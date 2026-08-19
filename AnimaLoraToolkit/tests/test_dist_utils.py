"""``utils/dist_utils`` 的本地单测 —— 不需要多卡、不需要 GPU。

覆盖的都是"错了不会报错、只会静默算错或静默卡死"的地方：

* **行为中立**：没有 torchrun 环境变量时，``DistContext`` 的每个方法都必须是恒等映射
  （这是"新功能 default-off、关掉时与改动前完全等价"这条铁律的可执行版本）。
* **分片语义**：各 rank 拿到的 batch 必须**互斥**、**长度严格相等**（长度不等 = 先跑完
  的 rank 退出循环、其余 rank 在集合通信上永远等下去），且并集是全局序列去掉尾部余数。
* **序号换算**：``reference_batches_for_batch_index`` 必须把局部序号换算回全局，
  否则 reference-step 记账会悄悄对错行。
* **不支持组合 fail-fast**：多卡下会死锁的配置必须在构造期报错而不是跑到第 300 步。
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import dist_utils  # noqa: E402


class _FakeSampler:
    """最小 batch sampler：产出 ``n`` 个 batch，并记录 set_epoch 是否被透传。"""

    def __init__(self, n, batch_size=4, seed=42):
        self.n = n
        self.batch_size = batch_size
        self.seed = seed
        self.epoch_set_to = None

    def __iter__(self):
        return iter([[i * 10, i * 10 + 1] for i in range(self.n)])

    def __len__(self):
        return self.n

    def set_epoch(self, e):
        self.epoch_set_to = e

    def reference_batches_for_batch_index(self, idx):
        # 直接回传全局序号，方便断言换算是否正确
        return idx


# ── 行为中立 ──────────────────────────────────────────────────────────────────

def test_disabled_context_is_identity():
    ctx = dist_utils.DistContext(enabled=False)
    assert ctx.is_main is True
    assert ctx.tag == ""            # 日志前缀为空 → 单卡输出逐字节不变
    assert ctx.all_ranks_clean(True) is True
    assert ctx.all_ranks_clean(False) is False
    assert ctx.mean_scalar(1.25) == 1.25
    assert ctx.all_reduce_grads_([]) == 0
    ctx.barrier()                   # no-op，不应抛
    ctx.shutdown()


def test_init_from_env_without_torchrun_is_noop(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    ctx = dist_utils.init_from_env()
    assert ctx.enabled is False
    assert ctx.world_size == 1
    assert ctx.rank == 0


def test_env_world_size_handles_garbage(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "not-a-number")
    assert dist_utils.env_world_size() == 1


def test_bind_device_is_noop_single_card(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert dist_utils.bind_device() == 0


# ── 分片语义 ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("total,world", [(24, 8), (25, 8), (31, 8), (8, 8), (100, 4), (7, 2)])
def test_shards_are_disjoint_and_equal_length(total, world):
    base = _FakeSampler(total)
    shards = [dist_utils.ShardedBatchSampler(base, r, world) for r in range(world)]

    lens = {len(s) for s in shards}
    assert len(lens) == 1, f"各 rank 的 batch 数必须相等，实际 {[len(s) for s in shards]}"
    assert lens.pop() == total // world

    seen = []
    for s in shards:
        got = list(s)
        assert len(got) == len(s), "__iter__ 产出的个数必须与 __len__ 一致"
        seen.extend(tuple(b) for b in got)

    assert len(seen) == len(set(seen)), "各 rank 的 batch 必须互斥，不能有重复样本"
    assert len(seen) == (total // world) * world

    # 并集必须是全局序列的**前缀集合**去掉尾部余数（每 epoch 重洗牌 → 丢的不是固定样本）
    all_global = [tuple(b) for b in _FakeSampler(total)]
    assert set(seen).issubset(set(all_global))


def test_shard_len_zero_when_fewer_batches_than_ranks():
    """batch 数少于卡数时每个 rank 拿 0 个 —— 必须是"一致地 0"，不能有的 1 有的 0。"""
    base = _FakeSampler(3)
    shards = [dist_utils.ShardedBatchSampler(base, r, 8) for r in range(8)]
    assert all(len(s) == 0 for s in shards)
    assert all(list(s) == [] for s in shards)


def test_local_index_is_translated_to_global():
    base = _FakeSampler(32)
    s = dist_utils.ShardedBatchSampler(base, rank=3, world_size=8)
    # 局部第 0 个 batch = 全局第 3 个；局部第 2 个 = 全局第 19 个
    assert s.reference_batches_for_batch_index(0) == 3
    assert s.reference_batches_for_batch_index(2) == 2 * 8 + 3


def test_attribute_passthrough_and_fingerprint_fields():
    base = _FakeSampler(16, batch_size=4, seed=1234)
    s = dist_utils.ShardedBatchSampler(base, rank=1, world_size=4)
    s.set_epoch(7)
    assert base.epoch_set_to == 7, "set_epoch 必须透传给底层 sampler"
    # dataloader_fingerprint 读的字段
    assert s.batch_size == 4
    assert s.seed == 1234
    assert s.world_size == 4, "world_size 必须是分片器自己的值（供指纹区分卡数）"
    with pytest.raises(AttributeError):
        _ = s.this_attribute_does_not_exist


def test_getattr_does_not_recurse_on_missing_base():
    """反 pickle 场景：__dict__ 为空时查属性不能无限递归。"""
    s = dist_utils.ShardedBatchSampler.__new__(dist_utils.ShardedBatchSampler)
    with pytest.raises(AttributeError):
        _ = s.base


def test_world_size_one_rejected():
    with pytest.raises(ValueError):
        dist_utils.ShardedBatchSampler(_FakeSampler(4), 0, 1)


# ── 不支持组合的 fail-fast ────────────────────────────────────────────────────

class _Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _multi():
    return dist_utils.DistContext(enabled=True, rank=0, world_size=8)


def test_guard_noop_when_single_card():
    # 单卡下即使配了这些项也不该报错（它们在单卡是合法的）
    args = _Args(effective_batch_size=32, dpo_enabled=True, lora_one_init_steps=5)
    dist_utils.guard_incompatible(args, dist_utils.DistContext(enabled=False))


def test_guard_rejects_sample_window_accumulation():
    with pytest.raises(ValueError, match="effective_batch_size"):
        dist_utils.guard_incompatible(_Args(effective_batch_size=32), _multi())


def test_guard_rejects_dpo_and_lora_one():
    with pytest.raises(ValueError, match="dpo_enabled"):
        dist_utils.guard_incompatible(_Args(dpo_enabled=True), _multi())
    with pytest.raises(ValueError, match="lora_one_init_steps"):
        dist_utils.guard_incompatible(_Args(lora_one_init_steps=4), _multi())


def test_guard_passes_clean_config():
    args = _Args(effective_batch_size=0, dpo_enabled=False, lora_one_init_steps=0,
                 gaf_enabled=False, adaptive_timestep=False)
    dist_utils.guard_incompatible(args, _multi())      # 不应抛


# ── 梯度打平/回写的语义（不需要真进程组）──────────────────────────────────────
# tests/test_dist_grad_reduce.py 会起两个进程走真 gloo 通信，但那个在起不了进程组的
# 机器上只能 skip。这里把 all_reduce 换成"加上另一个 rank 的缓冲"的桩，就能在**任何**
# 机器上验证真正易错的那部分：打平布局、grad=None 的占位、以及回写。
#
# 为什么这是最该测的：少放一个 None 占位 → 两个 rank 的缓冲布局错位一截 →
# all-reduce 回来的是逐元素串位的垃圾梯度。**不会报错**，loss 照降，只是在训错的东西。

torch = pytest.importorskip("torch")


def _mk_params(grads):
    """按 (shape, grad_value_or_None) 造一组参数。"""
    ps = []
    for shape, gv in grads:
        p = torch.nn.Parameter(torch.zeros(*shape))
        if gv is not None:
            p.grad = torch.full(shape, float(gv))
        ps.append(p)
    return ps


# rank0 / rank1 刻意用**不同的 None 模式**（模拟 module_dropout 在不同 rank 上跳过
# 了不同模块）—— 这正是打平布局最容易错位的场景。
_LAYOUT = [((4,), None), ((3,), None), ((2, 2), None)]
_R0 = [((4,), 1.0), ((3,), 8.0), ((2, 2), None)]
_R1 = [((4,), 3.0), ((3,), None), ((2, 2), 6.0)]


def _capture_flat(spec, monkeypatch):
    """跑一次 all_reduce_grads_，把它打平出来的缓冲截下来（all_reduce 桩成空操作）。"""
    import torch.distributed as tdist

    seen = {}

    def _fake(buf, op=None):
        seen["flat"] = buf.clone()

    monkeypatch.setattr(tdist, "all_reduce", _fake, raising=False)
    ctx = dist_utils.DistContext(enabled=True, rank=0, world_size=2,
                                 backend="gloo", device="cpu")
    ctx.all_reduce_grads_(_mk_params(spec))
    return seen["flat"]


def test_flat_layout_is_identical_across_ranks(monkeypatch):
    """两个 rank 的打平缓冲长度必须相同 —— 长度不同 = all-reduce 逐元素串位。"""
    f0 = _capture_flat(_R0, monkeypatch)
    f1 = _capture_flat(_R1, monkeypatch)
    assert f0.numel() == f1.numel() == 4 + 3 + 4
    # None 的那一段必须是**零占位**，不是被跳过
    assert f0[7:].abs().sum().item() == 0.0, "rank0 的 p2 是 None，应占位为零"
    assert f1[4:7].abs().sum().item() == 0.0, "rank1 的 p1 是 None，应占位为零"


@pytest.mark.parametrize("me,other", [(_R0, _R1), (_R1, _R0)])
def test_grad_writeback_matches_cross_rank_mean(me, other, monkeypatch):
    """把对端缓冲加进来，检查回写后的梯度 == 两 rank 均值（含 None 那一侧）。"""
    import torch.distributed as tdist

    other_flat = _capture_flat(other, monkeypatch)

    def _fake(buf, op=None):
        buf += other_flat

    monkeypatch.setattr(tdist, "all_reduce", _fake, raising=False)
    ctx = dist_utils.DistContext(enabled=True, rank=0, world_size=2,
                                 backend="gloo", device="cpu")
    params = _mk_params(me)
    n = ctx.all_reduce_grads_(params)
    assert n == 4 + 3 + 4

    # 期望：逐参数 (rank0值 + rank1值) / 2，None 当 0
    expect = []
    for (shape, a), (_, b) in zip(_R0, _R1):
        expect.append(((a or 0.0) + (b or 0.0)) / 2.0)

    for p, want in zip(params, expect):
        assert p.grad is not None, "grad=None 的参数也必须被补上平均值，否则该 rank 的权重会分叉"
        assert p.grad.flatten().tolist() == pytest.approx([want] * p.numel())


def test_buffer_is_reused_across_calls(monkeypatch):
    """缓冲要跨 step 复用（每步重新分配几十 MB 会打乱 allocator 的 segment 复用）。"""
    import torch.distributed as tdist

    monkeypatch.setattr(tdist, "all_reduce", lambda buf, op=None: None, raising=False)
    ctx = dist_utils.DistContext(enabled=True, rank=0, world_size=2,
                                 backend="gloo", device="cpu")
    params = _mk_params(_R0)
    ctx.all_reduce_grads_(params)
    first = ctx._flat
    ctx.all_reduce_grads_(params)
    assert ctx._flat is first, "同样的参数集，第二次调用不应重新分配缓冲"
