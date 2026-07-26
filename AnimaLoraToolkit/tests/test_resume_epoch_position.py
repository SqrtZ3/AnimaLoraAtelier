# -*- coding: utf-8 -*-
"""断点续训的 epoch 内位置恢复（resume_skip_consumed_batches）回归测试。

历史 bug：`save_training_state` 只存 epoch 号，主循环 `for epoch in range(start_epoch,
args.epochs)` 之后直接 `for batch_idx, batch in enumerate(dataloader)`，中间没有任何
"跳过已消费 batch"的逻辑 —— 于是从 step 200 的 state 恢复时，会把 step 200 所在的那个
epoch **从头重跑一遍**，那一段数据被重复训练。

这里验证三件事：
1. 同一个 epoch 号下 batch 序列可复现（skip 的正确性前提，data.py 的 sampler 只用
   `random.Random(seed + epoch)`，不依赖全局 RNG）；
2. "跑到一半保存 + resume 快进"消费的 batch 序列，与"一口气跑完"逐 batch 相同
   —— 既不重复也不遗漏；
3. state 的存/取往返带上了 epoch 内位置，且旧 state 文件（没有该字段）能优雅退化。
"""
import random
import tempfile
from pathlib import Path

import torch

from trainer.checkpoint import (
    dataloader_fingerprint,
    load_training_state,
    save_training_state,
)
from trainer.data import BucketBatchSampler
from trainer.lora import LoRAInjector


# ---------------------------------------------------------------- 测试替身

class _FakeDataset:
    """最小数据集替身：只需要 __len__ 与 bucket_for_index（sampler 只看这两样）。"""

    def __init__(self, n=37, buckets=((512, 512), (768, 512), (512, 768))):
        self._n = n
        rng = random.Random(0)
        self.bucket_for_index = [rng.choice(buckets) for _ in range(n)]

    def __len__(self):
        return self._n


class _FakeLoader:
    """把 batch_sampler 直接当 dataloader 迭代（真 DataLoader 只是按索引取数据，
    batch 的**顺序**完全由 batch_sampler 决定，而顺序正是这里要验证的东西）。"""

    def __init__(self, dataset, sampler):
        self.dataset = dataset
        self.batch_sampler = sampler

    def __iter__(self):
        return iter(self.batch_sampler)


def _make_loader(effective_batch_size=0, seed=42, batch_size=2):
    ds = _FakeDataset()
    sampler = BucketBatchSampler(
        ds, batch_size=batch_size, drop_last=False, shuffle=True, seed=seed,
        effective_batch_size=effective_batch_size,
    )
    return _FakeLoader(ds, sampler)


def _run_epochs(loader, epochs, start_epoch=0, skip_first=0, accum_offset=0,
                accum_pending=0, stop_after=None, effective_batch_size=0):
    """按 anima_train.py 主循环的 epoch/skip 语义消费 batch，返回实际训练到的 batch。

    返回 (consumed, position)：
      consumed —— [(epoch, batch_idx, tuple(batch)), ...] 真正参与训练的 batch
      position —— 停在 stop_after 个 batch 时的 (epoch, batch_in_epoch, accum_offset)
    """
    consumed = []
    pending = accum_offset
    position = None
    for epoch in range(start_epoch, epochs):
        _skip_until = skip_first if epoch == start_epoch else 0
        if _skip_until:
            pending = accum_offset
            skip_first = 0
        loader.batch_sampler.set_epoch(epoch)
        if effective_batch_size:
            loader.batch_sampler.set_accumulation_offset(pending)
        offset_this_epoch = pending
        if _skip_until and _skip_until >= len(loader.batch_sampler):
            # 断点恰在该 epoch 的最后一个 batch：整个 epoch 已训完，直接进下一个
            # epoch，不空转一遍 dataloader（对应 anima_train.py 里的同名 fast path）
            pending = accum_pending
            _skip_until = 0
            continue
        for batch_idx, batch in enumerate(loader):
            if batch_idx < _skip_until:
                if batch_idx + 1 == _skip_until:
                    pending = accum_pending
                    _skip_until = 0
                continue
            consumed.append((epoch, batch_idx, tuple(batch)))
            if effective_batch_size:
                pending = (pending + len(batch)) % effective_batch_size
            if stop_after is not None and len(consumed) == stop_after:
                position = (epoch, batch_idx + 1, offset_this_epoch, pending)
                return consumed, position
    return consumed, position


# ---------------------------------------------------------------- 1. 可复现性

def test_batch_sequence_is_reproducible_per_epoch():
    """同一 epoch 号 + 同一数据集 → batch 序列逐 batch 相同（skip 精确性的前提）。

    顺便证明它不依赖全局 RNG：中间故意打乱 random/torch 的全局状态。
    """
    loader_a = _make_loader()
    loader_a.batch_sampler.set_epoch(3)
    seq_a = [tuple(b) for b in loader_a]

    random.random()
    torch.randn(8)

    loader_b = _make_loader()
    loader_b.batch_sampler.set_epoch(3)
    seq_b = [tuple(b) for b in loader_b]

    assert seq_a == seq_b, "同一 epoch 的 batch 序列不可复现 —— 跳过已消费 batch 会跳错样本"
    loader_b.batch_sampler.set_epoch(4)
    assert [tuple(b) for b in loader_b] != seq_a, "不同 epoch 应给出不同顺序（测试前提失效）"


# ---------------------------------------------------------------- 2. skip 语义

def test_resume_midepoch_consumes_each_batch_exactly_once():
    """中途保存 + resume 快进后，消费的 batch 序列 == 一口气跑完的序列。

    旧行为（skip=0）在这里必然失败：resume 会把该 epoch 从头重跑。
    """
    reference, _ = _run_epochs(_make_loader(), epochs=3)

    stop_after = 25  # 落在某个 epoch 中间
    part1, pos = _run_epochs(_make_loader(), epochs=3, stop_after=stop_after)
    assert pos is not None, "测试前提：stop_after 必须落在总 batch 数之内"
    saved_epoch, batch_in_epoch, _offset, _pending = pos

    part2, _ = _run_epochs(_make_loader(), epochs=3, start_epoch=saved_epoch,
                           skip_first=batch_in_epoch)

    got = [b for _, _, b in part1 + part2]
    want = [b for _, _, b in reference]
    assert got == want, (
        f"resume 后的 batch 序列与不中断训练不一致：\n"
        f"  长度 {len(got)} vs {len(want)}（多出的是被重复训练的 batch）"
    )


def test_old_behavior_would_replay_the_epoch():
    """反向断言：不跳过（旧行为）确实会重复训练一整段 —— 证明上面的测试真的在测东西。"""
    reference, _ = _run_epochs(_make_loader(), epochs=3)
    part1, pos = _run_epochs(_make_loader(), epochs=3, stop_after=25)
    saved_epoch, batch_in_epoch, _offset, _pending = pos

    replayed, _ = _run_epochs(_make_loader(), epochs=3, start_epoch=saved_epoch, skip_first=0)
    got = [b for _, _, b in part1 + replayed]
    assert len(got) > len(reference), "旧行为本应多消费 batch（重跑整个 epoch）"
    assert batch_in_epoch > 0


def test_resume_with_sample_accumulation_window():
    """开 sample-window accumulation 时，切分序列依赖 epoch 起始 offset —— 必须复原它。"""
    eff = 6
    reference, _ = _run_epochs(_make_loader(effective_batch_size=eff), epochs=3,
                               effective_batch_size=eff)

    part1, pos = _run_epochs(_make_loader(effective_batch_size=eff), epochs=3,
                             stop_after=25, effective_batch_size=eff)
    saved_epoch, batch_in_epoch, offset_at_epoch_start, _pending = pos

    part2, _ = _run_epochs(_make_loader(effective_batch_size=eff), epochs=3,
                           start_epoch=saved_epoch, skip_first=batch_in_epoch,
                           accum_offset=offset_at_epoch_start, accum_pending=_pending,
                           effective_batch_size=eff)

    got = [b for _, _, b in part1 + part2]
    want = [b for _, _, b in reference]
    assert got == want, "accumulation 切分下 resume 的 batch 序列不一致（offset 没复原？）"


def test_resume_at_epoch_boundary():
    """保存点恰好是某 epoch 的最后一个 batch：resume 应整段跳过该 epoch，不重跑。"""
    reference, _ = _run_epochs(_make_loader(), epochs=3)
    loader = _make_loader()
    loader.batch_sampler.set_epoch(0)
    n_epoch0 = sum(1 for _ in loader)

    part1, pos = _run_epochs(_make_loader(), epochs=3, stop_after=n_epoch0)
    saved_epoch, batch_in_epoch, _o, _p = pos
    assert (saved_epoch, batch_in_epoch) == (0, n_epoch0)

    part2, _ = _run_epochs(_make_loader(), epochs=3, start_epoch=saved_epoch,
                           skip_first=batch_in_epoch)
    got = [b for _, _, b in part1 + part2]
    assert got == [b for _, _, b in reference]


# ---------------------------------------------------------------- 3. 存取往返

def _toy_injector():
    class Blocks(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn_q = torch.nn.Linear(16, 16, bias=False)

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([Blocks()])

    m = Toy()
    m.requires_grad_(False)
    inj = LoRAInjector(rank=2, alpha=2.0, targets=["attn_q"])
    inj.inject(m)
    opt = torch.optim.AdamW(inj.get_params(), lr=1e-4)
    return inj, opt


def test_epoch_position_roundtrip():
    inj, opt = _toy_injector()
    loader = _make_loader()
    position = {
        "epoch": 3,
        "batch_in_epoch": 17,
        "accum_pending": 0,
        "accum_offset_at_epoch_start": 4,
        "fingerprint": dataloader_fingerprint(loader, grad_accum=2),
    }
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "state.pt"
        save_training_state(path, inj, opt, epoch=3, global_step=200,
                            epoch_position=position)
        inj2, opt2 = _toy_injector()
        got = load_training_state(path, inj2, opt2)
    assert len(got) == 7, "load_training_state 应返回 7 元组（末位是 epoch_position）"
    assert got[6] == position


def test_old_state_without_position_degrades_gracefully():
    """旧版本存的 state 没有 epoch_position → 返回 None，调用方回退到旧行为。"""
    inj, opt = _toy_injector()
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "state.pt"
        save_training_state(path, inj, opt, epoch=1, global_step=10)  # 不传 position
        inj2, opt2 = _toy_injector()
        got = load_training_state(path, inj2, opt2)
    assert got[6] is None


# ---------------------------------------------------------------- 4. 指纹

def test_fingerprint_detects_incompatible_changes():
    base = dataloader_fingerprint(_make_loader(), grad_accum=1)
    assert base == dataloader_fingerprint(_make_loader(), grad_accum=1)

    # 换 seed / batch_size / grad_accum / 数据量 → 序列不再可复现，指纹必须变
    assert base != dataloader_fingerprint(_make_loader(seed=7), grad_accum=1)
    assert base != dataloader_fingerprint(_make_loader(batch_size=4), grad_accum=1)
    assert base != dataloader_fingerprint(_make_loader(), grad_accum=2)
    assert base != dataloader_fingerprint(_make_loader(effective_batch_size=6), grad_accum=1)

    bigger = _make_loader()
    bigger.dataset._n += 1
    assert base != dataloader_fingerprint(bigger, grad_accum=1)
