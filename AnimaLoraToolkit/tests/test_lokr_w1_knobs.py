# -*- coding: utf-8 -*-
"""LoKr w1 的两个旋钮 + _find_factor 静默降级修复的回归测试。

背景（2026-07-25 本地取证，见 memory krea2-lokr-fullw2-counterexample）：
kron 的第 (i,j) 个块 = w1[i,j]·w2 —— f² 个块全是同一个 w2 的标量倍，**w1 就是那
f² 个"块间调制标量"**。w1 冻在随机 init 时该结构的表达力上界是闭式的 1/f²
（f=8 → 1.56%，数值实测贴合），而 w1 学到位时最优可达 11~16%。

而 `loraplus_lr_ratio` 抬的是 **w2_b，不是 w1**（易记错，这里用测试锁死）。
"""
import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trainer.lora import LoRAInjector  # noqa: E402


def _toy(in_f=768, out_f=768, n_layers=1):
    class Blocks(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn_q = torch.nn.Linear(in_f, out_f, bias=False)

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([Blocks() for _ in range(n_layers)])

    m = Toy()
    m.requires_grad_(False)
    return m


def _inject(in_f=768, out_f=768, **kw):
    m = _toy(in_f, out_f)
    inj = LoRAInjector(rank=32, alpha=32.0, use_lokr=True, targets=["attn_q"], **kw)
    inj.inject(m)
    return inj


# ── w1_init_std ──────────────────────────────────────────────────────────────

def test_w1_init_std_default_is_unchanged():
    """默认 0.1 → w1 的 rms 落在 normal(0,0.1) 的量级（行为中立）。"""
    ad = list(_inject().injected.values())[0].adapter
    rms = ad.lokr_w1.pow(2).mean().sqrt().item()
    assert 0.06 < rms < 0.15, f"默认 init 的 w1 rms={rms}，偏离 std=0.1 太远"


def test_w1_init_std_scales_linearly():
    """std 放大 5× → w1 的 rms 也应放大约 5×。"""
    small = list(_inject(lokr_w1_init_std=0.1).injected.values())[0].adapter
    big = list(_inject(lokr_w1_init_std=0.5).injected.values())[0].adapter
    r_small = small.lokr_w1.pow(2).mean().sqrt().item()
    r_big = big.lokr_w1.pow(2).mean().sqrt().item()
    assert 3.5 < r_big / r_small < 7.0, f"比值 {r_big / r_small} 不在预期的 ~5×"


def test_w1_init_std_does_not_break_step0_neutrality():
    """step-0 中立只靠 w2_b=0，改 w1_init_std 不该破坏它 —— 无论多大，ΔW 都必须恒 0。"""
    for std in (0.1, 0.5, 2.0):
        ad = list(_inject(lokr_w1_init_std=std).injected.values())[0].adapter
        assert torch.count_nonzero(ad.lokr_w2_b) == 0, "w2_b 应零初始化"
        x = torch.randn(2, 768)
        ad.train()
        assert ad(x).abs().max().item() == 0.0, f"std={std} 时 step-0 净 delta 非 0"


def test_w1_init_std_zero_fails_fast():
    """w1=0 会让 ΔW 与 w2 的梯度同时恒零，整个 LoKr 永久死掉 → 构造期就该拦。"""
    with pytest.raises(ValueError, match="lokr_w1_init_std"):
        _inject(lokr_w1_init_std=0.0)


# ── w1_lr_ratio ──────────────────────────────────────────────────────────────

def _group_of(param_groups, tensor):
    for g in param_groups:
        if any(p is tensor for p in g["params"]):
            return g
    raise AssertionError("参数没有出现在任何 param_group 里（永远不会被更新）")


def test_w1_lr_ratio_default_neutral():
    inj = _inject()
    ad = list(inj.injected.values())[0].adapter
    groups = inj.get_param_groups(weight_decay=0.01, base_lr=1e-4)
    # 默认 1.0 → 不该给 w1 单独设 lr（走 optimizer 的全局 lr）
    assert "lr" not in _group_of(groups, ad.lokr_w1)


def test_w1_lr_ratio_applies_to_w1_only():
    inj = _inject(lokr_w1_lr_ratio=16.0)
    ad = list(inj.injected.values())[0].adapter
    groups = inj.get_param_groups(weight_decay=0.01, base_lr=1e-4)
    assert _group_of(groups, ad.lokr_w1)["lr"] == pytest.approx(1e-4 * 16.0)
    # w2_a / w2_b 不受它影响
    assert "lr" not in _group_of(groups, ad.lokr_w2_a)
    assert "lr" not in _group_of(groups, ad.lokr_w2_b)


def test_loraplus_targets_w2b_not_w1():
    """★ 锁死这个易记错的事实：loraplus_lr_ratio 抬的是 w2_b，w1 完全不受影响。"""
    inj = _inject(loraplus_lr_ratio=16.0)
    ad = list(inj.injected.values())[0].adapter
    groups = inj.get_param_groups(weight_decay=0.01, base_lr=1e-4)
    assert _group_of(groups, ad.lokr_w2_b)["lr"] == pytest.approx(1e-4 * 16.0)
    assert "lr" not in _group_of(groups, ad.lokr_w1), \
        "loraplus 不应作用于 w1 —— 若要抬 w1 请用 lokr_w1_lr_ratio"
    assert "lr" not in _group_of(groups, ad.lokr_w2_a)


def test_two_ratios_are_independent():
    inj = _inject(loraplus_lr_ratio=16.0, lokr_w1_lr_ratio=4.0)
    ad = list(inj.injected.values())[0].adapter
    groups = inj.get_param_groups(weight_decay=0.01, base_lr=1e-4)
    assert _group_of(groups, ad.lokr_w1)["lr"] == pytest.approx(4e-4)
    assert _group_of(groups, ad.lokr_w2_b)["lr"] == pytest.approx(16e-4)


def test_knobs_rejected_on_non_lokr():
    """非 LoKr 路径下这两个参数无效，静默忽略会让一整轮 A/B 白跑 → fail-fast。"""
    m = _toy()
    with pytest.raises(ValueError, match="只对 lora_type='lokr' 生效"):
        LoRAInjector(rank=32, alpha=32.0, use_lokr=False, targets=["attn_q"],
                     lokr_w1_lr_ratio=8.0).inject(m)


# ── _find_factor：不再静默跌到 4/2/1 ─────────────────────────────────────────

@pytest.mark.parametrize("in_f,out_f,target,expect", [
    (6144, 6144, 8, 8),      # 整除，原样
    (6144, 6144, 6, 6),      # 旧实现会跌到 4（6144%6==0，本该保留 6）
    (6144, 16384, 6, 2),     # 16384%6!=0、%5!=0、%4==0… 但 6144%4==0 → 4；实际公约取 2
    (768, 768, 5, 1),        # 768%5!=0、%4==0 → 4
    (1536, 6144, 6, 6),
])
def test_find_factor_picks_largest_divisor(in_f, out_f, target, expect):
    from trainer.lora import LoKrLayer
    got = LoKrLayer._find_factor(None, in_f, out_f, target)
    # 断言：结果必须整除两边，且是 <=target 里最大的那个
    assert in_f % got == 0 and out_f % got == 0
    best = max(f for f in range(1, target + 1) if in_f % f == 0 and out_f % f == 0)
    assert got == best, f"in={in_f} out={out_f} target={target}: 取到 {got}，最大公约因子是 {best}"


def test_find_factor_no_longer_skips_middle_values():
    """回归：旧实现 [target,4,2,1] 会跳过 5/6/7，把 factor=6 静默换成 4。"""
    from trainer.lora import LoKrLayer
    assert LoKrLayer._find_factor(None, 6144, 6144, 6) == 6
    # 6144%7!=0 → 往下第一个能整除的是 6（旧实现会直接跌到 4）
    assert LoKrLayer._find_factor(None, 6144, 6144, 7) == 6
