# -*- coding: utf-8 -*-
"""AC-LoRA（arXiv:2504.02231）训练期 RESTART + 导出期 SVD 压缩的单测。

覆盖：
  · svd_truncate_lora_pair：energy=1.0 数学恒等无损；报告丢弃能量 == 实测重建误差；
    max_rank 封顶。
  · aclora_restart_matrix：keep 随 p 单调不降；保留 top 能量方向；p→1 近乎 no-op。
  · LoRAInjector：fail-fast（aclora/compress 只允许标准 base LoRA）；step-0 中立；
    RESTART 触发时机（warmup + 间隔）；导出压缩件生成 + energy=1.0 行为中立。

全程 CPU、小张量，可在无 GPU 的 CI 上跑。
"""
import os
import tempfile

import torch

from trainer.lora import (
    LoRAInjector,
    aclora_restart_matrix,
    svd_truncate_lora_pair,
)


TARGETS = ["lin"]


def _make_model():
    m = torch.nn.Module()
    m.lin_a = torch.nn.Linear(64, 48)
    m.lin_b = torch.nn.Linear(48, 96)
    return m


# ── svd_truncate_lora_pair ───────────────────────────────────────────────
def test_truncate_energy1_lossless():
    torch.manual_seed(0)
    A = torch.randn(32, 64)
    B = torch.randn(96, 32)
    s = 0.5
    dW = (B @ A) * s
    down, up, keep, dropped = svd_truncate_lora_pair(A, B, s, energy=1.0)
    assert keep == 32 and dropped == 0.0
    rel = (up @ down - dW).norm() / dW.norm()
    assert rel < 1e-4, rel


def test_truncate_dropped_matches_reconstruction():
    torch.manual_seed(1)
    A = torch.randn(32, 64)
    B = torch.randn(96, 32)
    s = 1.0
    dW = (B @ A) * s
    for E in (0.9, 0.5):
        down, up, keep, dropped = svd_truncate_lora_pair(A, B, s, energy=E)
        actual = ((up @ down - dW).pow(2).sum() / dW.pow(2).sum()).item()
        assert abs(dropped - actual) < 1e-4, (E, dropped, actual)


def test_truncate_max_rank_cap():
    torch.manual_seed(2)
    A = torch.randn(24, 64)
    B = torch.randn(96, 24)
    _d, _u, keep, _dr = svd_truncate_lora_pair(A, B, 1.0, energy=1.0, max_rank=10)
    assert keep == 10


# ── aclora_restart_matrix ────────────────────────────────────────────────
def test_restart_keep_monotone_and_topsignal():
    torch.manual_seed(0)
    U = torch.linalg.qr(torch.randn(96, 32))[0]
    V = torch.linalg.qr(torch.randn(48, 32))[0]
    S = torch.tensor([10, 8, 6, 4, 3] + [0.05] * 27).float()
    M = (U * S.unsqueeze(0)) @ V.t()

    def top_sub(X, k):
        return torch.linalg.svd(X.float(), full_matrices=False)[0][:, :k]

    prev = 0
    align_hi = None
    for p in (0.5, 0.9, 0.99, 0.99999):
        Mn, keep = aclora_restart_matrix(M, p)
        assert keep >= prev, (p, keep, prev)
        prev = keep
        align = torch.linalg.svdvals(top_sub(M, 5).t() @ top_sub(Mn, 5)).mean().item()
        align_hi = align
    # p→1 时 top-5 信号子空间高度保真、近乎 no-op
    assert align_hi > 0.99


def test_restart_change_decreases_with_p():
    # 稳健不变量：p 越大 → 保留越多 → RESTART 改动越小（单调）。
    # 注意忠实 Eq.3 用严格 <（"保留累计能量跨过 p 之前的分量"），有效保留能量 ≤ p，
    # 因此 p<1 时总会重置至少最小的那个分量，change 不会严格到 0——这是设计使然。
    torch.manual_seed(3)
    U = torch.linalg.qr(torch.randn(96, 32))[0]
    V = torch.linalg.qr(torch.randn(32, 32))[0]
    S = torch.tensor([10.0, 8.0, 6.0, 4.0, 3.0] + [0.3] * 27)
    M = (U * S.unsqueeze(0)) @ V.t()
    changes = []
    for p in (0.5, 0.9, 0.99, 0.9999):
        Mn, _keep = aclora_restart_matrix(M, p)
        changes.append(((Mn - M).norm() / M.norm()).item())
    assert changes == sorted(changes, reverse=True), changes
    assert changes[-1] < changes[0], changes


# ── LoRAInjector fail-fast ───────────────────────────────────────────────
def test_failfast_aclora_requires_standard_lora():
    for kw in (dict(use_lokr=True), dict(use_abba=True), dict(lora_variant="dora")):
        try:
            LoRAInjector(rank=8, alpha=8.0, targets=TARGETS, aclora_enabled=True, **kw)
            raise AssertionError(f"未 fail-fast: {kw}")
        except ValueError:
            pass


def test_failfast_compress_requires_standard_lora():
    try:
        LoRAInjector(rank=8, alpha=8.0, targets=TARGETS, lora_variant="dora",
                     lora_compress_energy=0.99)
        raise AssertionError("compress+dora 未 fail-fast")
    except ValueError:
        pass


# ── LoRAInjector step-0 中立 + RESTART 时机 ──────────────────────────────
def test_step0_neutral_and_restart_timing():
    torch.manual_seed(0)
    m = _make_model()
    inj = LoRAInjector(rank=8, alpha=8.0, targets=TARGETS,
                       aclora_enabled=True, aclora_restart_every=5, aclora_warmup_steps=10)
    inj.inject(m)
    x = torch.randn(4, 64)
    assert torch.allclose(m.lin_a.original(x), m.lin_a(x), atol=1e-6)

    with torch.no_grad():
        for lora in inj.injected.values():
            lora.adapter.lora_up.weight.normal_(0, 0.02)
            lora.adapter.lora_down.weight.normal_(0, 0.02)

    fired = [s for s in range(1, 26) if inj.aclora_step(s, 0.4)[0]]
    assert fired == [10, 15, 20, 25], fired


# ── LoRAInjector 导出压缩件 + 行为中立 ───────────────────────────────────
def test_compressed_export_and_neutrality():
    torch.manual_seed(0)
    tmp = tempfile.mkdtemp()

    m = _make_model()
    inj = LoRAInjector(rank=8, alpha=8.0, targets=TARGETS, lora_compress_energy=0.9)
    inj.inject(m)
    with torch.no_grad():
        for lora in inj.injected.values():
            w = lora.adapter.lora_up.weight
            w.zero_(); w[:, :2].normal_(0, 0.5)
            lora.adapter.lora_down.weight.normal_(0, 0.5)
    main = os.path.join(tmp, "m.safetensors")
    inj.save(main)
    comp = os.path.join(tmp, "m.compressed.safetensors")
    assert os.path.exists(main) and os.path.exists(comp)
    from safetensors.torch import load_file
    sd = load_file(comp)
    for base in ("lora_unet_lin_a", "lora_unet_lin_b"):
        assert sd[f"{base}.lora_down.weight"].shape[0] <= 8

    # energy=1.0 默认 → 不写压缩件
    m2 = _make_model()
    inj2 = LoRAInjector(rank=8, alpha=8.0, targets=TARGETS)
    inj2.inject(m2)
    main2 = os.path.join(tmp, "n.safetensors")
    inj2.save(main2)
    assert not os.path.exists(os.path.join(tmp, "n.compressed.safetensors"))
