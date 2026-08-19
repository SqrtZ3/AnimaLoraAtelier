"""NCP-DPO CPU unit tests (arXiv 2406.17636, perceptual-space DPO loss).

Covers the model-decoupled pieces of trainer/ncp.py (no 2B model / no real DiT):

  - reverse_step_latent : x_{t'} = x_t − dt·v, with scalar and per-sample [B] dt broadcast.
  - ncp_perceptual_loss : per-sample shape; **exactness** (v_pred==v_true → PL≈0); gradient
                          flows to v_pred but NOT to the true anchor (it is detached);
                          feat_true sharing skips the redundant encoder call.
  - frozen_params       : inside the context a forward over frozen weights yields NO param
                          grad while input grad still flows; flags exactly restored on exit.
  - resolve_tap_block   : -1 → middle block; clamps into [0, n-1].

Run:  python tests/test_ncp.py     (also collectable by pytest)
"""

import pathlib
import sys

import torch
import torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trainer.ncp import (  # noqa: E402
    reverse_step_latent,
    ncp_perceptual_loss,
    frozen_params,
    resolve_tap_block,
    self_perceptual_per_sample,
)


def _identity(x):
    return x


# --------------------------------------------------------------------------- #
# reverse_step_latent
# --------------------------------------------------------------------------- #
def test_reverse_step_scalar():
    x_t = torch.randn(2, 16, 1, 8, 8)
    v = torch.randn(2, 16, 1, 8, 8)
    out = reverse_step_latent(x_t, v, 0.1)
    assert torch.allclose(out, x_t - 0.1 * v)


def test_reverse_step_per_sample_dt():
    x_t = torch.randn(3, 16, 1, 8, 8)
    v = torch.randn(3, 16, 1, 8, 8)
    dt = torch.tensor([0.1, 0.2, 0.3])
    out = reverse_step_latent(x_t, v, dt)
    for i in range(3):
        assert torch.allclose(out[i], x_t[i] - dt[i] * v[i])


# --------------------------------------------------------------------------- #
# ncp_perceptual_loss
# --------------------------------------------------------------------------- #
def test_perceptual_shape():
    x_t = torch.randn(4, 16, 1, 8, 8)
    v_pred = torch.randn(4, 16, 1, 8, 8)
    v_true = torch.randn(4, 16, 1, 8, 8)
    pl, feat_true = ncp_perceptual_loss(_identity, x_t, v_pred, v_true, 0.1)
    assert pl.shape == (4,)
    assert torch.all(pl >= 0)
    assert feat_true is not None


def test_perceptual_exactness():
    """v_pred == v_true → reverse latents coincide → PL ≈ 0 (validates the whole formula)."""
    x_t = torch.randn(4, 16, 1, 8, 8)
    v = torch.randn(4, 16, 1, 8, 8)
    pl, _ = ncp_perceptual_loss(_identity, x_t, v, v.clone(), 0.1)
    assert torch.allclose(pl, torch.zeros(4), atol=1e-6), pl


def test_perceptual_grad_only_through_pred():
    x_t = torch.randn(4, 16, 1, 8, 8)
    v_pred = torch.randn(4, 16, 1, 8, 8, requires_grad=True)
    v_true = torch.randn(4, 16, 1, 8, 8, requires_grad=True)
    pl, _ = ncp_perceptual_loss(_identity, x_t, v_pred, v_true, 0.1)
    pl.sum().backward()
    assert v_pred.grad is not None and torch.any(v_pred.grad != 0)
    assert v_true.grad is None, "true anchor must be detached (no preference signal through it)"


def test_perceptual_feat_true_sharing():
    """Passing a precomputed feat_true must skip the second (true-anchor) encoder call."""
    calls = {"n": 0}

    def counting_fn(x):
        calls["n"] += 1
        return x

    x_t = torch.randn(2, 16, 1, 8, 8)
    v_pred = torch.randn(2, 16, 1, 8, 8)
    v_true = torch.randn(2, 16, 1, 8, 8)
    # first call computes both pred + true features = 2 calls
    _, feat_true = ncp_perceptual_loss(counting_fn, x_t, v_pred, v_true, 0.1)
    assert calls["n"] == 2
    calls["n"] = 0
    # reusing feat_true = only the pred feature = 1 call
    _, _ = ncp_perceptual_loss(counting_fn, x_t, v_pred, v_true, 0.1, feat_true=feat_true)
    assert calls["n"] == 1


# --------------------------------------------------------------------------- #
# frozen_params
# --------------------------------------------------------------------------- #
def test_frozen_params_blocks_param_grad_keeps_input_grad():
    lin = nn.Linear(8, 8)
    x = torch.randn(4, 8, requires_grad=True)
    with frozen_params(lin.parameters()):
        assert all(not p.requires_grad for p in lin.parameters())
        y = lin(x)
        y.sum().backward()
    # input grad flows through frozen weights; weights themselves get no grad
    assert x.grad is not None and torch.any(x.grad != 0)
    assert lin.weight.grad is None
    assert lin.bias.grad is None
    # flags restored on exit
    assert all(p.requires_grad for p in lin.parameters())


def test_frozen_params_restores_mixed_flags():
    a = torch.zeros(2, requires_grad=True)
    b = torch.zeros(2, requires_grad=False)
    with frozen_params([a, b]):
        assert not a.requires_grad and not b.requires_grad
    assert a.requires_grad is True
    assert b.requires_grad is False


# --------------------------------------------------------------------------- #
# resolve_tap_block
# --------------------------------------------------------------------------- #
def test_resolve_tap_block():
    assert resolve_tap_block(28, -1) == 14          # auto = middle
    assert resolve_tap_block(28, 5) == 5            # explicit
    assert resolve_tap_block(28, 100) == 27         # clamp to last
    assert resolve_tap_block(28, 0) == 0


# --------------------------------------------------------------------------- #
# self_perceptual_per_sample (SFT aux term, arXiv 2401.00110)
# --------------------------------------------------------------------------- #
def test_self_perceptual_shape_and_nonneg():
    feat_p = torch.randn(4, 3, 8, 8)
    feat_t = torch.randn(4, 3, 8, 8)
    t = torch.tensor([0.1, 0.2, 0.3, 0.4])
    out = self_perceptual_per_sample(feat_p, feat_t, t, t_gate=0.5)
    assert out.shape == (4,)
    assert torch.all(out >= 0)
    assert out.dtype == torch.float32


def test_self_perceptual_exactness():
    """identical features → loss ≈ 0 (validates feature_mse reuse)."""
    feat = torch.randn(3, 16, 4, 4)
    t = torch.tensor([0.1, 0.2, 0.3])
    out = self_perceptual_per_sample(feat, feat.clone(), t, t_gate=0.9)
    assert torch.allclose(out, torch.zeros(3), atol=1e-6), out


def test_self_perceptual_t_gate_zeros_high_t():
    """Samples with t >= t_gate contribute exactly 0; low-t ones are nonzero."""
    feat_p = torch.randn(4, 8, 4, 4)
    feat_t = torch.randn(4, 8, 4, 4)
    t = torch.tensor([0.1, 0.4, 0.6, 0.9])   # gate 0.5 → first two active, last two gated
    out = self_perceptual_per_sample(feat_p, feat_t, t, t_gate=0.5)
    assert torch.all(out[:2] > 0)
    assert torch.all(out[2:] == 0)


def test_self_perceptual_all_gated():
    feat_p = torch.randn(2, 8, 4, 4)
    feat_t = torch.randn(2, 8, 4, 4)
    t = torch.tensor([0.7, 0.95])
    out = self_perceptual_per_sample(feat_p, feat_t, t, t_gate=0.5)
    assert torch.all(out == 0)


def test_self_perceptual_grad_only_through_pred():
    """Gradient flows to the predicted features; the target anchor must be detached."""
    feat_p = torch.randn(4, 8, 4, 4, requires_grad=True)
    feat_t = torch.randn(4, 8, 4, 4, requires_grad=True)
    t = torch.tensor([0.1, 0.2, 0.3, 0.4])
    out = self_perceptual_per_sample(feat_p, feat_t.detach(), t, t_gate=0.5)
    out.sum().backward()
    assert feat_p.grad is not None and torch.any(feat_p.grad != 0)
    assert feat_t.grad is None


def test_self_perceptual_scale_invariant():
    """归一化后 pl 只依赖相对偏差、与特征绝对尺度无关（锁 bugfix：DiT 裸激活不再爆量级）。

    旧实现 pl=‖fp−ft‖² 直接用裸激活，feat 放大 100× → pl 放大 10000×，未收敛时顶到数千、
    ×λ 爆梯度。归一化后 pl=‖fp−ft‖²/‖ft‖² 是相对能量，同比例放大 target 与 pred → pl 不变。
    """
    torch.manual_seed(0)
    feat_t = torch.randn(2, 8, 4, 4)
    feat_p = feat_t + 0.5 * torch.randn(2, 8, 4, 4)     # 固定相对偏差
    t = torch.tensor([0.1, 0.2])
    pl_small = self_perceptual_per_sample(feat_p, feat_t, t, t_gate=0.5)
    pl_large = self_perceptual_per_sample(feat_p * 100.0, feat_t * 100.0, t, t_gate=0.5)
    assert torch.allclose(pl_small, pl_large, atol=1e-4), (pl_small, pl_large)
    # 归一化后 pl 有界（旧实现对 100× 激活会给出 ~1e4 量级）
    assert torch.all(pl_small < 10.0), pl_small


if __name__ == "__main__":
    test_reverse_step_scalar()
    test_reverse_step_per_sample_dt()
    test_perceptual_shape()
    test_perceptual_exactness()
    test_perceptual_grad_only_through_pred()
    test_perceptual_feat_true_sharing()
    test_frozen_params_blocks_param_grad_keeps_input_grad()
    test_frozen_params_restores_mixed_flags()
    test_resolve_tap_block()
    test_self_perceptual_shape_and_nonneg()
    test_self_perceptual_exactness()
    test_self_perceptual_t_gate_zeros_high_t()
    test_self_perceptual_all_gated()
    test_self_perceptual_grad_only_through_pred()
    test_self_perceptual_scale_invariant()
    print("test_ncp: all passed")
