"""LeapAlign CPU unit tests (no 2B model / no real VAE).

Covers the pure-tensor pieces of the two-step self-distillation port (trainer/leap.py):

  - sample_two_timesteps : k>j, in (0,1), gap honored, deterministic under a generator.
  - leap_training_step   : per-sample shape; **exactness** (true velocity -> x̂0==x0 -> loss≈0,
                           which validates the connector + x̂0 = x_j - j·v_j formula end-to-end);
                           gradient flows to the forward params; exactly two forward calls;
                           nested_grad_coe=0.0 path stays finite.

Run:  python tests/test_leap.py     (also collectable by pytest)
"""

import pathlib
import sys

import torch
import torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trainer.leap import sample_two_timesteps, leap_training_step  # noqa: E402


def _data(bs=4, c=16, h=8, w=8):
    """A fake (x0, noise, cross, pad_mask) tuple in the training-latent shape [B,C,1,H,W]."""
    x0 = torch.randn(bs, c, 1, h, w)
    noise = torch.randn(bs, c, 1, h, w)
    cross = torch.zeros(bs, 512, 8)
    pad = torch.zeros(bs, 1, h, w)
    return x0, noise, cross, pad


# --------------------------------------------------------------------------- #
# sample_two_timesteps
# --------------------------------------------------------------------------- #
def test_two_timesteps_ordering_and_bounds():
    g = torch.Generator().manual_seed(0)
    k, j = sample_two_timesteps(256, "cpu", min_gap=0.1, generator=g)
    assert k.shape == (256,) and j.shape == (256,)
    assert (k > j).all(), "k must be strictly greater than j"
    assert (k > 0).all() and (k < 1).all(), "k must stay in (0,1)"
    assert (j > 0).all() and (j < 1).all(), "j must stay in (0,1)"
    # gap is opened to >= min_gap except where the (0,1) clamp at the boundary forces it down;
    # the firm guarantee is k - j >= eps (1e-3).
    assert (k - j >= 1e-3 - 1e-6).all(), "gap must be at least eps"
    assert (k - j).mean() >= 0.1, "typical gap should reach min_gap on average"


def test_two_timesteps_deterministic():
    a = sample_two_timesteps(16, "cpu", generator=torch.Generator().manual_seed(7))
    b = sample_two_timesteps(16, "cpu", generator=torch.Generator().manual_seed(7))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


# --------------------------------------------------------------------------- #
# leap_training_step
# --------------------------------------------------------------------------- #
def test_leap_shape():
    x0, noise, cross, pad = _data(bs=3)
    k, j = sample_two_timesteps(3, "cpu", generator=torch.Generator().manual_seed(1))
    fwd = lambda x, t, c, p: 0.05 * x
    out = leap_training_step(fwd, x0, noise, cross, pad, k, j)
    assert out.shape == (3,), f"expected per-sample (B,), got {tuple(out.shape)}"
    assert torch.isfinite(out).all()


def test_leap_exactness_true_velocity_gives_zero_loss():
    """If the model predicts the TRUE velocity v=noise-x0, the two-step leap recovers x0
    exactly -> loss ≈ 0. This validates the connector + x̂0 = x_j - j·v_j math end-to-end."""
    x0, noise, cross, pad = _data(bs=4)
    k, j = sample_two_timesteps(4, "cpu", generator=torch.Generator().manual_seed(2))
    true_v = noise - x0
    fwd = lambda x, t, c, p: true_v          # perfect velocity regardless of input
    out = leap_training_step(fwd, x0, noise, cross, pad, k, j, nested_grad_coe=0.3)
    assert out.abs().max().item() < 1e-8, f"true velocity must give ~0 loss, got {out.abs().max().item()}"


def test_leap_gradient_flows_to_params():
    """Loss gradient must reach the forward's parameters (both leaps are differentiable)."""
    x0, noise, cross, pad = _data(bs=2)
    k, j = sample_two_timesteps(2, "cpu", generator=torch.Generator().manual_seed(3))
    w = nn.Parameter(torch.tensor(0.1))
    fwd = lambda x, t, c, p: w * x
    loss = leap_training_step(fwd, x0, noise, cross, pad, k, j, nested_grad_coe=0.3).mean()
    loss.backward()
    assert w.grad is not None and torch.isfinite(w.grad) and w.grad.abs().item() > 0


def test_leap_two_forward_calls():
    x0, noise, cross, pad = _data(bs=2)
    k, j = sample_two_timesteps(2, "cpu", generator=torch.Generator().manual_seed(4))
    calls = {"n": 0}

    def fwd(x, t, c, p):
        calls["n"] += 1
        return 0.05 * x

    leap_training_step(fwd, x0, noise, cross, pad, k, j)
    assert calls["n"] == 2, f"leap must do exactly 2 forwards, got {calls['n']}"


def test_leap_nested_grad_coe_zero_is_finite():
    """coe=0 detaches the nested path; loss/grad must still be finite (first leap still trains)."""
    x0, noise, cross, pad = _data(bs=2)
    k, j = sample_two_timesteps(2, "cpu", generator=torch.Generator().manual_seed(5))
    w = nn.Parameter(torch.tensor(0.2))
    fwd = lambda x, t, c, p: w * x
    loss = leap_training_step(fwd, x0, noise, cross, pad, k, j, nested_grad_coe=0.0).mean()
    loss.backward()
    assert torch.isfinite(loss) and w.grad is not None and torch.isfinite(w.grad)


def test_leap_traj_sim_weighting_runs():
    x0, noise, cross, pad = _data(bs=3)
    k, j = sample_two_timesteps(3, "cpu", generator=torch.Generator().manual_seed(6))
    fwd = lambda x, t, c, p: 0.05 * x
    out = leap_training_step(fwd, x0, noise, cross, pad, k, j,
                             traj_sim_weighting=True, traj_sim_min=0.1)
    assert out.shape == (3,) and torch.isfinite(out).all()


def main():
    tests = [
        test_two_timesteps_ordering_and_bounds,
        test_two_timesteps_deterministic,
        test_leap_shape,
        test_leap_exactness_true_velocity_gives_zero_loss,
        test_leap_gradient_flows_to_params,
        test_leap_two_forward_calls,
        test_leap_nested_grad_coe_zero_is_finite,
        test_leap_traj_sim_weighting_runs,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
