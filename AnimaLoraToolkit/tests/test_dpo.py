"""Linear-DPO CPU unit tests (no 2B model / no real VAE).

Covers the pieces of the Linear-DPO preference-amplification phase that are pure
tensor math and therefore testable on CPU torch, per the spec "Testability"
section (docs/superpowers/specs/2026-06-15-linear-dpo-preference-amplification-design.md):

  - sample_latent  : training-shape latent out, no VAE decode, no-CFG = 1 fwd/step,
                     RNG isolated from the global stream, train/eval restored.
  - (step 2 will add) decision core ω' clip + Δ sign + gradient direction,
                      reference .data-swap correctness, DpoController round gating.

Run:  python tests/test_dpo.py     (also collectable by pytest)
"""

import pathlib
import sys

import torch
import torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trainer.sampling import sample_latent  # noqa: E402
from trainer.dpo import (  # noqa: E402
    DpoController,
    ReferenceAdapter,
    linear_dpo_loss,
)


# --------------------------------------------------------------------------- #
# tiny fake DiT — callable like the real model, counts forwards, no text path
# --------------------------------------------------------------------------- #
class FakeDiT(nn.Module):
    """Mimics the Anima DiT call signature used inside sample_latent:
    `model(x, sigma_b, cross_cond, padding_mask=...) -> velocity` with the same
    spatial shape as x. Returns a small, bounded velocity so the ER-SDE solver
    stays finite over a handful of steps. Counts every forward call."""

    def __init__(self):
        super().__init__()
        # a real parameter so .train()/.eval() and device moves behave like a Module
        self._p = nn.Parameter(torch.zeros(1))
        self.n_forward = 0

    def forward(self, x, sigma_b, cross_cond, padding_mask=None):
        self.n_forward += 1
        return 0.05 * x + self._p.to(x.dtype) * 0.0


def _fake_cond(dim: int = 8, dtype=torch.bfloat16):
    return torch.zeros(1, 512, dim, dtype=dtype)


# common small-but-real sampling config for CPU
_KW = dict(
    height=64, width=64, steps=3,
    sampler_name="er_sde", scheduler="simple",
    device="cpu", dtype=torch.bfloat16,
)


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def test_sample_latent_shape():
    """Returns a training-shape latent [1,16,1,h//8,w//8], float32, finite, not a PIL image."""
    model = FakeDiT()
    out = sample_latent(model, _fake_cond(), _fake_cond(), cfg_scale=4.0, **_KW)
    assert isinstance(out, torch.Tensor), f"expected a latent tensor, got {type(out)}"
    assert out.shape == (1, 16, 1, 64 // 8, 64 // 8), f"bad latent shape {tuple(out.shape)}"
    assert out.dtype == torch.float32, f"latent should be float32, got {out.dtype}"
    assert torch.isfinite(out).all(), "latent has non-finite values"


def test_sample_latent_nocfg_one_forward_per_step():
    """no-CFG branch (cross_uncond=None OR cfg==1.0) does exactly `steps` forwards;
    the CFG branch does 2*steps. ER-SDE calls denoise once per step boundary."""
    steps = _KW["steps"]

    m_none = FakeDiT()
    sample_latent(m_none, _fake_cond(), None, cfg_scale=4.0, **_KW)
    assert m_none.n_forward == steps, f"no-CFG (uncond=None): {m_none.n_forward} != {steps}"

    m_cfg1 = FakeDiT()
    sample_latent(m_cfg1, _fake_cond(), _fake_cond(), cfg_scale=1.0, **_KW)
    assert m_cfg1.n_forward == steps, f"no-CFG (cfg=1.0): {m_cfg1.n_forward} != {steps}"

    m_cfg = FakeDiT()
    sample_latent(m_cfg, _fake_cond(), _fake_cond(), cfg_scale=4.0, **_KW)
    assert m_cfg.n_forward == 2 * steps, f"CFG: {m_cfg.n_forward} != {2 * steps}"


def test_sample_latent_seed_determinism():
    """Same seed -> identical latent; different seed -> different latent."""
    m = FakeDiT()
    a = sample_latent(m, _fake_cond(), None, cfg_scale=1.0, seed=123, **_KW)
    b = sample_latent(m, _fake_cond(), None, cfg_scale=1.0, seed=123, **_KW)
    c = sample_latent(m, _fake_cond(), None, cfg_scale=1.0, seed=456, **_KW)
    assert torch.equal(a, b), "same seed must reproduce the same loser latent"
    assert not torch.equal(a, c), "different seed should give a different loser latent"


def test_sample_latent_rng_isolation():
    """A seeded sample_latent call must NOT consume from the global RNG stream
    (DPO in-loop sampling must not perturb the training noise sequence)."""
    torch.manual_seed(0)
    r_ref = torch.randn(4)

    torch.manual_seed(0)
    sample_latent(FakeDiT(), _fake_cond(), None, cfg_scale=1.0, seed=999, **_KW)
    r_after = torch.randn(4)

    assert torch.equal(r_ref, r_after), "seeded sample_latent leaked into the global RNG stream"


def test_sample_latent_restores_mode():
    """train/eval mode is restored on exit (and on error), for both starting modes."""
    m = FakeDiT()
    m.train()
    sample_latent(m, _fake_cond(), None, cfg_scale=1.0, **_KW)
    assert m.training is True, "training mode not restored after sample_latent"

    m.eval()
    sample_latent(m, _fake_cond(), None, cfg_scale=1.0, **_KW)
    assert m.training is False, "eval mode not preserved after sample_latent"

    class _Boom(FakeDiT):
        def forward(self, *a, **k):
            raise RuntimeError("boom")

    mb = _Boom()
    mb.train()
    try:
        sample_latent(mb, _fake_cond(), None, cfg_scale=1.0, **_KW)
    except RuntimeError:
        pass
    assert mb.training is True, "training mode not restored after an error mid-sampling"


# --------------------------------------------------------------------------- #
# decision core — linear_dpo_loss
# --------------------------------------------------------------------------- #
def test_linear_dpo_omega_clip():
    """ω' clips to [η, 1]: extreme +Δ -> 1, extreme −Δ -> η."""
    z = torch.zeros(1)
    _, hi = linear_dpo_loss(torch.tensor([100.0]), z, z, z, beta=1.0, eta=0.01)
    assert abs(hi["omega"].item() - 1.0) < 1e-6, f"upper clip {hi['omega'].item()}"
    _, lo = linear_dpo_loss(z, torch.tensor([100.0]), z, z, beta=1.0, eta=0.01)
    assert abs(lo["omega"].item() - 0.01) < 1e-6, f"lower clip {lo['omega'].item()}"


def test_linear_dpo_delta_sign():
    """Δ = (L_w − Lref_w) − (L_l − Lref_l)."""
    _, info = linear_dpo_loss(
        torch.tensor([2.0]), torch.tensor([1.0]),
        torch.tensor([0.5]), torch.tensor([0.5]), beta=1.0,
    )
    assert abs(info["delta"].item() - 1.0) < 1e-6, f"delta {info['delta'].item()}"


def test_linear_dpo_gradient_direction():
    """Minimizing L_DPO pushes L_θ(w) down (+grad) and L_θ(l) up (−grad); |grad| == ω'."""
    lw = torch.tensor([1.0], requires_grad=True)
    ll = torch.tensor([1.0], requires_grad=True)
    rw = torch.tensor([0.5])
    rl = torch.tensor([0.5])
    loss, info = linear_dpo_loss(lw, ll, rw, rl, beta=1.0, eta=0.01)
    loss.backward()
    assert lw.grad.item() > 0, "winner term should get +grad (pushed down on minimize)"
    assert ll.grad.item() < 0, "loser term should get −grad (pushed up on minimize)"
    assert abs(lw.grad.item() - info["omega"].item()) < 1e-6, "grad magnitude must equal ω'"


def test_dpo_loss_sft_anchor():
    """dpo_loss reproduces the hand-computed value; sft anchor adds λ·mean(L_θ(w))."""
    p = [nn.Parameter(torch.zeros(2))]
    base = DpoController(enabled=True, beta=2.0, eta=0.01, sft_anchor_lambda=0.0, params=p)
    loss, _ = base.dpo_loss(torch.tensor([1.0], requires_grad=True),
                            torch.tensor([2.0], requires_grad=True),
                            torch.tensor([1.0]), torch.tensor([1.0]))
    # Δ=-1 -> ω'=0.2*2*(-1)+0.5=0.1 ; margin=-1 ; loss=0.1*-1=-0.1
    assert abs(loss.item() - (-0.1)) < 1e-6, f"loss {loss.item()}"

    anch = DpoController(enabled=True, beta=2.0, eta=0.01, sft_anchor_lambda=0.5, params=p)
    loss_a, _ = anch.dpo_loss(torch.tensor([1.0], requires_grad=True),
                              torch.tensor([2.0], requires_grad=True),
                              torch.tensor([1.0]), torch.tensor([1.0]))
    # -0.1 + 0.5*mean([1.0]) = 0.4
    assert abs(loss_a.item() - 0.4) < 1e-6, f"anchored loss {loss_a.item()}"


# --------------------------------------------------------------------------- #
# reference adapter — .data swap
# --------------------------------------------------------------------------- #
def test_reference_swap_uses_frozen_and_restores():
    """In swap(): forward uses the frozen snapshot; on exit, policy .data is restored exactly."""
    lin = nn.Linear(4, 4, bias=False)
    x = torch.randn(3, 4)
    ref = ReferenceAdapter([lin.weight])              # snapshots the converged weight
    with torch.no_grad():
        y_snapshot = lin(x).clone()
        lin.weight.add_(1.0)                          # simulate further training
        y_policy = lin(x).clone()
        with ref.swap():
            y_ref = lin(x).clone()
        y_after = lin(x).clone()
    assert torch.allclose(y_ref, y_snapshot), "reference forward must use the frozen snapshot"
    assert not torch.allclose(y_ref, y_policy), "reference must differ from the moved policy"
    assert torch.allclose(y_after, y_policy), "policy .data must be restored exactly after swap"
    assert ref.ref_data[0].requires_grad is False, "frozen reference must not require grad"


def test_reference_ema_update():
    """ema=1.0 keeps the reference fixed; ema<1.0 moves it toward the current policy."""
    lin = nn.Linear(2, 2, bias=False)
    fixed = ReferenceAdapter([lin.weight], ema=1.0)
    orig = fixed.ref_data[0].clone()
    with torch.no_grad():
        lin.weight.add_(2.0)
    fixed.update_ema()
    assert torch.allclose(fixed.ref_data[0], orig), "ema=1.0 must keep the reference fixed"

    ema = ReferenceAdapter([lin.weight], ema=0.5)
    r0 = ema.ref_data[0].clone()
    with torch.no_grad():
        lin.weight.add_(4.0)
    p_now = lin.weight.detach().clone()
    ema.update_ema()
    assert torch.allclose(ema.ref_data[0], 0.5 * r0 + 0.5 * p_now), "ema=0.5 must move halfway"


# --------------------------------------------------------------------------- #
# controller — round gating, regen, disabled no-op
# --------------------------------------------------------------------------- #
def _ctrl(**kw):
    kw.setdefault("params", [nn.Parameter(torch.zeros(2))])
    kw.setdefault("beta", 0.1)
    return DpoController(**kw)


def test_controller_should_regen_gating():
    dpo = _ctrl(enabled=True, regen_every=100)
    assert dpo.should_regen(0), "round 1 must regen at step 0"
    assert not dpo.should_regen(1)
    assert not dpo.should_regen(99)
    assert dpo.should_regen(100)
    assert dpo.should_regen(200)


def test_controller_regen_subset_count():
    items = [(f"k{i}", i) for i in range(10)]
    sample_fn = lambda key, payload: torch.zeros(1, 16, 1, 8, 8)

    half = _ctrl(enabled=True, regen_every=100, loser_subset=0.5)
    assert half.regen_losers(sample_fn, items) == 5, "ceil(0.5*10)=5 refreshed"
    assert len(half.pool) == 5
    assert half.rounds == 1

    full = _ctrl(enabled=True, regen_every=100, loser_subset=1.0)
    assert full.regen_losers(sample_fn, items) == 10
    assert len(full.pool) == 10


def test_controller_disabled_and_no_params():
    off = _ctrl(enabled=False, regen_every=100)
    assert off.should_regen(0) is False
    assert off.regen_losers(lambda k, p: torch.zeros(1), [("k", 1)]) == 0
    with off.reference_mode():   # must not raise
        pass

    no_param = DpoController(enabled=True, beta=0.1, params=[])
    assert no_param.enabled is False, "enabled must downgrade to False without trainable params"
    assert no_param.reference is None


def main():
    tests = [
        test_sample_latent_shape,
        test_sample_latent_nocfg_one_forward_per_step,
        test_sample_latent_seed_determinism,
        test_sample_latent_rng_isolation,
        test_sample_latent_restores_mode,
        test_linear_dpo_omega_clip,
        test_linear_dpo_delta_sign,
        test_linear_dpo_gradient_direction,
        test_dpo_loss_sft_anchor,
        test_reference_swap_uses_frozen_and_restores,
        test_reference_ema_update,
        test_controller_should_regen_gating,
        test_controller_regen_subset_count,
        test_controller_disabled_and_no_params,
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
