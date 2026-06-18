"""Eisbach log-barrier CPU unit tests (arXiv 2606.07207, image port).

Covers the pure-tensor per-sample weight `eisbach_barrier_weight` (trainer/objective.py):

  - shape/bounds      : per-sample [B], every weight in [(1-λ), 1].
  - λ=0 no-op         : disabled → exactly 1.0 everywhere (zero training effect).
  - detached          : returned weight carries no grad (it only scales step size).
  - structural order  : peaked (low-entropy) sample weighted HIGHER than flat (high-entropy)
                        sample — the whole point of the barrier's Darwinian curation.
  - mask path         : invalid positions excluded from the energy distribution.

Run:  python tests/test_eisbach.py     (also collectable by pytest)
"""

import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trainer.objective import eisbach_barrier_weight  # noqa: E402


def _peaked(c=16, h=8, w=8):
    """A velocity-output sample with energy concentrated at a single position (low entropy)."""
    o = torch.zeros(c, 1, h, w)
    o[:, 0, h // 2, w // 2] = 5.0
    return o


def _flat(c=16, h=8, w=8):
    """A velocity-output sample with near-uniform energy (high entropy)."""
    return torch.ones(c, 1, h, w)


def test_shape_and_bounds():
    pred = torch.randn(4, 16, 1, 8, 8)
    lam = 0.5
    w = eisbach_barrier_weight(pred, lam)
    assert w.shape == (4,)
    assert torch.all(w <= 1.0 + 1e-6)
    assert torch.all(w >= (1.0 - lam) - 1e-6)


def test_lambda_zero_is_noop():
    pred = torch.randn(3, 16, 1, 8, 8)
    w = eisbach_barrier_weight(pred, 0.0)
    assert torch.allclose(w, torch.ones(3), atol=1e-6)


def test_detached():
    pred = torch.randn(2, 16, 1, 8, 8, requires_grad=True)
    w = eisbach_barrier_weight(pred, 0.5)
    assert not w.requires_grad


def test_structural_ordering():
    """Low-entropy (peaked) sample must receive a larger weight than a flat one."""
    pred = torch.stack([_peaked(), _flat()], dim=0)  # [2, C, 1, H, W]
    w = eisbach_barrier_weight(pred, 0.5)
    assert w[0] > w[1], f"peaked {w[0]:.4f} should exceed flat {w[1]:.4f}"
    # flat sample should sit near the (1-λ) floor; peaked near 1.0
    assert w[1] < 0.75
    assert w[0] > 0.9


def test_mask_excludes_positions():
    """With a mask, only valid positions enter the softmax/entropy."""
    pred = torch.randn(2, 16, 1, 8, 8)
    mask = torch.ones(2, 1, 1, 8, 8)
    mask[..., 4:, :] = 0.0  # drop half the rows
    w_full = eisbach_barrier_weight(pred, 0.5)
    w_masked = eisbach_barrier_weight(pred, 0.5, mask=mask)
    assert w_masked.shape == (2,)
    assert torch.all(w_masked <= 1.0 + 1e-6)
    # masking changes the distribution → weights differ from the full-grid case
    assert not torch.allclose(w_full, w_masked, atol=1e-4)


if __name__ == "__main__":
    test_shape_and_bounds()
    test_lambda_zero_is_noop()
    test_detached()
    test_structural_ordering()
    test_mask_excludes_positions()
    print("test_eisbach: all passed")
