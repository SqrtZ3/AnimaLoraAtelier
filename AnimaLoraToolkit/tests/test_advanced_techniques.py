"""Tests for the config-gated advanced techniques:

- Laplace timestep schedule (timestep_sampling="laplace")
- Configurable scheduled-Huber clamp (huber_snr_clamp_max)
- Contrastive Flow Matching negative term (dfm_lambda)
- Config plumbing (build_training_objective_config picks up the new knobs and
  defaults stay no-op / backward compatible).
"""

import pathlib
import sys
import types
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch
    HAS_TORCH = True
except ModuleNotFoundError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "torch is required for objective tests")
class LaplaceTimestepTests(unittest.TestCase):
    def test_laplace_in_range_and_shape(self):
        from trainer.objective import sample_t

        torch.manual_seed(0)
        t = sample_t(1000, "cpu", mode="laplace", laplace_mu=0.0, laplace_b=0.5)
        self.assertEqual(tuple(t.shape), (1000,))
        self.assertTrue(bool((t > 0).all()) and bool((t < 1).all()))
        self.assertTrue(bool(torch.isfinite(t).all()))

    def test_mu_shifts_toward_low_noise(self):
        # log-SNR λ; t = 1/(1+exp(λ/2)). μ>0 → higher SNR → lower t (low noise/detail).
        from trainer.objective import sample_t

        torch.manual_seed(0)
        t_neg = sample_t(40000, "cpu", mode="laplace", laplace_mu=-2.0, laplace_b=0.5).mean()
        t_zero = sample_t(40000, "cpu", mode="laplace", laplace_mu=0.0, laplace_b=0.5).mean()
        t_pos = sample_t(40000, "cpu", mode="laplace", laplace_mu=2.0, laplace_b=0.5).mean()
        self.assertLess(float(t_pos), float(t_zero))
        self.assertLess(float(t_zero), float(t_neg))
        # μ=0 should center near t=0.5 (λ≈0).
        self.assertAlmostEqual(float(t_zero), 0.5, delta=0.05)

    def test_b_controls_spread(self):
        from trainer.objective import sample_t

        torch.manual_seed(0)
        t_tight = sample_t(40000, "cpu", mode="laplace", laplace_mu=0.0, laplace_b=0.2)
        t_wide = sample_t(40000, "cpu", mode="laplace", laplace_mu=0.0, laplace_b=1.5)
        self.assertLess(float(t_tight.std()), float(t_wide.std()))


@unittest.skipUnless(HAS_TORCH, "torch is required for objective tests")
class HuberSnrClampTests(unittest.TestCase):
    def test_clamp_lowers_low_t_delta(self):
        from trainer.objective import _huber_delta_for_t

        t = torch.tensor([0.05])  # low noise: (1-t)/t = 19 → hits the upper clamp
        d_default = _huber_delta_for_t(t, 0.2, "snr", 10.0)
        d_tight = _huber_delta_for_t(t, 0.2, "snr", 3.0)
        self.assertAlmostEqual(float(d_default.flatten()[0]), 0.2 * 10.0, places=5)
        self.assertAlmostEqual(float(d_tight.flatten()[0]), 0.2 * 3.0, places=5)
        self.assertLess(float(d_tight.flatten()[0]), float(d_default.flatten()[0]))

    def test_default_matches_legacy_hardcoded_clamp(self):
        # Default snr_clamp_max=10.0 must reproduce the old behavior at every t.
        from trainer.objective import _huber_delta_for_t

        for tv in (0.05, 0.2, 0.5, 0.8, 0.95):
            t = torch.tensor([tv])
            d = _huber_delta_for_t(t, 0.2, "snr")
            snr_sqrt = min(max((1.0 - tv) / tv, 0.1), 10.0)
            self.assertAlmostEqual(float(d.flatten()[0]), 0.2 * snr_sqrt, places=5)

    def test_high_noise_unaffected_by_clamp_max(self):
        # At high noise the (1-t)/t term is small (< clamp), so changing the max
        # must not alter delta there.
        from trainer.objective import _huber_delta_for_t

        t = torch.tensor([0.8])
        d_default = _huber_delta_for_t(t, 0.2, "snr", 10.0)
        d_tight = _huber_delta_for_t(t, 0.2, "snr", 3.0)
        self.assertAlmostEqual(float(d_default.flatten()[0]), float(d_tight.flatten()[0]), places=6)


@unittest.skipUnless(HAS_TORCH, "torch is required for objective tests")
class ContrastiveFlowMatchingTests(unittest.TestCase):
    def test_singleton_batch_returns_zero(self):
        from trainer.objective import contrastive_flow_matching_neg

        pred = torch.randn(1, 4, 1, 8, 8)
        target = torch.randn_like(pred)
        neg = contrastive_flow_matching_neg(pred, target, loss_type="mse")
        self.assertEqual(tuple(neg.shape), (1,))
        self.assertEqual(float(neg.sum()), 0.0)

    def test_uses_rolled_partner(self):
        from trainer.objective import contrastive_flow_matching_neg, per_sample_loss

        torch.manual_seed(0)
        pred = torch.randn(3, 4, 1, 8, 8)
        target = torch.randn(3, 4, 1, 8, 8)
        neg = contrastive_flow_matching_neg(pred, target, loss_type="mse")
        perm = torch.roll(torch.arange(3), shifts=1)  # partner of i is i-1
        expected = per_sample_loss(pred, target.index_select(0, perm), loss_type="mse")
        self.assertTrue(torch.allclose(neg, expected))

    def test_equals_positive_when_all_targets_identical(self):
        # When every sample shares the same target, the contrastive ("negative")
        # term equals the positive term — i.e. no spurious repulsion when there is
        # nothing to contrast against.
        from trainer.objective import contrastive_flow_matching_neg, per_sample_loss

        torch.manual_seed(0)
        pred = torch.randn(4, 4, 1, 8, 8)
        target = torch.randn(1, 4, 1, 8, 8).expand(4, -1, -1, -1, -1).contiguous()
        neg = contrastive_flow_matching_neg(pred, target, loss_type="mse")
        pos = per_sample_loss(pred, target, loss_type="mse")
        self.assertTrue(torch.allclose(neg, pos))

    def test_packed_path_uses_mask(self):
        from trainer.objective import contrastive_flow_matching_neg

        pred = torch.randn(2, 3, 4)
        target = torch.randn(2, 3, 4)
        mask = torch.ones(2, 3)
        neg = contrastive_flow_matching_neg(pred, target, loss_type="mse", mask=mask)
        self.assertEqual(tuple(neg.shape), (2,))
        self.assertTrue(bool(torch.isfinite(neg).all()))


@unittest.skipUnless(HAS_TORCH, "torch is required for objective tests")
class ConfigPlumbingTests(unittest.TestCase):
    def test_build_config_reads_new_knobs(self):
        from trainer.objective import build_training_objective_config

        args = types.SimpleNamespace(
            timestep_laplace_mu=1.5,
            timestep_laplace_b=0.3,
            huber_snr_clamp_max=3.0,
            dfm_lambda=0.05,
        )
        cfg = build_training_objective_config(args)
        self.assertEqual(cfg.timestep.laplace_mu, 1.5)
        self.assertEqual(cfg.timestep.laplace_b, 0.3)
        self.assertEqual(cfg.loss.huber_snr_clamp_max, 3.0)
        self.assertEqual(cfg.loss.dfm_lambda, 0.05)

    def test_defaults_are_backward_compatible_noop(self):
        from trainer.objective import build_training_objective_config

        cfg = build_training_objective_config(types.SimpleNamespace())
        self.assertEqual(cfg.loss.dfm_lambda, 0.0)            # ΔFM off
        self.assertEqual(cfg.loss.huber_snr_clamp_max, 10.0)  # legacy clamp
        self.assertEqual(cfg.timestep.laplace_mu, 0.0)
        self.assertEqual(cfg.timestep.laplace_b, 0.5)


if __name__ == "__main__":
    unittest.main()
