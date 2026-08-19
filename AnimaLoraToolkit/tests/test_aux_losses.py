import pathlib
import sys
import types
import unittest

try:
    import torch
    HAS_TORCH = True
except ModuleNotFoundError:
    HAS_TORCH = False
    torch = None


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if HAS_TORCH:
    from trainer.aux_losses import AuxLossConfig, PerceptualLossModule, spectral_loss


class PerceptualGateTests(unittest.TestCase):
    def _module(self):
        module = PerceptualLossModule.__new__(PerceptualLossModule)
        torch.nn.Module.__init__(module)
        module.cfg = AuxLossConfig(
            perceptual_enabled=True,
            perceptual_t_gate=0.4,
            perceptual_use_checkpoint=True,
        )
        module.use_dino = False
        module.target_decode_calls = 0
        module.pred_calls = 0

        def decode_to_pixel(self, x0_latent, with_grad):
            if not with_grad:
                self.target_decode_calls += int(x0_latent.shape[0])
            return x0_latent

        def lpips_downsample(self, pixels):
            return pixels

        def compute_pred_against_target(self, x0_pred, pixels_target_lp, dino_feat_target):
            self.pred_calls += int(x0_pred.shape[0])
            return x0_pred.flatten(1).mean(dim=1)

        module._decode_to_pixel = types.MethodType(decode_to_pixel, module)
        module._lpips_downsample = types.MethodType(lpips_downsample, module)
        module._compute_pred_against_target = types.MethodType(compute_pred_against_target, module)
        return module

    @unittest.skipUnless(HAS_TORCH, "torch is required")
    def test_perceptual_checkpoint_only_runs_gate_active_samples(self):
        module = self._module()
        x0_pred = torch.arange(4.0).view(4, 1, 1, 1).requires_grad_(True)
        x0_target = torch.zeros_like(x0_pred)
        t = torch.tensor([0.1, 0.8, 0.9, 0.95])

        loss = module(x0_pred, x0_target, t)

        self.assertEqual(module.target_decode_calls, 1)
        self.assertEqual(module.pred_calls, 1)
        self.assertTrue(torch.equal(loss, x0_pred[0].mean()))


class SpectralLossTests(unittest.TestCase):
    @unittest.skipUnless(HAS_TORCH, "torch is required")
    def test_spectral_loss_has_finite_gradients_at_zero_amplitude(self):
        cfg = AuxLossConfig(
            spectral_enabled=True,
            spectral_use_wavelet=False,
            spectral_t_gate=0.5,
        )
        x0_pred = torch.zeros(2, 1, 1, 8, 8, requires_grad=True)
        x0_target = torch.randn_like(x0_pred)
        t = torch.tensor([0.1, 0.2])

        loss = spectral_loss(x0_pred, x0_target, t, cfg)
        loss.backward()

        self.assertTrue(torch.isfinite(loss).all())
        self.assertIsNotNone(x0_pred.grad)
        self.assertTrue(torch.isfinite(x0_pred.grad).all())


if __name__ == "__main__":
    unittest.main()
