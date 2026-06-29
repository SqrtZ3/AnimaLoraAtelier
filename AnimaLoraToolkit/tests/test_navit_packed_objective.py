"""Stage 3 of NaViT/FiT block-diagonal packing: the packed training-step core.

Validates ``navit_packed_forward_and_loss``: per-image noising, packed
block-diagonal forward, and per-image masked-token loss run end to end, produce
finite gradients, and the returned scalar equals the mean of the per-image
losses it reports.

Needs CUDA + xformers; skipped otherwise.
"""
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch
    from models.anima_modeling import Anima as TrainingAnima
    from models.anima_modeling_core import set_xformers_enabled
    from trainer.objective import (
        LossConfig,
        NoiseConfig,
        navit_packed_forward_and_loss,
    )
    HAS_TORCH = True
except ModuleNotFoundError:
    HAS_TORCH = False

HAS_CUDA = HAS_TORCH and torch.cuda.is_available()
try:
    import xformers  # noqa: F401
    HAS_XFORMERS = True
except Exception:
    HAS_XFORMERS = False


@unittest.skipUnless(HAS_TORCH and HAS_CUDA and HAS_XFORMERS,
                     "needs torch + CUDA + xformers")
class NavitPackedObjectiveTests(unittest.TestCase):
    def _model(self, dtype):
        torch.manual_seed(0)
        m = TrainingAnima(
            max_img_h=64, max_img_w=64, max_frames=1,
            in_channels=16, out_channels=16,
            patch_spatial=2, patch_temporal=1, concat_padding_mask=False,
            model_channels=128, num_blocks=2, num_heads=2, mlp_ratio=2.0,
            crossattn_emb_channels=128, pos_emb_cls="rope3d",
        )
        return m.to(device="cuda", dtype=dtype).train()

    def _pack(self, dtype):
        latent_shapes = [(4, 4), (6, 8), (4, 6)]
        text_lens = [5, 9, 3]
        lat_list, cross_list, tseq = [], [], []
        for (h, w), L in zip(latent_shapes, text_lens):
            lat_list.append(torch.randn(1, 16, 1, h, w, device="cuda", dtype=dtype))
            cross_list.append(torch.randn(1, L, 128, device="cuda", dtype=dtype))
            tseq.append(L)
        return lat_list, torch.cat(cross_list, dim=1), tseq

    def test_forward_loss_backward_finite(self):
        set_xformers_enabled(True)
        dtype = torch.float16
        model = self._model(dtype)
        lat_list, cross_packed, tseq = self._pack(dtype)
        t = torch.tensor([0.2, 0.6, 0.9], device="cuda")

        with torch.autocast("cuda", dtype=dtype):   # matches the real training loop
            loss, pred, info = navit_packed_forward_and_loss(
                model, lat_list, t, cross_packed, tseq,
                NoiseConfig(), LossConfig(loss_type="huber"),
            )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(info["visual_seqlens"], [4, 12, 6])     # (h/2)*(w/2)
        self.assertEqual(tuple(pred.shape[:2]), (1, 22))
        # reported scalar == mean of reported per-image losses
        torch.testing.assert_close(loss.float(), info["per_image_loss"].float().mean())

        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertGreater(len(grads), 0)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))

    def test_deterministic_noise_matches_manual_first_image(self):
        """With supplied noise, the per-image loss of image 0 equals a hand-computed
        masked MSE on the (Stage-2-proven) packed prediction slice."""
        from trainer.objective import masked_token_loss
        set_xformers_enabled(True)
        dtype = torch.float16
        model = self._model(dtype)
        lat_list, cross_packed, tseq = self._pack(dtype)
        t = torch.tensor([0.3, 0.5, 0.7], device="cuda")
        noise_list = [torch.randn_like(l) for l in lat_list]

        with torch.autocast("cuda", dtype=dtype):
            loss, pred, info = navit_packed_forward_and_loss(
                model, lat_list, t, cross_packed, tseq,
                NoiseConfig(), LossConfig(loss_type="mse"), noise_list=noise_list,
            )
        # recompute image-0 target and loss manually
        lat0 = lat_list[0]
        t0 = t[0].to(dtype).view(1, 1, 1, 1, 1)
        target0 = noise_list[0] - lat0
        ttok0, _, _, _ = model.patchify_latents_to_tokens(target0)
        n0 = info["visual_seqlens"][0]
        m = torch.ones(1, n0, device="cuda", dtype=dtype)
        manual0 = masked_token_loss(pred[:, :n0, :], ttok0, m, loss_type="mse")
        torch.testing.assert_close(
            info["per_image_loss"][0].float(), manual0[0].float(), rtol=2e-2, atol=2e-2
        )


if __name__ == "__main__":
    unittest.main()
