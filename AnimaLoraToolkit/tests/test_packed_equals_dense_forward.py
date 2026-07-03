"""Equivalence probe: ``forward_packed_tokens`` vs dense ``forward``.

The NaViT training path runs ``forward_packed_navit`` (block-diagonal pack) whose
per-image reference is ``forward_packed_tokens``. eval / ARB training run the dense
``forward``. The existing suite only proved ``forward_packed_navit`` ≡
``forward_packed_tokens`` (test_packed_navit_forward) — the dense↔packed equivalence
was never asserted. On top of that, those tests construct the model with
``concat_padding_mask=False``, whereas the real Anima is built with
``concat_padding_mask=True`` (trainer/models.py), so the mask-channel branch in
``prepare_embedded_sequence`` — which dense runs but packed skips — was unexercised.

This test builds the model the way the trainer does (``concat_padding_mask=True``)
and checks, for a single image with an all-zero padding mask (exactly what both the
ARB training loop and eval feed), that:

    unpatchify_tokens(forward_packed_tokens(...))  ==  forward(...)

If they diverge, the NaViT path is optimizing a different objective than eval
measures — the smoking gun for "eval loss climbs before it descends".

Needs CUDA + xformers. Run with the local GPU python:
    D:\\ArtificialIntelligence\\ComfyUI-aki-v1.5\\python\\python.exe -m pytest \\
        AnimaLoraToolkit/tests/test_packed_equals_dense_forward.py -v -s
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
class PackedEqualsDenseForwardTests(unittest.TestCase):
    def _model(self, dtype, concat_padding_mask=True):
        torch.manual_seed(0)
        m = TrainingAnima(
            max_img_h=64,
            max_img_w=64,
            max_frames=1,
            in_channels=16,
            out_channels=16,
            patch_spatial=2,
            patch_temporal=1,
            concat_padding_mask=concat_padding_mask,
            model_channels=128,   # /num_heads=2 -> head_dim 64
            num_blocks=2,
            num_heads=2,
            mlp_ratio=2.0,
            crossattn_emb_channels=128,
            pos_emb_cls="rope3d",
        )
        return m.to(device="cuda", dtype=dtype).eval()

    def _run_one(self, model, lat, t_val, cross, dtype):
        """Return (dense_out, packed_out) both as [1,C,T,h,w] grids for one image."""
        h, w = lat.shape[-2], lat.shape[-1]
        # The all-zero padding mask is exactly what ARB training (animas_train.py:2930)
        # and eval (anima_train.py:2603) feed into dense forward.
        pad_mask = torch.zeros(1, 1, h, w, device="cuda", dtype=dtype)
        t = torch.tensor([t_val], device="cuda", dtype=dtype)

        with torch.no_grad():
            # Dense path: returns unpatchified [1,C,T,H,W].
            dense_out = model.forward(lat, t.view(-1, 1), cross, padding_mask=pad_mask)

            # Packed path: returns patch tokens [1,N,M]; unpatchify back to grid.
            tok, grid, _mask, size = model.patchify_latents_to_tokens(lat)
            mask_ones = torch.ones(1, tok.shape[1], device="cuda", dtype=dtype)
            packed_tok = model.forward_packed_tokens(
                tok, t.view(-1, 1), cross, grid, mask_ones, size,
            )
            packed_out = model.unpatchify_tokens(packed_tok, size)
        return dense_out, packed_out

    def test_packed_equals_dense_concat_mask_true(self):
        """Real Anima config: concat_padding_mask=True, all-zero pad mask."""
        set_xformers_enabled(True)
        dtype = torch.float16
        model = self._model(dtype, concat_padding_mask=True)
        D = 128
        # A few shapes / timesteps / caption lengths to stress the path.
        cases = [
            ((8, 8), 0.2, 7),
            ((6, 10), 0.6, 11),
            ((10, 6), 0.9, 3),
        ]
        max_abs_err = 0.0
        max_rel_err = 0.0
        for (shape, t_val, L) in cases:
            lat = torch.randn(1, 16, 1, shape[0], shape[1], device="cuda", dtype=dtype)
            cross = torch.randn(1, L, D, device="cuda", dtype=dtype)
            dense_out, packed_out = self._run_one(model, lat, t_val, cross, dtype)
            self.assertEqual(tuple(dense_out.shape), tuple(packed_out.shape),
                             f"shape mismatch for case {shape}")
            # Report the worst-case error so a near-miss is visible, not just pass/fail.
            diff = (dense_out.float() - packed_out.float()).abs()
            max_abs_err = max(max_abs_err, float(diff.max().item()))
            denom = dense_out.float().abs().clamp(min=1e-3)
            max_rel_err = max(max_rel_err, float((diff / denom).max().item()))
        print(f"\n[concat_padding_mask=True] max_abs_err={max_abs_err:.6e} "
              f"max_rel_err={max_rel_err:.6e}")
        # fp16 + xformers varlen vs SDPA: allow a generous tolerance first; if it
        # passes tight, the paths are equivalent. If it fails, we've found the gap.
        torch.testing.assert_close(dense_out, packed_out, rtol=5e-2, atol=5e-2)

    def test_packed_equals_dense_concat_mask_false(self):
        """Sanity: with concat_padding_mask=False the mask-channel branch is gone,
        so dense and packed should be equivalent (this is what existing tests
        implicitly assumed). Confirms the test harness is sound."""
        set_xformers_enabled(True)
        dtype = torch.float16
        model = self._model(dtype, concat_padding_mask=False)
        D = 128
        lat = torch.randn(1, 16, 1, 8, 8, device="cuda", dtype=dtype)
        cross = torch.randn(1, 7, D, device="cuda", dtype=dtype)
        dense_out, packed_out = self._run_one(model, lat, 0.5, cross, dtype)
        diff = (dense_out.float() - packed_out.float()).abs()
        print(f"\n[concat_padding_mask=False] max_abs_err={float(diff.max().item()):.6e}")
        torch.testing.assert_close(dense_out, packed_out, rtol=5e-2, atol=5e-2)


if __name__ == "__main__":
    unittest.main()
