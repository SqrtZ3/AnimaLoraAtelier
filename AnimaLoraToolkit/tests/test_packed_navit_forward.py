"""Stage 2 of NaViT/FiT block-diagonal packing.

End-to-end model equivalence: ``MiniTrainDIT.forward_packed_navit`` on G
heterogeneous images packed into one sequence must equal running each image
independently through ``forward_packed_tokens`` (single image, all-valid mask,
its own timestep) and concatenating the per-image outputs.

This pins down everything the packed path adds at once: block-diagonal self
attention, block-diagonal cross attention to per-image captions, per-image
RoPE grids, and per-token AdaLN with one timestep per image.

Needs CUDA + xformers (varlen kernels are GPU-only); skipped otherwise.
Run with the local GPU python:
    D:\\ArtificialIntelligence\\ComfyUI-aki-v1.5\\python\\python.exe -m pytest \\
        AnimaLoraToolkit/tests/test_packed_navit_forward.py -v
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
class PackedNavitForwardTests(unittest.TestCase):
    def _model(self, dtype):
        torch.manual_seed(0)
        m = TrainingAnima(
            max_img_h=64,
            max_img_w=64,
            max_frames=1,
            in_channels=16,
            out_channels=16,
            patch_spatial=2,
            patch_temporal=1,
            concat_padding_mask=False,
            model_channels=128,   # /num_heads=2 -> head_dim 64 (xformers varlen needs 64/128/...)
            num_blocks=2,
            num_heads=2,
            mlp_ratio=2.0,
            crossattn_emb_channels=128,
            pos_emb_cls="rope3d",
        )
        return m.to(device="cuda", dtype=dtype).eval()

    def test_navit_equals_per_image_forward(self):
        set_xformers_enabled(True)
        dtype = torch.float16
        model = self._model(dtype)
        D = 128

        # Heterogeneous latents → token counts N = (h/2)*(w/2) = 4, 12, 6.
        latent_shapes = [(4, 4), (6, 8), (4, 6)]
        text_lens = [5, 9, 3]
        timesteps = [0.2, 0.6, 0.9]

        tokens_list, grid_list, size_list, vseq = [], [], [], []
        cross_list, tseq = [], []
        with torch.no_grad():
            for (h, w), L in zip(latent_shapes, text_lens):
                lat = torch.randn(1, 16, 1, h, w, device="cuda", dtype=dtype)
                tok, grid, _mask, size = model.patchify_latents_to_tokens(lat)
                tokens_list.append(tok)
                grid_list.append(grid)
                size_list.append(size)
                vseq.append(tok.shape[1])
                cross_list.append(torch.randn(1, L, D, device="cuda", dtype=dtype))
                tseq.append(L)

            # Per-image reference via the existing single-image packed forward.
            refs = []
            for tok, grid, size, cross, t in zip(
                tokens_list, grid_list, size_list, cross_list, timesteps
            ):
                mask = torch.ones(1, tok.shape[1], device="cuda", dtype=dtype)
                out = model.forward_packed_tokens(
                    tok,
                    torch.tensor([[t]], device="cuda", dtype=dtype),
                    cross,
                    grid,
                    mask,
                    size,
                )
                refs.append(out)
            ref = torch.cat(refs, dim=1)

            # Packed NaViT forward.
            tokens_packed = torch.cat(tokens_list, dim=1)
            grid_packed = torch.cat(grid_list, dim=2)
            cross_packed = torch.cat(cross_list, dim=1)
            ts = torch.tensor(timesteps, device="cuda", dtype=dtype)
            packed = model.forward_packed_navit(
                tokens_packed, ts, cross_packed, grid_packed, vseq, tseq
            )

        self.assertEqual(tuple(packed.shape), tuple(ref.shape))
        torch.testing.assert_close(packed, ref, rtol=3e-2, atol=3e-2)

    def test_navit_rejects_seqlen_mismatch(self):
        set_xformers_enabled(True)
        model = self._model(torch.float16)
        tok = torch.randn(1, 10, model.x_embedder.proj[1].in_features,
                          device="cuda", dtype=torch.float16)
        grid = torch.zeros(1, 2, 10, device="cuda", dtype=torch.float16)
        cross = torch.randn(1, 4, 128, device="cuda", dtype=torch.float16)
        with self.assertRaises(ValueError):
            model.forward_packed_navit(
                tok, torch.tensor([0.5], device="cuda"), cross, grid,
                visual_seqlens=[4, 4],  # sums to 8, not 10
                text_seqlens=[4],
            )


if __name__ == "__main__":
    unittest.main()
