import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch
    from models.anima_modeling_core import GeneralDIT
    HAS_TORCH = True
except ModuleNotFoundError:
    HAS_TORCH = False
    torch = None
    GeneralDIT = None


@unittest.skipUnless(HAS_TORCH, "torch is required for model tests")
class PackedTokenModelTests(unittest.TestCase):
    def _model(self):
        return GeneralDIT(
            max_img_h=16,
            max_img_w=16,
            max_frames=1,
            in_channels=16,
            out_channels=16,
            patch_spatial=2,
            patch_temporal=1,
            concat_padding_mask=False,
            model_channels=48,
            num_blocks=1,
            num_heads=4,
            mlp_ratio=2.0,
            crossattn_emb_channels=48,
            pos_emb_cls="rope3d",
        )

    def test_patchify_unpatchify_round_trip(self):
        model = self._model()
        latents = torch.arange(1 * 16 * 1 * 4 * 6, dtype=torch.float32).view(1, 16, 1, 4, 6)

        tokens, _grid, _mask, size = model.patchify_latents_to_tokens(latents)
        restored = model.unpatchify_tokens(tokens, size)

        self.assertTrue(torch.equal(restored, latents))

    def test_forward_packed_tokens_preserves_shape_and_masks_padding(self):
        model = self._model()
        tokens = torch.randn(1, 4, 64)
        grid = torch.tensor([[[0, 0, 1, 1], [0, 1, 0, 1]]])
        mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.float32)
        size = torch.tensor([[[2, 2]]], dtype=torch.int32)
        cross = torch.randn(1, 512, 48)

        out = model.forward_packed_tokens(tokens, torch.tensor([[0.5]]), cross, grid, mask, size)

        self.assertEqual(tuple(out.shape), (1, 4, 64))
        self.assertTrue(torch.allclose(out[:, 3], torch.zeros_like(out[:, 3]), atol=1e-6))

    def test_patchify_mask_keeps_partial_source_edge_tokens(self):
        model = self._model()
        latents = torch.zeros(1, 16, 1, 4, 4)
        mask = torch.zeros(1, 1, 4, 4)
        mask[:, :, :3, :3] = 1

        _tokens, _grid, token_mask, _size = model.patchify_latents_to_tokens(latents, mask)

        self.assertTrue(torch.equal(token_mask, torch.ones_like(token_mask)))

    def test_forward_packed_tokens_rejects_empty_token_masks(self):
        model = self._model()
        tokens = torch.randn(1, 4, 64)
        grid = torch.tensor([[[0, 0, 1, 1], [0, 1, 0, 1]]])
        mask = torch.zeros(1, 4)
        size = torch.tensor([[[2, 2]]], dtype=torch.int32)
        cross = torch.randn(1, 512, 48)

        with self.assertRaisesRegex(ValueError, "no valid tokens"):
            model.forward_packed_tokens(tokens, torch.tensor([[0.5]]), cross, grid, mask, size)

    def test_forward_packed_tokens_rejects_grid_above_rope_capacity(self):
        model = self._model()
        tokens = torch.randn(1, 1, 64)
        grid = torch.tensor([[[16], [0]]])
        mask = torch.ones(1, 1)
        size = torch.tensor([[[17, 1]]], dtype=torch.int32)
        cross = torch.randn(1, 512, 48)

        with self.assertRaisesRegex(ValueError, "RoPE capacity"):
            model.forward_packed_tokens(tokens, torch.tensor([[0.5]]), cross, grid, mask, size)


if __name__ == "__main__":
    unittest.main()
