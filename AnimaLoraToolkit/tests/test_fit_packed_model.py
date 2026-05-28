import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch
    from models.anima_modeling_core import GeneralDIT
    from models.anima_modeling import Anima as TrainingAnima
    HAS_TORCH = True
except ModuleNotFoundError:
    HAS_TORCH = False
    torch = None
    GeneralDIT = None
    TrainingAnima = None


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

    def _training_anima_model(self):
        return TrainingAnima(
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

    def test_training_anima_exposes_packed_token_helpers(self):
        model = self._training_anima_model()
        latents = torch.arange(1 * 16 * 1 * 4 * 4, dtype=torch.float32).view(1, 16, 1, 4, 4)

        tokens, grid, mask, size = model.patchify_latents_to_tokens(latents)
        restored = model.unpatchify_tokens(tokens, size)
        out = model.forward_packed_tokens(
            tokens,
            torch.tensor([[0.5]]),
            torch.randn(1, 512, 48),
            grid,
            mask,
            size,
        )

        self.assertTrue(torch.equal(restored, latents))
        self.assertEqual(tuple(out.shape), tuple(tokens.shape))

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

    def test_forward_packed_tokens_uses_key_only_padding_mask(self):
        model = self._model()
        captured = {}

        def fake_attention(q, k, v, attn_mask=None):
            captured["shape"] = tuple(attn_mask.shape)
            return torch.zeros(q.shape[0], q.shape[1], q.shape[2] * q.shape[3], device=q.device, dtype=q.dtype)

        model.blocks[0].self_attn.attn_op = fake_attention
        tokens = torch.randn(1, 4, 64)
        grid = torch.tensor([[[0, 0, 1, 1], [0, 1, 0, 1]]])
        mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.float32)
        size = torch.tensor([[[2, 2]]], dtype=torch.int32)
        cross = torch.randn(1, 512, 48)

        model.forward_packed_tokens(tokens, torch.tensor([[0.5]]), cross, grid, mask, size)

        self.assertEqual(captured["shape"], (1, 1, 1, 4))

    def test_packed_checkpoint_helper_matches_model_forward(self):
        from trainer.objective import forward_packed_with_optional_checkpoint

        torch.manual_seed(123)
        model = self._model()
        tokens = torch.randn(1, 4, 64)
        grid = torch.tensor([[[0, 0, 1, 1], [0, 1, 0, 1]]])
        mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.float32)
        size = torch.tensor([[[2, 2]]], dtype=torch.int32)
        cross = torch.randn(1, 512, 48)
        timesteps = torch.tensor([[0.5]])

        expected = model.forward_packed_tokens(tokens, timesteps, cross, grid, mask, size)
        actual = forward_packed_with_optional_checkpoint(
            model,
            tokens,
            timesteps,
            cross,
            grid,
            mask,
            size,
            use_checkpoint=True,
        )

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

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
