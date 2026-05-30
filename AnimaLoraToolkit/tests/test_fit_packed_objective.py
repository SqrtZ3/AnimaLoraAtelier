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
    torch_module = types.ModuleType("torch")
    sys.modules["torch"] = torch_module


@unittest.skipUnless(HAS_TORCH, "torch is required for objective tests")
class MaskedTokenLossTests(unittest.TestCase):
    def test_masked_token_loss_ignores_padding_tokens(self):
        from trainer.objective import masked_token_loss

        pred = torch.tensor([[[1.0], [10.0], [99.0]]])
        target = torch.tensor([[[0.0], [0.0], [0.0]]])
        mask = torch.tensor([[1, 1, 0]], dtype=torch.float32)

        loss = masked_token_loss(pred, target, mask, loss_type="mse")

        self.assertTrue(torch.allclose(loss, torch.tensor([50.5])))

    def test_masked_token_loss_clamps_empty_mask(self):
        from trainer.objective import masked_token_loss

        pred = torch.ones(1, 2, 1)
        target = torch.zeros(1, 2, 1)
        mask = torch.zeros(1, 2)

        loss = masked_token_loss(pred, target, mask)

        self.assertTrue(torch.isfinite(loss).all())
        self.assertEqual(float(loss.item()), 0.0)

    def test_masked_huber_uses_same_scale_as_grid_huber(self):
        from trainer.objective import masked_token_loss, per_sample_loss

        pred_tokens = torch.ones(1, 1, 1)
        target_tokens = torch.zeros_like(pred_tokens)
        mask = torch.ones(1, 1)
        pred_grid = torch.ones(1, 1, 1, 1, 1)
        target_grid = torch.zeros_like(pred_grid)

        token_loss = masked_token_loss(
            pred_tokens,
            target_tokens,
            mask,
            loss_type="huber",
            huber_c=0.2,
        )
        grid_loss = per_sample_loss(
            pred_grid,
            target_grid,
            loss_type="huber",
            huber_c=0.2,
            huber_schedule="constant",
        )

        self.assertTrue(torch.allclose(token_loss, grid_loss))

    def test_masked_huber_applies_timestep_schedule(self):
        from trainer.objective import masked_token_loss

        pred = torch.ones(2, 1, 1)
        target = torch.zeros_like(pred)
        mask = torch.ones(2, 1)
        t = torch.tensor([0.2, 0.8])

        loss = masked_token_loss(
            pred,
            target,
            mask,
            loss_type="huber",
            huber_c=0.2,
            huber_schedule="snr",
            t=t,
        )

        self.assertGreater(float(loss[0]), float(loss[1]))


@unittest.skipUnless(HAS_TORCH, "torch is required for objective tests")
class PackedForwardCheckpointTests(unittest.TestCase):
    def test_forward_packed_with_optional_checkpoint_checkpoints_each_block(self):
        from trainer import objective
        from trainer.objective import forward_packed_with_optional_checkpoint

        calls = []
        original_checkpoint = objective.checkpoint

        def fake_checkpoint(fn, x, use_reentrant=False):
            calls.append(use_reentrant)
            return fn(x)

        class Block(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.value = value

            def forward_tokens(self, x, *args, **kwargs):
                return x + self.value

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([Block(1.0), Block(2.0), Block(3.0)])
                embed = torch.nn.Linear(4, 4, bias=False)
                torch.nn.init.eye_(embed.weight)
                self.x_embedder = types.SimpleNamespace(proj=[None, embed])
                self.t_embedder = lambda timesteps: (torch.zeros(timesteps.shape[0], 1, 4), None)
                self.t_embedding_norm = torch.nn.Identity()
                self.final_layer = types.SimpleNamespace(
                    forward_tokens=lambda x, emb, adaln_lora_B_T_3D=None: x
                )

            def _packed_rope_from_grid(self, grid):
                return None

            def _output_tokens_to_patch_tokens(self, tokens, size=None):
                # The real model permutes within-token channels; the fake blocks use a flat
                # token dim, so an identity passthrough keeps this checkpoint-structure test
                # focused on per-block checkpointing.
                return tokens

            def forward_packed_tokens(self, tokens, timesteps, cross, grid, mask, size):
                raise AssertionError("whole packed forward should not be checkpointed")

        try:
            objective.checkpoint = fake_checkpoint
            model = Model()
            tokens = torch.zeros(1, 2, 4)
            out = forward_packed_with_optional_checkpoint(
                model,
                tokens,
                torch.tensor([[0.5]]),
                torch.zeros(1, 1, 4),
                torch.zeros(1, 2, 2, dtype=torch.long),
                torch.ones(1, 2),
                torch.tensor([[[1, 2]]], dtype=torch.int32),
                use_checkpoint=True,
            )
        finally:
            objective.checkpoint = original_checkpoint

        self.assertEqual(calls, [False, False, False])
        self.assertTrue(torch.equal(out, torch.full_like(tokens, 6.0)))


if __name__ == "__main__":
    unittest.main()
