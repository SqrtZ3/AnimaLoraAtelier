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


if __name__ == "__main__":
    unittest.main()
