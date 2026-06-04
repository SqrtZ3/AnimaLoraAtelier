import pathlib
import sys
import types
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch  # noqa: F401
    HAS_TORCH = True
    INSTALLED_FAKE_TORCH = False
except ModuleNotFoundError:
    HAS_TORCH = False
    INSTALLED_FAKE_TORCH = True
    torch_module = types.ModuleType("torch")
    torch_module.float32 = object()
    torch_module.float16 = object()
    torch_module.bfloat16 = object()
    torch_utils = types.ModuleType("torch.utils")
    torch_data = types.ModuleType("torch.utils.data")

    class Dataset:
        pass

    torch_data.Dataset = Dataset
    torch_utils.data = torch_data
    torch_module.utils = torch_utils
    sys.modules["torch"] = torch_module
    sys.modules["torch.utils"] = torch_utils
    sys.modules["torch.utils.data"] = torch_data

from trainer.data import BucketManager  # noqa: E402

if INSTALLED_FAKE_TORCH:
    sys.modules.pop("torch.utils.data", None)
    sys.modules.pop("torch.utils", None)
    sys.modules.pop("torch", None)


class TokenBucketManagerTests(unittest.TestCase):
    def test_token_bucket_mode_produces_correct_token_count(self):
        """Every bucket (W, H) must satisfy (W//16)*(H//16) == 4032."""
        manager = BucketManager(
            token_bucket=True,
            token_bucket_counts=[4032],
            token_bucket_min_dim=512,
            token_bucket_max_dim=2016,
        )
        self.assertTrue(len(manager.buckets) > 0, "token_bucket mode produced no buckets")
        for w, h in manager.buckets:
            tokens = (w // 16) * (h // 16)
            self.assertEqual(
                tokens, 4032,
                f"Bucket ({w}, {h}) has {tokens} tokens, expected 4032",
            )

    def test_get_bucket_returns_correct_token_count(self):
        """get_bucket(1000, 1000) should return a bucket with token count 4032."""
        manager = BucketManager(
            token_bucket=True,
            token_bucket_counts=[4032],
            token_bucket_min_dim=512,
            token_bucket_max_dim=2016,
        )
        w, h = manager.get_bucket(1000, 1000)
        tokens = (w // 16) * (h // 16)
        self.assertEqual(tokens, 4032, f"get_bucket returned ({w},{h}) with {tokens} tokens, expected 4032")

    def test_default_mode_is_arb_and_not_token_bucket(self):
        """Default BucketManager (token_bucket omitted) must use ARB path."""
        manager = BucketManager()
        self.assertFalse(manager.token_bucket, "Default manager should have token_bucket=False")
        self.assertIsNone(manager.token_bucket_counts, "Default manager token_bucket_counts should be None")
        self.assertTrue(len(manager.buckets) > 0, "Default ARB manager produced no buckets")

    def test_explicit_false_token_bucket_is_arb(self):
        """Explicitly passing token_bucket=False must also use ARB path."""
        manager = BucketManager(token_bucket=False)
        self.assertFalse(manager.token_bucket)
        self.assertTrue(len(manager.buckets) > 0)


if __name__ == "__main__":
    unittest.main()
