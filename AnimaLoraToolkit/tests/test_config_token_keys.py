import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trainer.config import YAML_TO_ARGS

EXPECTED_KEYS = [
    "token_bucket",
    "token_bucket_counts",
    "token_bucket_max_aspect_ratio",
    "token_bucket_min_dim",
    "token_bucket_max_dim",
    "torch_compile",
    "compile_mode",
]


class TestConfigTokenKeys(unittest.TestCase):
    def test_all_token_bucket_and_compile_keys_in_yaml_to_args(self):
        for key in EXPECTED_KEYS:
            with self.subTest(key=key):
                self.assertIn(
                    key,
                    YAML_TO_ARGS,
                    f"Key '{key}' missing from YAML_TO_ARGS",
                )

    def test_identity_mappings(self):
        for key in EXPECTED_KEYS:
            with self.subTest(key=key):
                self.assertEqual(
                    YAML_TO_ARGS[key],
                    key,
                    f"YAML_TO_ARGS['{key}'] should map to itself, got {YAML_TO_ARGS.get(key)!r}",
                )


if __name__ == "__main__":
    unittest.main()
