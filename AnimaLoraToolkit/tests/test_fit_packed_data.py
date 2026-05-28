import pathlib
import sys
import types
import unittest
from types import SimpleNamespace


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch
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

import trainer.data as data_module
from trainer.config import DEFAULTS, apply_yaml_config

if INSTALLED_FAKE_TORCH:
    sys.modules.pop("torch.utils.data", None)
    sys.modules.pop("torch.utils", None)
    sys.modules.pop("torch", None)


class NativeFitSizingTests(unittest.TestCase):
    def test_native_fit_size_preserves_source_when_aligned(self):
        plan = data_module.plan_native_fit_image(
            640, 384, max_tokens=1024, patch_size=2, vae_downsample=8
        )

        self.assertEqual((plan.width, plan.height), (640, 384))
        self.assertEqual((plan.source_width, plan.source_height), (640, 384))
        self.assertFalse(plan.was_resized)
        self.assertFalse(plan.was_padded)
        self.assertEqual(plan.token_count, 960)

    def test_native_fit_size_pads_unaligned_source_without_scaling(self):
        plan = data_module.plan_native_fit_image(
            513, 777, max_tokens=65536, patch_size=2, vae_downsample=8
        )

        self.assertEqual((plan.width, plan.height), (528, 784))
        self.assertEqual((plan.source_width, plan.source_height), (513, 777))
        self.assertTrue(plan.was_padded)
        self.assertFalse(plan.was_resized)

    def test_native_fit_over_budget_fails_by_default(self):
        with self.assertRaisesRegex(ValueError, "exceeds fit_max_tokens"):
            data_module.plan_native_fit_image(
                8192, 8192, max_tokens=65536, patch_size=2, vae_downsample=8
            )

    def test_native_fit_skip_uses_same_over_budget_signal(self):
        with self.assertRaisesRegex(ValueError, "exceeds fit_max_tokens"):
            data_module.plan_native_fit_image(
                8192,
                8192,
                max_tokens=65536,
                patch_size=2,
                vae_downsample=8,
                over_budget_strategy="skip",
            )

    def test_yaml_maps_fit_packed_flags(self):
        args = SimpleNamespace(**DEFAULTS)

        apply_yaml_config(args, {
            "fit_packed_training": True,
            "fit_max_tokens": 65536,
            "fit_warn_tokens": 16384,
            "fit_over_budget_strategy": "fail",
            "fit_align_mode": "pad",
            "fit_max_tokens_per_batch": 70000,
        })

        self.assertTrue(args.fit_packed_training)
        self.assertEqual(args.fit_max_tokens, 65536)
        self.assertEqual(args.fit_warn_tokens, 16384)
        self.assertEqual(args.fit_over_budget_strategy, "fail")
        self.assertEqual(args.fit_align_mode, "pad")
        self.assertEqual(args.fit_max_tokens_per_batch, 70000)


@unittest.skipUnless(HAS_TORCH, "torch is required for collate/sampler tests")
class NativeFitBatchingTests(unittest.TestCase):
    def test_collate_native_fit_pixels_pads_to_batch_max_and_keeps_token_counts(self):
        batch = [
            {
                "pixel_values": torch.ones(3, 32, 48),
                "pixel_mask": torch.ones(1, 32, 48),
                "caption": "a",
                "image": "a.png",
                "token_count": 6,
            },
            {
                "pixel_values": torch.ones(3, 48, 64),
                "pixel_mask": torch.ones(1, 48, 64),
                "caption": "b",
                "image": "b.png",
                "token_count": 12,
            },
        ]

        out = data_module.collate_fn_fit_packed(batch)

        self.assertEqual(tuple(out["pixel_values"].shape), (2, 3, 48, 64))
        self.assertEqual(tuple(out["pixel_mask"].shape), (2, 1, 48, 64))
        self.assertEqual(out["fit_token_counts"].tolist(), [6, 12])
        self.assertEqual(out["captions"], ["a", "b"])

    def test_fit_token_batch_sampler_groups_similar_token_counts(self):
        dataset = SimpleNamespace(token_count_for_index=[1024, 65536, 960, 64000])
        sampler = data_module.FitTokenBatchSampler(
            dataset, batch_size=2, max_tokens_per_batch=70000, shuffle=False
        )

        self.assertEqual(list(sampler), [[2, 0], [3], [1]])

    def test_fit_token_batch_sampler_len_uses_deterministic_packing(self):
        dataset = SimpleNamespace(token_count_for_index=[1024, 65536, 960, 64000])
        sampler = data_module.FitTokenBatchSampler(
            dataset, batch_size=2, max_tokens_per_batch=70000, shuffle=True, seed=7
        )

        self.assertEqual(len(sampler), 3)

    def test_image_dataset_fit_mode_pads_source_without_resampling(self):
        import tempfile
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            image_path = root / "sample.png"
            Image.new("RGB", (17, 19), (255, 0, 0)).save(image_path)
            image_path.with_suffix(".txt").write_text("tag", encoding="utf-8")

            dataset = data_module.ImageDataset(
                root,
                fit_packed=True,
                fit_max_tokens=16,
                fit_patch_size=2,
                fit_vae_downsample=8,
            )

            item = dataset[0]

            self.assertEqual(tuple(item["pixel_values"].shape), (3, 32, 32))
            self.assertEqual(tuple(item["pixel_mask"].shape), (1, 32, 32))
            self.assertEqual(float(item["pixel_mask"][:, :19, :17].sum()), 19 * 17)
            self.assertEqual(float(item["pixel_mask"][:, 19:, :].sum()), 0.0)
            self.assertEqual(item["token_count"], 4)


if __name__ == "__main__":
    unittest.main()
