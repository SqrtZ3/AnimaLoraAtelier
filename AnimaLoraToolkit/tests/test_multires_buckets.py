import inspect
import pathlib
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from PIL import Image
except ModuleNotFoundError:  # pragma: no cover - pillow is part of the trainer env
    Image = None

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

import trainer.data as data_module
from trainer.data import BucketManager, CachedLatentDataset, ImageDataset
from trainer.config import DEFAULTS, apply_yaml_config

if INSTALLED_FAKE_TORCH:
    sys.modules.pop("torch.utils.data", None)
    sys.modules.pop("torch.utils", None)
    sys.modules.pop("torch", None)


class _FakeArray:
    def __init__(self, shape):
        self.shape = shape
        self.ndim = len(shape)


class _FakeFinite:
    def all(self):
        return True


class _FakeNumpy:
    def __init__(self, files):
        self._files = files

    def load(self, _path):
        return self

    def __getitem__(self, key):
        return self._files[key]

    @property
    def files(self):
        return list(self._files.keys())

    def isfinite(self, _value):
        return _FakeFinite()


class MultiResolutionBucketManagerTests(unittest.TestCase):
    def test_bucket_manager_exposes_automatic_base_range_options(self):
        params = inspect.signature(BucketManager).parameters

        self.assertIn("min_base_reso", params)
        self.assertIn("max_base_reso", params)
        self.assertIn("base_reso_step", params)

    def test_generates_base_range_when_explicit_list_is_empty(self):
        manager = BucketManager(
            base_reso=1024,
            base_resos=[],
            min_base_reso=512,
            max_base_reso=2048,
            base_reso_step=256,
            min_reso=512,
            max_reso=3072,
            step=64,
        )

        self.assertEqual(manager.base_resos, [512, 768, 1024, 1280, 1536, 1792, 2048])
        self.assertIn((1792, 1792), manager.buckets)
        self.assertIn((2048, 2048), manager.buckets)

    def test_max_base_reso_defaults_to_bucket_minimum_start(self):
        manager = BucketManager(
            base_reso=1024,
            base_resos=[],
            max_base_reso=1024,
            base_reso_step=256,
            min_reso=512,
            max_reso=2048,
            step=64,
        )

        self.assertEqual(manager.base_resos, [512, 768, 1024])

    def test_explicit_base_resos_override_automatic_base_range(self):
        manager = BucketManager(
            base_reso=1024,
            base_resos=[768, 1024],
            min_base_reso=512,
            max_base_reso=2048,
            base_reso_step=256,
            min_reso=512,
            max_reso=2048,
            step=64,
        )

        self.assertEqual(manager.base_resos, [768, 1024])

    def test_generates_multiple_area_levels_and_avoids_upscaling(self):
        manager = BucketManager(
            base_reso=1024,
            base_resos=[512, 768, 1024],
            min_reso=512,
            max_reso=2048,
            step=64,
            no_upscale=True,
        )

        self.assertIn((512, 512), manager.buckets)
        self.assertIn((768, 768), manager.buckets)
        self.assertIn((1024, 1024), manager.buckets)
        self.assertEqual(manager.get_bucket(600, 600), (512, 512))
        self.assertEqual(manager.get_bucket(900, 900), (768, 768))
        self.assertEqual(manager.get_bucket(1400, 1400), (1024, 1024))

    def test_max_upscale_allows_small_controlled_enlargement(self):
        manager = BucketManager(
            base_reso=1024,
            base_resos=[512, 640, 768, 1024],
            min_reso=512,
            max_reso=2048,
            step=64,
            max_upscale=1.25,
        )

        self.assertEqual(manager.get_bucket(600, 600), (640, 640))
        self.assertEqual(manager.get_bucket(900, 900), (1024, 1024))


class MultiResolutionConfigTests(unittest.TestCase):
    def test_yaml_maps_multi_resolution_bucket_policy(self):
        args = SimpleNamespace(**DEFAULTS)

        apply_yaml_config(args, {
            "bucket_base_resos": [512, 768, 1024],
            "bucket_min_base_reso": 512,
            "bucket_max_base_reso": 2048,
            "bucket_base_reso_steps": 256,
            "bucket_no_upscale": True,
            "bucket_max_upscale": 1.25,
            "bucket_report": True,
        })

        self.assertEqual(args.bucket_base_resos, [512, 768, 1024])
        self.assertEqual(args.bucket_min_base_reso, 512)
        self.assertEqual(args.bucket_max_base_reso, 2048)
        self.assertEqual(args.bucket_base_reso_steps, 256)
        self.assertTrue(args.bucket_no_upscale)
        self.assertEqual(args.bucket_max_upscale, 1.25)
        self.assertTrue(args.bucket_report)


class BucketReportTests(unittest.TestCase):
    def test_formats_bucket_report_with_source_sizes_and_largest_downscales(self):
        self.assertTrue(
            hasattr(data_module, "format_bucket_report"),
            "trainer.data should expose format_bucket_report",
        )
        samples = [
            {"image": pathlib.Path("small.png"), "source_size": (600, 600), "bucket_key": (512, 512)},
            {"image": pathlib.Path("large.png"), "source_size": (4096, 4096), "bucket_key": (2048, 2048)},
            {"image": pathlib.Path("wide.png"), "source_size": (4096, 2048), "bucket_key": (1448, 2896)},
        ]

        report = data_module.format_bucket_report(samples, limit=2, label="train")

        self.assertIn("[BucketReport:train]", report)
        self.assertIn("2048x2048", report)
        self.assertIn("4096x4096 -> 2048x2048", report)
        self.assertIn("largest downscales", report)


@unittest.skipUnless(Image is not None and HAS_TORCH, "PIL and torch are required")
class MultiResolutionImageDatasetTests(unittest.TestCase):
    def test_no_upscale_dataset_keeps_small_crop_on_smaller_bucket(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = pathlib.Path(tmp)
            Image.new("RGB", (600, 600), (255, 0, 0)).save(data_dir / "detail.png")
            (data_dir / "detail.txt").write_text("detail crop", encoding="utf-8")

            manager = BucketManager(
                base_reso=1024,
                base_resos=[512, 768, 1024],
                min_reso=512,
                max_reso=2048,
                step=64,
                no_upscale=True,
            )
            dataset = ImageDataset(
                data_dir,
                resolution=1024,
                bucket_mgr=manager,
                prefer_json=False,
            )

            item = dataset[0]

        self.assertEqual(tuple(item["pixel_values"].shape), (3, 512, 512))
        self.assertEqual(dataset.bucket_for_index[0], (512, 512))


class CachedLatentBucketPolicyTests(unittest.TestCase):
    def test_cache_is_invalid_when_saved_bucket_differs_from_current_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            img_path = pathlib.Path(tmp) / "detail.png"
            img_path.write_bytes(b"fake image bytes")
            npz_path = img_path.with_suffix(".npz")
            npz_path.write_bytes(b"fake npz bytes")

            dataset = object.__new__(CachedLatentDataset)
            dataset.np = _FakeNumpy({
                "latent": _FakeArray((16, 1, 128, 128)),
                "bucket_w": 1024,
                "bucket_h": 1024,
                "dtype_kind": "fp32",
            })

            valid = dataset._is_cache_valid(
                {"image": img_path, "bucket_key": (512, 512)},
                npz_path,
            )

        self.assertFalse(valid)
        self.assertFalse(npz_path.exists())

    def test_legacy_cache_without_bucket_metadata_uses_latent_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            img_path = pathlib.Path(tmp) / "detail.png"
            img_path.write_bytes(b"fake image bytes")
            npz_path = img_path.with_suffix(".npz")
            npz_path.write_bytes(b"fake npz bytes")

            dataset = object.__new__(CachedLatentDataset)
            dataset.np = _FakeNumpy({
                "latent": _FakeArray((16, 1, 128, 128)),
                "dtype_kind": "fp32",
            })

            valid = dataset._is_cache_valid(
                {"image": img_path, "bucket_key": (512, 512)},
                npz_path,
            )

        self.assertFalse(valid)
        self.assertFalse(npz_path.exists())


if __name__ == "__main__":
    unittest.main()
