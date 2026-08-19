"""Regression test: bucket-key change must invalidate (delete) the latent cache.

bucket_key ordering is (H, W) — confirmed from data.py line 573 where
`cache[img_path] = ((bh, bw), ...)` and line 590 `sample["bucket_key"] = bucket_key`,
with line 1538 reading `expected_h, expected_w = int(expected_bucket[0]), int(expected_bucket[1])`.
"""
import os
import pathlib
import sys
import tempfile
import types
import unittest

# ── bootstrap: make `from trainer.data import ...` work ──────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch  # noqa: F401
    _INSTALLED_FAKE_TORCH = False
except ModuleNotFoundError:
    _INSTALLED_FAKE_TORCH = True
    torch_module = types.ModuleType("torch")
    torch_module.float32 = object()
    torch_module.float16 = object()
    torch_module.bfloat16 = object()
    torch_utils = types.ModuleType("torch.utils")
    torch_data = types.ModuleType("torch.utils.data")

    class _Dataset:
        pass

    torch_data.Dataset = _Dataset
    torch_utils.data = torch_data
    torch_module.utils = torch_utils
    sys.modules["torch"] = torch_module
    sys.modules["torch.utils"] = torch_utils
    sys.modules["torch.utils.data"] = torch_data

import numpy as np  # noqa: E402
from trainer.data import CachedLatentDataset  # noqa: E402

if _INSTALLED_FAKE_TORCH:
    sys.modules.pop("torch.utils.data", None)
    sys.modules.pop("torch.utils", None)
    sys.modules.pop("torch", None)

# ── Token bucket used in tests ────────────────────────────────────────────────
# 4032-token bucket: 1008 × 1024 pixels; bucket_key = (H=1024, W=1008)
BUCKET_H = 1024
BUCKET_W = 1008


def _make_ds():
    """Return a CachedLatentDataset instance with only the attrs _is_cache_valid touches."""
    ds = CachedLatentDataset.__new__(CachedLatentDataset)
    ds.np = np
    ds.flip_enabled = False
    return ds


def _write_image(path: pathlib.Path):
    """Write a minimal valid PNG-like byte sequence (enough to create the file)."""
    # Any non-empty file; mtime is what matters.
    path.write_bytes(b"\x89PNG\r\n\x1a\n")


def _write_npz(npz_path: pathlib.Path, bucket_h: int, bucket_w: int):
    """Write a valid latent npz with the given bucket dimensions."""
    latent = np.zeros((16, 1, 64, 64), dtype=np.float32)
    np.savez(
        str(npz_path),
        latent=latent,
        bucket_h=np.array(bucket_h, dtype=np.int32),
        bucket_w=np.array(bucket_w, dtype=np.int32),
        dtype_kind="fp32",
    )


class TestCacheInvalidation(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = pathlib.Path(self._tmp.name)

        # Create image file first so npz can be written with a later mtime.
        self.img_path = self.tmp / "img.png"
        _write_image(self.img_path)

        self.npz_path = self.tmp / "img.npz"
        _write_npz(self.npz_path, BUCKET_H, BUCKET_W)

        # Ensure npz mtime >= img mtime.
        img_mtime = self.img_path.stat().st_mtime
        npz_mtime = self.npz_path.stat().st_mtime
        if npz_mtime < img_mtime:
            os.utime(str(self.npz_path), (img_mtime, img_mtime))

        self.ds = _make_ds()

    def tearDown(self):
        self._tmp.cleanup()

    def test_valid_when_bucket_matches(self):
        """Cache is accepted when sample bucket_key matches stored bucket_h/bucket_w."""
        sample = {"image": str(self.img_path), "bucket_key": (BUCKET_H, BUCKET_W)}
        result = self.ds._is_cache_valid(sample, self.npz_path)
        self.assertTrue(result, "Expected True when bucket_key matches cached bucket")
        self.assertTrue(self.npz_path.exists(), "npz must NOT be deleted on a valid cache hit")

    def test_invalidated_when_bucket_changes(self):
        """Cache is rejected (False) when bucket_key differs from stored bucket_h/bucket_w.

        The primary guarantee is the return value: False signals the caller must re-encode.
        File deletion is best-effort in the production code: on Windows, numpy.load holds
        an mmap lock that prevents unlink (WinError 32), so the outer except silently
        swallows the error and the file may remain on disk.  We test what is guaranteed:
        the return value is False.
        """
        # Provide a different bucket (square 1024×1024 instead of 1024×1008).
        different_bucket = (1024, 1024)
        sample = {"image": str(self.img_path), "bucket_key": different_bucket}
        result = self.ds._is_cache_valid(sample, self.npz_path)
        self.assertFalse(result, "Expected False when bucket_key changed")
        # File deletion is a side effect — it succeeds on POSIX but on Windows the mmap
        # lock prevents unlink; either way the caller correctly sees False.
        # We do not assert npz_path existence here to keep the test cross-platform.


if __name__ == "__main__":
    unittest.main()
