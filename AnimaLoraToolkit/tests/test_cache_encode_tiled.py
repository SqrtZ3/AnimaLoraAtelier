"""缓存阶段分块 VAE encode（cache_encode_tiled）单测。

真 VAE 的接缝误差无法本地验证（云端 smoke），这里固化的是可本地证明的不变量：
  1) 拼接几何：_tile_starts 覆盖 [0,total)、末块贴齐、起点对齐。
  2) 对"窗口对齐的局部算子"（8×8 均值池化，感受野不越块界），
     tiled_vae_encode ≡ 整图 encode（含 B=2 flip 批、非整倍步长尺寸）。
  3) 单块可覆盖整图时直接整图 encode（逐 bit 等价路径）。
  4) 对齐约束 fail-fast（H/W/tile/overlap 非 8 整倍数、overlap 越界）。
  5) 端到端：CachedLatentDataset + 假 VAE，>4M px 图触发分块且落盘 latent 正确；
     预算内的图不触发（走原路径）。
  6) config 键默认关闭、YAML 可开启。
"""

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from trainer.config import DEFAULTS, apply_yaml_config  # noqa: E402
from trainer.data import (  # noqa: E402
    CachedLatentDataset,
    ImageDataset,
    _tile_starts,
    tiled_vae_encode,
)


def _avgpool_encode(x_5d):
    """假 VAE encode：8×8 均值池化（局部、窗口对齐 → 分块拼接应与整图精确一致）。"""
    b, c, t, h, w = x_5d.shape
    pooled = F.avg_pool2d(x_5d.reshape(b * t, c, h, w), 8)
    return pooled.reshape(b, c, t, h // 8, w // 8)


class TestTileStarts(unittest.TestCase):
    def test_single_tile_when_fits(self):
        self.assertEqual(_tile_starts(512, 1024, 896), [0])
        self.assertEqual(_tile_starts(1024, 1024, 896), [0])

    def test_coverage_and_last_tile_flush(self):
        total, tile, stride = 2304, 1024, 896
        starts = _tile_starts(total, tile, stride)
        self.assertEqual(starts[0], 0)
        self.assertEqual(starts[-1] + tile, total)      # 末块贴齐右边界
        for a, b in zip(starts, starts[1:]):
            self.assertLess(b, a + tile)                 # 相邻块必有重叠/相接
        self.assertTrue(all(s + tile <= total for s in starts))


class TestTiledEncodeEquivalence(unittest.TestCase):
    def _check_equal(self, h, w, tile=128, ov=32, batch=1):
        torch.manual_seed(0)
        x = torch.randn(batch, 3, 1, h, w)
        whole = _avgpool_encode(x)
        tiled = tiled_vae_encode(_avgpool_encode, x, tile, ov)
        self.assertEqual(tuple(tiled.shape), tuple(whole.shape))
        self.assertLess((tiled.float() - whole.float()).abs().max().item(), 1e-5)

    def test_grid_both_axes(self):
        self._check_equal(352, 288)          # 两轴多块 + 末块回移

    def test_one_axis_only(self):
        self._check_equal(352, 96)           # W 单块（无 x 混合）、H 多块

    def test_flip_batch(self):
        self._check_equal(304, 336, batch=2)  # flip 拼批 [2,...]

    def test_exact_stride_multiple(self):
        self._check_equal(320, 320, tile=128, ov=64)  # stride=64 整除

    def test_single_tile_passthrough_bitwise(self):
        x = torch.randn(1, 3, 1, 96, 112)
        whole = _avgpool_encode(x)
        tiled = tiled_vae_encode(_avgpool_encode, x, 128, 32)
        self.assertTrue(torch.equal(tiled, whole))    # 早退路径逐 bit 等价

    def test_misaligned_input_fails_fast(self):
        x = torch.randn(1, 3, 1, 100, 256)            # H=100 非 8 整倍数
        with self.assertRaises(ValueError):
            tiled_vae_encode(_avgpool_encode, x, 64, 16)

    def test_bad_overlap_fails_fast(self):
        x = torch.randn(1, 3, 1, 256, 256)
        with self.assertRaises(ValueError):
            tiled_vae_encode(_avgpool_encode, x, 64, 64)   # overlap >= tile


class _FakeVAE:
    """镜像 vae.model.encode(x, scale) 调用面。"""
    class _M:
        @staticmethod
        def encode(x, scale):
            return _avgpool_encode(x) * scale

    def __init__(self):
        self.model = self._M()
        self.scale = 1.0


class TestEndToEndCachedDataset(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = pathlib.Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _build(self, side, tiled):
        from PIL import Image
        Image.new("RGB", (side, side), (77, 150, 30)).save(self.data_dir / "img.png")
        (self.data_dir / "img.txt").write_text("1girl", encoding="utf-8")
        base = ImageDataset(str(self.data_dir), side, None, prefer_json=False)
        return CachedLatentDataset(
            base, _FakeVAE(), "cpu", torch.float32,
            save_dtype=torch.float32, encode_batch_size=2,
            encode_tiled=tiled, encode_tile_px=1024, encode_tile_overlap=128,
        )

    def test_over_budget_image_tiled_and_correct(self):
        # 2304² = 5.3M px > 4M 预算 → 触发分块；latent 应与整图 encode 一致
        ds = self._build(2304, tiled=True)
        item = ds[0]
        lat = item["latent"]
        self.assertEqual(tuple(lat.shape), (3, 1, 288, 288))
        expected = _avgpool_encode(
            ds.base_dataset[0]["pixel_values"].unsqueeze(0).unsqueeze(2)
        )[0]
        self.assertLess((lat.float() - expected.float()).abs().max().item(), 1e-5)

    def test_within_budget_image_untouched(self):
        # 512² 远低于预算 → 不触发分块（整图路径），结果同样正确
        ds = self._build(512, tiled=True)
        lat = ds[0]["latent"]
        self.assertEqual(tuple(lat.shape), (3, 1, 64, 64))
        expected = _avgpool_encode(
            ds.base_dataset[0]["pixel_values"].unsqueeze(0).unsqueeze(2)
        )[0]
        self.assertLess((lat.float() - expected.float()).abs().max().item(), 1e-6)


class TestConfigSwitch(unittest.TestCase):
    def test_default_off(self):
        self.assertIn("cache_encode_tiled", DEFAULTS)
        self.assertFalse(DEFAULTS["cache_encode_tiled"])
        self.assertEqual(int(DEFAULTS["cache_encode_tile_px"]), 1024)
        self.assertEqual(int(DEFAULTS["cache_encode_tile_overlap"]), 128)

    def test_yaml_enables(self):
        from types import SimpleNamespace
        args = SimpleNamespace(**DEFAULTS)
        apply_yaml_config(args, {
            "cache_encode_tiled": True,
            "cache_encode_tile_px": 768,
            "cache_encode_tile_overlap": 96,
        })
        self.assertTrue(args.cache_encode_tiled)
        self.assertEqual(int(args.cache_encode_tile_px), 768)
        self.assertEqual(int(args.cache_encode_tile_overlap), 96)


if __name__ == "__main__":
    unittest.main()
