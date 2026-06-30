"""navit 原生定尺寸（navit_native_resolution）相关单测。

覆盖三块新逻辑：
  1) anima_train.scan_max_image_side：跨 data/reg 目录取最大单边，忽略非图片，缺失目录安全跳过。
  2) 原生 floor 定尺寸 + 预算打包的数据链路不变量：plan_native_fit_image(floor) 零 padding、
     16px 整倍数、尺寸 ≤ 源；得到的异构 token 数能被 pack_indices_by_budget 在预算内无重叠覆盖。
  3) config：navit_native_resolution 默认 False 且能经 YAML 正确开启。
"""

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trainer.config import DEFAULTS, apply_yaml_config
from trainer.data import plan_native_fit_image, pack_indices_by_budget


def _write_png(path, w, h):
    from PIL import Image
    Image.new("RGB", (int(w), int(h)), (123, 222, 64)).save(path)


class TestScanMaxImageSide(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_max_side_across_dirs_and_ignores_non_images(self):
        import anima_train
        data_dir = self.root / "data"
        reg_dir = self.root / "reg"
        data_dir.mkdir()
        reg_dir.mkdir()
        _write_png(data_dir / "a.png", 800, 1200)   # 单边 1200
        _write_png(data_dir / "b.jpg", 1536, 512)   # 单边 1536
        _write_png(reg_dir / "c.webp", 2048, 768)   # 单边 2048（在 reg 里）
        (data_dir / "notes.txt").write_text("not an image", encoding="utf-8")

        self.assertEqual(
            anima_train.scan_max_image_side([str(data_dir), str(reg_dir)]), 2048
        )
        # 空 / 缺失目录安全跳过，仍只看 data
        self.assertEqual(
            anima_train.scan_max_image_side([str(data_dir), "", str(self.root / "nope")]),
            1536,
        )

    def test_empty_returns_zero(self):
        import anima_train
        self.assertEqual(anima_train.scan_max_image_side([None, ""]), 0)


class TestNativeFloorSizingAndPacking(unittest.TestCase):
    # 一组异构原生尺寸（含非 16 整倍数的）。floor 对齐后应裁到 16 整倍数、零 padding。
    SIZES = [(1000, 1500), (1536, 512), (777, 777), (2048, 768), (640, 1664)]

    def test_floor_alignment_invariants(self):
        for w, h in self.SIZES:
            plan = plan_native_fit_image(w, h, align_mode="floor", max_tokens=10 ** 9)
            # 16px 整倍数（VAE 下采样 8 × patch 2）
            self.assertEqual(plan.width % 16, 0)
            self.assertEqual(plan.height % 16, 0)
            # floor 只会裁小、不放大
            self.assertLessEqual(plan.width, w)
            self.assertLessEqual(plan.height, h)
            # 每边裁掉的像素 < 16（仅去掉不足一个对齐单元的余数）
            self.assertLess(w - plan.width, 16)
            self.assertLess(h - plan.height, 16)
            # 零 gray padding（navit 缓存路径无 mask 的前提）：floor 下规划尺寸 ≤ 源，
            # __getitem__ 把图裁到规划尺寸再贴满同尺寸画布 → 有效区填满整图 → mask 全 1。
            # 即 min(source, planned) == planned，对应有效像素覆盖整个 planned 网格。
            self.assertEqual(min(plan.source_width, plan.width), plan.width)
            self.assertEqual(min(plan.source_height, plan.height), plan.height)
            # token 数与 latent patch 网格一致
            self.assertEqual(plan.token_count, (plan.width // 16) * (plan.height // 16))

    def test_heterogeneous_packs_within_budget_and_cover(self):
        counts = [
            plan_native_fit_image(w, h, align_mode="floor", max_tokens=10 ** 9).token_count
            for (w, h) in self.SIZES
        ]
        budget = max(counts) * 2  # 保证每图都能进包、且能拼多张
        order = list(range(len(counts)))
        packs = pack_indices_by_budget(counts, budget, order)
        # 覆盖且仅覆盖每个 index 一次
        flat = [i for p in packs for i in p]
        self.assertEqual(sorted(flat), order)
        # 每个包的 token 数之和不超预算
        for p in packs:
            self.assertLessEqual(sum(counts[i] for i in p), budget)


class TestConfigSwitch(unittest.TestCase):
    def test_default_off(self):
        self.assertIn("navit_native_resolution", DEFAULTS)
        self.assertFalse(DEFAULTS["navit_native_resolution"])

    def test_yaml_enables(self):
        from types import SimpleNamespace
        args = SimpleNamespace(**DEFAULTS)
        apply_yaml_config(args, {"navit_native_resolution": True})
        self.assertTrue(args.navit_native_resolution)


if __name__ == "__main__":
    unittest.main()
