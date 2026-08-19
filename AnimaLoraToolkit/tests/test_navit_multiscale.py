"""navit 多尺度阶梯（navit_multiscale）单测。

覆盖：
  1) plan_multiscale_copy：只降不升采样、16px 整倍数、token 数 ≤ 目标档、
     等比（长宽比保持）、极端长宽比下不超预算。
  2) ImageDataset 展开：确定性追加副本条目（每图每命中档一条）、per-index 结构对齐、
     __getitem__ 副本像素尺寸 = 规划尺寸且 mask 全 1（navit 无 mask 前提）。
  3) CachedLatentDataset._get_npz_path：副本 sidecar 命名与原生互不覆盖。
  4) collate_fn_navit_pack：逐图 ms 标志透传。
  5) config：默认关闭、YAML 可开启。
"""

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from trainer.config import DEFAULTS, apply_yaml_config  # noqa: E402
from trainer.data import (  # noqa: E402
    CachedLatentDataset,
    ImageDataset,
    collate_fn_navit_pack,
    plan_multiscale_copy,
    plan_native_fit_image,
)


class TestPlanMultiscaleCopy(unittest.TestCase):
    def test_basic_downscale_invariants(self):
        # 用户场景：2814x4456 原生 ≈ 48650 token，缩到 ≤4096 token 档
        plan = plan_multiscale_copy(2814, 4456, 4096)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.width % 16, 0)
        self.assertEqual(plan.height % 16, 0)
        self.assertLessEqual(plan.token_count, 4096)
        self.assertEqual(plan.token_count, (plan.width // 16) * (plan.height // 16))
        # 等比：副本长宽比与源一致（16px 量化容差内）
        src_ar = 2814 / 4456
        ms_ar = plan.width / plan.height
        self.assertLess(abs(ms_ar - src_ar) / src_ar, 0.05)
        # 有效区 = 整张（navit 缓存路径 mask 恒全 1 的前提）
        self.assertEqual(plan.source_width, plan.width)
        self.assertEqual(plan.source_height, plan.height)
        self.assertTrue(plan.was_resized)
        self.assertFalse(plan.was_padded)

    def test_never_upscales_or_duplicates_native(self):
        # 源 token 数 == 目标档 → 不产出副本（避免与原生重复）
        self.assertIsNone(plan_multiscale_copy(1024, 1024, 4096))
        # 源比目标档还小 → 绝不上采样
        self.assertIsNone(plan_multiscale_copy(512, 512, 4096))
        # 非法输入
        self.assertIsNone(plan_multiscale_copy(0, 512, 4096))
        self.assertIsNone(plan_multiscale_copy(512, 512, 0))

    def test_strictly_larger_source_gets_copy(self):
        # 1040x1040 → floor 后 65*65=4225 > 4096 → 应产出 ≤4096 的副本
        plan = plan_multiscale_copy(1040, 1040, 4096)
        self.assertIsNotNone(plan)
        self.assertLessEqual(plan.token_count, 4096)

    def test_extreme_aspect_ratio_stays_within_budget(self):
        # 极扁图：8000x160 → 源 500*10=5000 token > 目标 4096
        plan = plan_multiscale_copy(8000, 160, 4096)
        self.assertIsNotNone(plan)
        self.assertLessEqual(plan.token_count, 4096)
        self.assertEqual(plan.width % 16, 0)
        self.assertEqual(plan.height % 16, 0)
        self.assertGreaterEqual(plan.token_h, 1)
        self.assertGreaterEqual(plan.token_w, 1)

    def test_ladder_tokens_below_native(self):
        # 副本 token 数必须严格小于原生（展开逻辑的跳过条件依赖这一点）
        native = plan_native_fit_image(2814, 4456, align_mode="floor", max_tokens=10 ** 9)
        plan = plan_multiscale_copy(2814, 4456, 16384)
        self.assertLess(plan.token_count, native.token_count)


class TestImageDatasetExpansion(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = pathlib.Path(self._tmp.name)
        from PIL import Image
        # big.png：704x1120 → 原生 44*70=3080 token（> 1024 档 → 应有副本）
        Image.new("RGB", (704, 1120), (200, 30, 90)).save(self.data_dir / "big.png")
        (self.data_dir / "big.txt").write_text("1girl, solo", encoding="utf-8")
        # small.png：288x288 → 18*18=324 token（≤ 1024 档 → 跳过）
        Image.new("RGB", (288, 288), (10, 130, 240)).save(self.data_dir / "small.png")
        (self.data_dir / "small.txt").write_text("1girl, sitting", encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def _make_dataset(self, ladder):
        return ImageDataset(
            str(self.data_dir), 1024, None,
            prefer_json=False,
            fit_packed=True,
            fit_max_tokens=10 ** 9,
            fit_warn_tokens=0,
            fit_min_tokens=0,
            fit_align_mode="floor",
            navit_ms_token_ladder=ladder,
        )

    def test_deterministic_expansion_and_index_alignment(self):
        ds = self._make_dataset([1024])
        # 2 原生 + 1 副本（仅 big 命中 1024 档）
        self.assertEqual(len(ds.samples), 3)
        ms = [s for s in ds.samples if s.get("ms_tokens_target")]
        self.assertEqual(len(ms), 1)
        self.assertEqual(ms[0]["ms_tokens_target"], 1024)
        self.assertTrue(str(ms[0]["image"]).endswith("big.png"))
        self.assertLessEqual(ms[0]["token_count"], 1024)
        # per-index 结构与 samples 对齐（打包器 / 缓存按索引取）
        self.assertEqual(len(ds.bucket_for_index), 3)
        self.assertEqual(len(ds.token_count_for_index), 3)
        for s, tc in zip(ds.samples, ds.token_count_for_index):
            self.assertEqual(int(s["token_count"]), int(tc))

    def test_ladder_off_is_neutral(self):
        ds = self._make_dataset(None)
        self.assertEqual(len(ds.samples), 2)
        self.assertFalse(any(s.get("ms_tokens_target") for s in ds.samples))

    def test_getitem_ms_copy_full_valid_mask(self):
        ds = self._make_dataset([1024])
        ms_idx = next(i for i, s in enumerate(ds.samples) if s.get("ms_tokens_target"))
        item = ds[ms_idx]
        plan = ds.samples[ms_idx]["fit_plan"]
        # 像素张量 = 规划尺寸（副本经 resize-cover + 中心裁剪，不 padding）
        self.assertEqual(tuple(item["pixel_values"].shape), (3, plan.height, plan.width))
        # mask 全 1：navit 缓存路径不携带 padding mask 的硬前提
        self.assertTrue(bool(torch.all(item["pixel_mask"] > 0.5)))
        self.assertEqual(int(item["token_count"]), int(plan.token_count))
        # caption 与原生共享
        self.assertIn("1girl", item["caption"])

    def test_native_entries_unchanged_by_expansion(self):
        ds_off = self._make_dataset(None)
        ds_on = self._make_dataset([1024])
        natives_on = [s for s in ds_on.samples if not s.get("ms_tokens_target")]
        self.assertEqual(len(natives_on), len(ds_off.samples))
        for a, b in zip(
            sorted(ds_off.samples, key=lambda s: str(s["image"])),
            sorted(natives_on, key=lambda s: str(s["image"])),
        ):
            self.assertEqual(a["bucket_key"], b["bucket_key"])
            self.assertEqual(a["token_count"], b["token_count"])


class TestNpzSidecarNaming(unittest.TestCase):
    def test_ms_copy_gets_own_sidecar(self):
        native = {"image": r"D:\ds\pic.png"}
        ms = {"image": r"D:\ds\pic.png", "ms_tokens_target": 4096}
        p_native = CachedLatentDataset._get_npz_path(None, native)
        p_ms = CachedLatentDataset._get_npz_path(None, ms)
        self.assertEqual(p_native.name, "pic.npz")
        self.assertEqual(p_ms.name, "pic.ms4096.npz")
        self.assertNotEqual(p_native, p_ms)

    def test_bare_path_backward_compatible(self):
        p = CachedLatentDataset._get_npz_path(None, r"D:\ds\pic.png")
        self.assertEqual(p.name, "pic.npz")


class TestCollateMsFlags(unittest.TestCase):
    def test_flags_passthrough(self):
        lat = torch.zeros(16, 1, 8, 8)
        batch = [
            {"latent": lat, "caption": "a", "image": "a.png", "ms_tokens_target": 0},
            {"latent": lat, "caption": "b", "image": "b.png", "ms_tokens_target": 4096},
            {"latent": lat, "caption": "c", "image": "c.png"},  # 旧条目无键 → False
        ]
        out = collate_fn_navit_pack(batch)
        self.assertEqual(out["navit_ms_flags"], [False, True, False])
        self.assertEqual(len(out["navit_latents"]), 3)


class TestConfigSwitch(unittest.TestCase):
    def test_default_off(self):
        self.assertIn("navit_multiscale", DEFAULTS)
        self.assertFalse(DEFAULTS["navit_multiscale"])
        self.assertEqual(str(DEFAULTS["navit_multiscale_token_ladder"]), "4096")
        self.assertEqual(float(DEFAULTS["navit_multiscale_loss_weight"]), 1.0)

    def test_yaml_enables(self):
        from types import SimpleNamespace
        args = SimpleNamespace(**DEFAULTS)
        apply_yaml_config(args, {
            "navit_multiscale": True,
            "navit_multiscale_token_ladder": "4096,16384",
            "navit_multiscale_loss_weight": 0.5,
        })
        self.assertTrue(args.navit_multiscale)
        self.assertEqual(args.navit_multiscale_token_ladder, "4096,16384")
        self.assertEqual(float(args.navit_multiscale_loss_weight), 0.5)


if __name__ == "__main__":
    unittest.main()
