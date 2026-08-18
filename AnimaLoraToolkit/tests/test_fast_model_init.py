"""meta device 构造 transformer（fast_model_init）单测。

固化的不变量：
  1) meta 构造 + to_empty + 显式重算，得到的**所有** buffer（含 non-persistent）与
     正常 CPU 构造逐位一致 —— 也就是 fast_init 只省掉那些随即被 checkpoint 覆盖的
     随机初始化，不改变任何真正被用到的数值。
  2) 显式重算的 key 集合 = RoPE 派生 buffer ∪ non-persistent buffer。
  3) _assert_no_uninitialized 是真防线：漏掉任何一个 key 就 raise，而不是让
     未初始化内存悄悄进训练。
  4) config 键默认关闭、YAML 可开启。

用 2 blocks 的小 config（真实是 28 blocks / 2.09B），结论与规模无关。
"""

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from trainer.config import DEFAULTS, apply_yaml_config  # noqa: E402
from trainer.models import (  # noqa: E402
    _assert_no_uninitialized,
    _build_anima_on_meta,
    ensure_models_namespace,
    load_module_from_path,
)

_MODELS = ROOT / "models"
ensure_models_namespace(_MODELS)
load_module_from_path("cosmos_predict2_modeling", _MODELS / "cosmos_predict2_modeling.py")
_am = load_module_from_path("anima_modeling", _MODELS / "anima_modeling.py")

SMALL_CFG = dict(
    max_img_h=64, max_img_w=64, max_frames=8,
    in_channels=16, out_channels=16, patch_spatial=2, patch_temporal=1,
    concat_padding_mask=True, model_channels=2048, num_blocks=2, num_heads=16,
    crossattn_emb_channels=1024, pos_emb_cls="rope3d", pos_emb_learnable=True,
    pos_emb_interpolation="crop", use_adaln_lora=True, adaln_lora_dim=256,
    rope_h_extrapolation_ratio=4.0, rope_w_extrapolation_ratio=4.0,
    rope_t_extrapolation_ratio=1.0,
)


class MetaInitTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.ref = _am.Anima(**SMALL_CFG)
        cls.model, cls.recomputed = _build_anima_on_meta(_am.Anima, SMALL_CFG)

    def _fill_from_ref(self, model):
        """模拟 checkpoint 加载：灌入除 recomputed 之外的所有 state_dict 条目。"""
        sd_ref, msd = self.ref.state_dict(), model.state_dict()
        for k, v in sd_ref.items():
            if k in self.recomputed:
                continue
            msd[k].data.copy_(v)
        return sorted(k for k in self.recomputed if k in sd_ref)

    def test_recomputed_set_is_exactly_rope_plus_non_persistent(self):
        persistent = set(self.ref.state_dict().keys())
        non_persistent = {n for n, _ in self.ref.named_buffers()} - persistent
        rope = {k for k in persistent
                if "pos_embedder.seq" in k or "pos_embedder.dim_" in k}
        self.assertTrue(non_persistent, "样例模型应当有 non-persistent buffer，否则这条测试没意义")
        self.assertEqual(self.recomputed, rope | non_persistent)

    def test_all_buffers_bit_identical_after_load(self):
        self._fill_from_ref(self.model)
        ref_b = dict(self.ref.named_buffers())
        got_b = dict(self.model.named_buffers())
        self.assertEqual(set(ref_b), set(got_b))
        for k in ref_b:
            with self.subTest(buffer=k):
                self.assertTrue(torch.equal(ref_b[k], got_b[k]), f"{k} 不一致")

    def test_all_params_filled(self):
        self._fill_from_ref(self.model)
        sd_ref, msd = self.ref.state_dict(), self.model.state_dict()
        for k in sd_ref:
            with self.subTest(key=k):
                self.assertTrue(torch.equal(sd_ref[k], msd[k]), f"{k} 不一致")

    def test_assert_passes_on_complete_load(self):
        skipped = self._fill_from_ref(self.model)
        _assert_no_uninitialized(self.model, {"missing": [], "skipped": skipped}, self.recomputed)

    def test_assert_raises_on_missing_key(self):
        skipped = self._fill_from_ref(self.model)
        with self.assertRaises(RuntimeError) as cm:
            _assert_no_uninitialized(
                self.model,
                {"missing": ["blocks.0.mlp.layer1.weight"], "skipped": skipped},
                self.recomputed,
            )
        self.assertIn("blocks.0.mlp.layer1.weight", str(cm.exception))

    def test_assert_raises_when_non_persistent_not_recomputed(self):
        skipped = self._fill_from_ref(self.model)
        persistent = set(self.ref.state_dict().keys())
        non_persistent = {n for n, _ in self.ref.named_buffers()} - persistent
        with self.assertRaises(RuntimeError):
            _assert_no_uninitialized(
                self.model, {"missing": [], "skipped": skipped},
                self.recomputed - non_persistent,
            )


class ConfigTest(unittest.TestCase):

    def test_default_off(self):
        self.assertIn("fast_model_init", DEFAULTS)
        self.assertFalse(DEFAULTS["fast_model_init"])

    def test_yaml_enables(self):
        from types import SimpleNamespace
        args = SimpleNamespace(**DEFAULTS)
        apply_yaml_config(args, {"fast_model_init": True})
        self.assertTrue(args.fast_model_init)


if __name__ == "__main__":
    unittest.main()
