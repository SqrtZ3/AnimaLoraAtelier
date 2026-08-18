"""VAE 自注意力 query 分块（vae_attn_chunk_tokens）单测。

固化的不变量：
  1) 数学恒等：_chunked_sdpa(q,k,v,chunk) ≡ F.scaled_dot_product_attention(q,k,v)
     （按 query 切块，每块仍对全部 K/V 做注意力，softmax 归一化域不变 → 非近似）。
  2) 整块 AttentionBlock 前向在开/关分块下一致（含 N 不被 chunk 整除的情况）。
  3) chunk=0 / N≤chunk 时是零开销回退，走同一条整块 SDPA 路径。
  4) 峰值中间量的规模：分块后不再物化 N×N（用 chunk 数量间接断言切分正确）。
  5) 环境变量 ANIMA_VAE_ATTN_CHUNK 覆盖；非法值 fail-fast。
  6) config 键默认关闭、YAML 可开启。
"""

import os
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from trainer.config import DEFAULTS, apply_yaml_config  # noqa: E402
from trainer.models import load_module_from_path  # noqa: E402

wan_vae = load_module_from_path("wan_vae_test", ROOT / "models" / "wan" / "vae2_1.py")


class ChunkedSdpaIdentityTest(unittest.TestCase):
    """分块 SDPA 与整块 SDPA 的逐元素对拍。"""

    def setUp(self):
        torch.manual_seed(0)
        wan_vae.set_vae_attn_chunk_tokens(0)
        os.environ.pop("ANIMA_VAE_ATTN_CHUNK", None)

    def test_identical_to_dense_various_chunks(self):
        # head_dim=384 与仓库配置最深层通道数一致（dim=96, dim_mult 末位 4）
        b, h, n, d = 2, 1, 130, 384
        q, k, v = (torch.randn(b, h, n, d, dtype=torch.float64) for _ in range(3))
        ref = F.scaled_dot_product_attention(q, k, v)
        for chunk in (1, 7, 64, 129, 130, 131, 4096):
            with self.subTest(chunk=chunk):
                got = wan_vae._chunked_sdpa(q, k, v, chunk)
                self.assertEqual(got.shape, ref.shape)
                torch.testing.assert_close(got, ref, rtol=0, atol=1e-12)

    def test_chunk_zero_and_oversize_are_passthrough(self):
        q, k, v = (torch.randn(1, 1, 33, 16, dtype=torch.float64) for _ in range(3))
        ref = F.scaled_dot_product_attention(q, k, v)
        for chunk in (0, -1, 33, 100):
            with self.subTest(chunk=chunk):
                torch.testing.assert_close(
                    wan_vae._chunked_sdpa(q, k, v, chunk), ref, rtol=0, atol=0)

    def test_splits_queries_not_keys(self):
        """分块必须切 query 维：每块的 K/V 是全量，否则就成了近似。"""
        calls = []
        real = F.scaled_dot_product_attention

        def spy(q, k, v, *a, **kw):
            calls.append((q.shape[-2], k.shape[-2]))
            return real(q, k, v, *a, **kw)

        q, k, v = (torch.randn(1, 1, 100, 8, dtype=torch.float64) for _ in range(3))
        F.scaled_dot_product_attention = spy
        try:
            wan_vae._chunked_sdpa(q, k, v, 32)
        finally:
            F.scaled_dot_product_attention = real
        self.assertEqual([c[0] for c in calls], [32, 32, 32, 4])   # query 被切
        self.assertTrue(all(c[1] == 100 for c in calls))           # key 始终全量


class AttentionBlockTest(unittest.TestCase):
    """AttentionBlock 端到端：开关分块结果一致。"""

    def setUp(self):
        torch.manual_seed(1)
        wan_vae.set_vae_attn_chunk_tokens(0)
        os.environ.pop("ANIMA_VAE_ATTN_CHUNK", None)

    def tearDown(self):
        wan_vae.set_vae_attn_chunk_tokens(0)
        os.environ.pop("ANIMA_VAE_ATTN_CHUNK", None)

    def _block_and_input(self):
        blk = wan_vae.AttentionBlock(32).double().eval()
        # proj 权重初始化为 0（残差恒等），置随机值才能真正测到注意力输出
        torch.nn.init.normal_(blk.proj.weight, std=0.05)
        x = torch.randn(1, 32, 1, 13, 11, dtype=torch.float64)  # N=143，不被 chunk 整除
        return blk, x

    def test_forward_matches_with_and_without_chunk(self):
        blk, x = self._block_and_input()
        with torch.no_grad():
            ref = blk(x)
            for chunk in (16, 64, 143, 512):
                with self.subTest(chunk=chunk):
                    wan_vae.set_vae_attn_chunk_tokens(chunk)
                    torch.testing.assert_close(blk(x), ref, rtol=0, atol=1e-12)

    def test_env_override(self):
        blk, x = self._block_and_input()
        with torch.no_grad():
            ref = blk(x)
            os.environ["ANIMA_VAE_ATTN_CHUNK"] = "16"
            self.assertEqual(wan_vae.get_vae_attn_chunk_tokens(), 16)
            torch.testing.assert_close(blk(x), ref, rtol=0, atol=1e-12)

    def test_env_garbage_is_ignored(self):
        os.environ["ANIMA_VAE_ATTN_CHUNK"] = "abc"
        wan_vae.set_vae_attn_chunk_tokens(4096)
        self.assertEqual(wan_vae.get_vae_attn_chunk_tokens(), 4096)

    def test_negative_setter_fails_fast(self):
        with self.assertRaises(ValueError):
            wan_vae.set_vae_attn_chunk_tokens(-1)


class ConfigTest(unittest.TestCase):

    def test_default_off(self):
        self.assertIn("vae_attn_chunk_tokens", DEFAULTS)
        self.assertEqual(DEFAULTS["vae_attn_chunk_tokens"], 0)

    def test_yaml_enables(self):
        import argparse
        args = argparse.Namespace(**dict(DEFAULTS))
        apply_yaml_config(args, {"vae_attn_chunk_tokens": 4096})
        self.assertEqual(args.vae_attn_chunk_tokens, 4096)


if __name__ == "__main__":
    unittest.main()
