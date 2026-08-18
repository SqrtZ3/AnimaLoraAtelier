"""sdpa_seg 段内 query 分块（navit_attn_chunk_tokens）单测。

固化的不变量：
  1) 前向数学恒等：分块 ≡ 整段 SDPA（按 query 切块，每块仍对本段全部 K/V 做注意力）。
  2) **梯度也恒等** —— 分块路径每块套 gradient checkpoint，backward 走的是重算的图，
     所以梯度必须单独对拍，不能只对前向。
  3) 块对角语义不变：_packed_attention_seg 在开分块后仍与稠密 bool mask 的参考实现一致
     （段间零泄漏）。
  4) 分块只切 query 维，K/V 每块全量（切错成 K 维就变成块状近似）。
  5) chunk=0 / s≤chunk 零开销回退；无梯度时不套 checkpoint。
  6) config 默认关；开了但后端不是 sdpa_seg 时 fail-fast。
"""

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from trainer.config import DEFAULTS, apply_yaml_config  # noqa: E402
from trainer.models import ensure_models_namespace  # noqa: E402

ensure_models_namespace(ROOT / "models")
from models import anima_modeling_core as amc  # noqa: E402


def _dense_block_diag_reference(q, k, v, seqlens):
    """参考实现：稠密 bool mask 的块对角注意力。q/k/v 为 [B,S,H,D]。"""
    b, s, h, d = q.shape
    mask = torch.zeros(s, s, dtype=torch.bool, device=q.device)
    off = 0
    for n in seqlens:
        mask[off:off + n, off:off + n] = True
        off += n
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask)
    return out.transpose(1, 2)


class SegChunkTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        amc.set_seg_attn_chunk_tokens(0)

    def tearDown(self):
        amc.set_seg_attn_chunk_tokens(0)

    def _qkv(self, s, h=4, d=16, requires_grad=False):
        return [torch.randn(1, s, h, d, dtype=torch.float64, requires_grad=requires_grad)
                for _ in range(3)]

    def test_forward_identity_various_chunks(self):
        seqlens = (37, 64, 11)
        q, k, v = self._qkv(sum(seqlens))
        seg = amc._SegLens(seqlens)
        ref = amc._packed_attention_seg(q, k, v, seg)
        for chunk in (1, 8, 16, 37, 64, 4096):
            with self.subTest(chunk=chunk):
                amc.set_seg_attn_chunk_tokens(chunk)
                got = amc._packed_attention_seg(q, k, v, seg)
                torch.testing.assert_close(got, ref, rtol=0, atol=1e-12)

    def test_gradients_identity(self):
        """分块路径 backward 走 checkpoint 重算的图，梯度必须单独对拍。"""
        seqlens = (37, 64, 11)
        base = self._qkv(sum(seqlens))
        seg = amc._SegLens(seqlens)

        def grads(chunk):
            amc.set_seg_attn_chunk_tokens(chunk)
            q, k, v = (t.detach().clone().requires_grad_(True) for t in base)
            out = amc._packed_attention_seg(q, k, v, seg)
            out.square().sum().backward()
            return q.grad, k.grad, v.grad

        ref = grads(0)
        for chunk in (8, 16, 37):
            with self.subTest(chunk=chunk):
                for g, r in zip(grads(chunk), ref):
                    torch.testing.assert_close(g, r, rtol=0, atol=1e-10)

    def test_matches_dense_block_diag_mask_with_chunking(self):
        """开了分块之后，块对角语义（段间零泄漏）仍然成立。"""
        seqlens = (23, 40)
        q, k, v = self._qkv(sum(seqlens))
        ref = _dense_block_diag_reference(q, k, v, seqlens)
        amc.set_seg_attn_chunk_tokens(8)
        got = amc._packed_attention_seg(q, k, v, amc._SegLens(seqlens))
        torch.testing.assert_close(got, ref, rtol=0, atol=1e-12)

    def test_splits_queries_not_keys(self):
        calls = []
        real = F.scaled_dot_product_attention

        def spy(q, k, v, *a, **kw):
            calls.append((q.shape[-2], k.shape[-2]))
            return real(q, k, v, *a, **kw)

        qs, ks, vs = (torch.randn(1, 4, 100, 16, dtype=torch.float64) for _ in range(3))
        F.scaled_dot_product_attention = spy
        try:
            amc._seg_sdpa_chunked(qs, ks, vs, 32)
        finally:
            F.scaled_dot_product_attention = real
        self.assertEqual([c[0] for c in calls], [32, 32, 32, 4])
        self.assertTrue(all(c[1] == 100 for c in calls))

    def test_no_checkpoint_without_grad(self):
        """eval / 采样路径没有 backward 图，不该套 checkpoint（纯开销）。"""
        used = []
        real_ckpt = amc.checkpoint
        amc.checkpoint = lambda fn, *a, **kw: (used.append(1), real_ckpt(fn, *a, **kw))[1]
        try:
            qs, ks, vs = (torch.randn(1, 4, 64, 16, dtype=torch.float64) for _ in range(3))
            amc._seg_sdpa_chunked(qs, ks, vs, 16)          # 无 requires_grad
            self.assertEqual(used, [])
            qs.requires_grad_(True)
            amc._seg_sdpa_chunked(qs, ks, vs, 16)
            self.assertEqual(len(used), 4)
        finally:
            amc.checkpoint = real_ckpt

    def test_passthrough(self):
        qs, ks, vs = (torch.randn(1, 4, 33, 16, dtype=torch.float64) for _ in range(3))
        ref = F.scaled_dot_product_attention(qs, ks, vs)
        for chunk in (0, -1, 33, 100):
            with self.subTest(chunk=chunk):
                torch.testing.assert_close(
                    amc._seg_sdpa_chunked(qs, ks, vs, chunk), ref, rtol=0, atol=0)

    def test_negative_fails_fast(self):
        with self.assertRaises(ValueError):
            amc.set_seg_attn_chunk_tokens(-1)


class ConfigTest(unittest.TestCase):

    def test_default_off(self):
        self.assertIn("navit_attn_chunk_tokens", DEFAULTS)
        self.assertEqual(DEFAULTS["navit_attn_chunk_tokens"], 0)

    def test_yaml_enables(self):
        from types import SimpleNamespace
        args = SimpleNamespace(**DEFAULTS)
        apply_yaml_config(args, {"navit_attn_chunk_tokens": 2048})
        self.assertEqual(args.navit_attn_chunk_tokens, 2048)


if __name__ == "__main__":
    unittest.main()
