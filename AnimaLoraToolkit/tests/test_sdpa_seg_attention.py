# -*- coding: utf-8 -*-
"""navit_attn_backend=sdpa_seg（逐段 dense SDPA）与块对角 mask 的数学恒等对拍。

背景：H20 微基准（tests/diag_h20_fp8_bench.py，2026-07-17）实测 dense SDPA
（cudnn）比 xformers FA2 varlen 快 1.56×，且云端 xformers 的 5D grouped 路径
没有 backward 算子。sdpa_seg 把块对角 mask 换成逐段 dense SDPA——段内全注意力
≡ 块对角语义，本文件用 fp32 对拍证明恒等（前向 + 反向梯度）。

本地跑法（不需要 xformers / GPU 有无皆可）：
    D:\\ArtificialIntelligence\\ComfyUI-aki-v1.5\\python\\python.exe -m pytest \\
        AnimaLoraToolkit/tests/test_sdpa_seg_attention.py -v
"""
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch
    HAS_TORCH = True
except ModuleNotFoundError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestSdpaSegAttention(unittest.TestCase):
    SEGS = [5, 9, 3]

    def _qkv(self, Hq, Hkv, D, dtype=torch.float32, seed=0):
        torch.manual_seed(seed)
        L = sum(self.SEGS)
        q = torch.randn(1, Hq, L, D, dtype=dtype, requires_grad=True)
        k = torch.randn(1, Hkv, L, D, dtype=dtype, requires_grad=True)
        v = torch.randn(1, Hkv, L, D, dtype=dtype, requires_grad=True)
        return q, k, v

    def _run_pair(self, Hq, Hkv, gqa):
        """同一份 q/k/v 分别走 _SegLens 路径与 bool 块对角 mask 路径，返回两组 (y, dq, dk, dv)。"""
        from models.krea2_modeling import attention, _SegLens, block_diag_bool_mask

        outs = []
        for use_seg in (True, False):
            q, k, v = self._qkv(Hq, Hkv, 16)
            if use_seg:
                mask = _SegLens(self.SEGS)
            else:
                mask = block_diag_bool_mask(self.SEGS, q.device)
            y = attention(q, k, v, mask=mask, gqa=gqa)
            y.square().sum().backward()
            outs.append((y.detach(), q.grad.detach(), k.grad.detach(), v.grad.detach()))
        return outs

    def _assert_close(self, a, b, what):
        self.assertTrue(
            torch.allclose(a, b, atol=1e-5, rtol=1e-5),
            f"{what} 不一致: max_abs={(a - b).abs().max().item():.3e}")

    def test_mha_forward_backward_equivalent(self):
        (y1, dq1, dk1, dv1), (y2, dq2, dk2, dv2) = self._run_pair(8, 8, gqa=False)
        self._assert_close(y1, y2, "MHA 前向")
        self._assert_close(dq1, dq2, "MHA dq")
        self._assert_close(dk1, dk2, "MHA dk")
        self._assert_close(dv1, dv2, "MHA dv")

    def test_gqa_forward_backward_equivalent(self):
        # krea2 同款头比：48/12 = 4:1（缩小为 8/2 保持 rep=4）
        (y1, dq1, dk1, dv1), (y2, dq2, dk2, dv2) = self._run_pair(8, 2, gqa=True)
        self._assert_close(y1, y2, "GQA 前向")
        self._assert_close(dq1, dq2, "GQA dq")
        self._assert_close(dk1, dk2, "GQA dk")
        self._assert_close(dv1, dv2, "GQA dv")

    def test_packed_navit_forward_equivalent_small_model(self):
        """全模型 packed 前向：sdpa_seg vs 默认（本地无 xformers = bool mask）。"""
        from models.krea2_modeling import (
            SingleStreamDiT, SingleMMDiTConfig, set_packed_attention_backend)
        cfg = SingleMMDiTConfig(features=128, tdim=64, txtdim=64, heads=8,
                                kvheads=2, multiplier=2, layers=2, patch=2,
                                channels=4, txtheads=4, txtkvheads=4, txtlayers=2)
        torch.manual_seed(0)
        m = SingleStreamDiT(cfg).float().eval()
        vis, txts = [16, 36], [7, 5]
        SN, SL, G = sum(vis), sum(txts), len(vis)
        tokens = torch.randn(1, SN, cfg.patch ** 2 * cfg.channels)
        t_G = torch.rand(G)
        cross = torch.randn(1, SL, cfg.txtlayers, cfg.txtdim)
        grid = torch.zeros(1, 2, SN)
        off = 0
        for s in vis:
            side = int(s ** 0.5)
            idx = torch.arange(s)
            grid[0, 0, off:off + s] = (idx // side).float()
            grid[0, 1, off:off + s] = (idx % side).float()
            off += s

        try:
            with torch.no_grad():
                set_packed_attention_backend("xformers")
                y_ref = m.forward_packed_navit(tokens, t_G, cross, grid, vis, txts)
                set_packed_attention_backend("sdpa_seg")
                y_seg = m.forward_packed_navit(tokens, t_G, cross, grid, vis, txts)
        finally:
            set_packed_attention_backend("xformers")
        self.assertTrue(
            torch.allclose(y_ref, y_seg, atol=1e-5, rtol=1e-5),
            f"packed 前向不一致: max_abs={(y_ref - y_seg).abs().max().item():.3e}")

    def test_invalid_backend_fails_fast(self):
        from models.krea2_modeling import set_packed_attention_backend
        with self.assertRaises(ValueError):
            set_packed_attention_backend("flash3")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
