"""Anima 打包注意力后端对拍：sdpa_seg / npu_tnd ≡ 块对角 mask。

判据：三个后端算的必须是**同一个东西**——打包序列里每张图只看自己的 token。参考实现
用稠密 bool 块对角 mask 走 SDPA（语义定义），被测实现是 ``_SegLens`` 路径。

覆盖：
  * self-attn（q/kv 段长相同）
  * cross-attn（q/kv 段长不等 —— Anima 的 visual↔text，krea2 的 sdpa_seg 不涉及这条）
  * 段间零泄漏（换掉别的段的 k/v，本段输出逐 bit 不变）
  * 反向可用且梯度与参考一致
  * 后端切换的行为中立（默认 xformers 不变）

npu_tnd 需要真机（torch_npu），本地自动 skip —— 它的真机验证在 tools/npu_probe.py。
"""

import math
import os
import sys
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.anima_modeling_core import (  # noqa: E402
    _SegLens,
    get_packed_attention_backend,
    set_packed_attention_backend,
    torch_attention_op,
)


# 模块导入即快照默认后端：任何用例改过它之后就问不到"出厂值"了。
_DEFAULT_BACKEND_AT_IMPORT = get_packed_attention_backend()


def _block_diag_reference(q_B_S_H_D, k_B_S_H_D, v_B_S_H_D, q_lens, kv_lens):
    """语义定义：稠密 bool 块对角 mask + SDPA。返回 [B, S, H*D]。"""
    Tq = sum(q_lens)
    Tkv = sum(kv_lens)
    mask = torch.zeros(Tq, Tkv, dtype=torch.bool, device=q_B_S_H_D.device)
    qo = ko = 0
    for sq, sk in zip(q_lens, kv_lens):
        mask[qo:qo + sq, ko:ko + sk] = True
        qo += sq
        ko += sk
    q = q_B_S_H_D.transpose(1, 2)
    k = k_B_S_H_D.transpose(1, 2)
    v = v_B_S_H_D.transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[None, None])
    B, H, S, D = out.shape
    return out.transpose(1, 2).reshape(B, S, H * D)


def _qkv(q_lens, kv_lens, H=4, D=16, dtype=torch.float32, seed=0, grad=False):
    torch.manual_seed(seed)
    q = torch.randn(1, sum(q_lens), H, D, dtype=dtype, requires_grad=grad)
    k = torch.randn(1, sum(kv_lens), H, D, dtype=dtype, requires_grad=grad)
    v = torch.randn(1, sum(kv_lens), H, D, dtype=dtype, requires_grad=grad)
    return q, k, v


class TestSdpaSegBackend(unittest.TestCase):
    """sdpa_seg：不依赖任何专有算子，本地就能完整验证。"""

    def setUp(self):
        self._saved = get_packed_attention_backend()
        set_packed_attention_backend("sdpa_seg")

    def tearDown(self):
        set_packed_attention_backend(self._saved)

    def test_self_attn_matches_block_diag(self):
        lens = [7, 13, 5]
        q, k, v = _qkv(lens, lens)
        got = torch_attention_op(q, k, v, attn_mask=_SegLens(lens))
        ref = _block_diag_reference(q, k, v, lens, lens)
        self.assertEqual(got.shape, ref.shape)
        self.assertLess((got - ref).abs().max().item(), 1e-5)

    def test_cross_attn_unequal_seglens(self):
        """Anima 的 cross-attn：visual 段长 ≠ text 段长。krea2 的 sdpa_seg 没这条。"""
        q_lens = [7, 13, 5]
        kv_lens = [3, 9, 2]
        q, k, v = _qkv(q_lens, kv_lens)
        got = torch_attention_op(q, k, v, attn_mask=_SegLens(q_lens, kv_lens))
        ref = _block_diag_reference(q, k, v, q_lens, kv_lens)
        self.assertLess((got - ref).abs().max().item(), 1e-5)

    def test_no_cross_segment_leakage(self):
        """换掉第 2/3 段的 k/v，第 1 段输出必须逐 bit 不变。"""
        q_lens = [7, 13, 5]
        kv_lens = [3, 9, 2]
        q, k, v = _qkv(q_lens, kv_lens)
        seg = _SegLens(q_lens, kv_lens)
        out1 = torch_attention_op(q, k, v, attn_mask=seg)

        k2, v2 = k.clone(), v.clone()
        k2[:, kv_lens[0]:] = torch.randn_like(k2[:, kv_lens[0]:])
        v2[:, kv_lens[0]:] = torch.randn_like(v2[:, kv_lens[0]:])
        out2 = torch_attention_op(q, k2, v2, attn_mask=seg)

        head = q_lens[0]
        self.assertTrue(torch.equal(out1[:, :head], out2[:, :head]),
                        "第 1 段输出受到了其他段 k/v 的影响 —— 存在跨图泄漏")
        self.assertFalse(torch.equal(out1[:, head:], out2[:, head:]),
                         "后续段应当变化，否则说明测试本身没生效")

    def test_backward_matches_reference(self):
        q_lens = [6, 10]
        kv_lens = [4, 7]
        q, k, v = _qkv(q_lens, kv_lens, grad=True)
        qr, kr, vr = (t.detach().clone().requires_grad_(True) for t in (q, k, v))

        torch_attention_op(q, k, v, attn_mask=_SegLens(q_lens, kv_lens)).pow(2).sum().backward()
        _block_diag_reference(qr, kr, vr, q_lens, kv_lens).pow(2).sum().backward()

        for name, a, b in (("q", q, qr), ("k", k, kr), ("v", v, vr)):
            self.assertIsNotNone(a.grad, f"{name} 没有梯度")
            self.assertLess((a.grad - b.grad).abs().max().item(), 1e-5,
                            f"{name} 的梯度与块对角参考不一致")

    def test_single_segment_equals_plain_attention(self):
        """只有一段时，块对角退化成全注意力。"""
        lens = [11]
        q, k, v = _qkv(lens, lens)
        got = torch_attention_op(q, k, v, attn_mask=_SegLens(lens))
        plain = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        plain = plain.transpose(1, 2).reshape(1, sum(lens), -1)
        self.assertLess((got - plain).abs().max().item(), 1e-5)


class TestBackendSwitching(unittest.TestCase):
    def test_default_is_xformers(self):
        """默认必须是 xformers —— 否则 CUDA 上的历史行为就被改掉了。

        用**导入本测试模块时**抓到的初值判断，不 reload 模块：reload 会造出一个新的
        ``_SegLens`` 类对象，本模块持有的旧引用随即 isinstance 失配，把后面的用例全带崩。
        """
        self.assertEqual(_DEFAULT_BACKEND_AT_IMPORT, "xformers")

    def test_illegal_backend_fails_fast(self):
        saved = get_packed_attention_backend()
        try:
            with self.assertRaises(ValueError):
                set_packed_attention_backend("flash_whatever")
        finally:
            set_packed_attention_backend(saved)

    def test_seg_lens_validates_shape(self):
        with self.assertRaises(ValueError):
            _SegLens([1, 2, 3], [1, 2])

    def test_seg_lens_defaults_kv_to_q(self):
        s = _SegLens([4, 5])
        self.assertEqual(s.kv_seqlens, (4, 5))


class TestPackedNavitEndToEnd(unittest.TestCase):
    """整条 ``forward_packed_navit`` 在 sdpa_seg 下的端到端隔离性验证（CPU 可跑）。

    判据：把 G 张图打包成一个序列跑一遍，每张图的输出必须与**它自己单独打包**（G=1）
    跑出来的一致。这一条同时覆盖块对角 self/cross attention、per-image AdaLN
    （mod_index）、每图独立 RoPE 网格——比只对拍 attention 算子强得多。

    注意：这个用例在本次改动之前**跑不了**——那时 ``forward_packed_navit`` 硬依赖
    xformers，而本地没有。sdpa_seg 后端把 NaViT 路径的验证从"必须上云"变成了本地免费。
    """

    @classmethod
    def setUpClass(cls):
        try:
            from models.anima_modeling import Anima  # noqa: F401
        except Exception as exc:  # pragma: no cover
            raise unittest.SkipTest(f"导不进 Anima: {exc}")

    def setUp(self):
        self._saved = get_packed_attention_backend()
        set_packed_attention_backend("sdpa_seg")

    def tearDown(self):
        set_packed_attention_backend(self._saved)

    def _model(self):
        from models.anima_modeling import Anima

        torch.manual_seed(0)
        return Anima(
            max_img_h=64, max_img_w=64, max_frames=1,
            in_channels=16, out_channels=16,
            patch_spatial=2, patch_temporal=1,
            concat_padding_mask=False,
            model_channels=64, num_blocks=2, num_heads=2, mlp_ratio=2.0,
            crossattn_emb_channels=64, pos_emb_cls="rope3d",
        ).to(torch.float32).eval()

    @staticmethod
    def _grid(h, w):
        rows = torch.arange(h).repeat_interleave(w)
        cols = torch.arange(w).repeat(h)
        return torch.stack([rows, cols], dim=0)          # [2, h*w]

    def test_packed_image_matches_solo_pack(self):
        model = self._model()
        D = 64
        shapes = [(4, 6), (6, 4), (2, 8)]                # patch 网格（已除 patch_spatial）
        text_lens = [5, 3, 7]
        t_vals = [0.2, 0.55, 0.9]

        toks, grids = [], []
        torch.manual_seed(1)
        for (h, w) in shapes:
            toks.append(torch.randn(h * w, 16 * 2 * 2 * 1))
            grids.append(self._grid(h, w))
        crosses = [torch.randn(L, D) for L in text_lens]
        vis_lens = [t.shape[0] for t in toks]

        with torch.no_grad():
            packed = model.forward_packed_navit(
                torch.cat(toks, 0).unsqueeze(0),
                torch.tensor(t_vals),
                torch.cat(crosses, 0).unsqueeze(0),
                torch.cat(grids, 1).unsqueeze(0),
                vis_lens, text_lens,
            )

        off = 0
        worst = 0.0
        for i, (n, L) in enumerate(zip(vis_lens, text_lens)):
            with torch.no_grad():
                solo = model.forward_packed_navit(
                    toks[i].unsqueeze(0),
                    torch.tensor([t_vals[i]]),
                    crosses[i].unsqueeze(0),
                    grids[i].unsqueeze(0),
                    [n], [L],
                )
            got = packed[:, off:off + n]
            worst = max(worst, (got - solo).abs().max().item())
            off += n
        print(f"\n[navit sdpa_seg] 打包 vs 单图 最大绝对差 = {worst:.3e}")
        self.assertLess(worst, 1e-4,
                        "打包后某张图的输出被别的图影响了 —— 块对角隔离性被破坏")

    def test_backward_flows_through_packed_path(self):
        """反向能穿过整条打包前向（sdpa_seg 不能只在 no_grad 下可用）。"""
        model = self._model()
        torch.manual_seed(2)
        shapes = [(4, 4), (2, 6)]
        text_lens = [4, 6]
        toks = [torch.randn(h * w, 16 * 2 * 2 * 1) for h, w in shapes]
        grids = [self._grid(h, w) for h, w in shapes]
        crosses = [torch.randn(L, 64) for L in text_lens]
        x = torch.cat(toks, 0).unsqueeze(0).requires_grad_(True)

        out = model.forward_packed_navit(
            x, torch.tensor([0.3, 0.7]),
            torch.cat(crosses, 0).unsqueeze(0),
            torch.cat(grids, 1).unsqueeze(0),
            [t.shape[0] for t in toks], text_lens,
        )
        out.pow(2).sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all(), "梯度里有 NaN/Inf")
        self.assertGreater(x.grad.abs().max().item(), 0.0)


class TestNpuTndBackend(unittest.TestCase):
    """npu_tnd 只能在真机验证；本地无 torch_npu 时 skip（不是通过，是没测）。"""

    @classmethod
    def setUpClass(cls):
        try:
            import torch_npu  # noqa: F401
        except Exception:
            raise unittest.SkipTest(
                "无 torch_npu —— npu_tnd 的验证在昇腾真机上由 tools/npu_probe.py 完成")

    def setUp(self):
        self._saved = get_packed_attention_backend()
        set_packed_attention_backend("npu_tnd")

    def tearDown(self):
        set_packed_attention_backend(self._saved)

    def test_cross_attn_matches_block_diag(self):
        q_lens = [64, 128]
        kv_lens = [16, 40]
        q, k, v = _qkv(q_lens, kv_lens, H=16, D=128, dtype=torch.bfloat16)
        q, k, v = q.npu(), k.npu(), v.npu()
        got = torch_attention_op(q, k, v, attn_mask=_SegLens(q_lens, kv_lens))
        ref = _block_diag_reference(q.float(), k.float(), v.float(), q_lens, kv_lens)
        rel = ((got.float() - ref).norm() / ref.norm()).item()
        self.assertLess(rel, 2e-2, f"npu_tnd 与块对角参考相对误差 {rel:.3e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
