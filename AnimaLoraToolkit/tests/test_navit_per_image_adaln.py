# -*- coding: utf-8 -*-
"""NaViT per-image AdaLN 调制路径（mod_index）≡ legacy 逐 token 展开路径。

背景：navit 打包前向原来把 t_emb/adaln_lora ``repeat_interleave`` 成 [1, ΣN, *] 后
喂给每个 block 的三个调制 MLP —— ΣN 行调制 matmul（唯一值只有 G 个）+ chunk 条带视图
喂逐元素 op。本地交错实测（RTX 5070, 2048ch×28blocks×16384tok，6 轮中位）新路径
（mod_index：调制算在 [1, G, *] 上、输出经 index_select gather 成连续逐 token 张量）
前向 −13%。同一行同值 → 数学等价，差异只可能来自 GEMM 行数不同的浮点归约顺序。

本测试在 fp32 下断言 Block / FinalLayer 两种布局逐位/近逐位一致（含 use_adaln_lora
两种分支），并断言 index_select 输出连续。model 级等价由 test_packed_navit_forward
（forward_packed_navit ≡ 逐图 forward_packed_tokens 拼接）继续覆盖。

Needs CUDA + xformers (block-diagonal varlen attention)；skipped otherwise。
Run with the local GPU python:
    D:\\ArtificialIntelligence\\ComfyUI-aki-v1.5\\python\\python.exe -m pytest \\
        AnimaLoraToolkit/tests/test_navit_per_image_adaln.py -v
"""
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch
    from models.anima_modeling_core import Block, FinalLayer
    HAS_TORCH = True
except ModuleNotFoundError:
    HAS_TORCH = False

HAS_CUDA = HAS_TORCH and torch.cuda.is_available()
try:
    from xformers.ops.fmha import BlockDiagonalMask
    HAS_XFORMERS = True
except Exception:
    HAS_XFORMERS = False


@unittest.skipUnless(HAS_TORCH and HAS_CUDA and HAS_XFORMERS,
                     "needs torch + CUDA + xformers")
class PerImageAdalnEquivalenceTests(unittest.TestCase):
    D = 128          # /num_heads=2 -> head_dim 64 (xformers varlen 支持 64/128)
    CTX = 96
    COUNTS = [4, 12, 6]          # G=3 张异构图的 token 数
    TEXT = [5, 9, 3]

    def _mk_inputs(self, dtype=torch.float32):
        torch.manual_seed(0)
        G = len(self.COUNTS)
        SN = sum(self.COUNTS)
        x = torch.randn(1, SN, self.D, device="cuda", dtype=dtype)
        cross = torch.randn(1, sum(self.TEXT), self.CTX, device="cuda", dtype=dtype)
        emb_G = torch.randn(1, G, self.D, device="cuda", dtype=dtype)
        lora_G = torch.randn(1, G, 3 * self.D, device="cuda", dtype=dtype)
        counts = torch.tensor(self.COUNTS, device="cuda")
        mod_index = torch.repeat_interleave(
            torch.arange(G, device="cuda"), counts
        )
        # legacy 逐 token 布局：同一行 repeat_interleave 展开
        emb_tok = emb_G[0].repeat_interleave(counts, dim=0).unsqueeze(0)
        lora_tok = lora_G[0].repeat_interleave(counts, dim=0).unsqueeze(0)
        self_bias = BlockDiagonalMask.from_seqlens(self.COUNTS)
        cross_bias = BlockDiagonalMask.from_seqlens(
            q_seqlen=self.COUNTS, kv_seqlen=self.TEXT
        )
        return (x, cross, emb_G, lora_G, emb_tok, lora_tok,
                mod_index, self_bias, cross_bias)

    def _block(self, use_adaln_lora):
        torch.manual_seed(1)
        blk = Block(self.D, self.CTX, num_heads=2, mlp_ratio=2.0,
                    use_adaln_lora=use_adaln_lora, adaln_lora_dim=16)
        return blk.to(device="cuda", dtype=torch.float32).eval()

    def _assert_block_equiv(self, use_adaln_lora):
        blk = self._block(use_adaln_lora)
        (x, cross, emb_G, lora_G, emb_tok, lora_tok,
         mod_index, self_bias, cross_bias) = self._mk_inputs()
        lora_G_arg = lora_G if use_adaln_lora else None
        lora_tok_arg = lora_tok if use_adaln_lora else None
        with torch.no_grad():
            legacy = blk.forward_tokens(
                x, emb_tok, cross, rope_emb_L_1_1_D=None,
                attn_mask=self_bias, token_mask_f=None,
                adaln_lora_B_T_3D=lora_tok_arg, cross_attn_mask=cross_bias,
                token_wise_mod=True,
            )
            new = blk.forward_tokens(
                x, emb_G, cross, rope_emb_L_1_1_D=None,
                attn_mask=self_bias, token_mask_f=None,
                adaln_lora_B_T_3D=lora_G_arg, cross_attn_mask=cross_bias,
                token_wise_mod=True, mod_index=mod_index,
            )
        # fp32：同值行经不同 M 的 GEMM，容差留浮点归约余量（实测通常逐 bit 相同）
        torch.testing.assert_close(new, legacy, rtol=1e-5, atol=1e-5)

    def test_block_equiv_adaln_lora(self):
        self._assert_block_equiv(use_adaln_lora=True)

    def test_block_equiv_plain_adaln(self):
        self._assert_block_equiv(use_adaln_lora=False)

    def test_final_layer_equiv(self):
        torch.manual_seed(2)
        fl = FinalLayer(self.D, spatial_patch_size=2, temporal_patch_size=1,
                        out_channels=16, use_adaln_lora=True, adaln_lora_dim=16)
        # init_weights 把调制输出权重置零 → shift/scale 恒 0，测试会空转；重新随机化
        for lin in (fl.adaln_modulation[1], fl.adaln_modulation[2]):
            torch.nn.init.normal_(lin.weight, std=0.05)
        fl = fl.to(device="cuda", dtype=torch.float32).eval()
        (x, _cross, emb_G, lora_G, emb_tok, lora_tok,
         mod_index, _sb, _cb) = self._mk_inputs()
        with torch.no_grad():
            legacy = fl.forward_tokens(
                x, emb_tok, adaln_lora_B_T_3D=lora_tok, token_wise_mod=True,
            )
            new = fl.forward_tokens(
                x, emb_G, adaln_lora_B_T_3D=lora_G, token_wise_mod=True,
                mod_index=mod_index,
            )
        torch.testing.assert_close(new, legacy, rtol=1e-5, atol=1e-5)

    def test_gathered_mod_is_contiguous(self):
        """性能不变量：index_select 出来的逐 token 调制张量必须连续
        （非连续条带视图正是 legacy 布局慢的根因之一）。"""
        emb_G = torch.randn(1, 3, self.D, device="cuda")
        counts = torch.tensor(self.COUNTS, device="cuda")
        mod_index = torch.repeat_interleave(torch.arange(3, device="cuda"), counts)
        chunk = torch.randn(1, 3, 3 * self.D, device="cuda").chunk(3, dim=-1)[0]
        self.assertFalse(chunk.is_contiguous())          # chunk 视图本身非连续
        gathered = chunk.index_select(1, mod_index)
        self.assertTrue(gathered.is_contiguous())        # gather 后连续
        self.assertEqual(tuple(gathered.shape), (1, sum(self.COUNTS), self.D))
        del emb_G


if __name__ == "__main__":
    unittest.main()
