# -*- coding: utf-8 -*-
"""``utils/npu_compat.expand_attn_mask`` 的行为中立性 / 语义等价 / 体积闸门。

背景：昇腾把 SDPA 落到 ``aclnnFlashAttentionScore``，它不接受 query 维为 1 的广播
attn_mask（真机报 ``get unsupported atten_mask shape ... [1,1,1,251]``），而仓库的
key-padding mask 正是这么造的。``expand_attn_mask`` 在 NPU 上把 ``[B,H,1,Skv]`` 展开成
``[B,H,Sq,Skv]``。本文件断言三件事：

1. **行为中立**：未 ``enable()`` 时是恒等映射（``is`` 同一对象），CUDA 路径不受影响；
2. **语义等价**：``LLMAdapter``（两份实现）在展开前后输出**逐 bit 相同**——广播本来
   就等于沿 Sq 复制，这条保证昇腾上的文本条件与 CUDA 上一致；
3. **体积闸门**：DiT 侧的加性 mask 展开是 O(B·Sq·Skv)，超阈值必须 fail-fast 抛错，
   而不是静默物化几百 MB~GB。

本地跑法（不需要 NPU；靠直接置 ``_NPU_ENABLED`` 模拟已启用）：
    D:\\ArtificialIntelligence\\ComfyUI-aki-v1.5\\python\\python.exe -m pytest \\
        AnimaLoraToolkit/tests/test_npu_mask_compat.py -v
"""
import contextlib
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

from utils import npu_compat as nc


@contextlib.contextmanager
def _pretend_npu():
    """把 ``_NPU_ENABLED`` 临时置 True。

    ``enable()`` 需要真实 torch_npu，本地不可用；而 ``expand_attn_mask`` 只读这个开关，
    展开本身是普通张量运算，在 CPU 上就能验。
    """
    old = nc._NPU_ENABLED
    nc._NPU_ENABLED = True
    try:
        yield
    finally:
        nc._NPU_ENABLED = old


@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestExpandAttnMask(unittest.TestCase):
    def test_identity_when_disabled(self):
        """未启用 NPU：原样返回同一个对象（不是拷贝），CUDA 路径逐字节不变。"""
        m = torch.zeros(2, 1, 1, 16, dtype=torch.bool)
        self.assertIs(nc.expand_attn_mask(m, 16), m)
        self.assertIsNone(nc.expand_attn_mask(None, 16))

    def test_expands_broadcast_mask(self):
        with _pretend_npu():
            m = torch.zeros(2, 1, 1, 16, dtype=torch.bool)
            m[0, 0, 0, 5:] = True
            out = nc.expand_attn_mask(m, 9)
            self.assertEqual(tuple(out.shape), (2, 1, 9, 16))
            self.assertTrue(out.is_contiguous())     # expand 出的 stride=0 对融合算子不安全
            self.assertEqual(out.dtype, torch.bool)
            # 每一行都是原 mask 的复制
            self.assertTrue(torch.equal(out, m.expand(2, 1, 9, 16)))

    def test_noop_shapes(self):
        """不该动的形状一律原样返回（同一对象）。"""
        with _pretend_npu():
            already = torch.zeros(2, 1, 9, 16, dtype=torch.bool)   # Sq 已是真实长度
            self.assertIs(nc.expand_attn_mask(already, 9), already)
            two_d = torch.zeros(9, 16, dtype=torch.bool)           # [Sq,Skv]，昇腾直接接受
            self.assertIs(nc.expand_attn_mask(two_d, 9), two_d)
            q1 = torch.zeros(2, 1, 1, 16, dtype=torch.bool)        # Sq=1，本来就合法
            self.assertIs(nc.expand_attn_mask(q1, 1), q1)
            self.assertIs(nc.expand_attn_mask("not-a-tensor", 9), "not-a-tensor")

    def test_size_guard_raises(self):
        """DiT 侧的 bf16 加性 mask 展开是 O(B·Sq·Skv)：超阈值 fail-fast，不静默物化。"""
        with _pretend_npu():
            big = torch.zeros(1, 1, 1, 16384, dtype=torch.bfloat16)  # 展开后 0.54 GB
            with self.assertRaises(RuntimeError) as cm:
                nc.expand_attn_mask(big, 16384)
            self.assertIn("navit_packing", str(cm.exception))        # 错误信息给出替代路径
            small = torch.zeros(1, 1, 1, 512, dtype=torch.bfloat16)  # 0.5 MB，放行
            self.assertEqual(tuple(nc.expand_attn_mask(small, 512).shape), (1, 1, 512, 512))


@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestLLMAdapterEquivalence(unittest.TestCase):
    """展开前后 ``LLMAdapter`` 输出逐 bit 相同（self-attn 与 cross-attn 都带真实 padding）。

    ``anima_modeling`` 与 ``anima_modeling_core`` 各有一份实现，两份都测。
    """

    def _run(self, module_name):
        mod = __import__(module_name, fromlist=["LLMAdapter"])
        torch.manual_seed(0)
        adapter = mod.LLMAdapter(source_dim=32, target_dim=64, model_dim=64,
                                 num_layers=2, num_heads=4).eval()
        B, L_target, L_source = 2, 11, 7
        ids = torch.randint(0, 1000, (B, L_target))
        src = torch.randn(B, L_source, 32)
        target_mask = torch.ones(B, L_target, dtype=torch.bool)
        target_mask[1, 8:] = False          # 第 2 条样本的 target 有 padding
        source_mask = torch.ones(B, L_source, dtype=torch.bool)
        source_mask[0, 5:] = False          # 第 1 条样本的 source 有 padding
        with torch.no_grad():
            ref = adapter(src, ids, target_attention_mask=target_mask,
                          source_attention_mask=source_mask)
            with _pretend_npu():
                got = adapter(src, ids, target_attention_mask=target_mask,
                              source_attention_mask=source_mask)
        self.assertTrue(torch.equal(ref, got),
                        f"{module_name}: 展开后输出变了，max_abs_diff="
                        f"{(ref - got).abs().max().item():.3e}")

    def test_anima_modeling(self):
        self._run("models.anima_modeling")

    def test_anima_modeling_core(self):
        self._run("models.anima_modeling_core")


if __name__ == "__main__":
    unittest.main()
