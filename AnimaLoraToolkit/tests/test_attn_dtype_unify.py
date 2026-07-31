"""回归：LoRA 注入后 cross-attn 的 q/k/v dtype 不一致 → xformers 直接 ValueError。

现场（云端 NaViT 训练第一步就崩）：
    ValueError: Query/Key/Value should either all have the same dtype ...
      query.dtype: torch.float32   key.dtype: torch.bfloat16   value.dtype: torch.bfloat16
链条（本地 torch 2.9 + CUDA 实测，见 _unify_attn_dtype 的 docstring）：
    autocast(bf16) 下 LayerNorm → fp32 → 被 LoKr/DoRA 包住的 q_proj 把输出 cast 回输入
    dtype → q 是 fp32；cross-attn 的 k/v 来自 bf16 的 crossattn_emb → bf16。
dense 路径走 SDPA（autocast 算子，自己会统一）所以一直没暴露；NaViT 块对角路径必走
xformers（非 autocast 算子）→ 崩。

本地跑：
    D:\\ArtificialIntelligence\\ComfyUI-aki-v1.5\\python\\python.exe -m pytest \\
        AnimaLoraToolkit/tests/test_attn_dtype_unify.py -v
"""
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch
    from models import anima_modeling_core as core
    from models.anima_modeling import Anima
    from trainer.lora import LoRAInjector
    HAS_TORCH = True
except ModuleNotFoundError:
    HAS_TORCH = False

HAS_CUDA = HAS_TORCH and torch.cuda.is_available()


@unittest.skipUnless(HAS_TORCH, "needs torch")
class UnifyAttnDtypeTests(unittest.TestCase):
    def test_same_dtype_is_passthrough(self):
        q = torch.randn(1, 4, 2, 8)
        k = torch.randn(1, 4, 2, 8)
        v = torch.randn(1, 4, 2, 8)
        oq, ok, ov = core._unify_attn_dtype(q, k, v)
        # 同 dtype 时必须是同一对象（零拷贝，行为逐字节不变）
        self.assertIs(oq, q)
        self.assertIs(ok, k)
        self.assertIs(ov, v)

    def test_mixed_without_autocast_promotes(self):
        q = torch.randn(1, 4, 2, 8, dtype=torch.float32)
        k = torch.randn(1, 4, 2, 8, dtype=torch.bfloat16)
        v = torch.randn(1, 4, 2, 8, dtype=torch.bfloat16)
        oq, ok, ov = core._unify_attn_dtype(q, k, v)
        # autocast 关（eval/采样）→ 提到最宽，不静默降精度
        self.assertEqual(oq.dtype, torch.float32)
        self.assertEqual(ok.dtype, torch.float32)
        self.assertEqual(ov.dtype, torch.float32)

    @unittest.skipUnless(HAS_CUDA, "needs CUDA for autocast")
    def test_mixed_under_autocast_uses_autocast_dtype(self):
        q = torch.randn(1, 4, 2, 8, device="cuda", dtype=torch.float32)
        k = torch.randn(1, 4, 2, 8, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, 4, 2, 8, device="cuda", dtype=torch.bfloat16)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            oq, ok, ov = core._unify_attn_dtype(q, k, v)
        self.assertEqual(oq.dtype, torch.bfloat16)
        self.assertEqual(ok.dtype, torch.bfloat16)
        self.assertEqual(ov.dtype, torch.bfloat16)


@unittest.skipUnless(HAS_CUDA, "needs torch + CUDA for autocast")
class ForceAutocastDtypeTests(unittest.TestCase):
    """attn_force_autocast_dtype=true：同为 fp32 的 q/k/v 也拉回 autocast dtype。"""

    def tearDown(self):
        core.set_attn_force_autocast_dtype(False)

    def _fp32_triplet(self):
        return tuple(torch.randn(1, 4, 2, 8, device="cuda", dtype=torch.float32)
                     for _ in range(3))

    def test_off_keeps_fp32(self):
        core.set_attn_force_autocast_dtype(False)
        q, k, v = self._fp32_triplet()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            oq, ok, ov = core._unify_attn_dtype(q, k, v)
        self.assertIs(oq, q)                       # 默认关 = 逐 bit 不变
        self.assertEqual((oq.dtype, ok.dtype, ov.dtype),
                         (torch.float32,) * 3)

    def test_on_downcasts_under_autocast(self):
        core.set_attn_force_autocast_dtype(True)
        q, k, v = self._fp32_triplet()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            oq, ok, ov = core._unify_attn_dtype(q, k, v)
        self.assertEqual((oq.dtype, ok.dtype, ov.dtype),
                         (torch.bfloat16,) * 3)

    def test_on_is_noop_without_autocast(self):
        core.set_attn_force_autocast_dtype(True)
        q, k, v = self._fp32_triplet()
        oq, ok, ov = core._unify_attn_dtype(q, k, v)   # eval/采样：autocast 关
        self.assertIs(oq, q)
        self.assertEqual(oq.dtype, torch.float32)


class _StrictXops:
    """复刻 xformers ``validate_inputs`` 的 dtype 校验（本地无 xformers 也能测）。"""

    def __init__(self):
        self.calls = []

    def memory_efficient_attention(self, q, k, v, attn_bias=None):
        if not (q.dtype == k.dtype == v.dtype):
            raise ValueError(
                "Query/Key/Value should either all have the same dtype, or ... "
                f"query.dtype: {q.dtype} key.dtype: {k.dtype} value.dtype: {v.dtype}"
            )
        self.calls.append((q.dtype, k.dtype, v.dtype))
        return torch.zeros_like(q)


@unittest.skipUnless(HAS_CUDA, "needs torch + CUDA")
class LoraInjectedCrossAttnDtypeTests(unittest.TestCase):
    """端到端：LoKr+DoRA 注入后走块对角（xformers）分支不能再因 dtype 崩。"""

    def tearDown(self):
        core.set_attn_force_autocast_dtype(False)

    def _block_forward(self, inject: bool):
        torch.manual_seed(0)
        dtype = torch.bfloat16
        m = Anima(
            max_img_h=64, max_img_w=64, max_frames=1,
            in_channels=16, out_channels=16, patch_spatial=2, patch_temporal=1,
            concat_padding_mask=False, model_channels=128, num_blocks=1, num_heads=2,
            mlp_ratio=2.0, crossattn_emb_channels=128, pos_emb_cls="rope3d",
        ).to(device="cuda", dtype=dtype)
        if inject:
            LoRAInjector(
                rank=8, alpha=8.0, use_lokr=True, factor=4, lora_variant="dora",
                targets=["q_proj", "k_proj", "v_proj", "output_proj",
                         "mlp.layer1", "mlp.layer2"],
            ).inject(m)
            m.to(device="cuda")
        m.train()

        stub = _StrictXops()
        sentinel = object()          # 冒充 BlockDiagonalMask
        old_xops, old_is_bias = core.xops, core._is_xformers_attn_bias
        core.xops = stub
        core._is_xformers_attn_bias = lambda b: b is sentinel
        try:
            x = torch.randn(1, 12, 128, device="cuda", dtype=dtype)
            t_emb = torch.randn(1, 1, 128, device="cuda", dtype=dtype)
            cross = torch.randn(1, 7, 128, device="cuda", dtype=dtype)
            with torch.autocast("cuda", dtype=dtype):
                m.blocks[0].forward_tokens(
                    x, t_emb, cross, rope_emb_L_1_1_D=None,
                    attn_mask=sentinel, token_mask_f=None,
                    adaln_lora_B_T_3D=None, cross_attn_mask=sentinel,
                    token_wise_mod=False,
                )
        finally:
            core.xops, core._is_xformers_attn_bias = old_xops, old_is_bias
        return stub.calls

    def test_lora_injected_block_runs(self):
        calls = self._block_forward(inject=True)
        self.assertEqual(len(calls), 2)               # self_attn + cross_attn
        for qd, kd, vd in calls:
            self.assertEqual(qd, kd)
            self.assertEqual(kd, vd)

    def test_force_autocast_dtype_makes_self_attn_bf16(self):
        # 默认关：self_attn 三者同为 fp32（xformers 不报错，但跑的是 fp32 kernel）
        calls = self._block_forward(inject=True)
        self.assertEqual(calls[0], (torch.float32,) * 3)
        # 开：self_attn 也拉回 bf16，与 dense/eval 的 SDPA 口径一致
        core.set_attn_force_autocast_dtype(True)
        calls = self._block_forward(inject=True)
        for qkv in calls:
            self.assertEqual(qkv, (torch.bfloat16,) * 3)

    def test_base_model_unchanged(self):
        # 未注入 LoRA 时本来就全 bf16，修复对它必须是 no-op
        calls = self._block_forward(inject=False)
        self.assertEqual(len(calls), 2)
        for qd, kd, vd in calls:
            self.assertEqual((qd, kd, vd),
                             (torch.bfloat16, torch.bfloat16, torch.bfloat16))


if __name__ == "__main__":
    unittest.main()
