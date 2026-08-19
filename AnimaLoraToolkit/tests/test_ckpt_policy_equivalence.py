# -*- coding: utf-8 -*-
"""选择性激活重算（grad_checkpoint_policy）：各档与全量 checkpoint 数学恒等。

SAC 只改变「哪些中间量被保存、哪些在 backward 重算」，不改任何数学。本测试在真的
SingleStreamBlock 上断言四档策略（full / sac_attn / sac_narrow / sac_all）的**输出与
输入梯度**一致，并断言 policy=full 与不经 _ckpt 包装的裸 checkpoint 逐位相同
（行为中立性）。

依据（tests/diag_navit_ckpt_policy.py --g-sweep，H20 28 块真栈）：
    策略          ms/token   激活 MB/token
    full            0.674        0.72
    sac_attn        0.653        1.03
    sac_narrow      0.562        2.43
    sac_all         0.484        4.07
    (无 ckpt 对照)  0.465        7.52
per-token 代价对每包图数不敏感（G=1..6 变化 <1%），故降 navit_token_budget 让出显存、
换更便宜的策略是净赚——这也是本功能的用法。

Needs CUDA（bf16 + SDPA 后端选择在 CPU 上不具代表性）；无卡时 skip。
Run:
    python -m pytest AnimaLoraToolkit/tests/test_ckpt_policy_equivalence.py -v
"""
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch
    from torch.utils.checkpoint import checkpoint
    HAS_TORCH = True
except ModuleNotFoundError:  # pragma: no cover
    HAS_TORCH = False

if HAS_TORCH:
    from models import krea2_modeling as K
    from models.krea2_modeling import SingleStreamBlock, _SegLens, set_checkpoint_policy

HAS_CUDA = HAS_TORCH and torch.cuda.is_available()
HAS_SAC = False
if HAS_TORCH:
    try:
        from torch.utils.checkpoint import create_selective_checkpoint_contexts  # noqa: F401
        HAS_SAC = True
    except Exception:  # noqa: BLE001  pragma: no cover
        HAS_SAC = False

# 小而真实的构型：features/heads 比例与 KREA2_LARGE_WIDE 一致（GQA 4:1，headdim 128）
FEATURES, HEADS, KVHEADS, MULT = 512, 4, 1, 4
SEGS = [24, 40]
N_TOK = sum(SEGS)


@unittest.skipUnless(HAS_CUDA, "需要 CUDA")
class TestCheckpointPolicyEquivalence(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.dev = "cuda"
        self.dtype = torch.float32          # fp32 让"恒等"断言不被 bf16 舍入掩盖
        self.blk = SingleStreamBlock(FEATURES, HEADS, MULT, False, KVHEADS).to(
            self.dev, self.dtype)
        self.vec = torch.randn(1, len(SEGS), FEATURES * 6, device=self.dev, dtype=self.dtype)
        self.mod_index = torch.repeat_interleave(
            torch.arange(len(SEGS), device=self.dev),
            torch.tensor(SEGS, device=self.dev))
        self.freqs = torch.randn(1, N_TOK, FEATURES // HEADS // 2, 2, 2,
                                 device=self.dev, dtype=torch.float32)
        self.mask = _SegLens(SEGS)
        self.x0 = torch.randn(1, N_TOK, FEATURES, device=self.dev, dtype=self.dtype)
        self.addCleanup(set_checkpoint_policy, "full", 0)

    def _run(self, policy, narrow_width=FEATURES, raw_checkpoint=False):
        set_checkpoint_policy(policy, narrow_width)
        x = self.x0.clone().requires_grad_(True)
        for p in self.blk.parameters():
            p.grad = None

        def _fn(inp):
            return self.blk(inp, self.vec, self.freqs, self.mask,
                            mod_index=self.mod_index)

        out = (checkpoint(_fn, x, use_reentrant=False) if raw_checkpoint
               else K._ckpt(_fn, x))
        out.float().pow(2).mean().backward()
        return out.detach().clone(), x.grad.detach().clone()

    def test_full_is_byte_identical_to_raw_checkpoint(self):
        """policy=full 必须与不经包装的 checkpoint 完全一致（行为中立）。"""
        o1, g1 = self._run("full", raw_checkpoint=True)
        o2, g2 = self._run("full", raw_checkpoint=False)
        self.assertTrue(torch.equal(o1, o2), "full 策略改变了前向输出")
        self.assertTrue(torch.equal(g1, g2), "full 策略改变了输入梯度")

    @unittest.skipUnless(HAS_SAC, "torch 无 create_selective_checkpoint_contexts")
    def test_sac_policies_match_full(self):
        ref_o, ref_g = self._run("full")
        for policy in ("sac_attn", "sac_narrow", "sac_all"):
            with self.subTest(policy=policy):
                o, g = self._run(policy)
                torch.testing.assert_close(o, ref_o, rtol=0, atol=1e-6,
                                           msg=f"{policy} 前向输出与 full 不一致")
                torch.testing.assert_close(g, ref_g, rtol=1e-5, atol=1e-6,
                                           msg=f"{policy} 输入梯度与 full 不一致")

    @unittest.skipUnless(HAS_SAC, "torch 无 create_selective_checkpoint_contexts")
    def test_param_grads_match_full(self):
        """参数梯度也要一致 —— 输出对了但参数梯度错了才是最阴的失败模式。"""
        self._run("full")
        ref = {n: p.grad.detach().clone() for n, p in self.blk.named_parameters()}
        for policy in ("sac_attn", "sac_all"):
            with self.subTest(policy=policy):
                self._run(policy)
                for n, p in self.blk.named_parameters():
                    torch.testing.assert_close(
                        p.grad, ref[n], rtol=1e-5, atol=1e-6,
                        msg=f"{policy} 的参数 {n} 梯度与 full 不一致")


class TestCheckpointPolicyValidation(unittest.TestCase):
    """构造期校验：非法值 fail-fast，不静默退回 full。"""

    @unittest.skipUnless(HAS_TORCH, "需要 torch")
    def test_unknown_policy_rejected(self):
        self.addCleanup(set_checkpoint_policy, "full", 0)
        with self.assertRaises(ValueError):
            set_checkpoint_policy("sac_everything")

    @unittest.skipUnless(HAS_TORCH, "需要 torch")
    def test_full_always_accepted(self):
        self.addCleanup(set_checkpoint_policy, "full", 0)
        set_checkpoint_policy("full")
        self.assertEqual(K._CKPT_POLICY, "full")


if __name__ == "__main__":
    unittest.main(verbosity=2)
