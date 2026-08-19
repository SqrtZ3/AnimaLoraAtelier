"""Automagic 优化器单测。

覆盖：lr_mask 符号一致性机制、钳位、fp32 master 防 bf16 ulp 冻结、
分解二阶矩的谱形保持（本仓库 muon_sf 不拟合归因的核心判据）、构造期 fail-fast。
"""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.automagic_optimizer import Automagic
from utils.optimizer_utils import create_optimizer


class TestLrMaskMechanism(unittest.TestCase):
    def test_consistent_sign_raises_lr(self):
        """梯度符号恒定 → lr_mask 每步 +lr_bump，最终顶到 max_lr。"""
        p = torch.zeros(4, 8, requires_grad=True)
        opt = Automagic([p], lr=1e-6, min_lr=1e-7, max_lr=1e-4, lr_bump=1e-5)
        for _ in range(50):
            p.grad = torch.ones_like(p)
            opt.step()
        mask = opt.state[p]["lr_mask"]
        self.assertTrue(torch.allclose(mask, torch.full_like(mask, 1e-4)),
                        f"恒定梯度下 lr 应顶到 max_lr，实际 {mask.min()}~{mask.max()}")

    def test_flipping_sign_lowers_lr(self):
        """梯度符号每步翻转 → lr_mask 一路降到 min_lr。"""
        p = torch.zeros(4, 8, requires_grad=True)
        opt = Automagic([p], lr=1e-4, min_lr=1e-7, max_lr=1e-3, lr_bump=1e-5)
        for i in range(50):
            p.grad = torch.full_like(p, 1.0 if i % 2 == 0 else -1.0)
            opt.step()
        mask = opt.state[p]["lr_mask"]
        self.assertTrue(torch.allclose(mask, torch.full_like(mask, 1e-7)),
                        f"翻转梯度下 lr 应降到 min_lr，实际 {mask.min()}~{mask.max()}")

    def test_lr_mask_is_elementwise(self):
        """半数元素梯度稳定、半数翻转 → lr_mask 分化（逐元素而非全张量）。"""
        p = torch.zeros(2, 8, requires_grad=True)
        opt = Automagic([p], lr=1e-5, min_lr=1e-7, max_lr=1e-4, lr_bump=1e-5)
        for i in range(30):
            g = torch.ones(2, 8)
            g[1] = 1.0 if i % 2 == 0 else -1.0   # 第 1 行来回翻转
            p.grad = g
            opt.step()
        mask = opt.state[p]["lr_mask"]
        self.assertGreater(mask[0].mean().item(), mask[1].mean().item() * 10,
                           "稳定行的 lr 应显著高于翻转行")

    def test_lr_mask_stays_fp32(self):
        """lr_mask 必须 fp32：bf16 下 1e-4 + 1e-6 会被尾数吃掉，机制失效。"""
        p = torch.zeros(4, 8, dtype=torch.bfloat16, requires_grad=True)
        opt = Automagic([p], lr=1e-6, max_lr=1e-3)
        p.grad = torch.ones_like(p)
        opt.step()
        self.assertEqual(opt.state[p]["lr_mask"].dtype, torch.float32)


class TestBf16MasterNoFreeze(unittest.TestCase):
    """对标 muon_optimizer 修复 #2：bf16 参数 + 亚 ulp 更新不得被静默冻结。"""

    def test_bf16_param_still_moves_with_subulp_updates(self):
        torch.manual_seed(0)
        init = torch.randn(64, 64) * 6e-3          # LoRA 参数典型量级
        p = init.clone().to(torch.bfloat16).requires_grad_(True)
        # 起始 lr=1e-6 → 每步更新 ~1e-6，远低于 bf16 在 6e-3 处的 ulp(~3e-5)
        opt = Automagic([p], lr=1e-6, min_lr=1e-7, max_lr=1e-4, lr_bump=1e-7)
        g = torch.randn(64, 64).to(torch.bfloat16)
        for _ in range(100):
            p.grad = g.clone()
            opt.step()
        master = opt.state[p]["master"]
        moved = (master - init).abs()
        self.assertGreater(moved.mean().item(), 1e-5,
                           "fp32 master 应累积住亚 ulp 更新")
        frozen = (p.float() == init.to(torch.bfloat16).float()).float().mean().item()
        self.assertLess(frozen, 0.5,
                        f"bf16 参数被冻结的条目比例过高: {frozen:.1%}")


class TestSpectralShapePreserved(unittest.TestCase):
    """核心判据：更新不得抹平梯度的谱形。

    muon_sf 在 Krea2 上不拟合的根因是 Newton-Schulz 把所有奇异值拉到 1
    （真实 checkpoint 取证：top1 能量 8.9% vs adamw 57.7%，谱条件数 1.7 vs 22.1）。
    Automagic 全链路逐元素，且 Adafactor 的 rank-1 缩放严格保持 rank-1 结构。
    """

    def test_rank1_gradient_yields_rank1_update(self):
        torch.manual_seed(0)
        u = torch.randn(128, 1).abs() + 0.5
        v = torch.randn(1, 256).abs() + 0.5
        g = u @ v                                   # 严格 rank-1、全正
        p = torch.zeros(128, 256, requires_grad=True)
        opt = Automagic([p], lr=1e-4, min_lr=1e-7, max_lr=1e-3, lr_bump=1e-6)
        p.grad = g.clone()
        opt.step()
        delta = -opt.state[p]["master"]             # 起点为 0，master 即 -update
        sv = torch.linalg.svdvals(delta)
        ratio = (sv[0] ** 2 / (sv ** 2).sum()).item()
        self.assertGreater(ratio, 0.99,
                           f"rank-1 梯度应产生 rank-1 更新，top1 能量仅 {ratio:.3f}")

    def test_spiky_spectrum_not_flattened(self):
        """尖谱梯度经 Automagic 后仍应保持尖谱（对照：NS 会压到条件数≈1）。"""
        torch.manual_seed(0)
        U, _ = torch.linalg.qr(torch.randn(128, 8))
        V, _ = torch.linalg.qr(torch.randn(256, 8))
        s = torch.tensor([1.0, .3, .1, .05, .03, .02, .01, .005])
        g = (U * s) @ V.T
        p = torch.zeros(128, 256, requires_grad=True)
        opt = Automagic([p], lr=1e-4, min_lr=1e-7, max_lr=1e-3, lr_bump=1e-6)
        p.grad = g.clone()
        opt.step()
        sv = torch.linalg.svdvals(-opt.state[p]["master"])
        cond = (sv[0] / sv[7]).item()
        self.assertGreater(cond, 20.0,
                           f"谱条件数被压平到 {cond:.1f}（梯度本身是 200）")


class TestFailFast(unittest.TestCase):
    def test_lr_outside_bounds_rejected(self):
        p = torch.zeros(4, 4, requires_grad=True)
        with self.assertRaises(ValueError):
            Automagic([p], lr=1e-3, min_lr=1e-7, max_lr=1e-4)

    def test_min_gt_max_rejected(self):
        p = torch.zeros(4, 4, requires_grad=True)
        with self.assertRaises(ValueError):
            Automagic([p], lr=1e-5, min_lr=1e-3, max_lr=1e-5)

    def test_bad_lr_bump_rejected(self):
        p = torch.zeros(4, 4, requires_grad=True)
        with self.assertRaises(ValueError):
            Automagic([p], lr=1e-5, lr_bump=0.0)


class TestFactoryWiring(unittest.TestCase):
    def test_create_optimizer_dispatches(self):
        p = torch.zeros(4, 8, requires_grad=True)
        opt = create_optimizer(
            optimizer_type="automagic", params=[p], learning_rate=1e-6,
            max_lr=1e-4, lr_bump=1e-6, weight_decay=1e-5,
        )
        self.assertIsInstance(opt, Automagic)
        self.assertEqual(opt.max_lr, 1e-4)
        self.assertEqual(opt.param_groups[0]["weight_decay"], 1e-5)

    def test_betas_silently_dropped(self):
        """Automagic 无一阶动量：betas 不适用，不得因此抛错。"""
        p = torch.zeros(4, 8, requires_grad=True)
        opt = create_optimizer(
            optimizer_type="automagic", params=[p], learning_rate=1e-6,
            betas=(0.9, 0.999), max_lr=1e-4,
        )
        self.assertIsInstance(opt, Automagic)


class TestStateRoundtrip(unittest.TestCase):
    def test_state_dict_roundtrip_preserves_mask(self):
        p = torch.zeros(4, 8, requires_grad=True)
        opt = Automagic([p], lr=1e-5, min_lr=1e-7, max_lr=1e-4, lr_bump=1e-5)
        for _ in range(10):
            p.grad = torch.ones_like(p)
            opt.step()
        saved = {k: v.clone() if torch.is_tensor(v) else v
                 for k, v in opt.state[p].items()}
        opt2 = Automagic([p], lr=1e-5, min_lr=1e-7, max_lr=1e-4, lr_bump=1e-5)
        opt2.load_state_dict(opt.state_dict())
        self.assertTrue(torch.allclose(opt2.state[p]["lr_mask"], saved["lr_mask"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
