"""AdamW-SNR 单测：默认行为中立、门控语义、谱形保持、fail-fast。"""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.adamw_snr_optimizer import AdamWSNR
from utils.optimizer_utils import create_optimizer


class TestDefaultsEquivalentToAdamW(unittest.TestCase):
    """默认参数下必须与 torch.optim.AdamW 数学恒等（行为中立的硬证据）。"""

    def _run(self, opt_factory, steps=40):
        torch.manual_seed(0)
        p = torch.randn(32, 64, dtype=torch.float32).requires_grad_(True)
        opt = opt_factory([p])
        g = torch.Generator().manual_seed(1)
        for _ in range(steps):
            p.grad = torch.randn(32, 64, generator=g)
            opt.step()
            p.grad = None
        return p.detach().clone()

    def test_matches_torch_adamw_no_wd(self):
        a = self._run(lambda ps: torch.optim.AdamW(ps, lr=1e-3, betas=(0.9, 0.999),
                                                   eps=1e-8, weight_decay=0.0))
        b = self._run(lambda ps: AdamWSNR(ps, lr=1e-3, betas=(0.9, 0.999),
                                          eps=1e-8, weight_decay=0.0))
        self.assertTrue(torch.allclose(a, b, atol=1e-6),
                        f"最大偏差 {(a - b).abs().max():.3e}")

    def test_matches_torch_adamw_with_wd(self):
        a = self._run(lambda ps: torch.optim.AdamW(ps, lr=1e-3, weight_decay=1e-2))
        b = self._run(lambda ps: AdamWSNR(ps, lr=1e-3, weight_decay=1e-2))
        self.assertTrue(torch.allclose(a, b, atol=1e-6),
                        f"最大偏差 {(a - b).abs().max():.3e}")

    def test_gates_actually_change_trajectory(self):
        base = self._run(lambda ps: AdamWSNR(ps, lr=1e-3))
        caut = self._run(lambda ps: AdamWSNR(ps, lr=1e-3, cautious=True))
        sharp = self._run(lambda ps: AdamWSNR(ps, lr=1e-3, snr_power=2.0))
        self.assertFalse(torch.allclose(base, caut, atol=1e-6))
        self.assertFalse(torch.allclose(base, sharp, atol=1e-6))


class TestSnrSharpening(unittest.TestCase):
    def test_sharpening_concentrates_energy_on_high_snr(self):
        """锐化后，高 SNR 坐标应占更大的更新能量份额。"""
        torch.manual_seed(0)
        p1 = torch.zeros(2048, dtype=torch.float32).requires_grad_(True)
        p2 = torch.zeros(2048, dtype=torch.float32).requires_grad_(True)
        # 前 10% 坐标是稳定信号，其余是噪声
        mu = torch.zeros(2048); mu[:205] = 1.0
        o1 = AdamWSNR([p1], lr=1e-3, snr_power=1.0)
        o2 = AdamWSNR([p2], lr=1e-3, snr_power=2.0)
        g = torch.Generator().manual_seed(3)
        for _ in range(200):
            noise = torch.randn(2048, generator=g)
            grad = mu + noise
            p1.grad = grad.clone(); o1.step(); p1.grad = None
            p2.grad = grad.clone(); o2.step(); p2.grad = None
        d1, d2 = p1.detach().abs(), p2.detach().abs()
        frac1 = (d1[:205] ** 2).sum() / (d1 ** 2).sum()
        frac2 = (d2[:205] ** 2).sum() / (d2 ** 2).sum()
        self.assertGreater(frac2.item(), frac1.item() + 0.05,
                           f"p=2 未显著集中能量: {frac1:.3f} -> {frac2:.3f}")

    def test_sharpening_preserves_step_magnitude(self):
        """重归一化应让不同 p 的总步长量级可比（lr 语义稳定）。"""
        torch.manual_seed(0)
        results = []
        for p_snr in (1.0, 1.5, 2.0):
            torch.manual_seed(0)
            p = torch.zeros(4096, dtype=torch.float32).requires_grad_(True)
            opt = AdamWSNR([p], lr=1e-3, snr_power=p_snr)
            g = torch.Generator().manual_seed(5)
            for _ in range(100):
                p.grad = torch.randn(4096, generator=g) + 0.3
                opt.step(); p.grad = None
            results.append(p.detach().norm().item())
        for r in results[1:]:
            self.assertLess(abs(r - results[0]) / results[0], 0.6,
                            f"步长量级偏离过大: {results}")


class TestCautious(unittest.TestCase):
    def test_mask_zeroes_disagreeing_coords(self):
        """update 与 g 反号的坐标应被掩掉。

        注意构造：需先用一致梯度把动量喂起来。只跑 1~2 步时动量太弱，
        一次反号就会把 m 也带成负号，届时 m 与 g 反而【同号】、掩码不该触发。
        """
        p = torch.zeros(4, dtype=torch.float32).requires_grad_(True)
        opt = AdamWSNR([p], lr=1e-2, cautious=True, betas=(0.9, 0.999))
        for _ in range(30):                      # 喂满动量：m -> ~+1
            p.grad = torch.ones(4); opt.step(); p.grad = None
        before = p.detach().clone()
        # 此时 m≈+1；坐标 1/3 梯度翻成 -1 → m 仍为正 → update 与 g 反号 → 掩掉
        p.grad = torch.tensor([1.0, -1.0, 1.0, -1.0]); opt.step(); p.grad = None
        moved = (p.detach() - before).abs()
        self.assertGreater(moved[0].item(), 0.0, "同号坐标应正常更新")
        self.assertAlmostEqual(moved[1].item(), 0.0, places=7)
        self.assertAlmostEqual(moved[3].item(), 0.0, places=7)

    def test_mask_renormalizes_kept_coords(self):
        """掩掉一半坐标后，保留坐标的步长应被放大以保持期望步长。"""
        p_a = torch.zeros(4, dtype=torch.float32).requires_grad_(True)
        p_b = torch.zeros(4, dtype=torch.float32).requires_grad_(True)
        oa = AdamWSNR([p_a], lr=1e-2, cautious=True)
        ob = AdamWSNR([p_b], lr=1e-2, cautious=False)
        for _ in range(30):
            p_a.grad = torch.ones(4); oa.step(); p_a.grad = None
            p_b.grad = torch.ones(4); ob.step(); p_b.grad = None
        a0, b0 = p_a.detach().clone(), p_b.detach().clone()
        flip = torch.tensor([1.0, -1.0, 1.0, -1.0])
        p_a.grad = flip.clone(); oa.step(); p_a.grad = None
        p_b.grad = flip.clone(); ob.step(); p_b.grad = None
        kept_a = (p_a.detach() - a0)[0].abs().item()
        kept_b = (p_b.detach() - b0)[0].abs().item()
        self.assertAlmostEqual(kept_a / kept_b, 2.0, places=4,
                               msg="保留 2/4 坐标时应放大 2 倍")


class TestSpectralShapePreserved(unittest.TestCase):
    """与 muon 的对照：逐元素门控不得抹平谱形。"""

    def test_rank1_gradient_stays_rank1(self):
        torch.manual_seed(0)
        u = torch.randn(96, 1).abs() + 0.5
        v = torch.randn(1, 192).abs() + 0.5
        g = u @ v
        for kwargs in ({}, {"cautious": True}, {"snr_power": 2.0}):
            p = torch.zeros(96, 192, dtype=torch.float32).requires_grad_(True)
            opt = AdamWSNR([p], lr=1e-3, **kwargs)
            for _ in range(5):
                p.grad = g.clone(); opt.step(); p.grad = None
            sv = torch.linalg.svdvals(p.detach())
            ratio = (sv[0] ** 2 / (sv ** 2).sum()).item()
            self.assertGreater(ratio, 0.99, f"{kwargs} 下 top1 能量仅 {ratio:.4f}")


class TestFailFast(unittest.TestCase):
    def test_snr_power_below_one_rejected(self):
        p = torch.zeros(4, 4, requires_grad=True)
        with self.assertRaises(ValueError):
            AdamWSNR([p], snr_power=0.5)

    def test_snr_power_too_large_rejected(self):
        p = torch.zeros(4, 4, requires_grad=True)
        with self.assertRaises(ValueError):
            AdamWSNR([p], snr_power=5.0)


class TestBf16Master(unittest.TestCase):
    def test_bf16_param_not_frozen(self):
        torch.manual_seed(0)
        init = torch.randn(64, 64) * 6e-3
        p = init.clone().to(torch.bfloat16).requires_grad_(True)
        opt = AdamWSNR([p], lr=1e-6)
        g = (torch.randn(64, 64) * 1e-3).to(torch.bfloat16)
        for _ in range(50):
            p.grad = g.clone(); opt.step(); p.grad = None
        moved = (opt.state[p]["master"] - init).abs().mean().item()
        self.assertGreater(moved, 1e-6, "fp32 master 未累积住亚 ulp 更新")


class TestFactoryWiring(unittest.TestCase):
    def test_dispatch_and_args(self):
        p = torch.zeros(4, 8, requires_grad=True)
        opt = create_optimizer(optimizer_type="adamw_snr", params=[p],
                               learning_rate=5e-5, cautious=True, snr_power=1.5,
                               weight_decay=1e-5)
        self.assertIsInstance(opt, AdamWSNR)
        self.assertTrue(opt.param_groups[0]["cautious"])
        self.assertEqual(opt.param_groups[0]["snr_power"], 1.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
