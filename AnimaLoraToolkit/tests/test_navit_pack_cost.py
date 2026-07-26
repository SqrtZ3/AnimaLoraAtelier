# -*- coding: utf-8 -*-
"""按代价装包（navit_pack_cost_lambda）：λ=0 与改动前逐包等价；λ>0 按二次代价重定价。

依据（见 docs/navit-packing.md §2.4）：包的步时不是 ΣN 的线性函数——attention 对每图
自身序列长度是二次的。H20 / Krea2 12B 实测同 ΣN=55778 下 G=1(3960ms) 比 G=16(1719ms)
慢 2.3×；拟合 t = a·ΣN + b·ΣN_i² 得 λ=b/a=2.742e-05（R²=0.9999），与真实 run 的
stage_timing 整步拟合值 3.279e-05 相差 16%。

本测试只覆盖装包算法本身（纯 Python，无需 GPU/torch 之外的东西）：
  1. λ=0 时 next_fit / ffd 的输出与不传 costs 完全一致（行为中立）；
  2. 归一后 n_ref 尺寸的图代价 == 其 token 数（均匀数据集容量不变）；
  3. λ>0 时大图占更多预算（每包装得更少）、小图占更少（装得更多）；
  4. 覆盖性：任何 λ 下每个索引恰好出现一次；
  5. max_images_per_pack 上限在两种策略下都仍然生效。

Run:
    python -m pytest AnimaLoraToolkit/tests/test_navit_pack_cost.py -v
"""
import pathlib
import random
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trainer.data import (  # noqa: E402
    navit_pack_costs,
    pack_indices_by_budget,
    pack_indices_ffd_windowed,
)

BUDGET = 49152


def _mixed_counts(n=200, seed=0):
    """混合尺寸数据集：大量 ~4k token 小图 + 少量 ~14k~30k 大图。"""
    rng = random.Random(seed)
    counts = []
    for _ in range(n):
        r = rng.random()
        if r < 0.7:
            counts.append(rng.choice([3072, 4096, 4608]))
        elif r < 0.95:
            counts.append(rng.choice([9216, 13944]))
        else:
            counts.append(rng.choice([24576, 30720]))
    return counts


class TestPackCostNeutral(unittest.TestCase):
    """λ=0 → 与改动前逐包等价。"""

    def test_costs_are_token_counts_when_lambda_zero(self):
        counts = _mixed_counts()
        costs, scale, ref = navit_pack_costs(counts, cost_lambda=0.0)
        self.assertEqual([int(c) for c in costs], counts)
        self.assertEqual(scale, 1.0)
        self.assertEqual(ref, 0)

    def test_next_fit_identical_without_costs(self):
        counts = _mixed_counts()
        order = list(range(len(counts)))
        random.Random(7).shuffle(order)
        legacy = pack_indices_by_budget(counts, BUDGET, order, 0)
        costs, _, _ = navit_pack_costs(counts, cost_lambda=0.0)
        with_costs = pack_indices_by_budget(counts, BUDGET, order, 0, costs=costs)
        self.assertEqual(legacy, with_costs)

    def test_ffd_identical_without_costs(self):
        counts = _mixed_counts()
        order = list(range(len(counts)))
        random.Random(11).shuffle(order)
        legacy = pack_indices_ffd_windowed(counts, BUDGET, order, 0, 64)
        costs, _, _ = navit_pack_costs(counts, cost_lambda=0.0)
        with_costs = pack_indices_ffd_windowed(counts, BUDGET, order, 0, 64, costs=costs)
        self.assertEqual(legacy, with_costs)


class TestPackCostRepricing(unittest.TestCase):
    """λ>0 → 按二次代价重定价，且归一点行为中立。"""

    LAM = 2.742e-05

    def test_reference_size_costs_its_token_count(self):
        counts = [4096] * 10 + [16384] * 3
        costs, _, ref = navit_pack_costs(counts, self.LAM, cost_ref_tokens=4096)
        self.assertEqual(ref, 4096)
        for c, n in zip(costs, counts):
            if n == 4096:
                self.assertAlmostEqual(c, 4096.0, places=6)

    def test_auto_reference_is_dataset_median(self):
        counts = [1000, 2000, 3000, 4000, 5000]
        _, _, ref = navit_pack_costs(counts, self.LAM, cost_ref_tokens=0)
        self.assertEqual(ref, 3000)

    def test_uniform_dataset_capacity_unchanged(self):
        """尺寸均匀时 λ>0 不该改变容量（归一点的意义）。"""
        counts = [4096] * 64
        order = list(range(len(counts)))
        base = pack_indices_by_budget(counts, BUDGET, order, 0)
        costs, _, _ = navit_pack_costs(counts, self.LAM, cost_ref_tokens=0)
        repriced = pack_indices_by_budget(counts, BUDGET, order, 0, costs=costs)
        self.assertEqual([len(p) for p in base], [len(p) for p in repriced])

    def test_large_images_get_more_expensive(self):
        counts = [4096] * 4 + [30720] * 4
        costs, _, _ = navit_pack_costs(counts, self.LAM, cost_ref_tokens=4096)
        small, large = costs[0], costs[-1]
        self.assertAlmostEqual(small / 4096.0, 1.0, places=6)
        # 30720 token 的图：(1+λ·30720)/(1+λ·4096) ≈ 1.64
        self.assertGreater(large / 30720.0, 1.5)

    def test_big_image_pack_holds_fewer(self):
        """同一预算下，大图包在 λ>0 时装得更少（消步时/显存尖峰）。"""
        counts = [13944] * 40
        order = list(range(len(counts)))
        base = pack_indices_by_budget(counts, BUDGET, order, 0)
        costs, _, _ = navit_pack_costs(counts, self.LAM, cost_ref_tokens=4096)
        repriced = pack_indices_by_budget(counts, BUDGET, order, 0, costs=costs)
        self.assertLess(max(len(p) for p in repriced), max(len(p) for p in base))

    def test_small_image_pack_holds_more(self):
        """小图包在 λ>0 时装得更多（提吞吐）。参考尺寸取数据集里的大图档。"""
        counts = [2048] * 60
        order = list(range(len(counts)))
        base = pack_indices_by_budget(counts, BUDGET, order, 0)
        costs, _, _ = navit_pack_costs(counts, self.LAM, cost_ref_tokens=13944)
        repriced = pack_indices_by_budget(counts, BUDGET, order, 0, costs=costs)
        self.assertGreater(max(len(p) for p in repriced), max(len(p) for p in base))


class TestPackCostInvariants(unittest.TestCase):
    """无论 λ 取值，装包的硬不变量都要成立。"""

    def _assert_covers_once(self, packs, n):
        flat = [i for p in packs for i in p]
        self.assertEqual(sorted(flat), list(range(n)))

    def test_coverage_exact_for_both_strategies(self):
        counts = _mixed_counts(n=137, seed=3)
        order = list(range(len(counts)))
        random.Random(5).shuffle(order)
        for lam in (0.0, 2.742e-05, 1e-4):
            costs, _, _ = navit_pack_costs(counts, lam)
            c = costs if lam > 0 else None
            self._assert_covers_once(
                pack_indices_by_budget(counts, BUDGET, order, 0, costs=c), len(counts))
            self._assert_covers_once(
                pack_indices_ffd_windowed(counts, BUDGET, order, 0, 32, costs=c), len(counts))

    def test_max_images_per_pack_still_enforced(self):
        counts = _mixed_counts(n=90, seed=9)
        order = list(range(len(counts)))
        costs, _, _ = navit_pack_costs(counts, 2.742e-05)
        for packs in (
            pack_indices_by_budget(counts, BUDGET, order, 4, costs=costs),
            pack_indices_ffd_windowed(counts, BUDGET, order, 4, 32, costs=costs),
        ):
            self.assertLessEqual(max(len(p) for p in packs), 4)

    def test_oversized_image_becomes_singleton(self):
        """单图代价超预算时仍单独成包（不能被静默丢掉）。"""
        counts = [4096, 4096, 200000, 4096]
        order = list(range(len(counts)))
        costs, _, _ = navit_pack_costs(counts, 2.742e-05, cost_ref_tokens=4096)
        packs = pack_indices_by_budget(counts, BUDGET, order, 0, costs=costs)
        self._assert_covers_once(packs, len(counts))
        self.assertIn([2], packs)

    def test_negative_inputs_fail_fast(self):
        """负值是笔误：静默按 0/自动兜底会让人误以为开关生效了。"""
        with self.assertRaises(ValueError):
            navit_pack_costs([4096, 8192], cost_lambda=-1e-5)
        with self.assertRaises(ValueError):
            navit_pack_costs([4096, 8192], cost_lambda=1e-5, cost_ref_tokens=-4096)


if __name__ == "__main__":
    unittest.main(verbosity=2)
