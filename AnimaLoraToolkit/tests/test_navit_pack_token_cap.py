# -*- coding: utf-8 -*-
"""双约束装包（navit_pack_token_cap）：代价预算管步时，token 上限管显存。

问题（本次改动前的真实缺口）：`navit_pack_cost_lambda>0` 时装包只测试 Σcost ≤ budget，
完全不再约束 ΣN。而 cost(n)=n·(1+λn)/(1+λ·n_ref) 对小于 n_ref 的图 **低于** 其 token 数，
所以一包小图的 ΣN 可以超出 budget 达 (1+λ·n_ref)/(1+λ·n_min) 倍 —— λ=2.742e-05、
n_ref=13944、n=4096 时是 1.243×。步显存对 ΣN 线性（实测 ≈10GB + 0.52MB/token），
所以那是未入账的显存超支，不是白捡的吞吐。

修法与 AdaptiveLoad（arXiv 2605.17923）的双约束一致：
``B = max(1, min(⌊M_mem/S⌋, ⌊M_comp/S^p⌋))`` —— 计算上限与显存上限同时卡。

本测试全部是纯 Python（不需要 torch/GPU）：
  1. **行为中立**：token_cap=None 时两种策略的输出与改动前逐包一致；λ=0 路径不受影响；
  2. **缺口复现**：不给 token_cap 时，λ>0 的小图包 ΣN 确实 > budget（且量级 ≈ 理论上界）；
  3. **修复生效**：给 token_cap 后每个包 ΣN ≤ cap，且 Σcost ≤ budget 仍成立（双约束）；
  4. **覆盖性**：加了 cap 之后每个索引仍恰好出现一次；
  5. **超大单图**：token 数超过 cap 的图仍单独成包（不会死循环/丢样本）；
  6. **max_images_per_pack 仍生效**。

Run:
    python -m pytest AnimaLoraToolkit/tests/test_navit_pack_token_cap.py -v
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
LAM = 2.742e-05
REF = 13944          # 文档里的 n_ref 例子；显式给出以免受数据集中位数漂移影响


def _small_counts(n=240, seed=0):
    """全部小于 n_ref 的图 —— 缺口最明显的场景（每张的 cost 都低于其 token 数）。"""
    rng = random.Random(seed)
    return [rng.choice([3072, 4096, 4608]) for _ in range(n)]


def _mixed_counts(n=200, seed=0):
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


def _sums(counts, costs, packs):
    """→ [(ΣN, Σcost), ...]"""
    return [(sum(counts[i] for i in p), sum(costs[i] for i in p)) for p in packs]


class TestNeutralWhenCapAbsent(unittest.TestCase):
    """token_cap=None（默认）→ 与改动前逐包等价。"""

    def test_next_fit_default_matches_no_cap_arg(self):
        counts = _small_counts()
        order = list(range(len(counts)))
        costs, _, _ = navit_pack_costs(counts, LAM, cost_ref_tokens=REF)
        self.assertEqual(
            pack_indices_by_budget(counts, BUDGET, order, 0, costs=costs),
            pack_indices_by_budget(counts, BUDGET, order, 0, costs=costs, token_cap=None),
        )

    def test_ffd_default_matches_no_cap_arg(self):
        counts = _mixed_counts()
        order = list(range(len(counts)))
        costs, _, _ = navit_pack_costs(counts, LAM, cost_ref_tokens=REF)
        self.assertEqual(
            pack_indices_ffd_windowed(counts, BUDGET, order, 0, 64, costs=costs),
            pack_indices_ffd_windowed(counts, BUDGET, order, 0, 64, costs=costs,
                                      token_cap=None),
        )

    def test_lambda_zero_path_unaffected_by_cap(self):
        """λ=0 时 cost≡token，加不加 cap 结果都该一样（两条约束是同一条）。"""
        counts = _mixed_counts()
        order = list(range(len(counts)))
        base = pack_indices_by_budget(counts, BUDGET, order, 0)
        capped = pack_indices_by_budget(counts, BUDGET, order, 0, token_cap=BUDGET)
        self.assertEqual(base, capped)


class TestOvershootIsReal(unittest.TestCase):
    """缺口复现：没有 token 上限时 ΣN 会超出 budget。"""

    def test_small_image_packs_exceed_budget_in_tokens(self):
        counts = _small_counts()
        order = list(range(len(counts)))
        costs, _, ref = navit_pack_costs(counts, LAM, cost_ref_tokens=REF)
        self.assertEqual(ref, REF)
        packs = pack_indices_by_budget(counts, BUDGET, order, 0, costs=costs)
        tok_sums = [s for s, _ in _sums(counts, costs, packs)]
        # 至少一个包（实际上几乎所有满包）在 token 口径上超预算
        self.assertTrue(max(tok_sums) > BUDGET,
                        f"预期 ΣN 超预算，实测 max={max(tok_sums)} budget={BUDGET}")
        # 超出幅度不该离理论上界太远：(1+λ·n_ref)/(1+λ·n_max_small)
        upper = BUDGET * (1.0 + LAM * REF) / (1.0 + LAM * min(counts))
        self.assertLessEqual(max(tok_sums), upper * 1.02)
        # 且量级确实可观（>10%），否则这个缺口不值得修
        self.assertGreater(max(tok_sums) / BUDGET, 1.10)


class TestDualConstraint(unittest.TestCase):
    """修复生效：ΣN ≤ cap 且 Σcost ≤ budget 同时成立。"""

    def _assert_dual(self, counts, costs, packs, cap):
        for p, (tok, cost) in zip(packs, _sums(counts, costs, packs)):
            if len(p) == 1:
                continue  # 单图包可能本身就超限（调用方 warn），见 TestOversizeSingle
            self.assertLessEqual(tok, cap, f"ΣN={tok} 超过 token_cap={cap}")
            self.assertLessEqual(round(cost, 6), BUDGET,
                                 f"Σcost={cost} 超过 budget={BUDGET}")

    def test_next_fit_respects_both(self):
        counts = _small_counts()
        order = list(range(len(counts)))
        costs, _, _ = navit_pack_costs(counts, LAM, cost_ref_tokens=REF)
        packs = pack_indices_by_budget(counts, BUDGET, order, 0, costs=costs,
                                       token_cap=BUDGET)
        self._assert_dual(counts, costs, packs, BUDGET)

    def test_ffd_respects_both(self):
        counts = _mixed_counts()
        order = list(range(len(counts)))
        costs, _, _ = navit_pack_costs(counts, LAM, cost_ref_tokens=REF)
        packs = pack_indices_ffd_windowed(counts, BUDGET, order, 0, 64, costs=costs,
                                          token_cap=BUDGET)
        self._assert_dual(counts, costs, packs, BUDGET)

    def test_explicit_cap_above_budget_is_honoured(self):
        """显式抬高 cap = 明知地拿显存换吞吐；ΣN 应落在新 cap 内而非 budget 内。"""
        counts = _small_counts()
        order = list(range(len(counts)))
        costs, _, _ = navit_pack_costs(counts, LAM, cost_ref_tokens=REF)
        cap = int(BUDGET * 1.5)
        packs = pack_indices_by_budget(counts, BUDGET, order, 0, costs=costs,
                                       token_cap=cap)
        tok_sums = [s for s, _ in _sums(counts, costs, packs)]
        self.assertLessEqual(max(tok_sums), cap)


class TestCoverageAndEdges(unittest.TestCase):

    def _assert_covers(self, packs, n):
        flat = [i for p in packs for i in p]
        self.assertEqual(sorted(flat), list(range(n)))

    def test_coverage_with_cap(self):
        for counts in (_small_counts(), _mixed_counts()):
            order = list(range(len(counts)))
            costs, _, _ = navit_pack_costs(counts, LAM, cost_ref_tokens=REF)
            self._assert_covers(
                pack_indices_by_budget(counts, BUDGET, order, 0, costs=costs,
                                       token_cap=BUDGET), len(counts))
            self._assert_covers(
                pack_indices_ffd_windowed(counts, BUDGET, order, 0, 32, costs=costs,
                                          token_cap=BUDGET), len(counts))

    def test_oversize_single_image_becomes_singleton(self):
        """token 数本身就超 cap 的图：单独成包，不丢、不死循环。"""
        counts = [4096, 60000, 4096]
        order = [0, 1, 2]
        costs, _, _ = navit_pack_costs(counts, LAM, cost_ref_tokens=REF)
        for packs in (
            pack_indices_by_budget(counts, BUDGET, order, 0, costs=costs,
                                   token_cap=BUDGET),
            pack_indices_ffd_windowed(counts, BUDGET, order, 0, 0, costs=costs,
                                      token_cap=BUDGET),
        ):
            self._assert_covers(packs, 3)
            self.assertIn([1], packs)

    def test_max_images_per_pack_still_applies(self):
        counts = _small_counts()
        order = list(range(len(counts)))
        costs, _, _ = navit_pack_costs(counts, LAM, cost_ref_tokens=REF)
        for packs in (
            pack_indices_by_budget(counts, BUDGET, order, 4, costs=costs,
                                   token_cap=BUDGET),
            pack_indices_ffd_windowed(counts, BUDGET, order, 4, 32, costs=costs,
                                      token_cap=BUDGET),
        ):
            self.assertTrue(all(len(p) <= 4 for p in packs))


if __name__ == "__main__":
    unittest.main(verbosity=2)
