"""Stage 1 of NaViT/FiT block-diagonal packing: the token-budget packer.

Pure-Python (no torch): validates that ``pack_indices_by_budget`` and
``NavitPackBatchSampler`` produce packs that (a) never exceed the summed token
budget except for unavoidable oversized singletons, (b) cover every sample
exactly once, (c) honour an optional per-pack image cap, and (d) reshuffle
across epochs.
"""
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trainer.data import (  # noqa: E402
    NavitPackBatchSampler,
    dataset_token_counts,
    pack_indices_by_budget,
)


class _FakeDataset:
    """Minimal dataset exposing ``token_count_for_index`` like CachedLatentDataset."""

    def __init__(self, token_counts):
        self.token_count_for_index = list(token_counts)

    def __len__(self):
        return len(self.token_count_for_index)


class _FakeCachedDataset:
    """Mimics the NaViT/non-FiT path: token_count_for_index is all-zero (FiT-only field),
    per-image size lives in bucket_for_index = (h, w) latent px."""

    def __init__(self, latent_shapes):
        self.bucket_for_index = list(latent_shapes)
        self.token_count_for_index = [0] * len(latent_shapes)  # FiT field unset → 0

    def __len__(self):
        return len(self.bucket_for_index)


class PackBudgetFunctionTests(unittest.TestCase):
    def test_sum_within_budget_and_full_coverage(self):
        counts = [10, 20, 30, 40, 15, 25, 5, 50]
        budget = 60
        order = list(range(len(counts)))
        packs = pack_indices_by_budget(counts, budget, order)

        seen = [i for p in packs for i in p]
        self.assertEqual(sorted(seen), sorted(order))          # every index once
        for p in packs:
            s = sum(counts[i] for i in p)
            # a pack may exceed budget only if it is a single oversized image
            self.assertTrue(s <= budget or len(p) == 1, (p, s))

    def test_oversized_image_becomes_singleton(self):
        counts = [100, 10, 10]
        packs = pack_indices_by_budget(counts, token_budget=50, order=[0, 1, 2])
        self.assertIn([0], packs)                              # 100 > 50 → its own pack

    def test_next_fit_flushes_on_overflow(self):
        counts = [30, 30, 30]
        packs = pack_indices_by_budget(counts, token_budget=60, order=[0, 1, 2])
        # 30+30 fills 60 exactly, third starts a new pack
        self.assertEqual(packs, [[0, 1], [2]])

    def test_max_images_per_pack_cap(self):
        counts = [1] * 10
        packs = pack_indices_by_budget(
            counts, token_budget=1000, order=list(range(10)), max_images_per_pack=3
        )
        self.assertTrue(all(len(p) <= 3 for p in packs))
        self.assertEqual(sum(len(p) for p in packs), 10)


class NavitPackBatchSamplerTests(unittest.TestCase):
    def test_full_coverage_default(self):
        ds = _FakeDataset([10, 20, 30, 40, 15, 25, 5, 50])
        sampler = NavitPackBatchSampler(ds, token_budget=60, shuffle=True, seed=1)
        packs = list(sampler)
        seen = [i for p in packs for i in p]
        self.assertEqual(sorted(seen), list(range(len(ds))))
        self.assertEqual(len(sampler), len(packs))

    def test_epoch_changes_packing(self):
        ds = _FakeDataset([7, 11, 13, 17, 19, 23, 29, 31, 37])
        sampler = NavitPackBatchSampler(ds, token_budget=50, shuffle=True, seed=1)
        sampler.set_epoch(0)
        p0 = list(sampler)
        sampler.set_epoch(1)
        p1 = list(sampler)
        # both cover everything...
        self.assertEqual(sorted(i for p in p0 for i in p), list(range(len(ds))))
        self.assertEqual(sorted(i for p in p1 for i in p), list(range(len(ds))))
        # ...but the grouping differs across epochs (reshuffle)
        self.assertNotEqual(p0, p1)

    def test_token_counts_derived_from_latent_shape_when_token_count_zero(self):
        # Regression: on the NaViT path token_count_for_index is all-zero; counts must be
        # derived from bucket_for_index (h//2)*(w//2), else everything packs into one
        # giant sequence → OOM.
        ds = _FakeCachedDataset([(128, 128), (96, 160), (160, 96)])  # latent px
        counts = dataset_token_counts(ds, patch_spatial=2)
        self.assertEqual(counts, [64 * 64, 48 * 80, 80 * 48])

    def test_pack_respects_budget_on_cached_path(self):
        ds = _FakeCachedDataset([(128, 128)] * 5)   # 4096 tokens each
        sampler = NavitPackBatchSampler(ds, token_budget=8192, shuffle=False)
        packs = list(sampler)
        for p in packs:
            self.assertLessEqual(len(p), 2)          # 2*4096 == 8192, never 3
        self.assertEqual(sum(len(p) for p in packs), 5)

    def test_fail_fast_on_all_zero_token_counts(self):
        ds = _FakeDataset([0, 0, 0])                 # nothing resolvable
        with self.assertRaises(RuntimeError):
            NavitPackBatchSampler(ds, token_budget=8192)

    def test_drop_last_removes_trailing_underfilled_pack(self):
        ds = _FakeDataset([60, 60, 5])
        sampler = NavitPackBatchSampler(
            ds, token_budget=60, shuffle=False, drop_last=True
        )
        packs = list(sampler)
        # [0],[1],[2]; last pack (5 < 60) dropped
        self.assertEqual(packs, [[0], [1]])


if __name__ == "__main__":
    unittest.main()
