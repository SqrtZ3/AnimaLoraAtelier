"""token_bucket fit 路径补全的数据层测试：

- ``collate_fn_cached_fit``：堆叠单一网格 latent + 重建全 1 latent_mask；混合尺寸报错。
- ``BucketBatchSampler``：对 fit-like 数据集按精确 (h,w) 网格分批，保证每 batch 单一网格
  （这是 cache_latents / aux / compile 三者正确性的共同前提）。
"""
import pathlib
import sys
import unittest

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trainer.data import collate_fn_cached_fit, BucketBatchSampler


class TestCollateCachedFit(unittest.TestCase):
    def _item(self, h, w, cap="c", img="x.png"):
        return {"latent": torch.randn(16, 1, h, w), "caption": cap, "image": img}

    def test_stacks_and_builds_all_ones_mask(self):
        batch = [self._item(4, 6, "a", "a.png"), self._item(4, 6, "b", "b.png")]
        out = collate_fn_cached_fit(batch)
        self.assertEqual(tuple(out["latents"].shape), (2, 16, 1, 4, 6))
        self.assertEqual(tuple(out["latent_mask"].shape), (2, 1, 4, 6))
        self.assertTrue(torch.equal(out["latent_mask"], torch.ones(2, 1, 4, 6)))
        self.assertEqual(out["captions"], ["a", "b"])
        self.assertEqual(out["images"], ["a.png", "b.png"])

    def test_mixed_shapes_raise(self):
        batch = [self._item(4, 6), self._item(4, 8)]
        with self.assertRaisesRegex(RuntimeError, "不同 latent 尺寸"):
            collate_fn_cached_fit(batch)


class _FakeFitDataset:
    """最小数据集：只暴露 BucketBatchSampler 需要的 bucket_for_index + __len__。"""

    def __init__(self, grids):
        # grids: list of (h, w) latent 网格键，每样本一个
        self.bucket_for_index = list(grids)

    def __len__(self):
        return len(self.bucket_for_index)


class TestBucketSamplerSingleGrid(unittest.TestCase):
    def test_every_batch_is_single_grid(self):
        # 三种网格混在一起；token 数可能相同但网格不同 —— 采样器必须按精确 (h,w) 分批。
        grids = [(63, 64)] * 5 + [(84, 48)] * 4 + [(64, 63)] * 3
        ds = _FakeFitDataset(grids)
        sampler = BucketBatchSampler(ds, batch_size=2, drop_last=False, shuffle=True, seed=7)
        seen = []
        for batch in sampler:
            keys = {ds.bucket_for_index[i] for i in batch}
            self.assertEqual(len(keys), 1, f"batch 含多种网格: {keys}")
            seen.extend(batch)
        # 所有样本都被覆盖（drop_last=False）
        self.assertEqual(sorted(seen), list(range(len(grids))))


if __name__ == "__main__":
    unittest.main()
