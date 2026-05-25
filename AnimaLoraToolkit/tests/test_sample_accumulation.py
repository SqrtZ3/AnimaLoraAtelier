import pathlib
import sys
import types
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ModuleNotFoundError:
    HAS_TORCH = False
    torch_module = types.ModuleType("torch")
    torch_module.float32 = object()
    torch_module.float16 = object()
    torch_module.bfloat16 = object()
    torch_utils = types.ModuleType("torch.utils")
    torch_data = types.ModuleType("torch.utils.data")

    class Dataset:
        pass

    torch_data.Dataset = Dataset
    torch_utils.data = torch_data
    torch_module.utils = torch_utils
    torch = torch_module
    sys.modules["torch"] = torch_module
    sys.modules["torch.utils"] = torch_utils
    sys.modules["torch.utils.data"] = torch_data

from trainer.data import BucketBatchSampler, compute_sample_accumulation_steps
if HAS_TORCH:
    from trainer.objective import LossConfig, apply_loss_weighting_per_sample
    from trainer.aux_losses import AuxLossConfig, spectral_loss_per_sample


class _BucketOnlyDataset:
    def __init__(self, bucket_sizes):
        self.bucket_for_index = []
        for bucket_id, size in enumerate(bucket_sizes):
            self.bucket_for_index.extend([(bucket_id, bucket_id)] * size)

    def __len__(self):
        return len(self.bucket_for_index)


class SampleAccumulationSamplerTests(unittest.TestCase):
    def test_splits_tail_micro_batch_to_hit_exact_effective_batch_without_loss(self):
        dataset = _BucketOnlyDataset([7] * 14)
        sampler = BucketBatchSampler(
            dataset,
            batch_size=48,
            effective_batch_size=96,
            drop_last=False,
            shuffle=False,
        )

        batches = list(iter(sampler))
        batch_lengths = [len(batch) for batch in batches]

        self.assertEqual(batch_lengths, ([7] * 13) + [5, 2])
        self.assertEqual(sum(batch_lengths), len(dataset))
        self.assertEqual(len(set(i for batch in batches for i in batch)), len(dataset))
        self.assertEqual(len(sampler), len(batches))

    def test_keeps_full_native_batches_when_boundary_already_matches(self):
        dataset = _BucketOnlyDataset([120])
        sampler = BucketBatchSampler(
            dataset,
            batch_size=48,
            effective_batch_size=96,
            drop_last=False,
            shuffle=False,
        )

        self.assertEqual([len(batch) for batch in sampler], [48, 48, 24])

    def test_respects_pending_samples_carried_from_previous_epoch(self):
        dataset = _BucketOnlyDataset([120])
        sampler = BucketBatchSampler(
            dataset,
            batch_size=48,
            effective_batch_size=96,
            drop_last=False,
            shuffle=False,
        )
        sampler.set_accumulation_offset(24)

        self.assertEqual([len(batch) for batch in sampler], [48, 24, 24, 24])
        flat = [i for batch in sampler for i in batch]
        self.assertEqual(sorted(flat), list(range(len(dataset))))

    def test_counts_optimizer_steps_across_epochs_with_one_final_flush(self):
        self.assertEqual(
            compute_sample_accumulation_steps(
                dataset_size=98,
                epochs=3,
                effective_batch_size=96,
            ),
            4,
        )

    @unittest.skipUnless(HAS_TORCH, "torch is required")
    def test_accumulation_loss_weighting_keeps_raw_per_sample_weights(self):
        cfg = LossConfig(
            weighting_scheme="detail_inv_t",
            detail_inv_t_min=1.0,
            detail_inv_t_max=10.0,
            weight_cap_ratio=0.0,
        )
        per_sample = torch.ones(2, dtype=torch.float32)
        t = torch.tensor([0.1, 1.0], dtype=torch.float32)

        weighted = apply_loss_weighting_per_sample(per_sample, t, cfg)

        torch.testing.assert_close(weighted, torch.tensor([10.0, 1.0]))

    @unittest.skipUnless(HAS_TORCH, "torch is required")
    def test_spectral_aux_loss_can_return_per_sample_values_without_active_mean(self):
        cfg = AuxLossConfig(
            spectral_enabled=True,
            spectral_use_wavelet=False,
            spectral_t_gate=0.5,
        )
        x0_pred = torch.zeros(2, 1, 1, 4, 4, dtype=torch.float32)
        x0_target = torch.zeros_like(x0_pred)
        x0_pred[0, :, :, 0, 0] = 1.0
        t = torch.tensor([0.1, 0.9], dtype=torch.float32)

        per_sample = spectral_loss_per_sample(x0_pred, x0_target, t, cfg)

        self.assertEqual(tuple(per_sample.shape), (2,))
        self.assertGreater(float(per_sample[0]), 0.0)
        self.assertEqual(float(per_sample[1]), 0.0)


if __name__ == "__main__":
    unittest.main()
