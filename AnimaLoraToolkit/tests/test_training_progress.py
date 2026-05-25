import pathlib
import sys
import unittest
from types import SimpleNamespace


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trainer.config import DEFAULTS, apply_yaml_config
from trainer.progress import ReferenceStepTracker, reference_interval_crossed
from train_monitor import get_state, restore_monitor_state, update_monitor


class ReferenceProgressTests(unittest.TestCase):
    def test_reference_tracker_uses_old_grad_accum(self):
        tracker = ReferenceStepTracker(grad_accum=2)

        self.assertEqual(tracker.commit_batches(1), (0, 0))
        self.assertEqual(tracker.grad_batches_pending, 1)
        self.assertEqual(tracker.commit_batches(3), (0, 2))
        self.assertEqual(tracker.grad_batches_pending, 0)

    def test_reference_interval_crosses_old_optimizer_step_ticks(self):
        self.assertFalse(
            reference_interval_crossed(
                previous_reference_step=24,
                current_reference_step=39,
                interval_reference_steps=40,
            )
        )
        self.assertTrue(
            reference_interval_crossed(
                previous_reference_step=39,
                current_reference_step=48,
                interval_reference_steps=40,
            )
        )
        self.assertFalse(
            reference_interval_crossed(
                previous_reference_step=48,
                current_reference_step=72,
                interval_reference_steps=40,
            )
        )

    def test_reference_interval_disabled_for_zero_interval(self):
        self.assertFalse(reference_interval_crossed(0, 48, 0))


class TrainingProgressConfigTests(unittest.TestCase):
    def test_yaml_maps_reference_progress_and_vae_cache_policy(self):
        args = SimpleNamespace(**DEFAULTS)

        apply_yaml_config(args, {
            "reference_batch_size": 4,
            "reference_grad_accum": 1,
            "sample_reference_steps": 40,
            "save_every_reference_steps": 80,
            "keep_vae_on_gpu": True,
            "empty_cache_after_sample": False,
        })

        self.assertEqual(args.reference_batch_size, 4)
        self.assertEqual(args.reference_grad_accum, 1)
        self.assertEqual(args.sample_reference_steps, 40)
        self.assertEqual(args.save_every_reference_steps, 80)
        self.assertTrue(args.keep_vae_on_gpu)
        self.assertFalse(args.empty_cache_after_sample)


class MonitorReferenceProgressTests(unittest.TestCase):
    def test_monitor_tracks_reference_progress_fields(self):
        restore_monitor_state(losses=[], lr_history=[], epoch=0, step=0, total_steps=0)
        update_monitor(step=7, ref_step=168, samples_seen=672)

        state = get_state()
        self.assertEqual(state["step"], 7)
        self.assertEqual(state["ref_step"], 168)
        self.assertEqual(state["samples_seen"], 672)


if __name__ == "__main__":
    unittest.main()
