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
            "sample_every_reference_steps": 40,
            "save_every_reference_steps": 80,
            "keep_vae_on_gpu": True,
            "empty_cache_after_sample": False,
        })

        self.assertEqual(args.reference_batch_size, 4)
        self.assertEqual(args.reference_grad_accum, 1)
        self.assertEqual(args.sample_every_reference_steps, 40)
        self.assertEqual(args.save_every_reference_steps, 80)
        self.assertTrue(args.keep_vae_on_gpu)
        self.assertFalse(args.empty_cache_after_sample)

    def test_renamed_cadence_keys_alias_old_yaml_names(self):
        """节奏参数统一为 *_every_<单位> 后，旧 yaml 键经 ALIASES 落到新属性，值不变。"""
        args = SimpleNamespace(**DEFAULTS)
        apply_yaml_config(args, {
            "save_every": 3,            # → save_every_epochs
            "sample_steps": 50,         # → sample_every_steps
            "log_every": 9,             # → log_every_steps
            "eval_every": 30,           # → eval_every_steps
            "grad_norm_log_every": 15,  # → grad_norm_log_every_steps
            "aclora_restart_every": 99, # → aclora_restart_every_steps
        })
        self.assertEqual(args.save_every_epochs, 3)
        self.assertEqual(args.sample_every_steps, 50)
        self.assertEqual(args.log_every_steps, 9)
        self.assertEqual(args.eval_every_steps, 30)
        self.assertEqual(args.grad_norm_log_every_steps, 15)
        self.assertEqual(args.aclora_restart_every_steps, 99)

    def test_renamed_cadence_new_key_wins_over_old(self):
        """新旧键同在时新键优先，旧键忽略（且旧 attr 不被设）。"""
        args = SimpleNamespace(**DEFAULTS)
        apply_yaml_config(args, {"save_every": 1, "save_every_epochs": 8})
        self.assertEqual(args.save_every_epochs, 8)
        self.assertFalse(hasattr(args, "save_every"))


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
