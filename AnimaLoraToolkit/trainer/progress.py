"""Training progress helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class ReferenceStepTracker:
    """Track old optimizer-step progress from completed reference batches."""

    step: int = 0
    grad_batches_pending: int = 0
    grad_accum: int = 1

    def commit_batches(self, completed_reference_batches: int) -> tuple[int, int]:
        previous_step = int(self.step)
        grad_accum = max(1, int(self.grad_accum or 1))
        total_batches = int(self.grad_batches_pending) + max(0, int(completed_reference_batches or 0))
        advanced_steps, self.grad_batches_pending = divmod(total_batches, grad_accum)
        self.step += advanced_steps
        return previous_step, int(self.step)


def reference_interval_crossed(
    previous_reference_step: int,
    current_reference_step: int,
    interval_reference_steps: int,
) -> bool:
    """Return True when reference optimizer steps cross an interval tick."""
    interval = int(interval_reference_steps or 0)
    if interval <= 0:
        return False

    previous_ref = max(0, int(previous_reference_step or 0))
    current_ref = max(0, int(current_reference_step or 0))
    if current_ref <= previous_ref:
        return False

    return math.floor(current_ref / interval) > math.floor(previous_ref / interval)
