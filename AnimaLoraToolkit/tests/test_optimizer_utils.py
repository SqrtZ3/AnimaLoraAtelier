import logging
import pathlib
import sys
import unittest

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.optimizer_utils import create_optimizer


class SoapOptimizerFactoryTests(unittest.TestCase):
    def test_create_soap_preserves_param_group_lr_and_weight_decay(self):
        matrix = torch.nn.Parameter(torch.randn(4, 3))
        vector = torch.nn.Parameter(torch.randn(4))
        groups = [
            {"params": [matrix], "lr": 2.0e-5, "weight_decay": 0.01},
            {"params": [vector], "lr": 8.0e-6, "weight_decay": 0.0},
        ]

        optimizer = create_optimizer(
            "soap",
            groups,
            learning_rate=2.0e-5,
            betas=(0.95, 0.95),
            weight_decay=0.01,
            eps=1.0e-8,
            precondition_frequency=10,
            max_precond_dim=512,
            precondition_1d=False,
            merge_dims=False,
        )

        self.assertEqual(type(optimizer).__name__, "SOAP")
        self.assertEqual(optimizer.param_groups[0]["lr"], 2.0e-5)
        self.assertEqual(optimizer.param_groups[1]["lr"], 8.0e-6)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.01)
        self.assertEqual(optimizer.param_groups[1]["weight_decay"], 0.0)

    def test_soap_step_updates_2d_parameter_with_finite_values(self):
        param = torch.nn.Parameter(torch.tensor([[1.0, -2.0], [0.5, 3.0]]))
        optimizer = create_optimizer(
            "soap",
            [param],
            learning_rate=1.0e-3,
            betas=(0.95, 0.95),
            weight_decay=0.0,
            precondition_frequency=1,
            max_precond_dim=16,
            merge_dims=False,
        )

        before = param.detach().clone()
        loss = (param.square()).sum()
        loss.backward()
        optimizer.step()

        self.assertFalse(torch.equal(before, param.detach()))
        self.assertTrue(torch.isfinite(param.detach()).all())

    def test_soap_step_updates_1d_parameter_without_preconditioner_when_disabled(self):
        param = torch.nn.Parameter(torch.tensor([1.0, -2.0, 0.5]))
        optimizer = create_optimizer(
            "soap",
            [param],
            learning_rate=1.0e-3,
            betas=(0.95, 0.95),
            weight_decay=0.0,
            precondition_frequency=1,
            max_precond_dim=16,
            precondition_1d=False,
            merge_dims=False,
        )

        loss = (param.square()).sum()
        loss.backward()
        optimizer.step()

        state = optimizer.state[param]
        self.assertTrue(torch.isfinite(param.detach()).all())
        self.assertNotIn("GG", state)
        self.assertNotIn("Q", state)

    def test_soap_ignores_unsupported_optimizer_args_with_warning(self):
        param = torch.nn.Parameter(torch.randn(2, 2))

        with self.assertLogs("utils.optimizer_utils", level=logging.WARNING) as logs:
            optimizer = create_optimizer(
                "soap",
                [param],
                learning_rate=1.0e-3,
                made_up_option=True,
            )

        self.assertEqual(type(optimizer).__name__, "SOAP")
        self.assertTrue(any("Ignored unsupported params" in line for line in logs.output))


if __name__ == "__main__":
    unittest.main()
