import logging
import pathlib
import sys
import unittest

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.optimizer_utils import create_optimizer


class AdamWOptimizerFactoryTests(unittest.TestCase):
    def test_adamw_accepts_yaml_string_optimizer_args(self):
        param = torch.nn.Parameter(torch.randn(2, 2))
        optimizer = create_optimizer(
            "adamw",
            [param],
            learning_rate=1.0e-4,
            lr="3e-4",
            betas=["0.9", "0.99"],
            weight_decay="0.03",
            eps="1e-8",
        )

        self.assertEqual(type(optimizer).__name__, "AdamW")
        self.assertEqual(optimizer.param_groups[0]["lr"], 3.0e-4)
        self.assertEqual(optimizer.param_groups[0]["betas"], (0.9, 0.99))
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.03)
        self.assertEqual(optimizer.param_groups[0]["eps"], 1.0e-8)


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


class AdoptOptimizerFactoryTests(unittest.TestCase):
    def test_create_adopt_preserves_param_group_lr_and_weight_decay(self):
        matrix = torch.nn.Parameter(torch.randn(4, 3))
        vector = torch.nn.Parameter(torch.randn(4))
        groups = [
            {"params": [matrix], "lr": 5.0e-5, "weight_decay": 0.01},
            {"params": [vector], "lr": 1.0e-5, "weight_decay": 0.0},
        ]

        optimizer = create_optimizer(
            "adopt",
            groups,
            learning_rate=5.0e-5,
            betas=(0.9, 0.9999),
            weight_decay=0.01,
            eps=1.0e-6,
            decoupled=True,
            use_clip=True,
            clip_exponent=0.25,
        )

        self.assertEqual(type(optimizer).__name__, "ADOPT")
        self.assertEqual(optimizer.param_groups[0]["lr"], 5.0e-5)
        self.assertEqual(optimizer.param_groups[1]["lr"], 1.0e-5)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.01)
        self.assertEqual(optimizer.param_groups[1]["weight_decay"], 0.0)

    def test_adopt_two_steps_update_param_with_finite_values(self):
        # ADOPT does NOT update on step 1 (initializes v with g_0^2 first).
        # Verify the first step is a no-op for params, but the second step moves them.
        param = torch.nn.Parameter(torch.tensor([[1.0, -2.0], [0.5, 3.0]]))
        optimizer = create_optimizer(
            "adopt",
            [param],
            learning_rate=1.0e-2,
            betas=(0.9, 0.9999),
            weight_decay=0.0,
        )

        before_step1 = param.detach().clone()
        loss = param.square().sum()
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.equal(before_step1, param.detach()),
                        "ADOPT must NOT update params on the first step")
        optimizer.zero_grad()

        loss = param.square().sum()
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(before_step1, param.detach()))
        self.assertTrue(torch.isfinite(param.detach()).all())

    def test_adopt_uses_large_beta2_default_when_not_specified(self):
        # The dispatch should upgrade the default (0.9, 0.999) → (0.9, 0.9999).
        param = torch.nn.Parameter(torch.randn(2, 2))
        optimizer = create_optimizer("adopt", [param], learning_rate=1.0e-4)
        self.assertEqual(optimizer.param_groups[0]["betas"], (0.9, 0.9999))

    def test_adopt_ignores_unsupported_optimizer_args_with_warning(self):
        param = torch.nn.Parameter(torch.randn(2, 2))

        with self.assertLogs("utils.optimizer_utils", level=logging.WARNING) as logs:
            optimizer = create_optimizer(
                "adopt",
                [param],
                learning_rate=1.0e-4,
                shampoo_beta=0.9,  # SOAP-specific, not valid for ADOPT
            )

        self.assertEqual(type(optimizer).__name__, "ADOPT")
        self.assertTrue(any("Ignored unsupported params" in line for line in logs.output))


class LionOptimizerFactoryTests(unittest.TestCase):
    def test_create_lion_factory_basic(self):
        param = torch.nn.Parameter(torch.randn(4, 4))
        optimizer = create_optimizer(
            "lion",
            [param],
            learning_rate=1.0e-5,
            weight_decay=0.05,
        )
        self.assertEqual(type(optimizer).__name__, "Lion")
        self.assertFalse(optimizer.param_groups[0]["cautious"])
        # Default betas should fall back to Lion's recommended (0.9, 0.99)
        # when the caller passes the AdamW default (0.9, 0.999).
        self.assertEqual(optimizer.param_groups[0]["betas"], (0.9, 0.99))

    def test_clion_dispatches_with_cautious_true(self):
        param = torch.nn.Parameter(torch.randn(4, 4))
        optimizer = create_optimizer(
            "clion",
            [param],
            learning_rate=1.0e-5,
            weight_decay=0.05,
        )
        self.assertEqual(type(optimizer).__name__, "Lion")
        self.assertTrue(optimizer.param_groups[0]["cautious"])

    def test_clion_accepts_yaml_string_lr_override(self):
        param = torch.nn.Parameter(torch.randn(4, 4))
        optimizer = create_optimizer(
            "clion",
            [param],
            learning_rate=1.0e-4,
            lr="3e-4",
            weight_decay=0.03,
        )

        self.assertEqual(type(optimizer).__name__, "Lion")
        self.assertEqual(optimizer.param_groups[0]["lr"], 3.0e-4)
        self.assertTrue(optimizer.param_groups[0]["cautious"])

    def test_lion_accepts_yaml_string_param_group_lr(self):
        matrix = torch.nn.Parameter(torch.randn(4, 3))
        vector = torch.nn.Parameter(torch.randn(4))
        groups = [
            {
                "params": [matrix],
                "lr": "3e-4",
                "weight_decay": "0.03",
                "betas": ["0.9", "0.99"],
            },
            {"params": [vector], "lr": "8e-5", "weight_decay": 0.0},
        ]

        optimizer = create_optimizer(
            "lion",
            groups,
            learning_rate=1.0e-4,
        )

        self.assertEqual(type(optimizer).__name__, "Lion")
        self.assertEqual(optimizer.param_groups[0]["lr"], 3.0e-4)
        self.assertEqual(optimizer.param_groups[1]["lr"], 8.0e-5)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.03)
        self.assertEqual(optimizer.param_groups[0]["betas"], (0.9, 0.99))

    def test_lion_step_updates_param_with_finite_values(self):
        param = torch.nn.Parameter(torch.tensor([[1.0, -2.0], [0.5, 3.0]]))
        optimizer = create_optimizer(
            "lion",
            [param],
            learning_rate=1.0e-3,
            weight_decay=0.0,  # isolate the sign-update behaviour; no WD drift
        )

        before = param.detach().clone()
        loss = param.square().sum()
        loss.backward()
        optimizer.step()

        self.assertFalse(torch.equal(before, param.detach()))
        self.assertTrue(torch.isfinite(param.detach()).all())
        # Lion sign-update: each coordinate moves by exactly lr (sign of grad).
        # Initial grad of x^2 is 2x; first call to .sign_() gives ±1.
        deltas = (param.detach() - before).abs()
        self.assertTrue(torch.allclose(deltas, torch.full_like(deltas, 1.0e-3), atol=1e-6))

    def test_clion_mask_zeros_disagreeing_coords(self):
        # We seed the momentum directly so the update direction is determined
        # (otherwise Lion needs many warm-up steps to make momentum dominate
        # the (1-β1)*g term).
        param = torch.nn.Parameter(torch.tensor([10.0, -10.0]))
        optimizer = create_optimizer(
            "clion",
            [param],
            learning_rate=1.0,
            betas=(0.9, 0.99),
            weight_decay=0.0,  # isolate the cautious mask, drop AdamW WD drift
        )
        # Bypass first-step init by pre-seeding the state.
        optimizer.state[param] = {
            "exp_avg": torch.tensor([1.0, 1.0], dtype=torch.float32),
        }

        # Small opposing grads — momentum (sign +) dominates the update direction
        # on both coords, but the cautious mask should kill coord 0 where grad < 0.
        param.grad = torch.tensor([-0.01, 0.01])
        before = param.detach().clone()
        optimizer.step()
        delta = param.detach() - before
        # update before mask: sign(0.9*[1,1] + 0.1*[-0.01,0.01]) = [+1, +1]
        # mask:   (update * grad > 0)   = [False, True] = [0, 1]
        # scale:  mask.mean()           = 0.5
        # update after mask/rescale     = [0, 2]
        # delta:  -lr * update          = [0, -2]
        self.assertTrue(torch.isfinite(param.detach()).all())
        self.assertAlmostEqual(delta[0].item(), 0.0, places=5)
        self.assertAlmostEqual(delta[1].item(), -2.0, places=5)

    def test_lion_drops_eps_from_optimizer_args_silently(self):
        # Lion has no eps; the factory should silently drop it rather than error.
        param = torch.nn.Parameter(torch.randn(2, 2))
        optimizer = create_optimizer(
            "lion",
            [param],
            learning_rate=1.0e-5,
            eps=1.0e-8,  # nonsensical for Lion, but common YAML passthrough
        )
        self.assertEqual(type(optimizer).__name__, "Lion")
        self.assertNotIn("eps", optimizer.param_groups[0])


if __name__ == "__main__":
    unittest.main()
