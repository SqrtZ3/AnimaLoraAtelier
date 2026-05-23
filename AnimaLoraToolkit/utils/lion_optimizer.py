"""
Lion (+ Cautious variant) for AnimaLoraToolkit.

Lion reference:
    "Symbolic Discovery of Optimization Algorithms"
    Chen et al., NeurIPS 2023.
    https://arxiv.org/abs/2302.06675
    Apache License 2.0, Google Research.

Cautious reference:
    "Cautious Optimizers: Improving Training with One Line of Code"
    Kaizhao Liang, Lizhang Chen, Bo Liu, Qiang Liu, 2024.
    https://arxiv.org/abs/2411.16085
    MIT License, https://github.com/kyleliang919/C-Optim

------------------------------------------------------------------------------
Algorithm
------------------------------------------------------------------------------
Lion update (per parameter):

    c_t = β1 * m_{t-1} + (1 - β1) * g_t
    update = sign(c_t)
    θ_t = θ_{t-1} - lr * (update + λ * θ_{t-1})   # decoupled WD, AdamW-style
    m_t = β2 * m_{t-1} + (1 - β2) * g_t           # separate slower EMA

Defaults: β1=0.9, β2=0.99. Lion paper recommends LR ~3–10× smaller than
AdamW and weight_decay ~3–10× larger, because sign-based updates have
constant magnitude regardless of gradient scale.

Cautious variant (cautious=True):
    Right before applying `update`, mask out coordinates whose sign
    disagrees with the current raw gradient direction:

        mask = (update * g_t > 0).float()
        mask = mask / (mask.mean() + 1e-8)   # rescale to preserve average step
        update = update * mask

    Theory (paper): preserves Lion's Hamiltonian convergence guarantee.
    Empirics: ~1.28× speedup on Llama-1B, and — relevant to this repo —
    naturally suppresses coordinates where main loss vs aux-loss gradients
    pull in opposite directions.

State tensors are kept in fp32 for bf16 training safety, matching the
convention in soap_optimizer.py and adopt_optimizer.py.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Tuple

import torch
from torch import Tensor
from torch.optim import Optimizer


class Lion(Optimizer):
    """Lion optimizer with optional Cautious modification.

    Args:
        params: parameter iterable or list of parameter group dicts.
        lr: learning rate. Recommended 1e-5 ~ 3e-5 for diffusion LoRA
            (about 1/3 of an AdamW LR you would otherwise pick).
        betas: (β1, β2). β1 controls the update-direction blend,
            β2 controls the momentum EMA decay.
        weight_decay: decoupled WD coefficient (AdamW-style).
            Lion typically needs 3–10× the AdamW value (try 0.05 ~ 0.1).
        cautious: if True, applies the Cautious mask (arxiv 2411.16085)
            — registers as the C-Lion variant.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        cautious: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            cautious=bool(cautious),
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            wd = group["weight_decay"]
            cautious = group["cautious"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients")

                grad = param.grad.detach().float()
                state = self.state[param]

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(param.detach(), dtype=torch.float32)

                exp_avg = state["exp_avg"]

                # Decoupled weight decay (AdamW-style).
                if wd != 0.0:
                    param.mul_(1.0 - lr * wd)

                # Update direction: sign of an interpolated momentum.
                update = exp_avg.mul(beta1).add_(grad, alpha=1.0 - beta1).sign_()

                # Cautious mask: drop coords where update sign disagrees with grad.
                if cautious:
                    align = (update * grad > 0).to(update.dtype)
                    scale = align.mean().clamp_(min=1e-8)
                    update.mul_(align).div_(scale)

                param.add_(update.to(dtype=param.dtype), alpha=-lr)

                # Update the actual momentum (slower EMA than the blend above).
                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)

        return loss
