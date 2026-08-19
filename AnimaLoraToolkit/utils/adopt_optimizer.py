"""
ADOPT optimizer for AnimaLoraToolkit.

Reference implementation:
    https://github.com/iShohei220/adopt
Paper:
    "ADOPT: Modified Adam Can Converge with Any β2 with the Optimal Rate"
    Shohei Taniguchi et al., NeurIPS 2024.
    https://arxiv.org/abs/2411.02853

MIT License
Copyright (c) 2024 Shohei Taniguchi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

------------------------------------------------------------------------------
Algorithm summary
------------------------------------------------------------------------------
ADOPT modifies Adam so that the second-moment EMA used to normalize the
current gradient is the *previous* step's value (v_{t-1}), and the update
to v happens *after* the parameter update. This breaks Adam's dependence
on a bounded-noise assumption — which is routinely violated in diffusion
models, VAEs, and multi-objective auxiliary losses. ADOPT therefore
converges at O(1/√T) for any β2 ∈ [0, 1).

Practical recommendation from the paper (Section 5): enable element-wise
clipping of the normalized gradient with bound c_t = step^(1/4). This
"ADOPT-clip" variant preserves the convergence guarantee while improving
numerical stability under heavy-tailed gradient noise. It is enabled by
default here (`use_clip=True`).

Implementation note: state tensors are kept in fp32 even when training
parameters are bf16, mirroring the convention used by this repo's SOAP
optimizer (utils/soap_optimizer.py) to avoid bf16 round-off accumulation.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Tuple

import torch
from torch import Tensor
from torch.optim import Optimizer


class ADOPT(Optimizer):
    """ADOPT optimizer (Taniguchi et al., NeurIPS 2024).

    Args:
        params: parameter iterable or list of parameter group dicts.
        lr: learning rate (default 1e-3). For diffusion LoRA with this
            project's aux-loss stack, 5e-5 ~ 1e-4 is a reasonable starting
            point — see config/train_my_adopt.yaml.
        betas: (β1, β2). β2 ≈ 0.9999 is recommended for long training
            since ADOPT removes Adam's β2 sensitivity.
        eps: clamp floor for sqrt(v_{t-1}) to avoid division by zero.
        weight_decay: L2 / decoupled WD coefficient.
        decoupled: if True, AdamW-style decoupled weight decay
            (θ ← θ * (1 - lr * wd)). If False, classical L2 (add wd*θ to grad).
        use_clip: enable ADOPT-clip stability variant (recommended).
        clip_exponent: exponent for the clip schedule c_t = step^clip_exponent.
            Paper default 0.25.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.9999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        decoupled: bool = True,
        use_clip: bool = True,
        clip_exponent: float = 0.25,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if clip_exponent <= 0.0:
            raise ValueError(f"clip_exponent must be > 0, got {clip_exponent}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            decoupled=bool(decoupled),
            use_clip=bool(use_clip),
            clip_exponent=float(clip_exponent),
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
            eps = group["eps"]
            decoupled = group["decoupled"]
            use_clip = group["use_clip"]
            clip_exponent = group["clip_exponent"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise RuntimeError("ADOPT does not support sparse gradients")

                grad = param.grad.detach().float()
                state = self.state[param]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param.detach(), dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(param.detach(), dtype=torch.float32)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # Weight decay (decoupled = AdamW-style, else classical L2)
                if wd != 0.0:
                    if decoupled:
                        param.mul_(1.0 - lr * wd)
                    else:
                        grad = grad.add(param.detach().float(), alpha=wd)

                state["step"] += 1
                step_t = state["step"]

                if step_t == 1:
                    # ADOPT initialization: v_1 = g_1^2, no param update yet.
                    exp_avg_sq.addcmul_(grad, grad)
                    continue

                # Normalize with the PREVIOUS step's v (not current).
                denom = torch.clamp(exp_avg_sq.sqrt(), min=eps)
                normed = grad / denom

                # ADOPT-clip stability: clamp element-wise to ±step^exponent.
                if use_clip:
                    clip = float(step_t) ** clip_exponent
                    normed.clamp_(-clip, clip)

                # Momentum on the normalized gradient.
                exp_avg.mul_(beta1).add_(normed, alpha=1.0 - beta1)

                # Parameter update.
                param.add_(exp_avg.to(dtype=param.dtype), alpha=-lr)

                # Update v_t AFTER the param update (uses current grad).
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

        return loss
