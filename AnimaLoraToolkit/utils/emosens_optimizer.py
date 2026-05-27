"""
EmoSens optimizer adapter for AnimaLoraToolkit.

Adapted from:
    https://github.com/muooon/EmoSens

Apache License 2.0
Copyright attribution belongs to the EmoSens authors/contributors.

This adapter intentionally removes EmoSens' global Tensor.backward monkey patch.
The training loop should pass the aggregated optimizer-step loss explicitly via
`set_loss(...)` before calling `step()`. Optimizer state tensors are kept in
fp32 so bf16 LoRA/LoKr training does not accumulate moment statistics in bf16.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Optional

import torch
from torch import Tensor
from torch.optim import Optimizer


class EmoSens(Optimizer):
    """Loss-driven Adam-style optimizer with EmoSens' dynamic emoPulse LR.

    Args:
        params: parameter iterable or list of parameter group dicts.
        lr: user scope for the dynamic LR. For DiT LoRA, upstream recommends
            starting around 0.1; for full fine-tuning, around 0.01.
        eps: denominator epsilon.
        betas: first and second moment EMA coefficients.
        weight_decay: decoupled AdamW-style weight decay.
        stopcoef: loss threshold used by the advisory `should_stop` flag.
        use_shadow: optional upstream shadow mechanism. Usually keep false.
        notify: print READY TO STOP messages when convergence is detected.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.1,
        eps: float = 1e-8,
        betas: tuple[float, float] = (0.9, 0.995),
        weight_decay: float = 0.0,
        stopcoef: float = 0.04,
        use_shadow: bool = False,
        notify: bool = False,
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
        if stopcoef < 0.0:
            raise ValueError(f"Invalid stopcoef: {stopcoef}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self._init_lr = float(lr)
        self.notify = bool(notify)
        self.should_stop = False
        self.stopcoef = float(stopcoef)
        self.use_shadow = bool(use_shadow)
        self.emoScope = float(lr)

        self.base_scale = 1e-4
        self.max_lim = 3e-3
        self.min_lim = 1e-8
        self.dNR_hist = 1.0
        self.noise_est = 1.0
        self.d_est = 0.02
        self.c_est = 0.0
        self.stop_base = 0.0
        self._manual_loss = 0.0
        self._group_lr_scales = []
        for group_idx, group in enumerate(self.param_groups):
            group_lr = float(group.get("lr", self._init_lr))
            scale = group_lr / self._init_lr if self._init_lr > 0 else 1.0
            self._group_lr_scales.append(scale)

    def set_loss(self, loss: float | Tensor) -> None:
        """Set the scalar loss used by the next dynamic LR update."""
        if isinstance(loss, Tensor):
            loss = float(loss.detach().float().cpu())
        loss = float(loss)
        if math.isfinite(loss):
            self._manual_loss = loss

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict["emo_internal"] = {
            "emoScope": self.emoScope,
            "dNR_hist": self.dNR_hist,
            "noise_est": self.noise_est,
            "d_est": self.d_est,
            "c_est": self.c_est,
            "should_stop": self.should_stop,
            "stopcoef": self.stopcoef,
            "manual_loss": self._manual_loss,
            "stop_base": self.stop_base,
            "group_lr_scales": list(self._group_lr_scales),
        }
        return state_dict

    def load_state_dict(self, state_dict):
        emo_internal = state_dict.pop("emo_internal", None)
        if emo_internal:
            self.emoScope = emo_internal.get("emoScope", self._init_lr)
            self.dNR_hist = emo_internal.get("dNR_hist", 1.0)
            self.noise_est = emo_internal.get("noise_est", 1.0)
            self.d_est = emo_internal.get("d_est", 0.02)
            self.c_est = emo_internal.get("c_est", 0.0)
            self.should_stop = emo_internal.get("should_stop", False)
            self.stopcoef = emo_internal.get("stopcoef", self.stopcoef)
            self._manual_loss = emo_internal.get("manual_loss", 0.0)
            self.stop_base = emo_internal.get("stop_base", 0.0)
            self._group_lr_scales = list(
                emo_internal.get("group_lr_scales", self._group_lr_scales)
            )
        super().load_state_dict(state_dict)

    def _update_ema(self, loss_val: float):
        ema = self.state.setdefault("ema", {})
        ema["short"] = 0.3 * loss_val + 0.7 * ema.get("short", loss_val)
        ema["medium"] = 0.05 * loss_val + 0.95 * ema.get("medium", loss_val)
        ema["long"] = 0.01 * loss_val + 0.99 * ema.get("long", loss_val)
        return ema

    @staticmethod
    def _compute_scalar(ema):
        scale_base_l = max(ema["long"], 1e-5)
        scale_base_m = max(ema["medium"], 1e-5)
        diff_base = ema["long"] - ema["short"]
        diff_l = diff_base / scale_base_l
        diff_m = diff_base / scale_base_m
        if abs(diff_l) < 0.05:
            res_scalar = math.tanh(diff_l)
        elif abs(diff_m) * scale_base_m < abs(diff_l) * scale_base_l:
            res_scalar = math.tanh(diff_m)
        else:
            res_scalar = math.tanh(diff_l)
        return res_scalar, scale_base_m

    def _decide_ratio(self, scalar: float) -> float:
        if not self.use_shadow:
            return 0.0
        if abs(scalar) > 0.625:
            return 1.0 - abs(scalar)
        return 0.0

    def _next_pulse(self, loss_val: float) -> float:
        ema = self._update_ema(loss_val)
        scalar, scale_base_m = self._compute_scalar(ema)
        trust = math.copysign((1.0 - abs(scalar)), scalar)

        self.noise_est = 0.97 * self.noise_est + 0.03 * abs(scalar)
        self.d_est = 0.97 * self.d_est + 0.03 * abs(trust)
        self.c_est = 0.7 * self.c_est + 0.3 * scalar
        noise = max(self.noise_est, 1e-10)
        d = self.d_est

        noise_base = abs(scalar - trust) + 0.1
        d_base = abs(noise - d) + 0.1
        dnr_now_val = (d_base / noise_base) ** 2
        if dnr_now_val >= self.dNR_hist and trust >= 0.5:
            self.dNR_hist = min(dnr_now_val, self.dNR_hist * 1.50)
        elif -0.5 <= trust <= 0.5:
            self.dNR_hist = dnr_now_val * 0.80

        emo_chain = self.emoScope * max((100.0 ** self.c_est), 1e-3)
        emo_pulse = float(
            max(
                min(
                    self.dNR_hist * (emo_chain * self.base_scale),
                    self.emoScope * self.max_lim,
                ),
                self.min_lim,
            )
        )

        self.stop_base = self.d_est - self.noise_est
        if self.stop_base >= 0.3 and scale_base_m <= self.stopcoef:
            self.should_stop = True
            if self.notify:
                print("[EmoSens] READY TO STOP")
        else:
            self.should_stop = False

        return emo_pulse

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], Tensor]] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
            self.set_loss(loss)

        loss_val = float(self._manual_loss)
        emo_pulse = self._next_pulse(loss_val)

        for group_idx, group in enumerate(self.param_groups):
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            group_lr_scale = (
                float(self._group_lr_scales[group_idx])
                if group_idx < len(self._group_lr_scales) else 1.0
            )
            group_pulse = emo_pulse * group_lr_scale
            ratio = self._decide_ratio(self.c_est)

            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise RuntimeError("EmoSens does not support sparse gradients")

                grad = param.grad.detach().float()
                state = self.state[param]

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(param.detach(), dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(param.detach(), dtype=torch.float32)

                if self.use_shadow and "shadow" not in state:
                    state["shadow"] = param.detach().clone().float()

                if self.use_shadow:
                    trust = max(1.0 - abs(self.c_est), 0.0)
                    if ratio > 0:
                        blended = (
                            param.detach().float().mul(1.0 - ratio)
                            .add(state["shadow"], alpha=trust)
                        )
                        param.copy_(blended.to(dtype=param.dtype))
                    else:
                        state["shadow"].lerp_(param.detach().float(), 0.1 * trust)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                denom = exp_avg_sq.sqrt().add_(eps)
                if weight_decay != 0.0:
                    param.mul_(1.0 - weight_decay * group_pulse)
                update = exp_avg / denom
                param.add_(update.to(dtype=param.dtype), alpha=-group_pulse)

        for group_idx, group in enumerate(self.param_groups):
            group_lr_scale = (
                float(self._group_lr_scales[group_idx])
                if group_idx < len(self._group_lr_scales) else 1.0
            )
            group["lr"] = emo_pulse * group_lr_scale

        return loss
