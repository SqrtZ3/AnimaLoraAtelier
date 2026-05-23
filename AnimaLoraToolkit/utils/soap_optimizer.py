"""
SOAP optimizer for AnimaLoraToolkit.

This file is adapted from the official SOAP reference implementation:
https://github.com/nikhilvyas/SOAP

MIT License
Copyright (c) 2024 Nikhil Vyas

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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

The implementation below keeps SOAP's Adam-in-Shampoo-eigenbasis update, but
uses fp32 optimizer state so bf16 LoRA/LoKr training remains numerically sane.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional

import torch
from torch import Tensor
from torch.optim import Optimizer


class SOAP(Optimizer):
    """Shampoo with Adam in the Shampoo eigenbasis.

    This optimizer is intended for matrix-heavy adapter parameters. If
    `precondition_1d` is false, vectors such as DoRA scales use AdamW-style
    updates without Shampoo preconditioning.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 3e-3,
        betas: tuple[float, float] = (0.95, 0.95),
        shampoo_beta: float = -1.0,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        precondition_frequency: int = 10,
        max_precond_dim: int = 10000,
        merge_dims: bool = False,
        precondition_1d: bool = False,
        normalize_grads: bool = False,
        data_format: str = "channels_first",
        correct_bias: bool = True,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 value: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 value: {betas[1]}")
        if shampoo_beta >= 1.0:
            raise ValueError(f"Invalid shampoo_beta value: {shampoo_beta}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if precondition_frequency < 1:
            raise ValueError("precondition_frequency must be >= 1")
        if max_precond_dim < 1:
            raise ValueError("max_precond_dim must be >= 1")
        if data_format not in {"channels_first", "channels_last"}:
            raise ValueError("data_format must be 'channels_first' or 'channels_last'")

        defaults = {
            "lr": lr,
            "betas": betas,
            "shampoo_beta": shampoo_beta,
            "eps": eps,
            "weight_decay": weight_decay,
            "precondition_frequency": int(precondition_frequency),
            "max_precond_dim": int(max_precond_dim),
            "merge_dims": bool(merge_dims),
            "precondition_1d": bool(precondition_1d),
            "normalize_grads": bool(normalize_grads),
            "correct_bias": bool(correct_bias),
        }
        super().__init__(params, defaults)
        self._data_format = data_format

    def _merge_dims(self, grad: Tensor, max_precond_dim: int) -> Tensor:
        if self._data_format == "channels_last" and grad.dim() == 4:
            grad = grad.permute(0, 3, 1, 2)

        new_shape: list[int] = []
        current = 1
        for size in grad.shape:
            candidate = current * int(size)
            if candidate > max_precond_dim:
                if current > 1:
                    new_shape.append(current)
                    current = int(size)
                else:
                    new_shape.append(int(size))
                    current = 1
            else:
                current = candidate
        if current > 1 or not new_shape:
            new_shape.append(current)
        return grad.reshape(new_shape)

    @staticmethod
    def _apply_matrix_along_dim(tensor: Tensor, matrix: Tensor, dim: int, transpose: bool) -> Tensor:
        mat = matrix.t() if transpose else matrix
        moved = tensor.movedim(dim, 0)
        flat = moved.reshape(moved.shape[0], -1)
        out = mat.to(device=flat.device, dtype=flat.dtype).matmul(flat)
        out = out.reshape(moved.shape)
        return out.movedim(0, dim)

    def _precondition_shape(self, grad: Tensor, merge_dims: bool, max_precond_dim: int) -> torch.Size:
        if not merge_dims:
            return grad.shape
        return self._merge_dims(grad, max_precond_dim).shape

    def _project(self, grad: Tensor, state: dict, merge_dims: bool, max_precond_dim: int) -> Tensor:
        original_shape = grad.shape
        if merge_dims:
            if self._data_format == "channels_last" and grad.dim() == 4:
                grad = grad.permute(0, 3, 1, 2)
            grad = self._merge_dims(grad, max_precond_dim)

        for dim, q in enumerate(state.get("Q", ())):
            if q is not None:
                grad = self._apply_matrix_along_dim(grad, q, dim, transpose=True)

        if merge_dims:
            if self._data_format == "channels_last" and len(original_shape) == 4:
                grad = grad.reshape(grad.new_empty(original_shape).permute(0, 3, 1, 2).shape)
                grad = grad.permute(0, 2, 3, 1)
            else:
                grad = grad.reshape(original_shape)
        return grad

    def _project_back(self, grad: Tensor, state: dict, merge_dims: bool, max_precond_dim: int) -> Tensor:
        original_shape = grad.shape
        if merge_dims:
            if self._data_format == "channels_last" and grad.dim() == 4:
                grad = grad.permute(0, 3, 1, 2)
            grad = self._merge_dims(grad, max_precond_dim)

        for dim, q in enumerate(state.get("Q", ())):
            if q is not None:
                grad = self._apply_matrix_along_dim(grad, q, dim, transpose=False)

        if merge_dims:
            if self._data_format == "channels_last" and len(original_shape) == 4:
                grad = grad.reshape(grad.new_empty(original_shape).permute(0, 3, 1, 2).shape)
                grad = grad.permute(0, 2, 3, 1)
            else:
                grad = grad.reshape(original_shape)
        return grad

    def _should_precondition(self, grad: Tensor, precondition_1d: bool, max_precond_dim: int) -> bool:
        if grad.dim() == 1:
            return bool(precondition_1d and grad.shape[0] <= max_precond_dim)
        return any(int(size) <= max_precond_dim for size in grad.shape)

    def _init_preconditioner(
        self,
        grad: Tensor,
        state: dict,
        shampoo_beta: float,
        max_precond_dim: int,
        precondition_1d: bool,
        merge_dims: bool,
        precondition_frequency: int,
    ) -> None:
        if not self._should_precondition(grad, precondition_1d, max_precond_dim):
            state["has_preconditioner"] = False
            return

        state["has_preconditioner"] = True
        state["GG"] = []
        precond_shape = self._precondition_shape(grad, merge_dims, max_precond_dim)
        for size in precond_shape:
            size_i = int(size)
            if size_i <= max_precond_dim:
                state["GG"].append(torch.zeros(size_i, size_i, device=grad.device, dtype=torch.float32))
            else:
                state["GG"].append(None)
        state["Q"] = None
        state["shampoo_beta"] = float(shampoo_beta)
        state["precondition_frequency"] = int(precondition_frequency)

    def _outer_for_dim(self, grad: Tensor, dim: int) -> Tensor:
        moved = grad.movedim(dim, 0).reshape(grad.shape[dim], -1)
        return moved.matmul(moved.t())

    def _orthogonal_matrix(self, matrices: list[Optional[Tensor]]) -> list[Optional[Tensor]]:
        bases: list[Optional[Tensor]] = []
        for matrix in matrices:
            if matrix is None:
                bases.append(None)
                continue
            eye = torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
            try:
                _, q = torch.linalg.eigh(matrix + 1e-30 * eye)
            except RuntimeError:
                _, q64 = torch.linalg.eigh(matrix.double() + 1e-30 * eye.double())
                q = q64.float()
            bases.append(torch.flip(q.float(), dims=[1]))
        return bases

    def _orthogonal_matrix_qr(
        self,
        state: dict,
        max_precond_dim: int,
        merge_dims: bool,
    ) -> list[Optional[Tensor]]:
        matrices = state["GG"]
        old_bases = state["Q"]
        exp_avg_sq = state["exp_avg_sq"]
        original_shape = exp_avg_sq.shape

        if merge_dims:
            if self._data_format == "channels_last" and exp_avg_sq.dim() == 4:
                permuted_shape = exp_avg_sq.permute(0, 3, 1, 2).shape
            else:
                permuted_shape = None
            exp_avg_sq = self._merge_dims(exp_avg_sq, max_precond_dim)
        else:
            permuted_shape = None

        new_bases: list[Optional[Tensor]] = []
        for dim, (matrix, old_q) in enumerate(zip(matrices, old_bases)):
            if matrix is None or old_q is None:
                new_bases.append(None)
                continue
            matrix = matrix.float()
            old_q = old_q.float()
            est_eig = torch.diag(old_q.t().matmul(matrix).matmul(old_q))
            sort_idx = torch.argsort(est_eig, descending=True)
            exp_avg_sq = exp_avg_sq.index_select(dim, sort_idx.to(exp_avg_sq.device))
            old_q = old_q[:, sort_idx]
            power_iter = matrix.matmul(old_q)
            q, _ = torch.linalg.qr(power_iter)
            new_bases.append(q.float())

        if merge_dims:
            if self._data_format == "channels_last" and len(original_shape) == 4 and permuted_shape is not None:
                exp_avg_sq = exp_avg_sq.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                exp_avg_sq = exp_avg_sq.reshape(original_shape)
        state["exp_avg_sq"] = exp_avg_sq
        return new_bases

    def _update_preconditioner(
        self,
        grad: Tensor,
        state: dict,
        max_precond_dim: int,
        merge_dims: bool,
        precondition_1d: bool,
    ) -> None:
        if not state.get("has_preconditioner", False):
            return

        exp_avg_was_projected = bool(state.get("exp_avg_projected", False))
        if exp_avg_was_projected and state.get("Q") is not None:
            state["exp_avg"] = self._project_back(state["exp_avg"], state, merge_dims, max_precond_dim)

        working_grad = grad
        if working_grad.dim() == 1:
            if precondition_1d and working_grad.shape[0] <= max_precond_dim:
                outer = working_grad.unsqueeze(1).matmul(working_grad.unsqueeze(0))
                state["GG"][0].lerp_(outer, 1.0 - state["shampoo_beta"])
        else:
            if merge_dims:
                working_grad = self._merge_dims(working_grad, max_precond_dim)
            for dim, size in enumerate(working_grad.shape):
                if int(size) <= max_precond_dim and state["GG"][dim] is not None:
                    state["GG"][dim].lerp_(self._outer_for_dim(working_grad, dim), 1.0 - state["shampoo_beta"])

        if state.get("Q") is None:
            state["Q"] = self._orthogonal_matrix(state["GG"])
        elif state["step"] > 0 and state["step"] % state["precondition_frequency"] == 0:
            state["Q"] = self._orthogonal_matrix_qr(state, max_precond_dim, merge_dims)

        state["exp_avg"] = self._project(state["exp_avg"], state, merge_dims, max_precond_dim)
        state["exp_avg_projected"] = True

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            shampoo_beta = group["shampoo_beta"] if group["shampoo_beta"] >= 0 else beta2
            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise RuntimeError("SOAP does not support sparse gradients")

                grad = param.grad.detach().float()
                state = self.state[param]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param.detach(), dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(param.detach(), dtype=torch.float32)
                    state["exp_avg_projected"] = False
                    self._init_preconditioner(
                        grad=grad,
                        state=state,
                        shampoo_beta=shampoo_beta,
                        max_precond_dim=group["max_precond_dim"],
                        precondition_1d=group["precondition_1d"],
                        merge_dims=group["merge_dims"],
                        precondition_frequency=group["precondition_frequency"],
                    )

                use_preconditioner = bool(state.get("has_preconditioner", False) and state.get("Q") is not None)
                if use_preconditioner:
                    grad_for_adam = self._project(
                        grad, state, merge_dims=group["merge_dims"], max_precond_dim=group["max_precond_dim"]
                    )
                else:
                    grad_for_adam = grad

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(grad_for_adam, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).add_(grad_for_adam.square(), alpha=1.0 - beta2)

                state["step"] += 1
                denom = exp_avg_sq.sqrt().add_(group["eps"])
                update = exp_avg / denom
                if use_preconditioner:
                    update = self._project_back(
                        update, state, merge_dims=group["merge_dims"], max_precond_dim=group["max_precond_dim"]
                    )
                if group["normalize_grads"]:
                    update = update / (update.pow(2).mean().sqrt() + 1e-30)

                step_size = group["lr"]
                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * (bias_correction2 ** 0.5) / bias_correction1

                if group["weight_decay"] > 0.0:
                    param.mul_(1.0 - group["lr"] * group["weight_decay"])
                param.add_(update.to(dtype=param.dtype), alpha=-step_size)

                self._update_preconditioner(
                    grad=grad,
                    state=state,
                    max_precond_dim=group["max_precond_dim"],
                    merge_dims=group["merge_dims"],
                    precondition_1d=group["precondition_1d"],
                )

        return loss
