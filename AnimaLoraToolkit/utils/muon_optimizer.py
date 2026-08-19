"""Muon & Muon-ScheduleFree optimizer.

Muon (MomentUm Orthogonalized by Newton-Schulz, Keller Jordan 2024):
  2D 参数走 SGD-momentum + Newton-Schulz 正交化（近似 SVD 的 UV^T），给
  出类似 Shampoo 的矩阵预条件效果，但 *不需要* GG/Q 矩阵 -- 优化器状态
  仅 1 个 momentum buffer（同参数大小），而非 SOAP 的 6×（exp_avg +
  exp_avg_sq + GG[0] + GG[1] + Q[0] + Q[1]）。

  Newton-Schulz 迭代原生在 bf16 下稳定（与 Shampoo 的 coupled-Newton
  不同，后者必须 fp32），因此不需要 SOAP 那套 _restore_fp32_state /
  _fp32_tree 的 bf16 补丁。

  1D 参数（bias、DoRA scale）走标准 AdamW fallback。

MuonScheduleFree:
  Schedule-Free 轨迹 + Newton-Schulz 正交化。保留 z base-sequence + 2D 走
  NS、1D 走 AdamW（second-moment only）。train()/eval() swap 遵循
  SOAPScheduleFree 的同一套 y/x 模式。

  ★ 2D 路径保留一个内层动量 buffer 作为 NS 的输入（momentum，默认 0.95，
  设 0 关闭）。SF 的 Polyak 平均替代的是 lr *schedule*，发生在权重轨迹上，
  替代不了 NS 之前的梯度平滑：极分解把所有奇异值拉到 1，喂裸梯度等于把
  batch 噪声也"归一化放大"。Keller 原版 Muon 的 NS 输入就是动量 buffer；
  ScheduleFree+ (arXiv:2605.19095) 实证把内层动量加回 SF 可修复大 batch
  发散；SF 官方 wrapper 文档也允许内外动量并存。

★ 2026-07-11 两处修复（krea2 LoRA 实测"完全不拟合"的根因，实验复现：
  同 lr=1e-4 下 AdamW 对 LoRA down 矩阵的位移是 muon_sf 的 165 倍）：
  1. RMS 缩放换成 Moonlight 口径 0.2·sqrt(max(rows,cols))（rms_scale，
     默认 "moonlight"）：NS 输出的条目 RMS = sqrt(min/(rows·cols))，乘该
     系数后恒为 0.2 —— 任意形状下更新量级与 AdamW lr 可比（arXiv:2502.16982）。
     旧的 Keller 口径 sqrt(max(1, rows/cols)) 只放大高瘦矩阵，LoRA 的矮宽
     down (r=24, in=6144) 被系统性缩小 ~16 倍（rms_scale="keller" 保留对照）。
  2. fp32 master：更新先落在 fp32 状态（Muon: state["master"]；SF: state["y"]），
     每步 param.copy_(master.to(bf16)) 只是展示视图。此前更新直接写 bf16
     参数，每步增量（缩放 bug 叠加 SF 的 ~0.1×lr 阻尼后 ~1e-7）远低于
     bf16 在 LoRA 参数量级(~6e-3)的 ulp(~3e-5)，78~93% 条目被四舍五入
     永久冻结。开销：每参数 +1 份 fp32（LoRA 级别可忽略）。

参考:
  - 原始 repo: github.com/KellerJordan/Muon
  - Kimi Moonlight scaling: arXiv:2502.16982
  - Fantastic Pretraining Optimizers: arXiv:2509.02046
  - Schedule-Free: arXiv:2405.15682
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Optional

import torch
from torch.optim import Optimizer
from torch import Tensor


# =============================================================================
# Newton-Schulz 核心迭代
# =============================================================================

@torch.no_grad()
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5) -> Tensor:
    """5-step Newton-Schulz 多项式迭代，近似计算 G 的正交极因子 (UV^T)。

    在 bf16 下数值稳定（显式设计目标）。返回与 G 同形状的正交化矩阵。

    系数 (3.4445, -4.7750, 2.0315) 拟合了 P(G) = 3X - 4X^3 + ... 的
    quintic Newton-Schulz 迭代，5 步内对 ||X|| <= 1 的矩阵收敛到
    正交极因子。
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    # 归一化到谱范数 <= 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X.float()


def _apply_rms_scale(update: Tensor, mode: str) -> Tensor:
    """NS 正交化输出的 update-RMS 对齐缩放。

    NS 输出条目 RMS = sqrt(min(rows,cols)/(rows·cols))。
    - "moonlight"（默认）：×0.2·sqrt(max(rows,cols)) → 条目 RMS 恒为 0.2，
      任意形状下与 AdamW 的归一化更新同量级（Moonlight arXiv:2502.16982）。
    - "keller"：×sqrt(max(1, rows/cols))（Keller Jordan 原版），只放大高瘦
      矩阵；对 LoRA 矮宽 down (r, in) 会缩小 ~sqrt(in/r) 倍，保留仅作对照。
    """
    rows, cols = update.shape
    if mode == "keller":
        return update * max(1.0, rows / cols) ** 0.5
    return update * (0.2 * math.sqrt(max(rows, cols)))


# =============================================================================
# Muon optimizer (momentum + Newton-Schulz)
# =============================================================================

class Muon(Optimizer):
    """Muon: SGD-momentum + Newton-Schulz 正交化 for 2D params, AdamW for 1D.

  Memory profile (per 2D param of shape (A, B)):
    SOAP:  exp_avg + exp_avg_sq + GG[A,A] + GG[B,B] + Q[A,A] + Q[B,B]
           = 2*A*B + 2*(A^2+B^2)  (bytes: fp32)
    Muon:  momentum_buffer = A*B       (bytes: fp32)
    -> Muon uses 6x less memory than SOAP for large dims.

  Hyperparameters:
    lr:           base learning rate (default 0.02, but with Kimi-style
                  update-RMS scaling this behaves like AdamW lr).
    momentum:     SGD momentum (default 0.95). "Usually fine" (original repo).
    nesterov:     Nesterov momentum (default True). Works better in all tests.
    ns_steps:     Newton-Schulz iterations (default 5). 10 = more accurate
                  but no better performance.
    weight_decay: decoupled weight decay (default 0).
    eps:          AdamW eps for 1D params (default 1e-8).
    betas:        (beta1, beta2) AdamW betas for 1D params only.
    correct_bias: AdamW bias correction for 1D params (default True).
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        eps: float = 1e-8,
        betas: tuple[float, float] = (0.9, 0.999),
        correct_bias: bool = True,
        rms_scale: str = "moonlight",
    ):
        if lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if rms_scale not in ("moonlight", "keller"):
            raise ValueError(f"Invalid rms_scale: {rms_scale!r} (moonlight | keller)")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            eps=eps,
            betas=betas,
            correct_bias=correct_bias,
            rms_scale=rms_scale,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]
            eps = group["eps"]
            beta1, beta2 = group["betas"]
            correct_bias = group["correct_bias"]
            rms_scale = group.get("rms_scale", "moonlight")

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad.detach()
                state = self.state[param]
                # fp32 master：更新累积在 fp32，bf16 参数只是每步刷新的展示视图
                # （旧 checkpoint 恢复的 state 无 master → 从当前参数惰性重建）。
                master = state.get("master")
                if master is None:
                    master = state["master"] = param.detach().clone().float()

                if param.ndim >= 2:
                    # ── 2D: Muon path (momentum + Newton-Schulz) ──────────
                    if "momentum_buffer" not in state:
                        state["step"] = 0
                        state["momentum_buffer"] = torch.zeros_like(
                            param, dtype=torch.float32
                        )

                    buf = state["momentum_buffer"]
                    buf.lerp_(grad.float(), 1.0 - momentum)

                    if nesterov:
                        update = grad.float().lerp_(buf, momentum)
                    else:
                        update = buf.clone()

                    # Newton-Schulz orthogonalization (runs in bf16 internally)
                    orig_shape = update.shape
                    if update.ndim > 2:
                        update = update.view(update.shape[0], -1)
                    update_ns = zeropower_via_newtonschulz5(update, steps=ns_steps)

                    # Update RMS scaling（moonlight：任意形状对齐 AdamW 量级）
                    update_ns = _apply_rms_scale(update_ns, rms_scale)

                    if wd != 0:
                        update_ns = update_ns.add(master.view(update_ns.shape), alpha=wd)

                    master.add_(update_ns.view(orig_shape), alpha=-lr)
                    param.copy_(master.to(dtype=param.dtype))
                    state["step"] += 1

                else:
                    # ── 1D: AdamW fallback (bias, DoRA scale, etc.) ─────────
                    if "exp_avg" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(param, dtype=torch.float32)
                        state["exp_avg_sq"] = torch.zeros_like(param, dtype=torch.float32)

                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    g = grad.float()

                    exp_avg.lerp_(g, 1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                    bc1 = 1.0 - beta1 ** (state["step"] + 1) if correct_bias else 1.0
                    bc2 = 1.0 - beta2 ** (state["step"] + 1) if correct_bias else 1.0
                    step_size = lr * (bc2 ** 0.5) / bc1 if correct_bias else lr

                    update = exp_avg / (exp_avg_sq.sqrt() + eps)
                    if wd != 0:
                        update = update.add(master, alpha=wd)

                    master.add_(update, alpha=-step_size)
                    param.copy_(master.to(dtype=param.dtype))
                    state["step"] += 1

        return loss


# =============================================================================
# MuonScheduleFree (Schedule-Free + Newton-Schulz)
# =============================================================================

class MuonScheduleFree(Optimizer):
    """Schedule-Free Muon: Polyak-Ruppert averaging + Newton-Schulz.

    2D params get NS-preconditioned momentum applied to z; 1D params get
    AdamW-normalized gradient applied to z.

    The parameter tensor holds the gradient-evaluation point ``y`` while in
    train mode; call :meth:`eval` before sampling/checkpointing to swap to
    the averaged iterate ``x``, and :meth:`train` to swap back.

    Schedule-Free specific args:
        weight_lr_power: power on lr in the Polyak averaging weight (default 2.0).
        r: power on step index (default 0.0 = uniform average).
        warmup_steps: linear lr warmup (default 0).
        momentum: inner momentum for the NS input on 2D params (default 0.95,
            same as Muon; 0 disables and feeds the raw gradient to NS — not
            recommended, see module docstring). Nesterov-style blend, matching
            the plain :class:`Muon` implementation.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.02,
        betas: tuple[float, float] = (0.9, 0.95),
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        eps: float = 1e-8,
        weight_lr_power: float = 2.0,
        r: float = 0.0,
        warmup_steps: int = 0,
        correct_bias: bool = True,
        momentum: float = 0.95,
        rms_scale: str = "moonlight",
    ):
        if lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f"Invalid momentum: {momentum}")
        if rms_scale not in ("moonlight", "keller"):
            raise ValueError(f"Invalid rms_scale: {rms_scale!r} (moonlight | keller)")
        defaults = dict(
            lr=lr,
            betas=betas,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            eps=eps,
            weight_lr_power=weight_lr_power,
            r=r,
            warmup_steps=warmup_steps,
            correct_bias=correct_bias,
            momentum=momentum,
            rms_scale=rms_scale,
            k=0,
            weight_sum=0.0,
            lr_max=0.0,
            train_mode=True,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def train(self) -> None:
        """Swap parameter from eval point x back to gradient point y.

        y 以 fp32 master（state["y"]）为准直接写回；旧 checkpoint（无 y master）
        退回逆插值恢复并就地建 master。
        """
        for group in self.param_groups:
            beta1 = group["betas"][0]
            if not group.get("train_mode", True):
                for param in group["params"]:
                    st = self.state.get(param, {})
                    z = st.get("z")
                    if z is None:
                        continue
                    y = st.get("y")
                    if y is None:
                        y = param.detach().float()
                        y.lerp_(z, weight=1.0 - beta1)
                        st["y"] = y
                    param.copy_(y.to(dtype=param.dtype))
                group["train_mode"] = True

    @torch.no_grad()
    def eval(self) -> None:
        """Swap parameter to the Polyak-averaged iterate x (for sampling/saving).

        x 从 fp32 master (y, z) 现算，y master 本身不动 —— train() 时无损换回。
        """
        for group in self.param_groups:
            beta1 = group["betas"][0]
            if group.get("train_mode", True):
                for param in group["params"]:
                    st = self.state.get(param, {})
                    z = st.get("z")
                    if z is None:
                        continue
                    y = st.get("y")
                    if y is None:
                        y = st["y"] = param.detach().clone().float()
                    x = y.lerp(z, weight=1.0 - 1.0 / beta1)
                    param.copy_(x.to(dtype=param.dtype))
                group["train_mode"] = False

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if not group.get("train_mode", True):
                raise RuntimeError(
                    "MuonScheduleFree.step() called in eval mode; call optimizer.train() first."
                )
            beta1, beta2 = group["betas"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]
            decay = group["weight_decay"]
            lr = group["lr"]
            warmup_steps = group["warmup_steps"]
            momentum = group.get("momentum", 0.95)

            rms_scale = group.get("rms_scale", "moonlight")

            k = group["k"]
            sched = (k + 1) / warmup_steps if (warmup_steps > 0 and k < warmup_steps) else 1.0
            bias_correction2 = (1.0 - beta2 ** (k + 1)) if group["correct_bias"] else 1.0
            lr_eff = lr * sched * (bias_correction2 ** 0.5)

            lr_max = group["lr_max"] = max(lr_eff, group["lr_max"])
            weight = ((k + 1) ** group["r"]) * (lr_max ** group["weight_lr_power"])
            weight_sum = group["weight_sum"] = group["weight_sum"] + weight
            ckp1 = weight / weight_sum if weight_sum > 0 else 0.0
            adaptive_y_lr = lr_eff * (beta1 * (1.0 - ckp1) - 1.0)

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad.detach().float()
                state = self.state[param]

                if "z" not in state:
                    state["step"] = 0
                    state["z"] = param.detach().clone().float()
                z = state["z"]
                # fp32 master y：SF 的每步 y 增量自带 ~(1-beta1)·lr 阻尼，直接写
                # bf16 参数会被 ulp 吞掉（见模块 docstring 修复 #2）。旧 state 恢复
                # 时惰性重建（train_mode 下参数即 y 的 bf16 舍入，可接受）。
                y = state.get("y")
                if y is None:
                    y = state["y"] = param.detach().clone().float()

                if param.ndim >= 2:
                    # ── 2D: Newton-Schulz path ─────────────────────────────
                    if grad.ndim > 2:
                        grad_flat = grad.view(grad.shape[0], -1)
                    else:
                        grad_flat = grad

                    # 内层动量平滑 NS 输入（与 Muon 同款 Nesterov blend）。
                    # 必须在 NS 之前：极分解会把噪声的奇异值也拉到 1，
                    # 喂裸梯度 = 放大 batch 噪声（见模块 docstring 引证）。
                    if momentum > 0.0:
                        buf = state.get("momentum_buffer")
                        if buf is None:
                            buf = state["momentum_buffer"] = torch.zeros_like(
                                grad_flat, dtype=torch.float32
                            )
                        buf.lerp_(grad_flat, 1.0 - momentum)
                        ns_input = grad_flat.lerp(buf, momentum)
                    else:
                        ns_input = grad_flat

                    update = zeropower_via_newtonschulz5(ns_input, steps=ns_steps)

                    # Update RMS scaling（moonlight：任意形状对齐 AdamW 量级）
                    update = _apply_rms_scale(update, rms_scale)
                    update = update.view(param.shape)

                    if decay != 0.0:
                        update = update.add(y, alpha=decay)

                    # Schedule-Free y/z update — 全程 fp32 master，bf16 仅视图
                    y.lerp_(z, weight=ckp1)
                    y.add_(update, alpha=adaptive_y_lr)
                    param.copy_(y.to(dtype=param.dtype))
                    z.sub_(update, alpha=lr_eff)

                else:
                    # ── 1D: AdamW-SF path (second moment only) ────────────
                    if "exp_avg_sq" not in state:
                        state["exp_avg_sq"] = torch.zeros_like(param, dtype=torch.float32)

                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    update = grad / (exp_avg_sq.sqrt() + eps)

                    if decay != 0.0:
                        update = update.add(y, alpha=decay)

                    y.lerp_(z, weight=ckp1)
                    y.add_(update, alpha=adaptive_y_lr)
                    param.copy_(y.to(dtype=param.dtype))
                    z.sub_(update, alpha=lr_eff)

                state["step"] += 1

            group["k"] = k + 1

        return loss
