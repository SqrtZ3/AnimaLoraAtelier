"""Automagic optimizer —— 逐元素自适应学习率（Adafactor 二阶矩 + 符号一致性 lr mask）。

移植自 ostris/ai-toolkit（`toolkit/optimizers/automagic.py`，Apache-2.0），
算法三段：

  1. **Adafactor 式分解二阶矩**：2D 参数只存行/列两个向量（`exp_avg_sq_row`
     `exp_avg_sq_col`），用 rank-1 外积近似完整的 v，再对 grad 做逐元素归一化。
     无一阶动量（这点与 AdamW/Lion 不同）。
  2. **RMS 裁剪**：`update /= max(1, rms(update)/clip_threshold)`，把每个张量的
     更新 RMS 压到 ≤ clip_threshold（默认 1.0），量级语义与 AdamW 的
     m/sqrt(v)（RMS≈1）对齐 —— 所以 lr 可以直接沿用 AdamW 的经验值。
  3. **逐元素 lr mask（本优化器的核心）**：每个权重一个独立 lr。本步更新符号
     与上步一致 → `+lr_bump`；翻转 → `-lr_bump`；钳在 [min_lr, max_lr]。
     直觉：梯度方向被数据稳定支持的权重自动加速；被噪声驱动、符号来回翻转的
     权重 lr 衰减到 min_lr，等于自动饿死噪声方向。

★ fp32 master（本仓库要求，非上游写法）：
  本仓库 LoRA 可训练参数是 **bf16** 创建的（见 trainer/checkpoint.py 的 dtype
  复原注释）。bf16 在 LoRA 参数量级(~6e-3)的 ulp ≈ 3e-5，而 Automagic 起始
  lr=1e-6 时每步更新 ≈1e-6，**比 ulp 小 30 倍会被四舍五入全部吞掉** —— 这正是
  muon_sf 踩过的坑（见 muon_optimizer.py 修复 #2）。上游用随机舍入
  (`copy_stochastic`) 规避，本仓库统一用 fp32 master：更新累积在
  `state["master"]`，bf16 参数只是每步刷新的展示视图。

显存账（每参数）：master fp32 4B + lr_mask fp32 4B + last_polarity bool 1B
  + 分解二阶矩（行/列向量，可忽略）≈ **9 B/param**，与 AdamW 的 8 B/param
  基本持平。**本优化器的卖点不是省显存**，是逐元素自适应 lr。
  lr_mask 必须 fp32：bf16 尾数仅 8 位，`1e-4 + 1e-6` 会直接舍回 1e-4，
  lr_bump 机制会被静默废掉。

参考:
  - 上游实现: github.com/ostris/ai-toolkit `toolkit/optimizers/automagic.py`
  - Adafactor 分解二阶矩: arXiv:1804.04235
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional

import torch
from torch import Tensor
from torch.optim import Optimizer


class Automagic(Optimizer):
    """逐元素自适应 lr 优化器。

    Args:
        lr: **起始** lr（每个权重的 lr 从这里出发，之后由 lr_mask 自行升降）。
            上游默认 1e-6；本仓库建议按 AdamW 已验证值的量级设 max_lr，见下。
        min_lr / max_lr: lr mask 的钳位区间。**max_lr 是真正决定训练强度的旋钮**
            —— 方向稳定的权重最终都会顶到 max_lr。
        lr_bump: 每步调整量。从 lr 爬到 max_lr 需要 (max_lr-lr)/lr_bump 步。
        beta2: 分解二阶矩的衰减。
        clip_threshold: 更新 RMS 上限。
        weight_decay: 解耦权重衰减，按逐元素 lr 缩放（与上游一致）。
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-6,
        min_lr: float = 1e-7,
        max_lr: float = 1e-3,
        lr_bump: float = 1e-6,
        beta2: float = 0.999,
        eps: float = 1e-30,
        clip_threshold: float = 1.0,
        weight_decay: float = 0.0,
    ):
        if not (0.0 < min_lr <= max_lr):
            raise ValueError(
                f"需满足 0 < min_lr <= max_lr，收到 min_lr={min_lr}, max_lr={max_lr}"
            )
        if not (min_lr <= lr <= max_lr):
            raise ValueError(
                f"起始 lr={lr} 必须落在 [min_lr, max_lr]=[{min_lr}, {max_lr}] 内。"
                f"Automagic 的 lr 是起点不是目标值——想调训练强度请改 max_lr。"
            )
        if lr_bump <= 0:
            raise ValueError(f"lr_bump 必须 > 0，收到 {lr_bump}")
        if not (0.0 <= beta2 < 1.0):
            raise ValueError(f"beta2 必须在 [0,1)，收到 {beta2}")
        self.min_lr = float(min_lr)
        self.max_lr = float(max_lr)
        self.lr_bump = float(lr_bump)
        defaults = dict(
            lr=lr, beta2=beta2, eps=eps,
            clip_threshold=clip_threshold, weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    # ── 供 trainer 日志用：Automagic 自己管 lr，group["lr"] 只是起始值 ──────
    def get_avg_learning_rate(self) -> float:
        total, n = 0.0, 0
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state.get(p, {})
                if "lr_mask" in st:
                    total += float(st["lr_mask"].mean())
                    n += 1
        return total / n if n else float(self.param_groups[0]["lr"])

    def get_learning_rates(self) -> list[float]:
        return [self.get_avg_learning_rate()]

    @staticmethod
    def _rms(t: Tensor) -> Tensor:
        return t.norm(2) / (t.numel() ** 0.5)

    def _init_state(self, p: Tensor, group: dict) -> None:
        state = self.state[p]
        state["step"] = 0
        state["master"] = p.detach().clone().float()
        # lr_mask 必须 fp32：bf16 下 lr + lr_bump 会被尾数吃掉（见模块 docstring）
        state["lr_mask"] = torch.full_like(
            p, float(group["lr"]), dtype=torch.float32
        )
        state["last_polarity"] = torch.zeros_like(p, dtype=torch.bool)
        if p.ndim >= 2:
            state["exp_avg_sq_row"] = torch.zeros(
                p.shape[:-1], device=p.device, dtype=torch.float32
            )
            state["exp_avg_sq_col"] = torch.zeros(
                p.shape[:-2] + p.shape[-1:], device=p.device, dtype=torch.float32
            )
        else:
            state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta2 = group["beta2"]
            eps = group["eps"]
            clip_threshold = group["clip_threshold"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.detach().float()
                if grad.is_sparse:
                    raise RuntimeError("Automagic 不支持稀疏梯度")

                state = self.state[p]
                if "lr_mask" not in state:
                    self._init_state(p, group)
                # 旧 checkpoint 恢复：master 缺失时按当前参数惰性重建
                master = state.get("master")
                if master is None:
                    master = state["master"] = p.detach().clone().float()

                # ── 1. Adafactor 分解二阶矩 → 逐元素归一化 ──────────────
                sq = grad * grad + eps
                if p.ndim >= 2:
                    row = state["exp_avg_sq_row"]
                    col = state["exp_avg_sq_col"]
                    row.mul_(beta2).add_(sq.mean(dim=-1), alpha=1.0 - beta2)
                    col.mul_(beta2).add_(sq.mean(dim=-2), alpha=1.0 - beta2)
                    r_factor = (
                        row / row.mean(dim=-1, keepdim=True)
                    ).rsqrt_().unsqueeze(-1)
                    c_factor = col.unsqueeze(-2).rsqrt()
                    update = r_factor * c_factor * grad
                else:
                    v = state["exp_avg_sq"]
                    v.mul_(beta2).add_(sq, alpha=1.0 - beta2)
                    update = v.rsqrt() * grad

                # ── 2. RMS 裁剪（量级对齐 AdamW 的 m/sqrt(v)）───────────
                update.div_((self._rms(update) / clip_threshold).clamp_(min=1.0))

                # ── 3. 逐元素 lr mask：符号一致 +bump / 翻转 -bump ───────
                # 注意：last_polarity 初始为全 False，故第 1 步所有正更新都算
                # "翻转"各减一次 bump（与上游一致，起始 lr 处影响可忽略）。
                current_polarity = update > 0
                agree = torch.where(
                    state["last_polarity"] == current_polarity, 1.0, -1.0
                )
                state["last_polarity"] = current_polarity
                new_lr = torch.clamp(
                    state["lr_mask"] + agree * self.lr_bump,
                    min=self.min_lr, max=self.max_lr,
                )
                state["lr_mask"] = new_lr

                # ── 4. 落到 fp32 master，bf16 参数只是展示视图 ───────────
                if wd != 0.0:
                    master.mul_(1.0 - wd * new_lr)
                master.add_(-(update * new_lr))
                p.copy_(master.to(dtype=p.dtype))
                state["step"] += 1

        return loss
