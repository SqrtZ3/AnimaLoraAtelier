"""LeapAlign 两步跳跃自蒸馏（去奖励模型版）—— flow-matching SFT 主目标增强。

移植自朋友 PR（saltysalrua/AnimaLoraStudio#276，`runtime/training/leap.py`），源头是
LeapAlign 论文（arXiv 2604.15311v2）的 two-step leap trajectory，但**去掉奖励模型**：把
"最大化 reward"换成"两步跳跃预测的 x̂0 逼近数据集真实 x0"的自蒸馏。

动机：标准 flow-matching 的速度 MSE 会回归条件均值 → 发灰 / "平均风格"。leap 用两步轨迹
把预测钉到**这一张真图**的 x0（而非均值），是打这个根因（朋友实测推理图更"原图风格"）。

约定（与 trainer/sampling 一致）：t=0 数据端、t=1 噪声端，`x_t=(1-t)·x0+t·noise`，
velocity `v=noise-x0`。一步跳跃（a→b，a>b）：`x̂_b = x_a - (a-b)·v_θ(x_a, a)`。
（自检：v 完美时 x̂_j 恰为真 x_j、x̂_0 恰为真 x0 → loss=0，见 tests/test_leap.py。）

本模块只做纯张量数学（两步前向 + latent connector + 嵌套梯度折扣 + 自蒸馏 MSE），与
模型 / TREAD / grad_checkpoint / T-LoRA **解耦**：`forward_fn` 由调用方注入（在那里关
TREAD、设 T-LoRA current_t、按需 checkpoint）。**一阶反向**（无 create_graph）→ 与
grad_checkpoint 兼容。CPU 可单测。
"""

from __future__ import annotations

import torch


def sample_two_timesteps(
    bs: int,
    device,
    *,
    min_gap: float = 0.1,
    dtype: torch.dtype = torch.float32,
    generator: "torch.Generator | None" = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """per-sample 采两个时刻 (k, j)，保证 k > j 且间隔 ≥ min_gap，均 ∈ (0,1)。

    k 偏噪声端（大 t），j 偏数据端（小 t）。间隔不足时各取一半缺口往两端撑开，再 clamp
    回开区间。用训练 RNG（leap 是训练步的一部分，不需要像 DPO 采样那样隔离）。
    """
    a = torch.rand(bs, device=device, dtype=dtype, generator=generator)
    b = torch.rand(bs, device=device, dtype=dtype, generator=generator)
    k = torch.maximum(a, b)
    j = torch.minimum(a, b)

    deficit = (float(min_gap) - (k - j)).clamp(min=0.0) * 0.5
    k = k + deficit
    j = j - deficit

    eps = 1e-3
    k = k.clamp(min=eps + float(min_gap), max=1.0 - eps)
    j = torch.minimum(j, k - eps).clamp(min=eps)
    return k, j


def leap_training_step(
    forward_fn,
    x0: torch.Tensor,
    noise: torch.Tensor,
    cross: torch.Tensor,
    pad_mask,
    t_k: torch.Tensor,
    t_j: torch.Tensor,
    *,
    nested_grad_coe: float = 0.3,
    traj_sim_weighting: bool = False,
    traj_sim_min: float = 0.1,
) -> torch.Tensor:
    """两步跳跃自蒸馏，返回 per-sample loss `(B,)`（未 reduction、未乘外部样本权重）。

    Args:
        forward_fn — `forward_fn(x, t_col, cross, pad_mask) -> velocity v`。调用方负责
                     关 TREAD（轨迹一致性）、设 T-LoRA current_t、按需 grad_checkpoint。
        x0         — 真实 latent `(B,C,T,H,W)`（赢家=真图，直接加噪，无需 rollout）。
        noise      — 噪声 x1，与 x0 同 shape（复用训练步的 noise）。
        cross/pad_mask — 文本条件 / padding mask。
        t_k, t_j   — per-sample 时刻 `(B,)`，t_k > t_j。
        nested_grad_coe — 嵌套梯度折扣 α（论文 Eq 9）：0=砍掉第二跳对 x_j 的嵌套梯度，1=不折扣。
        traj_sim_weighting / traj_sim_min — 轨迹相似度加权（论文 Eq 12，默认关）。
    """
    k = t_k.view(-1, *([1] * (x0.ndim - 1)))
    j = t_j.view(-1, *([1] * (x0.ndim - 1)))

    # 真实带噪 latent（数据集本有 x0 → 无需 online rollout）
    x_k = (1.0 - k) * x0 + k * noise
    x_j_real = (1.0 - j) * x0 + j * noise

    # ── 第一跳（带梯度）：x_k --v_k--> x̂_{j|k} ──
    v_k = forward_fn(x_k, t_k.view(-1, 1), cross, pad_mask)
    x_hat_j = x_k - (k - j) * v_k

    # ── latent connector（论文 Eq 6）：前向数值=真值 x_j_real，反向梯度流回 x̂_j → v_k ──
    x_j = x_hat_j + (x_j_real - x_hat_j).detach()

    # ── 嵌套梯度折扣（论文 Eq 9）：把第二跳经 x_j_in 回流到第一跳的梯度缩到 α 倍 ──
    a = float(nested_grad_coe)
    if a <= 0.0:
        x_j_in = x_j.detach()
    elif a >= 1.0:
        x_j_in = x_j
    else:
        x_j_in = a * x_j + (1.0 - a) * x_j.detach()

    # ── 第二跳（带梯度）：x_j --v_j--> x̂_{0|j} ──（x̂0 用 connector 值 x_j，非折扣后的 x_j_in）
    v_j = forward_fn(x_j_in, t_j.view(-1, 1), cross, pad_mask)
    x_hat_0 = x_j - j * v_j

    # ── 自蒸馏 loss：两步跳跃预测的 x̂0 逼近真实 x0（取代奖励信号）──
    loss_per_sample = (x_hat_0.float() - x0.float()).pow(2).mean(
        dim=tuple(range(1, x0.ndim))
    )

    # ── 轨迹相似度加权（论文 Eq 12）：跳跃越贴近真实路径权重越高 ──
    if traj_sim_weighting:
        with torch.no_grad():
            d_j = (x_j_real.float() - x_hat_j.float()).abs().mean(
                dim=tuple(range(1, x0.ndim))
            ).clamp(min=traj_sim_min)
            d_0 = (x0.float() - x_hat_0.float()).abs().mean(
                dim=tuple(range(1, x0.ndim))
            ).clamp(min=traj_sim_min)
            w_sim = 1.0 / (d_j + d_0)
        loss_per_sample = loss_per_sample * w_sim

    return loss_per_sample
