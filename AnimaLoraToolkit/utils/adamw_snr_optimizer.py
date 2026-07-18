"""AdamW-SNR —— AdamW + 可选的梯度信噪比门控（cautious 掩码 / SNR 锐化）。

## 出发点（本仓库实测，非论文转述）

AdamW 的更新 `m/sqrt(v)` 逐坐标的幅度**本身就是该坐标的梯度信噪比**：
设某坐标梯度 g ~ N(mu, sigma^2)，r = mu/sigma，则 |m|/sqrt(v) 单调于 r。
换言之 **AdamW ≡ sign(m) × SNR**。

对 Krea2 c12port 成功 run 的优化器状态做逐坐标测量（113.7M 坐标，step 800）：

    r 分位：  25% = 0（纯噪声）   50% = 0.094   90% = 0.66   95% = 0.93
    噪声坐标(r<0.15) 占 54% 的坐标数，但只占 6.84% 的更新能量
    信号坐标(r>0.5)  占 17% 的坐标数，占 66.0% 的更新能量

即 **AdamW 已经把绝大部分预算给了信号方向**。本模块提供的两个门控是在剩下
那 6.8% 上做文章，**上限有限**——引入时请如实标注预期收益，不要过度承诺。

## 两个门控

1. `cautious=True` —— C-AdamW（arXiv:2411.16085, ICLR 2025）。
   掩掉 `sign(update) != sign(g)` 的坐标并按保留比例重归一化。
   实测预测：信号能量占比 66.0% → 79.1%，噪声 6.84% → 2.96%。

2. `snr_power=p`（p>1）—— 本仓库设计的 SNR 锐化。
   更新改为 `sign(m) * SNR^p`，再按张量均值重归一化以保持 lr 语义不变。
   p=1 时**与 AdamW 逐元素恒等**。实测预测：p=1.5 → 信号 82.7%/噪声 1.82%；
   p=2.0 → 信号 92.2%/噪声 0.42%（p>=3 过于激进，会饿死大部分坐标，不建议）。

两个门控都是**逐元素**运算，不含任何谱平坦化算子，因此保持梯度的低秩谱结构
（这是 muon/muon_sf 在本仓库被结案弃用的原因，见 docs/optimizer-params.md §8
的警告框与 tools/lora_delta_forensics.py）。

## 默认行为中立

`cautious=False, snr_power=1.0`（默认）时与 `torch.optim.AdamW` 数学恒等，
单测 `TestDefaultsEquivalentToAdamW` 逐元素对拍断言此事。

## fp32 master

与 muon_optimizer / automagic_optimizer 一致：本仓库 LoRA 参数是 bf16，
更新先落 fp32 master，bf16 参数只是每步刷新的展示视图，防 ulp 冻结。
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional

import torch
from torch import Tensor
from torch.optim import Optimizer


class AdamWSNR(Optimizer):
    """AdamW + 可选 cautious 掩码 / SNR 锐化门控。

    Args:
        cautious: 开启 C-AdamW 掩码（arXiv:2411.16085）。
        snr_power: SNR 锐化指数。1.0 = 标准 AdamW（默认，行为中立）。
            建议范围 1.0~2.0；>=3 会饿死大部分坐标。
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        cautious: bool = False,
        snr_power: float = 1.0,
    ):
        if lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        if snr_power < 1.0:
            raise ValueError(
                f"snr_power={snr_power} < 1 会【放大】噪声坐标的相对步长，与本门控"
                f"的目的相反；如需削弱信号请直接降 lr。合法范围 >= 1.0（1.0=标准 AdamW）"
            )
        if snr_power > 4.0:
            raise ValueError(
                f"snr_power={snr_power} 过大：实测 p>=3 时 >99% 的更新能量集中到 "
                f"top-17% 坐标，其余坐标近乎冻结。建议 1.0~2.0"
            )
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            cautious=cautious, snr_power=snr_power,
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
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            cautious = group["cautious"]
            p_snr = group["snr_power"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad.detach().float()
                state = self.state[param]

                if "exp_avg" not in state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)
                    state["master"] = param.detach().clone().float()
                master = state.get("master")
                if master is None:   # 旧 checkpoint 恢复
                    master = state["master"] = param.detach().clone().float()

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                k = state["step"]

                exp_avg.lerp_(grad, 1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bc1 = 1.0 - beta1 ** k
                bc2 = 1.0 - beta2 ** k
                denom = (exp_avg_sq / bc2).sqrt_().add_(eps)
                update = (exp_avg / bc1) / denom        # ≡ AdamW 的逐坐标更新

                # ── SNR 锐化：|update| 本身就是该坐标的 SNR ────────────
                if p_snr != 1.0:
                    s = update.abs()
                    mean_s = s.mean()
                    sharp = s.pow(p_snr)
                    mean_sharp = sharp.mean()
                    # 按张量均值重归一化，保持 lr 语义与 p=1 可比
                    scale = mean_s / mean_sharp.clamp_min(1e-30)
                    update = torch.sign(update) * sharp * scale

                # ── cautious 掩码（arXiv:2411.16085）────────────────────
                if cautious:
                    mask = (update * grad > 0).to(update.dtype)
                    # 论文口径：按保留比例重归一化，保持期望步长
                    mask.mul_(mask.numel() / mask.sum().clamp_min(1.0))
                    update = update * mask

                if wd != 0.0:
                    master.mul_(1.0 - lr * wd)
                master.add_(update, alpha=-lr)
                param.copy_(master.to(dtype=param.dtype))

        return loss
