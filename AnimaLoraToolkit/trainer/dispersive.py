"""Dispersive Loss —— 中间隐表征空间的"无正样本对排斥"正则
（arXiv 2506.09027《Diffuse and Disperse: Image Generation with Representation
Regularization》, Wang & He 2025）。

机制：把一个 batch 内某个中间 transformer block 的隐表征 z_i 互相**推开**，对抗
"表征坍缩 / 均值化"。它是"没有正样本对的对比损失"——只有排斥项、没有需要多视图增广
的对齐项，因此**不干扰 FM 回归主目标、零额外可训练参数、不需要外部模型、不解码出图**。

总目标（论文 Eq.）：
    L = L_Diff + λ · L_Disp

最佳变体 InfoNCE-L2（论文 Table 实测最优）：
    L_Disp = log E_{i,j}[ exp(−‖z_i − z_j‖²₂ / τ) ]            # 含对角(i==j)，论文说可保留
最小化 L_Disp → 增大成对距离 → 表征在隐空间铺开。
cosine 变体：D_cosine = −cos(z_i,z_j)，exp(−D/τ)=exp(cos/τ)，最小化 → 降低成对余弦相似度。

★与原论文的两处工程取舍（都有依据，已在此说明）：
  1. **数值稳定**：论文 Algorithm 1 直接 `log mean exp(−D/τ)`；本实现用 logsumexp 等价式
     `logsumexp(−D/τ) − log(N_pair)`，避免大距离下 exp 下溢成 0 → log(0)=−inf。等价、更稳。
  2. **分辨率鲁棒（normalize_by_dim，l2 默认 True）**：论文在**定分辨率** ImageNet 上训练，
     用展平后**原始**平方距离（dim D=H·W·C 固定）。本仓库是 ARB 变分辨率分桶，跨 batch 的
     展平维度 D 会变 → 原始平方距离量级随之变 → τ 不可跨桶迁移。故 l2 距离除以特征维度
     （= 逐元素均方距离），让量级与分辨率解耦、τ 可迁移。设 normalize_by_dim=False 可还原
     论文原式。cosine 变体天然尺度/维度无关，不受此影响。

本模块只放**纯张量数学**（CPU 可单测）：表征降维 + 成对排斥损失。它与"如何从模型抽
block-K 激活"解耦——调用方注入 z（接线复用 trainer/ncp.perceptual_features 的截断前向，
不冻结 adapter 以让梯度回流 LoRA）。镜像 trainer/leap.py / ncp.py 的注入式风格。
"""

from __future__ import annotations

import math

import torch


def reduce_representation(z: torch.Tensor, pool: str = "flatten") -> torch.Tensor:
    """把任意形状的 block 激活 (B, ...) 规约成每样本一个向量 (B, D_feat)。

    - "flatten"：展平除 batch 外的全部维度（论文做法：D = H·W·C，保留空间信息）。
    - "mean"：对中间维度求均值、保留首维(B)与末维(channel) → (B, C)。把"全图平均表征"
      推开，比 flatten 粗但维度恒定。z 已是 2D 时两种模式都是 no-op。
    """
    if z.dim() <= 1:
        raise ValueError(f"dispersive 表征至少要 2 维 (B, ...)，收到 shape={tuple(z.shape)}")
    if z.dim() == 2:
        return z
    if pool == "mean":
        # 对 [1, ndim-1) 的中间维度求均值，保留 (B, C_last)
        return z.mean(dim=tuple(range(1, z.dim() - 1)))
    # 默认 flatten
    return z.reshape(z.shape[0], -1)


def dispersive_loss(
    z: torch.Tensor,
    variant: str = "infonce_l2",
    tau: float = 0.5,
    *,
    pool: str = "flatten",
    normalize_by_dim: bool = True,
) -> torch.Tensor:
    """成对排斥损失（标量）。z: (B, ...) 某中间 block 的激活；B>=2 才有意义。

    返回 0 维张量。变体：
      - "infonce_l2"   : log mean_{i,j} exp(−‖z_i−z_j‖²/τ)（logsumexp 稳定式；可选除以维度）
      - "infonce_cosine": log mean_{i,j} exp(cos(z_i,z_j)/τ)（尺度/维度无关，normalize_by_dim 无效）

    最小化该损失 → 表征在隐空间互相排斥（反坍缩）。
    """
    zf = reduce_representation(z, pool=pool).float()
    bs = zf.shape[0]
    if bs < 2:
        # 单样本无成对项 → 排斥无定义，返回 0（不贡献梯度）。
        return zf.new_zeros(())

    if variant == "infonce_cosine":
        zf = torch.nn.functional.normalize(zf, dim=1, eps=1e-8)
        sim = zf @ zf.t()                      # (B,B) 余弦相似度，对角=1
        logits = sim / float(tau)
    elif variant == "infonce_l2":
        # 成对平方 L2 距离矩阵 (B,B)，对角=0（论文保留对角）。
        d2 = torch.cdist(zf, zf, p=2.0).pow(2)
        if normalize_by_dim:
            d2 = d2 / float(zf.shape[1])       # 逐元素均方 → 分辨率/维度鲁棒、τ 可迁移
        logits = -d2 / float(tau)
    else:
        raise ValueError(f"未知 dispersive variant={variant!r}（支持 infonce_l2 / infonce_cosine）")

    # log mean exp(logits) = logsumexp(logits) − log(B*B)，含对角项（论文：作常数偏置可保留）。
    return torch.logsumexp(logits.reshape(-1), dim=0) - math.log(float(bs * bs))
