"""Noise-Conditioned Perceptual loss for DPO（arXiv 2406.17636，NCPPO）。

把 Diffusion-DPO 里逐样本的 **latent 空间去噪误差** `L_θ=‖v−v_θ‖²` 换成 **冻结 DiT 编码栈
特征空间的感知距离** `PL`，让偏好优化对齐"感知特征"而非"无语义结构的像素/latent 空间"。
论文在 SD1.5/SDXL 上只用原 DPO 7.5% 的 step×batch 就超过 latent-DPO（特征空间更省样本），
且 CPO 无参考模型也不发散（"与冻结副本匹配嵌入"自带强正则）。

机制（论文 Algorithm 1，移植到我们的 flow-matching / velocity 约定）：

    f      = 冻结参考 DiT 的前 K 个 block（"downsampling stack"的 DiT 类比，中间激活）
    x_t'^θ = x_t − (t−t')·v_θ           # 模型一步反演到 t'（= reverse step，用预测速度）
    x_t'^* = x_t − (t−t')·v_true        # 真值一步反演（用真速度 = 真实插值点，恒无梯度）
    PL_j   = ‖ f(x_t'^j, c, t') − f(x_t'^*, c, t') ‖²   # 逐样本特征 MSE（Eq.16）

DPO 包装沿用本项目既定的 **Linear-DPO**（trainer/dpo.py，arXiv 2605.21123）——NCP 是
"损失空间升级"（latent→感知），把四个 per-sample 项喂给 `linear_dpo_loss` 即可，不动其
线性效用 + η 裁剪 + 参考 KL 锚定。`f` 用**冻结参考权重**：θ 项要梯度经**输入** x_t'^θ
回流到策略（v_θ），但**不**经 f 权重 → 用 `frozen_params` 上下文把 adapter 参数临时
requires_grad=False（图盲：只碰 latent/特征张量，不解码出图、不外传）。

本模块只放与 2B 模型解耦的部分（纯张量数学 + 截断前向 + requires_grad 上下文），
`ncp_perceptual_loss` / `reverse_step_latent` 全 CPU 可单测（注入 feature_fn）；
`perceptual_features` 是接真模型的薄适配层。镜像 trainer/leap.py 的注入式风格。
"""

from __future__ import annotations

import contextlib

import torch

from .objective import _block_accepts_padding_mask


# ============================================================================
# 纯张量数学（CPU 可单测，注入 feature_fn）
# ============================================================================

def reverse_step_latent(x_t: torch.Tensor, v: torch.Tensor, dt) -> torch.Tensor:
    """Flow-matching 一步反演：x_{t'} = x_t − (t−t')·v（t'<t，朝数据端走 dt=t−t'）。

    dt: 标量、[B] 或可广播到 x_t 的张量；[B] 会自动右侧补维到 x_t 的 rank。
    """
    if torch.is_tensor(dt):
        while dt.ndim < x_t.ndim:
            dt = dt.unsqueeze(-1)
    return x_t - dt * v


def feature_mse(feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
    """逐样本特征 MSE：‖feat_a − feat_b‖²（对第 0 维以外全展平求均值）→ [B]。"""
    diff = feat_a.float() - feat_b.float()
    return diff.reshape(diff.shape[0], -1).pow(2).mean(dim=1)


def ncp_perceptual_loss(feature_fn, x_t, v_pred, v_true, dt, *, feat_true=None):
    """逐样本感知损失 PL = ‖f(x_t−dt·v_pred) − f(x_t−dt·v_true)‖²。

    - feature_fn(x) -> 特征张量；t'/cross/pad 由调用方 baked 进闭包（与 leap 注入风格一致）。
    - 真值锚 f(x_t−dt·v_true) **detach**（GT 不接收梯度）；梯度只经 v_pred。
    - feat_true 可由调用方预算好传入（同一 side 的 θ/ref 项共享，省一次 f 前向）；
      传入时必须已 detach。返回 (PL[B], feat_true)，feat_true 供同 side 复用。

    调用方负责 grad 上下文：θ 项在 frozen_params(+参考 .data) 下、**不** no_grad（输入要梯度）；
    ref 项整体 no_grad。两者 feature_fn 都跑**冻结参考**权重。
    """
    x_pred_tp = reverse_step_latent(x_t, v_pred, dt)
    feat_pred = feature_fn(x_pred_tp)
    if feat_true is None:
        x_true_tp = reverse_step_latent(x_t, v_true, dt)
        feat_true = feature_fn(x_true_tp).detach()
    return feature_mse(feat_pred, feat_true), feat_true


# ============================================================================
# requires_grad 上下文：冻结权重、保留输入梯度
# ============================================================================

@contextlib.contextmanager
def frozen_params(params):
    """临时把一组参数 requires_grad=False（退出精确还原原标志）。

    用于"f 用冻结参考权重、但梯度仍经其输入回流"：在此上下文内对参考权重跑前向不会给
    这些参数建梯度边（无 AccumulateGrad），而输入张量若自带 grad 则照常回传。主前向已在
    上下文外建好图（AccumulateGrad 已存在），故还原后 backward 不受影响。
    """
    params = list(params)
    saved = [bool(p.requires_grad) for p in params]
    try:
        for p in params:
            p.requires_grad_(False)
        yield
    finally:
        for p, s in zip(params, saved):
            p.requires_grad_(s)


# ============================================================================
# 截断前向：冻结 DiT 前 K 个 block 的中间激活 = 感知编码器 f（接真模型的薄适配层）
# ============================================================================

def resolve_tap_block(n_blocks: int, tap_block: int) -> int:
    """解析 tap_block：<0 → 自动取中间块 n//2；否则 clamp 到 [0, n-1]。"""
    if tap_block is None or int(tap_block) < 0:
        return max(0, n_blocks // 2)
    return max(0, min(int(tap_block), n_blocks - 1))


def perceptual_features(model, latents, timesteps, cross, padding_mask, tap_block: int):
    """跑 model 前 K 个 block，返回 block-K 后的网格隐状态 (B,T,H,W,D) 作为感知特征。

    复刻 forward_with_optional_checkpoint 的非 checkpoint 网格路径，但在第 K 个 block 后
    早停（不跑 final_layer/unpatchify）。f 的权重由调用方通过 .data 交换 + frozen_params
    指定为冻结参考；本函数不关心权重来源，只负责截断前向。use_checkpoint 恒 False（f 较浅）。
    """
    x_B_T_H_W_D, rope_emb, extra_pos_emb = model.prepare_embedded_sequence(
        latents, fps=None, padding_mask=padding_mask,
    )
    if timesteps.ndim == 1:
        timesteps = timesteps.unsqueeze(1)
    t_embedding, adaln_lora = model.t_embedder(timesteps)
    t_embedding = model.t_embedding_norm(t_embedding)
    block_kwargs = {
        "rope_emb_L_1_1_D": rope_emb,
        "adaln_lora_B_T_3D": adaln_lora,
        "extra_per_block_pos_emb": extra_pos_emb,
    }
    k = resolve_tap_block(len(model.blocks), tap_block)
    x = x_B_T_H_W_D
    for i, block in enumerate(model.blocks):
        if _block_accepts_padding_mask(block):
            x = block(x, t_embedding, cross, padding_mask=padding_mask, **block_kwargs)
        else:
            x = block(x, t_embedding, cross, **block_kwargs)
        if i >= k:
            break
    return x
