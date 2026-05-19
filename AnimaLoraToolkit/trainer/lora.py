"""LoRA / LoKr / DoRA 适配器与注入器。

包含：
- `LoRALayer`、`LoKrLayer` —— 两种低秩分解适配器（含 rank_dropout / module_dropout）
- `LoRALinear` —— 把原始 `torch.nn.Linear` 包成 adapter + base 的复合层，支持 DoRA
- `LoRAInjector` —— 全模型扫描 + regex 选择 + 模块级 rank / alpha / lr + LoRA+
                    + ComfyUI 兼容的 safetensors 保存/加载（native / diff / merged_model 三种导出格式）

被主训练脚本以及 checkpoint 模块依赖。本模块自身只 import torch / safetensors，不依赖任何
其它 trainer 子模块。
"""

from __future__ import annotations

import logging
import math
import re

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================================
# T-LoRA helpers
# ============================================================================

def _tlora_rank_mask(rank: int, r_min: int, current_t: torch.Tensor, alpha: float,
                     out_dtype: torch.dtype, out_device: torch.device) -> torch.Tensor:
    """Compute per-sample rank mask for T-LoRA.

    幂律 schedule: r(t) = floor(((1-t)^alpha) * (r - r_min)) + r_min
    论文 (arxiv:2507.05964) 的线性 schedule 是 alpha=1 的特例。
    返回形状 (B, rank) 的 0/1 mask。
    """
    t = current_t.to(device=out_device).clamp(0.0, 1.0).float()
    delta = max(rank - r_min, 0)
    r_t = (((1.0 - t) ** float(alpha)) * delta).floor().long() + int(r_min)
    r_t = r_t.clamp_(int(r_min), int(rank))
    ranks = torch.arange(int(rank), device=out_device)
    mask = (ranks.unsqueeze(0) < r_t.unsqueeze(1)).to(out_dtype)
    return mask


def _ortho_lora_init(in_features: int, out_features: int, rank: int):
    """Ortho-LoRA SVD init (arxiv:2507.05964 Eq.5).

    取 R ~ N(0, 1/sqrt(rank)) 的 SVD 最后 rank 个分量（sig_type="last"）：
        A_init = V^T[-rank:]                (rank, in_features)
        B_init = U[:, -rank:] * S[-rank:]   (out_features, rank)   # 把 S 折进 B
    返回 (A_init, B_init)；调用方负责拷贝到 nn.Linear 权重并保留 init 副本用于
    forward 时的初始 delta 补偿。
    """
    R = torch.randn(out_features, in_features) / math.sqrt(max(rank, 1))
    U, S, Vh = torch.linalg.svd(R, full_matrices=False)
    # SVD 返回 S 降序；最后 rank 个 = 最小奇异值（论文经验：抗过拟合最好）
    A_init = Vh[-rank:, :].contiguous()                          # (rank, in_features)
    S_init = S[-rank:]                                            # (rank,)
    B_init = (U[:, -rank:] * S_init.unsqueeze(0)).contiguous()    # (out_features, rank)
    return A_init, B_init


class LoRALayer(torch.nn.Module):
    """标准 LoRA 层（含 rank_dropout / module_dropout / 可选 T-LoRA）

    T-LoRA (arxiv:2507.05964)：
      - tlora_enabled=True 时，forward 内按 self.current_t（形状 (B,) 的 timestep
        tensor）应用 rank mask M_t，掩掉高 rank 通道；t→0 时全 rank，t→1 时只剩
        r_min 个。schedule 为幂律 r(t) = floor(((1-t)^alpha) * (r - r_min)) + r_min
        （alpha=1 即论文的线性 schedule）。
      - tlora_init="ortho" 时用 SVD 最后 rank 个分量做 Ortho-LoRA 初始化，并把
        初始 A/B 存为 buffer 用于 forward 时的初始 delta 补偿（论文 Eq.5），保证
        训练 step 0 时净 delta ≈ 0。
      - 用方法 set_current_t(t) 注入当前 timestep；调用方（train loop / sampling.py）
        在 model.forward 前调，forward 后 reset 成 None。
    """
    def __init__(self, in_features, out_features, rank=4, alpha=1.0, dropout=0.0,
                 rank_dropout=0.0, module_dropout=0.0,
                 tlora_enabled=False, tlora_rmin_ratio=0.5, tlora_alpha=1.0,
                 tlora_init="default"):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_down = torch.nn.Linear(in_features, rank, bias=False)
        self.lora_up = torch.nn.Linear(rank, out_features, bias=False)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        self.rank_dropout = float(rank_dropout or 0.0)
        self.module_dropout = float(module_dropout or 0.0)

        # T-LoRA state
        self.tlora_enabled = bool(tlora_enabled)
        self.tlora_alpha = float(tlora_alpha)
        self.tlora_init = (tlora_init or "default").lower()
        if self.tlora_enabled:
            self.r_min = max(int(round(rank * float(tlora_rmin_ratio))), 1)
        else:
            self.r_min = rank
        self.current_t: torch.Tensor | None = None

        # 初始化 + 可选 Ortho-LoRA + 初始 delta 补偿 buffer
        if self.tlora_enabled and self.tlora_init == "ortho":
            A_init, B_init = _ortho_lora_init(in_features, out_features, rank)
            with torch.no_grad():
                self.lora_down.weight.copy_(A_init)
                self.lora_up.weight.copy_(B_init)
            # 初始权重副本（不参与梯度），forward 时用同样的 mask 重新计算 init
            # 贡献并减去，让训练初始 step 净 delta ≈ 0（论文 Eq.5 的等价 2-matrix 版）
            self.register_buffer("lora_down_init", A_init.clone(), persistent=False)
            self.register_buffer("lora_up_init", B_init.clone(), persistent=False)
        else:
            torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
            torch.nn.init.zeros_(self.lora_up.weight)
            self.lora_down_init = None
            self.lora_up_init = None

    def set_current_t(self, t):
        """由训练循环 / 采样循环在 model.forward 前调用。t 可为 None（重置）。"""
        self.current_t = t

    def _apply_tlora_mask(self, h):
        """对 lora_down 的输出 h: (..., rank) 应用 per-sample T-LoRA mask。"""
        if not (self.tlora_enabled and self.current_t is not None):
            return h, None
        t = self.current_t
        # 输入可能是 (B, *, rank)；t 形状 (B,)
        if t.ndim == 0:
            t = t.view(1)
        mask = _tlora_rank_mask(self.rank, self.r_min, t, self.tlora_alpha,
                                h.dtype, h.device)
        # 广播 mask: (B, rank) → (B, 1, ..., 1, rank)
        if h.shape[0] != mask.shape[0]:
            # 单样本广播：t 形状 (1,) 而 h 第 0 维是别的（极少见，保守处理）
            if mask.shape[0] == 1:
                mask_view = mask.view(1, *([1] * (h.ndim - 2)), self.rank)
                return h * mask_view, mask
            raise RuntimeError(
                f"T-LoRA mask batch mismatch: h.shape[0]={h.shape[0]} vs current_t shape={tuple(t.shape)}"
            )
        mask_view = mask.view(mask.shape[0], *([1] * (h.ndim - 2)), self.rank)
        return h * mask_view, mask

    def forward(self, x):
        # Module dropout: 整个模块以 p 概率跳过（训练时）
        if self.training and self.module_dropout > 0:
            if torch.rand(1).item() < self.module_dropout:
                return torch.zeros(*x.shape[:-1], self.lora_up.out_features,
                                   device=x.device, dtype=x.dtype)
        x_drop = self.dropout(x)
        h = self.lora_down(x_drop)

        # T-LoRA: rank mask
        h, mask = self._apply_tlora_mask(h)

        # Rank dropout: 随机置零 rank 中的某些通道（训练时，inverted dropout）
        if self.training and self.rank_dropout > 0:
            rd_mask = torch.bernoulli(
                torch.full((self.rank,), 1.0 - self.rank_dropout, device=h.device)
            )
            h = h * rd_mask / (1.0 - self.rank_dropout + 1e-6)

        out = self.lora_up(h) * self.scaling

        # Ortho-LoRA 初始 delta 补偿：减去 (B_init · M_t · A_init) · x · scaling
        # 与上面的 (B · M_t · A) · x · scaling 相抵消，让训练 step 0 时净 delta ≈ 0
        if (self.tlora_enabled and self.lora_down_init is not None
                and self.lora_up_init is not None and self.current_t is not None):
            with torch.no_grad():
                pass  # buffers 已经 requires_grad=False
            h_init = F.linear(x_drop, self.lora_down_init)
            if mask is not None:
                h_init = h_init * mask.view(mask.shape[0], *([1] * (h_init.ndim - 2)), self.rank)
            out = out - F.linear(h_init, self.lora_up_init) * self.scaling

        return out


class LoKrLayer(torch.nn.Module):
    """LyCORIS LoKr 层 (ComfyUI 兼容) — w2 低秩分解版

    分解: ΔW = kron(w1, w2_a @ w2_b)
        w1   : (factor, factor)
        w2_a : (out_dim, rank)
        w2_b : (rank,    in_dim)
        其中 in_dim = in_features // factor, out_dim = out_features // factor

    forward 用 kron-bypass 数学等价但完全不实例化 (out_features, in_features) 全矩阵：
        将 x 视作 (..., factor, in_dim)，则 y = w1 @ ((x @ w2_b^T) @ w2_a^T)
        最后 reshape 回 (..., factor * out_dim)。
    复杂度 O(B*factor*in_dim*rank + B*factor*rank*out_dim + B*factor^2*out_dim)，
    远小于原本的 O(B*factor^2*in_dim*out_dim)，且无需在 bf16 中保存巨型 kron 矩阵。
    """
    def __init__(self, in_features, out_features, rank=4, alpha=1.0, factor=8, dropout=0.0,
                 rank_dropout=0.0, module_dropout=0.0,
                 tlora_enabled=False, tlora_rmin_ratio=0.5, tlora_alpha=1.0,
                 tlora_lokr_ortho_init=False):
        super().__init__()
        self.alpha = alpha
        self.in_features = in_features
        self.out_features = out_features

        # 自动调整 factor 确保能整除
        factor = self._find_factor(in_features, out_features, factor)
        self.factor = factor

        self.in_dim = in_features // factor
        self.out_dim = out_features // factor

        # cap rank 防止小层溢出
        self.rank = min(rank, self.out_dim, self.in_dim)
        self.scaling = alpha / self.rank

        # LoKr 分解: ΔW = kron(w1, w2_a @ w2_b)（命名与 LyCORIS 一致，可直接被 ComfyUI 加载）
        self.lokr_w1 = torch.nn.Parameter(torch.empty(factor, factor))
        self.lokr_w2_a = torch.nn.Parameter(torch.empty(self.out_dim, self.rank))
        self.lokr_w2_b = torch.nn.Parameter(torch.empty(self.rank, self.in_dim))
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        self.rank_dropout = float(rank_dropout or 0.0)
        self.module_dropout = float(module_dropout or 0.0)

        # T-LoRA (实验性，论文未覆盖 LoKr) state
        self.tlora_enabled = bool(tlora_enabled)
        self.tlora_alpha = float(tlora_alpha)
        if self.tlora_enabled:
            self.r_min = max(int(round(self.rank * float(tlora_rmin_ratio))), 1)
        else:
            self.r_min = self.rank
        self.current_t: torch.Tensor | None = None

        # ★ w1 用小 std 正态分布，配合 w2_b=0 初始时 ΔW=0；训练后 ΔW 量级由 scaling 控制
        torch.nn.init.normal_(self.lokr_w1, mean=0.0, std=0.1)

        if self.tlora_enabled and bool(tlora_lokr_ortho_init):
            # 实验性 ortho init：在 (w2_a, w2_b) 的简化维度上套 SVD 方案；
            # w1 保留原初始化。论文未覆盖该组合，需要 init delta 补偿避免初始 step 偏移。
            A_init, B_init = _ortho_lora_init(self.in_dim, self.out_dim, self.rank)
            with torch.no_grad():
                self.lokr_w2_a.data.copy_(B_init)  # (out_dim, rank)
                self.lokr_w2_b.data.copy_(A_init)  # (rank, in_dim)
            self.register_buffer("lokr_w2_a_init", B_init.clone(), persistent=False)
            self.register_buffer("lokr_w2_b_init", A_init.clone(), persistent=False)
            # w1 的初始副本（fp32），forward 时 init 分支用同一个 w1（它本身不掩 mask）
            self.register_buffer("lokr_w1_init", self.lokr_w1.detach().clone(), persistent=False)
        else:
            torch.nn.init.kaiming_uniform_(self.lokr_w2_a, a=5**0.5)
            torch.nn.init.zeros_(self.lokr_w2_b)
            self.lokr_w2_a_init = None
            self.lokr_w2_b_init = None
            self.lokr_w1_init = None

    def _find_factor(self, in_f, out_f, target_factor):
        """找到能同时整除 in_features 和 out_features 的 factor"""
        for f in [target_factor, 4, 2, 1]:
            if in_f % f == 0 and out_f % f == 0:
                return f
        return 1

    def set_current_t(self, t):
        """由训练循环 / 采样循环在 model.forward 前调用。t 可为 None（重置）。"""
        self.current_t = t

    def _apply_tlora_mask_kron(self, tmp_flat, x_orig_shape):
        """对 kron-bypass 中间 tensor `tmp_flat` (P, factor, rank) 应用 per-sample mask。
        P = prod(x_orig_shape[:-1]) = B * prod(中间维)。返回 (tmp_after_mask, mask_BR or None).
        """
        if not (self.tlora_enabled and self.current_t is not None):
            return tmp_flat, None
        t = self.current_t
        if t.ndim == 0:
            t = t.view(1)
        B = int(t.shape[0])
        P = tmp_flat.shape[0]
        if P % B != 0:
            # B 不能整除 P：罕见，可能是 fused 调用方式；退化为 broadcast 第一个 t
            t = t[:1]
            B = 1
        N_per_sample = P // B
        mask_BR = _tlora_rank_mask(self.rank, self.r_min, t, self.tlora_alpha,
                                   tmp_flat.dtype, tmp_flat.device)
        # mask: (B, rank) → (B, 1, 1, rank) → expand → (P, factor, rank)
        mask_view = mask_BR.view(B, 1, 1, self.rank).expand(
            B, N_per_sample, self.factor, self.rank
        ).reshape(P, self.factor, self.rank)
        return tmp_flat * mask_view, mask_BR

    def forward(self, x):
        # Module dropout: 整个模块以 p 概率跳过（训练时）
        if self.training and self.module_dropout > 0:
            if torch.rand(1).item() < self.module_dropout:
                return torch.zeros(*x.shape[:-1], self.out_features,
                                   device=x.device, dtype=x.dtype)

        # bf16 下 kron 容易数值放大，统一转 fp32 中间运算
        w1 = self.lokr_w1.float()
        w2_a = self.lokr_w2_a.float()
        w2_b = self.lokr_w2_b.float()

        # Rank dropout: 随机置零 rank 中的某些通道（训练时，inverted dropout）
        if self.training and self.rank_dropout > 0:
            mask = torch.bernoulli(
                torch.full((self.rank,), 1.0 - self.rank_dropout, device=w2_b.device)
            )
            scale = 1.0 / (1.0 - self.rank_dropout + 1e-6)
            w2_b = w2_b * (mask.unsqueeze(1) * scale)  # (rank, in_dim)

        x_drop = self.dropout(x)
        orig_shape = x_drop.shape
        # (..., in_features) → (B*, factor, in_dim)；保留前置维度
        x_flat = x_drop.reshape(-1, self.factor, self.in_dim).float()

        # 两段低秩矩阵乘代替 kron 全矩阵：
        #   tmp = x_flat @ w2_b^T  → (B*, factor, rank)
        #   T-LoRA mask 插在两段 matmul 之间（rank 维）
        #   tmp = tmp     @ w2_a^T → (B*, factor, out_dim)
        tmp = torch.matmul(x_flat, w2_b.transpose(0, 1))
        tmp, mask_BR = self._apply_tlora_mask_kron(tmp, orig_shape)
        tmp = torch.matmul(tmp, w2_a.transpose(0, 1))
        # 用 (factor, factor) 在前广播：w1 @ (B*, factor, out_dim) → (B*, factor, out_dim)
        y = torch.matmul(w1, tmp)

        # Ortho init compensation for LoKr T-LoRA：减去用 init 权重 + 同一 mask 的贡献，
        # 让训练 step 0 时净 delta ≈ 0。实验性，论文未覆盖。
        if (self.tlora_enabled and self.lokr_w2_a_init is not None
                and self.lokr_w2_b_init is not None and self.current_t is not None):
            w1_init = self.lokr_w1_init.float()
            w2a_init = self.lokr_w2_a_init.float()
            w2b_init = self.lokr_w2_b_init.float()
            tmp_i = torch.matmul(x_flat, w2b_init.transpose(0, 1))
            if mask_BR is not None:
                # 复用刚才算好的 mask
                B = int(mask_BR.shape[0])
                P = tmp_i.shape[0]
                N_per_sample = max(P // max(B, 1), 1)
                mask_view = mask_BR.view(B, 1, 1, self.rank).expand(
                    B, N_per_sample, self.factor, self.rank
                ).reshape(P, self.factor, self.rank)
                tmp_i = tmp_i * mask_view
            tmp_i = torch.matmul(tmp_i, w2a_init.transpose(0, 1))
            y_init = torch.matmul(w1_init, tmp_i)
            y = y - y_init

        # reshape 回 (..., out_features)
        y = y.reshape(*orig_shape[:-1], self.factor * self.out_dim)
        return y.to(dtype=x.dtype) * self.scaling

    def delta_weight(self, apply_rank_dropout: bool = False) -> torch.Tensor:
        """Materialize ΔW for DoRA weight decomposition and export checks."""
        w1 = self.lokr_w1.float()
        w2_a = self.lokr_w2_a.float()
        w2_b = self.lokr_w2_b.float()

        if apply_rank_dropout and self.training and self.rank_dropout > 0:
            mask = torch.bernoulli(
                torch.full((self.rank,), 1.0 - self.rank_dropout, device=w2_b.device)
            )
            scale = 1.0 / (1.0 - self.rank_dropout + 1e-6)
            w2_b = w2_b * (mask.unsqueeze(1) * scale)

        w2 = torch.matmul(w2_a, w2_b)
        return torch.kron(w1, w2) * self.scaling


class LoRALinear(torch.nn.Module):
    """LoRA 包装的 Linear 层（可选 DoRA / T-LoRA）"""
    def __init__(self, original, rank=4, alpha=1.0, dropout=0.0, use_lokr=False, factor=8,
                 rank_dropout=0.0, module_dropout=0.0, lora_variant="base",
                 tlora_rmin_ratio=0.5, tlora_alpha=1.0, tlora_init="default",
                 tlora_lokr_ortho_init=False):
        super().__init__()
        self.original = original
        self.use_lokr = use_lokr
        self.lora_variant = (lora_variant or "base").lower()
        self.use_dora = self.lora_variant == "dora"
        self.use_tlora = self.lora_variant == "tlora"

        # 组合兼容性校验：
        # - dora 仍只支持 lokr
        # - tlora 主路径只支持 lora；与 lokr 的组合属于实验性，由 injector 层显式 opt-in
        # - tlora × dora 不支持
        if self.use_dora and not use_lokr:
            raise ValueError("lora_variant='dora' is currently supported only with lora_type='lokr'")
        if self.use_tlora and self.use_dora:
            raise ValueError("lora_variant='tlora' is incompatible with DoRA")

        if use_lokr:
            self.adapter = LoKrLayer(
                original.in_features, original.out_features,
                rank=rank, alpha=alpha, factor=factor, dropout=dropout,
                rank_dropout=rank_dropout, module_dropout=module_dropout,
                tlora_enabled=self.use_tlora,
                tlora_rmin_ratio=tlora_rmin_ratio,
                tlora_alpha=tlora_alpha,
                tlora_lokr_ortho_init=tlora_lokr_ortho_init,
            )
        else:
            self.adapter = LoRALayer(
                original.in_features, original.out_features,
                rank=rank, alpha=alpha, dropout=dropout,
                rank_dropout=rank_dropout, module_dropout=module_dropout,
                tlora_enabled=self.use_tlora,
                tlora_rmin_ratio=tlora_rmin_ratio,
                tlora_alpha=tlora_alpha,
                tlora_init=tlora_init,
            )

        self.adapter.to(device=original.weight.device, dtype=original.weight.dtype)
        if self.use_dora:
            row_norm = original.weight.detach().float().norm(dim=1).clamp(min=1e-6)
            self.dora_scale = torch.nn.Parameter(row_norm.to(device=original.weight.device))
        for p in self.original.parameters():
            p.requires_grad = False

    def set_current_t(self, t):
        """转发给底层 adapter，使其在 forward 时能拿到当前 timestep。"""
        self.adapter.set_current_t(t)

    def forward(self, x):
        if self.use_dora:
            adapter = self.adapter
            if self.training and adapter.module_dropout > 0:
                if torch.rand(1, device=x.device).item() < adapter.module_dropout:
                    return self.original(x)

            delta = adapter.delta_weight(apply_rank_dropout=True).to(device=self.original.weight.device)
            base_w = self.original.weight.float()
            merged = base_w + delta
            denom = merged.norm(dim=1, keepdim=True).clamp(min=1e-6)
            scale = self.dora_scale.float().view(-1, 1) / denom
            dora_w = (merged * scale).to(dtype=self.original.weight.dtype)
            return F.linear(x, dora_w, self.original.bias)
        return self.original(x) + self.adapter(x)

    @property
    def weight(self):
        return self.original.weight

    @property
    def bias(self):
        return self.original.bias

    def merged_weight(self) -> torch.Tensor:
        base_w = self.original.weight.float()
        if self.use_lokr:
            delta = self.adapter.delta_weight(apply_rank_dropout=False).to(device=base_w.device)
        else:
            delta = torch.matmul(
                self.adapter.lora_up.weight.float(),
                self.adapter.lora_down.weight.float(),
            ) * self.adapter.scaling
            # T-LoRA Ortho init: 减去初始 delta 补偿（mask=I 即满 rank，等价于训练时
            # current_t=0 的 forward 行为）。这样合并出的权重表示"net 训练增量"。
            if (self.use_tlora
                    and getattr(self.adapter, "lora_down_init", None) is not None
                    and getattr(self.adapter, "lora_up_init", None) is not None):
                init_delta = torch.matmul(
                    self.adapter.lora_up_init.float(),
                    self.adapter.lora_down_init.float(),
                ).to(device=base_w.device) * self.adapter.scaling
                delta = delta - init_delta
        merged = base_w + delta
        if self.use_dora:
            denom = merged.norm(dim=1, keepdim=True).clamp(min=1e-6)
            merged = merged * (self.dora_scale.float().view(-1, 1) / denom)
        return merged


class LoRAInjector:
    """LoRA 注入器（支持 regex 模块选择、模块级 rank/lr、LoRA+）

    核心增强（从 kohya sd-scripts 借鉴）：
    - exclude_patterns / include_patterns: 用 re.fullmatch 替代前缀匹配
    - reg_dims: dict{regex: rank} 模块级 rank 控制
    - reg_lrs:  dict{regex: lr}   模块级学习率控制
    - loraplus_lr_ratio: LoRA+ w2_b/lora_up 用更高 lr
    """
    DEFAULT_TARGETS = ["q_proj", "k_proj", "v_proj", "output_proj", "mlp.layer1", "mlp.layer2"]
    DEFAULT_EXCLUDE_PATTERNS = [r".*llm_adapter.*"]

    def __init__(self, rank=32, alpha=16.0, dropout=0.0, use_lokr=False, factor=8,
                 targets=None, exclude_prefixes=None,
                 exclude_patterns=None, include_patterns=None,
                 reg_dims=None, reg_lrs=None, reg_alphas=None,
                 rank_dropout=0.0, module_dropout=0.0,
                 loraplus_lr_ratio=1.0, lora_variant="base", dora_export_mode="native",
                 # T-LoRA (arxiv:2507.05964)
                 tlora_rmin_ratio=0.5, tlora_alpha=1.0, tlora_init="ortho",
                 tlora_lokr_experimental=False, tlora_lokr_ortho_init=False):
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.use_lokr = use_lokr
        self.factor = factor
        self.lora_variant = (lora_variant or "base").lower()
        if self.lora_variant not in ("base", "dora", "tlora"):
            raise ValueError(f"Unknown lora_variant: {lora_variant}")
        if self.lora_variant == "dora" and not self.use_lokr:
            raise ValueError("lora_variant='dora' requires lora_type='lokr'")

        # T-LoRA 组合校验：
        # - 与 dora 互斥
        # - 与 lokr 组合属于实验性，论文未覆盖；需显式 opt-in（tlora_lokr_experimental=True）
        if self.lora_variant == "tlora":
            if self.use_lokr and not bool(tlora_lokr_experimental):
                raise ValueError(
                    "lora_variant='tlora' + lora_type='lokr' 属于实验性组合（论文 arxiv:2507.05964 仅覆盖标准 LoRA）。\n"
                    "如需启用，请显式在 YAML / CLI 中设 tlora_lokr_experimental=true，并注意：\n"
                    "  1) 没有现成的 ComfyUI loader 能在推理时给 LoKr 应用动态 mask；\n"
                    "  2) 导出的 checkpoint 在 ComfyUI 里会退化为标准 LoKr（满 rank）；\n"
                    "  3) Ortho 初始化在 Kronecker 分解上是我们做的启发式扩展，不在原论文范围。"
                )

        self.tlora_rmin_ratio = float(tlora_rmin_ratio or 0.5)
        self.tlora_alpha_param = float(tlora_alpha or 1.0)
        self.tlora_init = (tlora_init or "ortho").lower()
        if self.tlora_init not in ("ortho", "default"):
            raise ValueError(f"Unknown tlora_init: {tlora_init}")
        self.tlora_lokr_experimental = bool(tlora_lokr_experimental)
        self.tlora_lokr_ortho_init = bool(tlora_lokr_ortho_init)
        self.dora_export_mode = (dora_export_mode or "native").lower()
        if self.dora_export_mode not in ("native", "diff", "merged_model"):
            raise ValueError(f"Unknown dora_export_mode: {dora_export_mode}")
        self.targets = targets or self.DEFAULT_TARGETS
        self.rank_dropout = float(rank_dropout or 0.0)
        self.module_dropout = float(module_dropout or 0.0)
        self.loraplus_lr_ratio = max(float(loraplus_lr_ratio or 1.0), 1.0)

        # ── Regex 模块选择 ──────────────────────────────────────────────
        if exclude_patterns is not None:
            self.exclude_patterns = list(exclude_patterns)
        elif exclude_prefixes is not None:
            self.exclude_patterns = [re.escape(p) + r".*" for p in exclude_prefixes] if exclude_prefixes else []
        else:
            self.exclude_patterns = list(self.DEFAULT_EXCLUDE_PATTERNS)
        self.include_patterns = list(include_patterns or [])

        # ── 模块级 rank/lr 控制 ────────────────────────────────────────
        self.reg_dims = dict(reg_dims) if reg_dims else {}
        self.reg_alphas = dict(reg_alphas) if reg_alphas else {}
        self.reg_lrs = dict(reg_lrs) if reg_lrs else {}

        self.injected = {}
        self._module_ranks = {}
        self._module_alphas = {}
        self._module_lrs = {}

    def _should_inject(self, name):
        """判断模块是否应该被注入 LoRA（regex 匹配）"""
        if not any(t in name for t in self.targets):
            return False
        excluded = False
        for pat in self.exclude_patterns:
            if re.fullmatch(pat, name):
                excluded = True
                break
        if excluded:
            for pat in self.include_patterns:
                if re.fullmatch(pat, name):
                    return True
            return False
        return True

    def _get_reg_dim(self, name):
        """获取模块的 rank（优先 reg_dims 匹配，否则用全局 rank）"""
        for pat, dim in self.reg_dims.items():
            if re.fullmatch(pat, name):
                return int(dim)
        return self.rank

    def _get_reg_lr(self, name):
        """获取模块的自定义 lr（None 表示用全局）"""
        for pat, lr in self.reg_lrs.items():
            if re.fullmatch(pat, name):
                return float(lr)
        return None

    def _get_reg_alpha(self, name):
        for pat, alpha in self.reg_alphas.items():
            if re.fullmatch(pat, name):
                return float(alpha)
        return float(self.alpha)

    def inject(self, model):
        """注入 LoRA 到模型"""
        rank_summary = {}
        for name, module in list(model.named_modules()):
            if not isinstance(module, torch.nn.Linear):
                continue
            if not self._should_inject(name):
                continue

            mod_rank = self._get_reg_dim(name)
            mod_alpha = self._get_reg_alpha(name)
            mod_lr = self._get_reg_lr(name)

            lora_linear = LoRALinear(
                module, rank=mod_rank, alpha=mod_alpha,
                dropout=self.dropout, use_lokr=self.use_lokr, factor=self.factor,
                rank_dropout=self.rank_dropout, module_dropout=self.module_dropout,
                lora_variant=self.lora_variant,
                tlora_rmin_ratio=self.tlora_rmin_ratio,
                tlora_alpha=self.tlora_alpha_param,
                tlora_init=self.tlora_init,
                tlora_lokr_ortho_init=self.tlora_lokr_ortho_init,
            )

            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], lora_linear)
            self.injected[name] = lora_linear
            self._module_ranks[name] = mod_rank
            self._module_alphas[name] = mod_alpha
            self._module_lrs[name] = mod_lr
            rank_summary[mod_rank] = rank_summary.get(mod_rank, 0) + 1

        exc_str = ", ".join(self.exclude_patterns) or "无"
        inc_str = ", ".join(self.include_patterns) or "无"
        rank_dist = ", ".join(f"r{r}×{c}" for r, c in sorted(rank_summary.items()))
        if self.lora_variant == "tlora":
            variant_label = "T-LoKr (实验性)" if self.use_lokr else "T-LoRA"
        elif self.lora_variant == "dora":
            variant_label = "DoRA-LoKr"
        else:
            variant_label = "LoKr" if self.use_lokr else "LoRA"
        logger.info(
            f"注入 {variant_label} 到 {len(self.injected)} 层 "
            f"（排除: [{exc_str}], 包含: [{inc_str}], rank 分布: {rank_dist}）"
        )
        if self.lora_variant == "tlora":
            r_min_example = max(int(round(self.rank * self.tlora_rmin_ratio)), 1)
            logger.info(
                f"  T-LoRA: r_min_ratio={self.tlora_rmin_ratio} (r_min≈{r_min_example} for r={self.rank}), "
                f"alpha_schedule={self.tlora_alpha_param}, init={self.tlora_init}"
                + (f", lokr_ortho_init={self.tlora_lokr_ortho_init}" if self.use_lokr else "")
            )
            if self.use_lokr:
                logger.warning(
                    "  ⚠ T-LoRA + LoKr 属于实验性组合（论文未覆盖）。导出的 checkpoint 在 ComfyUI 中"
                    "会退化为标准 LoKr（满 rank 烘焙），动态 mask 仅训练 / sampling.py 出图时生效。"
                )
        if self.rank_dropout > 0 or self.module_dropout > 0:
            logger.info(f"  rank_dropout={self.rank_dropout}, module_dropout={self.module_dropout}")
        custom_alpha_modules = {
            n: a for n, a in self._module_alphas.items()
            if abs(float(a) - float(self.alpha)) > 1e-12
        }
        if custom_alpha_modules:
            for n, alpha in list(custom_alpha_modules.items())[:5]:
                rank = max(int(self._module_ranks.get(n, self.rank)), 1)
                logger.info(f"  module alpha: {n} -> {alpha:.3g} (scale={alpha / rank:.3g})")
            if len(custom_alpha_modules) > 5:
                logger.info(f"  ... {len(custom_alpha_modules)} modules have custom alpha")
        custom_lr_modules = {n: lr for n, lr in self._module_lrs.items() if lr is not None}
        if custom_lr_modules:
            for n, lr in list(custom_lr_modules.items())[:5]:
                logger.info(f"  模块级 lr: {n} → {lr:.2e}")
            if len(custom_lr_modules) > 5:
                logger.info(f"  ... 共 {len(custom_lr_modules)} 个模块有自定义 lr")
        return self.injected

    def get_params(self):
        """获取可训练参数"""
        params = []
        for lora in self.injected.values():
            params.extend(p for p in lora.parameters() if p.requires_grad)
        return params

    def set_current_t(self, t):
        """把当前 batch 的 timestep（形状 (B,) 或 None）写到每个 LoRA adapter。

        训练循环 / sampling.py 在 model.forward(...) 之前调用，forward 完成后传 None
        重置，避免跨 step 残留。仅 T-LoRA 变体在 forward 内会使用；其它 variant 忽略。
        """
        if self.lora_variant != "tlora":
            return
        for lora in self.injected.values():
            lora.set_current_t(t)

    def get_param_groups(self, weight_decay, base_lr: float = 1.0, loraplus_lr_ratio=None):
        """获取参数组（支持 LoRA+、模块级 lr、LoKr w1 排除 weight_decay）"""
        ratio = max(float(loraplus_lr_ratio or self.loraplus_lr_ratio), 1.0)
        groups_dict = {}  # (wd, lr_mult, custom_lr) -> [params]

        for name, lora in self.injected.items():
            custom_lr = self._module_lrs.get(name)
            if self.use_lokr:
                key_w1 = (0.0, 1.0, custom_lr)
                key_w2a = (weight_decay, 1.0, custom_lr)
                key_w2b = (weight_decay, ratio, custom_lr)
                groups_dict.setdefault(key_w1, []).append(lora.adapter.lokr_w1)
                groups_dict.setdefault(key_w2a, []).append(lora.adapter.lokr_w2_a)
                groups_dict.setdefault(key_w2b, []).append(lora.adapter.lokr_w2_b)
                if getattr(lora, "use_dora", False):
                    key_dora = (0.0, 1.0, custom_lr)
                    groups_dict.setdefault(key_dora, []).append(lora.dora_scale)
            else:
                key_down = (weight_decay, 1.0, custom_lr)
                key_up = (weight_decay, ratio, custom_lr)
                groups_dict.setdefault(key_down, []).append(lora.adapter.lora_down.weight)
                groups_dict.setdefault(key_up, []).append(lora.adapter.lora_up.weight)

        param_groups = []
        for (wd, lr_mult, custom_lr), params in groups_dict.items():
            if not params:
                continue
            group = {"params": params, "weight_decay": wd}
            if custom_lr is not None:
                group["lr"] = custom_lr * lr_mult
            elif lr_mult != 1.0:
                group["lr"] = float(base_lr) * lr_mult
            param_groups.append(group)

        if ratio > 1.0:
            lr_target = "LoKr w2_b" if self.use_lokr else "lora_up"
            logger.info(f"[LoRA+] {lr_target} lr ×{ratio:.1f}")
        return param_groups

    @staticmethod
    def comfy_weight_decompose(base_weight: torch.Tensor, diff_weight: torch.Tensor,
                               dora_scale: torch.Tensor) -> torch.Tensor:
        """Emulate ComfyUI output-axis DoRA for 2D linear weights."""
        base_f = base_weight.float()
        diff_f = diff_weight.float().to(device=base_f.device)
        calc = base_f + diff_f
        scale = dora_scale.float().to(device=base_f.device).reshape(-1, 1)
        base_norm = base_f.norm(dim=1, keepdim=True).clamp(min=1e-6)
        return calc * (scale / base_norm)

    @staticmethod
    def lycoris_output_axis_dora(base_weight: torch.Tensor, diff_weight: torch.Tensor,
                                 dora_scale: torch.Tensor) -> torch.Tensor:
        """LyCORIS-style output-axis DoRA formula used by training forward."""
        calc = base_weight.float() + diff_weight.float().to(device=base_weight.device)
        scale = dora_scale.float().to(device=calc.device).reshape(-1, 1)
        calc_norm = calc.norm(dim=1, keepdim=True).clamp(min=1e-6)
        return calc * (scale / calc_norm)

    @staticmethod
    def comfy_native_dora_scale(lora: LoRALinear) -> torch.Tensor:
        """Convert internal DoRA magnitude to a ComfyUI output-axis scale.

        ComfyUI's output-axis weight_decompose normalizes by ||W||, while the
        training forward normalizes by ||W + delta||. Export an adjusted scale
        so native .lokr_w* + .dora_scale reproduces the trained merged weight.
        """
        base_w = lora.original.weight.detach().float()
        delta = lora.adapter.delta_weight(apply_rank_dropout=False).detach().to(device=base_w.device)
        merged = base_w + delta
        base_norm = base_w.norm(dim=1).clamp(min=1e-6)
        merged_norm = merged.norm(dim=1).clamp(min=1e-6)
        magnitude = lora.dora_scale.detach().float().to(device=base_w.device)
        return magnitude * (base_norm / merged_norm)

    def state_dict(self, export_for_comfy=False):
        """导出 LoRA 权重。

        Training checkpoints keep raw DoRA magnitude. ComfyUI native export
        gets an adjusted output-axis scale so its weight_decompose matches
        LoRALinear.merged_weight() exactly for the exported checkpoint.

        T-LoRA 标准路径（lora_type=lora, lora_variant=tlora）使用 LyCORIS-style
        key 命名（`q_layer` / `p_layer` / `lambda_layer`），可被 bghira/ComfyUI-T-LoRA
        loader 识别并在推理时应用动态 mask。lambda_layer 是 Ortho-LoRA 初始 delta
        的补偿 buffer（B_init @ A_init），ComfyUI 节点用它重建 net delta。
        """
        sd = {}
        for name, lora in self.injected.items():
            base = "lora_unet_" + name.replace(".", "_")
            mod_alpha = self._module_alphas.get(name, self.alpha)
            sd[f"{base}.alpha"] = torch.tensor(float(mod_alpha))
            if self.use_lokr:
                # fp32 存储：训练时 Kronecker 积在 fp32 下计算，ComfyUI 加载后
                # 若张量是 fp32，合并时精度更接近训练行为（bf16 合并会损失小幅度 delta 的低位）
                # T-LoRA + LoKr 实验性路径：mask=I（满 rank）烘焙，直接存现有权重
                # （ComfyUI 加载即为标准 LoKr，无动态 mask）
                sd[f"{base}.lokr_w1"] = lora.adapter.lokr_w1.data.clone().float()
                sd[f"{base}.lokr_w2_a"] = lora.adapter.lokr_w2_a.data.clone().float()
                sd[f"{base}.lokr_w2_b"] = lora.adapter.lokr_w2_b.data.clone().float()
                if getattr(lora, "use_dora", False):
                    if export_for_comfy:
                        dora_scale = self.comfy_native_dora_scale(lora).cpu()
                    else:
                        dora_scale = lora.dora_scale.data.clone().float()
                    if export_for_comfy:
                        dora_scale = dora_scale.view(-1, 1)
                    sd[f"{base}.dora_scale"] = dora_scale
            else:
                if self.lora_variant == "tlora":
                    # LyCORIS-style key 命名，兼容 bghira/ComfyUI-T-LoRA loader
                    sd[f"{base}.q_layer.weight"] = lora.adapter.lora_down.weight.data.clone()
                    sd[f"{base}.p_layer.weight"] = lora.adapter.lora_up.weight.data.clone()
                    # 初始 delta 补偿（仅 ortho init 时有）：
                    # - lambda_layer: 完整的 (out_features, in_features) 初始 delta，
                    #   ComfyUI loader 用它实现 Ortho-LoRA 的"减去初始贡献"
                    # - q_layer_init / p_layer_init: 我们自己 resume_lora 时需要的
                    #   完整 A_init/B_init（lambda_layer 是其乘积，单独无法回推）
                    if (getattr(lora.adapter, "lora_down_init", None) is not None
                            and getattr(lora.adapter, "lora_up_init", None) is not None):
                        a_init = lora.adapter.lora_down_init.float().cpu()
                        b_init = lora.adapter.lora_up_init.float().cpu()
                        sd[f"{base}.lambda_layer"] = torch.matmul(b_init, a_init)
                        sd[f"{base}.q_layer_init.weight"] = a_init
                        sd[f"{base}.p_layer_init.weight"] = b_init
                else:
                    sd[f"{base}.lora_down.weight"] = lora.adapter.lora_down.weight.data.clone()
                    sd[f"{base}.lora_up.weight"] = lora.adapter.lora_up.weight.data.clone()
        return sd

    def _diff_state_dict(self):
        """Export exact ComfyUI .diff patches for final-checkpoint parity."""
        sd = {}
        for name, lora in self.injected.items():
            base = "lora_unet_" + name.replace(".", "_")
            diff = lora.merged_weight().detach().cpu().float() - lora.original.weight.detach().cpu().float()
            sd[f"{base}.diff"] = diff.contiguous()
        return sd

    def _merged_model_state_dict(self, model):
        """Export a full transformer state dict with adapter weights baked in."""
        if model is None:
            raise ValueError("dora_export_mode='merged_model' requires save(..., model=model)")
        sd = {}
        injected = dict(self.injected)
        for key, tensor in model.state_dict().items():
            if ".adapter." in key or key.endswith(".dora_scale"):
                continue
            if key.endswith(".original.weight"):
                prefix = key[: -len(".original.weight")]
                lora = injected.get(prefix)
                if lora is not None:
                    sd[f"{prefix}.weight"] = lora.merged_weight().detach().cpu()
                    continue
            if key.endswith(".original.bias"):
                prefix = key[: -len(".original.bias")]
                lora = injected.get(prefix)
                if lora is not None and lora.original.bias is not None:
                    sd[f"{prefix}.bias"] = lora.original.bias.detach().cpu()
                    continue
            sd[key] = tensor.detach().cpu()
        return sd

    def save(self, path, model=None):
        """保存为 safetensors (ComfyUI 兼容)"""
        from safetensors.torch import save_file

        if self.dora_export_mode == "merged_model":
            sd = self._merged_model_state_dict(model)
            save_file(sd, path, metadata={"format": "anima_merged_transformer"})
            logger.info(f"合并模型保存到: {path}")
            return

        if self.dora_export_mode == "diff":
            sd = self._diff_state_dict()
            meta = {
                "format": "anima_lora_diff",
                "ss_network_module": "diff",
                "anima_export_mode": "diff",
            }
            save_file(sd, path, metadata=meta)
            logger.info(f"LoRA diff 保存到: {path}")
            return

        sd = self.state_dict(export_for_comfy=True)
        network_args = f'{{"algo": "lokr", "factor": {self.factor}}}' if self.use_lokr else "{}"
        if self.use_lokr and self.lora_variant == "dora":
            network_args = f'{{"algo": "lokr", "factor": {self.factor}, "dora_wd": true}}'

        # T-LoRA 标准路径：用 bghira/ComfyUI-T-LoRA 能识别的 module 名 t_lora
        if self.lora_variant == "tlora" and not self.use_lokr:
            ss_network_module = "t_lora"
        elif self.use_lokr:
            ss_network_module = "lycoris.kohya"
        else:
            ss_network_module = "networks.lora"

        meta = {
            "ss_network_dim": str(self.rank),
            "ss_network_alpha": str(self.alpha),
            "ss_network_module": ss_network_module,
            "ss_network_args": network_args,
        }
        if self.use_lokr and self.lora_variant == "dora":
            meta["anima_dora_scale_format"] = "comfy_output_axis_adjusted"

        # T-LoRA 元数据（便于审计 + 下游 loader 解释 rank schedule）
        if self.lora_variant == "tlora":
            r_min_effective = max(int(round(self.rank * self.tlora_rmin_ratio)), 1)
            meta["anima_lora_variant"] = "tlora_lokr_experimental" if self.use_lokr else "tlora"
            meta["tlora_rmin"] = str(r_min_effective)
            meta["tlora_rmin_ratio"] = f"{self.tlora_rmin_ratio:.4f}"
            meta["tlora_schedule"] = "power_law"
            meta["tlora_alpha"] = f"{self.tlora_alpha_param:.4f}"
            meta["tlora_init"] = self.tlora_init
            if self.use_lokr:
                meta["anima_tlora_inference_supported"] = "false"
                meta["anima_tlora_inference_note"] = (
                    "LoKr+T-LoRA dynamic mask is training-only; baked checkpoint "
                    "loads as standard LoKr at inference."
                )
            else:
                meta["anima_tlora_inference_supported"] = "true"
                meta["anima_tlora_inference_note"] = (
                    "Compatible with bghira/ComfyUI-T-LoRA loader (LyCORIS-style keys: "
                    "q_layer/p_layer/lambda_layer)."
                )

        save_file(sd, path, metadata=meta)
        if self.lora_variant == "tlora" and self.use_lokr:
            logger.warning(
                f"⚠ LoKr+T-LoRA 实验性 checkpoint 已保存到 {path}（满 rank 烘焙；"
                f"ComfyUI 加载即为标准 LoKr，动态 mask 仅训练 / 训练期 sampling.py 生效）。"
            )
        else:
            logger.info(f"LoRA 保存到: {path}")

    def load(self, path):
        """从 safetensors 加载已有 LoRA 权重（用于继续训练）"""
        from safetensors import safe_open

        logger.info(f"加载已有 LoRA 权重: {path}")

        sd = {}
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                sd[k] = f.get_tensor(k)

        loaded_count = 0
        for name, lora in self.injected.items():
            base = "lora_unet_" + name.replace(".", "_")

            if self.use_lokr:
                w1_key = f"{base}.lokr_w1"
                w2a_key = f"{base}.lokr_w2_a"
                w2b_key = f"{base}.lokr_w2_b"
                dora_key = f"{base}.dora_scale"
                w2_old_key = f"{base}.lokr_w2"
                if w1_key in sd and w2a_key in sd and w2b_key in sd:
                    lora.adapter.lokr_w1.data.copy_(sd[w1_key])
                    lora.adapter.lokr_w2_a.data.copy_(sd[w2a_key])
                    lora.adapter.lokr_w2_b.data.copy_(sd[w2b_key])
                    if getattr(lora, "use_dora", False) and dora_key in sd:
                        dora_scale = sd[dora_key].reshape(-1)
                        lora.dora_scale.data.copy_(dora_scale.to(device=lora.dora_scale.device, dtype=lora.dora_scale.dtype))
                    loaded_count += 1
                elif w1_key in sd and w2_old_key in sd:
                    logger.warning(f"跳过旧格式 lokr_w2 全矩阵层: {name}（需重新训练）")
            else:
                down_key = f"{base}.lora_down.weight"
                up_key = f"{base}.lora_up.weight"
                # T-LoRA LyCORIS-style 命名（兼容 bghira/ComfyUI-T-LoRA）
                q_key = f"{base}.q_layer.weight"
                p_key = f"{base}.p_layer.weight"
                lambda_key = f"{base}.lambda_layer"
                if down_key in sd and up_key in sd:
                    lora.adapter.lora_down.weight.data.copy_(sd[down_key])
                    lora.adapter.lora_up.weight.data.copy_(sd[up_key])
                    loaded_count += 1
                elif q_key in sd and p_key in sd:
                    lora.adapter.lora_down.weight.data.copy_(sd[q_key])
                    lora.adapter.lora_up.weight.data.copy_(sd[p_key])
                    # 恢复 init A/B（resume_lora 必需，否则训练时的 Ortho-LoRA 补偿
                    # 会基于不同的随机 init → 续训行为偏离原 run）
                    q_init_key = f"{base}.q_layer_init.weight"
                    p_init_key = f"{base}.p_layer_init.weight"
                    if (q_init_key in sd and p_init_key in sd
                            and getattr(lora.adapter, "lora_down_init", None) is not None
                            and getattr(lora.adapter, "lora_up_init", None) is not None):
                        lora.adapter.lora_down_init.copy_(sd[q_init_key].to(
                            device=lora.adapter.lora_down_init.device,
                            dtype=lora.adapter.lora_down_init.dtype,
                        ))
                        lora.adapter.lora_up_init.copy_(sd[p_init_key].to(
                            device=lora.adapter.lora_up_init.device,
                            dtype=lora.adapter.lora_up_init.dtype,
                        ))
                    loaded_count += 1

        logger.info(f"从 checkpoint 加载了 {loaded_count}/{len(self.injected)} 层 LoRA 权重")
