"""训练目标（噪声 / timestep / loss / 自适应采样 / forward helpers）。

包含：
- dataclass: `TimestepConfig` / `NoiseConfig` / `LossConfig` / `TrainingObjectiveConfig`
  以及从 args 构造它们的 `build_training_objective_config`。
- timestep 采样：`sample_t`、`apply_timestep_schedule_shift`、`AdaptiveTimestepSampler`。
- 噪声生成：`make_noise`、`make_noise_from_config`
- per-sample loss：`per_sample_loss` + `_huber_delta_for_t`、`per_sample_highfreq_loss`
- adaptive 信号构造：`adaptive_timestep_metric_signal`
- 损失加权：`compute_loss_weight` + `apply_loss_weighting`
- 梯度范数（foreach 优化）：`compute_grad_norm`
- 前向 + 可选梯度检查点：`forward_with_optional_checkpoint`
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .aux_losses import AuxLossConfig, build_aux_loss_config

logger = logging.getLogger(__name__)


# ============================================================================
# Configs (dataclasses)
# ============================================================================

@dataclass(frozen=True)
class TimestepConfig:
    mode: str = "logit_normal"
    flow_shift: float = 3.0
    mix_low_prob: float = 0.25
    schedule_shift: float = 1.0
    # ── Laplace 噪声调度（Hang et al. 2024, arxiv:2407.03297）——仅 mode="laplace" 生效 ──
    # log-SNR 按 Laplace 分布采样：λ = μ - b·sgn(0.5-u)·log(1-2|u-0.5|)，再映射回 flow t。
    # μ>0 把采样推向高 SNR / 低噪声（细节）端；μ<0 推向高噪；b 控制集中度（越小越集中）。
    # 默认 μ=0,b=0.5 = 论文 ImageNet-256 设置（中噪聚焦）。
    # 建议与 schedule_shift=1.0 + adaptive_timestep=false 搭配做干净对照（避免二次偏移）。
    laplace_mu: float = 0.0
    laplace_b: float = 0.5


@dataclass(frozen=True)
class NoiseConfig:
    offset: float = 0.0
    offset_min: float = 0.0
    random_offset_strength: bool = False
    pyramid_iterations: int = 0
    pyramid_discount: float = 0.3


@dataclass(frozen=True)
class LossConfig:
    loss_type: str = "mse"
    huber_c: float = 0.1
    huber_schedule: str = "constant"
    weighting_scheme: str = "none"
    min_snr_gamma: float = 0.0
    weight_cap_ratio: float = 0.0
    # detail_inv_t weighting 的可调上下限。默认 [1, 5] = 历史行为，更保守可设
    # [1.5, 3]（hazy 画风）或彻底关掉 detail_inv_t（balanced 配方）。
    detail_inv_t_min: float = 1.0
    detail_inv_t_max: float = 5.0
    # ── 新增旋钮（默认值 = 历史行为，全部 no-op）──
    # snr 调度下低 t（细节区）δ 的上限。当前 snr 调度在 t→0 处 δ 可冲到 10·huber_c
    #   = 近乎纯 L2 = 对脏样本坏细节零 outlier 保护。调小（如 3）让低噪区也保留部分
    #   L1 鲁棒（抗脏数据，代价是极细节精度略降）。10.0 = 旧行为。
    huber_snr_clamp_max: float = 10.0
    # Contrastive Flow Matching（ΔFM, arxiv:2506.05350）排斥项权重 λ。
    #   per_sample ← per_sample - λ·||v_pred - v_另一样本||²，反"回归条件均值→发灰发雾"。
    #   零额外前向；0=关闭，论文甜点 0.05，≥0.15 会分布塌缩。
    dfm_lambda: float = 0.0


@dataclass(frozen=True)
class TrainingObjectiveConfig:
    timestep: TimestepConfig
    noise: NoiseConfig
    loss: LossConfig
    # ★ 辅助 loss（Spectral / Perceptual）；默认全关 → 完全 no-op 向后兼容
    aux: AuxLossConfig = field(default_factory=AuxLossConfig)


def build_training_objective_config(args) -> TrainingObjectiveConfig:
    return TrainingObjectiveConfig(
        timestep=TimestepConfig(
            mode=str(getattr(args, "timestep_sampling", "logit_normal") or "logit_normal"),
            flow_shift=float(getattr(args, "flow_shift", 3.0) or 3.0),
            mix_low_prob=float(getattr(args, "timestep_mix_low_prob", 0.25) or 0.0),
            schedule_shift=float(getattr(args, "schedule_shift", 1.0) or 1.0),
            laplace_mu=float(getattr(args, "timestep_laplace_mu", 0.0) or 0.0),
            laplace_b=float(getattr(args, "timestep_laplace_b", 0.5) or 0.5),
        ),
        noise=NoiseConfig(
            offset=float(getattr(args, "noise_offset", 0.0) or 0.0),
            offset_min=float(getattr(args, "noise_offset_min", 0.0) or 0.0),
            random_offset_strength=bool(getattr(args, "noise_offset_random_strength", False)),
            pyramid_iterations=int(getattr(args, "pyramid_noise_iterations", 0) or 0),
            pyramid_discount=float(getattr(args, "pyramid_noise_discount", 0.3) or 0.3),
        ),
        loss=LossConfig(
            loss_type=str(getattr(args, "loss_type", "mse") or "mse"),
            huber_c=float(getattr(args, "huber_c", 0.1) or 0.1),
            huber_schedule=str(getattr(args, "huber_schedule", "constant") or "constant"),
            weighting_scheme=str(getattr(args, "loss_weighting_scheme", "none") or "none"),
            min_snr_gamma=float(getattr(args, "min_snr_gamma", 0.0) or 0.0),
            weight_cap_ratio=float(getattr(args, "weight_cap_ratio", 0.0) or 0.0),
            detail_inv_t_min=float(getattr(args, "detail_inv_t_min", 1.0) or 1.0),
            detail_inv_t_max=float(getattr(args, "detail_inv_t_max", 5.0) or 5.0),
            huber_snr_clamp_max=float(getattr(args, "huber_snr_clamp_max", 10.0) or 10.0),
            dfm_lambda=float(getattr(args, "dfm_lambda", 0.0) or 0.0),
        ),
        aux=build_aux_loss_config(args),
    )


# ============================================================================
# Timestep sampling
# ============================================================================

def sample_t(
    bs,
    device,
    mode: str = "logit_normal",
    shift: float = 3.0,
    mix_low_prob: float = 0.25,
    laplace_mu: float = 0.0,
    laplace_b: float = 0.5,
):
    """采样 Flow Matching 时间步 t ∈ (0, 1)。

    mode:
      - "logit_normal": 经典 SD3/Anima 偏向中间 t 的分布，shift>1 进一步偏向高噪声端（默认）。
      - "uniform":      均匀采样 t，对低噪声端（细节）和高噪声端（结构）覆盖更均衡。
      - "logit_normal_low": logit-normal 但 shift 反向（推 t 偏向低噪声/细节端），适合刻画细节差的训练集。
      - "mode":         SD3 式 mode-distribution（用 sigma 形式，需要 shift）。
      - "mixed_uniform_low": 以 uniform 为主体，按 mix_low_prob 混入 logit_normal_low；
      - "mixed_logit_low_high"（别名 ushaped/bimodal）: U 形双峰。低噪(细节)峰 logit_normal_low
                        + 高噪(结构)峰 logit_normal，中段 t 被自然掏空 → 同时喂"细节(低t)"和
                        "脸型/构图/氛围(高t)"两端。mix_low_prob = 路由到低噪峰的比例（0.5=对称U），
                        flow_shift 控两峰间距（shift=3 → 峰约 t≈0.25/0.75）。
      - "laplace":      log-SNR 按 Laplace 分布采样（arxiv:2407.03297）。用 laplace_mu/laplace_b
                        控制峰位与集中度，μ>0 偏低噪声/细节端。是 detail_inv_t+mix_low+schedule_shift
                        那一堆 ad-hoc 旋钮的原理化替代。
    """
    mode = (mode or "logit_normal").lower()
    if mode == "uniform":
        return torch.rand(bs, device=device).clamp(1e-4, 1.0 - 1e-4)

    if mode == "laplace":
        # u ~ U(0,1) 作为分位点；λ = log-SNR 按 Laplace(μ, b) 逆 CDF 采样。
        u = torch.rand(bs, device=device).clamp(1e-4, 1.0 - 1e-4)
        sgn = torch.sign(0.5 - u)
        # log(1 - 2|u-0.5|)：u→0/1 时 →-inf（λ→±inf），u→0.5 时 →0（λ→μ）
        lam = float(laplace_mu) - float(laplace_b) * sgn * torch.log1p(-2.0 * (u - 0.5).abs())
        # 映射 log-SNR → flow-matching t：CONST 调度 SNR=((1-t)/t)^2 ⇒ t = 1/(1+exp(λ/2))
        t = 1.0 / (1.0 + torch.exp(0.5 * lam))
        return t.clamp(1e-4, 1.0 - 1e-4)

    if mode in ("mixed_uniform_low", "uniform_low_mix"):
        uniform_t = torch.rand(bs, device=device)
        low_t = sample_t(bs, device, mode="logit_normal_low", shift=shift)
        p_low = min(max(float(mix_low_prob), 0.0), 1.0)
        r = torch.rand(bs, device=device)
        t = torch.where(r < p_low, low_t, uniform_t)
        return t.clamp(1e-4, 1.0 - 1e-4)

    if mode in ("mixed_uniform_logit", "uniform_logit_mix"):
        uniform_t = torch.rand(bs, device=device)
        logit_t = sample_t(bs, device, mode="logit_normal", shift=shift)
        p = min(max(float(mix_low_prob), 0.0), 1.0)
        use_logit = (torch.rand(bs, device=device) < p)
        return torch.where(use_logit, logit_t, uniform_t).clamp(1e-4, 1.0 - 1e-4)

    if mode in ("mixed_logit_low_high", "ushaped", "u_shaped", "bimodal"):
        # U 形 / 双峰：低噪(细节)峰 logit_normal_low + 高噪(结构/脸型/氛围)峰 logit_normal。
        # 中段 t（最"易"、信息量最低）被自然掏空，两端同时获得监督。
        # mix_low_prob = 路由到低噪(细节)峰的比例；1-p 进高噪(结构)峰。0.5 ≈ 对称 U，
        # <0.5 偏结构端（保 v26 脸型/氛围收益），>0.5 偏细节端。
        # flow_shift 控两峰间距（shift=3 → 峰约 t≈0.25 / 0.75；越大两峰越分开）。
        low_t = sample_t(bs, device, mode="logit_normal_low", shift=shift)
        high_t = sample_t(bs, device, mode="logit_normal", shift=shift)
        p_low = min(max(float(mix_low_prob), 0.0), 1.0)
        use_low = (torch.rand(bs, device=device) < p_low)
        return torch.where(use_low, low_t, high_t).clamp(1e-4, 1.0 - 1e-4)

    # 基础 logit-normal
    u = torch.sigmoid(torch.randn(bs, device=device))

    if mode == "logit_normal_low":
        # shift 倒数，效果是把 t 推向 0 端（更多“低噪声/细节”样本）
        s = max(float(shift), 1e-4)
        u = (u * (1.0 / s)) / (1 + (1.0 / s - 1) * u)
        return u.clamp(1e-4, 1.0 - 1e-4)

    if mode == "mode":
        # SD3 mode sampling: 集中在某个 sigma 附近
        s = float(shift)
        u = 1 - u - s * (torch.cos(torch.pi * 0.5 * u) ** 2 - 1 + u)
        return u.clamp(1e-4, 1.0 - 1e-4)

    # 默认 logit_normal + shift
    s = float(shift)
    u = (u * s) / (1 + (s - 1) * u)
    return u.clamp(1e-4, 1.0 - 1e-4)


def apply_timestep_schedule_shift(t: torch.Tensor, schedule_shift: float) -> torch.Tensor:
    sched_shift = float(schedule_shift or 1.0)
    if sched_shift > 0 and abs(sched_shift - 1.0) > 1e-6:
        t = (t * sched_shift) / (1 + (sched_shift - 1) * t)
    return t.clamp(1e-4, 1.0 - 1e-4)


class AdaptiveTimestepSampler:
    """Conservative loss-aware resampler layered on top of sample_t().

    metric=entropy_rate 时改用 InfoNoise 风格信号：factor ∝ (mse_hat / t³) / w(t)，
    再经 low_noise_gate g(t) = t^n / (t^n + c^n) 抑制 t→0 端的失控分配。这是 arxiv
    2602.18647 的核心思路在 flow-matching t-空间下的重述（线性 FM 下 σ_t = t，
    I-MMSE 等式可直接搬过来）。论文只在 EDM/DDPM 上验证过，因此该 metric 默认关。
    """
    def __init__(
        self,
        enabled: bool = False,
        bins: int = 16,
        ema_decay: float = 0.95,
        burn_in_steps: int = 160,
        min_factor: float = 0.5,
        max_factor: float = 2.0,
        base_mix: float = 0.25,
        candidate_mult: int = 8,
        metric: str = "raw",
        highfreq_weight: float = 0.25,
        # InfoNoise / entropy_rate 模式专用
        low_noise_gate: bool = False,
        gate_n: float = 3.0,
        gate_c: float = 0.05,
        loss_weight_fn=None,
    ):
        self.enabled = bool(enabled)
        self.bins = max(int(bins or 16), 2)
        self.ema_decay = min(max(float(ema_decay), 0.0), 0.999)
        self.burn_in_steps = max(int(burn_in_steps or 0), 0)
        self.min_factor = max(float(min_factor), 1e-3)
        self.max_factor = max(float(max_factor), self.min_factor)
        self.base_mix = min(max(float(base_mix), 0.0), 1.0)
        self.candidate_mult = max(int(candidate_mult or 1), 1)
        self.metric = (metric or "raw").lower()
        if self.metric not in ("raw", "highfreq", "mixed", "entropy_rate"):
            raise ValueError(f"Unknown adaptive_timestep_metric: {metric}")
        self.highfreq_weight = max(float(highfreq_weight or 0.0), 0.0)
        self.loss_ema = torch.zeros(self.bins, dtype=torch.float32)
        self.counts = torch.zeros(self.bins, dtype=torch.long)
        # InfoNoise 闸门 + 损失权重补偿
        self.low_noise_gate = bool(low_noise_gate)
        self.gate_n = max(float(gate_n or 0.0), 1e-3)
        self.gate_c = max(float(gate_c or 0.0), 1e-6)
        # loss_weight_fn(t_tensor) -> tensor of same shape；用于 entropy_rate 模式
        # 把当前 loss-weighting scheme 的 w(t) 除掉，让 π·w ∝ ρ。
        self.loss_weight_fn = loss_weight_fn

    @property
    def ready(self) -> bool:
        return self.enabled and bool((self.counts > 0).all())

    def _bin_index(self, t: torch.Tensor) -> torch.Tensor:
        return torch.clamp((t.float().detach().cpu() * self.bins).long(), 0, self.bins - 1)

    def update(self, t: torch.Tensor, per_sample: torch.Tensor) -> None:
        if not self.enabled:
            return
        t_bins = self._bin_index(t)
        losses = per_sample.detach().float().cpu()
        for idx in range(self.bins):
            mask = t_bins == idx
            if not bool(mask.any()):
                continue
            val = losses[mask].mean()
            if self.counts[idx] == 0:
                self.loss_ema[idx] = val
            else:
                self.loss_ema[idx] = self.ema_decay * self.loss_ema[idx] + (1.0 - self.ema_decay) * val
            self.counts[idx] += int(mask.sum().item())

    def factors(self) -> torch.Tensor:
        if not self.ready:
            return torch.ones(self.bins, dtype=torch.float32)
        losses = self.loss_ema.clamp(min=1e-8)

        if self.metric == "entropy_rate":
            # InfoNoise: factor_k ∝ (mse_hat_k / t_k³) / w(t_k)
            # bin 中心：每个 bin 覆盖 [k/bins, (k+1)/bins]，取中点 (k+0.5)/bins
            bin_centers = (torch.arange(self.bins, dtype=torch.float32) + 0.5) / float(self.bins)
            # I-MMSE 风格熵率代理：mse / t³，clamp 防止 t→0 处爆炸
            t_cubed = bin_centers.pow(3).clamp(min=1e-6)
            entropy_rate = losses / t_cubed

            # 除以 w(t)，让 π·w ∝ ρ（与论文 Eq.16 一致：π(σ) ∝ ρ(σ)/w(σ)）
            if self.loss_weight_fn is not None:
                try:
                    w = self.loss_weight_fn(bin_centers).clamp(min=1e-6)
                    entropy_rate = entropy_rate / w
                except Exception as _e:
                    logger.warning(f"loss_weight_fn 调用失败，回退到不除 w(t): {_e}")

            # 低噪闸门 g(t) = t^n / (t^n + c^n)：t→0 时趋于 0，抑制极低噪 bin 的过度分配
            if self.low_noise_gate:
                t_n = bin_centers.pow(self.gate_n)
                c_n = float(self.gate_c) ** self.gate_n
                gate = t_n / (t_n + c_n)
                entropy_rate = entropy_rate * gate

            rel = entropy_rate / entropy_rate.mean().clamp(min=1e-8)
        else:
            rel = losses / losses.mean().clamp(min=1e-8)

        return rel.clamp(self.min_factor, self.max_factor)

    def sample(self, bs, device, *, mode: str, shift: float, mix_low_prob: float,
               schedule_shift: float = 1.0,
               laplace_mu: float = 0.0, laplace_b: float = 0.5,
               global_step: int) -> torch.Tensor:
        base_t = sample_t(bs, device, mode=mode, shift=shift, mix_low_prob=mix_low_prob,
                          laplace_mu=laplace_mu, laplace_b=laplace_b)
        if (not self.enabled) or global_step < self.burn_in_steps or not self.ready:
            return base_t

        adaptive_count = int(round(bs * (1.0 - self.base_mix)))
        if adaptive_count <= 0:
            return base_t

        candidates_n = max(adaptive_count * self.candidate_mult, adaptive_count)
        candidates = sample_t(candidates_n, device, mode=mode, shift=shift, mix_low_prob=mix_low_prob,
                              laplace_mu=laplace_mu, laplace_b=laplace_b)
        candidates_final = apply_timestep_schedule_shift(candidates, schedule_shift)
        candidate_bins = torch.clamp((candidates_final.float() * self.bins).long(), 0, self.bins - 1)
        weights = self.factors().to(device=candidates.device)[candidate_bins]
        probs = weights / weights.sum().clamp(min=1e-8)
        chosen = torch.multinomial(probs, adaptive_count, replacement=True)
        adapted = candidates[chosen]

        if adaptive_count >= bs:
            return adapted[:bs].clamp(1e-4, 1.0 - 1e-4)
        out = base_t.clone()
        out[:adaptive_count] = adapted
        perm = torch.randperm(bs, device=device)
        return out[perm].clamp(1e-4, 1.0 - 1e-4)

    def summary(self) -> str:
        factors = self.factors()
        extra = ""
        if self.metric == "entropy_rate":
            extra = f" (entropy_rate; gate={'on' if self.low_noise_gate else 'off'}"
            if self.low_noise_gate:
                extra += f" n={self.gate_n:.1f} c={self.gate_c:.3f}"
            extra += ")"
        return (
            f"metric={self.metric} hf_weight={self.highfreq_weight:.3f} "
            f"bins={self.bins} burn_in={self.burn_in_steps} "
            f"factor_min/max={float(factors.min()):.2f}/{float(factors.max()):.2f}{extra}"
        )


# ============================================================================
# Noise
# ============================================================================

def make_noise(latents, noise_offset: float = 0.0, pyramid_iters: int = 0,
               pyramid_discount: float = 0.3, random_offset_strength: bool = False,
               noise_offset_min: float = 0.0):
    """生成训练用噪声。

    base: standard normal
    noise_offset: 给每个样本/通道加一个低频偏移，缓解“总是中等亮度”的偏差，对学习明暗对比尤其有效（来自 SDXL 的 noise_offset 思路）。
    noise_offset_min: random_offset_strength=true 时的随机下限；默认 0 兼容旧行为。
    pyramid_iters: 叠加多尺度低频噪声，帮助模型快速学习全局光照/构图（参考 multires noise / pyramid noise）。

    ⚠ 顺序很重要：
      先做 pyramid 叠加 + 整体归一化（让噪声仍保持 std≈1，避免方差爆炸），
      然后再加 noise_offset。这样 offset 的实际幅度与配置数字一致。
      旧实现先 offset 再 pyramid 归一化，offset 会被 std-rescale 一起缩水。

    pyramid_discount 当前用 `discount ** (i+1)`（i 从 0 起），比 Whitaker/kohya 标准
    实现 `discount ** i` 弱一个量级。这是有意为之的"弱模式"：对追求绝对还原的训练
    更友好（pyramid 几乎不引入全局色调泛化）。若想要标准 multires noise 的强度，
    把配置里的 discount 从默认 0.3 提到 ~0.5-0.7 即可获得近似 Whitaker 效果。
    """
    out_dtype = latents.dtype
    noise = torch.randn_like(latents, dtype=torch.float32)

    # === Step 1: Pyramid 叠加 + 归一化（如果启用） ===
    if pyramid_iters and int(pyramid_iters) > 0:
        try:
            spatial_dims = list(latents.shape[-2:])
            cur = noise.clone()
            for i in range(int(pyramid_iters)):
                r = 2 ** (i + 1)
                small_h = max(spatial_dims[0] // r, 1)
                small_w = max(spatial_dims[1] // r, 1)
                # 5D latent: (B, C, T, H, W)；4D 也支持
                # NOTE: 用 bilinear 而非 nearest，与 Whitaker 原版 pyramid_noise_like 一致；
                # nearest 会产生块状低频结构，模型把"预测块状偏移"也作为目标的一部分学习，
                # 导致规则小结构（如扣子、文字、网格）训练后变形。bilinear 提供平滑的 LF 噪声。
                if latents.ndim == 5:
                    extra = torch.randn(latents.shape[0], latents.shape[1], latents.shape[2], small_h, small_w,
                                        device=latents.device, dtype=torch.float32)
                    extra = F.interpolate(extra.flatten(0, 1), size=spatial_dims, mode="bilinear",
                                          align_corners=False).view(
                        latents.shape[0], latents.shape[1], latents.shape[2], spatial_dims[0], spatial_dims[1])
                else:
                    extra = torch.randn(latents.shape[0], latents.shape[1], small_h, small_w,
                                        device=latents.device, dtype=torch.float32)
                    extra = F.interpolate(extra, size=spatial_dims, mode="bilinear", align_corners=False)
                cur = cur + extra * (float(pyramid_discount) ** (i + 1))
                if min(small_h, small_w) <= 1:
                    break
            # 归一到与原噪声相同的方差，保持训练稳定。
            reduce_dims = tuple(range(1, cur.ndim))
            cur = cur / cur.std(dim=reduce_dims, keepdim=True).clamp(min=1e-6)
            noise = cur
        except Exception as _e:
            logger.warning(f"pyramid_noise 计算失败，回退到标准噪声: {_e}")

    # === Step 2: noise_offset 加在归一化后的噪声上 ===
    # 这样配置里的 noise_offset 强度就是实际生效的强度（旧实现里这一步被
    # 后续 pyramid 的 cur/cur.std() 吃掉过一次）。
    if noise_offset and noise_offset > 0:
        # 形状: (B, C, T, 1, 1) 或 (B, C, 1, 1) — 与 latents 兼容的"低频"扰动
        leading_shape = list(latents.shape)
        for ax in range(2, latents.ndim):
            leading_shape[ax] = 1
        offset = torch.randn(*leading_shape, device=latents.device, dtype=torch.float32)
        scale = float(noise_offset)
        if random_offset_strength:
            lo = max(float(noise_offset_min or 0.0), 0.0)
            hi = max(scale, 0.0)
            if lo > hi:
                lo, hi = hi, lo
            scale_shape = [latents.shape[0]] + [1] * (latents.ndim - 1)
            scale = lo + (hi - lo) * torch.rand(scale_shape, device=latents.device, dtype=torch.float32)
        noise = noise + scale * offset

    return noise.to(dtype=out_dtype)


def make_noise_from_config(latents: torch.Tensor, cfg: NoiseConfig) -> torch.Tensor:
    return make_noise(
        latents,
        noise_offset=cfg.offset,
        pyramid_iters=cfg.pyramid_iterations,
        pyramid_discount=cfg.pyramid_discount,
        random_offset_strength=cfg.random_offset_strength,
        noise_offset_min=cfg.offset_min,
    )


# ============================================================================
# Loss
# ============================================================================

def _huber_delta_for_t(t: torch.Tensor | None, huber_c: float, schedule: str,
                       snr_clamp_max: float = 10.0):
    delta = max(float(huber_c), 1e-8)
    if t is None:
        return delta

    schedule = (schedule or "constant").lower()
    if schedule == "constant":
        return delta

    t_c = t.float().clamp(1e-4, 1.0 - 1e-4)
    if schedule == "snr":
        # High SNR / low-noise steps get a larger quadratic basin; high-noise steps become more L1-like.
        # snr_clamp_max 决定低 t（细节区）δ 上限：默认 10.0=旧行为（低 t 近纯 L2，脏数据零保护）；
        # 调小（如 3）让低噪区也保留部分 L1 鲁棒（抗脏数据集）。
        hi = max(float(snr_clamp_max), 0.1 + 1e-6)
        snr_sqrt = ((1.0 - t_c) / t_c).clamp(0.1, hi)
        return (delta * snr_sqrt).view(-1, *([1] * 4))
    if schedule == "sigma":
        return (delta * t_c.clamp(0.1, 1.0)).view(-1, *([1] * 4))

    return delta


def per_sample_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str = "mse",
                    huber_c: float = 0.1, huber_schedule: str = "constant",
                    t: torch.Tensor | None = None,
                    huber_snr_clamp_max: float = 10.0) -> torch.Tensor:
    """Return per-sample loss for tensors shaped (B, C, T, H, W)."""
    pred_f = pred.float()
    target_f = target.float()
    loss_type = (loss_type or "mse").lower()

    if loss_type in ("mse", "l2"):
        loss_map = F.mse_loss(pred_f, target_f, reduction="none")
    elif loss_type in ("l1", "mae"):
        loss_map = F.l1_loss(pred_f, target_f, reduction="none")
    elif loss_type in ("huber", "smooth_l1"):
        delta = _huber_delta_for_t(t, huber_c, huber_schedule, huber_snr_clamp_max)
        err = (pred_f - target_f).abs()
        if not torch.is_tensor(delta):
            delta_t = torch.tensor(float(delta), device=err.device, dtype=err.dtype)
        else:
            delta_t = delta.to(device=err.device, dtype=err.dtype)
        if loss_type == "huber":
            loss_map = torch.where(
                err < delta_t,
                0.5 * err.square(),
                delta_t * (err - 0.5 * delta_t),
            )
        else:
            loss_map = torch.where(
                err < delta_t,
                0.5 * err.square() / delta_t,
                err - 0.5 * delta_t,
            )
    else:
        logger.warning(f"Unknown loss_type={loss_type!r}; falling back to mse")
        loss_map = F.mse_loss(pred_f, target_f, reduction="none")

    return loss_map.view(loss_map.shape[0], -1).mean(dim=1)


def masked_token_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                      loss_type: str = "mse", huber_c: float = 0.1,
                      huber_schedule: str = "constant",
                      t: torch.Tensor | None = None,
                      huber_snr_clamp_max: float = 10.0) -> torch.Tensor:
    """Return per-sample token loss, ignoring padded FiT tokens.

    pred/target: (B, N, C)
    mask: (B, N), with non-zero entries marking valid tokens.
    """
    pred_f = pred.float()
    target_f = target.float()
    loss_type = (loss_type or "mse").lower()
    if loss_type in ("mse", "l2"):
        loss_map = F.mse_loss(pred_f, target_f, reduction="none")
    elif loss_type in ("l1", "mae"):
        loss_map = F.l1_loss(pred_f, target_f, reduction="none")
    elif loss_type in ("huber", "smooth_l1"):
        err = (pred_f - target_f).abs()
        delta = _huber_delta_for_t(t, huber_c, huber_schedule, huber_snr_clamp_max)
        if not torch.is_tensor(delta):
            delta_t = torch.tensor(float(delta), device=err.device, dtype=err.dtype)
        else:
            delta_t = delta.to(device=err.device, dtype=err.dtype)
            if delta_t.ndim > 1:
                delta_t = delta_t.reshape(delta_t.shape[0], -1).mean(dim=1)
            delta_t = delta_t.view(-1, 1, 1)
        if loss_type == "huber":
            loss_map = torch.where(
                err < delta_t,
                0.5 * err.square(),
                delta_t * (err - 0.5 * delta_t),
            )
        else:
            loss_map = torch.where(
                err < delta_t,
                0.5 * err.square() / delta_t,
                err - 0.5 * delta_t,
            )
    else:
        logger.warning(f"Unknown loss_type={loss_type!r}; falling back to mse")
        loss_map = F.mse_loss(pred_f, target_f, reduction="none")

    token_loss = loss_map.mean(dim=-1)
    valid = (mask.float() > 0).to(token_loss.dtype)
    weighted = token_loss * valid
    denom = valid.sum(dim=1).clamp(min=1.0)
    out = weighted.sum(dim=1) / denom
    empty = valid.sum(dim=1) <= 0
    if bool(empty.any()):
        out = out.masked_fill(empty, 0.0)
    return out


def contrastive_flow_matching_neg(pred: torch.Tensor, target: torch.Tensor,
                                  t: torch.Tensor | None = None, loss_type: str = "mse",
                                  huber_c: float = 0.1, huber_schedule: str = "constant",
                                  huber_snr_clamp_max: float = 10.0,
                                  mask: torch.Tensor | None = None) -> torch.Tensor:
    """ΔFM（Contrastive Flow Matching, arxiv:2506.05350）的负样本 per-sample loss。

    返回 ||v_pred_i - target_j||²（j = batch 内另一样本，用 roll(shifts=1) 取，
    bs>1 时保证 j≠i）。训练时 per_sample ← per_sample - λ·(本函数返回值)，形成排斥项，
    反"回归条件均值→发灰/材质难分"。

    不改噪声分布、复用已算好的 pred → 零额外前向。bs<=1 时返回全 0（无可配对样本）。
    mask 给 FiT packed token 路径用（与 masked_token_loss 对齐）；None 走 dense 路径。
    """
    bs = pred.shape[0]
    if bs <= 1:
        return pred.new_zeros((bs,), dtype=torch.float32)
    perm = torch.roll(torch.arange(bs, device=pred.device), shifts=1)
    target_neg = target.index_select(0, perm)
    if mask is not None:
        return masked_token_loss(
            pred, target_neg, mask, loss_type=loss_type, huber_c=huber_c,
            huber_schedule=huber_schedule, t=t, huber_snr_clamp_max=huber_snr_clamp_max,
        )
    return per_sample_loss(
        pred, target_neg, loss_type=loss_type, huber_c=huber_c,
        huber_schedule=huber_schedule, t=t, huber_snr_clamp_max=huber_snr_clamp_max,
    )


def per_sample_highfreq_loss(pred: torch.Tensor, target: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """Return per-sample high-frequency residual energy for latent tensors.

    This is only used as an adaptive timestep sampling signal; it does not
    change the training objective or gradients.
    """
    diff = (pred.float() - target.float())
    if diff.ndim < 4:
        return diff.square().view(diff.shape[0], -1).mean(dim=1)

    b = diff.shape[0]
    h, w = diff.shape[-2], diff.shape[-1]
    k = max(int(kernel_size or 5), 1)
    if k % 2 == 0:
        k += 1
    if min(h, w) <= 1 or k <= 1:
        return diff.square().view(b, -1).mean(dim=1)

    flat = diff.reshape(-1, 1, h, w)
    blur = F.avg_pool2d(flat, kernel_size=k, stride=1, padding=k // 2, count_include_pad=False)
    high = flat - blur
    return high.square().reshape(b, -1).mean(dim=1)


def adaptive_timestep_metric_signal(
    per_sample: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
    metric: str = "raw",
    highfreq_weight: float = 0.25,
) -> torch.Tensor:
    """Build the detached per-sample signal used by AdaptiveTimestepSampler."""
    metric = (metric or "raw").lower()
    raw = per_sample.detach().float()
    if metric in ("raw", "entropy_rate"):
        return raw
    highfreq = per_sample_highfreq_loss(pred.detach(), target.detach())
    if metric == "highfreq":
        return highfreq
    if metric == "mixed":
        return raw + max(float(highfreq_weight or 0.0), 0.0) * highfreq
    raise ValueError(f"Unknown adaptive_timestep_metric: {metric}")


def compute_loss_weight(t: torch.Tensor, scheme: str = "none", min_snr_gamma: float = 0.0,
                        weight_cap_ratio: float = 0.0,
                        detail_inv_t_min: float = 1.0, detail_inv_t_max: float = 5.0):
    """根据 scheme 返回每样本的 loss 权重 (B,)。

    Flow Matching CONST 调度下：alpha_t = 1 - t，sigma_t = t；SNR(t) = ((1-t)/t)^2

    scheme:
      - "none":          全 1 权重
      - "min_snr":       w = min(gamma / SNR, 1)，下调"几乎无噪声/高 SNR"的简单步
      - "max_snr_inv":   w = min(SNR / gamma, 1)，下调极高噪声/低 SNR 步（少用）
      - "logit_normal":  按 logit-normal 概率密度的倒数加权（debias 采样偏置）
      - "sigma_sqrt":    w = sqrt(sigma) = sqrt(t)，缓解 t→0 处梯度爆炸
      - "sigma_sqrt_sd3":SD3 论文 Eq.6 原始 σ^-2 权重；max=1000，**仅用于大 batch (>=64)**。
                         小 batch + Prodigy 会因单样本主导导致 d 估计崩坏（不学习）。
                         小 batch 想要细节强化请用 "detail_inv_t" 或 "cosmap"。
      - "detail_inv_t":  w = 1/t，clamp 到 [1, 5]；这是一个温和的细节端强化，配合
                         weight_cap_ratio (默认 5) 时单 batch 内 max/min 比 ≤ 5×。
                         小 batch + Prodigy 兼容，是 sigma_sqrt_sd3 的实用替代。
      - "cosmap":        SD3 cosmap weighting，对中间 t 更友好（max/min ≈ 1.81×）

    weight_cap_ratio: 单个 batch 内最大权重 / 最小权重的硬上限。0=禁用。
                       建议小 batch 训练设 5-10，避免单样本主导破坏 Prodigy 的 d 估计。
    """
    scheme = (scheme or "none").lower()
    if scheme == "none":
        return torch.ones_like(t)

    eps = 1e-4
    t_c = t.clamp(eps, 1 - eps)

    if scheme == "min_snr":
        if min_snr_gamma <= 0:
            return torch.ones_like(t)
        snr = ((1 - t_c) / t_c) ** 2
        w = torch.minimum(float(min_snr_gamma) / snr, torch.ones_like(t_c))
    elif scheme == "max_snr_inv":
        if min_snr_gamma <= 0:
            return torch.ones_like(t)
        snr = ((1 - t_c) / t_c) ** 2
        w = torch.minimum(snr / float(min_snr_gamma), torch.ones_like(t_c))
    elif scheme == "logit_normal":
        w = (t_c * (1 - t_c)).clamp(min=eps)
    elif scheme == "sigma_sqrt":
        # 【遗留】这是 sqrt(t)，不是 SD3 论文 Eq.6 的 sigma^-2。
        w = t_c.sqrt()
    elif scheme == "sigma_sqrt_sd3":
        # SD3 paper Eq. 6: w(sigma) = sigma^-2
        # ⚠️ 仅适合大 batch (>=64)。小 batch + Prodigy 会让单样本独占 loss → d 估计崩坏。
        w = (t_c ** -2).clamp(max=1000.0)
    elif scheme == "detail_inv_t":
        # 温和细节端强化：w = 1/t 但 clamp 到可配置上下限（默认 [1, 5]）；与小 batch + Prodigy 兼容。
        # 雾蒙蒙画风的 dataset 把上限调到 3 左右能显著减少细节溶解（hazy style_profile 的默认行为）。
        lo = float(detail_inv_t_min or 1.0)
        hi = float(detail_inv_t_max or 5.0)
        if lo > hi:
            lo, hi = hi, lo
        w = (1.0 / t.clamp(min=eps)).clamp(min=lo, max=hi)
    elif scheme == "cosmap":
        bot = 1 - 2 * t_c + 2 * t_c ** 2
        w = 2.0 / (math.pi * bot)
    else:
        return torch.ones_like(t)

    # batch 内 max/min 比上限：防止单样本主导（破坏 Prodigy d 估计）。
    if weight_cap_ratio and weight_cap_ratio > 1.0:
        w_min = w.min().clamp(min=eps)
        w_max_allowed = w_min * float(weight_cap_ratio)
        w = w.clamp(max=w_max_allowed)

    return w


def apply_loss_weighting(per_sample: torch.Tensor, t: torch.Tensor, cfg: LossConfig) -> torch.Tensor:
    if cfg.weighting_scheme == "none":
        return per_sample.mean()
    w = compute_loss_weight(
        t.float(),
        scheme=cfg.weighting_scheme,
        min_snr_gamma=cfg.min_snr_gamma,
        weight_cap_ratio=cfg.weight_cap_ratio,
        detail_inv_t_min=cfg.detail_inv_t_min,
        detail_inv_t_max=cfg.detail_inv_t_max,
    )
    w = w / w.mean().clamp(min=1e-6)
    return (per_sample * w).mean()


def apply_loss_weighting_per_sample(per_sample: torch.Tensor, t: torch.Tensor,
                                    cfg: LossConfig,
                                    normalize_weights: bool = False) -> torch.Tensor:
    """Return weighted per-sample losses without reducing across the batch.

    The regular training path keeps the historical per-micro-batch weight
    normalization in ``apply_loss_weighting``. Sample-window accumulation needs
    raw per-sample values so ARB micro-batch boundaries do not renormalize the
    timestep weights independently.
    """
    if cfg.weighting_scheme == "none":
        return per_sample
    w = compute_loss_weight(
        t.float(),
        scheme=cfg.weighting_scheme,
        min_snr_gamma=cfg.min_snr_gamma,
        weight_cap_ratio=cfg.weight_cap_ratio,
        detail_inv_t_min=cfg.detail_inv_t_min,
        detail_inv_t_max=cfg.detail_inv_t_max,
    )
    if normalize_weights:
        w = w / w.mean().clamp(min=1e-6)
    return per_sample * w


# ============================================================================
# Grad norm + forward helpers
# ============================================================================

def compute_grad_norm(parameters) -> float:
    """L2 范数（全局），用 foreach 一次性算所有 grad，避免逐参数 `.item()` 同步。

    LoKr 注入了 100+ 个小 Linear，旧实现每个 grad 都跑一次 .item()，每次 grad_norm
    日志都要触发 100+ 次 GPU↔CPU 同步；新实现只在最后做一次同步。
    """
    grads = [p.grad.detach() for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    # 任一 grad 含 NaN/Inf 直接报 inf（与旧实现语义一致）。
    finite_check = torch.stack([torch.isfinite(g).all() for g in grads])
    if not bool(finite_check.all()):
        return float("inf")
    # torch._foreach_norm 在新版 PyTorch 上是融合 kernel，比 Python loop 快很多。
    per_grad_norms = torch._foreach_norm(grads, 2.0)
    total = torch.linalg.vector_norm(torch.stack([n.to(torch.float32) for n in per_grad_norms]))
    return float(total.item())


_BLOCK_ACCEPTS_PAD_MASK_CACHE: "dict[int, bool]" = {}


def _block_accepts_padding_mask(block) -> bool:
    """Detect once per (block class, model instance) 是否接受 padding_mask kwarg。

    旧实现每个 block 每步 forward 都 try/except TypeError 兜底 —— 对 36 blocks ×
    几千 steps = 几十万次 try/except，虽然单次开销小，但累积非零。

    用 class id 做 cache key（同一模型类的所有 block 行为一致）。
    """
    key = id(type(block))
    cached = _BLOCK_ACCEPTS_PAD_MASK_CACHE.get(key)
    if cached is not None:
        return cached
    import inspect
    try:
        sig = inspect.signature(block.forward)
        accepts = ("padding_mask" in sig.parameters) or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
    except (TypeError, ValueError):
        # 无法 introspect（C++ binding 等）时保守地试一次 + 兜底
        accepts = True
    _BLOCK_ACCEPTS_PAD_MASK_CACHE[key] = bool(accepts)
    return bool(accepts)


def forward_with_optional_checkpoint(model, latents, timesteps, cross, padding_mask, use_checkpoint=False):
    """带可选梯度检查点的前向传播（per-block checkpoint 策略）。

    ⚠ 关于策略选择的历史教训：
    曾经一版实现把整个 `model.forward` 包进**单个** checkpoint 调用，理由是简单且永远不漏参数。
    后来发现对于大模型这意味着 backward 时要一次性重放整个 forward 的所有激活 →
    峰值显存 ≈ N × (单 block 激活)，把训练显存推到无法接受的高度（实测 10GB → 70GB 量级）。

    本实现回退到 per-block checkpoint：每个 transformer block 单独 checkpoint，峰值激活
    ≈ 1 × (单 block 激活)。同时把 `padding_mask` 显式透传给每个 block —— 这是上一版整体
    checkpoint 当初引入的本意（旧 per-block 实现漏传了 padding_mask）。

    ★ block 是否接受 padding_mask kwarg 用 inspect 一次性 introspect 并 cache，
       不再每个 forward 都 try/except TypeError。
    """
    if not use_checkpoint:
        return model(latents, timesteps, cross, padding_mask=padding_mask)
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

    for block in model.blocks:
        accepts_pad = _block_accepts_padding_mask(block)
        if accepts_pad:
            def custom_forward(x, blk=block):
                return blk(x, t_embedding, cross, padding_mask=padding_mask, **block_kwargs)
        else:
            def custom_forward(x, blk=block):
                return blk(x, t_embedding, cross, **block_kwargs)
        x_B_T_H_W_D = checkpoint(custom_forward, x_B_T_H_W_D, use_reentrant=False)

    x_B_T_H_W_O = model.final_layer(x_B_T_H_W_D, t_embedding, adaln_lora_B_T_3D=adaln_lora)
    return model.unpatchify(x_B_T_H_W_O)


def forward_packed_with_optional_checkpoint(
    model,
    tokens,
    timesteps,
    cross,
    grid,
    mask,
    size,
    use_checkpoint=False,
):
    """Forward packed FiT tokens with the same per-block checkpoint strategy.

    A whole-model checkpoint makes backward recompute the entire packed
    transformer at once. For long native FiT sequences, keeping the checkpoint
    boundary at each transformer block is much friendlier to peak VRAM.
    """
    if not use_checkpoint:
        return model.forward_packed_tokens(tokens, timesteps, cross, grid, mask, size)

    expected = model.x_embedder.proj[1].in_features
    if tokens.shape[-1] < expected:
        tokens = F.pad(tokens, (0, expected - tokens.shape[-1]))
    elif tokens.shape[-1] > expected:
        raise ValueError(
            f"packed tokens have dim={tokens.shape[-1]}, but x_embedder expects {expected}"
        )
    x = model.x_embedder.proj[1](tokens)

    if timesteps.ndim == 1:
        timesteps = timesteps.unsqueeze(1)
    t_embedding, adaln_lora = model.t_embedder(timesteps)
    t_embedding = model.t_embedding_norm(t_embedding)

    model.affline_scale_log_info = {"t_embedding_B_T_D": t_embedding.detach()}
    model.affline_emb = t_embedding
    model.crossattn_emb = cross

    rope_emb = model._packed_rope_from_grid(grid)
    for block in model.blocks:
        def custom_forward(x_in, blk=block):
            return blk.forward_tokens(
                x_in,
                t_embedding,
                cross,
                rope_emb_L_1_1_D=rope_emb,
                token_mask=mask,
                adaln_lora_B_T_3D=adaln_lora,
            )

        x = checkpoint(custom_forward, x, use_reentrant=False)

    out = model.final_layer.forward_tokens(x, t_embedding, adaln_lora_B_T_3D=adaln_lora)
    out = model._output_tokens_to_patch_tokens(out, size)
    return out * mask.to(dtype=out.dtype).unsqueeze(-1)


def validate_compile_requirements(torch_compile: bool, fit_packed_training: bool,
                                  token_bucket: bool, module_dropout: float = 0.0) -> None:
    """Fail fast if torch_compile is requested without its prerequisites.

    The compiled fast path runs through the packed-token forward
    (``block.forward_tokens``), so it requires ``fit_packed_training``; and it
    only pays off when the packed sequence length is fixed, which requires
    ``token_bucket`` (constant / N-token bucketing). ``module_dropout`` must be 0:
    its ``torch.rand().item()`` per-module drop is a data-dependent branch that
    graph-breaks every block under compile. No-op when compile is off.
    """
    if not torch_compile:
        return
    if not fit_packed_training:
        raise RuntimeError(
            "torch_compile=true requires fit_packed_training=true: the compiled fast "
            "path runs through the packed-token forward (block.forward_tokens)."
        )
    if not token_bucket:
        raise RuntimeError(
            "torch_compile=true requires token_bucket=true so the packed sequence "
            "length is fixed across the run (else torch.compile recompiles per shape)."
        )
    if module_dropout and float(module_dropout) > 0.0:
        raise RuntimeError(
            "torch_compile=true is incompatible with module_dropout>0: the per-module "
            "drop uses torch.rand().item(), a data-dependent branch that graph-breaks "
            "every block under compile (negating the speedup; fatal under CUDAGraph "
            "modes). Set module_dropout: 0.0 for the compile path "
            "(rank_dropout / lora_dropout are compile-safe and may stay)."
        )
