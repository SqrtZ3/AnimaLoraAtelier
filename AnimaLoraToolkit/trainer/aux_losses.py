"""辅助 loss：Spectral Regularization + Perceptual (LPIPS + DINOv2)。

Spectral (arxiv:2603.02447):
  在 latent space 上对 predicted x₀ 与 target x₀ 做 FFT 振幅 L1 匹配（可选附加
  Haar wavelet 系数 L1）。纯 tensor 运算，零额外模型/显存。直接惩罚高频细节缺失。

Perceptual (arxiv:2602.02493 PixelGen):
  把 predicted x₀ 通过 VAE decoder 解码到 pixel space，再用 VGG-LPIPS + DINOv2-B
  做感知相似度。VGG 特征对纹理敏感（眼/材质），DINO patch token 对语义结构敏感
  （脸型/比例）。梯度会从 pixel-space loss 一路反传到 LoRA 参数。

Flow Matching x₀ 恢复（线性 schedule）：
  x_t = (1-t)·x₀ + t·ε
  velocity = ε - x₀
  ⇒ x₀ = x_t - t·velocity

两个 loss 都有 t-gate：仅在 t < t_gate 的低噪声区间启用，因为高噪声下
predicted x₀ 偏差过大、感知/频域指标无意义。
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================================
# Config
# ============================================================================

@dataclass(frozen=True)
class AuxLossConfig:
    """辅助 loss 总开关 + 权重 + gate 配置。

    所有字段默认值都让 enabled=False 时彻底是 no-op，向后兼容。
    """
    # ---- Spectral (frequency-domain matching) ----
    spectral_enabled: bool = False
    spectral_lambda: float = 0.05
    spectral_use_wavelet: bool = False
    spectral_wavelet_lambda: float = 0.05
    spectral_t_gate: float = 0.7

    # ---- Perceptual (LPIPS + DINO via VAE decode) ----
    perceptual_enabled: bool = False
    perceptual_lambda_lpips: float = 0.1
    perceptual_lambda_dino: float = 0.01
    perceptual_t_gate: float = 0.7
    perceptual_lpips_net: str = "vgg"
    perceptual_dino_local_path: str = ""

    @property
    def any_enabled(self) -> bool:
        return self.spectral_enabled or self.perceptual_enabled

    @property
    def needs_vae_decoder(self) -> bool:
        return self.perceptual_enabled


def build_aux_loss_config(args) -> AuxLossConfig:
    """从 argparse Namespace 构造 AuxLossConfig。"""
    return AuxLossConfig(
        spectral_enabled=bool(getattr(args, "aux_spectral_enabled", False)),
        spectral_lambda=float(getattr(args, "aux_spectral_lambda", 0.05) or 0.0),
        spectral_use_wavelet=bool(getattr(args, "aux_spectral_use_wavelet", False)),
        spectral_wavelet_lambda=float(getattr(args, "aux_spectral_wavelet_lambda", 0.05) or 0.0),
        spectral_t_gate=float(getattr(args, "aux_spectral_t_gate", 0.7) or 0.7),
        perceptual_enabled=bool(getattr(args, "aux_perceptual_enabled", False)),
        perceptual_lambda_lpips=float(getattr(args, "aux_perceptual_lambda_lpips", 0.1) or 0.0),
        perceptual_lambda_dino=float(getattr(args, "aux_perceptual_lambda_dino", 0.01) or 0.0),
        perceptual_t_gate=float(getattr(args, "aux_perceptual_t_gate", 0.7) or 0.7),
        perceptual_lpips_net=str(getattr(args, "aux_perceptual_lpips_net", "vgg") or "vgg"),
        perceptual_dino_local_path=str(getattr(args, "aux_perceptual_dino_local_path", "") or ""),
    )


# ============================================================================
# x₀ recovery
# ============================================================================

def recover_x0_from_velocity(noisy: torch.Tensor, t: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """从 flow matching 的 velocity 预测恢复 predicted x₀。

    线性 schedule:  x_t = (1-t)·x₀ + t·ε ,  velocity = ε - x₀
    ⇒ x₀ = x_t - t·velocity

    Args:
        noisy: x_t, (B, C[, T], H, W)；与 anima_train.py 的 5D latent 一致
        t:     (B,) timesteps ∈ [0, 1]
        pred:  模型输出 (predicted velocity)，与 noisy 同 shape
    Returns:
        x₀_pred, 与 noisy 同 shape，强制 fp32（避免 bf16 减法误差）
    """
    n_f = noisy.float()
    p_f = pred.float()
    # broadcast t: (B,) → (B, 1, 1, ...)，维度数与 noisy 匹配
    t_exp = t.float().view(-1, *([1] * (noisy.ndim - 1)))
    return n_f - t_exp * p_f


# ============================================================================
# Spectral loss
# ============================================================================

def _haar_wavelet_coefs(x: torch.Tensor) -> torch.Tensor:
    """单层 Haar 小波分解，返回 [LL, LH, HL, HH] 拼到 channel 维上。

    支持 4D (B, C, H, W) 与 5D (B, C, T, H, W)。空间维 stride=2 下采样。
    """
    if x.ndim == 5:
        B, C, T, H, W = x.shape
        x_flat = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        reshape_back = (B, T, C, H, W)
    elif x.ndim == 4:
        x_flat = x
        B, C, H, W = x.shape
        T = 1
        reshape_back = None
    else:
        raise ValueError(f"_haar_wavelet_coefs: 仅支持 4D/5D 输入，收到 ndim={x.ndim}")

    # 2x2 Haar 滤波器 × 4 个方向
    # LL：低-低（平滑分量）   LH：水平边缘
    # HL：垂直边缘            HH：对角细节
    haar = torch.tensor(
        [
            [[1.0, 1.0], [1.0, 1.0]],     # LL
            [[1.0, 1.0], [-1.0, -1.0]],   # LH
            [[1.0, -1.0], [1.0, -1.0]],   # HL
            [[1.0, -1.0], [-1.0, 1.0]],   # HH
        ],
        dtype=x_flat.dtype, device=x_flat.device,
    ).unsqueeze(1) * 0.5  # (4, 1, 2, 2)

    # grouped conv：每个 input channel 独立卷 4 个 Haar 滤波器
    haar_g = haar.repeat(C, 1, 1, 1)  # (4*C, 1, 2, 2)

    coef = F.conv2d(x_flat, haar_g, stride=2, groups=C)  # (B*T, 4*C, H/2, W/2)

    if reshape_back is not None:
        # 把 T 维放回去：(B*T, 4*C, h, w) → (B, T, 4*C, h, w) → (B, 4*C, T, h, w)
        coef = coef.view(B, T, 4 * C, coef.shape[-2], coef.shape[-1])
        coef = coef.permute(0, 2, 1, 3, 4).contiguous()

    return coef


def spectral_loss(x0_pred: torch.Tensor, x0_target: torch.Tensor,
                  t: torch.Tensor, cfg: AuxLossConfig) -> torch.Tensor:
    """FFT 振幅 L1 匹配 + 可选 Haar wavelet 系数 L1。

    t-gate：仅 t < cfg.spectral_t_gate 的样本参与；其它样本对 batch loss 贡献为 0。
    返回标量（已按有效样本数归一）。
    """
    if not cfg.spectral_enabled:
        return x0_pred.new_zeros((), dtype=torch.float32)

    # gating mask：(B,) 0/1，只保留 t < gate 的样本
    mask = (t.float() < float(cfg.spectral_t_gate)).float()
    if mask.sum() == 0:
        return x0_pred.new_zeros((), dtype=torch.float32)

    pred_f = x0_pred.float()
    target_f = x0_target.float()

    # 2D FFT 在最后两个空间维（H, W）；4D/5D 都适用
    fft_pred = torch.fft.fft2(pred_f, dim=(-2, -1), norm="ortho")
    # target 不需要梯度
    with torch.no_grad():
        fft_target = torch.fft.fft2(target_f, dim=(-2, -1), norm="ortho")

    # L1 on amplitude spectrum
    amp_diff = (fft_pred.abs() - fft_target.abs()).abs()
    fft_per_sample = amp_diff.view(amp_diff.shape[0], -1).mean(dim=1)

    total = fft_per_sample

    if cfg.spectral_use_wavelet:
        coef_pred = _haar_wavelet_coefs(pred_f)
        with torch.no_grad():
            coef_target = _haar_wavelet_coefs(target_f)
        w_diff = (coef_pred - coef_target).abs()
        w_per_sample = w_diff.view(w_diff.shape[0], -1).mean(dim=1)
        total = total + float(cfg.spectral_wavelet_lambda) * w_per_sample

    # mask 加权求平均（只对参与样本归一）
    return (total * mask).sum() / mask.sum().clamp(min=1.0)


# ============================================================================
# Perceptual loss (LPIPS + DINOv2)
# ============================================================================

class PerceptualLossModule(torch.nn.Module):
    """LPIPS (VGG/Alex) + DINOv2-B 感知 loss，通过冻结 VAE decoder 解码到 pixel space。

    子模型（VAE / LPIPS / DINO）全部冻结，但 forward 仍是可微分的——梯度从
    pixel-space loss 经 LPIPS/DINO 网络反传到解码后像素，再经 VAE decoder 反传
    回 x0_pred，最终汇入 LoRA 参数。

    ⚠ 使用前提：anima_train.py 不能把 VAE.model 搬到 CPU。perceptual_enabled=True
    时训练主循环开始前需要跳过 VAE CPU offload。
    """

    def __init__(self, vae_wrapper, cfg: AuxLossConfig, device, compute_dtype=torch.bfloat16):
        super().__init__()
        self.cfg = cfg
        self.vae_wrapper = vae_wrapper  # 拿 .model 和 .scale，不作为子 module（避免被参数注册）
        self.compute_dtype = compute_dtype
        self.device = device

        # ★ VAE 主网络在我们这里只做 forward，不应该有梯度（不在优化器参数组里也不会更新，
        #    但 backward 时若 requires_grad=True 会无谓累积梯度占显存）。
        if hasattr(vae_wrapper, "model"):
            for p in vae_wrapper.model.parameters():
                p.requires_grad_(False)
            vae_wrapper.model.eval()

        # ---- LPIPS ----
        try:
            import lpips
        except ImportError as e:
            raise ImportError(
                "Perceptual loss 需要 lpips 包：pip install lpips\n"
                "（首次运行会下载 VGG16 / AlexNet / SqueezeNet 预训练权重到 ~/.cache/torch/）"
            ) from e

        self.lpips_fn = lpips.LPIPS(net=cfg.perceptual_lpips_net, verbose=False)
        self.lpips_fn.eval()
        for p in self.lpips_fn.parameters():
            p.requires_grad_(False)
        self.lpips_fn = self.lpips_fn.to(device)
        logger.info("LPIPS 加载完成 (net=%s)", cfg.perceptual_lpips_net)

        # ---- DINOv2 ----
        self.use_dino = float(cfg.perceptual_lambda_dino) > 0
        if self.use_dino:
            self.dino = self._load_dino(cfg.perceptual_dino_local_path, device)
            self.dino.eval()
            for p in self.dino.parameters():
                p.requires_grad_(False)
            logger.info("DINOv2-B 加载完成")
        else:
            self.dino = None
            logger.info("DINOv2 已跳过（perceptual_lambda_dino=0）")

        # ImageNet 归一化常量（用于 DINO 输入）
        self.register_buffer(
            "dino_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "dino_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    @staticmethod
    def _load_dino(local_path: str, device):
        """加载 DINOv2-B。优先级：local_path > torch.hub。

        local_path 支持两种形态：
          - 目录：clone 自 facebookresearch/dinov2 的本地仓库，走 torch.hub.load(source='local')
          - 文件：state_dict (.pth / .pt / .safetensors)；先用 torch.hub 拿模型架构再 load_state_dict
        """
        local_path = (local_path or "").strip()
        if local_path:
            p = Path(local_path)
            if not p.exists():
                raise FileNotFoundError(f"DINOv2 local_path 不存在: {local_path}")

            if p.is_dir():
                logger.info("DINOv2 从本地仓库加载: %s", local_path)
                dino = torch.hub.load(
                    str(p), "dinov2_vitb14",
                    source="local", pretrained=True, trust_repo=True,
                )
            else:
                logger.info("DINOv2 从 state_dict 加载: %s", local_path)
                # 先拿架构（无权重）
                dino = torch.hub.load(
                    "facebookresearch/dinov2", "dinov2_vitb14",
                    pretrained=False, trust_repo=True,
                )
                if local_path.endswith(".safetensors"):
                    from safetensors.torch import load_file
                    sd = load_file(local_path, device="cpu")
                else:
                    sd = torch.load(local_path, map_location="cpu")
                if isinstance(sd, dict) and "state_dict" in sd:
                    sd = sd["state_dict"]
                missing, unexpected = dino.load_state_dict(sd, strict=False)
                if missing or unexpected:
                    logger.warning(
                        "DINOv2 load_state_dict 不完全匹配：missing=%d unexpected=%d "
                        "（前 3 项 missing: %s）",
                        len(missing), len(unexpected), missing[:3],
                    )
        else:
            logger.info("DINOv2 从 torch.hub 加载 (facebookresearch/dinov2)…")
            dino = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitb14",
                pretrained=True, trust_repo=True,
            )

        return dino.to(device)

    def _decode_to_pixel(self, x0_latent: torch.Tensor, with_grad: bool) -> torch.Tensor:
        """通过冻结 VAE decoder 解码 latent → pixel [-1, 1]。

        with_grad=False 给 target 用（target 是固定的 ground truth，不需要梯度）。
        with_grad=True 给 pred 用（梯度要反传回去）。
        Anima latents 是 5D (B, 16, T, H, W)；4D 输入会自动 unsqueeze(2)。

        ★ 不在这里强制 .float()：保持 compute_dtype (bf16) 直到 LPIPS/DINO forward 完成。
          下游 LPIPS 内的 VGG forward 也用 bf16 跑（tensor core 加速），最后梯度回流时
          autograd 会自动 upcast 到 fp32。强制 fp32 会让 VGG forward 变慢且爆内存。
        """
        if x0_latent.ndim == 4:
            z = x0_latent.unsqueeze(2)
        else:
            z = x0_latent
        # VAE 主网络是 compute_dtype（bf16），把 z 也转到同 dtype 避免隐式 cast 开销
        z = z.to(dtype=self.compute_dtype)

        ctx = contextlib.nullcontext() if with_grad else torch.no_grad()
        with ctx, torch.autocast("cuda", dtype=self.compute_dtype):
            pixels = self.vae_wrapper.model.decode(z, self.vae_wrapper.scale)
        # decoder 输出 (B, 3, T_out, H_out, W_out)，单图情况 T_out=1
        if pixels.ndim == 5 and pixels.shape[2] == 1:
            pixels = pixels.squeeze(2)
        # WAN VAE 的 decoder 末端是 tanh-like，输出近似 [-1, 1]；clamp 防越界
        # 保持 compute_dtype；LPIPS/DINO 内部自己 autocast
        return pixels.clamp(-1.0, 1.0)

    def forward(self, x0_pred: torch.Tensor, x0_target: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        """计算 λ_lpips * LPIPS + λ_dino * (1 - cos_sim_DINO)，对 t < t_gate 样本做平均。"""
        if not self.cfg.perceptual_enabled:
            return x0_pred.new_zeros((), dtype=torch.float32)

        mask = (t.float() < float(self.cfg.perceptual_t_gate)).float()
        if mask.sum() == 0:
            return x0_pred.new_zeros((), dtype=torch.float32)

        # 解码到 pixel space（保持 compute_dtype，下游 LPIPS/DINO 在 bf16 下跑）
        pixels_pred = self._decode_to_pixel(x0_pred, with_grad=True)
        pixels_target = self._decode_to_pixel(x0_target, with_grad=False)

        # ---- LPIPS（在 bf16 autocast 内跑 VGG，速度 ~2× 且省显存）----
        with torch.autocast("cuda", dtype=self.compute_dtype):
            l_lpips = self.lpips_fn(pixels_pred, pixels_target)
        # 返回 (B, 1, 1, 1)，规范成 (B,)，再 upcast 到 fp32 与主 loss 同 dtype
        l_lpips = l_lpips.view(l_lpips.shape[0]).float()
        per_sample = float(self.cfg.perceptual_lambda_lpips) * l_lpips

        # ---- DINOv2 cosine similarity ----
        if self.use_dino:
            # [-1, 1] → [0, 1] → ImageNet normalize → resize to 224
            # mean/std buffers 是 fp32，与 pixels 做减除会上抬到 fp32；这里我们想保持 bf16，
            # 所以把 mean/std 转成 pixels 的 dtype
            mean = self.dino_mean.to(dtype=pixels_pred.dtype)
            std = self.dino_std.to(dtype=pixels_pred.dtype)
            p_pred = ((pixels_pred + 1.0) * 0.5).clamp(0.0, 1.0)
            p_target = ((pixels_target + 1.0) * 0.5).clamp(0.0, 1.0)
            p_pred = (p_pred - mean) / std
            p_target = (p_target - mean) / std
            p_pred = F.interpolate(p_pred, size=224, mode="bilinear", align_corners=False)
            p_target = F.interpolate(p_target, size=224, mode="bilinear", align_corners=False)

            with torch.autocast("cuda", dtype=self.compute_dtype):
                feat_pred = self.dino.forward_features(p_pred)["x_norm_patchtokens"]
                with torch.no_grad():
                    feat_target = self.dino.forward_features(p_target)["x_norm_patchtokens"]

            # cosine similarity 在 fp32 上做（数值更稳）；patch 维度平均
            cos = F.cosine_similarity(feat_pred.float(), feat_target.float(), dim=-1)  # (B, N)
            l_dino = (1.0 - cos).mean(dim=-1)  # (B,)
            per_sample = per_sample + float(self.cfg.perceptual_lambda_dino) * l_dino

        return (per_sample * mask).sum() / mask.sum().clamp(min=1.0)


# ============================================================================
# 一些便利的 logging
# ============================================================================

def summary_aux_loss_config(cfg: AuxLossConfig) -> str:
    """返回单行可读的 aux loss 配置摘要，供训练日志用。"""
    parts = []
    if cfg.spectral_enabled:
        s = f"spectral(λ={cfg.spectral_lambda:.3f},gate={cfg.spectral_t_gate:.2f}"
        if cfg.spectral_use_wavelet:
            s += f",+wavelet({cfg.spectral_wavelet_lambda:.3f})"
        s += ")"
        parts.append(s)
    if cfg.perceptual_enabled:
        parts.append(
            f"perceptual(λ_lpips={cfg.perceptual_lambda_lpips:.3f},"
            f"λ_dino={cfg.perceptual_lambda_dino:.3f},"
            f"gate={cfg.perceptual_t_gate:.2f},"
            f"net={cfg.perceptual_lpips_net})"
        )
    return " + ".join(parts) if parts else "disabled"
