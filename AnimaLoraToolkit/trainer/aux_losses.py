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
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)
_SPECTRAL_AMP_EPS = 1e-12


def _complex_abs_stable(x: torch.Tensor) -> torch.Tensor:
    return (x.real.square() + x.imag.square() + _SPECTRAL_AMP_EPS).sqrt()


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
    # ★ 给 LPIPS 内部 VGG / DINOv2 都重定向到这个本地目录（设为 TORCH_HOME）。
    #   目录结构必须是：<cache_dir>/hub/checkpoints/{vgg16-*.pth, dinov2_vitb14_pretrain.pth}
    #                 <cache_dir>/hub/facebookresearch_dinov2_main/ (DINOv2 仓库代码)
    #   设了这个后，torch.hub.load 和 torchvision.models.vgg16 都从这里加载，无需联网。
    perceptual_cache_dir: str = ""
    # ★ 把 VAE decode + LPIPS-VGG + DINOv2 这条 forward 用 torch.utils.checkpoint 包起来。
    #   原因：这三个模型都没启用 PyTorch 梯度检查点，激活会一直活到 backward。
    #   1024 分辨率 batch=4 时 VAE decoder + VGG 累计激活可达 15-25 GB，会拖爆 96GB 显存。
    #   开启后 backward 会重放一次这段 forward（compute 翻倍），但显存降到几 GB。
    #   除非你显存非常富裕想榨速度，强烈建议保持 True。
    perceptual_use_checkpoint: bool = True
    # ★ LPIPS 计算前的 pixel 下采样目标边长。0 = 不下采样（用原分辨率）。
    #   1024 训练时建议设 512（显存减半）或 256（再减半）；LPIPS-VGG 在 256-512 仍能可靠
    #   捕获纹理/细节相似度，对 1024 原图的差异不明显。
    perceptual_lpips_size: int = 0

    # ---- Self-Perceptual SFT (arXiv 2401.00110) ----
    # 冻结 DiT 编码栈特征空间距离 ‖f(x0_pred) − f(x0_target)‖²，当带 t_gate 的 SFT 辅助项叠进
    # 主 loss。复用 trainer/ncp.py:perceptual_features（编码器=当前模型、adapter 冻梯度），
    # 零外部模型/无 VAE decode（图盲、廉价：每步 +1~2 截断编码栈前向）。罚均值回归（"不够像"）。
    self_perceptual_enabled: bool = False
    self_perceptual_lambda: float = 0.1
    self_perceptual_t_gate: float = 0.5
    # tap_block：<0 自动取中间块 n//2（画风带）。越小越浅 = 更省/更偏结构-几何；越大 = 更画风特异。
    self_perceptual_tap_block: int = -1
    # encode_t：编码 clean x0 时喂给 DiT 的 timestep（贴数据端、留在训练域 [t_min,1] 内，默认 0.05）。
    self_perceptual_encode_t: float = 0.05

    @property
    def any_enabled(self) -> bool:
        return self.spectral_enabled or self.perceptual_enabled or self.self_perceptual_enabled

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
        perceptual_cache_dir=str(getattr(args, "aux_perceptual_cache_dir", "") or ""),
        perceptual_use_checkpoint=bool(getattr(args, "aux_perceptual_use_checkpoint", True)),
        perceptual_lpips_size=int(getattr(args, "aux_perceptual_lpips_size", 0) or 0),
        self_perceptual_enabled=bool(getattr(args, "aux_self_perceptual_enabled", False)),
        self_perceptual_lambda=float(getattr(args, "aux_self_perceptual_lambda", 0.1) or 0.0),
        self_perceptual_t_gate=float(getattr(args, "aux_self_perceptual_t_gate", 0.5) or 0.5),
        self_perceptual_tap_block=int(getattr(args, "aux_self_perceptual_tap_block", -1)),
        self_perceptual_encode_t=float(getattr(args, "aux_self_perceptual_encode_t", 0.05) or 0.05),
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
    per_sample = spectral_loss_per_sample(x0_pred, x0_target, t, cfg)
    active = t.float() < float(cfg.spectral_t_gate)
    if not bool(active.any()):
        return x0_pred.new_zeros((), dtype=torch.float32)
    return per_sample.index_select(0, active.nonzero(as_tuple=False).flatten()).mean()


def spectral_loss_per_sample(x0_pred: torch.Tensor, x0_target: torch.Tensor,
                             t: torch.Tensor, cfg: AuxLossConfig) -> torch.Tensor:
    """Return per-sample spectral loss, with inactive t-gated samples as zero."""
    out = x0_pred.new_zeros((x0_pred.shape[0],), dtype=torch.float32)
    if not cfg.spectral_enabled:
        return out

    active = t.float() < float(cfg.spectral_t_gate)
    if not bool(active.any()):
        return out

    active_idx = active.nonzero(as_tuple=False).flatten()
    pred_f = x0_pred.index_select(0, active_idx).float()
    target_f = x0_target.index_select(0, active_idx).float()

    # 2D FFT 在最后两个空间维（H, W）；4D/5D 都适用
    fft_pred = torch.fft.fft2(pred_f, dim=(-2, -1), norm="ortho")
    # target 不需要梯度
    with torch.no_grad():
        fft_target = torch.fft.fft2(target_f, dim=(-2, -1), norm="ortho")

    # L1 on amplitude spectrum. A tiny epsilon avoids NaN gradients at zero
    # complex amplitude while leaving nonzero amplitudes effectively unchanged.
    amp_diff = (_complex_abs_stable(fft_pred) - _complex_abs_stable(fft_target)).abs()
    fft_per_sample = amp_diff.view(amp_diff.shape[0], -1).mean(dim=1)

    total = fft_per_sample

    if cfg.spectral_use_wavelet:
        coef_pred = _haar_wavelet_coefs(pred_f)
        with torch.no_grad():
            coef_target = _haar_wavelet_coefs(target_f)
        w_diff = (coef_pred - coef_target).abs()
        w_per_sample = w_diff.view(w_diff.shape[0], -1).mean(dim=1)
        total = total + float(cfg.spectral_wavelet_lambda) * w_per_sample

    out.index_copy_(0, active_idx, total)
    return out


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

        # ★★★ 关键：在加载任何模型之前设置 TORCH_HOME ★★★
        # torch.hub.load (DINOv2 仓库代码 + 权重) 与 torchvision.models.vgg16 (LPIPS 内部)
        # 都从 ${TORCH_HOME}/hub/ 读取。设了这个就能完全离线运行。
        # 必须在 import lpips（构造 LPIPS 实例）和 torch.hub.load 之前。
        if cfg.perceptual_cache_dir:
            cache_dir = Path(cfg.perceptual_cache_dir).resolve()
            if not cache_dir.exists():
                raise FileNotFoundError(
                    f"aux_perceptual_cache_dir 不存在: {cache_dir}\n"
                    f"请先创建 {cache_dir}/hub/checkpoints/ 并把权重文件搬进去（详见 README 或 train_my.yaml 注释）"
                )
            os.environ["TORCH_HOME"] = str(cache_dir)
            logger.info("TORCH_HOME 已设为本地缓存目录: %s", cache_dir)
            # 顺便提示一下用户文件确实在那
            ckpt_dir = cache_dir / "hub" / "checkpoints"
            if ckpt_dir.exists():
                files = sorted(ckpt_dir.glob("*.pth"))
                logger.info("  发现 %d 个本地权重文件: %s",
                            len(files), [f.name for f in files])
            else:
                logger.warning("  ⚠ %s 不存在，加载时仍会尝试联网下载", ckpt_dir)

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
        # ★ 必须在 device 上构造，否则与 GPU 上的 pixels 做减除会因 device 不匹配崩。
        #   register_buffer 不直接接 device 参数，所以在 tensor 构造时就指定。
        self.register_buffer(
            "dino_mean",
            torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "dino_std",
            torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1),
        )

        # ★ 防御性：把整个 wrapper 移到 device，确保任何未来新增的 buffer/parameter
        #   也都在 GPU 上（lpips_fn / dino 已经各自 .to(device) 过，重复调用是 no-op）。
        self.to(device)

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

    def _dino_pixels(self, pixels: torch.Tensor, pixels_lp: torch.Tensor | None) -> torch.Tensor:
        """[-1,1] pixels → ImageNet-normalized, resize 到 224 供 DINOv2。

        当 perceptual_lpips_size >= 224 且 pixels_lp 可用时，从已经下采样的版本
        resize（512→224 vs 1024→224，interpolate 计算量和内存均降到 1/4）。
        """
        src = pixels_lp if (pixels_lp is not None and self.cfg.perceptual_lpips_size >= 224) else pixels
        mean = self.dino_mean.to(dtype=src.dtype, device=src.device)
        std = self.dino_std.to(dtype=src.dtype, device=src.device)
        p = ((src + 1.0) * 0.5).clamp(0.0, 1.0)
        p = (p - mean) / std
        return F.interpolate(p, size=224, mode="bilinear", align_corners=False)

    def _lpips_downsample(self, pixels: torch.Tensor) -> torch.Tensor:
        if self.cfg.perceptual_lpips_size > 0:
            return F.interpolate(pixels, size=int(self.cfg.perceptual_lpips_size),
                                 mode="bilinear", align_corners=False)
        return pixels

    def _compute_per_sample(self, x0_pred: torch.Tensor, x0_target: torch.Tensor) -> torch.Tensor:
        """完整的 per-sample perceptual loss 计算（VAE decode + LPIPS + DINO）。

        用于 eval / no-grad 路径（无 checkpoint），直接 batched 跑。
        """
        pixels_pred = self._decode_to_pixel(x0_pred, with_grad=True)
        pixels_target = self._decode_to_pixel(x0_target, with_grad=False)

        pixels_pred_lp = self._lpips_downsample(pixels_pred)
        pixels_target_lp = self._lpips_downsample(pixels_target)

        with torch.autocast("cuda", dtype=self.compute_dtype):
            l_lpips = self.lpips_fn(pixels_pred_lp, pixels_target_lp)
        l_lpips = l_lpips.view(l_lpips.shape[0]).float()
        per_sample = float(self.cfg.perceptual_lambda_lpips) * l_lpips

        if self.use_dino:
            p_pred = self._dino_pixels(pixels_pred, pixels_pred_lp)
            p_target = self._dino_pixels(pixels_target, pixels_target_lp)
            with torch.autocast("cuda", dtype=self.compute_dtype):
                feat_pred = self.dino.forward_features(p_pred)["x_norm_patchtokens"]
                with torch.no_grad():
                    feat_target = self.dino.forward_features(p_target)["x_norm_patchtokens"]
            cos = F.cosine_similarity(feat_pred.float(), feat_target.float(), dim=-1)
            l_dino = (1.0 - cos).mean(dim=-1)
            per_sample = per_sample + float(self.cfg.perceptual_lambda_dino) * l_dino

        return per_sample

    def _compute_pred_against_target(self, x0_pred: torch.Tensor,
                                     pixels_target_lp: torch.Tensor,
                                     dino_feat_target: torch.Tensor | None) -> torch.Tensor:
        """Checkpoint 优化路径：仅对 pred 做 VAE decode + VGG + DINO。

        Target 的 pixels 和 DINO features 在 checkpoint 边界外预计算完毕，
        通过闭包传入。backward replay 时这些 no_grad 张量直接复用，不再重跑
        VAE decode + DINO forward on target（每 step 省 B 次 VAE + B 次 DINO）。
        """
        pixels_pred = self._decode_to_pixel(x0_pred, with_grad=True)
        pixels_pred_lp = self._lpips_downsample(pixels_pred)

        with torch.autocast("cuda", dtype=self.compute_dtype):
            l_lpips = self.lpips_fn(pixels_pred_lp, pixels_target_lp)
        l_lpips = l_lpips.view(l_lpips.shape[0]).float()
        per_sample = float(self.cfg.perceptual_lambda_lpips) * l_lpips

        if self.use_dino and dino_feat_target is not None:
            p_pred = self._dino_pixels(pixels_pred, pixels_pred_lp)
            with torch.autocast("cuda", dtype=self.compute_dtype):
                feat_pred = self.dino.forward_features(p_pred)["x_norm_patchtokens"]
            cos = F.cosine_similarity(feat_pred.float(), dino_feat_target.float(), dim=-1)
            l_dino = (1.0 - cos).mean(dim=-1)
            per_sample = per_sample + float(self.cfg.perceptual_lambda_dino) * l_dino

        return per_sample

    def forward(self, x0_pred: torch.Tensor, x0_target: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        """计算 λ_lpips * LPIPS + λ_dino * (1 - cos_sim_DINO)，对 t < t_gate 样本做平均。

        ★ 显存控制：若 cfg.perceptual_use_checkpoint=True（默认），整个 VAE+VGG+DINO
          forward 会被 torch.utils.checkpoint 包起来 —— backward 时重放一次以释放
          forward 激活。这与主模型 per-block checkpoint 是对称的设计，否则 1024 训练
          时 perceptual 路径的激活会撑爆 96GB 显存。

        ★ 优化：target 的 VAE decode + DINO features 在 checkpoint 边界外一次性预计算。
          checkpoint 内仅对 pred 做 forward（backward replay 不再冗余重跑 target）。
          batch=4 时每 step 省 4 次 VAE decode + 4 次 DINO forward。
        """
        per_sample = self.forward_per_sample(x0_pred, x0_target, t)
        active = t.float() < float(self.cfg.perceptual_t_gate)
        if not bool(active.any()):
            return x0_pred.new_zeros((), dtype=torch.float32)
        return per_sample.index_select(0, active.nonzero(as_tuple=False).flatten()).mean()

    def forward_per_sample(self, x0_pred: torch.Tensor, x0_target: torch.Tensor,
                           t: torch.Tensor) -> torch.Tensor:
        """Return per-sample perceptual loss, with inactive t-gated samples as zero."""
        out = x0_pred.new_zeros((x0_pred.shape[0],), dtype=torch.float32)
        if not self.cfg.perceptual_enabled:
            return out

        active = t.float() < float(self.cfg.perceptual_t_gate)
        if not bool(active.any()):
            return out

        # PixelGen-style noise gating should save compute, not only zero out loss.
        # Slice to active samples before VAE decode / LPIPS / DINO so high-noise
        # samples do not pay the perceptual path cost.
        active_idx = active.nonzero(as_tuple=False).flatten()
        x0_pred_active = x0_pred.index_select(0, active_idx)
        x0_target_active = x0_target.index_select(0, active_idx)

        if self.cfg.perceptual_use_checkpoint and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint

            B = x0_pred_active.shape[0]

            # ★ 预计算 target：VAE decode + LPIPS downsample + DINO features。
            # 全部在 no_grad 下逐样本跑（避免 batched VAE decode 峰值显存），
            # 结果通过闭包传入 checkpoint 内部，backward replay 直接复用。
            pre_target = []
            with torch.no_grad():
                for i in range(B):
                    pt_i = self._decode_to_pixel(x0_target_active[i:i + 1], with_grad=False)
                    pt_lp_i = self._lpips_downsample(pt_i)
                    dino_feat_i = None
                    if self.use_dino:
                        p_t = self._dino_pixels(pt_i, pt_lp_i)
                        with torch.autocast("cuda", dtype=self.compute_dtype):
                            dino_feat_i = self.dino.forward_features(p_t)["x_norm_patchtokens"]
                    pre_target.append((pt_lp_i, dino_feat_i))
                    del pt_i  # 释放原始分辨率 target pixels

            per_sample_list = []
            for i in range(B):
                x0p_i = x0_pred_active[i:i + 1]
                pt_lp_i, dino_feat_i = pre_target[i]

                def _heavy_i(x, _pt_lp=pt_lp_i, _dino_feat=dino_feat_i):
                    return self._compute_pred_against_target(x, _pt_lp, _dino_feat)

                l_i = checkpoint(_heavy_i, x0p_i, use_reentrant=False)
                per_sample_list.append(l_i)

            per_sample = torch.cat(per_sample_list, dim=0)
        else:
            per_sample = self._compute_per_sample(x0_pred_active, x0_target_active)

        out.index_copy_(0, active_idx, per_sample.float())
        return out


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
    if cfg.self_perceptual_enabled:
        parts.append(
            f"self_perceptual(λ={cfg.self_perceptual_lambda:.3f},"
            f"gate={cfg.self_perceptual_t_gate:.2f},"
            f"tap={cfg.self_perceptual_tap_block},"
            f"enc_t={cfg.self_perceptual_encode_t:.3f})"
        )
    return " + ".join(parts) if parts else "disabled"
