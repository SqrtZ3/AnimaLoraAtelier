"""训练时推理 / 采样工具。

包含：
- `_time_snr_shift` / `_flow_sigmas_simple` —— ComfyUI ModelSamplingDiscreteFlow + simple_scheduler 的等价实现
- `_default_noise_sampler` —— 与 ComfyUI k_diffusion_sampling.default_noise_sampler 一致
- `_sample_er_sde_const_x0` —— ER-SDE-Solver-3 在 CONST flow schedule 下的去 ModelPatcher 化实现
- `sample_image` —— 训练循环中按 step / epoch 出预览图的入口

注意：训练时采样的 `shift=3.0` 是固定值（对齐 ComfyUI Anima supported_models 默认），
**与训练 t 分布的 `flow_shift` 不相关**。训练 t 分布默认更接近 uniform；采样时仍按
Anima 推荐的 shift=3 让推理过程在结构步上花更多算力。
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from trainer.text_encode import (
    _build_qwen_text_from_prompt,
    encode_qwen,
    tokenize_t5_weighted,
)

logger = logging.getLogger(__name__)


def _time_snr_shift(alpha: float, t: torch.Tensor) -> torch.Tensor:
    """ComfyUI ModelSamplingDiscreteFlow.time_snr_shift"""
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)


def _flow_sigmas_simple(steps: int, *, shift: float = 3.0, timesteps: int = 1000, device: str = "cpu") -> torch.Tensor:
    """
    复刻 ComfyUI:
    - supported_models.Anima 的 sampling_settings: shift=3.0, multiplier=1.0
    - ModelSamplingDiscreteFlow + simple_scheduler(model_sampling, steps)

    返回：sigmas (steps+1,) float32，从高到低，末尾带 0.0
    """
    ts = torch.arange(1, timesteps + 1, device=device, dtype=torch.float32) / float(timesteps)  # (0, 1]
    sigmas_full = _time_snr_shift(float(shift), ts)  # (0, 1]

    ss = len(sigmas_full) / float(steps)
    sigmas = [float(sigmas_full[-(1 + int(i * ss))]) for i in range(steps)]
    sigmas.append(0.0)
    sigmas = torch.tensor(sigmas, device=device, dtype=torch.float32)

    # ComfyUI offset_first_sigma_for_snr: CONST 下避免 sigma=1 导致 logit inf
    if sigmas.numel() > 0 and sigmas[0] >= 1.0:
        sigmas[0] = float(_time_snr_shift(float(shift), torch.tensor(1.0 - 1e-4, device=device, dtype=torch.float32)))
    return sigmas


def _default_noise_sampler(x: torch.Tensor, seed: int | None):
    """参考 ComfyUI k_diffusion_sampling.default_noise_sampler"""
    if seed is not None:
        if x.device.type == "cpu":
            seed = int(seed) + 1
        g = torch.Generator(device=x.device)
        g.manual_seed(int(seed))
    else:
        g = None

    def _sample(_sigma, _sigma_next):
        return torch.randn(x.size(), dtype=x.dtype, layout=x.layout, device=x.device, generator=g)

    return _sample


@torch.no_grad()
def _sample_er_sde_const_x0(
    denoise_fn,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    seed: int | None = None,
    s_noise: float = 1.0,
    max_stage: int = 3,
):
    """
    Extended Reverse-Time SDE solver（ER-SDE-Solver-3）在 CONST(flow) 噪声日程下的实现。
    参考 ComfyUI 的 k_diffusion_sampling.sample_er_sde（删去 model_patcher 依赖）。
    """
    sigmas = sigmas.to(device=x.device, dtype=torch.float32)
    if sigmas.numel() <= 1:
        return x

    noise_sampler = _default_noise_sampler(x, seed=seed)

    # CONST: half_log_snr = log((1 - t) / t) = -logit(t)
    eps = 1e-12
    t = sigmas.clamp(min=eps, max=1.0 - eps)
    half_log_snrs = torch.log((1 - t) / t)
    er_lambdas = half_log_snrs.neg().exp()  # er_lambda = t / (1 - t)

    old_denoised = None
    old_denoised_d = None

    def noise_scaler(lam: torch.Tensor) -> torch.Tensor:
        # default_er_sde_noise_scaler
        lam = lam.to(x.device, dtype=torch.float32)
        return lam * ((lam ** 0.3).exp() + 10.0)

    num_integration_points = 200.0
    point_indice = torch.arange(0, num_integration_points, dtype=torch.float32, device=x.device)

    for i in range(len(sigmas) - 1):
        sigma = sigmas[i]
        denoised = denoise_fn(x, sigma)

        stage_used = min(int(max_stage), i + 1)
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            er_lambda_s, er_lambda_t = er_lambdas[i], er_lambdas[i + 1]
            alpha_s = 1.0 - sigmas[i]
            alpha_t = 1.0 - sigmas[i + 1]
            r_alpha = alpha_t / alpha_s
            r = noise_scaler(er_lambda_t) / noise_scaler(er_lambda_s)

            # Stage 1 (Euler)
            x = r_alpha * r * x + alpha_t * (1 - r) * denoised

            if stage_used >= 2 and old_denoised is not None:
                dt = er_lambda_t - er_lambda_s
                lambda_step_size = -dt / num_integration_points
                lambda_pos = er_lambda_t + point_indice * lambda_step_size
                scaled_pos = noise_scaler(lambda_pos)

                # Stage 2
                s = torch.sum(1 / scaled_pos) * lambda_step_size
                denoised_d = (denoised - old_denoised) / (er_lambda_s - er_lambdas[i - 1])
                x = x + alpha_t * (dt + s * noise_scaler(er_lambda_t)) * denoised_d

                if stage_used >= 3 and old_denoised_d is not None:
                    # Stage 3
                    s_u = torch.sum((lambda_pos - er_lambda_s) / scaled_pos) * lambda_step_size
                    denoised_u = (denoised_d - old_denoised_d) / ((er_lambda_s - er_lambdas[i - 2]) / 2)
                    x = x + alpha_t * ((dt ** 2) / 2 + s_u * noise_scaler(er_lambda_t)) * denoised_u

                old_denoised_d = denoised_d

            # Stochastic term
            if s_noise and float(s_noise) > 0:
                noise = noise_sampler(float(sigmas[i]), float(sigmas[i + 1]))
                sde_scale = (er_lambda_t ** 2 - (er_lambda_s ** 2) * (r ** 2)).clamp(min=0).sqrt().nan_to_num(nan=0.0)
                x = x + alpha_t * noise * float(s_noise) * sde_scale

        old_denoised = denoised

    return x


@torch.no_grad()
def sample_image(
    model, vae, qwen_model, qwen_tokenizer, t5_tokenizer,
    prompt, height=1024, width=1024, steps=25, cfg_scale=4.0,
    negative_prompt=None,
    sampler_name: str = "er_sde",
    scheduler: str = "simple",
    device="cuda",
    dtype=torch.bfloat16,
    use_t5_token_weights: bool = True,
    injector=None,
):
    """训练时采样预览（尽量对齐 ComfyUI KSampler）。

    Args:
        negative_prompt: 负面提示词，默认使用标准负面提示词
        sampler_name: 采样器（推荐：er_sde）
        scheduler: 调度器（推荐：simple）
    """
    import numpy as np
    from PIL import Image
    model.eval()

    logger.info(f"[Debug] Sampling start. Prompt: {prompt[:50]}...")

    # Check VAE scale
    if isinstance(vae.scale, list) and len(vae.scale) == 2:
        m, s = vae.scale
        logger.info(f"[Debug] VAE scale: mean_shape={m.shape}, std_inv_shape={s.shape}")
        logger.info(f"[Debug] VAE scale values: mean={m.mean().item():.4f}, std_inv={s.mean().item():.4f}")

    # 默认负面提示词 (参考 Anima Prompt Guide)
    if negative_prompt is None:
        negative_prompt = "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, bad anatomy, bad hands, bad feet, missing fingers, extra fingers, text, watermark, logo, signature, username, artist name, copyright name"

    # 文本编码
    try:
        # 有条件 (positive prompt)
        qwen_text = _build_qwen_text_from_prompt(prompt)
        qwen_embeds, qwen_attn = encode_qwen(qwen_model, qwen_tokenizer, [qwen_text], device)
        logger.info(f"[Debug] Qwen embeds: {qwen_embeds.shape}, mean={qwen_embeds.mean().item():.4f}")

        t5_ids, t5_attn, t5_w = tokenize_t5_weighted(t5_tokenizer, [prompt], max_length=512)
        t5_ids = t5_ids.to(device)
        t5_attn = t5_attn.to(device)
        t5_w = t5_w.to(device, dtype=torch.float32)
        cross_cond = model.preprocess_text_embeds(qwen_embeds, t5_ids, t5_attn, qwen_attn)
        if (
            use_t5_token_weights
            and getattr(model, "llm_adapter", None) is not None
            and cross_cond.shape[1] == t5_w.shape[1]
        ):
            cross_cond = cross_cond * t5_w.to(cross_cond.dtype).unsqueeze(-1)
        if cross_cond.shape[1] < 512:
            cross_cond = F.pad(cross_cond, (0, 0, 0, 512 - cross_cond.shape[1]))

        # 无条件/负面提示词 (negative prompt)
        qwen_text_uncond = _build_qwen_text_from_prompt(negative_prompt)
        qwen_embeds_uncond, qwen_attn_uncond = encode_qwen(qwen_model, qwen_tokenizer, [qwen_text_uncond], device)
        t5_ids_uncond, t5_attn_uncond, t5_w_uncond = tokenize_t5_weighted(t5_tokenizer, [negative_prompt], max_length=512)
        t5_ids_uncond = t5_ids_uncond.to(device)
        t5_attn_uncond = t5_attn_uncond.to(device)
        t5_w_uncond = t5_w_uncond.to(device, dtype=torch.float32)
        cross_uncond = model.preprocess_text_embeds(qwen_embeds_uncond, t5_ids_uncond, t5_attn_uncond, qwen_attn_uncond)
        if (
            use_t5_token_weights
            and getattr(model, "llm_adapter", None) is not None
            and cross_uncond.shape[1] == t5_w_uncond.shape[1]
        ):
            cross_uncond = cross_uncond * t5_w_uncond.to(cross_uncond.dtype).unsqueeze(-1)
        if cross_uncond.shape[1] < 512:
            cross_uncond = F.pad(cross_uncond, (0, 0, 0, 512 - cross_uncond.shape[1]))

    except Exception as e:
        logger.error(f"[Debug] Encoding failed: {e}")
        raise e

    # sigmas（对齐 ComfyUI supported_models.Anima: shift=3.0, multiplier=1.0）
    lat_h, lat_w = height // 8, width // 8
    if str(scheduler).lower() != "simple":
        logger.warning(f"采样 scheduler={scheduler} 未实现，回退 simple")
    sigmas = _flow_sigmas_simple(steps, shift=3.0, device=device)

    # 初始化噪声（ComfyUI CONST.noise_scaling: x = sigma*noise + (1-sigma)*latent_image；txt2img latent_image=0）
    x = torch.randn(1, 16, 1, lat_h, lat_w, device=device, dtype=torch.float32) * float(sigmas[0])
    logger.info(f"[Debug] Latents init: {x.shape}, mean={x.mean().item():.4f}, std={x.std().item():.4f}")

    pad_mask = torch.zeros(1, 1, lat_h, lat_w, device=device, dtype=dtype)
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"

    def denoise_fn(x_in: torch.Tensor, sigma_in: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(sigma_in):
            sigma_in = torch.tensor(float(sigma_in), device=x_in.device, dtype=torch.float32)
        sigma_b = sigma_in.view(1, 1).to(device=x_in.device, dtype=dtype)
        sigma_5d = sigma_in.view(1, 1, 1, 1, 1).to(device=x_in.device, dtype=torch.float32)

        # T-LoRA：把当前 σ 写到 LoRA adapter（B=1 here）。其它 variant 自动跳过。
        # 在 try/finally 内确保即便 forward 抛错也能 reset，避免污染后续 step。
        if injector is not None:
            injector.set_current_t(sigma_in.view(-1).to(device=x_in.device, dtype=torch.float32))
        try:
            with torch.autocast(device_type=device_type, dtype=dtype):
                v_cond = model(x_in.to(device=x_in.device, dtype=dtype), sigma_b, cross_cond, padding_mask=pad_mask)
                v_uncond = model(x_in.to(device=x_in.device, dtype=dtype), sigma_b, cross_uncond, padding_mask=pad_mask)
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
        finally:
            if injector is not None:
                injector.set_current_t(None)

        if torch.isnan(v).any():
            raise RuntimeError("v contains NaN during sampling")

        # CONST(flow): denoised x0 = x - sigma * v
        return x_in - sigma_5d * v.float()

    sampler_name_l = str(sampler_name).lower().strip()
    logger.info(f"[Debug] Sampler={sampler_name_l}, Scheduler=simple, steps={steps}, cfg={cfg_scale}")

    if sampler_name_l == "er_sde":
        x = _sample_er_sde_const_x0(denoise_fn, x, sigmas, seed=None, s_noise=1.0, max_stage=3)
    else:
        # fallback: 简化 Euler ODE（deterministic），与 flow 兼容
        for i in range(len(sigmas) - 1):
            sigma = float(sigmas[i])
            sigma_next = float(sigmas[i + 1])
            denoised = denoise_fn(x, sigmas[i])
            d = (x - denoised) / max(sigma, 1e-6)
            x = x + d * (sigma_next - sigma)

    # VAE 解码
    latents = x.to(device=device, dtype=dtype)
    logger.info(f"[Debug] Final latents: mean={latents.mean().item():.4f}, std={latents.std().item():.4f}")
    try:
        images = vae.model.decode(latents, vae.scale)
        images = images.squeeze(2)  # [B,C,H,W]
        images = (images.clamp(-1, 1) + 1) / 2

        # 转 PIL
        img = images[0].permute(1, 2, 0).cpu().float().numpy()
        img = (img * 255).clip(0, 255).astype(np.uint8)

        model.train()
        return Image.fromarray(img)
    except Exception as e:
        logger.error(f"[Debug] VAE decode failed: {e}")
        raise e
