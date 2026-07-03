"""训练时推理 / 采样工具。

包含：
- `_time_snr_shift` / `_flow_sigmas_simple` —— ComfyUI ModelSamplingDiscreteFlow + simple_scheduler 的等价实现
- `_default_noise_sampler` —— 与 ComfyUI k_diffusion_sampling.default_noise_sampler 一致
- `_sample_er_sde_const_x0` —— ER-SDE-Solver-3 在 CONST flow schedule 下的去 ModelPatcher 化实现
- `_encode_text` —— 单条 prompt → cross_cond 编码（cond/uncond/DPO 共用）
- `sample_latent` —— 跑完采样返回训练 latent（不 decode）；DPO loser 生成 + sample_image 共用
- `sample_image` —— 训练循环中按 step / epoch 出预览图的入口（= sample_latent + VAE decode）

注意：训练时采样的 `shift=3.0` 是固定值（对齐 ComfyUI Anima supported_models 默认），
**与训练 t 分布的 `flow_shift` 不相关**。训练 t 分布默认更接近 uniform；采样时仍按
Anima 推荐的 shift=3 让推理过程在结构步上花更多算力。
"""

from __future__ import annotations

import logging
import math

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


def _beta_ppf(q: torch.Tensor, alpha: float, beta: float, grid: int = 4096) -> torch.Tensor:
    """Beta(α,β) 分位函数的免 scipy 数值实现（云端 venv 无 scipy）。

    经 x = sin²θ 换元后，pdf ∝ sin^{2α-1}θ · cos^{2β-1}θ 在 [0, π/2] 上有界光滑
    （α,β ≥ 0.5 时），梯形积分构造 CDF 再线性插值取逆。对 ComfyUI beta scheduler
    的用途（ppf×999 后取整）精度绰绰有余（vs scipy 误差 ~1e-4 量级）。
    """
    theta = torch.linspace(0.0, math.pi / 2, grid, dtype=torch.float64)
    g = torch.sin(theta).clamp(min=1e-12) ** (2 * alpha - 1) * torch.cos(theta).clamp(min=1e-12) ** (2 * beta - 1)
    cdf = torch.cumulative_trapezoid(g, theta, dim=0)
    cdf = torch.cat([torch.zeros(1, dtype=torch.float64), cdf])
    cdf = cdf / cdf[-1]
    q64 = q.to(torch.float64).clamp(0.0, 1.0)
    idx = torch.searchsorted(cdf, q64, right=True).clamp(1, grid - 1)
    c0, c1 = cdf[idx - 1], cdf[idx]
    th0, th1 = theta[idx - 1], theta[idx]
    frac = ((q64 - c0) / (c1 - c0).clamp(min=1e-18)).clamp(0.0, 1.0)
    th = th0 + frac * (th1 - th0)
    return torch.sin(th).square().to(q.dtype)


def _flow_sigmas_beta(steps: int, *, shift: float = 3.0, alpha: float = 0.6, beta: float = 0.6,
                      timesteps: int = 1000, device: str = "cpu") -> torch.Tensor:
    """复刻 ComfyUI beta_scheduler（默认 α=β=0.6）+ ModelSamplingDiscreteFlow。

    步位按 Beta 分位数集中到两端（首尾步更密、中段更稀），与 ComfyUI 推理工作流
    的 scheduler="beta" 对齐 —— 用于让训练内预览与外部评图的 sigma 调度一致。
    重复 timestep 折叠与 ComfyUI 行为一致（返回长度可能 < steps+1）。
    """
    ts_full = torch.arange(1, timesteps + 1, dtype=torch.float32) / float(timesteps)
    sigmas_full = _time_snr_shift(float(shift), ts_full)  # 升序，索引 t-1 对应时间步 t
    qs = 1.0 - torch.linspace(0.0, 1.0, steps + 1, dtype=torch.float32)[:-1]  # endpoint=False
    t_idx = torch.round(_beta_ppf(qs, float(alpha), float(beta)) * (timesteps - 1)).to(torch.long)
    sigs = []
    last = -1
    for t in t_idx.tolist():
        if t != last:
            sigs.append(float(sigmas_full[int(t)]))
            last = t
    sigs.append(0.0)
    sigmas = torch.tensor(sigs, device=device, dtype=torch.float32)
    if sigmas.numel() > 0 and sigmas[0] >= 1.0:
        sigmas[0] = float(_time_snr_shift(float(shift), torch.tensor(1.0 - 1e-4, dtype=torch.float32)))
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


def _encode_text(
    model, qwen_model, qwen_tokenizer, t5_tokenizer, prompt, device,
    *, use_t5_token_weights: bool = True,
) -> torch.Tensor:
    """编码单条 prompt -> cross_cond。

    - anima：[1, ≥512, D]，已 pad 到 512（Qwen hidden states + T5 token 权重）。
    - krea2（按 model.model_family 识别）：[1, L_valid, 12, D] Qwen3-VL 多层堆叠，
      压缩到有效 token（B=1 → 前向无需 mask）。此时 `qwen_model` 参数承载
      load_krea2_text_encoder 返回的 handles dict，t5/tokenizer 参数忽略。

    抽自 `sample_image` 的编码路径，cond / uncond 共用同一逻辑；DPO loser 生成也复用
    本函数（只编码正向 prompt，no-CFG 时无需 uncond）。
    """
    if getattr(model, "model_family", "anima") == "krea2":
        from trainer.model_family import encode_krea2_text
        cross, _ = encode_krea2_text(qwen_model, [prompt], device)
        return cross
    qwen_text = _build_qwen_text_from_prompt(prompt)
    qwen_embeds, qwen_attn = encode_qwen(qwen_model, qwen_tokenizer, [qwen_text], device)
    t5_ids, t5_attn, t5_w = tokenize_t5_weighted(t5_tokenizer, [prompt], max_length=512)
    t5_ids = t5_ids.to(device)
    t5_attn = t5_attn.to(device)
    t5_w = t5_w.to(device, dtype=torch.float32)
    cross = model.preprocess_text_embeds(qwen_embeds, t5_ids, t5_attn, qwen_attn)
    if (
        use_t5_token_weights
        and getattr(model, "llm_adapter", None) is not None
        and cross.shape[1] == t5_w.shape[1]
    ):
        cross = cross * t5_w.to(cross.dtype).unsqueeze(-1)
    if cross.shape[1] < 512:
        cross = F.pad(cross, (0, 0, 0, 512 - cross.shape[1]))
    return cross


@torch.no_grad()
def sample_latent(
    model,
    cross_cond,
    cross_uncond=None,
    *,
    height: int = 1024,
    width: int = 1024,
    steps: int = 25,
    cfg_scale: float = 4.0,
    sampler_name: str = "er_sde",
    scheduler: str = "simple",
    device="cuda",
    dtype=torch.bfloat16,
    injector=None,
    seed: int | None = None,
    shift: float | None = None,
) -> torch.Tensor:
    """运行 ER-SDE CONST 采样并返回训练 latent（`[1,16,1,h//8,w//8]`，float32），**不做 VAE decode**。

    `shift`：sigma 调度的 time-snr shift。None = 按 family 取默认——anima 固定 3.0
    （ComfyUI supported_models 口径），krea2 用官方分辨率感知 exp(mu(H,W))
    （1024²≈2.48；与 Anima 的 `_time_snr_shift` 代数同形，直接复用同一 scheduler）。

    `sample_image` = 本函数 + VAE decode。DPO loser 生成直接调用本函数，全程留在 latent
    空间（loser pool 存的就是返回值，与训练加噪管线直接兼容）。

    - **no-CFG 分支**：`cfg_scale == 1.0` 或 `cross_uncond is None` 时每步只做一次条件前向
      （CFG 公式在 cfg=1 时恒等于 v_cond），用于 `dpo_loser_cfg=1.0`，每步前向数减半、loser
      自然更弱。
    - **RNG 隔离**：传入 `seed` 时初始噪声与 ER-SDE 随机项都用独立 `Generator`，不污染训练
      噪声流（DPO in-loop 采样要求）。`seed=None`（预览路径）时维持旧行为（全局 RNG）。
    - `model.train()/eval()` 由本函数 try/finally 守护，中途抛错也能恢复。
    """
    _orig_training = bool(model.training)
    model.eval()
    try:
        lat_h, lat_w = height // 8, width // 8
        if shift is None:
            if getattr(model, "model_family", "anima") == "krea2":
                from trainer.model_family import krea2_sample_shift
                shift = krea2_sample_shift(height, width)
            else:
                shift = 3.0
        _sched = str(scheduler).lower()
        if _sched == "beta":
            sigmas = _flow_sigmas_beta(steps, shift=float(shift), device=device)
        else:
            if _sched != "simple":
                logger.warning(f"采样 scheduler={scheduler} 未实现，回退 simple")
            sigmas = _flow_sigmas_simple(steps, shift=float(shift), device=device)

        # 初始化噪声（ComfyUI CONST.noise_scaling: x = sigma*noise + (1-sigma)*latent_image；txt2img latent_image=0）
        if seed is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(int(seed))
            x = torch.randn(1, 16, 1, lat_h, lat_w, device=device, dtype=torch.float32, generator=gen) * float(sigmas[0])
        else:
            x = torch.randn(1, 16, 1, lat_h, lat_w, device=device, dtype=torch.float32) * float(sigmas[0])
        logger.info(f"[Debug] Latents init: {x.shape}, mean={x.mean().item():.4f}, std={x.std().item():.4f}")

        pad_mask = torch.zeros(1, 1, lat_h, lat_w, device=device, dtype=dtype)
        device_type = "cuda" if str(device).startswith("cuda") else "cpu"
        use_cfg = (cross_uncond is not None) and (float(cfg_scale) != 1.0)

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
                    if use_cfg:
                        v_uncond = model(x_in.to(device=x_in.device, dtype=dtype), sigma_b, cross_uncond, padding_mask=pad_mask)
                        v = v_uncond + cfg_scale * (v_cond - v_uncond)
                    else:
                        v = v_cond
            finally:
                if injector is not None:
                    injector.set_current_t(None)

            if torch.isnan(v).any():
                raise RuntimeError("v contains NaN during sampling")

            # CONST(flow): denoised x0 = x - sigma * v
            return x_in - sigma_5d * v.float()

        sampler_name_l = str(sampler_name).lower().strip()
        logger.info(f"[Debug] Sampler={sampler_name_l}, Scheduler={_sched}, steps={steps}, cfg={cfg_scale}, cfg_branch={'on' if use_cfg else 'off'}")

        if sampler_name_l == "er_sde":
            x = _sample_er_sde_const_x0(denoise_fn, x, sigmas, seed=seed, s_noise=1.0, max_stage=3)
        else:
            # fallback: 简化 Euler ODE（deterministic），与 flow 兼容
            for i in range(len(sigmas) - 1):
                sigma = float(sigmas[i])
                sigma_next = float(sigmas[i + 1])
                denoised = denoise_fn(x, sigmas[i])
                d = (x - denoised) / max(sigma, 1e-6)
                x = x + d * (sigma_next - sigma)

        return x
    finally:
        # 不论是否抛错都恢复 model train/eval 模式。
        if _orig_training:
            model.train()
        else:
            model.eval()


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

    ★ model.train()/eval() 状态用 try/finally 守护：
       旧实现是函数体内 model.eval() 进入、return 前 model.train() 退出。但若中途抛错
       （NaN/Inf、VAE decode 失败、模型分支跳错），return 永不到达，模型留在 eval 模式 →
       继续训练时 dropout 等被禁用，训练曲线静默偏移。
    """
    import numpy as np
    from PIL import Image
    # 记录原 mode（model.training 为 True/False），用于 finally 恢复
    _orig_training = bool(model.training)
    model.eval()

    logger.info(f"[Debug] Sampling start. Prompt: {prompt[:50]}...")

    try:
        # Check VAE scale
        if isinstance(vae.scale, list) and len(vae.scale) == 2:
            m, s = vae.scale
            logger.info(f"[Debug] VAE scale: mean_shape={m.shape}, std_inv_shape={s.shape}")
            logger.info(f"[Debug] VAE scale values: mean={m.mean().item():.4f}, std_inv={s.mean().item():.4f}")

        # 默认负面提示词：anima 参考 Anima Prompt Guide；krea2 对齐官方（空负面，
        # danbooru 风格质量 tag 对 krea2 语义未知，不做默认注入）。
        if negative_prompt is None:
            if getattr(model, "model_family", "anima") == "krea2":
                negative_prompt = ""
            else:
                negative_prompt = (
                    "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, "
                    "bad anatomy, bad hands, bad feet, missing fingers, extra fingers, text, watermark, "
                    "logo, signature, username, artist name, copyright name"
                )

        # 文本编码（cond + uncond，复用 _encode_text）
        try:
            cross_cond = _encode_text(
                model, qwen_model, qwen_tokenizer, t5_tokenizer, prompt, device,
                use_t5_token_weights=use_t5_token_weights,
            )
            cross_uncond = _encode_text(
                model, qwen_model, qwen_tokenizer, t5_tokenizer, negative_prompt, device,
                use_t5_token_weights=use_t5_token_weights,
            )
        except Exception as e:
            logger.error(f"[Debug] Encoding failed: {e}")
            raise

        # 采样（sigmas / 噪声初始化 / ER-SDE 求解，全部下放到 sample_latent；预览路径 seed=None
        # 维持旧的全局 RNG 行为，cfg!=1 走原 CFG 双前向分支，结果与重构前一致）
        x = sample_latent(
            model, cross_cond, cross_uncond,
            height=height, width=width, steps=steps, cfg_scale=cfg_scale,
            sampler_name=sampler_name, scheduler=scheduler,
            device=device, dtype=dtype, injector=injector, seed=None,
        )

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
            return Image.fromarray(img)
        except Exception as e:
            logger.error(f"[Debug] VAE decode failed: {e}")
            raise
    finally:
        # 不论是否抛错都恢复 model train/eval 模式（修了旧实现"中途抛错 → 模型留在 eval"的 bug）。
        if _orig_training:
            model.train()
        else:
            model.eval()
