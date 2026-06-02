"""YAML 配置加载、args 映射、weight_decay 解析。

主入口：
- `load_yaml_config(path)` —— 读 YAML 文件成 dict
- `apply_yaml_config(args, config)` —— 把 YAML 字段映射到 argparse args，处理过期键
  warning，最后调用 `_resolve_weight_decay` 统一 wd 来源
- `_resolve_weight_decay(args, config)` —— 处理顶层 `weight_decay` 与
  `optimizer_args.weight_decay` 同时出现时的冲突，确保 ProdigyPlus 看到的 per-group wd
  与用户显式设的值一致（详见 weight_decay 修复 commit 的 message）

依赖：仅 logging + pyyaml（懒加载）。本模块不导入任何其它 trainer 子模块。
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_yaml_config(config_path):
    """加载 YAML 配置文件"""
    try:
        import yaml
    except ImportError:
        print("PyYAML not installed. Install with: pip install pyyaml")
        raise SystemExit(1)

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    return config


# YAML key → argparse attribute 映射（保持稳定的对外接口）
YAML_TO_ARGS = {
    # 模型路径
    "transformer_path": "transformer",
    "vae_path": "vae",
    "text_encoder_path": "qwen",
    "t5_tokenizer_path": "t5_tokenizer",
    # 数据集
    "data_dir": "data_dir",
    "reg_data_dir": "reg_data_dir",
    "reg_repeats": "reg_repeats",
    "reg_caption": "reg_caption",
    "resolution": "resolution",
    # ARB 分桶细粒度参数（kohya 风格命名）
    "bucket_base_resos": "bucket_base_resos",
    "bucket_min_base_reso": "bucket_min_base_reso",
    "bucket_max_base_reso": "bucket_max_base_reso",
    "bucket_base_reso_steps": "bucket_base_reso_steps",
    "bucket_no_upscale": "bucket_no_upscale",
    "bucket_max_upscale": "bucket_max_upscale",
    "bucket_report": "bucket_report",
    "min_bucket_reso": "min_bucket_reso",
    "max_bucket_reso": "max_bucket_reso",
    "bucket_reso_steps": "bucket_reso_steps",
    "bucket_max_aspect_ratio": "bucket_max_aspect_ratio",
    "bucket_drop_last": "bucket_drop_last",
    "fit_packed_training": "fit_packed_training",
    "fit_max_tokens": "fit_max_tokens",
    "fit_warn_tokens": "fit_warn_tokens",
    "fit_min_tokens": "fit_min_tokens",
    "fit_patch_size": "fit_patch_size",
    "fit_vae_downsample": "fit_vae_downsample",
    "fit_over_budget_strategy": "fit_over_budget_strategy",
    "fit_align_mode": "fit_align_mode",
    "fit_pack_multiple_images": "fit_pack_multiple_images",
    "fit_max_tokens_per_batch": "fit_max_tokens_per_batch",
    "alpha_handling": "alpha_handling",
    "alpha_background": "alpha_background",
    "alpha_threshold": "alpha_threshold",
    # RoPE 位置嵌入上限（latent 单位）；0 = 自动从 max_bucket_reso 算
    "max_img_h": "max_img_h",
    "max_img_w": "max_img_w",
    "repeats": "repeats",
    "shuffle_caption": "shuffle_caption",
    "keep_tokens": "keep_tokens",
    "flip_augment": "flip_augment",
    "tag_dropout": "tag_dropout",
    "prefer_json": "prefer_json",
    "cache_latents": "cache_latents",
    "cache_encode_batch_size": "cache_encode_batch_size",
    # LoRA 配置
    "lora_type": "lora_type",
    "lora_rank": "lora_rank",
    "lora_alpha": "lora_alpha",
    "lora_dropout": "lora_dropout",
    "lokr_factor": "lokr_factor",
    "lora_variant": "lora_variant",
    "dora_export_mode": "dora_export_mode",
    "lora_targets": "lora_targets",
    "lora_exclude_prefixes": "lora_exclude_prefixes",
    "resume_lora": "resume_lora",
    # 训练参数
    "epochs": "epochs",
    "max_steps": "max_steps",
    "batch_size": "batch_size",
    "effective_batch_size": "effective_batch_size",
    "grad_accum": "grad_accum",
    "learning_rate": "lr",
    "lr_scheduler": "lr_scheduler",
    "lr_scheduler_t0": "lr_scheduler_t0",
    "lr_scheduler_t_mult": "lr_scheduler_t_mult",
    "lr_scheduler_eta_min": "lr_scheduler_eta_min",
    "weight_decay": "weight_decay",
    "grad_clip_max_norm": "grad_clip_max_norm",
    "mixed_precision": "mixed_precision",
    "grad_checkpoint": "grad_checkpoint",
    "num_workers": "num_workers",
    # 输出与保存
    "output_dir": "output_dir",
    "output_name": "output_name",
    "save_every": "save_every",
    "save_every_steps": "save_every_steps",
    "save_every_reference_steps": "save_every_reference_steps",
    "save_state_every": "save_state_every",
    "resume_state": "resume_state",
    "seed": "seed",
    # 采样
    "sample_every": "sample_every",
    "sample_steps": "sample_steps",
    "sample_prompt": "sample_prompt",
    "sample_prompts": "sample_prompts",
    "sample_cfg_scale": "sample_cfg_scale",
    "sample_negative_prompt": "sample_negative_prompt",
    "sample_width": "sample_width",
    "sample_height": "sample_height",
    "sample_seed": "sample_seed",
    "sample_infer_steps": "sample_infer_steps",
    "sample_sampler_name": "sample_sampler_name",
    "sample_scheduler": "sample_scheduler",
    "sample_reference_steps": "sample_reference_steps",
    "reference_batch_size": "reference_batch_size",
    "reference_grad_accum": "reference_grad_accum",
    "keep_vae_on_gpu": "keep_vae_on_gpu",
    "empty_cache_after_sample": "empty_cache_after_sample",
    # 进度显示与监控
    "loss_curve_steps": "loss_curve_steps",
    "no_progress": "no_progress",
    "log_every": "log_every",
    "no_monitor": "no_monitor",
    "monitor_host": "monitor_host",
    "monitor_port": "monitor_port",
    "no_browser": "no_browser",
    "debug_first_batches": "debug_first_batches",
    # 优化器配置映射
    "optimizer_type": "optimizer_type",
    "prodigyplus_d0": "prodigyplus_d0",
    "prodigyplus_use_stableadamw": "prodigyplus_use_stableadamw",
    # 优化器透明路由（任意 key 直接以 **kwargs 传入优化器，未识别的会被自动过滤并 warning）
    "optimizer_args": "optimizer_args",
    "use_t5_token_weights": "use_t5_token_weights",
    # Flow Matching / 损失权重 / 噪声增强
    "flow_shift": "flow_shift",
    "schedule_shift": "schedule_shift",
    "timestep_sampling": "timestep_sampling",
    "timestep_mix_low_prob": "timestep_mix_low_prob",
    "adaptive_timestep": "adaptive_timestep",
    "adaptive_timestep_metric": "adaptive_timestep_metric",
    "adaptive_timestep_highfreq_weight": "adaptive_timestep_highfreq_weight",
    "adaptive_timestep_bins": "adaptive_timestep_bins",
    "adaptive_timestep_ema_decay": "adaptive_timestep_ema_decay",
    "adaptive_timestep_burn_in": "adaptive_timestep_burn_in",
    "adaptive_timestep_min_factor": "adaptive_timestep_min_factor",
    "adaptive_timestep_max_factor": "adaptive_timestep_max_factor",
    "adaptive_timestep_base_mix": "adaptive_timestep_base_mix",
    "adaptive_timestep_candidate_mult": "adaptive_timestep_candidate_mult",
    # InfoNoise (arxiv:2602.18647) — entropy_rate 模式专用闸门 / 配置
    "adaptive_timestep_low_noise_gate": "adaptive_timestep_low_noise_gate",
    "adaptive_timestep_gate_n": "adaptive_timestep_gate_n",
    "adaptive_timestep_gate_c": "adaptive_timestep_gate_c",
    "min_snr_gamma": "min_snr_gamma",
    "loss_weighting_scheme": "loss_weighting_scheme",
    "weight_cap_ratio": "weight_cap_ratio",
    "loss_type": "loss_type",
    "huber_c": "huber_c",
    "huber_schedule": "huber_schedule",
    "noise_offset": "noise_offset",
    "noise_offset_min": "noise_offset_min",
    "noise_offset_random_strength": "noise_offset_random_strength",
    "pyramid_noise_iterations": "pyramid_noise_iterations",
    "pyramid_noise_discount": "pyramid_noise_discount",
    "caption_dropout_rate": "caption_dropout_rate",
    "grad_norm_log_every": "grad_norm_log_every",
    # Regex 模块选择（kohya 风格）
    "lora_exclude_patterns": "lora_exclude_patterns",
    "lora_include_patterns": "lora_include_patterns",
    # 模块级 rank/lr 控制（kohya 风格）
    "lora_reg_dims": "lora_reg_dims",
    "lora_reg_alphas": "lora_reg_alphas",
    "lora_reg_lrs": "lora_reg_lrs",
    # 精细正则化
    "rank_dropout": "rank_dropout",
    "module_dropout": "module_dropout",
    # T-LoRA (arxiv:2507.05964) — 仅 lora_variant=tlora 时生效
    "tlora_rmin_ratio": "tlora_rmin_ratio",
    "tlora_alpha": "tlora_alpha",
    "tlora_init": "tlora_init",
    "tlora_lokr_experimental": "tlora_lokr_experimental",
    "tlora_lokr_ortho_init": "tlora_lokr_ortho_init",
    # LoRA+ for LoKr
    "loraplus_lr_ratio": "loraplus_lr_ratio",
    # 频率均衡 tag dropout
    "freq_balanced_dropout_strength": "freq_balanced_dropout_strength",
    # 画风预设：sharp / hazy / balanced，详见 STYLE_PROFILES
    "style_profile": "style_profile",
    # detail_inv_t 权重的可调上下限（默认 [1, 5]）
    "detail_inv_t_min": "detail_inv_t_min",
    "detail_inv_t_max": "detail_inv_t_max",
    # ── 辅助 loss（Spectral + Perceptual），详见 trainer/aux_losses.py ──
    "aux_spectral_enabled": "aux_spectral_enabled",
    "aux_spectral_lambda": "aux_spectral_lambda",
    "aux_spectral_use_wavelet": "aux_spectral_use_wavelet",
    "aux_spectral_wavelet_lambda": "aux_spectral_wavelet_lambda",
    "aux_spectral_t_gate": "aux_spectral_t_gate",
    "aux_perceptual_enabled": "aux_perceptual_enabled",
    "aux_perceptual_lambda_lpips": "aux_perceptual_lambda_lpips",
    "aux_perceptual_lambda_dino": "aux_perceptual_lambda_dino",
    "aux_perceptual_t_gate": "aux_perceptual_t_gate",
    "aux_perceptual_lpips_net": "aux_perceptual_lpips_net",
    "aux_perceptual_dino_local_path": "aux_perceptual_dino_local_path",
    "aux_perceptual_cache_dir": "aux_perceptual_cache_dir",
    "aux_perceptual_use_checkpoint": "aux_perceptual_use_checkpoint",
    "aux_perceptual_lpips_size": "aux_perceptual_lpips_size",
}


# argparse 默认值（供 apply_yaml_config 判断"用户在命令行显式设置 vs 用了默认值"）
DEFAULTS = {
    "transformer": "",
    "vae": "",
    "qwen": "",
    "t5_tokenizer": "",
    "data_dir": "",
    "reg_data_dir": "",
    "reg_repeats": 1,
    "reg_caption": "",
    "resolution": 1024,
    "bucket_base_resos": None,
    "bucket_min_base_reso": 0,
    "bucket_max_base_reso": 0,
    "bucket_base_reso_steps": 256,
    "bucket_no_upscale": False,
    "bucket_max_upscale": 0.0,
    "bucket_report": False,
    "min_bucket_reso": 512,
    "max_bucket_reso": 2048,
    "bucket_reso_steps": 64,
    "bucket_max_aspect_ratio": 2.0,
    "bucket_drop_last": False,
    "fit_packed_training": False,
    "fit_max_tokens": 65536,
    "fit_warn_tokens": 16384,
    "fit_min_tokens": 16,
    "fit_patch_size": 2,
    "fit_vae_downsample": 8,
    "fit_over_budget_strategy": "fail",
    "fit_align_mode": "pad",
    "fit_pack_multiple_images": False,
    "fit_max_tokens_per_batch": 0,
    "alpha_handling": "none",
    "alpha_background": "neutral",
    "alpha_threshold": 0.01,
    "max_img_h": 0,
    "max_img_w": 0,
    "repeats": 1,
    "shuffle_caption": False,
    "keep_tokens": 0,
    "flip_augment": False,
    "tag_dropout": 0.0,
    "prefer_json": True,
    "cache_latents": False,
    "cache_encode_batch_size": 8,
    "lora_type": "lokr",
    "lora_rank": 32,
    "lora_alpha": 32.0,
    "lora_dropout": 0.0,
    "lokr_factor": 8,
    "lora_variant": "base",
    "dora_export_mode": "native",
    "lora_targets": None,
    "lora_exclude_prefixes": None,
    "resume_lora": "",
    "epochs": 10,
    "max_steps": 0,
    "batch_size": 1,
    "effective_batch_size": 0,
    "grad_accum": 1,
    "lr": 1e-4,
    "lr_scheduler": "none",
    "lr_scheduler_t0": 500,
    "lr_scheduler_t_mult": 2.0,
    "lr_scheduler_eta_min": 0.0,
    "weight_decay": 0.01,
    "grad_clip_max_norm": 1.0,
    "mixed_precision": "bf16",
    "grad_checkpoint": False,
    "num_workers": 0,
    "output_dir": "./output",
    "output_name": "anima_lora",
    "save_every": 0,
    "save_every_steps": 0,
    "save_every_reference_steps": 0,
    "save_state_every": 0,
    "resume_state": "",
    "seed": 42,
    "sample_every": 0,
    "sample_steps": 0,
    "sample_prompt": "1girl, masterpiece",
    "sample_prompts": [],
    "sample_cfg_scale": 4.0,
    "sample_negative_prompt": "",
    "sample_width": 0,
    "sample_height": 0,
    "sample_seed": 0,
    "sample_infer_steps": 25,
    "sample_sampler_name": "er_sde",
    "sample_scheduler": "simple",
    "sample_reference_steps": 0,
    "reference_batch_size": 0,
    "reference_grad_accum": 1,
    "keep_vae_on_gpu": False,
    "empty_cache_after_sample": True,
    "loss_curve_steps": 100,
    "no_progress": False,
    "log_every": 10,
    "no_monitor": False,
    "monitor_host": "0.0.0.0",
    "monitor_port": 8765,
    "no_browser": False,
    "debug_first_batches": 0,
    "optimizer_type": "adamw",
    "prodigyplus_d0": 1e-6,
    "prodigyplus_use_stableadamw": True,
    "optimizer_args": None,
    "use_t5_token_weights": True,
    "flow_shift": 3.0,
    "schedule_shift": 1.0,
    "timestep_sampling": "logit_normal",
    "timestep_mix_low_prob": 0.25,
    "adaptive_timestep": False,
    "adaptive_timestep_metric": "raw",
    "adaptive_timestep_highfreq_weight": 0.25,
    "adaptive_timestep_bins": 16,
    "adaptive_timestep_ema_decay": 0.95,
    "adaptive_timestep_burn_in": 160,
    "adaptive_timestep_min_factor": 0.5,
    "adaptive_timestep_max_factor": 2.0,
    "adaptive_timestep_base_mix": 0.25,
    "adaptive_timestep_candidate_mult": 8,
    "adaptive_timestep_low_noise_gate": False,
    "adaptive_timestep_gate_n": 3.0,
    "adaptive_timestep_gate_c": 0.05,
    "min_snr_gamma": 0.0,
    "loss_weighting_scheme": "none",
    "weight_cap_ratio": 5.0,
    "loss_type": "mse",
    "huber_c": 0.1,
    "huber_schedule": "constant",
    "noise_offset": 0.0,
    "noise_offset_min": 0.0,
    "noise_offset_random_strength": False,
    "pyramid_noise_iterations": 0,
    "pyramid_noise_discount": 0.3,
    "caption_dropout_rate": 0.0,
    "grad_norm_log_every": 0,
    "lora_exclude_patterns": None,
    "lora_include_patterns": None,
    "lora_reg_dims": None,
    "lora_reg_alphas": None,
    "lora_reg_lrs": None,
    "rank_dropout": 0.0,
    "module_dropout": 0.0,
    # T-LoRA 默认值（lora_variant != 'tlora' 时被忽略）
    "tlora_rmin_ratio": 0.5,
    "tlora_alpha": 1.0,
    "tlora_init": "ortho",
    "tlora_lokr_experimental": False,
    "tlora_lokr_ortho_init": False,
    "loraplus_lr_ratio": 1.0,
    "freq_balanced_dropout_strength": 0.0,
    "style_profile": "",
    "detail_inv_t_min": 1.0,
    "detail_inv_t_max": 5.0,
    # 辅助 loss 默认全关，配合 build_aux_loss_config 在 enabled=False 时彻底 no-op
    "aux_spectral_enabled": False,
    "aux_spectral_lambda": 0.05,
    "aux_spectral_use_wavelet": False,
    "aux_spectral_wavelet_lambda": 0.05,
    "aux_spectral_t_gate": 0.7,
    "aux_perceptual_enabled": False,
    "aux_perceptual_lambda_lpips": 0.1,
    "aux_perceptual_lambda_dino": 0.01,
    "aux_perceptual_t_gate": 0.7,
    "aux_perceptual_lpips_net": "vgg",
    "aux_perceptual_dino_local_path": "",
    "aux_perceptual_cache_dir": "",
    "aux_perceptual_use_checkpoint": True,
    "aux_perceptual_lpips_size": 0,
}


# ============================================================================
# Style profile presets
# ============================================================================

# 不同画师风格对 timestep / loss 权重栈的反应不同：
#   - sharp（清晰硬朗，高对比）：当前 train_my.yaml 默认值最适合
#   - hazy（雾蒙蒙 / 低饱和 / 滤镜风）：当前默认会出现细节溶解，因为 detail_inv_t
#     在低 t 把权重拉到 5×、adaptive_timestep 又把高 loss bin 反复采样，低饱和图的
#     "细节"本来就是低对比，被这两层加权 + adaptive 反复学习反而被磨平
#   - balanced：完全关掉 detail_inv_t 和 mixed_uniform_low，纯 uniform 训练（最保守）
#
# 用户在 YAML 顶层加 `style_profile: hazy` 即可应用对应 override；未设此字段时不做改动，
# 完全沿用 YAML 的其它字段（向后兼容）。
STYLE_PROFILES = {
    "sharp": {
        # 当前默认值，留作显式记录方便对照
        "loss_weighting_scheme": "detail_inv_t",
        "detail_inv_t_min": 1.0,
        "detail_inv_t_max": 5.0,
        "timestep_sampling": "mixed_uniform_low",
        "timestep_mix_low_prob": 0.15,
        "adaptive_timestep_base_mix": 0.25,
        "adaptive_timestep_max_factor": 2.0,
    },
    "hazy": {
        # 低饱和 / 雾蒙蒙画师：把 detail_inv_t 的强度收一半，让 adaptive 更保守，
        # 减少 mixed_uniform_low 的低 t 占比 —— 整体把"低 t 多看"的偏置降下来。
        "loss_weighting_scheme": "detail_inv_t",
        "detail_inv_t_min": 1.0,
        "detail_inv_t_max": 3.0,
        "timestep_sampling": "mixed_uniform_low",
        "timestep_mix_low_prob": 0.08,
        "adaptive_timestep_base_mix": 0.5,
        "adaptive_timestep_max_factor": 1.5,
    },
    "balanced": {
        # 关掉所有低 t 偏置，纯 uniform；最接近"原 logit_normal 加一点 uniform"的早期成功配置。
        "loss_weighting_scheme": "none",
        "timestep_sampling": "uniform",
        "timestep_mix_low_prob": 0.0,
        "adaptive_timestep_base_mix": 1.0,  # 等价禁用 adaptive 重采样
    },
}


def _apply_style_profile(args, config: dict | None = None):
    """根据 `style_profile` 覆盖一组 timestep/loss 相关参数。

    优先级：style_profile override > YAML 其它字段 > argparse 默认。
    既不破坏用户的细粒度自定义（不设 style_profile 就完全不动），又能一行切换风格。
    """
    profile_name = (getattr(args, "style_profile", "") or "").strip().lower()
    if not profile_name:
        return
    if profile_name not in STYLE_PROFILES:
        logger.warning(
            "Unknown style_profile=%s, ignoring. Available: %s",
            profile_name, ", ".join(STYLE_PROFILES.keys()),
        )
        return

    overrides = STYLE_PROFILES[profile_name]
    applied = []
    for attr, value in overrides.items():
        # 用户在 YAML 显式写了这个字段就尊重用户（细粒度优先）
        if config and attr in config and config[attr] is not None:
            continue
        old = getattr(args, attr, None)
        setattr(args, attr, value)
        applied.append(f"{attr}={value} (was {old})")
    logger.info(
        "[style_profile=%s] 应用了 %d 项 override: %s",
        profile_name, len(applied), "; ".join(applied),
    )


DEPRECATED_V5_KEYS = {
    "ip_noise_gamma",
    "ip_noise_target",
    "ip_noise_gamma_decay_steps",
    "immiscible_pool_size",
    "lora_ema_decay",
    "highfreq_loss_weight",
    "t_binned_module_groups",
    "timestep_mix_extreme_prob",
    "pyramid_zero_dc",
}


def _is_active_deprecated_value(value):
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return abs(float(value)) > 0.0
    if isinstance(value, str):
        return value.strip().lower() not in ("", "0", "false", "none", "null", "off")
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    return True


def apply_yaml_config(args, config):
    """将 YAML 配置应用到 args，命令行参数优先。

    判断"命令行是否显式设置"的方式：把 `args.<attr>` 与 `DEFAULTS[<attr>]` 比对，
    若相等或为 None 则视为没显式设，YAML 值生效。这种 heuristic 对绝大部分参数有效。
    """
    ignored = sorted(
        k for k in DEPRECATED_V5_KEYS
        if k in config and _is_active_deprecated_value(config.get(k))
    )
    if ignored:
        logger.warning(
            "Ignoring deprecated v5 training keys: %s. They no longer affect training.",
            ", ".join(ignored),
        )

    for yaml_key, arg_attr in YAML_TO_ARGS.items():
        if yaml_key not in config:
            continue
        yaml_value = config[yaml_key]
        if yaml_value is None:
            continue

        current_value = getattr(args, arg_attr, None)
        default_value = DEFAULTS.get(arg_attr)

        # 列表类型的默认值用 [] 表示，但 argparse 未定义时返回 None
        if current_value == default_value or current_value is None:
            setattr(args, arg_attr, yaml_value)

    # style_profile 在 wd 解析前应用 —— 它可能覆盖 timestep/loss 字段，但不动 wd
    _apply_style_profile(args, config)
    _resolve_weight_decay(args, config)

    return args


def _resolve_weight_decay(args, config: dict | None = None):
    """统一 `weight_decay` 来源，避免顶层 wd 与 optimizer_args.weight_decay 矛盾时的隐性吞值。

    历史 bug：当 YAML 顶层 `weight_decay: 0.0` 与 `optimizer_args.weight_decay: 0.01`
    同时存在时，`injector.get_param_groups(wd=args.weight_decay)` 会把每个 param group
    强制写成 wd=0.0，而 PyTorch 优化器规则是 per-group wd 完全覆盖 default wd。结果就是
    `optimizer_args.weight_decay` 被静默吞掉，真实 wd ≡ 0。

    解决（按用户偏好）：让 `optimizer_args.weight_decay` 作为权威来源。

    关键：要区分"用户在 YAML 显式写了 weight_decay"和"argparse 默认值 0.01"。前者是用户
    设置，需要参与冲突判断；后者只是兜底，不算"两个都设了"。靠 raw config dict 来判断。
    """
    opt_args = getattr(args, "optimizer_args", None) or {}
    if not isinstance(opt_args, dict):
        return

    user_set_top = bool(config and "weight_decay" in config and config["weight_decay"] is not None)
    user_set_opt = "weight_decay" in opt_args and opt_args["weight_decay"] is not None

    top_wd = getattr(args, "weight_decay", None)
    opt_wd = opt_args.get("weight_decay", None)

    # 两个都是用户显式设的 → 冲突就 warn，否则静默用 optimizer_args 的值
    if user_set_top and user_set_opt:
        if abs(float(top_wd) - float(opt_wd)) > 1e-12:
            logger.warning(
                "weight_decay 在 YAML 中被设置了两次（顶层=%s, optimizer_args=%s）。"
                "优先使用 optimizer_args.weight_decay=%s；顶层 %s 被忽略。"
                "建议把顶层的 weight_decay 删除或改成与 optimizer_args 一致，避免歧义。",
                top_wd, opt_wd, opt_wd, top_wd,
            )
        args.weight_decay = float(opt_wd)
        return

    # 只有 optimizer_args 是用户设的 → 把它同步回顶层
    if user_set_opt:
        args.weight_decay = float(opt_wd)
        return

    # 只有顶层是用户设的 → 把它推进 optimizer_args，建立 single source of truth
    if user_set_top:
        opt_args["weight_decay"] = float(top_wd)
        args.optimizer_args = opt_args
        return

    # 两个都没在 YAML 中显式设 → 不做任何事，沿用 argparse 默认值（顶层默认 0.01）。
    # 顶层默认值会通过 `create_optimizer(..., weight_decay=args.weight_decay)` 流向 opt_args。
