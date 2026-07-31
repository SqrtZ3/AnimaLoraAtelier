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
    # 模型族（DiT family）。"anima"（默认）/"krea2"（Krea 2 单流 MMDiT，opt-in）。
    # krea2 时 transformer_path 指 raw.safetensors，vae_path 仍是 Qwen-Image VAE（与
    # anima 同一文件），text_encoder_path/t5_tokenizer_path 不用，改用 krea2_text_encoder_path。
    "model_family": "model_family",
    "krea2_text_encoder_path": "krea2_text_encoder",
    "krea2_text_max_length": "krea2_text_max_length",
    "krea2_text_cache_entries": "krea2_text_cache_entries",
    "krea2_res_shift": "krea2_res_shift",
    "krea2_shift_min_res": "krea2_shift_min_res",
    "krea2_shift_max_res": "krea2_shift_max_res",
    "krea2_shift_y1": "krea2_shift_y1",
    "krea2_shift_y2": "krea2_shift_y2",
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
    "token_bucket": "token_bucket",
    "token_bucket_counts": "token_bucket_counts",
    "token_bucket_max_aspect_ratio": "token_bucket_max_aspect_ratio",
    "token_bucket_min_dim": "token_bucket_min_dim",
    "token_bucket_max_dim": "token_bucket_max_dim",
    "torch_compile": "torch_compile",
    "compile_mode": "compile_mode",
    "compile_dynamic": "compile_dynamic",
    # 冻结底模 Linear 量化（opt-in, default-off；见 docs/base-quant.md）
    "base_quant": "base_quant",
    "base_quant_gemm": "base_quant_gemm",
    "base_quant_fp8_scale": "base_quant_fp8_scale",
    "base_quant_fp8_grad": "base_quant_fp8_grad",
    "base_quant_include": "base_quant_include",
    "base_quant_skip": "base_quant_skip",
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
    "navit_packing": "navit_packing",
    "navit_token_budget": "navit_token_budget",
    "navit_max_images_per_pack": "navit_max_images_per_pack",
    "navit_text_trim_padding": "navit_text_trim_padding",
    "navit_pack_strategy": "navit_pack_strategy",
    "navit_pack_ffd_window": "navit_pack_ffd_window",
    "navit_pack_cost_lambda": "navit_pack_cost_lambda",
    "navit_pack_cost_ref_tokens": "navit_pack_cost_ref_tokens",
    "navit_drop_last": "navit_drop_last",
    "navit_native_resolution": "navit_native_resolution",
    "navit_multiscale": "navit_multiscale",
    "navit_multiscale_token_ladder": "navit_multiscale_token_ladder",
    "navit_multiscale_loss_weight": "navit_multiscale_loss_weight",
    "cache_encode_tiled": "cache_encode_tiled",
    "cache_encode_tile_px": "cache_encode_tile_px",
    "cache_encode_tile_overlap": "cache_encode_tile_overlap",
    "cache_encode_max_pixels": "cache_encode_max_pixels",
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
    "lokr_w1_init_std": "lokr_w1_init_std",
    "lokr_w1_lr_ratio": "lokr_w1_lr_ratio",
    "lokr_compute_dtype": "lokr_compute_dtype",
    "abba_alpha": "abba_alpha",
    "abba_export_kr": "abba_export_kr",
    "lora_variant": "lora_variant",
    "lora_init": "lora_init",
    "dora_export_mode": "dora_export_mode",
    "dora_fast_norm": "dora_fast_norm",
    "dora_detach_norm": "dora_detach_norm",
    # Layer A：导出期 SVD 压缩（save() 额外写 .compressed.safetensors 部署件）
    "lora_compress_energy": "lora_compress_energy",
    "lora_compress_max_rank": "lora_compress_max_rank",
    "lora_compress_budget_mb": "lora_compress_budget_mb",
    "lora_compress_replace_main": "lora_compress_replace_main",
    # Layer B：AC-LoRA 训练期 RESTART（arXiv:2504.02231）
    "aclora_enabled": "aclora_enabled",
    "aclora_restart_every": "aclora_restart_every",
    "aclora_warmup_steps": "aclora_warmup_steps",
    "aclora_p_mode": "aclora_p_mode",
    "aclora_p_start": "aclora_p_start",
    "aclora_p_end": "aclora_p_end",
    "aclora_p_floor": "aclora_p_floor",
    "aclora_total_steps": "aclora_total_steps",
    "aclora_loss_ema_beta": "aclora_loss_ema_beta",
    "aclora_reset_optimizer_state": "aclora_reset_optimizer_state",
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
    "grad_checkpoint_skip_last": "grad_checkpoint_skip_last",
    "grad_checkpoint_policy": "grad_checkpoint_policy",
    "num_workers": "num_workers",
    # 输出与保存
    "output_dir": "output_dir",
    "output_name": "output_name",
    "save_every": "save_every",
    "save_every_steps": "save_every_steps",
    "save_every_reference_steps": "save_every_reference_steps",
    "save_state_every": "save_state_every",
    "resume_state": "resume_state",
    # resume 时是否跳过该 epoch 内已消费的 batch（默认 true = 从中断处接续）。
    # 设 false 回到旧行为：从 checkpoint 所在 epoch 的开头重跑整个 epoch。
    "resume_skip_consumed_batches": "resume_skip_consumed_batches",
    "seed": "seed",
    # 采样
    "sample_every": "sample_every",
    "sample_steps": "sample_steps",
    "sample_prompt": "sample_prompt",
    "sample_prompts": "sample_prompts",
    # 预览提示词从训练集 caption 随机取（opt-in）。off=只用 sample_prompts（默认，
    # 与改动前等价）/ mix=按 ratio 概率混用 / only=全部取自训练集。
    "sample_dataset_prompts": "sample_dataset_prompts",
    "sample_dataset_prompt_ratio": "sample_dataset_prompt_ratio",
    "sample_dataset_prompt_pick": "sample_dataset_prompt_pick",
    "sample_dataset_prompt_count": "sample_dataset_prompt_count",
    "sample_dataset_prompt_suffix": "sample_dataset_prompt_suffix",
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
    "debug_nan_hooks": "debug_nan_hooks",
    "debug_nan_hooks_abort": "debug_nan_hooks_abort",
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
    "timestep_laplace_mu": "timestep_laplace_mu",
    "timestep_laplace_b": "timestep_laplace_b",
    # Style-Friendly logSNR 采样 (arXiv 2411.14793) + t 截断 + 分层采样
    "timestep_logsnr_mu": "timestep_logsnr_mu",
    "timestep_logsnr_sigma": "timestep_logsnr_sigma",
    "timestep_t_min": "timestep_t_min",
    "timestep_t_max": "timestep_t_max",
    "timestep_stratified": "timestep_stratified",
    # CSFlow（arXiv 2606.08833）：数据集功率谱×人眼CSF→逐t采样权重；timestep_sampling: csflow 启用
    "csflow_rapsd_path": "csflow_rapsd_path",
    "csflow_alpha": "csflow_alpha",
    "csflow_pixels_per_degree": "csflow_pixels_per_degree",
    "csflow_rapsd_max_images": "csflow_rapsd_max_images",
    # 训练内遥测总线（trainer/telemetry.py，图盲 opt-in default-off）
    "telemetry_enabled": "telemetry_enabled",
    "telemetry_freq_bands": "telemetry_freq_bands",
    "telemetry_slope_window": "telemetry_slope_window",
    "telemetry_optimizer_every": "telemetry_optimizer_every",
    "telemetry_capacity_every": "telemetry_capacity_every",
    # 训练步分阶段计时（trainer/stage_timer.py，opt-in default-off）—定位 NaViT vs
    # ARB 速度根因。CUDA event 计时 GPU 阶段、perf_counter 计时 CPU/IO；仅被采样步 sync。
    "stage_timing_every": "stage_timing_every",
    "stage_timing_warmup": "stage_timing_warmup",
    "stage_profile_step": "stage_profile_step",
    "stage_profile_trace": "stage_profile_trace",
    "navit_attn_backend": "navit_attn_backend",
    "attn_force_autocast_dtype": "attn_force_autocast_dtype",
    "base_quant_fuse_act_quant": "base_quant_fuse_act_quant",
    # LoRA-One 谱对齐初始化 (arXiv 2502.01235, KPSVD→LoKr)
    "lora_one_init_steps": "lora_one_init_steps",
    "lora_one_init_scale": "lora_one_init_scale",
    # LWD 小波显著性 time-gated 掩码 (arXiv 2506.00433)
    "lwd_mask_enabled": "lwd_mask_enabled",
    "lwd_mask_floor": "lwd_mask_floor",
    # 按 tag 覆盖 dropout 概率（YAML dict，仅 TXT caption 路径），如
    # tag_dropout_overrides: {"close-up": 0.5} —— 解开"特征绑定到条件 tag"
    "tag_dropout_overrides": "tag_dropout_overrides",
    # TREAD token 路由 (arXiv 2501.04765)
    "tread_enabled": "tread_enabled",
    "tread_ratio": "tread_ratio",
    "tread_start_layer": "tread_start_layer",
    "tread_end_layer": "tread_end_layer",
    # 三峰采样高噪路由 / 退火 / VeCoR 负样本 / 固定网格 eval
    "timestep_mix_high_prob": "timestep_mix_high_prob",
    "timestep_mix_anneal_start": "timestep_mix_anneal_start",
    "timestep_mix_anneal_end": "timestep_mix_anneal_end",
    "timestep_mix_low_prob_end": "timestep_mix_low_prob_end",
    "timestep_mix_high_prob_end": "timestep_mix_high_prob_end",
    "dfm_mode": "dfm_mode",
    "eval_every": "eval_every",
    "eval_count": "eval_count",
    "eval_t_grid": "eval_t_grid",
    "eval_seed": "eval_seed",
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
    # slope 模式专用：慢 EMA 衰减
    "adaptive_timestep_slope_slow_decay": "adaptive_timestep_slope_slow_decay",
    "min_snr_gamma": "min_snr_gamma",
    "loss_weighting_scheme": "loss_weighting_scheme",
    "weight_cap_ratio": "weight_cap_ratio",
    "loss_type": "loss_type",
    "huber_c": "huber_c",
    "huber_schedule": "huber_schedule",
    "huber_snr_clamp_max": "huber_snr_clamp_max",
    "dfm_lambda": "dfm_lambda",
    "eisbach_lambda": "eisbach_lambda",
    "noise_offset": "noise_offset",
    "noise_offset_min": "noise_offset_min",
    "noise_offset_random_strength": "noise_offset_random_strength",
    "pyramid_noise_iterations": "pyramid_noise_iterations",
    "pyramid_noise_discount": "pyramid_noise_discount",
    "immiscible_enabled": "immiscible_enabled",
    "immiscible_k": "immiscible_k",
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
    "aux_self_perceptual_enabled": "aux_self_perceptual_enabled",
    "aux_self_perceptual_lambda": "aux_self_perceptual_lambda",
    "aux_self_perceptual_t_gate": "aux_self_perceptual_t_gate",
    "aux_self_perceptual_tap_block": "aux_self_perceptual_tap_block",
    "aux_self_perceptual_encode_t": "aux_self_perceptual_encode_t",
    "aux_perceptual_enabled": "aux_perceptual_enabled",
    "aux_perceptual_lambda_lpips": "aux_perceptual_lambda_lpips",
    "aux_perceptual_lambda_dino": "aux_perceptual_lambda_dino",
    "aux_perceptual_t_gate": "aux_perceptual_t_gate",
    "aux_perceptual_lpips_net": "aux_perceptual_lpips_net",
    "aux_perceptual_dino_local_path": "aux_perceptual_dino_local_path",
    "aux_perceptual_cache_dir": "aux_perceptual_cache_dir",
    "aux_perceptual_use_checkpoint": "aux_perceptual_use_checkpoint",
    "aux_perceptual_lpips_size": "aux_perceptual_lpips_size",
    "aux_lpl_enabled": "aux_lpl_enabled",
    "aux_lpl_lambda": "aux_lpl_lambda",
    "aux_lpl_t_gate": "aux_lpl_t_gate",
    "aux_lpl_outlier_k": "aux_lpl_outlier_k",
    "aux_lpl_use_checkpoint": "aux_lpl_use_checkpoint",
    "aux_lpl_num_scales": "aux_lpl_num_scales",
    "aux_lpl_max_decode_px": "aux_lpl_max_decode_px",
    # ── GAF 梯度一致性过滤（脏数据鲁棒性 B1；默认关）──
    "gaf_enabled": "gaf_enabled",
    "gaf_every": "gaf_every",
    "gaf_warmup": "gaf_warmup",
    "gaf_mode": "gaf_mode",
    "gaf_threshold": "gaf_threshold",
    "gaf_temp": "gaf_temp",
    "gaf_floor": "gaf_floor",
    "gaf_min_keep": "gaf_min_keep",
    "gaf_trust_decay": "gaf_trust_decay",
    "gaf_log_path": "gaf_log_path",
    "gaf_backend": "gaf_backend",
    "gaf_proj_dim": "gaf_proj_dim",
    # ── Linear-DPO 偏好放大（in-loop, on-policy；默认关）见 trainer/dpo.py ──
    "dpo_enabled": "dpo_enabled",
    "dpo_beta": "dpo_beta",
    "dpo_eta": "dpo_eta",
    "dpo_ref_ema": "dpo_ref_ema",
    "dpo_regen_every": "dpo_regen_every",
    "dpo_loser_steps": "dpo_loser_steps",
    "dpo_loser_cfg": "dpo_loser_cfg",
    "dpo_loser_subset": "dpo_loser_subset",
    "dpo_sft_anchor_lambda": "dpo_sft_anchor_lambda",
    "dpo_share_noise": "dpo_share_noise",
    "dpo_log_path": "dpo_log_path",
    # NCP-DPO（arXiv 2406.17636）：感知特征空间损失（默认 latent = 原 Linear-DPO）见 trainer/ncp.py
    "dpo_loss_space": "dpo_loss_space",
    "ncp_tap_block": "ncp_tap_block",
    "ncp_dt": "ncp_dt",
    # ── Dispersive Loss 中间表征排斥正则（arXiv 2506.09027；默认关）见 trainer/dispersive.py ──
    "dispersive_enabled": "dispersive_enabled",
    "dispersive_lambda": "dispersive_lambda",
    "dispersive_tau": "dispersive_tau",
    "dispersive_tap_block": "dispersive_tap_block",
    "dispersive_variant": "dispersive_variant",
    # ── LeapAlign 两步跳跃自蒸馏（SFT 主目标增强；默认关）见 trainer/leap.py ──
    "leap_enabled": "leap_enabled",
    "leap_ratio": "leap_ratio",
    "leap_nested_grad_coe": "leap_nested_grad_coe",
    "leap_min_gap": "leap_min_gap",
    "leap_traj_sim_weighting": "leap_traj_sim_weighting",
    "leap_traj_sim_min": "leap_traj_sim_min",
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
    "token_bucket": False,
    "token_bucket_counts": "4032,4200",
    "token_bucket_max_aspect_ratio": 2.0,
    "token_bucket_min_dim": 512,
    "token_bucket_max_dim": 2016,
    "torch_compile": False,
    "compile_mode": None,
    "compile_dynamic": None,
    # ── 注意力精度口径（Anima family；opt-in, default-off）──────────────────
    # false（默认）= 只在 q/k/v dtype 不一致时归一（那是 LoRA 注入后 xformers 直接
    # ValueError 的必需修复，其余路径逐 bit 不变）。
    # true = autocast 开着时一律按 autocast dtype 算注意力。针对 NaViT 块对角路径：
    # 注入 LoRA 后 LayerNorm(fp32) → 被包的 q/k/v_proj 全吐 fp32 → 自注意力跑 fp32
    # kernel，而 dense/eval/采样走 SDPA 一直是 bf16。开了两条路口径才一致，并省下
    # navit 注意力的时间/显存；代价是该处数值从 fp32 变 bf16（需要 A/B 复核画质）。
    "attn_force_autocast_dtype": False,
    # ── 冻结底模 Linear 量化（opt-in, default-off）──────────────────────────
    # "none"（默认，行为与历史逐 bit 等价）/ "fp8"（e4m3 权重，H20 可走 fp8 GEMM）
    # / "fp4"（nvfp4 权重，Blackwell 可走 fp4 GEMM；H20 上自动退 dequant-bf16，
    # 仍有 4× 权重显存收益）。只量化冻结底模，LoRA adapter 全程 bf16。
    "base_quant": "none",
    # "auto"（探测到可用 _scaled_mm 就走量化 GEMM，否则 dequant-bf16）
    # / "on"（强制量化 GEMM，探测不到就报错）/ "off"（只省显存，不改计算）
    "base_quant_gemm": "auto",
    # fp8 scale 粒度："auto"（优先 rowwise，H20/sm90 支持；不支持退 tensorwise）
    # / "rowwise" / "tensorwise"
    "base_quant_fp8_scale": "auto",
    # 反向 dL/dx 也走 fp8 GEMM（e5m2 梯度 × e4m3 权重）。默认 false=反向 bf16
    # （更稳）；开了前反向都吃 fp8 吞吐，梯度多一层量化噪声 → 需 A/B 验证
    "base_quant_fp8_grad": False,
    # 量化哪些层（regex fullmatch 逻辑层名）。null → family 默认：
    # krea2=blocks.*/txtfusion.*/txtmlp.*（对齐推理端已验证画质的集合；
    # first/last/tproj/tmlp 保持 bf16），anima=blocks.*
    "base_quant_include": None,
    # 激活量化融合：torch.compile 只编译量化纯函数（abs→amax→div→clamp→cast
    # 融成 1-2 个 kernel），不碰模型整图。H20 profile 实测该链每步吃数秒。
    # 数值可能有 ulp 级差异（启用探针校验 dequant allclose），默认关。
    "base_quant_fuse_act_quant": False,
    # 在 include 基础上额外排除的 regex 列表（默认不排除）
    "base_quant_skip": None,
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
    # NaViT/Patch-n-Pack block-diagonal packing (opt-in, default-off). When enabled,
    # heterogeneous images are packed into one block-diagonal forward up to
    # navit_token_budget tokens (sum of per-image token counts), each with its own
    # timestep — decouples effective batch from per-image shape. token_budget=0 means
    # "must be set explicitly"; size it to VRAM (see docs/navit-packing.md).
    "navit_packing": False,
    "navit_token_budget": 0,
    "navit_max_images_per_pack": 0,
    # When True, pack each image's caption to its *valid* T5 length (from the T5
    # attention mask) instead of the full 512-pad. KEEP THIS OFF — the trade it was
    # meant to make does not exist: the upside is a fraction of a percent (text_encode
    # is 0.23% of a real navit step) while the downside is a measured train/eval
    # condition mismatch (training drops the 512-pad, eval/sampling/ARB keep it → the
    # cross-attn conditioning differs; A/B showed eval_loss spiking and worse fitting,
    # which reverted when turned off). Default False = byte-identical to the legacy
    # 512-pad path. See docs/navit-packing.md §2.0.
    "navit_text_trim_padding": False,
    # Pack-assembly strategy. "next_fit" (default) = the original order-preserving greedy
    # packer (byte-identical to before). "ffd" = First-Fit-Decreasing within shuffled
    # windows — packs fuller (fewer steps, less wasted budget) at the cost of some
    # batch-composition variety; main upside when image sizes are heterogeneous.
    "navit_pack_strategy": "next_fit",
    # FFD window size (images). Each epoch's shuffled order is split into windows of this
    # size and FFD runs inside each, so packs still vary across epochs. 0 = one global
    # window (max fill but epoch-static packs). Ignored when strategy != "ffd".
    "navit_pack_ffd_window": 256,
    # Cost-based packing (opt-in, default-off / byte-identical at 0.0). A pack's step time
    # is NOT linear in its summed tokens: attention is quadratic in each image's own
    # sequence length, so at a fixed ΣN a pack of few large images costs far more than one
    # of many small images (measured on H20 / Krea2 12B: 3960 ms at G=1 vs 1719 ms at
    # G=16 for the same ΣN=55778 — a 2.3× spread). With λ>0 the packer budgets on
    # cost(n) = n·(1+λ·n)/(1+λ·n_ref) instead of n, so oversized images take up more of
    # the budget (no step-time / VRAM spikes) and small ones take less (more images per
    # step). Fitted λ ≈ 2.7e-05 for Krea2 12B from tests/diag_navit_speed.py S5
    # (R²=0.9999), independently corroborated at 3.3e-05 by fitting a real run's
    # stage_timing.csv (R²=0.972). Re-fit it per model/card; see docs/navit-packing.md.
    "navit_pack_cost_lambda": 0.0,
    # Reference image size (tokens) for the cost normalisation above: an image of exactly
    # this size costs exactly its token count, so a uniformly-sized dataset keeps today's
    # pack capacity and only the size *spread* is repriced. 0 = auto (dataset median).
    "navit_pack_cost_ref_tokens": 0,
    # Drop the final (under-budget) pack each epoch. Default False: for packing the last
    # pack always holds real images, so dropping it wastes data on small datasets. This
    # is navit-specific and decoupled from bucket_drop_last (which drops incomplete ARB
    # batches, a different notion).
    "navit_drop_last": False,
    # Size each packed image at its *native* resolution (only the VAE+patch 16px-multiple
    # constraint), instead of quantizing to the ARB bucket grid. Forces floor alignment
    # (crop down to a 16px multiple ⇒ zero padding ⇒ no per-image mask needed; the navit
    # cached path carries no padding mask). Requires cache_latents. The longest single
    # side is still bounded by the model's RoPE cap (max_img_h/max_img_w); when those are
    # auto (0), they are derived from the dataset's largest image. Only meaningful with
    # navit_packing=true; default False keeps the existing ARB-bucket sizing.
    "navit_native_resolution": False,
    # NaViT multiscale ladder (opt-in, default-off; requires navit_packing +
    # navit_native_resolution). For every image whose native token count exceeds a
    # ladder entry, cache an extra aspect-preserving *downscaled* copy at ≤ that many
    # tokens and pack it as a regular dataset entry (deterministic: each image sees
    # each ladder scale exactly once per epoch). Fills otherwise-wasted pack budget
    # when native images are large relative to navit_token_budget, and exposes the
    # LoRA to the style at inference-scale token densities (mitigates the
    # train-large/infer-small scale shift). NaViT paper's resolution-sampling
    # counterpart (arXiv 2307.06304). Never upscales.
    "navit_multiscale": False,
    # Comma list / YAML list of per-copy token budgets, e.g. "4096" (~1024² px) or
    # "4096,16384". Each entry must be ≤ navit_token_budget.
    "navit_multiscale_token_ladder": "4096",
    # Per-image loss weight multiplier for the downscaled copies (native copies stay
    # 1.0). 1.0 = equal per-image weight (default; NaViT-paper-style). <1.0 keeps the
    # native scale dominant in the gradient while still exposing small scales.
    "navit_multiscale_loss_weight": 1.0,
    # Tiled VAE encode for the latent-cache phase (opt-in, default-off). Images whose
    # pixel count exceeds the cache encode budget (4M px) are encoded tile-by-tile
    # (overlapping pixel tiles → per-tile VAE encode → feather-blended latent stitch),
    # capping the one-off cache-phase VRAM spike at ~tile_px² instead of the full
    # native image (a 2814x4456 image is 12.5M px and can spike tens of GB when
    # encoded whole). APPROXIMATE at tile seams (VAE conv receptive field crosses
    # tile borders); larger overlap → smaller error. Images within the budget keep
    # the whole-image path byte-identical. Both knobs must be 16px multiples.
    "cache_encode_tiled": False,
    "cache_encode_tile_px": 1024,
    "cache_encode_tile_overlap": 128,
    # Pixel budget per VAE-encode call in the cache phase (flip copies included).
    # 0 = built-in conservative default (4M px ≈ two 1024² images with flip, sized
    # for small GPUs). Raising it lets same-shape small images batch deeper (e.g.
    # 16M on an 80GB card → 8×1024²-with-flip per call) — pure speed, no behavior
    # change per image. Also the cache_encode_tiled trigger threshold: only images
    # *above* this budget are tiled, so raising it also means e.g. a 12.5M-px image
    # is encoded whole under a 16M budget (make sure VRAM allows).
    "cache_encode_max_pixels": 0,
    "alpha_handling": "none",
    "alpha_background": "neutral",
    "alpha_threshold": 0.01,
    # ── 模型族（opt-in；"anima" 与历史行为逐一等价）─────────────────────────
    "model_family": "anima",
    # Krea2 专用：Qwen3-VL-4B-Instruct 的 HuggingFace 目录（model_family=krea2 必填）
    "krea2_text_encoder": "",
    # 官方 encoder 的 max_length（模板前缀/后缀之外的用户文本 token 预算）。
    # 512=官方 encoder.py 默认值（右截断，超长 caption 丢尾部）；<=0=无上限（opt-in）：
    # 不截断、动态 pad，超长 caption 全保留（ComfyUI 部署同样不截时用它对齐训练/推理）。
    "krea2_text_max_length": 512,
    # Krea2 文本特征 LRU cache 条目数（每条 [L,12,2560]，压缩后典型 tag caption 数 MB；
    # 128 条 ≈ 1GB 量级 VRAM，按数据集 caption 数量与显存调）
    "krea2_text_cache_entries": 128,
    # 训练 t 的分辨率感知 shift（Krea2 原生调度 α=exp(mu(seq_len))；仅 krea2 生效。
    # true=对齐官方推理时间步分布（musubi 的 krea2_shift 同物）；false=退回原始 t）
    "krea2_res_shift": True,
    # mu 线性内插端点（官方 sampling.py 默认：256px→y1、1280px→y2）
    "krea2_shift_min_res": 256,
    "krea2_shift_max_res": 1280,
    "krea2_shift_y1": 0.5,
    "krea2_shift_y2": 1.15,
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
    # LoKr 的 Kronecker 因子。in/out 各被切成 factor 份，ΔW=kron(w1(f,f), w2)。
    # f 不能整除某层维度时会自动下调到 ≤f 的最大公约因子（启动日志里会列实际分布）。
    # ★ f 直接决定"w1 冻在 init 时"的表达力上界 1/f²：f4=6.25% / f6=2.78% / f8=1.56%。
    #   第三方成功的 Krea2 LoKr 等效 factor 只有 4~6。f 越小 → 约束越弱、w2 越大、体积越大。
    "lokr_factor": 8,
    # w1 初始化的正态 std。默认 0.1（rms≈0.097）= 与改动前逐字节一致。
    # 依据（本地取证）：第三方成功件学成后 |w1|rms 中位 0.49、90% 分位 1.63；
    # w1 只有 f² 个参数、AdamW 每步至多走 lr，lr=1e-4 时从 0.097 爬到 0.49 要 ≥3900 步，
    # 典型 run（数百~千步）来不及 → 把起点直接设到目标量级是最便宜的补救。
    # step-0 中立只靠 w2_b=0 保证，与该值无关，怎么调都不破坏中立性。
    "lokr_w1_init_std": 0.1,
    # w1 的 lr 倍率，独立于 loraplus_lr_ratio（后者只抬 w2_b，**不抬 w1**）。
    # 1.0 = 与改动前一致。w1 是乘性块间门控，抬太猛会让 ΔW 剧烈摆动，建议从 4~16 起试。
    "lokr_w1_lr_ratio": 1.0,
    # LoKr 训练期中间量精度。"fp32"（默认）= 历史行为：把 x 与三个因子物化成 fp32。
    # "native" = 直接用参数原 dtype（bf16），省显存：
    #   ① x.reshape(...) 退化成 view，不再拷贝 N×in_features 的 fp32 副本
    #      （krea2 mlp.down @65536 token 就是 2.13 GB）；
    #   ② 两个 N×out_features 的中间量各省一半字节。
    # 本地实测（RTX 5070 Laptop，krea2 mlp.gate 6144→16384, f=4, r=32, N=4096）：
    #   峰值显存 −18.0%；fwd+bwd 耗时 2.9×（bf16 走 tensor core、fp32 不走）；
    #   输出相对差 5.1e-3（≈1.3 个 bf16 ulp）、梯度余弦相似度 0.99999。
    # 加速比在云端可能不同（取决于该卡 fp32/bf16 吞吐比与 TF32 设置），显存比值更稳。
    # 依据：w1/w2_a/w2_b 本来就是 bf16 参数，.float() 不增加信息量，而 torch 的
    # bf16 matmul 内部本就是 fp32 累加 → native 的数值口径与标准 LoRA 路径一致。
    # 与 tlora_lokr_ortho_init 互斥（补偿项要精确抵消，bf16 抵消不干净），构造期 fail-fast。
    "lokr_compute_dtype": "fp32",
    # ABBA（lora_type='abba'）：alpha1=alpha2 的统一覆盖；None → 官方口径 alpha=r（r=rank//2）
    "abba_alpha": None,
    # save() 是否额外写 KR 物化标准 LoRA 键（文件 ~8×，云端下载不友好）。
    # 默认 False：只存 native 因子（体积=同预算 LoRA），本地用 tools/abba_export_lora.py 转部署件
    "abba_export_kr": False,
    "lora_variant": "base",
    "lora_init": "default",
    "dora_export_mode": "native",
    "dora_fast_norm": False,
    "dora_detach_norm": False,
    # Layer A：导出期 SVD 压缩。三项全为默认值 → 关闭（行为中立，不写压缩件）。
    # 开启后每次标准 LoRA save() 额外写一份 SVD 截断的 .compressed.safetensors 部署件，
    # 主件（满 rank）不变、供 resume。仅 lora_type=lora + variant=base 可用。
    #
    # 两种分配策略，互斥（同开会构造期 fail-fast）：
    #  · lora_compress_budget_mb（推荐）：给定目标体积，全局 σ²/字节最优分配逐层 rank。
    #    Krea2 c12port epoch19 实测同等保留能量下比逐层阈值小 1.3–3.2×；
    #    35MB 档高频保留 0.997、8.9MB 档 0.975（ComfyUI 3 seed×2 prompt 配对实拍）。
    #  · lora_compress_energy / max_rank：逐层能量阈值（旧策略，保留兼容）。
    #
    # lora_compress_replace_main=true：不写满 rank 主件，epoch 成品直接就是压缩件。
    #   省磁盘与下载带宽，但满 rank 权重此后**只存在于 training_state(.pt)**，
    #   因此要求 save_state_every>0（否则 anima_train.py 构造期 fail-fast），
    #   且续训必须走 --resume-state（从压缩件 load 会 fail-fast）。
    "lora_compress_energy": 1.0,
    "lora_compress_max_rank": 0,
    "lora_compress_budget_mb": 0.0,
    "lora_compress_replace_main": False,
    # Layer B：AC-LoRA 训练期 RESTART。默认关。仅 lora_type=lora + variant=base + init=default。
    "aclora_enabled": False,
    "aclora_restart_every": 200,     # 每多少 optimizer step 做一次 RESTART（论文 E=10 epoch 的 step 类比）
    "aclora_warmup_steps": 200,      # 此前不 RESTART（保 step-0 中立 + 让信号先形成）
    "aclora_p_mode": "schedule",     # "schedule"=FM 稳健(p_start→p_end 线性)；"loss"=论文 p=1-l^α（假设 loss<1）
    "aclora_p_start": 0.7,           # schedule 模式起始保留能量占比
    "aclora_p_end": 0.99,            # schedule 模式终点保留能量占比
    "aclora_p_floor": 0.5,           # p 下限（防某步把层几乎清空）
    "aclora_total_steps": 0,         # schedule/loss 的进度分母；0=恒用 p_end / α=1（stop-anytime 友好）
    "aclora_loss_ema_beta": 0.98,    # loss EMA 平滑（仅 loss 模式用）
    "aclora_reset_optimizer_state": False,  # RESTART 后是否清零被改写参数的优化器动量（默认否=忠实论文）
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
    # 分块 grad checkpoint：最后 N 个 transformer block 不做 checkpoint（存全部激活、
    # backward 不重算），其余照常。0 = 全部 checkpoint（默认，与改动前逐字节等价）。
    # 用途：显存有富余时把它换成吞吐——每跳过一层省一次该层重算，数学恒等。
    # 每层激活量随 pack token 数线性增长，设置前先按显存余量估算（krea2 12B @16k
    # token 约 3.8GB/层）。仅 navit 打包路径 + krea2 模型族已接线。
    # ⚠ H20 实测的「时间×显存」前沿上，skip_last=8/12 被 grad_checkpoint_policy 的
    # sac_narrow **完全支配**（更慢且更占显存）——优先用下面那个键。
    "grad_checkpoint_skip_last": 0,
    # 选择性激活重算（SAC）策略。full（默认）= 现状：checkpoint 丢掉 block 内全部中间量、
    # backward 全部重算（重算代价 ≈ 一整个 forward，实测 bwd/fwd=2.73）。其余档保留贵的
    # matmul/SDPA 输出、只重算便宜的 norm/silu/rope/逐元素，用显存换掉大部分重算。
    # H20 28 块真栈实测（tests/diag_navit_ckpt_policy.py --g-sweep）：
    #     策略          ms/token   激活 MB/token
    #     full            0.674        0.72
    #     sac_attn        0.653        1.03
    #     sac_narrow      0.562        2.43     ← 稳妥档（+20% 吞吐）
    #     sac_all         0.484        4.07     ← 激进档（+39% 吞吐）
    #     (无 ckpt 对照)  0.465        7.52
    # 关键：块对角 attention 下 per-token 代价只取决于每图 seqlen、与一包几张图无关
    # （G=1..6 变化 <1%），所以**降 navit_token_budget 不损吞吐、只让出显存**——这正是
    # 换取更便宜策略的本钱。用法：把 budget 降到 1~2 张图的 token 数，再开 sac_*。
    # 数学恒等（只改哪些中间量被保存 vs 重算），有等价性单测。仅 krea2 模型族已接线。
    "grad_checkpoint_policy": "full",
    "num_workers": 0,
    "output_dir": "./output",
    "output_name": "anima_lora",
    "save_every": 0,
    "save_every_steps": 0,
    "save_every_reference_steps": 0,
    "save_state_every": 0,
    "resume_state": "",
    "resume_skip_consumed_batches": True,
    "seed": 42,
    "sample_every": 0,
    "sample_steps": 0,
    "sample_prompt": "1girl, masterpiece",
    "sample_prompts": [],
    "sample_dataset_prompts": "off",
    "sample_dataset_prompt_ratio": 0.5,
    "sample_dataset_prompt_pick": "each",
    "sample_dataset_prompt_count": 4,
    "sample_dataset_prompt_suffix": "",
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
    "debug_nan_hooks": False,
    "debug_nan_hooks_abort": True,
    "optimizer_type": "adamw",
    "prodigyplus_d0": 1e-6,
    "prodigyplus_use_stableadamw": True,
    "optimizer_args": None,
    "use_t5_token_weights": True,
    "flow_shift": 3.0,
    "schedule_shift": 1.0,
    "timestep_sampling": "logit_normal",
    "timestep_mix_low_prob": 0.25,
    "timestep_laplace_mu": 0.0,
    "timestep_laplace_b": 0.5,
    "timestep_logsnr_mu": -6.0,
    "timestep_logsnr_sigma": 2.0,
    "timestep_t_min": 0.0,
    "timestep_t_max": 1.0,
    "timestep_stratified": False,
    "csflow_rapsd_path": "",
    "csflow_alpha": 1.0,
    "csflow_pixels_per_degree": 50.0,
    "csflow_rapsd_max_images": 512,
    "telemetry_enabled": False,
    "telemetry_freq_bands": 3,
    "telemetry_slope_window": 6,
    "telemetry_optimizer_every": 0,
    "telemetry_capacity_every": 0,
    # 训练步分阶段计时 cadence（步）；0=关（noop 计时器，零开销/行为中立）。开启后每 N 步
    # 用 CUDA event 计时各阶段、末尾一次 sync 写 stage_timing.csv，定位 NaViT vs ARB 速度根因。
    "stage_timing_every": 0,
    # 跳过前 warmup 步不计时（cudnn autotune / cache 冷），避免冷启动污染统计；仅 every>0 时生效。
    "stage_timing_warmup": 10,
    # 在第 N 个 micro-step 用 torch.profiler 采样完整一步，区分「kernel 慢」vs
    # 「kernel 间空隙（CPU/分配器）」；0=关（默认，行为中立）。trace=额外导出 chrome json。
    "stage_profile_step": 0,
    "stage_profile_trace": False,
    # krea2 navit packed attention 后端：xformers（默认，历史行为逐 bit 不变）|
    # sdpa_seg（逐段 dense SDPA/cudnn，数学恒等，H20 上更快）。
    "navit_attn_backend": "xformers",
    "lora_one_init_steps": 0,
    "lora_one_init_scale": 0.01,
    "lwd_mask_enabled": False,
    "lwd_mask_floor": 0.3,
    "timestep_mix_high_prob": 0.25,
    "timestep_mix_anneal_start": 0,
    "timestep_mix_anneal_end": 0,
    "timestep_mix_low_prob_end": -1.0,
    "timestep_mix_high_prob_end": -1.0,
    "dfm_mode": "batch",
    "tag_dropout_overrides": None,
    "tread_enabled": False,
    "tread_ratio": 0.3,
    "tread_start_layer": 3,
    "tread_end_layer": -4,
    "eval_every": 0,
    "eval_count": 4,
    "eval_t_grid": "0.1,0.3,0.5,0.7,0.9",
    "eval_seed": 1234,
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
    "adaptive_timestep_slope_slow_decay": -1.0,
    "min_snr_gamma": 0.0,
    "loss_weighting_scheme": "none",
    "weight_cap_ratio": 5.0,
    "loss_type": "mse",
    "huber_c": 0.1,
    "huber_schedule": "constant",
    "huber_snr_clamp_max": 10.0,
    "dfm_lambda": 0.0,
    "eisbach_lambda": 0.0,
    "noise_offset": 0.0,
    "noise_offset_min": 0.0,
    "noise_offset_random_strength": False,
    "pyramid_noise_iterations": 0,
    "pyramid_noise_discount": 0.3,
    # Improved Immiscible Diffusion（arXiv 2505.18521）KNN 噪声选择；默认关 = no-op。
    # 开启后逐样本从 immiscible_k 个候选噪声中选与 latent 最近者（收敛加速，见 objective.py）。
    "immiscible_enabled": False,
    "immiscible_k": 4,
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
    # LPL：Latent Perceptual Loss（arXiv 2411.04873，默认关 → no-op）。冻结 VAE decoder
    # 中间特征空间对齐 pred/target x₀，专治 latent-MSE 丢高频质感。零外部模型；
    # 成本 ≈ 每个过 gate 样本 2 次 VAE decode。λ=0.1 是保守起点（论文未在小数据 LoRA
    # 场景验证，需 A/B 标定，开训看日志目标 ≈ 主 loss 的 10–30%）。
    "aux_lpl_enabled": False,
    "aux_lpl_lambda": 0.1,
    "aux_lpl_t_gate": 0.6,
    "aux_lpl_outlier_k": 8.0,
    "aux_lpl_use_checkpoint": True,
    "aux_lpl_num_scales": 4,
    "aux_lpl_max_decode_px": 1048576,
    # Self-Perceptual SFT（arXiv 2401.00110，默认关 → no-op）。冻结 DiT 编码栈特征空间距离，
    # 罚"糊/不像"（均值回归）比 latent-MSE 狠。复用 ncp.perceptual_features（编码器=当前模型、
    # adapter 冻梯度）；λ 起 0.05–0.1 且开训看日志标定（特征空间量纲未知，目标 ≈ 主 loss 的 10–30%）。
    # ★与 eisbach 的低 t 偏置有交互：t_gate 偏低会和 Eisbach 一起往低 t 堆 → 盯构图丰富度别被压回。
    "aux_self_perceptual_enabled": False,
    "aux_self_perceptual_lambda": 0.1,
    "aux_self_perceptual_t_gate": 0.5,
    "aux_self_perceptual_tap_block": -1,
    "aux_self_perceptual_encode_t": 0.05,
    # GAF 梯度一致性过滤（默认关 → 完全 no-op）
    "gaf_enabled": False,
    "gaf_every": 4,
    "gaf_warmup": 100,
    "gaf_mode": "soft",
    "gaf_threshold": 0.0,
    "gaf_temp": 0.15,
    "gaf_floor": 0.3,
    "gaf_min_keep": 2,
    "gaf_trust_decay": 0.9,
    "gaf_log_path": "",
    "gaf_backend": "autograd",
    "gaf_proj_dim": 16,
    # Linear-DPO 偏好放大（默认关 → 完全 no-op）。dpo_beta / dpo_regen_every 每 run 由用户调：
    # beta 起小（0.1），regen_every ≈ 一个 epoch 的步数（依 batch/数据集大小，3–5 轮）。
    "dpo_enabled": False,
    "dpo_beta": 0.1,
    "dpo_eta": 0.01,
    "dpo_ref_ema": 1.0,
    "dpo_regen_every": 1000,
    "dpo_loser_steps": 14,
    "dpo_loser_cfg": 1.0,
    "dpo_loser_subset": 1.0,
    "dpo_sft_anchor_lambda": 0.0,
    "dpo_share_noise": True,
    "dpo_log_path": "",
    # NCP-DPO（默认 latent = 原 Linear-DPO，零改动）。perceptual = 冻结编码栈特征空间。
    # ncp_tap_block: 取第几个 block 后的中间激活当感知特征；-1=自动取中间块 n//2。
    # ncp_dt: flow-matching 一步反演步长 t−t'（论文 t'=t−1 的连续类比）。
    "dpo_loss_space": "latent",
    "ncp_tap_block": -1,
    "ncp_dt": 0.05,
    # Dispersive Loss 中间表征排斥（默认关 → 完全 no-op）。在某中间 block 的隐表征空间做
    # "无正样本对的排斥"，反表征坍缩，与输出空间的 ΔFM/VeCoR、输出端 Eisbach 机理正交。
    # 接线复用 ncp.perceptual_features 截断前向（grads on，不冻结 → 梯度回流 LoRA）；dense
    # 路径限定，leap/dpo 步跳过。dispersive_lambda 论文默认 0.5，本仓库小数据 LoRA 起 0.05–0.1
    # （同 spectral/self-perceptual：辅助 loss 比论文调小一个量级起跑），开训看日志标定。
    # tap_block: -1=自动取中间块 n//2（沿用 ncp 语义）；论文建议前 1/4 ≈ n//4（28 块 → ~7），
    # 想贴论文请显式设。variant: infonce_l2（论文最优）/ infonce_cosine（尺度无关）。
    "dispersive_enabled": False,
    "dispersive_lambda": 0.0,
    "dispersive_tau": 0.5,
    "dispersive_tap_block": -1,
    "dispersive_variant": "infonce_l2",
    # LeapAlign 两步跳跃自蒸馏（默认关）。leap_ratio<1.0 = hybrid（非 leap 步保留三峰/
    # adaptive/aux/ΔFM，leap 步做轨迹蒸馏）；=1.0 = 纯 leap（adaptive_timestep 会变惰性）。
    "leap_enabled": False,
    "leap_ratio": 1.0,
    "leap_nested_grad_coe": 0.3,
    "leap_min_gap": 0.1,
    "leap_traj_sim_weighting": False,
    "leap_traj_sim_min": 0.1,
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
