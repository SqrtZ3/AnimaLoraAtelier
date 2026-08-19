r"""把**同一份训练 yaml** 翻成 TPU 后端的配置，并对没移植的功能 fail-fast。

## 这个模块的唯一职责：不让"悄悄少训了一个功能"发生

TPU 后端是 PyTorch 训练器的**子集**。子集本身不是问题，问题是子集的边界不可见 ——
用户拿着一份开了 12 个开关的 yaml 跑上来，其中 2 个没移植，训练照跑、loss 照降、
产物照出，几天后才从画面上发现不对。这在本仓库是明令要防的一类错（skill 第 5 节
fail-fast）。

所以这里的规则是：

  * **每一个会改变训练数学的 yaml 键都必须在下面的表里出现**（`_HANDLED` /
    `_CACHE_SIDE` / `_IGNORED` / `_UNPORTED` 四类之一）；
  * 落在 `_UNPORTED` 且**被打开**的键 -> 直接 raise，消息里写清为什么没移植、
    以及替代路径；
  * 表里完全没有的键 -> 收集起来报"未知键"，而不是默默忽略。

`--allow-unported` 可以把 raise 降级成警告，但它要显式传，且会打印一份清单。

## 全局 token 预算 vs 单卡预算

yaml 的 `navit_token_budget` 在 GPU 上是**一步一个 pack** 的预算。TPU 是 8 卡纯
DP、每卡一个 pack，所以 yaml 的 131072 在这里被解释成**全局**预算，单卡拿
131072/8 = 16384 —— 恰好是 v5e 单 chip(15.7GiB) 装得下的量级（真机 anima-mem-probe：
scan+full 档 budget 16384 可跑，32768 OOM）。这样"一步看多少 token"在两个后端上
是同一个数，梯度噪声的量级可比。不整除时 fail-fast，不四舍五入。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from . import adapters as AD
    from . import auxloss as X
    from . import flow as F
    from . import optim as O
    from . import sched as S
    from . import train as T
except ImportError:
    import adapters as AD
    import auxloss as X
    import flow as F
    import optim as O
    import sched as S
    import train as T


# ── 键的四分类 ────────────────────────────────────────────────────────────────
#: 已实现，直接进配置。
_HANDLED = {
    "transformer_path", "data_dir", "repeats", "seed",
    "navit_packing", "navit_native_resolution", "navit_token_budget",
    "navit_multiscale", "navit_pack_strategy", "navit_text_trim_padding",
    "navit_max_images_per_pack", "navit_multiscale_loss_weight",
    "flip_augment", "caption_dropout_rate",
    "lora_type", "lora_rank", "lora_alpha", "lokr_factor", "lora_variant",
    "lora_targets", "lora_exclude_patterns", "lora_reg_dims", "lora_reg_alphas",
    "rank_dropout", "module_dropout", "loraplus_lr_ratio", "lokr_w1_init_std",
    "lokr_compute_dtype",
    "optimizer_type", "learning_rate", "optimizer_args",
    "epochs", "max_steps", "grad_accum", "grad_clip_max_norm", "warmup_steps",
    "mixed_precision", "grad_checkpoint",
    "flow_shift", "schedule_shift", "timestep_sampling",
    "timestep_mix_low_prob", "timestep_mix_high_prob",
    "timestep_logsnr_mu", "timestep_logsnr_sigma",
    "timestep_t_min", "timestep_t_max", "timestep_stratified",
    "timestep_laplace_mu", "timestep_laplace_b",
    "timestep_mix_anneal_start", "timestep_mix_anneal_end",
    "timestep_mix_low_prob_end", "timestep_mix_high_prob_end",
    "adaptive_timestep", "adaptive_timestep_metric", "adaptive_timestep_bins",
    "adaptive_timestep_ema_decay", "adaptive_timestep_slope_slow_decay",
    "adaptive_timestep_burn_in", "adaptive_timestep_min_factor",
    "adaptive_timestep_max_factor", "adaptive_timestep_base_mix",
    "adaptive_timestep_candidate_mult", "adaptive_timestep_low_noise_gate",
    "adaptive_timestep_gate_n", "adaptive_timestep_gate_c",
    "adaptive_timestep_highfreq_weight",
    "loss_type", "huber_c", "huber_schedule", "huber_snr_clamp_max",
    "loss_weighting_scheme", "min_snr_gamma",
    "detail_inv_t_min", "detail_inv_t_max",
    "immiscible_enabled", "immiscible_k",
    "eisbach_lambda", "dfm_lambda", "dfm_mode",
    "aux_spectral_enabled", "aux_spectral_lambda", "aux_spectral_use_wavelet",
    "aux_spectral_wavelet_lambda", "aux_spectral_t_gate",
    "eval_every", "eval_count", "eval_t_grid", "eval_seed",
    "output_dir", "output_name", "save_every", "save_every_steps",
    "save_state_every", "resume_state",
    "log_every",
}

#: 由 **PyTorch 侧的离线缓存**负责，TPU 训练时不再需要（读的是缓存产物）。
#: 这些键在这里是 no-op，但**不是**"没实现"——语义已经烘焙进 npz 了。
_CACHE_SIDE = {
    "vae_path", "text_encoder_path", "t5_tokenizer_path", "resolution",
    "bucket_base_resos", "bucket_min_base_reso", "bucket_max_base_reso",
    "bucket_base_reso_steps", "min_bucket_reso", "max_bucket_reso",
    "bucket_reso_steps", "bucket_max_aspect_ratio", "bucket_no_upscale",
    "bucket_max_upscale", "bucket_report", "max_img_h", "max_img_w",
    "cache_latents", "cache_encode_tiled", "cache_encode_max_pixels",
    "navit_multiscale_token_ladder", "keep_tokens", "prefer_json",
    "fit_max_tokens", "fit_warn_tokens",
}

#: 与 TPU 后端无关（PyTorch/CUDA 专属，或纯展示）。
_IGNORED = {
    "batch_size", "num_workers", "torch_compile", "attn_force_autocast_dtype",
    "keep_vae_on_gpu", "empty_cache_after_sample", "no_progress", "no_monitor",
    "monitor_host", "monitor_port", "no_browser", "loss_curve_steps",
    "stage_timing_every", "stage_timing_warmup",
    "disable_tlora_hooks", "use_precommit_lora_forward",
    "use_per_block_checkpoint", "telemetry_freq_bands", "telemetry_slope_window",
    "telemetry_optimizer_every", "telemetry_capacity_every",
    "bucket_drop_last", "effective_batch_size", "lora_one_init_scale",
    # 纯展示/日志：TPU 侧每 log_every 步就打一次 gnorm，没有独立的遥测通道。
    # 归在这里而不是 _HANDLED —— 它们不改训练数学，但也确实没被读。
    "grad_norm_log_every", "telemetry_enabled",
}

#: 没移植。值 = (判定"是否被打开"的函数, 原因与替代路径)。
_UNPORTED: Dict[str, Tuple[Any, str]] = {
    "sample_every": (lambda v: int(v or 0) > 0,
                     "训练期采样出图未移植（要在 TPU 上跑完整 ODE 采样器 + VAE 解码）。"
                     "设 sample_every: 0，改用导出的 LoRA 在本地出图。"),
    "tread_enabled": (bool, "TREAD 未移植。"),
    "dispersive_enabled": (bool, "Dispersive loss 未移植（要额外一次截断前向）。"),
    "aux_self_perceptual_enabled": (bool, "self-perceptual 未移植（要额外模型前向）。"),
    "aux_perceptual_enabled": (bool, "LPIPS/DINO 感知 loss 未移植（要 VAE 解码 + 外部网络）。"),
    "aux_lpl_enabled": (bool, "LPL 未移植。"),
    "lwd_mask_enabled": (bool, "LWD 小波掩码未移植。"),
    "noise_offset": (lambda v: float(v or 0) > 0,
                     "noise_offset 未移植（NaViT 下要逐图在各自网格上加低频偏移）。"),
    "pyramid_noise_iterations": (lambda v: int(v or 0) > 0,
                                 "pyramid noise 未移植（同上，需逐图多尺度插值）。"),
    "shuffle_caption": (bool, "caption 洗牌要在编码前做，而 TPU 侧文本特征是离线缓存的。"
                              "要它就在缓存阶段做，或留 false。"),
    "tag_dropout": (lambda v: float(v or 0) > 0, "同 shuffle_caption：属于编码前的操作。"),
    "freq_balanced_dropout_strength": (lambda v: float(v or 0) > 0, "同上。"),
    "lora_dropout": (lambda v: float(v or 0) > 0,
                     "适配器输入侧 dropout 未移植（rank_dropout / module_dropout 已移植）。"),
    "lora_one_init_steps": (lambda v: int(v or 0) > 0, "LoRA-One 初始化未移植。"),
    "weight_cap_ratio": (lambda v: float(v or 0) > 0, "loss 权重上限比未移植。"),
    "fit_packed_training": (bool, "FiT 打包训练与 NaViT 打包是两条路，TPU 侧只走 NaViT。"),
    "token_bucket": (bool, "token_bucket 是 PyTorch DataLoader 侧的调度，TPU 侧走 packing.py。"),
    "dora_export_mode": (lambda v: str(v or "native").lower() != "native",
                         "TPU 侧的 export.py 只写 native（`dora_scale` 单独一个键）。"
                         "trainer/lora.py:1485 的另外两档 diff / merged_model 要在导出时"
                         "物化 ΔW 或整模合并，没移植。留 native，或用 tools/ 下的转换脚本"
                         "在本地转。"),
    "resume_lora": (lambda v: bool(str(v or "").strip()),
                    "从 safetensors 里读回 LoRA 未实现（export.py 只有写，没有读；"
                    "逐块 rank 还要按 rmax 重新补齐，键对不上会静默变成"
                    "\"从零开始训\"）。续训请用 resume_state —— 它连优化器一二阶矩"
                    "与自适应采样器的 EMA 一起存，才是真正的接棒。"),
}


#: 未移植功能的**子参数族**：主开关关着时，它的一堆子参数是死配置，不该被当成
#: "未知键"报错。主开关开着时由 `_UNPORTED` 那条负责 raise，这里不重复。
_FAMILY: Tuple[Tuple[str, str], ...] = (
    ("aux_self_perceptual_", "aux_self_perceptual_enabled"),
    ("aux_perceptual_", "aux_perceptual_enabled"),
    ("aux_lpl_", "aux_lpl_enabled"),
    ("dispersive_", "dispersive_enabled"),
    ("lwd_", "lwd_mask_enabled"),
    ("tread_", "tread_enabled"),
    ("noise_offset", "noise_offset"),
    ("pyramid_noise_", "pyramid_noise_iterations"),
    ("tag_dropout", "tag_dropout"),
)


@dataclass
class RunConfig:
    """一次 TPU 训练需要的全部东西。`tcfg` 是设备侧的，其余是 host 侧的。"""
    tcfg: T.TrainConfig
    adaptive: S.AdaptiveConfig
    data_dir: Path
    transformer_path: Path
    output_dir: Path
    output_name: str
    devices: int = 8
    #: **单卡** token 预算（= yaml 的 navit_token_budget / devices）
    budget: int = 16384
    quantum: int = 1024
    txt_len: int = 512
    max_images_per_pack: int = 0
    repeats: int = 1
    multiscale: bool = False
    ms_loss_weight: float = 1.0
    flip_prob: float = 0.0
    caption_dropout: float = 0.0
    epochs: int = 200
    max_steps: int = 0
    #: 每几个 epoch 存一次 LoRA（yaml 的 `save_every`）。0 = 只在 max_steps 收尾时存。
    save_every: int = 1
    save_every_steps: int = 0
    save_state_every: int = 0
    eval_every: int = 0
    eval_count: int = 0
    eval_t_grid: Tuple[float, ...] = ()
    eval_seed: int = 1234
    log_every: int = 1
    resume_state: str = ""
    #: 未移植但被显式放行（--allow-unported）的键，训练开始时要再打印一次。
    waived: Tuple[str, ...] = ()


def load_yaml(path) -> Dict[str, Any]:
    import yaml
    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    if not isinstance(d, dict):
        raise ValueError(f"{path} 不是一个 yaml 映射")
    return d


def _f(d, k, default=0.0):
    v = d.get(k, default)
    return float(default if v is None else v)


def _i(d, k, default=0):
    v = d.get(k, default)
    return int(default if v is None else v)


def _b(d, k, default=False):
    v = d.get(k, default)
    return bool(default if v is None else v)


def build(d: Dict[str, Any], devices: int = 8, allow_unported: bool = False,
          canvas_hw: Tuple[int, int] = (0, 0)) -> RunConfig:
    """yaml dict -> RunConfig。`canvas_hw` 由数据侧扫描后回填（aux_spectral 用）。"""
    waived = _check_coverage(d, allow_unported)
    _check_navit(d)

    # ── 适配器 ───────────────────────────────────────────────────────────────
    targets = _expand_targets(d.get("lora_targets") or [],
                              d.get("lora_exclude_patterns") or [])
    acfg = AD.AdapterConfig(
        kind=str(d.get("lora_type", "lora")).lower(),
        variant=str(d.get("lora_variant", "base") or "base").lower(),
        rank=_i(d, "lora_rank", 32),
        alpha=(None if d.get("lora_alpha") is None else _f(d, "lora_alpha")),
        factor=_i(d, "lokr_factor", 8),
        rank_dropout=_f(d, "rank_dropout"),
        module_dropout=_f(d, "module_dropout"),
        w1_init_std=_f(d, "lokr_w1_init_std", 0.1),
        compute_dtype=str(d.get("lokr_compute_dtype", "fp32") or "fp32").lower(),
        reg_dims={str(k): int(v) for k, v in (d.get("lora_reg_dims") or {}).items()},
        reg_alphas={str(k): float(v) for k, v in (d.get("lora_reg_alphas") or {}).items()},
    )

    # ── 优化器 ───────────────────────────────────────────────────────────────
    opt_t = str(d.get("optimizer_type", "adamw")).lower()
    if opt_t not in ("adamw", "adamw_snr", "adamwsnr"):
        raise ValueError(
            f"optimizer_type={opt_t!r} 未移植到 TPU 后端。已实现 adamw / adamw_snr"
            f"（后者含 SNR 锐化与 cautious 掩码）。别指望它悄悄退回 adamw —— 那会让"
            f"同一份 yaml 在两个后端上是两个优化器。")
    oa = d.get("optimizer_args") or {}
    betas = oa.get("betas") or [0.9, 0.999]
    adamw = O.AdamWConfig(
        lr=_f(d, "learning_rate", 1e-4),
        b1=float(betas[0]), b2=float(betas[1]),
        eps=float(oa.get("eps", 1e-8)),
        weight_decay=float(oa.get("weight_decay", 0.0)),
        max_grad_norm=_f(d, "grad_clip_max_norm", 1.0),
        warmup_steps=_i(d, "warmup_steps", 0),
        lora_plus_ratio=_f(d, "loraplus_lr_ratio", 1.0),
        snr_power=float(oa.get("snr_power", 1.0)),
        cautious=bool(oa.get("cautious", False)),
    )

    # ── flow / loss ──────────────────────────────────────────────────────────
    fcfg = F.FlowConfig(
        t_mode=str(d.get("timestep_sampling", "logit_normal")),
        flow_shift=_f(d, "flow_shift", 3.0),
        schedule_shift=_f(d, "schedule_shift", 1.0),
        logsnr_mu=_f(d, "timestep_logsnr_mu", -6.0),
        logsnr_sigma=_f(d, "timestep_logsnr_sigma", 2.0),
        mix_low_prob=_f(d, "timestep_mix_low_prob", 0.5),
        mix_high_prob=_f(d, "timestep_mix_high_prob", 0.25),
        laplace_mu=_f(d, "timestep_laplace_mu", 0.0),
        laplace_b=_f(d, "timestep_laplace_b", 0.5),
        t_min=_f(d, "timestep_t_min", 0.0),
        t_max=_f(d, "timestep_t_max", 1.0),
        stratified=_b(d, "timestep_stratified"),
        mix_anneal_start=_i(d, "timestep_mix_anneal_start", 0),
        mix_anneal_end=_i(d, "timestep_mix_anneal_end", 0),
        mix_low_prob_end=_f(d, "timestep_mix_low_prob_end", -1.0),
        mix_high_prob_end=_f(d, "timestep_mix_high_prob_end", -1.0),
        loss_type=str(d.get("loss_type", "mse")).lower(),
        huber_c=_f(d, "huber_c", 0.1),
        huber_schedule=str(d.get("huber_schedule", "constant")).lower(),
        huber_snr_clamp_max=_f(d, "huber_snr_clamp_max", 10.0),
        weighting=str(d.get("loss_weighting_scheme", "none")).lower(),
        min_snr_gamma=_f(d, "min_snr_gamma", 0.0) or 5.0,
        detail_inv_t_min=_f(d, "detail_inv_t_min", 1.0),
        detail_inv_t_max=_f(d, "detail_inv_t_max", 5.0),
        immiscible_k=(_i(d, "immiscible_k", 4) if _b(d, "immiscible_enabled") else 1),
    )

    aux = X.AuxConfig(
        eisbach_lambda=_f(d, "eisbach_lambda"),
        dfm_lambda=_f(d, "dfm_lambda"),
        dfm_mode=str(d.get("dfm_mode", "vecor")).lower(),
        spectral_enabled=_b(d, "aux_spectral_enabled"),
        spectral_lambda=_f(d, "aux_spectral_lambda", 0.05),
        spectral_use_wavelet=_b(d, "aux_spectral_use_wavelet"),
        spectral_wavelet_lambda=_f(d, "aux_spectral_wavelet_lambda", 0.05),
        spectral_t_gate=_f(d, "aux_spectral_t_gate", 0.7),
        canvas_hw=tuple(canvas_hw),
    )

    adaptive = S.AdaptiveConfig(
        enabled=_b(d, "adaptive_timestep"),
        metric=str(d.get("adaptive_timestep_metric", "raw")).lower(),
        bins=_i(d, "adaptive_timestep_bins", 16),
        ema_decay=_f(d, "adaptive_timestep_ema_decay", 0.95),
        slope_slow_decay=_f(d, "adaptive_timestep_slope_slow_decay", -1.0),
        burn_in=_i(d, "adaptive_timestep_burn_in", 160),
        min_factor=_f(d, "adaptive_timestep_min_factor", 0.5),
        max_factor=_f(d, "adaptive_timestep_max_factor", 2.0),
        base_mix=_f(d, "adaptive_timestep_base_mix", 0.25),
        candidate_mult=_i(d, "adaptive_timestep_candidate_mult", 8),
        low_noise_gate=_b(d, "adaptive_timestep_low_noise_gate"),
        gate_n=_f(d, "adaptive_timestep_gate_n", 3.0),
        gate_c=_f(d, "adaptive_timestep_gate_c", 0.05),
        highfreq_weight=_f(d, "adaptive_timestep_highfreq_weight", 0.25),
    )

    tcfg = T.TrainConfig(
        adapter=acfg, targets=targets,
        remat=("full" if _b(d, "grad_checkpoint", True) else "none"),
        grad_accum=max(_i(d, "grad_accum", 1), 1),
        seed=_i(d, "seed", 0),
        flow=fcfg, aux=aux, adamw=adamw,
    )

    total_budget = _i(d, "navit_token_budget", 16384)
    if total_budget % devices:
        raise ValueError(
            f"navit_token_budget={total_budget} 不能被 {devices} 卡整除。"
            f"TPU 侧把它当**全局**预算（8 卡各跑一个 pack），单卡拿 1/{devices}；"
            f"不整除就没法保证一步看到的 token 数与 GPU 侧一致。改成 {devices} 的倍数。")

    return RunConfig(
        tcfg=tcfg, adaptive=adaptive,
        data_dir=Path(str(d.get("data_dir", "."))),
        transformer_path=Path(str(d.get("transformer_path", ""))),
        output_dir=Path(str(d.get("output_dir", "./output"))),
        output_name=str(d.get("output_name", "anima-tpu")),
        devices=devices, budget=total_budget // devices,
        max_images_per_pack=_i(d, "navit_max_images_per_pack", 0),
        repeats=max(_i(d, "repeats", 1), 1),
        multiscale=_b(d, "navit_multiscale"),
        ms_loss_weight=_f(d, "navit_multiscale_loss_weight", 1.0),
        flip_prob=(0.5 if _b(d, "flip_augment") else 0.0),
        caption_dropout=_f(d, "caption_dropout_rate"),
        epochs=_i(d, "epochs", 200), max_steps=_i(d, "max_steps", 0),
        save_every=_i(d, "save_every", 1),
        save_every_steps=_i(d, "save_every_steps", 0),
        save_state_every=_i(d, "save_state_every", 0),
        eval_every=_i(d, "eval_every", 0), eval_count=_i(d, "eval_count", 0),
        eval_t_grid=_parse_grid(d.get("eval_t_grid", "")),
        eval_seed=_i(d, "eval_seed", 1234),
        log_every=max(_i(d, "log_every", 1), 1),
        resume_state=str(d.get("resume_state", "") or ""),
        waived=tuple(waived),
    )


# ── 校验 ──────────────────────────────────────────────────────────────────────
def _check_coverage(d: Dict[str, Any], allow: bool) -> List[str]:
    """未移植项的门。返回被显式放行的键。"""
    on, unknown = [], []
    for k, v in d.items():
        if k in _UNPORTED:
            pred, why = _UNPORTED[k]
            if pred(v):
                on.append(f"  {k} = {v!r}\n      {why}")
            continue
        if k in _HANDLED or k in _CACHE_SIDE or k in _IGNORED:
            continue
        if k.startswith("sample_"):          # 采样出图的一整族，由 sample_every 统管
            continue
        if any(k.startswith(pre) for pre, _ in _FAMILY):
            continue                         # 未移植功能的子参数，主开关那条已经管了
        unknown.append(k)
    if unknown:
        raise ValueError(
            "yaml 里有 TPU 后端不认识的键：\n  " + ", ".join(sorted(unknown))
            + "\n它们可能是新加的功能。这里不静默忽略 —— 请把它们归到 config.py 的"
              " _HANDLED / _CACHE_SIDE / _IGNORED / _UNPORTED 之一，"
              "顺便确认它是不是真的不影响训练数学。")
    if on and not allow:
        raise ValueError(
            "以下功能在 yaml 里是开着的，但 TPU 后端没有移植：\n"
            + "\n".join(on)
            + "\n\n要么在 yaml 里关掉它们（那样两个后端才是同一个实验），"
              "要么传 --allow-unported 明确接受这份差异。")
    return [x.strip().split(" =")[0] for x in on]


def _check_navit(d: Dict[str, Any]) -> None:
    """NaViT 相关的硬前提。都是"开了会静默出错"的那一类。"""
    if not _b(d, "navit_packing", True):
        raise ValueError("TPU 后端只实现了 NaViT 打包路线（与分桶对照路线），"
                         "navit_packing 必须为 true")
    if _b(d, "navit_text_trim_padding"):
        raise ValueError(
            "navit_text_trim_padding=true：训练去掉 512-pad 而 eval/采样/ARB 都带 pad，"
            "cross-attn 条件不一致会让 eval_loss 冲高且拟合变差（memory "
            "[[navit-text-trim-train-eval-mismatch]] 的 A/B 实证）。TPU 侧不实现这条路。")
    strat = str(d.get("navit_pack_strategy", "ffd")).lower()
    if strat != "ffd":
        raise ValueError(f"navit_pack_strategy={strat!r} 未移植；packing.py 只实现了 ffd")
    mp = str(d.get("mixed_precision", "bf16")).lower()
    if mp not in ("bf16", "bfloat16"):
        raise ValueError(f"mixed_precision={mp!r}：TPU 后端只走 bf16 前向 + fp32 master")


def _expand_targets(names, excludes) -> Tuple[str, ...]:
    """yaml 的裸 target 名（`q_proj`）展开成内部全名（`self_attn.q_proj` 等）。

    PyTorch 侧 `lora_targets` 是按模块名子串匹配的，所以 `q_proj` 会同时命中
    self_attn 与 cross_attn；这里照此展开，再套 `lora_exclude_patterns`。
    """
    import re
    all_t = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
             "self_attn.output_proj", "cross_attn.q_proj", "cross_attn.k_proj",
             "cross_attn.v_proj", "cross_attn.output_proj",
             "mlp.layer1", "mlp.layer2")
    if not names:
        picked = list(all_t)
    else:
        picked = [t for t in all_t if any(str(n) in t for n in names)]
        miss = [n for n in names if not any(str(n) in t for t in all_t)]
        if miss:
            raise ValueError(f"lora_targets 里这些没匹配到任何模块：{miss}；"
                             f"可选 {list(all_t)}")
    pats = [re.compile(str(p)) for p in (excludes or [])]
    picked = [t for t in picked if not any(p.fullmatch(f"blocks.0.{t}") or p.search(t)
                                           for p in pats)]
    if not picked:
        raise ValueError("lora_targets 与 lora_exclude_patterns 相消，一个模块都没剩下")
    return tuple(picked)


def _parse_grid(s) -> Tuple[float, ...]:
    if not s:
        return ()
    return tuple(float(x) for x in str(s).replace(" ", "").split(",") if x)


def summary(rc: RunConfig) -> str:
    a, f, x = rc.tcfg.adapter, rc.tcfg.flow, rc.tcfg.aux
    lines = [
        f"数据 {rc.data_dir}  输出 {rc.output_dir}/{rc.output_name}",
        f"预算 全局 {rc.budget * rc.devices} = {rc.devices} 卡 x {rc.budget}/卡"
        f" | quantum {rc.quantum} | repeats {rc.repeats}"
        f" | multiscale {'on' if rc.multiscale else 'off'}"
        f" | flip {rc.flip_prob} | caption_dropout {rc.caption_dropout}",
        f"适配器 {a.kind}"
        + (f"(f={a.factor})" if a.kind == "lokr" else "")
        + f"/{a.variant} r{a.rank} a{a.alpha} rd{a.rank_dropout} md{a.module_dropout}"
        f" | targets {len(rc.tcfg.targets)}",
        f"优化器 lr={rc.tcfg.adamw.lr} betas=({rc.tcfg.adamw.b1},{rc.tcfg.adamw.b2})"
        f" wd={rc.tcfg.adamw.weight_decay} snr_power={rc.tcfg.adamw.snr_power}"
        f" cautious={rc.tcfg.adamw.cautious} clip={rc.tcfg.adamw.max_grad_norm}"
        f" grad_accum={rc.tcfg.grad_accum}",
        f"t 采样 {f.t_mode} shift={f.flow_shift} low={f.mix_low_prob}"
        f" high={f.mix_high_prob} logsnr=({f.logsnr_mu},{f.logsnr_sigma})"
        f" 分层={f.stratified} 范围=[{f.t_min},{f.t_max}]",
        f"loss {f.loss_type}(c={f.huber_c}, {f.huber_schedule}, clamp={f.huber_snr_clamp_max})"
        f" 加权={f.weighting} immiscible_k={f.immiscible_k}",
        f"aux eisbach={x.eisbach_lambda} dfm={x.dfm_lambda}({x.dfm_mode})"
        f" spectral={'on' if x.spectral_enabled else 'off'}"
        + (f"(λ={x.spectral_lambda}, wavelet={x.spectral_wavelet_lambda},"
           f" gate={x.spectral_t_gate}, 画布={x.canvas_hw})"
           if x.spectral_enabled else ""),
        f"自适应 {'on' if rc.adaptive.enabled else 'off'}"
        + (f" metric={rc.adaptive.metric} bins={rc.adaptive.bins}"
           f" burn_in={rc.adaptive.burn_in} base_mix={rc.adaptive.base_mix}"
           if rc.adaptive.enabled else ""),
        f"remat={rc.tcfg.remat} eval_every={rc.eval_every}"
        f" t 网格={list(rc.eval_t_grid)}",
    ]
    if rc.waived:
        lines.append(f"[!] 已放行的未移植项：{', '.join(rc.waived)} —— 这份 run 与 "
                     f"GPU 侧同 yaml **不是同一个实验**")
    return "\n".join(lines)
