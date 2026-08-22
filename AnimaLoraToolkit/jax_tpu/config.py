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
    from . import krea2_jax as K2
    from . import optim as O
    from . import sched as S
    from . import train as T
except ImportError:
    import adapters as AD
    import auxloss as X
    import flow as F
    import krea2_jax as K2
    import optim as O
    import sched as S
    import train as T


# ── 键的四分类 ────────────────────────────────────────────────────────────────
#: 已实现，直接进配置。
_HANDLED = {
    "transformer_path", "data_dir", "repeats", "seed",
    "model_family",
    "krea2_res_shift", "krea2_shift_min_res", "krea2_shift_max_res",
    "krea2_shift_y1", "krea2_shift_y2",
    "navit_packing", "navit_native_resolution", "navit_token_budget",
    "navit_multiscale", "navit_pack_strategy", "navit_text_trim_padding",
    # navit_max_images_per_pack 已挪到 _UNPORTED —— 它以前在这里，但全仓从未被
    # 消费（packing.Packer 的签名里没有这个量），属于"在表里却是 no-op"。
    "navit_multiscale_loss_weight",
    "flip_augment", "caption_dropout_rate",
    "lora_type", "lora_rank", "lora_alpha", "lokr_factor", "lora_variant",
    "lora_targets", "lora_exclude_patterns", "lora_reg_dims", "lora_reg_alphas",
    "rank_dropout", "module_dropout", "loraplus_lr_ratio", "lokr_w1_init_std",
    "lokr_compute_dtype",
    "optimizer_type", "learning_rate", "optimizer_args",
    "epochs", "max_steps", "grad_accum", "grad_clip_max_norm",
    # `warmup_steps` 是 **TPU 独有的 yaml 键**（GPU 侧 trainer/config.py 的
    # YAML_TO_ARGS 里没有它，那边用 lr_scheduler 那一套）。写进 GPU yaml 会被
    # 它的 warn_unrecognized_keys 报未知。
    "warmup_steps",
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
    "krea2_text_encoder_path", "krea2_text_max_length", "krea2_text_cache_entries",
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
    # GPU 侧采样有**三个**独立触发器（anima_train.py：sample_every 每 N epoch、
    # sample_steps 每 N step、sample_reference_steps）。以前只判 sample_every，而
    # `_check_coverage` 里 `startswith("sample_")` 那条豁免把后两个当"未移植子参数"
    # 一并跳过了 —— 于是 `sample_every: 0` + `sample_steps: 100` 静默通过，TPU 一张
    # 图都不出、无提示。这正是本文件 docstring 要防的那类。
    "sample_steps": (lambda v: int(v or 0) > 0,
                     "同 sample_every（按 step 触发的那个）。TPU 侧不出图，设 0。"),
    "sample_reference_steps": (lambda v: int(v or 0) > 0,
                               "同 sample_every（参考图那路）。TPU 侧不出图，设 0。"),
    # ── LR 调度：GPU 有 cosine / cosine_with_restart，TPU 只有 warmup + 常数 lr ──
    # `optim._lr_at` 的 docstring 解释了为什么有意不做衰减（用户范式是极大 epoch
    # 一直训、随时手停、从任意 step 挑 checkpoint，带衰减会让"第 N 步的 checkpoint"
    # 的含义依赖总步数）。但"有意不做"必须显式告知，不能静默忽略。
    "lr_scheduler": (lambda v: str(v or "constant").lower()
                     not in ("", "constant", "none"),
                     "LR 调度未移植：TPU 侧只有线性 warmup + 之后恒定"
                     "（optim._lr_at，有意如此 —— 见那里的 docstring）。"
                     "要衰减请在 GPU 侧跑，或接受常 LR。"),
    "lr_scheduler_t0": (lambda v: int(v or 0) > 0, "同 lr_scheduler。"),
    "lr_scheduler_t_mult": (lambda v: float(v or 1) != 1.0, "同 lr_scheduler。"),
    "lr_scheduler_eta_min": (lambda v: float(v or 0) > 0, "同 lr_scheduler。"),
    # ── 顶层 weight_decay：TPU 只读 optimizer_args.weight_decay ──────────────
    # GPU 侧有 `_resolve_weight_decay`（trainer/config.py）处理顶层 vs optimizer_args
    # 的权威性。TPU 侧顶层写法完全接不上 —— 用户以为设了 wd 其实是 0，最危险的一类。
    "weight_decay": (lambda v: float(v or 0) > 0,
                     "顶层 weight_decay 未接线：TPU 侧只读 "
                     "`optimizer_args.weight_decay`。请把它移进 optimizer_args。"),
    "lora_include_patterns": (lambda v: bool(v),
                              "include（exclude 的豁免通道）未移植：TPU 侧只有 "
                              "lora_targets + lora_exclude_patterns 两级。"
                              "请把要保留的目标直接写进 lora_targets。"),
    # ── 正则化数据集 ────────────────────────────────────────────────────────
    "reg_data_dir": (lambda v: bool(str(v or "").strip()),
                     "正则化数据集未移植（TPU 侧 CacheDataset 只吃一个 latent 缓存目录）。"),
    "reg_repeats": (lambda v: int(v or 0) > 0, "同 reg_data_dir。"),
    "reg_caption": (lambda v: bool(str(v or "").strip()), "同 reg_data_dir。"),
    "navit_max_images_per_pack": (
        lambda v: int(v or 0) > 0,
        "逐 pack 图数上限未移植：TPU 侧 packing.ffd 只按 token budget 装箱"
        "（Packer 的签名里没有这个量）。以前它被解析进 RunConfig 但**从未被消费** —— "
        "属于'在表里却是 no-op'，比报错更坏。要限图数请调小 navit_token_budget。"),
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
    # GPU 侧默认值是 5.0 且**默认生效**（trainer/config.py 的 DEFAULTS +
    # objective.py 的 compute_loss_weight 尾部按 weight_cap_ratio 夹权重）。
    # 谓词的语义是"这个键在 GPU 上会真的改变权重吗"：
    #   * 0（或负）= 显式关掉 cap -> GPU 也不 cap -> 与 TPU 一致，**不该拦**
    #     （train_tpu_ashima.yaml 就是显式写 0 的）；
    #   * 缺省 / 5.0 = GPU 会按 5.0 夹，TPU 不夹 -> 是真差异，但 5.0 是 GPU 默认，
    #     从 GPU 拷来的 yaml 都带着它，拦了等于逼所有人加 --allow-unported ->
    #     所以只提示"非默认正值"这种明确是刻意调过的情况。
    "weight_cap_ratio": (
        lambda v: float(v or 0) > 0 and abs(float(v) - 5.0) > 1e-9,
        "loss 权重上限比未移植（TPU 侧 flow.loss_weight 不做 cap）。"
        "写 0 = 显式关掉（两边一致）；GPU 默认 5.0；调成别的正值不会在 TPU 上生效。"),
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
    transformer_path: Path          # 本地 Path，或 http(s) URL str（K2 流式 Range 加载）
    output_dir: Path
    output_name: str
    #: 模型族：anima（默认，与历史逐一等价）/ krea2（单流 MMDiT + FSDP）
    family: str = "anima"
    devices: int = 8
    #: **单卡** token 预算（= yaml 的 navit_token_budget / devices）
    budget: int = 16384
    quantum: int = 1024
    #: anima 路径的定长 cross-attn 文本槽（data.CacheDataset 用它断言缓存 shape）。
    #: K2 路径不用它（K2Packer 走 txt_quantum，文本槽是变长的），那边这个字段是死的。
    txt_len: int = 512
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
    #: krea2 分辨率感知 timestep shift（官方 sampling.py 的训练侧等价，
    #: trainer/model_family.py:399 同公式；family=krea2 时默认开）。
    krea2_res_shift: bool = True
    krea2_shift_min_res: int = 256
    krea2_shift_max_res: int = 1280
    krea2_shift_y1: float = 0.5
    krea2_shift_y2: float = 1.15


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


def _weight_path(v: Any):
    """底模路径：本地文件转 Path；http(s) URL（流式 Range 加载，见
    krea2_jax._read_safetensors_map）必须原样保留——Path() 会把
    "https://" 塌成 "https:/"。"""
    s = str(v or "")
    return s if s.startswith(("http://", "https://")) else Path(s)


def build(d: Dict[str, Any], devices: int = 8, allow_unported: bool = False,
          canvas_hw: Tuple[int, int] = (0, 0)) -> RunConfig:
    """yaml dict -> RunConfig。`canvas_hw` 由数据侧扫描后回填（aux_spectral 用）。"""
    waived = _check_coverage(d, allow_unported)
    family = str(d.get("model_family", "anima") or "anima").lower()
    if family not in ("anima", "krea2"):
        raise ValueError(f"model_family={family!r} 不认识（TPU 后端支持 anima / krea2）")
    _check_navit(d, family)

    # ── 适配器 ───────────────────────────────────────────────────────────────
    targets = _expand_targets(d.get("lora_targets") or [],
                              d.get("lora_exclude_patterns") or [], family)
    acfg = AD.AdapterConfig(
        # 下面四个默认值与 GPU 侧逐一对齐（trainer/config.py 的 DEFAULTS +
        # anima_train.py 的 argparse）。以前 lora_type 默认 "lora"、lora_alpha 默认
        # None(=rank)，于是一份"只写关心的开关"的 yaml（本仓库推荐用法）在两个后端上
        # 解出**不同的适配器结构**与**差 2× 的 scale**，且没有任何提示。
        kind=str(d.get("lora_type", "lokr")).lower(),            # GPU: DEFAULTS=lokr
        variant=str(d.get("lora_variant", "base") or "base").lower(),
        rank=_i(d, "lora_rank", 32),                             # GPU: 32（一致）
        # GPU 缺省是 32.0（不是 "=rank"）。显式写 null 才是"取 rank"（scale=1）。
        # 差别在 rank≠32 时显形：rank=64 + 不写 alpha -> GPU scale=0.5、旧 TPU
        # scale=1.0，等效学习率差 2×。
        alpha=(None if "lora_alpha" in d and d.get("lora_alpha") is None
               else _f(d, "lora_alpha", 32.0)),
        factor=_i(d, "lokr_factor", 8),                          # GPU: 8（一致）
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
    # `optimizer_args` 的**内部**以前是一片没有覆盖检查的盲区：只读这 5 个子键，
    # 其余静默丢弃（实测 `decouple` / `d0` / `weight_decouple` / `amsgrad` /
    # `unknown_knob` 全部无声消失）。这与本文件 docstring"每一个会改变训练数学的
    # yaml 键都必须在表里出现"直接矛盾 —— `optimizer_type` 那条 fail-fast 做得很好，
    # 它的参数却没有。所以这里也做覆盖检查。
    _OA_KNOWN = ("betas", "eps", "weight_decay", "snr_power", "cautious")
    unknown_oa = [k for k in oa if k not in _OA_KNOWN]
    if unknown_oa:
        raise ValueError(
            f"optimizer_args 里有 TPU 后端不认识的子键 {sorted(unknown_oa)}；"
            f"已实现 {list(_OA_KNOWN)}。它们不会被悄悄忽略 —— 若是别的优化器的参数"
            f"（Prodigy 的 d0/decouple、AdamW 变体的 amsgrad 等），那个优化器本身"
            f"就没移植（见 optimizer_type 的 fail-fast）；若确实无用请从 yaml 删掉。")
    betas = oa.get("betas")
    if betas is None:
        betas = [0.9, 0.999]
    # 显式判空而不是 `or 默认值`：`betas: []` 会被 `or` 静默换成默认值。
    if len(betas) != 2:
        raise ValueError(
            f"optimizer_args.betas 必须恰好 2 个（b1, b2），得到 {list(betas)}。"
            f"以前这里直接取前两个下标：3 个会静默丢掉第三个、1 个会报一个看不出"
            f"根因的 IndexError、空列表会被兜底成默认值。")
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
        # GPU 侧 `--timestep-mix-low-prob` / `--timestep-mix-high-prob` 默认都是 0.25
        # （anima_train.py 的 argparse；DEFAULTS 同值）。以前 low 默认 0.5，三峰路由
        # 的低噪份额比 GPU 多一倍 —— 同一份 yaml 两个后端的 t 分布不同。
        mix_low_prob=_f(d, "timestep_mix_low_prob", 0.25),
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
        # 与 GPU 侧默认对齐（trainer/config.py 的 DEFAULTS["min_snr_gamma"]=0.0、
        # argparse `--min-snr-gamma` default=0.0）。以前这里默认 5.0：只写
        # `loss_weighting_scheme: min_snr` 不写 gamma 时 GPU 不加权、TPU 加
        # min(5/snr,1) —— 同一份 yaml 在两个后端上不是同一个实验。
        min_snr_gamma=_f(d, "min_snr_gamma", 0.0),
        detail_inv_t_min=_f(d, "detail_inv_t_min", 1.0),
        detail_inv_t_max=_f(d, "detail_inv_t_max", 5.0),
        immiscible_k=(_i(d, "immiscible_k", 4) if _b(d, "immiscible_enabled") else 1),
    )

    # `min_snr_gamma <= 0` **不是**错配置：PyTorch 侧 `compute_loss_weight` 的
    # min_snr / max_snr_inv 两个分支都在 `min_snr_gamma <= 0` 时 `return ones_like(t)`
    # （trainer/objective.py 的 `compute_loss_weight`，:1112-1121 附近，行号可能漂移，
    # 以符号名为准），即"退化成不加权"，而不是权重恒 0。
    # 以前这里 raise，反而拦掉了 GPU 上完全合法的 `min_snr + 默认 gamma(0.0)` 组合
    # —— 从 GPU 侧拷 yaml 过来必撞。TPU 侧 `flow.loss_weight` 已同口径退化成 ones，
    # 所以这里不再拦，只在真的会静默失效时提示（gamma<0 是笔误的强信号）。
    if fcfg.weighting == "min_snr" and fcfg.min_snr_gamma < 0:
        raise ValueError(
            f"min_snr_gamma={fcfg.min_snr_gamma} < 0 无意义（权重 = min(gamma/snr, 1)）。"
            f"想要不加权就设 0（与 PyTorch 侧默认一致，两边都退化成 ones）；"
            f"想要真的加权请给正值（论文常用 5.0）。")

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
        seed=_i(d, "seed", 42),         # GPU: DEFAULTS/argparse 都是 42
        flow=fcfg, aux=aux, adamw=adamw,
    )

    # navit_token_budget 不给默认值：它是**全局** token 预算，直接决定单卡显存能否
    # 装下（K2 真机账：131072 撞 HLO temporaries 18.15G、98304 撞 executable reserve
    # 11.94G、81920 才跑通）。以前默认 16384 -> 单卡 2048，恰好整除、不报错，于是漏写
    # 这个键的 yaml 会静默跑一个极小预算的实验。
    if d.get("navit_token_budget") in (None, 0, ""):
        raise ValueError(
            "必须显式设 navit_token_budget（**全局** token 预算，8 卡各拿 1/8）："
            "它直接决定显存能否装下，没有一个安全的默认值。K2 真机已验工作点 "
            "81920（= 8 x 10240）；Anima 纯 DP 路线 131072（= 8 x 16384）。"
            "开训前请跑 tests/enum_quantum_advisor.py 看这个数据集的账。")
    total_budget = _i(d, "navit_token_budget", 0)
    if total_budget % devices:
        raise ValueError(
            f"navit_token_budget={total_budget} 不能被 {devices} 卡整除。"
            f"TPU 侧把它当**全局**预算（8 卡各跑一个 pack），单卡拿 1/{devices}；"
            f"不整除就没法保证一步看到的 token 数与 GPU 侧一致。改成 {devices} 的倍数。")

    return RunConfig(
        tcfg=tcfg, adaptive=adaptive,
        data_dir=Path(str(d.get("data_dir", "."))),
        transformer_path=_weight_path(d.get("transformer_path", "")),
        output_dir=Path(str(d.get("output_dir", "./output"))),
        output_name=str(d.get("output_name", "anima-tpu")),
        family=family,
        devices=devices, budget=total_budget // devices,
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
        krea2_res_shift=_b(d, "krea2_res_shift", True),
        krea2_shift_min_res=_i(d, "krea2_shift_min_res", 256),
        krea2_shift_max_res=_i(d, "krea2_shift_max_res", 1280),
        krea2_shift_y1=_f(d, "krea2_shift_y1", 0.5),
        krea2_shift_y2=_f(d, "krea2_shift_y2", 1.15),
    )


# ── 校验 ──────────────────────────────────────────────────────────────────────
def _check_coverage(d: Dict[str, Any], allow: bool) -> List[str]:
    """未移植项的门。返回被显式放行的键。"""
    on, unknown = [], []
    for k, v in d.items():
        if k in _UNPORTED:
            pred, why = _UNPORTED[k]
            try:
                hit = pred(v)
            except (TypeError, ValueError) as e:
                # 谓词大多是 `int(v or 0)` / `float(v or 0)` 这类，值写错类型会抛裸的
                # TypeError/ValueError（`sample_every: [1]` -> "int() argument must be
                # a string..."），不带键名，用户看不出是哪个字段。补上键名。
                raise ValueError(
                    f"yaml 键 {k!r} 的值 {v!r} 类型不对，无法判定该功能是否开启："
                    f"{type(e).__name__}: {e}") from e
            if hit:
                on.append((k, v, why))
            continue
        if k in _HANDLED or k in _CACHE_SIDE or k in _IGNORED:
            continue
        if k.startswith("sample_"):
            # 采样出图的一整族。**三个触发器（sample_every / sample_steps /
            # sample_reference_steps）已各自在 _UNPORTED 里**，上面那个分支先命中，
            # 所以这条豁免只兜住 sample_prompt / sample_steps_count 之类的纯参数。
            continue
        if any(k.startswith(pre) for pre, _ in _FAMILY):
            # 未移植功能的子参数，主开关那条已经管了。
            # **已知盲区**：这是前缀匹配，所以也会吞掉拼写错误（任何 `tread_` /
            # `lwd_` / `dispersive_` 开头的错拼键都不会被报出来）。GPU 侧对此有
            # difflib.get_close_matches 的拼写提示，TPU 这条豁免把那个能力关掉了。
            # 换成显式白名单能恢复，代价是要维护那 30 来个子参数名。
            continue
        unknown.append(k)
    if unknown:
        raise ValueError(
            "yaml 里有 TPU 后端不认识的键：\n  " + ", ".join(sorted(unknown))
            + "\n它们可能是新加的功能。这里不静默忽略 —— 请把它们归到 config.py 的"
              " _HANDLED / _CACHE_SIDE / _IGNORED / _UNPORTED 之一，"
              "顺便确认它是不是真的不影响训练数学。\n"
              "**若它是 GPU 侧真实生效的功能而 TPU 未移植，请补进 _UNPORTED"
              "（带判定谓词与替代路径），不要塞进 _IGNORED 或从 yaml 删掉** —— "
              "那样下一个人就看不出这份 yaml 在两个后端上不是同一个实验了。")
    if on and not allow:
        raise ValueError(
            "以下功能在 yaml 里是开着的，但 TPU 后端没有移植：\n"
            + "\n".join(f"  {k} = {v!r}\n      {why}" for k, v, why in on)
            + "\n\n要么在 yaml 里关掉它们（那样两个后端才是同一个实验），"
              "要么传 --allow-unported 明确接受这份差异。")
    return [k for k, _, _ in on]


def _check_navit(d: Dict[str, Any], family: str = "anima") -> None:
    """NaViT 相关的硬前提。都是"开了会静默出错"的那一类。"""
    if not _b(d, "navit_packing", True):
        raise ValueError("TPU 后端只实现了 NaViT 打包路线（与分桶对照路线），"
                         "navit_packing 必须为 true")
    if family == "anima" and _b(d, "navit_text_trim_padding"):
        raise ValueError(
            "navit_text_trim_padding=true：训练去掉 512-pad 而 eval/采样/ARB 都带 pad，"
            "cross-attn 条件不一致会让 eval_loss 冲高且拟合变差（memory "
            "[[navit-text-trim-train-eval-mismatch]] 的 A/B 实证）。TPU 侧不实现这条路。"
            "（krea2 无此问题：navit 本来就只打包有效 caption token。）")
    strat = str(d.get("navit_pack_strategy", "ffd")).lower()
    if strat != "ffd":
        raise ValueError(f"navit_pack_strategy={strat!r} 未移植；packing.py 只实现了 ffd")
    mp = str(d.get("mixed_precision", "bf16")).lower()
    if mp not in ("bf16", "bfloat16"):
        raise ValueError(f"mixed_precision={mp!r}：TPU 后端只走 bf16 前向 + fp32 master")
    if (family == "krea2" and _b(d, "krea2_res_shift", True)
            and abs(_f(d, "schedule_shift", 1.0) - 1.0) > 1e-6):
        # 与 docs/krea2-family.md 同一建议：res_shift 已在任何 timestep mode 之后
        # 按分辨率施加官方 shift，再叠 schedule_shift 就是双重偏移。
        raise ValueError(
            f"krea2_res_shift 开着时 schedule_shift 必须保持 1.0（得到 "
            f"{_f(d, 'schedule_shift', 1.0)}）—— 两者是同一个 shift 施加两次"
            f"（docs/krea2-family.md 的口径）。要手动调度就关 krea2_res_shift。")


def _expand_targets(names, excludes, family: str = "anima") -> Tuple[str, ...]:
    """yaml 的裸 target 名（`q_proj`）展开成内部全名（`self_attn.q_proj` 等）。

    PyTorch 侧 `lora_targets` 是按模块名子串匹配的，所以 `q_proj` 会同时命中
    self_attn 与 cross_attn；这里照此展开，再套 `lora_exclude_patterns`。
    krea2：全名表换成单流 MMDiT 的 264 个 Linear（krea2_jax.lora_target_shapes，
    缺省 = 官方/musubi 推荐的"DiT 全部 Linear"）。
    """
    import re
    if family == "krea2":
        all_t = tuple(sorted(K2.lora_target_shapes(K2.Krea2Config())))
    else:
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
    if family == "krea2":
        def excluded(t):
            # 与 Anima 同规则：展开名 fullmatch 或 target 名 search，任一命中即排除
            cnt = K2.lora_target_shapes(K2.Krea2Config())[t][0]
            ex = K2.expand_name(t, 0, cnt)
            return any(p.fullmatch(ex) or p.search(t) for p in pats)
        picked = [t for t in picked if not excluded(t)]
    else:
        picked = [t for t in picked if not any(
            p.fullmatch(f"blocks.0.{t}") or p.search(t) for p in pats)]
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
        f"模型族 {rc.family}"
        + ("（单流 MMDiT + FSDP 权重分片）" if rc.family == "krea2" else ""),
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
        f" 分层={f.stratified} 范围=[{f.t_min},{f.t_max}]"
        + (f" | krea2_res_shift mu({rc.krea2_shift_min_res}px={rc.krea2_shift_y1},"
           f"{rc.krea2_shift_max_res}px={rc.krea2_shift_y2})"
           if rc.family == "krea2" and rc.krea2_res_shift else ""),
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
