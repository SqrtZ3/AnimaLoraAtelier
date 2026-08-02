"""权重加载 / 路径规整 / 训练状态保存恢复。

包含两类完全独立的功能：

1. **权重加载助手**（被 `trainer/models.py` 使用）：
   - `_strip_prefixes` / `_pick_best_prefix_remap` —— 自动剥离 model./module./diffusion_model./...
     等常见前缀，命中最多 model_keys 的方案胜出
   - `_load_safetensors_state_dict` —— safetensors → dict
   - `_load_weights_best_effort` —— 加载 + 覆盖率检查 + 关键层缺失报错（避免"采样全噪点"）
   - `resolve_path_best_effort` —— 把相对路径按多个 base 尝试解析到真实存在的路径
   - `normalize_resume_paths` —— resume_lora 拿到 .pt 文件时自动找匹配的 .safetensors

2. **训练状态序列化**：
   - `save_training_state` —— 保存 LoRA + optimizer + scheduler + rng + 监控面板状态
   - `load_training_state` —— 反序列化并把状态填回 injector / optimizer
"""

from __future__ import annotations

import logging
import random
import re
import gc
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


# ============================================================================
# Weight loading helpers
# ============================================================================

def _strip_prefixes(key: str, prefixes: list[str]) -> str:
    """反复剥离前缀（支持 module.model. 这种复合前缀）"""
    if not prefixes:
        return key
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if key.startswith(p):
                key = key[len(p):]
                changed = True
    return key


def _pick_best_prefix_remap(sd_keys: list[str], model_keys: set[str]) -> tuple[list[str], int]:
    """从常见前缀组合里选择"命中最多 model_keys"的 remap 方案。

    返回 (prefixes, matched_count)。
    """
    candidates: list[tuple[str, list[str]]] = [
        ("none", []),
        ("net.", ["net."]),
        ("model.", ["model."]),
        ("module.", ["module."]),
        ("module.+model.", ["module.", "model."]),
        ("module.model.", ["module.model."]),
        ("diffusion_model.", ["diffusion_model."]),
        ("model.diffusion_model.", ["model.diffusion_model."]),
        ("transformer.", ["transformer."]),
        ("vae.", ["vae."]),
        ("first_stage_model.", ["first_stage_model."]),
        ("net.+model.", ["net.", "model."]),
        ("net.model.", ["net.model."]),
    ]

    best_prefixes: list[str] = []
    best_matched = -1
    for _name, prefixes in candidates:
        matched = 0
        for k in sd_keys:
            kk = _strip_prefixes(k, prefixes)
            if kk in model_keys:
                matched += 1
        if matched > best_matched:
            best_matched = matched
            best_prefixes = prefixes
    return best_prefixes, best_matched


def _load_safetensors_state_dict(path: Path) -> dict:
    from safetensors import safe_open

    sd = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            sd[k] = f.get_tensor(k)
    return sd




class _ShapeProxy:
    """Lightweight stand-in for a tensor, exposing only .shape.

    Used by ``_get_safetensors_shapes()`` so that
    ``infer_config_from_state_dict()`` can read tensor shapes without
    loading the actual weight data (~1000x lighter for 12B models).
    """
    __slots__ = ("shape",)

    def __init__(self, shape):
        self.shape = shape


def _get_safetensors_shapes(path: Path) -> dict:
    """Read all key -> shape pairs from a safetensors file **without** loading tensor data.

    Returns a ``dict[str, _ShapeProxy]``.  Only metadata is read -- no weight
    bytes are copied to RAM.  This is the right primitive for config inference
    (``infer_config_from_state_dict``) where only ``.shape`` and ``key in dict``
    are needed.
    """
    from safetensors import safe_open

    shapes = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            shapes[k] = _ShapeProxy(tuple(f.get_slice(k).get_shape()))
    return shapes


def _load_safetensors_into_model(
    model: torch.nn.Module,
    path: Path,
    prefixes: list[str] | None = None,
    label: str = "Model",
    skip_buffer_patterns: tuple[str, ...] | None = None,
    strict_missing: bool = False,
) -> dict:
    """Stream-load safetensors weights directly into *model* parameters, one tensor at a time.

    Avoids creating a full state-dict copy in RAM.  For a 12B model (bf16 ~24 GB)
    this cuts peak host-RAM from ~72-96 GB (3-4 simultaneous copies in the old
    ``_load_safetensors_state_dict`` + ``_filter_recomputable_buffers`` +
    ``_load_weights_best_effort`` pipeline) down to ~24 GB (model itself + one
    transient tensor).

    Parameters
    ----------
    model : nn.Module
        Target model (typically on CPU).  Weights are loaded via
        ``param.data.copy_(tensor)`` in-place, so the model's own storage
        is the only persistent allocation.
    path : Path
        Path to ``.safetensors`` file.
    prefixes : list[str] | None
        Prefix list to strip (e.g. ``["model."]``).  If ``None``, the best
        prefix is auto-detected via ``_pick_best_prefix_remap``.
    label : str
        Label for log messages (e.g. ``"Transformer"``, ``"VAE"``).
    skip_buffer_patterns : tuple[str, ...] | None
        If a model key contains any of these substrings, the tensor is
        shape-checked; a mismatch means the buffer is recomputable (e.g.
        RoPE position embeddings) and is **skipped** -- the model's own
        ``reset_parameters()`` will have set it correctly.
    strict_missing : bool
        If ``True``, **any** missing key raises ``RuntimeError`` (Krea2
        semantics).  If ``False``, only critical layers (``x_embedder.``,
        ``blocks.``, ``final_layer.``) or coverage < 60% raise (Anima / VAE).

    Returns
    -------
    dict  with keys ``remap``, ``coverage``, ``missing``, ``unexpected``.
    """
    from safetensors import safe_open

    model_sd = model.state_dict()
    model_keys = set(model_sd.keys())

    with safe_open(path, framework="pt", device="cpu") as f:
        sd_keys = list(f.keys())

        # Auto-detect best prefix if not provided
        if prefixes is None:
            prefixes, _ = _pick_best_prefix_remap(sd_keys, model_keys)
        remap_name = "+".join(prefixes) if prefixes else "none"

        # Build mapping: model_key -> safetensors_key
        key_map: dict[str, str] = {}
        for sk in sd_keys:
            mk = _strip_prefixes(sk, prefixes)
            if mk in model_keys:
                key_map[mk] = sk

        # Stream-load: one tensor at a time, copy into model param, then free
        loaded = 0
        skipped_buffers: list[tuple[str, tuple, tuple]] = []
        for mk, sk in key_map.items():
            # Skip recomputable derivative buffers whose shape doesn't match
            if skip_buffer_patterns and any(pat in mk for pat in skip_buffer_patterns):
                param_shape = tuple(model_sd[mk].shape)
                ckpt_shape = tuple(f.get_slice(sk).get_shape())
                if param_shape != ckpt_shape:
                    skipped_buffers.append((mk, ckpt_shape, param_shape))
                    continue
            tensor = f.get_tensor(sk)
            model_sd[mk].data.copy_(tensor)
            del tensor
            loaded += 1

    if skipped_buffers:
        for name, ck_shape, md_shape in skipped_buffers[:5]:
            logger.info(
                "Drop recomputable buffer from ckpt: %s (ckpt=%s, model=%s) - "
                "模型 reset_parameters() 会重算",
                name, ck_shape, md_shape,
            )
        if len(skipped_buffers) > 5:
            logger.info("  ... 共 %d 个 recomputable buffer 被跳过", len(skipped_buffers))

    # Coverage stats (mirror _load_weights_best_effort log format)
    matched_keys = set(key_map.keys())
    missing = sorted(model_keys - matched_keys)
    unexpected = sorted(
        _strip_prefixes(k, prefixes) for k in sd_keys
        if _strip_prefixes(k, prefixes) not in model_keys
    )
    coverage = loaded / max(1, len(model_keys))

    logger.info(
        f"{label} 权重加载: remap={remap_name}, 匹配 {loaded}/{len(model_keys)} "
        f"({coverage:.1%}), missing={len(missing)}, unexpected={len(unexpected)}"
    )

    # Fail-fast checks
    if strict_missing and missing:
        raise RuntimeError(
            f"{label} 权重缺失 {len(missing)} 个 key（前 5 个: {missing[:5]}）。"
        )
    critical_prefixes = ("x_embedder.", "blocks.", "final_layer.")
    critical_missing = [k for k in missing if k.startswith(critical_prefixes)]
    if coverage < 0.60 or len(critical_missing) > 0:
        preview_missing = ", ".join(critical_missing[:8])
        raise RuntimeError(
            f"{label} 权重看起来没有正确加载（remap={remap_name}, coverage={coverage:.1%}）。"
            f"关键参数缺失: {preview_missing or 'N/A'}。\n"
            f"这通常表示你选错了 .safetensors（不是完整 transformer/vae 权重），或 "
            f"checkpoint key 前缀不匹配。"
        )

    return {
        "remap": remap_name,
        "coverage": coverage,
        "missing": missing,
        "unexpected": unexpected,
    }


def resolve_path_best_effort(path_str: str, bases: list[Path]) -> str:
    """将相对路径按多个 base 尝试解析到一个真实存在的路径。

    主要用于：无论从 repo 根目录还是 AnimaLoraToolkit 目录启动，都能找到 models/* 文件。
    """
    if not path_str:
        return path_str

    p = Path(path_str)
    if p.is_absolute():
        return str(p)

    # 先按原样（相对 cwd）试一下
    if p.exists():
        return str(p)

    # 逐 base 拼接尝试
    for b in bases:
        if not b:
            continue
        try:
            cand = (Path(b) / p).resolve()
        except Exception:
            cand = Path(b) / p
        if cand.exists():
            return str(cand)

    # 常见：配置写了 AnimaLoraToolkit/xxx，但启动目录已经在 AnimaLoraToolkit 下
    parts = p.parts
    if parts and parts[0].lower() in ("animaloratoolkit", "anima_trainer", "anima-trainer"):
        p2 = Path(*parts[1:])
        if p2.exists():
            return str(p2)
        for b in bases:
            if not b:
                continue
            cand = Path(b) / p2
            if cand.exists():
                return str(cand)

    return path_str


def normalize_resume_paths(args, output_dir: Path):
    """Validate resume paths and recover common resume_lora/resume_state mixups."""
    resume_lora = str(getattr(args, "resume_lora", "") or "").strip()
    resume_state = str(getattr(args, "resume_state", "") or "").strip()

    if resume_lora:
        lora_path = Path(resume_lora)
        if lora_path.suffix.lower() == ".pt":
            if resume_state:
                logger.warning(
                    "resume_lora points to a .pt training state, but resume_state is already set; "
                    "ignoring resume_lora: %s",
                    resume_lora,
                )
                args.resume_lora = ""
            else:
                step_match = re.search(r"step(\d+)", lora_path.stem)
                companion = None
                if step_match:
                    step = step_match.group(1)
                    search_dir = lora_path.parent if str(lora_path.parent) != "." else output_dir
                    candidates = sorted(search_dir.glob(f"*_step{step}.safetensors"))
                    if candidates:
                        companion = candidates[0]

                if companion and companion.exists():
                    args.resume_lora = str(companion)
                    logger.warning(
                        "resume_lora was a .pt training state; using matching LoRA weights instead: %s",
                        companion,
                    )
                else:
                    args.resume_lora = ""
                    args.resume_state = resume_lora
                    logger.warning(
                        "resume_lora was a .pt training state and no matching .safetensors was found; "
                        "using resume_state instead: %s",
                        resume_lora,
                    )
        elif not lora_path.exists():
            logger.warning("resume_lora path does not exist; ignoring it: %s", resume_lora)
            args.resume_lora = ""

    resume_state = str(getattr(args, "resume_state", "") or "").strip()
    if resume_state and not Path(resume_state).exists():
        logger.warning("resume_state path does not exist; ignoring it: %s", resume_state)
        args.resume_state = ""


def _load_weights_best_effort(model: torch.nn.Module, sd: dict, label: str) -> dict:
    """更健壮的权重加载：

    - 自动尝试剥离常见前缀（model./module./...）
    - 打印匹配率、missing/unexpected
    - 关键模块未加载时直接报错（避免"采样全噪点"还继续训练）
    """
    model_keys = set(model.state_dict().keys())
    sd_keys = list(sd.keys())
    prefixes, matched = _pick_best_prefix_remap(sd_keys, model_keys)
    remapped = {_strip_prefixes(k, prefixes): v for k, v in sd.items()}

    incompatible = model.load_state_dict(remapped, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []) or [])
    unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])

    matched_after = len(set(remapped.keys()) & model_keys)
    coverage = matched_after / max(1, len(model_keys))
    remap_name = "+".join(prefixes) if prefixes else "none"

    logger.info(
        f"{label} 权重加载: remap={remap_name}, 匹配 {matched_after}/{len(model_keys)} ({coverage:.1%}), "
        f"missing={len(missing)}, unexpected={len(unexpected)}"
    )

    # 关键层缺失会直接导致输出接近 0，采样就是纯噪点
    critical_prefixes = ("x_embedder.", "blocks.", "final_layer.")
    critical_missing = [k for k in missing if k.startswith(critical_prefixes)]
    if coverage < 0.60 or len(critical_missing) > 0:
        preview_missing = ", ".join(critical_missing[:8])
        raise RuntimeError(
            f"{label} 权重看起来没有正确加载（remap={remap_name}, coverage={coverage:.1%}）。"
            f"关键参数缺失: {preview_missing or 'N/A'}。\n"
            f"这通常表示你选错了 .safetensors（不是完整 transformer/vae 权重），或 checkpoint key 前缀不匹配。"
        )
    return {
        "remap": remap_name,
        "coverage": coverage,
        "missing": missing,
        "unexpected": unexpected,
    }


# ============================================================================
# Training state save / load
# ============================================================================

def dataloader_fingerprint(dataloader, grad_accum=1):
    """数据/分批配置的指纹，用于判断 resume 时 batch 序列能否精确对齐。

    三个 batch sampler 的洗牌都只用 `random.Random(self.seed + self.epoch)`
    （data.py 的 BucketBatchSampler / FitTokenBatchSampler / NavitPackBatchSampler），
    不依赖全局 RNG —— 只要数据集与这些字段没变，同一个 epoch 号产出的 batch 序列
    就逐 batch 可复现，"跳过前 N 个 batch"才是精确的。任一字段变了（换数据集、
    改 batch_size/seed/drop_last…）指纹就不匹配，调用方放弃 skip。
    """
    sampler = getattr(dataloader, "batch_sampler", None)
    dataset = getattr(dataloader, "dataset", None)
    try:
        dataset_len = len(dataset) if dataset is not None else None
    except TypeError:
        dataset_len = None
    fp = {
        "sampler_class": type(sampler).__name__ if sampler is not None else None,
        "dataset_len": dataset_len,
        "grad_accum": int(grad_accum or 1),
    }
    for key in ("batch_size", "seed", "shuffle", "drop_last",
                "effective_batch_size", "reference_batch_size",
                "max_tokens_per_batch", "token_budget"):
        if sampler is not None and hasattr(sampler, key):
            value = getattr(sampler, key)
            if isinstance(value, bool):
                fp[key] = bool(value)
            elif isinstance(value, (int, float)):
                fp[key] = value
    return fp


def save_training_state(path, injector, optimizer, epoch, global_step,
                        loss_history=None, rng_state=None, monitor_state=None,
                        scheduler=None, samples_seen=None, reference_state=None,
                        epoch_position=None):
    """保存完整训练状态，支持断点续训

    `epoch_position`（可选 dict）记录 **epoch 内**的消费位置，让 resume 能从中断处
    接续而不是把该 epoch 从头重跑一遍。字段见 `anima_train.py` 的 `_epoch_position()`：
      - batch_in_epoch: 最后一次完成 optimizer step 时已消费的 batch 数（= batch_idx+1）
      - accum_pending / accum_offset_at_epoch_start: sample-window accumulation 的切分状态
      - fingerprint: 数据/分批配置指纹，resume 时不匹配就放弃 skip（fail-safe 回旧行为）
    缺省 None 时行为与旧版存盘完全一致（老 state 文件也照常能读，只是没有 skip 信息）。
    """
    state = {
        "lora_state_dict": injector.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "samples_seen": samples_seen,
        "reference_state": reference_state,
        "epoch_position": epoch_position,
        "loss_history": loss_history or [],
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "random": random.getstate(),
        },
        "monitor_state": monitor_state,  # 保存监控面板数据（用于恢复 loss 曲线）
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(state, path)
    _pos = ""
    if isinstance(epoch_position, dict) and epoch_position.get("batch_in_epoch"):
        _pos = f", batch_in_epoch={epoch_position['batch_in_epoch']}"
    logger.info(f"训练状态已保存: {path} (epoch={epoch}, step={global_step}{_pos})")


def load_training_state(path, injector, optimizer, scheduler=None):
    """加载训练状态，返回 (epoch, global_step, loss_history, monitor_state, samples_seen,
    reference_state, epoch_position)。

    ★ 旧实现这里有一份独立的 "拷贝 lora_w1/w2_a/w2_b" 逻辑，与 `LoRAInjector.load()`
    几乎完全重复。任何对 LoRA 存盘格式的修改（如 T-LoRA q/p_layer 命名、DoRA scale 维度）
    都要在两处同步改。这里改为：写一个临时 safetensors-like 接口给 injector.load_state_dict
    复用注入器自己的 loader，单一来源避免漂移。
    """
    logger.info(f"加载训练状态: {path}")
    if str(path).endswith(".safetensors"):
        raise ValueError(
            f"resume_state 指向了 safetensors 文件: {path}\n"
            "这是 LoRA 权重成品，不是训练状态。两者的用法：\n"
            "  - 只从权重继续（优化器从零）: resume_lora 填该 .safetensors，resume_state 留空\n"
            "  - 完整断点续训（含优化器/step/RNG）: resume_state 填 output 目录下的 "
            "training_state_step{N}.pt（save_state_every 产出）或 "
            "training_state_epoch{E}_step{N}.pt（save_state_every_epochs 产出），"
            "其中已含 LoRA 权重"
        )
    state = torch.load(path, map_location="cpu", weights_only=False)

    # 委托给 injector.load_state_dict_from_mapping —— 这是新加的 in-memory 加载入口，
    # 与 .safetensors 文件加载共享同一份 key 解析逻辑。
    lora_sd = state["lora_state_dict"]
    if hasattr(injector, "load_state_dict_from_mapping"):
        injector.load_state_dict_from_mapping(lora_sd)
    else:
        # 极旧的 injector 版本兜底（不应该走到这里）
        for name, lora in injector.injected.items():
            base = "lora_unet_" + name.replace(".", "_")
            if injector.use_lokr:
                w1_key = f"{base}.lokr_w1"
                w2a_key = f"{base}.lokr_w2_a"
                w2b_key = f"{base}.lokr_w2_b"
                dora_key = f"{base}.dora_scale"
                if w1_key in lora_sd and w2a_key in lora_sd and w2b_key in lora_sd:
                    lora.adapter.lokr_w1.data.copy_(lora_sd[w1_key])
                    lora.adapter.lokr_w2_a.data.copy_(lora_sd[w2a_key])
                    lora.adapter.lokr_w2_b.data.copy_(lora_sd[w2b_key])
                    if getattr(lora, "use_dora", False) and dora_key in lora_sd:
                        dora_scale = lora_sd[dora_key].reshape(-1)
                        lora.dora_scale.data.copy_(dora_scale.to(device=lora.dora_scale.device, dtype=lora.dora_scale.dtype))
            else:
                down_key = f"{base}.lora_down.weight"
                up_key = f"{base}.lora_up.weight"
                if down_key in lora_sd and up_key in lora_sd:
                    lora.adapter.lora_down.weight.data.copy_(lora_sd[down_key])
                    lora.adapter.lora_up.weight.data.copy_(lora_sd[up_key])

    # 加载优化器状态
    optimizer.load_state_dict(state["optimizer_state_dict"])

    # ── 优化器状态 dtype 复原（转回"保存时的 dtype"）──────────────────────
    # PyTorch 的 Optimizer.load_state_dict 会把每个"逐参数"浮点状态张量强制
    # 转成 *该参数* 的 dtype（_process_value_according_to_param_policy）。本仓库
    # 的 LoRA 参数是 bf16，于是 MuonSF/soap 等自定义优化器保存时本为 fp32 的
    # master 状态（momentum_buffer / z / y / exp_avg …）恢复时被静默降成 bf16
    # —— 触发 MuonSF lerp_ dtype 崩溃、废掉 fp32-master 防 ulp 冻结的 fix。
    # ★ 但不能无差别转 fp32（旧实现的 bug）：torch 原生 AdamW 的状态本来就是
    # 按参数 dtype（bf16）创建的，强转 fp32 会让 resume 后第一步
    # _foreach_lerp_(exp_avg_fp32, grad_bf16) 直接 RuntimeError。
    # 正确的不变式是"resume 后 dtype == 保存时 dtype"：按磁盘里每个状态张量的
    # 原始 dtype 逐个还原（fp32 master 回 fp32，bf16 态保持 bf16，各优化器自动正确）。
    _saved_opt = state["optimizer_state_dict"]
    _saved_state = _saved_opt.get("state", {})
    # torch load_state_dict 的映射语义：保存的 param id 与当前 params 按 group
    # 顺序 zip 对应，这里复用同一约定建 saved_id → 当前 param 的映射。
    _saved_ids = [pid for g in _saved_opt.get("param_groups", [])
                  for pid in g.get("params", [])]
    _cur_params = [p for g in optimizer.param_groups for p in g["params"]]
    _restored = 0
    for _pid, _p in zip(_saved_ids, _cur_params):
        _saved_st = _saved_state.get(_pid)
        _cur_st = optimizer.state.get(_p)
        if not isinstance(_saved_st, dict) or not isinstance(_cur_st, dict):
            continue
        for _k, _v in _cur_st.items():
            _sv = _saved_st.get(_k)
            if (isinstance(_v, torch.Tensor) and isinstance(_sv, torch.Tensor)
                    and _v.is_floating_point() and _sv.is_floating_point()
                    and _v.dtype != _sv.dtype):
                # ★ 必须从**磁盘上的原张量**搬值，不能 _v.to(_sv.dtype)：
                #   torch 已经把 fp32 master 降成 bf16 了，再转回 fp32 只恢复容器
                #   类型、低位一去不返（实测复原后 master 与 bf16 参数逐元素相等，
                #   等于 master 被重置成参数值，fp32-master 防 ulp 冻结形同虚设）。
                #   _sv 是 torch.load 出来的原始 fp32 张量，直接搬过来即可。
                _cur_st[_k] = _sv.detach().to(device=_v.device)
                _restored += 1
    if _restored:
        logger.info(f"优化器状态 dtype 复原: {_restored} 个浮点状态张量已转回保存时的 dtype"
                    f"（fp32 master 不变式；torch 原生优化器的 bf16 态不受影响）")

    # 加载调度器状态
    if scheduler is not None and "scheduler_state_dict" in state:
        try:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        except Exception as e:
            logger.warning(f"调度器状态恢复失败（将从头开始）: {e}")

    # 恢复随机数状态
    if "rng_state" in state:
        rng = state["rng_state"]
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(rng["cuda"])
        if rng.get("random") is not None:
            random.setstate(rng["random"])

    epoch = state.get("epoch", 0)
    global_step = state.get("global_step", 0)
    loss_history = state.get("loss_history", [])
    monitor_state = state.get("monitor_state", None)  # 恢复监控数据

    logger.info(f"训练状态已恢复: epoch={epoch}, step={global_step}")
    samples_seen = state.get("samples_seen")
    reference_state = state.get("reference_state")
    # 旧 state 文件没有这个键 → None，调用方据此回退到"从 epoch 头开始"的旧行为。
    epoch_position = state.get("epoch_position")
    if not isinstance(epoch_position, dict):
        epoch_position = None

    # Free the CPU-side checkpoint copy and any intermediate GPU tensors
    # (e.g. bf16 casts from load_state_dict) before training begins.
    del state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (epoch, global_step, loss_history, monitor_state, samples_seen,
            reference_state, epoch_position)
