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

def save_training_state(path, injector, optimizer, epoch, global_step,
                        loss_history=None, rng_state=None, monitor_state=None, scheduler=None):
    """保存完整训练状态，支持断点续训"""
    state = {
        "lora_state_dict": injector.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
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
    logger.info(f"训练状态已保存: {path} (epoch={epoch}, step={global_step})")


def load_training_state(path, injector, optimizer, scheduler=None):
    """加载训练状态，返回 (epoch, global_step, loss_history, monitor_state)。

    ★ 旧实现这里有一份独立的 "拷贝 lora_w1/w2_a/w2_b" 逻辑，与 `LoRAInjector.load()`
    几乎完全重复。任何对 LoRA 存盘格式的修改（如 T-LoRA q/p_layer 命名、DoRA scale 维度）
    都要在两处同步改。这里改为：写一个临时 safetensors-like 接口给 injector.load_state_dict
    复用注入器自己的 loader，单一来源避免漂移。
    """
    logger.info(f"加载训练状态: {path}")
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
    return epoch, global_step, loss_history, monitor_state
