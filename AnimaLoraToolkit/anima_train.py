#!/usr/bin/env python
"""
Anima LoRA Trainer v2 - 支持 LyCORIS + 训练时推理
基于 trainerV1.01 重构，轻量单文件

特性：
- 标准 LoRA 和 LyCORIS LoKr 双模式
- 训练时推理出图
- Flow Matching 训练
- ARB 分桶
- 依赖自动检测与安装
- Rich 进度条 + ASCII Loss 曲线
- 梯度检查点支持
- Caption 预处理 (shuffle/keep_tokens)
"""

import argparse
import logging
import math
import os
import random
import re
import subprocess
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# 尝试添加当前目录到路径，确保能找到 utils
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

# 强制导入优化器工具，不再提供静默回退（Fail fast）
try:
    from utils.optimizer_utils import create_optimizer, get_optimizer_info
except ImportError as e:
    # NOTE: logger 此时尚未定义，用 print 输出
    print(f"[FATAL] 无法导入高级优化器逻辑: {e}")
    print(f"[FATAL] 当前 Python 路径 (sys.path): {sys.path}")
    print(f"[FATAL] 脚本所在目录 (script_dir): {script_dir}")
    raise ImportError(
        "关键模块 utils.optimizer_utils 加载失败。\n"
        "1. 请确认 AnimaLoraToolkit/utils/optimizer_utils.py 文件存在\n"
        "2. 请确认已安装依赖: pip install prodigy-plus-schedule-free"
    ) from e

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# 依赖检测
# ============================================================================

def ensure_dependencies(auto_install=False):
    """检测并可选自动安装缺失依赖"""
    required = {
        "numpy": "numpy",
        "PIL": "Pillow",
        "safetensors": "safetensors",
        "transformers": "transformers",
        "einops": "einops",
        "torchvision": "torchvision",
        "yaml": "pyyaml",
    }
    missing = []
    for module_name, pip_name in required.items():
        try:
            __import__(module_name)
        except Exception:
            missing.append(pip_name)
    if not missing:
        return
    missing_list = ", ".join(sorted(set(missing)))
    print(f"Missing dependencies: {missing_list}")
    if not auto_install:
        print(f"Install them with:\n  {sys.executable} -m pip install {missing_list}")
        raise SystemExit(1)
    cmd = [sys.executable, "-m", "pip", "install", *sorted(set(missing))]
    print("Installing missing dependencies...")
    try:
        subprocess.run(cmd, check=False)
    except Exception as exc:
        print(f"Auto-install failed: {exc}")
        raise SystemExit(1)
    # Re-check after install
    still_missing = []
    for module_name, pip_name in required.items():
        try:
            __import__(module_name)
        except Exception:
            still_missing.append(pip_name)
    if still_missing:
        still_list = ", ".join(sorted(set(still_missing)))
        print(f"Still missing: {still_list}")
        raise SystemExit(1)


# ============================================================================
# YAML 配置加载
# ============================================================================

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


def apply_yaml_config(args, config):
    """将 YAML 配置应用到 args，命令行参数优先"""
    # 字段映射: YAML key -> args attribute
    mapping = {
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
        "repeats": "repeats",
        "shuffle_caption": "shuffle_caption",
        "keep_tokens": "keep_tokens",
        "flip_augment": "flip_augment",
        "tag_dropout": "tag_dropout",
        "prefer_json": "prefer_json",
        "cache_latents": "cache_latents",
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
        # LoRA+ for LoKr
        "loraplus_lr_ratio": "loraplus_lr_ratio",
        # 频率均衡 tag dropout：根据 tag 在数据集中的出现频率自适应地增加 dropout
        "freq_balanced_dropout_strength": "freq_balanced_dropout_strength",
    }

    deprecated_v5_keys = {
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

    ignored = sorted(k for k in deprecated_v5_keys if k in config and _is_active_deprecated_value(config.get(k)))
    if ignored:
        logger.warning(
            "Ignoring deprecated v5 training keys: %s. They no longer affect training.",
            ", ".join(ignored),
        )

    # 需要特殊处理的默认值（用于判断命令行是否显式设置）
    defaults = {
        "transformer": "",
        "vae": "",
        "qwen": "",
        "t5_tokenizer": "",
        "data_dir": "",
        "reg_data_dir": "",
        "reg_repeats": 1,
        "reg_caption": "",
        "resolution": 1024,
        "repeats": 1,
        "shuffle_caption": False,
        "keep_tokens": 0,
        "flip_augment": False,
        "tag_dropout": 0.0,
        "prefer_json": True,
        "cache_latents": False,
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
        "loraplus_lr_ratio": 1.0,
        "freq_balanced_dropout_strength": 0.0,
    }

    for yaml_key, arg_attr in mapping.items():
        if yaml_key not in config:
            continue
        yaml_value = config[yaml_key]
        if yaml_value is None:
            continue

        # 检查命令行是否显式设置了该参数（与默认值不同）
        current_value = getattr(args, arg_attr, None)
        default_value = defaults.get(arg_attr)

        # 如果当前值等于默认值，或者属性不存在（current_value 为 None），则使用 YAML 配置
        # 特殊处理：列表类型的默认值用 [] 表示，但 argparse 未定义时返回 None
        if current_value == default_value or current_value is None:
            setattr(args, arg_attr, yaml_value)

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


# ============================================================================
# 进度和 Loss 曲线可视化
# ============================================================================

def init_progress(show_progress, total_steps):
    """初始化 Rich 进度条"""
    if not show_progress:
        return None, None, None
    try:
        from rich.progress import (
            BarColumn, MofNCompleteColumn, Progress, TextColumn,
            TimeElapsedColumn, TimeRemainingColumn,
        )
        progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("loss={task.fields[loss]:.4f}"),
            TextColumn("lr={task.fields[lr]:.2e}"),
            TextColumn("speed={task.fields[speed]:.2f} it/s"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            refresh_per_second=10,
        )
        task = progress.add_task("train", total=total_steps, loss=0.0, lr=0.0, speed=0.0)
        return progress, task, "rich"
    except Exception:
        return "plain", None, None


def render_loss_curve(losses, width=60, height=10):
    """渲染 ASCII Loss 曲线"""
    if not losses:
        return ""
    if width < 5:
        width = 5
    values = losses
    if len(values) > width:
        step = len(values) / width
        buckets = []
        for i in range(width):
            start = int(i * step)
            end = int((i + 1) * step)
            end = max(end, start + 1)
            chunk = values[start:end]
            buckets.append(sum(chunk) / len(chunk))
        values = buckets
    min_v = min(values)
    max_v = max(values)
    if max_v == min_v:
        max_v = min_v + 1e-8
    grid = [[" " for _ in range(len(values))] for _ in range(height)]
    for i, v in enumerate(values):
        y = int((v - min_v) / (max_v - min_v) * (height - 1))
        y = height - 1 - y
        grid[y][i] = "*"
    lines = ["".join(row) for row in grid]
    lines.append(f"min={min_v:.4f} max={max_v:.4f}")
    return "\n".join(lines)


def render_curve_panel(losses, width=60, height=10):
    """渲染 Rich Panel 包装的 Loss 曲线"""
    try:
        from rich.panel import Panel
        from rich.text import Text
    except Exception:
        return None
    chart = render_loss_curve(losses, width=width, height=height)
    return Panel(Text(chart), title="Loss curve (recent)", expand=False)


# ============================================================================
# 模型加载工具
# ============================================================================

def find_diffusion_pipe_root():
    """查找 diffusion-pipe 模型代码路径"""
    candidates = [
        Path(__file__).parent / "diffusion_models",
        Path(__file__).parent / "models",
        Path(os.environ.get("DIFFUSION_PIPE_ROOT", "")) if os.environ.get("DIFFUSION_PIPE_ROOT") else None,
    ]
    for candidate in candidates:
        if candidate and (candidate / "anima_modeling.py").exists():
            return candidate
        if candidate and (candidate / "models" / "anima_modeling.py").exists():
            return candidate / "models"
    raise RuntimeError("找不到 anima_modeling.py，请设置 DIFFUSION_PIPE_ROOT 或放置模型代码")


def load_module_from_path(module_name, file_path):
    """动态加载 Python 模块"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _strip_prefixes(key: str, prefixes: list[str]) -> str:
    """反复剥离前缀（支持 module.model. 这种复合前缀）"""
    if not prefixes:
        return key
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if key.startswith(p):
                key = key[len(p) :]
                changed = True
    return key


def _pick_best_prefix_remap(sd_keys: list[str], model_keys: set[str]) -> tuple[list[str], int]:
    """
    从常见前缀组合里选择“命中最多 model_keys”的 remap 方案。
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
    """
    将相对路径按多个 base 尝试解析到一个真实存在的路径。
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
    """
    更健壮的权重加载：
    - 自动尝试剥离常见前缀（model./module./...）
    - 打印匹配率、missing/unexpected
    - 关键模块未加载时直接报错（避免“采样全噪点”还继续训练）
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


def ensure_models_namespace(repo_root):
    """确保 models 命名空间可用"""
    repo_root = Path(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if str(repo_root.parent) not in sys.path:
        sys.path.insert(0, str(repo_root.parent))


def load_anima_model(transformer_path, device, dtype, repo_root):
    """加载 Anima transformer 模型"""
    from safetensors import safe_open

    ensure_models_namespace(repo_root)

    # 加载模型类
    cosmos_modeling = load_module_from_path(
        "cosmos_predict2_modeling",
        repo_root / "cosmos_predict2_modeling.py",
    )
    anima_modeling = load_module_from_path(
        "anima_modeling",
        repo_root / "anima_modeling.py",
    )
    Anima = anima_modeling.Anima

    # 从 checkpoint 推断配置
    with safe_open(transformer_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.endswith("x_embedder.proj.1.weight"):
                w = f.get_tensor(k)
                break

    in_channels = (w.shape[1] // 4) - 1  # concat_padding_mask=True
    model_channels = w.shape[0]

    if model_channels == 2048:
        num_blocks, num_heads = 28, 16
    elif model_channels == 5120:
        num_blocks, num_heads = 36, 40
    else:
        raise RuntimeError(f"未知的 model_channels={model_channels}")

    config = dict(
        max_img_h=240, max_img_w=240, max_frames=128,
        in_channels=in_channels, out_channels=16,
        patch_spatial=2, patch_temporal=1,
        concat_padding_mask=True,
        model_channels=model_channels,
        num_blocks=num_blocks, num_heads=num_heads,
        crossattn_emb_channels=1024,
        pos_emb_cls="rope3d", pos_emb_learnable=True,
        pos_emb_interpolation="crop",
        use_adaln_lora=True, adaln_lora_dim=256,
        rope_h_extrapolation_ratio=4.0 if in_channels == 16 else 3.0,
        rope_w_extrapolation_ratio=4.0 if in_channels == 16 else 3.0,
        rope_t_extrapolation_ratio=1.0,
    )

    model = Anima(**config)

    # 加载权重
    sd = _load_safetensors_state_dict(Path(transformer_path))
    info = _load_weights_best_effort(model, sd, label="Transformer")

    # 如果 checkpoint 中完全没有 llm_adapter 权重，随机初始化会把 cross-attn 条件搞乱，直接禁用更安全
    has_llm_adapter = any("llm_adapter" in k for k in sd.keys())
    if not has_llm_adapter and hasattr(model, "llm_adapter"):
        try:
            model.llm_adapter = None
            logger.warning("检测到 checkpoint 不包含 llm_adapter 权重：已禁用 llm_adapter（回退为直接使用 Qwen embeddings）")
        except Exception:
            pass
    model = model.to(device=device, dtype=dtype)
    model.requires_grad_(False)

    logger.info(f"Anima 模型加载完成: {model_channels}ch, {num_blocks} blocks")
    return model


def load_vae(vae_path, device, dtype, repo_root):
    """加载 VAE"""
    wan_vae = load_module_from_path("wan_vae", repo_root / "wan" / "vae2_1.py")
    WanVAE = wan_vae.WanVAE_

    cfg = dict(
        dim=96, z_dim=16, dim_mult=[1, 2, 4, 4],
        num_res_blocks=2, attn_scales=[],
        temperal_downsample=[False, True, True], dropout=0.0,
    )

    model = WanVAE(**cfg).eval().requires_grad_(False)

    sd = _load_safetensors_state_dict(Path(vae_path))
    _load_weights_best_effort(model, sd, label="VAE")
    model = model.to(device=device, dtype=dtype)

    # VAE 归一化参数
    mean = torch.tensor([
        -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
    ], dtype=dtype, device=device)
    std = torch.tensor([
        2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
    ], dtype=dtype, device=device)

    class VAEWrapper:
        pass

    wrapper = VAEWrapper()
    wrapper.model = model
    wrapper.mean = mean
    wrapper.std = std
    wrapper.scale = [mean, 1.0 / std]

    logger.info("VAE 加载完成")
    return wrapper


def load_text_encoders(qwen_path, t5_tokenizer_path, device, dtype):
    """加载文本编码器"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer

    # Qwen
    qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
    qwen_model = AutoModelForCausalLM.from_pretrained(
        qwen_path, torch_dtype=dtype, trust_remote_code=True
    ).to(device).eval().requires_grad_(False)

    # T5 tokenizer
    if t5_tokenizer_path and Path(t5_tokenizer_path).exists():
        t5_tokenizer = T5Tokenizer.from_pretrained(t5_tokenizer_path)
    else:
        t5_tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")

    logger.info("文本编码器加载完成")
    return qwen_model, qwen_tokenizer, t5_tokenizer


# ============================================================================
# 文本编码 / 训练时采样：迁移到 trainer/text_encode.py 和 trainer/sampling.py
# ============================================================================
from trainer.text_encode import (
    _parse_weighted_tag,
    _build_qwen_text_from_prompt,
    encode_qwen,
    tokenize_t5_weighted,
)
from trainer.sampling import (
    _time_snr_shift,
    _flow_sigmas_simple,
    _default_noise_sampler,
    _sample_er_sde_const_x0,
    sample_image,
)
from trainer.objective import (
    TimestepConfig,
    NoiseConfig,
    LossConfig,
    TrainingObjectiveConfig,
    build_training_objective_config,
    sample_t,
    apply_timestep_schedule_shift,
    AdaptiveTimestepSampler,
    make_noise,
    make_noise_from_config,
    _huber_delta_for_t,
    per_sample_loss,
    per_sample_highfreq_loss,
    adaptive_timestep_metric_signal,
    compute_loss_weight,
    apply_loss_weighting,
    compute_grad_norm,
    forward_with_optional_checkpoint,
)
from trainer.lora import LoRALayer, LoKrLayer, LoRALinear, LoRAInjector


# ============================================================================
# 训练状态保存/恢复（断点续训）
# ============================================================================

def save_training_state(path, injector, optimizer, epoch, global_step, loss_history=None, rng_state=None, monitor_state=None, scheduler=None):
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
    """加载训练状态，返回 (epoch, global_step, loss_history, monitor_state)"""
    logger.info(f"加载训练状态: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    
    # 加载 LoRA 权重
    lora_sd = state["lora_state_dict"]
    for name, lora in injector.injected.items():
        base = "lora_unet_" + name.replace(".", "_")
        if injector.use_lokr:
            w1_key = f"{base}.lokr_w1"
            w2a_key = f"{base}.lokr_w2_a"
            w2b_key = f"{base}.lokr_w2_b"
            dora_key = f"{base}.dora_scale"
            w2_old_key = f"{base}.lokr_w2"
            if w1_key in lora_sd and w2a_key in lora_sd and w2b_key in lora_sd:
                lora.adapter.lokr_w1.data.copy_(lora_sd[w1_key])
                lora.adapter.lokr_w2_a.data.copy_(lora_sd[w2a_key])
                lora.adapter.lokr_w2_b.data.copy_(lora_sd[w2b_key])
                if getattr(lora, "use_dora", False) and dora_key in lora_sd:
                    dora_scale = lora_sd[dora_key].reshape(-1)
                    lora.dora_scale.data.copy_(dora_scale.to(device=lora.dora_scale.device, dtype=lora.dora_scale.dtype))
            elif w1_key in lora_sd and w2_old_key in lora_sd:
                logger.warning(f"跳过旧格式 lokr_w2 全矩阵层: {name}（需重新训练）")
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


# ============================================================================
# 数据集
# ============================================================================

class BucketManager:
    """ARB 分桶管理"""
    def __init__(self, base_reso=1024, min_reso=512, max_reso=2048, step=64):
        self.base_reso = base_reso
        self.buckets = self._generate(min_reso, max_reso, step, base_reso)

    def _generate(self, min_r, max_r, step, base):
        buckets = []
        base_area = base * base
        for w in range(min_r, max_r + 1, step):
            for h in range(min_r, max_r + 1, step):
                if abs(w * h - base_area) / base_area > 0.1:
                    continue
                if max(w/h, h/w) > 2.0:
                    continue
                buckets.append((w, h))
        return buckets

    def get_bucket(self, w, h):
        aspect = w / h
        best = (self.base_reso, self.base_reso)
        best_diff = float("inf")
        for bw, bh in self.buckets:
            diff = abs(aspect - bw/bh)
            if diff < best_diff:
                best_diff = diff
                best = (bw, bh)
        return best


class ImageDataset(Dataset):
    """
    图像数据集
    
    支持两种 caption 格式：
    1. JSON 文件（优先）- 支持分类 shuffle
    2. TXT 文件（回退）- 传统 shuffle
    """
    EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def __init__(self, data_dir, resolution=1024, bucket_mgr=None,
                 shuffle_caption=False, keep_tokens=0, flip_augment=False,
                 tag_dropout=0.0, prefer_json=True, caption_override=None,
                 freq_balanced_dropout_strength=0.0):
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.bucket_mgr = bucket_mgr
        self.shuffle_caption = shuffle_caption
        self.keep_tokens = keep_tokens
        self.flip_augment = flip_augment
        self.tag_dropout = tag_dropout
        self.prefer_json = prefer_json
        self.caption_override = caption_override  # 正则集：统一 caption，如 "1girl, solo"
        # ★ v5 ② frequency-balanced tag dropout
        # 0 = 关闭；>0 启用。在数据集 init 时统计 tag 频率，对在数据集中过度共现的 tag 额外提高 dropout
        # 概率，强迫模型把"风格"与"高频共现 tag"解耦。完全数据驱动，自动适配任何画师。
        self.freq_balanced_dropout_strength = float(freq_balanced_dropout_strength or 0.0)
        self.tag_freq = {}  # tag -> frequency in [0, 1]，scan 之后填充
        
        # 尝试导入 caption_utils（直接导入避开 __init__.py）
        self.caption_utils = None
        if prefer_json:
            try:
                import importlib.util
                import sys
                
                # 直接加载 caption_utils.py
                utils_path = Path(__file__).parent / "utils" / "caption_utils.py"
                if utils_path.exists():
                    spec = importlib.util.spec_from_file_location("caption_utils", utils_path)
                    caption_module = importlib.util.module_from_spec(spec)
                    sys.modules["caption_utils"] = caption_module
                    spec.loader.exec_module(caption_module)
                    
                    self.caption_utils = {
                        "load_and_build": caption_module.load_and_build_caption,
                        "load_json": caption_module.load_caption_json,
                        "normalize": caption_module.normalize_caption_json,
                        "build": caption_module.build_caption_from_json,
                    }
                    logger.info("JSON caption 模式已启用（分类 shuffle）")
                else:
                    logger.warning(f"caption_utils.py 未找到: {utils_path}")
            except Exception as e:
                logger.warning(f"caption_utils 加载失败: {e}，回退到 TXT 模式")
        
        self.samples = self._scan()
        json_count = sum(1 for s in self.samples if s.get("json_path"))
        txt_count = len(self.samples) - json_count
        logger.info(f"数据集: {len(self.samples)} 样本 (JSON: {json_count}, TXT: {txt_count})")

        # 用与 __getitem__ 完全一致的 PIL 路径填充 bucket_key
        self._finalize_bucket_keys()
        self.bucket_for_index = [s["bucket_key"] for s in self.samples]
        # 诊断：统计 bucket 分布
        from collections import Counter
        dist = Counter(self.bucket_for_index)
        logger.info(f"  bucket 分布: {len(dist)} 种, 例: {list(dist.most_common(5))}")

        # ★ v5 ② 统计 tag 频率（仅在启用 freq_balanced dropout 时执行；避免无谓 IO）
        if self.freq_balanced_dropout_strength > 0:
            if json_count > 0:
                logger.warning(
                    f"[freq_balanced] 检测到 {json_count} 个 JSON caption；本机制只对 TXT caption 生效。"
                    "JSON 走 caption_utils 内置的分类 shuffle / dropout，与频率均衡无关。"
                )
            self._compute_tag_freq()

    def _finalize_bucket_keys(self):
        """For each sample, compute bucket_key via the same code path that __getitem__ uses,
        so BucketBatchSampler can reliably group same-shape tensors. Caches per unique img path."""
        from PIL import Image as _PILImage
        cache = {}
        for sample in self.samples:
            img_path = sample["image"]
            key = cache.get(img_path)
            if key is None:
                if self.bucket_mgr is None:
                    key = (self.resolution, self.resolution)
                else:
                    try:
                        img = _PILImage.open(img_path)
                        # 与 __getitem__ 一致：读 width/height。.convert 不改变尺寸，可省。
                        w, h = img.width, img.height
                        try:
                            img.close()
                        except Exception:
                            pass
                        bw, bh = self.bucket_mgr.get_bucket(w, h)
                        key = (bh, bw)  # (h, w)
                    except Exception as e:
                        logger.warning(f"[bucket_key] 无法读取 {img_path}: {e}，回退到 ({self.resolution},{self.resolution})")
                        key = (self.resolution, self.resolution)
                cache[img_path] = key
            sample["bucket_key"] = key

    def _compute_tag_freq(self):
        """★ v5 ② 扫描全部 caption 文件，统计每个 tag 在数据集中出现的图片数 / 总图片数。

        每张图只算一次（即使 tag 在 caption 里重复也只计 1），相同图（被 repeat 而进入 samples
        多次）也只算一次，避免 repeat 高的图把它的 tag 频率人为放大。
        """
        from collections import Counter
        cnt = Counter()
        seen_imgs = set()
        total_imgs = 0

        for s in self.samples:
            img_key = str(s.get("image", ""))
            if img_key in seen_imgs:
                continue
            seen_imgs.add(img_key)

            caption_text = None
            # 优先 TXT（与 __getitem__ 的回退顺序一致）；JSON 结构性 tag 不参与此机制
            txt_path = s.get("txt_path")
            if txt_path:
                try:
                    caption_text = txt_path.read_text(encoding="utf-8").strip()
                except Exception:
                    caption_text = None

            if not caption_text:
                continue

            total_imgs += 1
            if "," in caption_text:
                tags = [t.strip() for t in caption_text.split(",") if t.strip()]
            else:
                tags = [t for t in caption_text.split() if t]
            # 每张图每个 tag 只计 1 次
            for t in set(tags):
                cnt[t] += 1

        if total_imgs == 0:
            logger.warning("[freq_balanced] 没有可统计的 TXT caption，禁用频率加权 dropout。")
            self.tag_freq = {}
            return

        self.tag_freq = {t: c / total_imgs for t, c in cnt.items()}
        # 诊断：打印共现最高的 10 个 tag
        top = sorted(self.tag_freq.items(), key=lambda kv: kv[1], reverse=True)[:10]
        top_str = ", ".join(f"{t}={f:.2f}" for t, f in top)
        logger.info(
            f"[freq_balanced] 已统计 {len(self.tag_freq)} 个 tag 的频率 "
            f"(strength={self.freq_balanced_dropout_strength:.2f}). Top10: {top_str}"
        )

    def _scan(self):
        samples = []
        # 扫描所有子目录，寻找图像及其对应的标签文件
        for img_path in self.data_dir.rglob("*"):
            if img_path.suffix.lower() not in self.EXTS:
                continue
            
            # 解析目录名中的重复次数 (例如 10_tags)
            repeats = 1
            parent_name = img_path.parent.name
            if "_" in parent_name:
                prefix = parent_name.split("_", 1)[0]
                if prefix.isdigit():
                    repeats = max(1, int(prefix))
            
            sample = {"image": img_path}
            
            # 优先查找 JSON
            json_path = img_path.with_suffix(".json")
            if self.prefer_json and json_path.exists():
                sample["json_path"] = json_path
                sample["txt_path"] = None
            else:
                # 回退到 TXT
                txt_path = img_path.with_suffix(".txt")
                if not txt_path.exists():
                    txt_path = img_path.with_suffix(".caption")
                if not txt_path.exists():
                    continue
                sample["json_path"] = None
                sample["txt_path"] = txt_path

            # bucket_key 留空，等会在 _finalize_bucket_keys 里用与 __getitem__ 一致
            # 的路径（Image.open + convert("RGB")）批量计算，避免 _scan 与运行时
            # PIL 行为差异导致 sampler 分桶失效。
            sample["bucket_key"] = None

            # 按重复次数添加样本
            for _ in range(repeats):
                samples.append(sample.copy())
        return samples

    def _process_caption_txt(self, caption):
        """处理 TXT caption: 传统 tag 打乱 + keep_tokens + tag_dropout

        tag_dropout 与 caption_utils.build_caption_from_json 中的语义一致：
          - 仅丢弃 keep_tokens 之后的可变标签；keep_tokens（角色名/触发词）始终保留
          - 若 dropout 后可变部分全空，强制保留其中随机一个，避免 caption 退化为只有触发词
        """
        if not caption:
            return ""
        if "," in caption:
            tags = [t.strip() for t in caption.split(",") if t.strip()]
        else:
            tags = [t for t in caption.split() if t]

        if not tags:
            return ""

        keep_n = max(int(self.keep_tokens or 0), 0)
        kept = tags[:keep_n]
        rest = tags[keep_n:]

        if self.shuffle_caption and rest:
            random.shuffle(rest)

        dropout = float(self.tag_dropout or 0.0)
        freq_strength = float(getattr(self, "freq_balanced_dropout_strength", 0.0) or 0.0)
        tag_freq = getattr(self, "tag_freq", {}) or {}

        if rest and (dropout > 0.0 or (freq_strength > 0.0 and tag_freq)):
            survivors = []
            for t in rest:
                # 通用 dropout
                if dropout > 0.0 and random.random() < dropout:
                    continue
                # ★ v5 ② 频率加权 dropout
                # 触发词在 kept 里完全免疫；这里只对 rest 起作用。
                # 公式：extra_drop = strength * max(0, freq - 0.3) / 0.7
                #   freq=1.0 → extra_drop = strength；freq=0.3 → 0；线性插值。
                #   0.3 是经验阈值：低于此值的 tag 视为"真实变量"不需要解耦。
                if freq_strength > 0.0:
                    freq = tag_freq.get(t, 0.0)
                    if freq > 0.3:
                        extra_drop = freq_strength * (freq - 0.3) / 0.7
                        if random.random() < extra_drop:
                            continue
                survivors.append(t)
            if not survivors:
                survivors = [random.choice(rest)]
            rest = survivors

        return ", ".join(kept + rest)

    def _process_caption_json(self, json_path):
        """处理 JSON caption: 分类 shuffle"""
        if self.caption_utils is None:
            return None
        
        try:
            raw_json = self.caption_utils["load_json"](json_path)
            if raw_json is None:
                return None
            
            # 检查是否已经是标准格式
            if "tags" in raw_json and "meta" in raw_json:
                normalized = raw_json
            else:
                normalized = self.caption_utils["normalize"](raw_json)
            
            # 构建 caption（分类 shuffle）
            return self.caption_utils["build"](
                normalized,
                shuffle_appearance=self.shuffle_caption,
                shuffle_tags=self.shuffle_caption,
                shuffle_environment=self.shuffle_caption,
                tag_dropout=self.tag_dropout,
            )
        except Exception as e:
            logger.warning(f"JSON 处理失败 {json_path}: {e}")
            return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import numpy as np
        from PIL import Image
        sample = self.samples[idx]
        img = Image.open(sample["image"]).convert("RGB")
        
        # 获取 caption（正则集可用 caption_override 统一覆盖）
        caption = None
        if self.caption_override is not None:
            caption = self.caption_override
        elif sample.get("json_path"):
            caption = self._process_caption_json(sample["json_path"])
        
        if caption is None and sample.get("txt_path"):
            caption = sample["txt_path"].read_text(encoding="utf-8").strip()
            caption = self._process_caption_txt(caption)
        
        if caption is None:
            caption = ""

        # ARB 分桶
        if self.bucket_mgr:
            tw, th = self.bucket_mgr.get_bucket(img.width, img.height)
        else:
            tw = th = self.resolution

        # 缩放裁剪
        scale = max(tw / img.width, th / img.height)
        nw, nh = int(img.width * scale), int(img.height * scale)
        img = img.resize((nw, nh), Image.LANCZOS)

        left = (nw - tw) // 2
        top = (nh - th) // 2
        img = img.crop((left, top, left + tw, top + th))

        # 水平翻转增强
        if self.flip_augment and random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # 转 tensor [-1, 1]
        arr = np.array(img).astype(np.float32) / 127.5 - 1.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)

        return {"pixel_values": tensor, "caption": caption, "image": str(sample["image"])}


class RepeatDataset(Dataset):
    """Kohya 风格数据集重复"""
    def __init__(self, dataset, repeats=1):
        self.dataset = dataset
        self.repeats = max(1, int(repeats))

    def __len__(self):
        return len(self.dataset) * self.repeats

    def __getitem__(self, idx):
        return self.dataset[idx % len(self.dataset)]


class MergedDataset(Dataset):
    """合并主数据集与正则数据集（Kohya 风格 reg）"""
    def __init__(self, main_dataset, reg_dataset):
        self.main_dataset = main_dataset
        self.reg_dataset = reg_dataset
        self._main_len = len(main_dataset)
        self._reg_len = len(reg_dataset)

        # 为 BucketBatchSampler 构建 bucket_for_index
        self.bucket_for_index = self._build_bucket_for_index()

    def _get_cached_dataset(self, d):
        bfi = getattr(d, "bucket_for_index", None)
        if bfi is not None and len(bfi) > 0:
            return d
        if hasattr(d, "dataset"):
            return self._get_cached_dataset(d.dataset)
        return None

    def _build_bucket_for_index(self):
        main_cached = self._get_cached_dataset(self.main_dataset)
        reg_cached = self._get_cached_dataset(self.reg_dataset)
        buckets = []
        if main_cached and main_cached.bucket_for_index:
            main_base_len = len(main_cached.bucket_for_index)
            for idx in range(self._main_len):
                b = main_cached.bucket_for_index[idx % main_base_len]
                buckets.append(b if b is not None else (0, 0))
        else:
            buckets.extend([(0, 0)] * self._main_len)
        if reg_cached and reg_cached.bucket_for_index:
            reg_base_len = len(reg_cached.bucket_for_index)
            for idx in range(self._reg_len):
                b = reg_cached.bucket_for_index[idx % reg_base_len]
                buckets.append(b if b is not None else (0, 0))
        else:
            buckets.extend([(0, 0)] * self._reg_len)
        return buckets

    def __len__(self):
        return self._main_len + self._reg_len

    def __getitem__(self, idx):
        if idx < self._main_len:
            return self.main_dataset[idx]
        return self.reg_dataset[idx - self._main_len]


class BucketBatchSampler:
    """Batch sampler that groups samples by bucket so tensors in each batch have the same size.

    Per-index resolution: walks the dataset wrapping chain (RepeatDataset / MergedDataset)
    for every outer index to look up the underlying ImageDataset / CachedLatentDataset's
    bucket_for_index. This avoids any indirection bugs in pre-built bucket_for_index lists.
    """
    def __init__(self, dataset, batch_size, drop_last=True, shuffle=True, seed=42):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        # Pre-compute per-index bucket keys via direct walk
        self._bucket_keys = self._build_keys(dataset)
        # Diagnostics
        unique = set(self._bucket_keys)
        none_count = sum(1 for k in self._bucket_keys if k is None)
        logger.info(
            "[BucketBatchSampler] dataset_len=%d unique_buckets=%d none=%d (e.g. %s)",
            len(self._bucket_keys), len(unique), none_count,
            list(unique)[:5],
        )
        if none_count == len(self._bucket_keys):
            logger.warning(
                "[BucketBatchSampler] 没有任何样本能解析到 bucket_key，"
                "将退化为顺序分批（可能在 ARB 模式下因尺寸不一致而崩溃）。"
                "请检查 ImageDataset/CachedLatentDataset 是否正确填充了 bucket_for_index。"
            )
        # 计算每个桶的批数（drop_last 在每个桶内独立生效）。旧实现 `n // bs` 在多桶 +
        # drop_last 时会高估批数 —— 例如 2 个桶各 5 张图、bs=4、drop_last=True，真实是
        # 1+1=2 个 batch，旧实现报 10//4=2（巧合相同）或在 [3,5] 这种分布下报 8//4=2
        # 但真实是 0+1=1。`total_steps = len(dataloader) * epochs / grad_accum` 会跟着
        # 偏，进而把 cosine 调度器的 T_max 设错。
        self._total_batches = self._compute_total_batches()

    def _compute_total_batches(self):
        from collections import Counter
        counts = Counter(tuple(k) if k is not None else (0, 0) for k in self._bucket_keys)
        bs = self.batch_size
        total = 0
        if self.drop_last:
            for n in counts.values():
                total += n // bs
        else:
            for n in counts.values():
                total += (n + bs - 1) // bs
        return total

    def _build_keys(self, dataset):
        n = len(dataset)
        keys = [None] * n
        for i in range(n):
            keys[i] = self._lookup(dataset, i)
        return keys

    def _lookup(self, d, idx):
        """Resolve the bucket key for a given outer index by walking dataset wrappers.

        Priority: MergedDataset routing → RepeatDataset (.dataset) → leaf bucket_for_index
        → CachedLatentDataset (.base_dataset). Leaf is preferred over base_dataset because
        CachedLatentDataset has its own complete bucket_for_index aligned with cached samples.
        """
        main = getattr(d, "main_dataset", None)
        reg = getattr(d, "reg_dataset", None)
        if main is not None and reg is not None:
            ml = getattr(d, "_main_len", len(main))
            if idx < ml:
                return self._lookup(main, idx)
            return self._lookup(reg, idx - ml)
        # RepeatDataset wraps another dataset via .dataset
        inner = getattr(d, "dataset", None)
        if inner is not None and inner is not d and not isinstance(inner, list):
            try:
                return self._lookup(inner, idx % len(inner))
            except TypeError:
                pass
        # Leaf: ImageDataset (.bucket_for_index present and indexed directly)
        bfi = getattr(d, "bucket_for_index", None)
        if bfi is not None and len(bfi) > 0:
            return bfi[idx % len(bfi)]
        # CachedLatentDataset fallback (rare: no bucket_for_index yet)
        inner = getattr(d, "base_dataset", None)
        if inner is not None and inner is not d:
            return self._lookup(inner, idx % len(inner))
        return None

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        # 用预计算的 per-bucket 加和（见 __init__ 末尾）。
        return self._total_batches

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        bucket_to_indices = {}
        for idx, key in enumerate(self._bucket_keys):
            if key is None:
                key = (0, 0)
            bucket_to_indices.setdefault(tuple(key), []).append(idx)

        buckets = list(bucket_to_indices.keys())
        if self.shuffle:
            rng.shuffle(buckets)
        for bucket in buckets:
            indices = bucket_to_indices[bucket]
            if self.shuffle:
                rng.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch


class CachedLatentDataset(Dataset):
    """Kohya 风格 npz 文件缓存的数据集"""
    def __init__(self, base_dataset, vae, device, dtype, cache_dir=None):
        import numpy as np
        self.base_dataset = base_dataset
        self.np = np
        # 获取原始数据集的 samples 列表
        self.samples = self._get_base_samples(base_dataset)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.bucket_for_index = []
        self._build_cache(vae, device, dtype)

    def _get_base_samples(self, dataset):
        """获取原始 ImageDataset 的 samples"""
        if hasattr(dataset, "samples"):
            return dataset.samples
        elif hasattr(dataset, "dataset"):
            return self._get_base_samples(dataset.dataset)
        return []

    def _get_npz_path(self, img_path):
        """获取图像对应的 npz 缓存路径"""
        img_path = Path(img_path)
        return img_path.with_suffix(".npz")

    def _is_cache_valid(self, img_path, npz_path):
        """检查缓存是否有效（图像未修改，且格式含 latent 键）。
        若为其他模型的不兼容缓存，则删除并返回 False。"""
        if not npz_path.exists():
            return False
        if npz_path.stat().st_mtime < img_path.stat().st_mtime:
            return False
        try:
            data = self.np.load(npz_path)
            if "latent" not in data.files:
                npz_path.unlink()
                logger.debug(f"已删除不兼容缓存: {npz_path.name}")
                return False
            latent = data["latent"]
            if latent.ndim != 4:
                logger.warning(f"删除异常 latent 缓存（维度应为 C,T,H,W）: {npz_path}")
                npz_path.unlink()
                return False
            if latent.shape[0] != 16:
                logger.warning(f"删除疑似非 Anima/Qwen VAE 缓存（C={latent.shape[0]}，应为 16）: {npz_path}")
                npz_path.unlink()
                return False
            if not self.np.isfinite(latent).all():
                logger.warning(f"删除非有限 latent 缓存: {npz_path}")
                npz_path.unlink()
                return False
        except Exception:
            try:
                npz_path.unlink()
            except Exception:
                pass
            return False
        return True

    def _build_cache(self, vae, device, dtype):
        """构建/加载 npz 缓存"""
        logger.info("检查 VAE latent 缓存...")
        to_encode = []
        for i, sample in enumerate(self.samples):
            img_path = sample["image"]
            npz_path = self._get_npz_path(img_path)
            if not self._is_cache_valid(img_path, npz_path):
                to_encode.append(i)

        if to_encode:
            logger.info(f"需要编码 {len(to_encode)}/{len(self.samples)} 张图像...")
            self._encode_and_save(to_encode, vae, device, dtype)
        else:
            logger.info(f"所有 {len(self.samples)} 张图像已缓存")

        self._fill_bucket_for_index()

    def _fill_bucket_for_index(self):
        """Fill bucket_for_index for all samples (needed for BucketBatchSampler).
        Uses latent spatial shape (h, w) as grouping key so batches have consistent tensor sizes."""
        self.bucket_for_index = [None] * len(self.samples)
        for i in range(len(self.samples)):
            npz_path = self._get_npz_path(self.samples[i]["image"])
            if not npz_path.exists():
                continue
            data = self.np.load(npz_path)
            latent = data["latent"]
            s = latent.shape
            if len(s) == 5:
                _, _, _, h, w = s
            else:
                _, _, h, w = s
            self.bucket_for_index[i] = (int(h), int(w))

    def _encode_and_save(self, indices, vae, device, dtype):
        """编码图像并保存为 npz"""
        for count, i in enumerate(indices):
            item = self.base_dataset[i]
            pixels = item["pixel_values"].unsqueeze(0).to(device, dtype=dtype)
            _, _, ph, pw = pixels.shape
            bucket_w, bucket_h = pw, ph
            with torch.no_grad():
                pixels_5d = pixels.unsqueeze(2)
                latent = vae.model.encode(pixels_5d, vae.scale)
            if not torch.isfinite(latent).all():
                logger.warning(f"VAE 编码产生非有限 latent，跳过缓存: {self.samples[i]['image']}")
                continue
            latent_np = latent.squeeze(0).cpu().float().numpy()
            npz_path = self._get_npz_path(self.samples[i]["image"])
            self.np.savez(npz_path, latent=latent_np, bucket_w=bucket_w, bucket_h=bucket_h)
            if (count + 1) % 10 == 0 or count == len(indices) - 1:
                logger.info(f"  编码进度: {count + 1}/{len(indices)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        npz_path = self._get_npz_path(sample["image"])
        data = self.np.load(npz_path)
        latent = torch.from_numpy(data["latent"])
        if not torch.isfinite(latent).all():
            raise RuntimeError(f"读取到非有限 latent 缓存: {npz_path}")
        
        # 获取 base_dataset 的引用（处理可能的嵌套）
        base = self.base_dataset
        while hasattr(base, "dataset"):
            base = base.dataset

        # NOTE: 不要在缓存 latent 上做空间 flip！
        # Anima/Qwen VAE 的 conv encoder 不是 flip-equivariant，
        # 即 flip(encode(img)) ≠ encode(flip(img))；
        # 在 latent 空间翻转会喂给训练"非自然"的潜变量，模型学到的是
        # 偏离真实分布的 latent，推理时表现为马赛克 / 边缘溶解。
        # flip 增强在缓存阶段（base ImageDataset 的 __getitem__ 里做图像 flip
        # 后再 encode）已经生效一次；想要每个 epoch 重新 flip，请关闭 cache_latents。

        # 处理 caption（正则集 caption_override 优先）
        caption = None
        if getattr(base, "caption_override", None) is not None:
            caption = base.caption_override
        elif sample.get("json_path") and hasattr(base, "_process_caption_json"):
            caption = base._process_caption_json(sample["json_path"])
        
        if caption is None and sample.get("txt_path"):
            caption = sample["txt_path"].read_text(encoding="utf-8").strip()
            if hasattr(base, "_process_caption_txt"):
                caption = base._process_caption_txt(caption)
        
        if caption is None:
            caption = ""
        
        return {"latent": latent, "caption": caption, "image": str(sample["image"])}


def collate_fn(batch):
    """DataLoader collate"""
    shapes = [tuple(b["pixel_values"].shape) for b in batch]
    if len(set(shapes)) > 1:
        # 诊断信息：BucketBatchSampler 应该已按 bucket 分组，出现混合尺寸说明分桶失效
        details = [
            f"  - {b.get('image', '?')}: shape={tuple(b['pixel_values'].shape)}"
            for b in batch
        ]
        raise RuntimeError(
            "[collate_fn] 同一 batch 出现不同尺寸张量，BucketBatchSampler 分桶失效。\n"
            "Batch 内容:\n" + "\n".join(details)
        )
    pixels = torch.stack([b["pixel_values"] for b in batch])
    captions = [b["caption"] for b in batch]
    images = [b.get("image", "") for b in batch]
    return {"pixel_values": pixels, "captions": captions, "images": images}


def collate_fn_cached(batch):
    """DataLoader collate for cached latents"""
    shapes = [tuple(b["latent"].shape) for b in batch]
    if len(set(shapes)) > 1:
        details = [
            f"  - {b.get('image', '?')}: latent_shape={tuple(b['latent'].shape)}"
            for b in batch
        ]
        raise RuntimeError(
            "[collate_fn_cached] 同一 batch 出现不同 latent 尺寸，BucketBatchSampler 分桶失效。\n"
            "Batch 内容:\n" + "\n".join(details)
        )
    latents = torch.stack([b["latent"] for b in batch])
    captions = [b["caption"] for b in batch]
    images = [b.get("image", "") for b in batch]
    return {"latents": latents, "captions": captions, "images": images}


# ============================================================================
# 参数解析
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Anima LoRA Trainer v2")
    # 配置文件
    p.add_argument("--config", default="", help="YAML 配置文件路径")
    # 路径
    p.add_argument("--data-dir", default="", help="数据集目录")
    p.add_argument("--transformer", default="", help="transformer safetensors")
    p.add_argument("--vae", default="", help="VAE safetensors")
    p.add_argument("--qwen", default="", help="Qwen 模型目录")
    p.add_argument("--t5-tokenizer", default="", help="T5 tokenizer 目录")
    p.add_argument("--output-dir", default="./output", help="输出目录")
    p.add_argument("--output-name", default="anima_lora", help="输出名称")

    # 训练参数
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-scheduler", default="none", choices=["none", "cosine", "cosine_with_restart"], help="学习率调度器")
    p.add_argument("--lr-scheduler-t0", type=int, default=500, help="cosine_with_restart: 首次 restart 周期 (step)")
    p.add_argument("--lr-scheduler-t-mult", type=float, default=2.0, help="cosine_with_restart: 每次 restart 周期倍数")
    p.add_argument("--lr-scheduler-eta-min", type=float, default=0.0, help="cosine/cosine_with_restart: 最小学习率")
    p.add_argument("--weight-decay", type=float, default=0.01, help="AdamW 权重衰减 (L2 正则, 0=禁用)")
    p.add_argument("--grad-clip-max-norm", type=float, default=1.0, help="梯度裁剪最大范数 (0=禁用；ProdigyPlus 推荐设 0)")
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--mixed-precision", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--grad-checkpoint", action="store_true", help="启用梯度检查点减少显存")
    p.add_argument("--max-steps", type=int, default=0, help="最大训练步数 (0=无限制)")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")

    # 数据集参数
    p.add_argument("--repeats", type=int, default=1, help="数据集重复次数 (Kohya 风格)")
    p.add_argument("--reg-data-dir", default="", help="正则数据集目录（防过拟合，Kohya 风格）")
    p.add_argument("--reg-repeats", type=int, default=1, help="正则集每张图重复次数")
    p.add_argument("--reg-caption", default="", help="正则集统一 caption，如 1girl, solo（空则用各图自带）")
    p.add_argument("--shuffle-caption", action="store_true", help="打乱 caption tags（分类 shuffle）")
    p.add_argument("--keep-tokens", type=int, default=0, help="保留前 N 个 tokens 不打乱")
    p.add_argument("--flip-augment", action="store_true", help="随机水平翻转增强")
    p.add_argument("--tag-dropout", type=float, default=0.0, help="Tag dropout 概率 (0-1)")
    p.add_argument("--no-prefer-json", action="store_true", help="禁用 JSON 优先模式")
    p.add_argument("--cache-latents", action="store_true", help="缓存 VAE latent 加速训练")

    # LoRA 参数
    p.add_argument("--lora-type", choices=["lora", "lokr"], default="lokr")
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=float, default=32.0)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--lokr-factor", type=int, default=8)
    p.add_argument("--lora-variant", choices=["base", "dora"], default="base",
                   help="Adapter variant. 'dora' enables LyCORIS/ComfyUI-compatible DoRA-LoKr.")
    p.add_argument("--dora-export-mode", choices=["native", "diff", "merged_model"], default="native",
                   help="Export mode: native LoKr + .dora_scale, exact ComfyUI .diff, or full merged transformer safetensors.")
    p.add_argument("--lora-targets", default=None,
                   help="逗号分隔的 Linear 模块名片段；默认使用 q/k/v/output_proj 和 mlp.layer1/2。")
    p.add_argument("--lora-exclude-prefixes", default=None,
                   help="逗号分隔，命中前缀的 Linear 不被注入 LoRA/LoKr，例如 'llm_adapter.'。默认空（与 base 一致）")
    p.add_argument("--resume-lora", default="", help="从已有 LoRA 继续训练（safetensors 路径）")

    # 采样参数
    p.add_argument("--sample-every", type=int, default=0, help="每 N 个 epoch 采样一次 (0=禁用)")
    p.add_argument("--sample-steps", type=int, default=0, help="每 N 个 step 采样一次 (0=禁用)")
    p.add_argument("--sample-prompt", default="1girl, masterpiece", help="采样提示词")
    p.add_argument("--sample-cfg-scale", type=float, default=4.0, help="采样 CFG（设为 1 表示不做 CFG，仅用正面条件）")
    p.add_argument("--sample-negative-prompt", default="", help="采样负面提示词（留空使用默认负面）")
    p.add_argument("--sample-width", type=int, default=0, help="采样宽度（0=跟随 resolution）")
    p.add_argument("--sample-height", type=int, default=0, help="采样高度（0=跟随 resolution）")
    p.add_argument("--sample-seed", type=int, default=0, help="采样随机种子（0=不固定）")
    p.add_argument("--sample-infer-steps", type=int, default=25, help="采样推理步数（对齐 ComfyUI 默认 25）")
    p.add_argument("--sample-sampler-name", default="er_sde", help="采样器名称（对齐 ComfyUI: er_sde）")
    p.add_argument("--sample-scheduler", default="simple", help="采样 scheduler（对齐 ComfyUI: simple）")

    # 保存参数
    p.add_argument("--save-every", type=int, default=0, help="每 N 个 epoch 保存 (0=仅结束时)")
    p.add_argument("--save-state-every", type=int, default=0, help="每 N 步保存完整训练状态（可断点续训）")
    p.add_argument("--resume-state", default="", help="从训练状态恢复（.pt 文件路径）")
    p.add_argument("--seed", type=int, default=42)

    # 进度显示
    p.add_argument("--no-progress", action="store_true", help="禁用动态进度显示")
    p.add_argument("--loss-curve-steps", type=int, default=100, help="Loss 曲线显示步数 (0=禁用)")
    p.add_argument("--no-live-curve", action="store_true", help="禁用实时 Loss 曲线刷新")
    p.add_argument("--no-monitor", action="store_true", help="禁用 Web 监控面板")
    p.add_argument("--monitor-host", default="0.0.0.0", help="监控面板绑定地址（默认 0.0.0.0 以支持远程访问）")
    p.add_argument("--monitor-port", type=int, default=8765, help="监控面板端口")
    p.add_argument("--no-browser", action="store_true", help="不自动打开监控面板浏览器")
    p.add_argument("--log-every", type=int, default=10, help="日志输出间隔")
    p.add_argument("--debug-first-batches", type=int, default=0, help="记录前 N 个优化步的张量统计，用于对齐 loss 标尺")
    p.add_argument("--grad-norm-log-every", type=int, default=0, help="每 N 个优化步记录梯度范数和裁切状态 (0=禁用)")

    # 优化器设置
    p.add_argument("--optimizer-type", default="adamw", choices=["adamw", "adamw8bit", "prodigyplus"], help="优化器类型")
    p.add_argument("--prodigyplus-d0", type=float, default=1e-6, help="ProdigyPlus 初始 d 估计值")
    p.add_argument("--prodigyplus-use-stableadamw", action="store_true", default=True, help="ProdigyPlus 是否使用 StableAdamW")

    # 依赖和交互
    p.add_argument("--auto-install", action="store_true", help="自动安装缺失依赖")
    p.add_argument("--interactive", action="store_true", help="交互模式，提示输入缺失参数")
    p.add_argument("--use-t5-token-weights", action="store_true", default=True)
    p.add_argument("--no-t5-token-weights", dest="use_t5_token_weights", action="store_false")
    p.add_argument("--flow-shift", type=float, default=3.0, help="logit/timestep shift used by shifted timestep samplers")
    p.add_argument("--timestep-sampling", default="logit_normal",
                   choices=["logit_normal", "uniform", "logit_normal_low", "mode", "mixed_uniform_low", "mixed_uniform_logit"],
                   help="timestep sampling distribution")
    p.add_argument("--timestep-mix-low-prob", type=float, default=0.25, help="mixed_uniform_low 中低噪声样本比例")
    p.add_argument("--adaptive-timestep", action="store_true",
                   help="启用保守自适应 timestep：按 per-timestep raw loss 重采样，不改变 loss 权重。")
    p.add_argument("--adaptive-timestep-metric", choices=["raw", "highfreq", "mixed"], default="raw",
                   help="自适应 timestep 的统计信号：raw / highfreq / mixed。")
    p.add_argument("--adaptive-timestep-highfreq-weight", type=float, default=0.25,
                   help="adaptive_timestep_metric=mixed 时的高频 residual 权重。")
    p.add_argument("--adaptive-timestep-bins", type=int, default=16, help="自适应 timestep loss 统计分桶数")
    p.add_argument("--adaptive-timestep-ema-decay", type=float, default=0.95, help="自适应 timestep loss EMA 衰减")
    p.add_argument("--adaptive-timestep-burn-in", type=int, default=160, help="开始重采样前的 warmup step 数")
    p.add_argument("--adaptive-timestep-min-factor", type=float, default=0.5, help="分桶采样倍率下限")
    p.add_argument("--adaptive-timestep-max-factor", type=float, default=2.0, help="分桶采样倍率上限")
    p.add_argument("--adaptive-timestep-base-mix", type=float, default=0.25, help="每个 batch 保留基础采样的比例")
    p.add_argument("--adaptive-timestep-candidate-mult", type=int, default=8, help="proposal-resampling 候选倍数")
    p.add_argument("--schedule-shift", type=float, default=1.0,
                   help="SD3 式 σ schedule shift（应用于所有 t 在噪声混合前）。1.0=不偏移；"
                        "1024 高分辨率训练 SD3 论文推荐 3.0；与 timestep_sampling 模式无关，对 uniform 也生效。")
    p.add_argument("--loss-type", default="mse", choices=["mse", "l2", "l1", "huber", "smooth_l1"], help="训练损失类型")
    p.add_argument("--huber-c", type=float, default=0.1, help="Huber/SmoothL1 切换阈值")
    p.add_argument("--huber-schedule", default="constant", choices=["constant", "snr", "sigma"], help="Huber 阈值随 timestep 的调度")
    p.add_argument("--loss-weighting-scheme", default="none",
                   choices=["none", "min_snr", "max_snr_inv", "logit_normal", "sigma_sqrt", "sigma_sqrt_sd3", "detail_inv_t", "cosmap"],
                   help="per-sample loss weighting scheme")
    p.add_argument("--min-snr-gamma", type=float, default=0.0, help="gamma used by min_snr/max_snr_inv weighting")
    p.add_argument("--weight-cap-ratio", type=float, default=5.0,
                   help="loss 加权时单 batch 内 max/min 比上限。0=禁用；推荐 5-10（小 batch + Prodigy）。"
                        "防止 detail_inv_t / sigma_sqrt_sd3 等激进权重让单样本主导 batch loss → 破坏 Prodigy 的 d 估计。")
    p.add_argument("--noise-offset-min", type=float, default=0.0, help="随机 noise_offset 的下限；仅 random_strength=true 时生效")

    p.add_argument("--noise-offset", type=float, default=0.0, help="low-frequency noise offset strength")
    p.add_argument("--noise-offset-random-strength", action="store_true", help="randomize noise_offset strength per sample")
    p.add_argument("--pyramid-noise-iterations", type=int, default=0, help="number of multires/pyramid noise levels")
    p.add_argument("--pyramid-noise-discount", type=float, default=0.3, help="pyramid noise decay per level")
    p.add_argument("--caption-dropout-rate", type=float, default=0.0, help="drop whole captions with this probability")

    return p.parse_args()


# ============================================================================
# 交互模式辅助函数
# ============================================================================

def _try_rich():
    try:
        from rich.prompt import Prompt, Confirm
        return Prompt, Confirm
    except Exception:
        return None, None


def _ask_str(label, default=""):
    Prompt, _ = _try_rich()
    if Prompt:
        return Prompt.ask(label, default=default) if default else Prompt.ask(label)
    raw = input(f"{label}{f' [{default}]' if default else ''}: ").strip()
    return raw or default


def _ask_bool(label, default=False):
    _, Confirm = _try_rich()
    if Confirm:
        return Confirm.ask(label, default=default)
    raw = input(f"{label} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes", "1", "true", "t")


def _ask_int(label, default):
    while True:
        raw = _ask_str(label, str(default))
        try:
            return int(raw)
        except ValueError:
            print("Please enter an integer.")


def _ask_float(label, default):
    while True:
        raw = _ask_str(label, str(default))
        try:
            return float(raw)
        except ValueError:
            print("Please enter a number.")


def _guess_default_paths():
    base = Path(__file__).resolve().parent
    transformer = base / "anima" / "diffusion_models" / "anima-preview.safetensors"
    vae = base / "anima" / "vae" / "qwen_image_vae.safetensors"
    qwen = base / "anima" / "text_encoders"
    return {
        "transformer": str(transformer) if transformer.exists() else "",
        "vae": str(vae) if vae.exists() else "",
        "qwen": str(qwen) if qwen.exists() else "",
    }


def prompt_for_args(args):
    """交互式提示输入缺失参数"""
    defaults = _guess_default_paths()
    args.data_dir = args.data_dir or _ask_str("数据集目录 (images + .txt)", "")
    args.transformer = args.transformer or _ask_str("Transformer 路径 (.safetensors)", defaults["transformer"])
    args.vae = args.vae or _ask_str("VAE 路径 (.safetensors)", defaults["vae"])
    args.qwen = args.qwen or _ask_str("Qwen 模型目录", defaults["qwen"])
    args.output_dir = _ask_str("输出目录", args.output_dir)
    args.output_name = _ask_str("输出名称", args.output_name)
    args.resolution = _ask_int("分辨率", args.resolution)
    args.batch_size = _ask_int("Batch size", args.batch_size)
    args.grad_accum = _ask_int("梯度累积", args.grad_accum)
    args.lr = _ask_float("学习率", args.lr)
    args.repeats = _ask_int("数据集重复次数", args.repeats)
    args.grad_checkpoint = _ask_bool("启用梯度检查点?", args.grad_checkpoint)
    args.epochs = _ask_int("Epochs", args.epochs)
    args.max_steps = _ask_int("最大步数 (0=无限制)", args.max_steps)
    args.lora_rank = _ask_int("LoRA rank", args.lora_rank)
    args.lora_alpha = _ask_float("LoRA alpha", args.lora_alpha)
    args.loss_curve_steps = _ask_int("Loss 曲线步数 (0=禁用)", args.loss_curve_steps)
    args.auto_install = _ask_bool("自动安装缺失依赖?", args.auto_install)
    args.save_every_epoch = _ask_bool("每个 epoch 保存?", args.save_every_epoch)
    args.mixed_precision = _ask_str("混合精度 (bf16/fp32)", args.mixed_precision)
    return args


# ============================================================================
# 主函数
# ============================================================================

def main():
    args = parse_args()

    # 加载 YAML 配置文件
    config_path = None
    config_dir = None
    if args.config:
        logger.info(f"加载配置文件: {args.config}")
        config_path = Path(args.config).resolve()
        config_dir = config_path.parent
        config = load_yaml_config(args.config)
        args = apply_yaml_config(args, config)

    # 处理 --no-prefer-json 参数
    if getattr(args, "no_prefer_json", False):
        args.prefer_json = False
    elif not hasattr(args, "prefer_json"):
        args.prefer_json = True  # 默认启用

    # 交互模式检查
    required = [args.data_dir, args.transformer, args.vae, args.qwen]
    if args.interactive or any(not x for x in required):
        args = prompt_for_args(args)

    # 依赖检测
    ensure_dependencies(auto_install=args.auto_install)

    # 延迟导入
    import numpy as np
    from PIL import Image

    # 设置随机种子
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ★ TF32 优化：在 Ampere/Hopper (A100/H100/RTX 30+/40+) GPU 上对 fp32 矩阵乘法
    # 启用 TensorFloat-32，单乘 ~8× 加速 fp32 路径，bf16 路径不受影响。
    # 训练里 fp32 残留路径主要在 loss 计算和某些 reduce 上，开 high 是安全的零代价收益。
    # 旧 GPU（V100、Turing）这个调用是 no-op，不会出错。
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        # SDPA 是 PyTorch 2.x 内置的注意力实现，会在 Flash / memory-efficient / math
        # 三个 backend 中自动选最优。Anima/Cosmos 的 attention 算子通过 F.scaled_dot_product_attention
        # 调用它，所以这里不需要手动启用 xformers；过往版本 `--xformers` 开关在 Anima 上
        # 实际上从未生效（钩子名不匹配），已经移除。
        logger.info("Attention backend: PyTorch SDPA (auto-selects flash/memory-efficient/math)")
    # 注意：cudnn.benchmark 这里**故意不开启**。
    # 理由：ARB 分桶导致 batch 之间 conv shape 在多个 bucket 之间切换，
    # 每个新 shape 都触发 ~1-5 秒的算法 profiling。短训练（几百-几千步）下
    # profiling 开销难以摊销，可能反而变慢；只有 shape 完全固定且训练很长才推荐开。

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(exist_ok=True)

    # 启动训练监控面板
    monitor_server = None
    if not getattr(args, "no_monitor", False):
        try:
            from train_monitor import start_monitor_server, update_monitor
            monitor_server = start_monitor_server(
                host=getattr(args, "monitor_host", "127.0.0.1"),
                port=int(getattr(args, "monitor_port", 8765) or 8765),
                output_dir=output_dir,
                open_browser=(not getattr(args, "no_browser", False)),
            )
            update_monitor(config={
                "model": "Anima LoKr" if args.lora_type == "lokr" else "Anima LoRA",
                "rank": args.lora_rank,
                "alpha": args.lora_alpha,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "lr": args.lr,
                "resolution": args.resolution,
                "data_dir": str(args.data_dir),
            })
        except Exception as e:
            logger.warning(f"监控面板启动失败: {e}")

    # 查找模型代码
    repo_root = find_diffusion_pipe_root()
    logger.info(f"模型代码路径: {repo_root}")

    # 解析路径：相对路径优先按 config 位置 / AnimaLoraToolkit 目录解析
    script_dir = Path(__file__).resolve().parent
    bases = [
        Path.cwd(),
        config_dir,
        config_dir.parent if config_dir else None,
        script_dir,
        script_dir.parent,
        repo_root,
        repo_root.parent,
    ]
    args.transformer = resolve_path_best_effort(args.transformer, bases)
    args.vae = resolve_path_best_effort(args.vae, bases)
    args.qwen = resolve_path_best_effort(args.qwen, bases)
    args.t5_tokenizer = resolve_path_best_effort(getattr(args, "t5_tokenizer", ""), bases)
    args.data_dir = resolve_path_best_effort(args.data_dir, bases)
    reg_data_dir = getattr(args, "reg_data_dir", "") or ""
    if reg_data_dir:
        args.reg_data_dir = resolve_path_best_effort(reg_data_dir, bases)
    args.resume_lora = resolve_path_best_effort(getattr(args, "resume_lora", ""), bases)
    args.resume_state = resolve_path_best_effort(getattr(args, "resume_state", ""), bases)
    normalize_resume_paths(args, output_dir)

    # 加载模型
    logger.info("加载 Transformer...")
    model = load_anima_model(args.transformer, device, dtype, repo_root)

    logger.info("加载 VAE...")
    vae = load_vae(args.vae, device, dtype, repo_root)

    logger.info("加载文本编码器...")
    qwen_model, qwen_tok, t5_tok = load_text_encoders(
        args.qwen, args.t5_tokenizer, device, dtype
    )

    # 注入 LoRA
    lora_variant = str(getattr(args, "lora_variant", "base") or "base").lower()
    dora_export_mode = str(getattr(args, "dora_export_mode", "native") or "native").lower()
    if lora_variant == "dora" and args.lora_type != "lokr":
        raise ValueError("lora_variant='dora' requires lora_type='lokr'")
    logger.info(f"注入 {args.lora_type.upper()} ({lora_variant})...")
    # exclude_prefixes 语义：
    #   - None / 未设置  → 用 DEFAULT_EXCLUDE_PREFIXES（默认排除 llm_adapter.*，与 Anima 官方建议一致）
    #   - 字符串/列表    → 完全替换默认值（例如 [] 表示一个不排除）
    raw_exclude = getattr(args, "lora_exclude_prefixes", None)
    raw_exclude_patterns = getattr(args, "lora_exclude_patterns", None)
    raw_include_patterns = getattr(args, "lora_include_patterns", None)

    # 兼容旧参数：exclude_prefixes → regex 转换在 LoRAInjector.__init__ 内处理
    injector_kwargs = {}
    if raw_exclude_patterns is not None:
        injector_kwargs["exclude_patterns"] = list(raw_exclude_patterns)
    elif raw_exclude is not None:
        if isinstance(raw_exclude, str):
            raw_exclude = [s.strip() for s in raw_exclude.split(",") if s.strip()]
        injector_kwargs["exclude_prefixes"] = tuple(raw_exclude)
    if raw_include_patterns is not None:
        injector_kwargs["include_patterns"] = list(raw_include_patterns)

    raw_targets = getattr(args, "lora_targets", None)
    if raw_targets is not None:
        if isinstance(raw_targets, str):
            raw_targets = [s.strip() for s in raw_targets.split(",") if s.strip()]
        else:
            raw_targets = [str(s).strip() for s in raw_targets if str(s).strip()]
        if raw_targets:
            injector_kwargs["targets"] = raw_targets
            logger.info("LoRA targets: %s", ", ".join(raw_targets))

    # 模块级 rank/lr 控制
    reg_dims = getattr(args, "lora_reg_dims", None)
    reg_alphas = getattr(args, "lora_reg_alphas", None)
    reg_lrs = getattr(args, "lora_reg_lrs", None)
    if reg_dims:
        injector_kwargs["reg_dims"] = dict(reg_dims)
    if reg_alphas:
        injector_kwargs["reg_alphas"] = dict(reg_alphas)
    if reg_lrs:
        injector_kwargs["reg_lrs"] = dict(reg_lrs)

    injector = LoRAInjector(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=float(getattr(args, "lora_dropout", 0.0) or 0.0),
        use_lokr=(args.lora_type == "lokr"),
        factor=args.lokr_factor,
        rank_dropout=float(getattr(args, "rank_dropout", 0.0) or 0.0),
        module_dropout=float(getattr(args, "module_dropout", 0.0) or 0.0),
        loraplus_lr_ratio=float(getattr(args, "loraplus_lr_ratio", 1.0) or 1.0),
        lora_variant=lora_variant,
        dora_export_mode=dora_export_mode,
        **injector_kwargs,
    )
    injector.inject(model)
    
    # 从已有 LoRA 继续训练
    if getattr(args, "resume_lora", "") and Path(args.resume_lora).exists():
        injector.load(args.resume_lora)
        logger.info(f"将从已有 LoRA 继续训练: {args.resume_lora}")

    # 数据集
    bucket_mgr = BucketManager(args.resolution)
    base_dataset = ImageDataset(
        args.data_dir, args.resolution, bucket_mgr,
        shuffle_caption=args.shuffle_caption,
        keep_tokens=args.keep_tokens,
        flip_augment=args.flip_augment,
        tag_dropout=args.tag_dropout,
        prefer_json=args.prefer_json,
        freq_balanced_dropout_strength=float(getattr(args, "freq_balanced_dropout_strength", 0.0) or 0.0),
    )
    dataset = base_dataset

    # 正则数据集（Kohya 风格，防过拟合）
    reg_data_dir = getattr(args, "reg_data_dir", "") or ""
    reg_dataset = None
    if reg_data_dir:
        if not Path(reg_data_dir).exists():
            logger.warning(f"正则数据集路径不存在，已跳过: {reg_data_dir}")
        elif len(base_dataset) == 0:
            logger.warning("主数据集为空，正则集已跳过")
        else:
            reg_caption = (getattr(args, "reg_caption", "") or "").strip()
            reg_repeats = max(1, int(getattr(args, "reg_repeats", 1)) or 1)
            reg_base = ImageDataset(
                reg_data_dir, args.resolution, bucket_mgr,
                shuffle_caption=args.shuffle_caption,
                keep_tokens=args.keep_tokens,
                flip_augment=args.flip_augment,
                tag_dropout=0.0,  # 正则集通常不用 dropout
                prefer_json=args.prefer_json,
                caption_override=reg_caption if reg_caption else None,
                freq_balanced_dropout_strength=0.0,  # 正则集不参与频率均衡
            )
            reg_dataset = reg_base
            cap_preview = f", caption=\"{reg_caption[:50]}{'...' if len(reg_caption) > 50 else ''}\"" if reg_caption else ""
            logger.info(f"正则数据集: {reg_data_dir} ({len(reg_base)} 张, repeats={reg_repeats}){cap_preview}")

    # 缓存 VAE latents（在 repeat 之前）
    use_cached = getattr(args, "cache_latents", False)
    if use_cached and bool(getattr(args, "flip_augment", False)):
        # flip_augment 在 ImageDataset.__getitem__ 中作用于像素图像；启用 cache_latents 后，
        # 每张图只在首次缓存时调用一次 __getitem__，是否 flip 在那一刻被随机决定并冻结。
        # 后续每个 epoch 永远拿到同一份 latent，flip 不再随机 → 增强等同于"50% 数据集预 flip"，
        # 失去逐 epoch 增广的本意。VAE encoder 非 flip-equivariant，也不能在 latent 上后补 flip。
        # 此处只警告，不强行覆盖用户配置。
        logger.warning(
            "[dataset] cache_latents=True 与 flip_augment=True 同时开启："
            "flip 仅在首次 latent 缓存时一次性生效，后续 epoch 不再随机翻转。"
            "若想让 flip 在每个 epoch 随机，请关闭 cache_latents；"
            "若想保持 cache_latents 的速度，请关闭 flip_augment。"
        )
    if use_cached:
        dataset = CachedLatentDataset(dataset, vae, device, dtype)
    if reg_dataset is not None and use_cached:
        reg_dataset = CachedLatentDataset(reg_dataset, vae, device, dtype)

    # repeat 放在缓存之后
    if args.repeats > 1:
        dataset = RepeatDataset(dataset, repeats=args.repeats)
    if reg_dataset is not None:
        reg_repeats = max(1, int(getattr(args, "reg_repeats", 1)) or 1)
        reg_dataset = RepeatDataset(reg_dataset, repeats=reg_repeats)
        dataset = MergedDataset(dataset, reg_dataset)

    if args.num_workers > 0 and os.name == "nt":
        logger.warning("num_workers > 0 在 Windows 上容易崩溃：已强制设为 0（避免多进程 spawn 问题）")
        args.num_workers = 0

    # num_workers>0 时启用 worker 持久化与适度预取，省每 epoch 的 spawn 开销
    _loader_kwargs = {}
    if args.num_workers > 0:
        _loader_kwargs["persistent_workers"] = True
        _loader_kwargs["prefetch_factor"] = 2

    if use_cached:
        batch_sampler = BucketBatchSampler(
            dataset, batch_size=args.batch_size,
            drop_last=True, shuffle=True,
            seed=getattr(args, "seed", 42),
        )
        dataloader = DataLoader(
            dataset, batch_sampler=batch_sampler,
            collate_fn=collate_fn_cached,
            num_workers=args.num_workers,
            **_loader_kwargs,
        )
    else:
        # Use BucketBatchSampler so same-resolution images are always batched together.
        # Without this, torch.stack fails when ARB produces tensors of different shapes.
        batch_sampler = BucketBatchSampler(
            dataset, batch_size=args.batch_size,
            drop_last=True, shuffle=True,
            seed=getattr(args, "seed", 42),
        )
        dataloader = DataLoader(
            dataset, batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            **_loader_kwargs,
        )

    # 训练前自检：VAE encode->decode 循环（快速排除 VAE/scale/shape 问题）
    try:
        if len(base_dataset) > 0:
            item0 = base_dataset[0]
            pixels0 = item0["pixel_values"].unsqueeze(0).to(device, dtype=dtype)  # [1,3,H,W]
            with torch.no_grad():
                z0 = vae.model.encode(pixels0.unsqueeze(2), vae.scale)   # [1,16,1,h,w]
                recon0 = vae.model.decode(z0, vae.scale).squeeze(2)      # [1,3,H,W]
                recon0 = (recon0.clamp(-1, 1) + 1) / 2
            arr0 = (recon0[0].permute(1, 2, 0).detach().cpu().float().numpy() * 255).clip(0, 255).astype("uint8")
            Image.fromarray(arr0).save(sample_dir / "vae_roundtrip.png")
            logger.info("VAE roundtrip 自检已保存: samples/vae_roundtrip.png")
    except Exception as e:
        logger.warning(f"VAE roundtrip 自检失败（若 sample 仍是噪点，请优先修这个）: {e}")

    # 优化器
    weight_decay = float(getattr(args, "weight_decay", 0.01) or 0.0)
    opt_type = getattr(args, "optimizer_type", "adamw")
    
    # 针对 ProdigyPlus 的学习率建议
    if opt_type == "prodigyplus" and args.lr != 1.0:
        logger.warning(f"检测到正在使用 ProdigyPlus 优化器，但学习率为 {args.lr}。建议将学习率设为 1.0 以获得最佳自适应效果。")

    # 获取参数组（支持 LoRA+、模块级 lr）
    param_groups = injector.get_param_groups(
        weight_decay,
        base_lr=args.lr,
        loraplus_lr_ratio=float(getattr(args, "loraplus_lr_ratio", 1.0) or 1.0),
    )

    # ── 透明路由 ──────────────────────────────────────────────────────────
    # YAML 中 `optimizer_args:` 下的所有 key 直接以 **kwargs 注入优化器。
    # 未被优化器签名识别的 key 会被 optimizer_utils 自动过滤并 warning（不会报错）。
    raw_opt_args = getattr(args, "optimizer_args", None) or {}
    if not isinstance(raw_opt_args, dict):
        logger.warning(f"optimizer_args 不是字典，已忽略: {type(raw_opt_args).__name__}")
        raw_opt_args = {}
    opt_args = dict(raw_opt_args)  # 拷贝，避免污染 args

    # YAML 中 list/tuple 互转：betas 这类参数惯例为 tuple
    for k in ("betas",):
        if k in opt_args and isinstance(opt_args[k], list):
            opt_args[k] = tuple(opt_args[k])

    # 顶层 weight_decay 作为兜底（若 optimizer_args 中没显式给）
    opt_args.setdefault("weight_decay", weight_decay)

    # 兼容旧字段（让旧 yaml 仍可工作）：prodigyplus_d0 / prodigyplus_use_stableadamw
    legacy_d0 = getattr(args, "prodigyplus_d0", None)
    if opt_type == "prodigyplus" and "d0" not in opt_args and legacy_d0 not in (None, 1e-6):
        opt_args["d0"] = legacy_d0
    legacy_sa = getattr(args, "prodigyplus_use_stableadamw", None)
    if opt_type == "prodigyplus" and "use_stableadamw" not in opt_args and legacy_sa is not None and legacy_sa is not True:
        # 默认 True，仅当显式改成 False 才透传（避免覆盖默认值）
        opt_args["use_stableadamw"] = legacy_sa

    if opt_args:
        logger.info(f"[optimizer_args] passthrough keys: {sorted(opt_args.keys())}")

    optimizer = create_optimizer(
        optimizer_type=opt_type,
        params=param_groups,
        learning_rate=args.lr,
        **opt_args,
    )
    # 打印优化器详细信息（确保用户知道当前用的是哪一个）
    opt_info = get_optimizer_info(optimizer)
    logger.info(f"优化器创建成功: {opt_info['type']}")
    logger.info(f"优化器配置详情: {opt_info}")

    if weight_decay > 0:
        wd_info = f"Optimizer: {opt_type}, weight_decay={weight_decay}"
        if injector.use_lokr:
            wd_info += "（w1 排除 weight_decay）"
        logger.info(wd_info)
    
    grad_clip = float(getattr(args, "grad_clip_max_norm", 0) or 0)
    if grad_clip > 0:
        logger.info(f"梯度裁剪 max_norm={grad_clip}")
    
    # 获取可训练参数用于梯度裁剪
    trainable_params = []
    for group in optimizer.param_groups:
        trainable_params.extend(group["params"])

    # 计算总步数
    try:
        steps_per_epoch = len(dataloader) // args.grad_accum
    except Exception:
        steps_per_epoch = None

    if args.max_steps and args.max_steps > 0:
        total_steps = args.max_steps
    elif steps_per_epoch is not None:
        total_steps = steps_per_epoch * args.epochs
    else:
        total_steps = None

    logger.info(f"数据集大小: {len(dataset)}, 每 epoch 步数: {steps_per_epoch}, 总步数: {total_steps}")

    # 学习率调度器
    scheduler = None
    lr_sched = getattr(args, "lr_scheduler", "none") or "none"
    
    # 如果是 ProdigyPlus 且启用了 schedule-free，则不使用调度器；
    # 关掉 schedule-free 时允许并推荐使用 cosine 等调度器，让后期 LR 衰减带来精修。
    if opt_type == "prodigyplus":
        sf_enabled = bool(opt_args.get("use_schedulefree", True))
        if sf_enabled and lr_sched != "none":
            logger.warning("ProdigyPlus (Schedule-Free) 不需要学习率调度器，已将其设为 none")
            lr_sched = "none"
        elif not sf_enabled and lr_sched == "none":
            logger.warning("ProdigyPlus 关闭 Schedule-Free 时建议配 cosine 调度器，否则后期没有 LR 衰减")
    
    if lr_sched == "cosine":
        eta_min = float(getattr(args, "lr_scheduler_eta_min", 0.0) or 0.0)
        if total_steps is None:
            logger.warning("cosine 调度器需要已知 total_steps，回退到 none")
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps, eta_min=eta_min
            )
            logger.info(f"学习率调度: cosine (T_max={total_steps}, eta_min={eta_min})")
    elif lr_sched == "cosine_with_restart":
        t0 = int(getattr(args, "lr_scheduler_t0", 500) or 500)
        t_mult = int(getattr(args, "lr_scheduler_t_mult", 2) or 2)
        eta_min = float(getattr(args, "lr_scheduler_eta_min", 0.0) or 0.0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=t0, T_mult=t_mult, eta_min=eta_min
        )
        logger.info(f"学习率调度: cosine_with_restart (T_0={t0}, T_mult={t_mult}, eta_min={eta_min})")

    # 初始化进度显示
    progress, task_id, progress_kind = init_progress(not args.no_progress, total_steps)
    use_rich = progress_kind == "rich"
    use_plain = progress == "plain"
    live = None
    loss_history = []
    speed_ema = None

    if use_rich:
        try:
            from rich.console import Group
            from rich.live import Live
            curve_panel = None
            if args.loss_curve_steps > 0 and not args.no_live_curve:
                curve_panel = render_curve_panel([], width=min(60, args.loss_curve_steps), height=10)
            group = Group(progress, curve_panel) if curve_panel is not None else Group(progress)
            live = Live(group, refresh_per_second=10)
            live.start()
        except Exception:
            live = None
            progress.start()

    def emit(msg):
        if use_plain:
            print()
        if live:
            live.console.print(msg)
        elif use_rich:
            progress.console.print(msg)
        else:
            print(msg)

    # 训练循环
    global_step = 0
    start_epoch = 0
    
    # 从训练状态恢复（断点续训）
    if getattr(args, "resume_state", "") and Path(args.resume_state).exists():
        start_epoch, global_step, loss_history, saved_monitor_state = load_training_state(
            args.resume_state, injector, optimizer, scheduler
        )
        emit(f"从断点恢复训练: epoch={start_epoch}, step={global_step}")
        
        # 恢复监控面板的历史数据（loss 曲线等）
        if monitor_server and saved_monitor_state:
            try:
                from train_monitor import restore_monitor_state
                restore_monitor_state(
                    losses=saved_monitor_state.get("losses"),
                    lr_history=saved_monitor_state.get("lr_history"),
                    epoch=start_epoch,
                    step=global_step,
                    total_steps=total_steps,
                )
                emit(f"监控面板历史数据已恢复: {len(saved_monitor_state.get('losses', []))} 个 loss 点")
            except Exception as e:
                emit(f"监控数据恢复失败: {e}")
    
    # Ctrl+C 信号处理：保存状态后退出
    interrupted = False
    current_epoch = start_epoch
    def signal_handler(sig, frame):
        nonlocal interrupted
        if interrupted:
            emit("强制退出...")
            sys.exit(1)
        interrupted = True
        emit("\n检测到 Ctrl+C，正在保存训练状态...")
        state_path = output_dir / f"training_state_step{global_step}.pt"
        if hasattr(optimizer, "eval") and global_step > 0: optimizer.eval()
        # 获取监控面板数据用于恢复 loss 曲线
        monitor_data = None
        if monitor_server:
            try:
                from train_monitor import get_state
                monitor_data = get_state()
            except Exception:
                pass
        save_training_state(state_path, injector, optimizer, current_epoch, global_step, loss_history, monitor_state=monitor_data, scheduler=scheduler)
        # 同时保存 LoRA 权重
        lora_path = output_dir / f"{args.output_name}_interrupted_step{global_step}.safetensors"
        injector.save(lora_path, model=model)
        emit(f"已保存！下次使用 --resume-state \"{state_path}\" 继续训练")
        sys.exit(0)
    
    import signal
    signal.signal(signal.SIGINT, signal_handler)
    
    # 确保优化器在训练模式
    if hasattr(optimizer, "train"):
        optimizer.train()
        
    model.train()
    step_start_time = time.perf_counter()

    # 设置采样提示词列表（支持多角色轮换）
    sample_prompts = getattr(args, "sample_prompts", []) or []
    if not sample_prompts and args.sample_prompt:
        sample_prompts = [args.sample_prompt]
    sample_prompt_idx = 0

    adaptive_ts = AdaptiveTimestepSampler(
        enabled=bool(getattr(args, "adaptive_timestep", False)),
        bins=int(getattr(args, "adaptive_timestep_bins", 16) or 16),
        ema_decay=float(getattr(args, "adaptive_timestep_ema_decay", 0.95) or 0.95),
        burn_in_steps=int(getattr(args, "adaptive_timestep_burn_in", 160) or 0),
        min_factor=float(getattr(args, "adaptive_timestep_min_factor", 0.5) or 0.5),
        max_factor=float(getattr(args, "adaptive_timestep_max_factor", 2.0) or 2.0),
        base_mix=float(getattr(args, "adaptive_timestep_base_mix", 0.25) or 0.0),
        candidate_mult=int(getattr(args, "adaptive_timestep_candidate_mult", 8) or 8),
        metric=str(getattr(args, "adaptive_timestep_metric", "raw") or "raw"),
        highfreq_weight=float(getattr(args, "adaptive_timestep_highfreq_weight", 0.25) or 0.0),
    )
    if adaptive_ts.enabled:
        logger.info("[adaptive_timestep] enabled: %s", adaptive_ts.summary())
    objective_cfg = build_training_objective_config(args)
    logger.info(
        "[objective] timestep=%s flow_shift=%.3f schedule_shift=%.3f mix_low=%.3f "
        "noise_offset=%.4f pyramid=%d discount=%.3f loss=%s huber=%s/%.3f weight=%s cap=%.3f",
        objective_cfg.timestep.mode,
        objective_cfg.timestep.flow_shift,
        objective_cfg.timestep.schedule_shift,
        objective_cfg.timestep.mix_low_prob,
        objective_cfg.noise.offset,
        objective_cfg.noise.pyramid_iterations,
        objective_cfg.noise.pyramid_discount,
        objective_cfg.loss.loss_type,
        objective_cfg.loss.huber_schedule,
        objective_cfg.loss.huber_c,
        objective_cfg.loss.weighting_scheme,
        objective_cfg.loss.weight_cap_ratio,
    )

    def get_next_sample_prompt():
        """获取下一个采样提示词（轮换）"""
        nonlocal sample_prompt_idx
        if not sample_prompts:
            return "1girl, masterpiece"
        prompt = sample_prompts[sample_prompt_idx % len(sample_prompts)]
        sample_prompt_idx += 1
        return prompt

    # Step 0 初始采样（基线效果，测试所有提示词）
    # 只在新训练时执行（global_step == 0），resume 时跳过
    sampling_enabled = args.sample_steps > 0 or args.sample_every > 0
    if global_step == 0 and sampling_enabled:
        emit("采样中 (step 0, 基线)...")
        model.eval()
            
        s_w = int(getattr(args, "sample_width", 0) or 0) or int(args.resolution)
        s_h = int(getattr(args, "sample_height", 0) or 0) or int(args.resolution)
        s_cfg = float(getattr(args, "sample_cfg_scale", 4.0) or 4.0)
        s_neg = str(getattr(args, "sample_negative_prompt", "") or "")
        s_seed = int(getattr(args, "sample_seed", 0) or 0)
        s_steps = int(getattr(args, "sample_infer_steps", 25) or 25)
        s_sampler = str(getattr(args, "sample_sampler_name", "er_sde") or "er_sde")
        s_sched = str(getattr(args, "sample_scheduler", "simple") or "simple")
        for i, prompt in enumerate(sample_prompts[:3]):  # 最多测试 3 个
            if s_seed:
                torch.manual_seed(s_seed + i)
            img = sample_image(
                model, vae, qwen_model, qwen_tok, t5_tok,
                prompt, height=s_h, width=s_w, steps=s_steps, cfg_scale=s_cfg,
                negative_prompt=(s_neg or None),
                sampler_name=s_sampler,
                scheduler=s_sched,
                device=device, dtype=dtype,
                use_t5_token_weights=bool(getattr(args, "use_t5_token_weights", True)),
            )
            sample_path = sample_dir / f"step_0_baseline_{i}.png"
            img.save(sample_path)
            emit(f"基线采样保存: step_0_baseline_{i}.png")
            if monitor_server:
                try:
                    update_monitor(sample_path=sample_path)
                except Exception:
                    pass
                    
        model.train()
    elif global_step > 0 and sampling_enabled:
        emit(f"跳过启动基线采样（从 step {global_step} 恢复，非 step 0）")

    # ★ Schedule-Free 需要显式进入 train 模式
    if hasattr(optimizer, "train"):
        optimizer.train()
    model.train()

    # 累积周期状态：当周期内任一 micro-batch 出现 NaN 时置 False。
    # 旧实现是出 NaN 立即 `zero_grad()` —— 会抹掉同周期内之前已经累计好的梯度，
    # 接着继续累计剩下的 micro-batch，最后用"半截"梯度调用 optimizer.step()。
    # 新做法：保留已累计的梯度，但记号本周期"脏了"，到周期边界时整体丢弃这次 step。
    accum_clean = True

    for epoch in range(start_epoch, args.epochs):
        current_epoch = epoch
        if hasattr(dataloader, "batch_sampler") and hasattr(dataloader.batch_sampler, "set_epoch"):
            dataloader.batch_sampler.set_epoch(epoch)
        for batch_idx, batch in enumerate(dataloader):
            # 在累积周期开始时记录时间 + 重置 clean 标志
            if batch_idx % args.grad_accum == 0:
                step_start_time = time.perf_counter()
                accum_clean = True

            captions = batch["captions"]

            # caption dropout：随机把 caption 替换为空字符串，提升 CFG 服从度（Anima 主要靠 CFG 出图）
            cap_drop_p = float(getattr(args, "caption_dropout_rate", 0.0) or 0.0)
            if cap_drop_p > 0:
                captions = ["" if random.random() < cap_drop_p else c for c in captions]

            # 获取 latents（缓存模式或实时编码）
            if use_cached:
                latents = batch["latents"].to(device, dtype=dtype)
            else:
                pixels = batch["pixel_values"].to(device, dtype=dtype)
                with torch.no_grad():
                    pixels_5d = pixels.unsqueeze(2)
                    # VAE 权重已是 bf16；与 _build_cache / roundtrip 自检保持一致，
                    # 不再强转 fp32（否则 conv3d 会因 input/bias dtype 不匹配而崩）。
                    latents = vae.model.encode(pixels_5d, vae.scale).to(dtype)
            bs = latents.shape[0]

            # 文本编码
            with torch.no_grad():
                # 参考指南/ComfyUI：Qwen 通道不传权重；T5 通道提供 token 权重
                qwen_texts = [_build_qwen_text_from_prompt(c) for c in captions]
                qwen_emb, qwen_attn = encode_qwen(qwen_model, qwen_tok, qwen_texts, device)
                t5_ids, t5_attn, t5_w = tokenize_t5_weighted(t5_tok, captions, max_length=512)
                t5_ids = t5_ids.to(device)
                t5_attn = t5_attn.to(device)
                t5_w = t5_w.to(device, dtype=torch.float32)
                cross = model.preprocess_text_embeds(qwen_emb, t5_ids, t5_attn, qwen_attn)
                if (
                    getattr(args, "use_t5_token_weights", True)
                    and getattr(model, "llm_adapter", None) is not None
                    and cross.shape[1] == t5_w.shape[1]
                ):
                    cross = cross * t5_w.to(cross.dtype).unsqueeze(-1)
                if cross.shape[1] < 512:
                    cross = F.pad(cross, (0, 0, 0, 512 - cross.shape[1]))

            # Flow Matching：t 采样、噪声生成、目标计算
            ts_mode = objective_cfg.timestep.mode
            f_shift = objective_cfg.timestep.flow_shift
            mix_low_prob = objective_cfg.timestep.mix_low_prob
            sched_shift = objective_cfg.timestep.schedule_shift
            t = adaptive_ts.sample(
                bs, device, mode=ts_mode, shift=f_shift,
                mix_low_prob=mix_low_prob, schedule_shift=sched_shift,
                global_step=global_step,
            )

            # SD3 式 σ schedule shift：作用于所有模式的 t（含 uniform / mixed_*），
            # 把噪声混合用的 sigma 整体偏向高噪声端。1.0=禁用，向后兼容。
            t = apply_timestep_schedule_shift(t, sched_shift)

            t_exp = t.view(-1, 1, 1, 1, 1)

            noise = make_noise_from_config(latents, objective_cfg.noise)

            noisy = (1 - t_exp) * latents + t_exp * noise
            target = noise - latents

            # 前向
            pad_mask = torch.zeros(bs, 1, latents.shape[-2], latents.shape[-1], device=device, dtype=dtype)
            with torch.autocast("cuda", dtype=dtype):
                pred = forward_with_optional_checkpoint(
                    model, noisy, t.view(-1, 1), cross, pad_mask,
                    use_checkpoint=args.grad_checkpoint
                )
                # ★ 损失始终 fp32 计算（per-sample，便于按 t 加权）
                per_sample = per_sample_loss(
                    pred,
                    target,
                    loss_type=objective_cfg.loss.loss_type,
                    huber_c=objective_cfg.loss.huber_c,
                    huber_schedule=objective_cfg.loss.huber_schedule,
                    t=t.float(),
                )

                loss = apply_loss_weighting(per_sample, t, objective_cfg.loss)

            # ★ 守护 1：forward 结果 NaN/Inf 检查
            debug_n = int(getattr(args, "debug_first_batches", 0) or 0)
            if debug_n > 0 and global_step < debug_n:
                try:
                    batch_images = batch.get("images", [])
                    preview = ", ".join(str(p) for p in batch_images[:4] if p)
                    logger.info(
                        "[debug step %s] loss=%.6f t(mean/min/max)=%.4f/%.4f/%.4f "
                        "latent(mean/std)=%.4f/%.4f noise(mean/std)=%.4f/%.4f "
                        "target(mean/std)=%.4f/%.4f pred(mean/std)=%.4f/%.4f "
                        "ts=%s shift=%.3f sched_shift=%.3f mix_low=%.3f "
                        "noise_offset=%.4f offset_min=%.4f "
                        "pyramid=%s loss_type=%s huber_c=%.4f loss_weight=%s "
                        "cap_drop=%.4f tag_drop=%.4f lora_drop=%.4f t5_weight=%s images=[%s]",
                        global_step,
                        float(loss.detach().cpu()),
                        float(t.float().mean().detach().cpu()),
                        float(t.float().min().detach().cpu()),
                        float(t.float().max().detach().cpu()),
                        float(latents.float().mean().detach().cpu()),
                        float(latents.float().std().detach().cpu()),
                        float(noise.float().mean().detach().cpu()),
                        float(noise.float().std().detach().cpu()),
                        float(target.float().mean().detach().cpu()),
                        float(target.float().std().detach().cpu()),
                        float(pred.float().mean().detach().cpu()),
                        float(pred.float().std().detach().cpu()),
                        ts_mode,
                        f_shift,
                        sched_shift,
                        mix_low_prob,
                        objective_cfg.noise.offset,
                        objective_cfg.noise.offset_min,
                        objective_cfg.noise.pyramid_iterations,
                        objective_cfg.loss.loss_type,
                        objective_cfg.loss.huber_c,
                        objective_cfg.loss.weighting_scheme,
                        float(getattr(args, "caption_dropout_rate", 0.0) or 0.0),
                        float(getattr(args, "tag_dropout", 0.0) or 0.0),
                        float(getattr(args, "lora_dropout", 0.0) or 0.0),
                        bool(getattr(args, "use_t5_token_weights", True)),
                        preview,
                    )
                except Exception as _debug_e:
                    logger.warning(f"debug_first_batches logging failed: {_debug_e}")

            if not torch.isfinite(loss):
                logger.warning(
                    f"[step {global_step}] Non-finite loss detected ({loss.item()}), "
                    f"skipping this micro-batch. "
                    f"pred stats: min={pred.float().min().item():.3e} "
                    f"max={pred.float().max().item():.3e}"
                )
                # 不要 zero_grad！保留同周期内其他 micro-batch 的梯度，整周期边界统一丢弃。
                accum_clean = False
                continue

            if adaptive_ts.enabled:
                adaptive_signal = adaptive_timestep_metric_signal(
                    per_sample,
                    pred,
                    target,
                    metric=adaptive_ts.metric,
                    highfreq_weight=adaptive_ts.highfreq_weight,
                )
                adaptive_ts.update(t.float(), adaptive_signal)
            loss_to_backward = loss / args.grad_accum
            loss_to_backward.backward()

            if (batch_idx + 1) % args.grad_accum == 0:
                # ★ 守护 1：周期内有 micro-batch NaN/Inf loss → 整周期作废，不做 step
                if not accum_clean:
                    logger.warning(
                        f"[step {global_step}] Accumulation cycle contained a non-finite "
                        f"micro-batch loss; discarding the entire cycle's gradients."
                    )
                    optimizer.zero_grad(set_to_none=True)
                    continue

                # ★ 守护 2：梯度 NaN/Inf 检查（即使 loss 全 finite，反向也可能出 NaN）
                bad_grad = False
                for p in trainable_params:
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        bad_grad = True
                        break
                if bad_grad:
                    batch_images = batch.get("images", [])
                    preview = ", ".join(str(p) for p in batch_images[:4] if p)
                    suffix = f" batch_images=[{preview}]" if preview else ""
                    logger.warning(f"[step {global_step}] Non-finite gradient, skipping update.{suffix}")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                # 梯度裁剪：grad_clip > 0 时启用，==0 表示用户显式关闭（推荐对 ProdigyPlus）。
                # ProdigyPlus 官方建议：use_stableadamw=True 时其内部已处理梯度归一化，
                # 外部裁剪会干扰 d 估计。AdamW 系优化器若想用裁剪，再把这个值设为 1.0。
                grad_norm_before = None
                grad_norm_log_every = int(getattr(args, "grad_norm_log_every", 0) or 0)
                should_log_grad_norm = grad_norm_log_every > 0 and (global_step + 1) % grad_norm_log_every == 0
                if should_log_grad_norm:
                    grad_norm_before = compute_grad_norm(trainable_params)
                if grad_clip > 0:
                    clipped_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=grad_clip)
                    if should_log_grad_norm:
                        clipped_norm_f = float(clipped_norm)
                        grad_norm_after = min(clipped_norm_f, grad_clip) if math.isfinite(clipped_norm_f) else float("inf")
                        logger.info(
                            "[step %s] grad_norm before=%.6f after<=%.6f clip=%.3f clipped=%s",
                            global_step + 1,
                            float(grad_norm_before),
                            float(grad_norm_after),
                            grad_clip,
                            bool(clipped_norm_f > grad_clip),
                        )
                elif should_log_grad_norm:
                    logger.info(
                        "[step %s] grad_norm=%.6f clip=disabled",
                        global_step + 1,
                        float(grad_norm_before),
                    )



                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                # ★ 守护 3：优化器状态污染检测（Prodigy 内部 d 变 NaN 会连锁崩溃）
                if opt_type == "prodigyplus" and global_step % 50 == 0:
                    try:
                        from utils.optimizer_utils import is_optimizer_state_healthy
                        if not is_optimizer_state_healthy(optimizer):
                            logger.error(
                                f"[step {global_step}] Optimizer state contaminated by NaN/Inf. "
                                f"Training cannot continue. Consider reloading from last checkpoint."
                            )
                            raise RuntimeError("Optimizer state NaN")
                    except ImportError:
                        pass

                # 记录 loss 历史（环形缓冲：始终保留最近 N 步）
                loss_val = float(loss.item() * args.grad_accum)
                if args.loss_curve_steps and args.loss_curve_steps > 0:
                    loss_history.append(loss_val)
                    if len(loss_history) > args.loss_curve_steps:
                        del loss_history[: len(loss_history) - args.loss_curve_steps]

                # 更新进度显示
                now = time.perf_counter()
                if opt_type == "prodigyplus" and optimizer.param_groups:
                    g = optimizer.param_groups[0]
                    # effective_lr 是 v2.0 新增的 logging 字段，d * effective_lr 近似真实 LR
                    d_val = g.get("d", 1.0)
                    eff_lr = g.get("effective_lr", g.get("lr", 1.0))
                    lr = float(d_val) * float(eff_lr)
                    # 每 50 步把 d / eff_lr / 实际 LR 分开打印一次，便于诊断 Prodigy 是否找到合理 LR
                    if global_step % 50 == 0:
                        logger.info(
                            "[step %d] prodigy d=%.3e effective_lr=%.3e real_lr=%.3e",
                            global_step, float(d_val), float(eff_lr), lr
                        )
                else:
                    lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else 0.0
                if adaptive_ts.enabled and global_step % 50 == 0:
                    logger.info("[step %d] adaptive_timestep %s", global_step, adaptive_ts.summary())
                
                # 更新训练监控面板
                if monitor_server:
                    try:
                        update_monitor(
                            loss=loss_val, lr=lr, epoch=epoch+1, step=global_step,
                            total_steps=total_steps, speed=speed_ema or 0
                        )
                    except Exception:
                        pass
                dt_step = now - step_start_time
                steps_per_sec = (1.0 / dt_step) if dt_step > 0 else 0.0
                speed_ema = steps_per_sec if speed_ema is None else (0.9 * speed_ema + 0.1 * steps_per_sec)

                if use_rich:
                    desc = f"epoch {epoch+1}/{args.epochs} step {global_step}/{total_steps or '?'}"
                    progress.update(task_id, advance=1, description=desc,
                                    loss=loss_val, lr=float(lr), speed=float(speed_ema or 0))
                    if live and args.loss_curve_steps > 0 and not args.no_live_curve:
                        panel = render_curve_panel(loss_history, width=min(60, args.loss_curve_steps), height=10)
                        if panel is not None:
                            from rich.console import Group
                            live.update(Group(progress, panel))
                elif use_plain:
                    print(f"epoch {epoch+1}/{args.epochs} step {global_step} loss={loss_val:.6f} lr={lr:.2e} speed={speed_ema:.2f} it/s", end="\r", flush=True)
                elif args.log_every and global_step % args.log_every == 0:
                    print(f"epoch={epoch} step={global_step} loss={loss_val:.6f} lr={lr:.2e} speed={steps_per_sec:.2f} it/s")

                # 按 step 采样（轮换提示词）
                if args.sample_steps > 0 and global_step % args.sample_steps == 0:
                    prompt = get_next_sample_prompt()
                    prompt_short = prompt[:50] + "..." if len(prompt) > 50 else prompt
                    emit(f"采样中 (step {global_step}): {prompt_short}")
                    model.eval()
                    if hasattr(optimizer, "eval"): optimizer.eval()
                    s_w = int(getattr(args, "sample_width", 0) or 0) or int(args.resolution)
                    s_h = int(getattr(args, "sample_height", 0) or 0) or int(args.resolution)
                    s_cfg = float(getattr(args, "sample_cfg_scale", 4.0) or 4.0)
                    s_neg = str(getattr(args, "sample_negative_prompt", "") or "")
                    s_steps = int(getattr(args, "sample_infer_steps", 25) or 25)
                    s_sampler = str(getattr(args, "sample_sampler_name", "er_sde") or "er_sde")
                    s_sched = str(getattr(args, "sample_scheduler", "simple") or "simple")
                    img = sample_image(
                        model, vae, qwen_model, qwen_tok, t5_tok,
                        prompt, height=s_h, width=s_w, steps=s_steps, cfg_scale=s_cfg,
                        negative_prompt=(s_neg or None),
                        sampler_name=s_sampler,
                        scheduler=s_sched,
                        device=device, dtype=dtype,
                        use_t5_token_weights=bool(getattr(args, "use_t5_token_weights", True)),
                    )
                    sample_path = sample_dir / f"step_{global_step}.png"
                    img.save(sample_path)
                    emit(f"采样保存: step_{global_step}.png")
                    if monitor_server:
                        try:
                            update_monitor(sample_path=sample_path)
                        except Exception:
                            pass
                    if hasattr(optimizer, "train"): optimizer.train()
                    model.train()

                # 定期保存 LoRA 权重（按 step）
                save_every_steps = getattr(args, "save_every_steps", 0)
                if save_every_steps > 0 and global_step % save_every_steps == 0:
                    if hasattr(optimizer, "eval"): optimizer.eval()
                    lora_path = output_dir / f"{args.output_name}_step{global_step}.safetensors"
                    injector.save(lora_path, model=model)
                    emit(f"Saved LoRA: {lora_path}")
                    if hasattr(optimizer, "train"): optimizer.train()

                # 定期保存训练状态（断点续训）
                save_state_every = getattr(args, "save_state_every", 0)
                if save_state_every > 0 and global_step % save_state_every == 0:
                    if hasattr(optimizer, "eval"): optimizer.eval()
                    state_path = output_dir / f"training_state_step{global_step}.pt"
                    # 获取监控面板数据用于恢复 loss 曲线
                    monitor_data = None
                    if monitor_server:
                        try:
                            from train_monitor import get_state
                            monitor_data = get_state()
                        except Exception:
                            pass
                    save_training_state(state_path, injector, optimizer, epoch, global_step, loss_history, monitor_state=monitor_data, scheduler=scheduler)
                    # 同时保存 LoRA 权重
                    lora_path = output_dir / f"{args.output_name}_step{global_step}.safetensors"
                    injector.save(lora_path, model=model)
                    if hasattr(optimizer, "train"): optimizer.train()

                # 检查 max_steps
                if args.max_steps and global_step >= args.max_steps:
                    break

        # epoch 结束后的操作
        current_epoch = epoch + 1
        if not args.max_steps or global_step < args.max_steps:
            # 保存 checkpoint
            if args.save_every > 0 and current_epoch % args.save_every == 0:
                if hasattr(optimizer, "eval"): optimizer.eval()
                save_path = output_dir / f"{args.output_name}_epoch{current_epoch}.safetensors"
                injector.save(save_path, model=model)
                emit(f"Saved LoRA: {save_path}")
                if hasattr(optimizer, "train"): optimizer.train()

            # 采样（轮换提示词）
            if args.sample_every > 0 and current_epoch % args.sample_every == 0:
                prompt = get_next_sample_prompt()
                prompt_short = prompt[:50] + "..." if len(prompt) > 50 else prompt
                emit(f"采样中 (epoch {current_epoch}): {prompt_short}")
                model.eval()
                if hasattr(optimizer, "eval"): optimizer.eval()
                s_w = int(getattr(args, "sample_width", 0) or 0) or int(args.resolution)
                s_h = int(getattr(args, "sample_height", 0) or 0) or int(args.resolution)
                s_cfg = float(getattr(args, "sample_cfg_scale", 4.0) or 4.0)
                s_neg = str(getattr(args, "sample_negative_prompt", "") or "")
                s_steps = int(getattr(args, "sample_infer_steps", 25) or 25)
                s_sampler = str(getattr(args, "sample_sampler_name", "er_sde") or "er_sde")
                s_sched = str(getattr(args, "sample_scheduler", "simple") or "simple")
                img = sample_image(
                    model, vae, qwen_model, qwen_tok, t5_tok,
                    prompt, height=s_h, width=s_w, steps=s_steps, cfg_scale=s_cfg,
                    negative_prompt=(s_neg or None),
                    sampler_name=s_sampler,
                    scheduler=s_sched,
                    device=device, dtype=dtype,
                    use_t5_token_weights=bool(getattr(args, "use_t5_token_weights", True)),
                )
                sample_path = sample_dir / f"epoch_{current_epoch}.png"
                img.save(sample_path)
                emit(f"采样保存: epoch_{current_epoch}.png")
                if hasattr(optimizer, "train"): optimizer.train()
                model.train()
                
                # 更新监控面板
                if monitor_server:
                    try:
                        update_monitor(sample_path=sample_path)
                    except Exception:
                        pass

        # 检查 max_steps
        if args.max_steps and global_step >= args.max_steps:
            break

    # 最终保存
    if hasattr(optimizer, "eval"): optimizer.eval()
    final_path = output_dir / f"{args.output_name}.safetensors"
    injector.save(final_path, model=model)

    # 清理进度显示
    if live:
        live.stop()
    elif use_rich:
        progress.stop()

    # 显示最终 loss 曲线
    if args.loss_curve_steps and loss_history:
        chart = render_loss_curve(loss_history, width=min(80, len(loss_history)), height=10)
        emit(f"Loss curve (first {len(loss_history)} steps):\n{chart}")

    emit(f"Saved final LoRA: {final_path}")
    logger.info("训练完成!")


if __name__ == "__main__":
    main()
