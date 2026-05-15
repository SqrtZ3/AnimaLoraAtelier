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
from trainer.data import (
    BucketManager,
    ImageDataset,
    RepeatDataset,
    MergedDataset,
    BucketBatchSampler,
    CachedLatentDataset,
    collate_fn,
    collate_fn_cached,
)
from trainer.checkpoint import (
    _strip_prefixes,
    _pick_best_prefix_remap,
    _load_safetensors_state_dict,
    resolve_path_best_effort,
    normalize_resume_paths,
    _load_weights_best_effort,
    save_training_state,
    load_training_state,
)
from trainer.models import (
    find_diffusion_pipe_root,
    load_module_from_path,
    ensure_models_namespace,
    load_anima_model,
    load_vae,
    load_text_encoders,
)
from trainer.config import (
    load_yaml_config,
    apply_yaml_config,
)


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

    # 风格预设：在 YAML 顶层设 `style_profile: hazy` 或命令行 `--style-profile hazy`
    # 会按预设覆盖一组相关参数（loss_weighting_scheme / detail_inv_t_* /
    # timestep_sampling / mix_low_prob / adaptive_*）。详见 trainer/config.py
    # 的 STYLE_PROFILES。YAML 里显式写的字段优先级 > 预设。
    p.add_argument("--style-profile", default="",
                   choices=["", "sharp", "hazy", "balanced"],
                   help="一行切换画风预设：sharp / hazy / balanced（详见 STYLE_PROFILES）")
    p.add_argument("--detail-inv-t-min", type=float, default=1.0,
                   help="detail_inv_t 加权下限（默认 1.0）")
    p.add_argument("--detail-inv-t-max", type=float, default=5.0,
                   help="detail_inv_t 加权上限（默认 5.0；hazy 画风可降到 ~3.0）")

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
