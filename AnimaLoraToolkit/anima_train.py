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


def summarize_bad_gradients(named_parameters, limit=8):
    summaries = []
    for name, p in named_parameters:
        grad = p.grad
        if grad is None:
            continue
        finite = torch.isfinite(grad)
        if bool(finite.all()):
            continue
        with torch.no_grad():
            nan_count = int(torch.isnan(grad).sum().item())
            posinf_count = int(torch.isposinf(grad).sum().item())
            neginf_count = int(torch.isneginf(grad).sum().item())
            finite_vals = grad.detach().float()[finite]
            finite_abs_max = (
                float(finite_vals.abs().max().item()) if finite_vals.numel() > 0 else float("nan")
            )
        summaries.append(
            f"{name}: shape={tuple(grad.shape)} nan={nan_count} "
            f"+inf={posinf_count} -inf={neginf_count} finite_abs_max={finite_abs_max:.3e}"
        )
        if len(summaries) >= int(limit):
            break
    return summaries


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
    sample_t_stratified,
    apply_timestep_schedule_shift,
    apply_t_range,
    anneal_mix_prob,
    AdaptiveTimestepSampler,
    make_noise,
    make_noise_from_config,
    _huber_delta_for_t,
    per_sample_loss,
    lwd_saliency_mask,
    vecor_contrastive_neg,
    LossBinEMA,
    contrastive_flow_matching_neg,
    per_sample_highfreq_loss,
    adaptive_timestep_metric_signal,
    compute_loss_weight,
    apply_loss_weighting,
    apply_loss_weighting_per_sample,
    compute_grad_norm,
    forward_with_optional_checkpoint,
    forward_packed_with_optional_checkpoint,
    masked_token_loss,
    validate_compile_requirements,
)
from trainer.lora import LoRALayer, LoKrLayer, LoRALinear, LoRAInjector
from trainer.gaf import GafController
from trainer.aux_losses import (
    build_aux_loss_config,
    recover_x0_from_velocity,
    spectral_loss,
    spectral_loss_per_sample,
    PerceptualLossModule,
    summary_aux_loss_config,
)
from trainer.data import (
    BucketManager,
    ImageDataset,
    RepeatDataset,
    MergedDataset,
    BucketBatchSampler,
    FitTokenBatchSampler,
    CachedLatentDataset,
    compute_sample_accumulation_steps,
    collate_fn,
    collate_fn_fit_packed,
    collate_fn_cached,
    collate_fn_cached_fit,
)
from trainer.progress import (
    ReferenceStepTracker,
    reference_interval_crossed,
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
    p.add_argument("--effective-batch-size", type=int, default=0,
                   help="按真实图片数累积到该数量后再 optimizer.step()。0=使用旧的 grad_accum 逻辑。")
    p.add_argument("--reference-batch-size", type=int, default=0,
                   help="用旧 batch size 模拟分桶 batch，换算旧 optimizer step。0=禁用 reference step。")
    p.add_argument("--reference-grad-accum", type=int, default=1,
                   help="reference step 使用的旧 grad_accum。默认 1。")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-scheduler", default="none", choices=["none", "cosine", "cosine_with_restart"], help="学习率调度器")
    p.add_argument("--lr-scheduler-t0", type=int, default=500, help="cosine_with_restart: 首次 restart 周期 (step)")
    p.add_argument("--lr-scheduler-t-mult", type=float, default=2.0, help="cosine_with_restart: 每次 restart 周期倍数")
    p.add_argument("--lr-scheduler-eta-min", type=float, default=0.0, help="cosine/cosine_with_restart: 最小学习率")
    p.add_argument("--weight-decay", type=float, default=0.01, help="AdamW 权重衰减 (L2 正则, 0=禁用)")
    p.add_argument("--grad-clip-max-norm", type=float, default=1.0, help="梯度裁剪最大范数 (0=禁用；ProdigyPlus 推荐设 0)")
    p.add_argument("--resolution", type=int, default=1024,
                   help="ARB 分桶的 base 边长（桶围着 base² 面积 ±10% 造）")
    p.add_argument("--bucket-base-resos", default=None,
                   help="多级 ARB base 边长，逗号分隔，如 512,768,1024。显式设置时优先于自动 base 范围。")
    p.add_argument("--bucket-min-base-reso", type=int, default=0,
                   help="自动生成多级 base 的起点；0=使用 min_bucket_reso。仅 bucket_base_resos 为空且 max>0 时生效。")
    p.add_argument("--bucket-max-base-reso", type=int, default=0,
                   help="自动生成多级 base 的终点；0=禁用自动 base 范围，回到单 resolution。")
    p.add_argument("--bucket-base-reso-steps", type=int, default=256,
                   help="自动 base 范围步长，如 512..2048 step 256。")
    p.add_argument("--bucket-no-upscale", action="store_true",
                   help="禁止把原图放大进更大的 bucket；小裁切会落到不超过原图尺寸的最近桶。")
    p.add_argument("--bucket-max-upscale", type=float, default=0.0,
                   help="允许的最大放大倍率；0=不限。与 bucket_no_upscale 同开时 no_upscale 优先。")
    p.add_argument("--bucket-report", action="store_true",
                   help="数据集初始化后打印原图尺寸到 bucket 的分配报告，用于调参。")
    p.add_argument("--min-bucket-reso", type=int, default=512,
                   help="ARB 单维下限（小于此的边长不会作为桶候选；默认 512）")
    p.add_argument("--max-bucket-reso", type=int, default=2048,
                   help="ARB 单维上限（大于此的边长不会作为桶候选；1024 base 默认 2048，"
                        "1536 base 想要全 AR=2.0 支持需要 2240+）")
    p.add_argument("--bucket-reso-steps", type=int, default=64,
                   help="ARB 桶的边长步长（默认 64；VAE 8× + patch 2× 要求是 16 的倍数，64 安全）")
    p.add_argument("--bucket-drop-last", action="store_true",
                   help="ARB 分桶时，每个桶里不足 batch_size 的余数图片是否丢弃。"
                        "默认 False —— 不丢弃，余数桶产生小 batch，保证每张图每 epoch 1 次曝光。"
                        "显式传该 flag 恢复旧行为（丢弃残缺桶）。")
    p.add_argument("--fit-packed-training", action="store_true",
                   help="Enable native-first FiT-style packed-token training; default preserves source pixels.")
    p.add_argument("--fit-max-tokens", type=int, default=65536,
                   help="Maximum FiT tokens per image before applying the over-budget policy.")
    p.add_argument("--fit-warn-tokens", type=int, default=16384,
                   help="Warn when a native FiT image exceeds this token count.")
    p.add_argument("--fit-min-tokens", type=int, default=16,
                   help="Diagnostic minimum token count for native FiT training.")
    p.add_argument("--fit-patch-size", type=int, default=2,
                   help="FiT token patch size; must match model patch_spatial.")
    p.add_argument("--fit-vae-downsample", type=int, default=8,
                   help="VAE spatial downsample factor used to estimate FiT token counts.")
    p.add_argument("--fit-over-budget-strategy", default="fail",
                   choices=["fail", "skip", "resize", "crop", "random_resize_crop"],
                   help="Policy for images above fit_max_tokens; default fail never silently resizes.")
    p.add_argument("--fit-align-mode", default="pad", choices=["pad", "ceil", "floor"],
                   help="How native FiT aligns source images to VAE+patch granularity.")
    p.add_argument("--fit-pack-multiple-images", action="store_true",
                   help="Pack multiple source images into one sequence; experimental and off by default.")
    p.add_argument("--fit-max-tokens-per-batch", type=int, default=0,
                   help="Maximum native FiT tokens per batch; 0 follows fit_max_tokens.")
    p.add_argument("--alpha-handling", default="none", choices=["none", "mask"],
                   help="How to handle transparent source pixels. mask excludes alpha-transparent pixels from FiT supervision.")
    p.add_argument("--alpha-background", default="neutral", choices=["neutral", "white", "black"],
                   help="RGB matte used before VAE encoding when alpha_handling=mask.")
    p.add_argument("--alpha-threshold", type=float, default=0.01,
                   help="Alpha values at or below this normalized threshold are excluded when alpha_handling=mask.")
    p.add_argument("--max-img-h", type=int, default=0,
                   help="RoPE 位置嵌入支持的最大单维（latent 单位 = image / 8）。0=自动从 "
                        "max_bucket_reso 推算并兜底到 240。preview3 训练在 240 = 120 patches，"
                        "1536 base 训练需要 >= 272 (image 2176)，建议 288。")
    p.add_argument("--max-img-w", type=int, default=0, help="同 max_img_h，宽度方向。")
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
    p.add_argument("--cache-latents-dtype", choices=["fp32", "bf16", "fp16"], default="bf16",
                   help="latent 缓存 dtype。默认 bf16 与训练 dtype 对齐，disk 占用减半，"
                        "且省去训练取数时的 fp32→bf16 转换开销。fp32 仅在需要最高精度时用。")
    p.add_argument("--cache-encode-batch-size", type=int, default=8,
                   help="缓存 VAE latent 时单个 micro-batch 的原图张数上限（同尺寸图 stack 一次编码，"
                        "flip 份拼进同批）。越大越快但越吃显存；仍受内部像素预算约束，OOM 会自动降级逐张。"
                        "显存紧张可调小（如 4 / 2 / 1）。")
    p.add_argument("--baseline-sample-count", type=int, default=3,
                   help="step 0 基线采样的提示词数量（最多取 sample_prompts 前 N 个）")
    p.add_argument("--dataloader-pin-memory", action="store_true", default=True,
                   help="DataLoader pin_memory=True，配合 .to(device, non_blocking=True) 让 H2D 拷贝与计算 overlap。"
                        "Windows 上某些场景可能引起问题，可用 --no-dataloader-pin-memory 关闭。")
    p.add_argument("--no-dataloader-pin-memory", dest="dataloader_pin_memory", action="store_false")

    # LoRA 参数
    p.add_argument("--lora-type", choices=["lora", "lokr"], default="lokr")
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=float, default=32.0)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--lokr-factor", type=int, default=8)
    p.add_argument("--lora-variant", choices=["base", "dora", "tlora"], default="base",
                   help="Adapter variant. 'dora' enables LyCORIS/ComfyUI-compatible DoRA-LoKr. "
                        "'tlora' enables timestep-dependent rank mask (arxiv:2507.05964); "
                        "ComfyUI 推理需 bghira/ComfyUI-T-LoRA。")
    # T-LoRA 参数（仅 lora_variant=tlora 时生效）
    p.add_argument("--tlora-rmin-ratio", type=float, default=0.5,
                   help="T-LoRA r_min 占 r 的比例。默认 0.5（论文推荐）。")
    p.add_argument("--tlora-alpha", type=float, default=1.0,
                   help="T-LoRA rank-schedule 幂律指数（alpha=1 即论文线性 schedule）。")
    p.add_argument("--tlora-init", choices=["ortho", "default"], default="ortho",
                   help="T-LoRA 初始化：ortho=SVD-based Ortho-LoRA（论文方案）；default=kaiming+zeros（仅 mask）。")
    p.add_argument("--tlora-lokr-experimental", action="store_true",
                   help="允许 lora_variant=tlora 与 lora_type=lokr 组合（实验性，论文未覆盖；"
                        "保存的 checkpoint 在 ComfyUI 中退化为标准 LoKr）。")
    p.add_argument("--tlora-lokr-ortho-init", action="store_true",
                   help="LoKr+T-LoRA 时启用 (w2_a, w2_b) 上的 Ortho 初始化（启发式扩展）。")
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
    p.add_argument("--sample-reference-steps", type=int, default=0,
                   help="每 N 个 reference step 采样一次。reference step 按旧 batch/grad_accum 分桶模拟。0=禁用")
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
    p.add_argument("--save-every-steps", type=int, default=0, help="每 N 个 optimizer step 保存 LoRA (0=禁用)")
    p.add_argument("--save-every-reference-steps", type=int, default=0,
                   help="每 N 个 reference step 保存 LoRA。reference step 按旧 batch/grad_accum 分桶模拟。0=禁用")
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
    p.add_argument("--keep-vae-on-gpu", action="store_true",
                   help="cache_latents=True 后仍让 VAE 留在 GPU，适合频繁采样，避免反复搬运。")
    p.add_argument("--empty-cache-after-sample", action="store_true", default=True,
                   help="VAE 采样后 offload 回 CPU 时调用 torch.cuda.empty_cache()。默认开启。")
    p.add_argument("--no-empty-cache-after-sample", dest="empty_cache_after_sample", action="store_false",
                   help="VAE 采样后不清空 CUDA allocator cache，适合频繁采样。")

    # 优化器设置
    p.add_argument("--optimizer-type", default="adamw", choices=["adamw", "adamw8bit", "prodigyplus", "soap", "adopt", "lion", "clion", "emosens"], help="优化器类型")
    p.add_argument("--prodigyplus-d0", type=float, default=1e-6, help="ProdigyPlus 初始 d 估计值")
    p.add_argument("--prodigyplus-use-stableadamw", action="store_true", default=True, help="ProdigyPlus 是否使用 StableAdamW")

    # 依赖和交互
    p.add_argument("--auto-install", action="store_true", help="自动安装缺失依赖")
    p.add_argument("--interactive", action="store_true", help="交互模式，提示输入缺失参数")
    p.add_argument("--use-t5-token-weights", action="store_true", default=True)
    p.add_argument("--no-t5-token-weights", dest="use_t5_token_weights", action="store_false")
    p.add_argument("--flow-shift", type=float, default=3.0, help="logit/timestep shift used by shifted timestep samplers")
    p.add_argument("--timestep-sampling", default="logit_normal",
                   choices=["logit_normal", "uniform", "logit_normal_low", "mode", "mixed_uniform_low", "mixed_uniform_logit", "mixed_logit_low_high", "ushaped", "u_shaped", "bimodal", "laplace", "logsnr", "mixed_logsnr_low_high", "ushaped_sf", "mixed_logsnr_three", "three_band"],
                   help="timestep sampling distribution")
    p.add_argument("--timestep-mix-low-prob", type=float, default=0.25, help="mixed_uniform_low 中低噪声样本比例")
    p.add_argument("--timestep-laplace-mu", type=float, default=0.0,
                   help="Laplace 噪声调度（timestep_sampling=laplace）log-SNR 峰位 μ；>0 偏低噪声/细节端，<0 偏高噪。")
    p.add_argument("--timestep-laplace-b", type=float, default=0.5,
                   help="Laplace 噪声调度的尺度 b（越小越集中在 μ 附近）。论文 256→0.5、512→0.75。")
    p.add_argument("--timestep-logsnr-mu", type=float, default=-6.0,
                   help="Style-Friendly logSNR 采样峰位 μ（arXiv 2411.14793）；-6 → t峰≈0.95 画风写入区。"
                        "仅 timestep_sampling=logsnr / mixed_logsnr_low_high 生效。")
    p.add_argument("--timestep-logsnr-sigma", type=float, default=2.0,
                   help="Style-Friendly logSNR 采样宽度 σ；论文 σ<2 损失多样性。")
    p.add_argument("--timestep-t-min", type=float, default=0.0,
                   help="t 下界截断；arXiv 2509.20952 证 t→0 velocity 目标病态，细节端建议 0.05。0=历史行为(1e-4)。")
    p.add_argument("--timestep-t-max", type=float, default=1.0,
                   help="t 上界截断。1.0=历史行为(1-1e-4)。")
    p.add_argument("--timestep-mix-high-prob", type=float, default=0.25,
                   help="mixed_logsnr_three 的高噪(氛围)峰路由概率；低噪用 --timestep-mix-low-prob，"
                        "其余进中噪(结构)峰。v2 教训：中段(0.4-0.85)不能为空，否则形体风格学不到。")
    p.add_argument("--timestep-mix-anneal-start", type=int, default=0,
                   help="三峰路由概率退火的起始 step（线性插值到 anneal-end）。")
    p.add_argument("--timestep-mix-anneal-end", type=int, default=0,
                   help="三峰路由概率退火的结束 step；0=禁用退火。课程式：前结构后细节"
                        "（v4 实证：绑定收紧发生在后段，末段提低噪份额=收尾重开无条件细节表达）。")
    p.add_argument("--timestep-mix-low-prob-end", type=float, default=-1.0,
                   help="退火终点的低噪峰概率；-1=不退火该项。")
    p.add_argument("--timestep-mix-high-prob-end", type=float, default=-1.0,
                   help="退火终点的高噪峰概率；-1=不退火该项。")
    p.add_argument("--dfm-mode", choices=["batch", "vecor"], default="batch",
                   help="ΔFM 负样本来源：batch=同批其它样本（原版）；vecor=对 target 做通道乱序/裁剪缩放"
                        "构造（arXiv 2511.18942 部分移植，不依赖 batch 大小，bs=1 也生效；实验性）。")
    p.add_argument("--eval-every", type=int, default=0,
                   help="每 N 步跑一次固定网格 eval loss（0=关）。固定样本+固定噪声+固定 t 网格 → "
                        "跨 run 可比的确定性曲线，写入 output_dir/eval_loss.csv。")
    p.add_argument("--eval-count", type=int, default=4,
                   help="eval 用的样本数（取数据集前 N 个；不从训练集中剔除——小画风集剔除代价更大，"
                        "曲线含义=拟合度而非泛化）。")
    p.add_argument("--eval-t-grid", default="0.1,0.3,0.5,0.7,0.9",
                   help="eval 的固定 timestep 网格（逗号分隔）。")
    p.add_argument("--eval-seed", type=int, default=1234, help="eval 固定噪声种子（CPU RNG，跨机器一致）。")
    p.add_argument("--tread-enabled", action="store_true",
                   help="TREAD token 路由（arXiv 2501.04765）：训练期让随机 ratio 的 token 绕过中段 blocks，"
                        "省 20-40% 算力，推理完全不变。dense 路径限定。")
    p.add_argument("--tread-ratio", type=float, default=0.3,
                   help="路由段内被绕过的 token 比例。LoRA 微调 >0.35 可能伤收敛（SimpleTuner 经验），"
                        "保守 0.3 起步。")
    p.add_argument("--tread-start-layer", type=int, default=3,
                   help="路由段起始 block 索引（含；支持负索引）。")
    p.add_argument("--tread-end-layer", type=int, default=-4,
                   help="路由段结束 block 索引（不含；支持负索引）。28 blocks 时 3→-4 = 路由 blocks 3..23，"
                        "首 3 尾 4 个 block 全量计算。")
    p.add_argument("--timestep-stratified", action="store_true",
                   help="batch 内 t 分位数分层采样（VDM 低差异思想）：消除小 batch 撞同一噪声段的"
                        "梯度噪声尖峰，对任意采样分布零开销生效。")
    p.add_argument("--lora-one-init-steps", type=int, default=0,
                   help="LoRA-One 谱对齐初始化（arXiv 2502.01235→LoKr KPSVD 版）：训练前累积 N 个 batch "
                        "的全参梯度做 SVD 初始化 LoKr 因子，加速前几百步收敛。0=关闭。"
                        "需要 fit_packed_training=false；建议 8-16。")
    p.add_argument("--lwd-mask-enabled", action="store_true",
                   help="LWD 小波显著性 time-gated 掩码（arXiv 2506.00433）：高细节区在更多 t 段受监督、"
                        "平坦区只在低噪段受监督。零额外开销；dense 路径限定；与 detail_inv_t/频域 aux 建议互斥。")
    p.add_argument("--lwd-mask-floor", type=float, default=0.3,
                   help="LWD 平坦区保底监督下限 ℓ（论文默认 0.3；t<ℓ 时全图受监督）。")
    p.add_argument("--lora-one-init-scale", type=float, default=0.01,
                   help="LoRA-One 初始 ΔW 的 Frobenius 范数相对 ||W0|| 的比例；成品画风 LoKr 通常在 "
                        "1-5%% 量级，0.01=保守。")
    p.add_argument("--adaptive-timestep", action="store_true",
                   help="启用保守自适应 timestep：按 per-timestep raw loss 重采样，不改变 loss 权重。")
    p.add_argument("--adaptive-timestep-metric", choices=["raw", "highfreq", "mixed", "entropy_rate"], default="raw",
                   help="自适应 timestep 的统计信号：raw / highfreq / mixed / entropy_rate "
                        "（entropy_rate = InfoNoise 风格 ρ(t)/w(t)，配合 --adaptive-timestep-low-noise-gate 使用）。")
    p.add_argument("--adaptive-timestep-low-noise-gate", action="store_true",
                   help="InfoNoise 低噪闸门 g(t) = t^n/(t^n + c^n)；entropy_rate 模式专用。")
    p.add_argument("--adaptive-timestep-gate-n", type=float, default=3.0,
                   help="InfoNoise 闸门指数 n（论文默认 3）。")
    p.add_argument("--adaptive-timestep-gate-c", type=float, default=0.05,
                   help="InfoNoise 闸门拐点 c（t 远小于 c 时压制采样）。")
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
    p.add_argument("--huber-snr-clamp-max", type=float, default=10.0,
                   help="huber_schedule=snr 时低 t（细节区）δ 的上限倍率。默认 10.0=旧行为（低噪近纯 L2）；"
                        "调小（如 3）让低噪区保留 L1 鲁棒，抗脏数据集 outlier。")
    p.add_argument("--dfm-lambda", type=float, default=0.0,
                   help="Contrastive Flow Matching（ΔFM, arxiv:2506.05350）排斥项权重。0=关闭；"
                        "论文甜点 0.05，≥0.15 易分布塌缩。反'回归条件均值→发灰发雾'，零额外前向。")
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

    # ── 辅助 loss：Spectral Regularization（arxiv:2603.02447）────────────────
    # FFT 振幅匹配 + 可选 Haar wavelet 系数匹配；纯 tensor 运算，零额外模型。
    p.add_argument("--aux-spectral-enabled", action="store_true",
                   help="启用 spectral regularization（在 latent space 上对 x₀_pred / x₀_target 做 FFT 振幅 L1 匹配）")
    p.add_argument("--aux-spectral-lambda", type=float, default=0.05,
                   help="spectral FFT 振幅 loss 的权重；推荐 0.02-0.1")
    p.add_argument("--aux-spectral-use-wavelet", action="store_true",
                   help="额外开启 Haar wavelet 系数 L1（多尺度结构匹配）")
    p.add_argument("--aux-spectral-wavelet-lambda", type=float, default=0.05,
                   help="wavelet loss 在 spectral 总和中的相对权重")
    p.add_argument("--aux-spectral-t-gate", type=float, default=0.7,
                   help="仅在 t < t_gate 时启用 spectral loss（高 t 区间 x₀_pred 不准，频域无意义）")
    # ── 辅助 loss：Perceptual（arxiv:2602.02493 PixelGen）──────────────────────
    # 通过 VAE decode → pixel → VGG16(LPIPS) + DINOv2-B 计算感知相似度。
    p.add_argument("--aux-perceptual-enabled", action="store_true",
                   help="启用 perceptual loss（LPIPS + DINOv2-B，VAE 解码到 pixel space）")
    p.add_argument("--aux-perceptual-lambda-lpips", type=float, default=0.1,
                   help="LPIPS 权重；PixelGen 论文经典值 0.1")
    p.add_argument("--aux-perceptual-lambda-dino", type=float, default=0.01,
                   help="DINOv2 cosine 距离权重；PixelGen 论文经典值 0.01；0=不加载 DINO")
    p.add_argument("--aux-perceptual-t-gate", type=float, default=0.7,
                   help="仅在 t < t_gate 时启用 perceptual loss")
    p.add_argument("--aux-perceptual-lpips-net", default="vgg",
                   choices=["vgg", "alex", "squeeze"],
                   help="LPIPS 主干网络；vgg 对纹理最敏感，alex 最快，squeeze 最轻量")
    p.add_argument("--aux-perceptual-dino-local-path", default="",
                   help="本地 DINOv2 权重路径（.pth/.safetensors 或 dinov2 仓库目录）；空=走 torch.hub 自动下载")
    p.add_argument("--aux-perceptual-cache-dir", default="",
                   help="本地 perceptual 模型缓存目录（设为 TORCH_HOME）；LPIPS-VGG 与 DINOv2 都从这里读，无需联网。"
                        "目录结构：<cache_dir>/hub/checkpoints/{vgg16-*.pth,dinov2_vitb14_pretrain.pth} + "
                        "<cache_dir>/hub/facebookresearch_dinov2_main/（DINOv2 仓库代码）")
    p.add_argument("--aux-perceptual-use-checkpoint", action="store_true", default=True,
                   help="把 VAE decode + LPIPS + DINO 这条 forward 用 torch.utils.checkpoint 包起来，"
                        "backward 重放一次以释放激活。1024 训练强烈建议开（默认 True）。")
    p.add_argument("--aux-perceptual-no-checkpoint", dest="aux_perceptual_use_checkpoint",
                   action="store_false",
                   help="禁用 perceptual 路径的 checkpoint（仅供调试/显存富余时用）")
    p.add_argument("--aux-perceptual-lpips-size", type=int, default=0,
                   help="LPIPS 之前对 pixel 下采样的目标边长。0=用原分辨率（默认）；512 让显存减半；"
                        "256 再减半，对 1024 训练几乎无质量损失")

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
    # ★ 旧实现 `args.save_every_epoch` 这个属性根本不存在（argparse 里只有 save_every: int）
    # 任何用户跑 --interactive 都会 AttributeError 崩溃。把它映射到 save_every（0=禁用, 1=每 epoch）。
    _save_each_epoch_default = bool(getattr(args, "save_every", 0))
    if _ask_bool("每个 epoch 保存?", _save_each_epoch_default):
        if not args.save_every:
            args.save_every = 1
    else:
        args.save_every = 0
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
    # 旧 GPU（V100、Turing）这些调用是 no-op，不会出错。
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        # 显式开 matmul + cudnn 的 TF32 flag。`high` 已经等价启用 matmul TF32，但
        # PyTorch 在不同小版本里的默认值会漂移；写出来更稳定。cudnn 那个 flag 对 conv 也生效，
        # 本训练里基本不走 conv（仅 VAE encode 阶段），不过开着也无害。
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # SDPA 是 PyTorch 2.x 内置的注意力实现，会在 Flash / memory-efficient / math
        # 三个 backend 中自动选最优。Anima/Cosmos 的 attention 算子通过 F.scaled_dot_product_attention
        # 调用它，所以这里不需要手动启用 xformers；过往版本 `--xformers` 开关在 Anima 上
        # 实际上从未生效（钩子名不匹配），已经移除。
        logger.info("Attention backend: PyTorch SDPA (auto-selects flash/memory-efficient/math)")
    # 注意：cudnn.benchmark 这里**故意不开启**。
    # 理由：ARB 分桶导致 batch 之间 conv shape 在多个 bucket 之间切换，
    # 每个新 shape 都触发 ~1-5 秒的算法 profiling。短训练（几百-几千步）下
    # profiling 开销难以摊销，可能反而变慢；只有 shape 完全固定且训练很长才推荐开。

    # ★ Allocator 提示：训练循环里 ARB 多 bucket 会触发频繁的 cudaMalloc/Free。
    # 设置 expandable_segments=True 让 PyTorch caching allocator 用增长式 segment 而非
    # 每个新 shape 都开新 block；减少 OOM 风险 + 降低分配延迟。
    # 仅在用户没显式设过 PYTORCH_CUDA_ALLOC_CONF 时设置。
    if torch.cuda.is_available() and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        logger.info("Allocator: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (ARB 多 bucket 友好)")

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
                "effective_batch_size": getattr(args, "effective_batch_size", 0),
                "reference_batch_size": getattr(args, "reference_batch_size", 0),
                "reference_grad_accum": getattr(args, "reference_grad_accum", 1),
                "sample_steps": getattr(args, "sample_steps", 0),
                "sample_reference_steps": getattr(args, "sample_reference_steps", 0),
                "save_every_steps": getattr(args, "save_every_steps", 0),
                "save_every_reference_steps": getattr(args, "save_every_reference_steps", 0),
                "keep_vae_on_gpu": getattr(args, "keep_vae_on_gpu", False),
                "empty_cache_after_sample": getattr(args, "empty_cache_after_sample", True),
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
    # 自动确定 max_img_h / max_img_w（RoPE 位置嵌入支持的单维上限）：
    #   - 用户在 YAML / CLI 显式设了 → 用用户给的
    #   - 否则按 max_bucket_reso 推算（latent 单位 = image / 8）
    #   - 历史最小值 240（= 1920 image 单维 = 120 patches）作为兜底
    # ARB 桶单维最大 = max_bucket_reso；要求 max_img_h >= max_bucket_reso / 8。
    # 留 8 个 latent 的余量 + 偶数对齐（patch_spatial=2）。
    _user_max_h = getattr(args, "max_img_h", 0)
    _user_max_w = getattr(args, "max_img_w", 0)
    _bucket_max = int(getattr(args, "max_bucket_reso", 2048) or 2048)
    _auto_max = max(240, ((_bucket_max // 8) + 8 + 1) // 2 * 2)
    max_img_h = int(_user_max_h) if _user_max_h and int(_user_max_h) > 0 else _auto_max
    max_img_w = int(_user_max_w) if _user_max_w and int(_user_max_w) > 0 else _auto_max
    logger.info("加载 Transformer...")
    model = load_anima_model(
        args.transformer, device, dtype, repo_root,
        max_img_h=max_img_h, max_img_w=max_img_w,
    )

    logger.info("加载 VAE...")
    vae = load_vae(args.vae, device, dtype, repo_root)

    logger.info("加载文本编码器...")
    qwen_model, qwen_tok, t5_tok = load_text_encoders(
        args.qwen, args.t5_tokenizer, device, dtype
    )

    # ★ Text encode cache：caption 在训练中是否会变？
    # - shuffle_caption / tag_dropout / caption_dropout_rate / freq_balanced_dropout 任一开启
    #   → caption 每次都不一样，cache 命中率几乎 0，纯浪费内存。
    # - 都关闭 → caption 完全静态，cache 命中率接近 100%，可省 Qwen forward + T5 tokenize 的 overhead。
    _caption_is_static = (
        not bool(getattr(args, "shuffle_caption", False))
        and float(getattr(args, "tag_dropout", 0.0) or 0.0) <= 0
        and float(getattr(args, "caption_dropout_rate", 0.0) or 0.0) <= 0
        and float(getattr(args, "freq_balanced_dropout_strength", 0.0) or 0.0) <= 0
    )
    try:
        from trainer.text_encode import set_text_encode_cache_enabled, reset_text_encode_cache
        reset_text_encode_cache()
        set_text_encode_cache_enabled(_caption_is_static)
        if _caption_is_static:
            logger.info("[text-encode] caption 静态（无 shuffle / dropout），启用编码 LRU cache（命中即跳过 Qwen forward）")
        else:
            logger.info("[text-encode] caption 启用了 shuffle / dropout，禁用编码 cache（命中率会很低）")
    except Exception as _e:
        logger.warning(f"[text-encode] cache toggle 失败（忽略）: {_e}")

    # 注入 LoRA
    lora_variant = str(getattr(args, "lora_variant", "base") or "base").lower()
    dora_export_mode = str(getattr(args, "dora_export_mode", "native") or "native").lower()
    if lora_variant == "dora" and args.lora_type != "lokr":
        raise ValueError("lora_variant='dora' requires lora_type='lokr'")
    # T-LoRA × LoKr 的组合校验交给 LoRAInjector.__init__（带详细错误提示）；
    # 这里只在标准 LoRA 路径下记录一行 info。
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
        tlora_rmin_ratio=float(getattr(args, "tlora_rmin_ratio", 0.5) or 0.5),
        tlora_alpha=float(getattr(args, "tlora_alpha", 1.0) or 1.0),
        tlora_init=str(getattr(args, "tlora_init", "ortho") or "ortho"),
        tlora_lokr_experimental=bool(getattr(args, "tlora_lokr_experimental", False)),
        tlora_lokr_ortho_init=bool(getattr(args, "tlora_lokr_ortho_init", False)),
        tlora_skip_lambda_layer=bool(getattr(args, "tlora_skip_lambda_layer", True)),
        **injector_kwargs,
    )
    injector.inject(model)
    
    # 从已有 LoRA 继续训练
    if getattr(args, "resume_lora", "") and Path(args.resume_lora).exists():
        injector.load(args.resume_lora)
        logger.info(f"将从已有 LoRA 继续训练: {args.resume_lora}")

    # 数据集
    fit_packed_training = bool(getattr(args, "fit_packed_training", False))
    if fit_packed_training:
        if bool(getattr(args, "fit_pack_multiple_images", False)):
            raise RuntimeError("fit_pack_multiple_images is reserved for a later slice; current FiT path uses one image sequence per sample.")
        fit_required_latent_dim = math.ceil(
            int(getattr(args, "fit_max_tokens", 65536) or 65536) ** 0.5
        ) * int(getattr(args, "fit_patch_size", 2) or 2)
        if max_img_h < fit_required_latent_dim or max_img_w < fit_required_latent_dim:
            logger.warning(
                "[FiT] max_img_h/max_img_w=%dx%d latent may be too small for fit_max_tokens=%d. "
                "A 4096x4096 image with patch=2, vae_downsample=8 needs max_img_h/max_img_w >= 512.",
                max_img_h,
                max_img_w,
                int(getattr(args, "fit_max_tokens", 65536) or 65536),
            )
        logger.info(
            "[FiT] native packed-token training enabled: max_tokens=%d warn_tokens=%d align=%s over_budget=%s",
            int(getattr(args, "fit_max_tokens", 65536) or 65536),
            int(getattr(args, "fit_warn_tokens", 16384) or 0),
            str(getattr(args, "fit_align_mode", "pad") or "pad"),
            str(getattr(args, "fit_over_budget_strategy", "fail") or "fail"),
        )

    bucket_min_reso = int(getattr(args, "min_bucket_reso", 512) or 512)
    bucket_max_reso = int(getattr(args, "max_bucket_reso", 2048) or 2048)
    bucket_step = int(getattr(args, "bucket_reso_steps", 64) or 64)
    bucket_base_resos = getattr(args, "bucket_base_resos", None)
    bucket_mgr = BucketManager(
        base_reso=args.resolution,
        min_reso=bucket_min_reso,
        max_reso=bucket_max_reso,
        step=bucket_step,
        base_resos=bucket_base_resos,
        min_base_reso=int(getattr(args, "bucket_min_base_reso", 0) or 0),
        max_base_reso=int(getattr(args, "bucket_max_base_reso", 0) or 0),
        base_reso_step=int(getattr(args, "bucket_base_reso_steps", 256) or 256),
        no_upscale=bool(getattr(args, "bucket_no_upscale", False)),
        max_upscale=float(getattr(args, "bucket_max_upscale", 0.0) or 0.0),
        max_aspect_ratio=float(getattr(args, "bucket_max_aspect_ratio", 2.0) or 2.0),
        token_bucket=bool(getattr(args, "token_bucket", False)),
        token_bucket_counts=getattr(args, "token_bucket_counts", None),
        token_bucket_max_aspect_ratio=float(getattr(args, "token_bucket_max_aspect_ratio", 2.0) or 2.0),
        token_bucket_min_dim=int(getattr(args, "token_bucket_min_dim", 512) or 512),
        token_bucket_max_dim=int(getattr(args, "token_bucket_max_dim", 2016) or 2016),
    )
    logger.info(
        "[BucketManager] bases=%s, min=%d, max=%d, step=%d, max_ar=%.2f, no_upscale=%s, max_upscale=%.3g, 桶数=%d, token_bucket=%s",
        bucket_mgr.base_resos, bucket_min_reso, bucket_max_reso, bucket_step,
        bucket_mgr.max_aspect_ratio, bucket_mgr.no_upscale, bucket_mgr.max_upscale,
        len(bucket_mgr.buckets), getattr(bucket_mgr, "token_bucket", False),
    )

    # 模型 RoPE 能容纳的最大单维（image pixels）= max_img_h * 8。
    # 在训练第一步崩溃之前，提前 fail-fast：把所有超出 RoPE 容量的桶过滤掉，
    # 并在控制台 emit 一个清楚的错误说明 —— 否则用户会看到一个看起来像随机崩溃
    # 的 AssertionError 出现在 forward 里。
    rope_max_image_dim = max_img_h * 8  # max_img_h 在 latent 单位（已经包含 patch_spatial=2 的余量）
    bad_buckets = [] if fit_packed_training else [
        (bw, bh) for (bw, bh) in bucket_mgr.buckets
        if bw > rope_max_image_dim or bh > rope_max_image_dim
    ]
    if bad_buckets:
        good = [b for b in bucket_mgr.buckets if b not in bad_buckets]
        if not good:
            raise RuntimeError(
                f"全部 {len(bucket_mgr.buckets)} 个 ARB 桶都超出了模型 RoPE 单维上限 "
                f"{rope_max_image_dim} image pixels。\n"
                f"解决方案：提高 max_img_h / max_img_w（latent 单位），或者降低 max_bucket_reso。\n"
                f"例如 resolution=1536 + AR=2.0 需要 max_img_h >= 272（image 2176），建议 288。"
            )
        logger.warning(
            "[BucketManager] 检测到 %d 个桶超出 RoPE 单维上限 %d image pixels（max_img_h=%d）："
            "%s ... 已从桶集合中过滤。如需保留这些桶，请提高 YAML 里的 max_img_h / max_img_w。",
            len(bad_buckets), rope_max_image_dim, max_img_h,
            ", ".join(f"{w}x{h}" for w, h in bad_buckets[:5]),
        )
        bucket_mgr.buckets = good
    base_dataset = ImageDataset(
        args.data_dir, args.resolution, None if fit_packed_training else bucket_mgr,
        shuffle_caption=args.shuffle_caption,
        keep_tokens=args.keep_tokens,
        flip_augment=args.flip_augment,
        tag_dropout=args.tag_dropout,
        tag_dropout_overrides=getattr(args, "tag_dropout_overrides", None),
        prefer_json=args.prefer_json,
        freq_balanced_dropout_strength=float(getattr(args, "freq_balanced_dropout_strength", 0.0) or 0.0),
        fit_packed=fit_packed_training,
        fit_max_tokens=int(getattr(args, "fit_max_tokens", 65536) or 65536),
        fit_warn_tokens=int(getattr(args, "fit_warn_tokens", 16384) or 0),
        fit_min_tokens=int(getattr(args, "fit_min_tokens", 16) or 0),
        fit_patch_size=int(getattr(args, "fit_patch_size", 2) or 2),
        fit_vae_downsample=int(getattr(args, "fit_vae_downsample", 8) or 8),
        fit_over_budget_strategy=str(getattr(args, "fit_over_budget_strategy", "fail") or "fail"),
        fit_align_mode=str(getattr(args, "fit_align_mode", "pad") or "pad"),
        alpha_handling=str(getattr(args, "alpha_handling", "none") or "none"),
        alpha_background=str(getattr(args, "alpha_background", "neutral") or "neutral"),
        alpha_threshold=float(getattr(args, "alpha_threshold", 0.01)),
    )
    if bool(getattr(args, "bucket_report", False)):
        logger.info("\n%s", base_dataset.bucket_report(label="train"))
    dataset = base_dataset

    # token_bucket 满覆盖前提的自检：每张图的 token 数都应落在配置的桶集合里。非桶尺寸图
    # 会各自成组（torch_compile 下多一张编译图），并可能与 cache/aux 的满覆盖假设不一致。
    if fit_packed_training and bool(getattr(args, "token_bucket", False)):
        _cfg_counts = set(int(c) for c in (getattr(bucket_mgr, "token_bucket_counts", None) or []))
        _seen_counts = set(int(c) for c in getattr(base_dataset, "token_count_for_index", []) if c)
        _off_counts = sorted(_seen_counts - _cfg_counts)
        if _off_counts and _cfg_counts:
            logger.warning(
                "[token_bucket] 检测到 %d 种不在配置桶集合 %s 内的 token 数：%s。\n"
                "token_bucket 假设数据集已被精确重采样到桶尺寸（满覆盖）。请用数据集工具按相同 "
                "token_bucket_counts 导出，或剔除这些图 —— 否则 compile 会多编译图、且 cache/aux "
                "的满覆盖假设可能不成立。",
                len(_off_counts), sorted(_cfg_counts), _off_counts[:8],
            )

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
                reg_data_dir, args.resolution, None if fit_packed_training else bucket_mgr,
                shuffle_caption=args.shuffle_caption,
                keep_tokens=args.keep_tokens,
                flip_augment=args.flip_augment,
                tag_dropout=0.0,  # 正则集通常不用 dropout
                prefer_json=args.prefer_json,
                caption_override=reg_caption if reg_caption else None,
                fit_packed=fit_packed_training,
                fit_max_tokens=int(getattr(args, "fit_max_tokens", 65536) or 65536),
                fit_warn_tokens=int(getattr(args, "fit_warn_tokens", 16384) or 0),
                fit_min_tokens=int(getattr(args, "fit_min_tokens", 16) or 0),
                fit_patch_size=int(getattr(args, "fit_patch_size", 2) or 2),
                fit_vae_downsample=int(getattr(args, "fit_vae_downsample", 8) or 8),
                fit_over_budget_strategy=str(getattr(args, "fit_over_budget_strategy", "fail") or "fail"),
                fit_align_mode=str(getattr(args, "fit_align_mode", "pad") or "pad"),
                alpha_handling=str(getattr(args, "alpha_handling", "none") or "none"),
                alpha_background=str(getattr(args, "alpha_background", "neutral") or "neutral"),
                alpha_threshold=float(getattr(args, "alpha_threshold", 0.01)),
                freq_balanced_dropout_strength=0.0,  # 正则集不参与频率均衡
            )
            reg_dataset = reg_base
            if bool(getattr(args, "bucket_report", False)):
                logger.info("\n%s", reg_base.bucket_report(label="reg"))
            cap_preview = f", caption=\"{reg_caption[:50]}{'...' if len(reg_caption) > 50 else ''}\"" if reg_caption else ""
            logger.info(f"正则数据集: {reg_data_dir} ({len(reg_base)} 张, repeats={reg_repeats}){cap_preview}")

    # 缓存 VAE latents（在 repeat 之前）
    use_cached = getattr(args, "cache_latents", False)
    _token_bucket = bool(getattr(args, "token_bucket", False))
    if fit_packed_training and use_cached and not _token_bucket:
        raise RuntimeError(
            "fit_packed_training without token_bucket requires cache_latents=false so pixel "
            "masks stay available (variable-size packing needs per-image padding masks). "
            "Enable token_bucket=true (full-coverage exact-grid buckets) to cache latents: the "
            "mask is then all-ones and rebuilt from latent shape."
        )
    if fit_packed_training and use_cached and _token_bucket:
        logger.info(
            "[cache] token_bucket + cache_latents：每张图填满桶 → latent_mask 全 1（从 latent 形状"
            "重建，不存进 npz）。注意：alpha 蒙版不经此缓存路径保留；带 alpha 的数据集请走非缓存 FiT。"
        )
    if use_cached and bool(getattr(args, "flip_augment", False)):
        # cache_latents 与 flip_augment 现已兼容（kohya 风格）：CachedLatentDataset 在缓存阶段
        # 为每张图额外编码一份"像素域水平翻转后再 encode"的 latent（VAE 非 flip-equivariant，
        # 翻转只能发生在像素域、encode 之前），训练时每次 __getitem__ 随机取原图 / 翻转其一，
        # 逐 epoch 随机翻转得以保留。代价：npz 体积 ≈ ×2、首次缓存的 VAE 编码量 ≈ ×2（一次性）。
        logger.info(
            "[dataset] cache_latents=True 且 flip_augment=True：将为每张图缓存原图 + 水平翻转两份 latent，"
            "训练时逐次随机二选一（保留逐 epoch 翻转）。npz 体积与首次编码量约翻倍；"
            "已有的旧缓存（仅含单份 latent）会被自动失效并重新编码一次。"
        )
    if use_cached:
        # cache_latents_dtype 让 npz 保存为 bf16/fp16 而非 fp32：
        #  - disk 占用 ≈ 一半（5120×5120×16ch 的 5D latent，一张 1024² 图约 0.6MB(fp32) → 0.3MB(bf16)）
        #  - 训练取数时 from_numpy(...).to(dtype=bf16) 跳过精度转换
        # fp32 = 旧默认（最高精度但 disk/读盘成本翻倍）；bf16 = 新默认（与训练 dtype 对齐）。
        _cache_dtype_str = str(getattr(args, "cache_latents_dtype", "bf16") or "bf16").lower()
        _cache_dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
        _cache_save_dtype = _cache_dtype_map.get(_cache_dtype_str, torch.bfloat16)
        _cache_encode_bs = int(getattr(args, "cache_encode_batch_size", 8) or 8)
        dataset = CachedLatentDataset(dataset, vae, device, dtype, save_dtype=_cache_save_dtype,
                                      encode_batch_size=_cache_encode_bs)
    if reg_dataset is not None and use_cached:
        reg_dataset = CachedLatentDataset(reg_dataset, vae, device, dtype, save_dtype=_cache_save_dtype,
                                          encode_batch_size=_cache_encode_bs)

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
    # ★ pin_memory + non_blocking 让 H2D 拷贝与下个 batch 计算 overlap。
    # 仅在 CUDA 可用时启用；Windows 上 num_workers=0 通常仍能受益（pinned host 内存自身有用）。
    if torch.cuda.is_available() and bool(getattr(args, "dataloader_pin_memory", True)):
        _loader_kwargs["pin_memory"] = True

    _bucket_drop_last = bool(getattr(args, "bucket_drop_last", False))
    _effective_batch_size = int(getattr(args, "effective_batch_size", 0) or 0)
    _reference_batch_size = int(getattr(args, "reference_batch_size", 0) or 0)
    if _effective_batch_size > 0:
        if _bucket_drop_last:
            logger.warning(
                "effective_batch_size 已启用，建议保持 bucket_drop_last=false；"
                "当前仍会先按 bucket_drop_last 丢弃不满 batch_size 的桶余数。"
            )
        if int(getattr(args, "grad_accum", 1) or 1) != 1:
            logger.warning(
                "effective_batch_size=%d 会接管梯度累积；grad_accum=%d 将被忽略。",
                _effective_batch_size, int(getattr(args, "grad_accum", 1) or 1),
            )
        logger.info(
            "Sample-window accumulation: native batch_size<=%d, effective_batch_size=%d",
            int(args.batch_size), _effective_batch_size,
        )
    if fit_packed_training:
        if int(getattr(args, "effective_batch_size", 0) or 0) > 0:
            raise RuntimeError("fit_packed_training currently uses token-based batches; effective_batch_size sample-window accumulation is not supported yet.")
        if _token_bucket:
            # token_bucket = 满覆盖 + 一组固定的精确桶尺寸。改用 BucketBatchSampler 按精确
            # (h, w) 网格分批 → 每个 batch 单一网格、mask 全 1。这让 cache_latents（latent_mask
            # 从形状重建）、aux（unpatchify 回单一网格）、torch.compile（每网格一张固定图）都干净。
            # 代价：不再按 token 预算混合不同网格打包；在小桶集 + 满覆盖下可忽略。
            batch_sampler = BucketBatchSampler(
                dataset, batch_size=args.batch_size,
                drop_last=_bucket_drop_last, shuffle=True,
                seed=getattr(args, "seed", 42),
            )
            _fit_collate = collate_fn_cached_fit if use_cached else collate_fn_fit_packed
            dataloader = DataLoader(
                dataset, batch_sampler=batch_sampler,
                collate_fn=_fit_collate,
                num_workers=args.num_workers,
                **_loader_kwargs,
            )
        else:
            max_tokens_per_batch = int(getattr(args, "fit_max_tokens_per_batch", 0) or 0)
            if max_tokens_per_batch <= 0:
                max_tokens_per_batch = int(getattr(args, "fit_max_tokens", 65536) or 65536)
            batch_sampler = FitTokenBatchSampler(
                dataset,
                batch_size=args.batch_size,
                max_tokens_per_batch=max_tokens_per_batch,
                shuffle=True,
                seed=getattr(args, "seed", 42),
            )
            dataloader = DataLoader(
                dataset, batch_sampler=batch_sampler,
                collate_fn=collate_fn_fit_packed,
                num_workers=args.num_workers,
                **_loader_kwargs,
            )
    elif use_cached:
        batch_sampler = BucketBatchSampler(
            dataset, batch_size=args.batch_size,
            drop_last=_bucket_drop_last, shuffle=True,
            seed=getattr(args, "seed", 42),
            effective_batch_size=_effective_batch_size,
            reference_batch_size=_reference_batch_size,
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
            drop_last=_bucket_drop_last, shuffle=True,
            seed=getattr(args, "seed", 42),
            effective_batch_size=_effective_batch_size,
            reference_batch_size=_reference_batch_size,
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

    # ── VAE CPU offload（仅在 cache_latents=True 时）────────────────────────
    # latent 缓存建好后，训练主循环再也不调用 vae.model（取 batch 时直接读 npz）。
    # 把 VAE 主网络挪到 CPU 上省 1-2GB 显存；sample_image 用到时再 .to(device)。
    # mean / std 张量小（fp32 16 channels × 2 = 128 bytes 量级），保持在 GPU 上没问题。
    # ★ 例外：若启用了 perceptual loss，训练主循环每步都要 vae.model.decode，offload
    #   到 CPU 会让每步都 swap，巨慢。这种情况下 VAE 必须常驻 GPU。
    aux_cfg = build_aux_loss_config(args)
    if fit_packed_training and aux_cfg.any_enabled and not _token_bucket:
        raise RuntimeError(
            "fit_packed_training without token_bucket supports the main masked token objective "
            "only: aux losses run on a grid x0 [B,C,H,W], which requires the single-grid batches "
            "that token_bucket=true provides. Enable token_bucket, or disable aux losses."
        )
    if fit_packed_training and aux_cfg.any_enabled and _token_bucket:
        logger.info(
            "[aux] token_bucket FiT：packed 预测将 unpatchify 回网格 [B,C,1,h,w] 后送入 "
            "spectral/perceptual（与非 FiT 路径同一套 aux 数学）。"
        )
    validate_compile_requirements(
        bool(getattr(args, "torch_compile", False)),
        fit_packed_training,
        bool(getattr(args, "token_bucket", False)),
        float(getattr(args, "module_dropout", 0.0) or 0.0),
    )
    # module-dropout 路径一次性定档：torch_compile 开 → compile-safe（预抽 keep 标量）；
    # 关 → eager 懒抽签（与最初实现等价，roll/clear 每步短路、零额外开销）。
    injector.set_module_dropout_compile_safe(bool(getattr(args, "torch_compile", False)))
    if bool(getattr(args, "torch_compile", False)):
        _compile_target = model.module if hasattr(model, "module") else model
        _compile_target.compile_blocks(mode=getattr(args, "compile_mode", None))
        logger.info(
            "[torch_compile] per-block token path compiled (compile_mode=%s); "
            "first step will be slow (Inductor warmup).",
            getattr(args, "compile_mode", None),
        )
    vae_offloaded_to_cpu = False
    keep_vae_on_gpu = bool(getattr(args, "keep_vae_on_gpu", False))
    if use_cached and keep_vae_on_gpu:
        logger.info("VAE kept on GPU (keep_vae_on_gpu=True); frequent sampling will not reload VAE.")
    elif use_cached and not aux_cfg.needs_vae_decoder:
        try:
            vae.model = vae.model.cpu()
            torch.cuda.empty_cache()
            vae_offloaded_to_cpu = True
            logger.info("VAE 主网络已 offload 到 CPU（cache_latents=True 后训练循环不再使用 VAE）")
        except Exception as _e:
            logger.warning(f"VAE CPU offload 失败（忽略，继续训练）: {_e}")
    elif use_cached and aux_cfg.needs_vae_decoder:
        logger.info(
            "Perceptual loss enabled → VAE 保留在 GPU 上（训练循环每步要 decode；CPU↔GPU swap 太贵）"
        )

    # ── 辅助 loss 模块（Spectral / Perceptual）────────────────────────────────
    # Spectral 是无状态 fn，按 cfg 调用即可；Perceptual 要预加载 LPIPS + DINO，做成 Module。
    perceptual_module = None
    if aux_cfg.perceptual_enabled:
        try:
            perceptual_module = PerceptualLossModule(
                vae_wrapper=vae,
                cfg=aux_cfg,
                device=device,
                compute_dtype=dtype,
            )
            # ★ 不能在这里 .eval()：会递归把 wrapper.training 设 False，
            #   而 PerceptualLossModule.forward 的 checkpoint 分支条件之一是
            #   self.training=True（已在 aux_losses.py 里去掉了，但稳妥起见这里也不调）。
            #   sub-modules (lpips_fn / dino) 在 __init__ 里已被独立 .eval()，无需父级再调。
            logger.info("PerceptualLossModule 构建完成: %s", summary_aux_loss_config(aux_cfg))
        except Exception as e:
            import dataclasses
            logger.error(
                "PerceptualLossModule 构建失败，自动禁用 perceptual loss（如缺少依赖请运行 `pip install lpips`）：%s",
                e,
            )
            # 把 perceptual 标记关掉，spectral 不受影响
            aux_cfg = dataclasses.replace(aux_cfg, perceptual_enabled=False)
            # ★ 同步禁用 args，否则下面 build_training_objective_config(args).aux 仍会带着
            #    perceptual_enabled=True，训练循环会再次进入 perceptual 分支但 module=None。
            args.aux_perceptual_enabled = False
    elif aux_cfg.spectral_enabled:
        logger.info("辅助 loss 配置: %s", summary_aux_loss_config(aux_cfg))

    # 优化器
    weight_decay = float(getattr(args, "weight_decay", 0.01) or 0.0)
    opt_type = str(getattr(args, "optimizer_type", "adamw") or "adamw").lower()
    
    # 针对 ProdigyPlus 的学习率建议
    if opt_type == "prodigyplus" and args.lr != 1.0:
        logger.warning(f"检测到正在使用 ProdigyPlus 优化器，但学习率为 {args.lr}。建议将学习率设为 1.0 以获得最佳自适应效果。")
    if opt_type == "emosens" and args.lr > 0.3:
        logger.warning(
            "检测到 EmoSens lr=%s。DiT LoRA 建议先从 0.1 起步；过高的 lr_scope 可能让 emoPulse 上限过大。",
            args.lr,
        )

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
    trainable_named_params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]

    # 计算总步数
    sample_accum_enabled = int(getattr(args, "effective_batch_size", 0) or 0) > 0
    effective_batch_size = int(getattr(args, "effective_batch_size", 0) or 0)
    try:
        if sample_accum_enabled:
            steps_per_epoch = compute_sample_accumulation_steps(
                dataset_size=len(dataset),
                epochs=1,
                effective_batch_size=effective_batch_size,
            )
        else:
            steps_per_epoch = (len(dataloader) + max(1, args.grad_accum) - 1) // max(1, args.grad_accum)
    except Exception:
        steps_per_epoch = None

    if args.max_steps and args.max_steps > 0:
        total_steps = args.max_steps
    elif sample_accum_enabled:
        total_steps = compute_sample_accumulation_steps(
            dataset_size=len(dataset),
            epochs=args.epochs,
            effective_batch_size=effective_batch_size,
        )
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
    if opt_type in ("soap_sf", "soapsf") and lr_sched != "none":
        # SOAP-SF 自带 schedule-free 平均，外部 cosine 会改写 group['lr'] 造成双重退火。
        logger.warning("SOAP-SF (Schedule-Free) 不需要学习率调度器，已将其设为 none")
        lr_sched = "none"
    if opt_type == "emosens" and lr_sched != "none":
        logger.warning("EmoSens 内部会根据 loss 序列动态写入学习率，已将外部 lr_scheduler 设为 none")
        lr_sched = "none"
    
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
    samples_seen = 0
    reference_batch_size = int(getattr(args, "reference_batch_size", 0) or 0)
    reference_grad_accum = max(1, int(getattr(args, "reference_grad_accum", 1) or 1))
    reference_tracker = ReferenceStepTracker(grad_accum=reference_grad_accum)
    
    # 从训练状态恢复（断点续训）
    if getattr(args, "resume_state", "") and Path(args.resume_state).exists():
        (
            start_epoch,
            global_step,
            loss_history,
            saved_monitor_state,
            saved_samples_seen,
            saved_reference_state,
        ) = load_training_state(
            args.resume_state, injector, optimizer, scheduler
        )
        if saved_samples_seen is not None:
            samples_seen = int(saved_samples_seen)
        elif sample_accum_enabled:
            samples_seen = int(global_step) * int(effective_batch_size)
        else:
            samples_seen = int(global_step) * int(args.batch_size) * int(args.grad_accum)
        if isinstance(saved_reference_state, dict):
            reference_tracker.step = int(saved_reference_state.get("step", 0) or 0)
            reference_tracker.grad_batches_pending = int(saved_reference_state.get("grad_batches_pending", 0) or 0)
            reference_tracker.grad_accum = int(saved_reference_state.get("grad_accum", reference_grad_accum) or reference_grad_accum)
        emit(f"从断点恢复训练: epoch={start_epoch}, step={global_step}")
        
        # 恢复监控面板的历史数据（loss 曲线等）
        if monitor_server and saved_monitor_state:
            try:
                from train_monitor import restore_monitor_state
                monitor_samples_seen = saved_monitor_state.get("samples_seen")
                if monitor_samples_seen is None:
                    monitor_samples_seen = samples_seen
                monitor_ref_step = saved_monitor_state.get("ref_step")
                if monitor_ref_step is None:
                    monitor_ref_step = reference_tracker.step
                restore_monitor_state(
                    losses=saved_monitor_state.get("losses"),
                    lr_history=saved_monitor_state.get("lr_history"),
                    epoch=start_epoch,
                    step=global_step,
                    total_steps=total_steps,
                    ref_step=monitor_ref_step,
                    samples_seen=monitor_samples_seen,
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
        save_training_state(
            state_path, injector, optimizer, current_epoch, global_step,
            loss_history, monitor_state=monitor_data, scheduler=scheduler,
            samples_seen=samples_seen,
            reference_state={
                "step": reference_tracker.step,
                "grad_batches_pending": reference_tracker.grad_batches_pending,
                "grad_accum": reference_tracker.grad_accum,
            },
        )
        # 同时保存 LoRA 权重
        lora_path = output_dir / f"{args.output_name}_interrupted_step{global_step}.safetensors"
        injector.save(lora_path, model=model)
        emit(f"已保存！下次使用 --resume-state \"{state_path}\" 继续训练")
        # ★ 优雅关闭监控 server。旧实现直接 sys.exit(0)，daemon thread 立刻被强杀，
        # 但 server socket 未 server_close() → Windows 下下次启动撞 "Address already in use"。
        if monitor_server is not None:
            try:
                from train_monitor import shutdown_monitor_server
                shutdown_monitor_server(monitor_server)
            except Exception:
                pass
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

    def _sample_with_vae_swap(*args_pos, **kwargs_pos):
        """sample_image 的薄包装：如果 VAE 被 offload 到 CPU 了，临时搬回 GPU 出图，
        出完再搬回去，省显存。否则直接透传。"""
        if vae_offloaded_to_cpu:
            try:
                vae.model = vae.model.to(device=device, dtype=dtype)
                return sample_image(*args_pos, **kwargs_pos)
            finally:
                try:
                    vae.model = vae.model.cpu()
                    if bool(getattr(args, "empty_cache_after_sample", True)):
                        torch.cuda.empty_cache()
                except Exception as _e:
                    logger.warning(f"VAE 出图后回 CPU 失败（忽略）: {_e}")
        return sample_image(*args_pos, **kwargs_pos)

    objective_cfg = build_training_objective_config(args)

    # InfoNoise entropy_rate 模式需要在 .factors() 里除以当前 loss-weighting w(t)。
    # 构造一个轻量闭包，复用 trainer.objective.compute_loss_weight，参数从 objective_cfg.loss 取。
    from trainer.objective import compute_loss_weight as _compute_loss_weight
    _loss_cfg_for_weight = objective_cfg.loss

    def _loss_weight_fn(t_tensor):
        return _compute_loss_weight(
            t_tensor.float(),
            scheme=_loss_cfg_for_weight.weighting_scheme,
            min_snr_gamma=_loss_cfg_for_weight.min_snr_gamma,
            weight_cap_ratio=0.0,  # bin-level 估计不需要 batch-level cap
            detail_inv_t_min=_loss_cfg_for_weight.detail_inv_t_min,
            detail_inv_t_max=_loss_cfg_for_weight.detail_inv_t_max,
        )

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
        low_noise_gate=bool(getattr(args, "adaptive_timestep_low_noise_gate", False)),
        gate_n=float(getattr(args, "adaptive_timestep_gate_n", 3.0) or 3.0),
        gate_c=float(getattr(args, "adaptive_timestep_gate_c", 0.05) or 0.05),
        loss_weight_fn=_loss_weight_fn,
    )
    if adaptive_ts.enabled:
        logger.info("[adaptive_timestep] enabled: %s", adaptive_ts.summary())
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

    def run_sample_checkpoint(label, filename_stem):
        prompt = get_next_sample_prompt()
        prompt_short = prompt[:50] + "..." if len(prompt) > 50 else prompt
        emit(f"采样中 ({label}): {prompt_short}")
        model.eval()
        if hasattr(optimizer, "eval"):
            optimizer.eval()
        s_w = int(getattr(args, "sample_width", 0) or 0) or int(args.resolution)
        s_h = int(getattr(args, "sample_height", 0) or 0) or int(args.resolution)
        s_cfg = float(getattr(args, "sample_cfg_scale", 4.0) or 4.0)
        s_neg = str(getattr(args, "sample_negative_prompt", "") or "")
        s_steps = int(getattr(args, "sample_infer_steps", 25) or 25)
        s_sampler = str(getattr(args, "sample_sampler_name", "er_sde") or "er_sde")
        s_sched = str(getattr(args, "sample_scheduler", "simple") or "simple")
        img = _sample_with_vae_swap(
            model, vae, qwen_model, qwen_tok, t5_tok,
            prompt, height=s_h, width=s_w, steps=s_steps, cfg_scale=s_cfg,
            negative_prompt=(s_neg or None),
            sampler_name=s_sampler,
            scheduler=s_sched,
            device=device, dtype=dtype,
            use_t5_token_weights=bool(getattr(args, "use_t5_token_weights", True)),
            injector=injector,
        )
        sample_path = sample_dir / f"{filename_stem}.png"
        img.save(sample_path)
        emit(f"采样保存: {sample_path.name}")
        if monitor_server:
            try:
                update_monitor(sample_path=sample_path)
            except Exception:
                pass
        if hasattr(optimizer, "train"):
            optimizer.train()
        model.train()
        return sample_path

    def save_lora_checkpoint(filename_stem):
        if hasattr(optimizer, "eval"):
            optimizer.eval()
        lora_path = output_dir / f"{args.output_name}_{filename_stem}.safetensors"
        injector.save(lora_path, model=model)
        emit(f"Saved LoRA: {lora_path}")
        if hasattr(optimizer, "train"):
            optimizer.train()
        return lora_path

    # Step 0 初始采样（基线效果，测试所有提示词）
    # 只在新训练时执行（global_step == 0），resume 时跳过
    sampling_enabled = (
        args.sample_steps > 0
        or args.sample_every > 0
        or int(getattr(args, "sample_reference_steps", 0) or 0) > 0
    )
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
        _baseline_n = max(int(getattr(args, "baseline_sample_count", 3) or 3), 1)
        for i, prompt in enumerate(sample_prompts[:_baseline_n]):
            if s_seed:
                torch.manual_seed(s_seed + i)
            img = _sample_with_vae_swap(
                model, vae, qwen_model, qwen_tok, t5_tok,
                prompt, height=s_h, width=s_w, steps=s_steps, cfg_scale=s_cfg,
                negative_prompt=(s_neg or None),
                sampler_name=s_sampler,
                scheduler=s_sched,
                device=device, dtype=dtype,
                use_t5_token_weights=bool(getattr(args, "use_t5_token_weights", True)),
                injector=injector,
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
    sample_accum_pending = 0
    sample_accum_loss_sum = None
    legacy_accum_samples = 0
    pending_reference_batches = 0
    step_start_time = time.perf_counter()

    # pad_mask 复用缓存：key=(B, 1, H_lat, W_lat) → tensor
    # ARB 多 bucket 时大概会有 5-20 个 unique shape，cache 几 KB 内存换掉每步的 cudaMalloc。
    _pad_mask_cache: dict = {}

    # LWD 掩码目前只在 dense 路径实现（packed token 路径需要 token 级掩码，未做）
    if objective_cfg.loss.lwd_enabled and fit_packed_training:
        raise RuntimeError("lwd_mask_enabled 目前仅支持 dense 路径（fit_packed_training=false）")

    # ── TREAD token 路由（arXiv 2501.04765，训练期省算力）────────────────────
    _tread_ratio = (float(getattr(args, "tread_ratio", 0.0) or 0.0)
                    if bool(getattr(args, "tread_enabled", False)) else 0.0)
    _tread_start = int(getattr(args, "tread_start_layer", 3) or 0)
    _tread_end = int(getattr(args, "tread_end_layer", -4) or 0)
    if _tread_ratio > 0.0:
        if fit_packed_training:
            raise RuntimeError("tread_enabled 目前仅支持 dense 路径（fit_packed_training=false）")
        if not (0.0 < _tread_ratio <= 0.9):
            raise ValueError(f"tread_ratio 须在 (0, 0.9] 内，当前 {_tread_ratio}")
        _tnb = len(model.blocks)
        _tts = _tread_start if _tread_start >= 0 else _tnb + _tread_start
        _tte = _tread_end if _tread_end > 0 else _tnb + _tread_end
        if not (0 <= _tts < _tte <= _tnb):
            raise ValueError(f"TREAD 路由段非法: blocks[{_tts}:{_tte}) / {_tnb} blocks")
        emit(f"[tread] token 路由启用: ratio={_tread_ratio} blocks[{_tts}:{_tte})/{_tnb}"
             f"（仅训练前向生效，采样/eval/推理不受影响）")

    # ── inv_loss_ema 加权（EDM2 解析变体）：与 adaptive_timestep 互斥 ────────────
    loss_bin_ema = None
    if objective_cfg.loss.weighting_scheme == "inv_loss_ema":
        if adaptive_ts.enabled:
            raise RuntimeError(
                "loss_weighting_scheme=inv_loss_ema 与 adaptive_timestep 互斥"
                "（一个改采样分布、一个改 loss 权重，叠加=双重补偿互相打架）。二选一。")
        loss_bin_ema = LossBinEMA(bins=8, decay=0.97, burn_in=100, min_w=0.25, max_w=4.0)
        emit("[inv_loss_ema] EDM2 解析式按-t loss 均衡加权已启用（burn-in 100 步）")

    # ── GAF（梯度一致性过滤，脏数据鲁棒性 B1；默认关，零影响）──────────────────────
    # 按梯度方向（非 loss 大小）降权"与 batch 共识方向不合"的脏样本：高细节干净图方向对
    # 不被罚、脏图方向歪被压。周期摊销（every 步评估一次）+ 每步施加缓存信任 → ~1.x× 成本。
    gaf_ctrl = None
    if bool(getattr(args, "gaf_enabled", False)):
        _gaf_log = str(getattr(args, "gaf_log_path", "") or "").strip()
        if not _gaf_log:
            _gaf_log = os.path.join(args.output_dir, f"{args.output_name}_gaf_trust.csv")
        gaf_ctrl = GafController(
            [p for _, p in trainable_named_params],
            enabled=True,
            every=int(getattr(args, "gaf_every", 4) or 4),
            warmup=int(getattr(args, "gaf_warmup", 100) or 100),
            mode=str(getattr(args, "gaf_mode", "soft") or "soft"),
            threshold=float(getattr(args, "gaf_threshold", 0.0) or 0.0),
            temp=float(getattr(args, "gaf_temp", 0.15) or 0.15),
            floor=float(getattr(args, "gaf_floor", 0.3) or 0.3),
            min_keep=int(getattr(args, "gaf_min_keep", 2) or 2),
            trust_decay=float(getattr(args, "gaf_trust_decay", 0.9) or 0.9),
            log_path=_gaf_log,
        )
        emit(f"[gaf] 梯度一致性过滤已启用：every={gaf_ctrl.every} warmup={gaf_ctrl.warmup} "
             f"mode={gaf_ctrl.mode} floor={gaf_ctrl.floor}（按方向降权脏样本；信任日志→{_gaf_log}）")

    # ── 固定网格 eval loss（确定性曲线，跨 run 可比）───────────────────────────
    _eval_every = int(getattr(args, "eval_every", 0) or 0)
    _eval_set = []          # [(latents_1xC1HW_gpu, cross_1xLxD_gpu)]
    _eval_t_grid = []
    if _eval_every > 0:
        _eval_t_grid = [float(s) for s in str(getattr(args, "eval_t_grid", "0.1,0.3,0.5,0.7,0.9")).split(",") if s.strip()]
        _eval_n = max(int(getattr(args, "eval_count", 4) or 4), 1)
        emit(f"[eval] 收集 {_eval_n} 个固定样本，t 网格 {_eval_t_grid} ...")
        with torch.no_grad():
            for _eb in dataloader:
                if len(_eval_set) >= _eval_n:
                    break
                if use_cached:
                    _elat = _eb["latents"].to(device, dtype=dtype)
                else:
                    _epx = _eb["pixel_values"].to(device, dtype=dtype)
                    _elat = vae.model.encode(_epx.unsqueeze(2), vae.scale).to(dtype)
                for _bi in range(_elat.shape[0]):
                    if len(_eval_set) >= _eval_n:
                        break
                    _ecap = _eb["captions"][_bi]
                    _eq_emb, _eq_attn = encode_qwen(qwen_model, qwen_tok,
                                                    [_build_qwen_text_from_prompt(_ecap)], device)
                    _et5_ids, _et5_attn, _et5_w = tokenize_t5_weighted(t5_tok, [_ecap], max_length=512)
                    _ecross = model.preprocess_text_embeds(
                        _eq_emb, _et5_ids.to(device), _et5_attn.to(device), _eq_attn)
                    if (getattr(args, "use_t5_token_weights", True)
                            and getattr(model, "llm_adapter", None) is not None
                            and _ecross.shape[1] == _et5_w.shape[1]):
                        _ecross = _ecross * _et5_w.to(device, dtype=torch.float32).to(_ecross.dtype).unsqueeze(-1)
                    if _ecross.shape[1] < 512:
                        _ecross = F.pad(_ecross, (0, 0, 0, 512 - _ecross.shape[1]))
                    _eval_set.append((_elat[_bi:_bi + 1].clone(), _ecross.clone()))
        emit(f"[eval] 固定 eval 集就绪：{len(_eval_set)} 个样本")

    def run_eval_loss(step):
        """固定样本 × 固定噪声 × 固定 t 网格的确定性 MSE eval。

        loss 统一用 MSE（与训练 loss_type 无关），保证跨配置可比；
        schedule-free 必须切 optimizer.eval() 测平均序列权重。
        结果 emit 一行 + 追加 output_dir/eval_loss.csv。
        """
        if not _eval_set:
            return
        model.eval()
        if hasattr(optimizer, "eval"):
            optimizer.eval()
        eval_seed = int(getattr(args, "eval_seed", 1234) or 1234)
        per_t_sums = [0.0] * len(_eval_t_grid)
        with torch.no_grad():
            for _i, (_lat, _cross) in enumerate(_eval_set):
                _pm = torch.zeros(1, 1, _lat.shape[-2], _lat.shape[-1], device=device, dtype=dtype)
                for _j, _tv in enumerate(_eval_t_grid):
                    _g = torch.Generator(device="cpu").manual_seed(eval_seed + 1000 * _i + _j)
                    _nz = torch.randn(_lat.shape, generator=_g, dtype=torch.float32).to(device=device, dtype=dtype)
                    _tt = torch.full((1,), float(_tv), device=device)
                    _noisy = (1 - _tv) * _lat + _tv * _nz
                    _tgt = _nz - _lat
                    with torch.autocast("cuda", dtype=dtype):
                        _pr = forward_with_optional_checkpoint(
                            model, _noisy, _tt.view(-1, 1), _cross, _pm, use_checkpoint=False)
                    per_t_sums[_j] += float(per_sample_loss(_pr, _tgt, loss_type="mse").item())
        n = max(len(_eval_set), 1)
        per_t = [s / n for s in per_t_sums]
        mean_v = sum(per_t) / max(len(per_t), 1)
        emit(f"[eval] step {step} mean={mean_v:.6f} " +
             " ".join(f"t{tv:g}={v:.6f}" for tv, v in zip(_eval_t_grid, per_t)))
        csv_path = output_dir / "eval_loss.csv"
        write_header = not csv_path.exists()
        with open(csv_path, "a", encoding="utf-8") as f:
            if write_header:
                f.write("step,mean," + ",".join(f"t{tv:g}" for tv in _eval_t_grid) + "\n")
            f.write(f"{step},{mean_v:.6f}," + ",".join(f"{v:.6f}" for v in per_t) + "\n")
        if hasattr(optimizer, "train"):
            optimizer.train()
        model.train()

    # ── LoRA-One (arXiv:2502.01235) 谱对齐初始化 ─────────────────────────────
    # 训练正式开始前：累积 N 个 batch 的全参梯度 → KPSVD → 初始化 LoKr 因子。
    # 只在全新训练（global_step==0）执行；resume 时跳过（因子已是训练后状态）。
    # 显存注意：目标模块的 base 权重要临时挂 grad（fp32 累积 ≈ 模型规模 ×4B），
    # 96GB 卡无压力；24GB 级卡慎开。
    _lora_one_steps = int(getattr(args, "lora_one_init_steps", 0) or 0)
    if _lora_one_steps > 0 and global_step == 0:
        from trainer.lora_one import lora_one_kpsvd_init
        if fit_packed_training:
            raise RuntimeError("lora_one_init_steps 目前仅支持 dense 路径（fit_packed_training=false）")
        emit(f"[lora-one] 收集 {_lora_one_steps} 个 batch 的全参梯度用于谱对齐初始化...")
        _lo_targets = dict(injector.injected)
        for _l in _lo_targets.values():
            _l.original.weight.requires_grad_(True)
        _lo_grads: dict = {}
        model.eval()  # 关 dropout/module_dropout 拿干净梯度；grad checkpoint 在 eval 下照常工作
        _lo_iter = iter(dataloader)
        _lo_done = 0
        while _lo_done < _lora_one_steps:
            try:
                _lb = next(_lo_iter)
            except StopIteration:
                _lo_iter = iter(dataloader)
                continue
            _lcaps = _lb["captions"]
            if use_cached:
                _llat = _lb["latents"].to(device, dtype=dtype)
            else:
                with torch.no_grad():
                    _lpx = _lb["pixel_values"].to(device, dtype=dtype)
                    _llat = vae.model.encode(_lpx.unsqueeze(2), vae.scale).to(dtype)
            with torch.no_grad():
                _lq_texts = [_build_qwen_text_from_prompt(c) for c in _lcaps]
                _lq_emb, _lq_attn = encode_qwen(qwen_model, qwen_tok, _lq_texts, device)
                _lt5_ids, _lt5_attn, _lt5_w = tokenize_t5_weighted(t5_tok, _lcaps, max_length=512)
                _lcross = model.preprocess_text_embeds(
                    _lq_emb, _lt5_ids.to(device), _lt5_attn.to(device), _lq_attn)
                if (getattr(args, "use_t5_token_weights", True)
                        and getattr(model, "llm_adapter", None) is not None
                        and _lcross.shape[1] == _lt5_w.shape[1]):
                    _lcross = _lcross * _lt5_w.to(device, dtype=torch.float32).to(_lcross.dtype).unsqueeze(-1)
                if _lcross.shape[1] < 512:
                    _lcross = F.pad(_lcross, (0, 0, 0, 512 - _lcross.shape[1]))
            _lbs = _llat.shape[0]
            # 用配置的 t 分布采样（梯度子空间应反映真实训练目标），分层进一步降估计方差
            _lt = sample_t_stratified(
                _lbs, device, mode=objective_cfg.timestep.mode,
                shift=objective_cfg.timestep.flow_shift,
                mix_low_prob=objective_cfg.timestep.mix_low_prob,
                laplace_mu=objective_cfg.timestep.laplace_mu,
                laplace_b=objective_cfg.timestep.laplace_b,
                logsnr_mu=objective_cfg.timestep.logsnr_mu,
                logsnr_sigma=objective_cfg.timestep.logsnr_sigma,
            )
            _lt = apply_timestep_schedule_shift(_lt, objective_cfg.timestep.schedule_shift)
            _lt = apply_t_range(_lt, objective_cfg.timestep.t_min, objective_cfg.timestep.t_max)
            _lnoise = make_noise_from_config(_llat, objective_cfg.noise)
            _lte = _lt.view(-1, 1, 1, 1, 1)
            _lnoisy = (1 - _lte) * _llat + _lte * _lnoise
            _ltarget = _lnoise - _llat
            _lpad = torch.zeros(_lbs, 1, _llat.shape[-2], _llat.shape[-1], device=device, dtype=dtype)
            with torch.autocast("cuda", dtype=dtype):
                _lpred = forward_with_optional_checkpoint(
                    model, _lnoisy, _lt.view(-1, 1), _lcross, _lpad,
                    use_checkpoint=bool(args.grad_checkpoint))
                _lloss = per_sample_loss(_lpred, _ltarget, loss_type="mse").mean()
            _lloss.backward()
            for _n, _l in _lo_targets.items():
                _g = _l.original.weight.grad
                if _g is None:
                    continue
                if _n in _lo_grads:
                    _lo_grads[_n] += _g.detach().float()
                else:
                    _lo_grads[_n] = _g.detach().float().clone()
            model.zero_grad(set_to_none=True)
            _lo_done += 1
            emit(f"[lora-one] grad batch {_lo_done}/{_lora_one_steps} loss={float(_lloss.detach()):.4f}")
        for _n in _lo_grads:
            _lo_grads[_n] /= float(_lora_one_steps)
        _lo_stats = lora_one_kpsvd_init(
            injector, _lo_grads,
            scale_rel=float(getattr(args, "lora_one_init_scale", 0.01) or 0.01))
        emit(f"[lora-one] 初始化完成: {_lo_stats}")
        for _l in _lo_targets.values():
            _l.original.weight.requires_grad_(False)
        model.zero_grad(set_to_none=True)
        _lo_grads.clear()
        del _lo_targets, _lo_iter
        model.train()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for epoch in range(start_epoch, args.epochs):
        current_epoch = epoch
        if hasattr(dataloader, "batch_sampler") and hasattr(dataloader.batch_sampler, "set_epoch"):
            dataloader.batch_sampler.set_epoch(epoch)
        if sample_accum_enabled and hasattr(dataloader, "batch_sampler") and hasattr(dataloader.batch_sampler, "set_accumulation_offset"):
            dataloader.batch_sampler.set_accumulation_offset(sample_accum_pending)
        for batch_idx, batch in enumerate(dataloader):
            # 在累积周期开始时记录时间 + 重置 clean 标志
            if sample_accum_enabled:
                if sample_accum_pending == 0:
                    step_start_time = time.perf_counter()
                    sample_accum_loss_sum = None
                    pending_reference_batches = 0
                    accum_clean = True
            elif batch_idx % args.grad_accum == 0:
                step_start_time = time.perf_counter()
                pending_reference_batches = 0
                accum_clean = True
            batch_reference_batches = 0
            if hasattr(dataloader, "batch_sampler") and hasattr(dataloader.batch_sampler, "reference_batches_for_batch_index"):
                batch_reference_batches = dataloader.batch_sampler.reference_batches_for_batch_index(batch_idx)
            pending_reference_batches += int(batch_reference_batches)

            captions = batch["captions"]

            # caption dropout：随机把 caption 替换为空字符串，提升 CFG 服从度（Anima 主要靠 CFG 出图）
            cap_drop_p = float(getattr(args, "caption_dropout_rate", 0.0) or 0.0)
            if cap_drop_p > 0:
                captions = ["" if random.random() < cap_drop_p else c for c in captions]

            # 获取 latents（缓存模式或实时编码）。
            # ★ 用 non_blocking=True 配合 DataLoader pin_memory，让 H2D 拷贝与下个 batch overlap。
            _non_blk = bool(_loader_kwargs.get("pin_memory", False))
            pixel_mask = None
            cached_latent_mask = None
            if use_cached:
                latents = batch["latents"].to(device, dtype=dtype, non_blocking=_non_blk)
                if fit_packed_training:
                    # token_bucket 缓存路径：latent_mask 由 collate 从形状重建（满覆盖→全 1）。
                    cached_latent_mask = batch["latent_mask"].to(
                        device, dtype=torch.float32, non_blocking=_non_blk
                    )
            else:
                pixels = batch["pixel_values"].to(device, dtype=dtype, non_blocking=_non_blk)
                if fit_packed_training:
                    pixel_mask = batch["pixel_mask"].to(device, dtype=torch.float32, non_blocking=_non_blk)
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
            sched_shift = objective_cfg.timestep.schedule_shift
            # 三峰路由概率退火（timestep_mix_anneal_end=0 时为 no-op，恒返基础值）
            _an_s = int(getattr(args, "timestep_mix_anneal_start", 0) or 0)
            _an_e = int(getattr(args, "timestep_mix_anneal_end", 0) or 0)
            mix_low_prob = anneal_mix_prob(
                objective_cfg.timestep.mix_low_prob,
                float(getattr(args, "timestep_mix_low_prob_end", -1.0)),
                global_step, _an_s, _an_e)
            mix_high_prob_cur = anneal_mix_prob(
                objective_cfg.timestep.mix_high_prob,
                float(getattr(args, "timestep_mix_high_prob_end", -1.0)),
                global_step, _an_s, _an_e)
            t = adaptive_ts.sample(
                bs, device, mode=ts_mode, shift=f_shift,
                mix_low_prob=mix_low_prob, schedule_shift=sched_shift,
                laplace_mu=objective_cfg.timestep.laplace_mu,
                laplace_b=objective_cfg.timestep.laplace_b,
                logsnr_mu=objective_cfg.timestep.logsnr_mu,
                logsnr_sigma=objective_cfg.timestep.logsnr_sigma,
                mix_high_prob=mix_high_prob_cur,
                stratified=objective_cfg.timestep.stratified,
                global_step=global_step,
            )

            # SD3 式 σ schedule shift：作用于所有模式的 t（含 uniform / mixed_*），
            # 把噪声混合用的 sigma 整体偏向高噪声端。1.0=禁用，向后兼容。
            t = apply_timestep_schedule_shift(t, sched_shift)
            # t 值域截断（timestep_t_min/t_max；默认 0/1 = 历史 1e-4 行为）
            t = apply_t_range(t, objective_cfg.timestep.t_min, objective_cfg.timestep.t_max)

            t_exp = t.view(-1, 1, 1, 1, 1)

            noise = make_noise_from_config(latents, objective_cfg.noise)

            noisy = (1 - t_exp) * latents + t_exp * noise
            target = noise - latents

            latent_mask = None
            if fit_packed_training:
                if cached_latent_mask is not None:
                    # 缓存路径：mask 已是 latent 分辨率（[B,1,h,w]），直接用。
                    latent_mask = cached_latent_mask
                else:
                    latent_mask = F.interpolate(
                        pixel_mask.float(),
                        size=latents.shape[-2:],
                        mode="nearest",
                    )

            # 前向
            # ★ pad_mask 是全零张量，每步都 zeros 分配 → ARB 多 bucket 时
            # 频繁触发 cudaMalloc/cudaFree。改成 by-shape cache，相同 shape 复用同一张量
            # （forward 内不修改它，只是把它当 attention mask 的占位）。
            pad_mask = None
            if not fit_packed_training:
                _pad_key = (int(bs), 1, int(latents.shape[-2]), int(latents.shape[-1]))
                pad_mask = _pad_mask_cache.get(_pad_key)
                if pad_mask is None or pad_mask.device != latents.device or pad_mask.dtype != dtype:
                    pad_mask = torch.zeros(*_pad_key, device=device, dtype=dtype)
                    _pad_mask_cache[_pad_key] = pad_mask
            # T-LoRA: 把当前 batch 的 timestep 写到每个注入的 LoRA adapter，让其在 forward
            # 内按 r(t) 应用 rank mask；其它 variant 该调用是空操作。完成后 reset 避免
            # 跨 step 残留（采样 / eval / 其它前向不应受影响）。
            injector.set_current_t(t.float().detach())
            # ★ module_dropout: compile-safe 模式下每 step 预抽 keep 标量（把 RNG 移出被 compile
            # 追踪的 forward）；与 current_t 同生命周期，必须存活到 backward 之后（grad checkpoint
            # recompute 要见同值），故 reset 放在 backward 之后与 NaN-continue 之前。
            # eager 模式（torch_compile 关，默认）本调用在 injector 层直接短路，零额外每步开销。
            injector.roll_module_dropout()
            # ★ T-LoRA: current_t 必须存活到 backward 之后。原本在这里 finally reset
            # 是错的：grad checkpoint 在 backward 时会 recompute forward，那一刻 current_t
            # 必须和原 forward 完全一致，否则 LoRALayer 的 _apply_tlora_mask / ortho 补偿
            # 分支会变，autograd 图保存的张量数对不上（torch.utils.checkpoint.CheckpointError）。
            # reset 已移到 backward 之后，以及 NaN-loss continue 路径之前。
            with torch.autocast("cuda", dtype=dtype):
                if fit_packed_training:
                    noisy_tokens, fit_grid, fit_mask, fit_size = model.patchify_latents_to_tokens(noisy, latent_mask)
                    target_tokens, _target_grid, _target_mask, _target_size = model.patchify_latents_to_tokens(target, latent_mask)
                    pred = forward_packed_with_optional_checkpoint(
                        model,
                        noisy_tokens,
                        t.view(-1, 1),
                        cross,
                        fit_grid,
                        fit_mask,
                        fit_size,
                        use_checkpoint=bool(getattr(args, "grad_checkpoint", False)),
                    )
                    target = target_tokens
                    per_sample = masked_token_loss(
                        pred,
                        target,
                        fit_mask,
                        loss_type=objective_cfg.loss.loss_type,
                        huber_c=objective_cfg.loss.huber_c,
                        huber_schedule=objective_cfg.loss.huber_schedule,
                        t=t.float(),
                        huber_snr_clamp_max=objective_cfg.loss.huber_snr_clamp_max,
                    )
                else:
                    pred = forward_with_optional_checkpoint(
                        model, noisy, t.view(-1, 1), cross, pad_mask,
                        use_checkpoint=args.grad_checkpoint,
                        tread_ratio=_tread_ratio,
                        tread_start=_tread_start,
                        tread_end=_tread_end,
                    )
                    # LWD 掩码从 clean latents 计算（与噪声无关），只作用于主 loss；
                    # ΔFM 负样本项保持全图（其量级由 λ 单独控制）。
                    lwd_w = None
                    if objective_cfg.loss.lwd_enabled:
                        lwd_w = lwd_saliency_mask(latents, t, objective_cfg.loss.lwd_floor)
                    per_sample = per_sample_loss(
                        pred,
                        target,
                        loss_type=objective_cfg.loss.loss_type,
                        huber_c=objective_cfg.loss.huber_c,
                        huber_schedule=objective_cfg.loss.huber_schedule,
                        t=t.float(),
                        huber_snr_clamp_max=objective_cfg.loss.huber_snr_clamp_max,
                        weight_map=lwd_w,
                    )

                # GAF（B1）：GAF 步抽每样本梯度→方向信任，更新逐图 EMA。用 ΔFM 之前的干净主
                # 重建 per_sample；autograd.grad 不污染 .grad、retain_graph 保后续主 backward。
                # 仅每 gaf_every 步触发（周期摊销）；warmup 内不介入。
                if gaf_ctrl is not None and gaf_ctrl.should_run(global_step):
                    gaf_ctrl.update_trust(per_sample, batch.get("images"), global_step)

                # ── inv_loss_ema：EDM2 不确定性加权的解析变体（loss_weighting_scheme）──
                # 先用原始 per_sample 更新 bin-EMA，再乘权重（ΔFM 项之后单独按 λ 控制）。
                if loss_bin_ema is not None:
                    loss_bin_ema.update(t.float(), per_sample)
                    per_sample = per_sample * loss_bin_ema.weight(t.float()).to(per_sample.dtype)

                # ── ΔFM: Contrastive Flow Matching（默认关闭，dfm_lambda=0）──
                # per_sample ← per_sample - λ·||v_pred - v_另一样本target||²，反"回归条件均值→发灰发雾"。
                # 不改噪声分布、零额外前向 → 不会重演 Immiscible 的推理斑块问题。
                # dfm_mode=vecor：负样本改为对 target 的破坏性增强（通道乱序/裁剪缩放），
                # 不依赖 batch 大小（VeCoR arXiv 2511.18942 部分移植）；dense 路径限定。
                _dfm_lambda = float(objective_cfg.loss.dfm_lambda or 0.0)
                _dfm_mode = str(getattr(args, "dfm_mode", "batch") or "batch")
                if _dfm_lambda > 0.0 and _dfm_mode == "vecor" and not fit_packed_training:
                    per_sample_neg = vecor_contrastive_neg(
                        pred, target, t.float(),
                        loss_type=objective_cfg.loss.loss_type,
                        huber_c=objective_cfg.loss.huber_c,
                        huber_schedule=objective_cfg.loss.huber_schedule,
                        huber_snr_clamp_max=objective_cfg.loss.huber_snr_clamp_max,
                    )
                    per_sample = per_sample - _dfm_lambda * per_sample_neg
                elif _dfm_lambda > 0.0 and bs > 1:
                    per_sample_neg = contrastive_flow_matching_neg(
                        pred, target, t.float(),
                        loss_type=objective_cfg.loss.loss_type,
                        huber_c=objective_cfg.loss.huber_c,
                        huber_schedule=objective_cfg.loss.huber_schedule,
                        huber_snr_clamp_max=objective_cfg.loss.huber_snr_clamp_max,
                        mask=fit_mask if fit_packed_training else None,
                    )
                    per_sample = per_sample - _dfm_lambda * per_sample_neg

                # GAF：把逐图方向信任当逐样本 loss 乘子（detached=按方向缩放该样本梯度贡献）。
                # 每步施加（未建档样本=1.0=不动）；高细节干净图≈1，方向常年不合的脏图被压。
                if gaf_ctrl is not None:
                    _gaf_w = gaf_ctrl.weight_for_batch(
                        batch.get("images"), device=per_sample.device, dtype=per_sample.dtype)
                    if _gaf_w is not None:
                        per_sample = per_sample * _gaf_w

                if sample_accum_enabled:
                    main_loss_per_sample = apply_loss_weighting_per_sample(
                        per_sample, t, objective_cfg.loss, normalize_weights=False
                    )
                    loss = main_loss_per_sample.sum() / float(effective_batch_size)
                else:
                    main_loss_per_sample = None
                    loss = apply_loss_weighting(per_sample, t, objective_cfg.loss)

            # ── 辅助 loss（Spectral / Perceptual）─────────────────────────────
            # 在 autocast 块之外计算：
            #   - spectral_loss 内部强制 fp32（FFT 数值稳定性）
            #   - perceptual_module 内部自己管 autocast（VAE decode + LPIPS/DINO 都用 bf16）
            # 顺序：先检查 t-gate → x₀ 恢复 → spectral → perceptual
            # x0_pred / x0_target / aux_total 在 None 兜底下声明，便于 NaN 路径统一 del
            x0_pred = None
            x0_target = None
            aux_total = None
            if objective_cfg.aux.any_enabled:
                # Early t-gate：若 batch 内所有样本的 t 都 >= 最大 gate，
                # 跳过 x₀ recovery（~1.5GB fp32 分配）和全部 aux forward。
                # 若只有部分样本命中 gate，则只恢复这些样本的 x₀，避免高 t 样本
                # 参与 latent FFT / VAE decode / LPIPS / DINO 的重路径。
                _aux = objective_cfg.aux
                _max_gate = max(
                    _aux.spectral_t_gate if _aux.spectral_enabled else 0.0,
                    _aux.perceptual_t_gate if _aux.perceptual_enabled else 0.0,
                )
                _aux_active = t.float() < _max_gate
                _any_below_gate = _aux_active.any().item()

                if _any_below_gate:
                    _aux_idx = _aux_active.nonzero(as_tuple=False).flatten()
                    # fit_packed 路径下 pred 是 patch-token；aux 需要网格 x0，先 unpatchify
                    # 回 [B,C,1,h,w]（token_bucket 单一网格满足 unpatchify_tokens 的 uniform-grid
                    # 前提）。非 fit 路径 pred 本就是网格。仅在过 t-gate 时才 unpatchify，省高 t 步的分配。
                    _pred_grid = (
                        model.unpatchify_tokens(pred, fit_size)
                        if fit_packed_training else pred
                    )
                    x0_pred = recover_x0_from_velocity(
                        noisy.index_select(0, _aux_idx),
                        t.index_select(0, _aux_idx),
                        _pred_grid.index_select(0, _aux_idx),
                    )
                    x0_target = latents.index_select(0, _aux_idx).float()
                    t_aux = t.index_select(0, _aux_idx)

                    aux_total = torch.zeros((), device=loss.device, dtype=torch.float32)
                    if _aux.spectral_enabled:
                        if sample_accum_enabled:
                            l_spec_vec = spectral_loss_per_sample(x0_pred, x0_target, t_aux, _aux)
                            aux_total = aux_total + (
                                float(_aux.spectral_lambda) * l_spec_vec.sum()
                                / float(effective_batch_size)
                            )
                        else:
                            l_spec = spectral_loss(x0_pred, x0_target, t_aux, _aux)
                            aux_total = aux_total + float(_aux.spectral_lambda) * l_spec

                    if _aux.perceptual_enabled and perceptual_module is not None:
                        if sample_accum_enabled:
                            l_perc_vec = perceptual_module.forward_per_sample(x0_pred, x0_target, t_aux)
                            aux_total = aux_total + l_perc_vec.sum() / float(effective_batch_size)
                        else:
                            l_perc = perceptual_module(x0_pred, x0_target, t_aux)
                            aux_total = aux_total + l_perc

                    loss = loss + aux_total.to(loss.dtype)

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
                if sample_accum_enabled:
                    sample_accum_pending += int(bs)
                    if sample_accum_pending >= effective_batch_size:
                        logger.warning(
                            f"[step {global_step}] Sample accumulation window contained a "
                            f"non-finite micro-batch loss; discarding the entire window."
                        )
                        optimizer.zero_grad(set_to_none=True)
                        sample_accum_pending = 0
                        sample_accum_loss_sum = None
                        pending_reference_batches = 0
                        accum_clean = True
                injector.set_current_t(None)  # T-LoRA: 本 micro-batch 跳过 backward，立即 reset
                injector.clear_module_dropout()
                # ★ 显式释放本 micro-batch 的 forward autograd 图。否则同周期内
                # 后续 micro-batch 的图会累积（pred / per_sample / loss 都还被 closure 引用），
                # 持续 NaN 时显存会迅速被这些"死图"占满 → OOM。
                # x0_pred / x0_target / aux_total 在 aux loss 启用时持有 pred / noisy 的
                # autograd 图引用，必须一并 del 才能让 GC 真正回收。
                del loss, per_sample, pred, target, noisy, cross
                if sample_accum_enabled and main_loss_per_sample is not None:
                    del main_loss_per_sample
                if x0_pred is not None:
                    del x0_pred
                if x0_target is not None:
                    del x0_target
                if aux_total is not None:
                    del aux_total
                continue

            if adaptive_ts.enabled and fit_packed_training:
                adaptive_ts.update(t.float(), per_sample.detach().float())
            elif adaptive_ts.enabled:
                adaptive_signal = adaptive_timestep_metric_signal(
                    per_sample,
                    pred,
                    target,
                    metric=adaptive_ts.metric,
                    highfreq_weight=adaptive_ts.highfreq_weight,
                )
                adaptive_ts.update(t.float(), adaptive_signal)
            if sample_accum_enabled:
                loss_to_backward = loss
                sample_accum_pending += int(bs)
                _loss_for_log = (
                    main_loss_per_sample.detach().float().sum()
                    if main_loss_per_sample is not None
                    else loss.detach().float() * float(effective_batch_size)
                )
                sample_accum_loss_sum = (
                    _loss_for_log if sample_accum_loss_sum is None
                    else sample_accum_loss_sum + _loss_for_log
                )
                step_boundary = sample_accum_pending >= effective_batch_size
            else:
                loss_to_backward = loss / args.grad_accum
                legacy_accum_samples += int(bs)
                step_boundary = (batch_idx + 1) % args.grad_accum == 0 or (batch_idx + 1) == len(dataloader)
            loss_to_backward.backward()
            # T-LoRA: backward 完成后才能 reset，否则 grad checkpoint recompute 会看到
            # current_t=None，与原 forward 走的分支不一致 → CheckpointError
            injector.set_current_t(None)
            injector.clear_module_dropout()

            if step_boundary:
                if sample_accum_enabled and sample_accum_pending != effective_batch_size:
                    raise RuntimeError(
                        f"Sample accumulation boundary mismatch: pending={sample_accum_pending}, "
                        f"effective_batch_size={effective_batch_size}. "
                        "BucketBatchSampler should split micro-batches before crossing the window."
                    )
                # ★ 守护 1：周期内有 micro-batch NaN/Inf loss → 整周期作废，不做 step
                if not accum_clean:
                    logger.warning(
                        f"[step {global_step}] Accumulation cycle contained a non-finite "
                        f"micro-batch loss; discarding the entire cycle's gradients."
                    )
                    optimizer.zero_grad(set_to_none=True)
                    if sample_accum_enabled:
                        sample_accum_pending = 0
                        sample_accum_loss_sum = None
                    else:
                        legacy_accum_samples = 0
                    pending_reference_batches = 0
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
                    for detail in summarize_bad_gradients(trainable_named_params):
                        logger.warning("[step %s] bad grad %s", global_step, detail)
                    optimizer.zero_grad(set_to_none=True)
                    if sample_accum_enabled:
                        sample_accum_pending = 0
                        sample_accum_loss_sum = None
                    else:
                        legacy_accum_samples = 0
                    pending_reference_batches = 0
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



                if hasattr(optimizer, "set_loss"):
                    if sample_accum_enabled:
                        optimizer_loss_val = (
                            float(sample_accum_loss_sum.detach().cpu()) / max(1, sample_accum_pending)
                            if sample_accum_loss_sum is not None else 0.0
                        )
                    else:
                        optimizer_loss_val = float(loss.item() * args.grad_accum)
                    optimizer.set_loss(optimizer_loss_val)
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
                committed_samples = int(effective_batch_size) if sample_accum_enabled else int(legacy_accum_samples)
                if sample_accum_enabled:
                    loss_val = (
                        float(sample_accum_loss_sum.detach().cpu()) / max(1, sample_accum_pending)
                        if sample_accum_loss_sum is not None else 0.0
                    )
                    sample_accum_pending = 0
                    sample_accum_loss_sum = None
                else:
                    loss_val = float(loss.item() * args.grad_accum)
                    legacy_accum_samples = 0
                samples_seen += max(0, committed_samples)
                previous_ref_step, ref_step = reference_tracker.commit_batches(pending_reference_batches)
                pending_reference_batches = 0
                sample_reference_steps = int(getattr(args, "sample_reference_steps", 0) or 0)
                save_reference_steps = int(getattr(args, "save_every_reference_steps", 0) or 0)
                sample_by_ref = reference_interval_crossed(
                    previous_ref_step,
                    ref_step,
                    sample_reference_steps,
                )
                save_by_ref = reference_interval_crossed(
                    previous_ref_step,
                    ref_step,
                    save_reference_steps,
                )
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
                            total_steps=total_steps, speed=speed_ema or 0,
                            ref_step=ref_step, samples_seen=samples_seen,
                        )
                    except Exception:
                        pass
                dt_step = now - step_start_time
                steps_per_sec = (1.0 / dt_step) if dt_step > 0 else 0.0
                speed_ema = steps_per_sec if speed_ema is None else (0.9 * speed_ema + 0.1 * steps_per_sec)
                ref_desc = f" ref={ref_step:.1f}" if reference_batch_size > 0 else ""

                if use_rich:
                    desc = f"epoch {epoch+1}/{args.epochs} step {global_step}/{total_steps or '?'}{ref_desc}"
                    progress.update(task_id, advance=1, description=desc,
                                    loss=loss_val, lr=float(lr), speed=float(speed_ema or 0))
                    if live and args.loss_curve_steps > 0 and not args.no_live_curve:
                        panel = render_curve_panel(loss_history, width=min(60, args.loss_curve_steps), height=10)
                        if panel is not None:
                            from rich.console import Group
                            live.update(Group(progress, panel))
                elif use_plain:
                    print(f"epoch {epoch+1}/{args.epochs} step {global_step}{ref_desc} loss={loss_val:.6f} lr={lr:.2e} speed={speed_ema:.2f} it/s", end="\r", flush=True)
                elif args.log_every and global_step % args.log_every == 0:
                    print(f"epoch={epoch} step={global_step}{ref_desc} loss={loss_val:.6f} lr={lr:.2e} speed={steps_per_sec:.2f} it/s")

                # Sample/checkpoint by optimizer step or by old-batch-equivalent reference step.
                if args.sample_steps > 0 and global_step % args.sample_steps == 0:
                    run_sample_checkpoint(f"step {global_step}", f"step_{global_step}")
                elif sample_by_ref:
                    ref_tag = int(ref_step)
                    run_sample_checkpoint(
                        f"ref_step {ref_step:.1f} (step {global_step})",
                        f"refstep_{ref_tag}_step_{global_step}",
                    )

                save_every_steps = getattr(args, "save_every_steps", 0)
                if save_every_steps > 0 and global_step % save_every_steps == 0:
                    save_lora_checkpoint(f"step{global_step}")
                elif save_by_ref:
                    ref_tag = int(ref_step)
                    save_lora_checkpoint(f"refstep{ref_tag}_step{global_step}")

                # 固定网格 eval loss（确定性曲线；eval_every=0 时 no-op）
                if _eval_every > 0 and global_step % _eval_every == 0:
                    run_eval_loss(global_step)

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
                    save_training_state(
                        state_path, injector, optimizer, epoch, global_step,
                        loss_history, monitor_state=monitor_data, scheduler=scheduler,
                        samples_seen=samples_seen,
                        reference_state={
                            "step": reference_tracker.step,
                            "grad_batches_pending": reference_tracker.grad_batches_pending,
                            "grad_accum": reference_tracker.grad_accum,
                        },
                    )
                    # 同时保存 LoRA 权重
                    lora_path = output_dir / f"{args.output_name}_step{global_step}.safetensors"
                    injector.save(lora_path, model=model)
                    if hasattr(optimizer, "train"): optimizer.train()

                # 检查 max_steps
                if args.max_steps and global_step >= args.max_steps:
                    break

        if sample_accum_enabled and sample_accum_pending > 0 and args.max_steps and global_step >= args.max_steps:
            logger.info(
                "max_steps reached; discarding %d pending samples in the unfinished accumulation window.",
                sample_accum_pending,
            )
            optimizer.zero_grad(set_to_none=True)
            sample_accum_pending = 0
            sample_accum_loss_sum = None
            pending_reference_batches = 0
            accum_clean = True

        # epoch 结束后的操作
        current_epoch = epoch + 1
        if not args.max_steps or global_step < args.max_steps:
            # 保存 checkpoint
            if args.save_every > 0 and current_epoch % args.save_every == 0:
                save_lora_checkpoint(f"epoch{current_epoch}")

            # 采样（轮换提示词）
            if args.sample_every > 0 and current_epoch % args.sample_every == 0:
                run_sample_checkpoint(f"epoch {current_epoch}", f"epoch_{current_epoch}")

        # 检查 max_steps
        if args.max_steps and global_step >= args.max_steps:
            break

    if sample_accum_enabled and sample_accum_pending > 0:
        if accum_clean:
            flushed_samples = sample_accum_pending
            logger.info(
                "Flushing final partial sample window: %d/%d samples.",
                sample_accum_pending, effective_batch_size,
            )
            # 复用常规 step 路径的核心保护逻辑，避免最后几张图只 backward 不更新。
            bad_grad = False
            for p in trainable_params:
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    bad_grad = True
                    break
            if bad_grad:
                logger.warning("[final flush] Non-finite gradient, skipping final partial update.")
                for detail in summarize_bad_gradients(trainable_named_params):
                    logger.warning("[final flush] bad grad %s", detail)
                optimizer.zero_grad(set_to_none=True)
            else:
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=grad_clip)
                if hasattr(optimizer, "set_loss"):
                    optimizer_loss_val = (
                        float(sample_accum_loss_sum.detach().cpu()) / max(1, sample_accum_pending)
                        if sample_accum_loss_sum is not None else 0.0
                    )
                    optimizer.set_loss(optimizer_loss_val)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                samples_seen += max(0, int(flushed_samples))
                reference_tracker.commit_batches(pending_reference_batches)
                pending_reference_batches = 0
                loss_val = (
                    float(sample_accum_loss_sum.detach().cpu()) / max(1, sample_accum_pending)
                    if sample_accum_loss_sum is not None else 0.0
                )
                if args.loss_curve_steps and args.loss_curve_steps > 0:
                    loss_history.append(loss_val)
                    if len(loss_history) > args.loss_curve_steps:
                        del loss_history[: len(loss_history) - args.loss_curve_steps]
        else:
            logger.warning(
                "[final flush] Accumulation window contained a non-finite micro-batch; "
                "discarding %d pending samples.",
                sample_accum_pending,
            )
            optimizer.zero_grad(set_to_none=True)
            pending_reference_batches = 0
        sample_accum_pending = 0
        sample_accum_loss_sum = None

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

    # ★ GAF：收尾 dump 逐图信任日志（低信任在前=脏样本提名表）。
    if gaf_ctrl is not None:
        gaf_ctrl.dump()
        emit(f"[gaf] {gaf_ctrl.summary()}；信任日志 → {gaf_ctrl.log_path}")

    # ★ 优雅关闭监控 server，让端口在训练结束后立即被释放（否则连续重启会撞端口）。
    if monitor_server is not None:
        try:
            from train_monitor import shutdown_monitor_server
            shutdown_monitor_server(monitor_server)
        except Exception:
            pass


if __name__ == "__main__":
    main()
