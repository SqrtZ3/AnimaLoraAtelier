"""数据集 / 分桶 / VAE latent 缓存。

包含：
- `BucketManager` —— ARB（aspect-ratio bucketing）桶生成与匹配
- `ImageDataset` —— 主数据集类，支持 JSON / TXT caption、tag dropout（含频率均衡）、
  flip 增强、按目录名前缀自动 repeat（kohya 风格 `10_xxx` → ×10）
- `RepeatDataset` —— 简单 N×重复包装
- `MergedDataset` —— 把主数据集与正则数据集（reg）首尾拼接
- `BucketBatchSampler` —— 把同桶样本聚一个 batch，避免 ARB 不同尺寸混 batch；
  `__len__` 用预计算的 per-bucket 加和（drop_last 在每个桶独立生效，旧的 `n // bs`
  实现在多桶时高估批数）
- `collect_dataset_captions` —— 抽出数据集原始 caption 列表（不做 shuffle/dropout），
  供训练期预览采样从训练集随机取 prompt
- `CachedLatentDataset` —— Kohya 风格 .npz 缓存，与上游 ImageDataset 共享 samples 列表
- `collate_fn` / `collate_fn_cached` —— 同 batch 内尺寸一致性检查 + 标准 stack

依赖 `utils/caption_utils.py`（动态 import，可选）。本模块对其它 trainer 子模块没有依赖。
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import logging
import math
import random
import sys
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

try:
    from .token_buckets import generate_token_buckets
except ImportError:  # when data.py is imported as a top-level module
    from token_buckets import generate_token_buckets

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NativeFitImagePlan:
    source_width: int
    source_height: int
    width: int
    height: int
    align_unit: int
    token_count: int
    token_h: int
    token_w: int
    was_padded: bool
    was_resized: bool = False
    was_cropped: bool = False


def _ceil_to_multiple(value: int, multiple: int) -> int:
    value = max(1, int(value))
    multiple = max(1, int(multiple))
    return ((value + multiple - 1) // multiple) * multiple


def _floor_to_multiple(value: int, multiple: int) -> int:
    value = max(1, int(value))
    multiple = max(1, int(multiple))
    return max(multiple, (value // multiple) * multiple)


def plan_native_fit_image(
    width: int,
    height: int,
    *,
    max_tokens: int = 65536,
    patch_size: int = 2,
    vae_downsample: int = 8,
    align_mode: str = "pad",
    over_budget_strategy: str = "fail",
) -> NativeFitImagePlan:
    """Plan native-first FiT sizing without implicit resize/crop.

    The returned size is the pixel size that will be handed to the VAE. The
    default mode pads up to the VAE+patch granularity so every source pixel is
    preserved and padded tokens can be masked later.
    """
    source_w = int(width)
    source_h = int(height)
    if source_w <= 0 or source_h <= 0:
        raise ValueError(f"image dimensions must be positive, got {source_w}x{source_h}")

    patch = max(1, int(patch_size))
    down = max(1, int(vae_downsample))
    align_unit = patch * down
    align_mode = (align_mode or "pad").lower()
    if align_mode == "pad" or align_mode == "ceil":
        planned_w = _ceil_to_multiple(source_w, align_unit)
        planned_h = _ceil_to_multiple(source_h, align_unit)
    elif align_mode == "floor":
        planned_w = _floor_to_multiple(source_w, align_unit)
        planned_h = _floor_to_multiple(source_h, align_unit)
    else:
        raise ValueError(f"unknown fit_align_mode={align_mode!r}; expected pad, ceil, or floor")

    token_w = planned_w // align_unit
    token_h = planned_h // align_unit
    token_count = token_w * token_h
    max_tokens = max(1, int(max_tokens))
    strategy = (over_budget_strategy or "fail").lower()
    if token_count > max_tokens:
        if strategy in ("fail", "skip"):
            raise ValueError(
                f"image {source_w}x{source_h} produces {token_count} FiT tokens, "
                f"which exceeds fit_max_tokens={max_tokens}. "
                "Increase fit_max_tokens or explicitly set fit_over_budget_strategy."
            )
        raise NotImplementedError(
            f"image {source_w}x{source_h} produces {token_count} FiT tokens, "
            f"above fit_max_tokens={max_tokens}; fit_over_budget_strategy={strategy!r} "
            "is reserved but not implemented yet in native-first FiT mode."
        )

    return NativeFitImagePlan(
        source_width=source_w,
        source_height=source_h,
        width=planned_w,
        height=planned_h,
        align_unit=align_unit,
        token_count=token_count,
        token_h=token_h,
        token_w=token_w,
        was_padded=(planned_w != source_w or planned_h != source_h),
        was_resized=False,
        was_cropped=False,
    )


def plan_multiscale_copy(
    width: int,
    height: int,
    target_tokens: int,
    *,
    patch_size: int = 2,
    vae_downsample: int = 8,
) -> Optional[NativeFitImagePlan]:
    """Plan an aspect-preserving *downscaled* copy of an image for the NaViT
    multiscale ladder (``navit_multiscale``).

    Returns a plan whose token count is ≤ ``target_tokens``, whose pixel dims are
    16px multiples, and whose ``source_* == width/height`` (the copy is produced by
    resize-cover + center-crop, so the valid region fills the whole latent → the
    all-ones-mask invariant of the navit cached path holds). Returns None when the
    source is not strictly larger than the target budget — the ladder never
    upscales and never emits a copy that duplicates the native size.
    """
    align = max(1, int(patch_size)) * max(1, int(vae_downsample))
    w, h = int(width), int(height)
    tgt = int(target_tokens)
    if w <= 0 or h <= 0 or tgt <= 0:
        return None
    # 不上采样、不产出与原生同档的副本：源图（floor 对齐后）token 数必须严格大于目标档。
    src_tokens = (w // align) * (h // align)
    if src_tokens <= tgt:
        return None
    # 等比缩放系数 s 使 (w·s/align)·(h·s/align) = tgt；floor 后 token 数必 ≤ tgt
    # （floor(a)·floor(b) ≤ a·b）。
    s = math.sqrt(tgt * align * align / float(w * h))
    tok_w = max(1, int(w * s) // align)
    tok_h = max(1, int(h * s) // align)
    # 极端长宽比下某轴被 max(1,·) 顶起时可能超预算；把另一轴压回来。
    if tok_w * tok_h > tgt:
        if tok_w >= tok_h:
            tok_w = max(1, tgt // tok_h)
        else:
            tok_h = max(1, tgt // tok_w)
    pw, ph = tok_w * align, tok_h * align
    return NativeFitImagePlan(
        source_width=pw,   # 有效区 = 整张（resize+crop 后无 padding）→ mask 恒全 1
        source_height=ph,
        width=pw,
        height=ph,
        align_unit=align,
        token_count=tok_w * tok_h,
        token_h=tok_h,
        token_w=tok_w,
        was_padded=False,
        was_resized=True,
        was_cropped=True,
    )


def compute_sample_accumulation_steps(dataset_size: int, epochs: int,
                                      effective_batch_size: int) -> int:
    """Return optimizer steps when flushing one final partial sample window.

    This helper is intentionally sample-count based: it does not care how many
    ARB buckets or micro-batches the DataLoader produces.
    """
    total_samples = max(0, int(dataset_size)) * max(0, int(epochs))
    eff_bs = max(1, int(effective_batch_size))
    return (total_samples + eff_bs - 1) // eff_bs


# numpy ↔ torch dtype 映射（CachedLatentDataset 保存 bf16 / fp16 时用）。
# numpy 1.x 无原生 bfloat16，所以 bf16 在磁盘上用 uint16 视图保存，读回来 view 成 bf16。
_NP_TO_TORCH = {
    "float32": torch.float32,
    "float16": torch.float16,
    "uint16_bf16": torch.bfloat16,  # 自定义 sentinel
}


class BucketManager:
    """ARB 分桶管理"""
    def __init__(self, base_reso=1024, min_reso=512, max_reso=2048, step=64,
                 base_resos=None, min_base_reso=0, max_base_reso=0,
                 base_reso_step=256, no_upscale=False, max_upscale=0.0,
                 ar_tolerance=0.05, max_aspect_ratio=2.0,
                 token_bucket=False, token_bucket_counts=None,
                 token_bucket_max_aspect_ratio=2.0,
                 token_bucket_min_dim=512, token_bucket_max_dim=2016):
        self.base_reso = int(base_reso)
        self.base_resos = self._normalize_base_resos(
            base_reso, base_resos, min_base_reso, max_base_reso,
            base_reso_step, min_reso
        )
        self.no_upscale = bool(no_upscale)
        self.max_upscale = float(max_upscale or 0.0)
        # 选桶的长宽比容差：见 get_bucket。多级分桶时把它和 detail-first 选择配合，
        # 避免大图被丢进长宽比略好的小面积桶。
        self.ar_tolerance = float(ar_tolerance)
        # 桶生成允许的最大单维长宽比 max(w/h, h/w)。历史写死 2.0；现可配置，
        # 让比 2:1 更扁的原生图（如 640x1664=2.6）也能得到匹配桶 → 训练侧零裁切零上采样，
        # 而不是被塞进 1.83:1 桶后上采样。受模型 RoPE 约束：~3.5:1 @1MP（长边 ≤1920）。
        self.max_aspect_ratio = max(1.0, float(max_aspect_ratio or 2.0))
        if token_bucket:
            counts = token_bucket_counts or [4032, 4200]
            if isinstance(counts, str):
                counts = [int(c) for c in counts.split(",") if c.strip()]
            self.token_bucket = True
            self.token_bucket_counts = [int(c) for c in counts]
            self.buckets = generate_token_buckets(
                self.token_bucket_counts,
                max_aspect_ratio=float(token_bucket_max_aspect_ratio or 2.0),
                min_dim_px=int(token_bucket_min_dim),
                max_dim_px=int(token_bucket_max_dim),
            )
        else:
            self.token_bucket = False
            self.token_bucket_counts = None
            self.buckets = self._generate(min_reso, max_reso, step, self.base_resos)

    @staticmethod
    def _normalize_base_resos(base_reso, base_resos, min_base_reso=0,
                              max_base_reso=0, base_reso_step=256,
                              min_reso=512):
        if base_resos is None or base_resos == "":
            values = []
        elif isinstance(base_resos, str):
            values = [v.strip() for v in base_resos.split(",")]
        else:
            values = list(base_resos)

        out = []
        for value in values:
            if value is None or value == "":
                continue
            ivalue = int(value)
            if ivalue <= 0:
                continue
            out.append(ivalue)
        if out:
            return sorted(set(out))

        max_base = int(max_base_reso or 0)
        if max_base > 0:
            min_base = int(min_base_reso or 0) or int(min_reso or base_reso)
            step = max(1, int(base_reso_step or 256))
            if min_base > max_base:
                min_base, max_base = max_base, min_base
            generated = list(range(min_base, max_base + 1, step))
            if not generated or generated[-1] != max_base:
                generated.append(max_base)
            return sorted(set(v for v in generated if v > 0))

        return [int(base_reso)]

    def _generate(self, min_r, max_r, step, bases):
        buckets = []
        seen = set()
        for base in bases:
            base_area = base * base
            for w in range(min_r, max_r + 1, step):
                for h in range(min_r, max_r + 1, step):
                    if abs(w * h - base_area) / base_area > 0.1:
                        continue
                    if max(w / h, h / w) > self.max_aspect_ratio + 1e-9:
                        continue
                    bucket = (w, h)
                    if bucket in seen:
                        continue
                    seen.add(bucket)
                    buckets.append(bucket)
        return sorted(buckets, key=lambda b: (b[0] * b[1], b[0], b[1]))

    def _bucket_allowed_for_image(self, bw, bh, w, h):
        scale = max(bw / max(1, w), bh / max(1, h))
        if self.no_upscale and scale > 1.0:
            return False
        if self.max_upscale > 0 and scale > self.max_upscale:
            return False
        return True

    def _score_bucket(self, bw, bh, w, h):
        aspect_diff = abs((w / h) - (bw / bh))
        scale = max(bw / max(1, w), bh / max(1, h))
        scale_diff = abs(math.log(max(scale, 1e-8)))
        area_diff = abs((bw * bh) - (w * h)) / max(1, w * h)
        return (aspect_diff, scale_diff, area_diff)

    def get_bucket(self, w, h):
        candidates = [
            (bw, bh) for (bw, bh) in self.buckets
            if self._bucket_allowed_for_image(bw, bh, w, h)
        ]
        if not candidates:
            candidates = list(self.buckets)
        if not candidates:
            return (self.base_reso, self.base_reso)

        def _aspect_diff(b):
            bw, bh = b
            return abs((w / max(1, h)) - (bw / max(1, bh)))

        # ★ detail-first 选桶（修复多级分桶下大图被丢进小面积桶的细节流失）：
        #   1) 先用长宽比容差圈出"裁切量可接受"的桶，避免为了塞进某个面积层而过度裁切；
        #   2) 在其中选 _score_bucket 的 (scale_diff, area_diff) 最小者 = 重采样最少。
        #      - 降采样场景：重采样最少 = 能容纳的最大面积桶 = 保留最多细节
        #        （旧实现以 aspect_diff 为唯一首要键，会把 2000x3000 丢进长宽比略好的
        #         640x960 小桶；现在它会落到最大的 ~1MP 桶）。
        #      - 上采样 fallback 场景：重采样最少 = 最接近源尺寸的桶 = 上采样幅度最小。
        best_ar = min(_aspect_diff(b) for b in candidates)
        tol = float(getattr(self, "ar_tolerance", 0.10))
        near = [b for b in candidates if _aspect_diff(b) <= best_ar + tol]
        return min(near, key=lambda b: self._score_bucket(b[0], b[1], w, h)[1:])


def _format_size(size):
    h, w = int(size[0]), int(size[1])
    return f"{w}x{h}"


def _downscale_ratio(source_size, bucket_key):
    source_h, source_w = source_size
    bucket_h, bucket_w = bucket_key
    return max(
        float(source_w) / max(1, float(bucket_w)),
        float(source_h) / max(1, float(bucket_h)),
    )


def _cover_scale(source_size, bucket_key):
    """训练 __getitem__ 的等比 cover-crop 缩放系数 = max(bw/sw, bh/sh)。
    >1 即"放大"（no_upscale 下不应出现）。注意它和 _downscale_ratio 不是互为倒数：
    cover-crop 取 max(bucket/source)，而 _downscale_ratio 取 max(source/bucket)，
    所以一张图可能 _downscale_ratio>1 却同时被放大（一维缩、另一维被放大裁切）。"""
    source_h, source_w = source_size
    bucket_h, bucket_w = bucket_key
    return max(
        float(bucket_w) / max(1, float(source_w)),
        float(bucket_h) / max(1, float(source_h)),
    )


def format_bucket_report(samples, limit=12, label="dataset"):
    """Return a compact text report of source sizes and bucket assignments."""
    from collections import Counter

    rows = []
    for sample in samples:
        bucket_key = sample.get("bucket_key")
        source_size = sample.get("source_size")
        if not bucket_key or not source_size:
            continue
        rows.append((sample.get("image"), source_size, bucket_key))

    if not rows:
        return f"[BucketReport:{label}] no bucketed samples"

    bucket_counts = Counter(bucket_key for _, _, bucket_key in rows)
    source_bins = Counter()
    downscales = []
    upscales = []
    for image, source_size, bucket_key in rows:
        longest = max(int(source_size[0]), int(source_size[1]))
        if longest < 768:
            source_bins["<768"] += 1
        elif longest < 1024:
            source_bins["768-1023"] += 1
        elif longest < 1536:
            source_bins["1024-1535"] += 1
        elif longest < 2048:
            source_bins["1536-2047"] += 1
        elif longest < 3072:
            source_bins["2048-3071"] += 1
        else:
            source_bins[">=3072"] += 1
        ratio = _downscale_ratio(source_size, bucket_key)
        if ratio > 1.0:
            downscales.append((ratio, image, source_size, bucket_key))
        cover = _cover_scale(source_size, bucket_key)
        if cover > 1.0 + 1e-6:
            upscales.append((cover, image, source_size, bucket_key))

    bucket_preview = ", ".join(
        f"{_format_size(bucket)}={count}"
        for bucket, count in bucket_counts.most_common(limit)
    )
    bin_order = ["<768", "768-1023", "1024-1535", "1536-2047", "2048-3071", ">=3072"]
    bin_preview = ", ".join(
        f"{name}={source_bins[name]}" for name in bin_order if source_bins[name]
    )
    downscales.sort(reverse=True, key=lambda row: row[0])
    downscale_preview = []
    for ratio, image, source_size, bucket_key in downscales[:limit]:
        name = Path(image).name if image is not None else "?"
        downscale_preview.append(
            f"{name}: {_format_size(source_size)} -> {_format_size(bucket_key)} ({ratio:.2f}x)"
        )

    lines = [
        f"[BucketReport:{label}] samples={len(rows)}, buckets={len(bucket_counts)}",
        f"  assigned buckets: {bucket_preview}",
        f"  source size bins: {bin_preview or 'none'}",
    ]
    if downscale_preview:
        lines.append("  largest downscales: " + "; ".join(downscale_preview))
    if upscales:
        upscales.sort(reverse=True, key=lambda row: row[0])
        upscale_preview = []
        for scale, image, source_size, bucket_key in upscales[:limit]:
            name = Path(image).name if image is not None else "?"
            upscale_preview.append(
                f"{name}: {_format_size(source_size)} -> {_format_size(bucket_key)} (放大 {scale:.2f}x)"
            )
        lines.append(
            f"  ⚠ UPSCALED {len(upscales)} 张（no_upscale 下不应出现：该长宽比在桶网格无匹配桶被放大）。"
            f"修复：bucket_max_aspect_ratio 提到覆盖这些图，或 bucket_reso_steps=64 补桶，或数据集端裁到更接近的 AR: "
            + "; ".join(upscale_preview)
        )
    return "\n".join(lines)


class ImageDataset(Dataset):
    """图像数据集，支持 JSON / TXT caption 与频率均衡的 tag dropout。"""
    EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def __init__(self, data_dir, resolution=1024, bucket_mgr=None,
                 shuffle_caption=False, keep_tokens=0, flip_augment=False,
                 tag_dropout=0.0, prefer_json=True, caption_override=None,
                 tag_dropout_overrides=None,
                 freq_balanced_dropout_strength=0.0,
                 fit_packed=False, fit_max_tokens=65536,
                 fit_warn_tokens=16384, fit_min_tokens=16,
                 fit_patch_size=2, fit_vae_downsample=8,
                 fit_over_budget_strategy="fail", fit_align_mode="pad",
                 alpha_handling="none", alpha_background="neutral",
                 alpha_threshold=0.01, navit_ms_token_ladder=None):
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.bucket_mgr = bucket_mgr
        self.shuffle_caption = shuffle_caption
        self.keep_tokens = keep_tokens
        self.flip_augment = flip_augment
        self.tag_dropout = tag_dropout
        # 按 tag 覆盖丢弃概率（仅 TXT caption 路径）：{"close-up": 0.5} 表示该 tag 以
        # 0.5 概率被丢（替代通用 tag_dropout，不叠加）。用途：解开"特征绑定到条件 tag"
        # ——如纹理特写集的纹理被 close-up 门控、正常尺度不表达时，提高该 tag 的
        # dropout 让特征向无条件/触发词泄漏。keep_tokens 内的 tag 不受影响。
        self.tag_dropout_overrides = {
            str(k).strip().lower(): float(v)
            for k, v in (tag_dropout_overrides or {}).items()
        }
        self.prefer_json = prefer_json
        self.caption_override = caption_override  # 正则集：统一 caption，如 "1girl, solo"
        self.fit_packed = bool(fit_packed)
        self.fit_max_tokens = int(fit_max_tokens or 65536)
        self.fit_warn_tokens = int(fit_warn_tokens or 0)
        self.fit_min_tokens = int(fit_min_tokens or 0)
        self.fit_patch_size = int(fit_patch_size or 2)
        self.fit_vae_downsample = int(fit_vae_downsample or 8)
        self.fit_over_budget_strategy = str(fit_over_budget_strategy or "fail").lower()
        self.fit_align_mode = str(fit_align_mode or "pad").lower()
        self.alpha_handling = str(alpha_handling or "none").lower()
        self.alpha_background = str(alpha_background or "neutral").lower()
        self.alpha_threshold = float(alpha_threshold if alpha_threshold is not None else 0.01)
        # navit 多尺度阶梯（navit_multiscale）：每图追加低 token 档的等比缩放副本，
        # 作为正式数据集条目参与打包（填满预算 + 缓解"只见过原生尺度"的分布偏移）。
        # 仅在 fit_packed（navit 原生定尺寸复用该路径）下生效；空/None = 关闭（默认）。
        self.navit_ms_token_ladder = sorted(
            {int(x) for x in (navit_ms_token_ladder or []) if int(x) > 0}
        )
        # ★ v5 ② frequency-balanced tag dropout
        # 0 = 关闭；>0 启用。在数据集 init 时统计 tag 频率，对在数据集中过度共现的 tag 额外提高 dropout
        # 概率，强迫模型把"风格"与"高频共现 tag"解耦。完全数据驱动，自动适配任何画师。
        self.freq_balanced_dropout_strength = float(freq_balanced_dropout_strength or 0.0)
        self.tag_freq = {}  # tag -> frequency in [0, 1]，scan 之后填充

        # 尝试导入 caption_utils（直接导入避开 __init__.py）
        self.caption_utils = None
        if prefer_json:
            try:
                # 直接加载 caption_utils.py。该文件在 AnimaLoraToolkit/utils/caption_utils.py。
                # trainer 包位于 AnimaLoraToolkit/trainer/ —— parent.parent 是仓库根，
                # parent 是 AnimaLoraToolkit/。两级合一指向 utils/caption_utils.py。
                utils_path = Path(__file__).resolve().parent.parent / "utils" / "caption_utils.py"
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

        # ★ JSON 预 normalize：__getitem__ 时只做 build（shuffle+dropout），不再每次 load+parse 文件。
        # 在 num_workers=0（Windows 默认）时，这一项能把数据 prep CPU 时间显著降下来。
        if json_count > 0 and self.caption_utils is not None:
            self._pre_normalize_json_captions()

        # 用与 __getitem__ 完全一致的 PIL 路径填充 bucket_key
        self._finalize_bucket_keys()
        # navit 多尺度阶梯：在 bucket_for_index / token_count_for_index 物化之前展开，
        # 使副本与原生条目在所有 per-index 结构里同等存在（打包器/缓存按索引对齐）。
        if self.fit_packed and self.navit_ms_token_ladder:
            self._expand_multiscale_samples()
        self.bucket_for_index = [s["bucket_key"] for s in self.samples]
        self.token_count_for_index = [int(s.get("token_count", 0) or 0) for s in self.samples]
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
        so BucketBatchSampler can reliably group same-shape tensors. Caches per unique img path.

        ★ 大数据集（几万张）时 PIL.Image.open 在 OS 层每张都要 open/close fd，
          init 阶段可能卡几十秒。加进度日志让用户知道还活着；并把 unique image set 预先收集，
          避免 repeats 重复样本重复 open。
        """
        from PIL import Image as _PILImage
        # 先收集 unique image paths（_scan 里同一张图可能被 repeats 次添加）
        unique_imgs = []
        seen = set()
        for sample in self.samples:
            ip = sample["image"]
            if ip not in seen:
                seen.add(ip)
                unique_imgs.append(ip)
        n_unique = len(unique_imgs)

        cache = {}
        for i, img_path in enumerate(unique_imgs):
            if self.fit_packed:
                try:
                    img = _PILImage.open(img_path)
                    w, h = img.width, img.height
                    try:
                        img.close()
                    except Exception:
                        pass
                    plan = plan_native_fit_image(
                        w,
                        h,
                        max_tokens=self.fit_max_tokens,
                        patch_size=self.fit_patch_size,
                        vae_downsample=self.fit_vae_downsample,
                        align_mode=self.fit_align_mode,
                        over_budget_strategy=self.fit_over_budget_strategy,
                    )
                    cache[img_path] = ((plan.height, plan.width), (h, w), plan)
                    if self.fit_warn_tokens > 0 and plan.token_count > self.fit_warn_tokens:
                        logger.warning(
                            "[FiT] %s -> %dx%d, tokens=%d exceeds fit_warn_tokens=%d",
                            img_path,
                            plan.width,
                            plan.height,
                            plan.token_count,
                            self.fit_warn_tokens,
                        )
                    if self.fit_min_tokens > 0 and plan.token_count < self.fit_min_tokens:
                        logger.warning(
                            "[FiT] %s -> %dx%d, tokens=%d below fit_min_tokens=%d",
                            img_path,
                            plan.width,
                            plan.height,
                            plan.token_count,
                            self.fit_min_tokens,
                        )
                except Exception as e:
                    logger.warning(f"[FiT] 无法读取 {img_path}: {e}")
                    if self.fit_over_budget_strategy == "skip":
                        cache[img_path] = None
                    else:
                        raise
            elif self.bucket_mgr is None:
                cache[img_path] = ((self.resolution, self.resolution), (self.resolution, self.resolution), None)
            else:
                try:
                    img = _PILImage.open(img_path)
                    w, h = img.width, img.height
                    try:
                        img.close()
                    except Exception:
                        pass
                    bw, bh = self.bucket_mgr.get_bucket(w, h)
                    if getattr(self.bucket_mgr, "no_upscale", False):
                        _cover = max(bw / max(1, w), bh / max(1, h))
                        if _cover > 1.0 + 1e-6:
                            logger.warning(
                                "[no_upscale] %s 源 %dx%d 无可容纳的桶，被放大 %.2fx 到 %dx%d"
                                "（该长宽比在桶网格无匹配桶）。建议：bucket_max_aspect_ratio 提到覆盖此图，"
                                "或 bucket_reso_steps=64 补桶，或数据集端把该图裁到 AR≤%.2f。",
                                Path(img_path).name, w, h, _cover, bw, bh,
                                float(getattr(self.bucket_mgr, "max_aspect_ratio", 2.0)),
                            )
                    cache[img_path] = ((bh, bw), (h, w), None)  # (bucket h,w), (source h,w), fit plan
                except Exception as e:
                    logger.warning(f"[bucket_key] 无法读取 {img_path}: {e}，回退到 ({self.resolution},{self.resolution})")
                    cache[img_path] = ((self.resolution, self.resolution), (0, 0), None)
            # 每 1000 张或最后一张时打印进度，让用户知道 init 没卡死
            if n_unique >= 2000 and ((i + 1) % 1000 == 0 or i == n_unique - 1):
                logger.info(f"  bucket_key 解析进度: {i + 1}/{n_unique}")

        if self.fit_packed:
            before = len(self.samples)
            self.samples = [sample for sample in self.samples if cache.get(sample["image"]) is not None]
            skipped = before - len(self.samples)
            if skipped:
                logger.warning("[FiT] skipped %d samples due to fit_over_budget_strategy=skip", skipped)

        for sample in self.samples:
            bucket_key, source_size, fit_plan = cache[sample["image"]]
            sample["bucket_key"] = bucket_key
            sample["source_size"] = source_size
            sample["fit_plan"] = fit_plan
            sample["token_count"] = int(fit_plan.token_count) if fit_plan is not None else 0

    def _expand_multiscale_samples(self):
        """navit 多尺度阶梯：为每个原生 FiT 条目追加低 token 档的等比缩放副本。

        副本是普通样本（自己的 fit_plan / bucket_key / token_count + ``ms_tokens_target``
        标记），经打包器与其它异尺寸图混包——确定性展开（每图每档每 epoch 恰见一次），
        而非随机填充：语义可复现、可归因（见 docs/navit-packing.md）。只降不升采样：
        源图 token 数 ≤ 目标档的条目直接跳过该档。caption 与原生共享（tag caption 不含
        分辨率语义，无训练/推理不一致）。
        """
        from collections import Counter
        added = []
        per_target = Counter()
        for sample in self.samples:
            plan = sample.get("fit_plan")
            if plan is None:
                continue
            src_h, src_w = sample.get("source_size", (0, 0))
            for tgt in self.navit_ms_token_ladder:
                ms_plan = plan_multiscale_copy(
                    src_w, src_h, tgt,
                    patch_size=self.fit_patch_size,
                    vae_downsample=self.fit_vae_downsample,
                )
                if ms_plan is None or ms_plan.token_count >= plan.token_count:
                    continue
                ms = dict(sample)
                ms["fit_plan"] = ms_plan
                ms["bucket_key"] = (ms_plan.height, ms_plan.width)
                ms["token_count"] = int(ms_plan.token_count)
                ms["ms_tokens_target"] = int(tgt)
                added.append(ms)
                per_target[int(tgt)] += 1
        if added:
            self.samples.extend(added)
        logger.info(
            "[navit-multiscale] token 阶梯 %s：追加 %d 个缩放副本（%s），数据集 %d → %d 条",
            self.navit_ms_token_ladder, len(added),
            ", ".join(f"≤{t}tok×{n}" for t, n in sorted(per_target.items())) or "无",
            len(self.samples) - len(added), len(self.samples),
        )

    def bucket_report(self, limit=12, label="dataset"):
        return format_bucket_report(self.samples, limit=limit, label=label)

    def _pre_normalize_json_captions(self):
        """一次性把所有 JSON caption 加载 + normalize，缓存到 sample["normalized_json"]。

        旧实现 `_process_caption_json` 在每次 __getitem__ 都做 load_json + normalize + build；
        normalize 是结构变换，对同一文件每次产出都一样，没必要重复做。
        load_json 也省下重复 IO（同一 sample dict 会被 repeats 次添加，每次都读盘）。

        预 normalize 后 __getitem__ 走 build_only 路径（只做 shuffle/dropout 这种随机的部分）。
        """
        if self.caption_utils is None:
            return
        cache = {}  # json_path -> normalized dict（同一 JSON 多次出现只算一次）
        ok = 0
        fail = 0
        for sample in self.samples:
            jp = sample.get("json_path")
            if jp is None:
                continue
            cached = cache.get(jp)
            if cached is None:
                try:
                    raw = self.caption_utils["load_json"](jp)
                    if raw is None:
                        fail += 1
                        continue
                    if "tags" in raw and "meta" in raw:
                        normalized = raw
                    else:
                        normalized = self.caption_utils["normalize"](raw)
                    cache[jp] = normalized
                    cached = normalized
                    ok += 1
                except Exception as e:
                    logger.warning(f"JSON pre-normalize 失败 {jp}: {e}")
                    fail += 1
                    continue
            sample["normalized_json"] = cached
        logger.info(f"[dataset] JSON 预 normalize 完成: {ok} unique, {fail} 失败")

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
            for t in set(tags):
                cnt[t] += 1

        if total_imgs == 0:
            logger.warning("[freq_balanced] 没有可统计的 TXT caption，禁用频率加权 dropout。")
            self.tag_freq = {}
            return

        self.tag_freq = {t: c / total_imgs for t, c in cnt.items()}
        top = sorted(self.tag_freq.items(), key=lambda kv: kv[1], reverse=True)[:10]
        top_str = ", ".join(f"{t}={f:.2f}" for t, f in top)
        logger.info(
            f"[freq_balanced] 已统计 {len(self.tag_freq)} 个 tag 的频率 "
            f"(strength={self.freq_balanced_dropout_strength:.2f}). Top10: {top_str}"
        )

    def _scan(self):
        samples = []
        dir_repeat_stats = {}  # dir_name -> repeat 次数（用于诊断日志）
        unique_images = 0
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
                    dir_repeat_stats[parent_name] = repeats

            sample = {"image": img_path}

            # 优先查找 JSON
            json_path = img_path.with_suffix(".json")
            if self.prefer_json and json_path.exists():
                sample["json_path"] = json_path
                sample["txt_path"] = None
            else:
                txt_path = img_path.with_suffix(".txt")
                if not txt_path.exists():
                    txt_path = img_path.with_suffix(".caption")
                if not txt_path.exists():
                    continue
                sample["json_path"] = None
                sample["txt_path"] = txt_path

            sample["bucket_key"] = None
            # ★ JSON 预 normalize 槽：__getitem__ 时直接读 dict 而非重新解析文件
            sample["normalized_json"] = None
            unique_images += 1

            for _ in range(repeats):
                samples.append(dict(sample))

        # ★ 旧实现：如果用户同时使用了 `10_xxx` 目录命名 + 顶层 `repeats: 10` YAML 参数，
        # 实际曝光会 ×100，用户无从察觉。这里明确 log 出 dir-level repeats 和 unique image 数。
        if dir_repeat_stats:
            preview = ", ".join(f"{name}×{r}" for name, r in list(dir_repeat_stats.items())[:5])
            logger.info(
                f"[dataset] 检测到 {len(dir_repeat_stats)} 个目录使用 kohya 风格 repeat 前缀: {preview}"
                + (" ..." if len(dir_repeat_stats) > 5 else "")
            )
            logger.info(
                f"[dataset] 唯一图片: {unique_images}, dir-level repeat 后样本: {len(samples)}。"
                f"注意：若 YAML 顶层再设 `repeats: N`，最终曝光 = dir_repeats × N，请确认是否符合预期。"
            )
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
        overrides = getattr(self, "tag_dropout_overrides", None) or {}
        freq_strength = float(getattr(self, "freq_balanced_dropout_strength", 0.0) or 0.0)
        tag_freq = getattr(self, "tag_freq", {}) or {}

        if rest and (dropout > 0.0 or overrides or (freq_strength > 0.0 and tag_freq)):
            survivors = []
            for t in rest:
                # 通用 dropout；tag_dropout_overrides 命中时按覆盖概率（替代非叠加）
                p_drop = overrides.get(t.strip().lower(), dropout)
                if p_drop > 0.0 and random.random() < p_drop:
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

    def _process_caption_json(self, json_path, normalized=None):
        """处理 JSON caption: 分类 shuffle。

        若调用方传入 `normalized`（init 时 _pre_normalize_json_captions 算好的），跳过 load + normalize；
        否则按旧路径 load + normalize（兼容外部直接调用，如 CachedLatentDataset）。
        """
        if self.caption_utils is None:
            return None

        try:
            if normalized is None:
                raw_json = self.caption_utils["load_json"](json_path)
                if raw_json is None:
                    return None

                if "tags" in raw_json and "meta" in raw_json:
                    normalized = raw_json
                else:
                    normalized = self.caption_utils["normalize"](raw_json)

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

    def _alpha_background_rgb(self):
        if self.alpha_background == "white":
            return (255, 255, 255)
        if self.alpha_background == "black":
            return (0, 0, 0)
        return (127, 127, 127)

    def _load_rgb_and_alpha_mask(self, image_path):
        from PIL import Image

        raw = Image.open(image_path)
        has_alpha = (
            "A" in raw.getbands()
            or raw.mode in ("LA", "PA")
            or raw.info.get("transparency") is not None
        )
        if self.alpha_handling == "mask" and has_alpha:
            rgba = raw.convert("RGBA")
            rgb = Image.new("RGB", rgba.size, self._alpha_background_rgb())
            rgb.paste(rgba.convert("RGB"), mask=rgba.getchannel("A"))
            alpha = rgba.getchannel("A")
            return rgb, alpha
        return raw.convert("RGB"), None

    def __getitem__(self, idx):
        import numpy as np
        from PIL import Image
        sample = self.samples[idx]
        img, alpha = self._load_rgb_and_alpha_mask(sample["image"])

        caption = None
        if self.caption_override is not None:
            caption = self.caption_override
        elif sample.get("json_path"):
            # 使用预 normalized 缓存（init 阶段算好），跳过每次的 load+normalize 开销
            caption = self._process_caption_json(
                sample["json_path"], normalized=sample.get("normalized_json"),
            )

        if caption is None and sample.get("txt_path"):
            caption = sample["txt_path"].read_text(encoding="utf-8").strip()
            caption = self._process_caption_txt(caption)

        if caption is None:
            caption = ""

        if self.fit_packed:
            plan = sample.get("fit_plan")
            if plan is None:
                raise RuntimeError(f"missing FiT image plan for {sample['image']}")

            if self.flip_augment and random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            if plan.was_resized:
                # navit 多尺度副本：等比缩小到覆盖规划尺寸再中心裁剪（与 ARB 桶路径同法，
                # 每轴至多裁 <16px 的对齐余量）。裁完恰为 plan 尺寸 → 下方 floor 裁剪 /
                # padding 均为 no-op，mask 全 1（plan.source_* == plan 尺寸保证这一点）。
                _cover = max(plan.width / img.width, plan.height / img.height)
                _rw = max(plan.width, int(math.ceil(img.width * _cover)))
                _rh = max(plan.height, int(math.ceil(img.height * _cover)))
                _left = (_rw - plan.width) // 2
                _top = (_rh - plan.height) // 2
                img = img.resize((_rw, _rh), Image.LANCZOS).crop(
                    (_left, _top, _left + plan.width, _top + plan.height))
                if alpha is not None:
                    alpha = alpha.resize((_rw, _rh), Image.LANCZOS).crop(
                        (_left, _top, _left + plan.width, _top + plan.height))

            if self.fit_align_mode == "floor":
                img = img.crop((0, 0, min(img.width, plan.width), min(img.height, plan.height)))
                if alpha is not None:
                    alpha = alpha.crop((0, 0, min(alpha.width, plan.width), min(alpha.height, plan.height)))
            padded = Image.new("RGB", (plan.width, plan.height), (127, 127, 127))
            padded.paste(img, (0, 0))

            arr = np.array(padded).astype(np.float32) / 127.5 - 1.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1)
            mask = torch.zeros((1, plan.height, plan.width), dtype=torch.float32)
            valid_h = min(plan.source_height, plan.height)
            valid_w = min(plan.source_width, plan.width)
            mask[:, :valid_h, :valid_w] = 1.0
            if alpha is not None:
                alpha_arr = np.array(alpha).astype(np.float32) / 255.0
                alpha_mask = torch.from_numpy((alpha_arr > self.alpha_threshold).astype(np.float32))
                mask[:, :valid_h, :valid_w] *= alpha_mask[:valid_h, :valid_w].unsqueeze(0)
            return {
                "pixel_values": tensor,
                "pixel_mask": mask,
                "caption": caption,
                "image": str(sample["image"]),
                "token_count": int(plan.token_count),
                "fit_plan": plan,
            }

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


def collect_dataset_captions(dataset, max_count: int = 0):
    """抽出数据集里每张图的 **原始** caption 文本，供训练期预览采样随机取用。

    与 `ImageDataset.__getitem__` 的 caption 路径的差别（有意为之）：
      - 不做 shuffle_caption / tag_dropout / caption_override 之外的任何随机化。
        预览图的用途是"这条 caption 现在被还原成什么样"，如果每次取到的还是被 dropout
        过的残缺 caption，不同 step 的预览就失去可比性，也看不出真实条件下的还原度。
      - JSON caption 走 `caption_utils.build`，但 shuffle_* 全 False、tag_dropout=0。
      - 按图片路径去重：目录名 `10_xxx` 前缀会把同一张图重复进 `samples`，
        不去重会让这些图在随机抽样里被加权。

    Args:
        dataset: `ImageDataset` 实例（只用到 `.samples` / `.caption_override` /
            `.caption_utils`，其它包装类请传底层的 base dataset）。
        max_count: >0 时最多返回前 N 条（按扫描顺序截断）。0 = 不限。

    Returns:
        list[str]：去重后的 caption 列表，空 caption 已剔除；顺序稳定（跟随扫描顺序）。
    """
    samples = getattr(dataset, "samples", None) or []
    override = getattr(dataset, "caption_override", None)
    cap_utils = getattr(dataset, "caption_utils", None)

    captions = []
    seen_images = set()
    for sample in samples:
        img_key = str(sample.get("image", ""))
        if img_key and img_key in seen_images:
            continue
        seen_images.add(img_key)

        text = None
        if override is not None:
            text = override
        elif sample.get("json_path") and cap_utils is not None:
            try:
                normalized = sample.get("normalized_json")
                if normalized is None:
                    raw_json = cap_utils["load_json"](sample["json_path"])
                    if raw_json is not None:
                        if "tags" in raw_json and "meta" in raw_json:
                            normalized = raw_json
                        else:
                            normalized = cap_utils["normalize"](raw_json)
                if normalized is not None:
                    text = cap_utils["build"](
                        normalized,
                        shuffle_appearance=False,
                        shuffle_tags=False,
                        shuffle_environment=False,
                        tag_dropout=0.0,
                    )
            except Exception as e:  # 单条坏 JSON 不该让整个 pool 构建失败
                logger.warning(f"[sample_prompts] JSON caption 读取失败 {sample.get('json_path')}: {e}")
                text = None

        if text is None and sample.get("txt_path"):
            try:
                text = Path(sample["txt_path"]).read_text(encoding="utf-8").strip()
            except Exception as e:
                logger.warning(f"[sample_prompts] TXT caption 读取失败 {sample.get('txt_path')}: {e}")
                text = None

        text = (text or "").strip()
        if not text:
            continue
        captions.append(text)
        if max_count > 0 and len(captions) >= max_count:
            break

    return captions


class RepeatDataset(Dataset):
    """Kohya 风格数据集重复"""
    def __init__(self, dataset, repeats=1):
        self.dataset = dataset
        self.repeats = max(1, int(repeats))
        counts = getattr(dataset, "token_count_for_index", None)
        if counts:
            self.token_count_for_index = [int(counts[i % len(counts)]) for i in range(len(self))]

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

        self.bucket_for_index = self._build_bucket_for_index()
        self.token_count_for_index = self._build_token_count_for_index()

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

    def _build_token_count_for_index(self):
        counts = []
        main_counts = getattr(self.main_dataset, "token_count_for_index", None)
        if main_counts:
            main_base_len = len(main_counts)
            for idx in range(self._main_len):
                counts.append(int(main_counts[idx % main_base_len]))
        else:
            counts.extend([0] * self._main_len)
        reg_counts = getattr(self.reg_dataset, "token_count_for_index", None)
        if reg_counts:
            reg_base_len = len(reg_counts)
            for idx in range(self._reg_len):
                counts.append(int(reg_counts[idx % reg_base_len]))
        else:
            counts.extend([0] * self._reg_len)
        return counts

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
    def __init__(self, dataset, batch_size, drop_last=False, shuffle=True, seed=42,
                 effective_batch_size=0, reference_batch_size=0):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.effective_batch_size = int(effective_batch_size or 0)
        if self.effective_batch_size <= 0:
            self.effective_batch_size = 0
        self.reference_batch_size = int(reference_batch_size or 0)
        if self.reference_batch_size <= 0:
            self.reference_batch_size = 0
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self.accumulation_offset = 0
        self._reference_counts_for_epoch = []
        self._bucket_keys = self._build_keys(dataset)

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
        # 让用户在训练开始时就直观看到 ARB 分桶对样本的影响（丢弃数或小 batch 数）
        from collections import Counter
        _counts = Counter(tuple(k) if k is not None else (0, 0) for k in self._bucket_keys)
        _bs = self.batch_size
        if self.drop_last:
            _dropped = sum(n % _bs for n in _counts.values())
            if _dropped > 0:
                logger.warning(
                    "[BucketBatchSampler] drop_last=True：将丢弃 %d/%d 张图片"
                    "（来自不满 batch_size=%d 的余数）。设 bucket_drop_last=false 可让所有图都参训。",
                    _dropped, len(self._bucket_keys), _bs,
                )
        else:
            _small_batches = sum(1 for n in _counts.values() if n % _bs != 0)
            logger.info(
                "[BucketBatchSampler] drop_last=False：所有 %d 张图都将参训；"
                "其中 %d 个桶会产生 1 个小于 batch_size=%d 的余数 batch。",
                len(self._bucket_keys), _small_batches, _bs,
            )
        if self.effective_batch_size:
            logger.info(
                "[BucketBatchSampler] sample-window accumulation enabled: "
                "effective_batch_size=%d, native batch_size<=%d.",
                self.effective_batch_size, self.batch_size,
            )
        if self.reference_batch_size:
            logger.info(
                "[BucketBatchSampler] reference progress enabled: "
                "reference_batch_size=%d.",
                self.reference_batch_size,
            )

        # 预计算 per-bucket 批数（drop_last 在每个桶内独立生效）。
        # 旧实现 `n // bs` 在多桶 + drop_last 时会高估批数，进而把 cosine 调度器的 T_max 设错。
        self._total_batches = self._compute_total_batches()

    def _compute_total_batches(self):
        total = 0
        pending = self.accumulation_offset
        for batch in self._native_batches():
            split_batches, pending = self._split_for_accumulation_window(batch, pending)
            total += len(split_batches)
        return total

    def _native_batches(self):
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

    def _split_for_accumulation_window(self, batch, pending):
        """Split a same-bucket native batch before forward to hit sample windows.

        The returned sub-batches preserve order and contain each input index
        exactly once. If `effective_batch_size` is disabled, the original batch
        is returned unchanged.
        """
        if not self.effective_batch_size:
            return [batch], pending

        eff = self.effective_batch_size
        pending = int(pending) % eff
        parts = []
        pos = 0
        while pos < len(batch):
            remaining = eff - pending
            take = min(len(batch) - pos, remaining)
            parts.append(batch[pos:pos + take])
            pos += take
            pending += take
            if pending == eff:
                pending = 0
        return parts, pending

    def _reference_boundaries_for_bucket(self, bucket_len):
        if not self.reference_batch_size:
            return []
        ref_bs = self.reference_batch_size
        full = bucket_len // ref_bs
        boundaries = [ref_bs * i for i in range(1, full + 1)]
        if not self.drop_last and bucket_len % ref_bs:
            boundaries.append(bucket_len)
        return boundaries

    def _native_batches_with_reference_boundaries(self):
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
            reference_boundaries = self._reference_boundaries_for_bucket(len(indices))
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                local_boundaries = [
                    boundary - i
                    for boundary in reference_boundaries
                    if i < boundary <= i + len(batch)
                ]
                yield batch, local_boundaries

    def _build_keys(self, dataset):
        """Build per-outer-index bucket keys。

        ★ 旧实现对每个 outer idx 都递归 walk wrapper（O(N × depth)）。
        优化：一次 walk 找到 leaf 的 bucket_for_index 列表，然后批量索引 → O(N + walk_depth)。
        对 100K samples + 3 层 wrap 提升约 2-3×。

        无法静态决定 leaf 时（如 MergedDataset(reg, main) 两个 leaf 不同），回退到 per-idx walk。
        """
        n = len(dataset)
        # 尝试 fast path：找到唯一 leaf 的 bucket_for_index
        leaf = self._find_unique_leaf(dataset)
        if leaf is not None:
            leaf_keys = getattr(leaf, "bucket_for_index", None)
            leaf_len = len(leaf_keys) if leaf_keys is not None else 0
            if leaf_len > 0:
                # outer idx 通过 % leaf_len 直接映射（RepeatDataset 的语义）
                return [leaf_keys[i % leaf_len] for i in range(n)]

        # 回退：复用旧的 walk 实现（MergedDataset 路径）
        keys = [None] * n
        for i in range(n):
            keys[i] = self._lookup(dataset, i)
        return keys

    def _find_unique_leaf(self, d):
        """Walk down a single chain（RepeatDataset/CachedLatentDataset）的 leaf。

        MergedDataset 有两条分支，不在 fast path 范围；返回 None 由 _lookup 处理。
        """
        # MergedDataset：拒绝
        if getattr(d, "main_dataset", None) is not None and getattr(d, "reg_dataset", None) is not None:
            return None
        cur = d
        # 已有 bucket_for_index 就是 leaf
        bfi = getattr(cur, "bucket_for_index", None)
        if bfi is not None and len(bfi) > 0:
            return cur
        # 一层层往下找
        for _ in range(10):  # 防御 max depth
            inner = getattr(cur, "dataset", None)
            if inner is None or inner is cur or isinstance(inner, list):
                inner = getattr(cur, "base_dataset", None)
            if inner is None or inner is cur:
                return None
            cur = inner
            # MergedDataset 出现在中间层就不行
            if getattr(cur, "main_dataset", None) is not None and getattr(cur, "reg_dataset", None) is not None:
                return None
            bfi = getattr(cur, "bucket_for_index", None)
            if bfi is not None and len(bfi) > 0:
                return cur
        return None

    def _lookup(self, d, idx):
        """Resolve the bucket key for a given outer index by walking dataset wrappers.

        Priority: MergedDataset routing → RepeatDataset (.dataset) → leaf bucket_for_index
        → CachedLatentDataset (.base_dataset).
        """
        main = getattr(d, "main_dataset", None)
        reg = getattr(d, "reg_dataset", None)
        if main is not None and reg is not None:
            ml = getattr(d, "_main_len", len(main))
            if idx < ml:
                return self._lookup(main, idx)
            return self._lookup(reg, idx - ml)
        inner = getattr(d, "dataset", None)
        if inner is not None and inner is not d and not isinstance(inner, list):
            try:
                return self._lookup(inner, idx % len(inner))
            except TypeError:
                pass
        bfi = getattr(d, "bucket_for_index", None)
        if bfi is not None and len(bfi) > 0:
            return bfi[idx % len(bfi)]
        inner = getattr(d, "base_dataset", None)
        if inner is not None and inner is not d:
            return self._lookup(inner, idx % len(inner))
        return None

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def set_accumulation_offset(self, pending_samples):
        if self.effective_batch_size:
            self.accumulation_offset = int(pending_samples) % self.effective_batch_size
        else:
            self.accumulation_offset = 0

    def reference_batches_for_batch_index(self, batch_idx):
        try:
            return int(self._reference_counts_for_epoch[int(batch_idx)])
        except (IndexError, TypeError, ValueError):
            return 0

    def __len__(self):
        if self.effective_batch_size:
            return self._compute_total_batches()
        return self._total_batches

    def __iter__(self):
        pending = self.accumulation_offset
        self._reference_counts_for_epoch = []
        for batch, reference_boundaries in self._native_batches_with_reference_boundaries():
            split_batches, pending = self._split_for_accumulation_window(batch, pending)
            split_start = 0
            for split_batch in split_batches:
                split_end = split_start + len(split_batch)
                reference_count = sum(
                    1 for boundary in reference_boundaries
                    if split_start < boundary <= split_end
                )
                self._reference_counts_for_epoch.append(reference_count)
                split_start = split_end
                yield split_batch


class FitTokenBatchSampler:
    """Batch native FiT samples by token count to avoid wasteful padding."""

    def __init__(self, dataset, batch_size, max_tokens_per_batch=0, shuffle=True, seed=42):
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        self.max_tokens_per_batch = max(0, int(max_tokens_per_batch or 0))
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        token_counts = self._build_token_counts(dataset)
        self.token_counts = [max(1, int(v)) for v in token_counts]

    def _build_token_counts(self, dataset):
        counts = getattr(dataset, "token_count_for_index", None)
        if counts is not None and len(counts) > 0:
            try:
                dataset_len = len(dataset)
            except TypeError:
                dataset_len = len(counts)
            return [int(counts[i % len(counts)]) for i in range(dataset_len)]
        return [int(self._lookup_token_count(dataset, i) or 0) for i in range(len(dataset))]

    def _lookup_token_count(self, d, idx):
        main = getattr(d, "main_dataset", None)
        reg = getattr(d, "reg_dataset", None)
        if main is not None and reg is not None:
            ml = getattr(d, "_main_len", len(main))
            if idx < ml:
                return self._lookup_token_count(main, idx)
            return self._lookup_token_count(reg, idx - ml)
        inner = getattr(d, "dataset", None)
        if inner is not None and inner is not d and not isinstance(inner, list):
            return self._lookup_token_count(inner, idx % len(inner))
        counts = getattr(d, "token_count_for_index", None)
        if counts is not None and len(counts) > 0:
            return int(counts[idx % len(counts)])
        inner = getattr(d, "base_dataset", None)
        if inner is not None and inner is not d:
            return self._lookup_token_count(inner, idx % len(inner))
        return 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.token_counts)))
        indices.sort(key=lambda i: (self.token_counts[i], i))
        if self.shuffle:
            # Keep coarse token locality, but vary order within similarly sized neighborhoods.
            chunks = [indices[i:i + self.batch_size * 8] for i in range(0, len(indices), self.batch_size * 8)]
            for chunk in chunks:
                rng.shuffle(chunk)
            rng.shuffle(chunks)
            indices = [idx for chunk in chunks for idx in chunk]

        batch = []
        batch_max = 0
        for idx in indices:
            tok = self.token_counts[idx]
            next_max = max(batch_max, tok)
            next_len = len(batch) + 1
            over_count = next_len > self.batch_size
            over_tokens = (
                self.max_tokens_per_batch > 0
                and next_max * next_len > self.max_tokens_per_batch
                and batch
            )
            if over_count or over_tokens:
                yield batch
                batch = []
                batch_max = 0
            if self.max_tokens_per_batch > 0 and tok > self.max_tokens_per_batch:
                logger.warning(
                    "[FiT] sample index %d has %d tokens, above fit_max_tokens_per_batch=%d; "
                    "it will be trained as a single-sample batch.",
                    idx,
                    tok,
                    self.max_tokens_per_batch,
                )
            batch.append(idx)
            batch_max = max(batch_max, tok)
        if batch:
            yield batch

    def __len__(self):
        total = 0
        indices = sorted(range(len(self.token_counts)), key=lambda i: (self.token_counts[i], i))
        batch = []
        batch_max = 0
        for idx in indices:
            tok = self.token_counts[idx]
            next_max = max(batch_max, tok)
            next_len = len(batch) + 1
            over_count = next_len > self.batch_size
            over_tokens = (
                self.max_tokens_per_batch > 0
                and next_max * next_len > self.max_tokens_per_batch
                and batch
            )
            if over_count or over_tokens:
                total += 1
                batch = []
                batch_max = 0
            batch.append(idx)
            batch_max = max(batch_max, tok)
        if batch:
            total += 1
        return total


def navit_pack_costs(token_counts, cost_lambda=0.0, cost_ref_tokens=0):
    """Per-image *cost* used as the packing "volume" — token count by default.

    Why a cost instead of the raw token count: a pack's step time is **not** linear in
    its summed tokens. The attention term is quadratic in each image's own sequence
    length, so at a fixed ΣN a pack of few large images is far more expensive than one
    of many small images. Measured on the training card (H20, Krea2 12B, one real
    ``SingleStreamBlock``, ``tests/diag_navit_speed.py`` S5): at ΣN = 55778 the
    fwd+bwd time is 3960 ms at G=1 versus 1719 ms at G=16 — a 2.3× spread at *identical*
    token count. Fitting ``t = a·ΣN + b·ΣN_i²`` gives R²=0.9999 with
    ``λ = b/a = 2.742e-05``; independently fitting whole-step times from a real run's
    ``stage_timing.csv`` gives ``λ = 3.279e-05`` (R²=0.972) — the two agree within 16%.
    (Same semi-empirical form as KnapFormer, arXiv 2508.06001, whose workload model is
    ``k·(24·L·d² + 4γ·L²·d)``.)

    So with ``cost_lambda > 0`` the packer budgets on::

        cost(n) = n · (1 + λ·n) / (1 + λ·n_ref)

    The ``n_ref`` normalisation keeps an image of the *reference* size costing exactly
    its token count, so a dataset of uniformly-sized images keeps its current pack
    capacity and only the size *spread* is repriced: images larger than the reference
    cost more (fewer per pack ⇒ no step-time / VRAM spike), smaller ones cost less
    (more per pack ⇒ higher throughput). Without it, enabling λ would silently shrink
    every pack, which reads as a regression rather than a rebalance.

    ``cost_lambda=0`` (default) returns the token counts unchanged, so every existing
    config packs byte-identically.

    Args:
        token_counts: per-index token counts.
        cost_lambda: λ in equivalent-tokens per token. 0 disables (default).
        cost_ref_tokens: n_ref; 0 = auto (median of the positive token counts).

    Returns:
        (costs, budget_scale, ref) — ``costs`` is a float list aligned with
        ``token_counts``; ``budget_scale`` is 1.0 (the budget stays in token units by
        construction); ``ref`` is the reference size actually used (for logging).
    """
    lam = float(cost_lambda or 0.0)
    # fail-fast：负值是笔误，静默按 0/自动兜底会让用户以为开关生效了。
    if lam < 0.0:
        raise ValueError(f"navit_pack_cost_lambda 必须 ≥0（0=关），收到 {lam}")
    if int(cost_ref_tokens or 0) < 0:
        raise ValueError(
            f"navit_pack_cost_ref_tokens 必须 ≥0（0=自动取数据集中位数），收到 {cost_ref_tokens}"
        )
    if lam == 0.0:
        return [float(int(c)) for c in token_counts], 1.0, 0
    pos = sorted(int(c) for c in token_counts if int(c) > 0)
    ref = int(cost_ref_tokens or 0)
    if ref <= 0:
        ref = pos[len(pos) // 2] if pos else 0
    denom = 1.0 + lam * float(ref)          # lam>0 且 ref≥0 ⇒ denom ≥ 1
    return [float(c) * (1.0 + lam * float(c)) / denom
            for c in (int(x) for x in token_counts)], 1.0, ref


def pack_indices_by_budget(token_counts, token_budget, order, max_images_per_pack=0,
                           costs=None):
    """Greedy next-fit packing of sample indices into packs whose *summed* cost stays
    within ``token_budget``.

    NaViT block-diagonal packing carries no padding, so a pack's cost is the exact sum
    of its images' token counts (unlike the padded FiT path, whose cost is
    ``max_tokens * n_images``). ``order`` is the already-shuffled index sequence; an
    image whose own token count exceeds the budget becomes a singleton pack (the caller
    warns). The result covers every index in ``order`` exactly once, order-preserving.

    ``costs`` (optional) replaces the raw token count as the packing volume — see
    :func:`navit_pack_costs`. ``None`` (default) means "cost == token count", which is
    the historical behaviour bit-for-bit.
    """
    packs = []
    cur, cur_sum = [], 0.0
    cap = int(max_images_per_pack or 0)
    budget = float(token_budget)
    vol = costs if costs is not None else token_counts
    for idx in order:
        n = float(vol[idx])
        over_budget = bool(cur) and (cur_sum + n > budget)
        over_count = cap > 0 and len(cur) >= cap
        if over_budget or over_count:
            packs.append(cur)
            cur, cur_sum = [], 0.0
        cur.append(idx)
        cur_sum += n
    if cur:
        packs.append(cur)
    return packs


def pack_indices_ffd_windowed(token_counts, token_budget, order,
                              max_images_per_pack=0, window=0, costs=None):
    """First-Fit-Decreasing packing within windows of the (already-shuffled) ``order``.

    Classic FFD (sort items by descending size, drop each into the first bin that fits)
    packs fuller than next-fit — fewer, tighter packs ⇒ fewer optimizer steps and less
    wasted token budget per step (see NeMo sequence-packing / ICLR'23 "Efficient Sequence
    Packing"). The trade-off: a *global* decreasing sort would group the same images
    together every epoch (size order is fixed), erasing the per-epoch shuffle that gives
    small-data SGD its batch-composition variety.

    The fix is ``window``: ``order`` is split into contiguous windows of ``window`` items
    and FFD runs *inside each window*. Because ``order`` is reshuffled each epoch, window
    membership (hence the grouping) changes across epochs, while the within-window
    decreasing sort still recovers most of the fill benefit. ``window<=0`` means one
    global window (max fill, but epoch-static packs — only sensible for single-pass data).

    Covers every index in ``order`` exactly once. An image larger than the budget becomes
    its own pack (the caller warns), matching :func:`pack_indices_by_budget`.

    ``costs`` (optional) replaces the raw token count as the packing volume — both the
    decreasing sort key and the bin fill test use it. See :func:`navit_pack_costs`;
    ``None`` keeps the historical token-count behaviour bit-for-bit.
    """
    budget = float(token_budget)
    cap = int(max_images_per_pack or 0)
    win = int(window or 0)
    order = list(order)
    vol = costs if costs is not None else token_counts
    if win <= 0:
        windows = [order]
    else:
        windows = [order[i:i + win] for i in range(0, len(order), win)]

    packs = []
    for w in windows:
        items = sorted(w, key=lambda i: float(vol[i]), reverse=True)
        bins = []  # each: [list_of_indices, summed_cost]
        for idx in items:
            n = float(vol[idx])
            placed = False
            for b in bins:
                over_count = cap > 0 and len(b[0]) >= cap
                if (not over_count) and (b[1] + n <= budget):
                    b[0].append(idx)
                    b[1] += n
                    placed = True
                    break
            if not placed:
                bins.append([[idx], n])
        packs.extend(b[0] for b in bins)
    return packs


def _lookup_token_count_walk(d, idx):
    """Resolve a sample's token count by walking dataset wrappers (mirror of
    :meth:`FitTokenBatchSampler._lookup_token_count`, as a free function so the NaViT
    packer can share it without disturbing the existing sampler)."""
    main = getattr(d, "main_dataset", None)
    reg = getattr(d, "reg_dataset", None)
    if main is not None and reg is not None:
        ml = getattr(d, "_main_len", len(main))
        if idx < ml:
            return _lookup_token_count_walk(main, idx)
        return _lookup_token_count_walk(reg, idx - ml)
    inner = getattr(d, "dataset", None)
    if inner is not None and inner is not d and not isinstance(inner, list):
        return _lookup_token_count_walk(inner, idx % len(inner))
    counts = getattr(d, "token_count_for_index", None)
    if counts is not None and len(counts) > 0:
        return int(counts[idx % len(counts)])
    inner = getattr(d, "base_dataset", None)
    if inner is not None and inner is not d:
        return _lookup_token_count_walk(inner, idx % len(inner))
    return 0


def _walk_attr_list(dataset, attr):
    """Find a leaf dataset's per-index list ``attr`` through single-chain wrappers
    (RepeatDataset/CachedLatentDataset), mapped to ``len(dataset)`` via ``% len`` (the
    RepeatDataset index semantics). Returns None for MergedDataset (two branches) or
    when the attribute is absent everywhere."""
    cur = dataset
    for _ in range(12):
        if getattr(cur, "main_dataset", None) is not None and getattr(cur, "reg_dataset", None) is not None:
            return None  # MergedDataset: not a single chain
        v = getattr(cur, attr, None)
        if v is not None and len(v) > 0:
            n = len(dataset)
            return [v[i % len(v)] for i in range(n)]
        nxt = getattr(cur, "dataset", None)
        if nxt is None or nxt is cur or isinstance(nxt, list):
            nxt = getattr(cur, "base_dataset", None)
        if nxt is None or nxt is cur:
            return None
        cur = nxt
    return None


def dataset_token_counts(dataset, patch_spatial=2):
    """Per-index token counts for NaViT packing.

    Prefers a populated ``token_count_for_index`` (the FiT path fills it). On the NaViT /
    non-FiT path that field is all-zero (``token_count`` is only set when an FiT plan
    exists, see ImageDataset), so fall back to deriving the count from the cached latent
    shape ``bucket_for_index = (h, w)`` (latent px) as ``(h // patch_spatial) * (w //
    patch_spatial)`` — exactly what ``patchify_latents_to_tokens`` produces. Without this
    fallback every count is 0 and the budget packer puts the *entire* dataset in one pack
    (→ a ~500k-token sequence → OOM)."""
    counts = _walk_attr_list(dataset, "token_count_for_index")
    if counts is not None and any(int(c) > 0 for c in counts):
        return [int(c) for c in counts]

    shapes = _walk_attr_list(dataset, "bucket_for_index")
    if shapes is not None:
        ps = max(1, int(patch_spatial))
        derived = []
        for s in shapes:
            if not s:
                derived.append(0)
                continue
            h, w = int(s[0]), int(s[1])
            derived.append((h // ps) * (w // ps))
        if any(c > 0 for c in derived):
            return derived

    return [int(_lookup_token_count_walk(dataset, i) or 0) for i in range(len(dataset))]


class NavitPackBatchSampler:
    """Yield packs of dataset indices for NaViT/Patch-n-Pack block-diagonal training.

    Each yielded list is one packed training sequence: the summed token count of its
    images stays within ``token_budget`` so the whole pack runs as a single
    block-diagonal forward (:meth:`MiniTrainDIT.forward_packed_navit`) with zero
    padding. This decouples "images per step" from per-image shape — unlike
    :class:`BucketBatchSampler` (one exact ``(h, w)`` per batch) or the padded FiT
    sampler, images of *different* token counts and aspect ratios share a pack, so a
    small multi-resolution dataset can still fill a large effective batch.
    """

    def __init__(self, dataset, token_budget, max_images_per_pack=0,
                 shuffle=True, seed=42, drop_last=False,
                 strategy="next_fit", ffd_window=256,
                 cost_lambda=0.0, cost_ref_tokens=0):
        self.dataset = dataset
        self.token_budget = int(token_budget)
        self.max_images_per_pack = int(max_images_per_pack or 0)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.strategy = str(strategy or "next_fit").lower()
        if self.strategy not in ("next_fit", "ffd"):
            raise ValueError(
                f"navit pack strategy 必须是 'next_fit' 或 'ffd'，收到 {strategy!r}"
            )
        self.ffd_window = int(ffd_window or 0)
        self.epoch = 0
        self.token_counts = dataset_token_counts(dataset)
        self._cached_packs = None
        # Fail-fast: all-zero token counts means the per-image size couldn't be resolved
        # (neither token_count_for_index nor bucket_for_index). Without this the budget
        # check `cur_sum + 0 > budget` never trips → the whole dataset packs into one
        # ~500k-token sequence → OOM. Far better to stop here with a clear message.
        if not self.token_counts or not any(int(c) > 0 for c in self.token_counts):
            raise RuntimeError(
                "[NavitPack] 无法解析任一样本的 token 数（token_count_for_index 与 "
                "bucket_for_index 都不可用/全 0）。NaViT 打包需要缓存数据集 "
                "（cache_latents=true）以拿到每图 latent 形状。"
            )
        # 按代价装包（opt-in）：cost_lambda=0 时 costs ≡ token_counts，逐字节等价。
        self.cost_lambda = float(cost_lambda or 0.0)
        self.costs, _scale, self.cost_ref = navit_pack_costs(
            self.token_counts, self.cost_lambda, cost_ref_tokens,
        )
        mx = max(self.token_counts) if self.token_counts else 0
        mx_cost = max(self.costs) if self.costs else 0.0
        if self.token_counts and self.token_budget < mx_cost:
            logger.warning(
                "[NavitPack] token_budget=%d < 最大单图代价=%.0f（token=%d）：该图将单独成包，"
                "可能超出预算并 OOM。建议 token_budget >= 最大单图代价。",
                self.token_budget, mx_cost, mx,
            )
        logger.info(
            "[NavitPack] dataset_len=%d token_budget=%d max_images_per_pack=%s "
            "strategy=%s ffd_window=%s (token 数范围 %d..%d)",
            len(self.token_counts), self.token_budget,
            self.max_images_per_pack or "∞", self.strategy,
            (self.ffd_window or "全局") if self.strategy == "ffd" else "-",
            min(self.token_counts) if self.token_counts else 0, mx,
        )
        if self.cost_lambda > 0.0:
            logger.info(
                "[NavitPack] 按代价装包已启用：λ=%.4g，参考尺寸 n_ref=%d token"
                "（该尺寸的图代价=token 数，容量不变）；代价范围 %.0f..%.0f。"
                "大图代价上浮=每包少装（消步时/显存尖峰），小图下浮=每包多装（提吞吐）。",
                self.cost_lambda, self.cost_ref,
                min(self.costs) if self.costs else 0.0, mx_cost,
            )
        if self.strategy == "ffd" and self.ffd_window <= 0:
            logger.warning(
                "[NavitPack] strategy=ffd 且 ffd_window<=0（全局 FFD）：每 epoch 的包将完全相同"
                "（按尺寸排序固定），削弱小数据 SGD 的 batch 多样性。多 epoch 训练建议设正窗口。"
            )

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        self._cached_packs = None

    def _build_packs(self):
        order = list(range(len(self.token_counts)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(order)
        # cost_lambda=0 时 self.costs 就是 token_counts 的浮点副本 → 装包结果逐包等价。
        _costs = self.costs if self.cost_lambda > 0.0 else None
        if self.strategy == "ffd":
            packs = pack_indices_ffd_windowed(
                self.token_counts, self.token_budget, order,
                self.max_images_per_pack, self.ffd_window, costs=_costs,
            )
        else:
            packs = pack_indices_by_budget(
                self.token_counts, self.token_budget, order,
                self.max_images_per_pack, costs=_costs,
            )
        if self.drop_last and len(packs) > 1:
            # 「未满」按与装包同一口径判断（开了 cost_lambda 就用代价），否则末包会被
            # 用另一套尺度误判。
            _vol = self.costs if self.cost_lambda > 0.0 else self.token_counts
            last_sum = sum(_vol[i] for i in packs[-1])
            if last_sum < self.token_budget:
                packs = packs[:-1]
        return packs

    def __iter__(self):
        packs = self._build_packs()
        self._cached_packs = packs
        for pack in packs:
            yield pack

    def __len__(self):
        if self._cached_packs is None:
            self._cached_packs = self._build_packs()
        return len(self._cached_packs)


def collate_fn_navit_pack(batch):
    """Collate one NaViT pack.

    Cached latents in a pack have *different* spatial shapes, so they cannot be stacked;
    they are kept as a list. The training loop patchifies each to tokens, concatenates
    the tokens and per-image RoPE grids, encodes the captions and concatenates them with
    matching ``text_seqlens``, then calls :meth:`MiniTrainDIT.forward_packed_navit`.
    """
    latents = [b["latent"] for b in batch]        # each [C, T, h_i, w_i]
    captions = [b["caption"] for b in batch]
    images = [b.get("image", "") for b in batch]
    # navit_multiscale：逐图"是否为缩放副本"标志（无副本时全 False，行为中立——
    # 训练循环仅在 navit_multiscale_loss_weight != 1.0 且存在副本时才用它构建权重）。
    ms_flags = [bool(int(b.get("ms_tokens_target", 0) or 0)) for b in batch]
    return {
        "navit_latents": latents,
        "captions": captions,
        "images": images,
        "navit_ms_flags": ms_flags,
    }


# 单次送入 VAE encode 的「总像素」软上限（含翻转份）默认值。VAE 3D encoder 中间激活很占
# 显存，用像素预算让大图自动减小每批张数，避免缓存阶段 OOM；真遇到 OOM 还有逐张兜底。
# 偏保守：1024² 原图在 flip 下每批 2 张（进网络 4 张），512² 每批可达 cache_encode_batch_size。
# 大显存卡可用 YAML `cache_encode_max_pixels` 覆盖（如 80GB 卡 16M）；它同时也是
# cache_encode_tiled 的分块触发阈值（单图像素 > 预算才分块）。
_CACHE_ENCODE_MAX_PIXELS = 4 * 1024 * 1024


def _tile_starts(total: int, tile: int, stride: int):
    """1D 分块起点：步长 stride 覆盖 [0, total)，末块贴齐右边界（保持满块尺寸，
    避免残块）。total <= tile 时单块。"""
    if total <= tile:
        return [0]
    starts = list(range(0, total - tile + 1, stride))
    if starts[-1] + tile < total:
        starts.append(total - tile)
    return starts


def _blend_ramp(n: int, ov: int, device, dtype):
    """块内 1D 混合权重：两端 ov 个位置线性升/降（严格 >0），中间恒 1。
    与邻块的对称权重在重叠区归一化后线性交叉渐变（羽化接缝）。"""
    w = torch.ones(n, device=device, dtype=dtype)
    if ov > 0:
        r = torch.linspace(1.0 / (ov + 1), ov / (ov + 1), ov, device=device, dtype=dtype)
        w[:ov] = torch.minimum(w[:ov], r)
        w[-ov:] = torch.minimum(w[-ov:], r.flip(0))
    return w


def tiled_vae_encode(encode_fn, pixels_5d, tile_px, overlap_px, down=8):
    """分块 VAE encode + latent 域羽化拼接（cache_encode_tiled）。

    把 ``pixels_5d [B,C,T,H,W]`` 按 ``tile_px``（重叠 ``overlap_px``）切成像素块，
    逐块 ``encode_fn`` 后在 latent 网格上按线性羽化权重累加并归一化——峰值显存从
    ∝ 整图像素降到 ∝ 单块像素。VAE conv 感受野越过块边界的部分是近似（非逐 bit
    等价），overlap 越大误差越小；对窗口对齐的局部算子（如 8×8 均值池化）拼接结果
    与整图 encode 精确一致（单测固化）。

    要求 H/W/tile/overlap 均为 ``down`` 的整倍数（latent 边界落整格）；不满足时
    fail-fast——调用方（缓存路径）里的图已按 16px 对齐，正常流程不会触发。
    单块可覆盖整图时直接整图 encode（逐 bit 等价，零混合开销）。
    """
    tile = int(tile_px)
    ov = int(overlap_px)
    d = max(1, int(down))
    B, C, T, H, W = pixels_5d.shape
    if H <= tile and W <= tile:
        return encode_fn(pixels_5d)
    for name, v in (("H", H), ("W", W), ("tile_px", tile), ("overlap_px", ov)):
        if v % d != 0:
            raise ValueError(
                f"tiled_vae_encode 要求 {name}={v} 是 VAE 下采样 {d} 的整倍数"
                "（latent 块边界需落在整格上）。"
            )
    if not (0 <= ov < tile):
        raise ValueError(f"overlap_px={ov} 必须满足 0 <= overlap < tile_px={tile}")
    stride = tile - ov
    ys = _tile_starts(H, tile, stride)
    xs = _tile_starts(W, tile, stride)

    canvas = None
    weight = None
    out_dtype = None
    for y0 in ys:
        y1 = min(y0 + tile, H)
        for x0 in xs:
            x1 = min(x0 + tile, W)
            lat = encode_fn(pixels_5d[..., y0:y1, x0:x1])
            out_dtype = lat.dtype
            lh, lw = lat.shape[-2], lat.shape[-1]
            if (lh, lw) != ((y1 - y0) // d, (x1 - x0) // d):
                raise ValueError(
                    f"encode_fn 输出空间尺寸 {lh}x{lw} 与预期 "
                    f"{(y1 - y0) // d}x{(x1 - x0) // d} 不符（down={d} 假设不成立）。"
                )
            if canvas is None:
                canvas = torch.zeros(
                    *lat.shape[:-2], H // d, W // d,
                    device=lat.device, dtype=torch.float32,
                )
                weight = torch.zeros(
                    H // d, W // d, device=lat.device, dtype=torch.float32,
                )
            ov_lat = ov // d
            wy = _blend_ramp(lh, ov_lat if len(ys) > 1 else 0, lat.device, torch.float32)
            wx = _blend_ramp(lw, ov_lat if len(xs) > 1 else 0, lat.device, torch.float32)
            w2d = wy[:, None] * wx[None, :]
            canvas[..., y0 // d:y1 // d, x0 // d:x1 // d] += lat.float() * w2d
            weight[y0 // d:y1 // d, x0 // d:x1 // d] += w2d
    return (canvas / weight).to(out_dtype)


def _plan_encode_batches(indices, bucket_of, max_batch, max_encode_pixels, flip):
    """把待编码样本索引按 bucket（像素尺寸）分组，再按张数 / 显存像素预算切成 micro-batch。

    同一 micro-batch 内尺寸一致 → 可 stack 成一个张量一次送进 VAE（替代逐张 batch=1）。
    flip=True 时每张图在 encode 时会额外拼一份水平翻转，一次进网络的张数是 2×，像素预算据此折半。

    参数：
        indices:           待编码样本索引（指向 self.samples）。
        bucket_of:         callable(idx) -> (h, w) 或 None；返回该样本的像素分桶尺寸。
        max_batch:         单个 micro-batch 的原图张数上限（>=1）。
        max_encode_pixels: 单次 encode 进网络的总像素上限（含翻转份）；<=0 表示不按像素限。
        flip:              是否会拼翻转份（影响像素预算折算）。
    返回：list[list[idx]]，保持桶内原顺序，覆盖且仅覆盖所有输入索引；同批尺寸一致。
    """
    max_batch = max(1, int(max_batch))
    groups = {}
    order = []
    for i in indices:
        key = bucket_of(i)
        key = tuple(key) if key else (0, 0)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(i)

    planned = []
    for key in order:
        idxs = groups[key]
        h, w = int(key[0] or 0), int(key[1] or 0)
        per = max_batch
        if max_encode_pixels and max_encode_pixels > 0 and h > 0 and w > 0:
            mult = 2 if flip else 1
            per = min(per, max(1, int(max_encode_pixels) // (mult * h * w)))
        for j in range(0, len(idxs), per):
            planned.append(idxs[j:j + per])
    return planned


class CachedLatentDataset(Dataset):
    """Kohya 风格 npz 文件缓存的数据集。

    save_dtype: torch.float32 / torch.bfloat16 / torch.float16
        - 默认 bf16：与训练 dtype 对齐，disk 占用 ≈ fp32 的 50%，读取时无精度转换开销。
        - numpy 1.x 没有原生 bf16，bf16 在磁盘上以 uint16 view 保存，并附带 sentinel
          key (`dtype_kind = "bf16"`) 让读端正确还原。
        - 旧 cache（无 dtype_kind 键）默认按 fp32 读，向后兼容。

    flip + cache 兼容（kohya 风格）：若底层 ImageDataset 启用了 flip_augment，缓存阶段会为
    每张图额外编码一份"像素域水平翻转后再 encode"的 latent，存入 npz 的 `latent_flipped`。
    __getitem__ 每次随机取原图 / 翻转其一，逐 epoch 随机翻转得以保留。绝不在 latent 空间翻转：
    VAE conv encoder 非 flip-equivariant（flip(encode(x)) ≠ encode(flip(x))），故翻转只能发生
    在像素域、encode 之前。代价：npz 体积与首次编码量约 ×2（一次性）。
    """
    def __init__(self, base_dataset, vae, device, dtype, cache_dir=None,
                 save_dtype: torch.dtype = torch.bfloat16, encode_batch_size=8,
                 encode_tiled=False, encode_tile_px=1024, encode_tile_overlap=128,
                 encode_max_pixels=0):
        import numpy as np
        self.base_dataset = base_dataset
        self.np = np
        self.samples = self._get_base_samples(base_dataset)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.bucket_for_index = []
        self.save_dtype = save_dtype
        # 缓存编码时单个 micro-batch 的原图张数上限（仍受 _CACHE_ENCODE_MAX_PIXELS 像素预算约束）。
        self.encode_batch_size = max(1, int(encode_batch_size or 1))
        # cache_encode_tiled（opt-in）：像素数超 _CACHE_ENCODE_MAX_PIXELS 的单张超大图
        # 改走分块 encode + latent 羽化拼接（tiled_vae_encode），峰值显存 ∝ 单块像素。
        # 阈值内的图路径不变（逐字节等价）。
        self.encode_tiled = bool(encode_tiled)
        self.encode_tile_px = int(encode_tile_px or 1024)
        self.encode_tile_overlap = int(encode_tile_overlap or 128)
        # 编码像素预算（cache_encode_max_pixels）：<=0 用内置保守默认 4M。
        # 同时决定每批张数（_plan_encode_batches）与 tiled 分块触发阈值。
        self.encode_max_pixels = (
            int(encode_max_pixels) if int(encode_max_pixels or 0) > 0
            else _CACHE_ENCODE_MAX_PIXELS
        )
        # 捕获用户的 flip 意图（最底层 ImageDataset.flip_augment）。必须在 _build_cache 之前设好：
        # _is_cache_valid 依赖它判断旧的"仅单份 latent"缓存是否需要失效重编码。
        leaf = base_dataset
        while hasattr(leaf, "dataset"):
            leaf = leaf.dataset
        self.flip_enabled = bool(getattr(leaf, "flip_augment", False))
        self._build_cache(vae, device, dtype)

    def _get_base_samples(self, dataset):
        """获取原始 ImageDataset 的 samples"""
        if hasattr(dataset, "samples"):
            return dataset.samples
        elif hasattr(dataset, "dataset"):
            return self._get_base_samples(dataset.dataset)
        return []

    def _get_npz_path(self, sample_or_img_path):
        """npz 缓存路径。接受 sample dict 或裸图片路径（向后兼容）。

        navit 多尺度副本（sample 带 ``ms_tokens_target``）与原生份共享同一张源图，
        故用独立的 sidecar 文件名 ``<stem>.ms<target>.npz``，互不覆盖；开关 multiscale
        不会使原生 npz 失效（无需重编码原生份）。
        """
        if isinstance(sample_or_img_path, dict):
            img_path = Path(sample_or_img_path["image"])
            ms = int(sample_or_img_path.get("ms_tokens_target", 0) or 0)
            if ms > 0:
                return img_path.with_name(f"{img_path.stem}.ms{ms}.npz")
        else:
            img_path = Path(sample_or_img_path)
        return img_path.with_suffix(".npz")

    def _is_cache_valid(self, sample_or_img_path, npz_path):
        """检查缓存是否有效（图像未修改，且格式含 latent 键）。
        若为其他模型的不兼容缓存，则删除并返回 False。"""
        if isinstance(sample_or_img_path, dict):
            img_path = Path(sample_or_img_path["image"])
            expected_bucket = sample_or_img_path.get("bucket_key")
        else:
            img_path = Path(sample_or_img_path)
            expected_bucket = None
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
            # flip + cache：启用 flip 时缓存必须含 latent_flipped；旧的"仅单份 latent"缓存失效重编码，
            # 否则会静默退化为"flip 关闭"（拿不到翻转份）。flip 关闭时本检查跳过，多余的 flipped 份无害。
            if getattr(self, "flip_enabled", False) and "latent_flipped" not in data.files:
                logger.info(
                    "删除缺少翻转 latent 的缓存（flip_augment 已启用，需原图+翻转两份）: %s", npz_path
                )
                npz_path.unlink()
                return False
            if expected_bucket is not None and "bucket_h" in data.files and "bucket_w" in data.files:
                expected_h, expected_w = int(expected_bucket[0]), int(expected_bucket[1])
                cached_h, cached_w = int(data["bucket_h"]), int(data["bucket_w"])
                if (cached_h, cached_w) != (expected_h, expected_w):
                    logger.info(
                        "删除 bucket 策略已变化的 latent 缓存: %s cached=%dx%d expected=%dx%d",
                        npz_path, cached_w, cached_h, expected_w, expected_h,
                    )
                    npz_path.unlink()
                    return False
            elif expected_bucket is not None:
                expected_h, expected_w = int(expected_bucket[0]), int(expected_bucket[1])
                cached_h, cached_w = int(latent.shape[-2]) * 8, int(latent.shape[-1]) * 8
                if (cached_h, cached_w) != (expected_h, expected_w):
                    logger.info(
                        "删除缺少 bucket 元数据且尺寸不匹配的 latent 缓存: %s cached≈%dx%d expected=%dx%d",
                        npz_path, cached_w, cached_h, expected_w, expected_h,
                    )
                    npz_path.unlink()
                    return False
            # bf16 cache 用 uint16 view 保存，跳过 isfinite 检查（uint16 永远 finite）。
            # 旧 fp32 / 新 fp16 cache 仍做 NaN/Inf 校验。
            dtype_kind = str(data["dtype_kind"]) if "dtype_kind" in data.files else "fp32"
            if dtype_kind != "bf16":
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
        logger.info("检查 VAE latent 缓存...")
        to_encode = []
        for i, sample in enumerate(self.samples):
            npz_path = self._get_npz_path(sample)
            if not self._is_cache_valid(sample, npz_path):
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
            npz_path = self._get_npz_path(self.samples[i])
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
        # flip + cache 兼容：编码期间临时关闭 base 的随机 flip，确保 self.base_dataset[i] 返回
        # 规范（未翻转）朝向。若用户启用了 flip，则下面对每张图再额外编码一份像素域水平翻转的 latent，
        # 两份一起存盘；__getitem__ 每次随机二选一，逐 epoch 随机翻转得以保留。
        leaf = self.base_dataset
        while hasattr(leaf, "dataset"):
            leaf = leaf.dataset
        _orig_flip = bool(getattr(leaf, "flip_augment", False))
        if _orig_flip:
            logger.info(
                "[cache] flip_augment + cache_latents：为每张图编码原图与水平翻转两份 latent，"
                "训练时每次随机取其一（逐 epoch 随机翻转）。编码期间临时关闭 base 随机 flip 以取规范朝向。"
            )
            leaf.flip_augment = False

        # latent → npz 落盘数组的 dtype 转换收敛到一处，原图与翻转份共用，避免分支重复。
        save_dtype = getattr(self, "save_dtype", torch.bfloat16)
        if save_dtype == torch.bfloat16:
            dtype_kind = "bf16"
            # bf16 → uint16 view（同等 bit pattern；numpy 1.x 没有原生 bf16）
            def _to_npz_array(lat):
                return lat.to(dtype=torch.bfloat16).cpu().contiguous().view(torch.uint16).numpy()
        elif save_dtype == torch.float16:
            dtype_kind = "fp16"
            def _to_npz_array(lat):
                return lat.to(dtype=torch.float16).cpu().numpy()
        else:
            dtype_kind = "fp32"
            def _to_npz_array(lat):
                return lat.cpu().float().numpy()

        # #1 同桶分组批量编码：把同尺寸的图 stack 成一个张量送进 VAE（底层 WanVAE_.encode 对 batch
        #    维通用），flip 份拼进同一 batch（[2N]）一次编码再切回，几乎零额外成本吃掉翻转开销。
        flip = bool(self.flip_enabled)
        planned = _plan_encode_batches(
            indices,
            lambda i: self.samples[i].get("bucket_key"),
            self.encode_batch_size,
            self.encode_max_pixels,
            flip,
        )
        use_cuda = str(getattr(device, "type", device)).startswith("cuda")
        oom_error = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)

        def _save_one(npz_path, lat, lat_flip, ph, pw):
            save_kwargs = {
                "latent": _to_npz_array(lat),
                "bucket_w": pw,
                "bucket_h": ph,
                "dtype_kind": dtype_kind,
            }
            if lat_flip is not None:
                save_kwargs["latent_flipped"] = _to_npz_array(lat_flip)
            self.np.savez(npz_path, **save_kwargs)

        def _save_batch(latent_cpu, batch_i, n, ph, pw):
            # #2 后台线程：dtype 转换 + 落盘，与下一批 GPU 编码重叠。latent_cpu 已在 CPU（只读，线程安全）。
            for k in range(n):
                lat_flip = latent_cpu[n + k] if flip else None
                _save_one(self._get_npz_path(self.samples[batch_i[k]]),
                          latent_cpu[k], lat_flip, ph, pw)

        def _load_batch(batch_idxs):
            # #3 预处理预取（线程）：PIL 解码 / resize / crop 与 GPU 编码重叠。
            return [(i, self.base_dataset[i]["pixel_values"]) for i in batch_idxs]

        from concurrent.futures import ThreadPoolExecutor
        save_pool = ThreadPoolExecutor(max_workers=2)
        prefetch_pool = ThreadPoolExecutor(max_workers=1)
        save_futures = []
        total = len(indices)
        done = 0
        try:
            next_future = prefetch_pool.submit(_load_batch, planned[0]) if planned else None
            for bi in range(len(planned)):
                loaded = next_future.result()
                if bi + 1 < len(planned):
                    next_future = prefetch_pool.submit(_load_batch, planned[bi + 1])

                batch_i = [i for i, _ in loaded]
                n = len(batch_i)
                if n == 0:
                    continue
                pixels = torch.stack([px for _, px in loaded]).to(device, dtype=dtype)  # [N,C,H,W]
                _, _, ph, pw = pixels.shape
                enc_in = pixels.unsqueeze(2)  # [N,C,1,H,W]
                if flip:
                    # 像素域水平翻转（dims=[-1]=宽）后拼进同一 batch；绝不在 latent 上翻转。
                    enc_in = torch.cat([enc_in, torch.flip(enc_in, dims=[-1])], dim=0)  # [2N,...]

                # cache_encode_tiled：仅超像素预算的图走分块（此时 _plan_encode_batches
                # 已保证该批只有 1 张原图，enc_in 为 [1 或 2(flip),C,1,H,W]）。
                if self.encode_tiled and ph * pw > self.encode_max_pixels:
                    logger.info(
                        "[cache-tiled] %dx%d 超像素预算，分块 encode（tile=%d overlap=%d）：%s",
                        pw, ph, self.encode_tile_px, self.encode_tile_overlap,
                        self.samples[batch_i[0]]["image"],
                    )
                    with torch.no_grad():
                        latent_all = tiled_vae_encode(
                            lambda x: vae.model.encode(x, vae.scale),
                            enc_in, self.encode_tile_px, self.encode_tile_overlap,
                        )
                    latent_cpu = latent_all.detach().to("cpu")
                    if torch.isfinite(latent_all).all().item():
                        save_futures.append(
                            save_pool.submit(_save_batch, latent_cpu, batch_i, n, ph, pw))
                    else:
                        logger.warning("VAE 分块编码产生非有限 latent，跳过缓存: %s",
                                       self.samples[batch_i[0]]["image"])
                    done += n
                    logger.info("  编码进度: %d/%d", min(done, total), total)
                    continue

                try:
                    with torch.no_grad():
                        latent_all = vae.model.encode(enc_in, vae.scale)
                except oom_error:
                    # 显存不够：清缓存并降级逐张编码（慢但不中断整个缓存任务）。
                    if use_cuda:
                        torch.cuda.empty_cache()
                    logger.warning(
                        "[cache] encode OOM at batch=%d (%dx%d)，降级逐张。可调小 cache_encode_batch_size。",
                        enc_in.shape[0], pw, ph,
                    )
                    with torch.no_grad():
                        latent_all = torch.cat(
                            [vae.model.encode(enc_in[s:s + 1], vae.scale) for s in range(enc_in.shape[0])],
                            dim=0,
                        )

                # #4 批级 isfinite：整批一次同步而非每张。全有限走后台落盘；否则逐张定位、跳过坏图。
                latent_cpu = latent_all.detach().to("cpu")
                if torch.isfinite(latent_all).all().item():
                    save_futures.append(save_pool.submit(_save_batch, latent_cpu, batch_i, n, ph, pw))
                else:
                    for k in range(n):
                        lat = latent_cpu[k]
                        lat_flip = latent_cpu[n + k] if flip else None
                        ok = bool(torch.isfinite(lat).all().item()) and (
                            lat_flip is None or bool(torch.isfinite(lat_flip).all().item())
                        )
                        if not ok:
                            logger.warning("VAE 编码产生非有限 latent，跳过缓存: %s",
                                           self.samples[batch_i[k]]["image"])
                            continue
                        _save_one(self._get_npz_path(self.samples[batch_i[k]]),
                                  lat, lat_flip, ph, pw)

                done += n
                logger.info("  编码进度: %d/%d", min(done, total), total)

            for f in save_futures:
                f.result()  # 传播后台落盘异常
        finally:
            save_pool.shutdown(wait=True)
            prefetch_pool.shutdown(wait=True)
            if _orig_flip:
                leaf.flip_augment = True

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        npz_path = self._get_npz_path(sample)
        data = self.np.load(npz_path)
        # dtype_kind sentinel：bf16 cache 在磁盘上是 uint16 view，要 view 回 bf16；
        # fp16 / fp32 直接 from_numpy。旧 cache 无 dtype_kind，按 fp32 兼容。
        dtype_kind = str(data["dtype_kind"]) if "dtype_kind" in data.files else "fp32"
        # flip + cache：缓存阶段已在像素域翻转后编码了 latent_flipped；这里每次随机二选一，
        # 恢复逐 epoch 随机水平翻转。绝不在 latent 空间翻转（VAE conv encoder 非 flip-equivariant，
        # flip(encode(x)) ≠ encode(flip(x))，在 latent 上翻会喂给训练偏离真实分布的潜变量，
        # 推理时表现为马赛克 / 边缘溶解）。
        latent_key = "latent"
        if (getattr(self, "flip_enabled", False)
                and "latent_flipped" in data.files
                and random.random() < 0.5):
            latent_key = "latent_flipped"
        latent_np = data[latent_key]
        if dtype_kind == "bf16":
            # uint16 → torch.uint16 → view as bf16（bit pattern 相同）
            latent = torch.from_numpy(latent_np).view(torch.bfloat16)
        else:
            latent = torch.from_numpy(latent_np)
        # bf16 不能 isfinite 直接判（pytorch 旧版本可能行为不一致），但我们在保存时已验证过；
        # 这里只对 fp32/fp16 cache 做防御性 isfinite 校验。
        if dtype_kind != "bf16" and not torch.isfinite(latent).all():
            raise RuntimeError(f"读取到非有限 latent 缓存: {npz_path}")

        # 获取 base_dataset 的引用（处理可能的嵌套）
        base = self.base_dataset
        while hasattr(base, "dataset"):
            base = base.dataset

        caption = None
        if getattr(base, "caption_override", None) is not None:
            caption = base.caption_override
        elif sample.get("json_path") and hasattr(base, "_process_caption_json"):
            # 使用预 normalized 缓存（避免每次 __getitem__ load+normalize JSON）
            caption = base._process_caption_json(
                sample["json_path"], normalized=sample.get("normalized_json"),
            )

        if caption is None and sample.get("txt_path"):
            caption = sample["txt_path"].read_text(encoding="utf-8").strip()
            if hasattr(base, "_process_caption_txt"):
                caption = base._process_caption_txt(caption)

        if caption is None:
            caption = ""

        return {
            "latent": latent,
            "caption": caption,
            "image": str(sample["image"]),
            # navit 多尺度副本标记（>0 = 该条目是 ≤N token 档的缩放副本）；
            # collate_fn_navit_pack 汇集成 per-image 标志，供逐图 loss 权重用。
            "ms_tokens_target": int(sample.get("ms_tokens_target", 0) or 0),
        }


def _require_uniform_batch_shape(batch, key, message, detail_key="shape"):
    """诊断守卫：同一 batch 内 ``b[key]`` 形状必须一致，否则说明分桶失效并抛出。

    BucketBatchSampler 应已按 bucket 分组；出现混合尺寸时把每个样本的形状打进
    错误信息便于定位。形状一致时直接返回，调用方继续正常 collate（行为不变）。
    """
    shapes = [tuple(b[key].shape) for b in batch]
    if len(set(shapes)) <= 1:
        return
    details = [
        f"  - {b.get('image', '?')}: {detail_key}={tuple(b[key].shape)}"
        for b in batch
    ]
    raise RuntimeError(message + "\nBatch 内容:\n" + "\n".join(details))


def collate_fn(batch):
    """DataLoader collate"""
    _require_uniform_batch_shape(
        batch, "pixel_values",
        "[collate_fn] 同一 batch 出现不同尺寸张量，BucketBatchSampler 分桶失效。",
    )
    pixels = torch.stack([b["pixel_values"] for b in batch])
    captions = [b["caption"] for b in batch]
    images = [b.get("image", "") for b in batch]
    return {"pixel_values": pixels, "captions": captions, "images": images}


def collate_fn_cached_fit(batch):
    """Collate cached latents for the token-bucket FiT path.

    Used only with ``token_bucket`` (full coverage): every image fills its bucket
    exactly, so batches are routed through :class:`BucketBatchSampler` (one exact
    grid per batch) and the latent mask is all-ones — rebuilt here from the latent
    spatial shape rather than stored in the ``.npz``. Alpha masks are therefore NOT
    preserved through the latent cache in this mode (warned at setup); for
    alpha-masked datasets use the non-cached FiT path.
    """
    _require_uniform_batch_shape(
        batch, "latent",
        "[collate_fn_cached_fit] 同一 batch 出现不同 latent 尺寸，token_bucket 的"
        "单一网格分批失效（应由 BucketBatchSampler 按精确桶尺寸分组）。",
        detail_key="latent_shape",
    )
    latents = torch.stack([b["latent"] for b in batch])  # [B, C, T, h, w]
    h, w = int(latents.shape[-2]), int(latents.shape[-1])
    latent_mask = torch.ones(len(batch), 1, h, w, dtype=torch.float32)
    captions = [b["caption"] for b in batch]
    images = [b.get("image", "") for b in batch]
    return {
        "latents": latents,
        "latent_mask": latent_mask,
        "captions": captions,
        "images": images,
    }


def collate_fn_fit_packed(batch):
    """Collate native FiT image batches by padding pixels and masks.

    This keeps source pixels intact. Any padding is explicit and carried in
    ``pixel_mask`` so later patchification can exclude padded regions.
    """
    import torch.nn.functional as F

    max_h = max(int(b["pixel_values"].shape[-2]) for b in batch)
    max_w = max(int(b["pixel_values"].shape[-1]) for b in batch)
    pixels = []
    masks = []
    token_counts = []
    captions = []
    images = []
    for b in batch:
        pixel = b["pixel_values"]
        mask = b.get("pixel_mask")
        if mask is None:
            mask = torch.ones(1, pixel.shape[-2], pixel.shape[-1], dtype=pixel.dtype)
        pad_h = max_h - int(pixel.shape[-2])
        pad_w = max_w - int(pixel.shape[-1])
        pixels.append(F.pad(pixel, (0, pad_w, 0, pad_h)))
        masks.append(F.pad(mask, (0, pad_w, 0, pad_h)))
        token_counts.append(int(b.get("token_count", 0) or 0))
        captions.append(b["caption"])
        images.append(b.get("image", ""))
    return {
        "pixel_values": torch.stack(pixels),
        "pixel_mask": torch.stack(masks),
        "fit_token_counts": torch.tensor(token_counts, dtype=torch.long),
        "captions": captions,
        "images": images,
    }


def collate_fn_cached(batch):
    """DataLoader collate for cached latents"""
    _require_uniform_batch_shape(
        batch, "latent",
        "[collate_fn_cached] 同一 batch 出现不同 latent 尺寸，BucketBatchSampler 分桶失效。",
        detail_key="latent_shape",
    )
    latents = torch.stack([b["latent"] for b in batch])
    captions = [b["caption"] for b in batch]
    images = [b.get("image", "") for b in batch]
    return {"latents": latents, "captions": captions, "images": images}
