"""从 upstream `trainer/data.py` 摘录的**图像 planning + 分块 VAE encode**。

逐字摘录（按 AST 行号切，函数体一个字符都没改）。来源行号与 sha256 见
`UPSTREAM.md`；漂移由 `tools/check_sync.py` 守。

这四个公开符号决定缓存 npz 的 **token 数与网格**，而 token 数直接决定 TPU 侧的
打包布局。所以必须与 upstream 逐字一致 —— 口径一偏，两个后端的 loss 曲线就悄悄
对不上，而且不报错。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

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
