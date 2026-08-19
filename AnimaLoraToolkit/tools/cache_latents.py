r"""把数据集的图像 latent 离线算好并落盘（PyTorch 侧工具，给 TPU 后端喂料）。

## 为什么需要它

TPU 后端（`jax_tpu/`）只跑 DiT，**不做任何编码**（`jax_tpu/data.py` 的 docstring 写明
两份缓存都由 PyTorch 侧离线产出）。文本侧有 `cache_text_features.py`，图像侧此前只有
`trainer/data.py:CachedLatentDataset` 这条**训练进程内**的缓存路径 —— 本地没有训练进程
可挂（训练只在云端跑），所以把缓存编码这一步拆成独立工具。

## 落盘口径（逐行复刻训练路径，抄错就是静默错）

像素管线复刻 `trainer/data.py:ImageDataset.__getitem__` 的 fit_packed 分支
（data.py:993-1040）与 `CachedLatentDataset._encode_and_save`（data.py:2333+）：

  * 原生份：`plan_native_fit_image(..., align_mode="floor")` —— navit_native_resolution
    强制 floor（anima_train.py:1749），即**左上角**裁到 16px 整倍数（patch2 × VAE8），
    零 padding、mask 恒全 1。居中对齐是 ARB 桶路径的做法，**别抄混**。
  * 多尺度 sidecar：`plan_multiscale_copy`（data.py:136）出规划，然后
    "等比缩小到覆盖规划尺寸再**中心裁剪**"（data.py:1005-1011，LANCZOS），
    存 `<stem>.ms<档>.npz`。只降不升采样；源图 token ≤ 档位的跳过。
  * 归一化：`arr / 127.5 - 1.0`，RGB。alpha 处理与 `_load_rgb_and_alpha_mask`
    的非 mask 路径一致（`convert("RGB")`）；数据集带 alpha 时会 fail-fast 提示。
  * 编码：`vae.model.encode(x[.,1,H,W], vae.scale)`，scale = [mean, 1/std]
    （trainer/models.py:313）。原图与像素域水平翻转各编一份（VAE conv encoder 非
    flip-equivariant，latent 空间翻转 ≠ 像素空间翻转，data.py:2167）。
  * npz 键：`latent` [C,T=1,H/8,W/8]、`bucket_w`/`bucket_h`（像素尺寸）、
    `dtype_kind="bf16"`（磁盘上 uint16 位模式，numpy 无原生 bf16）、`latent_flipped`。
    `np.savez`（非压缩），与 trainer 一致。

规划数学（floor 对齐 / 多尺度缩放系数）**直接 import trainer.data 的函数**，
不在本文件重新实现 —— 单一事实源，避免两处公式漂移。

## 显存

整图 encode 的峰值 ∝ 像素数（conv encoder 的全分辨率特征图）。超 `--tiled-threshold`
的图走 `tiled_vae_encode` 分块 + 羽化拼接（data.py:2055，与 cache_encode_tiled 同一条
路径，**分块缝处是近似**）。默认 1.5M 像素是 8GB 笔记本卡的稳妥值；大显存机器可抬到
训练侧内置默认 4M（能整图就整图，分块只是显存兜底）。

用法：
    <torch-python> cache_latents.py --data-dir <图片目录> \
        --vae <qwen_image_vae.safetensors> --ms-ladder 4096 [--overwrite]
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _prep_pixels(img, plan, was_resized: bool):
    """按训练路径把 PIL 图变成 [C,H,W] fp32 张量（/127.5-1）。

    was_resized=True（多尺度副本）：等比 cover 缩放 + 中心裁剪（data.py:1005-1011）。
    was_resized=False（原生份）：floor 对齐下 __getitem__ 的 crop 是 (0,0,w,h) 左上角
    —— 而 plan 尺寸本就是 floor16 后的尺寸，所以这里只需把对齐余量从右/下裁掉。
    """
    import torch

    if was_resized:
        cover = max(plan.width / img.width, plan.height / img.height)
        rw = max(plan.width, int(math.ceil(img.width * cover)))
        rh = max(plan.height, int(math.ceil(img.height * cover)))
        left = (rw - plan.width) // 2
        top = (rh - plan.height) // 2
        img = img.resize((rw, rh), Image.LANCZOS).crop(
            (left, top, left + plan.width, top + plan.height))
    else:
        img = img.crop((0, 0, min(img.width, plan.width), min(img.height, plan.height)))
    arr = np.array(img).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def _to_npz_bf16(lat) -> np.ndarray:
    import torch
    return lat.to(dtype=torch.bfloat16).cpu().contiguous().view(torch.uint16).numpy()


def _cache_valid(npz_path: Path, img_path: Path, need_flip: bool) -> bool:
    """与 CachedLatentDataset._is_cache_valid 同口径（mtime / latent 键 / C=16 / 翻转份）。"""
    if not npz_path.exists():
        return False
    if npz_path.stat().st_mtime < img_path.stat().st_mtime:
        return False
    try:
        data = np.load(npz_path)
    except Exception:
        return False
    if "latent" not in data.files or data["latent"].ndim != 4 or data["latent"].shape[0] != 16:
        return False
    if need_flip and "latent_flipped" not in data.files:
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--vae", required=True, help="qwen_image_vae.safetensors")
    ap.add_argument("--ms-ladder", default="",
                    help="逗号分隔的多尺度 token 档（如 4096）。留空 = 关，"
                         "与 navit_multiscale_token_ladder 同语义")
    ap.add_argument("--max-tokens", type=int, default=204800,
                    help="fit_max_tokens，超了直接报错（与 trainer 的 fail 策略一致）")
    ap.add_argument("--no-flip", action="store_true",
                    help="不编码 latent_flipped（yaml flip_augment: false 时用）")
    ap.add_argument("--tiled-threshold", type=int, default=500_000,
                    help="像素数超阈值走分块 encode。本机 8GB 卡实测：整图 768^2 峰值 4.5G、"
                         "1024^2 超 8G OOM（WanVAE feat_cache 把每层全分辨率特征图都留着），"
                         "所以默认 0.5M 几乎全部走分块；大显存机器可抬到 4M（训练侧默认），"
                         "分块缝处是近似，能整图就整图")
    ap.add_argument("--tile-px", type=int, default=640,
                    help="分块边长。1024 在 8GB 卡上自身就 OOM，640 实测峰值 ~3G")
    ap.add_argument("--tile-overlap", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    import torch
    from trainer.data import (plan_native_fit_image, plan_multiscale_copy,
                              tiled_vae_encode)
    from trainer.models import load_vae, find_diffusion_pipe_root

    ladder = sorted({int(x) for x in a.ms_ladder.split(",") if x.strip()})
    root = Path(a.data_dir)
    imgs = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT)
    if not imgs:
        print(f"{root} 下没找到图片")
        return 1

    dev = torch.device(a.device)
    vae = load_vae(a.vae, dev, torch.bfloat16, find_diffusion_pipe_root())
    print(f"{len(imgs)} 张图，ms 档 {ladder or '关'}，flip {'关' if a.no_flip else '开'}")

    n_ok = n_skip = n_ms = 0
    t0 = time.time()
    for idx, img_path in enumerate(imgs):
        raw = Image.open(img_path)
        if "A" in raw.getbands() or raw.mode in ("LA", "PA") \
                or raw.info.get("transparency") is not None:
            raise RuntimeError(
                f"{img_path.name} 带 alpha 通道：本工具只复刻了 convert('RGB') 路径，"
                f"训练侧另有 alpha 合成逻辑（_load_rgb_and_alpha_mask），先人工处理这张图")
        img = raw.convert("RGB")

        jobs = [("native", plan_native_fit_image(
            img.width, img.height, max_tokens=a.max_tokens, align_mode="floor"))]
        for tgt in ladder:
            ms = plan_multiscale_copy(img.width, img.height, tgt)
            if ms is None or ms.token_count >= jobs[0][1].token_count:
                continue
            jobs.append((f"ms{tgt}", ms))

        for tag, plan in jobs:
            npz = (img_path.with_suffix(".npz") if tag == "native"
                   else img_path.with_name(f"{img_path.stem}.{tag}.npz"))
            if not a.overwrite and _cache_valid(npz, img_path, not a.no_flip):
                n_skip += 1
                continue
            pixels = _prep_pixels(img, plan, was_resized=(tag != "native"))
            enc_in = pixels.unsqueeze(0).unsqueeze(2)  # [1,C,1,H,W]
            parts = [enc_in]
            if not a.no_flip:
                # 像素域水平翻转（宽轴），与 _encode_and_save 的 dims=[-1] 一致
                parts.append(torch.flip(enc_in, dims=[-1]))
            lats = []
            with torch.no_grad():
                for part in parts:
                    part = part.to(dev, torch.bfloat16)
                    _, _, _, ph, pw = part.shape
                    if ph * pw > a.tiled_threshold:
                        lat = tiled_vae_encode(
                            lambda x: vae.model.encode(x, vae.scale),
                            part, a.tile_px, a.tile_overlap)
                    else:
                        lat = vae.model.encode(part, vae.scale)
                    if not torch.isfinite(lat).all().item():
                        raise RuntimeError(f"VAE 编码产生非有限 latent: {img_path.name} ({tag})")
                    lats.append(lat)
            lat, lat_flip = lats[0][0], (lats[1][0] if len(lats) > 1 else None)
            kw = {"latent": _to_npz_bf16(lat), "bucket_w": plan.width,
                  "bucket_h": plan.height, "dtype_kind": "bf16"}
            if lat_flip is not None:
                kw["latent_flipped"] = _to_npz_bf16(lat_flip)
            np.savez(npz, **kw)
            del pixels, enc_in, parts, lats, lat, lat_flip
            if dev.type == "cuda":
                torch.cuda.empty_cache()   # 74 张异尺寸 → cuDNN workspace 碎片，每张清一次
            if tag == "native":
                n_ok += 1
            else:
                n_ms += 1
        print(f"  [{idx + 1}/{len(imgs)}] {img_path.name}"
              f"  native {jobs[0][1].width}x{jobs[0][1].height}"
              f" ({jobs[0][1].token_count} tok)"
              + (f"  +{len(jobs) - 1}ms" if len(jobs) > 1 else ""), flush=True)

    dt = time.time() - t0
    print(f"完成：原生 {n_ok} 编码 / {n_skip} 跳过（缓存有效），sidecar {n_ms}，{dt:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
