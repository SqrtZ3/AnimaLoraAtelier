"""离线计算数据集的径向平均功率谱（RAPSD），供 CSFlow（trainer/csflow.py）构造 t 采样权重。

像素域（立项决策 2026-06-25）：直接在**源训练图**上算——它们本就是像素，CSF 的 cycles/degree
轴能对上；不需要 VAE 解码（解码只会注入 VAE 的频响伪影，源图谱才是论文要的"自然图像功率谱 S_f"）。
若确想要 VAE 往返版（折进 VAE 频响），另开 --vae-roundtrip 分支即可，但默认源图谱更干净。

用法：
  python tools/compute_rapsd.py --data-dir ./Dataset/your-dataset \
      --out ./output/fdy12/rapsd.json --res 1024 --max-images 300

产出 json：{"rapsd": [...f=0..0.5 cycles/px...], "resolution": R, "n_images": N,
           "channels": "luma", "source": "<data_dir>"}。零额外训练期开销，一次性。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

try:
    from PIL import Image
except ImportError:
    print("需要 Pillow：pip install pillow", file=sys.stderr)
    raise

_IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
# BT.601 luma 权重（人眼亮度，CSF 本就定义在亮度通道上）
_LUMA = torch.tensor([0.299, 0.587, 0.114]).view(3, 1, 1)


def _list_images(data_dir: Path) -> list[Path]:
    return sorted(p for p in data_dir.rglob("*") if p.suffix.lower() in _IMG_EXT)


def _radial_average(power: torch.Tensor) -> torch.Tensor:
    """power: (R, R) fftshift 后的功率谱 → 径向平均 (R//2+1,)，索引 k 对应 f=k/R cycles/px ∈[0,0.5]。"""
    R = power.shape[-1]
    cy = cx = R // 2
    ys = torch.arange(R).view(R, 1).float() - cy
    xs = torch.arange(R).view(1, R).float() - cx
    r = torch.sqrt(ys * ys + xs * xs).round().long().clamp(max=R // 2)
    n_bins = R // 2 + 1
    sums = torch.bincount(r.flatten(), weights=power.flatten(), minlength=n_bins)[:n_bins]
    cnts = torch.bincount(r.flatten(), minlength=n_bins)[:n_bins].clamp(min=1)
    return sums / cnts


def compute_rapsd(data_dir: Path, res: int, max_images: int) -> dict:
    res = (res // 2) * 2  # 偶数边长，径向 bin 干净
    imgs = _list_images(data_dir)
    if not imgs:
        raise FileNotFoundError(f"{data_dir} 下没有图片")
    if max_images > 0 and len(imgs) > max_images:
        # 均匀抽样，避免只取目录前缀
        step = len(imgs) / max_images
        imgs = [imgs[int(i * step)] for i in range(max_images)]

    acc = torch.zeros(res // 2 + 1, dtype=torch.float64)
    used = 0
    for p in imgs:
        try:
            im = Image.open(p).convert("RGB").resize((res, res), Image.LANCZOS)
        except Exception as e:
            print(f"  跳过 {p.name}: {e}", file=sys.stderr)
            continue
        x = torch.from_numpy(_to_array(im)).float() / 255.0   # (3,R,R)
        luma = (x * _LUMA).sum(dim=0)                          # (R,R) 亮度
        luma = luma - luma.mean()                             # 去 DC（避免 0 频主导）
        fft = torch.fft.fftshift(torch.fft.fft2(luma))
        power = (fft.real.square() + fft.imag.square())        # |F|²
        acc += _radial_average(power).double()
        used += 1
    if used == 0:
        raise RuntimeError("没有可用图片")
    rapsd = (acc / used)
    # 归一到单位均值（绝对量纲无关，build 表里只用相对形状）
    rapsd = rapsd / rapsd.mean().clamp(min=1e-12)
    return {
        "rapsd": rapsd.float().tolist(),
        "resolution": res,
        "n_images": used,
        "channels": "luma",
        "source": str(data_dir),
    }


def _to_array(im):
    import numpy as np
    return np.asarray(im).transpose(2, 0, 1)  # HWC → CHW


def main():
    ap = argparse.ArgumentParser(description="数据集径向功率谱 → CSFlow RAPSD 档")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=1024, help="计算谱用的方形边长（默认 1024）")
    ap.add_argument("--max-images", type=int, default=300, help="抽样图数上限（0=全部）")
    args = ap.parse_args()

    prof = compute_rapsd(Path(args.data_dir), args.res, args.max_images)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(prof, f, ensure_ascii=False, indent=1)
    print(f"RAPSD 写入 {out}：n_images={prof['n_images']} res={prof['resolution']} "
          f"bins={len(prof['rapsd'])}")


if __name__ == "__main__":
    main()
