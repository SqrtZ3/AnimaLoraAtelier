"""把已训好的标准 LoRA safetensors 压成小体积部署件（全局 σ²/字节最优秩分配）。

与训练期 `lora_compress_budget_mb` 同一套分配器（trainer.lora.allocate_ranks_by_budget），
区别只是本工具作用于**已存在的 checkpoint 文件**——训练早于该功能上线的产物用这个补压。

机理与依据见 allocate_ranks_by_budget 的 docstring。Krea2 c12port epoch19 实测
（ComfyUI convrot w4a4 + Bypass LoRA，8 步 turbo，3 seed × 2 prompt 与原件配对）：

    体积      保留能量   高频(Laplacian)保留   判读
    216.8MB   100%       1.000                原件
    35.5MB    97.96%     0.997                几乎无损
    20.3MB    97.38%     —                    未实拍
    8.9MB     96.51%     0.975                发丝/缝线肉眼无差

对照：逐层能量阈值策略同等能量下体积大 1.3–3.2×（energy=0.95 → 85.5MB/97.79%）。

用法：
    python tools/lora_compress.py in.safetensors --budget-mb 35 [-o out.safetensors]
    python tools/lora_compress.py "dir/*.safetensors" --budget-mb 8.9    # 批量
    python tools/lora_compress.py in.safetensors --curve                 # 只看体积/能量曲线
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainer.lora import (  # noqa: E402
    allocate_ranks_by_budget, lora_pair_spectrum, svd_truncate_lora_pair,
)


def load_pairs(path: str):
    """返回 {模块名: (down, up, scaling)} 与原始 metadata 中的 alpha 表。"""
    sd = load_file(path)
    alphas = {k[: -len(".alpha")]: float(sd[k]) for k in sd if k.endswith(".alpha")}
    pairs = {}
    for k in sd:
        if not k.endswith(".lora_down.weight"):
            continue
        base = k[: -len(".lora_down.weight")]
        up = sd.get(base + ".lora_up.weight")
        if up is None:
            continue
        down = sd[k]
        rank = down.shape[0]
        pairs[base] = (down, up, alphas.get(base, rank) / rank)
    if not pairs:
        raise SystemExit(f"{path}: 没找到 lora_down/lora_up 键对——不是标准 LoRA 文件？")
    return pairs


def compress(path: str, budget_mb: float, out_path: str, device: str):
    pairs = load_pairs(path)
    spectra, per_rank = {}, {}
    for name, (down, up, sc) in pairs.items():
        spectra[name] = lora_pair_spectrum(
            down.float().to(device), up.float().to(device), sc).cpu()
        per_rank[name] = (up.shape[0] + down.shape[1]) * 2

    keep_map = allocate_ranks_by_budget(
        spectra, per_rank, budget_mb * 2 ** 20, min_rank=1)

    tot = sum((s ** 2).sum().item() for s in spectra.values())
    kept = sum((spectra[n][: keep_map[n]] ** 2).sum().item() for n in spectra)

    out = {}
    for name, (down, up, sc) in pairs.items():
        d, u, keep, _dropped = svd_truncate_lora_pair(
            down.float().to(device), up.float().to(device), sc,
            energy=1.0, max_rank=keep_map[name])
        out[f"{name}.lora_down.weight"] = d.to(torch.bfloat16).cpu().contiguous()
        out[f"{name}.lora_up.weight"] = u.to(torch.bfloat16).cpu().contiguous()
        # scaling 已折进因子 → alpha=keep 使 scaling'=1
        out[f"{name}.alpha"] = torch.tensor(float(keep))

    ks = sorted(keep_map.values())
    save_file(out, out_path, metadata={
        "anima_lora_variant": "svd_compressed",
        "anima_compress_budget_mb": str(budget_mb),
        "anima_compress_source": os.path.basename(path),
        "ss_network_alpha": "per-layer",
        "ss_network_dim": str(ks[-1]),
    })
    print(f"  {os.path.basename(path)} "
          f"{os.path.getsize(path) / 2 ** 20:.1f}MB -> "
          f"{os.path.basename(out_path)} {os.path.getsize(out_path) / 2 ** 20:.1f}MB  "
          f"保留能量 {kept / tot * 100:.2f}%  "
          f"逐层 rank min={ks[0]} 中位={ks[len(ks) // 2]} max={ks[-1]}")


def curve(path: str, device: str):
    """只打印体积/保留能量曲线，帮你选档位，不写文件。"""
    pairs = load_pairs(path)
    spectra, per_rank = {}, {}
    for name, (down, up, sc) in pairs.items():
        spectra[name] = lora_pair_spectrum(
            down.float().to(device), up.float().to(device), sc).cpu()
        per_rank[name] = (up.shape[0] + down.shape[1]) * 2
    tot = sum((s ** 2).sum().item() for s in spectra.values())
    full = sum(per_rank[n] * spectra[n].numel() for n in spectra) / 2 ** 20
    print(f"{os.path.basename(path)}：满 rank {full:.1f}MB")
    print(f"  {'预算MB':>8s}{'实际MB':>9s}{'保留能量':>11s}{'逐层rank中位':>13s}")
    for mb in (5, 8.9, 15, 20, 35, 50, 80):
        if mb > full:
            break
        km = allocate_ranks_by_budget(spectra, per_rank, mb * 2 ** 20, min_rank=1)
        kept = sum((spectra[n][: km[n]] ** 2).sum().item() for n in spectra)
        act = sum(km[n] * per_rank[n] for n in km) / 2 ** 20
        ks = sorted(km.values())
        print(f"  {mb:8.1f}{act:9.1f}{kept / tot * 100:10.2f}%{ks[len(ks) // 2]:13d}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="输入 safetensors（支持 glob）")
    ap.add_argument("--budget-mb", type=float, default=0.0,
                    help="目标体积（MB）。为 0 且未加 --curve 时报错。")
    ap.add_argument("-o", "--out", default="",
                    help="输出路径（仅单文件时有效）；默认 {stem}.c{budget}mb.safetensors")
    ap.add_argument("--curve", action="store_true", help="只打印体积/能量曲线，不写文件")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    files = []
    for pat in args.inputs:
        files.extend(sorted(glob.glob(pat)) or ([pat] if os.path.exists(pat) else []))
    if not files:
        raise SystemExit("没有匹配到任何输入文件")

    if args.curve:
        for f in files:
            curve(f, args.device)
        return
    if args.budget_mb <= 0:
        raise SystemExit("需要 --budget-mb（或用 --curve 先看曲线选档）")
    if args.out and len(files) > 1:
        raise SystemExit("-o 只能用于单个输入文件")

    for f in files:
        stem, _ = os.path.splitext(f)
        out = args.out or f"{stem}.c{args.budget_mb:g}mb.safetensors"
        compress(f, args.budget_mb, out, args.device)


if __name__ == "__main__":
    main()
