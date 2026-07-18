"""LoRA checkpoint 逐层增量取证：能量分布 + 谱形指纹。

用途：不用烧完整个 run 就能判断一个训练配置是否健康。核心指标是**谱集中度**
—— 2026-07-18 的 muon_sf 归因证明它能一刀切开"优化器是否在抹平谱形"：

    配置              总||dW||^2  top1能量  谱条件数  有效秩   结果
    adamw  c12port      112.4      57.7%     22.1     8.3    拟合成功
    muon_sf c12port      44.5       8.9%      1.7    25.0    几乎不拟合

真实画风增量在每层上近似 rank-1（adamw 最强层 top1 占 98%）。若某优化器产出
的 dW 谱条件数接近 1、有效秩接近满 rank，说明它在把预算平摊到所有方向，
调 lr 治不了（lr 只等比缩放全部奇异值 = 尺度问题，而这是形状问题）。

用法：
    python tools/lora_delta_forensics.py A.safetensors [B.safetensors ...]

实现说明：用 QR 技巧免物化全矩阵 —— B=Qb Rb, A^T=Qa Ra 时
svdvals(B@A) == svdvals(Rb @ Ra^T)，对 6144x16384 的层也是秒级。
"""

from __future__ import annotations

import argparse
import collections
import math
import re
import sys

import torch
from safetensors.torch import load_file


def group_of(name: str) -> str:
    """按模块类别分组（键名可能用 . 或 _ 分隔，两种都认）。"""
    if "txtfusion" in name:
        return "txtfusion"
    if "tproj" in name:
        return "tproj"
    if re.search(r"mlp[._](gate|up)$", name):
        return "mlp.gate/up"
    if re.search(r"mlp[._]down$", name):
        return "mlp.down"
    if re.search(r"attn[._](wq|wk|wv)$", name):
        return "attn.qkv"
    if re.search(r"attn[._](wo|gate)$", name):
        return "attn.o/gate"
    return "other"


def analyze(path: str, device: str = "cuda"):
    sd = load_file(path)
    if not torch.cuda.is_available():
        device = "cpu"
    pairs = {}
    for k in sd:
        if k.endswith(".lora_down.weight"):
            base = k[: -len(".lora_down.weight")]
            up = base + ".lora_up.weight"
            if up in sd:
                pairs[base] = (sd[k], sd[up])
    if not pairs:
        raise SystemExit(f"{path}: 没找到 lora_down/lora_up 键对——不是标准 LoRA 文件？")
    alphas = {k[: -len(".alpha")]: float(sd[k]) for k in sd if k.endswith(".alpha")}

    tot = 0.0
    per_group = collections.Counter()
    per_group_n = collections.Counter()
    rows = []
    for name, (down, up) in pairs.items():
        d = down.float().to(device)
        u = up.float().to(device)
        rank = d.shape[0]
        scale = alphas.get(name, rank) / rank
        _, ru = torch.linalg.qr(u)
        _, rd = torch.linalg.qr(d.T)
        sv = torch.linalg.svdvals((ru @ rd.T) * scale)
        e2 = sv ** 2
        energy = e2.sum().item()
        tot += energy
        g = group_of(name)
        per_group[g] += energy
        per_group_n[g] += 1
        p = e2 / max(e2.sum(), torch.tensor(1e-30, device=e2.device))
        p = p[p > 0]
        erank = float(torch.exp(-(p * p.log()).sum()))
        rows.append({
            "name": name, "rank": rank, "energy": energy, "erank": erank,
            "top1": (e2[0] / e2.sum()).item(),
            "cond": (sv[0] / sv[-1]).item(),
        })

    n = len(rows)
    print(f"\n=== {path} ===")
    print(f"模块数={n}  总 ||dW||^2 = {tot:.4g}  零增量层={sum(1 for r in rows if r['energy'] < 1e-12)}")
    print(f"★谱形指纹：top1 能量={sum(r['top1'] for r in rows)/n*100:.2f}%   "
          f"谱条件数={sum(r['cond'] for r in rows)/n:.1f}   "
          f"有效秩={sum(r['erank'] for r in rows)/n:.1f}")
    print(f"{'模块组':<14s}{'层数':>5s}{'能量占比':>10s}{'平均有效秩':>11s}")
    for g, e in sorted(per_group.items(), key=lambda x: -x[1]):
        ers = [r["erank"] for r in rows if group_of(r["name"]) == g]
        print(f"{g:<14s}{per_group_n[g]:>5d}{e/tot*100:>9.2f}%{sum(ers)/len(ers):>11.1f}")
    print("能量最高 5 层：")
    for r in sorted(rows, key=lambda x: -x["energy"])[:5]:
        print(f"   {r['name'][-46:]:<46s} r={r['rank']:<3d} E={r['energy']:8.4g} "
              f"top1={r['top1']*100:5.1f}% cond={r['cond']:7.1f} erank={r['erank']:5.1f}")
    return tot, rows


def verdict(rows):
    n = len(rows)
    top1 = sum(r["top1"] for r in rows) / n
    cond = sum(r["cond"] for r in rows) / n
    if top1 > 0.40 and cond > 10:
        return "谱形健康（adamw 侧）—— 继续训"
    if top1 < 0.15 and cond < 3:
        return "★谱形被抹平（muon 侧）—— 优化器在平摊预算，调 lr 治不了，建议停"
    return "介于两者之间 —— 证据不足，再训一段或与已知基线同步数对比"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="一个或多个 LoRA .safetensors")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    results = []
    for p in args.paths:
        tot, rows = analyze(p, args.device)
        results.append((p, tot, rows))
        print(f"判读：{verdict(rows)}")

    if len(results) > 1:
        print("\n" + "=" * 78)
        print(f"{'文件':<34s}{'总能量':>11s}{'top1':>8s}{'cond':>8s}{'erank':>7s}")
        for p, tot, rows in results:
            n = len(rows)
            print(f"{p.split(chr(92))[-1][-34:]:<34s}{tot:>11.4g}"
                  f"{sum(r['top1'] for r in rows)/n*100:>7.1f}%"
                  f"{sum(r['cond'] for r in rows)/n:>8.1f}"
                  f"{sum(r['erank'] for r in rows)/n:>7.1f}")


if __name__ == "__main__":
    main()
