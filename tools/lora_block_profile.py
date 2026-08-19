"""跨模型的 per-block ΔW 能量分布分析（权重侧画风定位）。

对每个 LoKr 模型：逐模块算 ‖ΔW‖_F = (alpha/dim)·‖w1‖_F·‖w2_a@w2_b‖_F
（kron 的 Frobenius 范数可分解，无需展开），按 block 聚合能量占比。
跨几十个风格模型对比，验证 goutong10-1 推理置零扫描的「画风带 7-20」结论
是否普适（权重更新质量集中在哪些块 = 训练实际把容量花在哪）。

用法:
  python lora_block_profile.py --base <anima-base.safetensors> --root D:/models/LoRA/anima
  # 自动取每个子目录最新 step 的 ckpt；--dirs 可指定子集
"""
import argparse
import glob
import os
import re

import torch
from safetensors import safe_open

STEP_RE = re.compile(r"_step(\d+)\.safetensors$")
BLOCK_RE = re.compile(r"lora_unet_blocks_(\d+)_")


def latest_ckpt(d):
    files = [(int(m.group(1)), f) for f in glob.glob(os.path.join(d, "*_step*.safetensors"))
             if (m := STEP_RE.search(f))]
    return max(files)[1] if files else None


def base_norms(base_path):
    """{lora_key_base: ||W0||_F}，由底模键名正向生成 lora 键名。"""
    norms = {}
    with safe_open(base_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            if not k.endswith(".weight") or not k.startswith("net."):
                continue
            path = k[len("net."):-len(".weight")]
            lk = "lora_unet_" + path.replace(".", "_")
            norms[lk] = float(f.get_tensor(k).float().norm())
    return norms


def profile_one(path, w0n):
    """返回 (per_block_energy[dict], adaln_energy, final_energy, total, n_mod, rel_list)"""
    blocks, e_adaln, e_final, total = {}, 0.0, 0.0, 0.0
    rels = []
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        bases = sorted({k.rsplit(".", 1)[0] for k in keys if k.endswith(".lokr_w1")})
        for b in bases:
            w1 = f.get_tensor(f"{b}.lokr_w1").float()
            w2a = f.get_tensor(f"{b}.lokr_w2_a").float()
            w2b = f.get_tensor(f"{b}.lokr_w2_b").float()
            alpha = float(f.get_tensor(f"{b}.alpha")) if f"{b}.alpha" in keys else w2b.shape[0]
            scale = alpha / w2b.shape[0]
            n = scale * float(w1.norm()) * float((w2a @ w2b).norm())
            e = n * n
            total += e
            if w0 := w0n.get(b):
                rels.append(n / w0)
            m = BLOCK_RE.search(b)
            if m:
                blocks[int(m.group(1))] = blocks.get(int(m.group(1)), 0.0) + e
                if "adaln_modulation" in b:
                    e_adaln += e
            elif "final_layer" in b:
                e_final += e
    return blocks, e_adaln, e_final, total, len(bases), rels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--root", help="父目录：每个子目录取最新 step ckpt")
    ap.add_argument("--dirs", nargs="*", help="只分析这些子目录名")
    ap.add_argument("--files", nargs="*", help="直接指定 ckpt 文件")
    ap.add_argument("--out", default="", help="可选 CSV 输出路径")
    args = ap.parse_args()

    w0n = base_norms(args.base)

    targets = []
    if args.files:
        targets = [(os.path.basename(os.path.dirname(f)) or f, f) for f in args.files]
    elif args.root:
        for d in sorted(os.listdir(args.root)):
            if args.dirs and d not in args.dirs:
                continue
            full = os.path.join(args.root, d)
            if os.path.isdir(full) and (ck := latest_ckpt(full)):
                targets.append((d, ck))
    if not targets:
        raise SystemExit("no checkpoints found")

    n_blocks_seen = 0
    profiles = {}   # name -> normalized per-block share (list)
    rows = []
    for name, path in targets:
        try:
            blocks, e_adaln, e_final, total, n_mod, rels = profile_one(path, w0n)
        except Exception as e:
            print(f"{name:<16} SKIP ({e})")
            continue
        if not blocks or total <= 0:
            print(f"{name:<16} SKIP (no lokr blocks)")
            continue
        nb = max(blocks) + 1
        n_blocks_seen = max(n_blocks_seen, nb)
        share = [blocks.get(i, 0.0) / total for i in range(nb)]
        profiles[name] = share
        bands = {
            "b00-06": sum(share[0:7]), "b07-13": sum(share[7:14]),
            "b14-20": sum(share[14:21]), "b21-27": sum(share[21:28]),
        }
        peak = max(range(nb), key=lambda i: share[i])
        med_rel = sorted(rels)[len(rels) // 2] if rels else 0.0
        rows.append((name, os.path.basename(path), bands, peak, e_adaln / total, e_final / total, med_rel))
        print(f"{name:<16} peak=b{peak:02d}  " +
              "  ".join(f"{k}={v:.1%}" for k, v in bands.items()) +
              f"  adaln={e_adaln / total:.1%} final={e_final / total:.2%} med|dW|/|W0|={med_rel:.4f}  ({n_mod}模块)")

    if len(profiles) > 1 and n_blocks_seen:
        import statistics
        print("\n=== 跨模型平均 per-block 能量占比 ===")
        for i in range(n_blocks_seen):
            vals = [p[i] for p in profiles.values() if len(p) > i]
            bar = "#" * int(statistics.mean(vals) * 400)
            print(f"b{i:02d}  {statistics.mean(vals):6.2%} ±{statistics.pstdev(vals):5.2%}  {bar}")

    if args.out:
        import csv
        with open(args.out, "w", newline="", encoding="utf-8") as fo:
            w = csv.writer(fo)
            w.writerow(["model", "ckpt", "b00-06", "b07-13", "b14-20", "b21-27",
                        "peak_block", "adaln_share", "final_share", "median_rel"])
            for name, ck, bands, peak, ad, fi, mr in rows:
                w.writerow([name, ck, *(f"{bands[k]:.4f}" for k in ("b00-06", "b07-13", "b14-20", "b21-27")),
                            peak, f"{ad:.4f}", f"{fi:.5f}", f"{mr:.5f}"])
        print(f"\nCSV -> {args.out}")


if __name__ == "__main__":
    main()
