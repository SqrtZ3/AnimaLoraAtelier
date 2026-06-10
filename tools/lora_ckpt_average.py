"""Post-hoc checkpoint averaging for LoKr/DoRA LoRA files (LAWA sliding window /
EDM2 post-hoc-EMA style snapshot weighting).

工作在因子空间（直接平均 lokr_w1 / w2_a / w2_b / dora_scale）。kron 与矩阵乘是双线性
的，对相邻 ckpt（schedule-free 轨迹上 40 步间隔）这是二阶小量近似；--check 会重建
ΔW = (alpha/dim)·kron(w1, w2_a@w2_b) 实测因子平均 vs 真 ΔW 平均的相对误差，
误差 ~1e-2 以下时因子平均的结论可信。

用法（ComfyUI python 运行）:
  python lora_ckpt_average.py --dir D:/models/LoRA/anima/goutong10-1 --last 4
  python lora_ckpt_average.py --dir ... --last 8 --decay 0.995   # EMA 风格快照加权
  python lora_ckpt_average.py --files a.safetensors b.safetensors --out avg.safetensors

--decay D: 快照权重 w_i ∝ D^(step_max - step_i)，D=1 为均匀（LAWA）。
"""
import argparse
import glob
import os
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def step_of(path: str) -> int:
    m = re.search(r"_step(\d+)\.safetensors$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def load_all(path: str):
    tensors, meta = {}, {}
    with safe_open(path, framework="pt", device="cpu") as f:
        meta = dict(f.metadata() or {})
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    return tensors, meta


def lokr_delta_w(t: dict, base: str) -> torch.Tensor:
    """Rebuild ΔW = (alpha/dim) * kron(w1, w2_a @ w2_b) in float32."""
    w1 = t[f"{base}.lokr_w1"].float()
    w2a = t[f"{base}.lokr_w2_a"].float()
    w2b = t[f"{base}.lokr_w2_b"].float()
    alpha = float(t[f"{base}.alpha"]) if f"{base}.alpha" in t else None
    dim = w2b.shape[0]
    scale = (alpha / dim) if alpha else 1.0
    return torch.kron(w1, w2a @ w2b) * scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", help="checkpoint directory (expects *_stepN.safetensors)")
    ap.add_argument("--last", type=int, default=4, help="use the last N checkpoints")
    ap.add_argument("--steps", help="comma list of explicit step numbers, e.g. 1080,1120,1200")
    ap.add_argument("--files", nargs="*", help="explicit file list (overrides --dir)")
    ap.add_argument("--decay", type=float, default=1.0,
                    help="per-step snapshot decay; w_i = decay^(step_max-step_i); 1.0=uniform")
    ap.add_argument("--out", help="output path (default <dir>/avg/<name>_avg<lo>-<hi>[_d<decay>].safetensors)")
    ap.add_argument("--check", type=int, default=4,
                    help="N module bases for factor-avg vs true-ΔW-avg error report (0=off)")
    args = ap.parse_args()

    if args.files:
        files = sorted(args.files, key=step_of)
    else:
        if not args.dir:
            sys.exit("need --dir or --files")
        files = sorted(glob.glob(os.path.join(args.dir, "*_step*.safetensors")), key=step_of)
        if args.steps:
            want = {int(s) for s in args.steps.split(",")}
            files = [f for f in files if step_of(f) in want]
        else:
            files = files[-args.last:]
    if len(files) < 2:
        sys.exit(f"need >=2 checkpoints, got {len(files)}")

    steps = [step_of(f) for f in files]
    smax = max(steps)
    raw_w = [args.decay ** (smax - s) for s in steps]
    tot = sum(raw_w)
    weights = [w / tot for w in raw_w]
    print("averaging:")
    for f, s, w in zip(files, steps, weights):
        print(f"  step {s:>6}  weight {w:.4f}  {os.path.basename(f)}")

    all_t = []
    meta = {}
    for f in files:
        t, m = load_all(f)
        all_t.append(t)
        meta = m  # keep newest (files sorted ascending)
    keys = set(all_t[0])
    for t in all_t[1:]:
        if set(t) != keys:
            sys.exit("key sets differ between checkpoints; refusing to average")

    out_t = {}
    for k in sorted(keys):
        acc = all_t[0][k].float() * weights[0]
        for t, w in zip(all_t[1:], weights[1:]):
            acc = acc + t[k].float() * w
        out_t[k] = acc.to(all_t[-1][k].dtype)

    # --- factor-space vs ΔW-space error report -----------------------------
    if args.check > 0:
        bases = sorted({k.rsplit(".", 1)[0] for k in keys if k.endswith(".lokr_w1")})
        # pick the largest modules (mlp first) for a conservative estimate
        bases.sort(key=lambda b: -all_t[0][f"{b}.lokr_w2_a"].numel())
        print("\nfactor-avg vs true-deltaW-avg relative error (Frobenius):")
        for b in bases[: args.check]:
            true_avg = None
            for t, w in zip(all_t, weights):
                d = lokr_delta_w(t, b) * w
                true_avg = d if true_avg is None else true_avg + d
            factor_avg = lokr_delta_w(out_t, b)
            rel = (factor_avg - true_avg).norm() / true_avg.norm().clamp(min=1e-12)
            print(f"  {b}: rel_err={rel.item():.3e}")
        print("  (<=1e-2: factor averaging is a faithful proxy for true weight averaging)")

    out = args.out
    if not out:
        d = args.dir or os.path.dirname(files[0])
        name = re.sub(r"_step\d+\.safetensors$", "", os.path.basename(files[-1]))
        tag = f"_avg{min(steps)}-{max(steps)}"
        if args.decay != 1.0:
            tag += f"_d{args.decay}"
        out = os.path.join(d, "avg", f"{name}{tag}.safetensors")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    meta["anima_ckpt_average"] = (
        f"steps={','.join(map(str, steps))};decay={args.decay};"
        f"weights={','.join(f'{w:.4f}' for w in weights)}"
    )
    save_file(out_t, out, metadata=meta)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
