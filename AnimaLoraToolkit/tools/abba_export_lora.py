# -*- coding: utf-8 -*-
"""ABBA native 成品 → ComfyUI 可载标准 LoRA 的本地转换/压缩工具。

云端 epoch 成品默认只含 native 因子（abba_a1/b1/a2/b2 + alpha1/alpha2，
体积 = 同预算标准 LoRA），本工具在本地完成：
  1. KR 物化：ΔW = √α1√α2 · (B1A1)∘(B2A2) = s·B_kr@A_kr（精确恒等，rank=r1·r2）
  2. 可选 SVD 截断压缩（QR 技巧，不物化 (out,in) 全矩阵）：
     --energy E     每层保留累计能量 ≥ E 的最小秩（默认 1.0 = 不截断）
     --max-rank N   每层秩上限（与 --energy 取更紧者）
  3. 输出 kohya 标准键（lora_down/lora_up/alpha，alpha=保留秩 → scaling=1，
     s 已折进因子），ComfyUI（含 Bypass loader）直载。

用法（本地 GPU python）：
  python tools/abba_export_lora.py in_abba.safetensors out_lora.safetensors \
      [--energy 0.999] [--max-rank 128] [--fp32]

截断损失以"每层被丢弃的能量占比"逐层报告；--energy 1.0 时输出与训练前向
逐 bit 等价（bf16 存储容差内）。
"""
import argparse
import math

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="ABBA native safetensors（含 abba_a1/b1/a2/b2 键）")
    ap.add_argument("output", help="输出标准 LoRA safetensors 路径")
    ap.add_argument("--energy", type=float, default=1.0,
                    help="每层保留累计能量阈值 (0,1]，1.0=不截断（默认）")
    ap.add_argument("--max-rank", type=int, default=0,
                    help="每层秩上限，0=不限（默认）")
    ap.add_argument("--fp32", action="store_true",
                    help="以 fp32 存储因子（默认 bf16，与训练精度一致）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if not (0.0 < args.energy <= 1.0):
        raise SystemExit(f"--energy 必须在 (0,1]，得到 {args.energy}")

    tensors, meta = {}, {}
    with safe_open(args.input, "pt") as f:
        meta = dict(f.metadata() or {})
        for k in f.keys():
            tensors[k] = f.get_tensor(k)

    bases = sorted({k[: -len(".abba_a1")] for k in tensors if k.endswith(".abba_a1")})
    if not bases:
        raise SystemExit("输入文件没有 abba_a1 键 —— 不是 ABBA native 成品？"
                         "（KR 物化成品本身已是标准 LoRA，无需转换）")

    out_sd = {}
    dev = args.device
    store_dtype = torch.float32 if args.fp32 else torch.bfloat16
    tot_in, tot_out, worst = 0, 0, (0.0, "")
    for base in bases:
        a1 = tensors[f"{base}.abba_a1"].to(dev, torch.float32)
        b1 = tensors[f"{base}.abba_b1"].to(dev, torch.float32)
        a2 = tensors[f"{base}.abba_a2"].to(dev, torch.float32)
        b2 = tensors[f"{base}.abba_b2"].to(dev, torch.float32)
        alpha1 = float(tensors.get(f"{base}.abba_alpha1", torch.tensor(float(a1.shape[0]))))
        alpha2 = float(tensors.get(f"{base}.abba_alpha2", torch.tensor(float(a2.shape[0]))))
        s = math.sqrt(alpha1) * math.sqrt(alpha2)
        r1, r2 = a1.shape[0], a2.shape[0]
        out_f, in_f = b1.shape[0], a1.shape[1]

        # KR 因子（不物化 out×in）：ΔW = s · b_kr @ a_kr
        a_kr = (a1.unsqueeze(1) * a2.unsqueeze(0)).reshape(r1 * r2, in_f)
        b_kr = (b1.unsqueeze(2) * b2.unsqueeze(1)).reshape(out_f, r1 * r2)

        # SVD via QR 技巧：b_kr=QbRb, a_krᵀ=QaRa → ΔW/s = Qb (Rb Raᵀ) Qaᵀ
        Qb, Rb = torch.linalg.qr(b_kr)
        Qa, Ra = torch.linalg.qr(a_kr.t())
        Uc, S, Vhc = torch.linalg.svd(Rb @ Ra.t())
        e = S ** 2
        e_tot = e.sum().clamp(min=1e-30)
        ce = torch.cumsum(e, 0) / e_tot
        keep = int((ce < args.energy).sum().item()) + 1 if args.energy < 1.0 else len(S)
        if args.max_rank > 0:
            keep = min(keep, args.max_rank)
        keep = max(min(keep, len(S)), 1)
        dropped = float(1.0 - ce[keep - 1].item()) if keep < len(S) else 0.0
        if dropped > worst[0]:
            worst = (dropped, base)

        # up = Qb Uc √S·√s, down = √s·√S Vhc Qaᵀ；alpha=keep → scaling=1
        s_sqrt = (S[:keep] * s).clamp(min=0).sqrt()
        up = (Qb @ Uc[:, :keep]) * s_sqrt.unsqueeze(0)         # (out, keep)
        down = (Vhc[:keep] @ Qa.t()) * s_sqrt.unsqueeze(1)     # (keep, in)
        out_sd[f"{base}.lora_up.weight"] = up.to(store_dtype).cpu().contiguous()
        out_sd[f"{base}.lora_down.weight"] = down.to(store_dtype).cpu().contiguous()
        out_sd[f"{base}.alpha"] = torch.tensor(float(keep))
        tot_in += (r1 + r2) * (in_f + out_f)
        tot_out += keep * (in_f + out_f)
        print(f"  {base.removeprefix('lora_unet_'):<44} kr={r1 * r2:>4} -> keep={keep:>4}"
              f"  丢弃能量 {dropped:.3%}")

    bpp = 4 if args.fp32 else 2
    meta_out = {
        "ss_network_module": "networks.lora",
        "ss_network_dim": str(max(int(t.shape[0]) for k, t in out_sd.items()
                                  if k.endswith("lora_down.weight"))),
        "ss_network_alpha": "per-layer",
        "anima_lora_variant": "abba_kr_export",
        "anima_abba_source": args.input,
        "anima_abba_energy": f"{args.energy}",
    }
    # 透传训练配置快照（若有）
    for k in ("anima_training_config", "anima_config_schema"):
        if k in meta:
            meta_out[k] = meta[k]
    save_file(out_sd, args.output, metadata=meta_out)
    print(f"\n完成: {args.output}")
    print(f"  层数 {len(bases)}，native {tot_in * 2 / 1e6:.1f} MB → 导出 {tot_out * bpp / 1e6:.1f} MB"
          f"（能量阈值 {args.energy}，max_rank {args.max_rank or '∞'}）")
    if worst[0] > 0:
        print(f"  最大逐层丢弃能量: {worst[0]:.3%} @ {worst[1]}")


if __name__ == "__main__":
    main()
