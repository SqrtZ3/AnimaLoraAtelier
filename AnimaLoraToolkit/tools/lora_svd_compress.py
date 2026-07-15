# -*- coding: utf-8 -*-
"""标准 LoRA 成品 → 逐层 SVD 自适应截断的本地压缩工具。

对每个 LoRA 层的 ΔW = scaling·(B@A) 做 SVD，按**该层自己的能量谱**保留累计
能量 ≥ --energy 的最小秩，重新发成逐层变 rank 的标准 LoRA（ComfyUI 直载）。

与"从静态表手动分配 rank"不同：本工具用的是**这次训练模型自己的谱**，因此换
数据集也自动自适应；截断损失（被丢弃的能量占比）逐层报告，--energy 1.0 时输出
与输入逐 bit 等价（fp32 QR-SVD 往返容差内）。QR 技巧，不物化 (out,in) 全矩阵。

用法（本地 GPU python）：
  python tools/lora_svd_compress.py in_lora.safetensors out_lora.safetensors \
      [--energy 0.99] [--max-rank 24] [--fp32]

依据：AC-LoRA (arXiv:2504.02231) 的信号-rank 抽取思想；截断准则与
tools/abba_export_lora.py 一致（保留累计能量 ≥ energy 的最小秩）。
"""
import argparse

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainer.lora import svd_truncate_lora_pair  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="标准 LoRA safetensors（含 lora_down/lora_up 键）")
    ap.add_argument("output", help="输出压缩后的标准 LoRA safetensors 路径")
    ap.add_argument("--energy", type=float, default=0.99,
                    help="每层保留累计能量阈值 (0,1]，1.0=不截断（默认 0.99）")
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

    # DoRA 成品的 ΔW 不是单纯 B@A（还有 dora_scale 的逐行重归一化），单纯截断 B@A
    # 会与训练语义不一致 → fail-fast，别悄悄产出错误成品。
    dora_keys = [k for k in tensors if k.endswith(".dora_scale")]
    if dora_keys:
        raise SystemExit(
            f"输入含 {len(dora_keys)} 个 dora_scale 键（DoRA 成品）。DoRA 的 ΔW 不是"
            "单纯 B@A，本工具不支持；请用非 DoRA 的标准 LoRA 成品。")

    bases = sorted({k[: -len(".lora_down.weight")]
                    for k in tensors if k.endswith(".lora_down.weight")})
    if not bases:
        raise SystemExit("输入文件没有 lora_down.weight 键 —— 不是标准 LoRA 成品？"
                         "（LoKr/ABBA native 成品请用对应工具）")

    out_sd = {}
    dev = args.device
    store_dtype = torch.float32 if args.fp32 else torch.bfloat16
    tot_in, tot_out, worst = 0, 0, (0.0, "")
    for base in bases:
        A = tensors[f"{base}.lora_down.weight"].to(dev, torch.float32)   # (r, in)
        B = tensors[f"{base}.lora_up.weight"].to(dev, torch.float32)     # (out, r)
        r, in_f = A.shape
        out_f = B.shape[0]
        # scaling = alpha / rank（无 alpha 键则默认 alpha=rank → scaling=1）
        alpha_key = f"{base}.alpha"
        alpha = float(tensors[alpha_key]) if alpha_key in tensors else float(r)
        scaling = alpha / max(r, 1)

        down, up, keep, dropped = svd_truncate_lora_pair(
            A, B, scaling, energy=args.energy, max_rank=args.max_rank)
        if dropped > worst[0]:
            worst = (dropped, base)
        out_sd[f"{base}.lora_down.weight"] = down.to(store_dtype).cpu().contiguous()
        out_sd[f"{base}.lora_up.weight"] = up.to(store_dtype).cpu().contiguous()
        out_sd[f"{base}.alpha"] = torch.tensor(float(keep))   # alpha=keep → scaling=1
        tot_in += r * (in_f + out_f)
        tot_out += keep * (in_f + out_f)
        print(f"  {base.removeprefix('lora_unet_'):<44} r={r:>3} -> keep={keep:>3}"
              f"  丢弃能量 {dropped:.3%}")

    bpp = 4 if args.fp32 else 2
    max_keep = max(int(t.shape[0]) for k, t in out_sd.items()
                   if k.endswith("lora_down.weight"))
    meta_out = {
        "ss_network_module": "networks.lora",
        "ss_network_dim": str(max_keep),
        "ss_network_alpha": "per-layer",
        "anima_lora_variant": "svd_compressed",
        "anima_compress_source": args.input,
        "anima_compress_energy": f"{args.energy}",
        "anima_compress_max_rank": str(args.max_rank or 0),
    }
    for k in ("anima_training_config", "anima_config_schema"):
        if k in meta:
            meta_out[k] = meta[k]
    save_file(out_sd, args.output, metadata=meta_out)
    print(f"\n完成: {args.output}")
    print(f"  层数 {len(bases)}，原 {tot_in * 2 / 1e6:.1f} MB(bf16) → 压缩 {tot_out * bpp / 1e6:.1f} MB"
          f"（能量阈值 {args.energy}，max_rank {args.max_rank or '∞'}）")
    if worst[0] > 0:
        print(f"  最大逐层丢弃能量: {worst[0]:.3%} @ {worst[1]}")


if __name__ == "__main__":
    main()
