"""测量训练后 LoKr+DoRA 的有效 ΔW 相对底模的幅度（Frobenius / 谱范数），
用于标定 LoRA-One 初始化的 scale。

python measure_lora_delta.py --base <anima-base.safetensors> --lora <lokr.safetensors>
"""
import argparse

import torch
from safetensors import safe_open


def load(f, key):
    return f.get_tensor(key).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--blocks", default="0,7,14,20,21,27")
    ap.add_argument("--modules", default="self_attn_q_proj,mlp_layer1")
    args = ap.parse_args()

    fb = safe_open(args.base, framework="pt", device="cpu")
    fl = safe_open(args.lora, framework="pt", device="cpu")

    print(f"{'module':<38} {'fro(dW)/fro(W0)':>16} {'s1(dW)/s1(W0)':>14}")
    fro_ratios, spec_ratios = [], []
    for b in args.blocks.split(","):
        for m in args.modules.split(","):
            lkey = f"lora_unet_blocks_{b}_{m}"
            bkey = f"net.blocks.{b}." + m.replace("self_attn_", "self_attn.").replace(
                "cross_attn_", "cross_attn.").replace("mlp_layer", "mlp.layer") + ".weight"
            try:
                w1 = load(fl, f"{lkey}.lokr_w1")
                w2a = load(fl, f"{lkey}.lokr_w2_a")
                w2b = load(fl, f"{lkey}.lokr_w2_b")
                alpha = float(load(fl, f"{lkey}.alpha"))
                ds = load(fl, f"{lkey}.dora_scale").reshape(-1)
                W0 = load(fb, bkey)
            except Exception as e:
                print(f"{lkey:<38} SKIP ({e})")
                continue
            scale = alpha / w2b.shape[0]
            dW = torch.kron(w1, w2a @ w2b) * scale
            merged = W0 + dW
            rn = merged.norm(dim=1, keepdim=True).clamp(min=1e-6)
            eff = ds.view(-1, 1) * merged / rn - W0   # ComfyUI output-axis DoRA 的有效 delta
            fro = (eff.norm() / W0.norm()).item()
            s1_d = torch.svd_lowrank(eff, q=8)[1][0].item()
            s1_w = torch.svd_lowrank(W0, q=8)[1][0].item()
            spec = s1_d / s1_w
            fro_ratios.append(fro)
            spec_ratios.append(spec)
            print(f"{lkey:<38} {fro:>16.5f} {spec:>14.5f}")

    if fro_ratios:
        t = torch.tensor(fro_ratios)
        s = torch.tensor(spec_ratios)
        print(f"\nfro ratio:  median={t.median():.5f}  mean={t.mean():.5f}  max={t.max():.5f}")
        print(f"spec ratio: median={s.median():.5f}  mean={s.mean():.5f}  max={s.max():.5f}")


if __name__ == "__main__":
    main()
