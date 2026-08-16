#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诊断：DoRA-LoKr 里 w1 收到的"收缩压力"在本平台上还在不在。

**要裁决的现象。** 同一套 C12 配方、**同一个训练集**（villainchin），CUDA 与昇腾
训出的 LoKr 成品，总增益因子 w1 的走向完全相反：

    CUDA：‖w1‖ 0.340 → 0.243（3 epoch），‖ΔW‖ 冲高后回落锁定 ~35.7
    昇腾：‖w1‖ 0.374 → 0.382（3 epoch），‖ΔW‖ 单调膨胀，12 epoch 到 113.2

从两边 training_state 里读 optimizer 动量做的方向取证（同 step 数对比）：

    cos(exp_avg, w1) 中位     CUDA +0.479（87.1% 的模块在收缩 w1）
                              昇腾 -0.134（只有 34.3%）

梯度量级、eps、lr 都排除了（|m| 同量级，sqrt(v)≈4.6e-6 ≫ eps=1e-8）。所以问题
不在步长，在**方向**：昇腾上 w1 没有收到那股一致的收缩梯度。

**收缩压力从哪来。** DoRA 把权重写成 ``m · (W+ΔW)/‖W+ΔW‖``。分母让"纯放大 ΔW 的
尺度"这个方向变得无效——放大多少就被除掉多少，于是尺度方向的梯度会自我抵消。
LoKr 下 ΔW = s·kron(w1, U)，w1 正是那个纯尺度/增益因子，所以这股抵消力**几乎全部
落在 w1 上**。实现见 ``trainer/lora.py`` 的 ``LoKrLayer.merged_row_norms``
（免物化算 ‖W+ΔW‖，梯度要能穿过它回到 w1/U）。

**本脚本的两个检验（第一个不需要第二台机器）：**

1. **DoRA 开/关 的自对照（自包含，单机可判）**
   同一层、同一输入、同一 loss，只切 ``lora_variant`` dora/base，比较 w1 梯度的
   **径向分量**（沿 w1 方向的投影）。DoRA 的归一化若正常生效，两者应有明显差别，
   且 DoRA 那侧的径向分量会明显偏向"收缩"。若两者几乎相同 → ‖W+ΔW‖ 的梯度通路
   在本平台上断了/被削平，这就是根因。

2. **跨设备对拍**（可选，有 CPU 就能做）
   同 seed 同输入，在目标设备与 CPU 各算一遍梯度，比 cos 与相对误差。
   CPU 是黄金参考，能把"平台算错"和"配方本身如此"分开。

用法::

    python AnimaLoraToolkit/tests/diag_dora_w1_grad.py --device npu
    python AnimaLoraToolkit/tests/diag_dora_w1_grad.py --device cuda   # 阳性对照
    python AnimaLoraToolkit/tests/diag_dora_w1_grad.py --device npu --no-ref   # 只做检验 1

单层、秒级，不需要底模、不需要数据集。
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trainer.lora import LoRALinear  # noqa: E402

# 真实层形状（Anima 2048ch）：self_attn.q_proj 与 mlp.layer1
SHAPES = [
    ("self_attn.q_proj", 2048, 2048, 64, 64.0),   # in, out, rank, alpha
    ("mlp.layer1", 2048, 8192, 48, 48.0),
]
FACTOR = 4
SEED = 1234


def _autocast_dev(dev: str) -> str:
    return "cpu" if dev == "cpu" else dev.split(":")[0]


def build(dev: str, in_f: int, out_f: int, rank: int, alpha: float, variant: str):
    """同 seed 造一层，保证跨设备/跨 variant 的初值逐字节一致。

    ⚠ **必须 warm-start**：LoKr 的 ``lokr_w2_b`` init 为零（保证 step-0 净 ΔW=0），
    此时 ``∂L/∂w1 ∝ w2_a@w2_b = 0``、``∂L/∂w2_a ∝ w2_b = 0`` —— w1 和 w2_a 的梯度
    **天然是零**，在 init 点上什么也测不出来（这是 LoKr 的正常性质，不是 bug）。
    所以这里给 w2_b 填一个固定 seed 的小随机值，把层挪到"训练中"的工作点。
    """
    torch.manual_seed(SEED)
    base = torch.nn.Linear(in_f, out_f, bias=False)
    base.weight.data = base.weight.data.to(torch.bfloat16)
    base = base.to(dev)
    torch.manual_seed(SEED)          # adapter 的 init 也吃这个 seed
    layer = LoRALinear(
        base, rank=rank, alpha=alpha, use_lokr=True, factor=FACTOR,
        rank_dropout=0.0, module_dropout=0.0,      # 关掉随机性，保证可对拍
        lora_variant=variant,
        lokr_w1_init_std=0.1, lokr_compute_dtype="fp32",
    ).to(dev)
    # warm-start：w2_b 从 0 挪开，量级取自真实 checkpoint（‖w2_a@w2_b‖ 在 ep1~ep3
    # 是 6~10，对应 w2_b 每元素 ~1e-2）。在 CPU 上生成再搬过去，保证跨设备逐位一致。
    g = torch.Generator(device="cpu").manual_seed(SEED + 31)
    wb = layer.adapter.lokr_w2_b
    with torch.no_grad():
        wb.copy_(torch.randn(*wb.shape, generator=g, dtype=torch.float32).mul_(1e-2)
                 .to(device=wb.device, dtype=wb.dtype))
    layer.train()
    return layer


def grads(layer, dev: str, in_f: int, batch: int = 8, seq: int = 256):
    g = torch.Generator(device="cpu").manual_seed(SEED + 7)
    x = torch.randn(batch, seq, in_f, generator=g, dtype=torch.float32).to(dev, torch.bfloat16)
    tgt = torch.randn(batch, seq, layer.original.out_features, generator=g,
                      dtype=torch.float32).to(dev, torch.bfloat16)
    for p in layer.parameters():
        if p.grad is not None:
            p.grad = None
    with torch.autocast(_autocast_dev(dev), dtype=torch.bfloat16):
        y = layer(x)
        loss = torch.nn.functional.mse_loss(y.float(), tgt.float())
    loss.backward()
    ad = layer.adapter
    out = {
        "w1": ad.lokr_w1.detach().float().cpu(),
        "w1.grad": ad.lokr_w1.grad.detach().float().cpu(),
        "w2_a.grad": ad.lokr_w2_a.grad.detach().float().cpu(),
        "w2_b.grad": ad.lokr_w2_b.grad.detach().float().cpu(),
        "loss": loss.item(),
    }
    if getattr(layer, "dora_scale", None) is not None and layer.dora_scale.grad is not None:
        out["dora_scale.grad"] = layer.dora_scale.grad.detach().float().cpu()
    return out


def radial(w1: torch.Tensor, g: torch.Tensor) -> tuple[float, float]:
    """返回 (cos(g, w1), 更新的径向分量 -<w1,g>/‖w1‖)。

    梯度下降走 -g，所以 -<w1,g>/‖w1‖ > 0 表示这一步在**放大** ‖w1‖，< 0 表示收缩。
    """
    wf, gf = w1.flatten(), g.flatten()
    denom = wf.norm() * gf.norm()
    cos = (torch.dot(wf, gf) / denom).item() if denom > 0 else float("nan")
    rad = (-torch.dot(wf, gf) / wf.norm()).item()
    return cos, rad


def relerr(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / (b.norm() + 1e-30)).item()


def cos_of(a: torch.Tensor, b: torch.Tensor) -> float:
    return (torch.dot(a.flatten(), b.flatten())
            / (a.norm() * b.norm() + 1e-30)).item()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DoRA-LoKr w1 梯度方向诊断")
    ap.add_argument("--device", default="cuda", help="npu / cuda / cpu")
    ap.add_argument("--no-ref", action="store_true", help="跳过与 CPU 的跨设备对拍")
    args = ap.parse_args(argv)
    dev = args.device

    if dev.startswith("npu"):
        try:
            import torch_npu  # noqa: F401
            import torch_npu.contrib.transfer_to_npu  # noqa: F401
        except Exception as e:  # pragma: no cover - 只在昇腾上走到
            print(f"导入 torch_npu 失败：{e}")
            return 1

    print(f"设备={dev}  torch={torch.__version__}")
    print()

    print("=" * 78)
    print("检验 1：DoRA 开/关 的自对照 —— 归一化的梯度通路还在不在")
    print("=" * 78)
    print(f"{'层':<18} {'variant':<6} {'cos(grad,w1)':>13} {'径向分量':>13} {'loss':>10}")
    verdicts = []
    for name, in_f, out_f, rank, alpha in SHAPES:
        rads = {}
        for variant in ("dora", "base"):
            layer = build(dev, in_f, out_f, rank, alpha, variant)
            r = grads(layer, dev, in_f)
            c, rad = radial(r["w1"], r["w1.grad"])
            rads[variant] = rad
            print(f"{name:<18} {variant:<6} {c:>13.4f} {rad:>13.3e} {r['loss']:>10.4f}")
        d, b = rads["dora"], rads["base"]
        rel = abs(d - b) / (abs(b) + 1e-30)
        verdicts.append((name, d, b, rel))
        print(f"{'':<18} {'差异':<6} dora 与 base 的径向分量相对差 = {rel:.3%}"
              f"{'  ← 几乎没差别，可疑' if rel < 0.05 else ''}")
    print()
    print("判读（只看**有没有差别**这个二值判据）：")
    print("      DoRA 的 ‖W+ΔW‖ 归一化生效时，w1 的梯度必然多出一条经过分母的通路，")
    print("      径向分量因此与关掉 DoRA 时显著不同。")
    print("      ▸ 相对差是几十个百分点量级 → 通路正常（CUDA 阳性对照实测 91% / 105%）。")
    print("      ▸ 相对差 ≈ 0 → ‖W+ΔW‖ 的梯度没有回到 w1，**这就是根因**。")
    print("      ⚠ 不要解读径向分量的**正负号**：这里的 loss 是对随机 target 的 MSE，")
    print("        不是真实训练目标，符号没有可比性（CUDA 对照上 mlp.layer1 的 dora 侧")
    print("        就是正的）。有没有差别才是判据。")
    print()

    if args.no_ref or dev == "cpu":
        return 0

    print("=" * 78)
    print("检验 2：与 CPU 的跨设备梯度对拍（CPU = 黄金参考）")
    print("=" * 78)
    print(f"{'层':<18} {'张量':<14} {'cos':>10} {'相对误差':>12}")
    worst = 0.0
    for name, in_f, out_f, rank, alpha in SHAPES:
        ref = grads(build("cpu", in_f, out_f, rank, alpha, "dora"), "cpu", in_f)
        cur = grads(build(dev, in_f, out_f, rank, alpha, "dora"), dev, in_f)
        for key in ("w1.grad", "w2_a.grad", "w2_b.grad", "dora_scale.grad"):
            if key not in ref or key not in cur:
                continue
            c = cos_of(cur[key], ref[key])
            e = relerr(cur[key], ref[key])
            worst = max(worst, e)
            flag = "  ← 偏离过大" if (c < 0.99 or e > 0.05) else ""
            print(f"{name:<18} {key:<14} {c:>10.6f} {e:>12.3%}{flag}")
    print()
    print(f"最大相对误差 = {worst:.3%}")
    print("判读：bf16 下 cos > 0.99、相对误差 < 5% 属正常数值抖动；")
    print("      w1.grad 的 cos 明显偏低（更别说变号）→ 平台把这个梯度算错了。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
