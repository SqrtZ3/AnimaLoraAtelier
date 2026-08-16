#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""算子级定位：2D×3D 广播 matmul 的**左操作数**梯度在本平台上对不对。

**为什么查这个。** `trainer/lora.py` 的 `LoKrLayer._compute` 最后一步是

    y = torch.matmul(w1, tmp)      # w1: (F, F) 2D    tmp: (P, F, D) 3D

w1 被广播到 P 个 batch 上。它的 backward 要把梯度沿**被广播的 P 维求和归约**：

    dL/dw1[i,j] = Σ_p Σ_o  dL/dy[p,i,o] · tmp[p,j,o]

而 tmp 那侧不涉及任何广播归约。上一层诊断（`diag_dora_w1_grad.py`）在昇腾上测到的
正是这个形状的偏差——只有 w1.grad 错，w2_a / w2_b / dora_scale 的梯度 cos 都
> 0.99997：

    self_attn.q_proj   w1.grad   cos=-0.527   相对误差 640%
    mlp.layer1         w1.grad   cos=+0.507   相对误差 1154%

假设：torch_npu 对"左操作数被广播"的 matmul backward 归约错了。本脚本用纯算子
（不含任何 LoRA 代码）复现它，并同时测几种数学等价的改写，找出平台上算对的那个。

用法::

    python AnimaLoraToolkit/tests/diag_npu_broadcast_matmul_grad.py --device npu
    python AnimaLoraToolkit/tests/diag_npu_broadcast_matmul_grad.py --device cuda   # 阳性对照

CPU 作黄金参考，同一进程内跑，秒级。
"""
from __future__ import annotations

import argparse

import torch

F_DIM = 4        # LoKr factor
D_DIM = 512      # out_dim
P_DIM = 2048     # 展平后的 token 数（真实量级：batch 8 × seq 256）
SEED = 20260816


def make(dev: str, dtype: torch.dtype):
    g = torch.Generator(device="cpu").manual_seed(SEED)
    w1 = torch.randn(F_DIM, F_DIM, generator=g, dtype=torch.float32)
    tmp = torch.randn(P_DIM, F_DIM, D_DIM, generator=g, dtype=torch.float32)
    gy = torch.randn(P_DIM, F_DIM, D_DIM, generator=g, dtype=torch.float32)
    w1 = w1.to(dev, dtype).requires_grad_(True)
    tmp = tmp.to(dev, dtype).requires_grad_(True)
    return w1, tmp, gy.to(dev, dtype)


# ── 几种数学等价的写法 ────────────────────────────────────────────
def impl_broadcast(w1, tmp):
    """现状：2D × 3D，左操作数隐式广播。"""
    return torch.matmul(w1, tmp)


def impl_einsum(w1, tmp):
    return torch.einsum("ij,pjo->pio", w1, tmp)


def impl_expand_bmm(w1, tmp):
    """显式 expand 成真 bmm（仍然依赖广播归约，只是把它写出来）。"""
    return torch.bmm(w1.unsqueeze(0).expand(tmp.shape[0], -1, -1), tmp)


def impl_flat2d(w1, tmp):
    """重排成**纯 2D mm**，完全避开广播归约。

    tmp (P,F,D) --transpose--> (F,P,D) --reshape--> (F, P*D)
    w1 (F,F) @ (F, P*D) -> (F, P*D) --reshape--> (F,P,D) --transpose--> (P,F,D)
    """
    P, Fd, D = tmp.shape
    flat = tmp.transpose(0, 1).reshape(Fd, P * D)
    out = torch.matmul(w1, flat)
    return out.reshape(Fd, P, D).transpose(0, 1)


def impl_right_operand(w1, tmp):
    """把 w1 挪到右操作数：(P,D,F) @ (F,F)^T。"""
    return torch.matmul(tmp.transpose(1, 2), w1.transpose(0, 1)).transpose(1, 2)


IMPLS = [
    ("matmul(w1, tmp) 现状", impl_broadcast),
    ("einsum ij,pjo->pio", impl_einsum),
    ("expand + bmm", impl_expand_bmm),
    ("重排成纯 2D mm", impl_flat2d),
    ("w1 挪到右操作数", impl_right_operand),
]


def run(dev: str, dtype: torch.dtype, fn):
    w1, tmp, gy = make(dev, dtype)
    y = fn(w1, tmp)
    y.backward(gy)
    return (w1.grad.detach().float().cpu(),
            tmp.grad.detach().float().cpu(),
            y.detach().float().cpu())


def cmp(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    cos = (torch.dot(a.flatten(), b.flatten())
           / (a.norm() * b.norm() + 1e-30)).item()
    rel = ((a - b).norm() / (b.norm() + 1e-30)).item()
    return cos, rel


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="2D×3D 广播 matmul 的左操作数梯度诊断")
    ap.add_argument("--device", default="cuda", help="npu / cuda / cpu")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    args = ap.parse_args(argv)
    dev = args.device
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16

    if dev.startswith("npu"):
        try:
            import torch_npu  # noqa: F401
            import torch_npu.contrib.transfer_to_npu  # noqa: F401
        except Exception as e:  # pragma: no cover
            print(f"导入 torch_npu 失败：{e}")
            return 1

    print(f"设备={dev}  dtype={args.dtype}  torch={torch.__version__}")
    print(f"形状：w1 ({F_DIM},{F_DIM})  tmp ({P_DIM},{F_DIM},{D_DIM})")
    print()

    # CPU 黄金参考：用现状写法 + fp32
    ref_w1g, ref_tmpg, ref_y = run("cpu", torch.float32, impl_broadcast)

    print(f"{'写法':<24} {'前向 cos':>10} {'w1.grad cos':>13} {'w1.grad relerr':>16} "
          f"{'tmp.grad cos':>13}")
    print("-" * 82)
    ok = []
    for name, fn in IMPLS:
        try:
            w1g, tmpg, y = run(dev, dtype, fn)
        except Exception as e:
            print(f"{name:<24} 跑不起来：{type(e).__name__}: {e}")
            continue
        cy, _ = cmp(y, ref_y)
        c1, r1 = cmp(w1g, ref_w1g)
        c2, _ = cmp(tmpg, ref_tmpg)
        good = (c1 > 0.999 and r1 < 0.05)
        ok.append((name, good, c1, r1))
        print(f"{name:<24} {cy:>10.6f} {c1:>13.6f} {r1:>15.3%} {c2:>13.6f}"
              f"{'' if good else '   ← 错'}")

    print()
    print("判读：所有写法在数学上完全等价，前向 cos 都应 ≈ 1。")
    print("      若某些写法的 w1.grad 偏离而另一些正确 → 平台的 backward 有 bug，")
    print("      且正确的那个写法就是可用的 workaround。")
    good_names = [n for n, g, _, _ in ok if g]
    bad_names = [n for n, g, _, _ in ok if not g]
    if bad_names and good_names:
        print()
        print(f"  ✗ 算错：{', '.join(bad_names)}")
        print(f"  ✓ 算对：{', '.join(good_names)}")
        print("  → 把 trainer/lora.py:973 的 y = torch.matmul(w1, tmp) 换成上面算对的写法。")
    elif not bad_names:
        print()
        print("  全部正确 —— 这个算子不是根因，回到上一层诊断另找。")
    else:
        print()
        print("  全部偏离 —— 先确认 CPU 参考本身没问题（换 --dtype fp32 再跑一次）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
