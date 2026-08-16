# -*- coding: utf-8 -*-
"""回归测试：LoKr 的 w1 梯度不得依赖"2D×3D 广播 matmul"的 backward。

背景（2026-08-16 昇腾真机取证，归因全过程见 docs/ascend-npu.md §4.2）：
`LoKrLayer._compute` 最后一步把 (factor,factor) 的 w1 作用到 (B*,factor,out_dim) 上。
原实现写作 `torch.matmul(w1, tmp)`——2D × 3D，左操作数隐式广播。torch_npu 2.6 上
这条路径**前向逐位正确、backward 对左操作数算错**（dL/dw1 需要沿被广播的 batch 维
求和归约，归约错了）：

    matmul(w1, tmp) 隐式广播        w1.grad cos=0.1656   相对误差 133.9%
    einsum / expand+bmm / 纯 2D mm  w1.grad cos=1.000000  相对误差 ≤0.001%

w1 是 LoKr 的总增益因子（ΔW = s·kron(w1, U)），它的梯度错不会表现为数值噪声，
而是训练动力学变质：总强度失去负反馈 → ‖ΔW‖ 单调膨胀（昇腾上 12 epoch 涨到
CUDA 同配方的 3.2 倍，成品推理时压过文本条件、几乎只输出同一张图）。

这个 bug 在 CUDA 上**不会复现**，所以不能靠"CI 在 CUDA 上跑过了"来防守。
下面的测试与后端无关：它断言几种数学等价写法给出同一个 w1 梯度。在有问题的平台上
（把实现换回隐式广播，或换一个 backward 有同类 bug 的后端）会直接失败。
"""
import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trainer.lora import LoRALinear  # noqa: E402

FACTOR = 4
SEED = 20260816


def _layer(in_f, out_f, rank, dev="cpu"):
    """造一层 LoKr，并把 w2_b 从零挪开。

    LoKr 的 w2_b init 为零（保证 step-0 净 ΔW=0），此时 ∂L/∂w1 ∝ w2_a@w2_b = 0，
    在 init 点上测不到任何东西——必须先 warm-start 到"训练中"的工作点。
    """
    torch.manual_seed(SEED)
    base = torch.nn.Linear(in_f, out_f, bias=False).to(dev)
    torch.manual_seed(SEED)
    layer = LoRALinear(
        base, rank=rank, alpha=float(rank), use_lokr=True, factor=FACTOR,
        rank_dropout=0.0, module_dropout=0.0, lora_variant="base",
        lokr_w1_init_std=0.1, lokr_compute_dtype="fp32",
    ).to(dev)
    g = torch.Generator(device="cpu").manual_seed(SEED + 31)
    wb = layer.adapter.lokr_w2_b
    with torch.no_grad():
        wb.copy_(torch.randn(*wb.shape, generator=g, dtype=torch.float32)
                 .mul_(1e-2).to(wb.device, wb.dtype))
    layer.train()
    return layer


def _w1_grad(layer, x, gy):
    for p in layer.parameters():
        if p.grad is not None:
            p.grad = None
    y = layer(x)
    y.backward(gy)
    return layer.adapter.lokr_w1.grad.detach().clone()


@pytest.mark.parametrize("in_f,out_f,rank", [(256, 256, 16), (256, 512, 8)])
def test_w1_grad_matches_reference_contractions(in_f, out_f, rank):
    """w1 的梯度必须与几种数学等价的显式收缩一致。

    参考值不走 LoKrLayer，而是直接按定义算：
        y[p,i,o] = Σ_j w1[i,j]·tmp[p,j,o]
        dL/dw1[i,j] = Σ_p Σ_o gy[p,i,o]·tmp[p,j,o]
    """
    layer = _layer(in_f, out_f, rank)
    ad = layer.adapter
    g = torch.Generator(device="cpu").manual_seed(SEED + 7)
    x = torch.randn(4, 32, in_f, generator=g)
    gy = torch.randn(4, 32, out_f, generator=g)

    got = _w1_grad(layer, x, gy)

    # 手工重建 tmp（与 _compute 的前两段 matmul 一致），再按定义收缩出 dL/dw1
    with torch.no_grad():
        w2_a = ad.lokr_w2_a.float()
        w2_b = ad.lokr_w2_b.float()
        x_flat = x.reshape(-1, ad.factor, ad.in_dim).float()
        tmp = torch.matmul(torch.matmul(x_flat, w2_b.t()), w2_a.t())  # (P,F,out_dim)
        # 前向出口是 y.reshape(...) * scaling，所以回传到 bmm 输出的梯度带 scaling
        gy_flat = gy.reshape(-1, ad.factor, ad.out_dim).float() * ad.scaling
        want = torch.einsum("pio,pjo->ij", gy_flat, tmp)

    assert torch.allclose(got, want, rtol=2e-4, atol=2e-6), (
        f"w1 梯度与按定义收缩的参考值不符\n"
        f"  cos={torch.dot(got.flatten(), want.flatten()) / (got.norm() * want.norm()):.6f}\n"
        f"  相对误差={(got - want).norm() / want.norm():.3%}\n"
        f"  → 本平台的 backward 可能对被广播操作数的归约有 bug；"
        f"见 tests/diag_npu_broadcast_matmul_grad.py"
    )


def test_implementation_does_not_use_implicit_2d_3d_broadcast():
    """锁住实现本身：w1 的作用不得写成 2D×3D 的隐式广播 matmul。

    纯静态断言。这个 bug 在 CUDA 上不复现，数值测试在 CUDA CI 上永远是绿的，
    所以额外用一条源码断言防止改回去。
    """
    import inspect
    import re

    from trainer.lora import LoKrLayer

    src = inspect.getsource(LoKrLayer._compute)
    # 只看真正的代码：剥掉行注释（那段说明里就写着这个字面量，否则会自己误伤自己）
    code = "\n".join(re.sub(r"#.*$", "", line) for line in src.splitlines())
    assert "torch.matmul(w1, tmp)" not in code, (
        "LoKrLayer._compute 又用回了 `torch.matmul(w1, tmp)`（2D×3D 隐式广播）。"
        "昇腾 torch_npu 2.6 上它的 backward 对 w1 算错（cos 0.166 / 相对误差 134%），"
        "会让 LoKr 的总增益因子失去负反馈、‖ΔW‖ 单调膨胀。"
        "请保持 expand+bmm（或其他把归约交给 autograd 的等价写法）。"
    )


def test_w1_grad_is_nonzero_after_warm_start():
    """防呆：warm-start 没生效的话上面的测试会变成"0 == 0"的假绿。"""
    layer = _layer(256, 256, 16)
    g = torch.Generator(device="cpu").manual_seed(SEED + 7)
    x = torch.randn(4, 32, 256, generator=g)
    gy = torch.randn(4, 32, 256, generator=g)
    assert _w1_grad(layer, x, gy).norm() > 0, "w1 梯度为零 —— warm-start 失效，测试无意义"
