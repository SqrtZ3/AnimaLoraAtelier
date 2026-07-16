# -*- coding: utf-8 -*-
"""冻结底模 FP8/FP4 量化（trainer/quant.py）单测。

覆盖：
  · fp8 rowwise/tensorwise 与 nvfp4 量化-反量化误差界；
  · e2m1 bucketize 舍入 == 逐点 argmin 参考实现；fp4 打包/解包往返一致；
  · QuantLinear dequant 路径前向/反向与"物化 dequant 权重的 F.linear"逐元素等价
    （backward 重新 dequant 的实现不改数学）；
  · LoRALinear 集成：换装 original 后 前向 = quant_base(x)+adapter(x)，
    adapter 梯度照常回传，merged_weight() 经 weight property 透明工作；
  · quantize_base_model：include/skip 选择、LoRALinear 被 skip 时其 .original
    不得以裸 Linear 身份绕过规则、adapter 永不量化、stats 口径；
  · validate_base_quant_compat fail-fast 组合；
  · [GPU] fp8/fp4 量化 GEMM 与逐元素 dequant 仿真对拍、fp8_grad 反向、
    形状不合规自动回退。

CPU 部分可在无 GPU 环境跑；GPU 部分自动 skip。
"""
import types

import pytest
import torch

from trainer.quant import (
    QuantLinear,
    _dequant_fp8,
    _dequant_nvfp4,
    _e2m1_codes,
    _gemm_probe,
    _pack_fp4,
    _quant_fp8_rowwise,
    _quant_fp8_tensorwise,
    _quant_nvfp4,
    _to_blocked,
    quantize_base_model,
    validate_base_quant_compat,
    _E2M1_VALUES,
)

_HAS_CUDA = torch.cuda.is_available()


def _args(**kw):
    base = dict(base_quant="none", base_quant_gemm="auto",
                base_quant_fp8_scale="auto", base_quant_fp8_grad=False,
                base_quant_include=None, base_quant_skip=None,
                mixed_precision="bf16", lora_variant="base",
                lora_one_init_steps=0, torch_compile=False)
    base.update(kw)
    return types.SimpleNamespace(**base)


# ── 量化数学 ─────────────────────────────────────────────────────────────

def test_fp8_dequant_relerr():
    torch.manual_seed(0)
    w = torch.randn(256, 320) * 0.02
    q, s = _quant_fp8_rowwise(w)
    rel = (_dequant_fp8(q, s, torch.float32) - w).norm() / w.norm()
    assert rel < 0.04, rel
    q, s = _quant_fp8_tensorwise(w)
    rel = (_dequant_fp8(q, s, torch.float32) - w).norm() / w.norm()
    assert rel < 0.06, rel


def test_fp8_rowwise_scale_handles_dead_rows():
    w = torch.zeros(8, 32)
    w[0] = torch.randn(32)
    q, s = _quant_fp8_rowwise(w)
    deq = _dequant_fp8(q, s, torch.float32)
    assert torch.isfinite(deq).all()
    assert (deq[1:] == 0).all()


def test_e2m1_codes_match_argmin_reference():
    torch.manual_seed(1)
    x = torch.randn(4096) * 3
    x = torch.cat([x, torch.tensor([0.0, 6.0, -6.0, 0.24, 0.26, 5.01, 4.99])])
    codes = _e2m1_codes(x)
    grid = torch.tensor(_E2M1_VALUES)
    idx_ref = (x.abs().clamp(max=6.0).unsqueeze(-1) - grid).abs().argmin(dim=-1)
    val = grid[(codes & 0x7).long()] * torch.where((codes & 0x8).bool(), -1.0, 1.0)
    val_ref = grid[idx_ref] * torch.where(x < 0, -1.0, 1.0)
    # tie 处两侧网格值等距 → 比较各自的重建误差而非码本身
    err = (val - x.clamp(-6, 6)).abs()
    err_ref = (val_ref - x.clamp(-6, 6)).abs()
    assert torch.all(err <= err_ref + 1e-6), "bucketize 舍入必须不劣于 argmin 最近邻"


def test_fp4_pack_roundtrip_and_relerr():
    torch.manual_seed(2)
    w = torch.randn(128, 64) * 0.05
    packed, bs, ts = _quant_nvfp4(w)
    assert packed.shape == (128, 32) and packed.dtype == torch.uint8
    assert bs.shape == (128, 4)
    deq = _dequant_nvfp4(packed, bs, ts, torch.float32)
    rel = (deq - w).norm() / w.norm()
    assert rel < 0.12, rel
    # 再量化-再解包 应当收敛（幂等性：网格上的值不再移动）
    packed2, bs2, ts2 = _quant_nvfp4(deq)
    deq2 = _dequant_nvfp4(packed2, bs2, ts2, torch.float32)
    assert (deq2 - deq).norm() / deq.norm() < 1e-3


def test_fp4_dead_block_no_nan():
    w = torch.zeros(4, 32)
    w[0, :16] = torch.randn(16)
    packed, bs, ts = _quant_nvfp4(w)
    deq = _dequant_nvfp4(packed, bs, ts, torch.float32)
    assert torch.isfinite(deq).all()
    assert (deq[1:] == 0).all() and (deq[0, 16:] == 0).all()


def test_to_blocked_shape():
    s = torch.randn(77, 5).to(torch.float8_e4m3fn)
    out = _to_blocked(s)
    assert out.shape == (128 * 8,)  # pad 到 128 行 × 8 列（2 个 4 列块），flatten


# ── QuantLinear dequant 路径（数学等价性）──────────────────────────────

@pytest.mark.parametrize("fmt", ["fp8", "fp4"])
def test_quantlinear_dequant_forward_backward_equiv(fmt):
    torch.manual_seed(3)
    lin = torch.nn.Linear(64, 48, bias=True)
    q = QuantLinear.from_linear(lin, fmt, "dequant", compute_dtype=torch.float32)
    w_deq = q._dequant_weight(torch.float32)

    x = torch.randn(5, 7, 64, requires_grad=True)
    y = q(x)
    x_ref = x.detach().clone().requires_grad_(True)
    y_ref = torch.nn.functional.linear(x_ref, w_deq, lin.bias)
    assert torch.allclose(y, y_ref, atol=1e-6), (y - y_ref).abs().max()

    g = torch.randn_like(y)
    y.backward(g)
    y_ref.backward(g)
    assert torch.allclose(x.grad, x_ref.grad, atol=1e-6)
    # 权重侧不产生梯度（冻结）
    assert q.weight_q.grad is None if hasattr(q.weight_q, "grad") else True


def test_quantlinear_weight_property():
    lin = torch.nn.Linear(32, 16, bias=False)
    q = QuantLinear.from_linear(lin, "fp8", "dequant", compute_dtype=torch.float32)
    w = q.weight
    assert w.shape == (16, 32) and not w.requires_grad
    rel = (w - lin.weight).norm() / lin.weight.norm()
    assert rel < 0.04
    assert q.in_features == 32 and q.out_features == 16 and q.bias is None


# ── LoRALinear 集成 ──────────────────────────────────────────────────────

def _toy_lora_model():
    from trainer.lora import LoRAInjector

    class Blocks(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn_q = torch.nn.Linear(64, 64, bias=False)
            self.mlp_up = torch.nn.Linear(64, 128, bias=False)

        def forward(self, x):
            return self.mlp_up(self.attn_q(x))

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([Blocks()])
            self.first = torch.nn.Linear(64, 64, bias=False)

        def forward(self, x):
            return self.blocks[0](self.first(x))

    m = Toy()
    m.requires_grad_(False)  # 对齐真实流程：底模加载后整体冻结，再注入 LoRA
    inj = LoRAInjector(rank=4, alpha=4.0, targets=["attn_q"])
    inj.inject(m)
    return m, inj


def test_lora_integration_forward_and_adapter_grad():
    torch.manual_seed(4)
    m, inj = _toy_lora_model()
    lora = m.blocks[0].attn_q
    # 给 adapter 一个非零增量，检验前向合成
    with torch.no_grad():
        lora.adapter.lora_up.weight.add_(0.01 * torch.randn_like(lora.adapter.lora_up.weight))

    x = torch.randn(3, 64)
    y_before = m(x)

    args = _args(base_quant="fp8", base_quant_gemm="off")
    stats = quantize_base_model(m, args, family="anima")
    assert stats["count"] == 2  # attn_q.original + mlp_up（裸 Linear）
    assert isinstance(lora.original, QuantLinear)
    assert isinstance(m.blocks[0].mlp_up, QuantLinear)
    assert isinstance(m.first, torch.nn.Linear)  # 不在 blocks.* → 不量化

    # 前向 = quant_base(x) + adapter(x)（compute_dtype=bf16 带来的量化+精度差异有界）
    y_after = m(x)
    rel = (y_after - y_before).norm() / y_before.norm()
    assert rel < 0.08, rel

    # adapter 梯度照常回传，QuantLinear 无可训练参数
    m(x).sum().backward()
    assert lora.adapter.lora_down.weight.grad is not None
    assert lora.adapter.lora_up.weight.grad is not None
    assert all(not p.requires_grad for p in lora.original.parameters())

    # merged_weight() 经 weight property 透明工作
    mw = lora.merged_weight()
    assert mw.shape == (64, 64) and torch.isfinite(mw).all()


def test_quantize_skip_rules_and_original_bypass():
    m, inj = _toy_lora_model()
    args = _args(base_quant="fp8", base_quant_gemm="off",
                 base_quant_skip=[r"blocks\.0\.attn_q"])
    stats = quantize_base_model(m, args, family="anima")
    lora = m.blocks[0].attn_q
    # attn_q 被 skip：original 必须保持 nn.Linear，不得以
    # "blocks.0.attn_q.original" 的裸 Linear 名字绕过 skip 被量化
    assert isinstance(lora.original, torch.nn.Linear)
    assert not isinstance(lora.original, QuantLinear)
    assert isinstance(m.blocks[0].mlp_up, QuantLinear)
    assert stats["count"] == 1
    # adapter 内部 Linear 永不量化
    assert isinstance(lora.adapter.lora_down, torch.nn.Linear)
    assert not isinstance(lora.adapter.lora_down, QuantLinear)


def test_quantize_include_override():
    m, _ = _toy_lora_model()
    args = _args(base_quant="fp4", base_quant_gemm="off",
                 base_quant_include=[r"first"])
    stats = quantize_base_model(m, args, family="anima")
    assert stats["count"] == 1
    assert isinstance(m.first, QuantLinear) and m.first.fmt == "fp4"
    assert isinstance(m.blocks[0].mlp_up, torch.nn.Linear)


# ── fail-fast 组合校验 ───────────────────────────────────────────────────

def test_validate_off_is_noop():
    validate_base_quant_compat(_args())  # base_quant=none → 不校验其余


@pytest.mark.parametrize("kw,frag", [
    (dict(base_quant="int8"), "base_quant"),
    (dict(base_quant="fp8", mixed_precision="fp32"), "bf16"),
    (dict(base_quant="fp8", lora_variant="dora"), "dora"),
    (dict(base_quant="fp8", lora_one_init_steps=4), "lora_one"),
    (dict(base_quant="fp8", torch_compile=True), "torch_compile"),
    (dict(base_quant="fp8", base_quant_gemm="maybe"), "base_quant_gemm"),
    (dict(base_quant="fp8", base_quant_fp8_scale="colwise"), "fp8_scale"),
    (dict(base_quant="fp8", base_quant_gemm="off", base_quant_fp8_grad=True),
     "fp8_grad"),
])
def test_validate_failfast(kw, frag):
    with pytest.raises((ValueError, RuntimeError)) as ei:
        validate_base_quant_compat(_args(**kw))
    assert frag in str(ei.value)


def test_fp4_with_fp8_grad_rejected_at_quantize():
    m, _ = _toy_lora_model()
    args = _args(base_quant="fp4", base_quant_gemm="off", base_quant_fp8_grad=True)
    with pytest.raises(ValueError):
        quantize_base_model(m, args, family="anima")


# ── GPU：量化 GEMM 对拍 ─────────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_CUDA, reason="需要 CUDA")
def test_fp8_gemm_forward_matches_emulation():
    if not _gemm_probe("fp8_tensorwise"):
        pytest.skip("设备不支持 fp8 _scaled_mm")
    torch.manual_seed(5)
    dev = "cuda"
    lin = torch.nn.Linear(256, 128, bias=True).to(dev, torch.bfloat16)
    q = QuantLinear.from_linear(lin, "fp8", "fp8_gemm",
                                fp8_rowwise=_gemm_probe("fp8_rowwise"))
    x = torch.randn(64, 256, device=dev, dtype=torch.bfloat16, requires_grad=True)
    y = q(x)
    # 仿真：dequant 权重 + 量化后的激活 dequant，bf16 GEMM
    w_deq = q._dequant_weight(torch.float32)
    if q.fp8_rowwise:
        xq, xs = _quant_fp8_rowwise(x.detach())
    else:
        xq, xs = _quant_fp8_tensorwise(x.detach())
    x_deq = _dequant_fp8(xq, xs, torch.float32)
    y_emu = x_deq @ w_deq.t() + lin.bias.float()
    rel = (y.float() - y_emu).norm() / y_emu.norm()
    assert rel < 0.02, rel  # 仅剩 GEMM 累加精度差，应远小于量化误差本身

    # 反向（默认 bf16 dequant 路径）
    g = torch.randn_like(y)
    y.backward(g)
    gx_ref = g.float() @ w_deq
    rel = (x.grad.float() - gx_ref).norm() / gx_ref.norm()
    assert rel < 0.02, rel


@pytest.mark.skipif(not _HAS_CUDA, reason="需要 CUDA")
def test_fp8_grad_gemm_close_to_bf16_grad():
    if not _gemm_probe("fp8_tensorwise"):
        pytest.skip("设备不支持 fp8 _scaled_mm")
    torch.manual_seed(6)
    dev = "cuda"
    lin = torch.nn.Linear(256, 128, bias=False).to(dev, torch.bfloat16)
    rowwise = _gemm_probe("fp8_rowwise")
    q_bf = QuantLinear.from_linear(lin, "fp8", "fp8_gemm", fp8_rowwise=rowwise)
    q_f8 = QuantLinear.from_linear(lin, "fp8", "fp8_gemm", fp8_rowwise=rowwise,
                                   fp8_grad=True)
    x1 = torch.randn(64, 256, device=dev, dtype=torch.bfloat16, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    g = torch.randn(64, 128, device=dev, dtype=torch.bfloat16)
    q_bf(x1).backward(g)
    q_f8(x2).backward(g)
    rel = (x1.grad.float() - x2.grad.float()).norm() / x1.grad.float().norm()
    assert rel < 0.1, rel  # e5m2 梯度量化噪声有界


@pytest.mark.skipif(not _HAS_CUDA, reason="需要 CUDA")
def test_fp4_gemm_forward_matches_emulation():
    if not _gemm_probe("fp4"):
        pytest.skip("设备不支持 fp4 _scaled_mm（非 Blackwell 或 torch 过旧）")
    torch.manual_seed(7)
    dev = "cuda"
    lin = torch.nn.Linear(256, 128, bias=False).to(dev, torch.bfloat16)
    q = QuantLinear.from_linear(lin, "fp4", "fp4_gemm")
    x = torch.randn(64, 256, device=dev, dtype=torch.bfloat16, requires_grad=True)
    y = q(x)
    xp, xbs, xts = _quant_nvfp4(x.detach())
    x_deq = _dequant_nvfp4(xp, xbs, xts, torch.float32)
    w_deq = q._dequant_weight(torch.float32)
    y_emu = x_deq @ w_deq.t()
    rel = (y.float() - y_emu).norm() / y_emu.norm()
    assert rel < 0.02, rel

    g = torch.randn_like(y)
    y.backward(g)
    gx_ref = g.float() @ w_deq
    rel = (x.grad.float() - gx_ref).norm() / gx_ref.norm()
    assert rel < 0.02, rel


@pytest.mark.skipif(not _HAS_CUDA, reason="需要 CUDA")
def test_shape_fallback():
    """fp8：维度非 16 倍数 → 自动 dequant 模式；fp4：K 非 16 倍数 → 保持 bf16。"""
    from trainer.lora import LoRAInjector

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList(
                [torch.nn.Sequential(torch.nn.Linear(72, 40, bias=False))])

    dev = "cuda"
    m = Toy().to(dev, torch.bfloat16).requires_grad_(False)
    args = _args(base_quant="fp8", base_quant_gemm="auto")
    stats = quantize_base_model(m, args, family="anima")
    ql = m.blocks[0][0]
    assert isinstance(ql, QuantLinear)
    assert ql.mode == "dequant"  # 72%16!=0 → GEMM 不合规自动回退

    m2 = Toy().to(dev, torch.bfloat16)
    m2.blocks[0][0] = torch.nn.Linear(72 + 8, 40, bias=False).to(dev, torch.bfloat16)
    m2.requires_grad_(False)
    args = _args(base_quant="fp4", base_quant_gemm="off")
    # 80 % 16 == 0 → fp4 可存储
    stats = quantize_base_model(m2, args, family="anima")
    assert stats["count"] == 1

    m3 = Toy().to(dev, torch.bfloat16)
    m3.blocks[0][0] = torch.nn.Linear(72, 40, bias=False).to(dev, torch.bfloat16)
    m3.requires_grad_(False)
    args = _args(base_quant="fp4", base_quant_gemm="off")
    stats = quantize_base_model(m3, args, family="anima")
    assert stats["count"] == 0 and stats["kept_bf16"] == 1
    assert isinstance(m3.blocks[0][0], torch.nn.Linear)


@pytest.mark.skipif(not _HAS_CUDA, reason="需要 CUDA")
def test_grad_checkpoint_recompute_deterministic():
    """梯度检查点会重算前向：量化前向必须确定（两次同输入同输出）。"""
    dev = "cuda"
    lin = torch.nn.Linear(64, 64, bias=False).to(dev, torch.bfloat16)
    mode = "fp8_gemm" if _gemm_probe("fp8_tensorwise") else "dequant"
    q = QuantLinear.from_linear(lin, "fp8", mode,
                                fp8_rowwise=_gemm_probe("fp8_rowwise"))
    x = torch.randn(8, 64, device=dev, dtype=torch.bfloat16)
    y1, y2 = q(x), q(x)
    assert torch.equal(y1, y2)


# ── GPU：真实 Krea2 结构端到端集成（小配置随机权重）────────────────────

@pytest.mark.skipif(not _HAS_CUDA, reason="需要 CUDA")
def test_krea2_end_to_end_quantized_forward_backward():
    """SingleStreamDiT 小配置：注入 LoRA → 量化 → forward_dense 误差有界，
    adapter 梯度回传正常（覆盖真实模型代码里的全部 Linear 调用位点）。"""
    from pathlib import Path
    from trainer.lora import LoRAInjector
    from trainer.model_family import KREA2_DEFAULT_LORA_TARGETS
    from trainer.models import load_module_from_path

    repo_models = Path(__file__).resolve().parent.parent / "models"
    k2 = load_module_from_path("krea2_modeling_bq_test",
                               repo_models / "krea2_modeling.py")
    cfg = k2.SingleMMDiTConfig(
        features=128, tdim=64, txtdim=64, heads=4, kvheads=2,
        multiplier=4, layers=2, patch=2, channels=16,
        txtlayers=3, txtheads=4, txtkvheads=4)
    dev = "cuda"
    torch.manual_seed(8)
    model = k2.SingleStreamDiT(cfg).to(dev, torch.bfloat16)
    model.requires_grad_(False)

    inj = LoRAInjector(rank=4, alpha=4.0, targets=list(KREA2_DEFAULT_LORA_TARGETS))
    inj.inject(model)
    # 给 adapter 非零增量，让量化前后的对比覆盖 base+adapter 合成路径
    with torch.no_grad():
        for lora in inj.injected.values():
            lora.adapter.lora_up.weight.add_(
                0.02 * torch.randn_like(lora.adapter.lora_up.weight))

    x = torch.randn(1, 16, 1, 8, 8, device=dev, dtype=torch.bfloat16)
    t = torch.rand(1, 1, device=dev, dtype=torch.bfloat16)
    cross = torch.randn(1, 7, 3, 64, device=dev, dtype=torch.bfloat16)

    with torch.no_grad():
        y_ref = model(x, t, cross).float()

    args = _args(base_quant="fp8", base_quant_gemm="auto")
    stats = quantize_base_model(model, args, family="krea2")
    # krea2 include 默认：blocks/txtfusion/txtmlp 命中；first/last/tproj/tmlp 保持
    # bf16（它们是 LoRA target 会被 LoRALinear 包住，看 .original 的类型）
    assert stats["count"] > 0

    def _base_of(mod):
        return mod.original if hasattr(mod, "original") else mod

    for sensitive in (model.first, model.last.linear, model.tproj[1], model.tmlp[0]):
        assert not isinstance(_base_of(sensitive), QuantLinear)
    # blocks 内的底模确实被量化了
    assert isinstance(_base_of(model.blocks[0].attn.wq), QuantLinear)
    assert isinstance(_base_of(model.blocks[1].mlp.down), QuantLinear)

    with torch.no_grad():
        y_q = model(x, t, cross).float()
    rel = (y_q - y_ref).norm() / y_ref.norm()
    assert torch.isfinite(y_q).all()
    assert rel < 0.10, rel  # 2 层小模型的 fp8 累积误差应在个位数百分比

    # 反向：adapter 梯度存在且有限
    y = model(x, t, cross)
    y.float().pow(2).mean().backward()
    grads = [l.adapter.lora_down.weight.grad for l in inj.injected.values()]
    n_with_grad = sum(g is not None and torch.isfinite(g).all() for g in grads)
    assert n_with_grad == len(grads), f"{n_with_grad}/{len(grads)} adapter 有梯度"


@pytest.mark.skipif(not _HAS_CUDA, reason="需要 CUDA")
def test_krea2_end_to_end_fp4_dequant_mode():
    """fp4 + gemm=off（H20 部署形态）：同一模型 dequant 路径前向/反向可用。"""
    from pathlib import Path
    from trainer.lora import LoRAInjector
    from trainer.model_family import KREA2_DEFAULT_LORA_TARGETS
    from trainer.models import load_module_from_path

    repo_models = Path(__file__).resolve().parent.parent / "models"
    k2 = load_module_from_path("krea2_modeling_bq_test2",
                               repo_models / "krea2_modeling.py")
    cfg = k2.SingleMMDiTConfig(
        features=128, tdim=64, txtdim=64, heads=4, kvheads=2,
        multiplier=4, layers=2, patch=2, channels=16,
        txtlayers=3, txtheads=4, txtkvheads=4)
    dev = "cuda"
    torch.manual_seed(9)
    model = k2.SingleStreamDiT(cfg).to(dev, torch.bfloat16)
    model.requires_grad_(False)
    inj = LoRAInjector(rank=4, alpha=4.0, targets=list(KREA2_DEFAULT_LORA_TARGETS))
    inj.inject(model)

    x = torch.randn(1, 16, 1, 8, 8, device=dev, dtype=torch.bfloat16)
    t = torch.rand(1, 1, device=dev, dtype=torch.bfloat16)
    cross = torch.randn(1, 7, 3, 64, device=dev, dtype=torch.bfloat16)
    with torch.no_grad():
        y_ref = model(x, t, cross).float()

    args = _args(base_quant="fp4", base_quant_gemm="off")
    stats = quantize_base_model(model, args, family="krea2")
    assert stats["count"] > 0 and stats["gemm"] == 0

    with torch.no_grad():
        y_q = model(x, t, cross).float()
    rel = (y_q - y_ref).norm() / y_ref.norm()
    assert torch.isfinite(y_q).all()
    assert rel < 0.25, rel  # fp4 权重误差 ~0.095，网络级放大后仍应有界

    y = model(x, t, cross)
    y.float().pow(2).mean().backward()
    grads = [l.adapter.lora_down.weight.grad for l in inj.injected.values()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
