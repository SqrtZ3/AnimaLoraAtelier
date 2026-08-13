#!/usr/bin/env python
"""昇腾 Ascend NPU（910B）能力探针 —— 在真机上实测，不做任何假设。

用途：在 OpenI/启智 NPU 容器里**烧算力跑训练之前**先跑这一个脚本，把"哪些算子/
特性在这台机器上真的能用"变成实测数据。每一项都独立 try/except，单项失败不影响
后续项，最后打一张汇总表。

用法：
    python tools/npu_probe.py               # 全部探测
    python tools/npu_probe.py --json out.json

判读：
    * OK   —— 实测通过（含数值校验的项会打印相对误差）
    * FAIL —— 实测失败，附错误摘要；训练里用到该路径就会崩
    * SKIP —— 前置条件不满足（例如 NPU 不可用）

本脚本只读不写，不加载任何底模权重，几秒到几十秒跑完。
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import subprocess
import sys
import time
import traceback

RESULTS: list[dict] = []


def record(name: str, status: str, detail: str = "") -> None:
    RESULTS.append({"name": name, "status": status, "detail": detail})
    mark = {"OK": "  OK  ", "FAIL": " FAIL ", "SKIP": " SKIP ", "INFO": " INFO "}.get(status, status)
    print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""), flush=True)


def probe(name: str):
    """装饰器：把一个返回 detail 字符串的函数包成一项探测。"""

    def deco(fn):
        def run(*a, **kw):
            try:
                detail = fn(*a, **kw)
                record(name, "OK", detail or "")
                return True
            except _Skip as s:
                record(name, "SKIP", str(s))
                return None
            except Exception as e:
                tb = traceback.format_exc(limit=2).strip().splitlines()[-1]
                record(name, "FAIL", f"{type(e).__name__}: {e} | {tb}")
                return False

        return run

    return deco


class _Skip(Exception):
    pass


# ── 0. 基础环境 ────────────────────────────────────────────────────────────────

def probe_env() -> None:
    record("python", "INFO", f"{platform.python_version()} @ {sys.executable}")
    record("platform", "INFO", platform.platform())
    for var in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_RT_VISIBLE_DEVICES",
                "PYTORCH_NPU_ALLOC_CONF", "LD_LIBRARY_PATH"):
        v = os.environ.get(var)
        if v:
            record(f"env:{var}", "INFO", v[:200])
    try:
        out = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=30)
        head = "\n".join(out.stdout.strip().splitlines()[:12])
        record("npu-smi info", "INFO", "\n" + head if head else "(空输出)")
    except Exception as e:
        record("npu-smi info", "SKIP", f"{type(e).__name__}: {e}")


@probe("import torch")
def probe_torch() -> str:
    import torch
    return f"torch {torch.__version__}"


@probe("import torch_npu")
def probe_torch_npu() -> str:
    import torch_npu
    return f"torch_npu {getattr(torch_npu, '__version__', 'unknown')}"


@probe("torch_npu.contrib.transfer_to_npu")
def probe_transfer() -> str:
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
    import torch
    # 补丁生效后，torch.cuda.is_available() 应当映射到 NPU
    return f"patched; torch.cuda.is_available()={torch.cuda.is_available()}"


@probe("NPU 可见性")
def probe_visible() -> str:
    import torch
    if not torch.npu.is_available():
        raise RuntimeError("torch.npu.is_available() == False")
    n = torch.npu.device_count()
    names = [torch.npu.get_device_name(i) for i in range(n)]
    return f"{n} 卡: {names}"


def _dev():
    import torch
    if not getattr(torch, "npu", None) or not torch.npu.is_available():
        raise _Skip("NPU 不可用")
    return "npu:0"


# ── 1. 数值与精度 ──────────────────────────────────────────────────────────────

@probe("bf16 matmul 数值")
def probe_bf16_matmul() -> str:
    import torch
    d = _dev()
    torch.manual_seed(0)
    a = torch.randn(512, 512)
    b = torch.randn(512, 512)
    ref = (a.double() @ b.double())
    got = (a.to(d, torch.bfloat16) @ b.to(d, torch.bfloat16)).float().cpu().double()
    rel = ((got - ref).norm() / ref.norm()).item()
    if rel > 5e-2:
        raise RuntimeError(f"bf16 matmul 相对误差过大 rel={rel:.3e}")
    return f"rel_err={rel:.3e}"


@probe("fp32 matmul 数值")
def probe_fp32_matmul() -> str:
    import torch
    d = _dev()
    torch.manual_seed(0)
    a = torch.randn(512, 512)
    b = torch.randn(512, 512)
    ref = a.double() @ b.double()
    got = (a.to(d) @ b.to(d)).cpu().double()
    rel = ((got - ref).norm() / ref.norm()).item()
    return f"rel_err={rel:.3e}（>1e-4 说明默认走了降精度模式，训练 fp32 残留路径需留意）"


@probe("autocast('npu', bf16)")
def probe_autocast() -> str:
    import torch
    d = _dev()
    x = torch.randn(64, 256, device=d)
    lin = torch.nn.Linear(256, 256).to(d)
    with torch.autocast("npu", dtype=torch.bfloat16):
        y = lin(x)
    return f"输出 dtype={y.dtype}（期望 torch.bfloat16）"


# ── 2. 注意力（训练热路径） ────────────────────────────────────────────────────

def _sdpa_case(mask_kind: str) -> str:
    import torch
    import torch.nn.functional as F
    d = _dev()
    B, H, S, E = 1, 16, 2048, 128
    torch.manual_seed(0)
    q = torch.randn(B, H, S, E, device=d, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(B, H, S, E, device=d, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(B, H, S, E, device=d, dtype=torch.bfloat16, requires_grad=True)
    if mask_kind == "none":
        mask = None
    elif mask_kind == "bool":
        mask = torch.ones(B, 1, S, S, device=d, dtype=torch.bool)
    elif mask_kind == "additive":
        mask = torch.zeros(B, 1, S, S, device=d, dtype=torch.bfloat16)
    else:
        raise ValueError(mask_kind)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    o.sum().backward()
    torch.npu.synchronize()
    dt = (time.perf_counter() - t0) * 1000
    peak = torch.npu.max_memory_allocated() / 1e9
    return f"fwd+bwd {dt:.1f} ms, peak {peak:.2f} GB, grad_ok={q.grad is not None}"


@probe("SDPA 无 mask (B1 H16 S2048 D128 bf16, fwd+bwd)")
def probe_sdpa_none() -> str:
    return _sdpa_case("none")


@probe("SDPA bool mask（稠密 SxS）")
def probe_sdpa_bool() -> str:
    return _sdpa_case("bool")


@probe("SDPA additive mask（稠密 SxS）")
def probe_sdpa_add() -> str:
    return _sdpa_case("additive")


def _keypad_mask_case(expand: bool):
    """构造带真实 padding 的 key-padding mask，返回 (NPU 输出, fp64 CPU 参考, 极性反转参考)。

    ``expand=True`` 模拟 ``utils/npu_compat.expand_attn_mask`` 展开后的形状
    ``[B,1,Sq,Skv]``；``expand=False`` 是仓库原本的广播形状 ``[B,1,1,Skv]``。
    """
    import torch
    import torch.nn.functional as F
    d = _dev()
    B, H, S, E, VALID = 1, 16, 251, 128, 151      # 与真机报错时的形状一致（Sq=Skv=251）
    torch.manual_seed(0)
    q = torch.randn(B, H, S, E, dtype=torch.bfloat16)
    k = torch.randn(B, H, S, E, dtype=torch.bfloat16)
    v = torch.randn(B, H, S, E, dtype=torch.bfloat16)
    keep = torch.zeros(B, S, dtype=torch.bool)
    keep[:, :VALID] = True                        # 前 VALID 个是真 token，其余是 padding
    mask = keep[:, None, None, :]                 # [B,1,1,Skv]，torch 语义：True=保留
    if expand:
        mask = mask.expand(B, 1, S, S).contiguous()

    out = F.scaled_dot_product_attention(
        q.to(d), k.to(d), v.to(d), attn_mask=mask.to(d)
    ).float().cpu().double()

    # fp64 CPU 参考：手写 softmax，两种极性各算一份
    qd, kd, vd = q.double(), k.double(), v.double()
    scores = (qd @ kd.transpose(-1, -2)) / (E ** 0.5)
    keep4 = keep[:, None, None, :].expand(B, H, S, S)
    ref_ok = (torch.softmax(scores.masked_fill(~keep4, float("-inf")), dim=-1) @ vd)
    ref_inv = (torch.softmax(scores.masked_fill(keep4, float("-inf")), dim=-1) @ vd)
    return out, ref_ok, ref_inv


@probe("SDPA key-padding mask 语义/极性（展开后 [B,1,Sq,Skv]）")
def probe_sdpa_keypad_semantics() -> str:
    """**这条才是 mask 的正确性检查**，上面三条只测了形状能不能被接受和快不快。

    背景：昇腾 ``aclnnFlashAttentionScore`` 的 ``atten_mask`` 约定是 **True=屏蔽**，
    与 PyTorch SDPA 的 **True=保留** 相反。这层转换由 torch_npu 负责，正常应当没问题，
    但如果没做，训练会照跑、loss 照降、不报任何错——只是文本条件变成"只看 padding"。
    全 True / 全 0 的 mask 对极性完全不敏感，测不出来，所以这里用真实 padding 对拍。
    """
    import torch
    out, ref_ok, ref_inv = _keypad_mask_case(expand=True)
    # 全部 query 行都用同一套 keep 集合（无整行被屏蔽的情况），所以不会出现 NaN 行，
    # NPU 输出与参考逐行可比。
    err_ok = ((out - ref_ok).norm() / ref_ok.norm()).item()
    err_inv = ((out - ref_inv).norm() / ref_inv.norm()).item()
    if not torch.isfinite(out).all():
        raise RuntimeError("输出含 NaN/Inf —— 全屏蔽行或 mask 处理有问题")
    if err_ok > 5e-2:
        if err_inv < err_ok:
            raise RuntimeError(
                f"mask 极性疑似反了！rel_err(torch 语义 True=保留)={err_ok:.3e} "
                f"> rel_err(反转语义)={err_inv:.3e}。"
                "若确认，昇腾上必须在下发前对 bool mask 取反，否则文本条件是错的。"
            )
        raise RuntimeError(f"key-padding mask 结果与 fp64 参考不符 rel_err={err_ok:.3e}")
    return (f"rel_err={err_ok:.3e}（反转极性参考 {err_inv:.3e}，差 "
            f"{err_inv / max(err_ok, 1e-12):.0f}×）→ 极性与 torch 语义一致")


@probe("SDPA 广播 key-padding mask [B,1,1,Skv]（记录昇腾是否仍拒收）")
def probe_sdpa_keypad_broadcast() -> str:
    """记录**未展开**的广播 mask 在当前 CANN/torch_npu 上是否还会被拒。

    真机曾报 ``get unsupported atten_mask shape ... [1,1,1,251]``，这是
    ``utils/npu_compat.expand_attn_mask`` 存在的唯一理由。如果某天这条变成 OK 且数值
    正确，那个展开（以及它 O(B·Sq·Skv) 的物化代价）就可以整体去掉。
    """
    try:
        out, ref_ok, ref_inv = _keypad_mask_case(expand=False)
    except Exception as e:  # 预期路径：昇腾拒收该形状
        return f"仍不支持（expand_attn_mask 有必要）：{type(e).__name__}: {str(e)[:200]}"
    err_ok = ((out - ref_ok).norm() / ref_ok.norm()).item()
    err_inv = ((out - ref_inv).norm() / ref_inv.norm()).item()
    ok = "数值正确" if err_ok < 5e-2 else f"但数值不对 rel_err={err_ok:.3e}"
    return (f"本机接受广播 mask，{ok}（反转极性参考 {err_inv:.3e}）"
            "→ 可考虑去掉 npu_compat.expand_attn_mask 的展开")


@probe("torch_npu.npu_fusion_attention（昇腾融合注意力，BSH）")
def probe_fusion_attn() -> str:
    import torch
    import torch_npu
    d = _dev()
    if not hasattr(torch_npu, "npu_fusion_attention"):
        raise _Skip("该 torch_npu 版本无 npu_fusion_attention")
    B, S, H, E = 1, 2048, 16, 128
    q = torch.randn(B, S, H * E, device=d, dtype=torch.bfloat16)
    k = torch.randn(B, S, H * E, device=d, dtype=torch.bfloat16)
    v = torch.randn(B, S, H * E, device=d, dtype=torch.bfloat16)
    out = torch_npu.npu_fusion_attention(q, k, v, H, input_layout="BSH")
    o = out[0] if isinstance(out, (tuple, list)) else out
    return f"输出 shape={tuple(o.shape)}（可作为 SDPA 的昇腾专用替代后端）"


# ── 2b. TND 变长打包注意力：NaViT 在昇腾上能不能活，全看这几项 ────────────────
# 语义目标：等价于 xformers BlockDiagonalMask —— 每张图只看自己的 token，段间零泄漏，
# 且不物化 O(ΣN²) 的稠密 mask。昇腾侧对应 npu_fusion_attention 的 input_layout="TND"
# + actual_seq_qlen / actual_seq_kvlen（累加和形式，host 侧 int 列表）。
# 几何取 Anima base 的真实形状：H=16、head_dim=128（model_channels=2048 / 16 头）。

_ANIMA_H, _ANIMA_D = 16, 128


def _seg_reference(q, k, v, q_lens, kv_lens, scale):
    """逐段 dense SDPA 参考实现（fp32）。段内全注意力 ≡ 块对角 mask。

    q: [Tq, H, D]，k/v: [Tkv, H, D]（TND 布局）。返回 [Tq, H, D] fp32。
    """
    import torch
    import torch.nn.functional as F

    outs = []
    qo = ko = 0
    for sq, sk in zip(q_lens, kv_lens):
        qs = q[qo:qo + sq].float().transpose(0, 1).unsqueeze(0)   # [1,H,sq,D]
        ks = k[ko:ko + sk].float().transpose(0, 1).unsqueeze(0)
        vs = v[ko:ko + sk].float().transpose(0, 1).unsqueeze(0)
        o = F.scaled_dot_product_attention(qs, ks, vs, scale=scale)
        outs.append(o.squeeze(0).transpose(0, 1))                 # [sq,H,D]
        qo += sq
        ko += sk
    return torch.cat(outs, dim=0)


def _tnd_case(q_lens, kv_lens, want_grad: bool) -> str:
    import math
    import torch
    import torch_npu

    d = _dev()
    if not hasattr(torch_npu, "npu_fusion_attention"):
        raise _Skip("该 torch_npu 版本无 npu_fusion_attention")
    H, E = _ANIMA_H, _ANIMA_D
    scale = 1.0 / math.sqrt(E)
    Tq, Tkv = sum(q_lens), sum(kv_lens)
    torch.manual_seed(0)
    q = torch.randn(Tq, H, E, device=d, dtype=torch.bfloat16, requires_grad=want_grad)
    k = torch.randn(Tkv, H, E, device=d, dtype=torch.bfloat16, requires_grad=want_grad)
    v = torch.randn(Tkv, H, E, device=d, dtype=torch.bfloat16, requires_grad=want_grad)

    # actual_seq_* 要的是**累加和**，不是每段长度本身。
    cu_q = list(itertools.accumulate(q_lens))
    cu_kv = list(itertools.accumulate(kv_lens))

    torch.npu.reset_peak_memory_stats()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    out = torch_npu.npu_fusion_attention(
        q, k, v, H,
        input_layout="TND",
        scale=scale,
        actual_seq_qlen=cu_q,
        actual_seq_kvlen=cu_kv,
    )
    o = out[0] if isinstance(out, (tuple, list)) else out
    grad_note = "无（未测反向）"
    if want_grad:
        o.float().sum().backward()
        missing = [n for n, t in (("q", q), ("k", k), ("v", v)) if t.grad is None]
        if missing:
            raise RuntimeError(f"反向未产生梯度：{missing}（该算子在本版本不可微 → 训练不可用）")
        grad_note = (f"‖dq‖={q.grad.float().norm():.3e} ‖dk‖={k.grad.float().norm():.3e} "
                     f"‖dv‖={v.grad.float().norm():.3e}")
    torch.npu.synchronize()
    dt = (time.perf_counter() - t0) * 1000
    peak = torch.npu.max_memory_allocated() / 1e9

    if tuple(o.shape) != (Tq, H, E):
        raise RuntimeError(f"输出 shape={tuple(o.shape)}，期望 {(Tq, H, E)}")

    # 数值对拍：段内全注意力 vs 逐段 dense SDPA 参考
    ref = _seg_reference(q.detach(), k.detach(), v.detach(), q_lens, kv_lens, scale)
    got = o.detach().float()
    rel = ((got - ref).norm() / ref.norm()).item()
    verdict = "数值一致" if rel < 2e-2 else f"⚠ 相对误差偏大 rel={rel:.3e}，别直接上生产"

    # 段间泄漏检查：把第 2 段的 k/v 换掉，第 1 段输出必须逐 bit 不变
    leak = "未测"
    if len(q_lens) > 1:
        k2 = k.detach().clone()
        v2 = v.detach().clone()
        k2[kv_lens[0]:] = torch.randn_like(k2[kv_lens[0]:])
        v2[kv_lens[0]:] = torch.randn_like(v2[kv_lens[0]:])
        out2 = torch_npu.npu_fusion_attention(
            q.detach(), k2, v2, H, input_layout="TND", scale=scale,
            actual_seq_qlen=cu_q, actual_seq_kvlen=cu_kv,
        )
        o2 = out2[0] if isinstance(out2, (tuple, list)) else out2
        same = torch.equal(o.detach()[:q_lens[0]], o2[:q_lens[0]])
        leak = "无泄漏（第1段逐bit不变）" if same else "⚠⚠ 段间有泄漏，语义不等于块对角！"

    return (f"rel_err={rel:.3e}（{verdict}）| {leak} | fwd{'+bwd' if want_grad else ''} "
            f"{dt:.1f} ms, peak {peak:.2f} GB | grad: {grad_note}")


@probe("TND 变长 self-attn（q=kv 段长，含反向）★ NaViT 主路径")
def probe_tnd_self() -> str:
    segs = [1024, 2048, 1024]
    return _tnd_case(segs, segs, want_grad=True) + f" | 段长={segs}"


@probe("TND 变长 cross-attn（q/kv 段长不等，含反向）★ Anima 必需")
def probe_tnd_cross() -> str:
    q_segs = [1024, 2048, 1024]
    kv_segs = [128, 256, 96]
    return _tnd_case(q_segs, kv_segs, want_grad=True) + f" | q={q_segs} kv={kv_segs}"


@probe("TND 显存线性性（budget 32k vs 8k，判断是否物化 O(ΣN²)）")
def probe_tnd_scaling() -> str:
    import math
    import torch
    import torch_npu

    d = _dev()
    if not hasattr(torch_npu, "npu_fusion_attention"):
        raise _Skip("该 torch_npu 版本无 npu_fusion_attention")
    H, E = _ANIMA_H, _ANIMA_D
    scale = 1.0 / math.sqrt(E)
    notes = []
    for total, nseg in ((8192, 4), (32768, 8)):
        seg = total // nseg
        lens = [seg] * nseg
        cu = list(itertools.accumulate(lens))
        torch.npu.empty_cache()
        torch.npu.reset_peak_memory_stats()
        q = torch.randn(total, H, E, device=d, dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(total, H, E, device=d, dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(total, H, E, device=d, dtype=torch.bfloat16, requires_grad=True)
        out = torch_npu.npu_fusion_attention(
            q, k, v, H, input_layout="TND", scale=scale,
            actual_seq_qlen=cu, actual_seq_kvlen=cu,
        )
        o = out[0] if isinstance(out, (tuple, list)) else out
        o.float().sum().backward()
        torch.npu.synchronize()
        notes.append(f"ΣN={total}({nseg}段) peak={torch.npu.max_memory_allocated()/1e9:.2f}GB")
        del q, k, v, out, o
    return " | ".join(notes) + "（peak 应近似线性；若 4× token 带来 ~16× 显存说明落到稠密实现）"


@probe("逐段 dense SDPA 保底路径（不依赖昇腾专有算子）")
def probe_sdpa_seg_fallback() -> str:
    """TND 挂了也要有活路：逐段 dense SDPA，段内全注意力 ≡ 块对角，数学恒等。"""
    import math
    import torch
    import torch.nn.functional as F

    d = _dev()
    H, E = _ANIMA_H, _ANIMA_D
    scale = 1.0 / math.sqrt(E)
    q_lens = [1024, 2048, 1024]
    kv_lens = [128, 256, 96]
    torch.manual_seed(0)
    q = torch.randn(sum(q_lens), H, E, device=d, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(sum(kv_lens), H, E, device=d, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(sum(kv_lens), H, E, device=d, dtype=torch.bfloat16, requires_grad=True)
    torch.npu.reset_peak_memory_stats()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    outs = []
    qo = ko = 0
    for sq, sk in zip(q_lens, kv_lens):
        qs = q[qo:qo + sq].transpose(0, 1).unsqueeze(0)
        ks = k[ko:ko + sk].transpose(0, 1).unsqueeze(0)
        vs = v[ko:ko + sk].transpose(0, 1).unsqueeze(0)
        outs.append(F.scaled_dot_product_attention(qs, ks, vs, scale=scale)
                    .squeeze(0).transpose(0, 1))
        qo += sq
        ko += sk
    o = torch.cat(outs, dim=0)
    o.float().sum().backward()
    torch.npu.synchronize()
    dt = (time.perf_counter() - t0) * 1000
    peak = torch.npu.max_memory_allocated() / 1e9
    return f"fwd+bwd {dt:.1f} ms, peak {peak:.2f} GB（与上面 TND 的耗时/显存直接可比）"


# ── 3. 训练器实际用到的算子 ────────────────────────────────────────────────────

@probe("torch.fft（spectral aux loss 依赖）")
def probe_fft() -> str:
    import torch
    d = _dev()
    x = torch.randn(2, 16, 64, 64, device=d, dtype=torch.float32)
    y = torch.fft.rfft2(x)
    return f"rfft2 -> {tuple(y.shape)} dtype={y.dtype}"


@probe("Conv2d/GroupNorm/SiLU bf16（VAE 路径）")
def probe_vae_ops() -> str:
    import torch
    d = _dev()
    m = torch.nn.Sequential(
        torch.nn.Conv2d(16, 32, 3, padding=1),
        torch.nn.GroupNorm(8, 32),
        torch.nn.SiLU(),
    ).to(d)
    x = torch.randn(1, 16, 128, 128, device=d)
    with torch.autocast("npu", dtype=torch.bfloat16):
        y = m(x)
    y.float().sum().backward()
    return f"输出 {tuple(y.shape)} dtype={y.dtype}"


@probe("torch.utils.checkpoint 梯度检查点")
def probe_grad_ckpt() -> str:
    import torch
    from torch.utils.checkpoint import checkpoint
    d = _dev()
    lin = torch.nn.Linear(512, 512).to(d)
    x = torch.randn(8, 512, device=d, requires_grad=True)
    y = checkpoint(lambda t: lin(t).relu(), x, use_reentrant=False)
    y.sum().backward()
    return f"grad_norm={x.grad.norm().item():.4f}"


@probe("AdamW 单步（fp32 master + bf16 参数）")
def probe_adamw() -> str:
    import torch
    d = _dev()
    p = torch.nn.Parameter(torch.randn(1024, 1024, device=d))
    opt = torch.optim.AdamW([p], lr=1e-4)
    before = p.detach().clone()
    (p * p).sum().backward()
    opt.step()
    delta = (p.detach() - before).norm().item()
    return f"‖Δp‖={delta:.4e}（非 0 即优化器 state 正常落到 NPU）"


@probe("torch.npu.Event 计时（stage_timer 依赖）")
def probe_event() -> str:
    import torch
    d = _dev()
    s = torch.npu.Event(enable_timing=True)
    e = torch.npu.Event(enable_timing=True)
    x = torch.randn(2048, 2048, device=d)
    s.record()
    (x @ x).sum()
    e.record()
    torch.npu.synchronize()
    return f"elapsed={s.elapsed_time(e):.3f} ms"


@probe("显存 API（memory_allocated / max / empty_cache）")
def probe_mem() -> str:
    import torch
    d = _dev()
    x = torch.empty(1024, 1024, 64, device=d, dtype=torch.bfloat16)  # ~128MB
    alloc = torch.npu.memory_allocated() / 1e9
    peak = torch.npu.max_memory_allocated() / 1e9
    del x
    torch.npu.empty_cache()
    after = torch.npu.memory_allocated() / 1e9
    return f"alloc={alloc:.3f}GB peak={peak:.3f}GB after_free={after:.3f}GB"


@probe("bf16 线性层 fwd+bwd 吞吐（粗略算力标定）")
def probe_throughput() -> str:
    import torch
    d = _dev()
    m = torch.nn.Linear(4096, 4096, bias=False).to(d, torch.bfloat16)
    x = torch.randn(8192, 4096, device=d, dtype=torch.bfloat16, requires_grad=True)
    for _ in range(3):
        m(x).sum().backward()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    iters = 10
    for _ in range(iters):
        m(x).sum().backward()
    torch.npu.synchronize()
    dt = (time.perf_counter() - t0) / iters
    # fwd 2*M*N*K + bwd 约 2×fwd
    flops = 3 * 2 * 8192 * 4096 * 4096
    return f"{dt*1000:.2f} ms/iter ≈ {flops/dt/1e12:.1f} TFLOPS(bf16, 含 bwd 粗估)"


# ── 4. 依赖包可用性（预期部分缺失） ───────────────────────────────────────────

def probe_packages() -> None:
    for mod, note in [
        ("safetensors", "权重读写，必需"),
        ("transformers", "文本编码器，必需"),
        ("einops", "必需"),
        ("PIL", "必需"),
        ("PyYAML/yaml", "配置解析，必需"),
        ("rich", "进度显示，必需"),
        # 以下都是**可选**。曾把 diffusers/omegaconf 标成"必需"是错的：AST 扫描
        # anima_train.py + trainer/ + utils/ + models/ 的真实运行路径，两者都未被 import
        # （tools/npu_setup_image.sh 的不装清单同源）。缺失不影响训练。
        ("diffusers", "可选：运行路径未 import，不装"),
        ("omegaconf", "可选：运行路径未 import，不装"),
        ("pillow_jxl", "可选：JXL 数据集才需要"),
        ("lpips", "可选：perceptual aux loss 才需要"),
        ("scipy", "可选：工具脚本"),
        ("xformers", "昇腾预期缺失 → 用 navit_attn_backend: npu_tnd / sdpa_seg 代替，"
                     "NaViT 打包仍可用"),
        ("bitsandbytes", "昇腾上预期缺失 → 8-bit 优化器不可用"),
        ("triton", "昇腾上预期缺失"),
        ("flash_attn", "昇腾上预期缺失"),
    ]:
        try:
            m = __import__(mod.split("/")[-1])
            record(f"pkg:{mod}", "OK", f"{getattr(m, '__version__', '?')} — {note}")
        except Exception as e:
            record(f"pkg:{mod}", "FAIL", f"{type(e).__name__} — {note}")


def main() -> int:
    ap = argparse.ArgumentParser(description="昇腾 NPU 能力探针")
    ap.add_argument("--json", default="", help="把结果写成 JSON 便于回传")
    args = ap.parse_args()

    print("=" * 78)
    print("昇腾 Ascend NPU 能力探针 —— 全部为真机实测，未通过项即为该机器的真实限制")
    print("=" * 78)

    probe_env()
    probe_torch()
    has_npu = probe_torch_npu()
    if has_npu:
        probe_transfer()
        probe_visible()

    probe_bf16_matmul()
    probe_fp32_matmul()
    probe_autocast()
    probe_sdpa_none()
    probe_sdpa_bool()
    probe_sdpa_add()
    probe_fusion_attn()
    probe_tnd_self()
    probe_tnd_cross()
    probe_tnd_scaling()
    probe_sdpa_seg_fallback()
    probe_fft()
    probe_vae_ops()
    probe_grad_ckpt()
    probe_adamw()
    probe_event()
    probe_mem()
    probe_throughput()
    probe_packages()

    ok = sum(1 for r in RESULTS if r["status"] == "OK")
    fail = sum(1 for r in RESULTS if r["status"] == "FAIL")
    skip = sum(1 for r in RESULTS if r["status"] == "SKIP")
    print("=" * 78)
    print(f"汇总：OK={ok}  FAIL={fail}  SKIP={skip}")
    if fail:
        print("\n失败项（这些是这台机器的真实限制，配置里必须绕开）：")
        for r in RESULTS:
            if r["status"] == "FAIL":
                print(f"  - {r['name']}: {r['detail']}")

    # ── NaViT 路径裁决：把探针结果直接翻译成该往 yaml 里写什么 ──────────────────
    def _st(prefix: str) -> str:
        for r in RESULTS:
            if r["name"].startswith(prefix):
                return r["status"]
        return "MISSING"

    tnd_self = _st("TND 变长 self-attn")
    tnd_cross = _st("TND 变长 cross-attn")
    seg_ok = _st("逐段 dense SDPA 保底路径")
    print("\nNaViT 打包路径裁决：")
    if tnd_self == "OK" and tnd_cross == "OK":
        print("  ✓ TND 变长融合注意力可用（含反向、q/kv 不等长）")
        print("    → navit_packing: true + navit_attn_backend: npu_tnd")
    elif seg_ok == "OK":
        print(f"  ✗ TND 不可用（self={tnd_self} cross={tnd_cross}），但逐段 dense SDPA 通过")
        print("    → navit_packing: true + navit_attn_backend: sdpa_seg（慢一些，语义等价）")
    else:
        print(f"  ✗ TND({tnd_self}/{tnd_cross}) 与逐段 SDPA({seg_ok}) 都不可用")
        print("    → navit_packing: false，退回 ARB 稠密路径")
    print("=" * 78)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(RESULTS, f, ensure_ascii=False, indent=2)
        print(f"已写出 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
