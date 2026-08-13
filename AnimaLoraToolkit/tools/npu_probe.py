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


@probe("torch_npu.npu_fusion_attention（昇腾融合注意力）")
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
        ("diffusers", "必需"),
        ("einops", "必需"),
        ("omegaconf", "配置，必需"),
        ("PIL", "必需"),
        ("pillow_jxl", "JXL 数据集才需要"),
        ("lpips", "perceptual aux loss 才需要"),
        ("scipy", "工具脚本"),
        ("xformers", "昇腾上预期缺失 → NaViT 打包路径不可用"),
        ("bitsandbytes", "昇腾上预期缺失 → 8-bit 优化器不可用"),
        ("triton", "昇腾上预期缺失"),
        ("flash_attn", "昇腾上预期缺失"),
    ]:
        try:
            m = __import__(mod)
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
    print("=" * 78)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(RESULTS, f, ensure_ascii=False, indent=2)
        print(f"已写出 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
