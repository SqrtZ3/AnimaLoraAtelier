#!/usr/bin/env python
"""DCU 上"除 math SDPA 之外还能用什么注意力后端"的一次性探针。

背景：海光 DCU（gfx936 / DTK26.04 / torch 2.9.0+das.opt1）实测 SDPA 只剩 MATH 后端
——mem-efficient 与 cuDNN attention 在**编译期**就没进 torch，flash 的 .so 缺失。
math 会物化 S×S 的 softmax，NaViT 打包路径单段 16384 token 时 backward 直接 OOM。

本脚本按"编译可行性由低到高的风险"依次探测四条路，每条独立、互不阻塞，
任何一条崩了都继续往下跑，最后给汇总表。**只读探测，不改环境。**

    python tools/dcu_attn_backend_probe.py

关注三列：可用 / 数值是否与 math 一致 / 峰值显存与耗时。
"""

from __future__ import annotations

import argparse
import time
import traceback

import torch
import torch.nn.functional as F

RESULTS: list[tuple[str, str, str]] = []


def _record(name: str, status: str, detail: str = "") -> None:
    RESULTS.append((name, status, detail))
    print(f"  → {name}: {status} {detail}")


def _bench(fn, q, k, v, warmup=2, iters=5):
    """返回 (输出, 峰值显存GiB, 每次耗时ms)。计时必须 warmup（跨平台通用教训）。"""
    for _ in range(warmup):
        fn(q, k, v)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(q, k, v)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters * 1e3
    peak = (torch.cuda.max_memory_allocated() - base) / 2 ** 30
    return out, peak, dt


def probe_env():
    print("=" * 78)
    print("[0] 环境")
    print(f"  torch {torch.__version__}  hip {torch.version.hip}")
    if not torch.cuda.is_available():
        raise SystemExit("  没有可用加速卡，退出")
    print(f"  device: {torch.cuda.get_device_properties(0).gcnArchName}")
    try:
        import triton
        print(f"  triton {triton.__version__}  ({triton.__file__})")
        return True
    except Exception as e:
        print(f"  triton 不可用: {type(e).__name__}: {e}")
        return False


def probe_triton_hello():
    """Triton 能不能在 gfx936 上编出并跑对一个最简单的 kernel。

    这是整份探针里最关键的一格：flash-attn 的 Triton 后端、torch FlexAttention、
    以及任何自己写的 kernel，全都建立在这一格是绿的基础上。
    """
    print("=" * 78)
    print("[1] Triton 基础可用性（决定下面 2/3 两条路是否值得走）")
    try:
        import triton
        import triton.language as tl

        @triton.jit
        def _add(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            m = offs < n
            tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=m) + tl.load(y_ptr + offs, mask=m), mask=m)

        n = 4096
        x = torch.randn(n, device="cuda")
        y = torch.randn(n, device="cuda")
        o = torch.empty_like(x)
        _add[(triton.cdiv(n, 256),)](x, y, o, n, BLOCK=256)
        torch.cuda.synchronize()
        ok = torch.allclose(o, x + y)
        _record("triton 最简 kernel", "OK" if ok else "跑通但结果错", "" if ok else "!!! 数值不对")
        return ok
    except Exception as e:
        _record("triton 最简 kernel", "FAIL", f"{type(e).__name__}: {str(e)[:160]}")
        traceback.print_exc()
        return False


def probe_flash_attn_triton(q, k, v, ref):
    """flash-attn 的 Triton AMD 后端。

    Dao-AILab 官方仓库自带，用 FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE 安装时**不编**
    CK 的 C++ kernel，纯 Triton JIT —— 因此不受 CK 那套按 gfx90a/gfx942 生成 instance
    的架构门槛限制，是 gfx936 上最有希望编出来的一条。
    """
    print("=" * 78)
    print("[2] flash-attn Triton AMD 后端（需先安装；本脚本只检测是否已可用）")
    try:
        from flash_attn import flash_attn_func
    except Exception as e:
        _record("flash_attn 包", "未安装", f"{type(e).__name__}")
        print("      安装方式（源码，纯 Triton、不编 CK）：")
        print("        git clone https://github.com/Dao-AILab/flash-attention")
        print("        cd flash-attention && FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE pip install -v -e . --no-build-isolation")
        return
    try:
        # flash_attn_func 吃 [B, S, H, D]
        fn = lambda a, b, c: flash_attn_func(a.transpose(1, 2), b.transpose(1, 2), c.transpose(1, 2)).transpose(1, 2)
        out, peak, dt = _bench(fn, q, k, v)
        err = (out.float() - ref.float()).abs().max().item()
        _record("flash_attn_func", "OK", f"max|Δ| vs math = {err:.2e}, 峰值 {peak:.2f} GiB, {dt:.1f} ms")
    except Exception as e:
        _record("flash_attn_func", "FAIL", f"{type(e).__name__}: {str(e)[:160]}")


def probe_flex_attention(q, k, v, ref):
    """torch 自带 FlexAttention（torch>=2.5）。

    零外部依赖，走 Inductor → Triton。原生支持 block-diagonal（document masking），
    正是 NaViT 打包路径需要的语义，且 BlockMask 会**跳过整块全 mask 的区域**。
    注意：必须 torch.compile，eager 下 flex_attention 会退化成物化 S×S 的实现。
    """
    print("=" * 78)
    print("[3] torch FlexAttention（自带，走 Inductor→Triton）")
    try:
        from torch.nn.attention.flex_attention import flex_attention
    except Exception as e:
        _record("flex_attention 导入", "不可用", f"{type(e).__name__}")
        return
    try:
        compiled = torch.compile(flex_attention, dynamic=False)
        out, peak, dt = _bench(lambda a, b, c: compiled(a, b, c), q, k, v, warmup=1, iters=3)
        err = (out.float() - ref.float()).abs().max().item()
        _record("flex_attention (compiled)", "OK",
                f"max|Δ| vs math = {err:.2e}, 峰值 {peak:.2f} GiB, {dt:.1f} ms")
    except Exception as e:
        _record("flex_attention (compiled)", "FAIL", f"{type(e).__name__}: {str(e)[:200]}")


def probe_math_baseline(q, k, v):
    print("=" * 78)
    print("[4] math SDPA 基线（当前 DCU 上唯一可用的）")
    from torch.nn.attention import SDPBackend, sdpa_kernel

    def fn(a, b, c):
        with sdpa_kernel(SDPBackend.MATH):
            return F.scaled_dot_product_attention(a, b, c)

    out, peak, dt = _bench(fn, q, k, v)
    _record("math SDPA", "OK", f"峰值 {peak:.2f} GiB, {dt:.1f} ms")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=4096, help="序列长度（真实痛点是 16384）")
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    probe_env()
    triton_ok = probe_triton_hello()

    dt = getattr(torch, args.dtype)
    q, k, v = (torch.randn(1, args.heads, args.seq, args.head_dim, device="cuda", dtype=dt)
               for _ in range(3))
    print(f"\n形状: B=1 H={args.heads} S={args.seq} D={args.head_dim} {args.dtype}")
    print(f"（math 后端会物化的 S×S: {args.heads * args.seq ** 2 * q.element_size() / 2**30:.2f} GiB）")

    ref = probe_math_baseline(q, k, v)
    probe_flash_attn_triton(q, k, v, ref)
    if triton_ok:
        probe_flex_attention(q, k, v, ref)
    else:
        _record("flex_attention (compiled)", "跳过", "triton 基础格没过，先解决那个")

    print("=" * 78)
    print("汇总")
    for name, status, detail in RESULTS:
        print(f"  {name:32s} {status:8s} {detail}")


if __name__ == "__main__":
    main()
