# -*- coding: utf-8 -*-
"""NaViT 链路加速裁决基准（Krea2 单流 12B 口径，云端跑）。

背景物证（krea2-C1 的 stage_timing.csv，10 个 step）：
  * 步内构成：forward 26.7% / backward 72.8%，其余（text_encode/optimizer/data）合计 0.5%
    → 模型内部之外的优化不值得做。
  * bwd/fwd = 2.73 ≈ 3 → grad checkpoint 重算 ≈ 一整个 forward ≈ 整步的 26.8%。
  * 代价模型 whole_step ≈ a·ΣN + b·ΣN_i²，a=7.411e-1 ms/tok、b=2.430e-5 ms/tok²，R²=0.972
    → 二次项（attention）占整步 ~24.6%，线性项（GEMM）~75%。
  * 等 token 不等 G 的对照（step300 G=5 vs step320 G=10，N≈66.3k）：forward 差 13.8%。

本脚本逐段裁决下列候选，全部给出「数值等价性 + 单块耗时 + 换算到整步的百分比」：

  S1 RoPE     ropeapply 现状(eager fp32) vs torch.compile 融合 vs bf16-native
  S2 Attention 逐段 SDPA(enable_gqa) vs 逐段 SDPA(repeat_kv) vs 强制 cudnn/flash 后端
              vs xformers BlockDiagonalMask vs torch varlen_attn(2.10+)
  S3 Block    真 SingleStreamBlock：无 ckpt / 全 ckpt / 选择性重算(SAC) 三档
              → grad_checkpoint_skip_last 的每块收益、SAC 是否值得
  S4 TREAD 上限  同一 block 在 keep=100%/75%/50% token 下的 fwd+bwd → 丢 token 的收益天花板
  S5 代价模型   在 (G, N) 网格上实测 block 耗时 → 拟合 a,b → 给出按代价装包的 λ

用法（云端 repo 根目录）::

    ./venv/bin/python AnimaLoraToolkit/tests/diag_navit_speed.py
    ./venv/bin/python AnimaLoraToolkit/tests/diag_navit_speed.py --only s1,s2
    ./venv/bin/python AnimaLoraToolkit/tests/diag_navit_speed.py --tokens 55778 --groups 6

只测速与数值等价，不加载权重、不写任何文件，随机初始化。预计 3~8 分钟、峰值显存
约 30~50 GB（--tokens 越大越高；OOM 会跳过该格并继续）。

诚实标注：单块 ×28 的外推**不含** LoRA/DoRA 注入、first/txtfusion/last 与 optimizer，
所以外推值会低于 CSV 的真实 forward；脚本会打印这条校准比，读结论时以**相对提升**
（各候选之间的比值）为准，不要把绝对值当整步预测。
"""

from __future__ import annotations

import argparse
import gc
import pathlib
import statistics
import sys
import time

import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Windows 控制台默认 GBK，报告里的 λ/² 会 UnicodeEncodeError；Linux 上是 no-op。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from models.krea2_modeling import (  # noqa: E402
    KREA2_LARGE_WIDE,
    PositionalEncoding,
    SingleStreamBlock,
    _SegLens,
    attention as krea2_attention,
    ropeapply,
)

# ── CSV 基准（krea2-C1，10 step 均值）───────────────────────────
CSV_WHOLE_MS = 53_836.6
CSV_FWD_MS = 15_433.9
CSV_BWD_MS = 42_107.8
CSV_A = 7.411e-1       # whole_step 线性系数 ms/token
CSV_B = 2.430e-5       # whole_step 二次系数 ms/token^2
N_BLOCKS = KREA2_LARGE_WIDE.layers          # 28
FEATURES = KREA2_LARGE_WIDE.features        # 6144
HEADS = KREA2_LARGE_WIDE.heads              # 48
KVHEADS = KREA2_LARGE_WIDE.kvheads          # 12
HEADDIM = FEATURES // HEADS                 # 128

DEV = "cuda"
DTYPE = torch.bfloat16


# ══════════════════════════════════════════════════════════════════════════
# 计时工具：交错轮换取中位，避免顺序测量把时钟/功耗漂移放大成假信号
# ══════════════════════════════════════════════════════════════════════════
def _sync():
    torch.cuda.synchronize()


def timeit(fn, warmup=2, rep=3):
    for _ in range(warmup):
        fn()
    _sync()
    ts = []
    for _ in range(rep):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def interleaved(cands: dict, rounds=3, warmup=1, rep=2):
    """cands: name -> callable。轮换执行顺序，每个 name 取各轮中位。
    某个候选抛异常（OOM / 内核不支持）时记为 None 并继续。"""
    names = list(cands)
    acc = {n: [] for n in names}
    for r in range(rounds):
        for n in names[r % len(names):] + names[: r % len(names)]:
            if acc[n] and acc[n][-1] is None:
                continue
            try:
                acc[n].append(timeit(cands[n], warmup=warmup, rep=rep))
            except Exception as e:  # noqa: BLE001
                acc[n].append(None)
                print(f"      [skip] {n}: {type(e).__name__}: {str(e)[:110]}")
                free_mem()
    return {n: (statistics.median([v for v in vs if v is not None])
                if any(v is not None for v in vs) else None)
            for n, vs in acc.items()}


def free_mem():
    gc.collect()
    torch.cuda.empty_cache()


def pct_of_step(ms_per_block_delta: float) -> float:
    """单块节省 ms → 占整步百分比（×28 块，除以 CSV 整步）。"""
    return ms_per_block_delta * N_BLOCKS / CSV_WHOLE_MS * 100.0


def fmt(v, unit="ms"):
    return "   n/a  " if v is None else f"{v:8.2f}{unit}"


# ══════════════════════════════════════════════════════════════════════════
# S0 环境探针
# ══════════════════════════════════════════════════════════════════════════
def probe_env():
    print("=" * 78)
    print("S0 环境")
    print("=" * 78)
    cap = torch.cuda.get_device_capability(0)
    print(f"  torch {torch.__version__}  cuda {torch.version.cuda}  "
          f"gpu {torch.cuda.get_device_name(0)}  sm{cap[0]}{cap[1]}  "
          f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB")
    caps = {}
    try:
        import xformers  # noqa: F401
        from xformers.ops.fmha import BlockDiagonalMask  # noqa: F401
        caps["xformers"] = True
    except Exception as e:  # noqa: BLE001
        caps["xformers"] = f"无（{type(e).__name__}）"
    try:
        from torch.nn.attention.varlen import varlen_attn  # noqa: F401
        caps["torch varlen_attn"] = True
    except Exception as e:  # noqa: BLE001
        caps["torch varlen_attn"] = f"无（{type(e).__name__}，需 torch>=2.10）"
    try:
        from torch.utils.checkpoint import create_selective_checkpoint_contexts  # noqa: F401
        caps["selective checkpoint(SAC)"] = True
    except Exception as e:  # noqa: BLE001
        caps["selective checkpoint(SAC)"] = f"无（{type(e).__name__}）"
    caps["cudnn attn 可用"] = torch.backends.cuda.cudnn_sdp_enabled()
    caps["flash 可用"] = torch.backends.cuda.flash_sdp_enabled()
    caps["mem_efficient 可用"] = torch.backends.cuda.mem_efficient_sdp_enabled()
    for k, v in caps.items():
        print(f"  {k:28s}: {v}")
    print()
    return caps


# ══════════════════════════════════════════════════════════════════════════
# S1 RoPE：eager fp32（现状） vs compile 融合 vs bf16-native
# ══════════════════════════════════════════════════════════════════════════
def _make_freqs(n_tok: int):
    """按真实 posemb 构造 freqs：pos [1,N,3]（frame=0, row, col），axes=[32,48,48]。"""
    axes = [HEADDIM - 12 * (HEADDIM // 16), 6 * (HEADDIM // 16), 6 * (HEADDIM // 16)]
    pe = PositionalEncoding(FEATURES, axes, theta=KREA2_LARGE_WIDE.theta, ntk=1.0)
    side = int(n_tok ** 0.5) + 1
    rows = (torch.arange(n_tok, device=DEV) // side).float()
    cols = (torch.arange(n_tok, device=DEV) % side).float()
    pos = torch.stack([torch.zeros_like(rows), rows, cols], dim=-1).unsqueeze(0)
    return pe(pos)                                   # [1, N, D/2, 2, 2] fp32


def _ropeapply_bf16(xq, xk, cos_b, sin_b):
    """bf16-native 变体：cos/sin 预转 bf16，旋转在 bf16 里做（会掉精度，仅作对照）。"""
    def _ap(x):
        x_ = x.reshape(*x.shape[:-1], -1, 2)
        x0, x1 = x_[..., 0], x_[..., 1]
        return torch.stack((x0 * cos_b - x1 * sin_b, x0 * sin_b + x1 * cos_b),
                           dim=-1).reshape(*x.shape)
    return _ap(xq), _ap(xk)


def section_s1(n_tok: int):
    print("=" * 78)
    print(f"S1 RoPE（ropeapply）  N={n_tok} tokens，q {HEADS}头 / k {KVHEADS}头 / D={HEADDIM}")
    print("=" * 78)
    freqs = _make_freqs(n_tok)
    q = torch.randn(1, HEADS, n_tok, HEADDIM, device=DEV, dtype=DTYPE)
    k = torch.randn(1, KVHEADS, n_tok, HEADDIM, device=DEV, dtype=DTYPE)
    cos_b = freqs[:, None, :, :, 0, 0].to(DTYPE)
    sin_b = freqs[:, None, :, :, 1, 0].to(DTYPE)

    rope_compiled = torch.compile(ropeapply, dynamic=True)
    try:
        rope_compiled(q, k, freqs)          # warm compile
        _sync()
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] torch.compile 预热失败：{type(e).__name__}: {str(e)[:120]}")

    print("  -- 纯前向 --")
    fwd = interleaved({
        "eager fp32(现状)": lambda: ropeapply(q, k, freqs),
        "compile fp32": lambda: rope_compiled(q, k, freqs),
        "bf16-native": lambda: _ropeapply_bf16(q, k, cos_b, sin_b),
    })
    base = fwd["eager fp32(现状)"]
    for n, v in fwd.items():
        sp = f"{base / v:5.2f}×" if (v and base) else "  -  "
        save = pct_of_step(base - v) if (v and base) else 0.0
        print(f"    {n:20s} {fmt(v)}  加速 {sp}   ×28块后省整步 {save:5.2f}%（仅前向）")

    # 数值等价
    ref = ropeapply(q, k, freqs)[0].float()
    for n, f in (("compile fp32", lambda: rope_compiled(q, k, freqs)),
                 ("bf16-native", lambda: _ropeapply_bf16(q, k, cos_b, sin_b))):
        try:
            got = f()[0].float()
            rel = ((ref - got).norm() / ref.norm()).item()
            print(f"    数值：{n:16s} 相对误差 {rel:.3e}"
                  + ("   ← 同 fp32 数学" if rel < 1e-4 else "   ← 有精度改变，需单独评估"))
        except Exception as e:  # noqa: BLE001
            print(f"    数值：{n:16s} 跳过（{type(e).__name__}）")

    # fwd+bwd（训练真实语境）
    print("  -- fwd+bwd（autograd）--")
    qg = q.clone().requires_grad_(True)
    kg = k.clone().requires_grad_(True)

    def _step(fn):
        def _f():
            qg.grad = None
            kg.grad = None
            a, b = fn(qg, kg)
            (a.float().pow(2).mean() + b.float().pow(2).mean()).backward()
        return _f

    try:
        _step(lambda a, b: rope_compiled(a, b, freqs))()
        _sync()
    except Exception:  # noqa: BLE001
        pass
    fb = interleaved({
        "eager fp32(现状)": _step(lambda a, b: ropeapply(a, b, freqs)),
        "compile fp32": _step(lambda a, b: rope_compiled(a, b, freqs)),
        "bf16-native": _step(lambda a, b: _ropeapply_bf16(a, b, cos_b, sin_b)),
    })
    base = fb["eager fp32(现状)"]
    for n, v in fb.items():
        sp = f"{base / v:5.2f}×" if (v and base) else "  -  "
        save = pct_of_step(base - v) if (v and base) else 0.0
        print(f"    {n:20s} {fmt(v)}  加速 {sp}   ×28块后省整步 {save:5.2f}%（fwd+bwd）")
    print("  注：fwd+bwd 里含测试用 loss 的开销，真实收益介于两组之间。\n")
    del q, k, qg, kg, freqs, cos_b, sin_b
    free_mem()


# ══════════════════════════════════════════════════════════════════════════
# S2 Attention 内核矩阵
# ══════════════════════════════════════════════════════════════════════════
def _seg_split(n_tok: int, g: int):
    base = n_tok // g
    segs = [base] * g
    segs[-1] += n_tok - base * g
    return segs


def _attn_seg_sdpa(q, k, v, segs, gqa_mode: str, backend=None):
    """逐段 dense SDPA（现状 sdpa_seg 后端）。gqa_mode: 'enable_gqa' | 'repeat'。"""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    ctx = sdpa_kernel([backend]) if backend is not None else None
    outs, off = [], 0
    if ctx is not None:
        ctx.__enter__()
    try:
        for s in segs:
            qs, ks, vs = q[:, :, off:off + s], k[:, :, off:off + s], v[:, :, off:off + s]
            if gqa_mode == "enable_gqa":
                o = F.scaled_dot_product_attention(qs, ks, vs, enable_gqa=True)
            else:
                rep = qs.shape[1] // ks.shape[1]
                o = F.scaled_dot_product_attention(
                    qs, ks.repeat_interleave(rep, dim=1), vs.repeat_interleave(rep, dim=1))
            outs.append(o)
            off += s
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)
    return torch.cat(outs, dim=2).transpose(1, 2).reshape(1, q.shape[2], -1)


def _attn_xformers(q, k, v, segs):
    from models.krea2_modeling import cached_block_diag_mask
    bias = cached_block_diag_mask(tuple(segs))
    return krea2_attention(q, k, v, mask=bias, gqa=True)


def _attn_varlen(q, k, v, segs):
    import inspect
    import itertools

    from torch.nn.attention.varlen import varlen_attn
    cu = torch.tensor([0] + list(itertools.accumulate(int(s) for s in segs)),
                      dtype=torch.int32, device=q.device)
    # GQA：新版签名有 enable_gqa，旧版（torch 2.11）没有 → 手动展开 KV 头（数值相同）。
    has_gqa_kw = "enable_gqa" in inspect.signature(varlen_attn).parameters
    if q.shape[1] != k.shape[1] and not has_gqa_kw:
        rep = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    qv = q[0].transpose(0, 1).contiguous()      # [T, Hq, D]
    kv = k[0].transpose(0, 1).contiguous()
    vv = v[0].transpose(0, 1).contiguous()
    mx = int(max(segs))
    kw = {"enable_gqa": True} if (has_gqa_kw and q.shape[1] != k.shape[1]) else {}
    out = varlen_attn(qv, kv, vv, cu, cu, mx, mx, **kw)
    return out.reshape(1, q.shape[2], -1)


def section_s2(n_tok: int, groups, caps):
    from torch.nn.attention import SDPBackend
    print("=" * 78)
    print(f"S2 Attention 内核（N={n_tok}，GQA {HEADS}/{KVHEADS}，D={HEADDIM}，bf16）")
    print("   基准：CSV 拟合出 attention(二次项) 占整步 ~24.6%")
    print("=" * 78)
    for g in groups:
        segs = _seg_split(n_tok, g)
        print(f"  ── G={g}（段长 ≈ {segs[0]}）")
        q = torch.randn(1, HEADS, n_tok, HEADDIM, device=DEV, dtype=DTYPE)
        k = torch.randn(1, KVHEADS, n_tok, HEADDIM, device=DEV, dtype=DTYPE)
        v = torch.randn(1, KVHEADS, n_tok, HEADDIM, device=DEV, dtype=DTYPE)

        cands = {
            "seg+enable_gqa(现状)": lambda: _attn_seg_sdpa(q, k, v, segs, "enable_gqa"),
            "seg+repeat_kv": lambda: _attn_seg_sdpa(q, k, v, segs, "repeat"),
            "seg+repeat_kv@CUDNN": lambda: _attn_seg_sdpa(q, k, v, segs, "repeat",
                                                          SDPBackend.CUDNN_ATTENTION),
            "seg+repeat_kv@FLASH": lambda: _attn_seg_sdpa(q, k, v, segs, "repeat",
                                                          SDPBackend.FLASH_ATTENTION),
            "seg+enable_gqa@MATH": lambda: _attn_seg_sdpa(q, k, v, segs, "enable_gqa",
                                                          SDPBackend.MATH),
        }
        if caps.get("xformers") is True:
            cands["xformers varlen"] = lambda: _attn_xformers(q, k, v, segs)
        if caps.get("torch varlen_attn") is True:
            cands["torch varlen_attn"] = lambda: _attn_varlen(q, k, v, segs)

        res = interleaved(cands, rounds=3)
        base = res.get("seg+enable_gqa(现状)")
        for n, val in res.items():
            sp = f"{base / val:5.2f}×" if (val and base) else "  -  "
            save = pct_of_step(base - val) if (val and base) else 0.0
            print(f"    {n:22s} {fmt(val)}  vs现状 {sp}   ×28块后省整步 {save:5.2f}%（仅前向）")

        # 数值等价（以 seg+repeat_kv 为参照；bf16 下 1e-2 量级即视为同数学不同内核）
        try:
            ref = _attn_seg_sdpa(q, k, v, segs, "repeat").float()
            for n, f in cands.items():
                if n == "seg+repeat_kv":
                    continue
                try:
                    got = f().float()
                    rel = ((ref - got).norm() / ref.norm()).item()
                    print(f"    数值：{n:22s} 相对误差 {rel:.3e}")
                except Exception as e:  # noqa: BLE001
                    print(f"    数值：{n:22s} 跳过（{type(e).__name__}）")
        except Exception as e:  # noqa: BLE001
            print(f"    数值参照构建失败：{type(e).__name__}")

        # backward 安全性（sm120 flash backward 有 IMA 前科，必须单独验）
        print("    -- backward 可用性/耗时 --")
        qg = q.clone().requires_grad_(True)
        kg = k.clone().requires_grad_(True)
        vg = v.clone().requires_grad_(True)

        def _bwd(fn):
            def _f():
                qg.grad = kg.grad = vg.grad = None
                fn(qg, kg, vg).float().pow(2).mean().backward()
            return _f

        bcands = {
            "seg+enable_gqa(现状)": _bwd(lambda a, b, c: _attn_seg_sdpa(a, b, c, segs, "enable_gqa")),
            "seg+repeat_kv": _bwd(lambda a, b, c: _attn_seg_sdpa(a, b, c, segs, "repeat")),
        }
        if caps.get("xformers") is True:
            bcands["xformers varlen"] = _bwd(lambda a, b, c: _attn_xformers(a, b, c, segs))
        if caps.get("torch varlen_attn") is True:
            bcands["torch varlen_attn"] = _bwd(lambda a, b, c: _attn_varlen(a, b, c, segs))
        bres = interleaved(bcands, rounds=2)
        bbase = bres.get("seg+enable_gqa(现状)")
        for n, val in bres.items():
            sp = f"{bbase / val:5.2f}×" if (val and bbase) else "  -  "
            print(f"    {n:22s} {fmt(val)}  vs现状 {sp}")
        del q, k, v, qg, kg, vg
        free_mem()
    print()


# ══════════════════════════════════════════════════════════════════════════
# S3 真 block：无 ckpt / 全 ckpt / SAC
# ══════════════════════════════════════════════════════════════════════════
def _build_block():
    blk = SingleStreamBlock(FEATURES, HEADS, KREA2_LARGE_WIDE.multiplier,
                            KREA2_LARGE_WIDE.bias, KVHEADS).to(DEV, DTYPE)
    for p in blk.parameters():
        p.requires_grad_(True)
    return blk


def _sac_context_fn():
    from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts
    save_ops = {
        torch.ops.aten.mm.default,
        torch.ops.aten.addmm.default,
        torch.ops.aten.bmm.default,
        torch.ops.aten._scaled_dot_product_efficient_attention.default,
        torch.ops.aten._scaled_dot_product_flash_attention.default,
        torch.ops.aten._scaled_dot_product_cudnn_attention.default,
    }

    def policy(ctx, op, *a, **kw):
        return (CheckpointPolicy.MUST_SAVE if op in save_ops
                else CheckpointPolicy.PREFER_RECOMPUTE)

    return lambda: create_selective_checkpoint_contexts(policy)


def section_s3(n_tok: int, g: int, caps):
    from torch.utils.checkpoint import checkpoint
    print("=" * 78)
    print(f"S3 真 SingleStreamBlock（{FEATURES}ch，N={n_tok}，G={g}）")
    print(f"   基准：CSV forward {CSV_FWD_MS:.0f}ms / backward {CSV_BWD_MS:.0f}ms / "
          f"整步 {CSV_WHOLE_MS:.0f}ms；bwd/fwd=2.73 → 重算约占整步 26.8%")
    print("=" * 78)
    segs = _seg_split(n_tok, g)
    blk = _build_block()
    x = torch.randn(1, n_tok, FEATURES, device=DEV, dtype=DTYPE, requires_grad=True)
    vec = torch.randn(1, g, FEATURES * 6, device=DEV, dtype=DTYPE)
    mod_index = torch.repeat_interleave(
        torch.arange(g, device=DEV), torch.tensor(segs, device=DEV))
    freqs = _make_freqs(n_tok)
    mask = _SegLens(segs)

    def run(inp):
        return blk(inp, vec, freqs, mask, mod_index=mod_index)

    def _fwd_only():
        with torch.no_grad():
            run(x)

    def _mk(fn):
        def _f():
            x.grad = None
            for p in blk.parameters():
                p.grad = None
            fn().float().pow(2).mean().backward()
        return _f

    cands = {
        "fwd only(no grad)": _fwd_only,
        "fwd+bwd 无ckpt": _mk(lambda: run(x)),
        "fwd+bwd 全ckpt(现状)": _mk(lambda: checkpoint(run, x, use_reentrant=False)),
    }
    if caps.get("selective checkpoint(SAC)") is True:
        ctx_fn = _sac_context_fn()
        cands["fwd+bwd SAC"] = _mk(
            lambda: checkpoint(run, x, use_reentrant=False, context_fn=ctx_fn))

    res = interleaved(cands, rounds=3, warmup=1, rep=2)
    for n, v in res.items():
        print(f"    {n:22s} {fmt(v)}   ×28块 = {fmt(v * N_BLOCKS / 1000 if v else None, 's')}")

    ck0 = res.get("fwd+bwd 全ckpt(现状)")
    if ck0:
        extrap = ck0 * N_BLOCKS
        r = extrap / (CSV_FWD_MS + CSV_BWD_MS)
        print(f"\n    [校准] 本机单块×28 = {extrap / 1000:.1f}s  vs  CSV(forward+backward) = "
              f"{(CSV_FWD_MS + CSV_BWD_MS) / 1000:.1f}s  → 比值 r={r:.2f}")
        print(f"           r 略小于 1 属正常（外推不含 LoRA/DoRA 与 first/txtfusion/last）；"
              f"若 r 严重偏离 1，上文所有『省整步 %』要按 1/r 折算后再读。")

    ck = res.get("fwd+bwd 全ckpt(现状)")
    nock = res.get("fwd+bwd 无ckpt")
    sac = res.get("fwd+bwd SAC")
    if ck and nock:
        per_blk = ck - nock
        # ★ 每块重算代价换算成整步占比时**不能**再乘 28（pct_of_step 里已经乘过一次）：
        #   skip_last=K 只省掉 K 块的重算，K=28（全关 ckpt）才等于 pct_of_step(per_blk)。
        per_blk_pct = per_blk / CSV_WHOLE_MS * 100.0
        print(f"\n    → 每块重算代价 = {per_blk:.1f} ms = 整步的 {per_blk_pct:.2f}%")
        print("      grad_checkpoint_skip_last 收益（线性）："
              + "  ".join(f"K={k} → {per_blk_pct * k:.1f}%" for k in (1, 4, 8, N_BLOCKS)))
        print(f"      代价：每 skip 1 块多占显存 ≈ 一块的激活量（用 nvidia-smi 峰值确认）")
    if ck and sac:
        print(f"    → SAC vs 全ckpt：{ck / sac:5.2f}×，省整步 {pct_of_step(ck - sac):.2f}%"
              f"（显存介于无ckpt与全ckpt之间）")

    # 显存峰值
    torch.cuda.reset_peak_memory_stats()
    try:
        _mk(lambda: checkpoint(run, x, use_reentrant=False))()
        print(f"    全ckpt 单块峰值显存 {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    except Exception as e:  # noqa: BLE001
        print(f"    峰值显存测量跳过：{type(e).__name__}")
    print()
    del blk, x, vec, freqs
    free_mem()
    return res


def section_s4(n_tok: int, g: int):
    """TREAD 上限：同一 block 在 keep=100/75/50% token 下的 fwd+bwd。"""
    from torch.utils.checkpoint import checkpoint
    print("=" * 78)
    print("S4 token 丢弃（TREAD）收益上限 —— 同 block 在不同保留率下的 fwd+bwd")
    print("=" * 78)
    blk = _build_block()
    res = {}
    for keep in (1.0, 0.75, 0.5):
        n = int(n_tok * keep)
        segs = _seg_split(n, g)
        x = torch.randn(1, n, FEATURES, device=DEV, dtype=DTYPE, requires_grad=True)
        vec = torch.randn(1, g, FEATURES * 6, device=DEV, dtype=DTYPE)
        mod_index = torch.repeat_interleave(
            torch.arange(g, device=DEV), torch.tensor(segs, device=DEV))
        freqs = _make_freqs(n)
        mask = _SegLens(segs)

        def _f(x=x, vec=vec, freqs=freqs, mask=mask, mod_index=mod_index):
            x.grad = None
            for p in blk.parameters():
                p.grad = None
            out = checkpoint(lambda i: blk(i, vec, freqs, mask, mod_index=mod_index),
                             x, use_reentrant=False)
            out.float().pow(2).mean().backward()

        try:
            res[keep] = timeit(_f, warmup=1, rep=3)
        except Exception as e:  # noqa: BLE001
            res[keep] = None
            print(f"    [skip] keep={keep}: {type(e).__name__}")
        del x, vec, freqs
        free_mem()
    base = res.get(1.0)
    for keep, v in res.items():
        sp = f"{base / v:5.2f}×" if (v and base) else "  -  "
        save = pct_of_step(base - v) if (v and base) else 0.0
        print(f"    keep={keep:4.0%}  {fmt(v)}  vs全量 {sp}   若中间层全程丢弃 → 省整步 {save:5.2f}%")
    print("    注：这是**上限**——TREAD 只在部分层丢 token，实际收益按丢弃层数比例折算。\n")
    del blk
    free_mem()


# ══════════════════════════════════════════════════════════════════════════
# S5 代价模型拟合（按代价装包的 λ）
# ══════════════════════════════════════════════════════════════════════════
def section_s5(tokens_grid, groups_grid):
    from torch.utils.checkpoint import checkpoint
    print("=" * 78)
    print("S5 代价模型：block fwd+bwd(全ckpt) 在 (N, G) 网格上实测 → 拟合 t = a·N + b·ΣL²")
    print(f"   CSV 侧参照：a={CSV_A:.4g} ms/tok, b={CSV_B:.4g} ms/tok², λ=b/a={CSV_B / CSV_A:.3e}")
    print("=" * 78)
    blk = _build_block()
    rows = []
    for n_tok in tokens_grid:
        for g in groups_grid:
            if g > n_tok // 512:
                continue
            segs = _seg_split(n_tok, g)
            try:
                x = torch.randn(1, n_tok, FEATURES, device=DEV, dtype=DTYPE, requires_grad=True)
                vec = torch.randn(1, g, FEATURES * 6, device=DEV, dtype=DTYPE)
                mod_index = torch.repeat_interleave(
                    torch.arange(g, device=DEV), torch.tensor(segs, device=DEV))
                freqs = _make_freqs(n_tok)
                mask = _SegLens(segs)

                def _f():
                    x.grad = None
                    for p in blk.parameters():
                        p.grad = None
                    out = checkpoint(lambda i: blk(i, vec, freqs, mask, mod_index=mod_index),
                                     x, use_reentrant=False)
                    out.float().pow(2).mean().backward()

                ms = timeit(_f, warmup=1, rep=2)
                s2 = float(sum(s * s for s in segs))
                rows.append((n_tok, g, s2, ms))
                print(f"    N={n_tok:6d} G={g:3d}  ΣL²={s2:.3e}  {ms:8.2f} ms")
            except Exception as e:  # noqa: BLE001
                print(f"    [skip] N={n_tok} G={g}: {type(e).__name__}")
            finally:
                x = vec = freqs = mask = mod_index = None
                free_mem()
    if len(rows) >= 3:
        A = torch.tensor([[r[0], r[2]] for r in rows], dtype=torch.float64)
        y = torch.tensor([r[3] for r in rows], dtype=torch.float64)
        sol = torch.linalg.lstsq(A, y.unsqueeze(1)).solution.squeeze(1)
        a_, b_ = float(sol[0]), float(sol[1])
        pred = (A @ sol)
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        print(f"\n    拟合（单块）：a={a_:.4e} ms/tok  b={b_:.4e} ms/tok²  R²={float(r2):.4f}")
        if a_ > 0:
            print(f"    λ = b/a = {b_ / a_:.3e}  → 按代价装包的预算函数：")
            print(f"        cost(pack) = ΣN_i + {b_ / a_:.3e} · ΣN_i²   （≤ 等效 token 预算 C）")
            print(f"        直觉：一张 {int(1 / (b_ / a_)):d} token 的图，代价 = 两张同 token 小图之和")
        print(f"    与 CSV 整步拟合的 λ={CSV_B / CSV_A:.3e} 对照：两者接近即说明单块外推可用于装包决策")
    print()
    del blk
    free_mem()


# ══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="NaViT 链路加速裁决基准（Krea2 口径）")
    ap.add_argument("--tokens", type=int, default=55778,
                    help="主测试 token 数（默认取 CSV 均值 55778）")
    ap.add_argument("--groups", type=int, default=6, help="主测试 G（默认 CSV 均值 6）")
    ap.add_argument("--s2-groups", type=str, default="4,6,10",
                    help="S2 用的 G 列表（对照等 token 不等 G）")
    ap.add_argument("--s5-tokens", type=str, default="16384,32768,55778",
                    help="S5 网格的 N 列表")
    ap.add_argument("--s5-groups", type=str, default="1,2,4,8,16",
                    help="S5 网格的 G 列表")
    ap.add_argument("--only", type=str, default="s1,s2,s3,s4,s5",
                    help="只跑其中几段，如 --only s1,s3")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("需要 CUDA。")
        return 1
    torch.manual_seed(0)
    want = {s.strip().lower() for s in args.only.split(",") if s.strip()}
    caps = probe_env()

    if "s1" in want:
        section_s1(args.tokens)
    if "s2" in want:
        section_s2(args.tokens, [int(x) for x in args.s2_groups.split(",")], caps)
    if "s3" in want:
        section_s3(args.tokens, args.groups, caps)
    if "s4" in want:
        section_s4(args.tokens, args.groups)
    if "s5" in want:
        section_s5([int(x) for x in args.s5_tokens.split(",")],
                   [int(x) for x in args.s5_groups.split(",")])

    print("=" * 78)
    print("读法提示")
    print("=" * 78)
    print("  * 「省整步 %」= 单块节省 ×28 ÷ CSV 整步 53.8s。它**低估**真实占比"
          "（外推不含 LoRA/DoRA 与块外层），各候选之间的**比值**才是可靠结论。")
    print("  * S1 只在相对误差 <1e-4 的候选里选（compile 应该满足；bf16-native 不满足，"
          "要单独评估画质影响后才谈）。")
    print("  * S2 若 seg+enable_gqa 与 seg+enable_gqa@MATH 耗时接近 → 说明现状确实落在 "
          "math 后端，换 repeat_kv/varlen 是纯赚。")
    print("  * S2 的 backward 段若 varlen/xformers 抛 IMA 或 illegal memory access，"
          "该路线在本卡上直接出局（sm120 有前科）。")
    print("  * S3 的「每块重算代价」直接给出 grad_checkpoint_skip_last 的收益曲线；"
          "配合 nvidia-smi 峰值决定能 skip 几块。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
