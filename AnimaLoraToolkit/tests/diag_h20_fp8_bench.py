# -*- coding: utf-8 -*-
"""H20 fp8/attention 微基准：裁决「fp8 quant-GEMM 前向为何不比 bf16 快」。

背景（2026-07 stage_timing 物证）：
  * fp8ab run（navit, budget 16384, adamw）G=4 前向 7.30s；
    soap_sf+lokr+dora run（同数据同 budget, bf16）G=4 前向 5.55–5.72s；
    ARB dense bf16（minimal_diag, bs4@1024, 同 token 几何）前向 3.75s。
  * 日志确认 fp8 走的是真 quant-GEMM（258 层 rowwise），非 dequant 回退。
  → 嫌疑：① rowwise ``_scaled_mm`` 在 H20 上本身慢；② 逐层激活量化开销；
    ③ packed(navit) attention 后端比 dense 慢。三者本脚本分段裁决。

用法（云端 H20，repo venv python）::

    python tests/diag_h20_fp8_bench.py            # 全部（gemm + bwd + attn）
    python tests/diag_h20_fp8_bench.py --skip-attn

纯 torch 即可跑 GEMM 部分；attention 部分依赖 xformers（可选 flash_attn），
缺哪个就跳过哪段并明说。总耗时 ~1–2 分钟。
形状 = Krea2 KREA2_LARGE_WIDE 主 block 的 8 个 Linear（28 块共 224 层，
占量化层的绝对大头）；量化/反向实现逐行对齐 trainer/quant.py 的热路径。
"""

from __future__ import annotations

import argparse
import sys

import torch
import torch.nn.functional as F

_E4M3_MAX = 448.0
_E5M2_MAX = 57344.0

# (标签, K=in, N=out, 每 block 个数) —— Krea2 主 block
MAIN_SHAPES = [
    ("wq/gate/wo 6144x6144", 6144, 6144, 3),
    ("wk/wv     6144x1536", 6144, 1536, 2),
    ("mlp g/u   6144x16384", 6144, 16384, 2),
    ("mlp.down 16384x6144", 16384, 6144, 1),
]
N_BLOCKS = 28

# token 数：4608=单图1024²+512text；18432=ARB bs4@1024 / navit 16k 口径；
# 36864≈budget 32768 + text
M_LIST = [4608, 18432, 36864]


def bench(fn, warmup: int = 10, iters: int = 30) -> float:
    """中位数 ms（CUDA event，每次 sync）。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


# ── 与 trainer/quant.py 逐行同款的量化原语 ────────────────────────────
def quant_rowwise(t, fmax=_E4M3_MAX, dtype=torch.float8_e4m3fn):
    scale = t.abs().amax(dim=1, keepdim=True).float().div(fmax).clamp(min=1e-12)
    return (t / scale.to(t.dtype)).clamp(-fmax, fmax).to(dtype), scale


def quant_tensorwise(t, fmax=_E4M3_MAX, dtype=torch.float8_e4m3fn):
    scale = t.abs().amax().float().div(fmax).clamp(min=1e-12)
    return (t / scale.to(t.dtype)).clamp(-fmax, fmax).to(dtype), scale.reshape(1, 1)


def dequant(q, s, dtype=torch.bfloat16):
    return (q.to(torch.float32) * s).to(dtype)


def section_gemm(dev):
    print("\n══ 1. GEMM 前向：bf16 F.linear vs fp8 _scaled_mm（含激活量化）══")
    print(f"{'shape':<22}{'M':>7} {'bf16':>8} {'fp8row':>8} {'fp8tsr':>8} "
          f"{'actq':>7} {'row加速':>8} {'tsr加速':>8}")
    totals = {}  # M -> [bf16, row, tsr] 单 block 合计
    for label, K, N, cnt in MAIN_SHAPES:
        w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
        wq_r, ws_r = quant_rowwise(w)
        wq_t, ws_t = quant_tensorwise(w)
        for M in M_LIST:
            try:
                x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
                t_bf16 = bench(lambda: F.linear(x, w))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"{label:<22}{M:>7} [skip] OOM（小显存设备）")
                continue

            def f_row():
                xq, xs = quant_rowwise(x)
                return torch._scaled_mm(xq, wq_r.t(), scale_a=xs,
                                        scale_b=ws_r.t(), out_dtype=torch.bfloat16)

            def f_tsr():
                xq, xs = quant_tensorwise(x)
                return torch._scaled_mm(xq, wq_t.t(), scale_a=xs,
                                        scale_b=ws_t, out_dtype=torch.bfloat16)

            try:
                t_row = bench(f_row)
            except Exception as ex:
                t_row = float("nan")
                print(f"  [!] rowwise _scaled_mm 失败: {type(ex).__name__}: {ex}")
            try:
                t_tsr = bench(f_tsr)
            except Exception as ex:
                t_tsr = float("nan")
                print(f"  [!] tensorwise _scaled_mm 失败: {type(ex).__name__}: {ex}")
            t_actq = bench(lambda: quant_rowwise(x))

            acc = totals.setdefault(M, [0.0, 0.0, 0.0])
            acc[0] += t_bf16 * cnt
            acc[1] += t_row * cnt
            acc[2] += t_tsr * cnt
            print(f"{label:<22}{M:>7} {t_bf16:>8.3f} {t_row:>8.3f} {t_tsr:>8.3f} "
                  f"{t_actq:>7.3f} {t_bf16 / t_row:>7.2f}x {t_bf16 / t_tsr:>7.2f}x")

    print("\n—— 单 block 8-Linear 前向合计（×28 块 = 整网 Linear 前向）——")
    for M, (b, r, t) in sorted(totals.items()):
        print(f"  M={M:<6} bf16 {b:7.3f}ms  fp8row {r:7.3f}ms ({b / r:4.2f}x)  "
              f"fp8tsr {t:7.3f}ms ({b / t:4.2f}x)   整网≈ bf16 {b * N_BLOCKS / 1000:.2f}s "
              f"/ fp8row {r * N_BLOCKS / 1000:.2f}s")


def section_backward(dev):
    print("\n══ 2. 反向 dL/dx：bf16 / dequant / fp8_grad（对齐 _Fp8GemmLinearFn.backward）══")
    print(f"{'shape':<22}{'M':>7} {'bf16':>8} {'dequant':>8} {'fp8grad':>8} "
          f"{'转置拷贝':>8}")
    for label, K, N, cnt in MAIN_SHAPES:
        w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
        wq, ws = quant_rowwise(w)
        for M in M_LIST:
            try:
                gy = torch.randn(M, N, device=dev, dtype=torch.bfloat16)
                t_bf16 = bench(lambda: gy @ w)
                t_deq = bench(lambda: gy @ dequant(wq, ws))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"{label:<22}{M:>7} [skip] OOM（小显存设备）")
                continue

            ones = ws.new_ones(1, K)

            def f_grad():
                # 逐行对齐 trainer/quant.py:301-311
                wq_cm = wq.t().contiguous().t()
                g = gy * ws.t().to(gy.dtype)
                gq, gs = quant_rowwise(g, fmax=_E5M2_MAX, dtype=torch.float8_e5m2)
                return torch._scaled_mm(gq, wq_cm, scale_a=gs, scale_b=ones,
                                        out_dtype=gy.dtype)

            try:
                t_g = bench(f_grad)
            except Exception as ex:
                t_g = float("nan")
                print(f"  [!] fp8_grad 路径失败: {type(ex).__name__}: {ex}")
            t_tp = bench(lambda: wq.t().contiguous())
            print(f"{label:<22}{M:>7} {t_bf16:>8.3f} {t_deq:>8.3f} {t_g:>8.3f} "
                  f"{t_tp:>8.3f}")


def section_attention(dev):
    print("\n══ 3. attention：dense SDPA vs xformers packed（navit 税裁决）══")
    # 段几何：4×4608（=ARB bs4@1024 同款）/ 2×9728（1536 大图）/ 10×2150（小图包）
    CASES = [("4x4608", [4608] * 4), ("2x9728", [9728] * 2), ("10x2150", [2150] * 10)]
    Hq, Hkv, D = 48, 12, 128
    try:
        import xformers.ops as xops
    except ImportError:
        print("  [skip] 无 xformers")
        xops = None

    try:
        from flash_attn import flash_attn_varlen_func
    except ImportError:
        flash_attn_varlen_func = None

    for name, segs in CASES:
        L = sum(segs)
        B = len(segs)
        print(f"\n  段几何 {name}（ΣL={L}）")

        # dense：B 个等长段 → [B, H, Lseg, D]（ARB 的实际布局）
        if len(set(segs)) == 1:
            Ls = segs[0]
            q = torch.randn(B, Hq, Ls, D, device=dev, dtype=torch.bfloat16,
                            requires_grad=True)
            k = torch.randn(B, Hkv, Ls, D, device=dev, dtype=torch.bfloat16,
                            requires_grad=True)
            v = torch.randn_like(k).requires_grad_(True)

            def f_dense():
                return F.scaled_dot_product_attention(q, k, v, enable_gqa=True)

            def fb_dense():
                F.scaled_dot_product_attention(q, k, v, enable_gqa=True).sum().backward()

            try:
                print(f"    dense SDPA(enable_gqa)    fwd {bench(f_dense):7.3f}ms  "
                      f"fwd+bwd {bench(fb_dense):7.3f}ms")
            except Exception as ex:
                print(f"    dense SDPA 失败: {ex}")

        if xops is None:
            continue
        mask = xops.fmha.BlockDiagonalMask.from_seqlens(segs)

        # navit 实际路径：5D grouped GQA [1, L, G_kv, rep, D]
        rep = Hq // Hkv
        q5 = torch.randn(1, L, Hkv, rep, D, device=dev, dtype=torch.bfloat16,
                         requires_grad=True)
        kv5 = torch.randn(1, L, Hkv, 1, D, device=dev, dtype=torch.bfloat16,
                          requires_grad=True)

        def f_5d():
            return xops.memory_efficient_attention(
                q5, kv5.expand(1, L, Hkv, rep, D), kv5.expand(1, L, Hkv, rep, D),
                attn_bias=mask)

        def fb_5d():
            xops.memory_efficient_attention(
                q5, kv5.expand(1, L, Hkv, rep, D), kv5.expand(1, L, Hkv, rep, D),
                attn_bias=mask).sum().backward()

        try:
            print(f"    xf 5D grouped (navit现路径) fwd {bench(f_5d):7.3f}ms  "
                  f"fwd+bwd {bench(fb_5d):7.3f}ms")
        except Exception as ex:
            print(f"    xf 5D grouped 失败: {ex}")

        # 4D + KV 物化展开（navit 的 fallback 路径）
        q4 = torch.randn(1, L, Hq, D, device=dev, dtype=torch.bfloat16,
                         requires_grad=True)
        k4 = torch.randn(1, L, Hq, D, device=dev, dtype=torch.bfloat16,
                         requires_grad=True)
        v4 = torch.randn_like(k4).requires_grad_(True)

        def f_4d():
            return xops.memory_efficient_attention(q4, k4, v4, attn_bias=mask)

        def fb_4d():
            xops.memory_efficient_attention(q4, k4, v4, attn_bias=mask).sum().backward()

        try:
            print(f"    xf 4D expanded-KV          fwd {bench(f_4d):7.3f}ms  "
                  f"fwd+bwd {bench(fb_4d):7.3f}ms")
        except Exception as ex:
            print(f"    xf 4D 失败: {ex}")

        if flash_attn_varlen_func is not None:
            cu = torch.tensor([0] + list(torch.tensor(segs).cumsum(0)),
                              device=dev, dtype=torch.int32)
            qf = torch.randn(L, Hq, D, device=dev, dtype=torch.bfloat16,
                             requires_grad=True)
            kf = torch.randn(L, Hkv, D, device=dev, dtype=torch.bfloat16,
                             requires_grad=True)
            vf = torch.randn_like(kf).requires_grad_(True)
            mx = max(segs)

            def f_fa():
                return flash_attn_varlen_func(qf, kf, vf, cu, cu, mx, mx)

            def fb_fa():
                flash_attn_varlen_func(qf, kf, vf, cu, cu, mx, mx).sum().backward()

            try:
                print(f"    flash-attn varlen(原生GQA) fwd {bench(f_fa):7.3f}ms  "
                      f"fwd+bwd {bench(fb_fa):7.3f}ms")
            except Exception as ex:
                print(f"    flash varlen 失败: {ex}")


def section_block(dev):
    """真实 SingleStreamBlock 闭环：dense vs packed vs fp8，×28 对 stage_timing 的账。

    背景：组件加总（Linear+attention）解释了 ARB bf16 的 3.75s 前向，却解释不了
    fp8+navit 的 7.3s（缺口 ~4.6s）。本段用真 block（含 RMSNorm/AdaLN/rope/门控
    全部 pointwise）复现，看缺口出在 block 内还是 block 外。无 LoRA（若本段
    ×28 仍对不上 stage_timing，嫌疑转向 LoRA 适配器/循环外逻辑）。
    """
    import sys as _sys
    from pathlib import Path
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    try:
        from models.krea2_modeling import (
            SingleStreamBlock, PositionalEncoding, block_diag_bool_mask)
    except ImportError as ex:
        print(f"\n══ 4. block 级闭环 [skip] 导入失败: {ex}（请在 AnimaLoraToolkit 目录下运行）══")
        return
    try:
        from models.krea2_modeling import cached_block_diag_mask
        import xformers  # noqa: F401  确认可用（云端路径）
        _make_mask = lambda segs: cached_block_diag_mask(tuple(segs))
        print("\n（packed mask = xformers BlockDiagonalMask，与云端训练一致）")
    except ImportError:
        _make_mask = lambda segs: block_diag_bool_mask(list(segs), dev)
        print("\n（无 xformers → packed mask 用 bool 回退，仅本地冒烟用，"
              "云端结果以 xformers 路径为准）")
    try:
        from trainer.quant import QuantLinear
    except ImportError:
        QuantLinear = None

    F_DIM, HEADS, KVH, MULT, HEADDIM = 6144, 48, 12, 4, 128
    N_LAYERS = 28
    print("\n══ 4. 真实 SingleStreamBlock 闭环（×28 外推 ≈ navit_model_forward）══")

    def make_block():
        blk = SingleStreamBlock(F_DIM, HEADS, MULT, bias=False, kvheads=KVH)
        blk = blk.to(dev, torch.bfloat16)
        for p in blk.parameters():
            p.requires_grad_(False)
        return blk

    def quantize_block(blk, rowwise=True):
        a, m = blk.attn, blk.mlp
        for parent, name in [(a, "wq"), (a, "wk"), (a, "wv"), (a, "gate"),
                             (a, "wo"), (m, "gate"), (m, "up"), (m, "down")]:
            lin = getattr(parent, name)
            setattr(parent, name, QuantLinear.from_linear(
                lin, "fp8", "fp8_gemm", fp8_rowwise=rowwise, fp8_grad=True))
        return blk

    posemb = PositionalEncoding(HEADDIM, [32, 48, 48], theta=1e3).to(dev)

    def make_pos(segs):
        # 每段一张方形网格图（frame=0, row, col）——与训练同构的 3D axial 位置
        parts = []
        for s in segs:
            side = int(s ** 0.5) + 1
            idx = torch.arange(s, device=dev)
            p = torch.zeros(s, 3, device=dev)
            p[:, 1] = (idx // side).float()
            p[:, 2] = (idx % side).float()
            parts.append(p)
        return torch.cat(parts).unsqueeze(0)

    import os
    if os.environ.get("DIAG_SMALL"):
        geoms = [("4x512(冒烟)", [512] * 4), ("2x1024(冒烟)", [1024] * 2)]
    else:
        geoms = [("4x4608", [4608] * 4), ("2x9728", [9728] * 2)]
    for name, segs in geoms:
      try:
        L, G = sum(segs), len(segs)
        print(f"\n  段几何 {name}（ΣL={L}）")
        freqs_packed = posemb(make_pos(segs))
        vec_g = torch.randn(1, G, 6 * F_DIM, device=dev, dtype=torch.bfloat16)
        vec_1 = vec_g[:, :1]
        counts = torch.tensor(segs, device=dev)
        mod_index = torch.repeat_interleave(torch.arange(G, device=dev), counts)
        mask = _make_mask(segs)

        blk = make_block()
        results = {}

        # dense：B=G 等长段（ARB 布局）
        if len(set(segs)) == 1:
            Ls = segs[0]
            freqs_d = posemb(make_pos([Ls]))
            xd = torch.randn(G, Ls, F_DIM, device=dev, dtype=torch.bfloat16,
                             requires_grad=True)
            vd = vec_g.reshape(G, 1, 6 * F_DIM)
            results["dense bf16 (ARB布局)"] = (
                bench(lambda: blk(xd, vd, freqs_d, None), warmup=3, iters=10),
                bench(lambda: blk(xd, vd, freqs_d, None).sum().backward(),
                      warmup=3, iters=10))

        xp = torch.randn(1, L, F_DIM, device=dev, dtype=torch.bfloat16,
                         requires_grad=True)
        results["packed bf16 广播调制"] = (
            bench(lambda: blk(xp, vec_1, freqs_packed, mask), warmup=3, iters=10),
            bench(lambda: blk(xp, vec_1, freqs_packed, mask).sum().backward(),
                  warmup=3, iters=10))
        results["packed bf16 mod_index"] = (
            bench(lambda: blk(xp, vec_g, freqs_packed, mask, mod_index=mod_index),
                  warmup=3, iters=10),
            bench(lambda: blk(xp, vec_g, freqs_packed, mask,
                              mod_index=mod_index).sum().backward(),
                  warmup=3, iters=10))

        if QuantLinear is not None:
            for rw in (True, False):
                try:
                    qblk = quantize_block(make_block(), rowwise=rw)
                    tag = "packed fp8  mod_index" + ("" if rw else " (tsr)")
                    results[tag] = (
                        bench(lambda: qblk(xp, vec_g, freqs_packed, mask,
                                           mod_index=mod_index), warmup=3, iters=10),
                        bench(lambda: qblk(xp, vec_g, freqs_packed, mask,
                                           mod_index=mod_index).sum().backward(),
                              warmup=3, iters=10))
                    break
                except Exception as ex:
                    print(f"    [!] fp8 block(rowwise={rw}) 失败: "
                          f"{type(ex).__name__}: {ex}")

        for k, (tf, tfb) in results.items():
            print(f"    {k:<26} fwd {tf:8.3f}ms (×28={tf * N_LAYERS / 1000:5.2f}s)   "
                  f"fwd+bwd {tfb:8.3f}ms (×28={tfb * N_LAYERS / 1000:5.2f}s)")
      except Exception as ex:
        print(f"    [!] 本段几何失败: {type(ex).__name__}: {ex}")


def section_fullmodel(dev):
    """全模型闭环：真 SingleStreamDiT × forward_packed_navit（训练同款入口）。

    第 4 段证明 block 数学正常（fp8 packed ×28 ≈ 3.1s），但训练实测
    navit_model_forward=7.3s —— 差一个 ~2.15× 乘数。本段在 block 与训练之间
    只剩的三层皮（checkpoint 包装 / LoRA 适配器 / txt-stack+first+last 胶水）
    上逐层加料，看乘数在哪一层出现。若本段全配置(e)仍 ≈3.5s，则乘数来自
    训练进程环境（分配器/CPU 争抢/时钟），不在模型代码。
    """
    import sys as _sys
    from pathlib import Path
    from types import SimpleNamespace
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    try:
        from models.krea2_modeling import (
            SingleStreamDiT, SingleMMDiTConfig, KREA2_LARGE_WIDE)
        from trainer.quant import quantize_base_model
        from trainer.lora import LoRAInjector
        from trainer.model_family import KREA2_DEFAULT_LORA_TARGETS
    except ImportError as ex:
        print(f"\n══ 5. 全模型闭环 [skip] 导入失败: {ex} ══")
        return

    import os
    if os.environ.get("DIAG_SMALL"):
        cfg = SingleMMDiTConfig(features=512, tdim=256, txtdim=256, heads=8,
                                kvheads=2, multiplier=4, layers=2, patch=2,
                                channels=16, txtheads=4, txtkvheads=4, txtlayers=2)
        vis, txts = [256] * 4, [32] * 4
        print("\n══ 5. 全模型闭环 [缩小冒烟版，数字无裁决意义] ══")
    else:
        cfg = KREA2_LARGE_WIDE
        vis, txts = [4096] * 4, [512] * 4
        print("\n══ 5. 全模型闭环（forward_packed_navit，G=4，随机权重）══")
    SN, SL, G = sum(vis), sum(txts), len(vis)

    tokens = torch.randn(1, SN, cfg.patch ** 2 * cfg.channels,
                         device=dev, dtype=torch.bfloat16)
    t_G = torch.rand(G, device=dev)
    cross = torch.randn(1, SL, cfg.txtlayers, cfg.txtdim,
                        device=dev, dtype=torch.bfloat16)
    grid = torch.zeros(1, 2, SN, device=dev)
    off = 0
    for s in vis:
        side = int(s ** 0.5)
        idx = torch.arange(s, device=dev)
        grid[0, 0, off:off + s] = (idx // side).float()
        grid[0, 1, off:off + s] = (idx % side).float()
        off += s

    def build(with_lora: bool):
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with torch.device(dev):
                m = SingleStreamDiT(cfg)
        finally:
            torch.set_default_dtype(old_dtype)
        for p in m.parameters():
            p.requires_grad_(False)
        if with_lora:
            inj = LoRAInjector(rank=32, alpha=32.0,
                               targets=list(KREA2_DEFAULT_LORA_TARGETS))
            inj.inject(m)
        return m

    def quant(m):
        ns = SimpleNamespace(base_quant="fp8", base_quant_gemm="auto",
                             base_quant_fp8_scale="auto", base_quant_fp8_grad=True,
                             base_quant_include=None, base_quant_skip=None)
        quantize_base_model(m, ns, family="krea2")
        return m

    def run(m, ckpt):
        return m.forward_packed_navit(tokens, t_G, cross, grid, vis, txts,
                                      use_checkpoint=ckpt)

    try:
        m = build(with_lora=False)
        print(f"    bf16          ckpt=off  fwd "
              f"{bench(lambda: run(m, False), warmup=1, iters=3) / 1000:6.2f}s")
        print(f"    bf16          ckpt=on   fwd "
              f"{bench(lambda: run(m, True), warmup=1, iters=3) / 1000:6.2f}s")
        quant(m)
        print(f"    fp8           ckpt=on   fwd "
              f"{bench(lambda: run(m, True), warmup=1, iters=3) / 1000:6.2f}s")
        del m
        torch.cuda.empty_cache()

        m = build(with_lora=True)
        print(f"    bf16+LoRA     ckpt=on   fwd "
              f"{bench(lambda: run(m, True), warmup=1, iters=3) / 1000:6.2f}s")
        quant(m)
        print(f"    fp8+LoRA      ckpt=on   fwd "
              f"{bench(lambda: run(m, True), warmup=1, iters=3) / 1000:6.2f}s"
              f"   ← 对标 stage_timing navit_model_forward≈7.3s")
        print(f"    fp8+LoRA      ckpt=on   fwd+bwd "
              f"{bench(lambda: run(m, True).float().pow(2).mean().backward(), warmup=1, iters=3) / 1000:6.2f}s"
              f"   ← 对标 forward+backward≈28.4s")
    except Exception as ex:
        import traceback
        traceback.print_exc()
        print(f"    [!] 全模型段中断: {type(ex).__name__}: {ex}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-attn", action="store_true")
    ap.add_argument("--skip-gemm", action="store_true")
    ap.add_argument("--skip-block", action="store_true")
    ap.add_argument("--skip-fullmodel", action="store_true")
    ap.add_argument("--small", action="store_true",
                    help="block 段用缩小几何（小显存本地冒烟）")
    args = ap.parse_args()
    if args.small:
        import os
        os.environ["DIAG_SMALL"] = "1"

    if not torch.cuda.is_available():
        sys.exit("需要 CUDA 设备")
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(dev)
    print(f"设备: {props.name} sm{props.major}{props.minor} "
          f"{props.total_memory / 2**30:.0f}GB | torch {torch.__version__}")

    if not args.skip_gemm:
        section_gemm(dev)
        section_backward(dev)
    if not args.skip_attn:
        section_attention(dev)
    if not args.skip_block:
        section_block(dev)
    if not args.skip_fullmodel:
        section_fullmodel(dev)


if __name__ == "__main__":
    main()
