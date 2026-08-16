# -*- coding: utf-8 -*-
"""逐图 AdaLN 的 gather 物化 vs 融合：值不值得做（本地免费取证）。

**测的是什么。** navit packed 路径下 `SingleStreamBlock.forward`
（models/krea2_modeling.py:672-678）拿到 per-image 的 6 组调制行 (1,G,D) 后，用
`index_select(...).unsqueeze(0)` **把 6 个 (1,ΣN,D) 张量全部物化**，再喂给逐元素的
`(1+scale)*RMSNorm(x)+shift` 与 `x + gate*(...)`。这些张量每行都是同一图的常数，
信息量只有 G×D，却按 ΣN×D 落了盘：Krea2 D=6144、ΣN=55778 时是 **6×55778×6144×2B
≈ 4.1 GB**（外加 RMSNorm 内部的 fp32 中间量 1.4 GB）。

AdaptiveLoad（arXiv 2605.17923）对同一位置做了 fused LayerNorm-Modulate CUDA kernel，
报该算子前向 3.21–3.39×、激活显存 −61.9%（8k–64k seqlen）。本脚本先用**不写 CUDA 的
版本**（torch.compile 让 inductor 把 gather 折进 pointwise kernel）量一下天花板，
决定要不要往下做。

**三个变体**（数学恒等，脚本会逐元素对拍）：
  A `eager_materialize` —— 现状：6 个 index_select 全物化后再算。
  B `eager_inline`      —— 不预先物化全部，表达式里就地 gather（少几个同时存活的大张量）。
  C `compiled_fused`    —— torch.compile(dynamic=True) 融合 RMSNorm+gather+仿射 / gate+残差。

**诚实标注：**
  * 这里只测被改动的算子链，**不含** attention/SwiGLU/LoRA，所以给出的是**该链自身**的
    倍数，不是整步收益。整步占比要用云端 stage_timing 才能定；脚本会按"每 block 调用
    2 次 × 28 block"把绝对毫秒外推出来供参考，但那是上界口径，别当整步预测。
  * 本地卡（8GiB 级）与云端 H20/RTX PRO 6000 的带宽/算力比不同，**倍数会移动**；
    本地结论只用于判断"是否值得在云端量"，不作为收益承诺。

用法::

    python AnimaLoraToolkit/tests/diag_navit_adaln_gather.py
    python AnimaLoraToolkit/tests/diag_navit_adaln_gather.py --tokens 4096,8192 --groups 4
    python AnimaLoraToolkit/tests/diag_navit_adaln_gather.py --features 3072  # 显存不够时
"""

from __future__ import annotations

import argparse
import gc
import statistics
import sys

import torch
import torch.nn.functional as F

try:  # Windows 控制台默认 GBK，报告里的 ≈/× 会 UnicodeEncodeError
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

# Krea2 KREA2_LARGE_WIDE 口径（见 models/krea2_modeling.py）
DEFAULT_FEATURES = 6144
N_BLOCKS = 28
CALLS_PER_BLOCK = 2      # 每个 SingleStreamBlock 有 pre / post 两条调制链


# ── 变体实现 ────────────────────────────────────────────────────────────────
def _rmsnorm(x, w, eps):
    """与 models/krea2_modeling.py 的 RMSNorm 同口径：fp32 计算后回原 dtype。"""
    t = x.float()
    t = F.rms_norm(t, (x.shape[-1],), eps=eps, weight=w)
    return t.to(x.dtype)


def variant_eager_materialize(x, mods, w, eps, mod_index, y):
    """A：现状。6 组调制行先各自 index_select 成 (1,ΣN,D)，再逐元素用。"""
    scale, shift, gate, pscale, pshift, pgate = (
        m[0].index_select(0, mod_index).unsqueeze(0) for m in mods
    )
    h = (1 + scale) * _rmsnorm(x, w, eps) + shift
    x = x + gate * h
    h2 = (1 + pscale) * _rmsnorm(x, w, eps) + pshift
    return x + pgate * (h2 + y)


def variant_eager_inline(x, mods, w, eps, mod_index, y):
    """B：不预先物化 6 个，表达式里就地 gather（同时存活的大张量更少）。"""
    h = (1 + mods[0][0].index_select(0, mod_index)) * _rmsnorm(x, w, eps) \
        + mods[1][0].index_select(0, mod_index)
    x = x + mods[2][0].index_select(0, mod_index) * h
    h2 = (1 + mods[3][0].index_select(0, mod_index)) * _rmsnorm(x, w, eps) \
        + mods[4][0].index_select(0, mod_index)
    return x + mods[5][0].index_select(0, mod_index) * (h2 + y)


def _norm_mod(x, w, eps, scale_g, shift_g, mod_index):
    t = x.float()
    t = F.rms_norm(t, (x.shape[-1],), eps=eps, weight=w)
    t = t.to(x.dtype)
    return (1 + scale_g.index_select(0, mod_index)) * t + shift_g.index_select(0, mod_index)


def _gate_add(x, gate_g, mod_index, h):
    return x + gate_g.index_select(0, mod_index) * h


_norm_mod_c = torch.compile(_norm_mod, dynamic=True)
_gate_add_c = torch.compile(_gate_add, dynamic=True)


def variant_compiled_fused(x, mods, w, eps, mod_index, y):
    """C：inductor 把 RMSNorm + gather + 仿射融进一个 kernel（理想情况下零物化）。"""
    h = _norm_mod_c(x, w, eps, mods[0][0], mods[1][0], mod_index)
    x = _gate_add_c(x, mods[2][0], mod_index, h)
    h2 = _norm_mod_c(x, w, eps, mods[3][0], mods[4][0], mod_index)
    return _gate_add_c(x, mods[5][0], mod_index, h2 + y)


VARIANTS = {
    "A eager_materialize": variant_eager_materialize,
    "B eager_inline": variant_eager_inline,
    "C compiled_fused": variant_compiled_fused,
}


# ── 计时 ────────────────────────────────────────────────────────────────────
def _bench(fn, args, backward, iters, warmup):
    for _ in range(warmup):
        out = fn(*args)
        if backward:
            out.float().pow(2).mean().backward()
            for a in args:
                if torch.is_tensor(a) and a.grad is not None:
                    a.grad = None
                elif isinstance(a, (list, tuple)):
                    for t in a:
                        if torch.is_tensor(t) and t.grad is not None:
                            t.grad = None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        out = fn(*args)
        if backward:
            out.float().pow(2).mean().backward()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
        if backward:
            for a in args:
                if torch.is_tensor(a) and a.grad is not None:
                    a.grad = None
                elif isinstance(a, (list, tuple)):
                    for t in a:
                        if torch.is_tensor(t) and t.grad is not None:
                            t.grad = None
    return statistics.median(times), torch.cuda.max_memory_allocated() / 2**20


def _make_inputs(S, G, D, dev, dtype, requires_grad):
    g = torch.Generator(device=dev).manual_seed(0)
    x = torch.randn(1, S, D, device=dev, dtype=dtype, generator=g,
                    requires_grad=requires_grad)
    y = torch.randn(1, S, D, device=dev, dtype=dtype, generator=g)
    mods = [torch.randn(1, G, D, device=dev, dtype=dtype, generator=g,
                        requires_grad=requires_grad) * 0.1 for _ in range(6)]
    mods = [m.detach().requires_grad_(requires_grad) for m in mods]
    w = torch.randn(D, device=dev, dtype=torch.float32, generator=g) * 0.1 + 1.0
    w.requires_grad_(requires_grad)
    # 每图 token 数尽量均分（真实包也大致如此）
    base, rem = S // G, S % G
    seg = [base + (1 if i < rem else 0) for i in range(G)]
    mod_index = torch.repeat_interleave(
        torch.arange(G, device=dev), torch.tensor(seg, device=dev)
    ).to(torch.int64)
    return x, mods, w, 1e-5, mod_index, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="4096,8192",
                    help="逗号分隔的 ΣN 列表")
    ap.add_argument("--groups", type=int, default=4, help="包内图数 G")
    ap.add_argument("--features", type=int, default=DEFAULT_FEATURES)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=8)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("需要 CUDA。")
        return 1
    dev, dtype = torch.device("cuda"), torch.bfloat16
    D, G = args.features, args.groups
    print(f"device={torch.cuda.get_device_name(0)}  torch={torch.__version__}")
    print(f"D(features)={D}  G(图/包)={G}  dtype={dtype}  "
          f"iters={args.iters}(warmup {args.warmup})")
    print(f"外推口径：每 block {CALLS_PER_BLOCK} 条调制链 × {N_BLOCKS} block\n")

    # ── 数值等价性 ──
    # 注意口径：融合会改变 bf16 的舍入路径，直接拿 A 当参考会把「舍入不同」误读成
    # 「算错」。判据用**全 fp32 参考**：各变体离真值多远，谁都不该超过 bf16 eps
    # (2^-8 ≈ 3.9e-03) 的量级，且融合版不应系统性更差。
    def _ref_fp32(x, mods, w, eps, mod_index, y):
        x, y = x.float(), y.float()
        m = [t.float() for t in mods]
        g = lambda t: t[0].index_select(0, mod_index)                      # noqa: E731
        n = lambda v: F.rms_norm(v, (D,), eps=eps, weight=w.float())       # noqa: E731
        h = (1 + g(m[0])) * n(x) + g(m[1])
        x = x + g(m[2]) * h
        h2 = (1 + g(m[3])) * n(x) + g(m[4])
        return x + g(m[5]) * (h2 + y)

    ref_args = _make_inputs(1024, G, D, dev, dtype, requires_grad=False)
    with torch.no_grad():
        ref = _ref_fp32(*ref_args)
        scale = ref.abs().max().item()
        print(f"  数值口径：全 fp32 参考，|ref|max={scale:.3f}，"
              f"bf16 eps ≈ {2**-8:.2e}（相对）")
        for name, fn in VARIANTS.items():
            err = (fn(*ref_args).float() - ref).abs().max().item()
            flag = "OK" if err / scale < 4 * 2**-8 else "⚠ 超出 bf16 噪声，需查"
            print(f"  vs fp32 参考 {name:22s} max_abs={err:.3e}  "
                  f"rel={err / scale:.2e}  {flag}")
    print()

    tok_list = [int(t) for t in args.tokens.split(",") if t.strip()]
    for S in tok_list:
        print(f"── ΣN = {S} ───────────────────────────────────────────────")
        for backward in (False, True):
            tag = "fwd+bwd" if backward else "fwd    "
            base_ms = None
            for name, fn in VARIANTS.items():
                gc.collect()
                torch.cuda.empty_cache()
                try:
                    a = _make_inputs(S, G, D, dev, dtype, requires_grad=backward)
                    ms, mem = _bench(fn, a, backward, args.iters, args.warmup)
                except torch.cuda.OutOfMemoryError:
                    print(f"  {tag}  {name:22s}  OOM（跳过）")
                    continue
                if base_ms is None:
                    base_ms = ms
                    rel = "基准"
                else:
                    rel = f"{base_ms / ms:.2f}× vs A"
                ext = ms * CALLS_PER_BLOCK * N_BLOCKS / 2.0  # 每变体已含 2 条链
                print(f"  {tag}  {name:22s}  {ms:8.3f} ms  峰值 {mem:7.0f} MiB  "
                      f"{rel:>12s}   ×28block ≈ {ext:8.1f} ms")
            print()
    print("提醒：以上是被改动算子链**自身**的倍数，不是整步收益；整步占比需云端 "
          "stage_timing。本地卡与云端卡带宽比不同，倍数会移动。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
