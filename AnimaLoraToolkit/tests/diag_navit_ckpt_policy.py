# -*- coding: utf-8 -*-
"""checkpoint 策略裁决：全 ckpt / skip_last / 选择性重算(SAC) 的「时间 × 峰值显存」前沿。

为什么要这个脚本
----------------
`diag_navit_speed.py` 的 S3 在 H20 上给出（真 SingleStreamBlock，N=55778，G=6）::

    fwd only            471.68 ms
    fwd+bwd 无 ckpt    1490.08 ms
    fwd+bwd 全 ckpt    1960.66 ms   ← 现状
    fwd+bwd SAC        1526.51 ms   ← 比现状快 1.28×，×28 块 = 整步的 22.6%

但那是**单块隔离**测的。SAC 的策略是"保存所有 mm/sdpa 输出"，而本 block 的显存大头正是
SwiGLU 的 16384 维中间量（N=32768 时每块约 4 GB）；28 块全保存约 110 GB，必 OOM。
所以 1.28× 能不能兑现，取决于**在 95 GB 里能用哪一档策略**——本脚本就测这条前沿。

测什么
------
在**真的 N_BLOCKS 层栈**上跑 fwd+bwd，逐策略报「耗时 + 峰值显存」：

  P0 全 ckpt（现状基线）
  P1 无 ckpt（速度上界，大概率 OOM，OOM 本身就是结论）
  P2 SAC-全mm      保存 mm/addmm/bmm/sdpa 输出（激进）
  P3 SAC-仅attn    只保存 sdpa 输出（重算便宜的 norm/silu/rope，MLP 中间量照旧重算）
  P4 SAC-窄mm      保存 sdpa + 输出宽度 ≤ --sac-width 的 mm（默认 6144，即放过 16384 维 MLP 中间量）
  P5 skip_last=K   前 N-K 块全 ckpt、末 K 块不 ckpt（已实现的旋钮，K 由 --skip-list 给）

基座权重**冻结**、只让输入带梯度 —— 贴近 LoRA 训练的显存形态（不含 LoRA 自身的
小梯度与优化器态，也不含 VAE/TE 常驻，所以真实训练的可用余量比这里更紧，读结论时留余量）。

用法（云端 repo 根目录）::

    ./venv/bin/python AnimaLoraToolkit/tests/diag_navit_ckpt_policy.py
    ./venv/bin/python AnimaLoraToolkit/tests/diag_navit_ckpt_policy.py --tokens 55778
    ./venv/bin/python AnimaLoraToolkit/tests/diag_navit_ckpt_policy.py --blocks 14 --tokens 16384

默认 --tokens 32768 --groups 6 --blocks 28。28 块 bf16 权重约 25.4 GB，先建栈再测；
任一策略 OOM 会打印 [OOM] 并继续（OOM 是有效结论，不是失败）。
"""

from __future__ import annotations

import argparse
import gc
import pathlib
import statistics
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from models.krea2_modeling import (  # noqa: E402
    KREA2_LARGE_WIDE,
    PositionalEncoding,
    SingleStreamBlock,
    _SegLens,
)

CSV_WHOLE_MS = 53_836.6          # krea2-C1 均值整步
CSV_FWD_BWD_MS = 57_541.7        # forward + backward
FEATURES = KREA2_LARGE_WIDE.features
HEADS = KREA2_LARGE_WIDE.heads
KVHEADS = KREA2_LARGE_WIDE.kvheads
HEADDIM = FEATURES // HEADS
DEV = "cuda"
DTYPE = torch.bfloat16


def free_mem():
    gc.collect()
    torch.cuda.empty_cache()


def make_freqs(n_tok: int):
    axes = [HEADDIM - 12 * (HEADDIM // 16), 6 * (HEADDIM // 16), 6 * (HEADDIM // 16)]
    pe = PositionalEncoding(FEATURES, axes, theta=KREA2_LARGE_WIDE.theta, ntk=1.0)
    side = int(n_tok ** 0.5) + 1
    rows = (torch.arange(n_tok, device=DEV) // side).float()
    cols = (torch.arange(n_tok, device=DEV) % side).float()
    pos = torch.stack([torch.zeros_like(rows), rows, cols], dim=-1).unsqueeze(0)
    return pe(pos)


def seg_split(n_tok: int, g: int):
    base = n_tok // g
    segs = [base] * g
    segs[-1] += n_tok - base * g
    return segs


# ── SAC 策略 ────────────────────────────────────────────────────────────────
def make_sac_ctx(mode: str, width_limit: int):
    """mode: 'all_mm' | 'attn_only' | 'narrow_mm'。返回 context_fn（给 checkpoint 用）。"""
    from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

    SDPA_OPS = {
        torch.ops.aten._scaled_dot_product_efficient_attention.default,
        torch.ops.aten._scaled_dot_product_flash_attention.default,
        torch.ops.aten._scaled_dot_product_cudnn_attention.default,
    }
    MM_OPS = {
        torch.ops.aten.mm.default,
        torch.ops.aten.addmm.default,
        torch.ops.aten.bmm.default,
    }

    def _out_width(op, args):
        """mm/addmm 的输出宽度（最后一维）——用来把 16384 维 MLP 中间量筛掉。"""
        try:
            if op is torch.ops.aten.addmm.default:
                return int(args[2].shape[-1])
            return int(args[1].shape[-1])
        except Exception:  # noqa: BLE001
            return 1 << 30

    def policy(ctx, op, *args, **kwargs):
        if op in SDPA_OPS:
            return CheckpointPolicy.MUST_SAVE
        if op in MM_OPS:
            if mode == "all_mm":
                return CheckpointPolicy.MUST_SAVE
            if mode == "narrow_mm" and _out_width(op, args) <= width_limit:
                return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    return lambda: create_selective_checkpoint_contexts(policy)


# ── 一次 fwd+bwd ────────────────────────────────────────────────────────────
def run_stack(blocks, x, vec, freqs, mask, mod_index, mode: str,
              sac_ctx=None, skip_last: int = 0):
    """mode: 'full' | 'none' | 'sac' | 'skip_last'。返回标量 loss（已 backward）。"""
    from torch.utils.checkpoint import checkpoint

    n = len(blocks)
    h = x
    for i, blk in enumerate(blocks):
        def _run(inp, _b=blk):
            return _b(inp, vec, freqs, mask, mod_index=mod_index)

        if mode == "none":
            h = _run(h)
        elif mode == "full":
            h = checkpoint(_run, h, use_reentrant=False)
        elif mode == "sac":
            h = checkpoint(_run, h, use_reentrant=False, context_fn=sac_ctx)
        elif mode == "skip_last":
            h = (_run(h) if i >= n - skip_last
                 else checkpoint(_run, h, use_reentrant=False))
        else:
            raise ValueError(mode)
    loss = h.float().pow(2).mean()
    loss.backward()
    return loss


def measure(label, blocks, x, vec, freqs, mask, mod_index, **kw):
    """返回 (中位耗时 ms, 峰值显存 GiB) 或 (None, None)。"""
    def _once():
        x.grad = None
        run_stack(blocks, x, vec, freqs, mask, mod_index, **kw)

    try:
        _once()                      # warmup（也把分配器的坑先踩了）
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        ts = []
        for _ in range(3):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _once()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
        peak = torch.cuda.max_memory_allocated() / 2 ** 30
        return statistics.median(ts), peak
    except torch.OutOfMemoryError:
        print(f"    [OOM] {label} —— 该策略在本卡放不下（这本身就是结论）")
        free_mem()
        return None, None
    except Exception as e:  # noqa: BLE001
        print(f"    [skip] {label}: {type(e).__name__}: {str(e)[:110]}")
        free_mem()
        return None, None


def main():
    ap = argparse.ArgumentParser(description="navit checkpoint 策略的时间×显存前沿")
    ap.add_argument("--tokens", type=int, default=32768)
    ap.add_argument("--groups", type=int, default=6)
    ap.add_argument("--blocks", type=int, default=KREA2_LARGE_WIDE.layers)
    ap.add_argument("--sac-width", type=int, default=FEATURES,
                    help="SAC-窄mm 保存的最大输出宽度（默认 6144，即放过 16384 维 MLP 中间量）")
    ap.add_argument("--skip-list", type=str, default="4,8,12",
                    help="要测的 grad_checkpoint_skip_last 值")
    ap.add_argument("--g-sweep", type=str, default="",
                    help="用 token 预算换 checkpoint 策略：给一串 G（如 '1,2,3,6'），"
                         "**固定每图 seqlen**（--seg-tokens）只改每包图数 → N=G×seg。"
                         "因为块对角 attention 让 per-token 代价只取决于每图 seqlen、"
                         "与预算无关（S5 实测 seg 固定时 ms/token 恒定），所以降预算不损"
                         "吞吐、却可能换来更便宜的 checkpoint 策略。判据是 ms/token 而非 ms/step。")
    ap.add_argument("--seg-tokens", type=int, default=0,
                    help="--g-sweep 用的每图 token 数；0=取 --tokens/--groups。")
    ap.add_argument("--vram-budget", type=float, default=0.0,
                    help="可用显存上限 GiB（超过即视为不可选）；0=自动取总显存的 80%%，"
                         "给 LoRA 梯度/优化器态/VAE/TE/碎片留余量。")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("需要 CUDA。")
        return 1
    torch.manual_seed(0)
    cap = torch.cuda.get_device_capability(0)
    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    print("=" * 78)
    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}  sm{cap[0]}{cap[1]}  "
          f"{total:.0f} GiB")
    print(f"栈：{args.blocks} × SingleStreamBlock({FEATURES}ch, {HEADS}/{KVHEADS} heads)  "
          f"N={args.tokens}  G={args.groups}")
    print(f"CSV 基准：整步 {CSV_WHOLE_MS / 1000:.1f}s（forward+backward {CSV_FWD_BWD_MS / 1000:.1f}s）")
    print("=" * 78)

    print("建栈中（权重冻结，只有输入带梯度 —— 贴近 LoRA 训练的显存形态）…")
    blocks = []
    for _ in range(args.blocks):
        b = SingleStreamBlock(FEATURES, HEADS, KREA2_LARGE_WIDE.multiplier,
                              KREA2_LARGE_WIDE.bias, KVHEADS).to(DEV, DTYPE)
        for p in b.parameters():
            p.requires_grad_(False)
        blocks.append(b)
    w_gib = sum(p.numel() * p.element_size() for b in blocks for p in b.parameters()) / 2 ** 30
    print(f"  权重常驻 {w_gib:.1f} GiB\n")

    vram_cap = float(args.vram_budget) if args.vram_budget > 0 else total * 0.80

    def build_inputs(n_tok: int, g: int):
        segs = seg_split(n_tok, g)
        x = torch.randn(1, n_tok, FEATURES, device=DEV, dtype=DTYPE, requires_grad=True)
        vec = torch.randn(1, g, FEATURES * 6, device=DEV, dtype=DTYPE)
        mod_index = torch.repeat_interleave(
            torch.arange(g, device=DEV), torch.tensor(segs, device=DEV))
        return dict(blocks=blocks, x=x, vec=vec, freqs=make_freqs(n_tok),
                    mask=_SegLens(segs), mod_index=mod_index)

    common = build_inputs(args.tokens, args.groups)

    cases = [
        ("P0 全ckpt(现状)", dict(mode="full")),
        ("P1 无ckpt(上界)", dict(mode="none")),
        ("P2 SAC-全mm", dict(mode="sac", sac_ctx=make_sac_ctx("all_mm", args.sac_width))),
        ("P3 SAC-仅attn", dict(mode="sac", sac_ctx=make_sac_ctx("attn_only", args.sac_width))),
        (f"P4 SAC-窄mm(≤{args.sac_width})",
         dict(mode="sac", sac_ctx=make_sac_ctx("narrow_mm", args.sac_width))),
    ]
    for k in (int(v) for v in args.skip_list.split(",") if v.strip()):
        if 0 < k < args.blocks:
            cases.append((f"P5 skip_last={k}", dict(mode="skip_last", skip_last=k)))

    print(f"{'策略':<22}{'耗时':>11}{'峰值显存':>12}{'vs现状':>9}{'省整步':>10}")
    print("-" * 78)
    results = {}
    for label, kw in cases:
        free_mem()
        torch.cuda.reset_peak_memory_stats()
        ms, peak = measure(label, **common, **kw)
        results[label] = (ms, peak)
        if ms is None:
            print(f"{label:<22}{'n/a':>11}{'n/a':>12}{'-':>9}{'-':>10}")
            continue
        base = results.get("P0 全ckpt(现状)", (None, None))[0]
        sp = f"{base / ms:.2f}×" if base else "-"
        # 本栈耗时 → 整步：栈已是 28 块的真实耗时，直接按 CSV 的 fwd+bwd 段折算
        save = ((base - ms) / CSV_WHOLE_MS * 100.0) if base else 0.0
        scale = args.blocks / KREA2_LARGE_WIDE.layers
        save = save / scale if scale else save
        print(f"{label:<22}{ms:9.1f}ms{peak:10.1f}GiB{sp:>9}{save:9.1f}%")

    # ── 用 token 预算换 checkpoint 策略（--g-sweep）────────────────────────────
    if args.g_sweep.strip():
        seg = int(args.seg_tokens or (args.tokens // max(1, args.groups)))
        g_list = [int(v) for v in args.g_sweep.split(",") if v.strip()]
        print()
        print("=" * 78)
        print(f"用预算换策略：固定每图 seqlen={seg}（per-token 代价只取决于它，与预算无关），")
        print(f"只改每包图数 G → N=G×seg。判据是 **ms/token**，显存上限取 {vram_cap:.0f} GiB。")
        print("=" * 78)
        print(f"{'G':>3}{'N':>8}  {'策略':<22}{'耗时':>10}{'ms/token':>11}{'峰值':>9}{'可用':>6}")
        print("-" * 78)
        best = []
        for g in g_list:
            n_tok = g * seg
            inputs = build_inputs(n_tok, g)
            for label, kw in cases:
                free_mem()
                torch.cuda.reset_peak_memory_stats()
                ms, peak = measure(f"G={g} {label}", **inputs, **kw)
                if ms is None:
                    continue
                per_tok = ms / n_tok
                ok = peak <= vram_cap
                print(f"{g:>3}{n_tok:>8}  {label:<22}{ms:8.0f}ms{per_tok:10.4f}{peak:8.1f}G"
                      f"{'  ✓' if ok else '  ✗':>6}")
                if ok:
                    best.append((per_tok, g, n_tok, label, ms, peak))
            inputs = None
            free_mem()
        print("-" * 78)
        if best:
            best.sort()
            b0 = best[0]
            print(f"最优（显存内）：G={b0[1]} N={b0[2]} {b0[3]} → {b0[0]:.4f} ms/token，峰值 {b0[5]:.1f} GiB")
            cur = [r for r in best if r[3].startswith("P0")]
            if cur:
                base_pt = min(r[0] for r in cur)
                print(f"对比现状（P0 全ckpt）最优 {base_pt:.4f} ms/token → "
                      f"吞吐提升 {(base_pt / b0[0] - 1) * 100:.1f}%")
                print("  换算：同样的图/秒下整步时间按此比例下降；步数变多由 grad_accum 补回"
                      "有效 batch（optimizer 只占步时 0.19%，多出的步开销可忽略）。")
        else:
            print("没有任何组合落在显存预算内 —— 提高 --vram-budget 或降 --seg-tokens 复测。")

    print("-" * 78)
    print("读法：")
    print("  * 「省整步%」= (基线−本策略) ÷ CSV 整步 53.8s，已按 --blocks/28 归一到满栈口径。")
    print("  * 峰值显存**不含** LoRA 梯度/优化器态、VAE、文本编码器与分配器碎片；")
    print("    真实训练要在此基础上留 10~20 GiB 余量再选策略。")
    print("  * 选择依据：在你能接受的峰值显存下，挑耗时最小的那一行；")
    print("    若 P4/P3 能在预算内跑到接近 P1，就是最优解（P1 是速度上界）。")
    print("  * 若某策略 OOM，可降 --tokens 复测，但**最终要按真实 navit_token_budget 决定**。")
    print("  * ⚠ 小 --tokens（几千以下）下 SAC 会比全 ckpt **慢** —— 逐 op 策略判定的固定开销")
    print("    盖过了省下的重算。要在真实 token 规模（≥16384）上读结论。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
