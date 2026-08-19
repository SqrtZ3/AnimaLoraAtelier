# -*- coding: utf-8 -*-
"""stage_timing.csv 汇总分析（stdlib-only，云端 venv 无 numpy/scipy 保证）。

用法：
    python tools/analyze_stage_timing.py --csv <output_dir>/stage_timing.csv

CSV 由 anima_train.py 的 stage_timing_every>0 写出（见 trainer/stage_timer.py）。
按 mode（navit / arb / fit …）分组，对每个阶段列输出 count / mean / median / p90，
以及占 whole_step 的均值份额——用来回答"一步时间花在哪、两条路径差在哪个阶段"。

注意：
* optimizer 列只在恰为梯度累积边界的采样行有值，其余行为空 → 按非空行统计。
* whole_step 跨整个 micro-batch；各阶段之和 ≤ whole_step（有未打点的零碎）。
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

META_COLS = {"step", "mode", "grad_checkpoint", "G"}


def _percentile(sorted_vals, q):
    """线性插值分位数（与 numpy 默认一致）；sorted_vals 非空。"""
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def main() -> int:
    ap = argparse.ArgumentParser(description="汇总 stage_timing.csv（按 mode 分组）")
    ap.add_argument("--csv", required=True, help="stage_timing.csv 路径")
    ap.add_argument("--last", type=int, default=0,
                    help="只统计最后 N 个采样行（0=全部）；用于跳过 run 前期不稳态")
    args = ap.parse_args()

    path = Path(args.csv)
    if not path.is_file():
        print(f"找不到文件: {path}", file=sys.stderr)
        return 1

    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("CSV 为空（没有采样行）。确认 stage_timing_every>0 且训练已跑过 warmup。")
        return 0
    if args.last > 0:
        rows = rows[-args.last:]

    by_mode: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_mode[r.get("mode", "?")].append(r)

    stage_cols = [c for c in rows[0].keys() if c not in META_COLS]

    for mode, mrows in by_mode.items():
        gs = [int(r["G"]) for r in mrows if (r.get("G") or "").strip()]
        g_note = f" G(mean)={sum(gs) / len(gs):.1f}" if gs else ""
        print(f"\n== mode={mode}  采样行={len(mrows)}{g_note} ==")
        whole_means: dict[str, float] = {}
        stats = []
        for col in stage_cols:
            vals = sorted(
                float(r[col]) for r in mrows if (r.get(col) or "").strip()
            )
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            whole_means[col] = mean
            stats.append((col, len(vals), mean,
                          _percentile(vals, 0.5), _percentile(vals, 0.9)))
        whole = whole_means.get("whole_step_ms", 0.0)
        print(f"{'stage':28s} {'n':>4s} {'mean_ms':>10s} {'median':>10s} "
              f"{'p90':>10s} {'%whole':>7s}")
        for col, n, mean, med, p90 in stats:
            share = f"{mean / whole * 100:6.1f}%" if whole > 0 else "      -"
            print(f"{col:28s} {n:4d} {mean:10.1f} {med:10.1f} {p90:10.1f} {share:>7s}")
    print("\n提示：跨 run 对比时固定同一数据/配置，只切 navit_packing；"
          "forward 内嵌套探针(navit_*)之和 ≤ forward（含未打点零碎）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
