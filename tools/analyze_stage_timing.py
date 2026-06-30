"""分析 stage_timing.csv：定位 NaViT vs ARB 的速度瓶颈。

读 anima_train.py 开启 ``--stage-timing-every`` 后写出的 ``stage_timing.csv``，按 mode
(navit/arb/fit) 分组，算各阶段 median/p90，并输出 NaViT-vs-ARB 逐阶段差值表，回答
"时间到底花在哪一段"。仅用标准库（csv/statistics），不依赖训练栈，本地直接跑。

用法:
  python tools/analyze_stage_timing.py --csv <output_dir>/stage_timing.csv
  # 多文件对比（如 navit 与 arb 分别跑的两次）：
  python tools/analyze_stage_timing.py --csv a/stage_timing.csv b/stage_timing.csv --label navit arb
"""
import argparse
import csv
import statistics
import sys
from pathlib import Path

# 与 anima_train.py _stage_timing_header 对齐的阶段列（ms），step/mode/grad_checkpoint/G 除外。
STAGE_COLS = [
    "data_fetch_ms", "text_encode_ms", "timestep_ms", "forward_ms",
    "navit_noise_patchify_ms", "navit_model_forward_ms", "navit_loss_loop_ms",
    "loss_assembly_ms", "aux_ms", "adaptive_ms", "backward_ms",
    "optimizer_ms", "whole_step_ms",
]


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"[analyze_stage_timing] CSV 不存在: {path}")
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _stats(values: list[float]) -> tuple[float, float, int]:
    """返回 (median, p90, count)，空列表返回 (nan, nan, 0)。"""
    xs = [v for v in values if v is not None and v == v]  # 过滤 None/NaN
    if not xs:
        return float("nan"), float("nan"), 0
    xs_sorted = sorted(xs)
    med = statistics.median(xs_sorted)
    # p90 线性插值（无 numpy）
    if len(xs_sorted) == 1:
        p90 = xs_sorted[0]
    else:
        pos = 0.9 * (len(xs_sorted) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(xs_sorted) - 1)
        p90 = xs_sorted[lo] + (xs_sorted[hi] - xs_sorted[lo]) * (pos - lo)
    return med, p90, len(xs_sorted)


def _collect(rows: list[dict], mode: str) -> dict[str, list[float]]:
    """从 rows 里筛 mode，收集每个阶段列的 float 值列表。"""
    out: dict[str, list[float]] = {c: [] for c in STAGE_COLS}
    for r in rows:
        if r.get("mode") != mode:
            continue
        for c in STAGE_COLS:
            raw = r.get(c)
            if raw is None or raw == "":
                continue
            try:
                out[c].append(float(raw))
            except ValueError:
                pass
    return out


def _fmt(v: float) -> str:
    return "  -  " if v != v else f"{v:7.2f}"


def report(rows: list[dict], label: str) -> None:
    modes = sorted({r.get("mode", "?") for r in rows})
    print(f"\n{'=' * 78}\n# {label}  (modes={modes}, rows={len(rows)})\n{'=' * 78}")
    for mode in modes:
        data = _collect(rows, mode)
        print(f"\n## mode = {mode}")
        print(f"  {'stage':<26} {'median_ms':>10} {'p90_ms':>10} {'count':>6}")
        print(f"  {'-' * 26} {'-' * 10} {'-' * 10} {'-' * 6}")
        # 按 median 降序，把大头排前面
        ranked = sorted(STAGE_COLS, key=lambda c: _stats(data[c])[0] if _stats(data[c])[0] == _stats(data[c])[0] else -1, reverse=True)
        for c in ranked:
            med, p90, n = _stats(data[c])
            if n == 0:
                continue
            print(f"  {c:<26} {_fmt(med):>10} {_fmt(p90):>10} {n:>6}")
        # 占比（相对 whole_step median）
        ws_med = _stats(data["whole_step_ms"])[0]
        if ws_med == ws_med and ws_med > 0:
            print(f"\n  占比（相对 whole_step median={ws_med:.2f}ms）:")
            for c in STAGE_COLS:
                if c == "whole_step_ms":
                    continue
                med, _, n = _stats(data[c])
                if n == 0 or med != med:
                    continue
                print(f"    {c:<24} {med / ws_med * 100:5.1f}%")


def compare(rows_a: list[dict], label_a: str, rows_b: list[dict], label_b: str) -> None:
    """逐阶段对比两个 CSV（同 mode 对同 mode）。"""
    modes = sorted({r.get("mode", "?") for r in rows_a} | {r.get("mode", "?") for r in rows_b})
    print(f"\n{'=' * 78}\n# 逐阶段对比: {label_a}  vs  {label_b}\n{'=' * 78}")
    for mode in modes:
        da = _collect(rows_a, mode)
        db = _collect(rows_b, mode)
        if all(_stats(da[c])[2] == 0 for c in STAGE_COLS):
            continue
        if all(_stats(db[c])[2] == 0 for c in STAGE_COLS):
            continue
        print(f"\n## mode = {mode}")
        print(f"  {'stage':<26} {label_a:>12} {label_b:>12} {'diff_ms':>10} {'ratio':>8}")
        print(f"  {'-' * 26} {'-' * 12} {'-' * 12} {'-' * 10} {'-' * 8}")
        for c in STAGE_COLS:
            ma, _, na = _stats(da[c])
            mb, _, nb = _stats(db[c])
            if na == 0 and nb == 0:
                continue
            diff = (mb - ma) if (ma == ma and mb == mb) else float("nan")
            ratio = (mb / ma) if (ma == ma and ma > 0 and mb == mb) else float("nan")
            rstr = "  -  " if ratio != ratio else f"{ratio:7.2f}x"
            print(f"  {c:<26} {_fmt(ma):>12} {_fmt(mb):>12} {_fmt(diff):>10} {rstr:>8}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", nargs="+", required=True, help="一个或多个 stage_timing.csv 路径")
    ap.add_argument("--label", nargs="+", help="与 --csv 一一对应的标签（默认文件名）")
    args = ap.parse_args()

    paths = [Path(p) for p in args.csv]
    labels = args.label if args.label else [p.parent.name for p in paths]
    if len(labels) != len(paths):
        sys.exit("--label 数量需与 --csv 一致")

    all_rows = []
    for p, lab in zip(paths, labels):
        rows = _read_rows(p)
        report(rows, lab)
        all_rows.append((rows, lab))

    if len(all_rows) == 2:
        compare(all_rows[0][0], all_rows[0][1], all_rows[1][0], all_rows[1][1])
    elif len(all_rows) > 2:
        print(f"\n[提示] 多于 2 个 CSV，仅分别报告；逐对对比请两两传入。")


if __name__ == "__main__":
    main()
