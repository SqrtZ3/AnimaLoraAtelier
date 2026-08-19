#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LoKr 成品的强度体检：逐 checkpoint 打印 ‖w1‖ / ‖w2‖ / ‖ΔW‖ 与能量分布。

**为什么需要它。** LoKr 的增量是 ΔW = (alpha/r)·kron(w1, w2_a@w2_b)，总强度是
**乘积** ‖w1‖·‖w2‖。w2 在被优化器推着单调变大是正常的；真正决定成品强不强的是
w1（每模块 4×4 的总增益因子）有没有反向收缩去抵消它。一次训练是"健康地收敛"还是
"强度失控"，看这两条曲线的相对走向最直接——比 loss 曲线灵敏得多，也不需要底模。

实测过的两条参照曲线（同一套 C12 配方、不同数据集/平台）：

    健康（成品可用）：  ‖w1‖ 0.340 → 0.155（降），‖ΔW‖ 冲高到 41.9 后回落锁定 35.7
    失控（成品废掉）：  ‖w1‖ 0.376 → 0.428（升），‖ΔW‖ 单调膨胀到 113.2

用法（可传多个文件，也可传 glob；Windows 上 glob 由脚本自己展开）::

    python AnimaLoraToolkit/tools/lokr_strength_probe.py out/*.safetensors
    python AnimaLoraToolkit/tools/lokr_strength_probe.py a_epoch1.safetensors a_epoch2.safetensors

只读权重，不需要底模、不需要 GPU。范数用 ‖kron(A,B)‖_F = ‖A‖_F·‖B‖_F 免物化算，
所以对 8192×2048 这种大层也是秒级。

**口径提醒**：范数占比 ≠ 因果重要性；这里给的是"增量住在哪、有多大"，
最终好不好仍以 A/B 画面为准。
"""
from __future__ import annotations

import argparse
import glob as _glob
import os
import re
import sys
from collections import defaultdict

import torch
from safetensors.torch import load_file


def _group_of(module: str) -> str:
    if "cross_attn" in module:
        return "cross_attn"
    if "self_attn" in module:
        return "self_attn"
    return "mlp"


def _sort_key(path: str):
    """按 epoch / step 数字排序，取不到就退回文件名。"""
    m = re.search(r"(?:epoch|step)(\d+)", os.path.basename(path))
    return (0, int(m.group(1))) if m else (1, os.path.basename(path))


def probe(path: str) -> dict:
    sd = load_file(path)
    modules = sorted({k.split(".", 1)[0] for k in sd})
    if not modules:
        raise SystemExit(f"{path}: 没有找到任何模块")

    w1_norms: list[float] = []
    w2_norms: list[float] = []
    energy: defaultdict[str, float] = defaultdict(float)
    total_sq = 0.0
    nan_modules: list[str] = []

    for m in modules:
        try:
            w1 = sd[m + ".lokr_w1"].float()
            a = sd[m + ".lokr_w2_a"].float()
            b = sd[m + ".lokr_w2_b"].float()
        except KeyError:
            # 不是 LoKr 模块（例如混了标准 LoRA 的 up/down）——跳过并在末尾提示
            continue
        alpha = sd[m + ".alpha"].float().item() if (m + ".alpha") in sd else float(a.shape[1])
        w2 = a @ b
        if torch.isnan(w1).any() or torch.isnan(w2).any():
            nan_modules.append(m)
        w1n = w1.norm().item()
        w2n = w2.norm().item()
        w1_norms.append(w1n)
        w2_norms.append(w2n)
        fro = (alpha / a.shape[1]) * w1n * w2n
        total_sq += fro * fro
        energy[_group_of(m)] += fro * fro

    if not w1_norms:
        raise SystemExit(f"{path}: 没有 LoKr 模块（lokr_w1/lokr_w2_a/lokr_w2_b）")

    n = len(w1_norms)
    return {
        "modules": n,
        "w1": sum(w1_norms) / n,
        "w2": sum(w2_norms) / n,
        "dw": total_sq ** 0.5,
        "energy": {k: v / total_sq * 100.0 for k, v in energy.items()},
        "nan": nan_modules,
        "has_dora": any(k.endswith(".dora_scale") for k in sd),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LoKr 成品强度体检")
    ap.add_argument("paths", nargs="+", help="safetensors 路径，可含通配符")
    args = ap.parse_args(argv)

    files: list[str] = []
    for p in args.paths:
        hits = _glob.glob(p)
        files.extend(hits if hits else [p])
    files = sorted(dict.fromkeys(files), key=_sort_key)
    if not files:
        print("没有匹配到文件", file=sys.stderr)
        return 1

    print(f"{'checkpoint':<46} {'模块':>4} {'‖w1‖均':>8} {'‖w2‖均':>8} "
          f"{'‖ΔW‖_F':>9} {'cross':>7} {'self':>7} {'mlp':>7}")
    print("-" * 104)
    prev = None
    for f in files:
        r = probe(f)
        e = r["energy"]
        name = os.path.basename(f)
        if len(name) > 45:
            name = "…" + name[-44:]
        arrow = ""
        if prev is not None:
            arrow = " ↑" if r["w1"] > prev + 1e-6 else (" ↓" if r["w1"] < prev - 1e-6 else " →")
        prev = r["w1"]
        print(f"{name:<46} {r['modules']:>4} {r['w1']:>8.4f}{arrow:<2} {r['w2']:>8.4f} "
              f"{r['dw']:>9.3f} {e.get('cross_attn', 0):>6.1f}% {e.get('self_attn', 0):>6.1f}% "
              f"{e.get('mlp', 0):>6.1f}%")
        if r["nan"]:
            print(f"    ⚠ NaN 模块 {len(r['nan'])} 个，例：{r['nan'][:3]}")

    print()
    print("判读：‖w2‖ 单调增长是正常的；看 ‖w1‖ 的箭头——")
    print("  ↓ 持续下降 = 总增益在被负反馈压住，‖ΔW‖ 会冲高后回落锁定（健康）")
    print("  ↑ 持续上升 = w1 与 w2 同向，乘积膨胀，成品会强到压过文本条件（失控）")
    print("参照实测值：健康 0.340→0.155（‖ΔW‖ 锁 35.7）；失控 0.376→0.428（‖ΔW‖ 到 113.2）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
