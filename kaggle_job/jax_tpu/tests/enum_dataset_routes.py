"""用**真实数据集**的图像尺寸，对比两条布局路线的调度代价（零 TPU 配额）。

打包路线（NaViT）：token 数量化到 Q -> FFD 装箱到 budget -> 同**段长元组**的
                   8 个 pack 成一步。
分桶路线（ragged）：token 数量化到 Q -> 同**桶长 L** 的 8*G 张图成一步。

回答三件事（**两条路线用同一组 Q、同一组口径**）：
  1. 有效填充率（= 线性层算力利用率，填充 token 一样要过 MLP）；
  2. 编译身份的种类数（打包是段长元组，分桶是一个整数 L）；
  3. **入步图率** —— 一轮里多少样本能凑成完整的 8 卡步，多少要顺延。

## 两条必须照做的口径（旧版本这里都错过）

**一、Q 要对两条路线都扫。** 旧版本给分桶扫 Q=128/512/1024、给打包锁死 Q=1024，
于是"打包填充率更低"这个结论有一部分是口径造出来的。真实数据集上 Q 对打包的
填充率可能完全没影响（your-dataset 实测：budget=32768 时 Q=128 与 Q=1024
的填充率都是 96.9%，只有编译身份数从 14 降到 10），所以必须扫了才知道。

**二、样本池必须含 multiscale 副本。** `navit_multiscale` 是**追加**等比缩放副本
（trainer/data.py:495），不是替换。your-dataset 实测 85 张原生图 + 80 条
4096 档副本 = 165 条，副本占了一半 —— 只读目录里的原生尺寸会评估出一个训练中
根本不存在的分布。本脚本用 `--ladder` 复刻 `plan_multiscale_copy`
（trainer/data.py:136）的算法：等比缩放、16px 对齐、floor、不上采样、
不产出与原生同档的副本。

## 谁划算取决于数据集，不能一般化

  * 打包的编译身份是段长**元组**（等价类细得多），要凑 8 个同元组的 pack；
    分桶 G=1 只要 8 张同 L 的图 —— 这一项分桶占便宜；
  * 打包能把不同大小的图混进同一个 pack，分桶一步只有一个 L —— 这一项打包占便宜。

token 数越集中，分桶越占便宜；越分散、或单图大到一个桶凑不齐 8 张，打包的混装
能力越值钱。**别拿手边某个目录的数字外推到别的数据集**（memory
`[[dont-infer-dataset-provenance]]`）。看 `--dump` 打出的布局组成往往比看汇总
数字更能解释结果：分布是双峰时 FFD 会产出少数几种 100% 填充的主布局 + 一条
只出现一次的长尾，长尾就是顺延的来源。

另需分清两件常被混为一谈的事：
  * **任意宽高比**：两条路线都原生支持（RoPE 逐 token 查 rows/cols，不依赖矩形
    网格，同一步里各图可以有各自的 (h, w)）；
  * **任意 token 数混装**：这才是序列打包独有的能力，分桶做不到（一步一个 L）。

token 数 = (H/16) * (W/16)：VAE 下采样 8 倍、patch 2x2。

跑法（需要 PIL，用 torch 那个解释器）：
    <torch-python> enum_dataset_routes.py <数据集目录> [更多目录...]
    <torch-python> enum_dataset_routes.py <目录> --ladder 4096 --dump 32768
"""

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

DEVICES = 8
EXT = (".png", ".jpg", ".jpeg", ".webp")
ALIGN = 16                    # patch 2 * vae 8
QUANTA = (128, 256, 512, 1024)
BUDGETS = (16384, 32768, 49152)


def image_sizes(root: Path):
    from PIL import Image
    out = []
    for p in sorted(root.iterdir()):
        if p.suffix.lower() not in EXT:
            continue
        try:
            w, h = Image.open(p).size
        except Exception:
            continue
        out.append((int(w), int(h)))
    return out


def multiscale_tokens(w: int, h: int, target: int, align: int = ALIGN):
    """复刻 trainer/data.py:136 `plan_multiscale_copy` 的 token 数。

    返回 None 表示不产出副本（源图不严格大于目标档 —— 阶梯从不上采样，也不产出
    与原生同档的副本）。
    """
    src = (w // align) * (h // align)
    if src <= target:
        return None
    s = math.sqrt(target * align * align / float(w * h))
    tw, th = max(1, int(w * s) // align), max(1, int(h * s) // align)
    if tw * th > target:                      # 极端长宽比下某轴被 max(1,·) 顶起
        if tw >= th:
            tw = max(1, target // th)
        else:
            th = max(1, target // tw)
    return tw * th


def build_pool(sizes, ladder):
    """原生 token + 各档 multiscale 副本。返回 (pool, native, 每档副本数)。"""
    native = [(w // ALIGN) * (h // ALIGN) for w, h in sizes]
    pool, per_rung = list(native), {}
    for tgt in ladder:
        cp = [n for w, h in sizes if (n := multiscale_tokens(w, h, tgt)) is not None]
        per_rung[tgt] = cp
        pool += cp
    return pool, native, per_rung


def quantize(n: int, q: int) -> int:
    return -(-n // q) * q


def ffd(sizes, budget):
    """First-Fit-Decreasing，与仓库 `navit_pack_strategy: ffd` 同策略。"""
    bins, used = [], []
    for i in sorted(range(len(sizes)), key=lambda i: -sizes[i]):
        s = sizes[i]
        if s > budget:
            return None
        for b, u in enumerate(used):
            if u + s <= budget:
                bins[b].append(i)
                used[b] += s
                break
        else:
            bins.append([i])
            used.append(s)
    return bins


def packed_route(toks, q, budget):
    """实段降序规范化 + 补齐到自然长度，与 packing.Packer.build_packs 一致。"""
    qs = [quantize(n, q) for n in toks]
    groups = ffd(qs, budget)
    if groups is None:
        return None
    layouts, imgs, real, padded = Counter(), defaultdict(int), 0, 0
    for g in groups:
        seg = sorted((qs[i] for i in g), reverse=True)
        total = quantize(sum(seg), 1024)       # 自然长度（packing.PACK_Q）
        if total > sum(seg):                   # 取整余量 -> 末尾的纯填充段
            seg.append(total - sum(seg))
        layouts[tuple(seg)] += 1
        imgs[tuple(seg)] += len(g)
        real += sum(toks[i] for i in g)
        padded += total
    steps = sum(c // DEVICES for c in layouts.values())
    in_imgs = sum(imgs[k] * (c // DEVICES * DEVICES) // c
                  for k, c in layouts.items())
    return {"fill": real / padded, "cap": padded / (len(groups) * budget),
            "ids": len(layouts),
            "steps": steps, "in_imgs": in_imgs, "n": len(toks),
            "packs": len(groups), "per_pack": len(toks) / len(groups),
            "layouts": layouts}


def ragged_route(toks, q, per_dev, budget):
    by = defaultdict(list)
    for n in toks:
        L = quantize(n, q)
        if L * per_dev > budget:
            return None
        by[L].append(n)
    per_step = DEVICES * per_dev
    steps = real = padded = in_imgs = 0
    for L, ns in by.items():
        k = len(ns) // per_step
        steps += k
        used = ns[:k * per_step]
        in_imgs += len(used)
        real += sum(used)
        padded += len(used) * L
    return {"fill": real / padded if padded else 0.0, "ids": len(by),
            "steps": steps, "in_imgs": in_imgs, "n": len(toks),
            "packs": 0, "per_pack": float(per_dev), "layouts": None}


def show(tag, r):
    if r is None:
        print(f"    {tag:<24} 单图/单桶放不下这个 budget")
        return
    extra = f"  pack {r['packs']} 图/pack {r['per_pack']:.2f}" if r["packs"] else ""
    cap = f"  容量 {r['cap']:>5.1%}" if "cap" in r else ""
    print(f"    {tag:<24} 填充 {r['fill']:>5.1%}{cap}  编译身份 {r['ids']:>3}"
          f"  成步 {r['steps']:>3}  入步图 {r['in_imgs']:>4}/{r['n']}"
          f" ({r['in_imgs'] / r['n']:>3.0%}){extra}")


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*", help="数据集目录（可多个）")
    ap.add_argument("--ladder", default="4096",
                    help="navit_multiscale_token_ladder，逗号分隔；空串=不加副本")
    ap.add_argument("--budgets", default=",".join(map(str, BUDGETS)))
    ap.add_argument("--dump", type=int, default=0,
                    help="打印该 budget 下打包的布局组成（0=不打）")
    a = ap.parse_args(argv)
    if not a.dirs:
        print(__doc__)
        return 2
    ladder = [int(x) for x in a.ladder.split(",") if x.strip()]
    budgets = [int(x) for x in a.budgets.split(",")]

    for d in a.dirs:
        root = Path(d)
        sizes = image_sizes(root)
        if not sizes:
            print(f"\n### {root.name}: 没找到图片")
            continue
        pool, native, per_rung = build_pool(sizes, ladder)
        st, ars = sorted(pool), sorted(w / h for w, h in sizes)
        print(f"\n### {root.name}  原生 {len(native)} 张"
              + "".join(f" + {len(v)} 条 {k} 档副本" for k, v in per_rung.items())
              + f" = 样本池 {len(pool)} 条")
        print(f"  token {st[0]}..{st[-1]} 中位 {st[len(st) // 2]}"
              f"  不同 token 数 {len(set(pool))} 种"
              f"  |  宽高比 {ars[0]:.2f}-{ars[-1]:.2f}")
        print(f"  **budget 下限 = {quantize(max(pool), 128)}**"
              f"（最大单图 {max(pool)} token，装不下就没法训这张图）")
        for q in QUANTA:
            qs = [quantize(t, q) for t in pool]
            print(f"  Q={q:<5} 不同量化长度 {len(set(qs)):<3}"
                  f" 量化填充率 {sum(pool) / sum(qs):.1%}")

        for budget in budgets:
            print(f"  —— budget {budget} ——")
            for q in QUANTA:
                show(f"打包 Q={q}", packed_route(pool, q, budget))
            for q in QUANTA:
                for g in (1, 2):
                    r = ragged_route(pool, q, g, budget)
                    if r is not None:
                        show(f"分桶 Q={q} 每卡{g}图", r)

        if a.dump:
            r = packed_route(pool, QUANTA[-1], a.dump)
            if r is None:
                print(f"\n  （budget {a.dump} 装不下，无法 dump）")
                continue
            print(f"\n  === budget {a.dump} / Q={QUANTA[-1]} 的打包布局组成 ===")
            print("  （x8 及以上才能自己凑成一步；x1 的长尾靠 plan_steps 顺延到下一轮）")
            for lay, c in r["layouts"].most_common(12):
                print(f"    {str(lay):<52} x{c:<3} 段数 {len(lay)}"
                      f" 容量 {sum(lay) / a.dump:.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
