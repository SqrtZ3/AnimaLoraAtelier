"""本地枚举（零成本）：ARB 桶 -> navit 段长 -> FFD 填充率 / 布局数 / 128 对齐。

回答"上 TPU 前必须先知道"的两件事：
  * 布局种类有多少（决定 splash 的 MaskInfo 预热成本会不会爆炸）
  * 填充率多少（padding 也要过 MLP，填充率**直接等于**线性层的算力利用率）

**当前是 Krea2 口径**（段 = [512 文本 ; 图像 token]，见 krea2_modeling.py:971）。
Anima 的文本走 cross-attn、不进主序列，段 = 纯图像 token —— 换算时把 TXT 设为 0。

依据：
  桶生成    trainer/data.py:286 `_generate`（面积容差 10%、步长 step、长宽比闸门）
  token 数  VAE 下采样 8 x patch 2 -> N = H*W/256（1024^2 -> 4096，与实现一致）
  multiscale  trainer/data.py:495 是**追加**缩放副本，不是替换
  对齐要求  splash block_kv=128，段长不对齐会产生 partial block

用法：python enum_pack_layouts.py [--txt 512] [--budgets 16384,24576,32768]
"""

import argparse
import random
from collections import Counter

# 取自 config/train_krea2_baseline_v2.yaml 的分桶段
MIN_R, MAX_R, STEP, MAX_AR = 512, 3072, 64, 2.7
BASES = [1536]
LADDER = [2304, 4096]          # navit_multiscale_token_ladder
N_IMAGES = 159                 # 数据集规模（只影响统计量级，不影响结论）


def gen_buckets():
    out, seen = [], set()
    for base in BASES:
        area = base * base
        for w in range(MIN_R, MAX_R + 1, STEP):
            for h in range(MIN_R, MAX_R + 1, STEP):
                if abs(w * h - area) / area > 0.1:
                    continue
                if max(w / h, h / w) > MAX_AR + 1e-9:
                    continue
                if (w, h) not in seen:
                    seen.add((w, h))
                    out.append((w, h))
    return out


def ffd(items, budget):
    """First-Fit-Decreasing 装箱，与 navit_pack_strategy: ffd 同策略。"""
    packs = []
    for s in sorted(items, reverse=True):
        for p in packs:
            if sum(p) + s <= budget:
                p.append(s)
                break
        else:
            packs.append([s])
    return packs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--txt", type=int, default=512,
                    help="每图文本段长；Anima 走 cross-attn 时设 0")
    ap.add_argument("--budgets", default="16384,24576,32768,49152")
    a = ap.parse_args()
    TXT = a.txt
    budgets = [int(x) for x in a.budgets.split(",")]

    random.seed(0)
    buckets = gen_buckets()
    full = [TXT + (w * h // 256) for (w, h) in random.choices(buckets, k=N_IMAGES)]
    copies = [TXT + n for n in LADDER for _ in range(N_IMAGES)]
    pool = full + copies
    print(f"ARB 桶 {len(buckets)} 个 | 样本池 {len(full)} 全分辨率 + "
          f"{len(copies)} multiscale 副本 = {len(pool)} 条（TXT={TXT}）")
    print(f"段长 {min(pool)}..{max(pool)}，不同段长 {len(set(pool))} 种\n")

    print(f"{'budget':>7} {'packs':>6} {'填充率':>8} {'图/pack':>9} {'布局数':>7}")
    for b in budgets:
        if b < max(pool):
            print(f"{b:>7}  单段 {max(pool)} 放不下，跳过")
            continue
        packs = ffd(pool, b)
        used = sum(sum(p) for p in packs)
        layouts = {tuple(sorted(p)) for p in packs}
        print(f"{b:>7} {len(packs):>6} {used / (len(packs) * b):>7.1%} "
              f"{len(pool) / len(packs):>9.2f} {len(layouts):>7}")

    b0 = budgets[0]
    print(f"\n—— budget={b0} 的 pack 组成（前 8 种）——")
    for comp, c in Counter(tuple(sorted(p)) for p in ffd(pool, b0)).most_common(8):
        print(f"  {str(comp):<36} x{c:<4} 填充 {sum(comp) / b0:.1%} 段数={len(comp)}")

    al = sum(1 for s in pool if s % 128 == 0)
    pad = sum((-s) % 128 for s in pool)
    print(f"\n段长对齐 128：按样本 {al}/{len(pool)}；"
          f"按种类 {sum(1 for s in set(pool) if s % 128 == 0)}/{len(set(pool))}")
    print(f"向上取整到 128 的倍数：平均每段多 {pad / len(pool):.1f} token "
          f"({pad / sum(pool):.2%})")


if __name__ == "__main__":
    main()
