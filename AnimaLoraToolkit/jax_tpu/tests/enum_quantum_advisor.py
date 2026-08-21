"""**训练前的 `--quantum` 补正**：用你这次要跑的那份 yaml + 那个数据集，扫出该传多少。

不是闸门，是**每次开训前跑一遍的 preflight**。零 TPU 配额、零 jax（只要 numpy +
pyyaml），几秒钟。它直接调 `packing.Packer / K2Packer` 和 `data.CacheDataset`，
不复刻任何逻辑 —— 口径与真训练一致，`packing.py` 改了这里自动跟着改。

## 为什么 quantum 值得每个数据集重扫一次

pack 补齐到自然长度（`packing.PACK_Q`）之后，**装箱紧不紧已经不再买到算力了**：

    每 epoch 总 token = Σ_图 round_up(token, quantum) + Σ_pack (< PACK_Q 的取整余量)

怎么分组都一样。所以填充率这个老指标不再是调 packing 的目标，剩下的只有四个轴，
而 `quantum` 同时压在这四个轴上、方向还相反：

  1. **算力**（Σ自然长度）—— quantum 越小越省。这是唯一剩下的 FLOPs 杠杆。
  2. **反向块档位** —— splash 反向块取 `min(实段)` 能容下的最大档
     （`attention.BWD_BLOCK_PREF`，真机：1024 fused 142ms / 512 fused 157ms /
     默认 128 非 fused 288ms）。quantum 小到让某张图的段短于 1024，**全盘**反向块
     就跟着掉档。这一轴是悬崖不是斜坡，所以本脚本把它当硬闸门而不是折算进分数。
  3. **布局数** = 全模型编译次数。Anima 路径只是多编译一次（磁盘缓存能摊掉）；
     K2 是单布局驻留 + 驱逐（`run_train.py:588`），多一个布局 = 多一次/epoch 的
     驱逐重载。quantum 越小布局越多。
  4. **入步图率** —— 同布局要凑够 `devices` 个 pack 才成步，凑不齐的进 carry 顺延。
     quantum 越小、布局越碎，顺延越多。**这里报的是单轮口径**：carry 会带到下一轮
     接着凑，样本不丢，所以小数据集上这个数天然偏低，别照着它一个人做决定。

## 怎么读结果

「相对算力」一列是 **Σ自然长度之比**，即步时的**推算**上界收益，**不是真机计时**：
host 侧采样/组 batch、优化器、all-reduce 这些固定开销不随 token 数缩，真机收益
只会更小。别把它当实测报出去。

布局数与入步图率在真训练里每个 epoch 都会因洗牌而抖，所以默认采样 `--draws 3`
个洗牌顺序，报均值与最坏值。

## 跑法

    python enum_quantum_advisor.py --config ../../../config/train_xxx.yaml
    python enum_quantum_advisor.py --config <yaml> --quanta 512,1024,2048 --draws 5
    python enum_quantum_advisor.py --config <yaml> --data-dir <另一个缓存目录>

数据集要的是**TPU 侧的 latent 缓存目录**（`<stem>.npz` + `<stem>.textfeat.npz`，
含 `*.ms<档>.npz` 的 multiscale sidecar），不是原图目录 —— 与 `run_train.py` 同一份。
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as C                                              # noqa: E402
import data as D                                                # noqa: E402
import packing as PK                                            # noqa: E402
from attention import BWD_BLOCK_PREF                            # noqa: E402

QUANTA = (256, 512, 1024, 2048, 4096)


def bwd_tier(seg_cap: int) -> int:
    """`attention._block_sizes` 的档位选择（这里只关心受 seg_cap 约束的那一半）。"""
    return next((b for b in BWD_BLOCK_PREF if b <= seg_cap), PK.BLOCK)


def one_draw(ds, rc, quantum):
    """跑一次完整的 plan_packed，返回这一轮的统计。装不下时抛 ValueError。"""
    if rc.family == "krea2":
        packer = PK.K2Packer(rc.budget, quantum, devices=rc.devices)
    else:
        packer = PK.Packer(rc.budget, quantum, rc.txt_len, rc.devices)
    steps, carry = ds.plan_packed(packer)
    packs = [p for st in steps for p in st] + list(carry)
    if not packs:
        raise ValueError("一个 pack 都没打出来（数据集是空的？）")

    by = Counter(p.layout for p in packs)
    in_step = sum(len(p.layout.real_seg_lens) for st in steps for p in st)
    total_img = sum(len(p.layout.real_seg_lens) for p in packs)

    natural = sum(p.layout.total_len for p in packs)
    if rc.family == "krea2":
        real = sum(sum(p.real_img_lens) + sum(p.real_txt_lens) for p in packs)
    else:
        real = sum(sum(p.real_lens) for p in packs)

    tiers = Counter()
    for layout, c in by.items():
        tiers[bwd_tier(min(layout.real_seg_lens))] += c
    return {
        "packs": len(packs), "layouts": len(by), "natural": natural, "real": real,
        "in_step": in_step, "total_img": total_img, "steps": len(steps),
        "tiers": tiers, "worst_tier": min(tiers),
        "worst_layout": min(by, key=lambda L: min(L.real_seg_lens)),
    }


def scan(ds, rc, quanta, draws):
    """对每个候选 quantum 采 `draws` 个洗牌顺序，聚合成一行。"""
    rows = []
    for q in quanta:
        if q % PK.BLOCK or rc.budget % q or rc.budget % PK.PACK_Q:
            rows.append({"q": q, "skip": f"budget {rc.budget}/卡 不能被它整除"})
            continue
        try:
            ds_runs = [one_draw(ds, rc, q) for _ in range(draws)]
        except ValueError as e:
            rows.append({"q": q, "skip": str(e).split("（")[0]})
            continue
        agg = {
            "q": q,
            "natural": float(np.mean([r["natural"] for r in ds_runs])),
            "fill": float(np.mean([r["real"] / r["natural"] for r in ds_runs])),
            "layouts": float(np.mean([r["layouts"] for r in ds_runs])),
            "layouts_max": max(r["layouts"] for r in ds_runs),
            "instep": float(np.mean([r["in_step"] / r["total_img"] for r in ds_runs])),
            "instep_min": min(r["in_step"] / r["total_img"] for r in ds_runs),
            "steps": float(np.mean([r["steps"] for r in ds_runs])),
            "worst_tier": min(r["worst_tier"] for r in ds_runs),
            "worst_layout": min((r["worst_layout"] for r in ds_runs),
                                key=lambda L: min(L.real_seg_lens)),
        }
        rows.append(agg)
    return rows


def advise(rows, max_layouts, min_instep):
    """规则式补正：先过三道硬闸门，再在通过者里取算力最省的。

    刻意**不折算成一个分数** —— 反向块档位是悬崖（256 fused 从没在真机上量过），
    编译一次的秒数也随模型大小变，硬凑一个加权和只会造出一个看着客观的假数字。
    """
    ok, rej = [], []
    for r in rows:
        if "skip" in r:
            continue
        bad = []
        if r["worst_tier"] < PK.PACK_Q:
            bad.append(f"反向块掉到 {r['worst_tier']}（最短实段 "
                       f"{min(r['worst_layout'].real_seg_lens)}）")
        if r["layouts_max"] > max_layouts:
            bad.append(f"布局数 {r['layouts_max']} > {max_layouts}")
        if r["instep_min"] < min_instep:
            bad.append(f"入步图率 {r['instep_min']:.0%} < {min_instep:.0%}")
        (rej if bad else ok).append((r, bad))
    if not ok:
        return None, rej
    best = min(ok, key=lambda kv: kv[0]["natural"])[0]
    return best, rej


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="与 run_train.py 同一份训练 yaml")
    ap.add_argument("--data-dir", default="", help="覆盖 yaml 的 data_dir")
    ap.add_argument("--devices", type=int, default=8)
    ap.add_argument("--quanta", default=",".join(map(str, QUANTA)))
    ap.add_argument("--draws", type=int, default=3,
                    help="采几个洗牌顺序（布局数/入步图率每 epoch 都会抖）")
    ap.add_argument("--current", type=int, default=1024,
                    help="这次准备传的 --quantum（run_train.py 的默认值是 1024）")
    ap.add_argument("--max-layouts", type=int, default=0,
                    help="布局数上限；0=按 family 取默认（krea2 6 / anima 16）")
    ap.add_argument("--min-instep", type=float, default=0.90,
                    help="入步图率下限（低于它说明太多样本在 carry 里顺延）")
    a = ap.parse_args(argv)

    raw = C.load_yaml(a.config)
    if a.data_dir:
        raw["data_dir"] = a.data_dir
    rc = C.build(raw, a.devices, allow_unported=True, canvas_hw=(1, 1))
    # caption_dropout 只影响取哪份文本特征，不影响打包；这里关掉，免得因为缺
    # `_empty.textfeat.npz` 报一个与本脚本无关的错。
    ds = D.CacheDataset(rc.data_dir, txt_len=rc.txt_len, flip_prob=rc.flip_prob,
                        repeats=rc.repeats, multiscale=rc.multiscale,
                        caption_dropout=0.0, family=rc.family,
                        rng=np.random.RandomState(rc.tcfg.seed))
    max_layouts = a.max_layouts or (6 if rc.family == "krea2" else 16)

    print(f"配置 {a.config}")
    print(f"数据 {rc.data_dir}  family={rc.family}  样本 {len(ds.samples)} 条"
          f"（含 multiscale 副本）×repeats {rc.repeats}")
    print(f"预算 全局 {rc.budget * rc.devices} = {rc.devices} 卡 × {rc.budget}/卡"
          f"  |  PACK_Q={PK.PACK_Q}  |  洗牌采样 {a.draws} 次\n")

    quanta = sorted({int(x) for x in a.quanta.split(",") if x.strip()})
    rows = scan(ds, rc, quanta, a.draws)

    print(f"{'quantum':>8} {'相对算力':>9} {'算力填充':>9} {'布局数':>12} "
          f"{'入步图率':>12} {'步/轮':>7} {'反向块':>8}")
    base = min((r["natural"] for r in rows if "skip" not in r), default=1.0)
    for r in rows:
        if "skip" in r:
            print(f"{r['q']:>8}  跳过：{r['skip']}")
            continue
        print(f"{r['q']:>8} {r['natural'] / base:>9.3f} {r['fill']:>8.1%} "
              f"{r['layouts']:>7.1f}(最坏{r['layouts_max']:>2}) "
              f"{r['instep']:>7.1%}(最坏{r['instep_min']:>4.0%}) "
              f"{r['steps']:>7.1f} {r['worst_tier']:>8}")

    best, rej = advise(rows, max_layouts, a.min_instep)
    print(f"\n硬闸门：反向块 = {PK.PACK_Q} | 布局数 <= {max_layouts} | "
          f"入步图率 >= {a.min_instep:.0%}")
    for r, bad in rej:
        print(f"  quantum {r['q']:<5} 不合格：{'；'.join(bad)}")
    if best is None:
        print("\n没有候选通过硬闸门。要么放宽 --max-layouts/--min-instep，"
              "要么先调 navit_token_budget 或数据集的分辨率分布。")
        return 1

    cur = next((r for r in rows if r.get("q") == a.current and "skip" not in r), None)
    print(f"\n建议：--quantum {best['q']}")
    if cur is None:
        print(f"  （当前的 --quantum {a.current} 不在扫描结果里，没法对比）")
    elif cur["q"] == best["q"]:
        print("  与当前一致，不用改。")
    else:
        gain = 1 - best["natural"] / cur["natural"]
        print(f"  相对当前的 --quantum {a.current}：算力口径 -{gain:.1%}"
              f"（**推算**，真机步时收益更小），布局数 "
              f"{cur['layouts_max']} -> {best['layouts_max']}，入步图率 "
              f"{cur['instep_min']:.0%} -> {best['instep_min']:.0%}")
    print("  依据：以上三道硬闸门内取 Σ自然长度最小者。反向块档位没有折算进分数"
          "（256 fused 真机没量过），它是硬闸门。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
