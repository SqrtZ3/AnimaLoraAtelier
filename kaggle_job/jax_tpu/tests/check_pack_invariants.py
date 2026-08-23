"""闸门⑭：「自然长度」打包的不变量（纯 numpy，几秒，不需要 jax）。

`packing.Packer / K2Packer` 从"补齐到全局 budget"改成"补齐到自然长度
`round_up(Σ实段, PACK_Q)`"之后，有四件事只能靠对拍守住 —— 它们都属于
**不报错的失效**，真机上只会表现成"曲线对不上"或"莫名多编译一次"：

  T1 RNG 兼容是双射
      新版仍按旧槽数 `Layout.t_slots` 采 t，再用 `t_gather` 重排（见
      `Layout` 与 `run_train._sample_t`）。这里逐 pack 复算一遍旧口径的段序，
      断言每个**实段**都映到旧版里段长相同的那个槽、且互不重复。错了不会报错，
      只会让新旧 loss 曲线悄悄错位，A/B 结论全废。

  T2 布局数方向：**新 >= 旧，不是"不变"**
      新布局身份 = 实段 multiset（单射）；旧身份 = `sorted(实段 ∪ {budget −
      Σ实段})`，**非单射** —— 填充段长撞上某个实段长时两种实段组合会被合并。
      最小反例（budget=10240）：实段 `{5120}` 与 `{5120, 5120}` 旧元组都是
      `(5120, 5120)`。T2a 把这个反例钉死，T2b 在 fuzz 里断言方向永不反转
      （新 < 旧 就说明去重逻辑出了问题）。

  T3 结构不变量
      `sum(seg_lens) == total_len <= budget`、`total_len % PACK_Q == 0`、
      段长 128 对齐（splash 块粒度，`attention._check_aligned` 的前置条件）、
      纯填充段恒在末尾且至多一个、实段 multiset 与旧口径逐 pack 一致。

  T4 quantum >= PACK_Q 时**不该有填充段**
      Σ实段 是 quantum 的倍数 -> 已经 PACK_Q 对齐 -> 取整余量恒 0。这条一旦破，
      说明自然长度算错了；它同时是 `real_seg_lens`（seg_cap）在 Anima 主工作点
      上等价于 `seg_lens` 的依据。

K2 侧多一条：combined 段 = 128 量化文本槽 + 图像槽，`Σ实段` 几乎从不 1024
对齐，所以填充段是常态 —— T5 顺带报告「旧口径下 `min(所有段)` 会落进哪一档
反向块」，这正是 `real_seg_lens` 要挡的东西（档位见 `attention.BWD_BLOCK_PREF`）。

跑法（任意带 numpy 的解释器）：
    python check_pack_invariants.py [--trials 1500] [--seed 0] [-v]
"""

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import packing as PK                                            # noqa: E402
from attention import BWD_BLOCK_PREF                            # noqa: E402

BUDGETS = (8192, 10240, 16384, 32768, 81920)
QUANTA = (128, 256, 512, 1024, 2048)


class Fail(Exception):
    pass


def check(cond, msg):
    if not cond:
        raise Fail(msg)


# ── 旧口径复算（唯一的真相来源：改动前的 build_packs）──────────────────────────
def old_anima(budget, quantum, token_counts):
    """旧版 Anima build_packs 的段序：补齐到 budget 的填充段**参与**整体降序。

    返回每个 pack 的 (段长元组, 每槽是否实段的标记列表)。
    """
    q = [PK.quantize_len(n, quantum) for n in token_counts]
    out = []
    for group in PK.ffd(q, budget):
        seg = [q[i] for i in group]
        is_real = [True] * len(seg)
        rest = budget - sum(seg)
        if rest:
            seg.append(rest)
            is_real.append(False)
        order = sorted(range(len(seg)), key=lambda i: (-seg[i], i))
        out.append((tuple(seg[i] for i in order),
                    [is_real[i] for i in order]))
    return out


def old_k2(budget, quantum, txt_quantum, token_counts, txt_lens):
    """旧版 K2 build_packs 的段序：实段降序，填充段恒在末尾（不参与排序）。"""
    vols = [PK.quantize_len(t, txt_quantum) + PK.quantize_len(n, quantum)
            for n, t in zip(token_counts, txt_lens)]
    out = []
    for group in PK.ffd(vols, budget):
        seg = sorted((vols[i] for i in group), reverse=True)
        is_real = [True] * len(seg)
        rest = budget - sum(seg)
        if rest:
            seg.append(rest)
            is_real.append(False)
        out.append((tuple(seg), is_real))
    return out


# ── 逐 pack 断言 ──────────────────────────────────────────────────────────────
def check_pack(L, oseg, oreal, budget, quantum, real_lens_of_pad_free):
    """对一个 pack 跑 T1/T3/T4。`oseg/oreal` 是旧口径的段序与实段标记。"""
    pad_idx = getattr(L, "pad_idx", None)
    if pad_idx is None:                       # K2Layout 没有 pad_idx 字段
        pad_idx = len(L.seg_lens) - 1 if len(L.seg_lens) > len(L.txt_segs) else -1

    # T3 结构
    check(sum(L.seg_lens) == L.total_len, f"Σ段长 != total_len：{L.seg_lens}")
    check(L.total_len <= budget, f"total_len {L.total_len} > budget {budget}")
    check(L.total_len % PK.PACK_Q == 0, f"total_len {L.total_len} 未对齐 PACK_Q")
    check(all(s % PK.BLOCK == 0 for s in L.seg_lens), f"段长未 128 对齐：{L.seg_lens}")
    check(sum(1 for i in range(len(L.seg_lens)) if i == pad_idx) <= 1, "填充段多于一个")
    check(sorted(L.real_seg_lens) == sorted(s for s, r in zip(oseg, oreal) if r),
          f"实段 multiset 与旧口径不一致：{L.real_seg_lens} vs {oseg}")

    # T4 quantum >= PACK_Q 时不该有填充段（只对 Anima 成立：K2 的段含 128 文本槽）
    if real_lens_of_pad_free and quantum % PK.PACK_Q == 0:
        check(pad_idx == -1, f"quantum={quantum} 仍产出了填充段：{L.seg_lens}")

    # T1 RNG 兼容双射
    check(L.t_slots == len(oseg), f"t_slots {L.t_slots} != 旧段数 {len(oseg)}")
    check(len(L.t_gather) == len(L.seg_lens), "t_gather 长度 != 新段数")
    used = []
    for p, slot in enumerate(L.t_gather):
        check(0 <= slot < L.t_slots, f"t_gather 越界：{slot} 不在 [0,{L.t_slots})")
        if p == pad_idx:
            continue                          # 新填充位复用旧填充槽，它的 t 反正被丢
        check(oreal[slot], f"实段第 {p} 段映到了旧填充槽 {slot}")
        check(oseg[slot] == L.seg_lens[p],
              f"实段第 {p} 段长 {L.seg_lens[p]} != 旧槽 {slot} 的 {oseg[slot]}")
        used.append(slot)
    check(len(set(used)) == len(used), f"两个实段映到同一个旧槽：{L.t_gather}")


# ── 反例回归（T2a）────────────────────────────────────────────────────────────
def t2a_counterexample(verbose):
    """budget=10240：实段 {5120} 与 {5120,5120} 旧同布局、新必须是两个布局。"""
    B, Q = 10240, 1024
    pk = PK.Packer(B, Q, 512, 8)
    one = pk.build_packs([0], [5120], [(64, 80)])
    two = PK.Packer(B, Q, 512, 8).build_packs([0, 1], [5120, 5120],
                                              [(64, 80), (64, 80)])
    check(len(one) == len(two) == 1, "反例构造失败：FFD 没装成一个 pack")
    l1, l2 = one[0].layout, two[0].layout
    check(l1.total_len == 5120 and l2.total_len == 10240,
          f"自然长度不对：{l1.total_len} / {l2.total_len}")
    check(l1 != l2, "反例失效：两个不同实段组合仍是同一个 Layout")
    o1 = old_anima(B, Q, [5120])[0][0]
    o2 = old_anima(B, Q, [5120, 5120])[0][0]
    check(o1 == o2 == (5120, 5120), f"旧口径复算不对：{o1} / {o2}")
    if verbose:
        print(f"  T2a 反例：旧 {o1} 一种布局 -> 新 {l1.seg_lens}/{l2.seg_lens} 两种")


# ── fuzz ──────────────────────────────────────────────────────────────────────
def fuzz_anima(trials, seed, verbose):
    rng = random.Random(seed)
    delta, n_case = Counter(), 0
    for _ in range(trials):
        budget = rng.choice(BUDGETS)
        quantum = rng.choice(QUANTA)
        if budget % quantum or budget % PK.PACK_Q:
            continue
        n = rng.randint(1, 24)
        hw = [(rng.randint(4, 120), rng.randint(4, 120)) for _ in range(n)]
        keep = [i for i, (h, w) in enumerate(hw)
                if PK.quantize_len(h * w, quantum) <= budget]
        if not keep:
            continue
        hw = [hw[i] for i in keep]
        tc = [h * w for h, w in hw]
        n_case += 1
        old = old_anima(budget, quantum, tc)
        new = PK.Packer(budget, quantum, 512, 8).build_packs(
            list(range(len(tc))), tc, hw)
        check(len(old) == len(new), f"pack 数变了：{len(old)} -> {len(new)}")
        for (oseg, oreal), p in zip(old, new):
            check_pack(p.layout, oseg, oreal, budget, quantum, True)
        n_old = len({o[0] for o in old})
        n_new = len({p.layout for p in new})
        check(n_new >= n_old, f"布局数反而变少了：{n_old} -> {n_new}（去重逻辑有问题）")
        delta[n_new - n_old] += 1
    if verbose:
        print(f"  anima fuzz {n_case} 组，布局数增量分布 {dict(sorted(delta.items()))}")
    return n_case


def fuzz_k2(trials, seed, verbose):
    rng = random.Random(seed + 1)
    delta, n_case, tier_old, tier_new = Counter(), 0, Counter(), Counter()
    for _ in range(trials):
        budget = rng.choice(BUDGETS)
        quantum = rng.choice(QUANTA)
        if budget % quantum or budget % PK.PACK_Q:
            continue
        n = rng.randint(1, 20)
        hw = [(rng.randint(4, 100), rng.randint(4, 100)) for _ in range(n)]
        txt = [rng.randint(1, 400) for _ in range(n)]
        keep = [i for i, (h, w) in enumerate(hw)
                if PK.quantize_len(txt[i], 128) + PK.quantize_len(h * w, quantum)
                <= budget]
        if not keep:
            continue
        hw = [hw[i] for i in keep]
        txt = [txt[i] for i in keep]
        tc = [h * w for h, w in hw]
        n_case += 1
        old = old_k2(budget, quantum, 128, tc, txt)
        new = PK.K2Packer(budget, quantum, 128, 8).build_packs(
            list(range(len(tc))), tc, txt, hw)
        check(len(old) == len(new), f"pack 数变了：{len(old)} -> {len(new)}")
        for (oseg, oreal), p in zip(old, new):
            check_pack(p.layout, oseg, oreal, budget, quantum, False)
            # T5 报告：旧口径的 min(所有段) 会掉到哪一档反向块
            reals = list(p.layout.real_seg_lens)
            rest_old = budget - sum(reals)
            oc = min(reals + [rest_old]) if rest_old else min(reals)
            tier_old[next((b for b in BWD_BLOCK_PREF if b <= oc), PK.BLOCK)] += 1
            tier_new[next((b for b in BWD_BLOCK_PREF if b <= min(reals)),
                          PK.BLOCK)] += 1
        n_old = len({o[0] for o in old})
        n_new = len({p.layout for p in new})
        check(n_new >= n_old, f"布局数反而变少了：{n_old} -> {n_new}")
        delta[n_new - n_old] += 1
    if verbose:
        print(f"  k2 fuzz {n_case} 组，布局数增量分布 {dict(sorted(delta.items()))}")
        print(f"  T5 反向块档位（旧 min 含填充段 -> 新 min 只看实段）："
              f"{dict(sorted(tier_old.items()))} -> {dict(sorted(tier_new.items()))}")
    return n_case


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    try:
        t2a_counterexample(a.verbose)
        na = fuzz_anima(a.trials, a.seed, a.verbose)
        nk = fuzz_k2(a.trials, a.seed, a.verbose)
    except Fail as e:
        print(f"打包不变量闸门 不通过：{e}")
        return 1
    print(f"打包不变量闸门 通过：T1 RNG 双射 / T2 布局数方向(新>=旧)+反例 / "
          f"T3 结构 / T4 quantum>=PACK_Q 无填充段 —— anima {na} 组、k2 {nk} 组")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
