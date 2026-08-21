"""NaViT 打包：把任意分辨率的图片装进 budget 容量上限内的 pack，并控制布局种类。

## 补齐到「自然长度」，不是 budget

每个布局本来就独立编译（见下），所以 pack **不需要统一长度**：FFD 只负责把量化
段装进全局 budget（显存容量上限，语义不变），pack 总长 = **自然长度**
`round_up(Σ实段, PACK_Q)`，取整余量（< PACK_Q，仅 quantum < PACK_Q 时非零）才
追加一个纯填充段。round 到 1024 是为了让 splash 反向块恒能取 1024（真机实测
1024 fused 142ms / 512 fused 157ms / 默认 128 非 fused 288ms，见
attention.BWD_BLOCK_PREF —— 掉到 512 只慢约 1.1x，掉到 128 才是 2x）。这样
budget 维度的 padding 浪费整个消失。

**收益是算力口径的推算，不是真机计时**：Σ自然长度 / (pack 数 × budget) 就是
步时的预期比值（填充 token 与真 token 一样过 MLP 与块对角注意力）。真机步时
收益会小于它 —— host 侧采样/组 batch、优化器、all-reduce 这些固定开销不随
token 数缩。

## 布局数：新口径 >= 旧口径，**不是"不变"**

新布局身份 = 实段 multiset（单射）；旧口径（补齐到 budget）的身份是
`sorted(实段 ∪ {budget − Σ实段})`，**非单射** —— 填充段长恰好撞上某个实段长时，
两种不同的实段组合会被合并成同一次编译。budget=10240 的最小反例：实段
`{5120}` 与 `{5120, 5120}` 的旧元组都是 `(5120, 5120)`，新口径拆成两个布局。

所以正确的说法是「**新布局数 >= 旧布局数，差多少取决于数据**」。
`tests/check_pack_invariants.py` 把这条方向性断言 + 上面那个反例固化成闸门；
随机 fuzz 里约 5% 的样本池会 +1..+3。真实数据集上要不要在意，跑
`tests/enum_quantum_advisor.py` 看，别靠推理免测：多一个布局在 Anima 路径上
只是多一次编译（磁盘缓存摊掉），在 K2 的单布局驻留路径上是多一次/epoch 的
驱逐重载。

## 这个模块存在的理由

TPU 上 splash 的块稀疏由**编译期 mask** 驱动（真机实测：编译期 mask 跳块比
0.252/理论 0.250；运行时 mask 只有 0.569，且 8 头以上放不下 HBM）。于是段长布局
进了编译身份 —— **每种布局要单独编译一次全模型**（真机实测首调约 60s）。

任意分辨率会产生几十种段长，直接打包出 43~47 种布局。所以这里做两件事：

  1. **段长量化**到 `quantum` 的倍数（默认 1024），把布局数压下来。
     本地枚举实测（159 张真实 ARB 图 + multiscale 副本 = 477 条）：

         budget  Q      布局数  有效填充率
         16384   128    17     84.8%
         16384   1024   4      84.8%
         32768   128    25     94.8%
         32768   1024   **4**  **94.8%**      <- 甜点
         49152   512    10     97.0%
         49152   4096   4      84.0%

     Q=1024 是拐点：布局数从 25 塌到 4，填充率一分不掉。

  2. **两级 mask 的索引构造**。量化会在图像段内留一截填充 token，它们会被同段的
     真 token 看见（静默污染，本地实测 max_abs=5.4）。而"哪些是填充"逐 pack 变化，
     编进编译期 mask 就前功尽弃。解法是 splash 的契约（SegmentIds 文档原文
     "The static mask is and-ed with the segment id mask"）：粗粒度进编译期负责
     跳块，精细边界走运行时 `segment_ids` 负责正确性，形状固定不触发重编译。

## 8 卡 DP 的额外约束

一个编译产物只有一种 mask，所以**一步的 8 个 pack 必须同布局**。`plan_steps`
按布局分组后每 8 个成一步，凑不满 8 个的余量顺延到下一轮（用户的训练范式是
"极大 epoch 一直训、随时挑 checkpoint"，顺延不会丢样本）。
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

BLOCK = 128          # splash 块粒度；quantum 必须是它的倍数
PAD_SEG = -1         # 自注意力里填充 token 的段号（它们彼此可见，行不空）
PACK_Q = 1024        # pack 自然长度的取整粒度：让 splash 反向块恒能取 1024


def quantize_len(n: int, quantum: int) -> int:
    """真实 token 数 -> 粗粒度段长（向上取整到 quantum 的倍数）。"""
    return int(math.ceil(n / quantum) * quantum)


@dataclass(frozen=True)
class Layout:
    """一个 pack 的**编译身份**。同 Layout 的 pack 共用一份编译产物。

    total_len 是该布局的**自然长度** = round_up(Σ实段, PACK_Q)，不再是全局 budget
    （budget 只剩 FFD 装箱容量上限一个语义，在 Packer 上）。seg_lens 实段降序、
    纯填充段（取整余量，长 < PACK_Q）恒在末尾，恒有 sum(seg_lens) == total_len。

    t_slots / t_gather 是 **RNG 兼容信息（不进编译身份，compare=False）**：
    旧版（补齐到 budget）每个 pack 按含填充段的段数采 t，填充段的 t 采了被丢；
    为让新旧两条代码路径的 RNG 消费逐 bit 一致，仍按旧槽数 t_slots 采样，再用
    t_gather（新段序第 p 段 -> 旧槽位下标）重排。直接构造 Layout 时留默认即可
    （t_slots=0 -> 按 n_seg 采样，不重排）。

    **这是一次性的 A/B 兼容层，做完新旧曲线对比就该摘掉。** 留着的代价是
    `budget`（一个 config 值）经 `rest_old` 永久地隐式决定 t 采样的 RNG 流 ——
    换 budget 就换随机流，而它本该只是显存容量上限。摘除时：删掉这两个字段、
    `Packer/K2Packer` 里 `rest_old/pad_old/t_gather` 的推导、`run_train._sample_t`，
    `_sample_t` 退回 `sampler.sample(rng, len(packs) * layout.n_seg, ...)`。
    """
    total_len: int
    seg_lens: Tuple[int, ...]
    txt_len: int
    pad_idx: int = -1                                # 纯填充段下标（恒在末尾）；-1=无
    t_slots: int = field(default=0, compare=False)
    t_gather: Tuple[int, ...] = field(default=(), compare=False)

    def __post_init__(self):
        if sum(self.seg_lens) != self.total_len:
            raise ValueError(f"段长和 {sum(self.seg_lens)} != total_len {self.total_len}")
        bad = [n for n in self.seg_lens if n % BLOCK]
        if bad:
            raise ValueError(f"段长必须是 {BLOCK} 的倍数（splash 块粒度），越界 {bad[:4]}")
        if self.txt_len % BLOCK:
            raise ValueError(f"txt_len 必须是 {BLOCK} 的倍数，得到 {self.txt_len}")
        if not (-1 <= self.pad_idx <= len(self.seg_lens) - 1):
            raise ValueError(f"pad_idx {self.pad_idx} 越界（-1 或 [0, {len(self.seg_lens)})）")
        if self.pad_idx != -1 and self.pad_idx != len(self.seg_lens) - 1:
            raise ValueError("纯填充段恒在末尾（实段降序规范化后追加）")
        if self.t_slots:
            if len(self.t_gather) != len(self.seg_lens) or \
                    any(not 0 <= g < self.t_slots for g in self.t_gather):
                raise ValueError(f"t_gather 必须是 {len(self.seg_lens)} 个 [0, "
                                 f"{self.t_slots}) 内的下标，得到 {self.t_gather}")

    @property
    def n_seg(self) -> int:
        return len(self.seg_lens)

    @property
    def real_seg_lens(self) -> Tuple[int, ...]:
        """实段段长（排除纯填充段）。反向块上限（seg_cap）只统计它们：< PACK_Q 的
        填充段若参与取 min 会把全盘反向块拖小。按 attention.BWD_BLOCK_PREF 的真机
        数：1024 fused 142ms / 512 fused 157ms（约 1.1x）/ 默认 128 非 fused 288ms
        （约 2x），256 fused 没量过。**这不是纯防御**：K2 的 combined 段 =
        128 量化文本槽 + 图像槽，Σ实段几乎从不 1024 对齐，所以每个 pack 都带取整
        余量段；本地合成池上 quantum=1024 时旧口径有 ~70% 的 pack 会掉进
        256/128 档。"""
        return tuple(s for i, s in enumerate(self.seg_lens) if i != self.pad_idx)

    @property
    def max_chunk(self) -> int:
        """`anima_jax.forward_packed(chunk=...)` 能用的最大 chunk。

        = 所有段长的最大公约数（这样每个 chunk 完整落在一张图内）。段长都是
        quantum 的倍数时，它 >= quantum。chunk 越大，AdaLN 的 gather 目标越小。
        **跨段的 chunk 会让调制静默用错图的 t**，所以这里取 gcd 而不是 quantum。
        """
        return math.gcd(*self.seg_lens) if len(self.seg_lens) > 1 else self.seg_lens[0]

    @property
    def kv_txt(self) -> int:
        """cross-attn 的 kv 长度：每段一个定长文本槽（含填充段，否则它的行会全 0）。"""
        return self.n_seg * self.txt_len


@dataclass
class Pack:
    """一个 pack 的**运行时**内容。layout 相同、内容不同 -> 不重编译。"""
    layout: Layout
    items: List[object] = field(default_factory=list)      # 每段的样本引用
    real_lens: List[int] = field(default_factory=list)     # 每段真实 token 数
    grids: List[Tuple[int, int]] = field(default_factory=list)   # 每段 (h, w)

    def index_arrays(self) -> Dict[str, np.ndarray]:
        """造出喂给 attention.py / anima_jax.py 的全部索引数组。

        seg_self   [B] 自注意力精细段号；段内填充统一 PAD_SEG（彼此可见，行不空）
        seg_cross  [B] cross-attn 精细段号；填充**沿用宿主段号**——文本侧没有配对的
                       填充段，给独立段号会让整行全 0 -> softmax 分母为 0
                       （splash 的 SegmentIds 文档对此有明确警告）
        seg_txt    [kv_txt] 文本侧段号
        mod_index  [B] AdaLN 的 token->段（= 粗粒度段号）
        rows/cols  [B] RoPE 网格坐标；填充位置填 0（输出被 loss_mask 丢弃）
        loss_mask  [B] 1=真 token
        """
        L = self.layout
        B = L.total_len
        seg_self = np.full(B, PAD_SEG, np.int32)
        seg_cross = np.empty(B, np.int32)
        mod_index = np.empty(B, np.int32)
        rows = np.zeros(B, np.int32)
        cols = np.zeros(B, np.int32)
        loss_mask = np.zeros(B, np.float32)
        off = 0
        for i, seg in enumerate(L.seg_lens):
            seg_cross[off:off + seg] = i
            mod_index[off:off + seg] = i
            r = self.real_lens[i] if i < len(self.real_lens) else 0
            if r:
                h, w = self.grids[i]
                if h * w != r:
                    raise ValueError(f"第 {i} 段网格 {h}x{w} != 真实 token 数 {r}")
                seg_self[off:off + r] = i
                loss_mask[off:off + r] = 1.0
                rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
                rows[off:off + r] = rr.reshape(-1)
                cols[off:off + r] = cc.reshape(-1)
            off += seg
        return {"seg_self": seg_self, "seg_cross": seg_cross,
                "seg_txt": np.repeat(np.arange(L.n_seg, dtype=np.int32), L.txt_len),
                "mod_index": mod_index, "rows": rows, "cols": cols,
                "loss_mask": loss_mask}

    @property
    def fill(self) -> float:
        """有效填充率：真实 token / 自然总长。它**直接等于**线性层的算力利用率
        （填充 token 一样要过 MLP）。"""
        return sum(self.real_lens) / self.layout.total_len


# ── 装箱 ──────────────────────────────────────────────────────────────────────
def ffd(sizes: Sequence[int], budget: int) -> List[List[int]]:
    """First-Fit-Decreasing 装箱，与仓库 `navit_pack_strategy: ffd` 同策略。
    返回的是**下标**分组，不是尺寸，调用方据此取回样本。"""
    order = sorted(range(len(sizes)), key=lambda i: -sizes[i])
    bins: List[List[int]] = []
    used: List[int] = []
    for i in order:
        s = sizes[i]
        if s > budget:
            raise ValueError(f"单段 {s} > budget {budget}，装不下（提高 budget "
                             f"或降低该图分辨率）")
        for b, u in enumerate(used):
            if u + s <= budget:
                bins[b].append(i)
                used[b] += s
                break
        else:
            bins.append([i])
            used.append(s)
    return bins


class Packer:
    """把 (样本, token 数) 列表打成 pack，并按布局分组成 8 卡一步的批。"""

    def __init__(self, budget: int, quantum: int = 1024, txt_len: int = 512,
                 devices: int = 8):
        """txt_len 默认 512，且**填充的文本 token 照常参与 cross-attn**。

        这不是疏忽：`navit_text_trim_padding` 在 trainer/config.py:540 默认 False，
        因为训练去掉 512-pad 而 eval/采样/ARB 都带 pad，会让 cross-attn 的条件
        不一致 —— memory `[[navit-text-trim-train-eval-mismatch]]` 记录了 A/B
        实证：开 trim 时 eval_loss 冲高且拟合变差，关掉后单调下降并追平 ARB。
        所以 TPU 侧也照此口径：定长 512、不 mask 掉 pad。
        """
        if quantum % BLOCK:
            raise ValueError(f"quantum 必须是 {BLOCK} 的倍数（splash 块粒度），"
                             f"得到 {quantum}")
        if budget % quantum:
            raise ValueError(f"budget {budget} 必须能被 quantum {quantum} 整除，"
                             f"否则装箱余量凑不出合法的填充段")
        if budget % PACK_Q:
            raise ValueError(f"budget {budget} 必须是 {PACK_Q} 的倍数：pack 总长向上"
                             f"取整到 {PACK_Q}（自然长度），budget 作为容量上限必须"
                             f"装得下取整结果")
        self.budget, self.quantum, self.txt_len = budget, quantum, txt_len
        self.devices = devices
        self._carry: List[Pack] = []      # 上一轮凑不满 8 个的余量

    def build_packs(self, items: Sequence[object],
                    token_counts: Sequence[int],
                    grids: Sequence[Tuple[int, int]]) -> List[Pack]:
        """items/token_counts/grids 一一对应。grids[i] = (h, w) 且 h*w == token_counts[i]。"""
        if not (len(items) == len(token_counts) == len(grids)):
            raise ValueError("items / token_counts / grids 长度必须一致")
        for i, (n, (h, w)) in enumerate(zip(token_counts, grids)):
            if h * w != n:
                raise ValueError(f"第 {i} 项网格 {h}x{w} != token 数 {n}")
        q = [quantize_len(n, self.quantum) for n in token_counts]
        packs = []
        for group in ffd(q, self.budget):
            seg = [q[i] for i in group]
            real = [token_counts[i] for i in group]
            gr = [grids[i] for i in group]
            it = [items[i] for i in group]
            # **实段降序规范化**：段在 pack 里的先后是自由的（每段各算各的，
            # AdaLN 走 mod_index、RoPE 走 rows/cols，都不依赖段序），但它进了
            # Layout 的编译身份。排序是免费的去重：同一段长 multiset 只编译一次
            # （否则 FFD 的装箱顺序会让同一 multiset 拿到两种 Layout，白编译一次
            # 全模型）。纯填充段不参与排序，恒在末尾。
            order = sorted(range(len(seg)), key=lambda i: (-seg[i], i))
            seg = [seg[i] for i in order]
            real = [real[i] for i in order]
            gr = [gr[i] for i in order]
            it = [it[i] for i in order]
            # 自然长度：只补齐到 Σ实段 向上取整 PACK_Q，不再补齐到 budget。
            total = quantize_len(sum(seg), PACK_Q)
            rest = total - sum(seg)
            # RNG 兼容（见 Layout.t_slots）：旧版补齐到 budget，段序是"实段 + 旧
            # 填充段"整体降序，旧填充段落在 pad_old = #{实段 >= 旧余量} 处。
            rest_old = self.budget - sum(seg)
            t_slots = len(seg) + (1 if rest_old else 0)
            pad_old = sum(1 for s in seg if s >= rest_old) if rest_old else -1
            pad_idx = -1
            if rest:                       # 取整余量 -> 一个纯填充段（恒在末尾）
                seg.append(rest)
                real.append(0)
                gr.append((0, 0))
                it.append(None)
                pad_idx = len(seg) - 1
            # 新段序第 p 段 -> 旧槽位：实段 rank rj 在旧版的位置是
            # rj + (rj >= pad_old)；新填充位复用旧填充槽（它的 t 反正被丢）。
            # rest > 0 时必有 rest_old > 0（total <= budget），pad_old 不会越界。
            t_gather = []
            rj = 0
            for p in range(len(seg)):
                if p == pad_idx:
                    t_gather.append(pad_old)
                else:
                    t_gather.append(rj + (1 if 0 <= pad_old <= rj else 0))
                    rj += 1
            packs.append(Pack(Layout(total, tuple(seg), self.txt_len, pad_idx,
                                     t_slots, tuple(t_gather)), it, real, gr))
        return packs

    def plan_steps(self, packs: Sequence[Pack]) -> Tuple[List[List[Pack]], List[Pack]]:
        """按布局分组，每 `devices` 个 pack 凑成一步。

        返回 (steps, carry)。carry 是凑不满一步的余量，应带到下一轮再凑
        （用户范式是极大 epoch 连续训，顺延不丢样本）。
        """
        by_layout: Dict[Layout, List[Pack]] = defaultdict(list)
        for p in list(self._carry) + list(packs):
            by_layout[p.layout].append(p)
        steps, carry = [], []
        for layout, ps in by_layout.items():
            n = len(ps) // self.devices * self.devices
            for i in range(0, n, self.devices):
                steps.append(ps[i:i + self.devices])
            carry.extend(ps[n:])
        self._carry = carry
        return steps, carry


# ── Krea2 打包（text+image 同一条单流序列）─────────────────────────────────────
#
# ## 与 Anima 打包的差别
#
# Anima 的段只有图像 token（文本走 cross-attn 的定长槽）；Krea2 是单流 MMDiT，
# 每图的段 = [该图 text ; 该图 image]，预算花在同一条 combined 序列上。于是：
#
#   * 段的体积 = txt_q + img_q（text 槽量化到 txt_quantum=128、image 槽量化到
#     quantum），FFD 按 combined 体积装箱；
#   * **编译身份是二元组**（combined 段长元组, 各实段的 text 槽长元组）——主序列
#     块对角内核看前者，txtfusion refiner 内核看后者；
#   * 取整余量段（round 到 PACK_Q 的余量）是纯填充（无 text 槽），与 Anima 同款；
#   * 文本侧不需要 `navit_text_trim_padding` 那种开关：Krea2 navit 本来就只打包
#     有效 caption token（量化槽内的一小截填充由精细 segment_ids 隔离），
#     训练/eval 口径天然一致。
@dataclass(frozen=True)
class K2Layout:
    """Krea2 一个 pack 的**编译身份**。

    total_len : 布局**自然长度** = round_up(Σ实段, PACK_Q)（budget 只是 FFD 装箱
        容量上限，语义同 Anima Layout）
    seg_lens  : combined 段长（实段降序；取整余量的纯填充段恒在末尾、无 text 槽），
        sum == total_len
    txt_segs  : 各**实**段的 text 槽长（与 seg_lens 前 len(txt_segs) 项一一对应）
    img_segs  : 各**实**段的 image 槽长（seg = txt + img）
    t_slots / t_gather : RNG 兼容信息（compare=False，语义同 Anima Layout）
    """
    total_len: int
    seg_lens: Tuple[int, ...]
    txt_segs: Tuple[int, ...]
    img_segs: Tuple[int, ...]
    t_slots: int = field(default=0, compare=False)
    t_gather: Tuple[int, ...] = field(default=(), compare=False)

    def __post_init__(self):
        if sum(self.seg_lens) != self.total_len:
            raise ValueError(f"段长和 {sum(self.seg_lens)} != total_len {self.total_len}")
        bad = [n for n in self.seg_lens if n % BLOCK]
        if bad:
            raise ValueError(f"段长必须是 {BLOCK} 的倍数（splash 块粒度），越界 {bad[:4]}")
        if not (len(self.txt_segs) == len(self.img_segs) <= len(self.seg_lens)):
            raise ValueError(f"txt/img 槽数 ({len(self.txt_segs)}/{len(self.img_segs)}) "
                             f"与段数 {len(self.seg_lens)} 对不上")
        if len(self.seg_lens) - len(self.txt_segs) > 1:
            raise ValueError("至多一个纯填充段（取整余量，恒在末尾）")
        if self.t_slots:                      # 校验与 Anima Layout 对称
            if len(self.t_gather) != len(self.seg_lens) or \
                    any(not 0 <= g < self.t_slots for g in self.t_gather):
                raise ValueError(f"t_gather 必须是 {len(self.seg_lens)} 个 [0, "
                                 f"{self.t_slots}) 内的下标，得到 {self.t_gather}")
        for i, (tq, iq) in enumerate(zip(self.txt_segs, self.img_segs)):
            if tq % BLOCK or iq % BLOCK:
                raise ValueError(f"第 {i} 段槽长未对齐 {BLOCK}：txt={tq} img={iq}")
            if tq + iq != self.seg_lens[i]:
                raise ValueError(f"第 {i} 段 txt+img={tq + iq} != 段长 {self.seg_lens[i]}")

    @property
    def n_seg(self) -> int:
        return len(self.seg_lens)

    @property
    def real_seg_lens(self) -> Tuple[int, ...]:
        """实段（combined）段长——反向块上限（seg_cap）只统计它们（同 Anima Layout）。"""
        return self.seg_lens[:len(self.txt_segs)]

    @property
    def n_img_tokens(self) -> int:
        """图像流总长（loss 侧）：各实段 image 槽之和。"""
        return sum(self.img_segs)

    @property
    def n_txt_tokens(self) -> int:
        return sum(self.txt_segs)

    def static_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        """(txt_pos, img_pos)：两流在 combined 序列中的位置。

        **只依赖布局字段**（与 pack 内容无关）——这正是它能进 make_grad_fn 的
        编译期闭包的原因；K2Pack.index_arrays 复用同一份，两边永远不会漂。
        """
        txt_pos, img_pos = [], []
        off = 0
        for i, seg in enumerate(self.seg_lens):
            if i < len(self.txt_segs):
                tq, iq = self.txt_segs[i], self.img_segs[i]
                txt_pos += list(range(off, off + tq))
                img_pos += list(range(off + tq, off + tq + iq))
            off += seg
        return (np.asarray(txt_pos, np.int32), np.asarray(img_pos, np.int32))


@dataclass
class K2Pack:
    """Krea2 一个 pack 的运行时内容。"""
    layout: K2Layout
    items: List[object] = field(default_factory=list)       # 各实段的样本引用
    real_img_lens: List[int] = field(default_factory=list)  # 各实段真实 image token 数
    real_txt_lens: List[int] = field(default_factory=list)  # 各实段真实 caption token 数
    grids: List[Tuple[int, int]] = field(default_factory=list)

    def index_arrays(self) -> Dict[str, np.ndarray]:
        """造出喂给 attention / krea2_jax / loss 的全部索引数组。

        combined 流 [B]：rows/cols（text/填充位=0）、mod_index（token→图，含填充
            段映射到宿主段）、seg_self（精细段号；一切填充位 = PAD_SEG，彼此可见，
            行不空 —— splash SegmentIds 的硬约束）
        图像流 [Σimg_q]：rows/cols（RoPE/ΔFM 用）、mod_index、loss_mask（1=真 token）
        文本流 [Σtxt_q]：txt_fine（精细段号，填充位 PAD_SEG）
        静态位图：txt_pos / img_pos（两流在 combined 序列中的位置，**进编译身份**，
            由 make_grad_fn 闭包持有，不走运行时数组）
        """
        L = self.layout
        B = L.total_len
        rows = np.zeros(B, np.int32)
        cols = np.zeros(B, np.int32)
        mod_index = np.empty(B, np.int32)
        seg_self = np.full(B, PAD_SEG, np.int32)
        i_rows = np.zeros(L.n_img_tokens, np.int32)
        i_cols = np.zeros(L.n_img_tokens, np.int32)
        i_mod = np.empty(L.n_img_tokens, np.int32)
        i_mask = np.zeros(L.n_img_tokens, np.float32)
        t_fine = np.full(L.n_txt_tokens, PAD_SEG, np.int32)
        txt_pos, img_pos = self.layout.static_positions()
        off = i_off = t_off = 0
        for i, seg in enumerate(L.seg_lens):
            mod_index[off:off + seg] = i
            if i < len(L.txt_segs):                     # 实段
                tq, iq = L.txt_segs[i], L.img_segs[i]
                rt = self.real_txt_lens[i]
                ri = self.real_img_lens[i]
                if not (0 < rt <= tq and 0 < ri <= iq):
                    raise ValueError(f"第 {i} 段实长越界：txt {rt}/{tq} img {ri}/{iq}")
                t_fine[t_off:t_off + rt] = i
                h, w = self.grids[i]
                if h * w != ri:
                    raise ValueError(f"第 {i} 段网格 {h}x{w} != 真实 token 数 {ri}")
                img_off = off + tq
                seg_self[off:off + rt] = i
                seg_self[img_off:img_off + ri] = i
                rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
                rows[img_off:img_off + ri] = rr.reshape(-1)
                cols[img_off:img_off + ri] = cc.reshape(-1)
                i_mod[i_off:i_off + iq] = i
                i_rows[i_off:i_off + ri] = rr.reshape(-1)
                i_cols[i_off:i_off + ri] = cc.reshape(-1)
                i_mask[i_off:i_off + ri] = 1.0
                i_off += iq
                t_off += tq
            off += seg
        return {"rows_c": rows, "cols_c": cols, "mod_index_c": mod_index,
                "seg_self": seg_self, "txt_fine": t_fine,
                "rows_i": i_rows, "cols_i": i_cols, "mod_index_i": i_mod,
                "loss_mask": i_mask,
                "txt_pos": np.asarray(txt_pos, np.int32),
                "img_pos": np.asarray(img_pos, np.int32)}

    @property
    def fill(self) -> float:
        """有效填充率 = （真实 image + 真实 text）/ 自然总长。它**直接等于**线性层
        的算力利用率（填充 token 一样要过 MLP）。"""
        return (sum(self.real_img_lens) + sum(self.real_txt_lens)) / self.layout.total_len


class K2Packer:
    """Krea2 版装箱/成步：与 Packer 同策略（FFD + 布局分组凑 8 卡一步）。"""

    def __init__(self, budget: int, quantum: int = 1024, txt_quantum: int = BLOCK,
                 devices: int = 8):
        if quantum % BLOCK or txt_quantum % BLOCK:
            raise ValueError(f"quantum/txt_quantum 必须是 {BLOCK} 的倍数，"
                             f"得到 {quantum}/{txt_quantum}")
        if budget % quantum:
            raise ValueError(f"budget {budget} 必须能被 quantum {quantum} 整除")
        if budget % PACK_Q:
            raise ValueError(f"budget {budget} 必须是 {PACK_Q} 的倍数：pack 总长向上"
                             f"取整到 {PACK_Q}（自然长度），budget 作为容量上限必须"
                             f"装得下取整结果")
        self.budget, self.quantum, self.txt_quantum = budget, quantum, txt_quantum
        self.devices = devices
        self._carry: List[K2Pack] = []

    def build_packs(self, items: Sequence[object],
                    token_counts: Sequence[int],
                    txt_lens: Sequence[int],
                    grids: Sequence[Tuple[int, int]]) -> List[K2Pack]:
        """items/token_counts(image)/txt_lens(caption)/grids 一一对应。"""
        if not (len(items) == len(token_counts) == len(txt_lens) == len(grids)):
            raise ValueError("items / token_counts / txt_lens / grids 长度必须一致")
        for n, tl, (h, w) in zip(token_counts, txt_lens, grids):
            if h * w != n:
                raise ValueError(f"网格 {h}x{w} != token 数 {n}")
            if tl < 1:
                raise ValueError("Krea2 要求每图至少 1 个有效 caption token"
                                 "（krea2_modeling.py:1006 同款 fail-fast）")
        vols = [quantize_len(n, self.quantum) + quantize_len(tl, self.txt_quantum)
                for n, tl in zip(token_counts, txt_lens)]
        packs = []
        for group in ffd(vols, self.budget):
            seg, tseg, iseg = [], [], []
            real_i, real_t, gr, it = [], [], [], []
            for i in group:
                iq = quantize_len(token_counts[i], self.quantum)
                tq = quantize_len(txt_lens[i], self.txt_quantum)
                seg.append(tq + iq)
                tseg.append(tq)
                iseg.append(iq)
                real_i.append(token_counts[i])
                real_t.append(txt_lens[i])
                gr.append(grids[i])
                it.append(items[i])
            # 段序规范化：**实段**降序、纯填充段恒在末尾。段在 pack 里的先后是
            # 自由的（调制走 mod_index、RoPE 走 rows/cols），但它进编译身份；
            # 规范化是免费的去重（同一段长 multiset 只编译一次），且填充段恒
            # 末尾让 K2Layout 的 txt/img 槽对齐规则（前 n 项为实段）天然成立。
            order = sorted(range(len(seg)), key=lambda i: (-seg[i], i))
            seg = [seg[i] for i in order]
            tseg = [tseg[i] for i in order]
            iseg = [iseg[i] for i in order]
            real_i = [real_i[i] for i in order]
            real_t = [real_t[i] for i in order]
            gr = [gr[i] for i in order]
            it = [it[i] for i in order]
            # 自然长度：只补齐到 Σ实段 向上取整 PACK_Q，不再补齐到 budget
            # （combined 段含 128 量化文本槽，余量段 < PACK_Q 是常态）。
            total = quantize_len(sum(seg), PACK_Q)
            # RNG 兼容（见 Layout.t_slots）：旧版填充段恒在末尾，实段槽位不变，
            # 只需记得旧槽数（含旧填充段），新填充位复用旧填充槽。
            rest_old = self.budget - sum(seg)
            t_slots = len(seg) + (1 if rest_old else 0)
            pad_old = len(seg) if rest_old else -1
            rest = total - sum(seg)
            t_gather = list(range(len(seg)))
            if rest:
                seg.append(rest)                # 纯填充段（无 text 槽，恒在末尾）
                t_gather.append(pad_old)
            packs.append(K2Pack(
                K2Layout(total, tuple(seg), tuple(tseg), tuple(iseg),
                         t_slots, tuple(t_gather)),
                it, real_i, real_t, gr))
        return packs

    def plan_steps(self, packs: Sequence[K2Pack]) -> Tuple[List[List[K2Pack]], List[K2Pack]]:
        """与 Packer.plan_steps 同语义：(steps, carry)，carry 跨轮顺延不丢样本。"""
        by_layout: Dict[K2Layout, List[K2Pack]] = defaultdict(list)
        for p in list(self._carry) + list(packs):
            by_layout[p.layout].append(p)
        steps, carry = [], []
        for layout, ps in by_layout.items():
            n = len(ps) // self.devices * self.devices
            for i in range(0, n, self.devices):
                steps.append(ps[i:i + self.devices])
            carry.extend(ps[n:])
        self._carry = carry
        return steps, carry


def report_k2(packs: Sequence[K2Pack], devices: int = 8, capacity: int = 0) -> str:
    """Krea2 打包质量报告（口径同 report()，填充率含 text 流）。

    有效填充率的分母是**自然总长**（= 线性层算力利用率）；`capacity`（全局 budget
    /卡）给定时额外打印容量利用率 = Σ自然总长 / (pack 数 × capacity)。
    """
    if not packs:
        return "（无 pack）"
    layouts = {p.layout for p in packs}
    by = defaultdict(int)
    for p in packs:
        by[p.layout] += 1
    full = sum(c // devices * devices for c in by.values())
    fill = sum(sum(p.real_img_lens) + sum(p.real_txt_lens) for p in packs) \
        / sum(p.layout.total_len for p in packs)
    cap = (f" | 容量利用率 "
           f"{sum(p.layout.total_len for p in packs) / (len(packs) * capacity):.1%}"
           if capacity else "")
    lines = [f"pack {len(packs)} 个 | 布局 {len(layouts)} 种 "
             f"(≈{len(layouts)}次全模型编译) | 有效填充率 {fill:.1%}{cap} | "
             f"可成步 {full}/{len(packs)} 个 pack = {full // devices} 步"]
    for layout, c in sorted(by.items(), key=lambda kv: -kv[1]):
        lines.append(f"  seg{str(layout.seg_lens):<40} txt{str(layout.txt_segs):<20} "
                     f"x{c:<4} 步 {c // devices} 余 {c % devices}")
    return "\n".join(lines)


# ── 分桶（ragged / 批维）调度 ─────────────────────────────────────────────────
#
# ## 与打包调度的关系
#
# 打包路线的**编译身份是段长元组**（如 `(10240, 9216, 4096, 4096, 3072, 2048)`），
# 分桶路线的编译身份只是**一个整数 L**。这个差别的方向是确定的：
#
#   * 打包的等价类更细 -> 要凑齐 8 个**同元组**的 pack（≈ 8*每 pack 图数 张同构图），
#     分桶 G=1 只要 8 张**同 L** 的图；
#   * 反过来，打包能把不同大小的图混进同一个 pack，分桶不能（一步一个 L）。
#
# **哪一边划算完全取决于数据集的 token 数分布，不能一般化。** token 数越集中，
# 分桶越占便宜（还能把 Q 降到 128 把填充率吃满）；越分散，打包的混装能力越值钱。
# 用 `tests/enum_dataset_routes.py` 对**实际要训的**数据集算一遍再定 —— 零配额、
# 秒级。不要拿别的数据集的数字外推：本仓库 memory `[[dont-infer-dataset-provenance]]`
# 记着这条教训（数据来源与预处理链只有用户知道）。
#
# 分散度高时缓解手段（尚未实现，仅记录方向）：跨桶做梯度累积 —— 一个优化步由若干
# 个不同 L 的微步组成，各自是独立的编译产物，LoRA 梯度累加后再更新。这样"凑不满
# 8 张同 L"就不再卡住优化步，代价是每个 L 多一份编译（可被持久化编译缓存摊掉）。
#
# 顺延语义与 `Packer.plan_steps` 一致：凑不满一步的余量带到下一轮，不丢样本。
@dataclass(frozen=True)
class Bucket:
    """一个分桶步的**编译身份**：桶长 L + 每卡图数 G + 文本槽长。"""
    length: int
    imgs: int
    txt_len: int = 512

    def __post_init__(self):
        for name, v in (("length", self.length), ("txt_len", self.txt_len)):
            if v % BLOCK:
                raise ValueError(f"{name}={v} 必须是 {BLOCK} 的倍数（splash 块粒度）")
        if self.imgs < 1:
            raise ValueError(f"imgs 必须 >= 1，得到 {self.imgs}")


@dataclass
class BucketStep:
    """一步的运行时内容：devices*imgs 张同 L 的图。"""
    bucket: Bucket
    items: List[object] = field(default_factory=list)
    real_lens: List[int] = field(default_factory=list)
    grids: List[Tuple[int, int]] = field(default_factory=list)

    def index_arrays(self, devices: int) -> Dict[str, np.ndarray]:
        """造出喂给 attention.make_bucket_attn / anima_jax.forward_ragged 的数组。

        形状都是 [devices, imgs, L]（seg/rows/cols/loss_mask），第 0 维是设备维。

        seg: 0=真 token、1=尾部量化填充。填充给**同一个**非零段号而不是逐 token
        独立，保证填充行彼此可见、softmax 分母不为 0（splash 的 SegmentIds 文档
        对全 0 行有明确警告）。
        """
        B, L = self.bucket, self.bucket.length
        n = devices * B.imgs
        if len(self.real_lens) != n:
            raise ValueError(f"一步要 {n} 张图（{devices} 卡 x {B.imgs}），"
                             f"得到 {len(self.real_lens)}")
        seg = np.ones((n, L), np.int32)
        rows = np.zeros((n, L), np.int32)
        cols = np.zeros((n, L), np.int32)
        loss_mask = np.zeros((n, L), np.float32)
        for i, (r, (h, w)) in enumerate(zip(self.real_lens, self.grids)):
            if h * w != r:
                raise ValueError(f"第 {i} 张网格 {h}x{w} != 真实 token 数 {r}")
            if r > L:
                raise ValueError(f"第 {i} 张 {r} token > 桶长 {L}")
            seg[i, :r] = 0
            loss_mask[i, :r] = 1.0
            rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
            rows[i, :r] = rr.reshape(-1)
            cols[i, :r] = cc.reshape(-1)
        rs = lambda a: a.reshape(devices, B.imgs, L)
        return {"seg": rs(seg), "rows": rs(rows), "cols": rs(cols),
                "loss_mask": rs(loss_mask)}

    @property
    def fill(self) -> float:
        """有效填充率 = 真实 token / 填充后 token。直接等于线性层的算力利用率。"""
        return sum(self.real_lens) / (len(self.real_lens) * self.bucket.length)


class BucketPlanner:
    """把 (样本, token 数) 列表按量化长度分桶，凑成 devices*imgs 一步。"""

    def __init__(self, quantum: int = 128, per_device: int = 1,
                 txt_len: int = 512, devices: int = 8,
                 max_length: Optional[int] = None):
        """`quantum` 默认 128（= splash 块粒度，能取到的最细量化）。

        打包路线要把它提到 1024 是为了压段长元组的组合数；分桶路线的编译身份只有
        一个 L，多几个桶只是多几次编译（且各自独立、可被持久化编译缓存摊掉），
        所以这里默认取最细的 128，把填充率吃满。
        """
        if quantum % BLOCK:
            raise ValueError(f"quantum 必须是 {BLOCK} 的倍数（splash 块粒度），得到 {quantum}")
        self.quantum, self.per_device, self.txt_len = quantum, per_device, txt_len
        self.devices, self.max_length = devices, max_length
        self._carry: Dict[int, List[Tuple]] = defaultdict(list)

    def plan_steps(self, items: Sequence[object], token_counts: Sequence[int],
                   grids: Sequence[Tuple[int, int]]
                   ) -> Tuple[List[BucketStep], List[Tuple]]:
        """返回 (steps, carry)。carry 是凑不满一步的余量，自动带到下一次调用。"""
        if not (len(items) == len(token_counts) == len(grids)):
            raise ValueError("items / token_counts / grids 长度必须一致")
        by = defaultdict(list)
        for k, v in self._carry.items():
            by[k].extend(v)
        for it, n, (h, w) in zip(items, token_counts, grids):
            if h * w != n:
                raise ValueError(f"网格 {h}x{w} != token 数 {n}")
            L = quantize_len(n, self.quantum)
            if self.max_length is not None and L > self.max_length:
                raise ValueError(f"图 {n} token -> 桶长 {L} 超过上限 "
                                 f"{self.max_length}（降分辨率或提高上限）")
            by[L].append((it, n, (h, w)))

        per_step = self.devices * self.per_device
        steps: List[BucketStep] = []
        self._carry = defaultdict(list)
        for L, entries in by.items():
            k = len(entries) // per_step
            for i in range(k):
                chunk = entries[i * per_step:(i + 1) * per_step]
                steps.append(BucketStep(
                    Bucket(L, self.per_device, self.txt_len),
                    [c[0] for c in chunk], [c[1] for c in chunk],
                    [c[2] for c in chunk]))
            self._carry[L] = entries[k * per_step:]
        carry = [e for v in self._carry.values() for e in v]
        return steps, carry


def report_buckets(steps: Sequence[BucketStep], carry_n: int = 0) -> str:
    """分桶调度质量报告。桶数决定编译次数，填充率决定线性层算力利用率。"""
    if not steps:
        return f"（无可成步的样本；顺延 {carry_n} 张）"
    by: Dict[Bucket, int] = defaultdict(int)
    for s in steps:
        by[s.bucket] += 1
    real = sum(sum(s.real_lens) for s in steps)
    pad = sum(len(s.real_lens) * s.bucket.length for s in steps)
    lines = [f"步 {len(steps)} | 桶 {len(by)} 种 (≈{len(by)} 次全模型编译) | "
             f"有效填充率 {real / pad:.1%} | 顺延 {carry_n} 张"]
    for b, c in sorted(by.items(), key=lambda kv: -kv[1]):
        lines.append(f"  L={b.length:<6} 每卡 {b.imgs} 图  x{c} 步")
    return "\n".join(lines)


# ── 诊断 ──────────────────────────────────────────────────────────────────────
def report(packs: Sequence[Pack], devices: int = 8, capacity: int = 0) -> str:
    """打包质量报告。**上真机前先看这个**：布局数决定编译成本，填充率决定
    线性层算力利用率，成步率决定有多少样本会被顺延。

    有效填充率的分母是**自然总长**（round_up(Σ实段, PACK_Q)）；`capacity`
    （budget/卡）给定时额外打印容量利用率 = Σ自然总长 / (pack 数 × capacity)，
    用于对照"装箱容量被吃掉多少"。
    """
    if not packs:
        return "（无 pack）"
    layouts = {p.layout for p in packs}
    by = defaultdict(int)
    for p in packs:
        by[p.layout] += 1
    full = sum(c // devices * devices for c in by.values())
    fill = sum(sum(p.real_lens) for p in packs) / sum(p.layout.total_len for p in packs)
    cap = (f" | 容量利用率 "
           f"{sum(p.layout.total_len for p in packs) / (len(packs) * capacity):.1%}"
           if capacity else "")
    lines = [f"pack {len(packs)} 个 | 布局 {len(layouts)} 种 "
             f"(≈{len(layouts)}次全模型编译) | 有效填充率 {fill:.1%}{cap} | "
             f"可成步 {full}/{len(packs)} 个 pack = {full // devices} 步"]
    for layout, c in sorted(by.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {str(layout.seg_lens):<44} x{c:<4} "
                     f"步 {c // devices} 余 {c % devices}")
    return "\n".join(lines)
