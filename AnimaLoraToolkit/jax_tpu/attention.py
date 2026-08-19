"""NaViT 块对角注意力后端（TPU splash / CPU 稠密参考）。

把「段几何」与「模型前向」解耦：`anima_jax.forward_packed` 只接收
`self_attn_fn(q,k,v)` / `cross_attn_fn(q,k,v)` 两个闭包，段信息全部闭在这里。

## 为什么是编译期 mask 而不是运行时 mask / segment_ids

真机三条实测（kaggle_job/output/anima-tpu-splash-probe、anima-tpu-arch-probe）：

  * T1  编译期块对角 mask：106.81ms vs 全通 423.20ms -> 实测比 **0.252**，理论 0.250
        —— 跳块真实生效。
  * E2  运行时 jax.Array mask：23.41/41.12ms -> 比 **0.569**（理论 0.250）
        —— **只部分跳块**，且 48/16/8 head 都放不下（HBM），只有 4 head 能跑。
  * D3  `segment_ids` 参数：数值正确，但块稀疏由编译期 mask 驱动，
        这条**大概率不跳块**，只作语义保底。

所以走编译期 mask。代价是每种段长布局要单独编译一次全模型（真机 D1 实测首步
60s），因此**布局数必须有界** —— 由 packing.py 的量化打包负责（Q=1024 时
budget=32768 下只有 4 种布局，填充率仍 94.8%）。

## 为什么用 _ComputableMask 而不是 NumpyMask

NumpyMask 要物化 (budget, budget) 的 bool 数组：budget=32768 时是 1GB numpy，
host 侧就撑不住。`_ComputableMask` 只在 MaskInfo 构造时按 block 切片惰性求值
（jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py
的 `_ComputableMask.__getitem__`：`mask_function(rows[:,None], cols[None,:])`）。

**段长必须 128 对齐**：对齐后每个 (128,128) 块要么整块在段内、要么整块在段外，
partial block 恒为 0（真机 B1：对齐 partial=0 / 未对齐 partial=429），
于是 `mask_function` 只在 host 侧被 numpy 调用，不需要进 kernel。
不对齐不会报错，只会产生 partial block -> 跳块率变差 + mask_function 被拖进
kernel（那时它必须是 jax-traceable 的）。这里显式 fail-fast，不让它静默劣化。

## splash 的两条硬约束（漏了都是静默错）

  1. **不内置 1/sqrt(d)** —— 必须自己预缩放 q，漏了会静默改变 softmax 温度。
  2. **block 大小必须整除序列长**（splash_attention_mask_info.py 硬 raise）。
"""

from __future__ import annotations

import functools
import math
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

BLOCK = 128          # splash 的 TPU 块粒度；段长必须是它的倍数
# 反向块大小：真机 arch_probe H1 实测 L=16384 下
#   默认(128/128, fused=False) 288ms | dkv=512 fused 157ms | dkv=1024 fused 142ms
#   dkv=2048 fused OOM
# -> 取 1024 + use_fused_bwd_kernel，比默认快 2.03x。整除不了时逐级退让。
BWD_BLOCK_PREF = (1024, 512, 256, 128)


# ── 段几何 ────────────────────────────────────────────────────────────────────
def segment_ids(seg_lens: Sequence[int]) -> np.ndarray:
    """[ΣL] token -> 段号。段号也直接用作 AdaLN 的 mod_index。"""
    return np.repeat(np.arange(len(seg_lens), dtype=np.int32),
                     np.asarray(seg_lens, dtype=np.int64))


def _check_aligned(seg_lens: Sequence[int], what: str) -> None:
    bad = [(i, n) for i, n in enumerate(seg_lens) if n % BLOCK]
    if bad:
        raise ValueError(
            f"{what} 段长必须是 {BLOCK} 的倍数（splash 块粒度），"
            f"越界段 {bad[:4]}{'...' if len(bad) > 4 else ''}。\n"
            f"不对齐不会报错但会产生 partial block：跳块率劣化，且 mask_function "
            f"会被拖进 kernel（真机 B1：对齐 partial=0 / 未对齐 partial=429）。\n"
            f"打包侧应把段长量化到 {BLOCK} 的倍数（packing.quantize，"
            f"实测代价约 0.26% token）。")


# ── 惰性块对角 mask ───────────────────────────────────────────────────────────
def _anima_jax():
    """兼容两种用法：作为包 `jax_tpu.attention` 导入，或按 tests/ 的老办法把
    `jax_tpu/` 直接放进 sys.path 后 `import anima_jax`。

    第三种情况是 Kaggle：build_job.py 会把 anima_jax.py 与本文件拼进**同一个**
    模块，此时两边都 import 不到，得回退到当前模块自身的命名空间。
    """
    try:
        from . import anima_jax as A          # 包内相对导入
        return A
    except ImportError:
        pass
    try:
        import anima_jax as A                 # sys.path 平铺（tests/ 走这条）
        return A
    except ImportError:
        import sys
        return sys.modules[__name__]          # 单文件拼接（Kaggle job）


def _import_splash():
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_kernel as sk,
        splash_attention_mask as sm,
    )
    return sk, sm


def block_diag_mask(q_seg: np.ndarray, kv_seg: np.ndarray):
    """段对齐的块对角 mask（q 与 kv 段号相同才可见）。矩形亦可（cross-attn）。

    **继承 `Mask` 而不是 `_ComputableMask`**，两条理由：

      1. `_ComputableMask` 的语义是"把 mask_function 编进 kernel 里现算"
         （splash_attention_mask_info.py:669 `hasattr(unique_mask,'mask_function')`）。
         段号是不等长段的查表，闭包会把两个 int32[S] 数组带进 pallas kernel，
         直接报 `captures constants [i32[768], i32[768]]`。
      2. `NumpyMask` 会物化 (q_len, kv_len) 的 bool 数组——budget=32768 时是 1GB
         numpy，host 侧不划算。

    继承 `Mask` 只实现惰性 `__getitem__`：MaskInfo 构造时按 (128,128) 块切片调用，
    每次只算一个小块。段长 128 对齐时**没有 partial block**（真机 B1 实证），
    所以 `partial_mask_blocks` 是空的，什么都不会被搬进 kernel。
    """
    _, sm = _import_splash()
    shape = (int(q_seg.shape[0]), int(kv_seg.shape[0]))

    class _BlockDiag(sm.Mask):
        @property
        def shape(self):
            return shape

        def __getitem__(self, idx):
            qs, ks = idx
            if not isinstance(qs, slice) or not isinstance(ks, slice):
                raise NotImplementedError(f"只支持切片索引，收到 {idx}")
            return (q_seg[qs][:, None] == kv_seg[ks][None, :])

        def __eq__(self, other):
            return (isinstance(other, _BlockDiag)
                    and np.array_equal(q_seg, other._q)
                    and np.array_equal(kv_seg, other._kv))

        def __hash__(self):
            return hash((q_seg.tobytes(), kv_seg.tobytes()))

    m = _BlockDiag()
    m._q, m._kv = q_seg, kv_seg
    return m


def _block_sizes(q_len: int, kv_len: int, seg_cap: Optional[int] = None):
    """挑一组整除 (q_len, kv_len) 的块大小；前向恒 128，反向尽量取大。

    `seg_cap` = 最短的段长。**反向块不能超过它**，否则一个反向块会横跨两个段，
    块内既有可见又有不可见 -> partial mask block，要被物化进 kernel。
    本地实测（jacknife 画像：图像段 2048、文本段 512）：

        cross-attn  block=(1024,1024)  partial_mask_blocks = 2
        cross-attn  block=(1024, 512)  partial_mask_blocks = 0
        self-attn   block=(1024,1024)  partial_mask_blocks = 0

    真机 B1 那条"128 对齐 -> partial=0"只覆盖**前向**（前向块恒 128 < 段长）。
    cross-attn 的 kv 段只有 txt_len=512，按 kv_len=n_seg*512 挑出来的 1024 会越界。
    """
    sk, _ = _import_splash()
    if q_len % BLOCK or kv_len % BLOCK:
        raise ValueError(f"序列长必须被 {BLOCK} 整除，得到 q={q_len} kv={kv_len}"
                         f"（splash_attention_mask_info.py 对此硬 raise）")
    pref = BWD_BLOCK_PREF if seg_cap is None else \
        tuple(b for b in BWD_BLOCK_PREF if b <= seg_cap) or (BLOCK,)
    bwd_q = next(b for b in pref if q_len % b == 0)
    bwd_kv = next(b for b in pref if kv_len % b == 0)
    return sk.BlockSizes(
        block_q=BLOCK, block_kv=BLOCK, block_kv_compute=BLOCK,
        block_q_dkv=bwd_q, block_kv_dkv=bwd_kv, block_kv_dkv_compute=bwd_kv,
        use_fused_bwd_kernel=True,
    )


# ── 后端工厂 ──────────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=64)
def _splash_kernel(q_seg_bytes: bytes, kv_seg_bytes: bytes, n_len: int,
                   kv_len: int, num_heads: int, interpret: bool,
                   seg_cap: Optional[int] = None):
    """按布局缓存已构造的 splash 可调用对象。

    真机 arch_probe G2 实测：**splash 自己没有缓存**（同一布局重复构造
    735ms -> 716ms，几乎没省），必须在调用侧自己缓存。这里用 lru_cache，
    键是段号数组的字节串（布局的完整身份）。
    """
    sk, sm = _import_splash()
    q_seg = np.frombuffer(q_seg_bytes, dtype=np.int32)
    kv_seg = np.frombuffer(kv_seg_bytes, dtype=np.int32)
    mask = sm.MultiHeadMask(masks=[block_diag_mask(q_seg, kv_seg)] * num_heads)
    return sk.make_splash_mha(mask, head_shards=1, q_seq_shards=1,
                              block_sizes=_block_sizes(n_len, kv_len, seg_cap),
                              interpret=interpret)


def bind_segments(attn: Callable, fine_q, fine_kv) -> Callable:
    """把运行时 segment_ids 绑到 attn 上，得到 `forward_packed` 要的三参签名。

    **在被 trace 的函数里只能调这个，不能调 `make_splash_attn`**：后者会构造
    splash kernel，其 MaskInfo 是 jax 数组；在 trace 内构造会让它们变成 tracer，
    又被 `_splash_kernel` 的 lru_cache 缓存下来，泄漏到下一次 trace ——
    真机不报错，本地 check_train_loop 抓到的是
    `UnexpectedTracerError: ... int8[1,16,8] ... escape the scope`。
    kernel 构造还很贵（真机实测 735ms），本就该提到 trace 外只做一次。
    """
    return lambda q, k, v: attn(q, k, v, fine_q, fine_kv)


def make_splash_attn(q_seg_lens: Sequence[int], kv_seg_lens: Sequence[int],
                     num_heads: int, head_dim: int,
                     fine_q=None, fine_kv=None, interpret: bool = False
                     ) -> Callable:
    """构造 (q,k,v)->out 的块对角注意力。q/k/v 布局 [S, H, D]（与模型侧一致）。

    ## 两级 mask —— 这是整套 NaViT-on-TPU 设计的关键

    `q_seg_lens` / `kv_seg_lens` 是**粗粒度**段长（量化到 1024，故 128 对齐），
    它们进编译期 mask，负责**跳块**（真机 T1 实测比 0.252 / 理论 0.250）。
    粗粒度 -> 布局种类少 -> 全模型编译次数少（Q=1024、budget=32768 时只有 4 种）。

    `fine_q` / `fine_kv` 是**运行时** segment id 数组（jnp，[S]/[T]），负责
    **精确边界**：段内的量化填充 token 归到独立段号，于是同段的真 token 看不见
    它们。它是运行时输入、形状固定，**不触发重编译**。

    依据是 splash 的契约（splash_attention_kernel.SegmentIds 文档原文）：
    "The static mask is and-ed with the segment id mask to form the actual
    attention mask." 本地 interpret 实测：加了之后段内真 token 受填充影响
    max_abs=0（不加时是 5.4，即静默污染）。

    注意该文档同时警告：**不能有整行全 0**（softmax 分母为 0）。所以
      * 自注意力：填充 token 用同一个独立段号 -> 它们彼此可见，行不空；
      * cross-attn：填充 token 必须沿用**宿主图**的段号（文本侧没有"填充段"
        给它们配对），其输出反正会被 loss mask 丢掉。
    这两条由 packing.py 构造，本函数不猜。

    自注意力：kv_seg_lens = q_seg_lens。
    cross-attn：q 是图像段、kv 是文本段，段数必须相同（第 i 图看第 i 段文本）。
    """
    import jax.numpy as jnp
    sk, _ = _import_splash()

    _check_aligned(q_seg_lens, "q")
    _check_aligned(kv_seg_lens, "kv")
    if len(q_seg_lens) != len(kv_seg_lens):
        raise ValueError(f"q 段数 {len(q_seg_lens)} != kv 段数 {len(kv_seg_lens)}；"
                         f"块对角要求逐段一一对应")
    if (fine_q is None) != (fine_kv is None):
        raise ValueError("fine_q / fine_kv 必须同时给或同时不给")
    q_seg, kv_seg = segment_ids(q_seg_lens), segment_ids(kv_seg_lens)
    # 反向块上限取最短段：跨段的反向块会产生 partial mask block（见 _block_sizes）
    seg_cap = min(min(q_seg_lens), min(kv_seg_lens))
    kernel = _splash_kernel(q_seg.tobytes(), kv_seg.tobytes(),
                            int(q_seg.shape[0]), int(kv_seg.shape[0]),
                            num_heads, interpret, seg_cap)
    scale = head_dim ** -0.5

    def attn(q, k, v, seg_q=fine_q, seg_kv=fine_kv):
        """seg_q/seg_kv 可在**调用时**给（见 bind_segments），也可在构造时给。
        构造时给只适合 trace 外用具体数组的场合（如探针脚本）。"""
        # splash **不内置 1/sqrt(d)**，必须显式预缩放 q（漏了是静默错：
        # softmax 温度变了，前几轮本地实测 rel 差到 9.5）。
        qs = (q.astype(jnp.float32) * scale).astype(q.dtype).transpose(1, 0, 2)
        kw = {} if seg_q is None else {
            "segment_ids": sk.SegmentIds(q=seg_q, kv=seg_kv)}
        o = kernel(qs, k.transpose(1, 0, 2), v.transpose(1, 0, 2), **kw)
        return o.transpose(1, 0, 2)

    return attn


# ── 网格宽度菜单后端（少量编译身份 + 保留跳块）────────────────────────────────
#
# ## 它解决什么
#
# 块对角后端快（真机 4x4096 布局：4505ms vs 全通 11508ms，有效 MFU 18.2% vs 11.2%），
# 但段长元组是**等价类**：布局不同就不能同步，稀有布局凑不满 8 个 pack 会被系统性
# 欠采样（真实数据集上 27~39 张图）。全通后端只有一个身份，但真机实测整轮慢
# **1.216x**（定死 S_max 后 1.275x）。
#
# 中间路线的依据在 splash 源码里：
#
#     grid_width = fwd_mask_info.data_next.shape[-1]     kernel:1137
#
# 跳块收益来自**网格宽度**，而它只是 `data_next` 最后一维的长度 —— 一个**上界**，
# 不是等价类。于是可以：
#
#   1. host 侧照常 `process_mask` 拿到已收缩的 MaskInfo（宽度 = 最大段长 / 128）；
#   2. 把 `data_next/mask_next/block_mask` 沿最后一维**补齐到菜单里的 W**
#      （补出来的格子 `block_mask=0` -> `should_run=False`，纯空转，不改数值）；
#   3. MaskInfo 作为**运行时参数**传进被 jit 的步函数。
#
# 于是编译身份 = W（菜单里的几个整数），而**任何 pack 都能向上取整到更大的 W**，
# 所以一步的 8 个 pack 永远凑得齐 —— 孤儿欠采样问题消失。
#
# 真实数据集 your-dataset（budget 16384、全宽 128）的 W 分布：
#     W=32 x19 | W=64~120 x15 | W=128 x59（单张 16384 的图，本来就无块可跳）
# 菜单 `[32, 128]` = 2 个编译身份。
#
# ## 已验证到哪一步
#
#   * 本地 CPU interpret：补齐到统一 W 后输出与官方 `make_splash_mha` **逐 bit
#     相同（max_abs=0.000e+00）**，且两种段几何 trace 次数 = 1。
#   * 真机：见 kaggle_job/anima_menu。
#
# ## 代价与约束（都要盯住）
#
#   * **私有 API**：`splash_attention_mask_info.process_mask/_dkv`、`MaskInfo._replace`、
#     `SplashAttentionKernel.__init__`。都在 `jax.experimental` 下，无稳定性保证；
#     `_preamble.py` 已把 jax 钉死在 0.11.0，升级时必须重跑对拍。
#   * **MaskInfo 必须当参数传，不能闭包捕获** —— 闭包里的 jax 数组会被当成编译期
#     常量烘进图里，那就退回"每布局一次编译"，白做。
#   * **反向块固定 1024**：`use_fused_bwd_kernel=True` 时 dkv 的 MaskInfo 本就不收缩
#     （`_make_splash_attention` 里 `shrink_grid=not use_fused_bwd_kernel`），
#     其形状只取决于 (q_len/1024, kv_len/1024)，天然恒定。但 1024 要求最短段 >= 1024
#     且段长是 1024 的倍数 —— 即 **quantum 必须是 1024**（Q=128 会让某些段产生
#     partial block）。这条在 `menu_mask_info` 里 fail-fast。
MENU_BWD_BLOCK = 1024


def _menu_block_sizes():
    """菜单后端的固定块大小。**不随布局变**，否则编译身份又回来了。"""
    sk, _ = _import_splash()
    return sk.BlockSizes(
        block_q=BLOCK, block_kv=BLOCK, block_kv_compute=BLOCK,
        block_q_dkv=MENU_BWD_BLOCK, block_kv_dkv=MENU_BWD_BLOCK,
        block_kv_dkv_compute=MENU_BWD_BLOCK, use_fused_bwd_kernel=True)


def menu_grid_width(seg_lens: Sequence[int]) -> int:
    """该布局的原生网格宽度 = 最大段长 / 128（每个 q block 最多看几个 kv block）。"""
    return max(int(n) for n in seg_lens) // BLOCK


def pick_menu_width(seg_lens: Sequence[int], menu: Sequence[int]) -> int:
    """从菜单里挑最小的、不小于原生宽度的 W。挑不到就是菜单设计错了，fail-fast。"""
    w = menu_grid_width(seg_lens)
    cand = [m for m in sorted(menu) if m >= w]
    if not cand:
        raise ValueError(f"布局 {tuple(seg_lens)} 的原生网格宽度 {w} 超出菜单 "
                         f"{sorted(menu)}；菜单里必须有一档 >= budget/{BLOCK}")
    return cand[0]


def menu_mask_info(q_seg_lens: Sequence[int], kv_seg_lens: Sequence[int],
                   num_heads: int, width: int):
    """host 侧造 MaskInfo 并补齐到网格宽度 `width`。返回 (fwd, dkv) 两个 pytree。

    **返回的是 numpy pytree**，调用方负责搬上设备并作为运行时参数传给步函数
    （闭包捕获会让它变成编译期常量，见本节开头）。
    """
    import numpy as _np
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_mask_info as smi)
    _, sm = _import_splash()

    _check_aligned(q_seg_lens, "q")
    _check_aligned(kv_seg_lens, "kv")
    for name, segs in (("q", q_seg_lens), ("kv", kv_seg_lens)):
        bad = [n for n in segs if n % MENU_BWD_BLOCK]
        if bad:
            raise ValueError(
                f"菜单后端要求 {name} 段长是 {MENU_BWD_BLOCK} 的倍数（反向块固定 "
                f"{MENU_BWD_BLOCK} 才能让 dkv MaskInfo 形状恒定），越界 {bad[:4]}。"
                f"把 packing.Packer 的 quantum 设成 {MENU_BWD_BLOCK}。")
    q_seg, kv_seg = segment_ids(q_seg_lens), segment_ids(kv_seg_lens)
    mask = sm.MultiHeadMask(masks=[block_diag_mask(q_seg, kv_seg)] * num_heads)
    bs = _menu_block_sizes()
    fwd, _ = smi.process_mask(mask, (bs.block_q, bs.block_kv))
    # dkv 用 shrink_grid=False（与 _make_splash_attention 在 fused 下的口径一致），
    # 形状本就只取决于 (q_len/1024, kv_len/1024)，不需要补齐。
    dkv, _ = smi.process_mask_dkv(mask, (bs.block_q_dkv, bs.block_kv_dkv),
                                  shrink_grid=False)
    native = fwd.data_next.shape[-1]
    if native > width:
        raise ValueError(f"原生网格宽度 {native} > 请求的 {width}；W 只能向上取整")

    def pad(a):
        if a is None:
            return None
        a = _np.asarray(a)
        if a.shape[-1] >= width:
            return a
        return _np.pad(a, [(0, 0)] * (a.ndim - 1) + [(0, width - a.shape[-1])])

    fwd = fwd._replace(data_next=pad(fwd.data_next), mask_next=pad(fwd.mask_next),
                       block_mask=pad(fwd.block_mask))
    return fwd, dkv


def make_menu_attn(num_heads: int, head_dim: int, interpret: bool = False
                   ) -> Callable:
    """构造 (q,k,v,seg_q,seg_kv,fwd_mi,dkv_mi)->out 的块对角注意力。

    q/k/v 布局 [S, H, D]，与另两个后端一致。**MaskInfo 是参数**，所以同一个 W
    下换任何段几何都不重编译。用 `bind_menu` 在 trace 内绑定。
    """
    import jax.numpy as jnp
    sk, _ = _import_splash()
    bs = _menu_block_sizes()
    scale = head_dim ** -0.5

    def attn(q, k, v, seg_q, seg_kv, fwd_mi, dkv_mi):
        kernel = sk.SplashAttentionKernel(
            fwd_mi, None, dkv_mi, block_sizes=bs, is_mqa=False,
            save_residuals=False, mask_value=sk.DEFAULT_MASK_VALUE,
            attn_logits_soft_cap=None, residual_checkpoint_name=None,
            mask_function=None, interpret=interpret)
        # splash **不内置 1/sqrt(d)**（三个后端共有的硬约束，漏了是静默错）
        qs = (q.astype(jnp.float32) * scale).astype(q.dtype).transpose(1, 0, 2)
        o = kernel(qs, k.transpose(1, 0, 2), v.transpose(1, 0, 2),
                   segment_ids=sk.SegmentIds(q=seg_q, kv=seg_kv))
        return o.transpose(1, 0, 2)

    return attn


def bind_menu(attn: Callable, fine_q, fine_kv, fwd_mi, dkv_mi) -> Callable:
    """把 segment_ids 与 MaskInfo 绑到 attn 上，得到 `forward_packed` 要的三参签名。

    **必须在被 trace 的函数里调，且 fwd_mi/dkv_mi 要是 tracer**（来自步函数的
    参数），不能是 trace 外造好的具体数组 —— 后者会被烘成编译期常量。
    """
    return lambda q, k, v: attn(q, k, v, fine_q, fine_kv, fwd_mi, dkv_mi)


# ── 全通打包后端（单编译身份）────────────────────────────────────────────────
#
# ## 它和上面块对角后端的关系
#
# 块对角后端把**段几何编进静态 mask**换取跳块（Σn² 而不是 L²），代价是段长元组
# 进了编译身份：每种布局编一次全模型、8 卡一步必须同布局、稀有布局凑不满 8 个
# pack 会被系统性欠采样。
#
# 本后端反过来：静态 mask 恒为 `FullMask`，**段边界完全交给运行时 segment_ids**。
# 于是编译身份退化成 (q_len, kv_len) 两个整数 —— 换任何段组成都不重编译，
# 一步可以由**任意** 8 个 pack 组成。代价是不跳块，注意力吃满 L²。
#
# ## 这是 NaViT 原文的选择，不是退让
#
# Patch n' Pack (Dehghani et al., arXiv:2307.06304) §2.1 用的就是自注意力 mask，
# 不是块稀疏内核；§2.3 专门论证这笔账：隐藏维越宽，注意力在总算力里占比越小，
# 打包的额外开销随之减小。§2.3 同时报告填充 token 通常不到 2%。
#
# ## 在 Anima 上这笔账多大（本地 FLOPs 口径，真实数据集 your-dataset）
#
# budget=16384、D=2048、28 层，按真实布局直方图加权：**全通/块对角 = 1.13x**
# （固定 S_max 的 cross-attn 一起算是 1.18x）。之所以这么小，是因为该数据集
# 93 个 pack 里 59 个只装得下一张 16384 的图 —— 那些 pack 本来就无块可跳。
# budget=32768 时涨到 1.69x，所以**这条路线只在 budget 贴着单图上限时划算**。
# 真机步时对照见 kaggle_job/anima_mask。
#
# ## 顺带消掉的两条约束
#
#   * **不需要段长量化**：Q 只为压布局数而存在，这里没有布局。
#   * **不需要 128 对齐**：对齐是为了让块对角 MaskInfo 不产生 partial block，
#     FullMask 按定义没有 partial block。只剩 q_len/kv_len 被 128 整除。
@functools.lru_cache(maxsize=32)
def _full_kernel(q_len: int, kv_len: int, num_heads: int, interpret: bool):
    """全通 splash kernel。与另两个后端同一条约束：**必须在 trace 外构造**
    （trace 内构造会让 MaskInfo 变 tracer 并被 lru_cache 泄漏到下一次 trace）。"""
    sk, sm = _import_splash()
    mask = sm.MultiHeadMask(masks=[sm.FullMask((q_len, kv_len))] * num_heads)
    return sk.make_splash_mha(mask, head_shards=1, q_seq_shards=1,
                              block_sizes=_block_sizes(q_len, kv_len),
                              interpret=interpret)


def make_full_attn(q_len: int, kv_len: int, num_heads: int, head_dim: int,
                   fine_q=None, fine_kv=None, interpret: bool = False) -> Callable:
    """构造 (q,k,v)->out 的**全通 + 运行时 segment_ids** 注意力。

    q/k/v 布局 [S, H, D]，与 `make_splash_attn` 完全一致 —— 两者可直接对换，
    `anima_jax.forward_packed` 不用改。

    `fine_q` / `fine_kv` 语义与 `make_splash_attn` 相同，但在这里它们**承担全部
    正确性**（那边只负责段内量化填充的精细边界，粗粒度由静态 mask 兜底）。
    因此**不给 segment_ids 就是错的**，这里 fail-fast，不让它静默退化成"一个
    pack 里所有图互相看得见"。用 `bind_segments` 在 trace 内绑定。

    splash 的 SegmentIds 文档警告不能有整行全 0，故沿用 packing.py 的约定：
      * 自注意力：段内填充 token 用同一个独立段号（彼此可见，行不空）；
      * cross-attn：填充 token 沿用宿主图的段号（文本侧没有配对的填充段）。
    """
    import jax.numpy as jnp
    sk, _ = _import_splash()

    for name, n in (("q_len", q_len), ("kv_len", kv_len)):
        if n % BLOCK:
            raise ValueError(f"{name}={n} 必须是 {BLOCK} 的倍数"
                             f"（splash_attention_mask_info.py 对此硬 raise）")
    if (fine_q is None) != (fine_kv is None):
        raise ValueError("fine_q / fine_kv 必须同时给或同时不给")
    kernel = _full_kernel(q_len, kv_len, num_heads, interpret)
    scale = head_dim ** -0.5

    def attn(q, k, v, seg_q=fine_q, seg_kv=fine_kv):
        if seg_q is None:
            raise ValueError(
                "全通后端必须给 segment_ids —— 静态 mask 是全通的，段边界只由它"
                "保证。不给不会报错，只会让一个 pack 里的图互相看得见（静默错）。"
                "请用 bind_segments 绑定。")
        # splash **不内置 1/sqrt(d)**（三个后端共有的硬约束，漏了是静默错）
        qs = (q.astype(jnp.float32) * scale).astype(q.dtype).transpose(1, 0, 2)
        o = kernel(qs, k.transpose(1, 0, 2), v.transpose(1, 0, 2),
                   segment_ids=sk.SegmentIds(q=seg_q, kv=seg_kv))
        return o.transpose(1, 0, 2)

    return attn


# ── 分桶（批维）后端 ──────────────────────────────────────────────────────────
#
# ## 与打包后端的关系
#
# 段长量化到 Q 之后，**块对角注意力与批维注意力在算力上完全等价**：
# 打包路径里第 i 段的 q 只看得见第 i 段的 kv，代价 Σ seg_i²；批路径里第 i 个批
# 元素同样只看自己，代价 Σ L²（同一桶内 L 相同）。所以"打包"相对"按量化 token 数
# 分桶"在注意力上买不到任何东西。
#
# 但两者在**布局**上天差地别：
#
#   打包  x [ΣN, D]      AdaLN 调制要 gather 成 [ΣN, 3D] -> 物化 -> 真机 1.01 MB/token
#   分桶  x [G, L, D]    AdaLN 调制是 [G, 1, 3D] 广播    -> 融合 -> 真机 0.02 MB/token
#
# 而且分桶路径不需要：块对角 mask、惰性 MaskInfo、两级 mask、按段长元组编译整模型。
# 编译身份退化成一个整数 L。代价是同一桶里要凑够 8 张图才能成一步。
#
# ## 任意宽高比仍然成立
#
# 桶是按**量化后的 token 总数**分的，不是按 (h, w) 分的。每图保留自己的 (h_i, w_i)，
# 只要 h_i*w_i <= L 即可同批；RoPE 逐 token 查 rows/cols，不依赖矩形网格。
# 这是把"NaViT 的设计哲学（任意宽高比）"与"NaViT 的实现手段（序列打包）"分开 ——
# 用户要的是前者。
@functools.lru_cache(maxsize=64)
def _bucket_kernel(q_len: int, kv_len: int, num_heads: int, interpret: bool):
    """整段全通的 splash kernel（`FullMask`）。同打包后端一样必须在 trace 外构造
    （trace 内构造会让 MaskInfo 变 tracer 并被 lru_cache 泄漏到下一次 trace）。"""
    sk, sm = _import_splash()
    mask = sm.MultiHeadMask(masks=[sm.FullMask((q_len, kv_len))] * num_heads)
    return sk.make_splash_mha(mask, head_shards=1, q_seq_shards=1,
                              block_sizes=_block_sizes(q_len, kv_len),
                              interpret=interpret)


def make_bucket_attn(q_len: int, kv_len: int, num_heads: int, head_dim: int,
                     use_segments: bool, interpret: bool = False) -> Callable:
    """构造 (q,k,v)->out 的**批维**注意力。q [G, Lq, H, D]，k/v [G, Lkv, H, D]。

    `use_segments=True`（自注意力）：需要运行时 `segment_ids` 把每图尾部的量化
      填充 token 隔离掉 —— 它们作为 key 会污染同图真 token（打包路径同样的坑，
      本地实测 max_abs=5.4，静默）。段号只有两个值（0=真 / 1=填充），
      形状固定，不触发重编译。填充 token 彼此可见，行不空。

    `use_segments=False`（cross-attn）：**完全不需要 mask**。每图 q 看自己那
      512 个文本 token 的全部（`navit_text_trim_padding` 默认 False，文本 pad
      照常参与，见 memory navit-text-trim-train-eval-mismatch）；图像侧的填充
      q 行算出来是垃圾但会被 loss_mask 丢掉。这一条是分桶路径相对打包路径省掉的
      整块复杂度：打包那边 cross-attn 是矩形块对角 splash（真机归因里占 4.2GB）。

    静态 mask 是全通 -> 不跳块，但桶内**没有块可跳**（只有每图尾部那点量化填充），
    浪费 = 1 - (real/L)²，Q=1024 时约 3-7%、Q=128 时 <1%。
    """
    import jax
    import jax.numpy as jnp
    sk, _ = _import_splash()

    for name, n in (("q_len", q_len), ("kv_len", kv_len)):
        if n % BLOCK:
            raise ValueError(f"{name}={n} 必须是 {BLOCK} 的倍数（splash 块粒度）")
    kernel = _bucket_kernel(q_len, kv_len, num_heads, interpret)
    scale = head_dim ** -0.5

    def attn(q, k, v, seg_q=None, seg_kv=None):
        # splash 不内置 1/sqrt(d)（与打包后端同一条硬约束）
        qs = (q.astype(jnp.float32) * scale).astype(q.dtype).transpose(0, 2, 1, 3)
        ks, vs = k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3)
        if use_segments:
            if seg_q is None:
                raise ValueError("use_segments=True 时必须经 bind_segments 绑定段号")
            o = jax.vmap(lambda a, b, c, sq, skv: kernel(
                a, b, c, segment_ids=sk.SegmentIds(q=sq, kv=skv)))(
                    qs, ks, vs, seg_q, seg_kv)
        else:
            o = jax.vmap(kernel)(qs, ks, vs)
        return o.transpose(0, 2, 1, 3)

    return attn


def bucket_segment_ids(real_lens: Sequence[int], q_len: int) -> np.ndarray:
    """[G, q_len] 的真/填充段号：0=真 token、1=尾部量化填充。

    填充给**同一个**非零段号（而不是逐 token 独立），保证填充行彼此可见、
    softmax 分母不为 0（splash 的 SegmentIds 文档对全 0 行有明确警告）。
    """
    idx = np.arange(q_len, dtype=np.int32)[None, :]
    return (idx >= np.asarray(real_lens, np.int32)[:, None]).astype(np.int32)


def make_dense_attn(q_seg_lens: Sequence[int], kv_seg_lens: Sequence[int],
                    fine_q=None, fine_kv=None) -> Callable:
    """稠密参考实现（本地对拍 / CPU 用）。O(S·T) 显存，只适合小 pack。

    与 splash 后端的差别只应是数值噪声——这正是
    tests/check_splash_blockdiag.py 要证的事。给了 fine_* 时按
    「粗粒度 AND 精细」构造 bias，与 splash 侧 `static mask and-ed with
    segment id mask` 的语义一致。
    """
    import jax.numpy as jnp
    A = _anima_jax()
    allow = np.asarray(A.block_diag_bias(q_seg_lens, kv_seg_lens)) == 0.0
    if fine_q is not None:
        fq, fkv = np.asarray(fine_q), np.asarray(fine_kv)
        allow = allow & (fq[:, None] == fkv[None, :])
    bias = jnp.asarray(np.where(allow, 0.0, -np.inf), dtype=jnp.float32)
    return lambda q, k, v: A.attention_dense(q, k, v, bias)
