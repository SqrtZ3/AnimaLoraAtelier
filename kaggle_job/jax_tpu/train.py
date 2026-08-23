r"""TPU 训练步：把 anima_jax / adapters / attention / packing / flow / aux / optim
装配成一步训练。

## 一步长什么样

  t        <- sched.AdaptiveTimestepSampler.sample(...)   host 侧，每图一个
  noise    <- immiscible KNN 选择（k=1 时即标准高斯）
  noisy    <- (1-t)*lat + t*noise
  target   <- noise - lat                                 速度场
  tok      <- concat(patchify(noisy), zeros(4))           ← 见下"零填充到 68"
  pred     <- forward_packed(...)                         冻结底模 + 适配器旁路
  per_img  <- 逐图 masked Huber(δ(t)) 均值                **逐图等权，不是逐 token**
  per_img  <- per_img × eisbach(pred) − λ_dfm × vecor_neg
  loss     <- Σ w(t)·per_img / Σ w(t)  +  λ_spec × spectral(命中 gate 的图的均值)
  grad     <- ∂loss/∂适配器                               shard_map 转置时自动 psum
  state    <- AdamW(-SNR)(state, grad)
  反馈     <- per_img（**未经 eisbach/dfm 乘算的那一份**）回 host 喂自适应采样器

## 为什么把"求梯度"和"更新参数"拆成两个 jit

`grad_accum > 1` 时一个优化步由若干微步组成，而微步之间**可以是不同布局**
（不同的段长元组 = 不同的编译产物）。梯度树的形状与布局无关，所以在 host 侧累加
梯度、最后调用一次 `apply_update` 是最自然的写法；两者塞进一个 jit 反而做不到。
代价是梯度树会物化一次（r32 全目标约 184MB fp32），相对权重与激活可以忽略。

## 零填充到 68（抄错不报错的一处）

x_embedder 吃 68 维 = (16 latent 通道 + 1 padding-mask 通道) x patch 2x2。
但 NaViT 训练路径里 padding-mask 通道**是全 0**，不是全 1 ——
models/anima_modeling_core.py:1843-1845 直接 `F.pad(tokens, (0, expected - dim))`
把 64 维零填充到 68。patchify 的通道序是 `(c pt ph pw)`（c 在最外），所以末尾
补 4 个 0 恰好等价于"第 17 个通道全 0"。填成 1 不会报错，只会让整模条件偏移。

## 每卡一个 pack

8 卡纯 DP：权重 bf16 3.91GB < 单 chip 15.7GiB，每卡各存一份、各跑自己的 pack，
每步只 all-reduce 适配器梯度。**一步的 8 个 pack 必须同布局**——一个编译产物只有
一种编译期 mask，由 packing.Packer.plan_steps 保证。

## 编译成本

每种布局编译一次全模型（真机实测首调约 60s）。跨 session 靠 JAX 持久化编译缓存
（真机 C2 已验证 `/kaggle/working/jax_cache` 可写、可作下一棒的 dataset 输入）。
"""

from __future__ import annotations

import functools
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

try:                                   # 作为包导入
    from . import adapters as AD
    from . import anima_jax as A
    from . import attention as AT
    from . import auxloss as X
    from . import export as EX
    from . import flow as F
    from . import optim as O
    from . import krea2_jax as K2
    from .packing import PAD_SEG, K2Layout, Layout, Pack
except ImportError:                    # jax_tpu/ 直接在 sys.path 上（tests/ 走这条）
    import adapters as AD
    import anima_jax as A
    import attention as AT
    import auxloss as X
    import export as EX
    import flow as F
    import optim as O
    import krea2_jax as K2
    from packing import PAD_SEG, K2Layout, Layout, Pack

PyTree = Any


@dataclass(frozen=True)
class TrainConfig:
    adapter: AD.AdapterConfig = field(default_factory=AD.AdapterConfig)
    targets: Tuple[str, ...] = ("self_attn.q_proj", "self_attn.k_proj",
                                "self_attn.v_proj", "self_attn.output_proj",
                                "cross_attn.q_proj", "cross_attn.k_proj",
                                "cross_attn.v_proj", "cross_attn.output_proj",
                                "mlp.layer1", "mlp.layer2")
    #: remat 强度，见 anima_jax.REMAT_CHOICES。默认 "full" —— 它是目前唯一在
    #: 真机上跑通过的档；换档前请先看 P4 扫描结果，别凭直觉调。
    remat: str = "full"
    #: 打包路线的 AdaLN 调制摆法。**两个正交的开关，都只在展开路径下有意义**
    #: （scan 本来就挡住了跨块提升；真机实测 scan 下开 chunk 反而慢 1~5%）。
    #:
    #: 真机 anima-layout-probe（budget 8192、展开路径、8 卡）：
    #:     朴素展开        budget 16384 的 full 档即 OOM（第一棒 20.60G）
    #:     +chunk          full 24.1k / **every2 30.2k（有效MFU 21.6%，全场最高）**
    #:     +barrier        full 24.6k / every2 28.9k（有效MFU 20.6%）
    #:
    #: 取舍：barrier 是恒等算子（零数值风险，full 档更快）；chunk 数学等价但会改
    #: matmul lowering（bf16 梯度有 ULP 漂移），省显存更多、every2 档更快。
    #: 按仓库约定默认全关（行为中立）。
    packed_chunk: bool = False
    packed_barrier: bool = False
    #: 展开路径。**这是"把 8 卡吃满"的主旋钮**，但它与 budget 强耦合，真机
    #: anima-layout-probe 的账（打包路线、8 卡、真tok/s / 有效MFU）：
    #:
    #:     budget 16384  scan+full          16.9k / 13.4%   <- 默认档
    #:     budget 16384  展开+barrier+full   16.7k / 13.2%
    #:     budget 16384  展开+chunk+every2   OOM (23.62G)
    #:     budget  8192  展开+chunk+every2   **30.2k / 21.6%（全场最高）**
    #:     budget  8192  展开+barrier+every2 28.9k / 20.6%
    #:
    #: 也就是说：**只在 budget 16384 上开展开是白开的**（16.7k vs 16.9k），
    #: 收益全在"budget 减半 + 展开 + every2"这个组合上。budget 减半会让一步看到的
    #: token 减半 = 改变有效 batch，要靠 grad_accum 2 补回来才是同一个实验。
    #: 默认关（行为中立），由 run_train 的 `--unrolled` 显式打开。
    unrolled: bool = False
    dtype: Any = jnp.bfloat16
    seed: int = 0
    grad_accum: int = 1
    flow: F.FlowConfig = field(default_factory=F.FlowConfig)
    aux: X.AuxConfig = field(default_factory=X.AuxConfig)
    adamw: O.AdamWConfig = field(default_factory=O.AdamWConfig)

    def __post_init__(self):
        if self.unrolled and not (self.packed_chunk or self.packed_barrier):
            raise ValueError(
                "展开路径（unrolled）下 AdaLN 调制会 28 组同时活着（真机归因 "
                "1.01 MB/token，anima-mem-probe V4），必须开 packed_chunk 或 "
                "packed_barrier 之一才压得住。**full 档也不例外** —— 朴素展开在 "
                "budget 16384 的 full 档实测就要 20.60G（anima-layout-probe 第一棒）。")
        if self.packed_barrier and not self.unrolled:
            # barrier 只在展开分支里插（anima_jax._forward_core 的
            # `isinstance(params["blocks"], (list, tuple))` 分支）；scan 路径下这个
            # 参数一路传到底、落进用不到它的 else 分支，**一个 bit 都不变也不提示**。
            # 而 packed_chunk 单开是真生效的（chunk 分支与 params 布局无关，
            # 真机实测 scan 下慢 1~5%）—— 两个"正交开关"在这一点上并不对称，
            # 所以这里明说，不让它静默无效（run_train.py 的 eval 那处同一教训）。
            raise ValueError(
                "packed_barrier 只在展开路径（unrolled）下生效 —— scan 路径的块边界"
                "由 lax.scan 自己隔开，barrier 参数会被静默忽略（不报错、不改数值、"
                "也没有任何提示）。请同时给 --unrolled，或去掉 --packed-barrier。"
                "注意 packed_chunk 不同：它单开是真生效的（scan 下实测慢 1~5%）。")


# ── 初始化 ────────────────────────────────────────────────────────────────────
def init_adapter(key, model_cfg: A.AnimaConfig, tcfg: TrainConfig,
                 stacked_params: PyTree):
    """初始化适配器并**堆成 scan 布局**。返回 (trainable, consts, plans)。

    训练一律走 scan 布局的参数树（哪怕 `unrolled=True` —— 展开路径会自己切片）：
    展开路径的**参数**若也按块摊开，加一层 python list 只会让 pytree 变胖。
    """
    return AD.init(key, model_cfg, tcfg.adapter, tcfg.targets, stacked_params)


def patchify(latent: np.ndarray, patch: int = 2) -> Tuple[np.ndarray, Tuple[int, int]]:
    """latent [C, H, W] -> tokens [N, C*patch^2]，通道序 `(c ph pw)`，N = (H/p)*(W/p)。

    与 models/anima_modeling_core.py:1585 的
    `rearrange(x, "b c (t pt) (h ph) (w pw) -> b (t h w) (c pt ph pw)")` 同序
    （T=1 故 pt 维省略）。这个序写错 -> 统计量正常、逐元素全错，历史上出过一次
    （rel 1.4），是本仓库记录在案的静默错之一。
    """
    C, H, W = latent.shape
    if H % patch or W % patch:
        raise ValueError(f"latent {H}x{W} 不能被 patch {patch} 整除")
    h, w = H // patch, W // patch
    t = latent.reshape(C, h, patch, w, patch).transpose(1, 3, 0, 2, 4)
    return t.reshape(h * w, C * patch * patch), (h, w)


def pad_to_in_dim(tokens: jnp.ndarray, in_dim: int) -> jnp.ndarray:
    """64 -> 68 的**零**填充（padding-mask 通道全 0）。见模块 docstring。

    只在最后一维上补，**前导维不限**：打包布局是 [ΣN, 64]，分桶布局是 [G, L, 64]。
    """
    d = tokens.shape[-1]
    if d == in_dim:
        return tokens
    if d > in_dim:
        raise ValueError(f"token 维度 {d} > x_embedder 的 {in_dim}")
    pad = [(0, 0)] * (tokens.ndim - 1) + [(0, in_dim - d)]
    return jnp.pad(tokens, pad)


# ── 一个 pack 的 loss（打包与分桶两条布局共用）────────────────────────────────
def _loss_core(pred, lat, noisy, target, b, g: int, seg, tcfg: TrainConfig,
               key, canvas_hw):
    """从 pred 到标量 loss 的全部逐图数学。**布局无关**：调用方把一切摊平成
    `[ΣN, ...]` + 段号 `seg` 即可（分桶路线就是 `repeat(arange(G), L)`）。

    返回 `(loss, per_image_clean)`。`per_image_clean` 是**没被 eisbach / ΔFM /
    multiscale 权重乘过**的那一份 —— 自适应采样器要的是"这个 t 学得怎么样"的
    度量，不是"要不要让这张图影响参数"的旋钮（objective.py:1071 同口径）。
    """
    fl, au = tcfg.flow, tcfg.aux
    mask = b["loss_mask"]
    ssum = lambda x: X.seg_sum(x, seg, g)
    bcast = lambda v: jnp.take(v, seg)
    t = b["t"]

    delta_tok = bcast(F.huber_delta(t, fl))
    per_img, den = F.per_image_loss(pred, target, mask, t, fl, ssum, delta_tok)
    valid = (den > 0).astype(jnp.float32)
    clean = per_img

    graded = per_img
    if au.eisbach_lambda > 0:
        graded = graded * X.eisbach_weight(pred, mask, seg, g, au.eisbach_lambda)
    if au.dfm_lambda > 0:
        # 逐图 50/50：通道乱序 vs 裁剪+resize（objective.py:955 同粒度）。
        # 两支都全量算出来再 where —— where 的常量折叠不存在于运行时数据，
        # 但这样能保住"一个布局一份编译产物"，分支不进编译身份。
        k_coin, k_perm, k_crop = jax.random.split(key, 3)
        coin = jax.random.uniform(k_coin, (g,)) < 0.5
        neg = jnp.where(
            bcast(coin)[:, None],
            X.vecor_negative(k_perm, target, seg, g),
            X.vecor_crop_resize(k_crop, target, seg, b["rows"], b["cols"],
                                mask, g, canvas_hw))
        neg_img, _ = F.per_image_loss(pred, neg, mask, t, fl, ssum, delta_tok)
        graded = graded - float(au.dfm_lambda) * neg_img
    if "ms_weight" in b:
        graded = graded * b["ms_weight"]
    loss = F.weighted_mean(graded, t, valid, fl)

    if au.spectral_enabled:
        # x0 恢复 + 散射回 latent 网格画布（见 aux.py 的"零填充到静态画布"一节）
        x0p = X.recover_x0(noisy, bcast(t)[:, None], pred)
        cv = lambda z: X.to_canvas(z, seg, b["rows"], b["cols"], mask, g, canvas_hw)
        # cover：画布上"这一格属于真实图像"的 0/1 图。用全 1 token 走同一条散射，
        # 保证它与数据画布的对齐方式**逐格一致**（自己另算一套索引最容易错位）。
        cover = (cv(jnp.ones_like(lat)) [:, 0] > 0).astype(jnp.float32)
        loss = loss + X.spectral_term(cv(x0p), cv(lat), cover, t, au, valid)
    return loss, clean


def local_loss(lora, consts, params, batch, model_cfg: A.AnimaConfig,
               tcfg: TrainConfig, plans, self_fn, cross_fn, key,
               chunk=None, barrier=False, g: int = 1):
    """打包布局下一个 pack 的 loss。batch 里的东西都是**本卡**的。"""
    b = batch
    seg = b["mod_index"]
    lat = b["latent"]
    ssum = lambda x: X.seg_sum(x, seg, g)
    bcast = lambda v: jnp.take(v, seg)

    k_noise, k_drop, k_neg = jax.random.split(key, 3)
    noise = F.immiscible_noise(k_noise, lat, b["loss_mask"], ssum, bcast,
                               tcfg.flow.immiscible_k)
    noisy, target = F.make_noisy_and_target(lat, noise, bcast(b["t"])[:, None])

    drop = AD.sample_dropout(k_drop, tcfg.adapter, plans, model_cfg.num_blocks)
    ctx = A.LoraCtx(tcfg.adapter, lora, consts, drop)
    tok = pad_to_in_dim(noisy.astype(tcfg.dtype), model_cfg.in_dim)
    pred = A.forward_packed(params, model_cfg, tok, b["t"], b["ctx"],
                            b["rows"], b["cols"], seg,
                            self_fn, cross_fn, loras=ctx, remat=tcfg.remat,
                            chunk=chunk, barrier=barrier)
    return _loss_core(pred, lat, noisy, target, b, g, seg, tcfg, k_neg,
                      tcfg.aux.canvas_hw)


def local_loss_ragged(lora, consts, params, batch, model_cfg: A.AnimaConfig,
                      tcfg: TrainConfig, plans, self_fn, cross_fn, key,
                      g: int, L: int):
    """分桶布局下一步在**本卡**上的 loss。

    与打包路线**共用** `_loss_core`：把 [G, L, ...] 摊平成 [G*L, ...]、段号取
    `repeat(arange(G), L)` 之后，两条布局在 loss 侧完全一样。这样"换布局"不会
    悄悄换掉 loss 的口径（历史上这类不一致最难查）。
    """
    seg = jnp.repeat(jnp.arange(g, dtype=jnp.int32), L)
    flat = lambda z: z.reshape(g * L, *z.shape[2:])
    b = {"t": batch["t"], "ctx": batch["ctx"],
         "rows": flat(batch["rows"]), "cols": flat(batch["cols"]),
         "loss_mask": flat(batch["loss_mask"])}
    if "ms_weight" in batch:
        b["ms_weight"] = batch["ms_weight"]
    lat = flat(batch["latent"])
    ssum = lambda x: X.seg_sum(x, seg, g)
    bcast = lambda v: jnp.take(v, seg)

    k_noise, k_drop, k_neg = jax.random.split(key, 3)
    noise = F.immiscible_noise(k_noise, lat, b["loss_mask"], ssum, bcast,
                               tcfg.flow.immiscible_k)
    noisy, target = F.make_noisy_and_target(lat, noise, bcast(b["t"])[:, None])

    drop = AD.sample_dropout(k_drop, tcfg.adapter, plans, model_cfg.num_blocks)
    ctx = A.LoraCtx(tcfg.adapter, lora, consts, drop)
    tok = pad_to_in_dim(noisy.reshape(g, L, -1).astype(tcfg.dtype), model_cfg.in_dim)
    pred = A.forward_ragged(params, model_cfg, tok, batch["t"], batch["ctx"],
                            batch["rows"], batch["cols"],
                            self_fn, cross_fn, loras=ctx, remat=tcfg.remat)
    return _loss_core(pred.reshape(g * L, -1), lat, noisy, target, b, g, seg,
                      tcfg, k_neg, tcfg.aux.canvas_hw)


# ── 编译一步 ──────────────────────────────────────────────────────────────────
def eval_config(tcfg: TrainConfig) -> TrainConfig:
    """eval 用的配置：关掉一切**随机的/额外的**东西，只留主 loss。

    关掉 rank/module dropout（否则同一份权重每次 eval 出的数不一样）、immiscible
    （噪声选择依赖 latent，会让 eval 与 train 的噪声分布不同）、以及全部 aux
    与 eisbach/ΔFM（那些是训练期的梯度整形，不是"模型学得怎么样"的度量）。
    留下的就是**逐图 masked Huber(δ(t))** —— 与训练主项同一口径，可直接比较。
    """
    a = replace(tcfg.adapter, rank_dropout=0.0, module_dropout=0.0)
    f = replace(tcfg.flow, immiscible_k=1)
    return replace(tcfg, adapter=a, flow=f, aux=X.AuxConfig())


def make_grad_fn(model_cfg: A.AnimaConfig, tcfg: TrainConfig, plans, layout: Layout,
                 mesh, interpret: bool = False, grad: bool = True):
    """为一种布局编译一个"求梯度"函数。

    返回 `fn(lora, consts, params, batch, key) -> (loss, per_image, grads)`：
      * `loss` 标量（8 卡均值）
      * `per_image` [devices, G] —— 回 host 喂自适应采样器
      * `grads` 与 `lora` 同构（8 卡已 all-reduce）

    batch 的每个字段第 0 维都是设备维（长度 = mesh 的 'd' 轴）。
    """
    from jax.sharding import PartitionSpec as P

    coarse = list(layout.seg_lens)
    txt = [layout.txt_len] * layout.n_seg
    g = layout.n_seg
    # chunk 化调制：段长的 gcd 是能用的最大 chunk（更大就会跨图 -> 调制静默用错 t）。
    chunk = layout.max_chunk if tcfg.packed_chunk else None
    if chunk is not None:
        # `forward_packed` 只校验总长能被 chunk 整除，段长整除性它查不了（收的是
        # mod_index 而不是 seg_lens）。但这里 seg_lens 是编译期已知的 python 元组，
        # 所以这个 host 侧断言是免费的 —— 而漏掉它的后果是"某个 chunk 跨两张图、
        # 调制静默用错图的 t，不报错、训练照跑、结果全错"（见 anima_jax.forward_packed
        # 的 chunk 一节）。max_chunk 按定义是 gcd，正常永远成立；这条拦的是将来有人
        # 手改 chunk 来源的情况。
        bad = [s for s in layout.seg_lens if s % chunk]
        if bad:
            raise ValueError(
                f"chunk={chunk} 不整除段长 {bad[:4]}：某个 chunk 会跨两张图，AdaLN "
                f"调制会静默用错图的 t（不报错，结果全错）。chunk 必须取段长的 gcd "
                f"（Layout.max_chunk）。")

    # **kernel 在 trace 外构造**：它的 MaskInfo 是 jax 数组，在 trace 内构造会变成
    # tracer 并被 _splash_kernel 的 lru_cache 缓存下来，泄漏到下一次 trace
    # （本地 check_train_loop 抓到过 UnexpectedTracerError）。而且构造很贵
    # （真机 735ms），本就该按布局只做一次。
    # seg_cap 只统计**实段**：自然长度布局里 <1024 的纯填充段若参与取 min，
    # 会把全盘反向块拖进退让链（见 attention.make_splash_attn 与 BWD_BLOCK_PREF）。
    real_cap = min(layout.real_seg_lens)
    self_attn = AT.make_splash_attn(coarse, coarse, model_cfg.num_heads,
                                    model_cfg.head_dim, interpret=interpret,
                                    seg_cap=real_cap)
    cross_attn = AT.make_splash_attn(coarse, txt, model_cfg.num_heads,
                                     model_cfg.head_dim, interpret=interpret,
                                     seg_cap=min(real_cap, layout.txt_len))

    def per_shard(lora, consts, params, b, key):
        # 精细 segment_ids 是**运行时**数组，随 pack 变化但形状固定 -> 不触发重编译。
        # 粗粒度段长进编译期 mask 负责跳块。两级 mask 的语义与依据见 attention.py。
        self_fn = AT.bind_segments(self_attn, b["seg_self"][0], b["seg_self"][0])
        cross_fn = AT.bind_segments(cross_attn, b["seg_cross"][0], b["seg_txt"][0])
        one = {k: v[0] for k, v in b.items()}
        # **每卡不同的 key**：同一份 key 会让 8 卡的 dropout 掩码/噪声候选完全相同，
        # 等价于 batch 里有 8 份重复扰动。用卡号 fold 进去。
        kk = jax.random.fold_in(key, jax.lax.axis_index("d"))
        loss, per = local_loss(lora, consts, params, one, model_cfg, tcfg, plans,
                               self_fn, cross_fn, kk, chunk, tcfg.packed_barrier, g)
        return loss[None], per[None]

    dspec = {k: P("d") for k in ("latent", "t", "ctx", "rows", "cols",
                                 "mod_index", "loss_mask", "seg_self",
                                 "seg_cross", "seg_txt", "ms_weight")}
    return _finish_grad_fn(per_shard, mesh, dspec, tcfg, grad)


def make_grad_fn_ragged(model_cfg: A.AnimaConfig, tcfg: TrainConfig, plans, bucket,
                        mesh, interpret: bool = False, grad: bool = True):
    """分桶路线的同名物。签名与 `make_grad_fn` 一致。

    相对打包路线省掉的东西：块对角 mask、惰性 MaskInfo、两级 mask、cross-attn 的
    矩形 splash。编译身份从段长元组退化成 (L, G)。
    """
    from jax.sharding import PartitionSpec as P

    L, T, g = bucket.length, bucket.txt_len, bucket.imgs
    self_attn = AT.make_bucket_attn(L, L, model_cfg.num_heads, model_cfg.head_dim,
                                    use_segments=True, interpret=interpret)
    # cross-attn **不需要任何 mask**：每图看自己那一份定长文本槽的全部
    # （navit_text_trim_padding 默认 False，见 memory
    #  navit-text-trim-train-eval-mismatch 的 A/B 实证）。
    cross_attn = AT.make_bucket_attn(L, T, model_cfg.num_heads, model_cfg.head_dim,
                                     use_segments=False, interpret=interpret)

    def per_shard(lora, consts, params, b, key):
        one = {k: v[0] for k, v in b.items()}
        self_fn = AT.bind_segments(self_attn, one["seg"], one["seg"])
        kk = jax.random.fold_in(key, jax.lax.axis_index("d"))
        loss, per = local_loss_ragged(lora, consts, params, one, model_cfg, tcfg,
                                      plans, self_fn, cross_attn, kk, g, L)
        return loss[None], per[None]

    dspec = {k: P("d") for k in ("latent", "t", "ctx", "rows", "cols",
                                 "seg", "loss_mask", "ms_weight")}
    return _finish_grad_fn(per_shard, mesh, dspec, tcfg, grad)


def _fwd_cast(lora, dtype):
    """把适配器参数降到前向 dtype，**但 `dora` 叶子保 fp32**。

    为什么单独放它：`dora_scale` 是整个适配器里唯一"初值恰为 ‖W‖_row、之后只叠很小
    增量"的参数，DoRA 的 step-0 中立性正是靠 `dora / merged_row_norm == 1` 精确成立
    （adapters.py 的 init 从同一个 `base_row_sq` 算两边）。整棵树无差别 astype 会破坏
    这一条，本地实测（in=out=256, r=16）：

        dora 全 fp32      step-0 输出相对偏差 = 0.000e+00   逐 bit 中立
        dora 降 bf16      step-0 输出相对偏差 = 1.659e-03   不中立

    而且 T1 闸门比的是 loss（偏差仅 2e-7），放过了它 —— 问题在输出域。

    第二重后果更要紧：bf16 在 ‖W‖ 量级上的 ulp 约 0.62%，

        ‖W‖=5.0  -> ulp 0.0312，lr=1e-4 要约  312 步才在前向可见
        ‖W‖=20.0 -> ulp 0.1250，lr=1e-4 要约 1250 步

    即 master 上学到的幅度增量在越过半个 ulp 之前对前向完全不可见，典型 run
    （数百~千步）下 DoRA 的幅度分量近似冻死。

    这不是新口径：`export.py` 与 `trainer/lora.py` 的 state_dict 都坚持把 dora_scale
    存 fp32，理由就是"降 bf16 会让幅度增量的最大相对误差 ~50%"。PyTorch 侧前向也是
    fp32（`LoRALinear.forward` 里的 `.float()`）。存盘认了这个理，前向不该再 round 掉。

    代价：`dora` 是 [L, out] 的小张量（r32 全目标下几 MB），可忽略。
    """
    def cast(path, x):
        last = path[-1]
        name = getattr(last, "key", None) or getattr(last, "name", None)
        return x if name == "dora" else x.astype(dtype)
    return jax.tree_util.tree_map_with_path(cast, lora)


def _finish_grad_fn(per_shard, mesh, dspec, tcfg: TrainConfig, grad: bool = True):
    from jax.sharding import PartitionSpec as P

    smapped = _shard_map(per_shard, mesh,
                         (P(), P(), P(), dspec, P()), (P("d"), P("d")))

    def loss_fn(lora, consts, params, b, key):
        loss, per = smapped(lora, consts, params, b, key)
        return jnp.mean(loss), per

    if not grad:
        @jax.jit
        def eval_fn(lora, consts, params, b, key):
            fwd = _fwd_cast(lora, tcfg.dtype)
            loss, per = loss_fn(fwd, consts, params, b, key)
            return loss, per, None
        return eval_fn

    @jax.jit
    def grad_fn(lora, consts, params, b, key):
        # lora/params 在 in_spec P() 上是复制的 -> 它们的余切在 shard_map 转置时
        # 自动 psum，即适配器梯度的 8 卡 all-reduce 已含在这一步里。
        fwd = _fwd_cast(lora, tcfg.dtype)
        (loss, per), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            fwd, consts, params, b, key)
        return loss, per, grads
    return grad_fn


@functools.partial(jax.jit, static_argnums=(2,), donate_argnums=(0,))
def apply_update(state, grads, cfg: O.AdamWConfig):
    """把（可能是累加过的）梯度作用到优化器状态上。与布局无关，只编译一次。

    `cfg` 走 static_argnums：AdamWConfig 是全标量的 frozen dataclass，可哈希，
    这样 lr/wd 等都被折叠成常量，不必每步传张量。
    """
    new_state, _, diag = O.update(state, grads, cfg, jnp.float32)
    return new_state, diag


def accumulate(acc, grads):
    """梯度累加（host 侧驱动，跨布局也成立）。

    **在 fp32 上累加**：`grad_fn` 里参数在进 `value_and_grad` 之前就被降到 bf16，
    所以回来的余切也是 bf16（8 位尾数）。前向/反向用 bf16 是这条路线的既定口径，
    但"把若干微步的梯度加起来"没有理由也在 bf16 上做 —— 那是纯粹白丢精度，
    而且丢得不均匀（先加的微步被后加的舍入吃掉），不报错。
    优化器本来就要把它转成 fp32（optim.update），这里只是把转换提前到累加之前。
    第一个微步也转，否则 grad_accum=1 与 >1 的 dtype 路径不一致。
    """
    g32 = jax.tree.map(lambda x: x.astype(jnp.float32), grads)
    if acc is None:
        return g32
    return jax.tree.map(lambda a, b: a + b, acc, g32)


def _shard_map(f, mesh, in_specs, out_specs):
    """shard_map 包装。**纯 DP 也必须用**：splash 是 Pallas/Mosaic 内核，
    XLA 的 GSPMD 自动分区处理不了（真机报 `Mosaic kernels cannot be
    automatically partitioned.`）。参数名在 0.8 前叫 check_rep、0.11 起叫 check_vma。"""
    try:
        from jax import shard_map as _sm
    except ImportError:
        from jax.experimental.shard_map import shard_map as _sm
    for kw in ({"check_vma": False}, {"check_rep": False}, {}):
        try:
            return _sm(f, mesh=mesh, in_specs=in_specs, out_specs=out_specs, **kw)
        except TypeError:
            continue
    raise RuntimeError("shard_map 不接受任何已知的 check_* 参数名")


# ── batch 组装 ────────────────────────────────────────────────────────────────
def assemble_batch(packs: Sequence[Pack], latents, ctxs, t: np.ndarray,
                   model_cfg: A.AnimaConfig, dtype, ms_weight=None
                   ) -> Dict[str, jnp.ndarray]:
    """把同布局的 N 个 pack 拼成 [N, ...] 的 batch。

    latents[i][j] = 第 i 个 pack 第 j 段的 patchified latent [n_j, 64]
    ctxs[i][j]    = 同上的文本特征 [txt_len, crossattn_dim]（已定长填充）
    纯填充段传 None。
    t: [N*G] 的 timestep（host 侧采好的，见模块 docstring 的反馈闭环）。
    """
    layout = packs[0].layout
    if any(p.layout != layout for p in packs):
        raise ValueError("同一步的 pack 必须同布局（一个编译产物只有一种 mask）")
    B, G, N = layout.total_len, layout.n_seg, len(packs)
    if t.shape != (N * G,):
        raise ValueError(f"t 的长度 {t.shape} != pack 数 × 段数 = {N * G}")
    out: Dict[str, list] = {k: [] for k in
                            ("latent", "ctx", "rows", "cols", "mod_index",
                             "loss_mask", "seg_self", "seg_cross", "seg_txt")}
    for p, lat_list, ctx_list in zip(packs, latents, ctxs):
        idx = p.index_arrays()
        if len(lat_list) != layout.n_seg or len(ctx_list) != layout.n_seg:
            # 少给一段就有一段被静默当成纯填充 —— 那张图整段丢失，不报错。
            raise ValueError(
                f"latents/ctxs 每 pack 必须给齐 {layout.n_seg} 段（纯填充段给 None），"
                f"拿到 {len(lat_list)}/{len(ctx_list)}")
        lat = np.zeros((B, model_cfg.out_dim), np.float32)
        ctx = np.zeros((layout.kv_txt, model_cfg.crossattn_dim), np.float32)
        off = 0
        for j, seg in enumerate(layout.seg_lens):
            L = lat_list[j]
            if L is not None:
                lat[off:off + L.shape[0]] = L
            C = ctx_list[j]
            if C is not None:
                ctx[j * layout.txt_len:(j + 1) * layout.txt_len] = C
            off += seg
        out["latent"].append(lat)
        out["ctx"].append(ctx)
        for k in ("rows", "cols", "mod_index", "loss_mask", "seg_self",
                  "seg_cross", "seg_txt"):
            out[k].append(idx[k])

    batch = {k: jnp.asarray(np.stack(v)) for k, v in out.items()}
    batch["ctx"] = batch["ctx"].astype(dtype)
    batch["t"] = jnp.asarray(t.reshape(N, G).astype(np.float32))
    # **ms_weight 恒存在**（不用时全 1）：batch 的键集进了 shard_map 的 in_specs，
    # 有时有、有时没有会让同一个布局编译出两份图，也会在 spec 不匹配时直接报
    # pytree 结构错。全 1 的乘法是 XLA 会折叠掉的常量。
    batch["ms_weight"] = jnp.asarray(
        np.ones((N, G), np.float32) if ms_weight is None
        else np.asarray(ms_weight, np.float32).reshape(N, G))
    return batch


def assemble_batch_ragged(step_plan, latents, ctxs, t: np.ndarray,
                          model_cfg: A.AnimaConfig, dtype, devices: int = 8,
                          ms_weight=None) -> Dict[str, jnp.ndarray]:
    """把一个 `packing.BucketStep` 组装成 [devices, G, ...] 的 batch。"""
    B, L = step_plan.bucket, step_plan.bucket.length
    idx = step_plan.index_arrays(devices)
    n = devices * B.imgs
    if not (len(latents) == len(ctxs) == n):
        raise ValueError(f"一步要 {n} 张图的 latent/ctx，得到 "
                         f"{len(latents)}/{len(ctxs)}")
    lat = np.zeros((n, L, model_cfg.out_dim), np.float32)
    ctx = np.zeros((n, B.txt_len, model_cfg.crossattn_dim), np.float32)
    for i, (a, c) in enumerate(zip(latents, ctxs)):
        lat[i, :a.shape[0]] = a
        ctx[i] = c
    batch = {k: jnp.asarray(v) for k, v in idx.items()}
    batch["latent"] = jnp.asarray(lat.reshape(devices, B.imgs, L, model_cfg.out_dim))
    batch["ctx"] = jnp.asarray(
        ctx.reshape(devices, B.imgs, B.txt_len, model_cfg.crossattn_dim)).astype(dtype)
    batch["t"] = jnp.asarray(t.reshape(devices, B.imgs).astype(np.float32))
    batch["ms_weight"] = jnp.asarray(
        np.ones((devices, B.imgs), np.float32) if ms_weight is None
        else np.asarray(ms_weight, np.float32).reshape(devices, B.imgs))
    return batch


# ── 状态存取（12h session 断点接棒）────────────────────────────────────────────
def save_state(path, state, tcfg: TrainConfig, extra: Optional[Dict] = None) -> Path:
    """存**完整**优化器状态：fp32 master + m + v + step。

    只存 master 是不够的——AdamW 的一二阶矩丢了，resume 后前几百步的有效学习率
    完全不同（bias-correction 从头开始），等价于每次接棒都重来一次 warmup。
    自适应采样器的 EMA/counts 也要一起存（走 `extra`），否则接棒后要重新 burn-in。
    """
    path = Path(path)
    flat: Dict[str, np.ndarray] = {}
    for part in ("master", "m", "v"):
        for name, mod in state[part].items():
            for leaf, v in mod.items():
                flat[f"{part}/{name}/{leaf}"] = np.asarray(v, np.float32)
    flat["step"] = np.asarray(state["step"], np.int32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **flat)
    meta = {"step": int(state["step"]), "config": _jsonable(tcfg), **(extra or {})}
    path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    return path


def load_state(path) -> Dict[str, Any]:
    z = np.load(Path(path))
    state: Dict[str, Any] = {"master": {}, "m": {}, "v": {},
                             "step": jnp.asarray(z["step"], jnp.int32)}
    for k in z.files:
        if k == "step":
            continue
        part, name, leaf = k.split("/")
        state[part].setdefault(name, {})[leaf] = jnp.asarray(z[k], jnp.float32)
    return state


def save_lora(path, state, tcfg: TrainConfig, plans, num_blocks: int,
              extra: Optional[Dict] = None) -> Path:
    """导出可直接给 ComfyUI 用的适配器文件（只有 master，不含优化器矩）。

    state 里是 scan 布局，先 `adapters.unstack` 回 `blocks.{i}.{target}` 扁平键、
    按各块自己的 rank 切片，再交给 export.py —— 那才是仓库/ComfyUI 认的键名。
    """
    flat = AD.unstack(state["master"], plans, num_blocks, tcfg.adapter)
    a = tcfg.adapter
    meta = {"step": str(int(state["step"])), "lora_type": a.kind,
            "lora_variant": a.variant, "rank": str(a.rank),
            "alpha": str(a.alpha if a.alpha is not None else a.rank),
            "lokr_factor": str(a.factor), "remat": tcfg.remat,
            "t_mode": tcfg.flow.t_mode, "flow_shift": str(tcfg.flow.flow_shift),
            "loss_type": tcfg.flow.loss_type, "lr": str(tcfg.adamw.lr),
            **{k: str(v) for k, v in (extra or {}).items()}}
    return EX.export_adapter(path, flat, meta)


def _jsonable(obj):
    d = asdict(obj) if hasattr(obj, "__dataclass_fields__") else obj
    if isinstance(d, dict):
        return {k: _jsonable(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_jsonable(v) for v in d]
    return d if isinstance(d, (int, float, str, bool, type(None))) else str(d)


# ═════════════════════════════════════════════════════════════════════════════
# Krea2（单流 MMDiT + FSDP）——与上面 Anima 部分共用 _loss_core / optim / flow /
# auxloss，差别只在：① 前向是 krea2_jax（text+image 同序列）；② 权重分片，
# all_gather 在 remat 内（spmd_probe P1）；③ loss 只作用在图像流上。
# ═════════════════════════════════════════════════════════════════════════════
def init_adapter_k2(key, model_cfg: K2.Krea2Config, acfg: AD.AdapterConfig,
                    targets, base_row_sq=None):
    """Krea2 适配器初始化（见 krea2_jax.init_adapters_k2）。"""
    return K2.init_adapters_k2(key, model_cfg, acfg, targets, base_row_sq)


def local_loss_k2(lora, consts, params, batch, model_cfg: K2.Krea2Config,
                  tcfg: TrainConfig, plans, main_fn, rf_fn, key,
                  txt_pos, img_pos, g: int, mesh_axis: str = "d",
                  fsdp_ndev: int = 8):
    """Krea2 打包布局下一个 pack 在**本卡**上的 loss。

    与 Anima 的 local_loss 的差别只有"模型怎么跑"：加噪/目标/逐图 loss/
    aux（eisbach/ΔFM/spectral）全部复用 —— 那些数学只作用在**图像流**
    [Σimg_q, 64] 上，与文本怎么进模型无关。
    """
    b = batch
    seg_i = b["mod_index_i"]                       # 图像流 token → 图
    lat = b["latent"]                              # [Σimg_q, 64]
    ssum = lambda x: X.seg_sum(x, seg_i, g)
    bcast = lambda v: jnp.take(v, seg_i)

    k_noise, k_drop, k_neg = jax.random.split(key, 3)
    noise = F.immiscible_noise(k_noise, lat, b["loss_mask"], ssum, bcast,
                               tcfg.flow.immiscible_k)
    noisy, target = F.make_noisy_and_target(lat, noise, bcast(b["t"])[:, None])

    drop = AD.sample_dropout(k_drop, tcfg.adapter, plans)
    ctxs = K2.split_ctx_regions(lora, consts, drop, tcfg.adapter)
    pred = K2.forward_packed(
        params, model_cfg, noisy.astype(tcfg.dtype), b["t"], b["txt"],
        b["rows_c"], b["cols_c"], b["mod_index_c"], txt_pos, img_pos,
        main_fn, rf_fn, K2.attention_dense_gqa,
        loras=ctxs, remat=tcfg.remat, mesh_axis=mesh_axis, fsdp_ndev=fsdp_ndev)

    # _loss_core 要的 b 是图像流的键名（rows/cols/loss_mask/t/ms_weight）
    b2 = {"t": b["t"], "loss_mask": b["loss_mask"],
          "rows": b["rows_i"], "cols": b["cols_i"]}
    if "ms_weight" in b:
        b2["ms_weight"] = b["ms_weight"]
    return _loss_core(pred, lat, noisy, target, b2, g, seg_i, tcfg, k_neg,
                      tcfg.aux.canvas_hw)


def fsdp_pspec(params, mesh_axis: str = "d", ndev: int = 8):
    """由**真实**权重树推 shard_map 的 in_spec：优先读数组自身的分片布局
    （加载/堆叠时定下的，最权威），host 侧未分片的数组才回退到
    krea2_jax.fsdp_shard_pred 的按形状判据。"""
    from jax.sharding import PartitionSpec as P

    def spec(x):
        sh = getattr(x, "sharding", None)
        s = getattr(sh, "spec", None)
        if s is not None:
            return s
        return (P(*([None] * (x.ndim - 1) + [mesh_axis]))
                if K2.fsdp_shard_pred(x.shape, ndev) else P())
    return jax.tree.map(spec, params)


def make_grad_fn_k2(model_cfg: K2.Krea2Config, tcfg: TrainConfig, plans,
                    layout: K2Layout, mesh, pspec, interpret: bool = False,
                    grad: bool = True, mesh_axis: str = "d"):
    """为一种 K2 布局编译"求梯度"函数。签名与 `make_grad_fn` 一致。

    两种注意力几何（见 krea2_jax 模块头）：
      * 主序列块对角：段 = combined（text+image），48 Q 头（KV 12→48 展开）
      * refiner 块对角：段 = text 槽，20 头
      * layerwise：逐 token 12×12 稠密（无 mask，不需要内核）

    `pspec` = fsdp_pspec(真实权重树)：分片权重的 in_spec。
    """
    from jax.sharding import PartitionSpec as P

    coarse = list(layout.seg_lens)
    txtc = list(layout.txt_segs)
    g = layout.n_seg
    txt_pos, img_pos = (jnp.asarray(a) for a in layout.static_positions())

    main_attn = AT.make_splash_attn(coarse, coarse, model_cfg.heads,
                                    model_cfg.head_dim, interpret=interpret,
                                    # seg_cap 只统计实段（排除 <1024 的纯填充段），
                                    # 同 make_grad_fn 的注记。
                                    seg_cap=min(layout.real_seg_lens))
    rf_attn = AT.make_splash_attn(txtc, txtc, model_cfg.txtheads,
                                  model_cfg.txt_head_dim, interpret=interpret)
    # GQA：splash 只认等头数，KV 头在内核外 repeat_interleave 展开（krea2_jax
    # 的 wrap_attn_gqa；与 torch 侧 repeat_interleave 逐 bit 相同）。
    # refiner 也要套：发布构型 txtheads == txtkvheads == 20 时 expand_gqa 是恒等
    # return（krea2_jax.py:230-231），零代价；但 _infer_config 推得出
    # txtkvheads != txtheads 的构型，那时 20 头 q / N 头 kv 直接喂给只认等头数的
    # splash 就错了。套上就永远对。
    main_gqa = K2.wrap_attn_gqa(main_attn, model_cfg.heads)
    rf_gqa = K2.wrap_attn_gqa(rf_attn, model_cfg.txtheads)

    def per_shard(lora, consts, params, b, key):
        self_fn = AT.bind_segments(main_gqa, b["seg_self"][0], b["seg_self"][0])
        rf_fn = AT.bind_segments(rf_gqa, b["txt_fine"][0], b["txt_fine"][0])
        one = {k: v[0] for k, v in b.items()}
        kk = jax.random.fold_in(key, jax.lax.axis_index(mesh_axis))
        loss, per = local_loss_k2(lora, consts, params, one, model_cfg, tcfg,
                                  plans, self_fn, rf_fn, kk, txt_pos, img_pos,
                                  g, mesh_axis,
                                  fsdp_ndev=mesh.shape[mesh_axis])
        return loss[None], per[None]

    dspec = {k: P(mesh_axis) for k in
             ("latent", "t", "txt", "rows_c", "cols_c", "mod_index_c", "seg_self",
              "txt_fine", "rows_i", "cols_i", "mod_index_i", "loss_mask",
              "ms_weight")}
    return _finish_grad_fn_k2(per_shard, mesh, pspec, dspec, tcfg, grad)


def _finish_grad_fn_k2(per_shard, mesh, pspec, dspec, tcfg: TrainConfig,
                       grad: bool = True):
    """k2 版 _finish_grad_fn：唯一差别是 params 的 in_spec 是分片树（pspec），
    其余（lora/consts 复制、梯度转置自动 psum、前向降 dtype 但 dora 保 fp32）
    与 Anima 相同。"""
    from jax.sharding import PartitionSpec as P

    smapped = _shard_map(per_shard, mesh,
                         (P(), P(), pspec, dspec, P()), (P("d"), P("d")))

    def loss_fn(lora, consts, params, b, key):
        loss, per = smapped(lora, consts, params, b, key)
        return jnp.mean(loss), per

    if not grad:
        @jax.jit
        def eval_fn(lora, consts, params, b, key):
            fwd = _fwd_cast(lora, tcfg.dtype)          # dora 保 fp32，见 _fwd_cast
            loss, per = loss_fn(fwd, consts, params, b, key)
            return loss, per, None
        return eval_fn

    @jax.jit
    def grad_fn(lora, consts, params, b, key):
        fwd = _fwd_cast(lora, tcfg.dtype)              # dora 保 fp32，见 _fwd_cast
        (loss, per), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            fwd, consts, params, b, key)
        return loss, per, grads
    return grad_fn


def assemble_batch_k2(packs, latents, ctxs, t: np.ndarray,
                      model_cfg: K2.Krea2Config, dtype, ms_weight=None
                      ) -> Dict[str, jnp.ndarray]:
    """把同布局的 N 个 K2 pack 拼成 [N, ...] 的 batch（K2Pack 版 assemble_batch）。

    latents[i][j] = 第 i 个 pack 第 j 段的 patchified latent [n_j, 64]（纯填充段 None）
    ctxs[i][j]    = 同上的**变长**文本特征 [L_j, 12, txtdim]（纯填充段 None）
    t: [N*G]（G = n_seg 含填充段；填充段的 t 采样了但被 loss_mask/valid 丢掉，
        与 Anima 同口径）
    """
    layout = packs[0].layout
    if any(p.layout != layout for p in packs):
        raise ValueError("同一步的 pack 必须同布局（一个编译产物只有一种 mask）")
    B, G, N = layout.total_len, layout.n_seg, len(packs)
    NI, NT = layout.n_img_tokens, layout.n_txt_tokens
    if t.shape != (N * G,):
        raise ValueError(f"t 的长度 {t.shape} != pack 数 × 段数 = {N * G}")
    n_layers = model_cfg.txtlayers
    keys = ("latent", "txt", "rows_c", "cols_c", "mod_index_c", "seg_self",
            "txt_fine", "rows_i", "cols_i", "mod_index_i", "loss_mask")
    out: Dict[str, list] = {k: [] for k in keys}
    for p, lat_list, ctx_list in zip(packs, latents, ctxs):
        idx = p.index_arrays()
        if len(lat_list) != len(layout.txt_segs) or len(ctx_list) != len(layout.txt_segs):
            raise ValueError(
                f"latents/ctxs 每 pack 必须给齐 {len(layout.txt_segs)} 个实段"
                f"（纯填充段不在其列），拿到 {len(lat_list)}/{len(ctx_list)}")
        lat = np.zeros((NI, model_cfg.in_dim), np.float32)
        # txt 缓存多半是 uint16 位模式（bf16）—— 零数组与赋值都按来料 dtype，
        # bitcast 留到最后一次性做（见 data._read_ctx 的内存注记）。
        c0 = next((c for c in ctx_list if c is not None), None)
        txt_dt = c0.dtype if c0 is not None else np.float32
        txt = np.zeros((NT, n_layers, model_cfg.txtdim), txt_dt)
        i_off = t_off = 0
        c_off = 0                              # combined 序列的段偏移（text 实位在段首）
        for j, (tq, iq) in enumerate(zip(layout.txt_segs, layout.img_segs)):
            L_ = lat_list[j]
            if L_ is not None:
                lat[i_off:i_off + L_.shape[0]] = L_
            C = ctx_list[j]
            if C is not None:
                if C.shape[0] > tq:
                    raise ValueError(
                        f"第 {j} 段文本 {C.shape[0]} token > 槽长 {tq}：caption "
                        f"dropout 换入的空 caption 比原 caption 还长？这不允许 —— "
                        f"空 caption 的 token 数必须 <= 槽长")
                txt[t_off:t_off + C.shape[0]] = C
                rt = p.real_txt_lens[j]
                if C.shape[0] < rt:
                    # caption dropout 换入的空 caption 比扫描时的原 caption 短：
                    # 精细 mask 必须按**实际填入**重标，否则 [len, rt) 的零值被当成
                    # 有效 caption —— refiner 里掺零 logit 稀释注意力，主序列里被
                    # txtmlp 的 bias 穿透成非零假 token 参与图像注意力（dropout
                    # 样本不再是干净的无条件）。seg_self/txt_fine 是运行时数组，
                    # 不进编译身份；mod_index/rows/cols 不受影响（调制按图、
                    # text 位坐标恒 0）。
                    idx["txt_fine"][t_off + C.shape[0]:t_off + rt] = PAD_SEG
                    idx["seg_self"][c_off + C.shape[0]:c_off + rt] = PAD_SEG
            i_off += iq
            t_off += tq
            c_off += layout.seg_lens[j]
        out["latent"].append(lat)
        out["txt"].append(txt)
        for k in ("rows_c", "cols_c", "mod_index_c", "seg_self", "txt_fine",
                  "rows_i", "cols_i", "mod_index_i", "loss_mask"):
            out[k].append(idx[k])

    batch = {k: jnp.asarray(np.stack(v)) for k, v in out.items()}
    if batch["txt"].dtype == jnp.uint16:
        batch["txt"] = jax.lax.bitcast_convert_type(batch["txt"], jnp.bfloat16)
    batch["txt"] = batch["txt"].astype(dtype)
    batch["t"] = jnp.asarray(t.reshape(N, G).astype(np.float32))
    # ms_weight 恒存在（同 assemble_batch 的理由：键集进 in_specs，全 1 会被 XLA 折叠）
    batch["ms_weight"] = jnp.asarray(
        np.ones((N, G), np.float32) if ms_weight is None
        else np.asarray(ms_weight, np.float32).reshape(N, G))
    return batch


def save_lora_k2(path, state, tcfg: TrainConfig, plans,
                 extra: Optional[Dict] = None) -> Path:
    """Krea2 适配器导出（torch 模块路径命名，ComfyUI 与仓库 PyTorch 侧同键名）。"""
    flat = K2.unstack_k2(state["master"], plans, tcfg.adapter)
    a = tcfg.adapter
    meta = {"step": str(int(state["step"])), "model_family": "krea2",
            "lora_type": a.kind, "lora_variant": a.variant, "rank": str(a.rank),
            "alpha": str(a.alpha if a.alpha is not None else a.rank),
            "lokr_factor": str(a.factor), "remat": tcfg.remat,
            "t_mode": tcfg.flow.t_mode, "flow_shift": str(tcfg.flow.flow_shift),
            "loss_type": tcfg.flow.loss_type, "lr": str(tcfg.adamw.lr),
            **{k: str(v) for k, v in (extra or {}).items()}}
    return EX.export_adapter(path, flat, meta)
