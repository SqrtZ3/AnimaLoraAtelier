"""Krea 2 (K2) SingleStreamDiT 的纯 JAX 前向（TPU 路线）——与
`models/krea2_modeling.py` 逐算子对齐（后者是官方 krea-ai/krea-2 推理实现的
训练侧移植，本文件是它的下一代亲属）。

## 与 Anima 的结构差异（决定本文件形态的）

  * **单流 MMDiT**：text token 与 image token 拼**一条**序列共享 attention/MLP
    （text 在前）。没有 cross-attn —— 每图的段 = [该图 text ; 该图 image]，
    块对角注意力按段隔离。文本特征是 Qwen3-VL **12 层** hidden states 的堆叠
    [ΣL, 12, 2560]，DiT 内 TextFusion 先沿层轴融合（2 block + Linear(12→1)）
    再沿序列轴精修（2 block，按 caption 块对角隔离），txtmlp 桥到 6144。
    所以本文件有**三种注意力几何**：主序列块对角（combined 段）、refiner 块对角
    （text 段）、layerwise 的逐 token 12×12 稠密小注意力。
  * **GQA**（48 Q 头 / 12 KV 头，headdim 128）+ attention 输出乘 sigmoid 门控
    （`wo(attn * sigmoid(gate(x)))`）。splash 不认 GQA，按 torch 同款
    `repeat_interleave` 展开 KV 头（分组约定与 SDPA enable_gqa 逐 bit 相同）。
  * **12.16B 参数 bf16 ≈ 24GB > 单 chip 15.7GiB** —— 必须 FSDP：权重沿输入维
    按 8 卡分片（每卡 ~3GB），每层在 **remat 边界内** all_gather 回全量、用完即弃
    （真机 spmd_probe P1 的方案与账目：9.8k tok/s @ L=16384/卡、单卡 3.05GiB；
    all_gather 放 checkpoint 外会让 28 层全量权重同时活着 = OOM 37.62G）。
  * **调制是 DoubleSharedModulation**：共享 tproj(t_vec) [G, 6F] + 每块裸 bias
    `mod.lin`（`vec + lin` → 6 chunk），替代 Anima 的 per-block AdaLN MLP。
    `mod.lin` 是裸张量不是 Linear，按官方口径**不挂 LoRA**。
  * **zero-center RMSNorm**（weight = scale + 1，fp32 算完再降 dtype）、QKNorm
    （逐 head_dim）、SwiGLU、GELU **tanh 近似**（Anima 是 erf 精确式，别抄错）。
  * **3D axial RoPE**：axes = [32, 48, 48] @ headdim 128，theta=1e3，
    **interleaved 对偶**形式（GPT-J 式，x 的相邻两维成对），fp32 计算；
    text token 的 pos 全 0（cos=1/sin=0 = 恒等，不需要特殊分支）。
  * **输出天然 (c ph pw) 序**（官方 sampling.py 直接按此 unpatchify），
    不需要 Anima 那个 (ph pw pt c)->(c pt ph pw) 的重排。

## 对齐依据（逐条指向 PyTorch 源）

  models/krea2_modeling.py:499   RMSNorm（zero-center，fp32 全程、末尾 .to(dtype)）
  models/krea2_modeling.py:518   QKNorm（逐 head_dim 的 RMSNorm，eps=1e-5）
  models/krea2_modeling.py:403   temb（t×1e3，period 1e4，cat(cos,sin)，cos 在前）
  models/krea2_modeling.py:272   rope/ropeapply（interleaved 对偶，fp32；freqs 矩阵
                                 [[cos,-sin],[sin,cos]]）
  models/krea2_modeling.py:705   posemb axes 切分（headdim−12·⌊hd/16⌋ / 6·⌊hd/16⌋ ×2）
  models/krea2_modeling.py:541   Attention（wq/wk/wv/gate 同源、QKNorm、sigmoid 门控
                                 在 wo **之前**）
  models/krea2_modeling.py:528   SwiGLU（mlpdim = ⌈int(2f/3)·mult⌉_128）
  models/krea2_modeling.py:471   DoubleSharedModulation（vec + lin → 6 chunk）
  models/krea2_modeling.py:657   SingleStreamBlock（pre/post 两组调制 + gate）
  models/krea2_modeling.py:612   TextFusionTransformer（2 layerwise + projector + 2 refiner）
  models/krea2_modeling.py:570   LastLayer（SimpleModulation：scale/shift = tvec + lin[2]）
  models/krea2_modeling.py:958   forward_packed_navit（逐图 [txt; img] 段组装、
                                 text pos=0 / image pos=(0,row,col)、mod_index 逐图调制）

本文件只做**前向**；优化器/数据/采样与 Anima 共用（flow/auxloss/optim/sched）。
"""

from __future__ import annotations

import json
import math
import os
import struct
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

try:
    from . import adapters as AD
    from . import anima_jax as A
except ImportError:                     # jax_tpu/ 直接在 sys.path 上（tests/ 走这条）
    import adapters as AD
    import anima_jax as A

PyTree = Any

#: LoRA 上下文的四个区域：主块栈（scan）、txtfusion 两个子栈、单例模块。
#: 每个区域是一棵 LoraCtx（键是**区域相对**名，如 "attn.wq" / "txtmlp.1"）。
CTX_REGIONS = ("blocks", "lw", "rf", "single")


# ── 构型（KREA2_LARGE_WIDE，与 krea2_modeling.py:442 相同）─────────────────────
@dataclass(frozen=True)
class Krea2Config:
    features: int = 6144
    tdim: int = 256
    txtdim: int = 2560
    heads: int = 48
    kvheads: int = 12
    multiplier: int = 4
    layers: int = 28
    patch: int = 2
    channels: int = 16
    theta: float = 1e3
    txtheads: int = 20
    txtkvheads: int = 20
    txtlayers: int = 12
    eps_rms: float = 1e-5

    @property
    def head_dim(self) -> int:
        return self.features // self.heads            # 128

    @property
    def txt_head_dim(self) -> int:
        return self.txtdim // self.txtheads           # 128

    @property
    def mlpdim(self) -> int:
        """SwiGLU 隐宽：⌈int(2f/3)·mult⌉ 到 128 的倍数（krea2_modeling.py:531）。"""
        d = int(2 * self.features / 3) * self.multiplier
        return 128 * ((d + 127) // 128)               # 16384

    @property
    def txt_mlpdim(self) -> int:
        d = int(2 * self.txtdim / 3) * self.multiplier
        return 128 * ((d + 127) // 128)               # 6912

    @property
    def in_dim(self) -> int:
        return self.channels * self.patch * self.patch   # 64（无 Anima 的 padding 通道）

    @property
    def rope_axes(self) -> Tuple[int, int, int]:
        """krea2_modeling.py:710 —— headdim 在 (frame, h, w) 三轴上的切分。"""
        hd = self.head_dim
        a = 6 * (hd // 16)
        return (hd - 2 * a, a, a)                     # (32, 48, 48) @ hd=128


# ── 基础算子 ──────────────────────────────────────────────────────────────────
def rms_norm_zc(x: jnp.ndarray, scale: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    """zero-center RMSNorm（krea2_modeling.py:510）。

    torch 是 `F.rms_norm(x.float(), (D,), eps, weight=scale.float()+1).to(dtype)`
    —— **fp32 全程**（含 weight 乘法），最后才降回 x.dtype。与 Anima 的"先降回
    再乘 weight"顺序不同，照抄各家的。
    """
    xf = x.astype(jnp.float32)
    w = scale.astype(jnp.float32) + 1.0
    out = xf * jax.lax.rsqrt(jnp.mean(xf ** 2, -1, keepdims=True) + eps) * w
    return out.astype(x.dtype)


def gelu_tanh(x: jnp.ndarray) -> jnp.ndarray:
    """nn.GELU(approximate="tanh")（tmlp/tproj/txtmlp 三处；**不是** Anima 的 erf 精确式）。"""
    return jax.nn.gelu(x, approximate=True)


def temb(t: jnp.ndarray, dim: int, period: float = 1e4, tfactor: float = 1e3
         ) -> jnp.ndarray:
    """krea2_modeling.py:403。freqs = exp(−log(period)·arange(half)/half)；
    args = (t·tfactor)·freqs；**cat(cos, sin)**（cos 在前，与 Anima 同序）。

    t: [G] fp32；返回 [G, dim] fp32（调用方降 dtype）。
    """
    half = dim // 2
    freqs = jnp.exp(-math.log(period)
                    * jnp.arange(half, dtype=jnp.float32) / float(half))
    args = (t.astype(jnp.float32) * tfactor)[:, None] * freqs[None, :]
    return jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)


# ── RoPE（3D axial，interleaved 对偶）──────────────────────────────────────────
def rope_cos_sin(rows: jnp.ndarray, cols: jnp.ndarray,
                 axes: Tuple[int, int, int] = (32, 48, 48), theta: float = 1e3
                 ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """逐 token 的 (cos, sin)，形状 [..., Σaxes/2]（frame/h/w 三轴频率 concat）。

    rows/cols 是 patch 网格坐标；**text token 与填充位置传 0** —— cos(0)=1/sin(0)=0
    是恒等，与官方 "text pos 全 0" 严格一致（krea2_modeling.py:1064-1066 的 pos
    组装；frame 轴恒 0）。

    torch 侧 `rope()` 用 float64 算频率再 .float()；这里直接用 fp32。差异**实测**
    （对 float64 参考）：pos ≤ 128 时 |Δcos|max = 4.32e-06、|Δsin|max = 4.45e-06；
    pos ≤ 4096 时涨到 1.46e-04（误差随 pos·ω 线性放大）。
    别按"fp32 相对误差 ~1e-7"估这个数（这里曾经就是那么写的）—— 差两个数量级，
    会让人以为 `--tol 1e-6` 的闸门安全，实际 128 格的网格就已经顶到 4e-6。
    对拍闸门（check_krea2_parity 默认 tol=1e-4）覆盖这一差异。
    """
    freqs = []
    for d in axes:
        scale = jnp.arange(0, d, 2, dtype=jnp.float32) / float(d)
        freqs.append(1.0 / (theta ** scale))          # [d/2]
    zeros = jnp.zeros(rows.shape, jnp.float32)
    ang = jnp.concatenate([zeros[..., None] * freqs[0],
                           rows.astype(jnp.float32)[..., None] * freqs[1],
                           cols.astype(jnp.float32)[..., None] * freqs[2]], axis=-1)
    return jnp.cos(ang), jnp.sin(ang)


def apply_rope_interleaved(x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray
                           ) -> jnp.ndarray:
    """krea2_modeling.py:283 ropeapply：相邻两维成对的旋转，**fp32** 计算后降回。

    x: [..., S, H, D]；cos/sin: [..., S, D//2]（head 维上没有位置差，广播）。
      out_2i   = cos·x_2i − sin·x_2i+1
      out_2i+1 = sin·x_2i + cos·x_2i+1
    """
    xf = x.astype(jnp.float32)
    pairs = xf.reshape(*xf.shape[:-1], xf.shape[-1] // 2, 2)
    x0, x1 = pairs[..., 0], pairs[..., 1]
    c = cos[..., None, :].astype(jnp.float32)         # [..., S, 1, D//2]
    s = sin[..., None, :].astype(jnp.float32)
    out = jnp.stack([c * x0 - s * x1, s * x0 + c * x1], axis=-1)
    return out.reshape(x.shape).astype(x.dtype)


# ── LoRA 上下文（复用 anima_jax 的 LoraCtx/_lora，键为区域相对名）──────────────
LoraCtx = A.LoraCtx
_lora = A._lora
_slice_ctx = A._slice_ctx


def dense(x: jnp.ndarray, w: jnp.ndarray, ad=None,
          b: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """y = x @ Wᵀ (+ b)，w 按 torch nn.Linear [out, in] 存放。"""
    y = x @ w.T.astype(x.dtype) if ad is None else AD.apply(x, w, *ad)
    return y if b is None else y + b.astype(x.dtype)


# ── 注意力 ────────────────────────────────────────────────────────────────────
def expand_gqa(k: jnp.ndarray, v: jnp.ndarray, q_heads: int
               ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """KV 头 repeat_interleave 展开到 Q 头数（krea2_modeling.py:373 同款）。

    [..., S, Hkv, D] → [..., S, Hq, D]；kv 头 j ↔ q 头 [j·rep, (j+1)·rep)，
    与 torch.repeat_interleave / SDPA enable_gqa 的分组约定逐 bit 相同。
    splash 只认等头数 MHA，所以必须在进内核前展开。
    """
    hkv = k.shape[-2]
    if hkv == q_heads:
        return k, v
    rep = q_heads // hkv
    return jnp.repeat(k, rep, axis=-2), jnp.repeat(v, rep, axis=-2)


def wrap_attn_gqa(attn_fn: Callable, q_heads: int) -> Callable:
    """把只认等头数的注意力后端包成认 GQA 的（KV 展开在内核外完成）。

    `*args` 透传（attention.bind_segments 绑进来的运行时 segment_ids）。
    """
    return lambda q, k, v, *args: attn_fn(q, *expand_gqa(k, v, q_heads), *args)


def attention_dense_gqa(q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray,
                        bias: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """稠密参考（本地对拍/小形状用）。q [..., S, Hq, D]，k/v [..., T, Hkv, D]。

    softmax 在 fp32（与 SDPA 一致）；GQA 在内部展开。
    """
    k, v = expand_gqa(k, v, q.shape[-2])
    d = q.shape[-1]
    logits = jnp.einsum("...shd,...thd->...hst", q, k).astype(jnp.float32) \
        / math.sqrt(d)
    if bias is not None:
        logits = logits + bias
    w = jax.nn.softmax(logits, axis=-1).astype(v.dtype)
    return jnp.einsum("...hst,...thd->...shd", w, v)


# ── 模块级前向 ────────────────────────────────────────────────────────────────
def attn_forward(x: jnp.ndarray, p: Dict[str, Any], heads: int, head_dim: int,
                 eps_qk: float, attn_fn, cos=None, sin=None,
                 loras=None, lp: str = "") -> jnp.ndarray:
    """krea2_modeling.py:556 Attention.forward（含 QKNorm / 可选 RoPE / sigmoid 门控）。

    x: [..., S, F]；attn_fn(q, k, v) 吃 [..., S, H, D] 布局（GQA 展开在后端
    wrapper 里）。`lp` 是 LoRA 键前缀（如 "attn."）。
    """
    q = dense(x, p["wq"], _lora(loras, f"{lp}wq"))
    k = dense(x, p["wk"], _lora(loras, f"{lp}wk"))
    v = dense(x, p["wv"], _lora(loras, f"{lp}wv"))
    g = dense(x, p["gate"], _lora(loras, f"{lp}gate"))
    kv_heads = p["wk"].shape[0] // head_dim
    hd = lambda z, h: z.reshape(*z.shape[:-1], h, head_dim)
    q = rms_norm_zc(hd(q, heads), p["qnorm"], eps_qk)
    k = rms_norm_zc(hd(k, kv_heads), p["knorm"], eps_qk)
    v = hd(v, kv_heads)
    if cos is not None:
        q, k = apply_rope_interleaved(q, cos, sin), apply_rope_interleaved(k, cos, sin)
    a = attn_fn(q, k, v)
    a = a.reshape(*a.shape[:-2], heads * head_dim)
    a = a * jax.nn.sigmoid(g)                          # 门控在 wo 之前（:566）
    return dense(a, p["wo"], _lora(loras, f"{lp}wo"))


def swiglu_forward(x: jnp.ndarray, p: Dict[str, Any],
                   loras=None, lp: str = "") -> jnp.ndarray:
    """krea2_modeling.py:537 —— down(silu(gate(x)) * up(x))。"""
    h = jax.nn.silu(dense(x, p["gate"], _lora(loras, f"{lp}gate")))
    h = h * dense(x, p["up"], _lora(loras, f"{lp}up"))
    return dense(h, p["down"], _lora(loras, f"{lp}down"))


def txtfusion_block(x: jnp.ndarray, p: Dict[str, Any], heads: int, head_dim: int,
                    eps: float, attn_fn, loras=None, lp: str = "") -> jnp.ndarray:
    """krea2_modeling.py:597 TextFusionBlock（无调制、无 RoPE）。"""
    h = rms_norm_zc(x, p["prenorm"], eps)
    x = x + attn_forward(h, p["attn"], heads, head_dim, eps, attn_fn,
                         cos=None, sin=None, loras=loras, lp=f"{lp}attn.")
    h = rms_norm_zc(x, p["postnorm"], eps)
    return x + swiglu_forward(h, p["mlp"], loras=loras, lp=f"{lp}mlp.")


#: `_resolve_remat` 认的档位（anima 的 REMAT_CHOICES 减去 `dots`）。
#: `none` 必须留着 —— `forward_packed` 的文本栈内层就是用它关掉嵌套 remat
#: （本文件 :736 与 tests/check_krea2_parity.py:106）；FSDP 下不许用 none 的
#: 判据在 `forward_packed` 的护栏里，不在这一层。
REMAT_CHOICES_K2 = ("full", "every2", "none")


def _norm_remat(remat):
    """旧签名归一：True -> "full"，False -> "none"（anima_jax.resolve_remat:405 同款）。"""
    if remat is True:
        return "full"
    if remat is False:
        return "none"
    return remat


def _resolve_remat(remat):
    """(fn, layer_idx)->fn 的包装。档位语义见 anima_jax.resolve_remat。

    **K2 路径禁 `dots`**：`dots_saveable` 把 matmul 的输入存给反向，而 FSDP 下
    dense 的输入正是 `gather_sharded` 当层 gather 出的**全量**权重（见
    gather_sharded 的"必须被 jax.checkpoint 包住"）—— 于是 28 层全量权重同时
    活着，正是探针第四跑 OOM 37.62G 的形态。语法上它曾被放行（只有 none 在
    forward_packed / run_train 被拦），真机上是一条静默的死路，这里 fail-fast。

    **未知档位一律 raise**（anima_jax.resolve_remat:409 同款白名单）：这里曾经
    是"落到末尾就当 full"的兜底，于是 "ful" / "every_2" / "None" / None 全都
    静默变成 full。而 `remat` 会被原样写进 checkpoint 元数据（train.py 的
    save_lora_k2 里 `"remat": tcfg.remat`），事后归因读到的是 "ful"、实际跑的
    是 full —— 账对不上，还没有任何线索指向拼写。
    """
    remat = _norm_remat(remat)
    if remat == "dots":
        raise ValueError(
            "Krea2 路径不支持 remat='dots'：FSDP 下 dots_saveable 会把每层 "
            "all_gather 出的全量权重存给反向（28 层同时活着 = 探针实测 37.62G "
            "OOM）。用 'full'（唯一在真机验证过的档）。")
    if remat not in REMAT_CHOICES_K2:
        raise ValueError(f"krea2 的 remat 只能是 {REMAT_CHOICES_K2} 之一"
                         f"（或 True/False），得到 {remat!r}")
    if remat == "none":
        return lambda fn, i: fn
    if remat == "every2":
        return lambda fn, i: fn if i % 2 == 0 else jax.checkpoint(fn)
    return lambda fn, i: jax.checkpoint(fn)


def text_fusion(params: Dict[str, Any], cfg: Krea2Config, x_txt: jnp.ndarray,
                layerwise_attn_fn, refiner_attn_fn,
                loras=None, remat: str = "full") -> jnp.ndarray:
    """krea2_modeling.py:626 TextFusionTransformer.forward。

    x_txt: [ΣL, n_layers=12, txtdim]（各图 caption 有效 token 按图序拼接，
    量化填充位补 0 —— refiner 的块对角精细 mask 会把它们隔掉）。
    返回 [ΣL, txtdim]（未过 txtmlp）。

    `loras`：{"lw": ctx(2 层堆叠), "rf": ctx(2 层堆叠), "single": ctx(单例)}。
    """
    wrap = _resolve_remat(remat)
    hd = cfg.txt_head_dim
    lw = None if loras is None else loras.get("lw")
    rf = None if loras is None else loras.get("rf")
    sg = None if loras is None else loras.get("single")

    x = x_txt
    for i in range(2):
        p = params["layerwise"][i]

        def lw_block(x_, _p=p, _i=i,
                     _lo=None if lw is None else _slice_ctx(lw, i)):
            # attn_fn/loras 走闭包 —— checkpoint 的入参只能是数组
            return txtfusion_block(x_, _p, cfg.txtheads, hd, cfg.eps_rms,
                                   layerwise_attn_fn, _lo, "")
        x = wrap(lw_block, i)(x)
    # projector：Linear(12→1, bias=False)，作用在最后一维（krea2_modeling.py:648-650）
    x = dense(x.transpose(0, 2, 1), params["projector"], _lora(sg, "txtfusion.projector"))
    x = x[..., 0]                                      # [ΣL, txtdim]
    for i in range(2):
        p = params["refiner"][i]

        def rf_block(x_, _p=p, _i=i,
                     _lo=None if rf is None else _slice_ctx(rf, i)):
            return txtfusion_block(x_, _p, cfg.txtheads, hd, cfg.eps_rms,
                                   refiner_attn_fn, _lo, "")
        x = wrap(rf_block, i)(x)
    return x


def txtmlp_forward(params: Dict[str, Any], x: jnp.ndarray,
                   loras=None) -> jnp.ndarray:
    """krea2_modeling.py:739 —— RMSNorm → Linear(txtdim→F) → GELU(tanh) → Linear(F→F)。"""
    h = rms_norm_zc(x, params["txtmlp_norm"])
    h = dense(h, params["txtmlp1"]["w"], _lora(loras, "txtmlp.1"),
              b=params["txtmlp1"]["b"])
    h = gelu_tanh(h)
    return dense(h, params["txtmlp3"]["w"], _lora(loras, "txtmlp.3"),
                 b=params["txtmlp3"]["b"])


def tvec_forward(params: Dict[str, Any], cfg: Krea2Config, t: jnp.ndarray,
                 dtype, loras=None) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """逐图 timestep 向量。返回 (t_vec [G, F], tvec6 [G, 6F])。

    tmlp = Linear(tdim→F) → GELU(tanh) → Linear(F→F)；tproj = GELU(tanh) →
    Linear(F→6F)（krea2_modeling.py:730/747）。
    """
    e = temb(t, cfg.tdim).astype(dtype)                # [G, tdim]
    h = dense(e, params["tmlp0"]["w"], _lora(loras, "tmlp.0"),
              b=params["tmlp0"]["b"])
    h = gelu_tanh(h)
    t_vec = dense(h, params["tmlp2"]["w"], _lora(loras, "tmlp.2"),
                  b=params["tmlp2"]["b"])
    tvec6 = dense(gelu_tanh(t_vec), params["tproj"]["w"], _lora(loras, "tproj.1"),
                  b=params["tproj"]["b"])
    return t_vec, tvec6


def block_forward(x: jnp.ndarray, p: Dict[str, Any], cfg: Krea2Config,
                  tvec6: jnp.ndarray, mod_bcast, cos, sin, attn_fn,
                  loras=None, layer: Optional[int] = None) -> jnp.ndarray:
    """krea2_modeling.py:668 SingleStreamBlock.forward。

    tvec6: [G, 6F]（共享 tproj 输出，**未加**本块 bias）；mod_bcast 把逐图调制
    摆成可与 x 广播的形状（打包 = take(mod_index)，分桶 = [:, None]），与
    Anima 的 `mod_bcast` 同约定 —— Anima 真机 1.01MB/token 的教训同样适用于此，
    所以训练必须走 scan（跨块提升被语义性挡住），展开路径的显存账另算。
    """
    mod = tvec6 + p["mod_lin"].astype(tvec6.dtype)     # [G, 6F]（DoubleSharedModulation）
    # **先广播再 split**（同 anima_jax.block_forward 的摆法：搬运相同、少算子）
    prescale, preshift, pregate, postscale, postshift, postgate = \
        jnp.split(mod_bcast(mod), 6, axis=-1)
    lp = "" if layer is None else f"blocks.{layer}."

    h = (1.0 + prescale) * rms_norm_zc(x, p["prenorm"], cfg.eps_rms) + preshift
    x = x + pregate * attn_forward(h, p["attn"], cfg.heads, cfg.head_dim,
                                   cfg.eps_rms, attn_fn, cos, sin,
                                   loras=loras, lp=f"{lp}attn.")
    h = (1.0 + postscale) * rms_norm_zc(x, p["postnorm"], cfg.eps_rms) + postshift
    x = x + postgate * swiglu_forward(h, p["mlp"], loras=loras, lp=f"{lp}mlp.")
    return x


def last_forward(x: jnp.ndarray, p: Dict[str, Any], cfg: Krea2Config,
                 t_vec: jnp.ndarray, mod_bcast, loras=None) -> jnp.ndarray:
    """krea2_modeling.py:570/583 LastLayer（SimpleModulation：scale/shift = tvec+lin）。

    t_vec: [G, F]（tmlp 输出，**不是** tvec6）。返回 [..., patch²·C]（天然
    (c ph pw) 序，krea2_modeling.py:952 的恒等口径）。
    """
    scale = mod_bcast(t_vec + p["mod_lin"][0].astype(t_vec.dtype))
    shift = mod_bcast(t_vec + p["mod_lin"][1].astype(t_vec.dtype))
    x = (1.0 + scale) * rms_norm_zc(x, p["norm"], cfg.eps_rms) + shift
    return dense(x, p["linear_w"], _lora(loras, "last.linear"), b=p["linear_b"])


# ── FSDP：分片与 all_gather ────────────────────────────────────────────────────
#
# 真机 spmd_probe P1 定案的形态（9.8k tok/s @ L=16384/卡，单卡 HBM 3.05GiB）：
#   * 权重沿**输入维**按 8 卡分片（P(..., "d") 在最后一维），每卡 ~3GB；
#   * 每层在 **remat 边界内** `all_gather` 回全量、用完即弃 —— 放边界外会让
#     28 层全量权重同时活着（探针第四跑 OOM 37.62G 的根因）；
#   * 反向重算时再 gather 一遍（通信 ×2，ICI all-gather 实测 34GB/s，占步时 ~10%）。
#
# **两种尺度别混**（混过一次，见 `gather_sharded`）：
#   `fsdp_shard_pred` / `fsdp_spec` 吃**全局**形状（trace 外：加载分片、in_spec）；
#   `_gather_pred_local` 吃 shard_map **体内**的局部形状。两者不是同一个判据。
#   两者何时等价，由 `check_gather_proxy_safe` 在加载时一次性审计。
def fsdp_shard_pred(shape, ndev: int) -> bool:
    """该张量**能不能**沿最后一维按 ndev 分片。**`shape` 必须是全局形状。**

    projector [1, 12] 这类最后一维不被 ndev 整除的张量必须复制（12 % 8 ≠ 0，
    硬分片直接 IndivisibleError）。

    权威调用点只有两处，都在 trace 外、都拿全局形状：`fsdp_spec`（→
    run_train.py 的 shard_rule，加载时逐张 device_put）与 train.py:653
    `fsdp_pspec`（shard_map 的 in_spec；它优先读数组自身的 `.sharding.spec`，
    只在 host 侧未分片时回退到本判据）。
    shard_map **体内**看到的是局部形状，不能用这个判据（曾经用了 —— 见
    `_gather_pred_local` 的注记）。
    """
    return len(shape) >= 2 and shape[-1] % ndev == 0


def fsdp_spec(shape, mesh_axis: str = "d", ndev: int = 8):
    """全局形状 -> PartitionSpec：可分片的沿**最后一维**（= torch 的 in_features），
    其余复制。加载分片与 in_spec 回退共用这一份实现（run_train.py 的
    `shard_rule` 就是它；别再各写一遍 —— 判据漂移 = gather 把复制张量拼成
    ndev 份的静默错）。
    """
    from jax.sharding import PartitionSpec as P
    if not fsdp_shard_pred(shape, ndev):
        return P()
    return P(*([None] * (len(shape) - 1) + [mesh_axis]))


def check_gather_proxy_safe(shapes, ndev: int = 8, where: str = "") -> None:
    """trace 外一次性审计：`gather_sharded` 的**局部形状代理判据**在这批全局
    形状 + 这个 ndev 下是否与真相等价。不等价就 fail-fast。

    为什么需要单独一条：trace 内只看得到局部末维 `shape[-1] // ndev`，用"局部
    末维还能不能被 ndev 整除"当代理，等价于要求全局末维能被 **ndev²** 整除。
    ndev=8 + 发布构型全部满足（末维 64/256/2560/6144/6912/16384/36864 都是 64
    的倍数）；ndev=16 就漏 —— `first.weight` 全局末维 64 → 局部 4，4 % 16 ≠ 0 →
    gather 被静默跳过。运气好的形状会撞出 dot_general 维度不匹配（响），运气不
    好就是 dense 拿 1/ndev 的权重去乘全宽激活（哑：loss 只是"看起来不太对"）。

    出路是把 trace 外算好的布尔树传给 `forward_packed(fsdp_sharded=...)`
    （`fsdp_sharded_tree` 产），那条路不看形状、任何 ndev 都对。本审计只管
    "没传布尔树、要靠代理"的调用方（当前 train.py 的 FSDP 训练路径就是）。

    `shapes`：{名字: 全局形状} 或形状的可迭代对象。
    """
    items = (shapes.items() if hasattr(shapes, "items")
             else ((f"#{i}", s) for i, s in enumerate(shapes)))
    bad = [(n, tuple(s)) for n, s in items
           if fsdp_shard_pred(s, ndev) and s[-1] % (ndev * ndev)]
    if not bad:
        return
    head = ", ".join(f"{n}{s}→局部末维 {s[-1] // ndev}" for n, s in bad[:4])
    raise ValueError(
        f"{where or 'FSDP'}：{len(bad)} 个张量沿最后一维按 {ndev} 卡分片后，"
        f"局部末维不再被 {ndev} 整除，shard_map 体内的 gather 代理判据"
        f"（_gather_pred_local）会把它们误判成未分片而**跳过 all_gather**。\n"
        f"  例：{head}\n"
        f"  ndev=8 的发布构型不触发（末维都是 64 的倍数）。要跑 ndev={ndev}，"
        f"把 krea2_jax.fsdp_sharded_tree(权重树) 的结果传给 "
        f"forward_packed(fsdp_sharded=...)，别依赖代理判据。")


def _gather_pred_local(local_shape, ndev: int) -> bool:
    """`gather_sharded` 在 shard_map **体内**用的代理判据（局部形状）。

    体内看到的是**局部**形状（实测：全局 (4, 6144) 分片 P(None,'d') → 体内
    `x.shape = (4, 768)`），所以"局部末维能否再被 ndev 整除"只是个代理，
    等价于要求全局末维能被 ndev² 整除 —— 不是 `fsdp_shard_pred` 那个判据。
    代理取假时 gather 被跳过，dense 就拿 1/ndev 的权重去乘全宽激活：可能撞出
    dot_general 维度不匹配（响），也可能只是数值错（哑）。ndev=8 + 发布构型下
    代理与真相恒等，这一点由 `check_gather_proxy_safe` 在加载时审计
    （run_train.py 的 k2 路径已接）；其它构型必须走 `sharded` 显式参数。

    JAX 0.11 里体内其实有权威答案（`jax.typeof(x).manual_axis_type.varying`
    = `{'d'}`），但本仓库的 shard_map 一律带 `check_vma=False`（train.py:451
    —— splash 是 Mosaic 内核，出参复制性推不出来），实测该字段被清成空集，
    所以指望不上。
    """
    return len(local_shape) >= 2 and local_shape[-1] % ndev == 0


def gather_sharded(x: jnp.ndarray, mesh_axis: str = "d", ndev: int = 8,
                   sharded: Optional[bool] = None) -> jnp.ndarray:
    """把沿最后一维分片的权重 gather 回全量；未分片的（向量/标量/不可整除
    矩阵）原样返回。

    `sharded` 给显式布尔（trace 外算好，权威）时按它走；None 时回退到
    `_gather_pred_local` 的局部形状代理（只在 `check_gather_proxy_safe` 审计
    放行的构型下与真相等价）。

    **必须被 jax.checkpoint 包住**（调用方的责任）：gather 产物存活期被限制在
    单层前向/重算内，活着的始终只有一层份。
    """
    if sharded is None:
        sharded = _gather_pred_local(x.shape, ndev)
    if not sharded:
        return x
    return jax.lax.all_gather(x, mesh_axis, axis=-1, tiled=True)


def gather_tree(p: PyTree, mesh_axis: str = "d", ndev: int = 8,
                sharded: Optional[PyTree] = None) -> PyTree:
    """`gather_sharded` 的 pytree 版。`sharded` 是与 `p` **同构**的布尔树
    （叶子是 Python bool，静态）；None 时逐叶回退到局部形状代理。"""
    if sharded is None:
        return jax.tree.map(lambda x: gather_sharded(x, mesh_axis, ndev), p)
    return jax.tree.map(lambda x, s: gather_sharded(x, mesh_axis, ndev, bool(s)),
                        p, sharded)


def fsdp_sharded_tree(params: PyTree, mesh_axis: str = "d", ndev: int = 8
                      ) -> PyTree:
    """由**真实**（trace 外）权重树算出 `forward_packed(fsdp_sharded=...)` 要的
    布尔树：优先读数组自身的 `.sharding.spec`（加载/堆叠时定下的，最权威），
    host 侧未分片的才回退到 `fsdp_shard_pred` 的全局形状判据。

    与 train.py:653 `fsdp_pspec` 同一份事实的布尔投影 —— 那边产 in_spec，
    这边产"该不该 gather"。堆叠过的 blocks 子树前导轴是 scan 轴（spec 的第 0
    项恒 None），逐叶的布尔与切片前后无关，所以这棵树可以直接喂给 scan 体内
    切片后的单块子树。

    也吃 pspec 树（`fsdp_pspec` 的输出，叶子是 PartitionSpec）：那样两边保证
    读的是同一份事实，一行接上。
    """
    def one(x):
        from jax.sharding import PartitionSpec
        if isinstance(x, PartitionSpec):
            # 已经是 spec（`fsdp_pspec` 的输出）。注意 `type(x).__name__` 是 "P"
            # 不是 "PartitionSpec"（jax 0.11 实测），别按名字认。
            return mesh_axis in tuple(x)
        spec = getattr(getattr(x, "sharding", None), "spec", None)
        if spec is not None:
            return mesh_axis in tuple(spec)
        return fsdp_shard_pred(x.shape, ndev)
    # PartitionSpec 在 jax 0.11 里本身就是 pytree **叶子**（实测
    # `jax.tree.leaves({"a": P(None,"d")})` 原样返回它），不需要 is_leaf。
    return jax.tree.map(one, params)


def stack_blocks(params: PyTree, mesh=None) -> PyTree:
    """`params["blocks"]`（list of dict）堆成 pytree of [L, ...]，供 scan 用。

    分片布局的保持规则：**新堆叠轴复制，其余维沿用输入分片**——
    按堆叠后的形状重新推断会出错（mod_lin 每块是 1D 向量、堆叠后 [L, 6F]
    看起来像可分片的矩阵，会被错切；scan 切片后剩下 1D 碎片，gather 判据
    就再也认不出它了）。传 `mesh` 时按上述规则 device_put 一次。
    """
    from jax.sharding import NamedSharding, PartitionSpec as P

    blocks = params["blocks"]
    if not isinstance(blocks, (list, tuple)):
        return params
    stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *blocks)
    if mesh is not None:
        def spec_of(x):
            # `P() if s is None else s`：不依赖 `bool(P()) is False` 这个隐晦语义
            # （实测 bool(P())=False 而 bool(P(None))=True —— 前者会被 `or` 换成
            # P()、后者不会，同为"复制"的两种写法走两条路，看代码看不出来）。
            spec = getattr(getattr(x, "sharding", None), "spec", None)
            return P() if spec is None else spec
        specs = jax.tree.map(spec_of, blocks[0])
        stacked = jax.tree.map(
            lambda x, s: jax.device_put(x, NamedSharding(mesh, P(None, *s))),
            stacked, specs)
    return {**params, "blocks": stacked}


# ── 整模前向（NaViT packed，text+image 同序列）─────────────────────────────────
def forward_packed(params: PyTree, cfg: Krea2Config,
                   img_tokens: jnp.ndarray, timesteps: jnp.ndarray,
                   txt_stack: jnp.ndarray,
                   rows: jnp.ndarray, cols: jnp.ndarray, mod_index: jnp.ndarray,
                   txt_pos: jnp.ndarray, img_pos: jnp.ndarray,
                   main_attn_fn, refiner_attn_fn, layerwise_attn_fn,
                   loras=None, remat: str = "full",
                   mesh_axis: Optional[str] = "d", fsdp_ndev: int = 8,
                   fsdp_sharded: Optional[PyTree] = None) -> jnp.ndarray:
    """**打包（NaViT）布局**：G 张图各携自己的 caption，拼成一条单流序列。

      img_tokens  [N_img, 64]     图像 patch token（按图序拼接，含段内量化填充，
                                  填充位补 0；是**加噪后**的 token，与 Anima 同约定）
      timesteps   [G]             每图一个 t
      txt_stack   [ΣL_txt, 12, txtdim]  各图 caption 的 12 层文本特征（量化填充补 0）
      rows/cols   [B]             combined 序列逐 token 的网格坐标（text/填充位 = 0）
      mod_index   [B]             combined 序列 token → 图 的索引（调制用）
      txt_pos     [ΣL_txt]        文本 token 在 combined 序列中的位置（静态，随布局）
      img_pos     [N_img]         图像 token 在 combined 序列中的位置（静态，随布局）
      返回        [N_img, 64]     仅图像位置的 velocity（天然 (c ph pw) 序）

    `mesh_axis` 非 None 时走 FSDP：params 是分片形态，逐层在 remat 内 gather；
    None 时 params 是全量（本地对拍/单卡小模型）。FSDP 下 remat **必须是 full**
    —— none 根本没 checkpoint、every2 的偶数层没被包住，gather 出的全量权重都会
    被存给反向（28/14 层全活 = OOM，就是探针第四跑那个 37.62G）。

    `fsdp_sharded`：与 `params` 同构的**布尔** pytree（trace 外用
    `fsdp_sharded_tree(真实权重树)` 算好），逐叶告诉 gather"该不该 all_gather"。
    None 时退回 `_gather_pred_local` 的局部形状代理 —— 只在
    `check_gather_proxy_safe` 审计放行的构型（ndev=8 + 发布构型）下与真相等价，
    判据与代价见 `_gather_pred_local`。
    """
    if mesh_axis and _norm_remat(remat) != "full":
        # 只拦 full 之外的全部档（曾经只拦 none，于是展开路径 + every2 的偶数层
        # 照样把 gather 出的全量权重存给反向 —— 14 层同时活着，和 dots 同一个坑；
        # 训练路径走不到（run_train.py:573 恒 stack_blocks，scan 分支对 every2
        # 硬 raise），但直接调本函数的对拍脚本敞开着）。full 也是唯一在真机
        # 验证过的档（见上方 FSDP 小节的注记）。
        raise ValueError(
            f"FSDP（分片权重）要求 remat='full'，得到 {remat!r}：all_gather 必须"
            f"整个落在 checkpoint 边界内，否则每层 gather 出的全量权重会被存给"
            f"反向（28 层全活 = 探针实测 OOM 37.62G）。")
    # ── 形状 fail-fast（对齐 torch 侧 forward_packed_navit:988-1013 的六条）──────
    # 这几条全是**编译期静态形状**，零运行时成本。上游 assemble_batch_k2 与
    # K2Layout.__post_init__（packing.py:375-395）已挡住大部分，但 txt_pos/img_pos
    # 与实际张量长度的关系在这里才第一次凑到一起。写错了会怎样：`.at[pos].set()`
    # 对越界索引**静默丢弃**、长度不齐则直接 broadcast 报错在别处，都比在这儿
    # 说清楚难查。
    if txt_pos.shape[0] != txt_stack.shape[0]:
        raise ValueError(
            f"txt_pos 有 {txt_pos.shape[0]} 个位置，但 txt_stack 有 "
            f"{txt_stack.shape[0]} 个 token —— 布局与文本张量不是同一份账")
    if img_pos.shape[0] != img_tokens.shape[0]:
        raise ValueError(
            f"img_pos 有 {img_pos.shape[0]} 个位置，但 img_tokens 有 "
            f"{img_tokens.shape[0]} 个 token —— 布局与图像张量不是同一份账")
    if img_tokens.shape[-1] != cfg.in_dim:
        raise ValueError(
            f"img_tokens 末维 {img_tokens.shape[-1]} != cfg.in_dim {cfg.in_dim}"
            f"（patch²·C）—— first 层会直接维度不匹配，但报错点在三层之外")
    if mod_index.shape[0] != rows.shape[0] or cols.shape[0] != rows.shape[0]:
        raise ValueError(
            f"mod_index/cols/rows 长度必须同为 combined 序列长 B，得到 "
            f"{mod_index.shape[0]}/{cols.shape[0]}/{rows.shape[0]}")
    # torch 侧第六条（`min(text_seqlens) > 0`）与 `timesteps.shape[0] >
    # mod_index.max()` 在这里**做不到**：mod_index 是运行时数组（batch 的
    # "mod_index_c"，在 jit/shard_map 内是 tracer），`.max()` 取不到具体值，
    # 在 trace 内做数据依赖的 raise 就更不可能。而这条恰恰是唯一全仓无人校验的
    # 关系：mod_bcast 走 jnp.take，越界索引在默认 fill 模式下产 **NaN**
    # （实测：jnp.take(h,[99]) -> nan，不是 clamp），NaN 会一路传 28 层才在 loss
    # 上显形。真要闸它，得在 host 侧（assemble_batch_k2 / K2Layout）加断言。
    F = cfg.features
    B = rows.shape[0]
    dt = img_tokens.dtype
    if mesh_axis:
        sub = (lambda key: None) if fsdp_sharded is None \
            else (lambda key: fsdp_sharded[key])

        def g(p, sharded=None):
            return gather_tree(p, mesh_axis, fsdp_ndev, sharded)
    else:
        sub = lambda key: None                              # noqa: E731
        g = lambda p, sharded=None: p                       # noqa: E731
    wrap = _resolve_remat(remat)
    mod_bcast = lambda h: jnp.take(h, mod_index, axis=0)

    # ── 文本栈：TextFusion（refiner 按 caption 块对角）→ txtmlp ──────────────
    # 只把文本侧子树传进 checkpoint（整树传进去会把 28 个主块也当残差存起来）
    _TXT_KEYS = ("txtfusion", "txtmlp_norm", "txtmlp1", "txtmlp3")

    def _text_stack(p_txt):
        p = g(p_txt, None if fsdp_sharded is None
              else {k: sub(k) for k in _TXT_KEYS})
        sg = None if loras is None else loras.get("single")
        # 内层 remat="none"：外层 jax.checkpoint 已丢掉文本栈全部中间量（反向
        # 整栈重算一次），内层再逐块包 checkpoint 只会让每块在反向时再重算
        # 一遍（嵌套 remat 双算），一分显存都省不了。
        h = text_fusion(p["txtfusion"], cfg, txt_stack,
                        layerwise_attn_fn, refiner_attn_fn, loras=loras,
                        remat="none")
        return txtmlp_forward(p, h, loras=sg)

    txt = jax.checkpoint(_text_stack)(
        {k: params[k] for k in _TXT_KEYS})                 # [ΣL_txt, F]

    # ── 图像嵌入 + combined 组装（每图 [txt_i ; img_i]）────────────────────────
    def _img_embed(p_first, tok):
        p = g(p_first, sub("first"))
        sg = None if loras is None else loras.get("single")
        return dense(tok, p["w"], _lora(sg, "first"), b=p["b"])

    img = jax.checkpoint(_img_embed)(params["first"], img_tokens)   # [N_img, F]
    combined = jnp.zeros((B, F), dt)
    combined = combined.at[txt_pos].set(txt.astype(dt))
    combined = combined.at[img_pos].set(img)

    # ── 逐图 timestep 向量 ────────────────────────────────────────────────────
    _TV_KEYS = ("tmlp0", "tmlp2", "tproj")

    def _tvec(p_tv):
        p = g(p_tv, None if fsdp_sharded is None
              else {k: sub(k) for k in _TV_KEYS})
        sg = None if loras is None else loras.get("single")
        return tvec_forward(p, cfg, timesteps, dt, loras=sg)

    t_vec, tvec6 = jax.checkpoint(_tvec)({k: params[k] for k in _TV_KEYS})
    t_vec, tvec6 = t_vec.astype(dt), tvec6.astype(dt)

    cos, sin = rope_cos_sin(rows, cols, cfg.rope_axes, cfg.theta)

    # ── 28 个主块（scan；FSDP 时每块在 remat 内 gather）────────────────────────
    blocks = params["blocks"]
    ctx_b = None if loras is None else loras.get("blocks")
    # blocks 的布尔子树：展开态是 list（逐块取），堆叠态每叶已是"整栈同一答案"
    # （堆叠只加了个复制的前导轴，见 stack_blocks / fsdp_sharded_tree）。
    sh_b = sub("blocks")
    if isinstance(blocks, (list, tuple)):
        # 展开路径（对拍按块比对用；真机训练走 scan —— Anima 1.01 MB/token 的
        # 教训同样适用于 K2 的单流逐 token 调制）。切片后的 ctx 键是区域相对的，
        # 所以 layer=None（与 anima_jax._forward_core 的 LoraCtx 分支同约定）。
        for i in range(cfg.layers):
            lo_i = None if ctx_b is None else _slice_ctx(ctx_b, i)

            def one(carry, p, _i=i, _lo=lo_i,
                    _sh=None if sh_b is None else sh_b[i]):
                return block_forward(carry, g(p, _sh), cfg, tvec6, mod_bcast,
                                     cos, sin,
                                     main_attn_fn, loras=_lo, layer=None)
            combined = wrap(one, i)(combined, blocks[i])
    else:
        if _norm_remat(remat) == "every2":
            # scan 下 every2 要成对重排 + 两块一个循环（anima_jax 有一套现成的
            # 写法）。FSDP 的验证档是 full，先不复制那套复杂度 —— 用到再移植。
            # （不推荐 dots/none：dots 在 _resolve_remat 直接 raise、none 在 FSDP
            # 护栏 raise，三个里两个是死的。）
            raise ValueError("krea2 scan 路径暂不支持 remat='every2'"
                             "（用 full；配对重排那套用到再从 anima_jax 移植）")

        def body(carry, layer):
            p, lo = layer

            def step(c):
                return block_forward(c, g(p, sh_b), cfg, tvec6, mod_bcast, cos, sin,
                                     main_attn_fn,
                                     loras=None if ctx_b is None
                                     else A.LoraCtx(ctx_b.cfg, *lo),
                                     layer=None)
            return wrap(step, 0)(carry), None

        xs = (blocks, (None, None, None) if ctx_b is None
              else (ctx_b.params, ctx_b.consts, ctx_b.drop))
        combined, _ = jax.lax.scan(body, combined, xs, length=cfg.layers)

    # ── LastLayer + 抽取图像位 ────────────────────────────────────────────────
    def _last(p_last, x):
        p = g(p_last, sub("last"))
        sg = None if loras is None else loras.get("single")
        return last_forward(x, p, cfg, t_vec, mod_bcast, loras=sg)

    out = jax.checkpoint(_last)(params["last"], combined)   # [B, 64]
    return out[img_pos]


# ── 权重加载（raw.safetensors → JAX pytree；可选逐张即刻分片）──────────────────
#: safetensors 的 dtype 串 -> **按位读**用的 numpy dtype。BF16 走 uint16 原样搬
#: （numpy 没有 bfloat16），由 `_to_jax` 的 bitcast 还原 —— 全程零舍入。
#: 曾另有一份 `_DT`（同名键 -> jnp dtype）从来没人用，删了：`_to_jax` 只用
#: `_NP` + 硬编码 bfloat16，留着一份平行的 dtype 表迟早被改岔。
_NP = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32}


def _read_safetensors_map(path: str, token: Optional[str] = None):
    """{name: (dtype_str, shape, reader)} + __metadata__——reader() 才真的读字节
    （每次自开文件，调用方可以在头部解析完之后的任何时刻读；24GB 文件不做
    一次性物化）。

    `path` 可以是 http(s) URL（HF resolve 链接）：此时走 HTTP Range 请求——
    header 两次小 GET（8B 长度 + JSON 头），之后每 tensor 一次 range GET。
    Kaggle 的 /kaggle/working 只有 ~20GB 装不下 26GB 的 Krea-2-Raw，而
    Kaggle←HF 实测 300+MB/s，流式读比"下载落盘再读"省一整圈磁盘与等待。
    safetensors 是纯字节寻址格式，Range 语义保证**取到的那段**与原文件逐 bit
    相同（长度不符时 `_http_range_get` 直接 raise），所以数值与本地文件路径
    完全等价 —— 但它保证不了"远端这份文件本身是完整的"，那条由 HTTP 版自己的
    完整性闸门管（`Content-Range` 的 total 对头部声明的数据区末尾，与本地路径
    的 getsize 闸门同一条判据）。gated repo 传 `token`
    （Bearer），重定向到 CDN 后**不再带**（签名 URL 自含授权，多带 Authorization
    有被 CDN 拒的案例）。"""
    if str(path).startswith(("http://", "https://")):
        return _read_safetensors_map_http(path, token)
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen))
        meta = hdr.pop("__metadata__", None) or {}
        base = 8 + hlen
    # 完整性闸门：safetensors 是纯字节寻址格式，文件长度**恒等于**头部末尾加上
    # 最大 data_offset。底模现在从 Kaggle Dataset 挂载读（26GB 上传过一次网络，
    # 本地那份的 sha256 核过、Kaggle 那份没有），截断的话不加这条会等到读到某个
    # tensor 才炸在 reshape 上 —— 已经烧掉几分钟 TPU 配额，报错还看不懂。
    # 代价是一次 getsize，读零字节数据。
    end = max((m["data_offsets"][1] for m in hdr.values()), default=0)
    size = os.path.getsize(path)
    if base + end != size:
        raise ValueError(
            f"{path} 不是完整的 safetensors：头部声明数据区到 {base + end} 字节，"
            f"实际文件 {size} 字节（差 {size - base - end:+d}）。\n"
            f"  下载/上传被截断，或写盘没写完。别继续加载 —— 重新传一份。")

    def make_reader(m):
        s0, e0 = m["data_offsets"]
        shape, dt = m["shape"], _NP[m["dtype"]]

        def read() -> np.ndarray:
            with open(path, "rb") as f2:
                f2.seek(base + s0)
                raw = f2.read(e0 - s0)
            return np.frombuffer(raw, dtype=dt).reshape(shape)
        return read

    return ({name: (m["dtype"], m["shape"], make_reader(m))
             for name, m in hdr.items()}, meta)


def _http_final_url(url: str, token: Optional[str]) -> Tuple[str, Optional[int]]:
    """解析 HF resolve 的 302 链，返回 (最终 CDN 签名 URL, 远端总字节数)。
    用 Range: 0-0 探，避免 HEAD 在某些 CDN 配置下不返回签名跳转。

    总字节数取自 206 响应的 `Content-Range: bytes 0-0/<total>` —— 这一次探测
    本来就要发，顺手把 total 读出来，给完整性闸门用（见
    `_read_safetensors_map_http`）。拿不到（服务端不给 Content-Range、或格式
    不认识）时返回 None，闸门降级为跳过 —— 探测本身的成功与否不该因此翻车。
    """
    import urllib.request
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=60) as r:
        cr = r.headers.get("Content-Range") or ""
        total = None
        if "/" in cr:
            tail = cr.rsplit("/", 1)[1].strip()
            if tail.isdigit():
                total = int(tail)
        return r.geturl(), total


def _http_range_get(url: str, start: int, end_excl: int,
                    token: Optional[str] = None, tries: int = 4) -> bytes:
    """GET [start, end_excl)。带有限重试（连接重置/5xx 时指数退避）。"""
    import time
    import urllib.request
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(
                url, headers={"Range": f"bytes={start}-{end_excl - 1}"})
            if token:
                req.add_header("Authorization", f"Bearer {token}")
            with urllib.request.urlopen(req, timeout=300) as r:
                body = r.read()
            if len(body) != end_excl - start:
                raise IOError(f"range 响应长度 {len(body)} != 期望 {end_excl - start}")
            return body
        except Exception as e:  # noqa: BLE001 —— 记录后重试，最后一次抛出
            last = e
            if i + 1 < tries:
                time.sleep(2.0 * (2 ** i))
    raise RuntimeError(f"HTTP range GET [{start},{end_excl}) 重试 {tries} 次仍失败"
                       f"（最后错误：{type(last).__name__}: {last}）")


def _read_safetensors_map_http(url: str, token: Optional[str],
                               workers: int = 8,
                               inflight_budget: int = 4 << 30):
    """`_read_safetensors_map` 的 HTTP Range 版（见它的 docstring）。

    **并发预取**：一个后台喂料线程按**文件序**（= 偏移升序，对 CDN 最友好）把
    range GET 提交给线程池（串行逐 tensor 的固定开销叠加实测让 26GB 加载拖到
    10 分钟级；并发把 TLS/RTT 开销摊掉）。reader() 只是取回对应 future 的结果，
    语义与同步读完全一致。

    `inflight_budget`（默认 4GB）限制**已提交但尚未被 reader() 消费**的字节数，
    预算在 `read()` 拿到字节、放掉 future 对结果的引用之后才归还 —— 于是 host
    驻留量真的被压在 ~4GB。三处曾经不对，一起修的：
      * 归还挂在 `fut.add_done_callback` 上 = future 一**完成**就归还，而
        `Future` 会一直持有 `_result`、`entries` dict 又活到加载结束，26GB 原始
        字节全程驻留 host RAM（Kaggle host 几百 GB 才没炸），与"host 从不持有
        全量副本"的口径直接矛盾；
      * 全部 tensor 在**建 map 时**一次提交完 —— 改成消费点归还后这必然死锁
        （实测：提交到第 2 个就预算耗尽、而消费还没开始），所以提交挪进喂料线程；
      * 消费顺序不是文件序（`load_safetensors_krea2` 按 params 树的组装序读），
        所以 `read()` 遇到还没排到的 tensor **就地插队提交**、绝不等喂料线程
        —— 等它 = 又一种死锁（它被预算卡住，而预算等着这次消费）。
    每 1GB 打一行进度（流式加载曾经"静默 10 分钟"被误判成卡死）。"""
    import concurrent.futures
    import threading
    import time

    final, remote_total = _http_final_url(url, token)
    # CDN 签名 URL 之后不再需要 Authorization（且不该再带）。
    cdn_token = token if final == url else None
    raw8 = _http_range_get(final, 0, 8, cdn_token)
    hlen = struct.unpack("<Q", raw8)[0]
    hdr = json.loads(_http_range_get(final, 8, 8 + hlen, cdn_token))
    meta = hdr.pop("__metadata__", None) or {}
    base = 8 + hlen
    # 完整性闸门：与本地路径那条（:851-862）**同一条判据**，只是"文件长度"改成
    # 从 Range 探测的 `Content-Range: bytes 0-0/<total>` 拿。Range 只保证单次
    # 请求的长度对，保证不了远端文件完整 —— HF 侧上传被截断、或 URL 指到一个
    # 半成品，逐 tensor 读到最后才炸在 reshape 上，TPU 配额已经烧掉几分钟。
    end = max((m["data_offsets"][1] for m in hdr.values()), default=0)
    if remote_total is not None and base + end != remote_total:
        raise ValueError(
            f"{url} 不是完整的 safetensors：头部声明数据区到 {base + end} 字节，"
            f"远端实际 {remote_total} 字节（差 {remote_total - base - end:+d}）。\n"
            f"  上传被截断，或 URL 指到了半成品。别继续加载 —— 换一份/重新传。")

    pool = concurrent.futures.ThreadPoolExecutor(workers)
    cond = threading.Condition()
    inflight = [0]                                # 已提交但未被消费的字节数
    futs: Dict[str, Any] = {}
    total_bytes = sum(m["data_offsets"][1] - m["data_offsets"][0]
                      for m in hdr.values())
    done_bytes = [0]
    t0 = time.time()

    def _submit(name: str, s0: int, e0: int) -> None:
        """提交一张的 range GET 并记账。**调用方必须持 cond。**"""
        futs[name] = pool.submit(_http_range_get, final, base + s0, base + e0,
                                 cdn_token)
        inflight[0] += e0 - s0
        cond.notify_all()

    order = sorted(((name, m["data_offsets"][0], m["data_offsets"][1])
                    for name, m in hdr.items()), key=lambda t: t[1])

    def _feed() -> None:
        for name, s0, e0 in order:
            sz = e0 - s0
            with cond:
                while name not in futs and inflight[0] + sz > inflight_budget:
                    cond.wait()
                if name not in futs:              # 否则已被 read() 插队提交过
                    _submit(name, s0, e0)

    threading.Thread(target=_feed, daemon=True, name="k2-st-prefetch").start()

    def make_reader(name: str, m):
        s0, e0 = m["data_offsets"]
        shape, dt = m["shape"], _NP[m["dtype"]]
        sz = e0 - s0

        def read() -> np.ndarray:
            with cond:
                fut = futs.get(name)
                if fut is None:                   # 还没排到（或已被读过一次）
                    _submit(name, s0, e0)
                    fut = futs[name]
            raw = fut.result()                    # **不持锁**等网络
            # 放掉 future 对字节的引用再归还预算。清 `_result` 动的是 CPython
            # concurrent.futures 的内部字段（有意为之：官方没给"取完就丢"的
            # API）；风险是未来版本改名/改语义，届时退化成"预算归还偏早" =
            # 恢复成修复前的行为，不影响数值正确性。摘出 futs 那半是纯公开
            # 语义的，任何版本都有效。
            with cond:
                if futs.get(name) is fut:
                    del futs[name]
                    inflight[0] -= sz
                    cond.notify_all()
            try:
                fut._result = None
            except Exception:                     # noqa: BLE001 —— 纯优化，失败无害
                pass
            del fut
            done_bytes[0] += len(raw)
            gb = done_bytes[0] / (1 << 30)
            if int(gb) > int((done_bytes[0] - len(raw)) / (1 << 30)):
                print(f"    [流式] {gb:.0f}/{total_bytes / (1 << 30):.0f}GiB "
                      f"({done_bytes[0] / 1e6 / max(time.time() - t0, 1e-9):.0f}MB/s)",
                      flush=True)
            return np.frombuffer(raw, dtype=dt).reshape(shape)
        return read

    return ({name: (m["dtype"], m["shape"], make_reader(name, m))
             for name, m in hdr.items()}, meta)


def _to_jax(arr: np.ndarray, dt_str: str, dtype) -> jnp.ndarray:
    if dt_str == "BF16":
        out = jax.lax.bitcast_convert_type(jnp.asarray(arr), jnp.bfloat16)
    else:
        out = jnp.asarray(arr)
    return out.astype(dtype)


def load_safetensors_krea2(path: str, dtype=jnp.bfloat16,
                           shard_put: Optional[Callable] = None,
                           row_norms_out: Optional[Dict[str, np.ndarray]] = None,
                           cfg: Optional["Krea2Config"] = None,
                           token: Optional[str] = None,
                           ) -> Tuple[PyTree, Krea2Config]:
    """把 Krea-2-Raw 的 safetensors 读成 JAX pytree。

    `shard_put` 非 None 时逐张读出**即刻** `jax.device_put(arr, shard_put(name, arr))`
    —— host 不持有全量副本（24GB 底模在 Kaggle 上的标准姿势）；None 时全量落
    默认设备（本地对拍/小模型用）。HTTP 流式路径下"不持全量"这句由
    `_read_safetensors_map_http` 的 `inflight_budget` 兜住（预取的原始字节在
    `read()` 消费后立即释放，驻留量 ~4GB 而不是 26GB —— 这条曾经是句空话，
    见该函数 docstring 里记的三处）。RMSNorm scale / mod.lin / bias 这类裸张量
    **随 `dtype` 存储**（不特殊保 fp32）：torch 训练侧是 `model.to(dtype)` 全转
    （model_family.py:135 附近），存储即经一轮 bf16 舍入；这里随 dtype 存 +
    用前升 fp32（rms_norm_zc）/ 降运行 dtype（调制加法、bias），两条路径都与
    torch 训练侧逐 bit 一致。fp32 对拍（K1）时 dtype=fp32 → 全 fp32，不受影响。

    `row_norms_out` 给一个 dict 时，边读边流式算好每个 Linear 权重的 fp32 逐行
    平方范数，按键 `{堆叠target: [count, out]}` 填入 —— DoRA 的 dora_scale 初值
    来源（权重随后就分片了，不该再 gather 回来算）。
    """
    entries, st_meta = _read_safetensors_map(
        path, token if token is not None else os.environ.get("HF_TOKEN"))
    if cfg is None:
        if "krea2_config" in st_meta:
            # 自描述 ckpt（本仓库 dump/转换工具会写）：非发布构型就靠它 ——
            # 形状推断假定 headdim=128，对自定义构型会静默猜错（真机踩过：
            # 猜错不报错，只会把 attention 头数搞错然后广播炸）。
            import dataclasses
            mc = (json.loads(st_meta["krea2_config"])
                  if isinstance(st_meta["krea2_config"], str)
                  else st_meta["krea2_config"])
            cfg = Krea2Config(**{f.name: mc[f.name]
                                 for f in dataclasses.fields(Krea2Config)
                                 if f.name in mc})
        else:
            # 发布构型（12B large_wide）能直接命中预设；非发布构型按 headdim=128
            # 推断（与 torch 侧 infer_config_from_state_dict 同一假设与局限）。
            cfg = _infer_config({k: v[1] for k, v in entries.items()})

    def note_row_sq(name: str, arr_f32: np.ndarray) -> None:
        """把 `xxx.weight` 映射回堆叠 target 并累加逐行平方范数。"""
        if row_norms_out is None or not name.endswith(".weight") or arr_f32.ndim != 2:
            return
        mod = name[:-len(".weight")]
        for s in _STACKS:
            if not mod.startswith(s + "."):
                continue
            rest = mod[len(s) + 1:]               # "{i}.attn.wq"
            idx, _, tail = rest.partition(".")
            if not idx.isdigit():
                continue
            target = f"{s}.{tail}"
            slot = row_norms_out.setdefault(
                target, np.zeros((lora_target_shapes(cfg)[target][0],
                                  arr_f32.shape[0]), np.float64))
            slot[int(idx)] += np.sum(arr_f32.astype(np.float64) ** 2, axis=-1)
            return
        row_norms_out.setdefault(
            mod, np.zeros((1, arr_f32.shape[0]), np.float64))[0] += \
            np.sum(arr_f32.astype(np.float64) ** 2, axis=-1)

    def get(name: str) -> jnp.ndarray:
        dt, _shape, rd = entries[name]
        raw = rd()
        if row_norms_out is not None:
            f32 = ((raw.astype(np.uint32) << 16).view(np.float32)
                   if dt == "BF16" else raw.astype(np.float32))
            note_row_sq(name, f32)
        out = _to_jax(raw, dt, dtype)
        if shard_put is not None:
            out = jax.device_put(out, shard_put(name, out))
        return out

    def lin(name: str, bias: bool = False) -> Dict[str, Any]:
        m = {"w": get(f"{name}.weight")}
        if bias:
            m["b"] = get(f"{name}.bias")
        return m

    def block(b: str, has_mod: bool) -> Dict[str, Any]:
        d = {
            "prenorm": get(f"{b}.prenorm.scale"),
            "postnorm": get(f"{b}.postnorm.scale"),
            "attn": {n: get(f"{b}.attn.{n}.weight") for n in ("wq", "wk", "wv", "gate", "wo")}
                   | {"qnorm": get(f"{b}.attn.qknorm.qnorm.scale"),
                      "knorm": get(f"{b}.attn.qknorm.knorm.scale")},
            "mlp": {n: get(f"{b}.mlp.{n}.weight") for n in ("gate", "up", "down")},
        }
        if has_mod:
            d["mod_lin"] = get(f"{b}.mod.lin")
        return d

    params = {
        "first": lin("first", bias=True),
        "tmlp0": lin("tmlp.0", bias=True), "tmlp2": lin("tmlp.2", bias=True),
        "tproj": lin("tproj.1", bias=True),
        "txtfusion": {
            "layerwise": [block(f"txtfusion.layerwise_blocks.{i}", False) for i in range(2)],
            "projector": get("txtfusion.projector.weight"),
            "refiner": [block(f"txtfusion.refiner_blocks.{i}", False) for i in range(2)],
        },
        "txtmlp_norm": get("txtmlp.0.scale"),
        "txtmlp1": lin("txtmlp.1", bias=True), "txtmlp3": lin("txtmlp.3", bias=True),
        "blocks": [block(f"blocks.{i}", True) for i in range(cfg.layers)],
        "last": {
            "norm": get("last.norm.scale"),
            "linear_w": get("last.linear.weight"),
            "linear_b": get("last.linear.bias"),
            "mod_lin": get("last.modulation.lin"),
        },
    }
    return params, cfg


def _infer_config(shapes: Dict[str, Tuple[int, ...]]) -> Krea2Config:
    """krea2_modeling.py:1103 的 JAX 侧对应物。发布构型直接命中预设。"""
    features = int(shapes["first.weight"][0])
    n_blocks = len({k.split(".")[1] for k in shapes if k.startswith("blocks.")})
    if features == 6144 and n_blocks == 28:
        return Krea2Config()
    headdim = 128
    kv_dim = int(shapes["blocks.0.attn.wk.weight"][0])
    txt_dim = int(shapes["txtmlp.1.weight"][1])
    txt_layers = int(shapes["txtfusion.projector.weight"][1])
    txt_kv = int(shapes["txtfusion.refiner_blocks.0.attn.wk.weight"][0])
    txt_hd = txt_dim // 20 if txt_dim % 20 == 0 else headdim
    return Krea2Config(
        features=features, txtdim=txt_dim,
        heads=features // headdim, kvheads=kv_dim // headdim,
        layers=n_blocks, txtlayers=txt_layers,
        txtheads=max(1, txt_dim // txt_hd), txtkvheads=max(1, txt_kv // txt_hd))


# ── LoRA 形状表（TPU 后端默认 = 官方推荐全部 264 个 Linear）────────────────────
#: 四个区域的堆叠份数：blocks=cfg.layers、lw/rf=2、single=1。scan 布局的参数树
#: 统一带 count 前导维（单例 count=1，前向时切 [0]），与 Anima 的 [L, ...] 同
#: 一套机制。
#: 曾是模块级常量 `LORA_STACKS`，把 `28` 硬编码在元组里 —— 全仓无人引用，却和
#: `lora_target_shapes`（正确地用 `cfg.layers`）构成一份影子常量：非 28 层的
#: 构型（`_infer_config` 推得出来）下两者会静默分家。改成从 cfg 派生。
def lora_stacks(cfg: Krea2Config) -> Tuple[Tuple[str, int], ...]:
    """(栈名, 堆叠份数) —— 与 `lora_target_shapes` 读的是同一份 cfg。"""
    return (("blocks", cfg.layers), ("txtfusion.layerwise_blocks", 2),
            ("txtfusion.refiner_blocks", 2))


def lora_target_shapes(cfg: Krea2Config) -> Dict[str, Tuple[int, int, int]]:
    """target → (count, in_features, out_features)。

    命名规则：堆叠目标是 `<栈名>.<模块>`（栈名带 layerwise/refiner 全路径），
    导出时展开成 `<栈名>.{i}.<模块>` —— 与 torch 模块路径**逐字一致**，于是
    export 键名 `lora_unet_<名>...` 与 PyTorch 侧产物相同（ComfyUI 直接认）。
    官方推荐默认 = 全部 264 个 Linear（trainer/model_family.py:443 同口径）。
    """
    F, T = cfg.features, cfg.txtdim
    shapes: Dict[str, Tuple[int, int, int]] = {}
    # 每栈的 (dim, heads, kvheads)；份数一律从 `lora_stacks(cfg)` 取（唯一来源）
    geom = {"blocks": (cfg.features, cfg.heads, cfg.kvheads),
            "txtfusion.layerwise_blocks": (cfg.txtdim, cfg.txtheads, cfg.txtkvheads),
            "txtfusion.refiner_blocks": (cfg.txtdim, cfg.txtheads, cfg.txtkvheads)}
    for stack, count in lora_stacks(cfg):
        dim, heads, kvheads = geom[stack]
        hd = dim // heads
        for n in ("wq", "gate", "wo"):
            shapes[f"{stack}.attn.{n}"] = (count, dim, dim)
        for n in ("wk", "wv"):
            shapes[f"{stack}.attn.{n}"] = (count, dim, kvheads * hd)
        mlp = cfg.mlpdim if dim == F else cfg.txt_mlpdim
        for n, (i_f, o_f) in (("gate", (dim, mlp)), ("up", (dim, mlp)),
                              ("down", (mlp, dim))):
            shapes[f"{stack}.mlp.{n}"] = (count, i_f, o_f)
    shapes["txtfusion.projector"] = (1, cfg.txtlayers, 1)
    shapes["txtmlp.1"] = (1, T, F)
    shapes["txtmlp.3"] = (1, F, F)
    shapes["tmlp.0"] = (1, cfg.tdim, F)
    shapes["tmlp.2"] = (1, F, F)
    shapes["tproj.1"] = (1, F, 6 * F)
    shapes["first"] = (1, cfg.in_dim, F)
    shapes["last.linear"] = (1, F, cfg.in_dim)
    return shapes


# ── 适配器（模型族层：plan / init / unstack / ctx 分区域）──────────────────────
#: 堆叠栈名（顺序敏感：长的在前，startswith 匹配先中长的）。
_STACKS = ("txtfusion.layerwise_blocks", "txtfusion.refiner_blocks", "blocks")

#: 区域名（LoraCtx 分桶用）：栈名 → 区域键。单例全进 "single"。
_REGION_OF_STACK = {"blocks": "blocks",
                    "txtfusion.layerwise_blocks": "lw",
                    "txtfusion.refiner_blocks": "rf"}


def expand_name(target: str, i: int, count: int) -> str:
    """堆叠目标展开成 torch 模块路径：`blocks.attn.wq`,2 → `blocks.2.attn.wq`；
    单例（count=1）原名返回（`tproj.1`）。"""
    if count == 1:
        return target
    for s in _STACKS:
        if target.startswith(s + "."):
            return f"{s}.{i}." + target[len(s) + 1:]
    raise ValueError(f"count>1 的 target {target!r} 不属于任何已知栈 {_STACKS}")


def _region_of(target: str) -> Tuple[str, str]:
    """target -> (区域键, 区域相对名)。区域相对名是 LoraCtx 里的键。"""
    for s in _STACKS:
        if target.startswith(s + "."):
            return _REGION_OF_STACK[s], target[len(s) + 1:]
    return "single", target


def plan_targets_k2(cfg: Krea2Config, acfg: AD.AdapterConfig,
                    targets: Sequence[str]) -> Dict[str, AD.TargetPlan]:
    """Krea2 版 plan_targets：按 reg_dims/reg_alphas 在**展开名**上解析逐位置
    rank/alpha（与 Anima 的 `blocks.{i}.{t}` 匹配语义一致，正则写法不变）。"""
    shapes = lora_target_shapes(cfg)
    out: Dict[str, AD.TargetPlan] = {}
    for t in targets:
        if t not in shapes:
            raise ValueError(f"未知 krea2 target {t!r}，可选 {sorted(shapes)}")
        count, i_f, o_f = shapes[t]
        if acfg.kind == "lokr":
            f = AD.find_factor(i_f, o_f, acfg.factor)
            in_dim, out_dim = i_f // f, o_f // f
            cap = min(in_dim, out_dim)
        else:
            f, in_dim, out_dim = 1, i_f, o_f
            cap = min(i_f, o_f)
        ranks, alphas = [], []
        for i in range(count):
            name = expand_name(t, i, count)
            r = int(AD.resolve_reg(acfg.reg_dims, name, acfg.rank))
            a = float(AD.resolve_reg(
                acfg.reg_alphas, name,
                acfg.alpha if acfg.alpha is not None else acfg.rank))
            # rank 夹到结构上限时 alpha 同比例夹，保住 scale = alpha/rank
            # （krea2 的 txtfusion.projector 是 Linear(12→1)、cap=1，不夹 alpha
            #  会让这一层的 scale 变成 32 而其它 264 层都是 1 —— 详见
            #  adapters.cap_rank_alpha 的注记）。
            r, a = AD.cap_rank_alpha(r, a, cap)
            ranks.append(r)
            alphas.append(a)
        out[t] = AD.TargetPlan(t, i_f, o_f, tuple(ranks), tuple(alphas),
                               f, in_dim, out_dim)
    return out


def init_adapters_k2(key, cfg: Krea2Config, acfg: AD.AdapterConfig,
                     targets: Optional[Sequence[str]],
                     base_row_sq: Optional[Dict[str, np.ndarray]] = None,
                     dtype=jnp.float32):
    """Krea2 版 init_adapter。targets=None -> 全部 264 个 Linear（官方推荐默认）。

    `base_row_sq` = 加载时流式算好的 {target: [count, out]}（DoRA 必需；
    权重分片后不该再 gather 求它，见 load_safetensors_krea2）。"""
    if targets is None:
        targets = list(lora_target_shapes(cfg))
    plans = plan_targets_k2(cfg, acfg, targets)
    fn = None
    if base_row_sq is not None:
        def fn(t):                            # noqa: F811
            v = base_row_sq.get(t)
            if v is None:
                raise ValueError(f"DoRA 初值缺 {t!r} 的逐行范数 —— 加载权重时"
                                 f"需要 want_row_norms=True")
            return jnp.asarray(v, jnp.float32)
    trainable, consts = AD.init_from_plans(key, acfg, plans, fn, dtype)
    return trainable, consts, plans


def unstack_k2(trainable: PyTree, plans: Dict[str, AD.TargetPlan],
               acfg: AD.AdapterConfig) -> Dict[str, Dict[str, np.ndarray]]:
    """scan 布局 -> torch 模块路径命名（`blocks.{i}.attn.wq` / `tproj.1` / ...）。"""
    return AD.unstack_named(
        trainable, plans, acfg,
        lambda t, i: expand_name(t, i, len(plans[t].ranks)))


def split_ctx_regions(params_tree: PyTree, consts_tree: PyTree, drop_tree: PyTree,
                      acfg: AD.AdapterConfig) -> Dict[str, "A.LoraCtx"]:
    """把整棵适配器树按区域拆成四个 LoraCtx（blocks/lw/rf/single）。

    堆叠区域的树带 count 前导维（scan 切片用）；**单例区域就地 squeeze 掉
    count=1 的前导维**（AD.apply 吃的是单块形状）。drop/consts 同步拆。
    """
    regions: Dict[str, Dict[str, Dict[str, Any]]] = {
        r: {"p": {}, "c": {}, "d": {}} for r in CTX_REGIONS}
    for t in params_tree:
        region, rel = _region_of(t)
        squeeze = region == "single"
        def sq(x):
            return x[0] if (squeeze and x is not None and hasattr(x, "shape")
                            and x.shape[0] == 1) else x
        regions[region]["p"][rel] = jax.tree.map(sq, params_tree[t])
        if consts_tree is not None and t in consts_tree:
            regions[region]["c"][rel] = jax.tree.map(sq, consts_tree[t])
        if drop_tree is not None and t in drop_tree:
            regions[region]["d"][rel] = jax.tree.map(sq, drop_tree[t])
    out = {}
    for r, d in regions.items():
        if not d["p"]:
            out[r] = None
            continue
        out[r] = A.LoraCtx(acfg, d["p"], d["c"] or None, d["d"] or None)
    return out
