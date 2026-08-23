"""Anima DiT 的纯 JAX 前向（TPU 路线）——与 PyTorch 实现逐算子对齐。

**为什么是纯 JAX 而不是 torch_xla**：TPU 侧所有已验证的物证（splash 块对角跳块
0.252、8 卡集合通信带宽、显存账）都在 JAX 侧取得；torch_xla 路线的端到端训练步
探测至今未通过（kaggle_job/output/anima-torchxla-navit-probe：L4.1 FAIL）。
底模冻结、只训 LoRA，需要移植的前向面很小，所以直接写 JAX。

**Anima 与 Krea2 的关键差异（决定 TPU 形态）**：
  * 参数量 2.091B、bf16 仅 4.18GB -> **单颗 v5e chip(15.7GiB) 放得下**，
    因此**不需要 FSDP/TP**，8 卡纯数据并行即可，每步只 all-reduce LoRA 梯度。
    （Krea2 12.16B/22.6GB 必须分片，那套 FSDP 方案在这里是多余的通信。）
  * 文本**不进主序列**，走 cross-attn（q=图像段 / kv=caption 段，块对角是**矩形**的）。

对齐依据（逐条指向 PyTorch 源，全部在 `models/anima_modeling_core.py` 里）。

**锚点是函数名/类名，行号只是辅助**：本文件上一版这一批引用写的是纯行号，PyTorch
侧插了一段代码后**十余处集体漂了 +83 行**（例如 `:1174` 实际已经是 `:1257`），逐条
修行号只会再漂一次。所以下表以符号名为准，行号带 `~` 表示"写这行注释时的位置"；
对不上就用 `grep -n "def forward_tokens" models/anima_modeling_core.py` 重新定位。

  RMSNorm._norm / RMSNorm.forward       fp32 归一、type_as(x) 回原 dtype、再乘 weight  ~:391
  GPT2FeedForward.__init__              nn.GELU() 默认 = erf 精确式，**不是** tanh 近似 ~:410
  Attention.__init__                    q_norm/k_norm 是**逐 head_dim** RMSNorm eps=1e-6 ~:576
  _apply_rotary_pos_emb_base            RoPE（rotate_half 式，cos/sin 转成 t.dtype 再乘）~:365
  Timesteps.forward                     cat[cos, sin]，注意 cos 在前                  ~:807
  TimestepEmbedding.forward             use_adaln_lora 时 emb=**原始正弦**，
                                        MLP 输出走 adaln_lora 这条支路                ~:847
  Block.__init__                        LayerNorm elementwise_affine=False eps=1e-6   ~:1084
  Block.forward_tokens                  三组 AdaLN + mod_index gather                 ~:1257
  FinalLayer.forward_tokens             2 chunk，取 adaln_lora 前 2D                  ~:1009
  MiniTrainDIT._packed_rope_from_grid   cat[t,h,w]*2，t 段恒 0                        ~:1746
  MiniTrainDIT.forward_packed_navit     整体装配                                      ~:1857

本文件只做**前向**，不含优化器/数据/采样——那些留在 PyTorch 侧离线完成。
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from typing import Any, Dict, NamedTuple, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

try:
    from . import adapters as AD
except ImportError:                     # jax_tpu/ 直接在 sys.path 上（tests/ 走这条）
    import adapters as AD

PyTree = Any


# ── 构型（从 anima-base-v1.0.safetensors 实测形状反推，非猜测）──────────────────
@dataclass(frozen=True)
class AnimaConfig:
    model_channels: int = 2048      # x_embedder.proj.1.weight [2048, 68]
    num_blocks: int = 28            # net.blocks.0..27
    num_heads: int = 16             # head_dim = 2048/16 = 128（q_norm.weight [128]）
    mlp_ratio: float = 4.0          # mlp.layer1 [8192, 2048]
    crossattn_dim: int = 1024       # cross_attn.k_proj [2048, 1024]
    adaln_lora_dim: int = 256       # adaln_modulation.1 [256, 2048]
    in_dim: int = 68                # (16 latent ch + 1 padding mask) * patch 2^2
    out_dim: int = 64               # final_layer.linear [64, 2048] = 16 * 2^2
    patch: int = 2
    eps_ln: float = 1e-6            # Block 的 LayerNorm
    eps_qk: float = 1e-6            # q_norm / k_norm
    # t_embedding_norm 的 eps。**1e-6，不是 RMSNorm 的类默认值 1e-5**：
    # `MiniTrainDIT.__init__` 里写的是 `RMSNorm(model_channels, eps=1e-6)`
    # （anima_modeling_core.py 的 MiniTrainDIT.__init__，~:1498），实例化时显式传参
    # 覆盖了 `RMSNorm.__init__(dim, eps=1e-5)` 的默认值。抄默认值就是抄错。
    #
    # 写错会怎样：t_embedding_norm 吃的是 timestep_sincos 的输出，而
    # mean(cat[cos,sin]²) 对任意 t 恒 ≈ 0.5（cos²+sin² 逐通道配对），所以 eps 差值
    # 不随 t 变化 —— emb 被一个**与 t 无关的常数因子**整体缩放，本地实测相对量
    # -9.13e-6（理论 -Δeps/(2·0.5) = -9e-6）。这个量级两侧闸门都抓不到：
    # fp32 对拍闸门 tol=1e-4 差 11 倍，bf16 下小于 1 个 ULP（2⁻⁸ = 3.9e-3）。
    # 也就是说错了不会被任何现有测试发现，只会让 TPU 与 GPU 的 emb 系统性差一点。
    eps_rms: float = 1e-6
    # RoPE 的 NTK 外推系数。trainer/models.py 的 load_anima_model 对 in_channels==16
    # 用 4.0（`rope_h_extrapolation_ratio=4.0 if in_channels == 16 else 3.0`，~:224），
    # 模型内换算成 ntk_factor = ratio**(dim/(dim-2))，再乘进 theta。
    # 漏掉它不会报错，只会静默改变位置编码频率 —— 必须跟着 checkpoint 走。
    rope_h_ratio: float = 4.0
    rope_w_ratio: float = 4.0

    @property
    def head_dim(self) -> int:
        return self.model_channels // self.num_heads


# ── 基础算子 ──────────────────────────────────────────────────────────────────
def layer_norm(x: jnp.ndarray, eps: float) -> jnp.ndarray:
    """nn.LayerNorm(elementwise_affine=False)。PyTorch 内部按 fp32 累加，这里显式对齐。"""
    xf = x.astype(jnp.float32)
    mu = xf.mean(-1, keepdims=True)
    var = jnp.mean((xf - mu) ** 2, -1, keepdims=True)
    return ((xf - mu) * jax.lax.rsqrt(var + eps)).astype(x.dtype)


def rms_norm(x: jnp.ndarray, w: jnp.ndarray, eps: float) -> jnp.ndarray:
    """`RMSNorm.forward`（models/anima_modeling_core.py，~:394；行号可能漂移，
    以函数名为准）—— fp32 归一、type_as(x) 回原 dtype、再乘 weight。

    注意乘 weight 的顺序：PyTorch 是 `self._norm(x.float()).type_as(x) * self.weight`，
    即**先降回 x.dtype 再乘**。bf16 下这与"fp32 乘完再降"有舍入差异，故照抄顺序。
    """
    xf = x.astype(jnp.float32)
    normed = (xf * jax.lax.rsqrt(jnp.mean(xf ** 2, -1, keepdims=True) + eps)).astype(x.dtype)
    return normed * w.astype(x.dtype)


def gelu_exact(x: jnp.ndarray) -> jnp.ndarray:
    """nn.GELU() 默认 approximate='none' -> erf 精确式（**不是** tanh 近似）。"""
    return jax.nn.gelu(x, approximate=False)


def silu(x: jnp.ndarray) -> jnp.ndarray:
    return x * jax.nn.sigmoid(x)


def rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rope(t: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray) -> jnp.ndarray:
    """t: [..., S, H, D]；cos/sin: [..., S, 1, D_rot]（前导维与 t 对齐或可广播）。

    PyTorch(_apply_rotary_pos_emb_base) 先把 cos/sin `.to(t.dtype)` 再相乘 —— bf16 下
    这一步的舍入会进结果，所以这里也先转 dtype 再算，不留在 fp32。
    """
    rot = cos.shape[-1]
    t_rot, t_pass = t[..., :rot], t[..., rot:]
    c, s = cos.astype(t.dtype), sin.astype(t.dtype)
    out = t_rot * c + rotate_half(t_rot) * s
    return jnp.concatenate([out, t_pass], axis=-1) if t_pass.shape[-1] else out


def timestep_sincos(t: jnp.ndarray, num_channels: int) -> jnp.ndarray:
    """`Timesteps.forward`（models/anima_modeling_core.py，~:807；行号可能漂移，
    以函数名为准）。**cos 在前、sin 在后**（与 diffusers 相反）。

    exponent = -log(10000) * arange(half) / half —— 分母是 half_dim 本身（不减 1）。

    `t` 必须是 1-D `[G]`（每图一个 timestep）。这条**不是可选的**：下面
    `t[:, None]` 在 2-D 输入上不会报错，只会广播成 `[G, T, half]` 再 concat 成
    `[G, T, D]`，后续 dense/rms_norm 全都能算，只有 emb 的 rank 悄悄多了一维，
    最后靠 mod_bcast 的广播把错误吃掉 —— 结果是错的但没有任何异常。
    PyTorch 侧同样有硬断言（`assert timesteps_B_T.ndim == 2`，那边是 [B, T] 布局，
    这里的打包/分桶布局都是逐图一个标量 t，故要求 1-D）。
    """
    if t.ndim != 1:
        raise ValueError(f"timestep_sincos 要求 1-D 的 [G] timesteps，得到 shape "
                         f"{t.shape}（rank {t.ndim}）；2-D 输入会被广播成 [G, T, D] "
                         f"而不报错，结果静默错误")
    half = num_channels // 2
    exponent = -math.log(10000.0) * jnp.arange(half, dtype=jnp.float32) / float(half)
    emb = t.astype(jnp.float32)[:, None] * jnp.exp(exponent)[None, :]
    return jnp.concatenate([jnp.cos(emb), jnp.sin(emb)], axis=-1)


def packed_rope_cos_sin(rows: jnp.ndarray, cols: jnp.ndarray, head_dim: int,
                        h_ratio: float = 4.0, w_ratio: float = 4.0
                        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """`MiniTrainDIT._packed_rope_from_grid`（models/anima_modeling_core.py，~:1746）
    + `VideoRopePosition3DEmb.__init__` 的 dim_h/dim_t 切分（~:710）。
    行号可能漂移，以函数名为准。

    head_dim=128 -> dim_h = dim_w = 128//6*2 = 42，dim_t = 128 - 84 = 44。
    emb = cat([t_half(22 个 0), h_half(21), w_half(21)] * 2) -> 128 维。
    时间轴恒 0（图像模型 T=1），但**必须保留这 44 个 0 通道**——它们占着 RoPE 的前
    44 维，去掉会让 h/w 的通道偏移错位。

    rows/cols 的前导维是**任意**的：打包布局给 [ΣN]、批布局给 [B, L]，
    返回 [..., 1, head_dim]，正好与 q/k 的 [..., S, H, D] 在 head 维上广播。
    """
    dim_h = head_dim // 6 * 2
    dim_t = head_dim - 2 * dim_h
    spatial_range = jnp.arange(0, dim_h, 2, dtype=jnp.float32)[: dim_h // 2] / dim_h
    # `VideoRopePosition3DEmb.__init__`（~:727）—— ntk_factor = ratio**(dim/(dim-2))，
    # theta 再乘它（行号可能漂移，以函数名为准）
    h_theta = 10000.0 * (h_ratio ** (dim_h / (dim_h - 2)))
    w_theta = 10000.0 * (w_ratio ** (dim_h / (dim_h - 2)))
    h_freqs = 1.0 / (h_theta ** spatial_range)
    w_freqs = 1.0 / (w_theta ** spatial_range)
    # t 轴位置恒 0（图像模型 T=1）-> half_emb_t 恒 0，与频率无关
    half_t = jnp.zeros(rows.shape + (dim_t // 2,), jnp.float32)
    half_h = rows.astype(jnp.float32)[..., None] * h_freqs
    half_w = cols.astype(jnp.float32)[..., None] * w_freqs
    emb = jnp.concatenate([half_t, half_h, half_w] * 2, axis=-1)   # [..., head_dim]
    return jnp.cos(emb)[..., None, :], jnp.sin(emb)[..., None, :]


# ── LoRA / LoKr / DoRA ───────────────────────────────────────────────────────
#
# 适配器的数学在 adapters.py（LoKr 的 kron-bypass、DoRA 的逐行幅度归一、逐块
# rank 掩码、rank/module dropout）。这里只负责"在哪些线性层上挂适配器"。
_DEFAULT_ACFG = AD.AdapterConfig(kind="lora")


def dense(x: jnp.ndarray, w: jnp.ndarray, ad=None) -> jnp.ndarray:
    """y = x @ Wᵀ，可选挂一个适配器（低秩旁路，不物化 ΔW）。

    权重按 PyTorch nn.Linear 存放（[out, in]），故这里转置。走**旁路**而非合并——
    与 ComfyUI 的 Bypass 加载路径同构（见 memory krea2-comfyui-lora-deploy）。

    `ad` 是 `(cfg, params, consts, drop)` 四元组；`{"a":..., "b":...}` 的旧格式
    仍然接受（按标准 LoRA、scale=1 处理），对拍脚本走这条。
    """
    if ad is None:
        return x @ w.T.astype(x.dtype)
    if isinstance(ad, dict):                       # 旧格式 {a, b}
        return AD.apply(x, w, _DEFAULT_ACFG, ad, None, None)
    return AD.apply(x, w, *ad)


class LoraCtx(NamedTuple):
    """一块之内的适配器上下文。三棵树的键都是 target 名（如 `self_attn.q_proj`）。

    分成三棵而不是一棵，是因为**只有 `params` 该收梯度**：`consts`（rmask/scale/
    base_row_sq）与 `drop`（每步的 dropout 掩码）是常量，混进同一棵树会让
    `jax.grad` 为它们也算一份梯度，优化器还会试着更新它们。
    """
    cfg: Any = None
    params: Optional[Dict[str, Any]] = None
    consts: Optional[Dict[str, Any]] = None
    drop: Optional[Dict[str, Any]] = None


def _lora(ctx, key: str):
    """取某个 target 的适配器四元组；没挂适配器返回 None。"""
    if ctx is None:
        return None
    if not isinstance(ctx, LoraCtx):               # 旧格式：扁平 dict of {a,b}
        return ctx.get(key)
    if ctx.params is None or key not in ctx.params:
        return None
    return (ctx.cfg or _DEFAULT_ACFG, ctx.params[key],
            None if ctx.consts is None else ctx.consts.get(key),
            None if ctx.drop is None else ctx.drop.get(key))


# ── 注意力后端 ────────────────────────────────────────────────────────────────
def attention_dense(q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray,
                    bias: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """q: [S, H, D]，k/v: [T, H, D]，bias: [S, T] 加性（可见 0 / 屏蔽 -1e4）。
    返回 [S, H, D]。

    参考实现（本地对拍 + 小规模用）。softmax 在 fp32 上做，与 SDPA 一致。
    O(S·T) 显存，只适合 S·T 不大的场合；大 pack 走 splash（TPU）。

    **`preferred_element_type=jnp.float32` 不是可选的**：`einsum(bf16, bf16)` 的输出
    dtype 就是 bf16，把 `.astype(fp32)` 放在 einsum **之后**只是把已经截断过的数搬到
    fp32 —— softmax 本身在 fp32 上算，但喂给它的 logits 已经掉了精度。本地实测
    （S=T=8, H=2, D=128, 标准正态输入）：对 fp32 参考的相对误差
    旧写法 2.06e-03 -> 新写法 1.33e-07，**分辨力差 1.5e4 倍**。
    这个函数是块对角语义的**参考实现**（tests/check_splash_blockdiag.py 拿它当判据，
    TOL=2e-2 的 bf16 口径），参考实现自己带 2e-3 的噪声会直接吃掉判据分辨力。

    加 `preferred_element_type` 只是把 TPU MXU 本就 fp32 累加的行为写实，不是"多要
    精度"：JAX Pallas matmul 官方文档明确写 "The native MXU bf16 matmul routine ...
    accumulates it in f32"。
    """
    d = q.shape[-1]
    logits = jnp.einsum("shd,thd->hst", q, k,
                        preferred_element_type=jnp.float32) / math.sqrt(d)
    if bias is not None:
        logits = logits + bias[None, :, :]
    w = jax.nn.softmax(logits, axis=-1).astype(v.dtype)
    return jnp.einsum("hst,thd->shd", w, v)


#: 块对角 bias 的屏蔽值。**有限负值，不是 -inf** —— 口径同 PyTorch 侧
#: `MiniTrainDIT._build_packed_masks` 的 `attn_mask.masked_fill(~key_valid, -1.0e4)`
#: （models/anima_modeling_core.py，~:1807；行号可能漂移，以函数名为准）。
_MASK_NEG = -1.0e4


def block_diag_bias(q_lens: Sequence[int], kv_lens: Optional[Sequence[int]] = None,
                    dtype=jnp.float32) -> jnp.ndarray:
    """块对角加性 mask：同一图的 q 只看得见同一图的 kv。kv_lens=None 时为自注意力。

    返回 [ΣQ, ΣKV]，可见位 0.0、屏蔽位 `_MASK_NEG`(-1e4)。

    **为什么屏蔽值是 -1e4 而不是 -inf**：整行全屏蔽时 -inf 会让 softmax 分母为 0 ->
    输出 NaN -> 反向把 NaN 灌进**全部**适配器梯度（本地实测：段数不匹配的例子里
    out_has_nan=True / grad_has_nan=True；换成 -1e4 后两者都 False）。有限值下那一行
    退化成均匀分布 —— 数值是"没意义"，但不会污染整棵梯度树，错能被 loss 看见而不是
    把训练变成 NaN 空转。段数匹配（正常情形）下两者**逐 bit 相同**（本地对拍
    max|diff|=0.0），所以这不改任何正常路径的数值。
    下游 `attention.make_dense_attn`（attention.py:666）用 `== 0.0` 判"可见"，
    只依赖可见位仍是 0.0，与屏蔽值取什么无关。

    **fp32 而不是 fp64**：`np.where(bool, 0.0, -1.0e4)` 里两个分支是 Python float，
    numpy 按 float64 出结果，尾巴上的 `dtype` 只作用于之后的 `jnp.asarray` —— 也就是
    host 上先物化一份 8 字节/元素的中间数组。budget 32768 时那是 8.00 GiB host RAM
    （fp32 4.00 / bool 1.00）。

    **两个分支传 `np.float32` 标量，而不是在 `np.where(...)` 外面套 `.astype`**：
    `.astype` 是先算出 fp64 再拷一份 fp32，两份**同时活着**，实测反而更差
    （n=4096 时 tracemalloc 峰值：fp64 直出 128.00 MiB / 套 astype 192.00 MiB /
    传 fp32 标量 64.00 MiB）。三者结果逐 bit 相同，所以这里取峰值最低的那个写法。
    """
    kv_lens = q_lens if kv_lens is None else kv_lens
    if len(q_lens) != len(kv_lens):
        # 段数不匹配 = 调用方把 q/kv 的段几何接错了。以前这里不拦：多出来的 q 段
        # 找不到对应 kv 段 -> 整行屏蔽 -> （-inf 时）NaN 顺着反向灌满全部适配器梯度，
        # 而 loss 只显示 nan，查不出是段几何错的。fail-fast 在这里就说清楚。
        raise ValueError(
            f"q/kv 段数必须一致：len(q_lens)={len(q_lens)} vs "
            f"len(kv_lens)={len(kv_lens)}。块对角要求第 i 个 q 段对第 i 个 kv 段"
            f"（自注意力传 kv_lens=None，cross-attn 传每图的文本段长）；"
            f"段数不等会让多出来的段整行被屏蔽，softmax 退化成均匀分布，"
            f"训练照跑但结果全错。口径同 PyTorch 侧 _SegLens.__init__ 的同名校验。")
    qi = np.repeat(np.arange(len(q_lens)), np.asarray(q_lens))
    ki = np.repeat(np.arange(len(kv_lens)), np.asarray(kv_lens))
    allow = qi[:, None] == ki[None, :]
    return jnp.asarray(np.where(allow, np.float32(0.0), np.float32(_MASK_NEG)),
                       dtype=dtype)


# ── 单块前向 ──────────────────────────────────────────────────────────────────
def attn_qkv(x: jnp.ndarray, ctx: Optional[jnp.ndarray], p: Dict[str, Any],
             cfg: AnimaConfig, loras, prefix: str,
             cos=None, sin=None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """q/k/v 投影 + 逐 head RMSNorm(+RoPE，仅自注意力)。

    reshape 保留**全部前导维**（不是 `reshape(-1, H, D)`）：打包布局是 [S, H, D]，
    批布局是 [B, L, H, D]。写死 -1 会把批维压进序列维 —— 不报错，但每张图会看见
    其他图的 token。
    """
    H, D = cfg.num_heads, cfg.head_dim
    src = x if ctx is None else ctx
    hd = lambda z: z.reshape(*z.shape[:-1], H, D)
    q = hd(dense(x, p["q_proj"], _lora(loras, f"{prefix}.q_proj")))
    k = hd(dense(src, p["k_proj"], _lora(loras, f"{prefix}.k_proj")))
    v = hd(dense(src, p["v_proj"], _lora(loras, f"{prefix}.v_proj")))
    q = rms_norm(q, p["q_norm"], cfg.eps_qk)
    k = rms_norm(k, p["k_norm"], cfg.eps_qk)
    if ctx is None and cos is not None:          # RoPE 只作用于自注意力
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    return q, k, v


def block_forward(x: jnp.ndarray, p: Dict[str, Any], cfg: AnimaConfig,
                  emb: jnp.ndarray, adaln_lora: jnp.ndarray, mod_bcast,
                  ctx: jnp.ndarray, cos, sin,
                  self_attn_fn, cross_attn_fn,
                  loras=None, layer: Optional[int] = 0) -> jnp.ndarray:
    """`Block.forward_tokens`（models/anima_modeling_core.py，~:1257；行号可能漂移，
    以函数名为准）的逐算子复刻。

    emb: [G, D] 逐图 timestep 向量；adaln_lora: [G, 3D]。

    `mod_bcast: [G, 3D] -> 可与 x 广播的张量` 是**两种布局的唯一差别**：

      * 打包布局 `lambda h: jnp.take(h, mod_index, 0)` -> [ΣN, 3D]，逐 token **物化**；
      * 批布局   `lambda h: h[:, None, :]`             -> [G, 1, 3D]，广播、**不物化**。

    这一处就是真机 1.01 vs 0.02 MB/token 的分水岭（anima-mem-probe 的 V4 单变量
    归因：关掉 AdaLN 调制后 20.60G -> 3.87G）。gather 出来的 [ΣN, D] 是实打实的
    缓冲区，而广播只是消费者算子上的一个退化 stride，XLA 会融合掉。
    """
    D = cfg.model_channels
    # layer=None -> loras 的键是**块内相对**的（如 "self_attn.q_proj"），scan 路径用；
    # layer=i    -> 键带 "blocks.i." 前缀，展开路径用。
    lp = "" if layer is None else f"blocks.{layer}."
    flat_d = lambda z: z.reshape(*z.shape[:-2], D)   # [..., H, Dh] -> [..., D]

    def modulation(name: str):
        h = silu(emb)
        # **这两个 `_lora(...)` 当前恒为 None，是刻意保留的钩子，不要删**：
        # TPU 侧 config 层已经把 adaln 挡在门外 —— `adapters.target_shapes`
        # （adapters.py:132-138）与 `config._expand_targets`（config.py:489-492）的
        # 全名表里都**没有** adaln 条目，所以 `plan_targets` 见到 "adaln_*" 会直接
        # fail-fast（adapters.py:174 `未知 target`），键根本进不了 LoraCtx.params。
        # 保留的理由：PyTorch 侧按模块名子串匹配注入，adaln_modulation 的两个 Linear
        # 是能被命中的；哪天要把这条打开，必须**同时**在
        # `adapters.target_shapes`（给 (in, out) 形状）与 `config._expand_targets`
        # （给全名表）加条目，只加一处会在另一处 fail-fast。
        h = dense(h, p[f"adaln_{name}_1"], _lora(loras, f"{lp}adaln_{name}.1"))
        h = dense(h, p[f"adaln_{name}_2"], _lora(loras, f"{lp}adaln_{name}.2"))
        h = h + adaln_lora                                             # [G, 3D]
        # **先广播再 split**，不是 split 完广播三次：搬运字节数相同，但少两次算子。
        # 调制 MLP 只在 G 行上算，与 PyTorch 的 mod_index 路径同布局
        # （memory navit-adaln-tokenwise-overhead：per-image 布局比逐 token 布局快 13%）。
        return jnp.split(mod_bcast(h), 3, axis=-1)

    sh, sc, gt = modulation("self")
    h = layer_norm(x, cfg.eps_ln) * (1 + sc) + sh
    q, k, v = attn_qkv(h, None, p["self_attn"], cfg, loras, f"{lp}self_attn", cos, sin)
    a = flat_d(self_attn_fn(q, k, v))
    x = x + gt * dense(a, p["self_attn"]["output_proj"],
                       _lora(loras, f"{lp}self_attn.output_proj"))

    sh, sc, gt = modulation("cross")
    h = layer_norm(x, cfg.eps_ln) * (1 + sc) + sh
    q, k, v = attn_qkv(h, ctx, p["cross_attn"], cfg, loras, f"{lp}cross_attn")
    a = flat_d(cross_attn_fn(q, k, v))
    x = x + gt * dense(a, p["cross_attn"]["output_proj"],
                       _lora(loras, f"{lp}cross_attn.output_proj"))

    sh, sc, gt = modulation("mlp")
    h = layer_norm(x, cfg.eps_ln) * (1 + sc) + sh
    h = gelu_exact(dense(h, p["mlp"]["layer1"], _lora(loras, f"{lp}mlp.layer1")))
    x = x + gt * dense(h, p["mlp"]["layer2"], _lora(loras, f"{lp}mlp.layer2"))
    return x


# ── 块堆叠（lax.scan 路径）────────────────────────────────────────────────────
#
# **为什么必须有这条路径**（真机实测，不是理论优化）：
#
# 用 Python for 循环展开 28 个块时，NaViT 打包路径的显存实测
# ≈ 4.0GB + **1.01 MB/token**（budget 16384 需 20.60G、32768 需 37.18G，都 OOM）。
# 而分桶路径只要 0.02 MB/token。归因 job（anima-mem-probe）的单变量结果：
#
#   V1 mod_index 恒 0（gather->广播）  20.60G  <- 没变，**不是 gather 的锅**
#   V3 反向块 128                      20.60G  <- 没变
#   V2 关 cross-attn                   16.40G  <- 降 4.2G
#   V4 **关 AdaLN 调制**               3.87G   <- 塌下来了，主因在这
#
# 对上账：shift/scale/gate 各 [ΣN, D] bf16 = 4KB/token，x3 = 12KB/token/次调制，
# x3 次/块 x28 块 = 1.0 MB/token，与实测 1.01 严丝合缝。
#
# 根因不是"激活没被 remat 掉"，而是**调度**：AdaLN 调制只依赖 emb 与块权重、
# **完全不依赖 x**，于是 28 个块的调制在展开的图里是 28 组彼此独立的计算，
# XLA 可以把它们全部提前算好，同时活着。remat 管不了这个（它管的是保存与重算，
# 不是调度顺序）。分桶路径没暴露是因为那边 mod_index 恒 0、张量被优化掉了。
#
# lax.scan 把 28 个块变成一个带循环的子图：跨块提升在语义上就不可能了，
# 顺带把编译量从 28 个块降到 1 个（展开时首调实测 49-57s）。
def stack_blocks(params: PyTree) -> PyTree:
    """把 `params["blocks"]`（list of dict）堆成 pytree of [L, ...]，供 scan 用。

    在**加载时调用一次**即可；不要放进 jit 的每步里（那是每步一次全权重拷贝）。
    """
    blocks = params["blocks"]
    if not isinstance(blocks, (list, tuple)):
        return params                                   # 已经是堆好的
    stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *blocks)
    return {**params, "blocks": stacked}


def stack_loras(loras: Optional[Dict[str, Any]], num_blocks: int
                ) -> Optional[Dict[str, Any]]:
    """`{"blocks.{i}.{target}": {a,b}}` -> `{target: {a: [L,...], b: [L,...]}}`。

    要求每个 target 在所有块上都存在且形状一致（本仓库的 LoRA 注入满足这点：
    每块结构相同）。缺块会在这里 fail-fast，而不是在 scan 里变成形状怪错。

    **契约：只支持标准 LoRA 的 `{a, b}` 两个因子**（下面的 `for s in ("a", "b")`
    是写死的）。LoKr（`{w1, w2a, w2b}`）/ DoRA（多一个 `dora`）进来会 KeyError。
    这不是遗漏而是分工：训练路径的适配器由 `adapters.init` **直接**产出 scan 布局
    （每个 target 一份 `[L, ...]`，见 adapters.init_from_plans），根本不需要"先扁平
    再堆叠"这一步；导出走 `adapters.unstack`（认全部三种因子）。本函数与
    `unstack_loras` 只服务 `init_lora` 的扁平产物 —— 也就是对拍脚本
    （tests/check_scan_equiv.py 的 C4 往返、tests/check_export.py）。
    """
    if loras is None:
        return None
    targets = sorted({k.split(".", 2)[2] for k in loras})
    out = {}
    for t in targets:
        keys = [f"blocks.{i}.{t}" for i in range(num_blocks)]
        missing = [k for k in keys if k not in loras]
        if missing:
            raise ValueError(f"target {t!r} 缺 {len(missing)} 个块（如 {missing[0]}）；"
                             f"scan 路径要求每块都有同一组 target")
        out[t] = {s: jnp.stack([loras[k][s] for k in keys]) for s in ("a", "b")}
    return out


def _slice_ctx(ctx: "LoraCtx", i: int) -> "LoraCtx":
    """取 LoraCtx 的第 i 块（展开路径用）。cfg 原样带过去，不参与切片。"""
    take = lambda t: None if t is None else jax.tree.map(lambda z: z[i], t)
    return LoraCtx(ctx.cfg, take(ctx.params), take(ctx.consts), take(ctx.drop))


def unstack_loras(stacked: Dict[str, Any], num_blocks: int) -> Dict[str, Any]:
    """`stack_loras` 的逆运算，对拍/取证时用。

    **同样只认标准 LoRA 的 `{a, b}`**（见 `stack_loras` 的契约说明）。
    LoKr/DoRA 的导出走 `adapters.unstack` -> `export.adapter_state_dict`
    （export.py:112 按有没有 `lokr_w1` 自动分流），不要往这里传。
    """
    return {f"blocks.{i}.{t}": {s: v[s][i] for s in ("a", "b")}
            for t, v in stacked.items() for i in range(num_blocks)}


# ── remat 强度 ────────────────────────────────────────────────────────────────
#: 可选档位。**冻结底模只训 LoRA 时反向没有 wgrad**，所以一次训练步 ≈
#: 前向(1) + 重算前向(1) + 激活梯度(1)；关掉重算最多省掉其中 1/3。
#: 显存与速度是直接对冲的，最优档只能实测（真机 D1：峰值 4.9/15.7 GiB，余量很大）。
REMAT_CHOICES = ("full", "dots", "every2", "none")


def resolve_remat(remat, unrolled: bool = True):
    """把 remat 设定翻成 `(mode, wrap(fn, layer_idx) -> fn)`。

    第一个返回值是**规范化后的档位字符串**（不是 checkpoint policy）：
    `True -> "full"`、`False -> "none"`，其余原样。调用方必须用它而不是原始参数
    去做分档判断 —— 见 `_forward_core` 里 `mode == "every2"` 那处。

      full   每块整个重算——激活最省、算力最贵（原实现，且是当前唯一真机验证过的档）
      dots   保留所有矩阵乘的输出，只重算 norm/激活函数等便宜算子
             （jax.checkpoint_policies.dots_saveable）——省掉绝大部分重算，
             但要存 q/k/v/attn/mlp 隐藏层，显存涨得最多。
             **仅展开路径可用**（`unrolled=True`）：scan 下未被 remat 掉的激活会按
             28 次迭代堆叠（≈1MB/token），budget 16384 必 OOM，故那边 fail-fast。
      every2 隔块 remat：偶数块存、奇数块重算——粗粒度的折中
      none   完全不 remat：最快，但按结构估算需要约 2.2MB/token 的激活
             （**估算值，未实测**），11.8GiB 余量只够约 5k token/chip

    `unrolled`：调用方是不是展开路径。默认 True = 老行为（放行全部档位），
    scan 路径必须显式传 False 才能拦住 `dots`。

    兼容旧签名：True -> "full"，False -> "none"。
    """
    if remat is True:
        remat = "full"
    elif remat is False:
        remat = "none"
    if remat not in REMAT_CHOICES:
        raise ValueError(f"remat 只能是 {REMAT_CHOICES} 之一（或 True/False），得到 {remat!r}")

    if remat == "none":
        return remat, (lambda fn, i: fn)
    if remat == "full":
        return remat, (lambda fn, i: jax.checkpoint(fn))
    if remat == "every2":
        return remat, (lambda fn, i: fn if i % 2 == 0 else jax.checkpoint(fn))
    if not unrolled:
        # 同 krea2_jax._resolve_remat 对 dots 的处理（那边是 FSDP 全量权重的账，
        # 这里是 scan 迭代堆叠的账）：语法上曾被放行，真机上是一条静默的死路。
        raise ValueError(
            "scan 路径不支持 remat='dots'：dots_saveable 会把 q/k/v/attn/mlp 隐藏层"
            "留给反向，而 scan 的语义是「未被 remat 掉的激活按 28 次迭代堆叠」"
            "（≈1MB/token），budget 16384 直接 OOM。"
            "dots 只在展开路径可用；scan 下请用 'full' 或 'every2'。")
    pol = jax.checkpoint_policies.dots_saveable
    return remat, (lambda fn, i: jax.checkpoint(fn, policy=pol))


# ── 整模前向（NaViT packed）───────────────────────────────────────────────────
def chunk_attn(attn):
    """把只认 `[S, H, D]` 的注意力后端包成认 `[C, Q, H, D]` 的（见 `forward_packed`
    的 `chunk` 参数）。

    只压平**图像侧**：cross-attn 的 k/v 来自 `ctx`，本来就是 `[ΣL, H, D]` 三维，
    按秩判断不动它。压平是纯 reshape（chunk 维与序列维在内存里连续），
    所以块对角 mask、段几何、splash 内核**全部不变** —— 这正是这条路径的要点：
    它改变的只有 AdaLN 调制的布局，不改变注意力语义。
    """
    def f(q, k, v):
        lead = q.shape[:-2]
        fl = lambda z: z.reshape(-1, *z.shape[-2:]) if z.ndim > 3 else z
        o = attn(fl(q), fl(k), fl(v))
        return o.reshape(*lead, *o.shape[-2:])
    return f


def forward_packed(params: PyTree, cfg: AnimaConfig,
                   tokens: jnp.ndarray, timesteps: jnp.ndarray, ctx: jnp.ndarray,
                   rows: jnp.ndarray, cols: jnp.ndarray, mod_index: jnp.ndarray,
                   self_attn_fn, cross_attn_fn,
                   loras=None, remat="full",
                   chunk: Optional[int] = None,
                   barrier: bool = False) -> jnp.ndarray:
    """**打包（NaViT）布局**：整个 pack 是一条长序列，靠块对角 mask 隔开各图。

      tokens     [ΣN, in_dim]   已 patchify 的图像 token（按图序拼接，含 padding 段）
      timesteps  [G]            每图一个 t
      ctx        [ΣL, 1024]     各图 caption 的文本特征（按图序拼接）
      rows/cols  [ΣN]           每个 image token 的 (row, col)，供 RoPE
      mod_index  [ΣN]           token -> 图 的索引
      返回       [ΣN, out_dim]

    ## `chunk`：消掉 AdaLN 物化，同时**完整保留 NaViT 语义**

    朴素打包路径里调制必须 gather 成逐 token 的 `[ΣN, 3D]`，是实打实的缓冲区
    （真机归因 1.01 MB/token，anima-mem-probe 的 V4 单变量：关掉 AdaLN 调制后
    20.60G -> 3.87G）。

    **它挡住的是"展开路径"，不是 remat 档位本身**（这一条被 anima-ragged-probe
    的真机数据修正过，别再按旧说法推理）：
      * `lax.scan` 已经能压住跨块提升，但 scan 的语义是"未被 remat 掉的激活按 28 次
        迭代堆叠"，所以 scan 下 every2/none 一定 OOM —— 这与布局无关，实测
        scan+every2 的 HLO temporaries 分桶 3.6 / 打包 3.8 MB/token，几乎相同。
      * 真正快的是**展开 + every2**（真机 35.5k 真tok/s，比同配置 scan+full 快 21.6%）。
      * 而朴素打包**展开时** 1.01 MB/token，budget 16384 的 full 档就 OOM，
        于是这条最快的路对它是关着的。chunk 化就是来开这扇门的。

    但段长已经量化到 Q 的倍数（`packing.quantize_len`），所以**整个 pack 可以按 Q
    切成 chunk，每个 chunk 完整落在一张图内**。于是：

        x:          [ΣN, D]      -> [ΣN/Q, Q, D]      纯 reshape，零成本
        调制:       [ΣN, 3D]     -> [ΣN/Q, 1, 3D]     广播，XLA 融合，不物化
        gather 目标: ΣN 行        -> ΣN/Q 行           小 Q 倍（Q=1024 时 1024x）

    注意力、块对角 mask、段几何、splash 内核**一律不变**（`chunk_attn` 只做 reshape）。
    也就是说任意 token 数混装、任意宽高比、跳块全部保留 —— 这条路径不牺牲 NaViT 的
    任何能力，只把调制换了个摆法。

    **硬前提：`chunk` 必须整除每一个段长**，否则某个 chunk 会跨两张图，`mod_index`
    在 chunk 内不恒定，调制就会静默用错图的 t（不报错，训练照跑，结果全错）。
    调用方必须保证（`packing.Layout.max_chunk` 会算出合法值）。这里只能校验能整除
    总长，段内一致性在 trace 里查不了。
    """
    if chunk is None:
        return _forward_core(params, cfg, tokens, timesteps, ctx, rows, cols,
                             lambda h: jnp.take(h, mod_index, axis=0),
                             self_attn_fn, cross_attn_fn, loras, remat, barrier)
    n = tokens.shape[0]
    if n % chunk:
        raise ValueError(f"chunk={chunk} 不能整除 pack 长度 {n}")
    c = n // chunk
    ci = mod_index[::chunk]                       # 每个 chunk 的宿主图
    out = _forward_core(params, cfg,
                        tokens.reshape(c, chunk, tokens.shape[-1]), timesteps, ctx,
                        rows.reshape(c, chunk), cols.reshape(c, chunk),
                        lambda h: jnp.take(h, ci, axis=0)[:, None, :],
                        chunk_attn(self_attn_fn), chunk_attn(cross_attn_fn),
                        loras, remat, barrier)
    return out.reshape(n, out.shape[-1])


def forward_ragged(params: PyTree, cfg: AnimaConfig,
                   tokens: jnp.ndarray, timesteps: jnp.ndarray, ctx: jnp.ndarray,
                   rows: jnp.ndarray, cols: jnp.ndarray,
                   self_attn_fn, cross_attn_fn,
                   loras=None, remat="full", barrier: bool = False) -> jnp.ndarray:
    """**批（量化分桶）布局**：每图占一个批元素，长度统一填充到 L。

      tokens     [G, L, in_dim]  每图 patchify 后零填充到 L（L = 量化后的 token 数）
      timesteps  [G]
      ctx        [G, T, 1024]    每图定长 T=512 的文本特征
      rows/cols  [G, L]          RoPE 网格坐标；填充位置填 0
      返回       [G, L, out_dim]

    与 `forward_packed` **共用同一份数值实现**（`_forward_core`），差别只有
    `mod_bcast` 与注意力闭包的形状。任意宽高比同样支持：桶是按**量化后的 token
    总数**分的，不是按 (h, w) 分的 —— 每图保留自己的 (h_i, w_i)，只要
    h_i*w_i <= L；RoPE 本来就是逐 token 查 rows/cols，不依赖矩形网格。
    """
    return _forward_core(params, cfg, tokens, timesteps, ctx, rows, cols,
                         lambda h: h[:, None, :],
                         self_attn_fn, cross_attn_fn, loras, remat, barrier)


def _forward_core(params: PyTree, cfg: AnimaConfig,
                  tokens: jnp.ndarray, timesteps: jnp.ndarray, ctx: jnp.ndarray,
                  rows: jnp.ndarray, cols: jnp.ndarray, mod_bcast,
                  self_attn_fn, cross_attn_fn,
                  loras=None, remat="full", barrier: bool = False) -> jnp.ndarray:
    """`MiniTrainDIT.forward_packed_navit`（models/anima_modeling_core.py，~:1857；
    行号可能漂移，以函数名为准）的复刻，**布局无关**。

    `self_attn_fn(q,k,v)` / `cross_attn_fn(q,k,v)` 由调用方注入（稠密 / splash /
    分桶 vmap），段几何被闭包在里面 —— 这样"换布局"只换函数、不动模型代码。
    `mod_bcast` 见 `block_forward` 的说明。
    """
    net = params
    x = tokens @ net["x_embedder"].T.astype(tokens.dtype)

    sincos = timestep_sincos(timesteps, cfg.model_channels).astype(x.dtype)
    adaln_lora = dense(silu(dense(sincos, net["t_embedder_1"])), net["t_embedder_2"])
    emb = rms_norm(sincos, net["t_embedding_norm"], cfg.eps_rms)      # [G, D]

    cos, sin = packed_rope_cos_sin(rows, cols, cfg.head_dim,
                                   cfg.rope_h_ratio, cfg.rope_w_ratio)

    # 逐块 remat：反向时一次只重算一块，峰值激活 ≈ 1 块而不是 28 块。
    # （Krea2 的 FSDP 经验"all_gather 必须在 remat 边界内"在这里不适用——
    #   Anima 权重不分片，块内没有集合通信。）
    #
    # **remat 强度是可调的，不是布尔量**（见 resolve_remat 的说明）：真机 D1
    # 实测峰值 HBM 只有 4.9/15.7 GiB，10.8 GiB 闲置，而全量 remat 要把前向整个
    # 重算一遍（约占总计算 1/3）。用哪档由 budget 与实测显存共同决定。
    #
    # **层号 i 必须走闭包默认参数，不能作为 checkpoint 的入参**：jax.checkpoint 会把
    # 入参统统 trace 成 tracer，于是 f"blocks.{i}" 拼出 "blocks.Traced<...>"，
    # LoRA 键全部匹配不上 -> LoRA 静默失效、梯度恒 0（本地冒烟实测 sum|grad_b|=0）。
    # 这种错不报异常，只会让"训练"变成空转，且 XLA 可能把反向整个 DCE 掉、
    # 使步时假性变快 —— 属于必须靠断言拦住的一类。
    # **返回值第一个是规范化后的档位字符串（mode），不是 checkpoint policy**：
    # `True -> "full"`、`False -> "none"`。下面 scan 分支的 `mode == "every2"` 必须
    # 用它，不能用原始的 `remat` 参数 —— 现在两者恰好等价（True/False 都不等于
    # "every2"），但只要 resolve_remat 将来加任何别名映射（如 "every_two"），
    # 用原始参数就会静默走错分支（走进普通 scan、悄悄改掉重算策略，不报错）。
    #
    # `unrolled` 决定 `dots` 放不放行：scan 下 dots_saveable 必 OOM，见 resolve_remat。
    unrolled = isinstance(params["blocks"], (list, tuple))
    mode, wrap = resolve_remat(remat, unrolled=unrolled)
    if unrolled:
        # 展开路径：逐块单独编译。对拍脚本按块比对需要它，**训练也用得上** ——
        # 但必须配 chunk 或 barrier（见下），否则就是 stack_blocks 注释里那个
        # 1.01 MB/token 的朴素展开（budget 16384 即 OOM）。压住之后它是真机
        # 最快的路径（budget 8192 + every2 = 30.2k 真tok/s，见 train.TrainConfig）。
        # `barrier`：在每块边界把 (x, emb, adaln_lora) 一起穿过 optimization_barrier，
        # **人为制造 emb -> x 的数据依赖**，于是第 i 块的 AdaLN 调制不可能早于第 i-1
        # 块的 x 算出来 —— 正是 anima-mem-probe 归因出的那个"28 组调制同时活着"。
        #
        # 相对 chunk 化的好处：`optimization_barrier` 是**恒等算子**，一个 bit 都不改，
        # 只给 XLA 加调度约束。chunk 化虽然数学等价，但会改变 matmul 的 lowering，
        # 真机实测 bf16 梯度有 ULP 量级的漂移。
        # 相对 chunk 化的坏处：它只压住调度，不缩小 gather 的目标（仍是 [ΣN, 3D]），
        # 所以省的是"28 份同时活着"而不是"每份本身的大小"。
        # 两者正交，可以叠加；哪个够用要实测。
        for i in range(cfg.num_blocks):
            # LoraCtx 的三棵树恒是 scan 布局（每个 target 一份 [L,...]），展开路径
            # 要自己切第 i 片；切完键就是块内相对的，故 layer=None。
            lo_i, lay = (_slice_ctx(loras, i), None) if isinstance(loras, LoraCtx) \
                else (loras, i)

            def one(carry, p, _lo=lo_i, _lay=lay):
                return block_forward(carry, p, cfg, emb, adaln_lora, mod_bcast, ctx,
                                     cos, sin, self_attn_fn, cross_attn_fn, _lo, _lay)
            x = wrap(one, i)(x, params["blocks"][i])
            if barrier and i + 1 < cfg.num_blocks:
                x, emb, adaln_lora = jax.lax.optimization_barrier(
                    (x, emb, adaln_lora))
    else:
        # scan 路径（训练用）：loras 的键必须是块内相对的（stack_loras 的产物）
        def run_block(carry, p, lo):
            return block_forward(carry, p, cfg, emb, adaln_lora, mod_bcast, ctx,
                                 cos, sin, self_attn_fn, cross_attn_fn, lo, None)

        # **cfg 不能进 xs**：它是 AdapterConfig（frozen dataclass，不是 pytree
        # 节点），jax.tree.map 会当成叶子去切第 0 维 -> 报 "不可下标"。所以三棵
        # 数据树进 xs、cfg 走闭包重建。
        if isinstance(loras, LoraCtx):
            acfg = loras.cfg
            xs = (params["blocks"], (loras.params, loras.consts, loras.drop))
            rebuild = lambda tri: LoraCtx(acfg, *tri)
        else:
            xs = (params["blocks"], loras)
            rebuild = lambda lo: lo
        _run = run_block
        run_block = lambda carry, p, lo: _run(carry, p, rebuild(lo))
        if mode == "every2":
            # 隔块 remat 在 scan 下要**成对**做：把 [L,...] 重排成 [L/2, 2, ...]，
            # 一次循环走两块，第一块不 remat、第二块 remat。
            # （wrap 的 `i % 2` 判据在 scan 里用不了 —— 循环变量是 tracer。）
            if cfg.num_blocks % 2:
                raise ValueError(f"remat='every2' 在 scan 路径下需要偶数块，"
                                 f"当前 {cfg.num_blocks}；改用 full/none"
                                 f"（dots 在 scan 下不可用，见 resolve_remat）")
            pair = lambda z: z.reshape(cfg.num_blocks // 2, 2, *z.shape[1:])
            xs = jax.tree.map(pair, xs)

            def body(carry, layer):
                p, lo = layer
                take = lambda t, j: jax.tree.map(lambda z: z[j], t)
                carry = run_block(carry, take(p, 0), take(lo, 0))
                carry = jax.checkpoint(
                    lambda c: run_block(c, take(p, 1), take(lo, 1)))(carry)
                return carry, None
            x, _ = jax.lax.scan(body, x, xs, length=cfg.num_blocks // 2)
        else:
            def body(carry, layer):
                p, lo = layer
                return run_block(carry, p, lo), None
            x, _ = jax.lax.scan(wrap(body, 0), x, xs, length=cfg.num_blocks)

    # FinalLayer：2 chunk，且只取 adaln_lora 的前 2D
    # （`FinalLayer.forward_tokens`，models/anima_modeling_core.py ~:1033）
    h = silu(emb)
    h = dense(h, net["final_adaln_1"])
    h = dense(h, net["final_adaln_2"]) + adaln_lora[:, : 2 * cfg.model_channels]
    shift, scale = jnp.split(mod_bcast(h), 2, axis=-1)
    x = layer_norm(x, cfg.eps_ln) * (1 + scale) + shift
    out = dense(x, net["final_linear"])
    return output_tokens_to_patch_tokens(out, cfg)


def output_tokens_to_patch_tokens(tokens: jnp.ndarray, cfg: AnimaConfig) -> jnp.ndarray:
    """`MiniTrainDIT._output_tokens_to_patch_tokens`（models/anima_modeling_core.py，
    ~:1721；行号可能漂移，以函数名为准）—— `(ph pw pt c) -> (c pt ph pw)`。

    **不是恒等重排**（Krea2 那边才是）。final_layer 吐的是 unpatchify 折叠用的
    (ph pw pt c) 序，而训练目标来自 patchify_latents_to_tokens 的 (c pt ph pw) 序。
    漏掉这一步：token 位置正确、每个 token 内部 64 个通道被打乱 —— 统计量完全正常、
    逐元素全错，本地对拍实测 rel 从 1.4 降到 3e-5 就是这一处。
    """
    p, c = cfg.patch, cfg.out_dim // (cfg.patch ** 2)
    lead = tokens.shape[:-1]                       # 打包 (ΣN,) / 批 (G, L)
    n = len(lead)
    t = tokens.reshape(*lead, p, p, 1, c)          # ..., ph, pw, pt, c
    t = jnp.transpose(t, (*range(n), n + 3, n + 2, n + 0, n + 1))
    return t.reshape(*lead, cfg.out_dim)


# ── 权重加载 ──────────────────────────────────────────────────────────────────
_DT = {"BF16": jnp.bfloat16, "F16": jnp.float16, "F32": jnp.float32}
_NP = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32}


def load_safetensors_anima(path: str, dtype=jnp.bfloat16) -> Tuple[PyTree, AnimaConfig]:
    """把 anima-base-v1.0.safetensors 读成 JAX pytree（丢掉 llm_adapter —— 文本侧
    在 PyTorch 离线算好，TPU 只跑 DiT）。

    bf16 用 uint16 中转再 bitcast：numpy 没有原生 bfloat16，直接 view 成 float16
    会把位模式解释错（静默出错，不报异常）。
    """
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen))
        hdr.pop("__metadata__", None)
        base = 8 + hlen

        def get(name: str) -> jnp.ndarray:
            meta = hdr[name]
            s, e = meta["data_offsets"]
            f.seek(base + s)
            raw = f.read(e - s)
            arr = np.frombuffer(raw, dtype=_NP[meta["dtype"]]).reshape(meta["shape"])
            if meta["dtype"] == "BF16":
                out = jax.lax.bitcast_convert_type(jnp.asarray(arr), jnp.bfloat16)
            else:
                out = jnp.asarray(arr)
            return out.astype(dtype)

        n_blocks = len({k.split(".")[2] for k in hdr if k.startswith("net.blocks.")})
        cfg = AnimaConfig(num_blocks=n_blocks)

        blocks = []
        for i in range(n_blocks):
            b = f"net.blocks.{i}"
            blocks.append({
                "self_attn": {n: get(f"{b}.self_attn.{n}.weight")
                              for n in ("q_proj", "k_proj", "v_proj", "output_proj",
                                        "q_norm", "k_norm")},
                "cross_attn": {n: get(f"{b}.cross_attn.{n}.weight")
                               for n in ("q_proj", "k_proj", "v_proj", "output_proj",
                                         "q_norm", "k_norm")},
                "mlp": {n: get(f"{b}.mlp.{n}.weight") for n in ("layer1", "layer2")},
                "adaln_self_1": get(f"{b}.adaln_modulation_self_attn.1.weight"),
                "adaln_self_2": get(f"{b}.adaln_modulation_self_attn.2.weight"),
                "adaln_cross_1": get(f"{b}.adaln_modulation_cross_attn.1.weight"),
                "adaln_cross_2": get(f"{b}.adaln_modulation_cross_attn.2.weight"),
                "adaln_mlp_1": get(f"{b}.adaln_modulation_mlp.1.weight"),
                "adaln_mlp_2": get(f"{b}.adaln_modulation_mlp.2.weight"),
            })

        params = {
            "x_embedder": get("net.x_embedder.proj.1.weight"),
            "t_embedder_1": get("net.t_embedder.1.linear_1.weight"),
            "t_embedder_2": get("net.t_embedder.1.linear_2.weight"),
            "t_embedding_norm": get("net.t_embedding_norm.weight"),
            "final_adaln_1": get("net.final_layer.adaln_modulation.1.weight"),
            "final_adaln_2": get("net.final_layer.adaln_modulation.2.weight"),
            "final_linear": get("net.final_layer.linear.weight"),
            "blocks": blocks,
        }
    return params, cfg


def lora_scaling(rank: int, alpha: Optional[float]) -> float:
    """ΔW 的缩放系数，口径同 `LoRALayer.__init__` 的 `self.scaling = alpha / rank`
    （trainer/lora.py:287）。

    alpha=None 时取 alpha=rank（scaling=1），这也是本仓库落地 adapter 的默认口径
    （`_pissa_init` 的 docstring 注明 PiSSA 等要求 alpha=rank 才能保证 step-0
    净增量为 0，trainer/lora.py:92）。
    导出 safetensors 时必须把这个 alpha 一起写进去，否则 ComfyUI 侧会按自己的
    默认值重算 scaling —— 权重对、强度错。
    """
    return 1.0 if alpha is None else float(alpha) / float(rank)


def init_lora(key, cfg: AnimaConfig, rank: int = 32,
              targets: Sequence[str] = ("self_attn.q_proj", "self_attn.k_proj",
                                        "self_attn.v_proj", "self_attn.output_proj",
                                        "cross_attn.q_proj", "cross_attn.k_proj",
                                        "cross_attn.v_proj", "cross_attn.output_proj",
                                        "mlp.layer1", "mlp.layer2"),
              dtype=jnp.bfloat16) -> Dict[str, Dict[str, jnp.ndarray]]:
    """B 恒为 0 -> step-0 净增量为 0（本仓库落地新 adapter 的硬约束，见 skill 2.2）。

    A 的初始化对齐 PyTorch 侧 `LoRALayer.__init__` 里的
    `kaiming_uniform_(self.lora_down.weight, a=5**0.5)`（trainer/lora.py:367）：
    该调用等价于 U(-b, b)，b = sqrt(6 / ((1+5) * fan_in)) = 1/sqrt(fan_in)，
    fan_in = in_features。

    （原实现用的是 N(0, 1/sqrt(in))，std 比这里大约 sqrt(3) 倍。B=0 时 step-0
      仍然中立，但 A 的量级会改变有效学习率与早期动力学 —— 两侧口径不一致会让
      "同一份 yaml 在 GPU 与 TPU 上训出不同结果"，且查不出原因。）

    形状表**直接取 `adapters.target_shapes(cfg)`**，不再在这里抄一份：两处以前是同一
    张表硬编码两遍（本地实测键集与逐键值完全相同，mlp_ratio=2.5/3.0/4.0 都一致），
    改一处漏一处就是"init 出的 A 形状与 plan_targets 算的 rank cap 对不上"，
    而两边都不报错。
    """
    shape = AD.target_shapes(cfg)
    out = {}
    keys = jax.random.split(key, cfg.num_blocks * len(targets))
    n = 0
    for i in range(cfg.num_blocks):
        for t in targets:
            i_dim, o_dim = shape[t]
            bound = 1.0 / math.sqrt(i_dim)      # = kaiming_uniform(a=sqrt(5)) 的边界
            out[f"blocks.{i}.{t}"] = {
                "a": jax.random.uniform(keys[n], (i_dim, rank), jnp.float32,
                                        minval=-bound, maxval=bound).astype(dtype),
                "b": jnp.zeros((rank, o_dim), dtype),
            }
            n += 1
    return out
