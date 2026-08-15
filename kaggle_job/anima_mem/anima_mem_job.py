# ！！自动生成，勿手改 —— 改 jax_tpu/*.py 或 _preamble/_job_body 后
# 重跑 build_job.py 生成。
from __future__ import annotations


# ==========================================================================
# ① bootstrap（_preamble.py）—— 必须在 import jax 之前
# ==========================================================================

"""生成脚本的**第一段**：在任何 `import jax` 之前把 jax/libtpu 装到指定版本。

Kaggle 默认镜像是 jax 0.10.2 + libtpu 构建于 2025-06-12，比 Pallas 的版本闸门
（`is_cloud_tpu_older_than`，硬 raise、**无环境变量旁路**）老约 14 个月，不升级则
所有 splash 探测以同一原因失败。**必须在 import jax 之前**——jax 一旦初始化后端
就换不掉 libtpu。这也是 anima_jax.py 的 import 必须排在本段之后的原因。

与上一版的区别：**钉死版本**而不是 `-U`。
  * 上一轮 `-U` 实际装到了 0.11.0（见 anima-tpu-dp-probe 日志），钉死同一版
    使真机结果可复现，也与本地对拍环境（jaxenv: Python 3.12 + jax 0.11.0）一致；
  * 本地/真机版本一旦漂移，"本地过了真机挂"会变成查不动的问题。
Kaggle 允许自由装依赖（docs/notebooks#modifying-a-notebook-specific-environment），
所以这里就是一次正常的 pip install，不是什么绕过手段。
"""

import os
import subprocess
import sys
import time

JAX_VERSION = "0.11.0"          # 与本地 jaxenv 对齐；改这里要同步改 tests/README.md

_BOOT = "未开启"
if os.environ.get("ANIMA_TPU_UPGRADE", "1") == "1":
    if "jax" in sys.modules:
        _BOOT = "[!] jax 已 import，安装无效"
    else:
        _t = time.time()
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q",
                 f"jax[tpu]=={JAX_VERSION}"],
                check=True, capture_output=True, timeout=1200)
            _BOOT = f"jax[tpu]=={JAX_VERSION} 安装 OK {time.time() - _t:.0f}s"
        except subprocess.CalledProcessError as _e:
            _tail = (_e.stderr or b"").decode("utf-8", "replace").strip().splitlines()[-3:]
            _BOOT = f"安装失败（继续跑，Pallas 大概率被闸门挡住）：{' | '.join(_tail)}"
        except Exception as _e:
            _BOOT = f"安装失败：{type(_e).__name__}: {_e}"
print(f"[ INFO ] bootstrap - {_BOOT}", flush=True)

# ==========================================================================
# ② 模型（jax_tpu/anima_jax.py）
# ==========================================================================

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

对齐依据（逐条指向 PyTorch 源）：
  models/anima_modeling_core.py:299  RMSNorm（fp32 归一后乘 weight）
  models/anima_modeling_core.py:324  GPT2FeedForward（nn.GELU 默认 = erf 精确式，非 tanh）
  models/anima_modeling_core.py:436  Attention（q/k_norm 是**逐 head_dim** RMSNorm eps=1e-6）
  models/anima_modeling_core.py:277  RoPE（rotate_half 式，cos/sin 转成 t.dtype 后相乘）
  models/anima_modeling_core.py:719  Timesteps（cat[cos, sin]，注意 cos 在前）
  models/anima_modeling_core.py:742  TimestepEmbedding（use_adaln_lora 时 emb=**原始正弦**，
                                     MLP 输出走 adaln_lora 这条支路）
  models/anima_modeling_core.py:967  Block（LayerNorm elementwise_affine=False eps=1e-6）
  models/anima_modeling_core.py:1174 Block.forward_tokens（三组 AdaLN + mod_index gather）
  models/anima_modeling_core.py:926  FinalLayer.forward_tokens（2 chunk，取 adaln_lora 前 2D）
  models/anima_modeling_core.py:1663 _packed_rope_from_grid（cat[t,h,w]*2，t 段恒 0）
  models/anima_modeling_core.py:1774 forward_packed_navit（整体装配）

本文件只做**前向**，不含优化器/数据/采样——那些留在 PyTorch 侧离线完成。
"""


import json
import math
import struct
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

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
    eps_rms: float = 1e-5           # t_embedding_norm
    # RoPE 的 NTK 外推系数。trainer/models.py:156 对 in_channels==16 用 4.0，
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
    """models/anima_modeling_core.py:308 —— fp32 归一、type_as(x) 回原 dtype、再乘 weight。

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
    """t: [..., S, H, D]；cos/sin: [S, 1, D_rot]。

    PyTorch(_apply_rotary_pos_emb_base) 先把 cos/sin `.to(t.dtype)` 再相乘 —— bf16 下
    这一步的舍入会进结果，所以这里也先转 dtype 再算，不留在 fp32。
    """
    rot = cos.shape[-1]
    t_rot, t_pass = t[..., :rot], t[..., rot:]
    c, s = cos.astype(t.dtype), sin.astype(t.dtype)
    out = t_rot * c + rotate_half(t_rot) * s
    return jnp.concatenate([out, t_pass], axis=-1) if t_pass.shape[-1] else out


def timestep_sincos(t: jnp.ndarray, num_channels: int) -> jnp.ndarray:
    """models/anima_modeling_core.py:724。**cos 在前、sin 在后**（与 diffusers 相反）。

    exponent = -log(10000) * arange(half) / half —— 分母是 half_dim 本身（不减 1）。
    """
    half = num_channels // 2
    exponent = -math.log(10000.0) * jnp.arange(half, dtype=jnp.float32) / float(half)
    emb = t.astype(jnp.float32)[:, None] * jnp.exp(exponent)[None, :]
    return jnp.concatenate([jnp.cos(emb), jnp.sin(emb)], axis=-1)


def packed_rope_cos_sin(rows: jnp.ndarray, cols: jnp.ndarray, head_dim: int,
                        h_ratio: float = 4.0, w_ratio: float = 4.0
                        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """models/anima_modeling_core.py:1663 + :605。

    head_dim=128 -> dim_h = dim_w = 128//6*2 = 42，dim_t = 128 - 84 = 44。
    emb = cat([t_half(22 个 0), h_half(21), w_half(21)] * 2) -> 128 维。
    时间轴恒 0（图像模型 T=1），但**必须保留这 44 个 0 通道**——它们占着 RoPE 的前
    44 维，去掉会让 h/w 的通道偏移错位。
    """
    dim_h = head_dim // 6 * 2
    dim_t = head_dim - 2 * dim_h
    spatial_range = jnp.arange(0, dim_h, 2, dtype=jnp.float32)[: dim_h // 2] / dim_h
    # models/anima_modeling_core.py:644 —— ntk_factor = ratio**(dim/(dim-2))，theta 再乘它
    h_theta = 10000.0 * (h_ratio ** (dim_h / (dim_h - 2)))
    w_theta = 10000.0 * (w_ratio ** (dim_h / (dim_h - 2)))
    h_freqs = 1.0 / (h_theta ** spatial_range)
    w_freqs = 1.0 / (w_theta ** spatial_range)
    # t 轴位置恒 0（图像模型 T=1）-> half_emb_t 恒 0，与频率无关
    n = rows.shape[0]
    half_t = jnp.zeros((n, dim_t // 2), jnp.float32)
    half_h = rows.astype(jnp.float32)[:, None] * h_freqs
    half_w = cols.astype(jnp.float32)[:, None] * w_freqs
    emb = jnp.concatenate([half_t, half_h, half_w] * 2, axis=-1)   # [N, head_dim]
    return jnp.cos(emb)[:, None, :], jnp.sin(emb)[:, None, :]


# ── LoRA ─────────────────────────────────────────────────────────────────────
def dense(x: jnp.ndarray, w: jnp.ndarray, lora: Optional[Dict[str, jnp.ndarray]] = None,
          scale: float = 1.0) -> jnp.ndarray:
    """y = x @ Wᵀ (+ scale * (x @ A) @ B)。

    权重按 PyTorch nn.Linear 存放（[out, in]），故这里转置。LoRA 走**低秩旁路**而非
    合并——与 ComfyUI 的 Bypass 加载路径同构（见 memory krea2-comfyui-lora-deploy），
    也避免物化 ΔW。
    """
    y = x @ w.T.astype(x.dtype)
    if lora is not None:
        y = y + scale * ((x @ lora["a"].astype(x.dtype)) @ lora["b"].astype(x.dtype))
    return y


def _lora(loras: Optional[Dict[str, Any]], key: str):
    return None if loras is None else loras.get(key)


# ── 注意力后端 ────────────────────────────────────────────────────────────────
def attention_dense(q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray,
                    bias: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """q: [S, H, D]，k/v: [T, H, D]，bias: [S, T] 加性（0 / -inf）。返回 [S, H, D]。

    参考实现（本地对拍 + 小规模用）。softmax 在 fp32 上做，与 SDPA 一致。
    O(S·T) 显存，只适合 S·T 不大的场合；大 pack 走 splash（TPU）。
    """
    d = q.shape[-1]
    logits = jnp.einsum("shd,thd->hst", q, k).astype(jnp.float32) / math.sqrt(d)
    if bias is not None:
        logits = logits + bias[None, :, :]
    w = jax.nn.softmax(logits, axis=-1).astype(v.dtype)
    return jnp.einsum("hst,thd->shd", w, v)


def block_diag_bias(q_lens: Sequence[int], kv_lens: Optional[Sequence[int]] = None,
                    dtype=jnp.float32) -> jnp.ndarray:
    """块对角加性 mask：同一图的 q 只看得见同一图的 kv。kv_lens=None 时为自注意力。"""
    kv_lens = q_lens if kv_lens is None else kv_lens
    qi = np.repeat(np.arange(len(q_lens)), np.asarray(q_lens))
    ki = np.repeat(np.arange(len(kv_lens)), np.asarray(kv_lens))
    allow = qi[:, None] == ki[None, :]
    return jnp.asarray(np.where(allow, 0.0, -np.inf), dtype=dtype)


# ── 单块前向 ──────────────────────────────────────────────────────────────────
def attn_qkv(x: jnp.ndarray, ctx: Optional[jnp.ndarray], p: Dict[str, Any],
             cfg: AnimaConfig, loras, prefix: str,
             cos=None, sin=None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """q/k/v 投影 + 逐 head RMSNorm(+RoPE，仅自注意力)。"""
    H, D = cfg.num_heads, cfg.head_dim
    src = x if ctx is None else ctx
    q = dense(x, p["q_proj"], _lora(loras, f"{prefix}.q_proj")).reshape(-1, H, D)
    k = dense(src, p["k_proj"], _lora(loras, f"{prefix}.k_proj")).reshape(-1, H, D)
    v = dense(src, p["v_proj"], _lora(loras, f"{prefix}.v_proj")).reshape(-1, H, D)
    q = rms_norm(q, p["q_norm"], cfg.eps_qk)
    k = rms_norm(k, p["k_norm"], cfg.eps_qk)
    if ctx is None and cos is not None:          # RoPE 只作用于自注意力
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    return q, k, v


def block_forward(x: jnp.ndarray, p: Dict[str, Any], cfg: AnimaConfig,
                  emb: jnp.ndarray, adaln_lora: jnp.ndarray, mod_index: jnp.ndarray,
                  ctx: jnp.ndarray, cos, sin,
                  self_attn_fn, cross_attn_fn,
                  loras=None, layer: int = 0) -> jnp.ndarray:
    """models/anima_modeling_core.py:1174 的逐算子复刻。

    emb: [G, D] 逐图 timestep 向量；adaln_lora: [G, 3D]；mod_index: [ΣN] token->图。
    调制 MLP 只在 G 行上算，再 gather 到逐 token —— 与 PyTorch 的 mod_index 路径同布局。
    """
    D = cfg.model_channels
    lp = f"blocks.{layer}"

    def modulation(name: str):
        h = silu(emb)
        h = dense(h, p[f"adaln_{name}_1"], _lora(loras, f"{lp}.adaln_{name}.1"))
        h = dense(h, p[f"adaln_{name}_2"], _lora(loras, f"{lp}.adaln_{name}.2"))
        h = h + adaln_lora                                             # [G, 3D]
        # **先 gather 再 split**，不是 split 完 gather 三次：搬运字节数相同，
        # 但少两次 gather 算子。调制 MLP 只在 G 行上算，与 PyTorch 的 mod_index
        # 路径同布局（memory navit-adaln-tokenwise-overhead：per-image 布局比
        # 逐 token 布局快 13%）。
        return jnp.split(jnp.take(h, mod_index, axis=0), 3, axis=-1)   # 3 x [ΣN, D]

    sh, sc, gt = modulation("self")
    h = layer_norm(x, cfg.eps_ln) * (1 + sc) + sh
    q, k, v = attn_qkv(h, None, p["self_attn"], cfg, loras, f"{lp}.self_attn", cos, sin)
    a = self_attn_fn(q, k, v).reshape(-1, D)
    x = x + gt * dense(a, p["self_attn"]["output_proj"],
                       _lora(loras, f"{lp}.self_attn.output_proj"))

    sh, sc, gt = modulation("cross")
    h = layer_norm(x, cfg.eps_ln) * (1 + sc) + sh
    q, k, v = attn_qkv(h, ctx, p["cross_attn"], cfg, loras, f"{lp}.cross_attn")
    a = cross_attn_fn(q, k, v).reshape(-1, D)
    x = x + gt * dense(a, p["cross_attn"]["output_proj"],
                       _lora(loras, f"{lp}.cross_attn.output_proj"))

    sh, sc, gt = modulation("mlp")
    h = layer_norm(x, cfg.eps_ln) * (1 + sc) + sh
    h = gelu_exact(dense(h, p["mlp"]["layer1"], _lora(loras, f"{lp}.mlp.layer1")))
    x = x + gt * dense(h, p["mlp"]["layer2"], _lora(loras, f"{lp}.mlp.layer2"))
    return x


# ── remat 强度 ────────────────────────────────────────────────────────────────
#: 可选档位。**冻结底模只训 LoRA 时反向没有 wgrad**，所以一次训练步 ≈
#: 前向(1) + 重算前向(1) + 激活梯度(1)；关掉重算最多省掉其中 1/3。
#: 显存与速度是直接对冲的，最优档只能实测（真机 D1：峰值 4.9/15.7 GiB，余量很大）。
REMAT_CHOICES = ("full", "dots", "every2", "none")


def resolve_remat(remat):
    """把 remat 设定翻成 (policy, wrap(fn, layer_idx)->fn)。

      full   每块整个重算——激活最省、算力最贵（原实现，且是当前唯一真机验证过的档）
      dots   保留所有矩阵乘的输出，只重算 norm/激活函数等便宜算子
             （jax.checkpoint_policies.dots_saveable）——省掉绝大部分重算，
             但要存 q/k/v/attn/mlp 隐藏层，显存涨得最多
      every2 隔块 remat：偶数块存、奇数块重算——粗粒度的折中
      none   完全不 remat：最快，但按结构估算需要约 2.2MB/token 的激活
             （**估算值，未实测**），11.8GiB 余量只够约 5k token/chip

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
    pol = jax.checkpoint_policies.dots_saveable
    return remat, (lambda fn, i: jax.checkpoint(fn, policy=pol))


# ── 整模前向（NaViT packed）───────────────────────────────────────────────────
def forward_packed(params: PyTree, cfg: AnimaConfig,
                   tokens: jnp.ndarray, timesteps: jnp.ndarray, ctx: jnp.ndarray,
                   rows: jnp.ndarray, cols: jnp.ndarray, mod_index: jnp.ndarray,
                   self_attn_fn, cross_attn_fn,
                   loras=None, remat="full") -> jnp.ndarray:
    """models/anima_modeling_core.py:1774 的复刻。

      tokens     [ΣN, in_dim]   已 patchify 的图像 token（按图序拼接，含 padding 段）
      timesteps  [G]            每图一个 t
      ctx        [ΣL, 1024]     各图 caption 的文本特征（按图序拼接）
      rows/cols  [ΣN]           每个 image token 的 (row, col)，供 RoPE
      mod_index  [ΣN]           token -> 图 的索引
      返回       [ΣN, out_dim]

    `self_attn_fn(q,k,v)` / `cross_attn_fn(q,k,v)` 由调用方注入（稠密 / splash），
    段几何被闭包在里面 —— 这样"换布局"只换函数、不动模型代码。
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
    policy, wrap = resolve_remat(remat)
    for i in range(cfg.num_blocks):
        def one(carry, p, _i=i):
            return block_forward(carry, p, cfg, emb, adaln_lora, mod_index, ctx,
                                 cos, sin, self_attn_fn, cross_attn_fn, loras, _i)
        step = wrap(one, i)
        x = step(x, params["blocks"][i])

    # FinalLayer：2 chunk，且只取 adaln_lora 的前 2D（models/...:950）
    h = silu(emb)
    h = dense(h, net["final_adaln_1"])
    h = dense(h, net["final_adaln_2"]) + adaln_lora[:, : 2 * cfg.model_channels]
    shift, scale = jnp.split(h, 2, axis=-1)
    take = lambda t: jnp.take(t, mod_index, axis=0)
    x = layer_norm(x, cfg.eps_ln) * (1 + take(scale)) + take(shift)
    out = dense(x, net["final_linear"])
    return output_tokens_to_patch_tokens(out, cfg)


def output_tokens_to_patch_tokens(tokens: jnp.ndarray, cfg: AnimaConfig) -> jnp.ndarray:
    """models/anima_modeling_core.py:1638 —— `(ph pw pt c) -> (c pt ph pw)`。

    **不是恒等重排**（Krea2 那边才是）。final_layer 吐的是 unpatchify 折叠用的
    (ph pw pt c) 序，而训练目标来自 patchify_latents_to_tokens 的 (c pt ph pw) 序。
    漏掉这一步：token 位置正确、每个 token 内部 64 个通道被打乱 —— 统计量完全正常、
    逐元素全错，本地对拍实测 rel 从 1.4 降到 3e-5 就是这一处。
    """
    p, c = cfg.patch, cfg.out_dim // (cfg.patch ** 2)
    n = tokens.shape[0]
    return tokens.reshape(n, p, p, 1, c).transpose(0, 4, 3, 1, 2).reshape(n, cfg.out_dim)


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
    """ΔW 的缩放系数，口径同 trainer/lora.py:287 `self.scaling = alpha / rank`。

    alpha=None 时取 alpha=rank（scaling=1），这也是本仓库落地 adapter 的默认口径
    （lora.py:92 注明 PiSSA 等要求 alpha=rank 才能保证 step-0 净增量为 0）。
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

    A 的初始化对齐 PyTorch 侧 trainer/lora.py:367 的
    `kaiming_uniform_(lora_down.weight, a=sqrt(5))`：该调用等价于
    U(-b, b)，b = sqrt(6 / ((1+5) * fan_in)) = 1/sqrt(fan_in)，fan_in = in_features。

    （原实现用的是 N(0, 1/sqrt(in))，std 比这里大约 sqrt(3) 倍。B=0 时 step-0
      仍然中立，但 A 的量级会改变有效学习率与早期动力学 —— 两侧口径不一致会让
      "同一份 yaml 在 GPU 与 TPU 上训出不同结果"，且查不出原因。）
    """
    D, C = cfg.model_channels, cfg.crossattn_dim
    shape = {
        "self_attn.q_proj": (D, D), "self_attn.k_proj": (D, D),
        "self_attn.v_proj": (D, D), "self_attn.output_proj": (D, D),
        "cross_attn.q_proj": (D, D), "cross_attn.k_proj": (C, D),
        "cross_attn.v_proj": (C, D), "cross_attn.output_proj": (D, D),
        "mlp.layer1": (D, int(D * cfg.mlp_ratio)),
        "mlp.layer2": (int(D * cfg.mlp_ratio), D),
    }
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

# ==========================================================================
# ③ 注意力后端（jax_tpu/attention.py）
# ==========================================================================

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


def _block_sizes(q_len: int, kv_len: int):
    """挑一组整除 (q_len, kv_len) 的块大小；前向恒 128，反向尽量取大。"""
    sk, _ = _import_splash()
    if q_len % BLOCK or kv_len % BLOCK:
        raise ValueError(f"序列长必须被 {BLOCK} 整除，得到 q={q_len} kv={kv_len}"
                         f"（splash_attention_mask_info.py 对此硬 raise）")
    bwd_q = next(b for b in BWD_BLOCK_PREF if q_len % b == 0)
    bwd_kv = next(b for b in BWD_BLOCK_PREF if kv_len % b == 0)
    return sk.BlockSizes(
        block_q=BLOCK, block_kv=BLOCK, block_kv_compute=BLOCK,
        block_q_dkv=bwd_q, block_kv_dkv=bwd_kv, block_kv_dkv_compute=bwd_kv,
        use_fused_bwd_kernel=True,
    )


# ── 后端工厂 ──────────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=64)
def _splash_kernel(q_seg_bytes: bytes, kv_seg_bytes: bytes, n_len: int,
                   kv_len: int, num_heads: int, interpret: bool):
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
                              block_sizes=_block_sizes(n_len, kv_len),
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
    kernel = _splash_kernel(q_seg.tobytes(), kv_seg.tobytes(),
                            int(q_seg.shape[0]), int(kv_seg.shape[0]),
                            num_heads, interpret)
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

# ==========================================================================
# ④ job 主体（_job_body.py）
# ==========================================================================

"""Kaggle TPU v5e-8 · NaViT 路径的显存归因（单变量二分）。

## 要回答什么

上一跑（anima-navit-probe）实测：NaViT 打包路径的显存 ≈ **4.0GB + 1.01 MB/token**
（由 16384/full 需 20.60G、32768/full 需 37.18G 两点拟合，8192 能跑且峰值 3.9GiB）。
而分桶批处理路径在 49152 token/chip 时总峰值才 4.9GiB —— 折合 0.02 MB/token，
**差约 50 倍**。

按结构算，full remat 下每块只需存 [B, D] 的块输入 = 4KB/token x 28 = 0.112 MB/token。
实测是它的 9 倍。所以有东西多占了，且是 NaViT 路径特有。**本 job 不猜，逐项关掉测。**

## 方法：OOM 也是数据

XLA 的 RESOURCE_EXHAUSTED 消息里带
`total memory required for HLO temporaries (X G)` —— 即使配置跑不起来，
这个数字也是可比的测量值。所以固定在 **budget=16384 / remat=full**（基线正好
OOM 且需求 20.60G），逐个关掉嫌疑项，看需求降到多少。降得多的那个就是主因。

## 变量表

  V0 基线                      —— 复现 20.60G
  V1 mod_index 恒 0            —— gather 退化成广播（**数值不对**，只测显存）
  V2 关 cross-attn             —— cross_fn 直接返回 0
  V3 反向块 128（而非 1024）   —— arch_probe H1 选的 1024 是为速度，代价未测
  V4 关 AdaLN 调制             —— shift=0/scale=0/gate=1
  V5 只前向不求梯度            —— 分出"前向本身"与"反向残差"的占比

V1/V2/V4 都会让结果数值上错，这是**故意的**：本 job 只测显存，不看 loss。
"""

import json
import os
import platform
import re
import sys
import tempfile
import time
import traceback

import numpy as np

RESULTS: list = []
_T0 = time.time()
_BOOT = globals().get("_BOOT", "未经 build_job.py 拼接，无 bootstrap")
OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else tempfile.gettempdir()

T_TXT = 256
LORA_RANK = 32
# 基线：上一跑实测需 20.60G（正好 OOM，是理想的对照点——降多少一目了然）
BUDGET = 16384
COARSE = [10240, 4096, 2048]
REAL = [10080, 4096, 0]


def _flush():
    try:
        with open(os.path.join(OUT_DIR, "anima_mem.json"), "w", encoding="utf-8") as f:
            json.dump(RESULTS, f, ensure_ascii=False, indent=2)
        with open(os.path.join(OUT_DIR, "anima_mem_report.txt"), "w", encoding="utf-8") as f:
            for r in RESULTS:
                f.write(f"[{r['status']:^6}] {r['name']}"
                        + (f" - {r['detail']}" if r["detail"] else "") + "\n")
    except OSError:
        pass


def record(name, status, detail=""):
    RESULTS.append({"name": name, "status": status, "detail": detail,
                    "t_elapsed_s": round(time.time() - _T0, 1)})
    print(f"[{status:^6}] {name}" + (f" - {detail}" if detail else ""), flush=True)
    _flush()


def probe(name):
    def deco(fn):
        def run(*a, **kw):
            try:
                record(name, "OK", fn(*a, **kw) or "")
                return True
            except Exception as e:
                tb = traceback.format_exc(limit=3).strip().splitlines()[-1]
                record(name, "FAIL", f"{type(e).__name__}: {e} | {tb}")
            return False
        return run
    return deco


_RE_MEM = re.compile(r"HLO temporaries \(([\d.]+)G\)")


def mem_required(build_and_run):
    """跑一次；返回 (状态, 需求GB, 步时ms)。

    **OOM 不算失败**：从异常消息里抠出 `HLO temporaries (X G)`，那就是这个配置的
    显存需求，是本 job 要的测量值。跑成功时读 peak_bytes_in_use。
    """
    import jax
    try:
        ms = build_and_run()
        return "跑通", _hbm_peak(), ms
    except Exception as e:
        m = _RE_MEM.search(str(e))
        if m:
            return "OOM", float(m.group(1)), None
        raise


def _hbm_peak():
    import jax
    st = jax.devices()[0].memory_stats() or {}
    return st.get("peak_bytes_in_use", 0) / 1024 ** 3


def _bench(fn, reps=2, warmup=1):
    import jax
    jax.block_until_ready(fn())
    for _ in range(warmup):
        jax.block_until_ready(fn())
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        jax.block_until_ready(fn())
        ts.append((time.perf_counter() - t) * 1e3)
    return min(ts)


def _smap(f, mesh, in_specs, out_specs):
    from jax.experimental.shard_map import shard_map
    for kw in ({"check_vma": False}, {"check_rep": False}, {}):
        try:
            return shard_map(f, mesh=mesh, in_specs=in_specs, out_specs=out_specs, **kw)
        except TypeError:
            continue
    raise RuntimeError("shard_map 不接受任何已知的 check_* 参数名")


def _cleanup(drop_kernels=False):
    import gc
    if drop_kernels:
        try:
            _splash_kernel.cache_clear()
        except Exception:
            pass
    gc.collect()


def _grid_for(n):
    h = max(d for d in range(1, int(n ** 0.5) + 1) if n % d == 0)
    return h, n // h


def build_pack(coarse, real, jnp):
    B = sum(coarse)
    seg_self = np.full(B, -1, np.int32)
    seg_cross = np.empty(B, np.int32)
    mod_index = np.empty(B, np.int32)
    rows = np.zeros(B, np.int32)
    cols = np.zeros(B, np.int32)
    loss_mask = np.zeros(B, np.float32)
    off = 0
    for i, (c, r) in enumerate(zip(coarse, real)):
        seg_cross[off:off + c] = i
        mod_index[off:off + c] = i
        if r:
            seg_self[off:off + r] = i
            loss_mask[off:off + r] = 1.0
            h, w = _grid_for(r)
            rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
            rows[off:off + r] = rr.reshape(-1)
            cols[off:off + r] = cc.reshape(-1)
        off += c
    to = jnp.asarray
    return {"seg_self": to(seg_self), "seg_cross": to(seg_cross),
            "mod_index": to(mod_index), "rows": to(rows), "cols": to(cols),
            "loss_mask": to(loss_mask)}


def make_random_params(key, cfg, dtype):
    import jax
    import jax.numpy as jnp
    D, C, R = cfg.model_channels, cfg.crossattn_dim, cfg.adaln_lora_dim
    F = int(D * cfg.mlp_ratio)
    ks = iter(jax.random.split(key, 8 + cfg.num_blocks * 16))
    w = lambda *s: jax.random.normal(next(ks), s, dtype) * 0.02
    one = lambda n: jnp.ones((n,), dtype)
    blocks = [{
        "self_attn": {"q_proj": w(D, D), "k_proj": w(D, D), "v_proj": w(D, D),
                      "output_proj": w(D, D), "q_norm": one(cfg.head_dim),
                      "k_norm": one(cfg.head_dim)},
        "cross_attn": {"q_proj": w(D, D), "k_proj": w(D, C), "v_proj": w(D, C),
                       "output_proj": w(D, D), "q_norm": one(cfg.head_dim),
                       "k_norm": one(cfg.head_dim)},
        "mlp": {"layer1": w(F, D), "layer2": w(D, F)},
        "adaln_self_1": w(R, D), "adaln_self_2": w(3 * D, R),
        "adaln_cross_1": w(R, D), "adaln_cross_2": w(3 * D, R),
        "adaln_mlp_1": w(R, D), "adaln_mlp_2": w(3 * D, R),
    } for _ in range(cfg.num_blocks)]
    return {"x_embedder": w(D, cfg.in_dim), "t_embedder_1": w(D, D),
            "t_embedder_2": w(3 * D, D), "t_embedding_norm": one(D),
            "final_adaln_1": w(R, D), "final_adaln_2": w(2 * D, R),
            "final_linear": w(cfg.out_dim, D), "blocks": blocks}


def run_variant(*, mod_broadcast=False, no_cross=False, bwd_block=1024,
                no_adaln=False, fwd_only=False, remat="full"):
    """按变量表跑一次 8 卡训练步（或纯前向）。返回步时 ms。"""
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    _cleanup(drop_kernels=True)

    devs = jax.devices()
    mesh = Mesh(np.array(devs).reshape(len(devs)), ("d",))
    cfg, dtype = AnimaConfig(), jnp.bfloat16
    pack = build_pack(COARSE, REAL, jnp)
    if mod_broadcast:
        # gather -> 广播（数值不对，只测显存）
        pack["mod_index"] = jnp.zeros_like(pack["mod_index"])

    # 反向块大小是本轮的变量之一 -> 临时改掉 attention.py 的偏好序
    global BWD_BLOCK_PREF
    saved = BWD_BLOCK_PREF
    BWD_BLOCK_PREF = tuple(b for b in saved if b <= bwd_block) or (128,)
    try:
        txt = [T_TXT] * len(COARSE)
        self_attn = make_splash_attn(COARSE, COARSE, cfg.num_heads, cfg.head_dim)
        cross_attn = make_splash_attn(COARSE, txt, cfg.num_heads, cfg.head_dim)
    finally:
        BWD_BLOCK_PREF = saved

    params = jax.jit(lambda: make_random_params(jax.random.PRNGKey(0), cfg, dtype),
                     out_shardings=NamedSharding(mesh, P()))()
    lora = jax.jit(lambda: init_lora(jax.random.PRNGKey(1), cfg, LORA_RANK, dtype=dtype),
                   out_shardings=NamedSharding(mesh, P()))()
    B, G = sum(COARSE), len(COARSE)
    ks = jax.random.split(jax.random.PRNGKey(2), 4)
    n = len(devs)
    spec = {k: P("d") for k in ("tok", "t", "ctx", "target")}
    batch = jax.jit(lambda: {
        "tok": jax.random.normal(ks[0], (n, B, cfg.in_dim), dtype),
        "t": jax.random.uniform(ks[1], (n, G), jnp.float32),
        "ctx": jax.random.normal(ks[2], (n, G * T_TXT, cfg.crossattn_dim), dtype),
        "target": jax.random.normal(ks[3], (n, B, cfg.out_dim), dtype),
    }, out_shardings={k: NamedSharding(mesh, v) for k, v in spec.items()})()

    zero_cross = lambda q, k, v: jnp.zeros_like(q)

    def local(lo, pa, b):
        sf = bind_segments(self_attn, pack["seg_self"], pack["seg_self"])
        cf = zero_cross if no_cross else bind_segments(
            cross_attn, pack["seg_cross"], jnp.asarray(segment_ids([T_TXT] * G)))
        mi = pack["mod_index"]
        if no_adaln:
            # 把调制关掉：shift=0 scale=0 gate=1 —— 等价于去掉 AdaLN 的全部 gather
            emb_zero = jnp.zeros((G, cfg.model_channels), dtype)
            out = forward_packed_noadaln(pa, cfg, b["tok"], b["t"], b["ctx"],
                                         pack["rows"], pack["cols"], mi, sf, cf,
                                         loras=lo, remat=remat)
        else:
            out = forward_packed(pa, cfg, b["tok"], b["t"], b["ctx"],
                                 pack["rows"], pack["cols"], mi, sf, cf,
                                 loras=lo, remat=remat)
        se = jnp.mean((out.astype(jnp.float32) - b["target"].astype(jnp.float32)) ** 2, -1)
        m = pack["loss_mask"]
        return jnp.sum(se * m) / jnp.maximum(jnp.sum(m), 1.0)

    def per_shard(lo, pa, b):
        return local(lo, pa, {k: v[0] for k, v in b.items()})[None]

    f = _smap(per_shard, mesh, (P(), P(), spec), P("d"))
    obj = lambda lo, pa, ba: jnp.mean(f(lo, pa, ba))
    fn = jax.jit(obj) if fwd_only else jax.jit(jax.grad(obj, argnums=0))
    return _bench(lambda: fn(lora, params, batch))


def forward_packed_noadaln(params, cfg, tokens, timesteps, ctx, rows, cols,
                           mod_index, self_fn, cross_fn, loras=None, remat="full"):
    """把 AdaLN 调制整个去掉的对照版（shift=0/scale=0/gate=1）。

    只为归因显存：如果这一版的需求塌下来，说明 1.01MB/token 主要来自
    `jnp.take(h, mod_index)` 那三次逐 token gather。数值当然是错的。
    """
    import jax
    import jax.numpy as jnp
    net = params
    x = tokens @ net["x_embedder"].T.astype(tokens.dtype)
    cos, sin = packed_rope_cos_sin(rows, cols, cfg.head_dim,
                                   cfg.rope_h_ratio, cfg.rope_w_ratio)
    _, wrap = resolve_remat(remat)

    def block(x, p, layer):
        D = cfg.model_channels
        lp = f"blocks.{layer}"
        h = layer_norm(x, cfg.eps_ln)
        q, k, v = attn_qkv(h, None, p["self_attn"], cfg, loras,
                           f"{lp}.self_attn", cos, sin)
        x = x + dense(self_fn(q, k, v).reshape(-1, D), p["self_attn"]["output_proj"],
                      _lora(loras, f"{lp}.self_attn.output_proj"))
        h = layer_norm(x, cfg.eps_ln)
        q, k, v = attn_qkv(h, ctx, p["cross_attn"], cfg, loras, f"{lp}.cross_attn")
        x = x + dense(cross_fn(q, k, v).reshape(-1, D), p["cross_attn"]["output_proj"],
                      _lora(loras, f"{lp}.cross_attn.output_proj"))
        h = layer_norm(x, cfg.eps_ln)
        h = gelu_exact(dense(h, p["mlp"]["layer1"], _lora(loras, f"{lp}.mlp.layer1")))
        return x + dense(h, p["mlp"]["layer2"], _lora(loras, f"{lp}.mlp.layer2"))

    for i in range(cfg.num_blocks):
        def one(carry, p, _i=i):
            return block(carry, p, _i)
        x = wrap(one, i)(x, params["blocks"][i])
    out = dense(layer_norm(x, cfg.eps_ln), net["final_linear"])
    return output_tokens_to_patch_tokens(out, cfg)


VARIANTS = [
    ("V0 基线（复现上一跑的 20.60G）", {}),
    ("V1 mod_index 恒 0（gather->广播）", {"mod_broadcast": True}),
    ("V2 关 cross-attn", {"no_cross": True}),
    ("V3 反向块 128（而非 1024）", {"bwd_block": 128}),
    ("V4 关 AdaLN 调制（去掉全部逐 token gather）", {"no_adaln": True}),
    ("V5 只前向不求梯度", {"fwd_only": True}),
]


@probe("P0 设备 / HBM")
def p0():
    import jax
    d = jax.devices()
    lim = (d[0].memory_stats() or {}).get("bytes_limit", 0) / 1024 ** 3
    return f"jax {jax.__version__} | {len(d)} x {d[0].device_kind} | HBM limit={lim:.1f}GiB"


@probe("M 显存归因（budget=16384 / remat=full，单变量）")
def m_attr():
    out = []
    for name, kw in VARIANTS:
        try:
            status, gb, ms = mem_required(lambda: run_variant(**kw))
            out.append(f"{name}: {status} {gb:.2f}G"
                       + (f" 步时{ms:.0f}ms" if ms else ""))
        except Exception as e:
            out.append(f"{name}: {type(e).__name__}: {str(e)[:120]}")
        _cleanup(drop_kernels=True)
    return "\n         ".join(out)


def main():
    record("python", "INFO", f"{platform.python_version()} @ {sys.executable}")
    record("bootstrap", "INFO", _BOOT)
    if not p0():
        _flush()
        return 0
    m_attr()
    record("汇总", "INFO", f"总耗时={time.time() - _T0:.0f}s")
    _flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())