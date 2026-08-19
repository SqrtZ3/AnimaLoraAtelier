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

    rows/cols 的前导维是**任意**的：打包布局给 [ΣN]、批布局给 [B, L]，
    返回 [..., 1, head_dim]，正好与 q/k 的 [..., S, H, D] 在 head 维上广播。
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
    half_t = jnp.zeros(rows.shape + (dim_t // 2,), jnp.float32)
    half_h = rows.astype(jnp.float32)[..., None] * h_freqs
    half_w = cols.astype(jnp.float32)[..., None] * w_freqs
    emb = jnp.concatenate([half_t, half_h, half_w] * 2, axis=-1)   # [..., head_dim]
    return jnp.cos(emb)[..., None, :], jnp.sin(emb)[..., None, :]


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
    """models/anima_modeling_core.py:1174 的逐算子复刻。

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


def unstack_loras(stacked: Dict[str, Any], num_blocks: int) -> Dict[str, Any]:
    """stack_loras 的逆运算，导出/取证时用（export.py 吃的是扁平键）。"""
    return {f"blocks.{i}.{t}": {s: v[s][i] for s in ("a", "b")}
            for t, v in stacked.items() for i in range(num_blocks)}


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
    """**打包（NaViT）布局**：整个 pack 是一条长序列，靠块对角 mask 隔开各图。

      tokens     [ΣN, in_dim]   已 patchify 的图像 token（按图序拼接，含 padding 段）
      timesteps  [G]            每图一个 t
      ctx        [ΣL, 1024]     各图 caption 的文本特征（按图序拼接）
      rows/cols  [ΣN]           每个 image token 的 (row, col)，供 RoPE
      mod_index  [ΣN]           token -> 图 的索引
      返回       [ΣN, out_dim]
    """
    return _forward_core(params, cfg, tokens, timesteps, ctx, rows, cols,
                         lambda h: jnp.take(h, mod_index, axis=0),
                         self_attn_fn, cross_attn_fn, loras, remat)


def forward_ragged(params: PyTree, cfg: AnimaConfig,
                   tokens: jnp.ndarray, timesteps: jnp.ndarray, ctx: jnp.ndarray,
                   rows: jnp.ndarray, cols: jnp.ndarray,
                   self_attn_fn, cross_attn_fn,
                   loras=None, remat="full") -> jnp.ndarray:
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
                         self_attn_fn, cross_attn_fn, loras, remat)


def _forward_core(params: PyTree, cfg: AnimaConfig,
                  tokens: jnp.ndarray, timesteps: jnp.ndarray, ctx: jnp.ndarray,
                  rows: jnp.ndarray, cols: jnp.ndarray, mod_bcast,
                  self_attn_fn, cross_attn_fn,
                  loras=None, remat="full") -> jnp.ndarray:
    """models/anima_modeling_core.py:1774 的复刻，**布局无关**。

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
    policy, wrap = resolve_remat(remat)
    if isinstance(params["blocks"], (list, tuple)):
        # 展开路径：逐块单独编译。保留它是因为对拍脚本按块比对需要，
        # **但训练不要用**——见 stack_blocks 的注释（显存 1.01 MB/token）。
        for i in range(cfg.num_blocks):
            def one(carry, p, _i=i):
                return block_forward(carry, p, cfg, emb, adaln_lora, mod_bcast, ctx,
                                     cos, sin, self_attn_fn, cross_attn_fn, loras, _i)
            x = wrap(one, i)(x, params["blocks"][i])
    else:
        # scan 路径（训练用）：loras 的键必须是块内相对的（stack_loras 的产物）
        def run_block(carry, p, lo):
            return block_forward(carry, p, cfg, emb, adaln_lora, mod_bcast, ctx,
                                 cos, sin, self_attn_fn, cross_attn_fn, lo, None)

        xs = (params["blocks"], loras)
        if remat == "every2":
            # 隔块 remat 在 scan 下要**成对**做：把 [L,...] 重排成 [L/2, 2, ...]，
            # 一次循环走两块，第一块不 remat、第二块 remat。
            # （wrap 的 `i % 2` 判据在 scan 里用不了 —— 循环变量是 tracer。）
            if cfg.num_blocks % 2:
                raise ValueError(f"remat='every2' 在 scan 路径下需要偶数块，"
                                 f"当前 {cfg.num_blocks}；改用 full/dots/none")
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

    # FinalLayer：2 chunk，且只取 adaln_lora 的前 2D（models/...:950）
    h = silu(emb)
    h = dense(h, net["final_adaln_1"])
    h = dense(h, net["final_adaln_2"]) + adaln_lora[:, : 2 * cfg.model_channels]
    shift, scale = jnp.split(mod_bcast(h), 2, axis=-1)
    x = layer_norm(x, cfg.eps_ln) * (1 + scale) + shift
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

# ==========================================================================
# ④ job 主体（_job_body.py）
# ==========================================================================

"""Kaggle TPU v5e-8 · **布局裁决**：NaViT 序列打包 vs 量化分桶（批维）。

## 为什么要这一跑

上一轮（anima-navit-probe）的结论是：打包 + 块对角 splash 能跑，但
  * 显存 ≈ 4.0GB + **1.01 MB/token**，必须靠 `lax.scan` 压住；
  * 压住之后 **`full` 成了唯一可用 remat 档**（every2/dots/none 全 OOM）；
  * 吞吐 25.4k 真tok/s，**仍未胜过**第一棒的分桶批处理 29.8k。

显存归因（anima-mem-probe 单变量）指向 AdaLN 调制：它不依赖 x，28 块展开后
XLA 把 28 组 [ΣN, 3D] 同时留着。但那是**打包布局**的产物 —— 打包路径必须把
逐图的 shift/scale/gate `gather` 成逐 token 的 [ΣN, 3D]（真物化），而批布局
只需要 [G, 1, 3D] 广播（XLA 融合进消费者，不物化）。分桶路径实测 0.02 MB/token，
正是这个差别。

于是本轮问的是一个**布局**问题，不是实现问题：

> 段长量化到 Q 之后，块对角注意力与批维注意力在算力上完全等价
> （都是 Σ L_i²）。那么"打包"到底还买到了什么？

**任意宽高比不是打包独有的**：桶按**量化后的 token 总数**分，不按 (h,w) 分；
每图保留自己的 (h_i, w_i)，只要 h_i*w_i <= L 即可同批，RoPE 逐 token 查
rows/cols、不依赖矩形网格。所以用户要的"任意分辨率哲学"两条路都满足。

## 本轮要回答的

R1 **数值**：`forward_ragged` ≡ `forward_packed`（真机真内核，同一批图）。
   过了这条，分桶路线就继承了打包路线全部已验证的数值口径（八道闸门）。
R2 **段内填充隔离**：分桶路径的 segment_ids 是否真把量化填充挡住（判据：动填充
   区的 k，真 token 输出必须逐 bit 不变）。打包路径不加它时实测污染 max_abs=5.4，
   静默 —— 分桶路径同样的坑。
R3 LoRA 接线自检（没有它，后面测的可能是个空转的模型）。
R4 **主判据**：两条路线的步时 / 真tok/s / 有效 MFU / 峰值，remat 四档全扫。
R5 **分桶路径还需不需要 scan**：同配置下 scan vs 展开。若展开可用，
   `lax.scan` 那条"未 remat 的激活按 28 次迭代堆叠"的约束就不存在了，
   `every2` 的 15% 才拿得回来。

## 不在本轮范围

优化器/数据/checkpoint 不接（随机权重 + MSE 到随机 target）；本轮只裁决布局。
形状一致则性能一致，故不上传 3.91GB 权重。
"""

import json
import os
import platform
import sys
import tempfile
import time
import traceback

import numpy as np

RESULTS: list = []
_T0 = time.time()
_BOOT = globals().get("_BOOT", "未经 build_job.py 拼接，无 bootstrap")
OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else tempfile.gettempdir()

T_TXT = 512          # 每图文本槽长度 —— **真实口径**（trainer/config.py 的 512，
                     # navit_text_trim_padding 默认 False，pad 照常参与 cross-attn）。
                     # 注意上一轮 anima-navit-probe 用的是 256，故本轮打包侧的
                     # 绝对数字与那一轮不可直接比；本轮内部 A/B 才是同口径的。
LORA_RANK = 32
V5E_PEAK_TFLOPS = 197.0      # 单 chip bf16 峰值；8 卡 = 1576

# ── 打包路线的布局（沿用上一轮，段长 = 真实 token 数量化到 1024）────────────────
# 每个元组：(粗粒度段长 -> 编译期 mask，真实 token 数 -> 运行时 segment_ids)
PACKED_LAYOUTS = {
    8192:  ([4096, 3072, 1024], [4096, 2304, 0]),
    16384: ([10240, 4096, 2048], [10080, 4096, 0]),
    32768: ([10240, 9216, 4096, 4096, 3072, 2048], [10080, 9216, 4096, 4096, 2304, 0]),
}

# ── 分桶路线的配置：(L 桶长, 每卡图数 G, 每图真实 token 数)────────────────────
# L 取自同样的量化格点，real 取自真实 ARB 桶：
#   3952 = jacknife-anima2 的典型值（token 3952-4160，近似均匀）
#   10080 = 96x105、9216 = 96x96、4096 = 64x64、2304 = 48x48（modare-anima3 侧）
RAGGED_CONFIGS = [
    (4096, 1, 3952), (4096, 2, 3952), (4096, 4, 3952), (4096, 8, 3952),
    (3072, 4, 2304), (3072, 8, 2304),
    (10240, 1, 10080), (10240, 2, 10080), (10240, 3, 10080),
]
REMATS = ("full", "every2", "dots", "none")


def _flush():
    try:
        with open(os.path.join(OUT_DIR, "anima_ragged.json"), "w", encoding="utf-8") as f:
            json.dump(RESULTS, f, ensure_ascii=False, indent=2)
        with open(os.path.join(OUT_DIR, "anima_ragged_report.txt"), "w", encoding="utf-8") as f:
            for r in RESULTS:
                f.write(f"[{r['status']:^6}] {r['name']}"
                        + (f" - {r['detail']}" if r["detail"] else "") + "\n")
            f.write(f"\n（截至 {time.time() - _T0:.0f}s 的快照）\n")
    except OSError:
        pass


def record(name, status, detail=""):
    RESULTS.append({"name": name, "status": status, "detail": detail,
                    "t_elapsed_s": round(time.time() - _T0, 1)})
    print(f"[{status:^6}] {name}" + (f" - {detail}" if detail else ""), flush=True)
    _flush()


class _Skip(Exception):
    pass


def probe(name):
    def deco(fn):
        def run(*a, **kw):
            try:
                record(name, "OK", fn(*a, **kw) or "")
                return True
            except _Skip as s:
                record(name, "SKIP", str(s))
            except Exception as e:
                tb = traceback.format_exc(limit=3).strip().splitlines()[-1]
                record(name, "FAIL", f"{type(e).__name__}: {e} | {tb}")
            return False
        return run
    return deco


# ── 测量工具 ──────────────────────────────────────────────────────────────────
def _bench(fn, reps=3, warmup=1):
    """block_until_ready 强制同步。不消费输出 -> 惰性执行根本没派发（测出假值）；
    不 warmup -> 首步编译时间混进步时。两个坑前几轮都踩过。"""
    import jax
    t0 = time.time()
    jax.block_until_ready(fn())
    first = time.time() - t0
    for _ in range(warmup):
        jax.block_until_ready(fn())
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        jax.block_until_ready(fn())
        ts.append((time.perf_counter() - t) * 1e3)
    ts.sort()
    return ts[len(ts) // 2], first


_PEAK_RESETTABLE = None


def _hbm(peak=False):
    import jax
    st = jax.devices()[0].memory_stats() or {}
    return st.get("peak_bytes_in_use" if peak else "bytes_in_use", 0) / 1024 ** 3


def _reset_peak():
    """尝试清零峰值统计；**返回是否成功**。失败时读数只是历史包络，报告里要标注
    （上一轮就出过"三个配置同报 3.9GiB"的废数）。"""
    global _PEAK_RESETTABLE
    import jax
    d = jax.devices()[0]
    for name in ("reset_memory_stats", "clear_memory_stats"):
        fn = getattr(d, name, None)
        if fn is not None:
            try:
                fn()
                _PEAK_RESETTABLE = True
                return True
            except Exception:
                pass
    _PEAK_RESETTABLE = False
    return False


def _smap(f, mesh, in_specs, out_specs):
    """shard_map 包装。**纯 DP 也必须用它**：splash 是 Pallas/Mosaic 内核，
    XLA 的 GSPMD 自动分区处理不了（真机报 `Mosaic kernels cannot be
    automatically partitioned.`）。参数名 0.8 前叫 check_rep、0.11 起叫 check_vma。"""
    from jax.experimental.shard_map import shard_map
    for kw in ({"check_vma": False}, {"check_rep": False}, {}):
        try:
            return shard_map(f, mesh=mesh, in_specs=in_specs, out_specs=out_specs, **kw)
        except TypeError:
            continue
    raise RuntimeError("shard_map 不接受任何已知的 check_* 参数名")


def _cleanup(drop_kernels: bool = False):
    """释放显存。**不逐个 delete `jax.live_arrays()`** —— splash kernel 的 MaskInfo
    是 jax 数组、活在 lru_cache 的可调用对象里，delete-all 会让缓存变空壳，
    表现为"第一格成功、同布局第二格必炸"（上一轮实测）。只 gc；换布局时清缓存。"""
    import gc
    if drop_kernels:
        for c in (globals().get("_splash_kernel"), globals().get("_bucket_kernel")):
            try:
                c.cache_clear()
            except Exception:
                pass
    gc.collect()


def _grid_for(n):
    """把 n 拆成尽量方的 h x w（h*w **恰好** = n）。真机第一跑栽过：
    round(sqrt(512))^2=529≠512 直接广播报错。token 数不是完全平方数是常态。"""
    h = max(d for d in range(1, int(n ** 0.5) + 1) if n % d == 0)
    return h, n // h


# ── 有效算力口径 ──────────────────────────────────────────────────────────────
def fwd_flops(cfg, n_pad, attn_pairs, cross_pairs):
    """一次**前向**的 FLOPs（单卡）。

      n_pad        本卡参与线性层的 token 数（含填充 —— 填充一样要过 MLP）
      attn_pairs   Σ L_i²        自注意力的 (q,kv) 对数
      cross_pairs  Σ L_i * T     cross-attn 的对数

    注意力每对 (q,kv) 的代价是 4*D（QKᵀ 的 2*D + AV 的 2*D）。
    AdaLN 只在 G 行上算，逐 token 项里不计。
    """
    D, C = cfg.model_channels, cfg.crossattn_dim
    F = int(D * cfg.mlp_ratio)
    p_lin = 4 * D * D + 2 * D * D + 2 * D * C + 2 * D * F
    return cfg.num_blocks * (2 * p_lin * n_pad + 4 * D * (attn_pairs + cross_pairs))


def useful_mfu(cfg, ms, n_dev, n_pad, attn_pairs, cross_pairs):
    """**有效 MFU**：只算"理论必需"的算力，不含 remat 的重算。

    冻结底模只训 LoRA -> 反向没有 wgrad -> 理论下界 = 前向(1) + 激活梯度(1) = 2x 前向。
    这样 remat 档位的代价会直接体现为有效 MFU 的下降，而不是被口径吸收掉 ——
    比"含重算的 MFU"更能回答"这一档值不值"。
    """
    total = 2.0 * fwd_flops(cfg, n_pad, attn_pairs, cross_pairs) * n_dev
    return total / (ms / 1e3) / (V5E_PEAK_TFLOPS * 1e12 * n_dev)


# ── 权重 ──────────────────────────────────────────────────────────────────────
def make_random_params(key, cfg, dtype, stack=True):
    """按真实形状随机造权重（性能只取决于形状）。**直接 bf16 生成**：
    前几轮两次 OOM 都是先在单卡物化 fp32 再转换造成的。

    `stack=False` 保留 list-of-dict（展开路径要），`True` 堆成 [L,...]（scan 路径要）。
    """
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
    p = {"x_embedder": w(D, cfg.in_dim), "t_embedder_1": w(D, D),
         "t_embedder_2": w(3 * D, D), "t_embedding_norm": one(D),
         "final_adaln_1": w(R, D), "final_adaln_2": w(2 * D, R),
         "final_linear": w(cfg.out_dim, D), "blocks": blocks}
    return stack_blocks(p) if stack else p


# ── 打包路线 ──────────────────────────────────────────────────────────────────
def build_pack(coarse, real, jnp):
    """由 (粗粒度段长, 真实 token 数) 造出一个 pack 的全部运行时数组。

    seg_self  段内填充统一给 -1（彼此可见，行不空）
    seg_cross 填充**沿用宿主段号**（文本侧没有配对填充段，独立段号 -> 整行全 0
              -> softmax 分母为 0，splash 的 SegmentIds 文档对此有明确警告）
    """
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
    to = lambda a: jnp.asarray(a)
    return {"seg_self": to(seg_self), "seg_cross": to(seg_cross),
            "mod_index": to(mod_index), "rows": to(rows), "cols": to(cols),
            "loss_mask": to(loss_mask)}


def build_packed_loss(cfg, coarse, pack, remat, interpret=False):
    """打包路线的 loss_fn(lora, params, batch)：冻结底模、只对 LoRA 求梯度。"""
    import jax.numpy as jnp
    txt = [T_TXT] * len(coarse)
    self_fn = bind_segments(
        make_splash_attn(coarse, coarse, cfg.num_heads, cfg.head_dim,
                         interpret=interpret), pack["seg_self"], pack["seg_self"])
    cross_fn = bind_segments(
        make_splash_attn(coarse, txt, cfg.num_heads, cfg.head_dim,
                         interpret=interpret),
        pack["seg_cross"], jnp.asarray(segment_ids(txt)))

    def loss_fn(lora, params, batch):
        out = forward_packed(params, cfg, batch["tok"], batch["t"], batch["ctx"],
                             pack["rows"], pack["cols"], pack["mod_index"],
                             self_fn, cross_fn, loras=lora, remat=remat)
        se = jnp.mean((out.astype(jnp.float32) - batch["target"].astype(jnp.float32)) ** 2,
                      axis=-1)
        m = pack["loss_mask"]        # padding token 的输出是垃圾，计进 loss 会污染梯度
        return jnp.sum(se * m) / jnp.maximum(jnp.sum(m), 1.0)

    return loss_fn


# ── 分桶路线 ──────────────────────────────────────────────────────────────────
def build_bucket(L, G, real, jnp):
    """由 (桶长 L, 每卡图数 G, 每图真实 token 数 real) 造出运行时数组。

    这里让 G 张图**真实 token 数相同**只是为了 A/B 干净；真实训练里同桶各图的
    real 可以各不相同（segment_ids 是 [G, L] 逐图的），(h_i, w_i) 也各不相同。
    """
    seg = bucket_segment_ids([real] * G, L)                # [G, L] 0=真 1=填充
    rows = np.zeros((G, L), np.int32)
    cols = np.zeros((G, L), np.int32)
    loss_mask = np.zeros((G, L), np.float32)
    h, w = _grid_for(real)
    rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    for g in range(G):
        rows[g, :real] = rr.reshape(-1)
        cols[g, :real] = cc.reshape(-1)
        loss_mask[g, :real] = 1.0
    to = lambda a: jnp.asarray(a)
    return {"seg": to(seg), "rows": to(rows), "cols": to(cols),
            "loss_mask": to(loss_mask)}


def build_ragged_loss(cfg, L, buck, remat, interpret=False):
    """分桶路线的 loss_fn。**注意 cross-attn 完全没有 mask** —— 每图看自己那一份
    定长文本槽的全部，这正是相对打包路线省掉的整块复杂度（打包那边是矩形块对角
    splash，anima-mem-probe 归因里占 4.2GB）。"""
    import jax.numpy as jnp
    self_fn = bind_segments(
        make_bucket_attn(L, L, cfg.num_heads, cfg.head_dim, use_segments=True,
                         interpret=interpret), buck["seg"], buck["seg"])
    cross_fn = make_bucket_attn(L, T_TXT, cfg.num_heads, cfg.head_dim,
                                use_segments=False, interpret=interpret)

    def loss_fn(lora, params, batch):
        out = forward_ragged(params, cfg, batch["tok"], batch["t"], batch["ctx"],
                             buck["rows"], buck["cols"],
                             self_fn, cross_fn, loras=lora, remat=remat)
        se = jnp.mean((out.astype(jnp.float32) - batch["target"].astype(jnp.float32)) ** 2,
                      axis=-1)
        m = buck["loss_mask"]
        return jnp.sum(se * m) / jnp.maximum(jnp.sum(m), 1.0)

    return loss_fn


def make_batch(key, cfg, n_dev, tok_shape, ctx_shape, n_img, dtype, mesh):
    """在设备上直接生成，省掉 host->device 搬运（也避免单卡物化 fp32 再转换）。"""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    ks = jax.random.split(key, 4)
    spec = {k: P("d") for k in ("tok", "t", "ctx", "target")}

    def mk():
        return {"tok": jax.random.normal(ks[0], (n_dev, *tok_shape, cfg.in_dim), dtype),
                "t": jax.random.uniform(ks[1], (n_dev, n_img), jnp.float32),
                "ctx": jax.random.normal(ks[2], (n_dev, *ctx_shape, cfg.crossattn_dim), dtype),
                "target": jax.random.normal(ks[3], (n_dev, *tok_shape, cfg.out_dim), dtype)}

    return jax.jit(mk, out_shardings={k: NamedSharding(mesh, v)
                                      for k, v in spec.items()})()


# ── 探测 ──────────────────────────────────────────────────────────────────────
@probe("R0 设备 / HBM")
def r0():
    import jax
    d = jax.devices()
    lim = (d[0].memory_stats() or {}).get("bytes_limit", 0) / 1024 ** 3
    _reset_peak()
    return (f"jax {jax.__version__} | {len(d)} x {d[0].device_kind} | "
            f"单设备 HBM limit={lim:.1f}GiB | peak 可清零={_PEAK_RESETTABLE}")


@probe("R1 forward_ragged ≡ forward_packed（真内核，同一批图）")
def r1():
    """**最关键的一道闸门**：过了它，分桶路线就继承打包路线全部已验证的数值口径。

    构造两张图（真实 token 数不同 -> 量化到同一个桶长 L），分别走两条路线，
    只在**真 token** 上比。两条路径共用 `_forward_core`，理论上差别只有布局与
    注意力后端，所以残差应当落在 bf16 噪声量级（~1e-3 相对）。
    """
    import jax
    import jax.numpy as jnp
    _cleanup(drop_kernels=True)
    cfg = AnimaConfig(num_blocks=4)          # 4 块足够暴露布局错误，省编译时间
    L, reals = 1024, [900, 768]
    G = len(reals)
    dtype = jnp.bfloat16
    params = make_random_params(jax.random.PRNGKey(0), cfg, dtype, stack=True)
    lora = stack_loras(init_lora(jax.random.PRNGKey(1), cfg, LORA_RANK, dtype=dtype),
                       cfg.num_blocks)
    # B 是零初始化 -> LoRA 净增量为 0，两条路线都一样；这里把 B 填成非零，
    # 好让 LoRA 支路真正参与对拍（否则等于没测 LoRA）
    lora = {k: {"a": v["a"], "b": jax.random.normal(
        jax.random.PRNGKey(7), v["b"].shape, dtype) * 0.02} for k, v in lora.items()}

    ks = jax.random.split(jax.random.PRNGKey(2), 3)
    tok_b = jax.random.normal(ks[0], (G, L, cfg.in_dim), dtype)
    t = jax.random.uniform(ks[1], (G,), jnp.float32)
    ctx_b = jax.random.normal(ks[2], (G, T_TXT, cfg.crossattn_dim), dtype)
    # 填充位置的输入置 0（真实数据里就是零填充），保证两条路线输入完全同源
    keep = (np.arange(L)[None, :] < np.asarray(reals)[:, None])
    tok_b = tok_b * jnp.asarray(keep, dtype)[..., None]

    buck = build_bucket(L, G, reals[0], jnp)
    buck["seg"] = jnp.asarray(bucket_segment_ids(reals, L))
    rows_b = np.zeros((G, L), np.int32)
    cols_b = np.zeros((G, L), np.int32)
    for g, r in enumerate(reals):
        h, w = _grid_for(r)
        rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        rows_b[g, :r], cols_b[g, :r] = rr.reshape(-1), cc.reshape(-1)
    buck["rows"], buck["cols"] = jnp.asarray(rows_b), jnp.asarray(cols_b)

    self_r = bind_segments(make_bucket_attn(L, L, cfg.num_heads, cfg.head_dim,
                                            use_segments=True), buck["seg"], buck["seg"])
    cross_r = make_bucket_attn(L, T_TXT, cfg.num_heads, cfg.head_dim,
                               use_segments=False)
    out_r = jax.jit(lambda: forward_ragged(params, cfg, tok_b, t, ctx_b,
                                           buck["rows"], buck["cols"],
                                           self_r, cross_r, loras=lora, remat="full"))()

    # 打包侧：同样两张图，段长都是 L（粗粒度），真实 token 数 = reals
    coarse = [L] * G
    pack = build_pack(coarse, reals, jnp)
    pack["rows"] = jnp.asarray(rows_b.reshape(-1))
    pack["cols"] = jnp.asarray(cols_b.reshape(-1))
    tok_p = tok_b.reshape(G * L, cfg.in_dim)
    ctx_p = ctx_b.reshape(G * T_TXT, cfg.crossattn_dim)
    self_p = bind_segments(make_splash_attn(coarse, coarse, cfg.num_heads, cfg.head_dim),
                           pack["seg_self"], pack["seg_self"])
    cross_p = bind_segments(
        make_splash_attn(coarse, [T_TXT] * G, cfg.num_heads, cfg.head_dim),
        pack["seg_cross"], jnp.asarray(segment_ids([T_TXT] * G)))
    out_p = jax.jit(lambda: forward_packed(params, cfg, tok_p, t, ctx_p,
                                           pack["rows"], pack["cols"], pack["mod_index"],
                                           self_p, cross_p, loras=lora, remat="full"))()

    a = np.asarray(out_r, np.float32).reshape(G * L, -1)
    b = np.asarray(out_p, np.float32)
    m = keep.reshape(-1)
    num = np.abs(a[m] - b[m]).max()
    den = max(float(np.abs(b[m]).max()), 1e-9)
    _cleanup(drop_kernels=True)
    if num / den > 5e-2:
        raise RuntimeError(f"两条路线不等价：rel={num / den:.3e}（bf16 噪声应 <5e-2）")
    return f"真 token 上 max_abs={num:.3e} rel={num / den:.3e}（bf16 噪声量级）"


@probe("R2 分桶路径的段内填充隔离（segment_ids 是否真生效）")
def r2():
    """判据是**行为**而不是数值接近：动填充区的 k，真 token 输出必须逐 bit 不变。
    不加 segment_ids 时打包路径实测污染 max_abs=5.4，且**静默**。"""
    import jax
    import jax.numpy as jnp
    _cleanup(drop_kernels=True)
    L, G, real, H, Dh = 1024, 2, 900, 16, 128
    r = lambda k: jax.random.normal(jax.random.PRNGKey(k), (G, L, H, Dh),
                                    jnp.float32).astype(jnp.bfloat16)
    q, k, v = r(0), r(1), r(2)
    seg = jnp.asarray(bucket_segment_ids([real] * G, L))
    attn = bind_segments(make_bucket_attn(L, L, H, Dh, use_segments=True), seg, seg)
    f = jax.jit(attn)
    kp = k.at[:, real:].set(r(9)[:, real:])
    leak = float(jnp.abs(f(q, k, v)[:, :real].astype(jnp.float32)
                         - f(q, kp, v)[:, :real].astype(jnp.float32)).max())
    # 反证：关掉 segment_ids 就应该被污染（证明这个测试本身有分辨力）
    bare = jax.jit(make_bucket_attn(L, L, H, Dh, use_segments=False))
    contam = float(jnp.abs(bare(q, k, v)[:, :real].astype(jnp.float32)
                           - bare(q, kp, v)[:, :real].astype(jnp.float32)).max())
    _cleanup(drop_kernels=True)
    if leak != 0.0:
        raise RuntimeError(f"填充泄漏进真 token（max_abs={leak:.3e}）—— 训练会静默学错")
    if contam == 0.0:
        raise RuntimeError("关掉 segment_ids 也没有污染 -> 这个测试没有分辨力，判据无效")
    return f"泄漏={leak:.1e}（必须 0）；关掉 segment_ids 的对照污染={contam:.3e}（证明判据有效）"


@probe("R3 LoRA 接线自检")
def r3():
    """jax.checkpoint 会把入参 trace 成 tracer，层号若作为入参，f"blocks.{i}" 拼出
    垃圾键 -> LoRA 静默失效、梯度恒 0，且反向可能被 DCE 掉使步时**假性变快**。
    判据：grad_b 非零（真接进图）+ grad_a 恒 0（B 零初始化 -> step-0 中立）。"""
    import jax
    import jax.numpy as jnp
    _cleanup(drop_kernels=True)
    cfg = AnimaConfig(num_blocks=2)
    L, G, real = 1024, 2, 900
    buck = build_bucket(L, G, real, jnp)
    params = make_random_params(jax.random.PRNGKey(0), cfg, jnp.bfloat16)
    lora = stack_loras(init_lora(jax.random.PRNGKey(1), cfg, LORA_RANK,
                                 dtype=jnp.bfloat16), cfg.num_blocks)
    ks = jax.random.split(jax.random.PRNGKey(2), 4)
    batch = {"tok": jax.random.normal(ks[0], (G, L, cfg.in_dim), jnp.bfloat16),
             "t": jax.random.uniform(ks[1], (G,), jnp.float32),
             "ctx": jax.random.normal(ks[2], (G, T_TXT, cfg.crossattn_dim), jnp.bfloat16),
             "target": jax.random.normal(ks[3], (G, L, cfg.out_dim), jnp.bfloat16)}
    g = jax.jit(jax.grad(build_ragged_loss(cfg, L, buck, "full"), argnums=0))(
        lora, params, batch)
    ga = sum(float(jnp.abs(v["a"]).sum()) for v in g.values())
    gb = sum(float(jnp.abs(v["b"]).sum()) for v in g.values())
    nz = sum(1 for v in g.values() if float(jnp.abs(v["b"]).max()) > 0)
    _cleanup(drop_kernels=True)
    if gb <= 0:
        raise RuntimeError(f"grad_b 全 0 —— LoRA 没接进图（{len(g)} 条）")
    if ga != 0:
        raise RuntimeError(f"grad_a={ga} 非 0 —— B 不是零初始化，step-0 不中立")
    return f"sum|grad_b|={gb:.3e}，非零 target {nz}/{len(g)}；sum|grad_a|=0（step-0 中立）"


# ── 主判据 ────────────────────────────────────────────────────────────────────
def _run_step(route, spec, remat, stack=True):
    """跑一次 8 卡 DP 训练步（前向 + LoRA 反向 + 跨卡 all-reduce），返回实测字典。

    route="packed" -> spec=budget；route="ragged" -> spec=(L, G, real)
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    _cleanup()
    _reset_peak()
    devs = jax.devices()
    if len(devs) < 8:
        raise _Skip(f"只有 {len(devs)} 个设备")
    n_dev = 8
    mesh = Mesh(np.array(devs[:n_dev]).reshape(n_dev), ("d",))
    cfg, dtype = AnimaConfig(), jnp.bfloat16

    if route == "packed":
        coarse, real = PACKED_LAYOUTS[spec]
        pack = build_pack(coarse, real, jnp)
        local_builder = lambda: build_packed_loss(cfg, coarse, pack, remat)
        tok_shape, ctx_shape = (sum(coarse),), (len(coarse) * T_TXT,)
        n_img = len(coarse)
        n_pad, tok_real = sum(coarse), sum(real)
        attn_pairs = sum(c * c for c in coarse)
        cross_pairs = sum(c * T_TXT for c in coarse)
        tag = f"packed/{spec}"
    else:
        L, G, real = spec
        buck = build_bucket(L, G, real, jnp)
        local_builder = lambda: build_ragged_loss(cfg, L, buck, remat)
        tok_shape, ctx_shape = (G, L), (G, T_TXT)
        n_img = G
        n_pad, tok_real = G * L, G * real
        attn_pairs, cross_pairs = G * L * L, G * L * T_TXT
        tag = f"ragged/L{L}xG{G}"

    params = jax.jit(lambda: make_random_params(jax.random.PRNGKey(0), cfg, dtype, stack),
                     out_shardings=NamedSharding(mesh, P()))()
    flat = init_lora(jax.random.PRNGKey(1), cfg, LORA_RANK, dtype=dtype)
    lora = jax.jit(lambda: stack_loras(flat, cfg.num_blocks) if stack else flat,
                   out_shardings=NamedSharding(mesh, P()))()
    batch = make_batch(jax.random.PRNGKey(2), cfg, n_dev, tok_shape, ctx_shape,
                       n_img, dtype, mesh)
    local = local_builder()

    def per_shard(lo, pa, ba):
        b = {k: v[0] for k, v in ba.items()}      # shard_map 给每卡 [1,...] 的切片
        return local(lo, pa, b)[None]

    dspec = {k: P("d") for k in ("tok", "t", "ctx", "target")}
    f = _smap(per_shard, mesh, (P(), P(), dspec), P("d"))
    # lora/params 是复制的(in_spec P()) -> 其余切在 shard_map 转置时自动 psum，
    # 即 LoRA 梯度的 8 卡 all-reduce 已含在实测步时里。
    grad = jax.jit(jax.grad(lambda lo, pa, ba: jnp.mean(f(lo, pa, ba)), argnums=0))
    ms, first = _bench(lambda: grad(lora, params, batch))
    mfu = useful_mfu(cfg, ms, n_dev, n_pad, attn_pairs, cross_pairs)
    return {"tag": tag, "ms": ms, "first": first,
            "tok_real": n_dev * tok_real, "tok_pad": n_dev * n_pad,
            "fill": tok_real / n_pad,
            "tok_s": n_dev * tok_real / (ms / 1e3),
            "mfu": mfu, "peak": _hbm(peak=True)}


def _fmt(r):
    return (f"{r['ms']:.0f}ms {r['tok_s'] / 1e3:.1f}k真tok/s "
            f"有效MFU {r['mfu']:.1%} 填充率 {r['fill']:.1%} "
            f"峰值{r['peak']:.1f}GiB 首调{r['first']:.0f}s")


def _sweep(route, specs, remats, stack=True, label=""):
    """**逐 spec 即时 record**：扫描很长（数十格 x 编译），Kaggle 超时/OOM 杀进程时
    只有已 record 的行落了盘。上一轮把整段扫描攒到最后一次性写，风险太大。"""
    out = []
    for spec in specs:
        _cleanup(drop_kernels=True)          # 换布局：旧 kernel 及其 MaskInfo 可回收
        row = []
        for remat in remats:
            try:
                r = _run_step(route, spec, remat, stack)
                row.append(f"{remat}: {_fmt(r)}")
            except Exception as e:
                row.append(f"{remat}: {type(e).__name__}: "
                           f"{str(e)[:200].replace(chr(10), ' ')}")
                _cleanup()
        line = " | ".join(row)
        record(f"  {label}{route}/{spec}", "DATA", line)
        out.append(f"{spec}: {line}")
    return out


@probe("R4a 分桶路线扫描（L x 每卡图数 x remat）")
def r4a():
    """**主判据的一半**。有效 MFU 是 remat 无关的口径（只算理论必需的 2x 前向），
    所以 remat 档位的代价会直接体现为 MFU 下降 —— 比"含重算的 MFU"更能回答
    "这一档值不值"。"""
    _sweep("ragged", RAGGED_CONFIGS, REMATS)
    return ("逐行见上方 DATA 条目"
            + ("" if _PEAK_RESETTABLE else "（**peak 不可清零，读数是历史包络**）"))


@probe("R4b 打包路线对照（同一份代码、同一 T_TXT=512 口径）")
def r4b():
    """上一轮 anima-navit-probe 用的是 T_TXT=256，绝对数字与本轮不可直接比。
    这里在同口径下重测，保证 A/B 是单变量的。"""
    _sweep("packed", sorted(PACKED_LAYOUTS), REMATS)
    return "逐行见上方 DATA 条目"


@probe("R5 分桶路线还需不需要 scan（scan vs 展开）")
def r5():
    """打包路线下 `lax.scan` 是必需的（展开 1.01 MB/token，budget 16384 即 OOM），
    但它的语义代价是：**未被 remat 掉的激活会按 28 次迭代堆叠成 [L,...]**，
    于是 every2/dots 在 scan 下比展开更贵，`full` 成了唯一可用档。

    如果分桶路线的显存本来就不成问题，展开路径就能用，`every2` 的那 15% 才拿得回来。
    """
    picks = [(4096, 2, 3952), (10240, 1, 10080)]
    _sweep("ragged", picks, REMATS, stack=False, label="展开 ")
    return "展开档见上方 DATA 条目；scan 档同配置见 R4a"


def main():
    record("python", "INFO", f"{platform.python_version()} @ {sys.executable}")
    record("bootstrap", "INFO", _BOOT)
    record("口径", "INFO", f"T_TXT={T_TXT} rank={LORA_RANK} "
                           f"有效MFU分母={V5E_PEAK_TFLOPS}TFLOPS/chip x8")
    if not r0():
        _flush()
        return 0
    if not r1():
        record("裁决", "FAIL", "两条路线不等价 -> 后续步时无可比性，停在这里")
        _flush()
        return 0
    if not r2():
        record("裁决", "FAIL", "分桶路径填充隔离失效 -> 训练会静默学错，停在这里")
        _flush()
        return 0
    if not r3():
        record("裁决", "FAIL", "LoRA 未接进图 -> 会测到一个空转的模型，停在这里")
        _flush()
        return 0
    r4a()
    r4b()
    r5()
    ok = sum(1 for r in RESULTS if r["status"] == "OK")
    bad = sum(1 for r in RESULTS if r["status"] == "FAIL")
    record("汇总", "INFO", f"OK={ok} FAIL={bad} 总耗时={time.time() - _T0:.0f}s")
    _flush()
    return 0   # 恒返回 0：单项 FAIL 是数据，不是脚本故障（Kaggle 把非零判成 ERROR）


if __name__ == "__main__":
    sys.exit(main())