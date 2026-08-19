# ！！自动生成，勿手改 —— 改 jax_tpu/anima_jax.py / _preamble.py / _job_body.py
# 后跑 build_job.py 重新生成。
from __future__ import annotations


# ==========================================================================
# ① bootstrap（_preamble.py）—— 必须在 import jax 之前
# ==========================================================================

"""生成脚本的**第一段**：在任何 `import jax` 之前把 libtpu/jax 升上去。

Kaggle 默认镜像是 jax 0.10.2 + libtpu 构建于 2025-06-12，比 Pallas 的版本闸门
（`is_cloud_tpu_older_than`，硬 raise、无环境变量旁路）老约 14 个月，不升级则
所有 splash 探测以同一原因失败。**必须在 import jax 之前**——jax 一旦初始化
后端就换不掉 libtpu。这也是 anima_jax.py 的 import 必须排在本段之后的原因。
"""

import os
import subprocess
import sys
import time

_BOOT = "未开启"
if os.environ.get("ANIMA_TPU_UPGRADE", "1") == "1":
    if "jax" in sys.modules:
        _BOOT = "[!] jax 已 import，升级无效"
    else:
        _t = time.time()
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "jax[tpu]"],
                           check=True, capture_output=True, timeout=900)
            _BOOT = f"jax[tpu] 升级 OK {time.time() - _t:.0f}s"
        except Exception as _e:
            _BOOT = f"升级失败（继续跑，Pallas 可能被闸门挡住）：{type(_e).__name__}"
print(f"[ INFO ] bootstrap - {_BOOT}", flush=True)

# ==========================================================================
# ② 模型（AnimaLoraToolkit/jax_tpu/anima_jax.py）
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
        shift, scale, gate = jnp.split(h + adaln_lora, 3, axis=-1)     # [G, D] each
        take = lambda t: jnp.take(t, mod_index, axis=0)                # -> [ΣN, D]
        return take(shift), take(scale), take(gate)

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


# ── 整模前向（NaViT packed）───────────────────────────────────────────────────
def forward_packed(params: PyTree, cfg: AnimaConfig,
                   tokens: jnp.ndarray, timesteps: jnp.ndarray, ctx: jnp.ndarray,
                   rows: jnp.ndarray, cols: jnp.ndarray, mod_index: jnp.ndarray,
                   self_attn_fn, cross_attn_fn,
                   loras=None, remat: bool = True) -> jnp.ndarray:
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
    # **层号 i 必须走闭包默认参数，不能作为 checkpoint 的入参**：jax.checkpoint 会把
    # 入参统统 trace 成 tracer，于是 f"blocks.{i}" 拼出 "blocks.Traced<...>"，
    # LoRA 键全部匹配不上 -> LoRA 静默失效、梯度恒 0（本地冒烟实测 sum|grad_b|=0）。
    # 这种错不报异常，只会让"训练"变成空转，且 XLA 可能把反向整个 DCE 掉、
    # 使步时假性变快 —— 属于必须靠断言拦住的一类。
    for i in range(cfg.num_blocks):
        def one(carry, p, _i=i):
            return block_forward(carry, p, cfg, emb, adaln_lora, mod_index, ctx,
                                 cos, sin, self_attn_fn, cross_attn_fn, loras, _i)
        step = jax.checkpoint(one) if remat else one
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


def init_lora(key, cfg: AnimaConfig, rank: int = 32,
              targets: Sequence[str] = ("self_attn.q_proj", "self_attn.k_proj",
                                        "self_attn.v_proj", "self_attn.output_proj",
                                        "cross_attn.q_proj", "cross_attn.k_proj",
                                        "cross_attn.v_proj", "cross_attn.output_proj",
                                        "mlp.layer1", "mlp.layer2"),
              dtype=jnp.bfloat16) -> Dict[str, Dict[str, jnp.ndarray]]:
    """B 恒为 0 -> step-0 净增量为 0（本仓库落地新 adapter 的硬约束，见 skill 2.2）。"""
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
            out[f"blocks.{i}.{t}"] = {
                "a": (jax.random.normal(keys[n], (i_dim, rank), jnp.float32)
                      * (1.0 / math.sqrt(i_dim))).astype(dtype),
                "b": jnp.zeros((rank, o_dim), dtype),
            }
            n += 1
    return out

# ==========================================================================
# ③ job 主体（_job_body.py）
# ==========================================================================

"""Kaggle TPU v5e-8 · Anima 8 卡纯数据并行训练步实测。

**这一轮要回答的**：Anima(2.09B) 在 v5e-8 上，8 卡纯 DP 的 tokens/s、峰值 HBM、
以及每卡 token 预算的真实上限。

**为什么是纯 DP 而不是前几轮为 Krea2 定下的 FSDP**：
Anima bf16 权重仅 4.18GB < 单 chip 15.7GiB，**权重放得下**，所以每卡各存一份、
各跑自己的数据，每步只 all-reduce LoRA 梯度（r32 全目标约 118MB bf16，按 P0 实测
的 all-reduce 110GB/s 约 1ms）。Krea2 那套"每层 all-gather 22.6GB x2"在这里是
纯多余的通信。这条由 F1 的权重账直接推出，本 job 用实测 HBM 复核。

**注意力形态（与前几轮的 NaViT 块对角不同，这是本轮要验的第二件事）**：
走「按 token 数分桶 + 批处理」——一个 pack 内 G 张图 token 数相同，张量是
[G, n, D]，于是：
  * 自注意力 = 每图独立 -> splash **FullMask** + vmap，**不需要块对角 mask、
    不需要 MaskInfo 预处理、没有 host 侧成本**（前几轮 G1/G2 实测 730-845ms/布局
    的那笔开销直接消失）；
  * cross-attn = q[n] x kv[T_txt]，T_txt 固定 -> 稠密即可（n*T_txt 很小）。
形状全静态、XLA 只编译一次。代价是同一 pack 内图片 token 数须一致（仓库已有
token_bucket 支持）。本 job 测它的绝对吞吐，与 NaViT 打包路线的取舍留给下一轮。

数值正确性**不在本 job 验**——已在本地用真权重对拍完成
（fp32、28 块全过、rel 4.6e-05，见 jax_tpu/ 与 scratchpad 的 check_parity）。
本 job 只测性能与显存，因此用随机权重（形状一致 -> 性能一致），
省掉 4.18GB 上传。
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
# _BOOT 由 ① 段的 preamble 设好（升级 jax[tpu] 的结果）；单独跑 _job_body 时兜底
_BOOT = globals().get("_BOOT", "未经 build_job.py 拼接，无 bootstrap")
OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else tempfile.gettempdir()

# ── 实验参数 ──────────────────────────────────────────────────────────────────
T_TXT = 256          # 每图 caption token 数（Anima 文本长度是动态的，取代表值）
LORA_RANK = 32
DEFAULT_TOKENS_PER_DEV = 16384   # 每卡 token 预算（= G * n）
N_IMG_TOKENS = 4096              # 每图 token 数（1024^2 -> 64x64 grid）


def _flush():
    try:
        with open(os.path.join(OUT_DIR, "anima_dp.json"), "w", encoding="utf-8") as f:
            json.dump(RESULTS, f, ensure_ascii=False, indent=2)
        with open(os.path.join(OUT_DIR, "anima_dp_report.txt"), "w", encoding="utf-8") as f:
            for r in RESULTS:
                f.write(f"[{r['status']:^6}] {r['name']}"
                        + (f" - {r['detail']}" if r["detail"] else "") + "\n")
            f.write(f"\n（截至 {time.time() - _T0:.0f}s 的快照）\n")
    except OSError:
        pass


def record(name, status, detail=""):
    RESULTS.append({"name": name, "status": status, "detail": detail,
                    "t_elapsed_s": round(time.time() - _T0, 1)})
    mark = {"OK": "  OK  ", "FAIL": " FAIL ", "SKIP": " SKIP ", "INFO": " INFO "}.get(status, status)
    print(f"[{mark}] {name}" + (f" - {detail}" if detail else ""), flush=True)
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
    """block_until_ready 强制同步。

    前几轮两次踩过的坑（都在这里被挡住）：① 不消费输出 -> 惰性执行根本没派发，
    测出比单个注意力还快的假值；② 不 warmup -> 首步的编译时间混进步时。
    """
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


def _hbm(peak=False):
    """**读 peak_bytes_in_use 而不是 bytes_in_use**。

    前一轮 spmd_probe 报的 "单卡 HBM 3.05GiB" 是常驻数组量、不是峰值，
    导致 "余量很大" 的结论与 "L=32768 立刻 OOM" 自相矛盾。这里两个都取。
    """
    import jax
    st = jax.devices()[0].memory_stats() or {}
    k = "peak_bytes_in_use" if peak else "bytes_in_use"
    return st.get(k, 0) / 1024 ** 3


def _reset_peak():
    import jax
    d = jax.devices()[0]
    for name in ("reset_memory_stats", "clear_memory_stats"):
        fn = getattr(d, name, None)
        if fn is not None:
            try:
                fn()
                return True
            except Exception:
                pass
    return False   # 无接口时 peak 是历史值，读数要按"自进程启动以来"解读


def _smap(f, mesh, in_specs, out_specs):
    """shard_map 包装，关掉静态复制检查。

    **为什么纯 DP 也必须用 shard_map**：splash 是 Pallas/Mosaic 内核，XLA 的
    GSPMD 自动分区处理不了它 —— 真机第一跑报
    `NotImplementedError: Mosaic kernels cannot be automatically partitioned.`
    所以哪怕逻辑上只是"每卡各算各的"，也得显式写成 shard_map 让每卡跑本地内核。

    参数名 jax 0.8 之前叫 check_rep、0.11 起叫 check_vma，两个都试。
    """
    from jax.experimental.shard_map import shard_map
    for kw in ({"check_vma": False}, {"check_rep": False}, {}):
        try:
            return shard_map(f, mesh=mesh, in_specs=in_specs,
                             out_specs=out_specs, **kw)
        except TypeError:
            continue
    raise RuntimeError("shard_map 不接受任何已知的 check_* 参数名")


def _cleanup():
    import gc
    import jax
    try:
        for a in jax.live_arrays():
            try:
                a.delete()
            except Exception:
                pass
    except Exception:
        pass
    gc.collect()


def _grid_for(n_tok):
    """把 n_tok 拆成尽量方的 h x w（h*w 必须**恰好**等于 n_tok）。

    真机第一跑就栽在这里：原来用 side=round(sqrt(n))，n=512 时得 23x23=529≠512，
    广播直接报错。token 数不是完全平方数是常态（2304=48^2 是巧合，512/8192 都不是），
    所以必须走因数分解而不是开方取整。
    """
    h = max(d for d in range(1, int(n_tok ** 0.5) + 1) if n_tok % d == 0)
    return h, n_tok // h


def _rowcol(n_tok, jnp):
    h, w = _grid_for(n_tok)
    rr, cc = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing="ij")
    return rr.reshape(-1).astype(jnp.int32), cc.reshape(-1).astype(jnp.int32)


# ── 模型装配（复用上半部分的 anima_jax）────────────────────────────────────────
def make_random_params(key, cfg, dtype):
    """按真实形状随机造权重（性能只取决于形状）。**直接分片/直接 bf16 生成**：
    前几轮两次 OOM 都是先在单卡物化 fp32 再转换造成的，不重犯。"""
    import jax
    import jax.numpy as jnp

    D, C, R = cfg.model_channels, cfg.crossattn_dim, cfg.adaln_lora_dim
    F = int(D * cfg.mlp_ratio)
    ks = iter(jax.random.split(key, 8 + cfg.num_blocks * 16))

    def w(*shape):
        return (jax.random.normal(next(ks), shape, dtype) * 0.02)

    blocks = []
    for _ in range(cfg.num_blocks):
        blocks.append({
            "self_attn": {"q_proj": w(D, D), "k_proj": w(D, D), "v_proj": w(D, D),
                          "output_proj": w(D, D),
                          "q_norm": jnp.ones((cfg.head_dim,), dtype),
                          "k_norm": jnp.ones((cfg.head_dim,), dtype)},
            "cross_attn": {"q_proj": w(D, D), "k_proj": w(D, C), "v_proj": w(D, C),
                           "output_proj": w(D, D),
                           "q_norm": jnp.ones((cfg.head_dim,), dtype),
                           "k_norm": jnp.ones((cfg.head_dim,), dtype)},
            "mlp": {"layer1": w(F, D), "layer2": w(D, F)},
            "adaln_self_1": w(R, D), "adaln_self_2": w(3 * D, R),
            "adaln_cross_1": w(R, D), "adaln_cross_2": w(3 * D, R),
            "adaln_mlp_1": w(R, D), "adaln_mlp_2": w(3 * D, R),
        })
    return {
        "x_embedder": w(D, cfg.in_dim),
        "t_embedder_1": w(D, D), "t_embedder_2": w(3 * D, D),
        "t_embedding_norm": jnp.ones((D,), dtype),
        "final_adaln_1": w(R, D), "final_adaln_2": w(2 * D, R),
        "final_linear": w(cfg.out_dim, D),
        "blocks": blocks,
    }


def make_splash_full(n_tok, n_heads, interpret=False):
    """每图独立的全注意力 -> splash FullMask。

    **不需要块对角**：分桶批处理下每张图自成一个 batch 元素，图间天然不可见。
    这直接省掉前几轮 G1/G2 实测的 730-845ms/布局 MaskInfo host 开销。
    反向块 1024 + fused 来自 arch_probe H1（比默认快 2.03x）。
    """
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_kernel as sk, splash_attention_mask as sm,
    )
    # **块大小必须整除序列长**（splash_attention_mask_info.py:569 硬 raise）。
    # 反向想用 1024（H1 实测比默认 128 快 2.03x），但 1024 不整除 2304/9216 这类
    # token 数，所以按 n_tok 自适应取最大可用块 —— 这也意味着「每图 token 数」
    # 不能任取：至少要被 128 整除（前向块），否则 splash 直接构造失败。
    fwd = max(b for b in (128,) if n_tok % b == 0)
    bwd = next((b for b in (1024, 512, 256, 128) if n_tok % b == 0), 128)
    bs = sk.BlockSizes(block_q=fwd, block_kv=fwd, block_kv_compute=fwd,
                       block_q_dkv=bwd, block_kv_dkv=bwd, block_kv_dkv_compute=bwd,
                       use_fused_bwd_kernel=True)
    mask = sm.MultiHeadMask(masks=[sm.FullMask((n_tok, n_tok))] * n_heads)
    return sk.make_splash_mha(mask, head_shards=1, q_seq_shards=1,
                              block_sizes=bs, interpret=interpret)


def build_step(cfg, n_tok, dtype, interpret=False):
    """返回 loss_fn(lora, params, batch) —— 冻结底模、只对 LoRA 求梯度。"""
    import jax
    import jax.numpy as jnp
    import math as _m

    splash = make_splash_full(n_tok, cfg.num_heads, interpret)
    scale = cfg.head_dim ** -0.5

    def self_fn(q, k, v):
        # splash 期望 [H, S, D]，且**不内置 1/sqrt(d)**（前几轮硬结论，漏了会静默
        # 改变 softmax 温度、本地实测 rel 差到 9.5）。这里显式预缩放 q。
        qs = (q.astype(jnp.float32) * scale).astype(q.dtype).transpose(1, 0, 2)
        o = splash(qs, k.transpose(1, 0, 2), v.transpose(1, 0, 2))
        return o.transpose(1, 0, 2)

    def cross_fn(q, k, v):
        # q[n, H, D] x kv[T_TXT, H, D]，T_TXT 小 -> 稠密就够（n*T 的 logits 很小）
        logits = jnp.einsum("shd,thd->hst", q, k).astype(jnp.float32) * scale
        w = jax.nn.softmax(logits, axis=-1).astype(v.dtype)
        return jnp.einsum("hst,thd->shd", w, v)

    def per_image(params, lora, tok, t, ctx, rows, cols, target):
        mod_index = jnp.zeros((n_tok,), jnp.int32)     # 单图 -> 全部指向第 0 行
        out = forward_packed(params, cfg, tok, t[None], ctx, rows, cols, mod_index,
                             self_fn, cross_fn, loras=lora, remat=True)
        return jnp.mean((out.astype(jnp.float32) - target.astype(jnp.float32)) ** 2)

    def loss_fn(lora, params, batch):
        # vmap 到 pack 内的 G 张图；DP 由 jit 的 sharding 负责跨卡
        losses = jax.vmap(per_image, in_axes=(None, None, 0, 0, 0, 0, 0, 0))(
            params, lora, batch["tok"], batch["t"], batch["ctx"],
            batch["rows"], batch["cols"], batch["target"])
        return jnp.mean(losses)

    return loss_fn


def make_batch(key, cfg, n_dev, G, n_tok, dtype, mesh):
    """造一个分片好的 batch：[n_dev*G, ...]，第 0 维按卡切。"""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    B = n_dev * G
    ks = jax.random.split(key, 5)
    sh = lambda spec: NamedSharding(mesh, spec)

    def mk():
        rr, cc = _rowcol(n_tok, jnp)
        return {
            "tok": jax.random.normal(ks[0], (B, n_tok, cfg.in_dim), dtype),
            "t": jax.random.uniform(ks[1], (B,), jnp.float32),
            "ctx": jax.random.normal(ks[2], (B, T_TXT, cfg.crossattn_dim), dtype),
            "rows": jnp.broadcast_to(rr[None], (B, n_tok)),
            "cols": jnp.broadcast_to(cc[None], (B, n_tok)),
            "target": jax.random.normal(ks[3], (B, n_tok, cfg.out_dim), dtype),
        }

    specs = {k: P("d") for k in ("tok", "t", "ctx", "rows", "cols", "target")}
    return jax.jit(mk, out_shardings={k: sh(v) for k, v in specs.items()})()


# ── 探测 ──────────────────────────────────────────────────────────────────────
def probe_env():
    record("python", "INFO", f"{platform.python_version()} @ {sys.executable}")
    record("bootstrap", "INFO", _BOOT)


@probe("0.1 设备 / HBM")
def p_basics():
    import jax
    d = jax.devices()
    lim = (d[0].memory_stats() or {}).get("bytes_limit", 0) / 1024 ** 3
    return (f"jax {jax.__version__} | {len(d)} x {d[0].device_kind} | "
            f"单设备 HBM limit={lim:.1f}GiB")


@probe("A1 splash 可用性（Pallas 闸门）")
def p_splash():
    import jax
    import jax.numpy as jnp
    fn = make_splash_full(256, 4)
    q = jnp.zeros((4, 256, 128), jnp.bfloat16)
    o = jax.jit(fn)(q, q, q)
    jax.block_until_ready(o)
    pv = jax.devices()[0].client.platform_version
    return f"splash 编译通过 | {pv.split('Built on')[-1].strip()[:24] if 'Built on' in pv else pv[:40]}"


@probe("W1 Anima 权重账（决定要不要分片）")
def p_weights():
    import jax
    cfg = AnimaConfig()
    D, C, R = cfg.model_channels, cfg.crossattn_dim, cfg.adaln_lora_dim
    F = int(D * cfg.mlp_ratio)
    per = (4 * D * D) + (2 * D * D + 2 * D * C) + (2 * D * F) + 3 * (R * D + 3 * D * R)
    total = per * cfg.num_blocks + D * cfg.in_dim + D * D + 3 * D * D + D + R * D + 2 * D * R + cfg.out_dim * D
    lim = (jax.devices()[0].memory_stats() or {}).get("bytes_limit", 0) / 1024 ** 3
    return (f"每块 {per / 1e6:.1f}M x {cfg.num_blocks} 块 = {total / 1e9:.3f}B | "
            f"bf16 {total * 2 / 1e9:.2f}GB vs 单 chip {lim:.1f}GiB -> "
            f"**{'放得下，纯 DP 即可，无需 FSDP/TP' if total * 2 / 1024**3 < lim else '放不下，须分片'}**")


@probe("B1 LoRA 接线自检（没有它，基准可能在测一个空转的模型）")
def p_lora_wired():
    """本地冒烟抓到过一次：jax.checkpoint 把层号 trace 成 tracer，f"blocks.{i}" 拼出
    垃圾键 -> LoRA 全部匹配不上、梯度恒 0，且 XLA 可能把反向整个 DCE 掉使步时假性
    变快。这类错不报异常，只能靠断言拦。故基准前先在真机上验一次。

    判据两条：① grad_b 必须非零（LoRA 真的在图里）；② grad_a 必须恒 0
    （B 零初始化 -> step-0 净增量为 0，本仓库落地 adapter 的硬约束）。
    """
    import jax
    import jax.numpy as jnp
    _cleanup()
    cfg = AnimaConfig(num_blocks=2)
    n = 512
    params = make_random_params(jax.random.PRNGKey(0), cfg, jnp.bfloat16)
    lora = init_lora(jax.random.PRNGKey(1), cfg, LORA_RANK, dtype=jnp.bfloat16)
    rr, cc = _rowcol(n, jnp)
    G = 2
    batch = {
        "tok": jax.random.normal(jax.random.PRNGKey(2), (G, n, cfg.in_dim), jnp.bfloat16),
        "t": jnp.array([0.3, 0.8], jnp.float32),
        "ctx": jax.random.normal(jax.random.PRNGKey(3), (G, T_TXT, cfg.crossattn_dim), jnp.bfloat16),
        "rows": jnp.broadcast_to(rr[None], (G, n)),
        "cols": jnp.broadcast_to(cc[None], (G, n)),
        "target": jax.random.normal(jax.random.PRNGKey(4), (G, n, cfg.out_dim), jnp.bfloat16),
    }
    g = jax.jit(jax.grad(build_step(cfg, n, jnp.bfloat16), argnums=0))(lora, params, batch)
    ga = sum(float(jnp.abs(v["a"]).sum()) for v in g.values())
    gb = sum(float(jnp.abs(v["b"]).sum()) for v in g.values())
    nz = sum(1 for v in g.values() if float(jnp.abs(v["b"]).max()) > 0)
    if gb <= 0:
        raise RuntimeError(f"grad_b 全 0 —— LoRA 没接进图，后续步时无意义（{len(g)} 条）")
    if ga != 0:
        raise RuntimeError(f"grad_a={ga} 非 0 —— B 不是零初始化，step-0 不中立")
    _cleanup()
    return (f"sum|grad_b|={gb:.3e}，非零条目 {nz}/{len(g)}；"
            f"sum|grad_a|=0（step-0 中立）-> LoRA 已接进计算图")


def _run_dp(G, n_tok, reset_peak=True):
    """跑一次 8 卡 DP 训练步，返回实测字典。"""
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    _cleanup()
    if reset_peak:
        _reset_peak()
    devs = jax.devices()
    if len(devs) < 8:
        raise _Skip(f"只有 {len(devs)} 个设备")
    mesh = Mesh(np.array(devs).reshape(8), ("d",))
    cfg = AnimaConfig()
    dtype = jnp.bfloat16
    k = jax.random.PRNGKey(0)

    params = jax.jit(lambda: make_random_params(k, cfg, dtype),
                     out_shardings=NamedSharding(mesh, P()))()      # 复制到每卡
    lora = jax.jit(lambda: init_lora(jax.random.PRNGKey(1), cfg, LORA_RANK, dtype=dtype),
                   out_shardings=NamedSharding(mesh, P()))()
    batch = make_batch(jax.random.PRNGKey(2), cfg, 8, G, n_tok, dtype, mesh)

    local = build_step(cfg, n_tok, dtype)

    def per_shard(lora, params, batch):
        # 每卡跑自己那 G 张图；返回 [1] 而不是标量（out_specs=P("d") 需要有轴可切）
        return local(lora, params, batch)[None]

    data_spec = {k: P("d") for k in ("tok", "t", "ctx", "rows", "cols", "target")}
    f = _smap(per_shard, mesh, (P(), P(), data_spec), P("d"))
    # lora/params 是复制的（in_spec P()）-> 它们的余切在 shard_map 转置时自动 psum，
    # 即 LoRA 梯度的 all-reduce 已包含在这一步的实测时间里。
    loss_fn = lambda lo, pa, ba: jnp.mean(f(lo, pa, ba))
    grad = jax.jit(jax.grad(loss_fn, argnums=0))
    ms, first = _bench(lambda: grad(lora, params, batch))
    tok = 8 * G * n_tok
    n_lora = sum(int(v["a"].size + v["b"].size) for v in lora.values())
    return {"ms": ms, "first": first, "tok": tok, "tok_s": tok / (ms / 1e3),
            "hbm": _hbm(), "peak": _hbm(peak=True), "G": G, "n": n_tok,
            "lora_mb": n_lora * 2 / 1e6}


@probe("D1 8 卡纯 DP 全 28 块训练步（主判据）")
def p_dp():
    G = DEFAULT_TOKENS_PER_DEV // N_IMG_TOKENS
    r = _run_dp(G, N_IMG_TOKENS)
    return (f"每卡 {r['G']} 图 x {r['n']} token = {r['G'] * r['n']} token/卡 | "
            f"8 卡共 {r['tok'] / 1000:.0f}k token/步 | 步时 {r['ms']:.0f}ms"
            f"（首步含编译 {r['first']:.0f}s）-> **{r['tok_s'] / 1e3:.1f}k tokens/s** | "
            f"常驻 HBM {r['hbm']:.2f}GiB / **峰值 {r['peak']:.2f}GiB** | "
            f"LoRA {r['lora_mb']:.0f}MB(bf16，每步 all-reduce 量)")


@probe("D2 每卡 token 预算扫描（找真实上限，不止步于'能跑的最大值'）")
def p_budget():
    out = []
    for budget in (8192, 16384, 24576, 32768, 49152):
        G = budget // N_IMG_TOKENS
        if G < 1:
            continue
        try:
            r = _run_dp(G, N_IMG_TOKENS)
            out.append(f"{budget}(G={G}): {r['ms']:.0f}ms {r['tok_s'] / 1e3:.1f}k tok/s "
                       f"峰值{r['peak']:.1f}GiB")
        except Exception as e:
            out.append(f"{budget}(G={G}): {type(e).__name__}")
            _cleanup()
    return " ; ".join(out)


@probe("D3 单图 token 数扫描（分辨率维度）")
def p_imgsize():
    out = []
    for n in (1024, 2304, 4096, 9216, 16384):
        G = max(1, 16384 // n)
        try:
            r = _run_dp(G, n)
            out.append(f"n={n}(G={G}): {r['ms']:.0f}ms {r['tok_s'] / 1e3:.1f}k tok/s "
                       f"峰值{r['peak']:.1f}GiB")
        except Exception as e:
            out.append(f"n={n}(G={G}): {type(e).__name__}")
            _cleanup()
    return " ; ".join(out)


@probe("D4 单卡对照（算并行效率，1 卡跑同样的每卡负载）")
def p_single():
    import jax
    import jax.numpy as jnp
    _cleanup()
    cfg = AnimaConfig()
    dtype = jnp.bfloat16
    G = DEFAULT_TOKENS_PER_DEV // N_IMG_TOKENS
    n = N_IMG_TOKENS
    dev = jax.devices()[0]
    params = jax.jit(lambda: make_random_params(jax.random.PRNGKey(0), cfg, dtype),
                     device=dev)()
    lora = jax.jit(lambda: init_lora(jax.random.PRNGKey(1), cfg, LORA_RANK, dtype=dtype),
                   device=dev)()
    ks = jax.random.split(jax.random.PRNGKey(2), 4)
    rr, cc = _rowcol(n, jnp)
    batch = jax.jit(lambda: {
        "tok": jax.random.normal(ks[0], (G, n, cfg.in_dim), dtype),
        "t": jax.random.uniform(ks[1], (G,), jnp.float32),
        "ctx": jax.random.normal(ks[2], (G, T_TXT, cfg.crossattn_dim), dtype),
        "rows": jnp.broadcast_to(rr[None], (G, n)),
        "cols": jnp.broadcast_to(cc[None], (G, n)),
        "target": jax.random.normal(ks[3], (G, n, cfg.out_dim), dtype),
    }, device=dev)()
    grad = jax.jit(jax.grad(build_step(cfg, n, dtype), argnums=0))
    ms, _ = _bench(lambda: grad(lora, params, batch))
    return (f"单卡 {G} 图 x {n} token | 步时 {ms:.0f}ms -> "
            f"{(G * n) / (ms / 1e3) / 1e3:.1f}k tokens/s（8 卡理想 = 此值 x8）")


def main():
    probe_env()
    if not p_basics():
        _flush()
        return 0
    p_weights()
    if not p_splash():
        record("裁决", "FAIL", "splash 不可用 -> 后续步时无意义，停在这里")
        _flush()
        return 0
    if not p_lora_wired():
        record("裁决", "FAIL", "LoRA 未接进图 -> 步时会测到一个空转模型，停在这里")
        _flush()
        return 0
    p_dp()
    p_single()
    p_budget()
    p_imgsize()
    ok = sum(1 for r in RESULTS if r["status"] == "OK")
    bad = sum(1 for r in RESULTS if r["status"] == "FAIL")
    record("汇总", "INFO", f"OK={ok} FAIL={bad} 总耗时={time.time() - _T0:.0f}s")
    _flush()
    return 0   # 恒返回 0：单项 FAIL 是数据，不是脚本故障（Kaggle 会把非零判成 ERROR）


if __name__ == "__main__":
    sys.exit(main())