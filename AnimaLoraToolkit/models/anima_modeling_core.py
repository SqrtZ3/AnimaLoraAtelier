# Auto-generated minimal Anima model file.
# Combines cosmos_predict2_modeling.py and anima_modeling.py.
# Original files are kept in anima_vendor/models.

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# PEP 563：注解延迟求值。本文件的 forward() 签名用了 ``X | Y`` 写法，Python 3.9
# 运行时求值注解会 TypeError（昇腾镜像里 Python 常见为 3.9）。加这一行后注解只作为
# 字符串保存，不改变任何运行时行为。
from __future__ import annotations

import contextlib
import functools
import math
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn
from torch.utils.checkpoint import checkpoint

_XFORMERS_AVAILABLE = False
_USE_XFORMERS = False
try:
    import xformers.ops as xops  # type: ignore
    _XFORMERS_AVAILABLE = True
except Exception:
    xops = None


def set_xformers_enabled(enabled: bool) -> bool:
    global _USE_XFORMERS
    _USE_XFORMERS = bool(enabled) and _XFORMERS_AVAILABLE
    return _USE_XFORMERS


# attn_force_autocast_dtype（opt-in, default-off）：见 _unify_attn_dtype 的 docstring。
# 关 = 只在 q/k/v dtype 不一致时归一（修 xformers 崩，其余逐 bit 不变）；
# 开 = autocast 开着时一律按 autocast dtype 算注意力（把 NaViT 块对角路径从 fp32
# 拉回 bf16，与 dense/eval/采样的 SDPA 口径一致）。
_ATTN_FORCE_AUTOCAST_DTYPE = False


def set_attn_force_autocast_dtype(enabled: bool) -> bool:
    global _ATTN_FORCE_AUTOCAST_DTYPE
    _ATTN_FORCE_AUTOCAST_DTYPE = bool(enabled)
    return _ATTN_FORCE_AUTOCAST_DTYPE


# ── NaViT 打包注意力后端（opt-in，默认 xformers = 历史行为逐 bit 不变）───────────
#
# 三个后端算的是**同一件事**：块对角注意力 —— 打包序列里每张图只看自己的 token，
# 段间零泄漏，且不物化 O(ΣN²) 的稠密 mask。区别只在用什么 kernel：
#
#   "xformers"  BlockDiagonalMask + memory_efficient_attention 的 varlen 快核。
#               历史默认，CUDA 上逐 bit 不变。昇腾无此包。
#   "npu_tnd"   torch_npu.npu_fusion_attention(input_layout="TND")，昇腾**原生**变长
#               融合注意力：actual_seq_qlen / actual_seq_kvlen 传累加和，语义与
#               BlockDiagonalMask 等价（cross-attn 的 q/kv 段长不等也支持）。
#               可用性必须由 tools/npu_probe.py 在真机上实测，不在这里假设。
#   "sdpa_seg"  逐段 dense SDPA。段内全注意力 ≡ 块对角，数学恒等（本文件的对拍单测
#               tests/test_anima_packed_attn_backends.py 用稠密 bool mask 对拍）。
#               不依赖任何专有算子，是 TND 挂掉时的保底路径。
#
# 与 krea2 的同名开关（models/krea2_modeling.py）是**各自独立**的模块级状态，训练入口
# 按 model_family 分别设置。
_PACKED_ATTN_BACKEND = "xformers"
_PACKED_ATTN_BACKENDS = ("xformers", "sdpa_seg", "npu_tnd")


def set_packed_attention_backend(name: str) -> str:
    """训练入口调用（trainer 读 ``navit_attn_backend``）。非法值构造期 fail-fast。"""
    global _PACKED_ATTN_BACKEND
    name = (name or "xformers").lower()
    if name not in _PACKED_ATTN_BACKENDS:
        raise ValueError(
            f"navit_attn_backend={name!r} 不认识；Anima family 可选 {_PACKED_ATTN_BACKENDS}")
    _PACKED_ATTN_BACKEND = name
    return _PACKED_ATTN_BACKEND


def get_packed_attention_backend() -> str:
    return _PACKED_ATTN_BACKEND


# ── sdpa_seg 的段内 query 分块（opt-in，默认 0 = 关）─────────────────────────────
#
# 只在**没有 flash / mem-efficient SDPA 后端**的平台上才需要（如海光 DCU：真机探针
# 显示 torch 编译时就没带 mem-efficient / cuDNN attention，flash 的 .so 也缺失，
# head_dim=128 / S=4096 只有 MATH 可用）。math backend 会逐元素物化 softmax(q@kᵀ)，
# 显存 O(S²)：一张 2048×2048 原生分辨率图是 16384 token，16 头 fp32 的 S×S 就是
# 16.00 GiB —— 真机上训练第一步的 backward 就是这样爆的。
#
# 关键的一点（本地 GPU 强制 MATH 后端实测，S=4096/H=16/bf16）：**光按 query 分块没用**，
# 因为 math SDPA 会把 S×S 的 softmax 结果存进 backward 图，切块只是把一个大张量拆成
# 若干小的，加起来还是 S×S：
#     整块 SDPA                    forward 后持有 1.12 GiB   峰值 2.41 GiB
#     query 分块 (c=1024)          forward 后持有 1.30 GiB   峰值 1.64 GiB   ← 没省
#     query 分块 + 每块 checkpoint  forward 后持有 0.02 GiB   峰值 0.65 GiB   ← 56×↓
# 所以这里每个 query 块**再套一层 gradient checkpoint**，backward 时逐块重算。
# 代价是注意力多跑一遍前向。
#
# 数学上：每个 query 的 softmax 归一化域仍是本段全部 key，与不分块逐元素恒等
# （tests/test_seg_attn_chunk.py 对拍前向与梯度）。0 = 关，走原来的整段 SDPA。
_SEG_ATTN_CHUNK_TOKENS = 0


def set_seg_attn_chunk_tokens(n: int) -> int:
    """设置 sdpa_seg 段内 query 分块大小（token 数）。0 = 关。返回生效值。"""
    global _SEG_ATTN_CHUNK_TOKENS
    n = int(n or 0)
    if n < 0:
        raise ValueError(f"navit_attn_chunk_tokens 必须 ≥0，收到 {n}")
    _SEG_ATTN_CHUNK_TOKENS = n
    return _SEG_ATTN_CHUNK_TOKENS


def get_seg_attn_chunk_tokens() -> int:
    return _SEG_ATTN_CHUNK_TOKENS


def _sdpa_plain(q, k, v):
    return F.scaled_dot_product_attention(q, k, v)


def _seg_sdpa_chunked(qs, ks, vs, chunk: int):
    """段内按 query 分块的 SDPA，每块套 gradient checkpoint。

    qs/ks/vs: [B, H, s, D]。与 ``F.scaled_dot_product_attention(qs, ks, vs)`` 数学恒等。
    无梯度时（eval / 采样）不套 checkpoint —— 那种场景没有 backward 图，分块本身就够。
    """
    sq = qs.shape[-2]
    if chunk <= 0 or sq <= chunk:
        return F.scaled_dot_product_attention(qs, ks, vs)
    use_ckpt = torch.is_grad_enabled() and (
        qs.requires_grad or ks.requires_grad or vs.requires_grad)
    outs = []
    for st in range(0, sq, chunk):
        qc = qs[:, :, st:st + chunk, :]
        if use_ckpt:
            outs.append(checkpoint(_sdpa_plain, qc, ks, vs, use_reentrant=False))
        else:
            outs.append(F.scaled_dot_product_attention(qc, ks, vs))
    return torch.cat(outs, dim=-2)


class _SegLens:
    """打包注意力的轻量段长标记（替代 xformers ``BlockDiagonalMask``）。

    纯 CPU 元数据：``q_seqlens`` 是每张图的 visual token 数，``kv_seqlens`` 在 self-attn
    等于前者、在 cross-attn 是每条 caption 的文本 token 数。``torch_attention_op``
    检测到它就走 sdpa_seg / npu_tnd 后端。
    """
    __slots__ = ("q_seqlens", "kv_seqlens")

    def __init__(self, q_seqlens, kv_seqlens=None):
        self.q_seqlens = tuple(int(s) for s in q_seqlens)
        self.kv_seqlens = (self.q_seqlens if kv_seqlens is None
                           else tuple(int(s) for s in kv_seqlens))
        if len(self.q_seqlens) != len(self.kv_seqlens):
            raise ValueError(
                f"q/kv 段数不一致：{len(self.q_seqlens)} vs {len(self.kv_seqlens)}")

    def __repr__(self) -> str:  # pragma: no cover - 仅调试用
        return f"_SegLens(q={self.q_seqlens}, kv={self.kv_seqlens})"


@functools.lru_cache(maxsize=256)
def _cached_seg_lens(q_seqlens: tuple, kv_seqlens: Optional[tuple] = None) -> _SegLens:
    """按 seqlens 缓存 ``_SegLens``，与 ``_cached_block_diag_mask`` 同样的复用理由。"""
    return _SegLens(q_seqlens, kv_seqlens)


def _packed_attention_seg(q_B_S_H_D, k_B_S_H_D, v_B_S_H_D, seg: _SegLens):
    """逐段 dense SDPA。输入/输出均为 [B, S, H, D]（打包路径 B==1）。"""
    outs = []
    qo = ko = 0
    for sq, sk in zip(seg.q_seqlens, seg.kv_seqlens):
        # [B, s, H, D] -> [B, H, s, D]
        qs = q_B_S_H_D[:, qo:qo + sq].transpose(1, 2)
        ks = k_B_S_H_D[:, ko:ko + sk].transpose(1, 2)
        vs = v_B_S_H_D[:, ko:ko + sk].transpose(1, 2)
        o = _seg_sdpa_chunked(qs, ks, vs, _SEG_ATTN_CHUNK_TOKENS)
        outs.append(o.transpose(1, 2))                       # 回 [B, s, H, D]
        qo += sq
        ko += sk
    return torch.cat(outs, dim=1)


def _packed_attention_npu_tnd(q_B_S_H_D, k_B_S_H_D, v_B_S_H_D, seg: _SegLens):
    """昇腾原生变长融合注意力（TND）。输入/输出 [B, S, H, D]，要求 B==1。

    TND 布局就是把 batch 维去掉的 [ΣS, H, D]；段边界由 ``actual_seq_*`` 的**累加和**
    给出。这与打包路径的内存布局天然一致，无需任何转置/拷贝。
    """
    import torch_npu  # 只有该后端需要；导入失败要吵，不静默回退

    B = q_B_S_H_D.shape[0]
    if B != 1:
        raise RuntimeError(f"npu_tnd 后端只支持打包路径的 B=1，收到 B={B}")
    H = q_B_S_H_D.shape[-2]
    D = q_B_S_H_D.shape[-1]
    q = q_B_S_H_D.reshape(-1, H, D)
    k = k_B_S_H_D.reshape(-1, H, D)
    v = v_B_S_H_D.reshape(-1, H, D)

    cu_q, cu_kv, acc_q, acc_kv = [], [], 0, 0
    for sq, sk in zip(seg.q_seqlens, seg.kv_seqlens):
        acc_q += sq
        acc_kv += sk
        cu_q.append(acc_q)
        cu_kv.append(acc_kv)

    out = torch_npu.npu_fusion_attention(
        q, k, v, H,
        input_layout="TND",
        scale=1.0 / math.sqrt(D),          # 与 SDPA / xformers 的默认缩放一致
        actual_seq_qlen=cu_q,
        actual_seq_kvlen=cu_kv,
    )
    o = out[0] if isinstance(out, (tuple, list)) else out
    return o.reshape(B, -1, H, D)


@contextlib.contextmanager
def _fp32_autocast(dev_type: str):
    """在支持 autocast 的加速器上开一个"不要降精度"的区域；其余设备是 no-op。

    只对 ``cuda`` / ``npu`` 生效：CPU autocast 只支持 bf16/fp16，开 fp32 会报错，而
    原来的 ``@torch.autocast('cuda', ...)`` 装饰器在 CPU 上本来就无效果——保持一致。

    **两条路的写法不同，原因见下（这不是笔误）：**

    - ``cuda``：``autocast(dtype=fp32)``，与原装饰器逐字节等价，行为不动。
    - ``npu``：``autocast(enabled=False)``。torch_npu 的 autocast **只支持
      fp16/bf16**，传 fp32 会走 ``torch/amp/autocast_mode.py`` 的
      ``if enabled and self.fast_dtype not in supported_dtype`` 分支——打一条
      ``"In npu autocast, but the target dtype is not supported. Disabling
      autocast."`` 然后把 ``enabled`` 直接置 False。也就是说昇腾上这段代码**一直**
      落在 enabled=False，只是靠一条 Python warning（默认每个位置只报一次）表达，
      属于静默降级。这里显式写出来：与 torch_npu 当前实际行为逐位一致，不再依赖
      被吞掉的警告，也不会在 torch_npu 将来支持 fp32 autocast 时行为突变。

    为什么 ``enabled=False`` 在本函数唯一的用途（:meth:`RMSNorm.forward`）上与 fp32
    autocast 等价：区域内只有 ``pow/mean/rsqrt/mul`` 这些**非 autocast 算子**
    （autocast 只改 matmul/conv 等白名单算子的 dtype），``x`` 已由调用方 ``.float()``
    显式提到 fp32，``output * self.weight`` 的结果由 type promotion
    （bf16 × fp32 → fp32）决定——两种写法给出同一个 dtype 链条。若以后往这个区域里
    加 matmul 之类的白名单算子，两条路就**不再等价**，届时必须重新裁决。
    """
    if dev_type == "cuda":
        with torch.autocast(dev_type, dtype=torch.float32):
            yield
    elif dev_type == "npu":
        with torch.autocast(dev_type, enabled=False):
            yield
    else:
        yield


def _is_xformers_attn_bias(m) -> bool:
    """True iff ``m`` is an xformers attention-bias object (e.g. ``BlockDiagonalMask``).

    Used by :func:`torch_attention_op` to distinguish the NaViT/FiT block-diagonal
    *packing* path (an ``AttentionBias`` routed through ``memory_efficient_attention``'s
    fast varlen kernel) from the legacy *additive float mask* path (a plain
    ``torch.Tensor`` routed through SDPA). Returns False — never raises — when xformers
    is absent, so non-packed callers are unaffected.
    """
    if m is None or isinstance(m, torch.Tensor):
        return False
    try:
        from xformers.ops.fmha.attn_bias import AttentionBias
    except Exception:
        return False
    return isinstance(m, AttentionBias)


def _unify_attn_dtype(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """把 q/k/v 统一到同一 dtype；三者本就同 dtype 时原样返回（零拷贝、行为不变）。

    为什么需要：autocast(bf16) 下 ``nn.LayerNorm`` 产出 **fp32**（PyTorch 的 autocast
    fp32 策略，本地 torch 2.9 实测），而被 LoRA/LoKr/DoRA 包住的 Linear 会把输出 cast
    回**输入** dtype（``trainer/lora.py`` 的 ``y.to(dtype=x.dtype)`` / DoRA 分支的
    ``.to(dtype=x.dtype)``）——于是 ``q_proj(normalized_x)`` 吐 fp32，而 cross-attn 的
    ``k_proj/v_proj`` 吃的是 bf16 的 ``crossattn_emb`` → 吐 bf16。未打 LoRA 时 Linear 走
    autocast 恒输出 bf16，三者一致，所以这条不一致只在注入 LoRA 后出现。

    SDPA 是 autocast 算子，会自己把三者统一成 bf16（dense 路径因此一直没暴露问题）；
    ``xops.memory_efficient_attention`` 不是 autocast 算子，``validate_inputs`` 直接
    ValueError —— NaViT 块对角路径必走它，于是训练第一步就崩。

    统一口径：autocast 开着就按 autocast dtype（= SDPA 在 dense 路径的既有行为，训练/
    评估两条路的注意力精度因此一致）；autocast 关着（eval/采样）才按最宽 dtype 提升，
    不静默降精度。

    ``set_attn_force_autocast_dtype(True)``（YAML: ``attn_force_autocast_dtype``，
    默认关）时更进一步：三者**已经**同为 fp32 也拉回 autocast dtype。这一条针对的是
    NaViT 块对角路径的 self-attn —— 注入 LoRA 后 q/k/v 全是 fp32（同 dtype，xformers
    不报错），于是整条自注意力跑 fp32 kernel；而 dense/eval/采样走 SDPA（autocast
    算子）一直是 bf16。开了它两条路口径才真的一致，代价是 navit 的注意力数值从 fp32
    变 bf16（本地 SDPA 代理测量 S=4096：fp32 比 bf16 慢 3.2×、峰值显存 1.84×；
    xformers 真实核未在本地验证）。
    """
    same = q.dtype == k.dtype == v.dtype
    if same and not _ATTN_FORCE_AUTOCAST_DTYPE:
        return q, k, v
    target = None
    dev_type = q.device.type
    try:
        if torch.is_autocast_enabled(dev_type):
            target = torch.get_autocast_dtype(dev_type)
    except TypeError:  # torch < 2.4：无 device_type 形参
        if dev_type == "cuda" and torch.is_autocast_enabled():
            target = torch.get_autocast_gpu_dtype()
    if same:
        # force 模式：autocast 关着（eval/采样/no_grad）时不动，保持既有精度
        return (q, k, v) if target is None or target == q.dtype else (
            q.to(target), k.to(target), v.to(target))
    if target is None:
        target = torch.promote_types(torch.promote_types(q.dtype, k.dtype), v.dtype)
    return q.to(target), k.to(target), v.to(target)


@functools.lru_cache(maxsize=256)
def _cached_block_diag_mask(q_seqlens: tuple, kv_seqlens: Optional[tuple] = None):
    """Build (and memoize) an xformers ``BlockDiagonalMask`` for the given seqlens.

    The mask is a pure CPU metadata object keyed only by the seqlen lists — the NaViT
    step already reuses one instance across all 28 blocks and the checkpoint recompute,
    so reusing it across steps with identical pack composition is equally safe. Small
    datasets cycle through few distinct packs per epoch → high hit rate; entries are a
    few ints + tiny tensors, so a bounded cache stays negligible.
    """
    from xformers.ops.fmha import BlockDiagonalMask  # lazy: only packed callers need it

    if kv_seqlens is None:
        return BlockDiagonalMask.from_seqlens(list(q_seqlens))
    return BlockDiagonalMask.from_seqlens(
        q_seqlen=list(q_seqlens), kv_seqlen=list(kv_seqlens)
    )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb_base(t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    if freqs.ndim == 5:
        freqs = freqs[:, : t.shape[1], 0]
    else:
        freqs = freqs[: t.shape[1]].transpose(0, 1)
    cos_ = torch.cos(freqs).to(t.dtype)
    sin_ = torch.sin(freqs).to(t.dtype)
    rot_dim = freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * cos_) + (_rotate_half(t) * sin_)
    return torch.cat((t, t_pass), dim=-1)


def apply_rotary_pos_emb(t: torch.Tensor, freqs: torch.Tensor, **_kwargs) -> torch.Tensor:
    return _apply_rotary_pos_emb_base(t, freqs)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原实现是 `@torch.autocast('cuda', dtype=torch.float32)` 装饰器——设备串写死
        # 在**类定义期**，在昇腾上等于给 npu 张量开了一个 cuda autocast 区域，
        # 完全不起作用（`output * self.weight` 会跟着外层 bf16 autocast 走，而不是
        # 原意的 fp32）。改成按张量实际设备取 device_type。
        # 行为中立性：CUDA 上 device_type=='cuda'，与原装饰器逐字节等价；CPU 上原来
        # 也是无效果（cuda autocast 不作用于 cpu 张量），这里显式跳过，保持无效果。
        with _fp32_autocast(x.device.type):
            output = self._norm(x.float()).type_as(x)
            return output * self.weight


# ---------------------- Feed Forward Network -----------------------
class GPT2FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.activation = nn.GELU()
        self.layer1 = nn.Linear(d_model, d_ff, bias=False)
        self.layer2 = nn.Linear(d_ff, d_model, bias=False)

        self._layer_id = None
        self._dim = d_model
        self._hidden_dim = d_ff
        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self._dim)
        torch.nn.init.trunc_normal_(self.layer1.weight, std=std, a=-3 * std, b=3 * std)

        # scale init by depth as in https://arxiv.org/abs/1908.11365 -- worked slightly better.
        std = 1.0 / math.sqrt(self._hidden_dim)
        if self._layer_id is not None:
            std = std / math.sqrt(2 * (self._layer_id + 1))
        torch.nn.init.trunc_normal_(self.layer2.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)

        x = self.activation(x)
        x = self.layer2(x)
        return x


def torch_attention_op(
    q_B_S_H_D: torch.Tensor,
    k_B_S_H_D: torch.Tensor,
    v_B_S_H_D: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Computes multi-head attention using PyTorch's native implementation.

    This function provides a PyTorch backend alternative to Transformer Engine's attention operation.
    It rearranges the input tensors to match PyTorch's expected format, computes scaled dot-product
    attention, and rearranges the output back to the original format.

    The input tensor names use the following dimension conventions:

    - B: batch size
    - S: sequence length
    - H: number of attention heads
    - D: head dimension

    Args:
        q_B_S_H_D: Query tensor with shape (batch, seq_len, n_heads, head_dim)
        k_B_S_H_D: Key tensor with shape (batch, seq_len, n_heads, head_dim)
        v_B_S_H_D: Value tensor with shape (batch, seq_len, n_heads, head_dim)

    Returns:
        Attention output tensor with shape (batch, seq_len, n_heads * head_dim)
    """
    # NaViT/FiT block-diagonal packing path: ``attn_mask`` is an xformers
    # ``AttentionBias`` (e.g. ``BlockDiagonalMask``) carrying per-image seqlens, not an
    # additive float tensor. Route it through ``memory_efficient_attention``'s fast
    # varlen kernel so each packed image attends only to its own tokens — no
    # cross-image leakage and no O(N²) dense mask. Requires xformers; raise loudly if
    # a bias was requested but xformers is unavailable (silently falling back to dense
    # SDPA would defeat the purpose and could OOM on long packed sequences).
    # LoRA 注入后 q 与 k/v 可能一个 fp32 一个 bf16（见 _unify_attn_dtype）。SDPA 自己会
    # 统一，xformers 不会——两条分支都先归一，保证走哪个后端语义一致。
    q_B_S_H_D, k_B_S_H_D, v_B_S_H_D = _unify_attn_dtype(q_B_S_H_D, k_B_S_H_D, v_B_S_H_D)

    # NaViT 打包路径的非 xformers 后端：``attn_mask`` 是 ``_SegLens``（纯段长元数据）。
    # 语义与 BlockDiagonalMask 完全相同，只是换 kernel。
    if isinstance(attn_mask, _SegLens):
        if _PACKED_ATTN_BACKEND == "npu_tnd":
            out = _packed_attention_npu_tnd(q_B_S_H_D, k_B_S_H_D, v_B_S_H_D, attn_mask)
        else:
            out = _packed_attention_seg(q_B_S_H_D, k_B_S_H_D, v_B_S_H_D, attn_mask)
        return rearrange(out, "b s h d -> b s (h d)")

    if _is_xformers_attn_bias(attn_mask):
        if xops is None:
            raise RuntimeError(
                "block-diagonal attention bias requires xformers, but xformers.ops "
                "is unavailable. Disable packed/NaViT training or install xformers."
            )
        out = xops.memory_efficient_attention(
            q_B_S_H_D, k_B_S_H_D, v_B_S_H_D, attn_bias=attn_mask
        )
        return rearrange(out, "b s h d -> b s (h d)")

    if attn_mask is None and _USE_XFORMERS and xops is not None:
        try:
            out = xops.memory_efficient_attention(q_B_S_H_D, k_B_S_H_D, v_B_S_H_D)
            return rearrange(out, "b s h d -> b s (h d)")
        except Exception:
            pass
    in_q_shape = q_B_S_H_D.shape
    in_k_shape = k_B_S_H_D.shape
    q_B_H_S_D = rearrange(q_B_S_H_D, "b ... h k -> b h ... k").view(in_q_shape[0], in_q_shape[-2], -1, in_q_shape[-1])
    k_B_H_S_D = rearrange(k_B_S_H_D, "b ... h v -> b h ... v").view(in_k_shape[0], in_k_shape[-2], -1, in_k_shape[-1])
    v_B_H_S_D = rearrange(v_B_S_H_D, "b ... h v -> b h ... v").view(in_k_shape[0], in_k_shape[-2], -1, in_k_shape[-1])
    # 昇腾：同上，DiT 侧若拿到 [B,H,1,Skv] 的 padding mask 也要先展开（CUDA 上是恒等）
    from utils.npu_compat import expand_attn_mask as _expand_attn_mask
    attn_mask = _expand_attn_mask(attn_mask, q_B_H_S_D.shape[-2])
    result_B_S_HD = rearrange(
        torch.nn.functional.scaled_dot_product_attention(
            q_B_H_S_D, k_B_H_S_D, v_B_H_S_D, attn_mask=attn_mask
        ),
        "b h ... l -> b ... (h l)",
    )

    return result_B_S_HD


class Attention(nn.Module):
    """
    A flexible attention module supporting both self-attention and cross-attention mechanisms.

    This module implements a multi-head attention layer that can operate in either self-attention
    or cross-attention mode. The mode is determined by whether a context dimension is provided.
    The implementation uses scaled dot-product attention and supports optional bias terms and
    dropout regularization.

    Args:
        query_dim (int): The dimensionality of the query vectors.
        context_dim (int, optional): The dimensionality of the context (key/value) vectors.
            If None, the module operates in self-attention mode using query_dim. Default: None
        n_heads (int, optional): Number of attention heads for multi-head attention. Default: 8
        head_dim (int, optional): The dimension of each attention head. Default: 64
        dropout (float, optional): Dropout probability applied to the output. Default: 0.0
        qkv_format (str, optional): Format specification for QKV tensors. Default: "bshd"
        backend (str, optional): Backend to use for the attention operation. Default: "transformer_engine"

    Examples:
        >>> # Self-attention with 512 dimensions and 8 heads
        >>> self_attn = Attention(query_dim=512)
        >>> x = torch.randn(32, 16, 512)  # (batch_size, seq_len, dim)
        >>> out = self_attn(x)  # (32, 16, 512)

        >>> # Cross-attention
        >>> cross_attn = Attention(query_dim=512, context_dim=256)
        >>> query = torch.randn(32, 16, 512)
        >>> context = torch.randn(32, 8, 256)
        >>> out = cross_attn(query, context)  # (32, 16, 512)
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        n_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
        qkv_format: str = "bshd",
        backend: str = 'torch',
    ) -> None:
        super().__init__()
        self.is_selfattn = context_dim is None  # self attention

        self.backend = 'torch'

        context_dim = query_dim if context_dim is None else context_dim
        inner_dim = head_dim * n_heads

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.qkv_format = qkv_format
        self.query_dim = query_dim
        self.context_dim = context_dim

        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=1e-6)

        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6)

        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.v_norm = nn.Identity()

        self.output_proj = nn.Linear(inner_dim, query_dim, bias=False)
        self.output_dropout = nn.Dropout(dropout) if dropout > 1e-4 else nn.Identity()
        self.backend = 'torch'
        self.attn_op = torch_attention_op

        self._query_dim = query_dim
        self._context_dim = context_dim
        self._inner_dim = inner_dim
        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self._query_dim)
        torch.nn.init.trunc_normal_(self.q_proj.weight, std=std, a=-3 * std, b=3 * std)
        std = 1.0 / math.sqrt(self._context_dim)
        torch.nn.init.trunc_normal_(self.k_proj.weight, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.v_proj.weight, std=std, a=-3 * std, b=3 * std)

        std = 1.0 / math.sqrt(self._inner_dim)
        torch.nn.init.trunc_normal_(self.output_proj.weight, std=std, a=-3 * std, b=3 * std)

        for layer in self.q_norm, self.k_norm, self.v_norm:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def compute_qkv(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        rope_emb: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.q_proj(x)
        context = x if context is None else context
        k = self.k_proj(context)
        v = self.v_proj(context)
        q, k, v = map(
            lambda t: rearrange(t, "b ... (h d) -> b ... h d", h=self.n_heads, d=self.head_dim),
            (q, k, v),
        )

        def apply_norm_and_rotary_pos_emb(
            q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, rope_emb: Optional[torch.Tensor]
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            q = self.q_norm(q)
            k = self.k_norm(k)
            v = self.v_norm(v)
            if self.is_selfattn and rope_emb is not None:  # only apply to self-attention!
                q = apply_rotary_pos_emb(q, rope_emb, tensor_format=self.qkv_format, fused=False)
                k = apply_rotary_pos_emb(k, rope_emb, tensor_format=self.qkv_format, fused=False)
            return q, k, v

        q, k, v = apply_norm_and_rotary_pos_emb(q, k, v, rope_emb)

        return q, k, v

    def compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        result = self.attn_op(q, k, v, attn_mask=attn_mask)  # [B, S, H, D]
        return self.output_dropout(self.output_proj(result))

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        rope_emb: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x (Tensor): The query tensor of shape [B, Mq, K]
            context (Optional[Tensor]): The key tensor of shape [B, Mk, K] or use x as context [self attention] if None
            rope_emb (Optional[Tensor]): RoPE embedding tensor, or no RoPE embeddings (i.e. in cross attention)
        """
        q, k, v = self.compute_qkv(x, context, rope_emb=rope_emb)
        return self.compute_attention(q, k, v, attn_mask=attn_mask)


class VideoPositionEmb(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._cp_group = None

    @property
    def seq_dim(self) -> int:
        return 1

    def forward(self, x_B_T_H_W_C: torch.Tensor, fps: Optional[torch.Tensor]) -> torch.Tensor:
        """
        With CP, the function assume that the input tensor is already split.
        It delegates the embedding generation to generate_embeddings function.
        """
        B_T_H_W_C = x_B_T_H_W_C.shape
        embeddings = self.generate_embeddings(B_T_H_W_C, fps=fps)

        return embeddings

    def generate_embeddings(self, B_T_H_W_C: torch.Size, fps: Optional[torch.Tensor]) -> Any:
        raise NotImplementedError


class VideoRopePosition3DEmb(VideoPositionEmb):
    def __init__(
        self,
        *,  # enforce keyword arguments
        head_dim: int,
        len_h: int,
        len_w: int,
        len_t: int,
        base_fps: int = 24,
        h_extrapolation_ratio: float = 1.0,
        w_extrapolation_ratio: float = 1.0,
        t_extrapolation_ratio: float = 1.0,
        **kwargs,  # used for compatibility with other positional embeddings; unused in this class
    ):
        del kwargs
        super().__init__()
        self.register_buffer("seq", torch.arange(max(len_h, len_w, len_t), dtype=torch.float))
        self.base_fps = base_fps
        self.max_h = len_h
        self.max_w = len_w
        self.max_t = len_t
        dim = head_dim
        dim_h = dim // 6 * 2
        dim_w = dim_h
        dim_t = dim - 2 * dim_h
        assert dim == dim_h + dim_w + dim_t, f"bad dim: {dim} != {dim_h} + {dim_w} + {dim_t}"
        self.register_buffer(
            "dim_spatial_range",
            torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h,
            persistent=True,
        )
        self.register_buffer(
            "dim_temporal_range",
            torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t,
            persistent=True,
        )
        self._dim_h = dim_h
        self._dim_t = dim_t

        self.h_ntk_factor = h_extrapolation_ratio ** (dim_h / (dim_h - 2))
        self.w_ntk_factor = w_extrapolation_ratio ** (dim_w / (dim_w - 2))
        self.t_ntk_factor = t_extrapolation_ratio ** (dim_t / (dim_t - 2))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        dim_h = self._dim_h
        dim_t = self._dim_t

        self.seq = torch.arange(max(self.max_h, self.max_w, self.max_t)).float().to(self.dim_spatial_range.device)
        self.dim_spatial_range = (
            torch.arange(0, dim_h, 2)[: (dim_h // 2)].float().to(self.dim_spatial_range.device) / dim_h
        )
        self.dim_temporal_range = (
            torch.arange(0, dim_t, 2)[: (dim_t // 2)].float().to(self.dim_spatial_range.device) / dim_t
        )

    def generate_embeddings(
        self,
        B_T_H_W_C: torch.Size,
        fps: Optional[torch.Tensor] = None,
        h_ntk_factor: Optional[float] = None,
        w_ntk_factor: Optional[float] = None,
        t_ntk_factor: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Generate embeddings for the given input size.

        Args:
            B_T_H_W_C (torch.Size): Input tensor size (Batch, Time, Height, Width, Channels).
            fps (Optional[torch.Tensor], optional): Frames per second. Defaults to None.
            h_ntk_factor (Optional[float], optional): Height NTK factor. If None, uses self.h_ntk_factor.
            w_ntk_factor (Optional[float], optional): Width NTK factor. If None, uses self.w_ntk_factor.
            t_ntk_factor (Optional[float], optional): Time NTK factor. If None, uses self.t_ntk_factor.

        Returns:
            Not specified in the original code snippet.
        """
        h_ntk_factor = h_ntk_factor if h_ntk_factor is not None else self.h_ntk_factor
        w_ntk_factor = w_ntk_factor if w_ntk_factor is not None else self.w_ntk_factor
        t_ntk_factor = t_ntk_factor if t_ntk_factor is not None else self.t_ntk_factor

        h_theta = 10000.0 * h_ntk_factor  # type: ignore
        w_theta = 10000.0 * w_ntk_factor  # type: ignore
        t_theta = 10000.0 * t_ntk_factor  # type: ignore

        h_spatial_freqs = 1.0 / (h_theta**self.dim_spatial_range)
        w_spatial_freqs = 1.0 / (w_theta**self.dim_spatial_range)
        temporal_freqs = 1.0 / (t_theta**self.dim_temporal_range)

        B, T, H, W, _ = B_T_H_W_C
        assert (
            H <= self.max_h and W <= self.max_w
        ), f"Input dimensions (H={H}, W={W}) exceed the maximum dimensions (max_h={self.max_h}, max_w={self.max_w})"
        half_emb_h = torch.outer(self.seq[:H], h_spatial_freqs)
        half_emb_w = torch.outer(self.seq[:W], w_spatial_freqs)
        half_emb_t = torch.outer(self.seq[:T], temporal_freqs)

        em_T_H_W_D = torch.cat(
            [
                repeat(half_emb_t, "t d -> t h w d", h=H, w=W),
                repeat(half_emb_h, "h d -> t h w d", t=T, w=W),
                repeat(half_emb_w, "w d -> t h w d", t=T, h=H),
            ]
            * 2,
            dim=-1,
        )

        return rearrange(em_T_H_W_D, "t h w d -> (t h w) 1 1 d").float()

    @property
    def seq_dim(self) -> int:
        return 0


class Timesteps(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps_B_T: torch.Tensor) -> torch.Tensor:
        assert timesteps_B_T.ndim == 2, f"Expected 2D input, got {timesteps_B_T.ndim}"
        in_dype = timesteps_B_T.dtype
        timesteps = timesteps_B_T.flatten().float()
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / (half_dim - 0.0)

        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]

        sin_emb = torch.sin(emb)
        cos_emb = torch.cos(emb)
        emb = torch.cat([cos_emb, sin_emb], dim=-1)

        return rearrange(emb.to(dtype=in_dype), "(b t) d -> b t d", b=timesteps_B_T.shape[0], t=timesteps_B_T.shape[1])


class TimestepEmbedding(nn.Module):
    def __init__(self, in_features: int, out_features: int, use_adaln_lora: bool = False):
        super().__init__()
        self.in_dim = in_features
        self.out_dim = out_features
        self.linear_1 = nn.Linear(in_features, out_features, bias=not use_adaln_lora)
        self.activation = nn.SiLU()
        self.use_adaln_lora = use_adaln_lora
        if use_adaln_lora:
            self.linear_2 = nn.Linear(out_features, 3 * out_features, bias=False)
        else:
            self.linear_2 = nn.Linear(out_features, out_features, bias=False)

        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.in_dim)
        torch.nn.init.trunc_normal_(self.linear_1.weight, std=std, a=-3 * std, b=3 * std)

        std = 1.0 / math.sqrt(self.out_dim)
        torch.nn.init.trunc_normal_(self.linear_2.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, sample: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        emb = self.linear_1(sample)
        emb = self.activation(emb)
        emb = self.linear_2(emb)

        if self.use_adaln_lora:
            adaln_lora_B_T_3D = emb
            emb_B_T_D = sample
        else:
            adaln_lora_B_T_3D = None
            emb_B_T_D = emb

        return emb_B_T_D, adaln_lora_B_T_3D


class PatchEmbed(nn.Module):
    """
    PatchEmbed is a module for embedding patches from an input tensor by applying either 3D or 2D convolutional layers,
    depending on the . This module can process inputs with temporal (video) and spatial (image) dimensions,
    making it suitable for video and image processing tasks. It supports dividing the input into patches
    and embedding each patch into a vector of size `out_channels`.

    Parameters:
    - spatial_patch_size (int): The size of each spatial patch.
    - temporal_patch_size (int): The size of each temporal patch.
    - in_channels (int): Number of input channels. Default: 3.
    - out_channels (int): The dimension of the embedding vector for each patch. Default: 768.
    - bias (bool): If True, adds a learnable bias to the output of the convolutional layers. Default: True.
    """

    def __init__(
        self,
        spatial_patch_size: int,
        temporal_patch_size: int,
        in_channels: int = 3,
        out_channels: int = 768,
    ):
        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size

        self.proj = nn.Sequential(
            Rearrange(
                "b c (t r) (h m) (w n) -> b t h w (c r m n)",
                r=temporal_patch_size,
                m=spatial_patch_size,
                n=spatial_patch_size,
            ),
            nn.Linear(
                in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size, out_channels, bias=False
            ),
        )
        self.dim = in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size

        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.dim)
        torch.nn.init.trunc_normal_(self.proj[1].weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PatchEmbed module.

        Parameters:
        - x (torch.Tensor): The input tensor of shape (B, C, T, H, W) where
            B is the batch size,
            C is the number of channels,
            T is the temporal dimension,
            H is the height, and
            W is the width of the input.

        Returns:
        - torch.Tensor: The embedded patches as a tensor, with shape b t h w c.
        """
        assert x.dim() == 5
        _, _, T, H, W = x.shape
        assert (
            H % self.spatial_patch_size == 0 and W % self.spatial_patch_size == 0
        ), f"H,W {(H, W)} should be divisible by spatial_patch_size {self.spatial_patch_size}"
        assert T % self.temporal_patch_size == 0
        x = self.proj(x)
        return x


class FinalLayer(nn.Module):
    """
    The final layer of video DiT.
    """

    def __init__(
        self,
        hidden_size: int,
        spatial_patch_size: int,
        temporal_patch_size: int,
        out_channels: int,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, spatial_patch_size * spatial_patch_size * temporal_patch_size * out_channels, bias=False
        )
        self.hidden_size = hidden_size
        self.n_adaln_chunks = 2
        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim
        if use_adaln_lora:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, self.n_adaln_chunks * hidden_size, bias=False),
            )
        else:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, self.n_adaln_chunks * hidden_size, bias=False)
            )

        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.hidden_size)
        torch.nn.init.trunc_normal_(self.linear.weight, std=std, a=-3 * std, b=3 * std)
        if self.use_adaln_lora:
            torch.nn.init.trunc_normal_(self.adaln_modulation[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.adaln_modulation[2].weight)
        else:
            torch.nn.init.zeros_(self.adaln_modulation[1].weight)

        self.layer_norm.reset_parameters()

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
    ):
        if self.use_adaln_lora:
            assert adaln_lora_B_T_3D is not None
            shift_B_T_D, scale_B_T_D = (
                self.adaln_modulation(emb_B_T_D) + adaln_lora_B_T_3D[:, :, : 2 * self.hidden_size]
            ).chunk(2, dim=-1)
        else:
            shift_B_T_D, scale_B_T_D = self.adaln_modulation(emb_B_T_D).chunk(2, dim=-1)

        shift_B_T_1_1_D, scale_B_T_1_1_D = rearrange(shift_B_T_D, "b t d -> b t 1 1 d"), rearrange(
            scale_B_T_D, "b t d -> b t 1 1 d"
        )

        def _fn(
            _x_B_T_H_W_D: torch.Tensor,
            _norm_layer: nn.Module,
            _scale_B_T_1_1_D: torch.Tensor,
            _shift_B_T_1_1_D: torch.Tensor,
        ) -> torch.Tensor:
            return _norm_layer(_x_B_T_H_W_D) * (1 + _scale_B_T_1_1_D) + _shift_B_T_1_1_D

        x_B_T_H_W_D = _fn(x_B_T_H_W_D, self.layer_norm, scale_B_T_1_1_D, shift_B_T_1_1_D)
        x_B_T_H_W_O = self.linear(x_B_T_H_W_D)
        return x_B_T_H_W_O

    def forward_tokens(
        self,
        x_B_N_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
        token_wise_mod: bool = False,
        mod_index: Optional[torch.Tensor] = None,
    ):
        # ``token_wise_mod`` is the NaViT/FiT packing path: each packed image carries its
        # own timestep, so AdaLN shift/scale must vary per token rather than broadcast
        # from a single ``[:, :1, :]`` slot. Two layouts:
        #
        # * ``mod_index`` given（NaViT 默认）: ``emb``/``adaln_lora`` are *per-image*
        #   ``[1, G, *]`` and ``mod_index`` ``[ΣN]`` maps each token to its image row —
        #   the modulation MLP runs on G rows only, then each chunk is gathered to a
        #   contiguous per-token tensor（同数学、免 ΣN 行 matmul 与跨 chunk 条带视图）。
        # * ``mod_index=None``: legacy per-token layout — inputs are already
        #   ``repeat_interleave`` 展开的 ``[1, ΣN, *]``，直接使用。
        #
        # Default ``False``/None keeps every existing caller (constant-N / token-bucket,
        # where the whole row shares one timestep) byte-identical.
        if self.use_adaln_lora:
            assert adaln_lora_B_T_3D is not None
            shift_B_T_D, scale_B_T_D = (
                self.adaln_modulation(emb_B_T_D) + adaln_lora_B_T_3D[:, :, : 2 * self.hidden_size]
            ).chunk(2, dim=-1)
        else:
            shift_B_T_D, scale_B_T_D = self.adaln_modulation(emb_B_T_D).chunk(2, dim=-1)

        if token_wise_mod and mod_index is not None:
            shift_mod = shift_B_T_D.index_select(1, mod_index)
            scale_mod = scale_B_T_D.index_select(1, mod_index)
        elif token_wise_mod:
            shift_mod, scale_mod = shift_B_T_D, scale_B_T_D
        else:
            shift_mod = shift_B_T_D[:, :1, :]
            scale_mod = scale_B_T_D[:, :1, :]
        x_B_N_D = self.layer_norm(x_B_N_D) * (1 + scale_mod) + shift_mod
        return self.linear(x_B_N_D)


class Block(nn.Module):
    """
    A transformer block that combines self-attention, cross-attention and MLP layers with AdaLN modulation.
    Each component (self-attention, cross-attention, MLP) has its own layer normalization and AdaLN modulation.

    Parameters:
        x_dim (int): Dimension of input features
        context_dim (int): Dimension of context features for cross-attention
        num_heads (int): Number of attention heads
        mlp_ratio (float): Multiplier for MLP hidden dimension. Default: 4.0
        use_adaln_lora (bool): Whether to use AdaLN-LoRA modulation. Default: False
        adaln_lora_dim (int): Hidden dimension for AdaLN-LoRA layers. Default: 256

    The block applies the following sequence:
    1. Self-attention with AdaLN modulation
    2. Cross-attention with AdaLN modulation
    3. MLP with AdaLN modulation

    Each component uses skip connections and layer normalization.
    """

    def __init__(
        self,
        x_dim: int,
        context_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        self_attention_backend: str = 'torch',
        cross_attention_backend: str = 'torch',
    ):
        super().__init__()
        self.x_dim = x_dim
        self.layer_norm_self_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = Attention(
            x_dim,
            None,
            num_heads,
            x_dim // num_heads,
            qkv_format="bshd",
            backend=self_attention_backend,
        )

        self.layer_norm_cross_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = Attention(
            x_dim, context_dim, num_heads, x_dim // num_heads, qkv_format="bshd", backend=cross_attention_backend
        )

        self.layer_norm_mlp = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = GPT2FeedForward(x_dim, int(x_dim * mlp_ratio))

        self.use_adaln_lora = use_adaln_lora
        if self.use_adaln_lora:
            self.adaln_modulation_self_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_cross_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
        else:
            self.adaln_modulation_self_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))
            self.adaln_modulation_cross_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))
            self.adaln_modulation_mlp = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))

    def reset_parameters(self) -> None:
        self.layer_norm_self_attn.reset_parameters()
        self.layer_norm_cross_attn.reset_parameters()
        self.layer_norm_mlp.reset_parameters()

        if self.use_adaln_lora:
            std = 1.0 / math.sqrt(self.x_dim)
            torch.nn.init.trunc_normal_(self.adaln_modulation_self_attn[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.trunc_normal_(self.adaln_modulation_cross_attn[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.trunc_normal_(self.adaln_modulation_mlp[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.adaln_modulation_self_attn[2].weight)
            torch.nn.init.zeros_(self.adaln_modulation_cross_attn[2].weight)
            torch.nn.init.zeros_(self.adaln_modulation_mlp[2].weight)
        else:
            torch.nn.init.zeros_(self.adaln_modulation_self_attn[1].weight)
            torch.nn.init.zeros_(self.adaln_modulation_cross_attn[1].weight)
            torch.nn.init.zeros_(self.adaln_modulation_mlp[1].weight)

    def init_weights(self) -> None:
        self.reset_parameters()
        self.self_attn.init_weights()
        self.cross_attn.init_weights()
        self.mlp.init_weights()

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
        extra_per_block_pos_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if extra_per_block_pos_emb is not None:
            x_B_T_H_W_D = x_B_T_H_W_D + extra_per_block_pos_emb

        if self.use_adaln_lora:
            shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = (
                self.adaln_modulation_self_attn(emb_B_T_D) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
            shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = (
                self.adaln_modulation_cross_attn(emb_B_T_D) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
            shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = (
                self.adaln_modulation_mlp(emb_B_T_D) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
        else:
            shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = self.adaln_modulation_self_attn(
                emb_B_T_D
            ).chunk(3, dim=-1)
            shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = self.adaln_modulation_cross_attn(
                emb_B_T_D
            ).chunk(3, dim=-1)
            shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = self.adaln_modulation_mlp(emb_B_T_D).chunk(3, dim=-1)

        # Reshape tensors from (B, T, D) to (B, T, 1, 1, D) for broadcasting
        shift_self_attn_B_T_1_1_D = rearrange(shift_self_attn_B_T_D, "b t d -> b t 1 1 d")
        scale_self_attn_B_T_1_1_D = rearrange(scale_self_attn_B_T_D, "b t d -> b t 1 1 d")
        gate_self_attn_B_T_1_1_D = rearrange(gate_self_attn_B_T_D, "b t d -> b t 1 1 d")

        shift_cross_attn_B_T_1_1_D = rearrange(shift_cross_attn_B_T_D, "b t d -> b t 1 1 d")
        scale_cross_attn_B_T_1_1_D = rearrange(scale_cross_attn_B_T_D, "b t d -> b t 1 1 d")
        gate_cross_attn_B_T_1_1_D = rearrange(gate_cross_attn_B_T_D, "b t d -> b t 1 1 d")

        shift_mlp_B_T_1_1_D = rearrange(shift_mlp_B_T_D, "b t d -> b t 1 1 d")
        scale_mlp_B_T_1_1_D = rearrange(scale_mlp_B_T_D, "b t d -> b t 1 1 d")
        gate_mlp_B_T_1_1_D = rearrange(gate_mlp_B_T_D, "b t d -> b t 1 1 d")

        B, T, H, W, D = x_B_T_H_W_D.shape

        def _fn(_x_B_T_H_W_D, _norm_layer, _scale_B_T_1_1_D, _shift_B_T_1_1_D):
            return _norm_layer(_x_B_T_H_W_D) * (1 + _scale_B_T_1_1_D) + _shift_B_T_1_1_D

        normalized_x_B_T_H_W_D = _fn(
            x_B_T_H_W_D,
            self.layer_norm_self_attn,
            scale_self_attn_B_T_1_1_D,
            shift_self_attn_B_T_1_1_D,
        )
        result_B_T_H_W_D = rearrange(
            self.self_attn(
                # normalized_x_B_T_HW_D,
                rearrange(normalized_x_B_T_H_W_D, "b t h w d -> b (t h w) d"),
                None,
                rope_emb=rope_emb_L_1_1_D,
            ),
            "b (t h w) d -> b t h w d",
            t=T,
            h=H,
            w=W,
        )
        x_B_T_H_W_D = x_B_T_H_W_D + gate_self_attn_B_T_1_1_D * result_B_T_H_W_D

        def _x_fn(
            _x_B_T_H_W_D: torch.Tensor,
            layer_norm_cross_attn: Callable,
            _scale_cross_attn_B_T_1_1_D: torch.Tensor,
            _shift_cross_attn_B_T_1_1_D: torch.Tensor,
        ) -> torch.Tensor:
            _normalized_x_B_T_H_W_D = _fn(
                _x_B_T_H_W_D, layer_norm_cross_attn, _scale_cross_attn_B_T_1_1_D, _shift_cross_attn_B_T_1_1_D
            )
            _result_B_T_H_W_D = rearrange(
                self.cross_attn(
                    rearrange(_normalized_x_B_T_H_W_D, "b t h w d -> b (t h w) d"),
                    crossattn_emb,
                    rope_emb=rope_emb_L_1_1_D,
                ),
                "b (t h w) d -> b t h w d",
                t=T,
                h=H,
                w=W,
            )
            return _result_B_T_H_W_D

        result_B_T_H_W_D = _x_fn(
            x_B_T_H_W_D,
            self.layer_norm_cross_attn,
            scale_cross_attn_B_T_1_1_D,
            shift_cross_attn_B_T_1_1_D,
        )
        x_B_T_H_W_D = result_B_T_H_W_D * gate_cross_attn_B_T_1_1_D + x_B_T_H_W_D

        normalized_x_B_T_H_W_D = _fn(
            x_B_T_H_W_D,
            self.layer_norm_mlp,
            scale_mlp_B_T_1_1_D,
            shift_mlp_B_T_1_1_D,
        )
        result_B_T_H_W_D = self.mlp(normalized_x_B_T_H_W_D)
        x_B_T_H_W_D = x_B_T_H_W_D + gate_mlp_B_T_1_1_D * result_B_T_H_W_D
        return x_B_T_H_W_D

    def forward_tokens(
        self,
        x_B_N_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        token_mask_f: Optional[torch.Tensor] = None,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
        token_wise_mod: bool = False,
        mod_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # ``attn_mask`` (additive key-padding mask) and ``token_mask_f`` (float
        # zeroing mask) are precomputed once per step by the caller via
        # ``MiniTrainDIT._build_packed_masks`` — not rebuilt here per block. Both are
        # None when every token is valid (constant-N / token-bucket), so attention
        # takes SDPA's fast maskless path and no output zeroing is needed.
        #
        # NaViT/FiT packing path (``token_wise_mod=True``): ``attn_mask`` and
        # ``cross_attn_mask`` are xformers ``BlockDiagonalMask`` biases (per-image
        # self / cross seqlens). AdaLN shift/scale/gate vary per token（每图各自 t），
        # 有两种输入布局：
        #
        # * ``mod_index`` given（NaViT 默认）: ``emb_B_T_D``/``adaln_lora_B_T_3D`` 是
        #   *per-image* ``[1, G, *]``，``mod_index`` ``[ΣN]`` 把每个 token 映射到所属图行。
        #   三个调制 MLP 只在 G 行上跑，各 chunk 经 ``index_select`` gather 成**连续**
        #   逐 token 张量。与逐 token 布局同数学（同一行同值），省掉 ΣN 行调制 matmul
        #   与喂给每个 AdaLN 逐元素 op 的跨 chunk 条带视图（本地交错实测前向中位 −13%，
        #   RTX 5070 Laptop；云端占比待 stage_timing 验证）。
        # * ``mod_index=None``: legacy 逐 token 布局——输入已 ``repeat_interleave`` 展开
        #   为 ``[1, ΣN, *]``，按原样使用。
        #
        # All defaults are None/False, so constant-N / token-bucket callers are
        # byte-identical.
        if self.use_adaln_lora:
            assert adaln_lora_B_T_3D is not None
            shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = (
                self.adaln_modulation_self_attn(emb_B_T_D) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
            shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = (
                self.adaln_modulation_cross_attn(emb_B_T_D) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
            shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = (
                self.adaln_modulation_mlp(emb_B_T_D) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
        else:
            shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = self.adaln_modulation_self_attn(
                emb_B_T_D
            ).chunk(3, dim=-1)
            shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = self.adaln_modulation_cross_attn(
                emb_B_T_D
            ).chunk(3, dim=-1)
            shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = self.adaln_modulation_mlp(emb_B_T_D).chunk(3, dim=-1)

        if token_wise_mod and mod_index is not None:
            _sel = lambda t: t.index_select(1, mod_index)   # [1,G,D] → 连续 [1,ΣN,D]
        elif token_wise_mod:
            _sel = lambda t: t
        else:
            _sel = lambda t: t[:, :1, :]
        shift_self_attn_B_1_D = _sel(shift_self_attn_B_T_D)
        scale_self_attn_B_1_D = _sel(scale_self_attn_B_T_D)
        gate_self_attn_B_1_D = _sel(gate_self_attn_B_T_D)
        shift_cross_attn_B_1_D = _sel(shift_cross_attn_B_T_D)
        scale_cross_attn_B_1_D = _sel(scale_cross_attn_B_T_D)
        gate_cross_attn_B_1_D = _sel(gate_cross_attn_B_T_D)
        shift_mlp_B_1_D = _sel(shift_mlp_B_T_D)
        scale_mlp_B_1_D = _sel(scale_mlp_B_T_D)
        gate_mlp_B_1_D = _sel(gate_mlp_B_T_D)

        def _fn(_x_B_N_D, _norm_layer, _scale_B_1_D, _shift_B_1_D):
            return _norm_layer(_x_B_N_D) * (1 + _scale_B_1_D) + _shift_B_1_D

        normalized_x_B_N_D = _fn(
            x_B_N_D,
            self.layer_norm_self_attn,
            scale_self_attn_B_1_D,
            shift_self_attn_B_1_D,
        )
        result_B_N_D = self.self_attn(
            normalized_x_B_N_D,
            None,
            rope_emb=rope_emb_L_1_1_D,
            attn_mask=attn_mask,
        )
        x_B_N_D = x_B_N_D + gate_self_attn_B_1_D * result_B_N_D

        normalized_x_B_N_D = _fn(
            x_B_N_D,
            self.layer_norm_cross_attn,
            scale_cross_attn_B_1_D,
            shift_cross_attn_B_1_D,
        )
        result_B_N_D = self.cross_attn(
            normalized_x_B_N_D, crossattn_emb, rope_emb=None, attn_mask=cross_attn_mask
        )
        x_B_N_D = result_B_N_D * gate_cross_attn_B_1_D + x_B_N_D

        normalized_x_B_N_D = _fn(
            x_B_N_D,
            self.layer_norm_mlp,
            scale_mlp_B_1_D,
            shift_mlp_B_1_D,
        )
        result_B_N_D = self.mlp(normalized_x_B_N_D)
        x_B_N_D = x_B_N_D + gate_mlp_B_1_D * result_B_N_D
        if token_mask_f is not None:
            x_B_N_D = x_B_N_D * token_mask_f
        return x_B_N_D


class MiniTrainDIT(nn.Module):
    """
    A clean impl of DIT that can load and  reproduce the training results of the original DIT model in~(cosmos 1)
    A general implementation of adaln-modulated VIT-like~(DiT) transformer for video processing.

    Args:
        max_img_h (int): Maximum height of the input images.
        max_img_w (int): Maximum width of the input images.
        max_frames (int): Maximum number of frames in the video sequence.
        in_channels (int): Number of input channels (e.g., RGB channels for color images).
        out_channels (int): Number of output channels.
        patch_spatial (tuple): Spatial resolution of patches for input processing.
        patch_temporal (int): Temporal resolution of patches for input processing.
        concat_padding_mask (bool): If True, includes a mask channel in the input to handle padding.
        model_channels (int): Base number of channels used throughout the model.
        num_blocks (int): Number of transformer blocks.
        num_heads (int): Number of heads in the multi-head attention layers.
        mlp_ratio (float): Expansion ratio for MLP blocks.
        crossattn_emb_channels (int): Number of embedding channels for cross-attention.
        pos_emb_cls (str): Type of positional embeddings.
        pos_emb_learnable (bool): Whether positional embeddings are learnable.
        pos_emb_interpolation (str): Method for interpolating positional embeddings.
        min_fps (int): Minimum frames per second.
        max_fps (int): Maximum frames per second.
        use_adaln_lora (bool): Whether to use AdaLN-LoRA.
        adaln_lora_dim (int): Dimension for AdaLN-LoRA.
        rope_h_extrapolation_ratio (float): Height extrapolation ratio for RoPE.
        rope_w_extrapolation_ratio (float): Width extrapolation ratio for RoPE.
        rope_t_extrapolation_ratio (float): Temporal extrapolation ratio for RoPE.
        extra_h_extrapolation_ratio (float): Height extrapolation ratio for extra embeddings.
        extra_w_extrapolation_ratio (float): Width extrapolation ratio for extra embeddings.
        extra_t_extrapolation_ratio (float): Temporal extrapolation ratio for extra embeddings.
    """

    def __init__(
        self,
        max_img_h: int,
        max_img_w: int,
        max_frames: int,
        in_channels: int,
        out_channels: int,
        patch_spatial: int,  # tuple,
        patch_temporal: int,
        concat_padding_mask: bool = True,
        # attention settings
        model_channels: int = 768,
        num_blocks: int = 10,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        # cross attention settings
        crossattn_emb_channels: int = 1024,
        # positional embedding settings
        pos_emb_cls: str = "sincos",
        pos_emb_learnable: bool = False,
        pos_emb_interpolation: str = "crop",
        min_fps: int = 1,
        max_fps: int = 30,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        rope_h_extrapolation_ratio: float = 1.0,
        rope_w_extrapolation_ratio: float = 1.0,
        rope_t_extrapolation_ratio: float = 1.0,
        extra_per_block_abs_pos_emb: bool = False,
        extra_h_extrapolation_ratio: float = 1.0,
        extra_w_extrapolation_ratio: float = 1.0,
        extra_t_extrapolation_ratio: float = 1.0,
        rope_enable_fps_modulation: bool = False,
    ) -> None:
        atten_backend = 'torch'

        super().__init__()
        self._blocks_compiled = False
        self.max_img_h = max_img_h
        self.max_img_w = max_img_w
        self.max_frames = max_frames
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.model_channels = model_channels
        self.concat_padding_mask = concat_padding_mask
        self.atten_backend = atten_backend
        # positional embedding settings
        self.pos_emb_cls = pos_emb_cls
        self.pos_emb_learnable = pos_emb_learnable
        self.pos_emb_interpolation = pos_emb_interpolation
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.rope_h_extrapolation_ratio = rope_h_extrapolation_ratio
        self.rope_w_extrapolation_ratio = rope_w_extrapolation_ratio
        self.rope_t_extrapolation_ratio = rope_t_extrapolation_ratio
        self.cuda_graphs = {}

        self.build_patch_embed()
        self.build_pos_embed()
        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim
        self.t_embedder = nn.Sequential(
            Timesteps(model_channels),
            TimestepEmbedding(model_channels, model_channels, use_adaln_lora=use_adaln_lora),
        )

        self.blocks = nn.ModuleList(
            [
                Block(
                    x_dim=model_channels,
                    context_dim=crossattn_emb_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_adaln_lora=use_adaln_lora,
                    adaln_lora_dim=adaln_lora_dim,
                    self_attention_backend=atten_backend,
                    cross_attention_backend=atten_backend,
                )
                for _ in range(num_blocks)
            ]
        )

        self.final_layer = FinalLayer(
            hidden_size=self.model_channels,
            spatial_patch_size=self.patch_spatial,
            temporal_patch_size=self.patch_temporal,
            out_channels=self.out_channels,
            use_adaln_lora=self.use_adaln_lora,
            adaln_lora_dim=self.adaln_lora_dim,
        )

        self.t_embedding_norm = RMSNorm(model_channels, eps=1e-6)
        self.init_weights()

    def init_weights(self) -> None:
        self.x_embedder.init_weights()
        self.pos_embedder.reset_parameters()

        self.t_embedder[1].init_weights()
        for block in self.blocks:
            block.init_weights()

        self.final_layer.init_weights()
        self.t_embedding_norm.reset_parameters()

    def build_patch_embed(self) -> None:
        (
            concat_padding_mask,
            in_channels,
            patch_spatial,
            patch_temporal,
            model_channels,
        ) = (
            self.concat_padding_mask,
            self.in_channels,
            self.patch_spatial,
            self.patch_temporal,
            self.model_channels,
        )
        in_channels = in_channels + 1 if concat_padding_mask else in_channels
        self.x_embedder = PatchEmbed(
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            in_channels=in_channels,
            out_channels=model_channels,
        )

    def build_pos_embed(self) -> None:
        if self.pos_emb_cls == "rope3d":
            cls_type = VideoRopePosition3DEmb
        else:
            raise ValueError(f"Unknown pos_emb_cls {self.pos_emb_cls}")

        kwargs = dict(
            model_channels=self.model_channels,
            len_h=self.max_img_h // self.patch_spatial,
            len_w=self.max_img_w // self.patch_spatial,
            len_t=self.max_frames // self.patch_temporal,
            max_fps=self.max_fps,
            min_fps=self.min_fps,
            is_learnable=self.pos_emb_learnable,
            interpolation=self.pos_emb_interpolation,
            head_dim=self.model_channels // self.num_heads,
            h_extrapolation_ratio=self.rope_h_extrapolation_ratio,
            w_extrapolation_ratio=self.rope_w_extrapolation_ratio,
            t_extrapolation_ratio=self.rope_t_extrapolation_ratio,
            enable_fps_modulation=False,
        )
        self.pos_embedder = cls_type(
            **kwargs,  # type: ignore
        )


    def compile_blocks(self, backend: str = "inductor", mode=None, dynamic=False):
        """Per-block torch.compile of the token forward path (``block.forward_tokens``).

        Intended for constant / N-token bucketing: when the packed sequence length is
        fixed across the run, each compiled block traces a tiny fixed set of graphs.
        The eager grid ``forward`` is unaffected (only ``forward_tokens`` is wrapped).
        ``backend='eager'`` validates compile-compatibility without Inductor/Triton;
        ``backend='inductor'`` is for the real speedup on CUDA/Linux.

        ``dynamic`` controls shape specialization (passed straight to ``torch.compile``):
        - ``False`` (historical default): every distinct input shape — including each
          distinct packed sequence length N *and* each distinct batch size B — is a
          separate static graph. Best per-step kernels, but if the dataset has several
          native token counts or ``bucket_drop_last=false`` produces variable-size
          remainder batches, the graph count explodes (and past ``cache_size_limit`` it
          recompiles/falls back every step). Use only when N and B are truly fixed.
        - ``None`` (auto): specialize the first shape statically, then mark the varying
          dim symbolic on the second distinct shape → one or two graphs cover all token
          counts and batch sizes. Recommended when the dataset has multiple native token
          counts (no resampling needed) or B varies.
        - ``True``: compile dynamic-shape kernels from the start (one graph, no static
          warmup specialization).
        """
        import torch._dynamo as _dynamo
        self._blocks_compiled = True
        _dynamo.config.cache_size_limit = max(_dynamo.config.cache_size_limit, 32)
        kwargs = {"backend": backend, "dynamic": dynamic}
        if mode is not None:
            kwargs["mode"] = mode
        for block in self.blocks:
            block.forward_tokens = torch.compile(block.forward_tokens, **kwargs)
        import logging
        logging.getLogger(__name__).info(
            "compile_blocks: compiled %d block.forward_tokens (backend=%s, mode=%s, dynamic=%s)",
            len(self.blocks), backend, mode, dynamic,
        )

    def prepare_embedded_sequence(
        self,
        x_B_C_T_H_W: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Prepares an embedded sequence tensor by applying positional embeddings and handling padding masks.

        Args:
            x_B_C_T_H_W (torch.Tensor): video
            fps (Optional[torch.Tensor]): Frames per second tensor to be used for positional embedding when required.
                                    If None, a default value (`self.base_fps`) will be used.
            padding_mask (Optional[torch.Tensor]): current it is not used

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - A tensor of shape (B, T, H, W, D) with the embedded sequence.
                - An optional positional embedding tensor, returned only if the positional embedding class
                (`self.pos_emb_cls`) includes 'rope'. Otherwise, None.

        Notes:
            - If `self.concat_padding_mask` is True, a padding mask channel is concatenated to the input tensor.
            - The method of applying positional embeddings depends on the value of `self.pos_emb_cls`.
            - If 'rope' is in `self.pos_emb_cls` (case insensitive), the positional embeddings are generated using
                the `self.pos_embedder` with the shape [T, H, W].
            - If "fps_aware" is in `self.pos_emb_cls`, the positional embeddings are generated using the
            `self.pos_embedder` with the fps tensor.
            - Otherwise, the positional embeddings are generated without considering fps.
        """
        if self.concat_padding_mask:
            padding_mask = F.interpolate(
                padding_mask,
                size=list(x_B_C_T_H_W.shape[-2:]),
                mode="nearest",
            )
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)], dim=1
            )
        x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)

        extra_pos_emb = None

        if "rope" in self.pos_emb_cls.lower():
            return x_B_T_H_W_D, self.pos_embedder(x_B_T_H_W_D, fps=fps), extra_pos_emb
        x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D)  # [B, T, H, W, D]

        return x_B_T_H_W_D, None, extra_pos_emb

    def unpatchify(self, x_B_T_H_W_M: torch.Tensor) -> torch.Tensor:
        x_B_C_Tt_Hp_Wp = rearrange(
            x_B_T_H_W_M,
            "B T H W (p1 p2 t C) -> B C (T t) (H p1) (W p2)",
            p1=self.patch_spatial,
            p2=self.patch_spatial,
            t=self.patch_temporal,
        )
        return x_B_C_Tt_Hp_Wp

    def patchify_latents_to_tokens(
        self,
        x_B_C_T_H_W: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert x_B_C_T_H_W.dim() == 5
        B, C, T, H, W = x_B_C_T_H_W.shape
        assert H % self.patch_spatial == 0 and W % self.patch_spatial == 0
        assert T % self.patch_temporal == 0
        token_t = T // self.patch_temporal
        token_h = H // self.patch_spatial
        token_w = W // self.patch_spatial
        tokens = rearrange(
            x_B_C_T_H_W,
            "b c (t pt) (h ph) (w pw) -> b (t h w) (c pt ph pw)",
            pt=self.patch_temporal,
            ph=self.patch_spatial,
            pw=self.patch_spatial,
        )

        rows = torch.arange(token_h, device=x_B_C_T_H_W.device)
        cols = torch.arange(token_w, device=x_B_C_T_H_W.device)
        rr, cc = torch.meshgrid(rows, cols, indexing="ij")
        grid_1 = torch.stack([rr.reshape(-1), cc.reshape(-1)], dim=0)
        if token_t > 1:
            grid_1 = grid_1.repeat(1, token_t)
        grid = grid_1.unsqueeze(0).repeat(B, 1, 1)

        if padding_mask is None:
            mask = torch.ones(B, tokens.shape[1], device=x_B_C_T_H_W.device, dtype=x_B_C_T_H_W.dtype)
        else:
            pm = padding_mask
            if pm.dim() == 5:
                pm = pm[:, :, 0]
            if pm.dim() == 3:
                pm = pm.unsqueeze(1)
            pm = F.interpolate(pm.float(), size=(H, W), mode="nearest")
            pooled = F.avg_pool2d(pm, kernel_size=self.patch_spatial, stride=self.patch_spatial)
            mask = (pooled > 0.0).flatten(1).to(dtype=x_B_C_T_H_W.dtype)
            if token_t > 1:
                mask = mask.repeat(1, token_t)

        size = torch.tensor([[[token_h, token_w]]], device=x_B_C_T_H_W.device, dtype=torch.int32).repeat(B, 1, 1)
        return tokens, grid, mask, size

    def unpatchify_tokens(self, tokens_B_N_M: torch.Tensor, size_B_1_2: torch.Tensor) -> torch.Tensor:
        sizes = size_B_1_2[:, 0, :].to(device="cpu", dtype=torch.long)
        if not bool((sizes == sizes[:1]).all()):
            raise ValueError("unpatchify_tokens currently requires a uniform token grid")
        token_h = int(sizes[0, 0].item())
        token_w = int(sizes[0, 1].item())
        token_t = max(1, tokens_B_N_M.shape[1] // max(1, token_h * token_w))
        channels = tokens_B_N_M.shape[-1] // (self.patch_temporal * self.patch_spatial * self.patch_spatial)
        return rearrange(
            tokens_B_N_M,
            "b (t h w) (c pt ph pw) -> b c (t pt) (h ph) (w pw)",
            t=token_t,
            h=token_h,
            w=token_w,
            pt=self.patch_temporal,
            ph=self.patch_spatial,
            pw=self.patch_spatial,
            c=channels,
        )

    def _output_tokens_to_patch_tokens(
        self,
        tokens_B_N_M: torch.Tensor,
        size_B_1_2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Reorder final-layer tokens into the ``patchify_latents_to_tokens`` channel layout.

        The final layer emits each token's patch as ``(ph pw pt c)`` — the order that
        :meth:`unpatchify` folds back into a latent grid — whereas the training targets come
        from :meth:`patchify_latents_to_tokens` in ``(c pt ph pw)`` order. The two differ only
        by a permutation *inside* each token; the token positions themselves are untouched. A
        single ``rearrange`` is therefore exact (verified bit-for-bit against the
        ``unpatchify`` -> ``patchify_latents_to_tokens`` round trip) while avoiding both the
        intermediate latent grid and the per-step device->host sync that reading
        ``size_B_1_2`` would force on CUDA.
        """
        del size_B_1_2  # grid shape is implicit in each token; kept only for call-site parity
        return rearrange(
            tokens_B_N_M,
            "b n (ph pw pt c) -> b n (c pt ph pw)",
            ph=self.patch_spatial,
            pw=self.patch_spatial,
            pt=self.patch_temporal,
        )

    def _packed_rope_from_grid(self, grid_B_2_N: torch.Tensor) -> Optional[torch.Tensor]:
        if "rope" not in self.pos_emb_cls.lower():
            return None
        pe = self.pos_embedder
        if grid_B_2_N.numel():
            # RoPE 容量 fail-fast：一次 GPU→CPU 同步取 (max_row, max_col)
            # （原实现两次 .item() = 每步两次同步；amax 后 tolist 与逐通道 max 同值）。
            max_row, max_col = (
                int(v) for v in grid_B_2_N.amax(dim=(0, 2)).tolist()
            )
        else:
            max_row = max_col = 0
        if max_row >= pe.max_h or max_col >= pe.max_w:
            raise ValueError(
                f"packed FiT token grid {(max_row + 1)}x{(max_col + 1)} exceeds RoPE capacity "
                f"{pe.max_h}x{pe.max_w}; increase max_img_h/max_img_w for this native resolution."
            )
        # freqs 只依赖 NTK 系数与 device（与 grid 无关），缓存避免每步重建；
        # 命中时张量与首算逐 bit 相同（同一对象）。
        dev = grid_B_2_N.device
        _key = (dev, float(pe.h_ntk_factor), float(pe.w_ntk_factor), float(pe.t_ntk_factor))
        _cache = getattr(self, "_packed_rope_freqs_cache", None)
        if _cache is not None and _cache[0] == _key:
            h_freqs, w_freqs, t_freqs = _cache[1]
        else:
            h_theta = 10000.0 * pe.h_ntk_factor
            w_theta = 10000.0 * pe.w_ntk_factor
            t_theta = 10000.0 * pe.t_ntk_factor
            h_freqs = 1.0 / (h_theta**pe.dim_spatial_range.to(dev))
            w_freqs = 1.0 / (w_theta**pe.dim_spatial_range.to(dev))
            t_freqs = 1.0 / (t_theta**pe.dim_temporal_range.to(dev))
            self._packed_rope_freqs_cache = (_key, (h_freqs, w_freqs, t_freqs))
        row = grid_B_2_N[:, 0, :].float()
        col = grid_B_2_N[:, 1, :].float()
        half_emb_t = row.new_zeros((row.shape[0], row.shape[1], t_freqs.shape[0]))
        half_emb_h = row.unsqueeze(-1) * h_freqs
        half_emb_w = col.unsqueeze(-1) * w_freqs
        emb = torch.cat([half_emb_t, half_emb_h, half_emb_w] * 2, dim=-1)
        return emb[:, :, None, None, :].float()

    def _build_packed_masks(self, token_mask: Optional[torch.Tensor], dtype: torch.dtype):
        """Build the additive attention mask + float zeroing mask ONCE for the whole
        block stack, instead of rebuilding them (and re-syncing) inside every block.

        Returns ``(attn_mask, token_mask_f)``. Both are None when every token is valid
        (constant-N / token-bucket): a None ``attn_mask`` lets SDPA take its fast
        maskless path (xformers/flash) instead of the slower additive-mask kernel, and
        no output zeroing is needed. The empty-sequence validation and the all-valid
        test are each a single data-dependent ``bool(...)`` GPU→CPU sync — done once
        here rather than once per block (was N_blocks syncs/step). Skipped under
        ``torch.compile``, where the token-bucket invariant guarantees a constant,
        all-valid sequence (and a ``bool(...)`` would graph-break the trace)."""
        if token_mask is None or torch.compiler.is_compiling():
            return None, None
        bool_mask = token_mask.to(dtype=torch.bool)
        if not bool(bool_mask.any(dim=1).all()):
            raise ValueError("packed FiT sequence contains a sample with no valid tokens")
        if bool(bool_mask.all()):
            return None, None
        key_valid = bool_mask[:, None, None, :]
        attn_mask = torch.zeros_like(key_valid, dtype=dtype)
        attn_mask = attn_mask.masked_fill(~key_valid, -1.0e4)
        token_mask_f = token_mask.to(dtype=dtype).unsqueeze(-1)
        return attn_mask, token_mask_f

    def forward_packed_tokens(
        self,
        tokens_B_N_M: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        grid_B_2_N: torch.Tensor,
        mask_B_N: torch.Tensor,
        size_B_1_2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if size_B_1_2 is None:
            raise ValueError("size_B_1_2 (token grid shape) is required for packed FiT")
        expected = self.x_embedder.proj[1].in_features
        if tokens_B_N_M.shape[-1] < expected:
            tokens_B_N_M = F.pad(tokens_B_N_M, (0, expected - tokens_B_N_M.shape[-1]))
        elif tokens_B_N_M.shape[-1] > expected:
            raise ValueError(
                f"packed tokens have dim={tokens_B_N_M.shape[-1]}, but x_embedder expects {expected}"
            )
        x_B_N_D = self.x_embedder.proj[1](tokens_B_N_M)

        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
        t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        self.affline_scale_log_info = {"t_embedding_B_T_D": t_embedding_B_T_D.detach()}
        self.affline_emb = t_embedding_B_T_D
        self.crossattn_emb = crossattn_emb

        rope_emb = self._packed_rope_from_grid(grid_B_2_N)
        attn_mask, token_mask_f = self._build_packed_masks(mask_B_N, x_B_N_D.dtype)
        for block in self.blocks:
            x_B_N_D = block.forward_tokens(
                x_B_N_D,
                t_embedding_B_T_D,
                crossattn_emb,
                rope_emb_L_1_1_D=rope_emb,
                attn_mask=attn_mask,
                token_mask_f=token_mask_f,
                adaln_lora_B_T_3D=adaln_lora_B_T_3D,
            )

        out = self.final_layer.forward_tokens(x_B_N_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        out = self._output_tokens_to_patch_tokens(out, size_B_1_2)
        return out * mask_B_N.to(dtype=out.dtype).unsqueeze(-1)

    def forward_packed_navit(
        self,
        tokens_1_N_M: torch.Tensor,
        timesteps_G: torch.Tensor,
        crossattn_packed_1_L_D: torch.Tensor,
        grid_1_2_N: torch.Tensor,
        visual_seqlens: Sequence[int],
        text_seqlens: Sequence[int],
        use_checkpoint: bool = False,
    ) -> torch.Tensor:
        """NaViT/Patch-n-Pack forward: ``G`` heterogeneous images concatenated into one
        sequence, each attending only to its own tokens (block-diagonal self-attention)
        and only to its own caption (block-diagonal cross-attention), each carrying its
        own sampled timestep (per-token AdaLN).

        ``use_checkpoint=True`` wraps each transformer block in a gradient checkpoint
        (per-block, ``use_reentrant=False``) so backward recomputes one block at a time
        — peak activation memory ≈ 1 block instead of N_blocks. The per-token timestep
        embedding, RoPE, and the two ``BlockDiagonalMask`` biases are built once and
        closed over, identical to the non-checkpoint path.

        Shapes (B is fixed at 1 — the whole pack is one sequence):
          tokens_1_N_M        [1, ΣN, M]   patch tokens, images concatenated in order
          timesteps_G         [G] or [G,1] one timestep per packed image
          crossattn_packed    [1, ΣL, D]   text embeddings, captions concatenated in order
          grid_1_2_N          [1, 2, ΣN]   per-token (row, col) for RoPE (per-image grids)
          visual_seqlens      length G     image token counts (sum == ΣN)
          text_seqlens        length G     caption token counts (sum == ΣL)

        Returns packed patch tokens ``[1, ΣN, O]`` in ``patchify_latents_to_tokens``
        channel order; the caller slices per image (via ``visual_seqlens``) for the loss.
        Self/cross masking uses xformers ``BlockDiagonalMask`` so attention runs the fast
        varlen kernel — there is no O(ΣN²) dense mask and no cross-image leakage (the
        invariant is asserted bit-for-bit in ``test_packed_block_diag_attention``).
        """
        # 后端可用性 fail-fast。xformers 只在 xformers 后端才是硬依赖——sdpa_seg /
        # npu_tnd 用 _SegLens 走等价路径（昇腾上没有 xformers，这是 NaViT 能活的前提）。
        if _PACKED_ATTN_BACKEND == "xformers":
            try:  # 实际构建走 _cached_block_diag_mask（按 seqlens 缓存）
                from xformers.ops.fmha import BlockDiagonalMask  # noqa: F401
            except Exception as exc:  # pragma: no cover - exercised only without xformers
                raise RuntimeError(
                    "forward_packed_navit requires xformers (BlockDiagonalMask) for "
                    "block-diagonal packed attention; it is unavailable. "
                    "在没有 xformers 的平台（如昇腾 NPU）请设 "
                    "navit_attn_backend: npu_tnd 或 sdpa_seg。"
                ) from exc
        elif _PACKED_ATTN_BACKEND == "npu_tnd":
            try:
                import torch_npu  # noqa: F401
            except Exception as exc:  # pragma: no cover - 只在非昇腾机器触发
                raise RuntimeError(
                    "navit_attn_backend=npu_tnd 需要 torch_npu；导入失败。"
                    "非昇腾平台请用 xformers（CUDA）或 sdpa_seg（通用）。"
                ) from exc

        visual_seqlens = [int(s) for s in visual_seqlens]
        text_seqlens = [int(s) for s in text_seqlens]
        if sum(visual_seqlens) != tokens_1_N_M.shape[1]:
            raise ValueError(
                f"visual_seqlens sum {sum(visual_seqlens)} != packed token count "
                f"{tokens_1_N_M.shape[1]}"
            )
        if sum(text_seqlens) != crossattn_packed_1_L_D.shape[1]:
            raise ValueError(
                f"text_seqlens sum {sum(text_seqlens)} != packed text token count "
                f"{crossattn_packed_1_L_D.shape[1]}"
            )

        expected = self.x_embedder.proj[1].in_features
        if tokens_1_N_M.shape[-1] < expected:
            tokens_1_N_M = F.pad(tokens_1_N_M, (0, expected - tokens_1_N_M.shape[-1]))
        elif tokens_1_N_M.shape[-1] > expected:
            raise ValueError(
                f"packed tokens have dim={tokens_1_N_M.shape[-1]}, but x_embedder expects {expected}"
            )
        x_1_N_D = self.x_embedder.proj[1](tokens_1_N_M)

        # Per-image timestep embedding kept at [1, G, *]; blocks receive ``mod_index``
        # ([ΣN] token→image row) and run AdaLN modulation on G rows, gathering each
        # chunk to a contiguous per-token tensor inside ``forward_tokens``. Same math
        # as the old repeat-interleave-to-token layout（同一行同值），but drops the
        # ΣN-row modulation matmuls and the strided chunk views（本地交错实测前向
        # 中位 −13%；云端占比待 stage_timing 验证）。
        if timesteps_G.ndim == 1:
            timesteps_G = timesteps_G.unsqueeze(1)            # [G, 1]
        t_emb_G_1_D, adaln_lora_G_1_3D = self.t_embedder(timesteps_G)
        t_emb_G_1_D = self.t_embedding_norm(t_emb_G_1_D)

        counts = torch.tensor(visual_seqlens, device=x_1_N_D.device)
        mod_index = torch.repeat_interleave(
            torch.arange(len(visual_seqlens), device=x_1_N_D.device), counts
        )                                                     # [ΣN]
        t_emb_1_G_D = t_emb_G_1_D[:, 0, :].unsqueeze(0)       # [1, G, D]
        if adaln_lora_G_1_3D is not None:
            adaln_lora_1_G_3D = adaln_lora_G_1_3D[:, 0, :].unsqueeze(0)
        else:
            adaln_lora_1_G_3D = None

        self.affline_scale_log_info = {"t_embedding_B_T_D": t_emb_1_G_D.detach()}
        self.affline_emb = t_emb_1_G_D
        self.crossattn_emb = crossattn_packed_1_L_D

        rope_emb = self._packed_rope_from_grid(grid_1_2_N)
        # BlockDiagonalMask 只依赖 seqlens（纯 CPU 元数据对象，块内已跨 28 block +
        # checkpoint 重算复用），按 seqlens 元组缓存避免每步重建。
        if _PACKED_ATTN_BACKEND == "xformers":
            self_bias = _cached_block_diag_mask(tuple(visual_seqlens))
            cross_bias = _cached_block_diag_mask(
                tuple(visual_seqlens), tuple(text_seqlens)
            )
        else:
            # sdpa_seg / npu_tnd：同样是纯 CPU 元数据，同样按 seqlens 缓存复用
            self_bias = _cached_seg_lens(tuple(visual_seqlens))
            cross_bias = _cached_seg_lens(tuple(visual_seqlens), tuple(text_seqlens))

        for block in self.blocks:
            def _run(x_in, blk=block):
                return blk.forward_tokens(
                    x_in,
                    t_emb_1_G_D,
                    crossattn_packed_1_L_D,
                    rope_emb_L_1_1_D=rope_emb,
                    attn_mask=self_bias,
                    token_mask_f=None,
                    adaln_lora_B_T_3D=adaln_lora_1_G_3D,
                    cross_attn_mask=cross_bias,
                    token_wise_mod=True,
                    mod_index=mod_index,
                )
            if use_checkpoint:
                x_1_N_D = checkpoint(_run, x_1_N_D, use_reentrant=False)
            else:
                x_1_N_D = _run(x_1_N_D)

        out = self.final_layer.forward_tokens(
            x_1_N_D, t_emb_1_G_D, adaln_lora_B_T_3D=adaln_lora_1_G_3D,
            token_wise_mod=True, mod_index=mod_index,
        )
        return self._output_tokens_to_patch_tokens(out, None)

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: (B, C, T, H, W) tensor of spatial-temp inputs
            timesteps: (B, ) tensor of timesteps
            crossattn_emb: (B, N, D) tensor of cross-attention embeddings
        """
        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = self.prepare_embedded_sequence(
            x_B_C_T_H_W,
            fps=fps,
            padding_mask=padding_mask,
        )

        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
        t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        # for logging purpose
        affline_scale_log_info = {}
        affline_scale_log_info["t_embedding_B_T_D"] = t_embedding_B_T_D.detach()
        self.affline_scale_log_info = affline_scale_log_info
        self.affline_emb = t_embedding_B_T_D
        self.crossattn_emb = crossattn_emb

        if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
            assert (
                x_B_T_H_W_D.shape == extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape
            ), f"{x_B_T_H_W_D.shape} != {extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape}"


        blocks = self.blocks

        block_kwargs = {
            "rope_emb_L_1_1_D": rope_emb_L_1_1_D,
            "adaln_lora_B_T_3D": adaln_lora_B_T_3D,
            "extra_per_block_pos_emb": extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
        }
        for block in blocks:
            x_B_T_H_W_D = block(
                x_B_T_H_W_D,
                t_embedding_B_T_D,
                crossattn_emb,
                **block_kwargs,
            )

        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        x_B_C_Tt_Hp_Wp = self.unpatchify(x_B_T_H_W_O)
        return x_B_C_Tt_Hp_Wp


# Anima model - MiniTrainDIT with LLMAdapter for bridging Qwen3 embeddings
# Based on ComfyUI's comfy/ldm/anima/model.py

import torch
from torch import nn
import torch.nn.functional as F



def rotate_half_llm(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_llm(x, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    x_embed = (x * cos) + (rotate_half_llm(x) * sin)
    return x_embed


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.rope_theta = 10000
        self._head_dim = head_dim
        self.register_buffer("inv_freq", self._make_inv_freq(), persistent=False)

    def _make_inv_freq(self) -> torch.Tensor:
        return 1.0 / (self.rope_theta ** (torch.arange(0, self._head_dim, 2, dtype=torch.int64).to(dtype=torch.float) / self._head_dim))

    def reset_parameters(self) -> None:
        """重算 ``inv_freq``。

        它是 non-persistent buffer（不进 state_dict），所以 checkpoint 永远填不到它。
        正常构造路径下 ``__init__`` 已经算好，这个方法不会被调用；只有 meta device
        构造（trainer/models.py 的 fast_init）在 ``to_empty()`` 之后需要它——那时
        buffer 里是未初始化内存。
        """
        self.inv_freq = self._make_inv_freq().to(self.inv_freq.device)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class LLMAdapterAttention(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads, head_dim):
        super().__init__()

        inner_dim = head_dim * n_heads
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.query_dim = query_dim
        self.context_dim = context_dim

        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)

        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)

        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)

        self.o_proj = nn.Linear(inner_dim, query_dim, bias=False)

    def forward(self, x, mask=None, context=None, position_embeddings=None, position_embeddings_context=None):
        context = x if context is None else context
        input_shape = x.shape[:-1]
        q_shape = (*input_shape, self.n_heads, self.head_dim)
        context_shape = context.shape[:-1]
        kv_shape = (*context_shape, self.n_heads, self.head_dim)

        query_states = self.q_norm(self.q_proj(x).view(q_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(context).view(kv_shape)).transpose(1, 2)
        value_states = self.v_proj(context).view(kv_shape).transpose(1, 2)

        if position_embeddings is not None:
            assert position_embeddings_context is not None
            cos, sin = position_embeddings
            query_states = apply_rotary_pos_emb_llm(query_states, cos, sin)
            cos, sin = position_embeddings_context
            key_states = apply_rotary_pos_emb_llm(key_states, cos, sin)

        # 注：昇腾需要的 "Sq=1 广播 mask 展开" 在 ``LLMAdapter.forward`` 里一次性做完
        # （两个 mask 的 query 都是 x，Sq 恒等），这里拿到的已经是可直接下发的形状。
        attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=mask)

        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output

    def init_weights(self):
        torch.nn.init.zeros_(self.o_proj.weight)


class LLMAdapterTransformerBlock(nn.Module):
    def __init__(self, source_dim, model_dim, num_heads=16, mlp_ratio=4.0, use_self_attn=False, layer_norm=False):
        super().__init__()
        self.use_self_attn = use_self_attn

        if self.use_self_attn:
            self.norm_self_attn = nn.LayerNorm(model_dim) if layer_norm else nn.RMSNorm(model_dim, eps=1e-6)
            self.self_attn = LLMAdapterAttention(
                query_dim=model_dim,
                context_dim=model_dim,
                n_heads=num_heads,
                head_dim=model_dim//num_heads,
            )

        self.norm_cross_attn = nn.LayerNorm(model_dim) if layer_norm else nn.RMSNorm(model_dim, eps=1e-6)
        self.cross_attn = LLMAdapterAttention(
            query_dim=model_dim,
            context_dim=source_dim,
            n_heads=num_heads,
            head_dim=model_dim//num_heads,
        )

        self.norm_mlp = nn.LayerNorm(model_dim) if layer_norm else nn.RMSNorm(model_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, int(model_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(model_dim * mlp_ratio), model_dim)
        )

    def forward(self, x, context, target_attention_mask=None, source_attention_mask=None, position_embeddings=None, position_embeddings_context=None):
        if self.use_self_attn:
            normed = self.norm_self_attn(x)
            attn_out = self.self_attn(normed, mask=target_attention_mask, position_embeddings=position_embeddings, position_embeddings_context=position_embeddings)
            x = x + attn_out

        normed = self.norm_cross_attn(x)
        attn_out = self.cross_attn(normed, mask=source_attention_mask, context=context, position_embeddings=position_embeddings, position_embeddings_context=position_embeddings_context)
        x = x + attn_out

        x = x + self.mlp(self.norm_mlp(x))
        return x

    def init_weights(self):
        torch.nn.init.zeros_(self.mlp[2].weight)
        self.cross_attn.init_weights()


class LLMAdapter(nn.Module):
    """
    LLMAdapter bridges Qwen3 embeddings to the diffusion model via a transformer-based module.

    Takes:
    - source_hidden_states: Qwen3 embeddings (B, seq_len, 1024)
    - target_input_ids: T5 token IDs (B, seq_len)

    Returns:
    - Processed embeddings for cross-attention in the diffusion model
    """
    def __init__(
            self,
            source_dim=1024,
            target_dim=1024,
            model_dim=1024,
            num_layers=6,
            num_heads=16,
            use_self_attn=True,
            layer_norm=False,
        ):
        super().__init__()

        self.embed = nn.Embedding(32128, target_dim)  # T5 vocab size
        if model_dim != target_dim:
            self.in_proj = nn.Linear(target_dim, model_dim)
        else:
            self.in_proj = nn.Identity()
        self.rotary_emb = RotaryEmbedding(model_dim//num_heads)
        self.blocks = nn.ModuleList([
            LLMAdapterTransformerBlock(source_dim, model_dim, num_heads=num_heads, use_self_attn=use_self_attn, layer_norm=layer_norm) for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(model_dim, target_dim)
        self.norm = nn.RMSNorm(target_dim, eps=1e-6)

    def forward(self, source_hidden_states, target_input_ids, target_attention_mask=None, source_attention_mask=None):
        if target_attention_mask is not None:
            target_attention_mask = target_attention_mask.to(torch.bool)
            if target_attention_mask.ndim == 2:
                target_attention_mask = target_attention_mask.unsqueeze(1).unsqueeze(1)

        if source_attention_mask is not None:
            source_attention_mask = source_attention_mask.to(torch.bool)
            if source_attention_mask.ndim == 2:
                source_attention_mask = source_attention_mask.unsqueeze(1).unsqueeze(1)

        x = self.in_proj(self.embed(target_input_ids))
        context = source_hidden_states

        # 昇腾：FlashAttentionScore 不接受 Sq 维为 1 的广播 mask（详见 utils/npu_compat.py）。
        # 在这里一次性展开而不是在每个 attention 里做：self-attn 与 cross-attn 的 query 都是
        # x（cross-attn 只换 k/v），两个 mask 的 Sq 都等于 x.shape[1]，而 mask 在整个 block
        # 栈里不变 —— 放在 attention 层里等于对同一个张量重复展开 2×num_layers 次。
        # CUDA/CPU 上 expand_attn_mask 是恒等映射，行为逐字节不变。
        from utils.npu_compat import expand_attn_mask as _expand_attn_mask
        target_attention_mask = _expand_attn_mask(target_attention_mask, x.shape[1])
        source_attention_mask = _expand_attn_mask(source_attention_mask, x.shape[1])

        position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        position_ids_context = torch.arange(context.shape[1], device=x.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(x, position_ids)
        position_embeddings_context = self.rotary_emb(x, position_ids_context)
        for block in self.blocks:
            x = block(x, context, target_attention_mask=target_attention_mask, source_attention_mask=source_attention_mask, position_embeddings=position_embeddings, position_embeddings_context=position_embeddings_context)
        return self.norm(self.out_proj(x))


class Anima(MiniTrainDIT):
    """
    Anima model - extends MiniTrainDIT (Cosmos-Predict2 base) with an LLMAdapter
    for processing dual text encoder outputs (Qwen3 + T5).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # LLMAdapter with default Anima configuration
        self.llm_adapter = LLMAdapter(
            source_dim=1024,   # Qwen3 embedding dimension
            target_dim=1024,   # Output dimension (matches crossattn_emb_channels)
            model_dim=1024,
            num_layers=6,
            num_heads=16,
            use_self_attn=True,
            layer_norm=False,
        )

    def preprocess_text_embeds(self, text_embeds, text_ids, target_attention_mask=None, source_attention_mask=None):
        """
        Process text embeddings through the LLM adapter.

        Args:
            text_embeds: Qwen3 embeddings (B, seq_len, 1024)
            text_ids: T5 token IDs (B, seq_len)

        Returns:
            Processed embeddings for cross-attention
        """
        if text_ids is not None:
            return self.llm_adapter(text_embeds, text_ids, target_attention_mask, source_attention_mask)
        else:
            return text_embeds


GeneralDIT = MiniTrainDIT
