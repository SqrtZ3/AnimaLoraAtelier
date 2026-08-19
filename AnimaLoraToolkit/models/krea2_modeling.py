# Krea 2 (K2) SingleStreamDiT —— 训练用移植。
#
# 来源：官方推理实现 https://github.com/krea-ai/krea-2 （mmdit.py / encoder.py / sampling.py，
# 模型权重与代码按 Krea 2 Community License 发布；详见 THIRD_PARTY_NOTICES.md）。
# 架构（Krea 2 Technical Report, 2026-06）：
#   - 单流 MMDiT：text token + image token 拼一条序列共享 attention/MLP（text 在前）。
#   - GQA（48 Q 头 / 12 KV 头，headdim 128）+ sigmoid 门控 attention 输出。
#   - SwiGLU MLP（multiplier 4，mlpdim 128 对齐）。
#   - 轻量 timestep 调制：共享 tproj(6·features) + 每 block 可学习 bias（DoubleSharedModulation），
#     替代传统 per-block AdaLN MLP。
#   - zero-center RMSNorm（weight = scale + 1）、QKNorm。
#   - 3D axial RoPE（frame/h/w = [32,48,48] @ headdim128，theta=1e3；text token pos 全 0）。
#   - 文本条件：Qwen3-VL 12 层 hidden states 堆叠 [B,L,12,2560]，DiT 内 TextFusionTransformer
#     先沿"层轴"融合（2 block + Linear(12→1) 投影）再沿"序列轴"精修（2 block），txtmlp 桥到 6144。
#
# 与官方实现的差异（全部为训练工程所需，数学上对有效 token 等价）：
#   1. 去掉 @torch.compile 装饰器（与 per-block gradient checkpoint / LoRA 注入冲突）。
#   2. 去掉"combined 序列 pad 到 256 倍数"（官方注释明确它只为稳定编译 kernel 形状；
#      pad 位置被 mask 屏蔽，对有效 token 输出无影响）。
#   3. dense 路径的 key-padding mask 在对角线上强制 True（全 mask 行在非 CUDNN SDPA 后端
#      会产生 NaN 并顺着 V 污染后续层；对角 True 让 pad 行只看自己 → 有限值。有效 query
#      行原本对角即 True，masked key 集合不变 → 有效 token 输出严格不变）。
#   4. 新增训练接口（duck-type 对齐本仓库 Anima 模型的调用契约）：
#      forward(x_B_C_T_H_W, timesteps_B_T, crossattn_emb, fps=None, padding_mask=None)
#      forward_dense(...)（内建 per-block checkpoint）、patchify_latents_to_tokens、
#      forward_packed_navit（NaViT block-diagonal 打包，text+image 同序列逐图隔离）。
#
# 模块/属性命名与官方逐一相同 —— `raw.safetensors` 可 strict=True 直载。

from __future__ import annotations

import functools
import logging
import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger(__name__)

# ── 梯度检查点策略（opt-in，默认 "full" = 历史行为逐字节不变）──────────────────
# 全量 checkpoint 把一个 block 的**所有**中间量都丢掉、backward 全部重算；重算代价
# ≈ 一整个 forward（实测 bwd/fwd=2.73≈3）。选择性重算（SAC）只丢便宜的（norm/silu/
# rope/逐元素），保留贵的（matmul / SDPA 输出），用显存换掉大部分重算。
#
# H20 实测（28 块真栈，tests/diag_navit_ckpt_policy.py --g-sweep，ms/token 对 G 不敏感，
# 故下表可直接按 token 数外推；截距均落在 22.6GiB=权重，自洽）：
#
#   策略            ms/token   激活 MB/token
#   full(现状)        0.674        0.72
#   sac_attn          0.653        1.03
#   sac_narrow        0.562        2.43
#   sac_all           0.484        4.07
#   （对照）无 ckpt   0.465        7.52   ← 只比 sac_all 快 4%，显存 1.85×
#
# 关键前提：块对角 attention 下 **per-token 代价只取决于每图 seqlen，与一包装几张图
# 无关**（同一策略在 G=1..6 上 ms/token 变化 <1%）。所以降 navit_token_budget 不损
# 吞吐、只让出显存 —— 这正是换取更便宜策略的本钱。
# 已发布的 grad_checkpoint_skip_last=8/12 在这张前沿上被 sac_narrow **完全支配**
# （更慢且更占显存），不建议再用。
_CKPT_POLICY = "full"
_CKPT_POLICIES = ("full", "sac_attn", "sac_narrow", "sac_all")
_CKPT_NARROW_WIDTH = 0          # 0 = 构造时按 config.features 定；见 set_checkpoint_policy


def set_checkpoint_policy(name: str, narrow_width: int = 0) -> None:
    """训练入口调用（trainer 读 grad_checkpoint_policy 配置）。非法值构造期 fail-fast。

    ``narrow_width``：sac_narrow 保存 mm 输出的最大宽度（0=不限，由调用方传模型
    features）。默认放过比它更宽的中间量 —— 对 Krea2 就是 SwiGLU 的 16384 维，
    显存大头正在那里。
    """
    global _CKPT_POLICY, _CKPT_NARROW_WIDTH
    name = (name or "full").lower()
    if name not in _CKPT_POLICIES:
        raise ValueError(
            f"grad_checkpoint_policy={name!r} 不认识；可选 {_CKPT_POLICIES}")
    if name != "full":
        try:
            from torch.utils.checkpoint import create_selective_checkpoint_contexts  # noqa: F401
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                f"grad_checkpoint_policy={name} 需要 torch 的选择性重算 API "
                "(torch.utils.checkpoint.create_selective_checkpoint_contexts)，"
                "当前 torch 版本没有。请升级 torch 或用 full。"
            ) from exc
    _CKPT_POLICY = name
    _CKPT_NARROW_WIDTH = int(narrow_width or 0)


@functools.lru_cache(maxsize=8)
def _sac_context_fn(policy: str, narrow_width: int):
    """按策略构造 SAC 的 context_fn（纯 CPU 元数据，按 (policy,width) 缓存）。"""
    from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

    sdpa_ops = {
        torch.ops.aten._scaled_dot_product_efficient_attention.default,
        torch.ops.aten._scaled_dot_product_flash_attention.default,
        torch.ops.aten._scaled_dot_product_cudnn_attention.default,
    }
    mm_ops = {
        torch.ops.aten.mm.default,
        torch.ops.aten.addmm.default,
        torch.ops.aten.bmm.default,
    }

    def _out_width(op, args):
        try:
            if op is torch.ops.aten.addmm.default:
                return int(args[2].shape[-1])
            return int(args[1].shape[-1])
        except Exception:  # noqa: BLE001
            return 1 << 30      # 认不出形状就当"很宽"→ 重算（保守省显存）

    def _policy(ctx, op, *args, **kwargs):
        if op in sdpa_ops:
            return CheckpointPolicy.MUST_SAVE
        if op in mm_ops:
            if policy == "sac_all":
                return CheckpointPolicy.MUST_SAVE
            if policy == "sac_narrow" and (
                narrow_width <= 0 or _out_width(op, args) <= narrow_width
            ):
                return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    return lambda: create_selective_checkpoint_contexts(_policy)


def _ckpt(fn, *args):
    """checkpoint 包装：policy=full 时与直接调用 checkpoint 完全一致（逐字节）。"""
    if _CKPT_POLICY == "full":
        return checkpoint(fn, *args, use_reentrant=False)
    return checkpoint(fn, *args, use_reentrant=False,
                      context_fn=_sac_context_fn(_CKPT_POLICY, _CKPT_NARROW_WIDTH))


# torch >= 2.5 的 SDPA 原生支持 GQA（enable_gqa）；老版本退回手动展开 KV 头（数值相同）。
try:
    _SDPA_HAS_GQA = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) >= (2, 5)
except Exception:  # pragma: no cover
    _SDPA_HAS_GQA = False

# xformers GQA：优先走 5D grouped 布局 [B, L, G, H/G, D]，KV 用 expand（stride-0
# 视图）广播到组内 —— 免去 repeat_interleave 物化 4×KV 的读写（28 块 × grad
# checkpoint recompute 每步各来一遍）。与展开后计算数学等价（同一 q 头分组约定：
# q 头 i ↔ kv 头 i//rep，与 repeat_interleave / SDPA enable_gqa 一致）。
# 可用性由**首次调用时的一次性小张量探针**决定（含 backward —— 部分 xformers
# 构建只有 fa2B/fa3B backward 算子、不支持 BMGHK 格式，前向能过 backward 会炸；
# 云端实测如此）。探针失败记一条 WARNING 后永久走 repeat_interleave（非静默）。
# 不在真实前向里试错：grad checkpoint 下中途换 kernel 会让同一步混两种
# bf16 舍入路径，探针前置可彻底避免。
# None=未探针；True/False=探针结果。测试可显式覆写。
_XF_GQA_5D_OK: Optional[bool] = None

# ── packed attention 后端（opt-in，默认 xformers = 历史行为逐 bit 不变）────────
# "sdpa_seg"：块对角 mask 换成逐段 dense SDPA（cudnn 后端）。段内全注意力 =
# 块对角语义，数学恒等（有单测对拍）；H20 微基准 dense SDPA 比 xformers FA2
# varlen 快 1.56×（fwd，三种段几何一致），且不依赖 xformers。
# G 通常 2~10，逐段 launch 开销可忽略。该接缝同时是未来低比特 attention
# 后端（如 SageBwd INT8）的插槽——任何"dense [B,H,L,D] 进出"的 kernel 都能挂。
_PACKED_ATTN_BACKEND = "xformers"
_PACKED_ATTN_BACKENDS = ("xformers", "sdpa_seg")


def set_packed_attention_backend(name: str) -> None:
    """训练入口调用（trainer 读 navit_attn_backend 配置）。非法值构造期 fail-fast。"""
    global _PACKED_ATTN_BACKEND
    name = (name or "xformers").lower()
    if name not in _PACKED_ATTN_BACKENDS:
        raise ValueError(
            f"navit_attn_backend={name!r} 不认识；可选 {_PACKED_ATTN_BACKENDS}")
    _PACKED_ATTN_BACKEND = name


class _SegLens:
    """packed 注意力的轻量段长标记（sdpa_seg 后端替代 BlockDiagonalMask/bool mask）。

    attention() 检测到它时逐段调用 dense SDPA；持有 (s0, s1, ...) 段长元组。
    """
    __slots__ = ("seg_lens",)

    def __init__(self, seg_lens):
        self.seg_lens = tuple(int(s) for s in seg_lens)


def _probe_xf_gqa_5d(device, dtype, headdim: int, rep: int) -> bool:
    """小张量探针：BMGHK 布局 + BlockDiagonalMask 的 forward+backward 是否可用。"""
    try:
        import xformers.ops as xops
        from xformers.ops.fmha import BlockDiagonalMask

        G = 2
        with torch.enable_grad():
            q = torch.randn(1, 8, G, rep, headdim, device=device, dtype=dtype,
                            requires_grad=True)
            kv = torch.randn(1, 8, G, 1, headdim, device=device, dtype=dtype,
                             requires_grad=True)
            k = kv.expand(1, 8, G, rep, headdim)
            v = kv.expand(1, 8, G, rep, headdim)
            bias = BlockDiagonalMask.from_seqlens([4, 4])
            out = xops.memory_efficient_attention(q, k, v, attn_bias=bias)
            out.float().sum().backward()
        return True
    except Exception as e:
        # 已知现状（2026-07 实测，本地 xformers/torch2.7 与云端 fa2 2.8.3 构建一致）：
        # BMGHK 只有 forward 算子，backward 一律缺失 → 训练场景探针必然失败。
        # 这是能力检测的常态而非异常，记 INFO；待未来 xformers 补上
        # cutlassB/fa 的 BMGHK backward 后此路径自动启用。
        logger.info(
            "xformers 5D grouped GQA 不可用（%s: %s），走 repeat_interleave "
            "展开路径（数值相同，仅多物化 KV 副本，实测开销 <0.5%% 步时）。",
            type(e).__name__, str(e).split("\n")[0],
        )
        return False


def _is_xformers_bias(mask) -> bool:
    """True iff mask 是 xformers 的 AttentionBias（BlockDiagonalMask 等）。"""
    if mask is None or torch.is_tensor(mask):
        return False
    mod = type(mask).__module__ or ""
    return mod.startswith("xformers")


_BLOCK_DIAG_CACHE: dict = {}


def cached_block_diag_mask(q_seqlens: tuple, kv_seqlens: Optional[tuple] = None):
    """构建（并缓存）xformers BlockDiagonalMask。纯 CPU 元数据，按 seqlens 缓存。"""
    key = (q_seqlens, kv_seqlens)
    hit = _BLOCK_DIAG_CACHE.get(key)
    if hit is not None:
        return hit
    from xformers.ops.fmha import BlockDiagonalMask  # lazy：仅 packed 路径需要

    if kv_seqlens is None:
        bias = BlockDiagonalMask.from_seqlens(list(q_seqlens))
    else:
        bias = BlockDiagonalMask.from_seqlens(list(q_seqlens), kv_seqlen=list(kv_seqlens))
    if len(_BLOCK_DIAG_CACHE) > 256:
        _BLOCK_DIAG_CACHE.clear()
    _BLOCK_DIAG_CACHE[key] = bias
    return bias


def _xformers_available() -> bool:
    try:
        from xformers.ops.fmha import BlockDiagonalMask  # noqa: F401
        return True
    except Exception:
        return False


def block_diag_bool_mask(seqlens: Sequence[int], device) -> Tensor:
    """无 xformers 时的 SDPA 回退：块对角 bool mask [1,1,Σ,Σ]（O(Σ²) 显存，仅小规模验证用）。"""
    total = int(sum(seqlens))
    m = torch.zeros(total, total, dtype=torch.bool, device=device)
    off = 0
    for n in seqlens:
        m[off:off + n, off:off + n] = True
        off += n
    return m.unsqueeze(0).unsqueeze(0)


def rope(pos: Tensor, dim: int, theta: float = 1e4, ntk: float = 1.0) -> Tensor:
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / ((theta * ntk) ** scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack(
        [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1
    )
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def ropeapply(xq: Tensor, xk: Tensor, freqs: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    freqs = freqs[:, None, :, :, :]
    xq_ = freqs[..., 0] * xq_[..., 0] + freqs[..., 1] * xq_[..., 1]
    xk_ = freqs[..., 0] * xk_[..., 0] + freqs[..., 1] * xk_[..., 1]
    return xq_.reshape(*xq.shape).to(xq.dtype), xk_.reshape(*xk.shape).to(xk.dtype)


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask=None,
    scale: Optional[float] = None,
    gqa: bool = False,
) -> Tensor:
    """q/k/v: [B,H,L,D]（k/v 头数可少于 q，GQA）。mask 可为 None / bool 张量 /
    xformers AttentionBias。返回 [B, L, H*D]。

    - SDPA 路径：torch>=2.5 用 enable_gqa；否则 repeat_interleave 展开 KV 头（数值相同）。
    - xformers 路径（BlockDiagonalMask，navit packed）：展开 KV 头后走
      memory_efficient_attention 的 varlen 快 kernel（musubi-tuner 对 GQA 同样做展开，
      与原生 GQA 数值一致）。
    - _SegLens 路径（navit_attn_backend=sdpa_seg）：逐段 dense SDPA（cudnn），
      段内全注意力 ≡ 块对角 mask，数学恒等（tests/test_sdpa_seg_attention.py 对拍）。
    """
    if isinstance(mask, _SegLens):
        outs = []
        off = 0
        for s in mask.seg_lens:
            qs = q[:, :, off:off + s]
            ks = k[:, :, off:off + s]
            vs = v[:, :, off:off + s]
            if gqa and ks.shape[1] != qs.shape[1]:
                if _SDPA_HAS_GQA:
                    o = F.scaled_dot_product_attention(qs, ks, vs, scale=scale,
                                                       enable_gqa=True)
                else:
                    rep = qs.shape[1] // ks.shape[1]
                    o = F.scaled_dot_product_attention(
                        qs, ks.repeat_interleave(rep, dim=1),
                        vs.repeat_interleave(rep, dim=1), scale=scale)
            else:
                o = F.scaled_dot_product_attention(qs, ks, vs, scale=scale)
            outs.append(o)
            off += s
        x = torch.cat(outs, dim=2)
        return rearrange(x, "B H L D -> B L (H D)")

    if _is_xformers_bias(mask):
        import xformers.ops as xops

        global _XF_GQA_5D_OK
        if gqa and k.shape[1] != q.shape[1]:
            B, Hq, L, D = q.shape
            G_kv = k.shape[1]
            rep = Hq // G_kv
            if _XF_GQA_5D_OK is None:
                _XF_GQA_5D_OK = _probe_xf_gqa_5d(q.device, q.dtype, D, rep)
                if _XF_GQA_5D_OK:
                    logger.info("xformers 5D grouped GQA 探针通过，启用免物化 KV 路径")
            if _XF_GQA_5D_OK:
                # 5D grouped 布局：q [B,L,G,rep,D]，kv [B,L,G,1,D]→expand（零拷贝）。
                # 可用性已由探针（含 backward）确认，此处不再试错——真实前向若仍
                # 失败应当 fail-fast 抛出，而不是在 grad checkpoint 中途换 kernel。
                q5 = q.transpose(1, 2).contiguous().view(B, L, G_kv, rep, D)
                k5 = k.transpose(1, 2).contiguous().unsqueeze(3).expand(B, L, G_kv, rep, D)
                v5 = v.transpose(1, 2).contiguous().unsqueeze(3).expand(B, L, G_kv, rep, D)
                x = xops.memory_efficient_attention(q5, k5, v5, attn_bias=mask, scale=scale)
                return x.reshape(B, L, Hq * D)

        if gqa and k.shape[1] != q.shape[1]:
            rep = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        # xformers 布局 [B, L, H, D]
        q_ = q.transpose(1, 2).contiguous()
        k_ = k.transpose(1, 2).contiguous()
        v_ = v.transpose(1, 2).contiguous()
        x = xops.memory_efficient_attention(q_, k_, v_, attn_bias=mask, scale=scale)
        return rearrange(x, "B L H D -> B L (H D)")

    if gqa and k.shape[1] != q.shape[1]:
        if _SDPA_HAS_GQA:
            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, scale=scale, enable_gqa=True
            )
            return rearrange(x, "B H L D -> B L (H D)")
        rep = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
    return rearrange(x, "B H L D -> B L (H D)")


def _mask(mask: Tensor) -> Optional[Tensor]:
    """(B, L) key-padding mask → (B, 1, L, L) bool attention mask（外积）。

    与官方实现的唯一差异：对角线强制 True。全 False 行（padding 位置的 query）在
    math/efficient SDPA 后端会 softmax 出 NaN，并在下一层作为 value 污染有效 token
    （官方靠 CUDNN kernel 对全 mask 行输出 0 规避）。对角 True 后 pad 行只 attend 自己
    → 输出有限；有效 query 行的 masked key 集合不变，输出严格不变。

    全可见（无任何 padding）时返回 None：attn_mask=None 数学上与全 True mask
    严格等价，但允许 SDPA 走 flash 后端。非 None mask 会把 SDPA 排除出 flash，
    某些平台（sm120 + torch 2.11 实测）进一步落到 math 后端，物化 heads×L² 的
    注意力矩阵——eval/采样的 batch=1 长序列稠密前向因此单次分配 ~8GB 直接 OOM。
    eval/采样恰好永远是无 padding 的满 mask，此返回 None 路径正中这两条链路。
    """
    m = mask.bool()
    if bool(m.all()):
        return None
    out = m.unsqueeze(1).unsqueeze(2) & m.unsqueeze(1).unsqueeze(3)
    L = m.shape[-1]
    idx = torch.arange(L, device=m.device)
    out[:, :, idx, idx] = True
    return out


def temb(
    t: Tensor,
    dim: int,
    period: float = 1e4,
    tfactor: float = 1e3,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(period)
        * torch.arange(half, dtype=torch.float32, device=device)
        / half
    )
    # t: (B,) -> args: (B, 1, half)，逐样本向量广播。
    args = (t.float() * tfactor)[:, None, None] * freqs
    sin, cos = torch.sin(args), torch.cos(args)
    return torch.cat((cos, sin), dim=-1).to(dtype=dtype)


@dataclass
class SingleMMDiTConfig:
    features: int
    tdim: int
    txtdim: int
    heads: int
    multiplier: int
    layers: int
    patch: int
    channels: int
    bias: bool = False
    theta: float = 1e3
    kvheads: Optional[int] = None
    txtlayers: int = 1
    txtheads: int = 20
    txtkvheads: int = 20


# Krea 2 12B 发布配置（krea-ai/krea-2 inference.py 的 single_mmdit_large_wide）。
KREA2_LARGE_WIDE = SingleMMDiTConfig(
    features=6144,
    tdim=256,
    txtdim=2560,
    heads=48,
    kvheads=12,
    multiplier=4,
    layers=28,
    patch=2,
    channels=16,
    txtheads=20,
    txtkvheads=20,
    txtlayers=12,
)


class SimpleModulation(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = torch.nn.Parameter(torch.zeros(2, dim))
        self.multiplier = 2

    # vec: (B, 1, D)
    def forward(self, vec: Tensor):
        out = vec + rearrange(self.lin, "two d -> 1 two d")
        scale, shift = out.chunk(self.multiplier, dim=1)
        return scale, shift


class DoubleSharedModulation(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = torch.nn.Parameter(torch.zeros(6 * dim))

    # vec: (B, 1, 6·D) 或 (1, G, 6·D)
    def forward(self, vec: Tensor):
        out = vec + self.lin
        return out.chunk(6, dim=-1)


class PositionalEncoding(torch.nn.Module):
    def __init__(self, dim, axdims: list, theta: float = 1e2, ntk: float = 1.0):
        super().__init__()
        self.axdims = axdims  # headdim 在各位置轴上的切分
        self.theta = theta
        self.ntk = ntk

    def forward(self, pos: Tensor) -> Tensor:
        return torch.cat(
            [
                rope(pos[..., i], d, self.theta, self.ntk)
                for i, d in enumerate(self.axdims)
            ],
            dim=-3,
        )


class RMSNorm(torch.nn.Module):
    """zero-center RMSNorm：存储 scale，生效 weight = scale + 1（fp32 计算后回原 dtype）。"""

    def __init__(self, features: int, eps: float = 1e-05, device=None):
        super().__init__()
        self.features = features
        self.eps = eps
        self.scale = torch.nn.Parameter(
            torch.zeros(features, device=device, dtype=torch.float32)
        )

    def forward(self, x: Tensor) -> Tensor:
        t, dtype = x.float(), x.dtype
        t = F.rms_norm(
            t, (self.features,), eps=self.eps, weight=(self.scale.float() + 1.0)
        )
        return t.to(dtype)


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.qnorm = RMSNorm(dim)
        self.knorm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor):
        return self.qnorm(q), self.knorm(k), v


class SwiGLU(torch.nn.Module):
    def __init__(self, features: int, multiplier: int, bias: bool = False, multiple: int = 128):
        super().__init__()
        mlpdim = int(2 * features / 3) * multiplier
        mlpdim = multiple * ((mlpdim + multiple - 1) // multiple)
        self.gate = torch.nn.Linear(features, mlpdim, bias=bias)
        self.up = torch.nn.Linear(features, mlpdim, bias=bias)
        self.down = torch.nn.Linear(mlpdim, features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Attention(torch.nn.Module):
    def __init__(self, dim: int, heads: int, kvheads: Optional[int] = None, bias: bool = False):
        super().__init__()
        self.heads = heads
        self.kvheads = kvheads if kvheads is not None else heads
        self.headdim = dim // self.heads

        self.wq = torch.nn.Linear(dim, self.headdim * self.heads, bias=bias)
        self.wk = torch.nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.wv = torch.nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.gate = torch.nn.Linear(dim, dim, bias=bias)
        self.qknorm = QKNorm(self.headdim)
        self.gqa = self.heads != self.kvheads
        self.wo = torch.nn.Linear(dim, dim, bias=bias)

    def forward(self, qkv: Tensor, freqs: Optional[Tensor] = None, mask=None) -> Tensor:
        q, k, v, gate = self.wq(qkv), self.wk(qkv), self.wv(qkv), self.gate(qkv)

        q = rearrange(q, "B L (H D) -> B H L D", H=self.heads)
        k = rearrange(k, "B L (H D) -> B H L D", H=self.kvheads)
        v = rearrange(v, "B L (H D) -> B H L D", H=self.kvheads)

        q, k, v = self.qknorm(q, k, v)
        if freqs is not None:
            q, k = ropeapply(q, k, freqs)
        out = self.wo(attention(q, k, v, mask=mask, gqa=self.gqa) * F.sigmoid(gate))
        return out


class LastLayer(torch.nn.Module):
    def __init__(self, features: int, patch: int, channels: int):
        super().__init__()
        self.norm = RMSNorm(features)
        self.linear = torch.nn.Linear(features, patch * patch * channels, bias=True)
        self.modulation = SimpleModulation(features)

    def forward(self, x: Tensor, tvec: Tensor) -> Tensor:
        """tvec: (B,1,D) 逐样本 timestep 向量（packed 逐图路径见 forward_mod）。"""
        scale, shift = self.modulation(tvec)
        x = (1 + scale) * self.norm(x) + shift
        return self.linear(x)

    def forward_mod(self, x: Tensor, tvec_1_G_D: Tensor, mod_index: Tensor) -> Tensor:
        """packed 路径：per-image 调制行 gather 成逐 token。

        SimpleModulation 语义：out = vec + lin[2] → chunk(dim=1) 得 (scale, shift)。
        对 (1,G,D) 输入等价于 scale_g = vec_g + lin[0], shift_g = vec_g + lin[1]。
        """
        scale_g = tvec_1_G_D + self.modulation.lin[0]        # (1,G,D)
        shift_g = tvec_1_G_D + self.modulation.lin[1]        # (1,G,D)
        scale = scale_g[0].index_select(0, mod_index).unsqueeze(0)   # (1,Σ,D)
        shift = shift_g[0].index_select(0, mod_index).unsqueeze(0)
        x = (1 + scale) * self.norm(x) + shift
        return self.linear(x)


class TextFusionBlock(torch.nn.Module):
    def __init__(self, features: int, heads: int, multiplier: int, bias: bool = False,
                 kvheads: Optional[int] = None):
        super().__init__()
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(dim=features, heads=heads, bias=bias, kvheads=kvheads)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(self, x: Tensor, mask=None) -> Tensor:
        x = x + self.attn(self.prenorm(x), mask=mask)
        x = x + self.mlp(self.postnorm(x))
        return x


class TextFusionTransformer(torch.nn.Module):
    # num_txt_layers 是"喂进来的 encoder hidden-state 层数"（投影到 1），不是 transformer 深度
    # —— 深度固定 2 + 2。
    def __init__(self, num_txt_layers: int, txt_dim: int, heads: int, multiplier: int,
                 bias: bool = False, kvheads: Optional[int] = None):
        super().__init__()
        self.layerwise_blocks = torch.nn.ModuleList(
            [TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads) for _ in range(2)]
        )
        self.projector = torch.nn.Linear(num_txt_layers, 1, bias=False)
        self.refiner_blocks = torch.nn.ModuleList(
            [TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads) for _ in range(2)]
        )

    def forward(self, x: Tensor, mask=None) -> Tensor:
        """x: [B, L, n_layers, D]。mask 作用于 refiner 阶段（序列轴）；
        layerwise 阶段每 token 独立沿层轴 attention，无需 mask。

        mask 可为 (B,1,L,L) bool（dense）或 xformers BlockDiagonalMask（navit packed，
        L=ΣL_i 时按 caption 隔离）。
        """
        b, l, n, d = x.shape
        x = x.reshape(b * l, n, d)
        # layerwise 阶段的 SDPA 形状极端：batch=B·L（navit packed 下=全部文本
        # token 数，随 token budget 线性增长）、seqlen=n_layers（个位数）。
        # RTX PRO 6000 (sm120) + torch 2.11 实测：该形状落 flash 后端时，
        # batch≈6k×20 头起 backward 稳定 illegal memory access（budget 49152/65536
        # step1 必崩，anomaly mode 两次指认 ScaledDotProductFlashAttentionBackward0
        # 于本调用链），更小 batch 疑似偶发梯度写坏。seqlen=12 的 attention 用
        # MATH 后端（纯 matmul+softmax，qkᵀ 仅 [.,.,12,12]）既绕开内核 bug 又
        # 几乎零开销；flash 在这个 seqlen 本就无收益。
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel([SDPBackend.MATH]):
            for block in self.layerwise_blocks:
                x = block(x.contiguous(), mask=None)
        x = rearrange(x, "(b l) n d -> b l d n", b=b, l=l)
        x = self.projector(x)
        x = x.squeeze(-1)

        for block in self.refiner_blocks:
            x = block(x, mask=mask)
        return x


class SingleStreamBlock(nn.Module):
    def __init__(self, features: int, heads: int, multiplier: int, bias: bool = False,
                 kvheads: Optional[int] = None):
        super().__init__()
        self.mod = DoubleSharedModulation(features)
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(dim=features, heads=heads, bias=bias, kvheads=kvheads)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(self, x: Tensor, vec: Tensor, freqs: Tensor, mask=None,
                mod_index: Optional[Tensor] = None) -> Tensor:
        """vec: (B,1,6D) 共享 timestep 向量；packed 模式 vec=(1,G,6D) + mod_index[Σ]
        （逐 token gather 到各自图的调制行，与逐图广播同值）。"""
        prescale, preshift, pregate, postscale, postshift, postgate = self.mod(vec)
        if mod_index is not None:
            prescale = prescale[0].index_select(0, mod_index).unsqueeze(0)
            preshift = preshift[0].index_select(0, mod_index).unsqueeze(0)
            pregate = pregate[0].index_select(0, mod_index).unsqueeze(0)
            postscale = postscale[0].index_select(0, mod_index).unsqueeze(0)
            postshift = postshift[0].index_select(0, mod_index).unsqueeze(0)
            postgate = postgate[0].index_select(0, mod_index).unsqueeze(0)
        x = x + pregate * self.attn((1 + prescale) * self.prenorm(x) + preshift, freqs, mask)
        x = x + postgate * self.mlp((1 + postscale) * self.postnorm(x) + postshift)
        return x


class SingleStreamDiT(nn.Module):
    """Krea 2 单流 MMDiT。

    训练侧 duck-type 契约（与本仓库 Anima 模型对齐，见 trainer/model_family.py）：
      - forward(x_B_C_T_H_W, timesteps_B_T, crossattn_emb, fps=None, padding_mask=None,
                cross_mask=None)  —— dense 前向，返回 velocity [B,C,T,H,W]
      - forward_dense(...)                        —— 同上 + 内建 per-block checkpoint
      - patchify_latents_to_tokens(x, padding_mask=None) → (tokens, grid, mask, size)
      - forward_packed_navit(tokens, t_G, cross_packed, grid, vseq, text_seqlens,
                             use_checkpoint=False) → [1, ΣN, patch²·C]
      - _output_tokens_to_patch_tokens(tokens, size) —— Krea2 输出天然就是
        patchify 的 (c ph pw) 通道序（官方 sampling.py 的 unpatchify 即此序），恒等返回。

    crossattn_emb 载荷（与 Anima 的 3D cross 区分）：[B, L, n_txt_layers, txt_dim]
    —— Qwen3-VL 多层 hidden states 原始堆叠；TextFusion 在本模型内部完成。
    cross_mask: [B, L] bool（1=有效）；None = 全有效。
    """

    # 供 trainer 侧区分模型族（Anima 模型无此属性 → getattr 默认 "anima"）。
    model_family = "krea2"

    def __init__(self, config: SingleMMDiTConfig):
        super().__init__()
        self.config = config

        headdim = config.features // config.heads
        axes = [
            headdim - 12 * (headdim // 16),
            6 * (headdim // 16),
            6 * (headdim // 16),
        ]
        assert sum(axes) == headdim, f"sum(axes) = {sum(axes)}, headdim = {headdim}"
        assert all(a % 2 == 0 for a in axes), f"axes = {axes}"

        self.posemb = PositionalEncoding(config.features, axes, theta=config.theta, ntk=1.0)
        self.first = nn.Linear(config.channels * config.patch**2, config.features, bias=True)

        self.blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    config.features, config.heads, config.multiplier,
                    config.bias, config.kvheads,
                )
                for _ in range(config.layers)
            ]
        )
        self.tmlp = nn.Sequential(
            nn.Linear(config.tdim, config.features),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.features, config.features),
        )
        self.txtfusion = TextFusionTransformer(
            config.txtlayers, config.txtdim, config.txtheads,
            config.multiplier, config.bias, config.txtkvheads,
        )
        self.txtmlp = nn.Sequential(
            RMSNorm(config.txtdim),
            nn.Linear(config.txtdim, config.features),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.features, config.features),
        )
        self.last = LastLayer(config.features, config.patch, config.channels)

        self.tproj = nn.Sequential(
            nn.GELU(approximate="tanh"), nn.Linear(config.features, config.features * 6)
        )

    # ------------------------------------------------------------------ #
    # 官方推理 forward（保持原始签名，用于对照验证）                        #
    # ------------------------------------------------------------------ #
    def forward_official(self, img: Tensor, context: Tensor, t: Tensor, pos: Tensor,
                         mask: Optional[Tensor] = None) -> Tensor:
        """img: [B, N_img, patch²·C]（已 patchify）；context: [B, L, n_layers, D_txt]；
        t: [B]；pos: [B, L+N_img, 3]；mask: [B, L+N_img] bool。
        返回 [B, N_img, patch²·C]。（不做官方的 256 对齐 pad，见文件头差异说明。）"""
        img = self.first(img)
        t_vec = self.tmlp(temb(t, self.config.tdim, device=img.device, dtype=img.dtype))
        tvec = self.tproj(t_vec)

        txtmask = _mask(mask[:, : context.shape[1]])
        context = self.txtfusion(context, mask=txtmask)
        context = self.txtmlp(context)

        txtlen, imglen = context.shape[1], img.shape[1]
        combined = torch.cat((context, img), dim=1)

        attn_mask = _mask(mask)
        freqs = self.posemb(pos)

        for block in self.blocks:
            combined = block(combined, tvec, freqs, attn_mask)

        final = self.last(combined, t_vec)
        return final[:, txtlen: txtlen + imglen, :]

    # ------------------------------------------------------------------ #
    # 仓库 dense 契约                                                      #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _image_pos(h_tok: int, w_tok: int, batch: int, device) -> Tensor:
        ids = torch.zeros((h_tok, w_tok, 3), device=device)
        ids[..., 1] = torch.arange(h_tok, device=device)[:, None]
        ids[..., 2] = torch.arange(w_tok, device=device)[None, :]
        return ids.reshape(1, h_tok * w_tok, 3).repeat(batch, 1, 1)

    def _normalize_inputs(self, x_B_C_T_H_W: Tensor, timesteps_B_T: Tensor,
                          crossattn_emb: Tensor):
        if x_B_C_T_H_W.dim() == 4:
            x_B_C_T_H_W = x_B_C_T_H_W.unsqueeze(2)
        B, C, T, H, W = x_B_C_T_H_W.shape
        if T != 1:
            raise ValueError(f"Krea2 是图像模型，期望 T=1，得到 T={T}")
        p = self.config.patch
        if H % p or W % p:
            raise ValueError(f"latent 尺寸 ({H},{W}) 不是 patch={p} 的倍数")
        t = timesteps_B_T.reshape(timesteps_B_T.shape[0], -1)[:, 0].float()
        if crossattn_emb.dim() != 4:
            raise ValueError(
                "Krea2 的 crossattn_emb 应为 [B, L, n_txt_layers, txt_dim]（Qwen3-VL 多层"
                f"堆叠），得到 shape={tuple(crossattn_emb.shape)}。Anima 的 3D cross 与"
                " Krea2 模型不兼容 —— 检查 model_family 配置与文本编码通道。"
            )
        return x_B_C_T_H_W, t, crossattn_emb

    def forward(self, x_B_C_T_H_W: Tensor, timesteps_B_T: Tensor, crossattn_emb: Tensor,
                fps=None, padding_mask=None, cross_mask: Optional[Tensor] = None) -> Tensor:
        """仓库 dense 契约（与 Anima.forward 同签名；fps/padding_mask 兼容占位，忽略）。

        crossattn_emb: [B, L, n_layers, D_txt]；cross_mask: [B, L] bool，None=全有效。
        返回 velocity [B, C, T, H, W]。
        """
        return self.forward_dense(
            x_B_C_T_H_W, timesteps_B_T, crossattn_emb,
            cross_mask=cross_mask, use_checkpoint=False,
        )

    def _checkpoint_from_block(self, use_checkpoint: bool, skip_last: int) -> int:
        """返回 checkpoint 的**开区间上界**：下标 `_i < 返回值` 的 block 才做 checkpoint。

        `skip_last=N` = 最后 N 个 block 不做 checkpoint（存全部激活、backward 不重算），
        其余照常 checkpoint。数学上与全量 checkpoint 恒等，纯粹是显存/计算的取舍。

        为什么把不 checkpoint 的层放在**末尾**而不是开头：backward 从后往前走，末尾这 N 层
        的激活最先被消费并释放，等轮到前面 checkpoint 层重算时它们已经不占显存 → 峰值
        ≈ N×每层激活（出现在 forward 末尾）。若放在开头，峰值会额外叠加一层重算的临时量。

        ★ 返回值语义是"checkpoint 到第几个 block 为止"，不是"从第几个开始"——
          use_checkpoint=False 必须返回 0（一个都不 checkpoint）。曾经这里返回
          len(self.blocks)，配合调用点的 `_i < 上界` 变成了"全部 checkpoint"，
          与 use_checkpoint=False 的语义正好相反。
        """
        n = len(self.blocks)
        if not use_checkpoint:
            return 0            # 一个都不 checkpoint
        skip = max(0, int(skip_last))
        return max(0, n - skip)

    def forward_dense(self, x_B_C_T_H_W: Tensor, timesteps_B_T: Tensor, crossattn_emb: Tensor,
                      cross_mask: Optional[Tensor] = None, use_checkpoint: bool = False,
                      checkpoint_skip_last: int = 0) -> Tensor:
        x5d, t, context = self._normalize_inputs(x_B_C_T_H_W, timesteps_B_T, crossattn_emb)
        B, C, _T, H, W = x5d.shape
        p = self.config.patch
        h_tok, w_tok = H // p, W // p
        device = x5d.device

        img_tokens = rearrange(x5d[:, :, 0], "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=p, pw=p)
        L = context.shape[1]
        if cross_mask is None:
            cross_mask = torch.ones(B, L, dtype=torch.bool, device=device)
        else:
            cross_mask = cross_mask.bool().to(device)

        # ── 块外层也纳入 checkpoint（use_checkpoint 时）──────────────────────
        # first / txtfusion / txtmlp 在 per-block checkpoint 之外，LoRA/DoRA 包住
        # 它们后（官方全 264 targets），DoRA 前向的 fp32 中间量会整段留到 backward
        # （first 一层即 ~0.4GB/16k token）。checkpoint 后只存输入、backward 重算，
        # 数学恒等；重算成本相对 28 个主 block 可忽略。
        txt_mask = _mask(cross_mask)

        def _img_stack(tok_in):
            return self.first(tok_in)

        def _text_stack(ctx_in):
            return self.txtmlp(self.txtfusion(ctx_in, mask=txt_mask))

        if use_checkpoint:
            img = _ckpt(_img_stack, img_tokens)
            txt = _ckpt(_text_stack, context)
        else:
            img = _img_stack(img_tokens)
            txt = _text_stack(context)
        t_vec = self.tmlp(temb(t, self.config.tdim, device=device, dtype=img.dtype))
        tvec = self.tproj(t_vec)

        combined = torch.cat((txt, img), dim=1)
        full_mask = torch.cat(
            (cross_mask, torch.ones(B, img.shape[1], dtype=torch.bool, device=device)), dim=1
        )
        attn_mask = _mask(full_mask)

        pos = torch.cat(
            (torch.zeros(B, L, 3, device=device), self._image_pos(h_tok, w_tok, B, device)),
            dim=1,
        )
        freqs = self.posemb(pos)

        _ckpt_until = self._checkpoint_from_block(use_checkpoint, checkpoint_skip_last)
        for _i, block in enumerate(self.blocks):
            if _i < _ckpt_until:
                combined = _ckpt(
                    lambda x_in, _b=block: _b(x_in, tvec, freqs, attn_mask), combined,
                )
            else:
                combined = block(combined, tvec, freqs, attn_mask)

        out = self.last(combined, t_vec)
        out = out[:, L: L + img_tokens.shape[1], :]
        v = rearrange(
            out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_tok, w=w_tok, ph=p, pw=p
        )
        return v.unsqueeze(2)

    # ------------------------------------------------------------------ #
    # NaViT packed 契约                                                    #
    # ------------------------------------------------------------------ #
    def patchify_latents_to_tokens(self, x_B_C_T_H_W: Tensor, padding_mask=None):
        """与 Anima 同契约：返回 (tokens[B,N,M], grid[B,2,N], mask[B,N], size[B,1,2])。

        token 通道序 (c ph pw)（pt=1 时与 Anima 的 (c pt ph pw) 相同布局）。
        Krea2 navit 打包不支持 padding_mask（native 尺寸 floor 对齐后无 padding）。
        """
        if x_B_C_T_H_W.dim() == 4:
            x_B_C_T_H_W = x_B_C_T_H_W.unsqueeze(2)
        assert x_B_C_T_H_W.dim() == 5
        if padding_mask is not None:
            raise ValueError("Krea2 packed 路径不支持 padding_mask（请用 navit_native_resolution 的无 padding 通路）")
        B, C, T, H, W = x_B_C_T_H_W.shape
        if T != 1:
            raise ValueError(f"Krea2 是图像模型，期望 T=1，得到 T={T}")
        p = self.config.patch
        assert H % p == 0 and W % p == 0
        h_tok, w_tok = H // p, W // p
        tokens = rearrange(x_B_C_T_H_W[:, :, 0], "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=p, pw=p)

        rows = torch.arange(h_tok, device=x_B_C_T_H_W.device)
        cols = torch.arange(w_tok, device=x_B_C_T_H_W.device)
        rr, cc = torch.meshgrid(rows, cols, indexing="ij")
        grid = torch.stack([rr.reshape(-1), cc.reshape(-1)], dim=0).unsqueeze(0).repeat(B, 1, 1)

        mask = torch.ones(B, tokens.shape[1], device=x_B_C_T_H_W.device, dtype=x_B_C_T_H_W.dtype)
        size = torch.tensor([[[h_tok, w_tok]]], device=x_B_C_T_H_W.device, dtype=torch.int32).repeat(B, 1, 1)
        return tokens, grid, mask, size

    def unpatchify_tokens(self, tokens_B_N_M: Tensor, size_B_1_2: Tensor) -> Tensor:
        """(c ph pw) token → latent 网格 [B,C,T=1,H,W]（与 Anima unpatchify_tokens 同契约）。"""
        sizes = size_B_1_2[:, 0, :].to(device="cpu", dtype=torch.long)
        if not bool((sizes == sizes[:1]).all()):
            raise ValueError("unpatchify_tokens 要求 batch 内统一 token 网格")
        h_tok = int(sizes[0, 0].item())
        w_tok = int(sizes[0, 1].item())
        p = self.config.patch
        out = rearrange(
            tokens_B_N_M, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=h_tok, w=w_tok, ph=p, pw=p,
        )
        return out.unsqueeze(2)

    def _output_tokens_to_patch_tokens(self, tokens_B_N_M: Tensor, size_B_1_2=None) -> Tensor:
        """Krea2 的 last.linear 输出天然就是 (c ph pw) 通道序（官方 sampling.py 直接按此
        unpatchify），与 patchify_latents_to_tokens 布局一致 —— 恒等。保留方法以对齐契约。"""
        del size_B_1_2
        return tokens_B_N_M

    def forward_packed_navit(
        self,
        tokens_1_N_M: Tensor,
        timesteps_G: Tensor,
        crossattn_packed: Tensor,
        grid_1_2_N: Tensor,
        visual_seqlens: Sequence[int],
        text_seqlens: Sequence[int],
        use_checkpoint: bool = False,
        checkpoint_skip_last: int = 0,
    ) -> Tensor:
        """NaViT/Patch-n-Pack：G 张异构图打进一条单流序列，每图携带自己的 timestep。

        单流架构下每张图的段 = [该图 caption 的有效 text token ; 该图 image token]，
        自注意力用块对角 mask 按"段"隔离（text+image 同段互见，跨图不可见）——
        与官方 dense 前向对有效 token 数学等价（RoPE: text pos=0 / image (0,row,col)，
        调制：逐图 timestep 行 gather 成逐 token，广播同值）。

        Shapes:
          tokens_1_N_M       [1, ΣN, patch²·C]  image patch token（按图序拼接）
          timesteps_G        [G] / [G,1]        每图一个 t
          crossattn_packed   [1, ΣL, n_layers, D_txt] 各图 caption 有效 token 按图序拼接
          grid_1_2_N         [1, 2, ΣN]         每 image token 的 (row, col)
          visual_seqlens     len G              各图 image token 数（Σ = ΣN）
          text_seqlens       len G              各图有效 caption token 数（Σ = ΣL）

        返回 [1, ΣN, patch²·C]（(c ph pw) 序，与 patchify_latents_to_tokens 对齐）。
        有 xformers 时走 BlockDiagonalMask varlen 快 kernel；否则回退 SDPA 块对角
        bool mask（O(Σ²) 显存，仅小规模/本地验证场景）。
        """
        visual_seqlens = [int(s) for s in visual_seqlens]
        text_seqlens = [int(s) for s in text_seqlens]
        G = len(visual_seqlens)
        if len(text_seqlens) != G:
            raise ValueError(f"text_seqlens 有 {len(text_seqlens)} 项，期望 G={G}")
        if sum(visual_seqlens) != tokens_1_N_M.shape[1]:
            raise ValueError(
                f"visual_seqlens 之和 {sum(visual_seqlens)} != packed image token 数 {tokens_1_N_M.shape[1]}"
            )
        if crossattn_packed.dim() != 4:
            raise ValueError(
                f"crossattn_packed 应为 [1, ΣL, n_layers, D_txt]，得到 {tuple(crossattn_packed.shape)}"
            )
        if sum(text_seqlens) != crossattn_packed.shape[1]:
            raise ValueError(
                f"text_seqlens 之和 {sum(text_seqlens)} != packed text token 数 {crossattn_packed.shape[1]}"
            )
        if min(text_seqlens) <= 0:
            raise ValueError("Krea2 navit 要求每图至少 1 个有效 caption token")

        device = tokens_1_N_M.device
        expected = self.first.in_features
        if tokens_1_N_M.shape[-1] != expected:
            raise ValueError(
                f"packed tokens dim={tokens_1_N_M.shape[-1]}，但 first 层期望 {expected}"
            )

        use_xf = _xformers_available()
        use_sdpa_seg = _PACKED_ATTN_BACKEND == "sdpa_seg"

        # ── 文本融合（varlen：refiner 阶段按 caption 块对角隔离）──────────────
        if use_sdpa_seg:
            txt_bias = _SegLens(text_seqlens)
        elif use_xf:
            txt_bias = cached_block_diag_mask(tuple(text_seqlens))
        else:
            txt_bias = block_diag_bool_mask(text_seqlens, device)

        # 块外层纳入 checkpoint（与 forward_dense 同理：官方全 targets 下这些层
        # 被 DoRA 包住后 fp32 中间量会留到 backward，checkpoint 后数学恒等）。
        def _text_stack(ctx_in):
            return self.txtmlp(self.txtfusion(ctx_in, mask=txt_bias))

        if use_checkpoint:
            txt = _ckpt(_text_stack, crossattn_packed)
            img = _ckpt(self.first, tokens_1_N_M)
        else:
            txt = _text_stack(crossattn_packed)                 # [1, ΣL, features]
            img = self.first(tokens_1_N_M)                      # [1, ΣN, features]

        # ── 逐图 timestep 向量（G 行）───────────────────────────────────────
        t_flat = timesteps_G.reshape(-1).float()
        if t_flat.shape[0] != G:
            raise ValueError(f"timesteps_G 有 {t_flat.shape[0]} 项，期望 G={G}")
        t_vec = self.tmlp(temb(t_flat, self.config.tdim, device=device, dtype=img.dtype))  # (G,1,F)
        t_vec_1_G = t_vec[:, 0, :].unsqueeze(0)                 # (1,G,F)
        tvec_1_G = self.tproj(t_vec_1_G)                        # (1,G,6F)

        # ── 组装 combined 序列：每图 [txt_i ; img_i]，并记录 image token 位置 ──
        seg_lens = [tl + vl for tl, vl in zip(text_seqlens, visual_seqlens)]
        total = sum(seg_lens)
        parts = []
        img_index = torch.empty(sum(visual_seqlens), dtype=torch.long, device=device)
        pos = torch.zeros(1, total, 3, device=device)
        t_off = 0
        v_off = 0
        c_off = 0
        i_off = 0
        for i in range(G):
            tl, vl = text_seqlens[i], visual_seqlens[i]
            parts.append(txt[:, t_off:t_off + tl])
            parts.append(img[:, v_off:v_off + vl])
            # image token 在 combined 中的位置（用于抽取输出）
            img_index[i_off:i_off + vl] = torch.arange(
                c_off + tl, c_off + tl + vl, device=device
            )
            # RoPE 位置：text 全 0；image (frame=0, row, col)
            pos[0, c_off + tl: c_off + tl + vl, 1] = grid_1_2_N[0, 0, v_off:v_off + vl].float()
            pos[0, c_off + tl: c_off + tl + vl, 2] = grid_1_2_N[0, 1, v_off:v_off + vl].float()
            t_off += tl
            v_off += vl
            c_off += tl + vl
            i_off += vl
        combined = torch.cat(parts, dim=1)                      # [1, Σ(L+N), F]

        # token → 图 的调制索引
        counts = torch.tensor(seg_lens, device=device)
        mod_index = torch.repeat_interleave(torch.arange(G, device=device), counts)

        if use_sdpa_seg:
            self_bias = _SegLens(seg_lens)
        elif use_xf:
            self_bias = cached_block_diag_mask(tuple(seg_lens))
        else:
            self_bias = block_diag_bool_mask(seg_lens, device)

        freqs = self.posemb(pos)

        _ckpt_until = self._checkpoint_from_block(use_checkpoint, checkpoint_skip_last)
        for _i, block in enumerate(self.blocks):
            def _run(x_in, _b=block):
                return _b(x_in, tvec_1_G, freqs, self_bias, mod_index=mod_index)
            if _i < _ckpt_until:
                combined = _ckpt(_run, combined)
            else:
                combined = _run(combined)

        out = self.last.forward_mod(combined, t_vec_1_G, mod_index)   # [1, Σ(L+N), patch²·C]
        out_img = out[:, img_index, :]                                # [1, ΣN, patch²·C]
        return out_img


# ---------------------------------------------------------------------- #
# checkpoint → config 推断                                                 #
# ---------------------------------------------------------------------- #
def infer_config_from_state_dict(sd: dict) -> SingleMMDiTConfig:
    """从官方 raw/turbo safetensors 的 state dict 推断构型。

    已知发布构型（12B large_wide）直接返回预设；否则按权重形状推断（headdim 假定 128 —
    发布模型如是；非 128 headdim 的自定义 ckpt 请显式传 config）。
    """
    first_w = sd["first.weight"]                      # [features, C·p²]
    features = int(first_w.shape[0])
    n_blocks = 0
    while f"blocks.{n_blocks}.attn.wq.weight" in sd:
        n_blocks += 1
    if features == 6144 and n_blocks == 28:
        return KREA2_LARGE_WIDE

    headdim = 128
    kv_dim = int(sd["blocks.0.attn.wk.weight"].shape[0])
    txt_dim = int(sd["txtmlp.1.weight"].shape[1])
    txt_layers = int(sd["txtfusion.projector.weight"].shape[1])
    txt_kv = int(sd["txtfusion.refiner_blocks.0.attn.wk.weight"].shape[0])
    mlp_dim = int(sd["blocks.0.mlp.gate.weight"].shape[0])
    # SwiGLU: mlpdim = ceil(int(2f/3)·mult / 128)·128 → mult 反推（发布模型 mult=4）
    mult = max(1, round(mlp_dim / (int(2 * features / 3))))
    in_feats = int(first_w.shape[1])                  # C·p²
    # 发布模型 C=16, p=2 → 64。这里按 p=2 假定拆分。
    patch = 2
    channels = in_feats // (patch * patch)
    txt_headdim = txt_dim // 20 if txt_dim % 20 == 0 else headdim
    return SingleMMDiTConfig(
        features=features,
        tdim=256,
        txtdim=txt_dim,
        heads=features // headdim,
        kvheads=kv_dim // headdim,
        multiplier=mult,
        layers=n_blocks,
        patch=patch,
        channels=channels,
        txtheads=max(1, txt_dim // txt_headdim),
        txtkvheads=max(1, txt_kv // txt_headdim),
        txtlayers=txt_layers,
    )
