"""Anima 的 **LLMAdapter**（文本条件桥）—— 从 upstream `models/anima_modeling.py`
摘录的 6 个符号（第 12~220 行），逐字。

## 为什么只要这一小块

`<stem>.textfeat.npz` 里的 `cross` 是
`llm_adapter(qwen_emb, t5_ids, t5_attn, qwen_attn) * t5_w`。
upstream 的 `cache_text_features.py` 走 `load_anima_model()` 加载**整个 4.2GB 底模**
只为调一句 `dit.preprocess_text_embeds(...)`，而那句在 `anima_modeling.py:241` 里
就是 `return self.llm_adapter(...)` —— DiT 主体（28 层 blocks / x_embedder /
final_layer）在文本缓存这一步一个字节都用不到。

所以这里只搬 adapter 本体，并配 `load_llm_adapter()` 从 safetensors 里**只读**
`net.llm_adapter.*` 那 118 个键：**134.7M 参数 / 269MB**，而不是 4182MB
（键名前缀是实测的，不是猜的 —— 见 `UPSTREAM.md`）。数值上与走全量底模完全等价：
同一批权重、同一个 `nn.Module`、同一条 forward。

## 与 upstream 的唯一差异

`LLMAdapter.forward` 里对 `utils.npu_compat.expand_attn_mask` 的两次调用已删除。
那是昇腾 NPU 专用（FlashAttentionScore 不接受 Sq 维为 1 的广播 mask），upstream
自己的注释写明「CUDA/CPU 上 expand_attn_mask 是恒等映射，行为逐字节不变」。
本仓库的缓存工具只跑 CUDA/CPU，所以删掉它是逐 bit 无影响的。

行号与 sha256 见 `UPSTREAM.md`；漂移由 `tools/check_sync.py` 守。
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    x_embed = (x * cos) + (rotate_half(x) * sin)
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
        """重算 ``inv_freq``（non-persistent buffer，checkpoint 永远填不到它）。

        正常构造路径下 ``__init__`` 已经算好、本方法不会被调用；只有 meta device 构造
        （trainer/models.py 的 fast_init）在 ``to_empty()`` 之后需要它——那时 buffer
        里是未初始化内存。与 anima_modeling_core.RotaryEmbedding 保持一致。
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
            query_states = apply_rotary_pos_emb(query_states, cos, sin)
            cos, sin = position_embeddings_context
            key_states = apply_rotary_pos_emb(key_states, cos, sin)

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

        # upstream 此处调 `utils.npu_compat.expand_attn_mask` 把 Sq=1 的广播 mask
        # 展开（昇腾 FlashAttentionScore 不接受广播 mask）。upstream 自己的注释写明
        # 「CUDA/CPU 上 expand_attn_mask 是恒等映射，行为逐字节不变」，而本仓库的
        # 缓存工具只跑 CUDA/CPU —— 所以这两行删掉是逐 bit 无影响的。
        position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        position_ids_context = torch.arange(context.shape[1], device=x.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(x, position_ids)
        position_embeddings_context = self.rotary_emb(x, position_ids_context)
        for block in self.blocks:
            x = block(x, context, target_attention_mask=target_attention_mask, source_attention_mask=source_attention_mask, position_embeddings=position_embeddings, position_embeddings_context=position_embeddings_context)
        return self.norm(self.out_proj(x))


# ─────────────────────────────────────────────────────────────────────────────
# 以下是本仓库新增（upstream 无对应物）：只读 adapter 权重的加载器
# ─────────────────────────────────────────────────────────────────────────────

#: 底模里 adapter 权重的键前缀。**实测值**（`anima-base-v1.0.safetensors`：685 个
#: 键，其中 118 个以此开头，134.7M 参数 / 269MB），不是猜的。若哪天底模换了导出
#: 工具、前缀变了，`load_llm_adapter` 会 fail-fast 并把实际前缀列出来。
_ADAPTER_PREFIX = "net.llm_adapter."

#: Anima 的 adapter 构型（upstream `anima_modeling.py:231-239` 的
#: `self.llm_adapter = LLMAdapter(...)` 实参，逐字）。
ADAPTER_CONFIG = dict(
    source_dim=1024,   # Qwen3 embedding dimension
    target_dim=1024,   # Output dimension (matches crossattn_emb_channels)
    model_dim=1024,
    num_layers=6,
    num_heads=16,
    use_self_attn=True,
    layer_norm=False,
)


def load_llm_adapter(transformer_path, device, dtype):
    """从 anima 底模 safetensors 里**只**加载 LLMAdapter（269MB，不是 4.2GB）。

    返回一个 eval 模式、requires_grad=False 的 `LLMAdapter`，调用口径与
    upstream `dit.preprocess_text_embeds(text_embeds, text_ids, target_mask,
    source_mask)` 完全一致 —— 后者就是转发给 `self.llm_adapter(...)`
    （`anima_modeling.py:241-258`）。

    严格加载：118 个键必须**全部**命中，缺一个就 fail-fast。adapter 只有 118 个
    张量，「部分加载」没有合理场景 —— 静默少载会让 cross 条件错得看不出来
    （数值合法、训练照跑、条件全歪）。
    """
    from safetensors import safe_open

    model = LLMAdapter(**ADAPTER_CONFIG).eval().requires_grad_(False)
    want = set(model.state_dict().keys())

    with safe_open(str(transformer_path), framework="pt", device="cpu") as f:
        all_keys = list(f.keys())
        hit = {k[len(_ADAPTER_PREFIX):]: k
               for k in all_keys if k.startswith(_ADAPTER_PREFIX)}
        if not hit:
            cands = sorted({k.rsplit(".", 1)[0] for k in all_keys
                            if "llm_adapter" in k})
            raise RuntimeError(
                f"{transformer_path} 里没有 {_ADAPTER_PREFIX}* 键（共 {len(all_keys)} 个键）。\n"
                f"  含 'llm_adapter' 的键前缀候选：{cands[:5] or '一个都没有'}\n"
                f"  —— 这个 checkpoint 可能不带 llm_adapter（那 cross 就只能是 Qwen "
                f"hidden，口径与本工具产出的缓存不同，不能混用），或导出前缀变了。")
        missing = sorted(want - set(hit))
        if missing:
            raise RuntimeError(
                f"adapter 权重缺 {len(missing)}/{len(want)} 个键，"
                f"前 5 个：{missing[:5]}。\n"
                f"  adapter 只有 {len(want)} 个张量，不存在'部分加载'的合理场景 —— "
                f"少载会让 cross 条件静默错误。")
        sd = model.state_dict()
        for mk in sorted(want):
            sd[mk].data.copy_(f.get_tensor(hit[mk]))

    model = model.to(device=device, dtype=dtype)
    unused = len(hit) - len(want)
    logger.info("LLMAdapter 加载完成：%d 个张量%s（只读 %s*，未加载 DiT 主体）",
                len(want), f"，另有 {unused} 个多余键被忽略" if unused else "",
                _ADAPTER_PREFIX)
    return model
