# Anima model - MiniTrainDIT with LLMAdapter for bridging Qwen3 embeddings
# Based on ComfyUI's comfy/ldm/anima/model.py

import torch
from torch import nn
import torch.nn.functional as F

from models.anima_modeling_core import Anima as CoreAnima
from models.anima_modeling_core import MiniTrainDIT


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


class Anima(CoreAnima):
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
        if text_ids is not None and self.llm_adapter is not None:
            return self.llm_adapter(
                text_embeds,
                text_ids,
                target_attention_mask=target_attention_mask,
                source_attention_mask=source_attention_mask,
            )
        return text_embeds
