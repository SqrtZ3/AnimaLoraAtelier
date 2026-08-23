r"""Qwen3-VL **文本塔**的纯 JAX 前向（TPU 侧现算 krea2 文本条件）。

## 这个文件解决什么

krea2 的文本条件是 Qwen3-VL-4B 的 **12 层 hidden 堆叠** `[L, 12, 2560]`，bf16 下
约 61.4KB/token —— 96 张图的 `<stem>.textfeat.npz` 合计 **1.7GB**，而它的全部
信息来源只是 96 条 caption（合计 0.14MB）。原链路把这 1.7GB 在本地算好再传上
Kaggle，既伤本地磁盘也伤上行带宽，还得为 caption 明文做脱敏。

本文件把这一步搬到 TPU：上传的只有 **token ids**（`tools/dump_caption_ids.py`
产出，96 条约 100KB），文本塔权重从 Kaggle Dataset / HF 直读，
`<stem>.textfeat.npz` 在真机上现算。

**只做前向、只跑到需要的那一层**，不训练、不挂 LoRA（TE 在本仓库里恒为冻结）。

## 结构（config.json 实测，非推断）

36 层 / hidden 2560 / GQA 32:8 / head_dim 128 / SwiGLU 9728 / RMSNorm eps 1e-6 /
rope_theta 5e6 / vocab 151936。与 Qwen3 稠密模型同构，没有 MoE、没有滑窗。

## 三条"猜错就静默全错"的口径，全部由闸门裁决过

1. **tap 语义**：`trainer/model_family.py:KREA2_SELECT_LAYERS = (2,5,…,35)`
   索引的是 HF 的 `hidden_states[k]`，闸门 `dump_qwen3vl_ref.py` 实测
   `hidden_states[k]`（k<层数）**= 第 k 层的输入**（未过 final norm），
   `hidden_states[层数]` 才是 final norm 输出。所以：
     * 最大 tap 35 → **只跑 0~34 共 35 层**，`layers.35.*` 与 `norm.weight`
       一个字节都不用加载（省 ~0.2GB HBM 与 1/36 的算力）；
     * 反过来，若哪天 tap 里出现 `层数`，本文件会 fail-fast 而不是静默少跑。
2. **位置编码**：Qwen3-VL 的 mrope 在**纯文本**输入下三路 position 完全相同
   （`Qwen3VLModel.compute_3d_position_ids` 在无图无视频时回 None，
   `Qwen3VLTextModel` 随即用 `arange`），`apply_interleaved_mrope` 于是退化成
   标准 RoPE。本文件直接按标准 RoPE（rotate_half 半分对偶，**不是** krea2 DiT
   那种 interleaved 相邻对偶）实现，等价性由 K3 闸门逐层比对。
3. **padding 摆法**：`position_ids` 用的是 `arange`，**不看 attention_mask**。
   于是 `trainer/model_family.py:_encode_krea2_batch` 在 `max_length<=0` 分支下
   把 suffix 拼在 padding **之后**，短 caption 的 suffix 会落在被推后的位置上
   —— 闸门实测同一条 caption「中段 padding」与「单条无 padding」的 hidden
   max|Δ|=6.7e-1（**不是**等价的）。已落盘的缓存是 `cache_text_features.py`
   的 B=1 逐条口径（无 padding），所以本文件一律按
   **[prefix ; caption ; suffix ; PAD…]（右侧 padding）** 摆放 —— 闸门实测
   右侧 padding 与单条逐 bit 相同（causal + 右 padding 下 pad 位不可见）。

## 数值口径

权重 bf16 存、matmul fp32 累加（TPU 默认）、RMSNorm 在 fp32 算完降回运行 dtype
再乘 scale（逐字对齐 `Qwen3VLTextRMSNorm.forward`）、softmax 在 fp32。
与 torch eager 唯一的非精确处是 **QKᵀ 的中间结果**：torch eager 在 bf16 下会把
它舍到 bf16 再 softmax，本文件保留 fp32（与 SDPA/flash 一致，也更准）。
fp32 对拍时两条路完全相同，该差异不出现；bf16 下的实际影响由
`check_qwen3vl_real.py` 拿真权重量化。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

try:
    from .krea2_jax import _read_safetensors_map, _to_jax
except ImportError:                     # jax_tpu/ 直接在 sys.path 上（tests/ 走这条）
    from krea2_jax import _read_safetensors_map, _to_jax

PyTree = Any

#: 与 `trainer/model_family.py:KREA2_SELECT_LAYERS` 同源（改一处必须改两处，
#: 本文件的 `select_layers` 参数默认取它）。
KREA2_SELECT_LAYERS: Tuple[int, ...] = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)

#: 真 checkpoint 里文本塔的键前缀（`model.safetensors.index.json` 实测）。
TEXT_PREFIX = "model.language_model."


@dataclass(frozen=True)
class QwenTextConfig:
    """Qwen3-VL-4B-Instruct 的 `text_config`（默认值 = 真模型实测值）。"""
    hidden: int = 2560
    layers: int = 36
    heads: int = 32
    kv_heads: int = 8
    head_dim: int = 128
    intermediate: int = 9728
    vocab: int = 151936
    rope_theta: float = 5_000_000.0
    eps: float = 1e-6

    @property
    def kv_groups(self) -> int:
        return self.heads // self.kv_heads


# ── 基础算子 ──────────────────────────────────────────────────────────────────
def rms_norm(x: jnp.ndarray, w: jnp.ndarray, eps: float) -> jnp.ndarray:
    """`Qwen3VLTextRMSNorm.forward` 逐字对齐：

        fp32 算 rsqrt(mean(x²)+eps) -> **先降回 x 的 dtype** -> 再乘 weight。

    先降 dtype 再乘 scale 这个次序不是随手写的：weight 是 bf16，torch 那边
    `self.weight * hidden_states.to(input_dtype)` 的乘法就发生在 bf16 上。
    """
    dt = x.dtype
    x32 = x.astype(jnp.float32)
    v = jnp.mean(jnp.square(x32), axis=-1, keepdims=True)
    return w * (x32 * jax.lax.rsqrt(v + eps)).astype(dt)


def rope_tables(seq_len: int, cfg: QwenTextConfig, dtype, offset: int = 0
                ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """标准 RoPE 的 (cos, sin)，形状 [L, head_dim]（half-split 拼接口径）。

    `inv_freq = 1/theta^(arange(0,hd,2)/hd)`、`emb = cat(freqs, freqs)`
    —— 与 `Qwen3VLTextRotaryEmbedding.compute_default_rope_parameters` +
    `forward` 相同。fp32 算完再降到运行 dtype（torch 也是 `cos.to(x.dtype)`）。
    """
    hd = cfg.head_dim
    inv = 1.0 / (cfg.rope_theta ** (jnp.arange(0, hd, 2, dtype=jnp.float32) / hd))
    pos = jnp.arange(seq_len, dtype=jnp.float32) + offset
    freqs = pos[:, None] * inv[None, :]                     # [L, hd/2]
    emb = jnp.concatenate([freqs, freqs], axis=-1)          # [L, hd]
    return jnp.cos(emb).astype(dtype), jnp.sin(emb).astype(dtype)


def _rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    """`modeling_qwen3_vl.rotate_half`：**前后半分**成对，不是相邻两维成对。"""
    half = x.shape[-1] // 2
    return jnp.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_rope(x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray) -> jnp.ndarray:
    """x [B, L, H, hd]；cos/sin [L, hd] 在 head 轴广播。"""
    c, s = cos[None, :, None, :], sin[None, :, None, :]
    return x * c + _rotate_half(x) * s


def _expand_kv(t: jnp.ndarray, groups: int) -> jnp.ndarray:
    """[B, L, Hkv, hd] -> [B, L, Hkv*groups, hd]，分组约定与 `repeat_kv` 相同
    （kv 头 j ↔ q 头 [j·g, (j+1)·g)）。"""
    return t if groups == 1 else jnp.repeat(t, groups, axis=-2)


def causal_attention(q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray,
                     scale: float) -> jnp.ndarray:
    """因果注意力（GQA 在内部展开）。q [B,L,Hq,hd]，k/v [B,L,Hkv,hd]。

    logits/softmax 全程 fp32（见模块 docstring 的"数值口径"）。
    """
    k = _expand_kv(k, q.shape[-2] // k.shape[-2])
    v = _expand_kv(v, q.shape[-2] // v.shape[-2])
    logits = jnp.einsum("blhd,bthd->bhlt", q, k,
                        preferred_element_type=jnp.float32) * scale
    L, T = logits.shape[-2], logits.shape[-1]
    mask = jnp.tril(jnp.ones((L, T), dtype=bool))
    logits = jnp.where(mask, logits, jnp.float32(-jnp.inf))
    w = jax.nn.softmax(logits, axis=-1).astype(v.dtype)
    return jnp.einsum("bhlt,bthd->blhd", w, v,
                      preferred_element_type=jnp.float32).astype(v.dtype)


# ── 单层前向 ──────────────────────────────────────────────────────────────────
def decoder_layer(h: jnp.ndarray, p: Dict[str, jnp.ndarray], cfg: QwenTextConfig,
                  cos: jnp.ndarray, sin: jnp.ndarray) -> jnp.ndarray:
    """`Qwen3VLTextDecoderLayer.forward`（pre-norm、无 bias、q/k-norm 在 head_dim 上）。"""
    hd, dt = cfg.head_dim, h.dtype
    x = rms_norm(h, p["ln1"], cfg.eps)
    shp = lambda z, n: z.reshape(*z.shape[:-1], n, hd)
    q = rms_norm(shp(x @ p["wq"].T.astype(dt), cfg.heads), p["q_norm"], cfg.eps)
    k = rms_norm(shp(x @ p["wk"].T.astype(dt), cfg.kv_heads), p["k_norm"], cfg.eps)
    v = shp(x @ p["wv"].T.astype(dt), cfg.kv_heads)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    a = causal_attention(q, k, v, hd ** -0.5)
    h = h + a.reshape(*a.shape[:-2], cfg.heads * hd) @ p["wo"].T.astype(dt)

    x = rms_norm(h, p["ln2"], cfg.eps)
    g = jax.nn.silu(x @ p["gate"].T.astype(dt)) * (x @ p["up"].T.astype(dt))
    return h + g @ p["down"].T.astype(dt)


# ── 整塔前向（scan over 层）────────────────────────────────────────────────────
def n_layers_needed(select_layers: Sequence[int]) -> int:
    """要跑多少层。tap k = 第 k 层的**输入**，所以最大 tap 就是要跑的层数。"""
    return int(max(select_layers))


def forward_taps(params: PyTree, cfg: QwenTextConfig, ids: jnp.ndarray,
                 select_layers: Sequence[int] = KREA2_SELECT_LAYERS,
                 compute_dtype=None) -> jnp.ndarray:
    """ids [B, L] int32 -> 堆叠 tap [B, L, n_taps, D]（与 torch 侧
    `torch.stack([hidden_states[i] for i in KREA2_SELECT_LAYERS], dim=2)` 同形同序）。

    右侧 padding 位照常参与前向（causal 下它们不影响任何有效位），调用方按有效
    长度切片即可 —— 见模块 docstring 口径 3。

    `compute_dtype` 指定时，残差流按该 dtype 走、权重逐用上转（存储仍是加载时的
    dtype）。给 fp32 就得到"bf16 权重 + fp32 计算"的参考线 —— `check_qwen3vl_real.py`
    用它把"移植误差"与"bf16 本身的噪声"分开。默认 None = 跟随权重 dtype。
    """
    n_run = n_layers_needed(select_layers)
    h = params["embed"][ids]                                  # [B, L, D]
    if compute_dtype is not None:
        h = h.astype(compute_dtype)
    cos, sin = rope_tables(ids.shape[1], cfg, h.dtype)

    def body(carry, p):
        return decoder_layer(carry, p, cfg, cos, sin), carry   # 先发 tap 再算

    h_out, ys = jax.lax.scan(body, h, params["layers"])        # ys [n_run, B, L, D]
    # ys[k] = hidden_states[k]（第 k 层输入）；h_out = hidden_states[n_run]。
    all_h = jnp.concatenate([ys, h_out[None]], axis=0)
    return jnp.stack([all_h[i] for i in select_layers], axis=2)


# ── 权重加载 ──────────────────────────────────────────────────────────────────
def _entry_map(src: str, token: Optional[str] = None):
    """单文件 / 目录（含 index.json 分片）/ URL -> 合并后的 {name: (dt, shape, reader)}。

    真 checkpoint 是 2 分片（`model-0000{1,2}-of-00002.safetensors`），
    `krea2_jax._read_safetensors_map` 只认单文件，这里做合并。
    """
    p = Path(src)
    if str(src).startswith(("http://", "https://")) or p.is_file():
        return _read_safetensors_map(str(src), token)[0]
    if not p.is_dir():
        raise FileNotFoundError(f"文本塔权重路径不存在: {src}")
    files = sorted(p.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"{p} 下没有 *.safetensors")
    merged: Dict[str, Any] = {}
    for f in files:
        for name, ent in _read_safetensors_map(str(f), token)[0].items():
            if name in merged:
                raise ValueError(f"分片间键冲突: {name}（{f.name}）")
            merged[name] = ent
    return merged


def infer_config(entries, prefix: str = TEXT_PREFIX) -> QwenTextConfig:
    """从权重形状推构型。head_dim 不从 hidden/heads 推 —— Qwen3 的
    `head_dim`(128) × `num_attention_heads`(32) = 4096 ≠ hidden(2560)，
    按"hidden/heads"猜会得到 80 然后一路广播错。所以 heads 由 q_proj 的输出维
    除以 head_dim 得到，而 head_dim 由 **q_norm 的长度**给出（它就是 head_dim）。
    """
    def shp(name):
        return tuple(entries[prefix + name][1])

    hidden = shp("layers.0.input_layernorm.weight")[0]
    head_dim = shp("layers.0.self_attn.q_norm.weight")[0]
    heads = shp("layers.0.self_attn.q_proj.weight")[0] // head_dim
    kv_heads = shp("layers.0.self_attn.k_proj.weight")[0] // head_dim
    inter = shp("layers.0.mlp.gate_proj.weight")[0]
    vocab = shp("embed_tokens.weight")[0]
    n = len({k.split("layers.")[1].split(".")[0]
             for k in entries if k.startswith(prefix + "layers.")})
    return QwenTextConfig(hidden=hidden, layers=n, heads=heads, kv_heads=kv_heads,
                          head_dim=head_dim, intermediate=inter, vocab=vocab)


#: 层内张量 -> checkpoint 里的后缀。
_LAYER_KEYS = {
    "ln1": "input_layernorm.weight",
    "ln2": "post_attention_layernorm.weight",
    "wq": "self_attn.q_proj.weight", "wk": "self_attn.k_proj.weight",
    "wv": "self_attn.v_proj.weight", "wo": "self_attn.o_proj.weight",
    "q_norm": "self_attn.q_norm.weight", "k_norm": "self_attn.k_norm.weight",
    "gate": "mlp.gate_proj.weight", "up": "mlp.up_proj.weight",
    "down": "mlp.down_proj.weight",
}


def load_text_tower(src: str, dtype=jnp.bfloat16,
                    select_layers: Sequence[int] = KREA2_SELECT_LAYERS,
                    prefix: str = TEXT_PREFIX,
                    token: Optional[str] = None,
                    keep_ids: Optional[np.ndarray] = None,
                    device_put: Optional[Callable] = None,
                    cfg: Optional[QwenTextConfig] = None,
                    ) -> Tuple[PyTree, QwenTextConfig]:
    """读文本塔权重 -> (params, cfg)。**只读用得上的那些张量**。

    * `layers.{n_run}..` 与 `norm.weight`、以及整个 `model.visual.*` 都不读
      （n_run = max(select_layers)，见模块 docstring 口径 1）；
    * `keep_ids` 给一个**升序去重**的 token id 数组时，embedding 只保留这些行
      （真模型全表 151936×2560 bf16 = 778MB，而一个数据集实际用到的 token
      通常只有几千个）。此时返回的 params 里带 `id_map`，`encode_ids` 会用它
      把原始 id 重映射到紧凑行号 —— 调用方不用管。
    * `device_put(name, arr)` 非 None 时逐张放置（分片/指定设备）。
    """
    entries = _entry_map(src, token)
    missing_prefix = not any(k.startswith(prefix) for k in entries)
    if missing_prefix:
        raise KeyError(
            f"{src} 里没有前缀 {prefix!r} 的张量。实际前 5 个键：{list(entries)[:5]}\n"
            f"  （HF 官方 Qwen3-VL checkpoint 是 'model.language_model.'；"
            f"纯文本导出件可能是 'model.' 或空前缀，用 prefix= 指定）")
    if cfg is None:
        cfg = infer_config(entries, prefix)
    n_run = n_layers_needed(select_layers)
    if n_run > cfg.layers:
        raise ValueError(f"select_layers 最大 {n_run} 超过模型层数 {cfg.layers}")
    if n_run == cfg.layers:
        # tap == 层数 时语义是 final norm 输出（闸门实测），本文件没实现那一支。
        raise ValueError(
            f"select_layers 含 {n_run} = 层数，该 tap 的语义是 final norm 的输出，"
            f"不是第 {n_run} 层的输入。本实现不覆盖，请显式处理。")

    need = ["embed_tokens.weight"] + [f"layers.{i}.{suf}"
                                      for i in range(n_run) for suf in _LAYER_KEYS.values()]
    lack = [n for n in need if prefix + n not in entries]
    if lack:
        raise KeyError(
            f"{src} 缺 {len(lack)} 个必需张量，例如 {lack[:3]}。\n"
            f"  官方 checkpoint 是**分片**的（model-0000x-of-0000y.safetensors）——\n"
            f"  传目录（本模块会自动合并分片），不要传单个分片文件。\n"
            f"  HTTP URL 只能指向单文件 checkpoint。")

    def put(name: str, arr: jnp.ndarray) -> jnp.ndarray:
        return arr if device_put is None else jax.device_put(arr, device_put(name, arr))

    # embedding：裁剪在 **host 侧**做（全表 151936×2560 bf16 = 778MB，先整块上卡
    # 再 take 会白占一次 HBM）。
    dt_e, _shape_e, rd_e = entries[prefix + "embed_tokens.weight"]
    raw_emb = rd_e()
    id_map = None
    if keep_ids is not None:
        keep = np.asarray(keep_ids, dtype=np.int64)
        raw_emb = raw_emb[keep]
        id_map = np.full(int(cfg.vocab), -1, np.int32)
        id_map[keep] = np.arange(keep.size, dtype=np.int32)
    emb = put("embed_tokens.weight", _to_jax(raw_emb, dt_e, dtype))
    del raw_emb

    # scan 要求逐张量沿层轴堆叠。**在 host 上按原始位模式堆完再上卡**：
    # 逐层 device_put 后再 jnp.stack 会让最大那组（down [35,2560,9728] bf16
    # = 1.74GB）瞬时占两份 HBM，白抬 ~1.7GB 峰值。
    def stack_layers(suf: str) -> jnp.ndarray:
        dts, arrs = [], []
        for i in range(n_run):
            dt, _shape, rd = entries[prefix + f"layers.{i}.{suf}"]
            dts.append(dt)
            arrs.append(rd())
        if len(set(dts)) != 1:
            raise ValueError(f"层间 dtype 不一致 ({suf}): {sorted(set(dts))}")
        a = np.stack(arrs)
        arrs.clear()
        return put(f"layers.*.{suf}", _to_jax(a, dts[0], dtype))

    layers = {k: stack_layers(suf) for k, suf in _LAYER_KEYS.items()}
    params = {"embed": emb, "layers": layers}
    if id_map is not None:
        params["id_map"] = id_map
    return params, cfg


# ── 编码入口 ──────────────────────────────────────────────────────────────────
def _bucket(n: int, q: int) -> int:
    return int(q * ((n + q - 1) // q))


def encode_ids(params: PyTree, cfg: QwenTextConfig, id_lists: Sequence[np.ndarray],
               prefix_len: int,
               select_layers: Sequence[int] = KREA2_SELECT_LAYERS,
               quantum: int = 128, batch: int = 1,
               progress: Optional[Callable[[int, int], None]] = None,
               compute_dtype=None) -> List[np.ndarray]:
    """一批 caption 的完整 id 序列 -> 每条 `[L_valid, n_taps, D]`（numpy，dtype 同权重）。

    `id_lists[i]` 是 **[prefix ; caption ; suffix]** 的完整 ids（由
    `tools/dump_caption_ids.py` 产出）；返回值已按 `prefix_len` 去掉系统提示词
    那一段，与 `trainer/model_family.py:_encode_krea2_batch` 的
    `hiddens[:, _KREA2_PREFIX_IDX:]` 同口径。

    长度按 `quantum` 向上取整后**右侧 padding**（口径 3：与单条编码逐 bit 等价），
    同长度的条目才会被凑进同一批 —— 每种 (batch, L) 形状编译一次。
    """
    fwd = jax.jit(lambda p, x: forward_taps(p, cfg, x, select_layers, compute_dtype))
    id_map = params.get("id_map")
    order = sorted(range(len(id_lists)), key=lambda i: len(id_lists[i]))
    out: List[Optional[np.ndarray]] = [None] * len(id_lists)

    groups: Dict[int, List[int]] = {}
    for i in order:
        groups.setdefault(_bucket(len(id_lists[i]), quantum), []).append(i)

    done = 0
    for L, idxs in sorted(groups.items()):
        for s in range(0, len(idxs), batch):
            chunk = idxs[s:s + batch]
            arr = np.zeros((len(chunk), L), np.int32)
            valid = np.zeros((len(chunk), L), bool)
            for r, i in enumerate(chunk):
                v = np.asarray(id_lists[i], np.int32)
                arr[r, :v.size] = v
                valid[r, :v.size] = True
            if id_map is not None:
                mapped = id_map[arr]
                # 只校验有效位：padding 位填的 0 未必在 keep_ids 里，而它在
                # causal + 右 padding 下对任何有效位都不可见（口径 3），随便
                # 指到哪一行都行 —— 但必须是**合法行号**，否则 gather 越界。
                if valid.any() and int(mapped[valid].min()) < 0:
                    bad = int(arr[valid][int(np.argmin(mapped[valid]))])
                    raise KeyError(
                        f"token id {bad} 不在 keep_ids 里 —— embedding 被裁剪过，"
                        f"但这条 caption 用到了没保留的行。用同一份 ids 重新加载。")
                arr = np.where(valid, mapped, 0).astype(np.int32)
            feats = np.asarray(fwd(params, jnp.asarray(arr)))
            for r, i in enumerate(chunk):
                out[i] = feats[r, prefix_len:len(id_lists[i])]
            done += len(chunk)
            if progress is not None:
                progress(done, len(id_lists))
    return [o for o in out]                     # type: ignore[misc]
