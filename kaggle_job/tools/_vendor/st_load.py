"""从 upstream 摘录的 **safetensors 流式加载 + VAE 构造**。

  * `_strip_prefixes` / `_pick_best_prefix_remap` / `_load_safetensors_into_model`
    <- `trainer/checkpoint.py`（逐字）
  * `load_vae` <- `trainer/models.py`（逐字，只把动态 `load_module_from_path`
    换成对 `wan_vae` 的直接 import —— 本仓库里 VAE 实现就在 `_vendor/wan_vae.py`，
    不需要满仓库找 `anima_modeling.py` 来定位模型代码目录）

行号与 sha256 见 `UPSTREAM.md`；漂移由 `tools/check_sync.py` 守。

`load_vae` 里的 latent mean/std 是 Anima/Qwen VAE 的**归一化常量**，抄错不会报错、
只会让所有 latent 偏移 —— 训练照跑，出图全灰。属于必须逐字对齐的那类数字。
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

def _strip_prefixes(key: str, prefixes: list[str]) -> str:
    """反复剥离前缀（支持 module.model. 这种复合前缀）"""
    if not prefixes:
        return key
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if key.startswith(p):
                key = key[len(p):]
                changed = True
    return key

def _pick_best_prefix_remap(sd_keys: list[str], model_keys: set[str]) -> tuple[list[str], int]:
    """从常见前缀组合里选择"命中最多 model_keys"的 remap 方案。

    返回 (prefixes, matched_count)。
    """
    candidates: list[tuple[str, list[str]]] = [
        ("none", []),
        ("net.", ["net."]),
        ("model.", ["model."]),
        ("module.", ["module."]),
        ("module.+model.", ["module.", "model."]),
        ("module.model.", ["module.model."]),
        ("diffusion_model.", ["diffusion_model."]),
        ("model.diffusion_model.", ["model.diffusion_model."]),
        ("transformer.", ["transformer."]),
        ("vae.", ["vae."]),
        ("first_stage_model.", ["first_stage_model."]),
        ("net.+model.", ["net.", "model."]),
        ("net.model.", ["net.model."]),
    ]

    best_prefixes: list[str] = []
    best_matched = -1
    for _name, prefixes in candidates:
        matched = 0
        for k in sd_keys:
            kk = _strip_prefixes(k, prefixes)
            if kk in model_keys:
                matched += 1
        if matched > best_matched:
            best_matched = matched
            best_prefixes = prefixes
    return best_prefixes, best_matched

def _load_safetensors_into_model(
    model: torch.nn.Module,
    path: Path,
    prefixes: list[str] | None = None,
    label: str = "Model",
    skip_buffer_patterns: tuple[str, ...] | None = None,
    strict_missing: bool = False,
) -> dict:
    """Stream-load safetensors weights directly into *model* parameters, one tensor at a time.

    Avoids creating a full state-dict copy in RAM.  For a 12B model (bf16 ~24 GB)
    this cuts peak host-RAM from ~72-96 GB (3-4 simultaneous copies in the old
    ``_load_safetensors_state_dict`` + ``_filter_recomputable_buffers`` +
    ``_load_weights_best_effort`` pipeline) down to ~24 GB (model itself + one
    transient tensor).

    Parameters
    ----------
    model : nn.Module
        Target model (typically on CPU).  Weights are loaded via
        ``param.data.copy_(tensor)`` in-place, so the model's own storage
        is the only persistent allocation.
    path : Path
        Path to ``.safetensors`` file.
    prefixes : list[str] | None
        Prefix list to strip (e.g. ``["model."]``).  If ``None``, the best
        prefix is auto-detected via ``_pick_best_prefix_remap``.
    label : str
        Label for log messages (e.g. ``"Transformer"``, ``"VAE"``).
    skip_buffer_patterns : tuple[str, ...] | None
        If a model key contains any of these substrings, the tensor is
        shape-checked; a mismatch means the buffer is recomputable (e.g.
        RoPE position embeddings) and is **skipped** -- the model's own
        ``reset_parameters()`` will have set it correctly.
    strict_missing : bool
        If ``True``, **any** missing key raises ``RuntimeError`` (Krea2
        semantics).  If ``False``, only critical layers (``x_embedder.``,
        ``blocks.``, ``final_layer.``) or coverage < 60% raise (Anima / VAE).

    Returns
    -------
    dict  with keys ``remap``, ``coverage``, ``missing``, ``unexpected``.
    """
    from safetensors import safe_open

    model_sd = model.state_dict()
    model_keys = set(model_sd.keys())

    with safe_open(path, framework="pt", device="cpu") as f:
        sd_keys = list(f.keys())

        # Auto-detect best prefix if not provided
        if prefixes is None:
            prefixes, _ = _pick_best_prefix_remap(sd_keys, model_keys)
        remap_name = "+".join(prefixes) if prefixes else "none"

        # Build mapping: model_key -> safetensors_key
        key_map: dict[str, str] = {}
        for sk in sd_keys:
            mk = _strip_prefixes(sk, prefixes)
            if mk in model_keys:
                key_map[mk] = sk

        # Stream-load: one tensor at a time, copy into model param, then free
        loaded = 0
        skipped_buffers: list[tuple[str, tuple, tuple]] = []
        for mk, sk in key_map.items():
            # Skip recomputable derivative buffers whose shape doesn't match
            if skip_buffer_patterns and any(pat in mk for pat in skip_buffer_patterns):
                param_shape = tuple(model_sd[mk].shape)
                ckpt_shape = tuple(f.get_slice(sk).get_shape())
                if param_shape != ckpt_shape:
                    skipped_buffers.append((mk, ckpt_shape, param_shape))
                    continue
            tensor = f.get_tensor(sk)
            model_sd[mk].data.copy_(tensor)
            del tensor
            loaded += 1

    if skipped_buffers:
        for name, ck_shape, md_shape in skipped_buffers[:5]:
            logger.info(
                "Drop recomputable buffer from ckpt: %s (ckpt=%s, model=%s) - "
                "模型 reset_parameters() 会重算",
                name, ck_shape, md_shape,
            )
        if len(skipped_buffers) > 5:
            logger.info("  ... 共 %d 个 recomputable buffer 被跳过", len(skipped_buffers))

    # Coverage stats (mirror _load_weights_best_effort log format)
    matched_keys = set(key_map.keys())
    missing = sorted(model_keys - matched_keys)
    unexpected = sorted(
        _strip_prefixes(k, prefixes) for k in sd_keys
        if _strip_prefixes(k, prefixes) not in model_keys
    )
    coverage = loaded / max(1, len(model_keys))

    logger.info(
        f"{label} 权重加载: remap={remap_name}, 匹配 {loaded}/{len(model_keys)} "
        f"({coverage:.1%}), missing={len(missing)}, unexpected={len(unexpected)}"
    )

    # Fail-fast checks
    if strict_missing and missing:
        raise RuntimeError(
            f"{label} 权重缺失 {len(missing)} 个 key（前 5 个: {missing[:5]}）。"
        )
    critical_prefixes = ("x_embedder.", "blocks.", "final_layer.")
    critical_missing = [k for k in missing if k.startswith(critical_prefixes)]
    if coverage < 0.60 or len(critical_missing) > 0:
        preview_missing = ", ".join(critical_missing[:8])
        raise RuntimeError(
            f"{label} 权重看起来没有正确加载（remap={remap_name}, coverage={coverage:.1%}）。"
            f"关键参数缺失: {preview_missing or 'N/A'}。\n"
            f"这通常表示你选错了 .safetensors（不是完整 transformer/vae 权重），或 "
            f"checkpoint key 前缀不匹配。"
        )

    return {
        "remap": remap_name,
        "coverage": coverage,
        "missing": missing,
        "unexpected": unexpected,
        # 在 ckpt 里有对应 key、但因 shape 不匹配被当作 recomputable 丢掉的 buffer。
        # 与 `missing` 合起来才是"模型里没被 checkpoint 填过的 key"的全集 —— fast_init
        # （meta 构造）路径必须拿到这个全集，才能验证没有未初始化的内存漏出去。
        "skipped": [name for name, _, _ in skipped_buffers],
    }

def load_vae(vae_path, device, dtype, repo_root=None, attn_chunk_tokens: int = 0):
    """加载 VAE

    `attn_chunk_tokens` > 0 时开启 VAE 自注意力的 query 分块（数学恒等，把 math-SDPA
    后端下 O(N²) 的峰值显存降到 O(chunk·N)）。0 = 关闭，走原来的整块 SDPA。

    与 upstream 的唯一差异：VAE 实现直接 `from . import wan_vae`，不再用
    `load_module_from_path(repo_root / "wan" / "vae2_1.py")` 动态定位 ——
    本仓库里它就在 `_vendor/wan_vae.py`（与上游 `models/wan/vae2_1.py` 逐字节相同）。
    `repo_root` 保留为可选参数只为调用兼容，本函数忽略它。
    """
    from . import wan_vae
    WanVAE = wan_vae.WanVAE_

    if int(attn_chunk_tokens or 0) > 0:
        wan_vae.set_vae_attn_chunk_tokens(int(attn_chunk_tokens))
        logger.info(
            "[vae-attn] query 分块已启用：chunk=%d token。VAE 中间块是单头全局注意力，"
            "SDPA 落到 math backend 时显存 O(N²)；分块后峰值 ≈ chunk·N，数学恒等（非近似）。",
            int(attn_chunk_tokens),
        )

    cfg = dict(
        dim=96, z_dim=16, dim_mult=[1, 2, 4, 4],
        num_res_blocks=2, attn_scales=[],
        temperal_downsample=[False, True, True], dropout=0.0,
    )

    model = WanVAE(**cfg).eval().requires_grad_(False)

    _load_safetensors_into_model(model, Path(vae_path), label="VAE")
    model = model.to(device=device, dtype=dtype)

    # VAE 归一化参数
    mean = torch.tensor([
        -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
    ], dtype=dtype, device=device)
    std = torch.tensor([
        2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
    ], dtype=dtype, device=device)

    class VAEWrapper:
        pass

    wrapper = VAEWrapper()
    wrapper.model = model
    wrapper.mean = mean
    wrapper.std = std
    wrapper.scale = [mean, 1.0 / std]

    logger.info("VAE 加载完成")
    return wrapper
