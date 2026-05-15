"""Anima transformer / VAE / 文本编码器加载。

通过动态 import 从 `diffusion-pipe` 仓库（`anima_modeling.py` / `cosmos_predict2_modeling.py`
/ `wan/vae2_1.py`）拿到模型类，避免把整套外部代码作为强依赖。

公开函数：
- `find_diffusion_pipe_root` —— 在常见目录查找 anima_modeling.py 所在路径
- `load_module_from_path` —— 通用动态 import 助手
- `ensure_models_namespace` —— 把 repo_root 加进 sys.path
- `load_anima_model` —— 加载 Anima transformer + 智能选择 model_channels/blocks 配置
- `load_vae` —— 加载 WanVAE，绑定 mean/std 归一化常量，返回 wrapper
- `load_text_encoders` —— 加载 Qwen3 (LM hidden_states 通道) + T5 tokenizer
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from pathlib import Path

import torch

from trainer.checkpoint import _load_safetensors_state_dict, _load_weights_best_effort

logger = logging.getLogger(__name__)


def find_diffusion_pipe_root():
    """查找 diffusion-pipe 模型代码路径"""
    candidates = [
        Path(__file__).resolve().parent.parent / "diffusion_models",
        Path(__file__).resolve().parent.parent / "models",
        Path(os.environ.get("DIFFUSION_PIPE_ROOT", "")) if os.environ.get("DIFFUSION_PIPE_ROOT") else None,
    ]
    for candidate in candidates:
        if candidate and (candidate / "anima_modeling.py").exists():
            return candidate
        if candidate and (candidate / "models" / "anima_modeling.py").exists():
            return candidate / "models"
    raise RuntimeError("找不到 anima_modeling.py，请设置 DIFFUSION_PIPE_ROOT 或放置模型代码")


def load_module_from_path(module_name, file_path):
    """动态加载 Python 模块"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_models_namespace(repo_root):
    """确保 models 命名空间可用"""
    repo_root = Path(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if str(repo_root.parent) not in sys.path:
        sys.path.insert(0, str(repo_root.parent))


def load_anima_model(transformer_path, device, dtype, repo_root):
    """加载 Anima transformer 模型"""
    from safetensors import safe_open

    ensure_models_namespace(repo_root)

    # 加载模型类
    cosmos_modeling = load_module_from_path(
        "cosmos_predict2_modeling",
        repo_root / "cosmos_predict2_modeling.py",
    )
    anima_modeling = load_module_from_path(
        "anima_modeling",
        repo_root / "anima_modeling.py",
    )
    Anima = anima_modeling.Anima

    # 从 checkpoint 推断配置
    with safe_open(transformer_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.endswith("x_embedder.proj.1.weight"):
                w = f.get_tensor(k)
                break

    in_channels = (w.shape[1] // 4) - 1  # concat_padding_mask=True
    model_channels = w.shape[0]

    if model_channels == 2048:
        num_blocks, num_heads = 28, 16
    elif model_channels == 5120:
        num_blocks, num_heads = 36, 40
    else:
        raise RuntimeError(f"未知的 model_channels={model_channels}")

    config = dict(
        max_img_h=240, max_img_w=240, max_frames=128,
        in_channels=in_channels, out_channels=16,
        patch_spatial=2, patch_temporal=1,
        concat_padding_mask=True,
        model_channels=model_channels,
        num_blocks=num_blocks, num_heads=num_heads,
        crossattn_emb_channels=1024,
        pos_emb_cls="rope3d", pos_emb_learnable=True,
        pos_emb_interpolation="crop",
        use_adaln_lora=True, adaln_lora_dim=256,
        rope_h_extrapolation_ratio=4.0 if in_channels == 16 else 3.0,
        rope_w_extrapolation_ratio=4.0 if in_channels == 16 else 3.0,
        rope_t_extrapolation_ratio=1.0,
    )

    model = Anima(**config)

    sd = _load_safetensors_state_dict(Path(transformer_path))
    _load_weights_best_effort(model, sd, label="Transformer")

    # 如果 checkpoint 中完全没有 llm_adapter 权重，随机初始化会把 cross-attn 条件搞乱，直接禁用更安全
    has_llm_adapter = any("llm_adapter" in k for k in sd.keys())
    if not has_llm_adapter and hasattr(model, "llm_adapter"):
        try:
            model.llm_adapter = None
            logger.warning("检测到 checkpoint 不包含 llm_adapter 权重：已禁用 llm_adapter（回退为直接使用 Qwen embeddings）")
        except Exception:
            pass
    model = model.to(device=device, dtype=dtype)
    model.requires_grad_(False)

    logger.info(f"Anima 模型加载完成: {model_channels}ch, {num_blocks} blocks")
    return model


def load_vae(vae_path, device, dtype, repo_root):
    """加载 VAE"""
    wan_vae = load_module_from_path("wan_vae", repo_root / "wan" / "vae2_1.py")
    WanVAE = wan_vae.WanVAE_

    cfg = dict(
        dim=96, z_dim=16, dim_mult=[1, 2, 4, 4],
        num_res_blocks=2, attn_scales=[],
        temperal_downsample=[False, True, True], dropout=0.0,
    )

    model = WanVAE(**cfg).eval().requires_grad_(False)

    sd = _load_safetensors_state_dict(Path(vae_path))
    _load_weights_best_effort(model, sd, label="VAE")
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


def load_text_encoders(qwen_path, t5_tokenizer_path, device, dtype):
    """加载文本编码器"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer

    qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
    qwen_model = AutoModelForCausalLM.from_pretrained(
        qwen_path, torch_dtype=dtype, trust_remote_code=True
    ).to(device).eval().requires_grad_(False)

    if t5_tokenizer_path and Path(t5_tokenizer_path).exists():
        t5_tokenizer = T5Tokenizer.from_pretrained(t5_tokenizer_path)
    else:
        t5_tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")

    logger.info("文本编码器加载完成")
    return qwen_model, qwen_tokenizer, t5_tokenizer
