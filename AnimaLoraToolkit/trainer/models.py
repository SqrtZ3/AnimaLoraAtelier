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
import time
from pathlib import Path

import torch

from trainer.checkpoint import _load_safetensors_into_model

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
    """动态加载 Python 模块

    必须在 exec_module 之前把模块注册到 sys.modules，否则模块内定义的
    @dataclass 装饰器在 CPython 3.12+ 会因 sys.modules.get(cls.__module__)
    返回 None 而抛 AttributeError（bpo-120492）。
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def ensure_models_namespace(repo_root):
    """确保 models 命名空间可用"""
    repo_root = Path(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if str(repo_root.parent) not in sys.path:
        sys.path.insert(0, str(repo_root.parent))


# RoPE 内部的 derivative buffer 名字（前缀剥离后）。这些都是从 arange/dim 重新算的
# 静态张量，不存储训练学到的信息，shape 跟着 max_img_h/w/frames 变化。我们在 load 前
# 把它们从 checkpoint 里筛掉，避免 PyTorch load_state_dict 因 shape 不匹配 raise。
# `dim_spatial_range` / `dim_temporal_range` 跟 head_dim 关联，不随 max_img_* 变，但
# 为统一性一并处理 —— 模型自己的 reset_parameters() 会用正确 shape 重新生成。
_RECOMPUTABLE_BUFFER_PATTERNS = (
    "pos_embedder.seq",
    "pos_embedder.dim_spatial_range",
    "pos_embedder.dim_temporal_range",
    "extra_pos_embedder.seq",
    "extra_pos_embedder.dim_spatial_range",
    "extra_pos_embedder.dim_temporal_range",
)


# RoPE 派生 buffer 所属模块的属性名。fast_init 路径下这些模块的 buffer 是从 arange
# 重算的（不是训练学到的），必须在 to_empty() 之后显式 reset_parameters()。
_POS_EMBEDDER_MODULE_SUFFIXES = ("pos_embedder", "extra_pos_embedder")


def _build_anima_on_meta(Anima, config):
    """在 meta device 上构造 Anima，再 to_empty 到 CPU。

    为什么：2.09B 参数的默认随机初始化（kaiming_uniform_ 等）是纯 CPU 串行开销，
    本地 16 线程实测 26.5s，算力受限的节点上实测 301.5s —— 而这些随机值随即会被
    checkpoint 权重整个覆盖，一个字节都用不上。meta 构造把这段降到接近 0。

    代价与防线：``to_empty()`` 分配的是**未初始化内存**，任何没被 checkpoint 填过的
    参数/buffer 都会是垃圾值（且不会报错，只会让训练悄悄跑歪）。所以本函数返回它显式
    重算过的 key 集合，由 ``_assert_no_uninitialized`` 与加载结果对账，对不上就 raise。

    返回 ``(model, recomputed_keys)``。
    """
    with torch.device("meta"):
        model = Anima(**config)
    model.to_empty(device="cpu")

    # 需要显式重算的两类模块：
    #   1) RoPE 位置嵌入 —— 它的 buffer 在 ckpt 里有，但 shape 跟着 max_img_* 变，
    #      加载时按 recomputable 丢弃（见 _RECOMPUTABLE_BUFFER_PATTERNS）。
    #   2) 任何持有 **non-persistent buffer** 的模块 —— 这类 buffer 不进 state_dict，
    #      checkpoint 永远填不到（如 llm_adapter.rotary_emb.inv_freq）。
    persistent = set(model.state_dict().keys())
    recomputed: set[str] = set()
    for mname, m in model.named_modules():
        own_buffers = [f"{mname}.{b}" if mname else b for b, _ in m.named_buffers(recurse=False)]
        needs = (mname.split(".")[-1].endswith(_POS_EMBEDDER_MODULE_SUFFIXES)
                 or any(k not in persistent for k in own_buffers))
        if not needs or not hasattr(m, "reset_parameters"):
            continue
        m.reset_parameters()
        recomputed.update(own_buffers)
    return model, recomputed


def _assert_no_uninitialized(model, load_info: dict, recomputed: set) -> None:
    """fail-fast：确认 fast_init 之后模型里没有残留的未初始化内存。

    未被 checkpoint 填过的 key = ``missing``（ckpt 里根本没有）∪ ``skipped``（ckpt 里有
    但 shape 不匹配、被当作 recomputable 丢掉）。这些必须全部落在 ``recomputed`` 里。
    另外 state_dict 不含 non-persistent buffer，单独再查一遍。
    """
    unfilled = set(load_info.get("missing", ())) | set(load_info.get("skipped", ()))
    persistent = set(model.state_dict().keys())
    non_persistent = {n for n, _ in model.named_buffers()} - persistent
    leftover = sorted((unfilled | non_persistent) - set(recomputed))
    if leftover:
        raise RuntimeError(
            f"fast_init（meta 构造）后有 {len(leftover)} 个参数/buffer 既没被 checkpoint "
            f"填充、也没被显式重算，内容是未初始化内存：{leftover[:10]}"
            f"{' ...' if len(leftover) > 10 else ''}。"
            f"请设 fast_init=false（回到 CPU 随机初始化路径）并反馈这个列表 —— "
            f"这说明模型结构里新增了 fast_init 不认识的、需要真初始化的张量。"
        )
    logger.info(
        "[fast_init] 校验通过：%d 个 key 由 checkpoint 填充，%d 个 RoPE 派生 buffer 已重算，"
        "无未初始化残留。", len(model.state_dict()) - len(unfilled), len(recomputed),
    )


def load_anima_model(transformer_path, device, dtype, repo_root,
                     max_img_h: int = 240, max_img_w: int = 240, max_frames: int = 128,
                     fast_init: bool = False):
    """加载 Anima transformer 模型。

    `max_img_h` / `max_img_w` 控制 RoPE position embedding 的 `seq` buffer 长度上限
    （单位：latent 位置；模型内部会再除以 patch_spatial=2 转成 patch 位置）。

    默认 240 对应 120 patches = 1920 image pixels 最大单维；
    1536² + AR=2.0 训练需要单维 2172 image = 272 latent → 至少 max_img_h=272；
    建议传入 288 / 320 留出余量。

    Anima 当前没有启用 `extra_per_block_abs_pos_emb`（learnable pos embedding 是关掉的），
    所以加大这两个值**只会改变 RoPE 的 arange buffer 长度**，不会破坏权重加载兼容性。
    seq buffer 是 derivative tensor（recomputable from arange），shape 不匹配时
    `_load_safetensors_into_model(skip_buffer_patterns=...)` 跳过，模型仍按构造时的正确 shape 工作。
    """
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

    # 保证 max_img_h/w 是 patch_spatial(=2) 的倍数，避免 len_h 取整丢精度
    max_img_h = int(max_img_h)
    max_img_w = int(max_img_w)
    if max_img_h % 2:
        max_img_h += 1
    if max_img_w % 2:
        max_img_w += 1

    logger.info(
        "Anima 模型构造: max_img_h=%d, max_img_w=%d, max_frames=%d "
        "(支持单维最大 %d image pixels = %d patches)",
        max_img_h, max_img_w, max_frames,
        max_img_h * 8, max_img_h // 2,
    )

    config = dict(
        max_img_h=max_img_h, max_img_w=max_img_w, max_frames=max_frames,
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

    # 构造与权重加载都是纯 CPU 且都以分钟计（2.09B 参数）。分开计时：算力受限的节点上
    # 这两段能占掉启动的绝大部分，不切开就只能看到一个几分钟的"卡住"。
    _t0 = time.perf_counter()
    if fast_init:
        model, _recomputed = _build_anima_on_meta(Anima, config)
    else:
        model, _recomputed = Anima(**config), None
    _t_build = time.perf_counter() - _t0

    _t0 = time.perf_counter()
    _info = _load_safetensors_into_model(
        model, Path(transformer_path), label="Transformer",
        skip_buffer_patterns=_RECOMPUTABLE_BUFFER_PATTERNS,
    )
    _t_load = time.perf_counter() - _t0
    logger.info(
        "Transformer 启动耗时: 构造%s %.1fs + 权重加载 %.1fs (torch 线程数=%d)",
        "(meta, 跳过随机初始化)" if fast_init else "(CPU 随机初始化)",
        _t_build, _t_load, torch.get_num_threads(),
    )
    if fast_init:
        _assert_no_uninitialized(model, _info, _recomputed)

    # 如果 checkpoint 中完全没有 llm_adapter 权重，随机初始化会把 cross-attn 条件搞乱，直接禁用更安全
    with safe_open(transformer_path, framework="pt", device="cpu") as f:
        has_llm_adapter = any("llm_adapter" in k for k in f.keys())
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


def load_vae(vae_path, device, dtype, repo_root, attn_chunk_tokens: int = 0):
    """加载 VAE

    `attn_chunk_tokens` > 0 时开启 VAE 自注意力的 query 分块（数学恒等，把 math-SDPA
    后端下 O(N²) 的峰值显存降到 O(chunk·N)）。0 = 关闭，走原来的整块 SDPA。
    """
    wan_vae = load_module_from_path("wan_vae", repo_root / "wan" / "vae2_1.py")
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


_QWEN_LEGACY_SUBDIR = "Qwen3-0.6B-Base"


def _resolve_qwen_dir(qwen_path):
    """兼容旧布局的路径解析。

    Qwen3-0.6B 的权重/tokenizer 原本平铺在 `models/text_encoders/` 根目录，与 Krea2 的
    `text_encoders/Qwen3-VL-4B-Instruct/` 层级不对称，已整理进 `text_encoders/Qwen3-0.6B-Base/`。
    但训练 yaml 不随代码一起推送，云端可能仍写着旧的根目录路径 —— 这里做一次显式兜底：
    只有当目标目录**没有** config.json、而其下 `Qwen3-0.6B-Base/` 有时才改写，并打 warning。
    路径写对时本函数是恒等的，不引入任何行为变化。
    """
    p = Path(qwen_path)
    if (p / "config.json").exists():
        return qwen_path
    legacy = p / _QWEN_LEGACY_SUBDIR
    if (legacy / "config.json").exists():
        logger.warning(
            "text_encoder_path=%r 下没有 config.json，已自动改用 %r（Qwen3-0.6B 已从 "
            "text_encoders/ 根目录整理进子目录）。建议更新 yaml 里的 text_encoder_path。",
            str(p), str(legacy),
        )
        return str(legacy)
    return qwen_path


def load_text_encoders(qwen_path, t5_tokenizer_path, device, dtype):
    """加载文本编码器。

    ★ 加 `low_cpu_mem_usage=True` 与 `device_map={"": device}` 避免双倍 RAM：
       旧实现先把权重 load 到 CPU 再 `.to(device)` 拷一份到 VRAM；Qwen3-0.6B ~1.2GB
       会让 CPU RAM 额外占用 1.2GB（在 32GB 服务器上不算大问题，但云上小机型 / 多 LoRA
       并行训练时会撞 OOM）。新写法直接 in-place 加载到目标 device。
       老版本 transformers 不支持 `device_map` 参数的情形：catch + fallback 旧路径。
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer

    qwen_path = _resolve_qwen_dir(qwen_path)
    qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)

    try:
        qwen_model = AutoModelForCausalLM.from_pretrained(
            qwen_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map={"": str(device)},
        ).eval().requires_grad_(False)
    except (TypeError, ValueError, ImportError) as e:
        logger.warning(f"Qwen 加载 device_map 路径失败 ({e})，回退到 .to(device)（CPU RAM 会临时翻倍）")
        qwen_model = AutoModelForCausalLM.from_pretrained(
            qwen_path, torch_dtype=dtype, trust_remote_code=True
        ).to(device).eval().requires_grad_(False)

    if t5_tokenizer_path and Path(t5_tokenizer_path).exists():
        t5_tokenizer = T5Tokenizer.from_pretrained(t5_tokenizer_path)
    else:
        t5_tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")

    logger.info("文本编码器加载完成")
    return qwen_model, qwen_tokenizer, t5_tokenizer
