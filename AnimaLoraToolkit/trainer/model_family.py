"""DiT 模型族（model family）通用接口。

设计目标：让 trainer 主体（objective/lora/checkpoint/optimizer/telemetry）对底模架构
保持无感，扩展新 DiT 底模时只需：
  1. 模型类实现"模型级鸭子契约"（与 Anima 模型对齐的方法签名）：
       forward(x_B_C_T_H_W, timesteps_B_T, crossattn_emb, fps=None, padding_mask=None)
       patchify_latents_to_tokens(x, padding_mask=None) → (tokens, grid, mask, size)
       unpatchify_tokens(tokens, size) → 5D latent
       forward_packed_navit(tokens, t_G, cross_packed, grid, vseq, text_seqlens,
                            use_checkpoint=False) → [1, ΣN, M]
       _output_tokens_to_patch_tokens(tokens, size)
     —— 满足契约后 objective.py 的 navit 打包 loss / eval / sample_latent 全部直接复用。
  2. 在本文件注册一个 family 对象，提供架构差异化的部分：模型/文本编码器加载、
     文本 → cross 载荷编码、LoRA 默认 targets、timestep shift、采样默认值。

当前支持：
  - "anima"（默认）：Cosmos MiniTrainDIT + Qwen3-0.6B/T5 cross-attn。全部走既有代码
    路径，行为与引入本文件前逐一等价。
  - "krea2"：Krea 2 单流 MMDiT（models/krea2_modeling.py）+ Qwen3-VL-4B 12 层特征。
    opt-in（model_family: krea2），cross 载荷为 [B, L, 12, 2560] 4D 堆叠。

Krea2 与 Anima 的 cross 载荷维度不同（4D vs 3D），模型侧有 fail-fast 校验，
配错 family 会立刻报清晰错误而不是静默出垃圾。
"""

from __future__ import annotations

import logging
import math
from collections import OrderedDict
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

VALID_FAMILIES = ("anima", "krea2")


def get_model_family(args) -> str:
    fam = str(getattr(args, "model_family", "anima") or "anima").lower()
    if fam not in VALID_FAMILIES:
        raise ValueError(f"未知 model_family={fam!r}；可选: {VALID_FAMILIES}")
    return fam


# ---------------------------------------------------------------------------
# Krea2：兼容性守卫（fail-fast，不静默降级）
# ---------------------------------------------------------------------------
# 这些特性当前实现读取 Anima 模型内部结构（block 布局 / cross-attn / t_embedder /
# llm_adapter），或依赖 Anima 专属前向路径。对 krea2 显式报错，待需要时逐个移植。
_KREA2_UNSUPPORTED = (
    ("fit_packed_training", "FiT padded-token 打包（Anima forward_packed_tokens 专属；krea2 请用 navit_packing）"),
    ("token_bucket", "token bucket（依赖 fit_packed_training）"),
    ("torch_compile", "torch.compile 快路径（依赖 fit_packed_training）"),
    ("dpo_enabled", "DPO（loser 采样/参考前向未对 krea2 验证）"),
    ("leap_enabled", "LeapAlign（两步跳跃前向绑定 Anima forward 助手）"),
    ("gaf_enabled", "GAF（hook 挂在 Anima block 输入上）"),
    ("ncp_enabled", "NCP（tap Anima block 中间特征）"),
    ("self_perceptual_enabled", "self-perceptual loss（tap 模型内部特征）"),
    ("dispersive_enabled", "Dispersive loss（tap 模型中间隐藏表征）"),
    ("tread_enabled", "TREAD token 路由（绑定 Anima forward_tokens）"),
    ("lora_one_init_steps", "LoRA-One 谱对齐初始化（编码/前向路径绑定 Anima）"),
)


def validate_family_compat(args, family: str) -> None:
    """krea2 下对未验证/未移植的特性 fail-fast。anima 恒 no-op。"""
    if family != "krea2":
        return
    bad = []
    for key, label in _KREA2_UNSUPPORTED:
        val = getattr(args, key, None)
        if val is None:
            continue
        if isinstance(val, bool):
            on = val
        else:
            try:
                on = float(val) > 0
            except (TypeError, ValueError):
                on = bool(val)
        if on:
            bad.append(f"  - {key}: {label}")
    if bad:
        raise RuntimeError(
            "model_family=krea2 暂不支持以下已启用的特性（Anima 专属实现，需单独移植）：\n"
            + "\n".join(bad)
            + "\n请在 YAML 中关闭后重试。"
        )
    if not bool(getattr(args, "cache_latents", False)):
        logger.warning(
            "[krea2] 建议开启 cache_latents（12B 模型步时长，现场 VAE encode 会进一步拖慢）"
        )


# ---------------------------------------------------------------------------
# Krea2：模型 / 文本编码器加载
# ---------------------------------------------------------------------------
def load_krea2_model(transformer_path, device, dtype, repo_root,
                     max_img_h: int = 0, max_img_w: int = 0):
    """加载 Krea 2 SingleStreamDiT（raw.safetensors / turbo.safetensors，key 与官方一致）。

    max_img_h/w 仅作日志参考 —— Krea2 的 RoPE 按需从 pos 现算，无容量 buffer 上限
    （与 Anima 的 pos_embedder.seq 预分配不同）。
    """
    from trainer.checkpoint import _get_safetensors_shapes, _load_safetensors_into_model
    from trainer.models import ensure_models_namespace, load_module_from_path

    ensure_models_namespace(repo_root)
    krea2_modeling = load_module_from_path(
        "krea2_modeling", Path(repo_root) / "krea2_modeling.py"
    )

    # 仅读 key + shape（不加载 tensor 数据），用于配置推断和前缀检测
    shapes = _get_safetensors_shapes(Path(transformer_path))
    # 剥离可能的 "model." / "diffusion_model." 前缀（ComfyUI 再打包等场景）
    prefixes = None
    for prefix in ("model.diffusion_model.", "diffusion_model.", "model."):
        if any(k.startswith(prefix) for k in shapes) and f"{prefix}first.weight" in shapes:
            shapes = {k[len(prefix):]: v for k, v in shapes.items() if k.startswith(prefix)}
            prefixes = [prefix]
            break
    if "first.weight" not in shapes:
        raise RuntimeError(
            f"{transformer_path} 里找不到 Krea2 权重（缺 first.weight）。"
            "请确认是官方 raw.safetensors（训练用 RAW，不要用 diffusers 分片目录）。"
        )

    config = krea2_modeling.infer_config_from_state_dict(shapes)
    model = krea2_modeling.SingleStreamDiT(config)

    # 流式加载：逐 tensor 拷入模型参数，避免在 RAM 中持有完整 state dict
    result = _load_safetensors_into_model(
        model, Path(transformer_path),
        prefixes=prefixes, label="Krea2 Transformer",
        strict_missing=True,
    )
    if result["unexpected"]:
        logger.warning("Krea2 checkpoint 多出 %d 个未用 key（前 5 个: %s）",
                       len(result["unexpected"]), result["unexpected"][:5])
    model = model.to(device=device, dtype=dtype)
    model.requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Krea2 模型加载完成: features=%d, %d blocks, GQA %d/%d, %.2fB params",
        config.features, config.layers, config.heads, config.kvheads or config.heads,
        n_params / 1e9,
    )
    return model


# 官方 encoder.py 的模板与常量（krea-ai/krea-2）。
_KREA2_PROMPT_PREFIX = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and background:"
    "<|im_end|>\n<|im_start|>user\n"
)
_KREA2_PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
_KREA2_PREFIX_IDX = 34
_KREA2_SUFFIX_START_IDX = 5
KREA2_SELECT_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)


def load_krea2_text_encoder(text_encoder_path, device, dtype):
    """加载 Qwen3-VL-4B-Instruct（HF 目录）。返回 dict handles。

    需要较新的 transformers（含 Qwen3VLForConditionalGeneration）；不满足时报清晰错误。
    """
    try:
        from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "当前 transformers 版本没有 Qwen3VLForConditionalGeneration（Krea2 文本编码"
            "需要）。请升级 transformers（>=4.57）。"
        ) from exc

    if not text_encoder_path or not Path(text_encoder_path).exists():
        raise FileNotFoundError(
            f"krea2_text_encoder_path 无效: {text_encoder_path!r}（应为 Qwen3-VL-4B-Instruct "
            "的 HuggingFace 目录）"
        )

    tokenizer = AutoTokenizer.from_pretrained(text_encoder_path)
    try:
        te = Qwen3VLForConditionalGeneration.from_pretrained(
            text_encoder_path, torch_dtype=dtype,
            low_cpu_mem_usage=True, device_map={"": str(device)},
        ).eval().requires_grad_(False)
    except (TypeError, ValueError, ImportError) as e:
        logger.warning("Qwen3-VL device_map 加载失败 (%s)，回退 .to(device)", e)
        te = Qwen3VLForConditionalGeneration.from_pretrained(
            text_encoder_path, torch_dtype=dtype,
        ).to(device).eval().requires_grad_(False)
    logger.info("Krea2 文本编码器 (Qwen3-VL) 加载完成: %s", text_encoder_path)
    return {"model": te, "tokenizer": tokenizer}


# ── Krea2 文本编码 LRU cache ────────────────────────────────────────────────
# 条目是"压缩后仅有效 token"的 [L_valid, 12, D] 特征（bf16 下典型 tag caption ~几 MB），
# 比 Anima 的 cache 大得多 → 容量默认小得多（cap 由调用方传入）。
_KREA2_TEXT_CACHE: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()
_KREA2_TEXT_CACHE_CAP = 128
_KREA2_TEXT_CACHE_ENABLED = True


def set_krea2_text_cache(enabled: bool, cap: int = 128):
    global _KREA2_TEXT_CACHE_ENABLED, _KREA2_TEXT_CACHE_CAP
    _KREA2_TEXT_CACHE_ENABLED = bool(enabled)
    _KREA2_TEXT_CACHE_CAP = max(int(cap), 1)
    if not enabled:
        _KREA2_TEXT_CACHE.clear()


def reset_krea2_text_cache():
    _KREA2_TEXT_CACHE.clear()


@torch.no_grad()
def _encode_krea2_batch(handles: dict, texts: list, device, max_length: int) -> list:
    """一次 TE 前向编码 N 条 prompt → list of [L_valid_i, 12, D]（官方 encoder.py 逐步对齐）。

    与逐条编码数学等价：所有条目本就 pad 到同一 max_length、拼同一 suffix，
    batch 内每行的 input_ids / attention_mask / position 布局与单条时完全相同，
    仅把 N 次串行 TE 前向合并成 1 次（caption 动态时 cache 全关、训练每步都走
    这条路径 —— 省 N−1 次 4B 模型的整趟启动）。

    压缩（去掉 attention_mask=0 的 padding 位）与官方"mask 屏蔽 padding"数学等价：
    text token 无 RoPE（pos 全 0）、padding 作为 key 被 mask 后贡献恒为 0，
    删除它们不改变任何有效 token 的注意力输出。
    """
    te, tokenizer = handles["model"], handles["tokenizer"]
    fulls = [_KREA2_PROMPT_PREFIX + (t if t else " ") for t in texts]
    inputs = tokenizer(
        fulls, truncation=True, return_overflowing_tokens=False,
        padding="max_length",
        max_length=max_length + _KREA2_PREFIX_IDX - _KREA2_SUFFIX_START_IDX,
        return_tensors="pt",
    ).to(te.device)
    suffix = tokenizer([_KREA2_PROMPT_SUFFIX], return_tensors="pt",
                       add_special_tokens=False).to(te.device)
    B = len(fulls)
    input_ids = torch.cat(
        [inputs["input_ids"], suffix["input_ids"].expand(B, -1)], dim=1
    )
    mask = torch.cat(
        [inputs["attention_mask"].bool(), suffix["attention_mask"].bool().expand(B, -1)],
        dim=1,
    )
    states = te(input_ids=input_ids, attention_mask=mask, output_hidden_states=True)
    hiddens = torch.stack([states.hidden_states[i] for i in KREA2_SELECT_LAYERS], dim=2)
    hiddens = hiddens[:, _KREA2_PREFIX_IDX:]          # [B, L, 12, D]
    mask = mask[:, _KREA2_PREFIX_IDX:]                # [B, L]

    out = []
    for i in range(B):
        valid = hiddens[i][mask[i]]                   # [L_valid, 12, D]（压缩）
        if valid.shape[0] == 0:
            valid = hiddens[i][:1]                    # 兜底：至少 1 个 token
        out.append(valid.detach())
    return out


def _encode_krea2_single(handles: dict, text: str, device, max_length: int) -> torch.Tensor:
    """单条 prompt → [L_valid, 12, D]。批量路径的 B=1 包装（含 cache 查询/写入）。"""
    key = (text, int(max_length), id(handles["model"]))
    if _KREA2_TEXT_CACHE_ENABLED:
        hit = _KREA2_TEXT_CACHE.get(key)
        if hit is not None:
            _KREA2_TEXT_CACHE.move_to_end(key)
            return hit.to(device)
    valid = _encode_krea2_batch(handles, [text], device, max_length)[0]
    if _KREA2_TEXT_CACHE_ENABLED:
        _KREA2_TEXT_CACHE[key] = valid
        _KREA2_TEXT_CACHE.move_to_end(key)
        while len(_KREA2_TEXT_CACHE) > _KREA2_TEXT_CACHE_CAP:
            _KREA2_TEXT_CACHE.popitem(last=False)
    return valid.to(device)


def encode_krea2_text(handles: dict, texts, device, max_length: int = 512):
    """编码一批 prompt。返回 (cross [B, L_max, 12, D], cross_mask [B, L_max] bool)。

    - 权重语法 `(tag:1.5)` 被剥离（Krea2 无 T5 token 权重通道；clean text 与 Anima
      的 Qwen 通道同处理）。
    - 每条先压缩到有效 token，再 pad 到 batch 内最长 + mask —— dense 前向用 mask，
      navit 打包直接用压缩长度（见 encode_krea2_text_packed）。
    - cache 未命中的条目合并成**一次** TE batch 前向（数学等价于逐条，见
      _encode_krea2_batch）；命中的直接取 cache。
    """
    from trainer.text_encode import _build_qwen_text_from_prompt

    if isinstance(texts, str):
        texts = [texts]
    cleaned = [_build_qwen_text_from_prompt(str(t or "")) for t in texts]

    feats: list = [None] * len(cleaned)
    missing = []
    if _KREA2_TEXT_CACHE_ENABLED:
        for i, txt in enumerate(cleaned):
            key = (txt, int(max_length), id(handles["model"]))
            hit = _KREA2_TEXT_CACHE.get(key)
            if hit is not None:
                _KREA2_TEXT_CACHE.move_to_end(key)
                feats[i] = hit.to(device)
            else:
                missing.append(i)
    else:
        missing = list(range(len(cleaned)))

    if missing:
        # 批内去重：navit_multiscale 的缩放副本与原图共 caption（cache 关闭时
        # 同一 pack 里是完全相同的字符串），只编码一次、多处引用（下游 cat/pad
        # 都是拷贝语义，共享安全）。
        uniq_texts = list(dict.fromkeys(cleaned[i] for i in missing))
        encoded_map = {}
        # 分块编码：output_hidden_states 会让 37 层 hidden states 同时驻留
        # [B, 541, D]×37 ≈ 102MB/条 —— 整包一次前向在 G≈12 时瞬时 +1.2GB。
        # 按 8 条一块编码把该峰值封顶在 ~0.8GB，数学与整批/逐条一致。
        _CHUNK = 8
        for s in range(0, len(uniq_texts), _CHUNK):
            chunk = uniq_texts[s:s + _CHUNK]
            for txt, f in zip(chunk, _encode_krea2_batch(handles, chunk, device, max_length)):
                encoded_map[txt] = f
        for i in missing:
            f = encoded_map[cleaned[i]]
            if _KREA2_TEXT_CACHE_ENABLED:
                key = (cleaned[i], int(max_length), id(handles["model"]))
                _KREA2_TEXT_CACHE[key] = f
                _KREA2_TEXT_CACHE.move_to_end(key)
            feats[i] = f.to(device)
        if _KREA2_TEXT_CACHE_ENABLED:
            while len(_KREA2_TEXT_CACHE) > _KREA2_TEXT_CACHE_CAP:
                _KREA2_TEXT_CACHE.popitem(last=False)

    L_max = max(f.shape[0] for f in feats)
    B = len(feats)
    n_layers, dim = feats[0].shape[1], feats[0].shape[2]
    cross = torch.zeros(B, L_max, n_layers, dim, device=device, dtype=feats[0].dtype)
    cmask = torch.zeros(B, L_max, dtype=torch.bool, device=device)
    for i, f in enumerate(feats):
        cross[i, : f.shape[0]] = f
        cmask[i, : f.shape[0]] = True
    return cross, cmask


def krea2_pack_text(cross: torch.Tensor, cross_mask: torch.Tensor):
    """dense (cross, mask) → navit 打包载荷 (cross_packed [1, ΣL, 12, D], text_seqlens)。

    压缩每样本的有效 token（本编码通道有效位天然连续在前，但按 mask gather 写法
    对任意 mask 布局都正确）。
    """
    G = cross.shape[0]
    parts, lens = [], []
    for i in range(G):
        f = cross[i][cross_mask[i]]
        if f.shape[0] == 0:
            f = cross[i][:1]
        parts.append(f)
        lens.append(int(f.shape[0]))
    return torch.cat(parts, dim=0).unsqueeze(0), lens


# ---------------------------------------------------------------------------
# Krea2：分辨率感知 timestep shift（官方 sampling.py 的训练侧等价）
# ---------------------------------------------------------------------------
def krea2_mu(image_tokens: float, min_res: int = 256, max_res: int = 1280,
             y1: float = 0.5, y2: float = 1.15, patch_px: int = 16) -> float:
    """image_tokens = (H/16)·(W/16)。mu 在 (x1,y1)-(x2,y2) 间线性内插（官方公式）。

    1024² → mu≈0.906，exp(mu)≈2.48（musubi 文档口径的 "shift 2.5 @1024"）。
    """
    x1 = (min_res // patch_px) ** 2
    x2 = (max_res // patch_px) ** 2
    slope = (y2 - y1) / (x2 - x1)
    return slope * float(image_tokens) + (y1 - slope * x1)


def krea2_shift_timesteps(t: torch.Tensor, image_tokens, min_res: int = 256,
                          max_res: int = 1280, y1: float = 0.5, y2: float = 1.15) -> torch.Tensor:
    """把已采样的 t ∈ (0,1) 做 Krea2 分辨率感知 shift：t' = αt / (1 + (α−1)t)，α=exp(mu)。

    与官方 timesteps() 的 exp(mu)/(exp(mu)+(1/t−1)) 代数同形；α>1 时把质量推向高噪端。
    `image_tokens` 可为标量（整 batch 同尺寸）或 len==B 的序列（navit 逐图）。
    作用在 base 分布采样之后 —— 与 Anima 的 schedule_shift 同一挂点，任何 timestep
    mode 之上都可组合。
    """
    if isinstance(image_tokens, (int, float)):
        alpha = math.exp(krea2_mu(image_tokens, min_res, max_res, y1, y2))
        return alpha * t / (1 + (alpha - 1) * t)
    toks = torch.tensor([float(x) for x in image_tokens], device=t.device, dtype=t.dtype)
    if toks.numel() != t.numel():
        raise ValueError(f"image_tokens 数 ({toks.numel()}) != t 数 ({t.numel()})")
    x1 = (min_res // 16) ** 2
    x2 = (max_res // 16) ** 2
    slope = (y2 - y1) / (x2 - x1)
    alpha = torch.exp(slope * toks + (y1 - slope * x1)).reshape(t.shape)
    return alpha * t / (1 + (alpha - 1) * t)


def krea2_sample_shift(height: int, width: int, min_res: int = 256, max_res: int = 1280,
                       y1: float = 0.5, y2: float = 1.15) -> float:
    """采样（推理）用的常数 shift = exp(mu(分辨率))，供 _flow_sigmas_* 复用
    （其 _time_snr_shift 与官方 shift 同形）。"""
    tokens = (height // 16) * (width // 16)
    return math.exp(krea2_mu(tokens, min_res, max_res, y1, y2))


# ---------------------------------------------------------------------------
# LoRA 默认 targets
# ---------------------------------------------------------------------------
# 官方推荐默认 = DiT 内全部 264 个 Linear（musubi krea2 文档；rank/alpha 32）：
# 28 block × (attn 5 + mlp 3) + txtfusion(4 block × 8 + projector) + txtmlp 2 +
# tmlp 2 + tproj 1 + first 1 + last.linear 1。
# 注入器按"子串命中 + nn.Linear"匹配，以下列表恰好覆盖全部 Linear 且不误伤。
#
# 关于 tproj.1（= nn.Linear(6144, 36864)，226M 参数，全网最大单层）：官方/musubi
# 默认就训它（modulation/RMSNorm 是裸张量、非 Linear，本就不被包）。训练侧无问题——
# 训练时采样预览正常即证。真正的坑在 **ComfyUI 部署侧**：标准 Load LoRA 节点会物化
# 完整 delta (36864×6144 ≈ 453MB bf16)，12B 占满显存时 "Allocation on device"（OOM），
# 该层被静默丢弃 → 全局调制错位 → 背景崩。解法不是不训它，而是推理改用 ComfyUI 内置
# **Load LoRA (Bypass, Model Only)**（前向低秩注入、不物化 delta）。详见 docs/krea2-family.md。
KREA2_DEFAULT_LORA_TARGETS = [
    "attn.wq", "attn.wk", "attn.wv", "attn.wo", "attn.gate",
    "mlp.gate", "mlp.up", "mlp.down",
    "txtfusion.projector",
    "txtmlp.1", "txtmlp.3",
    "tmlp.0", "tmlp.2",
    "tproj.1",
    "first",
    "last.linear",
]

# 官方"长训练"配置：只训 28 个主 block 的 attention 投影（140 Linear），
# 保 prompt 追随性。（txtfusion 的 attn 通过 exclude 排除。）
KREA2_ATTENTION_ONLY_TARGETS = ["attn.wq", "attn.wk", "attn.wv", "attn.wo", "attn.gate"]
KREA2_ATTENTION_ONLY_EXCLUDE = [r".*txtfusion.*"]


def family_default_lora_targets(family: str):
    if family == "krea2":
        return list(KREA2_DEFAULT_LORA_TARGETS)
    return None  # anima：沿用 LoRAInjector.DEFAULT_TARGETS
