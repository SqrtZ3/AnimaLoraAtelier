"""krea2 文本侧编码（Qwen3-VL 文本塔）—— 从 upstream `trainer/model_family.py`
摘录，逐字。

## 这是**回退**路径，默认别用

krea2 的 textfeat 是 12 层 hidden 堆叠 `[L, 12, 2560]`，bf16 约 61.4KB/token ——
96 张图就 **1.7GB**，而这 1.7GB 的信息源只有 96 条 caption（0.14MB）。

推荐路径是让 TPU **现算**：
    tools/dump_caption_ids.py         # 本地只 tokenize，产物约 40KB
    build_job.py --caption-ids ...    # 内嵌进 job，真机上跑 jax_tpu/text_cache.py
这样上传量从 1.7GB 降到 40KB（压缩比 ~4.8 万倍），caption 明文也不出本地。

本文件留着是为了三种情况：① 不想让 TE 权重进 Kaggle；② 想在本地核对真机
现算的结果（`jax_tpu/tests/check_qwen3vl_real.py` 就拿它的产物当参考）；
③ TPU 侧现算路径出问题时的退路。

## 四个常量是"抄错就静默全错"的那类

`_KREA2_PREFIX_IDX = 34` / `_KREA2_SUFFIX_START_IDX = 5` 是 chat 模板的**切片
位置**，`KREA2_SELECT_LAYERS` 是取哪 12 层。抄错不会报错，只会让条件全歪。
`tools/dump_caption_ids.py` 里那两道 tokenizer 闸门（prefix 单独 tokenize 必须
恰好 34 个 token）就是拦这个的 —— 别删。

行号与 sha256 见 `UPSTREAM.md`；漂移由 `tools/check_sync.py` 守。
"""

from __future__ import annotations

import logging
from collections import OrderedDict

import torch

logger = logging.getLogger(__name__)

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

    max_length>0：官方定长 pad 口径（默认 512）；max_length<=0：无上限（opt-in），不截断、
    动态 pad 到 batch 内最长。两种 padding 量在 causal LM + mask 下对有效 token 的输出等价，
    batch≡逐条仍成立（padding 量不同不影响压缩后结果）。
    """
    te, tokenizer = handles["model"], handles["tokenizer"]
    fulls = [_KREA2_PROMPT_PREFIX + (t if t else " ") for t in texts]
    if max_length and int(max_length) > 0:
        # 官方口径（default）：截断到 max_length 并 pad 到定长（默认 512，逐字对齐官方
        # encoder.py：truncation=True、max_length + prefix_idx - suffix_start_idx）。
        inputs = tokenizer(
            fulls, truncation=True, return_overflowing_tokens=False,
            padding="max_length",
            max_length=int(max_length) + _KREA2_PREFIX_IDX - _KREA2_SUFFIX_START_IDX,
            return_tensors="pt",
        ).to(te.device)
    else:
        # 无上限（opt-in，krea2_text_max_length<=0）：不截断、动态 pad 到 batch 内最长。
        # 与定长 pad 数学等价——Qwen3-VL 是因果 LM，右侧 padding 被 causal + attention_mask
        # 双重屏蔽，不改变任何有效 token 的 hidden state；下游按 mask 压缩，输出与逐条/定长
        # pad 完全一致，仅去掉截断这一步（超长 caption 的尾部标签不再被右截断丢弃）。
        inputs = tokenizer(
            fulls, truncation=False, return_overflowing_tokens=False,
            padding=True,
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

    max_length>0 截断到该预算（默认 512=官方 encoder.py 值）；max_length<=0 无上限
    （不截断，超长 caption 尾部保留）——见 _encode_krea2_batch。

    - 权重语法 `(tag:1.5)` 被剥离（Krea2 无 T5 token 权重通道；clean text 与 Anima
      的 Qwen 通道同处理）。
    - 每条先压缩到有效 token，再 pad 到 batch 内最长 + mask —— dense 前向用 mask，
      navit 打包直接用压缩长度（见 encode_krea2_text_packed）。
    - cache 未命中的条目合并成**一次** TE batch 前向（数学等价于逐条，见
      _encode_krea2_batch）；命中的直接取 cache。
    """
    # upstream 此处是 `from trainer.text_encode import ...`；本仓库里同一个函数
    # 在 `_vendor/t5_weighted.py`（逐字摘录），改成相对 import。
    from .t5_weighted import _build_qwen_text_from_prompt

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
