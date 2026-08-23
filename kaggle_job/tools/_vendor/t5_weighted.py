"""anima 文本侧编码 —— 从 upstream `trainer/text_encode.py` 摘录，逐字。

产出 `<stem>.textfeat.npz` 的 `cross` 需要三步：
  ① `_build_qwen_text_from_prompt` 把带权重语法的 tag 串洗成"干净文本"给 Qwen；
  ② `encode_qwen` 拿 Qwen3-0.6B 的 last hidden（padding 位清零）；
  ③ `tokenize_t5_weighted` 逐 tag 分词并算 **token 级权重**（`(xxx:1.2)` / `[xxx]`
     那套 A1111 语法），权重最后乘到 cross 上。
再喂给 `_vendor/llm_adapter.py` 的 adapter。

**LRU cache 一并照抄了**（`_cache_get`/`_cache_put`/`_TEXT_CACHE_*`）。它在离线
缓存场景下用不上（每条 caption 只编一次），但它的判断嵌在 `encode_qwen` /
`tokenize_t5_weighted` 的函数体里 —— 摘掉就不再是"逐字摘录"，也就丧失了
`check_sync.py` 与逐 bit 对拍能提供的保证。多 60 行换一个可机械校验的口径，值。

行号与 sha256 见 `UPSTREAM.md`；漂移由 `tools/check_sync.py` 守。
"""

from __future__ import annotations

import logging
import re
from collections import OrderedDict
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

#: upstream 在模块级建这两个 LRU；照搬（`encode_qwen` / `tokenize_t5_weighted`
#: 的函数体直接引用它们）。
_QWEN_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_T5_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()

_TEXT_CACHE_CAP = 2048

_TEXT_CACHE_ENABLED = True  # 可由训练脚本 toggle

def set_text_encode_cache_enabled(enabled: bool):
    """Toggle text-encode cache（caption shuffle/dropout 启用时建议关，因为命中率会很低）。"""
    global _TEXT_CACHE_ENABLED
    _TEXT_CACHE_ENABLED = bool(enabled)

def reset_text_encode_cache():
    """清空 cache。模型 reload 后调用以避免读到 stale embedding。"""
    _QWEN_CACHE.clear()
    _T5_CACHE.clear()

def _cache_get(cache: "OrderedDict", key):
    if not _TEXT_CACHE_ENABLED:
        return None
    v = cache.get(key)
    if v is not None:
        cache.move_to_end(key)
    return v

def _cache_put(cache: "OrderedDict", key, value):
    if not _TEXT_CACHE_ENABLED:
        return
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _TEXT_CACHE_CAP:
        cache.popitem(last=False)

def _parse_weighted_tag(tag: str) -> tuple[str, float]:
    """
    解析单个 tag 的权重。
    """
    s = tag.strip()
    if not s:
        return "", 1.0

    # 显式 (xxx:1.23)
    m = re.fullmatch(r"\(\s*(.+?)\s*:\s*([+-]?\d+(?:\.\d+)?)\s*\)", s)
    if m:
        return m.group(1).strip(), float(m.group(2))

    # 统计外层 () / [] 深度
    w = 1.0
    while True:
        s2 = s.strip()
        if len(s2) >= 2 and s2[0] == "(" and s2[-1] == ")":
            s = s2[1:-1].strip()
            w *= 1.1
            continue
        if len(s2) >= 2 and s2[0] == "[" and s2[-1] == "]":
            s = s2[1:-1].strip()
            w /= 1.1
            continue
        break
    return s.strip(), float(w)

def _build_qwen_text_from_prompt(prompt: str) -> str:
    """Qwen 通道不传权重，只传"干净标签文本"（参考 ComfyUI anima-kai 的做法）"""
    parts = [p.strip() for p in prompt.split(",") if p.strip()]
    clean = []
    for p in parts:
        t, _w = _parse_weighted_tag(p)
        if t:
            clean.append(t)
    return ", ".join(clean)

def encode_qwen(model, tokenizer, texts, device, max_length: int = 512):
    """Qwen 文本编码。

    Qwen3 tokenizer 对空字符串可能返回 0 tokens（会导致模型内部 reshape 失败）。
    ComfyUI 的 AnimaTokenizer 设置了 min_length=1，这里做同等兜底。

    ★ batch_size=1 时走 cache：caption 重复时直接命中，跳过 GPU forward。
      typical 训练 batch_size=1, grad_accum=4：4 个 micro-batch 各一次 forward，
      若 4 个 caption 都相同（shuffle off 时常见）则 4× 加速 Qwen 部分。
    """
    if isinstance(texts, str):
        texts = [texts]
    texts = [(" " if (t is None or str(t).strip() == "") else str(t)) for t in texts]

    # Cache 仅在 batch_size=1 时启用（多 sample 时 batched matmul 已经摊销，cache 命中率低）
    if len(texts) == 1 and _TEXT_CACHE_ENABLED:
        key = (texts[0], int(max_length), id(model))
        cached = _cache_get(_QWEN_CACHE, key)
        if cached is not None:
            hidden_cached, attn_cached = cached
            # 防御性 .to(device)（一般已经在 device 上；模型搬家后可能错）
            return hidden_cached.to(device), attn_cached.to(device)

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    # 仍可能出现空序列（极端 tokenizer 行为），强制塞 1 个 token
    if inputs["input_ids"].ndim == 2 and inputs["input_ids"].shape[1] == 0:
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0
        bs = len(texts)
        inputs["input_ids"] = torch.full((bs, 1), int(pad_id), dtype=torch.long)
        inputs["attention_mask"] = torch.ones((bs, 1), dtype=torch.long)
    inputs = inputs.to(device)

    with torch.inference_mode():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

    hidden = outputs.hidden_states[-1]
    # 清零 padding 位置
    mask = inputs["attention_mask"].unsqueeze(-1)
    hidden = hidden * mask

    # Cache 命中后的张量需要不与 autograd 图关联（hidden 已经在 inference_mode 下生成）
    if len(texts) == 1 and _TEXT_CACHE_ENABLED:
        _cache_put(_QWEN_CACHE, (texts[0], int(max_length), id(model)),
                   (hidden.detach(), inputs["attention_mask"].detach()))

    return hidden, inputs["attention_mask"]

def tokenize_t5_weighted(tokenizer, texts, max_length: int = 512):
    """逐 tag 分词并附带 token-level 权重。

    返回：input_ids, attention_mask(1=有效), token_weights
    padding 位的权重置 1（恒等），避免下游 cross *= t5_w 把 pad 位置的条件信号抹零。

    ★ batch_size=1 时走 cache：每 tag 单独 tokenize 加 regex 解析权重，CPU 开销不可忽略。
    """
    if isinstance(texts, str):
        texts = [texts]

    # batch=1 cache fast path
    if len(texts) == 1 and _TEXT_CACHE_ENABLED:
        key = (texts[0], int(max_length), id(tokenizer))
        cached = _cache_get(_T5_CACHE, key)
        if cached is not None:
            return cached[0].clone(), cached[1].clone(), cached[2].clone()

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1

    all_ids = []
    all_w = []
    for text in texts:
        tags = [t.strip() for t in str(text).split(",") if t.strip()]
        ids = []
        ws = []
        for tag in tags:
            clean_tag, weight = _parse_weighted_tag(tag)
            if not clean_tag:
                continue
            tok = tokenizer(clean_tag, add_special_tokens=False)
            for tid in tok["input_ids"]:
                ids.append(int(tid))
                ws.append(float(weight))

        # 末尾补一个 eos（ComfyUI 也是最后加一个终止 token）
        ids.append(int(eos_id))
        ws.append(1.0)

        # 截断到 max_length（保留最后一个 eos）
        if max_length and len(ids) > max_length:
            ids = ids[: max_length - 1] + [int(eos_id)]
            ws = ws[: max_length - 1] + [1.0]

        all_ids.append(torch.tensor(ids, dtype=torch.long))
        all_w.append(torch.tensor(ws, dtype=torch.float32))

    # pad 到 batch 内最长
    max_len = max(x.numel() for x in all_ids) if all_ids else 1
    input_ids = torch.full((len(all_ids), max_len), pad_id, dtype=torch.long)
    token_w = torch.ones((len(all_w), max_len), dtype=torch.float32)
    attention_mask = torch.zeros((len(all_ids), max_len), dtype=torch.long)

    for i, (ids, ws) in enumerate(zip(all_ids, all_w)):
        L = ids.numel()
        input_ids[i, :L] = ids
        token_w[i, :L] = ws
        attention_mask[i, :L] = 1

    # 写回 cache（仅 batch=1）
    if len(texts) == 1 and _TEXT_CACHE_ENABLED:
        _cache_put(_T5_CACHE, (texts[0], int(max_length), id(tokenizer)),
                   (input_ids.clone(), attention_mask.clone(), token_w.clone()))

    return input_ids, attention_mask, token_w


# ── 以下摘自 upstream `trainer/models.py`（同属文本编码侧）─────────────

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
