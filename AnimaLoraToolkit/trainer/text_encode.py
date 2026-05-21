"""文本编码与 prompt-tag 权重解析。

输入：
- Qwen 通道：清洗后的纯标签文本（不含权重），走 causal LM hidden_states。
- T5 通道：逐 tag tokenize，每个 token 附带权重；padding 位权重为 1（恒等）。

权重语法（参考 ComfyUI / Anima 指南）：
- `(tag:1.5)` 显式
- `(tag)` / `((tag))` → 1.1^n
- `[tag]` → 1/1.1

主训练脚本通过下游对 `cross *= t5_w` 来把每 token 权重注入 cross-attn 条件信号。
"""

from __future__ import annotations

import re
from collections import OrderedDict

import torch


# ============================================================================
# 文本编码 LRU cache（按 (text, max_length) 命中）
# ============================================================================
#
# 训练循环里 caption 大概率重复（shuffle_caption=False 时同图 caption 完全静态；
# JSON 模式有 shuffle 但归一化后的 character/series/artist 等"固定部分"重复）。
# 每个 batch 都跑一遍 Qwen + T5 forward 是大头时间开销（Qwen3-0.6B 0.1-0.3s/seq）。
#
# 这个 cache 给 _encode_qwen_single / _tokenize_t5_single 用，命中时跳过 GPU forward。
# - cache key: (text, max_length)
# - cache 仅在 batch_size=1 路径生效；batch>1 走原 batched 路径（合并 cache 不值得）
# - 容量限制 2048 条，避免长训练 caption 多样化时 OOM
# - 模型变更（reload checkpoint 等）后用 reset_text_encode_cache() 清空

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
