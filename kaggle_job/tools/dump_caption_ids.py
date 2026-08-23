r"""把数据集的 caption **只 tokenize 不编码**，落成一个几十 KB 的 ids 文件。

## 为什么

krea2 的文本条件是 Qwen3-VL 的 12 层 hidden 堆叠，bf16 下 61.4KB/token ——
96 张图的 `<stem>.textfeat.npz` 合计 **1.7GB**（实测），占上传量的 90%。
而它的信息来源只有 96 条 caption（合计 0.14MB）。

本工具产出的 `caption_ids.npz` 是那 1.7GB 的**完整前身**：几百 KB 的 token id。
把它连同文本塔权重一起给 TPU，`jax_tpu/text_cache.py` 在真机上现算出同样的
`<stem>.textfeat.npz`（数值口径由 K3 闸门守，见 `jax_tpu/qwen3vl_te.py`）。

顺带解决 caption 明文的暴露面：**上传的是整数 id，不是文本**，本文件也不把
caption 原文写进产物。

## 落盘口径（与 `trainer/model_family.py:_encode_krea2_batch` 逐行对齐）

    clean  = _build_qwen_text_from_prompt(caption)      # 剥掉 (tag:1.5) 权重
    full   = _KREA2_PROMPT_PREFIX + (clean or " ")
    ids    = tok(full).input_ids + tok(suffix, add_special_tokens=False).input_ids

**这里只做 B=1（无 padding）那一条路**，因为已落盘的缓存就是
`cache_text_features.py` 逐条编码出来的。batch 里的中段 padding 会改条件
（`jax_tpu/tests/dump_qwen3vl_ref.py` 实测 max|Δ|=6.7e-1），不是等价摆法。

两道构造期闸门（错了就停，不静默产出错口径的 ids）：
  * 前缀 34 个 token（`_KREA2_PREFIX_IDX`）必须**逐 id** 等于单独 tokenize
    prefix 的结果 —— BPE 若跨 prefix/caption 边界合并，下游 `[34:]` 那一刀
    就切错了位置；
  * suffix 必须逐 id 稳定。

## 产物 `caption_ids.npz`

    ids        int32 [N, Lmax]  右侧 0 填充的完整序列（prefix+caption+suffix）
    lens       int32 [N]        每条有效长度
    stems      <U   [N]         对应 `<stem>.npz` 的 stem（不含扩展名）
    empty_ids  int32 [Le]       空 caption 那一份（caption_dropout 用），可选
    prefix_len int32            要从 hidden 前面切掉的 token 数（=34）
    meta       json             tokenizer 路径 / max_length / 版本 / 词表大小

用法（torch 解释器 —— 只用它的 transformers tokenizer）：
    <py> dump_caption_ids.py --data-dir <图片或缓存目录> \
        --tokenizer <Qwen3-VL-4B-Instruct 目录> -o caption_ids.npz [--empty-caption]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

VERSION = "1"
IMG_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _stems_from(root: Path, prefer_json: bool):
    """数据目录 -> [(stem, caption)]。图片优先；没有图片时按 `<stem>.npz` 推
    （与 `jax_tpu/data.py:_stems` 的回退口径一致，缓存目录可以完全不含像素）。"""
    from cache_text_features import read_caption

    imgs = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT)
    if imgs:
        return [(p.with_suffix(""), read_caption(p, prefer_json=prefer_json))
                for p in imgs]
    out = []
    for p in sorted(root.rglob("*.npz")):
        base = p.name[: -len(".npz")]
        if base.endswith(".textfeat") or ".ms" in base:
            continue
        stem = p.with_suffix("")
        # 无图目录：read_caption 的查找顺序按扩展名走，给它一个同 stem 的
        # 虚拟图片路径即可（它只做 with_suffix，不 open 图片本身）。
        out.append((stem, read_caption(stem.with_suffix(".png"),
                                       prefer_json=prefer_json)))
    return out


def build_ids(tok, caption: str, max_length: int, prefix_idx: int,
              prefix: str, suffix: str, suffix_start_idx: int) -> np.ndarray:
    """一条 caption -> 完整 ids（prefix + caption + suffix）。"""
    from _vendor.t5_weighted import _build_qwen_text_from_prompt

    clean = _build_qwen_text_from_prompt(str(caption or ""))
    full = prefix + (clean if clean else " ")
    if max_length and int(max_length) > 0:
        # 官方定长口径：截断到 max_length + prefix_idx - suffix_start_idx。
        # **不 pad** —— padding 由 TPU 侧按 quantum 在 suffix 之后补（右侧）。
        enc = tok(full, truncation=True,
                  max_length=int(max_length) + prefix_idx - suffix_start_idx)
    else:
        enc = tok(full, truncation=False)
    suf = tok(suffix, add_special_tokens=False)["input_ids"]
    return np.asarray(list(enc["input_ids"]) + list(suf), np.int32)


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--tokenizer", required=True,
                    help="Qwen3-VL-4B-Instruct 的 HF 目录（只读 tokenizer，不加载权重）")
    ap.add_argument("-o", "--out", default="caption_ids.npz")
    ap.add_argument("--max-length", type=int, default=0,
                    help="0 = 不截断（与 krea2_text_max_length: 0 一致）")
    ap.add_argument("--empty-caption", action="store_true",
                    help="额外产出空 caption 的 ids（caption_dropout_rate>0 时必需）")
    ap.add_argument("--no-json", action="store_true")
    a = ap.parse_args()

    from transformers import AutoTokenizer

    from _vendor.krea2_te import (_KREA2_PREFIX_IDX, _KREA2_PROMPT_PREFIX,
                                  _KREA2_PROMPT_SUFFIX, _KREA2_SUFFIX_START_IDX)

    tok = AutoTokenizer.from_pretrained(a.tokenizer)

    # ── 闸门①：prefix 必须逐 id 稳定，且不与 caption 跨边界合并 ─────────────
    pref_alone = np.asarray(tok(_KREA2_PROMPT_PREFIX)["input_ids"], np.int32)
    if pref_alone.size != _KREA2_PREFIX_IDX:
        raise SystemExit(
            f"prefix 单独 tokenize 得到 {pref_alone.size} 个 token，"
            f"但 trainer/model_family.py 的 _KREA2_PREFIX_IDX = {_KREA2_PREFIX_IDX}。"
            f"\n  tokenizer 与训练侧不是同一份，切片位置会错 —— 停。")

    root = Path(a.data_dir)
    pairs = _stems_from(root, prefer_json=not a.no_json)
    if not pairs:
        raise SystemExit(f"{root} 下没找到样本")

    stems, seqs, missing = [], [], []
    for stem, cap in pairs:
        if cap is None:
            missing.append(stem.name)
            continue
        ids = build_ids(tok, cap, a.max_length, _KREA2_PREFIX_IDX,
                        _KREA2_PROMPT_PREFIX, _KREA2_PROMPT_SUFFIX,
                        _KREA2_SUFFIX_START_IDX)
        if not np.array_equal(ids[:_KREA2_PREFIX_IDX], pref_alone):
            raise SystemExit(
                f"{stem.name}: caption 与 prefix 在 BPE 上跨边界合并了 —— "
                f"前 {_KREA2_PREFIX_IDX} 个 id 与单独 tokenize 的 prefix 不同。"
                f"\n  下游按 [{_KREA2_PREFIX_IDX}:] 切 hidden 会切错位置，停。")
        stems.append(stem.name)
        seqs.append(ids)

    if missing:
        print(f"[!] {len(missing)} 条没有 caption，已跳过：{missing[:5]}")
    if not seqs:
        raise SystemExit("一条都没有 caption")

    lens = np.asarray([s.size for s in seqs], np.int32)
    ids = np.zeros((len(seqs), int(lens.max())), np.int32)
    for i, s in enumerate(seqs):
        ids[i, : s.size] = s

    payload = dict(
        ids=ids, lens=lens, stems=np.asarray(stems),
        prefix_len=np.int32(_KREA2_PREFIX_IDX),
        meta=np.array(json.dumps({
            "version": VERSION, "family": "krea2",
            "tokenizer": str(a.tokenizer), "max_length": int(a.max_length),
            "vocab_size": int(getattr(tok, "vocab_size", 0)),
            "txt_layers": 12,
        }, ensure_ascii=False)))
    if a.empty_caption:
        payload["empty_ids"] = build_ids(
            tok, "", a.max_length, _KREA2_PREFIX_IDX, _KREA2_PROMPT_PREFIX,
            _KREA2_PROMPT_SUFFIX, _KREA2_SUFFIX_START_IDX)

    out = Path(a.out)
    np.savez_compressed(out, **payload)
    sz = out.stat().st_size
    n_tok = int(lens.sum())
    print(f"{len(seqs)} 条 caption -> {out}（{sz / 1024:.1f}KB）")
    print(f"  token 数 {int(lens.min())}~{int(lens.max())}（含 {_KREA2_PREFIX_IDX} 前缀"
          f"+suffix），合计 {n_tok}")
    print(f"  同等内容的 textfeat 缓存约 {n_tok * 12 * 2560 * 2 / 1e6:.0f}MB"
          f"（压缩比 {n_tok * 12 * 2560 * 2 / max(sz, 1):.0f}×）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
