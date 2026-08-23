r"""把数据集的文本特征离线算好并落盘（PyTorch 侧工具，给 TPU 后端喂料）。

## 为什么需要它

TPU 后端（`jax_tpu/`）只跑 DiT，**前提是 latent 与文本特征都已离线算好**。
latent 侧仓库已有 Kohya 风格 `.npz` 缓存（`CachedLatentDataset`），文本侧目前
只有内存 LRU（`trainer/text_encode.py` 的 `_QWEN_CACHE`），进程一退就没了 ——
把 1.2GB 的 Qwen3-0.6B 搬上 TPU 只为了算一次 caption 是纯浪费。

本工具对 GPU/NPU 训练**没有影响**，它只是新增一份 sidecar 文件。

## 落盘口径（与训练路径逐行一致，抄错就是静默错）

复刻 anima_train.py:3720-3733 的整条链，**不是**简单存 Qwen hidden：

    qwen_texts = _build_qwen_text_from_prompt(caption)        # 剥掉 (tag:1.2) 权重标记
    qwen_emb, qwen_attn = encode_qwen(...)                     # 变长（本数据集实测 82）
    t5_ids, t5_attn, t5_w = tokenize_t5_weighted(..., 512)     # 定长 512
    cross = model.preprocess_text_embeds(qwen_emb, t5_ids, t5_attn, qwen_attn)
    if use_t5_token_weights and llm_adapter is not None:
        cross = cross * t5_w                                   # 逐 token 权重
    if cross.shape[1] < 512: cross = F.pad(cross, ..., 512)    # 尾部零填充

**关键点：底模里有 llm_adapter**（anima-base-v1.0.safetensors 实测 118 个键），
所以 `preprocess_text_embeds` 走的是 `llm_adapter(qwen_emb, t5_ids, t5_attn,
qwen_attn)` 这条支路，输出是 512 长——**cross 条件不是 Qwen 的 hidden**。
只存 Qwen hidden 会让 TPU 训练一直条件错误且不报错。
llm_adapter 默认不注入 LoRA（anima_train.py:1327 的 DEFAULT_EXCLUDE_PREFIXES），
是冻结的，所以它的输出可以安全地离线缓存。

**保留 512 的填充、不做 trim**：`navit_text_trim_padding` 在 trainer/config.py:540
默认 False，因为训练去 pad 而 eval/采样带 pad 会让 cross-attn 条件不一致
（memory `[[navit-text-trim-train-eval-mismatch]]` 有 A/B 实证）。

  * caption 来源与 trainer/data.py:823-834 一致：优先 `.json`，否则 `.txt` / `.caption`。
  * dtype 存 bf16（训练本就在 bf16 下做，fp32 只是让文件大一倍）。

## 产物

每张图一个 `<stem>.textfeat.npz`，含：
    cross   [512, 1024]  bf16（以 uint16 位模式存，numpy 没有原生 bf16）
    mask    [512]        uint8，t5 侧的有效位（**仅供诊断**，训练不用它 mask）
    caption str          原文，便于事后核对
    meta    json         max_length / 编码器路径 / 工具版本

用法：
    <torch-python> cache_text_features.py --data-dir <图片目录> \
        --text-encoder <models/text_encoders/Qwen3-0.6B-Base> [--overwrite]

## krea2 模式（--model-family krea2）

文本条件换成 Qwen3-VL-4B **12 层** hidden states 堆叠（krea2_modeling.py 头注），
编码走 `trainer/model_family.py:298 encode_krea2_text`（含 `(tag:1.5)` 剥离与
逐条压缩到有效 token）。落盘键是 `txt` [L, 12, 2560]（**变长**，bf16 位模式）
—— TPU 的 navit 打包只取有效 token，不定长槽；`_empty.textfeat.npz` 同格式
（caption_dropout 用）。`--transformer` 在该模式下不需要（TextFusion 在 DiT
内部，离线侧无冻结支路要过）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

VERSION = "1"
IMG_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _bf16_view(t) -> np.ndarray:
    """torch bf16 -> uint16 位模式（numpy 没有原生 bfloat16，直接 .numpy() 会报错）。"""
    import torch
    return t.to(torch.bfloat16).view(torch.uint16).cpu().numpy()


def read_caption(img: Path, prefer_json: bool = True) -> str | None:
    """与 trainer/data.py:823-834 同一套查找顺序。"""
    j = img.with_suffix(".json")
    if prefer_json and j.exists():
        try:
            d = json.loads(j.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        for k in ("caption", "tags", "prompt", "text"):
            v = d.get(k)
            if isinstance(v, str) and v.strip():
                return v
            if isinstance(v, list) and v:
                return ", ".join(str(x) for x in v)
        return None
    for ext in (".txt", ".caption"):
        p = img.with_suffix(ext)
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-family", default="anima", choices=("anima", "krea2"),
                    help="anima（默认，历史行为）/ krea2（Qwen3-VL 12 层堆叠，键 txt）")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--text-encoder", required=True,
                    help="anima: Qwen3-0.6B 目录；krea2: Qwen3-VL-4B-Instruct 目录")
    ap.add_argument("--transformer", default=None,
                    help="anima 必需：anima-base-v1.0.safetensors（cross 要过它里面的 "
                         "llm_adapter，见本文件 docstring）。krea2 不需要")
    ap.add_argument("--t5-tokenizer", default=None,
                    help="默认用仓库的 AnimaLoraToolkit/models/t5_tokenizer")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8, help="krea2 的批编码条数")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-json", action="store_true", help="忽略 .json，只读 .txt")
    ap.add_argument("--empty-caption", action="store_true",
                    help="额外产出 `_empty.textfeat.npz`（caption=\"\"）。"
                         "TPU 后端的 caption_dropout 要靠它：文本特征是离线缓存的，"
                         "训练时没有编码器可以现算空 caption 的条件。")
    a = ap.parse_args()

    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    import torch

    root = Path(a.data_dir)
    imgs = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT)
    if not imgs:
        print(f"{root} 下没找到图片")
        return 1
    print(f"{len(imgs)} 张图，编码器 {a.text_encoder}，max_length={a.max_length}，"
          f"family={a.model_family}")

    if a.model_family == "krea2":
        return main_krea2(a, here, root, imgs, torch)
    return main_anima(a, here, root, imgs, torch)


def main_krea2(a, here, root, imgs, torch) -> int:
    """krea2：Qwen3-VL 12 层堆叠 -> `txt` [L, 12, D]（变长，bf16 位模式）。"""
    from _vendor.krea2_te import encode_krea2_text, load_krea2_text_encoder

    dev = torch.device(a.device)
    handles = load_krea2_text_encoder(a.text_encoder, dev, torch.bfloat16)
    meta = json.dumps({"version": VERSION, "family": "krea2",
                       "max_length": a.max_length,
                       "text_encoder": str(a.text_encoder),
                       "txt_layers": 12}, ensure_ascii=False)

    def encode_one(cap: str) -> np.ndarray:
        with torch.no_grad():
            cross, cmask = encode_krea2_text(handles, [cap], dev, a.max_length)
        f = cross[0][cmask[0]]                       # 只留有效 token（navit 口径）
        return _bf16_view(f)

    done = skipped = missing = 0
    for i, img in enumerate(imgs):
        out = img.with_name(img.stem + ".textfeat.npz")
        if out.exists() and not a.overwrite:
            skipped += 1
            continue
        cap = read_caption(img, prefer_json=not a.no_json)
        if cap is None:
            missing += 1
            print(f"  [跳过] {img.name} 没有 caption")
            continue
        v = encode_one(cap)
        np.savez(out, txt=v, caption=np.array(cap), meta=np.array(meta))
        done += 1
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(imgs)} …")

    if a.empty_caption:
        out = root / "_empty.textfeat.npz"
        if out.exists() and not a.overwrite:
            print("_empty.textfeat.npz 已存在，跳过")
        else:
            v = encode_one("")
            np.savez(out, txt=v, caption=np.array(""), meta=np.array(meta))
            print(f"已写 {out.name}（caption_dropout 用，{v.shape[0]} token）")

    print(f"完成：新写 {done}，已存在跳过 {skipped}，无 caption {missing}")
    if done:
        sz = out.stat().st_size / 1e6
        print(f"单文件约 {sz:.2f}MB（12 层堆叠，约为 anima 格式的 60 倍/token），"
              f"{len(imgs)} 张合计约 {sz * len(imgs):.0f}MB")
    return 0


def main_anima(a, here, root, imgs, torch) -> int:
    import torch.nn.functional as Fn
    from _vendor.llm_adapter import load_llm_adapter
    from _vendor.t5_weighted import (_build_qwen_text_from_prompt, encode_qwen,
                                     load_text_encoders, tokenize_t5_weighted)

    if not a.transformer:
        print("--transformer 在 anima 模式下必需（cross 要过底模的 llm_adapter）")
        return 1
    if not a.t5_tokenizer:
        print("--t5-tokenizer 必需：指向 T5 tokenizer 目录（`spiece.model` 那份）。\n"
              "  没有本地副本时给 'google/t5-v1_1-xxl'，会从 HF 拉 tokenizer（只几 MB）。")
        return 1
    t5_dir = a.t5_tokenizer

    dev = torch.device(a.device)
    qwen, qwen_tok, t5_tok = load_text_encoders(a.text_encoder, t5_dir, dev,
                                                torch.bfloat16)
    # 只加载底模里的 llm_adapter（118 个键 / 269MB），**不是**整个 4.2GB DiT ——
    # 本工具对底模的全部用法就是过一遍 adapter。与 upstream 走
    # `dit.preprocess_text_embeds(...)` 的输出**逐 bit 相同**（已对拍验证），
    # 加载耗时 0.7s vs 42.2s。
    adapter = load_llm_adapter(a.transformer, dev, torch.bfloat16)
    # adapter 加载器缺键即 fail-fast，所以走到这里 adapter 必然完整。
    # （upstream 版本在这里还判 `getattr(dit,"llm_adapter",None) is not None`，
    #   因为它加载的是通用底模、adapter 可能整个不存在。）
    has_adapter = True
    meta = json.dumps({"version": VERSION, "max_length": a.max_length,
                       "text_encoder": str(a.text_encoder),
                       "transformer": str(a.transformer),
                       "llm_adapter": bool(has_adapter),
                       "t5_token_weights": bool(has_adapter),
                       "trim_padding": False}, ensure_ascii=False)

    done = skipped = missing = 0
    for i, img in enumerate(imgs):
        out = img.with_name(img.stem + ".textfeat.npz")
        if out.exists() and not a.overwrite:
            skipped += 1
            continue
        cap = read_caption(img, prefer_json=not a.no_json)
        if cap is None:
            missing += 1
            print(f"  [跳过] {img.name} 没有 caption")
            continue
        with torch.no_grad():
            qtxt = _build_qwen_text_from_prompt(cap)
            q_emb, q_attn = encode_qwen(qwen, qwen_tok, [qtxt], dev, a.max_length)
            t5_ids, t5_attn, t5_w = tokenize_t5_weighted(t5_tok, [cap],
                                                         max_length=a.max_length)
            # upstream: dit.preprocess_text_embeds(q_emb, t5_ids, t5_attn, q_attn)
            #   -> anima_modeling.py:241 转发成 llm_adapter(text_embeds, text_ids,
            #      target_attention_mask=..., source_attention_mask=...)
            # 这里直接调 adapter，实参顺序逐一对应（已逐 bit 对拍）。
            cross = adapter(q_emb, t5_ids.to(dev),
                            target_attention_mask=t5_attn.to(dev),
                            source_attention_mask=q_attn)
            if has_adapter and cross.shape[1] == t5_w.shape[1]:
                cross = cross * t5_w.to(dev, dtype=torch.float32).to(cross.dtype).unsqueeze(-1)
            if cross.shape[1] < a.max_length:
                cross = Fn.pad(cross, (0, 0, 0, a.max_length - cross.shape[1]))
        c = cross[0]
        if c.shape[0] != a.max_length:
            # 不该发生；发生了说明上游口径变了，必须停下来看而不是静默截断
            raise RuntimeError(f"{img.name}: cross 长度 {c.shape[0]} != "
                               f"max_length {a.max_length}，口径对不上，不能静默继续")
        np.savez(out, cross=_bf16_view(c),
                 mask=t5_attn[0].to(torch.uint8).cpu().numpy(),
                 caption=np.array(cap), meta=np.array(meta))
        done += 1
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(imgs)} …")

    if a.empty_caption:
        out = root / "_empty.textfeat.npz"
        if out.exists() and not a.overwrite:
            print("_empty.textfeat.npz 已存在，跳过")
        else:
            with torch.no_grad():
                q_emb, q_attn = encode_qwen(qwen, qwen_tok, [""], dev, a.max_length)
                t5_ids, t5_attn, t5_w = tokenize_t5_weighted(t5_tok, [""],
                                                             max_length=a.max_length)
                cross = adapter(q_emb, t5_ids.to(dev),
                                target_attention_mask=t5_attn.to(dev),
                                source_attention_mask=q_attn)
                if has_adapter and cross.shape[1] == t5_w.shape[1]:
                    cross = cross * t5_w.to(dev, dtype=torch.float32).to(cross.dtype).unsqueeze(-1)
                if cross.shape[1] < a.max_length:
                    cross = Fn.pad(cross, (0, 0, 0, a.max_length - cross.shape[1]))
            np.savez(out, cross=_bf16_view(cross[0]),
                     mask=t5_attn[0].to(torch.uint8).cpu().numpy(),
                     caption=np.array(""), meta=np.array(meta))
            print(f"已写 {out.name}（caption_dropout 用）")

    print(f"完成：新写 {done}，已存在跳过 {skipped}，无 caption {missing}")
    if done:
        sz = out.stat().st_size / 1e6
        print(f"单文件约 {sz:.2f}MB，{len(imgs)} 张合计约 {sz * len(imgs):.0f}MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
