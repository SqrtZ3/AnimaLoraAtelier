"""上传前校验缓存目录 —— 纯 numpy，不需要 jax / torch。

## 为什么要单独一个校验器

TPU 侧 `jax_tpu/data.py:CacheDataset._scan` 已经对缓存做了严格 fail-fast。
但它在**真机上**才跑：缺一个 `_empty.textfeat.npz`，代价是 push 一轮、等挂载、
起 TPU、然后挂掉 —— 白烧一分钟配额加十几分钟等待。

本工具把同一套判据搬到本地，跑一遍几秒钟。判据与 `jax_tpu/data.py` 对齐
（`_stems` 的 sidecar 排除口径、latent 形状、flip、krea2 的 `txt` 键），
外加两条**只在上传场景成立**的：

  * 目录里不许有 png/jpg/txt —— TPU 侧从不读像素，放原图是纯多余的暴露。
    2026-08-21 真实吃过一次：96 张原图跟缓存一起传上 Kaggle，被按 NSFW 条款
    **整个 dataset 删掉**并向账号告警。
  * 单图 token 数 ≤ 单卡预算（`navit_token_budget / 8`）—— 超了打包器会
    fail-fast（"单段 N > budget"），这同样是上机之后才发现。

## 另一重作用：格式契约的**执行体**

`docs/CACHE_FORMAT.md` 写了 npz 该长什么样，但文档不会拦住任何错误。
如果你用自己的 VAE 实现产缓存（完全可以 —— TPU 侧只认 npz，不认工具），
跑一遍这个就知道格式对不对，不必读我们的 torch 代码。

## 用法

    python verify_cache.py <缓存目录> [--config <训练 yaml>] [--upload-ready]

给 `--config` 就顺带核对 yaml 的开关与缓存是否配套（flip / caption_dropout /
multiscale / token 预算）—— 这是最容易出事的地方，强烈建议给。
`--upload-ready` 额外检查"能不能直接传上 Kaggle"（无像素、无明文 caption）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

IMG_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
_MS_SIDECAR = re.compile(r"\.ms\d+$")
LATENT_CHANNELS = 16
PATCH = 2

#: anima 的 cross 是定长 [512, 1024]；krea2 的 txt 是变长 [L, 12, 2560]。
ANIMA_CROSS = (512, 1024)
KREA2_TXT_LAYERS = 12

PROBLEMS: list = []
NOTES: list = []


def bad(msg: str) -> None:
    PROBLEMS.append(msg)


def note(msg: str) -> None:
    NOTES.append(msg)


def stems_of(d: Path) -> list:
    """与 `jax_tpu/data.py:CacheDataset._stems` 同口径：有图按图名推，
    没图按 `<stem>.npz` 推并排掉 sidecar。"""
    imgs = {p.with_suffix("") for p in d.iterdir() if p.suffix.lower() in IMG_EXT}
    if imgs:
        return sorted(imgs)
    out = set()
    for p in d.iterdir():
        if p.suffix.lower() != ".npz":
            continue
        base = p.name[:-len(".npz")]
        if base.endswith(".textfeat") or _MS_SIDECAR.search(base):
            continue
        out.add(p.with_suffix(""))
    return sorted(out)


def check_latent(npz: Path, label: str, need_flip: bool) -> int:
    """校验一个 latent npz，返回 token 数（出错返回 -1）。"""
    try:
        with np.load(npz) as z:
            files = set(z.files)
            if "latent" not in files:
                bad(f"{label}: 没有 `latent` 键（有 {sorted(files)}）")
                return -1
            shape = z["latent"].shape
            if need_flip and "latent_flipped" not in files:
                bad(f"{label}: 缺 `latent_flipped`，但 yaml 的 flip_augment 是开的。"
                    f"翻转必须在像素域做完再 encode（VAE 卷积非 flip-等变），"
                    f"训练时补不了 —— 带 flip 重跑缓存，或关掉 flip_augment")
            for k in ("bucket_w", "bucket_h"):
                if k not in files:
                    bad(f"{label}: 缺 `{k}`")
    except Exception as e:
        bad(f"{label}: 读不出来（{type(e).__name__}: {e}）")
        return -1

    if len(shape) != 4 or shape[0] != LATENT_CHANNELS:
        bad(f"{label}: latent 形状 {shape}，应为 [{LATENT_CHANNELS}, T, H, W]")
        return -1
    _, _, h, w = shape
    if h % PATCH or w % PATCH:
        bad(f"{label}: latent {h}x{w} 不能被 patch {PATCH} 整除")
        return -1
    return (h // PATCH) * (w // PATCH)


def check_textfeat(npz: Path, label: str, family: str) -> int:
    """校验一个 textfeat npz，返回 text token 数（anima 恒 0，krea2 是 L）。"""
    if not npz.exists():
        bad(f"{label}: 缺 `{npz.name}` —— 跑 tools/cache_text_features.py"
            + ("（krea2 也可用 dump_caption_ids.py 让 TPU 现算）" if family == "krea2" else ""))
        return -1
    try:
        with np.load(npz, allow_pickle=True) as z:
            files = set(z.files)
            if family == "krea2":
                if "txt" not in files:
                    bad(f"{label}: krea2 的 textfeat 要有 `txt` 键（有 {sorted(files)}）")
                    return -1
                sh = z["txt"].shape
                if len(sh) != 3 or sh[1] != KREA2_TXT_LAYERS:
                    bad(f"{label}: `txt` 形状 {sh}，应为 [L, {KREA2_TXT_LAYERS}, D]")
                    return -1
                return int(sh[0])
            if "cross" not in files:
                bad(f"{label}: anima 的 textfeat 要有 `cross` 键（有 {sorted(files)}）。\n"
                    f"    注意 cross **不是** Qwen hidden，是过了 llm_adapter 的输出 —— "
                    f"只存 Qwen hidden 会让训练一直条件错误且不报错")
                return -1
            sh = z["cross"].shape
            if tuple(sh) != ANIMA_CROSS:
                bad(f"{label}: `cross` 形状 {tuple(sh)}，应为 {ANIMA_CROSS}")
                return -1
            return 0
    except Exception as e:
        bad(f"{label}: 读不出来（{type(e).__name__}: {e}）")
        return -1


def check_upload_ready(d: Path) -> None:
    pixels = [p.name for p in d.iterdir() if p.suffix.lower() in IMG_EXT]
    texts = [p.name for p in d.iterdir() if p.suffix.lower() in (".txt", ".json")
             and p.name != "dataset-metadata.json"]
    if pixels:
        bad(f"[上传闸门] 目录里有 {len(pixels)} 个图片文件（如 {pixels[:3]}）。\n"
            f"    TPU 侧从不读像素，传上去是纯多余的暴露 —— 2026-08-21 真实因此"
            f"被 Kaggle 按 NSFW 条款删过整个 dataset。只传 npz。")
    if texts:
        bad(f"[上传闸门] 目录里有 {len(texts)} 个 .txt/.json（如 {texts[:3]}）——"
            f" caption 已烘焙进 textfeat，明文不必上传")
    # npz 内部的 caption 明文
    leaky = []
    for p in sorted(d.glob("*.textfeat.npz")):
        try:
            with np.load(p, allow_pickle=True) as z:
                if "caption" in z.files and str(z["caption"]) not in ("", "b''"):
                    leaky.append(p.name)
        except Exception:
            pass
    if leaky:
        note(f"[上传提示] {len(leaky)} 个 textfeat 内含 caption 明文（如 {leaky[:3]}）。"
             f"训练路径只读 `cross`/`txt`，用 tools/dataset_encrypt.py 剥掉逐 bit 无影响。"
             f"（tools/kaggle_fast_upload.py 默认也会拦这个）")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cache_dir", help="缓存目录（npz 所在处）")
    ap.add_argument("--config", default="",
                    help="训练 yaml。给了就核对开关与缓存是否配套（强烈建议给）")
    ap.add_argument("--upload-ready", action="store_true",
                    help="额外检查能否直接上传 Kaggle（无像素 / 无明文 caption）")
    ap.add_argument("--devices", type=int, default=8,
                    help="TPU 卡数，用于把全局 token 预算换成单卡预算")
    a = ap.parse_args()

    d = Path(a.cache_dir)
    if not d.is_dir():
        raise SystemExit(f"[ FATAL ] {d} 不是目录")

    family, need_flip, want_ms, ladder, cap_drop, budget = "anima", False, False, [], 0.0, 0
    if a.config:
        import yaml
        cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8")) or {}
        family = str(cfg.get("model_family", "anima") or "anima").lower()
        need_flip = bool(cfg.get("flip_augment", False))
        want_ms = bool(cfg.get("navit_multiscale", False))
        ladder = [int(x) for x in
                  str(cfg.get("navit_multiscale_token_ladder", "") or "").split(",")
                  if x.strip()]
        cap_drop = float(cfg.get("caption_dropout_rate", 0) or 0)
        gb = int(cfg.get("navit_token_budget", 0) or 0)
        if gb and gb % a.devices:
            bad(f"navit_token_budget={gb} 不能被 {a.devices} 整除 —— TPU 上它是"
                f"**全局**预算，单卡拿 1/{a.devices}，不整除会 fail-fast")
        budget = gb // a.devices if gb else 0
        print(f"配置: family={family} flip={need_flip} multiscale={want_ms}"
              f"{ladder or ''} caption_dropout={cap_drop} 单卡预算={budget or '未设'}")
    else:
        note("没给 --config：跳过了 flip / caption_dropout / multiscale / token 预算的"
             "配套检查。这几项恰是最容易出事的 —— 建议带上 yaml 再跑一遍")

    stems = stems_of(d)
    if not stems:
        raise SystemExit(f"[ FATAL ] {d} 下没找到任何样本（既无图片也无 `<stem>.npz`）")
    print(f"扫到 {len(stems)} 个样本\n")

    tokens, n_ms = [], 0
    for stem in stems:
        label = stem.name
        ntok = check_latent(stem.with_suffix(".npz"), label, need_flip)
        tl = check_textfeat(Path(str(stem) + ".textfeat.npz"), label, family)
        if ntok > 0:
            tokens.append((label, ntok + max(0, tl)))
        # multiscale sidecar
        for side in sorted(d.glob(f"{stem.name}.ms*.npz")):
            n_ms += 1
            stok = check_latent(side, side.name, need_flip)
            if stok > 0:
                tokens.append((side.name, stok + max(0, tl)))

    if want_ms and n_ms == 0:
        bad(f"yaml 开了 navit_multiscale（档位 {ladder}）但目录下没有任何 `*.ms<档>.npz`。"
            f"缓存时要带 --ms-ladder {','.join(map(str, ladder)) or '<档位>'}")
    if n_ms and not want_ms:
        note(f"目录里有 {n_ms} 个 ms sidecar，但 yaml 的 navit_multiscale 是关的 ——"
             f" 它们不会被用到（不影响训练，只是白占上传体积）")
    if want_ms and ladder:
        found = {int(m.group(1)) for p in d.glob("*.ms*.npz")
                 for m in [re.search(r"\.ms(\d+)\.npz$", p.name)] if m}
        extra, absent = found - set(ladder), set(ladder) - found
        if absent:
            bad(f"yaml 的档位 {sorted(absent)} 在缓存里一个 sidecar 都没有 ——"
                f" 档位不一致会让训练看到的样本集与 plan-only 不同")
        if extra:
            note(f"缓存里有 yaml 未列的档位 {sorted(extra)}，不会被加载")

    if cap_drop > 0 and not (d / "_empty.textfeat.npz").exists():
        bad(f"caption_dropout_rate={cap_drop} > 0 但没有 `_empty.textfeat.npz`。"
            f"文本特征是离线缓存的，训练时没有编码器现算空 caption —— "
            f"缓存时加 --empty-caption")

    if budget and tokens:
        over = [(n, t) for n, t in tokens if t > budget]
        mx = max(tokens, key=lambda x: x[1])
        print(f"token: 最大 {mx[1]}（{mx[0]}），单卡预算 {budget}")
        if over:
            bad(f"{len(over)} 个样本超单卡预算 {budget}（最大 {over[0]}）。"
                f"打包器会 fail-fast（'单段 N > budget'）—— 要么调大 "
                f"navit_token_budget，要么把这些图缩小/加 ms 档位")

    if a.upload_ready:
        check_upload_ready(d)

    print()
    for m in NOTES:
        print(f"  [注意] {m}")
    print("=" * 64)
    if PROBLEMS:
        print(f"{len(PROBLEMS)} 个问题：\n")
        for m in PROBLEMS:
            print(f"  [问题] {m}")
        print("\n这些在真机上都会 fail-fast —— 先在本地修掉，别烧配额去发现。")
    else:
        print(f"缓存检查通过：{len(stems)} 个样本"
              + (f" + {n_ms} 个 ms sidecar" if n_ms else "")
              + ("，可以上传" if a.upload_ready else ""))
    print("=" * 64)
    return 1 if PROBLEMS else 0


if __name__ == "__main__":
    raise SystemExit(main())
