"""Generate block-ablated variants of a trained LoKr file for style localization.

按 B-LoRA / SplitFlux 思路：把 28 个 transformer block 分成 N 个连续组，生成
only_<组>（只保留该组）与 drop_<组>（去掉该组）两套变体，外加 adaln / final_layer
专项开关。变体只是删键（被删模块回退到底模行为），不做任何数值修改，ComfyUI
加载格式与原文件完全一致。

用法:
  python lokr_block_scan.py --src D:/models/LoRA/anima/goutong10-1/xxx_step1200.safetensors
  # 输出到 <srcdir>/blockscan/，附 manifest.csv
"""
import argparse
import csv
import os
import re

from safetensors import safe_open
from safetensors.torch import save_file

BLOCK_RE = re.compile(r"lora_unet_blocks_(\d+)_")
ADALN_RE = re.compile(r"adaln_modulation")
FINAL_RE = re.compile(r"lora_unet_final_layer")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out-dir", help="default <srcdir>/blockscan")
    ap.add_argument("--groups", type=int, default=4, help="number of contiguous block groups")
    ap.add_argument("--modes", default="only,drop", help="comma subset of only,drop")
    ap.add_argument("--no-specials", action="store_true",
                    help="skip drop_adaln / only_adaln / drop_final variants")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(os.path.dirname(args.src), "blockscan")
    os.makedirs(out_dir, exist_ok=True)

    tensors, meta = {}, {}
    with safe_open(args.src, framework="pt", device="cpu") as f:
        meta = dict(f.metadata() or {})
        for k in f.keys():
            tensors[k] = f.get_tensor(k)

    blocks = sorted({int(m.group(1)) for k in tensors for m in [BLOCK_RE.search(k)] if m})
    n_blocks = len(blocks)
    print(f"src: {os.path.basename(args.src)} | {len(tensors)} keys | {n_blocks} blocks")

    def block_of(key):
        m = BLOCK_RE.search(key)
        return int(m.group(1)) if m else None  # None = final_layer etc.

    # contiguous group ranges
    per = (n_blocks + args.groups - 1) // args.groups
    ranges = [(blocks[i], blocks[min(i + per, n_blocks) - 1])
              for i in range(0, n_blocks, per)]

    variants = []  # (name, keep_fn, description)
    modes = {m.strip() for m in args.modes.split(",")}
    for lo, hi in ranges:
        tag = f"b{lo:02d}-{hi:02d}"
        if "only" in modes:
            variants.append((f"only_{tag}",
                             lambda k, lo=lo, hi=hi: (b := block_of(k)) is not None and lo <= b <= hi,
                             f"keep only blocks {lo}-{hi} (incl. their adaln); no final_layer"))
        if "drop" in modes:
            variants.append((f"drop_{tag}",
                             lambda k, lo=lo, hi=hi: not ((b := block_of(k)) is not None and lo <= b <= hi),
                             f"everything except blocks {lo}-{hi}"))
    if not args.no_specials:
        variants += [
            ("drop_adaln", lambda k: not ADALN_RE.search(k),
             "remove all adaln_modulation adapters (haze/contrast-drift suspect)"),
            ("only_adaln", lambda k: bool(ADALN_RE.search(k)),
             "only adaln_modulation adapters"),
            ("drop_final", lambda k: not FINAL_RE.search(k),
             "remove final_layer.linear adapter"),
        ]

    stem = re.sub(r"\.safetensors$", "", os.path.basename(args.src))
    manifest = []
    for name, keep, desc in variants:
        sub = {k: v for k, v in tensors.items() if keep(k)}
        if not sub:
            print(f"  skip {name}: empty")
            continue
        m = dict(meta)
        m["anima_block_scan"] = f"{name}: {desc}"
        path = os.path.join(out_dir, f"{stem}__{name}.safetensors")
        save_file(sub, path, metadata=m)
        manifest.append((name, len(sub), desc, os.path.basename(path)))
        print(f"  {name:<14} {len(sub):>5} keys  {desc}")

    with open(os.path.join(out_dir, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "n_keys", "description", "file"])
        w.writerows(manifest)
    print(f"\n{len(manifest)} variants -> {out_dir}")


if __name__ == "__main__":
    main()
