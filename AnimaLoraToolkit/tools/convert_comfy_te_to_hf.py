#!/usr/bin/env python
"""ComfyUI 单文件文本编码器 → HuggingFace 目录。

**为什么需要它**：`trainer/models.py:load_text_encoders` 走 `AutoModelForCausalLM`，
要的是一个 HF 目录（`config.json` + tokenizer + 权重）。而各平台（OpenI/启智的
`FoundationModel/Anima`、ComfyUI 生态的分发）给的是**单文件 safetensors**，键名少一层
或多一层前缀。仓库里 `models/text_encoders/Qwen3-0.6B-Base/` 的 config/tokenizer 是齐的，
缺的只有权重（`model.safetensors` 是 135 字节的 LFS 指针）。本工具把单文件补进去。

**设计原则：不猜键名。**
1. 期望键集**从 config.json 反推**（meta device 实例化模型取 state_dict 的键与形状），
   不是硬编码一张表——换模型/换 transformers 版本都不会失效。
2. 前缀差异只用**整体统一变换**修复（对所有键加/去同一个前缀），不做逐键模糊匹配。
   统一变换要么全对要么全错，可验证；逐键猜测会静默配错。
3. 每个键都做**形状校验**；有任何一个期望键没被覆盖就直接报错退出，不产出半成品。

**流式写出**：本地内存紧张（见 memory `[[comfyui-te-key-namespace]]`），不用
`safetensors.torch.save_file`（要求整个 state_dict 同时在内存）。这里按 safetensors
格式手写：先算好 header，再逐张量拷贝字节，峰值内存 ≈ 最大单个张量。

用法
----
    python tools/convert_comfy_te_to_hf.py \
        --src  /path/to/qwen_3_06b_base.safetensors \
        --out  models/text_encoders/Qwen3-0.6B-Base-full

    # 只体检不写盘（先确认键名对得上）
    python tools/convert_comfy_te_to_hf.py --src ... --out ... --dry-run

`--template` 默认取 `models/text_encoders/Qwen3-0.6B-Base`（config + tokenizer 从这里
拷到 --out）。转完可以直接把 `--out` 传给 yaml 的 `text_encoder_path`。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
from pathlib import Path

# Windows 控制台默认 GBK，输出里的 ✓/中文会 UnicodeEncodeError。云端 Linux 无此问题，
# 但本地预演也要能跑通，所以显式切 UTF-8（失败就算了，不影响功能）。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

DEFAULT_TEMPLATE = _REPO / "models" / "text_encoders" / "Qwen3-0.6B-Base"

# 会跟随 config/tokenizer 一起拷到输出目录的文件（存在才拷）
_TEMPLATE_FILES = (
    "config.json", "generation_config.json",
    "tokenizer.json", "tokenizer_config.json",
    "vocab.json", "merges.txt", "special_tokens_map.json",
    "added_tokens.json", "chat_template.jinja",
)

# 尝试的统一前缀变换。每一项是 (描述, 函数)；函数把**源文件的键**映射成 HF 键。
# 只保留能让「期望键集 ⊆ 变换后的源键集」成立的那个。
_PREFIX_CANDIDATES = (
    ("identity（键名已经是 HF 口径）", lambda k: k),
    ("去掉 'model.diffusion_model.'", lambda k: k[len("model.diffusion_model."):]
     if k.startswith("model.diffusion_model.") else k),
    ("补上 'model.'", lambda k: k if k.startswith("model.") else "model." + k),
    ("去掉 'text_encoders.qwen3.transformer.'",
     lambda k: k.split("transformer.", 1)[-1] if "transformer." in k else k),
    ("插入 'language_model.'（ComfyUI 比 HF 少这一层）",
     lambda k: k.replace("model.", "model.language_model.", 1)
     if k.startswith("model.") else k),
    ("去掉 'language_model.'（ComfyUI 比 HF 多这一层）",
     lambda k: k.replace("language_model.", "", 1)),
    ("去 'language_model.' 再补 'model.'",
     lambda k: ("model." + k.replace("language_model.", "", 1))
     if not k.replace("language_model.", "", 1).startswith("model.")
     else k.replace("language_model.", "", 1)),
)

# safetensors dtype 字符串 ↔ torch dtype
_DT2ST = {
    "torch.float32": "F32", "torch.float16": "F16", "torch.bfloat16": "BF16",
    "torch.float64": "F64", "torch.int64": "I64", "torch.int32": "I32",
    "torch.int16": "I16", "torch.int8": "I8", "torch.uint8": "U8", "torch.bool": "BOOL",
}


def _st_to_torch(name):
    """safetensors 的 dtype 字符串（'BF16'…）→ torch dtype。已经是 dtype 就原样返回。"""
    import torch
    if not isinstance(name, str):
        return name
    table = {
        "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
        "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
        "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool,
    }
    key = name.upper().replace("TORCH.", "")
    if key in table:
        return table[key]
    got = getattr(torch, name.replace("torch.", ""), None)
    if isinstance(got, torch.dtype):
        return got
    raise SystemExit(f"✗ 不认识的 dtype: {name!r}")


def _expected_keys(template_dir: Path) -> dict:
    """从 config.json 反推 HF 期望的 {键: 形状}。不硬编码键名表。"""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(str(template_dir), trust_remote_code=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)
    sd = model.state_dict()
    exp = {k: tuple(v.shape) for k, v in sd.items()}

    # tie_word_embeddings=True 时 lm_head.weight 与 embed_tokens 共享存储，
    # 单文件里通常不存它 —— 交给 transformers 加载时自动 tie，这里不当作缺失。
    if getattr(cfg, "tie_word_embeddings", False):
        exp.pop("lm_head.weight", None)
    return exp


def _pick_transform(src_keys: set, expected: dict):
    """在候选统一变换里挑出能覆盖全部期望键的那个。挑不出就抛错并打差异。"""
    best = None
    report = []
    for desc, fn in _PREFIX_CANDIDATES:
        mapped = {}
        collision = False
        for k in src_keys:
            nk = fn(k)
            if nk in mapped:
                collision = True
                break
            mapped[nk] = k
        if collision:
            report.append(f"  - {desc}: 变换后有键冲突，跳过")
            continue
        missing = set(expected) - set(mapped)
        report.append(f"  - {desc}: 覆盖 {len(expected) - len(missing)}/{len(expected)}")
        if not missing:
            best = (desc, mapped)
            break
    if best is None:
        raise SystemExit(
            "✗ 没有任何统一前缀变换能让源文件覆盖全部期望键。\n"
            "各候选的覆盖情况：\n" + "\n".join(report) +
            "\n\n源文件键样本（前 10）：\n  " + "\n  ".join(sorted(src_keys)[:10]) +
            "\n期望键样本（前 10）：\n  " + "\n  ".join(sorted(expected)[:10]) +
            "\n\n这说明源文件不是本 config 对应的模型，或键名规则超出已知几种。"
            "把上面两组样本贴出来再加一条变换规则，不要靠模糊匹配硬配。"
        )
    return best


def _stream_save(out_path: Path, plan: list, src_file, target_dtype) -> None:
    """按 safetensors 格式流式写出。plan = [(hf_key, src_key, shape, torch_dtype)]。

    格式：8 字节小端 header 长度 + header(JSON, UTF-8) + 各张量原始字节（按 header 里
    的 data_offsets 顺序紧密排列）。逐张量读→转 dtype→写，峰值内存 ≈ 最大单个张量。
    """
    import torch

    header = {}
    offset = 0
    for hf_key, _src_key, shape, dt in plan:
        nbytes = 1
        for s in shape:
            nbytes *= int(s)
        nbytes *= torch.empty((), dtype=dt).element_size()
        header[hf_key] = {
            "dtype": _DT2ST[str(dt)],
            "shape": [int(s) for s in shape],
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    header["__metadata__"] = {"format": "pt",
                              "converted_by": "anima tools/convert_comfy_te_to_hf.py"}

    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (-len(blob)) % 8                      # header 对齐到 8 字节边界
    blob += b" " * pad

    tmp = out_path.with_suffix(out_path.suffix + ".part")
    written = 0
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for hf_key, src_key, shape, dt in plan:
            t = src_file.get_tensor(src_key)
            if t.dtype != dt:
                t = t.to(dt)
            raw = t.contiguous().flatten().view(torch.uint8).numpy().tobytes()
            f.write(raw)
            written += len(raw)
            del t, raw
            print(f"    写入 {hf_key:60s} {tuple(shape)}", flush=True)
    if written != offset:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"✗ 写出字节数 {written} != header 声明 {offset}，已丢弃产物")
    os.replace(tmp, out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description="ComfyUI 单文件 TE → HF 目录")
    ap.add_argument("--src", required=True, help="ComfyUI 单文件 .safetensors")
    ap.add_argument("--out", required=True, help="输出的 HF 目录")
    ap.add_argument("--template", default=str(DEFAULT_TEMPLATE),
                    help="提供 config.json / tokenizer 的模板目录")
    ap.add_argument("--dtype", default="keep",
                    choices=["keep", "bf16", "fp16", "fp32"],
                    help="权重 dtype（默认保持源文件不变）")
    ap.add_argument("--dry-run", action="store_true", help="只校验键名与形状，不写盘")
    args = ap.parse_args()

    import torch
    from safetensors import safe_open

    src = Path(args.src)
    out = Path(args.out)
    template = Path(args.template)
    if not src.is_file():
        raise SystemExit(f"✗ 源文件不存在：{src}")
    if not (template / "config.json").is_file():
        raise SystemExit(f"✗ 模板目录里没有 config.json：{template}")

    print(f"源文件   : {src}  ({src.stat().st_size / 2**30:.2f} GiB)")
    print(f"模板目录 : {template}")
    print(f"输出目录 : {out}")

    expected = _expected_keys(template)
    print(f"\n期望键数 : {len(expected)}（由 {template.name}/config.json 反推）")

    with safe_open(str(src), framework="pt") as f:
        src_keys = set(f.keys())
        print(f"源文件键数: {len(src_keys)}")

        desc, mapped = _pick_transform(src_keys, expected)
        print(f"\n✓ 键名变换 : {desc}")

        # 形状逐个校验
        bad = []
        for hf_key, shape in sorted(expected.items()):
            got = f.get_slice(mapped[hf_key]).get_shape()
            if tuple(got) != tuple(shape):
                bad.append(f"  {hf_key}: 源 {tuple(got)} != 期望 {tuple(shape)}")
        if bad:
            raise SystemExit("✗ 形状不匹配：\n" + "\n".join(bad))
        print(f"✓ 形状校验 : {len(expected)}/{len(expected)} 全部一致")

        extra = set(mapped) - set(expected)
        if extra:
            print(f"ℹ 源文件多出 {len(extra)} 个键（不写入），例如："
                  f" {sorted(extra)[:3]}")

        want = {"keep": None, "bf16": torch.bfloat16,
                "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
        plan = []
        for hf_key, shape in sorted(expected.items()):
            src_key = mapped[hf_key]
            dt = _st_to_torch(f.get_slice(src_key).get_dtype())
            plan.append((hf_key, src_key, shape, want or dt))

        total = sum(
            torch.empty((), dtype=dt).element_size() * int(torch.tensor(sh).prod())
            for _, _, sh, dt in plan)
        print(f"✓ 将写出   : {len(plan)} 个张量，{total / 2**30:.2f} GiB")

        if args.dry_run:
            print("\n--dry-run：键名与形状都对得上，未写盘。去掉 --dry-run 即可产出。")
            return 0

        out.mkdir(parents=True, exist_ok=True)
        print("\n写入权重（流式，峰值内存 ≈ 最大单个张量）：")
        _stream_save(out / "model.safetensors", plan, f, want)

    # config / tokenizer 一并拷过去，产出一个自足的 HF 目录
    copied = []
    for name in _TEMPLATE_FILES:
        p = template / name
        if p.is_file() and p.stat().st_size > 1024:   # 跳过 LFS 指针那种小文件
            shutil.copy2(p, out / name)
            copied.append(name)
        elif p.is_file():
            shutil.copy2(p, out / name)               # config.json 本来就小
            copied.append(name)
    print(f"\n✓ 已拷贝配置/分词器: {copied}")

    print(f"\n完成。自检：\n"
          f"  python -c \"from transformers import AutoModelForCausalLM as M;"
          f" m=M.from_pretrained(r'{out}'); print(sum(p.numel() for p in m.parameters()))\"\n"
          f"然后把 yaml 的 text_encoder_path 指到 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
