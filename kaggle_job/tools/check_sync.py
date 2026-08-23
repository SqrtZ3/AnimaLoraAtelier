"""闸门：`_vendor/` 的摘录是否还跟得上 upstream。

## 守什么

`tools/_vendor/` 里的代码是从 upstream（`anima-lora-train/AnimaLoraToolkit`）
照抄的。抄完之后 upstream 会继续演进 —— 本工具检测这种**漂移**：

  * 整份复制的文件：sha256 比对（最严，逐字节）；
  * 按符号摘录的：`ast.unparse` 逐符号比对（忽略空白/注释排版，只看代码结构）；
  * 已知的有意改动：走白名单，只验"改动仍然只在预期的那几处"。

漂移本身**不一定是错**。上游改注释、改日志、加类型标注都不影响缓存产物。
所以本工具的输出是"哪个符号变了"，判断由人做：

  * 只动了文档/日志 → 重抄一遍，更新 `_vendor/UPSTREAM.md` 的 sha 即可；
  * 动了数值口径（对齐单位、归一化常量、切片位置）→ **必须**重跑
    `tools/tests/check_cache_parity.py`，并考虑已有缓存要不要重新生成。

## 与 check_cache_parity 的分工

|  | 发现「上游改了」 | 发现「抄的时候漏了」 |
|---|---|---|
| `check_sync.py`（本文件） | ✅ | ❌ |
| `check_cache_parity.py` | ❌（它只比两边现在的行为） | ✅ |

两条都要有。只跑 sync 会漏掉摘录本身的错，只跑 parity 会在上游变更后
突然爆一堆看不懂的差异。

## 用法

    export ANIMA_UPSTREAM=/path/to/anima-lora-train/AnimaLoraToolkit
    python check_sync.py            # 纯 stdlib，不需要 torch / jax

退出码 0 = 无漂移；1 = 有漂移（打印清单）。
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "_vendor"

#: 整份复制的文件：本地路径 -> upstream 相对路径。
#: 判据是**归一化行尾后**的 sha256 相同 —— 本仓库统一用 LF，upstream 那几份是
#: CRLF，行尾对 Python 语义零影响，拿它报漂移只会制造噪声。内容差异照样抓得到。
WHOLE_FILES = {
    "_vendor/wan_vae.py": "models/wan/vae2_1.py",
    "kaggle_fast_upload.py": "tools/kaggle_fast_upload.py",
    "dataset_encrypt.py": "tools/dataset_encrypt.py",
}

#: 按符号摘录：本地文件 -> [(upstream 相对路径, [符号名...])]
#: 一个本地文件可以摘自多个上游文件（t5_weighted 就是）。
EXTRACTED = {
    "_vendor/latent_plan.py": [
        ("trainer/data.py",
         ["NativeFitImagePlan", "_ceil_to_multiple", "_floor_to_multiple",
          "plan_native_fit_image", "plan_multiscale_copy",
          "_tile_starts", "_blend_ramp", "tiled_vae_encode"]),
    ],
    "_vendor/st_load.py": [
        ("trainer/checkpoint.py",
         ["_strip_prefixes", "_pick_best_prefix_remap",
          "_load_safetensors_into_model"]),
        ("trainer/models.py", ["load_vae"]),
    ],
    "_vendor/llm_adapter.py": [
        ("models/anima_modeling.py",
         ["rotate_half", "apply_rotary_pos_emb", "RotaryEmbedding",
          "LLMAdapterAttention", "LLMAdapterTransformerBlock", "LLMAdapter"]),
    ],
    "_vendor/t5_weighted.py": [
        ("trainer/text_encode.py",
         ["_QWEN_CACHE", "_T5_CACHE", "_TEXT_CACHE_CAP", "_TEXT_CACHE_ENABLED",
          "set_text_encode_cache_enabled", "reset_text_encode_cache",
          "_cache_get", "_cache_put", "_parse_weighted_tag",
          "_build_qwen_text_from_prompt", "encode_qwen", "tokenize_t5_weighted"]),
        ("trainer/models.py",
         ["_QWEN_LEGACY_SUBDIR", "_resolve_qwen_dir", "load_text_encoders"]),
    ],
    "_vendor/krea2_te.py": [
        ("trainer/model_family.py",
         ["_KREA2_PROMPT_PREFIX", "_KREA2_PROMPT_SUFFIX", "_KREA2_PREFIX_IDX",
          "_KREA2_SUFFIX_START_IDX", "KREA2_SELECT_LAYERS",
          "load_krea2_text_encoder", "_KREA2_TEXT_CACHE", "_KREA2_TEXT_CACHE_CAP",
          "_KREA2_TEXT_CACHE_ENABLED", "set_krea2_text_cache",
          "reset_krea2_text_cache", "_encode_krea2_batch",
          "_encode_krea2_single", "encode_krea2_text"]),
    ],
}

#: 有意改动的符号 -> 允许的差异行数上限与说明（见 `_vendor/UPSTREAM.md`）。
#: 判据不是"随便改"，而是**改动规模不超过记录值** —— 上游改了它、本地又叠一层
#: 改动时，diff 行数会涨，于是被抓出来。
MODIFIED = {
    "load_vae": (10, "VAE 实现改成直接 import wan_vae；repo_root 变可选"),
    "LLMAdapter": (4, "删掉 npu_compat.expand_attn_mask 的 import 与两次调用"),
    "encode_krea2_text": (5, "内部 import 改指 _vendor/t5_weighted"),
}

DRIFT: list = []


def _upstream(explicit: str) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not (p / "trainer" / "data.py").exists():
            raise SystemExit(f"[ FATAL ] --upstream={explicit} 里没有 trainer/data.py")
        return p
    sys.path.insert(0, str(HERE.parent / "jax_tpu" / "tests"))
    from _upstream import upstream
    return upstream("_vendor/ 摘录的全部来源文件")


def _norm_sha(p: Path) -> str:
    """归一化行尾后的 sha256。CRLF/LF 差异不算漂移（对 Python 语义无影响），
    内容差异照样抓得到。"""
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def symbols(path: Path) -> dict:
    """文件 -> {符号名: ast.unparse(节点)}。只看顶层，摘录的都是顶层符号。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = {}
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out[n.name] = ast.unparse(n)
        elif isinstance(n, (ast.Assign, ast.AnnAssign)):
            tgs = n.targets if isinstance(n, ast.Assign) else [n.target]
            for tg in tgs:
                if isinstance(tg, ast.Name):
                    out[tg.id] = ast.unparse(n)
    return out


def diff_lines(a: str, b: str) -> int:
    import difflib
    return sum(1 for l in difflib.unified_diff(a.splitlines(), b.splitlines(),
                                               lineterm="", n=0)
               if l[:1] in "+-" and l[:3] not in ("---", "+++"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default="",
                    help="AnimaLoraToolkit 路径。缺省读 ANIMA_UPSTREAM 或自动上溯")
    ap.add_argument("-v", "--verbose", action="store_true", help="打印每个符号")
    a = ap.parse_args()

    up = _upstream(a.upstream)
    print(f"upstream = {up}\n")

    print("[1] 整份复制（sha256，行尾归一化后）")
    for local, rel in WHOLE_FILES.items():
        lp, rp = HERE / local, up / rel
        if not rp.exists():
            DRIFT.append(f"{rel} 在 upstream 里不存在了（文件被移动/删除？）")
            print(f"  [MISS] {local}  <- {rel}")
            continue
        h1, h2 = _norm_sha(lp), _norm_sha(rp)
        ok = h1 == h2
        if not ok:
            DRIFT.append(f"{local} 与 {rel} 内容不同（{h1[:12]} vs {h2[:12]}）")
        print(f"  [{'OK  ' if ok else 'DIFF'}] {local:26s} {h1[:12]}"
              + ("" if ok else f"  upstream={h2[:12]}"))

    print("\n[2] 按符号摘录（AST 逐符号）")
    for local, sources in EXTRACTED.items():
        mine = symbols(HERE / local)
        n_ok = n_mod = 0
        for rel, names in sources:
            rp = up / rel
            if not rp.exists():
                DRIFT.append(f"{rel} 在 upstream 里不存在了")
                print(f"  [MISS] {local}  <- {rel}")
                continue
            theirs = symbols(rp)
            for nm in names:
                if nm not in theirs:
                    DRIFT.append(f"{rel}::{nm} 在 upstream 里没了（改名/删除？）")
                    print(f"    [GONE] {nm}  ({rel})")
                    continue
                if nm not in mine:
                    DRIFT.append(f"{local}::{nm} 本地缺失（摘录不完整？）")
                    print(f"    [MISS] {nm}")
                    continue
                if theirs[nm] == mine[nm]:
                    n_ok += 1
                    if a.verbose:
                        print(f"    [OK  ] {nm}")
                    continue
                # 不一致：是白名单里的有意改动吗？
                if nm in MODIFIED:
                    cap, why = MODIFIED[nm]
                    d = diff_lines(theirs[nm], mine[nm])
                    if d <= cap:
                        n_mod += 1
                        if a.verbose:
                            print(f"    [MOD ] {nm}  diff {d}/{cap} 行 — {why}")
                        continue
                    DRIFT.append(
                        f"{local}::{nm} 的差异涨到 {d} 行（记录值 {cap}）—— "
                        f"预期改动是「{why}」，现在多出来的是什么？")
                    print(f"    [GROW] {nm}  diff {d} > {cap}")
                    continue
                DRIFT.append(f"{local}::{nm} 与 upstream {rel} 不一致")
                print(f"    [DIFF] {nm}  ({rel})")
        print(f"  {local:26s} 逐字 {n_ok}"
              + (f" + 有意改动 {n_mod}" if n_mod else ""))

    print("\n" + "=" * 66)
    if DRIFT:
        print(f"检出 {len(DRIFT)} 处漂移：\n")
        for m in DRIFT:
            print(f"  - {m}")
        print("\n下一步：")
        print("  1. 看 upstream 那次改动是否影响缓存产物（数值口径 vs 注释/日志）；")
        print("  2. 影响的话，重抄 + 重跑 tools/tests/check_cache_parity.py，")
        print("     并考虑已有缓存是否需要重新生成；")
        print("  3. 更新 tools/_vendor/UPSTREAM.md 的 sha256 与 commit。")
    else:
        print("无漂移：_vendor/ 与 upstream 一致。")
        print("注意：本工具只证明「两边一样」，不证明「摘录当初抄对了」——")
        print("      后者由 tools/tests/check_cache_parity.py（逐 bit）负责。")
    print("=" * 66)
    return 1 if DRIFT else 0


if __name__ == "__main__":
    raise SystemExit(main())
