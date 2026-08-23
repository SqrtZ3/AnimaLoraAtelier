"""定位 **upstream**（父仓库 `AnimaLoraToolkit/`）—— 只给 torch 侧 dump 闸门用。

## 为什么需要这个

本仓库（TPU 路线）对父仓库**零运行时依赖**：`jax_tpu/*.py` 不 import 任何
`trainer.*` / `models.*`，训练、打包、导出、plan-only 全都自足。

但有一类闸门天然跨仓库：**parity 对拍**。它们要证明"JAX 实现 ≡ PyTorch 实现"，
所以必须能同时看到两边。这类闸门是**两步式**的：

    <torch-python>  dump_*.py            # 在 upstream 环境跑，产参考量 npz
    <jax-python>    check_*.py           # 在本仓库跑，读 npz 判等

第二步（`check_*`）只读 npz，**在本仓库独立可跑**；只有第一步需要 upstream。
参考量一旦产出就可长期复用（小的几份已随仓库分发，见 `tests/README.md`）。

## 用法

    export ANIMA_UPSTREAM=/path/to/anima-lora-train/AnimaLoraToolkit
    <torch-python> dump_adapter_ref.py

没设环境变量时会 fail-fast 并说清要什么 —— 而不是抛一个
`ModuleNotFoundError: No module named 'trainer'` 让人猜。

同目录并存时（本仓库是父仓库的子目录，开发期常见）会自动探到，不必设环境变量。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: upstream 的判据：这几个相对路径都在，才认。任挑一个都可能误命中同名目录。
_MARKERS = ("trainer/objective.py", "models/anima_modeling.py", "trainer/data.py")


def _looks_like_upstream(p: Path) -> bool:
    return p.is_dir() and all((p / m).exists() for m in _MARKERS)


def upstream(what: str) -> Path:
    """返回 upstream `AnimaLoraToolkit/` 的路径，并把它插进 `sys.path`。

    `what` 只用于报错文案，写清本闸门要 upstream 的哪个东西（如
    "trainer/objective.py 的 Flow Matching 参考实现"）。
    """
    env = os.environ.get("ANIMA_UPSTREAM", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if not _looks_like_upstream(p):
            raise SystemExit(
                f"[ FATAL ] ANIMA_UPSTREAM={env!r} 不像 AnimaLoraToolkit 目录\n"
                f"  判据：{' / '.join(_MARKERS)} 三者都要在。\n"
                f"  它该指向父仓库里的 `AnimaLoraToolkit`（**不是**仓库根）。")
    else:
        # 开发期本仓库常作为父仓库的子目录并存 —— 往上找，省掉设环境变量。
        here = Path(__file__).resolve()
        for anc in here.parents:
            for cand in (anc / "AnimaLoraToolkit", anc):
                if _looks_like_upstream(cand):
                    p = cand
                    break
            else:
                continue
            break
        else:
            raise SystemExit(
                f"[ FATAL ] 本闸门是 **upstream parity 对拍**，需要父仓库的 {what}。\n"
                f"  本仓库（TPU 路线）对父仓库零运行时依赖 —— 训练/打包/导出/plan-only\n"
                f"  都不需要它，只有这一类 `dump_*.py` 需要。\n"
                f"  设一下：\n"
                f"    export ANIMA_UPSTREAM=/path/to/anima-lora-train/AnimaLoraToolkit\n"
                f"  然后用 **torch 解释器**（不是 jax 那个）跑本脚本。\n"
                f"  哪些闸门需要 upstream、哪些参考量已随仓库分发：见 tests/README.md。")
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)
    return p
