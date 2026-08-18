#!/usr/bin/env python
"""生成 DCU 环境的 pip constraints —— 把「镜像已经装好的、绝不能被动的包」钉死。

用法：
    python tools/dcu_gen_constraints.py > constraints-dcu.txt
    pip install --prefer-binary -c constraints-dcu.txt -r requirements-dcu.txt

为什么需要它：DTK 镜像里的 torch 版本形如 ``2.9.0+das.opt2.dtk2604``。它满足
``torch>=2.0.0``，所以**正常情况下** pip 不会去动它；但只要有任何一个包写了
``torch==2.x.y`` 这类精确约束、或者你在一个没有 ``--system-site-packages`` 的
venv 里（那里根本看不见镜像的 torch），pip 就会从 PyPI 拉 CUDA 构建下来，
静默替换。钉死之后这种情况会变成一条**响亮的** ResolutionImpossible，而不是
一个能跑但 ``torch.cuda.is_available()==False`` 的环境。

numpy 也钉：das 版 torch/torchvision 是按镜像自带的 numpy ABI 编的，
被上层包顺手升到 numpy 2.x 有 ABI 断裂的风险（未实测，保守处理）。
"""
from __future__ import annotations

import importlib.metadata as md
import sys

# 顺序即输出顺序。缺失的静默跳过（镜像里不一定都有 torchaudio/triton）。
PINNED = (
    "torch",
    "torchvision",
    "torchaudio",
    "triton",
    "pytorch-triton-rocm",
    "numpy",
)


def main() -> int:
    lines = ["# 由 tools/dcu_gen_constraints.py 从当前环境生成，勿手改"]
    found = []
    for name in PINNED:
        try:
            lines.append(f"{name}=={md.version(name)}")
            found.append(name)
        except md.PackageNotFoundError:
            continue
    if "torch" not in found:
        print(
            "✗ 当前解释器里没有 torch —— 你多半在一个没有 --system-site-packages "
            "的 venv 里，镜像的 das 版 torch 被隔离掉了。\n"
            f"  当前解释器: {sys.executable}\n"
            "  见 docs/hygon-dcu.md「部署」一节。",
            file=sys.stderr,
        )
        return 1
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
