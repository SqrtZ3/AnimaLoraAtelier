#!/usr/bin/env python
"""DCU 镜像自检：构建期当硬门槛用，运行期当验收用。

★ 为什么是独立脚本而不是 Dockerfile 里的 heredoc：
  scnet 的镜像构建用的是 **docker legacy builder**，它把 `RUN` 的多行命令压成一行，
  heredoc 当场失效 ——

      Step 19/22 : RUN set -e;     python - <<'PY'
       ---> /bin/bash: line 1: warning: here-document at line 1 delimited by
            end-of-file (wanted `PY')

  校验一行都没跑，这一步却"成功"了。写成文件 + `python tools/dcu_image_selfcheck.py`
  就没有这个问题。

★ 构建期必须先 `. /opt/dtk/env.sh`：DTK 的环境是基础镜像的 `docker-entrypoint.sh`
  在**容器启动时**设的，`RUN` 不走 entrypoint，所以构建期裸 `import torch` 必然报
  `librocm_smi64.so.2` / `libgalaxyhip.so.5` 找不到。

用法：
    . /opt/dtk/env.sh && python tools/dcu_image_selfcheck.py            # 构建期
    . /opt/dtk/env.sh && python tools/dcu_image_selfcheck.py --runtime  # 进容器后验收

退出码非 0 即失败（构建期会让 build 挂掉，这正是想要的）。
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys

# 训练必需（anima_train.py:ensure_dependencies 的硬门槛）
REQUIRED = ["numpy", "PIL", "safetensors", "transformers", "einops", "sentencepiece", "yaml"]
# 本镜像额外装的两类：
#  - 强烈建议：EXPECTED
#  - DCU 注意力后端（DAS 预编译轮子，PyPI 无 DCU 版）：flash_attn / triton ——
#    flash_attn 激活 DAS torch 的 SDPA flash 后端；triton 是 flash_attn 的 import 硬依赖
#    （flash_attn/utils/sparse_utils.py 模块级 import），也是 torch.compile / FlexAttention 的前提
REQUIRED += ["flash_attn", "triton"]
EXPECTED = ["rich", "omegaconf", "wandb", "accelerate"]
# 平台要求，缺了实例起不来（见《构建镜像规则》一、1/2）
PLATFORM_BINS = ["/usr/sbin/sshd", "/usr/bin/sudo", "/opt/conda/bin/jupyter"]

fails: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'✓' if ok else '✗'} {label}{(' — ' + detail) if detail else ''}")
    if not ok:
        fails.append(label)


def main() -> int:
    ap = argparse.ArgumentParser(description="DCU 镜像自检")
    ap.add_argument("--runtime", action="store_true",
                    help="运行期模式：额外验证设备真的可见、SDPA 修复真的生效")
    args = ap.parse_args()

    print("=" * 70)
    print("DCU 镜像自检" + ("（运行期）" if args.runtime else "（构建期）"))
    print("=" * 70)

    # --- 1) torch 必须还是 DTK 的 das 构建 ---
    try:
        import torch
    except Exception as exc:
        check("import torch", False, f"{type(exc).__name__}: {exc}")
        print("\n  构建期报 librocm_smi64.so.2 / libgalaxyhip.so.5 找不到 = 忘了先"
              " `. /opt/dtk/env.sh`（RUN 不走 entrypoint）。")
        return 1
    hip = getattr(torch.version, "hip", None)
    cuda = getattr(torch.version, "cuda", None)
    check("torch 是 HIP 构建", bool(hip) and not cuda,
          f"{torch.__version__} hip={hip} cuda={cuda}")
    if cuda or not hip:
        print("     → das 版被 PyPI 的 CUDA 版覆盖了。用 `pip install --dry-run -v <包>` "
              "定位是哪个包拉的 torch，加进 requirements-dcu.txt 的排除名单。")

    # --- 2) numpy 没被顶到 2.x（das torch 是按 1.x ABI 编的，未做对拍）---
    try:
        import numpy
        check("numpy 仍是 1.x", numpy.__version__.startswith("1."), numpy.__version__)
    except Exception as exc:
        check("import numpy", False, f"{type(exc).__name__}: {exc}")

    # --- 3) 训练依赖 ---
    missing = []
    for mod in REQUIRED:
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(mod)
    check("训练必需依赖齐全", not missing, "缺 " + ", ".join(missing) if missing
          else f"{len(REQUIRED)}/{len(REQUIRED)}")

    soft = []
    for mod in EXPECTED:
        try:
            importlib.import_module(mod)
        except Exception:
            soft.append(mod)
    if soft:
        print(f"  · 可选依赖缺 {', '.join(soft)}（不阻断）")

    # --- 4) 平台必需组件（缺了创建实例会失败）---
    lost = [p for p in PLATFORM_BINS if not os.path.exists(p)]
    check("平台必需组件齐全", not lost, "缺 " + ", ".join(lost) if lost else "sshd/sudo/jupyter")

    # --- 5) 环境脚本进镜像了 ---
    check("/etc/profile.d/zz-scnet-env.sh 已安装",
          os.path.exists("/etc/profile.d/zz-scnet-env.sh"))

    # --- 6) 运行期才有意义的两项 ---
    if args.runtime:
        avail = torch.cuda.is_available()
        check("DCU 可见", avail,
              f"{torch.cuda.device_count()} 卡 {torch.cuda.get_device_name(0)}" if avail else "无")
        if avail:
            sys.path.insert(0, "/opt/anima-lora-train/AnimaLoraToolkit")
            try:
                from utils import dcu_compat
                info = dcu_compat.enable()
                import torch.nn.functional as F
                q = torch.randn(1, 2, 64, 32, device="cuda", dtype=torch.bfloat16)
                F.scaled_dot_product_attention(q, q, q)
                check("无 mask SDPA 可用", True)
            except Exception as exc:
                check("无 mask SDPA 可用", False, f"{type(exc).__name__}: {exc}")
            # ★ flash 后端必须激活：DAS 版 torch 把无 mask SDPA 派发给外部 flash-attn
            #   动态库（缺失时报 `No matching libraries found for flash_attn_2_cuda*.so`，
            #   dcu_compat 会实测后关掉 flash 退回 math —— 那是"能用但 O(S²)"）。
            #   镜像里带 DAS flash_attn 轮子后，configure_sdpa_backends 应什么都不关
            #   （flash 保持开），SDPA 峰值显存 O(S) 级（详见 docs/dcu-attn-backend-research.md）。
            sdpa_state = info.get("sdpa", {})
            flash_on = bool(sdpa_state.get("flash")) and not sdpa_state.get("disabled_by_anima")
            check("SDPA flash 后端激活（DAS flash_attn 轮子生效）", flash_on,
                  f"flash={sdpa_state.get('flash')} 关掉了 {sdpa_state.get('disabled_by_anima')}")

    print("-" * 70)
    if fails:
        print("✗ 自检未通过：" + "；".join(fails))
        return 1
    print("✓ 自检全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
