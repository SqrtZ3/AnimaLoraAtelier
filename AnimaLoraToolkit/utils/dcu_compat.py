"""海光 DCU（K100-AI / 深算系列）兼容层 —— opt-in，默认关，CUDA 路径逐字节不变。

启用方式（二选一，等价）：
    * 环境变量 ``ANIMA_DCU=1``
    * YAML / CLI ``device_backend: dcu``

与昇腾适配的**根本区别**（决定了本文件为什么这么短）
--------------------------------------------------
昇腾要 ``transfer_to_npu`` 把 ``torch.cuda.*`` 重定向到 ``torch.npu.*``；DCU 不需要。
DCU 的 DTK 软件栈是 ROCm/HIP 的衍生版，海光适配版 PyTorch（``torch==2.x+das.optN.dtkXXXX``）
本身就是 **HIP 后端的 torch**：``torch.cuda.is_available()`` / ``device_count()`` /
``get_device_name()`` / ``torch.device("cuda")`` 直接就是 DCU，``torch.version.hip``
非空、``torch.version.cuda`` 为空。

所以本文件**不做设备重映射**，只做三件事：

1. **确认真的在 DCU 上**（``enable()``）——防止"以为在国产卡上跑，其实 torch 是
   CPU-only / 被 pip 覆盖成了 CUDA 构建"这种静默失败。真机踩过的坑是
   ``pip install torch`` 会把镜像里适配好的 das 版本覆盖掉。
2. **已知不支持项的构造期 fail-fast**（``guard_unsupported()``），且**尽量用实测代替假设**：
   xformers / bitsandbytes / triton 这类"有没有"的问题一律 ``import`` 一下再下结论，
   而不是写死"DCU 上没有"。
3. **环境提示**（allocator / MIOpen 缓存）。

未在本文件做任何假设的事情
--------------------------
"某个算子在 K100-AI 上行不行、快不快"一律交给 ``tools/dcu_probe.py`` 真机实测。
特别是这三条，本地无法验证、且都有前科：

* **SDPA 走哪个后端**：ROCm 上 flash/mem-efficient 后端依赖 aotriton，DTK 里有没有编进去
  未知。若只剩 math 后端，注意力就是 O(N²) 物化 → NaViT 打包（ΣN 上万）必炸显存。
* **bf16 的实际精度与吞吐**：K100-AI 标称 BF16/FP16 ~192 TFLOPS（厂商口径），实测多少要跑。
* **广播 matmul 的 backward**：昇腾上 2D×3D 广播 matmul 前向对、**反向静默算错**，
  害得 LoKr 的 w1 梯度全错（见 memory ``npu-broadcast-matmul-grad-bug`` 与
  ``tests/test_lokr_w1_grad_no_broadcast.py``）。换平台必须重做 backward 对拍，
  探针里有对应项。

平台参考：DTK ≥ 24.04 才支持 K100-AI（gfx928）；镜像来自光源 sourcefind；
``rocm-smi``（DTK 侧）/ ``hy-smi``（驱动侧）看卡；``/opt/dtk`` 是 DTK 根目录。
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_DCU_ENABLED = False
_DCU_INFO: dict = {}

# K100-AI 的 GPU arch。DTK 的 torch 若不是为 gfx928 编的，需要 HSA_OVERRIDE_GFX_VERSION=9.2.8。
_K100AI_ARCH = "gfx928"


def dcu_requested(device_backend: str | None = None) -> bool:
    """是否请求了 DCU 后端（env 或配置字段任一命中）。"""
    if os.environ.get("ANIMA_DCU", "").strip().lower() in ("1", "true", "yes", "dcu"):
        return True
    return (device_backend or "").strip().lower() in ("dcu", "hygon", "rocm")


def is_dcu() -> bool:
    """当前进程是否已确认跑在 DCU 上。"""
    return _DCU_ENABLED


def device_str() -> str:
    """训练主设备字符串。

    DCU 上**就是** ``"cuda"`` —— HIP 后端复用了 CUDA 的设备命名空间，写 ``"hip"`` 反而报错。
    这不是"没适配"，是 DTK 的既定约定。
    """
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def autocast_device_type() -> str:
    """``torch.autocast(device_type=...)`` 该用的字符串。DCU 上同样是 ``"cuda"``。"""
    return "cuda"


def _torch_flavor(torch) -> str:
    """判定当前 torch 是 HIP 构建、CUDA 构建还是 CPU-only。"""
    if getattr(torch.version, "hip", None):
        return "hip"
    if getattr(torch.version, "cuda", None):
        return "cuda"
    return "cpu"


def _arch_name(torch) -> str:
    """取设备 0 的 gcnArchName（如 ``gfx928:sramecc+:xnack-``），取不到返回空串。"""
    try:
        return str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", "") or "")
    except Exception:
        return ""


def enable() -> dict:
    """确认运行环境确实是 DCU。失败即抛错，不静默回退。

    必须在**第一次真正使用设备之前**调用。返回一份环境信息 dict 供日志/telemetry 用。
    """
    global _DCU_ENABLED, _DCU_INFO
    if _DCU_ENABLED:
        return _DCU_INFO

    import torch

    flavor = _torch_flavor(torch)
    if flavor != "hip":
        raise RuntimeError(
            f"device_backend=dcu（或 ANIMA_DCU=1）要求海光适配版 PyTorch（HIP 后端），"
            f"但当前 torch {torch.__version__} 是 {flavor} 构建"
            f"（torch.version.hip={getattr(torch.version, 'hip', None)!r}）。\n"
            "最常见原因：在 DTK 镜像里执行过 `pip install torch`，把适配版覆盖成了 PyPI 的\n"
            "CUDA/CPU 构建。修复：重建容器，或从光源（sourcefind）重装 torch-*+das*.dtk* 轮子；\n"
            "**不要**用 pip 从 PyPI 装 torch/torchvision。"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "torch 是 HIP 构建，但 torch.cuda.is_available() 为 False —— 容器里没有可见的 DCU。\n"
            "检查：(1) `rocm-smi` / `hy-smi` 能否看到卡；(2) 环境变量 HIP_VISIBLE_DEVICES /\n"
            "ROCR_VISIBLE_DEVICES 是否把卡屏蔽了；(3) /opt/dtk 的环境变量是否 source 过\n"
            "（LD_LIBRARY_PATH 缺 /opt/dtk/lib 会表现为找不到 libhip*.so）。"
        )

    arch = _arch_name(torch)
    _DCU_ENABLED = True
    _DCU_INFO = {
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "device_count": torch.cuda.device_count(),
        "device_name": _safe_device_name(),
        "arch": arch,
        "dtk_root": os.environ.get("DTK_HOME") or ("/opt/dtk" if os.path.isdir("/opt/dtk") else ""),
    }
    logger.info(
        "[dcu] 已确认海光 DCU：torch=%s hip=%s devices=%s (%s, arch=%s)",
        _DCU_INFO["torch"], _DCU_INFO["hip"], _DCU_INFO["device_count"],
        _DCU_INFO["device_name"], arch or "未知",
    )
    if arch and _K100AI_ARCH not in arch:
        # 不是错误——DCU 有 Z100/K100/K100-AI 多个型号，只是提醒别把别的型号的实测结论套过来。
        logger.info(
            "[dcu] 注意：设备 arch=%s，不是 K100-AI 的 %s。docs/hygon-dcu.md 里的实测数字"
            "是在 K100-AI 上取的，换型号需重跑 tools/dcu_probe.py。",
            arch, _K100AI_ARCH,
        )
    _DCU_INFO["sdpa"] = configure_sdpa_backends()
    return _DCU_INFO


def configure_sdpa_backends() -> dict:
    """把**实际不可用**的 SDPA 后端关掉，让 dispatcher 落到能跑的那个。

    ★ 这不是性能调优，是可用性修复。【实测 · scnet BW(gfx936) / torch 2.9.0+das.dtk2604】：

        F.scaled_dot_product_attention(q, k, v)        # 不给 mask
        RuntimeError: No matching libraries found for flash_attn_2_cuda*.so

    DTK 的 torch 把 **无 mask 的 SDPA** 派发给一个外部的 flash-attn 动态库，而基础镜像里
    根本没有那个 .so（``find / -name '*flash_attn*'`` 为空）。它不会优雅回退到 math，
    而是直接抛 RuntimeError —— 于是「给了 mask 就能跑、不给 mask 就崩」，
    ``sdpa_seg``（NaViT 主路径）整条挂掉。

    ``torch.backends.cuda.enable_flash_sdp(False)`` 一行即可绕开：实测关掉之后
    同一个调用立刻通过（fwd+bwd 都对）。

    做法上**不写死"DCU 没有 flash"** —— 光源在持续补 flash-attn 的 das 版轮子，装上以后
    这条路是通的。所以这里实测一次：能跑就什么都不动，跑不通才逐个关。

    返回一份 dict 记录最终各后端的开关状态，供日志/telemetry 用。
    """
    import torch
    import torch.nn.functional as F

    backends = getattr(torch.backends, "cuda", None)
    if backends is None or not hasattr(backends, "enable_flash_sdp"):
        return {}

    def _probe() -> str:
        """跑一次无 mask 的小 SDPA，返回 "" 表示通过，否则返回错误摘要。"""
        try:
            q = torch.randn(1, 2, 64, 32, device="cuda", dtype=torch.bfloat16)
            F.scaled_dot_product_attention(q, q, q)
            return ""
        except Exception as exc:  # noqa: BLE001 —— 任何异常都算"这条路不通"
            return f"{type(exc).__name__}: {exc}"

    disabled = []
    err = _probe()
    if err and backends.flash_sdp_enabled():
        backends.enable_flash_sdp(False)
        disabled.append("flash")
        logger.warning("[dcu] 无 mask SDPA 失败（%s），已关闭 flash 后端后重试", err[:120])
        err = _probe()
    if err and backends.mem_efficient_sdp_enabled():
        backends.enable_mem_efficient_sdp(False)
        disabled.append("mem_efficient")
        logger.warning("[dcu] 仍失败（%s），已关闭 mem_efficient 后端后重试", err[:120])
        err = _probe()

    state = {
        "flash": bool(backends.flash_sdp_enabled()),
        "mem_efficient": bool(backends.mem_efficient_sdp_enabled()),
        "math": bool(backends.math_sdp_enabled()),
        "disabled_by_anima": disabled,
        "still_failing": err,
    }
    if err:
        # 到这一步还不通就别静默继续 —— 注意力是每一层都要走的路。
        raise RuntimeError(
            "DCU 上所有 SDPA 后端都跑不通，最后一次错误：" + err + "\n"
            "这台机器的注意力路径不可用，先跑 tools/dcu_probe.py 看「SDPA 可用后端」项。"
        )
    if disabled:
        logger.info(
            "[dcu] SDPA 后端最终状态：flash=%s mem_efficient=%s math=%s（本次关掉了 %s）"
            " → 只剩 math 时注意力会物化 O(S^2) 分数矩阵，NaViT 的 token 预算要按实测压。",
            state["flash"], state["mem_efficient"], state["math"], "/".join(disabled),
        )
    return state


def _safe_device_name() -> str:
    import torch

    try:
        return torch.cuda.get_device_name(0)
    except Exception:
        return "unknown"


def set_allocator_env() -> None:
    """DCU 侧 allocator 提示。

    ROCm 后端的 torch 读 ``PYTORCH_HIP_ALLOC_CONF``；较新的版本同时兼容
    ``PYTORCH_CUDA_ALLOC_CONF``。两个都设（只在用户没显式设过时），谁被认就是谁生效。
    ``expandable_segments`` 在 DTK 上是否受支持随版本变化，设置失败不影响训练，
    所以这里不做断言 —— 真值由 ``tools/dcu_probe.py`` 的 allocator 项实测。
    """
    if not _DCU_ENABLED:
        return
    for var in ("PYTORCH_HIP_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
        if var not in os.environ:
            os.environ[var] = "expandable_segments:True"
            logger.info("[dcu] %s=expandable_segments:True", var)


def miopen_cache_hint() -> str:
    """MIOpen 内核缓存位置提示。

    DTK 迁移的已知坑：``~/.cache/miopen`` 里的旧编译缓存会让**算子结果出错**（不是变慢），
    换 DTK 版本 / 换卡型号后尤其容易撞上。这里只返回提示串，不擅自删用户的缓存
    （删除属于难以撤销的操作）。
    """
    path = os.path.expanduser(os.environ.get("MIOPEN_USER_DB_PATH", "~/.cache/miopen"))
    return (f"MIOpen 缓存目录: {path}"
            f"（换 DTK 版本/卡型号后若出现数值异常，先 rm -rf 它再复现一次）")


# ── 不支持功能的 fail-fast 守卫 ────────────────────────────────────────────────
# 原则：**能 import 验证的就验证，不写死"DCU 上没有"**。海光的光源仓库在持续补生态，
# 写死会让本来能用的功能被永久拦住。

def _importable(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:
        return False


def guard_unsupported(args) -> None:
    """在 DCU 上逐条检查已知不支持的开关，命中即抛 ValueError（含替代路径）。"""
    if not _DCU_ENABLED:
        return

    def _on(name: str) -> bool:
        v = getattr(args, name, None)
        return bool(v) and str(v).lower() not in ("none", "off", "false", "0")

    problems: list[str] = []

    # ── FP8/FP4 冻结底模量化 ──────────────────────────────────────────────────
    # 依据：FP8 张量核在 AMD 谱系里从 CDNA3（gfx942/MI300）才有；K100-AI 是 gfx928，
    # 代际在其之前。**这是架构代际推断，不是实测**——所以留了放行开关，且探针里有实测项。
    if str(getattr(args, "base_quant", "none") or "none").lower() != "none":
        if os.environ.get("ANIMA_DCU_ALLOW_FP8", "").strip().lower() in ("1", "true", "yes"):
            logger.warning(
                "[dcu] base_quant=%s 被 ANIMA_DCU_ALLOW_FP8 放行。请确认 tools/dcu_probe.py 的"
                "「FP8 张量核」项在本机是 OK，否则会在训练中途报算子不支持。",
                getattr(args, "base_quant"),
            )
        else:
            problems.append(
                f"base_quant={getattr(args, 'base_quant')}（FP8/FP4 冻结底模量化）："
                f"K100-AI 的 arch 是 {_K100AI_ARCH}，按 AMD 谱系代际推断早于 FP8 张量核"
                "（gfx942 起）。请设 base_quant: none。\n"
                "      注：这条是**推断不是实测**。先跑 tools/dcu_probe.py，若「FP8」项实测 OK，"
                "可用 ANIMA_DCU_ALLOW_FP8=1 放行。"
            )

    # ── NaViT 打包的注意力后端 ────────────────────────────────────────────────
    if _on("navit_packing"):
        family = str(getattr(args, "model_family", "anima") or "anima").lower()
        backend = str(getattr(args, "navit_attn_backend", "xformers") or "xformers").lower()
        if backend == "npu_tnd":
            problems.append(
                "navit_attn_backend=npu_tnd 是昇腾 torch_npu 专有算子，DCU 上不存在。"
                "请改 sdpa_seg（逐段 dense SDPA，数学恒等，有单测对拍）。"
            )
        elif backend == "xformers" and not _importable("xformers.ops"):
            problems.append(
                f"navit_packing + navit_attn_backend=xformers，但本机 import xformers.ops 失败"
                f"（{family} family）。xformers 官方无 DCU/HIP 轮子。\n"
                "      请改 navit_attn_backend: sdpa_seg —— 逐段 dense SDPA，段内全注意力 ≡ "
                "块对角，数学恒等（tests/test_packed_equals_dense_forward.py 有对拍），"
                "不依赖任何专有算子。\n"
                "      ⚠ sdpa_seg 的显存/速度取决于 SDPA 在 DTK 上落到哪个后端："
                "落 math 后端会 O(段长²) 物化。先跑 tools/dcu_probe.py 的「SDPA 后端」"
                "与「sdpa_seg 显存线性性」两项再定 navit_token_budget。"
            )

    # ── torch.compile / triton ───────────────────────────────────────────────
    if _on("torch_compile") and not _importable("triton"):
        problems.append(
            "torch_compile 需要 triton，本机 import triton 失败（DTK 是否带 triton 随版本变化）。"
            "请设 torch_compile: false。"
        )

    # ── 8-bit 优化器 ──────────────────────────────────────────────────────────
    opt = str(getattr(args, "optimizer_type", "") or "").lower()
    if ("8bit" in opt or "bnb" in opt) and not _importable("bitsandbytes"):
        problems.append(
            f"optimizer_type={opt} 依赖 bitsandbytes，本机 import 失败。"
            "请换 adamw / adamw_snr / automagic 等纯 torch 实现。"
        )

    # ── perceptual aux loss ───────────────────────────────────────────────────
    # 与昇腾不同：DCU 侧海光提供适配版 torchvision，所以这里**不预先禁止**，只在真的
    # 装不上时拦。红线仍在：别从 PyPI 装 torchvision（会连带把 torch 换成 CUDA 构建）。
    if _on("aux_perceptual_enabled") and not _importable("torchvision"):
        problems.append(
            "aux_perceptual_enabled（LPIPS / DINOv2）依赖 torchvision，本机 import 失败。\n"
            "      ⚠ 不要 `pip install torchvision`：PyPI 版会连带拉 CUDA 构建的 torch，"
            "把 das 适配版覆盖掉（整个环境报废）。只能从光源装 torchvision-*+das*.dtk* 轮子。\n"
            "      否则请设 aux_perceptual_enabled: false。"
        )

    if problems:
        raise ValueError(
            "以下配置项在海光 DCU 上不受支持：\n  - " + "\n  - ".join(problems)
        )
