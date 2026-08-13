"""昇腾 Ascend NPU（910B）兼容层 —— opt-in，默认关，CUDA 路径逐字节不变。

启用方式（二选一，等价）：
    * 环境变量 ``ANIMA_NPU=1``
    * YAML / CLI ``device_backend: npu``

设计原则
--------
1. **默认关**：``device_backend: auto``（默认值）在有 CUDA 的机器上行为与本文件加入前
   完全一致——``enable()`` 不会被调用，``autocast_device_type()`` 返回 ``"cuda"``。
2. **不静默回退**：显式要求 ``npu`` 却导不进 ``torch_npu`` / 没有可用 NPU 时直接抛错，
   而不是悄悄退回 CPU 跑一个假训练。
3. **不臆测**：本文件只做两件确定的事——(a) 调用昇腾官方提供的 ``transfer_to_npu``
   兼容补丁把 ``torch.cuda.*`` 映射到 ``torch.npu.*``；(b) 把 autocast 的
   ``device_type`` 字符串换成 ``"npu"``。其余"NPU 上某算子行不行"的问题一律交给
   ``tools/npu_probe.py`` 在真机上实测，不在这里假设。

已知约束（在 NPU 上必须关掉的功能，见 ``guard_unsupported()``）
------------------------------------------------------------
* ``xformers``：昇腾无此包 → 默认的 NaViT 块对角打包后端不可用。**但打包本身可用**：
  Anima family 有 ``navit_attn_backend: npu_tnd``（torch_npu 原生 TND 变长融合注意力）
  与 ``sdpa_seg``（逐段 dense SDPA 保底）两条数学恒等的替代路径；krea2 family 有
  ``sdpa_seg``。真机可用性由 ``tools/npu_probe.py`` 实测，不在这里假设。
* FP8/FP4 冻结底模量化（``base_quant``）：910B 无 FP8 张量核。
* ``bitsandbytes`` 8-bit 优化器：无昇腾后端。
* ``torch.compile`` / triton：昇腾上支持度未验证，默认禁止。

参考：昇腾 PyTorch 适配 ``torch_npu.contrib.transfer_to_npu``（官方迁移工具）。
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# 是否已经成功切到 NPU。模块级单例——``enable()`` 幂等。
_NPU_ENABLED = False
_NPU_INFO: dict = {}


def npu_requested(device_backend: str | None = None) -> bool:
    """是否请求了 NPU 后端（env 或配置字段任一命中）。"""
    if os.environ.get("ANIMA_NPU", "").strip().lower() in ("1", "true", "yes", "npu"):
        return True
    return (device_backend or "").strip().lower() == "npu"


def is_npu() -> bool:
    """当前进程是否已经切到 NPU。"""
    return _NPU_ENABLED


def autocast_device_type() -> str:
    """``torch.autocast(device_type=...)`` 该用的字符串。

    CUDA 机器上恒为 ``"cuda"``（与本文件加入前逐字节等价）；只有 ``enable()``
    成功后才变成 ``"npu"``。
    """
    return "npu" if _NPU_ENABLED else "cuda"


def device_str() -> str:
    """训练主设备字符串。"""
    if _NPU_ENABLED:
        return "npu"
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def enable() -> dict:
    """导入 torch_npu 并打上官方兼容补丁。失败即抛错，不静默回退。

    必须在**第一次真正使用设备之前**调用（本仓库里 = ``main()`` 里 TF32/allocator
    那一段之前）。返回一份环境信息 dict 供日志/telemetry 用。
    """
    global _NPU_ENABLED, _NPU_INFO
    if _NPU_ENABLED:
        return _NPU_INFO

    import torch

    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:  # pragma: no cover - 只在昇腾环境触发
        raise RuntimeError(
            "device_backend=npu（或 ANIMA_NPU=1）要求安装 torch_npu，但导入失败。\n"
            "请确认镜像里装了与 torch 版本匹配的 torch_npu + CANN，并且 "
            "`source /usr/local/Ascend/ascend-toolkit/set_env.sh` 已执行。\n"
            f"原始错误：{exc}"
        ) from exc

    # 昇腾官方迁移补丁：把 torch.cuda.* 映射到 torch.npu.*、'cuda' 设备串映射到 'npu'。
    # 这样仓库里既有的 torch.cuda.empty_cache/synchronize/Event/OutOfMemoryError
    # 等调用点无需逐个改写。
    # ⚠ 该补丁对 autocast 的 device_type 字符串是否覆盖，各版本行为不一致——所以
    # 本仓库的 autocast 调用点一律显式走 autocast_device_type()，不依赖这里。
    try:
        from torch_npu.contrib import transfer_to_npu  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "torch_npu 已安装但 torch_npu.contrib.transfer_to_npu 不可用；"
            "该兼容补丁是本适配的前提。请换用带完整 torch_npu 的镜像。\n"
            f"原始错误：{exc}"
        ) from exc

    npu_mod = getattr(torch, "npu", None)
    if npu_mod is None or not npu_mod.is_available():
        raise RuntimeError(
            "torch_npu 已加载，但 torch.npu.is_available() 为 False —— 容器里没有"
            "可见的昇腾设备。检查 ASCEND_RT_VISIBLE_DEVICES 与 npu-smi info。"
        )

    _NPU_ENABLED = True
    _NPU_INFO = {
        "torch": torch.__version__,
        "torch_npu": getattr(__import__("torch_npu"), "__version__", "unknown"),
        "device_count": npu_mod.device_count(),
        "device_name": _safe_device_name(),
    }
    logger.info(
        "[npu] 已切换到昇腾 NPU：torch=%s torch_npu=%s devices=%s (%s)",
        _NPU_INFO["torch"],
        _NPU_INFO["torch_npu"],
        _NPU_INFO["device_count"],
        _NPU_INFO["device_name"],
    )
    return _NPU_INFO


def _safe_device_name() -> str:
    import torch

    try:
        return torch.npu.get_device_name(0)
    except Exception:
        return "unknown"


def set_allocator_env() -> None:
    """NPU 侧 allocator 提示（对应 CUDA 的 PYTORCH_CUDA_ALLOC_CONF）。

    只在用户没显式设过时才设。expandable_segments 在昇腾上是否受支持随 torch_npu
    版本变化——设置失败不影响训练，只是少一项优化，所以这里不做断言。
    """
    if not _NPU_ENABLED:
        return
    if "PYTORCH_NPU_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
        logger.info("[npu] PYTORCH_NPU_ALLOC_CONF=expandable_segments:True")


# ── 不支持功能的 fail-fast 守卫 ────────────────────────────────────────────────
# 首版刻意收窄变量面：这些开关在 NPU 上要么无后端、要么未验证，构造期直接报错并
# 指出替代路径，而不是跑到第 300 步才崩。

def guard_unsupported(args) -> None:
    """在 NPU 上逐条检查已知不支持的开关，命中即抛 ValueError（含替代方案）。"""
    if not _NPU_ENABLED:
        return

    def _on(name: str) -> bool:
        v = getattr(args, name, None)
        return bool(v) and str(v).lower() not in ("none", "off", "false", "0")

    problems: list[str] = []

    if str(getattr(args, "base_quant", "none") or "none").lower() != "none":
        problems.append(
            "base_quant（FP8/FP4 冻结底模量化）：910B 没有 FP8 张量核，"
            "该路径在昇腾上无对应实现。请设 base_quant: none。"
        )

    if _on("navit_packing"):
        family = str(getattr(args, "model_family", "anima") or "anima").lower()
        backend = str(getattr(args, "navit_attn_backend", "xformers") or "xformers").lower()
        # 昇腾没有 xformers，但块对角打包本身是可行的——两条等价路径见
        # models/anima_modeling_core.py 顶部的后端说明。只拦真正依赖 xformers 的取值。
        allowed = ("sdpa_seg",) if family == "krea2" else ("sdpa_seg", "npu_tnd")
        if backend not in allowed:
            problems.append(
                f"navit_packing + navit_attn_backend={backend} 依赖 xformers（昇腾无此包）。"
                f"{family} family 在昇腾上可选 {allowed}："
                "npu_tnd = torch_npu.npu_fusion_attention 的 TND 变长融合注意力"
                "（昇腾原生，先用 tools/npu_probe.py 确认真机可用）；"
                "sdpa_seg = 逐段 dense SDPA 保底路径（不依赖专有算子，数学恒等，有单测对拍）。"
                "两条都不通再设 navit_packing: false 走 ARB 稠密路径。"
            )

    if _on("torch_compile"):
        problems.append("torch_compile 在昇腾上未验证，首版禁止。请设 torch_compile: false。")

    opt = str(getattr(args, "optimizer_type", "") or "").lower()
    if "8bit" in opt or "bnb" in opt:
        problems.append(
            f"optimizer_type={opt} 依赖 bitsandbytes（无昇腾后端）。"
            "请换 adamw / adamw_snr / automagic 等纯 torch 实现。"
        )

    if problems:
        raise ValueError(
            "以下配置项在昇腾 NPU 上不受支持：\n  - "
            + "\n  - ".join(problems)
        )
