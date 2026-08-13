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

# ``expand_attn_mask()`` 允许物化的稠密 mask 上限（字节）。默认 256 MB：文本编码器那种
# 几百 KB 的 mask 随便过，DiT 侧 O(N²) 的加性 mask 会在失控前 fail-fast。见该函数 docstring。
_MASK_EXPAND_MAX_BYTES = int(os.environ.get("ANIMA_NPU_MASK_EXPAND_MAX_BYTES", str(256 * 1024 * 1024)))


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


def expand_attn_mask(mask, q_len: int):
    """把 ``[B, H, 1, Skv]`` 形状的广播 mask 展开成 ``[B, H, Sq, Skv]``（仅 NPU 上）。

    **为什么**：PyTorch 的 SDPA 允许 attn_mask 在 query 维上广播（`Sq=1`），仓库里
    key-padding mask 就是这么造的（`models/anima_modeling.py:181` 的
    `unsqueeze(1).unsqueeze(1)`）。但昇腾把 SDPA 落到 `aclnnFlashAttentionScore`，
    它**只接受** `[B,N,Sq,Skv]` / `[B,1,Sq,Skv]` / `[1,1,Sq,Skv]` / `[Sq,Skv]`
    —— `Sq` 必须是真实的 query 长度。真机报错原文：

        get unsupported atten_mask shape, the shape is [1, 1, 1, 251].
        B=[1], N=[16], Sq=[251], Skv=[251]

    展开后语义完全相同（广播本来就是把这一维复制 Sq 份）。`.contiguous()` 是因为
    `expand` 出来的那一维 stride=0，融合算子对非连续输入的支持没有保证。

    **代价是 O(B·Sq·Skv)，必须看调用点**：

    * 文本编码器（``LLMAdapter``）：Sq=Skv≈文本长度（几百），bool → 几十~几百 KB，可忽略。
    * DiT 稠密分支（``torch_attention_op``）：mask 来自
      ``anima_modeling_core.py:_build_packed_masks``，是 **bf16 加性** mask，Skv=打包序列长度 N。
      展开成 ``[B,1,N,N]`` 的体积是 B·N²·2 字节 —— N=8192 时每样本 134 MB，N=32768 时 2.1 GB。
      所以这里**不静默展开**：超过 ``_MASK_EXPAND_MAX_BYTES`` 直接抛错（fail-fast，符合本仓库
      "不静默吃掉资源" 的偏好），并在错误信息里给出替代路径。当前昇腾配置
      （``navit_packing: true``）走 ``_SegLens`` 变长分支，根本到不了这条路。

    CUDA/CPU 上直接原样返回 —— 未 `enable()` 时本函数是恒等映射，行为逐字节不变。
    """
    if not _NPU_ENABLED or mask is None:
        return mask
    import torch

    if not torch.is_tensor(mask) or mask.dim() != 4:
        return mask
    if mask.shape[-2] != 1 or int(q_len) == 1:
        return mask
    q_len = int(q_len)
    nbytes = mask.shape[0] * mask.shape[1] * q_len * mask.shape[-1] * mask.element_size()
    if nbytes > _MASK_EXPAND_MAX_BYTES:
        raise RuntimeError(
            f"昇腾 SDPA 需要把广播 attn_mask {tuple(mask.shape)} 展开成 "
            f"[{mask.shape[0]}, {mask.shape[1]}, {q_len}, {mask.shape[-1]}]"
            f"（dtype={mask.dtype}，{nbytes / 1e9:.2f} GB），超过上限 "
            f"{_MASK_EXPAND_MAX_BYTES / 1e9:.2f} GB。\n"
            "原因：昇腾把 SDPA 落到 aclnnFlashAttentionScore，它不接受 Sq=1 的广播 mask，"
            "而稠密 mask 的体积是 O(B·Sq·Skv)。\n"
            "解决：改用变长打包路径（navit_packing: true + navit_attn_backend: npu_tnd/sdpa_seg），"
            "它按段长走变长注意力、根本不物化稠密 mask；或缩小序列长度。\n"
            "确需放行可调大环境变量 ANIMA_NPU_MASK_EXPAND_MAX_BYTES（字节）。"
        )
    return mask.expand(mask.shape[0], mask.shape[1], q_len, mask.shape[-1]).contiguous()


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

    if _on("aux_perceptual_enabled"):
        # LPIPS 与 DINOv2 两条路都硬依赖 torchvision：lpips 包自身 import 它，
        # DINOv2 走 torch.hub.load（trainer/aux_losses.py:391-430）加载的仓库代码
        # （models/perceptual/hub/facebookresearch_dinov2_main/**）顶层也 import 它。
        # 而昇腾上装 torchvision 会从 PyPI 连带拉 CUDA 构建的 torch 覆盖掉 torch_npu
        # 配套版本（docs/ascend-npu.md §6「最大的环境杀手」）——这是环境级破坏，
        # 不是"少个包"，所以在这里拦住而不是等到运行时 ImportError。
        problems.append(
            "aux_perceptual_enabled（LPIPS / DINOv2 perceptual aux loss）依赖 torchvision，"
            "而昇腾上装 torchvision 会连带把 torch 换成 CUDA 构建、废掉 torch_npu 环境"
            "（docs/ascend-npu.md §6）。请设 aux_perceptual_enabled: false；"
            "确需在昇腾上用，只能先用 npu_setup_image.sh --with-perceptual"
            "（它带 constraints 钉死 torch）装好并自行验证。"
        )

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
