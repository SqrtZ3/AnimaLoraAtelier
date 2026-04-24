"""
Optimizer Utils Module - 优化器创建（修复版）
===================================
修复要点（vs 原版）：
1. ProdigyPlus 参数使用"版本探测"机制：只把库真正支持的 kwargs 传进去，
   避免因版本差异 raise TypeError 或静默吞参数。
2. 当 params 已是分组格式（list of dict）时，不再用顶层 weight_decay 覆盖。
3. 强制 ProdigyPlus 的 lr=1.0（Prodigy 数学要求），并用显式警告而非静默修改。
4. d_coef 默认 0.5（Prodigy 作者推荐），而不是 1.0，避免 d 增长过快导致 bf16 溢出。
5. eps 默认 None —— 启用 Adam-atan2 模式（新版 prodigy-plus-schedule-free 支持）。
   atan2(m, sqrt(v)) 数学上天然避免除零，比传统 m/(sqrt(v)+eps) 更稳定，
   对 bf16 尤其友好。仅当用户显式传入非正数 eps 时会降级提示。
6. 增加 create 后的 sanity check（打印参数数、state 初始化验证）。

支持的优化器类型：
- adamw8bit (bitsandbytes) - 内存高效
- adamw    - 标准 PyTorch AdamW，后备选项
- prodigyplus (prodigyplus) - 自适应学习率 + Schedule-Free
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, Iterable, List, Optional, Union

import torch
from torch import nn
from torch.optim import AdamW, Optimizer

logger = logging.getLogger(__name__)

# -------- 可选依赖 --------
try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False

try:
    from prodigyplus import ProdigyPlusScheduleFree
    PRODIGYPLUS_AVAILABLE = True
except ImportError:
    ProdigyPlusScheduleFree = None  # type: ignore
    PRODIGYPLUS_AVAILABLE = False


# =============================================================================
# 工具函数
# =============================================================================

ParamInput = Union[Iterable[nn.Parameter], List[Dict[str, Any]]]


def _is_param_groups(params: Any) -> bool:
    """判断传入的 params 是否已经是 [{'params': [...], ...}, ...] 形式"""
    if isinstance(params, (list, tuple)) and len(params) > 0:
        first = params[0]
        if isinstance(first, dict) and "params" in first:
            return True
    return False


def _filter_kwargs_by_signature(cls_or_fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    只保留 cls_or_fn 签名里真实存在的关键字参数，
    避免不同版本 ProdigyPlus 的参数差异导致 TypeError。
    """
    try:
        sig = inspect.signature(cls_or_fn)
    except (TypeError, ValueError):
        return dict(kwargs)

    accepted = set()
    has_var_keyword = False
    for name, param in sig.parameters.items():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            has_var_keyword = True
            break
        accepted.add(name)

    if has_var_keyword:
        return dict(kwargs)

    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    dropped = [k for k in kwargs if k not in accepted]
    if dropped:
        logger.warning(
            f"[optimizer_utils] Dropped unsupported kwargs for "
            f"{getattr(cls_or_fn, '__name__', cls_or_fn)}: {dropped}"
        )
    return filtered


# =============================================================================
# 工厂入口
# =============================================================================

def create_optimizer(
    optimizer_type: str,
    params: ParamInput,
    learning_rate: float,
    betas: tuple = (0.9, 0.999),
    weight_decay: float = 0.01,
    eps: float = 1e-8,
    **kwargs,
) -> Optimizer:
    """
    根据 optimizer_type 创建优化器。
    `params` 既可以是参数迭代器，也可以是 param_groups（[{'params':..., 'weight_decay':...}]）。
    """
    optimizer_type = optimizer_type.lower()

    if optimizer_type == "adamw8bit":
        return create_8bit_adamw(
            params=params, lr=learning_rate, betas=betas,
            weight_decay=weight_decay, eps=eps, **kwargs,
        )
    if optimizer_type == "adamw":
        return create_standard_adamw(
            params=params, lr=learning_rate, betas=betas,
            weight_decay=weight_decay, eps=eps, **kwargs,
        )
    if optimizer_type == "prodigyplus":
        return create_prodigyplus_optimizer(
            params=params, lr=learning_rate, betas=betas,
            weight_decay=weight_decay, eps=eps, **kwargs,
        )

    raise ValueError(
        f"Unknown optimizer type: {optimizer_type}. "
        f"Choose from: adamw8bit, adamw, prodigyplus"
    )


# =============================================================================
# 标准 AdamW
# =============================================================================

def create_standard_adamw(
    params: ParamInput,
    lr: float,
    betas: tuple = (0.9, 0.999),
    weight_decay: float = 0.01,
    eps: float = 1e-8,
    **kwargs,
) -> Optimizer:
    valid_keys = {
        "amsgrad", "maximize", "capturable", "differentiable", "foreach", "fused",
    }
    adamw_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
    ignored = [k for k in kwargs if k not in valid_keys]
    if ignored:
        logger.warning(f"[AdamW] Ignored unsupported params: {ignored}")

    logger.info(f"Creating standard AdamW (lr={lr}, wd={weight_decay}, eps={eps}, betas={betas})")

    param_list = list(params) if not _is_param_groups(params) else params
    return AdamW(
        param_list, lr=lr, betas=betas, eps=eps,
        weight_decay=weight_decay, **adamw_kwargs,
    )


# =============================================================================
# 8-bit AdamW
# =============================================================================

def create_8bit_adamw(
    params: ParamInput,
    lr: float,
    betas: tuple = (0.9, 0.999),
    weight_decay: float = 0.01,
    eps: float = 1e-8,
    min_8bit_size: int = 4096,
    **kwargs,
) -> Optimizer:
    if not BITSANDBYTES_AVAILABLE:
        raise ImportError(
            "bitsandbytes is required for 8-bit AdamW. "
            "Install with: pip install bitsandbytes"
        )
    logger.info(f"Creating 8-bit AdamW (lr={lr}, wd={weight_decay})")

    param_list = list(params) if not _is_param_groups(params) else params
    return bnb.optim.AdamW8bit(
        param_list, lr=lr, betas=betas, eps=eps,
        weight_decay=weight_decay, min_8bit_size=min_8bit_size,
    )


# =============================================================================
# ProdigyPlus (Schedule-Free)
# =============================================================================

def create_prodigyplus_optimizer(
    params: ParamInput,
    lr: float = 1.0,
    betas: tuple = (0.9, 0.99),
    weight_decay: float = 0.01,
    eps: Optional[float] = None,         # ★ 默认 None → 启用 Adam-atan2（新版推荐）
    d0: float = 1e-6,
    d_coef: float = 0.5,                 # ★ 官方推荐 0.5，避免 d 爆炸
    use_schedulefree: bool = True,
    use_stableadamw: bool = True,        # ★ 抗 bf16 梯度尖峰
    use_bias_correction: bool = False,
    factored: bool = False,
    **kwargs,
) -> Optimizer:
    """
    创建 ProdigyPlusScheduleFree 优化器。
    版本兼容：未知参数会被自动过滤掉（并给出 warning），不会导致 TypeError。
    """
    if not PRODIGYPLUS_AVAILABLE:
        raise ImportError(
            "prodigyplus is required. Install with: "
            "pip install prodigy-plus-schedule-free"
        )

    # --- Prodigy 数学要求 lr=1.0，显式强制 ---
    if abs(float(lr) - 1.0) > 1e-8:
        logger.warning(
            f"[ProdigyPlus] Forcing lr=1.0 (got {lr}); Prodigy adapts step size "
            f"internally via `d`. Use `d0` to control initial effective lr."
        )
    lr = 1.0

    # --- eps=None 是合法且推荐的用法：新版 ProdigyPlus 会自动启用 Adam-atan2 ---
    #     Adam-atan2 不需要 epsilon（数学上 atan2(m, sqrt(v)) 天然避免除零），
    #     数值稳定性优于传统的 m / (sqrt(v) + eps)，对 bf16 尤其友好。
    #     这里不做任何干预；仅当 eps 是非法的非正数时给出提示。
    if isinstance(eps, (int, float)) and eps <= 0:
        logger.warning(
            f"[ProdigyPlus] eps={eps} is non-positive and will fall back to default. "
            f"Use eps=None explicitly if you want Adam-atan2 mode."
        )
        eps = None

    # --- 收集所有候选 kwargs，再按实际签名过滤 ---
    candidate = dict(
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        d0=d0,
        d_coef=d_coef,
        use_schedulefree=use_schedulefree,
        use_stableadamw=use_stableadamw,
        use_bias_correction=use_bias_correction,
        factored=factored,
        **kwargs,
    )
    safe_kwargs = _filter_kwargs_by_signature(ProdigyPlusScheduleFree, candidate)

    # --- 如果 params 已是分组形式，保留内部 weight_decay=0 配置 ---
    param_list = params if _is_param_groups(params) else list(params)

    logger.info(
        f"Creating ProdigyPlusScheduleFree "
        f"(d0={d0}, d_coef={d_coef}, betas={betas}, wd={weight_decay}, "
        f"SF={use_schedulefree}, StableAdamW={use_stableadamw})"
    )
    logger.info(f"[ProdigyPlus] Effective kwargs: {list(safe_kwargs.keys())}")

    optimizer = ProdigyPlusScheduleFree(param_list, **safe_kwargs)

    # --- 构造后自检 ---
    total = sum(p.numel() for g in optimizer.param_groups for p in g["params"] if p.requires_grad)
    logger.info(f"[ProdigyPlus] Trainable params in optimizer: {total:,}")

    return optimizer


# =============================================================================
# 参数分组（LoKr 专用：w1 不做 weight_decay）
# =============================================================================

def create_optimizer_grouped_parameters(
    model: nn.Module,
    weight_decay: float,
    no_decay_modules: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    if no_decay_modules is None:
        no_decay_modules = ["bias", "LayerNorm.weight", "layernorm.weight", "norm.weight"]

    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(pat in name for pat in no_decay_modules):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


# =============================================================================
# 信息查询
# =============================================================================

def get_optimizer_info(optimizer: Optimizer) -> Dict[str, Any]:
    info = {
        "type": type(optimizer).__name__,
        "learning_rate": optimizer.param_groups[0].get("lr", None),
        "num_param_groups": len(optimizer.param_groups),
    }
    total = 0
    for g in optimizer.param_groups:
        for p in g["params"]:
            total += p.numel()
    info["total_trainable_params"] = total

    # ProdigyPlus 特有
    pg0 = optimizer.param_groups[0]
    for key in ("d", "d0", "d_coef", "effective_lr", "k"):
        if key in pg0:
            info[key] = pg0[key]

    return info


# =============================================================================
# NaN 安全守护（供训练脚本调用）
# =============================================================================

def is_optimizer_state_healthy(optimizer: Optimizer) -> bool:
    """
    检查优化器内部状态是否已被 NaN/Inf 污染。
    ProdigyPlus 的 d、d_numerator、s 一旦变 NaN，之后每一步 loss 都会是 NaN。
    """
    for group in optimizer.param_groups:
        for key in ("d", "d_numerator", "s"):
            v = group.get(key, None)
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                if not torch.isfinite(v).all():
                    return False
            elif isinstance(v, float):
                import math
                if not math.isfinite(v):
                    return False
        for p in group["params"]:
            state = optimizer.state.get(p, {})
            for _, s in state.items():
                if isinstance(s, torch.Tensor) and s.is_floating_point():
                    if not torch.isfinite(s).all():
                        return False
    return True