"""
Optimizer Utils Module - 优化器创建（修复版）
===================================
修复要点（vs 原版）：
1. ProdigyPlus 参数使用"版本探测"机制：只把库真正支持的 kwargs 传进去，
   避免因版本差异 raise TypeError 或静默吞参数。
2. 当 params 已是分组格式（list of dict）时，不再用顶层 weight_decay 覆盖。
3. 强制 ProdigyPlus 的 lr=1.0（Prodigy 数学要求），并用显式警告而非静默修改。
4. d_coef 默认 1.0：LoRA/LoKr 实战里低于 1 容易出现长期不学习；不再默认压制自适应步长。
5. eps 默认 None —— 启用 Adam-atan2 模式（新版 prodigy-plus-schedule-free 支持）。
   atan2(m, sqrt(v)) 数学上天然避免除零，比传统 m/(sqrt(v)+eps) 更稳定，
   对 bf16 尤其友好。仅当用户显式传入非正数 eps 时会降级提示。
6. 增加 create 后的 sanity check（打印参数数、state 初始化验证）。

支持的优化器类型：
- adamw8bit (bitsandbytes) - 内存高效
- adamw    - 标准 PyTorch AdamW，后备选项
- prodigyplus (prodigyplus) - 自适应学习率 + Schedule-Free
- soap     - Shampoo/Adam 矩阵预条件优化器
- soap_sf  - Schedule-Free SOAP（预条件 + Polyak 平均，无需 LR 调度）
- adopt    - ADOPT（NeurIPS 2024），diffusion 等无界梯度噪声下收敛保证的 Adam 变种
- lion     - Lion（NeurIPS 2023），sign-based 极简优化器
- clion    - Cautious Lion，Lion + 一行 mask（arxiv 2411.16085）
- emosens  - EmoSens，loss 序列驱动的动态 LR Adam-style 优化器
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, Iterable, List, Optional, Union

import torch
from torch import nn
from torch.optim import AdamW, Optimizer

logger = logging.getLogger(__name__)

try:
    from .soap_optimizer import SOAP, SOAPScheduleFree
except ImportError:  # pragma: no cover - fallback for direct script execution
    from soap_optimizer import SOAP, SOAPScheduleFree  # type: ignore

try:
    from .muon_optimizer import Muon, MuonScheduleFree
except ImportError:  # pragma: no cover
    from muon_optimizer import Muon, MuonScheduleFree  # type: ignore

try:
    from .automagic_optimizer import Automagic
except ImportError:  # pragma: no cover
    from automagic_optimizer import Automagic  # type: ignore

try:
    from .adamw_snr_optimizer import AdamWSNR
except ImportError:  # pragma: no cover
    from adamw_snr_optimizer import AdamWSNR  # type: ignore

try:
    from .adopt_optimizer import ADOPT
except ImportError:  # pragma: no cover
    from adopt_optimizer import ADOPT  # type: ignore

try:
    from .lion_optimizer import Lion
except ImportError:  # pragma: no cover
    from lion_optimizer import Lion  # type: ignore

try:
    from .emosens_optimizer import EmoSens
except ImportError:  # pragma: no cover
    from emosens_optimizer import EmoSens  # type: ignore

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


def _coerce_float(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric value for {name}: {value!r}") from exc


def _coerce_betas(value: Any) -> tuple:
    try:
        beta1, beta2 = value
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid betas value: {value!r}") from exc
    return (_coerce_float(beta1, "betas[0]"), _coerce_float(beta2, "betas[1]"))


def _normalize_param_groups_numeric_values(params: ParamInput) -> ParamInput:
    if not _is_param_groups(params):
        return params

    normalized = []
    for group in params:
        fixed = dict(group)
        if "lr" in fixed:
            fixed["lr"] = _coerce_float(fixed["lr"], "param_group.lr")
        if "weight_decay" in fixed:
            fixed["weight_decay"] = _coerce_float(
                fixed["weight_decay"], "param_group.weight_decay"
            )
        if "eps" in fixed and fixed["eps"] is not None:
            fixed["eps"] = _coerce_float(fixed["eps"], "param_group.eps")
        if "betas" in fixed:
            fixed["betas"] = _coerce_betas(fixed["betas"])
        normalized.append(fixed)
    return normalized


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
    params = _normalize_param_groups_numeric_values(params)
    learning_rate = _coerce_float(learning_rate, "learning_rate")
    betas = _coerce_betas(betas)
    weight_decay = _coerce_float(weight_decay, "weight_decay")
    eps = _coerce_float(eps, "eps") if eps is not None else None

    if optimizer_type == "adamw8bit":
        if "lr" in kwargs:
            learning_rate = _coerce_float(kwargs.pop("lr"), "optimizer_args.lr")
        if "betas" in kwargs:
            betas = _coerce_betas(kwargs.pop("betas"))
        if "weight_decay" in kwargs:
            weight_decay = _coerce_float(
                kwargs.pop("weight_decay"), "optimizer_args.weight_decay"
            )
        if "eps" in kwargs:
            eps = _coerce_float(kwargs.pop("eps"), "optimizer_args.eps")
        return create_8bit_adamw(
            params=params, lr=learning_rate, betas=betas,
            weight_decay=weight_decay, eps=eps, **kwargs,
        )
    if optimizer_type == "adamw":
        if "lr" in kwargs:
            learning_rate = _coerce_float(kwargs.pop("lr"), "optimizer_args.lr")
        if "betas" in kwargs:
            betas = _coerce_betas(kwargs.pop("betas"))
        if "weight_decay" in kwargs:
            weight_decay = _coerce_float(
                kwargs.pop("weight_decay"), "optimizer_args.weight_decay"
            )
        if "eps" in kwargs:
            eps = _coerce_float(kwargs.pop("eps"), "optimizer_args.eps")
        return create_standard_adamw(
            params=params, lr=learning_rate, betas=betas,
            weight_decay=weight_decay, eps=eps, **kwargs,
        )
    if optimizer_type == "prodigyplus":
        # `optimizer_args` may contain constructor-level keys. Pop them here so
        # YAML passthrough works without sending duplicate lr/betas/wd/eps.
        if "lr" in kwargs:
            learning_rate = _coerce_float(kwargs.pop("lr"), "optimizer_args.lr")
        if "betas" in kwargs:
            betas = _coerce_betas(kwargs.pop("betas"))
        if "weight_decay" in kwargs:
            weight_decay = _coerce_float(
                kwargs.pop("weight_decay"), "optimizer_args.weight_decay"
            )
        if "eps" in kwargs:
            eps_value = kwargs.pop("eps")
            eps = (
                _coerce_float(eps_value, "optimizer_args.eps")
                if eps_value is not None
                else None
            )
        return create_prodigyplus_optimizer(
            params=params, lr=learning_rate, betas=betas,
            weight_decay=weight_decay, eps=eps, **kwargs,
        )
    if optimizer_type == "soap":
        if "lr" in kwargs:
            learning_rate = _coerce_float(kwargs.pop("lr"), "optimizer_args.lr")
        if "betas" in kwargs:
            betas = _coerce_betas(kwargs.pop("betas"))
        elif betas == (0.9, 0.999):
            betas = (0.95, 0.95)
        if "weight_decay" in kwargs:
            weight_decay = _coerce_float(
                kwargs.pop("weight_decay"), "optimizer_args.weight_decay"
            )
        if "eps" in kwargs:
            eps = _coerce_float(kwargs.pop("eps"), "optimizer_args.eps")
        return create_soap_optimizer(
            params=params, lr=learning_rate, betas=betas,
            weight_decay=weight_decay, eps=eps, **kwargs,
        )
    if optimizer_type in {"soap_sf", "soapsf"}:
        if "lr" in kwargs:
            learning_rate = _coerce_float(kwargs.pop("lr"), "optimizer_args.lr")
        if "betas" in kwargs:
            betas = _coerce_betas(kwargs.pop("betas"))
        elif betas == (0.9, 0.999):
            # SF interpolation β1=0.9 (SF default) + SOAP's fast second moment β2=0.95.
            betas = (0.9, 0.95)
        if "weight_decay" in kwargs:
            weight_decay = _coerce_float(
                kwargs.pop("weight_decay"), "optimizer_args.weight_decay"
            )
        if "eps" in kwargs:
            eps = _coerce_float(kwargs.pop("eps"), "optimizer_args.eps")
        return create_soap_sf_optimizer(
            params=params, lr=learning_rate, betas=betas,
            weight_decay=weight_decay, eps=eps, **kwargs,
        )
    if optimizer_type == "adopt":
        if "lr" in kwargs:
            learning_rate = _coerce_float(kwargs.pop("lr"), "optimizer_args.lr")
        if "betas" in kwargs:
            betas = _coerce_betas(kwargs.pop("betas"))
        elif betas == (0.9, 0.999):
            # ADOPT recommends a larger β2 (the whole point is β2 robustness).
            betas = (0.9, 0.9999)
        if "weight_decay" in kwargs:
            weight_decay = _coerce_float(
                kwargs.pop("weight_decay"), "optimizer_args.weight_decay"
            )
        if "eps" in kwargs:
            eps = _coerce_float(kwargs.pop("eps"), "optimizer_args.eps")
        return create_adopt_optimizer(
            params=params, lr=learning_rate, betas=betas,
            weight_decay=weight_decay, eps=eps, **kwargs,
        )
    if optimizer_type in {"lion", "clion"}:
        if "lr" in kwargs:
            learning_rate = _coerce_float(kwargs.pop("lr"), "optimizer_args.lr")
        if "betas" in kwargs:
            betas = _coerce_betas(kwargs.pop("betas"))
        elif betas == (0.9, 0.999):
            betas = (0.9, 0.99)
        if "weight_decay" in kwargs:
            weight_decay = _coerce_float(
                kwargs.pop("weight_decay"), "optimizer_args.weight_decay"
            )
        kwargs.pop("eps", None)  # Lion has no eps; silently drop if passed.
        # `clion` is just lion + cautious=True; explicit kwargs win.
        if optimizer_type == "clion":
            kwargs.setdefault("cautious", True)
        return create_lion_optimizer(
            params=params, lr=learning_rate, betas=betas,
            weight_decay=weight_decay, **kwargs,
        )
    if optimizer_type == "emosens":
        if "lr" in kwargs:
            learning_rate = _coerce_float(kwargs.pop("lr"), "optimizer_args.lr")
        if "betas" in kwargs:
            betas = _coerce_betas(kwargs.pop("betas"))
        elif betas == (0.9, 0.999):
            betas = (0.9, 0.995)
        if "weight_decay" in kwargs:
            weight_decay = _coerce_float(
                kwargs.pop("weight_decay"), "optimizer_args.weight_decay"
            )
        if "eps" in kwargs:
            eps = _coerce_float(kwargs.pop("eps"), "optimizer_args.eps")
        return create_emosens_optimizer(
            params=params, lr=learning_rate, betas=betas,
            weight_decay=weight_decay, eps=eps, **kwargs,
        )

    if optimizer_type == "muon":
        if "lr" in kwargs:
            learning_rate = _coerce_float(kwargs.pop("lr"), "optimizer_args.lr")
        if "betas" in kwargs:
            betas = _coerce_betas(kwargs.pop("betas"))
        if "weight_decay" in kwargs:
            weight_decay = _coerce_float(
                kwargs.pop("weight_decay"), "optimizer_args.weight_decay"
            )
        if "eps" in kwargs:
            eps = _coerce_float(kwargs.pop("eps"), "optimizer_args.eps")
        return create_muon_optimizer(
            params=params, lr=learning_rate, betas=betas,
            weight_decay=weight_decay, eps=eps, **kwargs,
        )
    if optimizer_type in {"muon_sf", "muonsf"}:
        if "lr" in kwargs:
            learning_rate = _coerce_float(kwargs.pop("lr"), "optimizer_args.lr")
        if "betas" in kwargs:
            betas = _coerce_betas(kwargs.pop("betas"))
        elif betas == (0.9, 0.999):
            betas = (0.9, 0.95)
        if "weight_decay" in kwargs:
            weight_decay = _coerce_float(
                kwargs.pop("weight_decay"), "optimizer_args.weight_decay"
            )
        if "eps" in kwargs:
            eps = _coerce_float(kwargs.pop("eps"), "optimizer_args.eps")
        return create_muon_sf_optimizer(
            params=params, lr=learning_rate, betas=betas,
            weight_decay=weight_decay, eps=eps, **kwargs,
        )

    if optimizer_type in {"adamw_snr", "adamwsnr"}:
        # 默认 cautious=False / snr_power=1.0 时与 adamw 数学恒等（行为中立）。
        if "lr" in kwargs:
            learning_rate = _coerce_float(kwargs.pop("lr"), "optimizer_args.lr")
        if "betas" in kwargs:
            betas = _coerce_betas(kwargs.pop("betas"))
        if "weight_decay" in kwargs:
            weight_decay = _coerce_float(
                kwargs.pop("weight_decay"), "optimizer_args.weight_decay"
            )
        if "eps" in kwargs:
            eps = _coerce_float(kwargs.pop("eps"), "optimizer_args.eps")
        return create_adamw_snr_optimizer(
            params=params, lr=learning_rate, betas=betas,
            weight_decay=weight_decay, eps=eps, **kwargs,
        )

    if optimizer_type == "automagic":
        # Automagic 自己管每个权重的 lr：yaml 的 learning_rate 是**起始值**，
        # 训练强度由 optimizer_args.max_lr 决定（见 automagic_optimizer 文档）。
        if "lr" in kwargs:
            learning_rate = _coerce_float(kwargs.pop("lr"), "optimizer_args.lr")
        if "weight_decay" in kwargs:
            weight_decay = _coerce_float(
                kwargs.pop("weight_decay"), "optimizer_args.weight_decay"
            )
        if "eps" in kwargs:
            eps = _coerce_float(kwargs.pop("eps"), "optimizer_args.eps")
        else:
            eps = 1e-30  # Adafactor 分解二阶矩的稳定项，与 AdamW 的 1e-8 语义不同
        kwargs.pop("betas", None)  # 无一阶动量，betas 不适用；静默丢弃
        return create_automagic_optimizer(
            params=params, lr=learning_rate,
            weight_decay=weight_decay, eps=eps, **kwargs,
        )

    raise ValueError(
        f"Unknown optimizer type: {optimizer_type}. "
        f"Choose from: adamw8bit, adamw, prodigyplus, soap, soap_sf, "
        f"muon, muon_sf, adopt, lion, clion, emosens, automagic"
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
# SOAP
# =============================================================================

def create_soap_optimizer(
    params: ParamInput,
    lr: float,
    betas: tuple = (0.95, 0.95),
    weight_decay: float = 0.01,
    eps: float = 1e-8,
    **kwargs,
) -> Optimizer:
    valid_keys = {
        "shampoo_beta",
        "precondition_frequency",
        "max_precond_dim",
        "merge_dims",
        "precondition_1d",
        "normalize_grads",
        "data_format",
        "correct_bias",
        "precond_in_state",
    }
    soap_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
    ignored = [k for k in kwargs if k not in valid_keys]
    if ignored:
        logger.warning(f"[SOAP] Ignored unsupported params: {ignored}")

    param_list = params if _is_param_groups(params) else list(params)
    logger.info(
        "Creating SOAP optimizer "
        "(lr=%s, betas=%s, wd=%s, eps=%s, precondition_frequency=%s, "
        "max_precond_dim=%s, precondition_1d=%s, merge_dims=%s)",
        lr,
        betas,
        weight_decay,
        eps,
        soap_kwargs.get("precondition_frequency", 10),
        soap_kwargs.get("max_precond_dim", 10000),
        soap_kwargs.get("precondition_1d", False),
        soap_kwargs.get("merge_dims", False),
    )
    return SOAP(
        param_list,
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
        eps=eps,
        **soap_kwargs,
    )


# =============================================================================
# SOAP-SF (Schedule-Free SOAP)
# =============================================================================

def create_soap_sf_optimizer(
    params: ParamInput,
    lr: float,
    betas: tuple = (0.9, 0.95),
    weight_decay: float = 0.01,
    eps: float = 1e-8,
    **kwargs,
) -> Optimizer:
    valid_keys = {
        "shampoo_beta",
        "precondition_frequency",
        "max_precond_dim",
        "merge_dims",
        "precondition_1d",
        "normalize_grads",
        "data_format",
        "correct_bias",
        "precond_in_state",
        # Schedule-Free specific
        "weight_lr_power",
        "r",
        "warmup_steps",
    }
    sf_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
    ignored = [k for k in kwargs if k not in valid_keys]
    if ignored:
        logger.warning(f"[SOAP-SF] Ignored unsupported params: {ignored}")

    param_list = params if _is_param_groups(params) else list(params)
    logger.info(
        "Creating SOAP-SF (Schedule-Free) optimizer "
        "(lr=%s, betas=%s, wd=%s, eps=%s, precondition_frequency=%s, "
        "max_precond_dim=%s, weight_lr_power=%s, r=%s, warmup_steps=%s)",
        lr,
        betas,
        weight_decay,
        eps,
        sf_kwargs.get("precondition_frequency", 10),
        sf_kwargs.get("max_precond_dim", 10000),
        sf_kwargs.get("weight_lr_power", 2.0),
        sf_kwargs.get("r", 0.0),
        sf_kwargs.get("warmup_steps", 0),
    )
    return SOAPScheduleFree(
        param_list,
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
        eps=eps,
        **sf_kwargs,
    )


# =============================================================================
# ADOPT
# =============================================================================

def create_adopt_optimizer(
    params: ParamInput,
    lr: float,
    betas: tuple = (0.9, 0.9999),
    weight_decay: float = 0.0,
    eps: float = 1e-6,
    **kwargs,
) -> Optimizer:
    valid_keys = {"decoupled", "use_clip", "clip_exponent"}
    adopt_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
    ignored = [k for k in kwargs if k not in valid_keys]
    if ignored:
        logger.warning(f"[ADOPT] Ignored unsupported params: {ignored}")

    param_list = params if _is_param_groups(params) else list(params)
    logger.info(
        "Creating ADOPT optimizer "
        "(lr=%s, betas=%s, wd=%s, eps=%s, decoupled=%s, use_clip=%s, clip_exponent=%s)",
        lr,
        betas,
        weight_decay,
        eps,
        adopt_kwargs.get("decoupled", True),
        adopt_kwargs.get("use_clip", True),
        adopt_kwargs.get("clip_exponent", 0.25),
    )
    return ADOPT(
        param_list,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        **adopt_kwargs,
    )


# =============================================================================
# Lion / C-Lion
# =============================================================================

def create_lion_optimizer(
    params: ParamInput,
    lr: float,
    betas: tuple = (0.9, 0.99),
    weight_decay: float = 0.0,
    **kwargs,
) -> Optimizer:
    valid_keys = {"cautious"}
    lion_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
    ignored = [k for k in kwargs if k not in valid_keys]
    if ignored:
        logger.warning(f"[Lion] Ignored unsupported params: {ignored}")

    param_list = params if _is_param_groups(params) else list(params)
    cautious = bool(lion_kwargs.get("cautious", False))
    label = "C-Lion (cautious)" if cautious else "Lion"
    logger.info(
        "Creating %s optimizer (lr=%s, betas=%s, wd=%s)",
        label, lr, betas, weight_decay,
    )
    return Lion(
        param_list,
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
        **lion_kwargs,
    )


# =============================================================================
# EmoSens
# =============================================================================

def create_emosens_optimizer(
    params: ParamInput,
    lr: float = 0.1,
    betas: tuple = (0.9, 0.995),
    weight_decay: float = 0.0,
    eps: float = 1e-8,
    **kwargs,
) -> Optimizer:
    valid_keys = {"stopcoef", "use_shadow", "notify"}
    emosens_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
    ignored = [k for k in kwargs if k not in valid_keys]
    if ignored:
        logger.warning(f"[EmoSens] Ignored unsupported params: {ignored}")

    param_list = params if _is_param_groups(params) else list(params)
    logger.info(
        "Creating EmoSens optimizer "
        "(lr_scope=%s, betas=%s, wd=%s, eps=%s, stopcoef=%s, use_shadow=%s, notify=%s)",
        lr,
        betas,
        weight_decay,
        eps,
        emosens_kwargs.get("stopcoef", 0.04),
        emosens_kwargs.get("use_shadow", False),
        emosens_kwargs.get("notify", False),
    )
    return EmoSens(
        param_list,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        **emosens_kwargs,
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
    d_coef: float = 1.0,                 # LoRA 实战默认：低于 1 容易完全不学
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

    # Optimizer-specific telemetry
    pg0 = optimizer.param_groups[0]
    for key in ("d", "d0", "d_coef", "effective_lr", "k"):
        if key in pg0:
            info[key] = pg0[key]
    for key in ("emoScope", "dNR_hist", "noise_est", "d_est", "c_est", "stop_base", "should_stop"):
        if hasattr(optimizer, key):
            info[key] = getattr(optimizer, key)

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


# =============================================================================
# Muon (Newton-Schulz orthogonalized momentum)
# =============================================================================

def create_muon_optimizer(
    params: ParamInput,
    lr: float = 0.02,
    betas: tuple = (0.9, 0.999),
    weight_decay: float = 0.0,
    eps: float = 1e-8,
    **kwargs,
) -> Optimizer:
    valid_keys = {
        "momentum", "nesterov", "ns_steps", "correct_bias", "rms_scale",
    }
    muon_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
    ignored = [k for k in kwargs if k not in valid_keys]
    if ignored:
        logger.warning(f"[Muon] Ignored unsupported params: {ignored}")

    param_list = params if _is_param_groups(params) else list(params)
    logger.info(
        "Creating Muon optimizer "
        "(lr=%s, momentum=%s, nesterov=%s, ns_steps=%s, wd=%s, betas=%s)",
        lr,
        muon_kwargs.get("momentum", 0.95),
        muon_kwargs.get("nesterov", True),
        muon_kwargs.get("ns_steps", 5),
        weight_decay,
        betas,
    )
    return Muon(
        param_list,
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
        eps=eps,
        **muon_kwargs,
    )


# =============================================================================
# AdamW-SNR (AdamW + cautious / SNR 锐化门控)
# =============================================================================

def create_adamw_snr_optimizer(
    params: ParamInput,
    lr: float = 1e-4,
    betas: tuple = (0.9, 0.999),
    weight_decay: float = 0.0,
    eps: float = 1e-8,
    **kwargs,
) -> Optimizer:
    valid_keys = {"cautious", "snr_power"}
    snr_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
    ignored = [k for k in kwargs if k not in valid_keys]
    if ignored:
        logger.warning(f"[AdamW-SNR] Ignored unsupported params: {ignored}")

    param_list = params if _is_param_groups(params) else list(params)
    cautious = bool(snr_kwargs.get("cautious", False))
    snr_power = float(snr_kwargs.get("snr_power", 1.0))
    if not cautious and snr_power == 1.0:
        logger.info(
            "AdamW-SNR: cautious=False 且 snr_power=1.0 —— 与标准 adamw 数学恒等"
            "（如需门控请设 cautious=true 或 snr_power>1）"
        )
    logger.info(
        "Creating AdamW-SNR optimizer (lr=%s, betas=%s, wd=%s, cautious=%s, snr_power=%s)",
        lr, betas, weight_decay, cautious, snr_power,
    )
    return AdamWSNR(
        param_list, lr=lr, betas=betas, weight_decay=weight_decay, eps=eps,
        **snr_kwargs,
    )


# =============================================================================
# Automagic (逐元素自适应 lr)
# =============================================================================

def create_automagic_optimizer(
    params: ParamInput,
    lr: float = 1e-6,
    weight_decay: float = 0.0,
    eps: float = 1e-30,
    **kwargs,
) -> Optimizer:
    valid_keys = {"min_lr", "max_lr", "lr_bump", "beta2", "clip_threshold"}
    am_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
    ignored = [k for k in kwargs if k not in valid_keys]
    if ignored:
        logger.warning(f"[Automagic] Ignored unsupported params: {ignored}")

    param_list = params if _is_param_groups(params) else list(params)
    logger.info(
        "Creating Automagic optimizer "
        "(start_lr=%s, min_lr=%s, max_lr=%s, lr_bump=%s, beta2=%s, wd=%s) "
        "-- lr 逐权重自适应，训练强度由 max_lr 决定",
        lr,
        am_kwargs.get("min_lr", 1e-7),
        am_kwargs.get("max_lr", 1e-3),
        am_kwargs.get("lr_bump", 1e-6),
        am_kwargs.get("beta2", 0.999),
        weight_decay,
    )
    return Automagic(
        param_list, lr=lr, weight_decay=weight_decay, eps=eps, **am_kwargs,
    )


# =============================================================================
# Muon-SF (Schedule-Free Muon)
# =============================================================================

def create_muon_sf_optimizer(
    params: ParamInput,
    lr: float = 0.02,
    betas: tuple = (0.9, 0.95),
    weight_decay: float = 0.0,
    eps: float = 1e-8,
    **kwargs,
) -> Optimizer:
    valid_keys = {
        "ns_steps", "weight_lr_power", "r", "warmup_steps", "correct_bias",
        "momentum", "rms_scale",
    }
    sf_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
    ignored = [k for k in kwargs if k not in valid_keys]
    if ignored:
        logger.warning(f"[Muon-SF] Ignored unsupported params: {ignored}")

    param_list = params if _is_param_groups(params) else list(params)
    logger.info(
        "Creating Muon-SF (Schedule-Free) optimizer "
        "(lr=%s, betas=%s, wd=%s, ns_steps=%s, momentum=%s, weight_lr_power=%s, r=%s, warmup_steps=%s)",
        lr,
        betas,
        weight_decay,
        sf_kwargs.get("ns_steps", 5),
        sf_kwargs.get("momentum", 0.95),
        sf_kwargs.get("weight_lr_power", 2.0),
        sf_kwargs.get("r", 0.0),
        sf_kwargs.get("warmup_steps", 0),
    )
    return MuonScheduleFree(
        param_list,
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
        eps=eps,
        **sf_kwargs,
    )
