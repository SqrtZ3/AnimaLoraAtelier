"""
Optimizer Utils Module - 优化器创建
===================================
支持多种优化器：
1. 8-bit AdamW (bitsandbytes) - 内存高效
2. 标准 AdamW - 后备选项
3. ProdigyPlus (prodigyplus) - 自适应学习率 + Schedule-Free
"""

from typing import List, Dict, Any, Optional, Iterator

import torch
from torch import nn
from torch.optim import Optimizer, AdamW

# 尝试导入 bitsandbytes
try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False

# 尝试导入 prodigyplus
try:
    from prodigyplus import ProdigyPlusScheduleFree
    PRODIGYPLUS_AVAILABLE = True
except ImportError:
    PRODIGYPLUS_AVAILABLE = False


def create_optimizer(
    optimizer_type: str,
    params: Iterator[nn.Parameter],
    learning_rate: float,
    betas: tuple = (0.9, 0.999),
    weight_decay: float = 0.01,
    eps: float = 1e-8,
    **kwargs
) -> Optimizer:
    """
    创建优化器

    根据配置创建不同类型的优化器。这是工厂模式的应用，
    将优化器创建逻辑集中管理，便于维护和扩展。

    Args:
        optimizer_type: 优化器类型 ("adamw8bit", "adamw", "prodigyplus")
        params: 模型参数迭代器
        learning_rate: 学习率
        betas: Adam beta 参数 (beta1, beta2)
        weight_decay: 权重衰减系数
        eps: 数值稳定性 epsilon
        **kwargs: 其他优化器特定参数

    Returns:
        Optimizer: 创建的优化器实例

    Raises:
        ValueError: 如果优化器类型不支持
        ImportError: 如果需要的库未安装
    """
    optimizer_type = optimizer_type.lower()

    if optimizer_type == "adamw8bit":
        return create_8bit_adamw(
            params=params,
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
            **kwargs
        )

    elif optimizer_type == "adamw":
        return create_standard_adamw(
            params=params,
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
            **kwargs
        )

    elif optimizer_type == "prodigyplus":
        return create_prodigyplus_optimizer(
            params=params,
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
            **kwargs
        )

    else:
        raise ValueError(
            f"Unknown optimizer type: {optimizer_type}. "
            f"Choose from: adamw8bit, adamw, prodigyplus"
        )


def create_8bit_adamw(
    params: Iterator[nn.Parameter],
    lr: float,
    betas: tuple = (0.9, 0.999),
    weight_decay: float = 0.01,
    eps: float = 1e-8,
    min_8bit_size: int = 4096,
    **kwargs
) -> Optimizer:
    """
    创建 8-bit AdamW 优化器
    """
    if not BITSANDBYTES_AVAILABLE:
        raise ImportError(
            "bitsandbytes is required for 8-bit AdamW. "
            "Install with: pip install bitsandbytes"
        )
    
    print(f"Creating 8-bit AdamW optimizer (lr={lr}, weight_decay={weight_decay})")
    
    param_list = list(params)
    optimizer = bnb.optim.AdamW8bit(
        param_list,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        min_8bit_size=min_8bit_size,
        **kwargs
    )
    return optimizer


def create_standard_adamw(
    params: Iterator[nn.Parameter],
    lr: float,
    betas: tuple = (0.9, 0.999),
    weight_decay: float = 0.01,
    eps: float = 1e-8,
    **kwargs
) -> Optimizer:
    """
    创建标准 AdamW 优化器
    """
    # 过滤掉不属于 AdamW 的参数（如 d0, use_stableadamw 等）
    # 这样即便参数传递有误，也不会导致 TypeError
    valid_keys = ['betas', 'eps', 'weight_decay', 'amsgrad', 'maximize', 'capturable', 'differentiable', 'foreach']
    adamw_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
    
    if len(adamw_kwargs) < len(kwargs):
        ignored = [k for k in kwargs if k not in valid_keys]
        print(f"Warning: Standard AdamW does not support parameters {ignored}. They will be ignored.")

    print(f"Creating standard AdamW optimizer (lr={lr}, weight_decay={weight_decay})")
    param_list = list(params)
    optimizer = AdamW(
        param_list,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        **adamw_kwargs
    )
    return optimizer


def create_prodigyplus_optimizer(
    params: Iterator[nn.Parameter],
    lr: float = 1.0,
    betas: tuple = (0.9, 0.99),
    weight_decay: float = 0.01,
    eps: float = 1e-8,
    d0: float = 1e-6,
    use_schedulefree: bool = True,
    **kwargs
) -> Optimizer:
    """
    创建 ProdigyPlusScheduleFree 优化器

    ProdigyPlus 是一个先进的自适应优化器，它结合了 Prodigy 的自适应学习率
    和 Meta 的 Schedule-Free 训练技术。

    特点：
    - 无需手动设置学习率（通常设为 1.0）
    - 无需学习率调度器（Schedule-Free）
    - 训练和评估时需要切换 optimizer.train() / eval()
    """
    if not PRODIGYPLUS_AVAILABLE:
        raise ImportError(
            "prodigyplus is required for ProdigyPlus optimizer. "
            "Install with: pip install prodigy-plus-schedule-free"
        )

    print(f"Creating ProdigyPlus optimizer (base_lr={lr}, d0={d0}, weight_decay={weight_decay})")
    print(f"  Note: Schedule-Free is {'enabled' if use_schedulefree else 'disabled'}")

    param_list = list(params)
    optimizer = ProdigyPlusScheduleFree(
        param_list,
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
        eps=eps,
        d0=d0,
        use_schedulefree=use_schedulefree,
        **kwargs
    )

    return optimizer


def create_optimizer_grouped_parameters(
    model: nn.Module,
    weight_decay: float,
    no_decay_modules: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    创建分组的优化器参数
    """
    if no_decay_modules is None:
        no_decay_modules = ["bias", "LayerNorm.weight", "layernorm.weight", "norm.weight"]
    
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        needs_decay = True
        for no_decay_pattern in no_decay_modules:
            if no_decay_pattern in name:
                needs_decay = False
                break
        
        if needs_decay:
            decay_params.append(param)
        else:
            no_decay_params.append(param)
    
    optimizer_grouped_parameters = [
        {
            "params": decay_params,
            "weight_decay": weight_decay,
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


def get_optimizer_info(optimizer: Optimizer) -> Dict[str, Any]:
    """
    获取优化器信息
    """
    info = {
        "type": type(optimizer).__name__,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "num_param_groups": len(optimizer.param_groups),
    }
    
    total_params = 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            total_params += p.numel()
    
    info["total_trainable_params"] = total_params
    return info
