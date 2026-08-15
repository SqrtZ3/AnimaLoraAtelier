"""适配器专用 AdamW / AdamW-SNR（JAX），语义对齐 `torch.optim.AdamW` 与
`utils/adamw_snr_optimizer.py`。

## 为什么手写而不是用 optax

只训适配器，参数量几千万，优化器就几十行。手写的收益是**能逐行对齐 PyTorch 的
语义**——同一份 yaml 在 GPU 与 TPU 上应当训出同样的动力学，差一个 bias-correction
或把 weight decay 写成耦合式（L2 加进梯度）都会静默地变成"另一个优化器"，
而且事后极难归因。

torch/optim/adamw.py 的单步（decoupled weight decay）：

    p        <- p * (1 - lr * wd)
    m        <- b1*m + (1-b1)*g
    v        <- b2*v + (1-b2)*g^2
    denom    <- sqrt(v)/sqrt(1-b2^t) + eps          # eps 在**开方之后**加
    p        <- p - lr/(1-b1^t) * m/denom

注意两处容易写错、且不报错的地方：
  * `eps` 加在 sqrt 之后、且在 bias-correction2 之后（写成 sqrt(v+eps) 是另一个算法）；
  * weight decay 是**乘在参数上**（decoupled），不是加进梯度（那是 Adam+L2）。

## AdamW-SNR（`optimizer_type: adamw_snr`）

对齐 `utils/adamw_snr_optimizer.py:126-149`，在 AdamW 的逐坐标更新之后加两件事：

  1. **SNR 锐化**（`snr_power=p>1`）：`|update|` 本身就是该坐标的信噪比估计
     （m/√v），把它 p 次幂再**按张量均值重归一化**，保持 lr 语义可比：

         s = |u| ; u <- sign(u) * s^p * (mean(s) / mean(s^p))

     依据：memory `[[krea2-gradient-snr-measurement]]` —— 真实 LoRA 梯度里噪声
     坐标占 54% 但只占 6.84% 的更新能量，锐化就是把这 6.84% 再压下去。
  2. **cautious 掩码**（arXiv:2411.16085）：`update*grad <= 0` 的坐标置零，
     再按保留比例重归一化。

`snr_power=1.0 且 cautious=False` 时与上面的 AdamW **逐 bit 恒等**（代码里就是
同一条路径，不是"数学上等价"）。

## fp32 master 权重

适配器参数在优化器里恒以 **fp32** 保存，只在喂进前向时降到 bf16。bf16 只有 8 位
尾数，lr*update 相对参数小于 2^-8 时加法会被舍成 0 —— 参数看起来"冻住"、loss
不动，也不报错。仓库在移植 Automagic 时踩过同样的坑
（memory `[[automagic-optimizer-landing]]`）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp

PyTree = Any

#: LoRA+ 的"B 侧"叶子名。标准 LoRA 是 `b`（= lora_up），LoKr 的对应物是 `w2b`
#: （低秩侧、零初始化的那一支）。`lora_plus_ratio=1.0`（默认）时这个集合不产生
#: 任何影响。
B_SIDE = ("b", "w2b")


@dataclass(frozen=True)
class AdamWConfig:
    lr: float = 1e-4
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0          # 0 = 关闭
    warmup_steps: int = 0
    #: LoRA+ 的 B 侧学习率倍率（trainer/lora.py 支持模块级 lr）。1.0 = 关闭。
    #: 默认 1.0 保证与朴素 LoRA 数学等价 —— 新能力 opt-in、default-off。
    lora_plus_ratio: float = 1.0
    #: SNR 锐化指数。1.0 = 标准 AdamW（默认，行为中立）。
    snr_power: float = 1.0
    #: cautious 掩码（arXiv:2411.16085）。False = 标准 AdamW（默认）。
    cautious: bool = False

    def __post_init__(self):
        if self.snr_power < 1.0:
            raise ValueError(
                f"snr_power={self.snr_power} < 1 会【放大】噪声坐标的相对步长，"
                f"与该门控的意图相反（同 utils/adamw_snr_optimizer.py:76）")
        if self.snr_power > 4.0:
            raise ValueError(
                f"snr_power={self.snr_power} 过大：p>=3 时 >99% 的更新能量集中到"
                f"极少数坐标，等价于把 LoRA 退化成稀疏更新（同上，:81）")


def init_state(params: PyTree) -> Dict[str, Any]:
    """优化器状态：fp32 master 参数 + 一阶/二阶矩 + 步数。

    **必须拷贝而不是 astype**：输入若已是 fp32，`astype` 是空操作、返回同一个
    buffer；而训练步用 `donate_argnums` 捐献 state（r32 全目标约 552MB，TPU 上
    值得捐），捐献会把那个 buffer 删掉 —— 调用方手里的参数跟着失效，报
    `RuntimeError: Array has been deleted`。本地 check_train_loop 抓到过这一条。
    """
    f32 = jax.tree.map(lambda x: jnp.array(x, jnp.float32), params)
    return {"master": f32,
            "m": jax.tree.map(jnp.zeros_like, f32),
            "v": jax.tree.map(jnp.zeros_like, f32),
            "step": jnp.zeros((), jnp.int32)}


def global_norm(tree: PyTree) -> jnp.ndarray:
    leaves = jax.tree.leaves(tree)
    return jnp.sqrt(sum(jnp.sum(x.astype(jnp.float32) ** 2) for x in leaves))


def _lr_at(cfg: AdamWConfig, step: jnp.ndarray) -> jnp.ndarray:
    """线性 warmup，之后恒定。

    **有意不做衰减**：用户的训练范式是给一个极大 epoch 一直训、随时手动停、
    从任意 step 的 checkpoint 里挑模型（见 skill 2.5）。带衰减的调度会让
    "第 N 步的 checkpoint" 的含义依赖于总步数，与这个范式冲突。
    """
    if cfg.warmup_steps <= 0:
        return jnp.asarray(cfg.lr, jnp.float32)
    frac = jnp.minimum((step.astype(jnp.float32) + 1.0) / float(cfg.warmup_steps), 1.0)
    return jnp.asarray(cfg.lr, jnp.float32) * frac


def _is_b_side(path) -> bool:
    """叶子路径的最后一段是不是 LoRA+ 的 B 侧。"""
    last = path[-1]
    name = getattr(last, "key", None) or getattr(last, "name", None)
    return name in B_SIDE


def update(state: Dict[str, Any], grads: PyTree, cfg: AdamWConfig,
           dtype=jnp.bfloat16) -> Tuple[Dict[str, Any], PyTree, Dict[str, jnp.ndarray]]:
    """一步更新。返回 (新状态, 供前向用的低精度参数, 诊断量)。

    对**任意 pytree** 形状的参数都成立（标准 LoRA 的 {a,b}、LoKr 的
    {w1,w2a,w2b}、DoRA 多出来的 {dora}）—— 用 `tree_map_with_path` 而不是写死
    键名，加一种适配器结构不必再动优化器。

    诊断量里的 `gnorm` 是**裁剪前**的梯度全局范数——它是判断训练是否健康最便宜的
    指标（突然塌到 0 = 适配器掉线；持续贴着 max_grad_norm = lr 偏大）。
    """
    step = state["step"] + 1
    lr = _lr_at(cfg, state["step"])
    gnorm = global_norm(grads)

    if cfg.max_grad_norm and cfg.max_grad_norm > 0:
        scale = jnp.minimum(1.0, cfg.max_grad_norm / (gnorm + 1e-6))
        grads = jax.tree.map(lambda g: g.astype(jnp.float32) * scale, grads)
    else:
        grads = jax.tree.map(lambda g: g.astype(jnp.float32), grads)

    bc1 = 1.0 - cfg.b1 ** step.astype(jnp.float32)
    bc2 = 1.0 - cfg.b2 ** step.astype(jnp.float32)

    def step_one(path, p, m, v, g):
        m = cfg.b1 * m + (1.0 - cfg.b1) * g
        v = cfg.b2 * v + (1.0 - cfg.b2) * g * g
        # eps 加在 sqrt 与 bias-correction2 **之后**，与 torch 一致
        denom = jnp.sqrt(v) / jnp.sqrt(bc2) + cfg.eps
        upd = (m / bc1) / denom                       # ≡ AdamW 的逐坐标更新

        if cfg.snr_power != 1.0:
            s = jnp.abs(upd)
            sharp = s ** cfg.snr_power
            # 按**张量**均值重归一化（不是全局），与 PyTorch 侧逐 param 的做法一致
            upd = jnp.sign(upd) * sharp * (jnp.mean(s)
                                           / jnp.maximum(jnp.mean(sharp), 1e-30))
        if cfg.cautious:
            mask = (upd * g > 0).astype(jnp.float32)
            mask = mask * (mask.size / jnp.maximum(jnp.sum(mask), 1.0))
            upd = upd * mask

        lr_p = lr * (cfg.lora_plus_ratio if _is_b_side(path) else 1.0)
        p = p * (1.0 - lr_p * cfg.weight_decay)       # decoupled
        p = p - lr_p * upd
        return p, m, v

    # 三棵树（master/m/v/grads）结构相同 -> 同一个 treedef，按 flatten 的顺序
    # 一一对应。用 flatten 而不是 tree_map 返回元组再拆，是为了不依赖
    # "元组算不算叶子" 这种容易踩的判据。
    pl, treedef = jax.tree_util.tree_flatten_with_path(state["master"])
    ml = jax.tree.leaves(state["m"])
    vl = jax.tree.leaves(state["v"])
    gl = jax.tree.leaves(grads)
    if not (len(pl) == len(ml) == len(vl) == len(gl)):
        raise ValueError(f"master/m/v/grads 的叶子数不一致："
                         f"{len(pl)}/{len(ml)}/{len(vl)}/{len(gl)}")
    outs = [step_one(path, p, m, v, g)
            for (path, p), m, v, g in zip(pl, ml, vl, gl)]
    master = jax.tree.unflatten(treedef, [o[0] for o in outs])
    m_new = jax.tree.unflatten(treedef, [o[1] for o in outs])
    v_new = jax.tree.unflatten(treedef, [o[2] for o in outs])

    new_state = {"master": master, "m": m_new, "v": v_new, "step": step}
    fwd = jax.tree.map(lambda x: x.astype(dtype), master)
    return new_state, fwd, {"gnorm": gnorm, "lr": lr, "step": step}
