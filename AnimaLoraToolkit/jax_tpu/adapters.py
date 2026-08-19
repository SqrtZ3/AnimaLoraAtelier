r"""适配器（LoRA / LoKr，可选 DoRA）——与 `trainer/lora.py` 逐算子对齐的 JAX 实现。

## 为什么单独一个模块

原 `anima_jax.dense` 只认 `{a, b}` 两个矩阵的标准 LoRA。用户在用的配方是
`lora_type: lokr` + `lora_variant: dora` + 逐块 `lora_reg_dims`，这三件事分别改变
**参数结构**、**输出的归一化方式**、**每块的形状**，没法靠加几个 if 塞进去。

## 对齐依据（逐条指向 PyTorch 源）

  trainer/lora.py:774-776   LoKr 三因子 `w1 (f,f)` / `w2_a (out_dim, r)` / `w2_b (r, in_dim)`
  trainer/lora.py:963-1003  kron-bypass 前向：x->(P,f,in_dim) -> @w2_bᵀ -> @w2_aᵀ -> w1 @ ·
  trainer/lora.py:771       `scaling = alpha / rank`（**rank 是 cap 之后的**）
  trainer/lora.py:832       w1 ~ N(0, w1_init_std²)，默认 std=0.1
  trainer/lora.py:855-856   w2_a kaiming_uniform(a=√5)（fan_in=rank）、w2_b=0 -> step-0 净增量 0
  trainer/lora.py:861-877   factor 自动下调到能同时整除 in/out 的**最大**值
  trainer/lora.py:950-957   rank_dropout：对 w2_b 的行做 inverted dropout，scale=1/(1-p+1e-6)
  trainer/lora.py:1006-1010 module_dropout：整个适配器输出置 0（**不做 1/(1-p) 补偿**）
  trainer/lora.py:1016-1112 DoRA 的 `merged_row_norms`（免物化 ΔW 的逐行范数）
  trainer/lora.py:1262-1305 DoRA 前向：`(base + lora) * (dora_scale / ‖W+ΔW‖_row)`
  trainer/lora.py:1614-1632 `reg_dims` / `reg_alphas` 用 `re.fullmatch`，**首个命中的模式生效**

## 逐块不同 rank 怎么装进 `lax.scan`

`lora_reg_dims` 让第 7-20 块的 attn 是 r=64、第 21-27 块是 r=16。而训练必须走
`lax.scan`（展开路径 1.01 MB/token，budget 16384 即 OOM，见 anima_jax.stack_blocks），
scan 要求每块的参数**形状相同**。

解法：所有块按 `rmax = max(rank_l)` 分配，配一个**固定的** 0/1 掩码 `rmask [L, rmax]`，
在前向里乘到 `w2_b`（低秩侧）上。于是：

  * 多出来的那些 rank 通道对输出的贡献恒为 0；
  * 它们收到的梯度**恒为 0**（乘 0 的链式法则），不会偷偷学到东西；
  * 导出时按各块自己的 `rank_l` 切片，产物与 PyTorch 侧逐块 rank 的文件完全同形。

代价是显存/带宽按 rmax 记（r=16 的块也占 64 列）。r32 全目标下多出来的这部分
约 20MB fp32 master，相对 552MB 的优化器状态可以忽略。

## 掩码乘在哪一侧要紧

必须乘 `w2_b`（或标准 LoRA 的 `b`），不能乘 `w2_a`：`w2_b` 初值为 0，掩掉的通道
从一开始就是 0，前向逐 bit 不受影响；而 `w2_a` 是随机初始化的，掩它会让
`w2_a @ w2_b` 的乘积结构变了（虽然数值上仍是 0，但 DoRA 的 `merged_row_norms`
里 `U = w2_a @ w2_b` 会算错）。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

PyTree = Any

KINDS = ("lora", "lokr")
VARIANTS = ("base", "dora")


# ── 配置 ──────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class AdapterConfig:
    """与 yaml 的 `lora_*` 字段同名同义。"""
    kind: str = "lora"                     # lora_type
    variant: str = "base"                  # lora_variant: base / dora
    rank: int = 32                         # lora_rank
    alpha: Optional[float] = None          # lora_alpha（None -> = rank）
    factor: int = 8                        # lokr_factor
    rank_dropout: float = 0.0
    module_dropout: float = 0.0
    w1_init_std: float = 0.1               # lokr_w1_init_std
    #: "fp32"（默认，与 PyTorch 侧默认一致）/ "native"（省显存）。
    #: **TPU 上这个开关的数值意义比 GPU 上小**：XLA 在 TPU 上对 fp32 matmul 的
    #: 默认 precision 就是单趟 bf16，所以 fp32 主要是多占一份显存/带宽，
    #: 并不真的换来 fp32 的乘法精度。
    compute_dtype: str = "fp32"
    #: 逐模块 rank / alpha 覆盖（yaml 的 lora_reg_dims / lora_reg_alphas）。
    #: 键是**正则**，按插入顺序 fullmatch，首个命中生效（同 trainer/lora.py:1616）。
    reg_dims: Dict[str, int] = field(default_factory=dict)
    reg_alphas: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"lora_type 只支持 {KINDS}（TPU 后端），得到 {self.kind!r}")
        if self.variant not in VARIANTS:
            raise ValueError(f"lora_variant 只支持 {VARIANTS}（TPU 后端），"
                             f"得到 {self.variant!r}")
        if self.compute_dtype not in ("fp32", "native"):
            raise ValueError(f"lokr_compute_dtype 只支持 'fp32'/'native'，"
                             f"得到 {self.compute_dtype!r}")
        if not (0.0 <= self.rank_dropout < 1.0):
            raise ValueError(f"rank_dropout 必须在 [0,1)，得到 {self.rank_dropout}")
        if not (0.0 <= self.module_dropout < 1.0):
            raise ValueError(f"module_dropout 必须在 [0,1)，得到 {self.module_dropout}")

    @property
    def uses_dropout(self) -> bool:
        return self.rank_dropout > 0.0 or self.module_dropout > 0.0


def resolve_reg(table: Dict[str, Any], name: str, default):
    """`re.fullmatch` 逐条试，首个命中生效——与 trainer/lora.py:1616 同序。

    dict 在 Python 3.7+ 保插入顺序，而 yaml 的 `lora_reg_dims` 是有序映射，
    所以"更具体的模式写在前面"这件事在两侧含义相同。写反了不会报错，只会让
    `.*mlp.*: 32` 这种兜底项吃掉前面所有的逐块设定。
    """
    for pat, v in table.items():
        if re.fullmatch(pat, name):
            return v
    return default


def find_factor(in_f: int, out_f: int, target: int) -> int:
    """trainer/lora.py:861 —— 从 target 往下找**最大**的公约因子，找不到才退到 1。"""
    target = max(1, int(target))
    for f in range(target, 0, -1):
        if in_f % f == 0 and out_f % f == 0:
            return f
    return 1


# ── 形状表 ────────────────────────────────────────────────────────────────────
def target_shapes(model_cfg) -> Dict[str, Tuple[int, int]]:
    """target -> (in_features, out_features)，口径同 anima_jax.init_lora 的 shape 表。"""
    D, C = model_cfg.model_channels, model_cfg.crossattn_dim
    H = int(D * model_cfg.mlp_ratio)
    return {
        "self_attn.q_proj": (D, D), "self_attn.k_proj": (D, D),
        "self_attn.v_proj": (D, D), "self_attn.output_proj": (D, D),
        "cross_attn.q_proj": (D, D), "cross_attn.k_proj": (C, D),
        "cross_attn.v_proj": (C, D), "cross_attn.output_proj": (D, D),
        "mlp.layer1": (D, H), "mlp.layer2": (H, D),
    }


@dataclass(frozen=True)
class TargetPlan:
    """一个 target 在 28 个块上的结构（rank/alpha 可逐块不同）。"""
    name: str
    in_features: int
    out_features: int
    ranks: Tuple[int, ...]        # 每块**有效** rank（已按 out_dim/in_dim cap）
    alphas: Tuple[float, ...]
    factor: int = 1               # lokr 才有意义
    in_dim: int = 0
    out_dim: int = 0

    @property
    def rmax(self) -> int:
        return max(self.ranks)

    @property
    def scales(self) -> Tuple[float, ...]:
        return tuple(a / r for a, r in zip(self.alphas, self.ranks))


def plan_targets(model_cfg, cfg: AdapterConfig,
                 targets: Sequence[str]) -> Dict[str, TargetPlan]:
    """按 reg_dims / reg_alphas 解析出每个 target 在每块上的 rank/alpha/factor。

    模块名用 `blocks.{i}.{target}`（与导出键 `lora_unet_blocks_0_self_attn_q_proj`
    同源）。yaml 里的模式都带 `.*` 前缀，所以带不带 `net.` 前缀都能命中。
    """
    shapes = target_shapes(model_cfg)
    L = model_cfg.num_blocks
    out: Dict[str, TargetPlan] = {}
    for t in targets:
        if t not in shapes:
            raise ValueError(f"未知 target {t!r}，可选 {sorted(shapes)}")
        i_f, o_f = shapes[t]
        if cfg.kind == "lokr":
            f = find_factor(i_f, o_f, cfg.factor)
            in_dim, out_dim = i_f // f, o_f // f
            cap = min(in_dim, out_dim)
        else:
            f, in_dim, out_dim = 1, i_f, o_f
            cap = min(i_f, o_f)
        ranks, alphas = [], []
        for i in range(L):
            name = f"blocks.{i}.{t}"
            r = int(resolve_reg(cfg.reg_dims, name, cfg.rank))
            a = float(resolve_reg(cfg.reg_alphas, name,
                                  cfg.alpha if cfg.alpha is not None else cfg.rank))
            r = max(1, min(r, cap))
            ranks.append(r)
            alphas.append(a)
        out[t] = TargetPlan(t, i_f, o_f, tuple(ranks), tuple(alphas), f, in_dim, out_dim)
    return out


# ── 初始化 ────────────────────────────────────────────────────────────────────
def init(key, model_cfg, cfg: AdapterConfig, targets: Sequence[str],
         base_params: Optional[PyTree] = None,
         dtype=jnp.float32) -> Tuple[PyTree, PyTree, Dict[str, TargetPlan]]:
    """-> (trainable, consts, plans)，两棵树都是 scan 布局（每个 target 一份 [L, ...]）。

    `base_params` 只在 DoRA 时需要（`dora_scale` 的初值 = 冻结底模的逐行范数，
    trainer/lora.py:1241）。它必须是 `anima_jax.stack_blocks` 之后的堆叠布局。

    **step-0 中立**：LoKr 的 `w2_b` / 标准 LoRA 的 `b` 恒为 0 -> ΔW=0；DoRA 的
    `dora_scale` 初值恰为 ‖W‖_row -> 缩放因子恒 1。两条都必须成立，否则第一步就
    偏了（本仓库落地新 adapter 的硬约束，见 skill 2.2）。
    """
    if cfg.variant == "dora" and base_params is None:
        raise ValueError("DoRA 需要 base_params 来初始化 dora_scale（=‖W‖_row）")
    plans = plan_targets(model_cfg, cfg, targets)
    bw = None
    if base_params is not None:
        bw = lambda t: jnp.sum(_base_weight(base_params, t).astype(jnp.float32) ** 2,
                               axis=-1)
    return init_from_plans(key, cfg, plans, bw, dtype) + (plans,)


def init_from_plans(key, cfg: AdapterConfig, plans: Dict[str, TargetPlan],
                    base_row_sq_fn=None, dtype=jnp.float32
                    ) -> Tuple[PyTree, PyTree]:
    """init 的主体（与模型族无关）：按 plans 造 (trainable, consts)。

    每个 target 的堆叠份数取 `len(plan.ranks)`（Anima = 28 块；Krea2 = 28/2/1）。
    `base_row_sq_fn(target) -> [count, out]`（fp32 逐行平方范数）只在 DoRA 时
    需要（None 且开 DoRA 会 fail-fast）。Krea2 用它从"加载时流式算好的逐行
    范数"取值 —— 那边权重是分片的，不该为了取 DoRA 初值再 gather 一份全量。
    """
    trainable: Dict[str, Any] = {}
    consts: Dict[str, Any] = {}
    keys = jax.random.split(key, len(plans) * 2)
    n = 0
    for t, pl in plans.items():
        L = len(pl.ranks)
        rmax = pl.rmax
        rmask = np.zeros((L, rmax), np.float32)
        for i, r in enumerate(pl.ranks):
            rmask[i, :r] = 1.0
        c = {"rmask": jnp.asarray(rmask),
             "scale": jnp.asarray(np.asarray(pl.scales, np.float32))}

        if cfg.kind == "lokr":
            # w2_a 的 kaiming_uniform(a=√5) 边界按**该块自己的 rank** 算
            # （torch 对 2D 张量取 fan_in = size(1)），逐块不同 -> 逐块生成。
            bounds = np.asarray([1.0 / math.sqrt(r) for r in pl.ranks], np.float32)
            u = jax.random.uniform(keys[n], (L, pl.out_dim, rmax), jnp.float32,
                                   minval=-1.0, maxval=1.0)
            p = {"w1": (float(cfg.w1_init_std)
                        * jax.random.normal(keys[n + 1], (L, pl.factor, pl.factor),
                                            jnp.float32)),
                 "w2a": u * jnp.asarray(bounds)[:, None, None],
                 "w2b": jnp.zeros((L, rmax, pl.in_dim), jnp.float32)}
        else:
            bounds = np.asarray([1.0 / math.sqrt(pl.in_features)] * L, np.float32)
            u = jax.random.uniform(keys[n], (L, pl.in_features, rmax), jnp.float32,
                                   minval=-1.0, maxval=1.0)
            p = {"a": u * jnp.asarray(bounds)[:, None, None],
                 "b": jnp.zeros((L, rmax, pl.out_features), jnp.float32)}
        n += 2

        if cfg.variant == "dora":
            if base_row_sq_fn is None:
                raise ValueError("DoRA 需要 base_row_sq_fn 来初始化 dora_scale（=‖W‖_row）")
            row_sq = base_row_sq_fn(t).astype(jnp.float32)      # [L, out]
            p["dora"] = jnp.sqrt(jnp.maximum(row_sq, 1e-12))
            c["base_row_sq"] = row_sq                 # W 冻结 -> 全程不变，缓存
        trainable[t] = jax.tree.map(lambda x: x.astype(dtype), p)
        consts[t] = c
    return trainable, consts


def _base_weight(base_params: PyTree, target: str) -> jnp.ndarray:
    """取某个 target 的 [L, out, in] 权重。**两种块布局都接受**。

    scan 布局（`stack_blocks` 之后）直接取；展开布局（`blocks` 是 list）**只把这一个
    target 堆起来**——不要为了取 DoRA 初值把整棵 3.91GB 的权重树 stack 一份，
    那会让展开路径在 init 阶段就峰值翻倍（单 chip 只有 15.7GiB）。
    """
    def pick(node: Any) -> Any:
        for part in target.split("."):
            node = node[part]
        return node

    blocks = base_params["blocks"]
    node = (jnp.stack([pick(b) for b in blocks])
            if isinstance(blocks, (list, tuple)) else pick(blocks))
    if not hasattr(node, "ndim") or node.ndim != 3:
        raise ValueError(f"target {target!r} 取到的不是 [L,out,in]（拿到 "
                         f"{getattr(node, 'shape', type(node))}）")
    return node


# ── 每步的 dropout 掩码 ───────────────────────────────────────────────────────
def sample_dropout(key, cfg: AdapterConfig, plans: Dict[str, TargetPlan],
                   num_blocks: Optional[int] = None, training: bool = True) -> Optional[PyTree]:
    """逐块逐 target 抽 rank/module dropout 掩码。返回 None = 不做 dropout。

    与 PyTorch 的差别只有"什么时候抽"：那边是每个模块 forward 时各抽各的，
    这边一次性抽好整棵树再喂进 scan。分布相同（都是逐模块独立伯努利），
    只有 RNG 消费顺序不同 —— 不影响任何统计性质。

    `num_blocks` 缺省时按各 plan 自己的份数（len(ranks)）——Krea2 的
    28/2/1 混合堆叠走这条；Anima 调用方显式传 28，行为不变。
    """
    if not (training and cfg.uses_dropout):
        return None
    out: Dict[str, Any] = {}
    keys = jax.random.split(key, len(plans) * 2)
    n = 0
    for t, pl in plans.items():
        L = int(num_blocks) if num_blocks is not None else len(pl.ranks)
        d: Dict[str, jnp.ndarray] = {}
        if cfg.rank_dropout > 0:
            keep = (jax.random.uniform(keys[n], (L, pl.rmax))
                    >= cfg.rank_dropout).astype(jnp.float32)
            # inverted dropout，scale 与 trainer/lora.py:954 逐字节一致（含 1e-6）
            d["rd"] = keep * (1.0 / (1.0 - cfg.rank_dropout + 1e-6))
        if cfg.module_dropout > 0:
            # **不做 1/(1-p) 补偿**，与 trainer/lora.py:1006 一致
            d["md"] = (jax.random.uniform(keys[n + 1], (L,))
                       >= cfg.module_dropout).astype(jnp.float32)
        n += 2
        out[t] = d
    return out


# ── 前向 ──────────────────────────────────────────────────────────────────────
def apply(x: jnp.ndarray, w: jnp.ndarray, cfg: AdapterConfig,
          p: Optional[Dict[str, jnp.ndarray]],
          c: Optional[Dict[str, jnp.ndarray]] = None,
          d: Optional[Dict[str, jnp.ndarray]] = None) -> jnp.ndarray:
    """`y = base(x) + adapter(x)`（DoRA 时是 `(base+adapter) * 幅度比`）。

    p/c/d 都是**单块**的切片（w1 是 [f,f] 而不是 [L,f,f]）——在 scan 里天然如此。
    p=None 时退化成纯 base，逐 bit 等于 `x @ wᵀ`。
    """
    base = x @ w.T.astype(x.dtype)
    if p is None:
        return base
    if "a" in p:                                   # 标准 LoRA（也含 {a,b} 旧格式）
        delta, extra = _lora_delta(x, p, c, d, cfg)
    else:
        delta, extra = _lokr_delta(x, p, c, d, cfg)

    if "dora" not in p:
        out = base + delta.astype(base.dtype)
        md = None if d is None else d.get("md")
        return out if md is None else base + md.astype(base.dtype) * delta.astype(base.dtype)

    # ── DoRA：在输出域做幅度归一（trainer/lora.py:1270-1305）──────────────────
    # 不物化 (out, in) 的 ΔW —— 逐行范数由 `_merged_row_norm` 用因子闭式算出。
    merged = _merged_row_norm(w, p, c, extra, cfg)
    scale = p["dora"].astype(jnp.float32) / jnp.maximum(merged, 1e-6)
    dora = ((base.astype(jnp.float32) + delta.astype(jnp.float32))
            * scale).astype(base.dtype)
    md = None if d is None else d.get("md")
    if md is None:
        return dora
    # module dropout 命中时退回纯 base（= trainer/lora.py:1268 的 eager 分支，
    # 也 = :1301 的输出域混合式；两者在 keep∈{0,1} 上恒等）
    return base + md.astype(base.dtype) * (dora - base)


def _eff_low(p, c, d, key: str) -> jnp.ndarray:
    """低秩侧因子乘上 rmask（逐块有效 rank）与 rank-dropout 掩码。

    两个掩码都作用在 rank 维（`w2_b` 的行 / `b` 的行）。乘在这一侧的理由见模块
    docstring —— 换到 `w2_a` 那侧会让 DoRA 的 `U = w2_a @ w2_b` 算错。
    """
    v = p[key]
    m = None
    if c is not None and "rmask" in c:
        m = c["rmask"]
    if d is not None and "rd" in d:
        m = d["rd"] if m is None else m * d["rd"]
    return v if m is None else v * m[:, None].astype(v.dtype)


def _compute_dtype(cfg: AdapterConfig, x: jnp.ndarray):
    return jnp.float32 if cfg.compute_dtype == "fp32" else x.dtype


def _lokr_delta(x, p, c, d, cfg):
    """kron-bypass：`y = w1 @ ((x_(f,in_dim) @ w2_bᵀ) @ w2_aᵀ)`（lora.py:963-1003）。

    数学上等价于 `x @ kron(w1, w2_a @ w2_b)ᵀ`，但从不物化 (out, in) 的 kron 矩阵。
    """
    dt = _compute_dtype(cfg, x)
    w1 = p["w1"].astype(dt)
    w2a = p["w2a"].astype(dt)
    w2b = _eff_low(p, c, d, "w2b").astype(dt)
    f, od = w1.shape[0], w2a.shape[0]
    lead = x.shape[:-1]
    xf = x.astype(dt).reshape(*lead, f, w2b.shape[-1])
    tmp = jnp.einsum("...fi,ri->...fr", xf, w2b)
    tmp = jnp.einsum("...fr,or->...fo", tmp, w2a)
    y = jnp.einsum("ij,...jo->...io", w1, tmp)
    s = 1.0 if c is None else c["scale"]
    return y.reshape(*lead, f * od) * jnp.asarray(s, dt), w2b


def _lora_delta(x, p, c, d, cfg):
    """标准 LoRA：`y = ((x @ a) @ b) * scale`。a:(in,r) b:(r,out)。"""
    dt = _compute_dtype(cfg, x)
    a = p["a"].astype(dt)
    b = _eff_low(p, c, d, "b").astype(dt)
    s = 1.0 if c is None else c["scale"]
    return ((x.astype(dt) @ a) @ b) * jnp.asarray(s, dt), b


def _merged_row_norm(w, p, c, low, cfg) -> jnp.ndarray:
    """‖W + ΔW‖ 的逐行 L2 范数，**不物化 ΔW**（trainer/lora.py:1016）。

      ‖W+ΔW‖² = ‖W‖² + 2⟨W, ΔW⟩ + ‖ΔW‖²

    LoKr 下 ΔW = s·kron(w1, U)，U = w2_a @ w2_b：
      ‖ΔW_{f,o}‖² = s²·‖w1[f]‖²·‖U[o]‖²
      ⟨W_{f,o}, ΔW_{f,o}⟩ = s·Σ_{f_i} w1[f,f_i]·⟨W_r[f,o,f_i,:], U[o,:]⟩
    最大中间量是 U (out_dim×in_dim)，比 full ΔW 小 factor² 倍。

    `base_row_sq` 由 init 缓存（W 冻结、全程不变）；没有就现算。
    """
    wf = w.astype(jnp.float32)
    if c is not None and "base_row_sq" in c:
        w_sq = c["base_row_sq"].astype(jnp.float32)
    else:
        w_sq = jnp.sum(wf ** 2, axis=-1)
    s = jnp.asarray(1.0 if c is None else c["scale"], jnp.float32)

    if "w1" in p:
        w1 = p["w1"].astype(jnp.float32)
        u = p["w2a"].astype(jnp.float32) @ low.astype(jnp.float32)     # (od, idim)
        f, od = w1.shape[0], u.shape[0]
        delta_sq = (s ** 2) * (jnp.sum(w1 ** 2, -1)[:, None]
                               * jnp.sum(u ** 2, -1)[None, :]).reshape(-1)
        w_r = wf.reshape(f, od, f, u.shape[-1])
        wu = jnp.einsum("abcd,bd->abc", w_r, u)
        dot = s * jnp.einsum("abc,ac->ab", wu, w1)
        dot = dot.reshape(-1)
    else:
        # 标准 LoRA：ΔW = s·(a@b)ᵀ。同样不物化——
        #   ⟨W,ΔW⟩[o] = s·Σ_r b[r,o]·(W@a)[o,r]
        #   ‖ΔW‖²[o]  = s²·b[:,o]ᵀ (aᵀa) b[:,o]
        a = p["a"].astype(jnp.float32)
        b = low.astype(jnp.float32)
        wa = wf @ a                                     # (out, r)
        dot = s * jnp.sum(wa * b.T, axis=-1)
        g = a.T @ a                                     # (r, r)
        delta_sq = (s ** 2) * jnp.sum(b * (g @ b), axis=0)
    return jnp.sqrt(jnp.maximum(w_sq + 2.0 * dot + delta_sq, 1e-12))


# ── 导出 ──────────────────────────────────────────────────────────────────────
def unstack(trainable: PyTree, plans: Dict[str, TargetPlan],
            num_blocks: int, cfg: AdapterConfig) -> Dict[str, Dict[str, np.ndarray]]:
    """scan 布局 -> `{"blocks.{i}.{target}": {因子名: ndarray}}`（Anima 命名）。"""
    return unstack_named(trainable, plans, cfg,
                         lambda t, i: f"blocks.{i}.{t}", num_blocks)


def unstack_named(trainable: PyTree, plans: Dict[str, TargetPlan],
                  cfg: AdapterConfig, key_fn, num_blocks: Optional[int] = None
                  ) -> Dict[str, Dict[str, np.ndarray]]:
    """**按各块自己的 rank 切回去**（掩掉的通道恒为 0，切掉它们不改变任何数值，
    只是让产物与 PyTorch 侧逐块 rank 的文件同形）。

    `key_fn(target, i)` 决定展开名 —— Krea2 的栈名带 layerwise/refiner 路径、
    单例不展开，命名规则与 Anima 的 `blocks.{i}.{t}` 不同，但切片逻辑同一份。
    """
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for t, pl in plans.items():
        p = trainable[t]
        L = int(num_blocks) if num_blocks is not None else len(pl.ranks)
        for i in range(L):
            r = pl.ranks[i]
            key = key_fn(t, i)
            if cfg.kind == "lokr":
                mod = {"lokr_w1": np.asarray(p["w1"][i], np.float32),
                       "lokr_w2_a": np.asarray(p["w2a"][i][:, :r], np.float32),
                       "lokr_w2_b": np.asarray(p["w2b"][i][:r], np.float32)}
            else:
                mod = {"a": np.asarray(p["a"][i][:, :r], np.float32),
                       "b": np.asarray(p["b"][i][:r], np.float32)}
            if "dora" in p:
                mod["dora_scale"] = np.asarray(p["dora"][i], np.float32)
            mod["_alpha"] = np.asarray(pl.alphas[i], np.float32)
            out[key] = mod
    return out


def summary(plans: Dict[str, TargetPlan], cfg: AdapterConfig) -> str:
    """结构摘要。**上真机前先看这个**：factor 被自动下调、rank 被 cap、
    rmax 比实际 rank 大很多，这三件事都不报错但都有代价。"""
    lines = [f"adapter: {cfg.kind}"
             + (f"(factor={cfg.factor})" if cfg.kind == "lokr" else "")
             + f" variant={cfg.variant} rank_dropout={cfg.rank_dropout}"
             f" module_dropout={cfg.module_dropout}"]
    total = 0
    for t, pl in sorted(plans.items()):
        uniq = sorted(set(pl.ranks))
        if cfg.kind == "lokr":
            per = [pl.factor ** 2 + pl.out_dim * r + r * pl.in_dim for r in pl.ranks]
            note = (f"f={pl.factor}" + ("" if pl.factor == cfg.factor
                                        else f"(请求 {cfg.factor}，已下调)")
                    + f" {pl.out_dim}x{pl.in_dim}")
        else:
            per = [(pl.in_features + pl.out_features) * r for r in pl.ranks]
            note = f"{pl.out_features}x{pl.in_features}"
        if cfg.variant == "dora":
            per = [x + pl.out_features for x in per]
        total += sum(per)
        lines.append(f"  {t:<24} rank {uniq} rmax={pl.rmax:<3} {note} "
                     f"参数 {sum(per) / 1e6:.2f}M")
    lines.append(f"  合计可训练参数 {total / 1e6:.2f}M "
                 f"(fp32 master+m+v ≈ {total * 12 / 1e9:.2f}GB)")
    return "\n".join(lines)
