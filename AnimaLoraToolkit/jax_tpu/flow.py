"""Flow Matching 训练目标（JAX 侧），与 PyTorch `trainer/objective.py` 逐行对齐。

## 对齐依据（逐条指向 PyTorch 源）

  trainer/objective.py:1523  `noisy = (1 - t) * latent + t * noise`
  trainer/objective.py:1524  `target = noise - latent`（速度场，不是噪声预测）
  trainer/objective.py:287   基础 logit-normal：`u = sigmoid(randn)`
  trainer/objective.py:302   shift：`u = u*s / (1 + (s-1)*u)`，再 clamp 到 [1e-4, 1-1e-4]
  trainer/objective.py:291   logit_normal_low：把 shift 取倒数（推向低噪声/细节端）
  trainer/objective.py:215   logsnr：`λ ~ N(μ,σ)`，`t = sigmoid(-λ/2)`
  trainer/objective.py:218   mixed_logsnr_three：低/中/高三峰按 (p_low, p_high) 路由
  trainer/objective.py:249   laplace：λ 按 Laplace 逆 CDF，`t = 1/(1+exp(λ/2))`
  trainer/objective.py:307   schedule_shift：同 shift 公式再作用一次
  trainer/objective.py:329   apply_t_range：t_min/t_max 截断
  trainer/objective.py:340   sample_t_stratified：超采 ×32 后按分位层取
  trainer/objective.py:689   `_huber_delta_for_t`（constant / snr / sigma 三种调度）
  trainer/objective.py:713   `_huber_loss_map`（Huber 与 smooth-L1 的分支）
  trainer/objective.py:861   `masked_token_loss`（**逐图** token 均值）
  trainer/objective.py:642   Improved Immiscible 的 KNN 噪声选择
  trainer/objective.py:1081  compute_loss_weight 的各 scheme（本文件实现其中一部分）

## 一份采样实现，两个后端

`sample_t` 与 `sample_t_np` 共用 `_t_from_draws`：把"抽随机数"和"把随机数变成 t"
拆开，两边只有前半段不同。这样加一个 t 模式只需改一处 —— 两份平行实现在这种
"公式长得像但常数不同"的代码里迟早会漂，而漂了不报错。

**训练走 numpy 那条**（`sample_t_np`）：自适应重采样要拿上一步的逐图 loss 反馈，
状态在 host 上；t 只有几十个数，放 host 生成的开销可以忽略。

## 有意**不**移植的部分（这是 TPU 后端与 PyTorch 训练器的真实差距）

t 采样已覆盖 yaml 里用得到的全部模式；但 `objective.py` 里的 csflow / dpo / gaf /
leap / ncp、LossBinEMA、LWD 掩码等仍未移植，用到了要自己补，别以为它们已经在了。
loss weighting 只实现 none / detail_inv_t / cosmap / min_snr / logit_normal
这几个小 batch 场景常用的；`sigma_sqrt_sd3` 之类明确标注"仅大 batch"的没移。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

EPS = 1e-4          # 与 PyTorch 侧的 clamp 边界一致（objective.py:303）

T_MODES = ("logit_normal", "logit_normal_low", "logsnr", "uniform", "ushaped",
           "mixed_logsnr_three", "mixed_logsnr_low_high", "laplace",
           "mixed_uniform_low", "mode")
W_SCHEMES = ("none", "detail_inv_t", "cosmap", "min_snr", "logit_normal")
LOSS_TYPES = ("mse", "l1", "huber", "smooth_l1")
HUBER_SCHEDULES = ("constant", "snr", "sigma")

#: `ushaped` 在 PyTorch 侧的别名集合（objective.py:274），这里一并接受，
#: 免得同一份 yaml 在两个后端上一个能跑一个报错。
_ALIASES = {"mixed_logit_low_high": "ushaped", "u_shaped": "ushaped",
            "bimodal": "ushaped", "three_band": "mixed_logsnr_three",
            "ushaped_sf": "mixed_logsnr_low_high",
            "uniform_low_mix": "mixed_uniform_low"}


@dataclass(frozen=True)
class FlowConfig:
    """与 yaml 里的 `timestep_*` / `flow_shift` / `loss_*` 同名同义。"""
    t_mode: str = "logit_normal"
    flow_shift: float = 3.0
    schedule_shift: float = 1.0
    logsnr_mu: float = -6.0
    logsnr_sigma: float = 2.0
    #: 三峰/U 形的路由概率。ushaped 只用 low；three 用 low 与 high，其余进中噪峰。
    mix_low_prob: float = 0.5
    mix_high_prob: float = 0.25
    laplace_mu: float = 0.0
    laplace_b: float = 0.5
    t_min: float = 0.0
    t_max: float = 1.0
    stratified: bool = False
    stratified_oversample: int = 32
    # ── 路由概率的线性退火（objective.py:314 `anneal_mix_prob`）──────────────
    #: `*_end < 0` 或 `anneal_end <= anneal_start` = 禁用（yaml 默认就是这样）。
    #: **只影响 host 侧的 t 采样**，不进任何编译产物 —— 所以每步换一份
    #: FlowConfig 不会触发重编译（见 `at_step`）。
    mix_anneal_start: int = 0
    mix_anneal_end: int = 0
    mix_low_prob_end: float = -1.0
    mix_high_prob_end: float = -1.0
    # ── 主 loss ──────────────────────────────────────────────────────────────
    loss_type: str = "mse"
    huber_c: float = 0.1
    huber_schedule: str = "constant"
    huber_snr_clamp_max: float = 10.0
    weighting: str = "none"
    min_snr_gamma: float = 5.0
    detail_inv_t_min: float = 1.0
    detail_inv_t_max: float = 5.0
    # ── 噪声 ─────────────────────────────────────────────────────────────────
    immiscible_k: int = 1               # 1 = 关（标准高斯噪声）

    def __post_init__(self):
        object.__setattr__(self, "t_mode", _ALIASES.get(self.t_mode, self.t_mode))
        if self.t_mode not in T_MODES:
            raise ValueError(f"t_mode 只支持 {T_MODES}（TPU 后端是 PyTorch 侧的子集，"
                             f"见本文件 docstring），得到 {self.t_mode!r}")
        if self.weighting not in W_SCHEMES:
            raise ValueError(f"weighting 只支持 {W_SCHEMES}，得到 {self.weighting!r}")
        if self.loss_type not in LOSS_TYPES:
            raise ValueError(f"loss_type 只支持 {LOSS_TYPES}，得到 {self.loss_type!r}")
        if self.huber_schedule not in HUBER_SCHEDULES:
            raise ValueError(f"huber_schedule 只支持 {HUBER_SCHEDULES}，"
                             f"得到 {self.huber_schedule!r}")


# ── t 采样 ────────────────────────────────────────────────────────────────────
def _shift(u, s: float):
    """objective.py:302 的 shift 变换。s>1 推向高噪声端，s<1 推向低噪声端。"""
    s = float(s)
    return (u * s) / (1.0 + (s - 1.0) * u)


def _sigmoid(x, xp):
    return 1.0 / (1.0 + xp.exp(-x))


def _t_from_draws(d, cfg: FlowConfig, xp):
    """把一组标准随机数变成 t。`d` 的键见 `_draw_*`；`xp` 是 numpy 或 jnp。

    **不含** schedule_shift / t_range / clamp —— 那三步在 `_finish` 里，
    因为分层采样要在 clamp 之前排序。
    """
    m = cfg.t_mode
    if m == "uniform":
        return d["u"]
    if m == "logsnr":
        # FM CONST 调度 SNR=((1-t)/t)^2 => t = sigmoid(-λ/2)（objective.py:212-215）
        return _sigmoid(-0.5 * (cfg.logsnr_mu + cfg.logsnr_sigma * d["z_hi"]), xp)
    if m == "laplace":
        # λ = log-SNR 按 Laplace(μ,b) 逆 CDF；t = 1/(1+exp(λ/2))（objective.py:249）
        u = xp.clip(d["u"], EPS, 1.0 - EPS)
        sgn = xp.sign(0.5 - u)
        lam = cfg.laplace_mu - cfg.laplace_b * sgn * xp.log1p(-2.0 * xp.abs(u - 0.5))
        return 1.0 / (1.0 + xp.exp(0.5 * lam))

    lo = _shift(_sigmoid(d["z_lo"], xp), 1.0 / max(cfg.flow_shift, EPS))
    mid = _shift(_sigmoid(d["z_mid"], xp), cfg.flow_shift)
    if m == "logit_normal":
        return mid
    if m == "logit_normal_low":
        return lo
    if m == "mode":
        # SD3 mode sampling（objective.py:295）
        u = _sigmoid(d["z_mid"], xp)
        s = float(cfg.flow_shift)
        return 1 - u - s * (xp.cos(xp.pi * 0.5 * u) ** 2 - 1 + u)

    p_low = min(max(float(cfg.mix_low_prob), 0.0), 1.0)
    r = d["u_route"]
    if m == "ushaped":
        return xp.where(r < p_low, lo, mid)
    if m == "mixed_uniform_low":
        return xp.where(r < p_low, lo, d["u"])
    hi = _sigmoid(-0.5 * (cfg.logsnr_mu + cfg.logsnr_sigma * d["z_hi"]), xp)
    if m == "mixed_logsnr_low_high":
        return xp.where(r < p_low, lo, hi)
    # mixed_logsnr_three：低噪(细节) / 高噪(氛围) / 其余进中噪(结构)
    p_high = min(max(float(cfg.mix_high_prob), 0.0), 1.0 - p_low)
    return xp.where(r < p_low, lo, xp.where(r < p_low + p_high, hi, mid))


def _base(d, cfg: FlowConfig, xp):
    """≡ PyTorch 的 `sample_t(...)`：只到 clamp 为止，**不含** schedule_shift/t_range。

    这个边界不是随意划的：anima_train.py:3758-3792 里 schedule_shift 与 t_range 是
    在 `adaptive_ts.sample(...)` **返回之后**才施加的，而自适应重采样在内部要按
    schedule_shift **之后**的值分桶。两件事必须能分开调用，否则要么分桶用错了 t，
    要么 shift 被施加两次 —— 都不报错。
    """
    return xp.clip(_t_from_draws(d, cfg, xp), EPS, 1.0 - EPS)


def t_range_clip(t, cfg: FlowConfig, xp=jnp):
    """≡ `apply_t_range`（objective.py:329）：恒 clamp 到 [max(t_min,1e-4),
    min(t_max,1-1e-4)]，lo>hi 交换。finish_t 的最后一段与 K2 res_shift 之后
    共用 —— PyTorch 侧的顺序是 res_shift 之后再过它（anima_train.py:3962）。"""
    lo = max(float(cfg.t_min), EPS)
    hi = min(float(cfg.t_max), 1.0 - EPS)
    if lo > hi:
        lo, hi = hi, lo
    return xp.clip(t, lo, hi)


def finish_t(t, cfg: FlowConfig, xp=jnp):
    """≡ `apply_timestep_schedule_shift` + `apply_t_range`（objective.py:307/329）。"""
    if abs(cfg.schedule_shift - 1.0) > 1e-6 and cfg.schedule_shift > 0:
        t = _shift(t, cfg.schedule_shift)
    return t_range_clip(t, cfg, xp)


def _finish(t, cfg: FlowConfig, xp):
    return finish_t(t, cfg, xp)


def _draw_np(rng: np.random.RandomState, n: int):
    return {"z_lo": rng.randn(n).astype(np.float32),
            "z_mid": rng.randn(n).astype(np.float32),
            "z_hi": rng.randn(n).astype(np.float32),
            "u": rng.rand(n).astype(np.float32),
            "u_route": rng.rand(n).astype(np.float32)}


def _draw_jax(key, n: int):
    k = jax.random.split(key, 5)
    return {"z_lo": jax.random.normal(k[0], (n,), jnp.float32),
            "z_mid": jax.random.normal(k[1], (n,), jnp.float32),
            "z_hi": jax.random.normal(k[2], (n,), jnp.float32),
            "u": jax.random.uniform(k[3], (n,), jnp.float32),
            "u_route": jax.random.uniform(k[4], (n,), jnp.float32)}


def anneal_mix_prob(base: float, end: float, step: int,
                    start: int, stop: int) -> float:
    """objective.py:314 —— 路由概率的线性退火。`end<0` 或 `stop<=start` = 禁用。"""
    if end < 0 or stop <= start:
        return float(base)
    prog = min(max((step - start) / float(stop - start), 0.0), 1.0)
    return float(base) + (float(end) - float(base)) * prog


def at_step(cfg: FlowConfig, step: int) -> FlowConfig:
    """把第 `step` 步的退火后路由概率算进去，返回一份新的 FlowConfig。

    **只该喂给 host 侧的 t 采样**（`sample_t_np` / `sched.AdaptiveTimestepSampler`）。
    device 侧用到 FlowConfig 的只有 huber_delta / elem_loss / loss_weight，都不读
    mix 概率，所以这份逐步变化的配置不会进编译身份 —— 但也别把它塞进
    `TrainConfig` 再交给 `make_grad_fn`，那会让每步换一份 tcfg 从而每步重编译。

    退火关着时（yaml 默认 `*_prob_end: -1`）返回**原对象**，零开销、零行为差异。
    """
    from dataclasses import replace as _replace
    if cfg.mix_low_prob_end < 0 and cfg.mix_high_prob_end < 0:
        return cfg
    if cfg.mix_anneal_end <= cfg.mix_anneal_start:
        return cfg
    return _replace(
        cfg,
        mix_low_prob=anneal_mix_prob(cfg.mix_low_prob, cfg.mix_low_prob_end, step,
                                     cfg.mix_anneal_start, cfg.mix_anneal_end),
        mix_high_prob=anneal_mix_prob(cfg.mix_high_prob, cfg.mix_high_prob_end, step,
                                      cfg.mix_anneal_start, cfg.mix_anneal_end))


def sample_t(key, n: int, cfg: FlowConfig) -> jnp.ndarray:
    """采 n 个 timestep（JAX 后端）。返回 [n] fp32，落在 [EPS, 1-EPS]。

    **不含分层与自适应**（那两条都要 host 侧状态，见 `sample_t_np` / sched.py）。
    对拍脚本与不需要反馈闭环的场合用这条。
    """
    return _finish(_base(_draw_jax(key, n), cfg, jnp), cfg, jnp)


def base_t_np(rng: np.random.RandomState, n: int, cfg: FlowConfig,
              stratified: Optional[bool] = None) -> np.ndarray:
    """≡ PyTorch 的 `sample_t` / `sample_t_stratified`（**不含** schedule_shift/t_range）。

    分层（objective.py:340）：从同一分布超采 n×oversample 个候选并排序，再在每个
    分位层内随机取一个 —— batch 的 t 在分位空间均匀覆盖（消除"几个 t 全撞同一
    噪声段"的梯度噪声尖峰），边际分布不变。返回前随机重排，避免 batch 位置与
    t 大小相关。
    """
    strat = cfg.stratified if stratified is None else bool(stratified)
    if not strat:
        return _base(_draw_np(rng, n), cfg, np).astype(np.float32)
    over = max(int(cfg.stratified_oversample), 2)
    m = n * over
    cand = np.sort(_base(_draw_np(rng, m), cfg, np))
    per = m // n
    offs = rng.randint(0, per, size=n)
    picked = cand[np.arange(n) * per + offs]
    return picked[rng.permutation(n)].astype(np.float32)


def sample_t_np(rng: np.random.RandomState, n: int, cfg: FlowConfig) -> np.ndarray:
    """采 n 个 timestep（host 后端）。= base_t_np + schedule_shift + t_range。

    训练路径不直接用这条 —— 那边要经 `sched.AdaptiveTimestepSampler.sample`
    才能拿到自适应重采样。这条给 eval / 诊断 / 不开自适应的场合。
    """
    return finish_t(base_t_np(rng, n, cfg), cfg, np).astype(np.float32)


# ── krea2 分辨率感知 shift（官方 sampling.py 的训练侧等价）─────────────────────
def krea2_mu(image_tokens: float, min_res: int = 256, max_res: int = 1280,
             y1: float = 0.5, y2: float = 1.15, patch_px: int = 16) -> float:
    """trainer/model_family.py:387 的同公式（官方 mu 在两端点间线性内插）。

    image_tokens = (H/16)·(W/16)。1024² → mu≈0.906，exp(mu)≈2.48（"shift 2.5 @1024"）。
    """
    x1 = (min_res // patch_px) ** 2
    x2 = (max_res // patch_px) ** 2
    slope = (y2 - y1) / (x2 - x1)
    return slope * float(image_tokens) + (y1 - slope * x1)


def krea2_res_shift_np(t: np.ndarray, image_tokens: np.ndarray,
                       min_res: int = 256, max_res: int = 1280,
                       y1: float = 0.5, y2: float = 1.15) -> np.ndarray:
    """逐图施加 t' = αt/(1+(α−1)t)，α=exp(mu(该图 image token 数))。

    trainer/model_family.py:399 的 numpy 版。挂点与 schedule_shift 相同（任何
    timestep mode 采样之后、送进模型之前）；自适应采样器的分桶口径 = 施加后
    的最终 t（与 PyTorch 侧 anima_train.py 的顺序一致）。
    """
    t = np.asarray(t, np.float32)
    toks = np.asarray(image_tokens, np.float32).reshape(-1)
    if toks.shape[0] != t.reshape(-1).shape[0]:
        raise ValueError(f"image_tokens 数 {toks.shape[0]} != t 数 {t.reshape(-1).shape[0]}")
    x1 = (min_res // 16) ** 2
    x2 = (max_res // 16) ** 2
    slope = (y2 - y1) / (x2 - x1)
    alpha = np.exp(slope * toks + (y1 - slope * x1)).astype(np.float32)
    out = alpha * t / (1.0 + (alpha - 1.0) * t)
    return out.astype(np.float32)


# ── 加噪 / 目标 ───────────────────────────────────────────────────────────────
def make_noisy_and_target(latent: jnp.ndarray, noise: jnp.ndarray, t: jnp.ndarray):
    """objective.py:1523-1524。

      noisy  = (1 - t) * latent + t * noise
      target = noise - latent          <- **速度场**，不是噪声预测

    latent/noise: [..., N, C]（已 patchify 的 token）；t: 可广播到前面的形状。
    这两行写反或写成 eps-prediction 都不会报错，只会训出一个学不到东西的模型。
    """
    t = t.astype(jnp.float32)
    lat, noi = latent.astype(jnp.float32), noise.astype(jnp.float32)
    return (1.0 - t) * lat + t * noi, noi - lat


def immiscible_noise(key, latent: jnp.ndarray, weight: jnp.ndarray,
                     seg_sum, bcast, k: int) -> jnp.ndarray:
    """Improved Immiscible Diffusion 的 KNN 噪声选择（objective.py:642）。

    对每张图独立地采 k 个候选高斯噪声，选与该图 latent 的 L2 距离最小者。距离**只
    在真 token 上算**（`weight` 是 0/1 的 token 掩码）——把填充位算进去会让所有图
    的距离被同一堆零主导，选择退化成随机。

    `seg_sum(x) -> [G]` 与 `bcast([G]) -> token 形状` 由调用方给：打包布局是
    `segment_sum(·, mod_index)` / `take(·, mod_index)`，分桶布局是
    `sum(·, -1)` / `[:, None]`。这样两条布局共用同一份数学。
    k<=1 时直接返回标准高斯（零额外开销，逐 bit 等于原路径）。

    选择用 one-hot 加权和而不是 gather：k 只有 4，多算 3 次逐元素乘远比在两种
    布局上分别写对 `take_along_axis` 的索引形状可靠（那类形状错不报异常）。
    """
    if k is None or int(k) <= 1:
        return jax.random.normal(key, latent.shape, jnp.float32)
    k = int(k)
    cand = jax.random.normal(key, (k,) + latent.shape, jnp.float32)
    # [k, ...] 逐候选算逐图距离；只在真 token 上算（weight 是 0/1 掩码）
    d2 = jnp.stack([seg_sum(jnp.sum((cand[i] - latent) ** 2, axis=-1) * weight)
                    for i in range(k)])                # [k, G]
    oh = jax.nn.one_hot(jnp.argmin(d2, axis=0), k, dtype=jnp.float32)   # [G, k]
    return sum(cand[i] * bcast(oh[:, i])[..., None] for i in range(k))


# ── 主 loss ──────────────────────────────────────────────────────────────────
def huber_delta(t: jnp.ndarray, cfg: FlowConfig) -> jnp.ndarray:
    """objective.py:689 的 `_huber_delta_for_t`。返回逐图 δ [G]。

    `snr` 调度：δ = huber_c · clamp((1-t)/t, 0.1, snr_clamp_max)。低 t（细节区）
    得到更大的二次域（接近 L2），高 t 更接近 L1。**clamp 上界要紧**：默认 10 时
    低 t 几乎是纯 L2（对脏数据零保护），调小到 3 会让低噪区也保留部分 L1 鲁棒。
    """
    d = max(float(cfg.huber_c), 1e-8)
    if cfg.huber_schedule == "constant":
        return jnp.full(t.shape, d, jnp.float32)
    tc = jnp.clip(t.astype(jnp.float32), EPS, 1.0 - EPS)
    if cfg.huber_schedule == "snr":
        hi = max(float(cfg.huber_snr_clamp_max), 0.1 + 1e-6)
        return d * jnp.clip((1.0 - tc) / tc, 0.1, hi)
    return d * jnp.clip(tc, 0.1, 1.0)                  # sigma


def elem_loss(pred: jnp.ndarray, target: jnp.ndarray, delta_tok: jnp.ndarray,
              cfg: FlowConfig) -> jnp.ndarray:
    """逐 token 的 loss（已对通道维取均值）。pred/target [..., C]，返回 [...]。

    `delta_tok` 是**逐 token** 的 Huber δ（由逐图 δ gather/广播而来），形状能广播
    到 pred 的前导维。mse/l1 时它不被使用。
    """
    p, g = pred.astype(jnp.float32), target.astype(jnp.float32)
    if cfg.loss_type == "mse":
        m = (p - g) ** 2
    elif cfg.loss_type == "l1":
        m = jnp.abs(p - g)
    else:
        err = jnp.abs(p - g)
        dt = delta_tok[..., None]
        if cfg.loss_type == "smooth_l1":
            m = jnp.where(err < dt, 0.5 * err ** 2 / dt, err - 0.5 * dt)
        else:                                          # huber（objective.py:726）
            m = jnp.where(err < dt, 0.5 * err ** 2, dt * (err - 0.5 * dt))
    return jnp.mean(m, axis=-1)


def loss_weight(t: jnp.ndarray, cfg: FlowConfig) -> jnp.ndarray:
    """objective.py:1081 compute_loss_weight 的子集。返回 [n]。"""
    tc = jnp.clip(t.astype(jnp.float32), EPS, 1.0 - EPS)
    s = cfg.weighting
    if s == "none":
        return jnp.ones_like(tc)
    if s == "detail_inv_t":
        # 温和的细节端强化：w = 1/t，clamp 到 [lo, hi]（默认 [1,5]）
        lo, hi = sorted((float(cfg.detail_inv_t_min), float(cfg.detail_inv_t_max)))
        return jnp.clip(1.0 / tc, lo, hi)
    if s == "cosmap":
        return 2.0 / (jnp.pi * (1 - 2 * tc + 2 * tc ** 2))
    if s == "min_snr":
        snr = ((1 - tc) / tc) ** 2
        return jnp.minimum(float(cfg.min_snr_gamma) / snr, 1.0)
    # logit_normal：按采样密度的倒数去偏
    return jnp.maximum(tc * (1 - tc), EPS)


def per_image_loss(pred, target, mask, t, cfg: FlowConfig, seg_sum, delta_tok):
    """**逐图** loss 向量 [G]：每图先在自己的真 token 上取均值。

    这是 navit 路径的归约口径（objective.py:861 `masked_token_loss` + 训练循环
    :4109 的逐图组装），与"全局 token 加权平均"**不是同一个东西**：后者让 token
    多的大图按面积占权重，一张 16k token 的图能顶 4 张 4k 的。NaViT 论文与本仓库
    的 PyTorch 实现都是**逐图等权**。

    没有真 token 的段（FFD 装箱余量）分母为 0，这里返回 0；调用方靠
    `image_weight` 把它排除在平均之外，别让它把 loss 稀释掉。
    """
    se = elem_loss(pred, target, delta_tok, cfg)       # [ΣN] 或 [G, L]
    num = seg_sum(se * mask)
    den = seg_sum(mask)
    return num / jnp.maximum(den, 1.0), den


def weighted_mean(per_image: jnp.ndarray, t: jnp.ndarray, valid: jnp.ndarray,
                  cfg: FlowConfig) -> jnp.ndarray:
    """按 w(t) 加权的逐图平均（objective.py:1109 `apply_loss_weighting`）。

    `valid` = 该段是否有真 token（1/0）。scheme=none 时退化成"真实图的算术平均"。
    """
    w = loss_weight(t, cfg) * valid.astype(jnp.float32)
    return jnp.sum(per_image * w) / jnp.maximum(jnp.sum(w), 1e-6)


# ── 兼容旧接口（对拍脚本仍在用）────────────────────────────────────────────────
def masked_token_loss(pred: jnp.ndarray, target: jnp.ndarray,
                      token_weight: jnp.ndarray) -> jnp.ndarray:
    """逐 token 加权 MSE（**全局**归约口径）。保留给对拍脚本与消融对照。

    训练路径请用 `per_image_loss` + `weighted_mean` —— 两者在"每图 token 数相同"
    时相等，不同时不等（见 `per_image_loss` 的说明）。
    """
    se = jnp.mean((pred.astype(jnp.float32) - target.astype(jnp.float32)) ** 2, axis=-1)
    w = token_weight.astype(jnp.float32)
    return jnp.sum(se * w) / jnp.maximum(jnp.sum(w), 1.0)


def token_weights(loss_mask: jnp.ndarray, mod_index: jnp.ndarray,
                  t: jnp.ndarray, cfg: FlowConfig) -> jnp.ndarray:
    """打包布局：把逐图的 w(t) gather 到逐 token，再乘上 loss_mask。返回 [ΣN]。"""
    return loss_mask.astype(jnp.float32) * jnp.take(loss_weight(t, cfg), mod_index)


def token_weights_ragged(loss_mask: jnp.ndarray, t: jnp.ndarray,
                         cfg: FlowConfig) -> jnp.ndarray:
    """分桶布局：loss_mask [G, L] × w(t) [G] **广播**（无 gather）。返回 [G, L]。"""
    return loss_mask.astype(jnp.float32) * loss_weight(t, cfg)[:, None]
