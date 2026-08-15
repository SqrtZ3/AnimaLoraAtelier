"""自适应 timestep 重采样（host 侧），对齐 `trainer/objective.py:358`
`AdaptiveTimestepSampler`。

## 为什么在 host 上

它是一个**反馈闭环**：上一步各图的裸 loss -> 按 t 分桶的 EMA -> 逐桶重采样权重
-> 这一步的 t。状态只有 `bins`(=8) 个标量，但每步要读上一步的结果 —— 放进 jit
就得把它做成 carry 并让 t 采样也进图，得不偿失。逐图 loss 回 host 是 [G] 个
float32（G≈6），一步几十字节。

## metric=slope（本配方用的这个）

逐 bin 维护快/慢两条 EMA，`factor ∝ (slow−fast)/slow` 的正部 —— 即该 bin 的
loss **分数下降速度**：

  * 还在学的 bin（slope 大）被抬到 `max_factor`；
  * 已饱和（slope≈0）或在回升（slope<0，过拟合那段）的 bin 落到 `min_factor`。

除以 slow 归一化消掉 bin 间 loss 量级差（低 t 的 loss 天生大，不该因此天然占优）。
与 level-based 的 raw/entropy_rate 取向正交：把采样预算投向"边际收益高"而非
"绝对 loss 高"的 t。没有任何 bin 在学时退回全 1（不重采样）。

## 与 PyTorch 侧的差异（诚实标注）

  * `counts` 的累加、EMA 的更新、`factors()` 的公式、`sample()` 的候选池与多项式
    抽样都是逐行照抄，数值上应当一致。
  * 但 **RNG 流不同**（numpy vs torch），所以同 seed 不会得到同一条轨迹。这与
    "算法不同"是两回事，做 A/B 时别把它读成算法差异。
  * `metric` 只实现 raw / slope / entropy_rate。highfreq / mixed 需要"逐图高频
    loss"这个额外信号（PyTorch 侧走 `per_sample_highfreq_loss`），尚未移植，
    构造期直接 fail-fast 而不是悄悄退回 raw。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

try:
    from . import flow as F
except ImportError:
    import flow as F

METRICS = ("raw", "slope", "entropy_rate")
_NOT_PORTED = ("highfreq", "mixed")


@dataclass(frozen=True)
class AdaptiveConfig:
    """与 yaml 的 `adaptive_timestep_*` 同名同义。"""
    enabled: bool = False
    metric: str = "slope"
    bins: int = 16
    ema_decay: float = 0.95
    slope_slow_decay: float = -1.0      # <0 -> 自动（比 fast 慢 4×）
    burn_in: int = 160
    min_factor: float = 0.5
    max_factor: float = 2.0
    base_mix: float = 0.25
    candidate_mult: int = 8
    low_noise_gate: bool = False
    gate_n: float = 3.0
    gate_c: float = 0.05
    highfreq_weight: float = 0.25       # 只在 highfreq/mixed 下有意义（未移植）

    def __post_init__(self):
        if self.metric in _NOT_PORTED:
            raise ValueError(
                f"adaptive_timestep_metric={self.metric!r} 需要逐图高频 loss 信号"
                f"（trainer/objective.py:per_sample_highfreq_loss），TPU 后端尚未移植。"
                f"可选 {METRICS}；想要它就先移植那个信号，别让它悄悄退回 raw。")
        if self.metric not in METRICS:
            raise ValueError(f"adaptive_timestep_metric 只支持 {METRICS}，"
                             f"得到 {self.metric!r}")


class AdaptiveTimestepSampler:
    """objective.py:358 的 numpy 复刻。`enabled=False` 时是纯透传。"""

    def __init__(self, cfg: AdaptiveConfig,
                 loss_weight_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        self.cfg = cfg
        self.bins = max(int(cfg.bins), 2)
        self.ema_decay = min(max(float(cfg.ema_decay), 0.0), 0.999)
        if cfg.slope_slow_decay is None or float(cfg.slope_slow_decay) < 0.0:
            slow = 1.0 - (1.0 - self.ema_decay) * 0.25
        else:
            slow = min(max(float(cfg.slope_slow_decay), 0.0), 0.9999)
        # slow 不得快于 fast（decay 越大越慢）
        self.slow_decay = max(slow, self.ema_decay)
        self.min_factor = max(float(cfg.min_factor), 1e-3)
        self.max_factor = max(float(cfg.max_factor), self.min_factor)
        self.base_mix = min(max(float(cfg.base_mix), 0.0), 1.0)
        self.candidate_mult = max(int(cfg.candidate_mult), 1)
        self.loss_ema = np.zeros(self.bins, np.float32)
        self.loss_ema_slow = np.zeros(self.bins, np.float32)
        self.counts = np.zeros(self.bins, np.int64)
        self.loss_weight_fn = loss_weight_fn

    # ── 反馈 ──────────────────────────────────────────────────────────────────
    @property
    def ready(self) -> bool:
        return self.cfg.enabled and bool((self.counts > 0).all())

    def _bin(self, t: np.ndarray) -> np.ndarray:
        return np.clip((np.asarray(t, np.float32) * self.bins).astype(np.int64),
                       0, self.bins - 1)

    def update(self, t: np.ndarray, per_image_loss: np.ndarray) -> None:
        """喂**裸**逐图 loss（objective.py:1071：slope/raw/entropy_rate 都透传裸值）。

        注意"裸"的含义：eisbach 权重、ΔFM 负项、multiscale 权重都**不能**乘进来
        —— 那些是"要不要让这张图影响参数"的旋钮，不是"这个 t 学得怎么样"的度量。
        PyTorch 侧同样是拿未经乘算的 `per_image_loss` 喂进来的。
        """
        if not self.cfg.enabled:
            return
        idx = self._bin(t)
        vals = np.asarray(per_image_loss, np.float32)
        for b in range(self.bins):
            m = idx == b
            if not m.any():
                continue
            v = float(vals[m].mean())
            if self.counts[b] == 0:
                self.loss_ema[b] = v
                self.loss_ema_slow[b] = v
            else:
                self.loss_ema[b] = (self.ema_decay * self.loss_ema[b]
                                    + (1.0 - self.ema_decay) * v)
                self.loss_ema_slow[b] = (self.slow_decay * self.loss_ema_slow[b]
                                         + (1.0 - self.slow_decay) * v)
            self.counts[b] += int(m.sum())

    def factors(self) -> np.ndarray:
        if not self.ready:
            return np.ones(self.bins, np.float32)
        losses = np.maximum(self.loss_ema, 1e-8)

        if self.cfg.metric == "slope":
            fast = np.maximum(self.loss_ema, 1e-8)
            slow = np.maximum(self.loss_ema_slow, 1e-8)
            s = np.maximum((slow - fast) / slow, 0.0)   # 饱和/回升 -> 0 -> min_factor
            denom = max(float(s.mean()), 1e-8)
            if denom <= 1e-8:                            # 没有 bin 在学 -> 不重采样
                return np.ones(self.bins, np.float32)
            return np.clip(s / denom, self.min_factor, self.max_factor).astype(np.float32)

        if self.cfg.metric == "entropy_rate":
            centers = (np.arange(self.bins, dtype=np.float32) + 0.5) / float(self.bins)
            rate = losses / np.maximum(centers ** 3, 1e-6)
            if self.loss_weight_fn is not None:
                rate = rate / np.maximum(self.loss_weight_fn(centers), 1e-6)
            if self.cfg.low_noise_gate:
                tn = centers ** self.cfg.gate_n
                rate = rate * (tn / (tn + float(self.cfg.gate_c) ** self.cfg.gate_n))
            rel = rate / max(float(rate.mean()), 1e-8)
        else:
            rel = losses / max(float(losses.mean()), 1e-8)
        return np.clip(rel, self.min_factor, self.max_factor).astype(np.float32)

    # ── 采样 ──────────────────────────────────────────────────────────────────
    def sample(self, rng: np.random.RandomState, n: int, fcfg: F.FlowConfig,
               global_step: int) -> np.ndarray:
        """返回 [n] 的 t，**已含** schedule_shift 与 t_range（= PyTorch 侧那两行的效果）。

        base 份额走分层采样；自适应份额从 `candidate_mult × count` 个候选里按逐 bin
        权重做**有放回**多项式抽样。burn-in 未过或还有空 bin 时整批退回 base。
        """
        base = F.base_t_np(rng, n, fcfg)
        c = self.cfg
        if (not c.enabled) or global_step < c.burn_in or not self.ready:
            return F.finish_t(base, fcfg, np).astype(np.float32)

        cnt = int(round(n * (1.0 - self.base_mix)))
        if cnt <= 0:
            return F.finish_t(base, fcfg, np).astype(np.float32)

        m = max(cnt * self.candidate_mult, cnt)
        cand = F.base_t_np(rng, m, fcfg, stratified=False)
        # **分桶用 schedule_shift 之后的值、被选中的却是之前的值**——照抄
        # objective.py:520-525。理由：分桶要与 update() 里的 t 同一口径（那边拿到的
        # 是训练用的最终 t），而返回值随后还会被统一施加一次 shift。
        w = self.factors()[self._bin(F.finish_t(cand, fcfg, np))]
        p = w / max(float(w.sum()), 1e-8)
        chosen = rng.choice(m, size=cnt, replace=True, p=p)
        adapted = cand[chosen]

        if cnt >= n:
            out = adapted[:n]
        else:
            out = base.copy()
            out[:cnt] = adapted
            out = out[rng.permutation(n)]
        return F.finish_t(out, fcfg, np).astype(np.float32)

    def summary(self) -> str:
        f = self.factors()
        gate = ""
        if self.cfg.metric == "entropy_rate":
            gate = (f" gate={'on' if self.cfg.low_noise_gate else 'off'}"
                    f"(n={self.cfg.gate_n} c={self.cfg.gate_c})")
        return (f"adaptive: metric={self.cfg.metric} bins={self.bins} "
                f"burn_in={self.cfg.burn_in} base_mix={self.base_mix} "
                f"factor {f.min():.2f}~{f.max():.2f} "
                f"覆盖 {int((self.counts > 0).sum())}/{self.bins} 桶{gate}")

    # ── 断点接棒 ──────────────────────────────────────────────────────────────
    def state(self) -> dict:
        """EMA 与 counts 必须随 checkpoint 一起存：丢了就要重新 burn-in，
        续训的前 200 步等于关掉了自适应（而且看不出来）。"""
        return {"loss_ema": self.loss_ema.tolist(),
                "loss_ema_slow": self.loss_ema_slow.tolist(),
                "counts": self.counts.tolist()}

    def load_state(self, d: dict) -> None:
        if not d:
            return
        n = self.bins
        get = lambda k, dt: np.asarray(d[k], dt) if k in d else None
        for name, key, dt in (("loss_ema", "loss_ema", np.float32),
                              ("loss_ema_slow", "loss_ema_slow", np.float32),
                              ("counts", "counts", np.int64)):
            v = get(key, dt)
            if v is None:
                continue
            if v.shape != (n,):
                raise ValueError(f"adaptive 状态 {key} 长度 {v.shape} != bins {n}；"
                                 f"改过 adaptive_timestep_bins 就不能直接接棒")
            setattr(self, name, v)


def anneal_mix_prob(base: float, end: float, step: int,
                    start: int, stop: int) -> float:
    """三峰路由概率的线性退火（objective.py:314）。

    `end < 0` 或 `stop <= start` 视为禁用（恒返 base）——yaml 默认就是这样。
    """
    if end < 0 or stop <= start:
        return float(base)
    prog = min(max((step - start) / float(stop - start), 0.0), 1.0)
    return float(base) + (float(end) - float(base)) * prog
