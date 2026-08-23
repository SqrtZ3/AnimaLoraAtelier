"""自适应 timestep 重采样（host 侧），对齐 `trainer/objective.py`
`AdaptiveTimestepSampler`（:358，**行号可能漂移，以符号名为准**）。

## 为什么在 host 上

它是一个**反馈闭环**：上一步各图的裸 loss -> 按 t 分桶的 EMA -> 逐桶重采样权重
-> 这一步的 t。状态只有 `bins` 个标量（`AdaptiveConfig.bins` 默认 16，与 PyTorch
侧 argparse 的 `adaptive_timestep_bins` 同值；本仓库的 TPU yaml 多用 8），但每步
要读上一步的结果 —— 放进 jit 就得把它做成 carry 并让 t 采样也进图，得不偿失。
逐图 loss 回 host 是 [G] 个 float32（G≈6），一步几十字节。

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
        # metric 只在**开着的时候**校验：`adaptive_timestep: false` +
        # `metric: highfreq` 在 GPU 侧是合法的死配置（PyTorch 侧 metric 的白名单
        # 校验在 `AdaptiveTimestepSampler.__init__` 里，但 enabled=False 时那条路
        # 上的值根本不被读），TPU 侧要是照样 raise，就变成"关着的功能也拦人"。
        if not self.enabled:
            return
        if self.metric in _NOT_PORTED:
            raise ValueError(
                f"adaptive_timestep_metric={self.metric!r} 需要逐图高频 loss 信号"
                f"（trainer/objective.py `per_sample_highfreq_loss`），TPU 后端尚未移植。"
                f"可选 {METRICS}；想要它就先移植那个信号，别让它悄悄退回 raw。")
        if self.metric not in METRICS:
            raise ValueError(f"adaptive_timestep_metric 只支持 {METRICS}，"
                             f"得到 {self.metric!r}")


class AdaptiveTimestepSampler:
    """`objective.py` `AdaptiveTimestepSampler` 的 numpy 复刻（:358，行号可能漂移，
    以符号名为准）。`enabled=False` 时是纯透传。"""

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
        # 闸门参数的下界与 PyTorch 侧同（`AdaptiveTimestepSampler.__init__`：
        # `gate_n = max(float(gate_n or 0), 1e-3)` / `gate_c = max(..., 1e-6)`）。
        # 少了这两个 clamp 会静默坏掉而不报错：gate_c=0 -> c^n=0 -> gate ≡ 1（闸门
        # 等于没开）；gate_n<0 -> t^n 单调**递减** -> 闸门方向反过来（本该压低噪端，
        # 变成压高噪端）。
        self.gate_n = max(float(cfg.gate_n or 0.0), 1e-3)
        self.gate_c = max(float(cfg.gate_c or 0.0), 1e-6)
        self.loss_ema = np.zeros(self.bins, np.float32)
        self.loss_ema_slow = np.zeros(self.bins, np.float32)
        self.counts = np.zeros(self.bins, np.int64)
        self.loss_weight_fn = loss_weight_fn
        # entropy_rate 的公式是 π(σ) ∝ ρ(σ)/w(σ)（论文 Eq.16），那个 /w(t) 不是
        # 可选项。PyTorch 侧 anima_train.py 构造 `_loss_weight_fn` 闭包后**必传**；
        # 这里若为 None，那一除会被静默跳过 —— 实测（loss ∝ t³、weighting=min_snr
        # gamma=5）最低 t 桶采样权重差 4.1×、其余桶 ~9×，也就是"同一份 yaml 在两个
        # 后端上不是同一个实验"，而且没有任何迹象。所以宁可构造期就报错。
        if cfg.enabled and cfg.metric == "entropy_rate" and loss_weight_fn is None:
            raise ValueError(
                "adaptive_timestep_metric=entropy_rate 必须传 loss_weight_fn："
                "该 metric 的定义是 π(σ) ∝ ρ(σ)/w(σ)（objective.py `factors()` 的 "
                "entropy_rate 分支除以 `self.loss_weight_fn(bin_centers)`），"
                "少了 /w(t) 那一除，t 的分布与 GPU 侧同 yaml **不是同一个实验**"
                "（实测最低 t 桶权重差 4.1×，其余桶约 9×），且不会有任何报错。"
                "接线见 run_train.py："
                "loss_weight_fn=lambda c: np.asarray(F.loss_weight(jnp.asarray(c), rc.tcfg.flow))。")

    # ── 反馈 ──────────────────────────────────────────────────────────────────
    @property
    def ready(self) -> bool:
        return self.cfg.enabled and bool((self.counts > 0).all())

    def _bin(self, t: np.ndarray) -> np.ndarray:
        return np.clip((np.asarray(t, np.float32) * self.bins).astype(np.int64),
                       0, self.bins - 1)

    def update(self, t: np.ndarray, per_image_loss: np.ndarray) -> None:
        """喂**裸**逐图 loss（objective.py `adaptive_timestep_metric_signal`：
        slope/raw/entropy_rate 都透传裸值；:1059 附近，行号可能漂移，以符号名为准）。

        注意"裸"的含义：eisbach 权重、ΔFM 负项、multiscale 权重都**不能**乘进来
        —— 那些是"要不要让这张图影响参数"的旋钮，不是"这个 t 学得怎么样"的度量。
        PyTorch 侧同样是拿未经乘算的 `per_image_loss` 喂进来的。

        Krea2 口径说明（与 PyTorch 侧同构）：K2 训练喂进来的是 res_shift **后**
        的最终 t（模型实际经历的 t），而 sample() 里候选分桶用的是 res_shift
        **前**的值 —— 两者错一个逐图不同的单调映射。这是架构性的：采样发生在
        t 与图的配对之前，候选分桶拿不到各图的分辨率。错位方向安全（统计上
        偏向多练难区域，不会错训）；若哪天要消它，得让 sample 知道逐位置
        tokens 并重做配对语义（改动大、且要动 PyTorch 侧，暂未做）。
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
            # /w(t)：构造期已保证 loss_weight_fn 不是 None（见 __init__ 的 fail-fast），
            # 这里不再有"静默跳过"的分支。
            rate = rate / np.maximum(self.loss_weight_fn(centers), 1e-6)
            if self.cfg.low_noise_gate:
                tn = centers ** self.gate_n
                rate = rate * (tn / (tn + self.gate_c ** self.gate_n))
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
        # objective.py `AdaptiveTimestepSampler.sample` 的 `candidates_final`
        # （:520 附近，行号可能漂移，以符号名为准）。理由：分桶要与 update() 里的 t
        # 同一口径（那边拿到的是训练用的最终 t），而返回值随后还会被统一施加一次 shift。
        #
        # 这里必须是 `schedule_shift_only` 而**不是** `finish_t`：PyTorch 侧只施加
        # schedule_shift，t_range 是在 `adaptive_ts.sample()` 返回**之后**才作用的
        # （anima_train.py 里 sample -> apply_timestep_schedule_shift -> (res_shift)
        # -> apply_t_range 这条链）。多套一层 t_range clip 会把候选挤进边界桶：
        # 实测 t_min=0.3/t_max=0.7、bins=8、20000 候选时
        #     TPU（clip 过）: [   0    0 7440 2530 2480 7550    0    0]
        #     PyTorch      : [2486 2495 2459 2530 2480 2506 2502 2542]
        # 于是桶 0/1/6/7 的 factors 永远索引不到、桶 2/5 拿到 3× 质量；而 update()
        # 那侧喂的是被 clip 的最终 t，counts 必有空桶 -> `ready` 恒 False -> 整个
        # 自适应静默退回 base。两种失效模式二选一，都不报错。
        w = self.factors()[self._bin(F.schedule_shift_only(cand, fcfg, np))]
        # p 升到 float64 再归一化：rng.choice 会校验 |sum(p)-1| <= atol(1.49e-8)，
        # 而 float32 下 m=数千时实测 |sum(p)-1| 最大 1.3e-7 —— 现在没炸只是因为
        # numpy 内部用了 Kahan 求和，属于贴着容差跑。float64 把余量拉开 ~1e9 倍。
        w64 = np.asarray(w, np.float64)
        p = w64 / max(float(w64.sum()), 1e-8)
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


