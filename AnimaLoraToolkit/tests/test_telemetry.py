"""训练遥测总线 CPU 单测（无 model/VAE 依赖，注入合成张量，全程图盲）。

覆盖：
  L1 radial_band_power —— 频带能量分离（低频 vs 棋盘高频）
  L2 FreqEvalProbe     —— CSV schema + 逐 t-bin 斜率符号
  O1/O2/O3 optimizer_report —— 真 SOAPScheduleFree state 上的 sf_lag/kappa/update_norm/CSV
  O2 gg_anisotropy     —— 各向同性 κ≈1 / 各向异性 κ≫1
  C1 lokr_spectrum     —— Kronecker 谱恒等式 vs 暴力 kron SVD（核心正确性）
  C1 spectrum_stats / lokr_capacity_report —— 有效秩 + 逐 block CSV
"""

import math

import torch
import torch.nn as nn

from trainer.telemetry import (
    _block_of,
    _linfit_slope,
    radial_band_power,
    FreqEvalProbe,
    gg_anisotropy,
    optimizer_report,
    lokr_spectrum,
    spectrum_stats,
    lokr_capacity_report,
    adaptive_bin_report,
)
from utils.soap_optimizer import SOAPScheduleFree
from trainer.objective import AdaptiveTimestepSampler, adaptive_timestep_metric_signal


# ───────────────────────── 小工具 ─────────────────────────

def test_block_of_parses_block_id():
    assert _block_of("net.blocks.14.self_attn.q_proj") == "14"
    assert _block_of("blocks.0.mlp.fc1") == "0"
    assert _block_of("final_layer.linear") == "other"
    assert _block_of("x_embedder.proj") == "other"
    assert _block_of("") == "other"


def test_linfit_slope_sign():
    steps = [0.0, 40.0, 80.0, 120.0]
    assert _linfit_slope(steps, [1.0, 0.8, 0.6, 0.4]) < 0       # 单调降
    assert abs(_linfit_slope(steps, [0.5, 0.5, 0.5, 0.5])) < 1e-9  # 平
    assert _linfit_slope([1.0], [0.5]) == 0.0                   # 单点退化


# ───────────────────────── L1 ─────────────────────────

def test_radial_band_power_shape_and_nonneg():
    err = torch.randn(2, 4, 16, 16)
    bands = radial_band_power(err, n_bands=3)
    assert len(bands) == 3
    assert all(b >= 0 for b in bands)


def test_radial_band_power_separates_low_vs_high():
    H = W = 32
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")

    # 低频：单一低频正弦（沿 x 1 个周期）→ 能量应集中在最低带
    low = torch.sin(2 * math.pi * xx.float() / W).view(1, 1, H, W)
    lb = radial_band_power(low, n_bands=3)
    assert lb[0] > lb[2], f"低频信号应低带占优: {lb}"

    # 高频：棋盘（Nyquist）→ 能量应集中在最高带
    checker = (((yy + xx) % 2) * 2 - 1).float().view(1, 1, H, W)
    hb = radial_band_power(checker, n_bands=3)
    assert hb[2] > hb[0], f"棋盘高频应高带占优: {hb}"


# ───────────────────────── L2 ─────────────────────────

def test_freq_eval_probe_csv_schema(tmp_path):
    t_grid = [0.1, 0.5, 0.9]
    probe = FreqEvalProbe(tmp_path, t_grid, n_bands=3, slope_window=4)
    per_t_bands = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    probe.record(40, per_t_bands, per_t_loss=[0.3, 0.1, 0.12])
    probe.record(80, per_t_bands, per_t_loss=[0.25, 0.09, 0.10])

    freq_csv = (tmp_path / "eval_freq_loss.csv").read_text(encoding="utf-8").strip().splitlines()
    assert freq_csv[0] == "step,t0.1_fL,t0.1_fM,t0.1_fH,t0.5_fL,t0.5_fM,t0.5_fH,t0.9_fL,t0.9_fM,t0.9_fH"
    assert len(freq_csv) == 3  # header + 2 行
    assert freq_csv[1].startswith("40,1,2,3,4,5,6,7,8,9")

    slope_csv = (tmp_path / "eval_slope.csv").read_text(encoding="utf-8").strip().splitlines()
    assert slope_csv[0] == "step,t0.1_slope,t0.5_slope,t0.9_slope"
    # 第二行（step 80）每个 t-bin 都在降 → 斜率为负
    vals = [float(x) for x in slope_csv[2].split(",")[1:]]
    assert all(v < 0 for v in vals), f"下降序列斜率应为负: {vals}"


# ───────────────────────── O2 gg_anisotropy ─────────────────────────

def test_gg_anisotropy_isotropic_vs_anisotropic():
    iso = torch.eye(8) * 3.0
    k_iso, rr_iso = gg_anisotropy([iso])
    assert abs(k_iso - 1.0) < 1e-4, f"各向同性 κ 应≈1: {k_iso}"
    assert abs(rr_iso - 1.0) < 1e-4, f"各向同性有效秩比应≈1: {rr_iso}"

    aniso = torch.diag(torch.tensor([100.0, 1.0, 0.01]))
    k_an, rr_an = gg_anisotropy([aniso])
    assert k_an > 1e3, f"各向异性 κ 应≫1: {k_an}"
    assert rr_an < 0.8, f"各向异性有效秩比应明显<1: {rr_an}"

    assert math.isnan(gg_anisotropy([None])[0])


# ───────────────────────── O1/O2/O3 optimizer_report ─────────────────────────

def test_optimizer_report_on_soap_sf(tmp_path):
    torch.manual_seed(0)
    p = nn.Parameter(torch.randn(16, 8))
    opt = SOAPScheduleFree([p], lr=1e-2, precondition_frequency=1)
    opt._telemetry = True   # 让 step 把 ‖update‖ stash 进 state['_upd_norm']

    for _ in range(5):
        opt.zero_grad()
        p.grad = torch.randn(16, 8)
        opt.step()

    # grad 仍在（未 zero_grad）时调用，模拟训练循环里 step 后 / zero_grad 前的挂点
    p.grad = torch.randn(16, 8)
    agg = optimizer_report(opt, tmp_path, step=200,
                           named_params=[("net.blocks.7.mlp.fc1", p)])

    assert math.isfinite(agg["sf_lag"]), "SF-lag 应有限"
    assert math.isfinite(agg["kappa"]) and agg["kappa"] >= 1.0, "κ 应≥1"
    assert math.isfinite(agg["trust_ratio"]), "trust ratio 应有限（依赖 _upd_norm stash）"

    lines = (tmp_path / "telemetry_optimizer.csv").read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "step,scope,grad_norm,update_norm,trust_ratio,sf_lag,kappa,eff_rank_ratio"
    scopes = [ln.split(",")[1] for ln in lines[1:]]
    assert "7" in scopes and "all" in scopes, f"应有 block 7 行 + all 聚合行: {scopes}"


def test_soap_sf_telemetry_flag_off_by_default():
    p = nn.Parameter(torch.randn(8, 4))
    opt = SOAPScheduleFree([p], lr=1e-2)
    p.grad = torch.randn(8, 4)
    opt.step()
    # 默认不 stash（零热路径成本）
    assert "_upd_norm" not in opt.state[p]


# ───────────────────────── C1 lokr_spectrum / Kronecker 恒等式 ─────────────────────────

def test_lokr_spectrum_matches_bruteforce_kron():
    torch.manual_seed(1)
    factor, out_dim, in_dim, rank = 3, 4, 5, 2
    w1 = torch.randn(factor, factor)
    w2_a = torch.randn(out_dim, rank)
    w2_b = torch.randn(rank, in_dim)
    scaling = 0.75

    # 暴力：真的形成 ΔW = scaling·kron(w1, w2_a@w2_b) 再 SVD
    w2 = w2_a @ w2_b
    dW = scaling * torch.kron(w1, w2)
    sv_brute = torch.linalg.svdvals(dW)

    sv_fast = lokr_spectrum(w1, w2_a, w2_b, scaling=scaling)

    # frob（Σσ²）必须精确一致
    assert math.isclose(float(sv_brute.square().sum()), float(sv_fast.square().sum()), rel_tol=1e-5)
    # 非零奇异值个数 = rank(w1)·rank(w2) = 3·2 = 6，top-6 应一致
    top = min(6, sv_brute.numel(), sv_fast.numel())
    assert torch.allclose(sv_brute[:top], sv_fast[:top], atol=1e-4), \
        f"\nbrute={sv_brute[:top]}\nfast ={sv_fast[:top]}"

    # 有效秩两路一致（padding 的零不影响谱熵）
    _, er_brute, _ = spectrum_stats(sv_brute)
    _, er_fast, _ = spectrum_stats(sv_fast)
    assert math.isclose(er_brute, er_fast, rel_tol=1e-4)


def test_spectrum_stats_rank1_vs_full():
    # rank-1 谱 → 有效秩≈1
    s1 = torch.tensor([5.0, 0.0, 0.0, 0.0])
    _, er1, ratio1 = spectrum_stats(s1)
    assert abs(er1 - 1.0) < 1e-3 and ratio1 < 0.3

    # 均匀满谱 → 有效秩≈维数
    sfull = torch.ones(4)
    frob, erf, ratiof = spectrum_stats(sfull)
    assert abs(erf - 4.0) < 1e-3 and abs(ratiof - 1.0) < 1e-3
    assert math.isclose(frob, 2.0, rel_tol=1e-5)  # sqrt(4)


# ───────────────────────── C1 lokr_capacity_report ─────────────────────────

class _FakeLoKr(nn.Module):
    def __init__(self, factor, out_dim, in_dim, rank, scaling):
        super().__init__()
        self.lokr_w1 = nn.Parameter(torch.randn(factor, factor) * 0.1)
        self.lokr_w2_a = nn.Parameter(torch.randn(out_dim, rank))
        self.lokr_w2_b = nn.Parameter(torch.randn(rank, in_dim))
        self.scaling = scaling


def test_lokr_capacity_report_writes_per_block(tmp_path):
    root = nn.Module()
    root.blocks = nn.ModuleDict({
        "7": _FakeLoKr(3, 8, 6, 4, 1.0),
        "21": _FakeLoKr(3, 8, 6, 2, 1.0),
    })
    summary = lokr_capacity_report(root, tmp_path, step=120)

    assert set(summary.keys()) == {"7", "21"}
    lines = (tmp_path / "telemetry_capacity.csv").read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "step,block,module,frob,eff_rank,eff_rank_ratio"
    blocks = sorted(ln.split(",")[1] for ln in lines[1:])
    assert blocks == ["21", "7"]
    # 所有有效秩比 ∈ (0, 1]
    for ln in lines[1:]:
        ratio = float(ln.split(",")[-1])
        assert 0.0 < ratio <= 1.0 + 1e-6


# ───────────────────────── L3 adaptive_bin_report ─────────────────────────

def test_adaptive_bin_report_disabled_sampler_is_noop(tmp_path):
    # disabled sampler → 直接返回，不写 CSV
    s = AdaptiveTimestepSampler(enabled=False, bins=8)
    out = adaptive_bin_report(s, tmp_path, step=40)
    assert not (tmp_path / "telemetry_adaptive.csv").exists()
    assert out["ready"] is False


def test_adaptive_bin_report_not_ready_writes_all_ones(tmp_path):
    # enabled 但未 ready（counts 有空桶）→ factors() 全 1，仍写入便于看 burn-in 结束
    s = AdaptiveTimestepSampler(enabled=True, bins=8, metric="entropy_rate")
    assert not s.ready
    adaptive_bin_report(s, tmp_path, step=40)
    lines = (tmp_path / "telemetry_adaptive.csv").read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "step,ready,factor_min,factor_max,f0,f1,f2,f3,f4,f5,f6,f7"
    vals = lines[1].split(",")
    assert vals[0] == "40" and vals[1] == "0"   # ready=0
    factors = [float(x) for x in vals[4:]]
    assert all(abs(f - 1.0) < 1e-6 for f in factors), f"未 ready 应全 1: {factors}"


def test_adaptive_bin_report_ready_clamps_to_bounds(tmp_path):
    # 喂满每个 bin 的 loss → ready=True；factor 应被 clamp 到 [min_factor, max_factor]
    s = AdaptiveTimestepSampler(
        enabled=True, bins=4, metric="raw",
        min_factor=0.5, max_factor=2.5, base_mix=0.0)
    t = torch.tensor([0.125, 0.375, 0.625, 0.875])
    # 各 bin 给差异极大的 loss：高 loss bin 应被抬到 max_factor
    losses = torch.tensor([0.01, 0.01, 10.0, 0.01])
    for _ in range(3):   # 多次 update 让 EMA 稳定
        s.update(t, losses)
    assert s.ready
    out = adaptive_bin_report(s, tmp_path, step=80)
    assert out["ready"] is True
    assert abs(out["min"] - 0.5) < 1e-4 and abs(out["max"] - 2.5) < 1e-4, \
        f"factor 应被 clamp 到 [0.5, 2.5]: {out}"
    lines = (tmp_path / "telemetry_adaptive.csv").read_text(encoding="utf-8").strip().splitlines()
    assert lines[1].split(",")[1] == "1"   # ready=1


# ───────────────────────── L3 adaptive_bin_report ─────────────────────────

def test_adaptive_bin_report_disabled_is_noop(tmp_path):
    # adaptive 关闭 → 不写 CSV，返回 ready=False
    from trainer.objective import AdaptiveTimestepSampler
    sampler = AdaptiveTimestepSampler(enabled=False, bins=8)
    out = adaptive_bin_report(sampler, tmp_path, step=40)
    assert out["ready"] is False
    assert not (tmp_path / "telemetry_adaptive.csv").exists()


def test_adaptive_bin_report_burn_in_all_ones(tmp_path):
    # ready=False（仍有空桶）→ factors() 全 1，但仍写入（看 burn-in 何时结束）
    from trainer.objective import AdaptiveTimestepSampler
    sampler = AdaptiveTimestepSampler(enabled=True, bins=4, metric="raw")
    # 没调过 update → counts 全 0 → ready=False
    out = adaptive_bin_report(sampler, tmp_path, step=40)
    assert out["ready"] is False
    lines = (tmp_path / "telemetry_adaptive.csv").read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "step,ready,factor_min,factor_max,f0,f1,f2,f3"
    row = lines[1].split(",")
    assert row[0] == "40" and row[1] == "0"
    factors = [float(x) for x in row[4:]]
    assert all(abs(f - 1.0) < 1e-6 for f in factors), f"burn-in 应全 1: {factors}"


def test_adaptive_bin_report_ready_writes_factors(tmp_path):
    # 喂满所有 bin → ready=True，factor 被 clamp 到 [min_factor, max_factor]
    from trainer.objective import AdaptiveTimestepSampler
    sampler = AdaptiveTimestepSampler(
        enabled=True, bins=4, metric="raw",
        min_factor=0.5, max_factor=2.5, ema_decay=1.0,
    )
    # 每个 bin 各喂一个样本，制造不均匀 loss → factor 非全 1
    t = torch.tensor([0.05, 0.30, 0.60, 0.90])
    per_sample = torch.tensor([10.0, 1.0, 1.0, 1.0])  # bin0 高 loss
    sampler.update(t, per_sample)
    assert sampler.ready

    out = adaptive_bin_report(sampler, tmp_path, step=80)
    assert out["ready"] is True
    assert out["min"] >= 0.5 - 1e-6 and out["max"] <= 2.5 + 1e-6
    # bin0（高 loss）的 factor 应最大
    lines = (tmp_path / "telemetry_adaptive.csv").read_text(encoding="utf-8").strip().splitlines()
    row = lines[-1].split(",")
    assert row[1] == "1"  # ready
    factors = [float(x) for x in row[4:]]
    assert factors[0] == max(factors), f"高 loss bin 应得最大 factor: {factors}"


# ───────────────────────── slope metric（斜率感知重采样）─────────────────────────

def _ready_slope_sampler(bins, fast, slow, **kw):
    """构造一个已 ready 的 slope sampler，直接注入 fast/slow EMA 做 factors() 纯逻辑断言。"""
    s = AdaptiveTimestepSampler(enabled=True, bins=bins, metric="slope",
                                min_factor=0.5, max_factor=2.5, **kw)
    s.counts = torch.ones(bins, dtype=torch.long)          # 标记 ready
    s.loss_ema = torch.tensor(fast, dtype=torch.float32)
    s.loss_ema_slow = torch.tensor(slow, dtype=torch.float32)
    return s


def test_slope_invests_in_learning_bin():
    # 只有 bin1 在学（slow>fast）；饱和 bin → min，学习 bin → max
    s = _ready_slope_sampler(4, fast=[1.0, 1.0, 1.0, 0.5], slow=[1.0, 2.0, 1.0, 0.5])
    f = s.factors().tolist()
    assert abs(f[1] - 2.5) < 1e-5, f"还在学的 bin 应到 max_factor: {f}"
    assert all(abs(f[i] - 0.5) < 1e-5 for i in (0, 2, 3)), f"饱和 bin 应落 min_factor: {f}"


def test_slope_is_level_free():
    # bin0 绝对 loss 高但平（slow=fast=10）；bin1 绝对 loss 低但在降 → bin1 应胜出，
    # 证明 slope 不被 loss 量级带偏（raw/entropy_rate 会把 bin0 抬最高）。
    s = _ready_slope_sampler(2, fast=[10.0, 0.1], slow=[10.0, 0.2])
    f = s.factors().tolist()
    assert f[1] > f[0], f"低 loss 但在学的 bin 应赢过高 loss 但平的 bin: {f}"
    assert abs(f[0] - 0.5) < 1e-5 and f[1] > 1.5, f"{f}"


def test_slope_demotes_worsening_bin():
    # bin1 loss 在回升（fast>slow → s<0）→ 与饱和同等落 min；bin2 在学 → max
    s = _ready_slope_sampler(3, fast=[1.0, 1.5, 0.5], slow=[1.0, 1.0, 1.0])
    f = s.factors().tolist()
    assert abs(f[1] - 0.5) < 1e-5, f"回升(过拟合)的 bin 应落 min_factor: {f}"
    assert abs(f[2] - 2.5) < 1e-5, f"在学的 bin 应到 max_factor: {f}"


def test_slope_all_saturated_returns_ones():
    # 所有 bin slow==fast → 无人在学 → 不重采样（全 1，回退 base 分布）
    s = _ready_slope_sampler(4, fast=[1.0, 2.0, 0.5, 3.0], slow=[1.0, 2.0, 0.5, 3.0])
    f = s.factors().tolist()
    assert all(abs(v - 1.0) < 1e-6 for v in f), f"全饱和应返回全 1: {f}"


def test_slope_does_not_affect_raw_metric():
    # default-off 等价：metric=raw 时即便 slow EMA 不同，factors 仍走 level-based（用 loss_ema）
    s = AdaptiveTimestepSampler(enabled=True, bins=4, metric="raw",
                                min_factor=0.5, max_factor=2.5)
    s.counts = torch.ones(4, dtype=torch.long)
    s.loss_ema = torch.tensor([1.0, 1.0, 10.0, 1.0])      # bin2 绝对 loss 最高
    s.loss_ema_slow = torch.tensor([5.0, 5.0, 5.0, 5.0])  # 故意与 fast 不一致
    f = s.factors().tolist()
    assert f[2] == max(f) and abs(f[2] - 2.5) < 1e-5, f"raw 应按 level：高 loss bin 最大: {f}"


def test_slope_slow_decay_auto_and_clamp():
    # 自动派生：slow = 1-(1-fast)*0.25；显式过快(<fast)被夹到 fast
    s_auto = AdaptiveTimestepSampler(enabled=True, bins=4, metric="slope", ema_decay=0.95)
    assert abs(s_auto.slope_slow_decay - 0.9875) < 1e-9
    s_clamp = AdaptiveTimestepSampler(enabled=True, bins=4, metric="slope",
                                      ema_decay=0.95, slope_slow_decay=0.90)
    assert abs(s_clamp.slope_slow_decay - 0.95) < 1e-9, "slow 不得快于 fast"


def test_metric_signal_accepts_slope():
    # 回归：非 fit_packed 训练路径经 adaptive_timestep_metric_signal 取信号；
    # slope 必须被接受并透传裸 loss（曾因漏加白名单而 ValueError）。
    per_sample = torch.tensor([0.3, 0.1, 0.5, 0.2])
    pred = torch.randn(4, 4, 8, 8)
    target = torch.randn(4, 4, 8, 8)
    sig = adaptive_timestep_metric_signal(per_sample, pred, target, metric="slope")
    assert torch.allclose(sig, per_sample), "slope 应透传裸 per_sample loss"


def test_slope_end_to_end_via_update():
    # 走真实 update() 路径：bin1 喂持续下降 loss、其余恒定 → bin1 factor 最大
    s = AdaptiveTimestepSampler(enabled=True, bins=4, metric="slope",
                                min_factor=0.5, max_factor=2.5, ema_decay=0.8)
    t = torch.tensor([0.125, 0.375, 0.625, 0.875])
    for k in range(40):
        decreasing = max(2.0 - 0.04 * k, 0.2)
        per_sample = torch.tensor([1.0, decreasing, 1.0, 1.0])
        s.update(t, per_sample)
    assert s.ready
    f = s.factors().tolist()
    assert f[1] == max(f) and f[1] > f[0], f"持续下降的 bin 应得最大 factor: {f}"
