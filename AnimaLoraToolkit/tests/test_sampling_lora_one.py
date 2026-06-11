"""logsnr/分层采样 + LoRA-One KPSVD 初始化的单元测试。

pytest 可跑；也可直接 `python tests/test_sampling_lora_one.py`（本地无 pytest 时）。
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from trainer.objective import (  # noqa: E402
    sample_t, sample_t_stratified, apply_t_range,
    lwd_saliency_mask, per_sample_loss,
    vecor_contrastive_neg, LossBinEMA,
)
from trainer.lora_one import kpsvd_lokr_factors, lora_one_kpsvd_init  # noqa: E402


def test_logsnr_mode_peak():
    torch.manual_seed(0)
    t = sample_t(20000, "cpu", mode="logsnr", logsnr_mu=-6.0, logsnr_sigma=2.0)
    # μ=-6 ⇒ t 中位数 = sigmoid(3) ≈ 0.9526
    med = t.median().item()
    assert 0.93 < med < 0.97, med
    assert (t > 0.8).float().mean().item() > 0.75  # 大部分质量在高噪段


def test_mixed_logsnr_low_high_bimodal():
    torch.manual_seed(0)
    t = sample_t(20000, "cpu", mode="mixed_logsnr_low_high", shift=3.0,
                 mix_low_prob=0.45, logsnr_mu=-6.0, logsnr_sigma=2.0)
    low_frac = (t < 0.5).float().mean().item()
    assert 0.38 < low_frac < 0.52, low_frac           # 低噪峰份额 ≈ mix_low_prob
    assert (t > 0.85).float().mean().item() > 0.35    # 高噪峰显著存在
    assert ((t > 0.45) & (t < 0.75)).float().mean().item() < 0.15  # 中段被掏空


def test_stratified_coverage_and_marginal():
    torch.manual_seed(0)
    bs = 4
    kw = dict(mode="mixed_logit_low_high", shift=3.0, mix_low_prob=0.45)
    ref = torch.sort(sample_t(100000, "cpu", **kw)).values
    quartiles = [ref[int(q * len(ref))].item() for q in (0.25, 0.5, 0.75)]
    # 覆盖性：分层是对"候选池经验分位"的互斥保证；双峰分布下池内峰间二项波动
    # 使真四分位覆盖只能以平均值衡量。iid 采样的平均 distinct-bin 数 ≈ 2.73，
    # 分层应显著更高（≥3.2）。
    n_trials = 300
    distinct_sum = 0
    for _ in range(n_trials):
        t = sample_t_stratified(bs, "cpu", **kw)
        bins = {sum(v > q for q in quartiles) for v in t.tolist()}
        distinct_sum += len(bins)
    assert distinct_sum / n_trials > 3.2, distinct_sum / n_trials
    # 方差削减（真正的目标）：批内均值 t 的波动应远小于 iid 采样
    strat_means = torch.stack([sample_t_stratified(bs, "cpu", **kw).mean() for _ in range(2000)])
    iid_means = torch.stack([sample_t(bs, "cpu", **kw).mean() for _ in range(2000)])
    assert strat_means.std().item() < 0.5 * iid_means.std().item(), \
        (strat_means.std().item(), iid_means.std().item())
    # 边际分布一致性：分层大样本的中位数与普通采样接近
    big = torch.cat([sample_t_stratified(bs, "cpu", **kw) for _ in range(5000)])
    assert abs(big.median().item() - ref.median().item()) < 0.03


def test_apply_t_range():
    t = torch.tensor([0.00001, 0.5, 0.9999])
    out = apply_t_range(t, 0.05, 1.0)
    assert out.min().item() >= 0.05 - 1e-9
    assert out.max().item() <= 1.0 - 1e-4 + 1e-9
    assert abs(out[1].item() - 0.5) < 1e-9          # 中段不动
    # 默认参数 = 历史 1e-4 clamp 行为：1e-5 被抬到 1e-4，其余不动
    out2 = apply_t_range(t)
    assert abs(out2[0].item() - 1e-4) < 1e-9
    assert abs(out2[1].item() - 0.5) < 1e-9


def test_kpsvd_exact_recovery():
    torch.manual_seed(0)
    f, od, idim, r = 4, 32, 24, 6
    w1 = torch.randn(f, f)
    w2 = torch.randn(od, r) @ torch.randn(r, idim)   # 秩 r 的 W2
    G = torch.kron(w1, w2)
    a1, a2a, a2b = kpsvd_lokr_factors(G, factor=f, rank=r)
    approx = torch.kron(a1, a2a @ a2b)
    rel = (approx - G).norm() / G.norm()
    assert rel.item() < 1e-5, rel.item()  # 精确 Kronecker×低秩结构应近乎完美还原


def test_kpsvd_rank_padding():
    G = torch.randn(16, 8)
    a1, a2a, a2b = kpsvd_lokr_factors(G, factor=4, rank=10)  # rank > min(od=4, id=2)
    assert a2a.shape == (4, 10) and a2b.shape == (10, 2)
    assert torch.isfinite(torch.kron(a1, a2a @ a2b)).all()


def test_lora_one_init_end_to_end():
    torch.manual_seed(0)
    from trainer.lora import LoRALinear

    base = torch.nn.Linear(64, 64, bias=False)
    lora = LoRALinear(base, rank=8, alpha=8.0, use_lokr=True, factor=4,
                      lora_variant="dora")

    class FakeInjector:
        injected = {"blocks.0.self_attn.q_proj": lora}

    G = torch.randn(64, 64)
    scale_rel = 0.02
    stats = lora_one_kpsvd_init(FakeInjector(), {"blocks.0.self_attn.q_proj": G},
                                scale_rel=scale_rel)
    assert "applied=1" in stats, stats

    delta = lora.adapter.delta_weight(apply_rank_dropout=False).float()
    W0 = base.weight.detach().float()
    # 范数标定：||ΔW|| = scale_rel · ||W0||
    assert abs(delta.norm().item() - scale_rel * W0.norm().item()) / (scale_rel * W0.norm().item()) < 1e-3
    # 方向：ΔW 与 -G 同向（内积为正）
    assert (delta * (-G)).sum().item() > 0
    # DoRA 幅度已重算：m = rownorm(W0 + ΔW) ⇒ 初始有效权重 = W0 + ΔW（无幅度失配）
    expected_m = (W0 + delta).norm(dim=1)
    assert torch.allclose(lora.dora_scale.detach().float(), expected_m, rtol=1e-3, atol=1e-5)
    # merged_weight 与 W0+ΔW 一致
    merged = lora.merged_weight().float()
    assert torch.allclose(merged, W0 + delta, rtol=1e-3, atol=1e-5)


def test_mixed_logsnr_three_bands():
    torch.manual_seed(0)
    t = sample_t(30000, "cpu", mode="mixed_logsnr_three", shift=3.0,
                 mix_low_prob=0.40, mix_high_prob=0.25,
                 logsnr_mu=-6.0, logsnr_sigma=2.0)
    low = (t < 0.45).float().mean().item()       # 低噪峰（峰约 0.25）
    high = (t > 0.88).float().mean().item()      # 高噪峰（峰约 0.95）
    mid = ((t >= 0.45) & (t <= 0.88)).float().mean().item()  # 中噪峰（峰约 0.75）
    assert 0.33 < low < 0.48, low
    assert 0.15 < high < 0.32, high
    assert mid > 0.25, mid                       # 中段不再空洞（v2 教训）


def test_loss_bin_ema_weighting():
    ema = LossBinEMA(bins=4, decay=0.9, burn_in=5, min_w=0.25, max_w=4.0)
    t_lo = torch.full((8,), 0.1)   # bin 0：高 loss
    t_hi = torch.full((8,), 0.9)   # bin 3：低 loss
    t_all = torch.tensor([0.1, 0.35, 0.6, 0.9] * 2)
    # burn-in 前权重恒 1
    assert ema.weight(t_lo).min().item() == 1.0
    for _ in range(20):
        ema.update(t_all, torch.tensor([4.0, 2.0, 2.0, 1.0] * 2))
    assert ema.ready
    w_lo = ema.weight(t_lo)[0].item()
    w_hi = ema.weight(t_hi)[0].item()
    assert w_lo < 1.0 < w_hi, (w_lo, w_hi)       # 高 loss 段降权、低 loss 段升权（均衡化）
    assert 0.25 <= w_lo and w_hi <= 4.0
    # NaN 批次不污染 EMA
    before = ema.ema.clone()
    ema.update(t_all, torch.tensor([float("nan")] * 8))
    assert torch.allclose(ema.ema, before)


def test_vecor_contrastive_neg():
    torch.manual_seed(0)
    pred = torch.randn(1, 4, 1, 16, 16)          # bs=1 也成立（vs ΔFM 的 batch 依赖）
    target = torch.randn(1, 4, 1, 16, 16)
    for _ in range(8):                            # 覆盖两种增强分支
        neg = vecor_contrastive_neg(pred, target, loss_type="mse")
        assert neg.shape == (1,)
        assert torch.isfinite(neg).all() and neg.item() > 0
        pos = per_sample_loss(pred, target, loss_type="mse")
        assert abs(neg.item() - pos.item()) > 1e-8  # 负目标 != 原目标


def test_lwd_saliency_mask_gating():
    torch.manual_seed(0)
    # 左半棋盘格（高频）、右半纯平：显著图应左高右低
    lat = torch.zeros(2, 4, 1, 16, 16)
    checker = (torch.arange(16).view(-1, 1) + torch.arange(16).view(1, -1)) % 2
    lat[:, :, :, :, :8] = checker[:, :8].float() * 2 - 1

    # t=0.6 > floor=0.3：平坦区（显著度≈0）被门掉，高频区保留
    m_mid = lwd_saliency_mask(lat, torch.tensor([0.6, 0.6]), floor=0.3)
    assert m_mid.shape == (2, 1, 1, 16, 16)
    assert m_mid[..., :, :8].mean().item() > 0.9   # 高频半边几乎全保留
    assert m_mid[..., :, 10:].mean().item() < 0.1  # 平坦半边几乎全被门掉
    # t=0.2 < floor：全图受监督
    m_low = lwd_saliency_mask(lat, torch.tensor([0.2, 0.2]), floor=0.3)
    assert m_low.min().item() == 1.0
    # 纯平图 + 高 t：空 mask 兜底回退全 1
    flat = torch.zeros(1, 4, 1, 16, 16)
    m_flat = lwd_saliency_mask(flat, torch.tensor([0.99]), floor=0.3)
    assert m_flat.min().item() == 1.0


def test_per_sample_loss_weight_map():
    torch.manual_seed(0)
    pred = torch.randn(2, 4, 1, 8, 8)
    target = torch.randn(2, 4, 1, 8, 8)
    # 全 1 权重 == 不加权
    ones = torch.ones(2, 1, 1, 8, 8)
    a = per_sample_loss(pred, target, loss_type="mse", weight_map=ones)
    b = per_sample_loss(pred, target, loss_type="mse")
    assert torch.allclose(a, b, rtol=1e-5)
    # 只保留左半：等于左半的均值
    half = torch.zeros(2, 1, 1, 8, 8)
    half[..., :4] = 1.0
    c = per_sample_loss(pred, target, loss_type="mse", weight_map=half)
    expected = (pred - target)[..., :4].float().square().reshape(2, -1).mean(dim=1)
    assert torch.allclose(c, expected, rtol=1e-5)


def test_lora_one_global_eta_preserves_relative_magnitudes():
    """全局单一步长：梯度小 10× 的模块，初始 ΔW 也必须小 10×（防止每模块归一化
    把零梯度模块推满扰动 → 模型纯噪声的事故回归测试）。"""
    torch.manual_seed(1)
    from trainer.lora import LoRALinear

    base_a = torch.nn.Linear(64, 64, bias=False)
    base_b = torch.nn.Linear(64, 64, bias=False)
    lora_a = LoRALinear(base_a, rank=8, alpha=8.0, use_lokr=True, factor=4, lora_variant="dora")
    lora_b = LoRALinear(base_b, rank=8, alpha=8.0, use_lokr=True, factor=4, lora_variant="dora")

    class FakeInjector:
        injected = {"m.a": lora_a, "m.b": lora_b}

    G = torch.randn(64, 64)
    scale_rel = 0.02
    stats = lora_one_kpsvd_init(FakeInjector(), {"m.a": G, "m.b": 0.1 * G},
                                scale_rel=scale_rel)
    assert "applied=2" in stats, stats

    da = lora_a.adapter.delta_weight(apply_rank_dropout=False).float()
    db = lora_b.adapter.delta_weight(apply_rank_dropout=False).float()
    # 相对幅度保持：‖ΔW_b‖/‖ΔW_a‖ = 梯度比 0.1（同方向、同 η）
    ratio = (db.norm() / da.norm()).item()
    assert abs(ratio - 0.1) < 1e-3, ratio
    # 扰动最大的模块（a）恰好达到 scale_rel·‖W0_a‖
    W0a = base_a.weight.detach().float()
    assert abs(da.norm().item() - scale_rel * W0a.norm().item()) / (scale_rel * W0a.norm().item()) < 2e-2
    # b 远低于 scale_rel（这正是旧实现做不到的）
    W0b = base_b.weight.detach().float()
    assert db.norm().item() < 0.05 * scale_rel * W0b.norm().item() * 10


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nall {len(fns)} tests passed")
