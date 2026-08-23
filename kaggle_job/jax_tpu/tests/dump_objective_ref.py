r"""闸门⑬上半：用 PyTorch 侧的 `trainer/objective.py` / `trainer/aux_losses.py`
产出主 loss 与逐图 aux 的参考数据。

跑法：torch 解释器跑本脚本 -> jax 解释器跑 `check_objective_parity.py`。

覆盖：Huber δ 的三种调度、Huber/smooth-L1 的 loss map、逐图 masked 归约、
Eisbach 障碍权重、spectral（FFT 幅度 + Haar 小波）、三峰 t 采样的分位点。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from _upstream import upstream                                                 # noqa: E402
upstream("trainer/objective.py 与 trainer/aux_losses.py")

from trainer.aux_losses import (AuxLossConfig, recover_x0_from_velocity,      # noqa: E402
                                spectral_loss_per_sample)
from trainer.objective import (_huber_delta_for_t, _huber_loss_map,           # noqa: E402
                               eisbach_barrier_weight, masked_token_loss,
                               sample_t)

OUT = Path(__file__).parent / "_ref" / "objective_ref.npz"
C, H, W = 16, 8, 12          # latent 网格；token 网格是它的一半（patch=2）
G, N = 3, H * W // 4         # 3 张图、每图 24 个 token


def main() -> int:
    torch.manual_seed(0)
    out: dict[str, np.ndarray] = {}
    t = torch.tensor([0.08, 0.5, 0.93], dtype=torch.float32)
    out["t"] = t.numpy()

    # ── ① Huber δ 的三种调度 ─────────────────────────────────────────────────
    for sched in ("constant", "snr", "sigma"):
        d = _huber_delta_for_t(t, 0.2, sched, 2.0)
        out[f"delta_{sched}"] = (np.full(3, float(d), np.float32)
                                 if not torch.is_tensor(d)
                                 else d.reshape(3).numpy())

    # ── ② Huber / smooth-L1 的 loss map ──────────────────────────────────────
    err = torch.rand(200, dtype=torch.float32) * 1.5
    dt = torch.full((200,), 0.3)
    out["err"] = err.numpy()
    out["huber_map"] = _huber_loss_map(err, dt, False).numpy()
    out["smoothl1_map"] = _huber_loss_map(err, dt, True).numpy()

    # ── ③ 逐图 masked token loss（Huber + snr 调度）────────────────────────────
    pred = torch.randn(G, N, 64) * 0.5
    tgt = torch.randn(G, N, 64) * 0.5
    mask = torch.ones(G, N)
    mask[1, -6:] = 0.0                    # 第二张图有 6 个填充 token
    out.update(pred=pred.numpy(), tgt=tgt.numpy(), mask=mask.numpy())
    out["per_image"] = masked_token_loss(
        pred, tgt, mask, loss_type="huber", huber_c=0.2, huber_schedule="snr",
        t=t, huber_snr_clamp_max=2.0).numpy()

    # ── ④ Eisbach 障碍权重（逐图，dense 网格口径）────────────────────────────
    # pred 的 token 维 64 = (c=16, ph=2, pw=2)，还原成 [G, 16, 1, h*2, w*2]
    grid = pred.reshape(G, H // 2, W // 2, C, 2, 2).permute(0, 3, 1, 4, 2, 5)
    grid = grid.reshape(G, C, H, W).unsqueeze(2)          # [G, C, 1, H, W]
    out["eisbach"] = eisbach_barrier_weight(grid, 0.15).numpy()
    out["grid"] = grid.numpy()

    # ── ⑤ spectral（FFT 幅度 + Haar 小波），逐图、原生网格 ─────────────────────
    acfg = AuxLossConfig(spectral_enabled=True, spectral_lambda=0.065,
                         spectral_use_wavelet=True, spectral_wavelet_lambda=0.20,
                         spectral_t_gate=1.01)             # gate 全开，逐图都算
    noisy = torch.randn(G, C, 1, H, W)
    x0p = recover_x0_from_velocity(noisy, t, grid)
    x0t = torch.randn(G, C, 1, H, W)
    out.update(noisy=noisy.numpy(), x0t=x0t.numpy())
    out["spectral"] = spectral_loss_per_sample(x0p, x0t, t, acfg).numpy()

    # ── ⑥ 三峰 t 采样的分位点（RNG 不同，只能比分布）──────────────────────────
    ts = sample_t(200000, torch.device("cpu"), mode="mixed_logsnr_three",
                  shift=3.0, mix_low_prob=0.12, mix_high_prob=0.48,
                  logsnr_mu=-6.0, logsnr_sigma=2.4)
    out["three_q"] = np.quantile(ts.numpy(), np.linspace(0.05, 0.95, 19))

    # ── ⑦ VeCoR 裁剪+resize 负样本支路（固定参数逐点比）────────────────────────
    # 负样本的数学就是 objective.py:963-969 的那两行（裁 60-90% + align_corners=False
    # 双线性拉回），这里原样复刻并固定全部随机量，供 jax 侧逐点比。
    # 两张图**网格不同**（16x12 与 10x8 latent 像素），专门验"逐图运行时网格"。
    import torch.nn.functional as Fn
    va = torch.randn(C, 16, 12)
    vb = torch.randn(C, 10, 8)
    v_ratio = np.array([0.75, 0.62], np.float32)
    v_top = np.array([2, 1], np.int32)
    v_left = np.array([1, 2], np.int32)
    v_negs = []
    for im, r, tp, lf in zip((va, vb), v_ratio, v_top, v_left):
        h, w = im.shape[-2:]
        ch_, cw_ = max(int(h * float(r)), 2), max(int(w * float(r)), 2)
        crop = im[None, :, int(tp):int(tp) + ch_, int(lf):int(lf) + cw_]
        v_negs.append(Fn.interpolate(crop, size=(h, w), mode="bilinear",
                                     align_corners=False)[0])
    out.update(vecor_a=va.numpy(), vecor_b=vb.numpy(),
               vecor_ratio=v_ratio, vecor_top=v_top, vecor_left=v_left,
               vecor_neg_a=v_negs[0].numpy(), vecor_neg_b=v_negs[1].numpy())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, **out)
    print(f"已写 {OUT}（{len(out)} 项）")
    print(f"  δ(snr) = {out['delta_snr']}")
    print(f"  逐图 loss = {out['per_image']}")
    print(f"  eisbach = {out['eisbach']}")
    print(f"  spectral = {out['spectral']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
