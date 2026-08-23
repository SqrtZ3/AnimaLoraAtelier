r"""闸门⑬下半：JAX 的主 loss / 逐图 aux ≟ PyTorch 侧实现。

先用 torch 解释器跑 `dump_objective_ref.py`，再用 jax 解释器跑本脚本。

## 判据分三档，因为这三类项的可移植性本来就不同

  * **逐 bit / 1e-6 档**：Huber δ、loss map、逐图 masked 归约、Eisbach、
    VeCoR 裁剪+resize（运行时裁剪参数 + 静态画布 gather，全静态形状）。
    这些在打包布局下是精确可移植的（Eisbach 靠"熵对位置排列不变"这一条）。
  * **1e-3 档**：spectral 的 FFT 那一支。TPU 侧把每图散射进静态画布再 FFT，
    零填充 = 同一 DTFT 的更细采样，幅度谱的**均值**只差归一化（已补偿），
    但不同频点的采样位置不同 -> 不可能逐 bit 相同。这里判的是"补偿之后量级对得上"。
  * **分布档**：三峰 t 采样。两边 RNG 不同，只能比 19 个分位点。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax                                                    # noqa: E402
import jax.numpy as jnp                                       # noqa: E402
import numpy as np                                            # noqa: E402

import auxloss as X                                               # noqa: E402
import flow as F                                              # noqa: E402

REF = Path(__file__).parent / "_ref" / "objective_ref.npz"
OK = FAIL = 0
C, H, W = 16, 8, 12


def check(name, got, want, tol=1e-6):
    global OK, FAIL
    g, w = np.asarray(got, np.float64), np.asarray(want, np.float64)
    rel = np.max(np.abs(g - w)) / max(np.max(np.abs(w)), 1e-12)
    good = rel < tol and g.shape == w.shape
    if good:
        OK += 1
    else:
        FAIL += 1
    print(f"  {'[OK ]' if good else '[FAIL]'} {name:<32} rel={rel:.3e} "
          f"(阈值 {tol:g})" + ("" if g.shape == w.shape else f" 形状 {g.shape}/{w.shape}"))


def main() -> int:
    if not REF.exists():
        print(f"缺 {REF} —— 先用 torch 解释器跑 dump_objective_ref.py")
        return 2
    z = np.load(REF)
    t = jnp.asarray(z["t"])
    G = t.shape[0]
    N = z["pred"].shape[1]
    seg = jnp.repeat(jnp.arange(G), N)
    ssum = lambda a: X.seg_sum(a, seg, G)

    print("① Huber δ 的三种调度")
    for s in ("constant", "snr", "sigma"):
        cfg = F.FlowConfig(loss_type="huber", huber_c=0.2, huber_schedule=s,
                           huber_snr_clamp_max=2.0)
        check(f"δ({s})", F.huber_delta(t, cfg), z[f"delta_{s}"])

    print("\n② Huber / smooth-L1 的 loss map")
    err = jnp.asarray(z["err"])[:, None]
    dt = jnp.full((err.shape[0],), 0.3)
    # elem_loss 会对最后一维取均值；给最后一维长度 1 就等于逐元素
    for name, lt in (("huber", "huber"), ("smoothl1", "smooth_l1")):
        cfg = F.FlowConfig(loss_type=lt, huber_c=0.3)
        got = F.elem_loss(err, jnp.zeros_like(err), dt, cfg)
        check(f"{name} map", got, z[f"{name}_map"])

    print("\n③ 逐图 masked token loss（Huber + snr）")
    cfg = F.FlowConfig(loss_type="huber", huber_c=0.2, huber_schedule="snr",
                       huber_snr_clamp_max=2.0)
    pred = jnp.asarray(z["pred"]).reshape(G * N, -1)
    tgt = jnp.asarray(z["tgt"]).reshape(G * N, -1)
    mask = jnp.asarray(z["mask"]).reshape(G * N)
    delta_tok = jnp.take(F.huber_delta(t, cfg), seg)
    per, den = F.per_image_loss(pred, tgt, mask, t, cfg, ssum, delta_tok)
    check("逐图 loss", per, z["per_image"])
    check("逐图真 token 数", den, z["mask"].sum(1))

    print("\n④ Eisbach 障碍权重（熵对位置排列不变 -> 打包布局下可精确移植）")
    # 参考值是在**完整网格**上算的（真实 navit 路径里每图 unpatchify 成自己的网格，
    # 网格里没有填充位）。所以这里也要给全 1 掩码 —— 拿带填充的掩码去比，比的是
    # 两个不同的量（本地实测差 3.9e-4，正是那 6 个填充 token 的贡献）。
    full = jnp.ones_like(mask)
    check("逐图权重", X.eisbach_weight(pred, full, seg, G, 0.15), z["eisbach"])

    print("\n⑤ spectral（FFT 幅度 + Haar 小波）")
    rows = jnp.tile(jnp.repeat(jnp.arange(H // 2), W // 2), G)
    cols = jnp.tile(jnp.tile(jnp.arange(W // 2), H // 2), G)
    x0p_tok = X.recover_x0(_grid_to_tok(z["noisy"]), jnp.take(t, seg)[:, None],
                           pred)

    def spec(canvas):
        acfg = X.AuxConfig(spectral_enabled=True, spectral_lambda=0.065,
                           spectral_use_wavelet=True, spectral_wavelet_lambda=0.20,
                           spectral_t_gate=1.01, canvas_hw=canvas)
        cv = lambda a: X.to_canvas(a, seg, rows, cols, full, G, canvas)
        cover = (cv(jnp.ones_like(pred))[:, 0] > 0).astype(jnp.float32)
        return X.spectral_per_image(cv(x0p_tok), cv(_grid_to_tok(z["x0t"])),
                                    cover, acfg)

    # 5a 画布 == 原生网格：没有零填充，应当与 PyTorch 逐元素一致
    check("画布=原生网格（无填充）", spec((H // 2, W // 2)), z["spectral"], tol=1e-5)
    # 5b 画布比图大 2.7 倍：FFT 走零填充（DTFT 的更细采样）+ 归一化补偿。
    #    小波那一支仍是精确的（只统计完全落在图内的块）。这里判的是"补偿之后
    #    量级对得上"，**不是**逐 bit —— 数据集里图越接近画布，这个残差越小。
    check("画布放大 2.7x（零填充+补偿）", spec((6, 8)), z["spectral"], tol=0.10)

    # 5b 之后：VeCoR 的裁剪+resize 支路。两张图网格不同（16x12 与 10x8 latent
    # 像素），验"逐图运行时网格 + 固定参数逐点一致"。放在 ⑥ 前是因为它是
    # 1e-6 档的精确判据，不是分布档。
    print("\n⑥+ VeCoR 裁剪+resize 支路（固定参数，逐点比 token 域负样本）")
    ga = z["vecor_a"][None, :, None]          # [1, 16, 1, 16, 12]
    gb = z["vecor_b"][None, :, None]          # [1, 16, 1, 10, 8]
    ta_, tb_ = _grid_to_tok(ga), _grid_to_tok(gb)      # [48, 64], [20, 64]
    tok = jnp.concatenate([ta_, tb_], axis=0)
    vseg = jnp.asarray([0] * 48 + [1] * 20)
    vrows = jnp.concatenate([jnp.repeat(jnp.arange(8), 6),
                             jnp.repeat(jnp.arange(5), 4)])
    vcols = jnp.concatenate([jnp.tile(jnp.arange(6), 8),
                             jnp.tile(jnp.arange(4), 5)])
    vmask = jnp.ones((68,), jnp.float32)
    got = X._vecor_crop_resize_params(
        tok, vseg, vrows, vcols, vmask, 2, (8, 6),
        jnp.asarray(z["vecor_ratio"]), jnp.asarray(z["vecor_top"]),
        jnp.asarray(z["vecor_left"]))
    check("crop+resize 图A（满画布 8x6 网格）", np.asarray(got[:48]),
          np.asarray(_grid_to_tok(z["vecor_neg_a"][None, :, None])), tol=1e-5)
    check("crop+resize 图B（画布内 5x4 网格）", np.asarray(got[48:]),
          np.asarray(_grid_to_tok(z["vecor_neg_b"][None, :, None])), tol=1e-5)

    print("\n⑥ 三峰 t 采样的分位点（RNG 不同，只比分布）")
    fcfg = F.FlowConfig(t_mode="mixed_logsnr_three", flow_shift=3.0,
                        mix_low_prob=0.12, mix_high_prob=0.48,
                        logsnr_mu=-6.0, logsnr_sigma=2.4)
    ts = F.base_t_np(np.random.RandomState(0), 200000, fcfg)
    q = np.quantile(ts, np.linspace(0.05, 0.95, 19))
    check("19 个分位点", q, z["three_q"], tol=2e-2)

    print(f"\n{'*** 通过 ***' if FAIL == 0 else f'*** {FAIL} 项失败 ***'}"
          f"  ({OK} OK / {FAIL} FAIL)")
    return 0 if FAIL == 0 else 1


def _grid_to_tok(a: np.ndarray) -> jnp.ndarray:
    """[G, C, 1, H, W] -> [G*N, 64]，通道序 `(c ph pw)`（= patchify 的序）。"""
    g = a[:, :, 0]
    G_, c, h, w = g.shape
    t = g.reshape(G_, c, h // 2, 2, w // 2, 2).transpose(0, 2, 4, 1, 3, 5)
    return jnp.asarray(t.reshape(G_ * (h // 2) * (w // 2), c * 4))


if __name__ == "__main__":
    raise SystemExit(main())
