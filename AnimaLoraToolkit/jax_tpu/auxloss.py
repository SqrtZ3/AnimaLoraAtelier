r"""逐图辅助项：Eisbach 障碍权重 / ΔFM(VeCoR) 负样本 / spectral(FFT+小波)。

（文件名不叫 `aux.py`：AUX 是 Windows 的保留设备名，git 在 Windows 上
 直接打不开这个文件 —— `error: open("aux.py"): No such file or directory`，
 而 ls/python 都看得见它。踩过一次，别改回去。）

对齐依据：
  trainer/objective.py:814   `eisbach_barrier_weight`（空间能量熵 -> 逐样本 detached 权重）
  trainer/objective.py:934   `vecor_contrastive_neg`（对 target 做破坏性增强当负样本）
  trainer/aux_losses.py:170  `recover_x0_from_velocity`：x0 = x_t - t·v
  trainer/aux_losses.py:194  `_haar_wavelet_coefs`（单层 Haar，2x2 滤波器 ×0.5，stride 2）
  trainer/aux_losses.py:251  `spectral_loss_per_sample`（FFT 振幅 L1 + 可选小波 L1，t-gate）
  anima_train.py:4258        navit 逐图 aux 的组装：λ × **命中 gate 的图的均值**

## 三项在 TPU 上的可移植性各不相同，逐条说清

**Eisbach —— 可精确移植。** 它是"位置能量分布的熵"，对位置的**排列不变**：
只要拿到同一张图的那一组位置能量，摆成网格还是摆成一条 token 序列，softmax
与熵完全相同。所以打包布局下不需要还原网格，逐段做 masked softmax 即可。
唯一要注意的是**位置的粒度**：PyTorch 在 latent 像素上算（每个位置 16 通道），
而一个 token 是 2x2 个 latent 像素。这里把 token 的 64 维拆回 `(c=16, ph, pw)`
再对 c 取均值，得到每 token 4 个位置 —— 与 PyTorch 逐像素完全同粒度。
（若图省事直接对 64 维取均值，就变成 4 个像素先平均再算熵，熵会系统性偏低，
且不报错。）

**ΔFM(VeCoR) —— 只移植了两支增强中的一支（诚实标注）。**
PyTorch 每次调用在"通道乱序"与"随机裁剪后 resize 回原尺寸"之间各 50% 二选一。
通道乱序是逐 token 的置换，可精确移植；裁剪+resize 需要**该图的真实网格**，
而打包布局下网格是运行时量（不进编译身份，否则布局数爆炸）。所以这里**只做通道
乱序**，等价于把原实现的随机二选一固定到其中一支。这会改变负样本的分布 ——
`dfm_lambda=0.05` 下影响有限，但它是真实差异，不要当成等价实现。

**spectral —— 移植了，但 FFT 走"零填充到静态画布"。**
打包布局下每图的 (h, w) 是运行时量，而 FFT 需要静态形状。这里把每图散射进一个
固定大小的画布再做 FFT。零填充**不是近似**：补零后的 DFT 就是同一信号 DTFT 的
更细采样（幅度谱与平移无关，所以图放在画布左上角不影响幅度）。差别只有两处，
都已补偿：
  ① `norm="ortho"` 的归一化分母从"图面积"变成"画布面积" -> 乘 sqrt(N_画布/N_图)；
  ② 频点数变多（更细采样）-> 求均值时自然抵消。
小波那一支不需要补偿：Haar 是 2x2/stride 2，图的高宽都是偶数，块与图边界严格
对齐，所以**只统计完全落在图内的系数**就与原实现逐元素相同。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

try:
    from . import flow as F
except ImportError:
    import flow as F

LATENT_CH = 16          # Anima/Qwen VAE 的 latent 通道数
PATCH = 2


@dataclass(frozen=True)
class AuxConfig:
    """与 yaml 的 `eisbach_* / dfm_* / aux_spectral_*` 同名同义。"""
    eisbach_lambda: float = 0.0
    dfm_lambda: float = 0.0
    dfm_mode: str = "vecor"
    spectral_enabled: bool = False
    spectral_lambda: float = 0.05
    spectral_use_wavelet: bool = False
    spectral_wavelet_lambda: float = 0.05
    spectral_t_gate: float = 0.7
    #: FFT 画布（token 网格单位）。必须 >= 数据集里最大的 (h, w)，由
    #: `data.CacheDataset` 在建计划时算出来填进去；填小了会 fail-fast。
    canvas_hw: Tuple[int, int] = (0, 0)

    def __post_init__(self):
        if self.dfm_lambda > 0 and self.dfm_mode != "vecor":
            raise ValueError(
                f"dfm_mode={self.dfm_mode!r}：batch 模式要在同形 batch 内配对负样本，"
                f"NaViT 逐图异形打包下对不上（PyTorch 侧 anima_train.py:1509 同样"
                f"fail-fast）。请用 dfm_mode: vecor。")
        if self.spectral_enabled and min(self.canvas_hw) <= 0:
            raise ValueError("aux_spectral 需要 canvas_hw（FFT 画布尺寸），"
                             "由数据侧按数据集最大网格填入")

    @property
    def any_enabled(self) -> bool:
        return (self.eisbach_lambda > 0 or self.dfm_lambda > 0
                or self.spectral_enabled)


# ── 逐段归约的小工具 ──────────────────────────────────────────────────────────
def seg_sum(x: jnp.ndarray, seg: jnp.ndarray, g: int) -> jnp.ndarray:
    return jax.ops.segment_sum(x, seg, num_segments=g, indices_are_sorted=True)


def seg_max(x: jnp.ndarray, seg: jnp.ndarray, g: int) -> jnp.ndarray:
    return jax.ops.segment_max(x, seg, num_segments=g, indices_are_sorted=True)


def pixel_energy(pred: jnp.ndarray) -> jnp.ndarray:
    """token [N, 64] -> 每 token 的 4 个 latent 像素的能量 [N, 4]。

    通道序是 `(c pt ph pw)`（c 在最外，pt=1），所以 reshape 成 (16, 4) 后对第一维
    取均值就是"逐像素、跨 16 通道的均方"，与 PyTorch 的 `o.pow(2).mean(dim=1)` 同义。
    """
    o = pred.astype(jnp.float32).reshape(*pred.shape[:-1], LATENT_CH, PATCH * PATCH)
    return jnp.mean(o ** 2, axis=-2)


def eisbach_weight(pred: jnp.ndarray, mask: jnp.ndarray, seg: jnp.ndarray,
                   g: int, lam: float, eps: float = 1e-6) -> jnp.ndarray:
    """逐图的 Eisbach log-barrier 权重 [G]（objective.py:814，**整体 detach**）。

      e   = 逐位置能量
      p   = softmax(e)               # 只在该图的真实位置上
      H   = -Σ p log p / log(M)      # 归一化熵 ∈ [0,1]
      w   = 1/(1 + (-log(1-H)))      # 障碍 -> 权重 ∈ (0,1]
      out = (1-λ) + λ·w              # 论文的插值地板，保证平坦样本也有保底监督

    H→0（有结构）→ w→1；H→1（弥散/均值化）→ w→0。detach 是有意的：它只缩 step
    size、不改梯度方向（监督扩散的方向锁死在真值上，所以安全）。

    逐段 softmax 用"减段内最大值"稳定化。填充位置被 `mask` 排除在分子分母之外
    —— 若不排除，填充多的 pack 会因为一大片相同的值把熵抬到接近 1，权重被系统性
    压到地板（不报错，只是训练悄悄变慢）。
    """
    e = pixel_energy(pred)                                   # [N, 4]
    m = mask.astype(jnp.float32)[:, None]                    # [N, 1]
    big = jnp.where(m > 0, e, -jnp.inf)
    mx = seg_max(jnp.max(big, axis=-1), seg, g)              # [G]
    mx = jnp.where(jnp.isfinite(mx), mx, 0.0)
    ex = jnp.exp(e - mx[seg][:, None]) * m
    den = seg_sum(jnp.sum(ex, axis=-1), seg, g)              # [G]
    den = jnp.maximum(den, eps)
    p = ex / den[seg][:, None]
    ent = -seg_sum(jnp.sum(p * jnp.log(jnp.maximum(p, eps)) * m, axis=-1), seg, g)
    cnt = jnp.maximum(seg_sum(jnp.sum(m * jnp.ones_like(e), axis=-1), seg, g), 2.0)
    h = jnp.clip(ent / jnp.log(cnt), 0.0, 1.0)
    w = 1.0 / (1.0 + (-jnp.log(jnp.maximum(1.0 - h, eps))))
    return jax.lax.stop_gradient((1.0 - lam) + lam * w)


def vecor_negative(key, target: jnp.ndarray, seg: jnp.ndarray, g: int) -> jnp.ndarray:
    """VeCoR 负目标：对每张图独立地把 16 个 latent 通道乱序（objective.py:955-961）。

    保证非恒等（撞上恒等置换就 roll 一位），与原实现一致。返回与 target 同形。
    **只实现了通道乱序这一支**，理由见模块 docstring。
    """
    perms = jax.random.permutation(key, jnp.tile(jnp.arange(LATENT_CH), (g, 1)),
                                   axis=1, independent=True)          # [G, 16]
    ident = jnp.all(perms == jnp.arange(LATENT_CH), axis=1, keepdims=True)
    perms = jnp.where(ident, jnp.roll(perms, 1, axis=1), perms)
    t = target.astype(jnp.float32).reshape(*target.shape[:-1], LATENT_CH, PATCH * PATCH)
    return jnp.take_along_axis(t, perms[seg][:, :, None], axis=-2).reshape(target.shape)


# ── spectral ─────────────────────────────────────────────────────────────────
def to_canvas(tokens: jnp.ndarray, seg: jnp.ndarray, rows: jnp.ndarray,
              cols: jnp.ndarray, mask: jnp.ndarray, g: int,
              hw: Tuple[int, int]) -> jnp.ndarray:
    """token [N, 64] -> 逐图的 latent 网格画布 [G, 16, 2*H, 2*W]（图在左上角，其余补 0）。

    **必须用 `.add` 而不是 `.set`**：填充 token 的 (row, col) 都是 0，与该段真
    token 的 (0,0) 撞在同一个格子上；`.set` 下谁最后写谁赢（XLA 不保证顺序），
    真 token 可能被 0 覆盖 —— 不报错，只是那一格数据没了。`.add` 下填充写的是
    0（已乘 mask），加多少次都不改结果。
    """
    h, w = hw
    tok = (tokens.astype(jnp.float32) * mask.astype(jnp.float32)[:, None]).reshape(
        -1, LATENT_CH, PATCH, PATCH)
    canvas = jnp.zeros((g, h, w, LATENT_CH, PATCH, PATCH), jnp.float32)
    canvas = canvas.at[seg, rows, cols].add(tok)
    # [G, h, w, c, ph, pw] -> [G, c, h*ph, w*pw]
    return canvas.transpose(0, 3, 1, 4, 2, 5).reshape(g, LATENT_CH, h * PATCH,
                                                      w * PATCH)


def _haar(x: jnp.ndarray) -> jnp.ndarray:
    """单层 Haar 分解，返回 [B, 4*C, H/2, W/2]（aux_losses.py:194 的等价写法）。

    原实现用 grouped conv2d + 4 个 2x2 核 ×0.5；这里直接按 stride-2 切片组合，
    数值上是同一个线性变换（同样的 ±1 组合乘 0.5），少一次卷积调用。
    """
    a = x[..., 0::2, 0::2]
    b = x[..., 0::2, 1::2]
    c = x[..., 1::2, 0::2]
    d = x[..., 1::2, 1::2]
    ll = (a + b + c + d) * 0.5
    lh = (a + b - c - d) * 0.5
    hl = (a - b + c - d) * 0.5
    hh = (a - b - c + d) * 0.5
    return jnp.concatenate([ll, lh, hl, hh], axis=1)


def spectral_per_image(x0_pred: jnp.ndarray, x0_target: jnp.ndarray,
                       cover: jnp.ndarray, cfg: AuxConfig) -> jnp.ndarray:
    """逐图 spectral loss [G]。输入是 `to_canvas` 的产物 [G, 16, H, W]。

    `cover` [G, H, W] 是画布上"这一格属于真实图像"的 0/1 掩码，用来
      ① 把 FFT 的 ortho 归一化补偿回图自己的面积（见模块 docstring）；
      ② 给小波系数做 masked mean（只统计完全落在图内的块）。
    """
    fp = jnp.fft.fft2(x0_pred.astype(jnp.float32), axes=(-2, -1), norm="ortho")
    ft = jax.lax.stop_gradient(
        jnp.fft.fft2(x0_target.astype(jnp.float32), axes=(-2, -1), norm="ortho"))
    # 稳定的复模：sqrt(re²+im²+eps) —— 在 0 处的梯度会 NaN，加 eps 后对非零幅度
    # 实质无影响（aux_losses.py:44 同款处理）
    amp = lambda z: jnp.sqrt(jnp.real(z) ** 2 + jnp.imag(z) ** 2 + 1e-12)
    diff = jnp.abs(amp(fp) - amp(ft))
    n_canvas = float(x0_pred.shape[-1] * x0_pred.shape[-2])
    n_img = jnp.maximum(jnp.sum(cover, axis=(-2, -1)), 1.0)         # [G]
    # 零填充补偿：|A_padded| = sqrt(N_img/N_canvas)·|A_native|
    total = jnp.mean(diff, axis=(1, 2, 3)) * jnp.sqrt(n_canvas / n_img)

    if cfg.spectral_use_wavelet:
        cp = _haar(x0_pred.astype(jnp.float32))
        ct = jax.lax.stop_gradient(_haar(x0_target.astype(jnp.float32)))
        # 系数掩码：2x2 块四角都在图内才算数（图的高宽是偶数、块与边界对齐，
        # 所以这等价于"块完全落在图内"）
        cm = (cover[:, 0::2, 0::2] * cover[:, 0::2, 1::2]
              * cover[:, 1::2, 0::2] * cover[:, 1::2, 1::2])[:, None]
        num = jnp.sum(jnp.abs(cp - ct) * cm, axis=(1, 2, 3))
        den = jnp.maximum(jnp.sum(cm, axis=(1, 2, 3)) * cp.shape[1], 1.0)
        total = total + float(cfg.spectral_wavelet_lambda) * (num / den)
    return total


def spectral_term(x0_pred, x0_target, cover, t, cfg: AuxConfig,
                  valid: jnp.ndarray) -> jnp.ndarray:
    """λ × **命中 t-gate 的图的均值**（anima_train.py:4290-4295 的口径）。

    那一行的注释值得照抄进来：navit 侧是对命中的图**求和**再除以命中数，
    因为非 navit 路径的 `spectral_loss` 返回的是 batch 均值。少除这一下，
    aux 相对主 loss 会被放大 G 倍（G≈6 时就压死主损失了）。
    没有图命中 gate 时返回 0。
    """
    gate = ((t < float(cfg.spectral_t_gate)).astype(jnp.float32)
            * valid.astype(jnp.float32))
    per = spectral_per_image(x0_pred, x0_target, cover, cfg)
    return (float(cfg.spectral_lambda) * jnp.sum(per * gate)
            / jnp.maximum(jnp.sum(gate), 1.0))


def recover_x0(noisy: jnp.ndarray, t_tok: jnp.ndarray,
               pred: jnp.ndarray) -> jnp.ndarray:
    """aux_losses.py:170 —— 线性 FM 下 `x0 = x_t - t·v`。fp32。"""
    return noisy.astype(jnp.float32) - t_tok * pred.astype(jnp.float32)
