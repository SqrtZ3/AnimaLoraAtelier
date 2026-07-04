# Noise & Timestep Parameters Reference

> 覆盖范围：`trainer/objective.py` + `trainer/aux_losses.py` 中实现的所有
> timestep 采样、loss 类型、loss 加权、噪声生成、自适应时间步、辅助 loss 参数。

---

## 理论基础

Flow Matching 线性调度下：

```
x_t = (1-t)·x₀ + t·ε      t ∈ [0,1]
速度目标 v* = ε - x₀
x₀ 恢复：x₀_pred = x_t - t·v_pred
```

- **t ≈ 0**：低噪声端，决定发丝 / 纹理 / 眼睛等细节
- **t ≈ 1**：高噪声端，决定人物比例 / 构图等结构
- `SNR(t) = ((1−t)/t)²`，t 越小 SNR 越高（越"容易"）

---

## 一、Timestep 采样参数

### `timestep_sampling`

控制训练时 t 值从 `[0,1]` 的采样分布。

```python
# objective.py  sample_t()
if mode == "uniform":
    return torch.rand(bs, device=device)            # 均匀

if mode == "mixed_uniform_low":
    # mix_low_prob 概率从 logit_normal_low 采，其余均匀

if mode == "logit_normal":
    u_shifted = (u * s) / (1 + (s-1) * u)          # 偏向中高噪声

if mode == "logit_normal_low":
    u_shifted = (u / s) / (1 + (1/s - 1) * u)      # 偏向低噪声
```

| 取值 | 分布形状 | 对画面的影响 |
|---|---|---|
| `uniform` | 平坦，低/高 t 等概率 | 细节与结构均衡学习，最干净基线 |
| `logit_normal` | 中间 t 峰值 | 结构偏强，细纹理学不够 |
| `logit_normal_low` | 低 t 峰值 | 细节偏强，可能结构变形 |
| `mixed_uniform_low` | 均匀 + 低 t 混合 | 细节稍强于 uniform |
| `mixed_logit_low_high`（别名 `ushaped`/`bimodal`） | **U 形双峰**：低 t 峰 + 高 t 峰，中段掏空 | 细节(低t) 与 脸型/构图/氛围(高t) 两端**同时**喂；`mix_low_prob`=低噪峰占比(0.5=对称U)，`flow_shift`=两峰间距(shift3→峰≈0.25/0.75)。中段(t≈0.5，FM 最易、信息量最低)被刻意饿掉 |

> ⚠️ `logit_normal` / `logit_normal_low` / `mode` 这三个模式**不读** `laplace_mu`/`laplace_b`（仅 `laplace` 模式读）。在 logit 模式下设这两个值是 no-op。

---

### `flow_shift`

仅对 `logit_normal` / `logit_normal_low` / `mode` 有效。**`uniform` 模式下完全无效**（代码直接 `return rand`，shift 未被读取）。

```python
# logit_normal 时：u_shifted = (u * s) / (1 + (s-1)*u)
# s > 1 → 曲线向 t=1（高噪）偏；s < 1 → 向 t=0（低噪）偏
```

| 取值 | 影响 |
|---|---|
| 1.0 | 无偏移，logit-normal 原始形状 |
| 3.0（SD3 默认） | 偏向中高噪声端，结构感强 |
| > 1 + uniform 模式 | **完全无效** |

---

### `schedule_shift`

**所有 mode 下都有效**，是 `flow_shift` 的后处理补丁，在 `sample_t` 之后统一施加：

```python
# objective.py  apply_timestep_schedule_shift()
t_shifted = (t * s) / (1 + (s-1) * t)   # s = schedule_shift
```

| 取值 | 效果 |
|---|---|
| 1.0 | 恒等变换，不偏移 |
| > 1.0 | 把整体分布推向高噪端（结构向，伤细节） |
| < 1.0 | 把分布推向低噪端（细节向） |
| 0.9 | 微弱低噪偏置（对 uniform 也生效） |

---

### `timestep_mix_low_prob`

仅 `mixed_uniform_low` 模式下有效。每个 batch 中**该比例的样本**从 `logit_normal_low` 采样，其余从 uniform 采样。

| 取值 | 效果 |
|---|---|
| 0.0 | 等价纯 uniform |
| 0.08 | 轻微低噪偏置，适合低饱和画风 |
| 0.15 | 温和低噪偏置 |
| 0.24 | 约四分之一 batch 看细节步，细节追得积极 |
| > 0.4 | 低噪过多，自适应优化器 d 估计容易被低 t 步绑架 |

---

## 二、Loss 类型参数

### `loss_type`

| 取值 | 公式 | 特点 |
|---|---|---|
| `mse` | `(pred - target)²` | 标准 FM；大误差被平方放大，训练信号强但对异常样本敏感 |
| `l1` | `|pred - target|` | 各处均等惩罚；梯度方向稳，但精度不如 MSE |
| `huber` | 二次区间 + 线性区间的折中 | 抗尖峰，精度接近 MSE |

---

### `huber_c`

Huber 的折点（delta）基础值。配合 `huber_schedule: "snr"` 时，实际 delta 为：

```python
# objective.py  _huber_delta_for_t()  schedule="snr":
snr_sqrt = ((1-t)/t).clamp(0.1, 10)
actual_delta = huber_c * snr_sqrt
```

| 取值 | 效果 |
|---|---|
| 0.05 | 更 L1 化，对离群点更鲁棒，精度低 |
| 0.1（代码默认） | 平衡点 |
| 0.2 | 更 MSE 化，在低 t（高 SNR）时 delta 更大，精细纹理学习更精准 |
| 0.5+ | 几乎等同 MSE，失去 Huber 抗尖峰优势 |

---

### `huber_schedule`

| 取值 | 实际 delta(t) | 效果 |
|---|---|---|
| `constant` | 固定 = `huber_c` | 简单基线 |
| `snr` | `huber_c × sqrt(SNR) = huber_c × (1-t)/t` | 低 t（细节端）接近 MSE，高 t（结构端）接近 L1；自然适配 FM 的 SNR 分布 |
| `sigma` | `huber_c × t` | 与 snr 相反；高 t 时更 MSE，低 t 时更 L1 |

---

## 三、Loss 加权参数

### `loss_weighting_scheme`

```python
# objective.py  compute_loss_weight()
# min_snr:      w = min(γ/SNR, 1)    Flow-FM: SNR = ((1-t)/t)²
# detail_inv_t: w = 1/t,  clamp [min, max]
```

| 取值 | 加权曲线 | 对画面的影响 |
|---|---|---|
| `none` | 全部 1× | 无偏，结构/细节均衡 |
| `min_snr` | 压低高 SNR（低 t）步 | 防止细节步"太容易"→模型更多关注中高 t，结构稳定 |
| `detail_inv_t` | 1/t 上调低 t 步 | 细节/纹理权重提高，学得更快 |
| `sigma_sqrt` | sqrt(t) | 极轻微的低噪偏置 |
| `sigma_sqrt_sd3` | 1/t²（cap 1000） | 极激进，**仅大 batch (≥64) 可用** |
| `cosmap` | 2/(π·(1-2t+2t²)) | SD3 经典，中 t 友好，max/min ≈ 1.8× |

---

### `min_snr_gamma`

仅 `loss_weighting_scheme: "min_snr"` 时生效：

```
SNR(t) = ((1-t)/t)²
w = min(γ / SNR(t), 1)
```

| γ 取值 | 效果 |
|---|---|
| 0 | 等价 `none`（代码直接返回全 1） |
| 1~3 | 低 t 步被强力压权；细节步贡献下降，anime 风格常用 |
| 2.5 | 在细节追得积极与结构稳定之间的经验甜点 |
| 5（论文默认） | 保守，对高 SNR 步仅轻微压权 |
| 10+ | 几乎无加权效果 |

---

### `weight_cap_ratio`

batch 内最大权重 / 最小权重的硬上限，防止单个极端 t 样本绑架自适应优化器的步长估计：

```python
if weight_cap_ratio > 1.0:
    w_max_allowed = w.min() * weight_cap_ratio
    w = w.clamp(max=w_max_allowed)
```

| 取值 | 效果 |
|---|---|
| 0 | 禁用，权重无封顶 |
| 5~6 | 防止单样本主导，适合自适应优化器 + 小 batch |
| 10+ | 封顶过松，几乎无效 |

> `min_snr` 方案下 w 自然有 [0, 1] 上界，可设 0；`detail_inv_t` 方案建议设 5~6。

---

### `detail_inv_t_min` / `detail_inv_t_max`

仅 `detail_inv_t` 方案生效时使用，控制 `w = 1/t` 的 clamp 范围：

```
t=0.05 → 1/t=20  → clamp 到 detail_inv_t_max
t=0.20 → 1/t=5   → 恰在 max 附近
t=1.00 → 1/t=1   → clamp 到 detail_inv_t_min
```

| [min, max] | 效果 |
|---|---|
| [1, 3] | 保守，低饱和 / 低对比画风推荐 |
| [1, 4.5] | 标准细节强化 |
| [1, 5] | 代码默认，稍激进 |
| [1, 8] | 激进，自适应优化器 d 不稳定风险高 |

---

## 四、噪声生成参数

### `noise_offset`

给每个 batch 样本加一个**逐通道、空间均匀**的随机偏移：

```python
# offset shape: (B, C, 1, 1) 或 (B, C, T, 1, 1)
noise = noise + noise_offset * randn(B, C, 1, 1)
```

| 取值 | 效果 |
|---|---|
| 0（基线） | 标准各向同性高斯，最干净，细节还原最纯粹 |
| 0.02 | 轻微低频偏移，让模型学会生成极亮/极暗图像 |
| 0.05 | 明显色调丰富度提升，换一定纯净度 |
| 0.1+ | 强烈全局色调变化，可能模糊细纹理监督 |

---

### `noise_offset_min`

仅 `noise_offset_random_strength: true` 时有效，随机偏移强度在 `[noise_offset_min, noise_offset]` 之间均匀采样：

```python
scale = noise_offset_min + (noise_offset - noise_offset_min) * rand(B, 1, ...)
```

单独使用 `noise_offset` 不开 `random_strength` 时此参数**无效**。

---

### `noise_offset_random_strength`

| 取值 | 效果 |
|---|---|
| `false`（默认） | 全 batch 共用同一强度 `noise_offset` |
| `true` | 每样本独立随机化偏移强度，训练分布更丰富 |

---

### `pyramid_noise_iterations`

叠加多尺度低频噪声：

```python
# objective.py  make_noise()
for i in range(iterations):
    r = 2^(i+1)                          # 下采样倍数：2, 4, 8…
    small_noise = randn(B, C, H/r, W/r)
    extra = bilinear_upsample(small_noise, (H, W))
    noise += extra * discount^(i+1)      # 指数从 i+1 起（比 Whitaker 弱一级）
noise /= noise.std()                     # 归一化保持 std ≈ 1
```

| 取值 | 效果 |
|---|---|
| 0（基线） | 纯白噪声，与模型预训练噪声分布最接近 |
| 1~2 | 轻微低频成分，帮助学习全局光照/色调 |
| 3 | 中度低频，画面"丰富度"提升，细纹理还原略受影响 |
| 5+ | 强低频污染，可能引入全局光晕/色块伪影 |

---

### `pyramid_noise_discount`

每个 pyramid 层级的衰减系数：

```python
layer_weight = discount^(i+1)   # i=0 → discount^1, i=1 → discount^2, …
```

| 取值 | 效果 |
|---|---|
| 0.1~0.2 | 极弱，几乎等同 `iterations=0` |
| 0.3 | "弱模式"，对细节还原友好 |
| 0.5~0.7 | 接近 Whitaker/kohya 标准强度，全局色调学习积极 |
| > 0.9 | 各层几乎等权重叠加，噪声变非常"结构化" |

---

## 五、自适应时间步重采样

### `adaptive_timestep`

开启后（burn_in 步后），基于每个 t-bin 的 EMA 损失动态调整采样频率：高损失 bin 被多采样。关闭时为纯基础采样。

---

### `adaptive_timestep_metric`

决定"哪个 bin 算难"的信号：

```python
# entropy_rate 模式（objective.py  AdaptiveTimestepSampler.factors()）：
entropy_rate_k = loss_k / t_k³          # I-MMSE 风格熵率代理 (arXiv:2602.18647)
entropy_rate_k /= w(t_k)               # 除以 loss_weighting 的 w(t)，让 π·w ∝ ρ
gate = t^n / (t^n + c^n)               # 低噪闸门
factor = (entropy_rate * gate).normalized
```

| 取值 | 信号 | 适用场景 |
|---|---|---|
| `raw` | 直接 per-sample loss | 简单直观 |
| `highfreq` | 高频残差能量（avg_pool - original） | 关注空间细节误差 |
| `mixed` | raw + weight × highfreq | 综合 |
| `entropy_rate` | mse/t³ 除以 w(t)，理论最优 FM 重采样 | InfoNoise 方案 |

---

### `adaptive_timestep_low_noise_gate` / `gate_n` / `gate_c`

仅 `entropy_rate` 模式使用，防止 t→0 端爆炸分配：

```
g(t) = t^n / (t^n + c^n)
```

| 参数 | 作用 |
|---|---|
| `low_noise_gate: true` | 开启闸门，t ≪ c 时 factor 趋近 0，防止 t≈0 的 bin 被过度采样 |
| `gate_n` | 三次方（n=3）过渡较平滑 |
| `gate_c` | 约 t < c 的 bin 受到抑制；推荐 0.05~0.07 |

---

### 其他 adaptive 参数

| 参数 | 作用 | 推荐范围 |
|---|---|---|
| `adaptive_timestep_bins` | t 分成几个 bin | 8~16 |
| `adaptive_timestep_ema_decay` | EMA 衰减；越高越稳定但越慢响应 | 0.95~0.97 |
| `adaptive_timestep_burn_in` | 多少步后才激活 | 160~200 |
| `adaptive_timestep_min_factor` | 最低倍率（欠采 bin 的下限） | 0.5~0.75 |
| `adaptive_timestep_max_factor` | 最高倍率（过采 bin 的上限） | 1.5~2.0 |
| `adaptive_timestep_base_mix` | batch 中保留纯基础采样的比例 | 0.4~0.6；越高越保守 |
| `adaptive_timestep_candidate_mult` | 抽候选时的过采倍数 | 4~8 |
| `adaptive_timestep_highfreq_weight` | `mixed` metric 中 highfreq 的权重 | 0.1~0.3 |

---

## 六、辅助 Loss — Spectral

### `aux_spectral_enabled`

在 latent space 对 `x₀_pred` vs `x₀_target` 做 FFT 振幅 L1 匹配：

```python
# aux_losses.py  spectral_loss_per_sample()
fft_pred   = torch.fft.fft2(x0_pred,   dim=(-2,-1), norm="ortho")
fft_target = torch.fft.fft2(x0_target, dim=(-2,-1), norm="ortho")
loss = |amplitude(fft_pred) - amplitude(fft_target)|₁
```

零额外模型，零额外显存。直接惩罚高频缺失（眼睛线条、发丝等）。

---

### `aux_spectral_lambda`

| 取值 | 效果 |
|---|---|
| 0.02~0.04 | 保守，仅"提示"方向，主 loss 主导 |
| 0.05~0.07 | 平衡，推荐起点 |
| 0.1+ | 频域主导，边缘可能过锐 |

---

### `aux_spectral_use_wavelet` / `aux_spectral_wavelet_lambda`

附加 Haar 小波系数（LL/LH/HL/HH 四方向）L1 匹配：

```python
# aux_losses.py  _haar_wavelet_coefs()
# 单层 Haar 分解，grouped conv2d，stride=2
```

对有方向性的纹理（布料织纹、头发条纹）比纯 FFT 更有效。`wavelet_lambda` 是在 `spectral_lambda` 基础上的附加权重。

---

### `aux_spectral_t_gate`

仅 t < t_gate 的样本参与 spectral loss（高 t 时 x₀_pred 太不准，频域误差无意义）：

| 取值 | 覆盖范围 |
|---|---|
| 0.45~0.5 | 仅最低噪声端，最保守 |
| 0.7 | 低~中噪区间，覆盖更广 |
| 1.0 | 全程（不推荐，高 t 误导训练） |

---

## 七、辅助 Loss — Perceptual

### `aux_perceptual_enabled`

通过 VAE decode 把 `x₀_pred` 解码到 pixel space，然后：

- **LPIPS-VGG**：多层 VGG16 特征 L2，对纹理/局部细节敏感
- **DINOv2-B**：patch token cosine 距离，对语义结构（脸型、手）敏感

代价：约 +2~3 GB VRAM，训练时长约 3~4×。需要 `pip install lpips`。

---

### `aux_perceptual_lambda_lpips` / `aux_perceptual_lambda_dino`

| 参数 | 推荐值 | 效果 |
|---|---|---|
| `lambda_lpips` | 0.1（PixelGen 默认） | LPIPS 距离量级 0.1~0.5，此权重让贡献与主 loss 相当 |
| `lambda_dino` | 0.01（PixelGen 默认） | 语义结构补偿；设 0.0 则跳过 DINO，省 ~350 MB VRAM |

---

### `aux_perceptual_t_gate`

| 取值 | 效果 |
|---|---|
| 0.4~0.5 | 仅最细节端，最精准但覆盖少 |
| 0.7 | 低~中噪区间 |

---

### `aux_perceptual_lpips_net`

| 取值 | 特点 |
|---|---|
| `vgg`（推荐） | VGG16，纹理最敏感 |
| `alex` | AlexNet，更快但纹理感知弱 |
| `squeeze` | 最轻量 |

---

### `aux_perceptual_lpips_size` / `aux_perceptual_use_checkpoint`

| 参数 | 推荐 | 原因 |
|---|---|---|
| `lpips_size: 512` | 1024 训练必须设 | LPIPS 显存降 1/4；256~512 感知质量无明显差异 |
| `use_checkpoint: true` | 始终开启 | VAE decode + VGG + DINO 激活可达 15~25 GB，checkpoint 降到几 GB |

---

## 七.5、辅助 Loss — LPL（Latent Perceptual Loss）

来源：*Boosting Latent Diffusion with Perceptual Objectives*（arXiv 2411.04873，Meta FAIR）。
论文在 DDPM-ε/v 与 Flow Matching、256/512 分辨率、三个数据集上验证，FID +6~20%，
定性收益即"更锐、纹理更真实"。注意：论文场景是全参数（继续）训练，**小数据 LoRA/LoKr
微调下的收益无直接证据**，需 A/B 实测。

### 机理

latent-MSE（以及 latent 上的 FFT/wavelet 匹配）只关心 latent 数值逼近，与 decoder
如何把 latent 映射为像素**脱节**——VAE latent 空间高度不规则，latent 的小偏差可能被
decoder 放大成明显的质感/高频丢失。LPL 把 predicted x₀ 与 target x₀ 都过冻结 VAE
decoder，在其**多尺度中间特征空间**（H/8→H 共 4 个分辨率 stage 的末个 ResidualBlock
输出）做标准化后的 L2 匹配，梯度经 decoder 反传回 x₀_pred → LoRA。

与 `aux_perceptual`（LPIPS/DINO）的区别：不需要 VGG/DINO 等外部模型（零额外下载/依赖），
特征来自 VAE decoder 自身；与 `aux_spectral` 的区别：spectral 在 latent 上做频谱统计，
LPL 在"更接近像素"的 decoder 特征上做逐点匹配，二者机理轴不同、可叠加。

### 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `aux_lpl_enabled` | false | 总开关（default-off，关闭时完全 no-op） |
| `aux_lpl_lambda` | 0.1 | 权重。**保守起点、非论文值**（论文 w_LPL 在附录，正文未给；此处按仓库 aux 惯例取 0.1）。开训看日志把 LPL 项标定到主 loss 的 10~30% |
| `aux_lpl_t_gate` | 0.6 | 只对 t < gate 的样本启用（论文按 SNR 硬门控，FM 线性调度下等价于 t 阈值；低噪端 x̂₀ 才有意义 + 省算力） |
| `aux_lpl_outlier_k` | 8.0 | 标准化后 \|φ̂'\| > k 的特征元素不参与 loss（简化版 outlier 屏蔽；论文用 quantile+形态学操作）。0 = 关闭 |
| `aux_lpl_use_checkpoint` | true | 把 decode(pred)+特征 loss 包进梯度检查点（同 `aux_perceptual_use_checkpoint` 的理由） |
| `aux_lpl_num_scales` | 4 | 参与的 decoder 分辨率 stage 数（自低分辨率端起）。4=含全分辨率 tap（特征显存大头）；超大图预算紧张时降 3（显存约减半） |

### 代价与注意

- 每个过 gate 的样本 ≈ **2 次 VAE decoder 前向**（target 侧 no_grad + pred 侧带梯度，
  checkpoint 下 backward 再重放一次 pred 侧）。t_gate=0.6 且三峰采样偏高噪时，
  实际命中率 ≈ 低噪+部分中噪路由比例。
- `lpl_enabled=true` 时 VAE 自动保留在 GPU（同 perceptual 的 offload 逻辑）。
- NaViT 打包下逐图生效（与 spectral/perceptual 同一套逐图 unpatchify 路径）。
- 实现上与论文的两处已知偏离（都只影响 λ 标定，不影响梯度方向，见 aux_losses.py 注释）：
  ① 空间维取均值而非求和（navit 变尺寸图之间 per-image loss 可比）；
  ② outlier 屏蔽用硬阈值替代 quantile+形态学。

---

## 八、其他已实现参数

### `freq_balanced_dropout_strength`

按 tag 出现频率决定 dropout 概率，频率高的 tag 被 dropout 的概率更大（乘以此强度系数）。防止模型依赖高频共现 tag 而不学真正的风格特征。

| 取值 | 效果 |
|---|---|
| 0.0 | 禁用（等同普通 tag_dropout） |
| 0.3~0.5 | 温和均衡，让 LoRA 泛化更好 |
| 1.0 | 强均衡，罕见 tag 几乎不被 dropout |

---

## 九、综合对照表

### A. Timestep 采样参数

| 参数 | 小值/关 | 推荐范围 | 大值/开 | 核心影响 |
|---|---|---|---|---|
| `timestep_sampling` | `uniform`（最干净基线） | `mixed_uniform_low`（细节稍强）/ `mixed_logit_low_high`（U形双峰，两端兼顾） | `logit_normal`（结构偏强） | 整体 t 分布偏向 |
| `flow_shift` | 1.0（无偏移） | 3.0~5.5 | > 5 | 仅 logit_normal 类有效；越大越偏高噪结构端 |
| `schedule_shift` | 0.9（轻微细节偏） | 1.0（纯净基线） | > 1.0（结构偏，伤细节） | 后处理，全 mode 有效 |
| `timestep_mix_low_prob` | 0.0（纯 uniform） | 0.15~0.24 | > 0.4（低噪过饱和） | 混入低噪 bin 的比例 |

### B. Loss 函数参数

| 参数 | 小值 | 推荐范围 | 大值 | 核心影响 |
|---|---|---|---|---|
| `loss_type` | `l1`（稳定，精度低） | `huber` | `mse`（精准，不抗尖峰） | 误差的惩罚形状 |
| `huber_c` | 0.05（接近 L1） | 0.1~0.2 | 0.5+（接近 MSE） | Huber 折点基础值 |
| `huber_schedule` | `sigma`（高 t 精） | `snr`（低 t 细节端接近 MSE） | `constant`（均匀） | delta 随 t 的变化方式 |

### C. Loss 加权参数

| 参数 | 小值/关 | 推荐范围 | 大值/开 | 核心影响 |
|---|---|---|---|---|
| `loss_weighting_scheme` | `none` | `min_snr` 或 `detail_inv_t` | `sigma_sqrt_sd3`（仅大 batch） | t 步权重分配策略 |
| `min_snr_gamma` | 1~2（细节步被强压） | 2.5~5 | 10+（无效果） | min_snr 方案专用 |
| `weight_cap_ratio` | 0（禁用封顶） | 0（min_snr）或 5~6（detail_inv_t） | 10+（封顶过松） | 防单样本劫持优化器步长 |
| `detail_inv_t_min/max` | [1, 3]（低对比风格） | [1, 4.5] | [1, 8]（激进，不稳定） | detail_inv_t 方案的权重上下限 |

### D. 噪声生成参数

| 参数 | 小值/关 | 推荐范围 | 大值/开 | 核心影响 |
|---|---|---|---|---|
| `noise_offset` | 0（最纯净） | 0~0.02 或 0.05（丰富度） | 0.1+（全局色调变化强） | 低频 DC 偏移，明暗范围 |
| `noise_offset_random_strength` | false | false（基线）/ true（多样性） | — | 是否随机化偏移强度 |
| `noise_offset_min` | 0 | 0（与 random_strength 配合） | — | 随机偏移下限 |
| `pyramid_noise_iterations` | 0（最纯净） | 0（基线）或 2~3（丰富度） | 5+（结构性伪影风险） | 低频叠加层数 |
| `pyramid_noise_discount` | 0.1（极弱） | 0.3（弱模式，细节友好） | 0.6~0.7（Whitaker 标准强度） | 每层金字塔的衰减率 |

### E. 自适应时间步参数

| 参数 | 小值/关 | 推荐范围 | 大值/开 | 核心影响 |
|---|---|---|---|---|
| `adaptive_timestep` | false（纯基础采样） | true（精细训练） | — | 是否启用损失感知重采样 |
| `adaptive_timestep_metric` | `raw` | `entropy_rate`（理论最优） | `highfreq`（关注细节误差） | 衡量 bin 难度的信号 |
| `adaptive_timestep_bins` | 8（粗粒度） | 16 | 32（过细，噪声大） | t 空间分辨率 |
| `adaptive_timestep_ema_decay` | 0.9（反应快但噪） | 0.95~0.97 | 0.99（太稳，适应慢） | EMA 稳定性 |
| `adaptive_timestep_burn_in` | 50（太早） | 160~200 | 500+（太晚） | 激活前的热身步数 |
| `adaptive_timestep_min_factor` | 0.25（欠采 bin 极少） | 0.5 | 0.75（保守） | 欠采样 bin 的最低频率倍率 |
| `adaptive_timestep_max_factor` | 1.25（保守） | 1.5~2.0 | 3.0+（极端不稳定） | 过采样 bin 的最高频率倍率 |
| `adaptive_timestep_base_mix` | 0.0（全 adaptive） | 0.4~0.6 | 1.0（等价关闭） | 保留基础分布的比例 |
| `adaptive_timestep_low_noise_gate` | false | true（entropy_rate 必开） | — | 抑制极低 t bin 过采 |
| `adaptive_timestep_gate_n` | 1（缓慢过渡） | 3.0 | 5+（几乎阶跃函数） | 闸门陡峭度 |
| `adaptive_timestep_gate_c` | 0.01（抑制极少） | 0.05~0.07 | 0.2（抑制太多低噪） | 闸门切换点（t 值） |

### F. 辅助 Loss 参数

| 参数 | 小值/关 | 推荐范围 | 大值/开 | 核心影响 |
|---|---|---|---|---|
| `aux_spectral_enabled` | false（基线） | true（推荐，免费） | — | FFT 振幅匹配，补高频细节 |
| `aux_spectral_lambda` | 0.02（仅提示） | 0.05~0.07 | 0.1+（边缘过锐） | 频域 loss 权重 |
| `aux_spectral_use_wavelet` | false | true（有方向纹理时） | — | 附加 Haar 小波匹配 |
| `aux_spectral_wavelet_lambda` | 0.05 | 0.1~0.3 | 0.5+（小波主导） | 小波分量相对权重 |
| `aux_spectral_t_gate` | 0.3（极保守） | 0.45~0.5 | 0.7（低中噪都覆盖） | 频域 loss 的 t 截止 |
| `aux_perceptual_enabled` | false（快） | true（精度高，贵 3~4×） | — | LPIPS + DINO pixel-space loss |
| `aux_perceptual_lambda_lpips` | 0.05 | 0.1（PixelGen 默认） | 0.3+（主 loss 被压制） | 纹理/局部细节权重 |
| `aux_perceptual_lambda_dino` | 0.0（跳过 DINO） | 0.01 | 0.05+（语义强制） | 语义结构对齐权重 |
| `aux_perceptual_t_gate` | 0.3（极保守） | 0.4~0.5 | 0.7（中噪也算） | 感知 loss 的 t 截止 |
| `aux_perceptual_lpips_size` | 0（原分辨率，最准但 OOM） | 512（1024 训练必设） | 256（显存极紧张） | LPIPS 前的下采目标边长 |
| `freq_balanced_dropout_strength` | 0.0（普通 dropout） | 0.3~0.5 | 1.0（稀有 tag 零 dropout） | 按 tag 频率均衡化的 dropout 强度 |
