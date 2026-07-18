# Optimizer Parameters Reference

> 覆盖范围：`utils/optimizer_utils.py` + 各优化器实现文件中所有已支持的
> `optimizer_type` 及对应 `optimizer_args`。

---

## 一、总览

| `optimizer_type` | 算法核心 | 是否需要手动 LR | 显存开销 | 对应论文/来源 |
|---|---|---|---|---|
| `adamw` | AdamW（PyTorch 标准） | ✅ 是 | 2× 参数量 | Loshchilov & Hutter 2017 |
| `adamw8bit` | 8-bit 量化 AdamW | ✅ 是 | 约 1× 参数量 | Dettmers et al., bitsandbytes |
| `prodigyplus` | 自适应 d + Schedule-Free | ❌ 固定 lr=1.0 | 2× + d 状态 | arXiv:2306.06101 + SF-Adam |
| `soap` | Shampoo 特征基下的 Adam | ✅ 是 | 2× + 预条件矩阵 | arXiv:2409.11321 |
| `soap_sf` | SOAP 预条件 + Schedule-Free 平均 | ✅ 是（无需调度） | 2× + 预条件矩阵 | arXiv:2409.11321 + 2405.15682 |
| `adopt` | 延迟 v 归一化 Adam | ✅ 是 | 2× 参数量 | arXiv:2411.02853 (NeurIPS 2024) |
| `lion` | Sign-based 单 EMA | ✅ 是（更小） | 1× 参数量 | arXiv:2302.06675 (NeurIPS 2023) |
| `clion` | Lion + Cautious 掩码 | ✅ 是 | 1× 参数量 | arXiv:2411.16085 |
| `emosens` | Loss 序列驱动动态 LR | ⚠️ 仅设范围 | 2× 参数量 | github.com/muooon/EmoSens |
| `muon` | 动量 + Newton-Schulz 正交化（2D）| ✅ 是（≈AdamW 量级）| 2× 参数量（momentum+master）| github.com/KellerJordan/Muon + arXiv:2502.16982 |
| `muon_sf` | Muon + Schedule-Free 平均 | ✅ 是（无需调度）| 3× 参数量（z+y+momentum）| 同上 + arXiv:2405.15682 |
| `automagic` | 逐元素自适应 lr（Adafactor 二阶矩 + 符号一致性）| ⚠️ lr 是**起点**，强度看 `max_lr` | ~2.25× 参数量（master+mask+polarity）| github.com/ostris/ai-toolkit |

> ⛔ **muon / muon_sf 不适用于 LoRA 画风训练**（2026-07-18 实测结案，见 §8 警告框）。

**共同实现细节**：SOAP / ADOPT / Lion / EmoSens 的**状态张量全部强制存 fp32**，即使参数是 bf16 也不例外——防止 bf16 下二阶矩精度损失。

---

## 二、各优化器详解

---

### 1. `adamw` / `adamw8bit` — 标准 AdamW

**算法**（Loshchilov & Hutter 2017）：

```
m_t = β1·m_{t-1} + (1-β1)·g_t
v_t = β2·v_{t-1} + (1-β2)·g_t²
m̂ = m_t / (1-β1^t)           ← bias correction
v̂ = v_t / (1-β2^t)
θ_t = θ_{t-1}·(1 - lr·wd) - lr·m̂/(√v̂+ε)   ← decoupled WD
```

`adamw8bit` 使用 bitsandbytes 的 8-bit 量化存储 m/v，显存减半，精度损失极小。

**optimizer_args 完整列表：**

```yaml
optimizer_type: "adamw"
learning_rate: 1e-4
optimizer_args:
  betas: [0.9, 0.999]       # β1/β2，AdamW 默认值
  weight_decay: 0.01        # 解耦 L2 正则化系数
  eps: 1e-8                 # 分母小量，防除零
  # ──── adamw 独有 ────
  amsgrad: false            # AMSGrad 变体（用历史最大 v 替代当前 v）
  foreach: false            # 使用 torch _foreach_* 融合 kernel，速度更快
  fused: false              # CUDA fused kernel（需 cuda 且参数连续）
  maximize: false           # 最大化 objective（一般不用）
  # ──── adamw8bit 独有 ────
  min_8bit_size: 4096       # 参数量低于此值的矩阵不量化（用标准 fp32 state）
```

**参数影响：**

| 参数 | 影响 |
|---|---|
| `betas[0]` (β1) | 动量衰减；越大梯度方向越平滑，对噪声鲁棒，但响应慢 |
| `betas[1]` (β2) | 二阶矩衰减；越大历史方差影响越持久，步长更稳定 |
| `weight_decay` | L2 正则；LoRA 不必太大（0~0.01），防止表征被压死 |
| `eps` | 极小值；bf16 训练可适当调大到 `1e-7` 防数值问题 |
| `amsgrad` | 提高最差情形收敛保证，LoRA 小参数量不一定需要 |
| `min_8bit_size` | 仅 adamw8bit；小矩阵量化精度损失不值得，4096 是经验阈值 |

**推荐配置：**
```yaml
optimizer_type: "adamw"
learning_rate: 5e-5
optimizer_args:
  betas: [0.9, 0.999]
  weight_decay: 0.01
  eps: 1e-8
```

---

### 2. `prodigyplus` — ProdigyPlus + Schedule-Free

**算法**（Mishchenko & Defazio, arXiv:2306.06101 + Schedule-Free Adam）：

Prodigy 的核心是**自动估计有效学习率 d**：

```
# 每步估计 d（自适应 LR 放大因子）
d_numerator  += d · ⟨g, θ₀ - θ⟩       # 与初始参数的内积
d_denominator += (lr·d)² · ‖g‖²
d = d_coef · d_numerator / d_denominator

# 实际步长
effective_lr = lr · d     # lr 固定=1.0，所以 effective_lr ≈ d
```

**Schedule-Free** 消除了 warmup + decay scheduler 的需要，直接用 primal-dual 平均实现隐式调度。

**StableAdamW**（`use_stableadamw`）在更新前对梯度做 RMS 归一化：

```
g_rms = g / RMS(g)     # 抗 bf16 梯度尖峰
```

**optimizer_args 完整列表：**

```yaml
optimizer_type: "prodigyplus"
learning_rate: 1.0        # ★ 必须设 1.0（Prodigy 数学要求），代码会强制覆盖
optimizer_args:
  betas: [0.95, 0.99]     # β1/β2；LoRA 推荐比 AdamW 的 0.9/0.999 更小
  eps: null               # null → Adam-atan2 模式（新版推荐）；或正 float（传统模式）
  weight_decay: 0.0       # LoRA 通常不需要 WD
  d0: 2.0e-5              # d 的初始猜测值
  d_coef: 1.0             # d 估计的放大/缩小系数
  d_limiter: true         # 限制 d 单步增长上限，防止爆炸
  prodigy_steps: 0        # 多少步后停止更新 d；0 = 全程自适应
  use_schedulefree: true  # 是否启用 Schedule-Free
  use_stableadamw: true   # 梯度 RMS 归一化，bf16 大分辨率训练至关重要
  schedulefree_c: 0       # Schedule-Free 的 warmup 常数；0 = 无 warmup
  use_speed: false        # 实验性：speed-based d 更新
  use_bias_correction: false  # Adam 偏差校正；SF 模式下通常不需要
  factored: false         # 分解二阶矩（Adafactor 思路），极省显存但不稳定
```

**参数影响：**

| 参数 | 小值/关 | 推荐值 | 大值/开 | 影响 |
|---|---|---|---|---|
| `betas[0]` (β1) | 0.85（梯度响应快） | 0.95 | 0.99（过平滑） | 动量；LoRA 训练 0.95 比 0.9 更稳 |
| `betas[1]` (β2) | 0.9（历史短） | 0.99 | 0.999（几乎不收敛初期） | 二阶矩；影响 d 的估计精度 |
| `eps` | — | null（atan2 模式） | 1e-8（传统） | null 时 `atan2(m, √v)` 无需 ε，bf16 更安全 |
| `d0` | 1e-6（冷启动极慢） | 2e-5 | 1e-3（初期步长过大） | d 的起点；实际 LR ≈ d 最终收敛值 |
| `d_coef` | 0.3（学习缓慢） | 1.0 | 3.0（d 爆炸风险） | 全局 d 乘子；< 1 容易不学习 |
| `d_limiter` | false（无限制） | true | — | 防止单步 d 暴增 |
| `prodigy_steps` | 0（全程自适应） | 0 | 1000（早期固化 LR） | 固化后退化为 AdamW |
| `use_schedulefree` | false（退化为 Prodigy） | true | — | 消除 scheduler 需求 |
| `use_stableadamw` | false（bf16 尖峰风险） | true | — | RMS 归一化，大分辨率必开 |
| `schedulefree_c` | 0（无 warmup） | 0 | 100（慢 warmup） | Schedule-Free warmup 步数近似值 |
| `factored` | false | false | true（极省显存但精度差） | 分解二阶矩 |

> **LoRA+ 与 Prodigy 的交互**：`loraplus_lr_ratio` 给 A/B 矩阵不同 group 设不同 lr 初值。Prodigy 会对每组独立估计 d，ratio 过大（如 4.0）时两组的 d 估计轨迹差异过大，整体 d 不稳。推荐 ratio ≤ 2.0。

---

### 3. `soap` — SOAP (Shampoo 特征基下的 Adam)

**算法**（Vyas et al., arXiv:2409.11321）：

SOAP = "Shampoo with Adam in the Shampoo eigenbasis"。先用 Shampoo 的 Kronecker 矩阵估计梯度的协方差特征基 Q，然后在 Q 旋转后的空间里跑标准 Adam：

```
# Step 1: 更新 Shampoo 协方差矩阵 GG（每步）
GG[dim] = shampoo_beta·GG[dim] + (1-shampoo_beta)·g·gᵀ

# Step 2: 每 precondition_frequency 步求特征基 Q
Q[dim] = eigenvectors(GG[dim])

# Step 3: 把梯度投影到特征基
g' = Q⁻¹·g·Q

# Step 4: 在 g' 空间跑标准 Adam
m' = β1·m' + (1-β1)·g'
v' = β2·v' + (1-β2)·(g')²
update' = m'/(√v'+ε)

# Step 5: 投影回原空间
update = Q·update'·Q⁻¹
θ -= lr·update
```

仅对维度 ≤ `max_precond_dim` 的轴做 Shampoo 预条件；大维度轴自动退化为 Adam。

**optimizer_args 完整列表：**

```yaml
optimizer_type: "soap"
learning_rate: 3e-4
optimizer_args:
  betas: [0.95, 0.95]          # 代码默认；β1=β2 是 SOAP 论文推荐
  shampoo_beta: -1.0           # Shampoo 协方差的 EMA 系数；-1 = 使用 β2
  eps: 1e-8
  weight_decay: 0.01
  precondition_frequency: 10   # 每 N 步重新计算特征基 Q
  max_precond_dim: 10000       # 超过此维度的轴不做 Shampoo（退化为 Adam）
  merge_dims: false            # 合并小维度再预条件（节省内存）
  precondition_1d: false       # 是否对 1D 向量（如 bias）也做 Shampoo
  normalize_grads: false       # 对 update 做 RMS 归一化
  data_format: "channels_first" # 4D 张量的 channel 维度位置
  correct_bias: true           # Adam-style bias correction
```

**参数影响：**

| 参数 | 推荐值 | 说明 |
|---|---|---|
| `betas` | [0.95, 0.95] | SOAP 论文 β1=β2 实验结果最好；代码对 AdamW 默认值 (0.9, 0.999) 自动覆盖 |
| `shampoo_beta` | -1 | 复用 β2；显式设值则独立控制 GG 的 EMA 速度 |
| `precondition_frequency` | 10~20 | 越小越精准但 eigen-decompose 开销大；LoRA 小矩阵下 10 很快 |
| `max_precond_dim` | **256~512** | ★ **state 体积头号开关**。GG/Q 是「每维一个方阵」，对维度**二次增长**。LoKr 的 `w2_a=(out/factor, rank)`、`w2_b=(rank, in/factor)` 的大维（如 3072）若 ≤ 此值会建 3072² 矩阵（单参数 ~73MB），全模型轻松上 GB。设 256 → 只预条件 rank 维，大维退化 Adam，state ≈ 仅动量 |
| `precond_in_state` | true | GG/Q 可由梯度重算。设 **false** → 不把它们写进 checkpoint（resume 冷重建，约 1/(1-β2) 步重新预热），与 max_precond_dim 无关地把 state 砍到只剩动量。想保留大维预条件又要小 checkpoint 时用 |
| `merge_dims` | false | LoRA 矩阵维度已小，无需合并 |
| `precondition_1d` | false | DoRA 的 magnitude 向量可选开 |
| `correct_bias` | true | 初期梯度偏置校正，建议开 |

> **state 文件过大排查**：若 `training_state*.pt` 远大于导出的 LoRA（实测 27MB LoRA 配 `max_precond_dim=10000` → 3GB state），99% 是 GG/Q。两条独立解法：①把 `max_precond_dim` 降到 256~512（同时**加速** eigen-decompose）；②`precond_in_state: false`（保留训练期完整预条件，只是不存盘）。两者可叠加。动量（exp_avg/exp_avg_sq）和参数同形，从来不是大头。

---

### 3b. `soap_sf` — Schedule-Free SOAP

**算法**（SOAP 预条件 + Schedule-Free, Defazio et al. arXiv:2405.15682）：

把 Schedule-Free 机制（base-optimizer 无关，设计上就是包在 Adam/SGD 外面）套到 SOAP 的
"特征基里跑 Adam" 上。关键改动：**丢掉一阶动量 `exp_avg`（m'）**，用三序列插值 + Polyak
平均代替；二阶矩 `exp_avg_sq`（v'）仍留在 Shampoo 特征基里随 Q 旋转。

```
y_t   = (1-β1)·z_t + β1·x_t          # 梯度在 y 处求；β1=SF 插值动量
g'    = Qᵀ·g(y_t)                     # 投影到特征基（沿用 SOAP）
v'    = β2·v' + (1-β2)·g'²            # 二阶矩留特征基（保留）
u     = Q·(g' / (√v' + ε))           # ★ 分子是 g'，不是 m'；投影回参数空间
z_{t+1} = z_t - lr·u                  # base 步进作用在 z（参数空间，无需换基）
x_{t+1} = (1-c)·x_t + c·z_{t+1}       # Polyak 平均，c≈(r+1)/t
```

- **参数张量训练时存 y**（≈ 平均点 x 附近）；`eval()` 换成 x（采样/存档用），`train()` 换回 y。
  训练脚本已按 `hasattr(opt,"eval")` 自动调度，无需手动。
- **显存与 SOAP 持平**：`z` 顶替 `exp_avg`。比 SOAP 还少了一阶矩的换基记账。
- **无需 LR 调度器 / total_steps**：脚本检测到 `soap_sf` 会强制 `lr_scheduler=none`。

**optimizer_args 完整列表：**

```yaml
optimizer_type: "soap_sf"
learning_rate: 2.5e-4
lr_scheduler: "none"           # 脚本会强制；SF 自带平均退火
optimizer_args:
  betas: [0.9, 0.95]           # β1=SF 插值动量（不再是一阶矩 EMA）；β2=二阶矩
  shampoo_beta: -1.0
  eps: 1.0e-8
  weight_decay: 0.01           # 解耦 WD，在 y 点求值
  precondition_frequency: 5
  max_precond_dim: 10000
  merge_dims: false
  precondition_1d: false
  normalize_grads: false
  data_format: "channels_first"
  correct_bias: true           # 把 √(1-β2^t) 折进 lr
  # ── SF 专属 ──
  weight_lr_power: 2.0         # 平均权重里 lr 的幂（论文默认 2.0）
  r: 0.0                       # 迭代序号幂；0=均匀平均，调高让 x 更贴近最新 z
  warmup_steps: 0              # SF 通常不需要 warmup
```

**参数影响：**

| 参数 | 影响 |
|---|---|
| `betas[0]` (β1) | **SF 插值权重**（不是动量缓冲）。`eval()` 要除以 β1，必须 ∈ (0,1)；0.9 是 SF 默认 |
| `betas[1]` (β2) | 特征基二阶矩 EMA；保留 SOAP 的 0.95（快二阶矩 = 拟合快的来源之一） |
| `r` | **短跑关键旋钮**。c≈(r+1)/t：r 越大越偏重晚期迭代，x 越贴近最新 z。但 60 步级别下即便 r=4 也只让 c≈5/60，平均仍严重滞后（见选型建议） |
| `weight_lr_power` | 平均权重里 lr 的幂，默认 2.0，几乎不用动 |
| `warmup_steps` | SF 一般不需要；想稳早期预条件可设 20~50 |

> **核心直觉**：x 是整条轨迹的（加权）平均。1200 步时轨迹大部分时间在解附近，x 落点良好；
> 60 步时轨迹全程都在下降，x≈轨迹质心 = 严重欠拟合——这是 SF 的数学本性，没有旋钮能在
> 60 步内救回。详见选型建议。

---

### 4. `adopt` — ADOPT (NeurIPS 2024)

**算法**（Taniguchi et al., arXiv:2411.02853）：

Adam 的收敛性需要梯度噪声有界，但 diffusion / VAE / 多目标 loss 的梯度天然无界。ADOPT 修复：**用上一步的 v_{t-1} 来归一化当前梯度**，并在 v 更新之后再做参数更新：

```
step 1:   v₁ = g₁²               # 仅初始化
step t>1:
  denom = clamp(√v_{t-1}, min=ε)  # ← 用上一步的 v，不是当前
  normed = g_t / denom
  if use_clip: normed = clamp(normed, -t^c, t^c)   # ADOPT-clip
  m_t = β1·m_{t-1} + (1-β1)·normed
  θ_t = θ_{t-1}·(1-lr·wd) - lr·m_t                # ← 先更新参数
  v_t = β2·v_{t-1} + (1-β2)·g_t²                  # ← 再更新 v
```

这个顺序调换使得 ADOPT 对**任意 β2 ∈ [0,1)** 都有 O(1/√T) 收敛保证。

**optimizer_args 完整列表：**

```yaml
optimizer_type: "adopt"
learning_rate: 5e-5
optimizer_args:
  betas: [0.9, 0.9999]   # β2 推荐 0.9999（论文推荐可用极大 β2）
  eps: 1e-6              # 比 AdamW 的 1e-8 略大
  weight_decay: 0.0      # 论文默认 0；LoRA 可设小值 0.01
  decoupled: true        # AdamW 式解耦 WD（推荐 true）
  use_clip: true         # ADOPT-clip stability 变体（推荐 true）
  clip_exponent: 0.25    # 裁剪上界 c_t = t^exponent；论文默认 0.25
```

**参数影响：**

| 参数 | 影响 |
|---|---|
| `betas[1]` (β2) | ADOPT 的核心优势：β2 可以设得非常大（0.9999）而不破坏收敛；历史 v 积累更充分，步长更稳 |
| `eps` | 比 Adam 可设大一点（1e-6）；`v_{t-1}=0`（第一步）时用 eps 做分母，太小会导致初始步过大 |
| `decoupled` | true = AdamW 式；false = 经典 L2（梯度中加 wd·θ） |
| `use_clip` | ADOPT-clip 防数值爆炸，特别是初期 v 还小时；强烈建议 true |
| `clip_exponent` | 0.25 = 步数 t=1000 时 clip≈5.6，步数 t=10000 时 clip≈10.0，随训练逐渐放开 |

---

### 5. `lion` — Lion (NeurIPS 2023)

**算法**（Chen et al., arXiv:2302.06675）：

由程序自动搜索发现的优化器。核心是 **sign-based 更新**，每步更新量大小固定，只看方向：

```
c_t = β1·m_{t-1} + (1-β1)·g_t   # 临时混合方向
update = sign(c_t)               # 只取符号！
θ_t = θ_{t-1}·(1-lr·wd) - lr·update
m_t = β2·m_{t-1} + (1-β2)·g_t   # 更新动量（比 c_t 的 EMA 更慢）
```

**关键特性**：
- 更新量大小恒定（= lr），不受梯度缩放影响
- **仅 1 个状态张量**（m），显存 = 1× 参数量，比 Adam / SOAP 节省一半
- 需要 **LR 约为 AdamW 的 1/3~1/10**，WD 约为 **3~10×**

**optimizer_args 完整列表：**

```yaml
optimizer_type: "lion"
learning_rate: 2e-5       # 约 1/5 的 AdamW LR
optimizer_args:
  betas: [0.9, 0.99]      # 代码默认（不同于 Adam 的 0.999）
  weight_decay: 0.0
  # eps 无效（Lion 没有 eps），代码会静默丢弃
```

**参数影响：**

| 参数 | 影响 |
|---|---|
| `betas[0]` (β1) | update 方向的临时混合系数；越大越依赖历史动量 |
| `betas[1]` (β2) | 实际动量 EMA 衰减；推荐 β2 > β1（如 0.99 > 0.9） |
| `weight_decay` | 因为 sign update 步长固定，WD 效果比 Adam 强；建议 3~10× AdamW 的值 |

---

### 6. `clion` — Cautious Lion

**算法**（Liang et al., arXiv:2411.16085）：

在 Lion 基础上加一行 **Cautious 掩码**：

```
c_t = β1·m_{t-1} + (1-β1)·g_t
update = sign(c_t)

# ★ Cautious 新增：只保留与当前梯度方向一致的坐标
mask = (update * g_t > 0).float()
mask = mask / (mask.mean() + 1e-8)   # 重新归一化保持平均步长不变
update = update * mask

θ_t = θ_{t-1}·(1-lr·wd) - lr·update
m_t = β2·m_{t-1} + (1-β2)·g_t
```

**optimizer_args 完整列表：**

```yaml
optimizer_type: "clion"
learning_rate: 2e-5
optimizer_args:
  betas: [0.9, 0.99]
  weight_decay: 0.0
  cautious: true         # 代码对 clion 默认设 true；lion 下默认 false
```

Cautious 掩码在主 loss 与 aux loss（spectral / perceptual）梯度方向冲突时，会自动屏蔽冲突坐标，对多目标 loss 理论上比纯 Lion 更鲁棒。

---

### 7. `emosens` — EmoSens

**算法**（github.com/muooon/EmoSens）：

EmoSens 的独特点是 **用 loss 历史序列驱动学习率**，而非梯度或二阶矩：

```
# 维护 loss 的 3 个 EMA 时间尺度
ema_short  = 0.3·loss + 0.7·ema_short     # 快速（~3 步）
ema_medium = 0.05·loss + 0.95·ema_medium  # 中速（~20 步）
ema_long   = 0.01·loss + 0.99·ema_long    # 慢速（~100 步）

# 计算 scalar（表征 loss 是在改善还是恶化）
scalar = tanh((ema_long - ema_short) / max(ema_long, ema_medium))

# 动态 LR = emoPulse（基于信噪比估计）
noise_est = 0.97·noise_est + 0.03·|scalar|
d_est     = 0.97·d_est     + 0.03·|trust|
dNR = (d_est / noise_est)²
emo_pulse = clip(dNR · emoScope · 1e-4 · 100^c_est, min_lim, emoScope·3e-3)

# Adam 更新，lr = emo_pulse（每步动态）
m_t = β1·m + (1-β1)·g
v_t = β2·v + (1-β2)·g²
θ_t -= emo_pulse · m_t / (√v_t + ε)
```

**optimizer_args 完整列表：**

```yaml
optimizer_type: "emosens"
learning_rate: 0.1         # 这是 emoScope（LR 动态范围的上界），不是固定 LR
optimizer_args:
  betas: [0.9, 0.995]      # 代码默认；β2 略高于 Adam 默认
  eps: 1e-8
  weight_decay: 0.0
  stopcoef: 0.04           # 收敛检测阈值；medium EMA < stopcoef 且信噪比高时设 should_stop=True
  use_shadow: false        # 实验性：维护参数影子，loss 上升时部分回滚
  notify: false            # 收敛时打印 "READY TO STOP"
```

**参数影响：**

| 参数 | 影响 |
|---|---|
| `learning_rate` | 此处是 `emoScope`，是 emo_pulse 的放大基准；实际 LR = emo_pulse ≤ emoScope × 3e-3 |
| `betas` | Adam 动量系数，与普通 Adam 语义相同 |
| `stopcoef` | 收敛早停建议阈值；触发时 `should_stop` 会被设为 True（仅提示，不自动停止） |
| `use_shadow` | 维护参数的"影子拷贝"，scalar 极端时用 trust 加权回滚，实验性 |

> ⚠️ **重要**：EmoSens 需要训练脚本在每个 `optimizer.step()` 前调用 `optimizer.set_loss(loss_scalar)`，否则 emo_pulse 始终基于 `_manual_loss=0.0` 计算，等价于固定极小 LR。

---

### 8. `muon` / `muon_sf` — Newton-Schulz 正交化动量（±Schedule-Free）

**核心思想**（Keller Jordan 2024；Moonlight arXiv:2502.16982）：2D 参数的动量矩阵经
5 步 Newton-Schulz 迭代近似极分解 UV^T（全部奇异值→1），得到类 Shampoo 的谱预条件，
但优化器状态只有动量（+SF 的 z/y），无 GG/Q 矩阵。1D 参数（bias、DoRA scale）走
AdamW fallback。`muon_sf` 在此之上叠 Schedule-Free 平均（`lr_scheduler` 必须 `none`，
任意 step `eval()` 拿平均 checkpoint；短跑 ≤100 步的滞后警告同 soap_sf）。

> ⛔ **2026-07-18 实测结案：muon / muon_sf 不要用于 LoRA 画风训练。**
> Krea2 c12port 配方下换 muon_sf 后几乎不拟合，lr 从 5e-5 一路试到 5e-4 全部无效。
> 根因由同 epoch9 checkpoint 逐层取证锁定（`tools/lora_delta_forensics.py`）：
>
> | | 总 ‖ΔW‖² | top1 能量 | 谱条件数 | 有效秩 | 最强层 |
> |---|---|---|---|---|---|
> | adamw | 112.4 | **57.7%** | **22.1** | 8.3 | top1 98%, s_max 2.42 |
> | muon_sf | 44.5 | **8.9%** | **1.7** | 25.0 | top1 3.5%, s_max 0.162 |
>
> Newton-Schulz 每步把更新矩阵的**所有奇异值拉到 1**——这是谱预条件的设计目的，
> 对预训练全模型是优点，但 LoRA 画风增量的真实目标每层近似 **rank-1**（adamw 最强层
> top1 占 98%），NS 会主动把预算平摊到全部 48 个方向。主方向幅度差 **15×**，
> 这就是"不拟合"。**调 lr 治不了**：lr 只等比缩放全部奇异值（尺度问题），而这是
> 形状问题——所以 5e-5→5e-4 全程无效。
>
> 附带修正：下方"直接用 AdamW 的经验值"这句**本身也不准确**。本地实测（真实
> LoRA 形状拟合 rank-32 目标）追平 AdamW 5e-5 需要 muon_sf ≈ 3e-4，约 **4–6×**，
> 而非 1:1。位移量级确实对齐（1.5× 内），但 ΔW=B@A 是两因子乘积，位移对齐 ≠
> 拟合效率对齐。保留此条仅为历史记录，不建议按它调参。

```yaml
optimizer_type: "muon_sf"        # 或 "muon"
learning_rate: 1e-4              # ⚠ 见上方警告框：此口径不准确，实测需 4–6×
lr_scheduler: "none"
optimizer_args:
  betas: [0.9, 0.95]             # muon_sf: [SF 插值权重, 1D 二阶矩衰减]；muon: 1D AdamW betas
  momentum: 0.95                 # NS 输入的内层动量（0 关闭=喂裸梯度，不推荐——放大 batch 噪声）
  ns_steps: 5                    # Newton-Schulz 迭代步数（10 更精确但无实测收益）
  weight_decay: 0.0
  warmup_steps: 50               # 仅 muon_sf；线性 warmup 稳定早期
  rms_scale: "moonlight"         # 默认；"keller" 仅作历史对照（见下）
```

**参数影响：**

| 参数 | 影响 |
|---|---|
| `learning_rate` | moonlight 缩放使任意形状矩阵的更新条目 RMS ≡ 0.2×lr，语义与 AdamW lr 可比；从 AdamW 已验证值（如 1e-4）起步 |
| `momentum` | NS 之前的梯度平滑；SF 的 Polyak 平均替代的是 lr schedule，替代不了它（ScheduleFree+ arXiv:2605.19095） |
| `rms_scale` | `moonlight`=0.2·√max(rows,cols)（形状无关）；`keller`=√max(1,rows/cols) 只放大高瘦矩阵——对 LoRA 矮宽 down (r,in) 缩小 ~√(in/r) 倍 |

> ⚠️ **2026-07-11 修复（krea2 LoRA 实测"完全不拟合"根因，勿回退）**：
> ① 旧实现用 keller 缩放但文档声称 Kimi 口径——LoRA down (24,6144) 的更新被缩小 ~16 倍；
> ② 更新直接写 bf16 参数，每步增量低于 bf16 ulp（~3e-5 @ 6e-3 量级），78~93% 条目被
> 四舍五入永久冻结。现更新一律先落 fp32 master（muon: `state["master"]`；muon_sf:
> `state["y"]`），bf16 参数只是每步刷新的视图。数值复现：同 lr=1e-4 下修复前 AdamW
> 对 down 矩阵的位移是 muon_sf 的 165 倍，修复后 bf16≡fp32、量级与 AdamW 相当
> （tests/test_pissa_dora_muon_fixes.py::TestMuonRMSScaleAndMaster）。

---

### 9. `automagic` — 逐元素自适应学习率

**核心思想**（移植自 [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit)
`toolkit/optimizers/automagic.py`，Apache-2.0）：**每个权重一个独立的学习率**。
本步更新符号与上步一致 → `lr += lr_bump`；符号翻转 → `lr -= lr_bump`；钳在
`[min_lr, max_lr]`。直觉是：梯度方向被数据稳定支持的权重自动加速，被噪声驱动、
符号来回翻转的权重 lr 衰减到 `min_lr`，**等于自动饿死噪声方向**。

二阶矩用 **Adafactor 分解**（2D 参数只存行/列两个向量），并对更新做 RMS 裁剪
（`clip_threshold=1.0`），量级语义与 AdamW 的 `m/sqrt(v)` (RMS≈1) 对齐。
**无一阶动量**（与 AdamW/Lion 的关键差异，逐步更新噪声更大）。

```yaml
optimizer_type: "automagic"
learning_rate: 1.0e-6          # ★这是【起始 lr】，不是训练强度
lr_scheduler: "none"           # 自管逐权重 lr，外部调度器会被强制关掉
optimizer_args:
  min_lr: 1.0e-7
  max_lr: 1.0e-4               # ★真正的强度旋钮（方向稳定的权重最终顶到这里）
  lr_bump: 1.0e-6              # 每步调整量；从 lr 爬到 max_lr 需 (max_lr-lr)/lr_bump 步
  beta2: 0.999
  clip_threshold: 1.0
  weight_decay: 0.00001        # 解耦，按逐元素 lr 缩放
```

**参数影响：**

| 参数 | 影响 |
|---|---|
| `learning_rate` | **仅是起点**。构造期校验它必须落在 `[min_lr, max_lr]`，否则 fail-fast |
| `max_lr` | 真正决定训练强度。RMS 裁剪使量级语义≈AdamW，可直接沿用 AdamW 已验证值 |
| `lr_bump` | 爬升速度。太小则整个 run 都没爬到位（判读时先看 avg_lr，别误判成不拟合） |
| `beta2` | 分解二阶矩衰减 |

**判读**：日志会打 `avg_lr`（`optimizer.get_avg_learning_rate()`）。
- avg_lr 绝大多数顶到 `max_lr` → 退化成固定 lr 的 Adafactor，逐元素机制没起作用；
- avg_lr 明显低于 `max_lr` 且分布分化 → 机制在工作。

> ⚠️ **本仓库改动：fp32 master（非上游写法）**。本仓库 LoRA 可训练参数是 bf16，
> 起始 lr=1e-6 时每步更新 ≈1e-6，**低于 bf16 在 LoRA 参数量级(~6e-3)的 ulp(~3e-5)**，
> 直接写 bf16 会被四舍五入全部吞掉——即 muon_sf 踩过的同款坑。上游用随机舍入
> (`copy_stochastic`) 规避，本仓库统一用 fp32 master（`state["master"]`）。
> `lr_mask` 同样强制 fp32：bf16 尾数仅 8 位，`1e-4 + 1e-6` 会直接舍回 1e-4，
> lr_bump 机制会被静默废掉。回归测试见
> `tests/test_automagic_optimizer.py::TestBf16MasterNoFreeze`。

> **显存**：master 4B + lr_mask 4B + polarity 1B ≈ **9 B/param**，与 AdamW 的
> 8 B/param 基本持平（本仓库 LoRA ~1.14 亿参数 → 约 1.0 GB）。
> **不是省显存方案**——LoRA 训练的显存大头是激活（navit 实测 10GB + 0.52MB/token），
> 优化器状态从来不是瓶颈。

---

## 三、全参数对照表

### 共有参数（所有优化器均支持，写在 optimizer_args 下）

| 参数 | adamw | adamw8bit | prodigyplus | soap | adopt | lion/clion | emosens |
|---|---|---|---|---|---|---|---|
| `betas` 默认值 | [0.9, 0.999] | [0.9, 0.999] | **[0.95, 0.99]** | **[0.95, 0.95]** | **[0.9, 0.9999]** | **[0.9, 0.99]** | **[0.9, 0.995]** |
| `weight_decay` 默认值 | 0.01 | 0.01 | 0.0 | 0.01 | 0.0 | 0.0 | 0.0 |
| `eps` 默认値 | 1e-8 | 1e-8 | **null** | 1e-8 | **1e-6** | ❌ 无 | 1e-8 |

加粗值是代码对该优化器自动设置的非通用默认值（若不指定 betas/eps，工厂函数会自动覆盖）。

### 独有参数速查

| 优化器 | 独有参数 | 默认值 | 关键作用 |
|---|---|---|---|
| `adamw` | `amsgrad` | false | 使用历史最大 v 保证收敛 |
| `adamw` | `foreach` | false | 融合 kernel 提速 |
| `adamw` | `fused` | false | CUDA fused kernel |
| `adamw8bit` | `min_8bit_size` | 4096 | 小参数不量化 |
| `prodigyplus` | `d0` | 1e-6 | d 初始猜测（建议 2e-5~1e-4） |
| `prodigyplus` | `d_coef` | 1.0 | d 放大系数 |
| `prodigyplus` | `d_limiter` | true | 限制 d 单步增幅 |
| `prodigyplus` | `prodigy_steps` | 0 | 固化 LR 的步数（0=全程自适应） |
| `prodigyplus` | `use_schedulefree` | true | 消除 scheduler |
| `prodigyplus` | `use_stableadamw` | true | RMS 梯度归一化 |
| `prodigyplus` | `schedulefree_c` | 0 | SF warmup 常数 |
| `prodigyplus` | `use_speed` | false | 速度自适应（实验性） |
| `prodigyplus` | `use_bias_correction` | false | Adam 偏差校正 |
| `prodigyplus` | `factored` | false | 分解二阶矩 |
| `soap` | `shampoo_beta` | -1 | GG EMA（-1=复用 β2） |
| `soap` | `precondition_frequency` | 10 | 特征基更新频率（步） |
| `soap` | `max_precond_dim` | 10000 | Shampoo 适用的最大维度 |
| `soap` | `merge_dims` | false | 合并小维后预条件 |
| `soap` | `precondition_1d` | false | 对 1D 向量预条件 |
| `soap` | `normalize_grads` | false | update RMS 归一化 |
| `soap` | `correct_bias` | true | bias correction |
| `soap` | `max_precond_dim` | 256~512 | ★ state 体积头号开关（GG/Q 对维度二次增长） |
| `soap` | `precond_in_state` | true | false=不存盘 GG/Q（resume 冷重建），砍 state |
| `soap_sf` | `betas[0]` | 0.9 | SF 插值动量（非一阶矩；eval 除以它，须 ∈(0,1)） |
| `soap_sf` | `weight_lr_power` | 2.0 | 平均权重里 lr 的幂 |
| `soap_sf` | `r` | 0.0 | 迭代序号幂；调高让 x 贴近最新 z（短跑用） |
| `soap_sf` | `warmup_steps` | 0 | 线性 lr warmup（SF 通常不需要） |
| `soap_sf` | *(其余同 soap)* | — | shampoo_beta / precondition_* / max_precond_dim 等沿用 |
| `adopt` | `decoupled` | true | AdamW 式解耦 WD |
| `adopt` | `use_clip` | true | ADOPT-clip 稳定变体 |
| `adopt` | `clip_exponent` | 0.25 | clip 上界随 step^exp 增长 |
| `lion/clion` | `cautious` | false/true | Cautious 方向掩码 |
| `emosens` | `stopcoef` | 0.04 | 收敛检测阈值 |
| `emosens` | `use_shadow` | false | 参数影子回滚 |
| `emosens` | `notify` | false | 收敛时打印提示 |

---

## 四、选型建议

| 场景 | 推荐优化器 | 核心理由 |
|---|---|---|
| **LoRA-on-DiT（主流程）** | `prodigyplus` | LR 全自动，无需调参；Schedule-Free + StableAdamW 对 bf16 大分辨率极友好 |
| **想要固定 LR 做对照** | `adamw` | 清晰基线，lr=5e-5 起 |
| **显存极紧张** | `adamw8bit` + `lion` | 8bit 省一半优化器显存；Lion 只存 1 个 moment |
| **梯度噪声特别大（多 aux loss）** | `adopt` | β2=0.9999 下仍收敛，ADOPT-clip 抗尖峰 |
| **实验性：梯度协方差利用** | `soap` | 小矩阵 LoKr 下快速预条件 |
| **SOAP 的拟合速度 + 开放式无终点训练** | `soap_sf` | 保留 SOAP 预条件，去掉 LR 调度依赖；任意 step 停下 `eval()` 都得到良好平均 checkpoint |
| **极短训练（≤60~100 步）** | `soap`（**不要** soap_sf） | SF 的 Polyak 平均在短跑严重滞后，x≈轨迹质心=欠拟合；用纯 SOAP + 短 cosine 或常 lr |
