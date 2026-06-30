# NaViT / Patch-n-Pack 块对角打包训练

> 状态：**全链路已落地**——模型内核 + 数据/目标核心（本地单测通过）+ 训练循环接线与
> 逐图 aux（py_compile + 全套 263 单测过、静态扫描已 guard，**真训练待云端 smoke**）。
> opt-in / default-off。关掉时与改动前逐字节等价。

## 1. 解决什么问题

小数据集 + 多分辨率 + 想开高 batch 时，现有两条路都卡：

- **ARB / token_bucket（`BucketBatchSampler`）** 按**精确 `(h, w)`** 分桶（[`data.py` `_fill_bucket_for_index`](../trainer/data.py)）。
  分辨率一多 → 精确尺寸桶一多 → 每桶几张图 → 填不满高 batch。哪怕 token_bucket 把
  token 数收敛到 6 个，同一 token 数的不同宽高比仍裂成多个精确尺寸桶。
- **纯 padded FiT（`FitTokenBatchSampler`）** 按 token 数聚批后 pad 到本批最大 N，
  `_build_packed_masks` 退化成稠密 additive key-padding 掩码 → SDPA 带掩码内核、丢掉
  xformers/flash 快路径 → 慢。

**NaViT/Patch-n-Pack（arXiv 2307.06304）** 用真正的 *example packing*：把**多张异构图**拼进
一条序列，用**块对角注意力掩码**让每图只注意自己的 token（self）与自己的 caption（cross），
每图带**自己的 timestep**。于是“每步处理多少图”与“单图形状”彻底解耦——小数据集也能用
固定 token 预算填满任意有效 batch，且**零 padding、走 xformers varlen 快内核**。

对照：**FiT/FiTv2（arXiv 2402.12376）** 用的是 padding 打包（本仓库现有的 fit_packed 路径）；
NaViT 用块对角 packing。本实现选 NaViT 以彻底解耦形状与 batch。

## 2. 怎么开

```yaml
navit_packing: true            # 总开关（默认 false）
navit_token_budget: 16384      # 一个 pack 的 token 数之和上限（见下方显存对照表，必须按卡设）
navit_max_images_per_pack: 0   # 单 pack 最多几张图，0=不限（仅受 token_budget 约束）
```

### 提速/打包旋钮（均 opt-in，关时与上面的基础路径逐字节等价）

```yaml
navit_text_trim_padding: false # 块对角 cross-attn 按每图 T5 有效长度打包文本，去 512-pad。
                               #   开启 = 不再对文本 padding 位做注意力（cross-attn 提速，
                               #   anime tag caption 通常仅几十 token → 文本侧 token 砍约一个量级）。
                               #   小行为改变：标准/ARB 路径本就注意 padding 文本位，开启后不再注意。
navit_pack_strategy: next_fit  # 打包策略：next_fit（默认，顺序贪心）/ ffd（窗口内 First-Fit-
                               #   Decreasing，包更满、step 更少）。next-fit 在"图相对 budget 很小"时
                               #   已接近满（收尾浪费 ≤ 一张图）；ffd 主要在图尺寸异质时收益大。
navit_pack_ffd_window: 256     # ffd 的窗口大小（张）：每 epoch 洗牌后切窗、窗内 FFD，使包仍逐 epoch
                               #   变化（保 SGD 多样性）。0=全局窗口（最满但 epoch 间包固定）。
navit_drop_last: false         # 是否丢弃每 epoch 最后那个未满预算的包。默认不丢（打包路径下末包
                               #   总含真实图，丢了在小数据上是浪费）。与 bucket_drop_last 解耦。
navit_native_resolution: false # 单图按原生分辨率定尺寸，只受 VAE+patch 的 16px 整倍数约束，
                               #   解开 ARB 分桶对单图尺寸的量化（见下方“2.1 原生分辨率”）。
```

### 2.1 原生分辨率（`navit_native_resolution`，opt-in）

默认（关）时，navit 打包仍**用 ARB 桶给每张图定尺寸**：每图被 resize + 中心裁到最近的桶
`(h,w)` 再编码 latent，打包只解耦“每步几张图”，单图尺寸仍被桶网格量化。

开启 `navit_native_resolution: true` 后，单图改走**原生定尺寸**（复用 FiT 的
`plan_native_fit_image`，对齐单元 = `patch(2) × vae_downsample(8) = 16px`）：

- **不再 resize、不再按桶量化**，每张图保留自己的宽高比与近似原生尺寸；
- **强制 `floor` 对齐**——把每边裁到 16 的下整倍数（每边丢 ≤15px），**零 gray padding**。
  这点很关键：navit 的缓存路径（`CachedLatentDataset` / `collate_fn_navit_pack`）**不携带
  padding mask**，floor 保证有效区填满整张 latent（mask 恒为全 1），不会把灰边当内容训练。
  对比 `pad/ceil` 会补灰边、需要逐图 mask，v1 不走这条；
- **需 `cache_latents: true`**（与 navit 打包本身的要求一致：打包按每图 latent token 数分包）。

**仍存在的真实上界（不是 16px，而是 RoPE 单维上限）：** 每张图的单边 latent token 数必须
≤ 模型 RoPE 的 `max_h/max_w`（即单边像素 ≤ `max_img_h × 8`）。`max_img_h/max_img_w` 为 0（自动）
时，会**预扫数据集取最大单边自动推**；超限会在缓存编码前 fail-fast 并提示提高 `max_img_h/w`
或在数据集端裁掉超大图。`navit_token_budget` 仍须 ≥ 最大单图 token 数。

**关于 NaViT 论文的 “fractional PE（位置归一化到 [0,1]）”——本仓库不引入。** 该技巧是为
**可学习的绝对加性位置嵌入**（固定大小 learned table 需跨分辨率插值）设计的；本模型用的是
**RoPE3D**（`pos_emb_cls="rope3d"`，整数网格位置 + NTK 外推，见
[`models/anima_modeling_core.py`](../models/anima_modeling_core.py) 的 `RopePosEmbed*` / `_packed_rope_from_grid`），
本就按整数位置天然外推多分辨率，靠 NTK 系数与 `max_h/max_w` 承担，不需要归一化。且我们是在
**冻结底模（Cosmos DiT，整数网格 RoPE 预训练）上训 LoRA**，把位置改成 [0,1] 归一化会给冻结权重喂
OOD 的位置信号 → 大概率掉点。真正让“任意分辨率”成立的是 **原生 16px 定尺寸 + RoPE 现成的整数网格
外推（在 max_h/w 内）**，与 fractional PE 无关。

token 数换算：Anima 是 VAE 下采样 8 × patch 2 = **16 px/token 轴**，所以一张 `W×H` 图的
token 数 `N = (W//16) × (H//16)`。例：1024² ≈ 4096 token；768×1024 ≈ 3072 token。

### 显存 ↔ token_budget 对照（先按此起步，再按实测微调）

token_budget 决定单步打包序列的最大长度，自注意力显存 ∝ budget（块对角 varlen，不是 N²
全矩阵，但激活仍随总 token 数线性涨）。下表是**保守起点**，**不是实测定值**——你的卡、
rank、grad_checkpoint、底模大小（2048 vs 5120 通道）都会左右它，**务必首跑观察峰值显存再调**：

| 显存 | 起步 token_budget | 约等于（4096-token/张） |
|---|---|---|
下表是 **`grad_checkpoint: true`（推荐，navit 已接逐块梯度检查点）** 下的保守起点——峰值激活
≈ 1 个 block 而非 28 个 block，所以同样显存能装的 budget 比无 checkpoint 大一个量级。**不是
实测定值**，你的卡 / rank / 底模通道（2048 vs 5120）都会左右它，**首跑看峰值显存再调**：

| 显存 | 起步 token_budget（grad_checkpoint=true） | 约等于（4096-token/张） |
|---|---|---|
| 16 GB | 16384 | ~4 张 |
| 24 GB | 32768 | ~8 张 |
| 32 GB | 49152 | ~12 张 |
| 48 GB | 65536 | ~16 张 |
| 80 GB | 98304 | ~24 张 |
| 96 GB | 131072 | ~32 张 |

- `navit_token_budget` 必须 **≥ 最大单图 token 数**，否则那张图单独成包且可能超预算/OOM
  （sampler 会 warn）。
- **`grad_checkpoint: false`**：峰值激活 ≈ 28×，上表 budget 要砍到约 1/8–1/10（如 96 GB 从
  131072 降到 ~16384）。除非要省 backward 重算的算力，否则建议保持 true。
- 想要更平滑的梯度：用 `grad_accum` 跨步累积到目标有效 batch（navit 下 `batch_size` 被忽略，
  每步图片数由 budget 决定）。

## 3. v1 支持范围与门控（重要）

NaViT 改变了 batch 语义（一包异构图、逐图 t），许多按“`[B,C,T,h,w]` 批量网格 + 逐 batch 单
timestep”假设写的特性会语义错位。v1 的策略：

**支持**：basic flow-matching（逐图 t 采样 + 噪声 + masked token loss）、`grad_accum`、
**逐块梯度检查点（`grad_checkpoint`，与 fit-packed 同策略）**、LoRA 保存/恢复、训练中采样出图、
**aux_losses 中的 spectral / perceptual（逐图 unpatchify 回网格后按既有 aux 数学计算）**。
（self-perceptual 需逐图额外模型前向、尚未适配；dispersive 见下方互斥项。）

**已逐图适配（opt-in，与 dense 同数学，但按 pack 内每图独立施加）**：

- **`eisbach_lambda>0`（Eisbach log-barrier）**：逐图 unpatchify pred 回网格 → 空间能量障碍
  权重（detached，只缩 step、不改方向）→ 乘到该图带梯度的逐图 loss。全模式放行。
- **`dfm_lambda>0`（ΔFM）——仅 `dfm_mode=vecor`**：vecor 负样本是对该图自身 target 的破坏性
  增强（通道乱序 / 裁剪缩放），自洽逐图。**`dfm_mode=batch`（默认）在 NaViT 下仍 fail-fast**：
  batch 模式要 `||v_i − target_j||²` 跨样本配对，而一个 pack 内各图形状不同、对不上。
- **`adaptive_timestep`——`metric=raw/slope/entropy_rate`**：采样侧 NaViT 本就逐图走
  `adaptive_ts.sample(bs=G)`，这里补控制器 `update(t, 干净逐图重建 loss)`（用未被 eisbach/vecor
  乘过的 `per_image_loss`，与 dense 的 `_adaptive_raw` 快照语义一致）。**`metric=highfreq/mixed`
  仍 fail-fast**：它们需逐图高频残差网格，v1 未适配。

**互斥（同时开 → 启动即 fail-fast 报错，提示显式关掉）**：`token_bucket`/ARB 分桶、
`effective_batch_size` 样本窗口累积、`tread_enabled`、`leap_enabled`、`gaf_enabled`、
`dpo_enabled`、`timestep_sampling=csflow`、`torch_compile`（动态 pack 形状与固定图冲突）、
`dispersive_enabled`，以及 `dfm_mode=batch` 下的 `dfm_lambda>0` /
`adaptive_timestep metric=highfreq|mixed`（见上）。这些（尚未适配的部分）在 NaViT 路径会被
`_skip_main_extras` 跳过——**为避免"开着却悄悄不生效"的隐性行为改变，一律 fail-fast 要求显式
关闭或切到已适配的模式**。其余日后可逐个适配 NaViT，不在 v1 范围。

## 4. 实现地图（已落地部分）

| 层 | 符号 | 文件 |
|---|---|---|
| 注意力 op 块对角分支 | `torch_attention_op` 接 `BlockDiagonalMask`；`_is_xformers_attn_bias` | `models/anima_modeling_core.py` |
| 逐图 timestep 调制 | `Block.forward_tokens` / `FinalLayer.forward_tokens` 的 `token_wise_mod` / `cross_attn_mask` | 同上 |
| 打包前向 | `MiniTrainDIT.forward_packed_navit`（块对角 self/cross + 逐图 RoPE + 逐图 AdaLN） | 同上 |
| token 预算打包 | `pack_indices_by_budget` / `NavitPackBatchSampler` / `collate_fn_navit_pack` | `trainer/data.py` |
| 训练步核心 | `navit_packed_forward_and_loss`（逐图加噪 → 打包前向 → 逐图 loss） | `trainer/objective.py` |
| config 键 | `navit_packing` / `navit_token_budget` / `navit_max_images_per_pack` | `trainer/config.py` |

### 验证状态（诚实标注）

本地 GPU（CUDA + xformers 0.0.30，head_dim 64/128）+ 纯 Python 单测，**已通过**：

- `test_packed_block_diag_attention`：块对角 self/cross attention ≡ 各图独立 attention 拼接（零跨图泄漏）。
- `test_packed_navit_forward`：`forward_packed_navit` ≡ 各图单独 `forward_packed_tokens` 拼接；
  逐块梯度检查点（`use_checkpoint=true`）≡ 非检查点输出且可反向。
- `test_navit_pack_sampler`：打包预算/覆盖/超大图/张数上限/跨 epoch 重洗。
- `test_navit_packed_objective`：训练步前向+loss+反向梯度有限；逐图 loss 与手算一致。

**尚未本地验证（云端 smoke 必做）**：训练循环端到端接线、aux 逐图路径、真模型
（head_dim 128，cross q/kv 源维 2048/1024 不等）首跑、云端 `_USE_XFORMERS` 是否开
（关了会 fail-fast 报错，不会静默退化为 O(N²)）、与 LoRA/LoKr 注入的实跑交互。

## 5. 首次 A/B 建议

固定其它一切，仅切 `navit_packing`（关 = 现有 token_bucket / ARB 路径），对比：
单步吞吐（imgs/s）、峰值显存、相同 wall-clock 下若干 step 的 loss 曲线与采样图。
NaViT 的预期收益是**吞吐 / 有效 batch 填充**，不是画质本身——画质应与等价超参下持平。
