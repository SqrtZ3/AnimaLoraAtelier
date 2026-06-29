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
```

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

**互斥（同时开 → 启动即 fail-fast 报错，提示显式关掉）**：`token_bucket`/ARB 分桶、
`effective_batch_size` 样本窗口累积、`tread_enabled`、`leap_enabled`、`gaf_enabled`、
`dpo_enabled`、`timestep_sampling=csflow`、`torch_compile`（动态 pack 形状与固定图冲突）、
以及 `dfm_lambda>0` / `eisbach_lambda>0` / `dispersive_enabled` / `adaptive_timestep`
——后四个在 NaViT 路径会被 `_skip_main_extras` 跳过（它们假设批量网格 / 逐 batch 单 t /
batch 内负样本），**为避免"开着却悄悄不生效"的隐性行为改变，一律 fail-fast 要求显式关闭**。
这些日后可逐个适配 NaViT，不在 v1 范围。

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
