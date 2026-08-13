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
navit_text_trim_padding: false # ⚠ 实测有害，保持 false。见下方“2.0 为什么不要开 text-trim”。
navit_pack_strategy: next_fit  # 打包策略：next_fit（默认，顺序贪心）/ ffd（窗口内 First-Fit-
                               #   Decreasing，包更满、step 更少）。next-fit 在"图相对 budget 很小"时
                               #   已接近满（收尾浪费 ≤ 一张图）；ffd 主要在图尺寸异质时收益大。
navit_pack_ffd_window: 256     # ffd 的窗口大小（张）：每 epoch 洗牌后切窗、窗内 FFD，使包仍逐 epoch
                               #   变化（保 SGD 多样性）。0=全局窗口（最满但 epoch 间包固定）。
navit_pack_cost_lambda: 0.0    # 按代价装包（0=关，与改动前逐包等价）。见下方“2.4 按代价装包”。
navit_pack_cost_ref_tokens: 0  # 代价归一的参考尺寸（token）；0=自动取数据集中位数。
navit_pack_token_cap: 0        # 双约束的第二条：ΣN 上限（管显存）。0=自动=token_budget。
                               #   仅在 cost_lambda>0 时生效。见“2.4.1 为什么代价预算不够”。
navit_drop_last: false         # 是否丢弃每 epoch 最后那个未满预算的包。默认不丢（打包路径下末包
                               #   总含真实图，丢了在小数据上是浪费）。与 bucket_drop_last 解耦。
navit_native_resolution: false # 单图按原生分辨率定尺寸，只受 VAE+patch 的 16px 整倍数约束，
                               #   解开 ARB 分桶对单图尺寸的量化（见下方“2.1 原生分辨率”）。
navit_multiscale: false        # 多尺度阶梯：为大图追加低 token 档等比缩小副本参与打包，
                               #   填满大图包剩余预算 + 缓解大图训练/小图推理的尺度偏移
                               #   （见下方“2.2 多尺度阶梯”；需 navit_native_resolution）。
```

### 2.0 为什么不要开 `navit_text_trim_padding`

这个开关早期被描述为“小行为改变换 cross-attn 提速”。**两边都测过之后，这个权衡不成立**：

- **收益侧接近零。** 真实训练的 `stage_timing.csv`（Krea2 12B / navit / budget≈55k token）里
  `text_encode` 只占整步 **0.23%**，而文本 token 在单流序列里只是 ΣL 的一小截；trim 能省的
  量级在零点几个百分点。
- **代价侧是已实证的训练/评估条件不一致。** 训练时去掉 512-pad、而 eval / 采样 / ARB 路径
  仍带 pad → cross-attn 的条件分布不一致，A/B 实测表现为 eval_loss 冲高 + 拟合变差；关掉后
  eval_loss 恢复单调下降并追平 ARB 基线。

也就是说这不是“提速 vs 轻微行为改变”的取舍，而是**收益≈0、代价已知为负**。保持 `false`；
训练启动时若检测到它被打开会打 warning。

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

### 2.2 多尺度阶梯（`navit_multiscale`，opt-in）

```yaml
navit_multiscale: true              # 需 navit_packing + navit_native_resolution；默认 false
navit_multiscale_token_ladder: "4096"  # 副本 token 档（逗号分隔或 YAML 列表）；每档 ≤ token_budget
navit_multiscale_loss_weight: 1.0   # 副本逐图 loss 权重（原生恒 1.0）；1.0=等权
```

**解决什么问题：** 原生大图相对 budget 很大时（如 2814×4456 ≈ 48.6k token、budget 65536），
每包只装得下一张、尾部 ~26% 预算浪费；且 LoRA 只见过原生尺度的画风统计——小分辨率推理时
是分布外（train-large / infer-small 尺度偏移）。NaViT 论文（arXiv 2307.06304）的
resolution-sampling 是同一思路的随机版。

**怎么做：** 对原生 token 数超过阶梯档的每张图，缓存阶段额外编码一份**等比缩小**副本
（floor 对齐 16px、resize-cover + 中心裁剪 → 零 padding、mask 恒全 1），作为**正式数据集
条目**参与打包。展开是**确定性**的：每图每档每 epoch 恰好出现一次（而非随机填充——可复现、
可归因）；配合 `navit_pack_strategy: ffd` 会自然装出"1 大图 + N 小副本"的高填充包。
**只降不升采样**：源图 token 数 ≤ 档位的条目跳过该档。副本 caption 与原生共享。

**账目示例（纯 2814×4456 数据集 + budget 65536 + 阶梯 4096）：** 每图 48,650(原生) +
~4,000(副本) token；FFD 装包 ≈ 1 原生 + 4 副本 ≈ 64.9k/包，填充率 ~99%（原 ~74%）。

**注意：**

- 逐图 loss 是等权的（`per_image.mean()`）——1 张大图 + 4 张副本时原生只占 1/5 梯度权重。
  想让原生尺度主导，把 `navit_multiscale_loss_weight` 调 <1.0（只作用于副本）。
- 缓存体积 ×(1+命中档数)（flip 再 ×2）；副本 npz 是独立 sidecar（`<stem>.ms<档>.npz`），
  开关 multiscale 不会使原生缓存失效。
- eval loss 的随机子集取自展开后的数据集，副本会进入 eval 分布（各尺度都被评到）。
- 超大原生图**首次**缓存编码的 VAE 峰值显存不因此下降（原生份仍按原生编码）；
  解法见下方 2.3 `cache_encode_tiled`。

### 2.4 按代价装包（`navit_pack_cost_lambda`，opt-in）

**解决什么问题：** `navit_token_budget` 只约束 ΣN，但**一个包的步时不是 ΣN 的线性函数**——
attention 对每张图**自身**的序列长度是二次的，所以同样的 ΣN，装少数大图远比装很多小图贵。

实测（H20 / Krea2 12B / 一个真 `SingleStreamBlock`，`tests/diag_navit_speed.py` S5，
ΣN=55778 固定）：

| G | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| block fwd+bwd | 3960 ms | 2749 | 2150 | 1851 | **1719** |

**同样的 token 数，G=1 比 G=16 慢 2.3×。** 把 `t = a·ΣN + b·ΣN_i²` 拟合到这张网格得
R²=0.9999、`λ = b/a = 2.742e-05`；再独立地把同一张卡上真实训练的 `stage_timing.csv`
按整步拟合，得 `λ = 3.279e-05`（R²=0.972）——两条路径相差 16%，互为佐证。
（同一半经验形式见 KnapFormer，arXiv 2508.06001：`k·(24·L·d² + 4γ·L²·d)`。）

**怎么做：** 装包时把每张图的“体积”从 token 数换成

```
cost(n) = n · (1 + λ·n) / (1 + λ·n_ref)
```

`n_ref` 归一的用意是**行为中立的起点**：恰好 `n_ref` 大小的图代价等于它的 token 数，所以
尺寸均匀的数据集容量不变，只有尺寸**差异**被重新定价——大图占更多预算（每包少装 → 消掉
步时与显存尖峰），小图占更少（每包多装 → 提吞吐）。若不做归一，开 λ 会让**所有**包一起
缩水，看起来像退步而不是再平衡。

**怎么设：**

- `navit_pack_cost_lambda`：Krea2 12B 在 H20 上是 **2.7e-05**；换模型/换卡请用
  `tests/diag_navit_speed.py --only s5` 重新拟合（它直接打印推荐值）。
- `navit_pack_cost_ref_tokens`：留 0（自动取数据集 token 中位数）即可；想让某个尺寸档
  严格保持现有容量时才手动指定。
- `0.0`（默认）= 关闭，装包结果与改动前**逐包等价**。

**收益量级：** 以 λ=2.74e-05 计，4096-token 图的代价系数是 1.112、13944-token 图是 1.382。
对尺寸均匀的数据集近似无变化；**主要价值是消除 `G=1` 那种 2.3× 的步时/显存尖峰**。

**注意：** 这只改变**批的组成**，不改任何数学；但批组成会影响 SGD 统计（和 `ffd` 同性质），
所以首次开启建议按单变量 A/B 跑（只切这一个键），对比 imgs/s、峰值显存与 loss 曲线。

### 2.4.1 双约束：为什么只卡代价预算不够（`navit_pack_token_cap`）

**此前这里写着"小图为主的包能多装约 24% 的 token"，并把它当成纯收益——那句话漏了代价侧。**
`cost(n)` 对**小于 `n_ref`** 的图低于其 token 数，所以只测试 `Σcost ≤ budget` 时，
一包小图的 **ΣN 会超出 `navit_token_budget`**，倍数上界是

```
ΣN_max / budget = (1 + λ·n_ref) / (1 + λ·n_min)
```

λ=2.742e-05、n_ref=13944、n=4096 时是 **1.243×**。本地实跑（`tests/test_navit_pack_token_cap.py`
的同一数据集，budget=49152、全 3072~4608 token 的小图、ffd）实测 **ΣN 峰值 61440 = 1.250×
budget**。而步显存对 ΣN 线性（memory 实测 ≈ 10GB + 0.52MB/token），**12288 个超额 token
≈ +6.4GB 峰值显存**——显存对照表承诺的还是 budget 那一档。所以那不是白捡的吞吐，
是**未入账的显存超支**（本质是拿显存换吞吐，只是没写在账上）。

修法与 AdaptiveLoad（arXiv 2605.17923）的双约束一致——计算上限与显存上限同时卡
（`B = max(1, min(⌊M_mem/S⌋, ⌊M_comp/S^p⌋))`）。装包现在要求**两条同时成立**：

```
Σcost ≤ navit_token_budget      （管步时；attention 二次项）
ΣN    ≤ navit_pack_token_cap    （管显存；激活对 ΣN 线性）
```

- `navit_pack_token_cap: 0`（默认）= 自动取 `navit_token_budget` → ΣN 不再越过预算。
- 想**明知地**拿显存换吞吐，就把它显式设成大于 budget 的值（例如 `1.25 × budget`），
  这时超支是你选的、写在配置里的，而不是隐式发生的。
- `cost_lambda = 0` 时 cost ≡ token，两条约束是同一条，本键不生效，**装包结果逐包等价**。

（历史说明：改动落地时仓库里没有任何 config 设过 `navit_pack_cost_lambda > 0`，
所以没有既有 run 的行为被改变。）

### 2.5 用预算换重算：`grad_checkpoint_policy`（opt-in）

**先记住一条实测事实：** 块对角注意力下，**per-token 代价只取决于每张图自己的 seqlen，
与一个包里装几张图无关**——同一策略在 G=1/2/3/6 下 ms/token 变化 <1%（H20，28 块真栈，
`tests/diag_navit_ckpt_policy.py --g-sweep`）。

所以 **`navit_token_budget` 是纯粹的显存/粒度旋钮，降它不损吞吐**。而 checkpoint 策略
恰恰只被显存卡住——这就构成一笔交换：**降预算 → 腾显存 → 换更便宜的重算策略 → 净提吞吐。**

H20 28 块真栈实测的「时间×显存」前沿（截距均落在 22.6GiB=权重，自洽）：

| `grad_checkpoint_policy` | ms/token | 激活 MB/token |
|---|---|---|
| `full`（默认，现状） | 0.674 | 0.72 |
| `sac_attn` | 0.653 | 1.03 |
| `sac_narrow` | **0.562** | 2.43 |
| `sac_all` | **0.484** | 4.07 |
| （对照）完全不 checkpoint | 0.465 | 7.52 |

- `sac_*` 保留贵的 matmul / SDPA 输出，只重算便宜的 norm/silu/rope/逐元素；
  `sac_narrow` 额外放过宽于 `features` 的中间量（Krea2 是 SwiGLU 的 16384 维，显存大头）。
- **数学恒等**——只改"哪些中间量保存 vs 重算"，有等价性单测
  （`tests/test_ckpt_policy_equivalence.py`，前向/输入梯度/参数梯度三样都对）。
- 完全不 checkpoint 只比 `sac_all` 快 4%，却要 1.85× 显存 —— 没有采用价值。
- **`grad_checkpoint_skip_last=8/12` 在这张前沿上被 `sac_narrow` 完全支配**（更慢且更占
  显存），不建议再用；两者同开会 fail-fast。

**怎么用（关键：预算和策略要一起改）：**

```yaml
grad_checkpoint: true
grad_checkpoint_policy: sac_narrow    # 稳妥档；显存宽裕再上 sac_all
navit_token_budget: 12288             # 降到 1~2 张图的 token 数，把显存让给上面
grad_accum: 8                         # 用它把有效 batch 补回来
```

预算怎么定：`可用显存 - 权重 - (LoRA梯度/优化器/TE/VAE/碎片余量)` ÷ 上表的 MB/token。
**首跑务必盯峰值显存**——上表的 MB/token 来自不含 LoRA/DoRA 的合成栈，真实训练更高。

**副作用（要一起看）：** 每包图数变少 → 逐图 loss 的等权平均（`per_image.mean()`）在
包之间的权重差异被放大；G=1 时反而彻底消失（每图权重恒为 1.0）。步数变多由 `grad_accum`
补回有效 batch，optimizer 只占步时 0.19%，多出的步开销可忽略。

### 2.3 缓存分块 encode（`cache_encode_tiled`，opt-in）

```yaml
cache_encode_tiled: true        # 默认 false；只影响缓存阶段，与训练步显存无关
cache_encode_tile_px: 1024      # 块边长（16 的整倍数）；峰值显存 ∝ 块像素数
cache_encode_tile_overlap: 128  # 相邻块重叠（16 的整倍数、≤ tile 一半）
cache_encode_max_pixels: 0      # 单次 encode 像素预算（含 flip 份）；0=内置 4M（保守）。
                                #   大显存卡上调（如 80GB → 16777216=16M）可让同尺寸小图
                                #   批量更深（纯提速）；同时是 tiled 的分块触发阈值——
                                #   超预算的图才分块，预算内整图 encode。
```

**解决什么问题：** 缓存阶段的像素预算（默认 4M px，可用 `cache_encode_max_pixels` 覆盖）
只能"少装几张"，对单张超限图（如 2814×4456 = 12.5M px，flip 再 ×2）无约束力——整张过
VAE encoder 的全分辨率卷积激活曾实测把 80GB 卡顶满。

**怎么做：** 超过 4M px 的图切成带重叠的像素块（末块贴齐边界保满块），逐块 encode 后在
latent 网格上按线性羽化权重累加归一化拼回**完整原生分辨率 latent**——原生大图训练不受
影响，峰值显存封顶在 ~tile²（1024² ≈ 一张普通图）。预算内的图走原路径，逐字节等价。

**注意：** 接缝处是**近似**——VAE conv 感受野越过块边界的信息在分块下缺失，overlap 越大
误差越小（默认 128px；diffusers/ComfyUI 的 VAE tiling 同理）。对窗口对齐的局部算子拼接
≡ 整图（单测固化几何正确性）；真 VAE 的接缝误差量级**待云端首跑目检**（建议对一张大图
对比 tiled 与整图 encode 的 latent 差异后再批量用）。

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

### 2.x 注意力精度：`attn_force_autocast_dtype`（opt-in, default-off）

注入 LoRA/LoKr/DoRA 后，autocast(bf16) 下的 dtype 链条是：`nn.LayerNorm` 属 autocast 的
fp32 策略 → `normalized_x` 是 fp32 → 被包住的 Linear 把输出 cast 回**输入** dtype
（`trainer/lora.py` 的 `y.to(dtype=x.dtype)`）→ `q/k/v_proj` 全吐 fp32。后果两条：

1. **cross-attn 崩**：k/v 来自 bf16 的 `crossattn_emb` → q=fp32、k=v=bf16，而
   `xops.memory_efficient_attention` 不是 autocast 算子，`validate_inputs` 直接
   `ValueError`。dense 路径走 SDPA（autocast 算子，自己会统一）所以从没暴露。
   → 这条**无条件修复**（`_unify_attn_dtype`，dtype 不一致时归一），与开关无关。
2. **self-attn 悄悄跑 fp32**：三者同为 fp32，xformers 不报错，但整条自注意力是 fp32
   kernel；而 dense/eval/采样一直是 bf16。开 `attn_force_autocast_dtype: true` 后
   autocast 期间一律按 autocast dtype 计算，两条路口径一致，并省下 navit 注意力的
   时间/显存（本地 SDPA 代理测量 S=4096：fp32 比 bf16 慢 3.2×、峰值显存 1.84×；
   xformers 真实核未本地验证）。默认 false = 保持现状（fp32），改动逐 bit 中立。

只对 Anima family 生效；krea2 用 `models/krea2_modeling.py` 自己的 attention，配在
krea2 上会 fail-fast。

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

### 4.1 per-image AdaLN 调制（速度优化，行为等价、无开关）

**动机与实测（本地 RTX 5070 Laptop，真实维度 2048ch×28blocks×16 heads、pack=4×4096 token，
bf16+xformers）：** 旧实现把 t_emb/adaln_lora `repeat_interleave` 成逐 token `[1, ΣN, *]`
再喂进每个 block 的三个调制 MLP —— 即 ① 调制 matmul 在 ΣN=16384 行上跑（唯一值只有 G=4 个），
② `chunk(3)` 出的 9 组 `[1, ΣN, D]` **非连续条带视图**直接喂 AdaLN 逐元素 op。
交错测量（6 轮轮换顺序取中位，抵消笔记本时钟漂移；早期顺序测量曾给出 +85% 的夸大值，已弃用）：
新路径较旧路径**前向 −13.3%**。同基准下注意力形状（dense `[4,4096]` 无掩码 vs 块对角
`[1,16384]`）前向等速（±1.4%）、fwd+bwd 仅 +6% —— attention 不是瓶颈。
**诚实标注：** −13% 前向不足以解释云端全部差距（navit budget=16384 ≈ 0.25 it/s vs
ARB bs=4 ≈ 0.33 it/s，步时 +32%）；剩余部分的归因需要云端 `stage_timing_every` A/B 数据
（本地卡与云端卡的带宽/算力比不同，各开销占比会移动）。

**改法：** `forward_packed_navit` 改传 per-image `[1, G, *]` 的 emb/adaln_lora + `mod_index`
（`[ΣN]` token→图行映射）；`Block/FinalLayer.forward_tokens` 在 G 行上算调制 MLP，各 chunk 经
`index_select` gather 成**连续**逐 token 张量再应用。同一行同值 → 数学等价（fp32 单测
`test_navit_per_image_adaln` 固化 Block/FinalLayer 两布局一致，含 use_adaln_lora 两分支；
model 级等价仍由 `test_packed_navit_forward` 覆盖）。legacy 逐 token 布局（`mod_index=None`）
保留，ARB/token-bucket 路径逐字节不变。

**尚未落地的已知开销：gather 是物化的。** 当前 `Block.forward_tokens` /
`SingleStreamBlock.forward` 拿到 per-image 调制行后，用 `index_select(...).unsqueeze(0)`
**把 6 个 `(1, ΣN, D)` 张量全部物化**再喂逐元素运算。这些张量每行都是同一图的常数，
信息量只有 `G×D`，却按 `ΣN×D` 落了盘——Krea2 `D=6144`、`ΣN=55778` 时是
**6×55778×6144×2B ≈ 4.1 GB**（另加 RMSNorm 内部 fp32 中间量约 1.4 GB）。

本地已量过天花板（`tests/diag_navit_adaln_gather.py`，RTX 5070 Laptop / torch 2.9.1 /
真实宽度 D=6144 / G=4，只测被改动的算子链、不含 attention/SwiGLU/LoRA）：

| 变体 | fwd (ΣN=8192) | fwd+bwd | 峰值显存 |
|---|---|---|---|
| 现状（6× index_select 物化） | 19.5 ms | 59.5 ms | 2449 MiB |
| 表达式内就地 gather | 19.5 ms（1.00×） | 66.0 ms（0.90×） | 2449 MiB |
| `torch.compile(dynamic=True)` 融合 RMSNorm+gather+仿射 | **4.5 ms（4.35×）** | **32.6 ms（1.79×）** | **1585 MiB（−35%）** |

数值上不是"更准/更错"而是**舍入路径不同**：以全 fp32 为参考，现状与融合版的相对误差都是
5.4e-03，同为 bf16 eps（3.9e-03）量级（脚本已把这个判据固化，不再拿现状当参考对拍）。
`dynamic=True` 下三个不同 ΣN 只产生 2 张图，**没有逐形状重编译**（`unique_graphs=2`）。

同一位置正是 AdaptiveLoad（arXiv 2605.17923）写 fused LayerNorm-Modulate CUDA kernel 的
地方（其报该算子 fwd 3.21–3.39×、激活 −61.9%，与本地这条 pure-PyTorch 路线量级一致）。

**诚实标注：这是算子链自身的倍数，不是整步收益。** 该链在整步里占多少、云端卡（H20 /
sm120）上带宽比不同会把倍数移到哪，都需要云端 `stage_timing` 才能定。文档 §4.1 记录的
云端 +32% 步时差里仍有未归因的部分，这是其中一个**合理嫌疑人，但尚未证实**。
另需注意：全局 `torch_compile` 在 navit 下是 fail-fast 的（整模型动态形状），
这里用的是**叶子级纯逐元素函数的局部编译**，是另一回事——真要落地仍须先确认它与
`grad_checkpoint` / LoRA 注入的实跑交互。

顺手的小优化（同 commit）：`_packed_rope_from_grid` 两次 `.item()` 同步合并为一次 +
freqs 按 (device, NTK) 缓存；`BlockDiagonalMask` 按 seqlens 元组 lru 缓存（纯 CPU 元数据，
块内本就跨 28 block 复用）；补上 config 注释引用的 `tools/analyze_stage_timing.py`。

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
