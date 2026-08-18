# 海光 DCU（K100-AI）适配 + 单机 8 卡数据并行

> **本文档的证据等级**：分三档标注。
> **【实测】** = 在本仓库的机器上跑过的数字。
> **【资料】** = 来自公开资料/厂商口径，未验证。
> **【推断】** = 从架构代际或其它平台的经验外推，**最容易错**，都留了实测推翻的口子。
>
> 截至撰写时，**DCU 侧没有任何【实测】** —— 本适配是"照着已知约束写好、把不确定的
> 全部收敛到一个探针里"的状态。上机第一件事是跑探针，然后回来把本文档的【资料】/
> 【推断】替换成【实测】。

---

## 0. 30 秒版

```bash
# 1) 先体检 + 探针（2 分钟，能省一次白烧的卡时）
bash run_dcu.sh --probe

# 2) 单卡首跑（20 步，证明前向/反向/存盘都通）
bash run_dcu.sh

# 3) 8 卡通信实测 → 再决定 8 卡值不值
bash run_dcu.sh --nproc 8 --probe

# 4) 8 卡训练
bash run_dcu.sh --nproc 8
```

配置：`config/train_dcu_k100ai.yaml`（单卡）/ `config/train_dcu_k100ai_8card.yaml`（8 卡）。
两者除 7 个字段外逐字段一致，差异在 8 卡版头部列出。

---

## 1. 为什么 DCU 的适配比昇腾**短得多**

| | 昇腾 910B | 海光 DCU |
|---|---|---|
| 软件栈 | CANN + torch_npu | DTK（ROCm/HIP 衍生） |
| 设备 API | `torch.npu.*`，靠 `transfer_to_npu` 补丁映射 | **就是** `torch.cuda.*`（HIP 复用 CUDA 命名空间） |
| autocast device_type | `"npu"` | `"cuda"` |
| 集合通信 | HCCL | RCCL，**注册在 `nccl` 这个名字下** |
| 兼容层要做的事 | 设备重映射 + 算子绕行 + 守卫 | 只有确认 + 守卫 + 环境 |

所以 `utils/dcu_compat.py` 比 `utils/npu_compat.py` 短一半，而且 `device_str()`
返回的是 `"cuda"` —— **这不是没适配，是 DTK 的既定约定**，写 `"hip"` 反而报错。

判别当前 torch 是不是真的 DCU 版：`torch.version.hip` 非空且 `torch.version.cuda` 为空。

---

## 2. 平台事实

### 硬件【资料】
K100-AI：gfx928、64GB 显存、~896 GB/s 带宽、BF16/FP16 峰值 ~192 TFLOPS、350–400W。
显存与昇腾 910B 同为 64GB，所以 NaViT 的 token 预算可以直接沿用昇腾版的量级起步。

**卡间互联拓扑没有可信的公开数据** —— 查到的国产 8 卡互联带宽说法互相矛盾，
一律不采信。这个数字对 8 卡效率是决定性的，所以做成了探针里的实测项
（`run_dcu.sh --nproc 8 --probe` 会打 algbw/busbw）。

### 软件栈【资料】
* K100-AI 需要 **DTK ≥ 24.04**。DTK 25.04 配套 torch 2.4.1+das.opt2.dtk2504 / Python 3.10。
* 镜像来自光源 sourcefind：`image.sourcefind.com:5000/dcu/ecosystem/pytorch:<torch>-dtk<ver>-py<ver>-<os>`。
* 看卡：`rocm-smi`（DTK 侧）、`hy-smi`（驱动侧）、`rocminfo | grep gfx`。
* DTK 根目录 `/opt/dtk`；`LD_LIBRARY_PATH` 缺 `/opt/dtk/lib` 会表现为找不到 `libhip*.so`。
* `HSA_OVERRIDE_GFX_VERSION=9.2.8` 对应 K100-AI（仅当 torch 不是为 gfx928 编的时才需要）。

### 两条红线【资料】
1. **绝不 `pip install torch` / `pip install torchvision`**。PyPI 版会把镜像里适配好的
   das 构建覆盖成 CUDA 构建，之后所有报错都指向奇怪的地方。`run_dcu.sh` 的体检把
   "torch 必须是 HIP 构建"设成了**硬门槛**，不过直接拒绝启动。
2. **换 DTK 版本 / 换卡型号后先清 MIOpen 缓存**（`~/.cache/miopen`）。
   陈旧的内核缓存会让算子**结果出错**（不是变慢）。`dcu_compat.miopen_cache_hint()`
   会在启动日志里打出这条提示，但不会替你删（删除是难以撤销的操作）。

---

## 3. 必须实测、不能假设的四件事

`tools/dcu_probe.py` 就是为这四件事写的。它们的共同点是：**错了不会报错**，
只会让训练照跑、loss 照降，然后你拿到一个错的模型或者一张炸掉的卡。

### 3.1 ★ SDPA 落到哪个后端 —— 决定 NaViT 能不能用
ROCm 上 flash / mem-efficient 后端依赖 aotriton 被编进 torch，DTK 里有没有是未知数。

**只剩 math 后端的话**，注意力会物化 O(S²) 的分数矩阵。`sdpa_seg` 逐段跑 dense SDPA，
段长 = 单张图的 token 数；ladder 4096 时每段 4096²×2B×16头 ≈ 0.5 GB —— NaViT 直接不可用。

探针的两项：`SDPA 可用后端`（逐后端强制跑一次看谁能起）和
`sdpa_seg 显存线性性`（ΣN 8k vs 32k 的 peak 比值；线性 ≈4×，落 math 会远超）。

### 3.2 ★ 广播 matmul 的**反向**
昇腾上实测过 2D×3D 广播 matmul **前向逐位正确、反向静默算错**，害得 LoKr 的 w1
梯度整个是错的（见 memory `npu-broadcast-matmul-grad-bug`）。**前向对不代表反向对。**
探针用 fp64 CPU 参考对拍 dA/dB 两个梯度。这条过不了，LoKr 必须走 expand+bmm 绕行。

### 3.3 ★ key-padding mask 的极性
`True=保留`（torch 语义）还是 `True=屏蔽`。反了的话文本条件变成"只看 padding"，
训练毫无异常。全 True / 全 0 的 mask 测不出极性，探针用**真实 padding** 对拍。

### 3.4 8 卡 all-reduce 的实际带宽
见 §5。

---

## 4. 已知不支持项（`dcu_compat.guard_unsupported`）

守卫的设计原则是**能 import 验证的就验证，不写死"DCU 上没有"** —— 海光的光源仓库在
持续补生态，写死会让本来能用的功能被永久拦住。

| 开关 | 处理 | 依据 |
|---|---|---|
| `base_quant`（FP8/FP4） | 拦，可用 `ANIMA_DCU_ALLOW_FP8=1` 放行 | 【推断】FP8 张量核在 AMD 谱系里自 gfx942 起；gfx928 在其之前。探针有实测项 |
| `navit_attn_backend: xformers` | 先 `import xformers.ops`，失败才拦 | 【资料】xformers 无 DCU/HIP 轮子 |
| `navit_attn_backend: npu_tnd` | 直接拦 | 昇腾专有算子，与平台无关的事实 |
| `torch_compile` | 先 `import triton`，失败才拦 | DTK 是否带 triton 随版本变化 |
| 8-bit 优化器 | 先 `import bitsandbytes`，失败才拦 | 【资料】无 DCU 后端 |
| `aux_perceptual_enabled` | 先 `import torchvision`，失败才拦 | 与昇腾不同：DCU 侧有 das 版 torchvision，**不预先禁止** |

---

## 5. 单机 8 卡数据并行

### 5.1 为什么不是原生 `DistributedDataParallel`

本仓库训的是**冻结底模 + 注入的 LoRA/LoKr**，可训练参数只有几十 MB。
原生 DDP 的核心优势（all-reduce 与 backward overlap）要以"把 model 包一层"为代价，
而这一层在本训练器里很贵：

1. `anima_train.py` 有十几处直接调 `model.xxx` / 自定义前向（NaViT 打包、DPO、GAF、
   CSFlow、NCP、LoRA-One、telemetry 探针），包一层后全要改写成 `model.module.xxx`。
2. 更要命的是 **module_dropout / T-LoRA / LoKr 的逐步条件分支** 让"这一步哪些参数参与
   计算"每步都不同 → 原生 DDP 必须开 `find_unused_parameters=True`，且与 grad
   checkpoint 组合在若干 torch 版本上直接报错。

所以走**手动梯度 all-reduce**：只在累积边界、只对可训练参数做一次集合通信。
代价是没有通信/计算 overlap；收益是**模型侧零改动**，所有进阶功能原样可用。

实现在 `utils/dist_utils.py`，训练器侧只有 5 处接线。

### 5.2 数据怎么切

`ShardedBatchSampler` 按 **stride 切已经组好的 batch**：rank r 取全局第
`r, r+W, r+2W, ...` 个 batch。

之所以不"按样本切数据集"，是因为三个 sampler（`BucketBatchSampler` /
`FitTokenBatchSampler` / `NavitPackBatchSampler`）的分批逻辑本身携带约束
（同桶同分辨率、token 预算装包），从中间切样本会破坏它们；切 batch 序列则完全不碰。

**每 epoch 丢掉不足 W 个的尾部 batch。** 各 rank 迭代次数必须严格相等，否则先跑完的
rank 退出循环、其余永远等在集合通信上。每 epoch 重洗牌，所以丢的不是固定那几张图。
数据量小的时候这个比例值得算：实际训到的是 `全局batch数 // W * W`。

### 5.3 三个必须知道的语义变化

**① 学习率没有被自动缩放。** 8 卡 = 单步看到 8 倍数据 ≈ 等价 batch ×8。
本仓库不替你改 lr（那是未经请求的行为改变），只在启动日志里把有效 batch 打出来。
要不要按 √W 还是 W 放大是你的决定，建议先原样跑、对比单卡同 step 的 loss 曲线。

**② 跨 rank 是"无权平均"，不是"全局样本均值"。**
各 rank 的 micro-batch 样本数不同时（ARB 桶尾、NaViT 每包图数不等），
"各 rank 均值再取均值" ≠ "全局样本均值"，小 batch 的样本被加权更高。
**这与 PyTorch 原生 DDP 的语义完全一致**（原生 DDP 也是无权平均），不是本实现引入的
偏差 —— 但必须知道它存在。

**③ 进程本地的状态会分叉。** 构造期会警告：
* `gaf_enabled`：逐图信任是进程本地的，每 epoch 重洗牌后同一张图落到不同 rank →
  信任历史被打散，`trust_decay` 的累积效果稀释到 1/W 量级。不会崩，但多卡的 GAF
  与单卡不是同一个东西。
* `adaptive_timestep`：控制器状态也是本地的，各 rank 独立演化出不同的 t 采样分布。
  等价于 W 个控制器投票，不等价于单卡的一个控制器。

**日志里的 loss 是 rank0 的本地值**（全局 batch 的 1/W 子样本），无偏但更抖。

### 5.4 构造期直接拒绝的组合

| 配置 | 为什么 |
|---|---|
| `effective_batch_size > 0` | 累积边界由**每 batch 的样本数**决定，各 rank 分到的图数不同 → 在不同 `batch_idx` 触发边界 → 集合通信错位、**死锁**。多卡请用 `grad_accum`（边界只由 batch_idx 决定，跨 rank 恒等） |
| `dpo_enabled` | 输家池按 global_step 重生成、改的是各 rank 本地的池，无同步机制 → 池分叉，训的不再是同一个目标 |
| `lora_one_init_steps > 0` | 预热阶段自己迭代 dataloader 做全参 backward，那段没接梯度同步 → 各 rank 算出不同的初始化因子。可先单卡跑完预热存下 LoRA，再多卡 `resume_lora` |
| `device_backend: npu` + 多卡 | 昇腾要走 HCCL，本仓库尚未接线 |

### 5.5 死锁面 —— 本适配真正的难点

数据并行的坑不是算错，是**挂住**。任何"某个 rank 走了不同分支、少调了一次集合通信"
都会让整个 job 卡在通信上直到超时（而且不报错，就是不动了）。

训练器里有**两条数据相关的 per-rank 提前退出路径**，各自都能造成死锁：

1. **micro-batch loss 非有限** → `continue` 跳过 backward。若这个 micro-batch 恰是
   累积边界，其余 rank 正停在边界的集合通信上等它 → 全体挂死。
   处理：该路径补一次与正常边界**严格一一对应**的 `all_ranks_clean(False)`。
2. **整周期梯度非有限** → `continue` 跳过 step。
   处理：把 `accum_clean` 先取跨 rank 逻辑与（`all_ranks_clean`），让"要不要跳过"
   成为全体一致的决定。

另外，**梯度 all-reduce 刻意放在 NaN 检查之前**：all-reduce 会把任一 rank 的 NaN
传播到所有 rank，那道逐参数 `isfinite` 于是自动成为全体一致的判定，
不必再加一次集合通信（少一次通信 = 少一个可能错位的点）。

### 5.6 起点必须一致（一个差点踩进去的坑）

各 rank 的 LoRA 初始化**必须完全相同**。梯度同步只同步**增量**，不同步绝对值 ——
起点不同的话，同一份平均梯度作用在不同权重上，权重差永远不会收敛回来。
表现是完全静默的：loss 照降，只是 8 张卡在训 8 个不同的 LoRA，最后存下来的是其中一个。

所以：
* 全局种子在**模型构造之前**保持各 rank 一致（不加 rank 偏移）；
* 模型建好后从 rank0 **广播一次**可训练权重（PiSSA/OLoRA 这类要对底模做 SVD 的初始化，
  即使种子相同也可能因数值库的非确定性分叉，一次广播杜绝整类问题）；
* **然后**才给全局 RNG 加 rank 偏移，让训练期的噪声/timestep/caption-dropout 跨 rank 独立。

三个 batch sampler 的洗牌用的是它们自己的 `random.Random(seed + epoch)`，不读全局 RNG
→ 不受偏移影响，各 rank 的**全局 batch 序列仍然相同**，分片才有意义。

### 5.7 rank0 独占的副作用

写 checkpoint / 出采样图 / 起监控端口 / 推 wandb / 渲染进度条 —— 全部只在 rank0 做。
8 个进程同时写同一个 `.safetensors` 会写出**损坏文件且不报错**（后写的截断前写的）；
8 个进程抢 6006 端口会有 7 个失败。

非主 rank 不需要显式 barrier 等 rank0 出图 —— 它们会在下一个累积边界的集合通信处
自然等待。**也不能加** barrier：barrier 与 all_reduce 的调用次数必须跨 rank 完全对齐。

`dist_timeout_minutes`（默认 60）就是为这段等待留的：主 rank 独占跑一遍全量 eval
可能几分钟到几十分钟，NCCL 的默认超时会误杀。

### 5.8 断点续训

`dataloader_fingerprint` 新增了 `world_size` 字段：8 卡存的 state 拿到 4 卡上 resume 时，
每个 rank 的 batch 序列完全不同，"跳过前 N 个 batch"会跳到别的数据上 —— 而 sampler
类名和 dataset_len 都一样、原来的指纹察觉不到。加了这个字段后卡数变化会被指纹挡住。

同卡数 resume 是精确的：`ckpt_batch_in_epoch` 记的是 rank0 的**局部** batch 序号，
各 rank 局部序列等长且边界对齐，所以每个 rank 跳过同样多的局部 batch 即可复现位置。

### 5.9 怎么判断 8 卡值不值

`stage_timing_every > 0` 时 `stage_timing.csv` 多了一列 **`grad_sync_ms`**
（单卡恒为空）。判据：

* `grad_sync_ms / whole_step_ms` **< 5%** → 8 卡接近线性，继续。
* **> 20%** → 通信吃掉太多。正确的应对是**增大 `grad_accum`**（同步频率被摊薄
  grad_accum 倍），而不是继续加卡。

配合 `run_dcu.sh --nproc 8 --probe` 打出的 all-reduce 毫秒数（按 8/32/128 MB 三档扫，
覆盖 LoRA 梯度的真实量级），可以在真正开训之前就估出这个比例。

---

## 6. 首跑清单

1. `bash run_dcu.sh --probe` → 看末尾「配置裁决」，照它改 yaml
2. 广播 matmul 反向那项若 FAIL → **停下来**，LoKr 路径要先修，别开训
3. `sdpa_seg 显存线性性` 比值 >7× → 把 `navit_token_budget` 按实测 peak 压下去，
   或直接 `navit_packing: false`
4. `bash run_dcu.sh` 单卡 20 步 → 确认 checkpoint 真的写出来了、能下载
5. `max_steps: 20 → 0`，`eval_every: 0 → 40`，逐项放开
6. `bash run_dcu.sh --nproc 8 --probe` → 记下 all-reduce 毫秒
7. `bash run_dcu.sh --nproc 8` → 看 `grad_sync_ms/whole_step_ms`，决定 grad_accum

**跑完请把探针的 JSON 回传，本文档里的【资料】/【推断】应该被替换成【实测】。**
