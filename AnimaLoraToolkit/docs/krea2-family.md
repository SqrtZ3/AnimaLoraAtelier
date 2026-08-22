# Krea 2 (K2) 模型族训练指南

本仓库通过 **model family** 通用接口（`trainer/model_family.py`）支持在 Krea 2 上训练
LoRA/LoKr。`model_family: anima`（默认）时一切行为与历史逐一等价；本文档只讲
`model_family: krea2`。

## 架构速览（与 Anima 的差异）

| | Anima | Krea 2 |
|---|---|---|
| 骨干 | Cosmos MiniTrainDIT（cross-attn 条件） | 单流 MMDiT 12B（text+image 同序列） |
| 文本 | Qwen3-0.6B hidden states + T5 token 权重 | Qwen3-VL-4B **12 层** hidden states 堆叠，DiT 内 TextFusion 融合 |
| VAE | Qwen-Image VAE (Wan 系 f8/16ch) | **同一个**（latent 缓存/归一化直接复用） |
| 范式 | Flow Matching | Flow Matching（同参数化 v=noise−x0，t=1 纯噪） |
| timestep 调度 | flow_shift/schedule_shift | 分辨率感知 shift α=exp(mu(seq_len))（`krea2_res_shift`） |
| 官方工作流 | — | **RAW 上训、Turbo 上推理**（LoRA 直接可用于 Turbo） |

## 最小配置

```yaml
model_family: krea2
transformer_path: /path/to/raw.safetensors          # Krea-2-Raw（训练一律用 RAW）
vae_path: /path/to/qwen_image_vae.safetensors       # 与 anima 同一文件
krea2_text_encoder_path: /path/to/Qwen3-VL-4B-Instruct   # HF 目录
cache_latents: true

# navit 打包照常可用（推荐）：
navit_packing: true
navit_native_resolution: true
navit_token_budget: <按显存定>
navit_attn_backend: xformers   # xformers（默认，历史行为）| sdpa_seg
```

- **`navit_attn_backend: sdpa_seg`**（opt-in）：packed 注意力从 xformers
  BlockDiagonalMask varlen 换成**逐段 dense SDPA（cudnn）**。段内全注意力 ≡
  块对角语义，数学恒等（`tests/test_sdpa_seg_attention.py` 前向+梯度对拍）；
  H20 微基准 dense SDPA 比 xformers FA2 快 1.56×（云端 xformers 的 5D grouped
  路径无 backward 算子，训练实际一直走 4D 物化）。G=2~10 时逐段 launch 开销
  可忽略。该接缝同时是未来低比特 attention 后端（SageBwd INT8 等，2026-07 时
  上游 kernel 尚未开源）的插槽。

`text_encoder_path` / `t5_tokenizer_path` 在 krea2 下不使用。

## 关键行为

- **LoRA targets 默认 = DiT 全部 264 个 Linear**（官方/musubi 推荐口径，rank/alpha 32
  为作者默认；含 tproj.1。modulation/RMSNorm 是裸张量、非 Linear，本就不被包）。
  tproj.1 是全网最大单层，训练侧无问题（预览正常即证），部署时的显存坑见下方
  「ComfyUI 推理部署」。想复刻官方"长训练"的 attention-only（140 Linear）：
  ```yaml
  lora_targets: ["attn.wq","attn.wk","attn.wv","attn.wo","attn.gate"]
  lora_exclude_patterns: [".*txtfusion.*"]
  ```
- **timestep**：`krea2_res_shift: true`（默认）在任何 timestep mode 采样之后按
  每图 token 数施加官方分辨率感知 shift（navit 逐图、dense 全批同值）。开着它时
  `schedule_shift` 建议保持 1.0，避免双重偏移。mu 端点 256px→0.5 / 1280px→1.15
  可用 `krea2_shift_*` 微调。
- **文本**：`(tag:1.5)` 权重语法被剥离（无 T5 权重通道）。caption 静态时启用
  LRU cache（`krea2_text_cache_entries`，每条数 MB，按显存调）。navit 打包只带
  有效 token（数学上与官方 mask 等价——text token 无 RoPE，pad 作为 key 被 mask
  后贡献恒 0）。
- **采样预览**：走既有 `sample_latent`/ER-SDE；sigma 调度自动用 exp(mu(分辨率))。
  CFG 换算：官方 `--cfg g` 是 `cond + g·(cond−uncond)`，等价本仓库
  `cfg_scale = 1 + g`（官方 RAW 推荐 3.5 → 这里 **4.5**）。默认负面提示词为空
  （对齐官方；danbooru 质量 tag 对 krea2 语义未知）。RAW 是 CFG 模型，预览步数
  建议 ≥28。
- **不支持项（启动时 fail-fast）**：fit_packed_training/token_bucket/torch_compile、
  TREAD、DPO、LeapAlign、GAF、NCP、self-perceptual、dispersive、LoRA-One init——
  这些绑定 Anima 前向内部结构，需要时单独移植。ΔFM/VeCoR、Eisbach、LWD、
  min-SNR/EDM2 加权、三峰/Laplace/CSFlow 等 latent 空间/t 轴技术照常可用。

## ComfyUI 推理部署（tproj.1 的显存坑 + 解法）

**现象**：把训好的 LoRA 喂进 ComfyUI，日志刷 `ERROR lora diffusion_model.tproj.1.weight
Allocation on device`，出图人物风格在、**背景崩成乱图**。

**根因（已定位到 ComfyUI 源码，非训练/数据问题）**：ComfyUI 标准 *Load LoRA* 节点对每个
权重物化**完整 delta 矩阵**再并入底模（`comfy/weight_adapter/lora.py`：
`lora_diff = torch.mm(up, down).reshape(weight.shape)` → `weight += ...`）。tproj.1 =
`nn.Linear(6144, 36864)`，delta 是 `[36864,6144]`≈453MB(bf16)/905MB(fp32)；12B 底模占满
显存时这步 OOM，被 `except` 静默吞掉、该层回退底模。于是只有 tproj.1 用底模、其余 263 层是
训练后的 → 全局 timestep 调制（tproj 输出喂全部 28 个 block 共享）错位 → 背景崩。
（训练时的采样预览用活模型全 264 层、tproj.1 是训练后的，所以预览正常——问题只在 ComfyUI
这条实时加载路径。这也是 ComfyUI 已知问题，见 Comfy-Org/ComfyUI#12000 一类 fp8/大层 OOM。）

**解法：改用 ComfyUI 内置节点 `Load LoRA (Bypass, Model Only)`**（类
`LoraLoaderBypassModelOnly`，分类 `model/loaders`，节点菜单搜 "Bypass"；实现见
`comfy_extras/nodes_lora_debug.py` → `comfy.sd.load_bypass_lora_for_models`）。它把 LoRA
作为**前向低秩注入**（`out = base_forward(x) + strength·(x·downᵀ)·upᵀ·(α/r)`），**永不物化
完整 delta**，故 tproj.1 不再 OOM，全部 264 层（含 tproj.1）都正确生效。与"完整合并"数学
等价（本地用真实 tproj.1 因子实测：fp32 max_abs_diff 1.7e-6，bf16 1.6e-2 属正常舍入）。
接线：把原来的 *Load LoRA (Model Only)* 换成 *Load LoRA (Bypass, Model Only)* 即可，其余不变。

- 代价：每步每层多一次低秩 matmul（rank 32，相对底模 6144 维开销很小），略慢，画质不变。
- 该节点标 `EXPERIMENTAL`/"(for debugging)"，但是 ComfyUI 官方代码、机制正确。
- 若你的**推理** ComfyUI 是缺 `nodes_lora_debug.py` 的旧构建，把该文件作为 custom node
  放进 `custom_nodes/` 即可获得同名节点。
- 不要用"从 LoRA 里删掉 tproj.1 键"来绕：那等价于 tproj.1 用底模，背景照崩（已实测），
  且丢了官方推荐训练的一层。

## 显存与速度预期（未实测，推断）

12B bf16 冻结权重约 24GB；navit 显存系数与 Anima 的 `10GB+0.52MB/token` **不可比**
（模型大 6 倍），token_budget 需在云端重新标定，建议从小预算起步。RAW 训练分布
只到 1k：navit 多尺度大图超过 ~4096 token（1024²）属于底模分布外，谨慎。

## TPU 后端（jax_tpu，`model_family: krea2`）

Krea2 已接线到 Kaggle TPU v5e-8 训练链路（`run_train.py` 自动按 `model_family`
分派）。与 Anima TPU 路线共用同一份 yaml / 同一条缓存→打包→训练管线，差别：

| | Anima TPU | Krea2 TPU |
|---|---|---|
| 权重 | bf16 3.91GB，**单 chip 复制**，纯 DP | bf16 ~24GB > 15.7GiB，**FSDP 8 卡分片**（每卡 ~3GB），每层 remat 内 all_gather（真机 spmd_probe P1 定案：9.8k tok/s @ 16384/卡，单卡 3.05GiB） |
| 序列 | 图像段 + cross-attn 定长文本槽 | **单流**：每图段 = [text ; image]，预算两边一起花；text 槽量化到 128 |
| 文本缓存 | `cross` [512, 1024] | `txt` [L, 12, 2560]（12 层堆叠、**变长**），`tools/cache_text_features.py --model-family krea2` 产出（Qwen3-VL-4B） |
| timestep | flow_shift/schedule_shift | `krea2_res_shift: true`（默认）：采样后按每图 image token 数施官方 mu shift；开着时 `schedule_shift` 必须 1.0（构造期 fail-fast，防双重偏移）。shift 之后再过一次 `t_range` 截断（与 PyTorch 侧 `anima_train.py` 的顺序一致） |
| LoRA 默认 targets | 10 个投影层 | 官方推荐**全部 264 个 Linear**（28 块×8 + txtfusion 33 + tmlp/txtmlp/tproj/first/last）；`lora_targets`/`lora_exclude_patterns` 语义不变 |
| ΔFM/Eisbach/spectral | 已移植 | **同一份实现直接复用**（它们只作用在图像流 [ΣN, 64] 上，与文本怎么进模型无关） |
| 显存上限 | 单图 ≤ 16384 token/卡 | 单图 **image + text 合计** ≤ 16384/卡（FSDP 实测 32768 OOM，spmd_probe P5） |

- **remat 必须 != none**：FSDP 的 all_gather 要在 checkpoint 边界内，否则 28 层
  全量权重同时活着（spmd_probe 第四跑 OOM 37.62G 的根因；构造期 fail-fast）。
- **底模加载：HTTP Range 流式（2026-08-20 首训定案）**。`raw.safetensors`
  26.3GB 装不下 Kaggle `/kaggle/working`（~20GB），直下不可行；本地出口到 HF
  实测 0.15~0.4MB/s（aria2c 16 线程同），"本地下完再传 dataset"要 ~24h。
  定案：`krea2_jax._read_safetensors_map` 认 http(s) URL——header 两次小 GET，
  之后 8 线程并发预取逐 tensor range GET（在飞字节上限 4GB），读出字节与本地
  文件逐 bit 相同（tests/check_http_range.py 对拍 120 次读取全等，含 302 链）。
  真机实测 24GiB 用 219.6s（~110MB/s）。job 侧用 build_job `--hf-stream`
  （把 ANIMA_TRANSFORMER 写成 resolve URL）+ `--env HF_TOKEN=...`。
  gated repo 的 token：Kaggle Secrets 网页 attach 在 script kernel 上不可靠
  （未 attach 时服务回 HTTP 400，被 kaggle_web_client 的 except 顺序误报成
  "ConnectionError"），env 烘焙是可靠路径（build_job 对 TOKEN/SECRET/KEY
  类 env 的日志打印已脱敏；生成的 anima_train_job.py 被 gitignore，token
  不进 git）。
- **HBM 账目（真 12B 首训实测，与 spmd_probe 的裸模型口径不同）**：
  完整训练图（LoKr 264 层 + ΔFM/Eisbach/spectral + huber-snr）的单个
  executable 驻留 ~11.9G/布局（12288/卡时实测 reserve 11.94G），
  16384/卡时运行期 HLO temporaries 18.15G（均 > 15.75G 上限，两轮 OOM 实录）。
  spmd_probe P1 的"16384/卡 3.05GiB"是裸模型探针口径，别拿来规划真训练。
  **定案工作点**：`navit_token_budget: 81920`（8×10240/卡）+ K2 路径
  **单布局驻留**（run_train.py：换布局时驱逐旧 executable 并 gc，同布局下次
  靠 --jax-cache 磁盘编译缓存秒回；布局数只影响编译次数，不再决定生死）。
  modare 数据集（159 张，最大 4096 img token）上 5 布局 / 填充率 ~81% /
  每 epoch 9 步 / 18 epoch 178 步全程 ~53 分钟（含流式加载 3.7min +
  布局编译）。multiscale 对本数据集是空操作（plan_multiscale_copy 只降不升，
  全数据集 ≤ ladder 档 → 0 sidecar；GPU 侧同函数同行为），yaml 里关掉即可
  （TPU 侧 multiscale=true 且无 sidecar 会 fail-fast）。
- **caption dropout**：换入的空 caption 比扫描时的原 caption 短，
  `assemble_batch_k2` 会按实际长度把 `seg_self`/`txt_fine` 的剩余区间重标
  PAD_SEG（运行时数组，不进编译身份）——否则零值会被 txtmlp 的 bias 穿透成
  假 token 参与注意力（dropout 样本不再是干净的无条件）。
- **裸张量（norm scale / mod.lin / bias）随 `mixed_precision` 存储**：对齐
  PyTorch 训练侧 `model.to(dtype)` 全转的行为（用前升 fp32 的算子语义不变），
  两个后端的 bf16 训练数值因此逐 bit 可对照。
- **自适应采样器的分桶口径**：反馈喂的是 res_shift 后的最终 t，候选分桶是
  res_shift 前口径（采样发生在 t 与图配对之前，架构性错位，与 PyTorch 侧
  同构；方向安全，详见 `jax_tpu/sched.py` 的 `update` docstring）。
- `--unrolled/--packed-chunk/--packed-barrier` 是 Anima 打包路径的吞吐旋钮，
  K2 的 scan+FSDP 路径未验证，构造期 fail-fast。
- 底模走 build_job `--hf-model krea/Krea-2-Raw:<文件名>[:<rev>]`：**gated
  repo，需要在 HF 网页接受 Krea 2 Community License**。raw.safetensors 26.3GB
  > /kaggle/working 的 ~20GB 上限，直下装不下——必须加 `--hf-stream`（HTTP
  Range 流式加载，见上文「底模加载：HTTP Range 流式」），token 用
  `--env HF_TOKEN=...` 烘焙进 job（Secrets 的网页 attach 对 script kernel
  不可靠，同上）。
- 本地闸门：`jax_tpu/tests/` 的 K1（前向对拍，torch dump → jax check，fp32
  rel ≤ 4e-6）与 K2（FSDP 训练闭环，8 伪造 CPU 设备）。改 `krea2_jax.py` /
  `train.py` / `packing.py` / `data.py` / `run_train.py` 后先跑完这两道再上
  真机；动 HTTP Range 加载路径加跑 `check_http_range.py`。
- 训练配置模板：`config/train_krea2_tpu_template.yaml`。

## 验证状态（本地 RTX GPU，tests/test_krea2_modeling.py）

- packed navit ≡ 逐图 dense 前向：fp32 max_abs_diff **1.4e-06**（SDPA 回退）/
  **1.2e-06**（xformers varlen）；跨图隔离逐 bit 成立；per-block checkpoint 逐 bit 一致。
- 与官方 mmdit.py：state dict key 双向 strict 相同；同权重 bf16 前向
  max_abs_diff 1.6e-02（跨 kernel + 官方 256-pad，量级正常）。
- 12B 真权重加载、云端端到端训练、画质效果：**未验证**——首跑请先小步数冒烟。

## TPU 后端一轮全面审查的修复（2026-08-22）

审的是 `jax_tpu/` 全包 + 6 份 TPU yaml + 3 份真机训练日志。**影响已跑过的四轮
（JAN / ASK / modare / ashima）的有三条**，都不报错、都只在特定条件下显形：

1. **`auxloss.eisbach_weight` 会产 NaN 并污染整步**（四轮全部开着
   `eisbach_lambda: 0.15`）。掩码乘在 `exp` 之后，而"无有效 token 的段"的段内最大值
   被兜底成 0，于是填充位能量 > 88（幅度 ≳9.5）时 `exp` 溢出成 `inf`、`inf*0 = NaN`。
   `valid=0` **拦不住** —— `weighted_mean` 用 `jnp.sum`，NaN 直接传染成标量 loss，
   一步污染 fp32 master 全部参数。本地复现：填充位幅度 12 即触发。
   而 FFD 取整余量的纯填充段**每个 pack 都有**（`packing.py:301`）。
   修法：掩码进指数（`exp(where(m>0, e, -inf) - mx)`），与 PyTorch 侧
   `masked_fill(-inf)` 再 softmax 同口径；无填充位时逐 bit 不变。
   历史上没炸过大概是因为填充位能量还没到那个量级 —— 属于**未爆的雷**，不是已发生的错。

2. **krea2 `txtfusion.projector` 的等效学习率是 PyTorch 侧的 32 倍**（K2 三轮都不写
   `lora_targets`，默认 264 层含它）。它是 `Linear(12→1)`，rank 被夹到 1 而 alpha
   仍是 32，于是 `scale = alpha/rank = 32`，其余 263 层都是 1。PyTorch 侧标准 LoRA
   **不夹 rank**（`LoRALayer.__init__`），那边这层恒是 1。
   修法：`adapters.cap_rank_alpha` —— 夹 rank 时 alpha 同比例夹，scale 不变；
   导出侧写的是夹后的 alpha，推理端按 `alpha/rank` 算出的 scale 与训练时逐位相同。
   Anima 侧六个 target 最小维 1024，`scale` 全是 1，**不受影响**。

3. **DoRA 的 `dora_scale` 被降 bf16，step-0 中立性失效**（`ashima` 那轮
   `lora_variant: dora`）。前向 cast 整树 `astype(bf16)`，而 `base_row_sq` 在 consts
   里是 fp32，比值不再精确为 1：实测 step-0 输出 rel **1.659e-03**（改后 0）。
   更要紧的是 bf16 在 ‖W‖ 量级上 ulp 约 0.62%，lr=1e-4 下要 312~1250 步才在前向
   可见 —— master 上学到的幅度增量在越过半个 ulp 之前对前向完全不可见。
   `export.py` 与 `trainer/lora.py` 的 state_dict 都坚持把它存 fp32，理由正是这个。
   修法：`train._fwd_cast` / `optim.update` 按叶子路径跳过 `dora`。

其余修的（不影响已跑轮次，但都是"开关开着却什么都没发生"或休眠雷）：
`--packed-barrier` 不带 `--unrolled` 时被静默忽略（scan 分支根本不读它）；
`entropy_rate` 的 `/w(t)` 因为没人传 `loss_weight_fn` 而被静默跳过（采样权重差 4~9 倍）；
`sched.sample()` 候选分桶多套一层 `t_range` clip（`timestep_t_min/t_max` 收窄时
让部分桶的权重永远索引不到）；`sample_steps`/`lr_scheduler`/顶层 `weight_decay`/
`lora_include_patterns` 等 GPU 侧真实生效的键在 TPU 上静默无效；
`navit_max_images_per_pack` 解析了但从未被消费；`krea2_jax._resolve_remat` 对未知
档位静默退化成 `full`；`gather_sharded` 的分片判据在 `ndev != 8` 时错位（休眠）；
`anima_jax.eps_rms` 抄了 `RMSNorm` 的默认 1e-5 而实例化处是 1e-6。

闸门覆盖的盲区分析见 `jax_tpu/tests/README.md` 的「闸门的已知盲区」一节 ——
上面 5 条正确性问题里有 4 条是现有闸门**结构上**抓不到的。

### 效率侧：`LIBTPU_INIT_ARGS` 是可用通道（此前被误判为无法验证）

commit `3c1696c` 认定"抬 `--xla_tpu_scoped_vmem_limit_kib` 这条路走不通，因为
CPU jaxlib 不注册这个 flag、真机是否接受没法本地验"。**前半句对，结论下早了**：
`XLA_FLAGS` 确实会 F 级 abort（本地复现），但 `LIBTPU_INIT_ARGS` 由 libtpu 自己
解析、**在无 TPU 的 CPU 上完全无副作用**（本地实测照常 import jax 并跑通），
而这正是 Google 官方教程的用法（MaxDiffusion on v6e 那篇逐字写着
`LIBTPU_INIT_ARGS="... --xla_tpu_scoped_vmem_limit_kib=65536"`）。

JAN 那次 `CompileTimeScopedVmemOom` 报的 `limit 16.00M` 正是这个 flag 的默认值
（16384 KiB）。当时选的 `ANIMA_BWD_BLOCK_MAX=512` 代价约 **1.1×** attention 反向；
抬额度则数值与调度完全不变。用法与两条注意事项（v5e vmem 上限有官方口径冲突、
首次上机建议 flag 与 `BWD_BLOCK_MAX` 同时带）见 `kaggle_job/README.md` 的
「XLA 调优 flag」一节。**未在 Kaggle 真机验证过。**
