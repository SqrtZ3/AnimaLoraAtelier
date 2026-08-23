# AnimaLoraAtelier

> **Anima 画风 LoRA/LoKr 训练 · 全栈工程**：训练器内核（源自 AnimaLoraToolkit）+ 数十项前沿训练技术落地 + CUDA / 海光 DCU / 昇腾 NPU / TPU 异构适配 + 云端镜像与训练运维。

本仓库围绕 **Anima**（基于 NVIDIA Cosmos 的二次元特调 DiT，Flow Matching 线性调度范式，Qwen3-0.6B 文本编码器 + Qwen-Image VAE）的画风 LoRA/LoKr 训练，把「训练目标 — 优化器 — 打包与分桶 — 显存与量化 — 模型族 — 硬件 — 部署 — 可观测性」整条链路做了系统化工程。同一套训练器也支持 **Krea 2（K2）单流 MMDiT** 模型族。

工程哲学（贯穿全部改动）：

- **opt-in、default-off、行为中立** —— 每个新能力都有 YAML 开关，关掉时与改动前逐字节等价；
- **fail-fast** —— 不支持/未移植的组合直接报错，绝不静默降级；
- **真机裁决** —— 结论尽量来自真机实测（探针/基准/issue log），而不是推测；
- **留痕** —— 每项技术配 `AnimaLoraToolkit/docs/` 文档，讲清机理与依据。

---

## 分支模型：链式推进（建议先读）

本仓库采用**链式分支**推进：每一节新分支从上一节分支的尖端切出，链上任意一节都完整包含其全部前置工作。因此：

- **最新功能永远位于链条末端的分支**（当前为 `feat/tpu-probe`）。
- 各节分支并非并行的独立功能分支：从任何一节切出，都会自动带上之前所有节的工作。
- `main` 按里程碑收编整条链，不追踪每一步。**2026-08-19 已完成一次全链合并**（零冲突），当前克隆 main 即得全量功能；此后的日常更新仍发生在链条末端，main 于下一个里程碑再收编。

| 节（分支） | 收官 | 主要内容（节内主题与分支名大致对应，跨节协作是常态，精确归因见 `git log`） |
|---|---|---|
| `codex-soap-optimizer` | 2026-05-23 | 基础期（**已在 main**）：ARB 长宽比分桶、Kohya 式参数族、`trainer/` 模块化重构、Spectral/Perceptual 辅助 loss、T-LoRA / infonoise、ProdigyPlus |
| `dev/adopt-optimizer` | 2026-05-26 | 优化器实验起步：SOAP / ADOPT / Lion / C-Lion、参考步采样进度 |
| `feat/ntoken-bucketing` | 2026-06-04 | EmoSens、原生 FiT 打包训练路径、Schedule-Free SOAP、DeltaFM / Laplace timestep / scheduled-Huber、N-token 精确分桶 + `compile_blocks` |
| `dev/emosens-optimizer` | 2026-06-29 | 训练目标大扩容：三带课程、U 形双峰采样、LoRA-One(KPSVD)、LWD、TREAD、GAF、DPO 家族、self-perceptual、telemetry 总线、Dispersive |
| `feat/navit-token-packing` | 2026-07-03 | NaViT / Patch-n-Pack 打包训练全链路 + 多尺度 / 原生分辨率 / 按代价装包 / 分块缓存 + stage_timing |
| `feat/krea2-model-family` | 2026-08-02 | Krea 2 模型族与 `model_family` 通用接口、Muon / PiSSA / DoRA、ABBA、AC-LoRA、FP8/FP4 量化、导出 SVD 压缩、heretic 文本编码器 |
| `feat/npu-ascend` | 2026-08-14 | 昇腾 910B 全套适配（兼容层 / 探针 / 镜像装配 / 真机 bug 绕行）+ wandb 接线 |
| `feat/tpu-probe` | 2026-08-19 | Kaggle / TPU 探针与 torch_xla 路线、JAX 全套移植、FSDP 分布式裁决 + 海光 DCU 适配与训练镜像（**当前链条末端**） |

## 仓库结构

```
AnimaLoraToolkit/            训练器内核与主要增量开发
├── anima_train.py           训练入口（train_monitor.py 为 Web 监控面板）
├── trainer/                 目标/loss、LoRA 注入、数据集、分桶、采样、telemetry、checkpoint…
├── utils/                   优化器实现 + 硬件兼容层（dcu_compat / npu_compat / dist_utils）
├── models/                  Anima / Cosmos-Predict2 / Krea 2 建模
├── jax_tpu/                 TPU 侧全套 JAX 移植（16 个模块；**副本**，主拷贝在 kaggle_job/）
├── config/                  训练 YAML（train_all_args_annotated.yaml 为带注释全参考）
├── docs/                    技术文档（每项技术一篇）+ probe_results 真机探针日志
├── tests/                   40+ 回归测试与 diag_* 诊断取证脚本
├── tools/                   探针 / 镜像装配 / 压缩导出 / 取证工具
kaggle_job/                  **可独立发布的 TPU 训练仓库**（jax_tpu 主拷贝 + 数据缓存
                             工具自包含 + Kaggle CLI 全流程；对本仓库零运行时依赖）
tools/                       LoRA/LoKr 分析小工具（块能量、delta 测量、ckpt 平均…）
run.sh / run_dcu.sh / run_npu.sh    各硬件训练入口（含显存守卫接力）
Dockerfile.dcu               海光 DCU 训练镜像（DAS 预编译轮子 + 构建期自检）
gpu_vram_guard.py / gpu_hog.cu      共享 GPU 显存占位 / FP32 FMA 压测内核
docs/superpowers/            设计文档（plans / specs）
```

## 功能总览

### 1. 训练范式与适配器内核

- Flow Matching（线性调度）+ DiT；ARB 长宽比分桶 + **纯 N-token 精确分桶**（`BucketManager`，golden fixture 锁行为）
- 适配器族：**LoRA / LoKr(LyCORIS) / DoRA（含 LoKr+DoRA）/ PiSSA 初始化 / LoRA-One（KPSVD init）/ AC-LoRA（训练期 RESTART + 导出期 SVD 自适应压缩）/ ABBA（Hadamard 双低秩，arXiv:2505.14238）/ T-LoRA ortho init**，LoKr 暴露 `w1` init 尺度与 lr 倍率
- 断点续训全套：Ctrl+C 安全保存、epoch 内位置恢复、优化器状态按保存时 dtype 复原、fp32 master 复原、`save_state_every_epochs`
- 训练中推理出图（预览 prompt 可从训练集 caption 随机抽取）；Web 监控面板 + wandb（均 opt-in）
- 导出：ComfyUI 兼容 safetensors；**SVD 压缩 + 全局最优秩分配**；ABBA native-only 成品（235MB）
- 推荐训练范式：设极大 epoch 持续训练、随时手动停、从任意 step 的 checkpoint 里挑模型（详见 `docs/training-tips.md`）

### 2. 训练目标：timestep / loss / 噪声（`docs/noise-timestep-params.md`）

- timestep 采样器：stratified / logsnr / **U 形双峰**（`mixed_logit_low_high`）/ **三带采样**（结构优先课程）/ 三带混合概率退火
- **自适应 timestep 重采样**（loss-EMA 与 slope 两种度量、低噪门控，含 L3 探针盲区补测）
- loss 族：FM / Huber(snr) / scheduled-Huber clamp / `inv_loss_ema` 加权 / VeCoR 负样本 / fixed-grid eval loss
- 噪声族：noise offset（随机强度）/ 金字塔噪声 / **Improved Immiscible（KNN 噪声选择）**

### 3. 辅助 loss 与进阶技术

- **Spectral**（RAPSD，含小波）/ **Perceptual**（LPIPS+DINO，t-gate，只算活跃 timestep）/ **LPL**（冻结 VAE decoder 中间特征感知目标）
- **CSFlow**（首跑自动计算 RAPSD）/ **Dispersive**（中间表征排斥正则）/ **Self-perceptual SFT**（arXiv:2401.00110）
- 偏好优化家族：**Linear-DPO / NCP perceptual DPO / Eisbach log-barrier / LeapAlign 两步自蒸馏**
- **GAF 梯度一致过滤**（脏数据鲁棒；ghost ~1x 后端 = TRAK 随机投影逐样本梯度）
- **TREAD** token routing（dense 路径，arXiv:2501.04765）

### 4. 优化器（`docs/optimizer-params.md`）

`adamw` / `adamw8bit` / `prodigyplus` / `soap` / `soap_sf` / `adopt` / `lion` / `clion` / `emosens` / `muon` / `muon_sf` / `automagic`（逐元素自适应 lr）/ `adamw_snr`（梯度信噪比门控）；含 resume 状态与 fp32 master 一致性修复

### 5. NaViT / Patch-n-Pack 打包训练（`docs/navit-packing.md`）

- 块对角打包注意力（packed 与 dense 前向等价性有测试锁定）+ token 预算打包 sampler + 逐图 loss 组装
- 打包旋钮：原生分辨率定尺寸 / 多尺度阶梯 / 按代价装包 / 双约束 `navit_pack_token_cap` / 变长 `text_seqlens` / per-image AdaLN
- `sdpa_seg` 分块注意力（非 xformers 平台）；`grad_checkpoint_policy` 用 token 预算换更便宜的重算策略；`compile_blocks`（逐块 `torch.compile`）
- VAE 缓存：`cache_encode_tiled` 超大图分块 encode（封顶 cache 阶段显存峰值）/ `cache_encode_max_pixels` 编码像素预算

### 6. 显存与量化（`docs/base-quant.md`）

- **冻结底模 FP8/FP4 量化**（GEMM 能力探测带数值验收——能跑且算得对才算可用）
- `lokr_compute_dtype=native`（训练期中间量 bf16）；DoRA 输出域分解（避免物化全量权重矩阵）；meta device 快速初始化（跳过 2.09B 参数的无用随机初始化）

### 7. 模型族（`docs/krea2-family.md`）

- `model_family` 通用接口：**Anima 与 Krea 2（K2 单流 MMDiT）共用同一套训练器**（含 navit 打包）
- heretic 版 Krea2 文本编码器；caption token 上限可解除（默认 512=官方口径）；txtfusion layerwise SDPA 强制 MATH 后端（绕开 sm120 flash backward IMA）

### 8. 异构硬件

| 平台 | 状态 | 入口与文档 |
|---|---|---|
| NVIDIA CUDA | 基准平台 | `run.sh` / `AnimaLoraToolkit/README.md` |
| **海光 DCU K100-AI**（gfx936 / DTK） | 真机跑通 | `run_dcu.sh`；`docs/hygon-dcu.md`、`docs/dcu-image-release.md`、`docs/scnet-image-build.md`；`Dockerfile.dcu`（DAS 预编译 flash_attn/triton 轮子 + 构建期自检 + 前端 IDE）；单机 8 卡数据并行；VAE/注意力分块防 O(S²) |
| **昇腾 Ascend 910B**（启智/OpenI） | 真机跑通 | `run_npu.sh`；`docs/ascend-npu.md`；`utils/npu_compat.py` 兼容层 + `npu_probe.py` 能力探针 + `npu_setup_image.sh` 镜像装配；含 torch_npu 广播 matmul backward bug 的定位与绕行 |
| **Google TPU v5e-8** | JAX 全套移植 + Kaggle 实测 | `AnimaLoraToolkit/jax_tpu/`（C12 配方：LoKr+DoRA、三峰+自适应 t、Huber(snr)、Eisbach/ΔFM/spectral）；`kaggle_job/` CLI 工作台（探针 → torch_xla → 真训练）；FSDP 实测 9.8k tokens/s、并行效率 95% |

### 8.1 环境与依赖（跨架构参考）

DCU 侧有一份**实测**的完整环境锁定：[`AnimaLoraToolkit/requirements-dcu-freeze.txt`](AnimaLoraToolkit/requirements-dcu-freeze.txt) —— 训练镜像内 `pip freeze` 的原样输出（含 IDE 层与基础镜像自带包，不可直接 `pip install -r`）。核心组合：`torch==2.9.0+das.opt1.dtk2604` + `flash_attn==2.8.3+das.opt1.dtk2604.torch290` + `triton==3.5.1+das.opt1.dtk2604.torch290`（均为 DAS 适配轮子，PyPI 无）+ `transformers==4.57.6` + `numpy==1.25.0`（constraints 钉死）。

以该实测组合为基线，其余架构的对应关系推断如下——**除标注「实测」外均为推断，未逐项验证**：

| 组件 | 海光 DCU（实测） | NVIDIA CUDA（推断） | 昇腾 NPU（推断） | TPU（推断） |
|---|---|---|---|---|
| 框架 | `torch 2.9.0+das`（DTK 26.04 基础镜像自带） | `torch 2.9.x + cu130`（见快速开始） | `torch + torch_npu`，版本组合由 CANN 版本决定（`npu_recon.sh` 实测侦察） | JAX：Kaggle 镜像自带 jax 0.10.2，自举升级至 0.11.0 实测可行；torch_xla 钉旧版 libtpu（实测存在版本剪刀） |
| 注意力后端 | `flash_attn 2.8.3+das` / `triton 3.5.1+das` | 官方 `flash-attn` / `xformers`；sm120 有 flash backward IMA 已知问题（krea2 txtfusion 路径已强制 MATH 后端绕行） | 昇腾原生算子（仓库自实现 `npu_tnd` / `sdpa_seg` / 广播 mask 展开适配） | `jax_tpu` 自实现块对角打包注意力 |
| torchvision | 镜像内不装（constraints 防误装） | 正常安装（`requirements.txt`） | **不装**（实测：昇腾上安装会破坏环境） | 训练路径不依赖 |
| 核心训练依赖 | `transformers 4.57.6` / `tokenizers 0.22.2` / `safetensors 0.7.0` / `sentencepiece 0.2.2` / `einops 0.8.2` / `omegaconf 2.3.1` | 同左——纯 Python / 通用轮子包可直接对齐 DCU 实测版本 | 同左（与硬件无关；sentencepiece 已进启动预检） | JAX 路线为独立精简依赖集（`jax_tpu/`）；torch_xla 路线同左 |
| 需成套钉死的组合 | torch/triton/numpy（constraints-dcu.txt） | torch + flash-attn/xformers 的 CUDA·sm 匹配 | torch ↔ torch_npu ↔ CANN ↔ 驱动 四层对齐 | libtpu ↔ jax ↔ torch_xla 三者成套（Kaggle 上已实测裁决） |

### 9. 可观测性与训练运维

- **telemetry 总线**（image-blind，opt-in）+ `adaptive_bin_report` 探针
- `stage_timing` 分阶段计时（含 tokens 列，定位 NaViT vs ARB 速度根因）；NaN/Inf 诞生点定位 hook；单步 `torch.profiler` 旋钮
- 显存守卫体系：`gpu_vram_guard.py`（占位/让出/互斥接力）、`gpu_hog.cu`（FP32 FMA 压测内核）、`run.sh` 全程显存曲线、`npu_mem_monitor.sh`
- 诊断脚本归档（`tests/diag_*.py`：NPU 广播梯度、DoRA w1 梯度方向、ABBA 根因、精度下限…）

### 10. 工具脚本

- **训练侧**：`compute_rapsd` / `cache_text_features` / `dataset_encrypt`（数据集脱敏）/ `openi_fetch`（启智纯标准库下载器）/ `convert_comfy_te_to_hf`
- **分析侧**（根 `tools/`）：`lokr_block_scan` / `lora_block_profile` / `measure_lora_delta` / `lora_ckpt_average` / `comfy_lora_scan` / `inspect_lora_keys`
- **导出/取证侧**：`lora_compress` / `lora_svd_compress` / `abba_export_lora` / `lora_delta_forensics` / `lokr_strength_probe`

## 快速开始

详细教程见 [`AnimaLoraToolkit/README.md`](AnimaLoraToolkit/README.md)，四步走：

```bash
# 1) 环境
python -m venv .venv && .venv\Scripts\activate       # Linux: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r AnimaLoraToolkit/requirements.txt

# 2) 模型（Anima 主模型 + Qwen-Image VAE + Qwen3-0.6B，下载后放 AnimaLoraToolkit/models/）
cd AnimaLoraToolkit && python download_tokenizers.py

# 3) 数据集：图片 + 同名 .txt（Danbooru 风格标签）或 .json（结构化标签，推荐）

# 4) 训练
cp config/train_template.yaml config/my_training.yaml   # 按需编辑
python anima_train.py --config ./config/my_training.yaml
```

训练范式提示：本项目的常规用法是**设一个极大 epoch 一直训下去，随时手动停、从任意 step 的 checkpoint 里挑模型**——所以 `save_every_steps` 建议开小（如 500），配合 `save_state_every`（断点续训）使用。

## 文档索引

| 文档 | 内容 |
|---|---|
| [`AnimaLoraToolkit/docs/noise-timestep-params.md`](AnimaLoraToolkit/docs/noise-timestep-params.md) | timestep 采样 / loss 类型与加权 / 噪声 / 辅助 loss 全参数参考 |
| [`AnimaLoraToolkit/docs/optimizer-params.md`](AnimaLoraToolkit/docs/optimizer-params.md) | 全部优化器机理与参数 |
| [`AnimaLoraToolkit/docs/navit-packing.md`](AnimaLoraToolkit/docs/navit-packing.md) | NaViT 打包训练开启方式与全部旋钮 |
| [`AnimaLoraToolkit/docs/krea2-family.md`](AnimaLoraToolkit/docs/krea2-family.md) | Krea 2 模型族训练指南 |
| [`AnimaLoraToolkit/docs/base-quant.md`](AnimaLoraToolkit/docs/base-quant.md) | 冻结底模 FP8/FP4 量化 |
| [`AnimaLoraToolkit/docs/abba.md`](AnimaLoraToolkit/docs/abba.md) | ABBA 适配器 |
| [`AnimaLoraToolkit/docs/ascend-npu.md`](AnimaLoraToolkit/docs/ascend-npu.md) | 昇腾 910B 适配（含已知硬约束与结案 bug） |
| [`AnimaLoraToolkit/docs/hygon-dcu.md`](AnimaLoraToolkit/docs/hygon-dcu.md) / [`dcu-image-release.md`](AnimaLoraToolkit/docs/dcu-image-release.md) / [`scnet-image-build.md`](AnimaLoraToolkit/docs/scnet-image-build.md) | 海光 DCU 适配、训练镜像发布、scnet 镜像构建规范 |
| [`AnimaLoraToolkit/docs/regularization-analysis.md`](AnimaLoraToolkit/docs/regularization-analysis.md) / [`trainer-optimization-analysis.md`](AnimaLoraToolkit/docs/trainer-optimization-analysis.md) | 正则化方案与训练器架构分析 |
| [`AnimaLoraToolkit/docs/training-tips.md`](AnimaLoraToolkit/docs/training-tips.md) / [`tagging-guide.md`](AnimaLoraToolkit/docs/tagging-guide.md) / [`json-caption-format.md`](AnimaLoraToolkit/docs/json-caption-format.md) | 训练经验、打标与 caption 规范 |
| [`kaggle_job/README.md`](kaggle_job/README.md) | Kaggle TPU 训练全流程（**该目录可整体独立发布**：`jax_tpu` 主拷贝 + 自包含的数据缓存工具，对本仓库零运行时依赖；两边一致性由它的 `tools/check_sync.py` 与逐 bit 缓存对拍守） |
| [`docs/superpowers/`](docs/superpowers/) | 设计文档（plans / specs） |

## 上游与许可

- 训练器内核源自 [AnimaLoraToolkit](https://github.com/WalkingMeatAxolotl/AnimaLoraToolkit)（本仓库对其做了大量修复与再工程化）；训练好的 LoRA 推理推荐搭配 [ComfyUI-AnimaTool](https://github.com/Moeblack/ComfyUI-AnimaTool)
- 许可：**GPL-3.0**（含派生自 ComfyUI 的建模代码）；NVIDIA Cosmos 相关文件为 Apache-2.0。详见 [`AnimaLoraToolkit/THIRD_PARTY_NOTICES.md`](AnimaLoraToolkit/THIRD_PARTY_NOTICES.md)
