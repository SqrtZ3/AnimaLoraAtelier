# 从 Studio fork 发布 DCU 版：现状盘点与路线

**目标**：把 AnimaLoraStudio 的 GUI 工作台带到海光 DCU（scnet），作为可发布、可拿平台激励
（视频 / 镜像 / 使用例）的产品，同时保持向上游回馈 PR 的能力。

**决策（2026-08-25，用户拍板）**：走 fork 路线，发布门面就用现有 fork
`SqrtZ3/AnimaLoraStudio`。**一切对比与 PR base 用 `dev`，不是 `master`。**

> 本文中「已验证」= 读过代码 / 跑过命令确认；「推断」= 由已知事实外推，标了置信度；
> 「待验证」= 现在下不了结论，需要真机或用户信息。

---

## 1. 仓库拓扑（已验证）

```
WalkingMeatAxolotl/AnimaLoraStudio     上游。default_branch=master，开发在 dev
                                        我们的权限：pull=true, push=false（只读）
  └─ SqrtZ3/AnimaLoraStudio            我们的 fork（parent 指向上游）★ 发布门面
                                        fork/dev = f1aecb0（已快进对齐 upstream/dev，落后 0）
SqrtZ3/AnimaLoraAtelier                非 fork 的独立公开仓库 = 本地 anima-lora-train
                                        多平台研究线（NPU / DCU / TPU）
（kaggle_job → 独立仓库，TPU 托管训练）
```

本地工作副本：`D:\ArtificialIntelligence\animafix\AnimaLoraStudio-fork`
（remote：`fork` = 我们的 fork，`upstream` = 上游）。

Atelier 与 `studio/dev` 的共同祖先是 `497b76a6`，分叉后我们 303 commits / 上游 707 commits。
历史相关，git 层面能合，**但不要合** —— 上游把整个 runtime 重构过
（`anima_train.py` 5138 行单体 → 5.5KB 入口 + `runtime/training/{loop,phases,families,adapters,...}`），
硬合只会得到无意义冲突。**Atelier 的角色是"补丁来源"，不是"合并基"。**

### 已被上游合入的我们的贡献（估算增量时必须先扣掉）

| PR | 内容 | 状态 |
|---|---|---|
| #371 | NaViT / Patch-n-Pack 块对角打包 + 分块 VAE 缓存 | MERGED |
| #372 | navit 原生分辨率定尺寸 + 超预算 downscale | MERGED |
| #271 | SOAP + Schedule-Free SOAP optimizer | MERGED |
| #75 | huber loss + `losses/` 插件子包 | MERGED |
| #73 | mixed_uniform timestep 采样 + schedule_shift | MERGED |
| #72 | detail_inv_t loss weighting 上下限可配置 | MERGED |
| #272 | TREAD token routing | OPEN |
| #373 | navit_multiscale 多尺度阶梯 | DRAFT |

---

## 2. 配置面盘点：我们 253 flags × 上游 165 schema 字段

方法：`anima_train.py` 的 `--flag` 归一成下划线，与 `studio/domain/training.py`
的 `TrainingConfig` 字段名取差集。**注意这是名字层面的差集，不等于功能差集** ——
下面第 2.2 节就是被名字骗到的那部分。

### 2.1 数字

| 类别 | 数量 |
|---|---|
| 我们有、上游名字里没有 | 180 |
| ├ argparse 否定式 artifact（`no_*`） | 3 |
| ├ cadence 别名（`*_every_steps`，我们自己的兼容层） | 7 |
| ├ CLI-only / 路径别名 | 14 |
| └ **实质差异** | **156** |
| 上游有、我们没有（fork 后白拿） | 89 |

### 2.2 只是改了名字的（必须做名字映射，不是移植）

| 我们 | 上游 |
|---|---|
| `lr` | `learning_rate` |
| `transformer` / `vae` / `qwen` / `t5_tokenizer` | `transformer_path` / `vae_path` / `text_encoder_path` / `t5_tokenizer_path` |
| `flow_shift` / `schedule_shift` | `timestep_shift` / `timestep_schedule_shift` |
| `loss_weighting_scheme` | `loss_weighting` |
| `pyramid_noise_iterations` | `pyramid_noise_iters` |
| `prodigyplus_*` | `ppsf_*` |
| `sample_dataset_prompts` | `sample_prompts` |
| `fit_*`（10 个：`fit_packed_training` / `fit_max_tokens` / …） | `navit_*`（`navit_packing` / `navit_token_budget` / …） |
| `tlora_alpha` / `tlora_init` / `tlora_rmin_ratio` | `tlora_alpha_rank_scale` / `tlora_use_ortho` / `tlora_min_rank` |

`fit_* ↔ navit_*` 这一组尤其要小心：**是同一个功能**（我们送上去的 #371），
上游在合并时用了 `navit_` 前缀，我们本地后来改成了 `fit_`。按名字比会误判成"上游没有打包"。

### 2.3 fork 后白拿的（上游有、我们没有，89 项）

值得点名的：

- **SRA**（`sra_enabled` / `sra_block` / `sra_decay_*` / `sra_normalize` / `sra_weight`）
- **LEAP 完整参数面**（我们只有 `leap.py`，上游有 `leap_variant` / `leap_traj_sim_*` / `leap_nested_grad_coe` 等 8 项）
- **InfoNoise**（`infonoise_enabled` / `infonoise_beta` / `infonoise_gate_pivot_c`）
- **block swap**（`blocks_to_swap`）—— 低显存训练的关键，DCU 上很可能用得着
- **`attention_backend`**（统一的注意力后端选择）—— **DCU/NPU 适配的天然挂载点**
- **masked_loss / kv_trim / vae_tiling / text_encoder_cache**
- **DoRA / lora_module_dropout / lora_rank_dropout / lora_reg_dims / lora_rs**
- **CAME 优化器全参数面**，以及 automagic / lion / prodigy / ppsf / soap 的完整参数面
- **eval validation split**（`eval_validation_enabled` / `eval_validation_split_ratio|seed`）

### 2.4 真正只有我们有的（156 项，按功能簇）

| 簇 | 项数 | 上游有无 | 初判去向 |
|---|---|---|---|
| `adaptive_timestep_*` | 14 | 无 | 可上游（独立特性，能 opt-in） |
| `aux_perceptual_*`（DINO / LPIPS） | 10 | 无 | 可上游，但拖 `lpips`/`pytorch-fid` 依赖，需按可选依赖处理 |
| `timestep_*`（laplace / logsnr / stratified / mix anneal / t_min,t_max） | 12 | 无 | 可上游 → `timestep_samplers/` 插件 |
| `bucket_*`（我们的 ARB 扩展） | 9 | 无 | 需先确认与上游 dataset 的分桶实现是否冲突 |
| `aux_spectral_*`（小波 / 频谱） | 5 | 无 | 可上游 → `losses/` 插件 |
| `telemetry_*` + `stage_*` + `debug_*` | 11 | 无 | 可上游（可观测性，风险低）**且对 DCU 取证特别有用** |
| `tread_*` | 4 | PR #272 OPEN | 推动既有 PR，不重开 |
| `navit_multiscale*` + `navit_pack_cost_*` | 5 | #373 DRAFT | 推动既有 PR |
| **`navit_attn_backend`** | 1 | 无 | **平台线核心**：`npu_tnd` / `sdpa_seg` 保底路径，DCU 只有 math SDPA 时必需 |
| `csflow_*` | 4 | 无 | 实验性，先留 Atelier |
| `tlora_lokr_*`（LoKr × T-LoRA 组合） | 2 | 无 | 实验性 |
| `lokr_w1_*` / `lora_one_*` / `dfm_*` / `lwd_*` / `eisbach_lambda` | 8 | 无 | 实验性，先留 Atelier |
| `alpha_*`（RGBA / 透明通道处理） | 3 | 无 | 可上游（数据侧，独立） |
| `eval_*`（我们自己那套） | 4 | 上游有另一套 | 语义冲突，需先对齐再谈 |
| 其余 misc | ~30 | 混杂 | 逐项判 |

**这张表还没做"逐项读代码确认"，是按名字 + 已知上下文的初判，置信度中。**
每一项真要上游前都得单独确认上游是不是已用别的名字实现了（见 2.2 的教训）。

---

## 3. DCU 适配：真实工作面比预期小

### 3.1 好消息：DCU 不需要设备重映射（已验证）

依据 `AnimaLoraToolkit/utils/dcu_compat.py` 的设计说明：DTK 是 ROCm/HIP 衍生版，
海光适配版 PyTorch（`torch==2.x+das.optN.dtkXXXX`）**本身就是 HIP 后端的 torch**，
`torch.cuda.is_available()` / `torch.device("cuda")` 直接就是 DCU，
`torch.version.hip` 非空、`torch.version.cuda` 为空。

因此 `studio/dev` 的 runtime 里那些 `torch.cuda` 调用（`anima_daemon.py` 29 处、
`vae.py` 10、`sample_runner.py` 9、`families/anima/sampling.py` 9、`sysmem.py` 8、
`block_swap.py` 8，合计约 97 处）**在 DCU 上本来就是对的，不需要改**。

需要重映射的是昇腾 NPU（`torch_npu.contrib.transfer_to_npu`），而 NPU 因为启智不开端口，
本来就不是 GUI 的目标平台。

### 3.2 我们的兼容层是入口 shim，不是侵入式改动（已验证）

`utils/dcu_compat.py`(358 行) 与 `utils/npu_compat.py`(277 行) 都是
**opt-in / default-off**，整个训练器里的挂载点只有：

- `anima_train.py:1113-1164` —— 设备选择 / `guard_unsupported` / allocator env
- `models/anima_modeling.py:210`、`models/anima_modeling_core.py:507` —— NPU 的 mask 展开

移到上游的 `runtime/training/phases/bootstrap.py` 是同一类工作。

### 3.3 依赖与自更新层（读码后修正过两处，见下）

**上一版这张表有两条说重了，更正：**

- ~~`studio.sh:163` 的 `pip install torch --index-url <CUDA源>` 是最高风险~~ ——
  该行只在 `tools/select_torch_index.py` 检测到 **NVIDIA 驱动**时才触发
  （靠 `nvidia-smi`）。DCU 上没有 nvidia-smi → 脚本静默无输出 → **这行根本不执行**。
- ~~`studio/api/routers/system.py:270` 自更新会跑 pip install~~ —— 那只是一句 UI 文案
  （"预计 pip install 1-2 分钟"）。`studio/services/updater.py` 里没有 pip 调用，
  真正的安装发生在下次 `studio.sh` 启动时的 stale-marker 重同步路径。

**真正致命的是 venv 隔离，不是 index-url：**

| 位置 | 内容 | 风险 |
|---|---|---|
| `studio.sh` / `studio.bat` 建 venv 时**不带** `--system-site-packages` | DTK 的 das torch 装在系统解释器里，普通 venv 看不见 → `requirements.txt` 的 `torch>=2.0.0` 读作**未满足** → pip 从 PyPI 装一个同名 CPU wheel 进来，**exit 0** | **最高**，且完全静默 |
| stale-marker 重同步 `pip install -r requirements.txt` | 目前没有 `--upgrade` 所以一般不动 torch，但没有任何机制**保证**这一点 | 中 |
| `studio/api/routers/installs.py:263` | GUI 按钮 `pip install xformers --index-url <torch-cu-index>` | 中。DCU 上会装到 CUDA 版；但海光有专用编译版，所以该做的是**改索引源**而不是禁用按钮 |

**已落地的修复**（分支 `feat/pin-vendor-torch`）：

- `tools/pin_installed_torch.py` —— 把已装 torch 的**精确版本**（含 `+das` local 段）
  写成 pip constraints，两个启动脚本每一处 `pip install -r` 都带 `-c`。
  pip 之后想改动 torch 会**直接报错说明原因**，而不是静默替换。
- `studio.sh` / `studio.bat` 新增 `--system-site-packages`。
- 首装阶段已能 `import torch` 时跳过 torch 安装。

这不是 DCU 专属修复 —— ROCm 用户、自编译 torch 的用户同样受益，是个干净的上游 PR。

## 4. scnet 单端口：一个可能的阻断点（待验证）

已验证的部分：

- 服务端口是参数化的：`studio/api/main.py:26-27` 默认 `--host 127.0.0.1 --port 8765`，
  `studio/cli.py:867` 的注释里甚至直接写了 `studio.sh --port 6006` 的例子。
  改 `--host 0.0.0.0 --port <scnet 端口>` 没有障碍。
- 前端构建产物由 FastAPI 挂在**根路径 `/`** 下（ADR 0012），
  `vite.config.ts` 里 `base: '/'`，`client.ts` 的 API 调用是**同源相对路径**。
- 全仓 **没有 `root_path` / `base_href` 支持**（grep 无命中）。

**已解答（2026-08-25，用户）**：scnet 是在控制台填一个端口号，由**平台域名路由到内网端口**，
不是路径前缀代理；用户已实测「改个内网端口就可用」。

→ **不需要 `root_path` / `base` 支持**，M2 退化为「把 host 改 `0.0.0.0`、port 改成控制台
登记的内网端口」。`studio/api/main.py:26-27` 与 `studio/cli.py` 都已参数化，无代码障碍。

---

## 5. 其余待验证清单

| # | 问题 | 怎么验 | 阻断性 |
|---|---|---|---|
| ~~V1~~ | ~~scnet 端口暴露形式~~ | **已解答**：域名路由到内网端口，非路径代理 | — |
| V2 | Studio 的 runtime 在 DTK torch 上能否跑通一次 N 步训练 | 真机 | 高 |
| V3 | DCU 上 SDPA 落到哪个后端（若只剩 math → NaViT 打包必炸显存） | `tools/dcu_probe.py` | 高（已有探针） |
| V4 | `pip install -r requirements.txt` 是否真的不动 das torch | 真机 dry-run | 中 |
| V5 | `nvidia-ml-py` / `onnxruntime-gpu` / `spandrel` 在 DTK 环境装不装得上 | 真机 | 中 |
| V6 | 启智 NPU 能否通过 jupyter 做端口代理 | 真机 | 低（NPU 线可先只给 CLI） |
| V7 | 上游 dataset 的分桶实现与我们的 `bucket_*` 扩展是否冲突 | 读码 | 中 |

**在 V2 出结果之前，本文第 6 节之后的所有时间估计都不成立。**

---

## 6. 路线

### M0 · 把 fork 接回上游 ✅

- [x] clone fork 到本地，加 `upstream` remote
- [x] `fork/dev` 快进到 `upstream/dev` → 两边同为 `f1aecb0`，落后 0
- [x] 建长期分支 `dcu-dist`（发布线），已合入两条 feature 分支

分支约定：

| 分支 | 用途 |
|---|---|
| `dev` | 纯跟随上游，不放我们的东西。所有对比与 PR base |
| `feat/*` | 每个可上游的改动一条，从 `dev` 切，发 PR 到 `upstream:dev` |
| `dcu-dist` | 发布线。承载上游不会要的：DCU 镜像、Dockerfile、scnet 启动脚本、中文教程 |

### M1 · DCU 能跑（不动 GUI）—— 补丁已就绪，待真机

分支 `feat/device-backend-dcu`（本地已提交 `e93c580`，未推）：

- `utils/dcu_compat.py` —— 从 Atelier 移植，guard 改按上游字段名
  （`attention_backend` / `navit_packing` / `optimizer_type`）
- `studio/domain/common.py` —— `DeviceBackend = Literal["auto","cuda","dcu"]`
- `studio/domain/training.py` —— `device_backend` 字段，默认 `auto`
- `runtime/training/phases/bootstrap.py:174` —— 单点挂载
- `tests/test_device_backend_dcu.py` —— 22 项，不需要硬件

移植时修掉了 Atelier 原版的一处顺序错误：`set_allocator_env()` 原本在
`enable()` **之后**调用，而 `enable()` 里的 SDPA 探针已经分配过张量 ——
PyTorch 的 caching allocator 只在首次分配时解析一次 `PYTORCH_*_ALLOC_CONF`，
所以原版的 `expandable_segments:True` 在 DCU 上大概率**静默没生效**。
新版把它提到 `enable()` 最前面。（置信度高，但**未在真机验证**；
Atelier 侧同样的问题还没修。）

**仍需真机裁决**：V2 / V3 / V4。这一步是整条路线的生死判据。

### M2 · GUI 在 scnet 上可达 —— 脚本已就绪

`studio_scnet.sh`（`dcu-dist` 分支）：source DTK env + 从 jupyter 进程捞内网代理 +
`--host 0.0.0.0 --port <控制台端口>` + 一个 `--check` 体检（torch 必须是 HIP 构建
且能看到卡，否则拒绝启动）。本地非 DCU 环境跑 `--check` 已验证会正确拒绝。

### M3 · 依赖与自更新的平台感知 —— 主体已完成

分支 `feat/pin-vendor-torch`（本地 `2c7260e`，未推）。见 §3.3「已落地的修复」。
12 项测试。**未在 DTK 真机验证**；`studio.bat` 的 if/else 嵌套改动也没在
Windows 上实跑过。

未做：`installs.py` 的 xformers 按钮索引源（要先知道海光专用编译版怎么装）。

### M4 · 发布物

`Dockerfile.dcu` → scnet 镜像；中文 getting-started；使用例 / 视频素材。

### M5 · 特性回流

按 §2.4 的表，把可上游的簇逐个发 PR；只进发行版的留 `dcu-dist`。

### M6 · TPU 风格统一（最后）

TPU 是"本地准备 → 推 Kaggle 托管 → 只能停不能调"，不该塞进同一运行时。
统一点放在配置 schema 与视觉外壳：GUI 里表现为"准备并导出"向导 + 只读状态页。
