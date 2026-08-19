# Third-Party Notices

本仓库包含/改写/派生了部分第三方代码与实现片段。请在分发时遵守其许可并保留必要的版权与许可声明。

## ComfyUI (GPL-3.0)

- **来源**：`comfyanonymous/ComfyUI`（现由 Comfy-Org 维护）
- **许可**：GPL-3.0
- **涉及文件**：
  - `models/anima_modeling.py`（实现结构与 ComfyUI 的 `comfy/ldm/anima/model.py` 高度相关）

> 由于包含/派生自 GPL-3.0 代码，本项目整体以 GPL-3.0 发布（见 `LICENSE`）。

## NVIDIA Cosmos (Apache-2.0)

- **来源**：NVIDIA 相关实现（文件内含 SPDX 头）
- **许可**：Apache-2.0（见文件头 `SPDX-License-Identifier: Apache-2.0`）
- **涉及文件**：
  - `models/cosmos_predict2_modeling.py`
  - `models/anima_modeling_core.py`

本仓库额外提供 `LICENSE-APACHE` 以便分发 Apache-2.0 许可文本。

## SOAP Optimizer (MIT)

- **来源**：`nikhilvyas/SOAP` official implementation
- **许可**：MIT（许可文本已保留在派生文件头部）
- **涉及文件**：
  - `utils/soap_optimizer.py`

## ADOPT Optimizer (MIT)

- **来源**：`iShohei220/adopt` official implementation
- **论文**：Taniguchi et al., "ADOPT: Modified Adam Can Converge with Any β2
  with the Optimal Rate", NeurIPS 2024 — arXiv:2411.02853
- **许可**：MIT（许可文本已保留在派生文件头部）
- **涉及文件**：
  - `utils/adopt_optimizer.py`

## Lion Optimizer (Apache-2.0) + Cautious variant (MIT)

- **Lion 来源**：Chen et al., "Symbolic Discovery of Optimization Algorithms",
  NeurIPS 2023 — arXiv:2302.06675。Google Research，Apache License 2.0。
- **Cautious 来源**：Liang et al., "Cautious Optimizers: Improving Training
  with One Line of Code", arXiv:2411.16085；参考实现 `kyleliang919/C-Optim`，MIT。
- **涉及文件**：
  - `utils/lion_optimizer.py`（Lion 类，`cautious=True` 即 C-Lion）

## Automagic Optimizer (Apache-2.0)

- **来源**：`ostris/ai-toolkit` 的 `toolkit/optimizers/automagic.py`
  （逐元素自适应 lr：Adafactor 分解二阶矩 + 符号一致性 lr mask）
- **许可**：Apache-2.0
- **涉及文件**：
  - `utils/automagic_optimizer.py`
- **本地差异**：算法逻辑照搬上游；本仓库改用 **fp32 master** 累积更新，
  替代上游的随机舍入（`copy_stochastic`）——本仓库 LoRA 参数为 bf16，
  起始 lr 下每步更新低于 bf16 ulp，需防静默冻结（与 `muon_optimizer.py`
  的同款修复保持一致）。未移植上游的 8-bit lr_mask 量化（`Auto8bitTensor`）
  与参数交换（paramiter swapping）。

## EmoSens Optimizer (Apache-2.0)

- **来源**：`muooon/EmoSens` official implementation
- **许可**：Apache-2.0
- **涉及文件**：
  - `utils/emosens_optimizer.py`
- **本地改动**：移除了上游 ECC/backward monkey patch，改为训练循环显式注入聚合 loss；
  optimizer state 使用 fp32，以适配本项目的 bf16 LoRA/LoKr 训练。

## Krea 2 (K2) MMDiT (Apache-2.0 代码 / Krea 2 Community License 权重)

- **来源**：`krea-ai/krea-2` 官方推理实现（mmdit.py / encoder.py / sampling.py）；
  技术报告 https://www.krea.ai/blog/krea-2-technical-report
- **许可**：官方仓库代码为 Apache-2.0（见其 LICENSE.md）；模型权重
  （krea/Krea-2-Raw、krea/Krea-2-Turbo）按 **Krea 2 Community License** 分发，
  使用/部署须遵守其 Acceptable Use Policy —— 权重不随本仓库分发。
- **涉及文件**：
  - `models/krea2_modeling.py`（单流 MMDiT 训练移植；模块命名与官方一致以保证
    checkpoint 直载，训练侧差异见该文件头部说明）
  - `trainer/model_family.py`（Qwen3-VL 编码模板/常量、分辨率感知 timestep shift
    公式移植自官方 encoder.py / sampling.py）
  - `jax_tpu/krea2_jax.py`（同一架构的纯 JAX 前向移植，TPU 路线；与
    `models/krea2_modeling.py` 逐算子对齐）
  - `jax_tpu/flow.py` 的 `krea2_mu` / `krea2_res_shift_np`（官方 sampling.py
    分辨率感知 shift 公式的 numpy 移植）

## ABBA-Adapters（论文方法移植）

- **来源**：论文 *ABBA-Adapters: Efficient and Expressive Fine-Tuning of Foundation
  Models*（arXiv:2505.14238，ICLR 2026）；官方实现 `CERT-Lab/abba`（截至引入时
  上游仓库未见 LICENSE 文件——本仓库为按论文公式的独立重实现，未复制上游代码，
  仅对照其 init / scaling / Khatri-Rao 重排的数值口径）。
- **涉及文件**：
  - `trainer/lora.py`（`ABBALayer` 及 LoRAInjector 的 abba 分支）
- **本地差异**：适配本仓库 adapter-输出叠加式注入（官方是包 base_layer 的
  wrapper）；导出为 KR 物化标准 LoRA 键 + native `abba_*` 因子并存。

## AC-LoRA（论文方法移植：训练期 RESTART + 导出 SVD 压缩）

- **来源**：论文 *AC-LoRA: Auto Component LoRA for Personalized Artistic Style
  Image Generation*（arXiv:2504.02231）。本仓库为**按论文公式的独立重实现**，未
  复制上游代码，仅对照其信号/噪声切分（Eq.3 累计能量阈值）与 RESTART（Eq.2：
  信号奇异分量保留、丢弃分量重置为同方差高斯噪声）的数值口径。
- **涉及文件**：
  - `trainer/lora.py`（`aclora_restart_matrix`、`svd_truncate_lora_pair`，以及
    LoRAInjector 的 `aclora_*` / `_maybe_save_compressed` 方法）
  - `tools/lora_svd_compress.py`（导出期逐层 SVD 截断的本地工具）
- **本地差异 / 对论文的偏离（已在代码注释与配置头标注）**：
  1. 论文按 epoch 触发（E=10）、阈值 `p=1−l^α` 依赖 loss<1；本仓库是 step-based
     且底模是 Flow-Matching（loss 不保证 <1、不单调），故默认改用 FM-稳健的
     `schedule` 模式（p 线性从 p_start 升到 p_end），保留 `loss` 模式为论文口径。
  2. 论文未规定最终导出如何降 rank；本仓库把"按信号能量抽取逐层 rank"实现为
     导出期 SVD 截断（save() 额外写 `.compressed.safetensors` 部署件 + 独立工具）。
  3. 首版收窄变量面：仅标准 LoRA（lora_type=lora, variant=base, init=default），
     与 LoKr/ABBA/DoRA/PiSSA/T-LoRA 构造期 fail-fast。均 opt-in / default-off。

## torchao（fp4 block-scale swizzle 布局函数改写）

- **来源**：`pytorch/ao`（torchao，BSD-3-Clause）
  `torchao/prototype/mx_formats/utils.py` 的 `to_blocked`（cuBLAS nvfp4
  GEMM 要求的 128×4 tile block-scale 布局变换）。
- **涉及文件**：
  - `trainer/quant.py`（`_to_blocked`，独立改写；其余量化/GEMM 代码为
    本仓库原创实现，仅依赖 torch 核心的 `_scaled_mm`）
- **本地差异**：仅此一个纯 reshape/pad 函数，避免引入 torchao 整包依赖。

## Alibaba Wan2.1 VAE（请再次确认上游许可）

- **来源**：`Wan-Video/Wan2.1` 的 VAE 实现（与 `wan/modules/vae.py` 对应）
- **涉及文件**：
  - `models/wan/vae2_1.py`

该文件头目前仅包含版权声明（未显式 SPDX）。上游仓库通常宣称 Apache-2.0，但建议你在开源前**再次核对上游仓库的 LICENSE/NOTICE**，确保分发合规。

---

## DCU 镜像内分发的第三方组件（2026-08-18）

以下组件随 `Dockerfile.dcu` 构建的公开镜像（`anima-lora-dcu:*`）分发，发布镜像时请保留
各自许可并附版权声明（许可文本随 pip 包或官方仓库提供）：

| 组件 | 版本（镜像内实测） | 许可 | 来源 |
|---|---|---|---|
| flash_attn（海光 DAS 预编译轮子） | 2.8.3+das.opt1.dtk2604.torch290 | BSD-3-Clause | `Dao-AILab/flash-attention`，由光源 DAS1.8 编译分发（download.sourcefind.cn:65024） |
| triton（海光 DAS 预编译轮子） | 3.5.1+das.opt1.dtk2604.torch290 | MIT | `triton-lang/triton`，由光源 DAS1.8 编译分发 |
| code-server | 4.133.0（tarball 手动打包进镜像） | MIT | `coder/code-server`（github releases 下载） |
| jupyterlab | 4.6.3 | BSD-3-Clause | PyPI（清华镜像） |
| transformers / accelerate | 4.57.6 / 随 pip freeze | Apache-2.0 | PyPI |
| torch / torchvision（基础镜像自带 das 构建） | 2.9.0+das.opt1.dtk2604 | BSD-3-Clause（PyTorch） | 光源基础镜像 `jupyterlab-pytorch:2.9.0-ubuntu22.04-dtk26.04-py3.11-devel` |
| wandb / einops / safetensors / Pillow 等 | 随 pip freeze | 各包许可（MIT/Apache-2.0/BSD 为主） | PyPI |

注意事项：

1. **海光 DAS 预编译轮子（flash_attn/triton）的再分发**：上游开源许可允许再分发，但
   轮子本身由海光光源平台编译提供，公开发布镜像前建议确认光源平台的使用条款是否允许
   再分发其编译产物（如需，可改在发布说明中注明"包含海光 DAS 编译的
   flash_attn/triton，来源 download.sourcefind.cn:65024"）。
2. **镜像内依赖全量清单**：构建期已把 pip 基线冻结在镜像内
   `/opt/anima-build/base_pip_freeze.txt`；实例内可随时 `python -m pip list --format=freeze`
   生成完整清单（见 `docs/scnet-image-build.md` §11 的发布指令）。
3. 本仓库代码许可（GPL-3.0）适用于镜像内 `/opt/anima-lora-train` 的代码副本，
   与上述第三方组件许可相互独立。

---

如你希望把项目改为更宽松的许可（例如 MIT），需要先移除/替换所有 GPL-3.0 派生部分，并重新梳理第三方依赖的许可兼容性。

