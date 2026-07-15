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

## Alibaba Wan2.1 VAE（请再次确认上游许可）

- **来源**：`Wan-Video/Wan2.1` 的 VAE 实现（与 `wan/modules/vae.py` 对应）
- **涉及文件**：
  - `models/wan/vae2_1.py`

该文件头目前仅包含版权声明（未显式 SPDX）。上游仓库通常宣称 Apache-2.0，但建议你在开源前**再次核对上游仓库的 LICENSE/NOTICE**，确保分发合规。

---

如你希望把项目改为更宽松的许可（例如 MIT），需要先移除/替换所有 GPL-3.0 派生部分，并重新梳理第三方依赖的许可兼容性。

