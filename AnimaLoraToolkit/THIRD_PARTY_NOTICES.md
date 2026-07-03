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

## Alibaba Wan2.1 VAE（请再次确认上游许可）

- **来源**：`Wan-Video/Wan2.1` 的 VAE 实现（与 `wan/modules/vae.py` 对应）
- **涉及文件**：
  - `models/wan/vae2_1.py`

该文件头目前仅包含版权声明（未显式 SPDX）。上游仓库通常宣称 Apache-2.0，但建议你在开源前**再次核对上游仓库的 LICENSE/NOTICE**，确保分发合规。

---

如你希望把项目改为更宽松的许可（例如 MIT），需要先移除/替换所有 GPL-3.0 派生部分，并重新梳理第三方依赖的许可兼容性。

