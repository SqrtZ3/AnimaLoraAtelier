# 昇腾 Ascend 910B（OpenI/启智）适配

> 状态标注很重要，先说清楚：本文里**只有"本地可验证"那几条是实测**（CUDA 路径行为中立、
> 编译/导入通过）。所有"NPU 上 X 能不能用"的说法**都未在真机验证**——那正是
> `tools/npu_probe.py` 存在的原因。不要把本文当成"已经跑通"的记录。

## 0. 一句话结论

代码侧已经加了 **opt-in 的 NPU 后端开关**（默认关，CUDA 路径逐字节不变）+ **真机能力探针**。
但 910B 与我们现有配方之间有几条硬约束（xformers / FP8 / NaViT），**必须先跑探针再决定配方**。

## 1. 怎么开

二选一，等价：

```yaml
# config/*.yaml
device_backend: npu     # 默认 auto = 有 CUDA 走 CUDA（历史行为）
```

```bash
ANIMA_NPU=1 python anima_train.py --config config/xxx.yaml
```

开启后 `utils/npu_compat.py` 做三件事：

1. `import torch_npu` + 打上昇腾官方兼容补丁 `torch_npu.contrib.transfer_to_npu`
   （把 `torch.cuda.*` 映射到 `torch.npu.*`），这样仓库里既有的
   `torch.cuda.empty_cache/synchronize/Event/OutOfMemoryError` 等 ~36 个调用点无需逐个改写。
2. autocast 的 `device_type` 显式换成 `"npu"`（`anima_train.py` / `trainer/aux_losses.py` /
   `trainer/sampling.py` 三处已接线）。**不依赖** transfer_to_npu 是否覆盖 autocast——
   各版本行为不一致，这是热路径，不赌。
3. `guard_unsupported()` 在构造期 fail-fast 掉已知不可用的开关，并在报错里给替代路径。

导不进 `torch_npu` / 没有可见 NPU 时**直接抛错**，不静默回退 CPU。

## 2. 已知硬约束（`guard_unsupported()` 会拦）

| 功能 | 状态 | 替代 |
|---|---|---|
| `base_quant`（FP8/FP4 冻结底模量化） | 910B 无 FP8 张量核 | `base_quant: none` |
| `navit_packing`（Anima family） | 依赖 xformers `BlockDiagonalMask`，昇腾无 xformers | 关 navit，走 ARB 稠密路径 |
| `navit_packing`（krea2 family） | 同上 | `navit_attn_backend: sdpa_seg`（逐段 dense SDPA，与块对角语义数学恒等，有单测对拍） |
| `torch_compile` | 昇腾支持度未验证 | `torch_compile: false` |
| 8-bit 优化器（bitsandbytes） | 无昇腾后端 | `adamw` / `adamw_snr` / `automagic` 等纯 torch 实现 |

未拦但**需要探针确认**的：`torch.fft`（spectral aux loss 依赖）、稠密 mask 下的 SDPA 后端与显存、
`torch.npu.Event`（`stage_timing_every > 0` 才用到，默认关）。

## 3. 上机第一件事：跑探针

```bash
python tools/npu_probe.py --json /tmp/npu_probe.json
```

它不加载任何底模、几十秒跑完，逐项实测并打汇总表：torch/torch_npu/CANN 版本、`npu-smi`、
bf16/fp32 matmul 数值、`autocast("npu")`、SDPA（无 mask / bool mask / additive mask，含 backward
与峰值显存）、`torch_npu.npu_fusion_attention`、`torch.fft`、Conv/GroupNorm/SiLU、
`torch.utils.checkpoint`、AdamW 单步、Event 计时、显存 API、bf16 粗略 TFLOPS，以及依赖包可用性。

**FAIL 项就是这台机器的真实限制**，配方必须绕开它，不要靠猜。

## 4. 还没做的（诚实清单）

- 未在真机跑过任何一步训练；步时/显存/收敛全部未知。
- Anima family 在昇腾上**没有** NaViT 打包的等价路径（krea2 的 `sdpa_seg` 只在 krea2 接线）。
  如果要在 NPU 上跑 Anima + navit，需要把 `sdpa_seg` 那套逐段 dense SDPA 移植到
  `models/anima_modeling_core.py`——这是一笔独立工作，尚未做。
- 分布式（多卡 HCCL）未接线；当前只考虑单卡。
- `trainer/quant.py` 全部走 CUDA 专用路径，NPU 上由守卫拦掉，未做适配。

## 5. OpenI/启智 侧的事实（2026-08 查自平台帮助文档与镜像列表）

**可用镜像**：在 `/explore/images` 用「昇腾NPU + PyTorch」筛选，**全站公开镜像只有 2 个**：

| 镜像 | 框架 | Python | CANN | 用量 |
|---|---|---|---|---|
| `torch-npu-cann8-debug` | PyTorch 2.1.0 | 3.9 | 8.0.RC1 | 10538 |
| `openmind_ms_v8-notebook` | PyTorch 2.1.0 | 3.8 | 8.0.RC3 | 1062（`conda activate openmind`） |

标注为 MindSpore 但描述写「包含多个 conda 环境」的几个（`mindspeed-llm` CANN 8.0.RC3 /
`cann8_3_rc1_nnal` CANN 8.3.RC1 Py3.11「只能用于 D910B」）**可能**另含 torch-npu 环境
——未验证，上机 `conda env list` 才算数。

**torch 2.1.0 的影响**：`F.scaled_dot_product_attention(enable_gqa=)` 需要 torch≥2.5，但
`models/krea2_modeling.py:146` 已有版本探测与手动展开 KV 头的 fallback（数值相同），
不是阻塞项。为兼容 Python 3.9，仓库里 4 处 PEP 604 注解所在文件已补
`from __future__ import annotations`（行为中立）：`models/anima_modeling_core.py`、
`models/cosmos_predict2_modeling.py`、`utils/caption_utils.py`。全量 AST 扫描
（49 个文件）未发现其余 3.10+ 专属语法。

**自定义镜像**：支持。流程 = 在**运行中**的调试任务里装好环境 → 调试任务列表 →「更多 →
提交镜像」。限制：镜像大小 鹏城计算I ≤20GB / 其他算力中心 ≤40GB；`/tmp/code` 与
`/tmp/dataset` 不入镜像；**NPU 环境下 `/home/ma-user/work` 不入镜像**；提交期间任务转
WAITING 暂不可用。

**调试任务时长**：文档写明**默认 4 小时**上限，可手动停止。

**上传底模/数据集**：用官方 CLI，别用网页拖拽。
```bash
openi login --token <token>        # token: /user/settings/applications
openi model   upload <owner>/<模型名>   <本地路径>
openi dataset upload <owner>/<数据集名> <本地路径> -w 100
```
支持断点续传与并行。另有「新建 → 迁移外部模型」，填 HuggingFace 模型 ID 由平台直接拉取。

**资源挂载**：`from c2net.context import prepare; c2net_context = prepare()`；产物写到
`c2net_context.output_path`。注意 `upload_output()` 回传**只有训练任务**支持，调试任务
的产物需自行下载。

**底模**：平台已有他人同步的 `FoundationModel/Anima`（源 `circlestone-labs/Anima`），
`split_files/` 下三件套齐全：`diffusion_models/anima-base-v1.0.safetensors`(3.89GiB)、
`text_encoders/qwen_3_06b_base.safetensors`(1.11GiB)、`vae/qwen_image_vae.safetensors`(242MiB)。
⚠ 该 TE 是 ComfyUI 单文件格式，而 `trainer/models.py:259` 走 `AutoModelForCausalLM`
需要 HF 目录（`config.json` + tokenizer）；仓库内 `models/text_encoders/Qwen3-0.6B-Base/`
的 tokenizer 齐但 `model.safetensors` 只有 135 字节（LFS 指针，非权重）。**需要一次
单文件→HF 目录的键名转换**，同 memory `[[comfyui-te-key-namespace]]` 记的那类工作。

## 6. 镜像装配

用 `tools/npu_setup_image.sh`（在调试任务终端里跑）：

```bash
bash tools/npu_setup_image.sh --check     # 只体检
bash tools/npu_setup_image.sh             # 装必需依赖 + 跑探针
```

**Python 版本决策：锁 3.9，不升级。** 昇腾上绑死的是「Python ABI × torch 版本 ×
CANN 版本」三元组，换 Python 就要重新对齐另外两个，而 CANN 自带 Python 包
（`te` / `op_compile`，由 `set_env.sh` 注入 PYTHONPATH）是否跨版本通用**未验证**。
收益侧为零——AST 全量扫描已确认仓库在 3.9 上可跑。

**最大的环境杀手**：`pip install torchvision`（或任何间接依赖 torch 的包）会从 PyPI
拉 CUDA 版 torch 覆盖掉 torch_npu 配套的那个，环境当场报废。脚本用动态生成的
constraints 钉死 `torch` / `torch-npu` / `numpy<2`（torch 2.1.x 按 numpy 1.x ABI 编译），
并在安装后校验 torch 版本有没有被改动，被改动就拒绝继续。

**真实依赖比 `requirements.txt` 小得多**（AST 扫描运行路径得出）：必需只有
`numpy<2` / `Pillow` / `safetensors` / `transformers>=4.51,<5` / `PyYAML` / `rich` / `einops`。
`transformers>=4.51` 是硬下限（Qwen3 架构支持）。`diffusers` / `accelerate` / `peft` /
`lycoris-lora` / `torchvision` / `pytorch-fid` 运行路径未 import，不装。
`bitsandbytes`（`utils/optimizer_utils.py:76`）与 `lpips`（`trainer/aux_losses.py:344`）
都是 try/except 可选导入，昇腾上不装即可。

**site-packages 位置陷阱**：NPU 环境下 `/home/ma-user/work` 不入镜像。若解释器的
site-packages 落在那里，装的包提交镜像后全部丢失——脚本第 1 步会检查并报警。
