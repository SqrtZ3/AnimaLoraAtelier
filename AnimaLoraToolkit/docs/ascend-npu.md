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
| `navit_packing` + `navit_attn_backend: xformers` | 昇腾无 xformers | 换后端，**不用关 navit**（见下） |
| `torch_compile` | 昇腾支持度未验证 | `torch_compile: false` |
| 8-bit 优化器（bitsandbytes） | 无昇腾后端 | `adamw` / `adamw_snr` / `automagic` 等纯 torch 实现 |
| `aux_perceptual_enabled`（LPIPS / DINOv2） | 硬依赖 torchvision，装它会换掉 torch（见 §6） | `false`；确需则用 `npu_setup_image.sh --with-perceptual`（带 constraints）自行验证 |

未拦但**需要探针确认**的：`torch.fft`（spectral aux loss 依赖）、稠密 mask 下的 SDPA 后端与显存、
`torch.npu.Event`（`stage_timing_every > 0` 才用到，默认关）。

### 2.2 广播 attn_mask 要展开（`expand_attn_mask`）

昇腾把 SDPA 落到 `aclnnFlashAttentionScore`，它**只接受** `[B,N,Sq,Skv]` / `[B,1,Sq,Skv]` /
`[1,1,Sq,Skv]` / `[Sq,Skv]` —— `Sq` 必须是真实 query 长度。而 PyTorch SDPA 允许在 query 维广播，
仓库的 key-padding mask 正是 `[B,1,1,Skv]`（`LLMAdapter.forward` 的两次 `unsqueeze(1)`），真机报：

    get unsupported atten_mask shape, the shape is [1, 1, 1, 251].

`utils/npu_compat.expand_attn_mask()` 在 NPU 上把它展开成 `[B,1,Sq,Skv]` 并 `contiguous()`
（`expand` 出的 stride=0 维对融合算子不安全）。要点：

- **展开点在 `LLMAdapter.forward`，不在 attention 层**：self-attn 与 cross-attn 的 query 都是 `x`，
  两个 mask 的 `Sq` 相同且整个 block 栈不变，一次展开即可（放在 attention 里会重复 2×num_layers 次）。
- **行为中立 + 语义等价**：未 `enable()` 时是恒等映射；启用后 `LLMAdapter` 输出与展开前**逐 bit 相同**
  （`tests/test_npu_mask_compat.py`，两份实现都测，含真实 padding）。
- **体积闸门**：DiT 稠密分支（`torch_attention_op`）拿到的是 **bf16 加性** mask，Skv=打包序列长度 N，
  展开是 O(B·N²)（N=8192→134 MB/样本，N=32768→2.1 GB）。超过 256 MB 直接抛错并指向变长打包路径，
  不静默物化；阈值可用 `ANIMA_NPU_MASK_EXPAND_MAX_BYTES` 调。当前昇腾配方走 `navit_packing: true`
  的 `_SegLens` 变长分支，根本到不了这条路。
- **未验证**：真机是否还有其它 mask 形状触发同一限制（目前只覆盖 Sq=1 这一种）。

### 2.1 NaViT 打包在昇腾上是**可用**的（不是必须放弃）

"没有 xformers" 只否掉了默认的那个 kernel，没有否掉块对角打包本身。三个后端算的是同一件事
——打包序列里每张图只看自己的 token、段间零泄漏、不物化 O(ΣN²) 稠密 mask：

| `navit_attn_backend` | 实现 | family | 状态 |
|---|---|---|---|
| `xformers`（默认） | `BlockDiagonalMask` + varlen 快核 | anima / krea2 | CUDA 历史行为，逐 bit 不变 |
| `npu_tnd` | `torch_npu.npu_fusion_attention(input_layout="TND")` + `actual_seq_qlen/kvlen` 累加和 | **anima** | 昇腾原生变长融合注意力；**真机未验证**，先跑探针 |
| `sdpa_seg` | 逐段 dense SDPA | anima / krea2 | 不依赖任何专有算子的保底路径 |

`npu_tnd` 与 `sdpa_seg` 都支持 cross-attn 的 **q/kv 段长不等**（Anima 的 visual↔text 必需）。

**本地已验证**（`tests/test_anima_packed_attn_backends.py`，CPU）：

- `sdpa_seg` 与稠密 bool 块对角 mask 的前向/反向逐元素一致（<1e-5）
- 段间零泄漏：换掉别段的 k/v，本段输出逐 bit 不变
- 整条 `forward_packed_navit` 端到端：**G 张图打包的输出与每张图单独打包完全相同（max_abs_diff = 0.0）**，
  覆盖块对角 self/cross attention + per-image AdaLN(`mod_index`) + 每图独立 RoPE

**未验证**：`npu_tnd` 需要真机（本地无 `torch_npu`，单测自动 skip）；`xformers` 路径本地无法回归
（无 CUDA/xformers，相关单测全部 skip）——但该分支代码未改动，只是加了前置的后端判断。

## 3. 上机第一件事：先侦察，再跑探针

```bash
bash tools/npu_recon.sh            # 只读侦察，不装任何东西
python tools/npu_probe.py --json /tmp/npu_probe.json
```

`npu_recon.sh` 回答的是决定后面所有事的问题：CPU 架构（决定 torch 装法）、CANN 实际版本
（决定 torch 版本上限）、宿主 driver 版本（决定 CANN 能不能升）、有没有 torch/torch_npu、
site-packages 是否会随镜像提交、网络能不能拉代码。

探针跑完会直接打一行 **NaViT 路径裁决**，把结果翻译成该往 yaml 里写什么。

它不加载任何底模、几十秒跑完，逐项实测并打汇总表：torch/torch_npu/CANN 版本、`npu-smi`、
bf16/fp32 matmul 数值、`autocast("npu")`、SDPA（无 mask / bool mask / additive mask，含 backward
与峰值显存）、**SDPA key-padding mask 的语义与极性**（见下）、
`torch_npu.npu_fusion_attention`、`torch.fft`、Conv/GroupNorm/SiLU、
`torch.utils.checkpoint`、AdamW 单步、Event 计时、显存 API、bf16 粗略 TFLOPS，以及依赖包可用性。

**FAIL 项就是这台机器的真实限制**，配方必须绕开它，不要靠猜。

**mask 极性探针（`SDPA key-padding mask 语义/极性`）单独说一句**：另外三条 SDPA 用例是全 True /
全 0 的 mask，只测形状能否被接受和快不快，**对极性完全不敏感**。而昇腾 `aclnnFlashAttentionScore`
的 `atten_mask` 约定是 `True=屏蔽`，与 PyTorch SDPA 的 `True=保留` 相反（转换由 torch_npu 负责）。
若这层没做对，训练会照跑、loss 照降、不报任何错——只是文本条件变成"只看 padding"。该探针用真实
padding 的 mask 与 fp64 CPU 参考对拍，并同时算一份**反转极性**的参考：哪一份更接近就给出裁决。
配套的 `SDPA 广播 key-padding mask` 用例记录未展开的 `[B,1,1,Skv]` 是否仍被拒——哪天不再被拒且
数值正确，§2.2 的展开就可以整体去掉。

⚠ **昇腾的算子错误状态是粘滞的（sticky），一次失败调用会污染同进程后续的一切**。
2026-08-14 真机实测：广播 mask 用例在主进程里触发 `FlashAttentionScore` tiling 失败后，
同一次运行的后续探测**全部失真**——TND self/cross 的 `rel_err` 从 `2.2e-03` 跳到 `1.619e-01`
且两条一模一样、三个梯度范数完全相等、`checkpoint` 的 `grad_norm` 变成 `1.0000`、
AdamW 的 `‖Δp‖` 报出**负数**（范数不可能为负）、bf16 吞吐虚高一倍（275→545 TFLOPS）、
`Event` 计时 FAIL 并在报错里回吐同一条 `atten_mask` inner error。
因此该用例现在 fork 子进程执行，紧跟一条 `设备健康复检`（重算开头那个 bf16 matmul）确认隔离生效。
**推论适用于训练本身**：昇腾上遇到算子级报错，不要指望"接住异常继续跑"——进程状态已不可信，
必须重启进程。

## 4. 还没做的（诚实清单）

- 未在真机跑过任何一步训练；步时/显存/收敛全部未知。
- `npu_tnd` 后端的真机可用性未验证（探针的 TND 项就是为它准备的）。
- 分布式（多卡 HCCL）未接线；当前只考虑单卡。
- `trainer/quant.py` 全部走 CUDA 专用路径，NPU 上由守卫拦掉，未做适配。
- ~~文本编码器转换未用真实文件验证~~ → **已验证，见 §5.1。**

## 4.1 NPU 侧顺手修掉的一个真 bug

`RMSNorm.forward` 原本是 `@torch.autocast('cuda', dtype=torch.float32)` 装饰器
（`models/anima_modeling_core.py`、`models/cosmos_predict2_modeling.py` 各一处）。设备串写死在
**类定义期**，在昇腾上等于给 npu 张量开了个 cuda autocast 区域，完全不生效——`output * self.weight`
会跟着外层 bf16 autocast 走，而不是原意的 fp32。已改成按张量实际设备取 `device_type`。
CUDA 上 `device_type == 'cuda'`，与原装饰器等价；CPU 上原来就无效果（cuda autocast 不作用于 cpu
张量），新实现显式跳过，保持无效果。

**补一刀（真机日志实测）**：改成 `torch.autocast('npu', dtype=torch.float32)` 之后，昇腾真机
训练日志里会出现

```
UserWarning: In npu autocast, but the target dtype is not supported. Disabling autocast.
 npu Autocast only supports dtypes of torch.float16, torch.bfloat16 currently.
```

即 torch_npu 的 autocast **只支持 fp16/bf16**，传 fp32 会走 `torch/amp/autocast_mode.py` 的
`if enabled and self.fast_dtype not in supported_dtype` 分支：打一条 warning，然后把 `enabled`
直接置 False。所以昇腾上这个"fp32 区域"**从来没有真的按 fp32 autocast 跑过**，一直是
enabled=False——又一处靠 warning 表达的静默降级（Python warning 默认每个位置只报一次，很容易被
淹没在日志里）。

已把 npu 分支显式改成 `torch.autocast('npu', enabled=False)`：

- **与 torch_npu 当前实际行为逐位一致**，不是行为变更；
- 不再依赖一条会被吞掉的警告，也不会在 torch_npu 将来支持 fp32 autocast 时行为突变；
- CUDA 分支保持 `dtype=torch.float32` 原样，逐字节不动。

为什么 `enabled=False` 在这个唯一用途上和 fp32 autocast 等价：`RMSNorm.forward` 的区域里只有
`pow/mean/rsqrt/mul` 这些**非 autocast 算子**（autocast 只改 matmul/conv 等白名单算子的 dtype），
`x` 已由 `.float()` 显式提到 fp32，`output * self.weight` 的 dtype 由 type promotion
（bf16 × fp32 → fp32）决定。**若以后往这个区域里加 matmul 之类的白名单算子，两条路就不再等价，
必须重新裁决。**

顺带明确一句归因边界：这处降级**不足以解释**昇腾 run 训出的 LoRA 强度偏高（见 §4.2 的实验设计）——
按上面的分析它在 RMSNorm 上是数值中性的。修它的理由是消除静默降级本身，不是把它当作那个现象的根因。

## 4.2 待裁决：昇腾 run 训出的 LoRA 强度偏高（现象已确认，根因未定）

**客观现象**（本地 checkpoint 取证，两份成品都是同一套 C12 配方）：

| | ‖w1‖ 均值 ep1→末 | ‖w2‖ 均值 ep1→末 | ‖ΔW‖_F 轨迹 | cross_attn 能量占比 ep1→末 |
|---|---|---|---|---|
| 昇腾 / jima（206 张） | 0.376 → **0.428（升）** | 6.17 → 14.18 | 41.4 单调涨到 **113.2** | 25.8% → 31.5% |
| CUDA / villainchin | 0.340 → **0.155（降）** | 5.85 → 11.91 | 峰值 41.9 后回落锁定 **35.7** | 23.5% → **51.0%** |

`‖w2‖` 两边增长曲线几乎重合，差别全在 LoKr 的 w1（每模块 4×4 的总增益因子）走向相反：
CUDA 上 w1 收缩抵消 w2 的增长（净强度稳定），昇腾上两者同向 → 乘积膨胀到 3.2 倍。
方向动力学两边同构（相邻 epoch 的 ΔW 余弦 0.93~0.98）、无 NaN、grad_norm 0.002~0.005
从未触发 clip —— **不是"训坏了"，是强度失控**。

**混杂变量（三个，尚未拆开）**：① 数据集不同；② `navit_token_budget` 65536 vs 131072；
③ 平台/注意力后端 npu+npu_tnd vs cuda+xformers。**现有证据不足以归罪昇腾平台**，
本文档 §4.1 那处 autocast 降级按其语义分析是数值中性的，也不解释这个现象。

**裁决实验**：`config/train_npu_c12_ab_dataset.yaml`（昇腾 + villainchin + budget 65536，
与已跑完的 jima run **只差数据集**），跑 3 个 epoch 用
`tools/lokr_strength_probe.py` 看 ‖w1‖ 走向即可分出"数据集"与"平台/预算"。
判读方法、参照曲线与 fallback 都写在那个 yaml 的头部。

## 5. OpenI/启智 侧的事实（2026-08 查自平台帮助文档与镜像列表）

**可用镜像**：在 `/explore/images` 用「昇腾NPU + PyTorch」筛选，**全站公开镜像只有 2 个**：

| 镜像 | 框架 | Python | CANN | 用量 |
|---|---|---|---|---|
| `torch-npu-cann8-debug` | PyTorch 2.1.0 | 3.9 | 8.0.RC1 | 10538 |
| `openmind_ms_v8-notebook` | PyTorch 2.1.0 | 3.8 | 8.0.RC3 | 1062（`conda activate openmind`） |

标注为 MindSpore 但描述写「包含多个 conda 环境」的几个（`mindspeed-llm` CANN 8.0.RC3 /
`cann8_3_rc1_nnal` CANN 8.3.RC1 Py3.11「只能用于 D910B」）**可能**另含 torch-npu 环境
——未验证，上机 `conda env list` 才算数。

除这两个之外，另有第三方管理员镜像（`/explore/images` 能搜到），例如
`cann8.2.rc2-ms2.7-py3.11-910b`（CANN 8.2.RC2 / MindSpore 2.7 / Python 3.11）——CANN 档次
高得多，但**镜像名里的 `ms` 是 MindSpore，不是 torch**，torch/torch_npu 大概率要自己装
（`npu_setup_image.sh` 会按 CANN 自动选版本）。是否真含可用 torch 环境，上机 `conda env list` 才算数。

**低版本 torch 的影响**：`F.scaled_dot_product_attention(enable_gqa=)` 需要 torch≥2.5，但
`models/krea2_modeling.py:146` 已有版本探测与手动展开 KV 头的 fallback（数值相同），
不是阻塞项。为兼容 Python 3.9，仓库里 4 处 PEP 604 注解所在文件已补
`from __future__ import annotations`（行为中立）：`models/anima_modeling_core.py`、
`models/cosmos_predict2_modeling.py`、`utils/caption_utils.py`。全量 AST 扫描
（49 个文件）未发现其余 3.10+ 专属语法。

**自定义镜像**：支持。流程 = 在**运行中**的调试任务里装好环境 → 调试任务列表 →「更多 →
提交镜像」。限制：镜像大小 鹏城计算I ≤20GB / 其他算力中心 ≤40GB；`/tmp/code` 与
`/tmp/dataset` 不入镜像；**NPU 环境下 `/home/ma-user/work` 不入镜像**；提交期间任务转
WAITING 暂不可用。

**调试任务时长**：文档写明**默认 4 小时**上限，可手动停止。JupyterLab 工作区在 `/tmp/code`，
任务不被回收时内容保留（用户实测：重启环境后仓库仍在）。⚠ 但**重启任务会回到基础镜像**——
自定义镜像要在「新建调试任务」时选（用户实测）。

**监控**：平台不暴露额外端口，仓库内置的 `train_monitor.py`（6006，纯内存）看不到。
用 `wandb_enabled: true`（`trainer/wandb_logger.py`，opt-in）把指标外推；到 api.wandb.ai
的连通性未验证，首跑建议 `wandb_mode: offline` 再 `wandb sync`。

**解释器不在 PATH**：`cann8.2.rc2-ms2.7-py3.11-910b` 镜像里 `pip` / `python` 都不在 PATH，
要用绝对路径 `/usr/local/python3.11.13/bin/python -m pip ...`（用户实测 `pip: command not found`）。

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
的 tokenizer 齐但 `model.safetensors` 只有 135 字节（LFS 指针，非权重）。

### 5.0 按单文件拉底模（不要挂载整个模型）

挂载整个 `FoundationModel/Anima` 会拖慢调试任务启动（用户实测：新建工作区时卡住）。
平台的模型文件可以**按单个文件**取，公开模型**无需登录**。以下均为实测（2026-08-13，匿名请求）：

```
GET https://openi.pcl.ac.cn/api/v1/aimodel/file/meta?aimodel_name=<owner/model>&file_name=<含子目录的路径>&parent_dir=
GET https://openi.pcl.ac.cn/api/v1/aimodel/file?...（同参数）→ 301 到鹏城 OBS 签名直链，支持 Range
```

`FoundationModel/Anima` 三件套的精确 `file_name` 与字节数：

| 文件 | file_name | 字节 |
|---|---|---|
| DiT | `split_files/diffusion_models/anima-base-v1.0.safetensors` | 4182218328 |
| TE | `split_files/text_encoders/qwen_3_06b_base.safetensors` | 1192135096 |
| VAE | `split_files/vae/qwen_image_vae.safetensors` | 253806246 |

两种取法（`openi` CLI 的 `-f` 支持逗号分隔多文件；curl 直连支持 `-C -` 续传）：

```bash
openi model download FoundationModel/Anima -f "split_files/vae/qwen_image_vae.safetensors" -d ./anima_models
curl -L -C - -o vae.safetensors "https://openi.pcl.ac.cn/api/v1/aimodel/file?aimodel_name=FoundationModel/Anima&file_name=split_files/vae/qwen_image_vae.safetensors&parent_dir="
```

数据集同理：`openi dataset upload/download`（上传支持文件夹，保留子目录）。**注意数据集
归属项目仓库**——私有仓下的数据集不进公开列表（文档口径，未实建验证）。

### 5.1 文本编码器转换：已用真实文件验证

`tools/convert_comfy_te_to_hf.py` 对平台那份 `qwen_3_06b_base.safetensors`（与 ComfyUI 生态
分发的是同一文件，字节数都是 1192135096）实测：**键名变换 = identity**（ComfyUI 单文件的
键名已经就是 HF 口径，不需要加/去前缀）、期望键 310 / 源键 310、**形状 310/310 全部一致**；
转出目录用 `AutoModelForCausalLM.from_pretrained` 加载成功，参数量 596,049,920。

因此机上现场转即可（几十秒），**不需要在本地转好再上传**：

用 `tools/convert_comfy_te_to_hf.py` 转：

```bash
python tools/convert_comfy_te_to_hf.py \
    --src /path/to/qwen_3_06b_base.safetensors \
    --out models/text_encoders/Qwen3-0.6B-Base-full --dry-run   # 先只校验键名/形状
```

它不硬编码键名表：期望键集从 `config.json` 反推（meta device 实例化取 state_dict），
前缀差异只用**整体统一变换**修复（要么全对要么全错，可验证），逐键形状全部校验，
配不上就打出两边的键样本并拒绝产出。写出走流式（峰值内存 ≈ 最大单个张量）。

## 6. 镜像装配

用 `tools/npu_setup_image.sh`（在调试任务终端里跑）：

```bash
bash tools/npu_setup_image.sh --check     # 只体检
bash tools/npu_setup_image.sh             # 装必需依赖 + 跑探针
```

**版本决策：真正的硬约束是 CANN，不是 Python。**

查官方版本配套表（`Ascend/pytorch` 的 `COMPATIBILITY.md`），Python 是**独立的一根轴**——
同一个 CANN 8.0.RC1 下，torch_npu 2.1.0.post4 官方就支持到 Python 3.11。所以"换 Python 就要
重新对齐 torch 和 CANN"是**不成立**的（本文档早期版本这么写过，是错的）。

真正卡住 torch 版本的是 CANN：

| CANN | 可用的 torch_npu / PyTorch | Python |
|---|---|---|
| 8.0.RC1 | 2.2.0 / 2.1.0.post4 | 3.8–3.10 / 3.8–3.11 |
| 8.0.RC3 | 2.4.0 / 2.3.1.post2 / 2.1.0.post8 | 3.8–3.11 |
| 8.0.0 | 2.4.0.post2 / 2.3.1.post4 / 2.1.0.post10 | 3.8–3.11 |
| 8.1.RC1 | 2.5.1 / 2.4.0.post4 | 3.9–3.11 |
| 8.2.RC1 | 2.6.0 / 2.5.1.post1 | 3.9–3.11 |
| 8.3.RC1 | 2.8.0 / 2.7.1 / 2.6.0.post3 | 3.9–3.11+ |

推论：**选镜像 = 选 CANN = 选 torch 上限**。在旧镜像里升 CANN 受宿主 driver/firmware 版本卡
（容器内改不了驱动），所以正确做法是**直接挑一个自带高版本 CANN 的镜像**，而不是在低版本镜像里
折腾。`tools/npu_setup_image.sh` 已按这张表实现「读 CANN 实际版本 → 选配套 torch」。

**架构决定 torch 的安装源**（最容易废掉环境的一步）：`x86_64` 走默认 PyPI 会拉到 **CUDA 构建**的
torch（几个 GB 且 torch_npu 不认），必须 `--index-url https://download.pytorch.org/whl/cpu`；
`aarch64` 的 PyPI wheel 本来就是 CPU 构建，直接装。脚本已按 `uname -m` 分支。

**最大的环境杀手**：`pip install torchvision`（或任何间接依赖 torch 的包）会从 PyPI
拉 CUDA 版 torch 覆盖掉 torch_npu 配套的那个，环境当场报废。脚本用动态生成的
constraints 钉死 `torch` / `torch-npu` / `numpy<2`（torch 2.1.x 按 numpy 1.x ABI 编译），
并在安装后校验 torch 版本有没有被改动，被改动就拒绝继续。

**真实依赖比 `requirements.txt` 小得多**（AST 扫描运行路径得出）：必需只有
`numpy<2` / `Pillow` / `safetensors` / `transformers>=4.51,<5` / `PyYAML` / `rich` / `einops`。
`transformers>=4.51` 是硬下限（Qwen3 架构支持）。`diffusers` / `accelerate` / `peft` /
`lycoris-lora` / `pytorch-fid` 运行路径未 import，不装。

⚠ **本文档早期版本把 `torchvision` 也列进了"未 import"，那是错的**（真机上跑出
`Missing dependencies: torchvision` 才发现）：`models/cosmos_predict2_modeling.py` 顶层有
`from torchvision import transforms`，而 `trainer/models.py:105` 会动态加载该模块——是运行
路径必经的。它的**唯一**用途是给 `padding_mask` 做一次最近邻 resize，现已改成纯 torch 的
`F.interpolate(mode="nearest")`（本地对 8 组形状/dtype 与 torchvision 逐 bit 对拍一致），
并从 `anima_train.py` 的依赖预检里移除。**昇腾上不要装 torchvision** —— 它会连带把 torch
换成 CUDA 构建。
"唯一用途"是就**默认路径**而言：开 `aux_perceptual_enabled` 时 torchvision 会从两个方向回来
（`lpips` 包自身、以及 `torch.hub.load` 加载的 DINOv2 仓库代码 `models/perceptual/hub/**` 顶层
`from torchvision import transforms`）。所以 `guard_unsupported()` 现在直接拦掉这个开关，
而不是等运行时 ImportError 或让用户去 `pip install torchvision` 把环境废掉。
`bitsandbytes`（`utils/optimizer_utils.py:76`）与 `lpips`（`trainer/aux_losses.py:344`）
都是 try/except 可选导入，昇腾上不装即可。

**这两个 bug（`torchvision` / `sentencepiece`）是同一个根因**：手工维护的 REQUIRED 清单与真实
依赖会漂移，而 AST 扫描看不见 `requires_backends` 这类运行时门控、也看不见动态加载的模块。
上面那份"真实依赖"清单是**经验清单，不是推导结论**——真机报缺包时先查它是不是又漏了一项，
别急着照错误提示 `pip install`（在昇腾上照做一次就可能废掉环境）。

**site-packages 位置陷阱**：NPU 环境下 `/home/ma-user/work` 不入镜像。若解释器的
site-packages 落在那里，装的包提交镜像后全部丢失——脚本第 1 步会检查并报警。
