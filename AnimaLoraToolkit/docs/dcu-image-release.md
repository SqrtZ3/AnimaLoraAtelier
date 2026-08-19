# Anima LoRA/DCU 训练镜像 —— 公开发布说明

> 本文件是 `anima-lora-dcu` 镜像公开发布时可直接使用的介绍材料（镜像名称 / 简介 /
> 详细介绍三部分）。所有数字均为镜像实机输出或仓库真机验收记录，出处已标注：
> **【实测】** = 在真机/镜像内跑出的输出；**【资料】** = 平台或厂商口径；**【推断】** = 外推。
> 关联文档：`docs/scnet-image-build.md`（构建规范）、`docs/dcu-issue-log.md`（真机问题记录）、
> `docs/hygon-dcu.md`（DCU 适配总览）、`docs/dcu-attn-backend-research.md`（注意力后端调研）、
> `THIRD_PARTY_NOTICES.md`（第三方许可）。

---

## 一、镜像名称

```
anima-lora-dcu
```

- 命名符合平台规则（60 位以内小写字母/数字/点/中划线/下划线），**创建后不可修改，不带版本号**。
- 版本走 tag：本文档对应版本建议 `v0.1.2`（相对 v0.1.1 新增 DAS flash_attn/triton 轮子层、
  code-server IDE 层、fast_model_init，详见「版本记录」）。

## 二、镜像简介

面向海光 DCU（K100-AI / BW 系列）的 Anima 二次元画风 **LoRA / LoKr 训练镜像**。
基于光源官方 jupyterlab-pytorch 基础镜像（Ubuntu 22.04 / DTK 26.04 / Python 3.11 /
PyTorch 2.9.0 DAS 版），预装海光 DAS 预编译的 flash_attn 2.8.3 与 triton 3.5.1 轮子，
激活 SDPA flash 注意力后端，让 NaViT 大 token 训练在 DCU 上开箱可用；内置
JupyterLab 4.6.3 与 code-server 4.133.0（VSCode）双 IDE 及 SSH/sudo 等平台组件，
构建期自检硬门槛 + 运行期验收 7/7 通过，开箱即训。

## 三、详细介绍

### 1. 镜像概述

`anima-lora-dcu` 是把 **Anima 画风 LoRA/LoKr 训练**完整搬到海光 DCU 上的开箱即用环境。
镜像在构建期就把 DCU 平台已知的坑全部解决掉（DTK 环境变量、SDPA 注意力后端缺失、
依赖污染、显存 OOM、构建服务 legacy builder 限制等），用户拿到镜像建实例后不需要
任何额外安装即可开始训练。训练工具包本体（AnimaLoraToolkit）以只读副本固化在
`/opt/anima-lora-train`，支持 YAML 配置、JSON/TXT 双格式打标、实时监控面板、
训练中出图采样，输出兼容 ComfyUI，并额外支持 LoKr、多优化器（SOAP/ADOPT/Lion/
EmoSens/Automagic/Muon 等）、NaViT 变分辨率打包等进阶能力。

### 2. 软硬件环境一览（镜像内实测）

| 项 | 值 | 说明 |
|---|---|---|
| 操作系统 | Ubuntu 22.04 | 基础镜像自带，符合平台兼容列表 |
| Python | 3.11（实测 3.11.9） | 依赖装在系统 python `/usr/local/lib/python3.11/site-packages` |
| PyTorch | `2.9.0+das.opt1.dtk2604` | 海光 DAS 适配版，HIP 后端：`hip=6.3.26093`、`cuda=None`（DTK 约定 `torch.cuda.*` 即 DCU 设备） |
| DTK | 26.04（`/opt/dtk`） | 海光软件开发套件，环境脚本已接入登录 shell |
| flash_attn | `2.8.3+das.opt1.dtk2604.torch290` | 海光 DAS1.8 预编译轮子（torch290/cp311 段，与镜像配套）；模块内 `__version__` 字符串仍为上游 2.6.1，属打包细节，不影响功能 |
| triton | `3.5.1+das.opt1.dtk2604.torch290` | 海光自编轮子（针对自家 HIP 与 LLVM），为 flash_attn import 链与 torch.compile 的前提 |
| transformers | 4.57.6 | — |
| safetensors | 0.7.0 | — |
| einops | 0.8.2 | — |
| numpy | 1.25.0 | ★ 必须保持 1.x：das 版 torch 按 1.x ABI 编译，已用 constraints 钉死 |
| accelerate / wandb | 1.14.0 / 0.28.2 | 日志与实验跟踪（wandb 走镜像内代理脚本出网） |
| omegaconf / rich / PyYAML | 2.3.1 / 15.0.0 / 6.0.3 | 配置与日志 |
| tokenizers / sentencepiece | 0.22.2 / 0.2.2 | Qwen3 文本编码器 |
| JupyterLab | 4.6.3 | 平台 Jupyter IDE（`/opt/conda/bin/jupyter`） |
| code-server（VSCode） | 4.133.0（Code 1.133.0） | 平台 VSCode IDE（`/usr/lib/code-server/bin/code-server`），真机验收平台前端可连 |
| hy-smi / rocm-smi | 可用 | 海光监控工具（hy-smi 实测可输出系统管理接口信息） |

基础镜像与真机规格（scnet 单卡实例，**【实测】**）：卡名 `BW`、arch `gfx936`、
68.7 GB 显存、80 CU。基础镜像的 `AMDGPU_TARGETS=gfx906;gfx926;gfx928;gfx936;gfx938`，
覆盖海光主流 DCU 型号；本文档实测数字均出自 gfx936 这台机器。

### 3. 核心特性

#### 3.1 DAS 预编译注意力轮子：SDPA flash 后端激活（本镜像存在的首要理由）

- **背景**：DAS 版 torch 把无 mask 的 SDPA 派发给**外部 flash-attn 动态库**；基础镜像
  里没有该库时直接 `RuntimeError`（不优雅回退），且 PyPI 上的 flash_attn/triton 全是
  NVIDIA CUDA 版、装不得。上游 triton 在 gfx936 上同时死在 ABI 门与 LLVM codegen 门，
  无版本可用（详见 `dcu-attn-backend-research.md`）。
- **解法**：层 3.5 固化海光光源 DAS1.8 官方预编译轮子
  `flash_attn-2.8.3+das.opt1.dtk2604.torch290`（≈628 MB）与
  `triton-3.5.1+das.opt1.dtk2604.torch290`（≈105 MB），另装 pytest（flash_attn import 链硬依赖）。
- **收益（【实测】）**：SDPA flash 后端被激活，无 mask SDPA 峰值显存 **32.57 → 0.34 GiB**
  （S=16384），数值 `max|Δ|=4.88e-4`；NaViT 大 token 预算（`navit_token_budget=76800`）
  训练实跑通过，无 OOM。

#### 3.2 DCU 专项适配代码（随镜像分发）

- `utils/dcu_compat.py`：`device_str()` 返回 `"cuda"`（DTK 约定）、SDPA 后端按**实测**
  决定开关（不写死"DCU 没有 flash"）、`guard_unsupported` 对 FP8/量化/xformers/
  torch.compile/8-bit 优化器等逐项守卫（能 import 验证就验证，不预先禁止）。
- `tools/dcu_probe.py`：2 分钟体检探针（SDPA 后端、广播 matmul 反向、key-padding mask
  极性、8 卡 all-reduce 带宽），开机先探再训，避免白烧卡时。
- 三处显存安全网：`vae_attn_chunk_tokens`（VAE 缓存阶段 OOM 兜底，query 分块数学恒等）、
  `navit_attn_chunk_tokens`（训练注意力分块 + 每块梯度 checkpoint，backward 逐块重算）、
  `sdpa_seg`（NaViT 段内 query 分块）。均在真机上验证通过。

#### 3.3 启动提速：fast_model_init

2.09B 参数模型默认随机初始化在 DCU 上实测耗时 **301.5 s**（且随即被 checkpoint 权重覆盖）。
镜像内 `fast_model_init` 用 meta device 构造 + `to_empty` 跳过随机初始化：
本地实测构造 **26.9 s → 4.5 s（6.0×）**，真机启动静默停顿 375 秒问题消除，训练实跑通过。

#### 3.4 依赖保护：constraints 钉死不可动包

镜像内**没有 torchvision**，仓库原 requirements 的 `torchvision>=0.15.0` 会让 pip 从
PyPI 拉 CUDA 版 torch 静默替换 das 版（`torch.cuda.is_available()` 变 False、报错全指向
奇怪的地方）。`requirements-dcu.txt` 故意不列 torch 系 + 构建期由
`dcu_gen_constraints.py` 生成 constraints 把 `torch/numpy` 钉死——任何想换它们的包会
**报错**而不是静默替换。镜像内 `/opt/anima-build/` 保留构建期基线留痕（见 §5）。

#### 3.5 单机 8 卡数据并行（代码已就绪，真机未验证）

- 实现方式：**手动梯度 all-reduce**（非原生 DDP），只在累积边界对可训练参数做一次
  RCCL 通信；模型侧零改动，LoKr/DPO/GAF/CSFlow 等进阶功能原样可用。
- `grad_sync_ms / whole_step_ms < 5%` 视为接近线性，>20% 应增大 `grad_accum`。
- **⚠️ 诚实声明：8 卡侧仍是零实测**（验收实例只有 1 张卡），
  配置 `config/train_dcu_k100ai_8card.yaml` 已就绪，建议发布后在 8 卡实例跑
  `bash run_dcu.sh --nproc 8 --probe` 复核 all-reduce 带宽再开训。

#### 3.6 开箱即用的双 IDE 与平台组件

- 平台必需组件（缺了实例起不来）已核对齐全：`/usr/sbin/sshd`、`/usr/bin/sudo`、
  `/opt/conda/bin/jupyter`（软链，pip 升级不动它）。
- JupyterLab 4.6.3（含中文语言包 zh-CN）；VSCode 入口 code-server 4.133.0 装在平台
  约定路径 `/usr/lib/code-server/bin/code-server`，真机验收平台前端可连。
- 不覆盖基础镜像的 `CMD/ENTRYPOINT`（平台按启动路径扫描 IDE，覆盖会导致创建实例失败）。

#### 3.7 环境自愈脚本（zz-scnet-env.sh）

镜像内 `/etc/profile.d/zz-scnet-env.sh` 一次补齐两个平台级坑：① DTK 的 env.sh 不在
`/etc/profile.d`（不 source 则 `import torch` 报 `libgalaxyhip.so.5` 找不到）；
② 容器无直连出网、代理只存在于 jupyter 进程环境（SSH 会话里 pip/git/wandb 全部超时）。
脚本在登录 shell / bashrc 自动生效，代理凭据**优先从正在跑的 jupyter-lab 进程动态读取**
（重启后凭据可能变），读不到才用兜底值；`HF_ENDPOINT` 默认指向 hf-mirror.com。

#### 3.8 构建期自检硬门槛

`tools/dcu_image_selfcheck.py` 在构建末尾作为硬门槛运行（任一不过则 build 失败）：
torch 仍是 HIP 构建（`version.hip` 非空且 `version.cuda` 为空）、numpy 仍是 1.x、
训练必需依赖齐全（含 flash_attn/triton 包存在性）、平台必需组件齐全、环境脚本已安装。
运行期 `--runtime` 模式额外验证：DCU 真的可见、**SDPA flash 后端真的激活**
（`sdpa.flash=True` 且未被仓库关闭）——"构建成功"不等于"验收通过"，发布版本必须用
新镜像建实例跑一次 `--runtime`（当前版本 7/7 通过）。

### 4. 真机实测数据（2026-08-18，scnet / BW / gfx936 / DTK 26.04）

| 项 | 实测值 |
|---|---|
| bf16 matmul 吞吐（fwd+bwd 粗估） | 263 TFLOPS（标称 192 的 137%，该卡非 K100-AI） |
| bf16/fp16/fp32 matmul 数值 | rel_err 2.9e-3 / 3.6e-4 / 4.1e-7（正常） |
| 2D×3D 广播 matmul 反向（LoKr 关键路径） | fwd 1.3e-7 / dA 1.8e-7 / dB 1.1e-7（通过，无需昇腾式绕行） |
| SDPA 无 mask 峰值显存（S=16384） | 装 DAS 轮子后 32.57 → **0.34 GiB**，数值 max\|Δ\|=4.88e-4 |
| NaViT 训练（`navit_token_budget=76800`） | 单卡稳态 ~0.02–0.03 it/s（33–50 s/step），显存 40–60 GB 浮动，**无 OOM** |
| 启动提速 | 模型构造 301.5 s（随机初始化）→ meta 构造跳过，启动停顿消除 |
| VAE 缓存（1712×2432 整图） | 修复前 OOM 63.07 GiB（fp32 N×N 物化），开启 chunk 后通过 |
| key-padding mask 极性 | 与 torch 语义一致（反转极性差 780×） |
| FP8 (`torch._scaled_mm`) | 不支持（要求 ROCm MI300+），配置 `base_quant: none` |
| xformers / bitsandbytes | 无 DCU 轮子，训练路径不依赖，已从依赖清单排除 |

### 5. 构建期留痕（镜像内 `/opt/anima-build/`）

```text
base_pip_freeze.txt             # 层 1：装任何东西之前的 pip 基线（判断"有没有被污染"）
base_torch.txt                  # 层 1：BASE torch= 2.9.0 hip= 6.3.26093 cuda= None
constraints-dcu.txt             # 层 2：生成的钉死清单（torch/numpy 等）
dcu_gen_constraints.py          # 生成脚本（随代码分发）
requirements-dcu.txt            # DCU 依赖清单（故意不列 torch 系）
code-server-4.133.0-linux-amd64.tar.gz   # IDE 分发包（构建上下文带入）
```

代码 commit 留痕：`/opt/anima-lora-train/AnimaLoraToolkit/.image_commit`（构建时写入）。

### 6. 环境依赖清单

镜像内 Python 包共 **209 项**（`python -m pip list --format=freeze`，构建期基线冻结在
`/opt/anima-build/base_pip_freeze.txt`）。关键项见 §2 表格；全量清单见本文附录 A。
发布"环境依赖"一栏可直接贴附录 A。

### 7. 镜像内磁盘布局（实测）

| 路径 | 占用 | 内容 |
|---|---|---|
| `/opt/anima-lora-train` | 9.8 GB | 训练工具包（代码层本身约 45 MB，其余为该实例内数据/输出占用，随镜像分发的是代码副本） |
| `/usr/lib/code-server` | 736 MB | code-server（VSCode）IDE |
| `/usr/local/lib/python3.11/site-packages` | 5.6 GB | Python 依赖（含 DAS 轮子） |

体积按 docker **单层 ≤ 15 GB** 规则分层（依赖/代码/权重各自成层，任一层远小于上限）；
参考：基础镜像 62 层、压缩后 6.54 GB、最大单层 3.57 GB；v0.1.1 成品未压缩 22.95 GB。

### 8. 快速开始

```bash
# 1) 用镜像建实例后，第一件事：运行期验收（发布版本必须 7/7 通过）
. /opt/dtk/env.sh && python /opt/anima-lora-train/AnimaLoraToolkit/tools/dcu_image_selfcheck.py --runtime

# 2) 2 分钟体检探针（先探再训）
bash /opt/anima-lora-train/run_dcu.sh --probe

# 3) 单卡首跑（20 步，证明前向/反向/存盘都通）
bash /opt/anima-lora-train/run_dcu.sh

# 4) 正式训练（极大 epoch 持续训练，随时手动停、从任意 step 的 checkpoint 挑模型）
#    配置：/opt/anima-lora-train/AnimaLoraToolkit/config/train_dcu_k100ai.yaml
```

日常开发建议用平台持久盘上的 git 工作副本（代码随镜像固化的副本是只读参考）。

### 9. 使用注意事项（红线）

1. **绝不 `pip install torch/torchvision`**：会静默把 das 版换成 CUDA 构建（constraints
   会让这种行为报错而非静默，但别主动装）。
2. **不建 venv**：venv 要么放持久盘（不进镜像）要么放系统盘（重启回退丢失），且不带
   `--system-site-packages` 会把镜像的 das torch 整个隔离掉。依赖直接装系统 python。
3. **numpy 必须留在 1.25.0**：das 版 torch 按 1.x ABI 编译（未做 numpy2 对拍，保守不动）。
4. **不覆盖镜像的 CMD/ENTRYPOINT**，别动 `/opt/conda`，别删 sshd/sudo——平台创建实例依赖。
5. **权重默认不进镜像**：平台共享目录每个节点可读；要完全自包含的产物才用
   `--weights` 重新构建（5.6 GB 单层，远低于上限）。
6. **换 DTK 版本 / 换卡型号后先清 MIOpen 缓存**（`~/.cache/miopen`）：陈旧内核缓存会
   让算子**结果出错**（不是变慢），镜像会提示但不会替你删。
7. 重启实例会把 `/` 回退到基础镜像——系统盘改动（如新装依赖）会丢失，除非"保存开发环境"。
8. 监控面板无鉴权：不建议直接暴露公网，云端用 SSH 端口转发访问。

### 10. 第三方组件与许可合规

镜像内分发的主要第三方组件及其许可（完整清单与再分发注意事项见 `THIRD_PARTY_NOTICES.md`）：

| 组件 | 版本（镜像内实测） | 许可 |
|---|---|---|
| flash_attn（海光 DAS 预编译轮子） | 2.8.3+das.opt1.dtk2604.torch290 | BSD-3-Clause（光源 DAS1.8 编译分发） |
| triton（海光 DAS 预编译轮子） | 3.5.1+das.opt1.dtk2604.torch290 | MIT（光源 DAS1.8 编译分发） |
| code-server | 4.133.0 | MIT |
| jupyterlab | 4.6.3 | BSD-3-Clause |
| torch（基础镜像自带 das 构建） | 2.9.0+das.opt1.dtk2604 | BSD-3-Clause |
| transformers / accelerate | 4.57.6 / 1.14.0 | Apache-2.0 |
| wandb / einops / safetensors / Pillow 等 | 随 pip freeze | 各包许可（MIT/Apache-2.0/BSD 为主） |

**再分发注意事项**：DAS 预编译轮子由海光光源平台编译提供，公开发布前建议确认光源
平台使用条款是否允许再分发其编译产物；如需，发布说明中注明"包含海光 DAS 编译的
flash_attn/triton，来源 `download.sourcefind.cn:65024`"。镜像内 `/opt/anima-lora-train`
的代码副本按本仓库许可（GPL-3.0，含 ComfyUI 派生部分）分发，与上述第三方组件许可相互独立。

### 11. 版本记录

| 版本 | 内容 | 状态 |
|---|---|---|
| v0.1.0 | 首次构建 | ✗ 弃用：硬门槛因 legacy builder heredoc 失效而空转（"假成功"） |
| v0.1.1 | 硬门槛改为脚本文件；运行期自检 7/7 通过 | ✅ 可用（无 DAS 轮子，SDPA 走 math 兜底） |
| v0.1.2 | + DAS flash_attn/triton 轮子（层 3.5，flash 后端激活）+ code-server/JupyterLab 4.6.3（层 4.5）+ fast_model_init；训练实跑通过 | ✅ 当前版本（本文档对应） |

---

## 附录 A：Python 依赖全量清单（pip freeze，209 项）

```text
accelerate==1.14.0
aiofiles==25.1.0
aiohappyeyeballs==2.6.1
aiohttp==3.13.5
aiohttp-cors==0.8.1
aiosignal==1.4.0
annotated-doc==0.0.4
annotated-types==0.7.0
antlr4-python3-runtime==4.9.3
anyio==4.13.0
argon2-cffi==25.1.0
argon2-cffi-bindings==25.1.0
arrow==1.4.0
asttokens==3.0.1
async-lru==2.3.0
attrs==26.1.0
babel==2.18.0
beautifulsoup4==4.14.3
bleach==6.3.0
blinker==1.9.0
certifi==2026.2.25
cffi==2.0.0
charset-normalizer==3.4.7
click==8.2.1
cloudpickle==3.1.2
cmake==3.29.2
colorful==0.5.8
comm==0.2.3
contourpy==1.3.3
cryptography==46.0.6
cycler==0.12.1
debugpy==1.8.20
decorator==5.2.1
defusedxml==0.7.1
distlib==0.4.0
einops==0.8.2
executing==2.2.1
fastapi==0.135.3
fastjsonschema==2.21.2
filelock==3.25.2
flash_attn==2.8.3+das.opt1.dtk2604.torch290
Flask==3.1.3
fonttools==4.62.1
fqdn==1.5.1
frozenlist==1.8.0
fsspec==2026.3.0
git-lfs==1.6
google-api-core==2.30.2
googleapis-common-protos==1.74.0
google-auth==2.49.1
grpcio==1.80.0
h11==0.16.0
h2==4.3.0
hf-xet==1.4.3
hpack==4.1.0
httpcore==1.0.9
httptools==0.7.1
httpx==0.28.1
huggingface-hub==0.36.0
Hypercorn==0.18.0
hyperframe==6.1.0
hypothesis==5.35.1
idna==3.11
importlib_metadata==8.7.1
iniconfig==2.3.0
ipykernel==7.2.0
ipython==9.10.1
ipython_pygments_lexers==1.1.1
ipywidgets==8.1.8
isoduration==20.11.0
itsdangerous==2.2.0
jedi==0.19.2
Jinja2==3.1.6
json5==0.14.0
jsonpointer==3.1.1
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
jupyter_builder==1.2.2
jupyter_client==8.8.0
jupyter_core==5.9.1
jupyter-events==0.12.0
jupyter_ext_model==1.0.10
jupyter_ext_platform==1.0.14
jupyterlab==4.6.3
jupyterlab-language-pack-zh-CN==4.0.post6
jupyterlab_pygments==0.3.0
jupyterlab_server==2.28.0
jupyterlab_widgets==3.0.16
jupyter-lsp==2.3.1
jupyter_server==2.20.0
jupyter_server_terminals==0.5.4
kiwisolver==1.5.0
lark==1.3.1
markdown-it-py==4.2.0
MarkupSafe==3.0.3
matplotlib==3.10.8
matplotlib-inline==0.2.1
mdurl==0.1.2
mistune==3.2.0
mpmath==1.3.0
msgpack==1.1.2
multidict==6.7.1
nbclient==0.10.4
nbconvert==7.17.1
nbformat==5.10.4
nest-asyncio==1.6.0
networkx==3.6.1
ninja==1.11.1
notebook_shim==0.2.4
numa==1.4.6
numpy==1.25.0
omegaconf==2.3.1
opencensus==0.11.4
opencensus-context==0.1.3
opentelemetry-api==1.44.0
opentelemetry-exporter-prometheus==0.61b0
opentelemetry-proto==1.40.0
opentelemetry-sdk==1.40.0
opentelemetry-semantic-conventions==0.61b0
overrides==7.7.0
packaging==26.0
pandas==3.0.2
pandocfilters==1.5.1
parso==0.8.6
pexpect==4.9.0
pillow==12.2.0
pip==26.0.1
platformdirs==4.9.4
pluggy==1.6.0
priority==2.0.0
prometheus_client==0.24.1
prompt_toolkit==3.0.52
propcache==0.4.1
protobuf==6.33.6
proto-plus==1.27.2
psutil==7.2.2
ptyprocess==0.7.0
pure_eval==0.2.3
pyarrow==23.0.1
pyasn1==0.6.3
pyasn1_modules==0.4.2
pycparser==3.0
pydantic==2.12.5
pydantic_core==2.41.5
Pygments==2.20.0
pyparsing==3.3.2
py-spy==0.4.1
pytest==9.1.1
python-dateutil==2.9.0.post0
python-discovery==1.2.1
python-dotenv==1.2.2
python-json-logger==4.1.0
PyYAML==6.0.3
pyzmq==27.1.0
Quart==0.20.0
ray==2.48.0
referencing==0.37.0
regex==2026.4.4
requests==2.33.1
rfc3339-validator==0.1.4
rfc3986-validator==0.1.1
rfc3987-syntax==1.1.0
rich==15.0.0
rpds-py==0.30.0
safetensors==0.7.0
Send2Trash==2.1.0
sentencepiece==0.2.2
sentry-sdk==2.68.0
setuptools==80.8.0
six==1.17.0
smart_open==7.5.1
sortedcontainers==2.4.0
soupsieve==2.8.3
stack-data==0.6.3
starlette==1.0.0
sympy==1.14.0
tensorboardX==2.6.5
terminado==0.18.1
tinycss2==1.4.0
tokenizers==0.22.2
torch==2.9.0+das.opt1.dtk2604
torchdata==0.8.0
tornado==6.5.5
tqdm==4.67.3
traitlets==5.14.3
transformers==4.57.6
triton==3.5.1+das.opt1.dtk2604.torch290
typing_extensions==4.15.0
typing-inspection==0.4.2
tzdata==2026.1
uri-template==1.3.0
urllib3==2.6.3
uvicorn==0.44.0
uvloop==0.22.1
virtualenv==21.2.0
wandb==0.28.2
watchfiles==1.1.1
wcwidth==0.6.0
webcolors==25.10.0
webencodings==0.5.1
websocket-client==1.9.0
websockets==16.0
Werkzeug==3.1.8
wheel==0.46.3
widgetsnbextension==4.0.15
wrapt==2.1.2
wsproto==1.3.2
yarl==1.23.0
zipp==3.23.0
```
