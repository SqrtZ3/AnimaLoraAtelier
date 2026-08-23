# Anima / Krea2 LoRA on Kaggle TPU v5e-8

在 Kaggle 的免费 TPU（v5e-8，20 小时/周）上训 LoRA / LoKr，**全程不碰 notebook
网页界面**：代码在本地 git 里正常管理，一条命令 push 上去后台跑，跑完把产物拉回来。

Kaggle 只能写 notebook 是**网页工作台**的限制，不是平台的限制 ——
`kernel-metadata.json` 里写 `"kernel_type": "script"` 就能推纯 `.py`。

支持两个模型族：

| | Anima | Krea2 |
|---|---|---|
| 规模 | 2B（28 层 × 2048） | 12.16B（28 层 单流 MMDiT） |
| 并行 | 8 卡纯 DP | FSDP 权重分片（每卡 ~3GB） |
| 文本条件 | `cross` [512,1024]，过 llm_adapter | `txt` [L,12,2560]，Qwen3-VL 12 层堆叠 |
| 实测吞吐 | 16.9k tok/s（MFU 13.4%） | 9.8k tok/s @ 10240/卡 |

核心技术是 **NaViT 块对角打包**：一个 pack 里塞多张异尺寸图，注意力代价是
`Σn_i²` 而不是 `L²`。真机实测跳块提速与理论吻合到 1% 以内（4 段 0.252 vs 理论
0.250），反向也跳块。这条结论是整个路线成立的前提，账目见
[`docs/ARCH_FINDINGS.md`](docs/ARCH_FINDINGS.md)。

---

## 先跑通链路（零配额，5 分钟）

装环境见 [`SETUP.md`](SETUP.md)。然后不带加速器推一次，确认
push / 轮询 / 取回整条链路通：

```powershell
pwsh -File kaggle_run.ps1 -Username <你的 Kaggle 用户名> -Accelerator none -TimeoutSec 900
```

这一步**不消耗 TPU 配额**。链路通了再往下。

## 完整流程（六步）

```
图片 + 同名 .txt 标签
      │
      ① tools/cache_latents.py          图像 → latent npz（torch，需 GPU 更快）
      │  tools/cache_text_features.py   caption → 文本条件 npz（torch）
      ↓
      ② tools/verify_cache.py           上机前校验（纯 numpy，几秒，必做）
      ↓
      ③ tools/kaggle_fast_upload.py     只传 npz 到私有 Kaggle Dataset
      ↓
      ④ anima_train/build_job.py        代码+配置打成单文件 script kernel
      ↓
      ⑤ jax_tpu/run_train.py --plan-only  本地预演（零配额，看三个数）
      ↓
      ⑥ kaggle_run.ps1                  push → 轮询 → 拉回 checkpoint
```

逐步的命令与参数在 [`docs/GUIDE_new_training.md`](docs/GUIDE_new_training.md)。
下面只说每步的要点。

**① 缓存。** TPU 后端只跑 DiT，不做任何编码 —— latent 与文本条件都在本地
用 PyTorch 算好。这是唯一需要 torch 的环节，也是唯一需要 GPU 的环节（CPU 也能跑，
只是慢）。anima 的文本缓存只读底模里的 `llm_adapter`（269MB，不是整个 4.2GB）。

**② 校验。** `verify_cache.py` 把真机上的 fail-fast 判据搬到本地：键名、形状、
flip 配套、`_empty` 配套、multiscale 档位一致、单图 token 不超预算、目录里没有
像素文件。**跳过这步的代价是白烧一轮配额**（push + 挂载 + 起 TPU + 挂掉）。

**③ 上传。** 只传 npz。`kaggle_fast_upload.py` 是并行上传器（官方 CLI 串行，
几百个小 npz 会把带宽跑成锯齿），且默认带两道数据外泄闸门：目录里有图片/标签
直接拒绝，npz 内藏 caption 明文也拒绝。传完等 `datasets status` 变 ready 再推
kernel，否则挂载不到。

**④ 打包。** `build_job.py` 把 `jax_tpu/` 的 16 个模块 base64 进单文件脚本，
运行时写回磁盘再正常 import（**不是**首尾拼接 —— 模块间有命名空间引用，拼接会
静默覆盖同名顶层函数）。脚本带源码 sha，本地跑过的代码与真机逐字节相同。
**每次改了 `jax_tpu/*.py` 或 yaml 都要重跑**（生成物不进 git，过期 = 把旧代码推上去）。

**⑤ 预演。** `--plan-only` 零配额，打印配置摘要、token 分布、打包报告、适配器
参数量。看三个数决定 8 卡用得满不满：**填充率**（线性层算力利用率上界）、
**成步率**（能凑齐 8 个同布局 pack 的比例）、**布局数**（= 全模型编译次数）。

**⑥ 推送。** `-FilePattern` **必须改**：默认值只拉 `*_probe*` 与 `*.log`，
训练产物（`.safetensors` / 优化器状态 `.npz`）会被过滤掉，跑完了却什么也拿不回来。

```powershell
pwsh -File kaggle_run.ps1 -Username <用户名> -Accelerator TpuV5E8 -TimeoutSec 10800 `
    -JobDir anima_train -FilePattern '(.*\.safetensors|.*\.npz|.*\.json|.*\.log)$'
```

## 改了代码怎么验

改 `jax_tpu/` 之后**先跑闸门再推 TPU**。十几个闸门在本仓库**独立可跑**
（不需要 torch、不需要 upstream、不需要任何权重，参考量已随仓库分发）：

```bash
cd jax_tpu/tests
export XLA_FLAGS=--xla_force_host_platform_device_count=8
<jax-python> check_pack_invariants.py      # 打包不变量
<jax-python> check_train_full.py           # 一整套开关同时开
<jax-python> check_train_loop_k2.py        # krea2 FSDP 闭环
# 全清单与分层见 jax_tpu/tests/README.md
```

改 `tools/_vendor/` 之后另有两条（需要 upstream，见下）：
`tools/check_sync.py`（摘录是否跟得上上游）+
`tools/tests/check_cache_parity.py`（产物与上游逐 bit 相同）。

## 目录

| 路径 | 作用 |
|---|---|
| `jax_tpu/` | TPU 后端本体（16 个模块，纯 JAX，无第三方训练框架依赖） |
| `jax_tpu/tests/` | 闸门。哪些独立可跑、哪些要 upstream，见其 README |
| `tools/` | 数据准备：缓存、校验、上传、脱敏 |
| `tools/_vendor/` | 从 upstream 照抄的最小 PyTorch 代码集（见下） |
| `config/` | 两个模型族的训练 yaml 模板 |
| `anima_train/` | 真训练 job：`build_job.py` + `kernel-metadata.json` |
| `*_probe/`、`anima_*/` | 真机探针（架构裁决的证据来源，平时不用跑） |
| `kaggle_run.ps1` | 本地驱动：改 metadata → 看配额 → push → 轮询 → 拉产物 → 再看配额 |
| `docs/` | 格式契约、操作指南、实测账目 |
| `output/<slug>/` | 拉回的产物（驱动脚本自动创建，不进 git） |

## 与上游仓库的关系

本仓库是从 `anima-lora-train`（GPU/NPU/DCU 多后端训练器）里独立出来的 TPU 路线。
**运行时零依赖**：训练、打包、导出、plan-only 都不需要那个仓库。

只有两类事情需要它，都是可选的：

1. **parity 对拍的 dump 侧** —— 证明"JAX 实现 ≡ PyTorch 实现"必须同时看到两边。
   参考量小的几份已随仓库分发，所以多数 check 侧开箱即跑；大的几份要自己 dump。
2. **`tools/_vendor/` 的同步** —— 那里的 PyTorch 代码（VAE、图像 planning、
   文本编码）是从上游**逐字摘录**的，`_vendor/UPSTREAM.md` 记着每个符号的来源、
   行号、sha256 与有意改动的白名单。

两者都通过环境变量指路：

```bash
export ANIMA_UPSTREAM=/path/to/anima-lora-train/AnimaLoraToolkit
```

没设时相关脚本 fail-fast 并说清要什么，不会抛看不懂的 `ModuleNotFoundError`。

**为什么数据缓存留在 PyTorch 侧**：VAE 编码是一次性离线活，GPU 十分钟的事。
在 JAX 里重写一遍 VAE 技术上可行（约 400 行 + 对拍），但本地 jax 通常只有 CPU
后端，上 TPU 又要烧 20h/周 的配额干这件事。收益不抵成本。所以 torch 留在数据
准备侧是有意的设计 —— 但它**自包含**（`_vendor/` 里都有），不需要 clone 别的仓库。

如果你想完全绕开我们的 torch 工具（用自己的 VAE 实现、或复用别处的预处理产物），
看 [`docs/CACHE_FORMAT.md`](docs/CACHE_FORMAT.md)：TPU 侧只认 npz 的键名与形状，
不认产出它的工具。`tools/verify_cache.py` 会告诉你格式对不对。

## 硬约束与已知坑

- **配额 20h/周**（`python -m kaggle quota` 查）。驱动脚本 push 前后各打一次，
  两次一减就是本次消耗。单 session 上限 12h，长训靠 `--save-state-every` +
  `resume_state` 接棒（存完整优化器状态，不只权重）。
- **单图最大 token ≤ 单卡预算**（= `navit_token_budget / 8`）。预算在 TPU 上是
  **全局**语义（8 卡纯 DP、每卡一个 pack），必须被 8 整除。
- **dataset 刚传完有处理窗口**：`datasets status` 变 ready 之前 push 的 kernel
  挂载不到它，白烧 ~1 分钟 TPU。
- **驱动脚本会吃到陈旧 COMPLETE**：同一 kernel 连续 push 时，轮询第一拍可能读到
  上一跑的状态提前退出、拉回旧产物。以 `kernels status` 为准，跑完用 `-PullOnly` 补拉。
- **Windows 三条**（pwsh 7 / `PYTHONUTF8=1` / `MSYS_NO_PATHCONV=1`）见 `SETUP.md`。

## 延伸阅读

- [`SETUP.md`](SETUP.md) —— 两个 Python 环境、权重清单、验一遍装对了
- [`docs/GUIDE_new_training.md`](docs/GUIDE_new_training.md) —— 逐步操作，含 Krea2 差异
- [`docs/CACHE_FORMAT.md`](docs/CACHE_FORMAT.md) —— npz 契约，自己产缓存看这份
- [`docs/ARCH_FINDINGS.md`](docs/ARCH_FINDINGS.md) —— 真机实测与架构裁决（为什么是这个形态）
- [`jax_tpu/tests/README.md`](jax_tpu/tests/README.md) —— 闸门清单与分层
- [`tools/_vendor/UPSTREAM.md`](tools/_vendor/UPSTREAM.md) —— 照抄代码的溯源台账
