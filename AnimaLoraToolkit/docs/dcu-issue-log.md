# 海光 DCU 首跑问题记录

记录 2026-08-18 在 scnet DCU 单卡实例（`BW` / `gfx936:sramecc+:xnack-` / 68.7GB / 80 CU，
DTK 26.04 / `torch 2.9.0+das.opt1.dtk2604` / hip 6.3.26093 / python 3.11）上把
Anima LoKr 训练跑通过程中出现的全部问题、各自的证据与当前状态。

**本文只记录事实与判定，不含建议。** 区分三种标注：

- **【实测】** 在真机或本地跑出来的数字 / 报错原文。
- **【推断】** 由实测推出但未直接验证的结论，标注置信度。
- **【未验证】** 尚无证据的空白。

平台级背景事实（DTK env 不自动 source、出网代理只存在于 jupyter 进程环境、重启回退基础
镜像等）见 `docs/hygon-dcu.md` 与 `docs/scnet-image-build.md`，本文不重复。

---

## 目录

| # | 问题 | 状态 |
|---|---|---|
| 1 | VAE latent 缓存阶段 OOM | 已定位，已修，真机已通过缓存阶段 |
| 2 | 启动阶段静默停顿 375 秒 | 已定位，已修，真机未验证 |
| 3 | 训练第一步 backward OOM | 已定位，已修，真机未验证 |
| 4 | SDPA 只剩 math 后端 | 已取证，无解（DAS flash_attn 轮子或可解，见 5） |
| 5 | flash_attn DAS 包获取受阻 | 已定位获取路径，真机未验证 |
| 6 | 上游 Triton 不可用 | 上游路线已判死，DAS 适配版待真机验证 |
| 7 | NaViT 原生分辨率无 per-image token 上限 | 已确认，未处理 |
| 8 | torch 线程数 128 与初始化耗时的关系 | 未裁决 |

---

## 1. VAE latent 缓存阶段 OOM

### 现象【实测】

首次启动在 cache latents 阶段崩溃，训练未开始：

```
[cache] encode OOM at batch=4 (1712x2432)，降级逐张
torch.OutOfMemoryError: HIP out of memory. Tried to allocate 63.07 GiB.
（降级逐张后）                                    Tried to allocate 15.77 GiB.
```

崩溃点：`models/wan/vae2_1.py:248` 的 `F.scaled_dot_product_attention`。

### 定位

`wan/vae2_1.py` 的 `AttentionBlock` 是**单头全局**自注意力，位于 encoder 中间块（3 次
空间下采样之后，即 1/8 分辨率），token 数 `N = (H/8)·(W/8)`，`head_dim` 等于该层通道数
（本仓库配置 `dim=96, dim_mult=[1,2,4,4]` → 最深层 **384**）。

数值对账【实测】：

| 图 | N | fp32 的 N×N | ×4（batch） |
|---|---|---|---|
| 1712×2432 | 65056 | **15.77 GiB** | **63.07 GiB** |
| 2048×2048 | 65536 | 16.00 GiB | 64.00 GiB |
| 848×1216 | 16112 | 0.97 GiB | 3.87 GiB |

与报错的两个数字逐位吻合。

有 flash / mem-efficient SDPA 后端的平台上显存随 N 线性增长；SDPA 落到 math backend 时
逐元素物化 `softmax(q@kᵀ)`，显存 O(N²)。参见问题 4。

`cache_encode_tiled` 已开但未生效：其触发阈值 `cache_encode_max_pixels` 默认约 16.8M px
（4096²），而 1712×2432 = 4.16M px，走的是整图路径。该阈值是按 CUDA 上 VAE 卷积的峰值
显存定的。

### 处置

`vae_attn_chunk_tokens`（config，默认 0 = 关）。按 query 切块，每块对全部 K/V 做 SDPA；
每个 query 的 softmax 归一化域仍是全部 key，逐元素数学恒等。峰值 N² → chunk·N，
chunk=4096 / N=65056 时约 1.0 GiB。

单测 `tests/test_vae_attn_chunk.py`（9 条）：float64 下 `atol=1e-12` 对拍整块 SDPA、
`AttentionBlock` 端到端开关一致、`chunk=0` / `N≤chunk` 零开销回退、非法值 fail-fast、
config 默认关 + YAML 可开；另有一条 spy 测试钉死"切的是 query 维、K/V 每块全量"。

提交 `661721d`。

### 现状【实测】

开启 `vae_attn_chunk_tokens: 4096` 后，缓存阶段在真机上通过。

---

## 2. 启动阶段静默停顿 375 秒

### 现象【实测】

```
13:00:18,537 - INFO - Anima 模型构造: max_img_h=392, ...
13:06:33,969 - INFO - Transformer 权重加载: remap=net., 匹配 685/688 (99.6%), missing=3
```

两行之间 375 秒无任何输出。其后 `13:06:33 → 13:07:24`（51 秒）是 `model.to(device)`。

### 排查过程

- 首个假设"网络存储冷读慢"**被证伪**【实测】：
  `time cat <transformer>.safetensors > /dev/null` → `real 0m1.895s`。
- 在 `load_anima_model` 中加入分段计时后【实测】：
  ```
  Transformer 启动耗时: 构造(CPU 随机初始化) 301.5s + 权重加载 47.1s (torch 线程数=128)
  ```
- 本地对照【实测】：同一 config、同一模型代码，16 线程 CPU 构造 **26.5s**，
  参数量 2.091B。

即 301.5 秒全部花在 2.09B 参数的默认随机初始化（`kaiming_uniform_` 等）上，而这些随机值
随即被 checkpoint 权重整个覆盖。

### 处置

`fast_model_init`（config，默认 false）：`with torch.device("meta"): Anima(**config)`
再 `to_empty(device="cpu")`。本地实测 **26.9s → 4.5s（6.0×）**，余下为 8GB CPU 内存
分配本身。

`to_empty()` 分配的是未初始化内存，配套两道防线：

1. 显式重算两类模块 —— RoPE 派生 buffer（ckpt 里有但 shape 随 `max_img_*` 变，加载时按
   recomputable 丢弃）、以及任何持有 **non-persistent buffer** 的模块（不进 state_dict，
   checkpoint 永远填不到）。为此给 `anima_modeling.py` 与 `anima_modeling_core.py` 两处
   `RotaryEmbedding` 补了 `reset_parameters()`（正常路径不调用）。
2. `_assert_no_uninitialized` fail-fast 对账：`missing ∪ skipped ∪ non-persistent` 必须
   全部落在显式重算集合里，否则 raise。为此 `_load_safetensors_into_model` 的返回值补了
   `"skipped"` 字段。

第 2 道防线在开发过程中**实际抓到过一次遗漏**：`llm_adapter.rotary_emb.inv_freq`。

单测 `tests/test_fast_model_init.py`（8 条）：灌入 checkpoint 后**所有** buffer（含
non-persistent）与正常构造逐位一致；漏掉任一 key 时必 raise。

提交 `44784fa`。

### 现状

真机未验证。

---

## 3. 训练第一步 backward OOM

### 现象【实测】

开启 `vae_attn_chunk_tokens` 后缓存通过，训练第一步崩在 backward：

```
File "anima_modeling_core.py", line 138, in _packed_attention_seg
    o = F.scaled_dot_product_attention(qs, ks, vs)
torch.OutOfMemoryError: HIP out of memory. Tried to allocate 16.00 GiB.
GPU 0 has a total capacity of 63.98 GiB of which 3.11 GiB is free.
Of the allocated memory 49.61 GiB is allocated by PyTorch...
```

调用栈显示位于 `torch.utils.checkpoint` 的 `recompute_fn` 内（block 级梯度检查点的重算）。

### 定位

`16.00 GiB = 16 heads × 16384² × 4 B`。16384 token 对应一张 **2048×2048 原生分辨率**图
（latent 256×256 → patch 128×128）；该尺寸在 BucketReport 里有 7 张。

dtype 为 fp32 是由该等式反解得到的**【推断】**（置信度中等：16.00 GiB 与
`H=16, S=16384, fp32` 精确相符；而 config 中 `attn_force_autocast_dtype=true` 按语义
应为 bf16，二者尚未对上，未进一步排查）。

### 关键实测：光分块不能解决这个问题

本地 GPU 强制 MATH 后端，S=4096 / H=16 / D=128 / bf16【实测】：

| | forward 后仍持有 | 峰值 |
|---|---|---|
| 整块 SDPA | 1.12 GiB | 2.41 GiB |
| query 分块 (c=1024) | **1.30 GiB** | 1.64 GiB |
| query 分块 + 每块 checkpoint | **0.02 GiB** | 0.65 GiB |

math SDPA 把 S×S 的 softmax 结果存进 backward 图，按 query 切块只是把一个大张量拆成若干
小的，加起来仍是 S×S，降的只有瞬时峰值。问题 1 的 VAE 不受此影响，因为 VAE encode 在
`torch.no_grad()` 下运行，没有 backward 图。

### 处置

`navit_attn_chunk_tokens`（config，默认 0 = 关）：`_packed_attention_seg` 段内按 query
分块，**每块再套一层 gradient checkpoint**，backward 时逐块重算；无梯度时（eval / 采样）
不套 checkpoint。开启但 `navit_attn_backend != sdpa_seg` 时构造期 fail-fast。

单测 `tests/test_seg_attn_chunk.py`（9 条）：前向逐元素恒等、**梯度单独对拍**、开分块后
仍与稠密 bool mask 的块对角参考实现一致（段间零泄漏）、只切 query 维、无梯度时不套
checkpoint。既有 packed 注意力回归 51 条通过。

提交 `1c062f9`。

### 现状

真机未验证。额外前向带来的速度代价**未测**。

---

## 4. SDPA 只剩 math 后端

### 证据【实测】

`torch.nn.attention.sdpa_kernel` 四后端逐个试，两种形状（`head_dim=128 / S=4096` 对应
训练主循环，`head_dim=384 / S=8192` 对应 VAE）：

```
Torch was not compiled with memory efficient attention.
Torch was not compiled with cuDNN attention.
FLASH_ATTENTION: No matching libraries found for flash_attn_2_cuda*.so
FLASH_ATTENTION: requires q,k,v ... less than or equal to 256. Got Query.size(-1): 384
MATH: OK   （两种形状均只有 MATH 通过）
```

三类原因彼此不同：

- mem-efficient / cuDNN attention —— **torch 编译期就没进去**。
- flash —— torch 编译期带了分派逻辑，运行期缺 `.so`。
- VAE 的 `head_dim=384` —— 超过 flash 系列 256 的上限，属于形状本身不合格，与库是否
  存在无关。

### math 后端的性能【实测】

S=4096 / H=16 / D=128 / bf16：**11.0 ms、峰值 2.14 GiB**。
按 `2·2·S²·D·H` 计约 137 GFLOP → 约 **12.5 TFLOPS**，对该卡 bf16 实测峰值
263 TFLOPS 的 **5%**，与访存瓶颈的特征一致。

### 现状

无解。仓库侧以问题 1、3 的两个开关兜底。

---

## 5. flash_attn DAS 包获取受阻

### 已确认的事实【实测】

- torch 是标准 DAS 构建：`pip show torch` → `2.9.0+das.opt1.dtk2604`
  （注意 `torch.__version__` 只显示 `2.9.0`，local version 段被截掉，一度导致误判为
  私有构建）。
- 目标产物形如
  `flash_attn-<版本>+das.opt1.dtk2604-cp311-cp311-manylinux_2_28_x86_64.whl`。
- 官方分发站 `https://cancon.hpccube.com:65024`：用户浏览器打不开；本对话的抓取工具报
  `SSL: CERTIFICATE_VERIFY_FAILED / Hostname mismatch`。
- 容器内 pip 源是 `https://pypi.tuna.tsinghua.edu.cn/simple`，即上游 PyPI 镜像；
  `pip index versions flash-attn` 列出的 70 余个版本全部是 NVIDIA CUDA 版。
- 环境中原本没有 flash_attn。

### 更新（2026-08-18，调研）

- 正确入口是 **`https://download.sourcefind.cn:65024`**（scnet 官方 DAS 文档
  `scnet.cn/help/docs/mainsite/ai/appendix/DAS-introduction/` 指向
  `download.sourcefind.cn:65024/4/main/`）；`cancon.hpccube.com` 是 scnet 内网别名，
  证书 hostname mismatch 是内网域名导致的，**公网域名无此问题**。
- 【实测】本地 curl `download.sourcefind.cn:65024/directlink/4/flash_attn/DAS1.8/` 可列目录，
  存在 `flash_attn-2.8.3+das.opt1.dtk2604.torch290-cp311-cp311-manylinux_2_28_x86_64.whl`
  （torch290 段对 2.9.0、cp311 对 py3.11、dtk2604 对 DTK 26.04，全段匹配本机）；
  完整 URL HEAD 200，Content-Length 658,898,543（≈628 MB），证书校验正常（无需 -k）。

### 现状

已定位获取路径，真机未安装未验证。验证步骤见 `docs/dcu-attn-backend-research.md` §5。

---

## 6. 上游 Triton 不可用

### 背景【实测】

- `find /opt/dtk* -maxdepth 4 -iname "*triton*"` 无命中 —— DTK 26.04 不带 Triton。
- `/opt/dtk/llvm/bin/llc -march=amdgcn -mcpu=help` 列出：
  ```
  gfx936  - Select the gfx936 processor.
  gfx938  - Select the gfx938 processor.
  gfx936-insts / gfx938-insts  - Additional instructions for GFX936/GFX938.
  ```
  上游 LLVM 的 AMDGPU target 列表中没有 gfx936 / gfx938。
- 上游 `triton 3.7.1` 可从清华源下载（197.7 MB）。

### 现象【实测】

装上 `triton 3.7.1` 后运行最简 kernel：

```
RuntimeError: cannot get address for 'hipDrvLaunchKernelEx' from libamdhip64.so
```

调用栈位于 `triton/backends/amd/driver.py` 的 `HIPUtils()` 初始化。

### 判定

失败发生在 **HIP 驱动 ABI 这一关**（海光的 `libamdhip64.so` 没有该符号），Triton 尚未
开始编译任何内容。因此"上游 LLVM 能否给 gfx936 出码"这一问**未被验证** —— 这是两道
独立的门，第一道确认关着，第二道空白。

### 未验证的路径

- 老版 Triton（3.2 及以前使用 `hipModuleLaunchKernel`）能否过第一道门。**【未验证】**
- DTK LLVM 的版本是否接近 Triton pin 的 LLVM commit；`/opt/dtk/llvm/include/mlir`
  是否存在。**【未验证】**
- CK 后端 flash-attn 源码编译：CK 的 fmha instance 按 gfx90a / gfx942 / gfx950 逐架构
  生成，gfx936 不在列。**【推断】**，置信度中等偏高，未尝试。

### 更新（2026-08-18，调研）

- 【实测】核对 triton v3.2.0 源码（`third_party/amd/backend/driver.{py,c}`）：AMD 后端只用
  `hipModuleLaunchKernel` / `hipModuleLoadDataEx` / `hipModuleGetFunction` /
  `hipFuncGetAttribute` / `hipPointerGetAttribute` 等老符号，**不含** `hipDrvLaunchKernelEx`
  —— 3.2 及更早可以过第一道门。
- 【实测】llvm.org AMDGPUUsage（22.0.0git 时代）的 AMDGPU 处理器表**没有 gfx936 / gfx938**：
  上游 LLVM 出不了 gfx936 的码 —— 对上游 triton（自带捆绑 LLVM）而言，第二道门是关着的。
- 【实测】DAS1.8 目录存在海光自编
  `triton-3.3.0+das.opt1.dtk2604.torch290-cp311-...whl`（107 MB）与
  `triton-3.5.1+das.opt1.dtk2604.torch290-cp311-...whl`（105 MB），HEAD 均 200 ——
  针对自家 HIP/LLVM 编的版本，预期两道门都通。

### 现状

上游 triton 路线判死（无版本能同时过两道门）。DAS 适配版（3.3.0 / 3.5.1）待真机验证。
验证步骤见 `docs/dcu-attn-backend-research.md` §5。

---

## 7. NaViT 原生分辨率无 per-image token 上限

### 事实

`navit_native_resolution: true` 下，每张图按原生尺寸（仅受 VAE + 16px 对齐约束）进包，
**不做 per-image token 数上限**；唯一边界是模型 RoPE 上限 `max_img_h` / `max_img_w`。
`navit_token_budget` 约束的是整个 pack 的 ΣN，不约束单张图。
`navit_multiscale_token_ladder` 只**追加**降采样副本，原图仍按原生尺寸进包。

因此当前数据集（含 7 张 2048×2048）必然产生 16384 token 的单段。这是问题 3 的直接成因。

### 现状

未处理。仓库中不存在"单图 token 上限"这一配置项。

---

## 8. torch 线程数 128 与初始化耗时的关系

### 事实【实测】

DCU 节点 `torch.get_num_threads() = 128`，2.09B 参数初始化 301.5s；
本地 16 线程同 config 26.5s。折算下来每核吞吐相差约一个数量级。

### 现状

**未裁决**：没有区分"该节点 CPU 单核慢"与"128 线程超订导致争用"。问题 2 的 meta device
方案绕开了这段代码，未再继续排查。

---

## 已被证伪的假设（留档，避免重走）

| 假设 | 证伪依据 |
|---|---|
| 启动慢是网络存储冷读 | `time cat` 权重文件 = 1.9s |
| `torch.__version__` 干净 → 私有构建、DAS 包可能不兼容 | `pip show torch` = `2.9.0+das.opt1.dtk2604`，标准 DAS 构建 |
| VAE 的 O(N²) 换个 SDPA 后端就能解决 | head_dim=384 超过 flash 系列 256 上限，官方 warning 直接报出 |
| 训练路径照搬 VAE 的 query 分块即可 | 本地强制 MATH 实测：分块后持有量 1.12 → 1.30 GiB，未下降 |
| FlexAttention 是"零编译"路径 | 它依赖 Triton，而环境中没有 Triton |
| 上游 triton 会因 LLVM 不认 gfx936 而失败 | 实际失败在更早的 HIP 驱动 ABI；LLVM 那一问仍未验证 |

---

## 尚无数据的空白

- 真机训练的 it/s 与稳态显存占用（三个开关全开后）。
- `navit_attn_chunk_tokens` 的额外前向带来的速度代价。
- `fast_model_init` 在 DCU 上的实际提速。
- 问题 3 中注意力 dtype 为何是 fp32，而非 `attn_force_autocast_dtype=true` 所期望的 bf16。
- 8 卡的 all-reduce 带宽与拓扑（该实例只有 1 张卡）。

---

## 相关提交与产物

| 提交 | 内容 |
|---|---|
| `661721d` | VAE 自注意力 query 分块（问题 1） |
| `44784fa` | meta device 构造 transformer（问题 2） |
| `1c062f9` | sdpa_seg 段内 query 分块 + 每块 checkpoint（问题 3） |
| `5b55798` | 注意力后端探针 `tools/dcu_attn_backend_probe.py`（问题 4、6） |
| （本次） | 探针新增 [0b] HIP 符号 / [0c] aotriton / [5] SDPA FLASH 后端三项检测；调研文档 `docs/dcu-attn-backend-research.md`（问题 5、6） |

新增的三个 config 键（`vae_attn_chunk_tokens` / `fast_model_init` /
`navit_attn_chunk_tokens`）全部 opt-in、默认关，关闭时与改动前是同一条代码路径。
