# DCU（gfx936 / DTK 26.04）注意力后端调研

> 2026-08-18。本文回答 dcu-issue-log.md 问题 4/5/6 的"还能用什么后端"。
> 证据等级沿用 issue log 的标注：**【实测】** = 跑过/查到原始出处；**【推断】** = 由实测外推；
> **【未验证】** = 空白。本文给结论与验证步骤，**结论都需在真机复跑探针后转正**。
>
> 配套脚本：`tools/dcu_attn_backend_probe.py`（新增 [0b] HIP 符号、[0c] aotriton、[5] SDPA FLASH 后端三项检测）。

## 1. 结论（TL;DR）

| # | 后端 | 判定 | 一句话依据 |
|---|---|---|---|
| 1 | **flash_attn 2.8.3 DAS 轮子** | ✅ 首选 | DAS1.8 官方轮子已发布，与 torch 2.9.0+das.opt1.dtk2604 / py3.11 段段匹配，公网可下载（curl 200 已验证） |
| 2 | **triton 3.3.0 / 3.5.1 DAS 轮子** | ✅ 次选 | 海光自编（针对自家 HIP/LLVM），预期同时过 ABI 门与 gfx936 codegen 门 → 解锁 FlexAttention / torch.compile |
| 3 | xformers | ❌ 排除 | DAS1.8 只有 torch251 版（0.0.33），无 torch290 版；且 torch 的 mem_efficient SDPA 后端是编译期定死的 |
| 4 | aotriton / CK | ❌ 排除 | 上游 arch 实例列表无 gfx936；DAS torch 的 flash 走的是外部 dlopen flash_attn，不是 aotriton |
| 5 | 上游 triton（任何版本） | ❌ 排除 | 3.7.1 死 ABI 门（hipDrvLaunchKernelEx 缺失）；≤3.2 能过 ABI 门但必死 codegen 门（上游 LLVM 无 gfx936） |
| 6 | MIOpen attention | ❌ 排除 | torch SDPA 不派发给 MIOpen |

## 2. 机器与软件栈（背景，来自 `docs/probe_results/dcu_probe_bw_gfx936_dtk2604.log`）

- 卡：`BW` / `gfx936:sramecc+:xnack-` / 68.7 GB / 80 CU；torch `2.9.0+das.opt1.dtk2604`，hip `6.3.26093`。
- SDPA 现状【实测】：flash=NO（`RuntimeError: No matching libraries found for flash_attn_2_cuda*.so`）、
  mem_efficient=NO（编译期没编入）、math=OK。bf16 matmul 实测 263 TFLOPS。
- 训练痛点：NaViT 原生分辨率单段最大 16384 token，math 后端物化 S×S → 问题 3 的 backward OOM，
  已用 `navit_attn_chunk_tokens`（分块+每块 checkpoint）兜底，但**兜底有额外前向、速度未测**。

## 3. 两条可行路的机理与证据

### 3.1 flash_attn 2.8.3 DAS 轮子（首选）

**机理**：DAS 版 torch 把无 mask 的 SDPA 派发给一个**外部的 flash-attn 动态库**
（报错原文 `No matching libraries found for flash_attn_2_cuda*.so` 是直接证据；
upstream torch 2.9 的 HIP flash 是 CK/AOTriton 实现，`aten/src/ATen/native/transformers/hip/flash_attn/`
下是 ck/ 与 aot/ 两套源码，DAS 版显然改成了 dlopen 外部库【推断，置信度高——报错串与
"基础镜像里 find / -name '*flash_attn*' 为空"的事实吻合】）。
装上海光适配的 flash_attn 轮子后，该后端应被激活。`dcu_compat.configure_sdpa_backends()` 按实测
决定开关（能跑就不动），仓库代码无需改动即可受益。

**证据【实测】**：
- scnet 官方 DAS 文档（https://www.scnet.cn/help/docs/mainsite/ai/appendix/DAS-introduction/）
  指明已发布组件从 `https://download.sourcefind.cn:65024/4/main/` 获取（注意：**不是**
  issue log 里记的 `cancon.hpccube.com:65024`，后者是 scnet 内网别名、证书不匹配）。
- `https://download.sourcefind.cn:65024/directlink/4/flash_attn/DAS1.8/` 目录【实测：本地 curl 可列】：
  ```
  flash_attn-2.8.3+das.opt1.dtk2604.torch251-cp311-...whl   （torch 2.5.1 版）
  flash_attn-2.8.3+das.opt1.dtk2604.torch271-cp311-...whl   （torch 2.7.1 版）
  flash_attn-2.8.3+das.opt1.dtk2604.torch290-cp311-cp311-manylinux_2_28_x86_64.whl   ← 本机
  ```
  `torch290` 段与 torch 2.9.0 对应，`cp311` 与 python 3.11 对应，`dtk2604` 与 DTK 26.04 对应。
- 完整 URL 本地 curl HEAD → **200 OK，Content-Length 658,898,543（≈628 MB）**，证书正常（无需 -k）。
- 限制：flash 系列 head_dim ≤ 256。transformer 的 128 满足；VAE 的 384 不满足 →
  VAE 继续走 `vae_attn_chunk_tokens` 兜底（VAE encode 在 no_grad 下，无 backward 图，无碍）。

**风险 / 未验证**：
- DAS torch 对 flash_attn 动态库的具体调用约定（符号名/注册方式）装上前无法确认，
  装上后跑探针 [5] 一锤定音。
- 628 MB 下载在代理下可能慢；`pip install` 不会自动校验 torch290 段，装错版本会
  undefined symbol 崩在 import。

### 3.2 triton 3.3.0 / 3.5.1 DAS 轮子（次选，解锁 FlexAttention 与 torch.compile）

**机理**：上游 triton 在 gfx936 上有**两道门**：
- 第 1 道（HIP 驱动 ABI）【实测】：triton 3.7.1 在 `HIPUtils()` 初始化报
  `cannot get address for 'hipDrvLaunchKernelEx' from libamdhip64.so` —— DTK 26.04 的
  libamdhip64（hip 6.3.26093）没有该符号。HIP 6.4 才引入（ROCm 7.0.1 release notes 列为新 API）；
  LlamaFactory issue #10511 记录同病（ROCm 6.3 上 triton 3.7.0 报同样错误，workaround 是
  triton 3.5.1）。核对 triton v3.2.0 源码（`third_party/amd/backend/driver.{py,c}`）：
  其 AMD 后端只用 `hipModuleLaunchKernel` / `hipModuleLoadDataEx` / `hipModuleGetFunction` /
  `hipFuncGetAttribute` / `hipPointerGetAttribute` 等**老符号**，不含 hipDrvLaunchKernelEx。
- 第 2 道（LLVM codegen）【实测】：上游 LLVM（llvm.org AMDGPUUsage，22.0.0git 时代）
  的 AMDGPU 处理器表**没有 gfx936 / gfx938** —— 这两个 arch 是海光在 DTK 的 LLVM fork
  里私有加的。triton 轮子自带捆绑 LLVM（上游 commit），编不出 gfx936 的码。

结论：上游 triton **没有版本能同时过两道门**（≤3.2 过 1 挂 2；≥3.6 挂 1）。
DAS 自编的 triton 轮子（针对自家 HIP 与自家 LLVM 编）预期两道门都通，是唯一正路。

**证据【实测】**：
- `https://download.sourcefind.cn:65024/directlink/4/triton/DAS1.8/` 目录：
  ```
  triton-3.3.0+das.opt1.dtk2604.torch290-cp311-cp311-manylinux_2_28_x86_64.whl   （107 MB）
  triton-3.5.1+das.opt1.dtk2604.torch290-cp311-cp311-manylinux_2_28_x86_64.whl   （105 MB）
  ```
  完整 URL 本地 curl HEAD → 200 OK。两个版本都备着，inductor 兼容性哪个对用哪个。

**预期收益**：
- FlexAttention（torch 自带，走 Inductor→Triton）：原生支持 block-diagonal（document）
  mask，BlockMask 会**跳过整块全 mask 区域**——正是 NaViT 打包路径的语义，可能是
  `sdpa_seg` 之上的最终形态（sdpa_seg 是逐段 dense，全 mask 块也照样算）。
- `torch_compile: true` 顺带解锁（dcu_compat 的守卫是"能 import triton 就不拦"）。

**风险 / 未验证**：
- DAS triton 与 DAS torch 2.9 的 inductor 版本配套关系未验证（torch 2.9 对 triton 有
  版本下限检查；3.3.0 与 3.5.1 两个都试）。
- FlexAttention 的数值/显存/速度在 gfx936 上无先例。

## 4. 已排除 / 证伪的候选

| 候选 | 结论 | 证据 |
|---|---|---|
| xformers | ❌ | DAS1.8 只有 `xformers-0.0.33+das.opt1.dtk2604.torch251`（torch 2.5.1 版），无 torch290 版【实测：目录列表】；且 torch 的 mem_efficient SDPA 后端编译期未编入（"Torch was not compiled with memory efficient attention"），装 xformers 也激活不了 SDPA 的 mem_efficient |
| aotriton | ❌ | DAS torch 的 flash 报错是"找不到外部 flash_attn 库"，不是 aotriton 缺失；上游 aotriton 的实例按 gfx90a/gfx942/gfx1100 等生成，无 gfx936【推断，置信度中高】。探针 [0c] 负责取证（DTK 是否带 aotriton） |
| flash-attn 源码编译（CK 后端） | ❌（备胎） | CK 的 fmha instance 按 arch 生成，gfx936 不在上游列表【推断，置信度中高，未逐一核对 CK 源码】。海光 OpenDAS 有适配源码仓（developer.sourcefind.cn/codes/OpenDAS/flash-attention，分支 aicc-master-dev），**仅在 DAS 轮子不可用时**才考虑 clone 自编（DTK 的 hipcc/dcc 支持 gfx936） |
| 上游 triton | ❌ | 见 §3.2 两道门 |
| MIOpen attention | ❌ | torch SDPA 不派发给 MIOpen；DTK 的 MIOpen 只服务 conv 等路径 |
| HSA_OVERRIDE_GFX_VERSION=9.4.2 + 上游 triton | ⚠️ 不推荐 | 理论上可让 triton 为 gfx942 出码并在 gfx936 上跑（如果 ISA 兼容），但整卡被覆盖为 gfx942 会波及 DTK 原生内核，属于高风险未验证打法，不做 |

## 5. 真机验证步骤（scnet DCU 上，按顺序）

```bash
. /root/private_data/scnet_env.sh          # DTK env + 出网代理
cd /root/private_data

# 5.1 首选：flash_attn（628 MB，耐心等）
curl -O "https://download.sourcefind.cn:65024/directlink/4/flash_attn/DAS1.8/flash_attn-2.8.3+das.opt1.dtk2604.torch290-cp311-cp311-manylinux_2_28_x86_64.whl"
pip install flash_attn-2.8.3+das.opt1.dtk2604.torch290-cp311-cp311-manylinux_2_28_x86_64.whl

# 验证：探针 [5]「SDPA FLASH 后端」应从 FAIL 变 OK；顺带看 16384 的显存/速度
python /opt/anima-lora-train/AnimaLoraToolkit/tools/dcu_attn_backend_probe.py --seq 16384

# 5.2 次选：triton（先 3.5.1，不行再换 3.3.0）
curl -O "https://download.sourcefind.cn:65024/directlink/4/triton/DAS1.8/triton-3.5.1+das.opt1.dtk2604.torch290-cp311-cp311-manylinux_2_28_x86_64.whl"
pip install triton-3.5.1+das.opt1.dtk2604.torch290-cp311-cp311-manylinux_2_28_x86_64.whl

# 验证：探针 [0b] 符号项 + [1] triton 最简 kernel + [3] flex_attention
python /opt/anima-lora-train/AnimaLoraToolkit/tools/dcu_attn_backend_probe.py --seq 16384
```

安装红线（沿用 hygon-dcu.md）：**绝不 pip install torch/torchvision**；wheel 名字里的
`torch290` / `cp311` / `dtk2604` 段必须与本机一致，装错版本会 undefined symbol。

## 6. 遗留空白（装完才有数据）

- flash 后端激活后：16384 token 单段 fwd+bwd 的显存与耗时（对比 math+分块+checkpoint 兜底）。
- `navit_attn_chunk_tokens` 兜底是否应默认关（flash 可用时它是纯开销）。
- DAS triton 与 torch 2.9 的 inductor 配套；FlexAttention 在 gfx936 上的数值对拍。
- VAE head_dim=384 永远走 chunk 兜底（flash 上限 256），与后端无关。
