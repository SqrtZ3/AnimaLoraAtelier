# 冻结底模量化（base_quant）：FP8 / FP4

> opt-in、default-off。`base_quant: none`（默认）时训练行为与历史**逐 bit 等价**
> （不 import、不换装、不改任何前向）。
> 实现：`trainer/quant.py`；单测：`tests/test_base_quant.py`。

## 1. 它是什么、为什么可行

LoRA 训练里底模权重全程冻结，只通过 `LoRALinear.original(x)` 参与前向；
反向也只需要 `dL/dx = dL/dy @ W`（权重本身**不需要梯度**）。所以可以把
`original` 换成量化存储/量化 GEMM 的 `QuantLinear`，而：

- **LoRA adapter 全程 bf16 不受影响**（可训练参数、优化器状态、梯度精度都不变）；
- **PiSSA / ortho / ABBA / LoRA init 数学口径不变**——量化发生在 LoRA 注入
  **之后**，各种 init 在注入时读到的仍是 bf16 底模权重；
- checkpoint / resume 不受影响——training state 只存 adapter + optimizer，
  底模每次启动从原始 bf16 checkpoint 重新量化（确定性算法，结果一致）。

收益（Krea2 12.9B，权重 25.8GB bf16）：

| 格式 | 权重显存 | 释放 | 计算 |
|---|---|---|---|
| fp8 | ≈ 12.9GB | ≈ 12.9GB | H20 可走 fp8 GEMM（前向；反向 opt-in） |
| fp4 | ≈ 6.8GB | ≈ 19GB | H20 无 fp4 tensor core → 自动 dequant-bf16；Blackwell(sm120+) 可走 fp4 GEMM |

释放出来的显存可以直接换成更大的 `navit_token_budget`（显存 ≈ 常数 + 0.52MB/token，
见 memory navit-vram-linear）。

## 2. 配置

```yaml
# —— 全部默认值（即关闭状态）——
base_quant: none            # none | fp8 | fp4  底模权重存储格式
base_quant_gemm: auto       # auto | on | off   量化 GEMM（off = 只省显存）
base_quant_fp8_scale: auto  # auto | rowwise | tensorwise
base_quant_fp8_grad: false  # 反向 dL/dx 也走 fp8（e5m2 梯度 × e4m3 权重）
base_quant_include: null    # null = family 默认（见下）；regex fullmatch 列表
base_quant_skip: null       # 额外排除的 regex 列表
base_quant_fuse_act_quant: false  # torch.compile 融合激活量化链（见下）
```

- **`base_quant_fuse_act_quant`**：热路径的激活量化（abs→amax→div→clamp→cast）
  是 4-5 个分立 kernel，H20 profile 实测该 pointwise 家族每步吃数秒。开启后用
  torch.compile **只编译这两个量化纯函数**（不碰模型整图，与 navit /
  `torch_compile` 互斥无关），融成 1-2 个 kernel。启用时跑数值探针（dequant
  allclose）校验，编译失败/探针不过自动回退 eager 并打日志。数值可能有 ulp 级
  差异（fp8 码字边界值差 1 码），权重量化仍走 eager（保 resume 确定性）。

- **`base_quant_gemm: auto`**（推荐）：启动时用真实小 GEMM 探测当前设备/torch
  的 `_scaled_mm` 能力，可用则量化层走量化 GEMM，不可用自动退回
  「dequant 到 bf16 计算」（仍省显存）。`on` = 探测失败直接报错；`off` = 纯省显存。
- **`base_quant_fp8_scale: auto`**：优先 rowwise（每输出行一个 fp32 scale，精度
  更好；H20/sm90 + torch≥2.5 支持），不支持退 tensorwise。消费级 Blackwell
  （sm120，本地 5070）实测只有 tensorwise。
- **`base_quant_fp8_grad`**：默认 false = 反向 bf16（backward 时**重新 dequant**，
  bf16 权重不驻留显存）。true = 反向也走 fp8 GEMM，梯度先量化成 e5m2 —— 更快
  但 dL/dx 多一层量化噪声，**必须 A/B 验证拟合后再常开**。
- **include 默认（= 推理端 W4A4 已验证画质的同一集合）**：
  - krea2：`blocks.*`、`txtfusion.*`、`txtmlp.*`；**保持 bf16**：`first`、
    `last.*`、`tproj.*`、`tmlp.*`（timestep 调制通路最敏感）与所有非 Linear
    （RMSNorm/QKNorm/modulation 裸张量）。
  - anima：`blocks.*`。
- 形状不合规的层自动降级：fp8 GEMM 需要 in/out 都是 16 的倍数（不满足→该层
  dequant 模式）；fp4 需要 in 是 16 的倍数（不满足→该层保持 bf16，log 提示）。

### fail-fast 组合（v1 收窄变量面）

- `lora_variant: dora` —— DoRA 每步读底模全量权重算范数，量化后每步 dequant
  物化，收益归零且未验证。
- `lora_one_init_steps > 0` —— LoRA-One 要在底模权重上挂梯度，量化 buffer 做不到。
- `torch_compile: true` —— 自定义 autograd.Function + `_scaled_mm` 的图捕获
  行为未确认，先二选一。
- `mixed_precision` 必须 bf16。

## 3. 数值物证（本地 RTX 5070 sm120 / torch 2.9.1+cu130）

真实 Krea2 turbo.safetensors 权重逐层实测（`tests/test_base_quant.py` +
验证脚本）：

| 量 | fp8 (e4m3) | fp4 (nvfp4) |
|---|---|---|
| 权重 dequant relerr | **0.0264–0.0267**（rowwise≈tensorwise） | **0.093–0.095** |
| 单层前向 relerr（量化 GEMM，W×A 双侧量化） | **0.0375** | **0.134** |
| 单层前向 relerr（dequant-bf16 计算） | 0.0266 | 0.094 |

参照系：推理端 W4A4（int4，权重 relerr≈0.16）已实测**画质与 bf16 基本无差**
（见 ComfyUI 侧 KREA2_FP4_HANDOFF.md）。fp8 的误差比它小 4–6 倍、fp4 与推理端
nvfp4 完全同一口径（0.095 逐位吻合），据此**预期**训练侧质量安全——但
「LoRA 在量化底模上训练再部署到 bf16 底模」的端到端拟合影响没有直接证据，
需按第 5 节做云端 A/B。

本地微基准（5070 Laptop，krea2 主 block 形状，仅方向参考——H20 的
带宽/算力比完全不同，务必云端重测）：

| 路径 | 前向 | 前向+反向 |
|---|---|---|
| fp8 GEMM（反向 bf16） | 1.2–1.9× | ≈1.0× |
| fp8 GEMM + fp8_grad | — | 1.1–1.24× |
| fp4 GEMM（mlp 大层） | 最高 2.05× | ≈0.8×（反向 dequant 拖累） |
| dequant 模式（省显存不提速） | 0.6–0.9× | — |

解读：dequant 一次的带宽开销 ≈ 读一遍 bf16 权重，本地卡（算力/带宽比高）
上很贵；H20 带宽富裕（4TB/s）、bf16 算力孱弱（148 TFLOPS vs fp8 296），
预期 fp8 GEMM 的相对收益**更大**、dequant 模式的相对损失**更小**，
但这是推断（置信度中），以云端 stage_timing 实测为准。

## 4. 与其他链路的交互

- **训练中采样/导出 merged**：`QuantLinear.weight` property 现算 dequant 权重，
  `merged_weight()` / aclora 导出 / 取证脚本透明可用（结果带量化误差，与训练
  时前向看到的底模一致）。
- **部署口径**：成品 LoRA 是在"量化底模"上拟合的，部署到 bf16 底模（或推理端
  另一种量化）存在底模失配 —— QLoRA（nf4，误差更大）业界经验是可接受，
  但本仓库尚未实测；如出画质问题优先怀疑这一层。
- **梯度检查点**：量化前向确定性（同输入同输出），重算安全（有单测）。
- **navit / ARB / FiT**：只换 Linear 模块，与打包方式正交。

## 5. 上云清单（未验证项，按此裁决）

1. **烟测**：baseline yaml 只加 `base_quant: fp8`（其余逐字段一致），step-0
   采样正常、loss 曲线与 bf16 基线走势一致（前 ~200 步）。
2. **确认 H20 rowwise 生效**：日志应有 `fp8 scale=rowwise`；若退 tensorwise，
   检查云端 torch 版本（rowwise `_scaled_mm` 需 ≥2.5）。
3. **速度归因**：开/关 `base_quant_gemm`、`base_quant_fp8_grad` 各测 step 时间
   （fp8_grad 的 rowwise 分支本地无法验证，是唯一没跑过真机的代码路径——
   烟测时优先盯它的 loss/梯度范数是否正常）。
4. **拟合 A/B**：同数据同种子 bf16 vs fp8（vs fp8+fp8_grad），对比 loss 曲线
   与画面；过了再考虑 fp4（H20 上是纯省显存换 dequant 开销）。
5. 显存收益换 token budget 时，注意 cache 阶段 VAE encode 的独立显存尖峰
   （另一条链路，`cache_encode_tiled` 管）。

## 6. 云端环境要求

- fp8 GEMM：torch ≥ 2.5（rowwise scale 的 `_scaled_mm`）+ CUDA 12+，H20(sm90)
  硬件支持。tensorwise 路径 torch ≥ 2.2 即可。
- fp4 GEMM：仅 Blackwell（sm120+）+ torch ≥ 2.8（`float4_e2m1fn_x2` dtype +
  fp4 `_scaled_mm`）+ cuBLAS 13；H20 上自动退 dequant，无需任何额外依赖。
- 无新增 pip 依赖（swizzle 布局自实现，见 THIRD_PARTY_NOTICES torchao 条目）。
