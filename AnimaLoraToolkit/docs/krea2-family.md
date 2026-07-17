# Krea 2 (K2) 模型族训练指南

本仓库通过 **model family** 通用接口（`trainer/model_family.py`）支持在 Krea 2 上训练
LoRA/LoKr。`model_family: anima`（默认）时一切行为与历史逐一等价；本文档只讲
`model_family: krea2`。

## 架构速览（与 Anima 的差异）

| | Anima | Krea 2 |
|---|---|---|
| 骨干 | Cosmos MiniTrainDIT（cross-attn 条件） | 单流 MMDiT 12B（text+image 同序列） |
| 文本 | Qwen3-0.6B hidden states + T5 token 权重 | Qwen3-VL-4B **12 层** hidden states 堆叠，DiT 内 TextFusion 融合 |
| VAE | Qwen-Image VAE (Wan 系 f8/16ch) | **同一个**（latent 缓存/归一化直接复用） |
| 范式 | Flow Matching | Flow Matching（同参数化 v=noise−x0，t=1 纯噪） |
| timestep 调度 | flow_shift/schedule_shift | 分辨率感知 shift α=exp(mu(seq_len))（`krea2_res_shift`） |
| 官方工作流 | — | **RAW 上训、Turbo 上推理**（LoRA 直接可用于 Turbo） |

## 最小配置

```yaml
model_family: krea2
transformer_path: /path/to/raw.safetensors          # Krea-2-Raw（训练一律用 RAW）
vae_path: /path/to/qwen_image_vae.safetensors       # 与 anima 同一文件
krea2_text_encoder_path: /path/to/Qwen3-VL-4B-Instruct   # HF 目录
cache_latents: true

# navit 打包照常可用（推荐）：
navit_packing: true
navit_native_resolution: true
navit_token_budget: <按显存定>
navit_attn_backend: xformers   # xformers（默认，历史行为）| sdpa_seg
```

- **`navit_attn_backend: sdpa_seg`**（opt-in）：packed 注意力从 xformers
  BlockDiagonalMask varlen 换成**逐段 dense SDPA（cudnn）**。段内全注意力 ≡
  块对角语义，数学恒等（`tests/test_sdpa_seg_attention.py` 前向+梯度对拍）；
  H20 微基准 dense SDPA 比 xformers FA2 快 1.56×（云端 xformers 的 5D grouped
  路径无 backward 算子，训练实际一直走 4D 物化）。G=2~10 时逐段 launch 开销
  可忽略。该接缝同时是未来低比特 attention 后端（SageBwd INT8 等，2026-07 时
  上游 kernel 尚未开源）的插槽。

`text_encoder_path` / `t5_tokenizer_path` 在 krea2 下不使用。

## 关键行为

- **LoRA targets 默认 = DiT 全部 264 个 Linear**（官方/musubi 推荐口径，rank/alpha 32
  为作者默认；含 tproj.1。modulation/RMSNorm 是裸张量、非 Linear，本就不被包）。
  tproj.1 是全网最大单层，训练侧无问题（预览正常即证），部署时的显存坑见下方
  「ComfyUI 推理部署」。想复刻官方"长训练"的 attention-only（140 Linear）：
  ```yaml
  lora_targets: ["attn.wq","attn.wk","attn.wv","attn.wo","attn.gate"]
  lora_exclude_patterns: [".*txtfusion.*"]
  ```
- **timestep**：`krea2_res_shift: true`（默认）在任何 timestep mode 采样之后按
  每图 token 数施加官方分辨率感知 shift（navit 逐图、dense 全批同值）。开着它时
  `schedule_shift` 建议保持 1.0，避免双重偏移。mu 端点 256px→0.5 / 1280px→1.15
  可用 `krea2_shift_*` 微调。
- **文本**：`(tag:1.5)` 权重语法被剥离（无 T5 权重通道）。caption 静态时启用
  LRU cache（`krea2_text_cache_entries`，每条数 MB，按显存调）。navit 打包只带
  有效 token（数学上与官方 mask 等价——text token 无 RoPE，pad 作为 key 被 mask
  后贡献恒 0）。
- **采样预览**：走既有 `sample_latent`/ER-SDE；sigma 调度自动用 exp(mu(分辨率))。
  CFG 换算：官方 `--cfg g` 是 `cond + g·(cond−uncond)`，等价本仓库
  `cfg_scale = 1 + g`（官方 RAW 推荐 3.5 → 这里 **4.5**）。默认负面提示词为空
  （对齐官方；danbooru 质量 tag 对 krea2 语义未知）。RAW 是 CFG 模型，预览步数
  建议 ≥28。
- **不支持项（启动时 fail-fast）**：fit_packed_training/token_bucket/torch_compile、
  TREAD、DPO、LeapAlign、GAF、NCP、self-perceptual、dispersive、LoRA-One init——
  这些绑定 Anima 前向内部结构，需要时单独移植。ΔFM/VeCoR、Eisbach、LWD、
  min-SNR/EDM2 加权、三峰/Laplace/CSFlow 等 latent 空间/t 轴技术照常可用。

## ComfyUI 推理部署（tproj.1 的显存坑 + 解法）

**现象**：把训好的 LoRA 喂进 ComfyUI，日志刷 `ERROR lora diffusion_model.tproj.1.weight
Allocation on device`，出图人物风格在、**背景崩成乱图**。

**根因（已定位到 ComfyUI 源码，非训练/数据问题）**：ComfyUI 标准 *Load LoRA* 节点对每个
权重物化**完整 delta 矩阵**再并入底模（`comfy/weight_adapter/lora.py`：
`lora_diff = torch.mm(up, down).reshape(weight.shape)` → `weight += ...`）。tproj.1 =
`nn.Linear(6144, 36864)`，delta 是 `[36864,6144]`≈453MB(bf16)/905MB(fp32)；12B 底模占满
显存时这步 OOM，被 `except` 静默吞掉、该层回退底模。于是只有 tproj.1 用底模、其余 263 层是
训练后的 → 全局 timestep 调制（tproj 输出喂全部 28 个 block 共享）错位 → 背景崩。
（训练时的采样预览用活模型全 264 层、tproj.1 是训练后的，所以预览正常——问题只在 ComfyUI
这条实时加载路径。这也是 ComfyUI 已知问题，见 Comfy-Org/ComfyUI#12000 一类 fp8/大层 OOM。）

**解法：改用 ComfyUI 内置节点 `Load LoRA (Bypass, Model Only)`**（类
`LoraLoaderBypassModelOnly`，分类 `model/loaders`，节点菜单搜 "Bypass"；实现见
`comfy_extras/nodes_lora_debug.py` → `comfy.sd.load_bypass_lora_for_models`）。它把 LoRA
作为**前向低秩注入**（`out = base_forward(x) + strength·(x·downᵀ)·upᵀ·(α/r)`），**永不物化
完整 delta**，故 tproj.1 不再 OOM，全部 264 层（含 tproj.1）都正确生效。与"完整合并"数学
等价（本地用真实 tproj.1 因子实测：fp32 max_abs_diff 1.7e-6，bf16 1.6e-2 属正常舍入）。
接线：把原来的 *Load LoRA (Model Only)* 换成 *Load LoRA (Bypass, Model Only)* 即可，其余不变。

- 代价：每步每层多一次低秩 matmul（rank 32，相对底模 6144 维开销很小），略慢，画质不变。
- 该节点标 `EXPERIMENTAL`/"(for debugging)"，但是 ComfyUI 官方代码、机制正确。
- 若你的**推理** ComfyUI 是缺 `nodes_lora_debug.py` 的旧构建，把该文件作为 custom node
  放进 `custom_nodes/` 即可获得同名节点。
- 不要用"从 LoRA 里删掉 tproj.1 键"来绕：那等价于 tproj.1 用底模，背景照崩（已实测），
  且丢了官方推荐训练的一层。

## 显存与速度预期（未实测，推断）

12B bf16 冻结权重约 24GB；navit 显存系数与 Anima 的 `10GB+0.52MB/token` **不可比**
（模型大 6 倍），token_budget 需在云端重新标定，建议从小预算起步。RAW 训练分布
只到 1k：navit 多尺度大图超过 ~4096 token（1024²）属于底模分布外，谨慎。

## 验证状态（本地 RTX GPU，tests/test_krea2_modeling.py）

- packed navit ≡ 逐图 dense 前向：fp32 max_abs_diff **1.4e-06**（SDPA 回退）/
  **1.2e-06**（xformers varlen）；跨图隔离逐 bit 成立；per-block checkpoint 逐 bit 一致。
- 与官方 mmdit.py：state dict key 双向 strict 相同；同权重 bf16 前向
  max_abs_diff 1.6e-02（跨 kernel + 官方 256-pad，量级正常）。
- 12B 真权重加载、云端端到端训练、画质效果：**未验证**——首跑请先小步数冒烟。
