# Qwen3-VL-4B-Instruct (heretic) — Krea2 备用文本编码器

Krea2 链路的**可选**文本编码器，与 `../Qwen3-VL-4B-Instruct/`（官方权重）并存。
用法：把 yaml 里的 `krea2_text_encoder_path` 指向本目录即可，代码无需改动。

## 来源与生成方式

权重来自本机 ComfyUI 侧的单文件 `qwen3vl_4b_heretic_bf16.safetensors`（社区发布的
Qwen3-VL-4B-Instruct abliterated 变体）。该文件是 ComfyUI 的扁平键名布局，
**不能**直接被 `Qwen3VLForConditionalGeneration.from_pretrained` 加载，原因有二：

1. `from_pretrained` 需要 HF 目录（config.json + tokenizer 一套 + 权重），不接受单个 safetensors；
2. 键名空间不同 —— ComfyUI 版语言塔是 `model.layers.*`，HF 版是 `model.language_model.layers.*`。

本目录即按官方布局重新组织的产物：

- **权重**：纯键名重映射 `model.{embed_tokens,layers,norm}.*` → `model.language_model.{...}`；
  `model.visual.*` 原样保留。张量字节未做任何转换，按官方 `model.safetensors.index.json`
  的 shard 划分流式拷贝原始字节（不反序列化，内存 O(1)）。
- **配置/tokenizer**：从 `../Qwen3-VL-4B-Instruct/` 逐文件复制，未做修改。
  `model.safetensors.index.json` 也是原样复用 —— 键集与 shard 划分完全对齐。

## 验收结果（本地实测）

- 键集对拍：官方 index.json 713 键 ↔ 重映射后 713 键，**集合完全相等，无多无缺**。
- shard 划分：231 / 482 张量，与官方一致；shard-1 文件大小 4967229296 字节，
  与官方 LFS pointer 声明的字节数**逐字节相同**。
- 总数据量 8875631616 字节 == 官方 index.json 的 `metadata.total_size`。
- 抽样 7 个张量（embed / 语言塔首中尾层 / final norm / 视觉塔 block / deepstack merger）
  与源文件**逐 bit 相同**（bf16 按 int16 视图比较）。
- transformers 5.7.0 下用 meta device 建骨架比对：模型期望的键与本目录提供的键完全吻合；
  唯一"缺"的 `lm_head.weight` 由 `tie_word_embeddings: True` 在加载时与 `embed_tokens`
  绑定生成 —— 官方目录同样没有该键，行为一致。

**未验证**：没有在云端实际跑过训练，也没有做过与官方 TE 的 A/B 对比。以上仅证明
"能被正确加载且权重与源文件一致"，不构成任何关于训练效果的结论。

## 一个值得注意的技术前提

在 Krea2 链路里 Qwen3-VL 只作特征提取器用 —— 取 12 层 hidden states 喂 DiT 的 TextFusion
（见 `trainer/model_family.py` 的 `KREA2_SELECT_LAYERS`），**不走 lm_head、不生成 token**。
abliteration 是在 LLM 的生成/拒答回路上做的方向消融，对"hidden states 如何编码概念"属于
副作用而非设计目标，方向未定 —— 有可能只是给条件信号引入了一个分布偏移。

因此换用本目录后**必须做单变量 A/B**才能判断优劣：同一份数据、同一 yaml，只改
`krea2_text_encoder_path`，比 loss 曲线与同 seed 采样。不要默认它更好。
