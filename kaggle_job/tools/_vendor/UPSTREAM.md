# `_vendor` 溯源台账

本目录的代码来自 upstream 仓库 `anima-lora-train` 的 `AnimaLoraToolkit/`。
**这不是重新实现，是照抄** —— 所以必须能机械回答"抄的是哪个版本、抄了哪几行、
改过什么"。本文件就是那份台账，`tools/check_sync.py` 直接读它做漂移检查。

- 抄写时点：upstream `HEAD = 362eae3c`（2026-08-23）
- 校验方式：`ast.unparse` 逐符号比对（不看空白与注释格式，只看代码结构）；
  整份复制的走 sha256，**行尾归一化后**比（本仓库统一 LF，upstream 那几份是 CRLF）
- 复核命令：`python tools/check_sync.py --upstream <AnimaLoraToolkit 路径>`

## 一、整份复制（内容逐字节相同，行尾统一为 LF）

| 本地文件 | upstream 来源 | 归一化 sha256[:12] | 来源 commit |
|---|---|---|---|
| `_vendor/wan_vae.py` | `models/wan/vae2_1.py` | `890b9ec219a4` | `e5719093` |
| `../kaggle_fast_upload.py` | `tools/kaggle_fast_upload.py` | `22225dbf71cb` | `9a13ba6b` |
| `../dataset_encrypt.py` | `tools/dataset_encrypt.py` | `31c662423dbb` | `0a5547b7` |

`wan_vae.py` 保留了 Alibaba Wan Team 的版权头（Apache-2.0）。它只依赖
torch / einops / logging / os，本来就自包含，所以整份复制而非摘录 —— 摘出
Encoder3d 那条链要挑 9 个类约 450 行，省 260 行换来漂移风险，不值。

## 二、按 AST 行号摘录（函数体逐字，仅补 import 头）

行号是**抄写时**的 upstream 行号，仅供人工定位；`check_sync.py` 不依赖它们
（按符号名重新解析，上游插几行不会误报）。

### `_vendor/latent_plan.py` ← `trainer/data.py`（sha `b1c425b4f979ce6e`, commit `3507fa8f`）

| 符号 | upstream 行 |
|---|---|
| `NativeFitImagePlan` | 42-54 |
| `_ceil_to_multiple` | 57-60 |
| `_floor_to_multiple` | 63-66 |
| `plan_native_fit_image` | 69-133 |
| `plan_multiscale_copy` | 136-187 |
| `_tile_starts` | 2033-2041 |
| `_blend_ramp` | 2044-2052 |
| `tiled_vae_encode` | 2055-2115 |

**为什么这几个必须逐字**：它们决定每张图的 token 数与网格，而 token 数决定 TPU
侧的打包布局。口径偏一点，两个后端的 loss 曲线就悄悄对不上 —— 不报错。

### `_vendor/st_load.py` ← `trainer/checkpoint.py` + `trainer/models.py`

来源 sha：`checkpoint.py = 69705a0073b8e23d`（commit `4067b5f4`）、
`models.py = 9b3dd481252cf15d`（commit `4067b5f4`）。

| 符号 | upstream 位置 | 状态 |
|---|---|---|
| `_strip_prefixes` | `checkpoint.py:35-46` | 逐字 |
| `_pick_best_prefix_remap` | `checkpoint.py:49-81` | 逐字 |
| `_load_safetensors_into_model` | `checkpoint.py:126-254` | 逐字 |
| `load_vae` | `models.py:268-316` | **改 1 处**，见下 |

`load_vae` 的改动（`check_sync.py` 白名单里记着）：
- 签名 `repo_root` 改成可选（`repo_root=None`），本函数忽略它；
- `load_module_from_path("wan_vae", repo_root / "wan" / "vae2_1.py")`
  → `from . import wan_vae`。

本仓库里 VAE 实现就在 `_vendor/wan_vae.py`（与上游逐字节相同），不需要满仓库
找 `anima_modeling.py` 来定位模型代码目录。**其余全部逐字** —— 特别是那 16 维
latent mean/std，抄错不报错、只会让所有 latent 偏移。

### `_vendor/llm_adapter.py` ← `models/anima_modeling.py`（sha `8209f76f46e01436`, commit `4067b5f4`）

| 符号 | upstream 行 | 状态 |
|---|---|---|
| `rotate_half` | 12-15 | 逐字 |
| `apply_rotary_pos_emb` | 18-22 | 逐字 |
| `RotaryEmbedding` | 25-56 | 逐字 |
| `LLMAdapterAttention` | 59-106 | 逐字 |
| `LLMAdapterTransformerBlock` | 109-153 | 逐字 |
| `LLMAdapter` | 156-220 | **删 3 行**，见下 |
| `ADAPTER_CONFIG` | 231-239 的实参 | 转写成 dict |
| `load_llm_adapter` | — | 本仓库新增 |

`LLMAdapter.forward` 删掉的三行是 `utils.npu_compat.expand_attn_mask` 的
import 与两次调用。那是昇腾 NPU 专用（FlashAttentionScore 不接受 Sq=1 的广播
mask），upstream 自己的注释写明「CUDA/CPU 上 expand_attn_mask 是恒等映射，行为
逐字节不变」。本仓库缓存工具只跑 CUDA/CPU。

**`net.llm_adapter.` 前缀是实测的**（不是猜的）：
`anima-base-v1.0.safetensors` 共 685 个键，其中 118 个以此开头，
合计 134.7M 参数 / 269MB（全文件 4182MB）。

已验证：`load_llm_adapter` 与 upstream 走全量底模的
`dit.preprocess_text_embeds(...)` 输出**逐 bit 相同**（`torch.equal` 为 True），
加载耗时 **0.7s vs 42.2s**。

### `_vendor/t5_weighted.py` ← `trainer/text_encode.py` + `trainer/models.py`

来源 sha：`text_encode.py = b5a6382ffaa98dce`（commit `3e0574e8`）、
`models.py = 9b3dd481252cf15d`（commit `4067b5f4`）。

`text_encode.py` 的 12 个符号全部逐字：`_QWEN_CACHE`(38) `_T5_CACHE`(39)
`_TEXT_CACHE_CAP`(40) `_TEXT_CACHE_ENABLED`(41)
`set_text_encode_cache_enabled`(43-46) `reset_text_encode_cache`(49-52)
`_cache_get`(55-61) `_cache_put`(64-70) `_parse_weighted_tag`(73-99)
`_build_qwen_text_from_prompt`(102-110) `encode_qwen`(113-175)
`tokenize_t5_weighted`(178-243)。

`models.py` 的 3 个符号（Qwen/T5 加载，同属文本编码侧）也逐字：
`_QWEN_LEGACY_SUBDIR`(319) `_resolve_qwen_dir`(322-342)
`load_text_encoders`(345-379)。

**LRU cache 一并抄了**，尽管离线缓存场景用不上它（每条 caption 只编一次）。
原因：cache 判断嵌在 `encode_qwen` / `tokenize_t5_weighted` 的函数体里，摘掉就
不再是逐字摘录，也就丧失了机械校验能力。多 60 行换一个可 diff 的口径，值。

### `_vendor/krea2_te.py` ← `trainer/model_family.py`（sha `79d98f7b5f73b2c8`, commit `0be0d9ba`）

14 个符号全部逐字：`_KREA2_PROMPT_PREFIX`(154-158) `_KREA2_PROMPT_SUFFIX`(159)
`_KREA2_PREFIX_IDX`(160) `_KREA2_SUFFIX_START_IDX`(161) `KREA2_SELECT_LAYERS`(162)
`load_krea2_text_encoder`(165-196) `_KREA2_TEXT_CACHE`(202)
`_KREA2_TEXT_CACHE_CAP`(203) `_KREA2_TEXT_CACHE_ENABLED`(204)
`set_krea2_text_cache`(207-212) `reset_krea2_text_cache`(215-216)
`_encode_krea2_batch`(219-278) `_encode_krea2_single`(281-295)
`encode_krea2_text`(298-364)。

前四个常量是 chat 模板的**切片位置**，抄错不报错、只让条件全歪。
`tools/dump_caption_ids.py` 里那两道 tokenizer 闸门（prefix 单独 tokenize 必须
恰好 34 个 token）就是拦这个的 —— 别删。

`encode_krea2_text` 里有一处改动（`check_sync.py` 白名单）：函数体内的
`from trainer.text_encode import _build_qwen_text_from_prompt`
→ `from .t5_weighted import _build_qwen_text_from_prompt`。同一个函数，
本仓库里它在 `_vendor/t5_weighted.py`（逐字摘录）。

## 三、同步流程（上游改了怎么办）

```bash
export ANIMA_UPSTREAM=/path/to/anima-lora-train/AnimaLoraToolkit
python tools/check_sync.py            # 报哪个符号漂了
```

漂了之后：

1. 先看上游那次改动**是否影响缓存产物**。改注释/日志/类型标注 → 只需重抄并更新
   本文件的 sha；改数值口径（对齐单位、归一化常量、切片位置）→ **必须重跑**
   `tools/tests/check_cache_parity.py`，并考虑已有缓存是否要重新生成。
2. 重抄用 AST 行号，不要手抄。
3. 更新本文件的 sha256 与 commit。

**两道防线要一起跑**：`check_sync.py` 发现"上游改了"，
`check_cache_parity.py` 发现"抄的时候漏了什么"。只跑一个都不够。
