# 缓存格式契约

TPU 后端**只跑 DiT，不做任何编码**。它吃的是磁盘上的 npz，不认产出 npz 的工具。
所以这份文档写的是**契约**，不是"我们的工具怎么用" —— 只要你产出的 npz 符合下面
的键名与形状，用什么实现都行（自己的 VAE、别的框架、云端预处理产物都可以）。

配套有一个可执行的校验器：

```bash
python tools/verify_cache.py <缓存目录> --config <你的训练 yaml>
```

它把 `jax_tpu/data.py:CacheDataset._scan` 的判据搬到本地跑，几秒钟出结果。
**上机前跑一遍**：缺一个文件的代价是 push 一轮、等挂载、起 TPU、然后挂掉。

---

## 目录布局

一个平铺目录，每个样本一组同 stem 的文件：

```
<数据集目录>/
├── img001.npz              图像 latent（必需）
├── img001.textfeat.npz     文本条件（必需）
├── img001.ms2304.npz       多尺度副本（开了 navit_multiscale 才要）
├── img002.npz
├── img002.textfeat.npz
└── _empty.textfeat.npz     空 caption 条件（caption_dropout_rate>0 才要）
```

**stem 怎么认**（与 `jax_tpu/data.py:_stems` 同口径）：目录里有图片文件时按图片名
推 stem；**一张图片都没有时**按 `<stem>.npz` 推，并排掉 `.textfeat` 与 `.ms<档>`
这两类 sidecar。

所以缓存目录**可以完全不含像素** —— 而且上传时**必须**不含：TPU 侧从不读图片
（只取文件名），caption 也已烘焙进 textfeat，放原图纯属多余暴露。2026-08-21
真实吃过一次：96 张原图跟缓存一起传上 Kaggle，被按 NSFW 条款**整个 dataset 删掉**
并向账号告警。

## `<stem>.npz` —— 图像 latent

| 键 | 类型 / 形状 | 说明 |
|---|---|---|
| `latent` | `[16, T, H/8, W/8]`，见下方 dtype | VAE encode 的结果。C=16 固定，T=1（单帧） |
| `bucket_w` / `bucket_h` | 标量 int | 编码时的**像素**尺寸 |
| `dtype_kind` | 字符串 | `"bf16"` 时磁盘上是 uint16 位模式；`"fp32"` 时是 float32 |
| `latent_flipped` | 同 `latent` | 只在 `flip_augment: true` 时需要，见下 |

约束：

- `H/8` 与 `W/8` 都必须能被 **patch=2** 整除（token 网格 = `(H/16, W/16)`）。
- 单图 token 数 = `(H/16) × (W/16)`，**必须 ≤ 单卡预算** = `navit_token_budget / 8`。
  超了打包器 fail-fast（"单段 N > budget"）。
- 像素尺寸按 **16px（patch2 × VAE8）floor 对齐**：`plan_native_fit_image` 的
  `align_mode="floor"`，即左上角裁掉余数、零 padding、mask 恒全 1。
  **不要用 ceil/pad 对齐** —— 那是 ARB 桶路径的做法，抄混了 token 数就不对。

### bf16 的读法（抄错不报错）

numpy 没有原生 bfloat16，所以磁盘上存的是 **uint16 位模式**。
**直接 `.view(np.float16)` 会把位模式解释错**（静默出错，数值全废）。
正确做法是先转 uint16 再 bitcast：

```python
jax.lax.bitcast_convert_type(jnp.asarray(a_uint16), jnp.bfloat16)
```

写入侧对应地取高 16 位（**就近舍入**，不是截断）。

### `latent_flipped` 为什么必须单独存

翻转必须在**像素域**做完再 encode。VAE 的卷积 encoder 不是 flip-等变的
（latent 空间翻转 ≠ 像素空间翻转），训练时手上没有 VAE，补不了。

`flip_augment: true` 而缓存里没有这个键时，TPU 侧**直接报错**而不是静默不翻 ——
悄悄不翻会让 yaml 说的事没发生，且日志上看不出来。

## `<stem>.textfeat.npz` —— 文本条件

两个模型族格式不同。

### anima（`model_family` 缺省）

| 键 | 类型 / 形状 | 说明 |
|---|---|---|
| `cross` | `[512, 1024]` bf16 位模式 | 交叉注意力条件，**定长 512** |
| `mask` | `[L]` uint8 | T5 的 attention mask（1=有效） |
| `caption` | 字符串标量 | 原文，仅供追溯；训练路径不读 |
| `meta` | JSON 字符串 | 编码器路径 / max_length / 是否过 adapter |

**`cross` 不是 Qwen 的 hidden state** —— 这是最容易抄错的一处。真实链路是：

```
cross = llm_adapter(qwen_hidden, t5_ids, t5_attn, qwen_attn) * t5_token_weights
```

其中 `llm_adapter` 是 anima 底模里的一个 6 层 transformer 桥（底模里 118 个
`net.llm_adapter.*` 键，134.7M 参数），默认冻结、不注入 LoRA，所以它的输出可以
安全地离线缓存。`t5_token_weights` 是逐 token 权重（`(xxx:1.2)` / `[xxx]` 那套
A1111 语法解析出来的）。

只存 Qwen hidden 会让训练**一直条件错误且不报错**。本仓库的
`tools/cache_text_features.py` 里那句 `if c.shape[0] != max_length: raise`
就是拦这个的。

### krea2（`model_family: krea2`）

| 键 | 类型 / 形状 | 说明 |
|---|---|---|
| `txt` | `[L, 12, 2560]` bf16 位模式 | Qwen3-VL 文本塔的 **12 层 hidden 堆叠**，**变长** |
| `caption` | 字符串标量 | 同上，训练路径不读 |
| `meta` | JSON 字符串 | 同上 |

取的是哪 12 层由 `KREA2_SELECT_LAYERS = (2,5,8,...,35)` 定。注意 `hidden_states[k]`
（k < 层数）是**第 k 层的输入**，所以最大 tap 35 意味着只需跑 0~34 层。

krea2 的段是 `[text ; image]` 同一条序列，**预算两边一起花** ——
单图 `image_tokens + text_tokens ≤ 单卡预算`。

⚠ 体积提醒：`txt` 约 **61.4KB/token**，96 张图能到 1.7GB，而信息源只有 96 条
caption（0.14MB）。所以 krea2 推荐**让 TPU 现算**：本地只跑
`tools/dump_caption_ids.py` 产出约 40KB 的 token ids，用
`build_job.py --caption-ids` 内嵌进 job，真机上由 `jax_tpu/text_cache.py` 现算。

## `<stem>.ms<档>.npz` —— 多尺度副本

开了 `navit_multiscale` 时，每张图额外存一份**低 token 档的缩小副本**，
格式与 `<stem>.npz` 完全相同（自带自己的网格与 token 数）。

规则：

- 只降不升采样。源图 token 数 ≤ 档位时**不产出**这个档的 sidecar。
- 产出方式是"等比缩放到覆盖目标尺寸再**中心裁剪**"（LANCZOS），所以有效区填满
  整个 latent，mask 恒全 1。
- 文件名里的档位数字必须与 yaml 的 `navit_multiscale_token_ladder` 对得上 ——
  不一致会让训练看到的样本集与 plan-only 算的不同。
- 副本与本体**共用同一份 textfeat**（同一条 caption），不需要单独存。

`navit_multiscale: true` 而目录里一个 sidecar 都没有时，TPU 侧 fail-fast。

## `_empty.textfeat.npz` —— 空 caption

`caption_dropout_rate > 0` 时**必需**：文本特征是离线缓存的，训练时手上没有编码器，
没法临时算一个空 caption 的条件。格式与普通 textfeat 相同，caption 是空串。

缺了直接报错 —— 悄悄退回"不 dropout"会让同一份 yaml 在两个后端上训出不同的
条件鲁棒性。

## 自己实现的话，最容易错的五处

按"错了会不会报错"排序，越靠前越危险：

1. **`cross` 存成 Qwen hidden**（不过 llm_adapter）—— 形状对、不报错、条件全错。
2. **bf16 用 `.view(np.float16)`** —— 位模式解释错，数值全废，不报错。
3. **对齐用 ceil 而不是 floor** —— token 数变了，打包布局跟着变，loss 曲线与
   GPU 侧悄悄对不上。
4. **latent 域翻转代替像素域翻转** —— VAE 非 flip-等变，数值不对但看不出来。
5. **bf16 用截断而不是就近舍入** —— 半个 ulp 的系统偏差。

前四条 `tools/verify_cache.py` 只能查出第 2、3 条的一部分（形状与 token 数）；
第 1、4、5 条要靠对拍。要拿本仓库的 torch 工具做参考实现，跑：

```bash
export ANIMA_UPSTREAM=<不需要 —— 这条只对比本仓库工具与 upstream>
python tools/tests/check_cache_parity.py --vae <VAE> ...
```

或者更直接：拿本仓库的 `tools/cache_latents.py` 产一份，与你的实现逐 bit 比。
