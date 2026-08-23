# 启动一次新的 TPU 训练（新数据集 / 新参数）

Kaggle TPU v5e-8 上跑 Anima / Krea2 LoRA 训练的**完整启动流程**。
前置知识（环境搭建）在 `SETUP.md`，架构结论与实测账目在 `docs/ARCH_FINDINGS.md`，
本文件只讲"怎么操作"。命令都在**仓库根目录**跑（Windows 下用 Git Bash）。

## 约定：先设这几个变量

本文档用变量代替具体路径，照抄前先设好（或自己替换）：

```bash
TORCH_PY=/path/to/.venv-torch/bin/python      # Windows: .../Scripts/python.exe
JAX_PY=/path/to/.venv-jax/bin/python
KAGGLE_USER=<你的 Kaggle 用户名>
DATA=/path/to/your-dataset                    # 图片 + 同名 .txt 所在目录
MODELS=/path/to/your/models                   # VAE / 文本编码器 / 底模所在处
```

## 0. 心智模型

TPU 后端（`jax_tpu/`）**只跑 DiT，不做任何编码**。所以一次新训练的准备顺序
永远是：

```
图片+标签 ──①──> latent/textfeat 缓存（本地 PyTorch，一次性）
                ② yaml 适配（同一份 yaml 也能驱动 GPU 侧，如果你有那个仓库）
                ②b verify_cache 校验（零配额，几秒，别跳）
                ③ 缓存上传 Kaggle Dataset（私有）
                ④ build_job 打单文件脚本（底模走 HF 直下或挂 dataset，不用上传）
                ⑤ 本地 --plan-only 验证（零配额）
                ⑥ push 跑（kaggle_run.ps1）→ 拉回 checkpoint
```

两个 python 的分工（写死，别搞混，理由见 `SETUP.md`）：
- **torch 活**：`$TORCH_PY` —— 只用于 ① 的数据缓存
- **jax 活**：`$JAX_PY` —— jax 0.11.0，与真机钉同一版本（真机由 job preamble
  自动升级到这个版本）；用于 ⑤ 与所有闸门

## ① 数据缓存（新数据集必做，旧缓存可复用）

数据集目录 = 图片 + 同名 `.txt` 标签（tag 风格，trigger 词自己定）。放进目录后：

**文本特征**（CPU 即可，几分钟；`--empty-caption` 在 `caption_dropout_rate > 0` 时
**必须**带，否则 TPU 侧 fail-fast）：

```bash
$TORCH_PY tools/cache_text_features.py \
  --data-dir "$DATA" \
  --text-encoder "$MODELS/text_encoders/Qwen3-0.6B-Base" \
  --transformer "$MODELS/diffusion_models/anima-base-v1.0.safetensors" \
  --t5-tokenizer "$MODELS/t5_tokenizer" \
  --device cpu --empty-caption
```

`--transformer` 给的是 anima 底模，但**只会读里面的 `llm_adapter`**
（118 个键 / 269MB，不是整个 4.2GB）—— 与走全量底模的输出逐 bit 相同，
加载快 60 倍。原因见 `tools/_vendor/UPSTREAM.md`。

**图像 latent**（GPU 更快，CPU 也能跑；`--ms-ladder` 必须与 yaml 的
`navit_multiscale_token_ladder` 一致；yaml 开 `flip_augment` 时**不要**加
`--no-flip`）：

```bash
$TORCH_PY tools/cache_latents.py \
  --data-dir "$DATA" \
  --vae "$MODELS/vae/qwen_image_vae.safetensors" \
  --ms-ladder 4096
```

**分块编码是近似，整图直编才精确。** 超过 `--tiled-threshold`（默认约 1.05Mpx）
的图走分块 + 羽化拼接。注意 VAE encoder 的 mid block 是**全局注意力**，感受野
= 整块，所以分块误差是**全域的**而非只在缝附近（整图 vs 分块 mean|Δ|≈0.002，
不随离缝距离衰减），加大 overlap 只能压低、不能消除。

默认 tile 832 / overlap 256 是权衡后的值（比旧的 640/128 误差低约 40%、
峰值显存 4.4GB、多约 20% 耗时）。要完全精确就把 `--tiled-threshold` 抬到
4000000 并在 ≥24GB 显存的机器上跑。0.002 量级对 LoRA 风格训练可忽略，
但"近似"二字如实记录，取舍由你。

### ①b 校验（几秒，别跳）

```bash
python tools/verify_cache.py "$DATA" --config config/<你的>.yaml
```

它把真机上的 fail-fast 判据搬到本地一次性跑完：

- 每张图有 `<stem>.npz`（含 `latent`；flip 开时还要 `latent_flipped`）
- 每张图有 `<stem>.textfeat.npz`（anima 键 `cross` [512,1024]；krea2 键 `txt`）
- 开了 multiscale 就有 `<stem>.ms<档>.npz`，且档位与 yaml 一致
- `caption_dropout_rate>0` 时有 `_empty.textfeat.npz`
- 单图 token 数 ≤ 单卡预算

**跳过这步的代价是白烧一轮配额**：push + 等挂载 + 起 TPU + 挂掉。

## ② yaml 适配（换参数在这里）

从 `config/train_anima_tpu_template.yaml`（或 krea2 那份）改起 ——
那是真机跑过的配方。若你手上有 GPU 侧 yaml，**逐字段照抄**，只动该动的。TPU 侧
`jax_tpu/config.py` 把每个键强制归为四类，**不在任何一类里的键直接报错**
（fail-fast，不静默忽略）——所以删键比改键重要：

- **必须删掉的后端专属键**：`device_backend`、`navit_attn_backend`、`fast_model_init`、
  `base_quant`、`wandb_*`、整族 `sample_*`（TPU 无训练期采样）、`save_state_every_epochs`。
- **budget 规则**：`navit_token_budget` 在 TPU 解释为**全局**预算（8 卡纯 DP），
  必须被 8 整除；单卡 = 全局/8，v5e 实测上限量级是 16384/卡（32768 OOM）。
  它只是 **FFD 装箱的容量上限**：每个 pack 实际补齐到自己的自然长度
  `round_up(Σ实段, 1024)`，不再补齐到 budget。
  **硬约束：数据集里单图最大 token 数必须 ≤ 单卡预算**，否则打包器 fail-fast
  （"单段 N > budget"）。token 数 ≈ (W//16)×(H//16)。举个真实例子：某数据集
  最大 16268 token → 只能 16384/卡（这就是"工作点 B" 8192/卡 对它不成立的原因）。
- **开了会报错的未移植开关**（留着 false/0/空 就没事）：`sample_every>0`、
  `tread_enabled`、`dispersive_enabled`、`aux_perceptual_enabled`、`noise_offset>0`、
  `pyramid_noise_iterations>0`、`shuffle_caption`、`lora_dropout>0`、
  `resume_lora`（续训用 `resume_state`，它连优化器矩一起存）、`dora_export_mode`
  非 native。
- yaml 里的路径写**本地值**；上云路径由 build_job 的 `--env` 烘焙覆盖，yaml 不动。

改完先过一道（jax 侧），报错信息会直接告诉你哪个键有问题：

```bash
$JAX_PY -c "
import sys, yaml; sys.path.insert(0, 'jax_tpu'); import config
config.build(yaml.safe_load(open('config/<你的>.yaml', encoding='utf-8')),
             devices=8, canvas_hw=(128, 128))
print('配置 OK')"
```

## ③ 上传数据集

**只传 npz**。原图与 `.txt` 一个都不要带上去 —— `jax_tpu/data.py:CacheDataset`
「无图时按 npz 推 stem」，缓存目录里根本不需要图片文件（实测：一个纯 npz 的目录
plan-only 照常认出 151 条样本）。这不只是省体积：2026-08-21 真实吃过一次，
96 张原图跟缓存一起传上去，被按 NSFW 条款**整个 dataset 删掉**并向账号告警。

```bash
STAGING=/path/to/staging/<slug>
mkdir -p "$STAGING"
cp "$DATA"/*.npz "$STAGING"/          # 只挑 npz，原图/标签留在本地
cat > "$STAGING/dataset-metadata.json" <<EOF
{"title": "<slug>", "id": "$KAGGLE_USER/<slug>", "licenses": [{"name": "CC0-1.0"}]}
EOF

# 用并行上传器（官方 CLI 是串行的，几百个小 npz 会把带宽跑成锯齿）
PYTHONUTF8=1 python tools/kaggle_fast_upload.py "$STAGING" --mode create --workers 8
```

`kaggle_fast_upload.py` 相对 `python -m kaggle datasets create -p .`：

- **并行 + 连接复用**，把「token 往返时网卡空转」这段填掉；跑完会打印
  `空闲秒占比 / 变异系数`，可直接判断带宽有没有用满（判据定义见工具 docstring）。
- **两道数据外泄闸门默认开**：目录里有 png/jpg/txt 之类直接 fail-fast；
  `.npz` 内部藏 `caption` 明文（`cache_text_features.py` 会塞）也 fail-fast。
  `--dry-run` 只扫描不发字节。要放行得显式 `--allow-ext` / `--allow-npz-caption`。
- 传新版本：`--mode version -m "说明"`。

注意：
- 上传的是**私有** dataset（默认；`--public` 才公开）。**两种挂载布局并存（实测）**：
  有的 dataset 挂在 `/kaggle/input/datasets/<用户名>/<slug>/`，
  有的挂在 `/kaggle/input/<slug>/` —— 同一个 kernel 里两种都可能出现。
  推之前拿不准就用 `input_probe/` 零配额探一下（把 dataset_sources
  挂上后跑一遍，它会打印 /kaggle/input 的实际树）。
- `datasets status $KAGGLE_USER/<slug>` 显示 **ready 之后**才能 push kernel，
  否则挂载不到（白烧约 1 分钟 TPU）。
- 已经缓存好、但 npz 里带了 caption 明文的，用 `tools/dataset_encrypt.py`
  剥掉（训练路径只读 `cross`/`txt`，剥掉逐 bit 无影响）。

## ④ 打包 job

```bash
cd anima_train
MSYS_NO_PATHCONV=1 $JAX_PY build_job.py \
  --config ../config/<你的>.yaml \
  --env ANIMA_DATA_DIR=/kaggle/input/datasets/$KAGGLE_USER/<slug> \
  --env ANIMA_OUTPUT_DIR=/kaggle/working/out \
  --hf-model circlestone-labs/Anima:split_files/diffusion_models/anima-base-v1.0.safetensors:f7382c4bf9d7ffe4ceea593a0adbb470c56dd79b \
  [--extra-args "--max-steps 40"]
cd ..
```

- `MSYS_NO_PATHCONV=1` **必须**（否则 Git Bash 把 /kaggle/... 改成 C:/Program Files/...）。
- 底模走 HF 直下：Kaggle→HF 实测 10~13s 拉 4.2GB；revision 钉死，且 HF 文件
  sha256 已与本地对拍用权重核对一致（bd43b7cf…）。换底模就换 repo:文件:revision。
- 每次改了 `jax_tpu/*.py` 或 yaml 都要**重跑 build_job**（生成物不进 git，
  生成物过期 = 把旧代码推上去）。
- `anima_train/kernel-metadata.json` 的 `dataset_sources` 要挂上
  `$KAGGLE_USER/<slug>`（改一次即可）。模板里是空数组，第一次用必须填。
- `id` 字段里的 `KAGGLE_USERNAME` 占位符会被 `kaggle_run.ps1 -Username` 自动改写，
  不用手改。

## ⑤ 本地 --plan-only（零配额，必做）

```bash
cd jax_tpu
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  $JAX_PY run_train.py --config ../config/<你的>.yaml --plan-only
cd ..
```

看三个数（决定 8 卡用没用完）：
- **填充率**（线性层算力利用率上界，分母是自然总长；<90% 说明量化浪费大，
  调 quantum。"容量利用率"是分母为 budget 的对照口径，看装箱容量浪不浪费）
- **成步率**（能凑齐 8 个同布局 pack 的比例；小数据集天然低，顺延是设计行为）
- **布局数**（= 全模型编译次数；爆炸就调大 `--quantum`）

要调 quantum 时别靠猜，跑 preflight：
`jax_tpu/tests/enum_quantum_advisor.py --config <你的 yaml>` 会拿你这次的
数据集扫出建议值。

**改了 `jax_tpu/` 任何代码，先把闸门跑完再推**（见 `jax_tpu/tests/README.md`，
十几条在本仓库独立可跑，不需要 torch 与权重）。
改了 `tools/_vendor/` 则跑 `tools/check_sync.py` 与
`tools/tests/check_cache_parity.py`。

## ⑥ 推送 / 拉取

```bash
pwsh.exe -NoProfile -ExecutionPolicy Bypass -File "$PWD/kaggle_run.ps1" \
  -Username $KAGGLE_USER -Accelerator TpuV5E8 -TimeoutSec 10800 \
  -JobDir anima_train -FilePattern '(.*\.safetensors|.*\.npz|.*\.json|.*\.log)$'
```

- **必须 pwsh（PS7）**：Windows PowerShell 5.1 把无 BOM 的 UTF-8 当 GBK 读，
  直接解析失败。
- `-FilePattern` 必须含 safetensors/npz，否则训练产物拉不回来。
- 连续 push 同一 kernel 时，驱动脚本第一拍可能吃到**上一跑的陈旧 COMPLETE**
  提前退出拉回旧产物——以 `python -m kaggle kernels status <ref>` 为准；
  提前退了就用 `-PullOnly` 补拉。
- 产物在 `output/anima-tpu-train/out/`（该目录 gitignore，不进仓库）。
- 配额：TPU **20h/周**（`python -m kaggle quota` 查）；脚本 push 前后各打一次。

## 断点接棒（长训 / 12h session 上限）

`--save-state-every N`（extra-args 或 yaml `save_state_every`）存**完整**优化器
状态 + 自适应采样器 EMA；`resume_state` 接回。只存权重不接 state 的话 AdamW
一二阶矩与自适应 burn-in 都要重来——续训前几百步等于换了优化器。产物拉回后
传成新的 dataset version 挂上，就能跨 session 接。

## 工作点选择（布局探针真机实测账目）

| | A（默认） | B（快档） |
|---|---|---|
| 配置 | budget 131072(8×16384) + scan + full remat | budget 65536(8×8192) + grad_accum 2 + `--unrolled --packed-barrier --remat every2` |
| 吞吐 | 16.9k 真tok/s（MFU 13.4%） | 28.9-30.2k 真tok/s（~1.8×） |
| 适用 | 单图 token ≤ 16384 的数据集 | 单图 token ≤ 8192 的数据集 |
| 编译 | 9-20s/布局 | 56-83s/布局 |

两者有效 batch 相同（B 用 grad_accum 2 补回 131072），是同一个实验。
**单图最大 token 超过 8192 的数据集只能选 A**（那种图在 B 的 8192/卡 预算下
根本装不进一个 pack，打包器会 fail-fast）。B 的 packed-chunk 变体快 4%
但 bf16 梯度有 ULP 漂移，barrier 变体 fp32 对拍逐 bit。

## 实测参考（142 样本的 anima 数据集，工作点 A）

20 epoch / 148 步全程 29.4 分钟（含 14 种布局首次编译），稳态 6~9s/步，
单次 TPU 消耗 ~0.5h 配额。checkpoint 约 7~8 步一份（epoch 边界），32MB/份。
作为外推：这个数据集规模下"再训 10 倍"（200 epoch）约 5h，一周配额够跑 3~4 次。

## Krea2 的差异点（`model_family: krea2`）

整体流程不变（同一份 yaml、同一条缓存→上传→build→push 管线），差别只在：

- **① 文本缓存两条路**：
  - **推荐：TPU 现算**。本地只跑 `tools/dump_caption_ids.py --data-dir "$DATA"
    --tokenizer "$MODELS/text_encoders/Qwen3-VL-4B-Instruct" -o caption_ids.npz
    [--empty-caption]`（只读 tokenizer，不加载权重），产物约 40KB；
    然后 `build_job.py --caption-ids caption_ids.npz
    --env ANIMA_TE_PATH=<文本塔的 Kaggle Dataset 挂载路径或 HF URL>`。
    textfeat 由真机上的 `jax_tpu/text_cache.py` 现算（跑在子进程里，退出后 HBM
    全部交回，之后才加载 12B 底模）。
  - **回退：本地算**。`tools/cache_text_features.py --model-family krea2
    --text-encoder <Qwen3-VL-4B-Instruct 目录>`（不需要 `--transformer`）。
    产物键 `txt` [L,12,2560] **变长**，约 61.4KB/token —— 96 张就 1.7GB，
    上传前留意大小。这也是为什么推荐上一条。

  latent 缓存与 Anima 完全同一条命令（同一个 VAE）。
- **② yaml**：`model_family: krea2` + `krea2_res_shift: true`（默认；开着时
  `schedule_shift` 必须 1.0，构造期 fail-fast）。模板
  `config/train_krea2_tpu_template.yaml`。预算语义不变（全局 = 8×单卡），
  但 K2 的段含 text 槽 —— **单图 image+text 合计 ≤ 单卡预算**。
- **④ 底模**：两条路，**优先挂 dataset**（下面 a），流式（b）留作回退。

  **(a) 挂 Kaggle Dataset（推荐）**：权重预先传成 dataset，训练时直接从
  `/kaggle/input` 读本地文件 —— 无网络依赖、无 HF gated/token、无 302 CDN 抖动，
  且 26GB 只读挂载不占 `/kaggle/working` 的 ~20GB 配额。

  自己传的话：从 HF `krea/Krea-2-Raw` 拉 `raw.safetensors`
  （26,283,332,608 字节，全文件 sha256 =
  `f99bb0ff8e362b77342bc4994e0c50906fe7ef7074864b181b7d48d2fa6d03d7`
  —— 这个值等于 HF 上那份的 git-lfs 指针 oid，可以直接核对），
  传成自己的 dataset，`dataset_sources` 挂上，然后

  ```bash
  --env ANIMA_TRANSFORMER=/kaggle/input/<你的 slug>/<文件名>.safetensors
  ```

  **不要**再传 `--hf-model/--hf-stream`（传了会覆盖成 URL 又走回流式）。
  `_read_safetensors_map` 的非 URL 分支逐 tensor 读本地文件 + 即刻分片，
  host 同样不持全量，数值与流式完全等价。**注意 Kaggle 侧那份没有独立校验**，
  靠加载器的尺寸闸门兜底。挂载路径两种布局并存的老问题由 job 里的
  `_resolve_input` 兜底：路径不存在时按 basename 在 `/kaggle/input` 下搜，
  唯一命中就自动改用并打 WARN，否则 fail-fast 并列出实际树（不再是加载器抛一个
  看不懂的 FileNotFoundError）。

  **(b) HF 流式（回退）**：`--hf-model krea/Krea-2-Raw:<文件>[:<rev>] --hf-stream
  --env HF_TOKEN=<只读token>`。**gated** —— 先在 HF 网页接受协议。
  raw.safetensors 26.3GB > /kaggle/working 的 ~20GB 上限，直下装不下；
  `--hf-stream` 让 krea2_jax 走 HTTP Range 流式加载（8 线程预取，真机
  ~110MB/s，零磁盘占用，字节与文件逐 bit 相同，`jax_tpu/tests/check_http_range.py`
  对拍）。token 用 `--env` 烘焙（Secrets 的网页 attach 对 script kernel
  不可靠：未 attach 时服务回 HTTP 400，被误报成 "ConnectionError"）；
  build_job 对凭证类 env 的日志打印已脱敏，生成物不进 git。
  job 内 FSDP 分片加载（每卡 ~3GB），host 不持全量。
- **HBM**：真 12B 训练图（LoKr+aux 全开）单 executable 驻留 ~12G/布局，
  16384/卡 与 12288/卡 都 OOM 过——工作点 `navit_token_budget: 81920`
  （8×10240/卡）+ K2 单布局驻留（换布局驱逐旧 executable，run_train.py）。
  spmd_probe 的 16384/卡账是裸模型探针口径，别拿来规划真训练。
- **⑤ plan-only**：输出里布局行会带 `txt(...)`（各实段 text 槽长）。
- **真机约束**：remat 不能 none（FSDP all_gather 要在 checkpoint 内）；
  `--unrolled/--packed-chunk/--packed-barrier` 未在 K2 验证，构造期拦。
- **闸门**：改 `jax_tpu/` 后先跑 K2（`check_train_loop_k2.py`，本仓库独立可跑）
  再推；K1 需要 upstream + torch，见 `jax_tpu/tests/README.md`。
