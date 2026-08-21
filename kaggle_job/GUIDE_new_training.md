# 启动一次新的 TPU 训练（新数据集 / 新参数）

给项目负责人与后来的 agent：这是 Kaggle TPU v5e-8 上跑 Anima LoRA/LoKr 训练的
**完整启动流程**。前置知识（探针结论、架构裁决、配额约束）在 `kaggle_job/README.md`，
本文件只讲"怎么操作"。所有命令都在仓库根目录的 Git Bash 里跑（Windows）。

## 0. 心智模型

TPU 后端（`AnimaLoraToolkit/jax_tpu/`）**只跑 DiT，不做任何编码**。所以一次新训练
的准备顺序永远是：

```
图片+标签 ──①──> latent/textfeat 缓存（本地 PyTorch，一次性）
                ② yaml 适配（同一份 yaml 驱动 GPU/TPU 两个后端）
                ③ 缓存上传 Kaggle Dataset（私有）
                ④ build_job 打单文件脚本（底模走 HF 直下，不用上传）
                ⑤ 本地 --plan-only 验证（零配额）
                ⑥ push 跑（kaggle_run.ps1）→ 拉回 checkpoint
```

本地两个 python 的分工（写死，别搞混）：
- torch 活：`D:\ArtificialIntelligence\ComfyUI-aki-v1.5\python\python.exe`
- jax 活：`D:\ArtificialIntelligence\animafix\jaxenv\Scripts\python.exe`（jax 0.11.0，
  与 Kaggle 真机钉同一版本；真机由 job  preamble 自动升级到这个版本）

## ① 数据缓存（新数据集必做，旧缓存可复用）

数据集目录 = 图片 + 同名 `.txt` 标签（tag 风格，trigger 词自己定）。放进目录后：

**文本特征**（CPU 即可，几分钟；`--empty-caption` 在 `caption_dropout_rate > 0` 时
**必须**带，否则 TPU 侧 fail-fast）：

```bash
cd AnimaLoraToolkit
"/d/ArtificialIntelligence/ComfyUI-aki-v1.5/python/python.exe" tools/cache_text_features.py \
  --data-dir "D:/Datasets/<名字>" \
  --text-encoder "D:/ArtificialIntelligence/animafix/anima-lora-train/AnimaLoraToolkit/models/text_encoders/Qwen3-0.6B-Base" \
  --transformer "D:/ArtificialIntelligence/ComfyUI-aki-v1.5/models/diffusion_models/anima-base-v1.0.safetensors" \
  --device cpu --empty-caption
```

**图像 latent**（GPU；`--ms-ladder` 必须与 yaml 的 `navit_multiscale_token_ladder`
一致；yaml 开 `flip_augment` 时**不要**加 `--no-flip`）：

```bash
"/d/ArtificialIntelligence/ComfyUI-aki-v1.5/python/python.exe" tools/cache_latents.py \
  --data-dir "D:/Datasets/<名字>" \
  --vae "D:/ArtificialIntelligence/ComfyUI-aki-v1.5/models/vae/qwen_image_vae.safetensors" \
  --ms-ladder 4096
```

8GB 显存本机的默认档是全分块编码（tile 640）：**分块缝处是羽化近似**（实测均值
0.46%、缝列误差约为其他列 2.2 倍，加大 overlap 无改善）。要完全精确就把
`--tiled-threshold` 抬到 4000000 并在大显存机器（≥24GB）上跑——或复用云端
（DCU/超算）训练前缓存阶段产出的 npz。0.46% 量级对 LoRA 风格训练可忽略，
但"近似"二字已如实记录，取舍由你。

**自查清单**（缺任何一样，TPU 侧会 fail-fast 且说得很难听，照做就是）：
- 每张图有 `<stem>.npz`（含 `latent`，flip 开时还要 `latent_flipped`）
- 每张图有 `<stem>.textfeat.npz`（含 `cross` [512,1024]）
- 开了 multiscale 就有 `<stem>.ms<档>.npz` sidecar
- `caption_dropout_rate>0` 时有 `_empty.textfeat.npz`

## ② yaml 适配（换参数在这里）

从一份现成 yaml（如 DCU 版）**逐字段照抄**，只动该动的。TPU 侧
`jax_tpu/config.py` 把每个键强制归为四类，**不在任何一类里的键直接报错**
（fail-fast，不静默忽略）——所以删键比改键重要：

- **必须删掉的后端专属键**：`device_backend`、`navit_attn_backend`、`fast_model_init`、
  `base_quant`、`wandb_*`、整族 `sample_*`（TPU 无训练期采样）、`save_state_every_epochs`。
- **budget 规则**：`navit_token_budget` 在 TPU 解释为**全局**预算（8 卡纯 DP），
  必须被 8 整除；单卡 = 全局/8，v5e 实测上限量级是 16384/卡（32768 OOM）。
  它只是 **FFD 装箱的容量上限**：每个 pack 实际补齐到自己的自然长度
  `round_up(Σ实段, 1024)`，不再补齐到 budget。
  **硬约束：数据集里单图最大 token 数必须 ≤ 单卡预算**，否则打包器 fail-fast
  （"单段 N > budget"）。token 数 ≈ (W//16)×(H//16)。ashima 数据集最大 16268 →
  只能 16384/卡（这就是"工作点 B" 8192/卡 对本数据集不成立的原因）。
- **开了会报错的未移植开关**（留着 false/0/空 就没事）：`sample_every>0`、
  `tread_enabled`、`dispersive_enabled`、`aux_perceptual_enabled`、`noise_offset>0`、
  `pyramid_noise_iterations>0`、`shuffle_caption`、`lora_dropout>0`、
  `resume_lora`（续训用 `resume_state`，它连优化器矩一起存）、`dora_export_mode`
  非 native。
- yaml 里的路径写**本地值**；上云路径由 build_job 的 `--env` 烘焙覆盖，yaml 不动。

改完先过一道（jaxenv）：`python -c "import yaml,config; config.build(yaml.safe_load(...))"`
的报错信息会直接告诉你哪个键有问题。

## ③ 上传数据集

```bash
mkdir /d/Datasets/_staging && cp -r "/d/Datasets/<名字>" /d/Datasets/_staging/<slug>
cd /d/Datasets/_staging/<slug>
# 写 dataset-metadata.json：{"title":"<slug>","id":"<用户名>/<slug>","licenses":[{"name":"other"}]}
PYTHONUTF8=1 python -m kaggle datasets create -p .
```

注意：
- 上传的是**私有** dataset（默认）。**两种挂载布局并存（2026-08-20 实测）**：
  早先传的 ashima 缓存在 `/kaggle/input/datasets/<用户名>/<slug>/`，
  新传的 krea2-tiny-* 却在 `/kaggle/input/<slug>/` —— 同一个 kernel 里两种都在。
  推之前拿不准就用 `kaggle_job/input_probe/` 零配额探一下（把 dataset_sources
  挂上后跑一遍，它会打印 /kaggle/input 的实际树）。
- `datasets status <用户名>/<slug>` 显示 **ready 之后**才能 push kernel，
  否则挂载不到（白烧约 1 分钟 TPU）。
- PNG 必须一起传：TPU 扫描靠图片文件发现 stem（像素从不被读）。若在意隐私，
  用 `tools/dataset_encrypt.py`（像素洗牌 + 剥 caption 明文）。

## ④ 打包 job

```bash
cd kaggle_job/anima_train
MSYS_NO_PATHCONV=1 python build_job.py \
  --config ../../AnimaLoraToolkit/config/private/<你的>.yaml \
  --env ANIMA_DATA_DIR=/kaggle/input/datasets/<用户名>/<slug> \
  --env ANIMA_OUTPUT_DIR=/kaggle/working/out \
  --hf-model circlestone-labs/Anima:split_files/diffusion_models/anima-base-v1.0.safetensors:f7382c4bf9d7ffe4ceea593a0adbb470c56dd79b \
  [--extra-args "--max-steps 40"]
```

- `MSYS_NO_PATHCONV=1` **必须**（否则 Git Bash 把 /kaggle/... 改成 C:/Program Files/...）。
- 底模走 HF 直下：Kaggle→HF 实测 10~13s 拉 4.2GB；revision 钉死，且 HF 文件
  sha256 已与本地对拍用权重核对一致（bd43b7cf…）。换底模就换 repo:文件:revision。
- 每次改了 `jax_tpu/*.py` 或 yaml 都要**重跑 build_job**（生成物不进 git，
  生成物过期 = 把旧代码推上去）。
- kernel-metadata.json 的 `dataset_sources` 要挂上 `<用户名>/<slug>`（改一次即可）。

## ⑤ 本地 --plan-only（零配额，必做）

```bash
cd AnimaLoraToolkit/jax_tpu
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  "/d/ArtificialIntelligence/animafix/jaxenv/Scripts/python.exe" \
  run_train.py --config ../config/private/<你的>.yaml --plan-only
```

看三个数（决定 8 卡用没用完）：
- **填充率**（线性层算力利用率上界，分母是自然总长；<90% 说明量化浪费大，
  调 quantum。"容量利用率"是分母为 budget 的对照口径，看装箱容量浪不浪费）
- **成步率**（能凑齐 8 个同布局 pack 的比例；小数据集天然低，顺延是设计行为）
- **布局数**（= 全模型编译次数；爆炸就调大 `--quantum`）

agent 注意：改了 `jax_tpu/` 任何代码，先把 `jax_tpu/tests/` 的闸门跑完再推
（见 tests/README.md；对拍是两步式，先 torch 解释器 dump 再 jaxenv check）。

## ⑥ 推送 / 拉取

```bash
cd kaggle_job
pwsh.exe -NoProfile -ExecutionPolicy Bypass -File "$PWD/kaggle_run.ps1" \
  -Username <用户名> -Accelerator TpuV5E8 -TimeoutSec 10800 \
  -JobDir anima_train -FilePattern '(.*\.safetensors|.*\.npz|.*\.json|.*\.log)$'
```

- **必须 pwsh（PS7）**：Windows PowerShell 5.1 把无 BOM 的 UTF-8 当 GBK 读，
  直接解析失败。
- `-FilePattern` 必须含 safetensors/npz，否则训练产物拉不回来。
- 连续 push 同一 kernel 时，驱动脚本第一拍可能吃到**上一跑的陈旧 COMPLETE**
  提前退出拉回旧产物——以 `python -m kaggle kernels status <ref>` 为准；
  提前退了就用 `-PullOnly` 补拉。
- 产物在 `kaggle_job/output/anima-tpu-train/out/`（该目录 gitignore，不进仓库）。
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
**ashima 数据集（最大 16268 token）只能选 A**。B 的 packed-chunk 变体快 4%
但 bf16 梯度有 ULP 漂移，barrier 变体 fp32 对拍逐 bit。

## 实测参考（ashima-anima2，142 样本，工作点 A）

20 epoch / 148 步全程 29.4 分钟（含 14 种布局首次编译），稳态 6~9s/步，
单次 TPU 消耗 ~0.5h 配额。checkpoint 约 7~8 步一份（epoch 边界），32MB/份。
作为外推：这个数据集规模下"再训 10 倍"（200 epoch）约 5h，一周配额够跑 3~4 次。

## Krea2 的差异点（`model_family: krea2`）

整体流程不变（同一份 yaml、同一条缓存→上传→build→push 管线），差别只在：

- **① 文本缓存换工具档**：`tools/cache_text_features.py --model-family krea2`
  （`--text-encoder` 指 Qwen3-VL-4B-Instruct 目录，不需要 `--transformer`）。
  产物键 `txt` [L,12,2560] **变长**，体积约为 anima 格式的 60 倍/token，
  上传 dataset 前留意大小。latent 缓存与 Anima 同一条命令（同一个 VAE）。
- **② yaml**：`model_family: krea2` + `krea2_res_shift: true`（默认；开着时
  `schedule_shift` 必须 1.0，构造期 fail-fast）。模板
  `AnimaLoraToolkit/config/train_krea2_tpu_template.yaml`。预算语义不变
  （全局 = 8×单卡），但 K2 的段含 text 槽 —— **单图 image+text 合计 ≤ 16384/卡**。
- **④ 底模**：`--hf-model krea/Krea-2-Raw:<文件>[:<rev>] --hf-stream
  --env HF_TOKEN=<只读token>`。**gated** —— 先在 HF 网页接受协议。
  raw.safetensors 26.3GB > /kaggle/working 的 ~20GB 上限，直下装不下；
  `--hf-stream` 让 krea2_jax 走 HTTP Range 流式加载（8 线程预取，真机
  ~110MB/s，零磁盘占用，字节与文件逐 bit 相同，tests/check_http_range.py
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
- **闸门**：改 `jax_tpu/` 后先跑 tests 的 K1 + K2 再推（见 tests/README.md）。
