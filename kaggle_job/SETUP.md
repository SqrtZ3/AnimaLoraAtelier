# 环境搭建

需要**两个** Python 环境，分工写死，别混用：

| 环境 | 干什么 | 关键依赖 |
|---|---|---|
| **torch 侧** | 数据缓存（VAE 编码图像、文本编码器算条件） | torch + transformers |
| **jax 侧** | `--plan-only` 预演、跑闸门 | jax 0.11.0（CPU 版够用） |

为什么分两个：jax 与 torch 的 CUDA 依赖会互相踩，装一起迟早出事。而两边本来
就不需要同时在场 —— 缓存是一次性的离线活，训练在 Kaggle 的 TPU 上跑。

真机上不需要你操心：job 脚本的 preamble 会在 `import jax` **之前**把 Kaggle 镜像
自带的 jax 升级到与本地同一版本（Kaggle 默认镜像的 libtpu 太旧，Pallas 整体被挡，
详见 `docs/ARCH_FINDINGS.md`）。

## torch 侧

```bash
python -m venv .venv-torch
. .venv-torch/bin/activate          # Windows: .venv-torch\Scripts\activate
pip install torch numpy pillow safetensors transformers einops sentencepiece pyyaml
```

- **GPU 强烈建议**：VAE 编码是这一侧唯一的重活。CPU 也能跑（本仓库的对拍就在
  CPU 上做的），只是慢。
- `einops` 是 VAE 实现（`tools/_vendor/wan_vae.py`）要的，`sentencepiece` 是
  T5 tokenizer 要的 —— 少了会在跑到一半时才报，先装上。
- 已经有一个装了 torch 的环境（比如 ComfyUI 自带的那个）就直接用，不必新建。

## jax 侧

```bash
python -m venv .venv-jax
. .venv-jax/bin/activate
pip install "jax[cpu]==0.11.0" numpy pyyaml safetensors
```

**版本钉在 0.11.0**，与真机 preamble 升到的那一版一致。目的是让本地 `--plan-only`
算出来的布局数、填充率、成步率与真机逐一对得上 —— 不然本地预演就失去意义。
本地只需要 CPU 版：`--plan-only` 不做真前向，闸门也都在 CPU 上跑
（8 卡分片用 `XLA_FLAGS=--xla_force_host_platform_device_count=8` 伪造）。

## 权重要哪些

| 用途 | 文件 | 谁需要 |
|---|---|---|
| latent 缓存 | `qwen_image_vae.safetensors` | torch 侧，必需 |
| anima 文本缓存 | Qwen3-0.6B-Base（HF 目录）+ T5 tokenizer + anima 底模 | torch 侧，anima 必需 |
| krea2 文本缓存 | Qwen3-VL-4B-Instruct（HF 目录） | torch 侧；走 TPU 现算时**只需 tokenizer** |
| 训练底模 | anima 4.2GB / krea2-raw 26.3GB | 真机上拉，本地不需要 |

两点省事的地方：

- **anima 文本缓存只读底模里的 `llm_adapter`**（118 个键 / 269MB），不是整个
  4.2GB。`tools/_vendor/llm_adapter.py` 干这件事，与走全量底模的输出**逐 bit
  相同**（已对拍：`torch.equal` 为 True，0.7s vs 42.2s）。
- **krea2 的文本特征建议让 TPU 现算**：本地只跑 `tools/dump_caption_ids.py`
  （只要 tokenizer，不加载权重），产物约 40KB；textfeat 在真机上算。
  相比本地算完上传 1.7GB/96 张，省掉 4 万倍的上行。

## Kaggle CLI

```bash
python -m pip install --upgrade kaggle
python -m kaggle auth login
```

凭据缓存在本机。之后一律用 `python -m kaggle`（pip 装完 `Scripts` 目录未必在
PATH 上）。账号需要**手机验证**才能开 internet（job 要联网升级 jax、拉底模）。

## Windows 额外三条（都是真踩过的）

1. **驱动脚本必须用 pwsh（PowerShell 7）**：`kaggle_run.ps1` 是无 BOM 的 UTF-8，
   Windows PowerShell 5.1 会按 GBK 误读中文注释，直接解析失败。
2. **`PYTHONUTF8=1`**：`kaggle kernels output` 用系统默认编码（中文 Windows 是
   GBK）写日志，遇到 `✗` 之类字符会崩，日志落成 0 字节。驱动脚本内部已强制设了，
   自己手动调 kaggle CLI 时要记得带。
3. **Git Bash 里跑 `build_job.py` 要加 `MSYS_NO_PATHCONV=1`**：否则它会把
   `/kaggle/input/...` 改写成 `C:/Program Files/...`，烘焙进脚本的路径全废。

## 验一遍装对了

```bash
# jax 侧：模板能过配置构造
<jax-python> -c "import sys,yaml; sys.path.insert(0,'jax_tpu'); import config; \
  config.build(yaml.safe_load(open('config/train_anima_tpu_template.yaml',encoding='utf-8')), \
               devices=8, canvas_hw=(128,128)); print('config OK')"

# jax 侧：闸门（几分钟，不需要权重）
cd jax_tpu/tests
XLA_FLAGS=--xla_force_host_platform_device_count=8 <jax-python> check_pack_invariants.py
XLA_FLAGS=--xla_force_host_platform_device_count=8 <jax-python> check_adapter_parity.py

# torch 侧：造两张噪声图跑一遍 latent 缓存，再校验
<torch-python> tools/cache_latents.py --data-dir <某个有图的目录> \
    --vae <qwen_image_vae.safetensors> --device cpu --ms-ladder ""
python tools/verify_cache.py <那个目录>
```

四条都过，环境就齐了。接着看 `README.md` 的「完整流程」。
