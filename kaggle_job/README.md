# Kaggle CLI 工作台 · TPU v5e-8

用 Kaggle 官方 CLI 当工作台，**全程不碰 notebook 网页界面**：代码在本地 git 里正常管理，
一条命令 push 上去后台跑，跑完把产物拉回来。

Kaggle 只能写 notebook 是**网页工作台**的限制，不是平台的限制——
`kernel-metadata.json` 里 `"kernel_type": "script"` 就能推纯 `.py`。

## 前置（只需做一次）

```bash
python -m pip install --upgrade kaggle
```

```bash
python -m kaggle auth login
```

凭据缓存在本机。之后所有命令都用 `python -m kaggle`（pip 装完 `Scripts` 目录未必在 PATH 上）。

## 三步走

**第 0 步 · 零配额验证 CLI 链路。** 不带加速器推一次，确认 push/轮询/取回整条链路通：

```powershell
.\kaggle_run.ps1 -Username 你的用户名 -Accelerator none -TimeoutSec 900
```

**第 1 步 · 上 TPU 跑探针。** 这一步才消耗 TPU 配额，预计十几分钟：

```powershell
.\kaggle_run.ps1 -Username 你的用户名 -Accelerator TpuV5E8 -TimeoutSec 2700
```

**第 2 步 · 读 `output/anima-tpu-splash-probe/tpu_probe_report.txt`**，对照下面的判读表。

**补救：只重新拉产物，不重跑、不花配额**（上次拉取失败时用）：

```powershell
.\kaggle_run.ps1 -Username 你的用户名 -PullOnly
```

## 真训练 job（`anima_train/`）

探针阶段结束后的正式入口。吃的是**与 GPU 侧同一份训练 yaml**，
`jax_tpu/config.py` 负责翻译，并对没移植的开关 **fail-fast**（不静默忽略）。

```powershell
cd anima_train
python build_job.py --config ..\..\AnimaLoraToolkit\config\private\train_tpu_ashima.yaml `
    --extra-args "--max-steps 40" `
    --env ANIMA_DATA_DIR=/kaggle/input/datasets/ilovebg/ashima-anima2-tpu-cache `
    --env ANIMA_OUTPUT_DIR=/kaggle/working/out `
    --hf-model circlestone-labs/Anima:split_files/diffusion_models/anima-base-v1.0.safetensors:f7382c4bf9d7ffe4ceea593a0adbb470c56dd79b
cd ..
.\kaggle_run.ps1 -Username 你的用户名 -Accelerator TpuV5E8 -TimeoutSec 7200 `
    -JobDir anima_train -FilePattern '(.*\.safetensors|.*\.npz|.*\.json|.*\.log)$'
```

`-FilePattern` **必须改**：默认值只拉 `*_probe*` 与 `*.log`，训练产物
（`.safetensors` / 优化器状态 `.npz`）会被过滤掉，跑完了却什么也拿不回来。

`build_job.py` 把 `jax_tpu/` 的 14 个模块 base64 进单文件脚本，运行时写回磁盘再
正常 import —— **不是**首尾拼接（模块互相带命名空间引用，拼接会静默覆盖同名
顶层函数）。脚本带源码 sha，本地跑过的代码与真机逐字节相同。

### Krea2 训练（`model_family: krea2`）

同一条流水线，yaml 换 `model_family: krea2` 即可（run_train 自动分派到 FSDP
路线）。三处与 Anima 不同的操作：

- **底模**：`--hf-model krea/Krea-2-Raw:<文件名>[:<rev>]` —— **gated repo**，
  先在 HF 网页接受 Krea 2 Community License，并把 token 配成 Kaggle Secrets
  的 `HF_TOKEN`（job 启动时自动读，不落盘）。12.16B bf16 ~24GB，Kaggle→HF
  直下，job 内 FSDP 分片加载（每卡 ~3GB）。
- **文本缓存**：`tools/cache_text_features.py --model-family krea2`
  （Qwen3-VL-4B 编码，键 `txt` [L,12,2560]，变长；约为 anima 格式 60 倍/token，
  注意 dataset 体积）。latent 缓存与 Anima 完全同一份（同一个 VAE）。
- **闸门**：改 `jax_tpu/` 后先跑 `tests/` 的 K1（krea2 前向对拍）+ K2
  （FSDP 训练闭环），全绿再推。详见 `AnimaLoraToolkit/docs/krea2-family.md`
  的「TPU 后端」节与 `config/train_krea2_tpu_template.yaml`。

### 权重与数据从哪来（2026-08-19 首秀定案的形态）

- **底模走 job 内 HF 直下**（`--hf-model <repo>:<文件>[:<revision>]`），不走 Kaggle
  Models：本机上传 4.2GB 太慢，而 Kaggle → HF 实测 **10~13 秒**拉完。revision 钉死，
  且 HF 文件的 X-Linked-ETag 已与本地那份的 sha256 核对一致（bd43b7cf…），
  对拍环境与真机权重逐字节相同。repo 是 public 非 gated，不需要 token。
- **数据缓存走 Kaggle Dataset**（私有），`dataset_sources` 挂进 metadata。
  latent / textfeat 两份缓存都由 PyTorch 侧离线产出：`tools/cache_latents.py`
  （图像侧，本仓库新增）+ `tools/cache_text_features.py`（文本侧，含
  `--empty-caption` 给 caption_dropout 用）。
- **挂载布局两种并存（2026-08-20 实测）**：dataset 可能出现在
  `/kaggle/input/datasets/<owner>/<slug>/`，也可能是 `/kaggle/input/<slug>/` ——
  同一个 kernel 里两种同时存在（ashima 缓存走前者、新传的 krea2-tiny-* 走后者）。
  路径用 `--env ANIMA_DATA_DIR=...` 烘焙进脚本，运行时 `_override_paths` 用它覆盖
  yaml 里的本地路径（yaml 本身不动，两个后端共用同一份）。写错路径的代价是
  一轮白跑 —— 拿不准就先跑 `input_probe/`（CPU、零配额）把 `/kaggle/input`
  的实际布局打出来。

### 首秀实测（2026-08-19，v5e-8，ashima-anima2，C12 全套开关）

40 步全过：loss 0.005~0.017 随图数正常波动、gnorm 0.001~0.01 无爆炸、
填充率 93.6~99.4%；稳态步时与布局探针的 16.9k tok/s 工作点吻合；
40 步 + 14 种布局的首次全量编译合计 ~9.6 分钟。LoKr+DoRA 存档键名
`lora_unet_blocks_N_*.lokr_w1/lokr_w2_a/lokr_w2_b/dora_scale/alpha`，
ComfyUI 可直接加载。两个坑都记在案：

1. **dataset 刚传完有处理窗口**：`datasets status` 变 ready 之前 push 的 kernel
   挂载不到它（FileNotFoundError），白烧 ~1 分钟 TPU。等 ready 再推。
2. **驱动脚本会吃到陈旧 COMPLETE**：同一 kernel 连续 push 时，轮询的第一拍可能
   读到上一跑的 COMPLETE 提前退出、拉回旧产物。看到"疑似失败但状态还在 RUNNING"
   时以 `kernels status` 为准，跑完用 `-PullOnly` 补拉。

**先跑 `--plan-only`**（本地即可，零配额）：它把配置摘要、数据集 token 分布、
打包报告（布局数/填充率/成步率）、适配器结构与参数量全打出来。三个数决定 8 卡
用得满不满 —— 填充率是线性层算力利用率的上界、成步率决定有没有卡空转、布局数
决定编译次数。

### 全局 token 预算怎么对账

yaml 的 `navit_token_budget` 在 GPU 上是一步一个 pack 的预算；TPU 是 8 卡纯 DP、
每卡一个 pack，所以它在这里被解释成**全局**预算，单卡拿 1/8。C12 配方的 131072
正好落成 **8 x 16384**，而 16384 恰是 v5e 单 chip(15.7GiB) 在 scan+full 档装得下的
量级（anima-mem-probe：32768 OOM）。于是"一步看多少 token"在两个后端上是同一个数，
梯度噪声量级可比。不整除时直接报错，不四舍五入。

## 架构裁决（2026-08-14，`arch_probe` 第二跑，v5e-8 真机）

块对角内核可行之后，这一轮问的是**整条路能不能走**：静态图（E）、显存（F）、
主机侧成本（G）。结论是**能走，但形态被钉死了**。

### F 显存：Krea2 12B 在 v5e-8 上放得下，但余量不大

| 项 | 实测 |
|---|---|
| Krea2 参数量 | 12.16B（每层 attn 132.1M + mlp 302.0M = 434.1M ×28） |
| bf16 权重 | 22.6GB → **单 chip(15.7GiB) 放不下，必须分片** |
| 8 卡分片实测 | 每片 2.83GB，`in_use 0.00→2.83GiB` → **放得下** |
| 单层前向+反向 | temp 2.54GB / 每层输入 0.19GB（编译产物静态分析） |
| 28 层 activation | 全梯度检查点 ≈7.8GB；无检查点 ≈71.1GB |

粗算单卡峰值 ≈ 2.83(权重分片) + 0.81(all-gather 当层) + 7.8(存的激活) + 2.54(temp)
**≈ 14GB / 15.7GiB**。**结论：必须 FSDP 分片 + 全层梯度检查点，且 L=16384 已经贴顶**，
想留余量只能降 `navit_token_budget`。注意 GPU 上靠 fp8/fp4 省显存那条路在 TPU 上不存在。

### E 静态图：必须走「补齐固定预算 + 编译期 mask + 布局缓存」

- **运行时 `jax.Array` mask 路线被否决**（原本最理想的"一个图打天下"）：
  - 显存上死：L=16384 的稠密 bool mask 单 head 256MB，48 head 共 12GB，
    真机报 `HLO temporaries 36.00G` → 放不下（1 head 可以）。
  - 性能上也死：**同 4 head 对照**下运行时 mask 实测比 0.574，编译期 mask 0.259
    （理论 0.250）→ 运行时 mask **只部分跳块**。
- **补齐到固定 token 预算是便宜的**：padding 自成一段，填充率 95% 时相对开销仅 **1.3%**
  （99%→0.1%，90%→5.1%）。这样张量 shape 恒定，XLA 只编译一次。
- **不等长段有 ~18% 的额外税**：真实 ragged pack（7 段不等长含 padding 段）实测比
  0.203 vs 理论 0.172，偏差 18.2%；4 等长段时偏差 <1%。仍是 ~5× 加速，可接受。

### G 主机侧：MaskInfo 预处理必须自己缓存

- Kaggle 主机 CPU 实测 **730–845ms/布局**（4/8/16 段几乎一样，与段数无关）。
- **`make_splash_mha` 内部无缓存**：同布局重复构造 736ms → 729ms。
- 对比注意力步时 288ms —— 不缓存的话每步多付 2.5 倍。**必须在 trainer 侧按布局缓存
  可调用对象。**

### 由此得到的 TPU 版 navit 形态

1. 每图 token 数**量化到有限档位**，pack **补齐到 `navit_token_budget`**，padding 自成一段。
2. 段长对齐 128（前一轮结论），按「段长组成」缓存 `make_splash_mha` 可调用对象。
3. FSDP 分片权重 + 全层梯度检查点。
4. 布局种类数（本地枚举，`token_budget=16384`）：

   | 量化档位 | 布局数 | 首次全量预热 |
   |---|---|---|
   | 8 档（1k/2k/3k/4k/6k/8k/12k/16k） | 535 | ~535s |
   | 5 档（2k/4k/6k/8k/16k） | 53 | ~53s |
   | 3 档（4k/8k/16k） | 9 | ~9s |

   **布局爆炸不成立**——最细的 8 档也只要一次性 9 分钟预热，之后进程内全命中。
   （注意 XLA 编译有持久化缓存可跨 run 摊掉，但 MaskInfo 是 host 侧 numpy，
   **不在编译缓存覆盖范围内**，每个新会话要重付。）

## 结论（2026-08-14，v5e-8 真机，20 OK / 0 FAIL）

**NaViT 的块对角打包在 TPU 上成立，跳块真实生效，提速与理论吻合到 1% 以内。**

| 测量 | 块对角 | 全通 | 实测比 | 理论 Σn^2/L^2 |
|---|---|---|---|---|
| T1 前向（L=16384，4 段，48q/12kv×128，bf16） | 106.81 ms | 423.20 ms | **0.252** | 0.250 |
| T2 前向+反向 | 287.95 ms | 1136.30 ms | **0.253** | 0.250 |
| T3 8 段 | 53.91 ms | 423.30 ms | 0.127 | 0.125 |
| T3 16 段 | 27.76 ms | 423.30 ms | 0.066 | 0.062 |

反向也跳块（T2），这对训练是必要条件。段数越多收益越大，且一直贴着理论线。

其余关键结果：

- **自举升级成功**：43 秒把 libtpu 从 2025-06-12 拉到 **2026-07-27**、jax 到 0.11.0，
  A1 报"闸门通过"。**Kaggle 默认镜像必须先升级才能用 Pallas。**
- **D2 通过**：运行时 `jax.Array` mask 数值正确 → pack 组成可完全动态，不必枚举布局。
- **C1 编译成本极低**：3 种布局各 0.2–0.3 秒（**仅** XLA/Mosaic 编译）。
- **C3 mask 预处理才是要小心的地方**：`make_splash_mha` 里的 MaskInfo 预处理是同步
  numpy，本地 CPU 实测 L=16384/48heads 约 **500 ms**，与 T2 的 288 ms 步时同量级。
  训练时若每个 pack 都重建 mask，这一项会直接叠到步时上。两条规避路径：
  ① 按段长布局缓存已构建的可调用对象；② 走 D2 的运行时 mask 路径。
- 数值精度：bf16 下 rel ~2e-3（N1–N4），fp32 interpret 下 ~1e-6，符合预期。

单次运行 157 秒，耗 0.04h 配额。

## 上一跑的结论：libtpu 太旧，Pallas 整体被挡（非技术问题，已解决）

2026-08-14 在 v5e-8 上跑（耗时 0.02h）。所有 splash/Pallas 探测以**同一个原因**失败：

```
RuntimeError: Pallas TPU requires a libtpu version that's at most a month old.
Found version string: ... TFRT TPU v5 lite ... Built on Jun 12 2025 ...
```

Kaggle 默认镜像：**jax 0.10.2 + libtpu 构建于 2025-06-12**，比闸门要求老约 14 个月。
闸门在 `jax/_src/pallas/mosaic/lowering.py` 的 `is_cloud_tpu_older_than(...)`，
**硬 raise，没有环境变量旁路**。与块对角/NaViT 本身无关。

**修法**（已写进本目录，下一跑生效）：`kernel-metadata.json` 设
`"enable_internet": "true"`（账号需手机验证），探针顶部 `BOOTSTRAP_UPGRADE_JAX=True`
会在 **`import jax` 之前** `pip install -U "jax[tpu]"`。必须在 import 之前——
jax 一旦初始化后端就换不掉 libtpu。失败不致命，会记 FAIL 后继续。

**这一跑仍然拿到的真实数据**：

| 项 | 真机结果 |
|---|---|
| 设备 | 8 × `TPU v5 lite`，platform=tpu |
| 单设备 HBM | **limit = 15.7 GiB**（印证官方 16 GB/chip） |
| B1 块压缩率 | **与本地完全一致**：0.2500/0.2500、0.1582/0.1582、partial=0 |
| C2 持久化编译缓存 | 可用，写入 `/kaggle/working/jax_cache` |
| D1 API 面 | `make_splash_mha` 接受 `jax.Array` mask；`process_dynamic_mask` 存在 |
| M1 8 卡分片 | 通过 |
| 环境 | Python 3.12.13 / Linux 6.6.143 / `KAGGLE_KERNEL_RUN_TYPE=Batch` |

新增探测 `A1` 专门诊断这道闸门（二分反推 libtpu 构建日），把十条一样的 stack trace
压成一行；裁决段也加了 `[ENV]` 分支，不会再把环境问题误报成"块对角不行"。

## 首次真机运行踩到的三个坑（均已修）

1. **Kaggle 把非零退出码判为 ERROR。** 探针原本在有 FAIL 项时 `return 1`，结果脚本明明
   跑完了全部探测，run 仍被标成 ERROR。现在 `main()` **恒返回 0**——单项 FAIL 是数据，
   不是脚本故障。
2. **中文 Windows 下拉日志会崩。** `kernels output` 用系统默认编码（GBK）写日志文件，
   遇到 `✗` 之类字符报 `'gbk' codec can't encode character`，日志落成 **0 字节**。
   现在驱动脚本强制 `PYTHONUTF8=1`，探针本体也已清掉所有 GBK 编不了的字符
   （`²`/`⇒`/`⚠` → `^2`/`=>`/`[!]`）。
3. **script kernel 拿不到命令行参数。** 它是被裸 `python probe_tpu_splash.py` 拉起的，
   所以 `--json` 永远不生效。现在探针检测到 `/kaggle/working` 存在就自动写
   `tpu_probe.json` + `tpu_probe_report.txt`，不依赖内核日志。

另：探针会把 jax 编译缓存也写进 `/kaggle/working`（几十个文件）。驱动脚本默认只拉
报告和日志；要连缓存一起拉，传 `-FilePattern ""`。

驱动脚本在 push 前后各调一次 `kaggle quota`，两次一减就知道本次实际消耗了多少配额，
也顺便回答"一次任务最多能跑多久"——`-TimeoutSec` 是我们自己设的上限，Kaggle 的全局
上限会在超出时体现在状态里。

## 探针要回答的唯一问题

NaViT 的块对角打包，在 TPU/XLA 上能不能保留它的核心收益——即注意力代价是 **Σn_i²**
（每张图各算各的）而不是 **L²**（整个 pack 当一整条稠密序列算）。

如果只是"算完稠密再用 mask 抹掉"，NaViT 在 TPU 上就没有意义。所以重点不是"能不能跑通"，
而是 **T1/T2 的跳块提速是否真实**。

## 判读表

| 探测 | 含义 | 期望 |
|---|---|---|
| `B1` | MaskInfo 块压缩率（纯 numpy） | 存活块/稠密块 == 理论 Σn²/L²，partial=0 |
| `N1–N4` | 数值/梯度/GQA 对拍 | 全 OK，rel < 1e-5（fp32） |
| **`T1`/`T2`** | **跳块提速（决定性）** | 实测比 ≈ 理论 Σn²/L²（4 段时 0.25） |
| `T3` | 段数扩展性 | 8/16 段时比值继续下降 |
| `C1` | 换段长布局的编译代价 | 每布局一次，看单次多少秒 |
| `C2` | 持久化编译缓存 | 命中 → 编译成本可跨 run 摊掉 |
| `D2` | 运行时 `jax.Array` mask | OK → pack 组成可完全动态，无需枚举布局 |
| `D3` | `segment_ids` 保底路径 | 数值对但**大概率不跳块**，只作语义保底 |

**最关键的一行**：`T1` 会直接打印 `跳块生效` 或 `疑似未跳块`，脚本末尾的裁决段会把它翻译成
"TPU 路线该怎么走"。

## 已经在本地验证过的（不必再花 TPU 配额重验）

用 CPU 版 jax 0.11.0 + Pallas `interpret` 模式跑过：

- **B1 块压缩率精确等于理论值**：段长对齐到 `block_kv=128` 时，`存活块/稠密块` 与
  `Σn²/L²` 相等到小数点后 4 位，`partial_mask_blocks` 为 `None`；
  故意不对齐则多算约 9%（0.1675 vs 0.1538）并产生 429 个 partial 块。
  → **段长必须对齐到 128**，这是硬要求。
- **N1–N4 全过**：块对角前向、**反向**、GQA(4:1，同 Krea2) 数值全部正确，rel ~1e-6/1e-7。
- **D2 通过**：`make_splash_mha` 的公开签名就是
  `mask: np.ndarray | jax.Array | MultiHeadMask`，接受运行时 `jax.Array` 且数值正确。
- **坑**：splash **不内置** `1/sqrt(head_dim)` 缩放，调用方必须自己预缩放 q。
  漏掉不会报错，只会静默改变 softmax 温度（本地实测 rel 差到 9.5）。

**B1 只证明 Pallas grid 变小了，不证明墙钟时间同比例下降**（可能有固定开销、流水线气泡）。
真机 T1/T2 才是裁决。

本地复现（不需要 TPU，几秒）：

```bash
python probe_tpu_splash.py --interpret --ref-len 512 --skip-slow
```

> 真机上**不要**加 `--interpret`——那会走 CPU 解释器，慢且计时无意义。

## 文件

| 路径 | 作用 |
|---|---|
| `kaggle_run.ps1` | 本地驱动：改写 metadata → 看配额 → push → 轮询 → 拉产物 → 再看配额。**用 `pwsh`（PS7）跑**：文件是无 BOM 的 UTF-8，Windows PowerShell 5.1 会按 GBK 误读中文注释直接解析失败 |
| `splash_probe/kernel-metadata.json` | kernel 元数据（`kernel_type: script`，`id` 由驱动脚本填用户名） |
| `splash_probe/probe_tpu_splash.py` | 探针本体，自包含、无仓库依赖 |
| `anima_train/` | 真训练 job：`build_job.py` 打包（`--env`/`--hf-model`/`--extra-args`），生成物 `anima_train_job.py` 不进 git |
| `input_probe/` | 零配额（CPU）诊断：`/kaggle/input` 实际挂载布局打出来，dataset 路径拿不准时先跑它 |
| `output/<slug>/` | 拉回的产物（驱动脚本自动创建） |

## 后续（探针通过之后才做）

~~真正训练时的形态是：代码走 Kaggle Dataset，权重走 Kaggle Models~~ **已落地，
见上面"真训练 job"**：代码 base64 内嵌进单文件脚本，权重 job 内 HF 直下（钉 revision），
数据走私有 Kaggle Dataset。产物写 `/kaggle/working`，run 结束后拉回来传成新的
dataset version，作为下一棒的输入——12h 断点可以这样自动接棒。

真正的硬约束是 **20h/周**（本周已用 ~1.4h，含探针与首秀），接棒自动化解决不了它。
