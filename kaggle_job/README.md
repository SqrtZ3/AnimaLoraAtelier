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

## 真机第一跑的结论：libtpu 太旧，Pallas 整体被挡（非技术问题）

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
| `kaggle_run.ps1` | 本地驱动：改写 metadata → 看配额 → push → 轮询 → 拉产物 → 再看配额 |
| `splash_probe/kernel-metadata.json` | kernel 元数据（`kernel_type: script`，`id` 由驱动脚本填用户名） |
| `splash_probe/probe_tpu_splash.py` | 探针本体，自包含、无仓库依赖 |
| `output/<slug>/` | 拉回的产物（驱动脚本自动创建） |

## 后续（探针通过之后才做）

真正训练时的形态是：代码走 Kaggle Dataset，权重走 Kaggle Models，入口脚本只有几十行，
`dataset_sources` / `model_sources` 在 metadata 里挂上。产物写 `/kaggle/working`，
run 结束后拉回来传成新的 dataset version，作为下一棒的输入——12h 断点可以这样自动接棒。

真正的硬约束是 **20h/周**，接棒自动化解决不了它。
