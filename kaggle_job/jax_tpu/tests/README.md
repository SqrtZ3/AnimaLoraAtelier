# `jax_tpu/tests` —— TPU 路线的闸门

改动 `../*.py` 或 job 目录之后，**先在这里跑完再推 TPU**。
真机一轮几十分钟 + 配额，这里几分钟，能挡掉绝大多数错。

## 两类闸门：本仓库自足的 vs 需要 upstream 的

本仓库对 upstream（`anima-lora-train/AnimaLoraToolkit`）**零运行时依赖** ——
训练、打包、导出、plan-only 全都自足。但有一类闸门天然跨仓库：**parity 对拍**，
它要证明"JAX 实现 ≡ PyTorch 实现"，必须同时看到两边。

那类闸门是**两步式**的：

    <torch-python>  dump_*.py     # 需要 upstream + torch，产参考量 npz
    <jax-python>    check_*.py    # 只读 npz，本仓库独立可跑

**参考量小的几份已随仓库分发**（`_ref/` 里约 1.7MB），所以 ④⑫⑬K3 的 check 侧
开箱即跑。大的几份没法带（`export_ref.npz` 503MB / `anima_ref.npz` 37MB /
`optim_ref.npz` 9MB / K1 的 `tiny.safetensors` 11MB），要跑 ①⑤⑥K1 得自己 dump：

```bash
export ANIMA_UPSTREAM=/path/to/anima-lora-train/AnimaLoraToolkit
<torch-python> dump_torch_ref.py --ckpt <anima 底模>     # ①
```

没设 `ANIMA_UPSTREAM` 时 `dump_*.py` 会 fail-fast 说清要什么（`_upstream.py`），
不会抛一个看不懂的 `ModuleNotFoundError: trainer`。

| 闸门 | 脚本 | 证明什么 | 独立可跑 | 需要 |
|---|---|---|---|---|
| ① 数值对拍 | `dump_torch_ref.py` → `check_jax_parity.py` | JAX 前向 ≡ PyTorch | ✗ 要 dump + 真底模 | torch + jax + upstream |
| ② 块对角 / 两级 mask | `check_splash_blockdiag.py` | 打包路线的注意力 ≡ 稠密参考 | ✓ | jax |
| ③ scan ≡ 展开 | `check_scan_equiv.py` | `lax.scan` 路径与逐块展开等价 | ✓ | jax |
| ④ 训练目标 | `check_flow_parity.py` | Flow Matching ≡ upstream `trainer/objective.py` | ✓ **参考量已带** | jax |
| ⑤ 优化器 | `check_optim_parity.py --dump` → 同名 | AdamW ≡ `torch.optim.AdamW` | ✗ 参考量 9MB | torch + jax |
| ⑥ 导出 | `check_export.py --write` → 同名 | LoRA 文件 ≡ 键名规范 | ✗ 参考量 503MB | torch + jax |
| ⑦ 端到端（打包） | `check_train_loop.py` | 打包路线能学 + 8 卡分片走得通 | ✓ | jax |
| ⑧ 打包枚举 | `enum_pack_layouts.py` | 布局数不爆炸、填充率够高 | ✓ | 纯 stdlib |
| ⑨ **布局等价** | `check_ragged_equiv.py` | `forward_ragged` ≡ `forward_packed` | ✓ | jax |
| ⑩ **端到端（分桶）** | `check_train_loop_ragged.py` | 分桶路线能学 + 8 卡分片走得通 | ✓ | jax |
| ⑪ **全功能训练步** | `check_train_full.py` | 一整套开关同时开时**每条都真的接上了** | ✓ | jax |
| ⑫ 适配器对拍 | `dump_adapter_ref.py` → `check_adapter_parity.py` | LoKr/DoRA/adamw_snr ≡ upstream `trainer/lora.py` | ✓ **参考量已带** | jax |
| ⑬ 目标函数对拍 | `dump_objective_ref.py` → `check_objective_parity.py` | Huber(snr)/逐图归约/Eisbach/spectral ≡ upstream | ✓ **参考量已带** | jax |
| ⑭ **打包不变量** | `check_pack_invariants.py` | 自然长度打包的 RNG 双射 / 布局数方向 / 结构不变量 | ✓ | 纯 numpy |
| ⑮ **无图缓存** | `check_cache_scan.py` | 缓存不含图片时样本集逐条不变 + sidecar 不当样本 | ✓ | 纯 numpy |
| **preflight** | `enum_quantum_advisor.py --config <yaml>` | 用这次的 yaml+数据集扫出该传的 `--quantum` | ✓ | numpy + pyyaml |
| **K1** Krea2 前向对拍 | `dump_krea2_ref.py` → `check_krea2_parity.py` | `krea2_jax` ≡ upstream `models/krea2_modeling.py` | ✗ 权重 11MB | torch + jax + upstream |
| **K2** Krea2 端到端 | `check_train_loop_k2.py` | FSDP 分片训练闭环能学 + 分片≡全量 + 导出键名 | ✓ | jax |
| **K3** 文本塔前向对拍 | `dump_qwen3vl_ref.py` → `check_qwen3vl_parity.py` | `qwen3vl_te` ≡ HF `Qwen3VLTextModel` | ✓ **参考量已带** | jax |
| **K3-B** 文本塔真权重 | `check_qwen3vl_real.py` | 真权重重算的 `textfeat` ≡ 已缓存那份 | ✗ 要真 TE | jax + 真 TE |

**一条命令跑完所有独立闸门**（约十几分钟，不需要 torch / upstream / 任何权重）：

```bash
cd jax_tpu/tests
export XLA_FLAGS=--xla_force_host_platform_device_count=8
for t in check_splash_blockdiag check_scan_equiv check_ragged_equiv \
         enum_pack_layouts check_pack_invariants check_cache_scan \
         check_adapter_parity check_objective_parity check_flow_parity \
         check_qwen3vl_parity check_train_loop check_train_loop_ragged \
         check_train_full check_train_loop_k2; do
  printf '%-28s ' "$t"; <jax-python> $t.py >/dev/null 2>&1 && echo OK || echo FAIL
done
```

另有两条**数据缓存侧**的闸门在 `tools/`（不在本目录）：

| 闸门 | 脚本 | 证明什么 | 需要 |
|---|---|---|---|
| 摘录等价 | `tools/tests/check_cache_parity.py` | 本仓库缓存工具产物 ≡ upstream 产物（**逐 bit**） | torch + upstream + VAE |
| 摘录同步 | `tools/check_sync.py` | `_vendor/` 与 upstream 当前内容无漂移 | 纯 stdlib + upstream |


K1/K2 是 **Krea2 模型族**（`model_family: krea2`，单流 MMDiT 12B + FSDP 权重分片）
引入的。K1 用小构型随机权重（结构 parity 不需要 24GB 真权重），逐 tap 比对
txtfusion/txtmlp/first/t_vec/逐块/最终输出，再挂合成 LoRA 比对四个区域
（blocks/lw/rf/单例）的注入点；fp32 基线 rel ≤ 4e-6。K2 在 8 个伪造 CPU 设备上
跑 shard_map + all_gather 的 FSDP 路径：T0 分片≡全量逐设备对拍（fp32 实测
2.2e-07；bf16 下判据放宽到 5e-2 —— 拼权错误是 O(1) 量级，两种噪声差着
数量级）、T1 step-0 中立、T4/T6 图像与文本**两级**填充都不参与 loss
（K2 文本是变长的，填充全靠精细 segment_ids 隔离）、T2 真的在学、
T5 存取往返 + 导出键名（`lora_unet_blocks_0_attn_wq` / `lora_unet_tproj_1`
等 torch 模块路径）。

K3 是把 **Qwen3-VL 文本塔搬上 TPU** 引入的（`jax_tpu/qwen3vl_te.py` +
`text_cache.py`）：krea2 的 `.textfeat.npz` 是 12 层 hidden 堆叠、61.4KB/token，
96 张就 1.7GB，而信息源只有 96 条 caption（0.14MB）。移植后本地只 tokenize
（`tools/dump_caption_ids.py`，实测 **39.8KB**，压缩比 48658×），textfeat 在真机现算。

K3 用小构型随机权重判**结构**（fp32 rel ≤ 5.3e-7），顺带把三条"猜错就静默全错"的
口径**测**成断言而不是留在注释里：
  * `hidden_states[k]`（k<层数）**= 第 k 层的输入**、`hidden_states[层数]` 才是
    final norm 输出 —— 于是 `KREA2_SELECT_LAYERS` 最大 tap 35 意味着
    **只跑 0~34 层**，`layers.35.*` 与 `norm.weight` 一个字节都不用加载；
    tap 等于层数时 fail-fast（T7）。
  * mrope 在纯文本输入下三路 position 相同，退化为标准 RoPE（T1 逐元素比 cos/sin）。
  * **padding 摆法**：`position_ids` 用 `arange`、**不看 attention_mask**，所以
    `_encode_krea2_batch` 在 `max_length<=0` 下把 suffix 拼在 padding 之后的那种
    "中段 padding"会改条件 —— torch 侧探针实测 max|Δ| = **6.7e-1**；右侧 padding
    才是逐 bit 等价的摆法（探针 0.0）。已落盘的缓存是 B=1 无 padding 口径，
    jax 侧一律按右侧 padding 摆。

K3-B 判**口径**：拿本地已有的 96 份 `.textfeat.npz`（torch CUDA bf16 产出）当参考，
不需要再跑一次 torch。判据不设绝对阈值，而是同时量 A=JAX(bf16) vs 缓存、
B=JAX(fp32 计算) vs 缓存、C=JAX(bf16) vs JAX(fp32)，**A 不该比 B+C 大一个数量级**
（实测 A/(B+C)=0.43，‖Δ‖/‖ref‖≈1.1e-2 = bf16 本身的噪声，逐层平坦无离群层）。

⑮ 守的是一条**外部平台**约束：TPU 侧从不读像素（`_stems` 只取文件名），
所以缓存 dataset 里放原图是纯多余的暴露 —— 2026-08-21 真实吃过一次，
`jan-krea2-tpu-cache` 连 96 张原图一起传上 Kaggle，被按 NSFW 条款**整个删掉**
并向账号告警。现在 `CacheDataset` 无图时按 `<stem>.npz` 推 stem，上传目录只放
npz。它同时守 sidecar 不被当成样本本体 —— `a.textfeat.npz` 若被当样本会报一个
假的「缺 textfeat」，`a.ms4096.npz` 被当样本则让副本数两遍、**训练集凭空变大
而日志上看不出来**。

⑭ 是「pack 补齐到自然长度」引入的（`packing.PACK_Q`）。它守的四条都是**不报错的
失效**：RNG 兼容重排错位会让新旧 loss 曲线悄悄对不上，去重逻辑坏掉会白编译一次
全模型。它同时把一个**被写错的结论**钉成回归：commit 1b12432 声称「布局数不变」，
实际上旧口径的布局身份 `sorted(实段 ∪ {budget − Σ实段})` 是非单射的，正确的说法是
**新布局数 >= 旧布局数**（最小反例见脚本 T2a）。真实数据集上差几个要跑
`enum_quantum_advisor.py` 看，不能靠推理免测。

`enum_quantum_advisor.py` 不是闸门，是**每次开训前跑一遍的 preflight**。自然长度
打包之后装箱紧不紧已经不买算力了（每 epoch 总 token 与分组无关），`quantum` 成了
唯一剩下的 FLOPs 杠杆，同时又压着反向块档位、布局数、入步图率三个反向轴 ——
这三条的权衡是逐数据集的，必须扫。它直接调 `packing` 与 `data.CacheDataset`，
不复刻逻辑。

⑪⑫⑬ 是**功能移植**引入的（LoKr+DoRA、三峰/自适应 t、Huber、eisbach/ΔFM/spectral、
immiscible、逐块 reg_dims）。⑪ 的每条断言都在防一种"不报错的失效"：适配器键名对不上
会让梯度恒 0 而前向照跑；aux 开关接了但恒等于 0 也一样看不出来。所以它逐个关掉每个
aux 去看 loss 变没变，而不是只看"能不能跑完"。

⑪ 的 **T10 覆盖吞吐旋钮**（`run_train.py --unrolled --packed-chunk/--packed-barrier`）：
展开路径与 scan 路径必须同数学（fp32 下 barrier 逐 bit、chunk 1.8e-07），且
`--unrolled` 单开必须构造期报错 —— 朴素展开在真机 budget 16384 的 full 档就要 20.60G。
**T10 必须在 fp32 上判**：默认 bf16 前向下同一份代码梯度 rel 就有 7e-3，
那是舍入不是算法差异（同 README 下面"逐 bit 只在 fp32 下成立"那一节）。

⑫⑬ 的当前基线（fp32）：

| 项 | rel |
|---|---|
| LoKr kron-bypass / ≡ 物化 kron | 2.1e-07 / 1.2e-07 |
| DoRA `‖W+ΔW‖` 逐行范数 / 前向 | 6.4e-08 / 1.5e-07 |
| adamw_snr 6 步（plain/snr1.5/cautious/两者） | ≤ 5.2e-07 |
| Huber δ 三调度 / loss map | 0（逐 bit） |
| 逐图 masked 归约 / Eisbach | 9.5e-08 / 6.2e-07 |
| spectral（画布=原生网格） | 1.5e-07 |
| spectral（画布放大 2.7×，零填充+补偿） | **3.3e-02** |
| VeCoR crop+resize（图A 满画布 / 图B 画布内） | 9.0e-08 / 8.4e-08 |
| 三峰 t 的 19 个分位点 | 3.5e-03 |

**倒数第二行是唯一一处非精确移植，别把它读成 bug**：打包布局下每图的 (h,w) 是运行时量，
而 FFT 要静态形状，所以每图被散射进一个固定画布再做 FFT。零填充本身不是近似（补零后的
DFT 就是同一 DTFT 的更细采样，幅度谱与平移无关），差的只是 ortho 归一化分母（已按
`sqrt(N_画布/N_图)` 补偿）与频点采样位置。残差随"图/画布"的比例收敛到 0 —— 画布等于
原生网格时是 1.5e-07。小波那一支始终是精确的（只统计完全落在图内的 2x2 块）。

**VeCoR 裁剪+resize 一支的移植注记**（曾是"非等价移植"，已补正）：裁剪参数
(ratio/top/left) 是运行时**数值**而非形状 —— 采样坐标在 jnp 里现算、画布 gather
取四角双线性，全静态形状，所以不再需要"固定到通道乱序一支"的妥协。一个易错点
已写进 `auxloss.py:crop_resize_canvas` 的 docstring：坐标 clamp 必须在**裁剪框相对
坐标**里做（clip 到 [0, ch−1]）再平移 top/left，反了会让出界行插值到框外一行
（⑦ 的图 A 首行曾因此全错）。

⑨⑩ 是**布局裁决**引入的（NaViT 序列打包 vs 量化分桶，见 `../attention.py` 里
"分桶（批维）后端"一节）。⑨ 有两部分，当前基线都是 fp32 下**逐 bit**：
①–③ 分桶 ≡ 打包（整模 rel `0.000e+00`）、④ 打包的 chunk 化调制 ≡ 朴素
（chunk=32/64/128，max_abs `0.000e+00`）。三条路径共用 `anima_jax._forward_core`，
所以都自动继承 ①④⑤⑥ 的口径，不必对 PyTorch 再对拍。

**"逐 bit"只在 fp32 + 稠密注意力参考实现下成立**。bf16 下是舍入量级
（LoRA 梯度相对差 6.7e-5），且 loss 标量会有 ~1e-6 的差 —— 那是 XLA 归约融合
顺序不同，不是数学差异。再经 AdamW 的 `m/√v`（对近零梯度近似取符号）放大，
几步之内参数就有 O(lr) 分歧。**所以切换这些开关不能复现同一条轨迹**，
做 A/B 时别把这种分歧读成"算法变了"。

## 闸门的已知盲区（2026-08-22 一轮全面审查暴露的）

那一轮修的 5 条正确性问题里，**有 4 条是现有闸门结构上抓不到的**。记在这里，
是因为"闸门全绿"不等于"没问题"，下次审查别把绿灯当结论：

| 修的问题 | 为什么闸门抓不到 | 补了什么 |
|---|---|---|
| `auxloss.eisbach_weight` 掩码乘在 `exp` 之后 → 填充位能量 >88 时 `inf*0=NaN`，且 `valid=0` 拦不住（`weighted_mean` 用 `jnp.sum`），一步污染 fp32 master | ⑬ 的 Eisbach 用例是**全有效 mask**，永远走不到填充位那条路 | ⑬ 该加一个"含填充位、填充位 pred 幅度 12"的用例 |
| DoRA 的 `dora_scale` 被 `train._fwd_cast` 前身整树降 bf16 → step-0 中立性失效（输出 rel 1.7e-3）、幅度增量被量化到 0.62% 格子 | ⑦/K2 的 T1 比的是 **loss**（偏差仅 2e-7，远在阈值内），问题在**输出域** | T1 宜改成比输出/比 per-image 向量 |
| krea2 `txtfusion.projector`（`Linear(12→1)`）rank 被 cap 到 1 而 alpha 不跟着 → 该层 scale=32、其余 264 层 =1，等效 LR 差 32× | 没有任何闸门比较**逐层 scale** 与 PyTorch 侧 | ⑫ 宜加一条"全 target 的 scale 与 torch `LoRALayer.scaling` 逐层相等" |
| `anima_jax.AnimaConfig.eps_rms` 抄了 `RMSNorm.__init__` 的默认 1e-5，而实例化处是 1e-6 | ① 的 `t_emb` 分项 tol=1e-4，而偏差只有 9e-6；bf16 下又小于 1 个 ULP | ① 的 `t_emb` 阈值宜收到 1e-6 |
| `sched.sample()` 候选分桶多套了一层 `t_range` clip（PyTorch 只施加 schedule_shift） | **`sched.py` 完全没有对拍闸门** —— `AdaptiveTimestepSampler` 只在 ⑪ T9 被构造过，没有与 PyTorch 的数值对拍 | 缺一个 sched 对拍闸门 |

共性：**闸门比的量与 bug 所在的域不一致**（比 loss 而 bug 在输出域、比 fp32 而
bug 只在 bf16 显形、阈值宽于偏差量级），以及**没有闸门的模块**（`sched.py`）。

同一轮还修了几处"开关开着但什么都没发生"（`--packed-barrier` 不带 `--unrolled` 时被
静默忽略、`entropy_rate` 的 `/w(t)` 因为没人传 `loss_weight_fn` 而被跳过、
`navit_max_images_per_pack` 解析了但从未被消费）—— 这一类现在都是构造期 raise。

## 环境

torch 与 jax **装不进同一个解释器**（CUDA 依赖互相踩），所以对拍拆成两步、
经 `_ref/*.npz` 落盘交换。两个环境怎么建见仓库根的 `SETUP.md`。

约定的写法：本文档里 `<jax-python>` / `<torch-python>` 指你那两个 venv 的
解释器路径。**jax 侧版本钉 0.11.0**，与真机 preamble 升到的那一版一致 ——
本地/真机版本一旦漂移，"本地过了真机挂"会变成查不动的问题。

需要真底模的闸门（①K3-B）用 `--ckpt` 传，或设环境变量 `ANIMA_TRANSFORMER`
（与 job 侧同名，一处配置两处用）。

`_ref/` 里除了随仓库分发的那几份（见 `.gitignore` 的放行名单），其余都是生成物。

## 跑法

```bash
python check_splash_blockdiag.py check_scan_equiv.py check_ragged_equiv.py
```

（各脚本独立，逐个跑；两步式的在 docstring 里写了先 `--dump` / `--write`。）

⑦⑩⑪ 用 `XLA_FLAGS=--xla_force_host_platform_device_count=8` 在 CPU 上伪造 8 个设备，
于是 shard_map / 跨卡梯度 all-reduce / 每卡一份数据这些分支也验到了；
splash 走 `interpret=True`（Pallas 解释执行）。

## 判读

**① 对拍**：主判据是"整模 rel"，当前基线 **4.79e-05**（bf16 权重、28 块全过）。

逐算子里 `q`/`k` 有单独放宽的阈值（`STAGE_TOL`）：它们是 RMSNorm 之后的小幅值张量，
相对误差被放大约一个量级；判据是**下游是否收敛回来**。若 q/k 超标且下游**不再收敛**，
那才是真问题。逐块 rel 应稳定在 1e-5 量级、**不该单调放大**；某块突变 = 那块有结构错。

若"std 几乎相同但逐元素差很大"，多半是排列/通道顺序错——优先查
`output_tokens_to_patch_tokens` 和 q/k/v 的 head reshape。这类错历史上出过一次
（漏了 `(ph pw pt c)→(c pt ph pw)`，rel 1.4）。

**②⑨ 的"填充隔离"判据是行为判据，不是数值接近**：动填充区的输入，真 token 输出
必须**逐 bit 不变**（=0，不是"很小"）。两个脚本都带了**反证**（关掉隔离必须被污染），
防止判据因为构造错误而失去分辨力 —— 一个恒过的断言比没有断言更糟。

**⑪ 不带这个 flag 跑会假失败**：设备只有 1 个时 T6（逐图等权）的样本量掉到 1/8，
两次扰动的差从 1% 涨到 4%，撞破 2% 的容差。判据本身没问题，但它对样本量敏感 ——
看到 T6 单独红，先确认 flag 加了没有。

**⑦⑩ 里 T1/T3/T4 不能删**：

- T1 step-0 中立：B 零初始化 -> 接不接 LoRA，loss 必须逐 bit 相同
- T3 跨卡：改一卡数据参数必须变（否则 shard_map 没接对）
- T4 填充不参与 loss：改填充区 latent，loss 必须逐 bit 不变

另有一条同类断言在 TPU job 里（`grad_b` 非零 / `grad_a` 恒 0）。它存在的理由：
`jax.checkpoint` 会把入参 trace 成 tracer，层号若作为入参传入，`f"blocks.{i}"`
会拼出垃圾键 → LoRA 静默失效、梯度恒 0，而且反向可能被 XLA 整个 DCE 掉，
**使步时假性变快**。不断言根本发现不了。

**⑧ 打包枚举**：当前是 Krea2 口径（段含 512 文本）。Anima 文本走 cross-attn、
不进主序列，换算时 `--txt 0`。
