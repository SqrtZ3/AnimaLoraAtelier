# `jax_tpu/tests` —— TPU 路线的闸门

改动 `../*.py` 或 `kaggle_job/*` 之后，**先在这里跑完再推 TPU**。
真机一轮几十分钟 + 配额，这里几分钟，能挡掉绝大多数错。

| 闸门 | 脚本 | 证明什么 | 需要 |
|---|---|---|---|
| ① 数值对拍 | `dump_torch_ref.py --dump` → `check_jax_parity.py` | JAX 前向 ≡ PyTorch | torch + jax |
| ② 块对角 / 两级 mask | `check_splash_blockdiag.py` | 打包路线的注意力 ≡ 稠密参考 | jax |
| ③ scan ≡ 展开 | `check_scan_equiv.py` | `lax.scan` 路径与逐块展开等价 | jax |
| ④ 训练目标 | `check_flow_parity.py` | Flow Matching ≡ `trainer/objective.py` | torch + jax |
| ⑤ 优化器 | `check_optim_parity.py` | AdamW ≡ `torch.optim.AdamW` | torch + jax |
| ⑥ 导出 | `check_export.py --write` → `check_export.py` | LoRA 文件 ≡ 仓库键名规范 | torch + jax |
| ⑦ 端到端（打包） | `check_train_loop.py` | 打包路线能学 + 8 卡分片走得通 | jax |
| ⑧ 打包枚举 | `enum_pack_layouts.py` | 布局数不爆炸、填充率够高 | 纯 stdlib |
| ⑨ **布局等价** | `check_ragged_equiv.py` | `forward_ragged` ≡ `forward_packed` | jax |
| ⑩ **端到端（分桶）** | `check_train_loop_ragged.py` | 分桶路线能学 + 8 卡分片走得通 | jax |
| ⑪ **全功能训练步** | `check_train_full.py` | 用户 yaml 那一整套开关同时打开时**每一条都真的接上了** | jax |
| ⑫ 适配器对拍 | `dump_adapter_ref.py` → `check_adapter_parity.py` | LoKr/DoRA/adamw_snr ≡ `trainer/lora.py`+`utils/` | torch + jax |
| ⑬ 目标函数对拍 | `dump_objective_ref.py` → `check_objective_parity.py` | Huber(snr)/逐图归约/Eisbach/spectral ≡ `trainer/objective.py`+`aux_losses.py` | torch + jax |
| ⑭ **打包不变量** | `check_pack_invariants.py` | 「自然长度」打包的 RNG 兼容双射 / 布局数方向 / 结构不变量 | 纯 numpy |
| ⑮ **无图缓存** | `check_cache_scan.py` | 缓存目录不含图片时样本集逐条不变 + sidecar 不被当样本 | 纯 numpy |
| **preflight** | `enum_quantum_advisor.py --config <yaml>` | 用这次的 yaml+数据集扫出该传的 `--quantum` | numpy + pyyaml |
| **K1** Krea2 前向对拍 | `dump_krea2_ref.py` → `check_krea2_parity.py` | `krea2_jax` ≡ `models/krea2_modeling.py`（逐 tap + LoRA 四区域） | torch + jax |
| **K2** Krea2 端到端 | `check_train_loop_k2.py` | FSDP 分片训练闭环能学 + 分片≡全量 + 导出键名 | jax |

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

## 环境

本地 torch 与 jax **装不进同一个解释器**，所以对拍拆成两步、经 `_ref/*.npz` 落盘交换。

- torch（真权重）：`D:\ArtificialIntelligence\ComfyUI-aki-v1.5\python\python.exe`
  （见 memory `[[environment-reference]]`）
- jax：`D:\ArtificialIntelligence\animafix\jaxenv\Scripts\python.exe`
  （Python 3.12 + jax 0.11.0，与 Kaggle 钉的版本一致；本地/真机版本一旦漂移，
  "本地过了真机挂"会变成查不动的问题）
- 底模：`...\ComfyUI-aki-v1.5\models\diffusion_models\anima-base-v1.0.safetensors`

`_ref/` 是生成物，不必提交。

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
