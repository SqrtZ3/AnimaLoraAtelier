r"""TPU 训练入口：`python run_train.py --config train_anima.yaml`。

吃的是**与 GPU 侧同一份 yaml**（jax_tpu/config.py 负责翻译与 fail-fast）。

## 一步的时间线

  host: 取一步的 8 个同布局 pack -> 读 latent/ctx 缓存 -> 自适应采样器出 t
  dev : 8 卡各跑自己的 pack -> 逐图 loss -> 梯度（转置时自动 all-reduce）
  host: 累加 grad_accum 个微步 -> apply_update -> 逐图 loss 回喂自适应采样器

## 8 卡怎么才算"用满"

三件事分别对应三个报告，训练一开始就打印，**不要等跑起来才看**：

  1. **填充率**（`packing.report`）= 线性层的算力利用率上界。pack 里的填充 token
     照样要过 MLP，填充率 85% 就意味着 15% 的 FLOPs 是白烧的。
  2. **成步率** = 能凑齐 8 个同布局 pack 的比例。凑不齐的 pack 顺延到下一轮，
     8 卡里有一张空转就等于损失 12.5% —— 所以 `plan_steps` 宁可顺延也不空跑。
  3. **布局数** = 全模型编译次数。首跑每种布局约 60s，跨 session 靠持久化编译
     缓存摊掉（`--jax-cache`）。布局数爆炸时先调大 `quantum`。

## 两个可选的工作点（真机 anima-layout-probe，打包路线、8 卡、真tok/s）

默认档不是最快档，这是有意的（新旋钮默认关、行为中立），但差距很大，**上生产前
先决定用哪个**：

  A. 默认：`navit_token_budget: 131072`（= 8 x 16384）+ scan + full
       16.9k 真tok/s / 有效MFU 13.4%。一步 32 张图左右，与 GPU 侧同一份 yaml
       的有效 batch 完全对齐，零额外决策。

  B. 快档：yaml 改 `navit_token_budget: 65536`（= 8 x 8192）+ `grad_accum: 2`，
     跑 `--unrolled --packed-chunk --remat every2`
       30.2k 真tok/s / 有效MFU 21.6%（全场最高）。**约 1.8x**。
       `grad_accum: 2` 把"一步看多少 token"补回 131072，所以它与 A 是同一个
       有效 batch、同一个实验；变的只是 8 卡怎么把这些 token 走完。

为什么不能只开 `--unrolled` 而不动 budget：budget 16384 下展开+every2 直接 OOM
（23.62G），展开+full 是 16.7k —— 与 scan 的 16.9k 打平，白开。收益全在
"budget 减半才装得下 every2"这一步上。

代价（都已实测、可接受）：展开路径首调编译 56~83s vs scan 的 9~20s（靠
`--jax-cache` 跨 session 摊掉）；`--packed-chunk` 数学等价但改 matmul lowering，
bf16 梯度有 ULP 漂移，想要零数值风险就换 `--packed-barrier`（恒等算子，
本地 fp32 对拍逐 bit，但 every2 档略慢：28.9k）。

## 断点接棒（Kaggle 12h / 20h 每周）

`--save-state-every` 存完整优化器状态 + 自适应采样器的 EMA；`--resume-state`
接回来。两者都存了才是真的接棒 —— 只存权重的话，AdamW 的一二阶矩与自适应的
burn-in 都要重来，续训的前几百步等于换了个优化器（而且看不出来）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax                                                    # noqa: E402
import jax.numpy as jnp                                       # noqa: E402

import adapters as AD                                         # noqa: E402
import anima_jax as A                                         # noqa: E402
import config as C                                            # noqa: E402
import data as D                                              # noqa: E402
import flow as F                                              # noqa: E402
import krea2_jax as K2                                        # noqa: E402
import optim as O                                             # noqa: E402
import sched as S                                             # noqa: E402
import train as T                                             # noqa: E402
from packing import K2Packer, Packer, report, report_k2       # noqa: E402


def build_mesh(devices: int):
    from jax.sharding import Mesh
    dev = jax.devices()
    if len(dev) < devices:
        raise RuntimeError(f"只有 {len(dev)} 个设备，配置要 {devices} 个。"
                           f"CPU 上可用 XLA_FLAGS=--xla_force_host_platform_device_count=8 伪造")
    return Mesh(np.asarray(dev[:devices]).reshape(devices), ("d",))


class _Prefetcher:
    """下一步 latent/ctx 的单线程后台预取。

    **训练数学逐 bit 不变**：`ds.rng`（flip/caption dropout 的随机源）的消费顺序
    与全同步版完全一致 —— 预取严格按 steps 顺序一个接一个（take 先 join、再发
    下一个，任何时刻最多一个在飞），且**不跨 epoch 边界**（下一 epoch 的
    plan_packed 在主线程跑完前，不会有新预取）。唯一的差别是读盘与 TPU 计算
    重叠：TPU 不再空等 host IO。
    """

    def __init__(self, enabled: bool = True):
        import concurrent.futures
        self._ex = concurrent.futures.ThreadPoolExecutor(1) if enabled else None
        self._fut = None

    def start(self, fn, *args):
        if self._ex is not None:
            self._fut = self._ex.submit(fn, *args)

    def take(self, fn, *args):
        """取回上一个 start 的结果；没有在飞任务时同步执行。异常在此 re-raise。"""
        if self._ex is None:
            return fn(*args)
        fut, self._fut = self._fut, None
        return fn(*args) if fut is None else fut.result()


def _ms_weights(packs, w: float) -> Optional[np.ndarray]:
    """multiscale 缩放副本的逐图 loss 权重（原生份恒 1.0）。w==1 时返回 None。"""
    if abs(w - 1.0) < 1e-9:
        return None
    out = []
    for p in packs:
        for i in range(p.layout.n_seg):
            it = p.items[i] if i < len(p.items) else None
            out.append(w if (it is not None and getattr(it, "ms_target", 0)) else 1.0)
    return np.asarray(out, np.float32)


def _valid_mask(packs) -> np.ndarray:
    """哪些段是真实图（FFD 装箱余量段不是）。自适应反馈只能用真实图。"""
    out = []
    for p in packs:
        for i in range(p.layout.n_seg):
            out.append(1.0 if (i < len(p.real_lens) and p.real_lens[i] > 0) else 0.0)
    return np.asarray(out, np.float32)


def _sample_t(sampler, rng, packs, layout, fcfg, gstep) -> np.ndarray:
    """采本步的 t，**RNG 消费与「补齐到 budget」旧版逐 bit 一致**。

    旧版每 pack 按含纯填充段的段数采 t（填充段的 t 采了被丢）。自然长度布局的
    段数可能更少（Σ实段恰好 PACK_Q 对齐时填充段整个不存在），这里仍按旧槽数
    `layout.t_slots` 采样，再按 `layout.t_gather`（新段序第 p 段 -> 旧槽位）
    重排 —— 实段拿到与旧版完全相同的 t，新旧两条代码路径的 loss 曲线可逐点
    对比。直接构造的 Layout（t_slots=0）退化为按 n_seg 采样、不重排。
    """
    n_slots = layout.t_slots or layout.n_seg
    t = sampler.sample(rng, len(packs) * n_slots, fcfg, gstep)
    if layout.t_gather:
        t = np.asarray(t).reshape(len(packs), n_slots)[:,
              np.asarray(layout.t_gather, np.int64)].reshape(-1)
    return np.asarray(t, np.float32)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="与 GPU 侧同一份训练 yaml")
    ap.add_argument("--devices", type=int, default=8)
    ap.add_argument("--quantum", type=int, default=1024,
                    help="段长量化粒度（必须是 128 的倍数）。本地枚举实测 1024 是"
                         "拐点：布局数从 25 塌到 4，填充率一分不掉")
    ap.add_argument("--allow-unported", action="store_true",
                    help="放行 yaml 里那些没移植的开关（会打印清单）")
    ap.add_argument("--jax-cache", default="", help="XLA 持久化编译缓存目录")
    ap.add_argument("--max-steps", type=int, default=0, help="覆盖 yaml 的 max_steps")
    ap.add_argument("--plan-only", action="store_true",
                    help="只打印配置/打包/结构报告就退出，零设备开销")
    ap.add_argument("--interpret", action="store_true",
                    help="splash 走 Pallas 解释执行（CPU 冒烟用，真机上**不要**加）")
    ap.add_argument("--no-prefetch", action="store_true",
                    help="关掉下一步 latent/ctx 的后台预取（默认开；纯 host IO 重叠，"
                         "训练数学逐 bit 不变。排查数据侧问题时用它回到全同步）")
    # ── 吞吐旋钮（默认全关 = 与改动前逐位等价）──────────────────────────────
    ap.add_argument("--unrolled", action="store_true",
                    help="走展开路径而不是 lax.scan。必须同时开 --packed-chunk 或 "
                         "--packed-barrier。真机账见 train.TrainConfig.unrolled："
                         "收益只在 budget 8192 + every2 上（30.2k vs 16.9k 真tok/s），"
                         "而 budget 减半要用 grad_accum 补回有效 batch")
    ap.add_argument("--packed-chunk", action="store_true",
                    help="按段长 gcd 把调制切成广播（数学等价，bf16 有 ULP 漂移）")
    ap.add_argument("--packed-barrier", action="store_true",
                    help="块边界插 optimization_barrier（恒等算子，零数值风险）")
    ap.add_argument("--remat", default="", choices=("", "full", "dots", "every2", "none"),
                    help="覆盖 yaml 的 grad_checkpoint 推出来的档位。展开路径下 "
                         "every2 才是快的那一档，但只在 budget<=8192 装得下")
    a = ap.parse_args(argv)

    raw = C.load_yaml(a.config)
    family = str(raw.get("model_family", "anima") or "anima").lower()
    if family == "krea2":
        return main_k2(a, raw)
    if family != "anima":
        raise ValueError(f"model_family={family!r} 不认识（anima / krea2）")

    # ── 数据先扫：aux_spectral 的画布尺寸要从数据集的最大网格来 ────────────────
    # 第一次 build 只为拿到数据路径等 host 侧字段；画布尺寸此刻还不知道
    # （要先扫完数据集才知道最大网格），先给个占位值。
    tmp = C.build(raw, a.devices, a.allow_unported, canvas_hw=(1, 1))
    ds = D.CacheDataset(tmp.data_dir, txt_len=tmp.txt_len,
                        flip_prob=tmp.flip_prob, repeats=tmp.repeats,
                        multiscale=tmp.multiscale,
                        caption_dropout=tmp.caption_dropout,
                        rng=np.random.RandomState(tmp.tcfg.seed))
    rc = C.build(raw, a.devices, a.allow_unported, canvas_hw=ds.canvas_hw)
    rc = replace(rc, quantum=a.quantum,
                 max_steps=(a.max_steps or rc.max_steps))
    if rc.eval_every and not rc.eval_t_grid:
        # 以前这里静默不跑 eval（evalset 永远取不到）——开关开着却什么都没发生，
        # 正是本仓库要防的那类。PyTorch 侧 eval 走自己的默认网格。
        raise ValueError(
            f"eval_every={rc.eval_every} 但 eval_t_grid 为空：TPU 侧 eval 只在"
            f"固定 t 网格上跑，没有网格就没有 eval。请设 eval_t_grid（如 "
            f'"0.1,0.3,0.5,0.7,0.9"）或把 eval_every 关掉。')
    # 吞吐旋钮走命令行而不是 yaml：它们**不改训练数学**（chunk 是数学等价、
    # barrier 是恒等算子、remat 只改重算策略），所以不该混进"两个后端共用的
    # 同一份实验描述"里。TrainConfig.__post_init__ 会拦住非法组合。
    rc = replace(rc, tcfg=replace(
        rc.tcfg, unrolled=a.unrolled, packed_chunk=a.packed_chunk,
        packed_barrier=a.packed_barrier, remat=(a.remat or rc.tcfg.remat)))

    print(C.summary(rc))
    print(ds.report())

    packer = Packer(rc.budget, rc.quantum, rc.txt_len, rc.devices)
    steps, carry = ds.plan_packed(packer)
    print(report([p for st in steps for p in st] + carry, rc.devices, rc.budget))
    _advise_steps(steps, carry, rc)
    if rc.tcfg.aux.spectral_enabled:
        h, w = rc.tcfg.aux.canvas_hw
        print(f"aux_spectral 画布 {2*h}x{2*w} latent 像素/图 "
              f"(fp32 约 {2*h*2*w*16*4/1e6:.1f}MB/图，pred+target 两份)")

    # ── 结构报告（不碰设备）────────────────────────────────────────────────────
    if a.plan_only:
        cfg_probe = A.AnimaConfig()
        plans = AD.plan_targets(cfg_probe, rc.tcfg.adapter, rc.tcfg.targets)
        print(AD.summary(plans, rc.tcfg.adapter))
        return 0

    if a.jax_cache:
        from jax.experimental.compilation_cache import compilation_cache as cc
        cc.set_cache_dir(a.jax_cache)
        print(f"XLA 持久化编译缓存 -> {a.jax_cache}")

    mesh = build_mesh(rc.devices)
    print(f"设备 {rc.devices} x {jax.devices()[0].device_kind}")

    # ── 模型与适配器 ──────────────────────────────────────────────────────────
    t0 = time.time()
    params, mcfg = A.load_safetensors_anima(str(rc.transformer_path),
                                            dtype=rc.tcfg.dtype)
    # 展开路径要的就是 list-of-blocks 布局（`anima_jax._forward_core` 按它分派）。
    # **不能两份都留着**：3.91GB x2 在 15.7GiB 的单 chip 上就没余量了。
    if not rc.tcfg.unrolled:
        params = A.stack_blocks(params)
    print(f"底模载入 {time.time()-t0:.1f}s，{mcfg.num_blocks} 块"
          f"（{'展开' if rc.tcfg.unrolled else 'scan'} 路径）")
    if mcfg.crossattn_dim != ds.crossattn_dim:
        # cross 是定长槽，维度对不上会在组装 batch 时才炸（而且报的是 numpy 广播
        # 错，看不出根因）。在这里点破：多半是文本特征缓存用的编码器与底模不配套。
        raise ValueError(
            f"底模的 crossattn_dim={mcfg.crossattn_dim}，但文本特征缓存是 "
            f"{ds.crossattn_dim} 维。两者必须一致 —— 检查 textfeat 缓存是不是用"
            f"另一个底模/编码器跑的（tools/cache_text_features.py 会把口径写进 meta）。")

    key = jax.random.PRNGKey(rc.tcfg.seed)
    k_init, key = jax.random.split(key)
    lora, consts, plans = T.init_adapter(k_init, mcfg, rc.tcfg, params)
    print(AD.summary(plans, rc.tcfg.adapter))

    state = O.init_state(lora)
    sampler = S.AdaptiveTimestepSampler(rc.adaptive)
    start_step = 0
    if rc.resume_state:
        state = T.load_state(rc.resume_state)
        meta = Path(rc.resume_state).with_suffix(".json")
        if meta.exists():
            m = json.loads(meta.read_text(encoding="utf-8"))
            sampler.load_state(m.get("adaptive") or {})
            start_step = int(m.get("step", 0))
        print(f"接棒自 {rc.resume_state}，step={start_step}")

    rng = np.random.RandomState(rc.tcfg.seed + 1)
    ecfg = T.eval_config(rc.tcfg)
    grad_fns: Dict[Any, Any] = {}       # layout -> grad_fn
    eval_fns: Dict[Any, Any] = {}       # layout -> eval_fn（**惰性**：eval 关掉时
                                        # 一个都不构造 —— 每个 fn 含 2 个 splash 内核，
                                        # 真机各 ~735ms，纯 host 开销）

    def get_grad_fn(layout):
        if layout not in grad_fns:
            t = time.time()
            grad_fns[layout] = T.make_grad_fn(mcfg, rc.tcfg, plans, layout, mesh,
                                              a.interpret)
            print(f"  [布局 {layout.seg_lens}] 新建 splash 内核 {time.time()-t:.1f}s"
                  f"（首次调用还会触发一次全模型编译）")
        return grad_fns[layout]

    def get_eval_fn(layout):
        if layout not in eval_fns:
            eval_fns[layout] = T.make_grad_fn(mcfg, ecfg, plans, layout, mesh,
                                              a.interpret, grad=False)
        return eval_fns[layout]

    out_dir = Path(rc.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gstep, acc, acc_n = start_step, None, 0
    fb: List[Tuple[np.ndarray, np.ndarray]] = []      # 本优化步累积的自适应反馈
    evalset: Optional[Tuple] = None                   # 固定 eval 集，见 _pick_evalset
    t_epoch = time.time()
    pref = _Prefetcher(not a.no_prefetch)

    for epoch in range(rc.epochs):
        if epoch:
            steps, carry = ds.plan_packed(packer)
        for i_step, packs in enumerate(steps):
            layout = packs[0].layout
            grad_fn = get_grad_fn(layout)
            # 取（或同步读）本步数据后立刻预取下一步 —— rng 消费顺序与全同步版
            # 逐 bit 一致（预取不跨 epoch 边界，见 _Prefetcher）。
            lats, ctxs = pref.take(ds.materialize_packed, packs)
            if i_step + 1 < len(steps):
                pref.start(ds.materialize_packed, steps[i_step + 1])
            # 路由概率的线性退火只作用在 host 侧的 t 采样上，**不进编译产物**
            # （device 侧的 FlowConfig 只被 huber/加权读，不读 mix 概率）。
            # 退火关着时 at_step 返回原对象，逐位等价。
            fcfg = F.at_step(rc.tcfg.flow, gstep)
            t_vec = _sample_t(sampler, rng, packs, layout, fcfg, gstep)
            batch = T.assemble_batch(packs, lats, ctxs, t_vec, mcfg, rc.tcfg.dtype,
                                     _ms_weights(packs, rc.ms_loss_weight))
            k_step, key = jax.random.split(key)
            loss, per, grads = grad_fn(state["master"], consts, params, batch, k_step)
            if evalset is None and rc.eval_every:
                evalset = _pick_evalset(packs, lats, ctxs, layout,
                                        get_eval_fn(layout), rc)

            acc = T.accumulate(acc, grads)
            acc_n += 1
            # 逐图 loss 回喂自适应采样器（**只喂真实图**：填充段的 loss 恒 0，
            # 喂进去会把它所在的 t 桶的 EMA 直接拉到 0，采样权重整片失真）。
            # **每个微步都要收**：grad_accum>1 时只收最后一个微步，等于把
            # (accum-1)/accum 的反馈扔了，自适应会按一份偏小的样本更新 EMA。
            v = _valid_mask(packs) > 0
            fb.append((t_vec[v], np.asarray(per, np.float32).reshape(-1)[v]))
            if acc_n < rc.tcfg.grad_accum:
                continue
            if rc.tcfg.grad_accum > 1:
                acc = jax.tree.map(lambda x: x / rc.tcfg.grad_accum, acc)
            state, diag = T.apply_update(state, acc, rc.tcfg.adamw)
            acc, acc_n = None, 0
            gstep += 1
            sampler.update(np.concatenate([f[0] for f in fb]),
                           np.concatenate([f[1] for f in fb]))
            fb = []

            if gstep % rc.log_every == 0:
                print(f"e{epoch} s{gstep} loss {float(loss):.5f} "
                      f"gnorm {float(diag['gnorm']):.4f} lr {float(diag['lr']):.2e} "
                      f"| 图 {int(v.sum())} 填充率 "
                      f"{sum(sum(p.real_lens) for p in packs) / (len(packs)*layout.total_len):.1%}"
                      f" 容量 {layout.total_len / rc.budget:.1%}"
                      f" | {time.time()-t_epoch:.1f}s")
            if rc.eval_every and gstep % rc.eval_every == 0 and evalset:
                _run_eval(state, consts, params, mcfg, rc, evalset)
                print("  " + sampler.summary())
            if rc.save_every_steps and gstep % rc.save_every_steps == 0:
                _save(out_dir, rc, state, plans, mcfg, sampler, gstep)
            if rc.save_state_every and gstep % rc.save_state_every == 0:
                T.save_state(out_dir / f"state_step{gstep}.npz", state, rc.tcfg,
                             {"adaptive": sampler.state()})
            if rc.max_steps and gstep >= rc.max_steps:
                _save(out_dir, rc, state, plans, mcfg, sampler, gstep)
                print("到达 max_steps，结束")
                return 0
        # yaml 的 save_every 是"每几个 epoch 存一次"。以前这里无条件每 epoch 存，
        # save_every: 5 会被静默当成 1（多存不致命，但同一份 yaml 在两个后端上
        # 产物节奏不同，事后对不上）。0 = 不按 epoch 存。
        if rc.save_every and (epoch + 1) % rc.save_every == 0:
            _save(out_dir, rc, state, plans, mcfg, sampler, gstep)
    return 0


def _advise_steps(steps, carry, rc) -> None:
    """成步率的判读。**这是 8 卡路线最容易被低估的一处约束。**

    一步要 `devices` 个**同布局**的 pack。数据集小、布局又多的时候，一个 epoch
    可能一步都凑不出来 —— 不是 bug（余量会顺延到下一轮，`Packer._carry` 跨 epoch
    累积，样本一个不丢），但它意味着"一步"跨了好几个 epoch 的数据，日志上看起来
    像卡住了。这里把账算给用户看，免得等半天才发现。

    三个旋钮，代价各不相同：
      * `--quantum` 调大 -> 布局数塌下来（本地枚举：128->1024 时 25 种变 4 种、
        填充率一分不掉），是**最便宜**的一档；
      * yaml 的 `navit_token_budget` 调小 -> pack 变多、更容易凑齐，但一步看到的
        token 也变少（等价于减小 batch，会改变梯度噪声，属于**改实验**）；
      * `repeats` 调大 -> 同样让 pack 变多，且不改一步的 token 数。
    """
    n_pack = sum(len(s) for s in steps) + len(carry)
    if steps:
        print(f"首轮可成 {len(steps)} 步（{n_pack} 个 pack，顺延 {len(carry)} 个）")
        return
    print(f"[!] 首轮一步都没凑出来：{n_pack} 个 pack 分散在多种布局上，"
          f"每步要 {rc.devices} 个同布局的。\n"
          f"    这**不是**错误 —— 余量会跨 epoch 累积（一个样本都不丢），"
          f"但前几个 epoch 会看不到 step。\n"
          f"    想更快成步：先调大 --quantum（最便宜，填充率几乎不掉），"
          f"再考虑 repeats；调小 navit_token_budget 会改变有效 batch，算改实验。")


def _pick_evalset(packs, lats, ctxs, layout, eval_fn, rc,
                  assemble=T.assemble_batch):
    """把**第一步**的那批图冻成 eval 集。

    以前 eval 用的是"当前训练步"的 packs —— 每次 eval 换一批图，于是 eval 曲线
    上相邻两点的差里混着"换了图"和"权重变了"两件事，而它存在的全部理由就是
    把后者单独看出来。冻住之后每个点只有权重在动。

    图**故意取自训练集**（这条路线没有独立的 held-out 集，latent 缓存只有一份）：
    它量的是"对这批图的拟合到了什么程度"，不是泛化。

    yaml 的 `eval_count` 在这里只当上限提示打印出来：一步的图数由布局决定
    （8 卡 x 每 pack 段数），不能自由取 16 张 —— 少一张就换一个布局、换一次编译。

    **存的是 host 侧材料，不是组装好的 batch**：4 个 t 的 batch 常驻要几百 MB
    设备内存，而单 chip 只有 15.7GiB、budget 已经贴顶。每 40 步重装一次的
    host 开销可以忽略。
    """
    n_real = sum(1 for p in packs for i in range(layout.n_seg)
                 if _seg_is_real(p, i))
    note = (f"（yaml eval_count={rc.eval_count}；一步的图数由布局决定，取不到任意值）"
            if rc.eval_count else "")
    print(f"  eval 固定集：{n_real} 张图 x {len(rc.eval_t_grid)} 个 t{note}")
    return (eval_fn, assemble, packs, lats, ctxs, layout)


def _seg_is_real(p, i: int) -> bool:
    """第 i 段是不是真实图（anima 看 real_lens，krea2 看 real_img_lens）。"""
    rl = getattr(p, "real_lens", None)
    if rl is not None:
        return i < len(rl) and rl[i] > 0
    rl = getattr(p, "real_img_lens", None)
    return rl is not None and i < len(rl) and rl[i] > 0


def _run_eval(state, consts, params, mcfg, rc, evalset):
    """在**固定图 + 固定 t 网格**上跑一次 eval，只有权重在变。

    训练 loss 的波动大部分来自 t 的随机性与换图，盯着它看不出"学得怎么样"。
    """
    eval_fn, assemble, packs, lats, ctxs, layout = evalset
    n_img = len(packs) * layout.n_seg
    key = jax.random.PRNGKey(rc.eval_seed)
    outs = []
    for tv in rc.eval_t_grid:
        b = assemble(packs, lats, ctxs,
                     np.full(n_img, tv, np.float32), mcfg, rc.tcfg.dtype)
        loss, _, _ = eval_fn(state["master"], consts, params, b, key)
        outs.append(f"t={tv:g}:{float(loss):.5f}")
    print("  eval " + "  ".join(outs))


def _save(out_dir: Path, rc, state, plans, mcfg, sampler, gstep: int) -> None:
    p = out_dir / f"{rc.output_name}_step{gstep}.safetensors"
    T.save_lora(p, state, rc.tcfg, plans, mcfg.num_blocks,
                {"adaptive": sampler.summary()})
    print(f"  已存 {p.name}")


# ═════════════════════════════════════════════════════════════════════════════
# Krea2 路线（model_family: krea2）——单流 MMDiT + FSDP 权重分片。
# 主循环结构与 Anima 相同（同源代码只写一份的部分：自适应反馈/grad_accum/
# prefetch/断点接棒）；模型相关的每一步都换成了 K2 版本。
# ═════════════════════════════════════════════════════════════════════════════
def _valid_mask_k2(packs) -> np.ndarray:
    """哪些段是真实图（FFD 装箱余量段不是）。自适应反馈只能用真实图。"""
    out = []
    for p in packs:
        for i in range(p.layout.n_seg):
            out.append(1.0 if (i < len(p.real_img_lens) and p.real_img_lens[i] > 0)
                       else 0.0)
    return np.asarray(out, np.float32)


def _img_tokens_of(packs) -> np.ndarray:
    """每段的**真实 image token 数**（krea2_res_shift 的 mu 输入；填充段为 0，
    它的 t 反正被 mask 丢掉）。"""
    out = []
    for p in packs:
        for i in range(p.layout.n_seg):
            out.append(float(p.real_img_lens[i]) if i < len(p.real_img_lens) else 0.0)
    return np.asarray(out, np.float32)


def main_k2(a, raw) -> int:
    from jax.sharding import NamedSharding, PartitionSpec as P

    def shard_rule(x):
        return (P(*([None] * (x.ndim - 1) + ["d"]))
                if K2.fsdp_shard_pred(x.shape, a.devices) else P())

    # ── 数据先扫（画布尺寸 + fail-fast 清单，与 anima 同一顺序）────────────────
    tmp = C.build(raw, a.devices, a.allow_unported, canvas_hw=(1, 1))
    ds = D.CacheDataset(tmp.data_dir, flip_prob=tmp.flip_prob, repeats=tmp.repeats,
                        multiscale=tmp.multiscale,
                        caption_dropout=tmp.caption_dropout,
                        rng=np.random.RandomState(tmp.tcfg.seed), family="krea2")
    rc = C.build(raw, a.devices, a.allow_unported, canvas_hw=ds.canvas_hw)
    rc = replace(rc, quantum=a.quantum,
                 max_steps=(a.max_steps or rc.max_steps))
    if rc.eval_every and not rc.eval_t_grid:
        raise ValueError(
            f"eval_every={rc.eval_every} 但 eval_t_grid 为空：TPU 侧 eval 只在"
            f"固定 t 网格上跑。请设 eval_t_grid 或把 eval_every 关掉。")
    # 吞吐旋钮（--unrolled/--packed-chunk/--packed-barrier）是 Anima 打包路径的
    # 账；K2 的 scan/FSDP 路径还没验证过展开形态，先 fail-fast 不静默吞掉。
    if a.unrolled or a.packed_chunk or a.packed_barrier:
        raise ValueError("--unrolled/--packed-chunk/--packed-barrier 是 Anima 打包"
                         "路径的吞吐旋钮，Krea2（scan+FSDP）路径尚未验证，先不开。")
    rc = replace(rc, tcfg=replace(rc.tcfg, remat=(a.remat or rc.tcfg.remat)))
    if rc.tcfg.remat in ("none",):
        raise ValueError("krea2 必须 remat != none：FSDP 的 all_gather 要在 "
                         "checkpoint 边界内（spmd_probe 第四跑 OOM 37.62G 的根因）")

    print(C.summary(rc))
    print(ds.report())

    packer = K2Packer(rc.budget, rc.quantum, devices=rc.devices)
    steps, carry = ds.plan_packed(packer)
    print(report_k2([p for st in steps for p in st] + carry, rc.devices, rc.budget))
    _advise_steps(steps, carry, rc)
    if rc.tcfg.aux.spectral_enabled:
        h, w = rc.tcfg.aux.canvas_hw
        print(f"aux_spectral 画布 {2*h}x{2*w} latent 像素/图 "
              f"(fp32 约 {2*h*2*w*16*4/1e6:.1f}MB/图，pred+target 两份)")

    if a.plan_only:
        cfg_probe = K2.Krea2Config()
        plans = K2.plan_targets_k2(cfg_probe, rc.tcfg.adapter, rc.tcfg.targets)
        print(AD.summary(plans, rc.tcfg.adapter))
        return 0

    if a.jax_cache:
        from jax.experimental.compilation_cache import compilation_cache as cc
        cc.set_cache_dir(a.jax_cache)
        print(f"XLA 持久化编译缓存 -> {a.jax_cache}")

    mesh = build_mesh(rc.devices)
    print(f"设备 {rc.devices} x {jax.devices()[0].device_kind}（FSDP 权重分片）")

    # ── 模型与适配器（分片加载：逐张读出即刻 device_put，host 不持全量）────────
    t0 = time.time()
    row_norms: Dict[str, np.ndarray] = {}
    params, mcfg = K2.load_safetensors_krea2(
        str(rc.transformer_path), dtype=rc.tcfg.dtype,
        shard_put=lambda name, arr: NamedSharding(mesh, shard_rule(arr)),
        row_norms_out=row_norms)
    params = K2.stack_blocks(params, mesh)
    print(f"底模载入 {time.time()-t0:.1f}s，{mcfg.layers} 块 + txtfusion（scan 路径，"
          f"FSDP 每卡 ~1/{rc.devices}）")

    key = jax.random.PRNGKey(rc.tcfg.seed)
    k_init, key = jax.random.split(key)
    lora, consts, plans = T.init_adapter_k2(
        k_init, mcfg, rc.tcfg.adapter, rc.tcfg.targets,
        row_norms if rc.tcfg.adapter.variant == "dora" else None)
    print(AD.summary(plans, rc.tcfg.adapter))

    state = O.init_state(lora)
    sampler = S.AdaptiveTimestepSampler(rc.adaptive)
    start_step = 0
    if rc.resume_state:
        state = T.load_state(rc.resume_state)
        meta = Path(rc.resume_state).with_suffix(".json")
        if meta.exists():
            m = json.loads(meta.read_text(encoding="utf-8"))
            sampler.load_state(m.get("adaptive") or {})
            start_step = int(m.get("step", 0))
        print(f"接棒自 {rc.resume_state}，step={start_step}")

    rng = np.random.RandomState(rc.tcfg.seed + 1)
    ecfg = T.eval_config(rc.tcfg)
    pspec = T.fsdp_pspec(params, "d", rc.devices)
    grad_fns: Dict[Any, Any] = {}
    eval_fns: Dict[Any, Any] = {}

    def get_grad_fn(layout):
        # K2 的 executable 驻留 ~10G/份（v5e-8 真机：12288/卡布局 reserve
        # 11.94G），两个布局同时驻留必 OOM —— 只留当前布局，换布局时驱逐旧
        # 的（executable 析构释放 program 内存；同布局下次靠 --jax-cache 的
        # 磁盘编译缓存秒回，steps 按布局分组连续，驱逐次数 = 布局数/epoch）。
        if layout not in grad_fns:
            if grad_fns:
                import gc
                grad_fns.clear()
                eval_fns.clear()
                gc.collect()
            t = time.time()
            grad_fns[layout] = T.make_grad_fn_k2(mcfg, rc.tcfg, plans, layout,
                                                 mesh, pspec, a.interpret)
            print(f"  [布局 {layout.seg_lens} txt{layout.txt_segs}] 新建 splash "
                  f"内核 {time.time()-t:.1f}s（首次调用还会触发一次全模型编译）")
        return grad_fns[layout]

    def get_eval_fn(layout):
        if layout not in eval_fns:
            eval_fns[layout] = T.make_grad_fn_k2(mcfg, ecfg, plans, layout,
                                                 mesh, pspec, a.interpret, grad=False)
        return eval_fns[layout]

    out_dir = Path(rc.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gstep, acc, acc_n = start_step, None, 0
    fb: List[Tuple[np.ndarray, np.ndarray]] = []
    evalset: Optional[Tuple] = None
    t_epoch = time.time()
    pref = _Prefetcher(not a.no_prefetch)

    for epoch in range(rc.epochs):
        if epoch:
            steps, carry = ds.plan_packed(packer)
        for i_step, packs in enumerate(steps):
            layout = packs[0].layout
            grad_fn = get_grad_fn(layout)
            lats, ctxs = pref.take(ds.materialize_packed, packs)
            if i_step + 1 < len(steps):
                pref.start(ds.materialize_packed, steps[i_step + 1])
            fcfg = F.at_step(rc.tcfg.flow, gstep)
            # RNG 消费与旧版逐 bit 一致（含旧填充段槽位，见 _sample_t）；
            # res_shift 作用在**重排后**的新段序 t 上（逐段独立，实段与旧版同值）。
            t_vec = _sample_t(sampler, rng, packs, layout, fcfg, gstep)
            if rc.krea2_res_shift:
                t_vec = F.krea2_res_shift_np(
                    t_vec, _img_tokens_of(packs),
                    rc.krea2_shift_min_res, rc.krea2_shift_max_res,
                    rc.krea2_shift_y1, rc.krea2_shift_y2)
                # 与 PyTorch 同序：res_shift 之后再过 t_range（anima_train.py
                # :3962 的 apply_t_range 在 krea2_shift_timesteps 之后）——
                # 否则显式设 timestep_t_min/t_max 时移位后的 t 会越出值域。
                t_vec = np.asarray(F.t_range_clip(t_vec, fcfg, np), np.float32)
            batch = T.assemble_batch_k2(packs, lats, ctxs, t_vec, mcfg,
                                        rc.tcfg.dtype,
                                        _ms_weights(packs, rc.ms_loss_weight))
            k_step, key = jax.random.split(key)
            loss, per, grads = grad_fn(state["master"], consts, params, batch, k_step)
            if evalset is None and rc.eval_every:
                evalset = _pick_evalset(packs, lats, ctxs, layout,
                                        get_eval_fn(layout), rc,
                                        assemble=T.assemble_batch_k2)

            acc = T.accumulate(acc, grads)
            acc_n += 1
            v = _valid_mask_k2(packs) > 0
            fb.append((t_vec[v], np.asarray(per, np.float32).reshape(-1)[v]))
            if acc_n < rc.tcfg.grad_accum:
                continue
            if rc.tcfg.grad_accum > 1:
                acc = jax.tree.map(lambda x: x / rc.tcfg.grad_accum, acc)
            state, diag = T.apply_update(state, acc, rc.tcfg.adamw)
            acc, acc_n = None, 0
            gstep += 1
            sampler.update(np.concatenate([f[0] for f in fb]),
                           np.concatenate([f[1] for f in fb]))
            fb = []

            if gstep % rc.log_every == 0:
                print(f"e{epoch} s{gstep} loss {float(loss):.5f} "
                      f"gnorm {float(diag['gnorm']):.4f} lr {float(diag['lr']):.2e} "
                      f"| 图 {int(v.sum())} 填充率 "
                      f"{sum(sum(p.real_img_lens) + sum(p.real_txt_lens) for p in packs) / (len(packs)*layout.total_len):.1%}"
                      f" 容量 {layout.total_len / rc.budget:.1%}"
                      f" | {time.time()-t_epoch:.1f}s")
            if rc.eval_every and gstep % rc.eval_every == 0 and evalset:
                _run_eval(state, consts, params, mcfg, rc, evalset)
                print("  " + sampler.summary())
            if rc.save_every_steps and gstep % rc.save_every_steps == 0:
                _save_k2(out_dir, rc, state, plans, sampler, gstep)
            if rc.save_state_every and gstep % rc.save_state_every == 0:
                T.save_state(out_dir / f"state_step{gstep}.npz", state, rc.tcfg,
                             {"adaptive": sampler.state()})
            if rc.max_steps and gstep >= rc.max_steps:
                _save_k2(out_dir, rc, state, plans, sampler, gstep)
                print("到达 max_steps，结束")
                return 0
        if rc.save_every and (epoch + 1) % rc.save_every == 0:
            _save_k2(out_dir, rc, state, plans, sampler, gstep)
    return 0


def _save_k2(out_dir: Path, rc, state, plans, sampler, gstep: int) -> None:
    p = out_dir / f"{rc.output_name}_step{gstep}.safetensors"
    T.save_lora_k2(p, state, rc.tcfg, plans, {"adaptive": sampler.summary()})
    print(f"  已存 {p.name}")


if __name__ == "__main__":
    raise SystemExit(main())
