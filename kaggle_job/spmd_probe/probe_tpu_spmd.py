#!/usr/bin/env python
"""Kaggle TPU v5e-8 · 8 卡分布式策略裁决探针（纯 JAX / shard_map，真实 Krea2 形状）。

已知（前几轮真机实测）：块对角跳块生效 0.252；call_jax 桥接开销仅 1.5%；
Krea2 12B bf16 权重 22.6GB，单 chip(15.7GiB) 放不下、8 卡分片每片 2.83GB 放得下；
反向 BlockSizes 取 dkv=1024+fused 比默认快 2.03x。

**这一轮要回答：8 颗核心怎么用，吞吐差多少。**

关键前提：**底模冻结，只训 LoRA**。这让几条策略的通信代价差异极大：

  纯 DP    权重 22.6GB > 单 chip 15.7GiB，**复制不了，不可能**。
  FSDP     权重按卡切(2.83GB/卡)，每层前向 all-gather 回全量。8 卡各跑一个 pack
           -> 样本吞吐 x8。但权重是**冻结**的，每步 all-gather 同样的东西 ——
           这笔通信是纯浪费，要看 ICI 够不够快到可以无视。
  TP       48 head 切 8x6、mlpdim 16384 切 8x2048，权重**永久分片、零权重通信**，
           只在每层两处 all-reduce 激活（wo 之后、w_down 之后）。对冻结底模天然
           合适。splash 自带 head_shards 参数，说明这条路是被支持的。

判据是 **tokens/s**（不是步时）—— FSDP 一步吞 8 个 pack，TP 一步吞 1 个，
不换算成 tokens/s 没法比。

用法（Kaggle script kernel，无参数）。本地零 TPU 复现：--interpret --tiny
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np

RESULTS: list[dict] = []
_T0 = time.time()
_BOOTSTRAP_NOTE = ""
_INTERPRET = False

OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else tempfile.gettempdir()
BOOTSTRAP_UPGRADE_JAX = os.environ.get("ANIMA_TPU_UPGRADE", "1") == "1"

# 真实 Krea2 形状（models/krea2_modeling.py KREA2_LARGE_WIDE）
D_MODEL = 6144
Q_HEADS, KV_HEADS, HEAD_DIM = 48, 12, 128
MLPDIM = 16384
N_LAYERS = 28
LORA_R = 32

L = 16384          # navit_token_budget
N_SEG = 4          # pack 内 4 段
FWD_BLOCK = 128    # 段长对齐粒度（前几轮硬结论）
BWD_BLOCK = 1024   # arch_probe H1：dkv=1024+fused 比默认快 2.03x


def _flush() -> None:
    try:
        with open(os.path.join(OUT_DIR, "spmd_probe.json"), "w", encoding="utf-8") as f:
            json.dump(RESULTS, f, ensure_ascii=False, indent=2)
        with open(os.path.join(OUT_DIR, "spmd_probe_report.txt"), "w", encoding="utf-8") as f:
            for r in RESULTS:
                f.write(f"[{r['status']:^6}] {r['name']}"
                        + (f" - {r['detail']}" if r["detail"] else "") + "\n")
            f.write(f"\n（截至 {time.time() - _T0:.0f}s 的快照）\n")
    except OSError:
        pass


def record(name: str, status: str, detail: str = "") -> None:
    RESULTS.append({"name": name, "status": status, "detail": detail,
                    "t_elapsed_s": round(time.time() - _T0, 1)})
    mark = {"OK": "  OK  ", "FAIL": " FAIL ", "SKIP": " SKIP ", "INFO": " INFO "}.get(status, status)
    print(f"[{mark}] {name}" + (f" - {detail}" if detail else ""), flush=True)
    _flush()


_DEFINED_PROBES: list[str] = []


class _Skip(Exception):
    pass


def probe(name: str):
    _DEFINED_PROBES.append(name)

    def deco(fn):
        def run(*a, **kw):
            try:
                detail = fn(*a, **kw)
                record(name, "OK", detail or "")
                return True
            except _Skip as s:
                record(name, "SKIP", str(s))
                return None
            except Exception as e:
                tb = traceback.format_exc(limit=3).strip().splitlines()[-1]
                record(name, "FAIL", f"{type(e).__name__}: {e} | {tb}")
                return False

        run.__probe_name__ = name
        return run

    return deco


def _bootstrap_upgrade_jax() -> None:
    global _BOOTSTRAP_NOTE
    if not BOOTSTRAP_UPGRADE_JAX:
        _BOOTSTRAP_NOTE = "未开启"
        return
    if "jax" in sys.modules:
        _BOOTSTRAP_NOTE = "[!] jax 已 import，升级无效"
        return
    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "jax[tpu]"],
                           capture_output=True, text=True, timeout=1800)
    except Exception as e:
        _BOOTSTRAP_NOTE = f"升级异常 {type(e).__name__}: {e}"
        return
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "").strip().splitlines()[-3:]
        _BOOTSTRAP_NOTE = f"升级失败 {r.returncode}: {' | '.join(tail)}"
        return
    _BOOTSTRAP_NOTE = f"jax[tpu] 升级 OK {time.time() - t0:.0f}s"


# ── 公共 ──────────────────────────────────────────────────────────────────────

def _seg_lens(total, n, align=FWD_BLOCK):
    per = (total // n) // align * align
    lens = [per] * (n - 1) + [total - per * (n - 1)]
    assert sum(lens) == total and all(x % align == 0 for x in lens), lens
    return lens


def _splash(seg_lens, n_heads):
    """块对角 splash 可调用对象。前向块 128（段长对齐），反向块 1024（H1 实测最快）。"""
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_kernel as sk, splash_attention_mask as sm,
    )

    class BD(sm.Mask):
        def __init__(self, lens):
            self.lens = tuple(int(x) for x in lens)
            self._ids = np.repeat(np.arange(len(self.lens), dtype=np.int32), self.lens)
            n = int(self._ids.shape[0])
            self._shape = (n, n)

        @property
        def shape(self):
            return self._shape

        def __getitem__(self, idx):
            a, b = self._ids[idx[0]], self._ids[idx[1]]
            return (a[:, None] == b[None, :]).astype(np.bool_)

        def __eq__(self, o):
            return isinstance(o, BD) and self.lens == o.lens

        def __hash__(self):
            return hash((type(self), self.lens))

    bs = sk.BlockSizes(
        block_q=FWD_BLOCK, block_kv=FWD_BLOCK, block_kv_compute=FWD_BLOCK,
        block_q_dkv=BWD_BLOCK, block_kv_dkv=BWD_BLOCK, block_kv_dkv_compute=BWD_BLOCK,
        use_fused_bwd_kernel=True,
    )
    multi = sm.MultiHeadMask(masks=[BD(seg_lens)] * n_heads)
    return sk.make_splash_mha(multi, head_shards=1, q_seq_shards=1,
                              block_sizes=bs, interpret=_INTERPRET)


def _bench(fn, args, reps=5, warmup=2):
    import jax
    t0 = time.time()
    jax.block_until_ready(fn(*args))
    first = time.time() - t0
    for _ in range(warmup - 1):
        jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append((time.perf_counter() - t) * 1e3)
    return float(np.median(ts)), first


def _cleanup() -> float:
    """把上一项探测留下的设备数组全部释放，返回清理后的 HBM 占用。

    **第二跑两个 OOM 的根因**：报 "要 384M，只剩 201M" —— 不是策略本身放不下，
    而是 P0(1GB 缓冲) / P3(2 层全量权重 1.6GB) 的数组还活着，堆到了 15.5GB。
    jax 数组靠引用计数释放，探测函数返回后 Python 未必立刻 GC。这里显式删。
    每个重探测前都调，让各策略在**同样干净的起点**上比较。"""
    import gc
    import jax
    try:
        for a in jax.live_arrays():
            try:
                a.delete()
            except Exception:
                pass
    except Exception:
        pass
    gc.collect()
    return _hbm_used()


def _hbm_used():
    import jax
    st = jax.devices()[0].memory_stats() or {}
    return st.get("bytes_in_use", 0) / 1024 ** 3


def _mesh(shape=(8,), names=("d",)):
    import jax
    from jax.sharding import Mesh
    devs = jax.devices()
    if len(devs) < 8:
        raise _Skip(f"只有 {len(devs)} 个设备，8 卡策略不可测")
    return Mesh(np.array(devs).reshape(shape), names)


def _smap(f, mesh, in_specs, out_specs):
    """shard_map 包装：关掉静态复制检查。jax 0.8 之前叫 check_rep，
    0.11 起叫 check_vma —— 两个名字都试一遍，别为版本差异写死。"""
    from jax.experimental.shard_map import shard_map
    for kw in ({"check_vma": False}, {"check_rep": False}, {}):
        try:
            return shard_map(f, mesh=mesh, in_specs=in_specs, out_specs=out_specs, **kw)
        except TypeError:
            continue
    raise RuntimeError("shard_map 不接受任何已知的 check_* 参数名")


# ── 0. 环境 ───────────────────────────────────────────────────────────────────

def probe_env() -> None:
    record("python", "INFO", f"{platform.python_version()} @ {sys.executable}")
    record("bootstrap", "INFO", _BOOTSTRAP_NOTE)


@probe("0.1 设备 / HBM")
def probe_basics() -> str:
    import jax
    devs = jax.devices()
    st = devs[0].memory_stats() or {}
    return (f"jax {jax.__version__} | {len(devs)} x {devs[0].device_kind} | "
            f"单设备 HBM limit={st.get('bytes_limit', 0) / 1024 ** 3:.1f}GiB")


# ── P0. ICI 带宽（一切策略选择的基础）────────────────────────────────────────

@probe("P0 ICI 集合通信带宽（8 卡）")
def probe_ici() -> str:
    """FSDP 每层要 all-gather 一层权重(0.81GB)、TP 每层要 all-reduce 两次激活
    (各 0.19GB)。这两个数除以带宽就是通信下限，先把带宽量出来。

    口径：all-gather 的有效带宽按"每卡收到的字节数"算；all-reduce 按 ring 口径
    的 2*(N-1)/N * size 算总线字节。都只报**实测耗时**与由此反推的带宽，不做
    理论修正。"""
    import jax
    import jax.numpy as jnp
    from jax.experimental.shard_map import shard_map
    from jax.sharding import NamedSharding, PartitionSpec as P
    mesh = _mesh()
    rows = []
    for label, mb in (("64MB", 64), ("256MB", 256), ("1024MB", 1024)):
        n = mb * 1024 * 1024 // 2 // 8            # 每卡 bf16 元素数
        x = jax.device_put(jnp.ones((8, n), jnp.bfloat16),
                           NamedSharding(mesh, P("d", None)))

        @jax.jit
        def ar(a):
            return _smap(lambda t: jax.lax.psum(t, "d"),
                         mesh, P("d", None), P("d", None))(a)

        @jax.jit
        def ag(a):
            # all_gather 的输出在各卡上是同一份 -> out_specs 要写 P(None,...)，
            # 但 shard_map 的静态复制推断看不出来，必须关掉那个检查。
            # jax 0.11 把 check_rep 改名成 check_vma，_smap 里两个名字都试。
            return _smap(lambda t: jax.lax.all_gather(t, "d", tiled=True),
                         mesh, P("d", None), P(None, None))(a)

        t_ar, _ = _bench(ar, (x,), reps=5)
        t_ag, _ = _bench(ag, (x,), reps=5)
        tot_gb = mb / 1024.0                      # 全局张量大小
        rows.append(f"{label}: all-reduce {t_ar:.2f}ms({tot_gb * 2 * 7 / 8 / (t_ar / 1e3):.0f}GB/s) "
                    f"/ all-gather {t_ag:.2f}ms({tot_gb * 7 / 8 / (t_ag / 1e3):.0f}GB/s)")
    return " ; ".join(rows)


@probe("P0.2 FSDP/TP 每步通信量（纸面，由 P0 换算）")
def probe_comm_budget() -> str:
    """把每步要搬多少字节算清楚，配合 P0 的带宽就能预判谁快。"""
    w_layer = (D_MODEL * Q_HEADS * HEAD_DIM + 2 * D_MODEL * KV_HEADS * HEAD_DIM
               + 2 * D_MODEL * D_MODEL + 3 * D_MODEL * MLPDIM) * 2 / 1024 ** 3
    act = L * D_MODEL * 2 / 1024 ** 3
    lora_p = N_LAYERS * 2 * ((D_MODEL + Q_HEADS * HEAD_DIM) + (D_MODEL + D_MODEL)
                             + (D_MODEL + MLPDIM) + (MLPDIM + D_MODEL)) * LORA_R
    return (f"每层权重 {w_layer:.2f}GB，每层激活 {act:.2f}GB，LoRA 参数 "
            f"{lora_p / 1e6:.0f}M({lora_p * 2 / 1024 ** 2:.0f}MB bf16) | "
            f"FSDP: 前向 all-gather {w_layer * N_LAYERS:.1f}GB + 反向再一次（重算则更多），"
            f"梯度只有 LoRA -> reduce-scatter 仅 {lora_p * 2 / 1024 ** 2:.0f}MB | "
            f"TP: 每层 2 次 all-reduce x {act:.2f}GB = {2 * act * N_LAYERS:.1f}GB/步，"
            f"权重零通信（冻结底模永久分片）")


# ── 单层计算核（两种策略共用）────────────────────────────────────────────────

def _init_w(key, *shape):
    import jax
    import jax.numpy as jnp
    return (jax.random.normal(key, shape, jnp.float32) * 0.02).astype(jnp.bfloat16)


def _layer_full(W, lora, x, splash_fn, q_heads):
    """完整单层（本地持有全量权重）：attn(GQA+块对角) + SwiGLU，LoRA 旁路。
    **口径**：显存/算力口径的复刻，省掉 RMSNorm/RoPE/调制的精确形式。"""
    import jax
    import jax.numpy as jnp
    n = x.shape[0]

    def lin(t, name):
        y = t @ W[name]
        if name in lora:
            a, b = lora[name]
            y = y + (t @ a) @ b
        return y

    q = lin(x, "wq").reshape(n, q_heads, HEAD_DIM).transpose(1, 0, 2)
    k = (x @ W["wk"]).reshape(n, -1, HEAD_DIM).transpose(1, 0, 2)
    v = (x @ W["wv"]).reshape(n, -1, HEAD_DIM).transpose(1, 0, 2)
    q = (q.astype(jnp.float32) * (HEAD_DIM ** -0.5)).astype(jnp.bfloat16)  # splash 不内置缩放
    a = splash_fn(q, k, v).transpose(1, 0, 2).reshape(n, q_heads * HEAD_DIM)
    a = a * jax.nn.sigmoid(x @ W["gate"])
    x = x + lin(a, "wo")
    h = jax.nn.silu(lin(x, "w_gate")) * (x @ W["w_up"])
    return x + lin(h, "w_down")


def _layer_shapes(q_heads, kv_heads, mlpdim):
    return {
        "wq": (D_MODEL, q_heads * HEAD_DIM), "wk": (D_MODEL, kv_heads * HEAD_DIM),
        "wv": (D_MODEL, kv_heads * HEAD_DIM), "gate": (D_MODEL, D_MODEL),
        "wo": (q_heads * HEAD_DIM, D_MODEL), "w_gate": (D_MODEL, mlpdim),
        "w_up": (D_MODEL, mlpdim), "w_down": (mlpdim, D_MODEL),
    }


def _stacked_weights(key, n_layers, mesh, specs, q_heads=Q_HEADS,
                     kv_heads=KV_HEADS, mlpdim=MLPDIM):
    """**直接分片创建** [n_layers, in, out] 的权重堆。

    **第二/三跑两次 OOM 的真根因**（和之前 arch_probe F3 是同一个错）：原来先
    `_make_weights()` 在默认设备上把 28 层权重全物化（22.6GB 落在单卡）再
    device_put 分片 —— 创建阶段就爆了，跟策略本身没关系。所以必须用
    `jax.jit(..., out_shardings=...)` 让 XLA 一开始就分布式地生成。

    另外**直接生成 bfloat16**，不走 fp32 再 cast —— 那会多出 4 倍的中间量。"""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding
    shapes = _layer_shapes(q_heads, kv_heads, mlpdim)
    names = list(shapes)

    def make():
        ks = jax.random.split(key, len(names))
        return {n: jax.random.normal(ks[i], (n_layers,) + shapes[n], jnp.bfloat16) * 0.02
                for i, n in enumerate(names)}

    out_sh = {n: NamedSharding(mesh, specs[n]) for n in names}
    return jax.jit(make, out_shardings=out_sh)()


def _make_weights(key, q_heads, kv_heads, mlpdim, d_out_wo=None):
    import jax
    ks = jax.random.split(key, 8)
    return {
        "wq": _init_w(ks[0], D_MODEL, q_heads * HEAD_DIM),
        "wk": _init_w(ks[1], D_MODEL, kv_heads * HEAD_DIM),
        "wv": _init_w(ks[2], D_MODEL, kv_heads * HEAD_DIM),
        "gate": _init_w(ks[3], D_MODEL, d_out_wo or D_MODEL),
        "wo": _init_w(ks[4], q_heads * HEAD_DIM, D_MODEL),
        "w_gate": _init_w(ks[5], D_MODEL, mlpdim),
        "w_up": _init_w(ks[6], D_MODEL, mlpdim),
        "w_down": _init_w(ks[7], mlpdim, D_MODEL),
    }


def _make_lora(key, pairs):
    import jax
    import jax.numpy as jnp
    ks = jax.random.split(key, len(pairs))
    out = {}
    for i, (name, (i_dim, o_dim)) in enumerate(pairs.items()):
        out[name] = (_init_w(ks[i], i_dim, LORA_R),
                     jnp.zeros((LORA_R, o_dim), jnp.bfloat16))
    return out


# ── P1. FSDP：权重按卡切，每卡一个 pack ──────────────────────────────────────

@probe("P1 FSDP 全 28 层训练步（8 卡各一个 pack）")
def probe_fsdp() -> str:
    """权重沿输入维按卡切，每层 all_gather 拼回全量；数据是 8 个不同的 pack。
    冻结底模 -> 只有 LoRA 有梯度，reduce-scatter 极小。

    **第一跑 OOM(35.46G) 的根因**：把 28 层权重当 Python list 逐个展开传进
    shard_map，层与层之间没有强制的顺序依赖，XLA 就把所有层的 all_gather
    一起发了 —— 28 x 0.81GB 同时活着必然爆。
    改用 `lax.scan` 扫层：权重堆成 [28, in, out]，scan 逼 XLA 逐层 gather、
    用完即释放，活着的 gather 结果始终只有一层份。这是 XLA 上做 FSDP 的标准写法。"""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    r = _run_fsdp(L)
    return (f"28 层 x 8 pack | 步时 {r['ms']:.0f}ms（首步含编译 {r['first']:.0f}s）-> "
            f"**{r['tok_s'] / 1e3:.1f}k tokens/s**（每步 {r['tok'] / 1000:.0f}k token）| "
            f"单卡 HBM {r['hbm0']:.2f}->{r['hbm']:.2f}GiB")


def _run_fsdp(l_tok):
    """跑一次 8 卡 FSDP 的 28 层训练步，返回实测字典。P1 与 P5 共用。"""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    before = _cleanup()
    mesh = _mesh()
    lens = _seg_lens(l_tok, N_SEG)
    splash_fn = _splash(lens, Q_HEADS)
    ks = jax.random.split(jax.random.PRNGKey(0), N_LAYERS + 2)

    # 堆成 [28, in, out]，输入维（axis=1）按卡切；**直接分片创建**，不在单卡物化
    keys = list(_layer_shapes(Q_HEADS, KV_HEADS, MLPDIM))
    fsdp_specs = {k: P(None, "d", None) for k in keys}
    stacked = _stacked_weights(ks[0], N_LAYERS, mesh, fsdp_specs)
    lora = _make_lora(ks[N_LAYERS], {
        "wq": (D_MODEL, Q_HEADS * HEAD_DIM), "wo": (Q_HEADS * HEAD_DIM, D_MODEL),
        "w_gate": (D_MODEL, MLPDIM), "w_down": (MLPDIM, D_MODEL)})
    x = jax.jit(lambda: jax.random.normal(ks[1], (8, l_tok, D_MODEL), jnp.bfloat16),
                out_shardings=NamedSharding(mesh, P("d", None, None)))()

    def per_shard(lora, stk, xs):
        t = xs[0]

        def one(Wsh, carry):
            # **all_gather 必须在 remat 边界【内】**（第四跑 OOM 37.62G 的根因）：
            # 放在 checkpoint 外面时，28 层 gather 出来的全量权重全被存下来给反向
            # 用 = 28 x 0.81GB = 22.6GB。放进来则前向用完即弃、反向重新 gather，
            # 活着的始终只有一层份。代价是 all-gather 做两遍。
            W = {k: jax.lax.all_gather(v, "d", axis=0, tiled=True) for k, v in Wsh.items()}
            return _layer_full(W, lora, carry, splash_fn, Q_HEADS)

        def step(carry, Wsh):
            return jax.checkpoint(one)(Wsh, carry), None

        t, _ = jax.lax.scan(step, t, stk)
        return jnp.sum(t.astype(jnp.float32) ** 2)[None]

    def loss_fn(lora, stk, x):
        f = _smap(per_shard, mesh,
                  (P(), {k: P(None, "d", None) for k in keys}, P("d", None, None)),
                  P("d"))
        return jnp.sum(f(lora, stk, x))

    grad = jax.jit(jax.grad(loss_fn, argnums=0))
    ms, first = _bench(lambda: grad(lora, stacked, x), (), reps=3, warmup=1)
    tok = 8 * l_tok
    return {"ms": ms, "first": first, "tok": tok, "tok_s": tok / (ms / 1e3),
            "hbm0": before, "hbm": _hbm_used()}


# ── P2. TP：权重永久分片，1 个 pack ──────────────────────────────────────────

@probe("P2 2D 网格 2xDP x 4xTP 全 28 层训练步（权重零通信）")
def probe_tp() -> str:
    """**第一跑发现的真架构约束**：Krea2 是 GQA 4:1，只有 **12 个 KV head**。
    TP 度数必须整除 12（1/2/3/4/6/12），**8 路 TP 在这个模型上不成立**。
    所以 8 卡的正解是二维网格：**2 路数据并行 x 4 路张量并行**
    （每卡 48/4=12 个 q head、12/4=3 个 kv head，GQA 4:1 完好；mlpdim 16384/4=4096）。

    权重按 'tp' 轴永久分片、在 'dp' 轴上复制 -> 每卡持 22.6/4=5.66GB，**零权重通信**；
    每层只在 wo 后与 w_down 后各 psum 一次激活（只在 tp 轴内）。"""
    r = _run_tp(2, 4, L)
    return (f"网格 {r['dp']}dp x {r['tp']}tp | 每卡 {r['lq']}q/{r['lkv']}kv head、"
            f"mlp {r['lmlp']} | 28 层 x {r['dp']} pack | 步时 {r['ms']:.0f}ms"
            f"（首步含编译 {r['first']:.0f}s）-> **{r['tok_s'] / 1e3:.1f}k tokens/s**"
            f"（每步 {r['tok'] / 1000:.0f}k token）| 单卡 HBM {r['hbm0']:.2f}->{r['hbm']:.2f}GiB")


def _run_tp(DP, TP, l_tok):
    """跑一次 DPxTP 网格的 28 层训练步，返回实测字典。P2/P4 共用。"""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    before = _cleanup()
    if Q_HEADS % TP or KV_HEADS % TP or MLPDIM % TP:
        raise _Skip(f"{Q_HEADS}q/{KV_HEADS}kv/{MLPDIM} 不能被 TP={TP} 整除")
    mesh = _mesh((DP, TP), ("dp", "tp"))
    lq, lkv, lmlp = Q_HEADS // TP, KV_HEADS // TP, MLPDIM // TP
    splash_fn = _splash(_seg_lens(l_tok, N_SEG), lq)   # 每卡只算自己那份 q head
    ks = jax.random.split(jax.random.PRNGKey(1), 4)

    OUT_SPLIT = ("wq", "wk", "wv", "gate", "w_gate", "w_up")   # 输出维切
    specs = {k: (P(None, None, "tp") if k in OUT_SPLIT else P(None, "tp", None))
             for k in _layer_shapes(Q_HEADS, KV_HEADS, MLPDIM)}
    stacked = _stacked_weights(ks[0], N_LAYERS, mesh, specs)   # 直接分片创建
    x = jax.jit(lambda: jax.random.normal(ks[1], (DP, l_tok, D_MODEL), jnp.bfloat16),
                out_shardings=NamedSharding(mesh, P("dp", None, None)))()

    def tp_layer(W, x):
        q = (x @ W["wq"]).reshape(l_tok, lq, HEAD_DIM).transpose(1, 0, 2)
        k = (x @ W["wk"]).reshape(l_tok, lkv, HEAD_DIM).transpose(1, 0, 2)
        v = (x @ W["wv"]).reshape(l_tok, lkv, HEAD_DIM).transpose(1, 0, 2)
        q = (q.astype(jnp.float32) * (HEAD_DIM ** -0.5)).astype(jnp.bfloat16)
        a = splash_fn(q, k, v).transpose(1, 0, 2).reshape(l_tok, lq * HEAD_DIM)
        a = a * jax.nn.sigmoid(x @ W["gate"])
        x = x + jax.lax.psum(a @ W["wo"], "tp")           # 通信点 1（仅 tp 轴内）
        h = jax.nn.silu(x @ W["w_gate"]) * (x @ W["w_up"])
        return x + jax.lax.psum(h @ W["w_down"], "tp")    # 通信点 2

    def per_shard(stk, xs):
        t = xs[0]
        t, _ = jax.lax.scan(lambda c, W: (jax.checkpoint(tp_layer)(W, c), None), t, stk)
        return jnp.sum(t.astype(jnp.float32) ** 2)[None]

    def loss_fn(stk, x):
        return jnp.sum(_smap(per_shard, mesh, (specs, P("dp", None, None)), P("dp"))(stk, x))

    grad = jax.jit(jax.grad(loss_fn, argnums=1))
    ms, first = _bench(lambda: grad(stacked, x), (), reps=3, warmup=1)
    tok = DP * l_tok
    return {"dp": DP, "tp": TP, "lq": lq, "lkv": lkv, "lmlp": lmlp, "ms": ms,
            "first": first, "tok": tok, "tok_s": tok / (ms / 1e3),
            "hbm0": before, "hbm": _hbm_used()}


@probe("P4 TP 度数与 token_budget 扫描（用富余显存换更大 pack）")
def probe_tp_sweep() -> str:
    """P2 实测 2dp x 4tp 只用 5.87/15.7GiB —— 余量很大。两个方向可以吃掉它：
    ① 提高 navit_token_budget（一个 pack 装更多/更大的图）；
    ② 改 TP 度数（TP 度数必须整除 KV_HEADS=12，8 卡下可选 (dp,tp)=(4,2)/(2,4)）。
    这里扫一遍，给 config 定 token_budget 一个有依据的上限。"""
    import jax
    rows = []
    for dp, tp, l_tok in ((2, 4, 32768), (2, 4, 65536), (4, 2, 16384), (4, 2, 32768)):
        try:
            r = _run_tp(dp, tp, l_tok)
            rows.append(f"{dp}dp x {tp}tp L={l_tok}: {r['ms']:.0f}ms -> "
                        f"{r['tok_s'] / 1e3:.1f}k tok/s, HBM {r['hbm']:.2f}GiB")
        except Exception as e:
            msg = str(e)
            hit = "OOM" if ("exceed" in msg or "RESOURCE_EXHAUSTED" in msg
                            or "not possible" in msg) else type(e).__name__
            rows.append(f"{dp}dp x {tp}tp L={l_tok}: {hit}")
    return " ; ".join(rows)


@probe("P5 FSDP 的 token_budget 上限（权重只占 2.83GB/卡，余量大）")
def probe_fsdp_budget() -> str:
    """P4 已证明 TP 下 L=16384 就是顶（激活翻倍即超）。FSDP 权重只占 2.83GB/卡，
    余量大得多 —— 扫一遍看 navit_token_budget 能开到多少。budget 越大，一个 pack
    能装的图越多/越大，直接关系到 config 怎么填。"""
    rows = []
    for l_tok in (32768, 49152, 65536):
        try:
            r = _run_fsdp(l_tok)
            rows.append(f"L={l_tok}: {r['ms']:.0f}ms -> {r['tok_s'] / 1e3:.1f}k tok/s, "
                        f"HBM {r['hbm']:.2f}GiB")
        except Exception as e:
            msg = str(e)
            hit = ("OOM" if ("exceed" in msg or "RESOURCE_EXHAUSTED" in msg
                             or "not possible" in msg) else type(e).__name__)
            rows.append(f"L={l_tok}: {hit}")
    return " ; ".join(rows)


# ── P3. 单卡基线（不可能跑满 28 层，用少层外推）──────────────────────────────

@probe("P3 单卡基线（少层，用于算并行效率）")
def probe_single() -> str:
    """单卡放不下 28 层 12B 权重，用 N 层实测再线性外推，给 P1/P2 当分母。
    **标注清楚这是外推，不是实测 28 层。**"""
    import jax
    import jax.numpy as jnp
    _cleanup()
    n_l = 2
    lens = _seg_lens(L, N_SEG)
    splash_fn = _splash(lens, Q_HEADS)
    ks = jax.random.split(jax.random.PRNGKey(2), n_l + 2)
    layers = [_make_weights(ks[i], Q_HEADS, KV_HEADS, MLPDIM) for i in range(n_l)]
    lora = _make_lora(ks[n_l], {
        "wq": (D_MODEL, Q_HEADS * HEAD_DIM), "wo": (Q_HEADS * HEAD_DIM, D_MODEL),
        "w_gate": (D_MODEL, MLPDIM), "w_down": (MLPDIM, D_MODEL)})
    x = jax.random.normal(ks[n_l + 1], (L, D_MODEL), jnp.float32).astype(jnp.bfloat16)

    def loss_fn(lora, layers, t):
        for W in layers:
            t = jax.checkpoint(_layer_full, static_argnums=(3, 4))(
                W, lora, t, splash_fn, Q_HEADS)
        return jnp.sum(t.astype(jnp.float32) ** 2)

    grad = jax.jit(jax.grad(loss_fn, argnums=0))
    t, _ = _bench(lambda: grad(lora, layers, x), (), reps=3, warmup=1)
    per_layer = t / n_l
    est = per_layer * N_LAYERS
    return (f"{n_l} 层实测 {t:.0f}ms -> 每层 {per_layer:.1f}ms；**线性外推** 28 层 "
            f"{est:.0f}ms -> {L / (est / 1e3) / 1e3:.1f}k tokens/s（单卡放不下 28 层，"
            f"这是外推不是实测）")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    global _INTERPRET, N_LAYERS, L
    ap = argparse.ArgumentParser()
    ap.add_argument("--interpret", action="store_true")
    ap.add_argument("--tiny", action="store_true", help="本地用：缩小层数/序列")
    args, _u = ap.parse_known_args()
    _INTERPRET = args.interpret
    if args.tiny:
        N_LAYERS, L = 2, 2048

    print("=" * 78)
    print("Kaggle TPU v5e-8 · 8 卡分布式策略裁决（FSDP vs TP，判据 tokens/s）")
    print("=" * 78)
    if not args.interpret:
        _bootstrap_upgrade_jax()
    probe_env()
    probe_basics()

    probe_ici()
    probe_comm_budget()
    probe_single()
    probe_tp()
    probe_fsdp()
    probe_tp_sweep()
    probe_fsdp_budget()

    _ran = {r["name"] for r in RESULTS}
    missed = [n for n in _DEFINED_PROBES if n not in _ran]
    if missed:
        print("[!] 已定义但未调用的探测：" + ", ".join(missed))

    ok = sum(1 for r in RESULTS if r["status"] == "OK")
    fail = sum(1 for r in RESULTS if r["status"] == "FAIL")
    skip = sum(1 for r in RESULTS if r["status"] == "SKIP")
    print("=" * 78)
    print(f"汇总：OK={ok}  FAIL={fail}  SKIP={skip}  总耗时={time.time() - _T0:.0f}s")

    def _d(p):
        for r in RESULTS:
            if r["name"].startswith(p):
                return r["detail"]
        return ""

    print("\n" + "=" * 78)
    print("8 卡策略裁决（判据 tokens/s）：")
    print(f"  P0 ICI 带宽:   {_d('P0 ')}")
    print(f"  P0.2 通信量:   {_d('P0.2 ')}")
    print(f"  P3 单卡外推:   {_d('P3 ')}")
    print(f"  P2 TP:         {_d('P2 ')}")
    print(f"  P1 FSDP:       {_d('P1 ')}")
    print(f"  P4 TP 度数/预算: {_d('P4 ')}")
    print(f"  P5 FSDP 预算上限: {_d('P5 ')}")
    print("  -> 取 tokens/s 更高者；若接近，优先 TP（冻结底模零权重通信，")
    print("     且单卡显存占用更低，能给更大 token_budget 留余量）。")
    print("=" * 78)
    _flush()
    print(f"报告已写到 {OUT_DIR}/spmd_probe_report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
