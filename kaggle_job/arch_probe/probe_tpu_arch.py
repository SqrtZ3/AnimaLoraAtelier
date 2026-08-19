#!/usr/bin/env python
"""Kaggle TPU v5e-8 · NaViT-on-TPU **架构可行性**探针（纯 JAX，不碰 torch_xla）。

前一轮已裁决：块对角在 splash 上跳块真实生效（前向 0.252 / 前向+反向 0.253，
理论 0.250）。那是**注意力内核**的结论。这个探针问的是比它更靠上、也更决定
成败的三件事：

  E 静态图：navit 的 pack 组成每步都不同（ΣN 动态、段长组成动态）。XLA 是
    静态形状编译，这天然冲突。要么**补齐到固定 token 预算**把 shape 钉死，
    要么每个布局编译一次。E 组量的就是这两条路各自的真实代价。
  F 显存：v5e 单 chip 只有 15.7GiB（真机实测）。Krea2 12B 光 bf16 权重就
    ~24GB，单 chip 放不下；而我们在 GPU 上靠 fp8/fp4 省显存那条路在 TPU 上
    不存在。F 组用**真实 Krea2 层形状**量单层前向+反向的 HBM，外推 28 层。
  G 主机侧成本：MaskInfo 预处理是同步 numpy，本地 CPU 实测 ~500ms/布局，
    与 288ms 的注意力步时同量级。Kaggle 主机 CPU 上到底多少，要实测。

设计纪律沿用前两个探针：每项落盘、main 恒返回 0、漏调自检、裁决段区分
[ENV] 环境问题与技术结论。

用法（Kaggle script kernel，无参数）：直接被 `python probe_tpu_arch.py` 拉起。
本地零 TPU 复现：`python probe_tpu_arch.py --interpret --skip-slow`
"""

from __future__ import annotations

import argparse
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

# Kaggle 默认镜像 jax 0.10.2 + libtpu 2025-06-12，撞 Pallas 日期闸门（硬 raise，
# 无环境变量旁路）。必须在 import jax **之前**升级——jax 一旦初始化后端就换不掉
# libtpu。前一个探针已实测该绕法有效（43s → libtpu 2026-07-27 + jax 0.11.0）。
BOOTSTRAP_UPGRADE_JAX = os.environ.get("ANIMA_TPU_UPGRADE", "1") == "1"

# ── 真实模型形状（来自 models/krea2_modeling.py KREA2_LARGE_WIDE）────────────
# features=6144, heads=48, kvheads=12 (GQA 4:1), head_dim=128, layers=28,
# multiplier=4 → SwiGLU mlpdim = ceil(2*6144/3*4 / 128)*128 = 16384
KREA2_FEATURES = 6144
KREA2_Q_HEADS, KREA2_KV_HEADS, KREA2_HEAD_DIM = 48, 12, 128
KREA2_MLPDIM = 16384
KREA2_LAYERS = 28

TIME_L = 16384             # navit_token_budget 常用值（config/train_krea2_*.yaml）
BLOCK = 128                # splash 默认 block_kv，段长必须对齐到它


def _flush() -> None:
    try:
        with open(os.path.join(OUT_DIR, "arch_probe.json"), "w", encoding="utf-8") as f:
            json.dump(RESULTS, f, ensure_ascii=False, indent=2)
        with open(os.path.join(OUT_DIR, "arch_probe_report.txt"), "w", encoding="utf-8") as f:
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
        _BOOTSTRAP_NOTE = "未开启（本地/离线模式）"
        return
    if "jax" in sys.modules:
        _BOOTSTRAP_NOTE = "[!] jax 已被 import，升级无效"
        return
    t0 = time.time()
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-U", "jax[tpu]"],
            capture_output=True, text=True, timeout=1800)
    except Exception as e:
        _BOOTSTRAP_NOTE = f"升级异常 {type(e).__name__}: {e}"
        return
    dt = time.time() - t0
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "").strip().splitlines()[-3:]
        _BOOTSTRAP_NOTE = f"升级失败 退出码 {r.returncode}（{dt:.0f}s）: {' | '.join(tail)}"
        return
    _BOOTSTRAP_NOTE = f"jax[tpu] 升级 OK {dt:.0f}s"


# ── 公共工具 ──────────────────────────────────────────────────────────────────

def _seg_ids(seg_lens, dtype=np.int32) -> np.ndarray:
    return np.repeat(np.arange(len(seg_lens), dtype=dtype), seg_lens)


def _make_block_diag_mask_cls():
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as sm

    class BlockDiagonalMask(sm.Mask):
        def __init__(self, seg_lens):
            self.seg_lens = tuple(int(x) for x in seg_lens)
            self._ids = _seg_ids(self.seg_lens)
            n = int(self._ids.shape[0])
            self._shape = (n, n)

        @property
        def shape(self):
            return self._shape

        def __getitem__(self, idx):
            q, kv = self._ids[idx[0]], self._ids[idx[1]]
            return (q[:, None] == kv[None, :]).astype(np.bool_)

        def __eq__(self, o):
            return isinstance(o, BlockDiagonalMask) and self.seg_lens == o.seg_lens

        def __hash__(self):
            return hash((type(self), self.seg_lens))

    return BlockDiagonalMask


def _splash_fn(seg_lens, q_heads):
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_kernel as sk, splash_attention_mask as sm,
    )
    BD = _make_block_diag_mask_cls()
    multi = sm.MultiHeadMask(masks=[BD(seg_lens)] * q_heads)
    return sk.make_splash_mha(multi, head_shards=1, q_seq_shards=1, interpret=_INTERPRET)


def _rand_qkv(L, q_heads, kv_heads, dim, dtype, seed=0):
    import jax
    import jax.numpy as jnp
    ks = jax.random.split(jax.random.PRNGKey(seed), 3)
    q = jax.random.normal(ks[0], (q_heads, L, dim), jnp.float32).astype(dtype)
    k = jax.random.normal(ks[1], (kv_heads, L, dim), jnp.float32).astype(dtype)
    v = jax.random.normal(ks[2], (kv_heads, L, dim), jnp.float32).astype(dtype)
    return q, k, v


def _bench(fn, args, reps=10, warmup=3):
    """返回 (中位数毫秒, 首调秒数含编译)。"""
    import jax
    t0 = time.time()
    jax.block_until_ready(fn(*args))
    compile_s = time.time() - t0
    for _ in range(warmup - 1):
        jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append((time.perf_counter() - t) * 1e3)
    return float(np.median(ts)), compile_s


def _hbm(reset=False):
    """返回 (in_use_GiB, peak_GiB)。reset=True 先清峰值统计（若该 jax 版本支持）。"""
    import jax
    d = jax.devices()[0]
    if reset:
        for name in ("reset_memory_stats", "memory_stats_reset"):
            fn = getattr(d, name, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
    st = d.memory_stats() or {}
    g = 1024 ** 3
    return (st.get("bytes_in_use", 0) / g, st.get("peak_bytes_in_use", 0) / g)


# ── 0. 环境 ───────────────────────────────────────────────────────────────────

def probe_env() -> None:
    record("python", "INFO", f"{platform.python_version()} @ {sys.executable}")
    record("bootstrap", "INFO", _BOOTSTRAP_NOTE)


@probe("0.1 jax / 设备 / HBM")
def probe_basics() -> str:
    import jax
    devs = jax.devices()
    used, peak = _hbm()
    st = devs[0].memory_stats() or {}
    limit = st.get("bytes_limit", 0) / 1024 ** 3
    return (f"jax {jax.__version__} | {len(devs)} x {devs[0].device_kind} | "
            f"单设备 HBM limit={limit:.1f}GiB in_use={used:.2f}GiB")


# ── E. 静态图可行性（本探针的核心）────────────────────────────────────────────

@probe("E0 补齐到固定预算的浪费率（纸面，零成本）")
def probe_pad_waste() -> str:
    """navit 打包是 bin-packing，ΣN 每步不同。要在 XLA 上钉死 shape，最直接的
    办法是**把每个 pack 补齐到 token_budget**，多出来的 padding 自成一段
    （splash 的块对角 mask 天然支持，padding 段只与自己算）。

    代价 = 多算一个 pad×pad 的对角块，收益 = shape 恒定。

    **口径**：三个数都以「全通稠密 L²」为分母，才可比。
      有效代价 = Σseg²/L²（补不补齐都要算的）
      padding 额外代价 = pad²/L²
      相对开销 = pad²/Σseg²  ← 这才是"补齐到底亏多少"
    注意这是纯算术，不是实测——实测在 E2/E3。"""
    rows = []
    for fill in (0.99, 0.95, 0.90, 0.80):
        used = int(TIME_L * fill) // BLOCK * BLOCK
        pad = TIME_L - used
        segs = [used // 4 // BLOCK * BLOCK] * 3          # 有效部分切 4 段等长
        segs.append(used - sum(segs))
        useful = sum(n * n for n in segs)
        rows.append(f"填充{fill:.0%}(pad={pad}): 有效{useful / TIME_L ** 2:.4f}"
                    f" + padding{pad * pad / TIME_L ** 2:.4f}"
                    f" -> 相对开销 {pad * pad / useful:.1%}")
    return " ; ".join(rows) + "（FFD 打包实测填充率通常 >95%，故补齐的代价很小）"


@probe("E1 运行时 jax.Array mask 的显存/形状要求")
def probe_dynamic_mask_shape() -> str:
    """D2 只验了 REF_L=2048 下运行时 mask 数值正确。真实 L=16384 时，稠密 bool
    mask 是 L² = 268MB/head；48 head 就是 12.9GB —— 单 chip 15.7GiB 放不下。
    所以先问：mask 能不能只给 1 个 head 然后广播？不能的话运行时 mask 路线在
    真实形状下就是死的，必须走编译期 mask + 布局缓存。"""
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as sk
    import jax.numpy as jnp
    L = TIME_L
    per_head_mb = L * L / 1024 ** 2
    notes = [f"L={L} 稠密 bool mask 单 head {per_head_mb:.0f}MB，"
             f"{KREA2_Q_HEADS} head 共 {per_head_mb * KREA2_Q_HEADS / 1024:.1f}GB"]
    lens = [L // 4] * 4
    ids = _seg_ids(lens)
    for label, shape in (("1 head", (1, L, L)), ("广播到全 head", (KREA2_Q_HEADS, L, L))):
        try:
            m = jnp.asarray(ids[:, None] == ids[None, :])
            m = jnp.broadcast_to(m, shape)
            sk.make_splash_mha(m, head_shards=1, q_seq_shards=1, interpret=_INTERPRET)
            notes.append(f"{label}{shape}: 接受")
        except Exception as e:
            notes.append(f"{label}{shape}: 拒绝/失败 {type(e).__name__}: {str(e)[:120]}")
    return " | ".join(notes)


@probe("E2 运行时 mask 是否也跳块（决定能否一个图打天下）")
def probe_dynamic_mask_speed() -> str:
    """**这是本探针最关键的一项。**

    编译期 mask 跳块已裁决（0.252）。但编译期 mask 意味着每个 pack 布局一次
    编译 + 一次 ~500ms 的 MaskInfo 预处理。若运行时 jax.Array mask **也**跳块，
    就可以：补齐到固定预算 → shape 恒定 → 一次编译 → mask 当普通输入传进去。
    那 navit 在 TPU 上就是干净的静态图，torch_xla 路线的重编译风暴也一并消失。

    若不跳块，则必须走"段长量化 + 布局枚举 + 编译缓存"，工程复杂度高一档。"""
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as sk
    import jax.numpy as jnp
    L = TIME_L
    lens = [L // 4] * 4
    ids = _seg_ids(lens)

    def mk(m_np, h):
        m = jnp.broadcast_to(jnp.asarray(m_np), (h, L, L))
        return sk.make_splash_mha(m, head_shards=1, q_seq_shards=1, interpret=_INTERPRET)

    # 稠密运行时 mask 是 h*L*L 字节；48 head 要 12.9GB，单 chip 放不下。
    # 机理问题（跳不跳块）不需要 48 head，逐级降到能放下的头数为止。
    tried = []
    for H in (KREA2_Q_HEADS, 16, 8, 4):
        kv = KREA2_KV_HEADS if H == KREA2_Q_HEADS else H
        try:
            fn_bd = mk(ids[:, None] == ids[None, :], H)
            fn_full = mk(np.ones((L, L), np.bool_), H)
            q, k, v = _rand_qkv(L, H, kv, KREA2_HEAD_DIM, jnp.bfloat16, seed=40)
            t_bd, c_bd = _bench(fn_bd, (q, k, v), reps=5)
            t_full, _ = _bench(fn_full, (q, k, v), reps=5)
            break
        except Exception as e:
            tried.append(f"{H}head:{type(e).__name__}")
    else:
        raise _Skip(f"各头数下运行时 mask 均构造/运行失败（多半是显存）：{tried}")
    note = f"（降到 {H} head 才放得下；失败过：{tried}）" if tried else f"（{H} head）"
    ratio = t_bd / t_full
    theory = sum(n * n for n in lens) / L ** 2

    # **同头数对照**（第一跑漏了这个，导致 0.579 无法归因）：运行时 mask 只能降到
    # 4 head 才放得下，拿它和 48 head 的编译期 mask 数字比是伪对照。这里在**同一
    # 头数**下再测一遍编译期 mask，两者之差才是"运行时 vs 编译期"的真实代价。
    t_cbd, _ = _bench(_splash_fn(lens, H), (q, k, v), reps=5)
    t_cfull, _ = _bench(_splash_fn([L], H), (q, k, v), reps=5)
    c_ratio = t_cbd / t_cfull

    if ratio < c_ratio * 1.15:
        verdict = "运行时 mask 与编译期 mask 跳块效果相当"
    elif ratio < (theory + 1.0) / 2:
        verdict = "运行时 mask 只**部分**跳块（不如编译期 mask）"
    else:
        verdict = "运行时 mask 不跳块"
    return (f"同 {H} head 对照 | 运行时 mask: {t_bd:.2f}/{t_full:.2f}ms 比 {ratio:.3f} ; "
            f"编译期 mask: {t_cbd:.2f}/{t_cfull:.2f}ms 比 {c_ratio:.3f} ; 理论 {theory:.3f} "
            f"-> **{verdict}** {note}（首调 {c_bd:.1f}s）")


@probe("E3 真实 ragged pack 的实测比（不等长段 + padding 段）")
def probe_ragged_pack() -> str:
    """T1 用的是 4 等长段，是最理想的情况。真实 navit pack 是不等长的（大图小图
    混装），而且要补齐到预算。这里用一个更像真实 pack 的组成实测，看实测比是否
    仍贴理论线。"""
    import jax.numpy as jnp
    L, H = TIME_L, KREA2_Q_HEADS
    # 一个像真实 navit 的组成：两张大图 + 三张中图 + 一张小图 + padding 段
    lens = [4096, 3584, 2560, 2048, 1536, 1024]
    pad = L - sum(lens)
    assert pad >= 0 and pad % BLOCK == 0, (lens, pad)
    if pad:
        lens = lens + [pad]
    q, k, v = _rand_qkv(L, H, KREA2_KV_HEADS, KREA2_HEAD_DIM, jnp.bfloat16, seed=41)
    t_bd, _ = _bench(_splash_fn(lens, H), (q, k, v), reps=5)
    t_full, _ = _bench(_splash_fn([L], H), (q, k, v), reps=5)
    ratio = t_bd / t_full
    theory = sum(n * n for n in lens) / L ** 2
    return (f"段长={lens}（含 padding 段 {pad}） | 块对角 {t_bd:.2f}ms vs 全通 {t_full:.2f}ms "
            f"-> 实测比 {ratio:.3f}，理论 {theory:.3f}，"
            f"偏差 {abs(ratio - theory) / theory:.1%}")


# ── F. 显存与模型规模（go/no-go）───────────────────────────────────────────────

def _krea2_layer_params():
    """一层 Krea2 DiT 的参数量（按 models/krea2_modeling.py 的 Attention + SwiGLU）。"""
    d, hd = KREA2_FEATURES, KREA2_HEAD_DIM
    attn = (d * KREA2_Q_HEADS * hd            # wq
            + d * KREA2_KV_HEADS * hd         # wk
            + d * KREA2_KV_HEADS * hd         # wv
            + d * d                           # gate
            + d * d)                          # wo
    mlp = 3 * d * KREA2_MLPDIM                # gate / up / down
    return attn, mlp


@probe("F1 Krea2 12B 权重账 + 单 chip 是否放得下")
def probe_weight_budget() -> str:
    import jax
    attn, mlp = _krea2_layer_params()
    per_layer = attn + mlp
    total = per_layer * KREA2_LAYERS
    st = jax.devices()[0].memory_stats() or {}
    limit = st.get("bytes_limit", 0) / 1024 ** 3
    bf16 = total * 2 / 1024 ** 3
    return (f"每层 attn={attn / 1e6:.1f}M + mlp={mlp / 1e6:.1f}M = {per_layer / 1e6:.1f}M；"
            f"x{KREA2_LAYERS} 层 = {total / 1e9:.2f}B（不含 TE/VAE/embedding）| "
            f"bf16 权重 {bf16:.1f}GB vs 单 chip {limit:.1f}GiB -> "
            f"{'单 chip 放得下' if bf16 < limit * 0.9 else '**单 chip 放不下，必须跨 8 chip 分片**'}；"
            f"8 卡均分后 {bf16 / 8:.2f}GB/chip")


@probe("F2 单层真实 DiT block 的 HBM 峰值（前向+反向）")
def probe_layer_hbm() -> str:
    """用真实 Krea2 层形状跑一层前向+反向，量 HBM 峰值，外推 28 层。

    **口径声明**：这是 activation/显存口径的复刻（attn: wq/wk/wv/gate/wo + GQA
    + splash 块对角；mlp: SwiGLU 三矩阵），**不是数值复刻**（省掉 RMSNorm/RoPE/
    调制的精确形式，它们对显存量级影响很小）。目的是回答"放不放得下"，不是对拍。
    """
    import jax
    import jax.numpy as jnp
    L, H, KV, hd, d = TIME_L, KREA2_Q_HEADS, KREA2_KV_HEADS, KREA2_HEAD_DIM, KREA2_FEATURES
    lens = [L // 4] * 4
    splash = _splash_fn(lens, H)
    key = jax.random.PRNGKey(50)

    def init(k, *shape):
        return jax.random.normal(k, shape, jnp.float32).astype(jnp.bfloat16) * 0.02

    ks = jax.random.split(key, 7)
    W = {
        "wq": init(ks[0], d, H * hd), "wk": init(ks[1], d, KV * hd),
        "wv": init(ks[2], d, KV * hd), "gate": init(ks[3], d, d),
        "wo": init(ks[4], d, d), "w_gate": init(ks[5], d, KREA2_MLPDIM),
        "w_up": init(ks[5], d, KREA2_MLPDIM), "w_down": init(ks[6], KREA2_MLPDIM, d),
    }
    x = init(jax.random.PRNGKey(51), 1, L, d)[0]

    def layer(W, x):
        q = (x @ W["wq"]).reshape(L, H, hd).transpose(1, 0, 2)
        k = (x @ W["wk"]).reshape(L, KV, hd).transpose(1, 0, 2)
        v = (x @ W["wv"]).reshape(L, KV, hd).transpose(1, 0, 2)
        # splash 不内置 1/sqrt(d) 缩放，调用方预缩放 q
        q = (q.astype(jnp.float32) * (hd ** -0.5)).astype(jnp.bfloat16)
        a = splash(q, k, v).transpose(1, 0, 2).reshape(L, H * hd)
        a = a * jax.nn.sigmoid(x @ W["gate"])
        x = x + a @ W["wo"]
        h = jax.nn.silu(x @ W["w_gate"]) * (x @ W["w_up"])
        return x + h @ W["w_down"]

    def loss(W, x):
        return jnp.sum(layer(W, x).astype(jnp.float32) ** 2)

    # **口径修正（第一跑的坑）**：`memory_stats()["peak_bytes_in_use"]` 是**进程级
    # 历史峰值**，且这个 jax 版本没有 reset 接口 —— 第一跑因此把前面探测里分配的
    # mask 残留（14GB）当成了"单层峰值"，前向/反向读数还完全相同。
    # 改用编译产物的**静态内存分析**：只反映这一个 jit 函数自己需要多少，
    # 不受历史分配污染。
    g = 1024 ** 3
    rows = []
    for label, fn in (("前向", jax.jit(loss)),
                      ("前向+反向", jax.jit(jax.grad(loss, argnums=1)))):
        ma = fn.lower(W, x).compile().memory_analysis()
        temp = getattr(ma, "temp_size_in_bytes", 0) / g
        arg = getattr(ma, "argument_size_in_bytes", 0) / g
        out = getattr(ma, "output_size_in_bytes", 0) / g
        rows.append((label, temp, arg, out))

    w_gb = sum(int(np.prod(t.shape)) for t in W.values()) * 2 / g
    st = jax.devices()[0].memory_stats() or {}
    limit = st.get("bytes_limit", 0) / g
    bwd_temp = next(r[1] for r in rows if r[0] == "前向+反向")
    # 28 层的两个包络（**这是估算，标明口径，不是实测**）：
    #   下界 = 全层梯度检查点：每层只存输入 x（L*d*2B），temp 一层份可复用
    #   上界 = 无检查点：每层的 temp 都要留着给反向
    x_gb = L * d * 2 / g
    lo = KREA2_LAYERS * x_gb + bwd_temp
    hi = KREA2_LAYERS * bwd_temp
    detail = " | ".join(f"{l}: temp={t:.2f}GB arg={a:.2f}GB out={o:.2f}GB" for l, t, a, o in rows)
    return (f"L={L} 单层（编译产物静态分析，非历史峰值）：权重 {w_gb:.2f}GB | {detail} | "
            f"{KREA2_LAYERS} 层 activation 估算：全检查点≈{lo:.1f}GB / 无检查点≈{hi:.1f}GB"
            f"（每层输入 {x_gb:.2f}GB）| 单 chip {limit:.1f}GiB，权重另需 22.6GB 分片")


@probe("F3 8 卡 mesh 上分片放 12B 权重")
def probe_sharded_weights() -> str:
    """把一个 12B 规模的权重张量集合按 8 卡分片放上去，看是否真的放得下、
    以及每卡实际占用。这是 Krea2-on-v5e-8 的 go/no-go。"""
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    devs = jax.devices()
    if len(devs) < 8:
        raise _Skip(f"只有 {len(devs)} 个设备")
    mesh = Mesh(np.array(devs).reshape(8), ("fsdp",))
    sh = NamedSharding(mesh, P("fsdp"))
    attn, mlp = _krea2_layer_params()
    total_params = (attn + mlp) * KREA2_LAYERS
    # 用一个等参数量的二维张量代表全部权重，按第 0 轴 8 路分片
    rows = 8 * 4096
    cols = int(total_params // rows)
    before, _ = _hbm()
    # **第一跑的 bug**：`jax.device_put(jnp.zeros(...), sh)` 会先在**单设备上物化
    # 整个 22.6GB** 再分片，必然 OOM —— 那是我的写法问题，不是平台结论。
    # 正确做法：用 out_shardings 让 jit 直接把结果分片创建出来，全程不落单卡。
    w = jax.jit(lambda: jnp.zeros((rows, cols), jnp.bfloat16), out_shardings=sh)()
    jax.block_until_ready(w)
    used, _ = _hbm()
    st = devs[0].memory_stats() or {}
    limit = st.get("bytes_limit", 0) / 1024 ** 3
    gb = rows * cols * 2 / 1024 ** 3
    n_shards = len(w.addressable_shards)
    shard_gb = int(np.prod(w.addressable_shards[0].data.shape)) * 2 / 1024 ** 3
    del w
    return (f"分片创建 {rows}x{cols} bf16 = {gb:.1f}GB（≈{total_params / 1e9:.1f}B 参数）| "
            f"{n_shards} 个分片，每片 {shard_gb:.2f}GB | 单 chip in_use {before:.2f}->{used:.2f}GiB "
            f"/ limit={limit:.1f}GiB -> "
            f"{'**8 卡分片放得下**' if used < limit * 0.9 else '**即使分片也吃紧**'}")


# ── H. 反向 BlockSizes（torch_xla 第四跑撞的 24GB 墙）────────────────────────

@probe("H1 反向 BlockSizes 扫描（真实形状下反向放不放得下）")
def probe_bwd_blocksizes() -> str:
    """torch_xla 探针在真实形状（L=16384, 48q/12kv x128）跑端到端训练步时报：

        Allocation (size=24GB) would exceed memory (16GB)
        shape = bf16[128, 48, 16384, 128]
        tag = output of splash_mha_dkv_block_q_dkv_128...

    头一维 128 = q 块数（16384/128）—— 反向的 dk/dv kernel **按 q 块物化**中间结果。
    但纯 JAX 侧 T2 同形状前向+反向是跑通的（287.95ms），区别在那边用的是**默认**
    BlockSizes，而 torch_xla 探针显式传了 block_q_dkv=128。

    所以这里扫一遍：默认 / 各种 dkv 块大小 / fused 开关，看哪些组合放得下、多快。
    这决定 `navit_token_budget` 能开到多大，是很实的工程参数。"""
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_kernel as sk, splash_attention_mask as sm,
    )
    import jax
    import jax.numpy as jnp
    L, H, KV, D = TIME_L, KREA2_Q_HEADS, KREA2_KV_HEADS, KREA2_HEAD_DIM
    lens = [L // 4] * 4
    BD = _make_block_diag_mask_cls()
    multi = sm.MultiHeadMask(masks=[BD(lens)] * H)
    q, k, v = _rand_qkv(L, H, KV, D, jnp.bfloat16, seed=60)

    dflt = sk.BlockSizes.get_default()
    cands = [("默认", None)]
    for bq, bkv, fused in ((128, 128, True), (128, 128, False),
                           (512, 512, True), (1024, 1024, True), (2048, 2048, True)):
        cands.append((f"dkv={bq}/{bkv} fused={fused}",
                      dict(block_q_dkv=bq, block_kv_dkv=bkv, block_kv_dkv_compute=bkv,
                           use_fused_bwd_kernel=fused)))
    rows = []
    for label, kw in cands:
        try:
            if kw is None:
                fn = sk.make_splash_mha(multi, head_shards=1, q_seq_shards=1,
                                        interpret=_INTERPRET)
            else:
                bs = sk.BlockSizes(block_q=BLOCK, block_kv=BLOCK, block_kv_compute=BLOCK, **kw)
                fn = sk.make_splash_mha(multi, head_shards=1, q_seq_shards=1,
                                        block_sizes=bs, interpret=_INTERPRET)
            g = jax.jit(jax.grad(
                lambda a, b, c: jnp.sum(fn(a, b, c).astype(jnp.float32) ** 2),
                argnums=(0, 1, 2)))
            t, _ = _bench(g, (q, k, v), reps=3, warmup=1)
            rows.append(f"{label}: {t:.0f}ms")
        except Exception as e:
            msg = str(e)
            hit = "OOM" if ("exceed memory" in msg or "RESOURCE_EXHAUSTED" in msg) else type(e).__name__
            rows.append(f"{label}: {hit}")
    return (f"L={L} {H}q/{KV}kv x{D} 前向+反向 | 默认 BlockSizes="
            f"(q={dflt.block_q}, kv={dflt.block_kv}, q_dkv={dflt.block_q_dkv}, "
            f"kv_dkv={dflt.block_kv_dkv}, fused={dflt.use_fused_bwd_kernel}) | "
            + " ; ".join(rows))


# ── G. 主机侧成本 ─────────────────────────────────────────────────────────────

@probe("G1 MaskInfo 预处理成本（Kaggle 主机 CPU 实测）")
def probe_mask_prep() -> str:
    """本地 CPU 实测 L=16384/48heads ≈ 500ms，与 288ms 的注意力步时同量级。
    Kaggle 主机 CPU 上多少？若同量级，编译期 mask 路线**每步**都要付这笔，
    必须按布局缓存可调用对象。"""
    rows = []
    for label, lens in (("4段", [TIME_L // 4] * 4),
                        ("8段", [TIME_L // 8] * 8),
                        ("16段", [TIME_L // 16] * 16)):
        t0 = time.perf_counter()
        _ = _splash_fn(lens, KREA2_Q_HEADS)
        rows.append(f"{label}: {(time.perf_counter() - t0) * 1e3:.0f}ms")
    return " ; ".join(rows) + f"（48 heads, L={TIME_L}；与步时同量级则必须按布局缓存）"


@probe("G2 同一布局重复构造是否有缓存")
def probe_mask_prep_cache() -> str:
    """如果 make_splash_mha 内部对同一个 mask 对象有 memo，那"按布局缓存"就是
    免费的；没有的话要我们自己在 trainer 侧缓存。"""
    lens = [TIME_L // 4] * 4
    t0 = time.perf_counter()
    _splash_fn(lens, KREA2_Q_HEADS)
    first = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    _splash_fn(lens, KREA2_Q_HEADS)
    second = (time.perf_counter() - t0) * 1e3
    return (f"首次 {first:.0f}ms，同布局再构造 {second:.0f}ms -> "
            f"{'内部有缓存' if second < first * 0.3 else '**无缓存，必须自己在 trainer 侧缓存可调用对象**'}")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    global _INTERPRET
    ap = argparse.ArgumentParser()
    ap.add_argument("--interpret", action="store_true", help="Pallas 解释模式（本地无 TPU 时用）")
    ap.add_argument("--skip-slow", action="store_true")
    args, _unknown = ap.parse_known_args()
    _INTERPRET = args.interpret

    print("=" * 78)
    print("Kaggle TPU v5e-8 · NaViT-on-TPU 架构可行性探针（E 静态图 / F 显存 / G 主机成本）")
    print("=" * 78)

    if not args.interpret:
        _bootstrap_upgrade_jax()
    probe_env()
    probe_basics()

    # **顺序有讲究**：F 组量显存，必须排在 E 组**之前** —— E 组会分配几 GB 的稠密
    # mask，第一跑就是因为先跑 E 再跑 F，把 mask 残留当成了单层峰值。
    probe_weight_budget()
    probe_sharded_weights()
    if not args.skip_slow:
        probe_layer_hbm()

    probe_pad_waste()
    probe_dynamic_mask_shape()
    if not args.skip_slow:
        probe_dynamic_mask_speed()
        probe_ragged_pack()

    if not args.skip_slow:
        probe_bwd_blocksizes()

    probe_mask_prep()
    probe_mask_prep_cache()

    _ran = {r["name"] for r in RESULTS}
    missed = [n for n in _DEFINED_PROBES if n not in _ran]
    if missed:
        print("[!] 已定义但未调用的探测：" + ", ".join(missed))

    ok = sum(1 for r in RESULTS if r["status"] == "OK")
    fail = sum(1 for r in RESULTS if r["status"] == "FAIL")
    skip = sum(1 for r in RESULTS if r["status"] == "SKIP")
    print("=" * 78)
    print(f"汇总：OK={ok}  FAIL={fail}  SKIP={skip}  总耗时={time.time() - _T0:.0f}s")

    def _d(prefix):
        for r in RESULTS:
            if r["name"].startswith(prefix):
                return r["detail"]
        return ""

    def _s(prefix):
        for r in RESULTS:
            if r["name"].startswith(prefix):
                return r["status"]
        return "MISSING"

    print("\n" + "=" * 78)
    print("架构裁决：")
    e2 = _d("E2 ")
    if _s("E2 ") == "OK" and "运行时 mask 也跳块" in e2:
        print("  [静态图 YES] 运行时 jax.Array mask 也跳块")
        print("    -> 补齐到固定 token 预算 + mask 当普通输入 = 一次编译打天下。")
        print("       navit 在 TPU 上是干净的静态图，torch_xla 的重编译风暴也随之消失。")
    elif _s("E2 ") == "OK":
        print("  [静态图 NO] 运行时 mask 不跳块，块稀疏只由编译期 mask 驱动")
        print("    -> 必须走「段长量化到 128 + 布局枚举 + 编译/MaskInfo 缓存」。")
        print(f"       每布局 MaskInfo 成本见 G1: {_d('G1 ')}")
    else:
        print(f"  [静态图 未定] E2 {_s('E2 ')}: {e2[:200]}")
    print(f"  E3 真实 ragged pack: {_d('E3 ')}")
    print(f"  F1 权重账: {_d('F1 ')}")
    print(f"  F2 单层 HBM: {_d('F2 ')}")
    print(f"  F3 分片: {_d('F3 ')}")
    print(f"  H1 反向 BlockSizes: {_d('H1 ')}")
    print("=" * 78)
    _flush()
    print(f"报告已写到 {OUT_DIR}/arch_probe_report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
