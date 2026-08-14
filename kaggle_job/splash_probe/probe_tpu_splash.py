#!/usr/bin/env python
"""Kaggle TPU v5e-8 · NaViT 块对角注意力可行性探针 —— 真机实测，不做任何假设。

**这个脚本要回答的唯一问题**：NaViT 的块对角打包，在 TPU/XLA 上能不能保留它的核心
收益——即注意力代价是 Σn_i^2（每张图各算各的）而不是 L^2（整个 pack 当一整条序列稠密算）。

如果只是"算完稠密再用 mask 抹掉"，NaViT 在 TPU 上就没有意义，整条 TPU 路线要重估。
所以本脚本的重点不是"能不能跑通"，而是**块跳过是否真的省时间**（探测 T1/T2）。

判据（探测 T1/T2 会自动打印）：
    pack 长度 L、G 个等长段，理论加速比 = Σn_i^2 / L^2 = 1/G
    实测 t(块对角)/t(全通) ≈ 1/G  → 跳块生效，NaViT 在 TPU 上成立
    实测 t(块对角)/t(全通) ≈ 1    → 只是掩码没跳块，NaViT 在 TPU 上无收益

用法（在 Kaggle notebook/script kernel 里）：
    python probe_tpu_splash.py
    python probe_tpu_splash.py --json /kaggle/working/tpu_probe.json

注意：
  * 数值对拍用**小形状**（朴素参考实现要物化 [heads, L, L] 的注意力矩阵，
    真实形状下是几十 GB，放不下）；计时用**真实形状**、不带参考实现。
  * 所有计时都先 warmup 再测——XLA 首次调用含编译，不 warmup 测的是编译时间。
    （同类教训见本仓库昇腾 CANN 首调编译污染计时。）
  * 本脚本只读不写模型权重，几分钟跑完，尽量少烧 TPU 配额。
"""

from __future__ import annotations

import argparse
import functools
import inspect
import json
import os
import platform
import sys
import tempfile
import time
import traceback

import numpy as np

RESULTS: list[dict] = []
_T0 = time.time()
_BOOTSTRAP_NOTE = ""

# ── 自举：在 import jax 之前升级 jax + libtpu ──────────────────────────────────
# 为什么需要：Kaggle 默认 TPU 镜像的 libtpu 构建于 2025-06-12，而 Pallas 有一道
# **硬闸门** `is_cloud_tpu_older_than(...)`（jax/_src/pallas/mosaic/lowering.py），
# libtpu 超过一个月就直接 raise，且**没有环境变量旁路**。实测 2026-08-14 在
# v5e-8 上跑，所有 Pallas/splash 调用全部被这道闸门挡掉（jax 0.10.2 + libtpu
# Jun 12 2025），与块对角本身无关。
#
# 开关默认 False（本仓库惯例：新行为 opt-in）。要用它必须同时满足：
#   1) 这里改成 True，或设环境变量 ANIMA_TPU_UPGRADE=1
#   2) kernel-metadata.json 里 "enable_internet": "true"（需账号已手机验证）
# 失败不致命：记一条 FAIL 后继续跑，非 Pallas 的探测（A1/B1/C2/D1/M1）照样出数据。
# 当前置 True：2026-08-14 实测 Kaggle 默认镜像必然被闸门挡住，不升级就没有 Pallas。
# 等哪天 Kaggle 把镜像更新了（A1 会显示"闸门通过"），改回 False 可省掉一两分钟配额。
BOOTSTRAP_UPGRADE_JAX = os.environ.get("ANIMA_TPU_UPGRADE", "1") == "1"


def _bootstrap_upgrade_jax() -> None:
    """必须在任何 `import jax` **之前**调用——jax 一旦初始化后端就换不掉 libtpu。"""
    global _BOOTSTRAP_NOTE
    if not BOOTSTRAP_UPGRADE_JAX:
        _BOOTSTRAP_NOTE = "未开启（BOOTSTRAP_UPGRADE_JAX=False）"
        return
    if "jax" in sys.modules:
        _BOOTSTRAP_NOTE = "跳过：jax 已被导入，此时换 libtpu 无效"
        return
    import subprocess
    cmd = [sys.executable, "-m", "pip", "install", "-q", "-U", "jax[tpu]"]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except Exception as e:
        _BOOTSTRAP_NOTE = f"失败：{type(e).__name__}: {e}"
        return
    dt = time.time() - t0
    if r.returncode == 0:
        _BOOTSTRAP_NOTE = f"成功，耗时 {dt:.0f}s"
    else:
        tail = (r.stderr or r.stdout or "").strip().splitlines()[-3:]
        _BOOTSTRAP_NOTE = f"pip 退出码 {r.returncode}（{dt:.0f}s）：{' | '.join(tail)}"


_bootstrap_upgrade_jax()


def record(name: str, status: str, detail: str = "") -> None:
    RESULTS.append({"name": name, "status": status, "detail": detail,
                    "t_elapsed_s": round(time.time() - _T0, 1)})
    mark = {"OK": "  OK  ", "FAIL": " FAIL ", "SKIP": " SKIP ", "INFO": " INFO "}.get(status, status)
    print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""), flush=True)


# 所有被 @probe 声明过的探测名。main() 末尾核对它们是否都真的跑了——探测不是自动
# 发现的，必须在 main() 里显式调用；漏调只会让那一项静默消失。
_DEFINED_PROBES: list[str] = []


def probe(name: str):
    """装饰器：把一个返回 detail 字符串的函数包成一项探测。"""
    _DEFINED_PROBES.append(name)

    def deco(fn):
        @functools.wraps(fn)
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


class _Skip(Exception):
    pass


# interpret 模式：splash 的 Pallas kernel 可以在 CPU 上以解释器方式跑（数值等价、
# 极慢）。用它可以把**数值/梯度对拍在本地免费做完**，TPU 配额只留给计时项。
# 由 --interpret 设置；真机上恒为 False。
_INTERPRET = False


# ── 0. 基础环境 ────────────────────────────────────────────────────────────────

def probe_env() -> None:
    record("python", "INFO", f"{platform.python_version()} @ {sys.executable}")
    record("platform", "INFO", platform.platform())
    for var in ("TPU_NAME", "TPU_WORKER_ID", "COLAB_TPU_ADDR", "JAX_PLATFORMS",
                "LIBTPU_INIT_ARGS", "KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_DATA_PROXY_TOKEN"):
        v = os.environ.get(var)
        if v:
            # 不打印疑似凭据的完整值
            shown = v if "TOKEN" not in var else f"<set, {len(v)} chars>"
            record(f"env:{var}", "INFO", shown[:160])


@probe("import jax")
def probe_jax() -> str:
    import jax
    import jaxlib
    return f"jax {jax.__version__} / jaxlib {jaxlib.__version__}"


@probe("TPU 设备可见性")
def probe_devices() -> str:
    import jax
    devs = jax.devices()
    if not devs:
        raise RuntimeError("jax.devices() 为空")
    kinds = {d.device_kind for d in devs}
    if not any("TPU" in str(d.platform).upper() or "TPU" in str(k).upper()
               for d, k in zip(devs, [d.device_kind for d in devs])):
        raise RuntimeError(f"没有 TPU 设备，只看到 {devs}")
    return f"{len(devs)} 个设备，kind={kinds}，platform={devs[0].platform}"


@probe("单设备 HBM 容量")
def probe_hbm() -> str:
    import jax
    d = jax.local_devices()[0]
    stats = getattr(d, "memory_stats", lambda: None)()
    if not stats:
        raise _Skip("该 jax 版本/后端不提供 memory_stats()")
    lim = stats.get("bytes_limit") or stats.get("bytes_reservable_limit")
    used = stats.get("bytes_in_use", 0)
    if lim is None:
        raise _Skip(f"memory_stats 无容量字段，keys={sorted(stats)[:8]}")
    return (f"limit={lim / 2**30:.1f} GiB, in_use={used / 2**30:.2f} GiB"
            f"（v5e 官方规格 16 GB/chip，8 chip 共 128 GB）")


@probe("A1 Pallas libtpu 版本闸门")
def probe_libtpu_gate() -> str:
    """Pallas TPU 有一道硬闸门：libtpu 超过约一个月就直接 raise，**无环境变量旁路**
    （jax/_src/pallas/mosaic/lowering.py 的 `is_cloud_tpu_older_than`）。
    Kaggle 默认镜像的 libtpu 很旧，这一条不过则后面**所有** Pallas/splash 探测
    都会以同一个原因失败——那是镜像问题，不是块对角的问题。"""
    import jax
    from jax._src import xla_bridge
    ver = xla_bridge.get_backend().platform_version
    one_line = " / ".join(x.strip() for x in ver.splitlines() if x.strip())
    gated, gate_desc = None, "未知"
    if jax.devices()[0].platform != "tpu":
        # 非 TPU 后端上 is_cloud_tpu_older_than 恒 False，二分出来的日期是假象，
        # 打出来只会误导（本地 CPU 上实测会报成"2024-01-01"）。直接标 N/A。
        gate_desc = "非 TPU 后端，闸门不适用"
    else:
        try:
            import datetime
            from jax._src import cloud_tpu_init
            fn = getattr(cloud_tpu_init, "is_cloud_tpu_older_than", None)
            if fn is None:
                gate_desc = "该 jax 版本没有 is_cloud_tpu_older_than"
            else:
                client = jax.devices()[0].client
                # fn(D) == (libtpu 构建日 B < D)。二分求 B：
                #   fn(mid) 为真 => B < mid => 往前找（hi = mid）
                #   为假        => B >= mid => 往后找（lo = mid）
                d_lo, d_hi = datetime.date(2023, 1, 1), datetime.date(2031, 1, 1)
                while (d_hi - d_lo).days > 1:
                    mid = d_lo + (d_hi - d_lo) / 2
                    if fn(mid.year, mid.month, mid.day, client):
                        d_hi = mid
                    else:
                        d_lo = mid
                gate_desc = f"libtpu 构建日约 {d_lo.isoformat()}"
                gated = fn(2026, 4, 1, client)
        except Exception as e:
            gate_desc = f"探测闸门失败: {type(e).__name__}: {e}"
    verdict = ("**Pallas 被闸门挡住**" if gated else
               "Pallas 闸门通过" if gated is False else "闸门状态未知")
    return (f"jax {jax.__version__} | platform_version: {one_line} | {gate_desc} | "
            f"{verdict} | 自举升级: {_BOOTSTRAP_NOTE}")


@probe("import splash_attention")
def probe_import_splash() -> str:
    from jax.experimental.pallas.ops.tpu.splash_attention import (  # noqa: F401
        splash_attention_kernel as sk,
        splash_attention_mask as sm,
        splash_attention_mask_info as smi,
    )
    return "jax.experimental.pallas.ops.tpu.splash_attention 三个模块都可导入"


@probe("splash 默认 BlockSizes")
def probe_block_sizes() -> str:
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as sk
    bs = sk.BlockSizes.get_default()
    fields = {f: getattr(bs, f) for f in dir(bs)
              if f.startswith("block_") and isinstance(getattr(bs, f, None), int)}
    return f"{fields}（段长必须对齐到 block_kv，否则块对角会产生 partial block）"


@probe("B1 MaskInfo 块压缩率（纯 numpy，免费判据）")
def probe_maskinfo_compaction() -> str:
    """splash 的 MaskInfo 在**编译期**就把全 False 的块从 Pallas grid 里压缩掉了。
    这里直接数"存活块/稠密块"，它给出 T1/T2 应该达到的上限。

    本地 CPU 已实测（jax 0.11.0）：段长对齐到 block_kv 时，存活块/稠密块与理论
    Σn^2/L^2 精确相等、partial_mask_blocks 为 None；未对齐则多算约 26% 并产生
    partial 块。**注意这只证明 grid 变小了，不证明墙钟时间同比例下降**——后者
    由 T1/T2 在真机上裁决。"""
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_kernel as sk, splash_attention_mask as sm,
        splash_attention_mask_info as smi,
    )
    bs = sk.BlockSizes.get_default()
    BQ, BKV = bs.block_q, bs.block_kv
    BD = _make_block_diag_mask_cls()
    rows = []
    cases = [
        ("等长4段", TIME_L, _aligned_segments(TIME_L, 4, BKV)),
        ("不等长8段", TIME_L, [4096, 3072, 2048, 2048, 2048, 1536, 1024, 512]),
        ("未对齐(反例)", TIME_L, [4000, 3000, 2100, 2000, 2000, 1500, 1000, 784]),
    ]
    for label, L, lens in cases:
        if sum(lens) != L:
            rows.append(f"{label}:段长和≠L 跳过")
            continue
        mi, *_ = smi.process_mask(sm.MultiHeadMask(masks=[BD(lens)] * 2), (BQ, BKV))
        bm = np.asarray(mi.block_mask)[0]
        live = int((bm != 0).sum())
        dense = (L // BQ) * (L // BKV)
        partial = int((bm == 1).sum())
        theory = sum(n * n for n in lens) / (L * L)
        rows.append(f"{label} 存活{live / dense:.4f}/理论{theory:.4f} partial={partial}")
    return f"block=({BQ},{BKV}) | " + " ; ".join(rows)


# ── 形状常量：来自本仓库真实模型配置 ──────────────────────────────────────────
# Krea2 12B（models/krea2_modeling.py KREA2_LARGE_WIDE）：features=6144, heads=48,
#   kvheads=12（GQA 4:1）, head_dim=6144/48=128, layers=28
# Anima（models/anima_modeling_core.py 默认）：num_heads=16, head_dim=48
KREA2_Q_HEADS, KREA2_KV_HEADS, KREA2_HEAD_DIM = 48, 12, 128
ANIMA_Q_HEADS, ANIMA_HEAD_DIM = 16, 48

# 对拍用的小形状（朴素参考要物化 [h, L, L] fp32，真实形状下放不下）
REF_L, REF_HEADS, REF_DIM = 2048, 4, 128
# 计时用的真实形状：navit_token_budget 常用 16384（见 config/train_krea2_*.yaml）
TIME_L = 16384
TIME_SEGMENTS = 4          # 4 段等长 => 理论加速比 Σn^2/L^2 = 1/4 = 0.25


def _seg_ids(seg_lens, dtype=np.int32) -> np.ndarray:
    """把段长列表展开成逐 token 的 segment id（padding 段也占一个 id）。"""
    return np.repeat(np.arange(len(seg_lens), dtype=dtype), seg_lens)


def _make_block_diag_mask_cls():
    """构造一个块对角 Mask 子类（splash 的 Mask 是**编译期** numpy 对象，
    MaskInfo 由它派生，块跳过也由它驱动——这正是我们要验证的机制）。"""
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as sm

    class BlockDiagonalMask(sm.Mask):
        """段内全注意力、跨段不可见。等价于 xformers 的 BlockDiagonalMask，
        也等价于本仓库 Anima 的 sdpa_seg / npu_tnd 后端语义。"""

        def __init__(self, seg_lens):
            self.seg_lens = tuple(int(x) for x in seg_lens)
            self._ids = _seg_ids(self.seg_lens)
            n = int(self._ids.shape[0])
            self._shape = (n, n)

        @property
        def shape(self):
            return self._shape

        def __getitem__(self, idx):
            q_idx, kv_idx = idx
            q = self._ids[q_idx]
            kv = self._ids[kv_idx]
            return (q[:, None] == kv[None, :]).astype(np.bool_)

        def __eq__(self, other):
            return isinstance(other, BlockDiagonalMask) and self.seg_lens == other.seg_lens

        def __hash__(self):
            return hash((type(self), self.seg_lens))

    return BlockDiagonalMask


def _naive_attn(q, k, v, seg_ids_np, q_heads, kv_heads):
    """朴素参考实现：物化稠密 logits + 段掩码。仅用于小形状对拍。"""
    import jax
    import jax.numpy as jnp
    if kv_heads != q_heads:                      # GQA：把 kv head 广播到 q head
        rep = q_heads // kv_heads
        k = jnp.repeat(k, rep, axis=0)
        v = jnp.repeat(v, rep, axis=0)
    ids = jnp.asarray(seg_ids_np)
    allow = ids[:, None] == ids[None, :]
    # 重要：splash 的 kernel **不内置** 1/sqrt(head_dim) 缩放（见其自带的
    # _attention_reference_default：logits = einsum(q, k)，没有除以 sqrt(d)）。
    # 调用方必须自己预缩放 q。移植时漏掉这一步不会报错，只会静默改变 softmax
    # 温度——本地 interpret 对拍时实测 rel 差到 9.5，就是这个原因。
    # 这里的参考实现刻意与 splash 同口径：不缩放。
    logits = jnp.einsum("hsd,htd->hst", q.astype(jnp.float32), k.astype(jnp.float32))
    logits = jnp.where(allow[None], logits, -1e30)
    p = jax.nn.softmax(logits, axis=-1)
    return jnp.einsum("hst,htd->hsd", p, v.astype(jnp.float32))


def _splash_fn(seg_lens, q_heads, kv_heads):
    """按段长构造一个 splash MHA 可调用对象（编译期 mask 路径）。"""
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_kernel as sk, splash_attention_mask as sm,
    )
    BD = _make_block_diag_mask_cls()
    bd = BD(seg_lens)
    multi = sm.MultiHeadMask(masks=[bd] * q_heads)
    return sk.make_splash_mha(multi, head_shards=1, q_seq_shards=1, interpret=_INTERPRET)


def _rand_qkv(L, q_heads, kv_heads, dim, dtype, seed=0):
    import jax
    import jax.numpy as jnp
    ks = jax.random.split(jax.random.PRNGKey(seed), 3)
    q = jax.random.normal(ks[0], (q_heads, L, dim), dtype=jnp.float32).astype(dtype)
    k = jax.random.normal(ks[1], (kv_heads, L, dim), dtype=jnp.float32).astype(dtype)
    v = jax.random.normal(ks[2], (kv_heads, L, dim), dtype=jnp.float32).astype(dtype)
    return q, k, v


def _aligned_segments(L, n_seg, align):
    """把 L 切成 n_seg 段，每段对齐到 align 的整数倍（余数并入最后一段）。"""
    per = max(align, (L // n_seg) // align * align)
    lens = [per] * (n_seg - 1)
    lens.append(L - sum(lens))
    if lens[-1] <= 0 or lens[-1] % align != 0:
        raise ValueError(f"切不出对齐的段：L={L} n_seg={n_seg} align={align} -> {lens}")
    return lens


# ── N. 数值正确性（小形状对拍）────────────────────────────────────────────────

@probe("N1 全通 mask 数值对拍（基线）")
def probe_full_numeric() -> str:
    import jax.numpy as jnp
    q, k, v = _rand_qkv(REF_L, REF_HEADS, REF_HEADS, REF_DIM, jnp.float32, seed=1)
    lens = [REF_L]
    out = _splash_fn(lens, REF_HEADS, REF_HEADS)(q, k, v)
    ref = _naive_attn(q, k, v, _seg_ids(lens), REF_HEADS, REF_HEADS)
    err = float(jnp.max(jnp.abs(out.astype(jnp.float32) - ref)))
    rel = err / float(jnp.max(jnp.abs(ref)) + 1e-30)
    if not np.isfinite(err) or rel > 2e-2:
        raise RuntimeError(f"数值不匹配 max_abs={err:.3e} rel={rel:.3e}")
    return f"L={REF_L} 单段, max_abs={err:.3e}, rel={rel:.3e}"


@probe("N2 块对角 mask 数值对拍（前向）")
def probe_blockdiag_numeric() -> str:
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as sk
    import jax.numpy as jnp
    align = sk.BlockSizes.get_default().block_kv
    lens = _aligned_segments(REF_L, 3, align)
    q, k, v = _rand_qkv(REF_L, REF_HEADS, REF_HEADS, REF_DIM, jnp.float32, seed=2)
    out = _splash_fn(lens, REF_HEADS, REF_HEADS)(q, k, v)
    ref = _naive_attn(q, k, v, _seg_ids(lens), REF_HEADS, REF_HEADS)
    err = float(jnp.max(jnp.abs(out.astype(jnp.float32) - ref)))
    rel = err / float(jnp.max(jnp.abs(ref)) + 1e-30)
    if not np.isfinite(err) or rel > 2e-2:
        raise RuntimeError(f"数值不匹配 max_abs={err:.3e} rel={rel:.3e} segs={lens}")
    return f"段长={lens}(对齐{align}), max_abs={err:.3e}, rel={rel:.3e}"


@probe("N3 块对角 反向对拍（训练必须）")
def probe_blockdiag_grad() -> str:
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as sk
    import jax
    import jax.numpy as jnp
    align = sk.BlockSizes.get_default().block_kv
    lens = _aligned_segments(REF_L, 3, align)
    q, k, v = _rand_qkv(REF_L, REF_HEADS, REF_HEADS, REF_DIM, jnp.float32, seed=3)
    fn = _splash_fn(lens, REF_HEADS, REF_HEADS)
    ids = _seg_ids(lens)

    def loss_splash(q_, k_, v_):
        return jnp.sum(fn(q_, k_, v_).astype(jnp.float32) ** 2)

    def loss_ref(q_, k_, v_):
        return jnp.sum(_naive_attn(q_, k_, v_, ids, REF_HEADS, REF_HEADS) ** 2)

    g1 = jax.grad(loss_splash, argnums=(0, 1, 2))(q, k, v)
    g2 = jax.grad(loss_ref, argnums=(0, 1, 2))(q, k, v)
    worst, worst_name = 0.0, ""
    for name, a, b in zip("qkv", g1, g2):
        a32, b32 = a.astype(jnp.float32), b.astype(jnp.float32)
        rel = float(jnp.max(jnp.abs(a32 - b32))) / (float(jnp.max(jnp.abs(b32))) + 1e-30)
        if rel > worst:
            worst, worst_name = rel, name
    if not np.isfinite(worst) or worst > 5e-2:
        raise RuntimeError(f"梯度不匹配，最差 d{worst_name} rel={worst:.3e}")
    return f"dq/dk/dv 全部匹配，最差 d{worst_name} rel={worst:.3e}"


@probe("N4 GQA(48q/12kv) 块对角对拍")
def probe_gqa_numeric() -> str:
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as sk
    import jax.numpy as jnp
    align = sk.BlockSizes.get_default().block_kv
    lens = _aligned_segments(REF_L, 2, align)
    # Krea2 是 48q/12kv 的 GQA；用小 head 数保持同样 4:1 比例做对拍
    qh, kvh = 8, 2
    q, k, v = _rand_qkv(REF_L, qh, kvh, REF_DIM, jnp.float32, seed=4)
    out = _splash_fn(lens, qh, kvh)(q, k, v)
    ref = _naive_attn(q, k, v, _seg_ids(lens), qh, kvh)
    err = float(jnp.max(jnp.abs(out.astype(jnp.float32) - ref)))
    rel = err / float(jnp.max(jnp.abs(ref)) + 1e-30)
    if not np.isfinite(err) or rel > 2e-2:
        raise RuntimeError(f"GQA 数值不匹配 rel={rel:.3e}")
    return f"{qh}q/{kvh}kv(4:1，同 Krea2) rel={rel:.3e} → GQA 路径可用"


# ── T. 跳块提速（本脚本的核心）────────────────────────────────────────────────

def _bench(fn, args, reps=20, warmup=3):
    """先 warmup（XLA 首调含编译），再计时。返回 (中位数毫秒, 编译秒)."""
    import jax
    t_c0 = time.time()
    out = fn(*args)
    jax.block_until_ready(out)
    compile_s = time.time() - t_c0
    for _ in range(warmup - 1):
        jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts)), compile_s


@probe("T1 跳块提速 · 前向（决定性判据）")
def probe_speed_fwd() -> str:
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as sk
    import jax.numpy as jnp
    align = sk.BlockSizes.get_default().block_kv
    lens = _aligned_segments(TIME_L, TIME_SEGMENTS, align)
    q, k, v = _rand_qkv(TIME_L, KREA2_Q_HEADS, KREA2_KV_HEADS, KREA2_HEAD_DIM,
                        jnp.bfloat16, seed=10)
    t_bd, c_bd = _bench(_splash_fn(lens, KREA2_Q_HEADS, KREA2_KV_HEADS), (q, k, v))
    t_full, c_full = _bench(_splash_fn([TIME_L], KREA2_Q_HEADS, KREA2_KV_HEADS), (q, k, v))
    ratio = t_bd / t_full
    theory = sum(n * n for n in lens) / (TIME_L ** 2)
    verdict = ("跳块生效" if ratio < (theory + 1.0) / 2 else "疑似未跳块")
    return (f"L={TIME_L} {TIME_SEGMENTS}段 heads={KREA2_Q_HEADS}/{KREA2_KV_HEADS}x{KREA2_HEAD_DIM} bf16 | "
            f"块对角 {t_bd:.2f}ms vs 全通 {t_full:.2f}ms → 实测比 {ratio:.3f}，"
            f"理论 {theory:.3f} → {verdict}（编译 {c_bd:.1f}s/{c_full:.1f}s）")


@probe("T2 跳块提速 · 前向+反向")
def probe_speed_fwdbwd() -> str:
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as sk
    import jax
    import jax.numpy as jnp
    align = sk.BlockSizes.get_default().block_kv
    lens = _aligned_segments(TIME_L, TIME_SEGMENTS, align)
    q, k, v = _rand_qkv(TIME_L, KREA2_Q_HEADS, KREA2_KV_HEADS, KREA2_HEAD_DIM,
                        jnp.bfloat16, seed=11)

    def mk(seg):
        f = _splash_fn(seg, KREA2_Q_HEADS, KREA2_KV_HEADS)
        return jax.jit(jax.grad(lambda a, b, c: jnp.sum(f(a, b, c).astype(jnp.float32) ** 2),
                                argnums=(0, 1, 2)))

    t_bd, _ = _bench(mk(lens), (q, k, v), reps=10)
    t_full, _ = _bench(mk([TIME_L]), (q, k, v), reps=10)
    ratio = t_bd / t_full
    theory = sum(n * n for n in lens) / (TIME_L ** 2)
    verdict = ("跳块生效" if ratio < (theory + 1.0) / 2 else "疑似未跳块")
    return (f"块对角 {t_bd:.2f}ms vs 全通 {t_full:.2f}ms → 实测比 {ratio:.3f}，"
            f"理论 {theory:.3f} → {verdict}")


@probe("T3 段数扩展性（8/16 段）")
def probe_speed_scaling() -> str:
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as sk
    import jax.numpy as jnp
    align = sk.BlockSizes.get_default().block_kv
    q, k, v = _rand_qkv(TIME_L, KREA2_Q_HEADS, KREA2_KV_HEADS, KREA2_HEAD_DIM,
                        jnp.bfloat16, seed=12)
    t_full, _ = _bench(_splash_fn([TIME_L], KREA2_Q_HEADS, KREA2_KV_HEADS), (q, k, v), reps=10)
    parts = []
    for g in (8, 16):
        try:
            lens = _aligned_segments(TIME_L, g, align)
        except ValueError as e:
            parts.append(f"{g}段:切不出({e})")
            continue
        t, _ = _bench(_splash_fn(lens, KREA2_Q_HEADS, KREA2_KV_HEADS), (q, k, v), reps=10)
        parts.append(f"{g}段 {t:.2f}ms(比{t / t_full:.3f}/理论{1 / g:.3f})")
    return f"全通 {t_full:.2f}ms | " + " ; ".join(parts)


# ── C. 编译行为（决定"变长 pack"在 XLA 下要付多少代价）────────────────────────

@probe("C1 换段长布局是否触发重编译")
def probe_recompile() -> str:
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as sk
    import jax.numpy as jnp
    align = sk.BlockSizes.get_default().block_kv
    q, k, v = _rand_qkv(TIME_L, ANIMA_Q_HEADS, ANIMA_Q_HEADS, ANIMA_HEAD_DIM,
                        jnp.bfloat16, seed=20)
    # 三种不同的段长布局，同一个 L —— 现实中每个 pack 的组成都不同
    layouts = [
        _aligned_segments(TIME_L, 4, align),
        [align * 2, align * 6, TIME_L - align * 8],
        [align * 4, align * 4, align * 4, TIME_L - align * 12],
    ]
    costs = []
    for lens in layouts:
        _, c = _bench(_splash_fn(lens, ANIMA_Q_HEADS, ANIMA_Q_HEADS), (q, k, v), reps=2, warmup=1)
        costs.append(c)
    # 注意口径：_splash_fn(...) 在 _bench 之前就返回了，而 make_splash_mha 里的
    # MaskInfo 预处理是在那时候（同步、numpy）做掉的 —— **不在这个计时区间里**。
    # 所以这里量到的只是 XLA/Mosaic 编译，mask 预处理成本见 C3。
    return (f"3 种布局首调耗时 {[f'{c:.1f}s' for c in costs]}（**仅** XLA/Mosaic 编译，"
            f"不含 mask 预处理，后者见 C3）→ 每个布局一次")


@probe("C3 mask 预处理成本（每布局，逐步都要付）")
def probe_mask_prep_cost() -> str:
    """`make_splash_mha` 里的 MaskInfo 预处理是同步 numpy，且随 L 与 head 数增长。
    如果训练时每个 pack 都重建 mask，这个成本是**逐步**都要付的，会直接叠到步时上。
    本地 CPU 实测 L=16384/48heads 约 500ms —— 与 T2 的 288ms 注意力步时同量级，
    不能忽略。规避办法二选一：① 按布局缓存已构建的可调用对象；② 走 D2 的运行时
    jax.Array mask 路径。"""
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as sk
    align = sk.BlockSizes.get_default().block_kv
    rows = []
    for label, lens, heads in (
        ("L=16384 4段 48heads", _aligned_segments(TIME_L, 4, align), KREA2_Q_HEADS),
        ("L=16384 16段 48heads", _aligned_segments(TIME_L, 16, align), KREA2_Q_HEADS),
    ):
        t0 = time.perf_counter()
        _ = _splash_fn(lens, heads, heads)
        rows.append(f"{label}: {(time.perf_counter() - t0) * 1e3:.0f}ms")
    return " ; ".join(rows) + "（与 T2 步时同量级则必须按布局缓存或走 D2 路径）"


@probe("C2 JAX 持久化编译缓存是否命中")
def probe_persistent_cache() -> str:
    import jax
    import jax.numpy as jnp
    # 只有真的在 Kaggle 上才写 /kaggle/working；否则落到临时目录，免得在别人机器上
    # 平白造出一个 /kaggle 目录（本地 Windows 上实测会建到 D:\kaggle）。
    default_dir = ("/kaggle/working/jax_cache" if os.path.isdir("/kaggle/working")
                   else os.path.join(tempfile.gettempdir(), "anima_jax_cache"))
    cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR", default_dir)
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except OSError as e:
        raise _Skip(f"建不了缓存目录 {cache_dir}: {e}")
    try:
        jax.config.update("jax_compilation_cache_dir", cache_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    except Exception as e:
        raise _Skip(f"该 jax 版本不支持持久化编译缓存: {e}")
    n_before = sum(len(f) for _, _, f in os.walk(cache_dir))

    @jax.jit
    def f(x):
        return jnp.sin(x) @ jnp.cos(x).T

    x = jnp.ones((512, 512), jnp.float32)
    jax.block_until_ready(f(x))
    n_after = sum(len(f_) for _, _, f_ in os.walk(cache_dir))
    if n_after > n_before:
        how = f"新写入（{n_before}→{n_after}）"
    elif n_after > 0:
        # 同一份计算已在缓存里 → 文件数不变才是**命中**，不是失败
        how = f"命中已有缓存（{n_after} 个文件，未新增）"
    else:
        raise RuntimeError(f"缓存目录始终为空（{cache_dir}），持久化缓存没生效")
    return (f"{cache_dir} {how} → 编译产物可跨 run 复用"
            f"（落到 /kaggle/working 并作为下一棒的 dataset 输入，可摊掉编译成本）")


# ── D. 动态 mask API 面（能否免掉"每个布局一次编译"）──────────────────────────

@probe("D1 splash 动态 mask API 面")
def probe_dynamic_mask_api() -> str:
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_kernel as sk, splash_attention_mask_info as smi,
    )
    notes = []
    mi_fields = getattr(smi.MaskInfo, "_fields", ())
    notes.append(f"MaskInfo 字段={list(mi_fields)}")
    notes.append("有 is_dynamic_mask" if "is_dynamic_mask" in mi_fields else "无 is_dynamic_mask")
    try:
        sig = inspect.signature(sk.make_splash_mha)
        notes.append(f"make_splash_mha{sig}")
    except Exception as e:
        notes.append(f"取签名失败: {e}")
    # 是否存在接受 jax.Array 作为 mask 的处理入口（决定能否传运行时 mask）
    dyn_entries = [n for n in dir(smi) if "dynamic" in n.lower()]
    notes.append(f"mask_info 中含 dynamic 的符号={dyn_entries or '无'}")
    return " | ".join(notes)


@probe("D2 直接传 jax.Array 作为 mask（运行时布局）")
def probe_dynamic_mask_call() -> str:
    """如果这条通过，就不需要"每个段长布局编译一次"，pack 组成可以完全运行时可变。
    这是最理想但最不确定的一条——失败不代表 NaViT 不可行，只是要走 C1/C2 的枚举+缓存路线。"""
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_kernel as sk, splash_attention_mask_info as smi,
    )
    import jax.numpy as jnp
    align = sk.BlockSizes.get_default().block_kv
    lens = _aligned_segments(REF_L, 2, align)
    ids = _seg_ids(lens)
    dense = jnp.asarray(ids[:, None] == ids[None, :])
    dense = jnp.broadcast_to(dense, (REF_HEADS, REF_L, REF_L))
    try:
        fn = sk.make_splash_mha(dense, head_shards=1, q_seq_shards=1, interpret=_INTERPRET)
    except Exception as e:
        raise _Skip(f"make_splash_mha 不接受 jax.Array mask：{type(e).__name__}: {e}")
    q, k, v = _rand_qkv(REF_L, REF_HEADS, REF_HEADS, REF_DIM, jnp.float32, seed=30)
    out = fn(q, k, v)
    ref = _naive_attn(q, k, v, ids, REF_HEADS, REF_HEADS)
    rel = float(jnp.max(jnp.abs(out.astype(jnp.float32) - ref))) / (float(jnp.max(jnp.abs(ref))) + 1e-30)
    if not np.isfinite(rel) or rel > 2e-2:
        raise RuntimeError(f"运行时 mask 数值不匹配 rel={rel:.3e}")
    return f"接受 jax.Array mask 且数值正确 rel={rel:.3e} —— 可免枚举布局"


@probe("D3 segment_ids 参数是否可用（不跳块的保底路径）")
def probe_segment_ids() -> str:
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_kernel as sk,
    )
    import jax.numpy as jnp
    lens = _aligned_segments(REF_L, 2, sk.BlockSizes.get_default().block_kv)
    ids = jnp.asarray(_seg_ids(lens))
    fn = _splash_fn([REF_L], REF_HEADS, REF_HEADS)      # 全通静态 mask + 运行时 segment_ids
    q, k, v = _rand_qkv(REF_L, REF_HEADS, REF_HEADS, REF_DIM, jnp.float32, seed=31)
    out = fn(q, k, v, sk.SegmentIds(q=ids, kv=ids))
    ref = _naive_attn(q, k, v, _seg_ids(lens), REF_HEADS, REF_HEADS)
    rel = float(jnp.max(jnp.abs(out.astype(jnp.float32) - ref))) / (float(jnp.max(jnp.abs(ref))) + 1e-30)
    if not np.isfinite(rel) or rel > 2e-2:
        raise RuntimeError(f"segment_ids 数值不匹配 rel={rel:.3e}")
    return (f"数值正确 rel={rel:.3e}；注意这条**大概率不跳块**"
            f"（块稀疏由编译期 mask 驱动），只作语义保底")


# ── M. 多设备 ─────────────────────────────────────────────────────────────────

@probe("M1 8 卡分片 smoke")
def probe_sharding() -> str:
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    devs = jax.devices()
    if len(devs) < 8:
        raise _Skip(f"只有 {len(devs)} 个设备，跳过 8 卡分片")
    mesh = Mesh(np.array(devs).reshape(8), ("data",))
    s = NamedSharding(mesh, P("data"))
    x = jax.device_put(jnp.ones((8192, 1024), jnp.bfloat16), s)
    y = jax.block_until_ready(jax.jit(lambda a: (a * 2).sum(axis=1))(x))
    return f"mesh=(8,) 分片计算通过，结果 shape={y.shape}，devices={len(devs)}"


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Kaggle TPU v5e-8 NaViT 块对角注意力探针")
    ap.add_argument("--json", default="", help="把结果写成 JSON（建议 /kaggle/working/ 下）")
    ap.add_argument("--skip-slow", action="store_true", help="跳过 T3/C1 这些耗时项")
    ap.add_argument("--interpret", action="store_true",
                    help="用 Pallas interpret 模式在 CPU 上跑（数值等价、极慢）。"
                         "只做数值/梯度对拍，自动跳过所有计时项——本地免费预验证用，"
                         "真机上**不要**加这个开关。")
    ap.add_argument("--outdir", default="", help="报告落盘目录（Kaggle 上自动用 /kaggle/working）")
    ap.add_argument("--ref-len", type=int, default=0,
                    help="覆盖对拍序列长度（interpret 模式很慢，可设 256/512）")
    args = ap.parse_args()

    global _INTERPRET, REF_L
    _INTERPRET = bool(args.interpret)
    if args.ref_len:
        REF_L = int(args.ref_len)

    print("=" * 78)
    print("Kaggle TPU v5e-8 · NaViT 块对角注意力可行性探针")
    print("=" * 78)

    probe_env()
    has_jax = probe_jax()
    if not has_jax:
        record("总结", "FAIL", "jax 都导不进来，后续全部跳过")
        return 1

    probe_devices()
    probe_hbm()
    probe_libtpu_gate()
    ok_splash = probe_import_splash()
    if ok_splash:
        probe_block_sizes()
        probe_maskinfo_compaction()
        probe_full_numeric()
        probe_blockdiag_numeric()
        probe_blockdiag_grad()
        probe_gqa_numeric()
        if _INTERPRET:
            record("计时项 T1/T2/T3/C1", "SKIP",
                   "interpret 模式下 kernel 走 CPU 解释器，计时无意义（只验数值）")
        else:
            probe_speed_fwd()
            probe_speed_fwdbwd()
            if not args.skip_slow:
                probe_speed_scaling()
                probe_recompile()
                probe_mask_prep_cost()
        probe_persistent_cache()
        probe_dynamic_mask_api()
        probe_dynamic_mask_call()
        probe_segment_ids()
    probe_sharding()

    # 自检：@probe 声明了但 main() 忘了调用的探测
    _ran = {r["name"] for r in RESULTS}
    _missed = [n for n in _DEFINED_PROBES if n not in _ran]
    if _missed and ok_splash and not args.skip_slow and not _INTERPRET:
        print("=" * 78)
        print("[!] 以下探测已定义但 main() 没有调用，本次结果里缺这几项：")
        for n in _missed:
            print(f"  - {n}")
        print("  （不是这台机器的限制，是探针脚本自己的疏漏）")

    ok = sum(1 for r in RESULTS if r["status"] == "OK")
    fail = sum(1 for r in RESULTS if r["status"] == "FAIL")
    skip = sum(1 for r in RESULTS if r["status"] == "SKIP")
    print("=" * 78)
    print(f"汇总：OK={ok}  FAIL={fail}  SKIP={skip}  总耗时={time.time() - _T0:.0f}s")
    if fail:
        print("\n失败项（这些是这台机器的真实限制）：")
        for r in RESULTS:
            if r["status"] == "FAIL":
                print(f"  - {r['name']}: {r['detail']}")

    # ── 裁决：把探针结果翻译成"TPU 路线该怎么走" ──────────────────────────────
    def _st(prefix: str) -> str:
        for r in RESULTS:
            if r["name"].startswith(prefix):
                return r["status"]
        return "MISSING"

    def _detail(prefix: str) -> str:
        for r in RESULTS:
            if r["name"].startswith(prefix):
                return r["detail"]
        return ""

    print("\n" + "=" * 78)
    print("NaViT-on-TPU 裁决：")
    correct = all(_st(p) == "OK" for p in ("N2 ", "N3 ", "N4 "))
    t1 = _detail("T1 ")
    skipped_blocks = "跳块生效" in t1
    # libtpu 闸门是"环境不具备"，不是"块对角不行"——绝不能把它读成技术结论
    gate_blocked = "Pallas 被闸门挡住" in _detail("A1 ")
    pallas_err = any(r["status"] == "FAIL" and "libtpu version" in r["detail"]
                     for r in RESULTS)
    if gate_blocked or pallas_err:
        print("  [ENV]  **环境问题，非技术结论**：Kaggle 镜像的 libtpu 太旧，")
        print("         Pallas 硬闸门把所有 splash 调用挡在门外（与块对角无关）。")
        print(f"         {_detail('A1 ')}")
        print("         修法：kernel-metadata.json 设 enable_internet=true，并把")
        print("         BOOTSTRAP_UPGRADE_JAX 置 True（或环境变量 ANIMA_TPU_UPGRADE=1），")
        print("         在 import jax 之前 pip install -U 'jax[tpu]'。")
        print("         本次仍然有效的数据：A1/B1/C2/D1/M1 与设备/HBM 信息。")
    elif not correct:
        print("  [NO]   块对角数值/梯度对拍未全过 -> 先修正确性，性能结论无意义")
    elif _INTERPRET:
        # 计时项没跑，这里绝不能把"没有 T1 结果"读成"跳块没生效"
        print("  [PART] interpret 模式：数值+梯度对拍全过，但**没有测跳块提速**")
        print(f"         B1 块压缩率给出的期望上限：{_detail('B1 ')}")
        print("         结论待真机 T1/T2 裁决，现在还不能说 NaViT 在 TPU 上成立")
    elif skipped_blocks:
        print("  [YES]  块对角数值+梯度正确，且**跳块真实生效**")
        print("         -> NaViT 在 TPU 上成立，可以继续推进 TPU 路线")
        if _st("D2 ") == "OK":
            print("         -> 且支持运行时 mask：pack 组成可完全动态，无需枚举布局")
        else:
            print("         -> 运行时 mask 不可用：需把段长布局收敛成可枚举的有限集合，")
            print("            配合持久化编译缓存（C2）摊掉每布局一次的编译成本")
    else:
        print("  [NO]   数值正确但**跳块没生效**（实测比≈1，等于稠密算完再掩码）")
        print("         -> NaViT 在 TPU 上拿不到收益。要么自己写 Pallas 块对角 kernel，")
        print("            要么放弃 TPU 路线。别在这个状态下开始重构。")
    print("=" * 78)

    # ── 落盘 ──────────────────────────────────────────────────────────────────
    # script kernel 是被裸 `python probe_tpu_splash.py` 拉起的，**拿不到命令行参数**，
    # 所以 --json 在真机上永远不会生效。这里在 Kaggle 上自动把报告写进
    # /kaggle/working（会被 kernels output 收走），不依赖内核日志。
    out_dir = args.outdir or ("/kaggle/working" if os.path.isdir("/kaggle/working") else "")
    targets = []
    if args.json:
        targets.append(("json", args.json))
    if out_dir:
        targets.append(("json", os.path.join(out_dir, "tpu_probe.json")))
        targets.append(("txt", os.path.join(out_dir, "tpu_probe_report.txt")))
    for kind, path in targets:
        try:
            with open(path, "w", encoding="utf-8") as f:
                if kind == "json":
                    json.dump(RESULTS, f, ensure_ascii=False, indent=2)
                else:
                    for r in RESULTS:
                        f.write(f"[{r['status']:^6}] {r['name']}"
                                + (f" — {r['detail']}" if r["detail"] else "") + "\n")
                    f.write(f"\n汇总：OK={ok} FAIL={fail} SKIP={skip} "
                            f"总耗时={time.time() - _T0:.0f}s\n")
            print(f"已写出 {path}")
        except OSError as e:
            print(f"写 {path} 失败：{e}")

    # **恒返回 0**：本脚本的产出是"哪些能力可用"这张表，单项 FAIL 是数据不是脚本故障。
    # 返回非零会让 Kaggle 把整个 run 标成 ERROR（实测第一次真机跑就是这样：脚本明明
    # 跑完了全部探测，却因为 return 1 被判为失败）。
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
