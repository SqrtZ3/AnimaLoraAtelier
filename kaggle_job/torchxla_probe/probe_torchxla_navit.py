#!/usr/bin/env python
"""Kaggle TPU v5e-8 · torch_xla 路线可行性探针 —— 真机实测，不做任何假设。

**要回答的问题**：JAX 侧已实测块对角跳块真实生效（实测比 0.252 vs 理论 0.250）。
那么走 torch_xla 能不能拿到**同样的收益**？能的话就不必把 29k 行 torch 代码重写成 JAX。

分三层，逐层加码：

  L1 torch_xla 本身能不能在 Kaggle v5e-8 上跑起来（含与 jax 共存）。
  L2 torch_xla **自带**的 splash_attention 包装器 —— 读源码可知它只支持
     CausalMask / FullMask / LocalMask，段信息只能走**运行时** SegmentIds，
     编译期 mask 传的是 FullMask。按 JAX 侧已验证的机理，这条**预期不跳块**。
     这里要把"预期"变成实测数字。
  L3 绕开它的默认包装器，用 torch_xla 的 call_jax 桥接**我们自己的**块对角
     JAX 函数。这条通了，torch_xla 路线就成立。

已知的两个真风险（本脚本按它们设计）：
  * **版本冲突**：torch_xla master 钉 libtpu 0.0.24.dev20250929 + jax 0.8.0.dev20251001；
    与 JAX 探针里升到的 jax 0.11 / libtpu 2026-07 不是一套。混装可能 PJRT 报错。
  * **挂死**：PyTorch/XLA 文档明确要求导入 jax 前先调 jax_import_guard()，
    否则 jax 会锁住 TPU、torch_xla 拿不到设备，程序挂起（Kaggle 上会一直烧到超时）。

因此：**每完成一项探测就立刻把报告落盘**，被超时杀掉也能拿到已完成的部分。
风险最高的项放在最后。

用法（Kaggle script kernel，无参数）：直接被 `python probe_torchxla_navit.py` 拉起。
"""

from __future__ import annotations

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
# L3.2 实测的块对角注意力时间（ms）。L4.1 拿它当硬下限：一个**包含**这段注意力
# 的完整层，步时不可能显著快过注意力本身。峰值 FLOPS 推的下限太松（4.19ms vs
# 真值 108ms，差 25 倍），拦不住"没同步"这类假测量 —— 第三跑就是栽在这里。
_T_ATTN_BD: float | None = None

# 与 JAX 探针保持同一套形状，结果才可比。
# Krea2 12B: features=6144, heads=48, kvheads=12(GQA 4:1), head_dim=128
KREA2_Q_HEADS, KREA2_KV_HEADS, KREA2_HEAD_DIM = 48, 12, 128
TIME_L = 16384          # navit_token_budget 常用值
TIME_SEGMENTS = 4       # 4 等长段 => 理论 Sigma n^2/L^2 = 0.25
BLOCK = 128             # 与 JAX 侧一致；torch_xla 默认 2048 对 navit 段对齐太粗

# 落盘目录：Kaggle 上写 /kaggle/working（会被 kernels output 收走）
OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else tempfile.gettempdir()

# 安装 torch_xla。Kaggle 默认镜像没有它。默认开启——不装就什么都测不了。
BOOTSTRAP_INSTALL = os.environ.get("ANIMA_XLA_INSTALL", "1") == "1"

# **版本剪刀**（第一跑实测踩到的核心矛盾）：
#   * torch_xla 的 PJRT 层绑死一个较老的 libtpu —— v2.9.0 的 setup.py 钉
#     libtpu 0.0.21 (20250813)，真机上报的正是 "Built on Aug 15 2025"。
#   * jax 的 Pallas 闸门要求 libtpu 不超过一个月新。Kaggle 镜像自带 jax 0.10.2，
#     它的闸门直接拒掉 2025-08 的 libtpu，于是 call_jax 里的 splash 全挂。
# 所以**不能升 libtpu**（会破坏 torch_xla 的 PJRT），只能**把 jax 降到与之配套的
# 版本**。v2.9.0 的 setup.py 同时钉了 jax/jaxlib 0.7.1（同为 20250813），就用它。
# 注意装 jax 时**不要**带 [tpu] extra —— 那会拉进 jax 自己的 libtpu，把 torch_xla
# 的覆盖掉，剪刀又合上了。
# 第二跑实测：jax 0.7.1 过了日期闸门，但撞上更深一层 ——
#   Failed to deserialize the Mosaic module: Unsupported version: expected <= 7 but got 8
# 即 jax 0.7.1 的 Pallas 发出 Mosaic IR v8，而 torch_xla 2.9.0 带的 libtpu 只认 <= v7。
# （torch_xla 的 setup.py 写着配 jax 0.7.1，但 PyPI 上的 0.7.1 补丁版已经吐 v8。）
# 所以不能再一次 run 只试一个版本 —— 改成在**同一次 run 里扫一遍候选版本**：
# 每个候选装上后，在**子进程**里编一个最小 splash kernel，看 Mosaic 版本能否被接受。
# 子进程独占 TPU 后退出释放，所以这一步必须在父进程 import torch_xla **之前**做完。
INSTALL_TORCH_XLA = "torch~=2.9.0 torch_xla[tpu]~=2.9.0"
JAX_CANDIDATES = [c for c in os.environ.get(
    "ANIMA_JAX_CANDIDATES", "0.7.0,0.6.2,0.6.1,0.6.0,0.5.3").split(",") if c.strip()]

# 子进程里跑的最小 splash 编译测试：能打印 SPLASH_OK 就说明这个 jax 与当前 libtpu 配套。
# **smoke 必须走 L3 真正用的那条路**（第二跑的教训之二）：原版只编一个 FullMask，
# 但 L3 用的是自定义 sm.Mask 子类 + use_fused_bwd_kernel + vmap + 反向。
# 候选 jax 跨 0.5.x~0.7.x，splash 的 Mask API 差异不小，FullMask 编得过不代表
# 块对角编得过。所以这里直接用块对角 + 反向做 smoke，缩到 L=512 保证快。
# 判据收紧成："扫描通过 ⇒ L3 必定能编"，避免假绿灯。
_SPLASH_SMOKE = r"""
import sys
import numpy as np
try:
    import jax, jax.numpy as jnp
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_kernel as sk, splash_attention_mask as sm)

    class BD(sm.Mask):
        def __init__(self, lens):
            self.lens = tuple(int(x) for x in lens)
            self._ids = np.repeat(np.arange(len(self.lens), dtype=np.int32), self.lens)
            n = int(self._ids.shape[0]); self._shape = (n, n)
        @property
        def shape(self): return self._shape
        def __getitem__(self, idx):
            a, b = self._ids[idx[0]], self._ids[idx[1]]
            return (a[:, None] == b[None, :]).astype(np.bool_)
        def __eq__(self, o): return isinstance(o, BD) and self.lens == o.lens
        def __hash__(self): return hash((type(self), self.lens))

    L, H, D = 512, 2, 128
    mask = sm.MultiHeadMask(masks=(BD([256, 256]),) * H)
    bs = sk.BlockSizes(block_q=128, block_kv=128, block_kv_compute=128,
                       block_q_dkv=128, block_kv_dkv=128, block_kv_dkv_compute=128,
                       use_fused_bwd_kernel=True)
    f = sk.make_splash_mha(mask, head_shards=1, q_seq_shards=1, block_sizes=bs)
    q = jnp.zeros((1, H, L, D), jnp.bfloat16)
    vf = jax.vmap(f)                                   # L3 里就是这么调的
    jax.block_until_ready(vf(q, q, q))                 # 前向
    g = jax.grad(lambda a: jnp.sum(vf(a, a, a).astype(jnp.float32) ** 2))
    jax.block_until_ready(g(q))                        # 反向（训练必需）
    print("SPLASH_OK", jax.__version__, jax.devices()[0].device_kind)
except Exception as e:
    msg = str(e).replace(chr(10), " ")[:200]
    print("SPLASH_FAIL", type(e).__name__, msg)
"""


def _flush() -> None:
    """把当前结果写盘。**每项探测后都调**——这个探针有挂死风险，
    被超时杀掉时已完成的部分必须留得下来。"""
    try:
        with open(os.path.join(OUT_DIR, "xla_probe.json"), "w", encoding="utf-8") as f:
            json.dump(RESULTS, f, ensure_ascii=False, indent=2)
        with open(os.path.join(OUT_DIR, "xla_probe_report.txt"), "w", encoding="utf-8") as f:
            for r in RESULTS:
                f.write(f"[{r['status']:^6}] {r['name']}"
                        + (f" - {r['detail']}" if r["detail"] else "") + "\n")
            f.write(f"\n（截至 {time.time() - _T0:.0f}s 的快照；若报告到此为止，"
                    f"说明后面的探测挂死或被超时杀掉）\n")
    except OSError:
        pass


def record(name: str, status: str, detail: str = "") -> None:
    RESULTS.append({"name": name, "status": status, "detail": detail,
                    "t_elapsed_s": round(time.time() - _T0, 1)})
    mark = {"OK": "  OK  ", "FAIL": " FAIL ", "SKIP": " SKIP ", "INFO": " INFO "}.get(status, status)
    print(f"[{mark}] {name}" + (f" - {detail}" if detail else ""), flush=True)
    _flush()


_DEFINED_PROBES: list[str] = []


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


class _Skip(Exception):
    pass


def _bootstrap_install() -> None:
    """装 torch_xla。必须在 import torch_xla 之前。"""
    global _BOOTSTRAP_NOTE
    if not BOOTSTRAP_INSTALL:
        _BOOTSTRAP_NOTE = "未开启"
        return
    cmd = [sys.executable, "-m", "pip", "install", "-q"] + INSTALL_TORCH_XLA.split()
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except Exception as e:
        _BOOTSTRAP_NOTE = f"torch_xla 安装异常 {type(e).__name__}: {e}"
        return
    dt = time.time() - t0
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "").strip().splitlines()[-3:]
        _BOOTSTRAP_NOTE = f"torch_xla 安装失败 退出码 {r.returncode}（{dt:.0f}s）: {' | '.join(tail)}"
        return
    _BOOTSTRAP_NOTE = f"torch_xla OK {dt:.0f}s（{INSTALL_TORCH_XLA}）"


def _pip(spec: str, timeout=900) -> tuple[bool, str]:
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + spec.split(),
                       capture_output=True, text=True, timeout=timeout)
    if r.returncode == 0:
        return True, "ok"
    tail = (r.stderr or r.stdout or "").strip().splitlines()[-2:]
    return False, " | ".join(tail)


def sweep_jax_versions() -> str | None:
    """在**父进程 import torch_xla 之前**扫一遍 jax 候选版本，返回第一个能和当前
    libtpu 一起把 splash kernel 编出来的版本（None = 全军覆没）。

    为什么要子进程：TPU 是独占的，而 jax 一旦在某进程里初始化就换不掉 libtpu。
    子进程跑完退出即释放 TPU，可以逐个试。
    """
    for ver in JAX_CANDIDATES:
        ok, err = _pip(f"jax=={ver} jaxlib=={ver}")   # 不带 [tpu]：保住 torch_xla 的 libtpu
        if not ok:
            record(f"S jax {ver}", "SKIP", f"pip 装不上：{err}")
            continue
        try:
            r = subprocess.run([sys.executable, "-c", _SPLASH_SMOKE],
                               capture_output=True, text=True, timeout=600)
        except Exception as e:
            record(f"S jax {ver}", "FAIL", f"子进程异常 {type(e).__name__}: {e}")
            continue
        out = (r.stdout or "").strip().splitlines()
        line = next((l for l in out if l.startswith("SPLASH_")), "(无输出)")
        if line.startswith("SPLASH_OK"):
            record(f"S jax {ver}", "OK", f"splash 编译通过 -> {line}")
            return ver
        record(f"S jax {ver}", "FAIL", line[:220])
    return None


# ── L1 环境 ───────────────────────────────────────────────────────────────────

def probe_env() -> None:
    record("python", "INFO", f"{platform.python_version()} @ {sys.executable}")
    record("platform", "INFO", platform.platform())
    record("bootstrap 安装 torch_xla", "INFO", _BOOTSTRAP_NOTE)


@probe("L1.1 import torch / torch_xla")
def probe_import() -> str:
    import torch
    import torch_xla
    return (f"torch {torch.__version__} / torch_xla "
            f"{getattr(torch_xla, '__version__', 'unknown')}")


@probe("L1.2 XLA 设备可见性")
def probe_devices() -> str:
    import torch_xla.core.xla_model as xm
    dev = xm.xla_device()
    n = len(xm.get_xla_supported_devices() or [])
    return f"xla_device={dev}，可见设备数={n}"


@probe("L1.3 torch_xla 基础算子 + 反向")
def probe_basic() -> str:
    import torch
    import torch_xla.core.xla_model as xm
    d = xm.xla_device()
    x = torch.randn(1024, 1024, device=d, requires_grad=True)
    y = (x @ x.T).sum()
    y.backward()
    g = float(x.grad.abs().mean().cpu())
    if not np.isfinite(g):
        raise RuntimeError(f"梯度非有限：{g}")
    return f"matmul+backward 通过，grad.abs().mean()={g:.4f}"


@probe("L1.4 jax_import_guard + jax 共存")
def probe_jax_coexist() -> str:
    """PyTorch/XLA 文档：导入 jax 模块前必须先调 jax_import_guard()，否则 jax
    会锁住 TPU、torch_xla 拿不到设备而挂起。这一项要是挂了，L3 就别想了。"""
    from torch_xla.experimental.custom_kernel import jax_import_guard
    jax_import_guard()
    import jax
    devs = jax.devices()
    return (f"guard 后 jax {jax.__version__} 可用，jax.devices()={len(devs)} 个"
            f"（{devs[0].device_kind if devs else 'N/A'}）")


# ── 计时工具 ──────────────────────────────────────────────────────────────────

def _xla_bench(fn, reps=10, warmup=3):
    """torch_xla 惰性执行，必须 mark_step + 同步才能测到真时间。

    **必须消费输出**（第二跑踩的坑）：只调 fn() + mark_step() 而不用返回值，
    XLA 会把整个注意力当死代码消掉，测出来是 0.17ms —— 同形状 JAX 侧实测
    106.81ms，差 600 倍，物理上不可能。这里把结果 .sum() 搬回 CPU 强制物化。
    调用方还应配合 _implausible() 做量级自检。
    """
    import torch
    import torch_xla.core.xla_model as xm

    def once():
        out = fn()
        s = out.to(torch.float32).sum() if hasattr(out, "to") else out.sum()
        xm.mark_step()
        return float(s.cpu())          # .cpu() 强制同步 + 物化，杜绝 DCE

    for _ in range(warmup):
        once()
    xm.wait_device_ops()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        once()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


def _implausible(ms: float, L: int, qh: int, dim: int, segs: int) -> str | None:
    """量级自检：时间低于**物理下限**就说明没真算（DCE 或没同步）。

    块对角 fwd 的 FLOPs 约 2*2*qh*dim*Sigma n_i^2（QK^T 与 PV 各一次矩阵乘）。
    下限用 v5e 单 chip bf16 **峰值** 197 TFLOPS —— 达成率不可能超过 100%，
    所以 flops/peak 是时间的硬下界。再放宽 2 倍容错（我这个 FLOPs 估算可能偏大）。

    口径校准（别再搞错方向）：L=16384/4段/48头/128维 时 flops≈1.65 TFLOP，
    峰值下 8.4ms，阈值取 4.2ms。JAX 侧实测块对角 106.81ms（≈7.8% 达成率）
    远在阈值之上 -> 放行；第二跑那个 0.17ms 远在阈值之下 -> 拦下。
    最初我误用"5% 达成率"当下界得到 167ms，反而会把正确测量也拦掉。
    """
    n = L // segs
    flops = 4.0 * qh * dim * (segs * n * n)
    floor_ms = flops / 197e12 * 1e3 * 0.5      # 峰值时间的一半 = 绝无可能更快
    if ms < floor_ms:
        return (f"实测 {ms:.2f}ms 低于物理下限 {floor_ms:.2f}ms"
                f"（v5e 峰值 197TFLOPS 的 2 倍宽容）-> 没真算（DCE/未同步），数字不可信")
    return None


def _seg_lens(L, n, align):
    per = (L // n) // align * align
    lens = [per] * (n - 1) + [L - per * (n - 1)]
    assert sum(lens) == L and all(x % align == 0 for x in lens), lens
    return lens


def _seg_ids_t(seg_lens):
    import torch
    return torch.from_numpy(
        np.repeat(np.arange(len(seg_lens), dtype=np.int32), seg_lens)).to(torch.int32)


def _qkv(L, qh, kvh, dim):
    import torch
    import torch_xla.core.xla_model as xm
    d = xm.xla_device()
    g = torch.Generator().manual_seed(0)
    q = torch.randn(1, qh, L, dim, generator=g).to(torch.bfloat16).to(d)
    k = torch.randn(1, kvh, L, dim, generator=g).to(torch.bfloat16).to(d)
    v = torch.randn(1, kvh, L, dim, generator=g).to(torch.bfloat16).to(d)
    return q, k, v


# ── L2 torch_xla 自带的 splash 包装器 ─────────────────────────────────────────

@probe("L2.1 自带 splash_attention 可用性")
def probe_builtin_splash() -> str:
    from torch_xla.experimental.splash_attention import (  # noqa: F401
        splash_attention, SplashAttentionConfig,
    )
    return "torch_xla.experimental.splash_attention 可导入"


@probe("L1.5 libtpu / jax 版本剪刀")
def probe_version_scissors() -> str:
    """第一跑的杀手：torch_xla 绑老 libtpu，jax 的 Pallas 闸门要新 libtpu，两者
    差 12 个月。这一项在跑 L3 之前先把状态说清楚——它不过，L3 必然全挂。"""
    import datetime
    import re
    import jax
    from jax._src import xla_bridge
    ver = xla_bridge.get_backend().platform_version
    one_line = " / ".join(x.strip() for x in ver.splitlines() if x.strip())

    # **第二跑的教训之一**：原来用二分反推构建日，调的是
    # `is_cloud_tpu_older_than(y, m, d, client)` —— 真机报
    # `TypeError: takes 3 positional arguments but 4 were given`（各 jax 版本签名不同）。
    # 而 platform_version 字符串里**本来就写着** "Built on Aug 15 2025"，直接解析即可，
    # 既准确又不依赖 jax 私有 API 的签名。
    m = re.search(r"Built on (\w{3})\s+(\d+)\s+(\d{4})", ver)
    build = "未知"
    if m:
        try:
            build = datetime.datetime.strptime(
                f"{m.group(1)} {m.group(2)} {m.group(3)}", "%b %d %Y").date().isoformat()
        except ValueError:
            build = " ".join(m.groups())

    # 闸门状态改成"直接问"：签名不确定就逐个 arity 试，试不出来就老实报未知。
    gated = None
    try:
        from jax._src import cloud_tpu_init
        fn = getattr(cloud_tpu_init, "is_cloud_tpu_older_than", None)
        if fn is not None and jax.devices()[0].platform == "tpu":
            today = datetime.date.today()
            for args in ((today.year, today.month, today.day),
                         (today.year, today.month, today.day, jax.devices()[0].client)):
                try:
                    gated = bool(fn(*args))
                    break
                except TypeError:
                    continue
    except Exception:
        pass
    state = ("**闸门会挡住 Pallas**" if gated else
             "闸门通过" if gated is False else "闸门状态未知（不影响判断，看 S 扫描结果）")
    return (f"jax {jax.__version__} | libtpu 构建日 {build} | {state} | "
            f"platform_version: {one_line}")


@probe("L2.2 自带包装器用的是哪种 mask（读源码，零风险）")
def probe_builtin_mask_kind() -> str:
    """不去跑它，直接读它的源码 —— 结论一样确定，还不用趟 Mesh/mesh-string 的坑
    （第一跑就是栽在 Mesh.to_str 上，而这一项本来就不是决定性的）。

    要确认的是：它给 make_splash_mha 的**编译期** mask 只有 Causal/Full/Local，
    段信息只走运行时 SegmentIds —— 而块稀疏是由编译期 mask 驱动的，所以自带
    包装器拿不到块对角的跳块收益。"""
    import inspect
    from torch_xla.experimental import splash_attention as sa
    src = inspect.getsource(sa)
    kinds = [n for n in ("CausalMask", "FullMask", "LocalMask", "MultiHeadMask")
             if f"splash_attention_mask.{n}" in src or f"sm.{n}" in src]
    uses_segment_ids = "SegmentIds(" in src
    takes_custom_mask = "mask=" in src and "def splash_attention(" in src
    return (f"编译期 mask 种类={kinds}；用运行时 SegmentIds={uses_segment_ids}；"
            f"对外暴露自定义 mask 入口={'是' if takes_custom_mask and 'BlockDiagonal' in src else '否'}"
            f" -> 自带包装器**无法**表达块对角，必须走 L3 的 call_jax 自建 mask")


# ── L3 call_jax 注入我们自己的块对角 mask（决定性）────────────────────────────

def _bd_jax_fn(q, k, v, seg_lens):
    """在 JAX 侧构造**编译期**块对角 mask —— 块跳过就是靠它驱动的。
    q/k/v 布局 [B, H, S, D]（与 torch_xla 自带包装器一致），vmap 掉 batch 维。"""
    import jax
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_kernel as sk, splash_attention_mask as sm,
    )

    class BlockDiagonalMask(sm.Mask):
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
            return isinstance(o, BlockDiagonalMask) and self.lens == o.lens

        def __hash__(self):
            return hash((type(self), self.lens))

    n_heads = q.shape[1]
    lens = tuple(seg_lens)
    mask = (sm.FullMask(_shape=(sum(lens), sum(lens))) if len(lens) == 1
            else BlockDiagonalMask(lens))
    multi = sm.MultiHeadMask(masks=(mask,) * n_heads)
    # **反向块大小是显存与速度的关键旋钮**（arch_probe H1 在 L=16384/48head 实测）：
    #   默认(q_dkv=128,kv_dkv=128,fused=False) 288ms
    #   dkv=128/128 fused=True                 OOM  <- 第四跑 L4.1 撞的就是这个：
    #       反向 dk/dv kernel 按 q 块物化 [128,48,16384,128] = 24GB > 16GB
    #   dkv=512/512  fused=True                157ms
    #   dkv=1024/1024 fused=True               142ms  <- 最快，比默认快 2.03x
    #   dkv=2048/2048 fused=True               OOM
    # 前向块仍取 128（段长对齐粒度，见 tpu-navit-splash 结论）；反向块取 1024。
    BWD_BLOCK = 1024
    bs = sk.BlockSizes(
        block_q=BLOCK, block_kv=BLOCK, block_kv_compute=BLOCK,
        block_q_dkv=BWD_BLOCK, block_kv_dkv=BWD_BLOCK, block_kv_dkv_compute=BWD_BLOCK,
        use_fused_bwd_kernel=True,
    )
    kernel = sk.make_splash_mha(multi, head_shards=1, q_seq_shards=1, block_sizes=bs)
    return jax.vmap(kernel)(q, k, v)


@probe("L3.1 call_jax 桥接可用性 + 数值")
def probe_call_jax() -> str:
    """call_jax 是 torch_xla 的通用"用 torch 张量调 JAX 函数"入口
    （torch_xla.core.xla_builder.call_jax）。通了就能绕开自带包装器的 mask 限制。"""
    import torch
    from torch_xla.core.xla_builder import call_jax
    L, qh, dim = 2048, 4, 128
    lens = _seg_lens(L, 3, BLOCK)
    q, k, v = _qkv(L, qh, qh, dim)
    out = call_jax(_bd_jax_fn, (q, k, v, tuple(lens)), {}, "bd_splash_fw")
    # 朴素参考（torch 侧，fp32）
    ids = _seg_ids_t(lens).to(out.device)
    allow = (ids[:, None] == ids[None, :])
    qf, kf, vf = (t[0].to(torch.float32) for t in (q, k, v))
    logits = qf @ kf.transpose(-1, -2)          # splash 不内置 1/sqrt(d) 缩放
    logits = logits.masked_fill(~allow[None], float("-inf"))
    ref = torch.softmax(logits, dim=-1) @ vf
    rel = float((out[0].to(torch.float32) - ref).abs().max()
                / (ref.abs().max() + 1e-30))
    if not np.isfinite(rel) or rel > 5e-2:
        raise RuntimeError(f"数值不匹配 rel={rel:.3e}（段长 {lens}）")
    return f"call_jax 通，块对角数值正确 rel={rel:.3e}（L={L} 段长={lens}）"


@probe("L3.2 call_jax 块对角跳块提速（决定性判据）")
def probe_call_jax_speed() -> str:
    from torch_xla.core.xla_builder import call_jax
    lens = _seg_lens(TIME_L, TIME_SEGMENTS, BLOCK)
    q, k, v = _qkv(TIME_L, KREA2_Q_HEADS, KREA2_KV_HEADS, KREA2_HEAD_DIM)
    t_bd = _xla_bench(lambda: call_jax(_bd_jax_fn, (q, k, v, tuple(lens)), {}, "bd"))
    t_full = _xla_bench(lambda: call_jax(_bd_jax_fn, (q, k, v, (TIME_L,)), {}, "full"))
    # 先做量级自检 —— 数字不可信时**绝不**报"跳块/未跳块"的结论
    bad = (_implausible(t_bd, TIME_L, KREA2_Q_HEADS, KREA2_HEAD_DIM, TIME_SEGMENTS)
           or _implausible(t_full, TIME_L, KREA2_Q_HEADS, KREA2_HEAD_DIM, 1))
    if bad:
        raise RuntimeError(
            f"测量不可信，拒绝给结论：块对角 {t_bd:.2f}ms / 全通 {t_full:.2f}ms；{bad}")
    global _T_ATTN_BD
    _T_ATTN_BD = t_bd            # 给 L4.1 当"层时不可能快过自身注意力"的硬下限
    ratio = t_bd / t_full
    theory = sum(n * n for n in lens) / (TIME_L ** 2)
    verdict = "跳块生效" if ratio < (theory + 1.0) / 2 else "未跳块"
    return (f"L={TIME_L} {TIME_SEGMENTS}段 heads={KREA2_Q_HEADS}/{KREA2_KV_HEADS}x{KREA2_HEAD_DIM} bf16 | "
            f"块对角 {t_bd:.2f}ms vs 全通 {t_full:.2f}ms -> 实测比 {ratio:.3f}，"
            f"理论 {theory:.3f} -> {verdict}"
            f"（JAX 侧同形状实测 0.252 / 106.81ms，可直接对比）")


def _bd_jax_vjp(q, k, v, seg_lens, grad_out):
    """块对角 splash 的反向，JAX 侧用 vjp 求。与 torch_xla 自带包装器的
    _jax_grad_f 同构。"""
    import functools
    import jax
    f = functools.partial(_bd_jax_fn, seg_lens=seg_lens)
    _primals, f_vjp = jax.vjp(f, q, k, v)
    return f_vjp(grad_out)


def _make_bd_splash():
    """建一个 autograd.Function 包住 call_jax。

    **第二跑的教训**：call_jax **不透传 autograd**（报
    "element 0 of tensors does not require grad and does not have a grad_fn"）。
    torch_xla 自带的 SplashAttention 也是这么做的 —— 显式写 autograd.Function，
    前向和反向各调一次 call_jax。这不是 torch_xla 路线的死穴，只是必须自己写。
    """
    import torch
    from torch_xla.core.xla_builder import call_jax

    class BDSplash(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v, seg_lens):
            ctx.save_for_backward(q, k, v)
            ctx.seg_lens = seg_lens
            return call_jax(_bd_jax_fn, (q, k, v, seg_lens), {}, "bd_fw")

        @staticmethod
        def backward(ctx, grad_out):
            q, k, v = ctx.saved_tensors
            dq, dk, dv = call_jax(
                _bd_jax_vjp, (q, k, v, ctx.seg_lens, grad_out.contiguous()), {}, "bd_bw")
            return dq, dk, dv, None

    return BDSplash


@probe("L4.1 端到端训练步：attn+MLP+LoRA 前向反向+optimizer.step")
def probe_train_step() -> str:
    """**这才是"要不要动 29k 行"的裁决数据。** 前面测的都是孤立注意力；真训练里
    还有 MLP、LoRA 旁路、optimizer.step，以及 torch_xla 的图捕获/执行开销。

    用真实 Krea2 单层形状（features=6144, 48q/12kv x128, SwiGLU mlpdim=16384），
    冻结底模 + LoRA rank32 只训 LoRA —— 与本仓库的训练范式一致。"""
    import torch
    import torch_xla.core.xla_model as xm
    d, H, KV, hd, mlp, r = 6144, 48, 12, 128, 16384, 32
    L = TIME_L
    lens = _seg_lens(L, TIME_SEGMENTS, BLOCK)
    dev = xm.xla_device()
    BDSplash = _make_bd_splash()

    def frozen(*shape):
        return (torch.randn(*shape, dtype=torch.bfloat16, device=dev) * 0.02).requires_grad_(False)

    W = {k: frozen(*s) for k, s in {
        "wq": (d, H * hd), "wk": (d, KV * hd), "wv": (d, KV * hd),
        "gate": (d, d), "wo": (d, d),
        "w_gate": (d, mlp), "w_up": (d, mlp), "w_down": (mlp, d)}.items()}
    # LoRA 只挂在 wq/wo/w_gate/w_down 上（与仓库 targets 同量级即可）
    lora = {}
    for name, (i, o) in {"wq": (d, H * hd), "wo": (d, d),
                         "w_gate": (d, mlp), "w_down": (mlp, d)}.items():
        lora[name] = (
            (torch.randn(i, r, dtype=torch.bfloat16, device=dev) * 0.01).requires_grad_(True),
            torch.zeros(r, o, dtype=torch.bfloat16, device=dev).requires_grad_(True))
    params = [t for pair in lora.values() for t in pair]
    opt = torch.optim.AdamW(params, lr=1e-4)

    def lin(x, name):
        y = x @ W[name]
        if name in lora:
            a, b = lora[name]
            y = y + (x @ a) @ b
        return y

    x0 = torch.randn(L, d, dtype=torch.bfloat16, device=dev)

    def step():
        x = x0
        q = lin(x, "wq").reshape(1, L, H, hd).permute(0, 2, 1, 3)
        k = (x @ W["wk"]).reshape(1, L, KV, hd).permute(0, 2, 1, 3)
        v = (x @ W["wv"]).reshape(1, L, KV, hd).permute(0, 2, 1, 3)
        q = (q.to(torch.float32) * (hd ** -0.5)).to(torch.bfloat16)   # splash 不内置缩放
        a = BDSplash.apply(q, k, v, tuple(lens))
        a = a.permute(0, 2, 1, 3).reshape(L, H * hd)
        a = a * torch.sigmoid(x @ W["gate"])
        x = x + lin(a, "wo")
        h = torch.nn.functional.silu(lin(x, "w_gate")) * (x @ W["w_up"])
        x = x + lin(h, "w_down")
        loss = x.to(torch.float32).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        # **必须消费输出**（第三跑的坑）：torch_xla 惰性执行，只调 step() +
        # wait_device_ops() 根本没派发任何东西，wait 等了个空，测出 10.9ms ——
        # 而同形状注意力单独就要 108ms。.cpu() 强制物化+同步才是真时间。
        return float(loss.detach().cpu())

    t0 = time.time()
    first = step()
    compile_s = time.time() - t0
    if not np.isfinite(first):
        raise RuntimeError(f"loss 非有限：{first}")
    ts = []
    for _ in range(5):
        t = time.perf_counter()
        step()
        ts.append((time.perf_counter() - t) * 1e3)
    med = float(np.median(ts))

    # 双重量级自检。峰值 FLOPS 下限太松（4.19ms vs 真值 ~108ms），主判据用 L3.2
    # 的实测注意力时间：这一层**包含**那段注意力，不可能显著更快。
    bad = _implausible(med, L, H, hd, TIME_SEGMENTS)
    if bad is None and _T_ATTN_BD is not None and med < _T_ATTN_BD * 0.8:
        bad = (f"步时 {med:.2f}ms < L3.2 实测同形状注意力 {_T_ATTN_BD:.2f}ms 的 80% —— "
               f"整层不可能快过它自己的注意力，说明没真算/没同步")
    if bad:
        raise RuntimeError(f"步时 {med:.2f}ms 不可信，拒绝给结论：{bad}")
    return (f"真实 Krea2 单层形状 L={L} {TIME_SEGMENTS}段 LoRA r={r} | "
            f"首步（含编译）{compile_s:.1f}s，稳态步时 {med:.1f}ms/层"
            f"（其中注意力 {_T_ATTN_BD:.1f}ms，即 {_T_ATTN_BD / med:.0%}）-> "
            f"外推 28 层约 {med * 28 / 1000:.1f}s/step（不含 TE/VAE/优化器全量）")


@probe("L4.2 换段长布局的重编译代价（torch_xla 头号风险）")
def probe_relayout_cost() -> str:
    """JAX 侧已知：换布局要 ~250ms XLA 编译 + ~750ms MaskInfo（主机侧、无缓存）。
    torch_xla 上要问的是**更糟的那个可能**：换 splash 的 mask 会不会连累整张
    torch_xla 图重编译。若只有 splash 那一小块重编，代价与 JAX 侧同量级，
    布局有限集（8 档量化 = 535 种）就能一次性预热掉。"""
    import torch
    import torch_xla.core.xla_model as xm
    L, H, hd = TIME_L, 8, 128           # 头数缩小，只看编译代价不看绝对速度
    BDSplash = _make_bd_splash()
    q, k, v = _qkv(L, H, H, hd)
    q.requires_grad_(True)

    def run(lens):
        out = BDSplash.apply(q, k, v, tuple(lens))
        s = out.to(torch.float32).sum()
        xm.mark_step()
        return float(s.cpu())

    layouts = [_seg_lens(L, 4, BLOCK),
               [BLOCK * 2, BLOCK * 6, L - BLOCK * 8],
               [BLOCK * 4] * 3 + [L - BLOCK * 12],
               _seg_lens(L, 8, BLOCK)]
    costs = []
    for lens in layouts:
        t = time.time()
        run(lens)
        xm.wait_device_ops()
        costs.append(time.time() - t)
    # 回到第一个布局：若有缓存，这次应该明显更快
    t = time.time()
    run(layouts[0])
    xm.wait_device_ops()
    again = time.time() - t
    cached = again < costs[0] * 0.3
    return (f"4 种布局首次 {[f'{c:.1f}s' for c in costs]}；回到第 1 种 {again:.2f}s -> "
            f"{'有缓存，布局有限集可一次性预热' if cached else '**无缓存，每次换布局都要重付**'}"
            f"（JAX 侧同项 ~0.25s 编译 + ~0.75s MaskInfo）")


@probe("L3.3 反向能否穿过 call_jax（训练必需）")
def probe_call_jax_backward() -> str:
    """可微性是 torch_xla 路线成立的硬条件——不可微就只能推理用。
    注意：裸 call_jax 不可微（第二跑实测），必须自建 autograd.Function。"""
    import torch
    BDSplash = _make_bd_splash()
    L, qh, dim = 2048, 4, 128
    lens = _seg_lens(L, 3, BLOCK)
    q, k, v = _qkv(L, qh, qh, dim)
    q.requires_grad_(True)
    k.requires_grad_(True)
    out = BDSplash.apply(q, k, v, tuple(lens))
    out.to(torch.float32).pow(2).sum().backward()
    if q.grad is None or k.grad is None:
        raise RuntimeError("梯度为 None —— autograd.Function 包装后仍不可微")
    gq = float(q.grad.to(torch.float32).abs().mean().cpu())
    gk = float(k.grad.to(torch.float32).abs().mean().cpu())
    if not (np.isfinite(gq) and np.isfinite(gk)) or gq == 0.0 or gk == 0.0:
        raise RuntimeError(f"梯度异常：mean|dq|={gq}, mean|dk|={gk}")
    return f"autograd.Function 包装后梯度可穿过，mean|dq|={gq:.4e} mean|dk|={gk:.4e}"


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 78)
    print("Kaggle TPU v5e-8 · torch_xla 路线可行性探针")
    print("=" * 78)

    _bootstrap_install()
    probe_env()

    # ── jax 版本扫描：必须在**父进程 import torch_xla 之前**做完 ────────────────
    # TPU 独占：父进程一旦 import torch_xla 就占住设备，子进程再也初始化不了。
    won = sweep_jax_versions()
    if won:
        record("S 扫描结论", "OK", f"jax {won} 与 torch_xla 所带 libtpu 配套，用它继续")
    else:
        record("S 扫描结论", "FAIL",
               f"候选 {JAX_CANDIDATES} 全部与 torch_xla 2.9.0 的 libtpu 不配套 "
               f"-> Mosaic IR 版本对不上，torch_xla + Pallas 在此组合下无解")

    if not probe_import():
        record("裁决", "FAIL", "torch_xla 装不上/导不进 -> torch_xla 路线在 Kaggle 上不成立")
        _flush()
        return 0
    probe_devices()
    probe_basic()

    ok_coexist = probe_jax_coexist()
    if ok_coexist:
        probe_version_scissors()

    probe_builtin_splash()
    probe_builtin_mask_kind()

    # **短路**（第二跑的教训之三）：扫描全灭时父进程还带着最后一个候选（0.5.3）
    # 继续跑 L3，产出一堆同因 FAIL 噪声、还白烧配额。全灭就直接 SKIP。
    if not won:
        record("L3.* / L4.*", "SKIP",
               "jax 版本扫描全军覆没，call_jax + Pallas 在此组合下无解，后续探测无意义")
    elif not ok_coexist:
        record("L3.* / L4.*", "SKIP", "jax 与 torch_xla 无法共存，call_jax 路线不可测")
    else:
        probe_call_jax()
        probe_call_jax_speed()
        probe_call_jax_backward()
        # L4 是"要不要动 29k 行"的裁决数据，但也最重最容易挂，排在最后
        probe_train_step()
        probe_relayout_cost()

    _ran = {r["name"] for r in RESULTS}
    _missed = [n for n in _DEFINED_PROBES if n not in _ran]
    if _missed:
        print("[!] 已定义但未调用的探测：" + ", ".join(_missed))

    ok = sum(1 for r in RESULTS if r["status"] == "OK")
    fail = sum(1 for r in RESULTS if r["status"] == "FAIL")
    skip = sum(1 for r in RESULTS if r["status"] == "SKIP")
    print("=" * 78)
    print(f"汇总：OK={ok}  FAIL={fail}  SKIP={skip}  总耗时={time.time() - _T0:.0f}s")

    def _detail(prefix):
        for r in RESULTS:
            if r["name"].startswith(prefix):
                return r["detail"]
        return ""

    def _st(prefix):
        for r in RESULTS:
            if r["name"].startswith(prefix):
                return r["status"]
        return "MISSING"

    print("\n" + "=" * 78)
    print("torch_xla 路线裁决：")
    scissors = any(r["status"] == "FAIL"
                   and ("libtpu version" in r["detail"]
                        or "Unsupported version" in r["detail"]
                        or "Mosaic" in r["detail"])
                   for r in RESULTS)
    if _st("S 扫描结论") == "FAIL":
        print("  [NO]   **版本剪刀无解**：候选 jax 版本没有一个能和 torch_xla 2.9.0")
        print("         所带的 libtpu 把 splash kernel 编出来（Mosaic IR 版本对不上）。")
        print(f"         {_detail('S 扫描结论')}")
        print("         -> 想走 torch_xla 就得自己编 torch_xla（配新 libtpu），")
        print("            那已经超出'省 29k 行重写'的性价比了。倾向选纯 JAX 路线。")
    elif scissors:
        print("  [ENV]  **版本剪刀，非技术结论**：torch_xla 的 PJRT 绑老 libtpu，")
        print("         jax 的 Pallas 闸门要新 libtpu，两者对不上，splash 全挂。")
        print(f"         {_detail('L1.5 ')}")
        print("         下一步：把 jax 降到与 torch_xla 所带 libtpu 配套的版本")
        print("         （v2.9.0 钉 jax 0.7.1 + libtpu 0.0.21/20250813），")
        print("         且装 jax 时不要带 [tpu] extra，否则会覆盖 torch_xla 的 libtpu。")
        print("         **注意这本身就是个结论**：torch_xla 路线要长期承受这把剪刀。")
    elif _st("L3.2 ") == "OK" and "跳块生效" in _detail("L3.2 ") and _st("L3.3 ") == "OK":
        print("  [YES]  call_jax + 自建块对角 mask 跳块生效，且反向可穿过")
        print("         -> torch_xla 路线成立：29k 行 torch 代码不必重写成 JAX，")
        print("            只需把 navit 注意力换成一个 call_jax 桥接后端")
        print("            （对齐 models/anima_modeling_core.py 的 npu_tnd 先例）。")
        print(f"         {_detail('L3.2 ')}")
        print(f"         L4.1 端到端训练步: {_st('L4.1 ')} / {_detail('L4.1 ')}")
        print(f"         L4.2 换布局代价:   {_st('L4.2 ')} / {_detail('L4.2 ')}")
    elif _st("L3.3 ") == "FAIL":
        print("  [NO]   反向穿不过 call_jax -> 只能推理用，训练不行。")
        print("         -> torch_xla 路线不成立，回到 JAX 重写的评估。")
    elif _st("L3.2 ") in ("FAIL", "MISSING"):
        print("  [NO]   call_jax 桥接不通或没跑到 -> torch_xla 拿不到块对角收益。")
        print(f"         L1.4 jax 共存: {_st('L1.4 ')} / {_detail('L1.4 ')}")
    else:
        print("  [PART] 部分通过，看上面逐项结果。")
    print(f"  参考：自带包装器的 segment_ids 路径 -> {_detail('L2.2 ')}")
    print("=" * 78)
    _flush()
    print(f"报告已写到 {OUT_DIR}/xla_probe_report.txt")
    return 0        # 恒 0：单项 FAIL 是数据，非零会被 Kaggle 判为 ERROR


if __name__ == "__main__":
    raise SystemExit(main())
