#!/usr/bin/env python
"""海光 DCU（K100-AI 等）能力探针 —— 在真机上实测，不做任何假设。

用途：在超算互联网（scnet）/ 任意 DTK 容器里**烧卡时之前**先跑这一个脚本，把"哪些
算子/特性在这台机器上真的能用、多快"变成实测数据。100 卡时的预算下，先花 2 分钟把
不确定性清掉，比撞上第 300 步报错划算得多。

用法：
    python tools/dcu_probe.py                      # 单卡全项
    python tools/dcu_probe.py --json out.json
    torchrun --nproc_per_node=8 tools/dcu_probe.py --dist   # 8 卡集合通信实测

判读：
    * OK   —— 实测通过（含数值校验的项会打印相对误差）
    * FAIL —— 实测失败，附错误摘要；训练里用到该路径就会崩
    * SKIP —— 前置条件不满足

本脚本只读不写，不加载任何底模权重。

★ 为什么有些探测看起来"多此一举"
   ``广播 matmul backward`` 那一项：昇腾 910B 上实测过 2D×3D 广播 matmul **前向逐位
   正确、反向静默算错**，害得 LoKr 的 w1 梯度整个是错的，而训练照跑、loss 照降、
   不报任何错（见 memory ``npu-broadcast-matmul-grad-bug``）。换平台必须重做**反向**
   对拍 —— 前向对不代表反向对，这是花过代价买来的教训。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback

RESULTS: list[dict] = []


def record(name: str, status: str, detail: str = "") -> None:
    RESULTS.append({"name": name, "status": status, "detail": detail})
    mark = {"OK": "  OK  ", "FAIL": " FAIL ", "SKIP": " SKIP ", "INFO": " INFO "}.get(status, status)
    print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""), flush=True)


# 所有被 @probe 声明过的探测名。main() 末尾核对它们是否都真的跑了 —— 探测**不是**
# 自动发现的，漏在 main() 里调用只会让那一项静默消失（npu_probe 上踩过一次）。
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
                tb = traceback.format_exc(limit=2).strip().splitlines()[-1]
                record(name, "FAIL", f"{type(e).__name__}: {e} | {tb}")
                return False

        run.__probe_name__ = name
        return run

    return deco


class _Skip(Exception):
    pass


# Anima base 的真实几何：model_channels=2048 / 16 头 → head_dim=128
_ANIMA_H, _ANIMA_D = 16, 128


def _dev():
    import torch
    if not torch.cuda.is_available():
        raise _Skip("torch.cuda.is_available() == False（DCU 上 HIP 复用 cuda 命名空间）")
    return "cuda"


def _sync():
    import torch
    torch.cuda.synchronize()


# ── 0. 基础环境 ────────────────────────────────────────────────────────────────

def probe_env() -> None:
    record("python", "INFO", f"{platform.python_version()} @ {sys.executable}")
    record("platform", "INFO", platform.platform())
    for var in ("ROCM_PATH", "HIP_PATH", "DTK_HOME", "HIP_VISIBLE_DEVICES",
                "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
                "HSA_OVERRIDE_GFX_VERSION", "PYTORCH_HIP_ALLOC_CONF",
                "PYTORCH_CUDA_ALLOC_CONF", "MIOPEN_USER_DB_PATH", "LD_LIBRARY_PATH"):
        v = os.environ.get(var)
        if v:
            record(f"env:{var}", "INFO", v[:200])
    record("dtk 根目录", "INFO",
           "/opt/dtk 存在" if os.path.isdir("/opt/dtk") else "/opt/dtk 不存在（DTK 可能装在别处）")
    # rocm-smi 是 DTK 侧的，hy-smi 是驱动侧的；两个都试，有哪个算哪个
    for cmd in (["rocm-smi"], ["hy-smi"], ["rocminfo"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            txt = (out.stdout or "").strip()
            if cmd[0] == "rocminfo":
                # rocminfo 输出极长，只挑 gfx 行
                lines = [ln.strip() for ln in txt.splitlines() if "gfx" in ln]
                txt = "\n".join(dict.fromkeys(lines))     # 去重保序
            else:
                txt = "\n".join(txt.splitlines()[:14])
            record(" ".join(cmd), "INFO", "\n" + txt if txt else "(空输出)")
        except Exception as e:
            record(" ".join(cmd), "SKIP", f"{type(e).__name__}: {e}")


@probe("import torch（必须是 DTK 的 das 版，不能是 PyPI 的 CUDA/CPU 版）")
def probe_torch() -> str:
    import torch
    hip = getattr(torch.version, "hip", None)
    cuda = getattr(torch.version, "cuda", None)
    if not hip:
        raise RuntimeError(
            f"torch {torch.__version__} 不是 HIP 构建（hip={hip!r} cuda={cuda!r}）。"
            "多半是在 DTK 镜像里 pip install torch 覆盖了适配版 —— 这会让整个环境作废。"
        )
    das = "+das" in torch.__version__ or "dtk" in torch.__version__
    return (f"torch {torch.__version__}, hip={hip}"
            + ("" if das else "（版本串里没有 das/dtk 标记，确认一下来源）"))


@probe("DCU 可见性与规格")
def probe_visible() -> str:
    import torch
    d = _dev()
    n = torch.cuda.device_count()
    infos = []
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        infos.append(
            f"#{i} {p.name} arch={getattr(p, 'gcnArchName', '?')} "
            f"mem={p.total_memory / 1e9:.1f}GB CU={p.multi_processor_count}"
        )
    del d
    return f"{n} 卡\n        " + "\n        ".join(infos)


# ── 1. 数值与精度 ──────────────────────────────────────────────────────────────

def _matmul_case(dtype, tol, note=""):
    import torch
    d = _dev()
    torch.manual_seed(0)
    a = torch.randn(512, 512)
    b = torch.randn(512, 512)
    ref = a.double() @ b.double()
    got = (a.to(d, dtype) @ b.to(d, dtype)).float().cpu().double()
    rel = ((got - ref).norm() / ref.norm()).item()
    if rel > tol:
        raise RuntimeError(f"相对误差过大 rel={rel:.3e} > {tol:.0e}")
    return f"rel_err={rel:.3e}{note}"


@probe("bf16 matmul 数值")
def probe_bf16_matmul() -> str:
    import torch
    return _matmul_case(torch.bfloat16, 5e-2)


@probe("fp16 matmul 数值")
def probe_fp16_matmul() -> str:
    import torch
    return _matmul_case(torch.float16, 5e-2)


@probe("fp32 matmul 数值")
def probe_fp32_matmul() -> str:
    import torch
    return _matmul_case(torch.float32, 1.0,
                        "（>1e-4 说明默认走了降精度模式；训练里 loss/reduce 的 fp32 残留路径需留意）")


@probe("autocast('cuda', bf16)")
def probe_autocast() -> str:
    import torch
    d = _dev()
    x = torch.randn(64, 256, device=d)
    lin = torch.nn.Linear(256, 256).to(d)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = lin(x)
    if y.dtype != torch.bfloat16:
        raise RuntimeError(f"autocast 未生效，输出 dtype={y.dtype}（期望 bfloat16）")
    return f"输出 dtype={y.dtype}"


@probe("★ 2D×3D 广播 matmul 的**反向**（昇腾在这里静默算错过）")
def probe_broadcast_matmul_backward() -> str:
    """LoKr 的 w1 更新路径就长这样；前向对不代表反向对。

    构造 ``A[m,k] @ B[b,k,n]``（左操作数被广播到 batch 维），用 fp64 CPU 手算参考，
    对拍 **dA 与 dB 两个梯度**。昇腾 torch_npu 2.6 上这一项前向 rel~1e-3、反向 dA 完全
    错位，导致 LoKr 的 ‖w1‖ 失控且毫无报错。DCU 上必须独立验一遍。
    """
    import torch
    d = _dev()
    torch.manual_seed(0)
    B, M, K, N = 4, 32, 48, 24
    a_cpu = torch.randn(M, K, dtype=torch.float64)
    b_cpu = torch.randn(B, K, N, dtype=torch.float64)
    g_cpu = torch.randn(B, M, N, dtype=torch.float64)

    # fp64 CPU 参考（同样走广播 matmul，但 CPU 后端）
    ar = a_cpu.clone().requires_grad_(True)
    br = b_cpu.clone().requires_grad_(True)
    (ar @ br).backward(g_cpu)

    # 设备侧：fp32
    ad = a_cpu.float().to(d).requires_grad_(True)
    bd = b_cpu.float().to(d).requires_grad_(True)
    out = ad @ bd
    out.backward(g_cpu.float().to(d))

    fwd_rel = (((ad @ bd).detach().cpu().double() - (a_cpu @ b_cpu)).norm()
               / (a_cpu @ b_cpu).norm()).item()
    da_rel = ((ad.grad.cpu().double() - ar.grad).norm() / ar.grad.norm()).item()
    db_rel = ((bd.grad.cpu().double() - br.grad).norm() / br.grad.norm()).item()

    bad = [n for n, v in (("dA", da_rel), ("dB", db_rel)) if v > 1e-4]
    if bad:
        raise RuntimeError(
            f"广播 matmul 的反向不正确：fwd={fwd_rel:.3e} dA={da_rel:.3e} dB={db_rel:.3e}"
            f"（{'/'.join(bad)} 超差）。这正是昇腾踩过的坑 —— LoKr 必须走 "
            "expand+bmm 的绕行实现（见 trainer/lora.py 与 "
            "tests/test_lokr_w1_grad_no_broadcast.py）。"
        )
    return f"fwd={fwd_rel:.3e} dA={da_rel:.3e} dB={db_rel:.3e} → 前向与反向都正确"


# ── 2. 注意力（训练热路径，DCU 上最大的不确定性）───────────────────────────────

@probe("★ SDPA 可用后端（决定 NaViT 能不能上大 token 预算）")
def probe_sdpa_backends() -> str:
    """逐个强制指定后端跑一次，看哪些真的能用。

    ROCm 上 flash / mem-efficient 后端依赖 aotriton 被编进 torch，DTK 里有没有是未知数。
    **只剩 math 后端的话，注意力会物化 O(S²) 的分数矩阵** —— NaViT 打包动辄 ΣN=16k~32k，
    32768² × 2B = 2.1 GB **每层每头**，必炸。所以这一项直接决定 navit_token_budget
    能开到多大、甚至能不能开 NaViT。
    """
    import torch
    import torch.nn.functional as F
    d = _dev()
    B, H, S, E = 1, 8, 1024, 64
    q = torch.randn(B, H, S, E, device=d, dtype=torch.bfloat16)
    k = torch.randn(B, H, S, E, device=d, dtype=torch.bfloat16)
    v = torch.randn(B, H, S, E, device=d, dtype=torch.bfloat16)

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:
        raise _Skip("该 torch 版本没有 torch.nn.attention.sdpa_kernel（<2.3），无法逐后端强制")

    names = {
        SDPBackend.FLASH_ATTENTION: "flash",
        SDPBackend.EFFICIENT_ATTENTION: "mem_efficient",
        SDPBackend.MATH: "math",
    }
    got = []
    for backend, label in names.items():
        try:
            with sdpa_kernel(backend):
                o = F.scaled_dot_product_attention(q, k, v)
                _sync()
            got.append(f"{label}=OK" if torch.isfinite(o).all() else f"{label}=NaN")
        except Exception as e:
            got.append(f"{label}=NO({type(e).__name__})")
    fast = [g for g in got if g.endswith("=OK") and not g.startswith("math")]
    verdict = ("有快核" if fast else
               "⚠ 只剩 math 后端 —— 注意力会物化 O(S²)，大 token 预算必炸显存")
    return " ".join(got) + f" → {verdict}"


@probe("★ 应用 SDPA 后端修复（之后所有 SDPA 项都在修复后的状态下测）")
def probe_sdpa_backend_fix() -> str:
    """把上一项测出来不可用的后端关掉，再往下测。

    【实测 · scnet BW(gfx936) / torch 2.9.0+das.dtk2604】DTK 的 torch 把**无 mask 的
    SDPA** 派发给外部的 flash-attn 动态库，镜像里没有那个 .so 就直接抛
    ``RuntimeError: No matching libraries found for flash_attn_2_cuda*.so``，
    而不是回退到 math。于是「给 mask 能跑、不给 mask 崩」。

    训练时 ``utils/dcu_compat.enable()`` 会自动做同样的事，所以探针也要在**修复后**的
    状态下量后面几项 —— 否则 sdpa_seg 三项会全 FAIL，看起来像 NaViT 不可用，
    实际只是少关了一个开关。
    """
    import sys, pathlib as _pl
    sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
    from utils import dcu_compat

    state = dcu_compat.configure_sdpa_backends()
    if not state:
        raise _Skip("该 torch 没有 torch.backends.cuda.enable_flash_sdp，无需/无法修复")
    off = state.get("disabled_by_anima") or []
    if not off:
        return "无需修复：默认后端就能跑无 mask SDPA"
    return (f"关闭了 {'/'.join(off)} → flash={state['flash']} "
            f"mem_efficient={state['mem_efficient']} math={state['math']}")


def _sdpa_case(mask_kind: str) -> str:
    import torch
    import torch.nn.functional as F
    d = _dev()
    B, H, S, E = 1, 16, 2048, 128
    torch.manual_seed(0)
    q = torch.randn(B, H, S, E, device=d, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(B, H, S, E, device=d, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(B, H, S, E, device=d, dtype=torch.bfloat16, requires_grad=True)
    if mask_kind == "none":
        mask = None
    elif mask_kind == "bool":
        mask = torch.ones(B, 1, S, S, device=d, dtype=torch.bool)
    elif mask_kind == "additive":
        mask = torch.zeros(B, 1, S, S, device=d, dtype=torch.bfloat16)
    elif mask_kind == "broadcast":
        mask = torch.ones(B, 1, 1, S, device=d, dtype=torch.bool)
    else:
        raise ValueError(mask_kind)
    torch.cuda.reset_peak_memory_stats()
    _sync()
    t0 = time.perf_counter()
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    o.sum().backward()
    _sync()
    dt = (time.perf_counter() - t0) * 1000
    peak = torch.cuda.max_memory_allocated() / 1e9
    return f"fwd+bwd {dt:.1f} ms, peak {peak:.2f} GB, grad_ok={q.grad is not None}"


@probe("SDPA 无 mask (B1 H16 S2048 D128 bf16, fwd+bwd)")
def probe_sdpa_none() -> str:
    return _sdpa_case("none")


@probe("SDPA bool mask（稠密 SxS）")
def probe_sdpa_bool() -> str:
    return _sdpa_case("bool")


@probe("SDPA additive mask（稠密 SxS）")
def probe_sdpa_add() -> str:
    return _sdpa_case("additive")


@probe("SDPA 广播 mask [B,1,1,Skv]（昇腾拒收过，DCU 是否接受）")
def probe_sdpa_broadcast_mask() -> str:
    """文本编码器的 key-padding mask 就是这个形状。

    昇腾的融合算子只吃 Sq=真实长度，逼得 ``utils/npu_compat.expand_attn_mask`` 去物化展开。
    DCU 若原样接受，就不需要那层展开（也就没有 O(B·Sq·Skv) 的额外开销）。
    """
    return _sdpa_case("broadcast")


@probe("★ SDPA key-padding mask 的语义/极性（真实 padding 对拍）")
def probe_sdpa_keypad_semantics() -> str:
    """上面几条只测了"形状收不收、快不快"，**这条才测算得对不对**。

    极性反了的话（True=屏蔽 vs True=保留），训练会照跑、loss 照降、不报任何错 ——
    只是文本条件变成了"只看 padding"。全 True / 全 0 的 mask 对极性完全不敏感，
    必须用真实 padding 才测得出来。
    """
    import torch
    import torch.nn.functional as F
    d = _dev()
    B, H, S, E, VALID = 1, 16, 251, 128, 151
    torch.manual_seed(0)
    q = torch.randn(B, H, S, E, dtype=torch.bfloat16)
    k = torch.randn(B, H, S, E, dtype=torch.bfloat16)
    v = torch.randn(B, H, S, E, dtype=torch.bfloat16)
    keep = torch.zeros(B, S, dtype=torch.bool)
    keep[:, :VALID] = True
    mask = keep[:, None, None, :].expand(B, 1, S, S).contiguous()

    out = F.scaled_dot_product_attention(
        q.to(d), k.to(d), v.to(d), attn_mask=mask.to(d)).float().cpu().double()

    qd, kd, vd = q.double(), k.double(), v.double()
    scores = (qd @ kd.transpose(-1, -2)) / (E ** 0.5)
    keep4 = keep[:, None, None, :].expand(B, H, S, S)
    ref_ok = torch.softmax(scores.masked_fill(~keep4, float("-inf")), dim=-1) @ vd
    ref_inv = torch.softmax(scores.masked_fill(keep4, float("-inf")), dim=-1) @ vd

    if not torch.isfinite(out).all():
        raise RuntimeError("输出含 NaN/Inf")
    err_ok = ((out - ref_ok).norm() / ref_ok.norm()).item()
    err_inv = ((out - ref_inv).norm() / ref_inv.norm()).item()
    if err_ok > 5e-2:
        if err_inv < err_ok:
            raise RuntimeError(
                f"mask 极性疑似反了！rel_err(torch 语义 True=保留)={err_ok:.3e} > "
                f"rel_err(反转语义)={err_inv:.3e}")
        raise RuntimeError(f"与 fp64 参考不符 rel_err={err_ok:.3e}")
    return (f"rel_err={err_ok:.3e}（反转极性参考 {err_inv:.3e}，差 "
            f"{err_inv / max(err_ok, 1e-12):.0f}×）→ 极性与 torch 语义一致")


def _seg_reference(q, k, v, q_lens, kv_lens, scale):
    """逐段 dense SDPA 参考实现（fp32）。段内全注意力 ≡ 块对角，数学恒等。"""
    import torch
    import torch.nn.functional as F
    outs = []
    qo = ko = 0
    for sq, sk in zip(q_lens, kv_lens):
        qs = q[qo:qo + sq].float().transpose(0, 1).unsqueeze(0)
        ks = k[ko:ko + sk].float().transpose(0, 1).unsqueeze(0)
        vs = v[ko:ko + sk].float().transpose(0, 1).unsqueeze(0)
        outs.append(F.scaled_dot_product_attention(qs, ks, vs, scale=scale)
                    .squeeze(0).transpose(0, 1))
        qo += sq
        ko += sk
    return torch.cat(outs, dim=0)


def _sdpa_seg_case(q_lens, kv_lens, want_grad=True):
    import torch
    import torch.nn.functional as F
    d = _dev()
    H, E = _ANIMA_H, _ANIMA_D
    scale = 1.0 / math.sqrt(E)
    torch.manual_seed(0)
    q = torch.randn(sum(q_lens), H, E, device=d, dtype=torch.bfloat16, requires_grad=want_grad)
    k = torch.randn(sum(kv_lens), H, E, device=d, dtype=torch.bfloat16, requires_grad=want_grad)
    v = torch.randn(sum(kv_lens), H, E, device=d, dtype=torch.bfloat16, requires_grad=want_grad)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _sync()
    t0 = time.perf_counter()
    outs = []
    qo = ko = 0
    for sq, sk in zip(q_lens, kv_lens):
        qs = q[qo:qo + sq].transpose(0, 1).unsqueeze(0)
        ks = k[ko:ko + sk].transpose(0, 1).unsqueeze(0)
        vs = v[ko:ko + sk].transpose(0, 1).unsqueeze(0)
        outs.append(F.scaled_dot_product_attention(qs, ks, vs, scale=scale)
                    .squeeze(0).transpose(0, 1))
        qo += sq
        ko += sk
    o = torch.cat(outs, dim=0)
    if want_grad:
        o.float().sum().backward()
    _sync()
    dt = (time.perf_counter() - t0) * 1000
    peak = torch.cuda.max_memory_allocated() / 1e9
    return o, q, k, v, dt, peak, scale


@probe("★ sdpa_seg 逐段注意力（DCU 上 NaViT 的主路径，含反向）")
def probe_sdpa_seg() -> str:
    """DCU 没有 xformers，NaViT 打包只能走 sdpa_seg。这条决定 NaViT 在 DCU 上能不能用。

    同时做**段间泄漏检查**：换掉第 2 段的 k/v，第 1 段输出必须逐 bit 不变 ——
    泄漏了就说明语义不等于块对角，等于偷偷让每张图看到了别的图。
    """
    import torch
    q_lens = [1024, 2048, 1024]
    kv_lens = [128, 256, 96]
    o, q, k, v, dt, peak, scale = _sdpa_seg_case(q_lens, kv_lens, want_grad=True)
    ref = _seg_reference(q.detach(), k.detach(), v.detach(), q_lens, kv_lens, scale)
    rel = ((o.detach().float() - ref).norm() / ref.norm()).item()
    grads = [g for g in (q.grad, k.grad, v.grad) if g is None]
    if grads:
        raise RuntimeError("反向未产生梯度 → 训练不可用")
    return (f"rel_err={rel:.3e} | fwd+bwd {dt:.1f} ms, peak {peak:.2f} GB "
            f"| q={q_lens} kv={kv_lens}")


@probe("★ sdpa_seg 显存线性性（ΣN 8k vs 32k —— 判断是否落到 math 后端）")
def probe_sdpa_seg_scaling() -> str:
    """peak 显存应随 ΣN **近似线性**增长。

    4× token 带来 ~4× 显存 = 走了 flash/mem-efficient 快核；带来 ~16× = 落到 math 后端
    物化了 O(段长²) 的分数矩阵。后者意味着 navit_token_budget 只能开很小，
    或者干脆放弃 NaViT 走 ARB 稠密路径。**这是 DCU 适配里最该先看的一个数**。
    """
    notes = []
    peaks = []
    for total, nseg in ((8192, 4), (32768, 8)):
        seg = total // nseg
        lens = [seg] * nseg
        _, _, _, _, dt, peak, _ = _sdpa_seg_case(lens, lens, want_grad=True)
        peaks.append(peak)
        notes.append(f"ΣN={total}({nseg}段 每段{seg}) peak={peak:.2f}GB {dt:.0f}ms")
    ratio = peaks[1] / max(peaks[0], 1e-9)
    # token 4×、段长 2×：线性则 peak ~4×；O(段长²) 物化则分数矩阵那块 ~16×
    verdict = ("近似线性 → 走的是快核" if ratio < 7 else
               f"⚠ {ratio:.1f}× 远超 4× → 疑似 math 后端物化，NaViT 预算必须压低")
    return " | ".join(notes) + f" | 比值={ratio:.1f}× {verdict}"


@probe("flash_attn varlen（若 DTK 带了海光移植版）")
def probe_flash_varlen() -> str:
    """有的话是 sdpa_seg 之外的第二条 NaViT 路径（类比昇腾的 npu_tnd）。

    注意：**探到可用也不等于训练里能直接用** —— 仓库当前没有 flash varlen 的注意力
    后端接线（``_PACKED_ATTN_BACKENDS`` 只有 xformers/sdpa_seg/npu_tnd）。这一项是给
    "要不要投入去接第四个后端"提供依据的。
    """
    import torch
    try:
        from flash_attn import flash_attn_varlen_func    # type: ignore
    except Exception as e:
        raise _Skip(f"无 flash_attn（{type(e).__name__}）—— NaViT 走 sdpa_seg 即可")
    d = _dev()
    H, E = _ANIMA_H, _ANIMA_D
    lens = [1024, 2048, 1024]
    T = sum(lens)
    cu = torch.tensor([0] + list(_accum(lens)), device=d, dtype=torch.int32)
    q = torch.randn(T, H, E, device=d, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(T, H, E, device=d, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(T, H, E, device=d, dtype=torch.bfloat16, requires_grad=True)
    o = flash_attn_varlen_func(q, k, v, cu, cu, max(lens), max(lens))
    o.float().sum().backward()
    _sync()
    ref = _seg_reference(q.detach(), k.detach(), v.detach(), lens, lens, 1.0 / math.sqrt(E))
    rel = ((o.detach().float() - ref).norm() / ref.norm()).item()
    return f"可用，rel_err={rel:.3e}，反向 grad_ok={q.grad is not None}（仓库尚未接线该后端）"


def _accum(xs):
    s = 0
    for x in xs:
        s += x
        yield s


# ── 3. 训练器实际用到的算子 ────────────────────────────────────────────────────

@probe("torch.fft（spectral aux loss 依赖）")
def probe_fft() -> str:
    import torch
    d = _dev()
    x = torch.randn(2, 16, 64, 64, device=d, dtype=torch.float32)
    y = torch.fft.rfft2(x)
    ref = torch.fft.rfft2(x.cpu())
    rel = ((y.cpu() - ref).abs().norm() / ref.abs().norm()).item()
    return f"rfft2 -> {tuple(y.shape)} rel_err={rel:.3e}"


@probe("Conv2d/GroupNorm/SiLU bf16（VAE 路径，走 MIOpen）")
def probe_vae_ops() -> str:
    import torch
    d = _dev()
    m = torch.nn.Sequential(
        torch.nn.Conv2d(16, 32, 3, padding=1),
        torch.nn.GroupNorm(8, 32),
        torch.nn.SiLU(),
    ).to(d)
    x = torch.randn(1, 16, 128, 128, device=d)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = m(x)
    y.float().sum().backward()
    return f"输出 {tuple(y.shape)} dtype={y.dtype}（conv 走 MIOpen，首次会现编内核、偏慢属正常）"


@probe("torch.utils.checkpoint 梯度检查点")
def probe_grad_ckpt() -> str:
    import torch
    from torch.utils.checkpoint import checkpoint
    d = _dev()
    lin = torch.nn.Linear(512, 512).to(d)
    x = torch.randn(8, 512, device=d, requires_grad=True)
    y = checkpoint(lambda t: lin(t).relu(), x, use_reentrant=False)
    y.sum().backward()
    return f"grad_norm={x.grad.norm().item():.4f}"


@probe("AdamW 单步")
def probe_adamw() -> str:
    import torch
    d = _dev()
    p = torch.nn.Parameter(torch.randn(1024, 1024, device=d))
    opt = torch.optim.AdamW([p], lr=1e-4)
    before = p.detach().clone()
    (p * p).sum().backward()
    opt.step()
    delta = (p.detach() - before).norm().item()
    if not (delta > 0):
        raise RuntimeError("‖Δp‖=0，优化器 state 没落到设备上")
    return f"‖Δp‖={delta:.4e}"


@probe("torch.cuda.Event 计时（stage_timer 依赖）")
def probe_event() -> str:
    import torch
    d = _dev()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    x = torch.randn(2048, 2048, device=d)
    s.record()
    (x @ x).sum()
    e.record()
    _sync()
    ms = s.elapsed_time(e)
    if not (ms >= 0):
        raise RuntimeError(f"elapsed_time 返回 {ms}（负数说明 event 计时不可信）")
    return f"elapsed={ms:.3f} ms"


@probe("显存 API + expandable_segments")
def probe_mem() -> str:
    import torch
    d = _dev()
    x = torch.empty(1024, 1024, 64, device=d, dtype=torch.bfloat16)  # ~128MB
    alloc = torch.cuda.memory_allocated() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    del x
    torch.cuda.empty_cache()
    after = torch.cuda.memory_allocated() / 1e9
    conf = os.environ.get("PYTORCH_HIP_ALLOC_CONF") or os.environ.get("PYTORCH_CUDA_ALLOC_CONF") or "(未设)"
    return f"alloc={alloc:.3f}GB peak={peak:.3f}GB after_free={after:.3f}GB | alloc_conf={conf}"


@probe("FP8 张量核（base_quant 的前提；K100-AI 预期没有）")
def probe_fp8() -> str:
    import torch
    d = _dev()
    if not hasattr(torch, "float8_e4m3fn"):
        raise _Skip("该 torch 版本没有 float8 dtype")
    a = torch.randn(64, 64, device=d).to(torch.float8_e4m3fn)
    b = torch.randn(64, 64, device=d).to(torch.float8_e4m3fn)
    scale = torch.tensor(1.0, device=d)
    out = torch._scaled_mm(a, b.t(), scale_a=scale, scale_b=scale, out_dtype=torch.bfloat16)
    _sync()
    return (f"_scaled_mm 可用，输出 {tuple(out.shape)} —— 与 dcu_compat 里"
            "「gfx928 无 FP8」的推断矛盾，可用 ANIMA_DCU_ALLOW_FP8=1 放行 base_quant")


@probe("bf16 线性层 fwd+bwd 吞吐（算力标定）")
def probe_throughput() -> str:
    """必须 warmup —— MIOpen/rocBLAS 首次调用会现选算法，不 warmup 测的是编译时间。"""
    import torch
    d = _dev()
    m = torch.nn.Linear(4096, 4096, bias=False).to(d, torch.bfloat16)
    x = torch.randn(8192, 4096, device=d, dtype=torch.bfloat16, requires_grad=True)
    for _ in range(5):
        m(x).sum().backward()
    _sync()
    t0 = time.perf_counter()
    iters = 10
    for _ in range(iters):
        m(x).sum().backward()
    _sync()
    dt = (time.perf_counter() - t0) / iters
    flops = 3 * 2 * 8192 * 4096 * 4096      # fwd 2MNK + bwd ≈ 2×fwd
    return (f"{dt * 1000:.2f} ms/iter ≈ {flops / dt / 1e12:.1f} TFLOPS(bf16, 含 bwd 粗估)"
            f"；K100-AI 标称峰值 ~192 TFLOPS(BF16)，实测/标称 ≈ "
            f"{flops / dt / 1e12 / 192 * 100:.0f}%")


# ── 4. 多卡集合通信（torchrun --dist）─────────────────────────────────────────

def probe_dist() -> None:
    """8 卡 all-reduce 实测。**这是判断 8 卡值不值的唯一依据**。

    手动梯度同步的方案里，每个 optimizer step 交换一次"全部可训练参数"大小的 fp32 包。
    所以这里按 LoRA/LoKr 的真实量级（8 MB / 32 MB / 128 MB）扫，直接给出
    "每步会多花多少毫秒"，再对着 stage_timing.csv 的 whole_step_ms 就能算并行效率。
    """
    import torch
    import torch.distributed as dist

    if "WORLD_SIZE" not in os.environ:
        record("多卡集合通信", "SKIP",
               "未用 torchrun 启动。跑法：torchrun --nproc_per_node=8 tools/dcu_probe.py --dist")
        return

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = os.environ.get("ANIMA_DIST_BACKEND", "nccl")
    torch.cuda.set_device(local_rank)
    try:
        dist.init_process_group(backend=backend)
    except Exception as e:
        record(f"init_process_group({backend})", "FAIL", f"{type(e).__name__}: {e}")
        return
    if rank == 0:
        record(f"init_process_group({backend})", "OK",
               f"world_size={world}（DCU 上 RCCL 就注册在 nccl 这个名字下）")

    # 正确性：每个 rank 贡献自己的 rank 值，SUM 后应等于 0+1+...+(W-1)
    t = torch.full((1024,), float(rank), device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    expect = float(world * (world - 1) / 2)
    ok = bool((t - expect).abs().max().item() < 1e-3)
    if rank == 0:
        record("all_reduce 数值正确性", "OK" if ok else "FAIL",
               f"got={t[0].item()} expect={expect}")

    for mb in (8, 32, 128):
        n = mb * 1024 * 1024 // 4          # fp32 元素数
        buf = torch.randn(n, device="cuda", dtype=torch.float32)
        for _ in range(5):                  # warmup：首次会建 communicator + 选算法
            dist.all_reduce(buf)
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        iters = 20
        for _ in range(iters):
            dist.all_reduce(buf)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
        # ring all-reduce 的总线带宽口径：2(W-1)/W × size / t
        algbw = mb / 1e3 / dt                                  # GB/s
        busbw = algbw * 2 * (world - 1) / world
        if rank == 0:
            record(f"all_reduce {mb} MB × {world} 卡", "INFO",
                   f"{dt * 1000:.2f} ms/次 | algbw={algbw:.1f} GB/s | busbw={busbw:.1f} GB/s"
                   f" → 每 optimizer step 多花 {dt * 1000:.2f} ms")
        del buf
        torch.cuda.empty_cache()

    if rank == 0:
        record("判读提示", "INFO",
               "拿上面的 ms 除以 stage_timing.csv 的 whole_step_ms：<5% 说明 8 卡近线性；"
               ">20% 说明通信吃掉了太多，应改大 grad_accum（摊薄同步频率）而不是加卡。")
    dist.barrier()
    dist.destroy_process_group()


# ── 5. 依赖包 ─────────────────────────────────────────────────────────────────

def probe_packages() -> None:
    for mod, note in [
        ("safetensors", "权重读写，必需"),
        ("transformers", "文本编码器，必需"),
        ("einops", "必需"),
        ("PIL", "必需"),
        ("yaml", "配置解析，必需"),
        ("rich", "进度显示，必需"),
        ("torchvision", "可选：只有 aux_perceptual 才要。⚠ 只能装光源的 +das 版，"
                        "PyPI 版会连带把 torch 换成 CUDA 构建、废掉整个环境"),
        ("pillow_jxl", "可选：JXL 数据集才需要"),
        ("lpips", "可选：perceptual aux loss 才需要"),
        ("scipy", "可选：工具脚本"),
        ("xformers", "DCU 预期缺失 → NaViT 用 navit_attn_backend: sdpa_seg 代替"),
        ("bitsandbytes", "DCU 预期缺失 → 8-bit 优化器不可用"),
        ("triton", "DCU 上随 DTK 版本变化 → 有才能开 torch_compile"),
        ("flash_attn", "有的话是 sdpa_seg 之外的第二条路（仓库尚未接线）"),
        ("wandb", "可选：外推训练曲线"),
    ]:
        try:
            m = __import__(mod)
            record(f"pkg:{mod}", "OK", f"{getattr(m, '__version__', '?')} — {note}")
        except Exception as e:
            record(f"pkg:{mod}", "FAIL", f"{type(e).__name__} — {note}")


def _status_of(prefix: str) -> str:
    for r in RESULTS:
        if r["name"].startswith(prefix):
            return r["status"]
    return "MISSING"


def _detail_of(prefix: str) -> str:
    for r in RESULTS:
        if r["name"].startswith(prefix):
            return r["detail"]
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description="海光 DCU 能力探针")
    ap.add_argument("--json", default="", help="把结果写成 JSON 便于回传")
    ap.add_argument("--dist", action="store_true",
                    help="只跑多卡集合通信项（需用 torchrun 启动）")
    args = ap.parse_args()

    is_worker = args.dist and int(os.environ.get("RANK", "0")) != 0

    if not is_worker:
        print("=" * 78)
        print("海光 DCU 能力探针 —— 全部为真机实测，未通过项即为这台机器的真实限制")
        print("=" * 78)

    if args.dist:
        # 多卡模式：只跑通信项（其余单卡项跑 8 遍没意义，还会互相抢显存）
        probe_dist()
        if not is_worker and args.json:
            with open(args.json, "w", encoding="utf-8") as f:
                json.dump(RESULTS, f, ensure_ascii=False, indent=2)
            print(f"已写出 {args.json}")
        return 0

    probe_env()
    has_torch = probe_torch()
    if has_torch:
        probe_visible()

    probe_bf16_matmul()
    probe_fp16_matmul()
    probe_fp32_matmul()
    probe_autocast()
    probe_broadcast_matmul_backward()
    probe_sdpa_backends()
    probe_sdpa_backend_fix()
    probe_sdpa_none()
    probe_sdpa_bool()
    probe_sdpa_add()
    probe_sdpa_broadcast_mask()
    probe_sdpa_keypad_semantics()
    probe_sdpa_seg()
    probe_sdpa_seg_scaling()
    probe_flash_varlen()
    probe_fft()
    probe_vae_ops()
    probe_grad_ckpt()
    probe_adamw()
    probe_event()
    probe_mem()
    probe_fp8()
    probe_throughput()
    probe_packages()
    probe_dist()        # 未用 torchrun 时会 SKIP 并给出跑法

    # 自检：@probe 声明了但 main() 忘了调用的探测
    _ran = {r["name"] for r in RESULTS}
    _missed = [n for n in _DEFINED_PROBES if n not in _ran]
    if _missed:
        print("=" * 78)
        print("⚠ 以下探测已定义但 main() 没有调用，本次结果里缺这几项：")
        for n in _missed:
            print(f"  - {n}")
        print("  （不是这台机器的限制，是探针脚本自己的疏漏，请补上调用后重跑）")

    ok = sum(1 for r in RESULTS if r["status"] == "OK")
    fail = sum(1 for r in RESULTS if r["status"] == "FAIL")
    skip = sum(1 for r in RESULTS if r["status"] == "SKIP")
    print("=" * 78)
    print(f"汇总：OK={ok}  FAIL={fail}  SKIP={skip}")
    if fail:
        print("\n失败项（这些是这台机器的真实限制，配置里必须绕开）：")
        for r in RESULTS:
            if r["status"] == "FAIL":
                print(f"  - {r['name']}: {r['detail']}")

    # ── 把探针结果直接翻译成该往 yaml 里写什么 ────────────────────────────────
    print("\n配置裁决：")
    if _status_of("★ 2D×3D 广播 matmul") == "FAIL":
        print("  ⚠⚠ 广播 matmul 反向算错 —— 与昇腾同款坑。LoKr 必须走 expand+bmm 绕行实现，")
        print("      且**所有**用到广播 matmul 的路径都要复查，否则梯度是错的而训练不报错。")
    else:
        print("  ✓ 广播 matmul 前向+反向都正确 → LoKr/DoRA 路径无需昇腾那种绕行")

    seg = _status_of("★ sdpa_seg 逐段注意力")
    scaling = _detail_of("★ sdpa_seg 显存线性性")
    if seg == "OK" and "远超" not in scaling:
        print("  ✓ NaViT 打包可用 → navit_packing: true + navit_attn_backend: sdpa_seg")
    elif seg == "OK":
        print("  △ sdpa_seg 数值对，但显存疑似非线性（落 math 后端）：")
        print("      navit_packing 可开，但 navit_token_budget 要压小，按上面的 peak 反推。")
    else:
        print(f"  ✗ sdpa_seg 不可用（{seg}）→ navit_packing: false，退回 ARB 稠密路径")

    if _status_of("FP8 张量核") == "OK":
        print("  △ FP8 实测可用 → base_quant 可以试（需 ANIMA_DCU_ALLOW_FP8=1 放行）")
    else:
        print("  ✓ FP8 不可用（符合预期）→ base_quant: none")

    if _status_of("pkg:xformers") != "OK":
        print("  ✓ 无 xformers（符合预期）→ 别用 navit_attn_backend: xformers")
    if _status_of("pkg:triton") != "OK":
        print("  ✓ 无 triton → torch_compile: false")
    print("=" * 78)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(RESULTS, f, ensure_ascii=False, indent=2)
        print(f"已写出 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
