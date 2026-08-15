"""Kaggle TPU v5e-8 · Anima 8 卡纯数据并行训练步实测。

**这一轮要回答的**：Anima(2.09B) 在 v5e-8 上，8 卡纯 DP 的 tokens/s、峰值 HBM、
以及每卡 token 预算的真实上限。

**为什么是纯 DP 而不是前几轮为 Krea2 定下的 FSDP**：
Anima bf16 权重仅 4.18GB < 单 chip 15.7GiB，**权重放得下**，所以每卡各存一份、
各跑自己的数据，每步只 all-reduce LoRA 梯度（r32 全目标约 118MB bf16，按 P0 实测
的 all-reduce 110GB/s 约 1ms）。Krea2 那套"每层 all-gather 22.6GB x2"在这里是
纯多余的通信。这条由 F1 的权重账直接推出，本 job 用实测 HBM 复核。

**注意力形态（与前几轮的 NaViT 块对角不同，这是本轮要验的第二件事）**：
走「按 token 数分桶 + 批处理」——一个 pack 内 G 张图 token 数相同，张量是
[G, n, D]，于是：
  * 自注意力 = 每图独立 -> splash **FullMask** + vmap，**不需要块对角 mask、
    不需要 MaskInfo 预处理、没有 host 侧成本**（前几轮 G1/G2 实测 730-845ms/布局
    的那笔开销直接消失）；
  * cross-attn = q[n] x kv[T_txt]，T_txt 固定 -> 稠密即可（n*T_txt 很小）。
形状全静态、XLA 只编译一次。代价是同一 pack 内图片 token 数须一致（仓库已有
token_bucket 支持）。本 job 测它的绝对吞吐，与 NaViT 打包路线的取舍留给下一轮。

数值正确性**不在本 job 验**——已在本地用真权重对拍完成
（fp32、28 块全过、rel 4.6e-05，见 jax_tpu/ 与 scratchpad 的 check_parity）。
本 job 只测性能与显存，因此用随机权重（形状一致 -> 性能一致），
省掉 4.18GB 上传。
"""

import json
import os
import platform
import sys
import tempfile
import time
import traceback

import numpy as np

RESULTS: list = []
_T0 = time.time()
# _BOOT 由 ① 段的 preamble 设好（升级 jax[tpu] 的结果）；单独跑 _job_body 时兜底
_BOOT = globals().get("_BOOT", "未经 build_job.py 拼接，无 bootstrap")
OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else tempfile.gettempdir()

# ── 实验参数 ──────────────────────────────────────────────────────────────────
T_TXT = 256          # 每图 caption token 数（Anima 文本长度是动态的，取代表值）
LORA_RANK = 32
DEFAULT_TOKENS_PER_DEV = 16384   # 每卡 token 预算（= G * n）
N_IMG_TOKENS = 4096              # 每图 token 数（1024^2 -> 64x64 grid）


def _flush():
    try:
        with open(os.path.join(OUT_DIR, "anima_dp.json"), "w", encoding="utf-8") as f:
            json.dump(RESULTS, f, ensure_ascii=False, indent=2)
        with open(os.path.join(OUT_DIR, "anima_dp_report.txt"), "w", encoding="utf-8") as f:
            for r in RESULTS:
                f.write(f"[{r['status']:^6}] {r['name']}"
                        + (f" - {r['detail']}" if r["detail"] else "") + "\n")
            f.write(f"\n（截至 {time.time() - _T0:.0f}s 的快照）\n")
    except OSError:
        pass


def record(name, status, detail=""):
    RESULTS.append({"name": name, "status": status, "detail": detail,
                    "t_elapsed_s": round(time.time() - _T0, 1)})
    mark = {"OK": "  OK  ", "FAIL": " FAIL ", "SKIP": " SKIP ", "INFO": " INFO "}.get(status, status)
    print(f"[{mark}] {name}" + (f" - {detail}" if detail else ""), flush=True)
    _flush()


class _Skip(Exception):
    pass


def probe(name):
    def deco(fn):
        def run(*a, **kw):
            try:
                record(name, "OK", fn(*a, **kw) or "")
                return True
            except _Skip as s:
                record(name, "SKIP", str(s))
            except Exception as e:
                tb = traceback.format_exc(limit=3).strip().splitlines()[-1]
                record(name, "FAIL", f"{type(e).__name__}: {e} | {tb}")
            return False
        return run
    return deco


# ── 测量工具 ──────────────────────────────────────────────────────────────────
def _bench(fn, reps=3, warmup=1):
    """block_until_ready 强制同步。

    前几轮两次踩过的坑（都在这里被挡住）：① 不消费输出 -> 惰性执行根本没派发，
    测出比单个注意力还快的假值；② 不 warmup -> 首步的编译时间混进步时。
    """
    import jax
    t0 = time.time()
    jax.block_until_ready(fn())
    first = time.time() - t0
    for _ in range(warmup):
        jax.block_until_ready(fn())
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        jax.block_until_ready(fn())
        ts.append((time.perf_counter() - t) * 1e3)
    ts.sort()
    return ts[len(ts) // 2], first


def _hbm(peak=False):
    """**读 peak_bytes_in_use 而不是 bytes_in_use**。

    前一轮 spmd_probe 报的 "单卡 HBM 3.05GiB" 是常驻数组量、不是峰值，
    导致 "余量很大" 的结论与 "L=32768 立刻 OOM" 自相矛盾。这里两个都取。
    """
    import jax
    st = jax.devices()[0].memory_stats() or {}
    k = "peak_bytes_in_use" if peak else "bytes_in_use"
    return st.get(k, 0) / 1024 ** 3


def _reset_peak():
    import jax
    d = jax.devices()[0]
    for name in ("reset_memory_stats", "clear_memory_stats"):
        fn = getattr(d, name, None)
        if fn is not None:
            try:
                fn()
                return True
            except Exception:
                pass
    return False   # 无接口时 peak 是历史值，读数要按"自进程启动以来"解读


def _smap(f, mesh, in_specs, out_specs):
    """shard_map 包装，关掉静态复制检查。

    **为什么纯 DP 也必须用 shard_map**：splash 是 Pallas/Mosaic 内核，XLA 的
    GSPMD 自动分区处理不了它 —— 真机第一跑报
    `NotImplementedError: Mosaic kernels cannot be automatically partitioned.`
    所以哪怕逻辑上只是"每卡各算各的"，也得显式写成 shard_map 让每卡跑本地内核。

    参数名 jax 0.8 之前叫 check_rep、0.11 起叫 check_vma，两个都试。
    """
    from jax.experimental.shard_map import shard_map
    for kw in ({"check_vma": False}, {"check_rep": False}, {}):
        try:
            return shard_map(f, mesh=mesh, in_specs=in_specs,
                             out_specs=out_specs, **kw)
        except TypeError:
            continue
    raise RuntimeError("shard_map 不接受任何已知的 check_* 参数名")


def _cleanup():
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


def _grid_for(n_tok):
    """把 n_tok 拆成尽量方的 h x w（h*w 必须**恰好**等于 n_tok）。

    真机第一跑就栽在这里：原来用 side=round(sqrt(n))，n=512 时得 23x23=529≠512，
    广播直接报错。token 数不是完全平方数是常态（2304=48^2 是巧合，512/8192 都不是），
    所以必须走因数分解而不是开方取整。
    """
    h = max(d for d in range(1, int(n_tok ** 0.5) + 1) if n_tok % d == 0)
    return h, n_tok // h


def _rowcol(n_tok, jnp):
    h, w = _grid_for(n_tok)
    rr, cc = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing="ij")
    return rr.reshape(-1).astype(jnp.int32), cc.reshape(-1).astype(jnp.int32)


# ── 模型装配（复用上半部分的 anima_jax）────────────────────────────────────────
def make_random_params(key, cfg, dtype):
    """按真实形状随机造权重（性能只取决于形状）。**直接分片/直接 bf16 生成**：
    前几轮两次 OOM 都是先在单卡物化 fp32 再转换造成的，不重犯。"""
    import jax
    import jax.numpy as jnp

    D, C, R = cfg.model_channels, cfg.crossattn_dim, cfg.adaln_lora_dim
    F = int(D * cfg.mlp_ratio)
    ks = iter(jax.random.split(key, 8 + cfg.num_blocks * 16))

    def w(*shape):
        return (jax.random.normal(next(ks), shape, dtype) * 0.02)

    blocks = []
    for _ in range(cfg.num_blocks):
        blocks.append({
            "self_attn": {"q_proj": w(D, D), "k_proj": w(D, D), "v_proj": w(D, D),
                          "output_proj": w(D, D),
                          "q_norm": jnp.ones((cfg.head_dim,), dtype),
                          "k_norm": jnp.ones((cfg.head_dim,), dtype)},
            "cross_attn": {"q_proj": w(D, D), "k_proj": w(D, C), "v_proj": w(D, C),
                           "output_proj": w(D, D),
                           "q_norm": jnp.ones((cfg.head_dim,), dtype),
                           "k_norm": jnp.ones((cfg.head_dim,), dtype)},
            "mlp": {"layer1": w(F, D), "layer2": w(D, F)},
            "adaln_self_1": w(R, D), "adaln_self_2": w(3 * D, R),
            "adaln_cross_1": w(R, D), "adaln_cross_2": w(3 * D, R),
            "adaln_mlp_1": w(R, D), "adaln_mlp_2": w(3 * D, R),
        })
    return {
        "x_embedder": w(D, cfg.in_dim),
        "t_embedder_1": w(D, D), "t_embedder_2": w(3 * D, D),
        "t_embedding_norm": jnp.ones((D,), dtype),
        "final_adaln_1": w(R, D), "final_adaln_2": w(2 * D, R),
        "final_linear": w(cfg.out_dim, D),
        "blocks": blocks,
    }


def make_splash_full(n_tok, n_heads, interpret=False):
    """每图独立的全注意力 -> splash FullMask。

    **不需要块对角**：分桶批处理下每张图自成一个 batch 元素，图间天然不可见。
    这直接省掉前几轮 G1/G2 实测的 730-845ms/布局 MaskInfo host 开销。
    反向块 1024 + fused 来自 arch_probe H1（比默认快 2.03x）。
    """
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_kernel as sk, splash_attention_mask as sm,
    )
    # **块大小必须整除序列长**（splash_attention_mask_info.py:569 硬 raise）。
    # 反向想用 1024（H1 实测比默认 128 快 2.03x），但 1024 不整除 2304/9216 这类
    # token 数，所以按 n_tok 自适应取最大可用块 —— 这也意味着「每图 token 数」
    # 不能任取：至少要被 128 整除（前向块），否则 splash 直接构造失败。
    fwd = max(b for b in (128,) if n_tok % b == 0)
    bwd = next((b for b in (1024, 512, 256, 128) if n_tok % b == 0), 128)
    bs = sk.BlockSizes(block_q=fwd, block_kv=fwd, block_kv_compute=fwd,
                       block_q_dkv=bwd, block_kv_dkv=bwd, block_kv_dkv_compute=bwd,
                       use_fused_bwd_kernel=True)
    mask = sm.MultiHeadMask(masks=[sm.FullMask((n_tok, n_tok))] * n_heads)
    return sk.make_splash_mha(mask, head_shards=1, q_seq_shards=1,
                              block_sizes=bs, interpret=interpret)


def build_step(cfg, n_tok, dtype, interpret=False):
    """返回 loss_fn(lora, params, batch) —— 冻结底模、只对 LoRA 求梯度。"""
    import jax
    import jax.numpy as jnp
    import math as _m

    splash = make_splash_full(n_tok, cfg.num_heads, interpret)
    scale = cfg.head_dim ** -0.5

    def self_fn(q, k, v):
        # splash 期望 [H, S, D]，且**不内置 1/sqrt(d)**（前几轮硬结论，漏了会静默
        # 改变 softmax 温度、本地实测 rel 差到 9.5）。这里显式预缩放 q。
        qs = (q.astype(jnp.float32) * scale).astype(q.dtype).transpose(1, 0, 2)
        o = splash(qs, k.transpose(1, 0, 2), v.transpose(1, 0, 2))
        return o.transpose(1, 0, 2)

    def cross_fn(q, k, v):
        # q[n, H, D] x kv[T_TXT, H, D]，T_TXT 小 -> 稠密就够（n*T 的 logits 很小）
        logits = jnp.einsum("shd,thd->hst", q, k).astype(jnp.float32) * scale
        w = jax.nn.softmax(logits, axis=-1).astype(v.dtype)
        return jnp.einsum("hst,thd->shd", w, v)

    def per_image(params, lora, tok, t, ctx, rows, cols, target):
        mod_index = jnp.zeros((n_tok,), jnp.int32)     # 单图 -> 全部指向第 0 行
        out = forward_packed(params, cfg, tok, t[None], ctx, rows, cols, mod_index,
                             self_fn, cross_fn, loras=lora, remat=True)
        return jnp.mean((out.astype(jnp.float32) - target.astype(jnp.float32)) ** 2)

    def loss_fn(lora, params, batch):
        # vmap 到 pack 内的 G 张图；DP 由 jit 的 sharding 负责跨卡
        losses = jax.vmap(per_image, in_axes=(None, None, 0, 0, 0, 0, 0, 0))(
            params, lora, batch["tok"], batch["t"], batch["ctx"],
            batch["rows"], batch["cols"], batch["target"])
        return jnp.mean(losses)

    return loss_fn


def make_batch(key, cfg, n_dev, G, n_tok, dtype, mesh):
    """造一个分片好的 batch：[n_dev*G, ...]，第 0 维按卡切。"""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    B = n_dev * G
    ks = jax.random.split(key, 5)
    sh = lambda spec: NamedSharding(mesh, spec)

    def mk():
        rr, cc = _rowcol(n_tok, jnp)
        return {
            "tok": jax.random.normal(ks[0], (B, n_tok, cfg.in_dim), dtype),
            "t": jax.random.uniform(ks[1], (B,), jnp.float32),
            "ctx": jax.random.normal(ks[2], (B, T_TXT, cfg.crossattn_dim), dtype),
            "rows": jnp.broadcast_to(rr[None], (B, n_tok)),
            "cols": jnp.broadcast_to(cc[None], (B, n_tok)),
            "target": jax.random.normal(ks[3], (B, n_tok, cfg.out_dim), dtype),
        }

    specs = {k: P("d") for k in ("tok", "t", "ctx", "rows", "cols", "target")}
    return jax.jit(mk, out_shardings={k: sh(v) for k, v in specs.items()})()


# ── 探测 ──────────────────────────────────────────────────────────────────────
def probe_env():
    record("python", "INFO", f"{platform.python_version()} @ {sys.executable}")
    record("bootstrap", "INFO", _BOOT)


@probe("0.1 设备 / HBM")
def p_basics():
    import jax
    d = jax.devices()
    lim = (d[0].memory_stats() or {}).get("bytes_limit", 0) / 1024 ** 3
    return (f"jax {jax.__version__} | {len(d)} x {d[0].device_kind} | "
            f"单设备 HBM limit={lim:.1f}GiB")


@probe("A1 splash 可用性（Pallas 闸门）")
def p_splash():
    import jax
    import jax.numpy as jnp
    fn = make_splash_full(256, 4)
    q = jnp.zeros((4, 256, 128), jnp.bfloat16)
    o = jax.jit(fn)(q, q, q)
    jax.block_until_ready(o)
    pv = jax.devices()[0].client.platform_version
    return f"splash 编译通过 | {pv.split('Built on')[-1].strip()[:24] if 'Built on' in pv else pv[:40]}"


@probe("W1 Anima 权重账（决定要不要分片）")
def p_weights():
    import jax
    cfg = AnimaConfig()
    D, C, R = cfg.model_channels, cfg.crossattn_dim, cfg.adaln_lora_dim
    F = int(D * cfg.mlp_ratio)
    per = (4 * D * D) + (2 * D * D + 2 * D * C) + (2 * D * F) + 3 * (R * D + 3 * D * R)
    total = per * cfg.num_blocks + D * cfg.in_dim + D * D + 3 * D * D + D + R * D + 2 * D * R + cfg.out_dim * D
    lim = (jax.devices()[0].memory_stats() or {}).get("bytes_limit", 0) / 1024 ** 3
    return (f"每块 {per / 1e6:.1f}M x {cfg.num_blocks} 块 = {total / 1e9:.3f}B | "
            f"bf16 {total * 2 / 1e9:.2f}GB vs 单 chip {lim:.1f}GiB -> "
            f"**{'放得下，纯 DP 即可，无需 FSDP/TP' if total * 2 / 1024**3 < lim else '放不下，须分片'}**")


@probe("B1 LoRA 接线自检（没有它，基准可能在测一个空转的模型）")
def p_lora_wired():
    """本地冒烟抓到过一次：jax.checkpoint 把层号 trace 成 tracer，f"blocks.{i}" 拼出
    垃圾键 -> LoRA 全部匹配不上、梯度恒 0，且 XLA 可能把反向整个 DCE 掉使步时假性
    变快。这类错不报异常，只能靠断言拦。故基准前先在真机上验一次。

    判据两条：① grad_b 必须非零（LoRA 真的在图里）；② grad_a 必须恒 0
    （B 零初始化 -> step-0 净增量为 0，本仓库落地 adapter 的硬约束）。
    """
    import jax
    import jax.numpy as jnp
    _cleanup()
    cfg = AnimaConfig(num_blocks=2)
    n = 512
    params = make_random_params(jax.random.PRNGKey(0), cfg, jnp.bfloat16)
    lora = init_lora(jax.random.PRNGKey(1), cfg, LORA_RANK, dtype=jnp.bfloat16)
    rr, cc = _rowcol(n, jnp)
    G = 2
    batch = {
        "tok": jax.random.normal(jax.random.PRNGKey(2), (G, n, cfg.in_dim), jnp.bfloat16),
        "t": jnp.array([0.3, 0.8], jnp.float32),
        "ctx": jax.random.normal(jax.random.PRNGKey(3), (G, T_TXT, cfg.crossattn_dim), jnp.bfloat16),
        "rows": jnp.broadcast_to(rr[None], (G, n)),
        "cols": jnp.broadcast_to(cc[None], (G, n)),
        "target": jax.random.normal(jax.random.PRNGKey(4), (G, n, cfg.out_dim), jnp.bfloat16),
    }
    g = jax.jit(jax.grad(build_step(cfg, n, jnp.bfloat16), argnums=0))(lora, params, batch)
    ga = sum(float(jnp.abs(v["a"]).sum()) for v in g.values())
    gb = sum(float(jnp.abs(v["b"]).sum()) for v in g.values())
    nz = sum(1 for v in g.values() if float(jnp.abs(v["b"]).max()) > 0)
    if gb <= 0:
        raise RuntimeError(f"grad_b 全 0 —— LoRA 没接进图，后续步时无意义（{len(g)} 条）")
    if ga != 0:
        raise RuntimeError(f"grad_a={ga} 非 0 —— B 不是零初始化，step-0 不中立")
    _cleanup()
    return (f"sum|grad_b|={gb:.3e}，非零条目 {nz}/{len(g)}；"
            f"sum|grad_a|=0（step-0 中立）-> LoRA 已接进计算图")


def _run_dp(G, n_tok, reset_peak=True):
    """跑一次 8 卡 DP 训练步，返回实测字典。"""
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    _cleanup()
    if reset_peak:
        _reset_peak()
    devs = jax.devices()
    if len(devs) < 8:
        raise _Skip(f"只有 {len(devs)} 个设备")
    mesh = Mesh(np.array(devs).reshape(8), ("d",))
    cfg = AnimaConfig()
    dtype = jnp.bfloat16
    k = jax.random.PRNGKey(0)

    params = jax.jit(lambda: make_random_params(k, cfg, dtype),
                     out_shardings=NamedSharding(mesh, P()))()      # 复制到每卡
    lora = jax.jit(lambda: init_lora(jax.random.PRNGKey(1), cfg, LORA_RANK, dtype=dtype),
                   out_shardings=NamedSharding(mesh, P()))()
    batch = make_batch(jax.random.PRNGKey(2), cfg, 8, G, n_tok, dtype, mesh)

    local = build_step(cfg, n_tok, dtype)

    def per_shard(lora, params, batch):
        # 每卡跑自己那 G 张图；返回 [1] 而不是标量（out_specs=P("d") 需要有轴可切）
        return local(lora, params, batch)[None]

    data_spec = {k: P("d") for k in ("tok", "t", "ctx", "rows", "cols", "target")}
    f = _smap(per_shard, mesh, (P(), P(), data_spec), P("d"))
    # lora/params 是复制的（in_spec P()）-> 它们的余切在 shard_map 转置时自动 psum，
    # 即 LoRA 梯度的 all-reduce 已包含在这一步的实测时间里。
    loss_fn = lambda lo, pa, ba: jnp.mean(f(lo, pa, ba))
    grad = jax.jit(jax.grad(loss_fn, argnums=0))
    ms, first = _bench(lambda: grad(lora, params, batch))
    tok = 8 * G * n_tok
    n_lora = sum(int(v["a"].size + v["b"].size) for v in lora.values())
    return {"ms": ms, "first": first, "tok": tok, "tok_s": tok / (ms / 1e3),
            "hbm": _hbm(), "peak": _hbm(peak=True), "G": G, "n": n_tok,
            "lora_mb": n_lora * 2 / 1e6}


@probe("D1 8 卡纯 DP 全 28 块训练步（主判据）")
def p_dp():
    G = DEFAULT_TOKENS_PER_DEV // N_IMG_TOKENS
    r = _run_dp(G, N_IMG_TOKENS)
    return (f"每卡 {r['G']} 图 x {r['n']} token = {r['G'] * r['n']} token/卡 | "
            f"8 卡共 {r['tok'] / 1000:.0f}k token/步 | 步时 {r['ms']:.0f}ms"
            f"（首步含编译 {r['first']:.0f}s）-> **{r['tok_s'] / 1e3:.1f}k tokens/s** | "
            f"常驻 HBM {r['hbm']:.2f}GiB / **峰值 {r['peak']:.2f}GiB** | "
            f"LoRA {r['lora_mb']:.0f}MB(bf16，每步 all-reduce 量)")


@probe("D2 每卡 token 预算扫描（找真实上限，不止步于'能跑的最大值'）")
def p_budget():
    out = []
    for budget in (8192, 16384, 24576, 32768, 49152):
        G = budget // N_IMG_TOKENS
        if G < 1:
            continue
        try:
            r = _run_dp(G, N_IMG_TOKENS)
            out.append(f"{budget}(G={G}): {r['ms']:.0f}ms {r['tok_s'] / 1e3:.1f}k tok/s "
                       f"峰值{r['peak']:.1f}GiB")
        except Exception as e:
            out.append(f"{budget}(G={G}): {type(e).__name__}")
            _cleanup()
    return " ; ".join(out)


@probe("D3 单图 token 数扫描（分辨率维度）")
def p_imgsize():
    out = []
    for n in (1024, 2304, 4096, 9216, 16384):
        G = max(1, 16384 // n)
        try:
            r = _run_dp(G, n)
            out.append(f"n={n}(G={G}): {r['ms']:.0f}ms {r['tok_s'] / 1e3:.1f}k tok/s "
                       f"峰值{r['peak']:.1f}GiB")
        except Exception as e:
            out.append(f"n={n}(G={G}): {type(e).__name__}")
            _cleanup()
    return " ; ".join(out)


@probe("D4 单卡对照（算并行效率，1 卡跑同样的每卡负载）")
def p_single():
    import jax
    import jax.numpy as jnp
    _cleanup()
    cfg = AnimaConfig()
    dtype = jnp.bfloat16
    G = DEFAULT_TOKENS_PER_DEV // N_IMG_TOKENS
    n = N_IMG_TOKENS
    dev = jax.devices()[0]
    params = jax.jit(lambda: make_random_params(jax.random.PRNGKey(0), cfg, dtype),
                     device=dev)()
    lora = jax.jit(lambda: init_lora(jax.random.PRNGKey(1), cfg, LORA_RANK, dtype=dtype),
                   device=dev)()
    ks = jax.random.split(jax.random.PRNGKey(2), 4)
    rr, cc = _rowcol(n, jnp)
    batch = jax.jit(lambda: {
        "tok": jax.random.normal(ks[0], (G, n, cfg.in_dim), dtype),
        "t": jax.random.uniform(ks[1], (G,), jnp.float32),
        "ctx": jax.random.normal(ks[2], (G, T_TXT, cfg.crossattn_dim), dtype),
        "rows": jnp.broadcast_to(rr[None], (G, n)),
        "cols": jnp.broadcast_to(cc[None], (G, n)),
        "target": jax.random.normal(ks[3], (G, n, cfg.out_dim), dtype),
    }, device=dev)()
    grad = jax.jit(jax.grad(build_step(cfg, n, dtype), argnums=0))
    ms, _ = _bench(lambda: grad(lora, params, batch))
    return (f"单卡 {G} 图 x {n} token | 步时 {ms:.0f}ms -> "
            f"{(G * n) / (ms / 1e3) / 1e3:.1f}k tokens/s（8 卡理想 = 此值 x8）")


def main():
    probe_env()
    if not p_basics():
        _flush()
        return 0
    p_weights()
    if not p_splash():
        record("裁决", "FAIL", "splash 不可用 -> 后续步时无意义，停在这里")
        _flush()
        return 0
    if not p_lora_wired():
        record("裁决", "FAIL", "LoRA 未接进图 -> 步时会测到一个空转模型，停在这里")
        _flush()
        return 0
    p_dp()
    p_single()
    p_budget()
    p_imgsize()
    ok = sum(1 for r in RESULTS if r["status"] == "OK")
    bad = sum(1 for r in RESULTS if r["status"] == "FAIL")
    record("汇总", "INFO", f"OK={ok} FAIL={bad} 总耗时={time.time() - _T0:.0f}s")
    _flush()
    return 0   # 恒返回 0：单项 FAIL 是数据，不是脚本故障（Kaggle 会把非零判成 ERROR）


if __name__ == "__main__":
    sys.exit(main())
