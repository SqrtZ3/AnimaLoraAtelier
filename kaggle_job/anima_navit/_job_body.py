"""Kaggle TPU v5e-8 · Anima NaViT 打包路线的真机裁决。

上一轮（anima-tpu-dp-probe）走的是**分桶批处理**（每图一个 batch 元素 + splash
FullMask），拿到 8 卡纯 DP 30.1k tokens/s、并行效率≈100%。但那条路要求同 pack 内
每图 token 数相同，且 splash 要块整除序列长 -> token 数必须被 128 整除，而 128
整除的 token 数在 aspect<=2 下宽高比覆盖极差（4096=2^12 只有 1024x1024 正方形，
本地枚举实证）。用户要求保住 NaViT 的任意分辨率哲学，故本轮改走**打包 + 块对角**。

## 本轮要回答的四件事（都只有真机能答）

P1 数值：两级 mask（编译期粗粒度块对角跳块 + 运行时 segment_ids 精确边界）
   在**真内核**下是否 ≡ 稠密参考。本地已用 interpret=True 验过语义，但 interpret
   是解释执行，不代表 Mosaic 真内核一致。
P2 跳块是否真生效：块对角 vs 全通的步时比 vs 理论比。
   （前几轮在 Krea2 形状上测过 0.252/0.250，Anima 是 16 头 x128、形状不同，重测。）
P4 **remat x budget 的最优点（主判据）**：上一轮 28 块无条件全 remat，峰值 HBM
   只有 4.9/15.7 GiB，10.8 GiB 闲置，而全 remat 要把前向整个重算一遍（冻结底模
   只训 LoRA -> 反向无 wgrad -> 一步 ≈ 前向x3，重算占其中 1/3）。这一格能省多少、
   代价是 budget 要降多少，只能实测。
P5 布局切换的编译成本 + 持久化编译缓存是否命中（决定 4 种布局的一次性代价）。

## 已知不在本轮范围

优化器/数据/checkpoint 尚未接（仍是随机权重 + MSE 到随机 target），本轮只测
**注意力正确性与训练步的时间/显存**。形状一致则性能一致，故不上传 3.91GB 权重。
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
_BOOT = globals().get("_BOOT", "未经 build_job.py 拼接，无 bootstrap")
OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else tempfile.gettempdir()

T_TXT = 256          # 每图文本槽长度（128 的倍数，cross-attn 的 kv 侧）
LORA_RANK = 32

# 真实布局：段长 = 真实图像 token 数向上量化到 1024（本地枚举出的甜点 Q=1024，
# budget=32768 时只有 4 种布局、有效填充率 94.8%）。real 取自真实 ARB 桶的
# token 数（10080=96x105、4096=64x64、9216=96x96、2304=48x48）。
LAYOUTS = {
    8192:  ([4096, 3072, 1024], [4096, 2304, 0]),
    16384: ([10240, 4096, 2048], [10080, 4096, 0]),
    32768: ([10240, 9216, 4096, 4096, 3072, 2048], [10080, 9216, 4096, 4096, 2304, 0]),
    49152: ([10240] * 4 + [4096, 4096], [10080] * 4 + [4096, 4096]),
    65536: ([10240] * 5 + [9216, 4096, 1024], [10080] * 5 + [9216, 4096, 0]),
}
# 每个元组：(粗粒度段长 -> 编译期 mask，真实 token 数 -> 运行时 segment_ids)
# 真实数为 0 的段是纯 padding 段（FFD 装箱的余量）。


def _flush():
    try:
        with open(os.path.join(OUT_DIR, "anima_navit.json"), "w", encoding="utf-8") as f:
            json.dump(RESULTS, f, ensure_ascii=False, indent=2)
        with open(os.path.join(OUT_DIR, "anima_navit_report.txt"), "w", encoding="utf-8") as f:
            for r in RESULTS:
                f.write(f"[{r['status']:^6}] {r['name']}"
                        + (f" - {r['detail']}" if r["detail"] else "") + "\n")
            f.write(f"\n（截至 {time.time() - _T0:.0f}s 的快照）\n")
    except OSError:
        pass


def record(name, status, detail=""):
    RESULTS.append({"name": name, "status": status, "detail": detail,
                    "t_elapsed_s": round(time.time() - _T0, 1)})
    print(f"[{status:^6}] {name}" + (f" - {detail}" if detail else ""), flush=True)
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
    """block_until_ready 强制同步。不消费输出 -> 惰性执行根本没派发（测出假值）；
    不 warmup -> 首步编译时间混进步时。两个坑前几轮都踩过。"""
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


_PEAK_RESETTABLE = None


def _hbm(peak=False):
    import jax
    st = jax.devices()[0].memory_stats() or {}
    return st.get("peak_bytes_in_use" if peak else "bytes_in_use", 0) / 1024 ** 3


def _reset_peak():
    """尝试清零峰值统计；**返回是否成功**。

    上一轮的教训：`_reset_peak` 失败时静默降级，于是 D2/D3 五个配置全部报同一个
    "峰值 4.9GiB"（其实是自进程启动以来的历史最大值），被当成各配置的实测峰值。
    这里把成功与否记进全局，报告里显式标注，并把扫描按**显存递增序**跑，
    使历史峰值至少是个单调包络而不是废数。
    """
    global _PEAK_RESETTABLE
    import jax
    d = jax.devices()[0]
    for name in ("reset_memory_stats", "clear_memory_stats"):
        fn = getattr(d, name, None)
        if fn is not None:
            try:
                fn()
                _PEAK_RESETTABLE = True
                return True
            except Exception:
                pass
    _PEAK_RESETTABLE = False
    return False


def _smap(f, mesh, in_specs, out_specs):
    """shard_map 包装。**纯 DP 也必须用它**：splash 是 Pallas/Mosaic 内核，
    XLA 的 GSPMD 自动分区处理不了（真机报 `Mosaic kernels cannot be
    automatically partitioned.`）。参数名 0.8 前叫 check_rep、0.11 起叫 check_vma。"""
    from jax.experimental.shard_map import shard_map
    for kw in ({"check_vma": False}, {"check_rep": False}, {}):
        try:
            return shard_map(f, mesh=mesh, in_specs=in_specs, out_specs=out_specs, **kw)
        except TypeError:
            continue
    raise RuntimeError("shard_map 不接受任何已知的 check_* 参数名")


def _cleanup(drop_kernels: bool = False):
    """释放显存。

    **不再逐个 delete `jax.live_arrays()`** —— 上一跑就栽在这里：
    `make_splash_mha` 构造出的 MaskInfo 是 jax 数组，存活在被 lru_cache 缓存的
    可调用对象里；delete-all 把它们删了，缓存里的 kernel 就变成持有已删除数组的
    空壳，下次同布局命中缓存直接 RuntimeError。表现是"第一格成功、同布局第二格
    必炸"，与实测完全吻合（8192/full OK -> 8192/every2 RuntimeError）。

    改为只 gc；要真正换布局时传 drop_kernels=True 清掉 kernel 缓存，
    那时缓存里的数组才会一起被回收。
    """
    import gc
    if drop_kernels:
        try:
            _splash_kernel.cache_clear()
        except Exception:
            pass
    gc.collect()


def _grid_for(n):
    """把 n 拆成尽量方的 h x w（h*w **恰好** = n）。真机第一跑栽过：
    round(sqrt(512))^2=529≠512 直接广播报错。token 数不是完全平方数是常态。"""
    h = max(d for d in range(1, int(n ** 0.5) + 1) if n % d == 0)
    return h, n // h


# ── pack 构造 ─────────────────────────────────────────────────────────────────
def build_pack(coarse, real, jnp):
    """由 (粗粒度段长, 真实 token 数) 造出一个 pack 的全部运行时数组。

    返回 dict：
      seg_self  [B] 自注意力的精细段号；段内填充统一给 -1（彼此可见，行不空）
      seg_cross [B] cross-attn 的精细段号；填充**沿用宿主段号**
                    （文本侧没有配对的填充段，给独立段号会导致整行全 0 -> softmax 分母 0，
                     splash 的 SegmentIds 文档对此有明确警告）
      mod_index [B] AdaLN 的 token->段 索引（= 粗粒度段号）
      rows/cols [B] RoPE 用的网格坐标；填充位置填 0（其输出被 loss mask 丢弃）
      loss_mask [B] 1=真 token
    """
    B = sum(coarse)
    seg_self = np.full(B, -1, np.int32)
    seg_cross = np.empty(B, np.int32)
    mod_index = np.empty(B, np.int32)
    rows = np.zeros(B, np.int32)
    cols = np.zeros(B, np.int32)
    loss_mask = np.zeros(B, np.float32)
    off = 0
    for i, (c, r) in enumerate(zip(coarse, real)):
        seg_cross[off:off + c] = i
        mod_index[off:off + c] = i
        if r:
            seg_self[off:off + r] = i
            loss_mask[off:off + r] = 1.0
            h, w = _grid_for(r)
            rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
            rows[off:off + r] = rr.reshape(-1)
            cols[off:off + r] = cc.reshape(-1)
        off += c
    to = lambda a: jnp.asarray(a)
    return {"seg_self": to(seg_self), "seg_cross": to(seg_cross),
            "mod_index": to(mod_index), "rows": to(rows), "cols": to(cols),
            "loss_mask": to(loss_mask)}


def make_random_params(key, cfg, dtype):
    """按真实形状随机造权重（性能只取决于形状）。**直接 bf16 生成**：
    前几轮两次 OOM 都是先在单卡物化 fp32 再转换造成的。"""
    import jax
    import jax.numpy as jnp
    D, C, R = cfg.model_channels, cfg.crossattn_dim, cfg.adaln_lora_dim
    F = int(D * cfg.mlp_ratio)
    ks = iter(jax.random.split(key, 8 + cfg.num_blocks * 16))
    w = lambda *s: jax.random.normal(next(ks), s, dtype) * 0.02
    one = lambda n: jnp.ones((n,), dtype)
    blocks = [{
        "self_attn": {"q_proj": w(D, D), "k_proj": w(D, D), "v_proj": w(D, D),
                      "output_proj": w(D, D), "q_norm": one(cfg.head_dim),
                      "k_norm": one(cfg.head_dim)},
        "cross_attn": {"q_proj": w(D, D), "k_proj": w(D, C), "v_proj": w(D, C),
                       "output_proj": w(D, D), "q_norm": one(cfg.head_dim),
                       "k_norm": one(cfg.head_dim)},
        "mlp": {"layer1": w(F, D), "layer2": w(D, F)},
        "adaln_self_1": w(R, D), "adaln_self_2": w(3 * D, R),
        "adaln_cross_1": w(R, D), "adaln_cross_2": w(3 * D, R),
        "adaln_mlp_1": w(R, D), "adaln_mlp_2": w(3 * D, R),
    } for _ in range(cfg.num_blocks)]
    # **stack_blocks**：训练只走 lax.scan 路径。展开路径实测 1.01 MB/token
    # （anima-mem-probe 归因：AdaLN 调制不依赖 x -> 28 块的 [ΣN,3D] 被 XLA
    #  提前调度成同时活着），budget 16384 即 OOM。
    return stack_blocks({"x_embedder": w(D, cfg.in_dim), "t_embedder_1": w(D, D),
                         "t_embedder_2": w(3 * D, D), "t_embedding_norm": one(D),
                         "final_adaln_1": w(R, D), "final_adaln_2": w(2 * D, R),
                         "final_linear": w(cfg.out_dim, D), "blocks": blocks})


def build_loss(cfg, coarse, real, pack, remat, interpret=False):
    """返回 loss_fn(lora, params, batch)：冻结底模、只对 LoRA 求梯度。"""
    import jax.numpy as jnp
    txt = [T_TXT] * len(coarse)
    self_fn = bind_segments(
        make_splash_attn(coarse, coarse, cfg.num_heads, cfg.head_dim,
                         interpret=interpret), pack["seg_self"], pack["seg_self"])
    cross_fn = bind_segments(
        make_splash_attn(coarse, txt, cfg.num_heads, cfg.head_dim,
                         interpret=interpret),
        pack["seg_cross"], jnp.asarray(segment_ids(txt)))

    def loss_fn(lora, params, batch):
        out = forward_packed(params, cfg, batch["tok"], batch["t"], batch["ctx"],
                             pack["rows"], pack["cols"], pack["mod_index"],
                             self_fn, cross_fn, loras=lora, remat=remat)
        se = jnp.mean((out.astype(jnp.float32) - batch["target"].astype(jnp.float32)) ** 2,
                      axis=-1)
        # **padding token 必须排除**：它们的输出是垃圾，计进 loss 会污染梯度
        m = pack["loss_mask"]
        return jnp.sum(se * m) / jnp.maximum(jnp.sum(m), 1.0)

    return loss_fn


def make_batch(key, cfg, n_dev, coarse, dtype, mesh):
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    B, G = sum(coarse), len(coarse)
    ks = jax.random.split(key, 4)
    spec = {k: P("d") for k in ("tok", "t", "ctx", "target")}

    def mk():
        return {"tok": jax.random.normal(ks[0], (n_dev, B, cfg.in_dim), dtype),
                "t": jax.random.uniform(ks[1], (n_dev, G), jnp.float32),
                "ctx": jax.random.normal(ks[2], (n_dev, G * T_TXT, cfg.crossattn_dim), dtype),
                "target": jax.random.normal(ks[3], (n_dev, B, cfg.out_dim), dtype)}

    return jax.jit(mk, out_shardings={k: NamedSharding(mesh, v)
                                      for k, v in spec.items()})()


# ── 探测 ──────────────────────────────────────────────────────────────────────
@probe("P0 设备 / HBM")
def p0():
    import jax
    d = jax.devices()
    lim = (d[0].memory_stats() or {}).get("bytes_limit", 0) / 1024 ** 3
    _reset_peak()
    return (f"jax {jax.__version__} | {len(d)} x {d[0].device_kind} | "
            f"单设备 HBM limit={lim:.1f}GiB | peak 可清零={_PEAK_RESETTABLE}")


@probe("P1 两级 mask 数值正确性（真内核 vs 稠密参考）")
def p1():
    """本地 interpret=True 已验过语义，但 interpret 是解释执行，不代表 Mosaic
    真内核一致。这里在真机上复验，**前向 + 反向都验**（训练用的是反向）。"""
    import jax
    import jax.numpy as jnp
    _cleanup()
    coarse, real = [1024, 512, 512], [900, 500, 0]
    B, H, Dh = sum(coarse), 16, 128
    pack = build_pack(coarse, real, jnp)
    r = lambda k: jax.random.normal(jax.random.PRNGKey(k), (B, H, Dh),
                                    jnp.float32).astype(jnp.bfloat16)
    q, k, v = r(0), r(1), r(2)
    sp = make_splash_attn(coarse, coarse, H, Dh, pack["seg_self"], pack["seg_self"])
    dn = make_dense_attn(coarse, coarse, np.asarray(pack["seg_self"]),
                         np.asarray(pack["seg_self"]))
    f = lambda fn: float(jnp.abs(jax.jit(fn)(q, k, v).astype(jnp.float32)
                                 - dn(q, k, v).astype(jnp.float32)).max())
    fwd = f(sp)
    gs = jax.grad(lambda a: jnp.sum(sp(a, k, v).astype(jnp.float32) ** 2))(q)
    gd = jax.grad(lambda a: jnp.sum(dn(a, k, v).astype(jnp.float32) ** 2))(q)
    bwd = float(jnp.abs(gs.astype(jnp.float32) - gd.astype(jnp.float32)).max()
                / max(float(jnp.abs(gd).max()), 1e-9))
    # 行为判据：动段内填充区，真 token 输出必须逐 bit 不变
    kp = k.at[900:1024].set(r(9)[900:1024])
    leak = float(jnp.abs(sp(q, k, v)[:900].astype(jnp.float32)
                         - sp(q, kp, v)[:900].astype(jnp.float32)).max())
    if leak != 0.0:
        raise RuntimeError(f"段内填充泄漏进真 token（max_abs={leak:.3e}）—— "
                           f"segment_ids 没生效，训练会静默学错")
    _cleanup()
    return (f"前向 max_abs={fwd:.3e} | 反向 dq rel={bwd:.3e} | "
            f"段内填充泄漏={leak:.1e}（必须 0）")


@probe("P2 跳块是否真生效（块对角 vs 全通）")
def p2():
    import jax
    import jax.numpy as jnp
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_kernel as sk, splash_attention_mask as sm)
    _cleanup()
    coarse = [10240, 4096, 2048]
    B, H, Dh = sum(coarse), 16, 128
    q = jax.random.normal(jax.random.PRNGKey(0), (H, B, Dh), jnp.float32).astype(jnp.bfloat16)
    qs = segment_ids(coarse)
    bs = _block_sizes(B, B)
    diag = sk.make_splash_mha(sm.MultiHeadMask([block_diag_mask(qs, qs)] * H),
                              head_shards=1, q_seq_shards=1, block_sizes=bs)
    full = sk.make_splash_mha(sm.MultiHeadMask([sm.FullMask((B, B))] * H),
                              head_shards=1, q_seq_shards=1, block_sizes=bs)
    md, _ = _bench(lambda: jax.jit(diag)(q, q, q))
    mf, _ = _bench(lambda: jax.jit(full)(q, q, q))
    theo = sum(c * c for c in coarse) / (B * B)
    _cleanup()
    return (f"段长={coarse} | 块对角 {md:.2f}ms vs 全通 {mf:.2f}ms -> "
            f"实测比 **{md / mf:.3f}**，理论 {theo:.3f}"
            + ("（跳块生效）" if md / mf < theo * 1.5 else "（**跳块没生效**，查 mask 构造）"))


@probe("P3 LoRA 接线自检（没有它，后面测的可能是个空转的模型）")
def p3():
    """jax.checkpoint 会把入参 trace 成 tracer，层号若作为入参，f"blocks.{i}"
    拼出垃圾键 -> LoRA 静默失效、梯度恒 0，且反向可能被 DCE 掉使步时**假性变快**。
    判据：grad_b 非零（真接进图）+ grad_a 恒 0（B 零初始化 -> step-0 中立）。"""
    import jax
    import jax.numpy as jnp
    _cleanup()
    cfg = AnimaConfig(num_blocks=2)
    coarse, real = [1024, 1024], [900, 800]
    pack = build_pack(coarse, real, jnp)
    params = make_random_params(jax.random.PRNGKey(0), cfg, jnp.bfloat16)
    lora = stack_loras(init_lora(jax.random.PRNGKey(1), cfg, LORA_RANK,
                                 dtype=jnp.bfloat16), cfg.num_blocks)
    ks = jax.random.split(jax.random.PRNGKey(2), 4)
    B, G = sum(coarse), len(coarse)
    batch = {"tok": jax.random.normal(ks[0], (B, cfg.in_dim), jnp.bfloat16),
             "t": jax.random.uniform(ks[1], (G,), jnp.float32),
             "ctx": jax.random.normal(ks[2], (G * T_TXT, cfg.crossattn_dim), jnp.bfloat16),
             "target": jax.random.normal(ks[3], (B, cfg.out_dim), jnp.bfloat16)}
    g = jax.jit(jax.grad(build_loss(cfg, coarse, real, pack, "full"),
                         argnums=0))(lora, params, batch)
    ga = sum(float(jnp.abs(v["a"]).sum()) for v in g.values())
    gb = sum(float(jnp.abs(v["b"]).sum()) for v in g.values())
    nz = sum(1 for v in g.values() if float(jnp.abs(v["b"]).max()) > 0)
    if gb <= 0:
        raise RuntimeError(f"grad_b 全 0 —— LoRA 没接进图（{len(g)} 条）")
    if ga != 0:
        raise RuntimeError(f"grad_a={ga} 非 0 —— B 不是零初始化，step-0 不中立")
    _cleanup()
    return (f"sum|grad_b|={gb:.3e}，非零 target {nz}/{len(g)}（scan 布局：每项含 "
            f"{cfg.num_blocks} 块）；sum|grad_a|=0（step-0 中立）")


def _run_dp(budget, remat):
    """跑一次 8 卡 DP 训练步，返回实测字典。"""
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    _cleanup()
    _reset_peak()
    devs = jax.devices()
    if len(devs) < 8:
        raise _Skip(f"只有 {len(devs)} 个设备")
    mesh = Mesh(np.array(devs).reshape(8), ("d",))
    cfg, dtype = AnimaConfig(), jnp.bfloat16
    coarse, real = LAYOUTS[budget]
    pack = build_pack(coarse, real, jnp)

    params = jax.jit(lambda: make_random_params(jax.random.PRNGKey(0), cfg, dtype),
                     out_shardings=NamedSharding(mesh, P()))()
    lora = jax.jit(lambda: stack_loras(init_lora(jax.random.PRNGKey(1), cfg,
                                                 LORA_RANK, dtype=dtype),
                                       cfg.num_blocks),
                   out_shardings=NamedSharding(mesh, P()))()
    batch = make_batch(jax.random.PRNGKey(2), cfg, 8, coarse, dtype, mesh)
    local = build_loss(cfg, coarse, real, pack, remat)

    def per_shard(lo, pa, ba):
        # shard_map 给每卡 [1, ...] 的切片 -> 去掉那一维
        b = {k: v[0] for k, v in ba.items()}
        return local(lo, pa, b)[None]

    dspec = {k: P("d") for k in ("tok", "t", "ctx", "target")}
    f = _smap(per_shard, mesh, (P(), P(), dspec), P("d"))
    # lora/params 是复制的(in_spec P()) -> 其余切在 shard_map 转置时自动 psum，
    # 即 LoRA 梯度的 all-reduce 已含在实测步时里。
    grad = jax.jit(jax.grad(lambda lo, pa, ba: jnp.mean(f(lo, pa, ba)), argnums=0))
    ms, first = _bench(lambda: grad(lora, params, batch))
    tok_real = 8 * sum(real)
    return {"ms": ms, "first": first, "budget": budget,
            "tok_pad": 8 * budget, "tok_real": tok_real,
            "tok_s": tok_real / (ms / 1e3), "peak": _hbm(peak=True), "hbm": _hbm()}


@probe("P4 remat x budget 扫描（主判据）")
def p4():
    """按**显存递增序**跑：peak 若不可清零，历史峰值至少是单调包络。"""
    out = []
    for budget in sorted(LAYOUTS):
        _cleanup(drop_kernels=True)      # 换布局：旧 kernel 及其 MaskInfo 可以回收了
        for remat in ("full", "every2", "dots", "none"):
            try:
                r = _run_dp(budget, remat)
                out.append(f"{budget}/{remat}: {r['ms']:.0f}ms "
                           f"{r['tok_s'] / 1e3:.1f}k真tok/s "
                           f"({r['tok_pad'] / (r['ms'] / 1e3) / 1e3:.1f}k含填充) "
                           f"峰值{r['peak']:.1f}GiB 首调{r['first']:.0f}s")
            except Exception as e:
                out.append(f"{budget}/{remat}: {type(e).__name__}: "
                           f"{str(e)[:180].replace(chr(10), ' ')}")
                _cleanup()
    note = "" if _PEAK_RESETTABLE else "（**peak 不可清零，读数是历史包络、非各配置实测峰值**）"
    return " ; ".join(out) + note


@probe("P5 布局切换的编译成本 + 持久化缓存")
def p5():
    """每种布局要单独编译一次全模型。本地枚举：Q=1024 时 budget=32768 只有 4 种
    布局，故一次性成本 ≈ 4 x 这里测到的首调耗时；能否靠 /kaggle/working/jax_cache
    跨 run 摊掉，看第二列。"""
    import jax
    cache = os.path.join(OUT_DIR, "jax_cache")
    before = len(os.listdir(cache)) if os.path.isdir(cache) else 0
    try:
        jax.config.update("jax_compilation_cache_dir", cache)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    except Exception as e:
        return f"无法开启持久化缓存：{type(e).__name__}: {e}"
    firsts = []
    for budget in sorted(LAYOUTS):
        try:
            firsts.append(f"{budget}: {_run_dp(budget, 'full')['first']:.0f}s")
        except Exception as e:
            firsts.append(f"{budget}: {type(e).__name__}: "
                          f"{str(e)[:180].replace(chr(10), ' ')}")
            _cleanup()
    after = len(os.listdir(cache)) if os.path.isdir(cache) else 0
    return (f"各布局首调（含编译）: {' ; '.join(firsts)} | "
            f"编译缓存条目 {before}->{after}"
            + ("（可作下一棒 dataset 输入摊掉）" if after > before else "（**未写入，摊不掉**）"))


def main():
    record("python", "INFO", f"{platform.python_version()} @ {sys.executable}")
    record("bootstrap", "INFO", _BOOT)
    if not p0():
        _flush()
        return 0
    if not p1():
        record("裁决", "FAIL", "两级 mask 数值不对 -> 后续步时无意义，停在这里")
        _flush()
        return 0
    p2()
    if not p3():
        record("裁决", "FAIL", "LoRA 未接进图 -> 会测到一个空转的模型，停在这里")
        _flush()
        return 0
    p4()
    p5()
    ok = sum(1 for r in RESULTS if r["status"] == "OK")
    bad = sum(1 for r in RESULTS if r["status"] == "FAIL")
    record("汇总", "INFO", f"OK={ok} FAIL={bad} 总耗时={time.time() - _T0:.0f}s")
    _flush()
    return 0   # 恒返回 0：单项 FAIL 是数据，不是脚本故障（Kaggle 把非零判成 ERROR）


if __name__ == "__main__":
    sys.exit(main())
