"""Kaggle TPU v5e-8 · **布局裁决**：NaViT 序列打包 vs 量化分桶（批维）。

## 为什么要这一跑

上一轮（anima-navit-probe）的结论是：打包 + 块对角 splash 能跑，但
  * 显存 ≈ 4.0GB + **1.01 MB/token**，必须靠 `lax.scan` 压住；
  * 压住之后 **`full` 成了唯一可用 remat 档**（every2/dots/none 全 OOM）；
  * 吞吐 25.4k 真tok/s，**仍未胜过**第一棒的分桶批处理 29.8k。

显存归因（anima-mem-probe 单变量）指向 AdaLN 调制：它不依赖 x，28 块展开后
XLA 把 28 组 [ΣN, 3D] 同时留着。但那是**打包布局**的产物 —— 打包路径必须把
逐图的 shift/scale/gate `gather` 成逐 token 的 [ΣN, 3D]（真物化），而批布局
只需要 [G, 1, 3D] 广播（XLA 融合进消费者，不物化）。分桶路径实测 0.02 MB/token，
正是这个差别。

于是本轮问的是一个**布局**问题，不是实现问题：

> 段长量化到 Q 之后，块对角注意力与批维注意力在算力上完全等价
> （都是 Σ L_i²）。那么"打包"到底还买到了什么？

**任意宽高比不是打包独有的**：桶按**量化后的 token 总数**分，不按 (h,w) 分；
每图保留自己的 (h_i, w_i)，只要 h_i*w_i <= L 即可同批，RoPE 逐 token 查
rows/cols、不依赖矩形网格。所以用户要的"任意分辨率哲学"两条路都满足。

## 本轮要回答的

R1 **数值**：`forward_ragged` ≡ `forward_packed`（真机真内核，同一批图）。
   过了这条，分桶路线就继承了打包路线全部已验证的数值口径（八道闸门）。
R2 **段内填充隔离**：分桶路径的 segment_ids 是否真把量化填充挡住（判据：动填充
   区的 k，真 token 输出必须逐 bit 不变）。打包路径不加它时实测污染 max_abs=5.4，
   静默 —— 分桶路径同样的坑。
R3 LoRA 接线自检（没有它，后面测的可能是个空转的模型）。
R4 **主判据**：两条路线的步时 / 真tok/s / 有效 MFU / 峰值，remat 四档全扫。
R5 **分桶路径还需不需要 scan**：同配置下 scan vs 展开。若展开可用，
   `lax.scan` 那条"未 remat 的激活按 28 次迭代堆叠"的约束就不存在了，
   `every2` 的 15% 才拿得回来。

## 不在本轮范围

优化器/数据/checkpoint 不接（随机权重 + MSE 到随机 target）；本轮只裁决布局。
形状一致则性能一致，故不上传 3.91GB 权重。
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

T_TXT = 512          # 每图文本槽长度 —— **真实口径**（trainer/config.py 的 512，
                     # navit_text_trim_padding 默认 False，pad 照常参与 cross-attn）。
                     # 注意上一轮 anima-navit-probe 用的是 256，故本轮打包侧的
                     # 绝对数字与那一轮不可直接比；本轮内部 A/B 才是同口径的。
LORA_RANK = 32
V5E_PEAK_TFLOPS = 197.0      # 单 chip bf16 峰值；8 卡 = 1576

# ── 打包路线的布局（沿用上一轮，段长 = 真实 token 数量化到 1024）────────────────
# 每个元组：(粗粒度段长 -> 编译期 mask，真实 token 数 -> 运行时 segment_ids)
PACKED_LAYOUTS = {
    8192:  ([4096, 3072, 1024], [4096, 2304, 0]),
    16384: ([10240, 4096, 2048], [10080, 4096, 0]),
    32768: ([10240, 9216, 4096, 4096, 3072, 2048], [10080, 9216, 4096, 4096, 2304, 0]),
}

# ── 分桶路线的配置：(L 桶长, 每卡图数 G, 每图真实 token 数)────────────────────
# L 取自同样的量化格点，real 取自真实 ARB 桶：
#   3952 = jacknife-anima2 的典型值（token 3952-4160，近似均匀）
#   10080 = 96x105、9216 = 96x96、4096 = 64x64、2304 = 48x48（modare-anima3 侧）
RAGGED_CONFIGS = [
    (4096, 1, 3952), (4096, 2, 3952), (4096, 4, 3952), (4096, 8, 3952),
    (3072, 4, 2304), (3072, 8, 2304),
    (10240, 1, 10080), (10240, 2, 10080), (10240, 3, 10080),
]
REMATS = ("full", "every2", "dots", "none")


def _flush():
    try:
        with open(os.path.join(OUT_DIR, "anima_ragged.json"), "w", encoding="utf-8") as f:
            json.dump(RESULTS, f, ensure_ascii=False, indent=2)
        with open(os.path.join(OUT_DIR, "anima_ragged_report.txt"), "w", encoding="utf-8") as f:
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
    """尝试清零峰值统计；**返回是否成功**。失败时读数只是历史包络，报告里要标注
    （上一轮就出过"三个配置同报 3.9GiB"的废数）。"""
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
    """释放显存。**不逐个 delete `jax.live_arrays()`** —— splash kernel 的 MaskInfo
    是 jax 数组、活在 lru_cache 的可调用对象里，delete-all 会让缓存变空壳，
    表现为"第一格成功、同布局第二格必炸"（上一轮实测）。只 gc；换布局时清缓存。"""
    import gc
    if drop_kernels:
        for c in (globals().get("_splash_kernel"), globals().get("_bucket_kernel")):
            try:
                c.cache_clear()
            except Exception:
                pass
    gc.collect()


def _grid_for(n):
    """把 n 拆成尽量方的 h x w（h*w **恰好** = n）。真机第一跑栽过：
    round(sqrt(512))^2=529≠512 直接广播报错。token 数不是完全平方数是常态。"""
    h = max(d for d in range(1, int(n ** 0.5) + 1) if n % d == 0)
    return h, n // h


# ── 有效算力口径 ──────────────────────────────────────────────────────────────
def fwd_flops(cfg, n_pad, attn_pairs, cross_pairs):
    """一次**前向**的 FLOPs（单卡）。

      n_pad        本卡参与线性层的 token 数（含填充 —— 填充一样要过 MLP）
      attn_pairs   Σ L_i²        自注意力的 (q,kv) 对数
      cross_pairs  Σ L_i * T     cross-attn 的对数

    注意力每对 (q,kv) 的代价是 4*D（QKᵀ 的 2*D + AV 的 2*D）。
    AdaLN 只在 G 行上算，逐 token 项里不计。
    """
    D, C = cfg.model_channels, cfg.crossattn_dim
    F = int(D * cfg.mlp_ratio)
    p_lin = 4 * D * D + 2 * D * D + 2 * D * C + 2 * D * F
    return cfg.num_blocks * (2 * p_lin * n_pad + 4 * D * (attn_pairs + cross_pairs))


def useful_mfu(cfg, ms, n_dev, n_pad, attn_pairs, cross_pairs):
    """**有效 MFU**：只算"理论必需"的算力，不含 remat 的重算。

    冻结底模只训 LoRA -> 反向没有 wgrad -> 理论下界 = 前向(1) + 激活梯度(1) = 2x 前向。
    这样 remat 档位的代价会直接体现为有效 MFU 的下降，而不是被口径吸收掉 ——
    比"含重算的 MFU"更能回答"这一档值不值"。
    """
    total = 2.0 * fwd_flops(cfg, n_pad, attn_pairs, cross_pairs) * n_dev
    return total / (ms / 1e3) / (V5E_PEAK_TFLOPS * 1e12 * n_dev)


# ── 权重 ──────────────────────────────────────────────────────────────────────
def make_random_params(key, cfg, dtype, stack=True):
    """按真实形状随机造权重（性能只取决于形状）。**直接 bf16 生成**：
    前几轮两次 OOM 都是先在单卡物化 fp32 再转换造成的。

    `stack=False` 保留 list-of-dict（展开路径要），`True` 堆成 [L,...]（scan 路径要）。
    """
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
    p = {"x_embedder": w(D, cfg.in_dim), "t_embedder_1": w(D, D),
         "t_embedder_2": w(3 * D, D), "t_embedding_norm": one(D),
         "final_adaln_1": w(R, D), "final_adaln_2": w(2 * D, R),
         "final_linear": w(cfg.out_dim, D), "blocks": blocks}
    return stack_blocks(p) if stack else p


# ── 打包路线 ──────────────────────────────────────────────────────────────────
def build_pack(coarse, real, jnp):
    """由 (粗粒度段长, 真实 token 数) 造出一个 pack 的全部运行时数组。

    seg_self  段内填充统一给 -1（彼此可见，行不空）
    seg_cross 填充**沿用宿主段号**（文本侧没有配对填充段，独立段号 -> 整行全 0
              -> softmax 分母为 0，splash 的 SegmentIds 文档对此有明确警告）
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


def build_packed_loss(cfg, coarse, pack, remat, interpret=False):
    """打包路线的 loss_fn(lora, params, batch)：冻结底模、只对 LoRA 求梯度。"""
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
        m = pack["loss_mask"]        # padding token 的输出是垃圾，计进 loss 会污染梯度
        return jnp.sum(se * m) / jnp.maximum(jnp.sum(m), 1.0)

    return loss_fn


# ── 分桶路线 ──────────────────────────────────────────────────────────────────
def build_bucket(L, G, real, jnp):
    """由 (桶长 L, 每卡图数 G, 每图真实 token 数 real) 造出运行时数组。

    这里让 G 张图**真实 token 数相同**只是为了 A/B 干净；真实训练里同桶各图的
    real 可以各不相同（segment_ids 是 [G, L] 逐图的），(h_i, w_i) 也各不相同。
    """
    seg = bucket_segment_ids([real] * G, L)                # [G, L] 0=真 1=填充
    rows = np.zeros((G, L), np.int32)
    cols = np.zeros((G, L), np.int32)
    loss_mask = np.zeros((G, L), np.float32)
    h, w = _grid_for(real)
    rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    for g in range(G):
        rows[g, :real] = rr.reshape(-1)
        cols[g, :real] = cc.reshape(-1)
        loss_mask[g, :real] = 1.0
    to = lambda a: jnp.asarray(a)
    return {"seg": to(seg), "rows": to(rows), "cols": to(cols),
            "loss_mask": to(loss_mask)}


def build_ragged_loss(cfg, L, buck, remat, interpret=False):
    """分桶路线的 loss_fn。**注意 cross-attn 完全没有 mask** —— 每图看自己那一份
    定长文本槽的全部，这正是相对打包路线省掉的整块复杂度（打包那边是矩形块对角
    splash，anima-mem-probe 归因里占 4.2GB）。"""
    import jax.numpy as jnp
    self_fn = bind_segments(
        make_bucket_attn(L, L, cfg.num_heads, cfg.head_dim, use_segments=True,
                         interpret=interpret), buck["seg"], buck["seg"])
    cross_fn = make_bucket_attn(L, T_TXT, cfg.num_heads, cfg.head_dim,
                                use_segments=False, interpret=interpret)

    def loss_fn(lora, params, batch):
        out = forward_ragged(params, cfg, batch["tok"], batch["t"], batch["ctx"],
                             buck["rows"], buck["cols"],
                             self_fn, cross_fn, loras=lora, remat=remat)
        se = jnp.mean((out.astype(jnp.float32) - batch["target"].astype(jnp.float32)) ** 2,
                      axis=-1)
        m = buck["loss_mask"]
        return jnp.sum(se * m) / jnp.maximum(jnp.sum(m), 1.0)

    return loss_fn


def make_batch(key, cfg, n_dev, tok_shape, ctx_shape, n_img, dtype, mesh):
    """在设备上直接生成，省掉 host->device 搬运（也避免单卡物化 fp32 再转换）。"""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    ks = jax.random.split(key, 4)
    spec = {k: P("d") for k in ("tok", "t", "ctx", "target")}

    def mk():
        return {"tok": jax.random.normal(ks[0], (n_dev, *tok_shape, cfg.in_dim), dtype),
                "t": jax.random.uniform(ks[1], (n_dev, n_img), jnp.float32),
                "ctx": jax.random.normal(ks[2], (n_dev, *ctx_shape, cfg.crossattn_dim), dtype),
                "target": jax.random.normal(ks[3], (n_dev, *tok_shape, cfg.out_dim), dtype)}

    return jax.jit(mk, out_shardings={k: NamedSharding(mesh, v)
                                      for k, v in spec.items()})()


# ── 探测 ──────────────────────────────────────────────────────────────────────
@probe("R0 设备 / HBM")
def r0():
    import jax
    d = jax.devices()
    lim = (d[0].memory_stats() or {}).get("bytes_limit", 0) / 1024 ** 3
    _reset_peak()
    return (f"jax {jax.__version__} | {len(d)} x {d[0].device_kind} | "
            f"单设备 HBM limit={lim:.1f}GiB | peak 可清零={_PEAK_RESETTABLE}")


@probe("R1 forward_ragged ≡ forward_packed（真内核，同一批图）")
def r1():
    """**最关键的一道闸门**：过了它，分桶路线就继承打包路线全部已验证的数值口径。

    构造两张图（真实 token 数不同 -> 量化到同一个桶长 L），分别走两条路线，
    只在**真 token** 上比。两条路径共用 `_forward_core`，理论上差别只有布局与
    注意力后端，所以残差应当落在 bf16 噪声量级（~1e-3 相对）。
    """
    import jax
    import jax.numpy as jnp
    _cleanup(drop_kernels=True)
    cfg = AnimaConfig(num_blocks=4)          # 4 块足够暴露布局错误，省编译时间
    L, reals = 1024, [900, 768]
    G = len(reals)
    dtype = jnp.bfloat16
    params = make_random_params(jax.random.PRNGKey(0), cfg, dtype, stack=True)
    lora = stack_loras(init_lora(jax.random.PRNGKey(1), cfg, LORA_RANK, dtype=dtype),
                       cfg.num_blocks)
    # B 是零初始化 -> LoRA 净增量为 0，两条路线都一样；这里把 B 填成非零，
    # 好让 LoRA 支路真正参与对拍（否则等于没测 LoRA）
    lora = {k: {"a": v["a"], "b": jax.random.normal(
        jax.random.PRNGKey(7), v["b"].shape, dtype) * 0.02} for k, v in lora.items()}

    ks = jax.random.split(jax.random.PRNGKey(2), 3)
    tok_b = jax.random.normal(ks[0], (G, L, cfg.in_dim), dtype)
    t = jax.random.uniform(ks[1], (G,), jnp.float32)
    ctx_b = jax.random.normal(ks[2], (G, T_TXT, cfg.crossattn_dim), dtype)
    # 填充位置的输入置 0（真实数据里就是零填充），保证两条路线输入完全同源
    keep = (np.arange(L)[None, :] < np.asarray(reals)[:, None])
    tok_b = tok_b * jnp.asarray(keep, dtype)[..., None]

    buck = build_bucket(L, G, reals[0], jnp)
    buck["seg"] = jnp.asarray(bucket_segment_ids(reals, L))
    rows_b = np.zeros((G, L), np.int32)
    cols_b = np.zeros((G, L), np.int32)
    for g, r in enumerate(reals):
        h, w = _grid_for(r)
        rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        rows_b[g, :r], cols_b[g, :r] = rr.reshape(-1), cc.reshape(-1)
    buck["rows"], buck["cols"] = jnp.asarray(rows_b), jnp.asarray(cols_b)

    self_r = bind_segments(make_bucket_attn(L, L, cfg.num_heads, cfg.head_dim,
                                            use_segments=True), buck["seg"], buck["seg"])
    cross_r = make_bucket_attn(L, T_TXT, cfg.num_heads, cfg.head_dim,
                               use_segments=False)
    out_r = jax.jit(lambda: forward_ragged(params, cfg, tok_b, t, ctx_b,
                                           buck["rows"], buck["cols"],
                                           self_r, cross_r, loras=lora, remat="full"))()

    # 打包侧：同样两张图，段长都是 L（粗粒度），真实 token 数 = reals
    coarse = [L] * G
    pack = build_pack(coarse, reals, jnp)
    pack["rows"] = jnp.asarray(rows_b.reshape(-1))
    pack["cols"] = jnp.asarray(cols_b.reshape(-1))
    tok_p = tok_b.reshape(G * L, cfg.in_dim)
    ctx_p = ctx_b.reshape(G * T_TXT, cfg.crossattn_dim)
    self_p = bind_segments(make_splash_attn(coarse, coarse, cfg.num_heads, cfg.head_dim),
                           pack["seg_self"], pack["seg_self"])
    cross_p = bind_segments(
        make_splash_attn(coarse, [T_TXT] * G, cfg.num_heads, cfg.head_dim),
        pack["seg_cross"], jnp.asarray(segment_ids([T_TXT] * G)))
    out_p = jax.jit(lambda: forward_packed(params, cfg, tok_p, t, ctx_p,
                                           pack["rows"], pack["cols"], pack["mod_index"],
                                           self_p, cross_p, loras=lora, remat="full"))()

    a = np.asarray(out_r, np.float32).reshape(G * L, -1)
    b = np.asarray(out_p, np.float32)
    m = keep.reshape(-1)
    num = np.abs(a[m] - b[m]).max()
    den = max(float(np.abs(b[m]).max()), 1e-9)
    _cleanup(drop_kernels=True)
    if num / den > 5e-2:
        raise RuntimeError(f"两条路线不等价：rel={num / den:.3e}（bf16 噪声应 <5e-2）")
    return f"真 token 上 max_abs={num:.3e} rel={num / den:.3e}（bf16 噪声量级）"


@probe("R2 分桶路径的段内填充隔离（segment_ids 是否真生效）")
def r2():
    """判据是**行为**而不是数值接近：动填充区的 k，真 token 输出必须逐 bit 不变。
    不加 segment_ids 时打包路径实测污染 max_abs=5.4，且**静默**。"""
    import jax
    import jax.numpy as jnp
    _cleanup(drop_kernels=True)
    L, G, real, H, Dh = 1024, 2, 900, 16, 128
    r = lambda k: jax.random.normal(jax.random.PRNGKey(k), (G, L, H, Dh),
                                    jnp.float32).astype(jnp.bfloat16)
    q, k, v = r(0), r(1), r(2)
    seg = jnp.asarray(bucket_segment_ids([real] * G, L))
    attn = bind_segments(make_bucket_attn(L, L, H, Dh, use_segments=True), seg, seg)
    f = jax.jit(attn)
    kp = k.at[:, real:].set(r(9)[:, real:])
    leak = float(jnp.abs(f(q, k, v)[:, :real].astype(jnp.float32)
                         - f(q, kp, v)[:, :real].astype(jnp.float32)).max())
    # 反证：关掉 segment_ids 就应该被污染（证明这个测试本身有分辨力）
    bare = jax.jit(make_bucket_attn(L, L, H, Dh, use_segments=False))
    contam = float(jnp.abs(bare(q, k, v)[:, :real].astype(jnp.float32)
                           - bare(q, kp, v)[:, :real].astype(jnp.float32)).max())
    _cleanup(drop_kernels=True)
    if leak != 0.0:
        raise RuntimeError(f"填充泄漏进真 token（max_abs={leak:.3e}）—— 训练会静默学错")
    if contam == 0.0:
        raise RuntimeError("关掉 segment_ids 也没有污染 -> 这个测试没有分辨力，判据无效")
    return f"泄漏={leak:.1e}（必须 0）；关掉 segment_ids 的对照污染={contam:.3e}（证明判据有效）"


@probe("R3 LoRA 接线自检")
def r3():
    """jax.checkpoint 会把入参 trace 成 tracer，层号若作为入参，f"blocks.{i}" 拼出
    垃圾键 -> LoRA 静默失效、梯度恒 0，且反向可能被 DCE 掉使步时**假性变快**。
    判据：grad_b 非零（真接进图）+ grad_a 恒 0（B 零初始化 -> step-0 中立）。"""
    import jax
    import jax.numpy as jnp
    _cleanup(drop_kernels=True)
    cfg = AnimaConfig(num_blocks=2)
    L, G, real = 1024, 2, 900
    buck = build_bucket(L, G, real, jnp)
    params = make_random_params(jax.random.PRNGKey(0), cfg, jnp.bfloat16)
    lora = stack_loras(init_lora(jax.random.PRNGKey(1), cfg, LORA_RANK,
                                 dtype=jnp.bfloat16), cfg.num_blocks)
    ks = jax.random.split(jax.random.PRNGKey(2), 4)
    batch = {"tok": jax.random.normal(ks[0], (G, L, cfg.in_dim), jnp.bfloat16),
             "t": jax.random.uniform(ks[1], (G,), jnp.float32),
             "ctx": jax.random.normal(ks[2], (G, T_TXT, cfg.crossattn_dim), jnp.bfloat16),
             "target": jax.random.normal(ks[3], (G, L, cfg.out_dim), jnp.bfloat16)}
    g = jax.jit(jax.grad(build_ragged_loss(cfg, L, buck, "full"), argnums=0))(
        lora, params, batch)
    ga = sum(float(jnp.abs(v["a"]).sum()) for v in g.values())
    gb = sum(float(jnp.abs(v["b"]).sum()) for v in g.values())
    nz = sum(1 for v in g.values() if float(jnp.abs(v["b"]).max()) > 0)
    _cleanup(drop_kernels=True)
    if gb <= 0:
        raise RuntimeError(f"grad_b 全 0 —— LoRA 没接进图（{len(g)} 条）")
    if ga != 0:
        raise RuntimeError(f"grad_a={ga} 非 0 —— B 不是零初始化，step-0 不中立")
    return f"sum|grad_b|={gb:.3e}，非零 target {nz}/{len(g)}；sum|grad_a|=0（step-0 中立）"


# ── 主判据 ────────────────────────────────────────────────────────────────────
def _run_step(route, spec, remat, stack=True):
    """跑一次 8 卡 DP 训练步（前向 + LoRA 反向 + 跨卡 all-reduce），返回实测字典。

    route="packed" -> spec=budget；route="ragged" -> spec=(L, G, real)
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    _cleanup()
    _reset_peak()
    devs = jax.devices()
    if len(devs) < 8:
        raise _Skip(f"只有 {len(devs)} 个设备")
    n_dev = 8
    mesh = Mesh(np.array(devs[:n_dev]).reshape(n_dev), ("d",))
    cfg, dtype = AnimaConfig(), jnp.bfloat16

    if route == "packed":
        coarse, real = PACKED_LAYOUTS[spec]
        pack = build_pack(coarse, real, jnp)
        local_builder = lambda: build_packed_loss(cfg, coarse, pack, remat)
        tok_shape, ctx_shape = (sum(coarse),), (len(coarse) * T_TXT,)
        n_img = len(coarse)
        n_pad, tok_real = sum(coarse), sum(real)
        attn_pairs = sum(c * c for c in coarse)
        cross_pairs = sum(c * T_TXT for c in coarse)
        tag = f"packed/{spec}"
    else:
        L, G, real = spec
        buck = build_bucket(L, G, real, jnp)
        local_builder = lambda: build_ragged_loss(cfg, L, buck, remat)
        tok_shape, ctx_shape = (G, L), (G, T_TXT)
        n_img = G
        n_pad, tok_real = G * L, G * real
        attn_pairs, cross_pairs = G * L * L, G * L * T_TXT
        tag = f"ragged/L{L}xG{G}"

    params = jax.jit(lambda: make_random_params(jax.random.PRNGKey(0), cfg, dtype, stack),
                     out_shardings=NamedSharding(mesh, P()))()
    flat = init_lora(jax.random.PRNGKey(1), cfg, LORA_RANK, dtype=dtype)
    lora = jax.jit(lambda: stack_loras(flat, cfg.num_blocks) if stack else flat,
                   out_shardings=NamedSharding(mesh, P()))()
    batch = make_batch(jax.random.PRNGKey(2), cfg, n_dev, tok_shape, ctx_shape,
                       n_img, dtype, mesh)
    local = local_builder()

    def per_shard(lo, pa, ba):
        b = {k: v[0] for k, v in ba.items()}      # shard_map 给每卡 [1,...] 的切片
        return local(lo, pa, b)[None]

    dspec = {k: P("d") for k in ("tok", "t", "ctx", "target")}
    f = _smap(per_shard, mesh, (P(), P(), dspec), P("d"))
    # lora/params 是复制的(in_spec P()) -> 其余切在 shard_map 转置时自动 psum，
    # 即 LoRA 梯度的 8 卡 all-reduce 已含在实测步时里。
    grad = jax.jit(jax.grad(lambda lo, pa, ba: jnp.mean(f(lo, pa, ba)), argnums=0))
    ms, first = _bench(lambda: grad(lora, params, batch))
    mfu = useful_mfu(cfg, ms, n_dev, n_pad, attn_pairs, cross_pairs)
    return {"tag": tag, "ms": ms, "first": first,
            "tok_real": n_dev * tok_real, "tok_pad": n_dev * n_pad,
            "fill": tok_real / n_pad,
            "tok_s": n_dev * tok_real / (ms / 1e3),
            "mfu": mfu, "peak": _hbm(peak=True)}


def _fmt(r):
    return (f"{r['ms']:.0f}ms {r['tok_s'] / 1e3:.1f}k真tok/s "
            f"有效MFU {r['mfu']:.1%} 填充率 {r['fill']:.1%} "
            f"峰值{r['peak']:.1f}GiB 首调{r['first']:.0f}s")


def _sweep(route, specs, remats, stack=True, label=""):
    """**逐 spec 即时 record**：扫描很长（数十格 x 编译），Kaggle 超时/OOM 杀进程时
    只有已 record 的行落了盘。上一轮把整段扫描攒到最后一次性写，风险太大。"""
    out = []
    for spec in specs:
        _cleanup(drop_kernels=True)          # 换布局：旧 kernel 及其 MaskInfo 可回收
        row = []
        for remat in remats:
            try:
                r = _run_step(route, spec, remat, stack)
                row.append(f"{remat}: {_fmt(r)}")
            except Exception as e:
                row.append(f"{remat}: {type(e).__name__}: "
                           f"{str(e)[:200].replace(chr(10), ' ')}")
                _cleanup()
        line = " | ".join(row)
        record(f"  {label}{route}/{spec}", "DATA", line)
        out.append(f"{spec}: {line}")
    return out


@probe("R4a 分桶路线扫描（L x 每卡图数 x remat）")
def r4a():
    """**主判据的一半**。有效 MFU 是 remat 无关的口径（只算理论必需的 2x 前向），
    所以 remat 档位的代价会直接体现为 MFU 下降 —— 比"含重算的 MFU"更能回答
    "这一档值不值"。"""
    _sweep("ragged", RAGGED_CONFIGS, REMATS)
    return ("逐行见上方 DATA 条目"
            + ("" if _PEAK_RESETTABLE else "（**peak 不可清零，读数是历史包络**）"))


@probe("R4b 打包路线对照（同一份代码、同一 T_TXT=512 口径）")
def r4b():
    """上一轮 anima-navit-probe 用的是 T_TXT=256，绝对数字与本轮不可直接比。
    这里在同口径下重测，保证 A/B 是单变量的。"""
    _sweep("packed", sorted(PACKED_LAYOUTS), REMATS)
    return "逐行见上方 DATA 条目"


@probe("R5 分桶路线还需不需要 scan（scan vs 展开）")
def r5():
    """打包路线下 `lax.scan` 是必需的（展开 1.01 MB/token，budget 16384 即 OOM），
    但它的语义代价是：**未被 remat 掉的激活会按 28 次迭代堆叠成 [L,...]**，
    于是 every2/dots 在 scan 下比展开更贵，`full` 成了唯一可用档。

    如果分桶路线的显存本来就不成问题，展开路径就能用，`every2` 的那 15% 才拿得回来。
    """
    picks = [(4096, 2, 3952), (10240, 1, 10080)]
    _sweep("ragged", picks, REMATS, stack=False, label="展开 ")
    return "展开档见上方 DATA 条目；scan 档同配置见 R4a"


def main():
    record("python", "INFO", f"{platform.python_version()} @ {sys.executable}")
    record("bootstrap", "INFO", _BOOT)
    record("口径", "INFO", f"T_TXT={T_TXT} rank={LORA_RANK} "
                           f"有效MFU分母={V5E_PEAK_TFLOPS}TFLOPS/chip x8")
    if not r0():
        _flush()
        return 0
    if not r1():
        record("裁决", "FAIL", "两条路线不等价 -> 后续步时无可比性，停在这里")
        _flush()
        return 0
    if not r2():
        record("裁决", "FAIL", "分桶路径填充隔离失效 -> 训练会静默学错，停在这里")
        _flush()
        return 0
    if not r3():
        record("裁决", "FAIL", "LoRA 未接进图 -> 会测到一个空转的模型，停在这里")
        _flush()
        return 0
    r4a()
    r4b()
    r5()
    ok = sum(1 for r in RESULTS if r["status"] == "OK")
    bad = sum(1 for r in RESULTS if r["status"] == "FAIL")
    record("汇总", "INFO", f"OK={ok} FAIL={bad} 总耗时={time.time() - _T0:.0f}s")
    _flush()
    return 0   # 恒返回 0：单项 FAIL 是数据，不是脚本故障（Kaggle 把非零判成 ERROR）


if __name__ == "__main__":
    sys.exit(main())
