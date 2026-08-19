"""Kaggle TPU v5e-8 · **单编译身份的 NaViT**：全通静态 mask + 运行时 segment_ids
到底比块对角慢多少。

## 为什么问这一个问题

前四轮把块对角 splash 打通了（真机 T1 跳块比 0.252 / 理论 0.250），但它把
**段长元组变成了编译身份**，由此长出一整串并发症：每种布局编一次全模型、
一步的 8 个 pack 必须同布局、稀有布局凑不满 8 个就被系统性欠采样（真实数据集
上 27~39 张图落进这类布局）、MaskInfo 要自己按布局缓存、反向块大小受最短段限制。

NaViT 原文（Patch n' Pack, arXiv:2307.06304）用的**不是**块稀疏内核：§2.1 是
自注意力 mask，§2.3 专门论证隐藏维越宽、注意力占总算力比例越小，因此打包的
额外开销可以接受。若照原文做法——静态 mask 恒全通、段边界只由运行时
segment_ids 决定——编译身份退化成 (q_len, kv_len) 两个整数，上面那串并发症
**全部消失**，代价是不跳块。

本地按真实数据集 your-dataset 的布局直方图用 FLOPs 口径算出这笔账是
**1.13x**（含固定 S_max 的 cross 是 1.18x）。之所以这么小，是因为 93 个 pack
里 59 个只装得下一张 16384 的图，那些 pack 本来就无块可跳。

**FLOPs 口径不等于墙钟时间**（内核效率与形状有关，两条路径的反向块大小也不同），
所以本轮上真机把它钉死。

## 本轮要回答的

R0 设备 / HBM。
R1 **闸门**：全通+segment_ids 的前向输出 ≡ 块对角（同一 pack、真内核）。
   带破坏对照（把 segment_ids 抹平），确认这条判据测得出东西。
R2 **主判据**：四种真实布局 x 两个后端的步时 / 真tok/s / 有效MFU / 峰值，
   并按真实直方图加权，给出"实测 Nx vs 本地理论 1.13x"。
R3 换段组成的编译代价：全通应当只编一次，块对角每种布局都要编。
R4 固定 S_max 的 cross-attn 代价：单图方案要形状恒定，kv 只能定死成 4x512。

## 不在本轮范围

优化器 / 数据 / checkpoint 不接（随机权重 + MSE 到随机 target）；形状一致则
性能一致，故不上传 3.91GB 权重。remat 只跑 `full`（scan 路径下唯一在真机上
跑通过的档），与 anima_layout R4 同口径，数字可跨轮比。
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

def _flush():
    try:
        with open(os.path.join(OUT_DIR, "anima_layout.json"), "w", encoding="utf-8") as f:
            json.dump(RESULTS, f, ensure_ascii=False, indent=2)
        with open(os.path.join(OUT_DIR, "anima_layout_report.txt"), "w", encoding="utf-8") as f:
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
        for c in (globals().get("_splash_kernel"), globals().get("_bucket_kernel"),
                  globals().get("_full_kernel")):
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


def _gcd_all(xs):
    """段长的最大公约数 = 能用的最大 chunk（更大就会跨图 -> 调制静默用错 t）。"""
    import math
    g = 0
    for x in xs:
        g = math.gcd(g, int(x))
    return g


def build_packed_loss(cfg, coarse, pack, remat, chunk=None, barrier=False,
                      interpret=False):
    """打包路线的 loss_fn(lora, params, batch)：冻结底模、只对 LoRA 求梯度。

    `chunk` 见 `forward_packed`：把 pack 按 Q 切块后 AdaLN 调制走广播而不是逐 token
    gather。注意力、块对角 mask、段几何、splash 内核**一律不变**，数值上与朴素路径
    逐 bit 相同（本地 tests/check_ragged_equiv.py ④：chunk=32/64/128 全部 max_abs=0）。
    """
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
                             self_fn, cross_fn, loras=lora, remat=remat,
                             chunk=chunk, barrier=barrier)
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

# ── 本轮布局：全部取自真实数据集 your-dataset ────────────────────────────
#
# 本地枚举（tests/enum_dataset_routes.py，budget=16384 Q=1024，样本池 165 条
# = 85 张原生 + 80 条 4096 档 multiscale 副本）产出 93 个 pack，直方图：
#
#     x59  [16384]                 real [16320]                 Sn²/L² = 1.000
#     x19  [4096, 4096, 4096, 4096] real [4000,4080,4018,4048]  Sn²/L² = 0.250
#     x4   [8192, 8192]            real [8040, 8040]            Sn²/L² = 0.500
#     x2   [9216, 4096, 3072]      real [9196, 4080, 2924]      Sn²/L² = 0.414
#     ...  其余 9 个 pack 是只出现一两次的长尾
#
# **59/93 的 pack 只装得下一张 16384 的图 —— 那些 pack 本来就无块可跳。**
# 这正是本地 FLOPs 口径算出"全通/块对角只有 1.13x"的原因。本轮就是要把这个
# 推算钉成真机步时。
LAYOUTS = [
    ("单图 [16384]",        [16384],                 [16320],                  59),
    ("4x4096",              [4096] * 4,              [4000, 4080, 4018, 4048], 19),
    ("2x8192",              [8192, 8192],            [8040, 8040],              4),
    ("混装 [9216,4096,3072]", [9216, 4096, 3072],    [9196, 4080, 2924],        2),
]
S_MAX = 4          # 单图方案要把 cross 的 kv 长度定死；取真实布局的最大段数
BUDGET = 16384


def build_packed_loss2(cfg, coarse, pack, remat, backend, cross_segs=None,
                       interpret=False):
    """与 anima_layout 的 build_packed_loss 同构，只多一个 `backend` 开关。

    backend="bd"   段几何进静态 mask（块对角，跳块，段长元组进编译身份）
    backend="full" 静态 mask 全通，段边界全交给运行时 segment_ids（单编译身份）

    两条路径的 segment_ids **完全相同** —— 这是 A/B 干净的前提：唯一变量是静态
    mask。cross_segs 给定时把 cross 的 kv 定死成 S_MAX 个文本槽（单图方案要
    形状恒定），否则用实际段数。
    """
    import jax.numpy as jnp
    n_seg = len(coarse)
    kv_segs = n_seg if cross_segs is None else cross_segs
    txt = [T_TXT] * kv_segs
    seg_txt = jnp.asarray(segment_ids(txt))

    if backend == "bd":
        if cross_segs is not None and cross_segs != n_seg:
            raise ValueError("块对角后端要求 q/kv 段数一一对应，不能定死 S_MAX")
        self_raw = make_splash_attn(coarse, coarse, cfg.num_heads, cfg.head_dim,
                                    interpret=interpret)
        cross_raw = make_splash_attn(coarse, txt, cfg.num_heads, cfg.head_dim,
                                     interpret=interpret)
    elif backend == "full":
        L = sum(coarse)
        self_raw = make_full_attn(L, L, cfg.num_heads, cfg.head_dim,
                                  interpret=interpret)
        cross_raw = make_full_attn(L, kv_segs * T_TXT, cfg.num_heads,
                                   cfg.head_dim, interpret=interpret)
    else:
        raise ValueError(f"未知 backend {backend}")

    self_fn = bind_segments(self_raw, pack["seg_self"], pack["seg_self"])
    cross_fn = bind_segments(cross_raw, pack["seg_cross"], seg_txt)

    def loss_fn(lora, params, batch):
        out = forward_packed(params, cfg, batch["tok"], batch["t"], batch["ctx"],
                             pack["rows"], pack["cols"], pack["mod_index"],
                             self_fn, cross_fn, loras=lora, remat=remat)
        se = jnp.mean((out.astype(jnp.float32) - batch["target"].astype(jnp.float32)) ** 2,
                      axis=-1)
        m = pack["loss_mask"]
        return jnp.sum(se * m) / jnp.maximum(jnp.sum(m), 1.0)

    return loss_fn


def _theory_ratio(coarse, cfg, cross_kv_bd, cross_kv_full):
    """本地 FLOPs 口径的全通/块对角比（真机步时该向它靠）。"""
    L = sum(coarse)
    bd = fwd_flops(cfg, L, sum(c * c for c in coarse), L * cross_kv_bd)
    fu = fwd_flops(cfg, L, L * L, L * cross_kv_full)
    return fu / bd


# ── 探测 ──────────────────────────────────────────────────────────────────────
@probe("R0 设备 / HBM")
def r0():
    import jax
    d = jax.devices()
    lim = (d[0].memory_stats() or {}).get("bytes_limit", 0) / 1024 ** 3
    _reset_peak()
    return (f"jax {jax.__version__} | {len(d)} x {d[0].device_kind} | "
            f"单设备 HBM limit={lim:.1f}GiB | peak 可清零={_hbm(True) == 0}")


@probe("R1 【闸门】全通+segment_ids ≡ 块对角（同一 pack、真内核）")
def r1():
    """静态 mask 从块对角换成全通之后，**输出必须不变** —— 段边界由运行时
    segment_ids 保证（splash 的契约：静态 mask 与 segment id mask 取 AND）。

    判据带对照：把全通路径的 segment_ids 换成全 0（即一个 pack 里所有图互相
    看得见），污染量级必须显著大于 A/B 差，否则说明这条判据根本测不出东西。
    """
    import jax
    import jax.numpy as jnp
    _cleanup()
    cfg, dtype = AnimaConfig(), jnp.bfloat16
    coarse, real = LAYOUTS[1][1], LAYOUTS[1][2]      # 4x4096，段最多、最能暴露问题
    L = sum(coarse)
    pack = build_pack(coarse, real, jnp)
    key = jax.random.PRNGKey(0)
    params = make_random_params(key, cfg, dtype, stack=False)
    lora = init_lora(jax.random.PRNGKey(1), cfg, LORA_RANK, dtype=dtype)
    k = jax.random.split(jax.random.PRNGKey(2), 3)
    tok = jax.random.normal(k[0], (L, cfg.in_dim), dtype)
    t = jax.random.uniform(k[1], (len(coarse),), jnp.float32)
    ctx = jax.random.normal(k[2], (len(coarse) * T_TXT, cfg.crossattn_dim), dtype)

    def run(backend, break_segments=False):
        p = dict(pack)
        if break_segments:
            p["seg_self"] = jnp.zeros_like(pack["seg_self"])
        loss = build_packed_loss2(cfg, coarse, p, "none", backend)
        # 直接取前向输出而不是 loss —— 逐 token 比更严
        n_seg = len(coarse)
        txt = [T_TXT] * n_seg
        raw = (make_splash_attn(coarse, coarse, cfg.num_heads, cfg.head_dim)
               if backend == "bd" else
               make_full_attn(L, L, cfg.num_heads, cfg.head_dim))
        craw = (make_splash_attn(coarse, txt, cfg.num_heads, cfg.head_dim)
                if backend == "bd" else
                make_full_attn(L, n_seg * T_TXT, cfg.num_heads, cfg.head_dim))
        sf = bind_segments(raw, p["seg_self"], p["seg_self"])
        cf = bind_segments(craw, p["seg_cross"], jnp.asarray(segment_ids(txt)))
        return jax.jit(lambda: forward_packed(
            params, cfg, tok, t, ctx, pack["rows"], pack["cols"],
            pack["mod_index"], sf, cf, loras=lora, remat="none"))()

    m = np.asarray(pack["loss_mask"]) > 0
    a = np.asarray(run("bd"), np.float32)[m]
    b = np.asarray(run("full"), np.float32)[m]
    c = np.asarray(run("full", break_segments=True), np.float32)[m]
    diff = float(np.abs(a - b).max())
    rel = diff / max(float(np.abs(a).max()), 1e-9)
    ctrl = float(np.abs(a - c).max())
    if not (diff < ctrl / 10):
        raise RuntimeError(f"A/B 差 {diff:.3e} 未显著小于破坏对照 {ctrl:.3e} "
                           f"—— 全通路径的 segment_ids 可能没生效")
    return (f"真 token 上 max_abs={diff:.3e} rel={rel:.3e}（bf16 噪声量级）；"
            f"破坏 segment_ids 的对照污染={ctrl:.3e}（判据有效）")


@probe("R2 【主判据】全通 vs 块对角：真实布局的步时 / 峰值")
def r2():
    """8 卡 DP 一步（前向 + LoRA 反向 + 跨卡 all-reduce），scan 布局 + full remat
    —— 与 anima_layout R4 同口径，数字可直接跨轮比。

    每个布局报：两个后端的步时、实测比、以及本地 FLOPs 口径的理论比。
    """
    cfg = AnimaConfig()
    rows = []
    for name, coarse, real, cnt in LAYOUTS:
        got = {}
        for backend in ("bd", "full"):
            try:
                got[backend] = _run_masked(backend, coarse, real, "full")
            except _Skip:
                raise
            except Exception as e:
                got[backend] = {"err": f"{type(e).__name__}: {str(e).splitlines()[0]}"}
        line = f"{name}（真实占比 {cnt}/93）"
        for backend in ("bd", "full"):
            r = got[backend]
            tag = "块对角" if backend == "bd" else "全通  "
            line += ("\n      " + tag + " " +
                     (r["err"] if "err" in r else
                      f"{r['ms']:.0f}ms {r['tok_s'] / 1e3:.1f}k真tok/s "
                      f"有效MFU {r['mfu']:.1%} 峰值{r['peak']:.1f}GiB 首调{r['first']:.0f}s"))
        if all("err" not in got[b] for b in ("bd", "full")):
            meas = got["full"]["ms"] / got["bd"]["ms"]
            th = _theory_ratio(coarse, cfg, len(coarse) * T_TXT, len(coarse) * T_TXT)
            line += f"\n      -> 实测比 {meas:.3f}x   理论(FLOPs) {th:.3f}x"
            rows.append((cnt, got["bd"]["ms"], got["full"]["ms"], th))
        record("  " + name, "DATA", line.split("\n", 1)[1].strip())
    if rows:
        # **按总时长比，不是按比值的均值。** 一轮的墙钟 = Σ(pack 数 x 步时)，
        # 所以要先各自加权求和再相除。取比值的均值会高估 —— 它给"快但少见"的
        # 布局与"慢但常见"的布局同样的话语权（第一版就是这么写的，把 1.216x
        # 报成了 1.403x）。
        w = sum(c for c, _, _, _ in rows)
        tb = sum(c * b for c, b, _, _ in rows)
        tf = sum(c * f for c, _, f, _ in rows)
        wt = sum(c * t for c, _, _, t in rows) / w
        return (f"按真实直方图加权（覆盖 {w}/93 个 pack，总时长比）："
                f"**实测 {tf / tb:.3f}x** vs 本地理论(比值均值) {wt:.3f}x；"
                f"块对角一轮 {tb / 1e3:.0f}s vs 全通 {tf / 1e3:.0f}s")
    return "全部布局都失败，见上方 DATA 条目"


@probe("R3 换段组成的编译代价：全通应当零重编译")
def r3():
    """单图方案的核心卖点：编译身份只有 (q_len, kv_len)，换任何段组成都不重编译。
    块对角则每种段长元组编一次全模型。

    做法：同 budget 下依次跑 4 种段组成，记录每次的首调时间。全通路径除第一次外
    应当接近 0；块对角每次都应当是完整的编译时间。
    """
    out = []
    for backend in ("bd", "full"):
        firsts = []
        _cleanup(drop_kernels=True)
        for name, coarse, real, _ in LAYOUTS:
            try:
                r = _run_masked(backend, coarse, real, "full", reps=1, warmup=0)
                firsts.append(f"{name.split()[0]} {r['first']:.0f}s")
            except Exception as e:
                firsts.append(f"{name.split()[0]} {type(e).__name__}")
        tag = "块对角" if backend == "bd" else "全通"
        out.append(f"{tag}: " + " | ".join(firsts))
    return "；".join(out)


@probe("R4 固定 S_max 的 cross-attn 额外代价（单图方案要形状恒定）")
def r4():
    """单图方案下 cross 的 kv 长度也必须恒定，只能取 S_max*512（本数据集 S_max=4）。
    段数少于 S_max 的 pack 要为空文本槽白付一点算力 —— 这里量它。

    59/93 的 pack 只有 1 段（kv=512），定死成 4 段就是 4 倍 cross kv，
    所以拿单图布局测最坏情况。
    """
    name, coarse, real, _ = LAYOUTS[0]           # [16384]，n_seg=1，最坏
    got = {}
    for tag, cs in (("实际 kv=512", None), (f"定死 kv={S_MAX * T_TXT}", S_MAX)):
        try:
            got[tag] = _run_masked("full", coarse, real, "full", cross_segs=cs)
        except Exception as e:
            got[tag] = {"err": f"{type(e).__name__}: {str(e).splitlines()[0]}"}
    line = " | ".join(
        f"{k}: " + (v["err"] if "err" in v else f"{v['ms']:.0f}ms")
        for k, v in got.items())
    if all("err" not in v for v in got.values()):
        ks = list(got)
        line += f"  -> 定死的代价 {got[ks[1]]['ms'] / got[ks[0]]['ms']:.3f}x"
    return line


def _run_masked(backend, coarse, real, remat, cross_segs=None, reps=3, warmup=1):
    """跑一次 8 卡 DP 训练步。与 anima_layout 的 _run_step 同结构，
    只把 loss 构造换成带 backend 开关的版本。"""
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
    L, n_seg = sum(coarse), len(coarse)
    kv_segs = n_seg if cross_segs is None else cross_segs
    pack = build_pack(coarse, real, jnp)

    params = jax.jit(lambda: make_random_params(jax.random.PRNGKey(0), cfg, dtype, True),
                     out_shardings=NamedSharding(mesh, P()))()
    flat = init_lora(jax.random.PRNGKey(1), cfg, LORA_RANK, dtype=dtype)
    lora = jax.jit(lambda: stack_loras(flat, cfg.num_blocks),
                   out_shardings=NamedSharding(mesh, P()))()
    batch = make_batch(jax.random.PRNGKey(2), cfg, n_dev, (L,),
                       (kv_segs * T_TXT,), n_seg, dtype, mesh)
    local = build_packed_loss2(cfg, coarse, pack, remat, backend, cross_segs)

    def per_shard(lo, pa, ba):
        b = {k: v[0] for k, v in ba.items()}
        return local(lo, pa, b)[None]

    dspec = {k: P("d") for k in ("tok", "t", "ctx", "target")}
    f = _smap(per_shard, mesh, (P(), P(), dspec), P("d"))
    grad = jax.jit(jax.grad(lambda lo, pa, ba: jnp.mean(f(lo, pa, ba)), argnums=0))
    ms, first = _bench(lambda: grad(lora, params, batch), reps=reps, warmup=warmup)
    attn = sum(c * c for c in coarse) if backend == "bd" else L * L
    cross = L * kv_segs * T_TXT
    return {"ms": ms, "first": first, "peak": _hbm(True),
            "tok_s": n_dev * sum(real) / (ms / 1e3),
            "mfu": useful_mfu(cfg, ms, n_dev, L, attn, cross)}


def main():
    print(f"[ INFO ] python - {platform.python_version()} @ {sys.executable}", flush=True)
    print(f"[ INFO ] bootstrap - {_BOOT}", flush=True)
    print(f"[ INFO ] 口径 - T_TXT={T_TXT} rank={LORA_RANK} budget={BUDGET} "
          f"有效MFU分母={V5E_PEAK_TFLOPS}TFLOPS/chip x8", flush=True)
    for fn in (r0, r1, r2, r3, r4):
        fn()
    ok = sum(1 for r in RESULTS if r["status"] == "OK")
    bad = sum(1 for r in RESULTS if r["status"] == "FAIL")
    print(f"[ INFO ] 汇总 - OK={ok} FAIL={bad} 总耗时={time.time() - _T0:.0f}s",
          flush=True)
    _flush()
    return 0            # 单项 FAIL 是数据，不是脚本故障（Kaggle 把非零退出判 ERROR）


if __name__ == "__main__":
    sys.exit(main())
