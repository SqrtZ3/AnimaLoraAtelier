"""Kaggle TPU v5e-8 · **网格宽度做编译身份**：能不能同时拿到块对角的速度和
全通的单一编译身份。

## 上一轮（anima-mask-probe）钉死的事实

  块对角  4x4096 布局 4505ms（有效 MFU 18.2%）
  全通    同布局    11508ms（MFU 11.2%）
  一轮总时长比（真实直方图加权）**1.216x**，定死 S_max 后 **1.275x**

也就是说全通路线不只多算 FLOPs，单位算力效率也更低 —— 27% 的税太贵。
但块对角把**段长元组**变成了编译身份，稀有布局凑不满 8 个 pack 会被欠采样
（真实数据集 27~39 张图）。

## 本轮的想法

splash 源码 `splash_attention_kernel.py:1137`：

    grid_width = fwd_mask_info.data_next.shape[-1]

跳块收益来自**网格宽度**，而它只是一个**上界**，不是等价类。于是：
host 侧照常 process_mask 拿到收缩后的 MaskInfo，把 data_next/mask_next/block_mask
沿最后一维补齐到菜单里的 W（补出来的格子 block_mask=0 -> should_run=False，
纯空转），再把 MaskInfo 作为**运行时参数**传进步函数。

编译身份 = W（菜单里的几个整数），且任何 pack 都能向上取整到共用的 W
-> 一步的 8 个 pack 永远凑得齐 -> **孤儿欠采样消失**。

本地 CPU interpret 已验：补齐后输出与官方 make_splash_mha **逐 bit 相同**
（max_abs=0.000e+00），同 W 换段几何 trace 次数=1。本轮上真机验三件事。

## 本轮要回答的

R0 设备 / HBM。
R1 **闸门**：菜单后端 ≡ 块对角，含"原生 W=64 补齐到 W=128"这一档
   （空转格子若改了数值，这里必须炸）。走 scan 路径，避免上一轮 R1 的 793 秒。
R2 **主判据**：菜单原生 W 能否追平块对角（验证 MaskInfo 走参数不带额外开销），
   以及补齐到 W=128 的真实代价；据此算菜单 [32,128] 的一轮总时长。
R3 同 W 下换段几何是否真的零重编译（对照块对角每次都要编）。

## 不在本轮范围

优化器 / 数据 / checkpoint 不接（随机权重 + MSE 到随机 target）。
cross-attn 仍走块对角（文本槽 512 不是 1024 的倍数，用不了菜单后端的固定反向块；
且上一轮 R4 实测 cross 的 kv 从 512 撑到 2048 才 1.068x，占比小）。
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


# ── 本轮布局：真实数据集 your-dataset（budget 16384、Q=1024）───────────────
#
#     x59  [16384]              real [16320]                原生 W=128（无块可跳）
#     x19  [4096,4096,4096,4096] real [4000,4080,4018,4048]  原生 W=32
#     x4   [8192, 8192]         real [8040, 8040]            原生 W=64
#     x2   [9216, 4096, 3072]   real [9196, 4080, 2924]      原生 W=72
#
# 菜单 `[32, 128]`：4x4096 用 W=32，其余向上取整到 W=128。**2 个编译身份。**
LAYOUTS = [
    ("单图 [16384]",         [16384],               [16320],                  59),
    ("4x4096",               [4096] * 4,            [4000, 4080, 4018, 4048], 19),
    ("2x8192",               [8192, 8192],          [8040, 8040],              4),
    ("混装 [9216,4096,3072]", [9216, 4096, 3072],   [9196, 4080, 2924],        2),
]
MENU = [32, 128]
BUDGET = 16384
# 上一轮 anima-mask-probe 的真机基准（同口径：8 卡 DP、scan+full、budget 16384）
BASELINE_MS = {"单图 [16384]": (10518, 10515), "4x4096": (4505, 11508),
               "2x8192": (6594, 11034), "混装 [9216,4096,3072]": (6236, 11275)}


def build_loss(cfg, coarse, pack, remat, backend, width=None):
    """backend: "bd" 块对角（编译期 mask）/ "menu" 网格宽度菜单（MaskInfo 走参数）。

    menu 路径返回 (loss_fn, mask_info 字典)：MaskInfo **必须当参数传进步函数**，
    闭包捕获会被烘成编译期常量，就退回每布局一次编译了。
    """
    import jax.numpy as jnp
    n_seg, L = len(coarse), sum(coarse)
    txt = [T_TXT] * n_seg
    seg_txt = jnp.asarray(segment_ids(txt))

    if backend == "bd":
        self_raw = make_splash_attn(coarse, coarse, cfg.num_heads, cfg.head_dim)
        cross_raw = make_splash_attn(coarse, txt, cfg.num_heads, cfg.head_dim)

        def loss_fn(lora, params, batch):
            sf = bind_segments(self_raw, pack["seg_self"], pack["seg_self"])
            cf = bind_segments(cross_raw, pack["seg_cross"], seg_txt)
            return _loss(cfg, lora, params, batch, pack, sf, cf, remat)
        return loss_fn, {}

    w = width or pick_menu_width(coarse, MENU)
    sfwd, sdkv = menu_mask_info(coarse, coarse, cfg.num_heads, w)
    # cross 的 q 是图像段、kv 是文本段；文本槽 512 不是 1024 的倍数，用不了
    # 菜单后端的固定反向块 -> cross 仍走块对角（它的编译身份只是段数，不是段长元组，
    # 且真机实测 cross 只占 ~7%（anima-mask-probe R4：kv 512->2048 才 1.068x））。
    cross_raw = make_splash_attn(coarse, txt, cfg.num_heads, cfg.head_dim)
    attn = make_menu_attn(cfg.num_heads, cfg.head_dim)
    mi = {"fwd": sfwd, "dkv": sdkv}

    def loss_fn(lora, params, batch):
        sf = bind_menu(attn, pack["seg_self"], pack["seg_self"],
                       batch["mi_fwd"], batch["mi_dkv"])
        cf = bind_segments(cross_raw, pack["seg_cross"], seg_txt)
        return _loss(cfg, lora, params, batch, pack, sf, cf, remat)
    return loss_fn, mi


def _loss(cfg, lora, params, batch, pack, self_fn, cross_fn, remat):
    import jax.numpy as jnp
    out = forward_packed(params, cfg, batch["tok"], batch["t"], batch["ctx"],
                         pack["rows"], pack["cols"], pack["mod_index"],
                         self_fn, cross_fn, loras=lora, remat=remat)
    se = jnp.mean((out.astype(jnp.float32) - batch["target"].astype(jnp.float32)) ** 2,
                  axis=-1)
    m = pack["loss_mask"]
    return jnp.sum(se * m) / jnp.maximum(jnp.sum(m), 1.0)


def _run(backend, coarse, real, width=None, reps=3, warmup=1):
    """8 卡 DP 一步（前向 + LoRA 反向 + all-reduce）。与 anima-mask-probe 同口径。"""
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
    pack = build_pack(coarse, real, jnp)
    params = jax.jit(lambda: make_random_params(jax.random.PRNGKey(0), cfg, dtype, True),
                     out_shardings=NamedSharding(mesh, P()))()
    flat = init_lora(jax.random.PRNGKey(1), cfg, LORA_RANK, dtype=dtype)
    lora = jax.jit(lambda: stack_loras(flat, cfg.num_blocks),
                   out_shardings=NamedSharding(mesh, P()))()
    batch = make_batch(jax.random.PRNGKey(2), cfg, n_dev, (L,), (n_seg * T_TXT,),
                       n_seg, dtype, mesh)
    local, mi = build_loss(cfg, coarse, pack, "full", backend, width)
    if mi:
        # MaskInfo 复制到所有卡（in_spec P()），作为**参数**进入 jit
        rep = NamedSharding(mesh, P())
        batch["mi_fwd"] = jax.device_put(jax.tree.map(jnp.asarray, mi["fwd"]), rep)
        batch["mi_dkv"] = jax.device_put(jax.tree.map(jnp.asarray, mi["dkv"]), rep)

    def per_shard(lo, pa, ba, mif, mid):
        b = {k: v[0] for k, v in ba.items()}
        b["mi_fwd"], b["mi_dkv"] = mif, mid
        return local(lo, pa, b)[None]

    dspec = {k: P("d") for k in ("tok", "t", "ctx", "target")}
    dev_batch = {k: batch[k] for k in dspec}
    f = _smap(per_shard, mesh, (P(), P(), dspec, P(), P()), P("d"))
    grad = jax.jit(jax.grad(
        lambda lo, pa, ba, mif, mid: jnp.mean(f(lo, pa, ba, mif, mid)), argnums=0))
    args = (lora, params, dev_batch,
            batch.get("mi_fwd", ()), batch.get("mi_dkv", ()))
    ms, first = _bench(lambda: grad(*args), reps=reps, warmup=warmup)
    attn = sum(c * c for c in coarse)
    return {"ms": ms, "first": first, "peak": _hbm(True),
            "tok_s": n_dev * sum(real) / (ms / 1e3),
            "mfu": useful_mfu(cfg, ms, n_dev, L, attn, L * n_seg * T_TXT)}


# ── 探测 ──────────────────────────────────────────────────────────────────────
@probe("R0 设备 / HBM")
def r0():
    import jax
    d = jax.devices()
    lim = (d[0].memory_stats() or {}).get("bytes_limit", 0) / 1024 ** 3
    _reset_peak()
    return (f"jax {jax.__version__} | {len(d)} x {d[0].device_kind} | "
            f"单设备 HBM limit={lim:.1f}GiB | peak 可清零={_hbm(True) == 0}")


@probe("R1 【闸门】菜单后端 ≡ 块对角（真机、含补齐到更大的 W）")
def r1():
    """本地 interpret 已验逐 bit 相同；这里在真机上复验，并且**特意用一个原生
    W=64 的布局补齐到 W=128** —— 补出来的空转格子如果改了数值，这里必须炸。
    走 scan 路径（stacked 权重），避免上一轮 R1 展开编译吃掉 793 秒。"""
    import jax
    import jax.numpy as jnp
    _cleanup()
    cfg, dtype = AnimaConfig(), jnp.bfloat16
    coarse, real = LAYOUTS[2][1], LAYOUTS[2][2]          # 2x8192，原生 W=64
    L, n_seg = sum(coarse), len(coarse)
    pack = build_pack(coarse, real, jnp)
    flat = init_lora(jax.random.PRNGKey(1), cfg, LORA_RANK, dtype=dtype)
    lora = stack_loras(flat, cfg.num_blocks)
    params = make_random_params(jax.random.PRNGKey(0), cfg, dtype, True)
    k = jax.random.split(jax.random.PRNGKey(2), 3)
    batch = {"tok": jax.random.normal(k[0], (L, cfg.in_dim), dtype),
             "t": jax.random.uniform(k[1], (n_seg,), jnp.float32),
             "ctx": jax.random.normal(k[2], (n_seg * T_TXT, cfg.crossattn_dim), dtype),
             "target": jnp.zeros((L, cfg.out_dim), dtype)}

    def out_of(backend, width=None):
        local, mi = build_loss(cfg, coarse, pack, "full", backend, width)
        b = dict(batch)
        if mi:
            b["mi_fwd"] = jax.tree.map(jnp.asarray, mi["fwd"])
            b["mi_dkv"] = jax.tree.map(jnp.asarray, mi["dkv"])
        return np.asarray(jax.jit(lambda bb: local(lora, params, bb))(b), np.float64)

    bd = out_of("bd")
    native = out_of("menu", 64)
    padded = out_of("menu", 128)
    d1, d2 = abs(bd - native), abs(bd - padded)
    if d2 > 1e-9:
        raise RuntimeError(f"补齐到 W=128 后 loss 变了（Δ={d2:.3e}）—— 空转格子不干净")
    return (f"loss 块对角={bd:.9f} | 菜单W=64 Δ={d1:.1e} | 菜单W=128(补齐) "
            f"Δ={d2:.1e}（两者都必须 0）")


@probe("R2 【主判据】菜单 vs 块对角 vs 全通：真实布局的步时")
def r2():
    """三列：块对角（基准，上一轮实测）/ 菜单原生 W / 菜单补到 W=128。
    菜单原生 W 应当追平块对角（验证"MaskInfo 走参数"不带来额外开销）；
    菜单补齐档给出"统一到一个身份"的真实代价。"""
    rows = []
    for name, coarse, real, cnt in LAYOUTS:
        wn = menu_grid_width(coarse)
        got = {}
        for tag, w in (("菜单W%d" % wn, wn),
                       *(() if wn == 128 else (("菜单W128", 128),))):
            try:
                got[tag] = _run("menu", coarse, real, w)
            except _Skip:
                raise
            except Exception as e:
                got[tag] = {"err": f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"}
        bd_ms, full_ms = BASELINE_MS[name]
        line = (f"块对角{bd_ms}ms(上轮) 全通{full_ms}ms(上轮)  |  " + "  ".join(
            f"{t}: " + (r["err"] if "err" in r else
                        f"{r['ms']:.0f}ms MFU {r['mfu']:.1%} 峰值{r['peak']:.1f}GiB "
                        f"首调{r['first']:.0f}s（/块对角 {r['ms'] / bd_ms:.3f}x）")
            for t, r in got.items()))
        record("  " + name + f"（{cnt}/93，原生W={wn}）", "DATA", line)
        rows.append((cnt, bd_ms, full_ms, got, wn))

    # 菜单 [32,128] 的一轮总时长：4x4096 用原生 W=32，其余用 W=128
    tot_bd = tot_full = tot_menu = 0
    miss = 0
    for cnt, bd_ms, full_ms, got, wn in rows:
        tag = "菜单W%d" % wn if wn in (32, 128) else "菜单W128"
        r = got.get(tag) or {}
        if "ms" not in r:
            miss += 1
            continue
        tot_bd += cnt * bd_ms
        tot_full += cnt * full_ms
        tot_menu += cnt * r["ms"]
    if not tot_bd:
        return "全部失败，见上方 DATA"
    return (f"菜单{MENU}（2 个编译身份）一轮 {tot_menu / 1e3:.0f}s | "
            f"块对角(11~18 身份) {tot_bd / 1e3:.0f}s | 全通(1 身份) {tot_full / 1e3:.0f}s "
            f"-> 菜单/块对角 **{tot_menu / tot_bd:.3f}x**，菜单/全通 "
            f"{tot_menu / tot_full:.3f}x（缺 {miss} 档）")


@probe("R3 同 W 下换段几何：真机零重编译")
def r3():
    """菜单方案的核心卖点。判据：第二个布局的首调时间应当远小于第一个
    （只剩数据搬运，没有编译）。对照：块对角换布局每次都要重编译。"""
    a = _run("menu", [8192, 8192], [8040, 8040], 128, reps=1, warmup=0)
    b = _run("menu", [9216, 4096, 3072], [9196, 4080, 2924], 128, reps=1, warmup=0)
    c = _run("menu", [16384], [16320], 128, reps=1, warmup=0)
    _cleanup(drop_kernels=True)
    d = _run("bd", [8192, 8192], [8040, 8040], reps=1, warmup=0)
    e = _run("bd", [9216, 4096, 3072], [9196, 4080, 2924], reps=1, warmup=0)
    ok = b["first"] < a["first"] / 3
    return (f"菜单 W=128：2x8192 首调 {a['first']:.0f}s -> 混装 {b['first']:.0f}s -> "
            f"单图 {c['first']:.0f}s（零重编译={'是' if ok else '**否**'}）；"
            f"对照 块对角：2x8192 {d['first']:.0f}s -> 混装 {e['first']:.0f}s")


def main():
    print(f"[ INFO ] python - {platform.python_version()} @ {sys.executable}", flush=True)
    print(f"[ INFO ] bootstrap - {_BOOT}", flush=True)
    print(f"[ INFO ] 口径 - T_TXT={T_TXT} rank={LORA_RANK} budget={BUDGET} "
          f"菜单={MENU} 有效MFU分母={V5E_PEAK_TFLOPS}TFLOPS/chip x8", flush=True)
    for fn in (r0, r1, r2, r3):
        fn()
    ok = sum(1 for r in RESULTS if r["status"] == "OK")
    bad = sum(1 for r in RESULTS if r["status"] == "FAIL")
    print(f"[ INFO ] 汇总 - OK={ok} FAIL={bad} 总耗时={time.time() - _T0:.0f}s", flush=True)
    _flush()
    return 0            # 单项 FAIL 是数据，不是脚本故障


if __name__ == "__main__":
    sys.exit(main())
