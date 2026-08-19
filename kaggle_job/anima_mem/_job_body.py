"""Kaggle TPU v5e-8 · NaViT 路径的显存归因（单变量二分）。

## 要回答什么

上一跑（anima-navit-probe）实测：NaViT 打包路径的显存 ≈ **4.0GB + 1.01 MB/token**
（由 16384/full 需 20.60G、32768/full 需 37.18G 两点拟合，8192 能跑且峰值 3.9GiB）。
而分桶批处理路径在 49152 token/chip 时总峰值才 4.9GiB —— 折合 0.02 MB/token，
**差约 50 倍**。

按结构算，full remat 下每块只需存 [B, D] 的块输入 = 4KB/token x 28 = 0.112 MB/token。
实测是它的 9 倍。所以有东西多占了，且是 NaViT 路径特有。**本 job 不猜，逐项关掉测。**

## 方法：OOM 也是数据

XLA 的 RESOURCE_EXHAUSTED 消息里带
`total memory required for HLO temporaries (X G)` —— 即使配置跑不起来，
这个数字也是可比的测量值。所以固定在 **budget=16384 / remat=full**（基线正好
OOM 且需求 20.60G），逐个关掉嫌疑项，看需求降到多少。降得多的那个就是主因。

## 变量表

  V0 基线                      —— 复现 20.60G
  V1 mod_index 恒 0            —— gather 退化成广播（**数值不对**，只测显存）
  V2 关 cross-attn             —— cross_fn 直接返回 0
  V3 反向块 128（而非 1024）   —— arch_probe H1 选的 1024 是为速度，代价未测
  V4 关 AdaLN 调制             —— shift=0/scale=0/gate=1
  V5 只前向不求梯度            —— 分出"前向本身"与"反向残差"的占比

V1/V2/V4 都会让结果数值上错，这是**故意的**：本 job 只测显存，不看 loss。
"""

import json
import os
import platform
import re
import sys
import tempfile
import time
import traceback

import numpy as np

RESULTS: list = []
_T0 = time.time()
_BOOT = globals().get("_BOOT", "未经 build_job.py 拼接，无 bootstrap")
OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else tempfile.gettempdir()

T_TXT = 256
LORA_RANK = 32
# 基线：上一跑实测需 20.60G（正好 OOM，是理想的对照点——降多少一目了然）
BUDGET = 16384
COARSE = [10240, 4096, 2048]
REAL = [10080, 4096, 0]


def _flush():
    try:
        with open(os.path.join(OUT_DIR, "anima_mem.json"), "w", encoding="utf-8") as f:
            json.dump(RESULTS, f, ensure_ascii=False, indent=2)
        with open(os.path.join(OUT_DIR, "anima_mem_report.txt"), "w", encoding="utf-8") as f:
            for r in RESULTS:
                f.write(f"[{r['status']:^6}] {r['name']}"
                        + (f" - {r['detail']}" if r["detail"] else "") + "\n")
    except OSError:
        pass


def record(name, status, detail=""):
    RESULTS.append({"name": name, "status": status, "detail": detail,
                    "t_elapsed_s": round(time.time() - _T0, 1)})
    print(f"[{status:^6}] {name}" + (f" - {detail}" if detail else ""), flush=True)
    _flush()


def probe(name):
    def deco(fn):
        def run(*a, **kw):
            try:
                record(name, "OK", fn(*a, **kw) or "")
                return True
            except Exception as e:
                tb = traceback.format_exc(limit=3).strip().splitlines()[-1]
                record(name, "FAIL", f"{type(e).__name__}: {e} | {tb}")
            return False
        return run
    return deco


_RE_MEM = re.compile(r"HLO temporaries \(([\d.]+)G\)")


def mem_required(build_and_run):
    """跑一次；返回 (状态, 需求GB, 步时ms)。

    **OOM 不算失败**：从异常消息里抠出 `HLO temporaries (X G)`，那就是这个配置的
    显存需求，是本 job 要的测量值。跑成功时读 peak_bytes_in_use。
    """
    import jax
    try:
        ms = build_and_run()
        return "跑通", _hbm_peak(), ms
    except Exception as e:
        m = _RE_MEM.search(str(e))
        if m:
            return "OOM", float(m.group(1)), None
        raise


def _hbm_peak():
    import jax
    st = jax.devices()[0].memory_stats() or {}
    return st.get("peak_bytes_in_use", 0) / 1024 ** 3


def _bench(fn, reps=2, warmup=1):
    import jax
    jax.block_until_ready(fn())
    for _ in range(warmup):
        jax.block_until_ready(fn())
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        jax.block_until_ready(fn())
        ts.append((time.perf_counter() - t) * 1e3)
    return min(ts)


def _smap(f, mesh, in_specs, out_specs):
    from jax.experimental.shard_map import shard_map
    for kw in ({"check_vma": False}, {"check_rep": False}, {}):
        try:
            return shard_map(f, mesh=mesh, in_specs=in_specs, out_specs=out_specs, **kw)
        except TypeError:
            continue
    raise RuntimeError("shard_map 不接受任何已知的 check_* 参数名")


def _cleanup(drop_kernels=False):
    import gc
    if drop_kernels:
        try:
            _splash_kernel.cache_clear()
        except Exception:
            pass
    gc.collect()


def _grid_for(n):
    h = max(d for d in range(1, int(n ** 0.5) + 1) if n % d == 0)
    return h, n // h


def build_pack(coarse, real, jnp):
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
    to = jnp.asarray
    return {"seg_self": to(seg_self), "seg_cross": to(seg_cross),
            "mod_index": to(mod_index), "rows": to(rows), "cols": to(cols),
            "loss_mask": to(loss_mask)}


def make_random_params(key, cfg, dtype):
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
    return {"x_embedder": w(D, cfg.in_dim), "t_embedder_1": w(D, D),
            "t_embedder_2": w(3 * D, D), "t_embedding_norm": one(D),
            "final_adaln_1": w(R, D), "final_adaln_2": w(2 * D, R),
            "final_linear": w(cfg.out_dim, D), "blocks": blocks}


def run_variant(*, mod_broadcast=False, no_cross=False, bwd_block=1024,
                no_adaln=False, fwd_only=False, remat="full"):
    """按变量表跑一次 8 卡训练步（或纯前向）。返回步时 ms。"""
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    _cleanup(drop_kernels=True)

    devs = jax.devices()
    mesh = Mesh(np.array(devs).reshape(len(devs)), ("d",))
    cfg, dtype = AnimaConfig(), jnp.bfloat16
    pack = build_pack(COARSE, REAL, jnp)
    if mod_broadcast:
        # gather -> 广播（数值不对，只测显存）
        pack["mod_index"] = jnp.zeros_like(pack["mod_index"])

    # 反向块大小是本轮的变量之一 -> 临时改掉 attention.py 的偏好序
    global BWD_BLOCK_PREF
    saved = BWD_BLOCK_PREF
    BWD_BLOCK_PREF = tuple(b for b in saved if b <= bwd_block) or (128,)
    try:
        txt = [T_TXT] * len(COARSE)
        self_attn = make_splash_attn(COARSE, COARSE, cfg.num_heads, cfg.head_dim)
        cross_attn = make_splash_attn(COARSE, txt, cfg.num_heads, cfg.head_dim)
    finally:
        BWD_BLOCK_PREF = saved

    params = jax.jit(lambda: make_random_params(jax.random.PRNGKey(0), cfg, dtype),
                     out_shardings=NamedSharding(mesh, P()))()
    lora = jax.jit(lambda: init_lora(jax.random.PRNGKey(1), cfg, LORA_RANK, dtype=dtype),
                   out_shardings=NamedSharding(mesh, P()))()
    B, G = sum(COARSE), len(COARSE)
    ks = jax.random.split(jax.random.PRNGKey(2), 4)
    n = len(devs)
    spec = {k: P("d") for k in ("tok", "t", "ctx", "target")}
    batch = jax.jit(lambda: {
        "tok": jax.random.normal(ks[0], (n, B, cfg.in_dim), dtype),
        "t": jax.random.uniform(ks[1], (n, G), jnp.float32),
        "ctx": jax.random.normal(ks[2], (n, G * T_TXT, cfg.crossattn_dim), dtype),
        "target": jax.random.normal(ks[3], (n, B, cfg.out_dim), dtype),
    }, out_shardings={k: NamedSharding(mesh, v) for k, v in spec.items()})()

    zero_cross = lambda q, k, v: jnp.zeros_like(q)

    def local(lo, pa, b):
        sf = bind_segments(self_attn, pack["seg_self"], pack["seg_self"])
        cf = zero_cross if no_cross else bind_segments(
            cross_attn, pack["seg_cross"], jnp.asarray(segment_ids([T_TXT] * G)))
        mi = pack["mod_index"]
        if no_adaln:
            # 把调制关掉：shift=0 scale=0 gate=1 —— 等价于去掉 AdaLN 的全部 gather
            emb_zero = jnp.zeros((G, cfg.model_channels), dtype)
            out = forward_packed_noadaln(pa, cfg, b["tok"], b["t"], b["ctx"],
                                         pack["rows"], pack["cols"], mi, sf, cf,
                                         loras=lo, remat=remat)
        else:
            out = forward_packed(pa, cfg, b["tok"], b["t"], b["ctx"],
                                 pack["rows"], pack["cols"], mi, sf, cf,
                                 loras=lo, remat=remat)
        se = jnp.mean((out.astype(jnp.float32) - b["target"].astype(jnp.float32)) ** 2, -1)
        m = pack["loss_mask"]
        return jnp.sum(se * m) / jnp.maximum(jnp.sum(m), 1.0)

    def per_shard(lo, pa, b):
        return local(lo, pa, {k: v[0] for k, v in b.items()})[None]

    f = _smap(per_shard, mesh, (P(), P(), spec), P("d"))
    obj = lambda lo, pa, ba: jnp.mean(f(lo, pa, ba))
    fn = jax.jit(obj) if fwd_only else jax.jit(jax.grad(obj, argnums=0))
    return _bench(lambda: fn(lora, params, batch))


def forward_packed_noadaln(params, cfg, tokens, timesteps, ctx, rows, cols,
                           mod_index, self_fn, cross_fn, loras=None, remat="full"):
    """把 AdaLN 调制整个去掉的对照版（shift=0/scale=0/gate=1）。

    只为归因显存：如果这一版的需求塌下来，说明 1.01MB/token 主要来自
    `jnp.take(h, mod_index)` 那三次逐 token gather。数值当然是错的。
    """
    import jax
    import jax.numpy as jnp
    net = params
    x = tokens @ net["x_embedder"].T.astype(tokens.dtype)
    cos, sin = packed_rope_cos_sin(rows, cols, cfg.head_dim,
                                   cfg.rope_h_ratio, cfg.rope_w_ratio)
    _, wrap = resolve_remat(remat)

    def block(x, p, layer):
        D = cfg.model_channels
        lp = f"blocks.{layer}"
        h = layer_norm(x, cfg.eps_ln)
        q, k, v = attn_qkv(h, None, p["self_attn"], cfg, loras,
                           f"{lp}.self_attn", cos, sin)
        x = x + dense(self_fn(q, k, v).reshape(-1, D), p["self_attn"]["output_proj"],
                      _lora(loras, f"{lp}.self_attn.output_proj"))
        h = layer_norm(x, cfg.eps_ln)
        q, k, v = attn_qkv(h, ctx, p["cross_attn"], cfg, loras, f"{lp}.cross_attn")
        x = x + dense(cross_fn(q, k, v).reshape(-1, D), p["cross_attn"]["output_proj"],
                      _lora(loras, f"{lp}.cross_attn.output_proj"))
        h = layer_norm(x, cfg.eps_ln)
        h = gelu_exact(dense(h, p["mlp"]["layer1"], _lora(loras, f"{lp}.mlp.layer1")))
        return x + dense(h, p["mlp"]["layer2"], _lora(loras, f"{lp}.mlp.layer2"))

    for i in range(cfg.num_blocks):
        def one(carry, p, _i=i):
            return block(carry, p, _i)
        x = wrap(one, i)(x, params["blocks"][i])
    out = dense(layer_norm(x, cfg.eps_ln), net["final_linear"])
    return output_tokens_to_patch_tokens(out, cfg)


VARIANTS = [
    ("V0 基线（复现上一跑的 20.60G）", {}),
    ("V1 mod_index 恒 0（gather->广播）", {"mod_broadcast": True}),
    ("V2 关 cross-attn", {"no_cross": True}),
    ("V3 反向块 128（而非 1024）", {"bwd_block": 128}),
    ("V4 关 AdaLN 调制（去掉全部逐 token gather）", {"no_adaln": True}),
    ("V5 只前向不求梯度", {"fwd_only": True}),
]


@probe("P0 设备 / HBM")
def p0():
    import jax
    d = jax.devices()
    lim = (d[0].memory_stats() or {}).get("bytes_limit", 0) / 1024 ** 3
    return f"jax {jax.__version__} | {len(d)} x {d[0].device_kind} | HBM limit={lim:.1f}GiB"


@probe("M 显存归因（budget=16384 / remat=full，单变量）")
def m_attr():
    out = []
    for name, kw in VARIANTS:
        try:
            status, gb, ms = mem_required(lambda: run_variant(**kw))
            out.append(f"{name}: {status} {gb:.2f}G"
                       + (f" 步时{ms:.0f}ms" if ms else ""))
        except Exception as e:
            out.append(f"{name}: {type(e).__name__}: {str(e)[:120]}")
        _cleanup(drop_kernels=True)
    return "\n         ".join(out)


def main():
    record("python", "INFO", f"{platform.python_version()} @ {sys.executable}")
    record("bootstrap", "INFO", _BOOT)
    if not p0():
        _flush()
        return 0
    m_attr()
    record("汇总", "INFO", f"总耗时={time.time() - _T0:.0f}s")
    _flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
