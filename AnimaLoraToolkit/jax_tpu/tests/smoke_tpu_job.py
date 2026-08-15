"""冒烟闸门（本地 CPU，零 TPU 配额）：验生成的 Kaggle job 能不能跑、接线对不对。

**推 TPU 之前必跑。** 真机每跑一轮约 30 分钟 + 配额，本地几分钟就能挡掉大部分错。
历史上它挡下过两个：
  * `jax.checkpoint` 把层号 trace 成 tracer -> LoRA 静默失效、梯度恒 0
    （且反向可能被 DCE 掉，使步时**假性变快** —— 不断言根本发现不了）
  * 网格用 round(sqrt(n)) 取整 -> n 非完全平方数时形状对不上（真机第一跑就栽这）

两组：
  A 单设备：前向 + 反向、LoRA step-0 中立性、各档 token 数的网格/块大小合法性
  B 八设备（`--xla_force_host_platform_device_count=8` 造虚拟 CPU 设备）：
    shard_map 分片规格、梯度穿透、复制参数的余切是否被 psum
    （= LoRA 梯度 all-reduce 是否真的发生）

splash 在 CPU 上跑不了 Mosaic，B 组用稠密注意力顶替 —— 只验分片结构，不验性能。

用法（需要 jax 的解释器）：
    python smoke_tpu_job.py [--job 路径]
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path

# 必须在 import jax 之前设：① 关掉 job 的 pip 升级；② 造 8 个虚拟 CPU 设备
os.environ.setdefault("ANIMA_TPU_UPGRADE", "0")
os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "")
                           + " --xla_force_host_platform_device_count=8").strip()

import jax                                                   # noqa: E402
import jax.numpy as jnp                                      # noqa: E402
import numpy as np                                           # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P   # noqa: E402

DEFAULT_JOB = (Path(__file__).resolve().parents[3]
               / "kaggle_job" / "anima_dp" / "anima_tpu_job.py")
DKEYS = ("tok", "t", "ctx", "rows", "cols", "target")
FAILS: list = []


def check(name, cond, detail=""):
    print(f"  [{'OK ' if cond else 'BAD'}] {name}" + (f" - {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)
    return cond


def rowcol(m, n):
    """token 数 -> RoPE 的 (rows, cols)。

    **不要直接调 job 里的私有 helper。** 历史上这里写死过 `m._rowcol`，
    而 job 主体后来把它改名成 `_grid_for`（只返回 h, w），于是本闸门从
    anima_navit 那一轮起就一直崩在这一行、连着四轮没人跑成功 —— 探针是裸推
    真机的。现在只依赖 `_grid_for` 这个各版 job 都有的函数，rows/cols 在这里
    自己展开，闸门不再随 job 内部改名而失效。
    """
    h, w = m._grid_for(n)
    rr, cc = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing="ij")
    return rr.reshape(-1), cc.reshape(-1)


def load_job(path):
    spec = importlib.util.spec_from_file_location("job", str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules["job"] = m
    spec.loader.exec_module(m)
    return m


def make_batch(m, cfg, B, n, dtype=jnp.float32, shardings=None):
    rr, cc = rowcol(m, n)
    ks = jax.random.split(jax.random.PRNGKey(2), 4)
    mk = lambda: {
        "tok": jax.random.normal(ks[0], (B, n, cfg.in_dim), dtype),
        "t": jax.random.uniform(ks[1], (B,), jnp.float32),
        "ctx": jax.random.normal(ks[2], (B, m.T_TXT, cfg.crossattn_dim), dtype),
        "rows": jnp.broadcast_to(rr[None], (B, n)),
        "cols": jnp.broadcast_to(cc[None], (B, n)),
        "target": jax.random.normal(ks[3], (B, n, cfg.out_dim), dtype),
    }
    return jax.jit(mk, out_shardings=shardings)() if shardings else jax.jit(mk)()


def group_a(m):
    """单设备：形状合法性 + 前向反向 + LoRA step-0 中立。"""
    print("\nA 组 · 单设备")

    # A1 各档 token 数的网格分解与 splash 块大小
    for n in (512, 1024, 2304, 4096, 8192, 9216, 16384):
        h, w = m._grid_for(n)
        rr, cc = rowcol(m, n)
        bwd = next((b for b in (1024, 512, 256, 128) if n % b == 0), None)
        check(f"A1 n={n:<6} grid={h}x{w} bwd_block={bwd}",
              h * w == n and rr.shape == (n,) and n % 128 == 0)

    # A2 前向 + 反向 + step-0 中立
    cfg = m.AnimaConfig(num_blocks=2)
    m.T_TXT = 16
    n, G = 256, 2
    # **params 与 lora 的布局必须一致**：`make_random_params` 默认 stack=True
    # （scan 路径），LoRA 也得 `stack_loras` 堆成 [L,...]，否则 lax.scan 的
    # 前导轴对不上（真机训练一律走 scan，见 train.py:init_lora_stacked）。
    # 这条以前写成"stacked params + 扁平 lora"，闸门修好后第一跑就炸在这。
    params = m.make_random_params(jax.random.PRNGKey(0), cfg, jnp.float32)
    lora = m.stack_loras(m.init_lora(jax.random.PRNGKey(1), cfg, 4,
                                     dtype=jnp.float32), cfg.num_blocks)
    batch = make_batch(m, cfg, G, n)

    sc = cfg.head_dim ** -0.5

    def dense_attn(q, k, v):
        lg = jnp.einsum("shd,thd->hst", q, k).astype(jnp.float32) * sc
        return jnp.einsum("hst,thd->shd",
                          jax.nn.softmax(lg, -1).astype(v.dtype), v)

    def per_image(pa, lo, tok, t, ctx, rows, cols, target):
        mi = jnp.zeros((n,), jnp.int32)
        o = m.forward_packed(pa, cfg, tok, t[None], ctx, rows, cols, mi,
                             dense_attn, dense_attn, loras=lo, remat=True)
        return jnp.mean((o.astype(jnp.float32) - target.astype(jnp.float32)) ** 2)

    def loss(lo, pa, ba):
        return jnp.mean(jax.vmap(per_image, in_axes=(None, None, 0, 0, 0, 0, 0, 0))(
            pa, lo, ba["tok"], ba["t"], ba["ctx"], ba["rows"], ba["cols"],
            ba["target"]))

    l0 = float(loss(lora, params, batch))
    check("A2 前向可跑", np.isfinite(l0), f"loss={l0:.6f}")

    # LoRA 全置 None 应与 B=0 的 LoRA 完全同值 —— step-0 净增量为 0
    l_none = float(jax.vmap(
        lambda tok, t, ctx, rows, cols, target: per_image(
            params, None, tok, t, ctx, rows, cols, target))(
        batch["tok"], batch["t"], batch["ctx"], batch["rows"],
        batch["cols"], batch["target"]).mean())
    check("A3 step-0 中立（接 LoRA 与不接 LoRA 同值）",
          abs(l_none - l0) < 1e-6, f"Δ={abs(l_none - l0):.2e}")

    g = jax.grad(loss, argnums=0)(lora, params, batch)
    ga = sum(float(jnp.abs(v["a"]).sum()) for v in g.values())
    gb = sum(float(jnp.abs(v["b"]).sum()) for v in g.values())
    nz = sum(1 for v in g.values() if float(jnp.abs(v["b"]).max()) > 0)
    check("A4 LoRA 接进计算图（grad_b 非零）", gb > 0 and nz == len(g),
          f"sum|grad_b|={gb:.3e} 非零 {nz}/{len(g)}")
    check("A5 grad_a 恒 0（B 零初始化）", ga == 0, f"sum|grad_a|={ga:.1e}")
    return cfg, params, lora, dense_attn


def group_b(m, cfg, params, lora, dense_attn):
    """八设备：shard_map 分片 + 梯度 all-reduce。"""
    print("\nB 组 · 八设备（虚拟 CPU）")
    if not check("B0 拿到 8 个设备", len(jax.devices()) == 8,
                 f"实际 {len(jax.devices())} 个"):
        return
    n, G = 128, 2
    mesh = Mesh(np.array(jax.devices()).reshape(8), ("d",))
    sh = lambda s: NamedSharding(mesh, s)
    params = jax.device_put(params, sh(P()))
    lora = jax.device_put(lora, sh(P()))
    batch = make_batch(m, cfg, 8 * G, n,
                       shardings={k: sh(P("d")) for k in DKEYS})
    check("B1 batch 按卡切", batch["tok"].shape == (8 * G, n, cfg.in_dim))

    def per_image(pa, lo, tok, t, ctx, rows, cols, target):
        mi = jnp.zeros((n,), jnp.int32)
        o = m.forward_packed(pa, cfg, tok, t[None], ctx, rows, cols, mi,
                             dense_attn, dense_attn, loras=lo, remat=True)
        return jnp.mean((o.astype(jnp.float32) - target.astype(jnp.float32)) ** 2)

    def per_shard(lo, pa, ba):
        return jnp.mean(jax.vmap(per_image, in_axes=(None, None, 0, 0, 0, 0, 0, 0))(
            pa, lo, ba["tok"], ba["t"], ba["ctx"], ba["rows"], ba["cols"],
            ba["target"]))[None]

    f = m._smap(per_shard, mesh, (P(), P(), {k: P("d") for k in DKEYS}), P("d"))
    loss = lambda lo, pa, ba: jnp.mean(f(lo, pa, ba))
    l = float(jax.jit(loss)(lora, params, batch))
    check("B2 shard_map 前向可跑", np.isfinite(l), f"loss={l:.6f}")

    g = jax.jit(jax.grad(loss, argnums=0))(lora, params, batch)
    gb = sum(float(jnp.abs(v["b"]).sum()) for v in g.values())
    check("B3 梯度穿过 shard_map", gb > 0, f"sum|grad_b|={gb:.3e}")

    # 复制参数（in_spec=P()）的余切在转置时应被 psum -> 各卡分片必须完全一致
    one = list(g.values())[0]["b"]
    shards = [np.asarray(s.data) for s in one.addressable_shards]
    check("B4 LoRA 梯度已 all-reduce（各卡分片一致）",
          all(np.array_equal(shards[0], s) for s in shards[1:]),
          f"{len(shards)} 片")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default=str(DEFAULT_JOB))
    a = ap.parse_args()
    job = Path(a.job)
    if not job.exists():
        print(f"找不到 {job} —— 先跑 kaggle_job/anima_dp/build_job.py 生成")
        return 2
    m = load_job(job)
    print(f"job 加载 OK: {job.name}")
    group_b(m, *group_a(m))
    print(f"\n*** {'冒烟通过' if not FAILS else '冒烟失败'} ***"
          + ("" if not FAILS else f"  失败项: {FAILS}"))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
