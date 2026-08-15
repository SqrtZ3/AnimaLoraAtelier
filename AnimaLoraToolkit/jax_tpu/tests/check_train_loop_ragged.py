"""闸门 ⑨（端到端 · 分桶路线）：ragged 训练闭环在 CPU 上真的能学，且 8 卡分片走得通。

与 `check_train_loop.py` 是同一套判据、同一个规模，只换布局
（`BucketPlanner` + `make_step_ragged` + `assemble_batch_ragged`）。
两条路线的前向已由 `check_ragged_equiv.py` 证明**逐 bit 等价**（rel=0.000e+00），
这里补的是"装配"这一层：分桶的索引数组、shard_map 的 spec、loss 权重的广播口径。

用 `XLA_FLAGS=--xla_force_host_platform_device_count=8` 在 CPU 上伪造 8 个设备，
splash 走 `interpret=True`。

判据：
  T1 step-0 中立：B 零初始化 -> 接不接 LoRA，loss 必须**逐 bit 相同**
  T2 真的在学：固定一批数据反复训 N 步，loss 必须显著下降
  T3 梯度确实跨了 8 卡：换掉某一卡的数据，参数必须变
  T4 填充 token 不参与：改填充区的 latent，loss 必须**逐 bit 不变**
  T5 存取往返：save_state -> load_state 后再训一步，与不落盘逐 bit 一致
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import numpy as np                                             # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import anima_jax as A                                          # noqa: E402
import flow as F                                               # noqa: E402
import optim as O                                              # noqa: E402
import packing as PK                                           # noqa: E402
import adapters as AD
import train as T                                              # noqa: E402
import jax                                                     # noqa: E402
import jax.numpy as jnp                                        # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec    # noqa: E402

TXT, NDEV, STEPS = 128, 8, 30
QUANTUM, PER_DEV = 256, 1
_bad = []


def ok(name, cond, detail=""):
    print(f"  [{'OK ' if cond else 'BAD'}] {name:<30} {detail}")
    if not cond:
        _bad.append(name)


def make_params(cfg, key):
    ks = iter(jax.random.split(key, 8 + cfg.num_blocks * 16))
    D, C, R = cfg.model_channels, cfg.crossattn_dim, cfg.adaln_lora_dim
    Fd = int(D * cfg.mlp_ratio)
    w = lambda *s: (jax.random.normal(next(ks), s, jnp.float32) * 0.02).astype(jnp.bfloat16)
    one = lambda n: jnp.ones((n,), jnp.bfloat16)
    blocks = [{
        "self_attn": {"q_proj": w(D, D), "k_proj": w(D, D), "v_proj": w(D, D),
                      "output_proj": w(D, D), "q_norm": one(cfg.head_dim),
                      "k_norm": one(cfg.head_dim)},
        "cross_attn": {"q_proj": w(D, D), "k_proj": w(D, C), "v_proj": w(D, C),
                       "output_proj": w(D, D), "q_norm": one(cfg.head_dim),
                       "k_norm": one(cfg.head_dim)},
        "mlp": {"layer1": w(Fd, D), "layer2": w(D, Fd)},
        "adaln_self_1": w(R, D), "adaln_self_2": w(3 * D, R),
        "adaln_cross_1": w(R, D), "adaln_cross_2": w(3 * D, R),
        "adaln_mlp_1": w(R, D), "adaln_mlp_2": w(3 * D, R),
    } for _ in range(cfg.num_blocks)]
    return {"x_embedder": w(D, cfg.in_dim), "t_embedder_1": w(D, D),
            "t_embedder_2": w(3 * D, D), "t_embedding_norm": one(D),
            "final_adaln_1": w(R, D), "final_adaln_2": w(2 * D, R),
            "final_linear": w(cfg.out_dim, D), "blocks": blocks}


# ── 新 API 的薄包装 ───────────────────────────────────────────────────────────
# train.py 把"求梯度"与"更新参数"拆成了两个 jit（为了 grad_accum 跨布局），
# 这里把它们粘回一个 `step(state, params, batch)`，本文件其余断言逐字不动。
# `apply_update` 会捐献 state，所以先拷一份 —— 否则调用方手里的 state 会失效
# （`RuntimeError: Array has been deleted`，本仓库记录在案的一类坑）。
def make_step_compat(gf, consts, tcfg, key0=0):
    ctr = [0]

    def step(state, params, batch):
        ctr[0] += 1
        loss, per, grads = gf(state["master"], consts, params, batch,
                              jax.random.PRNGKey(key0))
        st = jax.tree.map(lambda x: jnp.array(x), state)
        st, diag = T.apply_update(st, grads, tcfg.adamw)
        return st, {"loss": loss, "per_image": per, **diag}
    return step


def leaf_names(lora):
    """遍历适配器树的 (模块名, 叶子名)，不写死 a/b —— LoKr 是 w1/w2a/w2b。"""
    return [(n, k) for n, mod in lora.items() for k in mod]


def main() -> int:
    devs = jax.devices()
    print(f"设备 {len(devs)} x {devs[0].device_kind}")
    if len(devs) < NDEV:
        print(f"只有 {len(devs)} 个设备，XLA_FLAGS 没生效？")
        return 2
    mesh = Mesh(np.array(devs[:NDEV]).reshape(NDEV), ("d",))
    rep = NamedSharding(mesh, PartitionSpec())
    dev = NamedSharding(mesh, PartitionSpec("d"))

    mcfg = A.AnimaConfig(num_blocks=2, model_channels=256, num_heads=2,
                         crossattn_dim=128, adaln_lora_dim=32)
    tcfg = T.TrainConfig(adapter=AD.AdapterConfig(kind="lora", rank=4),
                         remat="full",
                         flow=F.FlowConfig(t_mode="logit_normal"),
                         adamw=O.AdamWConfig(lr=1e-2, warmup_steps=0))

    # 8 张真实 token 数**各不相同**的图（这正是分桶路线要证明的能力：同一步里
    # 每图有自己的 (h, w)，只要量化后落进同一个桶）。384/448 -> 桶长 512。
    grids = [(16, 24), (16, 28), (24, 16), (28, 16),
             (16, 24), (14, 32), (32, 14), (16, 28)]
    counts = [h * w for h, w in grids]
    planner = PK.BucketPlanner(quantum=QUANTUM, per_device=PER_DEV,
                               txt_len=TXT, devices=NDEV)
    steps, carry = planner.plan_steps(list(range(len(counts))), counts, grids)
    if not steps:
        print(f"没凑出一步（{PK.report_buckets(steps, len(carry))}）")
        return 2
    plan = steps[0]
    print(PK.report_buckets(steps, len(carry)))
    print(f"真实 token 数 {plan.real_lens} -> 桶长 {plan.bucket.length}")

    rng = np.random.RandomState(0)
    lats = [rng.randn(r, mcfg.out_dim).astype(np.float32) for r in plan.real_lens]
    ctxs = [rng.randn(TXT, mcfg.crossattn_dim).astype(np.float32)
            for _ in plan.real_lens]

    params = jax.device_put(A.stack_blocks(make_params(mcfg, jax.random.PRNGKey(0))), rep)
    lora, consts, plans = T.init_adapter(jax.random.PRNGKey(1), mcfg, tcfg,
                                        params)
    t_vec = F.sample_t_np(np.random.RandomState(2),
                          NDEV * plan.bucket.imgs, tcfg.flow)
    batch = jax.device_put(
        T.assemble_batch_ragged(plan, lats, ctxs, t_vec, mcfg, tcfg.dtype,
                                devices=NDEV), dev)
    step = make_step_compat(
        T.make_grad_fn_ragged(mcfg, tcfg, plans, plan.bucket, mesh, interpret=True),
        consts, tcfg)

    print("\nT1 step-0 中立（B 零初始化 -> 接不接 LoRA 同值）：")
    st0, m0 = step(dict(O.init_state(lora)), params, batch)
    zero = jax.tree.map(jnp.zeros_like, lora)
    _, m_z = step(O.init_state(zero), params, batch)
    ok("loss 逐 bit 相同", float(m0["loss"]) == float(m_z["loss"]),
       f"{float(m0['loss']):.8f} vs {float(m_z['loss']):.8f}")

    print("\nT4 填充 token 不参与 loss：")
    b2 = dict(batch)
    lm = np.asarray(batch["loss_mask"])
    pert = np.asarray(batch["latent"]).copy()
    pert[lm == 0] += 100.0                     # 只动填充位置
    b2["latent"] = jnp.asarray(pert)
    _, m_p = step(O.init_state(lora), params, b2)
    ok("loss 逐 bit 不变", float(m0["loss"]) == float(m_p["loss"]),
       f"{float(m0['loss']):.8f} vs {float(m_p['loss']):.8f}")

    print("\nT3 梯度确实跨了 8 卡：")
    b3 = dict(batch)
    lat3 = np.asarray(batch["latent"]).copy()
    lat3[NDEV - 1] *= -1.0                     # 只改最后一卡的数据
    b3["latent"] = jnp.asarray(lat3)
    st_3, _ = step(O.init_state(lora), params, b3)
    d = max(float(jnp.abs(st_3["master"][n][s] - st0["master"][n][s]).max())
            for n, s in leaf_names(lora))
    ok("改一卡数据 -> 参数变了", d > 0, f"max_abs Δ={d:.3e}（=0 说明 shard_map 没接对）")

    print(f"\nT2 真的在学（固定一批训 {STEPS} 步）：")
    st = O.init_state(lora)
    losses = []
    for _ in range(STEPS):
        st, m = step(st, params, batch)
        losses.append(float(m["loss"]))
    print(f"  loss: {losses[0]:.6f} -> {losses[-1]:.6f} "
          f"(gnorm {float(m['gnorm']):.3e}, step {int(m['step'])})")
    ok("loss 显著下降", losses[-1] < losses[0] * 0.95,
       f"降幅 {(1 - losses[-1] / losses[0]) * 100:.1f}%（判据 >5%）")
    ok("loss 单调性尚可", sum(b < a for a, b in zip(losses, losses[1:])) >= STEPS * 0.6,
       f"{sum(b < a for a, b in zip(losses, losses[1:]))}/{STEPS - 1} 步在降")

    print("\nT5 状态存取往返：")
    tmp = Path(os.environ.get("TEMP", "/tmp")) / "anima_jax_ragged_state_test.npz"
    T.save_state(tmp, st, tcfg)
    st_r = T.load_state(tmp)
    a1, _ = step(dict(st), params, batch)
    a2, _ = step(st_r, params, batch)
    d = max(float(jnp.abs(a1["master"][n][s] - a2["master"][n][s]).max())
            for n, s in leaf_names(lora))
    ok("往返后再训一步逐 bit 一致", d == 0.0, f"max_abs Δ={d:.3e}")
    tmp.unlink(missing_ok=True)
    tmp.with_suffix(".json").unlink(missing_ok=True)

    print(f"\n*** {'通过' if not _bad else '失败'} ***"
          + ("" if not _bad else f"  失败项: {_bad}"))
    return 0 if not _bad else 1


if __name__ == "__main__":
    sys.exit(main())
