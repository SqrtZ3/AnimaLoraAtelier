"""闸门 K-⑦（Krea2 端到端）：FSDP 分片训练闭环在 CPU 上真的能学。

与 check_train_loop.py（闸门 ⑦，Anima）同一套判据，外加 K2 特有的两条：
  T0 FSDP 逐设备对拍：分片权重（all_gather 路径）跑出的逐图 loss 与全量权重
     在**同一份数据、同一把 key** 下必须一致 —— gather 放错位置/拼错轴会在这里
     显形（这类错不报错，只会把权重悄悄拼乱）。
  T6 文本填充不参与：改 text 槽填充区，loss 必须逐 bit 不变（闸 refiner +
     combined 两级精细 mask —— K2 的文本是变长的，填充全靠 fine ids 隔离）。

用 `XLA_FLAGS=--xla_force_host_platform_device_count=8` 伪造 8 设备，
shard_map + all_gather 这条 FSDP 路径才走得到。splash 走 interpret=True。

规模刻意开小（2 块、features 256、budget 2048），CPU 上分钟级。
"""

import os
import sys
from dataclasses import replace as dc_replace
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import numpy as np                                             # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import adapters as AD                                          # noqa: E402
import attention as AT                                         # noqa: E402
import flow as F                                               # noqa: E402
import krea2_jax as K2                                         # noqa: E402
import optim as O                                              # noqa: E402
import packing as PK                                           # noqa: E402
import train as T                                              # noqa: E402
import jax                                                     # noqa: E402
import jax.numpy as jnp                                        # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

BUDGET, NDEV, STEPS = 2048, 8, 30
_bad = []


def ok(name, cond, detail=""):
    print(f"  [{'OK ' if cond else 'BAD'}] {name:<30} {detail}")
    if not cond:
        _bad.append(name)


def shard_rule(x):
    return (P(*([None] * (x.ndim - 1) + ["d"]))
            if K2.fsdp_shard_pred(x.shape, NDEV) else P())


def make_params(cfg, key):
    """与 load_safetensors_krea2 同构的随机权重树（norm scale/mod.lin 非零 —
    零初始化会让一大批算子退化成恒等，闸不住错位）。"""
    ks = iter(jax.random.split(key, 400))
    F_, T_, M_, TM_ = cfg.features, cfg.txtdim, cfg.mlpdim, cfg.txt_mlpdim
    hd = cfg.head_dim

    def w(*s, std=0.02):
        return (jax.random.normal(next(ks), s, jnp.float32) * std).astype(jnp.bfloat16)

    def v(*s):
        return jax.random.normal(next(ks), s, jnp.float32) * 0.1

    def blk(dim, mlp, kv, has_mod):
        d = {
            "prenorm": v(dim), "postnorm": v(dim),
            "attn": {"wq": w(dim, dim), "wk": w(kv * hd, dim), "wv": w(kv * hd, dim),
                     "gate": w(dim, dim), "wo": w(dim, dim),
                     "qnorm": v(hd), "knorm": v(hd)},
            "mlp": {"gate": w(mlp, dim), "up": w(mlp, dim), "down": w(dim, mlp)},
        }
        if has_mod:
            d["mod_lin"] = v(6 * dim)
        return d

    return {
        "first": {"w": w(F_, cfg.in_dim), "b": v(F_)},
        "tmlp0": {"w": w(F_, cfg.tdim), "b": v(F_)},
        "tmlp2": {"w": w(F_, F_), "b": v(F_)},
        "tproj": {"w": w(6 * F_, F_), "b": v(6 * F_)},
        "txtfusion": {
            "layerwise": [blk(T_, TM_, cfg.txtkvheads, False) for _ in range(2)],
            "projector": w(1, cfg.txtlayers),
            "refiner": [blk(T_, TM_, cfg.txtkvheads, False) for _ in range(2)],
        },
        "txtmlp_norm": v(T_),
        "txtmlp1": {"w": w(F_, T_), "b": v(F_)},
        "txtmlp3": {"w": w(F_, F_), "b": v(F_)},
        "blocks": [blk(F_, M_, cfg.kvheads, True) for _ in range(cfg.layers)],
        "last": {"norm": v(F_), "linear_w": w(cfg.in_dim, F_),
                 "linear_b": v(cfg.in_dim), "mod_lin": v(2, F_)},
    }


def make_step_compat(gf, consts, tcfg, key0=0):
    def step(state, params, batch):
        loss, per, grads = gf(state["master"], consts, params, batch,
                              jax.random.PRNGKey(key0))
        st = jax.tree.map(lambda x: jnp.array(x), state)   # 先拷再捐（记录在案的坑）
        st, diag = T.apply_update(st, grads, tcfg.adamw)
        return st, {"loss": loss, "per_image": per, **diag}
    return step


def leaf_names(lora):
    return [(n, k) for n, mod in lora.items() for k in mod]


def main() -> int:
    devs = jax.devices()
    print(f"设备 {len(devs)} x {devs[0].device_kind}")
    if len(devs) < NDEV:
        print(f"只有 {len(devs)} 个设备，XLA_FLAGS 没生效？")
        return 2
    mesh = Mesh(np.array(devs[:NDEV]).reshape(NDEV), ("d",))

    mcfg = K2.Krea2Config(features=256, tdim=64, txtdim=128, heads=4, kvheads=1,
                          multiplier=2, layers=2, patch=2, channels=16,
                          txtheads=2, txtkvheads=2, txtlayers=3)
    tcfg = T.TrainConfig(adapter=AD.AdapterConfig(kind="lora", rank=4),
                         remat="full",
                         flow=F.FlowConfig(t_mode="logit_normal"),
                         adamw=O.AdamWConfig(lr=1e-2, warmup_steps=0))

    # 两张真图（1024/384 token）+ caption（100/200 -> 槽 128/256）
    packer = PK.K2Packer(BUDGET, quantum=512, devices=NDEV)
    one = packer.build_packs(["a", "b"], [1024, 384], [100, 200],
                             [(32, 32), (16, 24)])[0]
    batch_packs = [one] * NDEV
    layout = one.layout
    print(f"布局 seg={layout.seg_lens} txt={layout.txt_segs} img={layout.img_segs}"
          f" | 每步 {len(batch_packs)} 个 pack")

    rng = np.random.RandomState(0)
    lats, ctxs = [], []
    for p in batch_packs:
        lats.append([rng.randn(r, mcfg.in_dim).astype(np.float32) for r in p.real_img_lens])
        ctxs.append([(rng.randn(rt, mcfg.txtlayers, mcfg.txtdim)
                      * 0.5).astype(np.float32) for rt in p.real_txt_lens])

    # 分片权重（FSDP）与全量权重（T0 对照用）各备一份
    base = make_params(mcfg, jax.random.PRNGKey(0))
    params = jax.tree.map(
        lambda x: jax.device_put(x, NamedSharding(mesh, shard_rule(x))), base)
    params = K2.stack_blocks(params, mesh)
    params_rep = K2.stack_blocks(
        jax.tree.map(lambda x: jax.device_put(x, NamedSharding(mesh, P())), base),
        mesh)

    lora, consts, plans = T.init_adapter_k2(jax.random.PRNGKey(1), mcfg,
                                            tcfg.adapter, None)
    state = O.init_state(lora)
    t_vec = F.sample_t_np(np.random.RandomState(2),
                          len(batch_packs) * layout.n_seg, tcfg.flow)
    batch = T.assemble_batch_k2(batch_packs, lats, ctxs, t_vec, mcfg, tcfg.dtype)
    batch = jax.device_put(batch, NamedSharding(mesh, P("d")))
    pspec = T.fsdp_pspec(params, "d", NDEV)
    gf = T.make_grad_fn_k2(mcfg, tcfg, plans, layout, mesh, pspec, interpret=True)
    step = make_step_compat(gf, consts, tcfg)

    txt_pos, img_pos = (jnp.asarray(a) for a in layout.static_positions())
    idx = one.index_arrays()

    print("\nT0 FSDP 逐设备对拍（分片 ≡ 全量，同数据同 key）：")
    main_attn = AT.make_splash_attn(list(layout.seg_lens), list(layout.seg_lens),
                                    mcfg.heads, mcfg.head_dim, interpret=True)
    rf_attn = AT.make_splash_attn(list(layout.txt_segs), list(layout.txt_segs),
                                  mcfg.txtheads, mcfg.txt_head_dim, interpret=True)
    main_gqa = K2.wrap_attn_gqa(main_attn, mcfg.heads)
    key0 = jax.random.PRNGKey(0)
    _, per_sharded, _ = gf(state["master"], consts, params, batch, key0)
    diffs = []
    for i in range(NDEV):
        kk = jax.random.fold_in(key0, i)
        # **每卡取自己的 batch 切片**（各卡 t 不同！）—— 曾图省事全用第 0 卡的
        # 切片，差出 1~2% 还以为是 FSDP 拼错了权重；那是 t 不同的正常差异。
        one_b = {k: np.asarray(v)[i] for k, v in batch.items()}
        self_fn = AT.bind_segments(main_gqa, one_b["seg_self"], one_b["seg_self"])
        rf_fn = AT.bind_segments(rf_attn, one_b["txt_fine"], one_b["txt_fine"])
        loss_i, clean_i = T.local_loss_k2(
            jax.tree.map(lambda x: x.astype(tcfg.dtype), lora), consts, params_rep,
            one_b, mcfg, tcfg, plans, self_fn, rf_fn, kk, txt_pos, img_pos,
            layout.n_seg, mesh_axis=None)
        diffs.append(float(jnp.abs(clean_i - per_sharded[i]).max()
                           / max(float(jnp.abs(per_sharded[i]).max()), 1e-12)))
    # bf16 下 jit(scan+shard_map) 与 eager 的融合顺序不同，噪声 ~1e-2；
    # **fp32 同款对比实测 2.2e-07**（开发时验证），拼权错误会是 O(1) 量级。
    ok("逐设备逐图 loss 一致", max(diffs) < 5e-2,
       f"max rel Δ={max(diffs):.2e}（bf16 编译噪声；fp32 实测 2.2e-07）")

    print("\nT1 step-0 中立（B 零初始化 -> 接不接 LoRA 同值）：")
    st0, m0 = step(dict(state), params, batch)
    zero = jax.tree.map(jnp.zeros_like, lora)
    st_z, m_z = step(O.init_state(zero), params, batch)
    ok("loss 逐 bit 相同", float(m0["loss"]) == float(m_z["loss"]),
       f"{float(m0['loss']):.8f} vs {float(m_z['loss']):.8f}")

    print("\nT4 图像填充 token 不参与 loss：")
    b2 = dict(batch)
    lm = np.asarray(batch["loss_mask"])
    pert = np.asarray(batch["latent"]).copy()
    pert[lm == 0] += 100.0
    b2["latent"] = jnp.asarray(pert)
    _, m_p = step(O.init_state(lora), params, b2)
    ok("loss 逐 bit 不变", float(m0["loss"]) == float(m_p["loss"]),
       f"{float(m0['loss']):.8f} vs {float(m_p['loss']):.8f}")

    print("\nT6 文本填充不参与 loss：")
    b6 = dict(batch)
    pert6 = np.asarray(batch["txt"]).copy()
    tf = np.asarray(batch["txt_fine"])
    pert6[tf == PK.PAD_SEG] += 100.0
    b6["txt"] = jnp.asarray(pert6)
    _, m_6 = step(O.init_state(lora), params, b6)
    ok("loss 逐 bit 不变", float(m0["loss"]) == float(m_6["loss"]),
       f"{float(m0['loss']):.8f} vs {float(m_6['loss']):.8f}")

    print("\nT8 caption dropout：换入短 caption 后，原有效区间被精细 mask 隔离：")
    # 模拟 dropout：每卡第 0 段的 caption 截短（空 caption 的角色）。修复前
    # [len, rt) 的零值被按扫描长度标成有效，txtmlp 的 bias 会把它穿透成非零
    # 假 token 参与 refiner 与主序列注意力；修复后 assemble_batch_k2 把该区间
    # 重标 PAD_SEG —— 扰动它必须逐 bit 不影响 loss。
    ctxs_d = [[c[: max(1, c.shape[0] // 2)] for c in per] for per in ctxs]
    rt0 = batch_packs[0].real_txt_lens[0]
    lo = max(1, rt0 // 2)
    batch_d = T.assemble_batch_k2(batch_packs, lats, ctxs_d, t_vec, mcfg,
                                  tcfg.dtype)
    tf_d = np.asarray(batch_d["txt_fine"])
    ss_d = np.asarray(batch_d["seg_self"])
    remark_ok = bool((tf_d[:, lo:rt0] == PK.PAD_SEG).all()
                     and (ss_d[:, lo:rt0] == PK.PAD_SEG).all())
    ok("重标生效（[len, rt) -> PAD_SEG）", remark_ok,
       f"txt_fine/seg_self 区间 [:, {lo}:{rt0}]")
    _, m_d = step(O.init_state(lora), params, batch_d)
    b8 = dict(batch_d)
    pert8 = np.asarray(batch_d["txt"]).copy()
    pert8[:, lo:rt0] += 100.0
    b8["txt"] = jnp.asarray(pert8)
    _, m_8 = step(O.init_state(lora), params, b8)
    ok("扰动已隔离区间，loss 逐 bit 不变",
       float(m_d["loss"]) == float(m_8["loss"]),
       f"{float(m_d['loss']):.8f} vs {float(m_8['loss']):.8f}")

    print("\nT3 梯度确实跨了 8 卡：")
    b3 = dict(batch)
    lat3 = np.asarray(batch["latent"]).copy()
    lat3[NDEV - 1] *= -1.0
    b3["latent"] = jnp.asarray(lat3)
    st_3, _ = step(O.init_state(lora), params, b3)
    d = max(float(jnp.abs(st_3["master"][n][s] - st0["master"][n][s]).max())
            for n, s in leaf_names(lora))
    ok("改一卡数据 -> 参数变了", d > 0, f"max_abs Δ={d:.3e}（=0 说明 shard_map 没接对）")

    print(f"\nT2 真的在学（固定一批训 {STEPS} 步）：")
    st = O.init_state(lora)
    losses = []
    for i in range(STEPS):
        st, m = step(st, params, batch)
        losses.append(float(m["loss"]))
    print(f"  loss: {losses[0]:.6f} -> {losses[-1]:.6f} "
          f"(gnorm {float(m['gnorm']):.3e}, step {int(m['step'])})")
    ok("loss 显著下降", losses[-1] < losses[0] * 0.95,
       f"降幅 {(1 - losses[-1] / losses[0]) * 100:.1f}%（判据 >5%）")

    print("\nT7 aux 真的接上了（dfm+eisbach 同时开）：")
    # aux 只改"要不要让这张图影响参数"，不改**干净**逐图 loss（那是给自适应
    # 采样器的度量，与 aux 无关）—— 同一把 key 下 clean 必须逐 bit 相同，
    # 而 graded loss 必须变。两条一起才能区分"接了"与"接了个寂寞"。
    import auxloss as X
    # canvas_hw 必须 >= 数据集最大网格（生产里由 CacheDataset.canvas_hw 回填；
    # 这里数据集网格 (32,32)/(16,24)）—— ΔFM 的裁剪支路与 spectral 共用画布
    tcfg_aux = dc_replace(tcfg, aux=X.AuxConfig(
        eisbach_lambda=0.3, dfm_lambda=0.5, canvas_hw=(32, 32)))
    gf_aux = T.make_grad_fn_k2(mcfg, tcfg_aux, plans, layout, mesh, pspec,
                               interpret=True)
    loss_aux, per_aux, _ = gf_aux(state["master"], consts, params, batch,
                                  jax.random.PRNGKey(0))
    _, per_base, _ = gf(state["master"], consts, params, batch,
                        jax.random.PRNGKey(0))
    d_clean = float(jnp.abs(per_aux - per_base).max())
    ok("clean 逐图 loss 逐 bit 相同", d_clean == 0.0, f"max_abs Δ={d_clean:.3e}")
    ok("graded loss 变了（aux 生效）",
       float(loss_aux) != 2.12175393 and np.isfinite(float(loss_aux)),
       f"{2.12175393:.6f} -> {float(loss_aux):.6f}")

    print("\nT5 状态存取往返 + 导出：")
    tmp = Path(os.environ.get("TEMP", "/tmp")) / "krea2_jax_state_test.npz"
    T.save_state(tmp, st, tcfg)
    st_r = T.load_state(tmp)
    a1, _ = step(dict(st), params, batch)
    a2, _ = step(st_r, params, batch)
    d = max(float(jnp.abs(a1["master"][n][s] - a2["master"][n][s]).max())
            for n, s in leaf_names(lora))
    ok("往返后再训一步逐 bit 一致", d == 0.0, f"max_abs Δ={d:.3e}")
    out_p = Path(os.environ.get("TEMP", "/tmp")) / "krea2_lora_test.safetensors"
    T.save_lora_k2(out_p, st, tcfg, plans)
    import json as _json
    with open(out_p, "rb") as f:
        hlen = int.from_bytes(f.read(8), "little")
        hdr = _json.loads(f.read(hlen))
    sample_keys = [k for k in hdr if "lora_down" in k][:3]
    ok("导出键名是 torch 模块路径",
       any("lora_unet_blocks_0_attn_wq" in k for k in hdr)
       and any("lora_unet_tproj_1" in k for k in hdr)
       and any("lora_unet_txtfusion_refiner_blocks_0_attn_wo" in k for k in hdr),
       f"样例 {sample_keys}")
    tmp.unlink(missing_ok=True)
    tmp.with_suffix(".json").unlink(missing_ok=True)
    out_p.unlink(missing_ok=True)

    print(f"\n*** {'通过' if not _bad else '失败'} ***"
          + ("" if not _bad else f"  失败项: {_bad}"))
    return 0 if not _bad else 1


if __name__ == "__main__":
    sys.exit(main())
