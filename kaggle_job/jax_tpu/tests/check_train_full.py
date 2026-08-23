r"""闸门⑪：**全功能**训练步（LoKr+DoRA + 三峰/自适应 t + Huber(snr) + immiscible
+ eisbach + ΔFM(vecor) + spectral(FFT+小波) + rank/module dropout + 8 卡分片）。

`check_train_loop.py` 证的是"打包路线能学"；这个证的是"**用户 yaml 里那一整套
功能**同时打开时还能学，而且每一条都真的接上了"。差别很实在：一个开关接错了
（键名拼错、树的形状不匹配、被 XLA DCE 掉）通常不报错，只是那一项静默失效。

跑法（CPU 上伪造 8 卡）：

    XLA_FLAGS=--xla_force_host_platform_device_count=8 python check_train_full.py

## 断言清单（每一条都在防一种"不报错的失效"）

  T1 step-0 中立    LoKr 的 w2_b=0 且 DoRA 的 scale 初值 =‖W‖ -> 接不接适配器，
                    loss 必须相同（到 fp32 舍入）。破了说明初始化偏了。
  T2 梯度非零       w2_b / dora 必须有梯度。恒 0 = 适配器没接上（键名不匹配时
                    `_lora()` 返回 None，前向照跑、反向被 DCE、步时还变快）。
  T3 rank 掩码      被 reg_dims 掩掉的那些 rank 通道，梯度必须**恒 0**。
  T4 跨卡           改一张卡的数据，loss 必须变（否则 shard_map 没接对）。
  T5 填充不参与     改填充区的 latent，loss 必须**逐 bit 不变**。
  T6 逐图等权       pack 的 loss 精确 == 两张图各自单跑的算术平均（8:1 token 差）
                    （全局 token 加权的话大图会顶 4 张小图）。
  T7 aux 都在动     逐个关掉 eisbach/ΔFM/spectral，loss 必须变 —— 防"开关接了但
                    数值恒等于 0"。
  T8 优化器走了     apply_update 之后参数必须动，且 master 仍是 fp32。
  T9 自适应闭环     喂 200 步反馈后 factors() 必须不再全 1。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax                                                    # noqa: E402
import jax.numpy as jnp                                       # noqa: E402
import numpy as np                                            # noqa: E402

import adapters as AD                                         # noqa: E402
import anima_jax as A                                         # noqa: E402
import auxloss as X                                               # noqa: E402
import flow as F                                              # noqa: E402
import optim as O                                             # noqa: E402
import sched as S                                             # noqa: E402
import train as T                                             # noqa: E402
from packing import Layout, Pack                              # noqa: E402

OK = FAIL = 0


def check(name, cond, detail=""):
    global OK, FAIL
    tag = "[OK ]" if cond else "[FAIL]"
    if cond:
        OK += 1
    else:
        FAIL += 1
    print(f"  {tag} {name:<34} {detail}")


# ── 玩具底模（形状对，数值随机）───────────────────────────────────────────────
def toy_model(seed=0, cross=32):
    cfg = A.AnimaConfig(num_blocks=4, model_channels=64, num_heads=2,
                        crossattn_dim=cross, adaln_lora_dim=16, mlp_ratio=2.0)
    D, C = cfg.model_channels, cfg.crossattn_dim
    H = int(D * cfg.mlp_ratio)
    k = jax.random.PRNGKey(seed)
    n = [0]

    def r(*shape):
        n[0] += 1
        return jax.random.normal(jax.random.fold_in(k, n[0]), shape,
                                 jnp.float32) * 0.05

    blocks = []
    for _ in range(cfg.num_blocks):
        blocks.append({
            "self_attn": {"q_proj": r(D, D), "k_proj": r(D, D), "v_proj": r(D, D),
                          "output_proj": r(D, D), "q_norm": jnp.ones(cfg.head_dim),
                          "k_norm": jnp.ones(cfg.head_dim)},
            "cross_attn": {"q_proj": r(D, D), "k_proj": r(D, C), "v_proj": r(D, C),
                           "output_proj": r(D, D), "q_norm": jnp.ones(cfg.head_dim),
                           "k_norm": jnp.ones(cfg.head_dim)},
            "mlp": {"layer1": r(H, D), "layer2": r(D, H)},
            "adaln_self_1": r(cfg.adaln_lora_dim, D),
            "adaln_self_2": r(3 * D, cfg.adaln_lora_dim),
            "adaln_cross_1": r(cfg.adaln_lora_dim, D),
            "adaln_cross_2": r(3 * D, cfg.adaln_lora_dim),
            "adaln_mlp_1": r(cfg.adaln_lora_dim, D),
            "adaln_mlp_2": r(3 * D, cfg.adaln_lora_dim),
        })
    params = {"x_embedder": r(D, cfg.in_dim), "blocks": blocks,
              "t_embedder_1": r(cfg.adaln_lora_dim, D),
              "t_embedder_2": r(3 * D, cfg.adaln_lora_dim),
              "t_embedding_norm": jnp.ones(D),
              "final_adaln_1": r(cfg.adaln_lora_dim, D),
              "final_adaln_2": r(2 * D, cfg.adaln_lora_dim),
              "final_linear": r(cfg.out_dim, D)}
    return A.stack_blocks(params), cfg


def make_packs(devices, budget=512, quantum=256, txt=128, grids=((8, 16), (8, 8))):
    """两段：第一段 128 token(8x16) 量化到 256，第二段 64 token(8x8) 量化到 256。"""
    segs = tuple(quantum for _ in grids)
    rest = budget - sum(segs)
    seg_lens = segs + ((rest,) if rest else ())
    order = tuple(sorted(seg_lens, reverse=True))
    layout = Layout(budget, order, txt)
    packs = []
    for _ in range(devices):
        real = [g[0] * g[1] for g in grids] + ([0] if rest else [])
        gr = list(grids) + ([(0, 0)] if rest else [])
        packs.append(Pack(layout, [object()] * len(real), real, gr))
    return packs, layout


def toy_batch(packs, layout, mcfg, t, seed=0):
    rng = np.random.RandomState(seed)
    lats, ctxs = [], []
    for p in packs:
        ll, cc = [], []
        for i, seg in enumerate(layout.seg_lens):
            r = p.real_lens[i]
            ll.append(None if r == 0 else
                      rng.randn(r, mcfg.out_dim).astype(np.float32) * 0.5)
            cc.append(rng.randn(layout.txt_len, mcfg.crossattn_dim).astype(np.float32)
                      * 0.1)
        lats.append(ll)
        ctxs.append(cc)
    return T.assemble_batch(packs, lats, ctxs, t, mcfg, jnp.float32), lats, ctxs


def main() -> int:
    devices = len(jax.devices())
    print(f"设备 {devices} 个（{jax.devices()[0].platform}）\n")
    params, mcfg = toy_model()
    packs, layout = make_packs(devices)
    G = layout.n_seg

    acfg = AD.AdapterConfig(
        kind="lokr", variant="dora", rank=8, alpha=8.0, factor=4,
        rank_dropout=0.05, module_dropout=0.03,
        reg_dims={r".*blocks\.[23]\..*": 4}, reg_alphas={r".*blocks\.[23]\..*": 4})
    fcfg = F.FlowConfig(t_mode="mixed_logsnr_three", flow_shift=3.0,
                        mix_low_prob=0.12, mix_high_prob=0.48,
                        logsnr_mu=-6.0, logsnr_sigma=2.4, t_min=0.05,
                        stratified=True, loss_type="huber", huber_c=0.2,
                        huber_schedule="snr", huber_snr_clamp_max=2.0,
                        immiscible_k=4)
    xcfg = X.AuxConfig(eisbach_lambda=0.15, dfm_lambda=0.05,
                       spectral_enabled=True, spectral_lambda=0.065,
                       spectral_use_wavelet=True, spectral_wavelet_lambda=0.20,
                       spectral_t_gate=0.55, canvas_hw=(8, 16))
    tcfg = T.TrainConfig(adapter=acfg,
                         targets=("self_attn.q_proj", "mlp.layer1"),
                         remat="full", flow=fcfg, aux=xcfg,
                         adamw=O.AdamWConfig(lr=2e-3, b1=0.98, b2=0.999,
                                             weight_decay=1e-5, snr_power=1.5))
    lora, consts, plans = T.init_adapter(jax.random.PRNGKey(0), mcfg, tcfg, params)
    mesh = _mesh(devices)
    gf = T.make_grad_fn(mcfg, tcfg, plans, layout, mesh, interpret=True)

    rng = np.random.RandomState(0)
    t = F.sample_t_np(rng, devices * G, fcfg)
    batch, lats, ctxs = toy_batch(packs, layout, mcfg, t)
    key = jax.random.PRNGKey(7)

    print("T1 step-0 中立（LoKr w2_b=0 且 DoRA scale=‖W‖）")
    ecfg = T.eval_config(tcfg)
    ef = T.make_grad_fn(mcfg, ecfg, plans, layout, mesh, interpret=True, grad=False)
    l_on = float(ef(lora, consts, params, batch, key)[0])
    zero = jax.tree.map(lambda x: x, lora)
    ef0 = T.make_grad_fn(mcfg, replace_adapter_off(ecfg), plans, layout, mesh,
                         interpret=True, grad=False)
    l_off = float(ef0(zero, consts, params, batch, key)[0])
    check("接/不接适配器 loss 相同", abs(l_on - l_off) < 1e-4 * max(abs(l_on), 1),
          f"{l_on:.6f} vs {l_off:.6f}")

    print("\nT2/T3 梯度：非零 + 掩掉的 rank 恒 0")
    loss, per, grads = gf(lora, consts, params, batch, key)
    for name in tcfg.targets:
        gb = float(jnp.sum(jnp.abs(grads[name]["w2b"])))
        gd = float(jnp.sum(jnp.abs(grads[name]["dora"])))
        check(f"{name} w2b/dora 有梯度", gb > 0 and gd > 0, f"{gb:.3e} / {gd:.3e}")
        m = consts[name]["rmask"]
        off = float(jnp.max(jnp.abs(grads[name]["w2b"] * (1 - m)[:, :, None])))
        check(f"{name} 掩掉通道梯度=0", off == 0.0, f"max={off:.1e}")

    print("\nT4 跨卡：改一张卡的数据 loss 必须变")
    b2 = dict(batch)
    lat = np.asarray(batch["latent"]).copy()
    lat[-1] += 1.0
    b2["latent"] = jnp.asarray(lat)
    check("最后一卡数据改动生效",
          abs(float(gf(lora, consts, params, b2, key)[0]) - float(loss)) > 1e-6,
          f"{float(loss):.6f} -> {float(gf(lora, consts, params, b2, key)[0]):.6f}")

    print("\nT5 填充区不参与 loss")
    b3 = dict(batch)
    lat = np.asarray(batch["latent"]).copy()
    mask = np.asarray(batch["loss_mask"])
    lat[mask == 0] += 100.0
    b3["latent"] = jnp.asarray(lat)
    l3 = float(gf(lora, consts, params, b3, key)[0])
    check("改填充区 loss 逐 bit 不变", l3 == float(loss), f"{float(loss):.6f} vs {l3:.6f}")

    print("\nT6 逐图等权（大图不因 token 多而占更大权重）")
    _check_equal_weight(mcfg, tcfg, plans, lora, consts, params, mesh)

    print("\nT7 每个 aux 都在动")
    base = float(loss)
    for name, off in (("eisbach", dict(eisbach_lambda=0.0)),
                      ("ΔFM(vecor)", dict(dfm_lambda=0.0)),
                      ("spectral", dict(spectral_enabled=False))):
        import dataclasses
        c2 = dataclasses.replace(tcfg, aux=dataclasses.replace(xcfg, **off))
        g2 = T.make_grad_fn(mcfg, c2, plans, layout, mesh, interpret=True)
        l2 = float(g2(lora, consts, params, batch, key)[0])
        check(f"关掉 {name} loss 变了", abs(l2 - base) > 1e-7, f"{base:.6f} -> {l2:.6f}")

    print("\nT8 优化器：参数动了、master 仍是 fp32")
    state = O.init_state(lora)
    before = float(jnp.sum(jnp.abs(state["master"][tcfg.targets[0]]["w2b"])))
    state, diag = T.apply_update(state, grads, tcfg.adamw)
    after = float(jnp.sum(jnp.abs(state["master"][tcfg.targets[0]]["w2b"])))
    check("w2b 动了", after != before, f"{before:.3e} -> {after:.3e}")
    check("master 是 fp32",
          state["master"][tcfg.targets[0]]["w2b"].dtype == jnp.float32,
          str(state["master"][tcfg.targets[0]]["w2b"].dtype))
    check("gnorm 有限且 >0", 0 < float(diag["gnorm"]) < 1e9,
          f"{float(diag['gnorm']):.4f}")

    print("\nT9 自适应闭环")
    smp = S.AdaptiveTimestepSampler(S.AdaptiveConfig(
        enabled=True, metric="slope", bins=8, burn_in=5, base_mix=0.5,
        ema_decay=0.97, min_factor=0.5, max_factor=2.2))
    r2 = np.random.RandomState(1)
    for i in range(60):
        tt = smp.sample(r2, 16, fcfg, i)
        # 造一个"低 t 还在快速下降、高 t 已饱和"的假反馈
        smp.update(tt, np.where(tt < 0.5, 1.0 / (1 + i * 0.3), 0.5).astype(np.float32))
    f = smp.factors()
    check("factors 不再全 1", float(f.max() - f.min()) > 0.05,
          f"{f.min():.2f}~{f.max():.2f}")
    check("低 t 桶被抬高", float(f[:4].mean()) > float(f[4:].mean()),
          f"低 {f[:4].mean():.2f} vs 高 {f[4:].mean():.2f}")

    print("\nT10 展开路径（吞吐旋钮）≡ scan 路径")
    _check_unrolled(mcfg, tcfg, plans, lora, consts, params, batch, key, layout, mesh,
                    float(loss), grads)

    print(f"\n{'*** 通过 ***' if FAIL == 0 else f'*** {FAIL} 项失败 ***'}"
          f"  ({OK} OK / {FAIL} FAIL)")
    # 这里原来无条件 return 0 —— 闸门打印 FAIL 却以成功退出，跑批脚本一律当绿。
    # 其余 check_*.py 都是 `return 0 if not bad else 1`，这条是漏网的。
    return 0 if FAIL == 0 else 1


def _unstack_params(params):
    """scan 布局 -> 展开布局（`blocks` 变回 list）。`stack_blocks` 的逆。"""
    blk = params["blocks"]
    n = jax.tree.leaves(blk)[0].shape[0]
    return {**params, "blocks": [jax.tree.map(lambda z, i=i: z[i], blk)
                                 for i in range(n)]}


def _check_unrolled(mcfg, tcfg, plans, lora, consts, params, batch, key, layout,
                    mesh, ref_loss, ref_grads):
    """展开路径是 `--unrolled` 打开后训练真正走的那条，必须与 scan 同数学。

    两条断言各防一件事：
      * loss/梯度不一致 -> 展开路径的 LoRA 切片（`_slice_ctx`）或逐块键名错了；
      * 非法组合不报错 -> `--unrolled` 单开会退回朴素展开（真机 budget 16384
        的 full 档就要 20.60G），必须构造期就拦住。
    """
    import dataclasses
    # **在 fp32 上对拍**：这条断言问的是"两条路径是不是同一个数学"，而默认的
    # bf16 前向会把答案淹在舍入里（同一份代码 bf16 下梯度 rel 7e-3，fp32 下 1e-7；
    # 见 tests/README「逐 bit 只在 fp32 下成立」）。ref_loss/ref_grads 是 bf16 的，
    # 所以这里重新取一份 fp32 的 scan 参考。
    up = _unstack_params(params)
    f32 = dataclasses.replace(tcfg, dtype=jnp.float32)
    ref = T.make_grad_fn(mcfg, f32, plans, layout, mesh, interpret=True)
    l0, _, g0 = ref(lora, consts, params, batch, key)
    for name, kw in (("barrier", dict(packed_barrier=True)),
                     ("chunk", dict(packed_chunk=True))):
        c = dataclasses.replace(f32, unrolled=True, **kw)
        gf = T.make_grad_fn(mcfg, c, plans, layout, mesh, interpret=True)
        l2, _, g2 = gf(lora, consts, up, batch, key)
        rel = max(float(jnp.max(jnp.abs(a - b)) / (jnp.max(jnp.abs(b)) + 1e-30))
                  for a, b in zip(jax.tree.leaves(g2), jax.tree.leaves(g0)))
        check(f"展开+{name} loss ≡ scan",
              abs(float(l2) - float(l0)) < 1e-5 * max(abs(float(l0)), 1.0),
              f"{float(l0):.6f} vs {float(l2):.6f}")
        check(f"展开+{name} LoRA 梯度 ≡ scan", rel < 1e-5, f"rel={rel:.2e}")
    try:
        dataclasses.replace(tcfg, unrolled=True)
        check("单开 unrolled 被拒", False, "没有报错")
    except ValueError:
        check("单开 unrolled 被拒", True, "构造期 fail-fast")


def replace_adapter_off(tcfg):
    """把 targets 清空 -> 前向里一个适配器也挂不上（用来做 T1 的对照）。"""
    import dataclasses
    return dataclasses.replace(tcfg, targets=())


def _check_equal_weight(mcfg, tcfg, plans, lora, consts, params, mesh):
    """逐图等权：一个 pack 的 loss 必须**精确**等于两张图各自单跑的算术平均。

    ## 为什么不是"扰动两段看 Δloss 是否相等"

    旧版给两段各加同一个常数扰动、比 Δloss。那个判据有两个毛病，seed=1 上就
    是红的（Δ 4.0% > 2% 阈值），但它红的原因不是归约错了：

      * **阈值低于自己的噪声底。** 把两段做成**同样大**（128 vs 128 token，此时
        两种归约口径预测完全一致、不存在尺寸效应），8 个 seed 的 Δ 比值仍在
        [0.955, 1.037]，最大偏离 4.55% —— 比让它判失败的 4.02% 还大。红的是
        内容噪声，不是权重。（本仓库自己的方法论第一条：没有噪声基线的阈值
        不是判据。）
      * **前提本身不成立。** "给 latent 加常数 = 把两段误差放大同样倍数"是假的：
        扰动要过一遍网络，响应与该图的 token 数、内容都有关。实测 8:1 时比值
        系统性地是 1.16（8 个 seed 全部 > 1.08，远出噪声底），bump 0.05 时更是
        2.18 —— 这些都是真实的、与归约口径无关的非线性。

    两个口径的差距本来有 4~8 倍（4:1 时全局 token 加权预测 Δ 比值 = 4.0），
    所以旧判据**方向上**其实一直是对的（实测 ≈1.0，离 4.0 差得远），只是把
    容差调到了噪声以下。

    ## 现在这个判据是精确的

    块对角注意力下两段互不影响：把一段的 `loss_mask` 清零就等于"只有另一张图"
    （`train.py:195` 的 `valid = den > 0` 会把它排除），而留下那段的每 token 误差
    逐 bit 不变。于是有恒等式

        L(A+B) == (L(A) + L(B)) / 2          ← 逐图等权
        L(A+B) == (N_A·L(A) + N_B·L(B)) / (N_A+N_B)   ← 全局 token 加权（错的那个）

    实测（128 vs 16 token）：对前者 rel ≤ 5e-08（fp32 噪声），对后者 rel 4.1e-02
    —— 判别余量 ~10^6 倍，不再依赖任何拍脑袋的阈值。
    """
    import dataclasses
    packs, layout = make_packs(len(jax.devices()), grids=((8, 16), (4, 4)))
    c = dataclasses.replace(tcfg, aux=X.AuxConfig(), flow=dataclasses.replace(
        tcfg.flow, immiscible_k=1, loss_type="mse"),
        adapter=dataclasses.replace(tcfg.adapter, rank_dropout=0.0,
                                    module_dropout=0.0))
    gf = T.make_grad_fn(mcfg, c, plans, layout, mesh, interpret=True, grad=False)
    key = jax.random.PRNGKey(3)
    t = np.full(len(packs) * layout.n_seg, 0.5, np.float32)
    b0, lats, ctxs = toy_batch(packs, layout, mcfg, t, seed=1)

    m_ab = np.asarray(b0["loss_mask"]).copy()
    n_a, n_b = packs[0].real_lens[0], packs[0].real_lens[1]
    off_b = sum(layout.seg_lens[:1])

    def run(mask):
        b = dict(b0)
        b["loss_mask"] = jnp.asarray(mask)
        return float(gf(lora, consts, params, b, key)[0])

    m_a = m_ab.copy(); m_a[:, off_b:off_b + n_b] = 0        # 只留大图
    m_b = m_ab.copy(); m_b[:, 0:n_a] = 0                    # 只留小图
    l_a, l_b, l_ab = run(m_a), run(m_b), run(m_ab)

    # 两张图的 loss 必须真的不一样，否则两个假设重合、这条断言是空的
    check("两图 loss 可区分（判据非空）", abs(l_a - l_b) > 1e-3 * max(abs(l_a), 1e-9),
          f"L_大 {l_a:.6f} vs L_小 {l_b:.6f}")

    eq = (l_a + l_b) / 2                                    # 逐图等权
    tok = (n_a * l_a + n_b * l_b) / (n_a + n_b)             # 全局 token 加权（错的）
    r_eq = abs(l_ab - eq) / max(abs(eq), 1e-9)
    r_tok = abs(l_ab - tok) / max(abs(tok), 1e-9)
    check("pack loss == 两图单跑的算术平均", r_eq < 1e-6,
          f"L(A+B) {l_ab:.8f} vs (L_A+L_B)/2 {eq:.8f}  rel={r_eq:.2e}")
    check("与'全局 token 加权'口径可区分", r_tok > 100 * max(r_eq, 1e-8),
          f"离 token 加权 {tok:.8f} 有 rel={r_tok:.2e}（判别余量 {r_tok / max(r_eq, 1e-8):.0e}x）")


def _mesh(devices):
    from jax.sharding import Mesh
    return Mesh(np.asarray(jax.devices()[:devices]).reshape(devices), ("d",))


if __name__ == "__main__":
    raise SystemExit(main())
