"""闸门 ⑧：`lax.scan` 路径 ≡ 展开路径（前向 + LoRA 梯度）。

scan 路径是为了修显存引入的（真机归因：展开时 NaViT 路径要 1.01 MB/token，
主因是 AdaLN 调制不依赖 x、28 个块的 [ΣN,3D] 被 XLA 提前调度成同时活着；
见 anima_jax.stack_blocks 的注释）。但它改了两件容易出错的事：

  * 权重从 list-of-dict 堆成 [L, ...]  —— 堆错顺序 = 块顺序错，**不报错**
  * LoRA 的键从 "blocks.i.target" 变成块内相对 —— 配错 = LoRA 挂到别的块上

两者都只会让结果悄悄不对。所以逐项比：

  C1 前向：scan vs 展开，逐元素
  C2 LoRA 梯度：scan vs 展开，逐元素（训练真正用的是它）
  C3 scan 下 full/every2/none 三档彼此一致，且 dots 被 fail-fast 拦住
     （dots 在 scan 下是 CLI 可达的 OOM 死路，见 anima_jax.resolve_remat）
  C4 stack/unstack 往返：unstack_loras(stack_loras(x)) == x（导出链路靠它）

用法（jax 解释器）：python check_scan_equiv.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import anima_jax as A                                          # noqa: E402
import attention as AT                                         # noqa: E402
import jax                                                     # noqa: E402
import jax.numpy as jnp                                        # noqa: E402

VIS, TXT = [512, 256, 256], [128, 128, 128]
NB, RANK = 4, 4
_bad = []


def ok(name, cond, detail=""):
    print(f"  [{'OK ' if cond else 'BAD'}] {name:<30} {detail}")
    if not cond:
        _bad.append(name)


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-12))


def main() -> int:
    cfg = A.AnimaConfig(num_blocks=NB, model_channels=256, num_heads=2,
                        crossattn_dim=128, adaln_lora_dim=32)
    D, C, R = cfg.model_channels, cfg.crossattn_dim, cfg.adaln_lora_dim
    Fd = int(D * cfg.mlp_ratio)
    ks = iter(jax.random.split(jax.random.PRNGKey(0), 8 + NB * 16))
    w = lambda *s: jax.random.normal(next(ks), s, jnp.float32) * 0.02
    one = lambda n: jnp.ones((n,), jnp.float32)
    # 每块权重刻意用不同随机值 -> 堆叠顺序错会被 C1/C2 抓到
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
    } for _ in range(NB)]
    params = {"x_embedder": w(D, cfg.in_dim), "t_embedder_1": w(D, D),
              "t_embedder_2": w(3 * D, D), "t_embedding_norm": one(D),
              "final_adaln_1": w(R, D), "final_adaln_2": w(2 * D, R),
              "final_linear": w(cfg.out_dim, D), "blocks": blocks}

    lora = A.init_lora(jax.random.PRNGKey(1), cfg, RANK, dtype=jnp.float32)
    # B 零初始化 -> ΔW=0，验不出"挂错块"。灌上随机值。
    kb = jax.random.split(jax.random.PRNGKey(2), len(lora))
    lora = {n: {"a": m["a"], "b": jax.random.normal(kb[i], m["b"].shape, jnp.float32) * 0.1}
            for i, (n, m) in enumerate(sorted(lora.items()))}

    S, T = sum(VIS), sum(TXT)
    tok = jax.random.normal(jax.random.PRNGKey(3), (S, cfg.in_dim), jnp.float32)
    ctx = jax.random.normal(jax.random.PRNGKey(4), (T, C), jnp.float32)
    t = jnp.array([0.2, 0.5, 0.9])
    mi = jnp.asarray(AT.segment_ids(VIS))
    rows = jnp.asarray(np.concatenate([np.arange(n) // 16 for n in VIS]).astype(np.int32))
    cols = jnp.asarray(np.concatenate([np.arange(n) % 16 for n in VIS]).astype(np.int32))
    sf, cf = AT.make_dense_attn(VIS, VIS), AT.make_dense_attn(VIS, TXT)

    stacked_p = A.stack_blocks(params)
    stacked_l = A.stack_loras(lora, NB)

    def fwd_unroll(lo):
        return A.forward_packed(params, cfg, tok, t, ctx, rows, cols, mi, sf, cf,
                                loras=lo, remat="full")

    def fwd_scan(lo, remat="full"):
        return A.forward_packed(stacked_p, cfg, tok, t, ctx, rows, cols, mi, sf, cf,
                                loras=lo, remat=remat)

    print("C1 前向：scan vs 展开")
    r = rel(fwd_scan(stacked_l), fwd_unroll(lora))
    ok("整模输出", r < 1e-5, f"rel={r:.3e}")

    print("\nC2 LoRA 梯度：scan vs 展开（训练真正用的路径）")
    gu = jax.grad(lambda lo: jnp.sum(fwd_unroll(lo) ** 2))(lora)
    gs = jax.grad(lambda lo: jnp.sum(fwd_scan(lo) ** 2))(stacked_l)
    gs_flat = A.unstack_loras(gs, NB)
    worst = max((rel(gs_flat[n][s], gu[n][s]), f"{n}.{s}")
                for n in gu for s in ("a", "b"))
    ok("逐层梯度", worst[0] < 1e-4, f"最差 {worst[1]} rel={worst[0]:.3e}")

    print("\nC3 scan 下三档 remat 一致 + dots fail-fast")
    base = fwd_scan(stacked_l, "full")
    gb = jax.grad(lambda lo: jnp.sum(fwd_scan(lo, "full") ** 2))(stacked_l)
    for rm in ("every2", "none"):
        r1 = rel(fwd_scan(stacked_l, rm), base)
        g2 = jax.grad(lambda lo: jnp.sum(fwd_scan(lo, rm) ** 2))(stacked_l)
        r2 = max(rel(g2[t_][s], gb[t_][s]) for t_ in gb for s in ("a", "b"))
        ok(f"remat={rm}", r1 < 1e-6 and r2 < 1e-4, f"前向 rel={r1:.1e} 梯度 rel={r2:.1e}")
    # `dots` 在 scan 下**必须报错**而不是"跑得出同一个数"：dots_saveable 保留的激活
    # 会按 num_blocks 次迭代堆叠（≈1MB/token），budget 16384 必 OOM。这里段长很小所以
    # 数值上跑得通，正是这条 CLI 可达死路以前藏得住的原因（run_train.py --remat dots
    # 默认就走 scan），所以判据从"数值一致"改成"拦住了"。见 anima_jax.resolve_remat。
    try:
        fwd_scan(stacked_l, "dots")
        ok("remat=dots 被拦住", False, "未抛异常")
    except ValueError as e:
        ok("remat=dots 被拦住", "dots" in str(e), f"{str(e)[:34]}...")

    print("\nC4 stack/unstack 往返")
    back = A.unstack_loras(stacked_l, NB)
    d = max(float(jnp.abs(back[n][s] - lora[n][s]).max()) for n in lora for s in ("a", "b"))
    ok("逐 bit 还原", d == 0.0 and set(back) == set(lora), f"max_abs Δ={d:.1e}，"
       f"键数 {len(back)}/{len(lora)}")

    print(f"\n*** {'通过' if not _bad else '失败'} ***"
          + ("" if not _bad else f"  失败项: {_bad}"))
    return 0 if not _bad else 1


if __name__ == "__main__":
    sys.exit(main())
