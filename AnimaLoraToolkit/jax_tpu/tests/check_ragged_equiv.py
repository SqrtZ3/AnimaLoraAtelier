r"""闸门：`forward_ragged`（批/量化分桶布局）≡ `forward_packed`（NaViT 打包布局）。

## 这道闸门证明什么

两条路线共用 `anima_jax._forward_core`，差别只有两处：
  * `mod_bcast`：打包是 `jnp.take(h, mod_index)` -> [ΣN, 3D] 物化；
                 分桶是 `h[:, None, :]`          -> [G, 1, 3D] 广播。
  * 注意力闭包的形状：[S, H, D] vs [G, L, H, D]。

如果两者在真 token 上逐元素一致，那么分桶路线**自动继承**打包路线已验证的
全部数值口径（RoPE NTK 系数、输出重排、cos 在前、LoRA scaling/init 等），
不需要再对 PyTorch 重做一次对拍。

用稠密注意力（不用 splash）：本闸门要证的是**布局**等价，不是内核等价；
内核那一层由 check_splash_blockdiag 与真机 R1/R2 负责。CPU 上秒级完成。

## 为什么要显式检查"填充位置不参与"

分桶布局里每图尾部有量化填充 token。它们作为 **key** 会污染同图真 token
（打包路径同样的坑，本地实测污染 max_abs=5.4，**静默**）。所以这里除了
等价性，还单独验一条行为判据：动填充区的输入，真 token 输出必须逐 bit 不变。

跑法：
    <jaxenv-python> check_ragged_equiv.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax                                                    # noqa: E402
import jax.numpy as jnp                                       # noqa: E402

import anima_jax as A                                         # noqa: E402

RANK = 8
TXT = 128            # 文本槽长度（本闸门只看布局，取小值省时间）


def make_params(key, cfg, dtype=jnp.float32):
    D, C, R = cfg.model_channels, cfg.crossattn_dim, cfg.adaln_lora_dim
    F = int(D * cfg.mlp_ratio)
    ks = iter(jax.random.split(key, 16 + cfg.num_blocks * 16))
    w = lambda *s: jax.random.normal(next(ks), s, dtype) * 0.05
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


def grid_for(n):
    h = max(d for d in range(1, int(n ** 0.5) + 1) if n % d == 0)
    return h, n // h


def rowcol(reals, L):
    rows = np.zeros((len(reals), L), np.int32)
    cols = np.zeros((len(reals), L), np.int32)
    for g, r in enumerate(reals):
        h, w = grid_for(r)
        rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        rows[g, :r], cols[g, :r] = rr.reshape(-1), cc.reshape(-1)
    return rows, cols


def dense_bucket_attn(seg=None):
    """批维稠密注意力。seg [G, L]（0=真 1=填充）给了就按段隔离。"""
    def attn(q, k, v):
        d = q.shape[-1]
        logits = jnp.einsum("gshd,gthd->ghst", q, k).astype(jnp.float32) / np.sqrt(d)
        if seg is not None:
            allow = seg[:, :, None] == seg[:, None, :]
            logits = jnp.where(allow[:, None, :, :], logits, -jnp.inf)
        w = jax.nn.softmax(logits, axis=-1).astype(v.dtype)
        return jnp.einsum("ghst,gthd->gshd", w, v)
    return attn


def packed_only(reals, L, cfg, params, loras, tok, t, ctx, seg_np, chunk):
    """只走打包路线，可选 chunk（AdaLN 按 chunk 广播）。返回 [G*L, out_dim]。"""
    G = len(reals)
    rows, cols = rowcol(reals, L)
    coarse = [L] * G
    mod_index = jnp.asarray(np.repeat(np.arange(G, dtype=np.int32), L))
    fine = np.where(seg_np == 0, np.arange(G)[:, None], -1).reshape(-1)
    self_bias = jnp.asarray(np.where(fine[:, None] == fine[None, :], 0.0, -np.inf),
                            jnp.float32)
    cross_bias = A.block_diag_bias(coarse, [TXT] * G)
    return A.forward_packed(
        params, cfg, tok.reshape(G * L, -1), t, ctx.reshape(G * TXT, -1),
        jnp.asarray(rows.reshape(-1)), jnp.asarray(cols.reshape(-1)), mod_index,
        lambda q, k, v: A.attention_dense(q, k, v, self_bias),
        lambda q, k, v: A.attention_dense(q, k, v, cross_bias),
        loras=loras, remat="none", chunk=chunk)


def packed_only_bar(reals, L, cfg, params, loras, tok, t, ctx, seg_np, barrier):
    """打包路线 + 可选 optimization_barrier（走**展开**路径 —— barrier 只在展开时
    有意义，scan 本来就把跨块提升挡住了）。"""
    G = len(reals)
    rows, cols = rowcol(reals, L)
    mod_index = jnp.asarray(np.repeat(np.arange(G, dtype=np.int32), L))
    fine = np.where(seg_np == 0, np.arange(G)[:, None], -1).reshape(-1)
    self_bias = jnp.asarray(np.where(fine[:, None] == fine[None, :], 0.0, -np.inf),
                            jnp.float32)
    cross_bias = A.block_diag_bias([L] * G, [TXT] * G)
    return A.forward_packed(
        params, cfg, tok.reshape(G * L, -1), t, ctx.reshape(G * TXT, -1),
        jnp.asarray(rows.reshape(-1)), jnp.asarray(cols.reshape(-1)), mod_index,
        lambda q, k, v: A.attention_dense(q, k, v, self_bias),
        lambda q, k, v: A.attention_dense(q, k, v, cross_bias),
        loras=loras, remat="none", barrier=barrier)


def run(reals, L, cfg, params, loras, tok, t, ctx, seg_np):
    """同一批输入分别走两条路线，返回 (ragged_out_flat, packed_out)。"""
    G = len(reals)
    rows, cols = rowcol(reals, L)
    # ── 分桶 ──
    out_r = A.forward_ragged(
        params, cfg, tok, t, ctx, jnp.asarray(rows), jnp.asarray(cols),
        dense_bucket_attn(jnp.asarray(seg_np)),
        dense_bucket_attn(None), loras=loras, remat="none")
    # ── 打包 ──
    coarse = [L] * G
    mod_index = jnp.asarray(np.repeat(np.arange(G, dtype=np.int32), L))
    # 打包侧的精细段号：真 token -> 图号 i；填充 -> 独立段号（彼此可见，行不空）
    fine = np.where(seg_np == 0, np.arange(G)[:, None], -1).reshape(-1)
    self_bias = jnp.asarray(np.where(fine[:, None] == fine[None, :], 0.0, -np.inf),
                            jnp.float32)
    cross_bias = A.block_diag_bias(coarse, [TXT] * G)
    self_p = lambda q, k, v: A.attention_dense(q, k, v, self_bias)
    cross_p = lambda q, k, v: A.attention_dense(q, k, v, cross_bias)
    out_p = A.forward_packed(
        params, cfg, tok.reshape(G * L, -1), t, ctx.reshape(G * TXT, -1),
        jnp.asarray(rows.reshape(-1)), jnp.asarray(cols.reshape(-1)), mod_index,
        self_p, cross_p, loras=loras, remat="none")
    return out_r.reshape(G * L, -1), out_p


def main():
    cfg = A.AnimaConfig(model_channels=128, num_blocks=3, num_heads=2,
                        crossattn_dim=64, adaln_lora_dim=32)
    reals, L = [96, 60], 128
    G = len(reals)
    params = make_params(jax.random.PRNGKey(0), cfg)
    # B 填成非零：B=0 时 LoRA 净增量为 0，等于没测 LoRA 支路
    flat = A.init_lora(jax.random.PRNGKey(1), cfg, RANK, dtype=jnp.float32)
    loras = {k: {"a": v["a"],
                 "b": jax.random.normal(jax.random.PRNGKey(hash(k) % 2**31),
                                        v["b"].shape, jnp.float32) * 0.05}
             for k, v in flat.items()}

    ks = jax.random.split(jax.random.PRNGKey(2), 3)
    tok = jax.random.normal(ks[0], (G, L, cfg.in_dim), jnp.float32)
    keep = np.arange(L)[None, :] < np.asarray(reals)[:, None]
    tok = tok * jnp.asarray(keep, jnp.float32)[..., None]   # 填充位置输入置 0
    t = jax.random.uniform(ks[1], (G,), jnp.float32)
    ctx = jax.random.normal(ks[2], (G, TXT, cfg.crossattn_dim), jnp.float32)
    seg_np = (~keep).astype(np.int32)

    fails = []

    def check(name, got, tol, extra=""):
        ok = got < tol
        print(f"  [{'OK ' if ok else 'BAD'}] {name:<34} {got:.3e} (阈值 {tol:.0e}){extra}")
        if not ok:
            fails.append(name)

    print(f"配置：{cfg.num_blocks} 块 D={cfg.model_channels} | "
          f"G={G} L={L} 真实 token={reals} | TXT={TXT}\n")

    print("① 两条路线在真 token 上等价（主判据）：")
    a, b = run(reals, L, cfg, params, loras, tok, t, ctx, seg_np)
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    m = keep.reshape(-1)
    den = max(np.abs(b[m]).max(), 1e-12)
    check("整模 rel", np.abs(a[m] - b[m]).max() / den, 1e-5)
    off = 0
    for g, r in enumerate(reals):
        s = slice(g * L, g * L + r)
        print(f"      图{g}（{r} token）rel="
              f"{np.abs(a[s] - b[s]).max() / den:.3e}")
        off += r

    print("\n② 分桶布局：填充 token 不污染真 token（行为判据，不是数值接近）：")
    tok2 = tok.at[:, reals[0]:].set(
        jax.random.normal(jax.random.PRNGKey(9), (G, L - reals[0], cfg.in_dim)))
    # 只有第 0 图的填充区被改（第 1 图真实长度更短，故其部分真 token 也在该区间内，
    # 这里只比第 0 图，保证是"纯填充扰动"）
    a2, _ = run(reals, L, cfg, params, loras, tok2, t, ctx, seg_np)
    a2 = np.asarray(a2, np.float64)
    check("图0 真 token 逐 bit 不变", np.abs(a[:reals[0]] - a2[:reals[0]]).max(), 1e-30,
          "  ← =0 才算隔离住")

    print("\n③ 反证：这个判据有分辨力（关掉段隔离就该被污染）：")
    def run_bare(tk):
        rows, cols = rowcol(reals, L)
        return np.asarray(A.forward_ragged(
            params, cfg, tk, t, ctx, jnp.asarray(rows), jnp.asarray(cols),
            dense_bucket_attn(None), dense_bucket_attn(None),
            loras=loras, remat="none"), np.float64)
    c1, c2 = run_bare(tok), run_bare(tok2)
    contam = np.abs(c1[0, :reals[0]] - c2[0, :reals[0]]).max()
    ok = contam > 1e-6
    print(f"  [{'OK ' if ok else 'BAD'}] 无隔离时的污染                {contam:.3e} "
          f"(应 >1e-6){'' if ok else '  ← 判据失效'}")
    if not ok:
        fails.append("反证")

    print("\n④ 打包布局的 chunk 化调制 ≡ 朴素逐 token 调制（NaViT 语义不变）：")
    base = np.asarray(packed_only(reals, L, cfg, params, loras, tok, t, ctx,
                                  seg_np, None), np.float64)
    for q in (32, 64, 128):
        if L % q:
            continue
        got = np.asarray(packed_only(reals, L, cfg, params, loras, tok, t, ctx,
                                     seg_np, q), np.float64)
        d = np.abs(got - base).max()
        ok = d == 0.0
        print(f"  [{'OK ' if ok else 'BAD'}] chunk={q:<4} 逐 bit 一致           "
              f"      max_abs={d:.3e}")
        if not ok:
            fails.append(f"chunk={q}")

    print("\n⑤ optimization_barrier 是恒等算子（前向 + LoRA 梯度都不该变）：")
    # barrier 只给 XLA 加调度约束（阻止 AdaLN 调制跨块提前算），语义上是 identity。
    # **梯度也要验**：barrier 有自己的转置规则，只验前向等于没验反向。
    def with_barrier(bar, want_grad):
        fn = lambda lo: packed_only_bar(reals, L, cfg, params, lo, tok, t, ctx,
                                        seg_np, bar)
        if not want_grad:
            return np.asarray(fn(loras), np.float64)
        g = jax.grad(lambda lo: jnp.sum(fn(lo) ** 2))(loras)
        return np.concatenate([np.asarray(g[k][s], np.float64).ravel()
                               for k in sorted(g) for s in ("a", "b")])

    for want_grad, label in ((False, "前向"), (True, "LoRA 梯度")):
        d = np.abs(with_barrier(True, want_grad) - with_barrier(False, want_grad)).max()
        ok = d == 0.0
        print(f"  [{'OK ' if ok else 'BAD'}] barrier {label:<9} 逐 bit 一致"
              f"        max_abs={d:.3e}")
        if not ok:
            fails.append(f"barrier-{label}")

    print(f"\n*** {'通过' if not fails else '失败：' + ', '.join(fails)} ***")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
