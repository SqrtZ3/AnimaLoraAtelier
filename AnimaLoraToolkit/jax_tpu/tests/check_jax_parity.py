"""对拍闸门 · 第二步（jax 侧）：加载同一份真权重跑 anima_jax，与 torch 参考逐级比。

先跑 `dump_torch_ref.py` 产出 `_ref/anima_ref.npz`，再跑本脚本。

三档比对，从细到粗：
  ② 逐算子   —— 偏差落在哪个算子
  ③ 逐块     —— 跨块累积放大，还是某块突变
  ① 整模     —— 主判据，退出码由它决定

**这是改动 anima_jax.py 之后唯一的正确性闸门，任何改动后都要重跑。**
历史上它抓到过两个静默错（都不报异常）：
  * RoPE 漏 NTK theta 缩放 -> 位置编码频率悄悄变了
  * 输出漏 (ph pw pt c)->(c pt ph pw) 重排 -> 统计量完全正常、逐元素全错(rel 1.4)

用法（需要 jax 的解释器；本地是独立 venv，见 tests/README.md）：
    python check_jax_parity.py [--ckpt 路径] [--ref 目录] [--tol 1e-4]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import anima_jax as A                                        # noqa: E402
import jax.numpy as jnp                                      # noqa: E402

DEFAULT_CKPT = (r"D:\ArtificialIntelligence\ComfyUI-aki-v1.5\models"
                r"\diffusion_models\anima-base-v1.0.safetensors")

# 逐算子的**分项阈值**。默认取 --tol；这两项放宽是有依据的，不是为了让测试变绿：
# q/k 是 q_norm/k_norm(RMSNorm) 之后的小幅值张量（std≈0.18），而它们的输入
# pre_selfattn std≈5.2 —— RMSNorm 除以一个小 RMS，会把输入侧 1.3e-5 的相对误差
# 放大约一个量级。判据是**下游是否收敛回来**：selfattn_out 实测 4.0e-5、
# 逐块 4.0e-5、整模 4.6e-5，都在 1e-4 内 -> 属 fp32 噪声传播，非结构错。
# 若哪天 q/k 超过 1e-3，或超标后下游**不再收敛**，那才是真问题。
STAGE_TOL = {"q": 1e-3, "k": 1e-3}


def rel_err(got, exp) -> float:
    got, exp = np.asarray(got, np.float64), np.asarray(exp, np.float64)
    if got.shape != exp.shape:
        return float("nan")
    return float(np.abs(got - exp).max() / max(np.abs(exp).max(), 1e-12))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--ref", default=str(Path(__file__).parent / "_ref"))
    ap.add_argument("--tol", type=float, default=1e-4)
    a = ap.parse_args()

    ref_path = Path(a.ref) / "anima_ref.npz"
    if not ref_path.exists():
        print(f"找不到 {ref_path} —— 先在 torch 环境跑 dump_torch_ref.py")
        return 2
    ref = np.load(ref_path)
    vis, txt = list(ref["vis"]), list(ref["txt"])
    G = len(vis)

    params, cfg = A.load_safetensors_anima(a.ckpt, dtype=jnp.float32)
    print(f"cfg: blocks={cfg.num_blocks} channels={cfg.model_channels} "
          f"heads={cfg.num_heads} rope_ratio={cfg.rope_h_ratio}")

    tokens = jnp.asarray(ref["tokens"])
    ctx = jnp.asarray(ref["ctx"])
    t = jnp.asarray(ref["t"])
    rows, cols = jnp.asarray(ref["rows"]), jnp.asarray(ref["cols"])
    mod_index = jnp.asarray(np.repeat(np.arange(G), vis))
    self_bias = A.block_diag_bias(vis)
    cross_bias = A.block_diag_bias(vis, txt)
    self_fn = lambda q, k, v: A.attention_dense(q, k, v, self_bias)
    cross_fn = lambda q, k, v: A.attention_dense(q, k, v, cross_bias)

    bad = []

    def cmp(name, got, key=None):
        r = rel_err(got, ref[key or name])
        tol = STAGE_TOL.get(name, a.tol)
        ok = r == r and r < tol           # NaN(形状不符) 也算失败
        note = f"  (阈值 {tol:.0e}，见 STAGE_TOL)" if name in STAGE_TOL else ""
        print(f"  [{'OK ' if ok else 'BAD'}] {name:<16} rel={r:.3e}{note}")
        if not ok:
            bad.append(name)
        return ok

    # ── ② 逐算子 ──────────────────────────────────────────────────────────
    print("\n② 逐算子：")
    x = tokens @ params["x_embedder"].T
    cmp("x_embed", x)
    sincos = A.timestep_sincos(t, cfg.model_channels)
    cmp("sincos", sincos)
    adaln = A.dense(A.silu(A.dense(sincos, params["t_embedder_1"])),
                    params["t_embedder_2"])
    cmp("adaln_lora", adaln)
    emb = A.rms_norm(sincos, params["t_embedding_norm"], cfg.eps_rms)
    cmp("t_emb", emb)

    cos, sin = A.packed_rope_cos_sin(rows, cols, cfg.head_dim,
                                     cfg.rope_h_ratio, cfg.rope_w_ratio)
    # torch 存的是角度，这里比 cos/sin 反推的主值（角度本身可差 2π 的整数倍）
    ang = np.arctan2(np.asarray(sin[:, 0, :]), np.asarray(cos[:, 0, :]))
    exp = np.arctan2(np.sin(ref["rope"]), np.cos(ref["rope"]))
    d = float(np.abs(ang - exp).max())
    print(f"  [{'OK ' if d < a.tol else 'BAD'}] {'rope(角度主值)':<16} absdiff={d:.3e}")
    if d >= a.tol:
        bad.append("rope")

    p0 = params["blocks"][0]
    h = A.dense(A.dense(A.silu(emb), p0["adaln_self_1"]), p0["adaln_self_2"])
    sh, sc, _ = jnp.split(h + adaln, 3, axis=-1)
    take = lambda z: jnp.take(z, mod_index, axis=0)
    pre = A.layer_norm(x, cfg.eps_ln) * (1 + take(sc)) + take(sh)
    cmp("pre_selfattn", pre)
    q, k, v = A.attn_qkv(pre, None, p0["self_attn"], cfg, None, "", cos, sin)
    cmp("q", q)
    cmp("k", k)
    cmp("v", v)
    att = A.attention_dense(q, k, v, self_bias).reshape(-1, cfg.model_channels)
    cmp("selfattn_out", A.dense(att, p0["self_attn"]["output_proj"]))

    # ── ③ 逐块 ────────────────────────────────────────────────────────────
    print("\n③ 逐块（只报异常块；rel 应稳定在 1e-5 量级，不该单调放大）：")
    xb = x
    worst = (0.0, -1)
    for i in range(cfg.num_blocks):
        xb = A.block_forward(xb, params["blocks"][i], cfg, emb, adaln, take,
                             ctx, cos, sin, self_fn, cross_fn, None, i)
        r = rel_err(xb, ref[f"block{i}"])
        worst = max(worst, (r, i))
        if r >= a.tol:
            print(f"  [BAD] block{i:<10} rel={r:.3e}")
            bad.append(f"block{i}")
    print(f"  最差块: block{worst[1]} rel={worst[0]:.3e}")

    # ── ① 整模（主判据）───────────────────────────────────────────────────
    print("\n① 整模（主判据）：")
    out = A.forward_packed(params, cfg, tokens, t, ctx, rows, cols, mod_index,
                           self_fn, cross_fn, loras=None, remat=False)
    o, r = np.asarray(out, np.float64), ref["out"].astype(np.float64)
    total = rel_err(o, r)
    print(f"  torch mean={r.mean():+.6f} std={r.std():.6f}")
    print(f"  jax   mean={o.mean():+.6f} std={o.std():.6f}")
    print(f"  max_abs_diff={np.abs(o - r).max():.3e}  rel={total:.3e}")
    off = 0
    for i, n in enumerate(vis):
        print(f"    图{i} tokens[{off}:{off + n}] rel="
              f"{rel_err(o[off:off + n], r[off:off + n]):.3e}")
        off += n

    ok = total < a.tol and not bad
    print(f"\n*** {'对拍通过' if ok else '对拍失败'} ***"
          + ("" if ok else f"  失败项: {bad[:8]}"))
    # 只在**整模**失败时提醒：统计量一致但 rel 大 = 排列/通道顺序问题
    if total >= a.tol and abs(o.std() - r.std()) / max(r.std(), 1e-12) < 0.01:
        print("提示：std 几乎相同但逐元素差很大 -> 多半是排列/通道顺序错，"
              "优先查 output_tokens_to_patch_tokens 与 q/k/v 的 head reshape")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
