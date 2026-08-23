r"""K3 对拍闸门 · 第二步（jax 侧）：`jax_tpu/qwen3vl_te.py` ≡ HF Qwen3-VL 文本塔。

先跑 `dump_qwen3vl_ref.py`（torch 解释器）产出 `_ref/qwen3vl/`，再跑本脚本
（jax 解释器）。fp32 + 小构型随机权重，判据 rel ≤ 1e-5。

比什么：
  T0 加载器      真 checkpoint 的键名前缀 / 分片合并 / 构型推断
  T1 RoPE 表     cos/sin 逐元素 ≡ `Qwen3VLTextRotaryEmbedding`（含 mrope 退化）
  T2 逐层 hidden **每一层**的输入都比（不是只比最后一层）
  T3 tap 语义    官方 `hidden_states[k]` ≡ 手工"第 k 层输入"（torch 侧已断言，
                 这里再断言 jax 的 tap 取的就是同一个东西）
  T4 tap 堆叠序  [B, L, n_taps, D] 与 torch `stack(..., dim=2)` 同序
  T5 右侧 padding  加 pad 后有效位逐 bit 不变（口径 3 的 jax 侧复现）
  T6 embedding 裁剪  keep_ids 路径与全表路径逐 bit 相同
  T7 fail-fast   tap == 层数 时报错而不是静默取到 final norm
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import qwen3vl_te as TE                                             # noqa: E402

REF = Path(__file__).parent / "_ref" / "qwen3vl"
TOL = 1e-5


def rel(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    d = np.abs(a - b).max()
    s = max(np.abs(b).max(), 1e-12)
    return float(d / s)


def report(name: str, r: float, tol: float = TOL) -> bool:
    ok = r <= tol
    print(f"  [{'OK' if ok else '!!'}] {name:<34} rel={r:.3e}")
    return ok


def main() -> int:
    if not (REF / "qwen3vl_ref.npz").exists():
        print(f"缺 {REF}/qwen3vl_ref.npz —— 先用 torch 解释器跑 dump_qwen3vl_ref.py")
        return 1
    z = np.load(REF / "qwen3vl_ref.npz", allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    n_layers = int(meta["num_hidden_layers"])
    ok = True

    # ── T0 加载器 + 构型推断 ─────────────────────────────────────────────────
    taps_all = tuple(range(n_layers))                 # 0..n-1，n_run = n-1
    params, cfg = TE.load_text_tower(str(REF / "tiny_te.safetensors"),
                                     dtype=jnp.float32, select_layers=taps_all)
    want = dict(hidden=meta["hidden_size"], layers=n_layers,
                heads=meta["num_attention_heads"], kv_heads=meta["num_key_value_heads"],
                head_dim=meta["head_dim"], intermediate=meta["intermediate_size"],
                vocab=meta["vocab_size"])
    bad = {k: (getattr(cfg, k), v) for k, v in want.items() if getattr(cfg, k) != v}
    print(f"T0 加载器：cfg={cfg}")
    if bad:
        print(f"  [!!] 构型推断错: {bad}")
        ok = False
    else:
        print(f"  [OK] 构型推断 + 键名前缀 {TE.TEXT_PREFIX!r}")
    n_run = TE.n_layers_needed(taps_all)
    got = params["layers"]["wq"].shape[0]
    if got != n_run:
        print(f"  [!!] 只该加载 {n_run} 层，实际 {got}")
        ok = False
    else:
        print(f"  [OK] 只加载了 {n_run}/{n_layers} 层（max tap = 第 {n_run} 层的输入）")

    # ── T1 RoPE 表 ──────────────────────────────────────────────────────────
    ids_a, ids_b = z["ids_a"], z["ids_b"]
    cos, sin = TE.rope_tables(ids_a.size, cfg, jnp.float32)
    print("T1 RoPE 表（mrope 在纯文本下退化为标准 RoPE）：")
    ok &= report("cos", rel(cos, z["cos_a"]))
    ok &= report("sin", rel(sin, z["sin_a"]))

    # ── T2/T3/T4 逐层 hidden ────────────────────────────────────────────────
    print("T2 逐层 hidden（tap 0..%d = 每层的输入 + 最后一层输出）：" % n_run)
    for tag, ids, ref in (("a", ids_a, z["hs_a"]), ("b", ids_b, z["hs_b"])):
        got = np.asarray(TE.forward_taps(params, cfg, jnp.asarray(ids)[None], taps_all))
        # got [1, L, n_taps, D] -> [n_taps, L, D]
        got = np.transpose(got[0], (1, 0, 2))
        worst = 0.0
        for i in range(len(taps_all)):
            worst = max(worst, rel(got[i], ref[i]))
        ok &= report(f"序列 {tag}（L={ids.size}）逐层最差", worst)
        # T3：torch 侧官方 hidden_states 与手工栈的一致性（0.0 由 dump 侧保证），
        # 这里断言 jax 的 tap 命中的是官方那一份。
        ok &= report(f"序列 {tag} vs 官方 hidden_states",
                     rel(got, z[f"official_{tag}"][:len(taps_all)]))

    print("T4 tap 堆叠序（[B, L, n_taps, D]，与 torch stack(dim=2) 同序）：")
    sel = (2, min(5, n_layers - 1))
    got = np.asarray(TE.forward_taps(params, cfg, jnp.asarray(ids_a)[None], sel))
    exp = np.stack([z["hs_a"][i] for i in sel], axis=1)[None]
    ok &= report(f"select={sel} 形状 {got.shape}", rel(got, exp))

    # ── T5 右侧 padding 不改有效位 ───────────────────────────────────────────
    # 判据不是"逐 bit 0"：改 L 就改了 XLA 的分块与归约次序，fp32 下会有
    # ~1e-8 的噪声（torch 那侧恰好给出 0.0 是它的分块碰巧没变，不是更强的保证）。
    # 真的漏了 pad 位的话误差是 O(1) 量级，与这里差着 7~8 个数量级。
    print("T5 右侧 padding（口径 3：causal 下 pad 位不可见）：")
    pad = np.zeros(ids_a.size + 19, np.int32)
    pad[:ids_a.size] = ids_a
    got = np.asarray(TE.forward_taps(params, cfg, jnp.asarray(pad)[None], taps_all))
    base = np.asarray(TE.forward_taps(params, cfg, jnp.asarray(ids_a)[None], taps_all))
    ok &= report("有效位 vs 无 padding", rel(got[:, :ids_a.size], base), 1e-6)
    print(f"       torch 侧同一探针：中段 padding {float(z['probe_mid']):.3e}（≠0，"
          f"故不可用）/ 右侧 padding {float(z['probe_right']):.3e}")

    # ── T6 embedding 裁剪 ───────────────────────────────────────────────────
    # 与全表路径**同一条 encode_ids**（同 quantum、同形状）对比，把 padding 噪声
    # 从这一项里剔掉 —— 裁剪只换了 gather 的行号，要求逐 bit 相同。
    print("T6 embedding 裁剪（keep_ids）：")
    keep = np.unique(np.concatenate([ids_a, ids_b]))
    p2, _ = TE.load_text_tower(str(REF / "tiny_te.safetensors"), dtype=jnp.float32,
                               select_layers=taps_all, keep_ids=keep)
    kw = dict(prefix_len=0, select_layers=taps_all, quantum=8)
    got = np.asarray(TE.encode_ids(p2, cfg, [ids_a], **kw)[0])
    exp = np.asarray(TE.encode_ids(params, cfg, [ids_a], **kw)[0])
    d = float(np.abs(got - exp).max())
    saved = 1 - keep.size / cfg.vocab
    print(f"  [{'OK' if d == 0 else '!!'}] max|Δ| = {d:.3e}（要求逐 bit 0）；"
          f"词表 {cfg.vocab} -> {keep.size}（省 {saved:.1%}）")
    ok &= (d == 0.0)
    # 顺带闸 encode_ids 的切片：它去掉 prefix、按有效长度截断，结果应与
    # forward_taps 的对应切片一致（这里 prefix_len=0，等于整条）。
    ok &= report("encode_ids ≡ forward_taps 切片", rel(exp, base[0][:ids_a.size]), 1e-6)

    # ── T7 fail-fast ────────────────────────────────────────────────────────
    print("T7 fail-fast（tap == 层数 的语义是 final norm，不是层输入）：")
    try:
        TE.load_text_tower(str(REF / "tiny_te.safetensors"), dtype=jnp.float32,
                           select_layers=(2, n_layers))
        print("  [!!] 没报错")
        ok = False
    except ValueError as e:
        print(f"  [OK] {str(e).splitlines()[0][:70]}")

    print("\n" + ("K3 通过" if ok else "K3 未通过"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
