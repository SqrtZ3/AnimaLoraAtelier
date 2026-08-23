r"""K3-B 闸门：**真权重**下 `qwen3vl_te` 重算的文本条件 ≡ 本地已缓存的 `.textfeat.npz`。

K3（`check_qwen3vl_parity.py`）用小构型随机权重判**结构**；本闸门用 8.3GB 真权重
判**口径**：键名映射、tap 取的层号、prefix 切片位置、bf16 存储——任何一处错都
会在这里显形，而且不需要再跑一次 torch（参考量就是 `cache_text_features.py`
在 CUDA bf16 下已经落好的那 96 份缓存）。

## 判据怎么定

参考量是 **torch CUDA bf16** 算的，本闸门跑的是 **JAX CPU**，两者不可能逐 bit 相同
（bf16 尾数 8 位，35 层残差流累下来必然有噪声）。所以不设绝对阈值，而是同时量三个数：

    A = JAX(bf16)  vs 缓存(torch bf16)      移植误差 + 两侧 bf16 噪声
    B = JAX(fp32 计算, bf16 权重) vs 缓存    ≈ torch 那侧 bf16 本身的噪声
    C = JAX(bf16)  vs JAX(fp32 计算)        ≈ 本侧 bf16 本身的噪声

**A 与 B、C 同量级** ⇒ 残差全部来自 bf16、移植没有系统性偏差；
A 显著大于 B/C（比如大一个数量级）⇒ 有真错，别上机。

用法（jax 解释器；需要真 TE 权重与本地缓存目录）：
    <py> check_qwen3vl_real.py --ids <caption_ids.npz> \
        --cache-dir <已缓存的 textfeat 目录> \
        --te <Qwen3-VL-4B-Instruct 目录> [-n 2]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import qwen3vl_te as TE                                             # noqa: E402


def bf16_to_f32(a: np.ndarray) -> np.ndarray:
    """npz 里的 bf16 是 uint16 位模式（numpy 没有原生 bfloat16）。"""
    if a.dtype == np.uint16:
        return (a.astype(np.uint32) << 16).view(np.float32)
    return a.astype(np.float32)


def stats(got: np.ndarray, ref: np.ndarray) -> dict:
    g, r = got.astype(np.float64), ref.astype(np.float64)
    d = g - r
    denom = max(np.abs(r).max(), 1e-30)
    cos = float((g * r).sum() / max(np.linalg.norm(g) * np.linalg.norm(r), 1e-30))
    return {"max": float(np.abs(d).max()), "rel": float(np.abs(d).max() / denom),
            "fro": float(np.linalg.norm(d) / max(np.linalg.norm(r), 1e-30)),
            "cos": cos}


def line(tag: str, s: dict) -> str:
    return (f"  {tag:<28} max={s['max']:.3e}  rel={s['rel']:.3e}  "
            f"‖Δ‖/‖ref‖={s['fro']:.3e}  cos={s['cos']:.6f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help="dump_caption_ids.py 的产物")
    ap.add_argument("--cache-dir", required=True, help="本地已有 .textfeat.npz 的目录")
    ap.add_argument("--te", required=True, help="Qwen3-VL-4B-Instruct 目录/单文件")
    ap.add_argument("-n", type=int, default=2, help="抽几条比（真权重前向在 CPU 上不便宜）")
    ap.add_argument("--quantum", type=int, default=64)
    ap.add_argument("--skip-fp32", action="store_true",
                    help="只跑 A（省一趟前向；此时无法把移植误差与 bf16 噪声分开）")
    a = ap.parse_args()

    z = np.load(a.ids, allow_pickle=True)
    stems, ids, lens = list(z["stems"]), z["ids"], z["lens"]
    cache = Path(a.cache_dir)

    picked = []
    for i, st in enumerate(stems):
        p = cache / f"{st}.textfeat.npz"
        if p.exists():
            picked.append((i, st, p))
        if len(picked) >= a.n:
            break
    if not picked:
        raise SystemExit(f"{cache} 下没有与 ids 对应的 .textfeat.npz")

    seqs = [ids[i, : lens[i]] for i, _, _ in picked]
    keep = np.unique(np.concatenate(seqs))
    print(f"抽样 {len(picked)} 条：{[s for _, s, _ in picked]}")
    print(f"  序列长度 {[int(s.size) for s in seqs]}（含 {int(z['prefix_len'])} 前缀）"
          f"；用到 {keep.size} 个不同 token")

    t0 = time.time()
    params, cfg = TE.load_text_tower(a.te, dtype=jnp.bfloat16, keep_ids=keep)
    n_run = TE.n_layers_needed(TE.KREA2_SELECT_LAYERS)
    n_par = sum(int(np.prod(x.shape)) for x in jax.tree_util.tree_leaves(
        {k: v for k, v in params.items() if k != "id_map"}))
    print(f"  权重加载 {time.time() - t0:.0f}s；cfg={cfg}")
    print(f"  实载 {n_run}/{cfg.layers} 层 + 裁剪后 embedding = {n_par / 1e9:.3f}B 参数"
          f"（全量文本塔约 4.02B，省 {1 - n_par / 4.02e9:.0%}）")

    t0 = time.time()
    got_bf16 = TE.encode_ids(params, cfg, seqs, prefix_len=int(z["prefix_len"]),
                             quantum=a.quantum)
    print(f"  bf16 前向 {time.time() - t0:.0f}s")
    got_f32 = None
    if not a.skip_fp32:
        t0 = time.time()
        got_f32 = TE.encode_ids(params, cfg, seqs, prefix_len=int(z["prefix_len"]),
                                quantum=a.quantum, compute_dtype=jnp.float32)
        print(f"  fp32 计算前向 {time.time() - t0:.0f}s")

    ok = True
    for j, (i, st, p) in enumerate(picked):
        ref = bf16_to_f32(np.load(p, allow_pickle=True)["txt"])
        # got_bf16[j] 是 ml_dtypes.bfloat16 数组（jax 的 bf16 落到 numpy 的样子）
        g16 = np.asarray(got_bf16[j]).astype(np.float32)
        print(f"\n{st}  参考 {ref.shape} / 重算 {g16.shape}")
        if g16.shape != ref.shape:
            print(f"  [!!] 形状不一致 —— 切片口径或 tap 数错了")
            ok = False
            continue
        sA = stats(g16, ref)
        print(line("A JAX(bf16) vs 缓存", sA))
        if got_f32 is not None:
            g32 = np.asarray(got_f32[j], dtype=np.float32)
            sB, sC = stats(g32, ref), stats(g16, g32)
            print(line("B JAX(fp32计算) vs 缓存", sB))
            print(line("C JAX(bf16) vs JAX(fp32)", sC))
            # 判据：A 不该比"两侧 bf16 噪声之和"大出一个数量级。
            floor = sB["fro"] + sC["fro"]
            bad = sA["fro"] > 10 * max(floor, 1e-9)
            print(f"  -> A/(B+C) = {sA['fro'] / max(floor, 1e-30):.2f}"
                  f"（>10 判为有系统性偏差）{'  [!!]' if bad else '  [OK]'}")
            ok &= not bad
        # 逐层看：某一层单独崩说明 tap 层号或权重映射错，整体一致的噪声才是 bf16。
        per = [stats(g16[:, k], ref[:, k])["fro"] for k in range(ref.shape[1])]
        print("  逐层 ‖Δ‖/‖ref‖: " + " ".join(f"{v:.1e}" for v in per))
        if max(per) > 5 * (sum(per) / len(per)):
            print("  [!!] 某一层明显偏离其余层 —— 疑似 tap 层号/权重映射错")
            ok = False

    print("\n" + ("K3-B 通过" if ok else "K3-B 未通过"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
