r"""闸门⑫下半：JAX 的 LoKr / DoRA / adamw_snr ≟ PyTorch 侧实现。

先用 torch 解释器跑 `dump_adapter_ref.py --dump`，再用 jax 解释器跑本脚本。

判据全部是 fp32 下的**相对误差 < 1e-5**：两边都是 fp32 稠密线代，除了归约顺序
没有别的差异来源。超了就是算法口径不同，不是精度问题。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax                                                    # noqa: E402
import jax.numpy as jnp                                       # noqa: E402
import numpy as np                                            # noqa: E402

import adapters as AD                                         # noqa: E402
import optim as O                                             # noqa: E402

REF = Path(__file__).parent / "_ref" / "adapter_ref.npz"
TOL = 1e-5
OK = FAIL = 0


def check(name, got, want, tol=TOL):
    global OK, FAIL
    g, w = np.asarray(got, np.float64), np.asarray(want, np.float64)
    rel = np.max(np.abs(g - w)) / max(np.max(np.abs(w)), 1e-12)
    good = rel < tol and g.shape == w.shape
    globals()["OK" if good else "FAIL"] = globals()["OK" if good else "FAIL"] + 1
    print(f"  {'[OK ]' if good else '[FAIL]'} {name:<32} rel={rel:.3e} "
          f"{'' if g.shape == w.shape else f'形状 {g.shape} vs {w.shape}'}")


def main() -> int:
    if not REF.exists():
        print(f"缺 {REF} —— 先用 torch 解释器跑 dump_adapter_ref.py")
        return 2
    z = np.load(REF)
    x = jnp.asarray(z["x"])
    cfg32 = AD.AdapterConfig(kind="lokr", factor=4, rank=8, alpha=8.0)

    print("① LoKr kron-bypass（无 DoRA）")
    p = {"w1": jnp.asarray(z["w1"]), "w2a": jnp.asarray(z["w2a"]),
         "w2b": jnp.asarray(z["w2b"])}
    c = {"scale": jnp.asarray(z["scaling"])}
    delta, _ = AD._lokr_delta(x, p, c, None, cfg32)
    check("ΔW 旁路输出", delta, z["y_lokr"])
    # 与"物化 kron 再矩乘"对拍：证明 bypass 的拆分方向没反（反了形状照样对）
    kron = np.kron(z["w1"], z["w2a"] @ z["w2b"]) * float(z["scaling"])
    check("≡ 物化 kron 矩乘", delta, np.asarray(z["x"]) @ kron.T)

    print("\n② LoKr + DoRA（输出域幅度归一）")
    w = jnp.asarray(z["base_w"])
    pd = dict(p, dora=jnp.asarray(z["dora_scale"]))
    low = p["w2b"]
    check("‖W+ΔW‖ 逐行范数",
          AD._merged_row_norm(w, pd, c, low, cfg32), z["merged_norm"])
    check("DoRA 前向", AD.apply(x, w, cfg32, pd, c, None), z["y_dora"])

    print("\n③ 标准 LoRA + DoRA")
    cfgl = AD.AdapterConfig(kind="lora", rank=8, alpha=8.0)
    pl = {"a": jnp.asarray(z["lora_down"].T), "b": jnp.asarray(z["lora_up"].T),
          "dora": jnp.asarray(z["dora2_scale"])}
    cl = {"scale": jnp.asarray(z["scaling2"])}
    check("DoRA 前向", AD.apply(x, jnp.asarray(z["base2_w"]), cfgl, pl, cl, None),
          z["y_lora_dora"])

    print("\n④ adamw_snr（SNR 锐化 / cautious 掩码）")
    for tag, kw in (("plain", dict(snr_power=1.0, cautious=False)),
                    ("snr15", dict(snr_power=1.5, cautious=False)),
                    ("caut", dict(snr_power=1.0, cautious=True)),
                    ("both", dict(snr_power=1.5, cautious=True))):
        ocfg = O.AdamWConfig(lr=2e-3, b1=0.98, b2=0.999, eps=1e-8,
                             weight_decay=1e-5, max_grad_norm=0.0, **kw)
        st = O.init_state({"m": {"a": jnp.asarray(z[f"opt_{tag}_p0"])}})
        for g in z[f"opt_{tag}_g"]:
            st, _, _ = O.update(st, {"m": {"a": jnp.asarray(g)}}, ocfg, jnp.float32)
        check(f"6 步后参数 [{tag}]", st["master"]["m"]["a"], z[f"opt_{tag}_p"])

    print("\n⑤ 逐块 rank 掩码：被掩掉的通道对输出零贡献")
    # 掩掉一半 rank 之后，输出必须等于"直接用一半 rank 的因子"算出来的
    r = int(z["eff_rank"])
    m = np.zeros(r, np.float32)
    m[: r // 2] = 1.0
    cm = dict(c, rmask=jnp.asarray(m))
    got, _ = AD._lokr_delta(x, p, cm, None, cfg32)
    p_half = {"w1": p["w1"], "w2a": p["w2a"][:, : r // 2],
              "w2b": p["w2b"][: r // 2]}
    want, _ = AD._lokr_delta(x, p_half, c, None, cfg32)
    check("掩一半 ≡ 只留一半", got, want, tol=1e-6)

    print(f"\n{'*** 通过 ***' if FAIL == 0 else f'*** {FAIL} 项失败 ***'}"
          f"  ({OK} OK / {FAIL} FAIL)")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
