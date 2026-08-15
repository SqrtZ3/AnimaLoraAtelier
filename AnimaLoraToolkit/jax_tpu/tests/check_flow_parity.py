"""闸门 ④：Flow Matching 目标的 JAX 侧 ≡ PyTorch 侧（`trainer/objective.py`）。

t 采样、噪声/target 构造、loss 权重这三样写错都**不报错**，只会训出一个学不到
东西（或学错东西）的模型。尤其是 `target = noise - latent`（速度场）写成
`latent - noise` 或写成噪声预测——两者维度全对、loss 也会下降。

两步跑（torch 与 jax 装不进同一个解释器）：

    <torch-python> check_flow_parity.py --dump     # 产出 _ref/flow_ref.npz
    <jax-python>   check_flow_parity.py            # 比对，退出码 0 = 通过

判据分两类：
  * **确定性变换**（shift / loss 权重 / noisy+target）逐元素比，阈值 1e-6；
  * **随机采样**只能比分布：两边各采 200k 个，比 9 个分位点，阈值 5e-3
    （两边 RNG 不同，逐样本对不上是正常的；分位点对不上才是公式错）。
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REF = Path(__file__).parent / "_ref" / "flow_ref.npz"
QS = np.array([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])

# (t_mode, flow_shift, schedule_shift) —— 覆盖画风训练实际在用的几种
T_CASES = [("logit_normal", 3.0, 1.0), ("logit_normal", 1.0, 2.0),
           ("logit_normal_low", 3.0, 1.0), ("uniform", 1.0, 1.0)]
W_CASES = ["none", "detail_inv_t", "cosmap", "min_snr", "logit_normal"]
N_SAMPLE = 200_000


def dump():
    """在 torch 环境跑：调用仓库自己的 objective.py，产出参考量。"""
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # AnimaLoraToolkit/
    from trainer import objective as O

    torch.manual_seed(0)
    t_grid = np.linspace(1e-3, 1 - 1e-3, 257).astype(np.float32)
    tg = torch.from_numpy(t_grid)
    out = {"t_grid": t_grid}

    for scheme in W_CASES:
        out[f"w_{scheme}"] = O.compute_loss_weight(
            tg, scheme=scheme, min_snr_gamma=5.0,
            detail_inv_t_min=1.0, detail_inv_t_max=5.0).numpy()

    for mode, shift, sched in T_CASES:
        t = O.sample_t(N_SAMPLE, torch.device("cpu"), mode=mode, shift=shift)
        t = O.apply_timestep_schedule_shift(t, sched)
        out[f"q_{mode}_{shift}_{sched}"] = np.quantile(t.numpy(), QS).astype(np.float32)

    # noisy / target：随机 latent+noise，逐元素比
    lat = torch.randn(4, 64, 16)
    noi = torch.randn(4, 64, 16)
    tt = torch.tensor([0.1, 0.3, 0.7, 0.95]).view(4, 1, 1)
    out["lat"], out["noi"], out["tt"] = lat.numpy(), noi.numpy(), tt.numpy()
    out["noisy"] = ((1 - tt) * lat + tt * noi).numpy()
    out["target"] = (noi - lat).numpy()

    REF.parent.mkdir(parents=True, exist_ok=True)
    np.savez(REF, **out)
    print(f"已写 {REF}（{len(out)} 项）")
    return 0


def check():
    import jax
    import jax.numpy as jnp
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import flow as F

    if not REF.exists():
        print(f"找不到 {REF} —— 先在 torch 环境跑 `--dump`")
        return 2
    ref = np.load(REF)
    bad = []

    def cmp(name, got, exp, tol):
        r = float(np.abs(np.asarray(got, np.float64) - np.asarray(exp, np.float64)).max())
        ok = r < tol
        print(f"  [{'OK ' if ok else 'BAD'}] {name:<34} max_abs={r:.3e} (阈值 {tol:.0e})")
        if not ok:
            bad.append(name)

    print("① loss 权重（确定性）：")
    tg = jnp.asarray(ref["t_grid"])
    for scheme in W_CASES:
        cfg = F.FlowConfig(weighting=scheme, min_snr_gamma=5.0)
        cmp(f"w_{scheme}", F.loss_weight(tg, cfg), ref[f"w_{scheme}"], 1e-6)

    print("\n② noisy / target（确定性；target 是**速度场** noise-latent）：")
    noisy, target = F.make_noisy_and_target(
        jnp.asarray(ref["lat"]), jnp.asarray(ref["noi"]), jnp.asarray(ref["tt"]))
    cmp("noisy", noisy, ref["noisy"], 1e-6)
    cmp("target", target, ref["target"], 1e-6)

    print("\n③ t 采样分布（比分位点，两边 RNG 不同）：")
    for mode, shift, sched in T_CASES:
        cfg = F.FlowConfig(t_mode=mode, flow_shift=shift, schedule_shift=sched)
        t = F.sample_t(jax.random.PRNGKey(0), N_SAMPLE, cfg)
        cmp(f"分位 {mode} s={shift} ss={sched}",
            np.quantile(np.asarray(t), QS), ref[f"q_{mode}_{shift}_{sched}"], 5e-3)

    print(f"\n*** {'通过' if not bad else '失败'} ***" + ("" if not bad else f"  失败项: {bad}"))
    return 0 if not bad else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="store_true", help="在 torch 环境跑，产出参考量")
    sys.exit(dump() if ap.parse_args().dump else check())
