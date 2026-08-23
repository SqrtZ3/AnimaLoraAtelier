"""闸门 ⑤：JAX AdamW ≡ `torch.optim.AdamW`（逐步逐元素）。

优化器写错是最难归因的一类：loss 照降、参数照动，只是动力学悄悄变成了别的算法。
常见的三个静默错都在这里被判：
  * `eps` 加在 sqrt 之前（变成另一个算法）；
  * weight decay 加进梯度（那是 Adam+L2，不是 AdamW）；
  * 漏 bias-correction（前几百步的有效 lr 完全不同）。

两步跑：
    <torch-python> check_optim_parity.py --dump    # 产出 _ref/optim_ref.npz
    <jax-python>   check_optim_parity.py

判据：跑 20 步，比**每一步**的参数。阈值 1e-6（两边都是 fp32）。
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REF = Path(__file__).parent / "_ref" / "optim_ref.npz"
STEPS = 20
LR, B1, B2, EPS, WD, CLIP = 1e-4, 0.9, 0.999, 1e-8, 0.01, 1.0
SHAPES = {"blocks.0.mlp.layer1": ((2048, 8), (8, 8192)),
          "blocks.1.self_attn.q_proj": ((2048, 8), (8, 2048))}


def _fake(seed):
    """确定性地造出参数与每步梯度，两侧共用。"""
    rng = np.random.RandomState(seed)
    p = {n: {"a": rng.randn(*sa).astype(np.float32) * 0.02,
             "b": np.zeros(sb, np.float32)} for n, (sa, sb) in SHAPES.items()}
    g = [{n: {"a": rng.randn(*sa).astype(np.float32),
              "b": rng.randn(*sb).astype(np.float32)}
          for n, (sa, sb) in SHAPES.items()} for _ in range(STEPS)]
    return p, g


def dump():
    import torch
    p0, gs = _fake(0)
    params, order = [], []
    for n in SHAPES:
        for s in ("a", "b"):
            params.append(torch.tensor(p0[n][s], requires_grad=True))
            order.append((n, s))
    opt = torch.optim.AdamW(params, lr=LR, betas=(B1, B2), eps=EPS, weight_decay=WD)
    traj = []
    for g in gs:
        for t, (n, s) in zip(params, order):
            t.grad = torch.tensor(g[n][s])
        torch.nn.utils.clip_grad_norm_(params, CLIP)
        opt.step()
        traj.append(np.concatenate([t.detach().numpy().ravel() for t in params]))
    REF.parent.mkdir(parents=True, exist_ok=True)
    np.savez(REF, traj=np.stack(traj))
    print(f"已写 {REF}（{STEPS} 步 x {traj[0].size} 参数）")
    return 0


def check():
    import jax.numpy as jnp
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import optim as O

    if not REF.exists():
        print(f"找不到 {REF} —— 先在 torch 环境跑 `--dump`")
        return 2
    ref = np.load(REF)["traj"]
    p0, gs = _fake(0)
    cfg = O.AdamWConfig(lr=LR, b1=B1, b2=B2, eps=EPS, weight_decay=WD,
                        max_grad_norm=CLIP, warmup_steps=0)
    st = O.init_state({n: {s: jnp.asarray(v) for s, v in d.items()}
                       for n, d in p0.items()})
    worst, worst_i = 0.0, -1
    for i, g in enumerate(gs):
        st, _, _ = O.update(st, {n: {s: jnp.asarray(v) for s, v in d.items()}
                                 for n, d in g.items()}, cfg)
        flat = np.concatenate([np.asarray(st["master"][n][s]).ravel()
                               for n in SHAPES for s in ("a", "b")])
        d = float(np.abs(flat - ref[i]).max())
        if d > worst:
            worst, worst_i = d, i
    ok = worst < 1e-6
    print(f"  [{'OK ' if ok else 'BAD'}] 20 步逐元素最差偏差 max_abs={worst:.3e} "
          f"(在第 {worst_i} 步，阈值 1e-06)")
    print(f"\n*** {'通过' if ok else '失败'} ***")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="store_true")
    sys.exit(dump() if ap.parse_args().dump else check())
