"""GAF ghost-backend fidelity probe (CPU, confound-free).

Motivation
----------
On the real your-dataset-1 run, two 480-step trainings — one with the autograd
backend, one with the ghost (TRAK random-projection) backend — produced
per-image trust rankings that were *uncorrelated* (Spearman ~0.02, bottom-10
overlap 1/10). But those two runs had different seeds/trajectories, so the CSVs
alone cannot separate two hypotheses:

  (a) the ghost backend is an unfaithful estimator (projection noise destroys the
      per-sample cosine *ranking*), vs
  (b) the GAF trust signal is intrinsically trajectory-unstable run-to-run,
      regardless of backend.

This probe removes the confound: one model, one batch, one seed. We run BOTH
backends on the *same* forward/backward and compare their per-sample LOO cosines
directly. For a plain Linear (bias-free) the autograd per-sample gradient w.r.t.
the weight is exactly G_i = sum_t g_{i,t} (x) a_{i,t}, and the ghost sketch is
exactly Pg^T G_i Pa — so this is a faithful, exact test of whether the Kronecker
projection preserves the per-sample direction *ranking* that GAF acts on.

We sweep the regime from "large separation" (the original synthetic self-check
that reported corr~0.996) down to "subtle marginal" (matches a *not-clean-enough*
set: small true-cosine gaps between samples), plus proj_dim and n_layers.

Run:  <python> tests/test_gaf_ghost_fidelity.py
"""

import pathlib
import sys

import torch
import torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trainer.gaf import (  # noqa: E402
    GafGhostHooks,
    extract_per_sample_grads,
    leave_one_out_agreement,
)


# --------------------------------------------------------------------------- #
# rank-correlation helpers (no scipy — cloud venv lacks it)
# --------------------------------------------------------------------------- #
def _rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0] * len(v)
    for pos, i in enumerate(order):
        r[i] = pos
    return r


def _pearson(x, y):
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    sx = sum((xi - mx) ** 2 for xi in x) ** 0.5
    sy = sum((yi - my) ** 2 for yi in y) ** 0.5
    if sx == 0 or sy == 0:
        return float("nan")
    return cov / (sx * sy)


def _spearman(x, y):
    return _pearson(_rank(x), _rank(y))


def _bottomk_overlap(a, b, k):
    """overlap of the k lowest-cosine (most-marginal) samples between two rankings."""
    la = set(sorted(range(len(a)), key=lambda i: a[i])[:k])
    lb = set(sorted(range(len(b)), key=lambda i: b[i])[:k])
    return len(la & lb)


# --------------------------------------------------------------------------- #
# tiny faithful testbed
# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    """Bias-free Linear stack so each weight's per-sample grad == sum_t g⊗a (the
    exact same G the ghost hook sketches). tanh between layers for realistic a/g."""

    def __init__(self, dim, n_layers):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in range(n_layers))

    def forward(self, x):
        h = x
        for i, lin in enumerate(self.layers):
            h = lin(h)
            if i < len(self.layers) - 1:
                h = torch.tanh(h)
        return h


def _build_batch(B, T, dim, n_layers, spread, n_outliers, outlier_boost, seed,
                 off_dirs=None, off_mag=None):
    """Build model + inputs + per-sample top-layer output-grad coefficient c.

    c_i = consensus + spread*boost_i*noise_i  (+ optional persistent off_mag_i*off_dir_i).
    loss_i = (y_i*c_i).sum() => dL/dy_i = c_i exactly, giving a controlled
    consensus+outlier direction structure. `off_dirs`/`off_mag` (fixed across seeds)
    inject a *persistent* per-sample identity for the cross-seed stability probe.
    """
    g_cpu = torch.Generator().manual_seed(seed)
    model = MLP(dim, n_layers).double()
    for p in model.parameters():
        p.data = torch.randn(p.shape, generator=g_cpu, dtype=torch.float64) / (dim ** 0.5)
    params = [lin.weight for lin in model.layers]

    base_x = torch.randn(1, T, dim, generator=g_cpu, dtype=torch.float64)
    x = (base_x + 0.3 * torch.randn(B, T, dim, generator=g_cpu, dtype=torch.float64))
    x.requires_grad_(True)              # realistic: activations carry grad (silences hook warning)

    consensus = torch.randn(1, T, dim, generator=g_cpu, dtype=torch.float64)
    noise = torch.randn(B, T, dim, generator=g_cpu, dtype=torch.float64)
    boost = torch.ones(B, 1, 1, dtype=torch.float64)
    boost[:n_outliers] = outlier_boost
    c = consensus + spread * boost * noise
    if off_dirs is not None:            # persistent per-sample identity (stability probe)
        c = c + (off_mag.view(B, 1, 1) * off_dirs.view(B, 1, dim))
    return model, params, x, c


def _both_cosines(model, params, x, c, B, proj_dim, seed, trust_w=None, rescale=False):
    """Run BOTH backends on ONE shared forward/backward; return (cos_exact, cos_ghost).

    exact  : per-sample LOO cosine from the CLEAN per_sample (== real autograd backend).
    ghost  : per-sample LOO cosine from the sketch captured during the (optionally
             trust-weighted) summed backward (== real ghost backend). `rescale` divides
             each sketch row by trust_w_i before LOO (the proposed de-contamination fix).
    """
    hooks = GafGhostHooks(batch_size=B, proj_dim=proj_dim, seed=seed)
    hooks.register(list(model.layers))

    hooks.begin()                       # active=True -> _fwd stashes projected inputs
    y = model(x)
    per_sample = (y * c).sum(dim=(1, 2))    # [B]

    hooks.active = False                # exact extraction must NOT consume the stash
    exact = extract_per_sample_grads(per_sample, params)

    hooks.active = True
    weighted = per_sample if trust_w is None else (per_sample * trust_w)
    weighted.sum().backward()
    ghost = hooks.collect()
    hooks.remove()

    if rescale and trust_w is not None:
        ghost = ghost / trust_w.view(-1, 1).clamp_min(1e-6)

    return leave_one_out_agreement(exact).tolist(), leave_one_out_agreement(ghost).tolist()


def run_case(*, B=12, T=64, dim=512, n_layers=4, proj_dim=16, spread=1.0,
             n_outliers=3, outlier_boost=4.0, seed=0, trust_w=None, rescale=False):
    """Confound-free ghost-vs-exact comparison (same model/batch/seed)."""
    model, params, x, c = _build_batch(B, T, dim, n_layers, spread,
                                        n_outliers, outlier_boost, seed)
    cos_exact, cos_ghost = _both_cosines(model, params, x, c, B, proj_dim, seed,
                                         trust_w=trust_w, rescale=rescale)
    return {
        "spearman": _spearman(cos_ghost, cos_exact),
        "pearson": _pearson(cos_ghost, cos_exact),
        "bottom3_overlap": _bottomk_overlap(cos_ghost, cos_exact, n_outliers),
        "true_cos_std": torch.tensor(cos_exact).std().item(),
        "n_outliers": n_outliers,
        "cos_exact": cos_exact,
        "cos_ghost": cos_ghost,
    }


def _avg_pairwise_spearman(list_of_cos):
    vals = []
    n = len(list_of_cos)
    for i in range(n):
        for j in range(i + 1, n):
            vals.append(_spearman(list_of_cos[i], list_of_cos[j]))
    return sum(vals) / len(vals) if vals else float("nan")


def _avg_cross_spearman(a_list, b_list):
    """mean Spearman(a[i], b[j]) over i != j — the synthetic analog of comparing a
    ghost-backend run against an autograd-backend run on different trajectories."""
    vals = []
    for i in range(len(a_list)):
        for j in range(len(b_list)):
            if i != j:
                vals.append(_spearman(a_list[i], b_list[j]))
    return sum(vals) / len(vals) if vals else float("nan")


def run_stability(*, B=16, dim=512, n_layers=8, proj_dim=64, n_seeds=6,
                  scenario="structured", n_marg=4):
    """Cross-trajectory stability of the GAF trust signal itself.

    A *persistent* per-sample off-consensus identity (off_dirs/off_mag) is fixed
    across seeds; only nuisance (model init, input jitter, noise) varies per seed.
    - scenario='structured': a few samples are strongly off (a real dirty subset).
    - scenario='uniform'   : every sample equally mildly off ("不够干净", no subset).
    If even the EXACT backend's ranking fails to persist across seeds in 'uniform',
    then the real 0.02 is mostly signal instability, not a ghost bug.
    """
    gfix = torch.Generator().manual_seed(99)
    off_dirs = torch.randn(B, dim, generator=gfix, dtype=torch.float64)
    off_dirs = off_dirs / off_dirs.norm(dim=1, keepdim=True)
    if scenario == "structured":
        off_mag = torch.zeros(B, dtype=torch.float64)
        off_mag[:n_marg] = 3.0
    else:  # uniform: all samples equally, mildly off
        off_mag = torch.full((B,), 0.8, dtype=torch.float64)

    exacts, ghosts = [], []
    for s in range(n_seeds):
        model, params, x, c = _build_batch(B, T=64, dim=dim, n_layers=n_layers,
                                            spread=0.4, n_outliers=0, outlier_boost=1.0,
                                            seed=1000 + s, off_dirs=off_dirs, off_mag=off_mag)
        ce, cg = _both_cosines(model, params, x, c, B, proj_dim, seed=1000 + s)
        exacts.append(ce)
        ghosts.append(cg)

    return {
        "exact_cross_seed": _avg_pairwise_spearman(exacts),       # is the truth stable?
        "ghost_cross_seed": _avg_pairwise_spearman(ghosts),       # is ghost stable?
        "ghostA_vs_exactB": _avg_cross_spearman(ghosts, exacts),  # ~ the real 0.02 measurement
        "ghost_vs_exact_sameseed": sum(_spearman(g, e) for g, e in zip(ghosts, exacts)) / n_seeds,
    }


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def main():
    torch.manual_seed(0)
    print("GAF ghost-backend fidelity probe (one model / one batch / one seed)\n"
          "Metric = agreement between ghost per-sample LOO-cosine and the EXACT\n"
          "per-sample-gradient LOO-cosine. Spearman 1.0 = identical ranking.\n")

    print("== 1. regime sweep: separation 'spread' x proj_dim  (dim=512, 4 layers, B=12) ==")
    print(f"{'spread':>7} {'true_cosσ':>9} | "
          + " ".join(f"k={k:<4d}" for k in (16, 32, 64, 128)))
    for spread in (3.0, 1.0, 0.4, 0.15):
        cells = []
        true_sigma = None
        for k in (16, 32, 64, 128):
            r = run_case(spread=spread, proj_dim=k, seed=0)
            cells.append(f"{r['spearman']:+.2f}")
            true_sigma = r["true_cos_std"]
        print(f"{spread:>7.2f} {true_sigma:>9.3f} | " + " ".join(f"{c:>6}" for c in cells))

    print("\n  (read: each cell = Spearman(ghost,exact). 'true_cosσ' is how separated\n"
          "   the samples actually are — a *not-clean-enough* set lives at small σ.)\n")

    print("== 2. does concatenating MORE layers rescue the ranking? (subtle spread=0.4, k=16) ==")
    print(f"{'n_layers':>9} {'spearman':>9} {'bottom-overlap':>15}")
    for nl in (1, 4, 16, 64):
        r = run_case(spread=0.4, n_layers=nl, proj_dim=16, seed=1)
        print(f"{nl:>9} {r['spearman']:>+9.2f} {r['bottom3_overlap']:>11}/{r['n_outliers']}")

    print("\n== 3. stability across seeds (subtle spread=0.4, k=16, dim=512, 4 layers) ==")
    sps = [run_case(spread=0.4, proj_dim=16, seed=s)["spearman"] for s in range(6)]
    print("  spearman per seed:", " ".join(f"{v:+.2f}" for v in sps),
          f"  mean={sum(sps)/len(sps):+.2f}")

    print("\n== 4. anchor: large separation should reproduce the old ~0.99 self-check ==")
    r = run_case(spread=3.0, proj_dim=16, n_layers=4, seed=0)
    print(f"  spread=3.0, k=16  ->  spearman={r['spearman']:+.3f}  "
          f"pearson={r['pearson']:+.3f}  (sanity anchor)")

    print("\n== 5. contamination: ghost captures the TRUST-WEIGHTED backward (real loop) ==")
    print("   trust_w_i = floor + (1-floor)*sigmoid(cos_i/temp) from the clean cos")
    print("   (steady-state feedback). Compare clean vs contaminated vs 1/w rescale fix.\n")
    print(f"{'n_layers':>9} {'clean':>8} {'+trust(contam)':>15} {'+trust +rescale':>16}")
    for nl in (8, 64):
        base = run_case(spread=0.4, n_layers=nl, proj_dim=64, seed=2)
        cos_e = torch.tensor(base["cos_exact"])
        tw = (0.3 + 0.7 * torch.sigmoid(cos_e / 0.15)).double()
        contam = run_case(spread=0.4, n_layers=nl, proj_dim=64, seed=2, trust_w=tw)
        fixed = run_case(spread=0.4, n_layers=nl, proj_dim=64, seed=2, trust_w=tw, rescale=True)
        print(f"{nl:>9} {base['spearman']:>+8.2f} {contam['spearman']:>+15.2f} "
              f"{fixed['spearman']:>+16.2f}")

    print("\n== 6. is there a STABLE trust signal at all? (cross-trajectory, real model has ~280 layers) ==")
    print("   persistent per-sample off-consensus identity; only nuisance varies per seed.")
    print(f"{'scenario':>12} {'exact_xseed':>12} {'ghost_xseed':>12} "
          f"{'ghostA_vs_exactB':>17} {'ghost~exact(same)':>18}")
    for scen in ("structured", "uniform"):
        st = run_stability(scenario=scen, n_layers=8, proj_dim=64, n_seeds=6)
        print(f"{scen:>12} {st['exact_cross_seed']:>+12.2f} {st['ghost_cross_seed']:>+12.2f} "
              f"{st['ghostA_vs_exactB']:>+17.2f} {st['ghost_vs_exact_sameseed']:>+18.2f}")
    print("\n   'ghostA_vs_exactB' is the synthetic analog of the real ghost-run-CSV vs\n"
          "   autograd-run-CSV (different trajectories + different backend = the 0.02).")


if __name__ == "__main__":
    main()
