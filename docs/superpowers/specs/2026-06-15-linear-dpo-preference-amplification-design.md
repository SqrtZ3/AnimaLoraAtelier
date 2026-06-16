# Linear-DPO Preference Amplification (in-loop, on-policy) — Design

## Context

The dirty-data line (GAF) is retired: on a *uniformly* "not-clean-enough" set there is
no stable dirty subset to find, so GAF's per-image trust is trajectory noise (the
ghost-vs-autograd 480-step runs were rank-uncorrelated, Spearman +0.02; a confound-free
CPU probe reproduced this for the *exact* backend on a uniform-borderline batch). See
`tests/test_gaf_ghost_fidelity.py` and the `dirty-data-robustness` memory.

We pivot from "reduce dataset disadvantage" to "amplify fitting/aesthetic advantage".
Two facts make preference optimization (not representation alignment) the right lever:

1. The Anima base card states it did **no aesthetic-score training** → there is genuine
   aesthetic headroom the base never climbed.
2. Our hand-curated training sets are aesthetic by construction → we own a set of
   **positives**, but no reward model and no preference pairs.

DPO matches this asset: it needs preference *pairs*, not a reward model. We choose
**Linear-DPO** (arXiv 2605.21123), a flow-matching-native DPO whose objective is an
**offline, reweighted denoising loss** — it reuses our existing per-sample loss almost
verbatim, and its linear-utility + η-clip + EMA-reference design is explicitly engineered
against over-optimization (our top risk: style drift). Reward-RL methods (Flow-GRPO,
LeapAlign) are rejected here: they need a reward model and an in-loop sampling rollout
for *ranking*, and target a different product (preference reward, not artist fidelity).

Because we have no human-preference pairs for the artist, we **construct** pairs:
`winner = real dataset image` (the only ground-truth "good"; never generated),
`loser = a generation from the current policy for the same caption`. This makes the
objective simultaneously a fitting amplifier (push toward data, away from current output)
and an aesthetic shaper, with no external scorer.

## Goals

- Add an **opt-in, default-off** Linear-DPO post-training phase that runs as a short
  continuation from an already-converged LoRA, on the existing single cloud machine, in
  a single process.
- Reuse the existing per-sample velocity loss, the existing in-loop sampler, the existing
  LoRA injector, optimizer, checkpointing, and monitor.
- Generate losers **in-loop, on-policy, in coarse rounds** (regenerate the loser pool
  every `dpo_regen_every` steps using the current weights), keeping losers in **latent
  space** (no VAE decode/encode).
- Keep the reference model nearly free under LoRA (shared frozen base + a frozen/EMA copy
  of the small adapter weights — no second base in memory).
- Make the decision core (the ω' weight + the paired loss) unit-testable **without** the
  2B model or a real VAE.

## Non-Goals

- Do not train or require a reward / aesthetic scorer (the asset is a positive set).
- Do not introduce in-loop sampling for *ranking* (winners are always the real images).
- Do not change the LoRA/LoKr export format or the Comfy inference path.
- Do not make this part of the main from-scratch SFT run in the first slice (combined
  SFT×DPO is a deliberate later stress-test; v1 is a continuation from a converged LoRA).
- Do not run anything on the local 8GB box (2B is cloud-only, unchanged).

## Chosen Approach

A new opt-in training mode `dpo_enabled: true`, orchestrated by a `DpoController`
(mirrors the `GafController` opt-in/default-off pattern in `trainer/gaf.py`), wrapping
three pieces: a **loser pool** (on-policy, regenerated in rounds), a **reference adapter**
(frozen converged LoRA, optional EMA), and the **Linear-DPO loss** (reuses per-sample loss).

### Loss (mapped to our flow-matching / velocity setup)

Per-sample denoising error reuses the existing path (`per_sample_loss` /
`masked_token_loss` in `anima_train.py` ~2313/2336, velocity target, huber/snr unchanged):

```
L_θ(x0, c, t, ε) = ‖ v* − v_θ(x_t, t, c) ‖²            # existing per_sample_loss
x_t             = (1-t)·x0 + t·ε        (CONST flow)   # existing noising
D_θ(x0)         = L_θ(x0) − L_ref(x0)                  # policy minus frozen reference
Δ               = D_θ(x0^w) − D_θ(x0^l)               # winner minus loser margin
ω'              = clip( 0.2·(β̄·Δ) + 0.5 , η , 1 )     # linear utility, DETACHED
L_DPO           = mean_pairs[ sg(ω') · ( L_θ(x0^w) − L_θ(x0^l) ) ]
```

- `sg` = stop-gradient; gradient flows only through `(L_θ(w) − L_θ(l))`. Minimizing pushes
  the policy to denoise the **real** image better and its **own sample** worse.
- `β̄ = dpo_beta` (constant for v1; the paper's `β·T·λ(t)` folds the schedule weight — we
  keep our existing weighting separate and tune a single `dpo_beta`).
- `η = dpo_eta` (default 0.01; paper-optimal; the clip floor keeps a small sustained push
  and guards against over-optimization — too large η caused over-opt in the paper).
- **Variance reduction:** within a pair, share the sampled `t` and noise `ε` between
  winner and loser (as Diffusion-DPO does). This makes Δ a paired difference at matched t.
- v1 phase loss is **pure** Linear-DPO. The winner term `L_θ(w)` already contains a
  weighted fit-to-data signal; an optional `dpo_sft_anchor_lambda` can add a plain SFT
  term on winners if drift appears.

### Reference model (cheap under LoRA)

Reference = base + **frozen converged adapter**. Implementation (v1): the `DpoController`
holds a frozen copy of each trainable adapter tensor (`lokr_w1, lokr_w2_a, lokr_w2_b,
dora_scale, …`). To compute `y_ref`, a context manager swaps each adapter param's `.data`
to the frozen copy, runs the two reference forwards under `torch.no_grad()`, then restores.
Adapter-type-agnostic, minimal core change, ~free memory (small tensors), no second base.

- v1: reference is **fixed** (the converged snapshot) — maximal anchoring, minimal drift,
  zero EMA bookkeeping. `γ = 1`.
- Follow-up knob `dpo_ref_ema` (e.g. 0.995, the paper's best): EMA-update the frozen copy
  from the policy each step. Use only if a fixed reference plateaus.
- Later optimization: a `use_reference` forward flag inside `LoRALinear` (buffer weights)
  to avoid per-step `.data` swaps — not required for v1.

### Loser generation (in-loop, on-policy, latent-space)

Reuse `trainer/sampling.py`. `sample_image` (line 206) already does the full ER-SDE CONST
sampling with CFG, but **VAE-decodes to PIL at the end** (line 354–364). Factor out:

```
sample_latent(model, cross_cond, cross_uncond, h, w, steps, cfg, sampler, scheduler,
              injector, seed) -> latent x0   # the value of `x` at line 354, no decode
```

`sample_image` becomes `sample_latent` + decode. The loser pool stores **latents**
(`[1,16,1,h//8,w//8]`, directly compatible with the training noising pipeline) — no VAE
decode/encode in the DPO path at all.

Regeneration cadence (coarse rounds, not per-step):

- Every `dpo_regen_every` steps (round boundary; reuse the existing sampling-interval hook
  pattern at `anima_train.py` ~1851–1892), regenerate the loser for each
  `(caption, bucket)` using the **current policy** weights at eval.
- Captions/resolutions come from the dataset (loser shares the winner's caption & bucket).
- v1 round-1 losers come from the converged LoRA (= the policy at step 0).

Cost levers (configurable):

- `dpo_loser_steps` (default 14, vs 25–30 for previews): fewer sampler steps; a slightly
  weaker loser is acceptable/desirable.
- `dpo_loser_cfg` (default 1.0): no-CFG halves forwards per step and yields a naturally
  weaker loser — needs a small no-CFG branch in `sample_latent` (skip the uncond forward).
- `dpo_loser_subset` (default 1.0): regenerate only a fraction of the pool per round.
- Batched generation (B>1) is a later optimization; v1 may call per-caption (B=1).

### Per-step cost (v1)

Per DPO step: 2 policy forwards (winner, loser; with grad) + 2 reference forwards
(no grad) ≈ **4 forwards** (backward through 2 images). Plus generation amortized over
rounds: one round ≈ `N_pairs × dpo_loser_steps × (1 or 2)` forwards. For ~98 pairs at 14
steps no-CFG ≈ ~1.4k forwards/round ≈ a few hundred DPO steps' worth — so keep rounds few
(3–5) and `dpo_regen_every` ≥ ~one epoch. **Generation, not training, dominates a short
phase** → cadence is the cost knob.

## Config (trainer/config.py, `dpo_*`, all default-off)

```
dpo_enabled: false
dpo_beta: <tune>            # β̄ in ω'
dpo_eta: 0.01              # linear-utility clip floor
dpo_ref_ema: 1.0           # 1.0 = fixed reference (v1); 0.995 = EMA follow-up
dpo_regen_every: <≈1 epoch in steps>   # round boundary
dpo_loser_steps: 14
dpo_loser_cfg: 1.0
dpo_loser_subset: 1.0
dpo_sft_anchor_lambda: 0.0 # optional winner SFT anchor if drift appears
dpo_share_noise: true      # share (t, ε) within a pair (variance reduction)
dpo_log_path: ""           # per-pair Δ / ω' / margin CSV (diagnostic)
```

## Training-loop integration (anima_train.py)

1. Build `DpoController` near the GAF init (~1950) when `dpo_enabled`: snapshot converged
   adapter → reference; build empty loser pool; register the regen hook.
2. At round boundaries, `dpo.regen_losers(model, dataset_captions)` (eval, no_grad,
   `sample_latent`) refreshes the latent pool.
3. In the step body, when DPO mode is active, replace the standard single-sample loss with:
   draw winner latents (from the dataset cache) + matched loser latents (from the pool),
   share `(t, ε)`, compute `L_θ(w), L_θ(l)` (policy, with grad) and `L_ref(w), L_ref(l)`
   (reference swap, no grad), form ω' and `L_DPO`, backward.
4. Monitor/preview/checkpoint paths unchanged. Optional per-pair CSV for Δ/ω'.

## Run-1 (locked decisions)

- Continuation from the **already-converged LoRA**; reference = that snapshot, **fixed**.
- Losers: **converged-LoRA self-generation**, in-loop, latent-space, coarse rounds.
- Single process, single machine.
- Single variable: DPO phase on/off.
- **Judge (not SFT loss):** fixed eval prompts+seeds, side-by-side SFT-LoRA vs SFT+DPO —
  aesthetic ↑ **and** style not drifted (drift = veto). Optional waifu-scorer as a
  *diagnostic only* (never a training signal; cf. the LAION-aesthetic ban).

## Risks & mitigations

- **Style drift / over-optimization** (top risk): fixed reference anchor + η-clip + small
  `dpo_beta` + reuse SFT (small) LR + short phase + optional SFT anchor. Linear-DPO is
  built for this.
- **Feedback loop / diversity collapse** from on-policy losers: reference KL-anchor +
  limited rounds; watch the per-pair margin CSV.
- **Signal self-tapers**: as the policy improves, its losers improve → winner-loser margin
  shrinks → gradient fades. This is the built-in stopping criterion, not a bug.
- **In-loop sampling plumbing**: `sample_latent` must run eval + `no_grad`, isolate its RNG
  generator from the training noise stream, restore train/eval mode in `finally` (the
  existing `sample_image` already guards this), and freeze the base.
- **Self-constructed pairs are unvalidated** (the paper used human-preference pairs): this
  run is itself the test.

## Testability (no 2B model / no real VAE)

- **Decision core**: feed synthetic per-sample scalars `L_θ(w/l), L_ref(w/l)` →
  assert ω' clip range `[η,1]`, Δ sign, and that the loss gradient pushes `L_θ(w)` down /
  `L_θ(l)` up. Pure tensor math, CPU.
- **Reference swap**: on a tiny `nn.Linear` adapter, assert `y_ref` uses frozen weights and
  the trainable `.data` is restored exactly afterward, and that no grad leaks to reference.
- **sample_latent**: assert it returns a latent of the training shape and never calls the
  VAE decode path; assert no-CFG branch does one forward/step.
- **DpoController**: round gating, pool refresh count, fixed-vs-EMA reference update.

## Phased rollout

1. v1: continuation, fixed reference, self-gen losers, coarse rounds (this spec).
2. If it works: EMA reference (`dpo_ref_ema=0.995`), more rounds (true iterative DPO),
   batched generation.
3. Later stress-test: combined SFT×DPO in one run to surface interaction problems
   (deliberately, with attribution traded away).
```
