# N-Token Bucketing + Per-Block torch.compile Fast Path — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a configurable constant/N-token bucketing scheme to both the training repo and the Drop Studio dataset tool, then make the training repo's existing FiT token path compile-fast (per-block `torch.compile`) under that fixed-shape regime — without disturbing the default ARB + eager paths.

**Architecture:** A single pure function `generate_token_buckets(token_counts, max_ar, …)` is implemented **identically in both repos** (enforced by a shared golden fixture). The training `BucketManager` and the tool's `webapp/buckets.py` gain a token-bucket branch that calls it; image export resamples (up or down) to exact bucket pixels. For speed, training routes constant-token batches through the **already-tested** `forward_packed_tokens` path (uniform `N`, all-valid mask) and `torch.compile`s `block.forward_tokens`; RoPE is computed eagerly before the block loop so the compiled region is graph-break-free.

**Tech Stack:** Python 3.11 (local tests via `D:\ArtificialIntelligence\ComfyUI-aki-v1.5\python`), PyTorch (Inductor compile validated on cloud Ubuntu), FastAPI + React (Drop Studio), pytest/unittest.

---

## Design & Shared Contract

### Token math
Anima latent patching is VAE downsample 8 × patch 2 = **16 px per token axis**. Token count `T = (W // 16) * (H // 16)`. A "token family" = all integer factor pairs `(wp, hp)` with `wp*hp == T`; aspect ratios available in a family are exactly the `wp:hp` ratios (divisor-determined → pick **highly composite** `T`).

### Canonical generator (byte-identical in both repos)
```python
def generate_token_buckets(token_counts, max_aspect_ratio=2.0, patch_pixels=16,
                           min_dim_px=256, max_dim_px=4096):
    """Sorted list of (W, H) pixel buckets whose token count
    (W//patch_pixels)*(H//patch_pixels) is exactly one of token_counts.
    Deterministic: sorted by (token_count, W, H). Both orientations arise
    naturally from factor enumeration."""
    min_p = max(1, int(min_dim_px) // patch_pixels)
    max_p = max(min_p, int(max_dim_px) // patch_pixels)
    out = set()
    for T in token_counts:
        T = int(T)
        for wp in range(1, T + 1):
            if T % wp:
                continue
            hp = T // wp
            if wp < min_p or wp > max_p or hp < min_p or hp > max_p:
                continue
            ar = (wp / hp) if wp >= hp else (hp / wp)
            if ar > float(max_aspect_ratio) + 1e-9:
                continue
            out.add((wp * patch_pixels, hp * patch_pixels))
    return sorted(out, key=lambda wh: ((wh[0]//patch_pixels)*(wh[1]//patch_pixels), wh[0], wh[1]))
```
- **Default counts:** `[4032, 4200]` (proven by the reference; 4032=63×64 covers near-square, 4200 fills mid-AR gaps). `min_dim_px=512, max_dim_px=2016` for the ~1MP tier (RoPE per-axis cap 126 patches < 256).
- **N=1 extreme:** `token_counts=[4032]` → one graph. **Larger images:** add `k²·T` counts (×k each axis preserves the AR grid).
- **Resampling allowed (up & down).** No no-upscale guard is added; bucket selection snaps by nearest AR, then cover-crop/resample to exact `(W,H)`.

### Golden fixture (cross-repo contract)
A frozen JSON of the canonical config's output is committed to **both** repos at `tests/fixtures/token_buckets_canonical.json`. Both repos' tests assert `generate_token_buckets(canonical) == fixture`. Generated once (Phase 1 Task 2), then pinned — any drift between the two implementations fails a test.

Canonical config: `token_counts=[4032,4200], max_aspect_ratio=2.0, patch_pixels=16, min_dim_px=512, max_dim_px=2016`.

### Training config keys (added to `trainer/config.py` `YAML_TO_ARGS`)
| YAML key | arg attr | default | meaning |
|---|---|---|---|
| `token_bucket` | `token_bucket` | `false` | enable token-bucket mode (off → ARB unchanged) |
| `token_bucket_counts` | `token_bucket_counts` | `"4032,4200"` | comma list or list of T |
| `token_bucket_max_aspect_ratio` | `token_bucket_max_aspect_ratio` | `2.0` | AR cap for generation |
| `token_bucket_min_dim` | `token_bucket_min_dim` | `512` | min pixel dim |
| `token_bucket_max_dim` | `token_bucket_max_dim` | `2016` | max pixel dim (RoPE-bound) |
| `torch_compile` | `torch_compile` | `false` | per-block compile (requires token_bucket) |
| `compile_mode` | `compile_mode` | `null` | inductor mode (e.g. `reduce-overhead`) |

### Compile approach (training repo, Goal B)
- Add `MiniTrainDIT.compile_blocks(backend="inductor", mode=None)`: sets `self._blocks_compiled=True`, raises `torch._dynamo.config.cache_size_limit`, and replaces each `block.forward_tokens` with `torch.compile(block.forward_tokens, backend=…, dynamic=False, mode=…)`.
- When `torch_compile and token_bucket`, training forward uses `forward_packed_tokens` (uniform `N`, full mask) instead of the grid `forward`. RoPE (`_packed_rope_from_grid`, which has `.item()` syncs) stays **outside** the compiled block (already computed once before the loop at core.py:1361), so the compiled region is clean.
- Equivalence guardrail: existing `test_fit_packed_*` already assert packed-token ≈ grid for a full image; add a compile-on/off numerical-equivalence test (CPU, `backend="eager"`).
- **Speed validation runs on cloud Ubuntu** (Inductor). Local Windows only runs correctness (`backend="eager"` / `aot_eager`).

### Cache invalidation
Latent cache keys must include the token-bucket config (counts + dims). Changing `token_bucket_counts` ⇒ cache miss/rebuild. Implemented in Phase 1 Task 6.

### File Structure
**Training repo** (`D:\ArtificialIntelligence\animafix\anima-lora-train\AnimaLoraToolkit`):
- Create `trainer/token_buckets.py` — `generate_token_buckets` (single responsibility: pure bucket math).
- Modify `trainer/data.py` — `BucketManager.__init__` token branch.
- Modify `trainer/config.py` — `YAML_TO_ARGS` + defaults.
- Modify `anima_train.py` — wire config → `BucketManager`; call `model.compile_blocks()`; route forward.
- Modify `models/anima_modeling_core.py` — add `compile_blocks()` + `_blocks_compiled` flag.
- Create `tests/test_token_buckets.py`, `tests/test_compile_equivalence.py`.
- Create `tests/fixtures/token_buckets_canonical.json`.

**Tool repo** (`D:\Datasets\1_droptools`):
- Modify `webapp/buckets.py` — add `generate_token_buckets` (identical) + `choose_token_bucket`.
- Modify `webapp/routes_crop.py` — token-bucket export branch in `process_batch`/`auto_bucket_all`.
- Modify `webapp/schemas.py` — request fields `token_bucket`, `token_bucket_counts`.
- Create `tests/test_token_buckets.py` + copy `tests/fixtures/token_buckets_canonical.json`.

---

## Phase 1 — Training-repo token bucketing (data + config)

### Task 1: Pure bucket generator
**Files:** Create `AnimaLoraToolkit/trainer/token_buckets.py`; Test `AnimaLoraToolkit/tests/test_token_buckets.py`

- [ ] **Step 1: Write failing test**
```python
# tests/test_token_buckets.py
from AnimaLoraToolkit.trainer.token_buckets import generate_token_buckets

def test_all_buckets_hit_exact_token_count():
    bks = generate_token_buckets([4032, 4200], max_aspect_ratio=2.0,
                                 min_dim_px=512, max_dim_px=2016)
    assert bks, "expected non-empty bucket list"
    for w, h in bks:
        assert (w // 16) * (h // 16) in (4032, 4200)
        assert max(w / h, h / w) <= 2.0 + 1e-9
        assert 512 <= w <= 2016 and 512 <= h <= 2016

def test_near_square_present_for_4032():
    bks = generate_token_buckets([4032], max_aspect_ratio=2.0,
                                 min_dim_px=512, max_dim_px=2016)
    assert (1008, 1024) in bks  # 63x64

def test_n1_single_token_count():
    bks = generate_token_buckets([4032], max_aspect_ratio=2.0,
                                 min_dim_px=512, max_dim_px=2016)
    assert {(w // 16) * (h // 16) for w, h in bks} == {4032}

def test_deterministic_sorted():
    bks = generate_token_buckets([4032, 4200], max_aspect_ratio=2.0,
                                 min_dim_px=512, max_dim_px=2016)
    assert bks == sorted(bks, key=lambda wh: ((wh[0]//16)*(wh[1]//16), wh[0], wh[1]))
```
- [ ] **Step 2: Run, verify fail** — `python -m pytest AnimaLoraToolkit/tests/test_token_buckets.py -v` → FAIL (module missing).
- [ ] **Step 3: Implement** — paste the canonical `generate_token_buckets` (see Design) into `trainer/token_buckets.py`, with module docstring.
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** — `feat(trainer): pure N-token bucket generator`.

### Task 2: Freeze the golden fixture
**Files:** Create `AnimaLoraToolkit/tests/fixtures/token_buckets_canonical.json`; modify `tests/test_token_buckets.py`

- [ ] **Step 1:** Generate fixture once:
```bash
"D:\ArtificialIntelligence\ComfyUI-aki-v1.5\python" -c "import json; from AnimaLoraToolkit.trainer.token_buckets import generate_token_buckets as g; json.dump(g([4032,4200],2.0,16,512,2016), open('AnimaLoraToolkit/tests/fixtures/token_buckets_canonical.json','w'))"
```
- [ ] **Step 2: Add pinning test**
```python
import json, os
def test_matches_golden_fixture():
    here = os.path.dirname(__file__)
    expect = [tuple(x) for x in json.load(open(os.path.join(here, "fixtures", "token_buckets_canonical.json")))]
    got = generate_token_buckets([4032, 4200], 2.0, 16, 512, 2016)
    assert got == expect
```
- [ ] **Step 3: Run, verify pass.** **Step 4: Commit** — `test(trainer): pin canonical token-bucket fixture`.

### Task 3: BucketManager token branch
**Files:** Modify `AnimaLoraToolkit/trainer/data.py:149-169` (`BucketManager.__init__`); Test `tests/test_token_buckets.py`

- [ ] **Step 1: Failing test**
```python
from AnimaLoraToolkit.trainer.data import BucketManager
def test_bucketmanager_token_mode():
    bm = BucketManager(token_bucket=True, token_bucket_counts=[4032],
                       token_bucket_max_aspect_ratio=2.0,
                       token_bucket_min_dim=512, token_bucket_max_dim=2016)
    assert {(w//16)*(h//16) for w, h in bm.buckets} == {4032}
    w, h = bm.get_bucket(1000, 1000)  # near-square source
    assert (w//16)*(h//16) == 4032
```
- [ ] **Step 2: Verify fail** (unexpected kwargs).
- [ ] **Step 3: Implement** — add params to `__init__` signature with defaults `token_bucket=False, token_bucket_counts=None, token_bucket_max_aspect_ratio=2.0, token_bucket_min_dim=512, token_bucket_max_dim=2016`; before the existing `self.buckets = self._generate(...)`:
```python
if token_bucket:
    from .token_buckets import generate_token_buckets
    counts = token_bucket_counts or [4032, 4200]
    if isinstance(counts, str):
        counts = [int(c) for c in counts.split(",") if c.strip()]
    self.token_bucket = True
    self.buckets = generate_token_buckets(
        counts, max_aspect_ratio=token_bucket_max_aspect_ratio,
        min_dim_px=int(token_bucket_min_dim), max_dim_px=int(token_bucket_max_dim))
else:
    self.token_bucket = False
    self.buckets = self._generate(min_reso, max_reso, step, self.base_resos)
```
(Existing `get_bucket`, `_score_bucket`, `_bucket_allowed_for_image` are reused unchanged — selection is bucket-set-agnostic.)
- [ ] **Step 4: Run, verify pass.** **Step 5: Commit** — `feat(trainer): BucketManager token-bucket mode`.

### Task 4: Config keys
**Files:** Modify `AnimaLoraToolkit/trainer/config.py` (`YAML_TO_ARGS` ~line 68, and defaults ~line 263)

- [ ] **Step 1: Failing test**
```python
from AnimaLoraToolkit.trainer.config import YAML_TO_ARGS, DEFAULTS  # adjust import to actual names
def test_token_bucket_keys_present():
    for k in ("token_bucket","token_bucket_counts","token_bucket_max_aspect_ratio",
              "token_bucket_min_dim","token_bucket_max_dim","torch_compile","compile_mode"):
        assert k in YAML_TO_ARGS
```
- [ ] **Step 2: Verify fail.**
- [ ] **Step 3: Implement** — add the 7 keys to `YAML_TO_ARGS` (identity mapping) right after `"bucket_drop_last"`, and corresponding defaults to the defaults dict (`token_bucket=False, token_bucket_counts="4032,4200", token_bucket_max_aspect_ratio=2.0, token_bucket_min_dim=512, token_bucket_max_dim=2016, torch_compile=False, compile_mode=None`). (Inspect the actual defaults structure near line 263 first; match its style.)
- [ ] **Step 4: Run, verify pass.** **Step 5: Commit** — `feat(config): token-bucket + compile keys`.

### Task 5: Wire BucketManager in anima_train.py
**Files:** Modify `AnimaLoraToolkit/anima_train.py:992-1004`

- [ ] **Step 1:** Pass token-bucket args into `BucketManager(...)`:
```python
bucket_mgr = BucketManager(
    base_reso=args.resolution, min_reso=bucket_min_reso, max_reso=bucket_max_reso,
    step=bucket_step, base_resos=bucket_base_resos,
    min_base_reso=int(getattr(args, "bucket_min_base_reso", 0) or 0),
    max_base_reso=int(getattr(args, "bucket_max_base_reso", 0) or 0),
    base_reso_step=int(getattr(args, "bucket_base_reso_steps", 256) or 256),
    no_upscale=bool(getattr(args, "bucket_no_upscale", False)),
    max_upscale=float(getattr(args, "bucket_max_upscale", 0.0) or 0.0),
    max_aspect_ratio=float(getattr(args, "bucket_max_aspect_ratio", 2.0) or 2.0),
    token_bucket=bool(getattr(args, "token_bucket", False)),
    token_bucket_counts=getattr(args, "token_bucket_counts", None),
    token_bucket_max_aspect_ratio=float(getattr(args, "token_bucket_max_aspect_ratio", 2.0) or 2.0),
    token_bucket_min_dim=int(getattr(args, "token_bucket_min_dim", 512) or 512),
    token_bucket_max_dim=int(getattr(args, "token_bucket_max_dim", 2016) or 2016),
)
```
- [ ] **Step 2:** Update the `[BucketManager]` log line to also print `token_bucket=%s`.
- [ ] **Step 3: Smoke test** — run a dataset-only dry run (no GPU) to confirm bucket list logs. **Step 4: Commit** — `feat(train): wire token-bucket config`.

### Task 6: Cache key includes token-bucket config
**Files:** Modify wherever the latent cache key/signature is built (grep `cache` in `trainer/data.py` / cache module); Test in `tests/test_token_buckets.py`

- [ ] **Step 1:** Locate the cache-signature builder (`grep -n "cache" trainer/data.py` and the cache module). Add the token-bucket config (sorted counts + dims + max_ar) to the signature.
- [ ] **Step 2:** Add a test asserting two different `token_bucket_counts` produce different cache signatures.
- [ ] **Step 3: Run, verify pass.** **Step 4: Commit** — `fix(cache): invalidate on token-bucket config change`.

---

## Phase 2 — Drop Studio dataset tool (`D:\Datasets\1_droptools`)

### Task 7: Identical generator + golden fixture
**Files:** Modify `webapp/buckets.py`; Create `tests/test_token_buckets.py`, `tests/fixtures/token_buckets_canonical.json`

- [ ] **Step 1: Failing test** (mirror Task 1 + the golden-fixture pin, importing from `webapp.buckets`).
- [ ] **Step 2: Verify fail.**
- [ ] **Step 3: Implement** — paste the **identical** `generate_token_buckets` into `webapp/buckets.py`; **copy** `token_buckets_canonical.json` from the training repo into `tests/fixtures/`. (Identical code + identical fixture = enforced contract.)
- [ ] **Step 4: Run, verify pass** (`"D:\ArtificialIntelligence\ComfyUI-aki-v1.5\python" -m pytest tests/test_token_buckets.py -v` from the tool dir). **Step 5: Commit** — `feat(buckets): N-token bucket generator (matches trainer)`.

### Task 8: choose_token_bucket selector
**Files:** Modify `webapp/buckets.py`; Test `tests/test_token_buckets.py`

- [ ] **Step 1: Failing test**
```python
from webapp.buckets import choose_token_bucket
def test_choose_token_bucket_nearest_ar():
    w, h = choose_token_bucket(1000, 1000, token_counts=[4032])
    assert (w // 16) * (h // 16) == 4032
```
- [ ] **Step 2: Verify fail.**
- [ ] **Step 3: Implement** — mirror `choose_arb_bucket` (`buckets.py:171`) but source buckets from `generate_token_buckets`; reuse `_bucket_score`. Resampling allowed → no `no_upscale` filter by default.
- [ ] **Step 4: Run, verify pass.** **Step 5: Commit** — `feat(buckets): token-bucket selector`.

### Task 9: Export branch + request schema
**Files:** Modify `webapp/schemas.py`; `webapp/routes_crop.py:141-300` (`process_batch`, `auto_bucket_all`)

- [ ] **Step 1:** Add to the relevant Pydantic request models in `schemas.py`: `token_bucket: bool = False`, `token_bucket_counts: Optional[str] = None`.
- [ ] **Step 2:** In `process_batch`/`auto_bucket_all`, when `req.token_bucket`, replace `calculate_output_size(...)` with `choose_token_bucket(item["w"], item["h"], token_counts=...)`; existing `_export_crop` then resamples (LANCZOS/RealESRGAN) + center-crops to the exact bucket — no other change.
- [ ] **Step 3: Manual smoke** — start `python server.py`, POST `/api/process_batch {token_bucket:true, token_bucket_counts:"4032"}` on a tiny folder; confirm every output's `(W//16)*(H//16)==4032`.
- [ ] **Step 4: Commit** — `feat(export): token-bucket export mode`.

---

## Phase 3 — Training-repo per-block torch.compile fast path

### Task 10: compile_blocks() on the model
**Files:** Modify `AnimaLoraToolkit/models/anima_modeling_core.py` (`MiniTrainDIT`, near `__init__` and after the block list ~1071); Test `tests/test_compile_equivalence.py`

- [ ] **Step 1: Failing test** (CPU, `backend="eager"` so it runs on Windows):
```python
import torch
def test_compile_blocks_sets_flag_and_runs(monkeypatch):
    from AnimaLoraToolkit.models.anima_modeling_core import MiniTrainDIT
    m = _tiny_minitraindit()   # helper: small dims, 2 blocks (define in test)
    m.eval()
    m.compile_blocks(backend="eager")
    assert getattr(m, "_blocks_compiled", False) is True
    # forward still runs after compile
    out = _run_packed_full_image(m)   # helper builds a full-mask packed batch
    assert out.shape[0] == 1
```
- [ ] **Step 2: Verify fail** (`compile_blocks` missing).
- [ ] **Step 3: Implement** in `MiniTrainDIT`:
```python
def compile_blocks(self, backend: str = "inductor", mode=None):
    """Per-block torch.compile of the token forward path. Requires constant-token
    bucketing so the sequence length (and thus the traced graph) is fixed."""
    import torch._dynamo as _dynamo
    self._blocks_compiled = True
    _dynamo.config.cache_size_limit = max(_dynamo.config.cache_size_limit, 32)
    kwargs = {"backend": backend, "dynamic": False}
    if mode is not None:
        kwargs["mode"] = mode
    for block in self.blocks:
        block.forward_tokens = torch.compile(block.forward_tokens, **kwargs)
    import logging; logging.getLogger(__name__).info(
        "compile_blocks: compiled %d block.forward_tokens (backend=%s, mode=%s)",
        len(self.blocks), backend, mode)
```
Add `self._blocks_compiled = False` in `__init__`.
- [ ] **Step 4: Run, verify pass** (`backend="eager"`). **Step 5: Commit** — `feat(model): compile_blocks for token path`.

### Task 11: Numerical equivalence (compile off vs on, packed vs grid)
**Files:** `tests/test_compile_equivalence.py`

- [ ] **Step 1:** Test that for one full-mask image: grid `forward` ≈ `forward_packed_tokens` (reuse/extend existing `test_fit_packed_*` assertions), AND packed output is unchanged after `compile_blocks(backend="eager")` (atol/rtol e.g. 1e-4):
```python
def test_packed_equivalence_compile_off_on():
    m = _tiny_minitraindit(); m.eval()
    args = _full_image_inputs()           # x, t, cross, grid, mask, size
    ref = m.forward_packed_tokens(**args)
    m.compile_blocks(backend="eager")
    got = m.forward_packed_tokens(**args)
    torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)
```
- [ ] **Step 2: Run, verify pass.** **Step 3: Commit** — `test(model): compile equivalence`.

### Task 12: Route training forward + call compile_blocks
**Files:** Modify `AnimaLoraToolkit/anima_train.py` (after `load_anima_model` ~857; forward call site in the training step — grep `prepare_embedded_sequence`/`model(`/`forward_packed_tokens`)

- [ ] **Step 1:** After model load, if `args.torch_compile`: assert `args.token_bucket` (fail-fast with a clear message that compile requires constant-token), then `model.compile_blocks(mode=getattr(args,"compile_mode",None))`.
- [ ] **Step 2:** Ensure the training step uses `forward_packed_tokens` when `token_bucket` (and thus a uniform `N` + full mask per batch). If the trainer currently only calls grid `forward`, add the token-path branch mirroring the existing `fit_packed_training` wiring (reuse its loss path in `trainer/objective.py`; confirm by reading the fit_packed branch there).
- [ ] **Step 3: Local correctness run** — tiny dataset, `token_bucket=true torch_compile=true compile_mode=null` with `dynamo_backend` forced to `eager`/`aot_eager` on Windows; confirm a few steps run and loss is finite.
- [ ] **Step 4: Commit** — `feat(train): compile fast path under token bucketing`.

### Task 13: Cloud Ubuntu speed validation (manual gate)
**Files:** none (validation + notes in plan)

- [ ] **Step 1:** On the cloud Ubuntu box, run identical short trainings: (A) `token_bucket=true, torch_compile=false`; (B) `token_bucket=true, torch_compile=true` (inductor). Record s/step + peak VRAM after warmup.
- [ ] **Step 2:** Confirm graph count is small (set `TORCH_LOGS=recompiles`); no per-step recompiles. If recompiles appear, check batch divisibility per bucket (partial last batch ⇒ extra graph) and `bucket_drop_last`.
- [ ] **Step 3:** Record numbers in `docs/superpowers/plans/2026-06-04-ntoken-bucketing-and-compile.md` results section. Commit notes.

---

## Self-Review

- **Spec coverage:** Goal A (N-token, both repos, dataset sync, soft-area, resampling) → Tasks 1–9. Goal B (native fixed-shape + per-block compile + equivalence + Ubuntu validation) → Tasks 10–13. Default ARB unchanged (Task 3 branch is opt-in). Existing tests untouched (new files only). Cache invalidation → Task 6. Cross-repo contract → golden fixture (Tasks 2, 7). ✅
- **Placeholder scan:** Phase 1–2 steps carry full code. Phase 3 Tasks 10–11 carry full method/test code; Tasks 12–13 reference exact files but require reading the `fit_packed`/objective wiring + on-device runs (flagged as iterative — not placeholders, but verification loops). Test helpers `_tiny_minitraindit`/`_full_image_inputs` must be authored in Task 10 from the real `MiniTrainDIT.__init__` signature.
- **Type consistency:** `generate_token_buckets(token_counts, max_aspect_ratio, patch_pixels, min_dim_px, max_dim_px)` identical in both repos; `BucketManager(..., token_bucket, token_bucket_counts, token_bucket_max_aspect_ratio, token_bucket_min_dim, token_bucket_max_dim)`; `compile_blocks(backend, mode)`; flag `_blocks_compiled`. Consistent across tasks. ✅
- **Open risk:** Task 12 assumes the FiT token path + its objective loss are reusable for constant-token full-mask training. If `trainer/objective.py` cannot be reused as-is, fall back to compiling the grid `forward` after removing the `b t h w d` rearranges in `Block.forward` (core.py:827-860) — heavier; treat as a contingency, not the default.
