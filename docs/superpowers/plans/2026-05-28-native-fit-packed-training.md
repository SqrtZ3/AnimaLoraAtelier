# Native-FiT Packed Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in native-first FiT-style packed-token training path that preserves source image detail by default while keeping existing Comfy-compatible inference/export behavior.

**Architecture:** The old ARB grid path remains default. The new path pads images to VAE+patch alignment, groups by token count, patchifies noisy latents into variable-length token batches, applies masked full self-attention, and computes loss only on valid tokens.

**Tech Stack:** Python, PyTorch, unittest, existing Anima trainer modules.

---

## File Structure

- Modify `AnimaLoraToolkit/trainer/data.py`: native FiT sizing helpers, dataset mode, token-count-aware sampler, collate for padded native image batches.
- Modify `AnimaLoraToolkit/trainer/config.py`: YAML mapping and defaults for FiT flags.
- Modify `AnimaLoraToolkit/anima_train.py`: argparse flags, dataset selection, training-loop branch, unsupported-combination guards.
- Modify `AnimaLoraToolkit/trainer/objective.py`: masked token loss helper and packed forward helper.
- Modify `AnimaLoraToolkit/models/anima_modeling_core.py`: token patchify/unpatchify, masked attention path, packed-token block/model forward.
- Create `AnimaLoraToolkit/tests/test_fit_packed_data.py`: sizing/collate/sampler/config tests.
- Create `AnimaLoraToolkit/tests/test_fit_packed_objective.py`: masked token loss tests.
- Create `AnimaLoraToolkit/tests/test_fit_packed_model.py`: model token helper tests.

## Task 1: Native FiT Data Helpers And Config

**Files:**
- Modify: `AnimaLoraToolkit/trainer/data.py`
- Modify: `AnimaLoraToolkit/trainer/config.py`
- Modify: `AnimaLoraToolkit/anima_train.py`
- Create: `AnimaLoraToolkit/tests/test_fit_packed_data.py`

- [ ] **Step 1: Write failing data/config tests**

Add tests that import `trainer.data` and `trainer.config` and assert:

```python
def test_native_fit_size_preserves_source_when_aligned():
    plan = data_module.plan_native_fit_image(640, 384, max_tokens=1024, patch_size=2, vae_downsample=8)
    self.assertEqual((plan.width, plan.height), (640, 384))
    self.assertFalse(plan.was_resized)
    self.assertEqual(plan.token_count, 960)

def test_native_fit_size_pads_unaligned_source_without_scaling():
    plan = data_module.plan_native_fit_image(513, 777, max_tokens=65536, patch_size=2, vae_downsample=8)
    self.assertEqual((plan.width, plan.height), (528, 784))
    self.assertEqual((plan.source_width, plan.source_height), (513, 777))
    self.assertTrue(plan.was_padded)
    self.assertFalse(plan.was_resized)

def test_native_fit_over_budget_fails_by_default():
    with self.assertRaisesRegex(ValueError, "exceeds fit_max_tokens"):
        data_module.plan_native_fit_image(8192, 8192, max_tokens=65536, patch_size=2, vae_downsample=8)

def test_yaml_maps_fit_packed_flags():
    args = SimpleNamespace(**DEFAULTS)
    apply_yaml_config(args, {"fit_packed_training": True, "fit_max_tokens": 65536})
    self.assertTrue(args.fit_packed_training)
    self.assertEqual(args.fit_max_tokens, 65536)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
$env:UV_CACHE_DIR='D:\ArtificialIntelligence\animafix\anima-lora-train\.uv-cache'; uv run python -m unittest AnimaLoraToolkit.tests.test_fit_packed_data
```

Expected: fail because helpers/config keys do not exist yet.

- [ ] **Step 3: Implement minimal helpers and config**

Add:

- `NativeFitImagePlan` dataclass.
- `plan_native_fit_image(...)`.
- `fit_*` entries in `YAML_TO_ARGS` and `DEFAULTS`.
- argparse flags in `anima_train.py`.

- [ ] **Step 4: Run tests to verify pass**

Run the same unittest command. Expected: pass.

## Task 2: Native FiT Dataset Collate And Sampler

**Files:**
- Modify: `AnimaLoraToolkit/trainer/data.py`
- Modify: `AnimaLoraToolkit/tests/test_fit_packed_data.py`

- [ ] **Step 1: Write failing collate/sampler tests**

Add tests for:

```python
def test_collate_native_fit_pixels_pads_to_batch_max_and_masks_padding():
    batch = [
        {"pixel_values": torch.ones(3, 32, 48), "pixel_mask": torch.ones(1, 32, 48), "caption": "a", "image": "a.png", "token_count": 6},
        {"pixel_values": torch.ones(3, 48, 64), "pixel_mask": torch.ones(1, 48, 64), "caption": "b", "image": "b.png", "token_count": 12},
    ]
    out = data_module.collate_fn_fit_packed(batch)
    self.assertEqual(tuple(out["pixel_values"].shape), (2, 3, 48, 64))
    self.assertEqual(tuple(out["pixel_mask"].shape), (2, 1, 48, 64))
    self.assertEqual(out["fit_token_counts"].tolist(), [6, 12])

def test_fit_token_batch_sampler_groups_similar_token_counts():
    dataset = SimpleNamespace(token_count_for_index=[1024, 65536, 960, 64000])
    sampler = data_module.FitTokenBatchSampler(dataset, batch_size=2, max_tokens_per_batch=70000, shuffle=False)
    self.assertEqual(list(sampler), [[2, 0], [3], [1]])
```

- [ ] **Step 2: Run tests to verify they fail**

Run the Task 1 unittest command. Expected: fail because collate/sampler do not exist.

- [ ] **Step 3: Implement collate and sampler**

Add:

- `collate_fn_fit_packed`.
- `FitTokenBatchSampler`.
- dataset token count lookup support.

- [ ] **Step 4: Run tests to verify pass**

Run the same unittest command. Expected: pass.

## Task 3: Masked Token Loss

**Files:**
- Modify: `AnimaLoraToolkit/trainer/objective.py`
- Create: `AnimaLoraToolkit/tests/test_fit_packed_objective.py`

- [ ] **Step 1: Write failing masked loss tests**

Add tests:

```python
def test_masked_token_loss_ignores_padding_tokens():
    pred = torch.tensor([[[1.0], [10.0], [99.0]]])
    target = torch.tensor([[[0.0], [0.0], [0.0]]])
    mask = torch.tensor([[1, 1, 0]], dtype=torch.float32)
    loss = masked_token_loss(pred, target, mask, loss_type="mse")
    self.assertTrue(torch.allclose(loss, torch.tensor([50.5])))

def test_masked_token_loss_clamps_empty_mask():
    pred = torch.ones(1, 2, 1)
    target = torch.zeros(1, 2, 1)
    mask = torch.zeros(1, 2)
    loss = masked_token_loss(pred, target, mask)
    self.assertTrue(torch.isfinite(loss).all())
    self.assertEqual(float(loss.item()), 0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
$env:UV_CACHE_DIR='D:\ArtificialIntelligence\animafix\anima-lora-train\.uv-cache'; uv run python -m unittest AnimaLoraToolkit.tests.test_fit_packed_objective
```

Expected: import failure for `masked_token_loss`.

- [ ] **Step 3: Implement `masked_token_loss`**

Implement MSE/L1/Huber token loss returning `(B,)`, using only `mask > 0`.

- [ ] **Step 4: Run tests to verify pass**

Run the same unittest command. Expected: pass.

## Task 4: Packed Token Model Path

**Files:**
- Modify: `AnimaLoraToolkit/models/anima_modeling_core.py`
- Modify: `AnimaLoraToolkit/trainer/objective.py`
- Create: `AnimaLoraToolkit/tests/test_fit_packed_model.py`

- [ ] **Step 1: Write failing model tests**

Use a tiny model instance with `model_channels=32`, `num_blocks=1`, `num_heads=4`, `in_channels=16`, `out_channels=16`, `patch_spatial=2`, `patch_temporal=1`, `max_img_h=16`, `max_img_w=16`, `max_frames=1`, `crossattn_emb_channels=32`, `pos_emb_cls="rope3d"`.

Test:

```python
def test_patchify_unpatchify_round_trip():
    latents = torch.arange(1 * 16 * 1 * 4 * 6, dtype=torch.float32).view(1, 16, 1, 4, 6)
    tokens, grid, mask, size = model.patchify_latents_to_tokens(latents)
    restored = model.unpatchify_tokens(tokens, size)
    self.assertTrue(torch.equal(restored, latents))

def test_forward_packed_tokens_preserves_shape_and_masks_padding():
    tokens = torch.randn(1, 4, 64)
    grid = torch.tensor([[[0, 0, 1, 1], [0, 1, 0, 1]]])
    mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.float32)
    size = torch.tensor([[[2, 2]]], dtype=torch.int32)
    cross = torch.randn(1, 512, 32)
    out = model.forward_packed_tokens(tokens, torch.tensor([[0.5]]), cross, grid, mask, size)
    self.assertEqual(tuple(out.shape), (1, 4, 64))
    self.assertTrue(torch.allclose(out[:, 3], torch.zeros_like(out[:, 3]), atol=1e-6))
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
$env:UV_CACHE_DIR='D:\ArtificialIntelligence\animafix\anima-lora-train\.uv-cache'; uv run python -m unittest AnimaLoraToolkit.tests.test_fit_packed_model
```

Expected: methods missing.

- [ ] **Step 3: Implement packed token path**

Add:

- attention helper that accepts an optional boolean mask,
- `Attention.forward(..., mask=None)` support without changing old calls,
- `Block.forward_tokens`,
- model patchify/unpatchify helpers,
- model `forward_packed_tokens`.

- [ ] **Step 4: Run tests to verify pass**

Run the same unittest command. Expected: pass.

## Task 5: Training Loop Integration

**Files:**
- Modify: `AnimaLoraToolkit/anima_train.py`
- Modify: `AnimaLoraToolkit/trainer/objective.py`
- Modify: `AnimaLoraToolkit/tests/test_fit_packed_data.py`

- [ ] **Step 1: Write failing guard tests where practical**

Add tests for a pure helper if extracted:

```python
def test_fit_packed_rejects_cache_latents_by_default():
    args = SimpleNamespace(fit_packed_training=True, cache_latents=True)
    with self.assertRaisesRegex(RuntimeError, "cache_latents"):
        validate_fit_packed_training_args(args)
```

- [ ] **Step 2: Run tests to verify fail**

Run relevant unittest file. Expected: helper missing.

- [ ] **Step 3: Wire training branch**

In `anima_train.py`:

- choose `collate_fn_fit_packed` and `FitTokenBatchSampler` when enabled,
- encode padded pixels,
- interpolate/patchify pixel mask to token mask,
- patchify noisy/target,
- call `forward_packed_tokens`,
- compute `masked_token_loss`,
- reuse existing timestep weighting and optimizer flow,
- fail-fast for `cache_latents` and aux perceptual/spectral settings until explicitly supported.

- [ ] **Step 4: Run targeted tests**

Run all new tests. Expected: pass.

## Task 6: Verification

**Files:**
- All touched files.

- [ ] **Step 1: Run new test files**

```powershell
$env:UV_CACHE_DIR='D:\ArtificialIntelligence\animafix\anima-lora-train\.uv-cache'; uv run python -m unittest AnimaLoraToolkit.tests.test_fit_packed_data AnimaLoraToolkit.tests.test_fit_packed_objective AnimaLoraToolkit.tests.test_fit_packed_model
```

- [ ] **Step 2: Run existing multires tests**

```powershell
$env:UV_CACHE_DIR='D:\ArtificialIntelligence\animafix\anima-lora-train\.uv-cache'; uv run python -m unittest AnimaLoraToolkit.tests.test_multires_buckets
```

- [ ] **Step 3: Run py_compile**

```powershell
$env:UV_CACHE_DIR='D:\ArtificialIntelligence\animafix\anima-lora-train\.uv-cache'; uv run python -m py_compile AnimaLoraToolkit\anima_train.py AnimaLoraToolkit\trainer\data.py AnimaLoraToolkit\trainer\objective.py AnimaLoraToolkit\models\anima_modeling_core.py
```

- [ ] **Step 4: Review diff**

Run:

```powershell
git diff --stat
git diff -- AnimaLoraToolkit\trainer\data.py AnimaLoraToolkit\trainer\objective.py AnimaLoraToolkit\models\anima_modeling_core.py AnimaLoraToolkit\anima_train.py
```

Confirm no unrelated changes were reverted.
