# FiT-Style Packed Training Design

## Context

The repository already has multi-resolution ARB buckets. That path still produces a rectangular pixel batch, encodes it to VAE latents, runs the existing Anima transformer on a 5D latent grid, and exports LoRA weights for the current Comfy-compatible inference path.

FiT is more aggressive. It trains diffusion transformers on latent patch sequences with:

- a target maximum token length,
- per-token 2D grid positions,
- padding masks for variable sequence lengths,
- optional resize and crop variants for images over the token budget,
- loss normalization by valid tokens, not padded sequence length.

The project goal is stricter than the public FiT ImageNet preprocessing recipe: preserve source image detail whenever hardware allows it. Resize and crop are acceptable only as explicit fallback modes, not as the default behavior. A large source image should be represented as one continuous token sequence when it fits the configured memory/token budget.

## Goals

- Add an opt-in FiT-style training path for images with broad resolution and aspect-ratio variation.
- Preserve original pixel detail by default: no resize, no crop, and no bucket snapping unless the user explicitly enables a fallback.
- Keep the existing bucket/grid training path as the default.
- Keep exported LoRA/LoKr weights compatible with the current Anima/Comfy inference path.
- Reuse the existing Anima model weights, text conditioning, LoRA injection, timestep sampling, optimizers, checkpointing, and monitor infrastructure.
- Make the new path testable without a real VAE or large model.

## Non-Goals

- Do not replace the Anima architecture with the official FiT architecture.
- Do not require ComfyUI to support FiT-style token inference.
- Do not change LoRA export formats.
- Do not make arbitrary-resolution sampling part of the first implementation slice.
- Do not remove existing ARB buckets.
- Do not silently downsample or crop source images in the FiT path.

## Chosen Approach

Implement a FiT-style packed token path only for training. The standard 5D latent-grid forward remains the inference-compatible path.

At training time, when `fit_packed_training: true` is enabled:

1. The image pipeline loads the original image and only aligns dimensions to the VAE/patch granularity if needed.
2. No image is resized or cropped by default. If a source image exceeds the configured token/memory guard, the trainer fails fast with a clear estimate and remediation options.
3. VAE latents are patchified into token vectors using the model's existing spatial patch size.
4. A large image remains one continuous token sequence with a single coordinate grid. It is not split into disconnected crops.
5. Batches are padded to a shared sequence length and carry `grid`, `mask`, and `size` metadata.
6. The model runs a new packed-token forward path that reuses existing transformer blocks and final layers.
7. Loss is computed only on valid tokens and normalized by valid-token count.

The exported LoRA is still trained on the same underlying modules used by the old grid forward. During Comfy inference, those modules are invoked through the old 5D grid path, so no downstream loader change is required.

## Data Design

Add a small FiT preprocessing unit in `trainer/data.py`.

Configuration:

- `fit_packed_training`: bool, default false.
- `fit_max_tokens`: int, default 65536 for native-first high-memory training. This corresponds to a 4096x4096 image with VAE downsample 8 and patch size 2.
- `fit_warn_tokens`: int, default 16384. Above this, log an O(N^2) attention cost warning before training starts.
- `fit_min_tokens`: int, default 16.
- `fit_patch_size`: int, default 2, must match the model spatial patch size.
- `fit_vae_downsample`: int, default 8.
- `fit_over_budget_strategy`: `fail`, `skip`, `resize`, `crop`, or `random_resize_crop`, default `fail`.
- `fit_align_mode`: `pad`, `floor`, or `ceil`, default `pad`. Padding preserves all source pixels and marks padded tokens invalid where possible.
- `fit_pack_multiple_images`: bool, default false for the first slice.

Native-first image sizing:

- token count is `H * W / (vae_downsample^2 * patch_size^2)`.
- dimensions are aligned to multiples of `vae_downsample * patch_size`, preferably by padding instead of resizing.
- small images are not upscaled by default; they are padded to the required multiple.
- over-budget images are not modified unless `fit_over_budget_strategy` explicitly requests a fallback.
- if fallback is set to resize or crop, the log must identify each modified image and the exact source and target sizes.

For the first implementation slice, each training sample remains one source image. We still pad variable-length token sequences within a batch. Full cross-image packing can be added later using the same mask semantics, but it must not break a single large image into disconnected crops by default.

Concrete examples with `fit_vae_downsample=8` and `fit_patch_size=2`:

- `512x512` -> latent `64x64` -> token grid `32x32 = 1024`.
- `640x384` -> latent `80x48` -> token grid `40x24 = 960`; no bucket snapping.
- `4096x1024` -> latent `512x128` -> token grid `256x64 = 16384`; continuous full-image sequence.
- `4096x4096` -> latent `512x512` -> token grid `256x256 = 65536`; continuous full-image sequence if the token guard permits it.

## Batch Contract

The FiT collate path returns:

- `fit_latents`: `(B, C, T, H, W)` for real-time VAE encoding or cached source latents before patchification.
- `fit_tokens`: `(B, N, P)` after patchification when latents are cached in token form.
- `fit_grid`: `(B, 2, N)`, token row and column indices.
- `fit_mask`: `(B, N)`, 1 for valid tokens and 0 for padding.
- `fit_size`: `(B, 1, 2)`, token-grid height and width for each sample.
- `fit_pixel_mask` or `fit_latent_mask` when padding was needed before VAE encoding, so padded regions can be excluded after patchification.
- `captions` and `images`, unchanged.

The training loop can start with latent-grid batches and patchify on GPU after noise is created. Cached token storage can be added later if disk speed becomes a bottleneck.

For very large images, the sampler should group examples by similar token counts and allow `batch_size=1` without treating that as a degraded mode. Padding a 1024-token image to a 65536-token batch wastes memory, so token-count-aware batching is part of the first implementation slice.

## Model Design

Add helper methods to the Anima model:

- `patchify_latents_to_tokens(latents, padding_mask=None) -> tokens, grid, mask, size`
- `unpatchify_tokens(tokens, size) -> latent_grid`
- `forward_packed_tokens(tokens, timesteps, cross, grid, mask, size) -> tokens`

The packed path reuses:

- `x_embedder.proj[1]` for token projection,
- `t_embedder`,
- `t_embedding_norm`,
- existing transformer blocks,
- existing final layer linear projection.

Block changes are minimal but explicit:

- Existing `Block.forward` remains unchanged for grid tensors.
- Add `Block.forward_tokens` for `(B, N, D)` tensors.
- Self-attention receives `fit_mask` and builds a boolean attention mask so padding tokens do not contribute.
- Cross-attention uses text conditioning as today.
- Final output tokens are multiplied by `fit_mask[..., None]`.

Attention cost guard:

- The current model uses full self-attention. A `4096x4096` image produces 65,536 tokens, so attention cost is orders of magnitude larger than a 512x512 image's 1,024 tokens.
- The implementation must not hide this by downsampling. It should estimate token counts before training and warn or fail according to `fit_max_tokens`.
- If full attention proves too expensive even on 96GB VRAM, a later implementation can add tiled/windowed attention with explicit cross-window stitching, but that is a separate architecture change.

RoPE changes:

- Existing grid RoPE remains unchanged for normal forward.
- Add a grid-based RoPE helper that accepts `fit_grid` and returns per-token embeddings.
- The first implementation may use the same frequencies as existing RoPE, indexed by explicit token coordinates.
- Online/dynamic scaling from `size` can be added in a second pass if tests show the simple grid helper is stable.

## Loss Design

When `fit_packed_training` is disabled, the current loss path is unchanged.

When enabled:

1. Generate noise in latent-grid space.
2. Patchify `noisy` and `target` to tokens.
3. Run `forward_packed_tokens`.
4. Compute per-sample loss over token dimensions.
5. Multiply by `fit_mask`.
6. Normalize by valid token count per sample.
7. Apply existing timestep loss weighting.

Auxiliary spectral/perceptual losses can stay disabled or fall back to latent-grid reconstruction in the first slice. If enabled, predicted tokens can be unpatchified per sample before aux losses run, but this should be gated because variable shapes make it heavier.

## Cache Design

First slice:

- Existing latent cache remains supported for bucket training.
- FiT packed training can run without cache first.
- If `cache_latents` and `fit_packed_training` are both true, the trainer should fail fast with a clear message until token/variable-latent caching is implemented.

Second slice:

- Store FiT latent variants with metadata: `feature`, `grid`, `size`, `source_size`, `variant`, `dtype_kind`.
- Invalidate cache when token budget, patch size, VAE downsample, or variant strategy changes.

## Compatibility

The key compatibility rule is: FiT packed training must train the same module weights that the old grid forward uses.

This allows the exported LoRA to remain a standard Anima LoRA. The model may learn from variable token lengths during training, but Comfy inference will still call the rectangular-grid forward it already supports.

Risk: the token path may exercise attention masking and RoPE indexing differently from the grid path. Tests must confirm that for a single rectangular sample with all mask values valid, packed-token forward and grid forward produce matching shapes and close numeric behavior where possible.

## Testing Strategy

Add focused unit tests before implementation:

- Native FiT sizing does not resize or crop source images under `fit_max_tokens`.
- Alignment pads rather than scales source images by default.
- Over-budget images fail fast by default with source size, token count, and configured guard in the error.
- Explicit resize/crop fallback paths log source and target sizes.
- Collate pads variable-length token samples and creates correct `mask`, `grid`, and `size`.
- Masked token loss ignores padding and normalizes by valid tokens.
- Model token patchify/unpatchify round-trips a latent tensor.
- Packed-token forward returns `(B, N, P)` and zeros padded outputs.
- Default config keeps old path unchanged.

Run at least:

- `uv run python -m unittest AnimaLoraToolkit.tests.test_multires_buckets`
- new FiT data/loss/model unit tests
- `uv run python -m py_compile AnimaLoraToolkit/anima_train.py AnimaLoraToolkit/trainer/data.py AnimaLoraToolkit/trainer/objective.py AnimaLoraToolkit/models/anima_modeling_core.py`

## Rollout

1. Add config flags and tests for sizing/collate/loss.
2. Add native-first FiT data helpers while keeping old ARB behavior unchanged.
3. Add token loss helpers.
4. Add model packed-token methods behind a new forward helper.
5. Add token-count-aware batching and memory/cost warnings.
6. Wire training loop when `fit_packed_training` is true.
7. Add fail-fast guards for unsupported combinations, especially cached latents and aux losses.
8. Document the config and the Comfy inference compatibility boundary.

## Open Decisions

- Whether first implementation should allow `batch_size > 1` with variable token padding, or force `batch_size=1` until packed-token forward is validated.
- Whether auxiliary losses should auto-disable under FiT packed training or reconstruct latent grids per sample.
- Whether to add full cross-image packing immediately after single-image variable token batches work.
- Whether `fit_max_tokens=65536` should be the template default or only a documented high-memory preset.
