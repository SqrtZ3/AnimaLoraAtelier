# Token-Bucket 模式：补全 cache_latents / aux loss / module_dropout 支持

**日期:** 2026-06-07
**范围:** 仅针对 `token_bucket` 的 fit_packed 路径（用户确认的「满覆盖」配置：每张图已被精确重采样到桶尺寸）。**不**为一般变长带 padding 的 fit 路径做 cache/aux。

## 背景：三项限制的真实根因

| 功能 | 拦截点 | 根因 |
|---|---|---|
| `cache_latents` | `anima_train.py` `fit_packed and use_cached → raise` | 缓存 `.npz` 只存 `latent`，不存 `pixel_mask`；packed 步需要 mask 构 `latent_mask`。 |
| aux loss | `anima_train.py` `fit_packed and aux.any_enabled → raise` | aux 在网格 x0 `[B,C,H,W]` 上算；packed 路径只产 token 预测。 |
| `module_dropout` | `objective.validate_compile_requirements` | 仅 `torch_compile=true` 时：`torch.rand().item()` 数据依赖分支逐 block 打断编译图。 |

## 关键前提：按精确网格分批

`unpatchify_tokens` 与 `collate_fn_cached` 都要求 batch 内网格一致，但 `FitTokenBatchSampler` 按 token 数分组，会混不同网格（1008×1024 与 1344×768 同为 4032 token）。

**方案：`token_bucket=true` 时，fit batch 按精确 `(token_h, token_w)` 分组。** 每 batch = 单一网格 → mask 恒全 1 → `_build_packed_masks` 返回 `(None,None)` → SDPA 快路径（对 compile 最友好）。加 fail-fast 校验：每图 token 数须落在配置的桶集合内。

## 功能 1：cache_latents

- `CachedLatentDataset` 在 `.npz` 额外存 latent 分辨率的 `latent_mask`（`uint8 [1,h,w]`）+ `token_count`。存真实 mask（不假设全 1）以兼容 alpha 蒙版。
- 新增 `collate_fn_cached_fit`：堆叠单一网格 latent + latent_mask。
- 训练步：`fit_packed + use_cached` 下直接取 `latents`/`latent_mask`（跳过 VAE），后续 `patchify_latents_to_tokens` 路径不变。
- 解除 `fit_packed and use_cached` 拦截，条件为 `token_bucket`。

## 功能 2：aux loss

- packed 前向后，`pred` 是 patch-token；单一网格下 `model.unpatchify_tokens(pred, fit_size)` → 网格 `[B,16,1,H,W]`。`noisy`/`latents` 本就是网格。
- 把 `(noisy_grid, t, pred_grid)` 喂给现有 `recover_x0_from_velocity` + spectral/perceptual，aux 数学不改。
- 解除 `fit_packed and aux.any_enabled` 拦截，条件为 `token_bucket`。

## 功能 3：module_dropout（compile 下可用）

- RNG 外提：每层加标量 `_md_keep`；`injector.roll_module_dropout()`（每步 forward 前调，与 `set_current_t` 同生命周期）抽取，`clear_module_dropout()` 重置。
- forward 用「输出 × `_md_keep`」替代 `torch.rand().item()` 早返回分支。无 `.item()`、图内无 RNG → compile 干净；step 内固定 → grad-checkpoint recompute 一致；eager 等价。
- 覆盖 `LoRALayer` / `LoKrLayer` / `LoRALinear`(DoRA) 三处。
- 从 `validate_compile_requirements` 去掉 `module_dropout` 一条。

## 测试（CPU, backend="eager"）

- 精确网格采样器产出单一网格 batch。
- cached-fit 往返 latent+mask。
- `unpatchify_tokens(pred)` 与网格前向参考一致。
- `_md_keep` 图安全：compile 开/关在 p=0 数值等价、p>0 统计等价。
