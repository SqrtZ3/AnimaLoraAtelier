r"""度量整图 encode（feat_cache 跳过 + attn query 分块）vs 分块 encode 的工具。

回答三件事（cache_latents.py 显存口径变更的实证）：

  1. bit 等价性：T=1 跳过 feat_cache / attn 分块开启，输出是否逐 bit 不变。
  2. 显存与速度：同一张图，旧配置（feat_cache 常开 + attn 整块）分块 encode vs
     新配置整图 encode 的峰值显存和耗时。
  3. 接缝误差：整图 latent 与分块 latent 的差，按"到最近拼缝的距离"分桶统计
     --误差应随距离衰减，衰减宽度即 halo 宽度（mid block 全局注意力下不为零）。

用法：
    <torch-python> vae_tiled_verify.py --image <png> --vae <qwen_image_vae.safetensors> \
        [--attn-chunk 2048] [--tile-px 640] [--tile-overlap 128] [--runs 3]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def _timed(fn, runs):
    """跑 runs 次取最快，同时记录峰值显存（GB，allocated）。"""
    import torch
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    best, out = float("inf"), None
    for _ in range(runs):
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    peak = torch.cuda.max_memory_allocated() / 2**30
    return out, best, peak


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--attn-chunk", type=int, default=2048)
    ap.add_argument("--tile-px", type=int, default=640)
    ap.add_argument("--tile-overlap", type=int, default=128)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--cudnn-benchmark", action="store_true",
                    help="开 torch.backends.cudnn.benchmark（按形状选 conv 算法，"
                         "可能避开 conv3d 的巨型 im2col 工作区）")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import torch
    from PIL import Image
    from cache_latents import _prep_pixels
    from trainer.data import (plan_native_fit_image, tiled_vae_encode,
                              _tile_starts)
    from trainer.models import load_vae, find_diffusion_pipe_root

    dev = torch.device(a.device)
    if a.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    vae = load_vae(a.vae, dev, torch.bfloat16, find_diffusion_pipe_root())
    m, scale = vae.model, vae.scale
    wan_vae = sys.modules["wan_vae"]

    img = Image.open(a.image).convert("RGB")
    plan = plan_native_fit_image(img.width, img.height,
                                 max_tokens=204800, align_mode="floor")
    x = _prep_pixels(img, plan, was_resized=False) \
        .unsqueeze(0).unsqueeze(2).to(dev, torch.bfloat16)   # [1,C,1,H,W]
    H, W = x.shape[-2:]
    print(f"{Path(a.image).name}: {img.width}x{img.height} -> floor16 {W}x{H}"
          f" ({H * W / 1e6:.2f}Mpx)，tile {a.tile_px}/ov {a.tile_overlap}")

    def encode_old(t):
        """复刻旧配置（feat_cache 常开）的 T=1 encode，供对照计时。"""
        m.clear_cache()
        out = m.encoder(t[:, :, :1], feat_cache=m._enc_feat_map, feat_idx=[0])
        mu, _ = m.conv1(out).chunk(2, dim=1)
        mu = (mu - scale[0].view(1, m.z_dim, 1, 1, 1)) \
            * scale[1].view(1, m.z_dim, 1, 1, 1)
        m.clear_cache()
        return mu

    # ── 1. bit 等价性：feat_cache ────────────────────────────────────────────
    crop = x[..., :min(640, H), :min(640, W)]
    with torch.no_grad():
        m.clear_cache()
        o_old = m.encoder(crop, feat_cache=m._enc_feat_map, feat_idx=[0])
        m.clear_cache()
        o_new = m.encoder(crop, feat_cache=None, feat_idx=[0])
    eq = torch.equal(o_old, o_new)
    md = (o_old.float() - o_new.float()).abs().max().item()
    print(f"[feat_cache] T=1 跳过缓存: bitwise_equal={eq}  max_diff={md:.3e}")

    # ── 2. bit 等价性：attn query 分块 ───────────────────────────────────────
    probe = x[..., :min(640, H), :min(640, W)]     # N=6400 token，整块也放得下
    with torch.no_grad():
        l0 = m.encode(probe, scale)                  # chunk 关（当前默认）
        wan_vae.set_vae_attn_chunk_tokens(a.attn_chunk)
        l1 = m.encode(probe, scale)
    eq = torch.equal(l0, l1)
    md = (l0.float() - l1.float()).abs().max().item()
    print(f"[attn-chunk] chunk={a.attn_chunk}: bitwise_equal={eq}  max_diff={md:.3e}")

    # ── 3. 旧配置：整图尝试（预期 OOM）与分块计时 ───────────────────────────
    try:
        with torch.no_grad():
            _, t_old_whole, p_old_whole = _timed(
                lambda: encode_old(x), a.runs)
        print(f"[旧配置|整图] {t_old_whole:.2f}s  peak {p_old_whole:.2f}GB（没 OOM！）")
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print("[旧配置|整图] OOM 复现（feat_cache 常开 + attn 整块）")
    with torch.no_grad():
        lat_old_tiled, t_old, p_old = _timed(
            lambda: tiled_vae_encode(lambda t: encode_old(t), x,
                                     a.tile_px, a.tile_overlap), a.runs)
    print(f"[旧配置|分块] {t_old:.2f}s  peak {p_old:.2f}GB"
          f"  ({len(_tile_starts(H, a.tile_px, a.tile_px - a.tile_overlap))}"
          f"x{len(_tile_starts(W, a.tile_px, a.tile_px - a.tile_overlap))} 块)")
    torch.cuda.empty_cache()

    # ── 4. 新配置：整图直编（feat_cache 跳过 + attn 分块，均已开启）─────────
    lat_whole = None
    try:
        with torch.no_grad():
            lat_whole, t_new, p_new = _timed(lambda: m.encode(x, scale), a.runs)
        print(f"[新配置|整图] {t_new:.2f}s  peak {p_new:.2f}GB"
              f"  （提速 {t_old / t_new:.1f}x，显存 {p_old / max(p_new, 1e-9):.1f}x↓）")
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print("[新配置|整图] 仍 OOM：conv3d 的 im2col 工作区 ∝ 像素数"
              "（96ch×27×px×2B），8GB 卡装不下 2Mpx 整图")
    torch.cuda.empty_cache()

    # ── 5. 接缝误差：整图 vs 分块（同新配置编码）───────────────────────────
    with torch.no_grad():
        lat_tiled, t_nt, p_nt = _timed(
            lambda: tiled_vae_encode(lambda t: m.encode(t, scale), x,
                                     a.tile_px, a.tile_overlap), 1)
    print(f"[新配置|分块 {a.tile_px}/{a.tile_overlap}] {t_nt:.2f}s  peak {p_nt:.2f}GB")
    if lat_whole is not None:
        d = (lat_whole.float() - lat_tiled.float()).abs()      # [1,C,1,h,w]
        print(f"[接缝误差] 整图 vs 分块: mean={d.mean():.4f}  max={d.max():.4f}")
        ys = _tile_starts(H, a.tile_px, a.tile_px - a.tile_overlap)
        xs = _tile_starts(W, a.tile_px, a.tile_px - a.tile_overlap)
        per_cell = np.abs(d[0, :, 0].float().cpu().numpy()).max(0)   # [h,w] 每格最大通道差
        row_cov = np.zeros(per_cell.shape[0], int)
        for y0 in ys:
            row_cov[y0 // 8:(y0 + a.tile_px) // 8] += 1
        col_cov = np.zeros(per_cell.shape[1], int)
        for x0 in xs:
            col_cov[x0 // 8:(x0 + a.tile_px) // 8] += 1

        def axis_dist(cov):
            idx = np.flatnonzero(cov >= 2)
            if not len(idx):
                return np.full(len(cov), 10**9)
            return np.abs(np.arange(len(cov))[:, None] - idx[None, :]).min(1)

        dist = np.minimum(axis_dist(row_cov)[:, None], axis_dist(col_cov)[None, :])
        print("[halo 剖面] 距最近拼缝(latent格) | 格数 | 平均|差| | 最大|差|")
        for lo, hi in [(0, 0), (1, 8), (9, 16), (17, 32), (33, 64), (65, 10**9)]:
            sel = (dist >= lo) & (dist <= hi)
            if sel.any():
                print(f"  {lo}-{(hi if hi < 10**9 else 'inf'):>4} | {int(sel.sum()):6d}"
                      f" | {per_cell[sel].mean():.5f} | {per_cell[sel].max():.5f}")

    # ── 6. 与磁盘上现有 npz（旧管线产物）对比 ───────────────────────────────
    npz = Path(a.image).with_suffix(".npz")
    if npz.exists():
        with np.load(npz) as z:
            disk = torch.from_numpy(z["latent"]).to(dev).view(torch.bfloat16)
        ref = lat_tiled[0].to(torch.bfloat16)
        print(f"[磁盘npz] vs 新分块: bitwise_equal={torch.equal(disk, ref)}"
              f"  max_diff={(disk.float() - ref.float()).abs().max():.3e}")
        if lat_whole is not None:
            ref_w = lat_whole[0].to(torch.bfloat16)
            print(f"[磁盘npz] vs 新整图: max_diff={(disk.float() - ref_w.float()).abs().max():.4f}"
                  f"  mean={(disk.float() - ref_w.float()).abs().mean():.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
