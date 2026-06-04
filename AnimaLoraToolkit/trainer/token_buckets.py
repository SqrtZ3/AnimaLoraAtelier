"""Constant / N-token bucket generation for Anima LoRA training.

Anima latent patching is VAE downsample 8 x patch 2 = 16 px per token axis,
so a bucket's token count is ``T = (W // 16) * (H // 16)``. A "token family"
is the set of integer factor pairs ``(wp, hp)`` with ``wp * hp == T``; the
aspect ratios a family can represent are exactly those ``wp:hp`` ratios, so
``T`` should be highly composite (e.g. 4032 = 63*64, 4200 = 60*70) for dense
aspect-ratio coverage. Constraining training to a small set of token counts is
what lets ``torch.compile`` trace a fixed, tiny set of graphs.

Dependency-free (stdlib only) so it can run in the dataset tool and in test
environments without torch.
"""

from typing import List, Sequence, Tuple


def generate_token_buckets(
    token_counts: Sequence[int],
    max_aspect_ratio: float = 2.0,
    patch_pixels: int = 16,
    min_dim_px: int = 256,
    max_dim_px: int = 4096,
) -> List[Tuple[int, int]]:
    """Return sorted (W, H) pixel buckets whose token count
    ``(W // patch_pixels) * (H // patch_pixels)`` is exactly one of
    ``token_counts``.

    For each ``T`` in ``token_counts`` every integer factor pair ``(wp, hp)``
    with ``wp * hp == T`` is enumerated; pairs are kept when both pixel
    dimensions fall in ``[min_dim_px, max_dim_px]`` and the aspect ratio
    ``max(wp/hp, hp/wp)`` does not exceed ``max_aspect_ratio``. Both
    orientations arise naturally from the factor enumeration. The result is
    deterministic, sorted by ``(token_count, W, H)``.
    """
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
    return sorted(
        out,
        key=lambda wh: ((wh[0] // patch_pixels) * (wh[1] // patch_pixels), wh[0], wh[1]),
    )
