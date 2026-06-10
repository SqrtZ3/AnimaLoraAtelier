"""LoRA-One 式谱对齐初始化，移植到 LoKr（KPSVD 版）。

LoRA-One (arXiv:2502.01235, ICML 2025 Oral)：训练前用一次（小批累积的）全参梯度
G 的 top-r 奇异子空间初始化适配器，使 t=0 即对齐"正确"更新子空间，显著加速
前几百步收敛——正中小数据 LoRA 微调的预算痛点。

推广到 LoKr（SoKA, arXiv:2506.15251 证明 KPSVD 初始化在 Kronecker 结构上成立）：
  1. Van Loan–Pitsianis 重排：G(out,in) → R(G) ∈ (f², od·id)，
     使 ||G - kron(W1, W2)||_F = ||R(G) - vec(W1)vec(W2)ᵀ||_F
  2. R(G) 的 rank-1 SVD → w1 (f,f) 与 W2 (od,id)
  3. W2 的 top-r SVD → w2_a (od,r), w2_b (r,id)
  4. 整体缩放到 ΔW_init = -scale_rel·||W0||_F · Ĝ/||Ĝ||_F（负号 = 梯度下降方向）
  5. DoRA 幅度重算：m ← rownorm(W0 + ΔW_init)，保证初始有效权重 = W0 + ΔW_init
     （不重算会幅度失配，起点输出漂移）。

注意：
  - 初始 ΔW ≠ 0 是有意的（等价于在 top-r 子空间内做一次大步长首步）；导出的
    ckpt 自然包含它，推理侧无需任何特殊处理。
  - scale_rel 是唯一需要标定的超参：ΔW_init 的 Frobenius 范数相对 ||W0||_F 的
    比例。参考：训练完成的画风 LoKr ΔW 范数通常在 ||W0|| 的 1~5% 量级，默认
    0.01 = 保守起步。
  - 仅支持 use_lokr 且非 tlora 的模块；其余跳过（计入 skipped）。
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def kpsvd_lokr_factors(G: torch.Tensor, factor: int, rank: int):
    """对 G (out,in) 做 KPSVD，返回 (w1, w2_a, w2_b)，使
    kron(w1, w2_a @ w2_b) ≈ G 的 rank-(1×r) Kronecker 近似。

    要求 out/in 均被 factor 整除（与 LoKrLayer 同约束）。
    若 min(od,id) < rank，w2_a/w2_b 的多余 rank 列/行补零。
    """
    out_f, in_f = G.shape
    if out_f % factor or in_f % factor:
        raise ValueError(f"shape {tuple(G.shape)} not divisible by factor={factor}")
    od, idim = out_f // factor, in_f // factor

    # Van Loan–Pitsianis 重排：kron(A,B)[i1*od+i2, j1*id+j2] = A[i1,j1]·B[i2,j2]
    R = G.reshape(factor, od, factor, idim).permute(0, 2, 1, 3).reshape(factor * factor, od * idim)
    U, S, Vh = torch.linalg.svd(R, full_matrices=False)
    s1 = S[0].clamp_min(1e-20)
    w1 = (U[:, 0] * s1.sqrt()).reshape(factor, factor)
    W2 = (Vh[0, :] * s1.sqrt()).reshape(od, idim)

    U2, S2, V2h = torch.linalg.svd(W2, full_matrices=False)
    r_eff = min(rank, S2.shape[0])
    sq = S2[:r_eff].clamp_min(0).sqrt()
    w2_a = G.new_zeros(od, rank)
    w2_b = G.new_zeros(rank, idim)
    w2_a[:, :r_eff] = U2[:, :r_eff] * sq.unsqueeze(0)
    w2_b[:r_eff, :] = sq.unsqueeze(1) * V2h[:r_eff, :]
    return w1, w2_a, w2_b


@torch.no_grad()
def lora_one_kpsvd_init(injector, grads: dict, scale_rel: float = 0.01) -> str:
    """对 injector 中每个 LoKr 模块用累积梯度做谱对齐初始化。

    grads: {module_name: fp32 grad tensor (out,in)}（已对 batch 数取均值）。
    返回摘要字符串（applied/skipped/平均能量捕获率）。
    """
    applied = 0
    skipped = []
    capture_sum = 0.0
    for name, lora in injector.injected.items():
        G = grads.get(name)
        ad = lora.adapter
        if (G is None or not getattr(lora, "use_lokr", False)
                or getattr(lora, "use_tlora", False)):
            skipped.append(name)
            continue
        G = G.float()
        gn = G.norm()
        if not torch.isfinite(gn) or gn < 1e-12:
            skipped.append(name)
            continue

        w1, w2_a, w2_b = kpsvd_lokr_factors(G, int(ad.factor), int(ad.rank))

        # 当前近似的 ΔW（含 layer scaling），用于范数标定与能量捕获统计
        approx = torch.kron(w1, w2_a @ w2_b)
        an = approx.norm().clamp_min(1e-12)
        capture_sum += float(an / gn)  # rank-(1×r) Kronecker 近似捕获的梯度能量比

        W0 = lora.original.weight.detach().float()
        target_norm = float(scale_rel) * float(W0.norm())
        # ΔW_layer = scaling·kron(w1, w2a@w2b)；把"负号 + 范数标定 ÷ scaling"全部吸收进 w2_a
        coef = -(target_norm / (float(ad.scaling) * float(an)))
        w2_a = w2_a * coef

        ad.lokr_w1.data.copy_(w1.to(dtype=ad.lokr_w1.dtype, device=ad.lokr_w1.device))
        ad.lokr_w2_a.data.copy_(w2_a.to(dtype=ad.lokr_w2_a.dtype, device=ad.lokr_w2_a.device))
        ad.lokr_w2_b.data.copy_(w2_b.to(dtype=ad.lokr_w2_b.dtype, device=ad.lokr_w2_b.device))

        if getattr(lora, "use_dora", False):
            delta = ad.delta_weight(apply_rank_dropout=False).float().to(W0.device)
            m = (W0 + delta).norm(dim=1).clamp(min=1e-6)
            lora.dora_scale.data.copy_(m.to(dtype=lora.dora_scale.dtype,
                                            device=lora.dora_scale.device))
        applied += 1

    cap = capture_sum / max(applied, 1)
    if skipped:
        logger.info("[lora-one] skipped %d modules (non-lokr/tlora/no-grad): %s%s",
                    len(skipped), ", ".join(skipped[:5]), " ..." if len(skipped) > 5 else "")
    return (f"applied={applied} skipped={len(skipped)} "
            f"scale_rel={scale_rel} mean_energy_capture={cap:.3f}")
