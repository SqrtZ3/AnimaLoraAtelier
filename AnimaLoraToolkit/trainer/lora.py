"""LoRA / LoKr / DoRA 适配器与注入器。

包含：
- `LoRALayer`、`LoKrLayer`、`ABBALayer` —— 三种低秩分解适配器（含 rank_dropout / module_dropout；
  ABBA 为 Hadamard 双低秩，arXiv:2505.14238，不支持 rank_dropout）
- `LoRALinear` —— 把原始 `torch.nn.Linear` 包成 adapter + base 的复合层，支持 DoRA
- `LoRAInjector` —— 全模型扫描 + regex 选择 + 模块级 rank / alpha / lr + LoRA+
                    + ComfyUI 兼容的 safetensors 保存/加载（native / diff / merged_model 三种导出格式）

被主训练脚本以及 checkpoint 模块依赖。本模块自身只 import torch / safetensors，不依赖任何
其它 trainer 子模块。
"""

from __future__ import annotations

import collections
import logging
import math
import re

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ★ module_dropout 决策必须用 torch.rand 而非 Python random:
# torch.utils.checkpoint(use_reentrant=False) 在 backward recompute forward 时
# 会自动 fork + restore torch 的 rng state，让 forward 和 recompute 走完全相同的
# 分支；Python random 模块的 state **不被** checkpoint 保护，recompute 时返回不同
# 值会让 forward 选 "return zeros" 而 recompute 选 "正常计算"，autograd 图保存的
# tensor 与 recompute 的对不上 → CheckpointError。
#
# 至于 GPU 同步：torch.rand(1) 默认 device=cpu（且 generator 默认是 cpu 的），不
# 触发 GPU↔CPU 同步；.item() 在 CPU tensor 上是纯内存读取。曾经有一版改成
# random.random() 想"省同步"是误判，且破坏了 checkpoint 的确定性。


# ============================================================================
# T-LoRA helpers
# ============================================================================

def _tlora_rank_mask(rank: int, r_min: int, current_t: torch.Tensor, alpha: float,
                     out_dtype: torch.dtype, out_device: torch.device) -> torch.Tensor:
    """Compute per-sample rank mask for T-LoRA.

    幂律 schedule: r(t) = floor(((1-t)^alpha) * (r - r_min)) + r_min
    论文 (arxiv:2507.05964) 的线性 schedule 是 alpha=1 的特例。
    返回形状 (B, rank) 的 0/1 mask。
    """
    t = current_t.to(device=out_device).clamp(0.0, 1.0).float()
    delta = max(rank - r_min, 0)
    r_t = (((1.0 - t) ** float(alpha)) * delta).floor().long() + int(r_min)
    r_t = r_t.clamp_(int(r_min), int(rank))
    ranks = torch.arange(int(rank), device=out_device)
    mask = (ranks.unsqueeze(0) < r_t.unsqueeze(1)).to(out_dtype)
    return mask


def _ortho_lora_init(in_features: int, out_features: int, rank: int, device=None):
    """Ortho-LoRA SVD init (arxiv:2507.05964 Eq.5).

    取 R ~ N(0, 1/sqrt(rank)) 的 SVD 最后 rank 个分量（sig_type="last"）：
        A_init = V^T[-rank:]                (rank, in_features)
        B_init = U[:, -rank:] * S[-rank:]   (out_features, rank)   # 把 S 折进 B
    返回 (A_init, B_init)；调用方负责拷贝到 nn.Linear 权重并保留 init 副本用于
    forward 时的初始 delta 补偿。

    SVD 在传入的 device 上计算（建议传 GPU），CPU SVD 在 5120×5120 量级要 10+ 秒，
    Wan 14B 整网 240+ 层会累积到一小时。dtype 固定 fp32：bf16 SVD 数值上不稳定。
    """
    R = torch.randn(out_features, in_features, device=device, dtype=torch.float32) / math.sqrt(max(rank, 1))
    U, S, Vh = torch.linalg.svd(R, full_matrices=False)
    # SVD 返回 S 降序；最后 rank 个 = 最小奇异值（论文经验：抗过拟合最好）
    A_init = Vh[-rank:, :].contiguous()                          # (rank, in_features)
    S_init = S[-rank:]                                            # (rank,)
    B_init = (U[:, -rank:] * S_init.unsqueeze(0)).contiguous()    # (out_features, rank)
    return A_init, B_init


def _pissa_init(in_features: int, out_features: int, rank: int,
                base_weight: torch.Tensor, device=None):
    """PiSSA SVD initialization (arxiv:2404.02948, NeurIPS 2024 spotlight).

    取 W 的 SVD 的前 rank 个主成分（最大奇异值方向）初始化 A/B：
        A_init = V[:, :rank]^T               (rank, in_features)
        B_init = U[:, :rank] * S[:rank]      (out_features, rank)   # 把 S 折进 B

    与 Ortho-LoRA 取最小奇异值相反，PiSSA 取最大奇异值（最重要的方向）。
    配合 delta 补偿（forward 时减去 B_init@A_init * scaling），step 0 净 delta=0，
    但 A/B 起点在 W 的主方向子空间内，梯度方向天然对齐最重要的变化方向。

    要求 alpha = rank（scaling = 1），否则 step 0 净 delta ≠ 0。

    使用 torch.svd_lowrank（随机化截断 SVD）替代全量 torch.linalg.svd：
    只需 top-r 奇异向量，无需全量分解。对 6144×36864 的 tproj 层，
    全量 SVD ~30s，截断 SVD (r=24) <0.1s（~300x 加速）。
    """
    W = base_weight.detach().float().to(device)
    # W: (out_features, in_features)，svd_lowrank 要求 2D 输入
    # 返回 U: (out, q), S: (q,), V: (in, q)
    # ★ 注意：svd_lowrank 返回 V（不是 Vh），V 的形状是 (in, q)
    #   全量 svd 返回 Vh: (k, in)，截断版本返回 V: (in, q)
    q = min(rank + 8, min(in_features, out_features))  # 多取几个提高精度
    U, S, V = torch.svd_lowrank(W, q=q, niter=2)
    A_init = V[:, :rank].t().contiguous()                        # (rank, in_features)
    B_init = (U[:, :rank] * S[:rank].unsqueeze(0)).contiguous()  # (out_features, rank)
    return A_init, B_init


def svd_truncate_lora_pair(down: torch.Tensor, up: torch.Tensor, scaling: float,
                           energy: float = 1.0, max_rank: int = 0):
    """按每层能量阈值对标准 LoRA 的 (down, up) 做 SVD 截断压缩。

    输入（标准 LoRA 语义）：
        down = A : (r, in)        # lora_down.weight
        up   = B : (out, r)       # lora_up.weight
        scaling  = alpha / r      # 训练时的 ΔW = scaling · B @ A
    返回 (down', up', keep, dropped)：
        down' : (keep, in)、up' : (out, keep)，满足 up' @ down' ≈ scaling·(B@A) 的
        top-`keep` 秩近似；**scaling 已折进因子**，导出时应写 alpha'=keep（→ scaling'=1）。
        keep    = 保留的秩；dropped = 被丢弃的能量占比（0 表示无损）。

    截断准则与 tools/abba_export_lora.py 一致：保留累计能量 ≥ energy 的最小秩
    （实现 `keep=(ce<energy).sum()+1`，保守多留一个，实际保留能量 ≥ energy）。
    energy=1.0 → keep=r（不截断，数值容差内无损）。max_rank>0 时与能量准则取更紧者。

    ★ 截断口径（与 aclora_restart_matrix 相反，勿混淆）：
    本函数用于**导出期压缩**，口径是"宁可多留、别截过头"，故取 ≥ energy。
    同文件的 aclora_restart_matrix（训练期 RESTART 用）忠实论文 Eq.3 用**严格 `<`**，
    实际保留能量 ≤ p。两者 docstring 措辞相近但语义相反，改其中任一个前请先确认用途，
    不要"统一"成同一个口径。

    用 QR 技巧在 r×r 上做 SVD，**不物化 (out, in) 全矩阵**：
        B = Qb Rb, Aᵀ = Qa Ra → B@A = Qb (Rb Raᵀ) Qaᵀ，只对 (Rb Raᵀ) 这个 r×r SVD。
    对 6144×4096 的 mlp 层，全矩阵 SVD 要秒级且吃显存，这里 r×r（r≤64）近乎免费。
    """
    if not (0.0 < energy <= 1.0):
        raise ValueError(f"energy 必须在 (0,1]，得到 {energy}")
    A = down.detach().float()          # (r, in)
    B = up.detach().float()            # (out, r)
    r = A.shape[0]
    s = float(scaling)
    # QR：B=Qb Rb（Qb:(out,r), Rb:(r,r)），Aᵀ=Qa Ra（Qa:(in,r), Ra:(r,r)）
    Qb, Rb = torch.linalg.qr(B)
    Qa, Ra = torch.linalg.qr(A.t())
    Uc, S, Vhc = torch.linalg.svd(Rb @ Ra.t())   # r×r
    e = S ** 2
    e_tot = e.sum().clamp(min=1e-30)
    ce = torch.cumsum(e, 0) / e_tot
    if energy < 1.0:
        keep = int((ce < energy).sum().item()) + 1
    else:
        keep = int(S.numel())
    if max_rank and max_rank > 0:
        keep = min(keep, int(max_rank))
    keep = max(min(keep, int(S.numel())), 1)
    dropped = float(1.0 - ce[keep - 1].item()) if keep < S.numel() else 0.0
    # 把 scaling 折进因子：up'@down' = Qb Uc[:, :k] (S[:k]·s) Vhc[:k] Qaᵀ = s·(B@A) 截断
    s_sqrt = (S[:keep] * s).clamp(min=0).sqrt()
    down_new = (Vhc[:keep] @ Qa.t()) * s_sqrt.unsqueeze(1)   # (keep, in)
    up_new = (Qb @ Uc[:, :keep]) * s_sqrt.unsqueeze(0)       # (out, keep)
    return down_new, up_new, keep, dropped


def lora_pair_spectrum(down: torch.Tensor, up: torch.Tensor, scaling: float):
    """只算 ΔW 的奇异值谱（不做截断、不物化全矩阵）。

    用于全局预算分配的第一遍扫描：分配器需要先看到所有层的谱，才能决定
    每层该分几个 rank。QR 技巧同 svd_truncate_lora_pair。
    返回 S：(r,) 降序奇异值，已含 scaling。
    """
    A = down.detach().float()
    B = up.detach().float()
    _, Rb = torch.linalg.qr(B)
    _, Ra = torch.linalg.qr(A.t())
    return torch.linalg.svdvals((Rb @ Ra.t()) * float(scaling))


def allocate_ranks_by_budget(spectra: dict, bytes_per_rank: dict,
                             budget_bytes: float, min_rank: int = 1):
    """全局 σ²/字节最优秩分配：给定总字节预算，决定每层保留几个奇异分量。

    机理：把"再给某层加 1 个 rank"看成一次投资——收益是该层下一个奇异值的能量
    σ²，成本是该层每 rank 的字节数 (in+out)·2。贪心地总是先买"每字节能量增益"
    最大的那一个，即为该预算下的能量最优分配（各层收益随 rank 递减，贪心即最优）。

    为什么不用逐层能量阈值：逐层阈值对每层一视同仁，会在"范数极小但谱平"的层
    （如 attn.qkv）上浪费大量字节，而这些层对总增量几乎无贡献。Krea2 c12port
    epoch19 实测同等保留能量下，本方法体积小 1.3–3.2×：
        逐层 energy=0.95 → 85.5MB/97.79%   vs   全局预算 35.5MB → 97.96%
        逐层 energy=0.90 → 64.5MB/97.24%   vs   全局预算 20.3MB → 97.38%

    Args:
        spectra: {层名: 奇异值张量(降序)}
        bytes_per_rank: {层名: 每个 rank 的字节数}
        budget_bytes: 总预算（字节）
        min_rank: 每层至少保留的 rank（默认 1；置 0 允许整层丢弃）
    Returns:
        {层名: keep}
    """
    import heapq
    ranks = {n: 0 for n in spectra}
    used = 0.0
    heap = []
    for n, S in spectra.items():
        if S.numel() > 0:
            heapq.heappush(heap, (-(S[0] ** 2).item() / bytes_per_rank[n], n))
    while heap and used < budget_bytes:
        _gain, n = heapq.heappop(heap)
        S = spectra[n]
        if ranks[n] >= S.numel():
            continue
        ranks[n] += 1
        used += bytes_per_rank[n]
        if ranks[n] < S.numel():
            heapq.heappush(heap, (-(S[ranks[n]] ** 2).item() / bytes_per_rank[n], n))
    if min_rank > 0:
        for n in ranks:
            ranks[n] = max(ranks[n], min(min_rank, int(spectra[n].numel())))
    return ranks


def aclora_restart_matrix(M: torch.Tensor, p: float, generator=None):
    """AC-LoRA RESTART（arXiv:2504.02231 Eq.2/3）对单个矩阵 M（A 或 B）做一次。

    信号/噪声切分（Eq.3）：对 M 做 SVD 得奇异值 S（降序），按累计能量占比切分——
        保留集 S_keep = {i | cumsum_i < p·total}（严格小于，忠实论文 Eq.3 / Algo.1 L12），
        其余奇异分量清零。keep 至少 1。
    RESTART（Eq.2）：M_signal = U·diag(D')·Vᵀ（D' 把噪声奇异值清零重建），
        σ² = Var(M − M_signal)（被丢弃残差的方差），G ~ N(0, σ²) 与 M 同形，
        M' = M_signal + G。**丢弃分量不是永久删除，而是重置成同方差噪声继续训练。**

    返回 (M'（与 M 同 dtype/device）, keep)。不修改输入。
    注意：p 越大 → 保留越多 → 加噪越少；p→1 时近乎 no-op（信号≈全部，σ²≈0）。

    ★ 截断口径（与 svd_truncate_lora_pair 相反，勿混淆）：
    本函数用**严格 `<`**，因此实际保留能量 **≤ p**（总能重置 ≥1 个最小分量）。
    这是论文原意——RESTART 的目的就是主动重置噪声分量。而同文件的
    svd_truncate_lora_pair（导出压缩用）用 `(ce < energy).sum()+1`，实际保留
    **≥ energy**，是为"导出宁可多留、别截过头"的保守口径。两者 docstring 措辞
    相近但语义相反，改其中任一个前请先确认用途，不要"统一"。
    """
    orig_dtype, orig_device = M.dtype, M.device
    Mf = M.detach().float()
    U, S, Vh = torch.linalg.svd(Mf, full_matrices=False)   # U:(m,k) S:(k,) Vh:(k,n)
    e = S ** 2
    tot = e.sum().clamp(min=1e-30)
    cum = torch.cumsum(e, 0)
    keep = int((cum < float(p) * tot).sum().item())
    keep = max(min(keep, int(S.numel())), 1)
    Dp = S.clone()
    if keep < S.numel():
        Dp[keep:] = 0.0
    M_signal = (U * Dp.unsqueeze(0)) @ Vh                   # U·diag(D')·Vᵀ
    resid = Mf - M_signal
    sigma = resid.std(unbiased=False)                      # sqrt(Var(残差))
    if generator is not None:
        G = torch.randn(Mf.shape, device=Mf.device, dtype=Mf.dtype, generator=generator)
    else:
        G = torch.randn_like(Mf)
    M_new = M_signal + G * sigma
    return M_new.to(dtype=orig_dtype, device=orig_device), keep


class LoRALayer(torch.nn.Module):
    """标准 LoRA 层（含 rank_dropout / module_dropout / 可选 T-LoRA）

    T-LoRA (arxiv:2507.05964)：
      - tlora_enabled=True 时，forward 内按 self.current_t（形状 (B,) 的 timestep
        tensor）应用 rank mask M_t，掩掉高 rank 通道；t→0 时全 rank，t→1 时只剩
        r_min 个。schedule 为幂律 r(t) = floor(((1-t)^alpha) * (r - r_min)) + r_min
        （alpha=1 即论文的线性 schedule）。
      - tlora_init="ortho" 时用 SVD 最后 rank 个分量做 Ortho-LoRA 初始化，并把
        初始 A/B 存为 buffer 用于 forward 时的初始 delta 补偿（论文 Eq.5），保证
        训练 step 0 时净 delta ≈ 0。
      - 用方法 set_current_t(t) 注入当前 timestep；调用方（train loop / sampling.py）
        在 model.forward 前调，forward 后 reset 成 None。
    """
    def __init__(self, in_features, out_features, rank=4, alpha=1.0, dropout=0.0,
                 rank_dropout=0.0, module_dropout=0.0,
                 tlora_enabled=False, tlora_rmin_ratio=0.5, tlora_alpha=1.0,
                 tlora_init="default", device=None,
                 lora_init="default", base_weight=None):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        # 在目标 device 上直接构造，避免后续 cross-device copy_ 的隐式同步开销
        linear_kwargs = {"device": device} if device is not None else {}
        self.lora_down = torch.nn.Linear(in_features, rank, bias=False, **linear_kwargs)
        self.lora_up = torch.nn.Linear(rank, out_features, bias=False, **linear_kwargs)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        self.rank_dropout = float(rank_dropout or 0.0)
        self.module_dropout = float(module_dropout or 0.0)

        # T-LoRA state
        self.tlora_enabled = bool(tlora_enabled)
        self.tlora_alpha = float(tlora_alpha)
        self.tlora_init = (tlora_init or "default").lower()
        if self.tlora_enabled:
            self.r_min = max(int(round(rank * float(tlora_rmin_ratio))), 1)
        else:
            self.r_min = rank
        self.current_t: torch.Tensor | None = None
        # Module-dropout 两条路径，由 _md_compile_safe 选择（setup 时一次性设定）：
        #  - eager（torch_compile 关，默认 _md_compile_safe=False）：forward 内直接
        #    torch.rand().item() 懒抽签、命中即早返回，与最初实现逐字节一致、零额外每步开销；
        #    此模式下 roll/clear 根本不会被调用（injector 层短路）。
        #  - compile-safe（torch_compile 开）：keep 标量由 roll_module_dropout() 在 forward 外
        #    预抽，forward 只读它做乘法 —— 把数据依赖的 RNG 分支移出编译区域，避免逐 block 打断图。
        self._md_keep: torch.Tensor | None = None
        self._md_compile_safe: bool = False

        # 初始化 + 可选 Ortho-LoRA / PiSSA + 初始 delta 补偿 buffer
        #
        # ★ 退化层守卫：SVD 类补偿式 init（ortho/PiSSA）最多只能取出
        #   min(in, out) 个奇异分量。min(in, out) < rank 的层（如 krea2
        #   txtfusion.projector = Linear(12→1)）上 `[:rank]`/`[-rank:]` 切片会
        #   静默切少，后续 copy_ 的隐式广播把形状错误掩盖到前向才炸
        #   （物证：F.linear 报 (M,32)×(1,1)）。这类层回退 default init——
        #   lora_up 零 init 天然满足 step-0 净 delta=0，无需补偿。
        svd_init_ok = min(in_features, out_features) >= rank
        wants_svd_init = (
            (self.tlora_enabled and self.tlora_init == "ortho")
            or (lora_init == "pissa" and base_weight is not None)
        )
        if wants_svd_init and not svd_init_ok:
            logger.warning(
                "[lora-init] 层 (%d→%d) 的 min(in,out)=%d < rank=%d，SVD 补偿式 "
                "init（%s）在此退化 —— 该层回退 default init（step-0 净 delta 仍=0）。",
                in_features, out_features, min(in_features, out_features), rank,
                "ortho" if self.tlora_enabled else "pissa")
        if self.tlora_enabled and self.tlora_init == "ortho" and svd_init_ok:
            A_init, B_init = _ortho_lora_init(in_features, out_features, rank, device=device)
            assert A_init.shape == (rank, in_features) and B_init.shape == (out_features, rank), (
                f"ortho init 形状错误: A{tuple(A_init.shape)} B{tuple(B_init.shape)}，"
                f"期望 A({rank},{in_features}) B({out_features},{rank})")
            with torch.no_grad():
                self.lora_down.weight.copy_(A_init)
                self.lora_up.weight.copy_(B_init)
            # 初始权重副本（不参与梯度），forward 时用同样的 mask 重新计算 init
            # 贡献并减去，让训练初始 step 净 delta ≈ 0（论文 Eq.5 的等价 2-matrix 版）
            self.register_buffer("lora_down_init", A_init.clone(), persistent=False)
            self.register_buffer("lora_up_init", B_init.clone(), persistent=False)
        elif lora_init == "pissa" and base_weight is not None and svd_init_ok:
            # PiSSA (arxiv:2404.02948): SVD of base weight, top-r principal components
            # A/B 起点在 W 主方向子空间内，配合 delta 补偿保证 step 0 净 delta=0
            if abs(self.scaling - 1.0) > 1e-6:
                logger.warning(
                    "[PiSSA] alpha != rank (scaling=%.4f); step 0 net delta != 0. "
                    "Set alpha=rank for correct PiSSA behavior.", self.scaling)
            A_init, B_init = _pissa_init(in_features, out_features, rank,
                                         base_weight=base_weight, device=device)
            assert A_init.shape == (rank, in_features) and B_init.shape == (out_features, rank), (
                f"PiSSA init 形状错误: A{tuple(A_init.shape)} B{tuple(B_init.shape)}，"
                f"期望 A({rank},{in_features}) B({out_features},{rank})")
            with torch.no_grad():
                self.lora_down.weight.copy_(A_init.to(device=device))
                self.lora_up.weight.copy_(B_init.to(device=device))
            self.register_buffer("lora_down_init", A_init.clone(), persistent=False)
            self.register_buffer("lora_up_init", B_init.clone(), persistent=False)
        else:
            torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
            torch.nn.init.zeros_(self.lora_up.weight)
            self.lora_down_init = None
            self.lora_up_init = None

    def set_current_t(self, t):
        """由训练循环 / 采样循环在 model.forward 前调用。t 可为 None（重置）。"""
        self.current_t = t

    def set_module_dropout_compile_safe(self, flag):
        """setup 时设定一次：True=走 compile-safe（forward 外预抽 keep 标量）；
        False=走 eager 懒抽签（forward 内 rand 早返回，零额外每步开销）。"""
        self._md_compile_safe = bool(flag)
        if not self._md_compile_safe:
            self._md_keep = None

    def roll_module_dropout(self):
        """compile-safe 模式下每 step 在 forward 前预抽 keep 标量；eager 模式不会被调用。"""
        if self._md_compile_safe and self.training and self.module_dropout > 0:
            dev = self.lora_up.weight.device
            self._md_keep = (torch.rand((), device=dev) >= self.module_dropout).float()
        else:
            self._md_keep = None

    def clear_module_dropout(self):
        """forward 完成后重置，避免 keep=0 残留进后续采样/eval 前向。"""
        self._md_keep = None

    def _apply_tlora_mask(self, h):
        """对 lora_down 的输出 h: (..., rank) 应用 per-sample T-LoRA mask。"""
        if not (self.tlora_enabled and self.current_t is not None):
            return h, None
        t = self.current_t
        # 输入可能是 (B, *, rank)；t 形状 (B,)
        if t.ndim == 0:
            t = t.view(1)
        mask = _tlora_rank_mask(self.rank, self.r_min, t, self.tlora_alpha,
                                h.dtype, h.device)
        # 广播 mask: (B, rank) → (B, 1, ..., 1, rank)
        if h.shape[0] != mask.shape[0]:
            # 单样本广播：t 形状 (1,) 而 h 第 0 维是别的（极少见，保守处理）
            if mask.shape[0] == 1:
                mask_view = mask.view(1, *([1] * (h.ndim - 2)), self.rank)
                return h * mask_view, mask
            raise RuntimeError(
                f"T-LoRA mask batch mismatch: h.shape[0]={h.shape[0]} vs current_t shape={tuple(t.shape)}"
            )
        mask_view = mask.view(mask.shape[0], *([1] * (h.ndim - 2)), self.rank)
        return h * mask_view, mask

    def forward(self, x):
        # Module dropout 两路（见 __init__ 注释）：
        #  - eager：原版懒抽签早返回，零额外开销（torch.rand().item() 被 grad-checkpoint 的
        #    RNG fork/restore 保护，recompute 走同一分支）；
        #  - compile-safe：keep 标量在 forward 外预抽，仅末尾乘一次。compile 时
        #    `not self._md_compile_safe` 是常量 False → 整条 and 短路 → torch.rand 不进图。
        if (self.training and self.module_dropout > 0 and not self._md_compile_safe
                and torch.rand(1).item() < self.module_dropout):
            return torch.zeros(*x.shape[:-1], self.lora_up.out_features,
                               device=x.device, dtype=x.dtype)
        x_drop = self.dropout(x)
        h = self.lora_down(x_drop)

        # T-LoRA: rank mask
        h, mask = self._apply_tlora_mask(h)

        # 是否启用初始 delta 补偿（Ortho-LoRA 或 PiSSA）。先确定，因为 rank_dropout
        # 需要同时作用到两支。
        has_init = (
            self.lora_down_init is not None and self.lora_up_init is not None
        )
        use_ortho_comp = has_init and self.tlora_enabled and self.current_t is not None
        # PiSSA init compensation: no T-LoRA, just subtract init delta for step-0 = 0
        use_init_comp = use_ortho_comp or (has_init and not self.tlora_enabled)

        # 提前算出 init 分支的 h_init（应用同一个 T-LoRA mask），稍后与 train 分支共用 rd_mask
        h_init = None
        if use_init_comp:
            h_init = F.linear(x_drop, self.lora_down_init)
            if mask is not None:
                h_init = h_init * mask.view(mask.shape[0], *([1] * (h_init.ndim - 2)), self.rank)

        # Rank dropout: 随机置零 rank 中的某些通道（训练时，inverted dropout）
        # ★ rd_mask 必须显式落在 h.dtype 上：torch.full 默认 fp32，会让 h * rd_mask
        # 升精度（bf16 → fp32），后续 self.lora_up(h) 时 weight 是 bf16 即 dtype mismatch
        # ★★ 同一个 rd_mask 必须同时应用到 train 分支 h 和 init 分支 h_init。
        # 否则 step 0 时（lora_up=B_init, lora_down=A_init）两分支应当严格相消，但因
        # 一边 dropout 了、另一边没 dropout，会留下一个高方差 0 均值的随机 delta 叠加
        # 在 base 权重上，破坏 Ortho-LoRA 论文 Eq.5 的"step 0 净 delta=0"保证。
        # 观察上的表现：训练从一开始就输出色块噪声、Prodigy d 估计被噪声主导疯狂上调、
        # loss 反复冲到 1.5+ 不收敛（issue: rank_dropout × T-LoRA ortho init 互动）
        if self.training and self.rank_dropout > 0:
            rd_mask = torch.bernoulli(
                torch.full((self.rank,), 1.0 - self.rank_dropout, device=h.device)
            ).to(dtype=h.dtype)
            inv_keep = 1.0 / (1.0 - self.rank_dropout + 1e-6)
            h = h * rd_mask * inv_keep
            if h_init is not None:
                h_init = h_init * rd_mask.to(h_init.dtype) * inv_keep

        # ★ 防御性 cast：极端情况下（用户在 forward 前手工改了 dtype，或某些 hook 上溯精度）
        # 也能保证 lora_up 不会因 dtype mismatch 崩；正常路径下这是 no-op
        if h.dtype != self.lora_up.weight.dtype:
            h = h.to(dtype=self.lora_up.weight.dtype)
        out = self.lora_up(h) * self.scaling

        # Ortho-LoRA 初始 delta 补偿：减去 (B_init · M_t · A_init · rd_mask) · x · scaling
        # 与上面的 (B · M_t · A · rd_mask) · x · scaling 相抵消，让训练 step 0 时净 delta = 0
        if h_init is not None:
            if h_init.dtype != self.lora_up_init.dtype:
                h_init = h_init.to(dtype=self.lora_up_init.dtype)
            out = out - F.linear(h_init, self.lora_up_init) * self.scaling

        if self._md_keep is not None:
            out = out * self._md_keep.to(out.dtype)
        return out

    # ── DoRA 支持：标准 LoRA 路径的 _compute / merged_row_norms ──────────
    # 这些方法让 lora_type="lora" + lora_variant="dora" 可用（此前仅 LoKr 支持 DoRA）。
    # LoKr 版通过 Kronecker 结构避免全矩阵物化；标准 LoRA 版更简单，直接低秩收缩。

    def _compute(self, x):
        """LoRA delta output without base -- for DoRA forward path.

        等价于 self.forward(x)（adapter forward 只返回 delta，不含 base）。
        """
        return self.forward(x)

    def merged_row_norms(self, base_weight, base_row_sq=None, fast=False):
        """||W + ΔW|| per row for DoRA, without materializing full (out, in) matrix.

        ΔW = (B @ A - B_init @ A_init) * scaling  (含 PiSSA/ortho init 补偿)
        ||W + ΔW||² = ||W||² + 2·⟨W, ΔW⟩ + ||ΔW||²  per row

        fast=True：⟨W,·⟩ 收缩直接在 W 原 dtype（bf16 tensor core，fp32 累加）上做，
        不物化全量 fp32 W 副本 —— 关键的是 autograd 图里保存的是 base 权重的
        detached 引用而非每层一份 fp32 拷贝（12B/264 targets 下这是 GB 级差异）。
        范数相对误差 ~1e-3。语义与 LoKrLayer.merged_row_norms 的 fast 一致。
        此前该参数在标准 LoRA 路径上被静默忽略（只有 LoKr 实现了）。
        """
        Wd = base_weight.detach()
        if base_row_sq is None:
            if fast:
                base_row_sq = (Wd * Wd).sum(dim=1, dtype=torch.float32)
            else:
                base_row_sq = Wd.float().pow(2).sum(dim=1)
        A = self.lora_down.weight.float()  # (rank, in)
        B = self.lora_up.weight.float()    # (out, rank)
        s = self.scaling
        W = None if fast else Wd.float()
        # ⟨W, s·B@A⟩ per row = s * (B * (W @ A.T)).sum(dim=1)
        if fast:
            WA = torch.matmul(Wd, A.t().to(Wd.dtype)).float()  # (out, rank)
        else:
            WA = torch.matmul(W, A.t())  # (out, rank)
        dot = s * (B * WA).sum(dim=1)  # (out,)
        # ||s·B@A||² per row = s² * Σ_{r,r'} B[o,r]·B[o,r']·⟨A[r],A[r']⟩
        #                    = s² * (B @ (A@Aᵀ) * B).sum(dim=1)
        # ★ 必须带 A 的 rank×rank Gram：只取对角（B²@||A||²）仅在 A 行正交时
        # 成立（PiSSA 第 0 步），训练后 A 偏离正交会系统性低/高估范数。
        G = torch.matmul(A, A.t())  # (rank, rank)
        delta_sq = (s ** 2) * (torch.matmul(B, G) * B).sum(dim=1)  # (out,)
        # Init delta compensation (PiSSA / Ortho-LoRA):
        # ΔW = s·(B@A - B_i@A_i), so subtract init contributions
        if self.lora_down_init is not None and self.lora_up_init is not None:
            A_i = self.lora_down_init.float()
            B_i = self.lora_up_init.float()
            # ⟨W, -s·B_i@A_i⟩ per row
            if fast:
                WA_i = torch.matmul(Wd, A_i.t().to(Wd.dtype)).float()  # (out, rank)
            else:
                WA_i = torch.matmul(W, A_i.t())  # (out, rank)
            dot_i = s * (B_i * WA_i).sum(dim=1)  # (out,)
            # ||s·B_i@A_i||² per row（同样带 Gram；A_i 是 SVD 正交行时 G_i=I，
            # 与对角式一致，但不依赖这一假设）
            G_i = torch.matmul(A_i, A_i.t())  # (rank, rank)
            delta_i_sq = (s ** 2) * (torch.matmul(B_i, G_i) * B_i).sum(dim=1)  # (out,)
            # ⟨s·B@A, s·B_i@A_i⟩ per row = s² * (B * (B_i @ (A@A_i.T).T)).sum(dim=1)
            # A@A_i.T: (rank, rank), element [r,r'] = Σ_i A[r,i]·A_i[r',i]
            AAi = torch.matmul(A, A_i.t())  # (rank, rank)
            cross_inner = (B * torch.matmul(B_i, AAi.t())).sum(dim=1)  # (out,)
            cross = (s ** 2) * cross_inner  # (out,)
            # Combine: dot -= dot_i, delta_sq = delta_sq + delta_i_sq - 2·cross
            dot = dot - dot_i
            delta_sq = delta_sq + delta_i_sq - 2.0 * cross
        merged_sq = base_row_sq + 2.0 * dot + delta_sq
        return merged_sq.clamp(min=1e-12).sqrt()

    def delta_weight(self, apply_rank_dropout: bool = False) -> torch.Tensor:
        """Materialize ΔW for DoRA weight decomposition and export.

        含 PiSSA/ortho init 补偿：ΔW = (B@A - B_init@A_init) * scaling
        """
        A = self.lora_down.weight.float()
        B = self.lora_up.weight.float()
        if apply_rank_dropout and self.training and self.rank_dropout > 0:
            mask = torch.bernoulli(
                torch.full((self.rank,), 1.0 - self.rank_dropout, device=A.device)
            )
            scale = 1.0 / (1.0 - self.rank_dropout + 1e-6)
            A = A * (mask.unsqueeze(1) * scale)
        delta = torch.matmul(B, A) * self.scaling
        if self.lora_down_init is not None and self.lora_up_init is not None:
            init_delta = torch.matmul(
                self.lora_up_init.float(), self.lora_down_init.float()
            ) * self.scaling
            delta = delta - init_delta
        return delta


class ABBALayer(torch.nn.Module):
    """ABBA 适配器 (arXiv:2505.14238, ICLR 2026)：ΔW = (s1·B1@A1) ∘ (s2·B2@A2)

    Hadamard 双低秩：参数量 (r1+r2)(in+out)，有效秩上限 r1·r2。
    r1=r2=lora_rank/2 时参数预算与标准 LoRA rank=lora_rank 相同。
    可达集合严格包含标准 LoRA rank-r1 的全部解（B2@A2 学成全 1 矩阵即退化，
    全 1 矩阵 rank-1 可表示）——与 LoKr 的 kron 块共享结构不同，没有
    "对一般目标只能覆盖 ~1/f² 能量"的结构病理。

    前向用 Khatri-Rao 精确重排（官方 CERT-Lab/abba 同款，数学恒等无近似）：
        (B1@A1) ∘ (B2@A2) = B_kr @ A_kr
        B_kr[i,:] = B1[i,:] ⊗ B2[i,:]   → (out, r1·r2)   行向 KR
        A_kr[:,j] = A1[:,j] ⊗ A2[:,j]   → (r1·r2, in)    列向 KR
        ΔW·x = B_kr(A_kr·x) —— 全程不物化 (out, in) 矩阵。

    init（官方 init_weights_svd_mixed）：
        (B1,A1) ← W0 截断 SVD：B1=U√Σ, A1=√Σ·Vᵀ（svd_lowrank q=r1, niter=10）
        (B2,A2) ← B2=0, A2 kaiming → step 0 净 ΔW=0（行为中立）
    scaling 官方口径：s1=√alpha1, s2=√alpha2，self.scaling = s1·s2。

    不支持 rank_dropout / T-LoRA / DoRA（构造期 fail-fast，见 LoRALinear 校验）。
    """
    def __init__(self, in_features, out_features, r1=16, r2=16,
                 alpha1=16.0, alpha2=16.0, dropout=0.0, module_dropout=0.0,
                 rank_dropout=0.0, base_weight=None, device=None):
        super().__init__()
        if rank_dropout and float(rank_dropout) > 0:
            raise ValueError("ABBALayer 不支持 rank_dropout（4 因子乘性结构下 rank 通道"
                             "语义不明确）；请设 rank_dropout: 0")
        if base_weight is None:
            raise ValueError("ABBALayer 需要 base_weight 做 SVD init（官方 init 方案）")
        r_cap = min(in_features, out_features)
        if int(r1) > r_cap or int(r2) > r_cap:
            logger.warning(
                "[ABBA] r1/r2=%s/%s 超过层最小维 %d，收紧到 %d（SVD 截断秩上限）",
                r1, r2, r_cap, r_cap)
        self.r1 = min(int(r1), r_cap)
        self.r2 = min(int(r2), r_cap)
        self.rank = self.r1 * self.r2   # KR 有效秩（信息用途）
        self.alpha1 = float(alpha1)
        self.alpha2 = float(alpha2)
        # 官方口径：scaling1/2 = sqrt(alpha)，folded 进最终输出
        self.scaling = math.sqrt(self.alpha1) * math.sqrt(self.alpha2)

        self.abba_a1 = torch.nn.Parameter(torch.empty(self.r1, in_features, device=device))
        self.abba_b1 = torch.nn.Parameter(torch.empty(out_features, self.r1, device=device))
        self.abba_a2 = torch.nn.Parameter(torch.empty(self.r2, in_features, device=device))
        self.abba_b2 = torch.nn.Parameter(torch.empty(out_features, self.r2, device=device))
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        self.rank_dropout = 0.0
        self.module_dropout = float(module_dropout or 0.0)
        self.current_t = None
        self._md_keep: torch.Tensor | None = None
        self._md_compile_safe: bool = False

        # ── init（fp32 SVD，与 _pissa_init 同理由：bf16 SVD 数值不稳）─────
        with torch.no_grad():
            W = base_weight.detach().float().to(device)
            q = min(self.r1, min(in_features, out_features))
            U, S, V = torch.svd_lowrank(W, q=q, niter=10)
            # ★ svd_lowrank 返回 V (in, q) 而非 Vh —— 与 _pissa_init 同一坑位
            s_sqrt = S[:q].clamp(min=0).sqrt()
            self.abba_a1.copy_((V[:, :q] * s_sqrt.unsqueeze(0)).t())   # (r1, in)
            self.abba_b1.copy_(U[:, :q] * s_sqrt.unsqueeze(0))          # (out, r1)
            torch.nn.init.kaiming_uniform_(self.abba_a2)                # 官方默认口径
            torch.nn.init.zeros_(self.abba_b2)

    # ── module_dropout 机制：与 LoRALayer 语义一致 ─────────────────────
    def set_current_t(self, t):
        self.current_t = t   # ABBA 不消费 timestep；保留接口以兼容 LoRALinear 转发

    def set_module_dropout_compile_safe(self, flag):
        self._md_compile_safe = bool(flag)
        if not self._md_compile_safe:
            self._md_keep = None

    def roll_module_dropout(self):
        if self._md_compile_safe and self.training and self.module_dropout > 0:
            dev = self.abba_b2.device
            self._md_keep = (torch.rand((), device=dev) >= self.module_dropout).float()
        else:
            self._md_keep = None

    def clear_module_dropout(self):
        self._md_keep = None

    def khatri_rao_factors(self):
        """返回 (A_kr, B_kr) fp32：ΔW = scaling · B_kr @ A_kr（精确，供导出/取证）。"""
        a1 = self.abba_a1.float()
        a2 = self.abba_a2.float()
        b1 = self.abba_b1.float()
        b2 = self.abba_b2.float()
        a_kr = (a1.unsqueeze(1) * a2.unsqueeze(0)).reshape(self.r1 * self.r2, a1.shape[1])
        b_kr = (b1.unsqueeze(2) * b2.unsqueeze(1)).reshape(b1.shape[0], self.r1 * self.r2)
        return a_kr, b_kr

    def forward(self, x):
        # eager module dropout：与 LoRALayer 同款懒抽签早返回
        if (self.training and self.module_dropout > 0 and not self._md_compile_safe
                and torch.rand(1).item() < self.module_dropout):
            return torch.zeros(*x.shape[:-1], self.abba_b1.shape[0],
                               device=x.device, dtype=x.dtype)
        x_drop = self.dropout(x)
        a1, a2, b1, b2 = self.abba_a1, self.abba_a2, self.abba_b1, self.abba_b2
        # KR 因子逐 forward 重建（autograd 穿过；成本 O((r1·r2)(in+out))，远小于 matmul）
        a_kr = (a1.unsqueeze(1) * a2.unsqueeze(0)).reshape(self.r1 * self.r2, a1.shape[1])
        b_kr = (b1.unsqueeze(2) * b2.unsqueeze(1)).reshape(b1.shape[0], self.r1 * self.r2)
        if x_drop.dtype != a_kr.dtype:
            x_drop = x_drop.to(dtype=a_kr.dtype)
        h = F.linear(x_drop, a_kr)
        out = F.linear(h, b_kr) * self.scaling
        if self._md_keep is not None:
            out = out * self._md_keep.to(out.dtype)
        return out

    def delta_weight(self, apply_rank_dropout: bool = False) -> torch.Tensor:
        """Materialize ΔW（导出 / diff / merged_weight 用）。

        ΔW = scaling·(B1@A1)∘(B2@A2)。与 forward() 的 Khatri-Rao 重排 b_kr@a_kr
        严格恒等（(B1A1)∘(B2A2) = KR(B1,B2)@KR(A1,A2)，本地对拍 max_abs≈9e-9），
        这里用 Hadamard 形式：FLOPs 更少且直观。
        """
        del apply_rank_dropout  # ABBA 无 rank_dropout；保留签名与 LoKrLayer 一致
        d1 = torch.matmul(self.abba_b1.float(), self.abba_a1.float())
        d2 = torch.matmul(self.abba_b2.float(), self.abba_a2.float())
        return d1 * d2 * self.scaling


class LoKrLayer(torch.nn.Module):
    """LyCORIS LoKr 层 (ComfyUI 兼容) — w2 低秩分解版

    分解: ΔW = kron(w1, w2_a @ w2_b)
        w1   : (factor, factor)
        w2_a : (out_dim, rank)
        w2_b : (rank,    in_dim)
        其中 in_dim = in_features // factor, out_dim = out_features // factor

    forward 用 kron-bypass 数学等价但完全不实例化 (out_features, in_features) 全矩阵：
        将 x 视作 (..., factor, in_dim)，则 y = w1 @ ((x @ w2_b^T) @ w2_a^T)
        最后 reshape 回 (..., factor * out_dim)。
    复杂度 O(B*factor*in_dim*rank + B*factor*rank*out_dim + B*factor^2*out_dim)，
    远小于原本的 O(B*factor^2*in_dim*out_dim)，且无需在 bf16 中保存巨型 kron 矩阵。
    """
    def __init__(self, in_features, out_features, rank=4, alpha=1.0, factor=8, dropout=0.0,
                 rank_dropout=0.0, module_dropout=0.0,
                 tlora_enabled=False, tlora_rmin_ratio=0.5, tlora_alpha=1.0,
                 tlora_lokr_ortho_init=False, device=None, w1_init_std=0.1):
        super().__init__()
        self.alpha = alpha
        self.in_features = in_features
        self.out_features = out_features

        # 自动调整 factor 确保能整除。降级不是无害的（见 _find_factor 与 w1 的 1/f² 上界），
        # 所以把请求值留下来，由 injector 汇总成一条日志，让用户看得见实际生效的结构。
        self.requested_factor = int(factor)
        factor = self._find_factor(in_features, out_features, factor)
        self.factor = factor

        self.in_dim = in_features // factor
        self.out_dim = out_features // factor

        # cap rank 防止小层溢出
        self.rank = min(rank, self.out_dim, self.in_dim)
        self.scaling = alpha / self.rank

        # LoKr 分解: ΔW = kron(w1, w2_a @ w2_b)（命名与 LyCORIS 一致，可直接被 ComfyUI 加载）
        self.lokr_w1 = torch.nn.Parameter(torch.empty(factor, factor))
        self.lokr_w2_a = torch.nn.Parameter(torch.empty(self.out_dim, self.rank))
        self.lokr_w2_b = torch.nn.Parameter(torch.empty(self.rank, self.in_dim))
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        self.rank_dropout = float(rank_dropout or 0.0)
        self.module_dropout = float(module_dropout or 0.0)

        # T-LoRA (实验性，论文未覆盖 LoKr) state
        self.tlora_enabled = bool(tlora_enabled)
        self.tlora_alpha = float(tlora_alpha)
        if self.tlora_enabled:
            self.r_min = max(int(round(self.rank * float(tlora_rmin_ratio))), 1)
        else:
            self.r_min = self.rank
        self.current_t: torch.Tensor | None = None
        # Module-dropout keep 标量；语义同 LoRALayer._md_keep / _md_compile_safe。
        self._md_keep: torch.Tensor | None = None
        self._md_compile_safe: bool = False

        # ★ w1 用正态分布，配合 w2_b=0 初始时 ΔW=0（step-0 中立只靠 w2_b=0，与 w1 无关，
        #   所以 w1_init_std 怎么调都不破坏中立性）；训练后 ΔW 量级由 scaling 控制。
        #
        # 为什么这个 std 值要紧（2026-07-25 本地取证，见 memory
        # krea2-lokr-fullw2-counterexample）：kron 的第 (i,j) 个块 = w1[i,j]·w2，即 f² 个块
        # 全是同一个 w2 的标量倍，**w1 就是那 f² 个"块间调制标量"**。w1 若停在随机 init，
        # 该结构的表达力上界是闭式的 1/f²（f=8 时仅 1.56%，实测贴合），而 w1 学到位时
        # 最优可达 11–16% —— 差 7–10×。
        # 而 w1 只有 f² 个参数、AdamW 每步至多走 lr：默认 std=0.1（rms≈0.097）距离第三方
        # 成功件学成后的 |w1|rms 中位 0.49 差 5×，lr=1e-4 下要 ≥3900 步才爬得到，
        # 典型 run（数百~千步）根本来不及。把起点直接设到目标量级是最便宜的补救。
        # 默认保持 0.1 = 与改动前逐字节一致（行为中立）。
        _w1_std = float(w1_init_std)
        if not (_w1_std > 0.0):
            raise ValueError(
                f"lokr_w1_init_std 必须 > 0，得到 {_w1_std}。w1=0 会让 ΔW 恒为 0 且梯度全零"
                "（kron 对 w2 的梯度正比于 w1），整个 LoKr 永久死掉。")
        torch.nn.init.normal_(self.lokr_w1, mean=0.0, std=_w1_std)

        if self.tlora_enabled and bool(tlora_lokr_ortho_init):
            # 实验性 ortho init：在 (w2_a, w2_b) 的简化维度上套 SVD 方案；
            # w1 保留原初始化。论文未覆盖该组合，需要 init delta 补偿避免初始 step 偏移。
            A_init, B_init = _ortho_lora_init(self.in_dim, self.out_dim, self.rank, device=device)
            with torch.no_grad():
                self.lokr_w2_a.data.copy_(B_init)  # (out_dim, rank)
                self.lokr_w2_b.data.copy_(A_init)  # (rank, in_dim)
            self.register_buffer("lokr_w2_a_init", B_init.clone(), persistent=False)
            self.register_buffer("lokr_w2_b_init", A_init.clone(), persistent=False)
            # w1 的初始副本（fp32），forward 时 init 分支用同一个 w1（它本身不掩 mask）
            self.register_buffer("lokr_w1_init", self.lokr_w1.detach().clone(), persistent=False)
        else:
            torch.nn.init.kaiming_uniform_(self.lokr_w2_a, a=5**0.5)
            torch.nn.init.zeros_(self.lokr_w2_b)
            self.lokr_w2_a_init = None
            self.lokr_w2_b_init = None
            self.lokr_w1_init = None

    def _find_factor(self, in_f, out_f, target_factor):
        """找到能同时整除 in_features 和 out_features 的 factor（不超过 target）。

        ★ 旧实现只试 `[target, 4, 2, 1]`：target 一旦不整除就直接跌到 4/2/1，中间那些
          能整除的值（5、6、7…）全被跳过。后果不是报错而是**静默换结构** —— 例如
          target=6 在 6144×16384 上会退到 4，w2 变大 2.25×、体积翻数倍，用户看不出来。
          第三方成功件的等效 factor 正是 5、6 这类（w1 形状 (6,6)/(4,6)/(5,5)…）。

        现在改为从 target 往下找**最大的**公约因子，只有真的一个都没有才退到 1。
        f 的大小直接决定 w1 冻结时的表达力上界 1/f²（f=4→6.25%、f=8→1.56%），
        所以"悄悄换成更大或更小的 f"是有实际代价的，这里返回后由调用方打日志。
        """
        target = max(1, int(target_factor))
        for f in range(target, 0, -1):
            if in_f % f == 0 and out_f % f == 0:
                return f
        return 1

    def set_current_t(self, t):
        """由训练循环 / 采样循环在 model.forward 前调用。t 可为 None（重置）。"""
        self.current_t = t

    def set_module_dropout_compile_safe(self, flag):
        """语义同 LoRALayer.set_module_dropout_compile_safe。"""
        self._md_compile_safe = bool(flag)
        if not self._md_compile_safe:
            self._md_keep = None

    def roll_module_dropout(self):
        """compile-safe 模式下每 step 在 forward 前预抽 keep 标量；eager 模式不会被调用。"""
        if self._md_compile_safe and self.training and self.module_dropout > 0:
            dev = self.lokr_w1.device
            self._md_keep = (torch.rand((), device=dev) >= self.module_dropout).float()
        else:
            self._md_keep = None

    def clear_module_dropout(self):
        """forward 完成后重置，避免 keep=0 残留进后续采样/eval 前向。"""
        self._md_keep = None

    def _apply_tlora_mask_kron(self, tmp_flat, x_orig_shape):
        """对 kron-bypass 中间 tensor `tmp_flat` (P, factor, rank) 应用 per-sample mask。
        P = prod(x_orig_shape[:-1]) = B * prod(中间维)。返回 (tmp_after_mask, mask_BR or None).
        """
        if not (self.tlora_enabled and self.current_t is not None):
            return tmp_flat, None
        t = self.current_t
        if t.ndim == 0:
            t = t.view(1)
        B = int(t.shape[0])
        P = tmp_flat.shape[0]
        if P % B != 0:
            # B 不能整除 P：罕见，可能是 fused 调用方式；退化为 broadcast 第一个 t
            t = t[:1]
            B = 1
        N_per_sample = P // B
        mask_BR = _tlora_rank_mask(self.rank, self.r_min, t, self.tlora_alpha,
                                   tmp_flat.dtype, tmp_flat.device)
        # mask: (B, rank) → (B, 1, 1, rank) → expand → (P, factor, rank)
        mask_view = mask_BR.view(B, 1, 1, self.rank).expand(
            B, N_per_sample, self.factor, self.rank
        ).reshape(P, self.factor, self.rank)
        return tmp_flat * mask_view, mask_BR

    def _compute(self, x):
        self._rd_mask = None
        self._rd_scale = 1.0

        # ★ Training 路径：bf16 下 kron 容易数值放大，统一转 fp32 中间运算（必要）。
        # ★ Inference (eval + no_grad) 路径：可直接用原 dtype（通常 bf16），跳过 3 个 fp32 副本。
        #   - 推理时不积累梯度，bf16 精度对单步前向足够
        #   - 节省 ~3× LoKr 参数副本（对 5120ch model 大概 80MB / inject 层 -> 总省几 GB 临时显存）
        #   - 推理速度也快 ~1.5×（bf16 matmul tensor core）
        # rank_dropout / T-LoRA mask / ortho init 路径都需要严格的数值一致性 -> 训练路径仍 fp32。
        if self.training:
            w1 = self.lokr_w1.float()
            w2_a = self.lokr_w2_a.float()
            w2_b = self.lokr_w2_b.float()
            _compute_dtype = torch.float32
        else:
            _compute_dtype = self.lokr_w1.dtype  # 通常 bf16，与基模型 dtype 对齐
            w1 = self.lokr_w1
            w2_a = self.lokr_w2_a
            w2_b = self.lokr_w2_b

        # Rank dropout: 随机置零 rank 中的某些通道（训练时，inverted dropout）
        # ★ 同一个 rd_mask 必须同时作用到 w2_b 和后面的 w2b_init，否则破坏 step 0 净 delta=0
        # （详见 LoRALayer.forward 同名注释）。保留 rd 元组传到 ortho 补偿块用。
        if self.training and self.rank_dropout > 0:
            mask = torch.bernoulli(
                torch.full((self.rank,), 1.0 - self.rank_dropout, device=w2_b.device)
            )
            scale = 1.0 / (1.0 - self.rank_dropout + 1e-6)
            w2_b = w2_b * (mask.unsqueeze(1) * scale)  # (rank, in_dim)
            self._rd_mask = mask
            self._rd_scale = scale

        x_drop = self.dropout(x)
        orig_shape = x_drop.shape
        # (..., in_features) -> (B*, factor, in_dim)；保留前置维度
        # 推理路径用 compute_dtype（通常 bf16），训练路径转 fp32
        x_flat = x_drop.reshape(-1, self.factor, self.in_dim).to(dtype=_compute_dtype)

        # 两段低秩矩阵乘代替 kron 全矩阵：
        #   tmp = x_flat @ w2_b^T  -> (B*, factor, rank)
        #   T-LoRA mask 插在两段 matmul 之间（rank 维）
        #   tmp = tmp     @ w2_a^T -> (B*, factor, out_dim)
        tmp = torch.matmul(x_flat, w2_b.transpose(0, 1))
        tmp, mask_BR = self._apply_tlora_mask_kron(tmp, orig_shape)
        tmp = torch.matmul(tmp, w2_a.transpose(0, 1))
        # 用 (factor, factor) 在前广播：w1 @ (B*, factor, out_dim) -> (B*, factor, out_dim)
        y = torch.matmul(w1, tmp)

        # Ortho init compensation for LoKr T-LoRA：减去用 init 权重 + 同一 mask 的贡献，
        # 让训练 step 0 时净 delta ≈ 0。实验性，论文未覆盖。
        # ortho 补偿块只在 training=True 时有意义（推理时 T-LoRA mask 不动），
        # 因此一直走 fp32 是安全的（保数值精度）。
        if (self.tlora_enabled and self.lokr_w2_a_init is not None
                and self.lokr_w2_b_init is not None and self.current_t is not None):
            w1_init = self.lokr_w1_init.float()
            w2a_init = self.lokr_w2_a_init.float()
            w2b_init = self.lokr_w2_b_init.float()
            # 与 w2_b 用同一个 rd_mask，保 step 0 净 delta=0
            if self._rd_mask is not None:
                w2b_init = w2b_init * (self._rd_mask.unsqueeze(1) * self._rd_scale)
            tmp_i = torch.matmul(x_flat, w2b_init.transpose(0, 1))
            if mask_BR is not None:
                # 复用刚才算好的 mask
                B = int(mask_BR.shape[0])
                P = tmp_i.shape[0]
                N_per_sample = max(P // max(B, 1), 1)
                mask_view = mask_BR.view(B, 1, 1, self.rank).expand(
                    B, N_per_sample, self.factor, self.rank
                ).reshape(P, self.factor, self.rank)
                tmp_i = tmp_i * mask_view
            tmp_i = torch.matmul(tmp_i, w2a_init.transpose(0, 1))
            y_init = torch.matmul(w1_init, tmp_i)
            y = y - y_init

        # reshape 回 (..., out_features)
        y = y.reshape(*orig_shape[:-1], self.factor * self.out_dim)
        out = y.to(dtype=x.dtype) * self.scaling
        return out

    def forward(self, x):
        if (self.training and self.module_dropout > 0 and not self._md_compile_safe
                and torch.rand(1).item() < self.module_dropout):
            return torch.zeros(*x.shape[:-1], self.out_features,
                               device=x.device, dtype=x.dtype)
        out = self._compute(x)
        if self._md_keep is not None:
            out = out * self._md_keep.to(out.dtype)
        return out

    def merged_row_norms(self, base_weight, base_row_sq=None, fast=False):
        """Per-row L2 norms of (base_weight + ΔW) without materializing full delta.

        Uses: ||W + ΔW||² = ||W||² + 2⟨W, ΔW⟩ + ||ΔW||²
        For LoKr ΔW = scaling · kron(w1, U), U = w2a @ w2b:
          ||ΔW_{f,o}||² = scaling² · ||w1[f]||² · ||U[o]||²
          ⟨W_{f,o}, ΔW_{f,o}⟩ = scaling · Σ_{f_i} w1[f, f_i] · ⟨W_r[f,o,f_i,:], U[o,:]⟩

        峰值显存 = O(out_dim · in_dim)（即 U），远小于 O(out_features · in_features) 的 full delta。
        对 krea2 mlp 层：25MB vs 384MB（~15× 缩减）。

        必须在 _compute() 之后调用以复用同一 rank dropout mask。

        Args:
            base_row_sq: 可选的 ||W||² per-row 预计算缓存（fp32, (out_features,)）。
                W 冻结时该量全程不变，调用方缓存后可省去每次前向对全量 W 的
                cast + 平方 + 归约一整趟（LoRALinear.forward 的 DoRA 分支即如此）。
                数值与现算严格一致。
            fast: True 时 ⟨W,ΔW⟩ 的收缩直接用 base_weight 原 dtype（通常 bf16，
                matmul 内部 fp32 累加）计算，免去对全量 W 的 fp32 物化拷贝——
                读写流量 ~减半。范数量级 O(1)、仅作幅度归一，bf16 收缩的相对误差
                ~1e-3 量级；数值敏感场景保持 False（默认，逐 bit 不变）。
        """
        factor, out_dim, in_dim = self.factor, self.out_dim, self.in_dim

        if self.training:
            w1 = self.lokr_w1.float()
            w2_a = self.lokr_w2_a.float()
            w2_b = self.lokr_w2_b.float()
        else:
            w1 = self.lokr_w1
            w2_a = self.lokr_w2_a
            w2_b = self.lokr_w2_b

        # Apply same rank dropout mask as _compute()
        if getattr(self, '_rd_mask', None) is not None:
            w2_b = w2_b * (self._rd_mask.unsqueeze(1) * self._rd_scale)

        # U = w2a @ w2b: (out_dim, in_dim) -- 最大的中间量，远小于 full delta
        U = torch.matmul(w2_a, w2_b)

        # ||W||² per row（W 冻结 → 可由调用方缓存注入）
        if base_row_sq is not None:
            W_sq = base_row_sq.to(device=base_weight.device)  # (out_features,) fp32
        else:
            W_sq = (base_weight.float() ** 2).sum(dim=1)      # (out_features,)

        # ||ΔW||² per row = scaling² · ||w1[f_o]||² · ||U[o]||²
        w1_sq = (w1 ** 2).sum(dim=1)  # (factor,)
        U_sq = (U ** 2).sum(dim=1)    # (out_dim,)
        delta_sq = (self.scaling ** 2) * (
            w1_sq.unsqueeze(1) * U_sq.unsqueeze(0)  # (factor, out_dim)
        ).reshape(-1)  # (out_features,)

        # ⟨W, ΔW⟩ per row
        if fast:
            # 免物化：W 保持原 dtype（bf16 存储读它不损精度），einsum 走 tensor core
            # fp32 累加；仅乘法操作数是 bf16 舍入。
            W_r = base_weight.reshape(factor, out_dim, factor, in_dim)
            WU = torch.einsum(
                'abcd,bd->abc', W_r, U.to(base_weight.dtype)
            ).float()  # (f_o, o, f_i)
        else:
            W = base_weight.float()  # (out_features, in_features)
            W_r = W.reshape(factor, out_dim, factor, in_dim)  # (f_o, o, f_i, ii)
            WU = torch.einsum('abcd,bd->abc', W_r, U)  # (f_o, o, f_i)
        dot = self.scaling * torch.einsum('abc,ac->ab', WU, w1)  # (f_o, o)
        dot = dot.reshape(-1)  # (out_features,)

        # Ortho init: ΔW = scaling·(kron(w1,U) - kron(w1_init,U_init))
        if self.lokr_w2_a_init is not None and self.lokr_w2_b_init is not None:
            w1_init = (self.lokr_w1_init.float() if self.training else self.lokr_w1_init)
            w2a_init = (self.lokr_w2_a_init.float() if self.training else self.lokr_w2_a_init)
            w2b_init = (self.lokr_w2_b_init.float() if self.training else self.lokr_w2_b_init)
            if getattr(self, '_rd_mask', None) is not None:
                w2b_init = w2b_init * (self._rd_mask.unsqueeze(1) * self._rd_scale)
            U_init = torch.matmul(w2a_init, w2b_init)

            # Cross terms for ||ΔW||²
            w1_cross = (w1 * w1_init).sum(dim=1)   # (factor,)
            U_cross = (U * U_init).sum(dim=1)       # (out_dim,)
            w1_init_sq = (w1_init ** 2).sum(dim=1)  # (factor,)
            U_init_sq = (U_init ** 2).sum(dim=1)   # (out_dim,)
            delta_sq = delta_sq \
                - (self.scaling ** 2) * 2.0 * (w1_cross.unsqueeze(1) * U_cross.unsqueeze(0)).reshape(-1) \
                + (self.scaling ** 2) * (w1_init_sq.unsqueeze(1) * U_init_sq.unsqueeze(0)).reshape(-1)

            # Subtract init contribution from ⟨W, ΔW⟩（fast 模式 W_r 是 bf16，对齐 dtype）
            WU_init = torch.einsum(
                'abcd,bd->abc', W_r, U_init.to(W_r.dtype)
            ).float()
            dot_init = self.scaling * torch.einsum('abc,ac->ab', WU_init, w1_init)
            dot = dot - dot_init.reshape(-1)

        # ||W + ΔW||² = ||W||² + 2·dot + ||ΔW||²
        merged_sq = W_sq + 2.0 * dot + delta_sq
        return merged_sq.clamp(min=1e-12).sqrt()

    def delta_weight(self, apply_rank_dropout: bool = False) -> torch.Tensor:
        """Materialize 净 ΔW for DoRA weight decomposition and export checks.

        Ortho init（tlora_lokr_ortho_init）时训练 forward 的有效 delta 是
        kron(w1, w2a@w2b)·s − kron(w1_init, w2a_init@w2b_init)·s（init 贡献被补偿减去），
        这里同步减 init 项，merged_weight / diff / merged_model 导出才与训练语义一致。
        两个 kron 之差无法合并成单组标准 LoKr key —— native 导出模式在 LoRAInjector
        构造时已被 raise 拦下。
        """
        w1 = self.lokr_w1.float()
        w2_a = self.lokr_w2_a.float()
        w2_b = self.lokr_w2_b.float()
        w2b_init = (self.lokr_w2_b_init.float()
                    if self.lokr_w2_b_init is not None else None)

        if apply_rank_dropout and self.training and self.rank_dropout > 0:
            mask = torch.bernoulli(
                torch.full((self.rank,), 1.0 - self.rank_dropout, device=w2_b.device)
            )
            scale = 1.0 / (1.0 - self.rank_dropout + 1e-6)
            w2_b = w2_b * (mask.unsqueeze(1) * scale)
            # 与 forward 一致：同一 rd_mask 同时作用两支，保 step-0 净 delta=0
            if w2b_init is not None:
                w2b_init = w2b_init * (mask.unsqueeze(1) * scale)

        delta = torch.kron(w1, torch.matmul(w2_a, w2_b))
        if w2b_init is not None and self.lokr_w2_a_init is not None:
            delta = delta - torch.kron(
                self.lokr_w1_init.float(),
                torch.matmul(self.lokr_w2_a_init.float(), w2b_init),
            )
        return delta * self.scaling


class LoRALinear(torch.nn.Module):
    """LoRA 包装的 Linear 层（可选 DoRA / T-LoRA）"""
    def __init__(self, original, rank=4, alpha=1.0, dropout=0.0, use_lokr=False, factor=8,
                 rank_dropout=0.0, module_dropout=0.0, lora_variant="base",
                 tlora_rmin_ratio=0.5, tlora_alpha=1.0, tlora_init="default",
                 tlora_lokr_ortho_init=False,
                 dora_fast_norm=False, dora_detach_norm=False,
                 lora_init="default",
                 use_abba=False, abba_r1=None, abba_r2=None,
                 abba_alpha1=None, abba_alpha2=None, lokr_w1_init_std=0.1):
        super().__init__()
        self.original = original
        self.use_lokr = use_lokr
        self.use_abba = bool(use_abba)
        self.lora_variant = (lora_variant or "base").lower()
        self.use_dora = self.lora_variant == "dora"
        self.use_tlora = self.lora_variant == "tlora"
        self.lora_init = (lora_init or "default").lower()
        # DoRA 范数计算的两个 opt-in 加速开关（默认 False = 逐 bit 与历史一致）：
        # - fast_norm: ⟨W,ΔW⟩ 收缩用 W 原 dtype（bf16 tensor core, fp32 累加），
        #   免全量 fp32 物化；范数相对误差 ~1e-3。
        # - detach_norm: ||W+ΔW|| 对 LoKr 因子 detach（PEFT DoRA 参考实现同款，
        #   出自 DoRA 论文 Sec 4.3 的省显存梯度近似）；省 backward 里对全量 W
        #   的再收缩与范数支路的 autograd 图。梯度有轻微改变 → 必须 opt-in。
        self.dora_fast_norm = bool(dora_fast_norm)
        self.dora_detach_norm = bool(dora_detach_norm)
        # ||W||² per-row 缓存（W 冻结不变；惰性初始化以跟随 device 迁移）
        self._dora_base_row_sq = None

        # 组合兼容性校验：
        # - tlora 主路径只支持 lora；与 lokr 的组合属于实验性，由 injector 层显式 opt-in
        # - tlora × dora 不支持
        if self.use_tlora and self.use_dora:
            raise ValueError("lora_variant='tlora' is incompatible with DoRA")
        # - ABBA 首版保持最小变量面：不与 lokr / dora / tlora / pissa / rank_dropout 组合
        if self.use_abba:
            if self.use_lokr:
                raise ValueError("lora_type='abba' 与 lokr 互斥")
            if self.use_dora or self.use_tlora:
                raise ValueError("lora_type='abba' 暂不支持 lora_variant='dora'/'tlora'"
                                 "（首版单变量验证；如需组合，先在云端确认 ABBA 基线拟合）")
            if self.lora_init not in ("default",):
                raise ValueError("lora_type='abba' 自带官方 SVD init，与 lora_init="
                                 f"{self.lora_init!r} 冲突；请保持 lora_init: default")
            if rank_dropout and float(rank_dropout) > 0:
                raise ValueError("lora_type='abba' 不支持 rank_dropout；请设 rank_dropout: 0")

        # Ortho init 里的 SVD 走 original.weight.device（通常 GPU），避免 CPU SVD
        # 在大尺寸层（>5120）上每层 10–50 秒、整网累积一小时的开销
        svd_device = original.weight.device
        if self.use_abba:
            r_half = max(int(rank) // 2, 1)
            r1 = int(abba_r1) if abba_r1 else r_half
            r2 = int(abba_r2) if abba_r2 else r_half
            self.adapter = ABBALayer(
                original.in_features, original.out_features,
                r1=r1, r2=r2,
                alpha1=float(abba_alpha1) if abba_alpha1 else float(r1),
                alpha2=float(abba_alpha2) if abba_alpha2 else float(r2),
                dropout=dropout, module_dropout=module_dropout,
                rank_dropout=rank_dropout,
                base_weight=original.weight, device=svd_device,
            )
        elif use_lokr:
            self.adapter = LoKrLayer(
                original.in_features, original.out_features,
                rank=rank, alpha=alpha, factor=factor, dropout=dropout,
                rank_dropout=rank_dropout, module_dropout=module_dropout,
                tlora_enabled=self.use_tlora,
                tlora_rmin_ratio=tlora_rmin_ratio,
                tlora_alpha=tlora_alpha,
                tlora_lokr_ortho_init=tlora_lokr_ortho_init,
                device=svd_device,
                w1_init_std=lokr_w1_init_std,
            )
        else:
            self.adapter = LoRALayer(
                original.in_features, original.out_features,
                rank=rank, alpha=alpha, dropout=dropout,
                rank_dropout=rank_dropout, module_dropout=module_dropout,
                tlora_enabled=self.use_tlora,
                tlora_rmin_ratio=tlora_rmin_ratio,
                tlora_alpha=tlora_alpha,
                tlora_init=tlora_init,
                lora_init=self.lora_init,
                base_weight=original.weight,
                device=svd_device,
            )

        self.adapter.to(device=original.weight.device, dtype=original.weight.dtype)
        if self.use_dora:
            row_norm = original.weight.detach().float().norm(dim=1).clamp(min=1e-6)
            self.dora_scale = torch.nn.Parameter(row_norm.to(device=original.weight.device))
        for p in self.original.parameters():
            p.requires_grad = False

    def set_current_t(self, t):
        """转发给底层 adapter，使其在 forward 时能拿到当前 timestep。"""
        self.adapter.set_current_t(t)

    def set_module_dropout_compile_safe(self, flag):
        """转发给底层 adapter；setup 时由 injector 一次性设定。"""
        self.adapter.set_module_dropout_compile_safe(flag)

    def roll_module_dropout(self):
        """转发给底层 adapter；compile-safe 模式下每 step forward 前由 injector 调用。"""
        self.adapter.roll_module_dropout()

    def clear_module_dropout(self):
        """转发给底层 adapter；forward 完成后由 injector 调用。"""
        self.adapter.clear_module_dropout()

    def forward(self, x):
        if self.use_dora:
            adapter = self.adapter
            # eager module dropout: 命中即退回 base（无 LoRA / 无 DoRA）
            if (self.training and adapter.module_dropout > 0 and not adapter._md_compile_safe
                    and torch.rand(1).item() < adapter.module_dropout):
                return self.original(x)

            # ★ Memory-efficient DoRA: 不实例化 (out, in) 全矩阵，改在输出域分解。
            # y = (base_no_bias + lora_out) · (dora_scale / ||W + ΔW||) + bias
            # 其中 lora_out 走 LoKr 低秩前向（kron-bypass），||W + ΔW|| 走 merged_row_norms()。
            # 峰值显存从 O(out·in) 降到 O(out_dim·in_dim)（缩小 ~factor² 倍）。
            base_no_bias = F.linear(x, self.original.weight, None)
            lora_out = adapter._compute(x)  # 也存储 _rd_mask 供下一步复用
            # ||W||² per-row 缓存：W 冻结（requires_grad=False、不进 optimizer），
            # 该量全程不变 → 只算一次（device 迁移时重算）。严格等价于每次现算。
            _w = self.original.weight
            if (self._dora_base_row_sq is None
                    or self._dora_base_row_sq.device != _w.device):
                self._dora_base_row_sq = (_w.detach().float() ** 2).sum(dim=1)
            if self.dora_detach_norm:
                with torch.no_grad():
                    merged_norm = adapter.merged_row_norms(
                        _w, base_row_sq=self._dora_base_row_sq,
                        fast=self.dora_fast_norm,
                    )  # (out_features,), no grad
            else:
                merged_norm = adapter.merged_row_norms(
                    _w, base_row_sq=self._dora_base_row_sq,
                    fast=self.dora_fast_norm,
                )  # (out_features,)
            scale = (self.dora_scale.float() / merged_norm.clamp(min=1e-6))  # (out_features,)
            raw_out = base_no_bias.float() + lora_out.float()
            dora_no_bias = (raw_out * scale.unsqueeze(0)).to(dtype=x.dtype)

            # Module dropout compile-safe: blend in output domain
            # dora_w = base_w + keep·(dora_w − base_w)  =>  y = base + keep·(dora_y − base_no_bias)
            if adapter._md_keep is not None:
                keep = adapter._md_keep.to(dtype=dora_no_bias.dtype)
                dora_no_bias = base_no_bias + keep * (dora_no_bias - base_no_bias)

            if self.original.bias is not None:
                return dora_no_bias + self.original.bias.to(dtype=dora_no_bias.dtype)
            return dora_no_bias
        return self.original(x) + self.adapter(x)

    @property
    def weight(self):
        return self.original.weight

    @property
    def bias(self):
        return self.original.bias

    # nn.Linear 元数据透传：被包住的层对外仍表现为 Linear。
    # krea2 forward_packed_navit 读 self.first.in_features 校验 token 维、
    # Anima packed 路径读 x_embedder.proj[1].in_features —— targets 覆盖这些层
    # （如官方全 264 Linear 口径含 first）时缺透传会直接 AttributeError。
    @property
    def in_features(self):
        return self.original.in_features

    @property
    def out_features(self):
        return self.original.out_features

    def merged_weight(self) -> torch.Tensor:
        base_w = self.original.weight.float()
        if self.use_lokr or self.use_abba:
            delta = self.adapter.delta_weight(apply_rank_dropout=False).to(device=base_w.device)
        else:
            delta = torch.matmul(
                self.adapter.lora_up.weight.float(),
                self.adapter.lora_down.weight.float(),
            ) * self.adapter.scaling
            # T-LoRA Ortho init: 减去初始 delta 补偿（mask=I 即满 rank，等价于训练时
            # current_t=0 的 forward 行为）。这样合并出的权重表示"net 训练增量"。
            if (self.use_tlora
                    and getattr(self.adapter, "lora_down_init", None) is not None
                    and getattr(self.adapter, "lora_up_init", None) is not None):
                init_delta = torch.matmul(
                    self.adapter.lora_up_init.float(),
                    self.adapter.lora_down_init.float(),
                ).to(device=base_w.device) * self.adapter.scaling
                delta = delta - init_delta
            elif (not self.use_tlora
                    and getattr(self.adapter, "lora_down_init", None) is not None
                    and getattr(self.adapter, "lora_up_init", None) is not None):
                init_delta = torch.matmul(
                    self.adapter.lora_up_init.float(),
                    self.adapter.lora_down_init.float(),
                ).to(device=base_w.device) * self.adapter.scaling
                delta = delta - init_delta
        merged = base_w + delta
        if self.use_dora:
            denom = merged.norm(dim=1, keepdim=True).clamp(min=1e-6)
            merged = merged * (self.dora_scale.float().view(-1, 1) / denom)
        return merged


class LoRAInjector:
    """LoRA 注入器（支持 regex 模块选择、模块级 rank/lr、LoRA+）

    核心增强（从 kohya sd-scripts 借鉴）：
    - exclude_patterns / include_patterns: 用 re.fullmatch 替代前缀匹配
    - reg_dims: dict{regex: rank} 模块级 rank 控制
    - reg_lrs:  dict{regex: lr}   模块级学习率控制
    - loraplus_lr_ratio: LoRA+ w2_b/lora_up 用更高 lr
    """
    DEFAULT_TARGETS = ["q_proj", "k_proj", "v_proj", "output_proj", "mlp.layer1", "mlp.layer2"]
    DEFAULT_EXCLUDE_PATTERNS = [r".*llm_adapter.*"]

    def __init__(self, rank=32, alpha=16.0, dropout=0.0, use_lokr=False, factor=8,
                 targets=None, exclude_prefixes=None,
                 exclude_patterns=None, include_patterns=None,
                 reg_dims=None, reg_lrs=None, reg_alphas=None,
                 rank_dropout=0.0, module_dropout=0.0,
                 loraplus_lr_ratio=1.0, lora_variant="base", dora_export_mode="native",
                 # T-LoRA (arxiv:2507.05964)
                 tlora_rmin_ratio=0.5, tlora_alpha=1.0, tlora_init="ortho",
                 tlora_lokr_experimental=False, tlora_lokr_ortho_init=False,
                 # T-LoRA 导出体积控制
                 tlora_skip_lambda_layer=True,
                 # DoRA 范数加速开关（见 LoRALinear.__init__ 注释；默认关 = 行为不变）
                 dora_fast_norm=False, dora_detach_norm=False,
                 lora_init="default",
                 # ABBA (arXiv:2505.14238)：lora_type='abba' 时 use_abba=True。
                 # 每模块 r1=r2=mod_rank//2（参数预算=同 rank 标准 LoRA）；
                 # abba_alpha 覆盖 alpha1=alpha2（默认 None → 官方口径 alpha=r）。
                 # abba_export_kr：save() 时是否额外写 KR 物化的标准 LoRA 键
                 # （rank=r1·r2，体积 ~8×，仅当想让成品直接被 ComfyUI 加载时开；
                 # 默认 False = 只存 native 因子（体积 = 同预算 LoRA），部署件
                 # 用 tools/abba_export_lora.py 在本地转换/压缩）。
                 use_abba=False, abba_alpha=None, abba_export_kr=False,
                 # ── LoKr w1 的两个旋钮（默认值 = 与改动前逐字节一致）────────────────
                 # w1 是 kron 的"块间调制标量"（f² 个），冻在 init 时该结构表达力上界
                 # 只有 1/f²；而它参数少、AdamW 每步至多走 lr，典型 run 爬不到位。
                 # 详见 LoKrLayer.__init__ 的 w1_init_std 注释与 get_param_groups。
                 lokr_w1_init_std=0.1, lokr_w1_lr_ratio=1.0,
                 # ── Layer A：导出期 SVD 压缩（save() 额外写压缩件；默认 off = 行为中立）──
                 lora_compress_energy=1.0, lora_compress_max_rank=0,
                 lora_compress_budget_mb=0.0, lora_compress_replace_main=False,
                 # ── Layer B：AC-LoRA 训练期 RESTART（arXiv:2504.02231；默认 off）──
                 aclora_enabled=False, aclora_restart_every=200, aclora_warmup_steps=200,
                 aclora_p_mode="schedule", aclora_p_start=0.7, aclora_p_end=0.99,
                 aclora_p_floor=0.5, aclora_total_steps=0, aclora_loss_ema_beta=0.98):
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.use_lokr = use_lokr
        self.use_abba = bool(use_abba)
        self.abba_alpha = float(abba_alpha) if abba_alpha else None
        self.abba_export_kr = bool(abba_export_kr)
        self.factor = factor
        # ── LoKr w1 旋钮（校验放构造期，别等到跑起来才炸）────────────────────
        self.lokr_w1_init_std = float(lokr_w1_init_std)
        self.lokr_w1_lr_ratio = float(lokr_w1_lr_ratio)
        if self.use_lokr:
            if not (self.lokr_w1_init_std > 0.0):
                raise ValueError(
                    f"lokr_w1_init_std 必须 > 0，得到 {self.lokr_w1_init_std}。"
                    "w1=0 会让 ΔW 恒为 0 且 w2 的梯度也恒为 0（kron 对 w2 的梯度正比于 w1），"
                    "整个 LoKr 永久死掉。")
            if not (self.lokr_w1_lr_ratio > 0.0):
                raise ValueError(
                    f"lokr_w1_lr_ratio 必须 > 0，得到 {self.lokr_w1_lr_ratio}"
                    "（=1.0 表示与其余因子同 lr，即改动前的行为；<1 会更慢，通常不是你想要的）")
        elif (self.lokr_w1_init_std != 0.1) or (self.lokr_w1_lr_ratio != 1.0):
            # 非 LoKr 路径设这两个参数没有任何效果，静默忽略会让 A/B 白跑一轮
            raise ValueError(
                "lokr_w1_init_std / lokr_w1_lr_ratio 只对 lora_type='lokr' 生效"
                f"（当前 lora_type 不是 lokr）。请去掉这两个参数，或改用 lora_type: lokr。")
        if self.use_abba:
            if self.use_lokr:
                raise ValueError("lora_type='abba' 与 lokr 互斥")
            if (lora_variant or "base").lower() != "base":
                raise ValueError("lora_type='abba' 暂不支持 lora_variant="
                                 f"{lora_variant!r}（首版单变量验证）")
            if rank_dropout and float(rank_dropout) > 0:
                raise ValueError("lora_type='abba' 不支持 rank_dropout；请设 rank_dropout: 0")
        # 完整训练配置快照；save() 时整体写进 safetensors metadata（成品自描述 / 防云端 yaml 丢失）。
        self.training_metadata: dict = {}
        self.lora_variant = (lora_variant or "base").lower()
        if self.lora_variant not in ("base", "dora", "tlora"):
            raise ValueError(f"Unknown lora_variant: {lora_variant}")
        self.lora_init = lora_init

        # T-LoRA 组合校验：
        # - 与 dora 互斥
        # - 与 lokr 组合属于实验性，论文未覆盖；需显式 opt-in（tlora_lokr_experimental=True）
        if self.lora_variant == "tlora":
            if self.use_lokr and not bool(tlora_lokr_experimental):
                raise ValueError(
                    "lora_variant='tlora' + lora_type='lokr' 属于实验性组合（论文 arxiv:2507.05964 仅覆盖标准 LoRA）。\n"
                    "如需启用，请显式在 YAML / CLI 中设 tlora_lokr_experimental=true，并注意：\n"
                    "  1) 没有现成的 ComfyUI loader 能在推理时给 LoKr 应用动态 mask；\n"
                    "  2) 导出的 checkpoint 在 ComfyUI 里会退化为标准 LoKr（满 rank）；\n"
                    "  3) Ortho 初始化在 Kronecker 分解上是我们做的启发式扩展，不在原论文范围。"
                )

        self.tlora_rmin_ratio = float(tlora_rmin_ratio or 0.5)
        self.tlora_alpha_param = float(tlora_alpha or 1.0)
        self.tlora_init = (tlora_init or "ortho").lower()
        if self.tlora_init not in ("ortho", "default"):
            raise ValueError(f"Unknown tlora_init: {tlora_init}")
        self.tlora_lokr_experimental = bool(tlora_lokr_experimental)
        self.tlora_lokr_ortho_init = bool(tlora_lokr_ortho_init)
        # lambda_layer = b_init @ a_init 是 (out, in) 全矩阵，与基模型对应层等大。
        # 数学上完全冗余（q_layer_init / p_layer_init 已经低秩存了 a_init / b_init，
        # loader 可自行重建乘积；且 T-LoRA 的 per-step rank mask 必须在 rank 维中间
        # 插入，不能用预乘的全矩阵承载）。默认 True 跳过，文件从 GB 量级降到 100MB 量级。
        # 仅当下游 loader 明确要求 `.lambda_layer` 键存在时再设 False。
        self.tlora_skip_lambda_layer = bool(tlora_skip_lambda_layer)
        self.dora_export_mode = (dora_export_mode or "native").lower()
        if self.dora_export_mode not in ("native", "diff", "merged_model"):
            raise ValueError(f"Unknown dora_export_mode: {dora_export_mode}")
        self.dora_fast_norm = bool(dora_fast_norm)
        self.dora_detach_norm = bool(dora_detach_norm)

        # Ortho init 的净 ΔW = kron(w1,w2a@w2b) − kron(w1_init,w2a_init@w2b_init)，
        # 两个 Kronecker 积之差数学上无法合并成单组标准 LoKr key（lokr_w1/w2_a/w2_b）
        # → native 导出必然与训练语义不一致。在构造期 fail-fast，别等训练完才发现。
        if (self.lora_variant == "tlora" and self.use_lokr
                and self.tlora_lokr_ortho_init and self.dora_export_mode == "native"):
            raise ValueError(
                "tlora_lokr_ortho_init=true 与 dora_export_mode='native' 不兼容：\n"
                "ortho 补偿的净 ΔW 是两个 Kronecker 积之差，无法表达为标准 LoKr key，"
                "native 导出会把 init 贡献错误地烘焙进权重。可选：\n"
                "  1) tlora_lokr_ortho_init: false（推荐：w2_b=0 起步天然 ΔW=0，"
                "native 导出严格自洽，ComfyUI 标准 loader 直载）；\n"
                "  2) dora_export_mode: 'diff' 或 'merged_model'（全矩阵导出，"
                "可精确补偿 init 项）。"
            )
        self.targets = targets or self.DEFAULT_TARGETS
        self.rank_dropout = float(rank_dropout or 0.0)
        self.module_dropout = float(module_dropout or 0.0)
        # module-dropout 路径：默认 eager（False）。setup 时由 set_module_dropout_compile_safe
        # 按 torch_compile 设定；eager 下 roll/clear 短路，零额外每步开销。
        self._md_compile_safe = False
        self.loraplus_lr_ratio = max(float(loraplus_lr_ratio or 1.0), 1.0)

        # ── Regex 模块选择 ──────────────────────────────────────────────
        if exclude_patterns is not None:
            self.exclude_patterns = list(exclude_patterns)
        elif exclude_prefixes is not None:
            self.exclude_patterns = [re.escape(p) + r".*" for p in exclude_prefixes] if exclude_prefixes else []
        else:
            self.exclude_patterns = list(self.DEFAULT_EXCLUDE_PATTERNS)
        self.include_patterns = list(include_patterns or [])

        # ── 模块级 rank/lr 控制 ────────────────────────────────────────
        self.reg_dims = dict(reg_dims) if reg_dims else {}
        self.reg_alphas = dict(reg_alphas) if reg_alphas else {}
        self.reg_lrs = dict(reg_lrs) if reg_lrs else {}
        self._reg_conflict_warned: set = set()

        self.injected = {}
        self._module_ranks = {}
        self._module_alphas = {}
        self._module_lrs = {}

        # ── Layer A：导出期 SVD 压缩配置 ────────────────────────────────
        self.lora_compress_energy = float(lora_compress_energy)
        self.lora_compress_max_rank = int(lora_compress_max_rank or 0)
        self.lora_compress_budget_mb = float(lora_compress_budget_mb or 0.0)
        if not (0.0 < self.lora_compress_energy <= 1.0):
            raise ValueError(f"lora_compress_energy 必须在 (0,1]，得到 {self.lora_compress_energy}")
        if self.lora_compress_budget_mb < 0.0:
            raise ValueError(
                f"lora_compress_budget_mb 不能为负，得到 {self.lora_compress_budget_mb}")
        # 两种分配策略互斥：预算模式（全局 σ²/字节最优）vs 逐层阈值模式。
        # 同开会让"到底谁说了算"变得不可预测，故构造期 fail-fast。
        if self.lora_compress_budget_mb > 0.0 and (
                self.lora_compress_energy < 1.0 or self.lora_compress_max_rank > 0):
            raise ValueError(
                "lora_compress_budget_mb（全局预算分配）与 lora_compress_energy/"
                "lora_compress_max_rank（逐层阈值）互斥，只能开一个。\n"
                "  · 想按体积出件（推荐）：只设 lora_compress_budget_mb，如 35.0；\n"
                "  · 想按逐层能量阈值：只设 lora_compress_energy，如 0.99。\n"
                "  依据：Krea2 c12port epoch19 实测，同等保留能量下全局预算分配"
                "体积小 1.3–3.2×（逐层 0.95→85.5MB/97.79% vs 全局 35.5MB/97.96%）。")
        _compress_on = (self.lora_compress_energy < 1.0
                        or self.lora_compress_max_rank > 0
                        or self.lora_compress_budget_mb > 0.0)
        # 只留压缩件、不写满 rank 主件。省磁盘/下载带宽，但满 rank 权重就只剩
        # training_state(.pt) 一个副本了 —— 调用方（anima_train.py）必须确保
        # save_state_every>0，否则续训能力会被永久丢弃。这里只能校验"压缩本身开着"。
        self.lora_compress_replace_main = bool(lora_compress_replace_main)
        if self.lora_compress_replace_main and not _compress_on:
            raise ValueError(
                "lora_compress_replace_main=true 需要同时开启压缩，否则没有压缩件可写、"
                "主件又被跳过 = 什么都不保存。\n"
                "  请设 lora_compress_budget_mb（推荐，如 35.0）"
                "或 lora_compress_energy/lora_compress_max_rank。")
        if _compress_on and (self.use_lokr or self.use_abba or self.lora_variant != "base"):
            raise ValueError(
                "导出期 SVD 压缩（lora_compress_energy<1 / lora_compress_max_rank>0 / "
                "lora_compress_budget_mb>0）首版只支持标准 LoRA"
                "（lora_type=lora, lora_variant=base）。\n"
                "  · LoKr/ABBA：请用各自的 native 导出 + tools/abba_export_lora.py；\n"
                "  · DoRA：ΔW 含 dora_scale 逐行重归一化，非单纯 B@A，压缩语义不一致。")

        # ── Layer B：AC-LoRA 训练期 RESTART 配置 ────────────────────────
        self._aclora = {
            "enabled": bool(aclora_enabled),
            "restart_every": int(aclora_restart_every or 0),
            "warmup_steps": int(aclora_warmup_steps or 0),
            "mode": str(aclora_p_mode or "schedule").lower(),
            "p_start": float(aclora_p_start),
            "p_end": float(aclora_p_end),
            "p_floor": float(aclora_p_floor),
            "total_steps": int(aclora_total_steps or 0),
            "loss_ema_beta": float(aclora_loss_ema_beta),
        }
        self._aclora_loss_ema = None
        if self._aclora["enabled"]:
            if self.use_lokr or self.use_abba or self.lora_variant != "base" or self.lora_init != "default":
                raise ValueError(
                    "aclora_enabled=true 首版只支持标准 LoRA（lora_type=lora, "
                    "lora_variant=base, lora_init=default）——RESTART 直接对 A/B 两个"
                    "矩阵做 SVD，LoKr/ABBA 的因子结构、DoRA 的幅度向量、PiSSA/ortho 的"
                    "init 补偿都会被它破坏。请先在标准 LoRA 上单变量验证。")
            if self._aclora["mode"] not in ("schedule", "loss"):
                raise ValueError(f"aclora_p_mode 必须是 'schedule' 或 'loss'，得到 {self._aclora['mode']}")
            if not (0.0 < self._aclora["p_floor"] < 1.0):
                raise ValueError(f"aclora_p_floor 必须在 (0,1)，得到 {self._aclora['p_floor']}")

    def _should_inject(self, name):
        """判断模块是否应该被注入 LoRA（regex 匹配）"""
        if not any(t in name for t in self.targets):
            return False
        excluded = False
        for pat in self.exclude_patterns:
            if re.fullmatch(pat, name):
                excluded = True
                break
        if excluded:
            for pat in self.include_patterns:
                if re.fullmatch(pat, name):
                    return True
            return False
        return True

    def _get_reg_dim(self, name):
        """获取模块的 rank（优先 reg_dims 匹配，否则用全局 rank）"""
        for pat, dim in self.reg_dims.items():
            if re.fullmatch(pat, name):
                return int(dim)
        return self.rank

    def _get_reg_lr(self, name):
        """获取模块的自定义 lr（None 表示用全局）"""
        for pat, lr in self.reg_lrs.items():
            if re.fullmatch(pat, name):
                return float(lr)
        return None

    def _get_reg_alpha(self, name):
        for pat, alpha in self.reg_alphas.items():
            if re.fullmatch(pat, name):
                return float(alpha)
        return float(self.alpha)

    def inject(self, model):
        """注入 LoRA 到模型。

        ★ 旧实现对每个注入点都从 `model` 起 `getattr` walk 整条 dotted path
          （`for p in parts[:-1]: parent = getattr(parent, p)`），O(N × depth)。
        新实现一次性 `dict(model.named_modules())`（O(N)），按 parent dotted name 直接查表，
        O(1) 每层。对 Anima 5120ch + 36 blocks 共 200+ 注入点，inject 阶段提速 ~5×。
        """
        rank_summary = {}
        # 注意：要在改 model 之前快照所有 module，否则后续 setattr 会让新 LoRALinear
        # 出现在 named_modules() 里，把 isinstance Linear 检查带歪。
        modules_snapshot = list(model.named_modules())
        modules_by_name = {n: m for n, m in modules_snapshot}

        for name, module in modules_snapshot:
            if not isinstance(module, torch.nn.Linear):
                continue
            if not self._should_inject(name):
                continue

            mod_rank = self._get_reg_dim(name)
            mod_alpha = self._get_reg_alpha(name)
            mod_lr = self._get_reg_lr(name)

            # ── reg_dims/reg_alphas 多模式命中审计 ──────────────────────────
            # 首个 fullmatch 生效（kohya 语义，顺序即优先级），但静默遮蔽曾导致
            # 实际 bug：".*self_attn.*" 抢先命中 "blocks.N.adaln_modulation_self_attn.1"，
            # adaln 拿到 48 而不是配置在后面的 8。这里把"多模式命中且取值不同"显式
            # 告警（按命中组合去重，避免 84 个 adaln 模块刷屏）。
            for _tbl, _label, _used in ((self.reg_dims, "lora_reg_dims", mod_rank),
                                        (self.reg_alphas, "lora_reg_alphas", mod_alpha)):
                _hits = [(p, v) for p, v in _tbl.items() if re.fullmatch(p, name)]
                if len(_hits) > 1 and any(float(v) != float(_used) for _, v in _hits[1:]):
                    _key = (_label, tuple(p for p, _ in _hits))
                    if _key not in self._reg_conflict_warned:
                        self._reg_conflict_warned.add(_key)
                        # 危险型 = 更长（更特定）的模式被排在后面遮蔽（adaln-48 事故型）→ WARNING；
                        # 预期型 = 特定模式刻意放前面压过通用兜底（块段倾斜的正常写法）→ INFO。
                        _danger = any(len(p) > len(_hits[0][0]) for p, _ in _hits[1:])
                        if _danger:
                            logger.warning(
                                "[%s] 模块 %s 同时命中 %s，按首个 %r=%s 生效——但有更特定的模式"
                                "被遮蔽！若这不是有意为之，把更特定的模式移到 dict 更前面。",
                                _label, name, _hits, _hits[0][0], _hits[0][1])
                        else:
                            logger.info(
                                "[%s] %s: 特定模式 %r=%s 覆盖通用兜底 %s（预期优先级顺序）。",
                                _label, name, _hits[0][0], _hits[0][1],
                                [(p, v) for p, v in _hits[1:]])

            abba_kwargs = {}
            if self.use_abba:
                r_half = max(int(mod_rank) // 2, 1)
                abba_kwargs = {
                    "use_abba": True,
                    "abba_r1": r_half, "abba_r2": r_half,
                    "abba_alpha1": self.abba_alpha or float(r_half),
                    "abba_alpha2": self.abba_alpha or float(r_half),
                }
            lora_linear = LoRALinear(
                module, rank=mod_rank, alpha=mod_alpha,
                dropout=self.dropout, use_lokr=self.use_lokr, factor=self.factor,
                rank_dropout=self.rank_dropout, module_dropout=self.module_dropout,
                lora_variant=self.lora_variant,
                lora_init=self.lora_init,
                tlora_rmin_ratio=self.tlora_rmin_ratio,
                tlora_alpha=self.tlora_alpha_param,
                tlora_init=self.tlora_init,
                tlora_lokr_ortho_init=self.tlora_lokr_ortho_init,
                dora_fast_norm=self.dora_fast_norm,
                dora_detach_norm=self.dora_detach_norm,
                lokr_w1_init_std=self.lokr_w1_init_std,
                **abba_kwargs,
            )

            parent_name, _, child_name = name.rpartition(".")
            parent = modules_by_name[parent_name] if parent_name else model
            setattr(parent, child_name, lora_linear)
            self.injected[name] = lora_linear
            self._module_ranks[name] = mod_rank
            self._module_alphas[name] = mod_alpha
            self._module_lrs[name] = mod_lr
            rank_summary[mod_rank] = rank_summary.get(mod_rank, 0) + 1

        exc_str = ", ".join(self.exclude_patterns) or "无"
        inc_str = ", ".join(self.include_patterns) or "无"
        rank_dist = ", ".join(f"r{r}×{c}" for r, c in sorted(rank_summary.items()))
        if self.lora_variant == "tlora":
            variant_label = "T-LoKr (实验性)" if self.use_lokr else "T-LoRA"
        elif self.lora_variant == "dora":
            variant_label = "DoRA-LoKr"
        elif self.use_abba:
            variant_label = "ABBA"
        else:
            variant_label = "LoKr" if self.use_lokr else "LoRA"
        logger.info(
            f"注入 {variant_label} 到 {len(self.injected)} 层 "
            f"（排除: [{exc_str}], 包含: [{inc_str}], rank 分布: {rank_dist}）"
        )
        if self.use_abba:
            _r_half = max(int(self.rank) // 2, 1)
            logger.info(
                f"  ABBA: r1=r2=rank/2（全局 rank={self.rank} → {_r_half}，参数预算=同 rank 标准 LoRA），"
                f"alpha1=alpha2={self.abba_alpha or float(_r_half):g}，"
                f"init=SVD(W0)+B2=0（arXiv:2505.14238，KR 有效秩上限 {_r_half * _r_half}）"
            )
        if self.use_lokr:
            # 实际生效的 factor 分布 + w1 旋钮。factor 会被 _find_factor 按整除性下调，
            # 而 f 直接决定 w1 冻结时的表达力上界 1/f²，所以必须让用户看见实际值。
            _facs = collections.Counter(
                int(getattr(l.adapter, "factor", 0)) for l in self.injected.values())
            _downgraded = sum(
                c for fv, c in _facs.items() if fv != int(self.factor))
            logger.info(
                f"  LoKr: 请求 factor={self.factor} → 实际生效 {dict(sorted(_facs.items()))}"
                + (f"（{_downgraded} 层因整除性被下调）" if _downgraded else "")
            )
            logger.info(
                f"  LoKr w1: init_std={self.lokr_w1_init_std}（rms≈{self.lokr_w1_init_std:.3f}）, "
                f"lr_ratio={self.lokr_w1_lr_ratio}；w1 冻结时表达力上界 1/f² = "
                + ", ".join(f"f{fv}→{1.0 / (fv * fv) * 100:.2f}%" for fv in sorted(_facs))
            )
        if self.lora_variant == "tlora":
            r_min_example = max(int(round(self.rank * self.tlora_rmin_ratio)), 1)
            logger.info(
                f"  T-LoRA: r_min_ratio={self.tlora_rmin_ratio} (r_min≈{r_min_example} for r={self.rank}), "
                f"alpha_schedule={self.tlora_alpha_param}, init={self.tlora_init}"
                + (f", lokr_ortho_init={self.tlora_lokr_ortho_init}" if self.use_lokr else "")
            )
            if self.use_lokr:
                logger.warning(
                    "  ⚠ T-LoRA + LoKr 属于实验性组合（论文未覆盖）。导出的 checkpoint 在 ComfyUI 中"
                    "会退化为标准 LoKr（满 rank 烘焙），动态 mask 仅训练 / sampling.py 出图时生效。"
                )
        if self.rank_dropout > 0 or self.module_dropout > 0:
            logger.info(f"  rank_dropout={self.rank_dropout}, module_dropout={self.module_dropout}")
        custom_alpha_modules = {
            n: a for n, a in self._module_alphas.items()
            if abs(float(a) - float(self.alpha)) > 1e-12
        }
        if custom_alpha_modules:
            for n, alpha in list(custom_alpha_modules.items())[:5]:
                rank = max(int(self._module_ranks.get(n, self.rank)), 1)
                logger.info(f"  module alpha: {n} -> {alpha:.3g} (scale={alpha / rank:.3g})")
            if len(custom_alpha_modules) > 5:
                logger.info(f"  ... {len(custom_alpha_modules)} modules have custom alpha")
        custom_lr_modules = {n: lr for n, lr in self._module_lrs.items() if lr is not None}
        if custom_lr_modules:
            for n, lr in list(custom_lr_modules.items())[:5]:
                logger.info(f"  模块级 lr: {n} → {lr:.2e}")
            if len(custom_lr_modules) > 5:
                logger.info(f"  ... 共 {len(custom_lr_modules)} 个模块有自定义 lr")
        return self.injected

    def get_params(self):
        """获取可训练参数"""
        params = []
        for lora in self.injected.values():
            params.extend(p for p in lora.parameters() if p.requires_grad)
        return params

    def set_current_t(self, t):
        """把当前 batch 的 timestep（形状 (B,) 或 None）写到每个 LoRA adapter。

        训练循环 / sampling.py 在 model.forward(...) 之前调用，forward 完成后传 None
        重置，避免跨 step 残留。仅 T-LoRA 变体在 forward 内会使用；其它 variant 忽略。
        """
        if self.lora_variant != "tlora":
            return
        for lora in self.injected.values():
            lora.set_current_t(t)

    def set_module_dropout_compile_safe(self, flag):
        """setup 时一次性设定 module-dropout 路径：

        - flag=True（torch_compile 开）：走 compile-safe，每 step roll 预抽 keep 标量，
          forward 只读它做乘法，把 RNG 移出可能被 compile 追踪的 block.forward_tokens。
        - flag=False（默认/eager）：走原版懒抽签，forward 内 rand 早返回；此时 roll/clear
          在本类直接短路返回，不再每步遍历所有注入层 —— eager 路径零额外每步开销。"""
        self._md_compile_safe = bool(flag)
        if self.module_dropout <= 0:
            return
        for lora in self.injected.values():
            lora.set_module_dropout_compile_safe(flag)

    def roll_module_dropout(self):
        """compile-safe 模式下每个训练 step 在 model.forward 前调用，为每个注入层预抽 keep 标量。
        eager 模式（默认）直接短路 —— dropout 由 forward 内懒抽签处理，无需每步遍历。"""
        if not self._md_compile_safe or self.module_dropout <= 0:
            return
        for lora in self.injected.values():
            lora.roll_module_dropout()

    def clear_module_dropout(self):
        """compile-safe 模式下 forward + backward 完成后调用，重置每层 keep 标量。eager 直接短路。"""
        if not self._md_compile_safe or self.module_dropout <= 0:
            return
        for lora in self.injected.values():
            lora.clear_module_dropout()

    # ── AC-LoRA 训练期 RESTART（arXiv:2504.02231）────────────────────────
    def aclora_active(self) -> bool:
        return bool(self._aclora.get("enabled"))

    def _aclora_compute_p(self, global_step: int) -> float:
        """按配置算当前阈值 p（保留累计能量占比到 p 之前的信号分量）。

        - mode="schedule"（默认，FM-稳健）：p 从 p_start 线性升到 p_end，
          进度 = global_step/total_steps（total_steps<=0 时恒为 p_end）。
        - mode="loss"（论文 Eq.5-6 口径，假设 loss<1）：p = 1 − l^α，
          α = global_step/total_steps + 1（total_steps<=0 时 α=1），l=clamp(loss_ema)。
        统一夹到 [p_floor, 0.999999]。
        """
        cfg = self._aclora
        total = cfg["total_steps"]
        if cfg["mode"] == "loss":
            alpha = (global_step / total if total > 0 else 0.0) + 1.0
            l = self._aclora_loss_ema if self._aclora_loss_ema is not None else 0.5
            l = min(max(float(l), 0.0), 0.999)
            p = 1.0 - l ** alpha
        else:
            frac = min(global_step / total, 1.0) if total > 0 else 1.0
            p = cfg["p_start"] + (cfg["p_end"] - cfg["p_start"]) * frac
        return float(min(max(p, cfg["p_floor"]), 0.999999))

    def aclora_restart(self, p: float):
        """对所有注入的标准 LoRA 层的 A、B 各做一次 RESTART（就地改 .data）。

        返回统计 dict：{层数, A/B 平均保留秩, 用到的 p}。仅标准 LoRA 生效
        （构造期已 fail-fast 排除 LoKr/ABBA/DoRA/PiSSA）。
        """
        keeps_a, keeps_b = [], []
        with torch.no_grad():
            for lora in self.injected.values():
                ad = lora.adapter
                new_a, ka = aclora_restart_matrix(ad.lora_down.weight, p)
                new_b, kb = aclora_restart_matrix(ad.lora_up.weight, p)
                ad.lora_down.weight.copy_(new_a)
                ad.lora_up.weight.copy_(new_b)
                keeps_a.append(ka)
                keeps_b.append(kb)
        n = max(len(keeps_a), 1)
        return {
            "layers": len(keeps_a),
            "p": float(p),
            "mean_keep_down": sum(keeps_a) / n,
            "mean_keep_up": sum(keeps_b) / n,
            "min_keep": min(keeps_a + keeps_b) if keeps_a else 0,
            "max_keep": max(keeps_a + keeps_b) if keeps_a else 0,
        }

    def aclora_step(self, global_step: int, loss_val=None):
        """训练循环每个 optimizer step 后调一次。

        更新 loss EMA；若已过 warmup 且 global_step 命中 restart_every 间隔，则触发
        一次 RESTART。返回 (did_restart: bool, stats: dict | None)。
        RESTART 会就地改写 A/B → 调用方若配置了重置优化器状态，应在拿到 did_restart=True
        后自行清理这些参数的 optimizer state（本方法不持有 optimizer）。
        """
        cfg = self._aclora
        if not cfg["enabled"]:
            return False, None
        if loss_val is not None:
            try:
                lv = float(loss_val)
            except (TypeError, ValueError):
                lv = None
            if lv is not None and math.isfinite(lv):
                b = cfg["loss_ema_beta"]
                if self._aclora_loss_ema is None:
                    self._aclora_loss_ema = lv
                else:
                    self._aclora_loss_ema = b * self._aclora_loss_ema + (1.0 - b) * lv
        every = cfg["restart_every"]
        if every <= 0 or global_step < cfg["warmup_steps"] or global_step % every != 0:
            return False, None
        p = self._aclora_compute_p(global_step)
        stats = self.aclora_restart(p)
        logger.info(
            "[AC-LoRA] RESTART @ step %d：p=%.4f，%d 层，保留秩 down 均值 %.1f / up 均值 %.1f "
            "（min %d, max %d）%s",
            global_step, p, stats["layers"], stats["mean_keep_down"], stats["mean_keep_up"],
            stats["min_keep"], stats["max_keep"],
            f"，loss_ema={self._aclora_loss_ema:.4f}" if self._aclora_loss_ema is not None else "",
        )
        return True, stats

    def get_param_groups(self, weight_decay, base_lr: float = 1.0, loraplus_lr_ratio=None):
        """获取参数组（支持 LoRA+、模块级 lr、LoKr w1 排除 weight_decay）"""
        ratio = max(float(loraplus_lr_ratio or self.loraplus_lr_ratio), 1.0)
        # w1 的倍率独立于 LoRA+：LoRA+ 的理论针对的是"零初始化的那个因子"（这里是 w2_b），
        # w1 是乘性门控、问题性质不同（量级不够而非从 0 起步），所以给它单独一个旋钮。
        w1_ratio = float(getattr(self, "lokr_w1_lr_ratio", 1.0) or 1.0)
        groups_dict = {}  # (wd, lr_mult, custom_lr) -> [params]

        for name, lora in self.injected.items():
            custom_lr = self._module_lrs.get(name)
            if self.use_lokr:
                # w1 是 kron 的块间调制标量（只有 f² 个）。它冻在 init 时整个结构的
                # 表达力上界是 1/f²（f=8 → 1.56%，本地闭式+数值双验证），而 w1 学到位
                # 时最优可达 11–16%。它参数少、AdamW 每步至多走 lr，典型 run 爬不到
                # 第三方成功件的量级（|w1|rms 中位 0.49 vs 我们 init 0.097）。
                # ★ loraplus_lr_ratio 抬的是 w2_b，**不抬 w1** —— 想加速 w1 用这个。
                key_w1 = (0.0, w1_ratio, custom_lr)
                key_w2a = (weight_decay, 1.0, custom_lr)
                key_w2b = (weight_decay, ratio, custom_lr)
                groups_dict.setdefault(key_w1, []).append(lora.adapter.lokr_w1)
                groups_dict.setdefault(key_w2a, []).append(lora.adapter.lokr_w2_a)
                groups_dict.setdefault(key_w2b, []).append(lora.adapter.lokr_w2_b)
                if getattr(lora, "use_dora", False):
                    key_dora = (0.0, 1.0, custom_lr)
                    groups_dict.setdefault(key_dora, []).append(lora.dora_scale)
            elif self.use_abba:
                # ABBA 分组按参数来源区分（关键是把零初始化的 b2 单独隔离）：
                #   a1/b1 = W0 截断 SVD 暖启动（承载有意义的大值）→ wd=0，不衰减暖启动。
                #     （注：wd 侵蚀量级 = lr×wd ≈ 1e-6/step，本身可忽略；wd=0 只是干净）
                #   a2 = kaiming init → 常规 wd。
                #   b2 = zeros init，是 step-0 唯一有梯度、需从 0 长起的瓶颈因子 →
                #     单独进 LoRA+ ratio 组，使 loraplus_lr_ratio 能定向只给 b2 加速
                #     （本地取证：ABBA 在 lr=1e-4 下 b2 长得比 LoRA up 慢 ~20–30×，
                #      定向抬 b2 lr 可追平；见 memory krea2-fitting-experiment-matrix）。
                key_svd  = (0.0,          1.0,   custom_lr)  # wd=0  for SVD factors a1, b1
                key_a2   = (weight_decay, 1.0,   custom_lr)  # wd for kaiming-init a2
                key_b2   = (weight_decay, ratio, custom_lr)  # wd + LoRA+ ratio for zeros-init b2
                groups_dict.setdefault(key_svd, []).append(lora.adapter.abba_a1)
                groups_dict.setdefault(key_svd, []).append(lora.adapter.abba_b1)
                groups_dict.setdefault(key_a2,  []).append(lora.adapter.abba_a2)
                groups_dict.setdefault(key_b2,  []).append(lora.adapter.abba_b2)
            else:
                key_down = (weight_decay, 1.0, custom_lr)
                key_up = (weight_decay, ratio, custom_lr)
                groups_dict.setdefault(key_down, []).append(lora.adapter.lora_down.weight)
                groups_dict.setdefault(key_up, []).append(lora.adapter.lora_up.weight)
                # DoRA 幅度向量：与 LoKr 分支同款（无 weight_decay —— 它是范数尺度，
                # 衰减会系统性压暗输出）。漏掉的话 dora_scale 永远不进 optimizer/zero_grad，
                # 幅度冻结在 ||W|| 初值、梯度还跨步累积。
                if getattr(lora, "use_dora", False):
                    key_dora = (0.0, 1.0, custom_lr)
                    groups_dict.setdefault(key_dora, []).append(lora.dora_scale)

        param_groups = []
        for (wd, lr_mult, custom_lr), params in groups_dict.items():
            if not params:
                continue
            group = {"params": params, "weight_decay": wd}
            if custom_lr is not None:
                group["lr"] = custom_lr * lr_mult
            elif lr_mult != 1.0:
                group["lr"] = float(base_lr) * lr_mult
            param_groups.append(group)

        if ratio > 1.0:
            if self.use_lokr:
                lr_target = "LoKr w2_b"
            elif self.use_abba:
                lr_target = "ABBA b2"
            else:
                lr_target = "lora_up"
            logger.info(f"[LoRA+] {lr_target} lr ×{ratio:.1f}（注意：不作用于 LoKr w1）")
        if self.use_lokr and w1_ratio != 1.0:
            logger.info(f"[LoKr] w1 lr ×{w1_ratio:.1f}（块间调制标量，独立于 LoRA+）")
        return param_groups

    @staticmethod
    def comfy_weight_decompose(base_weight: torch.Tensor, diff_weight: torch.Tensor,
                               dora_scale: torch.Tensor) -> torch.Tensor:
        """Emulate ComfyUI output-axis DoRA for 2D linear weights."""
        base_f = base_weight.float()
        diff_f = diff_weight.float().to(device=base_f.device)
        calc = base_f + diff_f
        scale = dora_scale.float().to(device=base_f.device).reshape(-1, 1)
        base_norm = base_f.norm(dim=1, keepdim=True).clamp(min=1e-6)
        return calc * (scale / base_norm)

    @staticmethod
    def lycoris_output_axis_dora(base_weight: torch.Tensor, diff_weight: torch.Tensor,
                                 dora_scale: torch.Tensor) -> torch.Tensor:
        """LyCORIS-style output-axis DoRA formula used by training forward."""
        calc = base_weight.float() + diff_weight.float().to(device=base_weight.device)
        scale = dora_scale.float().to(device=calc.device).reshape(-1, 1)
        calc_norm = calc.norm(dim=1, keepdim=True).clamp(min=1e-6)
        return calc * (scale / calc_norm)

    @staticmethod
    def comfy_native_dora_scale(lora: LoRALinear) -> torch.Tensor:
        """Convert internal DoRA magnitude to a ComfyUI output-axis scale.

        ComfyUI's output-axis weight_decompose normalizes by ||W||, while the
        training forward normalizes by ||W + delta||. Export an adjusted scale
        so native .lokr_w* + .dora_scale reproduces the trained merged weight.
        """
        base_w = lora.original.weight.detach().float()
        delta = lora.adapter.delta_weight(apply_rank_dropout=False).detach().to(device=base_w.device)
        merged = base_w + delta
        base_norm = base_w.norm(dim=1).clamp(min=1e-6)
        merged_norm = merged.norm(dim=1).clamp(min=1e-6)
        magnitude = lora.dora_scale.detach().float().to(device=base_w.device)
        return magnitude * (base_norm / merged_norm)

    def state_dict(self, export_for_comfy=False):
        """导出 LoRA 权重。

        Training checkpoints keep raw DoRA magnitude. ComfyUI native export
        gets an adjusted output-axis scale so its weight_decompose matches
        LoRALinear.merged_weight() exactly for the exported checkpoint.

        T-LoRA 标准路径（lora_type=lora, lora_variant=tlora）使用 LyCORIS-style
        key 命名（`q_layer` / `p_layer` / `lambda_layer`），可被 bghira/ComfyUI-T-LoRA
        loader 识别并在推理时应用动态 mask。lambda_layer 是 Ortho-LoRA 初始 delta
        的补偿 buffer（B_init @ A_init），ComfyUI 节点用它重建 net delta。
        """
        sd = {}
        for name, lora in self.injected.items():
            base = "lora_unet_" + name.replace(".", "_")
            mod_alpha = self._module_alphas.get(name, self.alpha)
            sd[f"{base}.alpha"] = torch.tensor(float(mod_alpha))
            if self.use_lokr:
                # bf16 存储：训练在 bf16 mixed precision 下进行，参数的有效精度本身就是 bf16。
                # fp32 保存只是浪费空间——ComfyUI 推理时 kron 积也在 bf16 下计算。
                # 文件大小减半（64MB → ~32MB）且推理结果无差异。
                # T-LoRA + LoKr 实验性路径：mask=I（满 rank）烘焙，直接存现有权重
                # （ComfyUI 加载即为标准 LoKr，无动态 mask）
                sd[f"{base}.lokr_w1"] = lora.adapter.lokr_w1.data.clone().bfloat16().cpu()
                sd[f"{base}.lokr_w2_a"] = lora.adapter.lokr_w2_a.data.clone().bfloat16().cpu()
                sd[f"{base}.lokr_w2_b"] = lora.adapter.lokr_w2_b.data.clone().bfloat16().cpu()
                if getattr(lora, "use_dora", False):
                    if export_for_comfy:
                        dora_scale = self.comfy_native_dora_scale(lora).bfloat16().cpu()
                    else:
                        dora_scale = lora.dora_scale.data.clone().bfloat16().cpu()
                    if export_for_comfy:
                        dora_scale = dora_scale.view(-1, 1)
                    sd[f"{base}.dora_scale"] = dora_scale
                # T-LoRA ortho init buffers（persistent=False，不存的话 resume 会重新
                # 随机 _ortho_lora_init → 补偿基准漂移，续训语义偏离原 run）。
                # native ComfyUI 导出在构造期已被 raise 拦下，这些 key 只出现在
                # 训练 checkpoint / diff / merged_model 配置的 run 里。
                ad = lora.adapter
                if (getattr(ad, "lokr_w2_a_init", None) is not None
                        and getattr(ad, "lokr_w2_b_init", None) is not None):
                    sd[f"{base}.lokr_w1_init"] = ad.lokr_w1_init.data.clone().bfloat16().cpu()
                    sd[f"{base}.lokr_w2_a_init"] = ad.lokr_w2_a_init.data.clone().bfloat16().cpu()
                    sd[f"{base}.lokr_w2_b_init"] = ad.lokr_w2_b_init.data.clone().bfloat16().cpu()
            elif self.use_abba:
                ad = lora.adapter
                # native 4 因子：resume 必需（KR 乘积无法唯一回推因子）。bf16 与训练精度一致。
                sd[f"{base}.abba_a1"] = ad.abba_a1.data.clone().bfloat16().cpu()
                sd[f"{base}.abba_b1"] = ad.abba_b1.data.clone().bfloat16().cpu()
                sd[f"{base}.abba_a2"] = ad.abba_a2.data.clone().bfloat16().cpu()
                sd[f"{base}.abba_b2"] = ad.abba_b2.data.clone().bfloat16().cpu()
                sd[f"{base}.abba_alpha1"] = torch.tensor(float(ad.alpha1))
                sd[f"{base}.abba_alpha2"] = torch.tensor(float(ad.alpha2))
                if export_for_comfy and self.abba_export_kr:
                    # KR 物化成标准 LoRA（精确恒等）：ΔW = scaling·B_kr@A_kr
                    #   = (alpha/rank)·up@down，取 rank=r1·r2、alpha=scaling·r1·r2。
                    # ComfyUI 直接可载（rank 变大但数学无损）；native abba_* 键会被
                    # 标准 loader 忽略（仅 resume 用）。fp32 计算 KR 再降 bf16 存储。
                    # ★ 默认关（abba_export_kr=false）：KR 键让文件膨胀 ~8×（云端
                    # 下载不友好）；native 因子信息完备，部署件在本地用
                    # tools/abba_export_lora.py 转换（可顺带 SVD 截断压缩）。
                    a_kr, b_kr = ad.khatri_rao_factors()
                    kr_rank = a_kr.shape[0]
                    sd[f"{base}.lora_down.weight"] = a_kr.bfloat16().cpu()
                    sd[f"{base}.lora_up.weight"] = b_kr.bfloat16().cpu()
                    sd[f"{base}.alpha"] = torch.tensor(float(ad.scaling) * kr_rank)
            else:
                if self.lora_variant == "tlora":
                    # LyCORIS-style key 命名，兼容 bghira/ComfyUI-T-LoRA loader
                    sd[f"{base}.q_layer.weight"] = lora.adapter.lora_down.weight.data.clone()
                    sd[f"{base}.p_layer.weight"] = lora.adapter.lora_up.weight.data.clone()
                    # 初始 delta 补偿（仅 ortho init 时有）：
                    # - lambda_layer: 完整的 (out_features, in_features) 初始 delta，
                    #   ComfyUI loader 用它实现 Ortho-LoRA 的"减去初始贡献"
                    # - q_layer_init / p_layer_init: 我们自己 resume_lora 时需要的
                    #   完整 A_init/B_init（lambda_layer 是其乘积，单独无法回推）
                    if (getattr(lora.adapter, "lora_down_init", None) is not None
                            and getattr(lora.adapter, "lora_up_init", None) is not None):
                        # 始终存的低秩 init buffers（resume 必需；下游 loader 也能用它们
                        # 重建 lambda_layer = p_init @ q_init）。bf16 存储与训练精度一致。
                        a_init = lora.adapter.lora_down_init.float().cpu()
                        b_init = lora.adapter.lora_up_init.float().cpu()
                        sd[f"{base}.q_layer_init.weight"] = a_init.to(torch.bfloat16)
                        sd[f"{base}.p_layer_init.weight"] = b_init.to(torch.bfloat16)
                        # 可选：写入 (out, in) 全矩阵 lambda_layer = b_init @ a_init。
                        # 这是冗余信息（loader 可从上面两个低秩矩阵自己算），但某些下游
                        # loader 直接读这个 key。默认跳过：单层 lambda_layer 与 base 模型
                        # 对应层等大，Wan 1.3B 整网累积 ~3 GB，几乎全是这一项。
                        if not self.tlora_skip_lambda_layer:
                            sd[f"{base}.lambda_layer"] = torch.matmul(b_init, a_init).to(torch.bfloat16)
                else:
                    ad = lora.adapter
                    has_init = (getattr(ad, "lora_down_init", None) is not None
                                and getattr(ad, "lora_up_init", None) is not None)
                    if export_for_comfy and has_init:
                        # PiSSA/ortho 补偿式训练的净 ΔW = s·(B@A − B₀@A₀)，单组 rank-r
                        # 标准键表达不了 —— 直接存 B/A 会让 ComfyUI 把 base 权重的
                        # top-r 主成分加倍。按 PiSSA 官方转换（MuLabPKU/PiSSA：
                        # ΔW = [B | −B₀] @ [A; A₀]）折叠成 rank-2r 标准 LoRA。
                        # ComfyUI 的 scaling = alpha/dim，dim 翻倍 → alpha 同步 ×2
                        # 保持 s 不变。load_state_dict_from_mapping 能按 rank 切回
                        # A/B/A₀/B₀ 继续训练（无损往返）。
                        A = ad.lora_down.weight.data
                        B = ad.lora_up.weight.data
                        A0 = ad.lora_down_init.to(device=A.device, dtype=A.dtype)
                        B0 = ad.lora_up_init.to(device=B.device, dtype=B.dtype)
                        sd[f"{base}.lora_down.weight"] = torch.cat([A, A0], dim=0).clone()
                        sd[f"{base}.lora_up.weight"] = torch.cat([B, -B0], dim=1).clone()
                        sd[f"{base}.alpha"] = torch.tensor(float(mod_alpha) * 2.0)
                    else:
                        sd[f"{base}.lora_down.weight"] = ad.lora_down.weight.data.clone()
                        sd[f"{base}.lora_up.weight"] = ad.lora_up.weight.data.clone()
                        if has_init:
                            sd[f"{base}.lora_down_init.weight"] = ad.lora_down_init.data.clone().bfloat16().cpu()
                            sd[f"{base}.lora_up_init.weight"] = ad.lora_up_init.data.clone().bfloat16().cpu()
                    # DoRA 幅度：此前只有 LoKr 分支导出，标准 LoRA + DoRA 的成品
                    # 丢 dora_scale → 推理端连"按幅度重归一化"都复现不了；resume
                    # 也拿不回训练中的幅度。与 LoKr 分支同款处理。
                    if getattr(lora, "use_dora", False):
                        if export_for_comfy:
                            dora_scale = self.comfy_native_dora_scale(lora).bfloat16().cpu().view(-1, 1)
                        else:
                            dora_scale = lora.dora_scale.data.clone().bfloat16().cpu()
                        sd[f"{base}.dora_scale"] = dora_scale
        return sd

    def _diff_state_dict(self):
        """Export exact ComfyUI .diff patches for final-checkpoint parity."""
        sd = {}
        for name, lora in self.injected.items():
            base = "lora_unet_" + name.replace(".", "_")
            diff = lora.merged_weight().detach().cpu().float() - lora.original.weight.detach().cpu().float()
            sd[f"{base}.diff"] = diff.contiguous()
        return sd

    def _merged_model_state_dict(self, model):
        """Export a full transformer state dict with adapter weights baked in."""
        if model is None:
            raise ValueError("dora_export_mode='merged_model' requires save(..., model=model)")
        sd = {}
        injected = dict(self.injected)
        for key, tensor in model.state_dict().items():
            if ".adapter." in key or key.endswith(".dora_scale"):
                continue
            if key.endswith(".original.weight"):
                prefix = key[: -len(".original.weight")]
                lora = injected.get(prefix)
                if lora is not None:
                    sd[f"{prefix}.weight"] = lora.merged_weight().detach().cpu()
                    continue
            if key.endswith(".original.bias"):
                prefix = key[: -len(".original.bias")]
                lora = injected.get(prefix)
                if lora is not None and lora.original.bias is not None:
                    sd[f"{prefix}.bias"] = lora.original.bias.detach().cpu()
                    continue
            sd[key] = tensor.detach().cpu()
        return sd

    def set_training_metadata(self, config) -> None:
        """记录完整训练配置快照（通常传 `vars(args)`）。save() 时整体写进成品 metadata。

        只做浅拷贝；值的 JSON 友好降级延后到 save() 时（json.dumps default=str）。
        以 `_` 开头的私有键跳过。无副作用，调用失败不应影响训练。
        """
        try:
            self.training_metadata = {
                str(k): v for k, v in dict(config or {}).items() if not str(k).startswith("_")
            }
        except Exception as e:
            logger.warning(f"训练配置快照失败（忽略）: {e}")
            self.training_metadata = {}

    def _augment_meta_with_config(self, meta: dict) -> dict:
        """把完整训练配置以单个 JSON 字符串写进 safetensors metadata。

        key=`anima_training_config`（JSON），`anima_config_schema=v1`。safetensors 要求
        metadata 值全为 str；非 JSON 可序列化的值由 default=str 降级。空快照时原样返回。
        """
        if not self.training_metadata:
            return meta
        import json
        try:
            blob = json.dumps(self.training_metadata, ensure_ascii=False,
                              sort_keys=True, default=str)
        except Exception as e:
            logger.warning(f"训练配置 metadata 序列化失败，跳过全参快照: {e}")
            return meta
        out = dict(meta)
        out["anima_training_config"] = blob
        out["anima_config_schema"] = "v1"
        return out

    def save(self, path, model=None):
        """保存为 safetensors (ComfyUI 兼容)"""
        from safetensors.torch import save_file

        if self.dora_export_mode == "merged_model":
            sd = self._merged_model_state_dict(model)
            save_file(sd, path, metadata=self._augment_meta_with_config(
                {"format": "anima_merged_transformer"}))
            logger.info(f"合并模型保存到: {path}")
            return

        if self.dora_export_mode == "diff":
            sd = self._diff_state_dict()
            meta = {
                "format": "anima_lora_diff",
                "ss_network_module": "diff",
                "anima_export_mode": "diff",
            }
            save_file(sd, path, metadata=self._augment_meta_with_config(meta))
            logger.info(f"LoRA diff 保存到: {path}")
            return

        network_args = f'{{"algo": "lokr", "factor": {self.factor}}}' if self.use_lokr else "{}"
        if self.use_lokr and self.lora_variant == "dora":
            network_args = f'{{"algo": "lokr", "factor": {self.factor}, "dora_wd": true}}'
        if self.use_abba:
            _r_half = max(int(self.rank) // 2, 1)
            _a = self.abba_alpha or float(_r_half)
            network_args = (f'{{"algo": "abba", "r1": {_r_half}, "r2": {_r_half}, '
                            f'"alpha1": {_a:g}, "alpha2": {_a:g}}}')

        # T-LoRA 标准路径：用 bghira/ComfyUI-T-LoRA 能识别的 module 名 t_lora
        if self.lora_variant == "tlora" and not self.use_lokr:
            ss_network_module = "t_lora"
        elif self.use_lokr:
            ss_network_module = "lycoris.kohya"
        else:
            ss_network_module = "networks.lora"

        meta = {
            "ss_network_dim": str(self.rank),
            "ss_network_alpha": str(self.alpha),
            "ss_network_module": ss_network_module,
            "ss_network_args": network_args,
        }
        if self.use_lokr and self.lora_variant == "dora":
            meta["anima_dora_scale_format"] = "comfy_output_axis_adjusted"
        if self.use_abba:
            _r_half = max(int(self.rank) // 2, 1)
            _a = self.abba_alpha or float(_r_half)
            meta["anima_lora_variant"] = "abba"
            if self.abba_export_kr:
                # 标准键是 KR 物化（rank=r1·r2、alpha=scaling·r1·r2 已按层写进 tensor）；
                # ss_network_dim/alpha 元数据按 KR 口径覆盖，防下游按预算 rank 误读
                meta["ss_network_dim"] = str(_r_half * _r_half)
                meta["ss_network_alpha"] = str(float(_a) * _r_half * _r_half)
                meta["anima_abba_note"] = (
                    "Standard lora_down/lora_up keys are the exact Khatri-Rao "
                    "materialization of ABBA (arXiv:2505.14238); abba_* keys are the "
                    "native factors kept for resume and are safe for loaders to ignore."
                )
            else:
                # native-only（默认）：文件里没有标准 LoRA 键，标准 loader 无法直载。
                # 换成自有 module 名防误读；部署件用本地转换工具生成。
                meta["ss_network_module"] = "anima.abba"
                meta["anima_abba_note"] = (
                    "Native ABBA factors only (abba_a1/b1/a2/b2 + alpha1/alpha2 per "
                    "layer). Not loadable by standard LoRA loaders; convert with "
                    "tools/abba_export_lora.py (optionally with SVD truncation) "
                    "to produce a ComfyUI-loadable standard LoRA."
                )

        # T-LoRA 元数据（便于审计 + 下游 loader 解释 rank schedule）
        if self.lora_variant == "tlora":
            r_min_effective = max(int(round(self.rank * self.tlora_rmin_ratio)), 1)
            meta["anima_lora_variant"] = "tlora_lokr_experimental" if self.use_lokr else "tlora"
            meta["tlora_rmin"] = str(r_min_effective)
            meta["tlora_rmin_ratio"] = f"{self.tlora_rmin_ratio:.4f}"
            meta["tlora_schedule"] = "power_law"
            meta["tlora_alpha"] = f"{self.tlora_alpha_param:.4f}"
            meta["tlora_init"] = self.tlora_init
            if self.use_lokr:
                meta["anima_tlora_inference_supported"] = "false"
                meta["anima_tlora_inference_note"] = (
                    "LoKr+T-LoRA dynamic mask is training-only; baked checkpoint "
                    "loads as standard LoKr at inference."
                )
            else:
                meta["anima_tlora_inference_supported"] = "true"
                meta["anima_tlora_lambda_layer_present"] = (
                    "false" if self.tlora_skip_lambda_layer else "true"
                )
                if self.tlora_skip_lambda_layer:
                    meta["anima_tlora_inference_note"] = (
                        "Compact T-LoRA export: keys q_layer/p_layer/q_layer_init/p_layer_init. "
                        "lambda_layer (= p_init @ q_init) skipped to keep file size at ~LoRA scale. "
                        "Loaders that need lambda_layer must reconstruct it from the _init pieces."
                    )
                else:
                    meta["anima_tlora_inference_note"] = (
                        "Legacy T-LoRA export: keys q_layer/p_layer/lambda_layer/q_layer_init/p_layer_init. "
                        "Compatible with loaders that read lambda_layer directly."
                    )

        # replace_main：跳过满 rank 主件，直接把压缩件写到 path。构造期已保证此模式
        # 只可能出现在标准 base LoRA 上（压缩本身就 fail-fast 排除了 lokr/abba/dora/tlora）。
        # ★ 放在 state_dict() 之前：这个模式下满 rank 的 sd 根本用不上，264 层的
        #   CPU 拷贝纯属浪费（_maybe_save_compressed 自己从 adapter 取因子）。
        if self.lora_compress_replace_main:
            self._maybe_save_compressed(path, meta, replace_main=True)
            return

        sd = self.state_dict(export_for_comfy=True)
        save_file(sd, path, metadata=self._augment_meta_with_config(meta))
        if self.lora_variant == "tlora" and self.use_lokr:
            logger.warning(
                f"⚠ LoKr+T-LoRA 实验性 checkpoint 已保存到 {path}（满 rank 烘焙；"
                f"ComfyUI 加载即为标准 LoKr，动态 mask 仅训练 / 训练期 sampling.py 生效）。"
            )
        elif self.lora_variant == "tlora":
            mode = "compact (no lambda_layer)" if self.tlora_skip_lambda_layer else "legacy (+ lambda_layer)"
            logger.info(f"T-LoRA 保存到: {path}  [{mode}]")
        else:
            logger.info(f"LoRA 保存到: {path}")
            self._maybe_save_compressed(path, meta)

    @staticmethod
    def _compress_effective_pair(ad):
        """取出压缩用的等价 (down, up) 因子对。

        PiSSA/ortho 补偿式 init：净 ΔW = scaling·(B@A − B₀@A₀)，直接压 B@A 会把
        base 权重的 top-r 主成分算进去（成品部署全错）。压缩前先按官方 PiSSA 折叠
        （MuLabPKU/PiSSA：ΔW = scaling·[B|−B₀]@[A;A₀]）成 rank-2r 等价因子——与主
        comfy 导出（state_dict export_for_comfy 分支）同一口径。svd_truncate_lora_pair
        走 QR，支持任意秩输入，折叠后照压不物化全矩阵。lora_init=default 时
        lora_down_init 为 None → 走 else，与改动前逐字节一致（行为中立）。
        """
        if (getattr(ad, "lora_down_init", None) is not None
                and getattr(ad, "lora_up_init", None) is not None):
            A0 = ad.lora_down_init.to(device=ad.lora_down.weight.device,
                                      dtype=ad.lora_down.weight.dtype)
            B0 = ad.lora_up_init.to(device=ad.lora_up.weight.device,
                                    dtype=ad.lora_up.weight.dtype)
            return (torch.cat([ad.lora_down.weight, A0], dim=0),      # (2r, in)
                    torch.cat([ad.lora_up.weight, -B0], dim=1))       # (out, 2r)
        return ad.lora_down.weight, ad.lora_up.weight

    def _maybe_save_compressed(self, path, meta: dict, replace_main: bool = False):
        """Layer A：若开了导出压缩，额外写一份逐层 SVD 截断的部署件。

        主件（path，满 rank）保持不变——resume_lora 从它续训、信息完备；压缩件
        `{stem}.compressed.safetensors` 仅供部署（逐层变 rank，ComfyUI 直载）。
        默认关（energy=1.0 且 max_rank=0）→ 本方法直接返回，行为中立。
        仅标准 base LoRA 路径调用（构造期已 fail-fast 排除 lokr/abba/dora）。
        """
        energy = self.lora_compress_energy
        max_rank = self.lora_compress_max_rank
        budget_mb = self.lora_compress_budget_mb
        if not (energy < 1.0 or max_rank > 0 or budget_mb > 0.0):
            return
        import os as _os
        from safetensors.torch import save_file

        # 预算模式：先扫一遍全部层的谱，做全局 σ²/字节最优分配，得到逐层 keep。
        # 分两遍是为了省显存——第一遍只留奇异值（每层 ≤64 个数），不缓存因子；
        # 第二遍重算 QR 再截断。QR 在 r×r 上做，重算成本是秒级，可忽略。
        budget_keep = None
        if budget_mb > 0.0:
            spectra, per_rank_bytes = {}, {}
            for name, lora in self.injected.items():
                ad = lora.adapter
                eff_down, eff_up = self._compress_effective_pair(ad)
                spectra[name] = lora_pair_spectrum(eff_down, eff_up, float(ad.scaling))
                per_rank_bytes[name] = (eff_up.shape[0] + eff_down.shape[1]) * 2
            budget_keep = allocate_ranks_by_budget(
                spectra, per_rank_bytes, budget_mb * 2 ** 20, min_rank=1)

        comp = {}
        tot_in = tot_out = 0
        worst = (0.0, "")
        for name, lora in self.injected.items():
            base = "lora_unet_" + name.replace(".", "_")
            ad = lora.adapter
            eff_down, eff_up = self._compress_effective_pair(ad)
            if budget_keep is not None:
                # 预算模式：keep 已由全局分配器定死，用 max_rank 通道传进去
                # （energy=1.0 使能量准则不生效，两者取更紧者即 = 分配结果）。
                down, up, keep, dropped = svd_truncate_lora_pair(
                    eff_down, eff_up, float(ad.scaling),
                    energy=1.0, max_rank=budget_keep[name])
            else:
                down, up, keep, dropped = svd_truncate_lora_pair(
                    eff_down, eff_up, float(ad.scaling),
                    energy=energy, max_rank=max_rank)
            comp[f"{base}.lora_down.weight"] = down.to(torch.bfloat16).cpu().contiguous()
            comp[f"{base}.lora_up.weight"] = up.to(torch.bfloat16).cpu().contiguous()
            comp[f"{base}.alpha"] = torch.tensor(float(keep))
            r0 = eff_down.shape[0]
            in_f, out_f = eff_down.shape[1], eff_up.shape[0]
            tot_in += r0 * (in_f + out_f)
            tot_out += keep * (in_f + out_f)
            if dropped > worst[0]:
                worst = (dropped, base)
        if replace_main:
            comp_path = str(path)          # 压缩件即成品，不另起名
        else:
            stem, _ext = _os.path.splitext(str(path))
            comp_path = stem + ".compressed.safetensors"
        max_keep = max((int(t.shape[0]) for k, t in comp.items()
                        if k.endswith("lora_down.weight")), default=0)
        comp_meta = dict(meta)
        comp_meta.update({
            "ss_network_alpha": "per-layer",
            "ss_network_dim": str(max_keep),
            "anima_lora_variant": "svd_compressed",
            "anima_compress_energy": f"{energy}",
            "anima_compress_max_rank": str(max_rank or 0),
            "anima_compress_budget_mb": f"{budget_mb}",
        })
        save_file(comp, comp_path, metadata=self._augment_meta_with_config(comp_meta))
        if budget_keep is not None:
            ks = sorted(budget_keep.values())
            mode = (f"budget={budget_mb}MB（全局 σ²/字节最优分配；逐层 rank "
                    f"min={ks[0]} 中位={ks[len(ks) // 2]} max={ks[-1]}）")
        else:
            mode = f"energy={energy}, max_rank={max_rank or '∞'}（逐层阈值）"
        logger.info(
            "[压缩件%s] %s：满 rank %.1f MB → %.1f MB（%s，最大逐层丢弃能量 %.2f%% @ %s）",
            "·替代主件" if replace_main else "", comp_path,
            tot_in * 2 / 1e6, tot_out * 2 / 1e6, mode,
            worst[0] * 100.0, worst[1].removeprefix("lora_unet_"))
        if replace_main:
            logger.info(
                "  ↳ 未写满 rank 主件（lora_compress_replace_main=true）；"
                "满 rank 权重只在 training_state(.pt) 里，续训请用 --resume-state。")

    def load_state_dict_from_mapping(self, sd: dict, label: str = "checkpoint") -> int:
        """从 in-memory dict 加载 LoRA 权重。

        被 `load()`（safetensors）和 `checkpoint.load_training_state()`（torch.save 内嵌 dict）共享。
        旧实现两条路径各有一份独立的 lokr_w1/w2_a/w2_b 拷贝逻辑，存盘格式变动时容易漂移。

        返回成功加载的层数。
        """
        loaded_count = 0
        lokr_init_missing = 0
        pissa_init_missing = 0
        for name, lora in self.injected.items():
            base = "lora_unet_" + name.replace(".", "_")

            if self.use_lokr:
                w1_key = f"{base}.lokr_w1"
                w2a_key = f"{base}.lokr_w2_a"
                w2b_key = f"{base}.lokr_w2_b"
                dora_key = f"{base}.dora_scale"
                w2_old_key = f"{base}.lokr_w2"
                if w1_key in sd and w2a_key in sd and w2b_key in sd:
                    lora.adapter.lokr_w1.data.copy_(sd[w1_key])
                    lora.adapter.lokr_w2_a.data.copy_(sd[w2a_key])
                    lora.adapter.lokr_w2_b.data.copy_(sd[w2b_key])
                    if getattr(lora, "use_dora", False) and dora_key in sd:
                        dora_scale = sd[dora_key].reshape(-1)
                        lora.dora_scale.data.copy_(dora_scale.to(device=lora.dora_scale.device, dtype=lora.dora_scale.dtype))
                    # T-LoRA ortho init buffers：恢复补偿基准，保证续训语义与原 run
                    # 一致（语义同 plain-LoRA 分支的 q_layer_init/p_layer_init）
                    ad = lora.adapter
                    if getattr(ad, "lokr_w2_a_init", None) is not None:
                        init_keys = (f"{base}.lokr_w1_init", f"{base}.lokr_w2_a_init",
                                     f"{base}.lokr_w2_b_init")
                        if all(k in sd for k in init_keys):
                            for buf, k in zip((ad.lokr_w1_init, ad.lokr_w2_a_init,
                                               ad.lokr_w2_b_init), init_keys):
                                buf.copy_(sd[k].to(device=buf.device, dtype=buf.dtype))
                        else:
                            lokr_init_missing += 1
                    loaded_count += 1
                elif w1_key in sd and w2_old_key in sd:
                    logger.warning(f"跳过旧格式 lokr_w2 全矩阵层: {name}（需重新训练）")
            elif self.use_abba:
                keys = tuple(f"{base}.abba_{k}" for k in ("a1", "b1", "a2", "b2"))
                if all(k in sd for k in keys):
                    ad = lora.adapter
                    for p, k in zip((ad.abba_a1, ad.abba_b1, ad.abba_a2, ad.abba_b2), keys):
                        p.data.copy_(sd[k].to(device=p.device, dtype=p.dtype))
                    # alpha1/alpha2 若在文件里则以文件为准（跨配置 resume 时 scaling 不漂移）
                    a1k, a2k = f"{base}.abba_alpha1", f"{base}.abba_alpha2"
                    if a1k in sd and a2k in sd:
                        ad.alpha1 = float(sd[a1k])
                        ad.alpha2 = float(sd[a2k])
                        ad.scaling = math.sqrt(ad.alpha1) * math.sqrt(ad.alpha2)
                    loaded_count += 1
                elif f"{base}.lora_down.weight" in sd:
                    # 只有 KR 物化的标准 LoRA 键（缺 native 因子）：4 因子无法从乘积唯一
                    # 回推 → 明确拒绝，而不是静默加载错语义
                    logger.warning(
                        f"层 {name}: 文件只含 KR 物化的标准 LoRA 键、缺 abba_* native 因子，"
                        f"ABBA 无法 resume。请从含 abba_* 键的 checkpoint 恢复。")
            else:
                down_key = f"{base}.lora_down.weight"
                up_key = f"{base}.lora_up.weight"
                # T-LoRA LyCORIS-style 命名（兼容 bghira/ComfyUI-T-LoRA）
                q_key = f"{base}.q_layer.weight"
                p_key = f"{base}.p_layer.weight"
                if down_key in sd and up_key in sd:
                    ad = lora.adapter
                    down_t = sd[down_key]
                    up_t = sd[up_key]
                    r = ad.lora_down.weight.shape[0]
                    has_init = (getattr(ad, "lora_down_init", None) is not None
                                and getattr(ad, "lora_up_init", None) is not None)
                    folded = (down_t.shape[0] == 2 * r and up_t.shape[1] == 2 * r)
                    if folded:
                        # rank-2r 折叠导出（PiSSA/ortho 补偿，见 state_dict 注释）：
                        # down'=[A; A₀], up'=[B, −B₀] → 按 rank 切回四块，无损恢复训练态
                        A_t, A0_t = down_t[:r], down_t[r:]
                        B_t, B0_t = up_t[:, :r], -up_t[:, r:]
                        ad.lora_down.weight.data.copy_(A_t)
                        ad.lora_up.weight.data.copy_(B_t)
                        if not has_init:
                            # 当前 run 不是补偿式 init（如 lora_init=default 却 resume
                            # 了 PiSSA 成品）：补注册 init buffers，否则净 ΔW 语义会把
                            # B₀A₀（base 主成分）错误并入输出
                            dev = ad.lora_down.weight.device
                            dt = ad.lora_down.weight.dtype
                            del ad.lora_down_init
                            del ad.lora_up_init
                            ad.register_buffer(
                                "lora_down_init",
                                A0_t.to(device=dev, dtype=dt).clone(), persistent=False)
                            ad.register_buffer(
                                "lora_up_init",
                                B0_t.to(device=dev, dtype=dt).clone(), persistent=False)
                            logger.warning(
                                f"层 {name}: 加载了 rank-2r 折叠 PiSSA 成品，但当前配置"
                                f"不是补偿式 init —— 已从文件补建 init 补偿 buffers。")
                        else:
                            ad.lora_down_init.copy_(A0_t.to(
                                device=ad.lora_down_init.device,
                                dtype=ad.lora_down_init.dtype))
                            ad.lora_up_init.copy_(B0_t.to(
                                device=ad.lora_up_init.device,
                                dtype=ad.lora_up_init.dtype))
                    else:
                        ad.lora_down.weight.data.copy_(down_t)
                        ad.lora_up.weight.data.copy_(up_t)
                        # 训练态 raw 格式的 init buffers（PiSSA 补偿基准）。缺失时
                        # init 已在注入期被 svd_lowrank（随机化算法）重算 → 基准漂移
                        down_init_key = f"{base}.lora_down_init.weight"
                        up_init_key = f"{base}.lora_up_init.weight"
                        if has_init:
                            if down_init_key in sd and up_init_key in sd:
                                ad.lora_down_init.copy_(sd[down_init_key].to(
                                    device=ad.lora_down_init.device,
                                    dtype=ad.lora_down_init.dtype))
                                ad.lora_up_init.copy_(sd[up_init_key].to(
                                    device=ad.lora_up_init.device,
                                    dtype=ad.lora_up_init.dtype))
                            else:
                                pissa_init_missing += 1
                    # DoRA 幅度恢复。folded（=comfy 导出）里存的是换算后的
                    # output-axis scale（magnitude × ||W||/||W+ΔW||，见
                    # comfy_native_dora_scale），需按已恢复的 ΔW 精确逆换算回
                    # 训练态 magnitude；raw 格式直接拷贝。
                    dora_key = f"{base}.dora_scale"
                    if getattr(lora, "use_dora", False) and dora_key in sd:
                        scale_t = sd[dora_key].reshape(-1).float()
                        if folded:
                            base_w = lora.original.weight.detach().float()
                            delta = ad.delta_weight(apply_rank_dropout=False).detach().to(
                                device=base_w.device)
                            base_norm = base_w.norm(dim=1).clamp(min=1e-6)
                            merged_norm = (base_w + delta).norm(dim=1).clamp(min=1e-6)
                            scale_t = scale_t.to(device=base_w.device) * (merged_norm / base_norm)
                        lora.dora_scale.data.copy_(scale_t.to(
                            device=lora.dora_scale.device, dtype=lora.dora_scale.dtype))
                    loaded_count += 1
                elif q_key in sd and p_key in sd:
                    lora.adapter.lora_down.weight.data.copy_(sd[q_key])
                    lora.adapter.lora_up.weight.data.copy_(sd[p_key])
                    # 恢复 init A/B（resume_lora 必需，否则训练时的 Ortho-LoRA 补偿
                    # 会基于不同的随机 init → 续训行为偏离原 run）
                    q_init_key = f"{base}.q_layer_init.weight"
                    p_init_key = f"{base}.p_layer_init.weight"
                    if (q_init_key in sd and p_init_key in sd
                            and getattr(lora.adapter, "lora_down_init", None) is not None
                            and getattr(lora.adapter, "lora_up_init", None) is not None):
                        lora.adapter.lora_down_init.copy_(sd[q_init_key].to(
                            device=lora.adapter.lora_down_init.device,
                            dtype=lora.adapter.lora_down_init.dtype,
                        ))
                        lora.adapter.lora_up_init.copy_(sd[p_init_key].to(
                            device=lora.adapter.lora_up_init.device,
                            dtype=lora.adapter.lora_up_init.dtype,
                        ))
                    loaded_count += 1

        if lokr_init_missing:
            logger.warning(
                f"⚠ {lokr_init_missing} 层启用了 tlora_lokr_ortho_init，但 {label} 中没有 "
                f"lokr_*_init buffers（修复前的旧 checkpoint？）——这些层的 ortho 补偿基准"
                f"已重新随机生成，续训语义会偏离原 run。"
            )
        if pissa_init_missing:
            logger.warning(
                f"⚠ {pissa_init_missing} 层启用了 PiSSA/ortho 补偿式 init，但 {label} 中没有 "
                f"lora_*_init buffers（修复前的旧 checkpoint？）——这些层的补偿基准已由 "
                f"svd_lowrank（随机化算法）重算，续训净 ΔW 会与原 run 有漂移。"
            )
        logger.info(f"从 {label} 加载了 {loaded_count}/{len(self.injected)} 层 LoRA 权重")
        return loaded_count

    def load(self, path):
        """从 safetensors 加载已有 LoRA 权重（用于继续训练）"""
        from safetensors import safe_open

        logger.info(f"加载已有 LoRA 权重: {path}")

        sd = {}
        with safe_open(path, framework="pt", device="cpu") as f:
            # 压缩件是逐层变 rank 的 SVD 截断产物，只供部署。拿它续训会静默把
            # 训练权重换成截断版（丢掉尾部方向，且逐层 rank 与配置 rank 不符），
            # 因此在这里 fail-fast 而不是让后面的 shape 不匹配抛出难懂的错误。
            fmeta = f.metadata() or {}
            if fmeta.get("anima_lora_variant") == "svd_compressed":
                raise ValueError(
                    f"{path} 是导出压缩件（anima_lora_variant=svd_compressed），"
                    "不能用于续训——它是逐层 SVD 截断的部署件，尾部方向已丢弃。\n"
                    "  续训请用 training_state（.pt）：--resume-state <...>_state.pt\n"
                    "  （开了 lora_compress_replace_main 时，满 rank 权重只存在于 .pt 里。）")
            for k in f.keys():
                sd[k] = f.get_tensor(k)

        self.load_state_dict_from_mapping(sd, label=str(path))