"""Linear-DPO —— 偏好放大（in-loop, on-policy, latent 空间）的后训练阶段。

把"减小数据集劣势"换成"放大拟合/审美优势"：底模没做过美学评分训练 → 有真实审美
headroom；手筛训练集是审美正例集 → 我们拥有一批 **positives** 但没有打分器。Linear-DPO
（arXiv 2605.21123，flow-matching 原生 DPO）正好吃这个资产——它要的是偏好**配对**而非
reward 模型，损失本质 = 现有 per_sample 去噪误差的**加权差**。

配对构造（论文没给，自造）：`赢家 = 真·数据集图`（唯一 ground-truth"好"，**绝不生成**），
`输家 = 当前策略对同 caption 的生成`（in-loop、on-policy、留 latent 空间不 decode、粗轮重生成）。

本模块只负责**与 2B 模型解耦**的那几块（纯张量数学 / `.data` 交换 / 池子 / 轮门控），
全部 CPU 可单测；真正的前向、加噪、采样接线在 anima_train.py（见 spec 训练循环集成点）。
镜像 `trainer/gaf.py` 的 **opt-in / 默认关** 编排风格（enabled=False 时一切即时返回）。

损失（映射到我们的 flow-matching / velocity 设置）：

    L_θ(x0)  = per_sample_loss(v*, v_θ(x_t,t,c))        # 现有逐样本去噪误差
    D_θ(x0)  = L_θ(x0) − L_ref(x0)                       # 策略 − 冻结参考
    Δ        = D_θ(赢) − D_θ(输)                          # 配对 margin
    ω'       = clip(0.2·β̄·Δ + 0.5, η, 1)   （detached）  # 线性效用 + η 裁剪（抗过优化）
    L_DPO    = mean_pairs[ sg(ω')·(L_θ(赢) − L_θ(输)) ]

最小化 → 策略把**真图**去噪得更好、把**自己的采样**去噪得更差。ω' detach（stop-grad），
梯度只流经 `(L_θ(赢) − L_θ(输))`。配对内共享 (t, ε)（方差缩减，见 share_noise）。
"""

from __future__ import annotations

import contextlib
import csv
import math
import os

import torch

_EPS = 1e-12


# ============================================================================
# 决策核：Linear-DPO 配对损失（纯张量数学，CPU 可单测）
# ============================================================================

def linear_dpo_loss(
    l_theta_w: torch.Tensor,
    l_theta_l: torch.Tensor,
    l_ref_w: torch.Tensor,
    l_ref_l: torch.Tensor,
    *,
    beta: float,
    eta: float = 0.01,
):
    """Linear-DPO 配对损失。输入四个 per-pair 标量/向量（= per_sample_loss 的逐样本去噪误差）。

    - `l_theta_*` 必须带梯度（策略前向）；`l_ref_*` 来自参考前向（no_grad，已无图）。
    - ω' = clip(0.2·β̄·Δ + 0.5, η, 1) 整体 **detach**（stop-grad）：它只是逐配对权重，
      梯度只经 `(L_θ(赢) − L_θ(输))` 回传 → ∂L/∂L_θ(赢)=+ω'（压低真图误差），
      ∂L/∂L_θ(输)=−ω'（抬高自采样误差）。ω'≥η>0 保证恒有一点持续推力。

    返回 (loss_scalar, info)。info 里 omega/delta/margin 等均已 detach，供 CSV 诊断。
    """
    d_w = l_theta_w - l_ref_w
    d_l = l_theta_l - l_ref_l
    delta = d_w - d_l
    omega = torch.clamp(0.2 * float(beta) * delta + 0.5, min=float(eta), max=1.0).detach()
    margin = l_theta_w - l_theta_l
    loss = (omega * margin).mean()
    info = {
        "omega": omega.detach(),
        "delta": delta.detach(),
        "margin": margin.detach(),
        "d_w": d_w.detach(),
        "d_l": d_l.detach(),
    }
    return loss, info


# ============================================================================
# 参考 adapter：底模冻结共享 + 一份冻结的已收敛 adapter（.data 交换，近免费）
# ============================================================================

class ReferenceAdapter:
    """参考模型 = 底模（冻结共享）+ **一份冻结的已收敛 adapter**。

    持有每个可训 adapter 张量的冻结 `.data` 克隆；`swap()` 上下文里把 `p.data` 临时换成
    冻结副本、退出**精确还原**到策略权重。adapter 类型无关（lokr_w1/w2_a/w2_b/dora_scale…
    一视同仁），不在显存里再放一份底模。

    v1：参考**固定**（收敛快照，γ=1）。follow-up：`ema<1.0` 时每步 EMA 跟踪策略（论文最优
    0.995），仅当固定参考 plateau 才开。
    """

    def __init__(self, params, ema: float = 1.0):
        self.params = list(params)
        # 冻结副本：detach + clone（与 p 同 device/dtype，requires_grad=False）
        self.ref_data = [p.detach().clone() for p in self.params]
        self.ema = float(ema)

    @contextlib.contextmanager
    def swap(self):
        """把每个 adapter 参数的 .data 临时换成冻结参考副本；退出精确还原原 .data 张量对象。

        参考前向必须在 `torch.no_grad()` 下进行（调用方负责），故交换期间不建图、无梯度泄漏；
        优化器 step 在本上下文之外，看到的恒是还原后的策略 .data。
        """
        saved = [p.data for p in self.params]
        try:
            for p, ref in zip(self.params, self.ref_data):
                p.data = ref
            yield
        finally:
            for p, s in zip(self.params, saved):
                p.data = s

    @torch.no_grad()
    def update_ema(self) -> None:
        """EMA follow-up：ref ← ema·ref + (1−ema)·policy。ema>=1.0 时为固定参考（no-op）。"""
        if self.ema >= 1.0:
            return
        for ref, p in zip(self.ref_data, self.params):
            ref.mul_(self.ema).add_(p.detach().to(ref.dtype), alpha=1.0 - self.ema)


# ============================================================================
# 输家池：on-policy loser（latent 空间，粗轮重生成）
# ============================================================================

class DpoLoserPool:
    """on-policy 输家池：key（每个训练样本的稳定标识，如图片路径）→ loser latent（detach，
    训练 latent 形状 `[1,16,1,h,w]`，与加噪管线直接兼容）。

    粗轮重生成时只覆盖被选中的 subset，其余 key 保留上一轮的 loser（跨轮累积覆盖）。
    """

    def __init__(self):
        self._pool: dict[str, torch.Tensor] = {}

    def set(self, key, latent: torch.Tensor) -> None:
        self._pool[str(key)] = latent.detach()

    def get(self, key):
        return self._pool.get(str(key))

    def has(self, key) -> bool:
        return str(key) in self._pool

    def keys(self):
        return self._pool.keys()

    def __len__(self) -> int:
        return len(self._pool)

    def clear(self) -> None:
        self._pool.clear()


# ============================================================================
# 编排器：DpoController（默认关；enabled=False 时方法即时返回，零影响）
# ============================================================================

class DpoController:
    """Linear-DPO 阶段全流程编排，镜像 GafController 的 opt-in/默认关风格。

    职责（与 2B 模型解耦的部分）：
      - 轮门控 `should_regen(step)`（粗轮边界 = 每 regen_every 步，含 step 0 的第一轮）；
      - 输家池 + `regen_losers(sample_fn, items)`（按 loser_subset 选子集重生成）；
      - 参考 `.data` 交换上下文 `reference_mode()` + `update_reference_ema()`；
      - 决策核 `dpo_loss(...)`（调 linear_dpo_loss + 可选 winner SFT 锚 + 诊断 CSV）。

    真正的策略/参考前向、(t,ε) 加噪、sample_latent 采样在 anima_train.py 接线。
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        beta: float = 0.1,
        eta: float = 0.01,
        ref_ema: float = 1.0,
        regen_every: int = 1000,
        loser_steps: int = 14,
        loser_cfg: float = 1.0,
        loser_subset: float = 1.0,
        sft_anchor_lambda: float = 0.0,
        share_noise: bool = True,
        log_path: str = "",
        params=None,
        seed: int = 1234,
    ):
        self.enabled = bool(enabled)
        self.beta = float(beta)
        self.eta = float(eta)
        self.regen_every = max(int(regen_every or 1), 1)
        self.loser_steps = int(loser_steps)
        self.loser_cfg = float(loser_cfg)
        self.loser_subset = float(min(max(loser_subset, 0.0), 1.0))
        self.sft_anchor_lambda = float(sft_anchor_lambda or 0.0)
        self.share_noise = bool(share_noise)
        self.log_path = str(log_path or "")
        self.seed = int(seed)

        params = [p for p in (params or []) if getattr(p, "requires_grad", False)]
        self.reference = ReferenceAdapter(params, ema=ref_ema) if params else None
        # 启用要求有可训 adapter 才能算参考；否则降级为关（避免参考=策略的静默错算）
        self.enabled = self.enabled and self.reference is not None

        self.pool = DpoLoserPool()
        self.rounds = 0
        self.steps_done = 0
        self.last_info = None
        self._log_rows: list[tuple] = []

    # ── 轮门控 ──────────────────────────────────────────────────────────────
    def should_regen(self, step: int) -> bool:
        """粗轮边界：每 regen_every 步重生成一次输家池（step 0 = 第一轮，用收敛 LoRA 自生成）。"""
        return self.enabled and (int(step) % self.regen_every == 0)

    # ── 输家池重生成（sample_fn 由调用方注入 → CPU 可单测）────────────────────
    def regen_losers(self, sample_fn, items) -> int:
        """对 items 按 loser_subset 选子集，调 sample_fn 生成 loser latent 存入池。

        items: 可迭代的 (key, payload)；payload 透传给 sample_fn（caption/h/w 等由调用方约定）。
        sample_fn(key, payload) -> latent（或 None 表示本次跳过）。返回实际刷新条数。
        subset<1.0 时用 (seed+rounds) 种子随机选子集，跨轮逐步覆盖全池。
        """
        if not self.enabled:
            return 0
        items = list(items)
        n = len(items)
        if n == 0:
            self.rounds += 1
            return 0
        if self.loser_subset >= 1.0:
            chosen = list(range(n))
        else:
            k = max(1, math.ceil(self.loser_subset * n))
            g = torch.Generator().manual_seed(self.seed + self.rounds)
            chosen = torch.randperm(n, generator=g).tolist()[:k]
        cnt = 0
        for i in chosen:
            key, payload = items[i]
            latent = sample_fn(key, payload)
            if latent is not None:
                self.pool.set(key, latent)
                cnt += 1
        self.rounds += 1
        return cnt

    # ── 参考前向：.data 交换上下文 + EMA 更新 ─────────────────────────────────
    def reference_mode(self):
        """返回参考 .data 交换上下文管理器；未建参考时为 no-op（nullcontext）。

        用法（调用方在 torch.no_grad() 下）：
            with dpo.reference_mode():
                l_ref_w = per_sample_loss(...);  l_ref_l = per_sample_loss(...)
        """
        if self.reference is None:
            return contextlib.nullcontext()
        return self.reference.swap()

    def update_reference_ema(self) -> None:
        if self.reference is not None:
            self.reference.update_ema()

    # ── 决策核包装：算 L_DPO（+ 可选 winner SFT 锚）+ 诊断 ────────────────────
    def dpo_loss(self, l_theta_w, l_theta_l, l_ref_w, l_ref_l):
        """算 L_DPO；sft_anchor_lambda>0 时叠加 winner 普通 SFT 项（漂移兜底）。返回 (loss, info)。"""
        loss, info = linear_dpo_loss(
            l_theta_w, l_theta_l, l_ref_w, l_ref_l, beta=self.beta, eta=self.eta,
        )
        if self.sft_anchor_lambda > 0.0:
            loss = loss + self.sft_anchor_lambda * l_theta_w.mean()
        self.last_info = info
        self.steps_done += 1
        if self.log_path:
            self._log_rows.append((
                self.steps_done,
                float(info["omega"].float().mean().item()),
                float(info["delta"].float().mean().item()),
                float(info["margin"].float().mean().item()),
                float(info["d_w"].float().mean().item()),
                float(info["d_l"].float().mean().item()),
            ))
            if self.steps_done % 25 == 0:
                self.dump()
        return loss, info

    def dump(self) -> None:
        if not self.log_path or not self._log_rows:
            return
        d = os.path.dirname(self.log_path)
        if d:
            os.makedirs(d, exist_ok=True)
        tmp = self.log_path + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            wri = csv.writer(f)
            wri.writerow(["step", "omega", "delta", "margin", "d_w", "d_l"])
            wri.writerows(self._log_rows)
        os.replace(tmp, self.log_path)

    def summary(self) -> str:
        if not self.enabled:
            return "dpo: disabled"
        ema = self.reference.ema if self.reference is not None else 1.0
        return (f"dpo: rounds={self.rounds} steps={self.steps_done} pool={len(self.pool)} "
                f"beta={self.beta} eta={self.eta} ref_ema={ema} "
                f"loser(steps={self.loser_steps},cfg={self.loser_cfg},subset={self.loser_subset})")
