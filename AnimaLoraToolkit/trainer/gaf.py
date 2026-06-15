"""GAF —— Gradient Agreement Filtering（脏数据鲁棒性 B1）。

核心思想：按梯度**方向**（不按 loss 大小）判别"与 batch 共识方向不合"的样本。
高细节干净样本梯度**大但方向对** → 保留；脏 / 离群样本方向**歪** → 降权 / 剔除。
因此它逃开了"脏≈难细节"那个把 loss 加权法搞死的混淆，且与 Huber-SNR / adaptive-t 正交。

本模块只负责**决策核**：给定每样本展平梯度向量 → 每样本信任权重 / 接受掩码 +
逐样本拒绝统计。它与"如何抽取每样本梯度"（torch.func / ghost / 微批）**解耦**——
抽取后端是可替换的，决策核可独立单测。这样实现风险被隔离在抽取那一段。

约定：
- 输入 grads 形状 [B, P]（B 个样本，各自展平的可训练 LoRA 梯度），已 detach。
- 共识用**留一均值**（leave-one-out），避免样本自指；方向比较用 cosine（幅度无关
  = GAF 的灵魂：大但对齐的细节梯度不被罚）。
- soft 模式输出 (0,1] 信任权重，作为逐样本 loss 乘子施加（detached → 等价于按信任
  缩放该样本的梯度贡献，机制仍是"方向"驱动）；hard 模式输出接受掩码。
- min_keep 软下限：一个 batch 永不接受少于 min_keep 个样本，保护"合法但稀有"的干净图
  （它可能与共识不合，但不该被一票否决）。
"""

from __future__ import annotations

import torch

_EPS = 1e-12


def leave_one_out_agreement(grads: torch.Tensor) -> torch.Tensor:
    """每样本梯度与"其余样本均值方向"的 cosine。grads:[B,P] → [B]。

    幅度无关：cos 只看方向，所以"梯度大但对齐"的难细节样本 ≈ +1（保留），
    "方向相反"的脏样本 ≈ -1（降权）。
    """
    grads = grads.float()
    bs = grads.shape[0]
    if bs < 2:
        return grads.new_ones(bs)
    sum_g = grads.sum(0, keepdim=True)               # [1,P]
    loo = (sum_g - grads) / float(bs - 1)            # [B,P] 留一均值
    gn = grads.norm(dim=1).clamp_min(_EPS)
    ln = loo.norm(dim=1).clamp_min(_EPS)
    return (grads * loo).sum(1) / (gn * ln)          # [B] ∈ [-1,1]


def gaf_weights(grads: torch.Tensor, *, mode: str = "soft",
                threshold: float = 0.0, temp: float = 0.15,
                floor: float = 0.0, min_keep: int = 2):
    """决策核：每样本梯度 → 信任权重 + 诊断。

    mode="soft": w_i = floor + (1-floor)·sigmoid((cos_i - threshold)/temp)，∈[floor,1]。
                 平滑降权，方向越不合权重越低；threshold 是"中性点"（cos=threshold→0.5）。
    mode="hard": 接受 cos_i ≥ threshold 的样本（w=1），其余 w=floor；再用 min_keep 兜底
                 （cos 最高的若干个强制保留），避免一个 batch 几乎全被拒。

    返回 (weights[B], info)。info 含 cos、accepted 布尔、rejected 数，供逐图统计/日志。
    """
    cos = leave_one_out_agreement(grads)
    bs = cos.shape[0]
    floor = float(min(max(floor, 0.0), 1.0))

    if mode == "hard":
        accept = cos >= float(threshold)
        # min_keep 兜底：至少保留 cos 最高的 min_keep 个
        k = int(min(max(min_keep, 0), bs))
        if k > 0 and int(accept.sum()) < k:
            topk = torch.topk(cos, k).indices
            accept = torch.zeros_like(accept)
            accept[topk] = True
        weights = torch.where(accept, torch.ones_like(cos), cos.new_full((), floor))
    else:  # soft
        weights = floor + (1.0 - floor) * torch.sigmoid((cos - float(threshold)) / max(float(temp), 1e-4))
        # 软模式也保 min_keep：把 cos 最高的 min_keep 个权重抬回 1，确保共识核心不被稀释
        k = int(min(max(min_keep, 0), bs))
        if k > 0:
            topk = torch.topk(cos, k).indices
            weights = weights.clone()
            weights[topk] = 1.0
        accept = weights >= 0.5

    info = {
        "cos": cos.detach(),
        "weights": weights.detach(),
        "accepted": accept.detach(),
        "n_rejected": int((~accept).sum().item()),
    }
    return weights.detach(), info


class GafTrustState:
    """逐图信任 EMA + 拒绝频率统计（GAF 自带的"可疑图"提名表，免费）。

    键 = 图片路径串（来自 batch["images"]，已在内存，不读图 → 对 NSFW 安全）。
    trust_i = 该图历次 GAF 权重的 EMA；长期低 = 方向常年与共识不合 = 疑似脏/离群。
    """

    def __init__(self, decay: float = 0.9):
        self.decay = float(min(max(decay, 0.0), 0.9999)
                           )
        self.trust: dict[str, float] = {}
        self.seen: dict[str, int] = {}
        self.rejected: dict[str, int] = {}

    def update(self, keys, weights, accepted) -> None:
        if keys is None:
            return
        w = weights.detach().float().cpu().tolist()
        acc = accepted.detach().bool().cpu().tolist()
        for i, key in enumerate(keys):
            key = str(key)
            wi = float(w[i]) if i < len(w) else 1.0
            self.trust[key] = wi if key not in self.trust else (
                self.decay * self.trust[key] + (1.0 - self.decay) * wi)
            self.seen[key] = self.seen.get(key, 0) + 1
            if i < len(acc) and not acc[i]:
                self.rejected[key] = self.rejected.get(key, 0) + 1

    def dump(self, out_path: str) -> None:
        if not out_path or not self.trust:
            return
        import os, csv
        rows = sorted(self.trust.items(), key=lambda kv: kv[1])  # 低信任在前
        d = os.path.dirname(out_path)
        if d:
            os.makedirs(d, exist_ok=True)
        tmp = out_path + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            wri = csv.writer(f)
            wri.writerow(["rank", "trust", "seen", "rejected", "image"])
            for r, (key, tr) in enumerate(rows):
                wri.writerow([r, f"{tr:.4f}", self.seen.get(key, 0),
                              self.rejected.get(key, 0), key])
        os.replace(tmp, out_path)


# ============================================================================
# 抽取后端：每样本梯度（torch.autograd.grad 逐样本，retain_graph）
# ============================================================================

def extract_per_sample_grads(per_sample_loss: torch.Tensor, params) -> torch.Tensor:
    """对每样本 loss 逐个回传，抽出每样本在 `params` 上的展平梯度。返回 [B, P]。

    用 autograd.grad（不污染 .grad，对主 backward 零副作用）+ retain_graph，让后续主
    backward 仍可正常进行。autograd 自动处理 LoKr/DoRA 的因子链，**天然兼容 grad_checkpoint**
    （每次回传会按 checkpoint 语义重算 forward），无需为变体手写每样本公式。
    成本=B 次回传（GAF 步才付，靠 GafController.every 周期摊销）。
    params 应为可训练的 LoRA 参数（base 冻结）。
    """
    bs = per_sample_loss.shape[0]
    params = list(params)
    rows = []
    for i in range(bs):
        gi = torch.autograd.grad(
            per_sample_loss[i], params,
            retain_graph=True, allow_unused=True, create_graph=False,
        )
        flat = torch.cat([
            (g if g is not None else torch.zeros_like(p)).reshape(-1).float()
            for g, p in zip(gi, params)
        ])
        rows.append(flat)
    return torch.stack(rows, dim=0)  # [B, P]


class GafController:
    """GAF 全流程编排（默认关；enabled=False 时所有方法即时返回，零影响）。

    周期摊销：每 `every` 步做一次"逐样本梯度→方向信任"评估（昂贵的 B 次回传），更新逐图
    信任 EMA；**每步**都把信任当逐样本 loss 乘子施加（便宜）。GAF 步可同步刷新+施加（不延迟）。
    warmup 内不介入，让训练先稳定。所有 kernel 参数透传给 gaf_weights。
    """

    def __init__(self, params, *, enabled=False, every=4, warmup=100,
                 mode="soft", threshold=0.0, temp=0.15, floor=0.3, min_keep=2,
                 trust_decay=0.9, log_path=""):
        self.params = [p for p in params if getattr(p, "requires_grad", False)]
        self.enabled = bool(enabled) and len(self.params) > 0
        self.every = max(int(every or 1), 1)
        self.warmup = max(int(warmup or 0), 0)
        self.mode = str(mode or "soft")
        self.threshold = float(threshold)
        self.temp = float(temp)
        self.floor = float(floor)
        self.min_keep = int(min_keep)
        self.log_path = str(log_path or "")
        self.state = GafTrustState(decay=trust_decay)
        self.runs = 0
        self.last_info = None

    def should_run(self, step: int) -> bool:
        return self.enabled and step >= self.warmup and (int(step) % self.every == 0)

    def update_trust(self, per_sample_loss: torch.Tensor, keys, step: int) -> None:
        """GAF 步：抽每样本梯度→算方向信任→更新逐图 EMA。需在主 backward 前、graph 完好时调。"""
        if not self.enabled or keys is None:
            return
        if not bool(torch.isfinite(per_sample_loss).all()):
            return
        grads = extract_per_sample_grads(per_sample_loss, self.params)
        _, info = gaf_weights(
            grads, mode=self.mode, threshold=self.threshold, temp=self.temp,
            floor=self.floor, min_keep=self.min_keep,
        )
        self.state.update(keys, info["weights"], info["accepted"])
        self.last_info = info
        self.runs += 1
        if self.log_path and self.runs % 25 == 0:
            self.state.dump(self.log_path)

    def weight_for_batch(self, keys, *, device, dtype) -> "torch.Tensor | None":
        """每步：从信任 EMA 取本 batch 的逐样本乘子（未建档样本=1.0=不动）。"""
        if not self.enabled or keys is None:
            return None
        w = [float(self.state.trust.get(str(k), 1.0)) for k in keys]
        return torch.tensor(w, device=device, dtype=dtype)

    def dump(self) -> None:
        if self.enabled and self.log_path:
            self.state.dump(self.log_path)

    def summary(self) -> str:
        if not self.enabled:
            return "gaf: disabled"
        n = len(self.state.trust)
        lo = min(self.state.trust.values()) if n else 1.0
        return f"gaf: {n} 图已建档, GAF评估 {self.runs} 次, 最低信任={lo:.3f}"
