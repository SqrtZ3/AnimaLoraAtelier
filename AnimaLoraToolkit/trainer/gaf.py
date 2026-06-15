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


# ============================================================================
# 抽取后端 ②：ghost 随机投影 sketch（~1×，TRAK 式；在主 backward 内一次完成）
# ============================================================================

def _proj_matrix(dim: int, k: int, role: str, cache, seed: int):
    """按 (role, dim, k) 缓存的单个随机投影矩阵 [dim, k]（CPU 种子生成，用时搬设备）。
    a/g 独立投影 → 前向只需 in_dim、反向只需 out_dim，互不依赖；便于"前向就把 a 投影掉"省显存。"""
    key = (role, int(dim), int(k))
    if key not in cache:
        g = torch.Generator(device="cpu").manual_seed(
            int(seed) + (1009 if role == "g" else 0) + dim * 131 + k)
        cache[key] = torch.randn(dim, k, generator=g) * (1.0 / float(k) ** 0.5)
    return cache[key]


class GafGhostHooks:
    """在注入的 LoRALinear 边界捕获每样本 (输入 a, 输出梯度 g)，随机投影成 sketch
    s_i = Σ_t (Pg·g_i,t) ⊗ (Pa·a_i,t) ∈ R^{k×k}，拼接各层 → 每样本"功能梯度"方向向量。

    参数化无关（不碰 LoKr 因子 / DoRA；a/g 是真实激活与反向梯度，天然含 DoRA+dropout）；
    cosine 对每样本 trust 缩放不变（trust 只缩 g 的幅度不改方向）→ 可在 trust 加权的主
    backward 内捕获 → 一次 backward 完成 = ~1×。active=False 时 hook 立即返回（非 GAF 步零开销）。
    """

    def __init__(self, batch_size: int, proj_dim: int = 16, seed: int = 1234):
        self.bs = int(batch_size)
        self.k = max(int(proj_dim), 2)
        self.seed = int(seed)
        self._cache: dict = {}
        self._stash: dict = {}     # module -> 输入 a
        self._sketch: dict = {}    # module -> [B, k*k]
        self.active = False
        self.handles = []

    def register(self, modules) -> int:
        n = 0
        for m in modules:
            if not hasattr(m, "register_full_backward_hook"):
                continue
            self.handles.append(m.register_forward_hook(self._fwd))
            self.handles.append(m.register_full_backward_hook(self._bwd))
            n += 1
        return n

    def _fwd(self, module, inp, out):
        # ★ 关键省显存：前向就把输入投影成小的 pa=[B,T,k] 存下来，**不持有完整激活 a**
        #   （否则 hook 会 hold 住所有层输入、抵消 grad_checkpoint，1024 下必 OOM）。
        if not self.active:
            return
        a = inp[0] if isinstance(inp, (tuple, list)) else inp
        if not torch.is_tensor(a) or a.shape[0] != self.bs:
            return
        a = a.detach().reshape(self.bs, -1, a.shape[-1]).float()
        Pa = _proj_matrix(a.shape[-1], self.k, "a", self._cache, self.seed).to(a.device, torch.float32)
        # 外层 autocast 会把 matmul 降 bf16 → 与反向(无 autocast)的 float32 pg 撞 dtype；
        # 显式关 autocast 让投影恒 float32。
        with torch.autocast(device_type=a.device.type, enabled=False):
            self._stash[module] = (a @ Pa).float()    # [B, T, k]，float32，小

    def _bwd(self, module, grad_input, grad_output):
        if not self.active:
            return
        pa = self._stash.pop(module, None)
        g = grad_output[0] if isinstance(grad_output, (tuple, list)) else grad_output
        if pa is None or not torch.is_tensor(g) or g.shape[0] != self.bs:
            return
        g = g.detach().reshape(self.bs, -1, g.shape[-1]).float()
        if g.shape[1] != pa.shape[1]:                 # token 数不一致（异常路径）→ 跳过
            return
        Pg = _proj_matrix(g.shape[-1], self.k, "g", self._cache, self.seed).to(g.device, torch.float32)
        pg = (g @ Pg).float()                         # [B, T, k]，float32
        s = torch.einsum("bti,btj->bij", pg, pa.float()).reshape(self.bs, -1)  # [B, k*k]
        self._sketch[module] = s.detach()

    def begin(self) -> None:
        self._stash.clear()
        self._sketch.clear()
        self.active = True

    def collect(self):
        """返回 [B, D] 拼接 sketch（固定 id 顺序）；无捕获返回 None。同时关闭 active。"""
        self.active = False
        if not self._sketch:
            return None
        mods = sorted(self._sketch.keys(), key=id)
        out = torch.cat([self._sketch[m] for m in mods], dim=1)
        self._stash.clear()
        self._sketch.clear()
        return out

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


class GafController:
    """GAF 全流程编排（默认关；enabled=False 时所有方法即时返回，零影响）。

    两个抽取后端：
      - backend="autograd"（稳，~3×/GAF步）：GAF 步对每样本 loss 逐个 autograd.grad，精确；
        靠 `every` 周期摊销。需在主 backward 前调 update_autograd。
      - backend="ghost"（~1×，TRAK 投影 sketch）：在主 backward 内 hook 捕获 (a,g) 投影成
        方向 sketch，参数化无关、近似但 cosine 够用。before_forward 开捕获、after_backward 收集。

    两后端共用：每步把信任当逐样本 loss 乘子施加（weight_for_batch，便宜）；warmup 内不介入。
    """

    def __init__(self, *, backend="autograd", params=None, modules=None,
                 batch_size=1, proj_dim=16, enabled=False, every=4, warmup=100,
                 mode="soft", threshold=0.0, temp=0.15, floor=0.3, min_keep=2,
                 trust_decay=0.9, log_path="", proj_seed=1234):
        self.backend = str(backend or "autograd").lower()
        self.params = [p for p in (params or []) if getattr(p, "requires_grad", False)]
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
        self.hooks = None
        self.n_hooked = 0

        self.enabled = bool(enabled)
        if self.backend == "autograd":
            self.enabled = self.enabled and len(self.params) > 0
        elif self.backend == "ghost":
            if self.enabled:
                self.hooks = GafGhostHooks(batch_size, proj_dim, seed=proj_seed)
                self.n_hooked = self.hooks.register(list(modules or []))
                self.enabled = self.enabled and self.n_hooked > 0

    def should_run(self, step: int) -> bool:
        return self.enabled and step >= self.warmup and (int(step) % self.every == 0)

    def _score(self, matrix, keys) -> None:
        if matrix is None or keys is None or not bool(torch.isfinite(matrix).all()):
            return
        _, info = gaf_weights(matrix, mode=self.mode, threshold=self.threshold,
                              temp=self.temp, floor=self.floor, min_keep=self.min_keep)
        self.state.update(keys, info["weights"], info["accepted"])
        self.last_info = info
        self.runs += 1
        if self.log_path and self.runs % 25 == 0:
            self.state.dump(self.log_path)

    # ── ghost 后端：主 forward 前开捕获、主 backward 后收集 ──
    def before_forward(self, step: int) -> None:
        if self.backend != "ghost" or self.hooks is None:
            return
        if self.should_run(step):
            self.hooks.begin()
        else:
            self.hooks.active = False   # 非 GAF 步显式关闭，避免上一步 NaN-skip 残留 active

    def after_backward(self, keys, step: int) -> None:
        if self.backend == "ghost" and self.hooks is not None and self.should_run(step):
            self._score(self.hooks.collect(), keys)

    # ── autograd 后端：主 backward 前，对每样本 loss 抽梯度 ──
    def update_autograd(self, per_sample_loss: torch.Tensor, keys, step: int) -> None:
        if self.backend != "autograd" or not self.should_run(step) or keys is None:
            return
        if not bool(torch.isfinite(per_sample_loss).all()):
            return
        self._score(extract_per_sample_grads(per_sample_loss, self.params), keys)

    def weight_for_batch(self, keys, *, device, dtype) -> "torch.Tensor | None":
        """每步：从信任 EMA 取本 batch 的逐样本乘子（未建档样本=1.0=不动）。"""
        if not self.enabled or keys is None:
            return None
        w = [float(self.state.trust.get(str(k), 1.0)) for k in keys]
        return torch.tensor(w, device=device, dtype=dtype)

    def dump(self) -> None:
        if self.enabled and self.log_path:
            self.state.dump(self.log_path)

    def remove_hooks(self) -> None:
        if self.hooks is not None:
            self.hooks.remove()

    def summary(self) -> str:
        if not self.enabled:
            return "gaf: disabled"
        n = len(self.state.trust)
        lo = min(self.state.trust.values()) if n else 1.0
        extra = f" backend={self.backend}" + (f" hooked={self.n_hooked}" if self.hooks else "")
        return f"gaf: {n} 图已建档, GAF评估 {self.runs} 次, 最低信任={lo:.3f}{extra}"
