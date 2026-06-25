"""训练内遥测总线（image-blind, opt-in, default-off）。

服务 [[feedback-training-observability]]：把训练中"实际表达了什么"变成逐步可读的
数据，用内部信号反调参，而不是烧 run 猜预览。所有探针共享同一套纪律：

  * **图盲**：只碰 latent 空间的 error 张量 / 权重 / 优化器 state，绝不解码、不外传
    （[[feedback-image-blind-nsfw]]，NSFW 数据集）。
  * **复用**：骑现有 forward / 现有优化器 state，无新模型、无额外显存。
  * **≈0 边际成本**：norms / 小 eigvalsh / 稀疏 SVD，按 cadence 触发。
  * **闭环可执行**：每个探针回答一个否则要烧 run 才知道的调参问题。

探针（与产出 CSV）：
  L1 `radial_band_power`  —— eval velocity 残差按径向频带分解 → eval_freq_loss.csv
                            （高t/高频预算值不值、纹理 vs 结构谁先饱、CSFlow 前提）
  L2 `FreqEvalProbe` slope —— 逐 t-bin loss 的 Δ/step → eval_slope.csv（学饱 vs 还在学）
  O1 `optimizer_report` sf_lag —— ‖x−z‖/‖z‖（soap_sf Polyak 平均是否滞后 → run 够长吗）
  O2 `optimizer_report` kappa  —— GG 曲率谱 κ / 有效秩（二阶预条件值不值 / 还是 ≈Adam）
  O3 `optimizer_report` trust  —— 逐 block ‖update‖/‖grad‖ + grad norm（哪些 block 被饿）
  C1 `lokr_capacity_report`    —— 逐 block ‖ΔW‖ + LoKr ΔW 有效秩 → telemetry_capacity.csv
                                  （哪些 block 吃满/浪费 rank → 反调 reg_dims）

依赖：仅 torch + 标准库（云端 venv 无 scipy/numpy 保证，见 [[environment-reference]]）。
所有数值后处理在 fp32 CPU/GPU 上做，输出 python float。
"""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Iterable, Optional

import torch

_EPS = 1e-12


# ────────────────────────────── 通用小工具 ──────────────────────────────

def _block_of(name: str) -> str:
    """从参数/模块名解析 transformer block id（'blocks.14.xxx' → '14'）。

    非 block 模块（final_layer / x_embedder / adaln 等）归到 'other'。仅用于把
    逐参数指标按 block 聚合，便于对位 [[block-localization-goutong10]] 的 7-20/21-27 带。
    """
    if not name:
        return "other"
    idx = name.find("blocks.")
    if idx < 0:
        return "other"
    rest = name[idx + len("blocks."):].split(".", 1)[0]
    return rest if rest.isdigit() else "other"


def _percentile(values: list[float], q: float) -> float:
    """无 numpy 的简单分位数（线性插值），q∈[0,1]。空列表返回 nan。"""
    xs = sorted(v for v in values if v is not None and math.isfinite(v))
    if not xs:
        return float("nan")
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def _linfit_slope(steps: list[float], values: list[float]) -> float:
    """最小二乘斜率 Δvalue/Δstep。<2 点或退化时返回 0。"""
    n = len(steps)
    if n < 2:
        return 0.0
    sx = sum(steps)
    sy = sum(values)
    sxx = sum(s * s for s in steps)
    sxy = sum(s * v for s, v in zip(steps, values))
    denom = n * sxx - sx * sx
    if abs(denom) < _EPS:
        return 0.0
    return (n * sxy - sx * sy) / denom


def _append_csv(path: Path, header: list[str], row: list) -> None:
    """追加一行；文件不存在时先写 header。值为 None/nan 写空串。"""
    path = Path(path)
    write_header = not path.exists()
    with open(path, "a", encoding="utf-8") as f:
        if write_header:
            f.write(",".join(header) + "\n")

        def _fmt(v):
            if v is None:
                return ""
            if isinstance(v, float):
                return "" if not math.isfinite(v) else f"{v:.6g}"
            return str(v)

        f.write(",".join(_fmt(v) for v in row) + "\n")


# ───────────────────── L1：径向频带残差分解 ─────────────────────

@torch.no_grad()
def radial_band_power(err: torch.Tensor, n_bands: int = 3) -> list[float]:
    """把残差张量 err 按空间频率径向分成 n_bands 个频带，返回每带平均功率。

    err: (..., H, W) 实张量（典型 = velocity 预测误差 pred−target，shape (B,C,H,W)）。
    步骤：fft2 → fftshift → |.|² → 按到中心(DC)的归一化半径分桶（n_bands 等宽）→
    对所有前导维 + 带内像素取均值。返回长度 n_bands 的 python float 列表（低频→高频）。

    图盲：只吃 latent 空间 error 张量。⚠ latent 频率轴 ≠ 像素 cycles（VAE 8× 下采样），
    仅作**相对趋势**追踪，不做绝对（见 [[training-telemetry-spec]]）。
    """
    if err.dim() < 2:
        raise ValueError("radial_band_power 需要至少 2 维 (..., H, W)")
    if n_bands < 1:
        raise ValueError("n_bands 必须 >= 1")
    x = err.detach().float()
    H, W = x.shape[-2], x.shape[-1]
    fft = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1), norm="ortho"), dim=(-2, -1))
    power = fft.real.square() + fft.imag.square()        # (..., H, W)
    power = power.reshape(-1, H, W)                       # (N, H, W)

    cy, cx = H // 2, W // 2
    ys = torch.arange(H, device=x.device).view(H, 1).float() - cy
    xs = torch.arange(W, device=x.device).view(1, W).float() - cx
    r = torch.sqrt(ys * ys + xs * xs)
    rmax = float(r.max().clamp(min=_EPS))
    band = torch.clamp((r / rmax * n_bands).long(), max=n_bands - 1)   # (H, W)

    out: list[float] = []
    for b in range(n_bands):
        mask = band == b
        out.append(float(power[:, mask].mean()) if bool(mask.any()) else 0.0)
    return out


class FreqEvalProbe:
    """L1 + L2：挂在确定性 eval 循环上的频带残差 + 逐 t-bin 饱和斜率探针。

    每次 eval 调 `record(step, per_t_bands, per_t_loss)`：
      * 写 eval_freq_loss.csv：step + 每个 (t-bin × 频带) 的平均残差功率（宽表）。
      * 维护每个 t-bin 最近 `slope_window` 个 (step, loss)，最小二乘斜率写
        eval_slope.csv：step + 每个 t-bin 的 Δloss/step（负=还在降/未饱；≈0=学饱）。

    与 eval_loss.csv（既有逐 t-bin 绝对 loss）互补：那张是绝对值、这两张是频带分解 + 趋势。
    """

    def __init__(self, output_dir, t_grid: list[float], n_bands: int = 3,
                 slope_window: int = 6):
        self.output_dir = Path(output_dir)
        self.t_grid = list(t_grid)
        self.n_bands = int(n_bands)
        self.slope_window = max(int(slope_window), 2)
        self._hist = [deque(maxlen=self.slope_window) for _ in self.t_grid]
        self._band_names = self._make_band_names()

    def _make_band_names(self) -> list[str]:
        if self.n_bands == 3:
            return ["fL", "fM", "fH"]
        return [f"f{i}" for i in range(self.n_bands)]

    def freq_header(self) -> list[str]:
        cols = ["step"]
        for tv in self.t_grid:
            for bn in self._band_names:
                cols.append(f"t{tv:g}_{bn}")
        return cols

    def slope_header(self) -> list[str]:
        return ["step"] + [f"t{tv:g}_slope" for tv in self.t_grid]

    def record(self, step: int, per_t_bands: list[list[float]],
               per_t_loss: Optional[list[float]] = None) -> None:
        """per_t_bands[j] = t_grid[j] 的 n_bands 个频带功率；per_t_loss[j] = 该 t 的标量 loss。"""
        freq_row: list = [step]
        for bands in per_t_bands:
            freq_row.extend(bands)
        _append_csv(self.output_dir / "eval_freq_loss.csv", self.freq_header(), freq_row)

        if per_t_loss is not None:
            slope_row: list = [step]
            for j, _ in enumerate(self.t_grid):
                self._hist[j].append((float(step), float(per_t_loss[j])))
                steps = [s for s, _ in self._hist[j]]
                vals = [v for _, v in self._hist[j]]
                slope_row.append(_linfit_slope(steps, vals))
            _append_csv(self.output_dir / "eval_slope.csv", self.slope_header(), slope_row)


# ───────────────────── O1/O2/O3：优化器内部信号 ─────────────────────

@torch.no_grad()
def gg_anisotropy(gg_list: Iterable[Optional[torch.Tensor]]) -> tuple[float, float]:
    """从 SOAP 的 GG（Shampoo 协方差矩阵列表）算曲率谱指标。

    返回 (kappa, eff_rank_ratio)：
      * kappa = λmax/λmin，取各维 GG 里最差（最大）的——κ≫1 = 曲率高度各向异性，
        二阶预条件买到真东西；κ≈1 = 近各向同性，soap_sf ≈ Adam，Shampoo 机器白背。
      * eff_rank_ratio = 谱熵有效秩 / 维数（各维取均值）——曲率集中在少数方向(小)
        vs 摊平(≈1) 的 bulk 视角，与 κ 的极端比互补。
    无可用 GG 维时返回 (nan, nan)。
    """
    kappas: list[float] = []
    ratios: list[float] = []
    for gg in gg_list:
        if gg is None:
            continue
        ev = torch.linalg.eigvalsh(gg.float()).clamp(min=0.0)
        lam_max = float(ev.max())
        if lam_max <= _EPS:
            continue
        pos = ev[ev > _EPS * lam_max]
        lam_min = float(pos.min()) if pos.numel() > 0 else _EPS * lam_max
        kappas.append(lam_max / max(lam_min, _EPS))
        p = ev / ev.sum().clamp(min=_EPS)
        ent = float(-(p * (p.clamp(min=_EPS)).log()).sum())
        ratios.append(math.exp(ent) / ev.numel())
    if not kappas:
        return float("nan"), float("nan")
    return max(kappas), sum(ratios) / len(ratios)


@torch.no_grad()
def optimizer_report(optimizer, output_dir, step: int,
                     named_params: Optional[list] = None,
                     anisotropy: bool = True) -> dict:
    """读优化器 state（图盲：只碰权重/state 张量），逐 block 聚合后写
    telemetry_optimizer.csv（长表：step, scope, grad_norm, update_norm, trust_ratio,
    sf_lag, kappa, eff_rank_ratio）。scope = block id 或 'all'（全局分位汇总）。

    SOAPScheduleFree-aware：sf_lag 来自 state['z'] + 当前 param(=y)；update_norm 来自
    优化器在 _telemetry=True 时 stash 的 state['_upd_norm']；grad_norm 来自 param.grad
    （须在 zero_grad 之前调用）。其它优化器缺哪个信号就留空，不报错。
    返回全局聚合 dict 便于同时打 console。
    """
    name_of = {}
    if named_params is not None:
        name_of = {id(p): n for n, p in named_params}

    # 逐 block 累积
    per_block: dict[str, dict[str, list[float]]] = {}

    def _slot(block: str) -> dict[str, list[float]]:
        return per_block.setdefault(block, {
            "grad_norm": [], "update_norm": [], "trust_ratio": [],
            "sf_lag": [], "kappa": [], "eff_rank_ratio": [],
        })

    for group in optimizer.param_groups:
        betas = group.get("betas", None)
        beta1 = float(betas[0]) if betas else None
        for p in group["params"]:
            st = optimizer.state.get(p, {})
            if not st and p.grad is None:
                continue
            block = _block_of(name_of.get(id(p), ""))
            slot = _slot(block)

            gnorm = float(p.grad.detach().float().norm()) if p.grad is not None else None
            if gnorm is not None:
                slot["grad_norm"].append(gnorm)

            unorm = st.get("_upd_norm", None)
            if unorm is not None:
                unorm = float(unorm)
                slot["update_norm"].append(unorm)
                if gnorm is not None and gnorm > _EPS:
                    slot["trust_ratio"].append(unorm / gnorm)

            z = st.get("z", None)
            if z is not None and beta1 is not None:
                y = p.detach().float()
                zf = z.float()
                # x = Polyak/eval 迭代（SOAPScheduleFree.eval: x = y + (1−1/β1)(z−y)）
                x = y + (1.0 - 1.0 / beta1) * (zf - y)
                znorm = float(zf.norm())
                if znorm > _EPS:
                    slot["sf_lag"].append(float((zf - x).norm()) / znorm)

            if anisotropy and "GG" in st:
                k, rr = gg_anisotropy(st["GG"])
                if math.isfinite(k):
                    slot["kappa"].append(k)
                if math.isfinite(rr):
                    slot["eff_rank_ratio"].append(rr)

    metrics = ["grad_norm", "update_norm", "trust_ratio", "sf_lag", "kappa", "eff_rank_ratio"]
    header = ["step", "scope"] + metrics
    csv_path = Path(output_dir) / "telemetry_optimizer.csv"

    # 逐 block 行（中位数代表该 block）
    def _med(xs):
        return _percentile(xs, 0.5) if xs else None

    for block in sorted(per_block, key=lambda b: (b == "other", int(b) if b.isdigit() else 1e9)):
        slot = per_block[block]
        _append_csv(csv_path, header, [step, block] + [_med(slot[m]) for m in metrics])

    # 全局聚合行（'all'）：跨所有参数的中位 + p90（取代单 block 中位，给整体画像）
    flat = {m: [v for slot in per_block.values() for v in slot[m]] for m in metrics}
    agg = {m: _med(flat[m]) for m in metrics}
    _append_csv(csv_path, header, [step, "all"] + [agg[m] for m in metrics])

    agg["sf_lag_p90"] = _percentile(flat["sf_lag"], 0.9)
    agg["kappa_p90"] = _percentile(flat["kappa"], 0.9)
    return agg


# ───────────────────── C1：逐 block LoKr 容量 ─────────────────────

@torch.no_grad()
def lokr_spectrum(w1: torch.Tensor, w2_a: torch.Tensor, w2_b: torch.Tensor,
                  scaling: float = 1.0) -> torch.Tensor:
    """LoKr ΔW = scaling·kron(w1, w2_a @ w2_b) 的完整奇异谱——不实例化 ΔW。

    用 Kronecker 谱恒等式 σ(A⊗B) = {σ_i(A)·σ_j(B)}：只对小因子 w1、w2(=w2_a@w2_b)
    各做一次 SVD，外积得全谱。复杂度 O(factor³ + out·rank²)，远小于形成
    (out·factor)×(in·factor) 的 ΔW。返回降序奇异值张量（长度 factor·rank）。
    """
    w1f = w1.detach().float()
    w2 = w2_a.detach().float() @ w2_b.detach().float()
    s1 = torch.linalg.svdvals(w1f)
    s2 = torch.linalg.svdvals(w2)
    sigma = torch.outer(s1, s2).reshape(-1) * float(scaling)
    return torch.sort(sigma, descending=True).values


@torch.no_grad()
def spectrum_stats(sigma: torch.Tensor) -> tuple[float, float, float]:
    """从奇异谱算 (frob, eff_rank, eff_rank_ratio)。

    frob = ‖ΔW‖_F = sqrt(Σσ²)；eff_rank = exp(谱熵)（基于 σ² 归一化的能量分布）；
    eff_rank_ratio = eff_rank / len(σ)。eff_rank≈满 = rank 吃满；≪满 = 浪费 rank。
    """
    sig = sigma.detach().float()
    energy = sig.square()
    tot = float(energy.sum())
    frob = math.sqrt(max(tot, 0.0))
    if tot <= _EPS:
        return frob, 0.0, 0.0
    p = energy / tot
    ent = float(-(p * (p.clamp(min=_EPS)).log()).sum())
    eff_rank = math.exp(ent)
    return frob, eff_rank, eff_rank / max(sig.numel(), 1)


@torch.no_grad()
def lokr_capacity_report(model, output_dir, step: int) -> dict:
    """遍历模型里的 LoKrLayer，逐 block 记 ‖ΔW‖ + 有效秩 → telemetry_capacity.csv
    （长表：step, block, module, frob, eff_rank, eff_rank_ratio）。

    图盲：只读 LoKr 因子权重。回答"哪些 block 吃满/浪费分配的 rank"→ 反调
    lora_reg_dims（[[block-localization-goutong10]]、[[run-v4-ushape-harvest]]）。
    返回 {block: 平均 eff_rank_ratio} 便于 console 摘要。
    """
    header = ["step", "block", "module", "frob", "eff_rank", "eff_rank_ratio"]
    csv_path = Path(output_dir) / "telemetry_capacity.csv"
    per_block_ratio: dict[str, list[float]] = {}

    for name, mod in model.named_modules():
        if not (hasattr(mod, "lokr_w1") and hasattr(mod, "lokr_w2_a")
                and hasattr(mod, "lokr_w2_b")):
            continue
        scaling = float(getattr(mod, "scaling", 1.0))
        sigma = lokr_spectrum(mod.lokr_w1, mod.lokr_w2_a, mod.lokr_w2_b, scaling=scaling)
        frob, eff_rank, ratio = spectrum_stats(sigma)
        block = _block_of(name)
        _append_csv(csv_path, header, [step, block, name, frob, eff_rank, ratio])
        per_block_ratio.setdefault(block, []).append(ratio)

    return {b: (sum(v) / len(v) if v else float("nan")) for b, v in per_block_ratio.items()}
