"""CSFlow —— 把 flow-matching 的 timestep 采样对齐人眼对比敏感度（arXiv 2606.08833）。

核心思路：自然图像在去噪中按"粗→细"分频段恢复，而人眼对**中频**最敏感（CSF 峰）。
CSFlow 用 数据集功率谱 S_f × 人眼 CSF(f) 推出一条**逐 t 的重要性权重 w_CSFlow(t)**，
据此偏置训练时的 t 采样分布——把模型容量分配到"感知上重要的频段恰好开始可恢复"的噪声级。

落到本项目的 flow-matching CONST 调度（x_t=(1-t)·x0+t·ε，velocity 目标）：
    a_t = 1-t（信号系数）, b_t = t（噪声系数）
    rsignal(f,t) = a_t²·S_f / (a_t²·S_f + b_t²)      # 频率 f 在噪声级 t 的"信号占比"
    Δrsignal(f,t) = rsignal(f,t) − rsignal(f,t+Δt)   # t 降一档时 f 新变得可恢复的量（≥0）
    w_CSFlow(t)   = Σ_f Δrsignal(f,t)·CSF(f) / Σ_f Δrsignal(f,t)   # 该档恢复频段的 CSF 加权均值
    w_final(t)    = α·w̃_CSFlow(t) + (1−α)·1          # 与均匀基底插值（α=1 纯 CSFlow）
然后对 w_final 的 CDF 做 inverse-transform 采样 t。**零额外前向**（只换采样分布）。

设计取舍（2026-06-25 立项决策，见 memory advanced-techniques-shortlist-v6）：
- S_f 在**像素域**算（tools/compute_rapsd.py 一次性离线）：CSF 是按 cycles/degree 定义的，
  latent 8× 下采样后频率轴对不上；像素域 S_f 让 CSF 轴对得上。
- 实验对照 = **替换三峰做单变量 A/B**（其余一致 + adaptive 关以隔离 t 轴）。

本模块全 CPU 可单测（纯张量数学 + 注入式 RAPSD）；不依赖 model / VAE / 任何 trainer 子模块。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def mannos_csf(f_cpd: torch.Tensor) -> torch.Tensor:
    """Mannos–Sakrison 对比敏感度函数。f_cpd = 空间频率（cycles per degree）。

    CSF(f) = 2.6·(0.0192 + 0.114·f)·exp(−(0.114·f)^1.1)
    峰约在 f≈8 cpd，低频/高频两端衰减——这正是"中频最敏感"的来源。f=0 处给一个很小正值。
    """
    f = f_cpd.clamp(min=0.0)
    return 2.6 * (0.0192 + 0.114 * f) * torch.exp(-torch.pow(0.114 * f, 1.1))


def build_csflow_weight_table(
    rapsd: torch.Tensor,
    *,
    pixels_per_degree: float = 50.0,
    alpha: float = 1.0,
    n_t: int = 1024,
    t_min: float = 1e-3,
    t_max: float = 1.0 - 1e-3,
    eps: float = 1e-12,
):
    """从径向功率谱 rapsd 构造 (t_grid, w_final, cdf)。

    rapsd: 1D 张量，长度 = 径向频率 bin 数；rapsd[k] = 归一化频率 f_k = k/(2·(len−1)) ∈ [0,0.5]
           cycles/pixel 处的平均功率 S_f（DC 分量建议调用方已剔除或置小）。
    pixels_per_degree: 像素→cycles/degree 映射（cpd = f_cyc_per_px · ppd）。控制 CSF 峰落在哪个
           归一化频率；默认 50 让 CSF 峰(~8cpd)≈ f_norm 0.16，对应"中频"。可在 config 调。
    alpha: 与均匀基底的插值；1.0 = 纯 CSFlow（A/B 推荐），0 = 退化成均匀。
    返回的 w_final 已归一到单位均值；cdf 单调升 [0,1]，供 inverse-transform 采样。
    """
    rapsd = rapsd.detach().float().flatten()
    n_f = rapsd.shape[0]
    # 归一化频率 [0, 0.5] cycles/pixel → cycles/degree
    f_cyc_px = torch.linspace(0.0, 0.5, n_f)
    csf = mannos_csf(f_cyc_px * float(pixels_per_degree))            # (n_f,)

    t_grid = torch.linspace(float(t_min), float(t_max), int(n_t))    # (n_t,)
    a = (1.0 - t_grid).clamp(min=0.0)                                # 信号系数
    b = t_grid.clamp(min=0.0)                                        # 噪声系数
    # rsignal(t,f) = a²S_f / (a²S_f + b²)  → (n_t, n_f)
    a2S = (a.pow(2).unsqueeze(1)) * rapsd.unsqueeze(0)
    b2 = b.pow(2).unsqueeze(1)
    rsignal = a2S / (a2S + b2 + eps)
    # Δrsignal(t,f) = rsignal(t) − rsignal(t+Δt)：t 降一档新恢复的频段（t_grid 升序→相邻差取负）
    d = rsignal[:-1] - rsignal[1:]                                   # (n_t-1, n_f) ≥ 0
    d = d.clamp(min=0.0)
    num = (d * csf.unsqueeze(0)).sum(dim=1)                          # (n_t-1,)
    den = d.sum(dim=1).clamp(min=eps)
    w_mid = num / den                                               # (n_t-1,) 各档的 CSF 加权均值
    # 对齐回 n_t（末档复制），归一到单位均值
    w = torch.cat([w_mid, w_mid[-1:]], dim=0)                       # (n_t,)
    w = w / w.mean().clamp(min=eps)
    a_mix = float(alpha)
    w_final = a_mix * w + (1.0 - a_mix) * torch.ones_like(w)
    w_final = w_final.clamp(min=eps)
    cdf = torch.cumsum(w_final, dim=0)
    cdf = cdf / cdf[-1].clamp(min=eps)
    return t_grid, w_final, cdf


def load_or_compute_rapsd(rapsd_path: str, *, data_dir: str | None = None,
                          res: int = 1024, max_images: int = 512) -> dict:
    """读 RAPSD 档；**不存在则从 data_dir 源图自动计算（像素域）并缓存**到 rapsd_path。

    去掉"必须先手跑 tools/compute_rapsd.py"的脚手架——trainer 本就读数据集，首跑时自己算一次、
    落盘复用，下次直接读。只有"既无档、data_dir 又无效"时才报错。
    图盲：只在训练进程内读源图、算聚合功率谱标量，不解码 latent、不外传（[[feedback-image-blind-nsfw]]）。
    """
    p = Path(rapsd_path) if rapsd_path else None
    if p is not None and p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    if not data_dir or not Path(data_dir).exists():
        raise FileNotFoundError(
            f"CSFlow 需要 RAPSD：档 {rapsd_path!r} 不存在，且无法自动计算（data_dir={data_dir!r} 无效）。"
        )
    try:
        from tools.compute_rapsd import compute_rapsd  # 延迟导入，避免 trainer 顶层依赖 tools
    except Exception as e:
        raise RuntimeError(
            f"CSFlow 自动计算 RAPSD 失败：导入 tools.compute_rapsd 出错（{e}）；"
            f"可手动跑 `python tools/compute_rapsd.py --data-dir {data_dir} --out {rapsd_path}`。"
        ) from e
    logger.info("[csflow] RAPSD 档不存在，自动从数据集计算（像素域，一次性，max_images=%d）：%s",
                int(max_images), data_dir)
    prof = compute_rapsd(Path(data_dir), int(res), int(max_images))
    if p is not None:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(prof, f, ensure_ascii=False, indent=1)
            logger.info("[csflow] RAPSD 已缓存到 %s（n_images=%d，下次直接读）",
                        rapsd_path, int(prof.get("n_images", 0)))
        except Exception as e:
            logger.warning("[csflow] RAPSD 缓存写盘失败（本次用内存版继续）：%s", e)
    return prof


class CSFlowSampler:
    """从 CSFlow 权重表做 inverse-transform t 采样的小状态对象（替换三峰时用作 base 分布）。"""

    def __init__(self, t_grid: torch.Tensor, cdf: torch.Tensor,
                 t_min: float = 1e-4, t_max: float = 1.0 - 1e-4,
                 meta: dict | None = None):
        self.t_grid = t_grid.detach().float()
        self.cdf = cdf.detach().float()
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.meta = meta or {}

    @classmethod
    def from_profile(cls, rapsd_path: str, *, alpha: float = 1.0,
                     pixels_per_degree: float = 50.0,
                     t_min: float = 1e-4, t_max: float = 1.0 - 1e-4,
                     data_dir: str | None = None, res: int = 1024,
                     max_images: int = 512) -> "CSFlowSampler":
        """从 RAPSD 档构造采样器；**档不存在时从 data_dir 自动计算并缓存**（无需先手跑工具）。"""
        prof = load_or_compute_rapsd(rapsd_path, data_dir=data_dir, res=res, max_images=max_images)
        rapsd = torch.tensor(prof["rapsd"], dtype=torch.float32)
        t_grid, _w, cdf = build_csflow_weight_table(
            rapsd, pixels_per_degree=pixels_per_degree, alpha=alpha,
        )
        meta = {k: prof.get(k) for k in ("resolution", "n_images", "channels", "source")}
        meta.update({"alpha": alpha, "pixels_per_degree": pixels_per_degree})
        return cls(t_grid, cdf, t_min=t_min, t_max=t_max, meta=meta)

    def sample(self, bs: int, device) -> torch.Tensor:
        """采样 bs 个 t ∈ (t_min, t_max)，分布 ∝ w_final（inverse-transform on CDF）。"""
        u = torch.rand(bs, device="cpu")
        # searchsorted on cdf，再在所落 bin 内线性插值 t（避免阶梯量化）
        idx = torch.searchsorted(self.cdf, u.clamp(0.0, 1.0)).clamp(1, self.cdf.shape[0] - 1)
        c0 = self.cdf[idx - 1]
        c1 = self.cdf[idx]
        frac = ((u - c0) / (c1 - c0).clamp(min=1e-12)).clamp(0.0, 1.0)
        t0 = self.t_grid[idx - 1]
        t1 = self.t_grid[idx]
        t = (t0 + frac * (t1 - t0)).clamp(self.t_min, self.t_max)
        return t.to(device=device)

    def summary(self) -> str:
        return (f"CSFlow(alpha={self.meta.get('alpha')},ppd={self.meta.get('pixels_per_degree')},"
                f"res={self.meta.get('resolution')},n_img={self.meta.get('n_images')})")
