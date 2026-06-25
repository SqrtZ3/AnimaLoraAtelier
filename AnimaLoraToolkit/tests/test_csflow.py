"""CSFlow CPU 单测（无 model/VAE 依赖）：CSF 曲线、权重表、inverse-transform 采样、档案往返。"""

import json
import os
import tempfile

import torch

from trainer.csflow import (
    mannos_csf,
    build_csflow_weight_table,
    CSFlowSampler,
)


def test_mannos_csf_shape_and_peak():
    f = torch.linspace(0, 60, 200)
    csf = mannos_csf(f)
    assert csf.shape == f.shape
    assert torch.all(csf >= 0)
    # 峰应在中频（~8 cpd），不在 f=0 也不在 f=60
    peak = int(torch.argmax(csf))
    assert 0 < peak < len(f) - 1
    assert f[peak] < 20.0


def _toy_rapsd(n=128):
    # 自然图谱近似 1/f^2：低频能量高、高频低
    f = torch.linspace(1e-3, 0.5, n)
    return 1.0 / (f ** 2)


def test_weight_table_cdf_monotone_and_normalized():
    rapsd = _toy_rapsd()
    t_grid, w, cdf = build_csflow_weight_table(rapsd, alpha=1.0, n_t=512)
    assert t_grid.shape == w.shape == cdf.shape
    # w 单位均值
    assert abs(float(w.mean()) - 1.0) < 1e-4
    # cdf 单调不减，端点 [~0, 1]
    assert torch.all(cdf[1:] - cdf[:-1] >= -1e-6)
    assert float(cdf[-1]) == 1.0 or abs(float(cdf[-1]) - 1.0) < 1e-6
    # alpha=0 → 退化均匀（w 全 1）
    _, w0, _ = build_csflow_weight_table(rapsd, alpha=0.0, n_t=512)
    assert torch.allclose(w0, torch.ones_like(w0), atol=1e-5)


def test_sampler_in_range_and_biased():
    rapsd = _toy_rapsd()
    t_grid, w, cdf = build_csflow_weight_table(rapsd, alpha=1.0, n_t=1024)
    s = CSFlowSampler(t_grid, cdf, t_min=1e-4, t_max=1 - 1e-4)
    torch.manual_seed(0)
    t = s.sample(20000, torch.device("cpu"))
    assert t.shape == (20000,)
    assert float(t.min()) >= 1e-4 and float(t.max()) <= 1 - 1e-4
    # 非均匀：与 U(0,1) 的均值 0.5 应有可测偏离（CSFlow 偏感知中频→中/低 t）
    assert abs(float(t.mean()) - 0.5) > 0.02
    # 采样的经验密度应与 w 正相关：落在"高 w bin"的样本比例应高于这些 bin 在网格中的占比
    # （即比均匀采样更集中在高 w 区——这是 ∝w 采样的稳健不变量，不要求 >50%）。
    hi = (w > w.mean()).float()
    frac_hi = float(hi[torch.searchsorted(t_grid, t).clamp(max=len(t_grid) - 1)].mean())
    uniform_frac = float(hi.mean())
    assert frac_hi > uniform_frac + 0.03, (frac_hi, uniform_frac)


def test_from_profile_roundtrip():
    rapsd = _toy_rapsd(96)
    prof = {"rapsd": rapsd.tolist(), "resolution": 1024, "n_images": 50,
            "channels": "luma", "source": "toy"}
    tmp = os.path.join(tempfile.gettempdir(), "_csflow_prof.json")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(prof, f)
    s = CSFlowSampler.from_profile(tmp, alpha=1.0, pixels_per_degree=50.0)
    t = s.sample(100, torch.device("cpu"))
    assert t.shape == (100,)
    assert "alpha" in s.meta and s.meta["n_images"] == 50
    os.remove(tmp)


if __name__ == "__main__":
    test_mannos_csf_shape_and_peak()
    test_weight_table_cdf_monotone_and_normalized()
    test_sampler_in_range_and_biased()
    test_from_profile_roundtrip()
    print("ALL CSFLOW TESTS PASSED")
