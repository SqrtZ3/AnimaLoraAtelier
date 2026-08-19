"""CSFlow CPU 单测（无 model/VAE 依赖）：CSF 曲线、权重表、inverse-transform 采样、档案往返、
首跑自动计算 RAPSD（注入式假源，无需真实图片）。"""

import json
import os
import shutil
import sys
import tempfile
import types

import pytest
import torch

from trainer.csflow import (
    mannos_csf,
    build_csflow_weight_table,
    load_or_compute_rapsd,
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


# --- 首跑自动计算 RAPSD 分支（commit 7aaa6e1 引入但当时漏提交测试）---------------------

def _fake_profile(n=64):
    rapsd = _toy_rapsd(n)
    return {"rapsd": rapsd.tolist(), "resolution": 512, "n_images": 7,
            "channels": "luma", "source": "fake-auto"}


def _install_fake_compute_rapsd(profile):
    """把假的 tools.compute_rapsd 注入 sys.modules（避开真实 tools 导入与真实图片）。

    返回 (calls, cleanup)：calls['n'] 记录被调次数，cleanup() 还原 sys.modules。
    """
    calls = {"n": 0}

    def fake_compute(data_dir, res, max_images):
        calls["n"] += 1
        return profile

    saved = {k: sys.modules.get(k) for k in ("tools", "tools.compute_rapsd")}
    pkg = types.ModuleType("tools")
    pkg.__path__ = []  # 标记为包，允许 from tools.compute_rapsd import ...
    mod = types.ModuleType("tools.compute_rapsd")
    mod.compute_rapsd = fake_compute
    sys.modules["tools"] = pkg
    sys.modules["tools.compute_rapsd"] = mod

    def cleanup():
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    return calls, cleanup


def test_load_or_compute_raises_when_no_profile_and_no_datadir():
    missing = os.path.join(tempfile.gettempdir(), "_csflow_nonexistent_profile.json")
    if os.path.exists(missing):
        os.remove(missing)
    # 既无档、data_dir 又无效 → 明确报错（fail-fast，不静默）
    with pytest.raises(FileNotFoundError):
        load_or_compute_rapsd(missing, data_dir=None)


def test_load_or_compute_auto_computes_and_caches():
    profile = _fake_profile()
    calls, cleanup = _install_fake_compute_rapsd(profile)
    workdir = tempfile.mkdtemp()
    try:
        data_dir = os.path.join(workdir, "imgs")
        os.makedirs(data_dir)
        cache = os.path.join(workdir, "rapsd.json")
        # 档不存在 → 自动从 data_dir 计算并落盘缓存
        prof = load_or_compute_rapsd(cache, data_dir=data_dir, res=512, max_images=16)
        assert prof["source"] == "fake-auto"
        assert calls["n"] == 1
        assert os.path.exists(cache)  # 已缓存
        # 再次调用 → 命中缓存，不再触发计算
        prof2 = load_or_compute_rapsd(cache, data_dir=data_dir)
        assert calls["n"] == 1
        assert prof2["n_images"] == 7
    finally:
        cleanup()
        shutil.rmtree(workdir, ignore_errors=True)


def test_from_profile_auto_computes_when_missing():
    profile = _fake_profile()
    calls, cleanup = _install_fake_compute_rapsd(profile)
    workdir = tempfile.mkdtemp()
    try:
        data_dir = os.path.join(workdir, "imgs")
        os.makedirs(data_dir)
        cache = os.path.join(workdir, "rapsd.json")
        # from_profile 档缺失时应自动计算、构出可用采样器
        s = CSFlowSampler.from_profile(cache, alpha=1.0, data_dir=data_dir,
                                       res=512, max_images=16)
        t = s.sample(64, torch.device("cpu"))
        assert t.shape == (64,)
        assert calls["n"] == 1
        assert s.meta["n_images"] == 7
        assert os.path.exists(cache)
    finally:
        cleanup()
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    test_mannos_csf_shape_and_peak()
    test_weight_table_cdf_monotone_and_normalized()
    test_sampler_in_range_and_biased()
    test_from_profile_roundtrip()
    test_load_or_compute_raises_when_no_profile_and_no_datadir()
    test_load_or_compute_auto_computes_and_caches()
    test_from_profile_auto_computes_when_missing()
    print("ALL CSFLOW TESTS PASSED")
