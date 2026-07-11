"""ABBA (arXiv:2505.14238) 适配器的单元测试。

覆盖：
  1. step-0 净 ΔW=0（行为中立：注入后前向 == 原模型前向）
  2. Khatri-Rao 前向与 naive Hadamard 物化逐元素恒等（官方 Thm.1 的实现正确性）
  3. SVD init：(B1,A1) 逼近 W0 的最优 rank-r1 截断（EYM 口径）
  4. step-0 梯度流：只有 B2 有梯度（B2=0 阻断其余三因子），一步后全因子解冻
  5. get_param_groups：4 因子全部进组；LoRA+ ratio 只作用 b1/b2
  6. state_dict(raw) → load_state_dict_from_mapping 无损往返
  7. state_dict(export_for_comfy=True)：KR 物化标准 LoRA 键与 delta_weight 精确一致
     （bf16 存储容差内），且 native abba_* 键并存
  8. 非法组合 fail-fast（lokr / dora / tlora / pissa / rank_dropout）
  9. merged_weight == base + delta_weight

本地跑法：
    D:\\ArtificialIntelligence\\ComfyUI-aki-v1.5\\python\\python.exe -m pytest \\
        AnimaLoraToolkit/tests/test_abba.py -v
"""
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch
    HAS_TORCH = True
except ModuleNotFoundError:
    HAS_TORCH = False

IN_F, OUT_F = 24, 16


def _make_model(seed=0):
    torch.manual_seed(seed)

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(IN_F, OUT_F, bias=False)

    return Tiny()


def _inject(seed=0, rank=8, **kw):
    from trainer.lora import LoRAInjector
    model = _make_model(seed=seed)
    defaults = dict(rank=rank, alpha=float(rank), use_abba=True, targets=["q_proj"])
    defaults.update(kw)
    injector = LoRAInjector(**defaults)
    injector.inject(model)
    return model, injector


def _perturb(injector, seed=123):
    torch.manual_seed(seed)
    for lora in injector.injected.values():
        ad = lora.adapter
        with torch.no_grad():
            for p in (ad.abba_a1, ad.abba_b1, ad.abba_a2, ad.abba_b2):
                p.add_(torch.randn_like(p) * 0.05)


@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestABBA(unittest.TestCase):
    def test_step0_zero_delta(self):
        base_model = _make_model(seed=7)
        w0 = base_model.q_proj.weight.detach().clone()
        model, injector = _inject(seed=7)
        # 注入不改 base 权重
        torch.testing.assert_close(model.q_proj.original.weight, w0)
        x = torch.randn(5, IN_F)
        with torch.no_grad():
            out = model.q_proj(x)
            ref = torch.nn.functional.linear(x, w0)
        torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-6)

    def test_khatri_rao_identity(self):
        model, injector = _inject(seed=1)
        _perturb(injector)
        lora = next(iter(injector.injected.values()))
        ad = lora.adapter
        x = torch.randn(4, IN_F)
        with torch.no_grad():
            out_kr = ad(x)
            # naive: ΔW = s·(B1@A1) ∘ (B2@A2)
            d1 = ad.abba_b1 @ ad.abba_a1
            d2 = ad.abba_b2 @ ad.abba_a2
            delta = d1 * d2 * ad.scaling
            out_naive = x @ delta.t()
        torch.testing.assert_close(out_kr, out_naive, atol=1e-5, rtol=1e-5)
        # delta_weight() 与 naive 一致
        torch.testing.assert_close(ad.delta_weight(), delta, atol=1e-5, rtol=1e-5)

    def test_svd_init_near_optimal(self):
        model, injector = _inject(seed=2, rank=8)  # r1 = 4
        ad = next(iter(injector.injected.values())).adapter
        w0 = model.q_proj.original.weight.detach().float()
        approx = ad.abba_b1.detach().float() @ ad.abba_a1.detach().float()
        U, S, Vh = torch.linalg.svd(w0, full_matrices=False)
        r1 = ad.r1
        w_r = (U[:, :r1] * S[:r1]) @ Vh[:r1]
        opt_err = (w0 - w_r).norm()
        got_err = (w0 - approx).norm()
        # 随机化 SVD (niter=10) 应非常接近 EYM 最优截断
        self.assertLessEqual(got_err.item(), opt_err.item() * 1.05 + 1e-6)
        # B2 = 0
        self.assertEqual(ad.abba_b2.abs().max().item(), 0.0)

    def test_step0_grad_flow(self):
        model, injector = _inject(seed=3)
        ad = next(iter(injector.injected.values())).adapter
        x = torch.randn(6, IN_F)
        model.q_proj(x).square().mean().backward()
        # base ΔW=0 但 loss 对 b2 有梯度（经 B1A1 调制）；其余三因子被 B2A2=0 阻断
        self.assertGreater(ad.abba_b2.grad.abs().max().item(), 0.0)
        for p in (ad.abba_a1, ad.abba_b1, ad.abba_a2):
            self.assertEqual(p.grad.abs().max().item(), 0.0)
        # 一步后 b2 != 0 → 全因子解冻
        with torch.no_grad():
            ad.abba_b2.add_(-1.0 * ad.abba_b2.grad)
        for p in (ad.abba_a1, ad.abba_b1, ad.abba_a2, ad.abba_b2):
            p.grad = None
        model.q_proj(x).square().mean().backward()
        for p in (ad.abba_a1, ad.abba_b1, ad.abba_a2, ad.abba_b2):
            self.assertGreater(p.grad.abs().max().item(), 0.0)

    def test_param_groups(self):
        model, injector = _inject(seed=4, loraplus_lr_ratio=4.0)
        groups = injector.get_param_groups(weight_decay=0.01, base_lr=1e-4)
        all_params = [p for g in groups for p in g["params"]]
        self.assertEqual(len(all_params), 4)
        # b 组（LoRA+）lr = 4e-4
        b_group = [g for g in groups if any(p.shape[0] == OUT_F for p in g["params"])]
        self.assertTrue(any(abs(g.get("lr", 0) - 4e-4) < 1e-12 for g in b_group))

    def test_state_dict_roundtrip(self):
        model, injector = _inject(seed=5)
        _perturb(injector)
        x = torch.randn(3, IN_F)
        with torch.no_grad():
            ref = model.q_proj(x)
        sd = injector.state_dict()  # raw：native abba_* 键
        self.assertTrue(any(k.endswith(".abba_a1") for k in sd))
        self.assertFalse(any(k.endswith(".lora_down.weight") for k in sd))

        model2, injector2 = _inject(seed=99)  # 不同 seed → 不同 init
        n = injector2.load_state_dict_from_mapping(sd)
        self.assertEqual(n, 1)
        with torch.no_grad():
            out2 = model2.q_proj(x)
        # base 权重同 seed 不同 → 只比 delta
        d_ref = next(iter(injector.injected.values())).adapter.delta_weight()
        d_new = next(iter(injector2.injected.values())).adapter.delta_weight()
        # raw 导出按仓库惯例存 bf16（真实训练参数本就是 bf16 → 无损；本测试用
        # fp32 参数，只能到 bf16 量化容差）
        torch.testing.assert_close(d_ref, d_new, atol=2e-3, rtol=2e-2)
        del out2, ref

    def test_native_only_export_default(self):
        # 默认 abba_export_kr=False：comfy 导出也只有 native 键（体积=同预算 LoRA），
        # KR 物化在本地用 tools/abba_export_lora.py 完成
        model, injector = _inject(seed=6)
        sd = injector.state_dict(export_for_comfy=True)
        self.assertTrue(any(k.endswith(".abba_a1") for k in sd))
        self.assertFalse(any(k.endswith(".lora_down.weight") for k in sd))

    def test_export_tool_exact(self):
        # native 成品 → 转换工具（energy=1.0）→ 标准 LoRA delta 与训练态精确一致
        import subprocess, tempfile, os, sys as _sys
        model, injector = _inject(seed=10)
        _perturb(injector)
        ad = next(iter(injector.injected.values())).adapter
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "abba.safetensors")
            dst = os.path.join(td, "lora.safetensors")
            injector.save(src)
            tool = str(ROOT / "tools" / "abba_export_lora.py")
            r = subprocess.run([_sys.executable, tool, src, dst, "--device", "cpu"],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            from safetensors import safe_open
            with safe_open(dst, "pt") as f:
                keys = list(f.keys())
                base = next(k for k in keys if k.endswith(".lora_down.weight")).rsplit(".", 2)[0]
                down = f.get_tensor(f"{base}.lora_down.weight").float()
                up = f.get_tensor(f"{base}.lora_up.weight").float()
                alpha = float(f.get_tensor(f"{base}.alpha"))
            delta = (up @ down) * (alpha / down.shape[0])
            # 两次 bf16 量化（native 存盘 + 导出存盘）容差
            torch.testing.assert_close(delta, ad.delta_weight(), atol=3e-2, rtol=3e-2)

    def test_comfy_export_exact(self):
        model, injector = _inject(seed=6, abba_export_kr=True)
        _perturb(injector)
        ad = next(iter(injector.injected.values())).adapter
        sd = injector.state_dict(export_for_comfy=True)
        base = next(k for k in sd if k.endswith(".abba_a1")).rsplit(".", 1)[0]
        down = sd[f"{base}.lora_down.weight"].float()
        up = sd[f"{base}.lora_up.weight"].float()
        alpha = float(sd[f"{base}.alpha"])
        rank = down.shape[0]
        self.assertEqual(rank, ad.r1 * ad.r2)
        delta_export = (up @ down) * (alpha / rank)
        # bf16 存储容差
        torch.testing.assert_close(delta_export, ad.delta_weight(),
                                   atol=2e-2, rtol=2e-2)

    def test_incompatible_combos(self):
        from trainer.lora import LoRAInjector
        with self.assertRaises(ValueError):
            LoRAInjector(use_abba=True, use_lokr=True, targets=["q_proj"])
        with self.assertRaises(ValueError):
            LoRAInjector(use_abba=True, lora_variant="dora", targets=["q_proj"])
        with self.assertRaises(ValueError):
            LoRAInjector(use_abba=True, lora_variant="tlora", targets=["q_proj"])
        with self.assertRaises(ValueError):
            LoRAInjector(use_abba=True, rank_dropout=0.1, targets=["q_proj"])
        with self.assertRaises(ValueError):
            _inject(seed=0, lora_init="pissa")

    def test_merged_weight(self):
        model, injector = _inject(seed=8)
        _perturb(injector)
        lora = next(iter(injector.injected.values()))
        merged = lora.merged_weight()
        expect = lora.original.weight.float() + lora.adapter.delta_weight()
        torch.testing.assert_close(merged, expect, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
