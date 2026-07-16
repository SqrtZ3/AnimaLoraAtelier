"""PiSSA / DoRA(标准 LoRA) / Muon-SF 修复的回归测试。

覆盖的修复（均为 lora_type="lora" 路径，LoKr 路径原本就正确）：
  1. get_param_groups：dora_scale 必须进优化器参数组（此前只有 LoKr 分支加）
  2. state_dict(export_for_comfy=True) + PiSSA：折叠为 rank-2r 标准 LoRA
     （ΔW = [B|−B₀]@[A;A₀]，PiSSA 官方转换），alpha ×2 保持 scaling；
     直接存 B/A 会让 ComfyUI 把 base 的 top-r 主成分加倍
  3. state_dict：标准 LoRA + DoRA 必须导出 dora_scale（comfy 模式做
     output-axis 换算，raw 模式存训练态幅度）
  4. load_state_dict_from_mapping：
     - 折叠 rank-2r 文件按 rank 切回 A/B/A₀/B₀ 无损续训
     - raw 格式回读 lora_*_init buffers（否则 svd_lowrank 随机重算 → 基准漂移）
     - dora_scale 恢复（折叠文件做逆换算回训练态幅度）
  5. LoRALayer.merged_row_norms 的 fast 参数真正生效（此前静默忽略）
  6. MuonScheduleFree：2D 路径 NS 输入有内层动量 buffer（momentum=0 可关）

本地跑法：
    D:\\ArtificialIntelligence\\ComfyUI-aki-v1.5\\python\\python.exe -m pytest \\
        AnimaLoraToolkit/tests/test_pissa_dora_muon_fixes.py -v
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


def _make_model(seed=0, in_f=24, out_f=16):
    torch.manual_seed(seed)

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(in_f, out_f, bias=False)

    return Tiny()


def _make_injector(**kw):
    from trainer.lora import LoRAInjector
    defaults = dict(rank=4, alpha=4.0, use_lokr=False, targets=["q_proj"])
    defaults.update(kw)
    return LoRAInjector(**defaults)


def _inject(seed=0, **kw):
    model = _make_model(seed=seed)
    injector = _make_injector(**kw)
    injector.inject(model)
    return model, injector


def _perturb(injector, seed=123):
    """给 A/B（和 dora_scale）加扰动，模拟训练后状态（净 ΔW ≠ 0）。"""
    torch.manual_seed(seed)
    with torch.no_grad():
        for lora in injector.injected.values():
            lora.adapter.lora_down.weight.add_(
                torch.randn_like(lora.adapter.lora_down.weight) * 0.05)
            lora.adapter.lora_up.weight.add_(
                torch.randn_like(lora.adapter.lora_up.weight) * 0.05)
            if getattr(lora, "use_dora", False):
                lora.dora_scale.add_(torch.randn_like(lora.dora_scale) * 0.1)


# ---------------------------------------------------------------------------
# 1. 参数组
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestParamGroups(unittest.TestCase):
    def test_dora_scale_in_param_groups(self):
        _, injector = _inject(lora_variant="dora")
        groups = injector.get_param_groups(weight_decay=0.01)
        all_params = {id(p) for g in groups for p in g["params"]}
        for lora in injector.injected.values():
            self.assertIn(id(lora.dora_scale), all_params,
                          "dora_scale 必须进优化器参数组，否则幅度冻结、梯度跨步累积")
            # 幅度向量不应吃 weight_decay
            for g in groups:
                if any(id(p) == id(lora.dora_scale) for p in g["params"]):
                    self.assertEqual(g["weight_decay"], 0.0)


# ---------------------------------------------------------------------------
# 2/3/4. PiSSA 折叠导出与加载往返
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestPissaFoldExport(unittest.TestCase):
    def test_comfy_export_folds_to_rank_2r(self):
        model, injector = _inject(lora_init="pissa")
        _perturb(injector)
        sd = injector.state_dict(export_for_comfy=True)
        base = "lora_unet_q_proj"
        down = sd[f"{base}.lora_down.weight"]
        up = sd[f"{base}.lora_up.weight"]
        self.assertEqual(down.shape[0], 8, "折叠后 down 应为 2r 行")
        self.assertEqual(up.shape[1], 8, "折叠后 up 应为 2r 列")
        self.assertAlmostEqual(float(sd[f"{base}.alpha"]), 8.0,
                               msg="alpha 应 ×2 以保持 scaling 不变")
        # ComfyUI 语义重建：delta = (alpha/dim) * up @ down
        scaling_file = float(sd[f"{base}.alpha"]) / down.shape[0]
        delta_file = scaling_file * (up.float() @ down.float())
        lora = injector.injected["q_proj"]
        delta_train = lora.merged_weight() - lora.original.weight.float()
        self.assertTrue(
            torch.allclose(delta_file, delta_train, rtol=1e-5, atol=1e-6),
            f"折叠文件重建的 ΔW 必须等于训练净 ΔW，max abs diff="
            f"{(delta_file - delta_train).abs().max().item()}")
        # 折叠模式不应再冗余携带 init 键
        self.assertNotIn(f"{base}.lora_down_init.weight", sd)

    def test_comfy_export_without_pissa_unchanged(self):
        """default init（无补偿）导出格式必须保持 rank-r 原样。"""
        model, injector = _inject(lora_init="default")
        _perturb(injector)
        sd = injector.state_dict(export_for_comfy=True)
        base = "lora_unet_q_proj"
        self.assertEqual(sd[f"{base}.lora_down.weight"].shape[0], 4)
        self.assertAlmostEqual(float(sd[f"{base}.alpha"]), 4.0)

    def test_folded_roundtrip_restores_training_state(self):
        model, injector = _inject(lora_init="pissa")
        _perturb(injector)
        lora = injector.injected["q_proj"]
        A = lora.adapter.lora_down.weight.detach().clone()
        B = lora.adapter.lora_up.weight.detach().clone()
        A0 = lora.adapter.lora_down_init.detach().clone()
        B0 = lora.adapter.lora_up_init.detach().clone()
        sd = injector.state_dict(export_for_comfy=True)

        # 新 run：同一 base 权重，但先烧掉 RNG 让注入期 svd_lowrank 状态不同
        torch.manual_seed(999)
        torch.randn(64)
        model2, injector2 = _inject(lora_init="pissa")
        injector2.load_state_dict_from_mapping(sd)
        lora2 = injector2.injected["q_proj"]
        self.assertTrue(torch.allclose(lora2.adapter.lora_down.weight, A, atol=1e-6))
        self.assertTrue(torch.allclose(lora2.adapter.lora_up.weight, B, atol=1e-6))
        self.assertTrue(torch.allclose(lora2.adapter.lora_down_init, A0, atol=1e-6),
                        "折叠文件必须能切回 A₀（补偿基准不漂移）")
        self.assertTrue(torch.allclose(lora2.adapter.lora_up_init, B0, atol=1e-6))

    def test_folded_load_into_default_init_registers_buffers(self):
        model, injector = _inject(lora_init="pissa")
        _perturb(injector)
        sd = injector.state_dict(export_for_comfy=True)
        lora = injector.injected["q_proj"]
        delta_train = lora.merged_weight() - lora.original.weight.float()

        model2, injector2 = _inject(lora_init="default")
        injector2.load_state_dict_from_mapping(sd)
        lora2 = injector2.injected["q_proj"]
        self.assertIsNotNone(lora2.adapter.lora_down_init,
                             "加载折叠文件到非补偿式 run 必须补建 init buffers")
        delta2 = lora2.merged_weight() - lora2.original.weight.float()
        self.assertTrue(torch.allclose(delta2, delta_train, rtol=1e-5, atol=1e-6))

    def test_raw_state_dict_restores_init_buffers(self):
        model, injector = _inject(lora_init="pissa")
        _perturb(injector)
        lora = injector.injected["q_proj"]
        A0 = lora.adapter.lora_down_init.detach().clone()
        sd = injector.state_dict(export_for_comfy=False)
        self.assertIn("lora_unet_q_proj.lora_down_init.weight", sd)

        torch.manual_seed(777)
        torch.randn(64)
        model2, injector2 = _inject(lora_init="pissa")
        injector2.load_state_dict_from_mapping(sd)
        # raw 格式 init 以 bf16 存盘 → bf16 量化容差
        self.assertTrue(
            torch.allclose(injector2.injected["q_proj"].adapter.lora_down_init,
                           A0, rtol=1e-2, atol=1e-2),
            "resume 必须回读 init buffers 而不是 svd_lowrank 随机重算")


@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestDoraScaleExport(unittest.TestCase):
    def test_raw_state_dict_has_dora_scale_and_roundtrips(self):
        model, injector = _inject(lora_variant="dora")
        _perturb(injector)
        lora = injector.injected["q_proj"]
        mag = lora.dora_scale.detach().clone()
        sd = injector.state_dict(export_for_comfy=False)
        self.assertIn("lora_unet_q_proj.dora_scale", sd)

        model2, injector2 = _inject(lora_variant="dora")
        injector2.load_state_dict_from_mapping(sd)
        self.assertTrue(
            torch.allclose(injector2.injected["q_proj"].dora_scale, mag,
                           rtol=1e-2, atol=1e-2))

    def test_folded_comfy_dora_scale_inverse_conversion(self):
        """comfy 导出的 output-axis scale 加载回来要逆换算成训练态幅度。"""
        model, injector = _inject(lora_variant="dora", lora_init="pissa")
        _perturb(injector)
        lora = injector.injected["q_proj"]
        mag = lora.dora_scale.detach().clone()
        sd = injector.state_dict(export_for_comfy=True)
        self.assertIn("lora_unet_q_proj.dora_scale", sd)

        model2, injector2 = _inject(lora_variant="dora", lora_init="pissa")
        injector2.load_state_dict_from_mapping(sd)
        self.assertTrue(
            torch.allclose(injector2.injected["q_proj"].dora_scale, mag,
                           rtol=1e-2, atol=1e-2),
            "折叠文件的 dora_scale 逆换算应恢复训练态幅度（bf16 容差内）")


# ---------------------------------------------------------------------------
# 5. 标准 LoRA merged_row_norms 的 fast 路径
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestLoraLayerFastNorm(unittest.TestCase):
    def _adapter(self, pissa=True):
        from trainer.lora import LoRALayer
        torch.manual_seed(3)
        W = torch.randn(16, 24)
        ad = LoRALayer(24, 16, rank=4, alpha=4.0,
                       lora_init="pissa" if pissa else "default",
                       base_weight=W if pissa else None)
        with torch.no_grad():
            ad.lora_down.weight.add_(torch.randn_like(ad.lora_down.weight) * 0.05)
            ad.lora_up.weight.add_(torch.randn_like(ad.lora_up.weight) * 0.05)
        return ad, W

    def test_fast_close_to_exact_with_init_comp(self):
        ad, W = self._adapter(pissa=True)
        Wb = W.bfloat16()
        ref = ad.merged_row_norms(Wb)
        fast = ad.merged_row_norms(Wb, fast=True)
        self.assertFalse(torch.equal(ref, fast), "fast 路径应真正生效（此前被静默忽略）")
        self.assertTrue(
            torch.allclose(ref, fast, rtol=1e-2, atol=1e-3),
            f"fast 范数偏差过大: {(ref - fast).abs().max().item()}")

    def test_fast_matches_full_reference(self):
        ad, W = self._adapter(pissa=True)
        full = (W.float() + ad.delta_weight()).norm(dim=1)
        fast = ad.merged_row_norms(W, fast=True)
        self.assertTrue(torch.allclose(fast, full, rtol=1e-2, atol=1e-3))

    def test_exact_path_unchanged(self):
        ad, W = self._adapter(pissa=True)
        got = ad.merged_row_norms(W)
        full = (W.float() + ad.delta_weight()).norm(dim=1)
        self.assertTrue(torch.allclose(got, full, rtol=1e-5, atol=1e-6))


# ---------------------------------------------------------------------------
# 6. Muon-SF 内层动量
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestMuonSFMomentum(unittest.TestCase):
    def _step_twice(self, opt, p):
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            p.grad = torch.randn_like(p)
            opt.step()

    def test_momentum_buffer_created_and_used(self):
        from utils.muon_optimizer import MuonScheduleFree
        torch.manual_seed(0)
        p = torch.nn.Parameter(torch.randn(8, 6))
        opt = MuonScheduleFree([p], lr=1e-3)
        self._step_twice(opt, p)
        st = opt.state[p]
        self.assertIn("momentum_buffer", st, "默认应有内层动量 buffer 作 NS 输入")
        self.assertIn("z", st)
        # eval/train swap 正常
        opt.eval()
        opt.train()

    def test_momentum_zero_feeds_raw_grad(self):
        from utils.muon_optimizer import MuonScheduleFree
        torch.manual_seed(0)
        p = torch.nn.Parameter(torch.randn(8, 6))
        opt = MuonScheduleFree([p], lr=1e-3, momentum=0.0)
        self._step_twice(opt, p)
        self.assertNotIn("momentum_buffer", opt.state[p])

    def test_momentum_changes_trajectory(self):
        from utils.muon_optimizer import MuonScheduleFree
        results = []
        for m in (0.95, 0.0):
            torch.manual_seed(0)
            p = torch.nn.Parameter(torch.randn(8, 6))
            opt = MuonScheduleFree([p], lr=1e-2, momentum=m)
            torch.manual_seed(42)
            for _ in range(3):
                opt.zero_grad(set_to_none=True)
                p.grad = torch.randn(8, 6)
                opt.step()
            results.append(p.detach().clone())
        self.assertFalse(torch.allclose(results[0], results[1]))


# ---------------------------------------------------------------------------
# 7. Muon RMS 缩放（moonlight）+ fp32 master（2026-07-11 krea2 不拟合修复）
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestMuonRMSScaleAndMaster(unittest.TestCase):
    """锁定两条修复：
    1. moonlight 缩放让 NS 更新的条目 RMS ≈ 0.2·lr，与矩阵形状无关（LoRA 矮宽
       down 矩阵在 keller 口径下被缩小 ~sqrt(in/r) 倍 → 实测不拟合主因之一）。
    2. fp32 master：bf16 参数上的微小更新不再被 ulp 吞掉（此前 78~93% 条目冻结）。
    """

    def test_moonlight_rms_shape_invariant(self):
        from utils.muon_optimizer import zeropower_via_newtonschulz5, _apply_rms_scale
        torch.manual_seed(0)
        rms = []
        for shape in ((8, 512), (512, 8), (64, 64)):
            g = torch.randn(*shape)
            u = _apply_rms_scale(zeropower_via_newtonschulz5(g), "moonlight")
            rms.append(u.pow(2).mean().sqrt().item())
        for v in rms:
            self.assertAlmostEqual(v, 0.2, delta=0.05,
                                   msg=f"moonlight 缩放后 RMS 应 ≈0.2 与形状无关，得到 {rms}")

    def test_keller_mode_preserved_for_comparison(self):
        from utils.muon_optimizer import zeropower_via_newtonschulz5, _apply_rms_scale
        torch.manual_seed(0)
        g = torch.randn(8, 512)
        u = zeropower_via_newtonschulz5(g)
        self.assertTrue(torch.allclose(_apply_rms_scale(u, "keller"), u))  # rows<cols → ×1

    def test_bf16_master_no_freeze_muon_sf(self):
        """LoRA down 真实量级 (~6e-3 kaiming) 的 bf16 参数在 lr=1e-4 下必须能动。
        修复前：仅 ~22% 条目被动过、位移是 fp32 的 1/5；修复后 bf16 ≡ fp32。"""
        from utils.muon_optimizer import MuonScheduleFree
        deltas = {}
        for dtype in (torch.bfloat16, torch.float32):
            torch.manual_seed(0)
            base = torch.empty(8, 512)
            torch.nn.init.kaiming_uniform_(base, a=5 ** 0.5)
            p = torch.nn.Parameter(base.to(dtype).clone())
            p0 = p.detach().float().clone()
            opt = MuonScheduleFree([p], lr=1e-4)
            torch.manual_seed(42)
            for _ in range(100):
                opt.zero_grad(set_to_none=True)
                p.grad = (torch.randn(8, 512) * 1e-3).to(dtype)
                opt.step()
            d = (p.detach().float() - p0)
            deltas[dtype] = d.pow(2).mean().sqrt().item()
            frac = (d != 0).float().mean().item()
            self.assertGreater(frac, 0.9,
                               f"{dtype} 下应有 >90% 条目被更新（冻结修复），得到 {frac:.2%}")
        ratio = deltas[torch.bfloat16] / deltas[torch.float32]
        self.assertAlmostEqual(ratio, 1.0, delta=0.05,
                               msg=f"bf16 位移应与 fp32 一致（fp32 master），比值 {ratio:.3f}")

    def test_muon_sf_eval_train_roundtrip_keeps_y(self):
        """eval→train 往返后参数应精确回到 y master 的舍入值（master 不被 swap 破坏）。

        用 fp32 参数：bf16 下 x−y ~1e-4 会被 randn 量级的 ulp(~8e-3) 吞掉，
        看不出 swap（swap 本身正确）；fp32 才能断言可见的往返。"""
        from utils.muon_optimizer import MuonScheduleFree
        torch.manual_seed(0)
        p = torch.nn.Parameter(torch.randn(8, 6, dtype=torch.float32))
        opt = MuonScheduleFree([p], lr=1e-3)
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            p.grad = torch.randn(8, 6, dtype=torch.float32)
            opt.step()
        y_ref = opt.state[p]["y"].clone()
        before = p.detach().clone()
        opt.eval()
        self.assertFalse(torch.equal(p.detach(), before), "eval 应切到 Polyak 平均 x")
        opt.train()
        self.assertTrue(torch.equal(p.detach(), before), "train 应无损换回 y")
        self.assertTrue(torch.equal(opt.state[p]["y"], y_ref), "swap 不应改动 y master")

    def test_plain_muon_master_matches_fp32(self):
        from utils.muon_optimizer import Muon
        deltas = {}
        for dtype in (torch.bfloat16, torch.float32):
            torch.manual_seed(0)
            base = torch.empty(8, 512)
            torch.nn.init.kaiming_uniform_(base, a=5 ** 0.5)
            p = torch.nn.Parameter(base.to(dtype).clone())
            p0 = p.detach().float().clone()
            opt = Muon([p], lr=1e-4)
            torch.manual_seed(42)
            for _ in range(100):
                opt.zero_grad(set_to_none=True)
                p.grad = (torch.randn(8, 512) * 1e-3).to(dtype)
                opt.step()
            deltas[dtype] = (p.detach().float() - p0).pow(2).mean().sqrt().item()
        ratio = deltas[torch.bfloat16] / deltas[torch.float32]
        self.assertAlmostEqual(ratio, 1.0, delta=0.05,
                               msg=f"plain Muon bf16 应与 fp32 一致，比值 {ratio:.3f}")


@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestDegenerateLayerSvdInitGuard(unittest.TestCase):
    """7. 退化层守卫：min(in,out) < rank 时 SVD 补偿式 init 必须回退 default。

    物证（2026-07-17 云端）：krea2 txtfusion.projector = Linear(12→1) 注入
    rank=32 + lora_init=pissa 时，svd_lowrank 的 q 被钳到 1、[:rank] 切片静默
    切少，copy_ 的隐式广播把形状错误掩盖到前向才炸：
    F.linear 报 mat1(M,32) × mat2(1,1)。
    """

    def test_pissa_degenerate_falls_back_to_default(self):
        from trainer.lora import LoRALayer
        torch.manual_seed(0)
        W = torch.randn(1, 12)
        layer = LoRALayer(12, 1, rank=32, alpha=32.0,
                          lora_init="pissa", base_weight=W)
        # 回退 default：无补偿 buffers，lora_up 零 init
        self.assertIsNone(layer.lora_down_init)
        self.assertIsNone(layer.lora_up_init)
        y = layer(torch.randn(64, 12))
        self.assertEqual(tuple(y.shape), (64, 1))
        self.assertTrue(torch.equal(y, torch.zeros_like(y)),
                        "step-0 净 delta 必须为 0")

    def test_ortho_degenerate_falls_back_to_default(self):
        from trainer.lora import LoRALayer
        torch.manual_seed(0)
        layer = LoRALayer(12, 1, rank=32, alpha=32.0,
                          tlora_enabled=True, tlora_init="ortho")
        self.assertIsNone(layer.lora_down_init)
        self.assertIsNone(layer.lora_up_init)
        y = layer(torch.randn(8, 12))
        self.assertEqual(tuple(y.shape), (8, 1))

    def test_normal_layer_pissa_unchanged(self):
        from trainer.lora import LoRALayer
        torch.manual_seed(0)
        W = torch.randn(48, 64)
        layer = LoRALayer(64, 48, rank=16, alpha=16.0,
                          lora_init="pissa", base_weight=W)
        self.assertIsNotNone(layer.lora_up_init)
        self.assertEqual(tuple(layer.lora_up_init.shape), (48, 16))
        self.assertEqual(tuple(layer.lora_down_init.shape), (16, 64))
        y = layer(torch.randn(32, 64))
        self.assertLess(y.abs().max().item(), 1e-4,
                        "正常层 PiSSA 补偿应保证 step-0 净 delta≈0")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
