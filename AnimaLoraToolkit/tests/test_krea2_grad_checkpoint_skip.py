"""分块 grad checkpoint（`checkpoint_skip_last`）的数学恒等性测试。

`checkpoint_skip_last=N` 让最后 N 个 transformer block 跳过 gradient checkpointing
（存全部激活、backward 不重算）。这纯粹是显存/计算的取舍 —— 前向输出与**所有参数的
梯度**都必须与全量 checkpoint（N=0）以及完全不 checkpoint 逐元素一致。

覆盖：
  1. `_checkpoint_from_block` 的边界语义（N=0 / 0<N<L / N>=L / use_checkpoint=False）
  2. forward_packed_navit：N ∈ {0,1,3,6} 的输出 + 全参数梯度 vs 全量 checkpoint
  3. forward_dense：同上
  4. 阴性对照：checkpoint 关闭时该参数不改变行为（不误接线）

Needs CUDA。本地跑法：
    D:\\ArtificialIntelligence\\ComfyUI-aki-v1.5\\python\\python.exe -m pytest \\
        AnimaLoraToolkit/tests/test_krea2_grad_checkpoint_skip.py -v
"""
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch
    from models import krea2_modeling as k2
    HAS_TORCH = True
except ModuleNotFoundError:
    HAS_TORCH = False

HAS_CUDA = HAS_TORCH and torch.cuda.is_available()

N_LAYERS = 6


def _tiny_config():
    return k2.SingleMMDiTConfig(
        features=128, tdim=32, txtdim=64, heads=2, kvheads=1,
        multiplier=2, layers=N_LAYERS, patch=2, channels=16,
        txtheads=2, txtkvheads=1, txtlayers=3,
    )


def _tiny_model(device="cuda", dtype=torch.float32):
    torch.manual_seed(0)
    m = k2.SingleStreamDiT(_tiny_config())
    # 官方零初始化的调制/gate 会让路径退化成恒等 → 随机化以获得非平凡梯度
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    return m.to(device=device, dtype=dtype).train()


def _make_pack(model, device="cuda", dtype=torch.float32):
    latent_shapes = [(4, 4), (6, 8), (4, 6)]
    text_lens = [5, 9, 3]
    cfg = model.config
    lat_list, cross_list, tokens_list, grid_list, vseq = [], [], [], [], []
    torch.manual_seed(1)
    for (h, w), L in zip(latent_shapes, text_lens):
        lat = torch.randn(1, 16, 1, h, w, device=device, dtype=dtype)
        lat_list.append(lat)
        cross_list.append(torch.randn(1, L, cfg.txtlayers, cfg.txtdim, device=device, dtype=dtype))
        tok, grid, _m, _s = model.patchify_latents_to_tokens(lat)
        tokens_list.append(tok)
        grid_list.append(grid)
        vseq.append(tok.shape[1])
    t = torch.tensor([0.2, 0.6, 0.9], device=device, dtype=torch.float32)
    return (lat_list, cross_list,
            torch.cat(tokens_list, dim=1), torch.cat(grid_list, dim=2),
            torch.cat(cross_list, dim=1), vseq, text_lens, t)


def _grads_of(model, out):
    """对 out 做一个确定性的标量归约后 backward，返回 {name: grad}（已 detach 克隆）。"""
    model.zero_grad(set_to_none=True)
    # 用非均匀权重，避免对称归约掩盖某些方向上的差异
    w = torch.linspace(0.3, 1.7, out.numel(), device=out.device, dtype=out.dtype).reshape(out.shape)
    (out * w).sum().backward()
    return {n: (p.grad.detach().clone() if p.grad is not None else None)
            for n, p in model.named_parameters()}


@unittest.skipUnless(HAS_TORCH, "needs torch")
class CheckpointFromBlockSemanticsTests(unittest.TestCase):
    """纯索引语义，不需要 GPU。"""

    def setUp(self):
        self.model = k2.SingleStreamDiT(_tiny_config())

    def test_skip_zero_checkpoints_all(self):
        self.assertEqual(self.model._checkpoint_from_block(True, 0), N_LAYERS)

    def test_partial_skip(self):
        self.assertEqual(self.model._checkpoint_from_block(True, 1), N_LAYERS - 1)
        self.assertEqual(self.model._checkpoint_from_block(True, 3), N_LAYERS - 3)

    def test_skip_ge_layers_clamps_to_zero(self):
        """N >= 层数 → 一层都不 checkpoint（不应产生负下标）。"""
        self.assertEqual(self.model._checkpoint_from_block(True, N_LAYERS), 0)
        self.assertEqual(self.model._checkpoint_from_block(True, N_LAYERS + 10), 0)

    def test_checkpoint_disabled_ignores_skip(self):
        """use_checkpoint=False 时无论 skip 多少都一层不 checkpoint。

        返回值是"checkpoint 到第几个 block 为止"的开区间上界（调用点是
        `_i < 返回值`），所以"一个都不 checkpoint" == 返回 0。曾经这里断言
        N_LAYERS，把"全部 checkpoint"当成了"全不 checkpoint"，反而把 bug 锁死。
        """
        for skip in (0, 2, 99):
            self.assertEqual(self.model._checkpoint_from_block(False, skip), 0)


@unittest.skipUnless(HAS_TORCH and HAS_CUDA, "needs torch + CUDA")
class CheckpointSkipLastEquivalenceTests(unittest.TestCase):
    ATOL = 1e-5

    def _assert_grads_close(self, ref, got, label):
        self.assertEqual(set(ref), set(got), f"{label}: 参数集合不一致")
        worst, worst_name = 0.0, None
        for name in ref:
            a, b = ref[name], got[name]
            self.assertEqual(a is None, b is None, f"{label}: {name} 梯度有无不一致")
            if a is None:
                continue
            d = (a - b).abs().max().item()
            if d > worst:
                worst, worst_name = d, name
        self.assertLess(worst, self.ATOL,
                        f"{label}: 最大梯度差 {worst:.3e} 在 {worst_name}（>= {self.ATOL}）")
        return worst

    def test_navit_packed_all_skips_match_full_checkpoint(self):
        model = _tiny_model()
        _lat, _cross, tokens, grid, cross_packed, vseq, tseq, t = _make_pack(model)

        def run(**kw):
            out = model.forward_packed_navit(
                tokens, t, cross_packed, grid, vseq, tseq, **kw)
            return out, _grads_of(model, out)

        ref_out, ref_grads = run(use_checkpoint=True)          # 全量 checkpoint = 基线
        for skip in (0, 1, 3, N_LAYERS, N_LAYERS + 5):
            out, grads = run(use_checkpoint=True, checkpoint_skip_last=skip)
            od = (out - ref_out).abs().max().item()
            self.assertLess(od, self.ATOL, f"skip={skip}: 输出差 {od:.3e}")
            gd = self._assert_grads_close(ref_grads, grads, f"navit skip={skip}")
            print(f"[krea2] navit skip_last={skip}: out_diff={od:.3e} grad_diff={gd:.3e}")

    def test_navit_packed_no_checkpoint_matches(self):
        """完全不 checkpoint 也应与全量 checkpoint 一致（重算路径本身的正确性）。"""
        model = _tiny_model()
        _lat, _cross, tokens, grid, cross_packed, vseq, tseq, t = _make_pack(model)
        out_ref = model.forward_packed_navit(tokens, t, cross_packed, grid, vseq, tseq,
                                             use_checkpoint=True)
        g_ref = _grads_of(model, out_ref)
        out_no = model.forward_packed_navit(tokens, t, cross_packed, grid, vseq, tseq,
                                            use_checkpoint=False)
        g_no = _grads_of(model, out_no)
        od = (out_no - out_ref).abs().max().item()
        self.assertLess(od, self.ATOL, f"no-checkpoint 输出差 {od:.3e}")
        gd = self._assert_grads_close(g_ref, g_no, "navit no-checkpoint")
        print(f"[krea2] navit no-ckpt: out_diff={od:.3e} grad_diff={gd:.3e}")

    def test_dense_all_skips_match_full_checkpoint(self):
        model = _tiny_model()
        torch.manual_seed(2)
        lat = torch.randn(2, 16, 1, 6, 8, device="cuda", dtype=torch.float32)
        cross = torch.randn(2, 7, model.config.txtlayers, model.config.txtdim,
                            device="cuda", dtype=torch.float32)
        t = torch.tensor([[0.3], [0.8]], device="cuda", dtype=torch.float32)

        def run(**kw):
            out = model.forward_dense(lat, t, cross, **kw)
            return out, _grads_of(model, out)

        ref_out, ref_grads = run(use_checkpoint=True)
        for skip in (0, 2, N_LAYERS):
            out, grads = run(use_checkpoint=True, checkpoint_skip_last=skip)
            od = (out - ref_out).abs().max().item()
            self.assertLess(od, self.ATOL, f"dense skip={skip}: 输出差 {od:.3e}")
            gd = self._assert_grads_close(ref_grads, grads, f"dense skip={skip}")
            print(f"[krea2] dense skip_last={skip}: out_diff={od:.3e} grad_diff={gd:.3e}")

    def test_default_is_byte_identical_to_previous_behavior(self):
        """不传该参数 ≡ 传 0 —— 保证默认路径行为中立。"""
        model = _tiny_model()
        _lat, _cross, tokens, grid, cross_packed, vseq, tseq, t = _make_pack(model)
        a = model.forward_packed_navit(tokens, t, cross_packed, grid, vseq, tseq,
                                       use_checkpoint=True)
        b = model.forward_packed_navit(tokens, t, cross_packed, grid, vseq, tseq,
                                       use_checkpoint=True, checkpoint_skip_last=0)
        self.assertEqual((a - b).abs().max().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
