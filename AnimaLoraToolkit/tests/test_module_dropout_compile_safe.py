"""module_dropout 的两条路径：

- eager（默认，torch_compile 关）：forward 内 ``torch.rand().item()`` 懒抽签、命中即早返回，
  与最初实现逐字节一致、零额外每步开销；injector 的 roll/clear 短路、不遍历层。
- compile-safe（torch_compile 开）：keep 标量在 forward 外预抽（``roll_module_dropout``），
  forward 只读它做乘法 —— 把数据依赖的 RNG 分支移出编译区域，fullgraph 干净、不每步重编译。

由 ``set_module_dropout_compile_safe(flag)`` 在 setup 时一次性定档。
"""
import pathlib
import sys
import unittest
from unittest import mock

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trainer.lora import LoRALayer, LoKrLayer, LoRALinear, LoRAInjector


class CountingLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        self.forward_calls = 0

    def forward(self, x):
        self.forward_calls += 1
        return super().forward(x)


class TestModuleDropoutCompileSafe(unittest.TestCase):
    def _layer(self, p, compile_safe=True):
        torch.manual_seed(0)
        m = LoRALayer(8, 8, rank=4, alpha=4.0, module_dropout=p)
        # 让 LoRA 有非零输出：lora_up 默认 0 初始化 → 输出恒 0，无法区分 keep。
        with torch.no_grad():
            m.lora_up.weight.normal_()
        m.set_module_dropout_compile_safe(compile_safe)
        return m

    # ── compile-safe 路径 ────────────────────────────────────────────────
    def test_roll_draws_zero_and_one(self):
        m = self._layer(0.5, compile_safe=True)
        m.train()
        seen = set()
        for _ in range(200):
            m.roll_module_dropout()
            seen.add(float(m._md_keep.item()))
        self.assertEqual(seen, {0.0, 1.0})

    def test_keep_zero_zeros_output_keep_one_matches(self):
        m = self._layer(0.5, compile_safe=True)
        m.train()
        x = torch.randn(2, 3, 8)
        m.clear_module_dropout()  # keep=None → 无 dropout 的参考输出
        ref = m(x)
        self.assertTrue(torch.any(ref != 0), "参考输出不应恒为 0（否则测不出 dropout）")

        m._md_keep = torch.tensor(0.0)
        torch.testing.assert_close(m(x), torch.zeros_like(ref))

        m._md_keep = torch.tensor(1.0)
        torch.testing.assert_close(m(x), ref)

    def test_eval_mode_never_drops(self):
        m = self._layer(0.9, compile_safe=True)
        m.eval()
        m.roll_module_dropout()
        self.assertIsNone(m._md_keep)

    def test_forward_fullgraph_compiles(self):
        """fullgraph=True：compile-safe 下 forward 无 .item()/数据依赖分支（rand 分支被
        `not self._md_compile_safe`=False 常量短路掉），否则会抛错。"""
        m = self._layer(0.5, compile_safe=True)
        m.train()
        m.roll_module_dropout()
        x = torch.randn(2, 8)
        compiled = torch.compile(m.forward, backend="eager", fullgraph=True)
        out = compiled(x)
        self.assertEqual(out.shape, (2, 8))

    def test_no_recompile_when_keep_changes_each_step(self):
        """钉住编译稳定性：每步把 _md_keep 重新赋一个新 0-dim 标量，dynamo 只编译一次
        （它按 shape/dtype 而非 value/identity 设 guard）。若回归成每步重编译，本测试会失败。"""
        import torch._dynamo as dyn

        compiles = {"n": 0}

        def counting_backend(gm, ex):
            compiles["n"] += 1
            return gm.forward

        m = self._layer(0.5, compile_safe=True)
        m.train()
        m._md_keep = torch.tensor(1.0)
        x = torch.randn(2, 8)
        dyn.reset()
        cf = torch.compile(m.forward, backend=counting_backend, fullgraph=True)
        for i in range(8):
            m._md_keep = torch.tensor(float(i % 2))  # 新对象、值在 0/1 间变化
            cf(x)
        self.assertEqual(compiles["n"], 1, f"期望只编译 1 次，实际 {compiles['n']} 次（每步重编译说明回归）")

    def test_lokr_keep_scalar(self):
        torch.manual_seed(0)
        m = LoKrLayer(8, 8, rank=2, alpha=4.0, factor=2, module_dropout=0.5)
        with torch.no_grad():
            m.lokr_w2_a.normal_()
            m.lokr_w2_b.normal_()
        m.set_module_dropout_compile_safe(True)
        m.train()
        x = torch.randn(2, 8)
        m.clear_module_dropout()
        ref = m(x)
        m._md_keep = torch.tensor(0.0)
        torch.testing.assert_close(m(x), torch.zeros_like(ref))
        m._md_keep = torch.tensor(1.0)
        torch.testing.assert_close(m(x), ref)

    # ── eager 路径（零额外开销，等价最初实现）────────────────────────────
    def test_eager_default_is_not_compile_safe(self):
        m = LoRALayer(8, 8, rank=4, module_dropout=0.02)
        self.assertFalse(m._md_compile_safe)

    def test_eager_roll_is_noop(self):
        """eager 下 roll 不预抽 keep（forward 内懒抽签处理）。"""
        m = self._layer(0.5, compile_safe=False)
        m.train()
        m.roll_module_dropout()
        self.assertIsNone(m._md_keep)

    def test_eager_forward_drops_and_keeps(self):
        m = self._layer(0.5, compile_safe=False)
        m.train()
        x = torch.randn(2, 8)
        with mock.patch("torch.rand", return_value=torch.tensor([0.0])):  # < p → drop
            torch.testing.assert_close(m(x), torch.zeros(2, m.lora_up.out_features))
        with mock.patch("torch.rand", return_value=torch.tensor([0.99])):  # >= p → keep
            self.assertTrue(torch.any(m(x) != 0))

    def test_injector_eager_roll_clear_dont_touch_layers(self):
        """injector 在 eager 下 roll/clear 短路：不会把任何层的 _md_keep 置成非 None。"""
        class Tiny(torch.nn.Module):
            def __init__(s):
                super().__init__()
                s.q_proj = torch.nn.Linear(8, 8)

            def forward(s, x):
                return s.q_proj(x)

        inj = LoRAInjector(rank=4, alpha=4.0, targets=["q_proj"], module_dropout=0.5)
        inj.inject(Tiny())
        inj.set_module_dropout_compile_safe(False)
        for lora in inj.injected.values():
            lora.train()
        inj.roll_module_dropout()
        self.assertTrue(all(l.adapter._md_keep is None for l in inj.injected.values()))

    # ── DoRA 路径 ───────────────────────────────────────────────────────
    def _dora(self, base, compile_safe):
        torch.manual_seed(0)
        m = LoRALinear(base, rank=2, alpha=4.0, use_lokr=True, factor=2,
                       module_dropout=0.5, lora_variant="dora")
        m.set_module_dropout_compile_safe(compile_safe)
        m.train()
        return m

    def test_dora_eager_keep_does_not_run_extra_base_linear(self):
        base = CountingLinear(8, 8)
        m = self._dora(base, compile_safe=False)
        x = torch.randn(2, 8)
        with mock.patch("torch.rand", return_value=torch.tensor([0.99])):  # keep
            _ = m(x)
        self.assertEqual(base.forward_calls, 0,
                         "eager DoRA keep 应直接算 DoRA linear，不额外跑一次 base forward")

    def test_dora_eager_drop_skips_delta_materialization(self):
        base = CountingLinear(8, 8)
        m = self._dora(base, compile_safe=False)
        delta_calls = {"n": 0}
        orig_delta = m.adapter.delta_weight

        def counted(*a, **k):
            delta_calls["n"] += 1
            return orig_delta(*a, **k)

        m.adapter.delta_weight = counted
        x = torch.randn(2, 8)
        expected = torch.nn.functional.linear(x, base.weight, base.bias)
        with mock.patch("torch.rand", return_value=torch.tensor([0.0])):  # drop
            out = m(x)
        self.assertEqual(base.forward_calls, 1)
        self.assertEqual(delta_calls["n"], 0)
        torch.testing.assert_close(out, expected)

    def test_dora_compile_safe_keep_zero_blends_to_base(self):
        base = CountingLinear(8, 8)
        m = self._dora(base, compile_safe=True)
        m.adapter._md_keep = torch.tensor(0.0)
        x = torch.randn(2, 8)
        expected = torch.nn.functional.linear(x, base.weight, base.bias)
        out = m(x)
        self.assertEqual(base.forward_calls, 0,
                         "compile-safe DoRA dropout 在权重域 blend，不额外跑 base forward")
        torch.testing.assert_close(out, expected)


if __name__ == "__main__":
    unittest.main()
