"""module_dropout 的 keep 标量化改造：compile 安全 + 行为等价。

旧实现在 forward 里用 ``torch.rand(1).item() < p`` 早返回零，``.item()`` 在
``torch.compile(fullgraph=True)`` 下会打断图。新实现把 keep 标量在 forward 之外预抽
（``roll_module_dropout``），forward 只做乘法 —— 本测试钉住这两点。
"""
import pathlib
import sys
import unittest

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trainer.lora import LoRALayer, LoKrLayer


class TestModuleDropoutCompileSafe(unittest.TestCase):
    def _layer(self, p):
        torch.manual_seed(0)
        m = LoRALayer(8, 8, rank=4, alpha=4.0, module_dropout=p)
        # 让 LoRA 有非零输出：lora_up 默认 0 初始化 → 输出恒 0，无法区分 keep。
        with torch.no_grad():
            m.lora_up.weight.normal_()
        return m

    def test_roll_draws_zero_and_one(self):
        m = self._layer(0.5)
        m.train()
        seen = set()
        for _ in range(200):
            m.roll_module_dropout()
            seen.add(float(m._md_keep.item()))
        self.assertEqual(seen, {0.0, 1.0})

    def test_keep_zero_zeros_output_keep_one_matches(self):
        m = self._layer(0.5)
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
        m = self._layer(0.9)
        m.eval()
        m.roll_module_dropout()
        self.assertIsNone(m._md_keep)

    def test_forward_fullgraph_compiles(self):
        """fullgraph=True：若 forward 里还残留 .item()/数据依赖分支会直接抛错。"""
        m = self._layer(0.5)
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

        m = self._layer(0.5)
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
        m.train()
        x = torch.randn(2, 8)
        m.clear_module_dropout()
        ref = m(x)
        m._md_keep = torch.tensor(0.0)
        torch.testing.assert_close(m(x), torch.zeros_like(ref))
        m._md_keep = torch.tensor(1.0)
        torch.testing.assert_close(m(x), ref)


if __name__ == "__main__":
    unittest.main()
