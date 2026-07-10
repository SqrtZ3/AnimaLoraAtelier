"""Krea2 路径三项每步加速优化的数值等价性测试。

覆盖：
  1. DoRA merged_row_norms：
     - base_row_sq 预计算缓存 ≡ 现算（逐 bit）
     - fast=True（bf16 收缩）≈ fp32 精确路径（rtol 1e-2）
     - LoRALinear DoRA forward ≡ merged_weight() 全矩阵参照
     - dora_detach_norm：forward 值不变、梯度与非 detach 不同且仍回流 LoKr 因子
  2. encode_krea2_text 批量化：batch 前向 ≡ 逐条前向（用确定性 fake TE/tokenizer
     验证批处理/压缩/cache 逻辑本身，不依赖真 Qwen3-VL）
  3. xformers GQA 5D grouped 布局 ≡ repeat_interleave 展开（需 CUDA + xformers）

本地跑法：
    D:\\ArtificialIntelligence\\ComfyUI-aki-v1.5\\python\\python.exe -m pytest \\
        AnimaLoraToolkit/tests/test_krea2_optimizations.py -v
"""
import pathlib
import sys
import types
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch
    HAS_TORCH = True
except ModuleNotFoundError:
    HAS_TORCH = False

HAS_CUDA = HAS_TORCH and torch.cuda.is_available()
try:
    from xformers.ops.fmha import BlockDiagonalMask  # noqa: F401
    HAS_XFORMERS = True
except Exception:
    HAS_XFORMERS = False


# ---------------------------------------------------------------------------
# 1. DoRA merged_row_norms
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestMergedRowNorms(unittest.TestCase):
    def _make_adapter(self, in_f=24, out_f=16, factor=2, rank=4, seed=0):
        from trainer.lora import LoKrLayer
        torch.manual_seed(seed)
        ad = LoKrLayer(in_f, out_f, rank=rank, alpha=float(rank), factor=factor)
        with torch.no_grad():
            # w2_b 默认 0 → ΔW=0 平凡；随机化取非平凡值
            for p in ad.parameters():
                p.add_(torch.randn_like(p) * 0.3)
        ad.train()
        return ad

    def test_base_row_sq_cache_exact(self):
        ad = self._make_adapter()
        W = torch.randn(16, 24)
        ref = ad.merged_row_norms(W)
        cached = (W.float() ** 2).sum(dim=1)
        got = ad.merged_row_norms(W, base_row_sq=cached)
        self.assertTrue(torch.equal(ref, got), "base_row_sq 缓存必须逐 bit 等价")

    def test_fast_close_to_exact(self):
        ad = self._make_adapter()
        W = torch.randn(16, 24).bfloat16()
        ref = ad.merged_row_norms(W)
        fast = ad.merged_row_norms(W, fast=True)
        self.assertTrue(
            torch.allclose(ref, fast, rtol=1e-2, atol=1e-3),
            f"fast 范数偏差过大: max rel={(ref - fast).abs().max() / ref.abs().max()}",
        )

    def test_norms_match_full_delta(self):
        """解析范数 ≡ 全矩阵 ||W+ΔW|| 参照。"""
        ad = self._make_adapter()
        W = torch.randn(16, 24)
        ad._rd_mask = None
        got = ad.merged_row_norms(W)
        full = (W.float() + ad.delta_weight()).norm(dim=1)
        self.assertTrue(torch.allclose(got, full, rtol=1e-5, atol=1e-6))

    def test_dora_forward_matches_merged_weight(self):
        from trainer.lora import LoRALinear
        torch.manual_seed(1)
        base = torch.nn.Linear(24, 16, bias=True)
        lora = LoRALinear(base, rank=4, alpha=4.0, use_lokr=True, factor=2,
                          lora_variant="dora")
        with torch.no_grad():
            lora.adapter.lokr_w2_b.add_(torch.randn_like(lora.adapter.lokr_w2_b) * 0.3)
        lora.train()
        x = torch.randn(3, 24)
        y = lora(x)
        ref = torch.nn.functional.linear(
            x.float(), lora.merged_weight(), base.bias.float()
        )
        self.assertTrue(torch.allclose(y.float(), ref, rtol=1e-4, atol=1e-5))

    def test_detach_norm_same_value_diff_grad(self):
        from trainer.lora import LoRALinear
        torch.manual_seed(2)

        def build(detach):
            torch.manual_seed(2)
            base = torch.nn.Linear(24, 16, bias=False)
            lora = LoRALinear(base, rank=4, alpha=4.0, use_lokr=True, factor=2,
                              lora_variant="dora", dora_detach_norm=detach)
            with torch.no_grad():
                lora.adapter.lokr_w2_b.add_(
                    torch.randn_like(lora.adapter.lokr_w2_b) * 0.3)
            lora.train()
            return lora

        x = torch.randn(3, 24)
        a, b = build(False), build(True)
        ya, yb = a(x), b(x)
        # forward 值：detach 不改变数值（同一计算，只是切断梯度）
        self.assertTrue(torch.allclose(ya, yb, rtol=1e-6, atol=1e-7))

        ya.square().sum().backward()
        yb.square().sum().backward()
        # 两者 LoKr 因子都有梯度（detach 只切范数支路，主支路仍回流）
        self.assertIsNotNone(a.adapter.lokr_w2_a.grad)
        self.assertIsNotNone(b.adapter.lokr_w2_a.grad)
        self.assertIsNotNone(b.dora_scale.grad)
        # 梯度应不同（范数支路贡献被切掉）
        self.assertFalse(torch.allclose(
            a.adapter.lokr_w2_a.grad, b.adapter.lokr_w2_a.grad, rtol=1e-4))

    def test_fast_norm_grad_flows(self):
        from trainer.lora import LoRALinear
        torch.manual_seed(3)
        base = torch.nn.Linear(24, 16, bias=False).bfloat16()
        lora = LoRALinear(base, rank=4, alpha=4.0, use_lokr=True, factor=2,
                          lora_variant="dora", dora_fast_norm=True)
        with torch.no_grad():
            lora.adapter.lokr_w2_b.add_(
                torch.randn_like(lora.adapter.lokr_w2_b) * 0.3)
        lora.train()
        x = torch.randn(3, 24).bfloat16()
        y = lora(x)
        y.float().square().sum().backward()
        self.assertIsNotNone(lora.adapter.lokr_w2_a.grad)
        self.assertIsNotNone(lora.dora_scale.grad)


# ---------------------------------------------------------------------------
# 2. encode_krea2_text 批量化（fake TE / tokenizer）
# ---------------------------------------------------------------------------
class _FakeBatch(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    """确定性 fake：每条文本的有效 token 数 = 34(prefix) + 3 + len(text) % 9。"""

    def __call__(self, texts, truncation=False, return_overflowing_tokens=False,
                 padding=None, max_length=None, return_tensors=None,
                 add_special_tokens=True):
        if padding == "max_length":
            L = int(max_length)
            ids = torch.zeros(len(texts), L, dtype=torch.long)
            mask = torch.zeros(len(texts), L, dtype=torch.long)
            for i, t in enumerate(texts):
                n = min(34 + 3 + (len(t) % 9), L)
                for j in range(n):
                    ids[i, j] = 1 + (ord(t[j % len(t)]) + j) % 997 if t else 1
                mask[i, :n] = 1
            return _FakeBatch(input_ids=ids, attention_mask=mask)
        # suffix：固定 5 个 token，全有效
        L = 5
        ids = torch.arange(1, L + 1, dtype=torch.long).unsqueeze(0).repeat(len(texts), 1)
        mask = torch.ones(len(texts), L, dtype=torch.long)
        return _FakeBatch(input_ids=ids, attention_mask=mask)


class _FakeTE:
    """hidden_states[k][b,l,:] 只依赖该位置的 token id 与层号 → batch ≡ 逐条。"""
    device = "cpu"
    D = 8

    def __call__(self, input_ids, attention_mask=None, output_hidden_states=False):
        hs = []
        base = input_ids.float().unsqueeze(-1)  # [B, L, 1]
        for k in range(37):
            hs.append(base * (k + 1) / 37.0 + torch.arange(self.D).float() * 0.01)
        return types.SimpleNamespace(hidden_states=hs)


@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestEncodeKrea2Batch(unittest.TestCase):
    def setUp(self):
        from trainer import model_family as mf
        mf.reset_krea2_text_cache()
        mf.set_krea2_text_cache(False)
        self.handles = {"model": _FakeTE(), "tokenizer": _FakeTokenizer()}

    def tearDown(self):
        from trainer import model_family as mf
        mf.reset_krea2_text_cache()
        mf.set_krea2_text_cache(True, cap=128)

    def test_batch_equals_single(self):
        from trainer import model_family as mf
        texts = ["1girl, solo, smile", "landscape", "a" * 30]
        batch = mf._encode_krea2_batch(self.handles, texts, "cpu", max_length=16)
        singles = [
            mf._encode_krea2_batch(self.handles, [t], "cpu", max_length=16)[0]
            for t in texts
        ]
        for b, s in zip(batch, singles):
            self.assertEqual(b.shape, s.shape)
            self.assertTrue(torch.equal(b, s))

    def test_encode_krea2_text_pads_and_masks(self):
        from trainer import model_family as mf
        texts = ["short", "a much longer caption here"]
        cross, cmask = mf.encode_krea2_text(self.handles, texts, "cpu", max_length=16)
        self.assertEqual(cross.dim(), 4)
        self.assertEqual(cross.shape[0], 2)
        self.assertEqual(cross.shape[2], 12)  # KREA2_SELECT_LAYERS 层数
        # mask 有效长度 = 各自压缩长度
        lens = cmask.sum(dim=1).tolist()
        self.assertEqual(cross.shape[1], max(lens))
        # pad 区应为 0
        for i, n in enumerate(lens):
            if n < cross.shape[1]:
                self.assertEqual(float(cross[i, n:].abs().sum()), 0.0)

    def test_dedup_and_chunking_equal_single(self):
        """批内重复 caption（multiscale 副本场景）+ 超过 chunk 上限（8）的批，
        输出与逐条编码逐 bit 一致。"""
        from trainer import model_family as mf
        base = [f"caption number {i}" for i in range(6)]
        texts = base + base[:4]  # 10 条、其中 4 条重复 → 触发去重 + 分块
        cross, cmask = mf.encode_krea2_text(self.handles, texts, "cpu", max_length=16)
        for i, t in enumerate(texts):
            ref = mf._encode_krea2_batch(self.handles, [t], "cpu", max_length=16)[0]
            n = int(cmask[i].sum())
            self.assertEqual(n, ref.shape[0])
            self.assertTrue(torch.equal(cross[i, :n], ref))

    def test_cache_mixed_hit_miss(self):
        from trainer import model_family as mf
        mf.set_krea2_text_cache(True, cap=8)
        t1, t2 = "cached one", "cached two"
        ref, _ = mf.encode_krea2_text(self.handles, [t1, t2], "cpu", max_length=16)
        # t1 命中 cache、t3 新增 → 混合路径
        got, _ = mf.encode_krea2_text(self.handles, [t1, "new three"], "cpu", max_length=16)
        L1 = int((ref[0].abs().sum(dim=(1, 2)) > 0).sum())
        self.assertTrue(torch.equal(ref[0, :L1], got[0, :L1]))


# ---------------------------------------------------------------------------
# 3. xformers GQA 5D ≡ repeat_interleave
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAS_CUDA and HAS_XFORMERS, "needs CUDA + xformers")
class TestXformersGqa5D(unittest.TestCase):
    def test_probe_runs_once_on_first_use(self):
        """flag=None 时首次 GQA 调用应触发探针（含 backward）并落定 True/False，
        且无论落定结果如何调用都应产出有限输出。

        实测现状（2026-07）：xformers 的 BMGHK 只有 forward 算子、backward 缺失
        （本地 torch2.7 与云端 fa2 2.8.3 构建一致）→ 探针在当前环境应为 False。
        未来 xformers 支持后此断言自然翻转为 True，届时更新。"""
        from models import krea2_modeling as k2
        old_flag = k2._XF_GQA_5D_OK
        try:
            k2._XF_GQA_5D_OK = None
            q = torch.randn(1, 8, 32, 64, device="cuda", dtype=torch.bfloat16)
            kk = torch.randn(1, 2, 32, 64, device="cuda", dtype=torch.bfloat16)
            vv = torch.randn(1, 2, 32, 64, device="cuda", dtype=torch.bfloat16)
            mask = k2.cached_block_diag_mask((16, 16))
            out = k2.attention(q, kk, vv, mask=mask, gqa=True)
            self.assertIsNotNone(k2._XF_GQA_5D_OK, "探针应已运行并落定")
            self.assertIsInstance(k2._XF_GQA_5D_OK, bool)
            self.assertTrue(torch.isfinite(out.float()).all())
        finally:
            k2._XF_GQA_5D_OK = old_flag

    def test_probe_failure_falls_back(self):
        """探针失败（模拟云端 fa2-only 构建）→ flag=False → 走 repeat 路径且结果正确。"""
        from models import krea2_modeling as k2
        old_flag = k2._XF_GQA_5D_OK
        old_probe = k2._probe_xf_gqa_5d
        try:
            k2._probe_xf_gqa_5d = lambda *a, **kw: False
            k2._XF_GQA_5D_OK = None
            torch.manual_seed(0)
            q = torch.randn(1, 8, 32, 64, device="cuda", dtype=torch.bfloat16)
            kk = torch.randn(1, 2, 32, 64, device="cuda", dtype=torch.bfloat16)
            vv = torch.randn(1, 2, 32, 64, device="cuda", dtype=torch.bfloat16)
            mask = k2.cached_block_diag_mask((16, 16))
            out_fb = k2.attention(q, kk, vv, mask=mask, gqa=True)
            self.assertFalse(k2._XF_GQA_5D_OK)
            k2._XF_GQA_5D_OK = True
            out_5d = k2.attention(q, kk, vv, mask=mask, gqa=True)
            diff = (out_fb.float() - out_5d.float()).abs().max().item()
            self.assertLess(diff, 2e-2)
        finally:
            k2._XF_GQA_5D_OK = old_flag
            k2._probe_xf_gqa_5d = old_probe

    def test_5d_equals_repeat(self):
        from models import krea2_modeling as k2
        torch.manual_seed(0)
        device, dtype = "cuda", torch.bfloat16
        B, Hq, Hkv, D = 1, 8, 2, 64
        seqlens = (48, 96, 32)
        L = sum(seqlens)
        q = torch.randn(B, Hq, L, D, device=device, dtype=dtype)
        k = torch.randn(B, Hkv, L, D, device=device, dtype=dtype)
        v = torch.randn(B, Hkv, L, D, device=device, dtype=dtype)
        mask = k2.cached_block_diag_mask(seqlens)

        old_flag = k2._XF_GQA_5D_OK
        try:
            k2._XF_GQA_5D_OK = True
            out_5d = k2.attention(q, k, v, mask=mask, gqa=True)
            self.assertTrue(
                k2._XF_GQA_5D_OK,
                "5D grouped 路径应在本环境可用（若回退说明 xformers 版本不支持）",
            )
            k2._XF_GQA_5D_OK = False
            out_rep = k2.attention(q, k, v, mask=mask, gqa=True)
        finally:
            k2._XF_GQA_5D_OK = old_flag

        self.assertEqual(out_5d.shape, out_rep.shape)
        diff = (out_5d.float() - out_rep.float()).abs().max().item()
        self.assertLess(diff, 2e-2, f"5D vs repeat 偏差 {diff}")

    def test_training_grad_path_via_probe(self):
        """带梯度输入走探针决定的路径（当前环境探针 False → repeat），
        backward 必须可用且梯度有限——这是训练每步的真实路径。

        注：不再存在旧版的调用内静默回退；此前的 test_5d_backward 正是被
        那个回退掩盖成了假阳性（5D backward 实际不被 xformers 支持）。"""
        from models import krea2_modeling as k2
        torch.manual_seed(1)
        device, dtype = "cuda", torch.bfloat16
        q = torch.randn(1, 8, 64, 64, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(1, 2, 64, 64, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(1, 2, 64, 64, device=device, dtype=dtype, requires_grad=True)
        mask = k2.cached_block_diag_mask((32, 32))
        old_flag = k2._XF_GQA_5D_OK
        try:
            k2._XF_GQA_5D_OK = None   # 让探针现场决定（与真实训练首步一致）
            out = k2.attention(q, k, v, mask=mask, gqa=True)
            out.float().square().sum().backward()
        finally:
            k2._XF_GQA_5D_OK = old_flag
        self.assertIsNotNone(q.grad)
        self.assertIsNotNone(k.grad)
        self.assertTrue(torch.isfinite(k.grad.float()).all())


# ---------------------------------------------------------------------------
# 4. 全 264 targets 注入后的 packed 前向（云端实跑崩过的路径回归）
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAS_CUDA, "needs CUDA")
class TestInjectedPackedForward(unittest.TestCase):
    def test_dora_full_targets_packed_forward(self):
        """LoKr+DoRA 注入官方全部 Linear（含 first）后 forward_packed_navit 不应
        AttributeError（LoRALinear 需透传 in_features），且前向/反传有限。"""
        from models import krea2_modeling as k2
        from trainer.lora import LoRAInjector
        from trainer.model_family import KREA2_DEFAULT_LORA_TARGETS

        cfg = k2.SingleMMDiTConfig(
            features=128, tdim=32, txtdim=64, heads=2, kvheads=1,
            multiplier=2, layers=2, patch=2, channels=16,
            txtheads=2, txtkvheads=1, txtlayers=3,
        )
        torch.manual_seed(0)
        model = k2.SingleStreamDiT(cfg).cuda().bfloat16()
        model.requires_grad_(False)
        inj = LoRAInjector(rank=8, alpha=8.0, use_lokr=True, factor=2,
                           lora_variant="dora", dora_fast_norm=True,
                           targets=list(KREA2_DEFAULT_LORA_TARGETS))
        inj.inject(model)
        self.assertIn("first", inj.injected)
        # 属性透传
        self.assertEqual(model.first.in_features, 16 * 2 * 2)
        model.train()

        device, dtype = "cuda", torch.bfloat16
        lat_shapes = [(4, 4), (6, 8)]
        text_lens = [5, 3]
        toks, grids, vseq, crosses = [], [], [], []
        torch.manual_seed(1)
        for (h, w), L in zip(lat_shapes, text_lens):
            lat = torch.randn(1, 16, 1, h, w, device=device, dtype=dtype)
            tok, grid, _m, _s = model.patchify_latents_to_tokens(lat)
            toks.append(tok)
            grids.append(grid)
            vseq.append(tok.shape[1])
            crosses.append(torch.randn(1, L, 3, 64, device=device, dtype=dtype))
        tokens = torch.cat(toks, dim=1)
        grid = torch.cat(grids, dim=2)
        cross_packed = torch.cat(crosses, dim=1)
        t_g = torch.tensor([0.3, 0.8], device=device)

        with torch.autocast("cuda", dtype=dtype):
            out = model.forward_packed_navit(
                tokens, t_g, cross_packed, grid, vseq, text_lens,
                use_checkpoint=True,
            )
            # 块外层（first / txtfusion / txtmlp）纳入 checkpoint 后前向应与
            # 不 checkpoint 逐 bit 一致（同一批 kernel 只是重算时机不同）
            out_nock = model.forward_packed_navit(
                tokens, t_g, cross_packed, grid, vseq, text_lens,
                use_checkpoint=False,
            )
        self.assertTrue(torch.equal(out, out_nock),
                        "use_checkpoint 开/关前向应逐 bit 一致")
        self.assertEqual(out.shape[1], sum(vseq))
        loss = out.float().square().mean()
        loss.backward()
        for name, lin in inj.injected.items():
            for p in lin.adapter.parameters():
                if p.grad is not None:
                    self.assertTrue(
                        torch.isfinite(p.grad.float()).all(), f"{name} grad 非有限")


if __name__ == "__main__":
    unittest.main()
