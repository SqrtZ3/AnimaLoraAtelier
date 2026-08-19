"""Krea 2 (K2) 训练移植的等价性测试。

覆盖：
  1. packed navit ≡ 逐图 dense forward（块对角隔离 / 逐图 t / RoPE / varlen 文本，
     xformers varlen 与 SDPA 回退两条路径）
  2. 仓库 forward ≡ 官方 mmdit.py forward（同权重同输入；官方实现从 scratchpad 或
     环境变量 KREA2_OFFICIAL_DIR 加载，找不到则跳过该项）
  3. state dict key 与官方逐一相同（strict load 双向）
  4. KREA2_DEFAULT_LORA_TARGETS 恰好覆盖全部 nn.Linear（官方推荐"全部 264 Linear"口径）
  5. 分辨率感知 timestep shift 数学（mu@1024²≈0.906、逐图向量 == 标量路径）

Needs CUDA（xformers 项另需 xformers）；无 GPU 时跳过 GPU 项。
本地跑法：
    D:\\ArtificialIntelligence\\ComfyUI-aki-v1.5\\python\\python.exe -m pytest \\
        AnimaLoraToolkit/tests/test_krea2_modeling.py -v
"""
import math
import os
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
try:
    import xformers  # noqa: F401
    HAS_XFORMERS = True
except Exception:
    HAS_XFORMERS = False


def _tiny_config():
    # features/heads → headdim 64（xformers varlen 支持 64/128）；GQA 2/1；
    # txtfusion 3 层输入堆叠、txt 头 2/1；patch 2、latent 16ch 与发布模型一致。
    return k2.SingleMMDiTConfig(
        features=128, tdim=32, txtdim=64, heads=2, kvheads=1,
        multiplier=2, layers=2, patch=2, channels=16,
        txtheads=2, txtkvheads=1, txtlayers=3,
    )


def _tiny_model(dtype, device="cuda"):
    torch.manual_seed(0)
    m = k2.SingleStreamDiT(_tiny_config())
    # 官方权重零初始化的调制/norm 会让路径退化（gate=0 → 恒等），随机化以获得非平凡前向
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    return m.to(device=device, dtype=dtype).eval()


def _make_pack(model, dtype, device="cuda"):
    """3 张异构图 + varlen caption 的打包输入与逐图 dense 参照输入。"""
    latent_shapes = [(4, 4), (6, 8), (4, 6)]     # N = 4, 12, 6
    text_lens = [5, 9, 3]
    timesteps = [0.2, 0.6, 0.9]
    cfg = model.config

    lat_list, cross_list = [], []
    tokens_list, grid_list, vseq = [], [], []
    torch.manual_seed(1)
    for (h, w), L in zip(latent_shapes, text_lens):
        lat = torch.randn(1, 16, 1, h, w, device=device, dtype=dtype)
        lat_list.append(lat)
        cross_list.append(torch.randn(1, L, cfg.txtlayers, cfg.txtdim, device=device, dtype=dtype))
        tok, grid, _m, _s = model.patchify_latents_to_tokens(lat)
        tokens_list.append(tok)
        grid_list.append(grid)
        vseq.append(tok.shape[1])

    tokens = torch.cat(tokens_list, dim=1)
    grid = torch.cat(grid_list, dim=2)
    cross_packed = torch.cat(cross_list, dim=1)
    t = torch.tensor(timesteps, device=device, dtype=torch.float32)
    return lat_list, cross_list, tokens, grid, cross_packed, vseq, text_lens, t


@unittest.skipUnless(HAS_TORCH and HAS_CUDA, "needs torch + CUDA")
class Krea2PackedEqualsDenseTests(unittest.TestCase):
    def _dense_per_image(self, model, lat_list, cross_list, t):
        outs = []
        with torch.no_grad():
            for i, (lat, cross) in enumerate(zip(lat_list, cross_list)):
                v = model.forward(lat, t[i: i + 1].view(1, 1), cross)     # [1,C,1,h,w]
                tok, _g, _m, _s = model.patchify_latents_to_tokens(v)
                outs.append(tok)
        return torch.cat(outs, dim=1)                                     # [1, ΣN, 64]

    def _assert_close(self, a, b, atol, label):
        diff = (a.float() - b.float()).abs().max().item()
        self.assertLess(diff, atol, f"{label}: max_abs_diff={diff:.3e} >= {atol}")
        return diff

    def test_packed_navit_equals_dense_sdpa_fallback_fp32(self):
        """SDPA 块对角回退路径 vs 逐图 dense（同 SDPA kernel 族，fp32 紧公差）。"""
        model = _tiny_model(torch.float32)
        lat_list, cross_list, tokens, grid, cross_packed, vseq, tseq, t = _make_pack(model, torch.float32)
        dense = self._dense_per_image(model, lat_list, cross_list, t)

        orig = k2._xformers_available
        k2._xformers_available = lambda: False
        try:
            with torch.no_grad():
                packed = model.forward_packed_navit(tokens, t, cross_packed, grid, vseq, tseq)
        finally:
            k2._xformers_available = orig
        diff = self._assert_close(packed, dense, 5e-5, "sdpa packed vs dense fp32")
        print(f"[krea2] SDPA packed≡dense fp32 max_abs_diff={diff:.3e}")

    @unittest.skipUnless(HAS_XFORMERS, "needs xformers")
    def test_packed_navit_equals_dense_xformers_fp32(self):
        """xformers varlen 路径 vs 逐图 dense（跨 kernel，fp32 宽松公差）。"""
        model = _tiny_model(torch.float32)
        lat_list, cross_list, tokens, grid, cross_packed, vseq, tseq, t = _make_pack(model, torch.float32)
        dense = self._dense_per_image(model, lat_list, cross_list, t)
        with torch.no_grad():
            packed = model.forward_packed_navit(tokens, t, cross_packed, grid, vseq, tseq)
        diff = self._assert_close(packed, dense, 5e-4, "xformers packed vs dense fp32")
        print(f"[krea2] xformers packed≡dense fp32 max_abs_diff={diff:.3e}")

    @unittest.skipUnless(HAS_XFORMERS, "needs xformers")
    def test_packed_checkpoint_matches_no_checkpoint(self):
        """use_checkpoint=True 与 False 逐 bit 一致（推理态，无 dropout）。"""
        model = _tiny_model(torch.float32)
        _l, _c, tokens, grid, cross_packed, vseq, tseq, t = _make_pack(model, torch.float32)
        # 两侧都在 grad 模式下算：SDPA 的 backend 选择可能随 requires_grad 变化，
        # 只隔离 checkpoint 包装这一个变量。
        tokens_a = tokens.clone().requires_grad_(True)
        a = model.forward_packed_navit(tokens_a, t, cross_packed, grid, vseq, tseq, use_checkpoint=False)
        tokens_g = tokens.clone().requires_grad_(True)
        b = model.forward_packed_navit(tokens_g, t, cross_packed, grid, vseq, tseq, use_checkpoint=True)
        self.assertEqual((a.float() - b.detach().float()).abs().max().item(), 0.0)
        # backward 走通（LoRA 场景下的最小健全性）
        b.square().mean().backward()
        self.assertIsNotNone(tokens_g.grad)

    def test_cross_image_isolation(self):
        """改第 2 张图的内容/caption 不得影响第 1、3 张图的 packed 输出。"""
        model = _tiny_model(torch.float32)
        lat_list, cross_list, tokens, grid, cross_packed, vseq, tseq, t = _make_pack(model, torch.float32)
        orig = k2._xformers_available
        k2._xformers_available = lambda: False
        try:
            with torch.no_grad():
                base = model.forward_packed_navit(tokens, t, cross_packed, grid, vseq, tseq)
                tokens2 = tokens.clone()
                n0 = vseq[0]
                tokens2[:, n0: n0 + vseq[1]] += 1.0
                cross2 = cross_packed.clone()
                l0 = tseq[0]
                cross2[:, l0: l0 + tseq[1]] += 1.0
                pert = model.forward_packed_navit(tokens2, t, cross2, grid, vseq, tseq)
        finally:
            k2._xformers_available = orig
        self.assertEqual((base[:, :n0] - pert[:, :n0]).abs().max().item(), 0.0)
        self.assertEqual((base[:, n0 + vseq[1]:] - pert[:, n0 + vseq[1]:]).abs().max().item(), 0.0)
        self.assertGreater((base[:, n0: n0 + vseq[1]] - pert[:, n0: n0 + vseq[1]]).abs().max().item(), 0.0)


@unittest.skipUnless(HAS_TORCH and HAS_CUDA, "needs torch + CUDA")
class Krea2OfficialParityTests(unittest.TestCase):
    """与官方 mmdit.py 的对照（权重 key 双向 strict + 前向数值）。"""

    def _load_official(self):
        cand = os.environ.get("KREA2_OFFICIAL_DIR", "")
        paths = [pathlib.Path(cand)] if cand else []
        # scratchpad 克隆（本仓库调研期的默认位置）
        paths += list(pathlib.Path(os.environ.get("TEMP", "/tmp")).glob(
            "claude/*/*/scratchpad/krea-2"))
        for p in paths:
            if p and (p / "mmdit.py").exists():
                import importlib.util
                # 官方代码带 @torch.compile 装饰器；本地 Windows 无 triton → 全局禁用
                # dynamo 让其退化为 eager（数学不变，只影响 kernel 融合）。
                import torch._dynamo
                torch._dynamo.config.disable = True
                spec = importlib.util.spec_from_file_location("krea2_official_mmdit", p / "mmdit.py")
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod
        return None

    def test_state_dict_and_forward_parity(self):
        official = self._load_official()
        if official is None:
            self.skipTest("官方 krea-2 代码不可用（设 KREA2_OFFICIAL_DIR 指向克隆目录）")

        cfg = _tiny_config()
        ocfg = official.SingleMMDiTConfig(
            features=cfg.features, tdim=cfg.tdim, txtdim=cfg.txtdim, heads=cfg.heads,
            kvheads=cfg.kvheads, multiplier=cfg.multiplier, layers=cfg.layers,
            patch=cfg.patch, channels=cfg.channels,
            txtheads=cfg.txtheads, txtkvheads=cfg.txtkvheads, txtlayers=cfg.txtlayers,
        )
        ours = _tiny_model(torch.bfloat16)
        theirs = official.SingleStreamDiT(ocfg)

        # key 双向 strict：官方 ← 我们 / 我们 ← 官方
        theirs.load_state_dict(ours.state_dict(), strict=True)
        ours.load_state_dict(theirs.state_dict(), strict=True)
        theirs = theirs.to(device="cuda", dtype=torch.bfloat16).eval()

        # 同输入（官方签名：img tokens / context / t / pos / mask；官方内部 pad 到 256 倍数）
        torch.manual_seed(2)
        h, w = 6, 8
        lat = torch.randn(1, 16, 1, h, w, device="cuda", dtype=torch.bfloat16)
        L = 7
        cross = torch.randn(1, L, cfg.txtlayers, cfg.txtdim, device="cuda", dtype=torch.bfloat16)
        t = torch.tensor([0.4], device="cuda")

        with torch.no_grad():
            v_ours = ours.forward(lat, t.view(1, 1), cross)          # [1,16,1,h,w]
            tok_ours, _g, _m, _s = ours.patchify_latents_to_tokens(v_ours)

            img_tok, _g2, _m2, _s2 = ours.patchify_latents_to_tokens(lat)
            h_t, w_t = h // 2, w // 2
            imgids = torch.zeros((h_t, w_t, 3), device="cuda")
            imgids[..., 1] = torch.arange(h_t, device="cuda")[:, None]
            imgids[..., 2] = torch.arange(w_t, device="cuda")[None, :]
            pos = torch.cat(
                [torch.zeros(1, L, 3, device="cuda"),
                 imgids.reshape(1, h_t * w_t, 3)], dim=1)
            mask = torch.ones(1, L + h_t * w_t, dtype=torch.bool, device="cuda")
            tok_theirs = theirs(img=img_tok, context=cross, t=t, pos=pos, mask=mask)

        diff = (tok_ours.float() - tok_theirs.float()).abs().max().item()
        # bf16 + 官方 CUDNN kernel vs 我们默认 SDPA：跨 kernel 公差
        self.assertLess(diff, 5e-2, f"官方前向 vs 仓库前向 max_abs_diff={diff:.3e}")
        print(f"[krea2] official vs port (bf16) max_abs_diff={diff:.3e}")


@unittest.skipUnless(HAS_TORCH, "needs torch")
class Krea2UnitTests(unittest.TestCase):
    def test_patchify_roundtrip(self):
        model = k2.SingleStreamDiT(_tiny_config())
        x = torch.randn(2, 16, 1, 6, 8)
        tok, grid, mask, size = model.patchify_latents_to_tokens(x)
        self.assertEqual(tuple(tok.shape), (2, 12, 64))
        self.assertEqual(tuple(grid.shape), (2, 2, 12))
        back = model.unpatchify_tokens(tok, size)
        self.assertEqual((back - x).abs().max().item(), 0.0)
        # Krea2 输出 token 天然 (c ph pw) 序 → 恒等
        self.assertIs(model._output_tokens_to_patch_tokens(tok, size), tok)

    def test_lora_targets_cover_all_linears(self):
        import torch.nn as nn
        sys.path.insert(0, str(ROOT))
        from trainer.model_family import (
            KREA2_DEFAULT_LORA_TARGETS, KREA2_ATTENTION_ONLY_TARGETS,
        )
        model = k2.SingleStreamDiT(_tiny_config())
        linears = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
        hit = [n for n in linears if any(t in n for t in KREA2_DEFAULT_LORA_TARGETS)]
        self.assertEqual(sorted(hit), sorted(linears),
                         f"默认 targets 漏掉: {sorted(set(linears) - set(hit))}")
        # attention-only（txtfusion 排除前）：主 block attn 5 投影 × layers + txtfusion attn
        attn_hits = [n for n in linears if any(t in n for t in KREA2_ATTENTION_ONLY_TARGETS)]
        blk_hits = [n for n in attn_hits if n.startswith("blocks.")]
        self.assertEqual(len(blk_hits), 5 * model.config.layers)

    def test_lora_default_target_count_matches_author_264(self):
        """发布构型下默认 targets 命中数 == musubi 文档口径的 264 个 Linear。"""
        import torch.nn as nn
        from trainer.model_family import KREA2_DEFAULT_LORA_TARGETS
        with torch.device("meta"):
            model = k2.SingleStreamDiT(k2.KREA2_LARGE_WIDE)
        linears = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
        hit = [n for n in linears if any(t in n for t in KREA2_DEFAULT_LORA_TARGETS)]
        self.assertEqual(len(linears), 264)
        self.assertEqual(len(hit), 264)

    def test_krea2_shift_math(self):
        from trainer.model_family import krea2_mu, krea2_shift_timesteps, krea2_sample_shift
        mu_1024 = krea2_mu((1024 // 16) ** 2)
        self.assertAlmostEqual(mu_1024, 0.9062, places=3)
        self.assertAlmostEqual(math.exp(mu_1024), 2.475, places=2)   # musubi 口径 ≈2.5@1024
        # 标量路径 == 逐图向量路径
        t = torch.tensor([0.1, 0.5, 0.9])
        a = krea2_shift_timesteps(t, 4096.0)
        b = krea2_shift_timesteps(t, [4096, 4096, 4096])
        self.assertLess((a - b).abs().max().item(), 1e-6)
        # α>1 → t 上移（推向高噪端），且保持 (0,1) 内
        self.assertTrue(bool((a > t).all()) and bool((a < 1).all()))
        # 采样 shift 与训练 shift 同源
        self.assertAlmostEqual(krea2_sample_shift(1024, 1024), math.exp(mu_1024), places=6)

    def test_infer_config_from_state_dict(self):
        model = k2.SingleStreamDiT(_tiny_config())
        cfg = k2.infer_config_from_state_dict(model.state_dict())
        self.assertEqual(cfg.layers, 2)
        self.assertEqual(cfg.txtlayers, 3)
        self.assertEqual(cfg.channels, 16)


if __name__ == "__main__":
    unittest.main(verbosity=2)
