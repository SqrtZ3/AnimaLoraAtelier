"""Krea2 对拍闸门 · 第一步（torch 侧）：小构型随机权重跑
`models/krea2_modeling.py` 的单流 MMDiT，把参考量落盘。

配套 `check_krea2_parity.py`（jax 侧）。与 anima 闸门同一套方法论：
**数学对不对用 fp32 + 随机权重就能判**（结构 parity 不需要 24GB 真权重；
真权重只影响数值分布，不改算子拓扑）。

产出（`_ref/krea2/`）：
  `tiny.safetensors`   小构型全量权重（fp32，jax 侧直接走
                       `krea2_jax.load_safetensors_krea2` 读 —— 顺带闸了加载器）
  `krea2_ref.npz`      输入 / 各 tap 中间量 / 最终输出 /  packed 布局描述
  `krea2_ref_lora.npz` 一组合成 LoRA（down/up/alpha）+ 挂 LoRA 后的输出

packed 布局（故意不等长 + 带量化填充，填充错误会立刻显形）：
  图0 网格 4x6=24 token、caption 5 token；图1 网格 3x5=15、caption 9。
  jax 侧量化：text→8 的倍数 [8, 16]、image→16 的倍数 [32, 16]，
  combined 段长 [40, 32]，budget 72。torch 侧只打包有效 token（无填充），
  比对时按"实位映射"对齐（见 check_krea2_parity.py 的 REAL_IDX）。

用法（torch 解释器，见 tests/README.md）：
    python dump_krea2_ref.py [--out 目录]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from _upstream import upstream                                 # noqa: E402
REPO = upstream("models/krea2_modeling.py 的 SingleStreamDiT 参考实现")

from models.krea2_modeling import (SingleMMDiTConfig, SingleStreamDiT,  # noqa: E402
                                   set_packed_attention_backend)

DT = torch.float32

# 小构型：headdim=64（rope axes [16,24,24]，三轴均为偶数，合法）、GQA 4:1。
CFG = SingleMMDiTConfig(
    features=256, tdim=64, txtdim=128, heads=4, kvheads=1,
    multiplier=2, layers=3, patch=2, channels=16,
    txtheads=2, txtkvheads=2, txtlayers=3,
)

GRIDS = [(4, 6), (3, 5)]          # 每图的 (row, col) patch 网格
TEXT_SEQLENS = [5, 9]
T_VALS = [0.3, 0.7]

# 合成 LoRA：覆盖四个区域 + 单例，rank=3 alpha=6（scale=2，顺便闸缩放）。
LORA_RANK, LORA_ALPHA = 3, 6.0
LORA_TARGETS = [
    "blocks.0.attn.wq", "blocks.1.mlp.gate", "blocks.2.attn.wo",
    "txtfusion.layerwise_blocks.1.attn.wv",
    "txtfusion.refiner_blocks.0.attn.wo", "txtfusion.projector",
    "txtmlp.1", "tmlp.0", "tproj.1", "first", "last.linear",
]


def build_model() -> SingleStreamDiT:
    torch.manual_seed(1234)
    model = SingleStreamDiT(CFG).to(DT)
    # RMSNorm scale / mod.lin 零初始化会让一大批算子退化成恒等/常数，
    # 对拍拍不出错位 —— 全部换成小随机（真权重里它们也不是 0）。
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith(".scale") or name.endswith("mod.lin") \
                    or name.endswith("modulation.lin"):
                p.copy_(torch.randn_like(p) * 0.1)
    model.eval()
    return model


def make_inputs():
    g = torch.Generator().manual_seed(567)
    lat0 = torch.randn(1, 16, 1, GRIDS[0][0] * 2, GRIDS[0][1] * 2, generator=g).to(DT)
    lat1 = torch.randn(1, 16, 1, GRIDS[1][0] * 2, GRIDS[1][1] * 2, generator=g).to(DT)
    n_txt = sum(TEXT_SEQLENS)
    txt = torch.randn(1, n_txt, CFG.txtlayers, CFG.txtdim, generator=g).to(DT) * 0.5
    t = torch.tensor(T_VALS, dtype=DT)
    return lat0, lat1, txt, t


def packed_inputs(model, lat0, lat1):
    """两图各自的 patch token / 网格（torch packed 契约的载荷）。"""
    toks, grids, vseq = [], [], []
    for lat, (gh, gw) in ((lat0, GRIDS[0]), (lat1, GRIDS[1])):
        tok, grid, _mask, _size = model.patchify_latents_to_tokens(lat)
        toks.append(tok[0])                       # [N, 64]
        grids.append(grid[0])                     # [2, N]
        vseq.append(tok.shape[1])
    tokens = torch.cat(toks, dim=0).unsqueeze(0)  # [1, 39, 64]
    grid = torch.cat(grids, dim=1).unsqueeze(0)   # [1, 2, 39]
    return tokens, grid, vseq


def run_taps(model, tokens, grid, vseq, txt, t):
    """重走 forward_packed_navit 的装配，把每一级中间量都留下来。"""
    taps = {}
    with torch.no_grad():
        txt_bias = __import__("models.krea2_modeling", fromlist=["_SegLens"])._SegLens(
            TEXT_SEQLENS)
        txt_h = model.txtfusion(txt, mask=txt_bias)          # [1, 14, 128]
        taps["txtfusion_out"] = txt_h[0]
        txt_out = model.txtmlp(txt_h)                        # [1, 14, 256]
        taps["txt_stack_out"] = txt_out[0]
        img = model.first(tokens)                            # [1, 39, 256]
        taps["img_embed"] = img[0]

        t_flat = t.reshape(-1).float()
        t_vec = model.tmlp(__import__("models.krea2_modeling", fromlist=["temb"])
                           .temb(t_flat, CFG.tdim, dtype=DT))     # (2,1,256)
        taps["t_vec"] = t_vec[:, 0, :]
        tvec = model.tproj(t_vec[:, 0, :].unsqueeze(0))           # (1,2,6·256)
        taps["tvec6"] = tvec[0]

        # 逐图 [txt_i ; img_i] 组装（与 forward_packed_navit 同序）
        seg_lens = [tl + vl for tl, vl in zip(TEXT_SEQLENS, vseq)]
        total = sum(seg_lens)
        parts, pos = [], torch.zeros(1, total, 3)
        t_off = v_off = c_off = 0
        img_index = torch.empty(sum(vseq), dtype=torch.long)
        i_off = 0
        for i, (tl, vl) in enumerate(zip(TEXT_SEQLENS, vseq)):
            parts.append(txt_out[:, t_off:t_off + tl])
            parts.append(img[:, v_off:v_off + vl])
            img_index[i_off:i_off + vl] = torch.arange(c_off + tl, c_off + tl + vl)
            pos[0, c_off + tl:c_off + tl + vl, 1] = grid[0, 0, v_off:v_off + vl].float()
            pos[0, c_off + tl:c_off + tl + vl, 2] = grid[0, 1, v_off:v_off + vl].float()
            t_off += tl
            v_off += vl
            c_off += tl + vl
            i_off += vl
        combined = torch.cat(parts, dim=1)                 # [1, 53, 256]
        taps["combined_in"] = combined[0]
        counts = torch.tensor(seg_lens)
        mod_index = torch.repeat_interleave(torch.arange(2), counts)
        freqs = model.posemb(pos)

        from models.krea2_modeling import _SegLens
        self_bias = _SegLens(seg_lens)
        for i, block in enumerate(model.blocks):
            combined = block(combined, tvec, freqs, self_bias, mod_index=mod_index)
            taps[f"block{i}_out"] = combined[0]
        out = model.last.forward_mod(combined, t_vec[:, 0, :].unsqueeze(0), mod_index)
        taps["out"] = out[0, img_index, :]
    return taps


def attach_lora(model, loras):
    """把合成 LoRA 挂到指定 Linear 上（trainer/lora.py:115 的 delta 语义：
    y += ((x @ downᵀ) @ upᵀ)·(alpha/rank)，down [r,in]、up [out,r]）。"""
    mods = dict(model.named_modules())
    for name, (down, up, alpha) in loras.items():
        lin = mods[name]
        scale = alpha / down.shape[0]
        orig_fwd = lin.forward
        d, u, s = down.to(DT), up.to(DT), scale

        def make(orig, d, u, s):
            return lambda x: orig(x) + ((x @ d.T) @ u.T) * s
        lin.forward = make(orig_fwd, d, u, s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "_ref" / "krea2"))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_packed_attention_backend("sdpa_seg")    # 逐段 dense SDPA = 块对角参考数学
    model = build_model()
    lat0, lat1, txt, t = make_inputs()
    tokens, grid, vseq = packed_inputs(model, lat0, lat1)

    # ── 基线（无 LoRA）────────────────────────────────────────────────────────
    taps = run_taps(model, tokens, grid, vseq, txt, t)
    with torch.no_grad():
        ref = model.forward_packed_navit(tokens, t, txt, grid, vseq, TEXT_SEQLENS,
                                         use_checkpoint=False)
    np.testing.assert_allclose(taps["out"].numpy(), ref[0].numpy(), rtol=0, atol=0,
                               err_msg="run_taps 与 forward_packed_navit 不一致")
    print(f"[tap] forward_packed_navit ≡ 手动装配，逐 bit 一致，out {tuple(ref.shape)}")

    # 权重落 safetensors（jax 侧走 load_safetensors_krea2 读，顺带闸加载器）。
    # **写入自描述构型**：非发布构型没法从形状推断（推断假定 headdim=128，
    # 自定义构型会静默猜错 —— 真机踩过），krea2_jax 的加载器优先读这个键。
    import dataclasses as _dc
    from safetensors.torch import save_file
    sd = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(sd, str(out_dir / "tiny.safetensors"),
              metadata={"krea2_config": json.dumps(_dc.asdict(CFG))})

    np.savez(
        out_dir / "krea2_ref.npz",
        tokens=tokens[0].numpy(), grid=grid[0].numpy(), txt=txt[0].numpy(),
        t=t.numpy(), vseq=np.asarray(vseq), tseq=np.asarray(TEXT_SEQLENS),
        grids=np.asarray(GRIDS),
        **{k: v.numpy() for k, v in taps.items()},
    )

    # ── LoRA 档 ──────────────────────────────────────────────────────────────
    g = torch.Generator().manual_seed(99)
    loras = {}
    mods = dict(model.named_modules())
    for name in LORA_TARGETS:
        lin = mods[name]
        down = torch.randn(LORA_RANK, lin.in_features, generator=g) * 0.1
        up = torch.randn(lin.out_features, LORA_RANK, generator=g) * 0.1
        loras[name] = (down, up, LORA_ALPHA)
    attach_lora(model, loras)
    taps_lora = run_taps(model, tokens, grid, vseq, txt, t)
    np.savez(
        out_dir / "krea2_ref_lora.npz",
        **{f"lora/{k}/down": v[0].numpy() for k, v in loras.items()},
        **{f"lora/{k}/up": v[1].numpy() for k, v in loras.items()},
        lora_alpha=np.asarray(LORA_ALPHA), lora_rank=np.asarray(LORA_RANK),
        **{k: v.numpy() for k, v in taps_lora.items()},
    )
    print(f"[ok] 写出 {out_dir}/tiny.safetensors + krea2_ref.npz + krea2_ref_lora.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
