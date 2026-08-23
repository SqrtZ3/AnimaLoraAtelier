"""对拍闸门 · 第一步（torch 侧）：用**真权重**跑 PyTorch Anima，把参考量落盘成 npz。

配套 `check_jax_parity.py`（jax 侧）。两边分处不同 venv（本地 torch 环境装不了
新版 jax，jax venv 里没有 torch），所以通过 npz 落盘交换，互不污染。

一次跑出三档参考量，够定位任何偏差：
  ① 整模输出       —— 主判据
  ② 逐算子中间量   —— 偏差落在哪个算子
  ③ 逐块输出       —— 偏差是跨块累积放大，还是某一块突变

**跑在 CPU + fp32**：对拍要的是"数学对不对"，bf16 的舍入噪声会盖住真实偏差。

用法（需要 torch + safetensors 的解释器，见 SETUP.md 的「torch 侧」）：
    export ANIMA_UPSTREAM=/path/to/anima-lora-train/AnimaLoraToolkit
    python dump_torch_ref.py --ckpt <anima-base-v1.0.safetensors> [--out 目录]
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from _upstream import upstream                      # noqa: E402

REPO = upstream("models/anima_modeling_core.py 的 MiniTrainDIT 参考实现")
#: 底模路径。没有默认值 —— 每台机器不一样，猜一个只会报 FileNotFoundError。
#: 用 `--ckpt` 传，或设 `ANIMA_TRANSFORMER`（与 job 侧同名，一处配置两处用）。
DEFAULT_CKPT = os.environ.get("ANIMA_TRANSFORMER", "")
DT = torch.float32

# pack 构型：2 张不等大的图 + 不等长 caption。**故意不等长**——等长会掩盖
# 块对角 mask 的错误（等长时错位仍可能"看起来对"）。
GRIDS = [(8, 12), (6, 6)]      # 每图的 (row, col) patch 网格
TEXT_SEQLENS = [7, 11]


def build_model(acore):
    """构型与 trainer/models.py:145 一致，只把 max_img 缩小以省内存。"""
    return acore.MiniTrainDIT(
        max_img_h=256, max_img_w=256, max_frames=1,
        in_channels=16, out_channels=16,
        patch_spatial=2, patch_temporal=1, concat_padding_mask=True,
        model_channels=2048, num_blocks=28, num_heads=16,
        crossattn_emb_channels=1024,
        pos_emb_cls="rope3d", pos_emb_learnable=True, pos_emb_interpolation="crop",
        use_adaln_lora=True, adaln_lora_dim=256,
        # 这三个**必须**跟 trainer 一致：RoPE 的 theta 要乘 ntk_factor =
        # ratio**(dim/(dim-2))，漏了不报错、只是位置编码频率悄悄变了。
        rope_h_extrapolation_ratio=4.0, rope_w_extrapolation_ratio=4.0,
        rope_t_extrapolation_ratio=1.0,
    ).to(DT).eval()


def load_core():
    spec = importlib.util.spec_from_file_location(
        "acore", str(REPO / "models" / "anima_modeling_core.py"))
    acore = importlib.util.module_from_spec(spec)
    sys.modules["acore"] = acore
    spec.loader.exec_module(acore)
    # 本地无 xformers：块对角走 sdpa_seg（逐段 dense SDPA，与块对角数学等价）
    acore.set_packed_attention_backend("sdpa_seg")
    return acore


def load_weights(model, ckpt):
    from safetensors import safe_open
    sd = {}
    with safe_open(ckpt, framework="pt", device="cpu") as f:
        for k in f.keys():
            # llm_adapter 属于文本侧，TPU 路线上离线算好，不参与 DiT 对拍
            if k.startswith("net.") and "llm_adapter" not in k:
                sd[k[4:]] = f.get_tensor(k).to(DT)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if "llm_adapter" not in m]
    # pos_embedder.* 是 reset_parameters 重算的 buffer，缺失是正常的
    hard = [m for m in missing if not m.startswith("pos_embedder.")]
    if hard or unexpected:
        raise RuntimeError(f"权重不匹配 missing={hard[:5]} unexpected={unexpected[:5]}")
    return len(sd)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT,
                    help="anima 底模 safetensors。缺省读环境变量 ANIMA_TRANSFORMER")
    ap.add_argument("--out", default=str(Path(__file__).parent / "_ref"))
    a = ap.parse_args()
    if not a.ckpt:
        raise SystemExit(
            "[ FATAL ] 需要 anima 底模：--ckpt <anima-base-v1.0.safetensors>\n"
            "  或 export ANIMA_TRANSFORMER=<该文件路径>。\n"
            "  本闸门用**真权重**跑 torch 前向产参考量，随机权重证不了数值口径。")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    acore = load_core()
    model = build_model(acore)
    n = load_weights(model, a.ckpt)
    print(f"载入 {n} 个张量")

    G = len(GRIDS)
    vis = [h * w for h, w in GRIDS]
    txt = list(TEXT_SEQLENS)
    N, L = sum(vis), sum(txt)
    print(f"pack: G={G} visual_seqlens={vis} text_seqlens={txt} N={N} L={L}")

    tokens = torch.randn(1, N, 68, dtype=DT) * 0.5
    ctx = torch.randn(1, L, 1024, dtype=DT) * 0.5
    t = torch.tensor([0.3, 0.85], dtype=DT)
    rows, cols = [], []
    for h, w in GRIDS:
        rr, cc = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        rows.append(rr.reshape(-1))
        cols.append(cc.reshape(-1))
    grid = torch.stack([torch.cat(rows), torch.cat(cols)]).unsqueeze(0).to(DT)

    st = {}
    with torch.no_grad():
        # ── ① 整模 ────────────────────────────────────────────────────────
        st["out"] = model.forward_packed_navit(
            tokens, t, ctx, grid, vis, txt, use_checkpoint=False)[0].numpy()

        # ── ② 逐算子中间量 ────────────────────────────────────────────────
        x = model.x_embedder.proj[1](tokens)
        st["x_embed"] = x[0].numpy()
        st["sincos"] = model.t_embedder[0](t.unsqueeze(1))[:, 0, :].numpy()
        t_emb, adaln = model.t_embedder(t.unsqueeze(1))
        t_emb = model.t_embedding_norm(t_emb)
        st["t_emb"] = t_emb[:, 0, :].numpy()
        st["adaln_lora"] = adaln[:, 0, :].numpy()
        rope = model._packed_rope_from_grid(grid)
        st["rope"] = rope[0, :, 0, 0, :].numpy()      # 存的是**角度**，不是 cos/sin

        mi = torch.repeat_interleave(torch.arange(G), torch.tensor(vis))
        sb = acore._cached_seg_lens(tuple(vis))
        cb = acore._cached_seg_lens(tuple(vis), tuple(txt))
        blk = model.blocks[0]
        sh, sc, _ = (blk.adaln_modulation_self_attn(t_emb[:, 0, :].unsqueeze(0))
                     + adaln[:, 0, :].unsqueeze(0)).chunk(3, dim=-1)
        sel = lambda z: z.index_select(1, mi)
        h = blk.layer_norm_self_attn(x) * (1 + sel(sc)) + sel(sh)
        st["pre_selfattn"] = h[0].numpy()
        q, k, v = blk.self_attn.compute_qkv(h, None, rope_emb=rope)
        st["q"], st["k"], st["v"] = q[0].numpy(), k[0].numpy(), v[0].numpy()
        st["selfattn_out"] = blk.self_attn.compute_attention(
            q, k, v, attn_mask=sb)[0].numpy()

        # ── ③ 逐块 ────────────────────────────────────────────────────────
        xb = x
        for i, b in enumerate(model.blocks):
            xb = b.forward_tokens(
                xb, t_emb[:, 0, :].unsqueeze(0), ctx, rope_emb_L_1_1_D=rope,
                attn_mask=sb, adaln_lora_B_T_3D=adaln[:, 0, :].unsqueeze(0),
                cross_attn_mask=cb, token_wise_mod=True, mod_index=mi)
            st[f"block{i}"] = xb[0].numpy()

    np.savez(out / "anima_ref.npz",
             tokens=tokens[0].numpy(), ctx=ctx[0].numpy(), t=t.numpy(),
             rows=grid[0, 0].numpy().astype(np.int32),
             cols=grid[0, 1].numpy().astype(np.int32),
             vis=np.array(vis), txt=np.array(txt), **st)
    print(f"已写 {out / 'anima_ref.npz'}（{len(st)} 项参考量）")
    print(f"整模输出 mean={st['out'].mean():+.6f} std={st['out'].std():.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
