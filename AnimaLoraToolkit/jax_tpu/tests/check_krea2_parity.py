"""Krea2 对拍闸门 · 第二步（jax 侧）：加载 dump_krea2_ref.py 的产物，
用 krea2_jax 重跑同一个小构型 packed 前向，逐级比对。

两级判据：
  ① 逐 tap（txtfusion / txtmlp / first / t_vec / 逐块 / 最终输出）——定位偏差层
  ② LoRA 档（四个区域 + 单例各挂一个合成 LoRA）——闸适配器注入点与区域拆分

packed 布局（与 dump 脚本同一套）：combined 段 [40, 32]（text 量化到 8、
image 量化到 16），budget 72；torch 参考只有 53 个有效 token（无填充），
按 REAL_IDX 映射比对 —— **填充 token 被静默漏 mask 的话这里立刻显形**。

用法（jax 解释器）：
    python check_krea2_parity.py [--ref 目录] [--tol 1e-4]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax.numpy as jnp                                      # noqa: E402
import anima_jax as A                                        # noqa: E402
import adapters as AD                                        # noqa: E402
import krea2_jax as K                                        # noqa: E402

TXT_Q = (8, 16)                    # 每图 text 槽长（8 的倍数）
IMG_Q = (32, 16)                   # 每图 image 槽长（16 的倍数）
PAD = -1


def rel_err(got, exp) -> float:
    got, exp = np.asarray(got, np.float64), np.asarray(exp, np.float64)
    if got.shape != exp.shape:
        return float("nan")
    return float(np.abs(got - exp).max() / max(np.abs(exp).max(), 1e-12))


def build_layout(ref):
    """由数据集几何推出全部索引数组（与 krea2 打包器的规则同构，这里是手写固定值）。"""
    vseq, tseq = [int(x) for x in ref["vseq"]], [int(x) for x in ref["tseq"]]
    grids = ref["grids"]
    seg_lens = [tq + iq for tq, iq in zip(TXT_Q, IMG_Q)]
    budget = sum(seg_lens)

    txt_pos, img_pos = [], []
    rows = np.zeros(budget, np.int32)
    cols = np.zeros(budget, np.int32)
    mod_index = np.zeros(budget, np.int32)
    seg_self = np.full(budget, PAD, np.int32)
    off = 0
    for i, (tq, iq) in enumerate(zip(TXT_Q, IMG_Q)):
        txt_pos += list(range(off, off + tq))
        img_pos += list(range(off + tq, off + tq + iq))
        mod_index[off:off + tq + iq] = i
        # text 实位
        seg_self[off:off + tseq[i]] = i
        # image 实位（网格坐标写进 combined 序列）
        gh, gw = int(grids[i][0]), int(grids[i][1])
        rr, cc = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
        img_off = off + tq
        seg_self[img_off:img_off + gh * gw] = i
        rows[img_off:img_off + gh * gw] = rr.reshape(-1)
        cols[img_off:img_off + gh * gw] = cc.reshape(-1)
        off += tq + iq

    # torch 53 个有效位 ↔ jax 位置的映射（[txt0; img0; txt1; img1] 各段实部）
    real = []
    off = 0
    for i, (tq, iq) in enumerate(zip(TXT_Q, IMG_Q)):
        real += list(range(off, off + tseq[i]))                    # text 实位
        real += list(range(off + tq, off + tq + vseq[i]))          # image 实位
        off += tq + iq
    # txt 流（refiner）精细段号
    txt_fine = np.concatenate([np.full(tq, PAD, np.int32) for tq in TXT_Q])
    off = 0
    for i, tq in enumerate(TXT_Q):
        txt_fine[off:off + tseq[i]] = i
        off += tq
    # 输出（image 流）的实位
    img_real = []
    off = 0
    for i, iq in enumerate(IMG_Q):
        img_real += list(range(off, off + vseq[i]))
        off += iq
    return {
        "seg_lens": seg_lens, "budget": budget,
        "txt_pos": np.asarray(txt_pos, np.int32),
        "img_pos": np.asarray(img_pos, np.int32),
        "rows": rows, "cols": cols, "mod_index": mod_index,
        "seg_self": seg_self, "txt_fine": txt_fine,
        "real": np.asarray(real), "img_real": np.asarray(img_real),
    }


def dense_blockdiag(q_seg_lens, fine):
    """combined/refiner 的稠密参考注意力：粗粒度块对角 AND 精细段号。"""
    coarse = np.repeat(np.arange(len(q_seg_lens)), q_seg_lens)
    allow = (coarse[:, None] == coarse[None, :]) & (fine[:, None] == fine[None, :])
    bias = jnp.asarray(np.where(allow, 0.0, -np.inf), dtype=jnp.float32)
    return lambda q, k, v: K.attention_dense_gqa(q, k, v, bias)


def manual_forward(params, cfg, ref, lay, loras=None, remat="none"):
    """重走 forward_packed 的装配（逐块 tap 都留下来）。"""
    txt_stack = jnp.asarray(ref["txt_stack_pad"])
    img_tokens = jnp.asarray(ref["img_tokens_pad"])
    t = jnp.asarray(ref["t"])

    main_fn = dense_blockdiag(lay["seg_lens"], lay["seg_self"])
    rf_fn = dense_blockdiag(TXT_Q, lay["txt_fine"])
    lw_fn = lambda q, k, v: K.attention_dense_gqa(q, k, v)
    sg = None if loras is None else loras.get("single")

    h = K.text_fusion(params["txtfusion"], cfg, txt_stack, lw_fn, rf_fn,
                      loras=loras, remat=remat)
    taps = {"txtfusion_out": np.asarray(h)}
    txt = K.txtmlp_forward(params, h, loras=sg)
    taps["txt_stack_out"] = np.asarray(txt)

    img = K.dense(img_tokens, params["first"]["w"], _lora_sg(sg, "first"),
                  b=params["first"]["b"])
    taps["img_embed"] = np.asarray(img)

    t_vec, tvec6 = K.tvec_forward(
        {k: params[k] for k in ("tmlp0", "tmlp2", "tproj")}, cfg, t, jnp.float32,
        loras=sg)
    taps["t_vec"], taps["tvec6"] = np.asarray(t_vec), np.asarray(tvec6)

    B = lay["budget"]
    combined = jnp.zeros((B, cfg.features), jnp.float32)
    combined = combined.at[lay["txt_pos"]].set(txt)
    combined = combined.at[lay["img_pos"]].set(img)
    taps["combined_in"] = np.asarray(combined)

    mod_bcast = lambda v: jnp.take(v, lay["mod_index"], axis=0)
    cos, sin = K.rope_cos_sin(jnp.asarray(lay["rows"]), jnp.asarray(lay["cols"]),
                              cfg.rope_axes, cfg.theta)
    for i, p in enumerate(params["blocks"]):
        lo_i = None if loras is None else A._slice_ctx(loras["blocks"], i)
        combined = K.block_forward(combined, p, cfg, tvec6, mod_bcast, cos, sin,
                                   main_fn, loras=lo_i, layer=None)
        taps[f"block{i}_out"] = np.asarray(combined)
    out = K.last_forward(combined, params["last"], cfg, t_vec, mod_bcast, loras=sg)
    taps["out"] = np.asarray(out[jnp.asarray(lay["img_pos"])])
    return taps


def _lora_sg(sg, key):
    return None if sg is None else A._lora(sg, key)


def build_lora_ctxs(ref_l, cfg):
    """npz 里的合成 LoRA -> 区域 LoraCtx（a=downᵀ、b=upᵀ，scale=alpha/rank）。"""
    rank = int(ref_l["lora_rank"])
    alpha = float(ref_l["lora_alpha"])
    shapes = K.lora_target_shapes(cfg)
    trainable, consts = {}, {}
    names = [n for n in ref_l if n.startswith("lora/") and n.endswith("/down")]
    for nk in names:
        mod = nk[len("lora/"):-len("/down")]
        down, up = ref_l[nk], ref_l[f"lora/{mod}/up"]
        target, idx = _to_stacked(mod, shapes)
        count, i_f, o_f = shapes[target]
        a = np.zeros((count, i_f, rank), np.float32)
        b = np.zeros((count, rank, o_f), np.float32)
        a[idx] = down.T
        b[idx] = up.T
        ent = trainable.setdefault(target, {"a": a, "b": b})
        consts[target] = {"scale": np.full(count, alpha / rank, np.float32)}
    acfg = AD.AdapterConfig(kind="lora", rank=rank, alpha=alpha)
    return K.split_ctx_regions(trainable, consts, None, acfg)


def _to_stacked(mod: str, shapes):
    """torch 模块路径 -> (堆叠 target, 位置)。与 K.expand_name 互逆。"""
    for s in K._STACKS:
        if mod.startswith(s + "."):
            rest = mod[len(s) + 1:]
            idx, _, tail = rest.partition(".")
            return f"{s}.{tail}", int(idx)
    return mod, 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=str(Path(__file__).resolve().parent / "_ref" / "krea2"))
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()
    ref_dir = Path(args.ref)
    ref = dict(np.load(ref_dir / "krea2_ref.npz"))
    ref_l = dict(np.load(ref_dir / "krea2_ref_lora.npz"))

    # 不显式给 cfg —— 走 safetensors __metadata__ 的自描述构型（dump 写入的
    # krea2_config；非发布构型没法从形状推断，真机踩过"猜错不报错"）。
    params, cfg = K.load_safetensors_krea2(str(ref_dir / "tiny.safetensors"),
                                           dtype=jnp.float32)
    assert (cfg.features, cfg.heads, cfg.kvheads, cfg.layers, cfg.txtheads) == \
        (256, 4, 1, 3, 2), f"自描述构型读错：{cfg}"
    print(f"构型 F={cfg.features} heads={cfg.heads}/{cfg.kvheads} "
          f"layers={cfg.layers} txt={cfg.txtdim}x{cfg.txtlayers}层")

    # packed 载荷补齐到量化槽（dump 只存了有效 token）
    tseq = [int(x) for x in ref["tseq"]]
    txt_pad = np.zeros((sum(TXT_Q),) + ref["txt"].shape[1:], np.float32)
    off = 0
    for i, tq in enumerate(TXT_Q):
        txt_pad[off:off + tseq[i]] = ref["txt"][sum(tseq[:i]):sum(tseq[:i + 1])]
        off += tq
    vseq = [int(x) for x in ref["vseq"]]
    img_pad = np.zeros((sum(IMG_Q), ref["tokens"].shape[1]), np.float32)
    off = 0
    for i, iq in enumerate(IMG_Q):
        img_pad[off:off + vseq[i]] = ref["tokens"][sum(vseq[:i]):sum(vseq[:i + 1])]
        off += iq
    ref["txt_stack_pad"] = txt_pad
    ref["img_tokens_pad"] = img_pad

    lay = build_layout(ref)
    real, img_real = lay["real"], lay["img_real"]

    taps_at = lambda got, exp, name: (
        rel_err(got[real] if got.ndim == 2 and got.shape[0] == lay["budget"] else got,
                exp), name)
    fails = []

    def cmp(got, exp, name, tol=None):
        r = rel_err(got, exp)
        ok = r <= (tol or args.tol)
        print(f"  [{'OK ' if ok else 'FAIL'}] {name:<28} rel={r:.2e}")
        if not ok:
            fails.append(name)
        return r

    print("① 基线逐 tap（fp32，实位映射）")
    taps = manual_forward(params, cfg, ref, lay)
    cmp(taps["txtfusion_out"][real[:len(real) - sum(vseq)] if False else _txt_real(lay, tseq)],
        ref["txtfusion_out"], "txtfusion_out")
    cmp(taps["txt_stack_out"][_txt_real(lay, tseq)], ref["txt_stack_out"], "txt_stack_out")
    cmp(taps["img_embed"][img_real], ref["img_embed"], "img_embed")
    cmp(taps["t_vec"], ref["t_vec"], "t_vec")
    cmp(taps["tvec6"], ref["tvec6"], "tvec6")
    cmp(taps["combined_in"][real], ref["combined_in"], "combined_in")
    for i in range(cfg.layers):
        cmp(taps[f"block{i}_out"][real], ref[f"block{i}_out"], f"block{i}_out")
    cmp(taps["out"][img_real], ref["out"], "out(最终)")

    print("② forward_packed 整函数 ≡ 手动装配（含 scan 路径）")
    main_fn = dense_blockdiag(lay["seg_lens"], lay["seg_self"])
    rf_fn = dense_blockdiag(TXT_Q, lay["txt_fine"])
    lw_fn = lambda q, k, v: K.attention_dense_gqa(q, k, v)
    out_full = K.forward_packed(
        params, cfg, jnp.asarray(ref["img_tokens_pad"]), jnp.asarray(ref["t"]),
        jnp.asarray(ref["txt_stack_pad"]), jnp.asarray(lay["rows"]),
        jnp.asarray(lay["cols"]), jnp.asarray(lay["mod_index"]),
        jnp.asarray(lay["txt_pos"]), jnp.asarray(lay["img_pos"]),
        main_fn, rf_fn, lw_fn, loras=None, remat="full", mesh_axis=None)
    cmp(np.asarray(out_full), taps["out"], "forward_packed(展开) ≡ 手动", tol=0.0 + 1e-12)
    params_sc = K.stack_blocks(params)
    out_scan = K.forward_packed(
        params_sc, cfg, jnp.asarray(ref["img_tokens_pad"]), jnp.asarray(ref["t"]),
        jnp.asarray(ref["txt_stack_pad"]), jnp.asarray(lay["rows"]),
        jnp.asarray(lay["cols"]), jnp.asarray(lay["mod_index"]),
        jnp.asarray(lay["txt_pos"]), jnp.asarray(lay["img_pos"]),
        main_fn, rf_fn, lw_fn, loras=None, remat="full", mesh_axis=None)
    # scan 与展开是同一份数学的两份编译：XLA 融合顺序不同，fp32 下有 1e-7 量级
    # 噪声（anima 那边逐 bit 是巧合，不是契约）；1e-6 内都判等价。
    cmp(np.asarray(out_scan), taps["out"], "forward_packed(scan) ≡ 手动", tol=1e-6)

    print("③ LoRA 档（四区域 + 单例）")
    loras = build_lora_ctxs(ref_l, cfg)
    taps_l = manual_forward(params, cfg, ref, lay, loras=loras)
    cmp(taps_l["txt_stack_out"][_txt_real(lay, tseq)], ref_l["txt_stack_out"],
        "lora: txt_stack_out")
    cmp(taps_l["t_vec"], ref_l["t_vec"], "lora: t_vec")
    cmp(taps_l["tvec6"], ref_l["tvec6"], "lora: tvec6")
    for i in range(cfg.layers):
        cmp(taps_l[f"block{i}_out"][real], ref_l[f"block{i}_out"], f"lora: block{i}_out")
    cmp(taps_l["out"][img_real], ref_l["out"], "lora: out(最终)")

    if fails:
        print(f"\n*** 未过：{fails} ***")
        return 1
    print("\n*** 通过 ***")
    return 0


def _txt_real(lay, tseq):
    """txt 流（refiner 产物）的实位下标。"""
    out = []
    off = 0
    for i, tq in enumerate(TXT_Q):
        out += list(range(off, off + tseq[i]))
        off += tq
    return np.asarray(out)


if __name__ == "__main__":
    raise SystemExit(main())
