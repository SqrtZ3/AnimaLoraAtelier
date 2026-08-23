"""闸门 ⑥：导出的 safetensors 能被 torch 侧读回，且 ΔW 与 JAX 侧逐元素一致。

导出这一步的静默错很典型：转置漏了、alpha 没写、bf16 用截断而不是就近舍入——
文件都能正常加载，只是模型行为不对。

判据：
  E1 键名/形状符合仓库规范（lora.py:2054 `lora_unet_<path with _>`；
     down=(rank,in) / up=(out,rank)）
  E2 `safetensors` 官方库能读回（说明字节格式没写坏）
  E3 由 down/up/alpha 重建的 ΔW 与 JAX 侧 `scaling * a @ b` 逐元素一致
     （阈值按 bf16 存储精度定，取相对 4e-3）
  E4 bf16 是**就近舍入**而不是截断（用一个刻意落在半个 ulp 上的值判）

两步跑：
    <jax-python>   check_export.py --write     # 产出 _ref/export_test.safetensors
    <torch-python> check_export.py             # 读回并校验
"""

import argparse
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "_ref" / "export_test.safetensors"
NPY = Path(__file__).parent / "_ref" / "export_ref.npz"
RANK, ALPHA = 8, 4.0


def write():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import jax
    import jax.numpy as jnp
    import anima_jax as A
    import export as X

    cfg = A.AnimaConfig(num_blocks=2)
    lora = A.init_lora(jax.random.PRNGKey(0), cfg, RANK, dtype=jnp.float32)
    # B 是零初始化 -> ΔW 恒 0，验不出转置错。给它灌上随机值再导。
    k = jax.random.split(jax.random.PRNGKey(1), len(lora))
    lora = {n: {"a": m["a"], "b": jax.random.normal(k[i], m["b"].shape, jnp.float32) * 0.1}
            for i, (n, m) in enumerate(sorted(lora.items()))}
    X.export_lora(OUT, lora, RANK, ALPHA, {"src": "check_export"})

    scaling = A.lora_scaling(RANK, ALPHA)
    ref = {"lora_unet_" + n.replace(".", "_"):
           np.asarray(scaling * (m["a"] @ m["b"]), np.float32)      # (in, out)
           for n, m in lora.items()}
    np.savez(NPY, scaling=np.float32(scaling), **ref)
    print(f"已写 {OUT}（{len(lora)} 层）与 {NPY}")
    return 0


def check():
    import torch
    from safetensors.torch import load_file
    if not OUT.exists():
        print(f"找不到 {OUT} —— 先在 jax 环境跑 `--write`")
        return 2
    sd = {k: v.float().numpy() for k, v in load_file(str(OUT)).items()}
    ref = np.load(NPY)
    scaling = float(ref["scaling"])
    bad = []

    print("E1/E2 键名、形状、可读回：")
    names = sorted({k.rsplit(".", 2)[0] for k in sd})
    print(f"  {len(names)} 层，样例键 {names[0]}")
    if not names[0].startswith("lora_unet_blocks_0_"):
        bad.append("键名前缀不符合 lora_unet_ 规范")

    print("\nE3 ΔW 重建一致性：")
    worst, worst_n = 0.0, ""
    for n in names:
        down = sd[f"{n}.lora_down.weight"].astype(np.float32)   # (rank, in)
        up = sd[f"{n}.lora_up.weight"].astype(np.float32)       # (out, rank)
        a_ = float(sd[f"{n}.alpha"])
        if down.shape[0] != RANK or up.shape[1] != RANK:
            bad.append(f"{n}: 形状不符 down{down.shape} up{up.shape}（转置漏了？）")
            continue
        if abs(a_ - ALPHA) > 1e-6:
            bad.append(f"{n}: alpha={a_} != {ALPHA}")
        dw = (a_ / RANK) * (down.T @ up.T)                       # (in, out)
        exp = ref[n]          # 参考量用同样的 mangled 名存，不靠顺序配对
        r = float(np.abs(dw - exp).max() / max(np.abs(exp).max(), 1e-12))
        if r > worst:
            worst, worst_n = r, n
    ok3 = worst < 4e-3
    print(f"  [{'OK ' if ok3 else 'BAD'}] 最差层 {worst_n} rel={worst:.3e} (阈值 4e-03，bf16 存储精度)")
    if not ok3:
        bad.append("ΔW 重建不一致")

    print("\nE4 bf16 是就近舍入而非截断：")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import export as X
    # 1.00390625 = 1 + 2^-8，正好落在两个 bf16 之间上方；截断给 1.0，就近给 1.0078125
    v = np.array([1.0 + 2.0 ** -8 + 2.0 ** -12], np.float32)
    got = np.frombuffer(X._bf16_bytes(v), "<u2").astype(np.uint32) << 16
    got = got.view(np.float32)[0]
    ok4 = abs(got - 1.0078125) < 1e-7
    print(f"  [{'OK ' if ok4 else 'BAD'}] {v[0]:.8f} -> {got:.8f}（就近应为 1.00781250，"
          f"截断会给 1.00390625）")
    if not ok4:
        bad.append("bf16 用了截断")

    print(f"\n*** {'通过' if not bad else '失败'} ***" + ("" if not bad else f"  失败项: {bad[:5]}"))
    return 0 if not bad else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    sys.exit(write() if ap.parse_args().write else check())
