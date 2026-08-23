"""闸门：本仓库的缓存工具 ≡ upstream 的缓存工具（**逐 bit**）。

## 守什么

`tools/_vendor/` 里的代码是从 upstream（`anima-lora-train/AnimaLoraToolkit`）
摘录的。摘录有两种失效方式：

  1. **抄的时候漏了东西** —— 少一个 `.contiguous()`、少一次 dtype 转换、
     对齐口径抄成 `ceil` 而不是 `floor`。
  2. **上游后来改了** —— 由 `tools/check_sync.py` 的 sha/AST 比对守。

本闸门守第 1 种，也是唯一能守它的东西：`check_sync.py` 只知道"两边一样不一样"，
不知道"抄的时候是不是漏了"。

判据是**逐 bit 相同**，不是"误差够小"。理由：同一个 torch、同一份权重、同一条
算子序列，输出没有任何理由不逐 bit —— 一旦不是，就说明有条路走岔了，
"误差 1e-6 可以接受"这种话在这里是自欺。

## 为什么这条闸门值得跑

缓存口径错的代价是**静默的**：token 数变了 → 打包布局变了 → loss 曲线与 GPU 侧
悄悄对不上，而每一步都不报错。等发现时已经烧掉几小时 TPU 配额。

## 用法

    export ANIMA_UPSTREAM=/path/to/anima-lora-train/AnimaLoraToolkit
    <torch-python> check_cache_parity.py --vae <qwen_image_vae.safetensors> \\
        [--transformer <anima-base-v1.0.safetensors>] [--text-encoder <Qwen3-0.6B-Base>] \\
        [--t5-tokenizer <t5_tokenizer 目录>] [--device cuda]

给 `--vae` 就跑 latent 侧（T1）；再给 `--transformer` + `--text-encoder`
+ `--t5-tokenizer` 就additionally 跑文本侧（T2）。两侧都要 torch。

测试图是**当场生成**的（固定 seed 的噪声图，四种尺寸含非 16 整倍数），
不依赖任何数据集。
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent

#: 故意混入非 16 整倍数（floor 对齐才会把它裁掉 —— 抄成 ceil 就会在这里露出来），
#: 以及一张大到能触发 ms sidecar 的。
SIZES = [(517, 333), (256, 256), (640, 384), (1024, 768)]
#: 档位刻意选得比最大图（1024×768 = 64×48 = 3072 token）**小**，否则
#: `plan_multiscale_copy` 会一路 return None，sidecar 分支根本不被执行 ——
#: 那样这条闸门就只覆盖了原生份，等于漏掉一半摘录。
MS_LADDER = "2304"

FAILS: list = []


def check(name: str, cond: bool, detail: str = "") -> bool:
    tag = "OK  " if cond else "FAIL"
    print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""), flush=True)
    if not cond:
        FAILS.append(name)
    return cond


def _upstream() -> Path:
    sys.path.insert(0, str(TOOLS.parent / "jax_tpu" / "tests"))
    from _upstream import upstream
    return upstream("tools/cache_latents.py + tools/cache_text_features.py 的原版")


def make_images(root: Path, n_caption: bool = True) -> list:
    from PIL import Image
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(1234)
    out = []
    for i, (w, h) in enumerate(SIZES):
        p = root / f"img{i:02d}.png"
        Image.fromarray((rng.rand(h, w, 3) * 255).astype(np.uint8)).save(p)
        if n_caption:
            p.with_suffix(".txt").write_text(
                f"parity test {i}, (weighted:1.3), [light], noise tag",
                encoding="utf-8")
        out.append(p)
    return out


def run_tool(script: Path, args: list, cwd: Path) -> None:
    """在**子进程**里跑缓存工具。

    必须隔进程：本仓库的 `_vendor.t5_weighted` 与 upstream 的
    `trainer.text_encode` 是同名符号的两份实现，同进程内 `sys.modules` /
    `sys.path` 会互相污染，跑出来的"一致"就不可信了。
    """
    import subprocess
    r = subprocess.run([sys.executable, str(script), *args], cwd=str(cwd))
    if r.returncode != 0:
        raise SystemExit(f"[ FATAL ] {script.name} 退出码 {r.returncode}（上面是它的输出）")


def cmp_npz(mine: Path, ref: Path, keys: list, label: str) -> None:
    if not check(f"{label}: 两侧文件都在", mine.exists() and ref.exists(),
                 f"mine={mine.exists()} ref={ref.exists()}"):
        return
    a, b = np.load(mine, allow_pickle=True), np.load(ref, allow_pickle=True)
    for k in keys:
        if k not in b.files:
            check(f"{label}[{k}] upstream 侧不存在", False,
                  f"upstream 键: {sorted(b.files)}")
            continue
        if k not in a.files:
            check(f"{label}[{k}] 本仓库侧不存在", False, f"本仓库键: {sorted(a.files)}")
            continue
        x, y = a[k], b[k]
        if x.shape != y.shape:
            check(f"{label}[{k}] shape", False, f"{x.shape} vs {y.shape}")
            continue
        same = bool(np.array_equal(x, y))
        detail = f"shape={x.shape} dtype={x.dtype}"
        if not same and np.issubdtype(x.dtype, np.number):
            # bf16 存成 uint16 位模式，直接算差值没意义 —— 报不同位的个数
            nd = int((x != y).sum())
            detail += f"，{nd}/{x.size} 个元素不同"
        check(f"{label}[{k}] 逐 bit 相同", same, detail)


def t1_latent(a, up: Path, tmp: Path) -> None:
    print("\n[T1] latent 缓存（cache_latents.py）", flush=True)
    mine_dir, ref_dir = tmp / "t1_mine", tmp / "t1_ref"
    imgs = make_images(mine_dir)
    shutil.copytree(mine_dir, ref_dir)

    common = ["--vae", a.vae, "--ms-ladder", MS_LADDER, "--device", a.device]
    run_tool(TOOLS / "cache_latents.py", ["--data-dir", str(mine_dir), *common], TOOLS)
    run_tool(up / "tools" / "cache_latents.py",
             ["--data-dir", str(ref_dir), *common], up)

    keys = ["latent", "latent_flipped", "bucket_w", "bucket_h", "dtype_kind"]
    for p in imgs:
        stem = p.stem
        cmp_npz(mine_dir / f"{stem}.npz", ref_dir / f"{stem}.npz", keys, f"{stem}.npz")
    # ms sidecar：只对**两边都产出了**的比（产不产由 plan_multiscale_copy 决定，
    # 而"两边产出的集合是否相同"本身就是一条判据）
    ms_mine = sorted(p.name for p in mine_dir.glob("*.ms*.npz"))
    ms_ref = sorted(p.name for p in ref_dir.glob("*.ms*.npz"))
    # 先断言真的产出了 sidecar —— 两边都空时上面那句会"通过"，但那是假绿：
    # 说明 MS_LADDER 选得比所有图都大，多尺度那条摘录一行都没跑到。
    check("ms sidecar 非空（否则本闸门漏掉多尺度分支）", bool(ms_mine),
          f"档位 {MS_LADDER}，产出 {len(ms_mine)} 个")
    check("ms sidecar 集合相同", ms_mine == ms_ref, f"{ms_mine} vs {ms_ref}")
    for name in ms_mine:
        if name in ms_ref:
            cmp_npz(mine_dir / name, ref_dir / name, keys, name)


def t2_text(a, up: Path, tmp: Path) -> None:
    print("\n[T2] textfeat 缓存（cache_text_features.py，anima）", flush=True)
    mine_dir, ref_dir = tmp / "t2_mine", tmp / "t2_ref"
    imgs = make_images(mine_dir)
    shutil.copytree(mine_dir, ref_dir)

    common = ["--text-encoder", a.text_encoder, "--transformer", a.transformer,
              "--t5-tokenizer", a.t5_tokenizer, "--device", a.device,
              "--empty-caption"]
    run_tool(TOOLS / "cache_text_features.py",
             ["--data-dir", str(mine_dir), *common], TOOLS)
    run_tool(up / "tools" / "cache_text_features.py",
             ["--data-dir", str(ref_dir), *common], up)

    keys = ["cross", "mask", "caption"]
    for p in imgs:
        n = f"{p.stem}.textfeat.npz"
        cmp_npz(mine_dir / n, ref_dir / n, keys, n)
    cmp_npz(mine_dir / "_empty.textfeat.npz", ref_dir / "_empty.textfeat.npz",
            keys, "_empty.textfeat.npz")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", default="", help="给了就跑 T1（latent 侧）")
    ap.add_argument("--transformer", default="",
                    help="anima 底模；与 --text-encoder/--t5-tokenizer 一起给才跑 T2")
    ap.add_argument("--text-encoder", default="")
    ap.add_argument("--t5-tokenizer", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--keep", action="store_true", help="保留临时目录（排查用）")
    a = ap.parse_args()

    if not a.vae and not a.transformer:
        print(__doc__)
        raise SystemExit("[ FATAL ] 至少给 --vae（T1）或 --transformer 一套（T2）")

    up = _upstream()
    print(f"upstream = {up}")
    tmp = Path(tempfile.mkdtemp(prefix="anima_parity_"))
    print(f"临时目录 = {tmp}")
    try:
        if a.vae:
            t1_latent(a, up, tmp)
        if a.transformer:
            if not (a.text_encoder and a.t5_tokenizer):
                raise SystemExit("[ FATAL ] T2 还需要 --text-encoder 与 --t5-tokenizer")
            t2_text(a, up, tmp)
    finally:
        if a.keep:
            print(f"\n临时目录保留在 {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 64)
    if FAILS:
        print(f"FAIL {len(FAILS)} 项：{FAILS}")
        print("摘录与 upstream 不一致 —— 先修 _vendor/，别拿这份缓存去训练。")
    else:
        print("全部逐 bit 相同：_vendor 的摘录与 upstream 等价。")
    print("=" * 64)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
