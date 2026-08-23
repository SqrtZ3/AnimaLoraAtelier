"""闸门⑮：缓存目录**不含任何图片**时也能扫出完全相同的样本集（纯 numpy，1 秒）。

## 为什么要有这条

TPU 侧从不读像素：`data.CacheDataset._stems` 只拿图片的**文件名**推 stem，
caption 也早就烘焙进 `<stem>.textfeat.npz` 了。所以缓存目录里放原图是纯多余的
暴露 —— 而缓存目录是要上传到 Kaggle 这类外部平台的，那边的内容审查会因为训练图
**直接删库**（2026-08-21 真实发生过一次：`jan-krea2-tpu-cache` 连 96 张原图一起
传上去，被按 NSFW 条款移除并告警）。

`_stems` 因此有了「目录里一张图都没有就按 `<stem>.npz` 推」的回退。这条闸门守两件
只会**静默出错**的事：

  T1 两种模式样本集逐条相同
     同一批 npz，一份带图、一份不带，`(name, grid, tokens, ms_target, txt_len)`
     必须完全一致。错了不报错 —— 只会训练集悄悄少一半或多出鬼样本。

  T2 sidecar 不能被当成样本本体
     `a.textfeat.npz` 若被当成一个叫 `a.textfeat` 的样本，就会去找
     `a.textfeat.textfeat.npz`，报一个「缺 textfeat」的假故障；`a.ms4096.npz`
     同理，还会让多尺度副本被数两次（一次当本体、一次当 sidecar），
     **训练集凭空变大且 loss 口径不变**，从日志上完全看不出来。
     `_empty.textfeat.npz`（caption dropout 用）也必须排掉。

  T3 有图时仍以图片为准（回退不能反过来抢）

  T4 两样都没有时 fail-fast，且报错要同时提到两种模式

跑法（任意带 numpy 的解释器）：
    python check_cache_scan.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import data as D                                                # noqa: E402

OK = FAIL = 0


def check(name, cond, detail=""):
    global OK, FAIL
    if cond:
        OK += 1
        print(f"  [OK ] {name:<44} {detail}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name:<44} {detail}")


def _write(d: Path, stem: str, gh: int, gw: int, txt: int, ms: int = 0):
    """造一条最小合法样本（只写 _scan 会看的那几个键）。"""
    name = f"{stem}.ms{ms}" if ms else stem
    np.savez_compressed(d / f"{name}.npz",
                        latent=np.zeros((D.LATENT_CHANNELS, 1, gh * 2, gw * 2), np.uint8),
                        latent_flipped=np.zeros((D.LATENT_CHANNELS, 1, gh * 2, gw * 2),
                                                np.uint8))
    if not ms:                        # 副本共用本体的 caption，不单独存 textfeat
        np.savez_compressed(d / f"{stem}.textfeat.npz",
                            txt=np.zeros((txt, 12, 2560), np.uint8))


def build(root: Path, with_images: bool) -> Path:
    d = root / ("with_img" if with_images else "npz_only")
    d.mkdir(parents=True)
    plan = [("a", 32, 48, 300), ("b", 24, 24, 17), ("c", 40, 30, 128)]
    for stem, gh, gw, txt in plan:
        _write(d, stem, gh, gw, txt)
        _write(d, stem, gh // 2, gw // 2, txt, ms=4096)         # 多尺度 sidecar
    # caption dropout 的空 caption（不是样本，必须被排掉）
    np.savez_compressed(d / "_empty.textfeat.npz",
                        txt=np.zeros((1, 12, 2560), np.uint8))
    if with_images:
        for stem, *_ in plan:
            (d / f"{stem}.png").write_bytes(b"")                # 只要文件名
    return d


def load(d: Path):
    ds = D.CacheDataset(d, flip_prob=0.5, repeats=1, multiscale=True,
                        caption_dropout=0.0, family="krea2",
                        rng=np.random.RandomState(0))
    return sorted((s.name, s.grid, s.tokens, s.ms_target, s.txt_len)
                  for s in ds.samples)


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="cache_scan_"))
    try:
        img_dir, npz_dir = build(root, True), build(root, False)

        print("T1/T2 无图目录 ≡ 含图目录（且 sidecar 不被当成样本）")
        a, b = load(img_dir), load(npz_dir)
        check("含图 / 无图 样本集逐条相同", a == b,
              f"{len(a)} 条 vs {len(b)} 条")
        check("样本数 = 3 本体 + 3 副本", len(b) == 6, f"实得 {len(b)}")
        names = {n for n, *_ in b}
        check("没有 `.textfeat` 鬼样本",
              not any(".textfeat" in n for n in names), f"{sorted(names)}")
        check("`_empty` 没被当成样本", not any(n.startswith("_empty") for n in names))
        check("副本只出现一次（没被本体+sidecar 数两遍）",
              sum(1 for n in names if ".ms4096" in n) == 3,
              f"ms 副本 {sorted(n for n in names if '.ms4096' in n)}")
        check("无图目录里确实一张图都没有",
              not any(p.suffix.lower() in D.IMG_EXT for p in npz_dir.iterdir()))

        print("\nT3 有图时以图片为准（回退不抢）")
        extra = root / "extra"
        shutil.copytree(img_dir, extra)
        np.savez_compressed(extra / "zz.npz",                   # 只有 npz，没有 png
                            latent=np.zeros((D.LATENT_CHANNELS, 1, 8, 8), np.uint8),
                            latent_flipped=np.zeros((D.LATENT_CHANNELS, 1, 8, 8),
                                                    np.uint8))
        np.savez_compressed(extra / "zz.textfeat.npz",
                            txt=np.zeros((5, 12, 2560), np.uint8))
        check("多出的无图 npz 不进样本（图片模式优先）",
              {n for n, *_ in load(extra)} == names,
              "zz 应被忽略")

        print("\nT4 两样都没有 -> fail-fast")
        empty = root / "empty"
        empty.mkdir()
        try:
            D.CacheDataset(empty, family="krea2", rng=np.random.RandomState(0))
            check("空目录报错", False, "没报错")
        except FileNotFoundError as e:
            msg = str(e)
            check("空目录报错且两种模式都提到",
                  "图片" in msg and ".npz" in msg, msg.splitlines()[0][:70])
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print(f"\n{'*** 通过 ***' if FAIL == 0 else f'*** {FAIL} 项失败 ***'}"
          f"  ({OK} OK / {FAIL} FAIL)")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
