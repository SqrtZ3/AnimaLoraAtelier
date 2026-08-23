"""造 Krea2-TPU 真机冒烟资产（本地跑一次）：

  1. 数据 dataset：16 张噪声 PNG + 对应 latent npz（[16,1,H,W] fp32）+
     textfeat npz（`txt` [L, 3, 128]，tiny 构型的层数/维度）+ dataset-metadata.json
  2. 模型 dataset：tiny.safetensors（K1 闸门产的小构型权重）+ dataset-metadata.json

目录布局按 Kaggle Dataset 上传要求（_staging 下两个 slug 目录）。

用法：
    python make_k2_smoke_assets.py [--root <staging 目录>]
        # 缺省读环境变量 ANIMA_STAGING，再缺省落到 ./_staging
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

TINY = Path(__file__).resolve().parent / "_ref" / "krea2" / "tiny.safetensors"

SIZES = [(256, 256), (384, 256), (512, 512), (256, 384)]   # 四种尺寸 x4 张 = 16
TXT_LAYERS, TXT_DIM = 3, 128                                # tiny 构型


def bf16_u16(a: np.ndarray) -> np.ndarray:
    """fp32 -> bf16 位模式的 uint16（截断即可，冒烟数据不在乎那 0.4 ulp）。"""
    return (np.ascontiguousarray(a, np.float32).view(np.uint32) >> 16).astype(np.uint16)


def main() -> int:
    from PIL import Image

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("ANIMA_STAGING", "_staging"),
                    help="上传 staging 目录，两个 slug 子目录建在它下面。"
                         "缺省读环境变量 ANIMA_STAGING")
    ap.add_argument("--kaggle-user", default=os.environ.get("KAGGLE_USERNAME", ""),
                    help="dataset-metadata.json 的 id 前缀。缺省读 KAGGLE_USERNAME")
    a = ap.parse_args()
    if not a.kaggle_user:
        raise SystemExit(
            "[ FATAL ] 需要 Kaggle 用户名（dataset id 是 <user>/<slug>）：\n"
            "  --kaggle-user <你的用户名> 或 export KAGGLE_USERNAME=<你的用户名>")
    root = Path(a.root).expanduser()
    DATA, MODEL = root / "krea2-tiny-data", root / "krea2-tiny-model"

    rng = np.random.RandomState(7)
    DATA.mkdir(parents=True, exist_ok=True)
    MODEL.mkdir(parents=True, exist_ok=True)
    meta_data = {"title": "krea2-tiny-data", "id": f"{a.kaggle_user}/krea2-tiny-data",
                 "licenses": [{"name": "other"}]}
    meta_model = {"title": "krea2-tiny-model", "id": f"{a.kaggle_user}/krea2-tiny-model",
                  "licenses": [{"name": "other"}]}
    (DATA / "dataset-metadata.json").write_text(json.dumps(meta_data), encoding="utf-8")
    (MODEL / "dataset-metadata.json").write_text(json.dumps(meta_model), encoding="utf-8")

    for i in range(16):
        h, w = SIZES[i % len(SIZES)]
        stem = DATA / f"img{i:02d}"
        px = (rng.rand(h, w, 3) * 255).astype(np.uint8)
        Image.fromarray(px).save(stem.with_suffix(".png"))
        (stem.with_suffix(".txt")).write_text(
            f"tiny smoke caption {i}, test tag, noise", encoding="utf-8")
        # latent [16, 1, H/8, W/8] fp32（VAE f8；patch2 需要偶数 —— 尺寸都是 8 的倍数）
        lat = rng.randn(16, 1, h // 8, w // 8).astype(np.float32)
        np.savez(stem.with_suffix(".npz"), latent=lat,
                 bucket_w=np.array(w), bucket_h=np.array(h),
                 dtype_kind=np.array("fp32"))
        # textfeat：txt [L, 3, 128] bf16 位模式（真缓存就是 bf16 —— 顺带闸读取路径）
        L = int(rng.randint(20, 90))
        txt = (rng.randn(L, TXT_LAYERS, TXT_DIM) * 0.5).astype(np.float32)
        np.savez(str(stem) + ".textfeat.npz", txt=bf16_u16(txt),
                 caption=np.array(f"tiny smoke caption {i}"),
                 meta=np.array(json.dumps({"family": "krea2", "synthetic": True})))

    import shutil
    shutil.copy(TINY, MODEL / "tiny.safetensors")
    print(f"数据 -> {DATA}（16 张）")
    print(f"模型 -> {MODEL}（{TINY.stat().st_size / 1e6:.1f}MB）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
