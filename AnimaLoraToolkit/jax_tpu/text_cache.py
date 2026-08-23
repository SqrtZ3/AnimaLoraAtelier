r"""在 TPU 上现算 krea2 的文本缓存：`caption_ids.npz` + 文本塔权重 -> `<stem>.textfeat.npz`。

## 它替掉了什么

原链路：本地 CUDA 跑 Qwen3-VL -> 1.7GB 的 `.textfeat.npz` -> 上传 Kaggle。
新链路：本地只 tokenize（`tools/dump_caption_ids.py`，96 条 **39.8KB**）-> 上传 ->
真机上跑 `qwen3vl_te` 现算。压缩比实测 **48658×**，而且上传物里没有 caption 明文。

数值口径由两道闸门守：`check_qwen3vl_parity.py`（结构，fp32 rel ≤ 5.3e-7）与
`check_qwen3vl_real.py`（真权重，与本地已有缓存的差全部落在 bf16 噪声内）。

## 产物与目录布局

`/kaggle/input` 是只读的，textfeat 落不到 latent 旁边，而 `jax_tpu/data.py`
要求两者**同目录同 stem**。所以本模块产出一个 staging 目录：

    <out>/<stem>.npz            -> 软链到只读挂载的 latent（不拷贝字节）
    <out>/<stem>.ms<档>.npz     -> 同上（多尺度 sidecar）
    <out>/<stem>.textfeat.npz   -> 本模块现算写入的实体文件
    <out>/_empty.textfeat.npz   -> caption_dropout 用（--empty 时）

然后把 `data_dir` 指到 `<out>` 即可，`jax_tpu/data.py` 一行都不用改。

## 显存

文本塔按 tap 口径只加载 35/36 层 + 裁剪后的 embedding ≈ 3.53B 参数（bf16 7.1GB），
放**单张卡**（v5e 每 chip 16GB）。跑完 `free_params()` 释放，再由 run_train 加载
底模 —— 两者不同时驻留。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

try:
    from . import qwen3vl_te as TE
except ImportError:                     # jax_tpu/ 直接在 sys.path 上
    import qwen3vl_te as TE

VERSION = "1"

#: staging 时要软链过去的 latent 侧文件后缀（textfeat 是本模块自己写的，不链）。
_LINK_SUFFIX = ".npz"


def _as_uint16(a) -> np.ndarray:
    """bf16 数组 -> uint16 位模式（npz 存法，与 `cache_text_features.py` 一致）。"""
    # 切片出来的视图未必连续，.view() 对非连续数组会抛 —— 先落成连续块。
    a = np.ascontiguousarray(a)
    if a.dtype == np.uint16:
        return a
    if a.itemsize == 2:                 # ml_dtypes.bfloat16
        return a.view(np.uint16)
    raise TypeError(f"期望 bf16（2 字节），拿到 {a.dtype}")


def stage_latents(latent_dir: Path, out: Path) -> int:
    """把只读挂载里的 latent 与 ms sidecar 软链进 staging 目录。

    软链失败（某些文件系统不给建）时回退到硬链，再回退到拷贝 —— latent 侧
    96 张只有 127MB，拷贝也不致命，但要把实际走了哪条路打出来。
    """
    out.mkdir(parents=True, exist_ok=True)
    n, mode = 0, "symlink"
    for p in sorted(latent_dir.iterdir()):
        if p.suffix.lower() != _LINK_SUFFIX or p.name.endswith(".textfeat.npz"):
            continue
        q = out / p.name
        if q.exists() or q.is_symlink():
            continue
        try:
            q.symlink_to(p)
        except OSError:
            try:
                os.link(p, q)
                mode = "hardlink"
            except OSError:
                import shutil
                shutil.copy2(p, q)
                mode = "copy"
        n += 1
    stale = sum(1 for p in latent_dir.iterdir() if p.name.endswith(".textfeat.npz"))
    if stale:
        print(f"[textcache] 注意：只读挂载里还有 {stale} 个 .textfeat.npz，**没有链过来** "
              f"—— 本次用的是现算的那一份", flush=True)
    print(f"[textcache] staging latent {n} 个（{mode}）-> {out}", flush=True)
    return n


def fetch_hf(spec: str, token: Optional[str] = None) -> Path:
    """`hf://<repo>[@<rev>]` -> 本地快照目录（只拉权重分片 + index + config）。

    为什么不走 `krea2_jax` 那条 HTTP Range 流式：它在建 map 时就把**文件里所有**
    tensor 的 range GET 提交给线程池，而本模块**故意不读** `model.visual.*` 与
    `layers.35.*` —— 那些 future 永远不会被消费，会一直占着在飞字节预算。
    而且官方 checkpoint 是 2 分片，URL 口径只认单文件。
    文本塔 8.3GB 落 `/kaggle/working` 是装得下的（20GB），用完就删（见 build）。
    """
    body = spec[len("hf://"):]
    repo, _, rev = body.partition("@")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "huggingface_hub"], check=True)
        from huggingface_hub import snapshot_download
    import shutil as _sh
    for d in ("/", "/kaggle/working", str(Path.home())):
        if Path(d).exists():
            u = _sh.disk_usage(d)
            print(f"[textcache] 磁盘 {d}: 可用 {u.free / 1e9:.1f}GB / 共 "
                  f"{u.total / 1e9:.1f}GB", flush=True)
    t0 = time.time()
    p = snapshot_download(repo, revision=rev or None, token=token,
                          allow_patterns=["*.safetensors",
                                          "model.safetensors.index.json",
                                          "config.json"])
    sz = sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file())
    print(f"[textcache] HF {repo}@{rev or 'main'} -> {p}"
          f"（{sz / 1e9:.1f}GB，{time.time() - t0:.0f}s）", flush=True)
    return Path(p)


def build(ids_path: str, te_path: str, out_dir: str,
          latent_dir: Optional[str] = None,
          quantum: int = 128, batch: int = 1,
          token: Optional[str] = None,
          select_layers: Sequence[int] = TE.KREA2_SELECT_LAYERS,
          device: Optional[object] = None,
          drop_te_after: bool = True) -> Path:
    """跑完整条：读 ids -> 加载文本塔 -> 编码 -> 写 textfeat -> staging latent。

    返回 staging 目录（可直接当 `data_dir`）。`te_path` 支持 `hf://<repo>[@<rev>]`
    现拉；`drop_te_after` 在那种情况下用完删掉快照（给后面的 12B 底模腾磁盘）。
    """
    import jax
    import jax.numpy as jnp

    fetched = None
    if str(te_path).startswith("hf://"):
        fetched = fetch_hf(str(te_path), token)
        te_path = str(fetched)

    z = np.load(ids_path, allow_pickle=True)
    stems = [str(s) for s in z["stems"]]
    ids, lens = z["ids"], z["lens"]
    prefix_len = int(z["prefix_len"])
    meta_in = json.loads(str(z["meta"])) if "meta" in z.files else {}
    seqs: List[np.ndarray] = [ids[i, : lens[i]] for i in range(len(stems))]
    empty = z["empty_ids"] if "empty_ids" in z.files else None
    if empty is not None:
        seqs.append(np.asarray(empty, np.int32))
        stems.append("_empty")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if latent_dir:
        stage_latents(Path(latent_dir), out)

    keep = np.unique(np.concatenate(seqs))
    t0 = time.time()
    put = None if device is None else (lambda _n, _a: device)
    params, cfg = TE.load_text_tower(te_path, dtype=jnp.bfloat16,
                                     select_layers=select_layers,
                                     token=token, keep_ids=keep, device_put=put)
    n_par = sum(int(np.prod(x.shape)) for x in jax.tree_util.tree_leaves(
        {k: v for k, v in params.items() if k != "id_map"}))
    print(f"[textcache] 文本塔加载 {time.time() - t0:.0f}s："
          f"{TE.n_layers_needed(select_layers)}/{cfg.layers} 层 + "
          f"{keep.size}/{cfg.vocab} 词表行 = {n_par / 1e9:.2f}B 参数 "
          f"({n_par * 2 / 1e9:.1f}GB bf16)", flush=True)

    t0 = time.time()
    last = [0.0]

    def prog(done: int, total: int) -> None:
        now = time.time()
        if now - last[0] > 20 or done == total:
            last[0] = now
            print(f"[textcache]   {done}/{total} 条，{now - t0:.0f}s", flush=True)

    feats = TE.encode_ids(params, cfg, seqs, prefix_len=prefix_len,
                          select_layers=select_layers, quantum=quantum,
                          batch=batch, progress=prog)
    print(f"[textcache] 编码完成 {time.time() - t0:.0f}s", flush=True)

    meta = json.dumps({"version": VERSION, "family": "krea2",
                       "max_length": meta_in.get("max_length", 0),
                       "text_encoder": str(te_path),
                       "txt_layers": len(select_layers),
                       "select_layers": list(select_layers),
                       "source": "jax_tpu/text_cache.py",
                       "ids_meta": meta_in}, ensure_ascii=False)
    total = 0
    for stem, f in zip(stems, feats):
        arr = _as_uint16(f)
        p = out / f"{stem}.textfeat.npz"
        np.savez(p, txt=arr, meta=np.array(meta))
        total += p.stat().st_size
    print(f"[textcache] 已写 {len(feats)} 个 textfeat -> {out}"
          f"（合计 {total / 1e6:.0f}MB；上传物只有 "
          f"{Path(ids_path).stat().st_size / 1024:.0f}KB 的 ids）", flush=True)

    free_params(params)
    if fetched is not None and drop_te_after:
        # 权重已经在 HBM 上用完了，磁盘那 8.3GB 留着只会跟 26GB 底模抢
        # /kaggle/working 的 20GB。删的是 HF 缓存快照，不碰任何挂载。
        import shutil
        # snapshot_download 返回 <cache>/models--<org>--<repo>/snapshots/<sha>；
        # 布局不是这样就只删快照本身，绝不往上瞎跳。
        root = (fetched.parent.parent
                if fetched.parent.name == "snapshots"
                and fetched.parent.parent.name.startswith("models--")
                else fetched)
        shutil.rmtree(root, ignore_errors=True)
        print(f"[textcache] 已删掉 HF 快照 {root}（腾磁盘给底模）", flush=True)
    return out


def free_params(params) -> None:
    """显式释放文本塔占的 HBM —— 后面还要加载 12B 底模，两者不能同时驻留。"""
    import jax
    for x in jax.tree_util.tree_leaves(params):
        if hasattr(x, "delete"):
            try:
                x.delete()
            except Exception:                                   # noqa: BLE001
                pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--te", required=True,
                    help="Qwen3-VL 文本塔权重：目录 / 单 safetensors / HF resolve URL")
    ap.add_argument("--out", required=True, help="staging 目录（产出后当 data_dir 用）")
    ap.add_argument("--latent-dir", default=None,
                    help="只读的 latent 缓存目录，会软链进 --out")
    ap.add_argument("--quantum", type=int, default=128)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--jax-cache", default="", help="XLA 持久化编译缓存目录")
    a = ap.parse_args(argv)
    if a.jax_cache:
        from jax.experimental.compilation_cache import compilation_cache as cc
        cc.set_cache_dir(a.jax_cache)
    build(a.ids, a.te, a.out, a.latent_dir, a.quantum, a.batch,
          token=os.environ.get("HF_TOKEN"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
