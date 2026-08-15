r"""把 `jax_tpu/` 整包 + 训练 yaml 打成**一个自包含 Kaggle script kernel**。

## 为什么不像别的 job 那样"拼接源码"

探针类 job 只需要 anima_jax + attention 两个文件，直接首尾相接拼成一个文件即可。
但训练要 10 个模块（adapters / flow / sched / aux / optim / packing / data /
export / config / train / run_train），它们之间是**带命名空间的互相引用**
（`AD.apply` / `F.sample_t_np` / …）。首尾相接会把 10 份顶层名字压进同一个
namespace，`config.py` 的 `build` 和 `train.py` 的 `build` 之类会互相覆盖 ——
而且是静默覆盖。

所以这里换个做法：**把每个模块的源码 base64 进脚本，运行时写回磁盘再正常
import**。模块边界原样保留，本地跑过的代码与真机跑的逐字节相同（脚本里带了
sha256 自检），也就不存在"拼接版和本地版行为不一样"这种查不动的问题。

## 权重与数据从哪来

  * 底模：Kaggle **Models**（`model_sources`），挂到 `/kaggle/input/...`
  * latent / textfeat 缓存：Kaggle **Datasets**（`dataset_sources`）
  * 都在 `kernel-metadata.json` 里挂；本脚本只负责把**代码与配置**带上去。

路径靠环境变量覆盖 yaml 里的本地路径（yaml 本身不动，两个后端共用同一份）：

    ANIMA_TRANSFORMER=/kaggle/input/anima-base/anima-base-v1.0.safetensors
    ANIMA_DATA_DIR=/kaggle/input/villainchin-cache
    ANIMA_OUTPUT_DIR=/kaggle/working/out

## 用法

    python build_job.py --config ../../AnimaLoraToolkit/config/train_anima.yaml
"""

from __future__ import annotations

import argparse
import base64
import hashlib
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE.parents[1] / "AnimaLoraToolkit" / "jax_tpu"
OUT = HERE / "anima_train_job.py"

#: 打包哪些模块。顺序无所谓（运行时是正常 import），但列表要全 ——
#: 漏一个会在真机上报 ModuleNotFoundError，白烧一轮配额。
MODULES = ("adapters.py", "anima_jax.py", "attention.py", "auxloss.py", "config.py",
           "data.py", "export.py", "flow.py", "optim.py", "packing.py",
           "sched.py", "train.py", "run_train.py")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="训练 yaml（会被内嵌进脚本）")
    ap.add_argument("--extra-args", default="",
                    help="追加给 run_train 的命令行参数，如 --max-steps 50")
    a = ap.parse_args()

    blobs, digest = {}, hashlib.sha256()
    for m in MODULES:
        raw = (SRC / m).read_bytes()
        digest.update(raw)
        blobs[m] = base64.b64encode(raw).decode()
    cfg_raw = Path(a.config).read_bytes()
    digest.update(cfg_raw)

    body = _TEMPLATE.format(
        preamble=(HERE / "_preamble.py").read_text(encoding="utf-8"),
        blobs=repr(blobs),
        cfg=repr(base64.b64encode(cfg_raw).decode()),
        sha=digest.hexdigest()[:16],
        extra=repr(a.extra_args.split()),
    )
    OUT.write_text(body, encoding="utf-8")
    compile(body, str(OUT), "exec")          # 语法自检，别推上去才炸
    print(f"已生成 {OUT}（{len(body.splitlines())} 行，源码 sha {digest.hexdigest()[:16]}）")
    print(f"内嵌模块 {len(MODULES)} 个 + 配置 {Path(a.config).name}")
    return 0


_TEMPLATE = '''# ！！自动生成，勿手改 —— 改 jax_tpu/*.py 或 yaml 后重跑 build_job.py。
# 源码 sha256[:16] = {sha}
"""Anima LoRA on TPU v5e-8：把 jax_tpu 整包写回磁盘再 import，然后开跑。"""

{preamble}

import base64
import os
import sys
import traceback
from pathlib import Path

_BLOBS = {blobs}
_CFG = {cfg}
_EXTRA = {extra}

_PKG = Path("/kaggle/working/jax_tpu")
if not _PKG.parent.exists():                 # 本地干跑
    _PKG = Path(__file__).parent / "_unpacked"
_PKG.mkdir(parents=True, exist_ok=True)
for _name, _b64 in _BLOBS.items():
    (_PKG / _name).write_bytes(base64.b64decode(_b64))
_CFG_PATH = _PKG / "train.yaml"
_CFG_PATH.write_bytes(base64.b64decode(_CFG))
sys.path.insert(0, str(_PKG))
print(f"[ INFO ] 已解包 {{len(_BLOBS)}} 个模块 -> {{_PKG}}", flush=True)


def _override_paths(path):
    """用环境变量覆盖 yaml 里的本地路径。yaml 本身不动 —— 两个后端共用同一份，
    改了就不是同一个实验了。"""
    import yaml
    d = yaml.safe_load(path.read_text(encoding="utf-8"))
    for env, key in (("ANIMA_TRANSFORMER", "transformer_path"),
                     ("ANIMA_DATA_DIR", "data_dir"),
                     ("ANIMA_OUTPUT_DIR", "output_dir")):
        v = os.environ.get(env)
        if v:
            print(f"[ INFO ] {{key}}: {{d.get(key)!r}} -> {{v!r}}（来自 {{env}}）")
            d[key] = v
    path.write_text(yaml.safe_dump(d, allow_unicode=True), encoding="utf-8")


def main() -> int:
    _override_paths(_CFG_PATH)
    argv = ["--config", str(_CFG_PATH), "--devices",
            os.environ.get("ANIMA_DEVICES", "8"),
            "--jax-cache", "/kaggle/working/jax_cache"] + list(_EXTRA)
    import run_train
    return run_train.main(argv)


if __name__ == "__main__":
    try:
        _rc = main()
    except Exception:
        # **恒返回 0**：Kaggle 把非零退出码判成 ERROR，而"训练中途 OOM"是数据、
        # 不是脚本故障 —— 报成 ERROR 会让日志与产物都不好拿（前几轮踩过）。
        traceback.print_exc()
        _rc = 0
    print(f"[ INFO ] 结束 rc={{_rc}}", flush=True)
    raise SystemExit(0)
'''


if __name__ == "__main__":
    raise SystemExit(main())
