"""把 jax_tpu/{anima_jax,attention}.py 与 job 主体拼成一个自包含脚本。

与 anima_navit/build_job.py 同结构（Kaggle script kernel 只跑单个 code_file，
没有模块可 import，所以推送前在本地拼）。拼接而不是复制 —— 模型与注意力后端
各只有一份源，两条路线（打包 / 分桶）共用它，避免 A/B 时两边偷偷不同构。

顺序是硬要求：
  ① preamble 装 jax[tpu]，**必须排在任何 import jax 之前**（libtpu 一旦初始化
     就换不掉，Pallas 有硬版本闸门）；
  ② anima_jax 提供 forward_packed / forward_ragged / AnimaConfig / init_lora；
  ③ attention 依赖 ② 的 block_diag_bias（单文件下靠 _anima_jax() 回退到本模块）；
  ④ job 主体。
"""
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE.parents[1] / "AnimaLoraToolkit" / "jax_tpu"
PARTS = [
    ("① bootstrap（_preamble.py）—— 必须在 import jax 之前", HERE / "_preamble.py"),
    ("② 模型（jax_tpu/anima_jax.py）", SRC / "anima_jax.py"),
    ("③ 注意力后端（jax_tpu/attention.py）", SRC / "attention.py"),
    ("④ job 主体（_job_body.py）", HERE / "_job_body.py"),
]
OUT = HERE / "anima_ragged_job.py"


def strip_future(s: str) -> str:
    """`from __future__` 必须是文件第一条语句 -> 从各段剥掉，统一放最前面。"""
    return "\n".join(l for l in s.splitlines()
                     if not l.startswith("from __future__ import"))


def sep(title: str) -> str:
    return "\n\n# " + "=" * 74 + f"\n# {title}\n# " + "=" * 74 + "\n\n"


body = ("# ！！自动生成，勿手改 —— 改 jax_tpu/*.py 或 _preamble/_job_body 后\n"
        "# 重跑 build_job.py 生成。\n"
        "from __future__ import annotations\n")
for title, path in PARTS:
    body += sep(title) + strip_future(path.read_text(encoding="utf-8"))
OUT.write_text(body, encoding="utf-8")

n = len(OUT.read_text(encoding="utf-8").splitlines())
print(f"已生成 {OUT}  ({n} 行)")
# 语法自检：拼接顺序错/剥 future 剥坏了，在这里就该炸，而不是推上去浪费一轮配额
compile(OUT.read_text(encoding="utf-8"), str(OUT), "exec")
print("语法自检通过")
