"""把 jax_tpu/anima_jax.py 与 job 主体拼成一个自包含脚本。

Kaggle 的 script kernel 只跑 metadata 里的**单个** code_file，没有模块可 import，
所以推送前在本地拼。拼接而不是复制 —— 模型只有一份源（jax_tpu/anima_jax.py），
避免两边各改一半。
"""
from pathlib import Path

HERE = Path(__file__).parent
MODEL = HERE.parents[1] / "AnimaLoraToolkit" / "jax_tpu" / "anima_jax.py"
PRE = HERE / "_preamble.py"
BODY = HERE / "_job_body.py"
OUT = HERE / "anima_tpu_job.py"

pre_src = PRE.read_text(encoding="utf-8")
model_src = MODEL.read_text(encoding="utf-8")
body_src = BODY.read_text(encoding="utf-8")

# `from __future__` 必须是文件第一条语句 -> 从两边剥掉，统一放到最前面
def strip_future(s: str) -> str:
    return "\n".join(l for l in s.splitlines()
                     if not l.startswith("from __future__ import"))

def sep(title: str) -> str:
    return "\n\n# " + "=" * 74 + f"\n# {title}\n# " + "=" * 74 + "\n\n"


# 顺序是硬要求：preamble（升 jax）必须排在 anima_jax 的 `import jax` 之前。
OUT.write_text(
    "# ！！自动生成，勿手改 —— 改 jax_tpu/anima_jax.py / _preamble.py / _job_body.py\n"
    "# 后跑 build_job.py 重新生成。\n"
    "from __future__ import annotations\n"
    + sep("① bootstrap（_preamble.py）—— 必须在 import jax 之前")
    + strip_future(pre_src)
    + sep("② 模型（AnimaLoraToolkit/jax_tpu/anima_jax.py）")
    + strip_future(model_src)
    + sep("③ job 主体（_job_body.py）")
    + strip_future(body_src),
    encoding="utf-8")
print(f"已生成 {OUT}  ({len(OUT.read_text(encoding='utf-8').splitlines())} 行)")
