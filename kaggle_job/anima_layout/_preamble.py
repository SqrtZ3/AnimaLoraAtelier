"""生成脚本的**第一段**：在任何 `import jax` 之前把 jax/libtpu 装到指定版本。

Kaggle 默认镜像是 jax 0.10.2 + libtpu 构建于 2025-06-12，比 Pallas 的版本闸门
（`is_cloud_tpu_older_than`，硬 raise、**无环境变量旁路**）老约 14 个月，不升级则
所有 splash 探测以同一原因失败。**必须在 import jax 之前**——jax 一旦初始化后端
就换不掉 libtpu。这也是 anima_jax.py 的 import 必须排在本段之后的原因。

与上一版的区别：**钉死版本**而不是 `-U`。
  * 上一轮 `-U` 实际装到了 0.11.0（见 anima-tpu-dp-probe 日志），钉死同一版
    使真机结果可复现，也与本地对拍环境（jaxenv: Python 3.12 + jax 0.11.0）一致；
  * 本地/真机版本一旦漂移，"本地过了真机挂"会变成查不动的问题。
Kaggle 允许自由装依赖（docs/notebooks#modifying-a-notebook-specific-environment），
所以这里就是一次正常的 pip install，不是什么绕过手段。
"""

import os
import subprocess
import sys
import time

JAX_VERSION = "0.11.0"          # 与本地 jaxenv 对齐；改这里要同步改 tests/README.md

_BOOT = "未开启"
if os.environ.get("ANIMA_TPU_UPGRADE", "1") == "1":
    if "jax" in sys.modules:
        _BOOT = "[!] jax 已 import，安装无效"
    else:
        _t = time.time()
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q",
                 f"jax[tpu]=={JAX_VERSION}"],
                check=True, capture_output=True, timeout=1200)
            _BOOT = f"jax[tpu]=={JAX_VERSION} 安装 OK {time.time() - _t:.0f}s"
        except subprocess.CalledProcessError as _e:
            _tail = (_e.stderr or b"").decode("utf-8", "replace").strip().splitlines()[-3:]
            _BOOT = f"安装失败（继续跑，Pallas 大概率被闸门挡住）：{' | '.join(_tail)}"
        except Exception as _e:
            _BOOT = f"安装失败：{type(_e).__name__}: {_e}"
print(f"[ INFO ] bootstrap - {_BOOT}", flush=True)
