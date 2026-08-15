"""生成脚本的**第一段**：在任何 `import jax` 之前把 libtpu/jax 升上去。

Kaggle 默认镜像是 jax 0.10.2 + libtpu 构建于 2025-06-12，比 Pallas 的版本闸门
（`is_cloud_tpu_older_than`，硬 raise、无环境变量旁路）老约 14 个月，不升级则
所有 splash 探测以同一原因失败。**必须在 import jax 之前**——jax 一旦初始化
后端就换不掉 libtpu。这也是 anima_jax.py 的 import 必须排在本段之后的原因。
"""

import os
import subprocess
import sys
import time

_BOOT = "未开启"
if os.environ.get("ANIMA_TPU_UPGRADE", "1") == "1":
    if "jax" in sys.modules:
        _BOOT = "[!] jax 已 import，升级无效"
    else:
        _t = time.time()
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "jax[tpu]"],
                           check=True, capture_output=True, timeout=900)
            _BOOT = f"jax[tpu] 升级 OK {time.time() - _t:.0f}s"
        except Exception as _e:
            _BOOT = f"升级失败（继续跑，Pallas 可能被闸门挡住）：{type(_e).__name__}"
print(f"[ INFO ] bootstrap - {_BOOT}", flush=True)
