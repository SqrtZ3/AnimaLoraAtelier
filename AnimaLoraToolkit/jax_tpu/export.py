r"""JAX LoRA -> safetensors，键名/形状/dtype 与仓库 PyTorch 侧一致。

TPU 侧训出来的东西必须能被**现有链路**直接吃下：ComfyUI 的 Load LoRA、
仓库的 resume、以及 tools/ 下那些取证脚本。所以这里不发明新格式。

## 对齐依据

  trainer/lora.py:2054  `base = "lora_unet_" + name.replace(".", "_")`
  trainer/lora.py:2058  `sd[f"{base}.alpha"] = tensor(float(mod_alpha))`（标量 fp32）
  trainer/lora.py:115   `down = A : (rank, in)`  -> `lora_down.weight`
  trainer/lora.py:116   `up   = B : (out, rank)` -> `lora_up.weight`
  实测：D:\models\LoRA\anima\... 的真实产物键名为
        `lora_unet_blocks_0_self_attn_q_proj.*`（**没有** `net.` 前缀）

JAX 侧的存法是 `a: (in, rank)` / `b: (rank, out)`（因为 `dense` 走
`x @ a @ b`），所以导出时**两个都要转置**。转置写漏不会报错——形状恰好在
rank 是方阵或 in==out 时也能加载，然后静默出错。这里有断言。

## dtype

权重存 bf16（与仓库一致：trainer/lora.py:2062 注明"训练本就在 bf16 混精度下，
fp32 保存只是浪费空间"）。alpha 存 fp32 标量。
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def _to_np(x) -> np.ndarray:
    a = np.asarray(x)
    if a.dtype == np.float32 or a.dtype == np.float64:
        return a.astype(np.float32)
    return np.asarray(x, dtype=np.float32) if a.dtype.kind != "f" else a


def _bf16_bytes(a: np.ndarray) -> bytes:
    """fp32 -> bf16 的**就近舍入**（round-to-nearest-even），不是截断。

    直接取高 16 位（截断）会引入约 0.4 ulp 的系统性偏差；numpy 没有原生 bf16，
    所以手写一遍进位。
    """
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    rounding = ((u >> 16) & 1) + 0x7FFF
    return ((u + rounding) >> 16).astype("<u2").tobytes()


def lora_state_dict(lora: Dict[str, Dict[str, Any]], rank: int,
                    alpha: Optional[float] = None) -> Dict[str, tuple]:
    """把 JAX 的 lora pytree 翻成 {key: (dtype, shape, ndarray)}。

    alpha=None 取 alpha=rank（scaling=1），与 anima_jax.lora_scaling 同口径。
    """
    a_val = float(rank if alpha is None else alpha)
    sd: Dict[str, tuple] = {}
    for name, mod in sorted(lora.items()):
        base = "lora_unet_" + name.replace(".", "_")
        A = _to_np(mod["a"])          # (in, rank)
        B = _to_np(mod["b"])          # (rank, out)
        if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[0]:
            raise ValueError(f"{name}: 形状不成对 a{A.shape} b{B.shape}")
        if A.shape[1] != rank:
            raise ValueError(f"{name}: a 的第 1 维 {A.shape[1]} != rank {rank}"
                             f"（是不是把 (in,rank) 和 (rank,out) 搞反了？）")
        sd[f"{base}.lora_down.weight"] = ("BF16", list(A.T.shape), A.T)   # (rank, in)
        sd[f"{base}.lora_up.weight"] = ("BF16", list(B.T.shape), B.T)     # (out, rank)
        sd[f"{base}.alpha"] = ("F32", [], np.asarray(a_val, np.float32))
    return sd


def save_safetensors(path, sd: Dict[str, tuple],
                     metadata: Optional[Dict[str, str]] = None) -> Path:
    """手写 safetensors 序列化（不引 safetensors 依赖，TPU 侧环境越薄越好）。

    格式：<u8 头长度><utf-8 JSON 头><数据区>；头里每项
    {"dtype":..., "shape":[...], "data_offsets":[start,end]}，offset 相对数据区。
    """
    path = Path(path)
    header: Dict[str, Any] = {}
    blobs, off = [], 0
    for k, (dt, shape, arr) in sd.items():
        raw = _bf16_bytes(arr) if dt == "BF16" else \
            np.ascontiguousarray(arr, dtype=np.float32).tobytes()
        header[k] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    if metadata:
        header["__metadata__"] = {str(a): str(b) for a, b in metadata.items()}
    hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
    hb += b" " * ((8 - len(hb) % 8) % 8)          # 头按 8 字节对齐
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for b in blobs:
            f.write(b)
    return path


#: LoKr / DoRA 的键名，与 trainer/lora.py:2063-2078 的 `state_dict()` 逐字一致。
#: `dora_scale` 存 **fp32**（其余存 bf16）：它是整个适配器里唯一初值 ≈‖W‖、训练
#: 只在其上叠很小增量的参数，降到 bf16 会让 resume 丢掉学到的幅度（本地实测幅度
#: 整体动 1% 时 bf16 往返后该增量的最大相对误差 ~50%）。同 lora.py:2070 的注释。
_LOKR_KEYS = ("lokr_w1", "lokr_w2_a", "lokr_w2_b")


def adapter_state_dict(flat: Dict[str, Dict[str, Any]]) -> Dict[str, tuple]:
    """`adapters.unstack` 的产物 -> safetensors 条目。

    自动识别是 LoKr 还是标准 LoRA（看有没有 `lokr_w1`）。标准 LoRA 的两个矩阵
    **都要转置**：JAX 侧存 `a:(in,rank)` / `b:(rank,out)`（因为 `dense` 走
    `x @ a @ b`），而仓库/ComfyUI 认的是 `lora_down:(rank,in)` / `lora_up:(out,rank)`。
    转置写漏不会报错——rank 是方阵或 in==out 时也能加载，然后静默出错。
    """
    sd: Dict[str, tuple] = {}
    for name, mod in sorted(flat.items()):
        base = "lora_unet_" + name.replace(".", "_")
        sd[f"{base}.alpha"] = ("F32", [], np.asarray(mod["_alpha"], np.float32))
        if "lokr_w1" in mod:
            for k in _LOKR_KEYS:
                a = _to_np(mod[k])
                sd[f"{base}.{k}"] = ("BF16", list(a.shape), a)
        else:
            A = _to_np(mod["a"])          # (in, rank)
            B = _to_np(mod["b"])          # (rank, out)
            if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[0]:
                raise ValueError(f"{name}: 形状不成对 a{A.shape} b{B.shape}")
            sd[f"{base}.lora_down.weight"] = ("BF16", list(A.T.shape), A.T)
            sd[f"{base}.lora_up.weight"] = ("BF16", list(B.T.shape), B.T)
        if "dora_scale" in mod:
            d = _to_np(mod["dora_scale"]).reshape(-1, 1)
            sd[f"{base}.dora_scale"] = ("F32", list(d.shape), d)
    return sd


def export_adapter(path, flat: Dict[str, Dict[str, Any]],
                   metadata: Optional[Dict[str, str]] = None) -> Path:
    """一步导出（LoRA / LoKr / ±DoRA）。metadata 建议带上 step 与全部结构参数，
    便于事后归因（memory `[[dont-infer-dataset-provenance]]`：实跑参数读
    checkpoint 元数据，别信仓库里的同名 yaml）。

    **DoRA 的部署注意**：ComfyUI 标准 Load LoRA 会物化 ΔW，Anima 的大层上有 OOM
    静默丢层的前科；仓库改用内置的 Bypass(Model Only) 低秩注入，而**那条路径不支持
    `dora_scale`** —— DoRA 成品在该链路会静默丢掉幅度分量（memory
    `[[krea2-comfyui-lora-deploy]]`）。这不是本导出的 bug，但用之前要知道。
    """
    return save_safetensors(path, adapter_state_dict(flat), metadata)


def export_lora(path, lora, rank: int, alpha: Optional[float] = None,
                metadata: Optional[Dict[str, str]] = None) -> Path:
    """一步导出。metadata 建议带上 step / budget / t 采样配置，便于事后归因
    （memory `[[dont-infer-dataset-provenance]]`：实跑参数读 checkpoint 元数据，
      别信仓库里的同名 yaml）。"""
    return save_safetensors(path, lora_state_dict(lora, rank, alpha), metadata)
