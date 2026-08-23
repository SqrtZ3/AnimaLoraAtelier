"""Anima 的 TPU（纯 JAX）训练后端。

⚠ **这一份是 parity 对拍用的副本。** TPU 路线已独立成自己的仓库
（`kaggle_job/`，可整体迁出发布），那边是**主拷贝** —— 训练、打包、部署都走那边。

本目录保留的理由只有一个：本仓库的 GPU/NPU/DCU 侧改动了 `trainer/` 或 `models/`
之后，要能对拍"JAX 实现是否还跟 PyTorch 一致"。所以：

  * 改 TPU 功能 → 改**那边**，然后同步回这里（两边应当逐字相同）；
  * 改本仓库的 PyTorch 侧 → 跑那边的 `tools/check_sync.py` 与对拍闸门，
    确认没把 TPU 路线的口径带偏。

两边一致性由那边的 `tools/check_sync.py`（sha/AST）与
`tools/tests/check_cache_parity.py`（缓存产物逐 bit）守。

---

与 PyTorch 训练器**不共享代码**，只共享磁盘产物（latent/文本特征缓存进、
标准 LoRA safetensors 出）。模块职责见各文件 docstring：

  anima_jax  DiT 前向 + LoRA 低秩旁路（与 torch 对拍 rel 4.8e-05）
  attention  NaViT 块对角注意力（编译期粗粒度 mask 跳块 + 运行时 segment_ids）
  packing    量化 FFD 打包、布局词表、8 卡成步分组
  flow       Flow Matching 目标（与 trainer/objective.py 对拍）
  optim      LoRA-only AdamW（与 torch.optim.AdamW 对拍）
  train      训练步装配 + 状态存取
  export     -> ComfyUI 可用的 LoRA safetensors
"""
