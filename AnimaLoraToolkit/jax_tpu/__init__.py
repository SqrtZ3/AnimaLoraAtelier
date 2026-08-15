"""Anima 的 TPU（纯 JAX）训练后端。

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
