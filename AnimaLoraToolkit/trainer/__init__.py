"""anima_train 内部模块拆分包。

历史上所有训练相关代码都挤在 `anima_train.py` 一个 4500+ 行的大文件里。本包按职责
做中等粒度拆分，每个子模块只导出本职任务所需的公开接口：

- `text_encode`  — Qwen / T5 文本编码与 prompt-tag 权重解析
- `sampling`     — 训练时推理（flow sigmas、ER-SDE 采样、sample_image）
- `objective`    — 噪声 / timestep / loss / 自适应采样 / grad helpers
- `lora`         — LoRA / LoKr / DoRA 适配器、注入器、保存/加载
- `data`         — ARB 分桶、ImageDataset、CachedLatentDataset、collate
- `checkpoint`   — 训练状态保存恢复、resume 路径规整、权重前缀映射
- `models`       — Anima transformer / VAE / text encoder 加载
- `config`       — YAML→args 映射、TimestepConfig / NoiseConfig / LossConfig dataclass、wd 解析

主入口 `anima_train.py` 只保留 argparse、依赖检测、UI（rich 进度条、loss 曲线）、
监控启动、训练主循环 + 信号处理。每个 trainer/*.py 文件都可以被其它脚本独立 import
（例如未来想写一个独立的 dataset analyzer 或 LoRA merge 工具）。
"""
