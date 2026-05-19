"""utils package — 只保留实际被使用的导出。

历史上这里曾经导出 TagBasedDataset / CachedTagBasedDataset / 各种 caption helper，
但它们对应的实现依赖 diffusers / peft / accelerate，主训练脚本 anima_train.py 并不
使用，反而会在云端没装这些可选依赖时让 `import utils` 整个失败。

现在 anima_train.py 通过 `from utils.optimizer_utils import ...` 子模块直接 import，
__init__.py 只做兜底 namespace 声明，不再 eager import 任何内容。
"""
