"""`_vendor` —— 从 upstream（`anima-lora-train/AnimaLoraToolkit`）搬过来的最小代码集。

本仓库（TPU 路线）的**训练侧**对 upstream 零依赖。但**数据缓存**这一步要跑
PyTorch：VAE 编码图像、文本编码器算 cross 条件。那部分实现的权威版本在 upstream，
照抄一份到这里，是为了让「拿到本仓库的人不必先 clone 另一个仓库」。

## 里面有什么

| 文件 | 来源 | 方式 |
|---|---|---|
| `wan_vae.py` | `models/wan/vae2_1.py` | 整份复制（**逐字节相同**，含 Alibaba 版权头） |
| `latent_plan.py` | `trainer/data.py` 的 8 个符号 | 按 AST 行号摘录（函数体逐字） |
| `st_load.py` | `trainer/checkpoint.py` + `trainer/models.py` | 摘录；`load_vae` 改一行 import |
| `llm_adapter.py` | `models/anima_modeling.py:12-220` | 摘录 + 新增 `load_llm_adapter` |
| `t5_weighted.py` | `trainer/text_encode.py` | 摘录（去掉训练期 LRU cache） |
| `krea2_te.py` | `trainer/model_family.py` | 摘录（krea2 文本编码，回退路径） |

确切行号、上游 commit、sha256 都记在 `UPSTREAM.md`。

## 两道防线守着「抄了但没跟上游同步」

1. `tools/check_sync.py` —— 比对本地摘录段与 upstream 当前内容的 AST，漂了就报。
2. `tools/tests/check_cache_parity.py` —— 同一批图分别用 upstream 工具和本仓库工具
   产缓存，判 latent/cross **逐 bit 相同**。

这两条必须一起跑：sha 能发现「上游改了」，逐 bit 能发现「摘录时漏了什么」。
只有第一条会漏掉摘录本身的错，只有第二条会漏掉未来的漂移。
"""
