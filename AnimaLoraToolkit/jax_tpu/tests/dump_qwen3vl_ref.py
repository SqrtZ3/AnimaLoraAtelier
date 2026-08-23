r"""K3 对拍闸门 · 第一步（torch 侧）：Qwen3-VL **文本塔**的参考量落盘。

配套 `check_qwen3vl_parity.py`（jax 侧）。方法论与 K1 相同 ——
**结构对不对用 fp32 + 小构型随机权重就能判**，真权重（8.3GB）只影响数值分布，
不改算子拓扑。真权重那一路另有闸门（`check_qwen3vl_real.py`，拿本地已缓存的
`.textfeat.npz` 当参考，不需要再跑一次 torch）。

## 为什么要这个闸门

`jax_tpu/qwen3vl_te.py` 把 Qwen3-VL-4B 的文本塔搬到 TPU 上，让 krea2 的
`<stem>.textfeat.npz`（12 层 hidden 堆叠，96 张就 1.7GB）**不再需要在本地算好
再上传** —— 上传的只有 token ids（96 条约 100KB）。搬错的代价是"条件一直错
且不报错"，与 `tools/cache_text_features.py` docstring 里记的那个坑同一形态。

## 落盘内容（`_ref/qwen3vl/`）

  `tiny_te.safetensors`  小构型全量权重（fp32，键名带 `model.language_model.`
                         前缀 —— 与真 checkpoint 同名，顺带闸了 jax 侧加载器）
  `qwen3vl_ref.npz`      三组参考量：
      ① `ids_a/ids_b`（无 padding 的单条前向）+ `hs_a/hs_b` [n_layers+1, L, D]
         —— **全部** hidden_states，jax 侧逐层比对（不是只比最后一层）
      ② `probe_*`：batch 中段 padding 探针，见下
      ③ `meta`：层数/维度/位置编码口径等，jax 侧断言用

## 这个脚本要顺带裁决的两件事（都写进 npz，由 check 侧断言）

**(a) `hidden_states[k]` 到底是什么。** `trainer/model_family.py:KREA2_SELECT_LAYERS`
取 (2,5,...,35)，如果 `hidden_states[k]` = 第 k 层的**输入**（= 第 k−1 层输出，
未过 final norm），那么 35 层就够用、第 36 层（idx 35 那层）根本不用跑、
final `norm.weight` 也不用加载。这三条省下来的是真机上的时间与 HBM，但猜错
就是逐 bit 全错 —— 所以这里直接把 embedding 输出、每层输出、final norm 输出
一起落盘，让 check 侧用**恒等断言**判定，而不是靠读 transformers 源码推断。

**(b) batch padding 会不会改条件。** `_encode_krea2_batch` 在
`max_length<=0`（本仓库 JAN 那轮 config 的口径）下走 `padding=True`，而 suffix
是**在 padding 之后**才拼上去的 —— 于是短 caption 的 suffix token 落在
"被 padding 推后了的位置"上。而 `Qwen3VLTextModel.forward` 在 `position_ids=None`
时用的是 `arange`（**不看 attention_mask**）。若如此，同一条 caption 单独编码
与在 batch 里编码得到的 hidden 就**不相同**。
`cache_text_features.py` 走的是 B=1 逐条编码（无 padding），所以已落盘的缓存
口径 = 无 padding；jax 侧必须复刻这一份，**右侧 padding**（suffix 之后）才是
等价的。probe 把这件事测成一个数，而不是留在注释里当推断。

用法（torch 解释器，见 tests/README.md）：
    python dump_qwen3vl_ref.py [--out 目录]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

DT = torch.float32

#: 小构型：结构旋钮与真模型一致（GQA 4:1、head_dim 偶数、mrope 三段、
#: interleaved），只把宽度/层数缩小。层数取 7 是为了让 tap 的"每 3 层一取"
#: 模式（2,5,...）在小模型上也能取到两个以上。
TINY = dict(
    vocab_size=97, hidden_size=64, intermediate_size=160,
    num_hidden_layers=7, num_attention_heads=4, num_key_value_heads=1,
    head_dim=16, rms_norm_eps=1e-6,
)
TINY_ROPE = dict(rope_type="default", rope_theta=5000000.0,
                 mrope_section=[3, 3, 2], mrope_interleaved=True)


def build_tiny():
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    cfg = Qwen3VLTextConfig(rope_parameters=dict(TINY_ROPE), **TINY)
    # eager：逐算子语义最明确（sdpa 的 fp32 累加细节因后端而异）。fp32 下两者
    # 的差远小于本闸门的判据，选 eager 只是让"参考量"这三个字名副其实。
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    model = Qwen3VLTextModel(cfg).to(DT).eval()
    # 默认初始化的 RMSNorm weight 全是 1 —— 那样 scale 写错也看不出来。
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("norm.weight") or name.endswith("layernorm.weight"):
                p.copy_(1.0 + 0.1 * torch.randn_like(p))
    return model, cfg


@torch.no_grad()
def all_hidden(model, ids: torch.Tensor):
    """[1, L] ids -> (hs [n_layers+1, L, D], normed [L, D])。

    **不用 `output_hidden_states`** —— transformers 5.x 的 hidden_states 是由
    `@capture_outputs` 的 hook 收集的，语义随版本变；这里直接手工跑一遍
    embedding + 逐层，得到无歧义的"第 k 层输入"。同时用 `output_hidden_states=True`
    再跑一遍官方前向，把两者一起落盘 —— check 侧断言它们相等，等于把
    "官方 hidden_states[k] == 第 k 层输入" 这条**测出来**而不是假设。
    """
    emb = model.embed_tokens(ids)
    L = ids.shape[1]
    pos = torch.arange(L)[None].expand(4, ids.shape[0], -1)
    cos, sin = model.rotary_emb(emb, pos[1:])
    # 因果 mask 必须显式给：`eager_attention_forward` 在 attention_mask=None 时
    # **完全不加 mask**（= 双向注意力）。漏掉这句时 official 与手工版从第 1 层
    # 起就差 6.7e-2 —— 本闸门第一次跑就是这么发现的。
    causal = torch.full((L, L), float("-inf"), dtype=DT).triu(1)[None, None]
    h = emb
    outs = [h]
    for layer in model.layers:
        h = layer(h, position_embeddings=(cos, sin), attention_mask=causal)
        outs.append(h)
    manual = torch.stack(outs, 0)[:, 0]                 # [n+1, L, D]
    normed = model.norm(h)[0]

    official = model(input_ids=ids, output_hidden_states=True).hidden_states
    official = torch.stack([t[0] for t in official], 0)
    # cos/sin 是 [B, L, head_dim]（mrope 的 3 路在 apply_interleaved_mrope 里已
    # 合并掉），取 batch 0 -> [L, head_dim]。
    return manual, normed, official, (cos[0], sin[0])


@torch.no_grad()
def padding_probe(model, short: torch.Tensor, long: torch.Tensor, suffix: torch.Tensor):
    """裁决 (b)：同一条短序列，在「中段 padding」与「无 padding」两种摆法下，
    suffix token 的 hidden 是否相同。

    摆法完全复刻 `_encode_krea2_batch` 的 `max_length<=0` 分支：
        单条：[short ; suffix]
        批内：[short ; PAD×k ; suffix]  与  [long ; suffix]  同批
    attention_mask 把 PAD 位屏蔽掉。返回两者 suffix 段的最大绝对差。
    """
    pad_id = 0
    k = long.shape[1] - short.shape[1]
    single = torch.cat([short, suffix], 1)
    batched = torch.cat([short, torch.full((1, k), pad_id, dtype=torch.long), suffix], 1)
    mask = torch.ones_like(batched)
    mask[:, short.shape[1]:short.shape[1] + k] = 0

    h_single = model(input_ids=single).last_hidden_state[0, -suffix.shape[1]:]
    h_batch = model(input_ids=batched, attention_mask=mask
                    ).last_hidden_state[0, -suffix.shape[1]:]
    # 参照组：把 padding 挪到 suffix **之后**（右 padding），此时 suffix 的
    # 位置与单条一致 —— 期望逐 bit 相同（causal + mask 下右侧 padding 不可见）。
    right = torch.cat([short, suffix, torch.full((1, k), pad_id, dtype=torch.long)], 1)
    rmask = torch.ones_like(right)
    rmask[:, -k:] = 0
    h_right = model(input_ids=right, attention_mask=rmask
                    ).last_hidden_state[0, short.shape[1]:short.shape[1] + suffix.shape[1]]
    return (float((h_single - h_batch).abs().max()),
            float((h_single - h_right).abs().max()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "_ref" / "qwen3vl"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    model, cfg = build_tiny()
    rng = np.random.RandomState(7)
    ids_a = torch.from_numpy(rng.randint(0, TINY["vocab_size"], (1, 23)).astype(np.int64))
    ids_b = torch.from_numpy(rng.randint(0, TINY["vocab_size"], (1, 41)).astype(np.int64))

    hs_a, norm_a, off_a, (cos_a, sin_a) = all_hidden(model, ids_a)
    hs_b, norm_b, off_b, _ = all_hidden(model, ids_b)

    suffix = torch.from_numpy(rng.randint(0, TINY["vocab_size"], (1, 5)).astype(np.int64))
    d_mid, d_right = padding_probe(model, ids_a, ids_b, suffix)
    print(f"[probe] 中段 padding vs 单条：max|Δ| = {d_mid:.3e}")
    print(f"[probe] 右侧 padding vs 单条：max|Δ| = {d_right:.3e}")

    from safetensors.torch import save_file
    sd = {f"model.language_model.{k}": v.contiguous()
          for k, v in model.state_dict().items()}
    save_file(sd, str(out / "tiny_te.safetensors"))

    meta = dict(TINY, rope=TINY_ROPE, torch_version=torch.__version__,
                transformers_version=__import__("transformers").__version__)
    np.savez(out / "qwen3vl_ref.npz",
             ids_a=ids_a.numpy().astype(np.int32)[0],
             ids_b=ids_b.numpy().astype(np.int32)[0],
             hs_a=hs_a.float().numpy(), hs_b=hs_b.float().numpy(),
             official_a=off_a.float().numpy(), official_b=off_b.float().numpy(),
             norm_a=norm_a.float().numpy(), norm_b=norm_b.float().numpy(),
             cos_a=cos_a.float().numpy(), sin_a=sin_a.float().numpy(),
             probe_mid=np.float64(d_mid), probe_right=np.float64(d_right),
             meta=np.array(json.dumps(meta, ensure_ascii=False)))
    print(f"已写 {out}/tiny_te.safetensors（{len(sd)} 张量）与 qwen3vl_ref.npz")
    print(f"  hidden_states 官方 {off_a.shape[0]} 份 / 手工 {hs_a.shape[0]} 份 "
          f"（层数 {TINY['num_hidden_layers']}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
