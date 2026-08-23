r"""闸门⑫上半：用 **PyTorch 侧的真实实现**（trainer/lora.py）产出参考数据。

跑法（torch 解释器，见 tests/README.md）：

    python dump_adapter_ref.py

产物 `_ref/adapter_ref.npz`，再用 jax 解释器跑 `check_adapter_parity.py`。

## 为什么要有这一对

LoKr 的 kron-bypass 与 DoRA 的逐行范数都是"写错了不报错"的数学：
  * kron 的 (f, in_dim) 拆分方向反了 -> 形状照样对得上，输出全错；
  * `scaling = alpha / rank` 里的 rank 用了 cap 之前的值 -> 强度差一截；
  * DoRA 的 ‖W+ΔW‖ 少算了交叉项 -> 幅度慢慢漂。
这些都不会抛异常，只会让 TPU 训出来的东西和 GPU 不是一回事。所以必须对拍。

同一份 npz 里也带了 `adamw_snr` 的参考轨迹（SNR 锐化 + cautious 掩码的口径同样
容易抄错：锐化后要按**张量均值**重归一化、掩码要按保留比例重归一化）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from _upstream import upstream                                 # noqa: E402
upstream("trainer/lora.py 的 LoKr/LoRA 与 utils/adamw_snr_optimizer.py")

from trainer.lora import LoKrLayer, LoRALinear                 # noqa: E402
from utils.adamw_snr_optimizer import AdamWSNR                 # noqa: E402

OUT = Path(__file__).parent / "_ref" / "adapter_ref.npz"
IN_F, OUT_F, RANK, FACTOR = 64, 96, 8, 4
N_TOK = 7


def main() -> int:
    torch.manual_seed(0)
    out: dict[str, np.ndarray] = {}

    x = torch.randn(N_TOK, IN_F, dtype=torch.float32)
    base = torch.nn.Linear(IN_F, OUT_F, bias=False)
    torch.nn.init.normal_(base.weight, std=0.05)

    # ── ① 纯 LoKr（无 DoRA）──────────────────────────────────────────────────
    lk = LoKrLayer(IN_F, OUT_F, rank=RANK, alpha=float(RANK), factor=FACTOR)
    lk.train()
    with torch.no_grad():                     # w2_b 初值是 0 -> 手动填成随机，
        torch.nn.init.normal_(lk.lokr_w2_b, std=0.03)     # 否则对拍的是 0=0
    y_lokr = lk(x)
    out.update(x=x.numpy(), base_w=base.weight.detach().numpy(),
               w1=lk.lokr_w1.detach().numpy(),
               w2a=lk.lokr_w2_a.detach().numpy(),
               w2b=lk.lokr_w2_b.detach().numpy(),
               y_lokr=y_lokr.detach().numpy(),
               scaling=np.asarray(lk.scaling, np.float32),
               eff_rank=np.asarray(lk.rank, np.int32),
               eff_factor=np.asarray(lk.factor, np.int32))

    # ── ② LoKr + DoRA（走 LoRALinear 的输出域分解）────────────────────────────
    dora = LoRALinear(base, rank=RANK, alpha=float(RANK), use_lokr=True,
                      factor=FACTOR, lora_variant="dora")
    dora.train()
    with torch.no_grad():
        dora.adapter.lokr_w1.copy_(lk.lokr_w1)
        dora.adapter.lokr_w2_a.copy_(lk.lokr_w2_a)
        dora.adapter.lokr_w2_b.copy_(lk.lokr_w2_b)
        # dora_scale 初值就是 ‖W‖_row；扰动一下，免得对拍在"恒等缩放"上蒙混过关
        dora.dora_scale.mul_(1.0 + 0.05 * torch.randn_like(dora.dora_scale))
    out["dora_scale"] = dora.dora_scale.detach().numpy()
    out["y_dora"] = dora(x).detach().numpy()
    out["y_base"] = torch.nn.functional.linear(x, base.weight).detach().numpy()
    out["merged_norm"] = dora.adapter.merged_row_norms(
        base.weight, base_row_sq=None, fast=False).detach().numpy()

    # ── ③ 标准 LoRA + DoRA ───────────────────────────────────────────────────
    base2 = torch.nn.Linear(IN_F, OUT_F, bias=False)
    torch.nn.init.normal_(base2.weight, std=0.05)
    lin = LoRALinear(base2, rank=RANK, alpha=float(RANK), lora_variant="dora")
    lin.train()
    with torch.no_grad():
        torch.nn.init.normal_(lin.adapter.lora_up.weight, std=0.03)
        lin.dora_scale.mul_(1.0 + 0.05 * torch.randn_like(lin.dora_scale))
    out.update(base2_w=base2.weight.detach().numpy(),
               lora_down=lin.adapter.lora_down.weight.detach().numpy(),
               lora_up=lin.adapter.lora_up.weight.detach().numpy(),
               dora2_scale=lin.dora_scale.detach().numpy(),
               y_lora_dora=lin(x).detach().numpy(),
               scaling2=np.asarray(lin.adapter.scaling, np.float32))

    # ── ④ adamw_snr 的参考轨迹 ───────────────────────────────────────────────
    for tag, kw in (("plain", dict(snr_power=1.0, cautious=False)),
                    ("snr15", dict(snr_power=1.5, cautious=False)),
                    ("caut", dict(snr_power=1.0, cautious=True)),
                    ("both", dict(snr_power=1.5, cautious=True))):
        torch.manual_seed(1)
        p = torch.nn.Parameter(torch.randn(5, 9, dtype=torch.float32) * 0.1)
        p0 = p.detach().clone()
        opt = AdamWSNR([p], lr=2e-3, betas=(0.98, 0.999), eps=1e-8,
                       weight_decay=1e-5, **kw)
        grads = []
        for i in range(6):
            g = torch.randn(5, 9, generator=torch.Generator().manual_seed(100 + i))
            grads.append(g.numpy())
            p.grad = g.clone()
            opt.step()
        out[f"opt_{tag}_p0"] = p0.numpy()
        out[f"opt_{tag}_g"] = np.stack(grads)
        out[f"opt_{tag}_p"] = p.detach().numpy()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, **out)
    print(f"已写 {OUT}（{len(out)} 项）")
    print(f"  LoKr 生效 factor={int(out['eff_factor'])} rank={int(out['eff_rank'])} "
          f"scaling={float(out['scaling']):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
