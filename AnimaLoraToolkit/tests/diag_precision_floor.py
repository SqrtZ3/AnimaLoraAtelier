"""验证：bf16 精度地板假设 -- 为什么 LoKR 在 Anima 上工作但 Krea2 上失败。

核心假设：
- bf16 有 ~0.78% 的相对精度（7 bit mantissa）
- 如果 adapter_out / base_out < 0.78%，adapter 贡献在 bf16 加法中被丢失
- Anima: features=2048, lr=2e-3 -> adapter 贡献在精度地板以上
- Krea2: features=6144, lr=4e-4 -> adapter 贡献在精度地板以下

验证方法：模拟不同维度的单层训练，观察 adapter_out / base_out 比值
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

BF16_FLOOR = 2**(-7)  # ≈ 0.78%, bf16 的相对精度下限

def test_precision_floor(dim_in, dim_out, lora_type, lr, n_steps=30, device="cuda"):
    """测试不同维度下 adapter 输出与 base 输出的比值"""
    torch.manual_seed(42)
    dtype = torch.bfloat16

    layer = torch.nn.Linear(dim_in, dim_out, bias=False, device=device)
    torch.nn.init.kaiming_normal_(layer.weight, a=5**0.5)
    layer.requires_grad_(False)
    layer = layer.to(dtype=dtype)

    from trainer.lora import LoRAInjector
    kw = dict(rank=32, alpha=32.0, targets=["linear"], lora_variant="base")
    if lora_type == "abba":
        kw["use_abba"] = True
    elif lora_type == "lokr":
        kw["use_lokr"] = True
        kw["factor"] = 4

    injector = LoRAInjector(**kw)
    class M(torch.nn.Module):
        def __init__(self): super().__init__(); self.linear = layer
    model = M(); injector.inject(model); model = model.cuda()

    pg = injector.get_param_groups(weight_decay=0.01, base_lr=lr, loraplus_lr_ratio=1.0)
    opt = torch.optim.AdamW(pg, lr=lr)

    torch.manual_seed(123)
    x = torch.randn(4, 256, dim_in, device=device, dtype=dtype)
    target = torch.randn(4, 256, dim_out, device=device, dtype=dtype) * 0.1

    model.train()
    results = []
    for step in range(n_steps):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.linear(x)
        loss = F.mse_loss(out.float(), target.float())
        loss.backward()

        with torch.no_grad():
            base_out = F.linear(x, layer.weight)
            adapter_out = out.float() - base_out.float()
            ratio = adapter_out.norm().item() / max(base_out.float().norm().item(), 1e-8)
            above_floor = "YES" if ratio > BF16_FLOOR else "NO"

        gnorm = 0.0
        for p in injector.get_params():
            if p.grad is not None:
                gnorm += p.grad.float().norm().item() ** 2
        gnorm = gnorm ** 0.5

        opt.step()
        opt.zero_grad()

        if step < 3 or step % 5 == 0 or step == n_steps - 1:
            results.append((step, loss.item(), ratio, gnorm, above_floor))

    return results

if __name__ == "__main__":
    print(f"bf16 精度地板 = {BF16_FLOOR*100:.2f}% (adapter/base 比值低于此 = 丢失)")
    print()

    configs = [
        # (label, dim_in, dim_out, lora_type, lr)
        ("Anima LoRA  2048→8192  lr=2e-3", 2048, 8192, "lora", 2e-3),
        ("Anima LoKR  2048→8192  lr=2e-3", 2048, 8192, "lokr", 2e-3),
        ("Krea2 LoRA  6144→16384 lr=1e-4", 6144, 16384, "lora", 1e-4),
        ("Krea2 LoKR  6144→16384 lr=4e-4", 6144, 16384, "lokr", 4e-4),
        ("Krea2 LoKR  6144→16384 lr=2e-3", 6144, 16384, "lokr", 2e-3),
        ("Krea2 ABBA  6144→16384 lr=1e-4", 6144, 16384, "abba", 1e-4),
    ]

    for label, din, dout, lt, lr in configs:
        print(f"--- {label} ---")
        print(f"  {'step':>4s}  {'loss':>10s}  {'adapter/base':>12s}  {'above_floor':>11s}  {'grad_norm':>10s}")
        results = test_precision_floor(din, dout, lt, lr)
        for step, loss, ratio, gnorm, above in results:
            print(f"  {step:4d}  {loss:10.6f}  {ratio*100:10.4f}%  {above:>11s}  {gnorm:10.4e}")
        # 判断是否卡在精度地板以下
        final_ratio = results[-1][2]
        if final_ratio < BF16_FLOOR:
            print(f"  !! FINAL adapter/base = {final_ratio*100:.4f}% < {BF16_FLOOR*100:.2f}% -> STUCK below precision floor!")
        else:
            print(f"  OK FINAL adapter/base = {final_ratio*100:.4f}% > {BF16_FLOOR*100:.2f}% -> above precision floor")
        print()
