"""诊断脚本：对比 LoRA vs ABBA vs LoKR 在 krea2 尺度下的梯度流。

模拟 krea2 训练环境：
- 大维度 Linear (6144 -> 16384, 模拟 SwiGLU.gate/up)
- bf16 模型权重 + autocast
- 相同 rank/alpha/optimizer 设置

用法：
    python AnimaLoraToolkit/tests/diag_adapter_grad.py
"""
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

DIM_IN = 6144    # krea2 features
DIM_OUT = 16384  # krea2 mlpdim (SwiGLU gate/up)
RANK = 32
ALPHA = 32.0
LR = 1e-4


def make_frozen_linear(in_f, out_f, dtype=torch.bfloat16, device="cuda"):
    """模拟 krea2 预训练权重：随机初始化 + 冻结"""
    layer = torch.nn.Linear(in_f, out_f, bias=False, device=device)
    # 用 kaiming 初始化模拟预训练权重的量级
    torch.nn.init.kaiming_normal_(layer.weight, a=5**0.5)
    layer.requires_grad_(False)
    layer = layer.to(dtype=dtype)
    return layer


def inject_and_test(lora_type, device="cuda"):
    """注入指定类型的适配器，运行 forward+backward，返回梯度统计"""
    torch.manual_seed(42)

    # 创建冻结的 Linear 层
    original = make_frozen_linear(DIM_IN, DIM_OUT, dtype=torch.bfloat16, device=device)

    from trainer.lora import LoRAInjector

    injector = LoRAInjector(
        rank=RANK,
        alpha=ALPHA,
        use_lokr=(lora_type == "lokr"),
        use_abba=(lora_type == "abba"),
        factor=4,
        targets=["linear"],
        lora_variant="base",
    )

    # 注入
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = original

    model = TinyModel()
    injector.inject(model)
    model = model.to(device=device)

    # 验证参数
    adapter_params = list(injector.get_params())
    print(f"\n[{lora_type.upper()}] Adapter params: {len(adapter_params)}")
    for p in adapter_params:
        print(f"  shape={tuple(p.shape)}, dtype={p.dtype}, requires_grad={p.requires_grad}")

    # 创建 optimizer
    param_groups = injector.get_param_groups(weight_decay=0.01, base_lr=LR, loraplus_lr_ratio=1.0)
    optimizer = torch.optim.AdamW(param_groups, lr=LR)

    # 创建输入（模拟 krea2 hidden states）
    x = torch.randn(4, 256, DIM_IN, device=device, dtype=torch.bfloat16)  # (B, seq_len, dim)
    target = torch.randn(4, 256, DIM_OUT, device=device, dtype=torch.bfloat16) * 0.1

    # Forward (inside autocast, like training)
    model.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.linear(x)

    # 检查 adapter output 是否为 0（step 0）
    with torch.no_grad():
        base_out = F.linear(x, original.weight)
        adapter_out = out - base_out
        adapter_norm = adapter_out.float().norm().item()
        base_norm = base_out.float().norm().item()
        print(f"  Step 0: base_norm={base_norm:.4f}, adapter_norm={adapter_norm:.6e}, ratio={adapter_norm/max(base_norm,1e-8):.6e}")

    # Backward
    loss = F.mse_loss(out.float(), target.float())
    print(f"  Loss: {loss.item():.6f}")
    loss.backward()

    # 检查梯度
    print(f"  Gradient norms:")
    total_grad_norm = 0.0
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            gnorm = p.grad.float().norm().item()
            total_grad_norm += gnorm ** 2
            if "adapter" in name or "lora" in name or "abba" in name or "lokr" in name:
                print(f"    {name}: grad_norm={gnorm:.6e}, param_norm={p.float().norm().item():.6e}")
        elif p.requires_grad:
            print(f"    {name}: NO GRADIENT (grad is None)")
    total_grad_norm = total_grad_norm ** 0.5
    print(f"  Total grad_norm: {total_grad_norm:.6e}")

    # 检查 optimizer 是否能 step
    optimizer.step()
    optimizer.zero_grad()

    # 检查参数是否被更新
    print(f"  Post-step adapter param norms:")
    for name, p in model.named_parameters():
        if p.requires_grad and ("adapter" in name or "abba" in name or "lokr" in name or "lora" in name):
            print(f"    {name}: param_norm={p.float().norm().item():.6e}")

    return total_grad_norm, adapter_norm


def test_multi_step(lora_type, n_steps=20, device="cuda"):
    """多步训练，观察 loss 是否下降"""
    torch.manual_seed(42)
    original = make_frozen_linear(DIM_IN, DIM_OUT, dtype=torch.bfloat16, device=device)

    from trainer.lora import LoRAInjector

    injector = LoRAInjector(
        rank=RANK, alpha=ALPHA,
        use_lokr=(lora_type == "lokr"),
        use_abba=(lora_type == "abba"),
        factor=4, targets=["linear"], lora_variant="base",
    )

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = original

    model = TinyModel()
    injector.inject(model)
    model = model.to(device=device)

    param_groups = injector.get_param_groups(weight_decay=0.01, base_lr=LR, loraplus_lr_ratio=1.0)
    optimizer = torch.optim.AdamW(param_groups, lr=LR)

    # 固定输入和目标（便于比较）
    torch.manual_seed(123)
    x = torch.randn(4, 256, DIM_IN, device=device, dtype=torch.bfloat16)
    target = torch.randn(4, 256, DIM_OUT, device=device, dtype=torch.bfloat16) * 0.1

    model.train()
    losses = []
    for step in range(n_steps):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.linear(x)
        loss = F.mse_loss(out.float(), target.float())
        loss.backward()

        # 检查梯度
        total_gnorm = 0.0
        for p in injector.get_params():
            if p.grad is not None:
                total_gnorm += p.grad.float().norm().item() ** 2
        total_gnorm = total_gnorm ** 0.5

        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
        if step < 5 or step % 5 == 0:
            print(f"  [{lora_type.upper()}] step {step}: loss={loss.item():.6f}, grad_norm={total_gnorm:.6e}")

    first_loss = losses[0]
    last_loss = losses[-1]
    print(f"  [{lora_type.upper()}] loss change: {first_loss:.6f} -> {last_loss:.6f} ({(last_loss/first_loss - 1)*100:.2f}%)")
    return losses


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Device: {device} ===")
    print(f"=== Dimensions: in={DIM_IN}, out={DIM_OUT}, rank={RANK}, alpha={ALPHA} ===")

    print("\n" + "="*80)
    print("PART 1: Single-step gradient analysis")
    print("="*80)

    results = {}
    for lora_type in ["lora", "abba", "lokr"]:
        try:
            gnorm, anorm = inject_and_test(lora_type, device=device)
            results[lora_type] = (gnorm, anorm)
        except Exception as e:
            print(f"\n[{lora_type.upper()}] ERROR: {e}")
            import traceback; traceback.print_exc()

    print("\n" + "="*80)
    print("Summary: gradient norms comparison")
    print("="*80)
    for lt, (gn, an) in results.items():
        print(f"  {lt.upper():6s}: total_grad_norm={gn:.6e}, step0_adapter_norm={an:.6e}")

    print("\n" + "="*80)
    print("PART 2: Multi-step training (20 steps)")
    print("="*80)

    for lora_type in ["lora", "abba", "lokr"]:
        try:
            test_multi_step(lora_type, n_steps=20, device=device)
        except Exception as e:
            print(f"\n[{lora_type.upper()}] ERROR: {e}")
            import traceback; traceback.print_exc()
