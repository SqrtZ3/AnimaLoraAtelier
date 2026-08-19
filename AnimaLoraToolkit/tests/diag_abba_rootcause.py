"""深度诊断：ABBA 梯度远小于 LoRA 的根因分析。

测试假设：
1. bf16 精度问题 -> 用 fp32 对比
2. SVD 初始化问题 -> 用随机初始化对比
3. scaling 问题 -> 调整 scaling 对比
4. KR 结构本身的梯度衰减
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

DIM_IN, DIM_OUT, RANK, ALPHA = 6144, 16384, 32, 32.0


def test_single_layer(lora_type, dtype_str="bf16", svd_init=True, custom_scaling=None, device="cuda"):
    """单层测试，返回梯度范数和 adapter 输出范数"""
    torch.manual_seed(42)
    dtype = torch.bfloat16 if dtype_str == "bf16" else torch.float32

    # 创建冻结的 Linear 层
    layer = torch.nn.Linear(DIM_IN, DIM_OUT, bias=False, device=device)
    torch.nn.init.kaiming_normal_(layer.weight, a=5**0.5)
    layer.requires_grad_(False)
    layer = layer.to(dtype=dtype)

    from trainer.lora import LoRAInjector, LoRALinear, ABBALayer, LoRALayer, LoKrLayer

    if lora_type == "lora":
        injector = LoRAInjector(rank=RANK, alpha=ALPHA, targets=["linear"], lora_variant="base")
    elif lora_type == "abba":
        injector = LoRAInjector(rank=RANK, alpha=ALPHA, use_abba=True, targets=["linear"], lora_variant="base")
    elif lora_type == "lokr":
        injector = LoRAInjector(rank=RANK, alpha=ALPHA, use_lokr=True, factor=4, targets=["linear"], lora_variant="base")

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = layer
    model = M()
    injector.inject(model)
    model = model.to(device=device)

    # 如果是 ABBA 且不要 SVD init，用随机初始化
    if lora_type == "abba" and not svd_init:
        for lora in injector.injected.values():
            ad = lora.adapter
            with torch.no_grad():
                torch.nn.init.kaiming_uniform_(ad.abba_a1)
                torch.nn.init.kaiming_uniform_(ad.abba_b1)
                torch.nn.init.kaiming_uniform_(ad.abba_a2)
                torch.nn.init.zeros_(ad.abba_b2)

    # 如果有自定义 scaling
    if custom_scaling is not None:
        for lora in injector.injected.values():
            lora.adapter.scaling = custom_scaling

    # 输入
    torch.manual_seed(123)
    x = torch.randn(4, 256, DIM_IN, device=device, dtype=dtype)
    target = torch.randn(4, 256, DIM_OUT, device=device, dtype=dtype) * 0.1

    model.train()
    use_autocast = (dtype_str == "bf16")
    with torch.autocast("cuda", dtype=torch.bfloat16) if use_autocast else torch.cuda.amp.autocast(enabled=False):
        out = model.linear(x)

    # 检查 adapter output
    with torch.no_grad():
        base_out = F.linear(x, layer.weight)
        adapter_out = (out.float() - base_out.float()) if use_autocast else (out - base_out)
        adapter_norm = adapter_out.float().norm().item()
        base_norm = base_out.float().norm().item()

    loss = F.mse_loss(out.float(), target.float())
    loss.backward()

    # 收集梯度
    grads = {}
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            gnorm = p.grad.float().norm().item()
            if "adapter" in name or "abba" in name or "lokr" in name or "lora" in name:
                short_name = name.split(".")[-1] if "abba" in name or "lokr" in name else name.split(".")[-1]
                grads[short_name] = (gnorm, p.float().norm().item())

    total_gnorm = sum(g**2 for g, _ in grads.values()) ** 0.5
    return total_gnorm, adapter_norm, base_norm, grads, loss.item()


def test_multi_step_compare(lora_type, n_steps=50, dtype_str="bf16", device="cuda"):
    """多步训练对比"""
    torch.manual_seed(42)
    dtype = torch.bfloat16 if dtype_str == "bf16" else torch.float32

    layer = torch.nn.Linear(DIM_IN, DIM_OUT, bias=False, device=device)
    torch.nn.init.kaiming_normal_(layer.weight, a=5**0.5)
    layer.requires_grad_(False)
    layer = layer.to(dtype=dtype)

    from trainer.lora import LoRAInjector
    kw = dict(rank=RANK, alpha=ALPHA, targets=["linear"], lora_variant="base")
    if lora_type == "abba":
        kw["use_abba"] = True
    elif lora_type == "lokr":
        kw["use_lokr"] = True
        kw["factor"] = 4

    injector = LoRAInjector(**kw)
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = layer
    model = M()
    injector.inject(model)
    model = model.to(device=device)

    pg = injector.get_param_groups(weight_decay=0.01, base_lr=1e-4, loraplus_lr_ratio=1.0)
    opt = torch.optim.AdamW(pg, lr=1e-4)

    torch.manual_seed(123)
    x = torch.randn(4, 256, DIM_IN, device=device, dtype=dtype)
    target = torch.randn(4, 256, DIM_OUT, device=device, dtype=dtype) * 0.1

    use_autocast = (dtype_str == "bf16")
    model.train()
    first_loss = None
    for step in range(n_steps):
        with torch.autocast("cuda", dtype=torch.bfloat16) if use_autocast else torch.cuda.amp.autocast(enabled=False):
            out = model.linear(x)
        loss = F.mse_loss(out.float(), target.float())
        loss.backward()

        # 收集 adapter 梯度范数
        gnorm = 0.0
        for p in injector.get_params():
            if p.grad is not None:
                gnorm += p.grad.float().norm().item() ** 2
        gnorm = gnorm ** 0.5

        # 检查 adapter output
        with torch.no_grad():
            base_out = F.linear(x, layer.weight)
            adapter_out = out.float() - base_out.float()
            anorm = adapter_out.norm().item()

        opt.step()
        opt.zero_grad()

        if first_loss is None:
            first_loss = loss.item()
        if step < 3 or step % 10 == 0 or step == n_steps - 1:
            print(f"  step {step:3d}: loss={loss.item():.8f}, grad={gnorm:.4e}, adapter_out={anorm:.4e}")

    print(f"  -> loss: {first_loss:.8f} -> {loss.item():.8f} ({(loss.item()/first_loss - 1)*100:.3f}%)")
    return loss.item()


if __name__ == "__main__":
    device = "cuda"
    print(f"=== dims: in={DIM_IN}, out={DIM_OUT}, rank={RANK}, alpha={ALPHA} ===\n")

    # ===== Test 1: bf16 vs fp32 =====
    print("="*80)
    print("TEST 1: bf16 vs fp32 — 是否 bf16 精度导致梯度差异？")
    print("="*80)
    for dtype_str in ["bf16", "fp32"]:
        print(f"\n--- {dtype_str} ---")
        for lt in ["lora", "abba"]:
            gn, an, bn, grads, loss = test_single_layer(lt, dtype_str=dtype_str, device=device)
            print(f"  [{lt.upper():4s}] grad={gn:.4e}, adapter_out={an:.4e}, base_out={bn:.2f}, loss={loss:.6f}")
            for pname, (gnorm, pnorm) in grads.items():
                print(f"         {pname:20s}: grad={gnorm:.4e}, param={pnorm:.4e}")

    # ===== Test 2: ABBA with/without SVD init =====
    print("\n" + "="*80)
    print("TEST 2: ABBA SVD init vs random init — 是否 SVD 初始化导致梯度差异？")
    print("="*80)
    for svd in [True, False]:
        print(f"\n--- ABBA svd_init={svd} ---")
        gn, an, bn, grads, loss = test_single_layer("abba", dtype_str="fp32", svd_init=svd, device=device)
        print(f"  grad={gn:.4e}, adapter_out={an:.4e}, loss={loss:.6f}")
        for pname, (gnorm, pnorm) in grads.items():
            print(f"    {pname:20s}: grad={gnorm:.4e}, param={pnorm:.4e}")

    # ===== Test 3: Scaling comparison =====
    print("\n" + "="*80)
    print("TEST 3: ABBA scaling — scaling=16 (default) vs scaling=1.0")
    print("="*80)
    for sc in [16.0, 1.0, 4.0]:
        print(f"\n--- ABBA scaling={sc} ---")
        gn, an, bn, grads, loss = test_single_layer("abba", dtype_str="fp32", custom_scaling=sc, device=device)
        b2_grad = [g for n, (g, p) in grads.items() if "b2" in n]
        b2_g = b2_grad[0] if b2_grad else 0
        print(f"  grad={gn:.4e}, b2_grad={b2_g:.4e}, adapter_out={an:.4e}, loss={loss:.6f}")

    # ===== Test 4: Multi-step fp32 =====
    print("\n" + "="*80)
    print("TEST 4: Multi-step fp32 — fp32 下 ABBA 是否能收敛？")
    print("="*80)
    for lt in ["lora", "abba"]:
        print(f"\n--- {lt.upper()} fp32, 50 steps ---")
        test_multi_step_compare(lt, n_steps=50, dtype_str="fp32", device=device)

    # ===== Test 5: Multi-step bf16 =====
    print("\n" + "="*80)
    print("TEST 5: Multi-step bf16 — bf16 下行为对比")
    print("="*80)
    for lt in ["lora", "abba"]:
        print(f"\n--- {lt.upper()} bf16, 50 steps ---")
        test_multi_step_compare(lt, n_steps=50, dtype_str="bf16", device=device)
