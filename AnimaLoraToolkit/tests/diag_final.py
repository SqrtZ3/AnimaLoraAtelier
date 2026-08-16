"""最终验证：ABBA 梯度瓶颈的精确量化 + bf16 加法精度问题确认"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

DIM_IN, DIM_OUT, RANK = 6144, 16384, 32

def test_bf16_addition_precision():
    """验证 bf16 加法精度：base_out (大) + adapter_out (小) 是否丢失精度"""
    print("="*80)
    print("TEST A: bf16 加法精度 -- adapter_out 太小时是否被丢失？")
    print("="*80)
    base = torch.randn(4, 256, DIM_OUT, device="cuda", dtype=torch.bfloat16) * 50  # ~||base||~80000
    for adapter_scale in [100, 10, 1, 0.1, 0.01, 0.001]:
        adapter = torch.randn_like(base) * adapter_scale
        # bf16 加法
        sum_bf16 = (base + adapter).float()
        # fp32 加法 (ground truth)
        sum_fp32 = base.float() + adapter.float()
        # 计算相对误差
        diff = (sum_bf16 - sum_fp32).norm().item()
        adapter_norm = adapter.float().norm().item()
        base_norm = base.float().norm().item()
        print(f"  adapter_norm={adapter_norm:.4e}, base_norm={base_norm:.1f}, "
              f"ratio={adapter_norm/base_norm:.2e}, bf16_error={diff:.4e}, "
              f"adapter_lost={diff/max(adapter_norm,1e-12):.2%}")

def test_abba_vs_lora_gradient_breakdown():
    """精确量化 ABBA 梯度瓶颈的来源"""
    print("\n" + "="*80)
    print("TEST B: ABBA vs LoRA 梯度分解 -- 哪一步导致 62x 衰减？")
    print("="*80)

    torch.manual_seed(42)
    dtype = torch.float32  # fp32 排除精度干扰

    layer = torch.nn.Linear(DIM_IN, DIM_OUT, bias=False, device="cuda")
    torch.nn.init.kaiming_normal_(layer.weight, a=5**0.5)
    layer.requires_grad_(False)
    layer = layer.to(dtype=dtype)

    from trainer.lora import LoRAInjector
    x = torch.randn(4, 256, DIM_IN, device="cuda", dtype=dtype)
    target = torch.randn(4, 256, DIM_OUT, device="cuda", dtype=dtype) * 0.1

    # --- LoRA ---
    torch.manual_seed(42)
    inj_lora = LoRAInjector(rank=RANK, alpha=32.0, targets=["linear"])
    class M1(torch.nn.Module):
        def __init__(self): super().__init__(); self.linear = layer
    m1 = M1(); inj_lora.inject(m1)
    m1 = m1.cuda()

    lora_ad = m1.linear.adapter
    # 手动 forward 分解
    h_lora = lora_ad.lora_down(x)  # (4, 256, 32)
    # grad w.r.t. lora_up = scaling * h^T @ dL/d(out)
    with torch.autocast("cuda", enabled=False):
        out_lora = m1.linear(x)
    loss_lora = F.mse_loss(out_lora, target)
    loss_lora.backward()

    print(f"\n  LoRA:")
    print(f"    ||h_lora|| = {h_lora.norm().item():.4f}  (shape {tuple(h_lora.shape)})")
    print(f"    ||grad_lora_up|| = {lora_ad.lora_up.weight.grad.norm().item():.6e}")
    print(f"    scaling = {lora_ad.scaling}")
    print(f"    ||x|| = {x.norm().item():.4f}")
    print(f"    ||A (lora_down)|| = {lora_ad.lora_down.weight.norm().item():.4f}")

    # --- ABBA ---
    torch.manual_seed(42)
    inj_abba = LoRAInjector(rank=RANK, alpha=32.0, use_abba=True, targets=["linear"])
    class M2(torch.nn.Module):
        def __init__(self): super().__init__(); self.linear = layer
    m2 = M2(); inj_abba.inject(m2)
    m2 = m2.cuda()

    abba_ad = m2.linear.adapter
    # 手动 forward 分解
    a1, a2 = abba_ad.abba_a1, abba_ad.abba_a2
    b1, b2 = abba_ad.abba_b1, abba_ad.abba_b2
    a_kr = (a1.unsqueeze(1) * a2.unsqueeze(0)).reshape(abba_ad.r1 * abba_ad.r2, a1.shape[1])
    b_kr = (b1.unsqueeze(2) * b2.unsqueeze(1)).reshape(b1.shape[0], abba_ad.r1 * abba_ad.r2)
    h_abba = F.linear(x, a_kr)  # (4, 256, 256)

    out_abba = m2.linear(x)
    loss_abba = F.mse_loss(out_abba, target)
    loss_abba.backward()

    print(f"\n  ABBA:")
    print(f"    ||h_abba|| = {h_abba.norm().item():.4f}  (shape {tuple(h_abba.shape)})")
    print(f"    ||a_kr|| = {a_kr.norm().item():.4f}  (shape {tuple(a_kr.shape)})")
    print(f"    ||b_kr|| = {b_kr.norm().item():.4e}  (b2=0 -> b_kr=0)")
    print(f"    ||a1|| = {a1.norm().item():.4f}, ||a2|| = {a2.norm().item():.4f}")
    print(f"    ||b1|| = {b1.norm().item():.4f}, ||b2|| = {b2.norm().item():.4e}")
    print(f"    ||grad_b2|| = {b2.grad.norm().item():.6e}")
    print(f"    scaling = {abba_ad.scaling}")

    # --- 关键对比 ---
    print(f"\n  关键对比:")
    print(f"    ||h_lora|| / ||h_abba|| = {h_lora.norm().item() / h_abba.norm().item():.4f}")
    print(f"    ||grad_lora_up|| / ||grad_b2|| = {lora_ad.lora_up.weight.grad.norm().item() / b2.grad.norm().item():.1f}x")
    print(f"    scaling_lora / scaling_abba = {lora_ad.scaling / abba_ad.scaling:.4f}")

    # --- 模拟 b_kr 梯度 (如果有 b_kr 的梯度) ---
    # grad_b_kr = scaling * h^T @ dL/d(out)
    # 但 b_kr 不是 nn.Parameter, 所以我们手动计算
    dL_dout = 2.0 / (4 * 256 * DIM_OUT) * (out_abba - target)  # MSE grad
    grad_b_kr_manual = abba_ad.scaling * torch.einsum('bsd,bsv->vd', h_abba, dL_dout)
    # Wait, this isn't right. F.linear(h, b_kr) = h @ b_kr^T, so dL/db_kr = dL/dout^T @ h
    # Actually dL/db_kr[i,j] = sum_{b,s} dL/dout[b,s,i] * h[b,s,j]
    grad_b_kr = abba_ad.scaling * torch.einsum('bsi,bsj->ij', dL_dout, h_abba)  # (out, r1*r2)
    print(f"\n    ||grad_b_kr (manual)|| = {grad_b_kr.norm().item():.6e}")

    # grad_b2[i,l] = sum_k b1[i,k] * grad_b_kr[i, k*r2+l]
    r1, r2 = abba_ad.r1, abba_ad.r2
    grad_b_kr_reshaped = grad_b_kr.reshape(DIM_OUT, r1, r2)  # (out, r1, r2)
    grad_b2_manual = torch.einsum('ikl,ik->il', grad_b_kr_reshaped, b1)  # (out, r2)
    print(f"    ||grad_b2 (manual)|| = {grad_b2_manual.norm().item():.6e}")
    print(f"    ||grad_b2 (autograd)|| = {b2.grad.norm().item():.6e}")
    print(f"    ||b1|| = {b1.norm().item():.4f}")
    print(f"    衰减因子 = ||grad_b2|| / (||grad_b_kr|| * ||b1|| / sqrt(r1))")
    expected = grad_b_kr.norm().item() * b1.norm().item() / (r1 ** 0.5)
    print(f"      = {grad_b2_manual.norm().item():.6e} / {expected:.6e} = {grad_b2_manual.norm().item()/expected:.4f}")

    # --- 对比 LoRA 的等价量 ---
    # LoRA: grad_lora_up = scaling * h_lora^T @ dL/dout
    # h_lora shape (4, 256, 32), dL/dout shape (4, 256, 16384)
    grad_lora_up_manual = lora_ad.scaling * torch.einsum('bsi,bsj->ij', dL_dout, h_lora)
    print(f"\n    ||grad_lora_up (manual)|| = {grad_lora_up_manual.norm().item():.6e}")
    print(f"    ||h_lora|| = {h_lora.norm().item():.4f}")
    print(f"    衰减比: ||grad_b2|| / ||grad_lora_up|| = {grad_b2_manual.norm().item() / grad_lora_up_manual.norm().item():.6f}")
    print(f"    即 ABBA b2 的梯度是 LoRA lora_up 的 {grad_b2_manual.norm().item() / grad_lora_up_manual.norm().item() * 100:.2f}%")

    # --- 分解衰减来源 ---
    print(f"\n  衰减分解:")
    ratio_h = h_abba.norm().item() / h_lora.norm().item()
    ratio_scaling = abba_ad.scaling / lora_ad.scaling
    ratio_kr = grad_b2_manual.norm().item() / (grad_b_kr.norm().item() * b1.norm().item() / (r1 ** 0.5))
    total = ratio_h * ratio_scaling * ratio_kr
    print(f"    h_ratio (||h_abba||/||h_lora||) = {ratio_h:.4f}")
    print(f"    scaling_ratio (abba/lora) = {ratio_scaling:.4f}")
    print(f"    kr_contraction (实际/理论) = {ratio_kr:.4f}")
    print(f"    总衰减 = {total:.6f}")
    print(f"    实测衰减 = {grad_b2_manual.norm().item() / grad_lora_up_manual.norm().item():.6f}")


if __name__ == "__main__":
    test_bf16_addition_precision()
    test_abba_vs_lora_gradient_breakdown()
