# -*- coding: utf-8 -*-
"""resume 时优化器状态 dtype 复原的回归测试。

历史 bug（2026-07-16 云端实跑踩中）：load_training_state 的 "fp32 master 复原"
无差别把所有浮点状态张量强转 fp32。该 hack 是为 MuonSF/soap 等自定义优化器
（bf16 参数 + fp32 master 态）写的；但 torch 原生 AdamW 的 exp_avg/exp_avg_sq
本来就是按参数 dtype（bf16）创建的，强转 fp32 后 resume 的第一步
torch._foreach_lerp_(exp_avg_fp32, grad_bf16) 直接
RuntimeError: expected dtype float for `end` but got dtype c10::BFloat16。

正确不变式：resume 后状态 dtype == 保存时 dtype（按磁盘原始 dtype 逐个还原）。
"""
import tempfile
from pathlib import Path

import torch

from trainer.checkpoint import load_training_state, save_training_state
from trainer.lora import LoRAInjector


def _toy_setup(dtype=torch.bfloat16):
    class Blocks(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn_q = torch.nn.Linear(32, 32, bias=False)

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([Blocks()])

        def forward(self, x):
            return self.blocks[0].attn_q(x)

    m = Toy().to(dtype)
    m.requires_grad_(False)
    inj = LoRAInjector(rank=4, alpha=4.0, targets=["attn_q"])
    inj.inject(m)
    params = [p for lora in inj.injected.values()
              for p in lora.adapter.parameters() if p.requires_grad]
    return m, inj, params


def _one_step(model, params, optimizer, dtype=torch.bfloat16):
    x = torch.randn(4, 32, dtype=dtype)
    loss = model(x).float().pow(2).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def test_adamw_bf16_state_resume_and_step():
    """torch 原生 AdamW（bf16 态）：resume 后状态保持 bf16，且能继续 step。"""
    torch.manual_seed(0)
    m, inj, params = _toy_setup()
    opt = torch.optim.AdamW(params, lr=1e-3)
    _one_step(m, params, opt)
    assert all(st["exp_avg"].dtype == torch.bfloat16
               for st in opt.state.values()), "前提：AdamW 状态按参数 dtype 创建"

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "state.pt"
        save_training_state(path, inj, opt, epoch=1, global_step=1)

        m2, inj2, params2 = _toy_setup()
        opt2 = torch.optim.AdamW(params2, lr=1e-3)
        _one_step(m2, params2, opt2)  # 先造出状态槽位再 load（与训练脚本时序一致）
        load_training_state(path, inj2, opt2)

    for st in opt2.state.values():
        assert st["exp_avg"].dtype == torch.bfloat16, \
            "AdamW 的 bf16 状态被错误升成 fp32（旧 bug 复现）"
        assert st["exp_avg_sq"].dtype == torch.bfloat16

    # 旧 bug 在这里崩：_foreach_lerp_(exp_avg_fp32, grad_bf16)
    _one_step(m2, params2, opt2)


def test_fp32_master_state_restored_to_fp32():
    """MuonSF/soap 型（bf16 参数 + fp32 master 态）：resume 后必须还原 fp32。

    用 AdamW 手工把状态转成 fp32 模拟 fp32-master 优化器的保存形态——
    load_state_dict 会把它降成 bf16（参数 dtype），复原逻辑要按磁盘 dtype 转回。
    """
    torch.manual_seed(1)
    m, inj, params = _toy_setup()
    opt = torch.optim.AdamW(params, lr=1e-3)
    _one_step(m, params, opt)
    for st in opt.state.values():
        for k, v in list(st.items()):
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                st[k] = v.float()  # 模拟 fp32 master

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "state.pt"
        save_training_state(path, inj, opt, epoch=1, global_step=1)

        m2, inj2, params2 = _toy_setup()
        opt2 = torch.optim.AdamW(params2, lr=1e-3)
        _one_step(m2, params2, opt2)
        load_training_state(path, inj2, opt2)

    for st in opt2.state.values():
        assert st["exp_avg"].dtype == torch.float32, \
            "fp32 master 态没有被还原（会重新触发 muon ulp 冻结/ lerp 崩溃）"
        assert st["exp_avg_sq"].dtype == torch.float32
