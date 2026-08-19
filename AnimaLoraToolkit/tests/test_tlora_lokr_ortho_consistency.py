"""T-LoRA × LoKr ortho init 的训练/导出一致性测试。

Bug 背景（dev/emosens-optimizer，2026-06）：
  tlora_lokr_ortho_init=True 时训练 forward 会减去 init 贡献
  kron(w1_init, w2a_init @ w2b_init) * scaling 使 step-0 净 ΔW=0，
  但 delta_weight() / merged_weight / diff / merged_model 导出都不减
  → 训练语义与导出权重相差一个 init delta。
  且三个 init buffer persistent=False 又不进 LoKr state_dict
  → resume 时重新随机生成，补偿基准漂移。

pytest 可跑；也可直接 `python tests/test_tlora_lokr_ortho_consistency.py`。
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from trainer.lora import LoKrLayer, LoRALinear, LoRAInjector  # noqa: E402


def _train_forward_delta(layer_or_lora, in_features, t=0.0):
    """用单位矩阵探测训练 forward 的等效 ΔW（t=0 → T-LoRA mask 满 rank）。"""
    x = torch.eye(in_features)
    if isinstance(layer_or_lora, LoRALinear):
        layer_or_lora.set_current_t(torch.tensor([t]))
        y = layer_or_lora(x) - layer_or_lora.original(x)
        layer_or_lora.set_current_t(None)
    else:
        layer_or_lora.set_current_t(torch.tensor([t]))
        y = layer_or_lora(x)
        layer_or_lora.set_current_t(None)
    return y.transpose(0, 1).detach().float()   # y = x @ ΔW^T → ΔW = y^T


def _perturb(layer, std=0.05):
    with torch.no_grad():
        layer.lokr_w1.add_(std * torch.randn_like(layer.lokr_w1))
        layer.lokr_w2_a.add_(std * torch.randn_like(layer.lokr_w2_a))
        layer.lokr_w2_b.add_(std * torch.randn_like(layer.lokr_w2_b))


def test_lokr_ortho_train_forward_matches_delta_weight():
    """最小复现：ortho_init=True 时 delta_weight() 必须等于训练 forward 的净 ΔW。

    修复前：delta_weight 不减 init 项，两者相差恰好
    kron(w1_init, w2a_init @ w2b_init) * scaling ≠ 0。
    """
    torch.manual_seed(0)
    layer = LoKrLayer(16, 16, rank=4, alpha=4.0, factor=4,
                      tlora_enabled=True, tlora_rmin_ratio=0.5,
                      tlora_lokr_ortho_init=True)
    layer.train()

    # init delta 本身非零（否则测试是平凡的）
    init_kron = torch.kron(
        layer.lokr_w1_init.float(),
        layer.lokr_w2_a_init.float() @ layer.lokr_w2_b_init.float(),
    ) * layer.scaling
    assert init_kron.abs().max().item() > 1e-3

    # step 0：forward 净 ΔW ≈ 0（ortho 补偿），导出 delta 也必须 ≈ 0
    d_train0 = _train_forward_delta(layer, 16)
    assert d_train0.abs().max().item() < 1e-5
    d_export0 = layer.delta_weight(apply_rank_dropout=False).float()
    assert d_export0.abs().max().item() < 1e-5, \
        f"step-0 delta_weight 应为 0，实际 max={d_export0.abs().max().item():.4e}（= init delta 未补偿）"

    # 模拟训练后权重漂移：两条路径仍须一致
    _perturb(layer)
    d_train = _train_forward_delta(layer, 16)
    d_export = layer.delta_weight(apply_rank_dropout=False).float()
    assert torch.allclose(d_export, d_train, atol=1e-5), \
        (d_export - d_train).abs().max().item()
    # 显式公式核对：净 ΔW = kron(w1, w2a@w2b)·s − kron(w1_init, w2a_init@w2b_init)·s
    expected = torch.kron(
        layer.lokr_w1.detach().float(),
        layer.lokr_w2_a.detach().float() @ layer.lokr_w2_b.detach().float(),
    ) * layer.scaling - init_kron
    assert torch.allclose(d_export, expected, atol=1e-5)


def test_lokr_ortho_merged_weight_consistency():
    """LoRALinear.merged_weight（diff/merged_model 导出的底层）同样要减 init 项。"""
    torch.manual_seed(1)
    base = torch.nn.Linear(16, 16, bias=False)
    lora = LoRALinear(base, rank=4, alpha=4.0, use_lokr=True, factor=4,
                      lora_variant="tlora", tlora_lokr_ortho_init=True)
    lora.train()

    # step 0：merged == base（净 ΔW=0）
    merged0 = lora.merged_weight().float()
    assert torch.allclose(merged0, base.weight.detach().float(), atol=1e-5), \
        (merged0 - base.weight.float()).abs().max().item()

    # 漂移后：merged − base == 训练 forward 净 ΔW
    _perturb(lora.adapter)
    d_train = _train_forward_delta(lora, 16)
    merged = lora.merged_weight().float()
    assert torch.allclose(merged - base.weight.detach().float(), d_train, atol=1e-5)


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(16, 16, bias=False)


def _make_injector(**overrides):
    kw = dict(rank=4, alpha=4.0, use_lokr=True, factor=4,
              lora_variant="tlora", tlora_lokr_experimental=True,
              tlora_lokr_ortho_init=True, dora_export_mode="diff",
              targets=["q_proj"])
    kw.update(overrides)
    return LoRAInjector(**kw)


def test_lokr_ortho_state_dict_resume_roundtrip():
    """init buffers 必须随 checkpoint 保存并在 resume 时恢复（否则补偿基准重随机）。"""
    torch.manual_seed(2)
    model1 = _TinyModel()
    inj1 = _make_injector()
    inj1.inject(model1)
    a1 = inj1.injected["q_proj"].adapter
    _perturb(a1)
    sd = inj1.state_dict()

    base_key = "lora_unet_q_proj"
    for suffix in ("lokr_w1_init", "lokr_w2_a_init", "lokr_w2_b_init"):
        assert f"{base_key}.{suffix}" in sd, f"缺少 {suffix}（resume 必需）"

    # 不同 seed → 第二个 injector 的随机 init 必然不同；load 后必须被 checkpoint 覆盖
    torch.manual_seed(99)
    model2 = _TinyModel()
    inj2 = _make_injector()
    inj2.inject(model2)
    a2 = inj2.injected["q_proj"].adapter
    assert not torch.allclose(a2.lokr_w2_a_init.float(), a1.lokr_w2_a_init.float())

    loaded = inj2.load_state_dict_from_mapping(sd)
    assert loaded == 1
    # 存储为 bf16 → 恢复值应与原 buffer 的 bf16 投影逐 bit 一致
    for attr in ("lokr_w1_init", "lokr_w2_a_init", "lokr_w2_b_init"):
        restored = getattr(a2, attr).float()
        original = getattr(a1, attr).to(torch.bfloat16).float()
        assert torch.equal(restored, original), attr
    # 功能一致性：resume 后的净 ΔW 与原 run 一致（容忍 bf16 存储量化）
    d1 = a1.delta_weight(apply_rank_dropout=False).float()
    d2 = a2.delta_weight(apply_rank_dropout=False).float()
    assert torch.allclose(d2, d1, rtol=0.05, atol=5e-3), (d2 - d1).abs().max().item()


def test_lokr_ortho_native_export_mode_rejected():
    """净 ΔW 是两个 kron 之差，无法表达为标准 LoKr key → native 导出模式直接拒绝。"""
    try:
        _make_injector(dora_export_mode="native")
        assert False, "ortho_init=True + dora_export_mode='native' 应在构造时 raise"
    except ValueError as e:
        assert "native" in str(e)
    # diff / merged_model 可精确补偿 → 允许
    _make_injector(dora_export_mode="diff")
    _make_injector(dora_export_mode="merged_model")


def test_lokr_no_ortho_unaffected():
    """回归保护：ortho_init=False（推荐路径）行为不变 — 无 init key、native 可用、
    delta_weight 仍是单个 kron。"""
    torch.manual_seed(3)
    model = _TinyModel()
    inj = _make_injector(tlora_lokr_ortho_init=False, dora_export_mode="native")
    inj.inject(model)
    a = inj.injected["q_proj"].adapter
    # w2_b=0 起步 → step-0 ΔW 天然为 0
    assert a.delta_weight(apply_rank_dropout=False).abs().max().item() == 0.0
    _perturb(a)
    d = a.delta_weight(apply_rank_dropout=False).float()
    expected = torch.kron(
        a.lokr_w1.detach().float(),
        a.lokr_w2_a.detach().float() @ a.lokr_w2_b.detach().float(),
    ) * a.scaling
    assert torch.allclose(d, expected, atol=1e-6)
    sd = inj.state_dict()
    assert not any(k.endswith("_init") for k in sd), "非 ortho 路径不应出现 init key"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nall {len(fns)} tests passed")
