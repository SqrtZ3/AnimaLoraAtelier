"""Dispersive Loss CPU 单测（无 2B 模型 / 无 VAE）。

覆盖 trainer/dispersive.py 的纯张量数学：
  - reduce_representation：flatten / mean 两种降维的形状与 no-op 语义。
  - dispersive_loss：
      * 数值有限、形状为 0 维标量；bs<2 返回 0；
      * **方向正确性**：把表征推得更开 → 损失更小（l2 与 cosine 都验）；
      * 梯度回流输入 z；
      * normalize_by_dim 让 l2 损失对"维度复制(放大 D)"不敏感（分辨率鲁棒性代理）；
      * 大距离不溢出成 −inf（logsumexp 稳定性）；
      * 未知 variant 抛错。

Run:  python tests/test_dispersive.py   (also collectable by pytest)
"""

import math
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trainer.dispersive import dispersive_loss, reduce_representation  # noqa: E402


def test_reduce_representation_shapes():
    z = torch.randn(4, 3, 8, 8, 16)        # (B,T,H,W,D) 类网格激活
    flat = reduce_representation(z, pool="flatten")
    assert flat.shape == (4, 3 * 8 * 8 * 16), flat.shape
    mean = reduce_representation(z, pool="mean")
    assert mean.shape == (4, 16), mean.shape   # 保留 B 与末维 channel
    z2 = torch.randn(4, 32)
    assert reduce_representation(z2, pool="flatten").shape == (4, 32)
    assert reduce_representation(z2, pool="mean").shape == (4, 32)   # 2D 时 no-op


def test_scalar_and_finite():
    for variant in ("infonce_l2", "infonce_cosine"):
        z = torch.randn(4, 64)
        l = dispersive_loss(z, variant=variant, tau=0.5)
        assert l.dim() == 0, f"{variant} 应返回标量"
        assert torch.isfinite(l).all(), f"{variant} 非有限"


def test_bs_lt_2_returns_zero():
    z = torch.randn(1, 64)
    l = dispersive_loss(z, variant="infonce_l2")
    assert float(l) == 0.0, "bs<2 应返回 0"


def test_dispersed_has_lower_loss_l2():
    # 同一组点：聚拢 vs 撑开 —— 撑开后 dispersive 损失应更小（损失=log mean exp(-d²/τ)）。
    torch.manual_seed(0)
    base = torch.randn(6, 32)
    tight = base * 0.05            # 几乎重合（高相似/低距离）
    spread = base * 5.0            # 撑得很开
    l_tight = dispersive_loss(tight, variant="infonce_l2", tau=0.5)
    l_spread = dispersive_loss(spread, variant="infonce_l2", tau=0.5)
    assert float(l_spread) < float(l_tight), (float(l_spread), float(l_tight))


def test_dispersed_has_lower_loss_cosine():
    torch.manual_seed(0)
    # 高度共线（方向几乎一致）→ 余弦相似高 → 损失高；正交化后损失应更低。
    v = torch.randn(1, 32)
    aligned = v.repeat(5, 1) + 0.01 * torch.randn(5, 32)
    orthoish = torch.randn(5, 32)
    l_aligned = dispersive_loss(aligned, variant="infonce_cosine", tau=0.5)
    l_ortho = dispersive_loss(orthoish, variant="infonce_cosine", tau=0.5)
    assert float(l_ortho) < float(l_aligned), (float(l_ortho), float(l_aligned))


def test_gradient_flows():
    z = torch.randn(4, 48, requires_grad=True)
    l = dispersive_loss(z, variant="infonce_l2", tau=0.5)
    l.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    assert z.grad.abs().sum() > 0, "梯度应非零"


def test_normalize_by_dim_invariance():
    # normalize_by_dim=True 时，把每个样本的特征"复制一份拼接"(维度×2、每元素重复)
    # 不应显著改变 l2 dispersive 损失（逐元素均方距离不变）。这是分辨率鲁棒性的代理检验。
    torch.manual_seed(1)
    # 用小幅度 z，让 raw 模式的成对项落在非下溢区间（否则 off-diagonal 都 → 0、退化成
    # 常数 −log(B)，对照组失去敏感度，无法证明归一确实起作用）。
    z = torch.randn(5, 40) * 0.1
    z2 = torch.cat([z, z], dim=1)        # 维度翻倍、逐元素均方距离不变
    l1 = dispersive_loss(z, variant="infonce_l2", tau=0.5, normalize_by_dim=True)
    l2 = dispersive_loss(z2, variant="infonce_l2", tau=0.5, normalize_by_dim=True)
    assert abs(float(l1) - float(l2)) < 1e-4, (float(l1), float(l2))
    # 不归一时，维度翻倍会让距离翻倍 → 损失明显变化（对照，证明归一确实起作用）。
    l1_raw = dispersive_loss(z, variant="infonce_l2", tau=0.5, normalize_by_dim=False)
    l2_raw = dispersive_loss(z2, variant="infonce_l2", tau=0.5, normalize_by_dim=False)
    assert abs(float(l1_raw) - float(l2_raw)) > 1e-3, (float(l1_raw), float(l2_raw))


def test_large_distance_no_overflow():
    # 极大距离：朴素 log(mean(exp(-d²/τ))) 会下溢成 log(0)=-inf；logsumexp 应给有限值。
    z = torch.randn(4, 64) * 1e3
    l = dispersive_loss(z, variant="infonce_l2", tau=0.5, normalize_by_dim=False)
    assert torch.isfinite(l).all(), f"大距离下应有限，得到 {float(l)}"


def test_unknown_variant_raises():
    try:
        dispersive_loss(torch.randn(4, 8), variant="nope")
    except ValueError:
        return
    raise AssertionError("未知 variant 应抛 ValueError")


def main():
    tests = [
        test_reduce_representation_shapes,
        test_scalar_and_finite,
        test_bs_lt_2_returns_zero,
        test_dispersed_has_lower_loss_l2,
        test_dispersed_has_lower_loss_cosine,
        test_gradient_flows,
        test_normalize_by_dim_invariance,
        test_large_distance_no_overflow,
        test_unknown_variant_raises,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
