"""packed_per_image_token_loss（融合版逐图 token loss）与逐图 masked_token_loss 的等价性。

navit 训练步曾对每图单独调用 masked_token_loss(全 1 mask)——G 次小 kernel + G 次
全 1 mask 分配。融合版把整包一次 elementwise + 确定性 segment 均值。本测试固化契约：
两者对 mse / l1 / huber(constant|snr) / smooth_l1(sigma) 的值与梯度都一致。
纯 CPU、不需要模型/CUDA。
"""
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from trainer.objective import (
    LossConfig,
    masked_token_loss,
    packed_per_image_token_loss,
)

VSEQ = [4, 12, 6]
M = 32


def _reference_loop(pred, target, vseq, t, cfg):
    """训练循环旧实现：逐图切片 + 全 1 mask 的 masked_token_loss。"""
    off, out = 0, []
    for i, n in enumerate(vseq):
        m = torch.ones(1, n, dtype=pred.dtype)
        out.append(masked_token_loss(
            pred[:, off:off + n, :], target[:, off:off + n, :], m,
            loss_type=cfg.loss_type, huber_c=cfg.huber_c,
            huber_schedule=cfg.huber_schedule, t=t[i].reshape(1).float(),
            huber_snr_clamp_max=cfg.huber_snr_clamp_max,
        ))
        off += n
    return torch.cat(out)


class PackedPerImageLossTests(unittest.TestCase):
    def _data(self, seed=0):
        g = torch.Generator().manual_seed(seed)
        total = sum(VSEQ)
        pred = torch.randn(1, total, M, generator=g)
        target = torch.randn(1, total, M, generator=g)
        t = torch.tensor([0.2, 0.6, 0.9])
        return pred, target, t

    def _check(self, cfg, seed=0):
        pred, target, t = self._data(seed)
        pred_a = pred.clone().requires_grad_(True)
        pred_b = pred.clone().requires_grad_(True)

        fused = packed_per_image_token_loss(pred_a, target, VSEQ, t, cfg)
        ref = _reference_loop(pred_b, target, VSEQ, t, cfg)
        torch.testing.assert_close(fused, ref)

        fused.mean().backward()
        ref.mean().backward()
        torch.testing.assert_close(pred_a.grad, pred_b.grad)

    def test_mse(self):
        self._check(LossConfig(loss_type="mse"))

    def test_l1(self):
        self._check(LossConfig(loss_type="l1"))

    def test_huber_constant(self):
        self._check(LossConfig(loss_type="huber", huber_c=0.1, huber_schedule="constant"))

    def test_huber_snr_per_image_delta(self):
        # snr 调度下 δ 依赖各图自己的 t —— 覆盖逐图 δ 展开成逐 token 向量的路径
        self._check(LossConfig(loss_type="huber", huber_c=0.1, huber_schedule="snr",
                               huber_snr_clamp_max=10.0))

    def test_smooth_l1_sigma(self):
        self._check(LossConfig(loss_type="smooth_l1", huber_c=0.1, huber_schedule="sigma"))

    def test_single_image_pack(self):
        pred = torch.randn(1, 8, M)
        target = torch.randn(1, 8, M)
        t = torch.tensor([0.5])
        cfg = LossConfig(loss_type="huber", huber_schedule="snr")
        fused = packed_per_image_token_loss(pred, target, [8], t, cfg)
        ref = _reference_loop(pred, target, [8], t, cfg)
        torch.testing.assert_close(fused, ref)


if __name__ == "__main__":
    unittest.main()
