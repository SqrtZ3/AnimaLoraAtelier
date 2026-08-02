# -*- coding: utf-8 -*-
"""`save_state_every_epochs`（按 epoch 存训练状态）的回归测试。

背景：原来只有按 step 的 `save_state_every`。每个 epoch 的 step 数在 ARB / navit 打包 /
sample-window accumulation 下都不是固定值（`len(dataloader)` 随分桶变化），按 step 的
cadence 很难正好落在 epoch 边界上，于是"每个 epoch 末尾一定有一个可续训的点"这件事
无法保证。新开关在 epoch 末尾直接触发一次 `save_training_state`。

这里验证三件事：
1. YAML → args 的接线（key 在 YAML_TO_ARGS、默认 0=关、YAML 值能生效）；
2. 默认关时行为中立（不产生任何 epoch 触发）；
3. epoch 末尾存下的 `epoch_position` 能被 resume 正确消费 —— 两种落点都不重不漏：
   a) 最后一个 batch 恰好完成了 optimizer step（batch_in_epoch == 本 epoch batch 数）；
   b) 末尾还剩一个未参与 step 的 micro-batch（grad_accum 没整除，batch_in_epoch = n-1）。
   (b) 是 epoch cadence 比 step cadence 更容易撞上的落点：epoch 末尾并不保证是
   累积窗口的边界，此时 resume 必须把那个 batch 重跑一遍（它的梯度还没进权重）。
"""
import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from trainer.config import DEFAULTS, YAML_TO_ARGS, apply_yaml_config

# 复用姊妹测试里的 epoch/skip 语义模拟器（与 anima_train.py 主循环同构）
from test_resume_epoch_position import _make_loader, _run_epochs


# ---------------------------------------------------------------- 1. 配置接线

def test_yaml_key_is_wired():
    assert YAML_TO_ARGS.get("save_state_every_epochs") == "save_state_every_epochs"
    assert DEFAULTS["save_state_every_epochs"] == 0, "新功能必须 default-off"


def test_yaml_value_reaches_args():
    args = argparse.Namespace(save_state_every_epochs=0, optimizer_args={})
    apply_yaml_config(args, {"save_state_every_epochs": 3})
    assert args.save_state_every_epochs == 3


def test_absent_yaml_key_keeps_default():
    args = argparse.Namespace(save_state_every_epochs=0, optimizer_args={})
    apply_yaml_config(args, {})
    assert args.save_state_every_epochs == 0


# ---------------------------------------------------------------- 2. 触发节奏

def _triggered_epochs(n, total_epochs):
    """复刻 anima_train.py epoch 末尾的判据：current_epoch = epoch+1（1-based）。"""
    return [e + 1 for e in range(total_epochs)
            if n > 0 and (e + 1) % n == 0]


def test_disabled_by_default_triggers_nothing():
    assert _triggered_epochs(0, 10) == []


def test_cadence_counts_completed_epochs():
    assert _triggered_epochs(1, 5) == [1, 2, 3, 4, 5]
    assert _triggered_epochs(2, 6) == [2, 4, 6]
    assert _triggered_epochs(3, 7) == [3, 6]


# ---------------------------------------------------------------- 3. resume 语义

def _epoch_batch_count(epoch):
    loader = _make_loader()
    loader.batch_sampler.set_epoch(epoch)
    return sum(1 for _ in loader)


def test_state_saved_at_epoch_end_resumes_without_replay():
    """落点 (a)：epoch 最后一个 batch 完成了 step → resume 应整段跳过该 epoch。"""
    reference, _ = _run_epochs(_make_loader(), epochs=3)
    n0 = _epoch_batch_count(0)

    part1, pos = _run_epochs(_make_loader(), epochs=3, stop_after=n0)
    saved_epoch, batch_in_epoch, _o, _p = pos
    assert (saved_epoch, batch_in_epoch) == (0, n0)

    part2, _ = _run_epochs(_make_loader(), epochs=3, start_epoch=saved_epoch,
                           skip_first=batch_in_epoch)
    got = [b for _, _, b in part1 + part2]
    want = [b for _, _, b in reference]
    assert got == want, "epoch 末尾存的 state，resume 后 batch 序列与不中断训练不一致"


def test_state_saved_with_pending_microbatch_replays_only_the_pending_one():
    """落点 (b)：epoch 末尾还剩 1 个未进权重的 micro-batch → resume 必须重跑它、且只重跑它。

    `epoch_position` 记的是"最后一次完成 optimizer step"的位置，所以此时
    batch_in_epoch = n-1；快进跳过前 n-1 个后，第 n 个会被正常训练一次。
    """
    reference, _ = _run_epochs(_make_loader(), epochs=3)
    n0 = _epoch_batch_count(0)

    part1, pos = _run_epochs(_make_loader(), epochs=3, stop_after=n0 - 1)
    saved_epoch, batch_in_epoch, _o, _p = pos
    assert (saved_epoch, batch_in_epoch) == (0, n0 - 1)

    part2, _ = _run_epochs(_make_loader(), epochs=3, start_epoch=saved_epoch,
                           skip_first=batch_in_epoch)
    got = [b for _, _, b in part1 + part2]
    want = [b for _, _, b in reference]
    assert got == want, "末尾 pending micro-batch 被漏训或被重复训练"
