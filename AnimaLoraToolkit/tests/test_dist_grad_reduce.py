"""梯度 all-reduce 的**数值**对拍 —— 真起两个进程、真走 gloo 集合通信，CPU 即可跑。

为什么值得单独一个文件：``all_reduce_grads_`` 里那个 "``grad is None`` 也要占位" 的
处理，错了**不会报错**。少放一个占位 → 各 rank 的打平缓冲布局错位 → all-reduce 出来的
是逐元素串位的垃圾梯度，训练照常进行、loss 照常下降，只是在训一个错的东西。
这类 bug 只能靠"两个 rank 拿不同的 None 模式"的对拍抓出来，本文件就是干这个的。

同时验证 ``all_ranks_clean`` 的跨 rank 逻辑与 —— 它是多卡不死锁的前提。

跑法：``pytest tests/test_dist_grad_reduce.py``（不需要 GPU，不需要多卡）。
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")


WORLD = 2


def _worker(rank, port, out_q):
    """两个 rank 各自造一组梯度，跑一次 all_reduce_grads_，把结果回传给父进程。"""
    import torch.distributed as dist

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import dist_utils

    # gloo 会去解析本机 hostname 找网卡；Windows 上常解析失败（"unsupported gloo
    # device"）。显式钉到回环地址 + 指定网卡名，两个平台都能起来。
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(port))
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    try:
        dist.init_process_group(
            backend="gloo", init_method="env://", world_size=WORLD, rank=rank,
        )
    except Exception as e:
        # 环境起不来不是被测代码的问题 —— 回传标记让父进程 skip，而不是报 fail
        out_q.put({"rank": rank, "init_error": f"{type(e).__name__}: {e}"})
        return
    ctx = dist_utils.DistContext(enabled=True, rank=rank, world_size=WORLD,
                                 backend="gloo", device="cpu")

    # 三个参数，刻意构造非对称的 grad=None 模式：
    #   p0: 两个 rank 都有梯度
    #   p1: 只有 rank0 有（模拟 module_dropout 在 rank1 上跳过了这个模块）
    #   p2: 只有 rank1 有
    p0 = torch.nn.Parameter(torch.zeros(4))
    p1 = torch.nn.Parameter(torch.zeros(3))
    p2 = torch.nn.Parameter(torch.zeros(2))
    params = [p0, p1, p2]

    p0.grad = torch.full((4,), 1.0 if rank == 0 else 3.0)
    if rank == 0:
        p1.grad = torch.full((3,), 8.0)
    else:
        p2.grad = torch.full((2,), 6.0)

    ctx.all_reduce_grads_(params)

    clean_all = ctx.all_ranks_clean(True)
    clean_one_dirty = ctx.all_ranks_clean(rank == 0)   # rank1 报 dirty

    out_q.put({
        "rank": rank,
        "g0": p0.grad.tolist(),
        "g1": p1.grad.tolist(),
        "g2": p2.grad.tolist(),
        "clean_all": clean_all,
        "clean_one_dirty": clean_one_dirty,
    })
    dist.destroy_process_group()


def _free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_grad_average_across_ranks_with_asymmetric_none():
    import multiprocessing as mp

    ctx_mp = mp.get_context("spawn")
    q = ctx_mp.Queue()
    port = _free_port()
    procs = [ctx_mp.Process(target=_worker, args=(r, port, q)) for r in range(WORLD)]
    for p in procs:
        p.start()
    results = {}
    try:
        for _ in range(WORLD):
            r = q.get(timeout=300)
            results[r["rank"]] = r
    finally:
        for p in procs:
            p.join(timeout=60)
            if p.is_alive():
                p.terminate()

    init_errs = [r["init_error"] for r in results.values() if "init_error" in r]
    if init_errs:
        pytest.skip(f"本机起不了 gloo 进程组，跳过（Linux/DCU 上应能跑）：{init_errs[0]}")

    assert set(results) == {0, 1}, f"两个 rank 都要有结果，实际 {sorted(results)}"

    for rank, r in results.items():
        # p0：两边都有 → (1+3)/2 = 2
        assert r["g0"] == pytest.approx([2.0] * 4), f"rank{rank} p0 平均错了: {r['g0']}"
        # p1：只有 rank0 有 8 → (8+0)/2 = 4。**rank1 原本 grad=None，也必须拿到 4**，
        # 否则 rank1 的权重更新会与 rank0 分叉。
        assert r["g1"] == pytest.approx([4.0] * 3), f"rank{rank} p1 平均错了: {r['g1']}"
        # p2：只有 rank1 有 6 → (0+6)/2 = 3
        assert r["g2"] == pytest.approx([3.0] * 2), f"rank{rank} p2 平均错了: {r['g2']}"

    # 两个 rank 的结果必须逐元素相同 —— 这才是"权重不会分叉"的直接证据
    assert results[0]["g0"] == results[1]["g0"]
    assert results[0]["g1"] == results[1]["g1"]
    assert results[0]["g2"] == results[1]["g2"]

    # all_ranks_clean：全 clean → True；任一 dirty → 两个 rank 都得到 False
    assert results[0]["clean_all"] is True and results[1]["clean_all"] is True
    assert results[0]["clean_one_dirty"] is False, "rank0 报 clean，但 rank1 dirty → 必须一起 False"
    assert results[1]["clean_one_dirty"] is False
