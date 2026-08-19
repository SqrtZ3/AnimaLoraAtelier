"""训练步分阶段计时（图盲 opt-in default-off）。

服务 [[feedback-training-observability]]：把"一步训练的时间花在哪"变成可读数据，
定位 NaViT vs ARB 的速度差异根因——flash varlen 已实测与 xformers 等速（attention
不是瓶颈），本探针回答时间到底花在 text encode / 逐图噪声+patchify / 模型 forward /
逐图 loss / aux / backward 哪一段。

设计要点
--------
* **CUDA event 计时 GPU 阶段**：``record()`` 只往命令缓冲区插一条指令，不 sync、几乎
  零开销、不扰动 GPU 时序；仅在被采样步末尾做一次 ``torch.cuda.synchronize()`` 读
  ``elapsed_time``。逐图 Python 循环里的 kernel launch 开销会被计入该阶段（正是要抓的）。
* **perf_counter 计时 CPU/IO 阶段**（``data_fetch``）：CUDA event 测纯 CPU 区间会得 ~0，
  故 CPU/IO 用 ``start_cpu``/``stop_cpu`` 走 wall-clock。
* **行为中立**：默认 ``stage_timing_every=0`` 时 ``make_stage_timer(False)`` 返回
  ``_NOOP_TIMER``，start/stop 全是空操作，不记录 event、不 sync、不分配。开启时仅被采样
  步用真实 ``StageTimer``，非采样步用 noop——稳态 it/s 在非采样步测量，不受 sync 影响。

依赖：仅 torch + 标准库（云端 venv 无 scipy/numpy 保证，见 [[environment-reference]]）。
"""

from __future__ import annotations

import time
from typing import Union

import torch


class _NoopStageTimer:
    """空操作计时器：与 :class:`StageTimer` 同接口，但什么都不做。default-off 时使用。

    所有方法均为空体，``flush`` 返回空 dict。残余成本仅几次空方法调用（亚微秒），
    与 telemetry 每步 ``run_step_telemetry`` 早退同级。
    """

    __slots__ = ()

    def start(self, stage: str) -> None:  # noqa: D401 - deliberate no-op
        ...

    def stop(self, stage: str) -> None:
        ...

    def start_cpu(self, stage: str) -> None:
        ...

    def stop_cpu(self, stage: str) -> None:
        ...

    def flush(self) -> dict:
        return {}

    def reset(self) -> None:
        ...


_NOOP_TIMER = _NoopStageTimer()


class StageTimer:
    """被采样步的 CUDA-event 分阶段计时器。

    用法（在被采样步内）::

        timer.start("text_encode")        # GPU 阶段
        ... GPU 工作 ...
        timer.stop("text_encode")
        timer.start_cpu("data_fetch")     # CPU/IO 阶段
        ... CPU/IO 工作 ...
        timer.stop_cpu("data_fetch")
        summary = timer.flush()           # 触发一次 sync，返回 {stage: ms}

    每个 stage 每步只 start/stop 一次。GPU 阶段用 CUDA event（``start``/``stop``），
    CPU/IO 阶段用 perf_counter（``start_cpu``/``stop_cpu``）。嵌套子探针（如 navit
    forward 内的 noise_patchify / model_forward / loss_loop）也用 event，其和应 ≤ 外层
    forward（forward event 跨整段，含 cat 等未单独探针的零碎）。
    """

    def __init__(self) -> None:
        # GPU 阶段：每阶段一对 CUDA event（start.record() / stop.record() 各插一条标记）。
        self._starts: dict[str, torch.cuda.Event] = {}
        self._ends: dict[str, torch.cuda.Event] = {}
        # CPU/IO 阶段：累计 perf_counter delta（秒），flush 时转 ms。
        self._cpu_starts: dict[str, float] = {}
        self._cpu_deltas: dict[str, float] = {}

    def start(self, stage: str) -> None:
        evt = torch.cuda.Event(enable_timing=True)
        evt.record()
        self._starts[stage] = evt

    def stop(self, stage: str) -> None:
        evt = torch.cuda.Event(enable_timing=True)
        evt.record()
        self._ends[stage] = evt

    def start_cpu(self, stage: str) -> None:
        self._cpu_starts[stage] = time.perf_counter()

    def stop_cpu(self, stage: str) -> None:
        t0 = self._cpu_starts.pop(stage, None)
        if t0 is not None:
            self._cpu_deltas[stage] = self._cpu_deltas.get(stage, 0.0) + (
                time.perf_counter() - t0
            )

    def flush(self) -> dict:
        """synchronize 一次，返回 ``{stage: ms}``。

        GPU 阶段用 ``start.elapsed_time(end)``（已是 ms）；CPU/IO 阶段用累计 perf_counter
        delta（秒→ms）。调用后内部状态清空（等价于 :meth:`reset`），计时器可复用于下一采样步。
        """
        torch.cuda.synchronize()
        out: dict[str, float] = {}
        for stage, se in self._starts.items():
            ee = self._ends.get(stage)
            if ee is not None:
                out[stage] = float(se.elapsed_time(ee))
        for stage, delta in self._cpu_deltas.items():
            out[stage] = delta * 1000.0
        self.reset()
        return out

    def reset(self) -> None:
        self._starts.clear()
        self._ends.clear()
        self._cpu_starts.clear()
        self._cpu_deltas.clear()


def make_stage_timer(enabled: bool) -> Union[StageTimer, _NoopStageTimer]:
    """``enabled=True`` 返回真实 :class:`StageTimer`；``False`` 返回 ``_NOOP_TIMER``。

    训练步顶按采样 cadence 选用 real/noop：被采样步用 real（记录 event + 末尾一次 sync），
    非采样步用 noop（零成本），保证稳态 it/s 不受计时扰动。
    """
    return StageTimer() if enabled else _NOOP_TIMER
