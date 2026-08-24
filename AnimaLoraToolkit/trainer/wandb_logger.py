"""Weights & Biases 接线（opt-in，默认关）。

**为什么需要它**：`train_monitor.py` 是进程内 HTTP server（端口 6006，数据只在内存里），
在不给暴露端口的云平台（启智 OpenI 的调试任务、见 docs/ascend-npu.md）上完全看不到。
wandb 是反向的——训练进程主动往外推，只要有出网就能在浏览器里看。

**设计约束**（照仓库既有约定）：

1. **opt-in、默认关**：`wandb_enabled: false` 时本模块的每个函数都是 no-op，
   连 `import wandb` 都不会发生，与接线前逐字节等价。
2. **永不打断训练**：所有对外调用裹 try/except，wandb 挂了就降级成"没有 wandb"，
   打一条 warning 继续训。这与既有 `update_monitor(...)` 调用点的纪律一致。
3. **离线可用**：`wandb_mode: offline` 时只往本地 `wandb_dir` 写，事后
   `wandb sync <run 目录>` 补传。国内云到 api.wandb.ai 的连通性不保证，
   首次上机建议先 offline 跑通再切 online。

**SwanLab**：如果 wandb 连不上，SwanLab（国内同类）提供 `swanlab.sync_wandb()`
劫持 wandb 调用，本模块无需改动即可转发过去——在训练脚本外面先调一次即可。
本仓库不代为 import swanlab（不引入未验证依赖）。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

_RUN = None            # wandb Run 对象；None = 未启用/初始化失败
_WANDB = None          # 惰性 import 的 wandb 模块
_LOG_EVERY = 1
_LOG_IMAGES = True
_FAILED = False        # 出过错就不再重试，避免每步刷屏


def enabled() -> bool:
    """当前这次 run 是否真的在往 wandb 记（init 成功才为 True）。"""
    return _RUN is not None


def init(args, extra_config: Mapping[str, Any] | None = None) -> bool:
    """按 args 上的 wandb_* 字段初始化。未开启或失败都返回 False，调用方无需分支。"""
    global _RUN, _WANDB, _LOG_EVERY, _LOG_IMAGES, _FAILED

    if not bool(getattr(args, "wandb_enabled", False)):
        return False

    mode = str(getattr(args, "wandb_mode", "online") or "online")
    if mode == "disabled":
        return False

    _LOG_EVERY = max(1, int(getattr(args, "wandb_log_every_steps", 1) or 1))
    _LOG_IMAGES = bool(getattr(args, "wandb_log_images", True))

    try:
        import wandb  # 只有开启时才 import，未装 wandb 的环境不受影响
    except Exception as e:
        logger.warning("wandb_enabled=true 但导入 wandb 失败（跳过，不影响训练）: %s", e)
        _FAILED = True
        return False

    output_dir = Path(getattr(args, "output_dir", "./output"))
    wandb_dir = str(getattr(args, "wandb_dir", "") or output_dir)
    try:
        Path(wandb_dir).mkdir(parents=True, exist_ok=True)
    except Exception:
        wandb_dir = None

    # run 名默认跟 output_name 走：本仓库的产物就是按 output_name 命名的，
    # 这样 wandb 上的 run 与磁盘上的权重文件能直接对上号。
    run_name = str(getattr(args, "wandb_run_name", "") or getattr(args, "output_name", "") or "")
    entity = str(getattr(args, "wandb_entity", "") or "") or None

    config = dict(extra_config or {})
    try:
        _RUN = wandb.init(
            project=str(getattr(args, "wandb_project", "anima-lora") or "anima-lora"),
            entity=entity,
            name=run_name or None,
            mode=mode,
            dir=wandb_dir,
            config=config,
            resume="allow",
        )
        _WANDB = wandb
        logger.info(
            "[wandb] 已连接：project=%s run=%s mode=%s dir=%s",
            getattr(args, "wandb_project", "anima-lora"), run_name or "(auto)", mode, wandb_dir,
        )
        return True
    except Exception as e:
        logger.warning("[wandb] init 失败（训练继续，不记录）: %s", e)
        _RUN = None
        _FAILED = True
        return False


def log(metrics: Mapping[str, Any], step: int | None = None) -> None:
    """记一组标量。step 传训练的 global_step，wandb 侧就以它为横轴。"""
    global _FAILED
    if _RUN is None or _FAILED:
        return
    try:
        _RUN.log(dict(metrics), step=None if step is None else int(step))
    except Exception as e:
        logger.warning("[wandb] log 失败，后续不再尝试: %s", e)
        _FAILED = True


def log_step(
    step: int,
    loss: float,
    lr: float,
    speed: float | None = None,
    epoch: int | None = None,
    samples_seen: int | None = None,
    ref_step: float | None = None,
) -> None:
    """逐步指标。按 wandb_log_every_steps 抽稀（默认每步；长跑可设 10/50 降低请求量）。"""
    if _RUN is None or _FAILED:
        return
    if _LOG_EVERY > 1 and int(step) % _LOG_EVERY != 0:
        return
    m: dict[str, Any] = {"train/loss": float(loss), "train/lr": float(lr)}
    if speed is not None:
        m["train/it_per_s"] = float(speed)
    if epoch is not None:
        m["train/epoch"] = int(epoch)
    if samples_seen is not None:
        m["train/samples_seen"] = int(samples_seen)
    if ref_step is not None:
        m["train/ref_step"] = float(ref_step)
    log(m, step=step)


def log_eval(step: int, mean_loss: float, per_t: list[float], t_grid: list[float]) -> None:
    """eval 的逐 t-bin loss（与 output_dir/eval_loss.csv 同源，wandb 上多一条曲线而已）。"""
    if _RUN is None or _FAILED:
        return
    m: dict[str, Any] = {"eval/mean": float(mean_loss)}
    for tv, v in zip(t_grid, per_t):
        m[f"eval/t{tv:g}"] = float(v)
    log(m, step=step)


def log_image(path, step: int, caption: str | None = None) -> None:
    """采样预览图。wandb_log_images=false 可只记标量不传图（省带宽）。"""
    global _FAILED
    if _RUN is None or _FAILED or not _LOG_IMAGES:
        return
    try:
        p = Path(path)
        if not p.exists():
            return
        _RUN.log({"sample": _WANDB.Image(str(p), caption=caption or p.name)}, step=int(step))
    except Exception as e:
        logger.warning("[wandb] 图片上传失败，后续不再尝试: %s", e)
        _FAILED = True


def finish() -> None:
    """收尾。offline 模式下这一步才会把 run 目录写完整，别跳过。"""
    global _RUN
    if _RUN is None:
        return
    try:
        _RUN.finish()
    except Exception as e:
        logger.warning("[wandb] finish 失败（忽略）: %s", e)
    finally:
        _RUN = None
