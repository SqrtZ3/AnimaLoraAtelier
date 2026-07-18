"""NaN/Inf 诞生点定位 hook（opt-in, default-off；`debug_nan_hooks: true`）。

动机：训练出现非有限值时，现有守护（anima_train.py 的 loss/grad 守护）只能
告诉你"某个参数的梯度已经是 inf 了"，但 inf 可能在上游许多层之前就诞生、
一路传播过来。逐参数统计看到的是**污染结果**而非**污染源**。

本模块在每个叶子 module 上挂前向/反向 hook，报告第一个满足
「输入全有限 → 输出出现非有限」的模块 —— 那就是诞生点。附带打印该层的
输入/输出统计；若是 QuantLinear，额外打印量化元信息（scale 粒度与范围），
这正是排查 base-quant 数值问题需要的现场。

默认 `abort=True`：抓到第一例即抛 RuntimeError 中止训练，避免污染继续扩散
把现场冲掉（NaN 传播后所有下游层都"有问题"，晚一步就分不清源头）。

开销：每个 module 每次前向多两次 `isfinite().all()` 归约。仅诊断用，
不要在正式训练里开。关闭时本模块完全不被 import。
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def _tensors(obj):
    """从 tensor / tuple / list / dict 里摊出所有 tensor。"""
    if torch.is_tensor(obj):
        yield obj
    elif isinstance(obj, (tuple, list)):
        for o in obj:
            yield from _tensors(o)
    elif isinstance(obj, dict):
        for o in obj.values():
            yield from _tensors(o)


def _all_finite(obj) -> bool:
    for t in _tensors(obj):
        if t.is_floating_point() and not torch.isfinite(t).all():
            return False
    return True


def _stat(obj) -> str:
    """摘要字符串：shape/dtype + 有限部分的幅值范围 + nan/inf 计数。"""
    parts = []
    for t in _tensors(obj):
        if not t.is_floating_point():
            parts.append(f"{tuple(t.shape)}:{t.dtype}(non-float)")
            continue
        fin = torch.isfinite(t)
        nan = int(torch.isnan(t).sum())
        inf = int(torch.isinf(t).sum())
        if fin.any():
            amax = t[fin].abs().max().float().item()
            amin = t[fin].abs().min().float().item()
        else:
            amax = amin = float("nan")
        parts.append(
            f"{tuple(t.shape)}:{str(t.dtype).split('.')[-1]} "
            f"|x|∈[{amin:.3e},{amax:.3e}] nan={nan} inf={inf}"
        )
    return " ; ".join(parts) if parts else "(无 tensor)"


def _quant_meta(mod) -> str:
    """QuantLinear 的量化现场（其它 module 返回空串）。"""
    if type(mod).__name__ != "QuantLinear":
        return ""
    bits = [f"fmt={getattr(mod, 'fmt', '?')}", f"mode={getattr(mod, 'mode', '?')}"]
    if getattr(mod, "fmt", "") == "fp8":
        bits.append("scale=" + ("rowwise" if getattr(mod, "fp8_rowwise", False)
                                else "tensorwise"))
        bits.append(f"fp8_grad={getattr(mod, 'fp8_grad', False)}")
        ws = getattr(mod, "weight_scale", None)
        if torch.is_tensor(ws):
            bits.append(f"w_scale∈[{ws.min().item():.3e},{ws.max().item():.3e}]")
    return "  [" + " ".join(bits) + "]"


class NanHookState:
    """安装状态；`hits` 记录已报告的诞生点，供测试断言。"""

    def __init__(self, abort: bool, max_reports: int):
        self.abort = abort
        self.max_reports = max_reports
        self.hits: list[str] = []
        self.handles: list = []

    def _report(self, kind: str, name: str, mod, inp, out) -> None:
        if len(self.hits) >= self.max_reports:
            return
        self.hits.append(f"{kind}:{name}")
        msg = (
            f"\n{'=' * 72}\n"
            f"[nan-hook] 非有限值诞生点（{kind}）\n"
            f"  module : {name}  ({type(mod).__name__}){_quant_meta(mod)}\n"
            f"  输入   : {_stat(inp)}\n"
            f"  输出   : {_stat(out)}\n"
            f"{'=' * 72}"
        )
        logger.error(msg)
        print(msg, flush=True)
        if self.abort:
            raise RuntimeError(
                f"[nan-hook] {name} 在{kind}中产出非有限值（输入是有限的）——"
                f"这是 NaN/Inf 的诞生点。详见上方现场打印。"
                f"（debug_nan_hooks=true 时抓到即中止；设 debug_nan_hooks_abort=false "
                f"可只记录不中止。）"
            )

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


def install_nan_hooks(model: torch.nn.Module, *, abort: bool = True,
                      max_reports: int = 1) -> NanHookState:
    """给 `model` 的所有叶子 module 挂 NaN 诞生点 hook。返回状态对象。

    只报告「输入有限但输出非有限」的模块 —— 输入本就非有限的说明污染来自
    上游，不是诞生点，跳过以免刷屏。
    """
    state = NanHookState(abort=abort, max_reports=max_reports)

    def fwd_hook(name):
        def _hook(mod, inp, out):
            if _all_finite(out):
                return
            if not _all_finite(inp):
                return  # 上游已污染，本层只是传播
            state._report("前向", name, mod, inp, out)
        return _hook

    def bwd_hook(name):
        def _hook(mod, grad_in, grad_out):
            if _all_finite(grad_in):
                return
            if not _all_finite(grad_out):
                return  # 下游传来的梯度已污染
            state._report("反向", name, mod, grad_out, grad_in)
        return _hook

    n = 0
    for name, mod in model.named_modules():
        if list(mod.children()):
            continue  # 只挂叶子，避免父模块重复报告
        state.handles.append(mod.register_forward_hook(fwd_hook(name)))
        state.handles.append(mod.register_full_backward_hook(bwd_hook(name)))
        n += 1
    msg = (f"[nan-hook] 已在 {n} 个叶子 module 上安装 NaN 诞生点探测"
           f"（abort={abort}）。这是诊断开关，会拖慢每步，勿用于正式训练。")
    logger.warning(msg)
    print(msg, flush=True)
    return state
