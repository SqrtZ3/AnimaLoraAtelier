"""冻结底模 Linear 的 FP8 / FP4 量化（opt-in，default-off）。

动机（参见 docs/base-quant.md）：LoRA 训练里底模权重冻结、只通过
``LoRALinear.original(x)`` 参与前向 —— 把它换成量化存储/量化 GEMM 的
``QuantLinear``，可以：
  1. 省显存：12.9B bf16 权重 25.8GB → fp8 ≈ 12.9GB / fp4 ≈ 6.7GB，
     释放出来的显存能开更大的 navit token budget；
  2. 提速（可选）：H20 的 FP8 tensor core 峰值是 BF16 的 2×，
     前向（和 opt-in 的反向 dL/dx）GEMM 走 ``torch._scaled_mm``。

量化格式：
  - fp8：float8_e4m3fn 权重 + rowwise（[N,1]，H20/sm90 支持）或
    tensorwise（标量，兜底）fp32 scale。GEMM 时激活动态量化成 e4m3。
  - fp4：nvfp4 风格 —— e2m1 4bit 码（两值打包一个 uint8）+ 每 16 元素
    block 的 e4m3 scale + 全张量 fp32 scale。GEMM（需 Blackwell sm120+
    与 torch>=2.8 的 fp4 _scaled_mm）时激活同样动态量化；硬件不支持时
    自动退回"dequant 到 bf16 计算"（H20 上 fp4 仍可作纯省显存用）。

反向：底模权重冻结 → 只需要 dL/dx = dL/dy @ W。默认用 bf16 GEMM
（backward 时**重新** dequant，权重不以 bf16 驻留显存）；
``base_quant_fp8_grad=true`` 时 opt-in 走 e5m2×e4m3 的 fp8 GEMM。

数值口径的本地实测（RTX 5070 sm_120, torch 2.9.1+cu130，随机高斯数据）：
  - fp8 tensorwise W8A8 GEMM 相对误差 ≈ 0.037；
  - fp4 W4A4（swizzled block scale）≈ 0.134，与逐元素仿真差 0.0016
    （scale 布局正确的物证）；nvfp4 权重 dequant 误差 ≈ 0.095，
    与推理端 W4A4 交接文档的实测一致。

``_to_blocked``（fp4 block-scale 的 cuBLAS swizzle 布局）改写自
pytorch/ao (torchao) ``torchao/prototype/mx_formats/utils.py``，BSD-3
许可，见 THIRD_PARTY_NOTICES.md。
"""

from __future__ import annotations

import logging
import math
import re

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# fp8 e4m3fn 的最大有限值；量化前 clamp 到 ±该值防溢出成 NaN/inf
_E4M3_MAX = 448.0
# fp8 e5m2 的最大有限值（梯度量化用，动态范围优先）
_E5M2_MAX = 57344.0
# e2m1（fp4）可表示的非负值网格与相邻中点（round-to-nearest 用）
_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_MAX = 6.0
_FP4_BLOCK = 16  # nvfp4 block 大小（沿 in_features/K 轴）


# ──────────────────────────────────────────────────────────────────────
# 运行时能力探测（按当前 CUDA 设备缓存；用真实小 GEMM try/except，
# 不猜 compute capability —— torch 版本/平台差异太多，实跑才算数）
# ──────────────────────────────────────────────────────────────────────

_CAPS_CACHE: dict = {}
_CAPS_ERR: dict = {}  # kind -> 最近一次探测失败的原因字符串（诊断用）


def _gemm_probe(kind: str) -> bool:
    """kind: 'fp8_tensorwise' | 'fp8_rowwise' | 'fp4'。"""
    if not torch.cuda.is_available():
        return False
    key = (kind, torch.cuda.current_device())
    if key in _CAPS_CACHE:
        return _CAPS_CACHE[key]
    ok = False
    try:
        dev = torch.device("cuda", torch.cuda.current_device())
        M, K, N = 16, 32, 16
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        w = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
        if kind == "fp8_tensorwise":
            xq, xs = _quant_fp8_tensorwise(x)
            wq, ws = _quant_fp8_tensorwise(w)
            torch._scaled_mm(xq, wq.t(), scale_a=xs, scale_b=ws,
                             out_dtype=torch.bfloat16)
            ok = True
        elif kind == "fp8_rowwise":
            xq, xs = _quant_fp8_rowwise(x)
            wq, ws = _quant_fp8_rowwise(w)
            torch._scaled_mm(xq, wq.t(), scale_a=xs, scale_b=ws.t(),
                             out_dtype=torch.bfloat16)
            ok = True
        elif kind in ("fp8_grad_rowwise", "fp8_grad_tensorwise"):
            # 反向 dL/dx 的实际组合：e5m2 梯度 × e4m3 权重（部分平台的
            # rowwise 内核只收双 e4m3，必须按真实 dtype 组合探测）。
            # 形状按真实 backward：gy [M,N] @ W_cm [N,K] → [M,K]
            g = torch.randn(M, N, device=dev, dtype=torch.bfloat16)
            wq, ws = _quant_fp8_rowwise(w)
            wq_cm = wq.t().contiguous().t()
            if kind == "fp8_grad_rowwise":
                gq, gs = _quant_fp8_rowwise(g, fmax=_E5M2_MAX,
                                            dtype=torch.float8_e5m2)
                ones = ws.new_ones(1, wq.shape[1])
                torch._scaled_mm(gq, wq_cm, scale_a=gs, scale_b=ones,
                                 out_dtype=torch.bfloat16)
            else:
                gq, gs = _quant_fp8_tensorwise(g, fmax=_E5M2_MAX,
                                               dtype=torch.float8_e5m2)
                _, ws_t = _quant_fp8_tensorwise(w)
                torch._scaled_mm(gq, wq_cm, scale_a=gs, scale_b=ws_t,
                                 out_dtype=torch.bfloat16)
            ok = True
        elif kind == "fp4":
            if not hasattr(torch, "float4_e2m1fn_x2"):
                ok = False
            else:
                xp, xbs, xts = _quant_nvfp4(x)
                wp, wbs, wts = _quant_nvfp4(w)
                torch._scaled_mm(
                    xp.view(torch.float4_e2m1fn_x2),
                    wp.view(torch.float4_e2m1fn_x2).t(),
                    scale_a=_to_blocked(xbs), scale_b=_to_blocked(wbs),
                    out_dtype=torch.bfloat16)
                ok = True
        else:
            raise ValueError(f"未知 probe kind: {kind}")
    except Exception as e:  # noqa: BLE001 —— 探测失败即"该路径不可用"
        _CAPS_ERR[kind] = f"{type(e).__name__}: {e}"
        logger.info("[base-quant] GEMM 能力探测 %s 不可用: %s", kind, e)
        ok = False
    _CAPS_CACHE[key] = ok
    return ok


# ──────────────────────────────────────────────────────────────────────
# FP8 量化 / 反量化
# ──────────────────────────────────────────────────────────────────────

def _quant_fp8_tensorwise(t: torch.Tensor, fmax: float = _E4M3_MAX,
                          dtype=torch.float8_e4m3fn):
    """[*] → (q fp8, scale 0-dim fp32)。q * scale ≈ t。"""
    scale = t.abs().amax().float().div(fmax).clamp(min=1e-12)
    q = (t.float() / scale).clamp(-fmax, fmax).to(dtype)
    return q, scale


def _quant_fp8_rowwise(t: torch.Tensor, fmax: float = _E4M3_MAX,
                       dtype=torch.float8_e4m3fn):
    """[R,C] → (q fp8 [R,C], scale fp32 [R,1])。逐行 amax scale。"""
    scale = t.abs().amax(dim=1, keepdim=True).float().div(fmax).clamp(min=1e-12)
    q = (t.float() / scale).clamp(-fmax, fmax).to(dtype)
    return q, scale


def _dequant_fp8(q: torch.Tensor, scale: torch.Tensor, out_dtype) -> torch.Tensor:
    return (q.float() * scale).to(out_dtype)


# ── 激活量化的可选融合（base_quant_fuse_act_quant，opt-in default-off）────────
# 热路径的 abs→amax→div→clamp→cast 是 4-5 个分立 kernel（H20 profile 实测
# div/clamp/mul 家族 ≈ 每步数秒）。torch.compile 只编译这两个纯函数（不碰模型
# 整图，与 navit / torch_compile 互斥无关），把链融成 1-2 个 kernel。
# 数值口径：运算顺序不变，inductor 融合可能在 ulp 级改变中间舍入（fp8 码字
# 边界值可能差 1 码），启用探针里用 dequant 值 allclose 校验。
# 关闭（默认）时下面两个指针就是 eager 原函数，行为逐 bit 不变。
_act_quant_tensorwise = _quant_fp8_tensorwise
_act_quant_rowwise = _quant_fp8_rowwise


def enable_fused_act_quant() -> bool:
    """尝试把激活量化原语切到 torch.compile 融合版；成功返回 True。

    失败（无 triton / 编译异常 / 数值探针不过）自动留在 eager，非静默——
    调用方负责把结果打到 stdout。dynamic=True 避免逐形状重编译（M 每步变）。
    """
    global _act_quant_rowwise, _act_quant_tensorwise
    try:
        crow = torch.compile(_quant_fp8_rowwise, dynamic=True)
        cten = torch.compile(_quant_fp8_tensorwise, dynamic=True)
        x = torch.randn(96, 128, device="cuda", dtype=torch.bfloat16)
        for eager_fn, fused_fn in ((_quant_fp8_rowwise, crow),
                                   (_quant_fp8_tensorwise, cten)):
            for kw in ({}, {"fmax": _E5M2_MAX, "dtype": torch.float8_e5m2}):
                q0, s0 = eager_fn(x, **kw)
                q1, s1 = fused_fn(x, **kw)
                assert q1.shape == q0.shape and q1.dtype == q0.dtype
                d0 = q0.float() * s0
                d1 = q1.float() * s1
                assert torch.allclose(d0, d1, atol=1e-2, rtol=1e-2), \
                    f"融合量化 dequant 偏差 {(d0 - d1).abs().max().item():.3e}"
        _act_quant_rowwise, _act_quant_tensorwise = crow, cten
        return True
    except Exception as ex:  # noqa: BLE001 —— 速度旋钮，宁可回退不可崩
        logger.warning("[base-quant] fuse_act_quant 启用失败，保持 eager：%s", ex)
        return False


# ──────────────────────────────────────────────────────────────────────
# FP4（nvfp4 风格）量化 / 反量化 / swizzle
# ──────────────────────────────────────────────────────────────────────

def _e2m1_codes(scaled: torch.Tensor) -> torch.Tensor:
    """scaled: 已归一到 [-6,6] 的 float → uint8 码（bit3=符号, bit0..2=幅值下标）。

    round-to-nearest：bucketize 到相邻网格值的中点。tie 时取较小幅值
    （与逐点 argmin 的首个最小值语义一致，见单测对拍）。
    """
    boundaries = scaled.new_tensor(
        [(a + b) / 2 for a, b in zip(_E2M1_VALUES[:-1], _E2M1_VALUES[1:])]
    )
    sign = (scaled < 0).to(torch.uint8)
    mag = scaled.abs().clamp(max=_E2M1_MAX)
    idx = torch.bucketize(mag, boundaries).to(torch.uint8)  # 0..7
    return (sign << 3) | idx


def _pack_fp4(codes: torch.Tensor) -> torch.Tensor:
    """uint8 码 [..., C] → 打包 uint8 [..., C//2]。低 nibble 存偶数下标
    （布局已用 fp4 _scaled_mm 与逐元素仿真对拍验证）。"""
    lo = codes[..., 0::2]
    hi = codes[..., 1::2]
    return lo | (hi << 4)


def _quant_nvfp4(t: torch.Tensor):
    """[R,C]（C%16==0）→ (packed uint8 [R,C//2],
                          block scale e4m3 [R,C//16],
                          tensor scale fp32 0-dim)。

    dequant ≈ code_value * block_scale * tensor_scale。
    tensor scale 把 block amax 归一到 e4m3 可表示范围（nvfp4 标准做法），
    避免小幅值 block 的 scale 在 e4m3 下 flush 到 0。
    """
    R, C = t.shape
    if C % _FP4_BLOCK:
        raise ValueError(f"fp4 量化要求 C%{_FP4_BLOCK}==0，got {C}")
    tf = t.float().view(R, C // _FP4_BLOCK, _FP4_BLOCK)
    amax = tf.abs().amax(dim=-1, keepdim=True)  # [R,C/16,1]
    ts = (amax.max() / (_E2M1_MAX * _E4M3_MAX)).clamp(min=1e-12)
    bs = (amax / (_E2M1_MAX * ts)).clamp(min=1e-12, max=_E4M3_MAX) \
        .to(torch.float8_e4m3fn)
    # e4m3 subnormal 下限以下会 flush 成 0 → dequant 除 0 得 NaN；
    # 这些 block 本身 amax≈0，scale 换成 1 后全 block 量化为 0，安全。
    bs_f = bs.float()
    bs_f = torch.where(bs_f == 0, torch.ones_like(bs_f), bs_f)
    scaled = tf / (bs_f * ts)
    codes = _e2m1_codes(scaled.reshape(R, C))
    return _pack_fp4(codes), bs.view(R, C // _FP4_BLOCK), ts


_E2M1_LUT_CACHE: dict = {}


def _dequant_nvfp4(packed: torch.Tensor, bs: torch.Tensor, ts: torch.Tensor,
                   out_dtype) -> torch.Tensor:
    """打包 uint8 [R,C//2] → [R,C] out_dtype。"""
    key = packed.device
    lut = _E2M1_LUT_CACHE.get(key)
    if lut is None:
        vals = list(_E2M1_VALUES)
        lut = torch.tensor(vals + [-v for v in vals],
                           device=packed.device, dtype=torch.float32)
        _E2M1_LUT_CACHE[key] = lut
    R = packed.shape[0]
    lo = packed & 0xF
    hi = packed >> 4
    codes = torch.stack([lo, hi], dim=-1).view(R, -1)  # [R,C]
    vals = lut[codes.long()]
    C = vals.shape[1]
    bs_f = bs.float()
    bs_f = torch.where(bs_f == 0, torch.ones_like(bs_f), bs_f)
    vals = vals.view(R, C // _FP4_BLOCK, _FP4_BLOCK) * bs_f.view(R, -1, 1) * ts
    return vals.view(R, C).to(out_dtype)


def _to_blocked(scales: torch.Tensor) -> torch.Tensor:
    """fp4 block scale [R, C/16] e4m3 → cuBLAS 要求的 128×4 tile swizzle 布局。

    改写自 torchao ``to_blocked``（BSD-3-Clause，pytorch/ao）；
    正确性物证：swizzle 后 fp4 _scaled_mm 与逐元素 dequant 仿真差 0.0016。
    """
    rows, cols = scales.shape
    n_row_blocks = math.ceil(rows / 128)
    n_col_blocks = math.ceil(cols / 4)
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4
    padded = scales
    if (rows, cols) != (padded_rows, padded_cols):
        padded = scales.new_zeros(padded_rows, padded_cols)
        padded[:rows, :cols] = scales
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.flatten()


# ──────────────────────────────────────────────────────────────────────
# 自定义 autograd：权重冻结 → 只回传 dL/dx；backward 里重新 dequant，
# 不把 bf16 全量权重挂在 ctx 上驻留（否则省显存全白搭）。
# ──────────────────────────────────────────────────────────────────────

class _DequantLinearFn(torch.autograd.Function):
    """量化存储 + bf16 GEMM（fp8/fp4 通吃的"纯省显存"路径）。"""

    @staticmethod
    def forward(ctx, x2d, qlin):
        ctx.qlin = qlin
        w = qlin._dequant_weight(x2d.dtype)
        return F.linear(x2d, w, qlin.bias)

    @staticmethod
    def backward(ctx, gy):
        gx = None
        if ctx.needs_input_grad[0]:
            w = ctx.qlin._dequant_weight(gy.dtype)
            gx = gy @ w
        return gx, None


class _Fp8GemmLinearFn(torch.autograd.Function):
    """W8A8 fp8 前向（torch._scaled_mm）；反向默认 bf16，opt-in fp8。"""

    @staticmethod
    def forward(ctx, x2d, qlin):
        ctx.qlin = qlin
        wq, ws = qlin.weight_q, qlin.weight_scale
        if qlin.fp8_rowwise:
            xq, xs = _act_quant_rowwise(x2d)
            scale_b = ws.t()
        else:
            xq, xs = _act_quant_tensorwise(x2d)
            scale_b = ws
        return torch._scaled_mm(xq, wq.t(), scale_a=xs, scale_b=scale_b,
                                bias=qlin.bias, out_dtype=x2d.dtype)

    @staticmethod
    def backward(ctx, gy):
        if not ctx.needs_input_grad[0]:
            return None, None
        qlin = ctx.qlin
        wq, ws = qlin.weight_q, qlin.weight_scale
        if not qlin.fp8_grad:
            w = qlin._dequant_weight(gy.dtype)
            return gy @ w, None
        gy = gy.contiguous()
        # dL/dx = gy @ W = gy @ (wq·ws)。b 操作数要求 col-major [N,K]：
        # 每步做一次 fp8 转置拷贝（transient，~N·K bytes，远小于 GEMM 本身）。
        wq_cm = wq.t().contiguous().t()
        if qlin.fp8_rowwise:
            # ws [N,1] 折进 gy 的列 → g̃ = gy * wsᵀ，再逐行 e5m2 量化
            g = gy * ws.t().to(gy.dtype)
            gq, gs = _act_quant_rowwise(g, fmax=_E5M2_MAX, dtype=torch.float8_e5m2)
            ones = ws.new_ones(1, wq.shape[1])
            gx = torch._scaled_mm(gq, wq_cm, scale_a=gs, scale_b=ones,
                                  out_dtype=gy.dtype)
        else:
            gq, gs = _act_quant_tensorwise(gy, fmax=_E5M2_MAX,
                                           dtype=torch.float8_e5m2)
            gx = torch._scaled_mm(gq, wq_cm, scale_a=gs, scale_b=ws,
                                  out_dtype=gy.dtype)
        return gx, None


class _Fp4GemmLinearFn(torch.autograd.Function):
    """W4A4 nvfp4 前向（fp4 _scaled_mm + swizzled block scale）；反向 bf16。

    梯度不做 fp4：4bit 量化对梯度的信噪比破坏太大（无 Hadamard 旋转类
    保护时业界也不这么做），v1 反向固定 bf16 dequant GEMM。
    """

    @staticmethod
    def forward(ctx, x2d, qlin):
        ctx.qlin = qlin
        xp, xbs, xts = _quant_nvfp4(x2d)
        y = torch._scaled_mm(
            xp.view(torch.float4_e2m1fn_x2),
            qlin.weight_q.view(torch.float4_e2m1fn_x2).t(),
            scale_a=_to_blocked(xbs), scale_b=qlin.weight_scale_blocked,
            out_dtype=x2d.dtype)
        y = y * (xts * qlin.weight_scale2).to(y.dtype)
        if qlin.bias is not None:
            y = y + qlin.bias
        return y

    @staticmethod
    def backward(ctx, gy):
        gx = None
        if ctx.needs_input_grad[0]:
            w = ctx.qlin._dequant_weight(gy.dtype)
            gx = gy @ w
        return gx, None


# ──────────────────────────────────────────────────────────────────────
# QuantLinear
# ──────────────────────────────────────────────────────────────────────

class QuantLinear(torch.nn.Module):
    """量化存储的冻结 Linear，对外表现同 nn.Linear（forward / weight /
    bias / in_features / out_features）。

    ``weight`` property 返回**现算的 dequant bf16 张量**（带量化误差）——
    供 merged_weight() / 导出 / 取证等低频路径透明复用；热路径不要碰它。
    """

    def __init__(self, fmt: str, mode: str, in_features: int, out_features: int,
                 bias: torch.nn.Parameter | None, compute_dtype=torch.bfloat16):
        super().__init__()
        assert fmt in ("fp8", "fp4")
        # mode: fp8_gemm / fp4_gemm / dequant
        assert mode in ("fp8_gemm", "fp4_gemm", "dequant")
        self.fmt = fmt
        self.mode = mode
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype
        self.fp8_rowwise = False
        self.fp8_grad = False
        if bias is not None:
            bias.requires_grad_(False)
        self.bias = bias

    # —— 构造 ——————————————————————————————————————————————

    @classmethod
    def from_linear(cls, linear: torch.nn.Linear, fmt: str, mode: str,
                    fp8_rowwise: bool = False, fp8_grad: bool = False,
                    compute_dtype=torch.bfloat16) -> "QuantLinear":
        w = linear.weight.detach()
        self = cls(fmt, mode, linear.in_features, linear.out_features,
                   linear.bias, compute_dtype=compute_dtype)
        if fmt == "fp8":
            self.fp8_rowwise = bool(fp8_rowwise)
            self.fp8_grad = bool(fp8_grad)
            if self.fp8_rowwise or mode == "dequant":
                # dequant-only 模式固定 rowwise 存储（精度更好，GEMM 不受限）
                self.fp8_rowwise = True
                q, s = _quant_fp8_rowwise(w)
            else:
                q, s = _quant_fp8_tensorwise(w)
            self.register_buffer("weight_q", q, persistent=False)
            self.register_buffer("weight_scale", s, persistent=False)
        else:
            packed, bs, ts = _quant_nvfp4(w)
            self.register_buffer("weight_q", packed, persistent=False)
            self.register_buffer("weight_scale", bs, persistent=False)
            self.register_buffer("weight_scale2", ts, persistent=False)
            if mode == "fp4_gemm":
                self.register_buffer("weight_scale_blocked", _to_blocked(bs),
                                     persistent=False)
        return self

    # —— nn.Linear 接口透传 ——————————————————————————————————

    def _dequant_weight(self, dtype) -> torch.Tensor:
        if self.fmt == "fp8":
            return _dequant_fp8(self.weight_q, self.weight_scale, dtype)
        return _dequant_nvfp4(self.weight_q, self.weight_scale,
                              self.weight_scale2, dtype)

    @property
    def weight(self) -> torch.Tensor:
        return self._dequant_weight(self.compute_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2d = x.reshape(-1, self.in_features)
        if x2d.dtype != self.compute_dtype:
            x2d = x2d.to(self.compute_dtype)
        if not x2d.is_contiguous():
            x2d = x2d.contiguous()
        if self.mode == "fp8_gemm":
            y = _Fp8GemmLinearFn.apply(x2d, self)
        elif self.mode == "fp4_gemm":
            y = _Fp4GemmLinearFn.apply(x2d, self)
        else:
            y = _DequantLinearFn.apply(x2d, self)
        return y.reshape(*shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, fmt={self.fmt}, mode={self.mode}, "
                f"fp8_rowwise={self.fp8_rowwise}, fp8_grad={self.fp8_grad}")


# ──────────────────────────────────────────────────────────────────────
# 换装 pass + 配置校验
# ──────────────────────────────────────────────────────────────────────

# family 默认 include（量化哪些逻辑层；名字 = LoRALinear 注入点同款 dotted name）。
# krea2 对齐推理端已验证画质的 W4A4 集合：28 主 block×8 + txtfusion + txtmlp，
# 高精度保留 first / last.* / tproj / tmlp（时间步调制通路最敏感）与所有非 Linear。
_FAMILY_DEFAULT_INCLUDE = {
    "krea2": [r"blocks\..*", r"txtfusion\..*", r"txtmlp\..*"],
    "anima": [r"blocks\..*"],
}


def base_quant_requested(args) -> bool:
    return str(getattr(args, "base_quant", "none") or "none").lower() != "none"


def validate_base_quant_compat(args) -> None:
    """fail-fast 非法组合（v1 收窄变量面；解法写进报错）。"""
    if not base_quant_requested(args):
        return
    fmt = str(getattr(args, "base_quant", "none")).lower()
    if fmt not in ("fp8", "fp4"):
        raise ValueError(f"base_quant={fmt!r} 未知（可选: none / fp8 / fp4）")
    if str(getattr(args, "mixed_precision", "bf16")).lower() != "bf16":
        raise ValueError(
            "base_quant 需要 mixed_precision: bf16（量化 GEMM/ dequant 的计算 dtype "
            "固定 bf16；fp32 全精度训练与底模量化目标冲突）")
    if str(getattr(args, "lora_variant", "base") or "base").lower() == "dora":
        raise ValueError(
            "base_quant 暂不支持 lora_variant=dora：DoRA 每步前向都要读底模全量权重"
            "算合并范数，量化后该路径每步 dequant 物化，省显存收益归零且未验证。"
            "请先用 base（或关掉 base_quant）。")
    if int(getattr(args, "lora_one_init_steps", 0) or 0) > 0:
        raise ValueError(
            "base_quant 与 lora_one_init_steps 互斥：LoRA-One 初始化要在底模权重上"
            "临时挂梯度累积，量化存储（buffer 非 Parameter）做不到。"
            "先跑 LoRA-One 得到初始化，再在 resume 时开 base_quant，或二选一。")
    if bool(getattr(args, "torch_compile", False)):
        raise ValueError(
            "base_quant v1 未与 torch_compile 组合验证（自定义 autograd.Function + "
            "_scaled_mm 的图捕获行为未确认）。请先关掉其中一个。")
    gemm = str(getattr(args, "base_quant_gemm", "auto") or "auto").lower()
    if gemm not in ("auto", "on", "off"):
        raise ValueError(f"base_quant_gemm={gemm!r} 未知（可选: auto / on / off）")
    scale = str(getattr(args, "base_quant_fp8_scale", "auto") or "auto").lower()
    if scale not in ("auto", "rowwise", "tensorwise"):
        raise ValueError(
            f"base_quant_fp8_scale={scale!r} 未知（可选: auto / rowwise / tensorwise）")
    if bool(getattr(args, "base_quant_fp8_grad", False)) and gemm == "off":
        raise ValueError(
            "base_quant_fp8_grad=true 需要 base_quant_gemm 为 auto/on"
            "（反向 fp8 GEMM 建立在前向量化 GEMM 路径之上）")


def _resolve_patterns(raw, default):
    if raw is None:
        return list(default)
    if isinstance(raw, str):
        raw = [s.strip() for s in raw.split(",") if s.strip()]
    return [str(p) for p in raw]


def quantize_base_model(model, args, family: str = "anima"):
    """LoRA 注入完成后调用：把命中 include（且不命中 skip）的冻结底模
    Linear 换成 QuantLinear。返回 stats dict（供日志/测试断言）。

    换装对象：
      - LoRALinear.original（LoRA 包着的底模层）
      - 未被 LoRA 包住的裸 nn.Linear（例如只训 attention 时的 mlp 层）
    永不触碰 adapter 内部的 lora_up/lora_down 等可训练 Linear。
    """
    from trainer.lora import LoRALinear  # 局部 import 避免环形依赖

    fmt = str(getattr(args, "base_quant", "none")).lower()
    gemm_pref = str(getattr(args, "base_quant_gemm", "auto") or "auto").lower()
    scale_pref = str(getattr(args, "base_quant_fp8_scale", "auto") or "auto").lower()
    fp8_grad = bool(getattr(args, "base_quant_fp8_grad", False))
    include = _resolve_patterns(getattr(args, "base_quant_include", None),
                                _FAMILY_DEFAULT_INCLUDE.get(family, [r"blocks\..*"]))
    skip = _resolve_patterns(getattr(args, "base_quant_skip", None), [])

    # ── GEMM 路径决策（探测真实设备）─────────────────────────────
    fp8_rowwise = False
    if fmt == "fp8":
        if gemm_pref == "off":
            gemm_ok = False
        else:
            rowwise_ok = _gemm_probe("fp8_rowwise")
            tensorwise_ok = _gemm_probe("fp8_tensorwise")
            if scale_pref == "rowwise":
                gemm_ok, fp8_rowwise = rowwise_ok, True
                if not rowwise_ok and gemm_pref == "on":
                    raise RuntimeError(
                        "base_quant_fp8_scale=rowwise 但当前设备/torch 不支持 rowwise "
                        "_scaled_mm（H20/sm90 需要 torch>=2.5）。改 tensorwise 或 auto。")
            elif scale_pref == "tensorwise":
                gemm_ok, fp8_rowwise = tensorwise_ok, False
            else:  # auto：优先 rowwise（精度好），退 tensorwise
                if rowwise_ok:
                    gemm_ok, fp8_rowwise = True, True
                else:
                    gemm_ok, fp8_rowwise = tensorwise_ok, False
            if gemm_pref == "on" and not gemm_ok:
                raise RuntimeError(
                    "base_quant_gemm=on 但当前设备/torch 探测不到可用的 fp8 "
                    "_scaled_mm。用 auto（自动退 dequant bf16 计算）或升级环境。")
            # fp8_grad 的真实组合是 e5m2×e4m3，独立探测；失败降级 bf16 反向
            # （速度旋钮而非语义，宁可慢不可首步 backward 才崩）
            if fp8_grad and gemm_ok:
                _grad_kind = ("fp8_grad_rowwise" if fp8_rowwise
                              else "fp8_grad_tensorwise")
                if not _gemm_probe(_grad_kind):
                    logger.warning(
                        "[base-quant] base_quant_fp8_grad=true 但 %s 探测失败"
                        "（%s）——反向降级 bf16，前向量化 GEMM 不受影响",
                        _grad_kind, _CAPS_ERR.get(_grad_kind, "?"))
                    print("[base-quant] ⚠ fp8_grad 探测失败，反向已降级 bf16（详见日志）",
                          flush=True)
                    fp8_grad = False
    else:  # fp4
        gemm_ok = False if gemm_pref == "off" else _gemm_probe("fp4")
        if gemm_pref == "on" and not gemm_ok:
            raise RuntimeError(
                "base_quant_gemm=on 但当前设备/torch 不支持 fp4 _scaled_mm"
                "（需要 Blackwell sm120+ 与较新 torch/cuBLAS）。H20 上请用 auto："
                "fp4 自动退回 dequant-bf16 计算，仍有 4× 权重显存收益。")
        if fp8_grad:
            raise ValueError("base_quant_fp8_grad 仅对 base_quant=fp8 有意义")

    # ── 决策落地前先把探测/决策结果打到 stdout ─────────────────────
    # （train_monitor 等外层只收 stdout 日志文件时，logging 的 stderr 行会
    # 丢——GEMM 是否启用是判断"该不该有速度收益"的第一依据，必须可见。）
    def _say(msg: str, warn: bool = False):
        print(msg, flush=True)
        (logger.warning if warn else logger.info)(msg)

    if fmt == "fp8":
        _say("[base-quant] 探测: fp8_rowwise=%s fp8_tensorwise=%s"
             % (_gemm_probe("fp8_rowwise"), _gemm_probe("fp8_tensorwise")))
        for kind in ("fp8_rowwise", "fp8_tensorwise"):
            if not _gemm_probe(kind) and kind in _CAPS_ERR:
                _say("[base-quant]   %s 失败原因: %s" % (kind, _CAPS_ERR[kind]))
        _say("[base-quant] 决策: fmt=fp8 quant_gemm=%s scale=%s fp8_grad=%s"
             % (gemm_ok, "rowwise" if fp8_rowwise else "tensorwise", fp8_grad))
        if gemm_pref != "off" and not gemm_ok:
            _say("[base-quant] ⚠ 当前设备/torch 探测不到可用的 fp8 _scaled_mm，"
                 "全部量化层将走 dequant-bf16：显存照省，但**不会提速**（前向还会"
                 "略慢于纯 bf16）。检查 torch 版本与 GPU 架构，或显式设 "
                 "base_quant_gemm: off 消除本警告。", warn=True)
        # ── 激活量化融合（opt-in）：只 compile 量化纯函数，不碰模型整图 ──────
        if bool(getattr(args, "base_quant_fuse_act_quant", False)):
            if gemm_ok:
                _fused = enable_fused_act_quant()
                _say("[base-quant] fuse_act_quant=%s%s"
                     % (_fused, "" if _fused else "（编译/探针失败，已回退 eager，见日志）"))
            else:
                _say("[base-quant] fuse_act_quant 请求但 quant-GEMM 未启用，跳过")
    else:
        _say("[base-quant] 探测: fp4_scaled_mm=%s" % gemm_ok)
        _say("[base-quant] 决策: fmt=fp4 quant_gemm=%s（False=dequant-bf16，"
             "H20 等无 fp4 tensor core 硬件的预期形态：纯省显存不提速）" % gemm_ok)

    inc_re = [re.compile(p) for p in include]
    skip_re = [re.compile(p) for p in skip]

    def _selected(name: str) -> bool:
        return (any(r.fullmatch(name) for r in inc_re)
                and not any(r.fullmatch(name) for r in skip_re))

    stats = {"format": fmt, "count": 0, "gemm": 0, "dequant": 0, "kept_bf16": 0,
             "bytes_before": 0, "bytes_after": 0,
             "relerr_sum": 0.0, "relerr_max": 0.0, "relerr_max_layer": ""}

    def _quantize_one(lin: torch.nn.Linear, name: str):
        in_f, out_f = lin.in_features, lin.out_features
        # 形状约束：fp8 GEMM 要求 K%16==0 且 N%16==0（反向还要转置，两维都要）；
        # fp4 存储要求 K%16==0，GEMM 保守要求 K%32==0 且 N%16==0。
        mode = "dequant"
        if fmt == "fp8":
            if gemm_ok and in_f % 16 == 0 and out_f % 16 == 0:
                mode = "fp8_gemm"
        else:
            if in_f % _FP4_BLOCK != 0:
                stats["kept_bf16"] += 1
                logger.info("[base-quant] %s: in_features=%d 不满足 fp4 block 约束，"
                            "保持 bf16", name, in_f)
                return None
            if gemm_ok and in_f % 32 == 0 and out_f % 16 == 0:
                mode = "fp4_gemm"
        qlin = QuantLinear.from_linear(
            lin, fmt, mode, fp8_rowwise=fp8_rowwise, fp8_grad=fp8_grad)
        with torch.no_grad():
            w = lin.weight.detach()
            deq = qlin._dequant_weight(torch.float32)
            relerr = ((deq - w.float()).norm() / w.float().norm().clamp(min=1e-12)).item()
        stats["count"] += 1
        stats["gemm" if mode.endswith("_gemm") else "dequant"] += 1
        stats["bytes_before"] += w.numel() * w.element_size()
        qbytes = qlin.weight_q.numel() * qlin.weight_q.element_size()
        for extra in ("weight_scale", "weight_scale2", "weight_scale_blocked"):
            b = getattr(qlin, extra, None)
            if b is not None:
                qbytes += b.numel() * b.element_size()
        stats["bytes_after"] += qbytes
        stats["relerr_sum"] += relerr
        if relerr > stats["relerr_max"]:
            stats["relerr_max"] = relerr
            stats["relerr_max_layer"] = name
        return qlin

    modules_snapshot = list(model.named_modules())
    modules_by_name = {n: m for n, m in modules_snapshot}
    claimed = set()  # LoRALinear 已处理的 original 的 dotted name

    for name, module in modules_snapshot:
        if not isinstance(module, LoRALinear):
            continue
        # 无条件 claim：即使该层被 include/skip 排除，其 .original 子模块也
        # 不允许在下面的裸 Linear 循环里以 "xxx.original" 名字绕过规则被量化
        claimed.add(f"{name}.original")
        if _selected(name) and isinstance(module.original, torch.nn.Linear):
            qlin = _quantize_one(module.original, name)
            if qlin is not None:
                module.original = qlin
    for name, module in modules_snapshot:
        if not isinstance(module, torch.nn.Linear) or name in claimed:
            continue
        # adapter 内部的可训练 Linear（lora_up/lora_down 等）绝不量化
        if ".adapter" in name:
            continue
        if not _selected(name):
            continue
        if any(p.requires_grad for p in module.parameters()):
            # 量化存储是 buffer，无法回传权重梯度 → 可训练层量化会静默断训练。
            # 正常流程底模在注入前已整体 requires_grad_(False)，走到这说明有异常。
            logger.warning("[base-quant] %s 命中 include 但含 requires_grad 参数，"
                           "跳过量化（底模应在注入前整体冻结）", name)
            continue
        qlin = _quantize_one(module, name)
        if qlin is None:
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = modules_by_name[parent_name] if parent_name else model
        setattr(parent, child_name, qlin)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if stats["count"]:
        saved = (stats["bytes_before"] - stats["bytes_after"]) / (1 << 30)
        _say(
            "[base-quant] %s：量化 %d 层（quant-GEMM=%d, dequant-bf16=%d, 保持bf16=%d），"
            "权重 %.2fGB → %.2fGB（省 %.2fGB）；权重 relerr mean=%.4f max=%.4f (%s)%s"
            % (fmt, stats["count"], stats["gemm"], stats["dequant"], stats["kept_bf16"],
               stats["bytes_before"] / (1 << 30), stats["bytes_after"] / (1 << 30), saved,
               stats["relerr_sum"] / stats["count"], stats["relerr_max"],
               stats["relerr_max_layer"],
               "；fp8 scale=%s, fp8_grad=%s" % (
                   "rowwise" if fp8_rowwise else "tensorwise", fp8_grad)
               if fmt == "fp8" else "")
        )
    else:
        _say("[base-quant] ⚠ include/skip 规则没有命中任何 Linear，"
             "本次量化是 no-op（include=%s, skip=%s）" % (include, skip), warn=True)
    return stats
