"""闸门 ①b：块对角 splash 后端 ≡ 稠密参考（前向 + 反向），CPU 上跑。

`interpret=True` 让 Pallas 内核在 CPU 上解释执行，所以**不花 TPU 配额**就能验
attention.py 的段几何有没有接错。这是最容易出静默错的一处：mask 错了不会报异常，
只会让某些图看到别的图的 token（或看不到自己的），loss 照常下降、结果照常不对。

验四件事：
  S1 自注意力块对角 ≡ 稠密 + 块对角 bias
  S2 cross-attn 矩形块对角（q 图像段 / kv 文本段）≡ 稠密
  S3 反向（dq/dk/dv）≡ 稠密的反向 —— 训练用的是反向，前向对不代表反向对
  S4 段隔离的**行为判据**：改动第 j 段的 kv，第 i≠j 段的输出必须逐 bit 不变
     （S1~S3 都是与自己写的 bias 对拍，若 bias 也错了会一起错；S4 不依赖参考实现）

用法（jax 解释器）：python check_splash_blockdiag.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import anima_jax as A                                          # noqa: E402
import attention as AT                                         # noqa: E402
import jax                                                     # noqa: E402
import jax.numpy as jnp                                        # noqa: E402

jax.config.update("jax_enable_x64", False)

# 都是 128 的倍数（attention.py 的硬约束），故意取不等长以暴露段边界错位
VIS = [256, 384, 128]          # 图像段（最后一段可视作 padding 段）
TXT = [128, 128, 128]          # 每图文本段，定长
HEADS, HDIM = 2, 128
TOL = 2e-2                     # bf16 口径；splash 与稠密的 softmax 累加顺序不同

_bad = []


def rel(got, exp) -> float:
    got, exp = np.asarray(got, np.float64), np.asarray(exp, np.float64)
    if got.shape != exp.shape:
        return float("nan")
    return float(np.abs(got - exp).max() / max(np.abs(exp).max(), 1e-12))


def check(name, got, exp, tol=TOL):
    r = rel(got, exp)
    ok = r == r and r < tol
    print(f"  [{'OK ' if ok else 'BAD'}] {name:<28} rel={r:.3e}")
    if not ok:
        _bad.append(name)
    return ok


def rand(key, s):
    return jax.random.normal(jax.random.PRNGKey(key), (s, HEADS, HDIM), jnp.float32
                             ).astype(jnp.bfloat16)


def main() -> int:
    S, T = sum(VIS), sum(TXT)
    print(f"段: 图像 {VIS}(Σ={S}) / 文本 {TXT}(Σ={T}) | heads={HEADS} dim={HDIM}")

    self_sp = AT.make_splash_attn(VIS, VIS, HEADS, HDIM, interpret=True)
    self_dn = AT.make_dense_attn(VIS, VIS)
    cross_sp = AT.make_splash_attn(VIS, TXT, HEADS, HDIM, interpret=True)
    cross_dn = AT.make_dense_attn(VIS, TXT)

    q, k, v = rand(0, S), rand(1, S), rand(2, S)
    kt, vt = rand(3, T), rand(4, T)

    print("\nS1 自注意力（块对角）：")
    check("self_attn", self_sp(q, k, v), self_dn(q, k, v))

    print("\nS2 cross-attn（矩形块对角）：")
    check("cross_attn", cross_sp(q, kt, vt), cross_dn(q, kt, vt))

    print("\nS3 反向（训练真正用的路径）：")
    def loss(fn, qq, kk, vv):
        return jnp.sum(fn(qq, kk, vv).astype(jnp.float32) ** 2)
    gs = jax.grad(lambda a, b, c: loss(self_sp, a, b, c), argnums=(0, 1, 2))(q, k, v)
    gd = jax.grad(lambda a, b, c: loss(self_dn, a, b, c), argnums=(0, 1, 2))(q, k, v)
    for nm, a, b in zip(("dq", "dk", "dv"), gs, gd):
        check(f"self_{nm}", a, b)

    print("\nS4 段隔离行为判据（不依赖参考实现）：")
    # 只改第 2 段的 k/v，第 0/1 段的输出必须**逐 bit 不变**
    off = VIS[0] + VIS[1]
    k2 = k.at[off:].set(rand(9, VIS[2]))
    v2 = v.at[off:].set(rand(10, VIS[2]))
    o1, o2 = self_sp(q, k, v), self_sp(q, k2, v2)
    d_before = float(jnp.abs(o1[:off].astype(jnp.float32)
                             - o2[:off].astype(jnp.float32)).max())
    d_after = float(jnp.abs(o1[off:].astype(jnp.float32)
                            - o2[off:].astype(jnp.float32)).max())
    print(f"  [{'OK ' if d_before == 0 else 'BAD'}] 前两段输出不变        max_abs={d_before:.3e}")
    print(f"  [{'OK ' if d_after > 0 else 'BAD'}] 第三段输出确实变了    max_abs={d_after:.3e}")
    if d_before != 0:
        _bad.append("段泄漏：改第2段影响了前两段")
    if d_after <= 0:
        _bad.append("第3段没反应——mask 可能把整段屏蔽了")

    print("\nS5 两级 mask（粗粒度编译期跳块 + 运行时 segment_ids 精确边界）：")
    # 粗段 [256,384,128]，但真实图像只有 [200,300,100] token，其余是段内量化填充。
    # 自注意力：填充自成一段(99)；cross-attn：填充沿用宿主图段号（否则整行全 0）。
    REAL = [200, 300, 100]
    f_self, f_cross, off = np.empty(S, np.int32), np.empty(S, np.int32), 0
    for i, (c, r_) in enumerate(zip(VIS, REAL)):
        f_self[off:off + r_] = i
        f_self[off + r_:off + c] = 99
        f_cross[off:off + c] = i
        off += c
    f_txt = AT.segment_ids(TXT)
    sp = AT.make_splash_attn(VIS, VIS, HEADS, HDIM, jnp.asarray(f_self),
                             jnp.asarray(f_self), interpret=True)
    dn = AT.make_dense_attn(VIS, VIS, f_self, f_self)
    check("self_两级mask", sp(q, k, v), dn(q, k, v))
    csp = AT.make_splash_attn(VIS, TXT, HEADS, HDIM, jnp.asarray(f_cross),
                              jnp.asarray(f_txt), interpret=True)
    cdn = AT.make_dense_attn(VIS, TXT, f_cross, f_txt)
    check("cross_两级mask", csp(q, kt, vt), cdn(q, kt, vt))

    o_ref = sp(q, k, v)
    kp = k.at[200:256].set(rand(11, 56))       # 只动第 0 段的**段内填充区**
    d = float(jnp.abs(o_ref[:200].astype(jnp.float32)
                      - sp(q, kp, v)[:200].astype(jnp.float32)).max())
    print(f"  [{'OK ' if d == 0 else 'BAD'}] 真token不受段内填充影响  max_abs={d:.3e}")
    if d != 0:
        _bad.append("段内填充泄漏进真 token")
    if not bool(jnp.isfinite(o_ref).all()):
        _bad.append("两级 mask 下出现 NaN/Inf（多半是某行被全屏蔽）")

    print(f"\n*** {'通过' if not _bad else '失败'} ***"
          + ("" if not _bad else f"  失败项: {_bad}"))
    return 0 if not _bad else 1


if __name__ == "__main__":
    sys.exit(main())
