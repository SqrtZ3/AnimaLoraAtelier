# ABBA 适配器（`lora_type: "abba"`）

来源：*ABBA-Adapters: Efficient and Expressive Fine-Tuning of Foundation Models*
（arXiv:2505.14238，ICLR 2026；官方实现 CERT-Lab/abba）。opt-in / default-off：
不设 `lora_type: "abba"` 时训练器行为与引入前完全一致。

## 结构与动机

```
ΔW = √α1·√α2 · (B1@A1) ∘ (B2@A2)        ∘ = Hadamard 逐元素积
     B1:(out,r1) A1:(r1,in) B2:(out,r2) A2:(r2,in)
```

- 参数量 `(r1+r2)(in+out)`；**r1=r2=lora_rank/2 时与标准 LoRA rank=lora_rank
  完全同预算**；有效秩上限 r1·r2（r1=r2=16 → 256）。
- 可达集合**严格包含**标准 LoRA rank-r1 的全部解（B2@A2 学成全 1 矩阵即退化，
  全 1 矩阵 rank-1 可表示）——最坏情况是"半预算 LoRA"，有保底。这与 LoKr 的
  kron 块共享结构本质不同：本仓库对 Krea2 成功 LoRA 的取证显示其 ΔW 对 kron
  流形（f=8）能量可对齐率仅 ~2%，而 Hadamard 流形没有这一病理。
- 引入动机：在 Krea2 12B 上 LoKr（f4/f8 × adamw/muon-sf）全部不拟合后的
  结构化替代，目标是"体积低于纯 LoRA、拟合能力不塌方"。**在 12B DiT 画风
  任务上没有先验证据**，属于有结构性理由的实验项。

## 实现要点（`trainer/lora.py: ABBALayer`）

- **前向**：官方 Khatri-Rao 精确重排（Thm.1，数学恒等无近似）
  `ΔW·x = B_kr(A_kr·x)`，`B_kr = B1⊙r B2 (out, r1·r2)`，全程不物化 (out,in)。
  开销 ≈ 一个 rank-r1·r2 的标准 LoRA。
- **init**（官方 `init_weights_svd_mixed`）：`(B1,A1) ← svd_lowrank(W0, q=r1,
  niter=10)`（B1=U√Σ、A1=√ΣVᵀ），`B2=0`、A2 kaiming → **step 0 净 ΔW=0**。
  注入阶段逐层 SVD，264 层 GPU 上约 1 分钟。
- **scaling**：官方口径 `s1=√alpha1, s2=√alpha2`（不是 alpha/rank）。
- **导出（默认 native-only，云端下载友好）**：`injector.save()` 默认只写 native
  `abba_a1/b1/a2/b2 (+alpha1/alpha2)` —— **体积与同预算标准 LoRA 完全相同**
  （rank 32 预算 ≈ 235MB），resume 也用它。部署件在本地转换：

  ```
  python tools/abba_export_lora.py in_abba.safetensors out_lora.safetensors \
      [--energy 0.999] [--max-rank 128]
  ```

  转换 = KR 物化（精确恒等）+ 可选逐层 SVD 截断（QR 技巧，秒级），输出 kohya
  标准键、ComfyUI 直载；`--energy 1.0`（默认）时与训练前向 bf16 容差内一致。
  若确需云端直接产出可直载文件，设 `abba_export_kr: true`（文件 ~8×，慎用）。
  resume 只认 native 键（KR 乘积无法唯一回推 4 因子）。
- **step-0 梯度流**：B2=0 使 step 0 只有 B2 有梯度，一步后全因子解冻
  （与 LoRA 的 B=0 同类，单元测试覆盖）。

## 配置

```yaml
lora_type: "abba"
lora_rank: 32        # → r1=r2=16（预算 = LoRA r32）；模块级 lora_reg_dims 同样生效
# abba_alpha: 16.0   # alpha1=alpha2 统一覆盖；默认 None → 官方口径 alpha=r1
```

限制（构造期 fail-fast）：不与 `lokr` / `lora_variant: dora|tlora` /
`lora_init: pissa` / `rank_dropout>0` 组合（首版保持单变量面）。
`dropout` / `module_dropout` / `lora_reg_dims` / `loraplus_lr_ratio`
（作用于 B1/B2）/ 模块级 lr 正常支持。

## 已知不确定性

- LR 量纲：论文仅在 LLM 推理任务标定；Hadamard 调制（B1A1 携带 W0 主成分量级）
  下梯度尺度与 LoRA 不严格同量纲。建议首跑沿用基线 LR 做单变量 A/B，loss
  尖峰/发散再降半。
- 遥测 `telemetry_capacity`（eff_rank）目前只识别 LoKr 因子，ABBA 模块会被
  hasattr 守卫自然跳过（不报错、无数据）。
