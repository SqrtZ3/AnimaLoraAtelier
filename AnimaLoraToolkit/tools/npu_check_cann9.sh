#!/usr/bin/env bash
# CANN 9 升级可行性自检（只读，不装任何东西、不改任何环境）。
#
# 背景：openI/启智 的 NPU 镜像停在 CANN 8.2.RC2，而 CANN 9.x（9.0.0 正式版 /
# 9.2.0-beta.1）官方已发布。能不能装，卡点不在"容器里装不装得上"（CANN 支持
# pip/conda/yum 在线安装），而在宿主机的 **driver / firmware 版本**：
#   * 驱动在宿主机上，容器内改不了 —— 镜像 CANN 多新，驱动不配套就白搭
#   * 驱动决定 CANN 可用的特性上限；旧驱动配新 CANN 的典型后果：
#     driver version mismatch / acl op loading failed / 算子悄悄加载失败
#   * 官方 Toolkit 兼容区间 [N-1, N, N+1]：CANN 9.0.0 的 Toolkit 最低兼容
#     8.5.0 的 ops 包 —— 8.2 不在区间内
#
# 用法：
#     bash tools/npu_check_cann9.sh                 # 打印判定
#     bash tools/npu_check_cann9.sh > /tmp/c9.txt 2>&1   # 存档带回来
#
# 判读：
#   * verdict 行是结论；它只做**定性**判断（基于镜像自带 CANN ↔ 驱动配套的
#     关系 + 官方兼容区间），具体驱动版本号请用昇腾官方兼容性查询助手核对：
#     https://www.hiascend.com/hardware/compatibility

set -uo pipefail

say() { echo -e "\n\033[1m── $* ─────────────────────────────────────\033[0m"; }
kv()  { printf '   %-24s %s\n' "$1" "$2"; }

echo "════════════════════════════════════════════════════════════════════"
echo " CANN 9 升级可行性自检（只读）  $(date '+%F %T')"
echo "════════════════════════════════════════════════════════════════════"

# ── 1. 芯片与设备 ─────────────────────────────────────────────────────────────
say "1. 芯片与设备"
if command -v npu-smi >/dev/null 2>&1; then
  npu-smi info 2>&1 | head -n 14 | sed 's/^/   /'
else
  echo "   ⚠ 找不到 npu-smi —— 容器里没有 NPU 工具链，装 CANN 9 前先解决这个"
fi
kv "ASCEND_RT_VISIBLE_DEVICES" "${ASCEND_RT_VISIBLE_DEVICES:-<未设>}"
for f in /sys/class/npu/npu0/device/board_online /sys/class/npu/npu0/device/chip_info; do
  [ -f "$f" ] && kv "$f" "$(cat "$f" 2>/dev/null | tr '\n' ' ')"
done

# ── 2. 宿主机 driver / firmware（CANN 能升到多高的天花板） ─────────────────────
say "2. 宿主机 driver / firmware（CANN 升级的硬约束，容器内改不了）"
DRIVER_VER=""
FIRMWARE_VER=""
for f in /usr/local/Ascend/driver/version.info /usr/local/Ascend/driver/version.cfg \
         /usr/local/Ascend/firmware/version.info; do
  if [ -f "$f" ]; then
    echo "   $f:"
    sed 's/^/     /' "$f"
    # 抓版本号（如 24.1.rc2 / 25.0.0）
    v="$(grep -iE 'version|Version' "$f" | head -1 | grep -oE '[0-9]+\.[0-9]+[.a-zA-Z0-9]*' | head -1)"
    case "$f" in
      *driver*)  DRIVER_VER="${DRIVER_VER:-$v}" ;;
      *firmware*) FIRMWARE_VER="${FIRMWARE_VER:-$v}" ;;
    esac
  fi
done
# 另一个取驱动版本的路子：npu-smi board 信息里通常带驱动/固件版本行
if [ -z "$DRIVER_VER" ] && command -v npu-smi >/dev/null 2>&1; then
  DRIVER_VER="$(npu-smi info -t board -i 0 2>/dev/null | grep -iE 'driver|Version' | head -1 | grep -oE '[0-9]+\.[0-9]+[.a-zA-Z0-9]*' | head -1)"
fi
kv "driver 版本（解析）" "${DRIVER_VER:-<未解析到>}"
kv "firmware 版本（解析）" "${FIRMWARE_VER:-<未解析到>}"
if command -v npu-smi >/dev/null 2>&1; then
  echo "   npu-smi board 详情:"
  npu-smi info -t board -i 0 2>&1 | head -n 14 | sed 's/^/     /'
fi

# ── 3. 当前 CANN ──────────────────────────────────────────────────────────────
say "3. 当前 CANN（决定 torch/torch_npu 版本上限，也是与 CANN 9 的差距基准）"
CANN_VER=""
for f in /usr/local/Ascend/ascend-toolkit/latest/*/ascend_toolkit_install.info \
         /usr/local/Ascend/ascend-toolkit/latest/ascend_toolkit_install.info \
         /usr/local/Ascend/nnae/latest/ascend_nnae_install.info; do
  if [ -f "$f" ]; then
    echo "   $f:"
    sed 's/^/     /' "$f"
    v="$(grep -i '^version=' "$f" | head -1 | cut -d= -f2 | tr -d ' ')"
    CANN_VER="${CANN_VER:-$v}"
  fi
done
kv "CANN 版本（解析）" "${CANN_VER:-<未解析到>}"

# ── 4. torch / torch_npu ─────────────────────────────────────────────────────
say "4. torch / torch_npu"
for py in python3 python; do
  command -v "$py" >/dev/null 2>&1 || continue
  kv "$py" "$("$py" -V 2>&1) @ $(command -v "$py")"
  "$py" - <<'PY' 2>&1 | sed 's/^/     /'
import importlib
for mod in ("torch", "torch_npu"):
    try:
        m = importlib.import_module(mod)
        print(f"  {mod}: {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"  {mod}: <缺失: {type(e).__name__}>")
PY
  break   # 只查默认 python3
done

# ── 5. 判定 ──────────────────────────────────────────────────────────────────
say "5. 判定（定性；具体配套版本以昇腾兼容性查询助手为准）"
cat <<'EOF'
   经验对应关系（非官方，来自社区文档/案例的归纳，供定性参考）：
     驱动 23.0.x ↔ CANN 7.0 线     驱动 24.0.x ↔ CANN 8.0 线
     驱动 23.1.x ↔ CANN 7.1 线     驱动 24.1.x ↔ CANN 8.2 线
     驱动 25.x   ↔ CANN 8.5 / 9.0 线（需助手核对具体小版本）
   官方兼容区间（文档确认）：CANN 9.0.0 Toolkit 最低兼容 8.5.0 的 ops 包。
EOF

DRV_MAJ="$(echo "${DRIVER_VER:-}" | grep -oE '^[0-9]+' | head -1)"
CANN_MAJ="$(echo "${CANN_VER:-}" | grep -oE '^[0-9]+' | head -1)"
echo
case "$DRV_MAJ" in
  25|26)
    kv "verdict" "驱动 $DRIVER_VER 较新，CANN 9 有希望 —— 去兼容性查询助手核对具体配套版本后决定"
    ;;
  24)
    kv "verdict" "驱动 $DRIVER_VER 对应 CANN 8.x 线（8.2 镜像的典型配套），距 CANN 9 配套差 1-2 个版本线"
    echo "   → 硬装 CANN 9 大概率报 driver version mismatch / acl 初始化失败；不建议直接试"
    echo "   → 更现实的中间路：容器内装 CANN 8.5.0（在官方兼容区间内），上 torch_npu 2.7.1/2.8.0/2.9.0"
    ;;
  *)
    kv "verdict" "驱动版本未解析到（${DRIVER_VER:-空}）—— 把第 2 节原始输出贴给兼容性查询助手核对"
    ;;
esac
if [ -n "$CANN_MAJ" ] && [ "$CANN_MAJ" -lt 9 ] 2>/dev/null; then
  echo "   → 当前 CANN $CANN_VER 不在 CANN 9 的官方兼容区间 [8.5, 9.x] 内（8.2 < 8.5 下限）"
fi

# ── 6. 若想硬试：装前快照 + 装后验证清单 ──────────────────────────────────────
say "6. 若仍决定硬试 CANN 9：先留快照，再按清单验证"
cat <<'EOF'
   装前（记录基线，回滚用）：
     npu-smi info > /tmp/cann9_baseline_npu-smi.txt
     cp -r /usr/local/Ascend/ascend-toolkit/latest /tmp/cann9_baseline_toolkit 2>/dev/null || true
   装法参考（CANN 9.0.0 官方安装指南，支持 pip/conda/yum）：
     https://www.hiascend.com/document/detail/zh/canncommercial/latest/softwareinst/instg/instg_0000.html
     确切命令以该指南为准，别从博客抄（昇腾的 pip/conda 包名随版本变）。
   装后验证（按顺序，一步不过就别往下）：
     1) npu-smi info 还能列出卡（驱动层没崩）
     2) python -c "import torch_npu; print(torch_npu.__version__)"（新版 torch_npu 2.11/2.12 要配
        新版 CANN，装完先确认 import 不报 acl init failed）
     3) bash tools/npu_recon.sh        # 重新侦察，确认 CANN/torch_npu 配套合理
     4) python tools/npu_probe.py      # 算子级真机实测（matmul 数值、SDPA、backward）
     5) python tests/diag_npu_broadcast_matmul_grad.py --device npu
        # 2D×3D 广播 matmul 的 backward 是否已修（本次升级最想确认的一项）
     6) 短训练一版，对比 loss / LoKr w1 范数轨迹与 CUDA 是否同向
   回滚：容器内 CANN 是用户态，删掉 /usr/local/Ascend 下新装的部分、恢复 baseline 即可；
   但驱动若被平台更新，只能找平台方。
   平台侧选项：openI 支持「调试任务 → 提交镜像」，若确认 8.2 驱动配不了 CANN 9，
   唯一正路是向平台要更高 CANN 的镜像（或自带驱动 ≥25.x 的环境）。
EOF

echo
echo "════════════════════════════════════════════════════════════════════"
echo " 最终以昇腾官方兼容性查询助手为准："
echo "   https://www.hiascend.com/hardware/compatibility"
echo " 把第 2 节的 driver 版本与第 3 节的 CANN 版本填进去，选芯片型号即可"
echo "════════════════════════════════════════════════════════════════════"
