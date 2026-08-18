#!/usr/bin/env bash
# ==============================================================================
#  scnet Notebook：保存镜像前跑这个（幂等，可反复跑）
#
#  为什么需要它 —— 这个平台的目录规则决定了「什么进镜像」：
#    /                 系统盘，**随镜像保存**（限 15G）
#    /root/private_data 持久盘，**不随镜像保存**（但重启不丢）
#  所以依赖必须装在系统 python，代码要进镜像就得复制到 / 下面。
#  反过来，**重启实例会把 / 回退到基础镜像**——没保存过就等于白装。
#
#  用法：
#      bash AnimaLoraToolkit/tools/scnet_pack_image.sh          # 只装环境，不带代码
#      bash AnimaLoraToolkit/tools/scnet_pack_image.sh --code   # additionally 固化代码
#  跑完按提示去控制台「关机并保存开发环境」。
# ==============================================================================
set -uo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
IMAGE_CODE_DIR="${IMAGE_CODE_DIR:-/opt/anima-lora-train}"
WITH_CODE=false
[ "${1:-}" = "--code" ] && WITH_CODE=true

ok(){ echo "  ✓ $*"; }
bad(){ echo "  ✗ $*"; FAIL=1; }
FAIL=0

echo "=============================================================="
echo " scnet 镜像固化检查   仓库=$REPO_DIR   带代码=$WITH_CODE"
echo "=============================================================="

# --- 1) 环境脚本进系统盘 ------------------------------------------------------
echo "[1/5] 环境脚本"
if [ -f "$REPO_DIR/AnimaLoraToolkit/tools/scnet_env.sh" ]; then
    install -m 0755 "$REPO_DIR/AnimaLoraToolkit/tools/scnet_env.sh" /etc/profile.d/zz-scnet-env.sh
    ok "/etc/profile.d/zz-scnet-env.sh 已更新（登录 shell 自动生效）"
    # bashrc 指向系统盘那份，而不是 private_data —— 后者不进镜像，镜像换机器就断了
    sed -i '\#private_data/scnet_env.sh#d' /root/.bashrc 2>/dev/null
    grep -q "zz-scnet-env.sh" /root/.bashrc 2>/dev/null || \
        echo '[ -f /etc/profile.d/zz-scnet-env.sh ] && . /etc/profile.d/zz-scnet-env.sh' >> /root/.bashrc
    ok "/root/.bashrc 已挂上（交互式非登录 shell 也生效）"
else
    bad "找不到 tools/scnet_env.sh"
fi
# shellcheck disable=SC1091
. /etc/profile.d/zz-scnet-env.sh 2>/dev/null

# --- 2) torch 完整性（硬门槛）------------------------------------------------
echo "[2/5] torch 必须还是 HIP 构建"
python - <<'PY'
import sys
try:
    import torch
except Exception as e:
    sys.exit(f"  ✗ import torch 失败: {e}（DTK env 没生效？）")
hip, cuda = torch.version.hip, torch.version.cuda
print(f"  torch={torch.__version__} hip={hip} cuda={cuda} avail={torch.cuda.is_available()}")
if not hip or cuda:
    sys.exit("  ✗ torch 已被 PyPI 的 CUDA 版覆盖 —— 这个镜像不能存，先修")
print("  ✓ HIP 构建")
PY
[ $? -ne 0 ] && FAIL=1

# --- 3) 依赖必须在**系统** python 里，不能在任何 venv ------------------------
echo "[3/5] 必需依赖的安装位置"
python - <<'PY'
import importlib, sys
need = ["numpy","PIL","safetensors","transformers","einops","sentencepiece","yaml"]
extra = ["rich","omegaconf","wandb","accelerate"]
bad = []
for m in need + extra:
    try:
        mod = importlib.import_module(m)
        where = getattr(mod, "__file__", "") or ""
        tag = "系统" if "/usr/local/lib" in where else ("★venv/其它" if where else "?")
        mark = "✓" if (tag == "系统" or m in extra and tag != "★venv/其它") else "!"
        print(f"  {mark} {m:14s} {tag}")
        if tag == "★venv/其它":
            bad.append(m)
    except Exception:
        (bad if m in need else []).append(m)
        print(f"  {'✗' if m in need else '·'} {m:14s} 缺失{'（必需）' if m in need else '（可选）'}")
if bad:
    sys.exit(f"  ✗ 这些不在系统 python 里，存进镜像后会消失: {bad}")
PY
[ $? -ne 0 ] && FAIL=1

# --- 4) 可选：把代码固化进镜像 -----------------------------------------------
echo "[4/5] 代码"
if $WITH_CODE; then
    mkdir -p "$IMAGE_CODE_DIR"
    # 只带源码，不带 .git / 权重 / 数据集 / 输出（镜像限 15G）
    tar -C "$REPO_DIR" \
        --exclude=.git --exclude='venv' --exclude='.venv' --exclude='output' \
        --exclude='AnimaLoraToolkit/Dataset' --exclude='AnimaLoraToolkit/monitor_data' \
        --exclude='*.safetensors' --exclude='*.log' --exclude='__pycache__' \
        -cf - . | tar -C "$IMAGE_CODE_DIR" -xf -
    echo "$(cd "$REPO_DIR" && git rev-parse HEAD 2>/dev/null)" > "$IMAGE_CODE_DIR/.image_commit"
    ok "已固化到 $IMAGE_CODE_DIR（commit $(cat "$IMAGE_CODE_DIR/.image_commit" | cut -c1-8)，$(du -sh "$IMAGE_CODE_DIR" | cut -f1)）"
    echo "     注意：这是**镜像内的只读参考副本**；日常开发仍用 $REPO_DIR 的 git 工作副本，"
    echo "     每次要存镜像前重跑本脚本即可同步。"
else
    ok "跳过（不带代码；镜像只做通用环境层）"
fi

# --- 5) 体积 ------------------------------------------------------------------
echo "[5/5] 体积（平台上限 15G，口径未经验证，第一次存完对一下）"
du -sh /usr/local/lib/python3.11/site-packages 2>/dev/null | sed 's/^/  site-packages /'
$WITH_CODE && du -sh "$IMAGE_CODE_DIR" 2>/dev/null | sed 's/^/  代码          /'
echo "  ⚠ 权重（$([ -d /root/private_data/anima_models ] && du -sh /root/private_data/anima_models | cut -f1 || echo '未放置')）在持久盘，**不要**复制进 /"

echo "--------------------------------------------------------------"
if [ "$FAIL" -eq 0 ]; then
    echo "✓ 可以保存镜像了：控制台 → 关机 → 勾选「保存开发环境」"
else
    echo "✗ 有问题，先修完再存（上面标 ✗ 的项）"
fi
exit "$FAIL"
