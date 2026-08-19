#!/usr/bin/env bash
# ==============================================================================
#  把 Dockerfile 的构建上下文摆到平台要求的位置
#
#  平台规则：Dockerfile 里 COPY/ADD 的源路径**相对于家目录下的 dockerFileTemp**
#  （/public/home/${username}/dockerFileTemp）。在 Notebook 里这个目录就是
#  /root/private_data/dockerFileTemp —— 同一个挂载的两个名字。
#
#  用法：
#      bash AnimaLoraToolkit/tools/scnet_stage_dockerfile.sh              # 只带代码
#      bash AnimaLoraToolkit/tools/scnet_stage_dockerfile.sh --weights    # 连权重一起
#  跑完去控制台「镜像管理 → 构建镜像 → Dockerfile」，把 Dockerfile 内容贴进去。
# ==============================================================================
set -uo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
STAGE="${STAGE:-/root/private_data/dockerFileTemp}"
MODELS_DIR="${MODELS_DIR:-/root/private_data/anima_models}"
WITH_WEIGHTS=false
[ "${1:-}" = "--weights" ] && WITH_WEIGHTS=true

echo "=============================================================="
echo " 准备 Dockerfile 构建上下文"
echo " 仓库   : $REPO_DIR"
echo " 暂存区 : $STAGE"
echo " 带权重 : $WITH_WEIGHTS"
echo "=============================================================="

mkdir -p "$STAGE" || { echo "✗ 建不了 $STAGE"; exit 1; }

# --- 代码（排除 .git / 权重 / 数据集 / 输出 / venv）---------------------------
rm -rf "$STAGE/anima-lora-train"
mkdir -p "$STAGE/anima-lora-train"
tar -C "$REPO_DIR" \
    --exclude=.git --exclude='venv' --exclude='.venv' --exclude='output' \
    --exclude='AnimaLoraToolkit/Dataset' --exclude='AnimaLoraToolkit/monitor_data' \
    --exclude='*.safetensors' --exclude='*.log' --exclude='__pycache__' \
    -cf - . | tar -C "$STAGE/anima-lora-train" -xf -
echo "$(cd "$REPO_DIR" && git rev-parse HEAD 2>/dev/null)" > "$STAGE/anima-lora-train/.image_commit"
echo "  ✓ 代码 $(du -sh "$STAGE/anima-lora-train" | cut -f1)  commit=$(cut -c1-8 "$STAGE/anima-lora-train/.image_commit")"

# Dockerfile 本身也放一份，方便直接贴进控制台
cp "$REPO_DIR/Dockerfile.dcu" "$STAGE/Dockerfile" 2>/dev/null && echo "  ✓ Dockerfile 已复制到 $STAGE/Dockerfile"

# --- 权重（可选）-------------------------------------------------------------
if $WITH_WEIGHTS; then
    if [ ! -d "$MODELS_DIR" ]; then
        echo "  ✗ 找不到 $MODELS_DIR"; exit 1
    fi
    rm -rf "$STAGE/anima_models"
    cp -a "$MODELS_DIR" "$STAGE/anima_models"
    echo "  ✓ 权重 $(du -sh "$STAGE/anima_models" | cut -f1)"
    echo "     记得把 Dockerfile 里「层 6」的两行注释解开"
else
    rm -rf "$STAGE/anima_models"
    echo "  · 跳过权重（默认）。权重在平台共享目录里每个节点都能读，塞进镜像只会让"
    echo "    每次拉取多传几个 GB；要做完全自包含的参赛产物时再加 --weights"
fi

# --- 属主：平台的文件管理/构建服务用家目录属主的身份读，root 建的文件它可能读不到 ---
# motd 里那句「您在容器环境内创建的文件在文件管理中可能无法进行编辑修改，可以使用
# chown 命令赋权」说的就是这件事。家目录属主是谁就 chown 给谁。
HOME_OWNER="$(stat -c '%u:%g' "$(dirname "$STAGE")" 2>/dev/null)"
if [ -n "$HOME_OWNER" ] && [ "$HOME_OWNER" != "0:0" ]; then
    if chown -R "$HOME_OWNER" "$STAGE" 2>/dev/null; then
        echo "  ✓ 已把 $STAGE 的属主改成 $HOME_OWNER（平台构建服务才读得到）"
    else
        echo "  ⚠ chown 失败，若控制台看不到 dockerFileTemp 里的文件，手动执行："
        echo "      chown -R $HOME_OWNER $STAGE"
    fi
fi

echo "--------------------------------------------------------------"
echo "暂存区内容："
du -sh "$STAGE"/* 2>/dev/null | sed 's/^/  /'
echo
echo "单层体积检查（平台限制是 **docker 单层 ≤ 15GB**，不是镜像总大小）："
for d in "$STAGE"/*; do
    [ -e "$d" ] || continue
    sz=$(du -sm "$d" | cut -f1)
    if [ "$sz" -gt 15000 ]; then echo "  ✗ $(basename "$d") = ${sz}MB 超过单层上限"; else echo "  ✓ $(basename "$d") = ${sz}MB"; fi
done
echo
echo "下一步：控制台 → 镜像管理 → 构建镜像 → Dockerfile，"
echo "        「Dockerfile文件路径」填：$(dirname "$STAGE")/dockerFileTemp/Dockerfile"
echo
echo "当前 FROM： $(grep -m1 '^ARG BASE_IMAGE=' "$STAGE/Dockerfile" | cut -d= -f2-)"
echo "（该地址已用 config blob 的 Env 与运行中的实例对拍确认，见 docs/hygon-dcu.md §2.7）"
