#!/bin/bash

# ==============================================================================
#                Anima 统一训练与管理脚本 (Background Training & Management Script)
#
# 功能:
# 1. 检查 Python 环境和相关脚本/配置是否存在。
# 2. 启动训练前，自动让出 gpu_vram_guard.py 占着的显存（并等确认释放完毕）。
# 3. 在后台安全启动 AnimaLoraToolkit 训练过程。
# 4. 全程后台采样显存水位 + 当时的训练阶段，写入 vram_timeline.csv。
# 5. 训练结束/被停止后，自动把显存重新占回（保活），防止共享机器被抢卡。
# 6. 自动生成一个 'stop_anima_training.sh' 脚本，用于一键精准停止相关进程。
#
# 关于显存池的说明（重要）:
#   CUDA 显存不跨进程共享。guard 进程持有的显存，训练进程**拿不到**，
#   而且 guard 的释放是秒级轮询、训练的 OOM 是瞬时的，二者并存只会互相踩踏。
#   因此这里的策略是 **互斥接力**：guard 平时占卡 → 训练启动时完全让出 →
#   训练期间只做只读监控 → 训练停止后 guard 再占回来。
# ==============================================================================

# --- 用户配置区域 (User Configuration) ---

# 虚拟环境 Python 路径
PYTHON_BIN="./venv/bin/python"

# 训练脚本路径
TRAIN_SCRIPT="./AnimaLoraToolkit/anima_train.py"

# 训练配置文件路径
CONFIG_FILE="./AnimaLoraToolkit/config/train_my.yaml"

# 日志输出路径
LOG_FILE="./anima_training.log"

# CUDA caching allocator 配置。
# expandable_segments:True 让 allocator 用可扩展的虚拟内存段，而不是固定大小的 block，
# 显著减少变长 shape（ARB 分桶 / navit 变长打包）造成的显存碎片。
# 依据：H20 云端 c12port run 单变量 A/B 实测，整步时间稳定 −17~18%
#       （fwd/bwd/optimizer 等比例下降），同时降低碎片型 OOM 概率。
# 若某次要做 A/B 对照，把它置空即可回到 PyTorch 默认行为。
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# --- 显存 guard / 监控配置 ---

# 启动训练前是否自动停掉 guard 并让出显存；训练停止后是否自动占回
ENABLE_VRAM_GUARD=true
# 训练全程是否记录显存曲线
ENABLE_VRAM_MONITOR=true

GPU_INDEX=0                      # nvidia-smi 的卡号（注意与 CUDA_VISIBLE_DEVICES 的映射）
GUARD_LAUNCHER="./block.sh"      # guard 启动器（占位参数在它里面改）
MONITOR_SCRIPT="./vram_monitor.sh"
VRAM_CSV="./vram_timeline.csv"
MONITOR_INTERVAL=5               # 显存采样间隔（秒）
MONITOR_LOG="./vram_monitor.log"
GUARD_RELEASE_TIMEOUT=30         # 等 guard 退出并释放显存的最长秒数


# ==============================================================================
#                  脚本主体 (DO NOT MODIFY BELOW THIS LINE)
# ==============================================================================

STOP_SCRIPT="./stop_anima_training.sh"

# --- 1. 路径与文件检查 ---
echo "正在检查运行环境..."
if [ ! -f "$PYTHON_BIN" ]; then
    echo "错误: Python 虚拟环境未找到: $PYTHON_BIN"
    exit 1
fi

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "错误: 训练脚本未找到: $TRAIN_SCRIPT"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件未找到: $CONFIG_FILE"
    exit 1
fi
echo "环境检查通过。"
echo " "

# --- 1.5 让出 guard 占用的显存 ---
# guard 收到 SIGTERM 后会在 finally 里 blocks.clear() + empty_cache() 优雅释放，
# 因为它的循环每 interval 秒检查一次 stopping，所以正常 1~2 秒内退出。
show_vram() {
    nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.used,memory.free,memory.total \
        --format=csv,noheader 2>/dev/null \
        | awk -F, '{printf "  显存: 已用%s / 空闲%s / 总计%s\n", $1, $2, $3}'
}

if [ "$ENABLE_VRAM_GUARD" = "true" ]; then
    echo "正在检查显存占位进程 (gpu_vram_guard.py)..."
    show_vram

    # 先清掉可能残留的旧监控进程，否则它会在我们停 guard 后又把卡占回去
    pkill -f "vram_monitor.sh" > /dev/null 2>&1
    GUARD_PIDS=$(pgrep -f "gpu_vram_guard.py" 2>/dev/null)

    if [ -n "$GUARD_PIDS" ]; then
        echo "发现 guard 进程 (PIDs: $(echo $GUARD_PIDS | tr '\n' ' '))，正在请其让出显存..."
        kill -TERM $GUARD_PIDS 2>/dev/null

        WAITED=0
        while [ $WAITED -lt $GUARD_RELEASE_TIMEOUT ]; do
            pgrep -f "gpu_vram_guard.py" > /dev/null 2>&1 || break
            sleep 1
            WAITED=$((WAITED + 1))
        done

        if pgrep -f "gpu_vram_guard.py" > /dev/null 2>&1; then
            echo "警告: guard 在 ${GUARD_RELEASE_TIMEOUT}s 内未优雅退出，强制终止。"
            pkill -9 -f "gpu_vram_guard.py" 2>/dev/null
            sleep 2
        fi

        # 也要等 block.sh 这层 wrapper 退出，否则 pgrep 判断会有残留
        pkill -f "bash .*block\.sh" 2>/dev/null

        sleep 2   # 等驱动完成回收
        echo "guard 已停止，显存已让出:"
        show_vram
    else
        echo "未发现 guard 进程，跳过。"
    fi
    echo " "
fi

# --- 2. 启动训练 ---
# 后台运行训练，并将输出重定向到日志
echo "正在后台启动 Anima 训练脚本..."

# 注：PYTHONUNBUFFERED=1 + python -u 强制 stdout/stderr 无缓冲。
# 否则被 nohup 重定向到文件后 Python 走块缓冲，tail -f 会卡好几分钟才出新行。
echo "  PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
nohup env PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
    "$PYTHON_BIN" -u "$TRAIN_SCRIPT" --config="$CONFIG_FILE" \
    > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!

echo "训练任务已成功提交至后台运行 (PID: $TRAIN_PID)。"
echo "您可以通过以下命令实时查看训练进度和日志:"
echo "tail -f $LOG_FILE"
echo " "

# --- 2.5 启动显存监控 ---
MONITOR_PID=""
if [ "$ENABLE_VRAM_MONITOR" = "true" ]; then
    if [ ! -f "$MONITOR_SCRIPT" ]; then
        echo "警告: 未找到 $MONITOR_SCRIPT，跳过显存监控。"
    else
        # 训练停止后由 monitor 负责把卡占回去（自然退出和被 kill 两条路径都覆盖）
        GUARD_CMD=""
        if [ "$ENABLE_VRAM_GUARD" = "true" ] && [ -f "$GUARD_LAUNCHER" ]; then
            GUARD_CMD="bash $GUARD_LAUNCHER"
        fi

        nohup bash "$MONITOR_SCRIPT" "$GPU_INDEX" "$MONITOR_INTERVAL" "$VRAM_CSV" \
            "$TRAIN_PID" "$LOG_FILE" "$GUARD_CMD" > "$MONITOR_LOG" 2>&1 &
        MONITOR_PID=$!
        echo "显存监控已启动 (PID: $MONITOR_PID)，每 ${MONITOR_INTERVAL}s 采样一次。"
        echo "  曲线文件: $VRAM_CSV"
        echo "  监控日志: $MONITOR_LOG  (训练结束后这里会打印峰值摘要)"
        echo "  实时查看: tail -f $VRAM_CSV"
        echo " "
    fi
fi

# --- 3. 动态生成停止脚本 ---
echo "正在生成一键停止脚本 'stop_anima_training.sh'..."

cat > "$STOP_SCRIPT" <<EOF
#!/bin/bash
# 用法:
#   bash $STOP_SCRIPT             停训练，随后 guard 自动把显存占回（保活）
#   bash $STOP_SCRIPT --release   停训练 + 停监控 + 停 guard，彻底释放整张卡

RELEASE_ALL=false
[ "\$1" = "--release" ] && RELEASE_ALL=true

if [ "\$RELEASE_ALL" = "true" ]; then
    # 必须先杀监控，否则它在训练退出后会把卡重新占回去
    echo "正在停止显存监控..."
    pkill -f "vram_monitor.sh" > /dev/null 2>&1
fi

echo "正在查找 Anima 训练进程..."

# 查找匹配 anima_train.py 及其特定配置文件的进程
# 使用精确匹配，防止误杀您的其他无关 Python 进程
TRAIN_PIDS=\$(pgrep -f "anima_train.py.*$CONFIG_FILE")

if [ -z "\$TRAIN_PIDS" ]; then
    echo "未找到正在运行的 Anima 训练进程。"
else
    # 格式化 PID 输出，去除多余空格
    PIDS_TO_KILL=\$(echo \$TRAIN_PIDS | tr '\\n' ' ')
    echo "将要终止以下进程 (PIDs): \$PIDS_TO_KILL"

    # 强制终止进程并丢弃可能产生的错误信息
    kill -9 \$PIDS_TO_KILL > /dev/null 2>&1
    echo "所有相关进程已被终止。"
fi

if [ "\$RELEASE_ALL" = "true" ]; then
    echo "正在停止显存占位 guard..."
    pkill -TERM -f "gpu_vram_guard.py" > /dev/null 2>&1
    sleep 2
    pkill -9 -f "gpu_vram_guard.py" > /dev/null 2>&1
    pkill -f "bash .*block\\.sh" > /dev/null 2>&1
    echo "整张卡已释放。"
else
    echo "监控进程会在检测到训练退出后打印峰值摘要，并把显存重新占回。"
    echo "峰值摘要: tail -20 $MONITOR_LOG"
    echo " "
    echo "本脚本已保留（未自删），若还想连 guard 一起停、彻底释放整张卡，执行:"
    echo "  bash \$0 --release"
    exit 0
fi

echo "脚本执行完毕，将自动删除自身..."
rm -- "\$0"
EOF

# 赋予停止脚本执行权限
chmod +x "$STOP_SCRIPT"

echo "停止脚本 '$STOP_SCRIPT' 已成功创建。"
echo "--------------------------------------------------------"
echo "当您需要停止训练时，只需在终端运行这个命令:"
echo "bash $STOP_SCRIPT"
echo "--------------------------------------------------------"