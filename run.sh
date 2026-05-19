#!/bin/bash

# ==============================================================================
#                Anima 统一训练与管理脚本 (Background Training & Management Script)
#
# 功能:
# 1. 检查 Python 环境和相关脚本/配置是否存在。
# 2. 在后台安全启动 AnimaLoraToolkit 训练过程。
# 3. 自动生成一个 'stop_anima_training.sh' 脚本，用于一键精准停止相关进程。
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

# --- 2. 启动训练 ---
# 后台运行训练，并将输出重定向到日志
echo "正在后台启动 Anima 训练脚本..."

# 注：PYTHONUNBUFFERED=1 + python -u 强制 stdout/stderr 无缓冲。
# 否则被 nohup 重定向到文件后 Python 走块缓冲，tail -f 会卡好几分钟才出新行。
nohup env PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u "$TRAIN_SCRIPT" --config="$CONFIG_FILE" \
    > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!

echo "训练任务已成功提交至后台运行 (PID: $TRAIN_PID)。"
echo "您可以通过以下命令实时查看训练进度和日志:"
echo "tail -f $LOG_FILE"
echo " "

# --- 3. 动态生成停止脚本 ---
echo "正在生成一键停止脚本 'stop_anima_training.sh'..."

cat > "$STOP_SCRIPT" <<EOF
#!/bin/bash
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