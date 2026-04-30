#!/bin/bash

# ==========================================
# 1. 配置虚拟环境路径
# ==========================================
# 假设你的虚拟环境文件夹名字叫 "venv" 或 ".env"
# 如果你的名字不同（例如 "myenv"），请修改下面的变量
VENV_DIR="venv"

# 检查并激活当前目录下的虚拟环境
if [ -f "./$VENV_DIR/bin/activate" ]; then
    source "./$VENV_DIR/bin/activate"
    echo "成功激活虚拟环境: ./$VENV_DIR"
else
    echo "错误: 未找到虚拟环境激活脚本 (./$VENV_DIR/bin/activate)"
    echo "请确认虚拟环境名称是否正确。"
    exit 1
fi

# ==========================================
# 2. 硬编码参数配置区
# ==========================================
GPU_ID=0              # 想要占用的 GPU 索引号 (默认: 0)
KEEP_FREE=4096        # 目标保留的空闲显存，单位 MiB (例如 4096 就是留出 4GB 给突发任务)
HARD_MIN_FREE=2048    # 紧急释放显存的硬底线，单位 MiB (默认通常是 KEEP_FREE 的一半)
MAX_RESERVE=0         # 最大占用显存上限，单位 MiB (0 代表能占多少占多少)
BLOCK_SIZE=512        # 每次占用或释放的“步长”大小，单位 MiB (建议 256 或 512)
INTERVAL=1.0          # 检查显存状态的间隔时间，单位 秒 (默认 1.0)

# ==========================================
# 3. 启动 VRAM Guard
# ==========================================
echo "正在启动 GPU VRAM Guard (监控 cuda:$GPU_ID)..."
echo "目标保留空闲显存: ${KEEP_FREE} MiB"

# 执行 Python 脚本并传入参数
# 如果你不想看到密集的日志刷屏，可以在最后一行加上 --quiet
python3 gpu_vram_guard.py \
    --gpu "$GPU_ID" \
    --keep-free "$KEEP_FREE" \
    --hard-min-free "$HARD_MIN_FREE" \
    --max-reserve "$MAX_RESERVE" \
    --block "$BLOCK_SIZE" \
    --interval "$INTERVAL"
    # --quiet