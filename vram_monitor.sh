#!/usr/bin/env bash
# =============================================================================
# vram_monitor.sh — 训练全程显存水位采样
#
# 用法（一般由 run.sh 自动拉起，也可单独用）：
#   bash vram_monitor.sh <gpu> <interval_s> <out_csv> <train_pid> <train_log> [guard_cmd]
#
# 行为：
#   1. 每 interval 秒采一次 nvidia-smi，追加一行到 out_csv；
#   2. 每行附带训练日志最后一行的前 80 字符 —— OOM 后按 elapsed 对齐，
#      就能看出峰值出现在哪个阶段（cache / VAE encode / step / eval / sample）；
#   3. 训练进程消失后打印峰值摘要并退出；若给了 guard_cmd，退出前把卡重新占回。
#
# 注意：nvidia-smi 报的是 torch caching allocator 的 **reserved** 水位，
#       不是 allocated。判断"离 OOM 还有多少余量"看的正是 reserved，所以这个
#       口径是对的，但它会比训练日志里 torch 自报的 allocated 偏高。
# =============================================================================

GPU="${1:-0}"
INTERVAL="${2:-5}"
OUT="${3:-./vram_timeline.csv}"
TRAIN_PID="${4:-}"
TRAIN_LOG="${5:-}"
GUARD_CMD="${6:-}"

if [ ! -s "$OUT" ]; then
    echo "timestamp,elapsed_s,total_mib,used_mib,free_mib,train_proc_mib,gpu_util,stage" > "$OUT"
fi

START=$(date +%s)
PEAK_USED=0
PEAK_AT=0
PEAK_STAGE=""

# 取训练日志最后一行：先把 \r（进度条）拆成换行，再截断、清理 CSV 危险字符
log_tail() {
    [ -n "$TRAIN_LOG" ] && [ -f "$TRAIN_LOG" ] || return 0
    tail -c 4096 "$TRAIN_LOG" 2>/dev/null \
        | tr '\r' '\n' | grep -v '^[[:space:]]*$' | tail -n 1 \
        | cut -c1-80 | tr ',"' ';.' | tr -d '\n'
}

echo "[monitor] gpu=$GPU interval=${INTERVAL}s out=$OUT train_pid=$TRAIN_PID"

while true; do
    # 训练进程已退出（自然结束或被 stop 脚本 kill）→ 收尾
    if [ -n "$TRAIN_PID" ] && ! kill -0 "$TRAIN_PID" 2>/dev/null; then
        echo "[monitor] 训练进程 $TRAIN_PID 已退出，停止采样。"
        break
    fi

    SMI=$(nvidia-smi -i "$GPU" \
        --query-gpu=memory.total,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    if [ -z "$SMI" ]; then
        echo "[monitor] nvidia-smi 查询失败，${INTERVAL}s 后重试。"
        sleep "$INTERVAL"
        continue
    fi

    TOTAL=$(echo "$SMI" | cut -d, -f1)
    USED=$(echo "$SMI" | cut -d, -f2)
    FREE=$(echo "$SMI" | cut -d, -f3)
    UTIL=$(echo "$SMI" | cut -d, -f4)

    # 训练进程自己占了多少（区分于同卡上的其他进程）
    PROC_MIB=$(nvidia-smi -i "$GPU" --query-compute-apps=pid,used_memory \
        --format=csv,noheader,nounits 2>/dev/null \
        | tr -d ' ' | awk -F, -v p="$TRAIN_PID" '$1+0==p+0 {print $2+0; exit}')
    [ -z "$PROC_MIB" ] && PROC_MIB=0

    NOW=$(date +%s)
    ELAPSED=$((NOW - START))
    STAGE=$(log_tail)

    echo "$(date '+%F %T'),$ELAPSED,$TOTAL,$USED,$FREE,$PROC_MIB,$UTIL,\"$STAGE\"" >> "$OUT"

    if [ "$USED" -gt "$PEAK_USED" ] 2>/dev/null; then
        PEAK_USED=$USED
        PEAK_AT=$ELAPSED
        PEAK_STAGE=$STAGE
    fi

    sleep "$INTERVAL"
done

echo "-------------------------------------------------------------"
echo "[monitor] 峰值显存占用: ${PEAK_USED} MiB / ${TOTAL:-?} MiB"
echo "[monitor] 出现在 elapsed=${PEAK_AT}s，当时日志: ${PEAK_STAGE}"
echo "[monitor] 完整曲线: $OUT"
echo "-------------------------------------------------------------"

# 训练已停 → 把卡重新占回去（保活），避免共享机器被别人抢走
if [ -n "$GUARD_CMD" ]; then
    echo "[monitor] 训练已结束，重新启动显存占位 guard..."
    sleep 3   # 等驱动完成回收，否则 guard 抢到的是还没释放干净的水位
    nohup bash -c "$GUARD_CMD" > ./vram_guard.log 2>&1 &
    echo "[monitor] guard 已重启 (PID: $!)，日志: ./vram_guard.log"
fi
