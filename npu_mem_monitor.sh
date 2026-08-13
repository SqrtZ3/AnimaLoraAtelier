#!/usr/bin/env bash
# =============================================================================
# npu_mem_monitor.sh — 训练全程 HBM 水位采样（昇腾 910B / 启智 OpenI 版）
#
# 它是 vram_monitor.sh 的昇腾对应物。两点结构性差异：
#   1. 数据源从 nvidia-smi 换成 npu-smi；
#   2. **没有 guard 回占逻辑** —— 调试任务的 NPU 是独占分配的，不存在抢卡场景。
#
# 用法（一般由 run_npu.sh 自动拉起）：
#   bash npu_mem_monitor.sh <npu_id> <interval_s> <out_csv> <train_pid> <train_log>
#
# 自检（强烈建议首次上机先跑，5 秒出结果）：
#   bash npu_mem_monitor.sh --probe          # 打印原始 npu-smi + 解析出来的数值
#
# ⚠ 解析口径的可信度声明：
#   `npu-smi info` 主表每块芯片的第二行末尾是 `HBM-Usage(MB)` 的 `已用 / 总量`，
#   本脚本解析的就是它。**这个表格式我没有在真机上核对过**（本地无昇腾），
#   所以留了 --probe 自检和 -t usages 的备用解析。首次跑 --probe 看一眼，
#   数字对不上就把原始输出贴回来，我把解析器钉死。
# =============================================================================

set -u

PROBE=0
[ "${1:-}" = "--probe" ] && PROBE=1

NPU_ID="${1:-0}"
[ "$PROBE" = "1" ] && NPU_ID="${2:-0}"
INTERVAL="${2:-5}"
OUT="${3:-/tmp/code/output/npu_mem_timeline.csv}"
TRAIN_PID="${4:-}"
TRAIN_LOG="${5:-}"

# --- 解析器 A：主表 HBM-Usage 列（MB，精确值）--------------------------------
# 主表形如：
#   | 0     910B2               | OK            | 88.9  40      0    / 0        |
#   | 0                         | 0000:C1:00.0  | 0     0 / 0   3243 / 65536    |
# 第一行给 NPU id + 型号，第二行（带 Bus-Id，含冒号）给 Memory-Usage 与 HBM-Usage
# 两组 `a / b`。取**最后一组** = HBM。
parse_main_table() {
    npu-smi info 2>/dev/null | awk -F'|' -v want="$NPU_ID" '
        NF < 4 { next }
        {
            f2 = $2; f3 = $3; f4 = $4
            # 芯片首行：字段2 是 "id  型号"
            if (match(f2, /^[ \t]*[0-9]+[ \t]+[A-Za-z0-9]/)) {
                split(f2, a, /[ \t]+/); cur = a[2] == "" ? a[1] : a[2]
                if (cur == "") cur = a[1]
                # 取第一个纯数字
                for (i = 1; i <= 4; i++) if (a[i] ~ /^[0-9]+$/) { cur = a[i]; break }
                next
            }
            # 芯片数据行：字段3 含 Bus-Id（有冒号）
            if (f3 ~ /:/ && cur == want) {
                n = 0
                # 收集 f4 里所有 "数字 / 数字" 组
                s = f4
                while (match(s, /[0-9]+[ \t]*\/[ \t]*[0-9]+/)) {
                    pair = substr(s, RSTART, RLENGTH)
                    s = substr(s, RSTART + RLENGTH)
                    n++; last = pair
                }
                if (n > 0) { gsub(/[ \t]/, "", last); split(last, p, "/"); print p[1] "," p[2]; exit }
            }
        }'
}

# --- 解析器 B：-t usages（备用；只有百分比，1% ≈ 640MB，精度粗）-------------
parse_usages() {
    npu-smi info -t usages -i "$NPU_ID" -c 0 2>/dev/null | awk -F: '
        /HBM Capacity/   { gsub(/[ \t]/,"",$2); cap  = $2 }
        /HBM Usage Rate/ { gsub(/[ \t]/,"",$2); rate = $2 }
        END { if (cap != "" && rate != "") printf "%d,%d\n", cap * rate / 100, cap }'
}

read_hbm() {
    local r
    r=$(parse_main_table)
    if [ -z "$r" ]; then r=$(parse_usages); fi
    echo "$r"
}

# --- 训练进程自己占了多少（best-effort）--------------------------------------
# `npu-smi info` 尾部的进程表列出 "Process id / Process name / Process memory(MB)"。
# 匹配不到就报 0，不影响主曲线。
read_proc_mib() {
    [ -n "$TRAIN_PID" ] || { echo 0; return; }
    npu-smi info 2>/dev/null | awk -F'|' -v p="$TRAIN_PID" '
        NF < 4 { next }
        {
            for (i = 2; i <= NF; i++) { gsub(/^[ \t]+|[ \t]+$/, "", $i) }
            if ($3 ~ /^[0-9]+$/ && $3 + 0 == p + 0) {
                # 行尾有收尾的 "|"，$NF 是空串 —— 从后往前找第一个纯数字字段
                for (j = NF; j >= 4; j--) {
                    if ($j ~ /^[0-9]+$/) { print $j + 0; exit }
                }
            }
        }' || echo 0
}

# --- --probe 自检模式 --------------------------------------------------------
if [ "$PROBE" = "1" ]; then
    echo "=== npu-smi 是否存在 ==="
    command -v npu-smi >/dev/null 2>&1 && echo "  OK: $(command -v npu-smi)" || { echo "  ✗ 找不到 npu-smi"; exit 1; }
    echo
    echo "=== npu-smi info 原始输出（前 24 行）==="
    npu-smi info 2>&1 | head -n 24 | sed 's/^/  /'
    echo
    echo "=== 解析器 A（主表 HBM-Usage）npu_id=$NPU_ID ==="
    A=$(parse_main_table); echo "  已用MB,总量MB = ${A:-<解析失败>}"
    echo "=== 解析器 B（-t usages 百分比换算）==="
    B=$(parse_usages);     echo "  已用MB,总量MB = ${B:-<解析失败>}"
    if [ -n "${3:-}" ]; then
        TRAIN_PID="$3"
        echo "=== 进程表解析 pid=$TRAIN_PID ==="
        echo "  该进程占用MB = $(read_proc_mib)"
    fi
    echo
    echo "对照上面的原始输出，如果两个解析都不对，把这段完整输出贴回给我。"
    exit 0
fi

mkdir -p "$(dirname "$OUT")" 2>/dev/null
if [ ! -s "$OUT" ]; then
    echo "timestamp,elapsed_s,total_mib,used_mib,free_mib,train_proc_mib,stage" > "$OUT"
fi

START=$(date +%s)
PEAK_USED=0
PEAK_AT=0
PEAK_STAGE=""
TOTAL=0

# 取训练日志最后一行：先把 \r（进度条）拆成换行，再截断、清理 CSV 危险字符
log_tail() {
    [ -n "$TRAIN_LOG" ] && [ -f "$TRAIN_LOG" ] || return 0
    tail -c 4096 "$TRAIN_LOG" 2>/dev/null \
        | tr '\r' '\n' | grep -v '^[[:space:]]*$' | tail -n 1 \
        | cut -c1-80 | tr ',"' ';.' | tr -d '\n'
}

echo "[monitor] npu=$NPU_ID interval=${INTERVAL}s out=$OUT train_pid=$TRAIN_PID"

FAIL_STREAK=0
while true; do
    if [ -n "$TRAIN_PID" ] && ! kill -0 "$TRAIN_PID" 2>/dev/null; then
        echo "[monitor] 训练进程 $TRAIN_PID 已退出，停止采样。"
        break
    fi

    HBM=$(read_hbm)
    if [ -z "$HBM" ]; then
        FAIL_STREAK=$((FAIL_STREAK + 1))
        if [ "$FAIL_STREAK" = 1 ] || [ $((FAIL_STREAK % 12)) = 0 ]; then
            echo "[monitor] npu-smi 解析失败（第 ${FAIL_STREAK} 次）。跑 'bash $0 --probe' 看原始输出。"
        fi
        sleep "$INTERVAL"
        continue
    fi
    FAIL_STREAK=0

    USED=${HBM%%,*}
    TOTAL=${HBM##*,}
    FREE=$((TOTAL - USED))
    PROC_MIB=$(read_proc_mib)
    [ -z "$PROC_MIB" ] && PROC_MIB=0

    NOW=$(date +%s)
    ELAPSED=$((NOW - START))
    STAGE=$(log_tail)

    echo "$(date '+%F %T'),$ELAPSED,$TOTAL,$USED,$FREE,$PROC_MIB,\"$STAGE\"" >> "$OUT"

    if [ "$USED" -gt "$PEAK_USED" ] 2>/dev/null; then
        PEAK_USED=$USED
        PEAK_AT=$ELAPSED
        PEAK_STAGE=$STAGE
    fi

    sleep "$INTERVAL"
done

echo "-------------------------------------------------------------"
echo "[monitor] 峰值 HBM 占用: ${PEAK_USED} MiB / ${TOTAL:-?} MiB"
echo "[monitor] 出现在 elapsed=${PEAK_AT}s，当时日志: ${PEAK_STAGE}"
echo "[monitor] 完整曲线: $OUT"
echo "-------------------------------------------------------------"
echo "[monitor] 下一步：峰值离 65536 还有多少余量，决定 navit_token_budget 能不能从"
echo "          49152 往 65536 / 98304 抬（见 config/train_npu_c12.yaml 顶部的步骤 B）。"
