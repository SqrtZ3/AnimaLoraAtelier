#!/usr/bin/env bash
# ==============================================================================
#        Anima 训练启动脚本 · 昇腾 910B / 启智 OpenI 调试任务版
#
# 这是 run.sh 的昇腾对应物。相对 CUDA 版的**结构性**差异（不是简单换命令）：
#
#   1. ★ setsid：训练必须脱离终端会话。启智调试任务的 JupyterLab 启动参数含
#      TerminalManager.cull_inactive_timeout=1800，30 分钟无活动的终端会被回收，
#      同会话的子进程会被一起带走。CUDA 云上不需要这条，这里是硬要求。
#   2. 删掉了 gpu_vram_guard / block.sh 那一整套「让出→占回」接力。调试任务的
#      NPU 是独占分配的，没有抢卡场景；guard 只会白白吃掉 64GB 里的显存。
#   3. nvidia-smi → npu-smi（见 npu_mem_monitor.sh）。
#   4. PYTORCH_CUDA_ALLOC_CONF → PYTORCH_NPU_ALLOC_CONF，且**默认留空**。
#      expandable_segments 的 −17~18% 是 H20 CUDA 上的单变量实测，昇腾 allocator
#      是否支持这个键我没有验证过，首跑不引入未知变量。见下方说明。
#   5. 解释器用绝对路径：镜像里 python / pip 都不在 PATH。
#
# 用法:
#   bash run_npu.sh              启动训练（后台，脱离终端）
#   bash run_npu.sh --follow     启动后顺便 tail -f 日志（Ctrl-C 只断 tail，不停训练）
#   bash run_npu.sh --dry-run    只做体检，不启动
# ==============================================================================

set -u

# --- 用户配置区域 -------------------------------------------------------------
# 下面这些都支持环境变量覆盖，换路径不用改脚本，例如：
#   CONFIG_FILE=config/train_npu_other.yaml bash run_npu.sh

# 仓库在机器上的位置（JupyterLab 侧栏根目录是 /tmp/code）
REPO_DIR="${REPO_DIR:-/tmp/code/anima-lora-train}"

# 训练进程的工作目录。配置文件里的相对路径都以它为基准；
# train_npu_c12.yaml 的模型/数据/输出路径全是绝对路径，所以这里选哪个都能跑。
WORKDIR="${WORKDIR:-$REPO_DIR/AnimaLoraToolkit}"

# 解释器（★ 镜像里不在 PATH，必须绝对路径。换镜像后用
#   ls -d /usr/local/python3.* 确认版本号）
PYTHON_BIN="${PYTHON_BIN:-/usr/local/python3.11.13/bin/python}"

# 训练脚本 / 配置（相对 WORKDIR）
TRAIN_SCRIPT="${TRAIN_SCRIPT:-anima_train.py}"
CONFIG_FILE="${CONFIG_FILE:-config/train_npu_c12.yaml}"

# 日志与产物：放 /tmp/code 下，JupyterLab 侧栏能直接看到并下载
OUT_DIR="${OUT_DIR:-/tmp/code/output}"
LOG_FILE="$OUT_DIR/anima_training_npu.log"
MEM_CSV="$OUT_DIR/npu_mem_timeline.csv"
MONITOR_LOG="$OUT_DIR/npu_mem_monitor.log"

# --- 昇腾 allocator 配置 ---
# 空 = PyTorch/torch_npu 默认行为（首跑推荐）。
# 想做 A/B 时填 "expandable_segments:True"：CUDA 上它把变长 shape（ARB 分桶 /
# navit 变长打包）的碎片压下去，H20 实测整步 −17~18%。昇腾上**未验证**，
# 若 torch_npu 不认这个键，可能报错或静默忽略——所以要单独作为一次 A/B 来测，
# 不要和首跑混在一起。
PYTORCH_NPU_ALLOC_CONF=""

# --- 监控配置 ---
ENABLE_MEM_MONITOR="${ENABLE_MEM_MONITOR:-true}"
NPU_ID="${NPU_ID:-0}"          # npu-smi 的卡号
MONITOR_SCRIPT="${MONITOR_SCRIPT:-$REPO_DIR/npu_mem_monitor.sh}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-5}"

# ==============================================================================
#                  脚本主体 (DO NOT MODIFY BELOW THIS LINE)
# ==============================================================================

FOLLOW=false
DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --follow)  FOLLOW=true ;;
        --dry-run) DRY_RUN=true ;;
        *) echo "未知参数: $arg（可用: --follow / --dry-run）"; exit 1 ;;
    esac
done

STOP_SCRIPT="$REPO_DIR/stop_anima_training_npu.sh"
FAIL=0

note()  { echo "  $*"; }
bad()   { echo "  ✗ $*"; FAIL=1; }
warn()  { echo "  ⚠ $*"; }

echo "=============================================================="
echo " Anima 训练启动 · 昇腾 910B / 启智 OpenI"
echo "=============================================================="
echo
echo "[1/5] 环境体检"

[ -d "$WORKDIR" ]     || bad "工作目录不存在: $WORKDIR"
[ -x "$PYTHON_BIN" ]  || bad "解释器不存在或不可执行: $PYTHON_BIN  （试 ls -d /usr/local/python3.*）"
[ -f "$WORKDIR/$TRAIN_SCRIPT" ] || bad "训练脚本未找到: $WORKDIR/$TRAIN_SCRIPT"
[ -f "$WORKDIR/$CONFIG_FILE" ]  || bad "配置文件未找到: $WORKDIR/$CONFIG_FILE"

if [ "$FAIL" = 0 ]; then
    note "解释器: $("$PYTHON_BIN" -V 2>&1)"
    if "$PYTHON_BIN" -c "import torch, torch_npu" 2>/dev/null; then
        note "torch/torch_npu: $("$PYTHON_BIN" -c \
            'import torch,torch_npu;print(f"torch {torch.__version__} / torch_npu {torch_npu.__version__}")' 2>&1)"
        NPU_CNT=$("$PYTHON_BIN" -c 'import torch,torch_npu;print(torch.npu.device_count())' 2>/dev/null)
        [ "${NPU_CNT:-0}" -ge 1 ] 2>/dev/null \
            && note "可见 NPU 数量: $NPU_CNT" \
            || bad "torch 看不到 NPU（device_count=${NPU_CNT:-?}）"
    else
        bad "import torch / torch_npu 失败 —— 先跑 bash AnimaLoraToolkit/tools/npu_setup_image.sh --check"
    fi
fi

command -v npu-smi >/dev/null 2>&1 || warn "找不到 npu-smi，显存曲线会采不到（训练本身不受影响）"

echo
echo "[2/5] 配置文件路径检查（从 yaml 里读，逐条 stat）"

# 极简 yaml 取值：只支持 `key: "value"` / `key: value` 这种顶层标量，够用。
cfg_get() {
    sed -n "s/^[[:space:]]*$1[[:space:]]*:[[:space:]]*//p" "$WORKDIR/$CONFIG_FILE" \
        | head -n 1 | sed 's/[[:space:]]*#.*$//' | sed 's/^"//; s/"$//' | sed "s/^'//; s/'$//"
}

check_path() {  # <标签> <路径> <file|dir>
    local label="$1" p="$2" kind="$3"
    if [ -z "$p" ]; then bad "$label: 配置里没读到值"; return; fi
    if [ "$kind" = dir ]; then
        [ -d "$p" ] && note "$label: $p" || bad "$label 目录不存在: $p"
    else
        [ -f "$p" ] && note "$label: $p ($(du -h "$p" 2>/dev/null | cut -f1))" \
                    || bad "$label 文件不存在: $p"
    fi
}

check_path "DiT"        "$(cfg_get transformer_path)"   file
check_path "VAE"        "$(cfg_get vae_path)"           file
check_path "TE(HF目录)" "$(cfg_get text_encoder_path)"  dir
check_path "t5_tokenizer" "$(cfg_get t5_tokenizer_path)" dir
check_path "数据集"     "$(cfg_get data_dir)"           dir

DATA_DIR="$(cfg_get data_dir)"
if [ -d "$DATA_DIR" ]; then
    # 递归统计，与 trainer/data.py:808 的 `data_dir.rglob("*")` 口径一致
    #（所以 zip 解出来多套一层子目录不影响训练，图照样能被枚举到）
    IMG_N=$(find "$DATA_DIR" -type f \
        \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.webp' \) 2>/dev/null | wc -l)
    TXT_N=$(find "$DATA_DIR" -type f -iname '*.txt' 2>/dev/null | wc -l)
    note "数据集内容: 图 $IMG_N 张 / caption $TXT_N 个（递归统计）"
    [ "$IMG_N" -gt 0 ] || bad "数据集目录里没找到图片 —— zip 解开了吗？data_dir 指对了吗？"
    if [ "$IMG_N" -gt 0 ] && [ "$TXT_N" -lt "$IMG_N" ]; then
        warn "caption 数少于图片数（$TXT_N < $IMG_N），确认是否有漏打标"
    fi
fi

DEV_BACKEND="$(cfg_get device_backend)"
ATTN_BACKEND="$(cfg_get navit_attn_backend)"
[ "$DEV_BACKEND" = "npu" ] || bad "device_backend 应为 npu，实际是 '${DEV_BACKEND:-<未设>}'"
case "$ATTN_BACKEND" in
    npu_tnd|sdpa_seg) note "navit_attn_backend: $ATTN_BACKEND" ;;
    "")       warn "navit_attn_backend 未设 —— 默认 xformers 在昇腾上会 fail-fast" ;;
    *)        bad  "navit_attn_backend='$ATTN_BACKEND' 在昇腾上不可用，改 npu_tnd 或 sdpa_seg" ;;
esac
note "max_steps: $(cfg_get max_steps)   navit_token_budget: $(cfg_get navit_token_budget)"

echo
echo "[3/5] 磁盘与已有进程"
mkdir -p "$OUT_DIR"
note "输出目录: $OUT_DIR（剩余空间 $(df -h "$OUT_DIR" 2>/dev/null | awk 'NR==2{print $4}')）"

RUNNING=$(pgrep -f "$TRAIN_SCRIPT.*$CONFIG_FILE" 2>/dev/null | tr '\n' ' ')
if [ -n "$RUNNING" ]; then
    bad "已经有训练进程在跑 (PIDs: $RUNNING)。先 bash $STOP_SCRIPT，或确认是不是重复启动。"
fi

if [ "$FAIL" != 0 ]; then
    echo
    echo "体检未通过，已中止启动。修完上面的 ✗ 再来。"
    exit 1
fi
echo "  体检通过。"

if [ "$DRY_RUN" = true ]; then
    echo
    echo "[--dry-run] 到此为止，未启动训练。"
    exit 0
fi

echo
echo "[4/5] 启动训练（setsid 脱离终端，cull 杀不到）"

# PYTHONUNBUFFERED=1 + python -u：否则输出被重定向到文件后走块缓冲，
# tail -f 要卡好几分钟才出新行。
ENV_ARGS=(PYTHONUNBUFFERED=1)
if [ -n "$PYTORCH_NPU_ALLOC_CONF" ]; then
    ENV_ARGS+=(PYTORCH_NPU_ALLOC_CONF="$PYTORCH_NPU_ALLOC_CONF")
    note "PYTORCH_NPU_ALLOC_CONF=$PYTORCH_NPU_ALLOC_CONF （★非默认，记得这次是 A/B）"
else
    note "PYTORCH_NPU_ALLOC_CONF=<未设，用 torch_npu 默认>"
fi

cd "$WORKDIR" || exit 1
setsid nohup env "${ENV_ARGS[@]}" \
    "$PYTHON_BIN" -u "$TRAIN_SCRIPT" --config="$CONFIG_FILE" \
    > "$LOG_FILE" 2>&1 < /dev/null &

# setsid 可能自己 fork 一层，`$!` 拿到的不一定是 python 的 PID。
# 所以按命令行特征回查真实 PID（最多等 30 秒 —— 昇腾首次 import torch_npu 较慢）。
TRAIN_PID=""
for _ in $(seq 1 30); do
    TRAIN_PID=$(pgrep -f "$TRAIN_SCRIPT.*$CONFIG_FILE" 2>/dev/null | head -n 1)
    [ -n "$TRAIN_PID" ] && break
    sleep 1
done

if [ -z "$TRAIN_PID" ]; then
    echo "  ✗ 30 秒内没找到训练进程，多半是启动即崩。日志尾部："
    tail -n 30 "$LOG_FILE" 2>/dev/null | sed 's/^/      /'
    exit 1
fi

note "训练已在后台运行 (PID: $TRAIN_PID)"
note "会话 ID: $(ps -o sid= -p "$TRAIN_PID" 2>/dev/null | tr -d ' ')  （与当前终端不同即为脱离成功）"
note "日志: $LOG_FILE"

echo
echo "[5/5] 显存监控与停止脚本"

MONITOR_PID=""
if [ "$ENABLE_MEM_MONITOR" = true ]; then
    if [ ! -f "$MONITOR_SCRIPT" ]; then
        warn "未找到 $MONITOR_SCRIPT，跳过显存监控。"
    else
        setsid nohup bash "$MONITOR_SCRIPT" "$NPU_ID" "$MONITOR_INTERVAL" \
            "$MEM_CSV" "$TRAIN_PID" "$LOG_FILE" > "$MONITOR_LOG" 2>&1 < /dev/null &
        sleep 1
        MONITOR_PID=$(pgrep -f "npu_mem_monitor.sh $NPU_ID" 2>/dev/null | head -n 1)
        note "显存监控已启动 (PID: ${MONITOR_PID:-?})，每 ${MONITOR_INTERVAL}s 采样"
        note "  曲线: $MEM_CSV      峰值摘要: $MONITOR_LOG"
        note "  若曲线一直是空的，跑: bash $MONITOR_SCRIPT --probe"
    fi
fi

cat > "$STOP_SCRIPT" <<EOF
#!/usr/bin/env bash
# 由 run_npu.sh 生成。用法: bash $STOP_SCRIPT
# 昇腾版没有 guard 接力，停就是停干净，不需要 --release。

echo "正在停止显存监控..."
pkill -f "npu_mem_monitor.sh" > /dev/null 2>&1

echo "正在查找 Anima 训练进程..."
TRAIN_PIDS=\$(pgrep -f "$TRAIN_SCRIPT.*$CONFIG_FILE")

if [ -z "\$TRAIN_PIDS" ]; then
    echo "未找到正在运行的 Anima 训练进程。"
else
    PIDS=\$(echo \$TRAIN_PIDS | tr '\\n' ' ')
    echo "将要终止 (PIDs): \$PIDS"
    # 先 TERM 给 checkpoint 收尾的机会，10 秒后还在就 KILL
    kill -TERM \$PIDS 2>/dev/null
    for _ in \$(seq 1 10); do
        pgrep -f "$TRAIN_SCRIPT.*$CONFIG_FILE" > /dev/null 2>&1 || break
        sleep 1
    done
    pkill -9 -f "$TRAIN_SCRIPT.*$CONFIG_FILE" > /dev/null 2>&1
    echo "已终止。"
fi

echo
echo "峰值显存摘要: tail -20 $MONITOR_LOG"
echo "⚠ 调试任务被回收后 /tmp/code 不保证还在 —— 权重记得及时下走:"
echo "   $(cd "$WORKDIR" 2>/dev/null && sed -n 's/^[[:space:]]*output_dir[[:space:]]*:[[:space:]]*//p' "$CONFIG_FILE" | head -n1 | tr -d '"')"
EOF
chmod +x "$STOP_SCRIPT"
note "停止脚本已生成: $STOP_SCRIPT"

echo
echo "=============================================================="
echo " 看日志:   tail -f $LOG_FILE"
echo " 看显存:   tail -f $MEM_CSV"
echo " 停训练:   bash $STOP_SCRIPT"
echo "--------------------------------------------------------------"
echo " ⚠ 调试任务上限 4 小时；/tmp/code 在任务被回收后不保证还在。"
echo "   首跑 max_steps=$(cd "$WORKDIR" && sed -n 's/^[[:space:]]*max_steps[[:space:]]*:[[:space:]]*//p' "$CONFIG_FILE" | head -n1)，"
echo "   跑完先确认 $OUT_DIR 下存出了 safetensors，再放长训练。"
echo "=============================================================="

if [ "$FOLLOW" = true ]; then
    echo
    echo "（Ctrl-C 只中断 tail，训练继续在后台跑）"
    echo
    tail -f "$LOG_FILE"
fi
