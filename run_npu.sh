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
#
# ★★ CANN 9.1.0 / torch 2.11 环境改造（2026-08-16）★★
#
#   5. ★ 脚本自己 source /root/act9.sh。这是本次最关键的改动。
#      旧镜像在 /etc/profile 里自动 source 8.2 的 set_env.sh，所以脚本什么都不做
#      就能拿到 CANN 环境。新环境把那行停用了，改由 act9.sh 统一激活
#      （CANN 9.1.0 set_env + nnal/atb set_env + venv）。而 `bash run_npu.sh`
#      起的是**非交互非登录 shell**，~/.bashrc 和 /etc/profile.d/ 一个都不读——
#      不在脚本里显式 source，训练会以「找不到 torch_npu」或「acl 初始化失败」
#      的形式挂掉。
#   6. 解释器改为 venv 里的 python（/root/venv-cann9/bin/python），不再是
#      /usr/local/python3.11.13/bin/python。后者是系统 python，配的是旧 8.2 环境，
#      且 numpy 版本与新 venv 不兼容。act9.sh 激活后 `python` 已指向 venv，
#      PYTHON_BIN 仍写绝对路径是为了体检阶段能给出准确报错。
#   7. 体检新增三项硬门槛：torch 必须是 2.11.x、pip 里不能有 nvidia/cuda/
#      torchvision 残留、PATH 里不能有 8.2 的 ascend-toolkit。任何一项不过就拒绝
#      启动——这三样都会以「训练跑起来了但数值不对」的方式坑人，比直接崩更难查。
#   8. NPU_ID 改为自动探测（npu-smi 的物理卡号在不同任务里会变，上个任务是 7）。
#   9. conv.allow_hf32：昇腾默认为 True，会截尾数，与 GPU 基线不可比。
#      matmul 侧在 CANN 9.1.0 上默认已是 False（实测 fp32 relerr 2e-07），
#      conv 侧仍需手动关。脚本只做检查与告警，实际关闭要在训练脚本里写：
#          import torch_npu; torch_npu.npu.conv.allow_hf32 = False
#
# 用法:
#   bash run_npu.sh              启动训练（后台，脱离终端）
#   bash run_npu.sh --follow     启动后顺便 tail -f 日志（Ctrl-C 只断 tail，不停训练）
#   bash run_npu.sh --dry-run    只做体检，不启动
# ==============================================================================

set -u

# --- 环境激活（★ 必须在最前面，其余一切都依赖它）-------------------------------
# 非交互非登录 shell 不读 ~/.bashrc 和 /etc/profile.d/，所以这里显式 source。
# act9.sh 是幂等的（基线快照法），重复 source 不会让 PATH/LD_LIBRARY_PATH 增长。
#
# ★ set +u 是必须的：nnal 的 atb/set_env.sh 里引用了 $ZSH_VERSION 却没给默认值，
#   在 set -u 下会以 "ZSH_VERSION: unbound variable" 直接中断。这是昇腾脚本自身的
#   毛病，只能在调用侧让路。source 完立刻恢复 set -u。
ACT_SCRIPT="${ACT_SCRIPT:-/root/act9.sh}"
if [ -f "$ACT_SCRIPT" ]; then
    set +u
    # shellcheck disable=SC1090
    . "$ACT_SCRIPT"
    set -u
else
    echo "✗ 找不到环境激活脚本: $ACT_SCRIPT"
    echo "  这个镜像可能不是 CANN 9.1.0 版。检查: ls -d /usr/local/Ascend9 /root/venv-cann9"
    echo "  若确实在用旧 8.2 镜像，设 ACT_SCRIPT=/usr/local/Ascend/ascend-toolkit/set_env.sh"
    echo "  并把 PYTHON_BIN 改回 /usr/local/python3.11.13/bin/python。"
    exit 1
fi

# --- 用户配置区域 -------------------------------------------------------------
# 下面这些都支持环境变量覆盖，换路径不用改脚本，例如：
#   CONFIG_FILE=config/train_npu_other.yaml bash run_npu.sh

# 仓库在机器上的位置（JupyterLab 侧栏根目录是 /tmp/code）
REPO_DIR="${REPO_DIR:-/tmp/code/anima-lora-train}"

# 训练进程的工作目录。配置文件里的相对路径都以它为基准；
# train_npu_c12.yaml 的模型/数据/输出路径全是绝对路径，所以这里选哪个都能跑。
WORKDIR="${WORKDIR:-$REPO_DIR/AnimaLoraToolkit}"

# 解释器（★ CANN 9 环境：venv 里的 python，不是系统 python）
PYTHON_BIN="${PYTHON_BIN:-/root/venv-cann9/bin/python}"

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

# --- 环境门槛（★ 新增。设 STRICT_ENV=false 可降级为告警，但不建议）-------------
STRICT_ENV="${STRICT_ENV:-true}"
EXPECT_TORCH="${EXPECT_TORCH:-2.11}"

# --- 监控配置 ---
ENABLE_MEM_MONITOR="${ENABLE_MEM_MONITOR:-true}"
# NPU_ID 留空则自动探测 npu-smi 的物理卡号（不同任务分到的卡号会变）
NPU_ID="${NPU_ID:-}"
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
# 严格模式下按 bad 处理，否则只告警
strict(){ if [ "$STRICT_ENV" = true ]; then bad "$*"; else warn "$*（STRICT_ENV=false，放行）"; fi; }

echo "=============================================================="
echo " Anima 训练启动 · 昇腾 910B / 启智 OpenI · CANN 9.1.0 / torch 2.11"
echo "=============================================================="
echo
echo "[1/6] 环境体检"

note "已激活: $ACT_SCRIPT"
[ -d "$WORKDIR" ]     || bad "工作目录不存在: $WORKDIR"
[ -x "$PYTHON_BIN" ]  || bad "解释器不存在或不可执行: $PYTHON_BIN  （试 ls -d /root/venv-cann9/bin/python）"
[ -f "$WORKDIR/$TRAIN_SCRIPT" ] || bad "训练脚本未找到: $WORKDIR/$TRAIN_SCRIPT"
[ -f "$WORKDIR/$CONFIG_FILE" ]  || bad "配置文件未找到: $WORKDIR/$CONFIG_FILE"

if [ "$FAIL" = 0 ]; then
    note "解释器: $("$PYTHON_BIN" -V 2>&1)  ($PYTHON_BIN)"
    if "$PYTHON_BIN" -c "import torch, torch_npu" 2>/dev/null; then
        TORCH_VER=$("$PYTHON_BIN" -c 'import torch;print(torch.__version__)' 2>/dev/null)
        TNPU_VER=$("$PYTHON_BIN" -c 'import torch,torch_npu;print(torch_npu.__version__)' 2>/dev/null)
        note "torch $TORCH_VER / torch_npu $TNPU_VER"
        case "$TORCH_VER" in
            "$EXPECT_TORCH"*) : ;;
            *) strict "torch 版本是 $TORCH_VER，期望 ${EXPECT_TORCH}.x —— GPU 基线是在 ${EXPECT_TORCH} 上训的，版本不同则 loss 曲线不可比" ;;
        esac
        NPU_CNT=$("$PYTHON_BIN" -c 'import torch,torch_npu;print(torch.npu.device_count())' 2>/dev/null)
        [ "${NPU_CNT:-0}" -ge 1 ] 2>/dev/null \
            && note "可见 NPU 数量: $NPU_CNT" \
            || bad "torch 看不到 NPU（device_count=${NPU_CNT:-?}）"
    else
        bad "import torch / torch_npu 失败 —— 先跑 npu-check 看环境哪一层断了"
    fi
fi

echo
echo "[2/6] 环境纯净度（这三项都会以「跑得起来但数值不对」的方式坑人）"

# (a) CUDA / torchvision 残留。aarch64 上 PyPI 的 torch 捆 CUDA 13，且不存在配
#     torch 2.11 的 CPU 版 torchvision——一旦有人 pip install torchvision，
#     整个 venv 的 torch 会被换成 CUDA 版，NPU 直接不可用。
POLLUTED=$("$PYTHON_BIN" -m pip list 2>/dev/null | grep -iE "^(nvidia-|cuda-|torchvision)" | tr '\n' ' ')
if [ -n "$POLLUTED" ]; then
    strict "venv 里有 CUDA/torchvision 残留: $POLLUTED"
    note "  修复: pip uninstall -y 上述包，然后重装 torch:"
    note "    pip install torch==2.11.0 --no-index -f https://mirrors.aliyun.com/pytorch-wheels/cpu/ --no-deps"
else
    note "无 CUDA / torchvision 残留"
fi

# (b) 旧 8.2 的 ccec_compiler 若留在 PATH 上，算子在线编译可能走错编译器。
#     判据必须写全路径：/usr/local/Ascend9/ascend-toolkit/ 也存在，
#     用 "ascend-toolkit" 子串匹配会误报。
OLD82=$(echo "$PATH" | tr ':' '\n' | grep -c "^/usr/local/Ascend/ascend-toolkit")
if [ "$OLD82" -gt 0 ]; then
    strict "PATH 里还有 $OLD82 条 8.2 的路径 —— 检查 /etc/profile 与 ~/.bashrc 是否还在 source 旧 set_env.sh"
else
    note "PATH 里无 8.2 残留"
fi

# (c) conv 的 HF32。昇腾默认 True（截尾数），与 GPU 基线不可比。
#     matmul 侧 CANN 9.1.0 默认已是 False，不用管。
HF32=$("$PYTHON_BIN" -c 'import torch,torch_npu
m=getattr(torch_npu.npu,"matmul",None); c=getattr(torch_npu.npu,"conv",None)
print(getattr(m,"allow_hf32","NA"), getattr(c,"allow_hf32","NA"))' 2>/dev/null)
note "allow_hf32  matmul/conv = ${HF32:-探测失败}"
if grep -qs "allow_hf32" "$WORKDIR/$TRAIN_SCRIPT"; then
    note "训练脚本里有 allow_hf32 设置"
else
    warn "训练脚本里没找到 allow_hf32 —— 要与 GPU 基线严格可比，在 $TRAIN_SCRIPT 开头加:"
    warn "    import torch_npu; torch_npu.npu.conv.allow_hf32 = False"
fi

echo
echo "[3/6] 配置文件路径检查（从 yaml 里读，逐条 stat）"

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

# 模型权重多为 LFS 对象。若 clone 时用了 GIT_LFS_SKIP_SMUDGE=1，这里拿到的
# 会是几百字节的指针文件——大小检查能当场抓出来，比训练跑到一半报格式错好。
for _lbl in transformer_path vae_path; do
    _p="$(cfg_get $_lbl)"
    if [ -f "$_p" ]; then
        _sz=$(stat -c %s "$_p" 2>/dev/null || echo 0)
        if [ "$_sz" -lt 1048576 ]; then
            bad "$_lbl 只有 ${_sz} 字节，像是未取回的 LFS 指针文件: $_p"
            note "  取回: cd \$REPO && git lfs pull --include=\"$_p\""
        fi
    fi
done

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
echo "[4/6] 磁盘与已有进程"
mkdir -p "$OUT_DIR"
note "输出目录: $OUT_DIR（剩余空间 $(df -h "$OUT_DIR" 2>/dev/null | awk 'NR==2{print $4}')）"

# npu-smi 的物理卡号自动探测（不同任务分到的卡号不同，上个任务是 7 号）
if command -v npu-smi >/dev/null 2>&1; then
    if [ -z "$NPU_ID" ]; then
        NPU_ID=$(npu-smi info -l 2>/dev/null | sed -n 's/^[[:space:]]*NPU ID[[:space:]]*:[[:space:]]*//p' | head -n 1)
        NPU_ID="${NPU_ID:-0}"
        note "npu-smi 卡号（自动探测）: $NPU_ID"
    else
        note "npu-smi 卡号（手动指定）: $NPU_ID"
    fi
else
    NPU_ID="${NPU_ID:-0}"
    warn "找不到 npu-smi，显存曲线会采不到（训练本身不受影响）"
fi

RUNNING=$(pgrep -f "$TRAIN_SCRIPT.*$CONFIG_FILE" 2>/dev/null | tr '\n' ' ')
if [ -n "$RUNNING" ]; then
    bad "已经有训练进程在跑 (PIDs: $RUNNING)。先 bash $STOP_SCRIPT，或确认是不是重复启动。"
fi

if [ "$FAIL" != 0 ]; then
    echo
    echo "体检未通过，已中止启动。修完上面的 ✗ 再来。"
    echo "（若确信某项可以放行: STRICT_ENV=false bash run_npu.sh）"
    exit 1
fi
echo "  体检通过。"

if [ "$DRY_RUN" = true ]; then
    echo
    echo "[--dry-run] 到此为止，未启动训练。"
    exit 0
fi

echo
echo "[5/6] 启动训练（setsid 脱离终端，cull 杀不到）"

# PYTHONUNBUFFERED=1 + python -u：否则输出被重定向到文件后走块缓冲，
# tail -f 要卡好几分钟才出新行。
ENV_ARGS=(PYTHONUNBUFFERED=1)
if [ -n "$PYTORCH_NPU_ALLOC_CONF" ]; then
    ENV_ARGS+=(PYTORCH_NPU_ALLOC_CONF="$PYTORCH_NPU_ALLOC_CONF")
    note "PYTORCH_NPU_ALLOC_CONF=$PYTORCH_NPU_ALLOC_CONF （★非默认，记得这次是 A/B）"
else
    note "PYTORCH_NPU_ALLOC_CONF=<未设，用 torch_npu 默认>"
fi

# 把本次运行的环境指纹留在日志开头，事后对比两次跑的差异时能少走弯路。
mkdir -p "$OUT_DIR"
{
    echo "===== 环境指纹 $(date '+%F %T') ====="
    echo "act:        $ACT_SCRIPT"
    echo "python:     $PYTHON_BIN"
    echo "torch:      ${TORCH_VER:-?} / torch_npu ${TNPU_VER:-?}"
    echo "CANN:       ${ASCEND_TOOLKIT_HOME:-<未设>}"
    echo "allow_hf32: ${HF32:-?}  (matmul conv)"
    echo "config:     $WORKDIR/$CONFIG_FILE"
    echo "npu-smi id: $NPU_ID"
    echo "=========================================="
} > "$LOG_FILE"

cd "$WORKDIR" || exit 1
setsid nohup env "${ENV_ARGS[@]}" \
    "$PYTHON_BIN" -u "$TRAIN_SCRIPT" --config="$CONFIG_FILE" \
    >> "$LOG_FILE" 2>&1 < /dev/null &

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
echo "[6/6] 显存监控与停止脚本"

MONITOR_PID=""
if [ "$ENABLE_MEM_MONITOR" = true ]; then
    if [ ! -f "$MONITOR_SCRIPT" ]; then
        warn "未找到 $MONITOR_SCRIPT，跳过显存监控。"
    else
        setsid nohup bash "$MONITOR_SCRIPT" "$NPU_ID" "$MONITOR_INTERVAL" \
            "$MEM_CSV" "$TRAIN_PID" "$LOG_FILE" > "$MONITOR_LOG" 2>&1 < /dev/null &
        sleep 1
        MONITOR_PID=$(pgrep -f "npu_mem_monitor.sh $NPU_ID" 2>/dev/null | head -n 1)
        note "显存监控已启动 (PID: ${MONITOR_PID:-?})，每 ${MONITOR_INTERVAL}s 采样 卡号 $NPU_ID"
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
