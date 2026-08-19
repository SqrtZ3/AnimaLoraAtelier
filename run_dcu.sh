#!/usr/bin/env bash
# ==============================================================================
#      Anima 训练启动脚本 · 海光 DCU（K100-AI）/ 国家超算互联网 scnet 版
#
# 这是 run.sh（CUDA）与 run_npu.sh（昇腾）的第三个对应物。相对那两个的**结构性**差异：
#
#   1. ★ 不需要设备重映射。DCU 的 DTK 是 ROCm/HIP 衍生栈，海光适配版 torch 本身就是
#      HIP 后端 —— torch.cuda.* 直接就是 DCU。所以没有 transfer_to_npu 那种补丁，
#      device_backend: dcu 只做「确认 + 守卫 + 环境」，不改设备语义。
#   2. ★ 支持 8 卡：--nproc N 走 torchrun，梯度在累积边界做一次 all-reduce
#      （只同步 LoRA 参数，几十 MB）。见 utils/dist_utils.py 的设计说明。
#   3. nvidia-smi → rocm-smi（DTK 侧）/ hy-smi（驱动侧）。
#   4. 体检里有一条**硬门槛**：torch 必须是 HIP 构建。DTK 镜像里一句
#      `pip install torch` 就会把适配版换成 PyPI 的 CUDA 构建，之后所有报错都会
#      指向奇怪的地方。这条不过直接拒绝启动。
#
# ⚠ 未在真机验证。本脚本按公开资料（DTK ≥24.04 支持 K100-AI/gfx928、镜像来自光源、
#   rocm-smi/hy-smi、/opt/dtk）写成，首次上机请先跑：
#       python tools/dcu_probe.py --json dcu_probe.json
#   探针会把「SDPA 走哪个后端、bf16 实测多少 TFLOPS、广播 matmul 反向对不对」这些
#   决定成败的问题变成实测数据，再照它末尾的「配置裁决」调 yaml。
#
# 用法:
#   bash run_dcu.sh                    单卡启动（后台，脱离终端）
#   bash run_dcu.sh --nproc 8          8 卡启动（torchrun）
#   bash run_dcu.sh --dry-run          只做体检，不启动
#   bash run_dcu.sh --probe            只跑能力探针（单卡）
#   bash run_dcu.sh --nproc 8 --probe  只跑 8 卡集合通信探针
#   bash run_dcu.sh --follow           启动后 tail -f 日志（Ctrl-C 只断 tail）
# ==============================================================================

set -u

# --- 用户配置区域（都支持环境变量覆盖）----------------------------------------
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
WORKDIR="${WORKDIR:-$REPO_DIR/AnimaLoraToolkit}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-anima_train.py}"
CONFIG_FILE="${CONFIG_FILE:-config/train_dcu_k100ai.yaml}"
OUT_DIR="${OUT_DIR:-$REPO_DIR/output}"
LOG_FILE="$OUT_DIR/anima_training_dcu.log"
MEM_CSV="$OUT_DIR/dcu_mem_timeline.csv"

# 卡数：1 = 单进程直跑；>1 = torchrun。scnet 上单卡/8卡两种规格各 100 卡时。
NPROC="${NPROC:-1}"
# torchrun 的 rendezvous 端口（同机多任务并存时改掉，否则撞端口）
MASTER_PORT="${MASTER_PORT:-29500}"

# allocator：DCU 上 expandable_segments 是否受支持随 DTK 版本变化，**默认留空**
# 不引入未知变量。dcu_probe.py 的「显存 API」项会打出当前生效值。
PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-}"

STRICT_ENV="${STRICT_ENV:-true}"
ENABLE_MEM_MONITOR="${ENABLE_MEM_MONITOR:-true}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-5}"

# --- 参数解析 -----------------------------------------------------------------
FOLLOW=false
DRY_RUN=false
PROBE_ONLY=false
while [ $# -gt 0 ]; do
    case "$1" in
        --follow)  FOLLOW=true ;;
        --dry-run) DRY_RUN=true ;;
        --probe)   PROBE_ONLY=true ;;
        --nproc)   shift; NPROC="${1:-1}" ;;
        --nproc=*) NPROC="${1#*=}" ;;
        -h|--help) sed -n '1,32p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "未知参数: $1（-h 看用法）"; exit 1 ;;
    esac
    shift
done

mkdir -p "$OUT_DIR"
cd "$WORKDIR" || { echo "✗ 进不去工作目录: $WORKDIR"; exit 1; }

echo "=============================================================="
echo " Anima 训练 · 海光 DCU"
echo " 工作目录 : $WORKDIR"
echo " 配置     : $CONFIG_FILE"
echo " 卡数     : $NPROC"
echo "=============================================================="

# --- 体检 0：DTK 环境 ---------------------------------------------------------
# 【实测 · scnet BW 64GB 容器】DTK 的 env.sh **不会**被 /etc/profile.d 自动 source，
# 交互 shell 和 ssh 非登录 shell 里 LD_LIBRARY_PATH 都是空的，直接 `import torch`
# 报 `libgalaxyhip.so.5: cannot open shared object file`。所以这里无条件补一次。
# 已经 source 过是幂等的；DTK_ROOT 可用环境变量覆盖（多版本共存时指到具体版本目录）。
DTK_ROOT="${DTK_ROOT:-/opt/dtk}"
if [ -z "${ROCM_PATH:-}" ] && [ -f "$DTK_ROOT/env.sh" ]; then
    # env.sh 里大量 `export X=...:$X` 的写法，X 未定义时在 `set -u` 下是致命错误，
    # 会让整个脚本在这里静默退出（RC=1，连体检都没打完）。所以这一段临时关掉 -u。
    set +u
    # shellcheck disable=SC1091
    . "$DTK_ROOT/env.sh" >/dev/null 2>&1
    set -u
    echo "  · 已 source $DTK_ROOT/env.sh（ROCM_PATH=${ROCM_PATH:-未设置}）"
fi

FAIL=0
warn() { echo "  ⚠ $*"; }
bad()  { echo "  ✗ $*"; FAIL=1; }
good() { echo "  ✓ $*"; }

# --- 体检 1：设备可见 ---------------------------------------------------------
echo "[1/6] 设备"
if command -v rocm-smi >/dev/null 2>&1; then
    rocm-smi 2>/dev/null | head -12
    good "rocm-smi 可用"
elif command -v hy-smi >/dev/null 2>&1; then
    hy-smi 2>/dev/null | head -12
    good "hy-smi 可用（rocm-smi 不在 PATH，DTK 环境变量可能没 source）"
else
    bad "rocm-smi / hy-smi 都找不到。DTK 环境没生效？试 source /opt/dtk/env.sh"
fi

# --- 体检 2：torch 必须是 HIP 构建（★ 硬门槛）---------------------------------
echo "[2/6] PyTorch"
TORCH_INFO=$("$PYTHON_BIN" - <<'PY' 2>&1
try:
    import torch
    print(f"OK|{torch.__version__}|{torch.version.hip}|{torch.version.cuda}|"
          f"{torch.cuda.is_available()}|{torch.cuda.device_count()}")
except Exception as e:
    print(f"ERR|{type(e).__name__}: {e}")
PY
)
case "$TORCH_INFO" in
    OK\|*)
        IFS='|' read -r _ TV THIP TCUDA TAVAIL TCOUNT <<< "$TORCH_INFO"
        echo "  torch=$TV hip=$THIP cuda=$TCUDA available=$TAVAIL devices=$TCOUNT"
        if [ "$THIP" = "None" ] || [ -z "$THIP" ]; then
            bad "torch 不是 HIP 构建（torch.version.hip=None）。"
            echo "     多半是在 DTK 镜像里 pip install torch 覆盖了适配版。"
            echo "     修复：重建容器，或从光源(sourcefind)重装 torch-*+das*.dtk* 轮子。"
        else
            good "HIP 构建"
        fi
        [ "$TAVAIL" = "True" ] || bad "torch.cuda.is_available()=False，容器看不到 DCU"
        if [ "$NPROC" -gt 1 ] && [ "$TCOUNT" != "None" ] && [ "$TCOUNT" -lt "$NPROC" ] 2>/dev/null; then
            bad "要 $NPROC 卡但只看到 $TCOUNT 张。检查 HIP_VISIBLE_DEVICES。"
        fi
        ;;
    *) bad "import torch 失败: $TORCH_INFO" ;;
esac

# --- 体检 3：环境污染 ---------------------------------------------------------
echo "[3/6] 依赖污染"
POLLUTED=$("$PYTHON_BIN" -m pip list 2>/dev/null | grep -iE "^(nvidia-|cuda-)" | tr '\n' ' ')
if [ -n "$POLLUTED" ]; then
    warn "pip 里有 NVIDIA 相关包：$POLLUTED"
    echo "     它们通常是 PyPI torch 拖进来的，说明适配版可能已被覆盖。"
fi
TV_INFO=$("$PYTHON_BIN" -c "import torchvision;print(torchvision.__version__)" 2>/dev/null)
if [ -n "$TV_INFO" ]; then
    case "$TV_INFO" in
        *das*|*dtk*) good "torchvision=$TV_INFO（das 适配版）" ;;
        *) warn "torchvision=$TV_INFO 不像 das 适配版 —— PyPI 版会连带把 torch 换成 CUDA 构建" ;;
    esac
fi

# --- 体检 4：配置文件与关键字段 -----------------------------------------------
echo "[4/6] 配置"
if [ ! -f "$CONFIG_FILE" ]; then
    bad "配置文件不存在: $WORKDIR/$CONFIG_FILE"
else
    good "配置文件存在"
    cfg_get() {
        "$PYTHON_BIN" - "$CONFIG_FILE" "$1" <<'PY' 2>/dev/null
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8")) or {}
v = cfg.get(sys.argv[2], "")
print("" if v is None else v)
PY
    }
    DEV_BACKEND="$(cfg_get device_backend)"
    ATTN_BACKEND="$(cfg_get navit_attn_backend)"
    NAVIT="$(cfg_get navit_packing)"
    BASE_QUANT="$(cfg_get base_quant)"
    EFF_BS="$(cfg_get effective_batch_size)"
    DPO="$(cfg_get dpo_enabled)"
    LORA_ONE="$(cfg_get lora_one_init_steps)"

    case "$DEV_BACKEND" in
        dcu|hygon|rocm) good "device_backend=$DEV_BACKEND" ;;
        *) bad "device_backend=$DEV_BACKEND（应为 dcu）。也可用环境变量 ANIMA_DCU=1。" ;;
    esac

    case "$NAVIT" in
        True|true|1)
            case "$ATTN_BACKEND" in
                sdpa_seg) good "navit_attn_backend=sdpa_seg（DCU 上唯一已接线的打包后端）" ;;
                xformers) bad "navit_attn_backend=xformers —— DCU 无 xformers 轮子，请改 sdpa_seg" ;;
                npu_tnd)  bad "navit_attn_backend=npu_tnd 是昇腾专有算子，请改 sdpa_seg" ;;
                *) warn "navit_attn_backend=$ATTN_BACKEND（未识别，DCU 上应为 sdpa_seg）" ;;
            esac
            ;;
        *) good "navit_packing 未开（走 ARB 稠密路径）" ;;
    esac

    case "$BASE_QUANT" in
        ""|none|None) good "base_quant=none" ;;
        *) bad "base_quant=$BASE_QUANT —— K100-AI 按代际推断无 FP8 张量核。"
           echo "     先看 dcu_probe.py 的「FP8」项；确实可用再 ANIMA_DCU_ALLOW_FP8=1 放行。" ;;
    esac

    if [ "$NPROC" -gt 1 ]; then
        case "$EFF_BS" in
            ""|0) good "effective_batch_size 未用（多卡要求走 grad_accum）" ;;
            *) bad "多卡下 effective_batch_size=$EFF_BS 会导致各 rank 累积边界错位 → 死锁。"
               echo "     改用 grad_accum（边界只由 batch_idx 决定，跨 rank 恒等）。" ;;
        esac
        case "$DPO" in True|true|1) bad "多卡首版不支持 dpo_enabled（输家池会跨 rank 分叉）" ;; esac
        case "$LORA_ONE" in ""|0) ;; *) bad "多卡首版不支持 lora_one_init_steps=$LORA_ONE（预热未接梯度同步）" ;; esac
    fi
fi

# --- 体检 5：探针结果 ---------------------------------------------------------
echo "[5/6] 能力探针"
if [ -f "$OUT_DIR/dcu_probe.json" ]; then
    good "已有探针结果: $OUT_DIR/dcu_probe.json"
    "$PYTHON_BIN" - "$OUT_DIR/dcu_probe.json" <<'PY'
import json, sys
rs = json.load(open(sys.argv[1], encoding="utf-8"))
fails = [r for r in rs if r["status"] == "FAIL"]
print(f"     探针: OK={sum(1 for r in rs if r['status']=='OK')} FAIL={len(fails)}")
for r in fails[:8]:
    print(f"     ✗ {r['name']}")
PY
else
    warn "还没跑过能力探针。强烈建议先跑（2 分钟，能省一次白烧的卡时）："
    echo "     bash run_dcu.sh --probe"
fi

# --- 体检 6：重复启动 ---------------------------------------------------------
echo "[6/6] 进程"
RUNNING=$(pgrep -f "$TRAIN_SCRIPT.*$CONFIG_FILE" 2>/dev/null | tr '\n' ' ')
if [ -n "$RUNNING" ]; then
    bad "已有训练在跑（PID: $RUNNING）。先停掉或换 CONFIG_FILE。"
fi

echo "--------------------------------------------------------------"
if [ "$FAIL" -ne 0 ]; then
    if [ "$STRICT_ENV" = "true" ]; then
        echo "✗ 体检未通过，拒绝启动（STRICT_ENV=false 可强行跳过）"
        exit 1
    fi
    warn "体检未通过，但 STRICT_ENV=false —— 继续启动，出问题请先回头看上面的 ✗"
fi

# --- 探针模式 -----------------------------------------------------------------
if [ "$PROBE_ONLY" = true ]; then
    if [ "$NPROC" -gt 1 ]; then
        echo "→ 跑 $NPROC 卡集合通信探针"
        torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" \
                 tools/dcu_probe.py --dist --json "$OUT_DIR/dcu_probe_dist.json"
    else
        echo "→ 跑单卡能力探针"
        "$PYTHON_BIN" tools/dcu_probe.py --json "$OUT_DIR/dcu_probe.json"
    fi
    exit $?
fi

if [ "$DRY_RUN" = true ]; then
    echo "✓ 体检通过（--dry-run，不启动训练）"
    exit 0
fi

# --- 启动 ---------------------------------------------------------------------
ENV_ARGS=(PYTHONUNBUFFERED=1 ANIMA_DCU=1)
[ -n "$PYTORCH_HIP_ALLOC_CONF" ] && ENV_ARGS+=("PYTORCH_HIP_ALLOC_CONF=$PYTORCH_HIP_ALLOC_CONF")

if [ "$NPROC" -gt 1 ]; then
    # torchrun 会给每个进程注入 RANK/WORLD_SIZE/LOCAL_RANK，anima_train.py 据此
    # 自动进入多卡分支（没有这些变量时全程 no-op，单卡逐字节不变）。
    CMD=(torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT"
         "$TRAIN_SCRIPT" --config "$CONFIG_FILE" --no-browser)
else
    CMD=("$PYTHON_BIN" "$TRAIN_SCRIPT" --config "$CONFIG_FILE" --no-browser)
fi

echo "启动: ${CMD[*]}"
echo "日志: $LOG_FILE"

# setsid：让训练脱离当前终端会话。云平台的 Notebook/终端有空闲回收策略，
# 同会话的子进程会被一起带走 —— 这在启智上是实测过的（见 memory
# openi-platform-workflow-facts），scnet 同类平台按同样方式防一手。
setsid nohup env "${ENV_ARGS[@]}" "${CMD[@]}" > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!
echo "训练 PID: $TRAIN_PID"

# --- 显存采样 -----------------------------------------------------------------
if [ "$ENABLE_MEM_MONITOR" = "true" ] && command -v rocm-smi >/dev/null 2>&1; then
    echo "timestamp,card,vram_used_mb,util_pct" > "$MEM_CSV"
    setsid nohup bash -c '
        while true; do
            ts=$(date +%s)
            rocm-smi --showmeminfo vram --showuse --csv 2>/dev/null \
              | awk -F, -v ts="$ts" "NR>1 && NF>2 {print ts\",\"\$1\",\"\$2\",\"\$3}" \
              >> "'"$MEM_CSV"'"
            sleep '"$MONITOR_INTERVAL"'
        done' >/dev/null 2>&1 &
    echo "显存曲线: $MEM_CSV（rocm-smi 每 ${MONITOR_INTERVAL}s 采样）"
fi

sleep 3
if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    echo "✗ 训练进程启动后立刻退出，日志尾部："
    tail -30 "$LOG_FILE"
    exit 1
fi
echo "✓ 已启动。停止: pkill -f \"$TRAIN_SCRIPT.*$CONFIG_FILE\""

if [ "$FOLLOW" = true ]; then
    echo "--- tail -f（Ctrl-C 只断 tail，训练继续）---"
    tail -f "$LOG_FILE"
fi
