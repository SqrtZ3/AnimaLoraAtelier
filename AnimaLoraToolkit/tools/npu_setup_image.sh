#!/usr/bin/env bash
# 昇腾 NPU 调试任务：环境装配脚本（为「提交镜像」准备）
#
# 用法（在 OpenI 调试任务的终端里）：
#     bash tools/npu_setup_image.sh            # 装必需依赖
#     bash tools/npu_setup_image.sh --check    # 只体检，不装任何东西
#     bash tools/npu_setup_image.sh --with-perceptual --with-jxl
#
# 设计要点：
#   1. **绝不碰 torch / torch_npu**。生成一份 pip constraints 把已装的
#      torch / torch_npu / numpy 版本钉死——否则任何一个依赖（尤其 torchvision）
#      都可能从 PyPI 拉一个 CUDA 版 torch 覆盖掉 torch_npu，环境当场报废。
#   2. 只装**真实运行路径**用得到的包。requirements.txt 里的 diffusers /
#      accelerate / peft / lycoris-lora / torchvision / pytorch-fid 经 AST 扫描
#      确认未被 import，一律不装（镜像有 20GB/40GB 上限）。
#   3. 幂等：可重复执行。
#   4. 会检查 site-packages 是否落在「不随镜像提交」的目录里——NPU 环境下
#      /home/ma-user/work 不入镜像，装在那儿的包提交完就没了。
#
# 装完之后：**不要停止调试任务**，回任务列表 →「更多」→「提交镜像」。

set -uo pipefail

CHECK_ONLY=0
WITH_PERCEPTUAL=0
WITH_JXL=0
WITH_EXTRA_OPT=0
for arg in "$@"; do
  case "$arg" in
    --check)            CHECK_ONLY=1 ;;
    --with-perceptual)  WITH_PERCEPTUAL=1 ;;
    --with-jxl)         WITH_JXL=1 ;;
    --with-extra-opt)   WITH_EXTRA_OPT=1 ;;
    *) echo "未知参数: $arg"; exit 2 ;;
  esac
done

PYBIN="${PYBIN:-python3}"
PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

hr() { printf '%.0s─' {1..76}; echo; }
say() { echo -e "\n\033[1m>> $*\033[0m"; }

# ── 0. CANN 环境 ──────────────────────────────────────────────────────────────
say "0. CANN 环境"
for f in /usr/local/Ascend/ascend-toolkit/set_env.sh \
         /usr/local/Ascend/nnae/set_env.sh \
         "$HOME/Ascend/ascend-toolkit/set_env.sh"; do
  if [ -f "$f" ]; then
    # shellcheck disable=SC1090
    source "$f" && echo "   已 source $f"
    break
  fi
done
if command -v npu-smi >/dev/null 2>&1; then
  npu-smi info 2>&1 | head -n 12
else
  echo "   ⚠ 找不到 npu-smi —— 确认这是 NPU 容器"
fi

# ── 1. 解释器与安装位置 ───────────────────────────────────────────────────────
say "1. 解释器与 site-packages 位置"
command -v "$PYBIN" >/dev/null 2>&1 || { echo "找不到 $PYBIN"; exit 1; }
"$PYBIN" - <<'PY'
import sys, site
print(f"   python  : {sys.version.split()[0]}  @ {sys.executable}")
sp = site.getsitepackages()
print(f"   site-pkg: {sp}")
bad = [p for p in sp if p.startswith("/home/ma-user/work") or p.startswith("/tmp/code") or p.startswith("/tmp/dataset")]
if bad:
    print("   ⚠⚠ 警告：site-packages 位于不随镜像提交的目录", bad)
    print("        （NPU 环境下 /home/ma-user/work、以及 /tmp/code、/tmp/dataset 都不入镜像）")
    print("        装在这里的包提交镜像后会全部丢失。请改用系统 python 或 /opt 下的 conda 环境。")
else:
    print("   ✓ site-packages 在会被打包进镜像的路径下")
PY

if command -v conda >/dev/null 2>&1; then
  echo "   检测到 conda，现有环境："
  conda env list 2>/dev/null | sed 's/^/     /'
  echo "   （若当前环境没有 torch_npu，先 conda activate 到带 torch-npu 的那个再跑本脚本）"
fi

# ── 2. torch / torch_npu 现状 ─────────────────────────────────────────────────
say "2. torch / torch_npu 现状（本脚本绝不修改它们）"
TORCH_VER="$("$PYBIN" -c 'import torch;print(torch.__version__)' 2>/dev/null)"
if [ -z "$TORCH_VER" ]; then
  echo "   ✗ 导不进 torch。这个镜像不适合，换 torch-npu-cann8-debug。"; exit 1
fi
TORCH_NPU_VER="$("$PYBIN" -c 'import torch_npu;print(torch_npu.__version__)' 2>/dev/null)"
NUMPY_VER="$("$PYBIN" -c 'import numpy;print(numpy.__version__)' 2>/dev/null)"
echo "   torch      = $TORCH_VER"
echo "   torch_npu  = ${TORCH_NPU_VER:-<未安装 / 导入失败>}"
echo "   numpy      = ${NUMPY_VER:-<未安装>}"
if [ -z "$TORCH_NPU_VER" ]; then
  echo "   ✗ torch_npu 不可用 —— 后面所有 NPU 训练都无从谈起，先解决这个。"
fi

# ── 3. constraints：钉死 torch 系，防止被依赖解析悄悄换掉 ─────────────────────
say "3. 生成 pip constraints（防止 torchvision 等把 torch 换成 CUDA 版）"
CONS="/tmp/anima_npu_constraints.txt"
{
  echo "torch==$TORCH_VER"
  [ -n "$TORCH_NPU_VER" ] && echo "torch-npu==$TORCH_NPU_VER"
  # torch 2.1.x 是按 numpy 1.x ABI 编译的，numpy 2 会在运行期报 _ARRAY_API 错误
  echo "numpy<2"
} > "$CONS"
cat "$CONS" | sed 's/^/   /'

PIP="$PYBIN -m pip"
PIP_ARGS="--index-url $PIP_INDEX --constraint $CONS"

# ── 4. 依赖清单 ───────────────────────────────────────────────────────────────
# 必需：由 AST 扫描 anima_train.py + trainer/ + utils/ + models/ 的真实 import 得出。
# transformers>=4.51 是硬下限：trainer/models.py:259 用 AutoModelForCausalLM 加载
# Qwen3-0.6B，更早的版本不认识 Qwen3 架构。
REQUIRED=(
  "numpy<2"
  "Pillow"
  "safetensors"
  "transformers>=4.51,<5"
  "PyYAML"
  "rich"
  "einops"
)
# 可选组
PERCEPTUAL=( "lpips" )          # trainer/aux_losses.py，仅 perceptual aux loss 用；会拉 torchvision（危险，已被 constraints 挡住）
JXL=( "pillow-jxl-plugin" )     # 仅当数据集是 JXL
EXTRA_OPT=( "prodigy-plus-schedule-free" )  # 仅当用 prodigyplus 优化器

# 明确**不装**（并说明原因），避免下次有人「顺手补全 requirements.txt」
say "4. 明确不安装的包（不是遗漏）"
cat <<'EOF'
   xformers        —— 昇腾无此包；NaViT 打包路径因此不可用（配置里必须关）
   bitsandbytes    —— 无昇腾后端；utils/optimizer_utils.py:76 是 try/except 可选导入
   triton          —— 昇腾无
   flash-attn      —— 昇腾无
   torchvision     —— 运行路径未 import；且极易连带把 torch 换成 CUDA 版
   diffusers / accelerate / peft / lycoris-lora / pytorch-fid
                   —— AST 扫描确认运行路径未 import，纯占镜像体积
EOF

if [ "$CHECK_ONLY" = "1" ]; then
  say "--check 模式：到此为止，未安装任何东西"
  exit 0
fi

# ── 5. 安装 ───────────────────────────────────────────────────────────────────
say "5. 安装必需依赖"
$PIP config set global.index-url "$PIP_INDEX" >/dev/null 2>&1 || true
# shellcheck disable=SC2086
$PIP install $PIP_ARGS "${REQUIRED[@]}" || { echo "✗ 必需依赖安装失败"; exit 1; }

if [ "$WITH_PERCEPTUAL" = "1" ]; then
  say "5b. 安装 perceptual 组（会拉 torchvision —— constraints 已钉死 torch）"
  # shellcheck disable=SC2086
  $PIP install $PIP_ARGS "${PERCEPTUAL[@]}" || echo "   ⚠ perceptual 组安装失败（非致命，aux loss 关掉即可）"
fi
if [ "$WITH_JXL" = "1" ]; then
  say "5c. 安装 JXL 组"
  # shellcheck disable=SC2086
  $PIP install $PIP_ARGS "${JXL[@]}" || echo "   ⚠ JXL 组安装失败（非致命）"
fi
if [ "$WITH_EXTRA_OPT" = "1" ]; then
  say "5d. 安装第三方优化器组"
  # shellcheck disable=SC2086
  $PIP install $PIP_ARGS "${EXTRA_OPT[@]}" || echo "   ⚠ 优化器组安装失败（非致命，用内置 adamw）"
fi

# ── 6. 安装后校验：torch 有没有被换掉 ────────────────────────────────────────
say "6. 安装后校验"
NEW_TORCH="$("$PYBIN" -c 'import torch;print(torch.__version__)' 2>/dev/null)"
if [ "$NEW_TORCH" != "$TORCH_VER" ]; then
  echo "   ✗✗ torch 被改动了：$TORCH_VER -> $NEW_TORCH"
  echo "        环境很可能已经损坏。不要提交镜像，重开任务从头来。"
  exit 1
fi
echo "   ✓ torch 未被改动（$NEW_TORCH）"
"$PYBIN" - <<'PY'
mods = ["torch", "torch_npu", "transformers", "safetensors", "einops", "numpy", "yaml", "rich", "PIL"]
for m in mods:
    try:
        mod = __import__(m)
        print(f"   ✓ {m:14s} {getattr(mod, '__version__', '')}")
    except Exception as e:
        print(f"   ✗ {m:14s} {type(e).__name__}: {e}")
PY

# ── 7. 真机能力探针 ───────────────────────────────────────────────────────────
say "7. 跑能力探针（这才是判断这台机器到底能干什么的依据）"
if [ -f "$WORKDIR/tools/npu_probe.py" ]; then
  ( cd "$WORKDIR" && "$PYBIN" tools/npu_probe.py --json /tmp/npu_probe.json ) || \
    echo "   ⚠ 探针非零退出——把上面的 FAIL 项抄下来，那就是这台机器的真实限制"
  echo "   结果已写入 /tmp/npu_probe.json（记得在任务超时前下载）"
else
  echo "   ⚠ 找不到 tools/npu_probe.py（代码还没挂载？）"
fi

# ── 8. 瘦身与体积 ─────────────────────────────────────────────────────────────
say "8. 清理与体积（镜像上限：鹏城计算I 20GB / 其他 40GB）"
$PIP cache purge >/dev/null 2>&1 || true
rm -rf /root/.cache/pip /home/ma-user/.cache/pip 2>/dev/null || true
echo "   各目录占用（大到小，前 12）："
du -sh /usr/local/* /opt/* 2>/dev/null | sort -rh | head -n 12 | sed 's/^/     /'
echo "   根文件系统："
df -h / 2>/dev/null | sed 's/^/     /'

hr
cat <<'EOF'
下一步（顺序不能反）：
  1. 确认上面第 6 步全是 ✓、第 7 步没有致命 FAIL
  2. **不要停止本调试任务**
  3. 回到调试任务列表 →「更多」下拉 →「提交镜像」，填名称如 anima-npu-torch2.1-py39
  4. 提交期间任务转 WAITING，耐心等
  5. 之后每次新建任务直接选这个自定义镜像，不必重装
EOF
hr
