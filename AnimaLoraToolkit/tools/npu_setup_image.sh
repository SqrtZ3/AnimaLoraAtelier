#!/usr/bin/env bash
# 昇腾 NPU 调试任务：环境装配脚本（为「提交镜像」准备）
#
# 用法（在 OpenI 调试任务的终端里）：
#     bash tools/npu_setup_image.sh            # 装必需依赖
#     bash tools/npu_setup_image.sh --check    # 只体检，不装任何东西
#     bash tools/npu_setup_image.sh --with-perceptual --with-jxl
#     bash tools/npu_setup_image.sh --with-workbench   # + wandb + code-server + /proxy 通道
#
# 设计要点：
#   1. **绝不碰 torch / torch_npu**。生成一份 pip constraints 把已装的
#      torch / torch_npu / numpy 版本钉死——否则任何一个依赖（尤其 torchvision）
#      都可能从 PyPI 拉一个 CUDA 版 torch 覆盖掉 torch_npu，环境当场报废。
#   2. 只装**真实运行路径**用得到的包。requirements.txt 里的 diffusers /
#      accelerate / peft / lycoris-lora / pytorch-fid 经 AST 扫描确认未被 import，
#      一律不装（镜像有 20GB/40GB 上限）。
#      ⚠ torchvision 一度也被列在这里，那是**错的**：models/cosmos_predict2_modeling.py
#      顶层 `from torchvision import transforms` 是运行路径必经的（trainer/models.py:105
#      动态加载它），anima_train.py 的依赖预检会直接 fail。现已把那处唯一用途
#      （padding_mask 最近邻 resize）改成纯 torch 实现，torchvision 才真正成为非必需。
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
WITH_WANDB=0
WITH_WORKBENCH=0
for arg in "$@"; do
  case "$arg" in
    --check)            CHECK_ONLY=1 ;;
    --with-perceptual)  WITH_PERCEPTUAL=1 ;;
    --with-jxl)         WITH_JXL=1 ;;
    --with-extra-opt)   WITH_EXTRA_OPT=1 ;;
    --with-wandb)       WITH_WANDB=1 ;;
    --with-workbench)   WITH_WORKBENCH=1; WITH_WANDB=1 ;;
    *) echo "未知参数: $arg"; exit 2 ;;
  esac
done

# code-server 版本：改这里就能换。aarch64 用 linux-arm64 包。
CODE_SERVER_VER="${CODE_SERVER_VER:-4.132.0}"

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
# 规则：**已经装了 torch 就绝不动它**（换版本 = 赌上整个环境）。只有完全没有 torch
# 时才按「CANN 版本 → 配套 torch 版本」+「CPU 架构 → 装哪个源」自动装一套。
say "2. torch / torch_npu 现状"
ARCH="$(uname -m)"
TORCH_VER="$("$PYBIN" -c 'import torch;print(torch.__version__)' 2>/dev/null)"
TORCH_NPU_VER="$("$PYBIN" -c 'import torch_npu;print(torch_npu.__version__)' 2>/dev/null)"
NUMPY_VER="$("$PYBIN" -c 'import numpy;print(numpy.__version__)' 2>/dev/null)"
PYVER="$("$PYBIN" -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
echo "   架构       = $ARCH"
echo "   python     = $PYVER"
echo "   torch      = ${TORCH_VER:-<未安装>}"
echo "   torch_npu  = ${TORCH_NPU_VER:-<未安装 / 导入失败>}"
echo "   numpy      = ${NUMPY_VER:-<未安装>}"

# CANN 实际版本（install.info 里的 version= 才是权威，镜像名不可信）
CANN_VER=""
for f in /usr/local/Ascend/ascend-toolkit/latest/*/ascend_toolkit_install.info \
         /usr/local/Ascend/ascend-toolkit/latest/ascend_toolkit_install.info; do
  [ -f "$f" ] || continue
  CANN_VER="$(grep -iE '^version=' "$f" 2>/dev/null | head -1 | cut -d= -f2 | tr -d '[:space:]')"
  [ -n "$CANN_VER" ] && break
done
echo "   CANN       = ${CANN_VER:-<读不到 install.info>}"

# 官方版本配套表（Ascend/pytorch 的 COMPATIBILITY.md）。CANN 是硬上限：
#   CANN 8.3.RC1 → torch 2.8.0 / 2.7.1 / 2.6.0.post3
#   CANN 8.2.RC1 → torch 2.6.0 / 2.5.1.post1
#   CANN 8.1.RC1 → torch 2.5.1 / 2.4.0.post4
#   CANN 8.0.0   → torch 2.4.0.post2 / 2.3.1.post4
#   CANN 8.0.RC3 → torch 2.4.0 / 2.3.1.post2
#   CANN 8.0.RC1 → torch 2.2.0 / 2.1.0.post4
# 取每档里**偏保守的一个**（不取最新那个），少踩 post 版本的坑。
pick_torch_for_cann() {
  case "${1:-}" in
    8.3*)            echo "2.7.1  2.7.1" ;;
    8.2*)            echo "2.6.0  2.6.0" ;;
    8.1*)            echo "2.5.1  2.5.1" ;;
    8.0.0*)          echo "2.3.1  2.3.1.post4" ;;
    8.0.RC3*|8.0.rc3*) echo "2.3.1  2.3.1.post2" ;;
    8.0.RC1*|8.0.rc1*) echo "2.1.0  2.1.0.post4" ;;
    *)               echo "" ;;
  esac
}

if [ -z "$TORCH_VER" ]; then
  say "2b. 没有 torch —— 按 CANN/架构自动装一套"
  if [ -z "$CANN_VER" ]; then
    echo "   ✗ 读不到 CANN 版本，无法决定 torch 版本。先跑 tools/npu_recon.sh 看第 2 节。"
    exit 1
  fi
  read -r WANT_TORCH WANT_TORCH_NPU <<<"$(pick_torch_for_cann "$CANN_VER")"
  if [ -z "${WANT_TORCH:-}" ]; then
    echo "   ✗ CANN=$CANN_VER 不在已知配套表里。"
    echo "     查 https://github.com/Ascend/pytorch 的 COMPATIBILITY.md，手动指定："
    echo "     ANIMA_TORCH=2.6.0 ANIMA_TORCH_NPU=2.6.0 bash tools/npu_setup_image.sh"
    exit 1
  fi
  WANT_TORCH="${ANIMA_TORCH:-$WANT_TORCH}"
  WANT_TORCH_NPU="${ANIMA_TORCH_NPU:-$WANT_TORCH_NPU}"
  echo "   CANN=$CANN_VER → 选定 torch==$WANT_TORCH / torch-npu==$WANT_TORCH_NPU"

  # 架构决定装哪个源。这是最容易废掉环境的一步：
  #   x86_64 走默认 PyPI 会拿到**CUDA 构建**的 torch（几个 GB，且 torch_npu 不认）。
  #   aarch64 的 PyPI wheel 本来就是 CPU 构建，直接装即可。
  case "$ARCH" in
    x86_64)
      echo "   x86_64 → 强制走 CPU 专用源，避免拉到 CUDA 构建"
      $PYBIN -m pip install "torch==${WANT_TORCH}" \
        --index-url https://download.pytorch.org/whl/cpu || { echo "✗ torch 安装失败"; exit 1; }
      ;;
    aarch64)
      echo "   aarch64 → PyPI wheel 即 CPU 构建"
      $PYBIN -m pip install --index-url "$PIP_INDEX" "torch==${WANT_TORCH}" \
        || { echo "✗ torch 安装失败"; exit 1; }
      ;;
    *)
      echo "   ✗ 未知架构 $ARCH，不敢自动装。手动确认有对应 wheel 再来。"; exit 1 ;;
  esac
  $PYBIN -m pip install --index-url "$PIP_INDEX" "torch-npu==${WANT_TORCH_NPU}" \
    || { echo "✗ torch-npu 安装失败（确认该版本有 cp${PYVER//./} 的 $ARCH wheel）"; exit 1; }

  TORCH_VER="$("$PYBIN" -c 'import torch;print(torch.__version__)' 2>/dev/null)"
  TORCH_NPU_VER="$("$PYBIN" -c 'import torch_npu;print(torch_npu.__version__)' 2>/dev/null)"
  echo "   装完：torch=${TORCH_VER:-失败} torch_npu=${TORCH_NPU_VER:-失败}"
  [ -n "$TORCH_VER" ] || { echo "   ✗ 装完还是导不进 torch"; exit 1; }
else
  echo "   → 已有 torch，本脚本不会改动它（换版本 = 赌上整个环境）"
fi

if [ -z "$TORCH_NPU_VER" ]; then
  echo "   ✗ torch_npu 不可用 —— 后面所有 NPU 训练都无从谈起，先解决这个。"
fi

# ── 3. constraints：钉死 torch 系，防止被依赖解析悄悄换掉 ─────────────────────
say "3. 生成 pip constraints（防止 torchvision 等把 torch 换成 CUDA 版）"
CONS="/tmp/anima_npu_constraints.txt"
# numpy 上限跟 torch 走：torch 2.1/2.2 按 numpy 1.x ABI 编译，numpy 2 会在运行期报
# _ARRAY_API 错误；torch>=2.3 官方支持 numpy 2，不该无谓钉死。
NUMPY_PIN="numpy<2"
case "${TORCH_VER%%+*}" in
  2.1.*|2.2.*|1.*) NUMPY_PIN="numpy<2" ;;
  *)               NUMPY_PIN="numpy" ;;
esac
{
  echo "torch==$TORCH_VER"
  [ -n "$TORCH_NPU_VER" ] && echo "torch-npu==$TORCH_NPU_VER"
  [ "$NUMPY_PIN" = "numpy<2" ] && echo "numpy<2"
} > "$CONS"
cat "$CONS" | sed 's/^/   /'

PIP="$PYBIN -m pip"
PIP_ARGS="--index-url $PIP_INDEX --constraint $CONS"

# ── 4. 依赖清单 ───────────────────────────────────────────────────────────────
# 必需：由 AST 扫描 anima_train.py + trainer/ + utils/ + models/ 的真实 import 得出。
# transformers>=4.51 是硬下限：trainer/models.py:259 用 AutoModelForCausalLM 加载
# Qwen3-0.6B，更早的版本不认识 Qwen3 架构。
REQUIRED=(
  "$NUMPY_PIN"
  "Pillow"
  "safetensors"
  "transformers>=4.51,<5"
  "PyYAML"
  "rich"
  "einops"
  # ⚠ sentencepiece：AST 扫描**看不见**它。transformers 的 T5Tokenizer 用
  # requires_backends 做运行时门控，源码里没有 `import sentencepiece`，
  # 直到 trainer/models.py:279 真正 from_pretrained 才 ImportError ——
  # 而那已经是加载完 DiT + Qwen 之后了。requirements.txt 一直列着它，
  # 是这份精简清单漏了。无 torch 依赖，装它不会动 torch。
  "sentencepiece"
)
# 可选组
PERCEPTUAL=( "lpips" )          # trainer/aux_losses.py，仅 perceptual aux loss 用；会拉 torchvision（危险，已被 constraints 挡住）
JXL=( "pillow-jxl-plugin" )     # 仅当数据集是 JXL
EXTRA_OPT=( "prodigy-plus-schedule-free" )  # 仅当用 prodigyplus 优化器

# 明确**不装**（并说明原因），避免下次有人「顺手补全 requirements.txt」
say "4. 明确不安装的包（不是遗漏）"
cat <<'EOF'
   xformers        —— 昇腾无此包；改用 navit_attn_backend: npu_tnd / sdpa_seg，NaViT 打包仍可用
   bitsandbytes    —— 无昇腾后端；utils/optimizer_utils.py:76 是 try/except 可选导入
   triton          —— 昇腾无
   flash-attn      —— 昇腾无
   torchvision     —— 极易连带把 torch 换成 CUDA 版。曾是运行路径硬依赖
                      (cosmos_predict2_modeling.py 顶层 import)，该处已改纯 torch 实现
   diffusers / accelerate / peft / lycoris-lora / pytorch-fid
                   —— AST 扫描确认运行路径未 import，纯占镜像体积
   （accelerate 缺席会打一条 WARNING：Qwen 的 device_map 路径失败、回退 .to(device)，
     "CPU RAM 会临时翻倍"。TE 只有 1.1GB 而机器 192GB RAM —— 属预期行为，不是错误。）
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

if [ "$WITH_WANDB" = "1" ]; then
  say "5e. 安装 wandb（配 config 的 wandb_enabled: true；见 trainer/wandb_logger.py）"
  # shellcheck disable=SC2086
  $PIP install $PIP_ARGS wandb || echo "   ⚠ wandb 安装失败（非致命，训练不受影响）"
fi

# ── 5f. 工作台组（opt-in）──────────────────────────────────────────────────────
# 解决两个平台限制：① JupyterLab 3 的文件浏览器不支持上传目录；② 平台不暴露额外端口，
# 内置的 train_monitor(6006) 看不到。jupyter-server-proxy 把容器内任意本地端口挂到
# **平台自己那个 HTTPS 地址**的 /proxy/<port>/ 下（不出平台域名，不是内网穿透），
# code-server 则提供 VS Code 网页版（其资源管理器支持拖入整个文件夹）。
#
# 注意：装完**必须提交镜像 + 新建任务**才生效——jupyter server 扩展只在 server 启动时
# 加载，而 server 是平台拉起来的，在跑着的任务里重启它有丢会话的风险。
if [ "$WITH_WORKBENCH" = "1" ]; then
  say "5f. 安装工作台组（jupyter-server-proxy + code-server）"

  # jupyter-server-proxy 4.x 要求 jupyter-server>=1.24；低于它退回 3.2.4。
  # **不升级 jupyter-server / jupyterlab**：平台的启动命令可能带 jupyter-server 1.x
  #  专属参数，升到 2.x 会让 server 起不来——那样提交出去的镜像是个进不去的砖。
  JS_VER="$("$PYBIN" -c 'import importlib.metadata as m;print(m.version("jupyter-server"))' 2>/dev/null || echo "0")"
  echo "   已装 jupyter-server: ${JS_VER:-无}"
  PROXY_SPEC="jupyter-server-proxy"
  case "$JS_VER" in
    0|"") PROXY_SPEC="jupyter-server-proxy" ;;
    1.*)
      JS_MINOR="$(echo "$JS_VER" | cut -d. -f2)"
      if [ "${JS_MINOR:-0}" -lt 24 ]; then
        PROXY_SPEC="jupyter-server-proxy==3.2.4"
        echo "   jupyter-server < 1.24 → 装 $PROXY_SPEC（4.x 要求 >=1.24）"
      fi
      ;;
  esac
  # shellcheck disable=SC2086
  $PIP install $PIP_ARGS "$PROXY_SPEC" jupyter-codeserver-proxy \
    || echo "   ⚠ proxy 安装失败（非致命，只是没有 /proxy/<port>/ 通道）"

  ARCH="$(uname -m)"
  case "$ARCH" in
    aarch64|arm64) CS_ARCH="arm64" ;;
    x86_64|amd64)  CS_ARCH="amd64" ;;
    *) CS_ARCH="" ; echo "   ⚠ 未知架构 $ARCH，跳过 code-server" ;;
  esac
  if [ -n "$CS_ARCH" ]; then
    CS_DIR="/opt/code-server-${CODE_SERVER_VER}-linux-${CS_ARCH}"
    if [ -x "$CS_DIR/bin/code-server" ]; then
      echo "   ✓ code-server 已存在：$CS_DIR"
    else
      CS_URL="https://github.com/coder/code-server/releases/download/v${CODE_SERVER_VER}/code-server-${CODE_SERVER_VER}-linux-${CS_ARCH}.tar.gz"
      echo "   下载 $CS_URL"
      mkdir -p /opt
      # 下载器三选一：启智的 NPU 镜像里 curl 和 wget **都可能没有**（实测
      # cann8.2.rc2-ms2.7-py3.11-910b 两个都缺），python 标准库是唯一保底。
      fetch_stdout() {
        if command -v curl >/dev/null 2>&1; then curl -fL --retry 3 "$1"
        elif command -v wget >/dev/null 2>&1; then wget -qO- "$1"
        else "$PYBIN" -c 'import sys,urllib.request as u
r=u.urlopen(u.Request(sys.argv[1], headers={"User-Agent":"anima-lora-train"}), timeout=60)
b=sys.stdout.buffer
while True:
    c=r.read(1<<20)
    if not c: break
    b.write(c)' "$1"
        fi
      }
      if fetch_stdout "$CS_URL" | tar xz -C /opt; then
        echo "   ✓ 解包到 $CS_DIR"
      else
        echo "   ⚠ code-server 下载/解包失败（非致命）"
      fi
    fi
    # /usr/local/bin 进镜像；jupyter-codeserver-proxy 靠 PATH 找 code-server
    [ -x "$CS_DIR/bin/code-server" ] && ln -sf "$CS_DIR/bin/code-server" /usr/local/bin/code-server
    command -v code-server >/dev/null && code-server --version | head -1
  fi

  cat <<'EOF'

   用法（**提交镜像 → 新建任务**之后才生效）：
     code-server --auth none --bind-addr 127.0.0.1:8080 /tmp/code
     然后把浏览器里 JupyterLab 地址的 /lab 换成 /proxy/8080/
     训练监控同理：不设 no_monitor，访问 /proxy/6006/
   ⚠ 平台的 nginx 是否放行 /proxy/ 子路径未验证——这是整条通道的单点，先小成本试。
EOF
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
  3. 回到调试任务列表 →「更多」下拉 →「提交镜像」，填名称如 anima-npu-cann82-torch26-py311
  4. 提交期间任务转 WAITING，耐心等
  5. 之后每次新建任务直接选这个自定义镜像，不必重装
EOF
hr
