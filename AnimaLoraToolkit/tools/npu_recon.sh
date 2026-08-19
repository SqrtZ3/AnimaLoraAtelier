#!/usr/bin/env bash
# 昇腾 NPU 容器：只读侦察脚本。
#
# 上机第一件事跑它，**不装任何东西、不改任何环境**。它回答的是决定后面所有事的
# 那几个问题：
#   1. CPU 架构 —— 决定 torch 该从哪个源装（x86_64 走错源会拉到 CUDA 版，环境当场废）
#   2. CANN 实际版本 —— 决定 torch/torch_npu 的版本上限（配套表是硬约束）
#   3. 宿主 driver 版本 —— 决定 CANN 能不能升（容器里升不了驱动）
#   4. 有没有 torch / torch_npu、在哪个 conda 环境里
#   5. site-packages 会不会随镜像提交（NPU 环境下 /home/ma-user/work 不入镜像）
#   6. 磁盘余量（镜像上限 鹏城计算I 20GB / 其他 40GB）
#
# 用法：
#     bash tools/npu_recon.sh
#     bash tools/npu_recon.sh > /tmp/recon.txt 2>&1    # 存档带回来

set -uo pipefail

say() { echo -e "\n\033[1m── $* ─────────────────────────────────────\033[0m"; }
kv()  { printf '   %-22s %s\n' "$1" "$2"; }

echo "════════════════════════════════════════════════════════════════════"
echo " 昇腾 NPU 容器侦察（只读，不修改任何东西）  $(date '+%F %T')"
echo "════════════════════════════════════════════════════════════════════"

# ── 1. 机器与架构 ─────────────────────────────────────────────────────────────
say "1. 机器与架构"
kv "uname -m"      "$(uname -m)"
kv "uname -r"      "$(uname -r)"
kv "CPU 核心"      "$(nproc 2>/dev/null || echo '?')"
kv "内存"          "$(free -g 2>/dev/null | awk '/^Mem:/{print $2" GB"}')"
if [ -f /etc/os-release ]; then
  kv "发行版" "$(. /etc/os-release; echo "$PRETTY_NAME")"
fi
case "$(uname -m)" in
  aarch64) echo "   → torch 装法：pip install torch==<版本>（PyPI 的 aarch64 wheel 本来就是 CPU 构建，正合 torch_npu）" ;;
  x86_64)  echo "   → torch 装法：**必须** --index-url https://download.pytorch.org/whl/cpu"
           echo "      走默认 PyPI 会拉到 CUDA 构建，几个 GB 且与 torch_npu 冲突" ;;
  *)       echo "   → 未知架构，装 torch 前先确认有对应 wheel" ;;
esac

# ── 2. 昇腾软件栈 ─────────────────────────────────────────────────────────────
say "2. 昇腾软件栈（CANN / driver / firmware）"
if [ -d /usr/local/Ascend ]; then
  kv "/usr/local/Ascend" "$(ls /usr/local/Ascend 2>/dev/null | tr '\n' ' ')"
else
  echo "   ⚠ 没有 /usr/local/Ascend —— 这可能不是昇腾容器"
fi
# CANN toolkit 版本：install.info 里 version= 那一行才是权威
for f in /usr/local/Ascend/ascend-toolkit/latest/*/ascend_toolkit_install.info \
         /usr/local/Ascend/ascend-toolkit/latest/ascend_toolkit_install.info \
         /usr/local/Ascend/nnae/latest/ascend_nnae_install.info; do
  [ -f "$f" ] && { echo "   $f:"; sed 's/^/     /' "$f"; }
done
# driver / firmware 在宿主机装，容器只读得到版本号——它是 CANN 能升到多高的天花板
for f in /usr/local/Ascend/driver/version.info /usr/local/Ascend/firmware/version.info; do
  [ -f "$f" ] && { echo "   $f:"; sed 's/^/     /' "$f"; }
done
if command -v npu-smi >/dev/null 2>&1; then
  echo "   npu-smi info:"
  npu-smi info 2>&1 | head -n 16 | sed 's/^/     /'
  echo "   npu-smi 版本:"
  npu-smi info -t board -i 0 2>&1 | head -n 12 | sed 's/^/     /'
else
  echo "   ⚠ 找不到 npu-smi"
fi
kv "ASCEND_RT_VISIBLE_DEVICES" "${ASCEND_RT_VISIBLE_DEVICES:-<未设>}"

# ── 3. Python 环境盘点 ────────────────────────────────────────────────────────
say "3. Python 环境盘点"
for p in python3 python; do
  command -v "$p" >/dev/null 2>&1 && kv "$p" "$("$p" -V 2>&1) @ $(command -v $p)"
done
if command -v conda >/dev/null 2>&1; then
  echo "   conda 环境列表："
  conda env list 2>/dev/null | sed 's/^/     /'
  echo "   ↑ 镜像名里的 ms=MindSpore。若这些环境里没有 torch，就得自己装（见第 4 节结论）"
else
  echo "   （无 conda）"
fi

# ── 4. 各环境里的 torch / torch_npu / mindspore ───────────────────────────────
say "4. 各 Python 环境里的框架"
_probe_py() {
  local py="$1" label="$2"
  [ -x "$py" ] || return 0
  echo "   [$label] $py"
  "$py" - <<'PY' 2>&1 | sed 's/^/     /'
import importlib, sys
print(f"python {sys.version.split()[0]}")
for m in ("torch", "torch_npu", "mindspore", "transformers", "numpy", "safetensors", "einops"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m:14s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"  {m:14s} <缺失: {type(e).__name__}>")
import site
sp = site.getsitepackages()
print(f"  site-packages: {sp}")
bad = [p for p in sp if p.startswith("/home/ma-user/work") or p.startswith("/tmp/code") or p.startswith("/tmp/dataset")]
if bad:
    print(f"  !! site-packages 在不入镜像的目录 {bad} —— 装在这里的包提交镜像后会丢")
PY
}
_probe_py "$(command -v python3 || true)" "默认 python3"
if command -v conda >/dev/null 2>&1; then
  while read -r name path _; do
    case "$name" in ''|'#'*) continue ;; esac
    [ "$path" = "*" ] && continue
    [ -x "$path/bin/python" ] && _probe_py "$path/bin/python" "conda:$name"
  done < <(conda env list 2>/dev/null | tr -s ' ')
fi

# ── 5. 代码与数据挂载点 ───────────────────────────────────────────────────────
say "5. 挂载点与产物目录"
for d in /tmp/code /tmp/dataset /tmp/output /tmp/pretrainmodel /home/ma-user/work /cache; do
  if [ -d "$d" ]; then
    kv "$d" "存在，$(ls -1 "$d" 2>/dev/null | head -5 | tr '\n' ' ')"
  else
    kv "$d" "<不存在>"
  fi
done
echo "   注意：/tmp/code 与 /tmp/dataset **不入镜像**；NPU 环境下 /home/ma-user/work 也不入镜像。"

# ── 6. 网络（决定代码怎么上来、pip 走哪个源） ─────────────────────────────────
say "6. 网络连通性（各 5 秒超时）"
for url in https://pypi.tuna.tsinghua.edu.cn/simple/ https://pypi.org/simple/ \
           https://cnb.cool https://github.com https://download.pytorch.org; do
  if curl -sS -o /dev/null -m 5 -w '%{http_code}' "$url" >/tmp/_rc 2>/dev/null; then
    kv "$url" "HTTP $(cat /tmp/_rc)"
  else
    kv "$url" "不可达 / 超时"
  fi
done
rm -f /tmp/_rc
command -v git >/dev/null 2>&1 && kv "git" "$(git --version)" || kv "git" "<未安装>"

# ── 7. 磁盘 ───────────────────────────────────────────────────────────────────
say "7. 磁盘（镜像上限：鹏城计算I 20GB / 其他算力中心 40GB）"
df -h / /tmp /home 2>/dev/null | sed 's/^/   /'
echo "   大目录（前 10）："
du -sh /usr/local/* /opt/* 2>/dev/null | sort -rh | head -n 10 | sed 's/^/     /'

echo
echo "════════════════════════════════════════════════════════════════════"
cat <<'EOF'
侦察完了，把上面整段贴回来。接下来的判断依据：
  * 第 1 节 uname -m  → 决定 torch 装法（x86_64 必须走 whl/cpu 源）
  * 第 2 节 CANN 版本 → 决定 torch 版本上限（见 docs/ascend-npu.md 的配套表）
  * 第 4 节 torch_npu → 有就直接跑探针；没有就先 npu_setup_image.sh 装
  * 第 6 节 网络     → cnb.cool/github 通不通决定代码怎么上来
EOF
echo "════════════════════════════════════════════════════════════════════"
