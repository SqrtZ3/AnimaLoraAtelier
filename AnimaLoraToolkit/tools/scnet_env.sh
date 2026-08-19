#!/bin/bash
# scnet Notebook 容器的环境补全 —— SSH 会话必须 source，Jupyter 终端不需要。
#
# 为什么需要它：
#   1) DTK 的 env.sh 不在 /etc/profile.d 里，不 source 就 `import torch` 报
#      libgalaxyhip.so.5 找不到。
#   2) ★ 这个容器**没有直连出网**，全部走内网 HTTP 代理。这套 proxy 变量只存在于
#      Jupyter 进程的环境里，sshd 由 pid 1 拉起、不继承 —— 所以 SSH 会话里
#      pip / git / wandb 全部超时，看起来像"完全断网"。
#
# 代理地址优先从正在跑的 jupyter-lab 进程里读（重启后凭据可能变），读不到才用兜底值。

# --- 1) DTK ---
if [ -z "${ROCM_PATH:-}" ] && [ -f /opt/dtk/env.sh ]; then
    _u=$(set +o | grep nounset); set +u
    . /opt/dtk/env.sh >/dev/null 2>&1
    eval "$_u"; unset _u
fi

# --- 2) 出网代理 ---
if [ -z "${https_proxy:-}" ]; then
    _jp=$(ps -eo pid,args 2>/dev/null | grep "[j]upyter-lab" | awk "{print \$1}" | head -1)
    _px=""
    if [ -n "$_jp" ] && [ -r "/proc/$_jp/environ" ]; then
        _px=$(tr "\0" "\n" < "/proc/$_jp/environ" | sed -n "s/^https_proxy=//p" | head -1)
    fi
    [ -z "$_px" ] && _px="http://preset:6e298f07@10.16.1.51:3128"
    export http_proxy="$_px" https_proxy="$_px" ftp_proxy="$_px"
    export HTTP_PROXY="$_px" HTTPS_PROXY="$_px"
    export no_proxy="*.zzai2.scnet.cn,zzai2.scnet.cn,localhost,127.0.0.1"
    export NO_PROXY="$no_proxy"
    unset _jp _px
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
