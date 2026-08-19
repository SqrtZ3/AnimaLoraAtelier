#!/usr/bin/env python
"""从启智 OpenI 按**单个文件**拉模型/数据集（纯标准库，无需 curl/wget/pip 装包）。

**为什么需要它**：
1. 挂载整个模型（如 `FoundationModel/Anima`）会拖慢调试任务启动；平台其实支持按单文件取。
2. 启智的 NPU 镜像里**可能既没有 curl 也没有 wget**（实测 `cann8.2.rc2-ms2.7-py3.11-910b`
   两个都没有），而 `pip install openi` 又要先有网、要装 8 个依赖。标准库 urllib 一定有。

**接口**（2026-08-13 匿名实测可用，公开模型无需 token）：

    GET /api/v1/aimodel/file/meta?aimodel_name=<owner/name>&file_name=<路径>&parent_dir=
        → {"code":0,"data":{"ContentLength":...}}
    GET /api/v1/aimodel/file?<同参数>
        → 301 到鹏城 OBS 签名直链（支持 Range，签名约 1 小时过期）

数据集把 `aimodel` 换成 `dataset`、参数名换成 `dataset_name`（`--type dataset`）。

**断点续传**：本地已有部分字节时带 `Range: bytes=<已有>-` 续传；服务端不支持 Range（返回 200
而非 206）就从头重下，不会把两段拼错。下完按 meta 的 ContentLength 校验大小，不符直接报错。

用法
----
    # 拉 Anima 三件套到 /opt/anima_models
    python tools/openi_fetch.py FoundationModel/Anima -d /opt/anima_models \
        -f split_files/diffusion_models/anima-base-v1.0.safetensors \
        -f split_files/text_encoders/qwen_3_06b_base.safetensors \
        -f split_files/vae/qwen_image_vae.safetensors

    # 只看大小不下载
    python tools/openi_fetch.py FoundationModel/Anima -f <路径> --meta-only

    # 私有仓需要 token（/user/settings/applications 生成）
    python tools/openi_fetch.py you/private-ds --type dataset -f data.zip --token <TOKEN>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

DEFAULT_ENDPOINT = "https://openi.pcl.ac.cn"
_SUBJECT = {  # --type → (URL 段, 查询参数名)
    "model": ("aimodel", "aimodel_name"),
    "dataset": ("dataset", "dataset_name"),
}


def _url(endpoint: str, seg: str, tail: str, params: dict) -> str:
    return f"{endpoint}/api/v1/{seg}{tail}?" + urllib.parse.urlencode(params)


def _open(url: str, token: str | None, headers: dict | None = None):
    req = urllib.request.Request(url)
    # 平台默认 UA 也能过，但显式带一个便于对方日志排查
    req.add_header("User-Agent", "anima-lora-train/openi_fetch")
    if token:
        req.add_header("Authorization", f"token {token}")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    return urllib.request.urlopen(req, timeout=60)


def fetch_meta(endpoint: str, seg: str, param: str, repo: str, name: str, token: str | None) -> int:
    """返回文件字节数；文件不存在/接口报错时抛 RuntimeError。"""
    url = _url(endpoint, seg, "/file/meta", {param: repo, "file_name": name, "parent_dir": ""})
    with _open(url, token) as r:
        body = json.loads(r.read().decode("utf-8"))
    if body.get("code") != 0:
        raise RuntimeError(f"meta 失败：{name} → {body.get('msg', body)}")
    size = int(body.get("data", {}).get("ContentLength", 0))
    if size <= 0:
        raise RuntimeError(f"meta 返回的大小不合法：{name} → {body}")
    return size


def _human(n: float) -> str:
    for u in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024 or u == "GiB":
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}GiB"


def download(endpoint: str, seg: str, param: str, repo: str, name: str,
             out: Path, size: int, token: str | None, force: bool) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    have = out.stat().st_size if out.exists() else 0

    if have == size and not force:
        print(f"  ✓ 已完整存在，跳过（{_human(size)}）")
        return
    if have > size:
        print(f"  ⚠ 本地比云端大（{have} > {size}），从头重下")
        have = 0
    if force:
        have = 0

    url = _url(endpoint, seg, "/file", {param: repo, "file_name": name, "parent_dir": ""})
    headers = {"Range": f"bytes={have}-"} if have else {}
    resp = _open(url, token, headers)
    # 服务端不认 Range（返回 200 而不是 206）时必须从头写，否则会把两段拼错
    resuming = have > 0 and getattr(resp, "status", 200) == 206
    if have and not resuming:
        print("  ⚠ 服务端未接受 Range，从头重下")
        have = 0

    mode = "ab" if resuming else "wb"
    done = have if resuming else 0
    t0 = time.time()
    last = 0.0
    with resp, open(out, mode) as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            now = time.time()
            if now - last > 1.0:
                last = now
                spd = done / max(now - t0, 1e-6)
                pct = 100.0 * done / size if size else 0.0
                print(f"\r  {pct:5.1f}%  {_human(done)}/{_human(size)}  {_human(spd)}/s   ",
                      end="", flush=True)
    print()

    got = out.stat().st_size
    if got != size:
        raise RuntimeError(f"大小不符：{got} != {size}（下载不完整，重跑本命令可续传）")
    print(f"  ✓ {out}  {_human(got)}")


def main() -> int:
    p = argparse.ArgumentParser(description="从启智 OpenI 按单文件拉模型/数据集（纯标准库）")
    p.add_argument("repo_id", help="完整名称，格式 `拥有者/名称`")
    p.add_argument("-f", "--file", dest="files", action="append", required=True,
                   help="云端文件名（含子目录，如 split_files/vae/x.safetensors）。可重复")
    p.add_argument("-d", "--dir", default=".", help="本地保存根目录（默认当前目录）")
    p.add_argument("--type", choices=sorted(_SUBJECT), default="model", help="model（默认）/ dataset")
    p.add_argument("--flatten", action="store_true",
                   help="只按文件名保存，不复刻云端子目录结构")
    p.add_argument("--token", default=os.environ.get("OPENI_TOKEN", ""),
                   help="私有仓需要；也可用环境变量 OPENI_TOKEN")
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    p.add_argument("--force", action="store_true", help="忽略本地已有文件，从头重下")
    p.add_argument("--meta-only", action="store_true", help="只查大小，不下载")
    args = p.parse_args()

    seg, param = _SUBJECT[args.type]
    token = args.token or None
    root = Path(args.dir)

    failed = []
    for name in args.files:
        print(f"\n[{args.type}] {args.repo_id} :: {name}")
        try:
            size = fetch_meta(args.endpoint, seg, param, args.repo_id, name, token)
        except (urllib.error.URLError, RuntimeError, json.JSONDecodeError) as e:
            print(f"  ✗ {e}")
            failed.append(name)
            continue
        print(f"  云端大小 {size} 字节（{_human(size)}）")
        if args.meta_only:
            continue
        out = root / (Path(name).name if args.flatten else name)
        try:
            download(args.endpoint, seg, param, args.repo_id, name, out, size, token, args.force)
        except (urllib.error.URLError, RuntimeError, OSError) as e:
            print(f"  ✗ {e}")
            failed.append(name)

    if failed:
        print(f"\n✗ 失败 {len(failed)} 个：{', '.join(failed)}")
        return 1
    print("\n✓ 全部完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
