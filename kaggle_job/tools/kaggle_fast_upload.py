r"""**并行 Kaggle Dataset 上传器** —— 把上行带宽跑成一条平线，而不是一串尖峰。

## 为什么要自己写一个

官方 `kaggle datasets create/version` 的上传是**严格串行**的
（`kaggle/api/kaggle_api_extended.py:upload_files` 里一个
`for file_name in os.listdir(folder)` 循环），每个文件要走两跳：

    ① POST api.kaggle.com  start_blob_upload   -> 拿 GCS 可续传 URL + token
    ② PUT  storage.googleapis.com <那个 URL>    -> 真正吐字节

①是一次**跨太平洋 RTT**（还常常是新建 TCP+TLS 连接），期间网卡上行是 0；
②才是打满带宽的那一下。串行地 ①②①②①②… 下去，任务管理器上就是
「尖峰—归零—尖峰—归零」的锯齿，而不是平线。文件越小、数量越多，锯齿的
占空比越差 —— 本仓库的 latent 缓存正好是「几百个 8MB npz」这种最坏形状。

本工具做三件事，对应三个损失来源：

  1. **并行**（`--workers`，默认 8）：N 个文件的 ①②交错进行，甲的 token
     往返期间乙丙丁正在 PUT。这是填平波谷的主力。
  2. **连接复用**：每个 worker 线程**只建一次** KaggleClient 并全程持有
     （官方实现是每个文件 `with self.build_kaggle_client()` 新建新连接、
     新 TLS 握手），token 往返从「3 次 RTT」降到「1 次 RTT」。
  3. **实时吞吐计**：每秒打一行瞬时/均值速率与在途数，**跑完给出波动指标**
     （见下）。不是为了好看 —— 是为了让「我调 workers 到底有没有用」可测。

不重复造的部分：认证、协议对象、最后的 create_dataset / create_dataset_version
调用全部走官方 `kaggle` 包，所以它和官方 CLI 传出来的东西是同一种东西。

## 波动指标怎么读

结束时打印

    带宽利用: 均值 268 Mbps | 中位 291 Mbps | 空闲秒占比 2.1% | 变异系数 0.18

  * **空闲秒占比** = 采样窗口里瞬时速率 < 均值 10% 的秒数占比。串行上传这个数
    很大（波谷全是 0），并行后应当趋近 0。**这是「有没有把带宽用好」的直接判据。**
  * **变异系数** = 瞬时速率的 标准差/均值。越小越平。
  * 两个数都是**本机测得的应用层字节/墙钟时间**，不含 TCP/TLS/HTTP 头开销，
    所以会比任务管理器的读数略低几个百分点 —— 别拿两者做逐位对比。

## 数据外泄闸门（默认开，本仓库红线）

上传目录里出现下列东西时**直接 fail-fast、不传**：

  * 图片扩展名（png/jpg/jpeg/webp/bmp/gif/tif/tiff/avif）与 caption 文本
    （txt/caption）—— TPU 侧训练根本不读它们（`jax_tpu/data.py` 只按 npz 推
    stem），传上去纯是暴露面。要传就显式 `--allow-ext .png,.txt`。
  * `.npz` 内部藏着 `caption` 字段 —— `tools/cache_text_features.py` 会把
    caption 明文一起塞进 `<stem>.textfeat.npz`。本工具读 zip 中央目录即可判定
    （不解压、几乎不耗时）。要传就 `--allow-npz-caption`。
    剥除方法见 `tools/dataset_encrypt.py`（训练路径不读 `caption`，剥掉逐 bit 无影响）。

`--dry-run` 只跑扫描 + 打印计划，一个字节都不发。

## 传完才失败？不用重传

blob token 每成功一个就落盘（默认在系统临时目录，按目录路径哈希命名，
`--state-file` 可指定）。所以「247 个文件全传完、最后 create_dataset 被拒」
这种最贵的失败，改掉冲突字段后重跑同一条命令，只会重发提交那一跳。
token 默认 6 小时过期（`--token-ttl-hours`），文件大小/mtime 变了也不复用；
`--fresh` 强制全部重传。

## 一个 Kaggle 的坑：「title already in use」其实说的是 slug

服务端回

    The requested title "<X>" is already in use by a dataset. Please choose another title.

时，**改 title 没有用**。本仓库 2026-08-21 实测：同一个 slug 换一个完全不同的
title 再传，报错原样出现、且回显的是新 title —— 冲突判在 `id` 的 slug 上，文案
写成了 title。

而且这种被占的 slug 可能**读不到**：`datasets status/files/metadata` 三个接口全回
403，而 Kaggle 对**不存在**的 dataset 也回 403（已存在的回 `ready`）。所以
`--mode create` 前的 preflight 只能判出「已存在且可读」，判不出这一种。

**已知会造成这种状态的原因：dataset 被 Kaggle 按条款下架。** 本仓库的
`jan-krea2-tpu-cache` 就是同日因为误传了训练原图 + caption 被以 NSFW 条款整个删除
（见 commit a473957），之后该 slug 既读不到、也不能重建。撞上了就换 slug，
**别去猜是不是自己删过**。

## 用法

    # 新建数据集（目录里要有 dataset-metadata.json，格式同官方 CLI）
    python kaggle_fast_upload.py <目录> --mode create

    # 传新版本
    python kaggle_fast_upload.py <目录> --mode version -m "重缓存 ms4096"

    # 先看看要传什么、闸门过不过
    python kaggle_fast_upload.py <目录> --mode create --dry-run

    # 链路好/差时调并发（8 是 100Mbps 家宽 + 几 MB 文件的经验值）
    python kaggle_fast_upload.py <目录> --mode create --workers 16

跑完照旧用 `python -m kaggle datasets status <owner>/<slug>` 等它转 ready
再 push kernel。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 官方包：认证 + 协议对象 + 最后的 create 调用都走它，不自己拼 HTTP
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.blobs.types.blob_api_service import ApiBlobType, ApiStartBlobUploadRequest
from kagglesdk.datasets.types.dataset_api_service import (
    ApiCreateDatasetRequest,
    ApiCreateDatasetVersionRequest,
    ApiCreateDatasetVersionRequestBody,
    ApiDatasetNewFile,
)

#: 默认拦下的扩展名。理由见模块 docstring 的「数据外泄闸门」。
DENY_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff",
            ".avif", ".txt", ".caption"}

#: 与官方 CLI 一致：这些是元数据 / 封面图，不当作数据文件传
META_NAMES = {"dataset-metadata.json", "datapackage.json", "kernel-metadata.json",
              "model-metadata.json", "model-instance-metadata.json"}

#: 目录里这些噪声一律跳过
SKIP_NAMES = {".DS_Store", "Thumbs.db", "__pycache__", ".ipynb_checkpoints", ".git"}


# ────────────────────────────────────────────────────────────── 扫描与闸门


@dataclass
class Plan:
    files: List[Path]
    total_bytes: int
    meta: Dict


def scan(folder: Path) -> Plan:
    """列出要传的文件（不递归 —— 与官方 `--dir-mode skip` 同口径）。"""
    meta_file = folder / "dataset-metadata.json"
    if not meta_file.exists():
        raise SystemExit(f"[FATAL] 缺 {meta_file} —— 格式与官方 CLI 相同：\n"
                         '  {"title": "...", "id": "<owner>/<slug>", '
                         '"licenses": [{"name": "CC0-1.0"}]}')
    meta = json.loads(meta_file.read_text(encoding="utf-8"))

    files, total = [], 0
    for p in sorted(folder.iterdir()):
        if p.name in META_NAMES or p.name in SKIP_NAMES or p.name.startswith("."):
            continue
        if p.is_dir():
            print(f"[WARN] 跳过子目录 {p.name}（本工具不递归，与官方 --dir-mode skip 同口径）")
            continue
        if not p.is_file():
            continue
        files.append(p)
        total += p.stat().st_size
    if not files:
        raise SystemExit(f"[FATAL] {folder} 下没有可传的文件")
    return Plan(files=files, total_bytes=total, meta=meta)


def gate_extensions(files: List[Path], allow_ext: set) -> None:
    bad: Dict[str, List[str]] = {}
    for p in files:
        ext = p.suffix.lower()
        if ext in DENY_EXT and ext not in allow_ext:
            bad.setdefault(ext, []).append(p.name)
    if not bad:
        return
    lines = [f"  {ext}  {len(names)} 个（如 {names[:3]}）" for ext, names in sorted(bad.items())]
    raise SystemExit(
        "[FATAL] 上传目录里有默认禁传的扩展名：\n" + "\n".join(lines) +
        "\n\n  原图与 caption 明文上公有云是本仓库的红线，且 TPU 训练路径压根不读它们"
        "\n  （jax_tpu/data.py 无图时按 npz 推 stem）。"
        "\n  确实要传就显式放行，如 --allow-ext " + ",".join(sorted(bad)))


def gate_npz_caption(files: List[Path], allow: bool) -> None:
    """扫 npz 内部有没有 `caption` 字段。只读 zip 中央目录，不解压。"""
    if allow:
        return
    bad = []
    for p in files:
        if p.suffix.lower() != ".npz":
            continue
        try:
            with zipfile.ZipFile(p) as z:
                if "caption.npy" in z.namelist():
                    bad.append(p.name)
        except zipfile.BadZipFile:
            raise SystemExit(f"[FATAL] {p.name} 不是合法 npz（zip 损坏）—— 先修数据再传")
    if bad:
        raise SystemExit(
            f"[FATAL] {len(bad)} 个 npz 内部带 `caption` 明文字段（如 {bad[:3]}）。\n"
            "  cache_text_features.py 会把 caption 一起塞进 <stem>.textfeat.npz，\n"
            "  但训练只读 `txt` —— 剥掉对训练逐 bit 无影响。\n"
            "  剥法见 tools/dataset_encrypt.py；确实要连明文一起传就 --allow-npz-caption")


# ──────────────────────────────────────────────────────── token 断点（state）
#
# 上传完 247 个文件之后才在最后一跳 create_dataset 上失败（比如「标题已被占用」），
# 如果 token 只活在进程内存里，就得把 2GB 重传一遍。所以每传完一个文件就把
# blob token 落盘；下次启动时凡是「文件大小 + mtime 没变、token 没过期」的直接
# 复用，只补传缺的，然后重新提交。


@dataclass
class TokenState:
    path: Path
    ttl_sec: float
    _lock: threading.Lock = None            # type: ignore[assignment]
    _d: Dict[str, Dict] = None              # type: ignore[assignment]

    def __post_init__(self):
        self._lock = threading.Lock()
        self._d = {}
        if self.path.exists():
            try:
                self._d = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:                                   # noqa: BLE001
                print(f"[WARN] 断点文件 {self.path} 读不动，当作空的")
                self._d = {}

    @staticmethod
    def default_path(folder: Path) -> Path:
        key = hashlib.sha1(str(folder).encode("utf-8")).hexdigest()[:16]
        d = Path(tempfile.gettempdir()) / "kaggle_fast_upload"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{key}.json"

    def get(self, p: Path) -> Optional[str]:
        """只在「文件没变过 + token 没过期」时复用，否则当没有。"""
        e = self._d.get(p.name)
        if not e:
            return None
        st = p.stat()
        if e.get("size") != st.st_size or int(e.get("mtime", -1)) != int(st.st_mtime):
            return None
        if time.time() - e.get("ts", 0) > self.ttl_sec:
            return None
        return e.get("token")

    def put(self, p: Path, token: str) -> None:
        st = p.stat()
        with self._lock:
            self._d[p.name] = {"token": token, "size": st.st_size,
                               "mtime": int(st.st_mtime), "ts": time.time()}
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._d), encoding="utf-8")
            os.replace(tmp, self.path)

    def clear(self) -> None:
        with self._lock:
            self._d = {}
            self.path.unlink(missing_ok=True)


# ────────────────────────────────────────────────────────────── 吞吐计


class Meter:
    """全局字节计数 + 每秒采样。线程安全（只有一个 += 在锁里）。"""

    def __init__(self, total_bytes: int, total_files: int, report_sec: float):
        self.total_bytes = total_bytes
        self.total_files = total_files
        self.report_sec = report_sec
        self._lock = threading.Lock()
        self._sent = 0
        self._done = 0
        self._inflight = 0
        self.samples: List[float] = []       # 每个采样窗口的 bit/s
        self._stop = threading.Event()
        self.t0 = 0.0
        self.t1 = 0.0

    def add(self, n: int) -> None:
        with self._lock:
            self._sent += n

    def file_started(self) -> None:
        with self._lock:
            self._inflight += 1

    def file_done(self) -> None:
        with self._lock:
            self._inflight -= 1
            self._done += 1

    def snapshot(self) -> Tuple[int, int, int]:
        with self._lock:
            return self._sent, self._done, self._inflight

    def run(self) -> None:
        self.t0 = time.monotonic()
        last_t, last_b = self.t0, 0
        while not self._stop.wait(self.report_sec):
            now = time.monotonic()
            sent, done, inflight = self.snapshot()
            dt = now - last_t
            inst = (sent - last_b) * 8 / dt if dt > 0 else 0.0
            self.samples.append(inst)
            last_t, last_b = now, sent
            mean = sent * 8 / (now - self.t0) if now > self.t0 else 0.0
            pct = 100.0 * sent / self.total_bytes if self.total_bytes else 0.0
            print(f"[{now - self.t0:7.1f}s] {_mb(sent)}/{_mb(self.total_bytes)} "
                  f"({pct:5.1f}%) | 瞬时 {inst / 1e6:7.1f} Mbps | 均值 {mean / 1e6:7.1f} Mbps "
                  f"| 在途 {inflight} | 完成 {done}/{self.total_files}", flush=True)

    def stop(self) -> None:
        self.t1 = time.monotonic()
        self._stop.set()

    def summary(self) -> str:
        wall = max(self.t1 - self.t0, 1e-9)
        sent, done, _ = self.snapshot()
        mean = sent * 8 / wall
        out = [f"共 {done} 个文件 / {_mb(sent)}，墙钟 {wall:.1f}s，均值 {mean / 1e6:.1f} Mbps"]
        if len(self.samples) >= 3:
            med = statistics.median(self.samples)
            idle = sum(1 for s in self.samples if s < 0.1 * mean) / len(self.samples)
            cv = statistics.pstdev(self.samples) / mean if mean > 0 else float("nan")
            out.append(f"带宽利用: 均值 {mean / 1e6:.1f} Mbps | 中位 {med / 1e6:.1f} Mbps "
                       f"| 空闲秒占比 {idle * 100:.1f}% | 变异系数 {cv:.2f}")
            out.append("（空闲秒 = 瞬时 < 均值 10% 的采样窗口；串行上传这个数很大，"
                       "并行后应趋近 0）")
        else:
            out.append("（采样窗口不足 3 个，波动指标略过）")
        return "\n".join(out)


def _mb(n: int) -> str:
    return f"{n / 1e6:.1f} MB" if n < 1e9 else f"{n / 1e9:.2f} GB"


class _CountingReader:
    """包一层 read()，边读边记账。`__len__` 让 requests 自己算 Content-Length
    （有它就不会退化成 chunked 传输编码）。"""

    def __init__(self, fp, size: int, meter: Meter):
        self._fp, self._left, self._meter = fp, size, meter

    def __len__(self) -> int:
        return self._left

    def read(self, n: int = -1) -> bytes:
        b = self._fp.read(n)
        self._left -= len(b)
        self._meter.add(len(b))
        return b


# ────────────────────────────────────────────────────────────── 上传


class Uploader:
    """每个 worker 线程持有：一个常驻 KaggleClient（token 往返复用连接）
    + 一个常驻 requests.Session（PUT 到 GCS 复用连接）。"""

    def __init__(self, api: KaggleApi, meter: Meter, retries: int, timeout: int):
        self.api = api
        self.meter = meter
        self.retries = retries
        self.timeout = timeout
        self._tl = threading.local()

    def _client(self):
        c = getattr(self._tl, "client", None)
        if c is None:
            c = self.api.build_kaggle_client()
            c.__enter__()                     # 全程持有：连接/TLS 只握手一次
            self._tl.client = c
        return c

    def _session(self) -> requests.Session:
        s = getattr(self._tl, "session", None)
        if s is None:
            s = requests.Session()
            adapter = HTTPAdapter(max_retries=Retry(total=5, backoff_factor=0.5,
                                                    status_forcelist=[502, 503, 504]))
            s.mount("https://", adapter)
            s.mount("http://", adapter)
            self._tl.session = s
        return s

    def close_thread_local(self) -> None:
        c = getattr(self._tl, "client", None)
        if c is not None:
            try:
                c.__exit__(None, None, None)
            except Exception:
                pass
            self._tl.client = None

    def upload(self, path: Path) -> str:
        """传一个文件，返回 blob token。失败抛异常（外层已重试 self.retries 次）。"""
        size = path.stat().st_size
        last_err: Optional[BaseException] = None
        for attempt in range(self.retries + 1):
            try:
                req = ApiStartBlobUploadRequest()
                req.type = ApiBlobType.DATASET
                req.name = path.name
                req.content_length = size
                req.last_modified_epoch_seconds = int(path.stat().st_mtime)
                resp = self._client().blobs.blob_api_client.start_blob_upload(req)

                self.meter.file_started()
                try:
                    with open(path, "rb", buffering=1 << 20) as fp:
                        reader = _CountingReader(fp, size, self.meter)
                        r = self._session().put(resp.create_url, data=reader,
                                                timeout=self.timeout)
                    if r.status_code not in (200, 201):
                        raise RuntimeError(f"PUT 回 {r.status_code}: {r.text[:200]}")
                finally:
                    self.meter.file_done()
                return resp.token
            except BaseException as e:                      # noqa: BLE001
                last_err = e
                # 重试从头开始（换一个新的 blob 会话）—— 断点续传要 308 探测，
                # 一次多一个 RTT，而这里文件都是几 MB 级，重传比探测便宜。
                if attempt < self.retries:
                    wait = 0.5 * 2 ** attempt
                    print(f"[WARN] {path.name} 第 {attempt + 1} 次失败（{type(e).__name__}: {e}），"
                          f"{wait:.1f}s 后重试", flush=True)
                    time.sleep(wait)
        raise RuntimeError(f"{path.name} 上传失败（重试 {self.retries} 次）") from last_err


def upload_all(api: KaggleApi, plan: Plan, workers: int, retries: int,
               timeout: int, report_sec: float, state: TokenState) -> Dict[str, str]:
    """并行传完所有文件，返回 {文件名: token}。任一文件最终失败即整体抛错
    （宁可不建 dataset，也不建一个缺文件的）。已有有效 token 的文件直接跳过。"""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    tokens: Dict[str, str] = {}
    todo: List[Path] = []
    for p in plan.files:
        t = state.get(p)
        if t:
            tokens[p.name] = t
        else:
            todo.append(p)
    if tokens:
        print(f"断点复用 {len(tokens)} 个文件的 token（{state.path}），需要传 {len(todo)} 个")
    if not todo:
        print("所有文件都已有有效 token，直接提交。")
        return tokens

    todo_bytes = sum(p.stat().st_size for p in todo)
    meter = Meter(todo_bytes, len(todo), report_sec)
    up = Uploader(api, meter, retries, timeout)

    reporter = threading.Thread(target=meter.run, daemon=True)
    reporter.start()
    errors: List[str] = []
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(up.upload, p): p for p in todo}
            for fut in as_completed(futs):
                p = futs[fut]
                try:
                    tok = fut.result()
                    tokens[p.name] = tok
                    state.put(p, tok)                  # 每成功一个就落盘
                except Exception as e:                 # noqa: BLE001
                    errors.append(f"{p.name}: {e}")
    finally:
        meter.stop()
        reporter.join(timeout=2 * report_sec)

    print("\n" + meter.summary(), flush=True)
    if errors:
        raise SystemExit("[FATAL] 以下文件没传成功，已放弃建库（不会留下缺文件的 dataset）：\n  "
                         + "\n  ".join(errors) +
                         f"\n  已成功的 token 存在 {state.path}，重跑本命令会只补传缺的。")
    return tokens


# ────────────────────────────────────────────────────────────── 建库 / 建版本


def _new_files(plan: Plan, tokens: Dict[str, str]) -> List[ApiDatasetNewFile]:
    out = []
    for p in plan.files:                       # 保持目录序，结果可复现
        f = ApiDatasetNewFile()
        f.token = tokens[p.name]
        out.append(f)
    return out


def preflight_create(api: KaggleApi, plan: Plan) -> None:
    """`--mode create` 的**上传前**校验。传完 2GB 才发现「名字被占」是最贵的
    失败方式，所以能在本地判的全放这儿，能问服务端的先问一次。"""
    meta = plan.meta
    ref = meta.get("id") or ""
    title = meta.get("title") or ""
    licenses = meta.get("licenses") or []
    owner_slug, _, dataset_slug = ref.partition("/")
    if not owner_slug or not dataset_slug:
        raise SystemExit(f"[FATAL] metadata 的 id 要写成 <owner>/<slug>，收到 {ref!r}")
    if len(licenses) != 1:
        raise SystemExit("[FATAL] licenses 必须恰好一项")
    if not (6 <= len(dataset_slug) <= 50):
        raise SystemExit("[FATAL] dataset slug 长度必须在 6~50")
    if not (6 <= len(title) <= 50):
        raise SystemExit("[FATAL] dataset title 长度必须在 6~50")

    # 服务端占用探测。注意 Kaggle 对**不存在**的 dataset 也回 403（实测：
    # 已存在的回 "ready"，不存在的回 403 Forbidden），所以这里只能判「已存在」，
    # 判不出「slug 被保留但读不到」那种状态 —— 那种只有 create 时才会撞上。
    try:
        status = api.dataset_status(ref)
    except Exception:                                          # noqa: BLE001
        status = None
    if status is not None:
        raise SystemExit(
            f"[FATAL] {ref} 已存在（状态 {status}）。要更新它请改用 "
            f"--mode version -m \"<说明>\"，别用 --mode create。")


def do_create(api: KaggleApi, plan: Plan, tokens: Dict[str, str], public: bool):
    meta = plan.meta
    title = meta["title"]
    licenses = meta.get("licenses") or []
    owner_slug, _, dataset_slug = meta["id"].partition("/")

    req = ApiCreateDatasetRequest()
    req.title = title
    req.slug = dataset_slug
    req.owner_slug = owner_slug
    req.license_name = licenses[0]["name"]
    req.subtitle = meta.get("subtitle")
    req.description = meta.get("description")
    req.files = _new_files(plan, tokens)
    req.is_private = not public
    req.category_ids = meta.get("keywords", [])
    with api.build_kaggle_client() as kaggle:
        return api.with_retry(kaggle.datasets.dataset_api_client.create_dataset)(req)


def do_version(api: KaggleApi, plan: Plan, tokens: Dict[str, str], notes: str,
               delete_old: bool):
    meta = plan.meta
    owner_slug, _, dataset_slug = meta["id"].partition("/")
    body = ApiCreateDatasetVersionRequestBody()
    body.version_notes = notes
    body.subtitle = meta.get("subtitle")
    body.description = meta.get("description")
    body.files = _new_files(plan, tokens)
    body.category_ids = meta.get("keywords", [])
    body.delete_old_versions = delete_old

    req = ApiCreateDatasetVersionRequest()
    req.owner_slug = owner_slug
    req.dataset_slug = dataset_slug
    req.body = body
    with api.build_kaggle_client() as kaggle:
        return api.with_retry(kaggle.datasets.dataset_api_client.create_dataset_version)(req)


# ────────────────────────────────────────────────────────────── main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="并行 Kaggle Dataset 上传器（把上行带宽跑成平线）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="要上传的目录（里面要有 dataset-metadata.json）")
    ap.add_argument("--mode", choices=("create", "version"), required=True,
                    help="create=新建数据集；version=给已有数据集传新版本")
    ap.add_argument("-m", "--version-notes", default="",
                    help="--mode version 的版本说明")
    ap.add_argument("--delete-old-versions", action="store_true",
                    help="--mode version：建完新版本后删掉旧版本")
    ap.add_argument("--public", action="store_true",
                    help="--mode create：建成公开数据集（**默认私有**）")
    ap.add_argument("--workers", type=int, default=8,
                    help="并发上传的文件数（默认 8）")
    ap.add_argument("--retries", type=int, default=3, help="单文件重试次数")
    ap.add_argument("--timeout", type=int, default=300, help="单次 PUT 超时秒数")
    ap.add_argument("--report-sec", type=float, default=1.0, help="吞吐采样间隔")
    ap.add_argument("--allow-ext", default="",
                    help="放行默认禁传的扩展名，逗号分隔，如 .png,.txt")
    ap.add_argument("--allow-npz-caption", action="store_true",
                    help="放行内部带 caption 明文的 npz")
    ap.add_argument("--dry-run", action="store_true",
                    help="只扫描 + 打印计划与闸门结果，一个字节都不发")
    ap.add_argument("--state-file", default="",
                    help="blob token 断点文件（默认在系统临时目录，按目录路径哈希命名）。"
                         "提交那一跳失败时重跑本命令就不用重传")
    ap.add_argument("--token-ttl-hours", type=float, default=6.0,
                    help="断点里的 token 多久算过期（默认 6 小时）")
    ap.add_argument("--fresh", action="store_true",
                    help="忽略并清空断点文件，全部重传")
    a = ap.parse_args(argv)

    folder = Path(a.folder).resolve()
    if not folder.is_dir():
        raise SystemExit(f"[FATAL] 不是目录：{folder}")

    plan = scan(folder)
    allow_ext = {e.strip().lower() for e in a.allow_ext.split(",") if e.strip()}
    gate_extensions(plan.files, allow_ext)
    gate_npz_caption(plan.files, a.allow_npz_caption)

    by_ext: Dict[str, int] = {}
    for p in plan.files:
        by_ext[p.suffix.lower() or "<无扩展名>"] = by_ext.get(p.suffix.lower() or "<无扩展名>", 0) + 1
    print(f"目录 {folder}")
    print(f"数据集 {plan.meta.get('id')}  模式 {a.mode}"
          + ("（公开）" if a.public else "（私有）"))
    print(f"文件 {len(plan.files)} 个 / {_mb(plan.total_bytes)}  "
          f"构成 {dict(sorted(by_ext.items()))}")
    print(f"闸门通过：无禁传扩展名，npz 内无 caption 明文")
    print(f"并发 {a.workers} | 重试 {a.retries} | 采样 {a.report_sec}s")
    if a.dry_run:
        print("\n--dry-run：到此为止，没有发送任何字节。")
        return 0

    if a.mode == "version" and not a.version_notes:
        raise SystemExit("[FATAL] --mode version 需要 -m/--version-notes")

    api = KaggleApi()
    api.authenticate()
    if a.mode == "create":
        preflight_create(api, plan)

    state = TokenState(Path(a.state_file) if a.state_file else TokenState.default_path(folder),
                       ttl_sec=a.token_ttl_hours * 3600)
    if a.fresh:
        state.clear()

    print()
    tokens = upload_all(api, plan, a.workers, a.retries, a.timeout, a.report_sec, state)

    print("\n提交…", flush=True)
    if a.mode == "create":
        resp = do_create(api, plan, tokens, a.public)
    else:
        resp = do_version(api, plan, tokens, a.version_notes, a.delete_old_versions)

    err = getattr(resp, "error", None)
    if err:
        print(f"[FATAL] Kaggle 返回错误：{err}")
        print(f"  文件都已传完，token 存在 {state.path}（{a.token_ttl_hours}h 内有效）。"
              f"\n  改掉 metadata 里冲突的字段后重跑同一条命令，只会重发提交那一跳。")
        return 1
    state.clear()                                  # 提交成功，断点没用了
    url = getattr(resp, "url", "") or f"https://www.kaggle.com/datasets/{plan.meta.get('id')}"
    print(f"完成：{url}")
    print(f"注意：还要等它转 ready 才能挂进 kernel —— "
          f"python -m kaggle datasets status {plan.meta.get('id')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
