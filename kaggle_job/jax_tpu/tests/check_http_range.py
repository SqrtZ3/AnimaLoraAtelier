"""HTTP Range 加载路径的本地对拍（krea2_jax._read_safetensors_map）。

背景：Krea-2-Raw 26GB 装不下 Kaggle /kaggle/working（~20GB），底模改为
HTTP Range 流式加载。safetensors 是纯字节寻址格式，Range 读出的字节与
本地文件必须**逐 bit 相同**——本测试起一个支持 Range 与 302 的本地 HTTP
server，对同一份文件分别走两条加载路径逐 tensor 对拍。

用法（jax 侧解释器）：
    python tests/check_http_range.py
"""
from __future__ import annotations

import http.server
import re
import struct
import sys
import threading
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import krea2_jax as K2                                        # noqa: E402

SERVE_DIR = HERE / "_ref"                # export_test.safetensors 所在目录


class RangeHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler 不支持 Range —— 补上，外加 /redir 302 跳转
    （模拟 HF resolve -> CDN 的重定向链，验证最终 URL 解析与 token 剥除）。"""

    def do_GET(self):
        if self.path.startswith("/redir"):
            self.send_response(302)
            self.send_header("Location", "/export_test.safetensors")
            self.end_headers()
            return
        rng = self.headers.get("Range")
        if not rng:
            return super().do_GET()
        m = re.fullmatch(r"bytes=(\d+)-(\d*)", rng.strip())
        if not m:
            self.send_error(416)
            return
        start = int(m.group(1))
        data = (SERVE_DIR / "export_test.safetensors").read_bytes()
        end = int(m.group(2)) + 1 if m.group(2) else len(data)
        chunk = data[start:end]
        self.send_response(206)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Range",
                         f"bytes {start}-{start + len(chunk) - 1}/{len(data)}")
        self.send_header("Content-Length", str(len(chunk)))
        self.end_headers()
        self.wfile.write(chunk)

    def log_message(self, *args):        # 安静
        pass


def main() -> int:
    src = SERVE_DIR / "export_test.safetensors"
    if not src.exists():
        print(f"缺对拍文件 {src}")
        return 1

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    entries_f, meta_f = K2._read_safetensors_map(str(src))
    n_ok = 0
    try:
        for prefix, label in ((f"http://127.0.0.1:{port}/export_test.safetensors",
                               "直链"),
                              (f"http://127.0.0.1:{port}/redir", "302 重定向")):
            entries_h, meta_h = K2._read_safetensors_map(prefix)
            assert set(entries_h) == set(entries_f), "键集不一致"
            assert meta_h == meta_f, f"__metadata__ 不一致: {meta_h} vs {meta_f}"
            for name in entries_f:
                a = entries_f[name][2]()          # 文件路径读
                b = entries_h[name][2]()          # HTTP Range 读
                assert a.dtype == b.dtype and a.shape == b.shape, name
                assert np.array_equal(a.tobytes(), b.tobytes()), \
                    f"{name} 字节不一致（{label}）"
                n_ok += 1
            print(f"[{label}] {len(entries_f)} 个 tensor 全部逐 bit 一致 "
                  f"（含 __metadata__ 对拍）")

        # 大 range 与边界：整文件一次 range 读（等价于不落盘的"全量下载"）
        size = src.stat().st_size
        body = K2._http_range_get(f"http://127.0.0.1:{port}/export_test.safetensors",
                                  0, size)
        assert body == src.read_bytes(), "整文件 range 读字节不一致"
        print(f"[整文件 range] {size} 字节逐 bit 一致；tensor 级对拍共 {n_ok} 次")
    finally:
        httpd.shutdown()

    print("OK —— HTTP Range 路径与文件路径逐 bit 等价")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
