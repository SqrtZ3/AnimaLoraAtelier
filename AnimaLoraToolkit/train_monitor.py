"""
训练监控服务器
实时显示 loss 曲线和采样图片

线程模型：
- 训练主线程通过 update_monitor() 写入 MONITOR_STATE。
- ThreadingHTTPServer 在多个 handler 线程里调 get_state() 读取并 JSON 序列化。
- 二者共享 _STATE_LOCK；list 改用 collections.deque(maxlen=N) 避免每步 O(n) memcpy。

★ 关于 state.json:
旧版本会在每个 update_monitor() 把整个 MONITOR_STATE（含最多 50000 个 loss 点）
序列化写到 monitor_data/state.json。但搜遍仓库这个文件**从未被读取**：
  - HTTP /api/state 直接从内存 MONITOR_STATE 读
  - 断点续训的 monitor 历史走 training_state_step{N}.pt 里的 monitor_state 字段
  - 没有任何 json.load(state.json) 路径

所以 state.json 是纯写入孤儿文件。最干净的解决方案 = **不写**。
保留 save_state() / shutdown 钩子作为 API 兼容（旧脚本若调过 save_state 不会崩），
但实际是 no-op。
"""
import json
import threading
import time
from collections import deque
from pathlib import Path
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import webbrowser
from urllib.parse import urlparse, parse_qs

# 历史保留点数上限（loss/lr/samples）
_MAX_LOSS_POINTS = 50000
_MAX_LR_POINTS = 50000
_MAX_SAMPLES = 50

# 全局状态。list 改成 deque(maxlen)：append 与左侧弹出都是 O(1)，省每步 [-50000:] 的 memcpy。
# samples 仍可用 list（变更频率低，最多 50 个）。
MONITOR_STATE = {
    "losses": deque(maxlen=_MAX_LOSS_POINTS),
    "lr_history": deque(maxlen=_MAX_LR_POINTS),
    "epoch": 0,
    "step": 0,
    "total_steps": 0,
    "speed": 0.0,
    "samples": [],
    "start_time": None,
    "config": {},
}

# 训练线程写 / handler 线程读 共享锁。Python list/dict 的单操作大多是原子的，但
# HTTP 线程 JSON serialize 时可能撞到 deque mutated；用 lock + deque 双保险。
_STATE_LOCK = threading.RLock()


def _state_snapshot_for_serialization():
    """在 lock 下生成一个可 JSON serialize 的快照（deque → list）。"""
    with _STATE_LOCK:
        return {
            "losses": list(MONITOR_STATE["losses"]),
            "lr_history": list(MONITOR_STATE["lr_history"]),
            "epoch": MONITOR_STATE["epoch"],
            "step": MONITOR_STATE["step"],
            "total_steps": MONITOR_STATE["total_steps"],
            "speed": MONITOR_STATE["speed"],
            "samples": list(MONITOR_STATE["samples"]),
            "start_time": MONITOR_STATE["start_time"],
            "config": dict(MONITOR_STATE["config"]) if MONITOR_STATE["config"] else {},
        }


def update_monitor(loss=None, lr=None, epoch=None, step=None, total_steps=None, speed=None, sample_path=None, config=None):
    """更新监控状态（线程安全；纯内存，无 disk IO）。"""
    with _STATE_LOCK:
        # 先更新 step/epoch 等，使本次写入的 loss/lr 点位正确
        if epoch is not None:
            MONITOR_STATE["epoch"] = epoch
        if step is not None:
            MONITOR_STATE["step"] = step
        if total_steps is not None:
            MONITOR_STATE["total_steps"] = total_steps
        if speed is not None:
            MONITOR_STATE["speed"] = speed

        if loss is not None:
            MONITOR_STATE["losses"].append({"step": MONITOR_STATE["step"], "loss": loss, "time": time.time()})
        if lr is not None:
            MONITOR_STATE["lr_history"].append({"step": MONITOR_STATE["step"], "lr": lr})
        if sample_path is not None:
            MONITOR_STATE["samples"].append({"path": str(sample_path), "step": MONITOR_STATE["step"], "time": time.time()})
            if len(MONITOR_STATE["samples"]) > _MAX_SAMPLES:
                # samples 仍是 list（采样很少触发），直接截断
                MONITOR_STATE["samples"] = MONITOR_STATE["samples"][-_MAX_SAMPLES:]
        if config is not None:
            MONITOR_STATE["config"] = config

        if MONITOR_STATE["start_time"] is None:
            MONITOR_STATE["start_time"] = time.time()


def save_state(force: bool = False):
    """No-op，保留作为 API 兼容入口。

    旧版本会把 MONITOR_STATE 写到 monitor_data/state.json，但该文件从未被读取（HTTP API
    / 断点续训都走其它路径），属于纯写入孤儿。已移除磁盘 IO 以减少长训练的持续写盘开销。
    若未来需要 forensic dump，可显式调用 get_state() 自行序列化。
    """
    return None


def get_state():
    """获取当前状态（用于 HTTP handler；deque 转 list 以便 JSON 序列化）。"""
    return _state_snapshot_for_serialization()


def restore_monitor_state(losses=None, lr_history=None, epoch=None, step=None, total_steps=None, start_time=None, config=None):
    """恢复监控状态（用于断点续训）。

    Args:
        losses: 历史 loss 列表，格式 [{"step": int, "loss": float, "time": float}, ...]
        lr_history: 历史 lr 列表，格式 [{"step": int, "lr": float}, ...]
        epoch, step, total_steps: 训练进度
        start_time: 训练开始时间
        config: 配置字典
    """
    with _STATE_LOCK:
        if losses is not None:
            MONITOR_STATE["losses"] = deque(losses, maxlen=_MAX_LOSS_POINTS)
        if lr_history is not None:
            MONITOR_STATE["lr_history"] = deque(lr_history, maxlen=_MAX_LR_POINTS)
        if epoch is not None:
            MONITOR_STATE["epoch"] = epoch
        if step is not None:
            MONITOR_STATE["step"] = step
        if total_steps is not None:
            MONITOR_STATE["total_steps"] = total_steps
        if start_time is not None:
            MONITOR_STATE["start_time"] = start_time
        if config is not None:
            MONITOR_STATE["config"] = config


def _downsample_uniform(points, target_points: int):
    """均匀降采样到 target_points（保留首尾，适合 loss/lr 长序列）"""
    if not isinstance(target_points, int) or target_points <= 0:
        return points
    n = len(points)
    if n <= target_points:
        return points
    if target_points == 1:
        return [points[-1]]
    step = (n - 1) / (target_points - 1)
    out = []
    for i in range(target_points):
        idx = round(i * step)
        out.append(points[idx])
    return out


# HTML 页面
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Anima Training Monitor</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { 
            text-align: center; 
            margin-bottom: 20px;
            background: linear-gradient(90deg, #00d4ff, #7c3aed);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2em;
        }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
        .card {
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
        }
        .card h2 { 
            font-size: 1.1em; 
            margin-bottom: 15px; 
            color: #00d4ff;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; }
        .stat-item {
            background: rgba(0,212,255,0.1);
            border-radius: 12px;
            padding: 15px;
            text-align: center;
        }
        .stat-value { font-size: 1.8em; font-weight: bold; color: #00d4ff; }
        .stat-label { font-size: 0.85em; color: #888; margin-top: 5px; }
        .chart-container { height: 300px; position: relative; }
        .samples-grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); 
            gap: 15px;
        }
        .sample-item {
            background: rgba(0,0,0,0.3);
            border-radius: 12px;
            overflow: hidden;
            transition: transform 0.2s;
        }
        .sample-item:hover { transform: scale(1.02); }
        .sample-item img { 
            width: 100%; 
            height: 200px; 
            object-fit: cover;
        }
        .sample-info {
            padding: 10px;
            font-size: 0.85em;
            color: #888;
        }
        .progress-bar {
            height: 8px;
            background: rgba(255,255,255,0.1);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 10px;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #00d4ff, #7c3aed);
            transition: width 0.3s;
        }
        .config-list {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            font-size: 0.9em;
        }
        .config-item {
            display: flex;
            justify-content: space-between;
            padding: 8px 12px;
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
        }
        .config-key { color: #888; }
        .config-value { color: #00d4ff; font-weight: 500; }
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #00ff88;
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .full-width { grid-column: 1 / -1; }
        @media (max-width: 900px) {
            .grid { grid-template-columns: 1fr; }
            .stats-grid { grid-template-columns: repeat(2, 1fr); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎨 Anima Training Monitor</h1>
        
        <div class="stats-grid" style="margin-bottom: 20px;">
            <div class="stat-item">
                <div class="stat-value" id="epoch">-</div>
                <div class="stat-label">Epoch</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="step">-</div>
                <div class="stat-label">Step</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="loss">-</div>
                <div class="stat-label">Loss</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="speed">-</div>
                <div class="stat-label">Speed (it/s)</div>
            </div>
        </div>
        
        <div class="progress-bar">
            <div class="progress-fill" id="progress" style="width: 0%"></div>
        </div>
        <p style="text-align: center; margin: 10px 0; color: #888;" id="progress-text">等待训练开始...</p>
        
        <div class="grid">
            <div class="card">
                <h2><span class="status-dot"></span> Loss 曲线 <span style="font-size:0.7em;color:#00ff88;margin-left:10px">绿色=平滑趋势</span></h2>
                <div class="chart-container">
                    <canvas id="lossChart"></canvas>
                </div>
            </div>
            
            <div class="card">
                <h2>📊 Learning Rate</h2>
                <div class="chart-container">
                    <canvas id="lrChart"></canvas>
                </div>
            </div>
            
            <div class="card full-width">
                <h2>🖼️ 采样预览</h2>
                <div class="samples-grid" id="samples">
                    <p style="color: #666;">等待采样...</p>
                </div>
            </div>
            
            <div class="card full-width">
                <h2>⚙️ 训练配置</h2>
                <div class="config-list" id="config">
                    <p style="color: #666;">加载中...</p>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // 图表配置
        const chartOptions = {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 0 },
            scales: {
                x: { 
                    grid: { color: 'rgba(255,255,255,0.1)' },
                    ticks: { color: '#888' }
                },
                y: { 
                    grid: { color: 'rgba(255,255,255,0.1)' },
                    ticks: { color: '#888' }
                }
            },
            plugins: {
                legend: { display: false }
            }
        };
        
        const lossChart = new Chart(document.getElementById('lossChart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: '原始',
                        data: [],
                        borderColor: 'rgba(0,212,255,0.3)',
                        backgroundColor: 'rgba(0,212,255,0.05)',
                        fill: true,
                        tension: 0.1,
                        pointRadius: 0,
                        borderWidth: 1
                    },
                    {
                        label: '平滑 (EMA)',
                        data: [],
                        borderColor: '#00ff88',
                        backgroundColor: 'transparent',
                        fill: false,
                        tension: 0.4,
                        pointRadius: 0,
                        borderWidth: 2
                    }
                ]
            },
            options: {
                ...chartOptions,
                plugins: {
                    legend: { 
                        display: true,
                        labels: { color: '#888', boxWidth: 12 }
                    }
                }
            }
        });
        
        // 计算 EMA 平滑
        function calcEMA(data, alpha = 0.05) {
            if (data.length === 0) return [];
            const ema = [data[0]];
            for (let i = 1; i < data.length; i++) {
                ema.push(alpha * data[i] + (1 - alpha) * ema[i - 1]);
            }
            return ema;
        }
        
        const lrChart = new Chart(document.getElementById('lrChart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    data: [],
                    borderColor: '#7c3aed',
                    backgroundColor: 'rgba(124,58,237,0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0
                }]
            },
            options: chartOptions
        });
        
        // 更新函数
        async function updateData() {
            try {
                const resp = await fetch('/api/state?' + Date.now());
                const data = await resp.json();
                
                // 更新统计
                document.getElementById('epoch').textContent = data.epoch || 0;
                document.getElementById('step').textContent = data.step || 0;
                document.getElementById('speed').textContent = (data.speed || 0).toFixed(2);
                
                // Loss
                if (data.losses && data.losses.length > 0) {
                    const lastLoss = data.losses[data.losses.length - 1].loss;
                    document.getElementById('loss').textContent = lastLoss.toFixed(4);
                    
                    // 更新图表（最多显示 500 个点）
                    const displayLosses = data.losses.slice(-500);
                    const rawLosses = displayLosses.map(l => l.loss);
                    const smoothLosses = calcEMA(rawLosses, 0.02);  // alpha=0.02 更平滑
                    
                    lossChart.data.labels = displayLosses.map(l => l.step);
                    lossChart.data.datasets[0].data = rawLosses;      // 原始曲线
                    lossChart.data.datasets[1].data = smoothLosses;   // 平滑曲线
                    lossChart.update('none');
                    
                    // 显示平滑后的趋势（最近 100 步 vs 之前 100 步）
                    if (smoothLosses.length >= 200) {
                        const recent = smoothLosses.slice(-100).reduce((a,b) => a+b, 0) / 100;
                        const before = smoothLosses.slice(-200, -100).reduce((a,b) => a+b, 0) / 100;
                        const trend = ((recent - before) / before * 100).toFixed(2);
                        const trendText = trend < 0 ? `↓${Math.abs(trend)}%` : `↑${trend}%`;
                        const trendColor = trend < 0 ? '#00ff88' : '#ff6b6b';
                        document.getElementById('loss').innerHTML = 
                            `${lastLoss.toFixed(4)} <span style="font-size:0.5em;color:${trendColor}">${trendText}</span>`;
                    }
                }
                
                // LR
                if (data.lr_history && data.lr_history.length > 0) {
                    const displayLr = data.lr_history.slice(-500);
                    lrChart.data.labels = displayLr.map(l => l.step);
                    lrChart.data.datasets[0].data = displayLr.map(l => l.lr);
                    lrChart.update('none');
                }
                
                // 进度
                if (data.total_steps > 0) {
                    const pct = Math.min(100, (data.step / data.total_steps) * 100);
                    document.getElementById('progress').style.width = pct + '%';
                    
                    const elapsed = data.start_time ? (Date.now()/1000 - data.start_time) : 0;
                    const eta = data.speed > 0 ? (data.total_steps - data.step) / data.speed : 0;
                    document.getElementById('progress-text').textContent = 
                        `${pct.toFixed(1)}% | 已用: ${formatTime(elapsed)} | 预计剩余: ${formatTime(eta)}`;
                }
                
                // 采样图片
                if (data.samples && data.samples.length > 0) {
                    const samplesHtml = data.samples.slice(-6).reverse().map(s => `
                        <div class="sample-item">
                            <img src="/samples/${s.path.split(/[\\\\/]/).pop()}" onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22/>'">
                            <div class="sample-info">Step ${s.step}</div>
                        </div>
                    `).join('');
                    document.getElementById('samples').innerHTML = samplesHtml;
                }
                
                // 配置
                if (data.config && Object.keys(data.config).length > 0) {
                    const configHtml = Object.entries(data.config).map(([k, v]) => `
                        <div class="config-item">
                            <span class="config-key">${k}</span>
                            <span class="config-value">${v}</span>
                        </div>
                    `).join('');
                    document.getElementById('config').innerHTML = configHtml;
                }
                
            } catch (e) {
                console.log('Update failed:', e);
            }
        }
        
        function formatTime(seconds) {
            if (!seconds || seconds < 0) return '--:--';
            const h = Math.floor(seconds / 3600);
            const m = Math.floor((seconds % 3600) / 60);
            const s = Math.floor(seconds % 60);
            if (h > 0) return `${h}h ${m}m`;
            return `${m}m ${s}s`;
        }
        
        // 每秒更新
        setInterval(updateData, 1000);
        updateData();
    </script>
</body>
</html>
"""
# VRAM 查询缓存：浏览器 + 多客户端每秒 hit /api/state，重复查询 mem_get_info 触发 CUDA 同步。
# 用一个 500ms cache 把 N/秒收敛到 ≤2/秒，对实时性影响微乎其微。
_VRAM_CACHE = {"data": None, "ts": 0.0}
_VRAM_TTL = 0.5  # seconds


def get_vram_info():
    """获取 CUDA VRAM 信息（带 500ms cache）"""
    now = time.time()
    if _VRAM_CACHE["data"] is not None and (now - _VRAM_CACHE["ts"]) < _VRAM_TTL:
        return _VRAM_CACHE["data"]
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            free, total = torch.cuda.mem_get_info(device)
            free_mb = free // (1024 * 1024)
            total_mb = total // (1024 * 1024)
            used_mb = total_mb - free_mb
            result = {
                "free": free_mb,
                "total": total_mb,
                "used": used_mb,
                "percentage": round(used_mb / total_mb * 100, 1) if total_mb > 0 else 0,
            }
            _VRAM_CACHE["data"] = result
            _VRAM_CACHE["ts"] = now
            return result
    except Exception:
        pass
    return None


def _read_log_tail(log_file: Path, max_lines: int = 100, max_bytes: int = 256 * 1024) -> list:
    """读 log 文件末尾 max_lines 行，最多读 max_bytes 字节。

    旧实现 readlines() 把整个 log 文件读进内存 → 长训练 log 数十 MB 时，每秒
    /api/logs 都触发一次全文读 + 字符串切片，主机 IO + Python 解析都被占满。
    新实现 seek 到末尾倒读最多 256KB，足够覆盖 100 行（每行 ~200 字节是典型）。
    """
    try:
        size = log_file.stat().st_size
        with open(log_file, "rb") as f:
            offset = max(size - max_bytes, 0)
            f.seek(offset)
            chunk = f.read()
        text = chunk.decode("utf-8", errors="ignore")
        lines = text.splitlines()
        # 第一行可能因 seek 中截断，丢掉；除非我们就是从文件起始读的
        if offset > 0 and lines:
            lines = lines[1:]
        return [ln.strip() for ln in lines[-max_lines:]]
    except Exception:
        return []


class MonitorHandler(SimpleHTTPRequestHandler):
    """监控服务器 Handler"""
    
    def __init__(self, *args, output_dir=None, **kwargs):
        self.output_dir = output_dir or Path("./output")
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            
            # 优先使用 monitor_smooth.html
            smooth_html = Path(__file__).resolve().parent / "monitor_smooth.html"
            if smooth_html.exists():
                with open(smooth_html, "r", encoding="utf-8") as f:
                    content = f.read()
                self.wfile.write(content.encode("utf-8"))
            else:
                print(f"[Monitor] Warning: smooth UI not found at {smooth_html}, using fallback.")
                self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
        elif self.path.startswith("/api/state"):
            # 支持 query 参数：max_points（对 losses/lr_history 降采样，降低传输和前端渲染压力）
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query or "")
            try:
                max_points = int(qs.get("max_points", ["0"])[0] or 0)
            except Exception:
                max_points = 0

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            state = get_state()
            vram = get_vram_info()
            if vram:
                state["vram"] = vram
            if max_points > 0:
                # 不修改全局状态，只对返回值裁剪
                if "losses" in state:
                    state["losses"] = _downsample_uniform(state["losses"], max_points)
                if "lr_history" in state:
                    state["lr_history"] = _downsample_uniform(state["lr_history"], max_points)
            self.wfile.write(json.dumps(state).encode("utf-8"))
        elif self.path.startswith("/api/logs"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            logs = []
            log_file = Path("anima_training.log")
            if not log_file.exists():
                log_file = Path(__file__).resolve().parent.parent / "anima_training.log"

            if log_file.exists():
                # tail 读法（seek 到末尾 256KB），比每秒 readlines() 全文快几十倍
                logs = _read_log_tail(log_file, max_lines=100, max_bytes=256 * 1024)
            self.wfile.write(json.dumps({"logs": logs}).encode("utf-8"))
        elif self.path.startswith("/samples/"):
            # 提供采样图片
            filename = self.path.split("/")[-1]
            sample_path = self.output_dir / "samples" / filename
            if sample_path.exists():
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.end_headers()
                with open(sample_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404)
        else:
            self.send_error(404)
    
    def log_message(self, format, *args):
        pass  # 静默日志


def start_monitor_server(port=8765, host="0.0.0.0", output_dir=None, open_browser=True, max_port_retries=5):
    """启动监控服务器。

    使用 ThreadingHTTPServer 而非 HTTPServer：浏览器同时请求 /api/state + /api/logs +
    多张 /samples/*.png 时，旧 HTTPServer 是单线程串行 → 一个慢请求阻塞其它。
    Threading 版每个请求一个 worker thread。

    端口被占时自动 +1 重试，最多 max_port_retries 次（避免训练因端口冲突直接失败）。
    """
    output_dir = Path(output_dir) if output_dir else Path("./output")

    def handler(*args, **kwargs):
        return MonitorHandler(*args, output_dir=output_dir, **kwargs)

    server = None
    last_err = None
    actual_port = port
    for attempt in range(max(int(max_port_retries), 1)):
        try_port = port + attempt
        try:
            server = ThreadingHTTPServer((host, try_port), handler)
            actual_port = try_port
            break
        except OSError as e:
            last_err = e
            continue

    if server is None:
        raise RuntimeError(
            f"[Monitor] 端口 {port}..{port + max_port_retries - 1} 都被占用，监控面板启动失败: {last_err}"
        )

    # 提示端口被改写
    if actual_port != port:
        print(f"[Monitor] 端口 {port} 被占用，自动切换到 {actual_port}")
        port = actual_port

    def run():
        shown_host = "localhost" if host in ("0.0.0.0", "127.0.0.1") else host
        try:
            print(f"[Monitor] 训练监控面板: http://{shown_host}:{port}")
        except UnicodeEncodeError:
            try:
                print(f"[Monitor] Training Monitor: http://{shown_host}:{port}")
            except Exception:
                pass
        except Exception:
            pass
        server.serve_forever()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    if open_browser:
        time.sleep(0.5)
        webbrowser.open(f"http://{('localhost' if host in ('0.0.0.0','127.0.0.1') else host)}:{port}")

    # 暴露端口给调用方（自动 fallback 时主程序需要知道实际端口）
    server.actual_port = port
    return server


def shutdown_monitor_server(server):
    """优雅关闭监控服务器（signal handler 与训练正常结束时都应调用）。"""
    if server is None:
        return
    try:
        server.shutdown()
    except Exception:
        pass
    try:
        server.server_close()
    except Exception:
        pass


if __name__ == "__main__":
    # 测试模式
    import random
    
    server = start_monitor_server(port=8765)
    
    print("测试模式：模拟训练数据...")
    for i in range(1000):
        update_monitor(
            loss=0.5 * (0.95 ** (i / 10)) + random.random() * 0.05,
            lr=1e-4 * (0.99 ** (i / 50)),
            epoch=i // 100 + 1,
            step=i,
            total_steps=1000,
            speed=2.5 + random.random() * 0.5,
            config={
                "model": "Anima LoKr",
                "rank": 64,
                "epochs": 10,
                "batch_size": 4,
            }
        )
        time.sleep(0.1)
