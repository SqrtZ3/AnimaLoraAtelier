"""
训练监控服务器
实时显示 loss 曲线和采样图片
"""
import json
import os
import threading
import time
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import webbrowser
from urllib.parse import urlparse, parse_qs

# 全局状态
MONITOR_STATE = {
    "losses": [],
    "lr_history": [],
    "epoch": 0,
    "step": 0,
    "total_steps": 0,
    "speed": 0.0,
    "samples": [],
    "start_time": None,
    "config": {},
}

MONITOR_DIR = Path(__file__).resolve().parent / "monitor_data"
MONITOR_DIR.mkdir(exist_ok=True)


def update_monitor(loss=None, lr=None, epoch=None, step=None, total_steps=None, speed=None, sample_path=None, config=None):
    """更新监控状态"""
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
        # 保留最近 50000 个点（支持长时间训练）
        if len(MONITOR_STATE["losses"]) > 50000:
            MONITOR_STATE["losses"] = MONITOR_STATE["losses"][-50000:]
    
    if lr is not None:
        MONITOR_STATE["lr_history"].append({"step": MONITOR_STATE["step"], "lr": lr})
        if len(MONITOR_STATE["lr_history"]) > 50000:
            MONITOR_STATE["lr_history"] = MONITOR_STATE["lr_history"][-50000:]
    if sample_path is not None:
        MONITOR_STATE["samples"].append({"path": str(sample_path), "step": MONITOR_STATE["step"], "time": time.time()})
        # 只保留最近 50 张
        if len(MONITOR_STATE["samples"]) > 50:
            MONITOR_STATE["samples"] = MONITOR_STATE["samples"][-50:]
    if config is not None:
        MONITOR_STATE["config"] = config
    
    if MONITOR_STATE["start_time"] is None:
        MONITOR_STATE["start_time"] = time.time()
    
    # 写入 JSON 文件
    save_state()


def save_state():
    """保存状态到 JSON"""
    state_file = MONITOR_DIR / "state.json"
    try:
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(MONITOR_STATE, f)
    except Exception:
        pass


def get_state():
    """获取当前状态"""
    return MONITOR_STATE.copy()


def restore_monitor_state(losses=None, lr_history=None, epoch=None, step=None, total_steps=None, start_time=None, config=None):
    """恢复监控状态（用于断点续训）
    
    Args:
        losses: 历史 loss 列表，格式 [{"step": int, "loss": float, "time": float}, ...]
        lr_history: 历史 lr 列表，格式 [{"step": int, "lr": float}, ...]
        epoch, step, total_steps: 训练进度
        start_time: 训练开始时间
        config: 配置字典
    """
    if losses is not None:
        MONITOR_STATE["losses"] = losses
    if lr_history is not None:
        MONITOR_STATE["lr_history"] = lr_history
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
    save_state()


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
def get_vram_info():
    """获取 CUDA VRAM 信息"""
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            free, total = torch.cuda.mem_get_info(device)
            free_mb = free // (1024 * 1024)
            total_mb = total // (1024 * 1024)
            used_mb = total_mb - free_mb
            return {
                "free": free_mb,
                "total": total_mb,
                "used": used_mb,
                "percentage": round(used_mb / total_mb * 100, 1) if total_mb > 0 else 0
            }
    except Exception:
        pass
    return None


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
                try:
                    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                        lines = f.readlines()
                        logs = [line.strip() for line in lines[-100:]]
                except Exception:
                    pass
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


def start_monitor_server(port=8765, host="127.0.0.1", output_dir=None, open_browser=True):
    """启动监控服务器"""
    output_dir = Path(output_dir) if output_dir else Path("./output")
    
    def handler(*args, **kwargs):
        return MonitorHandler(*args, output_dir=output_dir, **kwargs)
    
    server = HTTPServer((host, port), handler)
    
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
    
    return server


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
