import os
import json
import socket
import traceback
import urllib.request

# ── 1. secrets 客户端依赖的环境变量 ────────────────────────────────
t = os.environ.get("KAGGLE_USER_SECRETS_TOKEN")
print(f"[PROBE] KAGGLE_USER_SECRETS_TOKEN: {'set len=%d' % len(t) if t else 'MISSING'}")
print(f"[PROBE] KAGGLE_URL_BASE: {os.environ.get('KAGGLE_URL_BASE')!r}")
print(f"[PROBE] KAGGLE_IAP_TOKEN: {'set' if os.environ.get('KAGGLE_IAP_TOKEN') else 'MISSING'}")

# ── 2. 网络连通性分层：DNS -> TCP -> HTTPS GET -> secrets POST ─────
try:
    print(f"[PROBE] DNS www.kaggle.com -> {socket.getaddrinfo('www.kaggle.com', 443)[0][4]}")
except Exception as e:
    print(f"[PROBE] DNS www.kaggle.com 失败: {type(e).__name__}: {e}")

try:
    s = socket.create_connection(("www.kaggle.com", 443), timeout=10)
    print(f"[PROBE] TCP www.kaggle.com:443 OK {s.getpeername()}")
    s.close()
except Exception as e:
    print(f"[PROBE] TCP www.kaggle.com:443 失败: {type(e).__name__}: {e}")

for url in ("https://huggingface.co/api/models/krea/Krea-2-Raw",
            "https://www.kaggle.com/"):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "probe"})
        with urllib.request.urlopen(req, timeout=15) as r:
            print(f"[PROBE] GET {url} -> {r.status}")
    except Exception as e:
        print(f"[PROBE] GET {url} 失败: {type(e).__name__}: {e}")

# ── 3. get_secret 完整 traceback（URLError 的 reason 是关键）────────
try:
    from kaggle_secrets import UserSecretsClient
    tok = UserSecretsClient().get_secret("HF_TOKEN")
    print(f"[PROBE] HF_TOKEN OK len={len(tok)}")
except Exception:
    print("[PROBE] get_secret 完整 traceback:")
    traceback.print_exc()

# ── 4. 绕过 kaggle_secrets，手工 POST secrets 端点（看 HTTP 层应答）──
if t:
    try:
        body = json.dumps({"Label": "HF_TOKEN"}).encode()
        req = urllib.request.Request(
            "https://www.kaggle.com/requests/GetUserSecretByLabelRequest",
            data=body,
            headers={"Content-type": "application/json",
                     "X-Kaggle-Authorization": f"Bearer {t}"})
        with urllib.request.urlopen(req, timeout=20) as r:
            print(f"[PROBE] 手工 POST secrets 端点 -> {r.status}: {r.read()[:200]}")
    except Exception:
        print("[PROBE] 手工 POST secrets 端点 traceback:")
        traceback.print_exc()

print("[PROBE] done")
