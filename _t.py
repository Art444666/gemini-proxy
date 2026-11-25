import httpx, json, sys

H = {"Authorization": "Bearer rnd_8El1E5WO8lA86Rbnc32bQSE5pnuP", "Accept": "application/json"}
env = httpx.get(
    "https://api.render.com/v1/services/srv-dc267btg1s2s7393gd5g/env-vars?limit=50",
    headers=H, timeout=30,
).json()
vals = {}
for item in env:
    v = item.get("envVar") or item
    vals[v["key"]] = v.get("value")
tok = vals["PROXY_AUTH_TOKEN"]
base = "https://gemini-proxy-bssr.onrender.com"

r = httpx.get(f"{base}/health", timeout=30)
print("health:", r.status_code, r.text[:200])

model = sys.argv[1] if len(sys.argv) > 1 else "gemini-3.5-flash-lite"
body = {"model": model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 8}
r = httpx.post(
    f"{base}/v1/chat/completions", headers={"Authorization": f"Bearer {tok}"},
    json=body, timeout=120,
)
print("chat:", r.status_code)
print(r.text[:600])
