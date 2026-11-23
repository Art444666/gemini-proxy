# gemini-proxy

Streaming reverse proxy (FastAPI + httpx) that forwards any request to the Google
Generative Language API, preserving all headers (including `x-goog-api-key`) and
streaming the response back unmodified (SSE / `alt=sse` supported end-to-end).

## Upstream

`UPSTREAM_BASE` defaults to `https://generativelanguage.googleapis.com`.
Override it with an env var to point at another `*.googleapis.com` host.

## Local run

```powershell
pip install -r requirements.txt
$env:GEMINI_API_KEY="<key>"
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Usage

```powershell
curl.exe -s "http://127.0.0.1:8000/v1beta/models?key=$env:GEMINI_API_KEY"

curl.exe -N -s -X POST `
  "http://127.0.0.1:8000/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse" `
  -H "x-goog-api-key: $env:GEMINI_API_KEY" `
  -H "Content-Type: application/json" `
  -d '{\"contents\":[{\"parts\":[{\"text\":\"Say hi\"}]}]}'
```

Point the Gemini CLI at the proxy:

```powershell
[Environment]::SetEnvironmentVariable("GOOGLE_GEMINI_BASE_URL", "https://<service>.onrender.com", "User")
```

## Health

`GET /healthz` -> `{"status":"ok","upstream":"..."}`

## Deploy (Render)

```powershell
$env:RENDER_API_KEY="<token>"; python deploy_render.py
```

The script creates a free `web_service` from this repo, waits for the deploy and
prints the public URL. It never stores the token in the repository.
