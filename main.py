import os
import json
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

GROQ_BASE = "https://api.groq.com/openai/v1"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
TIMEOUT = httpx.Timeout(connect=15.0, read=120.0, write=60.0, pool=15.0)
RETRY_STATUS = {401, 403, 429, 503}


def parse_keys(name: str):
    raw = os.getenv(name, "") or ""
    for ch in (";", "\n", "\r", "\t", " "):
        raw = raw.replace(ch, ",")
    return [k.strip() for k in raw.split(",") if k.strip()]


class KeyPool:
    def __init__(self, env_name: str):
        self.env_name = env_name
        self.keys = []
        self.cursor = 0

    def refresh(self):
        self.keys = parse_keys(self.env_name)
        if self.keys:
            self.cursor %= len(self.keys)
        else:
            self.cursor = 0

    def order(self):
        if not self.keys:
            return []
        start = self.cursor % len(self.keys)
        self.cursor = (start + 1) % len(self.keys)
        return self.keys[start:] + self.keys[:start]


groq_pool = KeyPool("GROQ_API_KEYS")
gemini_pool = KeyPool("GEMINI_API_KEYS")


@asynccontextmanager
async def lifespan(app: FastAPI):
    groq_pool.refresh()
    gemini_pool.refresh()
    yield


app = FastAPI(title="multi-provider-openai-gateway", lifespan=lifespan)


def auth_token(request: Request) -> str:
    header = request.headers.get("authorization", "") or ""
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return (request.headers.get("x-api-key", "") or "").strip()


def error_response(message: str, status: int, etype: str = "invalid_request_error", code: str = None, param: str = None):
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": etype, "param": param, "code": code}},
    )


def guard(request: Request):
    expected = (os.getenv("PROXY_AUTH_TOKEN", "") or "").strip()
    if not expected:
        return None
    if auth_token(request) != expected:
        return error_response("Invalid API key provided.", 401, "invalid_request_error", "invalid_api_key")
    return None


def upstream_headers(key: str, gemini: bool, stream: bool = False):
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if stream else "application/json",
        "User-Agent": "openai-gateway/1.0",
    }
    if gemini:
        headers["x-goog-api-key"] = key
    return headers


def pick_pool(model: str):
    if model.strip().lower().startswith("gemini"):
        return gemini_pool, GEMINI_BASE, True
    return groq_pool, GROQ_BASE, False


def extract_error(text: str, status: int):
    try:
        payload = json.loads(text)
        if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
            return payload["error"].get("message") or text[:400]
        if isinstance(payload, dict) and isinstance(payload.get("error"), str):
            return payload["error"]
    except Exception:
        pass
    return (text or f"upstream returned {status}")[:400]


@app.get("/")
@app.get("/health")
@app.get("/healthz")
async def health():
    return {
        "status": "ok",
        "groq_keys": len(groq_pool.keys),
        "gemini_keys": len(gemini_pool.keys),
    }


async def json_chat(url: str, body: dict, keys, gemini: bool):
    last_status = 502
    last_message = "no keys available"
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        for key in keys:
            try:
                resp = await client.post(url, json=body, headers=upstream_headers(key, gemini))
            except httpx.HTTPError as exc:
                last_status = 502
                last_message = f"upstream connection failed: {exc.__class__.__name__}"
                continue
            if resp.status_code in RETRY_STATUS:
                last_status = resp.status_code
                last_message = extract_error(resp.text, resp.status_code)
                continue
            media = resp.headers.get("content-type", "application/json")
            return Response(content=resp.content, status_code=resp.status_code, media_type=media)
    status = last_status if last_status in RETRY_STATUS else 502
    return error_response(f"all keys failed ({last_status}): {last_message}", status, "api_error", "upstream_error")


async def stream_chat(url: str, body: dict, keys, gemini: bool):
    client = httpx.AsyncClient(timeout=TIMEOUT)
    upstream = None
    last_status = 502
    last_message = "no keys available"
    for key in keys:
        try:
            request = client.build_request("POST", url, json=body, headers=upstream_headers(key, gemini, True))
            resp = await client.send(request, stream=True)
        except httpx.HTTPError as exc:
            last_status = 502
            last_message = f"upstream connection failed: {exc.__class__.__name__}"
            continue
        if resp.status_code in RETRY_STATUS:
            last_status = resp.status_code
            try:
                last_message = extract_error((await resp.aread()).decode("utf-8", "ignore"), resp.status_code)
            except Exception:
                last_message = f"upstream returned {resp.status_code}"
            await resp.aclose()
            continue
        if resp.status_code >= 400:
            try:
                text = (await resp.aread()).decode("utf-8", "ignore")
                payload = json.loads(text)
            except Exception:
                text, payload = f"upstream returned {resp.status_code}", None
            await resp.aclose()
            await client.aclose()
            if payload is not None:
                return JSONResponse(status_code=resp.status_code, content=payload)
            return error_response(text, resp.status_code, "api_error", "upstream_error")
        upstream = resp
        break

    if upstream is None:
        await client.aclose()
        status = last_status if last_status in RETRY_STATUS else 502
        return error_response(f"all keys failed ({last_status}): {last_message}", status, "api_error", "upstream_error")

    async def relay():
        try:
            async for chunk in upstream.aiter_bytes():
                if chunk:
                    yield chunk
        except httpx.HTTPError:
            return
        finally:
            try:
                await upstream.aclose()
            except Exception:
                pass
            await client.aclose()

    return StreamingResponse(
        relay(),
        status_code=200,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def chat_handler(request: Request):
    blocked = guard(request)
    if blocked is not None:
        return blocked
    try:
        body = await request.json()
    except Exception:
        return error_response("Invalid JSON body.", 400, "invalid_request_error", "invalid_json")
    if not isinstance(body, dict):
        return error_response("JSON body must be an object.", 400, "invalid_request_error", "invalid_json")
    model = str(body.get("model") or "")
    if not model:
        return error_response("Missing required parameter: 'model'.", 400, "invalid_request_error", "missing_model", "model")
    pool, base, gemini = pick_pool(model)
    keys = pool.order()
    if not keys:
        provider = "gemini" if gemini else "groq"
        return error_response(f"No API keys configured for provider '{provider}'.", 503, "api_error", "no_keys")
    url = f"{base}/chat/completions"
    if body.get("stream") is True:
        return await stream_chat(url, body, keys, gemini)
    return await json_chat(url, body, keys, gemini)


async def models_handler(request: Request):
    blocked = guard(request)
    if blocked is not None:
        return blocked
    data = []
    seen = set()
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        for pool, base, gemini in ((groq_pool, GROQ_BASE, False), (gemini_pool, GEMINI_BASE, True)):
            keys = pool.order()
            if not keys:
                continue
            for key in keys:
                try:
                    resp = await client.get(f"{base}/models", headers=upstream_headers(key, gemini))
                except httpx.HTTPError:
                    continue
                if resp.status_code in RETRY_STATUS:
                    continue
                if resp.status_code != 200:
                    break
                try:
                    payload = resp.json()
                except Exception:
                    break
                items = payload.get("data") if isinstance(payload, dict) else payload
                for item in items or []:
                    if not isinstance(item, dict):
                        continue
                    model_id = str(item.get("id") or item.get("name") or "").strip()
                    if not model_id:
                        continue
                    if model_id.startswith("models/"):
                        model_id = model_id[7:]
                    if model_id in seen:
                        continue
                    seen.add(model_id)
                    data.append(
                        {
                            "id": model_id,
                            "object": "model",
                            "created": item.get("created") or 0,
                            "owned_by": item.get("owned_by") or ("google" if gemini else "groq"),
                        }
                    )
                break
    return JSONResponse(content={"object": "list", "data": data})


app.add_api_route("/v1/chat/completions", chat_handler, methods=["POST"])
app.add_api_route("/chat/completions", chat_handler, methods=["POST"])
app.add_api_route("/v1/models", models_handler, methods=["GET"])
app.add_api_route("/models", models_handler, methods=["GET"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "10000")))
