import base64
import os
import json
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

GROQ_BASE = "https://api.groq.com/openai/v1"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
TOKENHARBOR_BASE = (os.getenv("TOKENHARBOR_BASE_URL", "") or "https://tokenharbor.ai/v1").rstrip("/")
GEMINI_MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_FALLBACK_MODELS = ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.5-pro"]
TIMEOUT = httpx.Timeout(connect=20.0, read=300.0, write=300.0, pool=30.0)
RETRY_STATUS = {401, 402, 403, 429, 503}


def parse_keys(name: str):
    raw = os.getenv(name, "") or ""
    for ch in (";", "\n", "\r", "\t", " "):
        raw = raw.replace(ch, ",")
    return [k.strip() for k in raw.split(",") if k.strip()]


class KeyPool:
    def __init__(self, *env_names: str):
        self.env_names = tuple(n for n in env_names if n)
        self.keys = []
        self.cursor = 0

    def refresh(self):
        keys = []
        for env_name in self.env_names:
            for key in parse_keys(env_name):
                if key not in keys:
                    keys.append(key)
        self.keys = keys
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
harbor_pool = KeyPool("TOKENHARBOR_API_KEYS", "TOKENHARBOR_API_KEY")

PROVIDER_BASE = {
    "groq": GROQ_BASE,
    "gemini": GEMINI_BASE,
    "tokenharbor": TOKENHARBOR_BASE,
}
PROVIDER_POOL = {
    "groq": groq_pool,
    "gemini": gemini_pool,
    "tokenharbor": harbor_pool,
}
PROVIDER_IS_GEMINI = {"gemini": True}
PROVIDER_ORDER = ("tokenharbor", "groq", "gemini")
HARBOR_ALIASES = ("tokenharbor/", "harbor/", "th/")
GROQ_MODEL_PREFIXES = (
    "llama",
    "whisper",
    "mixtral",
    "gemma",
    "groq/",
    "openai/gpt-oss",
    "meta-llama/",
    "distil-whisper/",
    "playai-",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    groq_pool.refresh()
    gemini_pool.refresh()
    harbor_pool.refresh()
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


def harbor_model_prefixes():
    raw = os.getenv("TOKENHARBOR_MODEL_PREFIXES", "") or ""
    for ch in (";", "\n", "\r", "\t", " "):
        raw = raw.replace(ch, ",")
    return [p.strip().lower() for p in raw.split(",") if p.strip()]


def route_model(model: str):
    """Return (provider, upstream_model) for a client-requested model id."""
    name = model.strip()
    low = name.lower()
    for alias in HARBOR_ALIASES:
        if low.startswith(alias):
            return "tokenharbor", name[len(alias):].strip() or name
    if low.startswith("gemini"):
        return "gemini", name
    if low.startswith(GROQ_MODEL_PREFIXES):
        return "groq", name
    for prefix in harbor_model_prefixes():
        if low.startswith(prefix):
            return "tokenharbor", name
    default_provider = (os.getenv("DEFAULT_PROVIDER", "") or "").strip().lower()
    if default_provider in PROVIDER_POOL:
        return default_provider, name
    return "groq", name


def is_upstream_failure(result) -> bool:
    if isinstance(result, JSONResponse):
        return result.status_code in RETRY_STATUS or result.status_code >= 500
    return False


def is_model_not_found(result) -> bool:
    """True when the provider replied 404 / model_not_found."""
    return isinstance(result, JSONResponse) and result.status_code == 404


def translate_model(provider: str, model: str) -> str:
    """Upstream model id to use on `provider` for a client-requested id."""
    routed, upstream = route_model(model)
    if routed == provider:
        return upstream
    return model.strip()


def provider_can_serve(provider: str, model: str) -> bool:
    """True when `provider` plausibly serves this exact model id."""
    low = model.strip().lower()
    if provider == "gemini":
        return low.startswith("gemini")
    if provider == "groq":
        return low.startswith(GROQ_MODEL_PREFIXES)
    if provider == "tokenharbor":
        if low.startswith(HARBOR_ALIASES):
            return True
        if low.startswith("gemini"):
            return False
        return low.startswith(tuple(harbor_model_prefixes()) or ("\0",))
    return True


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
        "tokenharbor_keys": len(harbor_pool.keys),
        "tokenharbor_base": TOKENHARBOR_BASE,
        "default_provider": (os.getenv("DEFAULT_PROVIDER", "") or "").strip().lower() or None,
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

MAX_BODY_BYTES = int(os.getenv("MAX_BODY_BYTES", "") or 48 * 1024 * 1024)
MAX_INLINE_FILE_BYTES = int(os.getenv("MAX_INLINE_FILE_BYTES", "") or 16 * 1024 * 1024)
TEXT_PART_TYPES = {"text", "input_text", "output_text", "summary_text"}
IMAGE_PART_TYPES = {"image_url", "input_image", "image", "image_base64", "input_image_url"}
FILE_PART_TYPES = {
    "file", "input_file", "file_data", "file_url", "input_file_data", "attachment",
    "document", "audio", "input_audio", "audio_url", "video", "input_video",
}
TEXTUAL_MIME = (
    "text/", "application/json", "application/xml", "application/csv",
    "application/x-yaml", "application/javascript",
)


async def unhandled_exception_handler(request: Request, exc: Exception):
    return error_response(
        f"Gateway internal error: {exc.__class__.__name__}: {str(exc)[:200]}",
        500,
        "api_error",
        "gateway_error",
    )


app.add_exception_handler(Exception, unhandled_exception_handler)


def _part_text(part):
    for key in ("text", "input_text", "content", "summary"):
        value = part.get(key)
        if isinstance(value, str):
            return value
    return ""


def _part_url(part):
    for key in (
        "image_url", "url", "file_url", "file_data", "file_data_url", "data",
        "image_base64", "base64", "b64_json", "input_image", "file", "image", "audio",
    ):
        value = part.get(key)
        if isinstance(value, dict):
            value = _part_url(value)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _mime_of(url):
    if not url.startswith("data:"):
        return ""
    head = url[5:].split(",", 1)[0]
    return head.split(";", 1)[0].strip().lower()


def _decode_text_data_url(url):
    if not url.startswith("data:") or ";base64," not in url:
        return ""
    payload = url.split(";base64,", 1)[1]
    try:
        raw = base64.b64decode(payload + "===")
    except Exception:
        return ""
    for encoding in ("utf-8", "cp1251", "latin-1"):
        try:
            return raw.decode(encoding)
        except Exception:
            continue
    return ""


def _part_name(part):
    for key in ("filename", "name", "file_name", "title"):
        value = part.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key in ("file", "image_url", "file_data", "input_file", "document", "attachment"):
        value = part.get(key)
        if isinstance(value, dict):
            found = _part_name(value)
            if found:
                return found
    return ""


def normalize_multimodal(body):
    """Coerce OpenAI/Responses style multimodal parts into parts upstream can digest."""
    messages = body.get("messages")
    if not isinstance(messages, list):
        return body
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if content is None or isinstance(content, str):
            continue
        if not isinstance(content, list):
            message["content"] = str(content)
            continue
        parts = []
        notes = []
        for part in content:
            if isinstance(part, str):
                parts.append({"type": "text", "text": part})
                continue
            if not isinstance(part, dict):
                continue
            kind = str(part.get("type") or "").strip().lower().replace("-", "_")
            text = _part_text(part)
            url = _part_url(part)
            mime = _mime_of(url)
            name = _part_name(part) or kind or "file"
            if kind in TEXT_PART_TYPES or (text and not url):
                parts.append({"type": "text", "text": text})
                continue
            if kind in IMAGE_PART_TYPES and url:
                parts.append({"type": "image_url", "image_url": {"url": url}})
                continue
            if kind in FILE_PART_TYPES or url:
                if not url:
                    if text:
                        parts.append({"type": "text", "text": text})
                    else:
                        notes.append(f"skipped attachment {name}: empty payload")
                    continue
                if mime.startswith(TEXTUAL_MIME):
                    decoded = _decode_text_data_url(url)
                    if decoded:
                        parts.append({"type": "text", "text": f"--- {name} ---\n{decoded}"})
                        continue
                parts.append({"type": "image_url", "image_url": {"url": url}})
                if not mime.startswith("image/"):
                    notes.append(f"attachment {name} ({mime or 'unknown type'}) forwarded")
                continue
            if text:
                parts.append({"type": "text", "text": text})
            else:
                notes.append(f"skipped part {name}: unsupported type")
        if notes:
            parts.append({"type": "text", "text": "[gateway] " + "; ".join(notes)})
        if not parts:
            parts.append({"type": "text", "text": ""})
        message["content"] = parts
    return body


async def multipart_body(request: Request):
    form = await request.form()
    fields = {}
    uploads = []
    for key, value in form.multi_items():
        if hasattr(value, "read"):
            uploads.append((str(key), value))
        else:
            fields[str(key)] = value if isinstance(value, str) else str(value)
    messages = []
    raw_messages = fields.get("messages")
    if raw_messages:
        try:
            parsed = json.loads(raw_messages)
            if isinstance(parsed, dict):
                parsed = parsed.get("messages") or []
            if isinstance(parsed, list):
                messages = [m for m in parsed if isinstance(m, dict)]
        except Exception:
            messages = []
    attachments = []
    for key, upload in uploads:
        try:
            data = await upload.read()
        except Exception:
            continue
        if not data:
            continue
        name = getattr(upload, "filename", "") or key or "file"
        if MAX_INLINE_FILE_BYTES > 0 and len(data) > MAX_INLINE_FILE_BYTES:
            attachments.append({"type": "text", "text": f"[gateway] {name}: file is larger than {MAX_INLINE_FILE_BYTES} bytes, skipped"})
            continue
        mime = (getattr(upload, "content_type", "") or "").strip() or "application/octet-stream"
        encoded = base64.b64encode(data).decode("ascii")
        attachments.append(
            {
                "type": "file",
                "file": {"filename": name, "file_data": f"data:{mime};base64,{encoded}"},
            }
        )
    if attachments:
        messages.append({"role": "user", "content": attachments})
    if not messages:
        prompt = fields.get("prompt") or fields.get("message") or fields.get("q") or ""
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    body = {
        "model": str(fields.get("model") or fields.get("model_name") or ""),
        "messages": messages,
    }
    if str(fields.get("stream", "")).strip().lower() in ("1", "true", "yes"):
        body["stream"] = True
    for key in (
        "temperature", "top_p", "max_tokens", "max_completion_tokens", "stop",
        "presence_penalty", "frequency_penalty", "seed", "n", "response_format",
        "tools", "tool_choice", "user",
    ):
        if key in fields:
            try:
                body[key] = json.loads(fields[key])
            except Exception:
                body[key] = fields[key]
    return body


async def read_chat_body(request: Request):
    """Read a chat body (JSON or multipart with files), enforce a size cap, never raise."""
    try:
        raw = await request.body()
    except Exception as exc:
        return None, error_response(
            f"Failed to read request body: {exc.__class__.__name__}",
            400,
            "invalid_request_error",
            "invalid_body",
        )
    if MAX_BODY_BYTES > 0 and len(raw) > MAX_BODY_BYTES:
        return None, error_response(
            f"Request body is {len(raw)} bytes, the gateway limit is {MAX_BODY_BYTES}. Attach a smaller file or pass an http(s) URL.",
            413,
            "invalid_request_error",
            "payload_too_large",
        )
    ctype = (request.headers.get("content-type") or "").lower()
    if "multipart/form-data" in ctype:
        try:
            body = await multipart_body(request)
        except Exception as exc:
            return None, error_response(
                f"Malformed multipart body: {exc.__class__.__name__}",
                400,
                "invalid_request_error",
                "invalid_multipart",
            )
    else:
        stripped = raw.strip()
        if not stripped:
            return None, error_response("Invalid JSON body.", 400, "invalid_request_error", "invalid_json")
        try:
            body = json.loads(stripped.decode("utf-8", "ignore"))
        except Exception:
            return None, error_response(
                "Invalid JSON body. Send application/json; multipart/form-data is accepted for file uploads.",
                400,
                "invalid_request_error",
                "invalid_json",
            )
    if not isinstance(body, dict):
        return None, error_response("JSON body must be an object.", 400, "invalid_request_error", "invalid_json")
    try:
        body = normalize_multimodal(body)
    except Exception:
        pass
    return body, None

async def chat_handler(request: Request):
    blocked = guard(request)
    if blocked is not None:
        return blocked
    body, failure = await read_chat_body(request)
    if failure is not None:
        return failure
    model = str(body.get("model") or "")
    if not model:
        return error_response("Missing required parameter: 'model'.", 400, "invalid_request_error", "missing_model", "model")
    provider, upstream_model = route_model(model)

    # Fallback only to providers that actually serve this model id.
    candidates = [provider] + [
        p for p in PROVIDER_ORDER
        if p != provider and p in PROVIDER_POOL and provider_can_serve(p, model)
    ]
    last_failure = None
    primary_failure = None
    for name in candidates:
        keys = PROVIDER_POOL[name].order()
        if not keys:
            continue
        base = PROVIDER_BASE[name]
        gemini = PROVIDER_IS_GEMINI.get(name, False)
        url = f"{base}/chat/completions"
        body["model"] = upstream_model if name == provider else translate_model(name, model)
        if body.get("stream") is True:
            result = await stream_chat(url, body, keys, gemini)
        else:
            result = await json_chat(url, body, keys, gemini)
        if not is_upstream_failure(result):
            return result
        if name != provider and is_model_not_found(result):
            continue
        if name == provider:
            primary_failure = result
        last_failure = result

    if primary_failure is not None and (
        last_failure is None or is_model_not_found(last_failure)
    ):
        return primary_failure
    if last_failure is not None:
        return last_failure
    return error_response(f"No API keys configured for provider '{provider}'.", 503, "api_error", "no_keys")


async def fetch_groq_models(client: httpx.AsyncClient):
    models = []
    for key in groq_pool.order():
        try:
            resp = await client.get(f"{GROQ_BASE}/models", headers=upstream_headers(key, False))
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
            model_id = str(item.get("id") or "").strip()
            if model_id:
                models.append({"id": model_id, "created": item.get("created") or 0, "owned_by": item.get("owned_by") or "groq"})
        break
    return models


async def fetch_gemini_models(client: httpx.AsyncClient):
    models = []
    for key in gemini_pool.order():
        try:
            resp = await client.get(
                GEMINI_MODELS_URL,
                params={"key": key, "pageSize": 200},
                headers={"x-goog-api-key": key, "Accept": "application/json"},
            )
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
        for item in payload.get("models", []) if isinstance(payload, dict) else []:
            if not isinstance(item, dict):
                continue
            methods = item.get("supportedGenerationMethods") or []
            if methods and "generateContent" not in methods:
                continue
            model_id = str(item.get("name") or "").strip()
            if model_id.startswith("models/"):
                model_id = model_id[7:]
            if model_id:
                models.append({"id": model_id, "created": 0, "owned_by": "google"})
        if models:
            break
    if not models:
        models = [{"id": m, "created": 0, "owned_by": "google"} for m in GEMINI_FALLBACK_MODELS]
    return models


async def fetch_tokenharbor_models(client: httpx.AsyncClient):
    models = []
    for key in harbor_pool.order():
        try:
            resp = await client.get(f"{TOKENHARBOR_BASE}/models", headers=upstream_headers(key, False))
        except httpx.HTTPError:
            continue
        if resp.status_code in RETRY_STATUS or resp.status_code != 200:
            continue
        try:
            payload = resp.json()
        except Exception:
            continue
        items = payload.get("data") if isinstance(payload, dict) else payload
        for item in items or []:
            if not isinstance(item, dict):
                continue
            model_id = str(item.get("id") or "").strip()
            if model_id:
                models.append({"id": model_id, "created": item.get("created") or 0, "owned_by": item.get("owned_by") or "tokenharbor"})
        if models:
            break
    return models


async def models_handler(request: Request):
    blocked = guard(request)
    if blocked is not None:
        return blocked
    data = []
    seen = set()
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        gathered = []
        if groq_pool.keys:
            gathered.extend(await fetch_groq_models(client))
        if gemini_pool.keys:
            gathered.extend(await fetch_gemini_models(client))
        if harbor_pool.keys:
            gathered.extend(await fetch_tokenharbor_models(client))
    for item in gathered:
        if item["id"] in seen:
            continue
        seen.add(item["id"])
        data.append({"id": item["id"], "object": "model", "created": item["created"], "owned_by": item["owned_by"]})
    return JSONResponse(content={"object": "list", "data": data})


app.add_api_route("/v1/chat/completions", chat_handler, methods=["POST"])
app.add_api_route("/chat/completions", chat_handler, methods=["POST"])
app.add_api_route("/v1/models", models_handler, methods=["GET"])
app.add_api_route("/models", models_handler, methods=["GET"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "10000")))
