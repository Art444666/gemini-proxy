"""Streaming reverse proxy for the Google Generative Language (Gemini) API.

Forwards every request (method, path, query, headers, body) to the upstream
googleapis host, preserving ``x-goog-api-key`` and other auth headers, and
streams the response back chunk by chunk (both directions are streamed).
"""

import logging
import os

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("gemini-proxy")

# Upstream host. Override with UPSTREAM_BASE to point at another *.googleapis.com
# service (e.g. https://googleapis.com literally, or a regional vertex endpoint).
UPSTREAM_BASE = os.getenv("UPSTREAM_BASE", "https://generativelanguage.googleapis.com").rstrip("/")

TIMEOUT = httpx.Timeout(connect=15.0, read=600.0, write=60.0, pool=15.0)

# RFC 7230 hop-by-hop headers: never forwarded in either direction.
HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "trailers",
    "transfer-encoding",
    "upgrade",
}
REQUEST_STRIP = HOP_BY_HOP | {"host", "content-length"}
RESPONSE_STRIP = HOP_BY_HOP | {"content-length"}

ALL_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]

app = FastAPI(
    title="gemini-proxy",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

client = httpx.AsyncClient(
    timeout=TIMEOUT,
    limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
    follow_redirects=False,
    http2=True,
)


async def _close_client() -> None:
    await client.aclose()


@app.on_event("shutdown")
async def shutdown() -> None:
    await _close_client()


@app.get("/healthz")
async def healthz() -> JSONResponse:
    return JSONResponse({"status": "ok", "upstream": UPSTREAM_BASE})


@app.api_route("/{path:path}", methods=ALL_METHODS)
async def proxy(request: Request, path: str):
    target_url = f"{UPSTREAM_BASE}/{path}"
    if request.url.query:
        target_url = f"{target_url}?{request.url.query}"

    headers = {k: v for k, v in request.headers.items() if k.lower() not in REQUEST_STRIP}

    # Stream the client body straight upstream instead of buffering it.
    body_less = request.headers.get("content-length") in (None, "0") and "transfer-encoding" not in request.headers
    content = None if body_less else request.stream()

    upstream_request = client.build_request(
        method=request.method,
        url=target_url,
        headers=headers,
        content=content,
    )

    try:
        upstream_response = await client.send(upstream_request, stream=True)
    except httpx.HTTPError as exc:
        log.warning("upstream error %s %s: %s", request.method, path, exc)
        return JSONResponse({"error": "upstream_unreachable", "detail": str(exc)}, status_code=502)

    log.info("%s /%s -> %s", request.method, path, upstream_response.status_code)

    response_headers = {
        k: v for k, v in upstream_response.headers.items() if k.lower() not in RESPONSE_STRIP
    }
    response_headers.setdefault("x-proxy-upstream", UPSTREAM_BASE)

    async def stream_body():
        try:
            # aiter_raw keeps the upstream bytes untouched (gzip/SSE/chunked).
            async for chunk in upstream_response.aiter_raw():
                yield chunk
        finally:
            await upstream_response.aclose()

    return StreamingResponse(
        stream_body(),
        status_code=upstream_response.status_code,
        headers=response_headers,
        background=BackgroundTask(upstream_response.aclose),
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, proxy_headers=True)
