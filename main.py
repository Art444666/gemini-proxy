import os
import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

app = FastAPI()

TARGET_BASE = "https://googleapis.com"
CLIENT_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0)

# Shared async client with connection pooling
client = httpx.AsyncClient(
    base_url=TARGET_BASE,
    timeout=CLIENT_TIMEOUT,
    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    http2=True,
)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
async def proxy(request: Request, path: str):
    # Build target URL
    target_url = f"/{path}"
    if request.url.query:
        target_url += f"?{request.url.query}"

    # Prepare headers - forward all except host
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)

    # Get request body
    body = await request.body()

    # Make upstream request with streaming
    upstream_req = client.build_request(
        method=request.method,
        url=target_url,
        headers=headers,
        content=body,
    )

    try:
        upstream_resp = await client.send(upstream_req, stream=True)
    except httpx.RequestError as e:
        return Response(content=str(e), status_code=502)

    # Prepare response headers - forward all except hop-by-hop
    excluded_headers = {
        "content-encoding",
        "content-length",
        "transfer-encoding",
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "upgrade",
    }
    response_headers = {
        k: v for k, v in upstream_resp.headers.items()
        if k.lower() not in excluded_headers
    }

    # Stream response body
    async def stream_body():
        async for chunk in upstream_resp.aiter_bytes():
            yield chunk

    return StreamingResponse(
        stream_body(),
        status_code=upstream_resp.status_code,
        headers=response_headers,
        background=BackgroundTask(upstream_resp.aclose),
    )


@app.on_event("shutdown")
async def shutdown():
    await client.aclose()


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)