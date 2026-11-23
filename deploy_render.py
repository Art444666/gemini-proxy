"""Create a Render web service from this repo and wait for the deploy.

Token is read from the RENDER_API_KEY environment variable; never committed.
"""

import os
import sys
import time

import requests

API = "https://api.render.com/v1"
REPO = os.getenv("REPO_URL", "https://github.com/Art444666/gemini-proxy")
BRANCH = os.getenv("BRANCH", "master")
NAME = os.getenv("SERVICE_NAME", "gemini-proxy")

TOKEN = os.getenv("RENDER_API_KEY")
if not TOKEN:
    sys.exit("RENDER_API_KEY is not set")

HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json", "Accept": "application/json"}


def api(method: str, path: str, **kwargs):
    resp = requests.request(method, f"{API}{path}", headers=HEADERS, timeout=60, **kwargs)
    if resp.status_code >= 400:
        sys.exit(f"{method} {path} -> {resp.status_code}: {resp.text}")
    return resp.json() if resp.text else {}


def main() -> None:
    owners = api("GET", "/owners")
    owner_id = owners[0]["owner"]["id"]

    payload = {
        "type": "web_service",
        "name": NAME,
        "ownerId": owner_id,
        "repo": REPO,
        "branch": BRANCH,
        "autoDeploy": "yes",
        "serviceDetails": {
            "env": "python",
            "plan": "free",
            "region": "frankfurt",
            "healthCheckPath": "/healthz",
            "envSpecificDetails": {
                "buildCommand": "pip install -r requirements.txt",
                "startCommand": "uvicorn main:app --host 0.0.0.0 --port $PORT",
            },
            "envVars": [{"key": "PYTHON_VERSION", "value": "3.12.0"}],
        },
    }

    service = api("POST", "/services", json=payload)
    service_id = service["id"]
    print(f"service_id={service_id}")

    deadline = time.time() + 900
    while time.time() < deadline:
        svc = api("GET", f"/services/{service_id}")
        details = svc.get("serviceDetails", {})
        status = details.get("deployStatus")
        print(f"deploy={status}")
        if status == "live":
            print(f"url={details.get('url')}")
            return
        if status in {"build_failed", "update_failed", "canceled", "deactivated"}:
            sys.exit(f"deploy failed: {status}")
        time.sleep(10)
    sys.exit("deploy timeout")


if __name__ == "__main__":
    main()
