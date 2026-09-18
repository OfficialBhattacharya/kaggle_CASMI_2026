"""Minimal Kaggle REST client using a KGAT_ bearer token.

Why not the official CLI: the `kaggle` package caps at 1.7.4.5 on Python 3.9,
and that version predates KGAT_ tokens — it hard-requires a legacy
kaggle.json (username + key) and fails at import otherwise. The REST API
accepts the token as a plain bearer, so this talks to it directly.

Token resolution order: KAGGLE_API_TOKEN env var, then ~/.kaggle/access_token.
The token is never logged.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

BASE = "https://www.kaggle.com/api/v1"


class KaggleError(RuntimeError):
    pass


def get_token() -> str:
    tok = os.environ.get("KAGGLE_API_TOKEN")
    if not tok:
        p = Path.home() / ".kaggle" / "access_token"
        if p.exists():
            tok = p.read_text().strip()
    if not tok:
        raise KaggleError(
            "No Kaggle token. Create one at kaggle.com -> Settings -> API, then:\n"
            "  mkdir -p ~/.kaggle && echo <TOKEN> > ~/.kaggle/access_token"
            " && chmod 600 ~/.kaggle/access_token")
    return tok


def request(method: str, path: str, body: dict | None = None,
            params: dict | None = None) -> Any:
    url = f"{BASE}{path}"
    if params:
        from urllib.parse import urlencode
        url += "?" + urlencode(params)
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {get_token()}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            raw = r.read().decode()
    except urllib.error.HTTPError as e:
        detail = e.read().decode()[:600]
        if e.code == 403:
            raise KaggleError(
                f"403 Forbidden on {path}. Usually means the competition rules have "
                f"not been accepted, or the token lacks scope.\n{detail}") from e
        raise KaggleError(f"HTTP {e.code} on {method} {path}: {detail}") from e
    return json.loads(raw) if raw.strip() else None


def whoami() -> str | None:
    """Best-effort username. The API has no /me endpoint for this token type,
    so fall back to KAGGLE_USERNAME or a stored note."""
    u = os.environ.get("KAGGLE_USERNAME")
    if u:
        return u
    p = Path.home() / ".kaggle" / "username"
    return p.read_text().strip() if p.exists() else None


def push_kernel(*, slug: str, title: str, notebook_path: Path, competition: str,
                private: bool = True, gpu: bool = False, internet: bool = False,
                dataset_sources: list[str] | None = None,
                kernel_sources: list[str] | None = None) -> dict:
    """Create or update a notebook. `slug` is 'username/kernel-name'."""
    payload = {
        "slug": slug,
        "newTitle": title,
        "text": notebook_path.read_text(),
        "language": "python",
        "kernelType": "notebook",
        "isPrivate": private,
        "enableGpu": gpu,
        "enableTpu": False,
        "enableInternet": internet,
        "datasetDataSources": dataset_sources or [],
        "competitionDataSources": [competition],
        "kernelDataSources": kernel_sources or [],
        "categoryIds": [],
    }
    return request("POST", "/kernels/push", payload)


def kernel_status(slug: str) -> dict:
    user, name = slug.split("/", 1)
    return request("GET", "/kernels/status", params={"userName": user, "kernelSlug": name})


def list_kernels(user: str | None = None, page_size: int = 50) -> list[dict]:
    """Kernels belonging to `user`. Note the `isPrivate` field in this response
    is not populated — check visibility by fetching the URL unauthenticated."""
    return request("GET", "/kernels/list",
                   params={"user": user or whoami(), "pageSize": page_size,
                           "sortBy": "dateCreated"}) or []


def competition_files(competition: str) -> list[dict]:
    r = request("GET", f"/competitions/data/list/{competition}")
    return r.get("files", r) if isinstance(r, dict) else r


def submissions(competition: str) -> list[dict]:
    return request("GET", f"/competitions/submissions/list/{competition}")
