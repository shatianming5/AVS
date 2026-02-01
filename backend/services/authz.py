from __future__ import annotations

import json

from fastapi import HTTPException, Request

from backend.services.db import fetch_one
from backend.services.init_db import init_db


def get_access_token(request: Request) -> str | None:
    token = request.headers.get("X-Pack-Token") or request.headers.get("X-Owner-Token")
    if token:
        token = token.strip()
    if not token:
        token = request.query_params.get("token")
        token = token.strip() if token else None
    return token or None


def require_pack_access(*, request: Request, pack_id: str) -> dict:
    init_db()
    token = get_access_token(request)
    pack = fetch_one("SELECT * FROM skillpacks WHERE pack_id = ?", (pack_id,))
    if pack is None:
        raise HTTPException(status_code=404, detail="SkillPack not found.")

    visibility = pack.get("visibility") or "private"
    if visibility == "public":
        return _inflate_pack(pack)

    if token is None:
        raise HTTPException(status_code=403, detail="Missing access token.")
    if token == pack.get("owner_token") or (pack.get("share_token") and token == pack.get("share_token")):
        return _inflate_pack(pack)

    raise HTTPException(status_code=403, detail="Forbidden.")


def require_pack_owner(*, request: Request, pack_id: str) -> dict:
    init_db()
    token = get_access_token(request)
    if token is None:
        raise HTTPException(status_code=403, detail="Missing access token.")
    pack = fetch_one("SELECT * FROM skillpacks WHERE pack_id = ?", (pack_id,))
    if pack is None:
        raise HTTPException(status_code=404, detail="SkillPack not found.")
    if token != pack.get("owner_token"):
        raise HTTPException(status_code=403, detail="Owner token required.")
    return _inflate_pack(pack)


def require_job_access(*, request: Request, job_id: str) -> dict:
    init_db()
    token = get_access_token(request)
    if token is None:
        raise HTTPException(status_code=403, detail="Missing access token.")
    job = fetch_one("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if token != job.get("owner_token"):
        raise HTTPException(status_code=403, detail="Forbidden.")
    return job


def _inflate_pack(pack_row: dict) -> dict:
    pdf_ids = []
    try:
        pdf_ids = json.loads(pack_row.get("pdf_ids_json") or "[]")
    except Exception:  # noqa: BLE001
        pdf_ids = []
    out = dict(pack_row)
    out["pdf_ids"] = pdf_ids
    return out

