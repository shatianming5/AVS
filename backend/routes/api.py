from __future__ import annotations

import json
import uuid
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from backend.services.jobs import create_build_job, get_job, run_build_in_background
from backend.services.authz import get_access_token, require_job_access, require_pack_access, require_pack_owner
from backend.services.db import fetch_all
from backend.services.skillpacks import generate_share_token, set_share_token
from backend.services.storage import (
    get_pdf_file_path,
    get_pdf_record,
    ensure_pdf_owned,
    save_uploaded_pdf,
)
from backend.services.annotations import create_annotation as create_annotation_record
from backend.services.annotations import delete_annotation as delete_annotation_record
from backend.services.annotations import list_annotations as list_annotation_records
from shared.config import data_paths


api_router = APIRouter()


def _get_owner_token(request: Request) -> str | None:
    token = request.headers.get("X-Owner-Token")
    if token:
        token = token.strip()
    if not token:
        token = request.query_params.get("owner_token")
        token = token.strip() if token else None
    return token or None


@api_router.post("/pdfs/upload")
async def upload_pdf(request: Request, file: Annotated[UploadFile, File(...)]) -> dict:
    if file.filename is None or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf uploads are supported.")

    owner_token = _get_owner_token(request) or uuid.uuid4().hex
    pdf_record = await save_uploaded_pdf(file=file, owner_token=owner_token)
    return {"pdf_id": pdf_record["pdf_id"], "owner_token": owner_token}


@api_router.get("/pdfs/{pdf_id}/file")
async def get_pdf_file(request: Request, pdf_id: str) -> FileResponse:
    owner_token = _get_owner_token(request)
    if not owner_token:
        raise HTTPException(status_code=403, detail="Missing owner_token.")
    ensure_pdf_owned(pdf_id=pdf_id, owner_token=owner_token)
    path = get_pdf_file_path(pdf_id)
    return FileResponse(path, media_type="application/pdf", filename=f"{pdf_id}.pdf")


@api_router.get("/pdfs/{pdf_id}/page/{page_index}.png")
async def get_pdf_page_image(request: Request, pdf_id: str, page_index: int) -> FileResponse:
    from backend.services.pdf_pages import render_page_png_cached

    if page_index < 1:
        raise HTTPException(status_code=400, detail="page_index must be >= 1.")
    owner_token = _get_owner_token(request)
    if not owner_token:
        raise HTTPException(status_code=403, detail="Missing owner_token.")
    ensure_pdf_owned(pdf_id=pdf_id, owner_token=owner_token)
    try:
        png_path = render_page_png_cached(pdf_id=pdf_id, page_index=page_index)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return FileResponse(png_path, media_type="image/png")


@api_router.get("/skillpacks/{pack_id}/pdfs/{pdf_id}/file")
async def get_pack_pdf_file(request: Request, pack_id: str, pdf_id: str) -> FileResponse:
    pack = require_pack_access(request=request, pack_id=pack_id)
    if pdf_id not in (pack.get("pdf_ids") or []):
        raise HTTPException(status_code=404, detail="PDF not part of this SkillPack.")
    path = get_pdf_file_path(pdf_id)
    return FileResponse(path, media_type="application/pdf", filename=f"{pdf_id}.pdf")


@api_router.get("/skillpacks/{pack_id}/pdfs/{pdf_id}/meta")
async def get_pack_pdf_meta(request: Request, pack_id: str, pdf_id: str) -> dict:
    pack = require_pack_access(request=request, pack_id=pack_id)
    if pdf_id not in (pack.get("pdf_ids") or []):
        raise HTTPException(status_code=404, detail="PDF not part of this SkillPack.")
    rec = get_pdf_record(pdf_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="PDF not found.")
    return {
        "pdf_id": pdf_id,
        "num_pages": int(rec.get("num_pages") or 0),
        "title": rec.get("title"),
        "original_filename": rec.get("original_filename"),
    }


@api_router.get("/skillpacks/{pack_id}/pdfs/{pdf_id}/page/{page_index}.png")
async def get_pack_pdf_page_image(request: Request, pack_id: str, pdf_id: str, page_index: int) -> FileResponse:
    from backend.services.pdf_pages import render_page_png_cached

    pack = require_pack_access(request=request, pack_id=pack_id)
    if pdf_id not in (pack.get("pdf_ids") or []):
        raise HTTPException(status_code=404, detail="PDF not part of this SkillPack.")
    if page_index < 1:
        raise HTTPException(status_code=400, detail="page_index must be >= 1.")
    try:
        png_path = render_page_png_cached(pdf_id=pdf_id, page_index=page_index)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return FileResponse(png_path, media_type="image/png")


@api_router.post("/skillpacks/build")
async def build_skillpack(request: Request, payload: dict) -> dict:
    pdf_ids = payload.get("pdf_ids")
    pack_name = payload.get("pack_name")
    field_hint = payload.get("field_hint")
    target_venue_hint = payload.get("target_venue_hint")
    language = payload.get("language", "English")
    owner_token = _get_owner_token(request) or payload.get("owner_token")

    if not isinstance(pdf_ids, list) or len(pdf_ids) != 3 or not all(isinstance(x, str) for x in pdf_ids):
        raise HTTPException(status_code=400, detail="pdf_ids must be a list of 3 pdf_id strings.")
    if not isinstance(pack_name, str) or not pack_name.strip():
        raise HTTPException(status_code=400, detail="pack_name is required.")
    if not isinstance(owner_token, str) or not owner_token.strip():
        raise HTTPException(status_code=403, detail="Missing owner_token.")

    owner_token = owner_token.strip()
    for pdf_id in pdf_ids:
        ensure_pdf_owned(pdf_id=pdf_id, owner_token=owner_token)

    job = create_build_job(
        pdf_ids=pdf_ids,
        pack_name=pack_name.strip(),
        field_hint=field_hint,
        target_venue_hint=target_venue_hint,
        language=language,
        owner_token=owner_token,
    )
    run_build_in_background(job["job_id"])
    return {"job_id": job["job_id"], "owner_token": owner_token}


@api_router.get("/jobs/{job_id}")
async def get_job_status(request: Request, job_id: str) -> dict:
    require_job_access(request=request, job_id=job_id)
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


@api_router.get("/jobs")
async def list_jobs(request: Request, limit: int = 50) -> dict:
    token = get_access_token(request)
    if token is None:
        raise HTTPException(status_code=403, detail="Missing access token.")
    lim = int(limit)
    if lim < 1:
        lim = 1
    if lim > 200:
        lim = 200

    rows = fetch_all(
        """
        SELECT job_id, kind, status, progress, stage, error, attempt, payload_json, result_json, metrics_json, created_at, updated_at
        FROM jobs
        WHERE owner_token = ?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (token, lim),
    )
    jobs: list[dict] = []
    def _loads(val: str | None):  # noqa: ANN001
        if not val:
            return None
        try:
            return json.loads(val)
        except Exception:  # noqa: BLE001
            return val

    for r in rows:
        payload = _loads(r.get("payload_json"))
        result = _loads(r.get("result_json"))
        metrics = _loads(r.get("metrics_json"))
        jobs.append(
            {
                "job_id": r["job_id"],
                "kind": r["kind"],
                "status": r["status"],
                "progress": r["progress"],
                "stage": r["stage"],
                "error": r.get("error"),
                "attempt": r.get("attempt", 0),
                "payload": payload,
                "result": result,
                "metrics": metrics,
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
            }
        )
    return {"jobs": jobs}


@api_router.get("/jobs/{job_id}/events")
async def get_job_events(request: Request, job_id: str) -> dict:
    require_job_access(request=request, job_id=job_id)
    rows = fetch_all(
        "SELECT ts, level, stage, message, data_json FROM job_events WHERE job_id = ? ORDER BY id ASC",
        (job_id,),
    )
    events = []
    for r in rows:
        data = None
        if r.get("data_json"):
            try:
                data = json.loads(r["data_json"])
            except Exception:  # noqa: BLE001
                data = r["data_json"]
        events.append({"ts": r["ts"], "level": r["level"], "stage": r["stage"], "message": r["message"], "data": data})
    return {"job_id": job_id, "events": events}


@api_router.get("/skillpacks/{pack_id}")
async def get_skillpack(request: Request, pack_id: str) -> JSONResponse:
    require_pack_access(request=request, pack_id=pack_id)
    path = data_paths().skillpack_json(pack_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="SkillPack not found.")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@api_router.get("/evidence/{pack_id}")
async def get_evidence_index(request: Request, pack_id: str) -> JSONResponse:
    require_pack_access(request=request, pack_id=pack_id)
    path = data_paths().evidence_index_json(pack_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Evidence index not found.")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@api_router.get("/annotations")
async def list_annotations(request: Request, pack_id: str, pdf_id: str) -> dict:
    pack = require_pack_access(request=request, pack_id=pack_id)
    if pdf_id not in (pack.get("pdf_ids") or []):
        raise HTTPException(status_code=404, detail="PDF not part of this SkillPack.")
    owner_token = str(pack.get("owner_token") or "")
    annotations = list_annotation_records(owner_token=owner_token, pack_id=pack_id, pdf_id=pdf_id)
    return {"annotations": annotations}


@api_router.post("/annotations")
async def create_annotation(request: Request, payload: dict) -> dict:
    pack_id = payload.get("pack_id")
    pdf_id = payload.get("pdf_id")
    page_index = payload.get("page_index")
    bbox_norm = payload.get("bbox_norm")
    note = payload.get("note")
    skill_ref_type = payload.get("skill_ref_type")
    skill_ref_id = payload.get("skill_ref_id")
    skill_title = payload.get("skill_title")

    if not isinstance(pack_id, str) or not pack_id.strip():
        raise HTTPException(status_code=400, detail="pack_id is required.")
    if not isinstance(pdf_id, str) or not pdf_id.strip():
        raise HTTPException(status_code=400, detail="pdf_id is required.")
    if not isinstance(page_index, int):
        raise HTTPException(status_code=400, detail="page_index must be an integer.")
    if not isinstance(bbox_norm, list):
        raise HTTPException(status_code=400, detail="bbox_norm must be a list.")
    if note is not None and not isinstance(note, str):
        raise HTTPException(status_code=400, detail="note must be a string.")
    if skill_ref_type is not None and not isinstance(skill_ref_type, str):
        raise HTTPException(status_code=400, detail="skill_ref_type must be a string.")
    if skill_ref_id is not None and not isinstance(skill_ref_id, str):
        raise HTTPException(status_code=400, detail="skill_ref_id must be a string.")
    if skill_title is not None and not isinstance(skill_title, str):
        raise HTTPException(status_code=400, detail="skill_title must be a string.")

    pack = require_pack_owner(request=request, pack_id=pack_id.strip())
    if pdf_id not in (pack.get("pdf_ids") or []):
        raise HTTPException(status_code=404, detail="PDF not part of this SkillPack.")

    owner_token = str(pack.get("owner_token") or "")
    ann = create_annotation_record(
        owner_token=owner_token,
        pack_id=pack_id.strip(),
        pdf_id=pdf_id.strip(),
        page_index=int(page_index),
        bbox_norm=bbox_norm,
        note=(note.strip() if isinstance(note, str) else None),
        skill_ref_type=(skill_ref_type.strip() if isinstance(skill_ref_type, str) else None),
        skill_ref_id=(skill_ref_id.strip() if isinstance(skill_ref_id, str) else None),
        skill_title=(skill_title.strip() if isinstance(skill_title, str) else None),
    )
    return {"annotation": ann}


@api_router.delete("/annotations/{annotation_id}")
async def delete_annotation(request: Request, annotation_id: str) -> dict:
    token = get_access_token(request)
    if token is None:
        raise HTTPException(status_code=403, detail="Missing access token.")
    delete_annotation_record(owner_token=token, annotation_id=annotation_id)
    return {"ok": True}


@api_router.get("/skillpacks/{pack_id}/download")
async def download_skillpack(request: Request, pack_id: str, format: str = "json") -> FileResponse:  # noqa: A002
    require_pack_access(request=request, pack_id=pack_id)
    fmt = (format or "json").lower()
    if fmt not in {"json", "yaml"}:
        raise HTTPException(status_code=400, detail="format must be json|yaml")
    path = data_paths().skillpack_json(pack_id) if fmt == "json" else data_paths().skillpack_yaml(pack_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="SkillPack not found.")
    media = "application/json" if fmt == "json" else "application/yaml"
    return FileResponse(path, media_type=media, filename=path.name)


@api_router.post("/skillpacks/{pack_id}/share")
async def share_skillpack(request: Request, pack_id: str) -> dict:
    require_pack_owner(request=request, pack_id=pack_id)
    token = generate_share_token()
    set_share_token(pack_id=pack_id, share_token=token)
    return {"pack_id": pack_id, "share_token": token, "share_url": f"/pack/{pack_id}?token={token}"}


@api_router.post("/skillpacks/{pack_id}/unshare")
async def unshare_skillpack(request: Request, pack_id: str) -> dict:
    require_pack_owner(request=request, pack_id=pack_id)
    set_share_token(pack_id=pack_id, share_token=None)
    return {"pack_id": pack_id, "share_token": None}
