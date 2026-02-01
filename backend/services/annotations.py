from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from fastapi import HTTPException

from backend.services.db import execute, fetch_all, fetch_one
from backend.services.init_db import init_db


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_bbox_norm(raw: str) -> list[float]:
    try:
        val = json.loads(raw or "[]")
    except Exception:  # noqa: BLE001
        val = []
    if not (isinstance(val, list) and len(val) == 4 and all(isinstance(x, (int, float)) for x in val)):
        return []
    x0, y0, x1, y1 = [float(x) for x in val]
    return [x0, y0, x1, y1]


def list_annotations(*, owner_token: str, pack_id: str, pdf_id: str) -> list[dict]:
    init_db()
    rows = fetch_all(
        """
        SELECT annotation_id, owner_token, pack_id, pdf_id, page_index, bbox_norm_json, note, skill_ref_type, skill_ref_id, skill_title, created_at, updated_at
        FROM annotations
        WHERE owner_token = ? AND pack_id = ? AND pdf_id = ?
        ORDER BY created_at ASC
        """,
        (owner_token, pack_id, pdf_id),
    )
    out: list[dict] = []
    for r in rows:
        out.append(
            {
                "annotation_id": r["annotation_id"],
                "pack_id": r.get("pack_id"),
                "pdf_id": r.get("pdf_id"),
                "page_index": int(r.get("page_index") or 1),
                "bbox_norm": _parse_bbox_norm(r.get("bbox_norm_json") or "[]"),
                "note": r.get("note"),
                "skill_ref_type": r.get("skill_ref_type"),
                "skill_ref_id": r.get("skill_ref_id"),
                "skill_title": r.get("skill_title"),
                "created_at": r.get("created_at"),
                "updated_at": r.get("updated_at"),
            }
        )
    return out


def create_annotation(
    *,
    owner_token: str,
    pack_id: str | None,
    pdf_id: str,
    page_index: int,
    bbox_norm: list[float],
    note: str | None,
    skill_ref_type: str | None,
    skill_ref_id: str | None,
    skill_title: str | None,
) -> dict:
    init_db()

    if not isinstance(pdf_id, str) or not pdf_id.strip():
        raise HTTPException(status_code=400, detail="pdf_id is required.")
    if not isinstance(page_index, int) or page_index < 1:
        raise HTTPException(status_code=400, detail="page_index must be >= 1.")
    if not (isinstance(bbox_norm, list) and len(bbox_norm) == 4 and all(isinstance(x, (int, float)) for x in bbox_norm)):
        raise HTTPException(status_code=400, detail="bbox_norm must be a list of 4 numbers.")

    x0, y0, x1, y1 = [float(x) for x in bbox_norm]
    x0, x1 = (min(x0, x1), max(x0, x1))
    y0, y1 = (min(y0, y1), max(y0, y1))
    x0 = max(0.0, min(1.0, x0))
    x1 = max(0.0, min(1.0, x1))
    y0 = max(0.0, min(1.0, y0))
    y1 = max(0.0, min(1.0, y1))

    if x1 - x0 < 0.002 or y1 - y0 < 0.002:
        raise HTTPException(status_code=400, detail="Annotation bbox too small.")

    annotation_id = uuid.uuid4().hex
    ts = now_iso()
    execute(
        """
        INSERT INTO annotations(annotation_id, owner_token, pack_id, pdf_id, page_index, bbox_norm_json, note, skill_ref_type, skill_ref_id, skill_title, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            annotation_id,
            owner_token,
            pack_id,
            pdf_id,
            int(page_index),
            json.dumps([x0, y0, x1, y1]),
            note,
            skill_ref_type,
            skill_ref_id,
            skill_title,
            ts,
            ts,
        ),
    )
    row = fetch_one(
        "SELECT annotation_id, owner_token, pack_id, pdf_id, page_index, bbox_norm_json, note, skill_ref_type, skill_ref_id, skill_title, created_at, updated_at FROM annotations WHERE annotation_id = ?",
        (annotation_id,),
    )
    if row is None:
        raise HTTPException(status_code=500, detail="Failed to create annotation.")
    return {
        "annotation_id": row["annotation_id"],
        "pack_id": row.get("pack_id"),
        "pdf_id": row.get("pdf_id"),
        "page_index": int(row.get("page_index") or 1),
        "bbox_norm": _parse_bbox_norm(row.get("bbox_norm_json") or "[]"),
        "note": row.get("note"),
        "skill_ref_type": row.get("skill_ref_type"),
        "skill_ref_id": row.get("skill_ref_id"),
        "skill_title": row.get("skill_title"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


def delete_annotation(*, owner_token: str, annotation_id: str) -> None:
    init_db()
    row = fetch_one("SELECT annotation_id, owner_token FROM annotations WHERE annotation_id = ?", (annotation_id,))
    if row is None:
        raise HTTPException(status_code=404, detail="Annotation not found.")
    if row.get("owner_token") != owner_token:
        raise HTTPException(status_code=403, detail="Owner token required.")
    execute("DELETE FROM annotations WHERE annotation_id = ?", (annotation_id,))

