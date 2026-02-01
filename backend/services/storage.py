from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import HTTPException, UploadFile

from backend.services.db import execute, fetch_one
from backend.services.init_db import init_db
from shared.config import data_paths


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def get_pdf_record(pdf_id: str) -> dict | None:
    init_db()
    return fetch_one("SELECT * FROM pdfs WHERE pdf_id = ?", (pdf_id,))


def get_pdf_file_path(pdf_id: str) -> Path:
    path = data_paths().pdf_file(pdf_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="PDF not found.")
    return path


def ensure_pdf_exists(pdf_id: str) -> None:
    _ = get_pdf_file_path(pdf_id)


def ensure_pdf_owned(*, pdf_id: str, owner_token: str) -> dict:
    init_db()
    rec = fetch_one("SELECT * FROM pdfs WHERE pdf_id = ?", (pdf_id,))
    if rec is None:
        raise HTTPException(status_code=404, detail="PDF not found.")
    if rec.get("owner_token") != owner_token:
        raise HTTPException(status_code=403, detail="Forbidden.")
    return rec


async def save_uploaded_pdf(*, file: UploadFile, owner_token: str) -> dict:
    init_db()
    raw = await file.read()
    file_hash = sha256_bytes(raw)
    existing = fetch_one("SELECT * FROM pdfs WHERE owner_token = ? AND file_hash = ?", (owner_token, file_hash))
    if existing is not None:
        return existing

    pdf_id = uuid.uuid4().hex
    pdf_path = data_paths().pdf_file(pdf_id)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(raw)

    execute(
        """
        INSERT INTO pdfs(pdf_id, owner_token, file_hash, original_filename, num_pages, title, toc_json, created_at)
        VALUES (?, ?, ?, ?, 0, NULL, NULL, ?)
        """,
        (pdf_id, owner_token, file_hash, file.filename or "upload.pdf", now_iso()),
    )
    return fetch_one("SELECT * FROM pdfs WHERE pdf_id = ?", (pdf_id,)) or {"pdf_id": pdf_id}
