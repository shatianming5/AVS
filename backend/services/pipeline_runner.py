from __future__ import annotations

from backend.services.storage import ensure_pdf_exists, ensure_pdf_owned
from shared.config import data_paths
from worker.build import build_skillpack


def build_skillpack_pipeline(*, job_id: str, payload: dict, on_progress) -> dict:
    pdf_ids: list[str] = payload["pdf_ids"]
    owner_token: str = payload.get("owner_token") or ""
    for pdf_id in pdf_ids:
        if owner_token:
            ensure_pdf_owned(pdf_id=pdf_id, owner_token=owner_token)
        else:
            ensure_pdf_exists(pdf_id)

    pack_name: str = payload["pack_name"]
    field_hint: str | None = payload.get("field_hint")
    target_venue_hint: str | None = payload.get("target_venue_hint")
    language: str = payload.get("language", "English")
    result = build_skillpack(
        job_id=job_id,
        pdf_ids=pdf_ids,
        pack_name=pack_name,
        field_hint=field_hint,
        target_venue_hint=target_venue_hint,
        language=language,
        owner_token=owner_token,
        data_dir=data_paths().root,
        on_progress=on_progress,
    )
    return result
