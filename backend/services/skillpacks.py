from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from backend.services.db import execute, fetch_one
from backend.services.init_db import init_db


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_skillpack_row(pack_id: str) -> dict | None:
    init_db()
    return fetch_one("SELECT * FROM skillpacks WHERE pack_id = ?", (pack_id,))


def upsert_skillpack_row(*, pack_id: str, owner_token: str, pack_name: str, pdf_ids: list[str]) -> None:
    init_db()
    execute(
        """
        INSERT INTO skillpacks(pack_id, owner_token, pack_name, pdf_ids_json, visibility, share_token, created_at)
        VALUES (?, ?, ?, ?, 'private', NULL, ?)
        ON CONFLICT(pack_id) DO UPDATE SET
          owner_token = excluded.owner_token,
          pack_name = excluded.pack_name,
          pdf_ids_json = excluded.pdf_ids_json
        """,
        (pack_id, owner_token, pack_name, json.dumps(pdf_ids, ensure_ascii=False), now_iso()),
    )


def set_share_token(*, pack_id: str, share_token: str | None) -> str | None:
    init_db()
    execute("UPDATE skillpacks SET share_token = ? WHERE pack_id = ?", (share_token, pack_id))
    return share_token


def generate_share_token() -> str:
    return uuid.uuid4().hex

