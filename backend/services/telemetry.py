from __future__ import annotations

import json
from datetime import datetime, timezone

from backend.services.db import execute
from backend.services.init_db import init_db


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_job_event(*, job_id: str, level: str, stage: str, message: str, data: dict | None = None) -> None:
    init_db()
    execute(
        """
        INSERT INTO job_events(job_id, ts, level, stage, message, data_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            job_id,
            now_iso(),
            level,
            stage,
            message,
            json.dumps(data, ensure_ascii=False) if data is not None else None,
        ),
    )

