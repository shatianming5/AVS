from __future__ import annotations

import json
import os
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone

from backend.services.db import execute, fetch_one
from backend.services.init_db import init_db
from backend.services.pipeline_runner import build_skillpack_pipeline
from backend.services.telemetry import log_job_event


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _truncate_traceback(tb: str, *, limit: int = 8192) -> tuple[str, bool]:
    t = tb or ""
    if len(t) <= limit:
        return t, False
    return t[: max(0, limit - 20)] + "\n…(traceback truncated)…", True


def create_build_job(
    *,
    pdf_ids: list[str],
    pack_name: str,
    field_hint: str | None,
    target_venue_hint: str | None,
    language: str,
    owner_token: str,
) -> dict:
    init_db()
    job_id = uuid.uuid4().hex
    payload = {
        "pdf_ids": pdf_ids,
        "pack_name": pack_name,
        "field_hint": field_hint,
        "target_venue_hint": target_venue_hint,
        "language": language,
        "owner_token": owner_token,
    }
    ts = now_iso()
    execute(
        """
        INSERT INTO jobs(job_id, kind, status, progress, stage, error, payload_json, result_json, metrics_json, owner_token, attempt, lease_until, created_at, updated_at)
        VALUES (?, 'build', 'queued', 0.0, 'queued', NULL, ?, NULL, NULL, ?, 0, NULL, ?, ?)
        """,
        (job_id, json.dumps(payload, ensure_ascii=False), owner_token, ts, ts),
    )
    log_job_event(job_id=job_id, level="info", stage="queued", message="Job queued.", data={"kind": "build"})
    return {"job_id": job_id}


def get_job(job_id: str) -> dict | None:
    init_db()
    job = fetch_one("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
    if job is None:
        return None
    payload = json.loads(job["payload_json"])
    result = json.loads(job["result_json"]) if job["result_json"] else None
    metrics = json.loads(job["metrics_json"]) if job.get("metrics_json") else None
    return {
        "job_id": job["job_id"],
        "kind": job["kind"],
        "status": job["status"],
        "progress": job["progress"],
        "stage": job["stage"],
        "error": job["error"],
        "payload": payload,
        "result": result,
        "metrics": metrics,
        "attempt": job.get("attempt", 0),
        "lease_until": job.get("lease_until"),
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
    }


def update_job(job_id: str, *, status: str | None = None, progress: float | None = None, stage: str | None = None) -> None:
    fields: list[str] = []
    params: list[object] = []
    if status is not None:
        fields.append("status = ?")
        params.append(status)
    if progress is not None:
        fields.append("progress = ?")
        params.append(float(progress))
    if stage is not None:
        fields.append("stage = ?")
        params.append(stage)
    fields.append("updated_at = ?")
    params.append(now_iso())
    params.append(job_id)
    execute(f"UPDATE jobs SET {', '.join(fields)} WHERE job_id = ?", params)
    if stage is not None:
        log_job_event(job_id=job_id, level="info", stage=stage, message="Stage update.", data={"status": status, "progress": progress})


def update_job_metrics(job_id: str, *, metrics: dict) -> None:
    execute(
        "UPDATE jobs SET metrics_json = ?, updated_at = ? WHERE job_id = ?",
        (json.dumps(metrics, ensure_ascii=False), now_iso(), job_id),
    )


def fail_job(job_id: str, *, error: str, traceback_str: str | None = None) -> None:
    execute(
        "UPDATE jobs SET status = 'failed', error = ?, lease_until = NULL, updated_at = ? WHERE job_id = ?",
        (error, now_iso(), job_id),
    )
    data: dict[str, object] = {"error": error}
    if traceback_str:
        tb, truncated = _truncate_traceback(traceback_str)
        data["traceback"] = tb
        if truncated:
            data["traceback_truncated"] = True
    log_job_event(job_id=job_id, level="error", stage="failed", message="Job failed.", data=data)


def finish_job(job_id: str, *, result: dict) -> None:
    execute(
        "UPDATE jobs SET status = 'succeeded', progress = 1.0, stage = 'done', error = NULL, result_json = ?, lease_until = NULL, updated_at = ? WHERE job_id = ?",
        (json.dumps(result, ensure_ascii=False), now_iso(), job_id),
    )
    log_job_event(job_id=job_id, level="info", stage="done", message="Job succeeded.", data={"result": result})


def run_build_in_background(job_id: str) -> None:
    # Default: use the external worker runner (worker/runner.py).
    # For local dev/tests, allow inline background execution.
    if os.environ.get("PAPER_SKILL_INLINE_WORKER") != "1":
        return
    thread = threading.Thread(target=_run_build_job, args=(job_id,), daemon=True)
    thread.start()


def _run_build_job(job_id: str) -> None:
    init_db()
    job_row = fetch_one("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
    if job_row is None:
        return

    payload = json.loads(job_row["payload_json"])
    try:
        update_job(job_id, status="running", progress=0.01, stage="starting")
        result = build_skillpack_pipeline(job_id=job_id, payload=payload, on_progress=_progress_cb(job_id))
        finish_job(job_id, result=result)
    except Exception as e:  # noqa: BLE001
        fail_job(job_id, error=str(e), traceback_str=traceback.format_exc())


def _progress_cb(job_id: str):
    def cb(stage: str, progress: float) -> None:
        update_job(job_id, stage=stage, progress=progress)

    return cb
