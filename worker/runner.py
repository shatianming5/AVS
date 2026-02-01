from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from datetime import datetime, timedelta, timezone

from backend.services.db import connect, fetch_one
from backend.services.init_db import init_db
from backend.services.jobs import finish_job, update_job
from backend.services.pipeline_runner import build_skillpack_pipeline
from backend.services.telemetry import log_job_event
from shared.env import load_env


load_env()


def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _truncate_traceback(tb: str, *, limit: int = 8192) -> tuple[str, bool]:
    t = tb or ""
    if len(t) <= limit:
        return t, False
    return t[: max(0, limit - 20)] + "\n…(traceback truncated)…", True


def now_iso() -> str:
    return now_utc().isoformat()


def _lease_until_iso(seconds: int) -> str:
    return (now_utc() + timedelta(seconds=seconds)).isoformat()


def _claim_one_job(*, lease_seconds: int) -> str | None:
    init_db()
    with connect() as conn:
        conn.execute("BEGIN IMMEDIATE;")
        row = conn.execute(
            """
            SELECT job_id, attempt
            FROM jobs
            WHERE status = 'queued' AND (lease_until IS NULL OR lease_until <= ?)
            ORDER BY created_at ASC
            LIMIT 1
            """,
            (now_iso(),),
        ).fetchone()
        if row is None:
            conn.execute("COMMIT;")
            return None
        job_id = str(row["job_id"])
        attempt = int(row["attempt"] or 0) + 1
        conn.execute(
            """
            UPDATE jobs
            SET status = 'running',
                stage = 'starting',
                progress = 0.01,
                attempt = ?,
                lease_until = ?,
                updated_at = ?
            WHERE job_id = ?
            """,
            (attempt, _lease_until_iso(lease_seconds), now_iso(), job_id),
        )
        conn.execute("COMMIT;")
        return job_id


def _retry_backoff_seconds(attempt: int) -> int:
    # 1st failure -> 5s, then 15s, then 45s...
    return int(min(300, 5 * (3 ** max(0, attempt - 1))))


def run_worker_once(*, lease_seconds: int = 600, max_attempts: int = 3) -> int:
    job_id = _claim_one_job(lease_seconds=lease_seconds)
    if job_id is None:
        return 0

    job_row = fetch_one("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
    if job_row is None:
        return 0
    payload = json.loads(job_row["payload_json"])

    log_job_event(job_id=job_id, level="info", stage="starting", message="Worker claimed job.", data={"attempt": job_row.get("attempt")})

    try:
        result = build_skillpack_pipeline(job_id=job_id, payload=payload, on_progress=_progress_cb(job_id))
        finish_job(job_id, result=result)
        return 1
    except Exception as e:  # noqa: BLE001
        tb_full = traceback.format_exc()
        tb, truncated = _truncate_traceback(tb_full)
        attempt = int(job_row.get("attempt") or 1)
        if attempt >= max_attempts:
            from backend.services.jobs import fail_job

            fail_job(job_id, error=str(e), traceback_str=tb)
            return 1

        wait_s = _retry_backoff_seconds(attempt)
        with connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'queued',
                    stage = 'retry_wait',
                    error = ?,
                    lease_until = ?,
                    updated_at = ?
                WHERE job_id = ?
                """,
                (str(e), _lease_until_iso(wait_s), now_iso(), job_id),
            )
            conn.commit()
        log_job_event(
            job_id=job_id,
            level="error",
            stage="retry_wait",
            message="Job error; will retry.",
            data={"error": str(e), "wait_s": wait_s, "traceback": tb, "traceback_truncated": truncated or None},
        )
        return 1


def run_worker_loop(*, poll_interval_s: float = 0.5, lease_seconds: int = 600, max_attempts: int = 3) -> None:
    init_db()
    while True:
        processed = run_worker_once(lease_seconds=lease_seconds, max_attempts=max_attempts)
        if processed == 0:
            time.sleep(poll_interval_s)


def _progress_cb(job_id: str):
    def cb(stage: str, progress: float) -> None:
        update_job(job_id, stage=stage, progress=progress)

    return cb


def main() -> None:
    parser = argparse.ArgumentParser(description="paper_skill worker runner")
    parser.add_argument("--once", action="store_true", help="Process at most one job then exit")
    parser.add_argument("--poll-interval", type=float, default=float(os.environ.get("PAPER_SKILL_WORKER_POLL", "0.5")))
    parser.add_argument("--lease-seconds", type=int, default=int(os.environ.get("PAPER_SKILL_WORKER_LEASE", "600")))
    parser.add_argument("--max-attempts", type=int, default=int(os.environ.get("PAPER_SKILL_WORKER_MAX_ATTEMPTS", "3")))
    args = parser.parse_args()

    if args.once:
        run_worker_once(lease_seconds=args.lease_seconds, max_attempts=args.max_attempts)
        return
    run_worker_loop(poll_interval_s=args.poll_interval, lease_seconds=args.lease_seconds, max_attempts=args.max_attempts)


if __name__ == "__main__":
    main()
