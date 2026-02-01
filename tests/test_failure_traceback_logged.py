from __future__ import annotations

import os
import time
from pathlib import Path

import fitz


def _make_pdf_bytes(title: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_textbox(fitz.Rect(50, 40, 545, 110), f"{title}\n\nAbstract\nWe study a task.", fontsize=12)
    page.insert_textbox(fitz.Rect(50, 130, 545, 160), "1 Introduction", fontsize=14)
    page.insert_textbox(fitz.Rect(50, 170, 545, 260), "However, this is a test intro.", fontsize=11)
    out = doc.tobytes()
    doc.close()
    return out


def test_failure_logs_traceback(tmp_path: Path) -> None:
    os.environ["PAPER_SKILL_DATA_DIR"] = str(tmp_path / "data")

    from backend.app import app
    from fastapi.testclient import TestClient

    client = TestClient(app)
    owner_token = "test_owner_token"

    pdf_ids: list[str] = []
    for i in range(3):
        files = {"file": (f"paper_{i+1}.pdf", _make_pdf_bytes(f"Synthetic {i+1}"), "application/pdf")}
        r = client.post("/api/pdfs/upload", files=files, headers={"X-Owner-Token": owner_token})
        assert r.status_code == 200, r.text
        pdf_ids.append(r.json()["pdf_id"])

    # Delete one underlying file after upload to force a worker-side failure.
    from shared.config import data_paths

    victim = pdf_ids[0]
    data_paths().pdf_file(victim).unlink()

    r = client.post(
        "/api/skillpacks/build",
        json={"pdf_ids": pdf_ids, "pack_name": "ShouldFail", "field_hint": "NLP", "language": "English"},
        headers={"X-Owner-Token": owner_token},
    )
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]

    from worker.runner import run_worker_once

    # Force a terminal failure on first attempt.
    run_worker_once(max_attempts=1)

    # Wait a moment for DB update.
    deadline = time.time() + 5
    job = None
    while time.time() < deadline:
        jr = client.get(f"/api/jobs/{job_id}", headers={"X-Pack-Token": owner_token})
        assert jr.status_code == 200, jr.text
        job = jr.json()
        if job["status"] == "failed":
            break
        time.sleep(0.1)
    assert job is not None
    assert job["status"] == "failed"

    ev = client.get(f"/api/jobs/{job_id}/events", headers={"X-Pack-Token": owner_token})
    assert ev.status_code == 200, ev.text
    events = ev.json().get("events") or []
    failed = [e for e in events if e.get("stage") == "failed"]
    assert failed, events
    data = failed[-1].get("data") or {}
    assert "traceback" in data
    tb = str(data["traceback"])
    assert len(tb) <= 8192 + 64

