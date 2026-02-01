from __future__ import annotations

import os
import time
from pathlib import Path

import fitz


def _make_pdf_bytes(title: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_textbox(
        fitz.Rect(50, 40, 545, 110),
        f"{title}\n\nAbstract\nWe study a task that has attracted significant attention in recent years.",
        fontsize=12,
    )
    page.insert_textbox(fitz.Rect(50, 130, 545, 160), "1 Introduction", fontsize=14)
    page.insert_textbox(fitz.Rect(50, 170, 545, 360), "However, existing approaches still lack robustness. In this paper, we propose MethodX.", fontsize=11)
    out = doc.tobytes()
    doc.close()
    return out


def test_worker_processes_jobs(tmp_path: Path) -> None:
    os.environ["PAPER_SKILL_DATA_DIR"] = str(tmp_path / "data")

    from backend.app import app  # noqa: WPS433
    from fastapi.testclient import TestClient  # noqa: WPS433
    from worker.runner import run_worker_once  # noqa: WPS433

    client = TestClient(app)
    owner_token = "queue_owner"

    def create_job(pack_name: str) -> str:
        pdf_ids: list[str] = []
        for i in range(3):
            files = {"file": (f"paper_{i+1}.pdf", _make_pdf_bytes(f"{pack_name} {i+1}"), "application/pdf")}
            r = client.post("/api/pdfs/upload", files=files, headers={"X-Owner-Token": owner_token})
            assert r.status_code == 200, r.text
            pdf_ids.append(r.json()["pdf_id"])
        r = client.post("/api/skillpacks/build", json={"pdf_ids": pdf_ids, "pack_name": pack_name}, headers={"X-Owner-Token": owner_token})
        assert r.status_code == 200, r.text
        return r.json()["job_id"]

    job1 = create_job("Pack1")
    job2 = create_job("Pack2")

    # Before running worker, jobs should be queued
    r = client.get(f"/api/jobs/{job1}", headers={"X-Pack-Token": owner_token})
    assert r.status_code == 200
    assert r.json()["status"] in {"queued", "running"}

    run_worker_once()
    r = client.get(f"/api/jobs/{job1}", headers={"X-Pack-Token": owner_token})
    assert r.status_code == 200
    assert r.json()["status"] == "succeeded"

    run_worker_once()
    r = client.get(f"/api/jobs/{job2}", headers={"X-Pack-Token": owner_token})
    assert r.status_code == 200
    assert r.json()["status"] == "succeeded"

    # No more work
    assert run_worker_once() == 0

