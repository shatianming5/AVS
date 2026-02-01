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
    intro = (
        "Recent progress in representation learning has enabled strong performance across tasks. "
        "However, existing approaches still lack robustness under distribution shift. "
        "In this paper, we propose MethodX, which may improve stability by combining calibration and data augmentation. "
        "Our contributions are threefold: (1) a simple method; (2) a theoretical insight; (3) extensive experiments. "
        "The rest of this paper is organized as follows: Section 2 describes the method; Section 3 reports results."
    )
    page.insert_textbox(fitz.Rect(50, 170, 545, 360), intro, fontsize=11)
    page.insert_textbox(fitz.Rect(50, 380, 545, 420), "Figure 1: Overview of the proposed pipeline.", fontsize=10)
    page.insert_textbox(fitz.Rect(50, 440, 545, 470), "2 Method", fontsize=14)

    out = doc.tobytes()
    doc.close()
    return out


def _wait_job(client, *, job_id: str, owner_token: str, timeout_s: float = 30.0) -> dict:
    from worker.runner import run_worker_once

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        run_worker_once()
        jr = client.get(f"/api/jobs/{job_id}", headers={"X-Pack-Token": owner_token})
        assert jr.status_code == 200, jr.text
        job = jr.json()
        if job["status"] == "failed":
            raise AssertionError(job.get("error") or "Job failed")
        if job["status"] == "succeeded":
            return job
        time.sleep(0.2)
    raise AssertionError("Timeout waiting for job")


def test_stage_cache_hits(tmp_path: Path) -> None:
    os.environ["PAPER_SKILL_DATA_DIR"] = str(tmp_path / "data")

    from backend.app import app
    from fastapi.testclient import TestClient

    client = TestClient(app)
    owner_token = "test_owner_token"

    # Upload 3 PDFs
    pdf_ids: list[str] = []
    for i in range(3):
        content = _make_pdf_bytes(f"Synthetic Paper {i+1}")
        files = {"file": (f"paper_{i+1}.pdf", content, "application/pdf")}
        r = client.post("/api/pdfs/upload", files=files, headers={"X-Owner-Token": owner_token})
        assert r.status_code == 200, r.text
        pdf_ids.append(r.json()["pdf_id"])

    payload = {
        "pdf_ids": pdf_ids,
        "pack_name": "Pack A",
        "field_hint": "NLP",
        "target_venue_hint": "NeurIPS",
        "language": "English",
    }

    r1 = client.post("/api/skillpacks/build", json=payload, headers={"X-Owner-Token": owner_token})
    assert r1.status_code == 200, r1.text
    job1 = _wait_job(client, job_id=r1.json()["job_id"], owner_token=owner_token)
    assert job1["result"] and job1["result"].get("pack_id")

    payload["pack_name"] = "Pack B"
    r2 = client.post("/api/skillpacks/build", json=payload, headers={"X-Owner-Token": owner_token})
    assert r2.status_code == 200, r2.text
    job2 = _wait_job(client, job_id=r2.json()["job_id"], owner_token=owner_token)

    metrics = job2.get("metrics") or {}
    cache_hits = metrics.get("cache_hits") or {}

    # Per-pdf stages
    for stage in ["sections", "locate_intro", "style_features", "moves"]:
        assert stage in cache_hits, cache_hits
        for pdf_id in pdf_ids:
            assert cache_hits[stage].get(pdf_id) is True, (stage, cache_hits[stage])

    # Per-pack stages (single key)
    for stage in ["blueprint", "templates", "storyboard", "quality"]:
        assert stage in cache_hits, cache_hits
        assert any(bool(v) for v in (cache_hits[stage] or {}).values()) is True, (stage, cache_hits[stage])

