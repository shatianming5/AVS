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
    # Single intro block is fine; evidence-first should downgrade any rule that cannot find evidence.
    intro = (
        "Recent progress has enabled strong performance. "
        "However, existing approaches still lack robustness. "
        "In this paper, we propose MethodX. "
        "The rest of this paper is organized as follows: Section 2 describes the method; Section 3 reports results."
    )
    page.insert_textbox(fitz.Rect(50, 170, 545, 320), intro, fontsize=11)
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


def test_evidence_first_rules_have_evidence(tmp_path: Path) -> None:
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

    r = client.post(
        "/api/skillpacks/build",
        json={"pdf_ids": pdf_ids, "pack_name": "EvidenceFirst", "field_hint": "NLP", "language": "English"},
        headers={"X-Owner-Token": owner_token},
    )
    assert r.status_code == 200, r.text
    job = _wait_job(client, job_id=r.json()["job_id"], owner_token=owner_token)
    pack_id = job["result"]["pack_id"]

    pack = client.get(f"/api/skillpacks/{pack_id}", headers={"X-Pack-Token": owner_token})
    assert pack.status_code == 200, pack.text
    bp = (pack.json().get("intro_blueprint") or {})
    story_rules = bp.get("story_rules") or []
    claim_rules = bp.get("claim_rules") or []

    assert len(story_rules) >= 1
    for rule in list(story_rules) + list(claim_rules):
        ev = rule.get("supporting_evidence") or []
        assert isinstance(ev, list)
        assert len(ev) >= 1
        assert all(isinstance(x, str) and x for x in ev)

