from __future__ import annotations

import os
import time
from pathlib import Path

import fitz


def _make_pdf_bytes(title: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4-ish points
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


def test_end_to_end_build(tmp_path: Path) -> None:
    os.environ["PAPER_SKILL_DATA_DIR"] = str(tmp_path / "data")

    # Import after env var to ensure config points at tmpdir.
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

    # Start build
    r = client.post(
        "/api/skillpacks/build",
        json={"pdf_ids": pdf_ids, "pack_name": "Test Pack", "field_hint": "NLP", "target_venue_hint": "NeurIPS", "language": "English"},
        headers={"X-Owner-Token": owner_token},
    )
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]

    from worker.runner import run_worker_once

    pack_id = None
    deadline = time.time() + 30
    while time.time() < deadline:
        run_worker_once()
        jr = client.get(f"/api/jobs/{job_id}", headers={"X-Pack-Token": owner_token})
        assert jr.status_code == 200, jr.text
        job = jr.json()
        if job["status"] == "failed":
            raise AssertionError(job.get("error") or "Job failed")
        if job["status"] == "succeeded":
            pack_id = job["result"]["pack_id"]
            break
        time.sleep(0.2)

    assert pack_id is not None

    # Fetch pack and evidence
    pack = client.get(f"/api/skillpacks/{pack_id}", headers={"X-Pack-Token": owner_token})
    assert pack.status_code == 200, pack.text
    evidence = client.get(f"/api/evidence/{pack_id}", headers={"X-Pack-Token": owner_token})
    assert evidence.status_code == 200, evidence.text

    pack_json = pack.json()
    evidence_json = evidence.json()

    assert pack_json["pack_id"] == pack_id
    assert len(pack_json["pdf_ids"]) == 3
    assert "intro_blueprint" in pack_json
    assert len(evidence_json.get("evidence", [])) >= 1
