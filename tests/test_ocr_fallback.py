from __future__ import annotations

import os
import time
from pathlib import Path

import fitz


def _make_text_pdf_bytes(title: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_textbox(fitz.Rect(50, 40, 545, 110), f"{title}\n\nAbstract\nTest abstract.", fontsize=12)
    page.insert_textbox(fitz.Rect(50, 130, 545, 160), "1 Introduction", fontsize=18)
    page.insert_textbox(
        fitz.Rect(50, 170, 545, 360),
        "However, existing approaches still lack robustness. In this paper, we propose MethodX. The rest of this paper is organized as follows: ...",
        fontsize=14,
    )
    out = doc.tobytes()
    doc.close()
    return out


def _make_scanned_pdf_bytes(title: str) -> bytes:
    # Create a page with text, render to an image, then embed that image into a new PDF.
    base = fitz.open()
    page = base.new_page(width=595, height=842)
    page.insert_textbox(fitz.Rect(50, 40, 545, 110), f"{title}\n\nAbstract\nThis is a scanned PDF test.", fontsize=18)
    page.insert_textbox(fitz.Rect(50, 130, 545, 160), "1 Introduction", fontsize=22)
    page.insert_textbox(
        fitz.Rect(50, 170, 545, 360),
        "However, existing approaches still lack robustness. In this paper, we propose MethodX.",
        fontsize=18,
    )
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    img = pix.tobytes("png")
    base.close()

    doc = fitz.open()
    p = doc.new_page(width=595, height=842)
    p.insert_image(fitz.Rect(0, 0, 595, 842), stream=img)
    out = doc.tobytes()
    doc.close()
    return out


def test_ocr_fallback_sets_flag(tmp_path: Path) -> None:
    os.environ["PAPER_SKILL_DATA_DIR"] = str(tmp_path / "data")

    from backend.app import app  # noqa: WPS433
    from fastapi.testclient import TestClient  # noqa: WPS433
    from worker.runner import run_worker_once  # noqa: WPS433

    client = TestClient(app)
    owner_token = "ocr_owner"

    pdf_ids: list[str] = []
    pdf_bytes = [_make_scanned_pdf_bytes("Scanned 1"), _make_text_pdf_bytes("Text 2"), _make_text_pdf_bytes("Text 3")]
    for i, content in enumerate(pdf_bytes):
        files = {"file": (f"paper_{i+1}.pdf", content, "application/pdf")}
        r = client.post("/api/pdfs/upload", files=files, headers={"X-Owner-Token": owner_token})
        assert r.status_code == 200, r.text
        pdf_ids.append(r.json()["pdf_id"])

    r = client.post("/api/skillpacks/build", json={"pdf_ids": pdf_ids, "pack_name": "OCR Pack"}, headers={"X-Owner-Token": owner_token})
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]

    deadline = time.time() + 60
    pack_id = None
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
    pack = client.get(f"/api/skillpacks/{pack_id}", headers={"X-Pack-Token": owner_token})
    assert pack.status_code == 200, pack.text
    assert pack.json()["quality_report"]["ocr_used"] is True

