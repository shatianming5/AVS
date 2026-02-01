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
    out = doc.tobytes()
    doc.close()
    return out


def _build_pack(*, tmp_path: Path, owner_token: str) -> tuple[str, list[str]]:
    os.environ["PAPER_SKILL_DATA_DIR"] = str(tmp_path / "data")

    from backend.app import app  # noqa: WPS433
    from fastapi.testclient import TestClient  # noqa: WPS433
    from worker.runner import run_worker_once  # noqa: WPS433

    client = TestClient(app)

    pdf_ids: list[str] = []
    for i in range(3):
        files = {"file": (f"paper_{i+1}.pdf", _make_pdf_bytes(f"Synthetic Paper {i+1}"), "application/pdf")}
        r = client.post("/api/pdfs/upload", files=files, headers={"X-Owner-Token": owner_token})
        assert r.status_code == 200, r.text
        pdf_ids.append(r.json()["pdf_id"])

    r = client.post("/api/skillpacks/build", json={"pdf_ids": pdf_ids, "pack_name": "Privacy Pack"}, headers={"X-Owner-Token": owner_token})
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]

    deadline = time.time() + 30
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
    assert pack_id is not None
    return pack_id, pdf_ids


def test_pack_access_tokens(tmp_path: Path) -> None:
    owner_token = "owner_a"
    other_token = "owner_b"

    pack_id, pdf_ids = _build_pack(tmp_path=tmp_path, owner_token=owner_token)

    from backend.app import app  # noqa: WPS433
    from fastapi.testclient import TestClient  # noqa: WPS433

    client = TestClient(app)

    # No token -> forbidden
    r = client.get(f"/api/skillpacks/{pack_id}")
    assert r.status_code == 403

    # Wrong token -> forbidden
    r = client.get(f"/api/skillpacks/{pack_id}", headers={"X-Pack-Token": other_token})
    assert r.status_code == 403

    # Owner token -> ok
    r = client.get(f"/api/skillpacks/{pack_id}", headers={"X-Pack-Token": owner_token})
    assert r.status_code == 200

    # Share -> get share token
    r = client.post(f"/api/skillpacks/{pack_id}/share", headers={"X-Pack-Token": owner_token})
    assert r.status_code == 200, r.text
    share_token = r.json()["share_token"]
    assert isinstance(share_token, str) and share_token

    # Share token -> ok
    r = client.get(f"/api/skillpacks/{pack_id}", headers={"X-Pack-Token": share_token})
    assert r.status_code == 200

    # Pack-scoped PDF access should work with share token
    r = client.get(f"/api/skillpacks/{pack_id}/pdfs/{pdf_ids[0]}/file", headers={"X-Pack-Token": share_token})
    assert r.status_code == 200

    # Unshare -> share token stops working
    r = client.post(f"/api/skillpacks/{pack_id}/unshare", headers={"X-Pack-Token": owner_token})
    assert r.status_code == 200, r.text
    r = client.get(f"/api/skillpacks/{pack_id}", headers={"X-Pack-Token": share_token})
    assert r.status_code == 403

