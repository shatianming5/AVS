from __future__ import annotations

import os
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
    page.insert_textbox(
        fitz.Rect(50, 170, 545, 360),
        "However, existing approaches still lack robustness. In this paper, we propose MethodX. The rest of this paper is organized as follows: Section 2...",
        fontsize=11,
    )
    out = doc.tobytes()
    doc.close()
    return out


def test_annotations_crud_and_share_readonly(tmp_path: Path) -> None:
    os.environ["PAPER_SKILL_DATA_DIR"] = str(tmp_path / "data")

    from backend.app import app  # noqa: WPS433
    from fastapi.testclient import TestClient  # noqa: WPS433
    from worker.runner import run_worker_once  # noqa: WPS433

    client = TestClient(app)
    owner_token = "ann_owner"

    pdf_ids: list[str] = []
    for i in range(3):
        files = {"file": (f"paper_{i+1}.pdf", _make_pdf_bytes(f"Ann Test {i+1}"), "application/pdf")}
        r = client.post("/api/pdfs/upload", files=files, headers={"X-Owner-Token": owner_token})
        assert r.status_code == 200, r.text
        pdf_ids.append(r.json()["pdf_id"])

    r = client.post("/api/skillpacks/build", json={"pdf_ids": pdf_ids, "pack_name": "Ann Pack", "field_hint": "cv"}, headers={"X-Owner-Token": owner_token})
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]

    run_worker_once()
    jr = client.get(f"/api/jobs/{job_id}", headers={"X-Pack-Token": owner_token})
    assert jr.status_code == 200, jr.text
    job = jr.json()
    assert job["status"] == "succeeded"
    pack_id = job["result"]["pack_id"]

    pdf_id = pdf_ids[0]

    # Create annotation (owner only)
    cr = client.post(
        "/api/annotations",
        json={
            "pack_id": pack_id,
            "pdf_id": pdf_id,
            "page_index": 1,
            "bbox_norm": [0.1, 0.1, 0.2, 0.2],
            "note": "Important region",
            "skill_ref_type": "template",
            "skill_ref_id": "t0",
            "skill_title": "Template: IntroOpening",
        },
        headers={"X-Pack-Token": owner_token},
    )
    assert cr.status_code == 200, cr.text
    ann = cr.json()["annotation"]
    assert ann["pdf_id"] == pdf_id
    assert ann["page_index"] == 1
    assert ann["note"] == "Important region"
    assert ann["bbox_norm"] == [0.1, 0.1, 0.2, 0.2]
    ann_id = ann["annotation_id"]

    # List annotations (owner)
    lr = client.get(f"/api/annotations?pack_id={pack_id}&pdf_id={pdf_id}", headers={"X-Pack-Token": owner_token})
    assert lr.status_code == 200, lr.text
    anns = lr.json()["annotations"]
    assert any(a["annotation_id"] == ann_id for a in anns)

    # Share token can read but cannot write
    sr = client.post(f"/api/skillpacks/{pack_id}/share", headers={"X-Pack-Token": owner_token})
    assert sr.status_code == 200, sr.text
    share_token = sr.json()["share_token"]

    lr2 = client.get(f"/api/annotations?pack_id={pack_id}&pdf_id={pdf_id}", headers={"X-Pack-Token": share_token})
    assert lr2.status_code == 200, lr2.text
    anns2 = lr2.json()["annotations"]
    assert any(a["annotation_id"] == ann_id for a in anns2)

    cr_forbidden = client.post(
        "/api/annotations",
        json={"pack_id": pack_id, "pdf_id": pdf_id, "page_index": 1, "bbox_norm": [0.3, 0.3, 0.4, 0.4]},
        headers={"X-Pack-Token": share_token},
    )
    assert cr_forbidden.status_code == 403

    dr_forbidden = client.delete(f"/api/annotations/{ann_id}", headers={"X-Pack-Token": share_token})
    assert dr_forbidden.status_code == 403

    # Owner can delete
    dr = client.delete(f"/api/annotations/{ann_id}", headers={"X-Pack-Token": owner_token})
    assert dr.status_code == 200, dr.text

    lr3 = client.get(f"/api/annotations?pack_id={pack_id}&pdf_id={pdf_id}", headers={"X-Pack-Token": owner_token})
    assert lr3.status_code == 200, lr3.text
    assert all(a["annotation_id"] != ann_id for a in lr3.json()["annotations"])

