from __future__ import annotations

import os
from pathlib import Path

import fitz


def _make_pdf_bytes_multiline_caption() -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_textbox(fitz.Rect(50, 130, 545, 160), "1 Introduction", fontsize=14)
    page.insert_textbox(
        fitz.Rect(50, 170, 545, 260),
        "We refer to Fig. 1 for an overview.",
        fontsize=11,
    )
    # Two adjacent blocks that should be merged into one caption with bbox_list
    page.insert_textbox(fitz.Rect(50, 380, 545, 405), "Figure 1: Overview of the proposed pipeline.", fontsize=10)
    page.insert_textbox(fitz.Rect(50, 405, 545, 430), "Continued caption line with more details.", fontsize=10)

    out = doc.tobytes()
    doc.close()
    return out


def test_caption_merge_produces_bbox_list(tmp_path: Path) -> None:
    os.environ["PAPER_SKILL_DATA_DIR"] = str(tmp_path / "data")

    from backend.app import app
    from fastapi.testclient import TestClient

    client = TestClient(app)
    owner_token = "test_owner_token"

    files = {"file": ("paper.pdf", _make_pdf_bytes_multiline_caption(), "application/pdf")}
    r = client.post("/api/pdfs/upload", files=files, headers={"X-Owner-Token": owner_token})
    assert r.status_code == 200, r.text
    pdf_id = r.json()["pdf_id"]

    from shared.config import data_paths
    from worker.pipeline.extract_blocks import extract_blocks_cached
    from worker.pipeline.evidence import EvidenceBuilder

    blocks = extract_blocks_cached(pdf_id, data_paths().root)
    cap = None
    for b in blocks:
        if b.block_type == "caption" and b.bbox_list and len(b.bbox_list) >= 2:
            cap = b
            break

    assert cap is not None, [b.text for b in blocks if b.block_type == "caption"]
    assert isinstance(cap.bbox_list, list)
    assert len(cap.bbox_list) >= 2

    ev = EvidenceBuilder(data_dir=data_paths().root).from_block(block=cap, reason="test", confidence=1.0, kind="caption")
    assert ev.bbox_norm_list is not None
    assert len(ev.bbox_norm_list) >= 2
