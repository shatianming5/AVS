from __future__ import annotations

import json
from pathlib import Path

import fitz


def extract_pdf_metadata(*, pdf_id: str, data_dir: Path) -> dict:
    pdf_path = data_dir / "pdfs" / f"{pdf_id}.pdf"
    doc = fitz.open(pdf_path)
    try:
        num_pages = int(doc.page_count)
        title = (doc.metadata.get("title") or "").strip() or None
        toc_raw = doc.get_toc() or []
        toc: list[dict] = []
        for item in toc_raw:
            if not item or len(item) < 3:
                continue
            level, t, page = item[0], item[1], item[2]
            toc.append({"level": int(level), "title": str(t), "page_index": int(page)})
        return {"num_pages": num_pages, "title": title, "toc": toc}
    finally:
        doc.close()


def cache_pdf_metadata(*, pdf_id: str, data_dir: Path, meta: dict) -> None:
    out = data_dir / "cache" / "pdf_meta" / f"{pdf_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

