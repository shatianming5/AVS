from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

from backend.services.storage import get_pdf_file_path
from shared.config import data_paths


def render_page_png_cached(*, pdf_id: str, page_index: int, zoom: float = 2.0) -> Path:
    out_path = data_paths().page_png(pdf_id=pdf_id, page_index=page_index, zoom=zoom)
    if out_path.exists():
        return out_path

    pdf_path = get_pdf_file_path(pdf_id)
    doc = fitz.open(pdf_path)
    try:
        if page_index < 1 or page_index > doc.page_count:
            raise ValueError(f"page_index out of range: {page_index}/{doc.page_count}")
        page = doc.load_page(page_index - 1)
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pix.save(out_path.as_posix())
        return out_path
    finally:
        doc.close()

