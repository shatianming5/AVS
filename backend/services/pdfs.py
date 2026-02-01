from __future__ import annotations

import json

from backend.services.db import execute
from backend.services.init_db import init_db


def update_pdf_metadata(*, pdf_id: str, num_pages: int, title: str | None, toc: list[dict] | None) -> None:
    init_db()
    execute(
        "UPDATE pdfs SET num_pages = ?, title = ?, toc_json = ? WHERE pdf_id = ?",
        (int(num_pages), title, json.dumps(toc, ensure_ascii=False) if toc is not None else None, pdf_id),
    )

