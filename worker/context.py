from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BuildContext:
    job_id: str
    pack_id: str
    pdf_ids: list[str]
    pack_name: str
    field_hint: str | None
    target_venue_hint: str | None
    language: str
    data_dir: Path
    owner_token: str
