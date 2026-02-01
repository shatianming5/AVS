from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class DataPaths:
    root: Path

    @property
    def db_path(self) -> Path:
        return self.root / "paper_skill.sqlite3"

    @property
    def cache_root(self) -> Path:
        return self.root / "cache"

    def pdf_file(self, pdf_id: str) -> Path:
        return self.root / "pdfs" / f"{pdf_id}.pdf"

    def blocks_json(self, pdf_id: str) -> Path:
        return self.root / "blocks" / f"{pdf_id}.json"

    def page_png(self, *, pdf_id: str, page_index: int, zoom: float) -> Path:
        z = str(zoom).replace(".", "_")
        return self.root / "pages" / pdf_id / f"p{page_index}_z{z}.png"

    def skillpack_json(self, pack_id: str) -> Path:
        return self.root / "skillpacks" / f"{pack_id}.json"

    def skillpack_yaml(self, pack_id: str) -> Path:
        return self.root / "skillpacks" / f"{pack_id}.yaml"

    def evidence_index_json(self, pack_id: str) -> Path:
        return self.root / "evidence" / f"{pack_id}.json"


def data_paths() -> DataPaths:
    return DataPaths(root=Path(os.environ.get("PAPER_SKILL_DATA_DIR", "data")))


def ensure_data_dirs() -> None:
    p = data_paths().root
    (p / "pdfs").mkdir(parents=True, exist_ok=True)
    (p / "pages").mkdir(parents=True, exist_ok=True)
    (p / "blocks").mkdir(parents=True, exist_ok=True)
    (p / "skillpacks").mkdir(parents=True, exist_ok=True)
    (p / "evidence").mkdir(parents=True, exist_ok=True)
    (p / "cache").mkdir(parents=True, exist_ok=True)
