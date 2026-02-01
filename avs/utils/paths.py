from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    base = os.environ.get("AVS_DATA_DIR")
    if base:
        return Path(base)
    return repo_root() / "data"

