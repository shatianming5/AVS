from __future__ import annotations

from backend.services.migrations import migrate


def init_db() -> None:
    migrate()

