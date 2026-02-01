from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable

from shared.config import data_paths


def get_db_path() -> Path:
    return data_paths().db_path


def connect() -> sqlite3.Connection:
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def exec_script(sql: str) -> None:
    with connect() as conn:
        conn.executescript(sql)


def fetch_one(query: str, params: Iterable[Any] = ()) -> dict | None:
    with connect() as conn:
        row = conn.execute(query, tuple(params)).fetchone()
        return dict(row) if row is not None else None


def fetch_all(query: str, params: Iterable[Any] = ()) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(query, tuple(params)).fetchall()
        return [dict(r) for r in rows]


def execute(query: str, params: Iterable[Any] = ()) -> None:
    with connect() as conn:
        conn.execute(query, tuple(params))
        conn.commit()


def execute_returning_id(query: str, params: Iterable[Any] = ()) -> int:
    with connect() as conn:
        cur = conn.execute(query, tuple(params))
        conn.commit()
        return int(cur.lastrowid)
