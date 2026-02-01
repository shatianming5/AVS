from __future__ import annotations

import json
import sqlite3
import uuid

from backend.services.db import connect


LATEST_SCHEMA_VERSION = 2


def migrate() -> None:
    with connect() as conn:
        cur = conn.execute("PRAGMA user_version;").fetchone()
        current = int(cur[0] if cur else 0)
        while current < LATEST_SCHEMA_VERSION:
            if current == 0:
                _migrate_0_to_1(conn)
                current = 1
                conn.execute("PRAGMA user_version = 1;")
                conn.commit()
                continue
            if current == 1:
                _migrate_1_to_2(conn)
                current = 2
                conn.execute("PRAGMA user_version = 2;")
                conn.commit()
                continue

            raise RuntimeError(f"Unsupported schema version: {current}")

        _ensure_indexes(conn)
        return


def _migrate_0_to_1(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS pdfs (
          pdf_id TEXT PRIMARY KEY,
          owner_token TEXT NOT NULL DEFAULT '',
          file_hash TEXT NOT NULL,
          original_filename TEXT NOT NULL,
          num_pages INTEGER NOT NULL DEFAULT 0,
          title TEXT,
          toc_json TEXT,
          created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS jobs (
          job_id TEXT PRIMARY KEY,
          kind TEXT NOT NULL,
          status TEXT NOT NULL,
          progress REAL NOT NULL,
          stage TEXT NOT NULL,
          error TEXT,
          payload_json TEXT NOT NULL,
          result_json TEXT,
          metrics_json TEXT,
          owner_token TEXT NOT NULL DEFAULT '',
          attempt INTEGER NOT NULL DEFAULT 0,
          lease_until TEXT,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS skillpacks (
          pack_id TEXT PRIMARY KEY,
          owner_token TEXT NOT NULL DEFAULT '',
          pack_name TEXT NOT NULL,
          pdf_ids_json TEXT NOT NULL,
          visibility TEXT NOT NULL DEFAULT 'private',
          share_token TEXT,
          created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS job_events (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          job_id TEXT NOT NULL,
          ts TEXT NOT NULL,
          level TEXT NOT NULL,
          stage TEXT NOT NULL,
          message TEXT NOT NULL,
          data_json TEXT,
          FOREIGN KEY(job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
        );
        """
    )

    # Add missing columns to legacy tables (best-effort).
    _ensure_column(conn, "pdfs", "owner_token", "TEXT", not_null=True, default_sql="''")
    _ensure_column(conn, "pdfs", "num_pages", "INTEGER", not_null=True, default_sql="0")
    _ensure_column(conn, "pdfs", "title", "TEXT")
    _ensure_column(conn, "pdfs", "toc_json", "TEXT")

    _ensure_column(conn, "jobs", "metrics_json", "TEXT")
    _ensure_column(conn, "jobs", "owner_token", "TEXT", not_null=True, default_sql="''")
    _ensure_column(conn, "jobs", "attempt", "INTEGER", not_null=True, default_sql="0")
    _ensure_column(conn, "jobs", "lease_until", "TEXT")

    _ensure_column(conn, "skillpacks", "owner_token", "TEXT", not_null=True, default_sql="''")
    _ensure_column(conn, "skillpacks", "visibility", "TEXT", not_null=True, default_sql="'private'")
    _ensure_column(conn, "skillpacks", "share_token", "TEXT")

    # Backfill legacy empty owner_token to a non-empty placeholder to avoid accidental "empty token" access.
    legacy_token = f"legacy_{uuid.uuid4().hex}"
    conn.execute("UPDATE pdfs SET owner_token = ? WHERE owner_token = ''", (legacy_token,))
    conn.execute("UPDATE jobs SET owner_token = ? WHERE owner_token = ''", (legacy_token,))
    conn.execute("UPDATE skillpacks SET owner_token = ? WHERE owner_token = ''", (legacy_token,))

    # Ensure payload/result/metrics are valid JSON strings where present
    _coerce_json_columns(conn, "jobs", ["payload_json", "result_json", "metrics_json"])
    _coerce_json_columns(conn, "pdfs", ["toc_json"])

    _ensure_indexes(conn)
    conn.commit()


def _ensure_indexes(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_pdfs_owner_hash ON pdfs(owner_token, file_hash);
        CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
        CREATE INDEX IF NOT EXISTS idx_jobs_lease ON jobs(lease_until);
        CREATE INDEX IF NOT EXISTS idx_job_events_job ON job_events(job_id);
        CREATE INDEX IF NOT EXISTS idx_skillpacks_share ON skillpacks(share_token);
        CREATE INDEX IF NOT EXISTS idx_skillpacks_owner ON skillpacks(owner_token);
        """
    )
    if _table_exists(conn, "annotations"):
        conn.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_annotations_owner_pack_pdf ON annotations(owner_token, pack_id, pdf_id);
            CREATE INDEX IF NOT EXISTS idx_annotations_pdf_page ON annotations(pdf_id, page_index);
            """
        )


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
    return {str(r[1]) for r in rows}

def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1;", (table,)).fetchone()
    return row is not None


def _ensure_column(
    conn: sqlite3.Connection,
    table: str,
    name: str,
    col_type: str,
    *,
    not_null: bool = False,
    default_sql: str | None = None,
) -> None:
    cols = _table_columns(conn, table)
    if name in cols:
        return
    sql = f"ALTER TABLE {table} ADD COLUMN {name} {col_type}"
    if not_null:
        sql += " NOT NULL"
    if default_sql is not None:
        sql += f" DEFAULT {default_sql}"
    conn.execute(sql)


def _coerce_json_columns(conn: sqlite3.Connection, table: str, columns: list[str]) -> None:
    existing = _table_columns(conn, table)
    cols = [c for c in columns if c in existing]
    if not cols:
        return

    # Best-effort: if invalid JSON, wrap as JSON string.
    rows = conn.execute(f"SELECT rowid, {', '.join(cols)} FROM {table};").fetchall()
    for row in rows:
        rowid = row[0]
        updates: dict[str, str] = {}
        for i, col in enumerate(cols, start=1):
            val = row[i]
            if val is None:
                continue
            if not isinstance(val, str):
                updates[col] = json.dumps(val, ensure_ascii=False)
                continue
            try:
                json.loads(val)
            except Exception:  # noqa: BLE001
                updates[col] = json.dumps(val, ensure_ascii=False)
        if updates:
            set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
            params = list(updates.values()) + [rowid]
            conn.execute(f"UPDATE {table} SET {set_clause} WHERE rowid = ?", params)


def _migrate_1_to_2(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS annotations (
          annotation_id TEXT PRIMARY KEY,
          owner_token TEXT NOT NULL DEFAULT '',
          pack_id TEXT,
          pdf_id TEXT NOT NULL,
          page_index INTEGER NOT NULL,
          bbox_norm_json TEXT NOT NULL,
          note TEXT,
          skill_ref_type TEXT,
          skill_ref_id TEXT,
          skill_title TEXT,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL
        );
        """
    )

    _ensure_column(conn, "annotations", "owner_token", "TEXT", not_null=True, default_sql="''")
    _ensure_column(conn, "annotations", "pack_id", "TEXT")
    _ensure_column(conn, "annotations", "pdf_id", "TEXT", not_null=True, default_sql="''")
    _ensure_column(conn, "annotations", "page_index", "INTEGER", not_null=True, default_sql="1")
    _ensure_column(conn, "annotations", "bbox_norm_json", "TEXT", not_null=True, default_sql="'[]'")
    _ensure_column(conn, "annotations", "note", "TEXT")
    _ensure_column(conn, "annotations", "skill_ref_type", "TEXT")
    _ensure_column(conn, "annotations", "skill_ref_id", "TEXT")
    _ensure_column(conn, "annotations", "skill_title", "TEXT")
    _ensure_column(conn, "annotations", "created_at", "TEXT", not_null=True, default_sql="''")
    _ensure_column(conn, "annotations", "updated_at", "TEXT", not_null=True, default_sql="''")

    _ensure_indexes(conn)
    conn.commit()
