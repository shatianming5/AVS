from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def stage_cache_path(*, data_dir: Path, stage: str, key: str) -> Path:
    safe_stage = stage.replace("/", "_")
    return data_dir / "cache" / safe_stage / f"{key}.json"


def load_stage_payload(*, path: Path, version: int, input_hash: str) -> dict | list | None:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    if raw.get("version") != version:
        return None
    if raw.get("input_hash") != input_hash:
        return None
    return raw.get("payload")


def save_stage_payload(*, path: Path, version: int, input_hash: str, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "version": int(version),
        "input_hash": str(input_hash),
        "created_at": now_iso(),
        "payload": payload,
    }
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

