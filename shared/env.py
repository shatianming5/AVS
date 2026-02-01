from __future__ import annotations

from pathlib import Path


def load_env() -> None:
    """
    Load `.env` from the repository root into process environment.

    - Does nothing if python-dotenv isn't installed.
    - Does not override already-set environment variables.
    """

    try:
        from dotenv import load_dotenv  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return

    root = Path(__file__).resolve().parents[1]
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=str(env_path), override=False)

