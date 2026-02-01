from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _wait_job(client, *, job_id: str, owner_token: str, timeout_s: float) -> dict:
    from worker.runner import run_worker_once

    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        run_worker_once()
        jr = client.get(f"/api/jobs/{job_id}", headers={"X-Owner-Token": owner_token})
        if jr.status_code != 200:
            raise RuntimeError(jr.text)
        job = jr.json()
        last = job
        if job.get("status") == "failed":
            raise RuntimeError(job.get("error") or "Job failed")
        if job.get("status") == "succeeded":
            return job
        time.sleep(0.5)
    raise RuntimeError(f"Timeout waiting for job. last={last}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a SkillPack from 3 local PDF files (in-process)")
    ap.add_argument("--pdf", action="append", default=[], help="Path to a PDF. Pass exactly 3 times.")
    ap.add_argument("--pdf-dir", default="", help="Directory containing paper_1.pdf, paper_2.pdf, paper_3.pdf")
    ap.add_argument("--pack-name", default="Arxiv CV Robust (2y)")
    ap.add_argument("--field-hint", default="cv")
    ap.add_argument("--target-venue-hint", default="")
    ap.add_argument("--language", default="English")
    ap.add_argument("--data-dir", default="data", help="PAPER_SKILL_DATA_DIR override (default: ./data)")
    ap.add_argument("--timeout-s", type=float, default=900.0)
    ap.add_argument("--owner-token", default="", help="Optional fixed owner token")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    os.environ["PAPER_SKILL_DATA_DIR"] = str(data_dir)

    pdf_paths: list[Path] = []
    if args.pdf_dir:
        d = Path(args.pdf_dir).expanduser()
        pdf_paths = [d / "paper_1.pdf", d / "paper_2.pdf", d / "paper_3.pdf"]
    else:
        pdf_paths = [Path(p).expanduser() for p in (args.pdf or [])]

    if len(pdf_paths) != 3:
        print("Need exactly 3 PDFs. Use --pdf-dir or pass --pdf 3 times.", file=sys.stderr)
        return 2
    for p in pdf_paths:
        if not p.exists():
            print(f"Missing PDF: {p}", file=sys.stderr)
            return 2

    # Import after env var to ensure backend uses same data dir.
    from backend.app import app
    from fastapi.testclient import TestClient

    client = TestClient(app)
    owner_token = (args.owner_token or "").strip() or uuid.uuid4().hex

    pdf_ids: list[str] = []
    for p in pdf_paths:
        b = p.read_bytes()
        files = {"file": (p.name, b, "application/pdf")}
        r = client.post("/api/pdfs/upload", files=files, headers={"X-Owner-Token": owner_token})
        if r.status_code != 200:
            raise RuntimeError(r.text)
        pdf_ids.append(r.json()["pdf_id"])

    payload = {
        "pdf_ids": pdf_ids,
        "pack_name": str(args.pack_name),
        "field_hint": (str(args.field_hint).strip() or None),
        "target_venue_hint": (str(args.target_venue_hint).strip() or None),
        "language": str(args.language),
    }
    r = client.post("/api/skillpacks/build", json=payload, headers={"X-Owner-Token": owner_token})
    if r.status_code != 200:
        raise RuntimeError(r.text)
    job_id = r.json()["job_id"]

    job = _wait_job(client, job_id=job_id, owner_token=owner_token, timeout_s=float(args.timeout_s))
    pack_id = (job.get("result") or {}).get("pack_id")

    print(f"job_id={job_id}")
    print(f"pack_id={pack_id}")
    print(f"owner_token={owner_token}")
    print("\nOpen in browser (if your server is running):")
    print(f"- /pack/{pack_id}?token={owner_token}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

