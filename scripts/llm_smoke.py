from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _make_synth_pdf_bytes(title: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4-ish points
    page.insert_textbox(
        fitz.Rect(50, 40, 545, 110),
        f"{title}\n\nAbstract\nWe study a task that has attracted significant attention in recent years.",
        fontsize=12,
    )
    page.insert_textbox(fitz.Rect(50, 130, 545, 160), "1 Introduction", fontsize=14)
    intro = (
        "Recent progress in representation learning has enabled strong performance across tasks. "
        "However, existing approaches still lack robustness under distribution shift. "
        "In this paper, we propose MethodX, which may improve stability by combining calibration and data augmentation. "
        "Our contributions are threefold: (1) a simple method; (2) a theoretical insight; (3) extensive experiments. "
        "The rest of this paper is organized as follows: Section 2 describes the method; Section 3 reports results."
    )
    page.insert_textbox(fitz.Rect(50, 170, 545, 360), intro, fontsize=11)
    page.insert_textbox(fitz.Rect(50, 380, 545, 420), "Figure 1: Overview of the proposed pipeline.", fontsize=10)
    out = doc.tobytes()
    doc.close()
    return out


def _wait_job(client, *, job_id: str, owner_token: str, timeout_s: float) -> dict:
    from worker.runner import run_worker_once

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        run_worker_once()
        jr = client.get(f"/api/jobs/{job_id}", headers={"X-Pack-Token": owner_token})
        if jr.status_code != 200:
            raise RuntimeError(jr.text)
        job = jr.json()
        if job["status"] == "failed":
            raise RuntimeError(job.get("error") or "Job failed")
        if job["status"] == "succeeded":
            return job
        time.sleep(0.2)
    raise RuntimeError("Timeout waiting for job")


def main() -> int:
    parser = argparse.ArgumentParser(description="paper_skill local-LLM smoke run (in-process)")
    parser.add_argument("--data-dir", type=str, default="", help="Override PAPER_SKILL_DATA_DIR (default: temp dir)")
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--field-hint", type=str, default="NLP")
    parser.add_argument("--target-venue-hint", type=str, default="NeurIPS")
    parser.add_argument("--use-upload-pdfs", action="store_true", help="Use data/playwright_uploads/paper_*.pdf instead of synthetic PDFs")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser() if args.data_dir else Path(tempfile.mkdtemp(prefix="paper_skill_llm_smoke_")) / "data"
    os.environ["PAPER_SKILL_DATA_DIR"] = str(data_dir)

    # Import after env var.
    from backend.app import app
    from fastapi.testclient import TestClient

    from shared.llm_client import load_llm_config, redact_base_url

    cfg = load_llm_config()
    if cfg is None:
        print("Missing PAPER_SKILL_LLM_MODEL (and/or .env not loaded).", file=sys.stderr)
        return 2

    print(f"LLM base_url={redact_base_url(cfg.base_url)} model={cfg.model}")
    print(f"data_dir={data_dir}")

    client = TestClient(app)
    owner_token = "llm_smoke_owner"

    pdf_ids: list[str] = []
    if args.use_upload_pdfs:
        root = Path(__file__).resolve().parents[1]
        pdf_paths = [
            root / "data" / "playwright_uploads" / "paper_1.pdf",
            root / "data" / "playwright_uploads" / "paper_2.pdf",
            root / "data" / "playwright_uploads" / "paper_3.pdf",
        ]
        for p in pdf_paths:
            b = p.read_bytes()
            files = {"file": (p.name, b, "application/pdf")}
            r = client.post("/api/pdfs/upload", files=files, headers={"X-Owner-Token": owner_token})
            if r.status_code != 200:
                raise RuntimeError(r.text)
            pdf_ids.append(r.json()["pdf_id"])
    else:
        for i in range(3):
            b = _make_synth_pdf_bytes(f"Synthetic LLM Smoke {i+1}")
            files = {"file": (f"paper_{i+1}.pdf", b, "application/pdf")}
            r = client.post("/api/pdfs/upload", files=files, headers={"X-Owner-Token": owner_token})
            if r.status_code != 200:
                raise RuntimeError(r.text)
            pdf_ids.append(r.json()["pdf_id"])

    payload = {
        "pdf_ids": pdf_ids,
        "pack_name": f"LLM Smoke {int(time.time())}",
        "field_hint": args.field_hint,
        "target_venue_hint": args.target_venue_hint,
        "language": "English",
    }
    r = client.post("/api/skillpacks/build", json=payload, headers={"X-Owner-Token": owner_token})
    if r.status_code != 200:
        raise RuntimeError(r.text)
    job_id = r.json()["job_id"]

    job = _wait_job(client, job_id=job_id, owner_token=owner_token, timeout_s=float(args.timeout_s))
    pack_id = job["result"]["pack_id"]

    pack = client.get(f"/api/skillpacks/{pack_id}", headers={"X-Pack-Token": owner_token})
    if pack.status_code != 200:
        raise RuntimeError(pack.text)
    pack_json = pack.json()

    metrics = job.get("metrics") or {}
    print(f"job_id={job_id}")
    print(f"pack_id={pack_id}")
    print(f"evidence_coverage={pack_json.get('quality_report', {}).get('evidence_coverage')}")
    print(f"llm_health={metrics.get('llm_health')}")
    print(f"cache_hits={metrics.get('cache_hits')}")

    # Basic thresholds (conservative): fail fast if something is clearly broken.
    ev_cov = float(pack_json.get("quality_report", {}).get("evidence_coverage") or 0.0)
    llm_health = metrics.get("llm_health") or {}
    mv_cov = float(llm_health.get("moves_return_coverage") or 0.0)
    bp_cov = float(llm_health.get("blueprint_rules_with_evidence_ratio") or 0.0)
    sb_cov = float(llm_health.get("storyboard_items_with_evidence_ratio") or 0.0)

    if ev_cov < 0.2 or mv_cov < 0.6 or (bp_cov < 0.2 and sb_cov < 0.2):
        print("Smoke thresholds not met.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
