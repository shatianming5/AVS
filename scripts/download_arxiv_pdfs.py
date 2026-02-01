from __future__ import annotations

import argparse
import json
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx


ARXIV_API = "https://export.arxiv.org/api/query"


@dataclass(frozen=True)
class ArxivPaper:
    arxiv_id: str  # versionless when possible
    arxiv_id_version: str  # as provided by the feed (often includes vN)
    title: str
    published: str
    updated: str
    abs_url: str
    pdf_url: str


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso8601(s: str) -> datetime:
    raw = (s or "").strip()
    if not raw:
        raise ValueError("Missing datetime string.")
    # arXiv Atom uses e.g. 2024-01-01T00:00:00Z
    raw = raw.replace("Z", "+00:00")
    return datetime.fromisoformat(raw)


def _normalize_title(s: str) -> str:
    return " ".join((s or "").split())


def _extract_arxiv_id(abs_url: str) -> str:
    m = re.search(r"/abs/([^?#]+)", abs_url)
    if not m:
        raise ValueError(f"Could not parse arXiv abs URL: {abs_url!r}")
    return m.group(1)


def _strip_version(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id)


def _build_search_query(query: str) -> str:
    q = (query or "").strip()
    if not q:
        raise ValueError("Empty query.")
    # If user already supplied a lucene query (contains field:), keep it.
    if ":" in q:
        return q
    tokens = [t for t in re.split(r"\s+", q) if t]
    if not tokens:
        raise ValueError("Empty query tokens.")

    # Convenience: treat "CV" as the arXiv cs.CV category unless user already specified one.
    has_cat = any(t.lower().startswith("cat") for t in tokens)
    tokens_l = [t.lower() for t in tokens]
    if not has_cat and any(t == "cv" for t in tokens_l):
        kept = [t for t in tokens if t.lower() != "cv"]
        if not kept:
            kept = ["vision"]
        return "cat:cs.CV AND " + " AND ".join([f"all:{t}" for t in kept])

    return " AND ".join([f"all:{t}" for t in tokens])


def _fetch_feed(*, client: httpx.Client, search_query: str, max_results: int) -> str:
    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": int(max(1, max_results)),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    r = client.get(ARXIV_API, params=params)
    r.raise_for_status()
    return r.text


def _parse_feed(feed_xml: str) -> list[ArxivPaper]:
    root = ET.fromstring(feed_xml)
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    out: list[ArxivPaper] = []
    for entry in root.findall("atom:entry", ns):
        abs_url = (entry.findtext("atom:id", default="", namespaces=ns) or "").strip()
        if not abs_url:
            continue
        arxiv_id_version = _extract_arxiv_id(abs_url)
        arxiv_id = _strip_version(arxiv_id_version)

        title = _normalize_title(entry.findtext("atom:title", default="", namespaces=ns) or "")
        published = (entry.findtext("atom:published", default="", namespaces=ns) or "").strip()
        updated = (entry.findtext("atom:updated", default="", namespaces=ns) or "").strip()
        if not published:
            published = updated

        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        out.append(
            ArxivPaper(
                arxiv_id=arxiv_id,
                arxiv_id_version=arxiv_id_version,
                title=title,
                published=published,
                updated=updated,
                abs_url=abs_url,
                pdf_url=pdf_url,
            )
        )
    return out


def _download_pdf(*, client: httpx.Client, url: str) -> bytes:
    r = client.get(url, follow_redirects=True)
    r.raise_for_status()
    b = r.content
    if not b.startswith(b"%PDF"):
        raise RuntimeError(f"Downloaded file does not look like a PDF: url={url!r} head={b[:12]!r}")
    return b


def main() -> int:
    ap = argparse.ArgumentParser(description="Search arXiv and download N PDFs (for paper_skill demo)")
    ap.add_argument("--query", default="CV robust", help='Search keywords, e.g. "CV robust"')
    ap.add_argument("--years", type=int, default=2, help="Keep papers published within last N years")
    ap.add_argument("--max-results", type=int, default=30, help="Max entries to fetch from arXiv API")
    ap.add_argument("--num-pdfs", type=int, default=3, help="Number of PDFs to download")
    ap.add_argument("--out-dir", default="data/arxiv_pdfs/cv_robust_2y", help="Output directory")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing paper_*.pdf")
    args = ap.parse_args()

    query = str(args.query).strip()
    years = int(args.years)
    max_results = int(args.max_results)
    num_pdfs = int(args.num_pdfs)
    out_dir = Path(args.out_dir)
    overwrite = bool(args.overwrite)

    if years < 0:
        raise SystemExit("--years must be >= 0")
    if num_pdfs < 1:
        raise SystemExit("--num-pdfs must be >= 1")

    search_query = _build_search_query(query)
    cutoff = _now_utc() - timedelta(days=365 * max(0, years))

    headers = {
        "User-Agent": "paper_skill/0.1 (arxiv downloader; contact: local)",
        "Accept": "application/atom+xml, application/xml;q=0.9, text/xml;q=0.8, */*;q=0.1",
    }
    with httpx.Client(timeout=30.0, headers=headers) as client:
        feed_xml = _fetch_feed(client=client, search_query=search_query, max_results=max_results)
        papers = _parse_feed(feed_xml)

        kept: list[ArxivPaper] = []
        for p in papers:
            try:
                dt = _parse_iso8601(p.published)
            except Exception:  # noqa: BLE001
                continue
            if dt >= cutoff:
                kept.append(p)
            if len(kept) >= num_pdfs:
                break

        if len(kept) < num_pdfs:
            raise SystemExit(f"Not enough results within last {years} years. Found {len(kept)}.")

        out_dir.mkdir(parents=True, exist_ok=True)
        local_paths: list[str] = []
        for i, p in enumerate(kept, start=1):
            out_path = out_dir / f"paper_{i}.pdf"
            local_paths.append(str(out_path))
            if out_path.exists() and not overwrite:
                continue
            # Retry a couple times; arXiv can be flaky.
            last_err: Exception | None = None
            for attempt in range(3):
                try:
                    b = _download_pdf(client=client, url=p.pdf_url)
                    out_path.write_bytes(b)
                    break
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    time.sleep(0.6 * (attempt + 1))
            else:
                raise SystemExit(f"Failed to download {p.pdf_url}: {last_err}")

        manifest = {
            "query": query,
            "search_query": search_query,
            "years": years,
            "generated_at": _now_utc().isoformat(),
            "papers": [
                {
                    "arxiv_id": p.arxiv_id,
                    "arxiv_id_version": p.arxiv_id_version,
                    "title": p.title,
                    "published": p.published,
                    "updated": p.updated,
                    "abs_url": p.abs_url,
                    "pdf_url": p.pdf_url,
                    "local_path": str(out_dir / f"paper_{i+1}.pdf"),
                }
                for i, p in enumerate(kept)
            ],
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Downloaded arXiv PDFs:")
    for p in local_paths:
        print(f"- {p}")
    print("\nSuggested form values:")
    print("- Pack name: Arxiv CV Robust (2y)")
    print("- Field hint: cv")
    print("- Target venue hint: (leave empty)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
