from __future__ import annotations

import argparse
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import httpx


REPO_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = REPO_ROOT / "data" / "playwright_uploads"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_http_ok(url: str, *, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code == 200:
                return
            last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:  # noqa: BLE001
            last_err = e
        time.sleep(0.2)
    raise RuntimeError(f"Timed out waiting for {url}: {last_err}")


@dataclass(frozen=True)
class Proc:
    name: str
    popen: subprocess.Popen
    log_path: Path


def _start_proc(*, name: str, cmd: list[str], env: dict[str, str]) -> Proc:
    log_path = Path(tempfile.gettempdir()) / f"paper_skill_{name}_{int(time.time())}.log"
    f = log_path.open("wb")
    p = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=f,
        stderr=subprocess.STDOUT,
    )
    return Proc(name=name, popen=p, log_path=log_path)


def _stop_proc(proc: Proc) -> None:
    p = proc.popen
    if p.poll() is not None:
        return
    try:
        p.send_signal(signal.SIGTERM)
    except Exception:  # noqa: BLE001
        return
    try:
        p.wait(timeout=5)
    except Exception:  # noqa: BLE001
        try:
            p.kill()
        except Exception:  # noqa: BLE001
            pass


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def run_e2e(*, base_url: str, owner_token_hint: str | None = None) -> None:
    from playwright.sync_api import sync_playwright  # type: ignore[import-not-found]

    pdf1 = str((UPLOAD_DIR / "paper_1.pdf").resolve())
    pdf2 = str((UPLOAD_DIR / "paper_2.pdf").resolve())
    pdf3 = str((UPLOAD_DIR / "paper_3.pdf").resolve())
    _assert(Path(pdf1).exists() and Path(pdf2).exists() and Path(pdf3).exists(), "Missing playwright upload fixtures.")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        # A) Home page loads
        page.goto(f"{base_url}/", wait_until="domcontentloaded")
        _assert(page.title() == "paper_skill", f"Unexpected title: {page.title()!r}")
        _assert(page.locator("#pdfs").count() == 1, "Missing #pdfs input.")
        _assert(page.locator("#pack_name").count() == 1, "Missing #pack_name input.")
        _assert(page.locator("#status").count() == 1, "Missing #status.")

        # B) Validation: 2 PDFs
        page.fill("#pack_name", "E2E Pack (validation)")
        page.set_input_files("#pdfs", [pdf1, pdf2])
        page.click("button[type=submit]")
        page.wait_for_timeout(200)
        status = page.locator("#status").inner_text()
        _assert(status == "Please select exactly 3 PDF files.", f"Expected 2-file error; got: {status!r}")

        # C/D) Build: 3 PDFs => /job => /pack
        page.fill("#pack_name", "E2E Pack")
        page.set_input_files("#pdfs", [pdf1, pdf2, pdf3])
        page.click("button[type=submit]")
        page.wait_for_url(re.compile(r".*/job/[^/?]+.*"), timeout=60_000)
        job_url = page.url
        m = re.search(r"/job/([^/?#]+)", job_url)
        _assert(bool(m), f"Could not extract job_id from URL: {job_url}")
        job_id = m.group(1)

        owner_token = page.evaluate("() => localStorage.getItem('paper_skill_owner_token')")
        _assert(isinstance(owner_token, str) and owner_token.strip(), "Missing owner token in localStorage.")
        if owner_token_hint:
            _assert(owner_token == owner_token_hint, "owner token changed unexpectedly.")

        page.wait_for_url(re.compile(r".*/pack/[^/?]+.*"), timeout=180_000)
        pack_url = page.url
        m = re.search(r"/pack/([^/?#]+)", pack_url)
        _assert(bool(m), f"Could not extract pack_id from URL: {pack_url}")
        pack_id = m.group(1)

        # E) Pack page sections
        _assert(page.locator("h2", has_text="Intro Blueprint").count() == 1, "Missing Intro Blueprint section.")
        _assert(page.locator("h2", has_text="Templates").count() == 1, "Missing Templates section.")
        _assert(page.locator("h2", has_text="Figure storyboard").count() == 1, "Missing Figure storyboard section.")
        _assert(page.locator("h2", has_text="Quality report").count() == 1, "Missing Quality report section.")

        # H) Downloads (via direct HTTP)
        download_json = page.evaluate("() => document.getElementById('download_json').href")
        download_yaml = page.evaluate("() => document.getElementById('download_yaml').href")
        _assert(isinstance(download_json, str) and download_json, "Missing download_json href.")
        _assert(isinstance(download_yaml, str) and download_yaml, "Missing download_yaml href.")
        rj = httpx.get(download_json, timeout=10.0)
        _assert(rj.status_code == 200 and '"pack_id"' in rj.text, "Download JSON failed or missing pack_id.")
        ry = httpx.get(download_yaml, timeout=10.0)
        _assert(ry.status_code == 200 and "pack_id:" in ry.text, "Download YAML failed or missing pack_id.")

        # F/G) Evidence -> Viewer
        first_ev = page.locator("a.ev-chip").first
        _assert(first_ev.count() == 1, "No evidence chip found.")
        first_ev.click()
        page.wait_for_url(re.compile(r".*/viewer/[^/?#]+.*"), timeout=30_000)
        page.wait_for_function(
            "() => { const el=document.getElementById('meta'); return !!(el && el.textContent && el.textContent.includes('page=')); }",
            timeout=30_000,
        )
        page.wait_for_selector(".hl", timeout=30_000)
        page.locator(".hl").first.hover(force=True)
        page.wait_for_selector("#tooltip:not(.hidden)", timeout=10_000)
        tt = page.locator("#tooltip").inner_text()
        _assert("evidence_id" in tt and "Used by" in tt, f"Viewer tooltip missing expected fields: {tt!r}")
        meta = page.locator("#meta").inner_text()
        _assert("page=" in meta and "/" in meta, f"Viewer meta missing page count: {meta!r}")
        _assert(page.locator("#prev").is_disabled(), "Viewer prev should be disabled on page 1.")

        # Create one user annotation (owner only)
        _assert(page.locator("#annotate_toggle").count() == 1, "Missing annotate toggle.")
        page.click("#annotate_toggle")
        overlay_box = page.locator("#overlay").bounding_box()
        _assert(isinstance(overlay_box, dict) and overlay_box.get("width", 0) > 20, "Bad overlay bounding box.")
        x0 = float(overlay_box["x"]) + float(overlay_box["width"]) * 0.62
        y0 = float(overlay_box["y"]) + float(overlay_box["height"]) * 0.62
        x1 = float(overlay_box["x"]) + float(overlay_box["width"]) * 0.80
        y1 = float(overlay_box["y"]) + float(overlay_box["height"]) * 0.78
        page.mouse.move(x0, y0)
        page.mouse.down()
        page.mouse.move(x1, y1)
        page.mouse.up()
        page.wait_for_selector("#ann_dialog[open]", timeout=10_000)
        page.fill("#ann_note", "E2E annotation")
        page.click("#ann_save")
        page.wait_for_selector(".ann", timeout=10_000)
        page.locator(".ann").first.hover(force=True)
        page.wait_for_selector("#tooltip:not(.hidden)", timeout=10_000)
        ann_tt = page.locator("#tooltip").inner_text()
        _assert("annotation_id" in ann_tt, f"Annotation tooltip missing annotation_id: {ann_tt!r}")

        # Jump to last page if multi-page
        m = re.search(r"page=(\\d+)(?:/(\\d+))?", meta)
        if m and m.group(2):
            total_pages = int(m.group(2))
            if total_pages >= 2:
                url = page.url
                if "page=" in url:
                    url = re.sub(r"([?&])page=\\d+", rf"\\1page={total_pages}", url)
                else:
                    url = url + ("&" if "?" in url else "?") + f"page={total_pages}"
                page.goto(url, wait_until="domcontentloaded")
                _assert(page.locator("#next").is_disabled(), "Viewer next should be disabled on last page.")

        # I) Share / Unshare
        page.goto(pack_url, wait_until="domcontentloaded")
        page.click("#share_btn")
        page.wait_for_selector("#share_box a", timeout=10_000)
        share_href = page.locator("#share_box a").get_attribute("href") or ""
        _assert(share_href.startswith("/pack/"), f"Bad share href: {share_href!r}")
        share_url = f"{base_url}{share_href}"

        share_context = browser.new_context()
        share_page = share_context.new_page()
        share_page.goto(share_url, wait_until="domcontentloaded")
        share_page.wait_for_function("() => (document.getElementById('title')||{}).textContent?.length > 0", timeout=30_000)
        _assert(share_page.locator("h2", has_text="Intro Blueprint").count() == 1, "Shared pack missing Intro Blueprint.")

        # Unshare as owner and ensure old share token becomes invalid
        page.goto(pack_url, wait_until="domcontentloaded")
        page.click("#unshare_btn")
        page.wait_for_function("() => (document.getElementById('share_box')||{}).textContent === 'Unshared.'", timeout=10_000)

        share_token = None
        m = re.search(r"[?&]token=([^&#]+)", share_href)
        if m:
            share_token = m.group(1)
        _assert(bool(share_token), f"Could not extract share token from href: {share_href!r}")
        r_forbidden = httpx.get(f"{base_url}/api/skillpacks/{pack_id}?token={share_token}", timeout=10.0)
        _assert(r_forbidden.status_code == 403, f"Expected 403 after unshare; got {r_forbidden.status_code}.")

        # J) Jobs page includes llm metrics
        jobs_url = f"{base_url}/jobs?token={owner_token}"
        page.goto(jobs_url, wait_until="domcontentloaded")
        page.wait_for_selector(".rule", timeout=10_000)
        jobs_text = page.locator("#content").inner_text()
        _assert("llm_health" in jobs_text and '"llm"' in jobs_text, "Jobs metrics missing llm/llm_health.")

        # K) Events page has entries
        events_url = f"{base_url}/job/{job_id}/events?token={owner_token}"
        page.goto(events_url, wait_until="domcontentloaded")
        page.wait_for_selector(".rule", timeout=10_000)
        _assert(page.locator(".rule").count() >= 1, "No events found.")

        # Access control: missing token behavior
        anon = browser.new_context()
        anon_page = anon.new_page()
        anon_page.goto(f"{base_url}/jobs", wait_until="domcontentloaded")
        anon_page.wait_for_timeout(200)
        _assert("Missing token" in anon_page.locator("#content").inner_text(), "Expected missing-token message on /jobs.")

        # Cache hits: second build with same PDFs (dedup -> same pdf_ids -> pack_cache_key hit)
        page.goto(f"{base_url}/", wait_until="domcontentloaded")
        page.fill("#pack_name", "E2E Pack (2nd run)")
        page.set_input_files("#pdfs", [pdf1, pdf2, pdf3])
        page.click("button[type=submit]")
        page.wait_for_url(re.compile(r".*/job/[^/?]+.*"), timeout=60_000)
        page.wait_for_url(re.compile(r".*/pack/[^/?]+.*"), timeout=180_000)

        jobs_api = httpx.get(f"{base_url}/api/jobs?limit=5&token={owner_token}", timeout=10.0)
        _assert(jobs_api.status_code == 200, f"/api/jobs failed: {jobs_api.status_code}")
        jobs = jobs_api.json().get("jobs") or []
        _assert(len(jobs) >= 2, "Expected at least 2 jobs after second build.")
        latest_metrics = (jobs[0].get("metrics") or {}) if isinstance(jobs[0], dict) else {}
        cache_hits = latest_metrics.get("cache_hits") if isinstance(latest_metrics, dict) else None
        _assert(isinstance(cache_hits, dict), "Latest job missing cache_hits.")
        pack_level = cache_hits.get("blueprint") if isinstance(cache_hits, dict) else None
        _assert(
            isinstance(pack_level, dict) and any(bool(v) for v in pack_level.values()),
            "Expected blueprint cache hit on 2nd run.",
        )

        try:
            context.close()
            share_context.close()
        except Exception:  # noqa: BLE001
            pass
        browser.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="paper_skill E2E via Playwright (headless)")
    ap.add_argument("--base-url", default=None, help="If provided, do not start services; run against this base URL.")
    ap.add_argument("--timeout", type=float, default=15.0, help="Startup timeout seconds when starting services.")
    args = ap.parse_args()

    procs: list[Proc] = []
    data_dir = Path(tempfile.mkdtemp(prefix="paper_skill_e2e_data_"))
    llm_port = _free_port()
    api_port = _free_port()

    try:
        base_url = args.base_url
        if not base_url:
            env = os.environ.copy()
            env.update(
                {
                    "PAPER_SKILL_DATA_DIR": str(data_dir),
                    "PAPER_SKILL_LLM_BASE_URL": f"http://127.0.0.1:{llm_port}/v1",
                    "PAPER_SKILL_LLM_MODEL": "fake-model",
                    "PAPER_SKILL_REQUIRE_LLM": "1",
                }
            )

            procs.append(
                _start_proc(
                    name="fake_llm",
                    cmd=[sys.executable, str(REPO_ROOT / "scripts" / "fake_llm_server.py")],
                    env={**env, "FAKE_LLM_PORT": str(llm_port)},
                )
            )
            procs.append(
                _start_proc(
                    name="worker",
                    cmd=[sys.executable, "-m", "worker.runner"],
                    env=env,
                )
            )
            procs.append(
                _start_proc(
                    name="api",
                    cmd=[
                        sys.executable,
                        "-m",
                        "uvicorn",
                        "backend.app:app",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(api_port),
                        "--log-level",
                        "warning",
                    ],
                    env=env,
                )
            )
            base_url = f"http://127.0.0.1:{api_port}"
            _wait_http_ok(f"{base_url}/", timeout_s=args.timeout)

        run_e2e(base_url=base_url, owner_token_hint=None)
        print("E2E PASS")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"E2E FAIL: {e}", file=sys.stderr)
        for p in procs:
            if p.log_path.exists():
                tail = p.log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-200:]
                print(f"\n--- {p.name} log tail: {p.log_path} ---", file=sys.stderr)
                print("\n".join(tail), file=sys.stderr)
        return 1
    finally:
        for p in reversed(procs):
            _stop_proc(p)


if __name__ == "__main__":
    raise SystemExit(main())
