#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path


CONCLUSION_RE = re.compile(r"^\s*-\s*\[(?P<check>[ xX])\]\s*(?P<cid>C\d{4})\s*:\s*(?P<title>.*\S)\s*$")
# Match `runs/...` paths inside Markdown code spans.
RUNS_PATH_RE = re.compile(r"`(runs/[^`\s]+)`")


@dataclass
class ConclusionBlock:
    cid: str
    checked: bool
    title: str
    text_lines: list[str]

    def evidence_paths(self) -> list[str]:
        paths: list[str] = []
        for ln in self.text_lines:
            for m in RUNS_PATH_RE.finditer(ln):
                paths.append(m.group(1))
        # de-dupe while preserving order
        seen: set[str] = set()
        out: list[str] = []
        for p in paths:
            if p in seen:
                continue
            seen.add(p)
            out.append(p)
        return out


def _parse_conclusions(plan_text: str) -> list[ConclusionBlock]:
    lines = plan_text.splitlines()
    blocks: list[ConclusionBlock] = []

    cur: ConclusionBlock | None = None
    for ln in lines:
        m = CONCLUSION_RE.match(ln)
        if m:
            if cur is not None:
                blocks.append(cur)
            cur = ConclusionBlock(
                cid=str(m.group("cid")),
                checked=str(m.group("check")).strip().lower() == "x",
                title=str(m.group("title")).strip(),
                text_lines=[ln],
            )
            continue
        if cur is not None:
            # Keep all following lines until the next conclusion bullet.
            cur.text_lines.append(ln)

    if cur is not None:
        blocks.append(cur)
    return blocks


def _write_markdown(*, out_md: Path, plan_rel: str, rows: list[dict]) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)

    def _fmt_paths(paths: list[str], missing: bool) -> str:
        if not paths:
            return ""
        max_show = 8
        shown = paths[:max_show]
        suffix = "" if len(paths) <= max_show else f"\n- ... (+{len(paths) - max_show} more)"
        tag = "missing" if missing else "present"
        return "\n".join([f"- `{p}` ({tag})" for p in shown]) + suffix

    lines: list[str] = []
    lines.append("# Evidence Matrix")
    lines.append("")
    lines.append(f"- Generated: `{time.strftime('%Y-%m-%d %H:%M:%S')}`")
    lines.append(f"- Plan: `{plan_rel}`")
    lines.append("")
    lines.append("| Conclusion | Checked in plan | Local artifacts present? | Notes |")
    lines.append("| --- | --- | --- | --- |")
    for r in rows:
        cid = r["cid"]
        checked = "yes" if r["checked"] else "no"
        present = "yes" if (r["missing_count"] == 0 and r["evidence_count"] > 0) else "no"
        note = ""
        if r["evidence_count"] == 0:
            note = "no `runs/...` paths found in conclusion block"
        elif r["missing_count"] > 0:
            note = f"missing {r['missing_count']}/{r['evidence_count']} artifacts"
        lines.append(f"| `{cid}` | {checked} | {present} | {note} |")
    lines.append("")

    for r in rows:
        lines.append(f"## {r['cid']}: {r['title']}")
        lines.append("")
        if r["evidence_count"] == 0:
            lines.append("- No `runs/...` evidence paths found under this conclusion.")
            lines.append("")
            continue
        if r["missing_count"] == 0:
            lines.append("- All referenced artifacts exist locally.")
            lines.append("")
        else:
            lines.append("### Missing")
            lines.append(_fmt_paths(r["missing_paths"], missing=True))
            lines.append("")
            lines.append("### Present")
            lines.append(_fmt_paths(r["present_paths"], missing=False))
            lines.append("")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build a local evidence matrix by scanning docs/plan.md conclusions for runs/* artifacts.")
    p.add_argument("--plan", type=Path, default=Path("docs/plan.md"))
    p.add_argument("--out-dir", type=Path, default=Path("runs") / f"evidence_matrix_{time.strftime('%Y%m%d-%H%M%S')}")
    p.add_argument("--write-docs-md", action="store_true", help="Also write docs/evidence_matrix.md (tracked).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan_path = Path(args.plan)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plan_text = plan_path.read_text(encoding="utf-8")
    blocks = _parse_conclusions(plan_text)

    rows: list[dict] = []
    for b in blocks:
        ev = b.evidence_paths()
        present: list[str] = []
        missing: list[str] = []
        for p in ev:
            if Path(p).exists():
                present.append(p)
            else:
                missing.append(p)
        rows.append(
            {
                "cid": b.cid,
                "checked": bool(b.checked),
                "title": b.title,
                "evidence_count": int(len(ev)),
                "present_count": int(len(present)),
                "missing_count": int(len(missing)),
                "present_paths": present,
                "missing_paths": missing,
            }
        )

    payload = {
        "ok": True,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "plan": str(plan_path),
        "conclusions": rows,
        "summary": {
            "total_conclusions": int(len(rows)),
            "checked_in_plan": int(sum(1 for r in rows if r["checked"])),
            "with_any_evidence_paths": int(sum(1 for r in rows if r["evidence_count"] > 0)),
            "with_missing_artifacts": int(sum(1 for r in rows if r["missing_count"] > 0)),
        },
    }

    out_json = out_dir / "evidence_matrix.json"
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out_json)

    out_md = out_dir / "evidence_matrix.md"
    _write_markdown(out_md=out_md, plan_rel=str(plan_path), rows=rows)
    print(out_md)

    if bool(args.write_docs_md):
        docs_md = Path("docs") / "evidence_matrix.md"
        _write_markdown(out_md=docs_md, plan_rel=str(plan_path), rows=rows)
        print(docs_md)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
