#!/usr/bin/env python3
"""
Bucketed "when does X help?" report for long-video QA eval runs.

Inputs are the streamed `predictions.jsonl` artifacts produced by:
- avs.experiments.intentqa_vlm_eval
- avs.experiments.avqa_vlm_eval
- avs.experiments.egoschema_vlm_eval (subset; labels available)

This script is intentionally lightweight: it does not re-run any model, it only
aggregates correctness/invalid flags and joins optional metadata for semantic
buckets.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:  # noqa: BLE001
                raise ValueError(f"failed to parse jsonl at {path}:{ln}: {e}") from e


def _detect_task_and_id_keys(first_row: dict[str, Any]) -> tuple[str, list[str]]:
    if "qid" in first_row:
        # IntentQA has many duplicated qid values across videos; use a composite key.
        if "video_id" not in first_row:
            raise ValueError("IntentQA predictions must include video_id to form a unique key")
        return "IntentQA", ["video_id", "qid"]
    if "item_id" in first_row:
        return "AVQA", ["item_id"]
    if "question_idx" in first_row:
        # In practice question_idx should be unique, but include video_idx for safety.
        if "video_idx" in first_row:
            return "EgoSchema", ["video_idx", "question_idx"]
        return "EgoSchema", ["question_idx"]
    keys = ", ".join(sorted(first_row.keys()))
    raise ValueError(f"cannot detect task from predictions row keys: {keys}")


def _extract_q_bar(row: dict[str, Any]) -> float | None:
    if "q_bar" in row and row["q_bar"] is not None:
        try:
            return float(row["q_bar"])
        except Exception:  # noqa: BLE001
            return None
    sel = row.get("selected")
    if isinstance(sel, dict) and ("q_bar" in sel) and sel["q_bar"] is not None:
        try:
            return float(sel["q_bar"])
        except Exception:  # noqa: BLE001
            return None
    return None


def _safe_mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


@dataclass(frozen=True)
class MethodStats:
    n: int
    acc: float
    invalid_rate: float


def _compute_method_stats(rows: list[dict[str, Any]]) -> MethodStats:
    ys: list[float] = []
    invs: list[float] = []
    for r in rows:
        if r.get("correct") is None:
            # Unlabeled split; skip.
            continue
        ys.append(1.0 if bool(r.get("correct")) else 0.0)
        invs.append(1.0 if bool(r.get("invalid")) else 0.0)
    return MethodStats(n=len(ys), acc=_safe_mean(ys), invalid_rate=_safe_mean(invs))


def _join_key(rec: dict[str, Any], keys: list[str]) -> str:
    return "|".join(str(rec.get(k)) for k in keys)


def _load_meta_map(meta_path: Path, meta_id_keys: list[str]) -> dict[str, dict[str, Any]]:
    if meta_path.suffix.lower() == ".csv":
        out: dict[str, dict[str, Any]] = {}
        with meta_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for k in meta_id_keys:
                if k not in (reader.fieldnames or []):
                    raise ValueError(f"meta_id_key={k} not found in csv columns: {reader.fieldnames}")
            for row in reader:
                out[_join_key(row, meta_id_keys)] = row
        return out

    if meta_path.suffix.lower() == ".json":
        obj = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(obj, list):
            raise ValueError(f"expected meta json to be a list, got {type(obj)}: {meta_path}")
        out = {}
        for rec in obj:
            if not isinstance(rec, dict):
                continue
            if any(k not in rec for k in meta_id_keys):
                continue
            out[_join_key(rec, meta_id_keys)] = rec
        return out

    raise ValueError(f"unsupported meta file type: {meta_path} (expected .csv or .json)")


def _bucket_by_meta(
    item_ids: list[str],
    meta_map: dict[str, dict[str, Any]],
    meta_bucket_field: str,
) -> dict[str, str]:
    out: dict[str, str] = {}
    for it in item_ids:
        rec = meta_map.get(str(it))
        if rec is None:
            out[str(it)] = "UNKNOWN"
            continue
        v = rec.get(meta_bucket_field)
        out[str(it)] = "UNKNOWN" if v is None else str(v)
    return out


def _bucket_by_qbar_quantiles(
    item_to_rows: dict[str, dict[str, dict[str, Any]]],
    qbar_method: str,
    quantiles: list[float],
) -> tuple[dict[str, str], dict[str, Any]]:
    qs: list[float] = []
    q_by_item: dict[str, float] = {}
    for it, m2r in item_to_rows.items():
        row = m2r.get(qbar_method)
        if not row:
            continue
        q = _extract_q_bar(row)
        if q is None:
            continue
        q_by_item[str(it)] = float(q)
        qs.append(float(q))

    if not qs:
        return {}, {"ok": False, "reason": f"no q_bar values found for method={qbar_method}"}

    qs_arr = np.asarray(qs, dtype=np.float64)
    qs_arr.sort()
    qts = [float(x) for x in np.quantile(qs_arr, np.asarray(quantiles, dtype=np.float64), method="linear")]
    # Ensure non-decreasing edges.
    qts = [max(qts[i], qts[i - 1]) if i > 0 else qts[i] for i in range(len(qts))]

    def label(q: float) -> str:
        # 3 bins for 2 quantiles; if more quantiles provided, generalize.
        edges = qts
        if len(edges) == 0:
            return "all"
        if q <= edges[0]:
            return "low"
        for i in range(1, len(edges)):
            if q <= edges[i]:
                return f"mid{i}"
        return "high"

    out: dict[str, str] = {}
    for it, q in q_by_item.items():
        out[str(it)] = label(float(q))

    meta = {
        "ok": True,
        "method": qbar_method,
        "quantiles": quantiles,
        "edges": qts,
        "n_items_with_qbar": int(len(q_by_item)),
    }
    return out, meta


def _compute_bucket_table(
    item_to_rows: dict[str, dict[str, dict[str, Any]]],
    buckets_by_item: dict[str, str],
    methods: list[str],
    uniform_method: str,
    compare_methods: list[str],
) -> list[dict[str, Any]]:
    bucket_to_items: dict[str, list[str]] = {}
    for it, b in buckets_by_item.items():
        bucket_to_items.setdefault(str(b), []).append(str(it))

    out_rows: list[dict[str, Any]] = []
    for b, items in sorted(bucket_to_items.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        by_method: dict[str, MethodStats] = {}
        for m in methods:
            paired: list[dict[str, Any]] = []
            for it in items:
                rows = item_to_rows.get(it, {})
                if uniform_method not in rows or m not in rows:
                    continue
                paired.append(rows[m])
            st = _compute_method_stats(paired)
            if st.n == 0:
                continue
            by_method[m] = st

        if uniform_method not in by_method:
            continue

        deltas: dict[str, float] = {}
        u_acc = float(by_method[uniform_method].acc)
        for m in compare_methods:
            if m == uniform_method:
                continue
            if m not in by_method:
                continue
            deltas[m] = float(by_method[m].acc - u_acc)

        out_rows.append(
            {
                "bucket": str(b),
                "n_items": int(len(items)),
                "by_method": {m: by_method[m].__dict__ for m in sorted(by_method.keys())},
                "delta_vs_uniform": deltas,
            }
        )

    return out_rows


def _render_bucket_md(
    *,
    title: str,
    primary_method: str,
    table: list[dict[str, Any]],
    uniform_method: str,
    top_k: int,
) -> str:
    # Keep markdown small and readable: show top helpful/harmful buckets for primary_method.
    rows = []
    for r in table:
        d = r.get("delta_vs_uniform", {})
        if primary_method not in d:
            continue
        rows.append((float(d[primary_method]), r))
    rows.sort(key=lambda x: x[0], reverse=True)

    helpful = rows[:top_k]
    harmful = list(reversed(rows[-top_k:])) if len(rows) >= top_k else list(reversed(rows))

    def fmt_row(delta: float, r: dict[str, Any]) -> str:
        bm = r.get("by_method", {})
        u = bm.get(uniform_method, {})
        p = bm.get(primary_method, {})
        return (
            f"| {r.get('bucket')} | {r.get('n_items')} | "
            f"{u.get('acc', 0.0):.4f} | {p.get('acc', 0.0):.4f} | {delta:+.4f} |"
        )

    def section(name: str, pairs: list[tuple[float, dict[str, Any]]]) -> str:
        if not pairs:
            return f"### {name}\n\n(none)\n"
        lines = [
            f"### {name}",
            "",
            f"Primary method: `{primary_method}` vs `{uniform_method}`.",
            "",
            "| bucket | n_items | uniform_acc | primary_acc | delta |",
            "|---|---:|---:|---:|---:|",
        ]
        for delta, r in pairs:
            lines.append(fmt_row(delta, r))
        lines.append("")
        return "\n".join(lines)

    parts = [f"## {title}", ""]
    parts.append(section("Top Helpful Buckets", helpful))
    parts.append(section("Top Harmful Buckets", harmful))
    return "\n".join(parts).rstrip() + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description="Bucketed analysis for long-video QA eval outputs (predictions.jsonl).")
    p.add_argument("--predictions", type=Path, required=True, help="Path to runs/.../predictions.jsonl")
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory (will be created)")
    p.add_argument("--uniform-method", type=str, default="uniform")
    p.add_argument(
        "--compare-methods",
        type=str,
        default="",
        help="Comma-separated methods to compute delta_vs_uniform for (default: all methods except uniform).",
    )
    p.add_argument("--primary-method", type=str, default="audio", help="Method used to rank helpful/harmful buckets.")

    # Optional meta bucket.
    p.add_argument("--meta", type=Path, default=None, help="Optional meta file (.csv or .json list of dicts).")
    p.add_argument("--meta-id-key", type=str, default="", help="Meta id key (e.g., qid or id). If empty, auto.")
    p.add_argument("--meta-bucket-field", type=str, default="", help="Meta field name to bucket by (e.g., type).")

    # Optional q_bar bucket.
    p.add_argument("--qbar-method", type=str, default="", help="Method to read q_bar from (e.g., audio).")
    p.add_argument(
        "--qbar-quantiles",
        type=str,
        default="0.33,0.66",
        help="Comma-separated quantiles (0-1) to form q_bar buckets (default: 0.33,0.66).",
    )
    p.add_argument("--no-qbar", action="store_true", help="Disable q_bar bucketing even if --qbar-method is set.")

    args = p.parse_args()
    pred_path = Path(args.predictions)
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    it = iter(_read_jsonl(pred_path))
    try:
        first = next(it)
    except StopIteration:
        raise ValueError(f"empty predictions file: {pred_path}")

    task, id_keys = _detect_task_and_id_keys(first)

    item_to_rows: dict[str, dict[str, dict[str, Any]]] = {}
    methods_set: set[str] = set()

    def ingest(row: dict[str, Any]) -> None:
        if "method" not in row:
            return
        item_id = "|".join(str(row.get(k)) for k in id_keys)
        m = str(row.get("method"))
        methods_set.add(m)
        item_to_rows.setdefault(item_id, {})[m] = row

    ingest(first)
    for row in it:
        ingest(row)

    methods = sorted(methods_set)
    uniform_method = str(args.uniform_method)
    if uniform_method not in methods:
        raise ValueError(f"uniform_method={uniform_method} not present in methods={methods}")

    compare_methods = [m for m in methods if m != uniform_method]
    if str(args.compare_methods).strip():
        compare_methods = [m.strip() for m in str(args.compare_methods).split(",") if m.strip()]

    primary_method = str(args.primary_method)
    if primary_method not in methods:
        # Fall back to first compare method for the report (still keep the user-provided value in payload).
        primary_method_effective = compare_methods[0] if compare_methods else uniform_method
    else:
        primary_method_effective = primary_method

    # Overall stats.
    overall: dict[str, Any] = {}
    for m in methods:
        rows = []
        for it_id, m2r in item_to_rows.items():
            if m not in m2r:
                continue
            rows.append(m2r[m])
        st = _compute_method_stats(rows)
        overall[m] = st.__dict__

    # Bucket reports.
    bucket_reports: list[dict[str, Any]] = []
    item_ids = sorted(item_to_rows.keys())

    # Meta bucket.
    if args.meta_bucket_field:
        if args.meta is None:
            raise ValueError("--meta-bucket-field requires --meta")
        meta_path = Path(args.meta)
        if not meta_path.exists():
            raise FileNotFoundError(meta_path)

        meta_id_key_str = str(args.meta_id_key).strip()
        if not meta_id_key_str:
            meta_id_key_str = "video_id,qid" if task == "IntentQA" else ("id" if task == "AVQA" else ",".join(id_keys))
        meta_id_keys = [k.strip() for k in meta_id_key_str.split(",") if k.strip()]
        meta_map = _load_meta_map(meta_path, meta_id_keys=meta_id_keys)
        buckets_by_item = _bucket_by_meta(item_ids, meta_map, meta_bucket_field=str(args.meta_bucket_field))
        table = _compute_bucket_table(
            item_to_rows=item_to_rows,
            buckets_by_item=buckets_by_item,
            methods=methods,
            uniform_method=uniform_method,
            compare_methods=compare_methods,
        )
        md = _render_bucket_md(
            title=f"Bucketed by meta:{args.meta_bucket_field}",
            primary_method=primary_method_effective,
            table=table,
            uniform_method=uniform_method,
            top_k=10,
        )
        bucket_reports.append(
            {
                "bucket": {
                    "kind": "meta",
                    "meta_path": str(meta_path),
                    "meta_id_keys": meta_id_keys,
                    "field": str(args.meta_bucket_field),
                },
                "table": table,
                "md": md,
            }
        )

    # q_bar bucket.
    qbar_method = str(args.qbar_method).strip()
    if qbar_method and (not bool(args.no_qbar)):
        if qbar_method not in methods:
            # If the method isn't present, skip silently (still record in payload).
            qbar_bucket = {}
            qbar_meta = {"ok": False, "reason": f"qbar_method={qbar_method} not in methods={methods}"}
        else:
            quantiles = [float(x) for x in str(args.qbar_quantiles).split(",") if x.strip()]
            qbar_bucket, qbar_meta = _bucket_by_qbar_quantiles(item_to_rows, qbar_method=qbar_method, quantiles=quantiles)

        if qbar_bucket and qbar_meta.get("ok"):
            table = _compute_bucket_table(
                item_to_rows=item_to_rows,
                buckets_by_item=qbar_bucket,
                methods=methods,
                uniform_method=uniform_method,
                compare_methods=compare_methods,
            )
            md = _render_bucket_md(
                title=f"Bucketed by q_bar quantiles (method={qbar_method})",
                primary_method=primary_method_effective,
                table=table,
                uniform_method=uniform_method,
                top_k=10,
            )
            bucket_reports.append(
                {
                    "bucket": {"kind": "q_bar_quantile", **qbar_meta},
                    "table": table,
                    "md": md,
                }
            )
        else:
            bucket_reports.append({"bucket": {"kind": "q_bar_quantile", **qbar_meta}, "table": [], "md": ""})

    # Render a compact markdown summary.
    md_lines = [
        f"# QA Bucket Report",
        "",
        f"- task: `{task}`",
        f"- predictions: `{pred_path}`",
        f"- n_items: `{len(item_to_rows)}`",
        f"- methods: `{', '.join(methods)}`",
        f"- uniform_method: `{uniform_method}`",
        f"- primary_method: `{primary_method}` (effective: `{primary_method_effective}`)",
        "",
        "## Overall",
        "",
        "| method | n | acc | invalid_rate |",
        "|---|---:|---:|---:|",
    ]
    for m in methods:
        st = overall[m]
        md_lines.append(f"| `{m}` | {int(st['n'])} | {float(st['acc']):.4f} | {float(st['invalid_rate']):.4f} |")
    md_lines.append("")

    for br in bucket_reports:
        md = br.get("md", "")
        if md:
            md_lines.append(md.rstrip())
            md_lines.append("")

    md_text = "\n".join(md_lines).rstrip() + "\n"

    payload = {
        "ok": True,
        "task": task,
        "predictions_jsonl": str(pred_path),
        "n_items": int(len(item_to_rows)),
        "id_keys": id_keys,
        "uniform_method": uniform_method,
        "methods": methods,
        "compare_methods": compare_methods,
        "primary_method": primary_method,
        "primary_method_effective": primary_method_effective,
        "overall": overall,
        "bucket_reports": [
            {k: v for k, v in br.items() if k != "md"}  # md lives in bucket_report.md
            for br in bucket_reports
        ],
        "artifacts": {
            "bucket_report_json": str(out_dir / "bucket_report.json"),
            "bucket_report_md": str(out_dir / "bucket_report.md"),
        },
    }

    (out_dir / "bucket_report.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "bucket_report.md").write_text(md_text, encoding="utf-8")
    print(out_dir / "bucket_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
