from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from avs.experiments.qa_bucket_report import (
    _bucket_by_meta,
    _bucket_by_qbar_quantiles,
    _detect_task_and_id_keys,
    _load_meta_map,
    _read_jsonl,
)


def _bootstrap_mean_ci_and_p(xs: np.ndarray, *, seed: int, n_boot: int = 5000, alpha: float = 0.05) -> dict[str, float | int]:
    rng = np.random.default_rng(int(seed))
    n = int(xs.size)
    if n <= 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "p_two_sided": 1.0, "n": 0, "n_boot": int(n_boot)}
    means = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        means.append(float(xs[idx].mean()))
    arr = np.asarray(means, dtype=np.float64)
    lo = float(np.quantile(arr, float(alpha / 2.0)))
    hi = float(np.quantile(arr, float(1.0 - alpha / 2.0)))
    p_two = float(2.0 * min(np.mean(arr <= 0.0), np.mean(arr >= 0.0)))
    return {"mean": float(xs.mean()), "lo": lo, "hi": hi, "p_two_sided": p_two, "n": n, "n_boot": int(n_boot)}


def _stable_offset(s: str, mod: int = 100000) -> int:
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()
    return int(h[:12], 16) % int(mod)


def _build_item_to_rows(pred_path: Path) -> tuple[str, list[str], dict[str, dict[str, dict[str, Any]]], list[str]]:
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
    return task, id_keys, item_to_rows, sorted(methods_set)


def _eval_bucket_significance(
    *,
    item_to_rows: dict[str, dict[str, dict[str, Any]]],
    buckets_by_item: dict[str, str],
    uniform_method: str,
    primary_method: str,
    n_boot: int,
    bootstrap_seed: int,
    min_n: int,
) -> list[dict[str, Any]]:
    bucket_to_items: dict[str, list[str]] = {}
    for item_id, b in buckets_by_item.items():
        bucket_to_items.setdefault(str(b), []).append(str(item_id))

    rows: list[dict[str, Any]] = []
    for b, items in sorted(bucket_to_items.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        diffs: list[float] = []
        acc_u: list[float] = []
        acc_p: list[float] = []
        for it in items:
            pair = item_to_rows.get(it, {})
            ru = pair.get(uniform_method)
            rp = pair.get(primary_method)
            if ru is None or rp is None:
                continue
            yu = ru.get("correct")
            yp = rp.get("correct")
            if yu is None or yp is None:
                continue
            u = 1.0 if bool(yu) else 0.0
            p = 1.0 if bool(yp) else 0.0
            diffs.append(float(p - u))
            acc_u.append(u)
            acc_p.append(p)
        arr = np.asarray(diffs, dtype=np.float64)
        stat = _bootstrap_mean_ci_and_p(
            arr,
            seed=int(bootstrap_seed) + _stable_offset(f"bucket:{b}", mod=104729),
            n_boot=int(n_boot),
            alpha=0.05,
        )
        rows.append(
            {
                "bucket": str(b),
                "n_items": int(len(items)),
                "n_paired": int(arr.size),
                "uniform_acc": float(np.mean(acc_u)) if acc_u else 0.0,
                "primary_acc": float(np.mean(acc_p)) if acc_p else 0.0,
                "delta_mean": float(stat["mean"]),
                "delta_ci95": {"lo": float(stat["lo"]), "hi": float(stat["hi"])},
                "p_bootstrap_two_sided": float(stat["p_two_sided"]),
                "significant": bool((arr.size >= int(min_n)) and (float(stat["lo"]) > 0.0 or float(stat["hi"]) < 0.0)),
            }
        )
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description="Bucket-level significance report for QA predictions (primary vs uniform).")
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--uniform-method", type=str, default="uniform")
    p.add_argument("--primary-method", type=str, required=True)
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--bootstrap-seed", type=int, default=0)
    p.add_argument("--min-n", type=int, default=20)

    p.add_argument("--meta", type=Path, default=None)
    p.add_argument("--meta-id-key", type=str, default="")
    p.add_argument("--meta-bucket-field", type=str, default="")

    p.add_argument("--qbar-method", type=str, default="")
    p.add_argument("--qbar-quantiles", type=str, default="0.33,0.66")
    p.add_argument("--no-qbar", action="store_true")
    args = p.parse_args()

    task, id_keys, item_to_rows, methods = _build_item_to_rows(Path(args.predictions))
    uniform_method = str(args.uniform_method)
    primary_method = str(args.primary_method)
    if uniform_method not in methods:
        raise ValueError(f"uniform_method={uniform_method} not found in methods={methods}")
    if primary_method not in methods:
        raise ValueError(f"primary_method={primary_method} not found in methods={methods}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reports: list[dict[str, Any]] = []
    item_ids = sorted(item_to_rows.keys())

    if str(args.meta_bucket_field).strip():
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
        table = _eval_bucket_significance(
            item_to_rows=item_to_rows,
            buckets_by_item=buckets_by_item,
            uniform_method=uniform_method,
            primary_method=primary_method,
            n_boot=int(args.n_boot),
            bootstrap_seed=int(args.bootstrap_seed),
            min_n=int(args.min_n),
        )
        reports.append(
            {
                "bucket": {
                    "kind": "meta",
                    "meta_path": str(meta_path),
                    "meta_id_keys": meta_id_keys,
                    "field": str(args.meta_bucket_field),
                },
                "table": table,
            }
        )

    qbar_method = str(args.qbar_method).strip()
    if qbar_method and (not bool(args.no_qbar)):
        quantiles = [float(x) for x in str(args.qbar_quantiles).split(",") if x.strip()]
        buckets, qmeta = _bucket_by_qbar_quantiles(item_to_rows, qbar_method=qbar_method, quantiles=quantiles)
        if qmeta.get("ok") and buckets:
            table = _eval_bucket_significance(
                item_to_rows=item_to_rows,
                buckets_by_item=buckets,
                uniform_method=uniform_method,
                primary_method=primary_method,
                n_boot=int(args.n_boot),
                bootstrap_seed=int(args.bootstrap_seed) + 97,
                min_n=int(args.min_n),
            )
        else:
            table = []
        reports.append({"bucket": {"kind": "q_bar_quantile", **qmeta}, "table": table})

    payload = {
        "ok": True,
        "task": task,
        "predictions_jsonl": str(args.predictions),
        "methods": methods,
        "uniform_method": uniform_method,
        "primary_method": primary_method,
        "n_boot": int(args.n_boot),
        "bootstrap_seed": int(args.bootstrap_seed),
        "min_n": int(args.min_n),
        "reports": reports,
        "artifacts": {
            "bucket_significance_json": str(out_dir / "bucket_significance.json"),
            "bucket_significance_md": str(out_dir / "bucket_significance.md"),
        },
    }

    md = [
        "# QA Bucket Significance",
        "",
        f"- task: `{task}`",
        f"- predictions: `{args.predictions}`",
        f"- primary_method: `{primary_method}` vs `{uniform_method}`",
        f"- min_n: `{int(args.min_n)}`",
        "",
    ]
    for rep in reports:
        bk = rep["bucket"]
        md.append(f"## Bucket: `{bk.get('kind')}`")
        if bk.get("kind") == "meta":
            md.append(f"- field: `{bk.get('field')}`")
        if bk.get("kind") == "q_bar_quantile":
            md.append(f"- method: `{bk.get('method', qbar_method)}`")
        md += [
            "",
            "| bucket | n_items | n_paired | uniform_acc | primary_acc | delta | ci95 | p_boot | significant |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rep.get("table", []):
            ci = row.get("delta_ci95", {"lo": 0.0, "hi": 0.0})
            md.append(
                f"| {row.get('bucket')} | {row.get('n_items')} | {row.get('n_paired')} | "
                f"{row.get('uniform_acc', 0.0):.4f} | {row.get('primary_acc', 0.0):.4f} | "
                f"{row.get('delta_mean', 0.0):+.4f} | "
                f"[{ci.get('lo', 0.0):+.4f}, {ci.get('hi', 0.0):+.4f}] | "
                f"{row.get('p_bootstrap_two_sided', 1.0):.4f} | {str(bool(row.get('significant', False))).lower()} |"
            )
        md.append("")

    (out_dir / "bucket_significance.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "bucket_significance.md").write_text("\n".join(md), encoding="utf-8")
    print(out_dir / "bucket_significance.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
