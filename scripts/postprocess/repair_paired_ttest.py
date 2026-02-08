#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _ttest_rel(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    from scipy.stats import t as student_t  # type: ignore

    dx = np.asarray(y, dtype=np.float64) - np.asarray(x, dtype=np.float64)
    n = int(dx.size)
    if n < 2:
        return {"t": float("nan"), "p": float("nan")}
    mean = float(dx.mean())
    std = float(dx.std(ddof=1))
    if std <= 0.0:
        if mean == 0.0:
            return {"t": 0.0, "p": 1.0}
        return {"t": float("inf") if mean > 0.0 else float("-inf"), "p": 0.0}

    t = mean / (std / float(np.sqrt(float(n))))
    p = float(2.0 * float(student_t.sf(abs(t), df=float(n - 1))))
    p = float(max(0.0, min(1.0, p)))
    return {"t": float(t), "p": float(p)}


def _paired_tests_from_results(results_by_seed: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_baseline: dict[str, list[float]] = {}
    for r in results_by_seed:
        baselines = r.get("baselines") or {}
        for b, v in baselines.items():
            if not isinstance(v, dict):
                continue
            if "val_acc" not in v:
                continue
            by_baseline.setdefault(str(b), []).append(float(v["val_acc"]))

    def arr(name: str) -> np.ndarray | None:
        vals = by_baseline.get(name)
        if not vals:
            return None
        return np.asarray(vals, dtype=np.float64)

    tests: dict[str, dict[str, float]] = {}

    anchored = arr("anchored_top2")
    uniform = arr("uniform")
    random_top2 = arr("random_top2")
    oracle = arr("oracle_top2")
    audio_concat_uniform = arr("audio_concat_uniform")
    audio_concat_anchored = arr("audio_concat_anchored_top2")
    audio_feat_concat_uniform = arr("audio_feat_concat_uniform")

    if anchored is not None and uniform is not None and anchored.size == uniform.size:
        tests["anchored_vs_uniform"] = _ttest_rel(uniform, anchored)
        if random_top2 is not None and random_top2.size == anchored.size:
            tests["anchored_vs_random"] = _ttest_rel(random_top2, anchored)
        if audio_concat_uniform is not None and audio_concat_uniform.size == anchored.size:
            tests["anchored_vs_audio_concat_uniform"] = _ttest_rel(audio_concat_uniform, anchored)
        if audio_feat_concat_uniform is not None and audio_feat_concat_uniform.size == anchored.size:
            tests["anchored_vs_audio_feat_concat_uniform"] = _ttest_rel(audio_feat_concat_uniform, anchored)

    if oracle is not None and uniform is not None and oracle.size == uniform.size:
        tests["oracle_vs_uniform"] = _ttest_rel(uniform, oracle)

    if audio_concat_anchored is not None and audio_concat_uniform is not None and audio_concat_anchored.size == audio_concat_uniform.size:
        tests["audio_concat_anchored_vs_audio_concat_uniform"] = _ttest_rel(audio_concat_uniform, audio_concat_anchored)

    return tests


def _repair_metrics(path: Path, *, overwrite: bool) -> bool:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False

    results_by_seed = obj.get("results_by_seed")
    if not isinstance(results_by_seed, list) or not results_by_seed:
        return False

    tests = _paired_tests_from_results(results_by_seed)
    if not tests:
        return False

    changed = False
    if obj.get("paired_ttest") != tests:
        obj["paired_ttest"] = tests
        changed = True

    if changed and overwrite:
        path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return changed


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Repair paired t-test fields in metrics.json files using SciPy.")
    p.add_argument("paths", nargs="+", type=Path, help="metrics.json file(s) or directories to scan recursively")
    p.add_argument("--overwrite", action="store_true", help="Write changes back to disk (default: dry-run)")
    args = p.parse_args(argv)

    metrics_files: list[Path] = []
    for raw in args.paths:
        if raw.is_dir():
            metrics_files.extend(sorted(raw.rglob("metrics.json")))
        else:
            metrics_files.append(raw)

    changed_paths: list[Path] = []
    for mp in metrics_files:
        if _repair_metrics(mp, overwrite=bool(args.overwrite)):
            changed_paths.append(mp)

    report = {
        "overwrite": bool(args.overwrite),
        "num_scanned": int(len(metrics_files)),
        "num_changed": int(len(changed_paths)),
        "changed": [str(p) for p in changed_paths[:200]],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

