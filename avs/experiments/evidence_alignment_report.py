from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from avs.datasets.ave import AVEIndex, ensure_ave_meta
from avs.metrics.time_windows import TimeWindow, coverage_at_tau


def _extract_run_and_metrics(obj: dict) -> tuple[dict, dict]:
    # Prefer end-to-end runner format: {"metrics": {...}, "meta_dir": ..., "eval_ids": ...}
    if isinstance(obj.get("metrics"), dict):
        return obj, obj["metrics"]
    return obj, obj


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask].astype(np.float64)
    y = y[mask].astype(np.float64)
    if x.size < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt((x * x).sum()) * np.sqrt((y * y).sum()))
    if denom <= 0.0:
        return float("nan")
    return float((x * y).sum() / denom)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask].astype(np.float64)
    y = y[mask].astype(np.float64)
    if x.size < 2:
        return float("nan")

    try:
        from scipy.stats import rankdata  # type: ignore[import-not-found]

        rx = rankdata(x, method="average")
        ry = rankdata(y, method="average")
    except Exception:
        rx = np.empty_like(x, dtype=np.float64)
        ry = np.empty_like(y, dtype=np.float64)
        rx[np.argsort(x)] = np.arange(x.size, dtype=np.float64)
        ry[np.argsort(y)] = np.arange(y.size, dtype=np.float64)

    return _pearson(rx, ry)


def _segment_windows_from_labels(labels: list[int]) -> list[TimeWindow]:
    out: list[TimeWindow] = []
    for i, lab in enumerate(labels):
        if int(lab) == 0:
            continue
        out.append(TimeWindow(start_s=float(i), end_s=float(i + 1)))
    return out


def _windows_from_anchors(anchors: list[int], *, num_segments: int, delta_s: float) -> list[TimeWindow]:
    out: list[TimeWindow] = []
    d = float(delta_s)
    for a in anchors:
        i = int(a)
        if i < 0 or i >= int(num_segments):
            continue
        start = max(0.0, float(i) - d)
        end = min(float(num_segments), float(i + 1) + d)
        out.append(TimeWindow(start_s=float(start), end_s=float(end)))
    return out


def _parse_csv_floats(value: str) -> list[float]:
    out: list[float] = []
    for s in str(value).split(","):
        s = s.strip()
        if not s:
            continue
        out.append(float(s))
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evidence Alignment report: Coverage@τ and correlation with anchored gains.")
    p.add_argument("--in-metrics", type=Path, required=True, help="Path to ave_p0 metrics.json (or ave_p0_end2end wrapper).")
    p.add_argument(
        "--meta-dir",
        type=Path,
        default=None,
        help="AVE meta dir. If not provided, attempts to infer from the metrics file (ave_p0_end2end format only).",
    )
    p.add_argument("--out-dir", type=Path, default=Path("runs") / f"E0202_evidence_alignment_{time.strftime('%Y%m%d-%H%M%S')}")
    p.add_argument("--tau-grid", type=str, default="0.3,0.5,0.7", help="Comma-separated IoU thresholds τ for Coverage@τ.")
    p.add_argument("--delta-s", type=float, default=0.0, help="Expand predicted anchor windows by ±delta_s seconds.")
    p.add_argument("--top-n", type=int, default=20)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    obj = json.loads(args.in_metrics.read_text(encoding="utf-8"))
    run, metrics = _extract_run_and_metrics(obj)

    clip_ids_eval = run.get("eval_ids") or metrics.get("debug_eval_clip_ids")
    if not clip_ids_eval:
        raise SystemExit("need eval ids: pass ave_p0_end2end metrics.json (or a metrics.json that contains debug_eval_clip_ids)")
    clip_ids_eval = [str(x) for x in clip_ids_eval]

    meta_dir = run.get("meta_dir")
    if meta_dir is None:
        meta_dir = args.meta_dir
    if meta_dir is None:
        raise SystemExit("need --meta-dir (or pass ave_p0_end2end metrics.json that contains meta_dir)")
    meta_dir = Path(meta_dir)
    ensure_ave_meta(meta_dir)
    index = AVEIndex.from_meta_dir(meta_dir)
    clip_by_id = {c.video_id: c for c in index.clips}

    debug_eval = metrics.get("debug_eval") or {}
    dbg = debug_eval.get("anchored_top2") or {}
    if not dbg:
        raise SystemExit("metrics missing debug_eval.anchored_top2 (anchors); rerun with updated code")

    results = metrics.get("results_by_seed") or []
    if not results:
        raise SystemExit("metrics missing results_by_seed")

    # Build per-sample mean delta across seeds.
    deltas_by_seed: list[np.ndarray] = []
    for r in results:
        b = r.get("baselines") or {}
        u = np.asarray(b.get("uniform", {}).get("val_acc_by_sample") or [], dtype=np.float32)
        a = np.asarray(b.get("anchored_top2", {}).get("val_acc_by_sample") or [], dtype=np.float32)
        if u.size != a.size or u.size != len(clip_ids_eval):
            raise SystemExit("per-sample arrays missing/wrong length; need val_acc_by_sample for uniform/anchored_top2")
        deltas_by_seed.append(a - u)
    delta_mean_by_sample = np.mean(np.stack(deltas_by_seed, axis=0), axis=0)

    # Aggregate per unique clip id (some runs can contain duplicates if allow-missing filtering differs).
    idxs_by_clip: dict[str, list[int]] = {}
    for i, cid in enumerate(clip_ids_eval):
        idxs_by_clip.setdefault(cid, []).append(int(i))

    delta_by_clip: dict[str, float] = {}
    cov_by_clip_by_tau: dict[str, dict[str, float]] = {}

    taus = _parse_csv_floats(args.tau_grid)
    for cid, idxs in idxs_by_clip.items():
        if cid not in clip_by_id or cid not in dbg:
            continue

        clip = clip_by_id[cid]
        labels = [int(x) for x in index.segment_labels(clip)]
        evidence = _segment_windows_from_labels(labels)

        anchors = [int(x) for x in (dbg.get(cid) or {}).get("anchors") or []]
        pred_windows = _windows_from_anchors(anchors, num_segments=10, delta_s=float(args.delta_s))

        delta_by_clip[cid] = float(np.mean(delta_mean_by_sample[idxs]))
        cov_by_tau: dict[str, float] = {}
        for tau in taus:
            cov_by_tau[f"{tau:.3f}"] = float(coverage_at_tau(windows=pred_windows, evidence=evidence, tau=float(tau)))
        cov_by_clip_by_tau[cid] = cov_by_tau

    common = sorted(set(delta_by_clip.keys()).intersection(cov_by_clip_by_tau.keys()))
    if not common:
        raise SystemExit("no usable clips after aligning metrics/debug/meta")

    out_cov_by_tau: dict[str, dict] = {}
    out_corr_by_tau: dict[str, dict] = {}

    delta_arr = np.asarray([delta_by_clip[c] for c in common], dtype=np.float32)
    for tau in taus:
        k = f"{tau:.3f}"
        cov_arr = np.asarray([cov_by_clip_by_tau[c][k] for c in common], dtype=np.float32)
        out_cov_by_tau[k] = {"mean": float(cov_arr.mean()), "std": float(cov_arr.std(ddof=1)) if cov_arr.size > 1 else 0.0}
        out_corr_by_tau[k] = {"pearson": _pearson(cov_arr, delta_arr), "spearman": _spearman(cov_arr, delta_arr), "n": int(cov_arr.size)}

    # Worst-case lists (low coverage + negative delta).
    top_n = max(1, int(args.top_n))
    # Choose a default tau for ranking (middle of grid).
    tau_rank = taus[len(taus) // 2] if taus else 0.5
    tau_key = f"{float(tau_rank):.3f}"
    ordered = sorted(common, key=lambda c: (cov_by_clip_by_tau[c][tau_key], delta_by_clip[c], c))
    worst = ordered[:top_n]
    best = ordered[-top_n:][::-1]

    def _row(c: str) -> dict:
        return {"clip_id": str(c), "delta": float(delta_by_clip[c]), "cov_by_tau": cov_by_clip_by_tau[c]}

    out = {
        "ok": True,
        "in_metrics": str(args.in_metrics),
        "meta_dir": str(meta_dir),
        "delta_s": float(args.delta_s),
        "tau_grid": [float(x) for x in taus],
        "num_eval_ids": int(len(clip_ids_eval)),
        "num_unique_eval_ids": int(len(idxs_by_clip)),
        "num_clips_used": int(len(common)),
        "cov_by_tau": out_cov_by_tau,
        "corr_by_tau": out_corr_by_tau,
        "top_improve": [_row(c) for c in best],
        "top_degrade": [_row(c) for c in worst],
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "evidence_alignment.json"
    out_json.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

