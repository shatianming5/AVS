from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from avs.datasets.ave import AVEIndex, ensure_ave_meta
from avs.metrics.anchors import recall_at_k


def _extract_run_and_metrics(obj: dict) -> tuple[dict, dict]:
    # Prefer end-to-end runner format: {"metrics": {...}, "meta_dir": ..., "eval_ids": ...}
    if isinstance(obj.get("metrics"), dict):
        return obj, obj["metrics"]
    return obj, obj


def _to_float_array(xs: list[float | None]) -> np.ndarray:
    out = np.empty((len(xs),), dtype=np.float32)
    for i, x in enumerate(xs):
        out[i] = np.nan if x is None else float(x)
    return out


def _percentiles(x: np.ndarray, ps: list[int]) -> dict[str, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"p{p}": float("nan") for p in ps}
    vals = np.percentile(x, ps).astype(np.float64)
    return {f"p{p}": float(v) for p, v in zip(ps, vals, strict=True)}


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
        # Fallback: ordinal ranks (ties get arbitrary order). Good enough for quick diagnostics.
        rx = np.empty_like(x, dtype=np.float64)
        ry = np.empty_like(y, dtype=np.float64)
        rx[np.argsort(x)] = np.arange(x.size, dtype=np.float64)
        ry[np.argsort(y)] = np.arange(y.size, dtype=np.float64)

    return _pearson(rx, ry)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Diagnose why anchored gains are small using per-clip deltas and anchor recall correlation.")
    p.add_argument("--in-metrics", type=Path, required=True, help="Path to ave_p0_end2end metrics.json (or ave_p0.py metrics.json).")
    p.add_argument("--out-dir", type=Path, default=Path("runs") / f"AVE_P0_DIAGNOSE_{time.strftime('%Y%m%d-%H%M%S')}")
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--deltas", type=str, default="0,1,2", help="Comma-separated dilation deltas for Recall@K,Δ.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    obj = json.loads(args.in_metrics.read_text(encoding="utf-8"))
    run, metrics = _extract_run_and_metrics(obj)

    results = metrics.get("results_by_seed") or []
    if not results:
        raise SystemExit("metrics missing results_by_seed")

    # Eval ids are only available in end-to-end runs; otherwise attempt to read debug field.
    clip_ids_eval = run.get("eval_ids") or metrics.get("debug_eval_clip_ids")
    if not clip_ids_eval:
        raise SystemExit("need eval ids: pass ave_p0_end2end metrics.json")
    clip_ids_eval = [str(x) for x in clip_ids_eval]

    # Per-clip (unique) aggregation.
    idxs_by_clip: dict[str, list[int]] = {}
    for i, cid in enumerate(clip_ids_eval):
        idxs_by_clip.setdefault(cid, []).append(int(i))

    # Collect per-seed deltas.
    deltas_by_seed: list[np.ndarray] = []
    deltas_event_by_seed: list[np.ndarray] = []
    for r in results:
        b = r.get("baselines") or {}
        if "uniform" not in b or "anchored_top2" not in b:
            raise SystemExit("missing uniform/anchored_top2 in results_by_seed")

        u = np.asarray(b["uniform"].get("val_acc_by_sample") or [], dtype=np.float32)
        a = np.asarray(b["anchored_top2"].get("val_acc_by_sample") or [], dtype=np.float32)
        if u.size != a.size or u.size != len(clip_ids_eval):
            raise SystemExit("per-sample arrays missing or wrong length; rerun with updated code to record val_acc_by_sample")

        deltas_by_seed.append(a - u)

        ue = _to_float_array(b["uniform"].get("val_acc_event_by_sample") or [])
        ae = _to_float_array(b["anchored_top2"].get("val_acc_event_by_sample") or [])
        if ue.size != ae.size or ue.size != len(clip_ids_eval):
            raise SystemExit("event per-sample arrays missing or wrong length; rerun with updated code to record val_acc_event_by_sample")
        deltas_event_by_seed.append(ae - ue)

    delta_mean_by_sample = np.mean(np.stack(deltas_by_seed, axis=0), axis=0)
    delta_event_mean_by_sample = np.nanmean(np.stack(deltas_event_by_seed, axis=0), axis=0)

    delta_by_clip: dict[str, float] = {}
    delta_event_by_clip: dict[str, float] = {}
    for cid, idxs in idxs_by_clip.items():
        d = float(np.mean(delta_mean_by_sample[idxs]))
        delta_by_clip[cid] = d

        de = delta_event_mean_by_sample[idxs]
        de = de[np.isfinite(de)]
        delta_event_by_clip[cid] = float(np.mean(de)) if de.size else float("nan")

    # Anchor recall correlation.
    debug_eval = metrics.get("debug_eval") or {}
    dbg = debug_eval.get("anchored_top2") or {}
    if not dbg:
        raise SystemExit("metrics missing debug_eval.anchored_top2 (anchors/scores/plan); rerun ave_p0_end2end with updated code")

    meta_dir = run.get("meta_dir")
    if meta_dir is None:
        raise SystemExit("metrics missing meta_dir (need AVE annotations for GT segments)")
    meta_dir = Path(meta_dir)
    ensure_ave_meta(meta_dir)
    index = AVEIndex.from_meta_dir(meta_dir)
    clip_by_id = {c.video_id: c for c in index.clips}

    deltas = [int(x) for x in str(args.deltas).split(",") if str(x).strip()]
    recall_by_delta: dict[int, dict[str, float]] = {d: {} for d in deltas}

    for cid in idxs_by_clip.keys():
        if cid not in dbg or cid not in clip_by_id:
            continue
        anchors = [int(x) for x in dbg[cid].get("anchors") or []]
        gt = [i for i, lab in enumerate(index.segment_labels(clip_by_id[cid])) if int(lab) != 0]
        for d in deltas:
            recall_by_delta[int(d)][cid] = float(recall_at_k(gt, anchors, num_segments=10, delta=int(d)).recall)

    # Correlations (use Δ0 by default).
    d0 = deltas[0] if deltas else 0
    common = sorted(set(delta_by_clip.keys()).intersection(recall_by_delta.get(int(d0), {}).keys()))
    delta_arr = np.asarray([delta_by_clip[c] for c in common], dtype=np.float32)
    recall_arr = np.asarray([recall_by_delta[int(d0)][c] for c in common], dtype=np.float32)
    corr = {"pearson": _pearson(delta_arr, recall_arr), "spearman": _spearman(delta_arr, recall_arr), "n": int(len(common))}

    # Top/bottom lists.
    top_n = max(1, int(args.top_n))
    ordered = sorted(common, key=lambda c: float(delta_by_clip.get(c, float("nan"))))
    worst = ordered[:top_n]
    best = ordered[-top_n:][::-1]

    top_improve = [
        {
            "clip_id": c,
            "delta": float(delta_by_clip[c]),
            "delta_event": float(delta_event_by_clip.get(c, float("nan"))),
            "recall_d0": float(recall_by_delta[int(d0)].get(c, float("nan"))),
        }
        for c in best
    ]
    top_degrade = [
        {
            "clip_id": c,
            "delta": float(delta_by_clip[c]),
            "delta_event": float(delta_event_by_clip.get(c, float("nan"))),
            "recall_d0": float(recall_by_delta[int(d0)].get(c, float("nan"))),
        }
        for c in worst
    ]

    delta_vals = np.asarray(list(delta_by_clip.values()), dtype=np.float32)
    delta_event_vals = np.asarray(list(delta_event_by_clip.values()), dtype=np.float32)

    # Anchor/plan diagnostics (root-cause: how often we fall back; when 2-high anchors hurt).
    from collections import Counter

    anchors_len_by_clip: dict[str, int] = {}
    high_count_by_clip: dict[str, int] = {}
    gap_by_clip: dict[str, float] = {}
    dist_by_clip: dict[str, int] = {}
    fallback_used_by_clip: dict[str, bool] = {}
    fallback_reason_by_clip: dict[str, str | None] = {}
    conf_value_by_clip: dict[str, float] = {}

    for cid in common:
        info = dbg.get(cid) or {}
        anchors = [int(x) for x in info.get("anchors") or []]
        scores = [float(x) for x in info.get("scores") or []]
        plan_res = [int(x) for x in info.get("plan_resolutions") or []]

        anchors_len_by_clip[cid] = int(len(anchors))
        fallback_used_by_clip[cid] = bool(info.get("fallback_used", len(anchors) == 0))
        fallback_reason_by_clip[cid] = info.get("fallback_reason")
        try:
            conf_value_by_clip[cid] = float(info.get("conf_value"))
        except Exception:
            conf_value_by_clip[cid] = float("nan")

        uniq = sorted(set(plan_res))
        if len(uniq) <= 1:
            high_count_by_clip[cid] = 0
        else:
            high_res = max(uniq)
            high_count_by_clip[cid] = int(sum(1 for r in plan_res if int(r) == int(high_res)))

        if len(scores) >= 2:
            order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
            gap_by_clip[cid] = float(scores[order[0]] - scores[order[1]])
        else:
            gap_by_clip[cid] = 0.0

        if len(anchors) >= 2:
            dist_by_clip[cid] = int(abs(int(anchors[0]) - int(anchors[1])))

    anchors_len_hist = Counter(anchors_len_by_clip.values())
    high_count_hist = Counter(high_count_by_clip.values())
    dist_hist = Counter(dist_by_clip.values())
    fallback_reason_hist = Counter(
        str(fallback_reason_by_clip.get(c) or "unknown") for c in common if bool(fallback_used_by_clip.get(c, False))
    )

    def _mean(xs: list[float]) -> float:
        return float(np.asarray(xs, dtype=np.float64).mean()) if xs else float("nan")

    delta_by_anchor_len = {
        str(k): {"n": int(v), "mean_delta": _mean([delta_by_clip[c] for c in common if anchors_len_by_clip.get(c) == k])}
        for k, v in sorted(anchors_len_hist.items())
    }
    delta_by_high_count = {
        str(k): {"n": int(v), "mean_delta": _mean([delta_by_clip[c] for c in common if high_count_by_clip.get(c) == k])}
        for k, v in sorted(high_count_hist.items())
    }
    delta_by_anchor_dist = {
        str(k): {"n": int(v), "mean_delta": _mean([delta_by_clip[c] for c in common if dist_by_clip.get(c) == k])}
        for k, v in sorted(dist_hist.items())
    }

    gap_arr = np.asarray([gap_by_clip[c] for c in common], dtype=np.float32)
    conf_arr = np.asarray([conf_value_by_clip[c] for c in common], dtype=np.float32)
    conf_fallback_arr = np.asarray(
        [conf_value_by_clip[c] for c in common if bool(fallback_used_by_clip.get(c, False))], dtype=np.float32
    )

    payload = {
        "in_metrics": str(args.in_metrics),
        "num_eval_ids": int(len(clip_ids_eval)),
        "num_unique_eval_ids": int(len(idxs_by_clip)),
        "deltas": deltas,
        "delta_summary": {
            "mean": float(np.nanmean(delta_vals)),
            "std": float(np.nanstd(delta_vals, ddof=1)) if delta_vals.size > 1 else 0.0,
            **_percentiles(delta_vals, [10, 50, 90]),
        },
        "delta_event_summary": {
            "mean": float(np.nanmean(delta_event_vals)),
            "std": float(np.nanstd(delta_event_vals, ddof=1)) if delta_event_vals.size > 1 else 0.0,
            **_percentiles(delta_event_vals, [10, 50, 90]),
        },
        "anchor_plan_stats": {
            "anchors_len_hist": {str(k): int(v) for k, v in sorted(anchors_len_hist.items())},
            "anchors_len_fallback_frac": float(anchors_len_hist.get(0, 0) / max(1, len(common))),
            "fallback_used_frac": float(sum(1 for c in common if bool(fallback_used_by_clip.get(c, False))) / max(1, len(common))),
            "fallback_reason_hist": {str(k): int(v) for k, v in sorted(fallback_reason_hist.items())},
            "high_count_hist": {str(k): int(v) for k, v in sorted(high_count_hist.items())},
            "anchor_dist_hist": {str(k): int(v) for k, v in sorted(dist_hist.items())},
            "gap_summary": {**_percentiles(gap_arr, [10, 50, 90]), "mean": float(np.nanmean(gap_arr))},
            "conf_summary": {**_percentiles(conf_arr, [10, 50, 90]), "mean": float(np.nanmean(conf_arr))},
            "conf_fallback_summary": {**_percentiles(conf_fallback_arr, [10, 50, 90]), "mean": float(np.nanmean(conf_fallback_arr))},
            "delta_by_anchor_len": delta_by_anchor_len,
            "delta_by_high_count": delta_by_high_count,
            "delta_by_anchor_dist": delta_by_anchor_dist,
        },
        "corr_delta_vs_recall": corr,
        "top_improve": top_improve,
        "top_degrade": top_degrade,
        "notes": [
            "delta = mean_over_seeds(acc_anchored - acc_uniform) per clip (eval split).",
            "recall_d0 uses anchors from debug_eval.anchored_top2 and GT segments from AVE annotations.",
            "If corr is low, anchored gains may be dominated by classifier noise or misalignment; inspect top_degrade clips.",
        ],
    }

    # Keep full per-clip maps in a separate section (still small: ~402 entries).
    payload["per_clip"] = {
        "delta": delta_by_clip,
        "delta_event": delta_event_by_clip,
        "recall_by_delta": {str(d): recall_by_delta[int(d)] for d in deltas},
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "diagnose.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
