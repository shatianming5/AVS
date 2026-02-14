from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from avs.datasets.layout import videomme_paths
from avs.datasets.videomme import VideoMMEItem, load_videomme_split
from avs.pipeline.qa_plan_generation import (
    QASecSelection,
    build_scores,
    random_seconds,
    select_seconds_alpha_mixture,
    select_seconds_maxinfo_maxvol_clip,
    select_seconds_mdp3_dpp_clip,
    select_seconds_qframe_gumbel_clip,
    uniform_seconds,
)
from avs.preprocess.video_extract import ensure_processed_fps1
from avs.vlm.qwen_vl import QwenVL, QwenVLConfig


@dataclass(frozen=True)
class MethodSummary:
    method: str
    n: int
    acc: float
    invalid_rate: float


def _bootstrap_ci_mean(xs: np.ndarray, *, seed: int, n_boot: int = 1000, alpha: float = 0.05) -> dict:
    rng = np.random.default_rng(int(seed))
    n = int(xs.size)
    if n <= 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n": 0}
    means = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        means.append(float(xs[idx].mean()))
    means = np.asarray(means, dtype=np.float64)
    lo = float(np.quantile(means, float(alpha / 2.0)))
    hi = float(np.quantile(means, float(1.0 - alpha / 2.0)))
    return {"mean": float(xs.mean()), "lo": lo, "hi": hi, "n": n, "n_boot": int(n_boot)}


def _duration_seconds_from_frames(frames_dir: Path) -> int:
    frames = sorted(frames_dir.glob("*.jpg"))
    if not frames:
        return 0
    try:
        mx = max(int(p.stem) for p in frames)
        return int(mx + 1)
    except Exception:
        return int(len(frames))


def _frame_paths(processed_dir: Path, seconds: list[int]) -> list[Path]:
    frames_dir = Path(processed_dir) / "frames"
    out: list[Path] = []
    for t in seconds:
        p = frames_dir / f"{int(t)}.jpg"
        if not p.exists():
            raise FileNotFoundError(f"missing frame: {p}")
        out.append(p)
    return out


def _eval_one(
    *,
    model: QwenVL,
    item: VideoMMEItem,
    processed_dir: Path,
    method: str,
    budget_frames: int,
    seed: int,
    max_seconds: int,
    strategy: str,
    ql2l_clap_device: str,
    ql2l_asr_device: str,
    ql2l_clip_device: str,
) -> dict:
    dur = min(int(_duration_seconds_from_frames(processed_dir / "frames")), int(max_seconds))
    if dur <= 0:
        raise ValueError(f"empty processed frames for youtube_id={item.youtube_id}")

    t0 = time.time()
    sel: QASecSelection | None = None
    scores_debug: dict | None = None
    m = str(method)

    row_budget = int(budget_frames)
    if m == "uniform":
        seconds = uniform_seconds(dur, int(budget_frames))
        sel = QASecSelection(
            selected_seconds=seconds,
            background_seconds=seconds,
            anchor_seconds=[],
            anchors=[],
            alpha=1.0,
            q_bar=0.0,
            reliability_metric="n/a",
        )
    elif m == "random":
        seconds = random_seconds(dur, int(budget_frames), seed=int(seed))
        sel = QASecSelection(
            selected_seconds=seconds,
            background_seconds=[],
            anchor_seconds=seconds,
            anchors=[],
            alpha=0.0,
            q_bar=0.0,
            reliability_metric="n/a",
        )
    elif m == "random_frame1":
        seconds = random_seconds(dur, 1, seed=int(seed))
        row_budget = 1
        sel = QASecSelection(
            selected_seconds=seconds,
            background_seconds=[],
            anchor_seconds=seconds,
            anchors=[],
            alpha=0.0,
            q_bar=0.0,
            reliability_metric="n/a",
        )
    elif m == "text_only":
        sel = QASecSelection(
            selected_seconds=[],
            background_seconds=[],
            anchor_seconds=[],
            anchors=[],
            alpha=1.0,
            q_bar=0.0,
            reliability_metric="n/a",
        )
    elif m == "qframe_gumbel_clip":
        sel, dbg = select_seconds_qframe_gumbel_clip(
            processed_dir=processed_dir,
            query=item.question,
            num_segments=int(dur),
            budget_frames=int(budget_frames),
            seed=int(seed),
            clip_device=str(ql2l_clip_device),
        )
        scores_debug = {"details": dbg}
    elif m == "maxinfo_maxvol_clip":
        sel, dbg = select_seconds_maxinfo_maxvol_clip(
            processed_dir=processed_dir,
            query=item.question,
            num_segments=int(dur),
            budget_frames=int(budget_frames),
            clip_device=str(ql2l_clip_device),
        )
        scores_debug = {"details": dbg}
    elif m == "mdp3_dpp_clip":
        sel, dbg = select_seconds_mdp3_dpp_clip(
            processed_dir=processed_dir,
            query=item.question,
            num_segments=int(dur),
            budget_frames=int(budget_frames),
            seed=int(seed),
            clip_device=str(ql2l_clip_device),
        )
        scores_debug = {"details": dbg}
    else:
        scores_debug = build_scores(
            processed_dir=processed_dir,
            query=item.question,
            num_segments=int(dur),
            method=m,
            seed=int(seed),
            clap_device=str(ql2l_clap_device),
            clip_device=str(ql2l_clip_device),
            asr_device=str(ql2l_asr_device),
        )
        scores = scores_debug["scores"]
        sel = select_seconds_alpha_mixture(scores=scores, budget_frames=int(budget_frames), seed=int(seed))

    stage1_s = float(time.time() - t0)
    img_paths = [] if m == "text_only" else _frame_paths(processed_dir, sel.selected_seconds)

    if str(strategy) == "ppl":
        ans = model.answer_mcq_ppl(image_paths=img_paths, question=item.question, options=item.options)
    elif str(strategy) == "generate":
        ans = model.answer_mcq_generate(image_paths=img_paths, question=item.question, options=item.options, max_new_tokens=32)
    else:
        raise ValueError(f"unknown strategy={strategy!r}; expected 'ppl' or 'generate'")

    ok = bool(ans.ok) and ans.pred_idx is not None
    correct = bool(ok and item.answer_idx is not None and int(ans.pred_idx) == int(item.answer_idx))
    return {
        "ok": True,
        "question_id": str(item.question_id),
        "video_id": str(item.video_id),
        "youtube_id": str(item.youtube_id),
        "duration_bucket": str(item.duration),
        "domain": str(item.domain),
        "sub_category": str(item.sub_category),
        "task_type": str(item.task_type),
        "method": str(m),
        "budget_frames": int(row_budget),
        "max_seconds": int(max_seconds),
        "duration_seconds": int(dur),
        "selected": sel.to_jsonable(),
        "scores_debug": None if scores_debug is None else scores_debug.get("details"),
        "pred_idx": None if ans.pred_idx is None else int(ans.pred_idx),
        "answer_idx": None if item.answer_idx is None else int(item.answer_idx),
        "correct": bool(correct),
        "invalid": bool(not ok),
        "timings": {"stage1_s": float(stage1_s), **ans.timings},
        "raw_text": str(ans.raw_text),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Video-MME VLM evaluation under budgeted frame selection (controlled transfer).")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--limit", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--order", type=str, default="hash", choices=["hash", "original"])
    p.add_argument("--methods", type=str, required=True, help="Comma-separated methods (uniform is auto-added if missing).")
    p.add_argument("--budget-frames", type=int, default=16)
    p.add_argument("--max-seconds", type=int, default=180)
    p.add_argument("--strategy", type=str, default="ppl", choices=["ppl", "generate"])
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--allow-missing-videos", action="store_true")
    p.add_argument("--min-items", type=int, default=128)

    p.add_argument("--model-name", type=str, default=QwenVLConfig.model_name)
    p.add_argument("--device", type=str, default=QwenVLConfig.device)
    p.add_argument("--dtype", type=str, default=QwenVLConfig.dtype)
    p.add_argument("--attn-implementation", type=str, default=None)

    p.add_argument("--ql2l-clap-device", type=str, default="cpu")
    p.add_argument("--ql2l-asr-device", type=str, default="cpu")
    p.add_argument("--ql2l-clip-device", type=str, default="cpu")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = load_videomme_split(split=str(args.split), limit=int(args.limit), seed=int(args.seed), order=str(args.order))
    methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]
    if not methods:
        raise SystemExit("empty --methods")
    if "uniform" not in methods:
        methods = ["uniform"] + methods

    # Preprocess required videos once (YouTube-backed; allow missing/unavailable videos if requested).
    p = videomme_paths()
    vids = sorted({it.youtube_id for it in items})
    pre_meta: dict[str, dict] = {}
    ok_vids: set[str] = set()
    skipped_vids: list[dict] = []
    t_pre = time.time()
    for i, vid in enumerate(vids):
        video_path = p.raw_video_path(vid)
        if not video_path.exists():
            if bool(args.allow_missing_videos):
                rec = {"youtube_id": str(vid), "reason": "missing_video", "video_path": str(video_path)}
                skipped_vids.append(rec)
                pre_meta[str(vid)] = {"ok": False, **rec}
                continue
            raise FileNotFoundError(f"missing VideoMME raw video: {video_path}")

        try:
            meta = ensure_processed_fps1(
                video_path=video_path,
                out_dir=p.processed_video_dir(vid),
                sample_rate=16000,
                start_offset_sec=0.5,
                max_seconds=int(args.max_seconds),
                force=False,
            )
            pre_meta[str(vid)] = meta
            ok_vids.add(str(vid))
        except Exception as e:  # noqa: BLE001
            if not bool(args.allow_missing_videos):
                raise
            rec = {"youtube_id": str(vid), "reason": "preprocess_failed", "video_path": str(video_path), "error": repr(e)}
            skipped_vids.append(rec)
            pre_meta[str(vid)] = {"ok": False, **rec}

        if (i + 1) % 20 == 0 or (i + 1) == len(vids):
            print(f"[videomme] preprocessed {i+1}/{len(vids)} videos", flush=True)
    pre_s = float(time.time() - t_pre)

    if bool(args.allow_missing_videos):
        items_before = int(len(items))
        items = [it for it in items if str(it.youtube_id) in ok_vids]
        if len(items) < int(args.min_items):
            raise SystemExit(
                f"too few items after filtering missing/corrupted videos: kept={len(items)} "
                f"< min_items={int(args.min_items)} (before={items_before}, skipped_videos={len(skipped_vids)})"
            )

    (out_dir / "preprocess_meta.json").write_text(json.dumps(pre_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    model = QwenVL(
        QwenVLConfig(
            model_name=str(args.model_name),
            device=str(args.device),
            dtype=str(args.dtype),
            attn_implementation=str(args.attn_implementation) if args.attn_implementation else None,
        )
    )

    correct_by_method: dict[str, list[float]] = {m: [] for m in methods}
    invalid_by_method: dict[str, list[float]] = {m: [] for m in methods}
    t0 = time.time()
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for j, it in enumerate(items):
            proc_dir = p.processed_video_dir(it.youtube_id)
            h = hashlib.sha1(f"{it.question_id}|{it.youtube_id}".encode("utf-8")).hexdigest()
            item_seed = int(args.seed) + int(h[:8], 16)
            for m in methods:
                row = _eval_one(
                    model=model,
                    item=it,
                    processed_dir=proc_dir,
                    method=str(m),
                    budget_frames=int(args.budget_frames),
                    seed=int(item_seed),
                    max_seconds=int(args.max_seconds),
                    strategy=str(args.strategy),
                    ql2l_clap_device=str(args.ql2l_clap_device),
                    ql2l_asr_device=str(args.ql2l_asr_device),
                    ql2l_clip_device=str(args.ql2l_clip_device),
                )
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                correct_by_method[str(m)].append(1.0 if row["correct"] else 0.0)
                invalid_by_method[str(m)].append(1.0 if row["invalid"] else 0.0)
            if (j + 1) % 10 == 0 or (j + 1) == len(items):
                f.flush()
                dt = time.time() - t0
                print(f"[videomme] eval {j+1}/{len(items)} items ({dt:.1f}s)", flush=True)
    eval_s = float(time.time() - t0)

    summaries: list[MethodSummary] = []
    acc_by_method: dict[str, np.ndarray] = {}
    for m in methods:
        correct = np.asarray(correct_by_method[str(m)], dtype=np.float64)
        invalid = np.asarray(invalid_by_method[str(m)], dtype=np.float64)
        acc_by_method[str(m)] = correct
        summaries.append(
            MethodSummary(
                method=str(m),
                n=int(correct.size),
                acc=float(correct.mean()) if correct.size > 0 else 0.0,
                invalid_rate=float(invalid.mean()) if invalid.size > 0 else 0.0,
            )
        )

    # Paired deltas vs uniform (item-wise).
    uniform = acc_by_method["uniform"]
    deltas: dict[str, dict] = {}
    for m in methods:
        if m == "uniform":
            continue
        diff = acc_by_method[str(m)] - uniform
        deltas[str(m)] = _bootstrap_ci_mean(diff, seed=int(args.seed), n_boot=300, alpha=0.05)

    payload = {
        "ok": True,
        "task": "VideoMME",
        "split": str(args.split),
        "n_items": int(len(items)),
        "methods": methods,
        "budget_frames": int(args.budget_frames),
        "max_seconds": int(args.max_seconds),
        "seed": int(args.seed),
        "strategy": str(args.strategy),
        "model": {"model_name": str(args.model_name), "device": str(args.device), "dtype": str(args.dtype)},
        "subset": {"order": str(args.order), "limit": int(args.limit)},
        "timings_s": {"preprocess": float(pre_s), "eval": float(eval_s), "total": float(pre_s + eval_s)},
        "summary": [s.__dict__ for s in summaries],
        "delta_vs_uniform": deltas,
        "allow_missing_videos": bool(args.allow_missing_videos),
        "skipped_videos": skipped_vids,
        "artifacts": {
            "predictions_jsonl": str(out_dir / "predictions.jsonl"),
            "metrics_json": str(out_dir / "metrics.json"),
            "preprocess_meta_json": str(out_dir / "preprocess_meta.json"),
        },
    }

    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out_dir / "metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
