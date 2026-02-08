from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from avs.datasets.egoschema import EgoSchemaItem, load_egoschema_split
from avs.datasets.layout import egoschema_paths
from avs.pipeline.qa_plan_generation import QASecSelection, build_scores, random_seconds, select_seconds_alpha_mixture, uniform_seconds
from avs.preprocess.video_extract import ensure_processed_fps1
from avs.vlm.qwen_vl import QwenVL, QwenVLConfig


@dataclass(frozen=True)
class MethodSummary:
    method: str
    n: int
    acc: float | None
    invalid_rate: float


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
    item: EgoSchemaItem,
    processed_dir: Path,
    method: str,
    budget_frames: int,
    seed: int,
    max_seconds: int,
    strategy: str,
    ql2l_clap_device: str,
    ql2l_asr_device: str,
) -> dict:
    dur = min(int(_duration_seconds_from_frames(processed_dir / "frames")), int(max_seconds))
    if dur <= 0:
        raise ValueError(f"empty processed frames for video_idx={item.video_idx}")

    t0 = time.time()
    sel: QASecSelection | None = None
    scores_debug: dict | None = None

    m = str(method)
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
    else:
        scores_debug = build_scores(
            processed_dir=processed_dir,
            query=item.question,
            num_segments=int(dur),
            method=m,
            seed=int(seed),
            clap_device=str(ql2l_clap_device),
            asr_device=str(ql2l_asr_device),
        )
        scores = scores_debug["scores"]
        sel = select_seconds_alpha_mixture(scores=scores, budget_frames=int(budget_frames), seed=int(seed))

    stage1_s = float(time.time() - t0)
    img_paths = _frame_paths(processed_dir, sel.selected_seconds)

    if str(strategy) == "ppl":
        ans = model.answer_mcq_ppl(image_paths=img_paths, question=item.question, options=item.options)
    elif str(strategy) == "generate":
        ans = model.answer_mcq_generate(image_paths=img_paths, question=item.question, options=item.options, max_new_tokens=32)
    else:
        raise ValueError(f"unknown strategy={strategy!r}; expected 'ppl' or 'generate'")

    ok = bool(ans.ok) and ans.pred_idx is not None
    correct = None
    if item.answer_idx is not None:
        correct = bool(ok and int(ans.pred_idx) == int(item.answer_idx))

    return {
        "ok": True,
        "question_idx": str(item.question_idx),
        "video_idx": str(item.video_idx),
        "method": str(m),
        "budget_frames": int(budget_frames),
        "duration_seconds": int(dur),
        "selected": sel.to_jsonable(),
        "scores_debug": None if scores_debug is None else scores_debug["details"],
        "pred_idx": None if ans.pred_idx is None else int(ans.pred_idx),
        "answer_idx": None if item.answer_idx is None else int(item.answer_idx),
        "correct": correct,
        "invalid": bool(not ok),
        "timings": {"stage1_s": float(stage1_s), **ans.timings},
        "raw_text": str(ans.raw_text),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EgoSchema VLM eval / prediction generation under budgeted frame selection.")
    p.add_argument("--config", type=str, default="MC", choices=["MC", "MC_PPL", "Subset", "GENERATION"])
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--limit", type=int, default=64)
    p.add_argument("--methods", type=str, default="uniform,ql2l_clap,ql2l_asr_bm25", help="Comma-separated methods")
    p.add_argument("--budget-frames", type=int, default=16)
    p.add_argument("--max-seconds", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--strategy", type=str, default="ppl", choices=["ppl", "generate"])
    p.add_argument("--out-dir", type=Path, required=True)

    p.add_argument("--model-name", type=str, default=QwenVLConfig.model_name)
    p.add_argument("--device", type=str, default=QwenVLConfig.device)
    p.add_argument("--dtype", type=str, default=QwenVLConfig.dtype)
    p.add_argument("--attn-implementation", type=str, default=None)

    p.add_argument("--ql2l-clap-device", type=str, default="cpu")
    p.add_argument("--ql2l-asr-device", type=str, default="cpu")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = load_egoschema_split(config=str(args.config), split=str(args.split), limit=int(args.limit) if args.limit else None)
    methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]
    if not methods:
        raise SystemExit("empty --methods")
    if "uniform" not in methods:
        methods = ["uniform"] + methods

    p = egoschema_paths()
    if not p.videos_dir.exists():
        raise FileNotFoundError(f"missing extracted EgoSchema videos dir: {p.videos_dir} (run scripts/datasets/egoschema_extract_videos.sh)")

    # Preprocess only the videos we need (bounded by --limit).
    vids = sorted({it.video_idx for it in items})
    pre_meta: dict[str, dict] = {}
    t_pre = time.time()
    for i, vid in enumerate(vids):
        video_path = p.video_path(vid)
        if not video_path.exists():
            raise FileNotFoundError(f"missing EgoSchema video: {video_path}")
        meta = ensure_processed_fps1(
            video_path=video_path,
            out_dir=p.processed_video_dir(vid),
            sample_rate=16000,
            start_offset_sec=0.5,
            max_seconds=int(args.max_seconds),
            force=False,
        )
        pre_meta[str(vid)] = meta
        if (i + 1) % 50 == 0 or (i + 1) == len(vids):
            print(f"[egoschema] preprocessed {i+1}/{len(vids)} videos", flush=True)
    pre_s = float(time.time() - t_pre)

    model = QwenVL(
        QwenVLConfig(
            model_name=str(args.model_name),
            device=str(args.device),
            dtype=str(args.dtype),
            attn_implementation=str(args.attn_implementation) if args.attn_implementation else None,
        )
    )

    per_method: dict[str, list[dict]] = {m: [] for m in methods}
    t0 = time.time()
    for j, it in enumerate(items):
        proc_dir = p.processed_video_dir(it.video_idx)
        h = hashlib.sha1(f"{it.question_idx}|{it.video_idx}".encode("utf-8")).hexdigest()
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
            )
            per_method[str(m)].append(row)
        if (j + 1) % 10 == 0 or (j + 1) == len(items):
            dt = time.time() - t0
            print(f"[egoschema] eval {j+1}/{len(items)} items ({dt:.1f}s)", flush=True)

    eval_s = float(time.time() - t0)

    summaries: list[MethodSummary] = []
    for m in methods:
        rows = per_method[str(m)]
        invalid = np.asarray([1.0 if r["invalid"] else 0.0 for r in rows], dtype=np.float64)
        acc = None
        if all(r["answer_idx"] is not None for r in rows):
            correct = np.asarray([1.0 if r["correct"] else 0.0 for r in rows], dtype=np.float64)
            acc = float(correct.mean() if correct.size else 0.0)
        summaries.append(MethodSummary(method=str(m), n=int(len(rows)), acc=acc, invalid_rate=float(invalid.mean() if invalid.size else 0.0)))

    payload = {
        "ok": True,
        "task": "EgoSchema",
        "config": str(args.config),
        "split": str(args.split),
        "n_items": int(len(items)),
        "methods": methods,
        "budget_frames": int(args.budget_frames),
        "max_seconds": int(args.max_seconds),
        "seed": int(args.seed),
        "strategy": str(args.strategy),
        "model": {"model_name": str(args.model_name), "device": str(args.device), "dtype": str(args.dtype)},
        "timings_s": {"preprocess": float(pre_s), "eval": float(eval_s), "total": float(pre_s + eval_s)},
        "summary": [s.__dict__ for s in summaries],
        "artifacts": {
            "predictions_jsonl": str(out_dir / "predictions.jsonl"),
            "metrics_json": str(out_dir / "metrics.json"),
            "preprocess_meta_json": str(out_dir / "preprocess_meta.json"),
        },
    }

    (out_dir / "preprocess_meta.json").write_text(json.dumps(pre_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for m in methods:
            for r in per_method[str(m)]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out_dir / "metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
