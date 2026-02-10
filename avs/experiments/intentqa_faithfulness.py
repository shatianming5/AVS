from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from avs.datasets.intentqa import load_intentqa_split
from avs.datasets.layout import intentqa_paths
from avs.pipeline.qa_plan_generation import build_scores, select_seconds_alpha_mixture, uniform_seconds
from avs.preprocess.video_extract import ensure_processed_fps1
from avs.vlm.qwen_vl import QwenVL, QwenVLConfig


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


def _build_deleted_seconds(*, selected: list[int], background: list[int], duration_seconds: int, budget_frames: int) -> list[int]:
    """
    Delete the anchor-selected seconds by replacing them with a uniform schedule, keeping compute constant.
    """
    dur = int(duration_seconds)
    target = int(budget_frames)
    bg = [int(x) for x in background]
    sel_set = set(int(x) for x in selected)
    bg_set = set(int(x) for x in bg)

    # Candidate replacements from a uniform schedule, excluding already-kept background.
    uni = uniform_seconds(dur, target)
    repl = [int(t) for t in uni if int(t) not in bg_set]

    out: list[int] = []
    out_set: set[int] = set()
    for t in bg:
        if t not in out_set:
            out.append(int(t))
            out_set.add(int(t))
    for t in repl:
        if len(out) >= target:
            break
        if int(t) in out_set:
            continue
        out.append(int(t))
        out_set.add(int(t))

    # If still short (rare), fill with earliest remaining seconds.
    for t in range(dur):
        if len(out) >= target:
            break
        if int(t) in out_set:
            continue
        out.append(int(t))
        out_set.add(int(t))

    out = sorted(out[:target])
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="IntentQA delete-and-predict faithfulness proxy (budget-matched).")
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--limit", type=int, default=64)
    p.add_argument("--method", type=str, default="ql2l_clap")
    p.add_argument("--budget-frames", type=int, default=16)
    p.add_argument("--max-seconds", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--strategy", type=str, default="ppl", choices=["ppl", "generate"])
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--allow-missing-videos",
        action="store_true",
        help="If set, skip missing/corrupted videos during preprocessing and filter out affected items.",
    )
    p.add_argument(
        "--min-items",
        type=int,
        default=16,
        help="Minimum number of items required after filtering (only used when --allow-missing-videos is set).",
    )

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

    items = load_intentqa_split(str(args.split))
    if args.limit is not None and int(args.limit) > 0:
        items = items[: int(args.limit)]

    if not items:
        raise SystemExit("no items to evaluate (check --split/--limit)")

    p = intentqa_paths()
    vids = sorted({it.video_id for it in items})

    # Preprocess videos once (with an optional skip-on-failure policy for real-world corruptions).
    pre_meta: dict[str, dict] = {}
    ok_vids: set[str] = set()
    skipped_vids: list[dict] = []
    t_pre = time.time()
    for i, vid in enumerate(vids):
        video_path = p.raw_video_path(vid)
        if not video_path.exists():
            if bool(args.allow_missing_videos):
                rec = {"video_id": str(vid), "reason": "missing_video", "video_path": str(video_path)}
                skipped_vids.append(rec)
                pre_meta[str(vid)] = {"ok": False, **rec}
                continue
            raise FileNotFoundError(f"missing IntentQA video: {video_path}")

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
            rec = {"video_id": str(vid), "reason": "preprocess_failed", "video_path": str(video_path), "error": repr(e)}
            skipped_vids.append(rec)
            pre_meta[str(vid)] = {"ok": False, **rec}

        if (i + 1) % 50 == 0 or (i + 1) == len(vids):
            print(f"[faithfulness] preprocessed {i+1}/{len(vids)} videos", flush=True)
    pre_s = float(time.time() - t_pre)

    if bool(args.allow_missing_videos):
        items_before = int(len(items))
        items = [it for it in items if str(it.video_id) in ok_vids]
        if len(items) < int(args.min_items):
            raise SystemExit(
                f"too few items after filtering missing/corrupted videos: kept={len(items)} "
                f"< min_items={int(args.min_items)} (before={items_before}, skipped_videos={len(skipped_vids)})"
            )

    # Persist preprocessing metadata early so partial runs still produce debuggable artifacts.
    (out_dir / "preprocess_meta.json").write_text(json.dumps(pre_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    model = QwenVL(
        QwenVLConfig(
            model_name=str(args.model_name),
            device=str(args.device),
            dtype=str(args.dtype),
            attn_implementation=str(args.attn_implementation) if args.attn_implementation else None,
        )
    )

    # Stream rows as we go to make long runs more failure-tolerant.
    n_rows = 0
    sum_correct = 0
    sum_correct_del = 0
    sum_invalid = 0
    sum_invalid_del = 0
    sum_changed = 0

    t0 = time.time()
    with (out_dir / "rows.jsonl").open("w", encoding="utf-8") as f:
        for j, it in enumerate(items):
            proc_dir = p.processed_video_dir(it.video_id)
            dur = min(int(_duration_seconds_from_frames(proc_dir / "frames")), int(args.max_seconds))
            if dur <= 0:
                continue

            h = hashlib.sha1(f"{it.qid}|{it.video_id}".encode("utf-8")).hexdigest()
            item_seed = int(args.seed) + int(h[:8], 16)

            scores_debug = build_scores(
                processed_dir=proc_dir,
                query=it.question,
                num_segments=int(dur),
                method=str(args.method),
                seed=int(item_seed),
                clap_device=str(args.ql2l_clap_device),
                asr_device=str(args.ql2l_asr_device),
            )
            sel = select_seconds_alpha_mixture(scores=scores_debug["scores"], budget_frames=int(args.budget_frames), seed=int(item_seed))
            deleted_seconds = _build_deleted_seconds(
                selected=sel.selected_seconds,
                background=sel.background_seconds,
                duration_seconds=int(dur),
                budget_frames=int(args.budget_frames),
            )

            imgs = _frame_paths(proc_dir, sel.selected_seconds)
            imgs_del = _frame_paths(proc_dir, deleted_seconds)

            if str(args.strategy) == "ppl":
                ans = model.answer_mcq_ppl(image_paths=imgs, question=it.question, options=it.options)
                ans_del = model.answer_mcq_ppl(image_paths=imgs_del, question=it.question, options=it.options)
            else:
                ans = model.answer_mcq_generate(image_paths=imgs, question=it.question, options=it.options, max_new_tokens=32)
                ans_del = model.answer_mcq_generate(image_paths=imgs_del, question=it.question, options=it.options, max_new_tokens=32)

            ok = bool(ans.ok) and ans.pred_idx is not None
            ok_del = bool(ans_del.ok) and ans_del.pred_idx is not None
            correct = bool(ok and int(ans.pred_idx) == int(it.answer_idx))
            correct_del = bool(ok_del and int(ans_del.pred_idx) == int(it.answer_idx))
            pred_changed = bool(ok and ok_del and int(ans.pred_idx) != int(ans_del.pred_idx))

            row = {
                "qid": str(it.qid),
                "video_id": str(it.video_id),
                "method": str(args.method),
                "budget_frames": int(args.budget_frames),
                "duration_seconds": int(dur),
                "selected_seconds": [int(x) for x in sel.selected_seconds],
                "anchor_seconds": [int(x) for x in sel.anchor_seconds],
                "background_seconds": [int(x) for x in sel.background_seconds],
                "deleted_seconds": [int(x) for x in deleted_seconds],
                "pred_idx": None if ans.pred_idx is None else int(ans.pred_idx),
                "pred_idx_deleted": None if ans_del.pred_idx is None else int(ans_del.pred_idx),
                "answer_idx": int(it.answer_idx),
                "correct": bool(correct),
                "correct_deleted": bool(correct_del),
                "invalid": bool(not ok),
                "invalid_deleted": bool(not ok_del),
                "pred_changed": bool(pred_changed),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_rows += 1
            sum_correct += 1 if correct else 0
            sum_correct_del += 1 if correct_del else 0
            sum_invalid += 1 if (not ok) else 0
            sum_invalid_del += 1 if (not ok_del) else 0
            sum_changed += 1 if pred_changed else 0

            if (j + 1) % 10 == 0 or (j + 1) == len(items):
                f.flush()
                dt = time.time() - t0
                print(f"[faithfulness] {j+1}/{len(items)} ({dt:.1f}s)", flush=True)

    eval_s = float(time.time() - t0)

    denom = float(n_rows) if n_rows > 0 else 1.0
    acc = float(sum_correct / denom) if n_rows > 0 else 0.0
    acc_del = float(sum_correct_del / denom) if n_rows > 0 else 0.0
    payload = {
        "ok": True,
        "task": "IntentQA_faithfulness",
        "split": str(args.split),
        "method": str(args.method),
        "n_items": int(n_rows),
        "budget_frames": int(args.budget_frames),
        "max_seconds": int(args.max_seconds),
        "seed": int(args.seed),
        "strategy": str(args.strategy),
        "model": {"model_name": str(args.model_name), "device": str(args.device), "dtype": str(args.dtype)},
        "timings_s": {"preprocess": float(pre_s), "eval": float(eval_s), "total": float(pre_s + eval_s)},
        "accuracy": float(acc),
        "accuracy_deleted": float(acc_del),
        "acc_drop": float(acc - acc_del),
        "invalid_rate": float(sum_invalid / denom) if n_rows > 0 else 0.0,
        "invalid_rate_deleted": float(sum_invalid_del / denom) if n_rows > 0 else 0.0,
        "pred_change_rate": float(sum_changed / denom) if n_rows > 0 else 0.0,
        "allow_missing_videos": bool(args.allow_missing_videos),
        "skipped_videos": skipped_vids,
        "artifacts": {
            "rows_jsonl": str(out_dir / "rows.jsonl"),
            "faithfulness_json": str(out_dir / "faithfulness.json"),
            "preprocess_meta_json": str(out_dir / "preprocess_meta.json"),
        },
    }

    # Write final metrics (rows + preprocess_meta are already streamed/persisted above).
    (out_dir / "faithfulness.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out_dir / "faithfulness.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
