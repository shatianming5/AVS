from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from avs.datasets.ave import AVEIndex, ensure_ave_meta
from avs.datasets.layout import ave_paths
from avs.experiments.ave_p0 import P0Config, run_p0_from_caches
from avs.train.train_loop import TrainConfig


def _load_scores_json(path: Path) -> dict[str, list[float]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict) and "scores" in obj and isinstance(obj["scores"], dict):
        scores_obj = obj["scores"]
    elif isinstance(obj, dict):
        scores_obj = obj
    else:
        raise ValueError("scores-json must be a JSON object")

    out: dict[str, list[float]] = {}
    for k, v in scores_obj.items():
        if not isinstance(v, list):
            raise ValueError(f"scores[{k!r}] must be a list, got {type(v)}")
        out[str(k)] = [float(x) for x in v]
    return out


def _read_ids_file(path: Path, limit: int | None) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        s = str(line).strip()
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        ids.append(s)
        if limit is not None and len(ids) >= int(limit):
            break
    return ids


def _split_ids(index: AVEIndex, split: str, limit: int | None) -> list[str]:
    ids = index.splits[str(split)]
    if limit is not None:
        ids = ids[: int(limit)]
    return [index.clips[int(i)].video_id for i in ids]


def _filter_missing(*, ids: list[str], caches_dir: Path, processed_dir: Path) -> list[str]:
    cached = {p.stem for p in caches_dir.glob("*.npz")}
    out: list[str] = []
    for cid in ids:
        if cid not in cached:
            continue
        if not (processed_dir / cid / "audio.wav").exists():
            continue
        out.append(cid)
    return out


def _labels_for_ids(index: AVEIndex, ids: list[str]) -> dict[str, list[int]]:
    clip_by_id = {c.video_id: c for c in index.clips}
    out: dict[str, list[int]] = {}
    for cid in ids:
        clip = clip_by_id.get(cid)
        if clip is None:
            continue
        out[cid] = [int(x) for x in index.segment_labels(clip)]
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Confirm fusion adds on top of sampling (audio_concat_* baselines).")
    p.add_argument("--meta-dir", type=Path, default=ave_paths().meta_dir)
    p.add_argument("--processed-dir", type=Path, default=ave_paths().processed_dir, help="Processed dir containing <clip_id>/audio.wav")
    p.add_argument("--caches-dir", type=Path, required=True, help="Dir containing <clip_id>.npz feature caches")
    p.add_argument("--config-json", type=Path, required=True, help="Path to best_config.json from E0011 sweep.")
    p.add_argument("--train-ids-file", type=Path, default=None, help="Optional file with one train video_id per line.")
    p.add_argument("--eval-ids-file", type=Path, default=None, help="Optional file with one eval video_id per line.")
    p.add_argument("--split-train", type=str, default="train", choices=["train", "val", "test"])
    p.add_argument("--split-eval", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-eval", type=int, default=None)
    p.add_argument("--allow-missing", action="store_true", help="Skip clips with missing caches/audio instead of failing.")

    p.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--train-device", type=str, default="cuda:0")

    p.add_argument("--eventness-method", type=str, default="energy")
    p.add_argument("--audio-device", type=str, default="cuda:0")
    p.add_argument("--ast-pretrained", action="store_true")
    p.add_argument("--panns-checkpoint", type=Path, default=None)
    p.add_argument("--panns-random", action="store_true")
    p.add_argument("--audiomae-checkpoint", type=Path, default=None)
    p.add_argument("--audiomae-random", action="store_true")
    p.add_argument(
        "--scores-json",
        type=Path,
        default=None,
        help="Optional JSON cache of per-second eventness scores keyed by clip_id (from E0014). "
        "If provided, skips recomputing scores for expensive audio backends.",
    )

    p.add_argument("--out-dir", type=Path, default=Path("runs") / f"E0013_ave_fusion_confirm_{time.strftime('%Y%m%d-%H%M%S')}")
    return p


def _load_config_json(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("config-json must be a JSON object")
    return obj


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ensure_ave_meta(args.meta_dir)
    index = AVEIndex.from_meta_dir(args.meta_dir)

    if args.train_ids_file is not None:
        train_ids = _read_ids_file(args.train_ids_file, args.limit_train)
    else:
        train_ids = _split_ids(index, args.split_train, args.limit_train)

    if args.eval_ids_file is not None:
        eval_ids = _read_ids_file(args.eval_ids_file, args.limit_eval)
    else:
        eval_ids = _split_ids(index, args.split_eval, args.limit_eval)

    caches_dir = Path(args.caches_dir)
    processed_dir = Path(args.processed_dir)
    if bool(args.allow_missing):
        train_ids = _filter_missing(ids=train_ids, caches_dir=caches_dir, processed_dir=processed_dir)
        eval_ids = _filter_missing(ids=eval_ids, caches_dir=caches_dir, processed_dir=processed_dir)

    if not train_ids or not eval_ids:
        raise SystemExit(f"no usable ids after filtering (train={len(train_ids)} eval={len(eval_ids)})")

    labels_by_clip = {**_labels_for_ids(index, train_ids), **_labels_for_ids(index, eval_ids)}
    cfg_obj = _load_config_json(args.config_json)

    cfg = P0Config(
        k=int(cfg_obj.get("k", 2)),
        low_res=int(cfg_obj["low_res"]),
        base_res=int(cfg_obj["base_res"]),
        high_res=int(cfg_obj["high_res"]),
        patch_size=16,
        max_high_anchors=int(cfg_obj.get("max_high_anchors")) if cfg_obj.get("max_high_anchors") is not None else None,
        anchor_shift=int(cfg_obj.get("anchor_shift", 0)),
        anchor_std_threshold=float(cfg_obj.get("anchor_std_threshold", 0.0)),
        anchor_select=str(cfg_obj.get("anchor_select", "topk")),
        anchor_nms_radius=int(cfg_obj.get("anchor_nms_radius", 1)),
        anchor_nms_strong_gap=float(cfg_obj.get("anchor_nms_strong_gap", 0.6)),
        anchor_window=int(cfg_obj.get("anchor_window", 3)),
        anchor_smooth_window=int(cfg_obj.get("anchor_smooth_window", 0)),
        anchor_smooth_mode=str(cfg_obj.get("anchor_smooth_mode", "mean")),
        anchor_base_alloc=str(cfg_obj.get("anchor_base_alloc", "distance")),
        anchor_conf_metric=str(cfg_obj.get("anchor_conf_metric")) if cfg_obj.get("anchor_conf_metric") is not None else None,
        anchor_conf_threshold=float(cfg_obj.get("anchor_conf_threshold")) if cfg_obj.get("anchor_conf_threshold") is not None else None,
        anchor_high_policy=str(cfg_obj.get("anchor_high_policy", "fixed")),
        anchor_high_adjacent_dist=int(cfg_obj.get("anchor_high_adjacent_dist", 1)),
        anchor_high_gap_threshold=float(cfg_obj.get("anchor_high_gap_threshold", 0.0)),
        head=str(cfg_obj.get("head", "temporal_conv")),
        temporal_kernel_size=int(cfg_obj.get("temporal_kernel_size", 3)),
    )

    seeds = [int(s) for s in str(args.seeds).split(",") if str(s).strip()]
    if len(seeds) < 2:
        raise SystemExit("--seeds must contain at least 2 seeds to compute paired p-values")

    train_cfg = TrainConfig(
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    scores_by_clip_override = None
    if args.scores_json is not None:
        if not args.scores_json.exists():
            raise SystemExit(f"--scores-json not found: {args.scores_json}")
        scores_by_clip_override = _load_scores_json(args.scores_json)
        all_ids = sorted(set(train_ids + eval_ids))
        missing = [cid for cid in all_ids if cid not in scores_by_clip_override]
        if missing:
            raise SystemExit(f"scores-json missing {len(missing)} clip_ids (e.g. {missing[:3]})")

    baselines = ["uniform", "anchored_top2", "audio_concat_uniform", "audio_concat_anchored_top2"]
    metrics = run_p0_from_caches(
        clip_ids_train=train_ids,
        clip_ids_eval=eval_ids,
        labels_by_clip=labels_by_clip,
        caches_dir=caches_dir,
        audio_dir=processed_dir,
        cfg=cfg,
        baselines=baselines,
        seeds=seeds,
        train_cfg=train_cfg,
        train_device=str(args.train_device),
        num_classes=index.num_classes,
        class_names=[str(index.idx_to_label[i]) for i in range(int(index.num_classes))],
        num_segments=10,
        eventness_method=str(args.eventness_method),
        audio_device=str(args.audio_device),
        ast_pretrained=bool(args.ast_pretrained),
        panns_random=bool(args.panns_random),
        panns_checkpoint=args.panns_checkpoint,
        audiomae_random=bool(args.audiomae_random),
        audiomae_checkpoint=args.audiomae_checkpoint,
        scores_by_clip_override=scores_by_clip_override,
    )

    out_path = args.out_dir / "metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
