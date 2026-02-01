from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

from avs.datasets.ave import AVEIndex, ensure_ave_meta
from avs.datasets.layout import ave_paths
from avs.experiments.ave_p0 import P0Config, run_p0_from_caches
from avs.train.train_loop import TrainConfig


def _read_ids_file(path: Path, limit: int | None) -> list[str]:
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = str(line).strip()
        if not s:
            continue
        ids.append(s)
        if limit is not None and len(ids) >= int(limit):
            break
    return ids


def _split_ids(index: AVEIndex, split: str, limit: int | None) -> list[str]:
    ids = index.splits[str(split)]
    if limit is not None:
        ids = ids[: int(limit)]
    return [index.clips[int(i)].video_id for i in ids]


@dataclass(frozen=True)
class CandidateConfig:
    name: str
    low_res: int
    base_res: int
    high_res: int
    head: str
    temporal_kernel_size: int
    anchor_shift: int
    anchor_std_threshold: float
    anchor_select: str
    anchor_nms_radius: int
    anchor_nms_strong_gap: float
    anchor_window: int
    anchor_smooth_window: int
    anchor_smooth_mode: str
    anchor_base_alloc: str
    anchor_conf_metric: str | None
    anchor_conf_threshold: float | None

    def to_jsonable(self) -> dict:
        return {
            "name": str(self.name),
            "low_res": int(self.low_res),
            "base_res": int(self.base_res),
            "high_res": int(self.high_res),
            "head": str(self.head),
            "temporal_kernel_size": int(self.temporal_kernel_size),
            "anchor_shift": int(self.anchor_shift),
            "anchor_std_threshold": float(self.anchor_std_threshold),
            "anchor_select": str(self.anchor_select),
            "anchor_nms_radius": int(self.anchor_nms_radius),
            "anchor_nms_strong_gap": float(self.anchor_nms_strong_gap),
            "anchor_window": int(self.anchor_window),
            "anchor_smooth_window": int(self.anchor_smooth_window),
            "anchor_smooth_mode": str(self.anchor_smooth_mode),
            "anchor_base_alloc": str(self.anchor_base_alloc),
            "anchor_conf_metric": str(self.anchor_conf_metric) if self.anchor_conf_metric is not None else None,
            "anchor_conf_threshold": float(self.anchor_conf_threshold) if self.anchor_conf_threshold is not None else None,
        }


def _default_candidates() -> list[CandidateConfig]:
    """
    Fixed, compact search space for reproducibility.

    This list should be edited deliberately (not auto-expanded) so results are comparable across runs.
    """
    base = dict(
        low_res=160,
        base_res=224,
        high_res=352,
        head="temporal_conv",
        temporal_kernel_size=3,
        anchor_shift=1,
        anchor_std_threshold=1.0,
        anchor_select="topk",
        anchor_nms_radius=2,
        anchor_nms_strong_gap=0.6,
        anchor_window=3,
        anchor_smooth_window=0,
        anchor_smooth_mode="mean",
        anchor_base_alloc="distance",
        anchor_conf_metric=None,
        anchor_conf_threshold=None,
    )

    out: list[CandidateConfig] = []
    out.append(CandidateConfig(name="base_160_224_352_topk", **base))
    out.append(CandidateConfig(name="base_160_224_352_nmsR2", **{**base, "anchor_select": "nms"}))
    out.append(CandidateConfig(name="base_160_224_352_window3", **{**base, "anchor_select": "window_topk", "anchor_window": 3}))
    out.append(CandidateConfig(name="base_160_224_352_window5", **{**base, "anchor_select": "window_topk", "anchor_window": 5}))
    out.append(CandidateConfig(name="base_160_224_352_scoreAlloc", **{**base, "anchor_base_alloc": "score"}))
    out.append(CandidateConfig(name="base_160_224_352_mixedAlloc", **{**base, "anchor_base_alloc": "mixed"}))
    out.append(CandidateConfig(name="base_160_224_352_k5", **{**base, "temporal_kernel_size": 5}))

    # A more extreme triad that sometimes helps (requires caches with 112/224/448).
    out.append(
        CandidateConfig(
            name="extreme_112_224_448_window3",
            **{
                **base,
                "low_res": 112,
                "high_res": 448,
                "anchor_select": "window_topk",
                "anchor_window": 3,
            },
        )
    )

    # Confidence gating variants (opt-in): use top1-top2 gap on raw scores.
    out.append(
        CandidateConfig(
            name="base_160_224_352_gapGate0.6",
            **{**base, "anchor_conf_metric": "top12_gap", "anchor_conf_threshold": 0.6},
        )
    )
    return out


def _extract_delta_and_p(metrics: dict) -> tuple[float | None, float | None]:
    try:
        summary = metrics["summary"]
        delta = float(summary["anchored_top2"]["mean"]) - float(summary["uniform"]["mean"])
    except Exception:
        delta = None

    try:
        p = float(metrics.get("paired_ttest", {})["anchored_vs_uniform"]["p"])
    except Exception:
        p = None
    return delta, p


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--meta-dir", type=Path, default=ave_paths().meta_dir)
    p.add_argument("--processed-dir", type=Path, default=ave_paths().processed_dir, help="Processed dir containing <clip_id>/audio.wav")
    p.add_argument("--caches-dir", type=Path, required=True, help="Dir containing <clip_id>.npz feature caches")
    p.add_argument("--train-ids-file", type=Path, default=None, help="Optional file with one train video_id per line.")
    p.add_argument("--eval-ids-file", type=Path, default=None, help="Optional file with one eval video_id per line.")
    p.add_argument("--split-train", type=str, default="train", choices=["train", "val", "test"])
    p.add_argument("--split-eval", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-eval", type=int, default=None)
    p.add_argument("--allow-missing", action="store_true", help="Skip clips with missing caches instead of failing.")

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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fixed-space AVE config sweep (val selection → test reproduction).")
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("sweep", help="Run fixed candidate configs on split-eval and write sweep_summary.json + best_config.json.")
    _build_common_args(ps)
    ps.add_argument("--out-dir", type=Path, default=Path("runs") / f"E0011_ave_p0_sweep_{time.strftime('%Y%m%d-%H%M%S')}")
    ps.add_argument("--p-filter", type=float, default=0.1, help="Pre-filter configs by p-value before selecting the best. 0 disables.")

    pr = sub.add_parser("run", help="Run one config from best_config.json on a given eval ids file/split (e.g., test402).")
    _build_common_args(pr)
    pr.add_argument("--config-json", type=Path, required=True, help="Path to best_config.json from the sweep step.")
    pr.add_argument("--out-dir", type=Path, default=Path("runs") / f"E0012_ave_p0_best_{time.strftime('%Y%m%d-%H%M%S')}")
    return p


def _load_or_select_ids(
    *,
    index: AVEIndex,
    ids_file: Path | None,
    split: str,
    limit: int | None,
) -> list[str]:
    if ids_file is not None:
        return _read_ids_file(ids_file, limit)
    return _split_ids(index, split, limit)


def _filter_missing(*, ids: list[str], caches_dir: Path) -> list[str]:
    cached = {p.stem for p in caches_dir.glob("*.npz")}
    return [cid for cid in ids if cid in cached]


def _labels_for_ids(index: AVEIndex, ids: list[str]) -> dict[str, list[int]]:
    clip_by_id = {c.video_id: c for c in index.clips}
    out: dict[str, list[int]] = {}
    for cid in ids:
        clip = clip_by_id.get(cid)
        if clip is None:
            continue
        out[cid] = [int(x) for x in index.segment_labels(clip)]
    return out


def _run_one(
    *,
    index: AVEIndex,
    train_ids: list[str],
    eval_ids: list[str],
    labels_by_clip: dict[str, list[int]],
    caches_dir: Path,
    processed_dir: Path,
    seeds: list[int],
    train_cfg: TrainConfig,
    train_device: str,
    eventness_method: str,
    audio_device: str,
    ast_pretrained: bool,
    panns_random: bool,
    panns_checkpoint: Path | None,
    audiomae_random: bool,
    audiomae_checkpoint: Path | None,
    cand: CandidateConfig,
) -> dict:
    cfg = P0Config(
        k=2,
        low_res=int(cand.low_res),
        base_res=int(cand.base_res),
        high_res=int(cand.high_res),
        patch_size=16,
        anchor_shift=int(cand.anchor_shift),
        anchor_std_threshold=float(cand.anchor_std_threshold),
        anchor_select=str(cand.anchor_select),
        anchor_nms_radius=int(cand.anchor_nms_radius),
        anchor_nms_strong_gap=float(cand.anchor_nms_strong_gap),
        anchor_window=int(cand.anchor_window),
        anchor_smooth_window=int(cand.anchor_smooth_window),
        anchor_smooth_mode=str(cand.anchor_smooth_mode),
        anchor_conf_metric=str(cand.anchor_conf_metric) if cand.anchor_conf_metric is not None else None,
        anchor_conf_threshold=float(cand.anchor_conf_threshold) if cand.anchor_conf_threshold is not None else None,
        anchor_base_alloc=str(cand.anchor_base_alloc),
        head=str(cand.head),
        temporal_kernel_size=int(cand.temporal_kernel_size),
    )

    metrics = run_p0_from_caches(
        clip_ids_train=train_ids,
        clip_ids_eval=eval_ids,
        labels_by_clip=labels_by_clip,
        caches_dir=caches_dir,
        audio_dir=processed_dir,
        cfg=cfg,
        baselines=["uniform", "random_top2", "anchored_top2"],
        seeds=seeds,
        train_cfg=train_cfg,
        train_device=str(train_device),
        num_classes=index.num_classes,
        num_segments=10,
        eventness_method=str(eventness_method),
        audio_device=str(audio_device),
        ast_pretrained=bool(ast_pretrained),
        panns_random=bool(panns_random),
        panns_checkpoint=panns_checkpoint,
        audiomae_random=bool(audiomae_random),
        audiomae_checkpoint=audiomae_checkpoint,
    )
    return metrics


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    ensure_ave_meta(args.meta_dir)
    index = AVEIndex.from_meta_dir(args.meta_dir)

    train_ids = _load_or_select_ids(index=index, ids_file=args.train_ids_file, split=args.split_train, limit=args.limit_train)
    eval_ids = _load_or_select_ids(index=index, ids_file=args.eval_ids_file, split=args.split_eval, limit=args.limit_eval)

    caches_dir = Path(args.caches_dir)
    if args.allow_missing:
        train_ids = _filter_missing(ids=train_ids, caches_dir=caches_dir)
        eval_ids = _filter_missing(ids=eval_ids, caches_dir=caches_dir)
    if not train_ids or not eval_ids:
        raise SystemExit(f"no usable ids after filtering (train={len(train_ids)} eval={len(eval_ids)})")

    labels_by_clip = {**_labels_for_ids(index, train_ids), **_labels_for_ids(index, eval_ids)}

    seeds = [int(s) for s in str(args.seeds).split(",") if str(s).strip()]
    if len(seeds) < 2:
        raise SystemExit("--seeds must contain at least 2 seeds to compute paired p-values")

    train_cfg = TrainConfig(
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    if args.cmd == "sweep":
        out_dir: Path = args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        candidates = _default_candidates()
        results: list[dict] = []
        for cand in candidates:
            run_dir = out_dir / cand.name
            run_dir.mkdir(parents=True, exist_ok=True)

            metrics = _run_one(
                index=index,
                train_ids=train_ids,
                eval_ids=eval_ids,
                labels_by_clip=labels_by_clip,
                caches_dir=caches_dir,
                processed_dir=Path(args.processed_dir),
                seeds=seeds,
                train_cfg=train_cfg,
                train_device=str(args.train_device),
                eventness_method=str(args.eventness_method),
                audio_device=str(args.audio_device),
                ast_pretrained=bool(args.ast_pretrained),
                panns_random=bool(args.panns_random),
                panns_checkpoint=args.panns_checkpoint,
                audiomae_random=bool(args.audiomae_random),
                audiomae_checkpoint=args.audiomae_checkpoint,
                cand=cand,
            )

            metrics_path = run_dir / "metrics.json"
            _write_json(metrics_path, metrics)

            delta, pval = _extract_delta_and_p(metrics)
            results.append(
                {
                    "candidate": cand.to_jsonable(),
                    "metrics_path": str(metrics_path),
                    "anchored_minus_uniform_mean": delta,
                    "anchored_vs_uniform_p": pval,
                }
            )

        ordered = sorted(results, key=lambda r: float(r["anchored_minus_uniform_mean"] or float("-inf")), reverse=True)

        # Optional p-value pre-filter (usually for val selection).
        p_filter = float(args.p_filter)
        filtered = ordered
        if p_filter > 0.0:
            filtered = [r for r in ordered if r["anchored_vs_uniform_p"] is not None and float(r["anchored_vs_uniform_p"]) < p_filter]
            if not filtered:
                filtered = ordered

        best = filtered[0]
        best_config = best["candidate"]
        _write_json(out_dir / "best_config.json", best_config)

        summary = {
            "ok": True,
            "meta_dir": str(args.meta_dir),
            "processed_dir": str(args.processed_dir),
            "caches_dir": str(args.caches_dir),
            "split_train": str(args.split_train),
            "split_eval": str(args.split_eval),
            "num_train_ids": int(len(train_ids)),
            "num_eval_ids": int(len(eval_ids)),
            "seeds": seeds,
            "train_cfg": {"epochs": train_cfg.epochs, "batch_size": train_cfg.batch_size, "lr": train_cfg.lr, "weight_decay": train_cfg.weight_decay},
            "eventness_method": str(args.eventness_method),
            "audio_device": str(args.audio_device),
            "train_device": str(args.train_device),
            "p_filter": p_filter,
            "candidates": results,
            "best": best,
            "top3": ordered[:3],
        }
        _write_json(out_dir / "sweep_summary.json", summary)
        print(out_dir / "sweep_summary.json")
        return 0

    # args.cmd == "run"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cand_obj = json.loads(Path(args.config_json).read_text(encoding="utf-8"))
    cand = CandidateConfig(
        name=str(cand_obj.get("name", "best_config")),
        low_res=int(cand_obj["low_res"]),
        base_res=int(cand_obj["base_res"]),
        high_res=int(cand_obj["high_res"]),
        head=str(cand_obj.get("head", "temporal_conv")),
        temporal_kernel_size=int(cand_obj.get("temporal_kernel_size", 3)),
        anchor_shift=int(cand_obj.get("anchor_shift", 0)),
        anchor_std_threshold=float(cand_obj.get("anchor_std_threshold", 0.0)),
        anchor_select=str(cand_obj.get("anchor_select", "topk")),
        anchor_nms_radius=int(cand_obj.get("anchor_nms_radius", 1)),
        anchor_nms_strong_gap=float(cand_obj.get("anchor_nms_strong_gap", 0.6)),
        anchor_window=int(cand_obj.get("anchor_window", 3)),
        anchor_smooth_window=int(cand_obj.get("anchor_smooth_window", 0)),
        anchor_smooth_mode=str(cand_obj.get("anchor_smooth_mode", "mean")),
        anchor_base_alloc=str(cand_obj.get("anchor_base_alloc", "distance")),
        anchor_conf_metric=str(cand_obj["anchor_conf_metric"]) if cand_obj.get("anchor_conf_metric") is not None else None,
        anchor_conf_threshold=float(cand_obj["anchor_conf_threshold"]) if cand_obj.get("anchor_conf_threshold") is not None else None,
    )

    metrics = _run_one(
        index=index,
        train_ids=train_ids,
        eval_ids=eval_ids,
        labels_by_clip=labels_by_clip,
        caches_dir=caches_dir,
        processed_dir=Path(args.processed_dir),
        seeds=seeds,
        train_cfg=train_cfg,
        train_device=str(args.train_device),
        eventness_method=str(args.eventness_method),
        audio_device=str(args.audio_device),
        ast_pretrained=bool(args.ast_pretrained),
        panns_random=bool(args.panns_random),
        panns_checkpoint=args.panns_checkpoint,
        audiomae_random=bool(args.audiomae_random),
        audiomae_checkpoint=args.audiomae_checkpoint,
        cand=cand,
    )
    _write_json(out_dir / "metrics.json", metrics)
    print(out_dir / "metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

