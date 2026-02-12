from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

from avs.datasets.ave import AVEIndex, ensure_ave_meta
from avs.datasets.ave_download import materialize_ave_videos
from avs.datasets.layout import ave_paths
from avs.experiments.ave_p0 import P0Config, run_p0_from_caches
from avs.preprocess.ave_dataset import preprocess_ave_videos
from avs.train.train_loop import TrainConfig
from avs.vision.clip_vit import ClipVisionEncoder, ClipVisionEncoderConfig
from avs.vision.feature_cache import build_clip_feature_cache


def _select_split_ids(meta_dir: Path, split: str, limit: int | None) -> list[str]:
    ensure_ave_meta(meta_dir)
    index = AVEIndex.from_meta_dir(meta_dir)
    ids = index.splits[split]
    if limit is not None:
        ids = ids[:limit]
    return [index.clips[int(i)].video_id for i in ids]


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


def _parse_devices_csv(value: str) -> list[str]:
    out = [x.strip() for x in str(value).split(",") if str(x).strip()]
    if not out:
        raise ValueError("empty devices list")
    return out


@dataclass(frozen=True)
class _CacheWorkerResult:
    worker_id: int
    device: str
    num_assigned: int
    num_done: int
    num_skipped: int
    errors: list[dict]
    elapsed_s: float

    def to_jsonable(self) -> dict:
        return {
            "worker_id": int(self.worker_id),
            "device": str(self.device),
            "num_assigned": int(self.num_assigned),
            "num_done": int(self.num_done),
            "num_skipped": int(self.num_skipped),
            "errors": list(self.errors),
            "elapsed_s": float(self.elapsed_s),
            "ok": len(self.errors) == 0,
        }


def _cache_worker(
    *,
    worker_id: int,
    processed_dir: Path,
    caches_dir: Path,
    clip_ids: list[str],
    resolutions: list[int],
    device: str,
    model_name: str,
    pretrained: bool,
    skip_existing: bool,
    out_path: Path,
) -> None:
    import time as _time
    import traceback as _traceback

    start = _time.time()
    errors: list[dict] = []
    num_done = 0
    num_skipped = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        encoder = ClipVisionEncoder(
            ClipVisionEncoderConfig(
                model_name=str(model_name),
                pretrained=bool(pretrained),
                device=str(device),
            )
        )

        for cid in clip_ids:
            cache_path = caches_dir / f"{cid}.npz"
            if skip_existing and cache_path.exists():
                num_skipped += 1
                continue
            try:
                frames_dir = processed_dir / cid / "frames"
                cache = build_clip_feature_cache(frames_dir=frames_dir, resolutions=resolutions, encoder=encoder)
                cache.save_npz(cache_path)
                num_done += 1
            except Exception as e:  # noqa: BLE001 - best-effort worker: record and continue
                errors.append({"clip_id": str(cid), "error": repr(e)})
    except Exception as e:  # noqa: BLE001 - worker init failure
        errors.append({"clip_id": None, "error": repr(e), "traceback": _traceback.format_exc()})

    res = _CacheWorkerResult(
        worker_id=int(worker_id),
        device=str(device),
        num_assigned=int(len(clip_ids)),
        num_done=int(num_done),
        num_skipped=int(num_skipped),
        errors=errors,
        elapsed_s=float(_time.time() - start),
    )
    out_path.write_text(json.dumps(res.to_jsonable(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run AVE-P0 end-to-end (raw→preprocess→cache→metrics) on a subset.")
    p.add_argument("--mode", type=str, default="local", choices=["local", "yt-dlp", "none"])
    p.add_argument("--src-dir", type=Path, default=None, help="Local dir containing <video_id>.mp4 (mode=local)")
    p.add_argument("--ytdlp", type=str, default="yt-dlp", help="yt-dlp binary (mode=yt-dlp)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--allow-missing",
        action="store_true",
        help="Continue even if some videos are missing/unavailable; drops those clips from train/eval.",
    )

    p.add_argument("--meta-dir", type=Path, default=ave_paths().meta_dir)
    p.add_argument("--raw-videos-dir", type=Path, default=ave_paths().raw_videos_dir)
    p.add_argument("--processed-dir", type=Path, default=ave_paths().processed_dir)
    p.add_argument("--preprocess-skip-existing", action="store_true", help="Skip preprocessing for clips already in processed-dir.")
    p.add_argument("--preprocess-jobs", type=int, default=1, help="Number of parallel preprocessing workers (ffmpeg-heavy).")
    p.add_argument("--caches-dir", type=Path, default=None, help="Dir for <clip_id>.npz caches (default: <out-dir>/caches)")

    p.add_argument("--split-train", type=str, default="train", choices=["train", "val", "test"])
    p.add_argument("--split-eval", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--limit-train", type=int, default=8)
    p.add_argument("--limit-eval", type=int, default=4)
    p.add_argument("--train-ids-file", type=Path, default=None, help="Optional file with one train video_id per line.")
    p.add_argument("--eval-ids-file", type=Path, default=None, help="Optional file with one eval video_id per line.")

    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--k", type=int, default=2, help="Top-K audio anchors to use (budget may cap effective K).")
    p.add_argument("--anchor-shift", type=int, default=0, help="Shift anchor indices by this many segments (A/V misalignment).")
    p.add_argument(
        "--anchor-std-threshold",
        type=float,
        default=0.0,
        help="If std(scores) < threshold, fall back to uniform sampling (anchored baseline). 0 disables.",
    )
    p.add_argument(
        "--anchor-select",
        type=str,
        default="topk",
        choices=["topk", "nms", "nms_strong", "adjacent_top2", "window_topk"],
        help="Anchor selection strategy on per-second eventness scores.",
    )
    p.add_argument(
        "--anchor-window",
        type=int,
        default=3,
        help="For --anchor-select window_topk: window size for score aggregation (odd; e.g., 3 or 5).",
    )
    p.add_argument(
        "--anchor-smooth-window",
        type=int,
        default=0,
        help="Optional score smoothing window (odd). Applied before anchor selection. 0 disables.",
    )
    p.add_argument(
        "--anchor-smooth-mode",
        type=str,
        default="mean",
        choices=["mean", "sum"],
        help="For --anchor-smooth-window: how to aggregate scores inside the smoothing window.",
    )
    p.add_argument(
        "--anchor-nms-radius",
        type=int,
        default=1,
        help="For --anchor-select nms: suppress anchors within ±radius segments of a selected anchor. For adjacent_top2: adjacent search radius.",
    )
    p.add_argument(
        "--anchor-nms-strong-gap",
        type=float,
        default=0.6,
        help="For --anchor-select nms_strong: accept a far anchor only if (top1_score - best_far_score) <= gap. For adjacent_top2: pick an adjacent 2nd anchor only if (top1_score - best_adj_score) <= gap.",
    )
    p.add_argument(
        "--anchor-conf-metric",
        type=str,
        default=None,
        choices=["std", "std_norm", "top1_med", "top1_med_norm", "top12_gap", "top12_gap_norm", "gini"],
        help="Anchor confidence metric. If set, uses --anchor-conf-threshold to decide fallback to uniform (replaces std-only fallback).",
    )
    p.add_argument(
        "--anchor-conf-threshold",
        type=float,
        default=None,
        help="For --anchor-conf-metric: if confidence < threshold, fall back to uniform (return empty anchors).",
    )
    p.add_argument(
        "--anchor-base-alloc",
        type=str,
        default="distance",
        choices=["distance", "score", "farthest", "mixed"],
        help="How to allocate base-res segments in the equal-budget anchored plan. distance=closest-to-anchor (legacy); score=highest eventness scores; farthest=farthest-from-anchor (preserve context); mixed=half near anchors + half far (context).",
    )
    p.add_argument(
        "--anchor-high-policy",
        type=str,
        default="fixed",
        choices=["fixed", "adaptive_v1", "adaptive_v2", "adaptive_v3"],
        help="How many anchors get high-res allocation: fixed uses --max-high-anchors; adaptive_v1 demotes the 2nd high anchor when anchors are adjacent (and optionally when top2 gap is large); adaptive_v2 adds confidence-based demotion for medium-confidence clips; adaptive_v3 only allows 2-high when anchors are adjacent (and demotes far anchors to preserve context).",
    )
    p.add_argument(
        "--anchor-high-adjacent-dist",
        type=int,
        default=1,
        help="For --anchor-high-policy adaptive_v1: if top2 anchors are within this distance, allocate high-res to only 1 anchor.",
    )
    p.add_argument(
        "--anchor-high-gap-threshold",
        type=float,
        default=0.0,
        help="For --anchor-high-policy adaptive_v1: if (top1_score - top2_score) >= threshold, allocate high-res to only 1 anchor. 0 disables.",
    )
    p.add_argument(
        "--eventness-method",
        type=str,
        default="energy",
        choices=[
            "energy",
            "energy_delta",
            "energy_stride_max",
            "asr_vad",
            "energy_nonspeech_ast",
            "energy_autoshift_clipdiff",
            "energy_autoshift_clipdiff_pos",
            "av_clap_clip_agree",
            "clap_evt",
            "clap_lr",
            "clap_mlp_cls",
            "clap_mlp_cls_target",
            "av_fused",
            "av_fused_prod",
            "av_fused_clipdiff",
            "av_fused_clipdiff_prod",
            "moe_energy_clipdiff",
            "av_basic_lr",
            "av_basic_mlp",
            "av_clipdiff_lr",
            "av_clipdiff_mlp",
            "av_clipdiff_accflip_mlp",
            "av_clipdiff_speech_mlp",
            "av_clipdiff_framediff_mlp",
            "av_clipdiff_fbank_mlp",
            "av_ast_clipdiff_mlp",
            "av_ast_clipdiff_mil_mlp",
            "av_ast_clipdiff_tcn",
            "av_ast_clipalign_nce",
            "av_ast_clipalign_bce",
            "av_clipdiff_vec_mlp",
            "av_clipdiff_mlp_cls",
            "av_clipdiff_mlp_cls_target",
            "av_clip_mlp_cls",
            "av_clip_mlp_cls_target",
            "av_clipdiff_tcn",
            "vision_clipdiff",
            "vision_binary_lr",
            "vision_binary_mlp",
            "vision_mlp_cls",
            "vision_mlp_cls_target",
            "ast",
            "ast_nonspeech_max",
            "ast_lr",
            "ast_emb_lr",
            "ast_evt_mlp",
            "ast_mlp_cls",
            "ast_mlp_cls_target",
            "panns",
            "audiomae",
            "audio_basic_lr",
            "audio_basic_mlp",
            "audio_basic_tcn",
            "audio_fbank_mlp",
            "audio_fbank_tcn",
            "audio_basic_mlp_cls",
            "audio_basic_mlp_cls_target",
        ],
    )
    p.add_argument("--ast-pretrained", action="store_true", help="Use pretrained AST weights (downloads from HF)")
    p.add_argument("--panns-checkpoint", type=Path, default=None, help="Path to PANNs Cnn14 checkpoint (.pth)")
    p.add_argument("--panns-random", action="store_true", help="Use random PANNs weights (no checkpoint; smoke/debug only)")
    p.add_argument("--audiomae-checkpoint", type=Path, default=None, help="Path to AudioMAE(-style) checkpoint (optional)")
    p.add_argument("--audiomae-random", action="store_true", help="Use random AudioMAE(-style) weights (no checkpoint; smoke/debug only)")

    p.add_argument("--low-res", type=int, default=112)
    p.add_argument("--base-res", type=int, default=224)
    p.add_argument("--high-res", type=int, default=448)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument(
        "--max-high-anchors",
        type=int,
        default=None,
        help="Optional cap on how many anchors get high-res allocation (budget-aware). Default: use as many as budget allows.",
    )

    p.add_argument("--triad-policy", type=str, default="fixed", choices=["fixed", "top1med_tiered_v1"])
    p.add_argument("--triad-alt-conf-threshold", type=float, default=0.0, help="0 disables tiered triad.")
    p.add_argument("--triad-alt-low-res", type=int, default=112)
    p.add_argument("--triad-alt-high-res", type=int, default=448)
    p.add_argument(
        "--triad-alt-max-high-anchors",
        type=int,
        default=1,
        help="Cap high-res anchors under the alt triad. Use -1 for no extra cap.",
    )

    p.add_argument("--head", type=str, default="mlp", choices=["mlp", "temporal_conv"])
    p.add_argument("--head-hidden-dim", type=int, default=128)
    p.add_argument("--head-dropout", type=float, default=0.0)
    p.add_argument("--temporal-kernel-size", type=int, default=3, help="Only for --head temporal_conv; must be odd.")
    p.add_argument("--train-device", type=str, default="cpu", help="Device for training the classifier head (cpu or cuda:<i>).")

    p.add_argument("--vision-pretrained", action="store_true", help="Use pretrained CLIP weights (downloads from HF)")
    p.add_argument(
        "--vision-model-name",
        type=str,
        default=ClipVisionEncoderConfig().model_name,
        help="HuggingFace model id for the CLIP vision backbone used to build caches.",
    )
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--audio-device",
        type=str,
        default=None,
        help="Device for audio probe inference (default: --device).",
    )

    p.add_argument("--cache-num-workers", type=int, default=1, help="Number of parallel workers for cache build (multi-process).")
    p.add_argument(
        "--cache-devices",
        type=str,
        default=None,
        help="Comma-separated device list for cache workers (e.g., cuda:0,cuda:1). Default: --device.",
    )
    p.add_argument("--cache-skip-existing", action="store_true", help="Skip cache build for clips with existing <clip_id>.npz.")
    p.add_argument("--cache-resolutions", type=str, default="112,224,448", help="Comma-separated cache resolutions.")

    p.add_argument("--out-dir", type=Path, default=Path("runs") / f"ave_p0_end2end_{time.strftime('%Y%m%d-%H%M%S')}")
    p.add_argument(
        "--cache-only",
        action="store_true",
        help="Run through cache build only (download/preprocess/cache), then exit without running P0 training/eval.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    caches_dir = args.caches_dir or (out_dir / "caches")

    # 1) Resolve subset ids.
    ensure_ave_meta(args.meta_dir)
    index = AVEIndex.from_meta_dir(args.meta_dir)
    if args.train_ids_file is not None:
        train_ids = _read_ids_file(args.train_ids_file, args.limit_train)
    else:
        train_ids = _select_split_ids(args.meta_dir, args.split_train, args.limit_train)

    if args.eval_ids_file is not None:
        eval_ids = _read_ids_file(args.eval_ids_file, args.limit_eval)
    else:
        eval_ids = _select_split_ids(args.meta_dir, args.split_eval, args.limit_eval)
    all_ids = sorted(set(train_ids + eval_ids))

    # 2) Materialize raw videos (optional).
    download_results = None
    if args.mode != "none":
        download_results = materialize_ave_videos(
            video_ids=all_ids,
            out_dir=args.raw_videos_dir,
            mode=args.mode,
            src_dir=args.src_dir,
            ytdlp=args.ytdlp,
            overwrite=bool(args.overwrite),
        )
        if not all(r.ok for r in download_results) and not bool(args.allow_missing):
            payload = {"ok": False, "stage": "download", "results": [r.__dict__ for r in download_results]}
            (out_dir / "download.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            return 2

    # If some downloads failed (or mode=none with partial availability), optionally drop missing clips.
    available = {vid for vid in all_ids if (args.raw_videos_dir / f"{vid}.mp4").exists()}
    if all_ids and not available:
        payload = {"ok": False, "stage": "download", "error": "no raw videos available", "requested": all_ids}
        (out_dir / "download.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return 2
    if bool(args.allow_missing):
        train_ids = [vid for vid in train_ids if vid in available]
        eval_ids = [vid for vid in eval_ids if vid in available]
        all_ids = sorted(set(train_ids + eval_ids))
        if not train_ids or not eval_ids:
            payload = {
                "ok": False,
                "stage": "download",
                "error": "insufficient available videos after filtering",
                "train_ids": train_ids,
                "eval_ids": eval_ids,
                "available": sorted(available),
            }
            (out_dir / "download.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            return 2

    # 3) Preprocess (audio + frames).
    preprocessed = preprocess_ave_videos(
        raw_videos_dir=args.raw_videos_dir,
        out_dir=args.processed_dir,
        video_ids=all_ids,
        skip_existing=bool(args.preprocess_skip_existing),
        allow_missing=bool(args.allow_missing),
        jobs=int(args.preprocess_jobs),
    )
    if bool(args.allow_missing):
        preprocessed_set = set(preprocessed)
        train_ids = [vid for vid in train_ids if vid in preprocessed_set]
        eval_ids = [vid for vid in eval_ids if vid in preprocessed_set]
        all_ids = sorted(set(train_ids + eval_ids))
        if not train_ids or not eval_ids:
            payload = {
                "ok": False,
                "stage": "preprocess",
                "error": "insufficient clips after preprocess filtering",
                "train_ids": train_ids,
                "eval_ids": eval_ids,
                "num_preprocessed": int(len(preprocessed_set)),
            }
            (out_dir / "preprocess.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return 2

    # 4) Build caches.
    resolutions = [int(x) for x in str(args.cache_resolutions).split(",") if str(x).strip()]
    if not resolutions:
        raise SystemExit("cache_resolutions must be non-empty")
    required_res = {int(args.low_res), int(args.base_res), int(args.high_res)}
    missing_res = sorted(required_res - set(int(r) for r in resolutions))
    if missing_res:
        raise SystemExit(f"cache_resolutions missing required resolutions {missing_res}; pass --cache-resolutions to include them.")

    caches_dir.mkdir(parents=True, exist_ok=True)
    cache_num_workers = max(1, int(args.cache_num_workers))
    cache_devices = _parse_devices_csv(args.cache_devices if args.cache_devices is not None else args.device)

    ids_to_cache = list(all_ids)
    if bool(args.cache_skip_existing):
        ids_to_cache = [cid for cid in all_ids if not (caches_dir / f"{cid}.npz").exists()]

    if not ids_to_cache and bool(args.cache_skip_existing):
        cache_payload = {
            "ok": True,
            "skipped_all": True,
            "cache_num_workers": int(cache_num_workers),
            "cache_devices": [str(x) for x in cache_devices],
            "cache_resolutions": [int(x) for x in resolutions],
            "skip_existing": True,
            "vision_model_name": str(args.vision_model_name),
            "vision_pretrained": bool(args.vision_pretrained),
            "workers": [],
            "missing_caches": [],
            "exitcodes": [],
            "num_total_ids": int(len(all_ids)),
            "num_to_cache": 0,
        }
        (out_dir / "cache_build.json").write_text(json.dumps(cache_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        if cache_num_workers == 1:
            encoder = ClipVisionEncoder(
                ClipVisionEncoderConfig(
                    model_name=str(args.vision_model_name),
                    pretrained=bool(args.vision_pretrained),
                    device=args.device,
                )
            )
            cached: set[str] = set()
            for cid in ids_to_cache:
                out_path = caches_dir / f"{cid}.npz"
                if bool(args.cache_skip_existing) and out_path.exists():
                    cached.add(str(cid))
                    continue
                frames_dir = args.processed_dir / cid / "frames"
                try:
                    cache = build_clip_feature_cache(frames_dir=frames_dir, resolutions=resolutions, encoder=encoder)
                    cache.save_npz(out_path)
                    cached.add(str(cid))
                except Exception:
                    if not bool(args.allow_missing):
                        raise
        else:
            import multiprocessing

            cache_build_dir = out_dir / "cache_build"
            cache_build_dir.mkdir(parents=True, exist_ok=True)

            cache_num_workers = min(cache_num_workers, len(ids_to_cache))
            shards: list[list[str]] = [[] for _ in range(cache_num_workers)]
            for i, cid in enumerate(ids_to_cache):
                shards[i % cache_num_workers].append(cid)

            ctx = multiprocessing.get_context("spawn")
            procs: list[multiprocessing.Process] = []
            for wid in range(cache_num_workers):
                device = cache_devices[wid % len(cache_devices)]
                worker_out = cache_build_dir / f"worker_{wid:02d}.json"
                p = ctx.Process(
                    target=_cache_worker,
                    kwargs={
                        "worker_id": int(wid),
                        "processed_dir": args.processed_dir,
                        "caches_dir": caches_dir,
                        "clip_ids": shards[wid],
                        "resolutions": resolutions,
                        "device": str(device),
                        "model_name": str(args.vision_model_name),
                        "pretrained": bool(args.vision_pretrained),
                        "skip_existing": bool(args.cache_skip_existing),
                        "out_path": worker_out,
                    },
                )
                p.start()
                procs.append(p)

            for p in procs:
                p.join()

            worker_results: list[dict] = []
            for wid in range(cache_num_workers):
                worker_out = cache_build_dir / f"worker_{wid:02d}.json"
                if worker_out.exists():
                    worker_results.append(json.loads(worker_out.read_text(encoding="utf-8")))
                else:
                    worker_results.append(
                        {
                            "worker_id": int(wid),
                            "device": str(cache_devices[wid % len(cache_devices)]),
                            "ok": False,
                            "error": "missing worker result file",
                        }
                    )

            missing_caches = [cid for cid in all_ids if not (caches_dir / f"{cid}.npz").exists()]
            cache_ok = all(r.get("ok") for r in worker_results) and not missing_caches and all(p.exitcode == 0 for p in procs)
            cache_payload = {
                "ok": bool(cache_ok),
                "cache_num_workers": int(cache_num_workers),
                "cache_devices": [str(x) for x in cache_devices],
                "cache_resolutions": [int(x) for x in resolutions],
                "skip_existing": bool(args.cache_skip_existing),
                "vision_model_name": str(args.vision_model_name),
                "vision_pretrained": bool(args.vision_pretrained),
                "workers": worker_results,
                "missing_caches": missing_caches,
                "exitcodes": [p.exitcode for p in procs],
            }
            (out_dir / "cache_build.json").write_text(json.dumps(cache_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            if not cache_ok and not bool(args.allow_missing):
                payload = {"ok": False, "stage": "cache_build", "cache_build": cache_payload}
                (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                return 2

    if bool(args.allow_missing):
        cached_set = {p.stem for p in caches_dir.glob("*.npz")}
        train_ids = [vid for vid in train_ids if vid in cached_set]
        eval_ids = [vid for vid in eval_ids if vid in cached_set]
        all_ids = sorted(set(train_ids + eval_ids))
        if not train_ids or not eval_ids:
            payload = {
                "ok": False,
                "stage": "cache_build",
                "error": "insufficient caches after filtering",
                "train_ids": train_ids,
                "eval_ids": eval_ids,
                "num_cached": int(len(cached_set)),
            }
            (out_dir / "cache_build.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return 2

    if bool(args.cache_only):
        payload = {
            "ok": True,
            "stage": "cache_only",
            "meta_dir": str(args.meta_dir),
            "raw_videos_dir": str(args.raw_videos_dir),
            "processed_dir": str(args.processed_dir),
            "caches_dir": str(caches_dir),
            "cache_resolutions": resolutions,
            "cache_num_workers": int(cache_num_workers),
            "cache_devices": cache_devices,
            "cache_skip_existing": bool(args.cache_skip_existing),
            "train": {
                "split": str(args.split_train),
                "ids_file": str(args.train_ids_file) if args.train_ids_file is not None else None,
                "limit": int(args.limit_train),
                "num_ids": int(len(train_ids)),
            },
            "eval": {
                "split": str(args.split_eval),
                "ids_file": str(args.eval_ids_file) if args.eval_ids_file is not None else None,
                "limit": int(args.limit_eval),
                "num_ids": int(len(eval_ids)),
            },
            "num_total_ids": int(len(all_ids)),
        }
        out_path = out_dir / "cache_only.json"
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(out_path)
        return 0

    # 5) Labels.
    labels_by_clip: dict[str, list[int]] = {}
    clip_by_id = {c.video_id: c for c in index.clips}
    for cid in all_ids:
        clip = clip_by_id[cid]
        labels_by_clip[cid] = [int(x) for x in index.segment_labels(clip)]

    # 6) Run P0.
    seeds = [int(s) for s in str(args.seeds).split(",") if str(s).strip()]
    baselines = [
        "uniform",
        "uniform_low",
        "audio_concat_uniform",
        "audio_feat_concat_uniform",
        "random_top2",
        "anchored_top2",
        "audio_concat_anchored_top2",
        "audio_feat_concat_anchored_top2",
        "oracle_top2",
    ]
    metrics = run_p0_from_caches(
        clip_ids_train=train_ids,
        clip_ids_eval=eval_ids,
        labels_by_clip=labels_by_clip,
        caches_dir=caches_dir,
        audio_dir=args.processed_dir,
        cfg=P0Config(
            k=int(args.k),
            low_res=int(args.low_res),
            base_res=int(args.base_res),
            high_res=int(args.high_res),
            patch_size=int(args.patch_size),
            max_high_anchors=args.max_high_anchors,
            triad_policy=str(args.triad_policy),
            triad_alt_conf_threshold=float(args.triad_alt_conf_threshold),
            triad_alt_low_res=int(args.triad_alt_low_res),
            triad_alt_high_res=int(args.triad_alt_high_res),
            triad_alt_max_high_anchors=(
                None if int(args.triad_alt_max_high_anchors) < 0 else int(args.triad_alt_max_high_anchors)
            ),
            anchor_shift=int(args.anchor_shift),
            anchor_std_threshold=float(args.anchor_std_threshold),
            anchor_select=str(args.anchor_select),
            anchor_nms_radius=int(args.anchor_nms_radius),
            anchor_nms_strong_gap=float(args.anchor_nms_strong_gap),
            anchor_window=int(args.anchor_window),
            anchor_smooth_window=int(args.anchor_smooth_window),
            anchor_smooth_mode=str(args.anchor_smooth_mode),
            anchor_conf_metric=str(args.anchor_conf_metric) if args.anchor_conf_metric is not None else None,
            anchor_conf_threshold=float(args.anchor_conf_threshold) if args.anchor_conf_threshold is not None else None,
            anchor_base_alloc=str(args.anchor_base_alloc),
            anchor_high_policy=str(args.anchor_high_policy),
            anchor_high_adjacent_dist=int(args.anchor_high_adjacent_dist),
            anchor_high_gap_threshold=float(args.anchor_high_gap_threshold),
            head=str(args.head),
            head_hidden_dim=int(args.head_hidden_dim),
            head_dropout=float(args.head_dropout),
            temporal_kernel_size=int(args.temporal_kernel_size),
        ),
        baselines=baselines,
        seeds=seeds,
        train_cfg=TrainConfig(epochs=5, batch_size=16, lr=2e-3),
        num_classes=index.num_classes,
        class_names=[str(index.idx_to_label[i]) for i in range(int(index.num_classes))],
        num_segments=10,
        eventness_method=str(args.eventness_method),
        audio_device=str(args.audio_device) if args.audio_device is not None else str(args.device),
        ast_pretrained=bool(args.ast_pretrained),
        panns_random=bool(args.panns_random),
        panns_checkpoint=args.panns_checkpoint,
        audiomae_random=bool(args.audiomae_random),
        audiomae_checkpoint=args.audiomae_checkpoint,
        train_device=str(args.train_device),
    )

    payload = {
        "ok": True,
        "meta_dir": str(args.meta_dir),
        "raw_videos_dir": str(args.raw_videos_dir),
        "processed_dir": str(args.processed_dir),
        "caches_dir": str(caches_dir),
        "train_ids": train_ids,
        "eval_ids": eval_ids,
        "download": [r.__dict__ for r in download_results] if download_results is not None else None,
        "metrics": metrics,
    }
    out_path = out_dir / "metrics.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
