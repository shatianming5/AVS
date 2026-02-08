from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from avs.datasets.epic_sounds import EpicSoundsIndex, ensure_epic_sounds_meta
from avs.datasets.layout import epic_sounds_paths
from avs.models.video_multilabel_head import VideoMultiLabelHead
from avs.pipeline.long_plan_generation import long_plan_from_wav_hybrid
from avs.pipeline.plan_generation import infer_num_segments_from_wav
from avs.preprocess.epic_sounds_audio import extract_epic_sounds_audio
from avs.preprocess.epic_sounds_frames import extract_epic_sounds_frames
from avs.sampling.plans import SamplingPlan, uniform_plan
from avs.vision.clip_vit import ClipVisionEncoder, ClipVisionEncoderConfig
from avs.vision.feature_cache import FeatureCache, build_clip_feature_cache_from_seconds


def average_precision_binary(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """
    Average precision (AP) for a single class.
    Uses the "mean precision at true positives" formulation.
    """
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if y_true.size == 0:
        return None
    pos = int(y_true.sum())
    if pos <= 0:
        return None

    order = np.argsort(-y_score, kind="mergesort")  # stable
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted == 1)
    idx = np.arange(1, y_true.size + 1, dtype=np.float64)
    prec = tp / idx
    ap = float(np.sum(prec[y_sorted == 1]) / float(pos))
    return ap


def mean_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Mean AP over classes with at least one positive in y_true.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.ndim != 2 or y_score.ndim != 2 or y_true.shape != y_score.shape:
        raise ValueError(f"expected y_true/y_score to be [N,C] with same shape, got {y_true.shape} vs {y_score.shape}")
    aps: list[float] = []
    for c in range(y_true.shape[1]):
        ap = average_precision_binary(y_true[:, c], y_score[:, c])
        if ap is not None and math.isfinite(ap):
            aps.append(float(ap))
    return float(np.mean(np.asarray(aps, dtype=np.float64))) if aps else 0.0


def macro_f1(y_true: np.ndarray, y_prob: np.ndarray, *, threshold: float = 0.5) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if y_true.shape != y_prob.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_prob.shape}")
    y_pred = (y_prob >= float(threshold)).astype(np.int64)

    f1s: list[float] = []
    for c in range(y_true.shape[1]):
        tp = int(((y_true[:, c] == 1) & (y_pred[:, c] == 1)).sum())
        fp = int(((y_true[:, c] == 0) & (y_pred[:, c] == 1)).sum())
        fn = int(((y_true[:, c] == 1) & (y_pred[:, c] == 0)).sum())
        if tp == 0 and (fp > 0 or fn > 0):
            f1s.append(0.0)
            continue
        denom = (2 * tp + fp + fn)
        f1s.append(float((2 * tp) / denom) if denom > 0 else 0.0)
    return float(np.mean(np.asarray(f1s, dtype=np.float64))) if f1s else 0.0


def _features_from_cache(cache: FeatureCache, plan: SamplingPlan) -> np.ndarray:
    t = len(plan.resolutions)
    any_r = cache.resolutions[0]
    d = cache.features_by_resolution[any_r].shape[-1]
    out = np.empty((t, d), dtype=np.float32)
    for i, r in enumerate(plan.resolutions):
        out[i] = cache.features_by_resolution[int(r)][i]
    return out


def select_seconds_uniform(*, duration_seconds: int, max_steps: int) -> list[int]:
    dur = max(0, int(duration_seconds))
    if dur <= 0:
        return []
    target = min(max(1, int(max_steps)), dur)
    if dur <= target:
        return list(range(dur))

    # Centered stratified selection (deterministic).
    secs: list[int] = []
    used: set[int] = set()
    for i in range(target):
        t = int(((i + 0.5) * dur) / target)
        t = max(0, min(dur - 1, t))
        if t not in used:
            secs.append(t)
            used.add(t)

    # Fill any gaps (rare due to rounding collisions).
    if len(secs) < target:
        for t in range(dur):
            if t in used:
                continue
            secs.append(t)
            used.add(t)
            if len(secs) >= target:
                break

    if len(secs) != target:
        raise AssertionError(f"bug: expected {target} seconds, got {len(secs)}")
    return secs


def select_seconds_random(*, duration_seconds: int, max_steps: int, rng: random.Random) -> list[int]:
    dur = max(0, int(duration_seconds))
    if dur <= 0:
        return []
    target = min(max(1, int(max_steps)), dur)
    return sorted(rng.sample(range(dur), k=target))


@dataclass(frozen=True)
class EpicVideoPack:
    video_id: str
    duration_seconds: int
    selected_seconds: list[int]
    anchors_seconds: list[int]
    plan: SamplingPlan
    cache_path: Path | None

    def to_jsonable(self) -> dict:
        return {
            "video_id": str(self.video_id),
            "duration_seconds": int(self.duration_seconds),
            "selected_seconds": [int(x) for x in self.selected_seconds],
            "anchors_seconds": [int(x) for x in self.anchors_seconds],
            "plan": self.plan.to_jsonable(),
            "cache_path": str(self.cache_path) if self.cache_path is not None else None,
        }


def pack_one_video(
    *,
    video_id: str,
    videos_dir: Path,
    out_dir: Path,
    selection: str,
    max_seconds: int | None,
    max_steps: int,
    eventness_method: str,
    k: int,
    anchor_radius: int,
    background_stride: int,
    low_res: int,
    base_res: int,
    high_res: int,
    patch_size: int,
    anchor_shift: int,
    anchor_std_threshold: float,
    encode: bool,
    cache_resolutions: list[int],
    encoder: ClipVisionEncoder | None,
    clip_model_name: str,
    clip_device: str,
    clip_dtype: str,
    clip_pretrained: bool,
    start_offset_sec: float,
    jpg_quality: int,
    ast_pretrained: bool,
    panns_random: bool,
    panns_checkpoint: Path | None,
    audiomae_random: bool,
    audiomae_checkpoint: Path | None,
) -> EpicVideoPack:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    (out_dir / "plans").mkdir(parents=True, exist_ok=True)
    (out_dir / "caches").mkdir(parents=True, exist_ok=True)

    # 1) Extract audio + frames (deterministic).
    extract_epic_sounds_audio(videos_dir=videos_dir, out_audio_dir=out_dir / "audio", video_ids=[video_id])
    extract_epic_sounds_frames(
        videos_dir=videos_dir,
        out_frames_dir=out_dir / "frames",
        video_ids=[video_id],
        start_offset_sec=float(start_offset_sec),
        max_seconds=int(max_seconds) if max_seconds is not None else None,
        jpg_quality=int(jpg_quality),
    )

    wav_path = out_dir / "audio" / f"{video_id}.wav"
    duration = infer_num_segments_from_wav(wav_path)
    if max_seconds is not None:
        duration = min(int(duration), int(max_seconds))

    # 2) Select seconds + build an equal-budget plan.
    anchors_seconds: list[int] = []
    selected_seconds: list[int] = []
    if selection == "uniform":
        selected_seconds = select_seconds_uniform(duration_seconds=duration, max_steps=max_steps)
        plan = uniform_plan(num_segments=len(selected_seconds), resolution=int(base_res), patch_size=int(patch_size))
    elif selection == "random":
        import hashlib

        seed = int.from_bytes(hashlib.md5(str(video_id).encode("utf-8")).digest()[:8], "little", signed=False)
        rng = random.Random(int(seed))
        selected_seconds = select_seconds_random(duration_seconds=duration, max_steps=max_steps, rng=rng)
        plan = uniform_plan(num_segments=len(selected_seconds), resolution=int(base_res), patch_size=int(patch_size))
    elif selection == "audio_anchored":
        rec = long_plan_from_wav_hybrid(
            clip_id=str(video_id),
            wav_path=wav_path,
            max_seconds=int(max_seconds) if max_seconds is not None else None,
            eventness_method=str(eventness_method),
            k=int(k),
            anchor_radius=int(anchor_radius),
            background_stride=int(background_stride),
            max_steps=int(max_steps),
            low_res=int(low_res),
            base_res=int(base_res),
            high_res=int(high_res),
            patch_size=int(patch_size),
            anchor_shift=int(anchor_shift),
            anchor_std_threshold=float(anchor_std_threshold),
            ast_pretrained=bool(ast_pretrained),
            panns_random=bool(panns_random),
            panns_checkpoint=panns_checkpoint,
            audiomae_random=bool(audiomae_random),
            audiomae_checkpoint=audiomae_checkpoint,
        )
        anchors_seconds = [int(x) for x in rec.anchors_seconds]
        selected_seconds = [int(x) for x in rec.selected_seconds]
        plan = rec.plan
    else:
        raise ValueError(f"unknown selection={selection!r}; expected 'uniform', 'random', or 'audio_anchored'")

    plan_path = out_dir / "plans" / f"{video_id}.{selection}.plan.json"
    plan.save_json(plan_path)

    # 3) Encode + cache features (optional).
    cache_path: Path | None = None
    if encode:
        if encoder is None:
            encoder = ClipVisionEncoder(
                ClipVisionEncoderConfig(
                    model_name=str(clip_model_name),
                    device=str(clip_device),
                    dtype=str(clip_dtype),
                    pretrained=bool(clip_pretrained),
                )
            )
        frames_dir = out_dir / "frames" / video_id / "frames"
        cache = build_clip_feature_cache_from_seconds(
            frames_dir=frames_dir,
            seconds=selected_seconds,
            resolutions=[int(r) for r in cache_resolutions],
            encoder=encoder,
        )
        cache_path = out_dir / "caches" / f"{video_id}.{selection}.npz"
        cache.save_npz(cache_path)

    pack = EpicVideoPack(
        video_id=str(video_id),
        duration_seconds=int(duration),
        selected_seconds=[int(x) for x in selected_seconds],
        anchors_seconds=[int(x) for x in anchors_seconds],
        plan=plan,
        cache_path=cache_path,
    )
    (out_dir / "plans" / f"{video_id}.{selection}.pack.json").write_text(
        json.dumps(pack.to_jsonable(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return pack


def _collect_epic_labels(index: EpicSoundsIndex, split: str) -> dict[str, set[int]]:
    if split == "train":
        segs = index.train
    elif split == "val":
        segs = index.val
    else:
        raise ValueError("only train/val are supported for supervised recognition")

    by_video: dict[str, set[int]] = {}
    for s in segs:
        if s.class_id is None:
            continue
        by_video.setdefault(s.video_id, set()).add(int(s.class_id))
    return by_video


def _dense_class_mapping(by_video_train: dict[str, set[int]], by_video_val: dict[str, set[int]]) -> dict[int, int]:
    ids: set[int] = set()
    for d in (by_video_train, by_video_val):
        for v in d.values():
            ids.update(int(x) for x in v)
    if not ids:
        return {}
    ordered = sorted(ids)
    return {int(cid): int(i) for i, cid in enumerate(ordered)}


def _labels_to_multihot(labels: set[int], mapping: dict[int, int], num_classes: int) -> np.ndarray:
    y = np.zeros((num_classes,), dtype=np.float32)
    for cid in labels:
        if int(cid) in mapping:
            y[int(mapping[int(cid)])] = 1.0
    return y


def _pad_batch(xs: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad variable-length [T,D] arrays into (x, mask):
      - x: [B,Tmax,D]
      - mask: [B,Tmax] (1 valid, 0 pad)
    """
    b = len(xs)
    tmax = max(int(x.shape[0]) for x in xs)
    d = int(xs[0].shape[1])
    x = torch.zeros((b, tmax, d), dtype=torch.float32)
    mask = torch.zeros((b, tmax), dtype=torch.float32)
    for i, arr in enumerate(xs):
        t = int(arr.shape[0])
        x[i, :t] = torch.from_numpy(arr).float()
        mask[i, :t] = 1.0
    return x, mask


def train_and_eval_multilabel(
    *,
    x_train: list[np.ndarray],
    y_train: np.ndarray,
    x_val: list[np.ndarray],
    y_val: np.ndarray,
    num_classes: int,
    device: str,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 2e-3,
    weight_decay: float = 0.0,
    hidden_dim: int = 256,
    dropout: float = 0.1,
) -> dict:
    if not x_train or not x_val:
        raise ValueError("empty train/val data")
    if y_train.shape[1] != num_classes or y_val.shape[1] != num_classes:
        raise ValueError("label shape mismatch")

    in_dim = int(x_train[0].shape[1])
    model = VideoMultiLabelHead(in_dim=in_dim, num_classes=int(num_classes), hidden_dim=int(hidden_dim), dropout=float(dropout)).to(
        torch.device(device)
    )

    # pos_weight for imbalance.
    pos = y_train.sum(axis=0).astype(np.float64)
    neg = float(y_train.shape[0]) - pos
    pos_weight = np.ones((num_classes,), dtype=np.float32)
    for c in range(num_classes):
        if pos[c] > 0:
            pos_weight[c] = float(neg[c] / pos[c])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(pos_weight).to(torch.device(device)))
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    rng = np.random.default_rng(0)
    n = len(x_train)
    steps = max(1, int(math.ceil(n / float(batch_size))))

    for _epoch in range(int(epochs)):
        order = rng.permutation(n)
        model.train()
        for si in range(steps):
            idx = order[si * int(batch_size) : (si + 1) * int(batch_size)]
            xb, mask = _pad_batch([x_train[int(i)] for i in idx.tolist()])
            yb = torch.from_numpy(y_train[idx]).float()
            xb = xb.to(torch.device(device))
            mask = mask.to(torch.device(device))
            yb = yb.to(torch.device(device))

            logits = model(xb, mask=mask)
            loss = loss_fn(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    # Eval
    model.eval()
    with torch.no_grad():
        all_logits: list[np.ndarray] = []
        n_val = len(x_val)
        steps_val = max(1, int(math.ceil(n_val / float(batch_size))))
        for si in range(steps_val):
            start = si * int(batch_size)
            end = min(n_val, (si + 1) * int(batch_size))
            xb, mask = _pad_batch(x_val[start:end])
            xb = xb.to(torch.device(device))
            mask = mask.to(torch.device(device))
            logits = model(xb, mask=mask).detach().cpu().numpy().astype(np.float64)
            all_logits.append(logits)
        logits_val = np.concatenate(all_logits, axis=0)

    prob_val = 1.0 / (1.0 + np.exp(-logits_val))
    metrics = {
        "mAP": mean_average_precision(y_val, prob_val),
        "macro_f1@0.5": macro_f1(y_val, prob_val, threshold=0.5),
    }
    return {"metrics": metrics, "pos_weight": [float(x) for x in pos_weight.tolist()]}


def run_synth_smoke(*, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    n_train = 32
    n_val = 16
    num_classes = 7
    d = 32

    def _rand_seq(n: int) -> list[np.ndarray]:
        xs: list[np.ndarray] = []
        for _ in range(n):
            t = int(rng.integers(low=4, high=12))
            xs.append(rng.standard_normal((t, d), dtype=np.float32))
        return xs

    x_train = _rand_seq(n_train)
    x_val = _rand_seq(n_val)

    y_train = (rng.random((n_train, num_classes)) < 0.2).astype(np.float32)
    y_val = (rng.random((n_val, num_classes)) < 0.2).astype(np.float32)

    res = train_and_eval_multilabel(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        num_classes=num_classes,
        device="cpu",
        epochs=2,
        batch_size=8,
        lr=2e-3,
        hidden_dim=64,
        dropout=0.0,
    )
    payload = {"ok": True, "synthetic": True, "metrics": res["metrics"], "pos_weight": res["pos_weight"]}
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EPIC-SOUNDS downstream proxy: video-level multi-label recognition from selected visual seconds.")
    p.add_argument("--videos-dir", type=Path, default=epic_sounds_paths().raw_videos_dir, help="Dir containing <video_id>.mp4")
    p.add_argument("--meta-dir", type=Path, default=epic_sounds_paths().meta_dir)
    p.add_argument("--out-dir", type=Path, default=Path("runs") / f"epic_sounds_video_cls_{time.strftime('%Y%m%d-%H%M%S')}")

    p.add_argument("--selection", type=str, default="audio_anchored", choices=["uniform", "random", "audio_anchored"])
    p.add_argument("--max-seconds", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=120)

    p.add_argument("--eventness-method", type=str, default="energy", choices=["energy", "energy_delta", "ast", "panns", "audiomae"])
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--anchor-radius", type=int, default=2)
    p.add_argument("--background-stride", type=int, default=5)
    p.add_argument("--anchor-shift", type=int, default=0)
    p.add_argument("--anchor-std-threshold", type=float, default=0.0)

    p.add_argument("--low-res", type=int, default=112)
    p.add_argument("--base-res", type=int, default=224)
    p.add_argument("--high-res", type=int, default=448)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--cache-resolutions", type=str, default="112,224,448", help="Comma-separated resolutions to cache for selected seconds.")

    p.add_argument("--encode", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--clip-pretrained", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--clip-model-name", type=str, default="openai/clip-vit-base-patch16")
    p.add_argument("--clip-device", type=str, default="cpu")
    p.add_argument("--clip-dtype", type=str, default="float32")

    p.add_argument("--start-offset-sec", type=float, default=0.5)
    p.add_argument("--jpg-quality", type=int, default=2)

    p.add_argument("--limit-train-videos", type=int, default=64)
    p.add_argument("--limit-val-videos", type=int, default=64)
    p.add_argument(
        "--allow-missing-videos",
        action="store_true",
        help="If set, select up to limit videos from available mp4s instead of failing on missing fixed IDs.",
    )
    p.add_argument(
        "--min-train-videos",
        type=int,
        default=16,
        help="Minimum available train videos required when --allow-missing-videos is enabled.",
    )
    p.add_argument(
        "--min-val-videos",
        type=int,
        default=16,
        help="Minimum available val videos required when --allow-missing-videos is enabled.",
    )

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--train-device", type=str, default="cpu")

    p.add_argument("--ast-pretrained", action="store_true")
    p.add_argument("--panns-checkpoint", type=Path, default=None)
    p.add_argument("--panns-random", action="store_true")
    p.add_argument("--audiomae-checkpoint", type=Path, default=None)
    p.add_argument("--audiomae-random", action="store_true")

    p.add_argument("--synthetic", action="store_true", help="Run a synthetic smoke-like experiment (no videos required).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if bool(args.synthetic):
        payload = run_synth_smoke(out_dir=out_dir)
        print(out_dir / "metrics.json")
        return 0 if bool(payload.get("ok")) else 2

    ensure_epic_sounds_meta(Path(args.meta_dir))
    index = EpicSoundsIndex.from_meta_dir(Path(args.meta_dir))

    by_video_train = _collect_epic_labels(index, "train")
    by_video_val = _collect_epic_labels(index, "val")
    mapping = _dense_class_mapping(by_video_train, by_video_val)
    num_classes = int(len(mapping))
    if num_classes <= 0:
        raise SystemExit("no labeled classes found in EPIC-SOUNDS meta (train/val)")

    videos_dir = Path(args.videos_dir)
    all_train_videos = sorted(by_video_train.keys())
    all_val_videos = sorted(by_video_val.keys())

    if bool(args.allow_missing_videos):
        available_train = [v for v in all_train_videos if (videos_dir / f"{v}.mp4").exists()]
        available_val = [v for v in all_val_videos if (videos_dir / f"{v}.mp4").exists()]
        train_videos = available_train[: int(args.limit_train_videos)]
        val_videos = available_val[: int(args.limit_val_videos)]

        min_train = int(args.min_train_videos)
        min_val = int(args.min_val_videos)
        if len(train_videos) < min_train or len(val_videos) < min_val:
            raise SystemExit(
                "insufficient available videos under "
                f"{videos_dir}: train={len(train_videos)}(<{min_train}) val={len(val_videos)}(<{min_val})"
            )
    else:
        train_videos = all_train_videos[: int(args.limit_train_videos)]
        val_videos = all_val_videos[: int(args.limit_val_videos)]
        missing = [v for v in (train_videos + val_videos) if not (videos_dir / f"{v}.mp4").exists()]
        if missing:
            raise SystemExit(f"missing mp4s under {videos_dir}: {missing[:5]}{' ...' if len(missing) > 5 else ''}")

    cache_resolutions = [int(x) for x in str(args.cache_resolutions).split(",") if str(x).strip()]
    if not cache_resolutions:
        raise SystemExit("empty --cache-resolutions")

    encoder = None
    if bool(args.encode):
        encoder = ClipVisionEncoder(
            ClipVisionEncoderConfig(
                model_name=str(args.clip_model_name),
                device=str(args.clip_device),
                dtype=str(args.clip_dtype),
                pretrained=bool(args.clip_pretrained),
            )
        )

    packs_train: list[EpicVideoPack] = []
    packs_val: list[EpicVideoPack] = []
    for split, vids in (("train", train_videos), ("val", val_videos)):
        for vid in vids:
            pack_out = out_dir / "pack" / split
            pack = pack_one_video(
                video_id=str(vid),
                videos_dir=videos_dir,
                out_dir=pack_out,
                selection=str(args.selection),
                max_seconds=args.max_seconds,
                max_steps=int(args.max_steps),
                eventness_method=str(args.eventness_method),
                k=int(args.k),
                anchor_radius=int(args.anchor_radius),
                background_stride=int(args.background_stride),
                low_res=int(args.low_res),
                base_res=int(args.base_res),
                high_res=int(args.high_res),
                patch_size=int(args.patch_size),
                anchor_shift=int(args.anchor_shift),
                anchor_std_threshold=float(args.anchor_std_threshold),
                encode=bool(args.encode),
                cache_resolutions=cache_resolutions,
                encoder=encoder,
                clip_model_name=str(args.clip_model_name),
                clip_device=str(args.clip_device),
                clip_dtype=str(args.clip_dtype),
                clip_pretrained=bool(args.clip_pretrained),
                start_offset_sec=float(args.start_offset_sec),
                jpg_quality=int(args.jpg_quality),
                ast_pretrained=bool(args.ast_pretrained),
                panns_random=bool(args.panns_random),
                panns_checkpoint=args.panns_checkpoint,
                audiomae_random=bool(args.audiomae_random),
                audiomae_checkpoint=args.audiomae_checkpoint,
            )
            if split == "train":
                packs_train.append(pack)
            else:
                packs_val.append(pack)

    # Load caches into features for training/eval.
    x_train: list[np.ndarray] = []
    y_train_rows: list[np.ndarray] = []
    for p in packs_train:
        if p.cache_path is None:
            raise SystemExit("--no-encode is not supported for training (need caches)")
        cache = FeatureCache.load_npz(p.cache_path)
        x_train.append(_features_from_cache(cache, p.plan))
        y_train_rows.append(_labels_to_multihot(by_video_train[p.video_id], mapping, num_classes))

    x_val: list[np.ndarray] = []
    y_val_rows: list[np.ndarray] = []
    for p in packs_val:
        if p.cache_path is None:
            raise SystemExit("--no-encode is not supported for training (need caches)")
        cache = FeatureCache.load_npz(p.cache_path)
        x_val.append(_features_from_cache(cache, p.plan))
        y_val_rows.append(_labels_to_multihot(by_video_val[p.video_id], mapping, num_classes))

    y_train = np.stack(y_train_rows, axis=0).astype(np.float32)
    y_val = np.stack(y_val_rows, axis=0).astype(np.float32)

    train_out = train_and_eval_multilabel(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        num_classes=num_classes,
        device=str(args.train_device),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
    )

    payload = {
        "ok": True,
        "meta_dir": str(args.meta_dir),
        "videos_dir": str(videos_dir),
        "selection": str(args.selection),
        "max_seconds": int(args.max_seconds) if args.max_seconds is not None else None,
        "max_steps": int(args.max_steps),
        "budget_def": "max_steps × base_res (token-equivalent); audio_anchored uses equal-budget low/base/high plan.",
        "low_res": int(args.low_res),
        "base_res": int(args.base_res),
        "high_res": int(args.high_res),
        "patch_size": int(args.patch_size),
        "cache_resolutions": cache_resolutions,
        "encode": bool(args.encode),
        "num_classes": int(num_classes),
        "num_train_videos": int(len(packs_train)),
        "num_val_videos": int(len(packs_val)),
        "allow_missing_videos": bool(args.allow_missing_videos),
        "requested_limit_train_videos": int(args.limit_train_videos),
        "requested_limit_val_videos": int(args.limit_val_videos),
        "train": train_out,
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out_dir / "metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
