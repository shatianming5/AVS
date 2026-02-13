#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import cv2
from PIL import Image
from tqdm import tqdm


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Export per-second eventness scores from a pretrained PSP-family AVEL model "
            "(from the CPSP repo) into AVS stage-1 scores JSON format."
        )
    )
    p.add_argument("--processed-dir", type=Path, required=True, help="Processed dir containing <video_id>/frames/{0..9}.jpg")
    p.add_argument("--meta-dir", type=Path, default=Path("data/AVE/meta"), help="Contains Annotations.txt")
    p.add_argument(
        "--audio-feature-h5",
        type=Path,
        default=Path("data/AVE/eccv18_features/audio_feature.h5"),
        help="ECCV18 VGGish audio features (avadataset: [4143,10,128])",
    )
    p.add_argument(
        "--model-ckpt",
        type=Path,
        default=Path("data/AVE/cpsp_pretrained/cpsp_avel/vgg/model_vgg_FullySupervised_best_model.pth.tar"),
        help="Pretrained PSP-family checkpoint (state_dict; keys prefixed with 'module.')",
    )
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument(
        "--ids-file",
        type=Path,
        default=None,
        help="Optional: text file of video_ids (one per line). Default: export all ids in processed-dir.",
    )
    p.add_argument("--shard-idx", type=int, default=0, help="Shard index in [0, num-shards).")
    p.add_argument("--num-shards", type=int, default=1, help="Number of shards to split ids into.")
    p.add_argument("--num-segments", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32, help="Batch size for VGG19 forward (in frames).")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", type=str, default="float64", choices=["float64", "float32"])
    p.add_argument(
        "--visual-source",
        type=str,
        default="processed_frames",
        choices=["processed_frames", "raw_video_avg16"],
        help=(
            "processed_frames: use per-second middle frames from processed-dir (fast). "
            "raw_video_avg16: sample 16 frames per second from raw videos and average VGG pool5 features (slower; closer to ECCV18 extraction)."
        ),
    )
    p.add_argument("--raw-videos-dir", type=Path, default=None, help="Required when --visual-source=raw_video_avg16.")
    p.add_argument("--sample-num", type=int, default=16, help="Frames per second to sample for raw_video_avg16.")
    return p.parse_args()


def _load_ids(processed_dir: Path, ids_file: Path | None) -> list[str]:
    if ids_file is not None:
        ids = []
        for line in ids_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = str(line).strip()
            if s:
                ids.append(s)
        # stable unique
        seen: set[str] = set()
        out: list[str] = []
        for x in ids:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out
    return sorted([p.name for p in processed_dir.iterdir() if p.is_dir()])


def _shard_ids(ids: list[str], *, shard_idx: int, num_shards: int) -> list[str]:
    if num_shards <= 0:
        raise ValueError(f"num_shards must be >0, got {num_shards}")
    if shard_idx < 0 or shard_idx >= num_shards:
        raise ValueError(f"shard_idx must be in [0, {num_shards}), got {shard_idx}")
    return [vid for i, vid in enumerate(ids) if (i % num_shards) == shard_idx]


def _build_first_index_by_video_id(meta_dir: Path) -> dict[str, int]:
    """
    Map each unique video_id -> first row index in Annotations.txt (matches ECCV18 feature row indices).

    ECCV18 released audio_feature.h5 has shape [4143,...] while official splits have only 4097 unique video_ids
    (some ids repeat with different labels). We consistently use the first row for duplicates since the media is
    identical.
    """
    ann_path = meta_dir / "Annotations.txt"
    if not ann_path.is_file():
        raise FileNotFoundError(f"missing annotations: {ann_path}")
    out: dict[str, int] = {}
    i = 0
    for line in ann_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = str(line).split("&")
        if len(parts) != 5:
            continue
        _label, vid, _quality, start, end = parts
        try:
            float(start)
            float(end)
        except Exception:
            continue
        vid = str(vid).strip()
        if vid not in out:
            out[vid] = i
        i += 1
    if i <= 0:
        raise ValueError(f"failed to parse any rows from {ann_path}")
    return out


def _load_psp_model(*, ckpt_path: Path, device: str) -> torch.nn.Module:
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"missing model checkpoint: {ckpt_path}")

    # Import third-party CPSP PSP-family model.
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.append(str(repo_root / "third_party" / "CPSP" / "cpsp_avel"))
    from model.psp_family import fully_psp_net  # type: ignore

    state_dict = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    if not isinstance(state_dict, dict):
        raise TypeError(f"unexpected ckpt type: {type(state_dict)}")
    if all(str(k).startswith("module.") for k in state_dict.keys()):
        state_dict = {str(k)[len("module.") :]: v for k, v in state_dict.items()}

    model = fully_psp_net(vis_fea_type="vgg", flag="psp", category_num=28)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)
    model.double()
    return model


def _load_vgg19_pool5(device: str) -> torch.nn.Module:
    # Use the last maxpool output (block5_pool), matching the ECCV18 scripts conceptually.
    model = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1).features
    model.eval()
    model.to(device)
    return model


def _preprocess_vgg19(img: Image.Image) -> torch.Tensor:
    w = torchvision.models.VGG19_Weights.IMAGENET1K_V1
    tfm = w.transforms()
    return tfm(img.convert("RGB"))


def _load_frames_tensor(processed_dir: Path, vid: str, *, num_segments: int) -> torch.Tensor:
    # Returns [T, 3, 224, 224]
    frames_dir = processed_dir / vid / "frames"
    xs: list[torch.Tensor] = []
    for t in range(int(num_segments)):
        p = frames_dir / f"{t}.jpg"
        if not p.is_file():
            raise FileNotFoundError(f"missing frame: {p}")
        xs.append(_preprocess_vgg19(Image.open(p)))
    return torch.stack(xs, dim=0)


@torch.no_grad()
def _vgg19_pool5_batch(vgg19: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    # x: [B,3,224,224] -> [B,512,7,7]
    return vgg19(x)


def _pool5_from_raw_video_avg16(
    *,
    vgg19: torch.nn.Module,
    raw_videos_dir: Path,
    vid: str,
    num_segments: int,
    sample_num: int,
    device: str,
    batch_size: int,
) -> np.ndarray:
    video_path = raw_videos_dir / f"{vid}.mp4"
    if not video_path.is_file():
        raise FileNotFoundError(f"missing raw video: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"invalid frame_count={total_frames} for {video_path}")

    # Follow ECCV18 feature script: split into T chunks, sample `sample_num` frames per chunk.
    interval = float(total_frames) / float(num_segments)
    indices: list[int] = []
    for t in range(int(num_segments)):
        for i in range(int(sample_num)):
            idx = int(t * interval + (float(i) / float(sample_num)) * interval)
            idx = max(0, min(total_frames - 1, idx))
            indices.append(idx)

    need = sorted(set(indices))
    need_ptr = 0
    cur = 0
    frames: dict[int, np.ndarray] = {}
    while need_ptr < len(need):
        ok, frame = cap.read()
        if not ok:
            break
        if cur == need[need_ptr]:
            frames[cur] = frame
            need_ptr += 1
        cur += 1
    cap.release()

    missing = [idx for idx in need if idx not in frames]
    if missing:
        raise RuntimeError(f"failed to decode {len(missing)} sampled frames for {vid} (example idx={missing[0]})")

    # Preprocess sampled frames in the exact requested order.
    xs: list[torch.Tensor] = []
    for idx in indices:
        frame = frames[idx]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        xs.append(_preprocess_vgg19(Image.fromarray(rgb)))
    x = torch.stack(xs, dim=0)  # [T*sample_num,3,224,224]

    # VGG19 forward in minibatches.
    feats: list[torch.Tensor] = []
    for i in range(0, int(x.shape[0]), int(batch_size)):
        xb = x[i : i + int(batch_size)].to(device=device, dtype=torch.float32)
        feats.append(_vgg19_pool5_batch(vgg19, xb).to("cpu"))
    pool5 = torch.cat(feats, dim=0)  # [T*sample_num,512,7,7]

    pool5 = pool5.view(int(num_segments), int(sample_num), 512, 7, 7).mean(dim=1)  # [T,512,7,7]
    pool5 = pool5.permute(0, 2, 3, 1).contiguous().numpy()  # [T,7,7,512]
    return pool5


def main() -> None:
    args = _parse_args()
    processed_dir = Path(args.processed_dir)
    if not processed_dir.is_dir():
        raise FileNotFoundError(f"processed-dir not found: {processed_dir}")

    ids = _load_ids(processed_dir, args.ids_file)
    ids = _shard_ids(ids, shard_idx=int(args.shard_idx), num_shards=int(args.num_shards))
    if not ids:
        raise ValueError("no ids to export")

    first_idx = _build_first_index_by_video_id(Path(args.meta_dir))

    # Audio features (ECCV18 VGGish embeddings).
    audio_h5_path = Path(args.audio_feature_h5)
    if not audio_h5_path.is_file():
        raise FileNotFoundError(f"audio_feature.h5 not found: {audio_h5_path}")

    device = str(args.device)
    psp = _load_psp_model(ckpt_path=Path(args.model_ckpt), device=device)
    vgg19 = _load_vgg19_pool5(device=device)

    scores: dict[str, list[float]] = {}

    # Keep H5 open for random access.
    with h5py.File(str(audio_h5_path), "r") as f_audio:
        if "avadataset" not in f_audio:
            raise KeyError(f"missing 'avadataset' in {audio_h5_path}")
        audio_ds = f_audio["avadataset"]

        raw_videos_dir = None
        if str(args.visual_source) == "raw_video_avg16":
            if args.raw_videos_dir is None:
                raise ValueError("--raw-videos-dir is required when --visual-source=raw_video_avg16")
            raw_videos_dir = Path(args.raw_videos_dir)
            if not raw_videos_dir.is_dir():
                raise FileNotFoundError(f"raw-videos-dir not found: {raw_videos_dir}")

        for vid in tqdm(ids, desc="export_psp_evt", unit="vid"):
            if vid not in first_idx:
                raise KeyError(f"video_id {vid!r} not found in {Path(args.meta_dir) / 'Annotations.txt'}")
            a = audio_ds[int(first_idx[vid])]  # [10,128], float64
            a = np.asarray(a, dtype=np.float64)[: int(args.num_segments)]
            if a.shape != (int(args.num_segments), 128):
                raise ValueError(f"bad audio shape for {vid}: {a.shape}")

            if str(args.visual_source) == "processed_frames":
                x = _load_frames_tensor(processed_dir, vid, num_segments=int(args.num_segments))  # [T,3,224,224]
                x = x.to(device=device, dtype=torch.float32)
                # pool5: [T,512,7,7] -> [T,7,7,512]
                v = _vgg19_pool5_batch(vgg19, x).detach().to("cpu")
                v = v.permute(0, 2, 3, 1).contiguous().numpy()  # [T,7,7,512]
            elif str(args.visual_source) == "raw_video_avg16":
                assert raw_videos_dir is not None
                v = _pool5_from_raw_video_avg16(
                    vgg19=vgg19,
                    raw_videos_dir=raw_videos_dir,
                    vid=vid,
                    num_segments=int(args.num_segments),
                    sample_num=int(args.sample_num),
                    device=device,
                    batch_size=int(args.batch_size),
                )
            else:
                raise ValueError(f"unsupported visual_source: {args.visual_source}")

            # Model expects:
            #   video: [B,10,7,7,512]
            #   audio: [B,10,128]
            video = torch.from_numpy(v)[None, ...].to(device=device, dtype=torch.float64)
            audio = torch.from_numpy(a)[None, ...].to(device=device, dtype=torch.float64)

            event_logits, _category_logits, _avps, _fusion = psp(video, audio)
            event_prob = torch.sigmoid(event_logits).squeeze(0)  # [T]
            scores[vid] = [float(x) for x in event_prob.detach().to("cpu").tolist()]

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "eventness_method": "psp_avel_evt",
        "visual_source": str(args.visual_source),
        "num_segments": int(args.num_segments),
        "shard_idx": int(args.shard_idx),
        "num_shards": int(args.num_shards),
        "unique_vids": int(len(scores)),
        "model_ckpt": str(Path(args.model_ckpt)),
        "audio_feature_h5": str(Path(args.audio_feature_h5)),
        "scores": scores,
    }
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"OK: wrote {args.out_json} (unique_vids={len(scores)})")


if __name__ == "__main__":
    main()
