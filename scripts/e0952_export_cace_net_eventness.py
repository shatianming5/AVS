#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch


def _install_braincog_stub() -> None:
    """
    CACE-Net imports BrainCog for an optional SNN encoder variant. For the supervised AVE config
    we use, that codepath is not exercised. To keep this exporter self-contained, install a stub
    module so `import braincog.model_zoo.fc_snn` succeeds even when BrainCog is not installed.
    """
    import types

    if "braincog" in sys.modules:
        return

    braincog = types.ModuleType("braincog")
    model_zoo = types.ModuleType("braincog.model_zoo")
    fc_snn = types.ModuleType("braincog.model_zoo.fc_snn")

    class SHD_SNN:  # noqa: N801 (match upstream symbol)
        def __init__(self, *args, **kwargs):
            raise RuntimeError("BrainCog is not installed; SHD_SNN is unavailable in this environment.")

    fc_snn.SHD_SNN = SHD_SNN
    model_zoo.fc_snn = fc_snn
    braincog.model_zoo = model_zoo

    sys.modules["braincog"] = braincog
    sys.modules["braincog.model_zoo"] = model_zoo
    sys.modules["braincog.model_zoo.fc_snn"] = fc_snn


def _load_vgg19_block5_pool(*, device: torch.device) -> tuple[torch.nn.Module, object]:
    """
    Torchvision VGG19 feature extractor up to `block5_pool` equivalent:
    returns a module mapping [N,3,224,224] -> [N,512,7,7].
    """
    from torchvision import transforms as T
    from torchvision.models import VGG19_Weights, vgg19

    weights = VGG19_Weights.IMAGENET1K_V1
    model = vgg19(weights=weights).features
    model = model.to(device=device, dtype=torch.float32)
    model.eval()

    # Torchvision image classification normalization (ImageNet).
    # Some torchvision weight enums don't expose mean/std in `.meta`, so keep it explicit.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )
    return model, transform


def _preprocess_bgr_frame_to_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
    """
    Convert an OpenCV BGR uint8 frame to a normalized float32 tensor [3,224,224] (RGB, ImageNet norm).
    """
    import cv2

    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(f"unexpected frame shape: {frame_bgr.shape}")
    x = cv2.resize(frame_bgr, (224, 224), interpolation=cv2.INTER_LINEAR)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x.astype(np.float32) / 255.0
    # ImageNet normalization.
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    return torch.from_numpy(x).permute(2, 0, 1).contiguous()


def _sample_video_frame_indices(*, vid_len: int, seconds: int, frames_per_second: int) -> list[int]:
    if seconds <= 0:
        raise ValueError("--seconds must be > 0")
    if frames_per_second <= 0:
        raise ValueError("--frames-per-second must be > 0")
    if vid_len <= 0:
        raise ValueError("video has no frames")

    frame_interval = float(vid_len) / float(seconds)
    idxs: list[int] = []
    for sec in range(int(seconds)):
        for i in range(int(frames_per_second)):
            n = int(sec * frame_interval + (float(i) / float(frames_per_second)) * frame_interval)
            if n < 0:
                n = 0
            if n >= vid_len:
                n = vid_len - 1
            idxs.append(int(n))
    return idxs


def _parse_ave_annotations(path: Path) -> list[dict]:
    clips: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        parts = line.split("&")
        if len(parts) != 5:
            raise ValueError(f"unexpected annotation line format: {line!r}")
        label, video_id, quality, start, end = parts
        try:
            float(start)
            float(end)
        except Exception:
            # Header row.
            continue
        clips.append(
            {
                "label": str(label).strip(),
                "video_id": str(video_id).strip(),
                "quality": str(quality).strip(),
                "start_sec": float(start),
                "end_sec": float(end),
            }
        )
    return clips


def _load_cace_model(
    *,
    cace_root: Path,
    weights_path: Path,
    device: torch.device,
    guide: str,
    psai: float,
    contrastive: bool,
    lambda_: float,
) -> torch.nn.Module:
    cace_root = cace_root.resolve()
    sys.path.insert(0, str(cace_root))

    cfg_path = cace_root / "configs" / "main.json"
    model_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))["model"]

    if device.type == "cuda":
        torch.cuda.set_device(int(device.index or 0))

    try:
        import braincog.model_zoo.fc_snn  # type: ignore  # noqa: F401
    except Exception:
        _install_braincog_stub()

    from model.main_model import supv_main_model  # type: ignore

    model = supv_main_model(model_cfg, psai=float(psai), guide=str(guide), contrastive=bool(contrastive), Lambda=float(lambda_))

    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"unexpected weights type: {type(state)}")
    cleaned = {}
    for k, v in state.items():
        if not isinstance(k, str):
            continue
        kk = k[7:] if k.startswith("module.") else k
        cleaned[kk] = v

    model.load_state_dict(cleaned, strict=True)
    model = model.to(device=device, dtype=torch.float32)
    model.eval()
    return model


def main() -> int:
    p = argparse.ArgumentParser(description="Export CACE-Net per-second AVE eventness logits to scores JSON (for AVE-P0 Stage-1).")
    p.add_argument("--meta-dir", type=Path, default=Path("data/AVE/meta"))
    p.add_argument("--annotations", type=Path, default=None, help="Optional override; default: <meta-dir>/Annotations.txt")
    p.add_argument("--audio-h5", type=Path, default=Path("data/AVE/eccv18_features/audio_feature.h5"))
    p.add_argument("--visual-h5", type=Path, default=Path("data/AVE/eccv18_features/visual_feature.h5"))
    p.add_argument(
        "--visual-source",
        type=str,
        default="h5",
        choices=["h5", "processed_frames", "raw_video_sample16"],
        help="How to supply visual features: 'h5' reads --visual-h5['avadataset']; "
        "'processed_frames' extracts VGG19 block5_pool from <processed-dir>/<video_id>/frames/{0..9}.jpg; "
        "'raw_video_sample16' extracts per-second features by averaging 16 sampled frames per second from raw videos.",
    )
    p.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/processed"),
        help="Processed dir containing <video_id>/frames/{0..9}.jpg (required when --visual-source=processed_frames).",
    )
    p.add_argument(
        "--raw-videos-dir",
        type=Path,
        default=Path("data/AVE/raw/videos"),
        help="Raw video directory containing <video_id>.mp4 (required when --visual-source=raw_video_sample16).",
    )
    p.add_argument("--seconds", type=int, default=10, help="Video length in seconds for sampling (AVE clips are 10s).")
    p.add_argument("--frames-per-second", type=int, default=16, help="Number of frames to sample per second (raw_video_sample16).")
    p.add_argument("--weights", type=Path, required=True, help="Path to a CACE-Net .pth.tar (state_dict).")
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--vgg-batch-size", type=int, default=128, help="Batch size for VGG19 frame feature extraction.")
    p.add_argument("--max-items", type=int, default=None, help="Optional cap on number of annotation rows (debug only).")
    p.add_argument("--aggregate", type=str, choices=["max"], default="max", help="How to aggregate multi-label duplicate video_ids.")

    # These must match the weights.
    p.add_argument("--guide", type=str, default="Co-Guide")
    p.add_argument("--psai", type=float, default=0.3)
    p.add_argument("--contrastive", action="store_true", default=True)
    p.add_argument("--lambda", dest="lambda_", type=float, default=0.6)

    args = p.parse_args()

    annotations_path = args.annotations if args.annotations is not None else (args.meta_dir / "Annotations.txt")
    clips = _parse_ave_annotations(annotations_path)
    if not clips:
        raise SystemExit(f"no clips parsed from {annotations_path}")

    cace_root = Path(__file__).resolve().parent.parent / "third_party" / "CACE-Net" / "CACE"
    if not cace_root.exists():
        raise SystemExit(f"missing CACE-Net repo at {cace_root} (expected third_party/CACE-Net)")

    device = torch.device(str(args.device))
    model = _load_cace_model(
        cace_root=cace_root,
        weights_path=args.weights,
        device=device,
        guide=args.guide,
        psai=float(args.psai),
        contrastive=bool(args.contrastive),
        lambda_=float(args.lambda_),
    )

    from PIL import Image

    if str(args.visual_source) in ("processed_frames", "raw_video_sample16"):
        if not args.processed_dir.exists():
            if str(args.visual_source) == "processed_frames":
                raise SystemExit(f"--processed-dir not found: {args.processed_dir}")
        vgg, vgg_transform = _load_vgg19_block5_pool(device=device)
    else:
        vgg = None
        vgg_transform = None

    with h5py.File(args.audio_h5, "r") as fa:
        a_ds = fa["avadataset"]
        n = int(len(a_ds))
        if n != int(len(clips)):
            raise SystemExit(f"feature length mismatch: audio={len(a_ds)} annotations={len(clips)} (wrong Annotations.txt?)")

        idxs = list(range(n))
        if args.max_items is not None:
            idxs = idxs[: int(args.max_items)]

        # Aggregate duplicate video_ids (multi-label videos) by per-segment max of logits.
        scores_by_vid: dict[str, np.ndarray] = {}

        bs = int(args.batch_size)
        vgg_bs = int(args.vgg_batch_size)

        def _consume_batch(*, batch_idxs: list[int], v_t: torch.Tensor, a_t: torch.Tensor) -> None:
            with torch.no_grad():
                is_evt, _, _, _, _, _ = model(v_t, a_t)

            # is_evt: [T,B,1] (as in supv_main.py) -> [B,T]
            if is_evt.ndim == 3:
                s = is_evt.squeeze(-1).transpose(0, 1)
            elif is_evt.ndim == 2:
                s = is_evt.transpose(0, 1)
            else:
                raise RuntimeError(f"unexpected is_event_scores shape: {tuple(is_evt.shape)}")
            s_np = s.detach().cpu().numpy().astype(np.float32, copy=False)

            for j, idx in enumerate(batch_idxs):
                vid = str(clips[int(idx)]["video_id"])
                cur = s_np[int(j)]
                prev = scores_by_vid.get(vid)
                if prev is None:
                    scores_by_vid[vid] = cur
                else:
                    scores_by_vid[vid] = np.maximum(prev, cur)

        if str(args.visual_source) == "h5":
            with h5py.File(args.visual_h5, "r") as fv:
                v_ds = fv["avadataset"]
                if int(len(v_ds)) != n:
                    raise SystemExit(f"feature length mismatch: audio={len(a_ds)} visual={len(v_ds)}")

                for i0 in range(0, len(idxs), bs):
                    batch_idxs = idxs[i0 : i0 + bs]
                    a_np = np.asarray(a_ds[batch_idxs], dtype=np.float32)
                    v_np = np.asarray(v_ds[batch_idxs], dtype=np.float32)
                    a_t = torch.from_numpy(a_np).to(device=device, dtype=torch.float32, non_blocking=True)
                    v_t = torch.from_numpy(v_np).to(device=device, dtype=torch.float32, non_blocking=True)
                    _consume_batch(batch_idxs=batch_idxs, v_t=v_t, a_t=a_t)

                    if (i0 + len(batch_idxs)) % 500 == 0 or (i0 + len(batch_idxs)) == len(idxs):
                        print(
                            f"[cace export] {i0+len(batch_idxs)}/{len(idxs)} rows; unique_vids={len(scores_by_vid)}",
                            flush=True,
                        )
        else:
            if vgg is None or vgg_transform is None:
                raise RuntimeError("internal error: vgg not initialized for visual_source!=h5")

            for i0 in range(0, len(idxs), bs):
                batch_idxs = idxs[i0 : i0 + bs]
                a_np = np.asarray(a_ds[batch_idxs], dtype=np.float32)
                a_t = torch.from_numpy(a_np).to(device=device, dtype=torch.float32, non_blocking=True)

                v_list: list[torch.Tensor] = []
                for idx in batch_idxs:
                    vid = str(clips[int(idx)]["video_id"])
                    if str(args.visual_source) == "processed_frames":
                        frame_tensors: list[torch.Tensor] = []
                        frames_dir = args.processed_dir / vid / "frames"
                        for t in range(10):
                            fp = frames_dir / f"{t}.jpg"
                            if not fp.is_file():
                                raise FileNotFoundError(f"missing frame: {fp}")
                            with Image.open(fp) as img:
                                frame_tensors.append(vgg_transform(img.convert("RGB")))

                        x = torch.stack(frame_tensors, dim=0)
                        feats: list[torch.Tensor] = []
                        for j0 in range(0, int(x.shape[0]), vgg_bs):
                            xb = x[j0 : j0 + vgg_bs].to(device=device, dtype=torch.float32, non_blocking=True)
                            with torch.no_grad():
                                yb = vgg(xb)
                            feats.append(yb)
                        y = torch.cat(feats, dim=0)  # [10, 512, 7, 7]
                        v_one = y.view(10, 512, 7, 7).permute(0, 2, 3, 1).contiguous()
                    else:
                        import cv2

                        if not args.raw_videos_dir.exists():
                            raise SystemExit(f"--raw-videos-dir not found: {args.raw_videos_dir}")
                        video_path = args.raw_videos_dir / f"{vid}.mp4"
                        if not video_path.is_file():
                            raise FileNotFoundError(f"missing raw video: {video_path}")

                        cap = cv2.VideoCapture(str(video_path))
                        try:
                            vid_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            frame_num = _sample_video_frame_indices(
                                vid_len=vid_len, seconds=int(args.seconds), frames_per_second=int(args.frames_per_second)
                            )
                            need_set = set(frame_num)
                            picked: dict[int, torch.Tensor] = {}

                            fi = 0
                            while True:
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                if fi in need_set and fi not in picked:
                                    picked[fi] = _preprocess_bgr_frame_to_tensor(frame)
                                    if len(picked) == len(need_set):
                                        break
                                fi += 1
                        finally:
                            cap.release()

                        if len(picked) != len(need_set):
                            missing = sorted(need_set.difference(picked.keys()))
                            raise RuntimeError(f"failed to sample {len(missing)} frames from {video_path} (e.g. {missing[:5]})")

                        # Preserve duplicates / ordering.
                        x = torch.stack([picked[i] for i in frame_num], dim=0)  # [T*F,3,224,224]
                        feats: list[torch.Tensor] = []
                        for j0 in range(0, int(x.shape[0]), vgg_bs):
                            xb = x[j0 : j0 + vgg_bs].to(device=device, dtype=torch.float32, non_blocking=True)
                            with torch.no_grad():
                                yb = vgg(xb)
                            feats.append(yb)
                        y = torch.cat(feats, dim=0)  # [T*F, 512, 7, 7]
                        sec = int(args.seconds)
                        fps = int(args.frames_per_second)
                        y = y.view(sec, fps, 512, 7, 7).mean(dim=1)  # [T,512,7,7]
                        v_one = y.permute(0, 2, 3, 1).contiguous()

                    v_list.append(v_one)

                v_t = torch.stack(v_list, dim=0)  # [B,10,7,7,512]

                _consume_batch(batch_idxs=batch_idxs, v_t=v_t, a_t=a_t)

                if (i0 + len(batch_idxs)) % 500 == 0 or (i0 + len(batch_idxs)) == len(idxs):
                    print(f"[cace export] {i0+len(batch_idxs)}/{len(idxs)} rows; unique_vids={len(scores_by_vid)}", flush=True)

    payload = {
        "ok": True,
        "eventness_method": "cace_net_evt",
        "num_segments": 10,
        "source": {
            "model": "CACE-Net (MM'24) supervised",
            "weights": str(args.weights),
            "audio_h5": str(args.audio_h5),
            "visual_source": str(args.visual_source),
            "visual_h5": str(args.visual_h5) if str(args.visual_source) == "h5" else None,
            "processed_dir": str(args.processed_dir) if str(args.visual_source) == "processed_frames" else None,
            "raw_videos_dir": str(args.raw_videos_dir) if str(args.visual_source) == "raw_video_sample16" else None,
            "seconds": int(args.seconds) if str(args.visual_source) == "raw_video_sample16" else None,
            "frames_per_second": int(args.frames_per_second) if str(args.visual_source) == "raw_video_sample16" else None,
            "annotations": str(annotations_path),
            "guide": str(args.guide),
            "psai": float(args.psai),
            "contrastive": bool(args.contrastive),
            "lambda": float(args.lambda_),
        },
        "scores": {k: [float(x) for x in v.tolist()] for k, v in sorted(scores_by_vid.items(), key=lambda kv: kv[0])},
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
