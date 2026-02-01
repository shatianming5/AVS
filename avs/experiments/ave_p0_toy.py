from __future__ import annotations

import argparse
import json
import math
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from avs.audio.eventness import compute_eventness_wav_energy, topk_anchors
from avs.models.per_segment_mlp import PerSegmentMLP
from avs.sampling.plans import SamplingPlan, equal_token_budget_anchored_plan
from avs.train.train_loop import TrainConfig, train_per_segment_classifier
from avs.vision.clip_vit import ClipVisionEncoder, ClipVisionEncoderConfig


@dataclass(frozen=True)
class ToyConfig:
    num_clips: int = 32
    num_segments: int = 10
    num_events_per_clip: int = 2
    seed: int = 0


def _write_wav_with_eventness(path: Path, *, sr: int = 16000, num_segments: int = 10, event_segments: tuple[int, ...]) -> None:
    dur_s = num_segments
    amps = [0.05] * dur_s
    for t in event_segments:
        amps[int(t)] = 0.8

    tt = np.arange(sr * dur_s, dtype=np.float32) / sr
    wave_data = np.zeros_like(tt, dtype=np.float32)
    for sec in range(dur_s):
        mask = (tt >= sec) & (tt < sec + 1)
        wave_data[mask] = amps[sec] * np.sin(2 * math.pi * 440.0 * tt[mask])

    pcm16 = np.clip(wave_data * 32767.0, -32768, 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())


def _make_frame(color: tuple[int, int, int], *, size: int = 256) -> Image.Image:
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[:, :] = np.array(color, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def build_toy_processed_dir(
    cfg: ToyConfig, out_dir: Path
) -> tuple[list[str], dict[str, Path], dict[str, tuple[int, ...]]]:
    """
    Builds:
      out_dir/<clip_id>/audio.wav
      out_dir/<clip_id>/frames/{0..9}.jpg
    Returns:
      clip_ids, audio_paths
    """
    rng = np.random.default_rng(cfg.seed)
    clip_ids: list[str] = []
    audio_paths: dict[str, Path] = {}
    events_by_clip: dict[str, tuple[int, ...]] = {}

    for i in range(cfg.num_clips):
        clip_id = f"toy_{i:04d}"
        clip_ids.append(clip_id)
        clip_dir = out_dir / clip_id
        frames_dir = clip_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        # Randomize colors a bit per clip so embeddings are not identical.
        bg_color = tuple(int(x) for x in rng.integers(0, 40, size=3))
        ev_color = tuple(int(x) for x in rng.integers(200, 255, size=3))

        events = tuple(sorted(rng.choice(cfg.num_segments, size=cfg.num_events_per_clip, replace=False).tolist()))
        events_by_clip[clip_id] = events

        for t in range(cfg.num_segments):
            im = _make_frame(ev_color if t in events else bg_color, size=256)
            im.save(frames_dir / f"{t}.jpg", quality=95)

        wav_path = clip_dir / "audio.wav"
        _write_wav_with_eventness(wav_path, num_segments=cfg.num_segments, event_segments=events)
        audio_paths[clip_id] = wav_path

    return clip_ids, audio_paths, events_by_clip


def _labels_for_toy(cfg: ToyConfig, clip_ids: list[str], events_by_clip: dict[str, tuple[int, ...]]) -> torch.Tensor:
    y = torch.zeros(len(clip_ids), cfg.num_segments, dtype=torch.long)
    for i, cid in enumerate(clip_ids):
        for t in events_by_clip[cid]:
            y[i, int(t)] = 1
    return y


def _extract_features(
    *,
    encoder: ClipVisionEncoder,
    processed_dir: Path,
    clip_ids: list[str],
    plans_by_clip: dict[str, SamplingPlan],
    events_by_clip: dict[str, tuple[int, ...]],
    device: torch.device,
    noise_mode: str,
    seed: int,
) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    feats: list[torch.Tensor] = []

    def _noise_sigma(res: int, *, is_event: bool) -> float:
        if noise_mode == "none":
            return 0.0
        if noise_mode == "resolution":
            return {112: 0.6, 224: 0.35, 448: 0.15}.get(res, 0.35)
        if noise_mode == "event_resolution":
            return 0.08 if not is_event else {112: 0.9, 224: 0.5, 448: 0.2}.get(res, 0.5)
        raise ValueError(f"unknown noise_mode: {noise_mode}")

    for clip_id in clip_ids:
        plan = plans_by_clip[clip_id]
        event_set = set(events_by_clip[clip_id])
        frames_dir = processed_dir / clip_id / "frames"
        images = [Image.open(frames_dir / f"{t}.jpg") for t in range(len(plan.resolutions))]
        try:
            per_t = torch.empty(len(plan.resolutions), encoder.encode([images[0]], resolution=plan.resolutions[0]).shape[-1])
        finally:
            for im in images:
                im.close()

        # Re-open to avoid holding file handles across multiple encodes.
        for t, res in enumerate(plan.resolutions):
            with Image.open(frames_dir / f"{t}.jpg") as im:
                emb = encoder.encode([im], resolution=res)[0]
            sigma = _noise_sigma(res, is_event=t in event_set)
            if sigma > 0:
                emb = emb + torch.from_numpy(rng.standard_normal(size=emb.shape).astype(np.float32)) * float(sigma)
            per_t[t] = emb
        feats.append(per_t)

    x = torch.stack(feats, dim=0).to(device)
    return x


def run_toy_p0(
    *,
    cfg: ToyConfig,
    out_dir: Path,
    device: str = "cpu",
    noise_mode: str = "event_resolution",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = out_dir / "processed"
    clip_ids, audio_paths, events_by_clip = build_toy_processed_dir(cfg, processed_dir)
    y = _labels_for_toy(cfg, clip_ids, events_by_clip)

    dev = torch.device(device)
    enc = ClipVisionEncoder(ClipVisionEncoderConfig(pretrained=False, device=device, dtype="float32"))

    # Anchors from audio energy.
    anchors_by_clip: dict[str, list[int]] = {}
    for clip_id in clip_ids:
        ev = compute_eventness_wav_energy(audio_paths[clip_id], num_segments=cfg.num_segments)
        anchors_by_clip[clip_id] = topk_anchors(ev.scores, k=2)

    # Sampling plans (equal token budget by construction). Anchored is per-clip.
    plans_uniform = {cid: equal_token_budget_anchored_plan(num_segments=cfg.num_segments, anchors=[]) for cid in clip_ids}
    plans_random = {cid: equal_token_budget_anchored_plan(num_segments=cfg.num_segments, anchors=[0, 1]) for cid in clip_ids}
    plans_anchored = {
        cid: equal_token_budget_anchored_plan(num_segments=cfg.num_segments, anchors=anchors_by_clip[cid]) for cid in clip_ids
    }

    # Extract features (note: plan affects resolution -> affects noise injection).
    x_uniform = _extract_features(
        encoder=enc,
        processed_dir=processed_dir,
        clip_ids=clip_ids,
        plans_by_clip=plans_uniform,
        events_by_clip=events_by_clip,
        device=dev,
        noise_mode=noise_mode,
        seed=cfg.seed,
    )
    x_random = _extract_features(
        encoder=enc,
        processed_dir=processed_dir,
        clip_ids=clip_ids,
        plans_by_clip=plans_random,
        events_by_clip=events_by_clip,
        device=dev,
        noise_mode=noise_mode,
        seed=cfg.seed,
    )
    x_anchored = _extract_features(
        encoder=enc,
        processed_dir=processed_dir,
        clip_ids=clip_ids,
        plans_by_clip=plans_anchored,
        events_by_clip=events_by_clip,
        device=dev,
        noise_mode=noise_mode,
        seed=cfg.seed,
    )

    split = int(0.8 * len(clip_ids))
    y_train, y_val = y[:split].to(dev), y[split:].to(dev)

    def _train_eval(name: str, x: torch.Tensor) -> float:
        x_train, x_val = x[:split], x[split:]
        model = PerSegmentMLP(in_dim=x.shape[-1], num_classes=2, hidden_dim=128).to(dev)
        m = train_per_segment_classifier(
            model=model,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            cfg=TrainConfig(epochs=12, batch_size=16, lr=2e-3),
        )
        return float(m["val_acc"])

    metrics = {
        "uniform": _train_eval("uniform", x_uniform),
        "random_top2": _train_eval("random_top2", x_random),
        "anchored_top2": _train_eval("anchored_top2", x_anchored),
    }

    payload = {
        "toy_cfg": cfg.__dict__,
        "noise_mode": noise_mode,
        "anchors_by_clip_sample": {k: anchors_by_clip[k] for k in clip_ids[:3]},
        "events_by_clip_sample": {k: list(events_by_clip[k]) for k in clip_ids[:3]},
        "plans": {
            "uniform": plans_uniform[clip_ids[0]].to_jsonable(),
            "random_top2": plans_random[clip_ids[0]].to_jsonable(),
            "anchored_top2": plans_anchored[clip_ids[0]].to_jsonable(),
        },
        "metrics": metrics,
    }
    out_path = out_dir / "metrics.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return out_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Toy AVE-P0 runner (processed frames+audio, equal-token baselines).")
    p.add_argument("--out-dir", type=Path, default=Path("runs") / f"AVE_P0_toy_{time.strftime('%Y%m%d-%H%M%S')}")
    p.add_argument("--num-clips", type=int, default=32)
    p.add_argument("--num-events-per-clip", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--noise-mode", type=str, default="event_resolution", choices=["none", "resolution", "event_resolution"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = ToyConfig(num_clips=args.num_clips, num_events_per_clip=args.num_events_per_clip, seed=args.seed)
    out_path = run_toy_p0(cfg=cfg, out_dir=args.out_dir, device=args.device, noise_mode=args.noise_mode)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
