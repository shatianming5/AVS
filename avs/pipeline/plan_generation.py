from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from avs.audio.eventness import anchors_from_scores, compute_eventness_wav_energy, compute_eventness_wav_energy_delta
from avs.sampling.plans import SamplingPlan, equal_token_budget_anchored_plan

_DEFAULT_SEGMENT_SECONDS: Final[float] = 1.0


@dataclass(frozen=True)
class PlanRecord:
    clip_id: str
    anchors: list[int]
    scores: list[float]
    plan: SamplingPlan

    def to_jsonable(self) -> dict:
        return {
            "clip_id": self.clip_id,
            "anchors": self.anchors,
            "scores": self.scores,
            "plan": self.plan.to_jsonable(),
        }


def plan_from_wav(
    *,
    clip_id: str,
    wav_path: Path,
    eventness_method: str = "energy",
    audio_device: str = "cpu",
    k: int = 2,
    num_segments: int = 10,
    low_res: int = 112,
    base_res: int = 224,
    high_res: int = 448,
    patch_size: int = 16,
    ast_pretrained: bool = False,
    panns_random: bool = False,
    panns_checkpoint: Path | None = None,
    audiomae_random: bool = False,
    audiomae_checkpoint: Path | None = None,
    anchor_shift: int = 0,
    anchor_std_threshold: float = 0.0,
) -> PlanRecord:
    if eventness_method == "energy":
        ev = compute_eventness_wav_energy(wav_path, num_segments=num_segments)
        scores = ev.scores
    elif eventness_method == "energy_delta":
        ev = compute_eventness_wav_energy_delta(wav_path, num_segments=num_segments)
        scores = ev.scores
    elif eventness_method == "ast":
        from avs.audio.ast_probe import ASTEventnessProbe, ASTProbeConfig

        probe = ASTEventnessProbe(ASTProbeConfig(pretrained=ast_pretrained, device=str(audio_device)))
        scores = probe.eventness_per_second(wav_path, num_segments=num_segments)
    elif eventness_method == "panns":
        from avs.audio.panns_probe import PANNsEventnessProbe, PANNsProbeConfig

        probe = PANNsEventnessProbe(
            PANNsProbeConfig(pretrained=not panns_random, checkpoint_path=panns_checkpoint, device=str(audio_device))
        )
        scores = probe.eventness_per_second(wav_path, num_segments=num_segments)
    elif eventness_method == "audiomae":
        from avs.audio.audiomae_probe import AudioMAEEventnessProbe, AudioMAEProbeConfig

        probe = AudioMAEEventnessProbe(
            AudioMAEProbeConfig(
                pretrained=(audiomae_checkpoint is not None) and (not audiomae_random),
                checkpoint_path=audiomae_checkpoint if (audiomae_checkpoint is not None) and (not audiomae_random) else None,
                device=str(audio_device),
            )
        )
        scores = probe.eventness_per_second(wav_path, num_segments=num_segments)
    else:
        raise ValueError(f"unsupported eventness_method: {eventness_method}")

    anchors = anchors_from_scores(
        [float(x) for x in scores],
        k=k,
        num_segments=num_segments,
        shift=int(anchor_shift),
        std_threshold=float(anchor_std_threshold),
    )
    plan = equal_token_budget_anchored_plan(
        num_segments=num_segments,
        anchors=anchors,
        low_res=low_res,
        base_res=base_res,
        high_res=high_res,
        patch_size=patch_size,
    )
    return PlanRecord(clip_id=clip_id, anchors=anchors, scores=[float(x) for x in scores], plan=plan)


def infer_num_segments_from_wav(wav_path: Path, *, segment_seconds: float = _DEFAULT_SEGMENT_SECONDS) -> int:
    import wave

    if float(segment_seconds) <= 0.0:
        raise ValueError("segment_seconds must be positive")

    with wave.open(str(wav_path), "rb") as wf:
        nframes = int(wf.getnframes())
        framerate = int(wf.getframerate())
    if framerate <= 0:
        return 1
    duration_s = float(nframes) / float(framerate)
    return max(1, int(duration_s / float(segment_seconds)))


def plan_from_wav_auto_segments(
    *,
    clip_id: str,
    wav_path: Path,
    segment_seconds: float = _DEFAULT_SEGMENT_SECONDS,
    eventness_method: str = "energy",
    audio_device: str = "cpu",
    k: int = 2,
    low_res: int = 112,
    base_res: int = 224,
    high_res: int = 448,
    patch_size: int = 16,
    ast_pretrained: bool = False,
    panns_random: bool = False,
    panns_checkpoint: Path | None = None,
    audiomae_random: bool = False,
    audiomae_checkpoint: Path | None = None,
    anchor_shift: int = 0,
    anchor_std_threshold: float = 0.0,
) -> PlanRecord:
    num_segments = infer_num_segments_from_wav(wav_path, segment_seconds=float(segment_seconds))
    return plan_from_wav(
        clip_id=clip_id,
        wav_path=wav_path,
        eventness_method=str(eventness_method),
        audio_device=str(audio_device),
        k=int(k),
        num_segments=int(num_segments),
        low_res=int(low_res),
        base_res=int(base_res),
        high_res=int(high_res),
        patch_size=int(patch_size),
        ast_pretrained=bool(ast_pretrained),
        panns_random=bool(panns_random),
        panns_checkpoint=panns_checkpoint,
        audiomae_random=bool(audiomae_random),
        audiomae_checkpoint=audiomae_checkpoint,
        anchor_shift=int(anchor_shift),
        anchor_std_threshold=float(anchor_std_threshold),
    )


def plan_from_wav_energy(
    *,
    clip_id: str,
    wav_path: Path,
    k: int = 2,
    num_segments: int = 10,
    low_res: int = 112,
    base_res: int = 224,
    high_res: int = 448,
    patch_size: int = 16,
) -> PlanRecord:
    return plan_from_wav(
        clip_id=clip_id,
        wav_path=wav_path,
        eventness_method="energy",
        k=k,
        num_segments=num_segments,
        low_res=low_res,
        base_res=base_res,
        high_res=high_res,
        patch_size=patch_size,
    )


def write_plan_jsonl(path: Path, records: list[PlanRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_jsonable(), ensure_ascii=False) + "\n")
