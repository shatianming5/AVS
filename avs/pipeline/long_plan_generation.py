from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from avs.audio.eventness import anchors_from_scores, compute_eventness_wav_energy, compute_eventness_wav_energy_delta
from avs.pipeline.plan_generation import infer_num_segments_from_wav
from avs.sampling.plans import SamplingPlan, equal_token_budget_anchored_plan

_DEFAULT_SEGMENT_SECONDS: Final[float] = 1.0


@dataclass(frozen=True)
class LongPlanRecord:
    clip_id: str
    wav_path: Path
    duration_seconds: int
    anchors_seconds: list[int]
    scores: list[float]
    selected_seconds: list[int]
    plan: SamplingPlan

    def to_jsonable(self) -> dict:
        return {
            "clip_id": self.clip_id,
            "wav_path": str(self.wav_path),
            "duration_seconds": int(self.duration_seconds),
            "anchors_seconds": [int(x) for x in self.anchors_seconds],
            "scores": [float(x) for x in self.scores],
            "selected_seconds": [int(x) for x in self.selected_seconds],
            "plan": self.plan.to_jsonable(),
        }

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_jsonable(), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def select_seconds_hybrid(
    *,
    duration_seconds: int,
    anchors_seconds: list[int],
    anchor_radius: int = 2,
    background_stride: int = 5,
    max_steps: int = 120,
) -> list[int]:
    """
    Select a fixed-size set of seconds from a long video using a hybrid policy:
      - Always include a dense window around audio anchors (anchor_radius).
      - Add uniformly spaced background seconds (background_stride).
      - Fill remaining slots with earliest remaining seconds for determinism.

    Returns: sorted-by-priority list of unique seconds (length == min(max_steps, duration_seconds)).
    """
    dur = max(0, int(duration_seconds))
    if dur <= 0:
        return []

    radius = max(0, int(anchor_radius))
    stride = max(1, int(background_stride))
    target = max(1, int(max_steps))
    target = min(target, dur)

    anchors: list[int] = []
    for a in anchors_seconds:
        a = int(a)
        if 0 <= a < dur and a not in anchors:
            anchors.append(a)

    anchor_window: set[int] = set()
    for a in anchors:
        for t in range(a - radius, a + radius + 1):
            if 0 <= t < dur:
                anchor_window.add(int(t))

    background = list(range(0, dur, stride))
    background_set = set(int(t) for t in background)

    anchor_list = sorted(anchor_window)
    background_list = [int(t) for t in background if int(t) not in anchor_window]
    remaining_list = [int(t) for t in range(dur) if int(t) not in anchor_window and int(t) not in background_set]

    out: list[int] = []
    out_set: set[int] = set()

    for seq in (anchor_list, background_list, remaining_list):
        for t in seq:
            if t not in out_set:
                out.append(int(t))
                out_set.add(int(t))
            if len(out) >= target:
                break
        if len(out) >= target:
            break

    if len(out) != target:
        raise AssertionError(f"bug: expected {target} selected seconds, got {len(out)}")
    return out


def long_plan_from_wav_hybrid(
    *,
    clip_id: str,
    wav_path: Path,
    max_seconds: int | None = None,
    segment_seconds: float = _DEFAULT_SEGMENT_SECONDS,
    eventness_method: str = "energy",
    audio_device: str = "cpu",
    k: int = 10,
    anchor_radius: int = 2,
    background_stride: int = 5,
    max_steps: int = 120,
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
) -> LongPlanRecord:
    duration_seconds = infer_num_segments_from_wav(wav_path, segment_seconds=float(segment_seconds))
    if max_seconds is not None:
        duration_seconds = min(int(duration_seconds), int(max_seconds))
    duration_seconds = max(0, int(duration_seconds))
    if duration_seconds <= 0:
        raise ValueError(f"invalid duration_seconds={duration_seconds} for wav: {wav_path}")

    if eventness_method == "energy":
        ev = compute_eventness_wav_energy(wav_path, num_segments=duration_seconds)
        scores = ev.scores
    elif eventness_method == "energy_delta":
        ev = compute_eventness_wav_energy_delta(wav_path, num_segments=duration_seconds)
        scores = ev.scores
    elif eventness_method == "ast":
        from avs.audio.ast_probe import ASTEventnessProbe, ASTProbeConfig

        probe = ASTEventnessProbe(ASTProbeConfig(pretrained=ast_pretrained, device=str(audio_device)))
        scores = probe.eventness_per_second(wav_path, num_segments=duration_seconds)
    elif eventness_method == "panns":
        from avs.audio.panns_probe import PANNsEventnessProbe, PANNsProbeConfig

        probe = PANNsEventnessProbe(
            PANNsProbeConfig(pretrained=not panns_random, checkpoint_path=panns_checkpoint, device=str(audio_device))
        )
        scores = probe.eventness_per_second(wav_path, num_segments=duration_seconds)
    elif eventness_method == "audiomae":
        from avs.audio.audiomae_probe import AudioMAEEventnessProbe, AudioMAEProbeConfig

        probe = AudioMAEEventnessProbe(
            AudioMAEProbeConfig(
                pretrained=(audiomae_checkpoint is not None) and (not audiomae_random),
                checkpoint_path=audiomae_checkpoint if (audiomae_checkpoint is not None) and (not audiomae_random) else None,
                device=str(audio_device),
            )
        )
        scores = probe.eventness_per_second(wav_path, num_segments=duration_seconds)
    else:
        raise ValueError(f"unsupported eventness_method: {eventness_method}")

    scores = [float(x) for x in scores]
    anchors_seconds = anchors_from_scores(
        scores,
        k=int(k),
        num_segments=int(duration_seconds),
        shift=int(anchor_shift),
        std_threshold=float(anchor_std_threshold),
    )

    selected_seconds = select_seconds_hybrid(
        duration_seconds=int(duration_seconds),
        anchors_seconds=anchors_seconds,
        anchor_radius=int(anchor_radius),
        background_stride=int(background_stride),
        max_steps=int(max_steps),
    )

    anchor_set = set(int(x) for x in anchors_seconds)
    anchor_positions = [i for i, t in enumerate(selected_seconds) if int(t) in anchor_set]

    plan = equal_token_budget_anchored_plan(
        num_segments=len(selected_seconds),
        anchors=anchor_positions,
        low_res=int(low_res),
        base_res=int(base_res),
        high_res=int(high_res),
        patch_size=int(patch_size),
    )

    return LongPlanRecord(
        clip_id=str(clip_id),
        wav_path=wav_path,
        duration_seconds=int(duration_seconds),
        anchors_seconds=[int(x) for x in anchors_seconds],
        scores=[float(x) for x in scores],
        selected_seconds=[int(x) for x in selected_seconds],
        plan=plan,
    )
