from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from avs.utils.paths import data_dir


@dataclass(frozen=True)
class AVEPaths:
    root: Path

    @property
    def meta_dir(self) -> Path:
        return self.root / "meta"

    @property
    def raw_videos_dir(self) -> Path:
        return self.root / "raw" / "videos"

    @property
    def processed_dir(self) -> Path:
        return self.root / "processed"

    def raw_video_path(self, video_id: str) -> Path:
        return self.raw_videos_dir / f"{video_id}.mp4"

    def processed_clip_dir(self, video_id: str) -> Path:
        return self.processed_dir / video_id

    def processed_audio_path(self, video_id: str) -> Path:
        return self.processed_clip_dir(video_id) / "audio.wav"

    def processed_frames_dir(self, video_id: str) -> Path:
        return self.processed_clip_dir(video_id) / "frames"

    def processed_frame_path(self, video_id: str, t: int) -> Path:
        return self.processed_frames_dir(video_id) / f"{int(t)}.jpg"


def ave_paths() -> AVEPaths:
    return AVEPaths(root=data_dir() / "AVE")


@dataclass(frozen=True)
class EpicSoundsPaths:
    root: Path

    @property
    def meta_dir(self) -> Path:
        return self.root / "meta"

    @property
    def raw_videos_dir(self) -> Path:
        return self.root / "raw" / "videos"

    @property
    def audio_dir(self) -> Path:
        return self.root / "audio"

    @property
    def frames_dir(self) -> Path:
        return self.root / "frames"

    @property
    def plans_dir(self) -> Path:
        return self.root / "plans"

    @property
    def caches_dir(self) -> Path:
        return self.root / "caches"

    @property
    def selected_frames_dir(self) -> Path:
        return self.root / "selected_frames"

    def raw_video_path(self, video_id: str) -> Path:
        return self.raw_videos_dir / f"{video_id}.mp4"

    def audio_path(self, video_id: str) -> Path:
        return self.audio_dir / f"{video_id}.wav"

    def frames_video_dir(self, video_id: str) -> Path:
        return self.frames_dir / video_id / "frames"

    def long_plan_path(self, video_id: str) -> Path:
        return self.plans_dir / f"{video_id}.long_plan.json"

    def cache_path(self, video_id: str) -> Path:
        return self.caches_dir / f"{video_id}.npz"


def epic_sounds_paths() -> EpicSoundsPaths:
    return EpicSoundsPaths(root=data_dir() / "EPIC_SOUNDS")


@dataclass(frozen=True)
class VGGSoundPaths:
    root: Path

    @property
    def meta_dir(self) -> Path:
        return self.root / "meta"


def vggsound_paths() -> VGGSoundPaths:
    return VGGSoundPaths(root=data_dir() / "VGGSound")
