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


@dataclass(frozen=True)
class AVQAPaths:
    """
    Default AVQA layout.

    - Metadata: data/AVQA/meta/{train_qa.json,val_qa.json}
    - Raw clips: data/AVQA/raw/videos/<video_name>.mp4
    - Processed: data/AVQA/processed/<video_name>/{audio.wav,frames/*.jpg}
    """

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

    def raw_video_path(self, video_name: str) -> Path:
        return self.raw_videos_dir / f"{str(video_name)}.mp4"

    def processed_video_dir(self, video_name: str) -> Path:
        return self.processed_dir / str(video_name)


def avqa_paths() -> AVQAPaths:
    return AVQAPaths(root=data_dir() / "AVQA")


@dataclass(frozen=True)
class IntentQAPaths:
    """
    Default IntentQA layout.

    We support both:
      - data/IntentQA/IntentQA/... (preferred canonical path), and
      - data/hf_repos/IntentQA/IntentQA/... (common when cloned from HF).
    """

    root: Path

    @property
    def intentqa_dir(self) -> Path:
        return self.root / "IntentQA"

    @property
    def videos_dir(self) -> Path:
        return self.intentqa_dir / "videos"

    @property
    def processed_dir(self) -> Path:
        return self.root / "processed"

    def split_csv_path(self, split: str) -> Path:
        return self.intentqa_dir / f"{str(split)}.csv"

    def raw_video_path(self, video_id: str) -> Path:
        return self.videos_dir / f"{str(video_id)}.mp4"

    def processed_video_dir(self, video_id: str) -> Path:
        return self.processed_dir / str(video_id)


def intentqa_paths() -> IntentQAPaths:
    d = data_dir() / "IntentQA"
    if (d / "IntentQA").exists():
        return IntentQAPaths(root=d)
    # Common local clone location (gitignored).
    alt = data_dir() / "hf_repos" / "IntentQA"
    return IntentQAPaths(root=alt)


@dataclass(frozen=True)
class EgoSchemaPaths:
    """
    Default EgoSchema layout.

    - HF repo clone (parquet metadata + zip shards): data/hf_repos/egoschema/
    - Extracted videos: data/EgoSchema/videos/*.mp4
    """

    hf_repo_dir: Path
    root: Path

    @property
    def videos_dir(self) -> Path:
        return self.root / "videos"

    @property
    def processed_dir(self) -> Path:
        return self.root / "processed"

    def processed_video_dir(self, video_idx: str) -> Path:
        return self.processed_dir / str(video_idx)

    def video_path(self, video_idx: str) -> Path:
        return self.videos_dir / f"{str(video_idx)}.mp4"


def egoschema_paths() -> EgoSchemaPaths:
    return EgoSchemaPaths(hf_repo_dir=data_dir() / "hf_repos" / "egoschema", root=data_dir() / "EgoSchema")
