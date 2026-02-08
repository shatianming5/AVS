from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetStatus:
    name: str
    required: bool
    ok: bool
    details: dict
    next_steps: list[str]

    def to_jsonable(self) -> dict:
        return {
            "name": str(self.name),
            "required": bool(self.required),
            "ok": bool(self.ok),
            "details": dict(self.details),
            "next_steps": [str(x) for x in self.next_steps],
        }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _count_mp4_dir(path: Path) -> int:
    # Follow symlinks.
    try:
        return sum(1 for _ in path.glob("*.mp4"))
    except Exception:
        return 0


def check_ave(*, root: Path) -> DatasetStatus:
    meta_dir = root / "data" / "AVE" / "meta"
    raw_videos_dir = root / "data" / "AVE" / "raw" / "videos"
    ann = meta_dir / "Annotations.txt"
    mp4_count = _count_mp4_dir(raw_videos_dir) if raw_videos_dir.exists() else 0

    ok = ann.exists() and raw_videos_dir.exists() and mp4_count > 0
    next_steps = []
    if not ok:
        next_steps.append("bash scripts/ave_install_official.sh")

    return DatasetStatus(
        name="AVE",
        required=True,
        ok=ok,
        details={"meta_dir": str(meta_dir), "raw_videos_dir": str(raw_videos_dir), "annotations": str(ann), "mp4_count": int(mp4_count)},
        next_steps=next_steps,
    )


def check_epic_sounds(*, root: Path) -> DatasetStatus:
    meta_dir = root / "data" / "EPIC_SOUNDS" / "meta"
    raw_videos_dir = root / "data" / "EPIC_SOUNDS" / "raw" / "videos"
    has_meta = meta_dir.exists()
    mp4_count = _count_mp4_dir(raw_videos_dir) if raw_videos_dir.exists() else 0
    ok = has_meta and mp4_count > 0

    next_steps = []
    if has_meta and mp4_count == 0:
        next_steps.append("Place EPIC-KITCHENS videos as data/EPIC_SOUNDS/raw/videos/<video_id>.mp4 (requires credentials).")
        next_steps.append("Then run: python -m avs.preprocess.epic_sounds_audio --help")
        next_steps.append("Then run: python -m avs.preprocess.epic_sounds_frames --help")

    return DatasetStatus(
        name="EPIC_SOUNDS",
        required=False,
        ok=ok,
        details={"meta_dir": str(meta_dir), "raw_videos_dir": str(raw_videos_dir), "mp4_count": int(mp4_count)},
        next_steps=next_steps,
    )


def check_egoschema(*, root: Path) -> DatasetStatus:
    d = root / "data" / "EgoSchema"
    ok = d.exists()
    return DatasetStatus(
        name="EgoSchema",
        required=False,
        ok=ok,
        details={"root": str(d)},
        next_steps=["Install EgoSchema under data/EgoSchema/ (terms may require manual download)."] if not ok else [],
    )


def check_intentqa(*, root: Path) -> DatasetStatus:
    d = root / "data" / "IntentQA"
    ok = d.exists()
    return DatasetStatus(
        name="IntentQA",
        required=False,
        ok=ok,
        details={"root": str(d)},
        next_steps=["Install IntentQA under data/IntentQA/ (terms may require manual download)."] if not ok else [],
    )


def main() -> int:
    root = _repo_root()
    out_dir = root / "runs" / f"datasets_verify_{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    statuses = [
        check_ave(root=root),
        check_epic_sounds(root=root),
        check_egoschema(root=root),
        check_intentqa(root=root),
    ]

    ok = all((not s.required) or s.ok for s in statuses)
    payload = {"ok": ok, "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "datasets": [s.to_jsonable() for s in statuses]}

    out_json = out_dir / "datasets_verify.json"
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out_json)

    # Do not hard-fail on optional datasets; keep this as a guide + log generator.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

