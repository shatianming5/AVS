from __future__ import annotations

import json
from pathlib import Path

from avs.smoke import SmokeResult


def _write_json(run_dir: Path, name: str, payload: dict) -> None:
    (run_dir / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def check_ave_meta(run_dir: Path) -> SmokeResult:
    from avs.datasets.ave import AVEIndex, ensure_ave_meta
    from avs.utils.paths import data_dir

    meta_dir = data_dir() / "AVE" / "meta"
    ensure_ave_meta(meta_dir)

    index = AVEIndex.from_meta_dir(meta_dir)
    split_sizes = {k: len(v) for k, v in index.splits.items()}

    ok = (
        len(index.clips) == 4143
        and split_sizes.get("train") == 3339
        and split_sizes.get("val") == 402
        and split_sizes.get("test") == 402
    )

    sample = index.clips[index.splits["train"][0]]
    seg = index.segment_labels(sample)
    ok = ok and len(seg) == 10 and all(0 <= x < index.num_classes for x in seg)

    payload = {
        "meta_dir": str(meta_dir),
        "num_clips": len(index.clips),
        "num_classes": index.num_classes,
        "split_sizes": split_sizes,
        "sample": {
            "video_id": sample.video_id,
            "label": sample.label,
            "start": sample.start_sec,
            "end": sample.end_sec,
            "segment_labels": seg,
        },
    }
    _write_json(run_dir, "ave_meta.json", payload)
    return SmokeResult(name="ave_meta", ok=ok, details=payload if ok else {"error": "unexpected meta counts/labels"})


def check_epic_sounds_meta(run_dir: Path) -> SmokeResult:
    from avs.datasets.epic_sounds import EpicSoundsIndex

    meta_dir = run_dir / "epic_sounds_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Minimal schema fixture based on EPIC-SOUNDS annotations repo README.
    (meta_dir / "EPIC_Sounds_train.csv").write_text(
        "\n".join(
            [
                "annotation_id,participant_id,video_id,start_timestamp,stop_timestamp,start_sample,stop_sample,description,class,class_id",
                "P01_01_0,P01,P01_01,00:00:02.466,00:00:05.315,59184,127560,paper rustle,rustle,4",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (meta_dir / "EPIC_Sounds_validation.csv").write_text(
        "\n".join(
            [
                "annotation_id,participant_id,video_id,start_timestamp,stop_timestamp,start_sample,stop_sample,description,class,class_id",
                "P02_03_0,P02,P02_03,00:01:00.000,00:01:01.000,1440000,1464000,door close,door,7",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (meta_dir / "EPIC_Sounds_recognition_test_timestamps.csv").write_text(
        "\n".join(
            [
                "annotation_id,participant_id,video_id,start_timestamp,stop_timestamp,start_sample,stop_sample",
                "P03_05_0,P03,P03_05,00:00:10.000,00:00:12.000,240000,288000",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (meta_dir / "sound_events_not_categorised.csv").write_text(
        "\n".join(
            [
                "annotation_id,participant_id,video_id,start_timestamp,stop_timestamp,start_sample,stop_sample,description",
                "P01_01_NC_0,P01,P01_01,00:00:15.000,00:00:16.000,360000,384000,unknown sound",
                "",
            ]
        ),
        encoding="utf-8",
    )

    index = EpicSoundsIndex.from_meta_dir(meta_dir)
    segs = index.segments("P01_01")

    ok = len(index.train) == 1 and len(index.val) == 1 and len(index.test) == 1 and len(index.not_categorised) == 1
    ok = ok and len(segs) == 2 and segs[0].start_sec == 2.466 and segs[0].label == "rustle" and segs[1].label is None

    payload = {
        "meta_dir": str(meta_dir),
        "counts": {"train": len(index.train), "val": len(index.val), "test": len(index.test), "not_categorised": len(index.not_categorised)},
        "sample_video_id": "P01_01",
        "segments": [
            {"annotation_id": s.annotation_id, "start_sec": s.start_sec, "stop_sec": s.stop_sec, "label": s.label, "split": s.split}
            for s in segs
        ],
    }
    _write_json(run_dir, "epic_sounds_meta.json", payload)
    return SmokeResult(name="epic_sounds_meta", ok=ok, details=payload if ok else {"error": "EPIC-SOUNDS parse mismatch", **payload})


def check_avqa_meta(run_dir: Path) -> SmokeResult:
    from avs.datasets.avqa import AVQAIndex
    from avs.prompts.contrastive import build_contrastive_prompt

    meta_dir = run_dir / "avqa_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    sample = [
        {
            "id": 1,
            "video_name": "demo_video_000000",
            "video_id": 123,
            "question_text": "What happened in the video?",
            "multi_choice": ["A", "B", "C", "D"],
            "answer": 2,
            "question_relation": "View",
            "question_type": "Happening",
        }
    ]
    (meta_dir / "train_qa.json").write_text(json.dumps(sample, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (meta_dir / "val_qa.json").write_text(json.dumps(sample, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    index = AVQAIndex.from_meta_dir(meta_dir)
    item = index.train[0]
    prompt = build_contrastive_prompt(
        question_text=item.question_text,
        choices=item.multi_choice,
        anchors_s=[6.0, 7.0],
        delta_s=1.0,
    )

    ok = len(index.train) == 1 and len(index.val) == 1 and "Audio-anchor windows" in prompt and "Question:" in prompt and "Choices:" in prompt
    payload = {"meta_dir": str(meta_dir), "prompt": prompt, "sample_video_name": item.video_name, "answer": item.answer}
    _write_json(run_dir, "avqa_meta.json", payload)
    return SmokeResult(name="avqa_meta", ok=ok, details=payload if ok else {"error": "AVQA parse/prompt mismatch", **payload})


def check_vggsound_meta(run_dir: Path) -> SmokeResult:
    from avs.datasets.vggsound import VGGSoundIndex

    meta_dir = run_dir / "vggsound_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # VGGSound has a headerless CSV: youtube_id,start_sec,label,split
    (meta_dir / "vggsound.csv").write_text(
        "\n".join(
            [
                "---g-f_I2yQ,1,people marching,test",
                "--0PQM4-hqg,30,waterfall burbling,train",
                "--56QUhyDQM,185,playing tennis,train",
                "",
            ]
        ),
        encoding="utf-8",
    )

    index = VGGSoundIndex.from_meta_dir(meta_dir)
    ok = len(index.clips) == 3 and set(index.by_split.keys()) == {"train", "test"} and len(index.by_split["train"]) == 2

    payload = {
        "meta_dir": str(meta_dir),
        "num_clips": len(index.clips),
        "splits": {k: len(v) for k, v in index.by_split.items()},
        "sample": {"youtube_id": index.clips[0].youtube_id, "start_sec": index.clips[0].start_sec, "label": index.clips[0].label, "split": index.clips[0].split},
    }
    _write_json(run_dir, "vggsound_meta.json", payload)
    return SmokeResult(name="vggsound_meta", ok=ok, details=payload if ok else {"error": "VGGSound parse mismatch", **payload})


def check_epic_sounds_audio(run_dir: Path) -> SmokeResult:
    import json as _json
    import subprocess

    from avs.preprocess.epic_sounds_audio import extract_epic_sounds_audio

    videos_dir = run_dir / "epic_videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    out_audio_dir = run_dir / "epic_audio"
    out_audio_dir.mkdir(parents=True, exist_ok=True)

    video_id = "P01_01"
    in_video = videos_dir / f"{video_id}.mp4"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=160x120:rate=25:duration=3",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=44100:duration=3",
        "-shortest",
        str(in_video),
    ]
    subprocess.run(cmd, check=True)  # noqa: S603,S607 - controlled args

    done = extract_epic_sounds_audio(videos_dir=videos_dir, out_audio_dir=out_audio_dir, video_ids=[video_id])
    wav_path = out_audio_dir / f"{video_id}.wav"

    probe = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels",
            "-of",
            "json",
            str(wav_path),
        ],
        text=True,
    )
    info = _json.loads(probe)
    stream = (info.get("streams") or [{}])[0]
    sr = int(stream.get("sample_rate", -1))
    ch = int(stream.get("channels", -1))

    ok = done == [video_id] and wav_path.exists() and sr == 16000 and ch == 1
    payload = {"videos_dir": str(videos_dir), "out_audio_dir": str(out_audio_dir), "done": done, "wav_path": str(wav_path), "sr": sr, "channels": ch}
    _write_json(run_dir, "epic_sounds_audio.json", payload)
    return SmokeResult(name="epic_sounds_audio", ok=ok, details=payload if ok else {"error": "EPIC-SOUNDS audio extraction failed", **payload})


def check_epic_sounds_frames(run_dir: Path) -> SmokeResult:
    import subprocess

    from avs.preprocess.epic_sounds_frames import extract_epic_sounds_frames

    videos_dir = run_dir / "epic_videos_frames"
    videos_dir.mkdir(parents=True, exist_ok=True)
    out_frames_dir = run_dir / "epic_frames"
    out_frames_dir.mkdir(parents=True, exist_ok=True)

    video_id = "P01_01"
    in_video = videos_dir / f"{video_id}.mp4"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=160x120:rate=25:duration=12",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=44100:duration=12",
        "-shortest",
        str(in_video),
    ]
    subprocess.run(cmd, check=True)  # noqa: S603,S607 - controlled args

    counts = extract_epic_sounds_frames(
        videos_dir=videos_dir,
        out_frames_dir=out_frames_dir,
        video_ids=[video_id],
        start_offset_sec=0.5,
        max_seconds=5,
        jpg_quality=2,
    )

    frames_dir = out_frames_dir / video_id / "frames"
    expected = [frames_dir / f"{i}.jpg" for i in range(5)]
    ok = counts.get(video_id) == 5 and all(p.exists() and p.stat().st_size > 0 for p in expected)
    payload = {"videos_dir": str(videos_dir), "out_frames_dir": str(out_frames_dir), "counts": counts, "frames_dir": str(frames_dir)}
    _write_json(run_dir, "epic_sounds_frames.json", payload)
    return SmokeResult(name="epic_sounds_frames", ok=ok, details=payload if ok else {"error": "EPIC-SOUNDS frame extraction failed", **payload})


def check_epic_sounds_long_pack(run_dir: Path) -> SmokeResult:
    import json as _json
    import subprocess

    import numpy as np

    from avs.experiments.epic_sounds_long_pack import run_epic_sounds_long_pack
    from avs.sampling.token_budget import TokenBudget

    videos_dir = run_dir / "epic_videos_long_pack"
    videos_dir.mkdir(parents=True, exist_ok=True)
    out_dir = run_dir / "epic_long_pack_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    video_id = "P01_01"
    in_video = videos_dir / f"{video_id}.mp4"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=160x120:rate=25:duration=12",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=44100:duration=12",
        "-shortest",
        str(in_video),
    ]
    subprocess.run(cmd, check=True)  # noqa: S603,S607 - controlled args

    payload = run_epic_sounds_long_pack(
        videos_dir=videos_dir,
        out_dir=out_dir,
        video_ids=[video_id],
        max_seconds=12,
        max_steps=8,
        method="energy",
        k=4,
        anchor_radius=1,
        background_stride=3,
        encode=True,
        clip_pretrained=False,
        clip_model_name="openai/clip-vit-base-patch16",
        clip_device="cpu",
        clip_dtype="float32",
        start_offset_sec=0.5,
        jpg_quality=2,
        anchor_shift=0,
        anchor_std_threshold=0.0,
        ast_pretrained=False,
        panns_random=True,
        panns_checkpoint=None,
        audiomae_random=True,
        audiomae_checkpoint=None,
    )

    manifest_path = out_dir / "manifest.jsonl"
    ok = manifest_path.exists() and manifest_path.stat().st_size > 0
    record = None
    if ok:
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
        ok = len(lines) == 1
        if ok:
            record = _json.loads(lines[0])
            selected_seconds = record.get("selected_seconds") or []
            resolutions = record.get("resolutions") or []
            cache_path = record.get("feature_cache_path")
            selected_frames_dir = Path(record.get("selected_frames_dir") or "")

            budget = TokenBudget(patch_size=16).tokens_for_resolution(224)
            expected_total = int(len(selected_seconds)) * int(budget)

            ok = (
                len(selected_seconds) == 8
                and len(resolutions) == 8
                and int(record.get("plan_total_tokens", -1)) == expected_total
                and cache_path is not None
                and Path(cache_path).exists()
                and selected_frames_dir.exists()
                and len(list(selected_frames_dir.glob("*.jpg"))) == 8
            )

            if ok:
                with np.load(str(cache_path)) as z:
                    for r in (112, 224, 448):
                        key = f"res_{r}"
                        arr = z[key]
                        if not (arr.ndim == 2 and arr.shape[0] == 8 and arr.shape[1] > 0):
                            ok = False
                            break

    details = {"videos_dir": str(videos_dir), "out_dir": str(out_dir), "payload": payload, "record": record}
    _write_json(run_dir, "epic_sounds_long_pack.json", details)
    return SmokeResult(
        name="epic_sounds_long_pack",
        ok=bool(ok),
        details=details if ok else {"error": "EPIC-SOUNDS long pack failed", **details},
    )


def check_epic_sounds_video_cls_synth(run_dir: Path) -> SmokeResult:
    from avs.experiments.epic_sounds_video_cls import run_synth_smoke

    out_dir = run_dir / "epic_sounds_video_cls_synth"
    payload = run_synth_smoke(out_dir=out_dir)
    ok = bool(payload.get("ok")) and "mAP" in (payload.get("metrics") or {})

    details = {"out_dir": str(out_dir), "metrics_path": str(out_dir / "metrics.json")}
    _write_json(run_dir, "epic_sounds_video_cls_synth.json", {"details": details, "payload": payload})
    return SmokeResult(
        name="epic_sounds_video_cls_synth",
        ok=ok,
        details=details if ok else {"error": "synthetic video-cls smoke failed", **details},
    )


def check_ave_download(run_dir: Path) -> SmokeResult:
    import subprocess
    import shutil

    from avs.datasets import ave_download

    src_dir = run_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    video_ids = ["toy_video", "-toy_video"]
    src_video = src_dir / f"{video_ids[0]}.mp4"

    # Create a deterministic local MP4 as the "source".
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=160x120:rate=25:duration=10",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=44100:duration=10",
        "-shortest",
        str(src_video),
    ]
    subprocess.run(cmd, check=True)  # noqa: S603,S607 - controlled args

    out_dir = run_dir / "raw" / "videos"
    # Duplicate the file with a leading '-' in the name to verify ids-file handling
    # (argparse would otherwise treat it as an option).
    shutil.copyfile(src_video, src_dir / f"{video_ids[1]}.mp4")

    ids_file = run_dir / "ids.txt"
    ids_file.write_text("\n".join(video_ids) + "\n", encoding="utf-8")
    meta_dir = run_dir / "meta"
    out_json = run_dir / "ave_download.json"

    rc = ave_download.main(
        [
            "--mode",
            "local",
            "--src-dir",
            str(src_dir),
            "--out-dir",
            str(out_dir),
            "--ids-file",
            str(ids_file),
            "--jobs",
            "2",
            "--meta-dir",
            str(meta_dir),
            "--out-json",
            str(out_json),
            "--write-meta-lists",
            "--lists-tag",
            "toy",
        ]
    )

    ok = (
        rc == 0
        and (out_dir / f"{video_ids[0]}.mp4").exists()
        and (out_dir / f"{video_ids[1]}.mp4").exists()
        and out_json.exists()
        and (meta_dir / "download_ok_toy_auto.txt").exists()
        and (meta_dir / "download_fail_toy_auto.txt").exists()
    )
    payload = {"src_dir": str(src_dir), "out_dir": str(out_dir), "ids_file": str(ids_file), "out_json": str(out_json), "meta_dir": str(meta_dir), "rc": int(rc)}
    _write_json(run_dir, "ave_download_smoke.json", payload)
    return SmokeResult(name="ave_download", ok=ok, details=payload if ok else {"error": "download helper failed", **payload})

def check_preprocess_one(run_dir: Path) -> SmokeResult:
    from PIL import Image

    from avs.preprocess.ave_extract import preprocess_one

    in_video = run_dir / "sample.mp4"
    out_root = run_dir / "processed"

    # Generate a deterministic 10s sample A/V clip.
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=320x240:rate=25:duration=10",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=44100:duration=10",
        "-shortest",
        str(in_video),
    ]
    import subprocess

    subprocess.run(cmd, check=True)  # noqa: S603,S607 - controlled args

    paths = preprocess_one(in_video, out_root, clip_id="sample")

    audio_path = paths["audio"]
    frames = paths["frames"]

    import subprocess

    probe = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels",
            "-of",
            "json",
            str(audio_path),
        ],
        text=True,
    )
    info = json.loads(probe)
    stream = (info.get("streams") or [{}])[0]
    sr = int(stream.get("sample_rate", -1))
    ch = int(stream.get("channels", -1))

    ok = (sr == 16000) and (ch == 1) and (len(frames) == 10)
    if frames:
        with Image.open(frames[0]) as im:
            ok = ok and im.size[0] > 0 and im.size[1] > 0

    payload = {
        "in_video": str(in_video),
        "audio_path": str(audio_path),
        "sr": int(sr),
        "channels": int(ch),
        "num_frames": len(frames),
        "frame0": str(frames[0]) if frames else None,
    }
    _write_json(run_dir, "preprocess_one.json", payload)
    return SmokeResult(name="preprocess_one", ok=ok, details=payload if ok else {"error": "preprocess output mismatch"})


def check_anchors(run_dir: Path) -> SmokeResult:
    import math
    import wave

    import numpy as np

    from avs.audio.eventness import compute_eventness_wav_energy, topk_anchors
    from avs.metrics.anchors import recall_at_k

    wav_path = run_dir / "anchors_sample.wav"
    sr = 16000
    dur_s = 10

    amps = [0.1] * dur_s
    amps[6] = 0.9
    amps[7] = 0.9

    t = np.arange(sr * dur_s, dtype=np.float32) / sr
    wave_data = np.zeros_like(t, dtype=np.float32)
    for sec in range(dur_s):
        mask = (t >= sec) & (t < sec + 1)
        wave_data[mask] = amps[sec] * np.sin(2 * math.pi * 440.0 * t[mask])

    pcm16 = np.clip(wave_data * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())

    ev = compute_eventness_wav_energy(wav_path, num_segments=10)
    anchors = topk_anchors(ev.scores, k=2)
    metrics = recall_at_k([6, 7], anchors, num_segments=10, delta=0)

    payload = {
        "wav_path": str(wav_path),
        "scores": ev.scores,
        "anchors": anchors,
        "gt_segments": [6, 7],
        "recall_at_2": metrics.recall,
        "covered": metrics.covered,
    }
    _write_json(run_dir, "anchors.json", {"anchors": anchors})
    _write_json(run_dir, "anchors_metrics.json", payload)

    ok = metrics.recall == 1.0 and set(anchors) == {6, 7}
    return SmokeResult(name="anchors", ok=ok, details=payload if ok else {"error": "unexpected anchors/recall"})


def check_preprocess_dataset(run_dir: Path) -> SmokeResult:
    import subprocess

    from avs.preprocess.ave_dataset import preprocess_ave_videos

    raw_videos_dir = run_dir / "raw" / "videos"
    raw_videos_dir.mkdir(parents=True, exist_ok=True)
    out_dir = run_dir / "processed_dataset"

    for vid in ("v0", "v1"):
        out_path = raw_videos_dir / f"{vid}.mp4"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=160x120:rate=25:duration=10",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=44100:duration=10",
            "-shortest",
            str(out_path),
        ]
        subprocess.run(cmd, check=True)  # noqa: S603,S607 - controlled args

    done = preprocess_ave_videos(raw_videos_dir=raw_videos_dir, out_dir=out_dir, video_ids=["v0", "v1"])
    audio0 = out_dir / "v0" / "audio.wav"
    frame0 = out_dir / "v0" / "frames" / "0.jpg"
    ok = (done == ["v0", "v1"]) and audio0.exists() and frame0.exists()

    payload = {"raw_videos_dir": str(raw_videos_dir), "out_dir": str(out_dir), "done": done}
    _write_json(run_dir, "preprocess_dataset.json", payload)
    return SmokeResult(name="preprocess_dataset", ok=ok, details=payload if ok else {"error": "dataset preprocess failed"})


def check_sampling_plan(run_dir: Path) -> SmokeResult:
    from avs.sampling.plans import (
        budget_band_anchored_plan_scored,
        equal_token_budget_anchored_plan,
        equal_token_budget_anchored_plan_scored,
        uniform_plan,
    )

    num_segments = 10
    uniform = uniform_plan(num_segments=num_segments, resolution=224, patch_size=16)
    anchored = equal_token_budget_anchored_plan(
        num_segments=num_segments,
        anchors=[2, 5],
        low_res=112,
        base_res=224,
        high_res=448,
        patch_size=16,
    )

    # Scored plan variant: smoke that new allocators keep the strict equal-token constraint.
    anchored_bridge = equal_token_budget_anchored_plan_scored(
        num_segments=num_segments,
        anchors=[3, 8],
        scores=None,
        base_alloc="bridge",
        low_res=160,
        base_res=224,
        high_res=352,
        max_high_anchors=None,
        patch_size=16,
    )

    anchored_distance_high = equal_token_budget_anchored_plan_scored(
        num_segments=num_segments,
        anchors=[3, 8],
        scores=None,
        base_alloc="distance_high",
        low_res=160,
        base_res=224,
        high_res=352,
        max_high_anchors=1,
        patch_size=16,
    )

    anchored_band = budget_band_anchored_plan_scored(
        num_segments=num_segments,
        anchors=[2, 5],
        scores=None,
        base_alloc="distance",
        low_res=160,
        base_res=224,
        high_res=352,
        extra_resolutions=[112],
        max_high_anchors=None,
        patch_size=16,
        epsilon_frac=0.01,
    )

    payload = {
        "uniform": uniform.to_jsonable(),
        "anchored": anchored.to_jsonable(),
        "anchored_bridge": anchored_bridge.to_jsonable(),
        "anchored_distance_high": anchored_distance_high.to_jsonable(),
        "anchored_band": anchored_band.to_jsonable(),
    }
    _write_json(run_dir, "sampling_plan.json", payload)
    anchored.save_json(run_dir / "plan.json")

    ok_equal = uniform.total_tokens() == anchored.total_tokens() == anchored_bridge.total_tokens() == anchored_distance_high.total_tokens()
    ok_band = (anchored_band.total_tokens() <= uniform.total_tokens()) and (
        float(uniform.total_tokens() - anchored_band.total_tokens()) <= float(uniform.total_tokens()) * 0.01 + 1e-9
    )
    ok = bool(ok_equal and ok_band)
    return SmokeResult(name="sampling_plan", ok=ok, details=payload if ok else {"error": "token budget mismatch"})


def check_plan_jsonl(run_dir: Path) -> SmokeResult:
    import math
    import wave

    import numpy as np

    from avs.pipeline.plan_generation import plan_from_wav_energy, write_plan_jsonl

    wav_path = run_dir / "plan_audio.wav"
    sr = 16000
    dur_s = 10

    amps = [0.05] * dur_s
    amps[6] = 0.8
    amps[7] = 0.8
    t = np.arange(sr * dur_s, dtype=np.float32) / sr
    wave_data = np.zeros_like(t, dtype=np.float32)
    for sec in range(dur_s):
        mask = (t >= sec) & (t < sec + 1)
        wave_data[mask] = amps[sec] * np.sin(2 * math.pi * 440.0 * t[mask])

    pcm16 = np.clip(wave_data * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())

    rec = plan_from_wav_energy(clip_id="sample", wav_path=wav_path, k=2)
    out_jsonl = run_dir / "plan.jsonl"
    write_plan_jsonl(out_jsonl, [rec])

    lines = out_jsonl.read_text(encoding="utf-8").splitlines()
    ok = len(lines) == 1 and '"clip_id": "sample"' in lines[0] and rec.plan.total_tokens() == 1960
    payload = {"out_jsonl": str(out_jsonl), "record": rec.to_jsonable()}
    _write_json(run_dir, "plan_jsonl.json", payload)
    return SmokeResult(name="plan_jsonl", ok=ok, details=payload if ok else {"error": "plan jsonl output mismatch"})


def check_plan_jsonl_long(run_dir: Path) -> SmokeResult:
    import math
    import wave

    import numpy as np

    from avs.pipeline.plan_generation import infer_num_segments_from_wav, plan_from_wav_auto_segments, write_plan_jsonl
    from avs.sampling.token_budget import TokenBudget

    wav_path = run_dir / "plan_audio_long.wav"
    sr = 16000
    dur_s = 17

    amps = [0.05] * dur_s
    amps[6] = 0.9
    amps[7] = 0.9
    t = np.arange(sr * dur_s, dtype=np.float32) / sr
    wave_data = np.zeros_like(t, dtype=np.float32)
    for sec in range(dur_s):
        mask = (t >= sec) & (t < sec + 1)
        wave_data[mask] = amps[sec] * np.sin(2 * math.pi * 440.0 * t[mask])

    pcm16 = np.clip(wave_data * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())

    inferred = infer_num_segments_from_wav(wav_path, segment_seconds=1.0)
    rec = plan_from_wav_auto_segments(clip_id="sample_long", wav_path=wav_path, k=2)
    out_jsonl = run_dir / "plan_long.jsonl"
    write_plan_jsonl(out_jsonl, [rec])

    budget = TokenBudget(patch_size=16)
    target_tokens = inferred * int(budget.tokens_for_resolution(224))
    lines = out_jsonl.read_text(encoding="utf-8").splitlines()

    ok = inferred == dur_s and len(rec.plan.resolutions) == inferred and rec.plan.total_tokens() == target_tokens and len(lines) == 1
    payload = {
        "wav_path": str(wav_path),
        "inferred_num_segments": inferred,
        "target_tokens": target_tokens,
        "record": rec.to_jsonable(),
        "out_jsonl": str(out_jsonl),
    }
    _write_json(run_dir, "plan_jsonl_long.json", payload)
    return SmokeResult(name="plan_jsonl_long", ok=ok, details=payload if ok else {"error": "long plan jsonl mismatch", **payload})


def check_ast_eventness(run_dir: Path) -> SmokeResult:
    import math
    import wave

    import numpy as np

    from avs.audio.ast_probe import ast_eventness

    wav_path = run_dir / "ast_sample.wav"
    sr = 16000
    dur_s = 10

    tt = np.arange(sr * dur_s, dtype=np.float32) / sr
    wave_data = 0.1 * np.sin(2 * math.pi * 440.0 * tt)
    pcm16 = np.clip(wave_data * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())

    scores = ast_eventness(wav_path, pretrained=False, num_segments=10)
    ok = len(scores) == 10 and all(np.isfinite(scores)) and all(0.0 <= float(x) <= 1.0 for x in scores)
    payload = {"wav_path": str(wav_path), "scores": [float(x) for x in scores]}
    _write_json(run_dir, "ast_eventness.json", payload)
    return SmokeResult(name="ast_eventness", ok=ok, details=payload if ok else {"error": "AST scores invalid"})


def check_energy_delta_eventness(run_dir: Path) -> SmokeResult:
    import math
    import wave

    import numpy as np

    from avs.audio.eventness import compute_eventness_wav_energy_delta, topk_anchors

    wav_path = run_dir / "energy_delta_sample.wav"
    sr = 16000
    dur_s = 10

    tt = np.arange(sr * dur_s, dtype=np.float32) / sr
    amps = [0.05] * dur_s
    amps[6] = 0.9
    amps[7] = 0.9
    wave_data = np.zeros_like(tt, dtype=np.float32)
    for sec in range(dur_s):
        mask = (tt >= sec) & (tt < sec + 1)
        wave_data[mask] = amps[sec] * np.sin(2 * math.pi * 440.0 * tt[mask])
    pcm16 = np.clip(wave_data * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())

    ev = compute_eventness_wav_energy_delta(wav_path, num_segments=dur_s)
    anchors = topk_anchors([float(x) for x in ev.scores], k=2)
    ok = len(ev.scores) == dur_s and set(anchors) == {6, 8}
    payload = {"wav_path": str(wav_path), "scores": [float(x) for x in ev.scores], "anchors_top2": anchors}
    _write_json(run_dir, "energy_delta_eventness.json", payload)
    return SmokeResult(name="energy_delta_eventness", ok=ok, details=payload if ok else {"error": "unexpected anchors/scores", **payload})


def check_panns_eventness(run_dir: Path) -> SmokeResult:
    import math
    import wave

    import numpy as np

    from avs.audio.panns_probe import panns_eventness

    wav_path = run_dir / "panns_sample.wav"
    sr = 16000
    dur_s = 10

    tt = np.arange(sr * dur_s, dtype=np.float32) / sr
    wave_data = 0.1 * np.sin(2 * math.pi * 440.0 * tt)
    pcm16 = np.clip(wave_data * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())

    scores = panns_eventness(wav_path, pretrained=False, checkpoint_path=None, num_segments=10)
    ok = len(scores) == 10 and all(np.isfinite(scores)) and all(0.0 <= float(x) <= 1.0 for x in scores)
    payload = {"wav_path": str(wav_path), "scores": [float(x) for x in scores]}
    _write_json(run_dir, "panns_eventness.json", payload)
    return SmokeResult(name="panns_eventness", ok=ok, details=payload if ok else {"error": "PANNs scores invalid"})


def check_audiomae_eventness(run_dir: Path) -> SmokeResult:
    import math
    import wave

    import numpy as np

    from avs.audio.audiomae_probe import audiomae_eventness

    wav_path = run_dir / "audiomae_sample.wav"
    sr = 16000
    dur_s = 10

    tt = np.arange(sr * dur_s, dtype=np.float32) / sr
    wave_data = 0.1 * np.sin(2 * math.pi * 440.0 * tt)
    pcm16 = np.clip(wave_data * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())

    scores = audiomae_eventness(wav_path, pretrained=False, checkpoint_path=None, num_segments=10)
    ok = len(scores) == 10 and all(np.isfinite(scores)) and all(0.0 <= float(x) <= 1.0 for x in scores)
    payload = {"wav_path": str(wav_path), "scores": [float(x) for x in scores]}
    _write_json(run_dir, "audiomae_eventness.json", payload)
    return SmokeResult(name="audiomae_eventness", ok=ok, details=payload if ok else {"error": "AudioMAE scores invalid"})


def check_anchor_knobs(run_dir: Path) -> SmokeResult:
    from avs.audio.eventness import anchors_from_scores
    from avs.sampling.plans import equal_token_budget_anchored_plan

    scores_flat = [0.0] * 10
    anchors_fallback = anchors_from_scores(scores_flat, k=2, num_segments=10, shift=0, std_threshold=0.1)
    plan_fallback = equal_token_budget_anchored_plan(num_segments=10, anchors=anchors_fallback)

    scores_peak = [0.0] * 10
    scores_peak[6] = 1.0
    scores_peak[7] = 0.9
    anchors_shifted = anchors_from_scores(scores_peak, k=2, num_segments=10, shift=-1, std_threshold=0.0)
    plan_shifted = equal_token_budget_anchored_plan(num_segments=10, anchors=anchors_shifted)

    ok = anchors_fallback == [] and plan_fallback.resolutions == [224] * 10 and anchors_shifted == [5, 6]
    payload = {
        "anchors_fallback": anchors_fallback,
        "plan_fallback": plan_fallback.to_jsonable(),
        "anchors_shifted": anchors_shifted,
        "plan_shifted": plan_shifted.to_jsonable(),
    }
    _write_json(run_dir, "anchor_knobs.json", payload)
    return SmokeResult(name="anchor_knobs", ok=ok, details=payload if ok else {"error": "unexpected anchor knob behavior", **payload})


def check_anchor_window_select(run_dir: Path) -> SmokeResult:
    from avs.audio.eventness import anchors_from_scores

    scores = [0.0] * 10
    scores[5] = 1.0
    scores[6] = 0.99

    anchors_topk = anchors_from_scores(scores, k=2, num_segments=10, select="topk")
    anchors_window = anchors_from_scores(scores, k=2, num_segments=10, select="window_topk", anchor_window=3, nms_radius=1)

    ok = len(anchors_topk) == 2 and anchors_topk == [5, 6]
    ok = ok and len(anchors_window) == 2 and abs(int(anchors_window[0]) - int(anchors_window[1])) > 1

    payload = {"anchors_topk": anchors_topk, "anchors_window_topk": anchors_window, "scores": scores}
    _write_json(run_dir, "anchor_window_select.json", payload)
    return SmokeResult(name="anchor_window_select", ok=ok, details=payload if ok else {"error": "window_topk behavior unexpected", **payload})


def check_anchor_confidence_gate(run_dir: Path) -> SmokeResult:
    from avs.audio.eventness import anchors_from_scores_with_debug

    flat = [0.0] * 10
    sel_flat = anchors_from_scores_with_debug(flat, k=2, num_segments=10, conf_metric="std", conf_threshold=0.1)

    peak = [0.0] * 10
    peak[6] = 1.0
    peak[7] = 0.9
    sel_peak = anchors_from_scores_with_debug(peak, k=2, num_segments=10, conf_metric="std", conf_threshold=0.1)

    ok = sel_flat.fallback_used and sel_flat.fallback_reason == "conf_below_threshold" and sel_flat.anchors == []
    ok = ok and (not sel_peak.fallback_used) and len(sel_peak.anchors) > 0

    payload = {
        "flat": sel_flat.__dict__,
        "peak": sel_peak.__dict__,
    }
    _write_json(run_dir, "anchor_confidence_gate.json", payload)
    return SmokeResult(
        name="anchor_confidence_gate",
        ok=ok,
        details=payload if ok else {"error": "confidence gating unexpected", **payload},
    )


def check_anchors_dataset(run_dir: Path) -> SmokeResult:
    import math
    import wave

    import numpy as np

    from avs.experiments.ave_anchor_eval import AnchorEvalClip, evaluate_anchor_quality

    sr = 16000
    dur_s = 10

    clips: list[AnchorEvalClip] = []
    for i, gt in enumerate(([1, 2], [3, 8], [0, 9], [6, 7], [4, 5])):
        wav_path = run_dir / f"clip_{i}.wav"
        tt = np.arange(sr * dur_s, dtype=np.float32) / sr
        wave_data = np.zeros_like(tt, dtype=np.float32)
        amps = [0.05] * dur_s
        amps[gt[0]] = 0.9
        amps[gt[1]] = 0.9
        for sec in range(dur_s):
            mask = (tt >= sec) & (tt < sec + 1)
            wave_data[mask] = amps[sec] * np.sin(2 * math.pi * 440.0 * tt[mask])

        pcm16 = np.clip(wave_data * 32767.0, -32768, 32767).astype(np.int16)
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm16.tobytes())

        clips.append(AnchorEvalClip(clip_id=f"clip_{i}", wav_path=wav_path, gt_segments=list(gt)))

    metrics = evaluate_anchor_quality(clips, method="energy", k=2, deltas=[0, 1, 2], seed=0)
    ours0 = metrics["by_delta"][0]["ours_mean_recall"]
    rand0 = metrics["by_delta"][0]["random_mean_recall"]

    payload = {"metrics": metrics}
    _write_json(run_dir, "anchors_dataset_metrics.json", payload)
    ok = float(ours0) == 1.0 and float(rand0) < 0.7
    return SmokeResult(
        name="anchors_dataset",
        ok=ok,
        details=payload if ok else {"error": "unexpected recall values", "ours0": ours0, "rand0": rand0},
    )


def check_epic_sounds_anchor_eval(run_dir: Path) -> SmokeResult:
    import math
    import wave

    import numpy as np

    from avs.experiments.epic_sounds_anchor_eval import EpicSoundsAnchorEvalClip, evaluate_anchor_quality

    sr = 16000
    dur_s = 20

    amps = [0.05] * dur_s
    amps[6] = 0.9
    amps[7] = 0.9

    t = np.arange(sr * dur_s, dtype=np.float32) / sr
    wave_data = np.zeros_like(t, dtype=np.float32)
    for sec in range(dur_s):
        mask = (t >= sec) & (t < sec + 1)
        wave_data[mask] = amps[sec] * np.sin(2 * math.pi * 440.0 * t[mask])

    wav_path = run_dir / "epic_sounds_sample.wav"
    pcm16 = np.clip(wave_data * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())

    clips = [EpicSoundsAnchorEvalClip(clip_id="clip0", wav_path=wav_path, gt_segments=[6, 7], num_segments=dur_s)]
    metrics = evaluate_anchor_quality(clips, method="energy", k=2, deltas=[0], seed=0)
    ours0 = metrics["by_delta"][0]["ours_mean_recall"]
    rand0 = metrics["by_delta"][0]["random_mean_recall"]

    ok = float(ours0) == 1.0 and float(rand0) == 0.0
    payload = {"wav_path": str(wav_path), "metrics": metrics}
    _write_json(run_dir, "epic_sounds_anchor_eval.json", payload)
    return SmokeResult(
        name="epic_sounds_anchor_eval",
        ok=ok,
        details=payload if ok else {"error": "unexpected epic_sounds recall values", "ours0": ours0, "rand0": rand0},
    )
def check_vision_encoder(run_dir: Path) -> SmokeResult:
    import numpy as np
    from PIL import Image

    from avs.vision.clip_vit import ClipVisionEncoder, ClipVisionEncoderConfig

    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, size=(256, 256, 3), dtype=np.uint8), mode="RGB")

    enc = ClipVisionEncoder(ClipVisionEncoderConfig(pretrained=False, device="cpu", dtype="float32"))
    e112 = enc.encode([img], resolution=112)
    e224 = enc.encode([img], resolution=224)
    e448 = enc.encode([img], resolution=448)

    ok = e112.shape == e224.shape == e448.shape and e112.ndim == 2 and e112.shape[0] == 1
    payload = {
        "shapes": {"112": list(e112.shape), "224": list(e224.shape), "448": list(e448.shape)},
        "dim": int(e112.shape[-1]) if e112.ndim == 2 else None,
    }
    _write_json(run_dir, "vision_encoder.json", payload)
    return SmokeResult(name="vision_encoder", ok=ok, details=payload if ok else {"error": "unexpected encoder shapes"})


def check_vision_efficiency(run_dir: Path) -> SmokeResult:
    from avs.experiments.vision_efficiency import VisionEfficiencyConfig, bench_clip_vision_encoder

    payload = bench_clip_vision_encoder(
        VisionEfficiencyConfig(resolutions=[112, 224, 448], batch_size=2, warmup=1, iters=1, seed=0, device="cpu", dtype="float32", pretrained=False)
    )
    _write_json(run_dir, "vision_efficiency.json", payload)

    results = payload.get("results_by_resolution", [])
    by_res = {int(r.get("resolution")): r for r in results if "resolution" in r}
    ok = set(by_res.keys()) == {112, 224, 448}
    for r in (112, 224, 448):
        row = by_res.get(r, {})
        ok = ok and float(row.get("ms_per_image", -1.0)) >= 0.0 and int(row.get("tokens_per_image", 0)) > 0
        ok = ok and int(row.get("approx_flops_per_image", 0)) > 0

    if ok:
        ok = int(by_res[112]["approx_flops_per_image"]) < int(by_res[224]["approx_flops_per_image"]) < int(by_res[448]["approx_flops_per_image"])

    details = {"results_by_resolution": results}
    return SmokeResult(
        name="vision_efficiency",
        ok=ok,
        details=details if ok else {"error": "unexpected/missing benchmark fields", **details},
    )


def check_train_smoke(run_dir: Path) -> SmokeResult:
    import torch

    from avs.models.per_segment_mlp import PerSegmentMLP
    from avs.sampling.plans import equal_token_budget_anchored_plan
    from avs.train.synthetic_ave import SyntheticAVEConfig, make_synthetic_ave
    from avs.train.train_loop import TrainConfig, train_per_segment_classifier

    device = torch.device("cpu")
    synth = SyntheticAVEConfig(num_samples=256, num_segments=10, num_classes=8, feat_dim=32, seed=0)

    plan = equal_token_budget_anchored_plan(num_segments=10, anchors=[6, 7])
    x, y = make_synthetic_ave(synth, plan=plan, device=device)
    split = int(0.8 * synth.num_samples)
    x_train, y_train = x[:split], y[:split]
    x_val, y_val = x[split:], y[split:]

    model = PerSegmentMLP(in_dim=synth.feat_dim, num_classes=synth.num_classes, hidden_dim=128).to(device)
    metrics = train_per_segment_classifier(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        cfg=TrainConfig(epochs=12, batch_size=32, lr=2e-3),
    )

    payload = {"synth": synth.__dict__, "plan": plan.to_jsonable(), "val_acc": float(metrics["val_acc"])}
    _write_json(run_dir, "train_smoke.json", payload)

    ok = float(metrics["val_acc"]) >= 0.9
    return SmokeResult(name="train_smoke", ok=ok, details=payload if ok else {"error": "val_acc below threshold", **payload})


def check_temporal_head(run_dir: Path) -> SmokeResult:
    import torch

    from avs.models.temporal_conv import TemporalConvHead
    from avs.sampling.plans import equal_token_budget_anchored_plan
    from avs.train.synthetic_ave import SyntheticAVEConfig, make_synthetic_ave
    from avs.train.train_loop import TrainConfig, train_per_segment_classifier

    device = torch.device("cpu")
    synth = SyntheticAVEConfig(num_samples=256, num_segments=10, num_classes=8, feat_dim=32, seed=1)

    plan = equal_token_budget_anchored_plan(num_segments=10, anchors=[6, 7])
    x, y = make_synthetic_ave(synth, plan=plan, device=device)
    split = int(0.8 * synth.num_samples)
    x_train, y_train = x[:split], y[:split]
    x_val, y_val = x[split:], y[split:]

    model = TemporalConvHead(in_dim=synth.feat_dim, num_classes=synth.num_classes, hidden_dim=128, kernel_size=3, dropout=0.0).to(device)
    metrics = train_per_segment_classifier(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        cfg=TrainConfig(epochs=12, batch_size=32, lr=2e-3),
    )

    payload = {"synth": synth.__dict__, "plan": plan.to_jsonable(), "val_acc": float(metrics["val_acc"])}
    _write_json(run_dir, "temporal_head.json", payload)

    ok = float(metrics["val_acc"]) >= 0.9
    return SmokeResult(name="temporal_head", ok=ok, details=payload if ok else {"error": "val_acc below threshold", **payload})


def check_feature_cache(run_dir: Path) -> SmokeResult:
    import subprocess

    from avs.preprocess.ave_extract import preprocess_one
    from avs.vision.clip_vit import ClipVisionEncoder, ClipVisionEncoderConfig
    from avs.vision.feature_cache import FeatureCache, build_clip_feature_cache

    in_video = run_dir / "fc_sample.mp4"
    out_root = run_dir / "fc_processed"

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=320x240:rate=25:duration=10",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=44100:duration=10",
        "-shortest",
        str(in_video),
    ]
    subprocess.run(cmd, check=True)  # noqa: S603,S607 - controlled args

    preprocess_one(in_video, out_root, clip_id="sample")
    frames_dir = out_root / "sample" / "frames"

    encoder = ClipVisionEncoder(ClipVisionEncoderConfig(pretrained=False, device="cpu", dtype="float32"))
    cache = build_clip_feature_cache(frames_dir=frames_dir, resolutions=[112, 224, 448], encoder=encoder)

    out_npz = run_dir / "features_cache" / "sample.npz"
    cache.save_npz(out_npz)
    loaded = FeatureCache.load_npz(out_npz)

    ok = True
    for r in (112, 224, 448):
        arr = loaded.features_by_resolution[r]
        ok = ok and arr.shape[0] == 10 and arr.ndim == 2

    payload = {"npz": str(out_npz), "resolutions": loaded.resolutions, "shapes": {r: list(loaded.features_by_resolution[r].shape) for r in loaded.resolutions}}
    _write_json(run_dir, "feature_cache.json", payload)
    return SmokeResult(name="feature_cache", ok=ok, details=payload if ok else {"error": "cache shapes invalid"})


def check_ave_p0_toy(run_dir: Path) -> SmokeResult:
    import math
    import random
    import wave

    import numpy as np

    from avs.experiments.ave_p0 import P0Config, run_p0_from_caches
    from avs.train.train_loop import TrainConfig
    from avs.vision.feature_cache import FeatureCache

    rng = random.Random(0)

    caches_dir = run_dir / "p0_caches"
    audio_dir = run_dir / "p0_processed"
    caches_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    num_clips = 40
    num_segments = 10
    dim = 16

    proto_bg = np.zeros((dim,), dtype=np.float32)
    proto_ev = np.ones((dim,), dtype=np.float32)

    def _sigma(res: int, is_event: bool) -> float:
        if not is_event:
            return 0.15
        return {112: 1.0, 224: 0.6, 448: 0.25}[int(res)]

    clip_ids: list[str] = []
    labels_by_clip: dict[str, list[int]] = {}

    sr = 16000
    tt = np.arange(sr * num_segments, dtype=np.float32) / sr

    for i in range(num_clips):
        cid = f"toy_{i:04d}"
        clip_ids.append(cid)
        gt = sorted(rng.sample(range(num_segments), 2))
        labels = [1 if t in gt else 0 for t in range(num_segments)]
        labels_by_clip[cid] = labels

        # Audio: high energy at gt seconds.
        wav_path = audio_dir / cid / "audio.wav"
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        amps = [0.05] * num_segments
        amps[gt[0]] = 0.9
        amps[gt[1]] = 0.9
        wave_data = np.zeros_like(tt, dtype=np.float32)
        for sec in range(num_segments):
            mask = (tt >= sec) & (tt < sec + 1)
            wave_data[mask] = amps[sec] * np.sin(2 * math.pi * 440.0 * tt[mask])
        pcm16 = np.clip(wave_data * 32767.0, -32768, 32767).astype(np.int16)
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm16.tobytes())

        # Feature caches: three resolutions.
        feats: dict[int, np.ndarray] = {}
        for res in (112, 224, 448):
            rows = []
            for t in range(num_segments):
                is_event = labels[t] == 1
                base = proto_ev if is_event else proto_bg
                noise = np.asarray([rng.gauss(0.0, 1.0) for _ in range(dim)], dtype=np.float32) * float(_sigma(res, is_event))
                rows.append(base + noise)
            feats[int(res)] = np.stack(rows, axis=0).astype(np.float32)

        FeatureCache(resolutions=[112, 224, 448], features_by_resolution=feats).save_npz(caches_dir / f"{cid}.npz")

    split = int(0.75 * num_clips)
    clip_ids_train = clip_ids[:split]
    clip_ids_eval = clip_ids[split:]

    cfg = P0Config()
    baselines = ["uniform", "audio_concat_uniform", "random_top2", "anchored_top2", "oracle_top2"]
    metrics = run_p0_from_caches(
        clip_ids_train=clip_ids_train,
        clip_ids_eval=clip_ids_eval,
        labels_by_clip=labels_by_clip,
        caches_dir=caches_dir,
        audio_dir=audio_dir,
        cfg=cfg,
        baselines=baselines,
        seeds=[0],
        train_cfg=TrainConfig(epochs=8, batch_size=16, lr=2e-3),
        num_classes=2,
        num_segments=num_segments,
        eventness_method="energy",
    )

    _write_json(run_dir, "ave_p0_toy.json", metrics)
    ok = set(metrics["summary"].keys()) == set(baselines) and int(metrics["token_budget"]) == 1960
    return SmokeResult(
        name="ave_p0_toy",
        ok=ok,
        details={"token_budget": metrics["token_budget"], "summary": metrics["summary"]}
        if ok
        else {"error": "metrics missing/invalid"},
    )


def check_ave_p0_uniform_low(run_dir: Path) -> SmokeResult:
    import random

    import numpy as np

    from avs.experiments.ave_p0 import P0Config, run_p0_from_caches
    from avs.train.train_loop import TrainConfig
    from avs.vision.feature_cache import FeatureCache

    rng = random.Random(0)

    caches_dir = run_dir / "p0_uniform_low_caches"
    caches_dir.mkdir(parents=True, exist_ok=True)

    num_clips = 8
    num_segments = 10
    dim = 8

    clip_ids: list[str] = []
    labels_by_clip: dict[str, list[int]] = {}

    for i in range(num_clips):
        cid = f"toy_ul_{i:04d}"
        clip_ids.append(cid)
        labels = [rng.randint(0, 1) for _ in range(num_segments)]
        labels_by_clip[cid] = labels

        feats: dict[int, np.ndarray] = {}
        for res in (112, 224):
            rows = []
            for _t in range(num_segments):
                rows.append(np.asarray([rng.gauss(0.0, 1.0) for _ in range(dim)], dtype=np.float32))
            feats[int(res)] = np.stack(rows, axis=0).astype(np.float32)
        FeatureCache(resolutions=[112, 224], features_by_resolution=feats).save_npz(caches_dir / f"{cid}.npz")

    split = int(0.75 * num_clips)
    clip_ids_train = clip_ids[:split]
    clip_ids_eval = clip_ids[split:]

    baselines = ["uniform", "uniform_low"]
    metrics = run_p0_from_caches(
        clip_ids_train=clip_ids_train,
        clip_ids_eval=clip_ids_eval,
        labels_by_clip=labels_by_clip,
        caches_dir=caches_dir,
        audio_dir=None,
        cfg=P0Config(),
        baselines=baselines,
        seeds=[0],
        train_cfg=TrainConfig(epochs=2, batch_size=8, lr=2e-3),
        num_classes=2,
        num_segments=num_segments,
        eventness_method="energy",
    )

    seed0 = metrics["results_by_seed"][0]["baselines"]
    uniform_tokens = int(seed0["uniform"]["token_budget"])
    uniform_low_tokens = int(seed0["uniform_low"]["token_budget"])
    ok = uniform_low_tokens < uniform_tokens and uniform_low_tokens > 0 and uniform_tokens > 0
    _write_json(
        run_dir,
        "ave_p0_uniform_low.json",
        {"baselines": baselines, "uniform_tokens": uniform_tokens, "uniform_low_tokens": uniform_low_tokens, "metrics": metrics},
    )
    return SmokeResult(
        name="ave_p0_uniform_low",
        ok=ok,
        details={"uniform_tokens": uniform_tokens, "uniform_low_tokens": uniform_low_tokens}
        if ok
        else {"error": "unexpected token budgets", "uniform_tokens": uniform_tokens, "uniform_low_tokens": uniform_low_tokens},
    )


def check_viz(run_dir: Path) -> SmokeResult:
    import math
    import wave

    import numpy as np

    from avs.audio.eventness import compute_eventness_wav_energy, topk_anchors
    from avs.sampling.plans import equal_token_budget_anchored_plan
    from avs.visualize.anchors_plot import plot_anchor_plan

    wav_path = run_dir / "viz.wav"
    sr = 16000
    dur_s = 10

    tt = np.arange(sr * dur_s, dtype=np.float32) / sr
    amps = [0.05] * dur_s
    amps[6] = 0.9
    amps[7] = 0.9
    wave_data = np.zeros_like(tt, dtype=np.float32)
    for sec in range(dur_s):
        mask = (tt >= sec) & (tt < sec + 1)
        wave_data[mask] = amps[sec] * np.sin(2 * math.pi * 440.0 * tt[mask])
    pcm16 = np.clip(wave_data * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())

    ev = compute_eventness_wav_energy(wav_path, num_segments=10)
    anchors = topk_anchors(ev.scores, k=2)
    plan = equal_token_budget_anchored_plan(num_segments=10, anchors=anchors)

    out_path = run_dir / "viz.png"
    plot_anchor_plan(
        scores=ev.scores,
        anchors=anchors,
        gt_segments=[6, 7],
        resolutions=plan.resolutions,
        out_path=out_path,
    )
    ok = out_path.exists() and out_path.stat().st_size > 0
    payload = {"png": str(out_path), "anchors": anchors, "scores": ev.scores, "resolutions": plan.resolutions}
    _write_json(run_dir, "viz.json", payload)
    return SmokeResult(name="viz", ok=ok, details=payload if ok else {"error": "viz.png missing/empty"})


def check_efficiency_curve(run_dir: Path) -> SmokeResult:
    import json

    from avs.visualize.efficiency_curve import main as curve_main

    metrics = {
        "baselines": ["uniform", "uniform_low", "random_top2", "anchored_top2"],
        "results_by_seed": [
            {
                "seed": 0,
                "baselines": {
                    "uniform": {"val_acc": 0.25, "token_budget": 1960},
                    "uniform_low": {"val_acc": 0.22, "token_budget": 490},
                    "random_top2": {"val_acc": 0.70, "token_budget": 1960},
                    "anchored_top2": {"val_acc": 0.75, "token_budget": 1960},
                },
            }
        ],
        "summary": {
            "uniform": {"mean": 0.25, "std": 0.0},
            "uniform_low": {"mean": 0.22, "std": 0.0},
            "random_top2": {"mean": 0.70, "std": 0.0},
            "anchored_top2": {"mean": 0.75, "std": 0.0},
        },
        "token_budget": 1960,
    }
    in_metrics = run_dir / "efficiency_curve_input_metrics.json"
    in_metrics.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rc = curve_main(["--in-metrics", str(in_metrics), "--out-dir", str(run_dir), "--title", "Smoke: Accuracy vs Tokens"])

    out_png = run_dir / "efficiency_curve.png"
    out_json = run_dir / "efficiency_curve.json"
    ok = rc == 0 and out_png.exists() and out_png.stat().st_size > 0 and out_json.exists()
    details = {"in_metrics": str(in_metrics), "out_png": str(out_png), "out_json": str(out_json)}
    return SmokeResult(name="efficiency_curve", ok=ok, details=details if ok else {"error": "curve plot missing/empty", **details})


def check_ave_p0_end2end(run_dir: Path) -> SmokeResult:
    import json
    import shutil
    import subprocess

    from avs.datasets.ave import AVEIndex, ensure_ave_meta
    from avs.pipeline.ave_p0_end2end import main as e2e_main
    from avs.utils.paths import data_dir

    meta_dir = data_dir() / "AVE" / "meta"
    ensure_ave_meta(meta_dir)
    index = AVEIndex.from_meta_dir(meta_dir)

    limit_train = 1
    limit_eval = 1
    train_ids = [index.clips[int(i)].video_id for i in index.splits["train"][:limit_train]]
    eval_ids = [index.clips[int(i)].video_id for i in index.splits["val"][:limit_eval]]
    all_ids = sorted(set(train_ids + eval_ids))

    src_dir = run_dir / "e2e_src"
    src_dir.mkdir(parents=True, exist_ok=True)
    template = src_dir / "__template__.mp4"

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=160x120:rate=25:duration=10",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=44100:duration=10",
        "-shortest",
        str(template),
    ]
    subprocess.run(cmd, check=True)  # noqa: S603,S607 - controlled args

    for vid in all_ids:
        shutil.copyfile(template, src_dir / f"{vid}.mp4")

    raw_videos_dir = run_dir / "e2e_raw" / "videos"
    processed_dir = run_dir / "e2e_processed"
    out_dir = run_dir / "ave_p0_end2end"

    rc = e2e_main(
        [
            "--mode",
            "local",
            "--src-dir",
            str(src_dir),
            "--meta-dir",
            str(meta_dir),
            "--raw-videos-dir",
            str(raw_videos_dir),
            "--processed-dir",
            str(processed_dir),
            "--out-dir",
            str(out_dir),
            "--split-train",
            "train",
            "--split-eval",
            "val",
            "--limit-train",
            str(limit_train),
            "--limit-eval",
            str(limit_eval),
            "--seeds",
            "0",
            "--eventness-method",
            "energy",
            "--device",
            "cpu",
            "--cache-num-workers",
            "2",
            "--cache-devices",
            "cpu,cpu",
        ]
    )

    metrics_path = out_dir / "metrics.json"
    ok = rc == 0 and metrics_path.exists()
    payload = None
    if ok:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        baselines = payload.get("metrics", {}).get("baselines")
        ok = bool(payload.get("ok")) and baselines is not None and set(baselines) >= {"uniform", "random_top2", "anchored_top2"}

    details = {
        "meta_dir": str(meta_dir),
        "src_dir": str(src_dir),
        "raw_videos_dir": str(raw_videos_dir),
        "processed_dir": str(processed_dir),
        "out_dir": str(out_dir),
        "train_ids": train_ids,
        "eval_ids": eval_ids,
        "metrics_path": str(metrics_path),
    }
    _write_json(run_dir, "ave_p0_end2end.json", {"details": details, "payload": payload})
    return SmokeResult(
        name="ave_p0_end2end",
        ok=ok,
        details=details if ok else {"error": "end-to-end run failed", **details},
    )


def check_evidence_windows(run_dir: Path) -> SmokeResult:
    from avs.metrics.time_windows import TimeWindow, coverage_at_tau, iou_1d, max_iou

    evidence = [TimeWindow(1.0, 2.0), TimeWindow(5.0, 6.0)]
    windows = [TimeWindow(1.0, 2.0), TimeWindow(4.0, 6.0)]

    iou_e1 = [iou_1d(w, evidence[0]) for w in windows]
    iou_e2 = [iou_1d(w, evidence[1]) for w in windows]

    cov_05 = coverage_at_tau(windows=windows, evidence=evidence, tau=0.5)
    cov_06 = coverage_at_tau(windows=windows, evidence=evidence, tau=0.6)

    ok = (
        abs(float(iou_e1[0]) - 1.0) < 1e-6
        and abs(float(iou_e2[1]) - 0.5) < 1e-6
        and abs(float(max_iou(windows, evidence[0])) - 1.0) < 1e-6
        and abs(float(max_iou(windows, evidence[1])) - 0.5) < 1e-6
        and abs(float(cov_05) - 1.0) < 1e-6
        and abs(float(cov_06) - 0.5) < 1e-6
    )

    payload = {
        "evidence": [{"start_s": e.start_s, "end_s": e.end_s} for e in evidence],
        "windows": [{"start_s": w.start_s, "end_s": w.end_s} for w in windows],
        "iou_evidence_0": iou_e1,
        "iou_evidence_1": iou_e2,
        "cov@0.5": cov_05,
        "cov@0.6": cov_06,
    }
    _write_json(run_dir, "evidence_windows.json", payload)
    return SmokeResult(name="evidence_windows", ok=ok, details=payload if ok else {"error": "coverage/iou mismatch", **payload})


def check_ltl_budget_allocator(run_dir: Path) -> SmokeResult:
    from avs.budget.vis_budget import VisualConfig, token_cost_window
    from avs.metrics.time_windows import TimeWindow
    from avs.sampling.allocator import allocate_budgeted_windows

    windows = [TimeWindow(0.0, 2.0), TimeWindow(2.0, 4.0), TimeWindow(4.0, 6.0)]
    weights = [0.1, 0.9, 0.5]
    cfg_low = VisualConfig(name="low", fps=1.0, resolution=112, r_keep=1.0)
    cfg_high = VisualConfig(name="high", fps=2.0, resolution=224, r_keep=1.0)

    # Budget that allows exactly one high window and two low windows.
    b = float(
        token_cost_window(cfg=cfg_high, window=windows[0])
        + token_cost_window(cfg=cfg_low, window=windows[1])
        + token_cost_window(cfg=cfg_low, window=windows[2])
    )

    rep = allocate_budgeted_windows(windows=windows, weights=weights, configs=[cfg_low, cfg_high], budget=b)

    chosen = [a.cfg.name if a.cfg is not None else None for a in rep.allocations]
    costs = [a.cost for a in rep.allocations]
    ok = rep.ok and abs(float(rep.total_cost) - float(b)) < 1e-6 and chosen == ["low", "high", "low"]

    payload = {"budget": b, "chosen": chosen, "costs": costs, "report": rep.to_jsonable()}
    _write_json(run_dir, "ltl_budget_allocator.json", payload)
    return SmokeResult(name="ltl_budget_allocator", ok=ok, details=payload if ok else {"error": "unexpected allocation", **payload})


def check_ltl_anchor_windows(run_dir: Path) -> SmokeResult:
    import math
    import wave

    import numpy as np

    from avs.audio.augment import shift_audio
    from avs.audio.eventness import anchor_windows_from_scores, compute_eventness_wav_energy_stride

    sr = 16000
    dur_s = 4.0
    t = np.arange(int(sr * dur_s), dtype=np.float32) / float(sr)

    wave_data = np.zeros_like(t, dtype=np.float32)

    def _inject_pulse(t0: float) -> None:
        mask = (t >= float(t0)) & (t < float(t0) + 0.2)
        wave_data[mask] = 0.9 * np.sin(2.0 * math.pi * 440.0 * t[mask])

    _inject_pulse(1.0)
    _inject_pulse(3.0)

    def _write_wav(path: Path, x: np.ndarray) -> None:
        pcm16 = np.clip(x * 32767.0, -32768, 32767).astype(np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm16.tobytes())

    wav_path = run_dir / "ltl_anchor_windows.wav"
    _write_wav(wav_path, wave_data)

    stride_s = 0.2
    win_s = 0.4
    delta_s = 0.5

    ev = compute_eventness_wav_energy_stride(wav_path, stride_s=stride_s, win_s=win_s)
    aw = anchor_windows_from_scores(ev.scores, stride_s=stride_s, win_s=win_s, delta_s=delta_s, k=2, nms_radius=1)

    wav_shift = run_dir / "ltl_anchor_windows_shifted.wav"
    wave_shifted = shift_audio(audio=wave_data, sample_rate=sr, shift_s=0.2)
    _write_wav(wav_shift, wave_shifted)

    ev2 = compute_eventness_wav_energy_stride(wav_shift, stride_s=stride_s, win_s=win_s)
    aw2 = anchor_windows_from_scores(ev2.scores, stride_s=stride_s, win_s=win_s, delta_s=delta_s, k=2, nms_radius=1)

    def _near(xs: list[float], targets: list[float], tol: float) -> bool:
        if len(xs) != len(targets):
            return False
        xs = sorted(float(x) for x in xs)
        targets = sorted(float(x) for x in targets)
        return all(abs(a - b) <= float(tol) for a, b in zip(xs, targets, strict=True))

    ok = len(aw.anchors_s) == 2 and _near(aw.anchors_s, [1.0, 3.0], tol=0.21) and _near(
        aw2.anchors_s, [x + 0.2 for x in aw.anchors_s], tol=0.21
    )

    payload = {
        "wav_path": str(wav_path),
        "wav_shift_path": str(wav_shift),
        "orig": aw.to_jsonable(),
        "shifted": aw2.to_jsonable(),
    }
    _write_json(run_dir, "ltl_anchor_windows.json", payload)
    return SmokeResult(name="ltl_anchor_windows", ok=ok, details=payload if ok else {"error": "unexpected anchors", **payload})


def check_ltl_eventness_stride_max(run_dir: Path) -> SmokeResult:
    import math
    import wave

    import numpy as np

    from avs.audio.eventness import compute_eventness_wav_energy_stride_max

    sr = 16000
    dur_s = 10.0
    t = np.arange(int(sr * dur_s), dtype=np.float32) / float(sr)

    wave_data = np.zeros_like(t, dtype=np.float32)
    mask = (t >= 1.0) & (t < 1.2)
    wave_data[mask] = 0.9 * np.sin(2.0 * math.pi * 440.0 * t[mask])

    wav_path = run_dir / "ltl_eventness_stride_max.wav"
    pcm16 = np.clip(wave_data * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())

    ev = compute_eventness_wav_energy_stride_max(wav_path, num_segments=10, stride_s=0.2, win_s=0.4)
    scores = [float(x) for x in ev.scores]
    top = int(np.argmax(np.asarray(scores, dtype=np.float32))) if scores else -1

    ok = len(scores) == 10 and top == 1
    payload = {"wav_path": str(wav_path), "scores": scores, "top_second": top}
    _write_json(run_dir, "ltl_eventness_stride_max.json", payload)
    return SmokeResult(
        name="ltl_eventness_stride_max",
        ok=ok,
        details=payload if ok else {"error": "unexpected top second for stride_max eventness", **payload},
    )


def check_ltl_eventness_av_fused(run_dir: Path) -> SmokeResult:
    import wave

    import numpy as np
    from PIL import Image

    from avs.audio.eventness import anchors_from_scores_with_debug, compute_eventness_wav_energy_stride_max
    from avs.utils.scores import AV_FUSED_SCORE_SCALE, fuse_max, minmax_01, scale
    from avs.vision.cheap_eventness import frame_diff_eventness, list_frames

    # Make a visual scene-change at t=4: black *4 -> white *6.
    frames_dir = run_dir / "ltl_eventness_av_fused_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    colors = [0, 0, 0, 0, 255, 255, 255, 255, 255, 255]
    for i, c in enumerate(colors):
        img = Image.new("RGB", (64, 64), (c, c, c))
        img.save(frames_dir / f"{i}.jpg")

    # Silent audio: forces fusion to rely on the visual change.
    sr = 16000
    dur_s = 10.0
    wave_data = np.zeros((int(sr * dur_s),), dtype=np.float32)
    wav_path = run_dir / "ltl_eventness_av_fused_silent.wav"
    pcm16 = np.clip(wave_data * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())

    ev = compute_eventness_wav_energy_stride_max(wav_path, num_segments=10, stride_s=0.2, win_s=0.4)
    a = minmax_01([float(x) for x in ev.scores])

    frames = list_frames(frames_dir)
    v = frame_diff_eventness(frames, size=32)
    v = minmax_01([float(x) for x in v])

    fused = scale(fuse_max(a, v, num_segments=10), AV_FUSED_SCORE_SCALE)
    top = int(np.argmax(np.asarray(fused, dtype=np.float32))) if fused else -1

    sel = anchors_from_scores_with_debug(fused, k=1, num_segments=10, std_threshold=1.0)

    ok = len(fused) == 10 and top == 4 and (not sel.fallback_used)
    payload = {
        "wav_path": str(wav_path),
        "frames_dir": str(frames_dir),
        "fused_scores": fused,
        "top_second": top,
        "conf_value": float(sel.conf_value),
        "conf_threshold": float(sel.conf_threshold),
        "fallback_used": bool(sel.fallback_used),
        "fallback_reason": sel.fallback_reason,
    }
    _write_json(run_dir, "ltl_eventness_av_fused.json", payload)
    return SmokeResult(
        name="ltl_eventness_av_fused",
        ok=ok,
        details=payload if ok else {"error": "unexpected top second for av_fused eventness", **payload},
    )


def check_ltl_eventness_autoshift(run_dir: Path) -> SmokeResult:
    from avs.utils.scores import best_shift_by_corr, minmax_01, shift_scores

    # Synthetic: audio peak at t=2, visual peak at t=3 => best shift should be +1.
    audio_raw = [0.0, 0.0, 10.0, 0.0, 0.0]
    visual = [0.0, 0.0, 0.0, 1.0, 0.0]
    s = best_shift_by_corr(minmax_01(audio_raw), minmax_01(visual), shifts=[-2, -1, 0, 1, 2])
    shifted = shift_scores(audio_raw, int(s))

    top_audio = int(max(range(len(audio_raw)), key=lambda i: audio_raw[i]))
    top_visual = int(max(range(len(visual)), key=lambda i: visual[i]))
    top_shifted = int(max(range(len(shifted)), key=lambda i: shifted[i]))

    ok = int(s) == 1 and top_audio == 2 and top_visual == 3 and top_shifted == 3
    payload = {
        "audio_raw": audio_raw,
        "visual": visual,
        "best_shift": int(s),
        "shifted_scores": shifted,
        "top_audio": top_audio,
        "top_visual": top_visual,
        "top_shifted": top_shifted,
    }
    _write_json(run_dir, "ltl_eventness_autoshift.json", payload)
    return SmokeResult(
        name="ltl_eventness_autoshift",
        ok=ok,
        details=payload if ok else {"error": "autoshift did not align peaks as expected", **payload},
    )


def check_cheap_visual_eventness(run_dir: Path) -> SmokeResult:
    from PIL import Image

    from avs.audio.eventness import topk_anchors
    from avs.vision.cheap_eventness import frame_diff_eventness, list_frames

    frames_dir = run_dir / "cheap_visual_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Make a simple scene-change sequence: black -> black -> white -> white -> white.
    colors = [0, 0, 255, 255, 255]
    for i, c in enumerate(colors):
        img = Image.new("RGB", (64, 64), (c, c, c))
        img.save(frames_dir / f"{i}.jpg")

    frames = list_frames(frames_dir)
    scores = frame_diff_eventness(frames, size=32)
    anchors = topk_anchors(scores, k=1)

    ok = len(scores) == len(colors) and anchors == [2]
    payload = {"frames_dir": str(frames_dir), "scores": scores, "anchors": anchors}
    _write_json(run_dir, "cheap_visual_eventness.json", payload)
    return SmokeResult(name="cheap_visual_eventness", ok=ok, details=payload if ok else {"error": "unexpected frame-diff anchors", **payload})


def check_mde_pareto_toy(run_dir: Path) -> SmokeResult:
    from avs.experiments.mde_ltl import write_toy_pareto_report

    out_dir = run_dir / "mde_pareto_toy"
    rep = write_toy_pareto_report(out_dir=out_dir)

    ok = Path(rep["out_json"]).exists() and Path(rep["out_png"]).exists()
    payload = {"out_dir": str(out_dir), **rep}
    _write_json(run_dir, "mde_pareto_toy.json", payload)
    return SmokeResult(name="mde_pareto_toy", ok=ok, details=payload if ok else {"error": "pareto outputs missing", **payload})


def check_ltl_degradation_suite_toy(run_dir: Path) -> SmokeResult:
    from avs.experiments.degradation_suite import run_toy_degradation_suite

    out_dir = run_dir / "degradation_suite_toy"
    rep = run_toy_degradation_suite(out_dir=out_dir)
    ok = Path(rep["out_json"]).exists() and int(rep["num_rows"]) > 0
    payload = {"out_dir": str(out_dir), **rep}
    _write_json(run_dir, "ltl_degradation_suite_toy.json", payload)
    return SmokeResult(name="ltl_degradation_suite_toy", ok=ok, details=payload if ok else {"error": "degradation outputs missing", **payload})


def check_dataset_integrity_audit(run_dir: Path) -> SmokeResult:
    import subprocess

    from avs.experiments.dataset_integrity_audit import run_integrity_audit

    videos_dir = run_dir / "dataset_integrity_videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    good_mp4 = videos_dir / "good.mp4"
    bad_mp4 = videos_dir / "bad.mp4"

    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=96x96:rate=10:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000:duration=1",
            "-shortest",
            str(good_mp4),
        ],
        check=True,
    )  # noqa: S603,S607 - controlled args
    bad_mp4.write_text("not a valid mp4", encoding="utf-8")

    out_dir = run_dir / "dataset_integrity_audit"
    rep = run_integrity_audit(
        videos_dir=videos_dir,
        pattern="*.mp4",
        out_dir=out_dir,
        decode_check="full",
        decode_limit=None,
        limit=None,
    )

    summary = rep["summary"]
    ok = int(summary["total_files"]) == 2 and int(summary["corrupted_files"]) >= 1
    payload = {"out_dir": str(out_dir), **rep}
    _write_json(run_dir, "dataset_integrity_audit.json", payload)
    return SmokeResult(
        name="dataset_integrity_audit",
        ok=ok,
        details=payload if ok else {"error": "integrity audit did not flag corrupted sample", **payload},
    )


def check_root_cause_report(run_dir: Path) -> SmokeResult:
    from avs.experiments.root_cause_report import build_root_cause_report

    in_dir = run_dir / "root_cause_inputs"
    in_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "summary": {
            "uniform": {"mean": 0.71, "std": 0.01},
            "anchored_top2": {"mean": 0.717, "std": 0.01},
        },
        "paired_ttest": {"anchored_vs_uniform": {"p": 0.18}},
        "token_budget": 1960,
        "debug_eval": {
            "anchored_top2": {
                "c1": {"fallback_used": True},
                "c2": {"fallback_used": True},
                "c3": {"fallback_used": False},
            }
        },
    }
    ovp = {
        "oracle_minus_predicted": 0.03,
        "predicted": {"summary": {"uniform": {"mean": 0.71}, "anchored_top2": {"mean": 0.717}}, "p_anchored_vs_uniform": 0.18},
    }
    evidence = {"cov_by_tau": {"0.3": 0.09, "0.5": 0.08, "0.7": 0.07}, "corr_by_tau": {"0.3": {"pearson": 0.01}, "0.5": {"pearson": 0.02}}}
    degradation = {"alpha_floor_checks": {"num_fail": 2}, "rows": [{"acc_delta": {"anchored_minus_uniform": -0.03}}, {"acc_delta": {"anchored_minus_uniform": 0.01}}]}
    epic = {"delta_mAP": -0.05, "delta_macro_f1@0.5": -0.01}

    metrics_json = in_dir / "metrics.json"
    ovp_json = in_dir / "oracle_vs_predicted.json"
    evidence_json = in_dir / "evidence_alignment.json"
    degradation_json = in_dir / "degradation_accuracy.json"
    epic_json = in_dir / "epic_compare.json"
    metrics_json.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ovp_json.write_text(json.dumps(ovp, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    evidence_json.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    degradation_json.write_text(json.dumps(degradation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    epic_json.write_text(json.dumps(epic, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    out_dir = run_dir / "root_cause_report"
    rep = build_root_cause_report(
        metrics_json=metrics_json,
        oracle_vs_predicted_json=ovp_json,
        evidence_alignment_json=evidence_json,
        degradation_accuracy_json=degradation_json,
        epic_compare_json=epic_json,
        out_dir=out_dir,
        target_delta=0.02,
        target_p=0.05,
        oracle_gap_threshold=0.015,
        fallback_threshold=0.6,
        coverage_threshold=0.2,
        corr_threshold=0.15,
    )

    report = json.loads(Path(rep["out_json"]).read_text(encoding="utf-8"))
    reason_ids = [str(x.get("id")) for x in (report.get("reasons") or []) if isinstance(x, dict)]
    ok = ("R_TARGET_DELTA" in reason_ids) and ("R_ORACLE_GAP" in reason_ids) and ("R_EPIC_DOWNSTREAM" in reason_ids)
    payload = {"out_dir": str(out_dir), **rep, "reason_ids": reason_ids}
    _write_json(run_dir, "root_cause_report.json", payload)
    return SmokeResult(
        name="root_cause_report",
        ok=ok,
        details=payload if ok else {"error": "root-cause report did not surface expected reasons", **payload},
    )


handlers = {
    "ave_meta": check_ave_meta,
    "epic_sounds_meta": check_epic_sounds_meta,
    "avqa_meta": check_avqa_meta,
    "vggsound_meta": check_vggsound_meta,
    "epic_sounds_audio": check_epic_sounds_audio,
    "epic_sounds_frames": check_epic_sounds_frames,
    "epic_sounds_long_pack": check_epic_sounds_long_pack,
    "epic_sounds_video_cls_synth": check_epic_sounds_video_cls_synth,
    "ave_download": check_ave_download,
    "preprocess_one": check_preprocess_one,
    "preprocess_dataset": check_preprocess_dataset,
    "anchors": check_anchors,
    "anchors_dataset": check_anchors_dataset,
    "epic_sounds_anchor_eval": check_epic_sounds_anchor_eval,
    "sampling_plan": check_sampling_plan,
    "plan_jsonl": check_plan_jsonl,
    "plan_jsonl_long": check_plan_jsonl_long,
    "ast_eventness": check_ast_eventness,
    "energy_delta_eventness": check_energy_delta_eventness,
    "panns_eventness": check_panns_eventness,
    "audiomae_eventness": check_audiomae_eventness,
    "anchor_knobs": check_anchor_knobs,
    "anchor_window_select": check_anchor_window_select,
    "anchor_confidence_gate": check_anchor_confidence_gate,
    "feature_cache": check_feature_cache,
    "vision_encoder": check_vision_encoder,
    "vision_efficiency": check_vision_efficiency,
    "train_smoke": check_train_smoke,
    "temporal_head": check_temporal_head,
    "ave_p0_toy": check_ave_p0_toy,
    "ave_p0_uniform_low": check_ave_p0_uniform_low,
    "efficiency_curve": check_efficiency_curve,
    "viz": check_viz,
    "ave_p0_end2end": check_ave_p0_end2end,
    "evidence_windows": check_evidence_windows,
    "ltl_budget_allocator": check_ltl_budget_allocator,
    "ltl_anchor_windows": check_ltl_anchor_windows,
    "ltl_eventness_stride_max": check_ltl_eventness_stride_max,
    "ltl_eventness_av_fused": check_ltl_eventness_av_fused,
    "ltl_eventness_autoshift": check_ltl_eventness_autoshift,
    "cheap_visual_eventness": check_cheap_visual_eventness,
    "mde_pareto_toy": check_mde_pareto_toy,
    "ltl_degradation_suite_toy": check_ltl_degradation_suite_toy,
    "dataset_integrity_audit": check_dataset_integrity_audit,
    "root_cause_report": check_root_cause_report,
}
