from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from avs.audio.augment import add_noise_snr_db, apply_silence_ratio, shift_audio
from avs.audio.features import audio_features_per_second, audio_features_per_second_from_array
from avs.audio.eventness import (
    anchor_windows_from_scores,
    eventness_energy_stride,
    eventness_energy_stride_max_per_second,
    anchors_from_scores_with_debug,
    eventness_energy_delta_per_second,
    eventness_energy_per_second,
    load_wav_mono,
)
from avs.datasets.ave import AVEIndex, ensure_ave_meta
from avs.datasets.layout import ave_paths
from avs.metrics.anchors import recall_at_k
from avs.utils.scores import best_shift_by_corr, fuse_max, fuse_prod, minmax_01, shift_scores, stride_max_pool_per_second
from avs.vision.feature_cache import FeatureCache
from avs.vision.cheap_eventness import clip_feature_diff_eventness


def run_toy_degradation_suite(*, out_dir: Path) -> dict:
    """
    Toy-only degradation suite used by smoke tests.

    Produces a JSON "heatmap" keyed by {shift_s, snr_db, silence_ratio, alpha}.
    Metric is a simple anchor hit-rate on a synthetic pulse event.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sr = 16000
    dur_s = 4.0
    t = np.arange(int(sr * dur_s), dtype=np.float32) / float(sr)

    # One pulse at 1.0s.
    wave = np.zeros_like(t, dtype=np.float32)
    mask = (t >= 1.0) & (t < 1.2)
    wave[mask] = 0.9 * np.sin(2.0 * math.pi * 440.0 * t[mask])

    stride_s = 0.2
    win_s = 0.4
    delta_s = 0.5
    event_time = 1.0

    shift_grid = [-0.5, 0.0, 0.5]
    snr_grid = [20.0, 10.0, 0.0]
    silence_grid = [0.0, 0.5]
    alpha_grid = [0.0, 0.5, 1.0]

    rows: list[dict] = []
    for shift_s in shift_grid:
        for snr_db in snr_grid:
            for silence_ratio in silence_grid:
                # Deterministic noise/silence per condition.
                seed = int((shift_s + 1.0) * 1000 + (snr_db + 100.0) * 10 + silence_ratio * 100)
                rng = np.random.default_rng(seed)

                x = shift_audio(audio=wave, sample_rate=sr, shift_s=float(shift_s))
                x = add_noise_snr_db(audio=x, snr_db=float(snr_db), rng=rng)
                x = apply_silence_ratio(audio=x, silence_ratio=float(silence_ratio), rng=rng)

                scores = eventness_energy_stride(x, sr, stride_s=float(stride_s), win_s=float(win_s), pad=True)
                aw = anchor_windows_from_scores(scores, stride_s=float(stride_s), win_s=float(win_s), delta_s=float(delta_s), k=1, nms_radius=1)
                hit = any(float(w.start_s) <= float(event_time) <= float(w.end_s) for w in aw.windows)

                for alpha in alpha_grid:
                    rows.append(
                        {
                            "shift_s": float(shift_s),
                            "snr_db": float(snr_db),
                            "silence_ratio": float(silence_ratio),
                            "alpha": float(alpha),
                            "hit": bool(hit),
                            "anchors_s": [float(x) for x in aw.anchors_s],
                        }
                    )

    out = {
        "ok": True,
        "mode": "toy",
        "params": {"stride_s": float(stride_s), "win_s": float(win_s), "delta_s": float(delta_s), "event_time": float(event_time)},
        "grid": {"shift_s": shift_grid, "snr_db": snr_grid, "silence_ratio": silence_grid, "alpha": alpha_grid},
        "rows": rows,
    }

    out_json = out_dir / "degradation_suite.json"
    out_json.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"out_dir": str(out_dir), "out_json": str(out_json), "num_rows": len(rows)}


def _parse_csv_floats(value: str) -> list[float]:
    out: list[float] = []
    for s in str(value).split(","):
        s = s.strip()
        if not s:
            continue
        out.append(float(s))
    return out


def _read_ids_file(path: Path, limit: int | None) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        s = str(line).strip()
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        ids.append(s)
        if limit is not None and len(ids) >= int(limit):
            break
    return ids


def _split_ids(index: AVEIndex, split: str, limit: int | None) -> list[str]:
    ids = index.splits[str(split)]
    if limit is not None:
        ids = ids[: int(limit)]
    return [index.clips[int(i)].video_id for i in ids]


def _load_or_select_ids(index: AVEIndex, ids_file: Path | None, split: str, limit: int | None) -> list[str]:
    if ids_file is not None:
        return _read_ids_file(ids_file, limit)
    return _split_ids(index, split, limit)


def _filter_missing(*, ids: list[str], caches_dir: Path) -> list[str]:
    cached = {p.stem for p in caches_dir.glob("*.npz")}
    return [cid for cid in ids if cid in cached]


def _labels_for_ids(index: AVEIndex, ids: list[str]) -> dict[str, list[int]]:
    clip_by_id = {c.video_id: c for c in index.clips}
    out: dict[str, list[int]] = {}
    for cid in ids:
        clip = clip_by_id.get(cid)
        if clip is None:
            continue
        out[cid] = [int(x) for x in index.segment_labels(clip)]
    return out


def _eventness_scores_from_augmented(
    *,
    wav_path: Path,
    num_segments: int,
    method: str,
    shift_s: float,
    snr_db: float,
    silence_ratio: float,
    rng: np.random.Generator,
    audio_model: object | None = None,
    audio_feature_set: str | None = None,
    num_classes: int | None = None,
    panns_probe: object | None = None,
    ast_probe: object | None = None,
    ast_lr_model: object | None = None,
    ast_emb_lr_model: object | None = None,
    ast_evt_mlp_model: object | None = None,
    ast_mlp_model: object | None = None,
) -> list[float]:
    audio, sr = load_wav_mono(wav_path)
    x = shift_audio(audio=audio, sample_rate=int(sr), shift_s=float(shift_s))
    x = add_noise_snr_db(audio=x, snr_db=float(snr_db), rng=rng)
    x = apply_silence_ratio(audio=x, silence_ratio=float(silence_ratio), rng=rng)

    if str(method) in ("energy", "moe_energy_clipdiff", "energy_autoshift_clipdiff", "energy_autoshift_clipdiff_pos"):
        return [float(s) for s in eventness_energy_per_second(x, int(sr), num_segments=int(num_segments))]
    if str(method) == "energy_delta":
        return [float(s) for s in eventness_energy_delta_per_second(x, int(sr), num_segments=int(num_segments))]
    if str(method) in ("energy_stride_max", "av_fused", "av_fused_prod", "av_fused_clipdiff", "av_fused_clipdiff_prod"):
        return [
            float(s)
            for s in eventness_energy_stride_max_per_second(x, int(sr), num_segments=int(num_segments), stride_s=0.2, win_s=0.4)
        ]
    if str(method) == "panns":
        if panns_probe is None:
            raise ValueError("panns probe not initialized")
        return [float(s) for s in panns_probe.eventness_from_array(x, int(sr), num_segments=int(num_segments))]
    if str(method) == "ast":
        if ast_probe is None:
            raise ValueError("ast probe not initialized")
        return [float(s) for s in ast_probe.eventness_from_array(x, int(sr), num_segments=int(num_segments))]
    if str(method) == "ast_lr":
        if ast_probe is None or ast_lr_model is None:
            raise ValueError("ast_lr model not initialized")

        import torch

        feats = ast_probe.logits_from_array(x, int(sr), num_segments=int(num_segments))
        xt = torch.from_numpy(feats).to(dtype=torch.float32, device=torch.device("cpu"))
        with torch.no_grad():
            scores = ast_lr_model(xt).squeeze(-1)
        return [float(v) for v in scores.detach().cpu().numpy().astype("float32").tolist()]
    if str(method) == "ast_evt_mlp":
        if ast_probe is None or ast_evt_mlp_model is None:
            raise ValueError("ast_evt_mlp model not initialized")

        import torch

        feats = ast_probe.logits_from_array(x, int(sr), num_segments=int(num_segments))
        xt = torch.from_numpy(feats).to(dtype=torch.float32, device=torch.device("cpu"))
        with torch.no_grad():
            scores = ast_evt_mlp_model(xt).squeeze(-1)
        return [float(v) for v in scores.detach().cpu().numpy().astype("float32").tolist()]
    if str(method) == "ast_emb_lr":
        if ast_probe is None or ast_emb_lr_model is None:
            raise ValueError("ast_emb_lr model not initialized")

        import torch

        feats = ast_probe.embeddings_from_array(x, int(sr), num_segments=int(num_segments))
        xt = torch.from_numpy(feats).to(dtype=torch.float32, device=torch.device("cpu"))
        with torch.no_grad():
            scores = ast_emb_lr_model(xt).squeeze(-1)
        return [float(v) for v in scores.detach().cpu().numpy().astype("float32").tolist()]
    if str(method) in ("ast_mlp_cls", "ast_mlp_cls_target"):
        if ast_probe is None or ast_mlp_model is None:
            raise ValueError("ast_mlp_cls model not initialized")

        import torch

        feats = ast_probe.logits_from_array(x, int(sr), num_segments=int(num_segments))
        xt = torch.from_numpy(feats).to(dtype=torch.float32, device=torch.device("cpu"))
        with torch.no_grad():
            logits = ast_mlp_model(xt)
            bg = logits[:, 0]
            if str(method) == "ast_mlp_cls":
                mx = logits[:, 1:].max(dim=-1).values
                scores = mx - bg
            else:
                clip_logits = logits.mean(dim=0)
                clip_logits = clip_logits.clone()
                clip_logits[0] = float("-inf")
                cls = int(torch.argmax(clip_logits).item())
                scores = logits[:, cls] - bg

        return [float(x) for x in scores.detach().cpu().numpy().astype("float32").tolist()]
    if str(method).startswith("audio_"):
        if audio_model is None or audio_feature_set is None:
            raise ValueError(f"audio model not initialized for method={method}")

        import torch

        feats = audio_features_per_second_from_array(x, int(sr), num_segments=int(num_segments), feature_set=str(audio_feature_set))
        xt = torch.from_numpy(feats).to(dtype=torch.float32, device=torch.device("cpu"))
        with torch.no_grad():
            logits = audio_model(xt)

        if str(method) in ("audio_basic_lr", "audio_basic_mlp", "audio_basic_tcn", "audio_fbank_mlp", "audio_fbank_tcn"):
            scores = logits.squeeze(-1)
        elif str(method) in ("audio_basic_mlp_cls", "audio_basic_mlp_cls_target"):
            if num_classes is None:
                raise ValueError(f"num_classes required for method={method}")
            if int(num_classes) <= 1:
                raise ValueError(f"num_classes must be > 1, got {num_classes}")
            if str(method) == "audio_basic_mlp_cls":
                probs = torch.softmax(logits, dim=-1)
                scores = 1.0 - probs[:, 0]
            else:
                clip_logits = logits.mean(dim=0)
                clip_logits = clip_logits.clone()
                clip_logits[0] = float("-inf")
                cls = int(torch.argmax(clip_logits).item())
                scores = logits[:, cls]
        else:
            raise ValueError(f"unsupported learned audio method: {method}")

        return [float(x) for x in scores.detach().cpu().numpy().astype("float32").tolist()]
    raise ValueError(f"unsupported eventness method: {method}")


def run_ave_official_degradation_suite(
    *,
    out_dir: Path,
    meta_dir: Path,
    processed_dir: Path,
    caches_dir: Path,
    split_train: str,
    split_eval: str,
    train_ids_file: Path | None,
    eval_ids_file: Path | None,
    limit_train: int | None,
    limit_eval: int | None,
    allow_missing: bool,
    eventness_method: str,
    k: int,
    anchor_select: str,
    anchor_nms_radius: int,
    anchor_nms_strong_gap: float,
    anchor_shift_segments: int,
    anchor_window: int,
    anchor_smooth_window: int,
    anchor_smooth_mode: str,
    audio_device: str,
    ast_pretrained: bool,
    shift_grid: list[float],
    snr_grid: list[float],
    silence_grid: list[float],
    deltas: list[int],
) -> dict:
    """
    Full-mode degradation suite (Stage-1 focus): run anchor robustness under shift/noise/silence.

    This produces an anchor-quality heatmap (Recall@K,Δ) and fallback stats. Downstream accuracy evaluation
    is intentionally left to E0203 follow-ups (grid size can be very large on full AVE).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ensure_ave_meta(meta_dir)
    index = AVEIndex.from_meta_dir(meta_dir)

    train_ids = _load_or_select_ids(index, train_ids_file, split_train, limit_train)
    eval_ids = _load_or_select_ids(index, eval_ids_file, split_eval, limit_eval)
    if allow_missing:
        train_ids = _filter_missing(ids=train_ids, caches_dir=caches_dir)
        eval_ids = _filter_missing(ids=eval_ids, caches_dir=caches_dir)
    if not train_ids or not eval_ids:
        raise ValueError(f"no usable ids after filtering (train={len(train_ids)} eval={len(eval_ids)})")

    labels_by_clip = {**_labels_for_ids(index, train_ids), **_labels_for_ids(index, eval_ids)}
    clip_by_id = {c.video_id: c for c in index.clips}

    audio_model = None
    audio_feature_set = None
    if str(eventness_method).startswith("audio_"):
        from avs.experiments.ave_p0 import (
            _train_audio_basic_lr_eventness,
            _train_audio_basic_mlp_cls_eventness,
            _train_audio_basic_mlp_eventness,
            _train_audio_tcn_eventness,
        )

        # Precompute train features and fit once; then score augmented audio in the main loop.
        audio_feature_set = "fbank_stats" if str(eventness_method) in ("audio_fbank_mlp", "audio_fbank_tcn") else "basic"
        audio_feats_by_clip: dict[str, np.ndarray] = {}
        for cid in train_ids:
            wav_path = processed_dir / cid / "audio.wav"
            audio_feats_by_clip[cid] = audio_features_per_second(wav_path, num_segments=10, feature_set=str(audio_feature_set))

        if str(eventness_method) == "audio_basic_lr":
            audio_model = _train_audio_basic_lr_eventness(
                clip_ids_train=train_ids,
                labels_by_clip=labels_by_clip,
                audio_feats_by_clip=audio_feats_by_clip,
                device="cpu",
            )
        elif str(eventness_method) == "audio_basic_mlp":
            audio_model = _train_audio_basic_mlp_eventness(
                clip_ids_train=train_ids,
                labels_by_clip=labels_by_clip,
                audio_feats_by_clip=audio_feats_by_clip,
                device="cpu",
            )
        elif str(eventness_method) == "audio_fbank_mlp":
            audio_model = _train_audio_basic_mlp_eventness(
                clip_ids_train=train_ids,
                labels_by_clip=labels_by_clip,
                audio_feats_by_clip=audio_feats_by_clip,
                device="cpu",
                hidden_dim=128,
            )
        elif str(eventness_method) in ("audio_basic_tcn", "audio_fbank_tcn"):
            audio_model = _train_audio_tcn_eventness(
                clip_ids_train=train_ids,
                labels_by_clip=labels_by_clip,
                audio_feats_by_clip=audio_feats_by_clip,
                device="cpu",
                epochs=50,
                batch_size=128,
                lr=1e-3,
                hidden_channels=64,
                kernel_size=3,
                dropout=0.1,
            )
        elif str(eventness_method) in ("audio_basic_mlp_cls", "audio_basic_mlp_cls_target"):
            audio_model = _train_audio_basic_mlp_cls_eventness(
                clip_ids_train=train_ids,
                labels_by_clip=labels_by_clip,
                audio_feats_by_clip=audio_feats_by_clip,
                num_classes=int(index.num_classes),
                device="cpu",
            )
        else:
            raise ValueError(f"unsupported learned audio method: {eventness_method}")

        import torch

        audio_model = audio_model.to(torch.device("cpu"))
        audio_model.eval()

    av_model = None
    clipdiff_scores_by_clip: dict[str, list[float]] | None = None
    framediff_scores_by_clip: dict[str, list[float]] | None = None
    flow_scores_by_clip: dict[str, list[float]] | None = None
    if str(eventness_method) in ("av_clipdiff_mlp", "av_clipdiff_framediff_mlp", "av_clipdiff_flow_mlp", "av_clipdiff_flow_mlp_stride"):
        from avs.experiments.ave_p0 import _train_audio_basic_mlp_eventness
        from avs.vision.cheap_eventness import frame_diff_eventness, list_frames, optical_flow_mag_eventness

        all_ids = sorted(set(train_ids + eval_ids))
        clipdiff_scores_by_clip = {}
        if str(eventness_method) == "av_clipdiff_framediff_mlp":
            framediff_scores_by_clip = {}
        if str(eventness_method) in ("av_clipdiff_flow_mlp", "av_clipdiff_flow_mlp_stride"):
            flow_scores_by_clip = {}

        for i, cid in enumerate(all_ids):
            cache = FeatureCache.load_npz(caches_dir / f"{cid}.npz")
            # Prefer 112 when available to match the Stage-1 default in AVE-P0 (avoid coupling to Stage-2 triad knobs).
            vis_res = 112 if 112 in cache.features_by_resolution else int(min(cache.features_by_resolution))
            feats = cache.features_by_resolution[int(vis_res)]
            cd = clip_feature_diff_eventness(feats, metric="cosine")
            clipdiff_scores_by_clip[cid] = minmax_01([float(x) for x in cd])

            if framediff_scores_by_clip is not None:
                frames_dir = processed_dir / cid / "frames"
                frames = list_frames(frames_dir) if frames_dir.exists() else []
                fd = frame_diff_eventness(frames, size=32) if frames else []
                framediff_scores_by_clip[cid] = minmax_01([float(x) for x in fd])
            if flow_scores_by_clip is not None:
                frames_dir = processed_dir / cid / "frames"
                frames = list_frames(frames_dir) if frames_dir.exists() else []
                flow = optical_flow_mag_eventness(frames, size=64) if frames else []
                flow_scores_by_clip[cid] = minmax_01([float(x) for x in flow])

            if (i + 1) % 500 == 0 or (i + 1) == len(all_ids):
                print(f"[{eventness_method}] cached proxies {i+1}/{len(all_ids)} clips", flush=True)

        feats_by_train: dict[str, np.ndarray] = {}
        for i, cid in enumerate(train_ids):
            wav_path = processed_dir / cid / "audio.wav"
            a = audio_features_per_second(wav_path, num_segments=10, feature_set="basic").astype(np.float32, copy=False)

            cd = clipdiff_scores_by_clip.get(cid) or []

            v_cd = np.zeros((10, 1), dtype=np.float32)
            for t, s in enumerate(cd[:10]):
                v_cd[int(t), 0] = float(s)

            if a.shape[0] != 10:
                raise ValueError(f"unexpected audio feature shape for {cid}: {a.shape}")
            if framediff_scores_by_clip is not None:
                fd = framediff_scores_by_clip.get(cid) or []
                v_fd = np.zeros((10, 1), dtype=np.float32)
                for t, s in enumerate(fd[:10]):
                    v_fd[int(t), 0] = float(s)
                feats_by_train[cid] = np.concatenate([a, v_cd, v_fd], axis=1).astype(np.float32, copy=False)
            elif flow_scores_by_clip is not None:
                flow = flow_scores_by_clip.get(cid) or []
                v_flow = np.zeros((10, 1), dtype=np.float32)
                for t, s in enumerate(flow[:10]):
                    v_flow[int(t), 0] = float(s)
                feats_by_train[cid] = np.concatenate([a, v_cd, v_flow], axis=1).astype(np.float32, copy=False)
            else:
                feats_by_train[cid] = np.concatenate([a, v_cd], axis=1).astype(np.float32, copy=False)

            if (i + 1) % 500 == 0 or (i + 1) == len(train_ids):
                print(f"[{eventness_method}] feats train {i+1}/{len(train_ids)} clips", flush=True)

        import torch

        av_model = _train_audio_basic_mlp_eventness(
            clip_ids_train=train_ids,
            labels_by_clip=labels_by_clip,
            audio_feats_by_clip=feats_by_train,
            device="cpu",
            hidden_dim=128,
        )
        av_model = av_model.to(torch.device("cpu"))
        av_model.eval()

    ast_probe = None
    ast_lr_model = None
    ast_emb_lr_model = None
    ast_evt_mlp_model = None
    ast_mlp_model = None
    if str(eventness_method) in ("ast", "ast_lr", "ast_emb_lr", "ast_evt_mlp", "ast_mlp_cls", "ast_mlp_cls_target"):
        from avs.audio.ast_probe import ASTEventnessProbe, ASTProbeConfig

        ast_probe = ASTEventnessProbe(ASTProbeConfig(pretrained=bool(ast_pretrained), device=str(audio_device)))
        if str(eventness_method) == "ast_lr":
            from avs.experiments.ave_p0 import _train_ast_lr_eventness

            ast_lr_model, _train_logits_by_clip = _train_ast_lr_eventness(
                clip_ids_train=train_ids,
                labels_by_clip=labels_by_clip,
                audio_dir=processed_dir,
                ast_probe=ast_probe,
                num_segments=10,
                device="cpu",
            )

            import torch

            ast_lr_model = ast_lr_model.to(torch.device("cpu"))
            ast_lr_model.eval()
        if str(eventness_method) == "ast_emb_lr":
            from avs.experiments.ave_p0 import _train_ast_emb_lr_eventness

            ast_emb_lr_model, _train_emb_by_clip = _train_ast_emb_lr_eventness(
                clip_ids_train=train_ids,
                labels_by_clip=labels_by_clip,
                audio_dir=processed_dir,
                ast_probe=ast_probe,
                num_segments=10,
                device="cpu",
            )

            import torch

            ast_emb_lr_model = ast_emb_lr_model.to(torch.device("cpu"))
            ast_emb_lr_model.eval()
        if str(eventness_method) == "ast_evt_mlp":
            from avs.experiments.ave_p0 import _train_ast_evt_mlp_eventness

            ast_evt_mlp_model, _train_logits_by_clip = _train_ast_evt_mlp_eventness(
                clip_ids_train=train_ids,
                labels_by_clip=labels_by_clip,
                audio_dir=processed_dir,
                ast_probe=ast_probe,
                num_segments=10,
                device="cpu",
                hidden_dim=128,
            )

            import torch

            ast_evt_mlp_model = ast_evt_mlp_model.to(torch.device("cpu"))
            ast_evt_mlp_model.eval()
        if str(eventness_method) in ("ast_mlp_cls", "ast_mlp_cls_target"):
            from avs.experiments.ave_p0 import _train_ast_mlp_cls_eventness

            ast_mlp_model, _train_logits_by_clip = _train_ast_mlp_cls_eventness(
                clip_ids_train=train_ids,
                labels_by_clip=labels_by_clip,
                audio_dir=processed_dir,
                ast_probe=ast_probe,
                num_classes=int(index.num_classes),
                num_segments=10,
                device="cpu",
                hidden_dim=128,
            )

            import torch

            ast_mlp_model = ast_mlp_model.to(torch.device("cpu"))
            ast_mlp_model.eval()

    panns_probe = None
    if str(eventness_method) == "panns":
        from avs.audio.panns_probe import PANNsEventnessProbe, PANNsProbeConfig

        panns_probe = PANNsEventnessProbe(PANNsProbeConfig(pretrained=True, device=str(audio_device)))

    visual_scores_by_clip: dict[str, list[float]] | None = None
    if str(eventness_method) in ("av_fused", "av_fused_prod"):
        from avs.vision.cheap_eventness import frame_diff_eventness, list_frames

        visual_scores_by_clip = {}
        for cid in eval_ids:
            frames_dir = processed_dir / cid / "frames"
            frames = list_frames(frames_dir) if frames_dir.exists() else []
            v = frame_diff_eventness(frames, size=32) if frames else []
            if len(v) < 10:
                v = list(v) + [0.0] * (10 - len(v))
            visual_scores_by_clip[cid] = [float(x) for x in v[:10]]
    elif str(eventness_method) in (
        "vision_clipdiff",
        "av_fused_clipdiff",
        "av_fused_clipdiff_prod",
        "moe_energy_clipdiff",
        "energy_autoshift_clipdiff",
        "energy_autoshift_clipdiff_pos",
    ):
        # Cached CLIP embedding diffs (semantic motion) as a cheap visual eventness proxy.
        visual_scores_by_clip = {}
        for cid in eval_ids:
            cache = FeatureCache.load_npz(caches_dir / f"{cid}.npz")
            vis_res = 160
            if vis_res not in cache.features_by_resolution:
                vis_res = int(min(cache.features_by_resolution))
            feats = cache.features_by_resolution[int(vis_res)]
            v = clip_feature_diff_eventness(feats, metric="cosine")
            if len(v) < 10:
                v = list(v) + [0.0] * (10 - len(v))
            visual_scores_by_clip[cid] = [float(x) for x in v[:10]]

    rows: list[dict] = []
    for shift_s in shift_grid:
        for snr_db in snr_grid:
            for silence_ratio in silence_grid:
                # Deterministic seed per condition.
                seed = int((shift_s + 10.0) * 1000 + (snr_db + 100.0) * 10 + silence_ratio * 100)
                rng = np.random.default_rng(seed)

                recalls: dict[str, list[float]] = {str(d): [] for d in deltas}
                fallback_used = 0
                used = 0

                for cid in eval_ids:
                    wav_path = processed_dir / cid / "audio.wav"
                    if str(eventness_method) == "vision_clipdiff":
                        scores = [float(x) for x in (visual_scores_by_clip or {}).get(cid, [])]
                    elif str(eventness_method) in ("av_clipdiff_mlp", "av_clipdiff_framediff_mlp", "av_clipdiff_flow_mlp", "av_clipdiff_flow_mlp_stride"):
                        if av_model is None or clipdiff_scores_by_clip is None:
                            raise ValueError(f"{eventness_method} model/proxies not initialized")

                        audio, sr = load_wav_mono(wav_path)
                        x = shift_audio(audio=audio, sample_rate=int(sr), shift_s=float(shift_s))
                        x = add_noise_snr_db(audio=x, snr_db=float(snr_db), rng=rng)
                        x = apply_silence_ratio(audio=x, silence_ratio=float(silence_ratio), rng=rng)

                        a = audio_features_per_second_from_array(
                            x, int(sr), num_segments=10, feature_set="basic"
                        ).astype(np.float32, copy=False)
                        cd = clipdiff_scores_by_clip.get(cid) or []

                        v_cd = np.zeros((10, 1), dtype=np.float32)
                        for t, s in enumerate(cd[:10]):
                            v_cd[int(t), 0] = float(s)

                        if str(eventness_method) == "av_clipdiff_framediff_mlp":
                            if framediff_scores_by_clip is None:
                                raise ValueError("av_clipdiff_framediff_mlp proxies not initialized")
                            fd = framediff_scores_by_clip.get(cid) or []
                            v_fd = np.zeros((10, 1), dtype=np.float32)
                            for t, s in enumerate(fd[:10]):
                                v_fd[int(t), 0] = float(s)
                            feats = np.concatenate([a, v_cd, v_fd], axis=1).astype(np.float32, copy=False)
                        elif str(eventness_method) in ("av_clipdiff_flow_mlp", "av_clipdiff_flow_mlp_stride"):
                            if flow_scores_by_clip is None:
                                raise ValueError("av_clipdiff_flow_mlp proxies not initialized")
                            flow = flow_scores_by_clip.get(cid) or []
                            v_flow = np.zeros((10, 1), dtype=np.float32)
                            for t, s in enumerate(flow[:10]):
                                v_flow[int(t), 0] = float(s)
                            feats = np.concatenate([a, v_cd, v_flow], axis=1).astype(np.float32, copy=False)
                        else:
                            feats = np.concatenate([a, v_cd], axis=1).astype(np.float32, copy=False)

                        import torch

                        with torch.no_grad():
                            logits = av_model(torch.from_numpy(feats).float()).squeeze(-1)
                        s_np = logits.detach().cpu().numpy().astype("float32")
                        if str(eventness_method) == "av_clipdiff_flow_mlp_stride":
                            s_np = np.asarray(
                                stride_max_pool_per_second(
                                    [float(x) for x in s_np.tolist()],
                                    num_segments=10,
                                    stride_s=0.2,
                                    win_s=0.6,
                                ),
                                dtype=np.float32,
                            )
                        scores = [float(x) for x in s_np.tolist()]
                    else:
                        scores = _eventness_scores_from_augmented(
                            wav_path=wav_path,
                            num_segments=10,
                            method=str(eventness_method),
                            shift_s=float(shift_s),
                            snr_db=float(snr_db),
                            silence_ratio=float(silence_ratio),
                            rng=rng,
                            audio_model=audio_model,
                            audio_feature_set=audio_feature_set,
                            num_classes=int(index.num_classes),
                            panns_probe=panns_probe,
                            ast_probe=ast_probe,
                            ast_lr_model=ast_lr_model,
                            ast_emb_lr_model=ast_emb_lr_model,
                            ast_evt_mlp_model=ast_evt_mlp_model,
                            ast_mlp_model=ast_mlp_model,
                        )

                    if str(eventness_method) in ("av_fused", "av_fused_prod", "av_fused_clipdiff", "av_fused_clipdiff_prod"):
                        v = (visual_scores_by_clip or {}).get(cid, [])
                        if str(eventness_method) in ("av_fused", "av_fused_clipdiff"):
                            scores = fuse_max(
                                minmax_01([float(x) for x in scores]),
                                minmax_01([float(x) for x in v]),
                                num_segments=10,
                            )
                        else:
                            scores = fuse_prod(
                                minmax_01([float(x) for x in scores]),
                                minmax_01([float(x) for x in v]),
                                num_segments=10,
                            )
                    elif str(eventness_method) == "moe_energy_clipdiff":
                        v = (visual_scores_by_clip or {}).get(cid, [])
                        audio_raw = [float(x) for x in scores]
                        if float(np.asarray(audio_raw, dtype=np.float32).std()) < 1.0:
                            scores = minmax_01([float(x) for x in v])
                        else:
                            scores = audio_raw
                    elif str(eventness_method) in ("energy_autoshift_clipdiff", "energy_autoshift_clipdiff_pos"):
                        v = (visual_scores_by_clip or {}).get(cid, [])
                        audio_raw = [float(x) for x in scores]
                        a01 = minmax_01(audio_raw)
                        v01 = minmax_01([float(x) for x in v])
                        if str(eventness_method) == "energy_autoshift_clipdiff_pos":
                            shifts = [0, 1, 2]
                        else:
                            shifts = [-2, -1, 0, 1, 2]
                        s = best_shift_by_corr(a01, v01, shifts=shifts)
                        scores = shift_scores(audio_raw, int(s))

                    shift_used = (
                        0
                        if str(eventness_method) in ("energy_autoshift_clipdiff", "energy_autoshift_clipdiff_pos")
                        else int(anchor_shift_segments)
                    )
                    sel = anchors_from_scores_with_debug(
                        scores,
                        k=int(k),
                        num_segments=10,
                        shift=int(shift_used),
                        std_threshold=0.0,
                        select=str(anchor_select),
                        nms_radius=int(anchor_nms_radius),
                        nms_strong_gap=float(anchor_nms_strong_gap),
                        anchor_window=int(anchor_window),
                        smooth_window=int(anchor_smooth_window),
                        smooth_mode=str(anchor_smooth_mode),
                        conf_metric=None,
                        conf_threshold=None,
                    )
                    anchors = [int(x) for x in sel.anchors]
                    fallback_used += 1 if bool(sel.fallback_used) else 0
                    used += 1

                    clip = clip_by_id.get(cid)
                    if clip is None:
                        continue
                    gt = [i for i, lab in enumerate(index.segment_labels(clip)) if int(lab) != 0]
                    for d in deltas:
                        recalls[str(d)].append(float(recall_at_k(gt, anchors, num_segments=10, delta=int(d)).recall))

                row = {
                    "shift_s": float(shift_s),
                    "snr_db": float(snr_db),
                    "silence_ratio": float(silence_ratio),
                    "num_eval_ids": int(len(eval_ids)),
                    "anchors_fallback_used_frac": float(fallback_used / max(1, used)),
                    "recall_by_delta": {
                        str(d): float(np.mean(np.asarray(recalls[str(d)], dtype=np.float32))) if recalls[str(d)] else 0.0 for d in deltas
                    },
                }
                rows.append(row)

    out = {
        "ok": True,
        "mode": "ave_official",
        "meta_dir": str(meta_dir),
        "processed_dir": str(processed_dir),
        "caches_dir": str(caches_dir),
        "split_train": str(split_train),
        "split_eval": str(split_eval),
        "num_train_ids": int(len(train_ids)),
        "num_eval_ids": int(len(eval_ids)),
        "eventness_method": str(eventness_method),
        "uses_visual": bool(
            str(eventness_method)
            in (
                "av_fused",
                "av_fused_prod",
                "av_fused_clipdiff",
                "av_fused_clipdiff_prod",
                "vision_clipdiff",
                "moe_energy_clipdiff",
                "energy_autoshift_clipdiff",
                "energy_autoshift_clipdiff_pos",
            )
        ),
        "anchor_cfg": {
            "k": int(k),
            "anchor_select": str(anchor_select),
            "anchor_nms_radius": int(anchor_nms_radius),
            "anchor_nms_strong_gap": float(anchor_nms_strong_gap),
            "anchor_shift_segments": int(anchor_shift_segments),
            "anchor_window": int(anchor_window),
            "anchor_smooth_window": int(anchor_smooth_window),
            "anchor_smooth_mode": str(anchor_smooth_mode),
        },
        "grid": {"shift_s": shift_grid, "snr_db": snr_grid, "silence_ratio": silence_grid, "deltas": deltas},
        "rows": rows,
    }

    out_json = out_dir / "degradation_suite.json"
    out_json.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"out_dir": str(out_dir), "out_json": str(out_json), "num_rows": len(rows)}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Listen-then-Look degradation suite (toy + AVE official).")
    p.add_argument("--mode", type=str, default="toy", choices=["toy", "ave_official"])
    p.add_argument("--out-dir", type=Path, default=Path("runs") / f"E0203_degradation_{time.strftime('%Y%m%d-%H%M%S')}")

    # AVE official params.
    p.add_argument("--meta-dir", type=Path, default=ave_paths().meta_dir)
    p.add_argument("--processed-dir", type=Path, default=ave_paths().processed_dir)
    p.add_argument("--caches-dir", type=Path, required=False, default=None)
    p.add_argument("--train-ids-file", type=Path, default=None)
    p.add_argument("--eval-ids-file", type=Path, default=None)
    p.add_argument("--split-train", type=str, default="train", choices=["train", "val", "test"])
    p.add_argument("--split-eval", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-eval", type=int, default=None)
    p.add_argument("--allow-missing", action="store_true")

    p.add_argument(
        "--eventness-method",
        type=str,
        default="energy",
        choices=[
            "energy",
            "energy_delta",
            "energy_stride_max",
            "energy_nonspeech_ast",
            "energy_autoshift_clipdiff",
            "energy_autoshift_clipdiff_pos",
            "av_fused",
            "av_fused_prod",
            "av_fused_clipdiff",
            "av_fused_clipdiff_prod",
            "moe_energy_clipdiff",
            "vision_clipdiff",
            "av_clipdiff_mlp",
            "av_clipdiff_speech_mlp",
            "av_clipdiff_framediff_mlp",
            "av_clipdiff_flow_mlp",
            "av_clipdiff_flow_mlp_stride",
            "ast",
            "ast_lr",
            "ast_emb_lr",
            "ast_evt_mlp",
            "ast_mlp_cls",
            "ast_mlp_cls_target",
            "panns",
            # Supervised audio-only eventness (computed from AVE train split).
            "audio_basic_lr",
            "audio_basic_mlp",
            "audio_basic_tcn",
            "audio_fbank_mlp",
            "audio_fbank_tcn",
            "audio_basic_mlp_cls",
            "audio_basic_mlp_cls_target",
        ],
    )
    p.add_argument("--audio-device", type=str, default="cpu", help="Device for audio probes like PANNs (e.g., cpu, cuda:0).")
    p.add_argument("--ast-pretrained", action="store_true", help="Use pretrained AST weights (downloads from HF).")
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--anchor-select", type=str, default="topk", choices=["topk", "nms", "nms_strong", "window_topk"])
    p.add_argument("--anchor-nms-radius", type=int, default=2)
    p.add_argument("--anchor-nms-strong-gap", type=float, default=0.6)
    p.add_argument("--anchor-shift-segments", type=int, default=0)
    p.add_argument("--anchor-window", type=int, default=3)
    p.add_argument("--anchor-smooth-window", type=int, default=0)
    p.add_argument("--anchor-smooth-mode", type=str, default="mean", choices=["mean", "sum"])
    p.add_argument("--deltas", type=str, default="0,1,2", help="Comma-separated dilation deltas for Recall@K,Δ.")

    # Reduced-by-default grids (full grid can be supplied explicitly).
    p.add_argument("--shift-grid", type=str, default="-0.5,0,0.5")
    p.add_argument("--snr-grid", type=str, default="20,10,0")
    p.add_argument("--silence-grid", type=str, default="0,0.5")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "toy":
        rep = run_toy_degradation_suite(out_dir=args.out_dir)
        print(rep["out_json"])
        return 0
    if args.mode == "ave_official":
        caches_dir = Path(args.caches_dir) if args.caches_dir is not None else (Path("runs") / "REAL_AVE_LOCAL" / "caches")
        deltas = [int(x) for x in str(args.deltas).split(",") if str(x).strip()]
        rep = run_ave_official_degradation_suite(
            out_dir=args.out_dir,
            meta_dir=Path(args.meta_dir),
            processed_dir=Path(args.processed_dir),
            caches_dir=caches_dir,
            split_train=str(args.split_train),
            split_eval=str(args.split_eval),
            train_ids_file=args.train_ids_file,
            eval_ids_file=args.eval_ids_file,
            limit_train=int(args.limit_train) if args.limit_train is not None else None,
            limit_eval=int(args.limit_eval) if args.limit_eval is not None else None,
            allow_missing=bool(args.allow_missing),
            eventness_method=str(args.eventness_method),
            k=int(args.k),
            anchor_select=str(args.anchor_select),
            anchor_nms_radius=int(args.anchor_nms_radius),
            anchor_nms_strong_gap=float(args.anchor_nms_strong_gap),
            anchor_shift_segments=int(args.anchor_shift_segments),
            anchor_window=int(args.anchor_window),
            anchor_smooth_window=int(args.anchor_smooth_window),
            anchor_smooth_mode=str(args.anchor_smooth_mode),
            audio_device=str(args.audio_device),
            ast_pretrained=bool(args.ast_pretrained),
            shift_grid=_parse_csv_floats(args.shift_grid),
            snr_grid=_parse_csv_floats(args.snr_grid),
            silence_grid=_parse_csv_floats(args.silence_grid),
            deltas=deltas,
        )
        print(rep["out_json"])
        return 0
    raise SystemExit(f"unknown mode: {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
