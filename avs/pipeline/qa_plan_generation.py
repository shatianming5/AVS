from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np

from avs.audio.eventness import anchors_from_scores, compute_eventness_wav_energy_stride_max
from avs.qa.query_relevance import (
    clap_query_relevance_from_embeddings,
    clip_query_relevance_from_embeddings,
    load_or_compute_asr_per_sec_text,
    load_or_compute_clap_audio_embeddings,
    load_or_compute_clip_image_embeddings,
    normalize_relevance,
    relevance_from_bm25,
    relevance_from_tfidf,
)
from avs.qa.reliability import reliability_and_alpha
from avs.utils.scores import fuse_max, minmax_01
from avs.vision.cheap_eventness import frame_diff_eventness, list_frames, optical_flow_mag_eventness


@dataclass(frozen=True)
class QASecSelection:
    selected_seconds: list[int]
    background_seconds: list[int]
    anchor_seconds: list[int]
    anchors: list[int]
    alpha: float
    q_bar: float
    reliability_metric: str

    def to_jsonable(self) -> dict:
        return {
            "selected_seconds": [int(x) for x in self.selected_seconds],
            "background_seconds": [int(x) for x in self.background_seconds],
            "anchor_seconds": [int(x) for x in self.anchor_seconds],
            "anchors": [int(x) for x in self.anchors],
            "alpha": float(self.alpha),
            "q_bar": float(self.q_bar),
            "reliability_metric": str(self.reliability_metric),
        }


def uniform_seconds(duration_seconds: int, n: int) -> list[int]:
    """Deterministic seconds selection spread uniformly across `[0, duration_seconds)`."""
    return _uniform_seconds(duration_seconds, n)


def random_seconds(duration_seconds: int, n: int, *, seed: int) -> list[int]:
    """Deterministic random seconds selection (uses `random.Random(seed)`)."""
    return _random_seconds(duration_seconds, n, seed=int(seed))


def _uniform_seconds(duration_seconds: int, n: int) -> list[int]:
    dur = int(duration_seconds)
    k = int(n)
    if dur <= 0 or k <= 0:
        return []
    if k >= dur:
        return list(range(dur))
    if k == 1:
        return [dur // 2]
    out = []
    for i in range(k):
        t = int(round(float(i) * float(dur - 1) / float(k - 1)))
        if t not in out:
            out.append(t)
    # pad if rounding caused duplicates
    if len(out) < k:
        for t in range(dur):
            if t not in out:
                out.append(t)
            if len(out) >= k:
                break
    return out[:k]


def _random_seconds(duration_seconds: int, n: int, *, seed: int) -> list[int]:
    dur = int(duration_seconds)
    k = int(n)
    if dur <= 0 or k <= 0:
        return []
    if k >= dur:
        return list(range(dur))
    rng = random.Random(int(seed))
    return sorted(rng.sample(list(range(dur)), k))


def _expand_anchors(anchors: list[int], *, radius: int, duration_seconds: int) -> list[int]:
    dur = int(duration_seconds)
    r = max(0, int(radius))
    out: list[int] = []
    seen: set[int] = set()
    for a in [int(x) for x in anchors]:
        # Prefer center, then outward.
        for d in range(0, r + 1):
            for t in (a - d, a + d) if d > 0 else (a,):
                if 0 <= int(t) < dur and int(t) not in seen:
                    out.append(int(t))
                    seen.add(int(t))
    return out


def select_seconds_alpha_mixture(
    *,
    scores: list[float],
    budget_frames: int,
    alpha_policy: str = "reliability",
    alpha_fixed: float = 0.3,
    reliability_metric: str = "top3_bottom3_gap_norm",
    alpha_min: float = 0.10,
    alpha_max: float = 0.60,
    k_anchors: int = 10,
    nms_radius: int = 1,
    anchor_radius: int = 2,
    seed: int = 0,
) -> QASecSelection:
    xs = [float(x) for x in scores]
    dur = int(len(xs))
    b = int(budget_frames)
    if b <= 0:
        return QASecSelection(
            selected_seconds=[],
            background_seconds=[],
            anchor_seconds=[],
            anchors=[],
            alpha=float(alpha_fixed),
            q_bar=0.0,
            reliability_metric=str(reliability_metric),
        )
    if dur <= 0:
        raise ValueError("scores is empty")

    rep = reliability_and_alpha(xs, metric=str(reliability_metric), alpha_min=float(alpha_min), alpha_max=float(alpha_max))
    if str(alpha_policy) == "fixed":
        alpha = float(max(0.0, min(1.0, float(alpha_fixed))))
        q_bar = float(rep.q_bar)
    elif str(alpha_policy) == "reliability":
        alpha = float(rep.alpha)
        q_bar = float(rep.q_bar)
    else:
        raise ValueError(f"unknown alpha_policy={alpha_policy!r}; expected 'fixed' or 'reliability'")

    n_back = int(round(float(alpha) * float(b)))
    n_back = max(0, min(b, n_back))
    n_anchor = int(b - n_back)

    back = uniform_seconds(dur, n_back) if n_back > 0 else []

    # Anchor proposal (query-aware scores already included if desired).
    k = min(int(k_anchors), dur)
    anchors = anchors_from_scores(xs, k=int(k), num_segments=int(dur), select="nms", nms_radius=int(nms_radius))
    expanded = _expand_anchors(anchors, radius=int(anchor_radius), duration_seconds=int(dur))

    anchor_seconds: list[int] = []
    chosen: set[int] = set(int(t) for t in back)
    for t in expanded:
        if len(anchor_seconds) >= int(n_anchor):
            break
        if int(t) in chosen:
            continue
        anchor_seconds.append(int(t))
        chosen.add(int(t))

    if len(anchor_seconds) < int(n_anchor):
        # Fill remaining anchor budget by score ranking.
        order = sorted(range(dur), key=lambda i: (-float(xs[i]), int(i)))
        for t in order:
            if len(anchor_seconds) >= int(n_anchor):
                break
            if int(t) in chosen:
                continue
            anchor_seconds.append(int(t))
            chosen.add(int(t))

    selected = sorted(list(chosen))
    # If still short (duration < budget), just take everything.
    if len(selected) < b:
        for t in range(dur):
            if t not in chosen:
                selected.append(int(t))
                chosen.add(int(t))
            if len(selected) >= b:
                break
        selected = sorted(selected)

    # Deterministic: if we over-selected due to rounding/dupes, trim by score then time.
    if len(selected) > b:
        selected.sort(key=lambda t: (-float(xs[int(t)]), int(t)))
        selected = sorted([int(t) for t in selected[:b]])

    return QASecSelection(
        selected_seconds=[int(x) for x in selected],
        background_seconds=[int(x) for x in back],
        anchor_seconds=[int(x) for x in anchor_seconds],
        anchors=[int(x) for x in anchors],
        alpha=float(alpha),
        q_bar=float(q_bar),
        reliability_metric=str(reliability_metric),
    )


def _gumbel_topk_indices(weights: list[float], k: int, *, seed: int) -> list[int]:
    """
    Deterministic Gumbel-TopK sampling (Q-Frame-style core primitive).

    We treat `weights` as non-negative unnormalized probabilities.
    """
    n = int(len(weights))
    kk = max(0, min(int(k), n))
    if kk <= 0:
        return []
    rng = random.Random(int(seed))
    eps = 1e-12
    scored: list[tuple[float, int]] = []
    for i, w in enumerate(weights):
        ww = float(w)
        ww = ww if ww > eps else eps
        # Gumbel noise: -log(-log(U))
        u = rng.random()
        u = u if u > eps else eps
        g = -math.log(-math.log(u))
        s = math.log(ww) + float(g)
        scored.append((float(s), int(i)))
    scored.sort(key=lambda x: (-float(x[0]), int(x[1])))
    return [int(i) for _, i in scored[:kk]]


def _greedy_maxvol_indices(emb: np.ndarray, k: int) -> list[int]:
    """
    MaxInfo/MaxVol-style greedy diversity selection via pivoted Gram-Schmidt.
    """
    x = np.asarray(emb, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected emb as 2D array, got shape={x.shape}")
    n = int(x.shape[0])
    kk = max(0, min(int(k), n))
    if kk <= 0:
        return []

    norms2 = np.einsum("ij,ij->i", x, x)
    i0 = int(np.argmax(norms2))
    selected = [i0]
    # Orthonormal basis rows.
    q: list[np.ndarray] = []
    v = x[i0].copy()
    nv = float(np.linalg.norm(v))
    if nv > 1e-12:
        q.append(v / nv)

    resid2 = norms2.copy()
    if q:
        proj = x @ q[0]
        resid2 = resid2 - proj * proj
        resid2 = np.maximum(resid2, 0.0)

    while len(selected) < kk:
        # Pick the largest residual norm among unselected.
        resid2[selected] = -1.0
        i = int(np.argmax(resid2))
        if resid2[i] <= 1e-12:
            break
        selected.append(i)

        v = x[i].copy()
        for qi in q:
            v = v - float(np.dot(v, qi)) * qi
        nv = float(np.linalg.norm(v))
        if nv <= 1e-12:
            continue
        qi = v / nv
        q.append(qi)
        proj = x @ qi
        resid2 = resid2 - proj * proj
        resid2 = np.maximum(resid2, 0.0)

    if len(selected) < kk:
        # Fill by original norm if we got rank-deficient vectors.
        rest = [i for i in np.argsort(-norms2) if int(i) not in set(selected)]
        selected.extend([int(i) for i in rest[: (kk - len(selected))]])

    return sorted({int(i) for i in selected})[:kk]


def _greedy_dpp_map_indices(L: np.ndarray, k: int) -> list[int]:
    """
    Greedy MAP inference for an L-ensemble DPP.

    This is a simple (but deterministic) Schur-complement greedy: pick items with
    maximum marginal `L_ii - L_iS L_SS^{-1} L_Si`.
    """
    L = np.asarray(L, dtype=np.float64)
    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError(f"expected square L, got shape={L.shape}")
    n = int(L.shape[0])
    kk = max(0, min(int(k), n))
    if kk <= 0:
        return []

    diag = np.diag(L).copy()
    selected: list[int] = []
    remaining = set(range(n))

    for _ in range(kk):
        if not remaining:
            break
        if not selected:
            i = int(max(remaining, key=lambda j: (float(diag[j]), -int(j))))
            selected.append(i)
            remaining.remove(i)
            continue

        S = selected
        Lss = L[np.ix_(S, S)]
        try:
            inv = np.linalg.inv(Lss)
        except Exception:
            inv = np.linalg.pinv(Lss)

        best_i = None
        best_m = -1e30
        for i in remaining:
            v = L[np.ix_([i], S)]
            m = float(L[i, i] - (v @ inv @ v.T)[0, 0])
            if m > best_m:
                best_m = m
                best_i = int(i)
        if best_i is None or best_m <= 1e-12:
            break
        selected.append(int(best_i))
        remaining.remove(int(best_i))

    return sorted({int(i) for i in selected})[:kk]


def _segments(n: int, G: int) -> list[list[int]]:
    nn = int(n)
    gg = max(1, int(G))
    out: list[list[int]] = []
    for g in range(gg):
        lo = int(round(float(g) * float(nn) / float(gg)))
        hi = int(round(float(g + 1) * float(nn) / float(gg)))
        idx = list(range(lo, min(hi, nn)))
        if idx:
            out.append(idx)
    return out if out else [list(range(nn))]


def _allocate_segment_quotas(weights: list[float], k: int, segs: list[list[int]]) -> list[int]:
    kk = int(k)
    masses = [float(sum(float(weights[i]) for i in seg)) for seg in segs]
    total = float(sum(masses))
    if total <= 0.0:
        masses = [float(len(seg)) for seg in segs]
        total = float(sum(masses))
    raw = [float(kk) * float(m) / float(total) for m in masses]
    base = [int(math.floor(x)) for x in raw]
    # Cap by segment size.
    base = [min(int(b), int(len(segs[i]))) for i, b in enumerate(base)]
    rem = int(kk - sum(base))
    frac = [float(raw[i] - float(base[i])) for i in range(len(segs))]
    # Distribute remaining to segments with largest fractional part (and capacity).
    order = sorted(range(len(segs)), key=lambda i: (-float(frac[i]), -int(len(segs[i])), int(i)))
    i = 0
    while rem > 0 and i < len(order) * 2:
        si = int(order[i % len(order)])
        if base[si] < int(len(segs[si])):
            base[si] += 1
            rem -= 1
        i += 1
    # If still short (due to caps), fill any segment with capacity.
    if rem > 0:
        caps = [int(len(segs[i])) - int(base[i]) for i in range(len(segs))]
        order2 = sorted(range(len(segs)), key=lambda i: (-int(caps[i]), int(i)))
        for si in order2:
            take = min(int(rem), max(0, int(caps[si])))
            base[si] += int(take)
            rem -= int(take)
            if rem <= 0:
                break
    return [int(x) for x in base]


def _clip_rel_and_emb(
    *,
    processed_dir: Path,
    query: str,
    num_segments: int,
    clip_model_name: str,
    clip_device: str,
    clip_dtype: str,
    clip_resolution: int,
) -> tuple[np.ndarray, list[float], dict | None]:
    proc = Path(processed_dir)
    frames_dir = proc / "frames"
    cache_dir = proc / "q_l2l"
    emb_path = cache_dir / f"clip_image_emb_T{int(num_segments)}_r{int(clip_resolution)}.npz"
    emb, rel_art = load_or_compute_clip_image_embeddings(
        frames_dir=frames_dir,
        num_segments=int(num_segments),
        cache_path=emb_path,
        model_name=str(clip_model_name),
        device=str(clip_device),
        dtype=str(clip_dtype),
        resolution=int(clip_resolution),
        batch_size=32,
        pretrained=True,
    )
    rel_raw = clip_query_relevance_from_embeddings(emb[: int(num_segments)], str(query), model_name=str(clip_model_name), device=str(clip_device))
    rel = normalize_relevance(rel_raw)
    art = None if rel_art is None else {"kind": rel_art.kind, "path": str(rel_art.cache_path), **rel_art.details}
    return np.asarray(emb[: int(num_segments)]), [float(x) for x in rel], art


def select_seconds_qframe_gumbel_clip(
    *,
    processed_dir: Path,
    query: str,
    num_segments: int,
    budget_frames: int,
    seed: int,
    clip_model_name: str = "openai/clip-vit-base-patch16",
    clip_device: str = "cpu",
    clip_dtype: str = "float32",
    clip_resolution: int = 224,
) -> tuple[QASecSelection, dict]:
    """
    Q-Frame-like baseline: query-aware relevance + Gumbel-TopK sampling.
    """
    emb, rel, art = _clip_rel_and_emb(
        processed_dir=Path(processed_dir),
        query=str(query),
        num_segments=int(num_segments),
        clip_model_name=str(clip_model_name),
        clip_device=str(clip_device),
        clip_dtype=str(clip_dtype),
        clip_resolution=int(clip_resolution),
    )
    chosen = sorted(_gumbel_topk_indices(rel, int(budget_frames), seed=int(seed)))
    sel = QASecSelection(
        selected_seconds=[int(x) for x in chosen],
        background_seconds=[],
        anchor_seconds=[int(x) for x in chosen],
        anchors=[],
        alpha=0.0,
        q_bar=float(sum(rel) / max(1, len(rel))),
        reliability_metric="clip_rel_gumbel_topk",
    )
    return sel, {"clip_relevance_norm": rel, "relevance_artifact": art, "emb_shape": list(emb.shape)}


def select_seconds_maxinfo_maxvol_clip(
    *,
    processed_dir: Path,
    query: str,
    num_segments: int,
    budget_frames: int,
    clip_model_name: str = "openai/clip-vit-base-patch16",
    clip_device: str = "cpu",
    clip_dtype: str = "float32",
    clip_resolution: int = 224,
) -> tuple[QASecSelection, dict]:
    """
    MaxInfo baseline: diversity selection via max-vol on CLIP image embeddings.
    """
    emb, rel, art = _clip_rel_and_emb(
        processed_dir=Path(processed_dir),
        query=str(query),
        num_segments=int(num_segments),
        clip_model_name=str(clip_model_name),
        clip_device=str(clip_device),
        clip_dtype=str(clip_dtype),
        clip_resolution=int(clip_resolution),
    )
    # L2 normalize per-second embeddings for scale-invariant diversity.
    x = np.asarray(emb, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    x = x / norms
    chosen = _greedy_maxvol_indices(x, int(budget_frames))
    sel = QASecSelection(
        selected_seconds=[int(x) for x in chosen],
        background_seconds=[],
        anchor_seconds=[int(x) for x in chosen],
        anchors=[],
        alpha=0.0,
        q_bar=float(sum(rel) / max(1, len(rel))),
        reliability_metric="clip_emb_maxvol",
    )
    return sel, {"clip_relevance_norm": rel, "relevance_artifact": art, "emb_shape": list(emb.shape)}


def select_seconds_mdp3_dpp_clip(
    *,
    processed_dir: Path,
    query: str,
    num_segments: int,
    budget_frames: int,
    seed: int,
    beta: float = 5.0,
    eps: float = 1e-3,
    segments: int = 8,
    clip_model_name: str = "openai/clip-vit-base-patch16",
    clip_device: str = "cpu",
    clip_dtype: str = "float32",
    clip_resolution: int = 224,
) -> tuple[QASecSelection, dict]:
    """
    MDP3-like baseline: list-wise diversity (DPP) with query-aware weighting, applied per segment.
    """
    emb, rel, art = _clip_rel_and_emb(
        processed_dir=Path(processed_dir),
        query=str(query),
        num_segments=int(num_segments),
        clip_model_name=str(clip_model_name),
        clip_device=str(clip_device),
        clip_dtype=str(clip_dtype),
        clip_resolution=int(clip_resolution),
    )
    x = np.asarray(emb, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    x = x / norms

    n = int(x.shape[0])
    segs = _segments(n, int(segments))
    quotas = _allocate_segment_quotas(rel, int(budget_frames), segs)

    selected: list[int] = []
    for seg, q in zip(segs, quotas):
        if q <= 0 or not seg:
            continue
        xs = x[np.asarray(seg)]
        S = xs @ xs.T
        w = np.exp(float(beta) * np.asarray([rel[i] for i in seg], dtype=np.float64))
        L = (w[:, None] * S) * w[None, :]
        L = L + float(eps) * np.eye(int(L.shape[0]), dtype=np.float64)
        pick_local = _greedy_dpp_map_indices(L, int(q))
        selected.extend([int(seg[i]) for i in pick_local])

    # Fill if short (e.g., many tiny segments).
    chosen = sorted({int(i) for i in selected})
    if len(chosen) < int(budget_frames):
        order = sorted(range(n), key=lambda i: (-float(rel[i]), int(i)))
        for i in order:
            if len(chosen) >= int(budget_frames):
                break
            if int(i) in chosen:
                continue
            chosen.append(int(i))
        chosen = sorted(chosen)
    chosen = chosen[: int(budget_frames)]

    sel = QASecSelection(
        selected_seconds=[int(x) for x in chosen],
        background_seconds=[],
        anchor_seconds=[int(x) for x in chosen],
        anchors=[],
        alpha=0.0,
        q_bar=float(sum(rel) / max(1, len(rel))),
        reliability_metric="clip_rel_dpp_segmented",
    )
    return sel, {
        "clip_relevance_norm": rel,
        "relevance_artifact": art,
        "emb_shape": list(emb.shape),
        "beta": float(beta),
        "eps": float(eps),
        "segments": int(segments),
        "segment_quotas": [int(x) for x in quotas],
        "seed": int(seed),
    }


def _combine_q_l2l(base_norm: list[float], rel_norm: list[float]) -> list[float]:
    # Query-aware modulation: keep non-zero floor so silence/empty transcript doesn't zero out.
    if not base_norm:
        return []
    if not rel_norm:
        return list(base_norm)
    n = min(len(base_norm), len(rel_norm))
    out = []
    for i in range(n):
        out.append(float(base_norm[i]) * (0.5 + 0.5 * float(rel_norm[i])))
    return out


def build_scores(
    *,
    processed_dir: Path,
    query: str,
    num_segments: int,
    method: str,
    seed: int = 0,
    ql2l_backend: str = "clap",
    clap_model_name: str = "laion/clap-htsat-fused",
    clap_device: str = "cpu",
    clip_model_name: str = "openai/clip-vit-base-patch16",
    clip_device: str = "cpu",
    clip_dtype: str = "float32",
    clip_resolution: int = 224,
    asr_model_size: str = "tiny.en",
    asr_device: str = "cpu",
    cheap_visual: str = "frame_diff",
) -> dict:
    """
    Build per-second scores for QA plan generation.

    Returns a JSONable dict including:
      - scores: list[float] length T
      - details: method-specific debug
    """
    t0 = time.time()
    proc = Path(processed_dir)
    wav_path = proc / "audio.wav"
    frames_dir = proc / "frames"

    if not wav_path.exists():
        raise FileNotFoundError(f"missing processed wav: {wav_path}")
    if not frames_dir.exists():
        raise FileNotFoundError(f"missing processed frames dir: {frames_dir}")

    # Audio base: dense stride max pooled to per-second.
    ev = compute_eventness_wav_energy_stride_max(wav_path, num_segments=int(num_segments), stride_s=0.2, win_s=0.4)
    audio = [float(x) for x in ev.scores]
    audio_norm = minmax_01(audio)

    # Cheap visual base.
    frames = list_frames(frames_dir)
    if cheap_visual == "frame_diff":
        v = frame_diff_eventness(frames, size=32)
    elif cheap_visual == "flow":
        v = optical_flow_mag_eventness(frames, size=64)
    else:
        raise ValueError(f"unknown cheap_visual={cheap_visual!r}; expected 'frame_diff' or 'flow'")
    vis = [float(x) for x in v[: int(num_segments)]]
    if len(vis) < int(num_segments):
        vis = vis + [0.0] * (int(num_segments) - len(vis))
    vis_norm = minmax_01(vis)

    m = str(method)
    details: dict = {"elapsed_s": float(time.time() - t0), "method": m}

    if m == "uniform":
        scores = [1.0] * int(num_segments)
    elif m == "random":
        rng = random.Random(int(seed))
        scores = [rng.random() for _ in range(int(num_segments))]
    elif m == "audio":
        scores = list(audio_norm)
    elif m == "cheap_visual":
        scores = list(vis_norm)
    elif m == "fused":
        scores = minmax_01(fuse_max(audio_norm, vis_norm, num_segments=int(num_segments)))
    elif m in ("ql2l_clap", "ql2l_asr_bm25", "ql2l_asr_tfidf"):
        base = minmax_01(fuse_max(audio_norm, vis_norm, num_segments=int(num_segments)))
        rel: list[float] = []
        rel_art = None
        if m == "ql2l_clap":
            cache_dir = proc / "q_l2l"
            emb_path = cache_dir / f"clap_audio_emb_T{int(num_segments)}.npz"
            emb, rel_art = load_or_compute_clap_audio_embeddings(
                wav_path=wav_path,
                num_segments=int(num_segments),
                cache_path=emb_path,
                model_name=str(clap_model_name),
                device=str(clap_device),
                dtype="float32",
                batch_size=16,
                pretrained=True,
            )
            rel_raw = clap_query_relevance_from_embeddings(emb[: int(num_segments)], str(query), model_name=str(clap_model_name), device=str(clap_device))
            rel = normalize_relevance(rel_raw)
        else:
            cache_dir = proc / "q_l2l"
            asr_path = cache_dir / f"asr_per_sec_T{int(num_segments)}_{str(asr_model_size).replace('/', '_')}.json"
            per_sec_text, rel_art = load_or_compute_asr_per_sec_text(
                wav_path=wav_path,
                num_segments=int(num_segments),
                cache_path=asr_path,
                model_size=str(asr_model_size),
                device=str(asr_device),
                compute_type=None,  # auto: int8 on CPU, float16 on CUDA
                language="en",
            )
            if m == "ql2l_asr_bm25":
                rel_raw = relevance_from_bm25(per_sec_text, str(query))
            else:
                rel_raw = relevance_from_tfidf(per_sec_text, str(query))
            rel = normalize_relevance(rel_raw)

        scores = minmax_01(_combine_q_l2l(base, rel))
        details["relevance_artifact"] = None if rel_art is None else {"kind": rel_art.kind, "path": str(rel_art.cache_path), **rel_art.details}
        details["relevance_norm"] = rel
        details["base_norm"] = base
    elif m == "ql2l_clip":
        base = minmax_01(fuse_max(audio_norm, vis_norm, num_segments=int(num_segments)))
        cache_dir = proc / "q_l2l"
        emb_path = cache_dir / f"clip_image_emb_T{int(num_segments)}_r{int(clip_resolution)}.npz"
        emb, rel_art = load_or_compute_clip_image_embeddings(
            frames_dir=frames_dir,
            num_segments=int(num_segments),
            cache_path=emb_path,
            model_name=str(clip_model_name),
            device=str(clip_device),
            dtype=str(clip_dtype),
            resolution=int(clip_resolution),
            batch_size=32,
            pretrained=True,
        )
        rel_raw = clip_query_relevance_from_embeddings(emb[: int(num_segments)], str(query), model_name=str(clip_model_name), device=str(clip_device))
        rel = normalize_relevance(rel_raw)
        scores = minmax_01(_combine_q_l2l(base, rel))
        details["relevance_artifact"] = None if rel_art is None else {"kind": rel_art.kind, "path": str(rel_art.cache_path), **rel_art.details}
        details["relevance_norm"] = rel
        details["base_norm"] = base
    else:
        raise ValueError(f"unknown method={method!r}")

    if len(scores) != int(num_segments):
        scores = [float(x) for x in scores[: int(num_segments)]]
        if len(scores) < int(num_segments):
            scores = scores + [0.0] * (int(num_segments) - len(scores))

    details["elapsed_s"] = float(time.time() - t0)
    return {"ok": True, "scores": [float(x) for x in scores], "audio_norm": audio_norm, "vis_norm": vis_norm, "details": details}


def write_scores_debug(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
