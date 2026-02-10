from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from avs.utils.scores import minmax_01


_WORD_RE = re.compile(r"[A-Za-z0-9]+")

# Lightweight in-process caches to avoid repeatedly re-loading large models during eval loops.
# This keeps evaluation deterministic (no randomness introduced) and drastically reduces overhead.
_ASR_MODEL_CACHE: dict[tuple[str, str, str], object] = {}
_CLAP_PROBE_CACHE: dict[tuple[str, str, str, bool], object] = {}


def _tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(str(text))]


@dataclass(frozen=True)
class RelevanceArtifact:
    kind: str
    cache_path: Path
    num_segments: int
    ok: bool
    details: dict


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def relevance_from_tfidf(per_sec_text: list[str], query: str) -> list[float]:
    """
    TFIDF cosine similarity between query and per-second ASR text.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    docs = [str(x) for x in per_sec_text]
    q = str(query)
    if not q.strip():
        return [0.0 for _ in per_sec_text]
    vec = TfidfVectorizer(lowercase=True, token_pattern=r"[A-Za-z0-9]+")
    try:
        X = vec.fit_transform(docs + [q])
    except ValueError:
        # e.g. empty vocabulary (all docs and query empty after tokenization)
        return [0.0 for _ in per_sec_text]
    # last row is query
    qv = X[-1]
    dv = X[:-1]
    # cosine sim since rows are L2-normalized in TFIDF (scikit does normalize by default)
    sims = (dv @ qv.T).toarray().reshape(-1)
    return [float(x) for x in sims.tolist()]


def relevance_from_bm25(per_sec_text: list[str], query: str) -> list[float]:
    """
    BM25 relevance between query tokens and per-second tokens.
    """
    from rank_bm25 import BM25Okapi

    docs_tok = [_tokenize(x) for x in per_sec_text]
    if not docs_tok or not any(len(toks) > 0 for toks in docs_tok):
        return [0.0 for _ in per_sec_text]
    bm25 = BM25Okapi(docs_tok)
    q_tok = _tokenize(query)
    if not q_tok:
        return [0.0 for _ in per_sec_text]
    scores = bm25.get_scores(q_tok)
    return [float(x) for x in scores.tolist()]


def normalize_relevance(scores: list[float]) -> list[float]:
    return minmax_01([float(x) for x in scores])


def load_or_compute_asr_per_sec_text(
    *,
    wav_path: Path,
    num_segments: int,
    cache_path: Path,
    model_size: str = "tiny.en",
    device: str = "cpu",
    compute_type: str | None = None,
    language: str | None = "en",
) -> tuple[list[str], RelevanceArtifact]:
    """
    Compute per-second transcript text using faster-whisper (with word timestamps) and cache it.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        obj = json.loads(cache_path.read_text(encoding="utf-8"))
        txt = obj.get("per_sec_text")
        if isinstance(txt, list) and len(txt) >= int(num_segments):
            return [str(x) for x in txt[: int(num_segments)]], RelevanceArtifact(
                kind="asr_per_sec",
                cache_path=cache_path,
                num_segments=int(num_segments),
                ok=True,
                details={"cached": True, "model_size": obj.get("model_size"), "device": obj.get("device")},
            )

    from faster_whisper import WhisperModel  # heavy import

    dev = str(device)
    ct = "auto" if compute_type is None else str(compute_type)
    if ct == "auto":
        d = dev.lower()
        # faster-whisper/ctranslate2 doesn't support efficient float16 on CPU.
        if d == "cpu" or d.startswith("cpu"):
            ct = "int8"
        elif d == "cuda" or d.startswith("cuda"):
            ct = "float16"
        else:
            ct = "float32"

    t0 = time.time()
    key = (str(model_size), str(device), str(ct))
    model = _ASR_MODEL_CACHE.get(key)
    if model is None:
        model = WhisperModel(str(model_size), device=str(device), compute_type=str(ct))
        _ASR_MODEL_CACHE[key] = model
    segments, _info = model.transcribe(
        str(wav_path),
        language=None if language is None else str(language),
        word_timestamps=True,
        vad_filter=True,
        beam_size=1,
    )

    # Build per-second text.
    per_sec: list[list[str]] = [[] for _ in range(int(num_segments))]
    words_out: list[dict] = []
    for seg in segments:
        for w in seg.words or []:
            wtext = str(getattr(w, "word", "")).strip()
            if not wtext:
                continue
            start = float(getattr(w, "start", 0.0))
            end = float(getattr(w, "end", start))
            mid = 0.5 * (start + end)
            sec = int(mid)
            if 0 <= sec < int(num_segments):
                per_sec[sec].append(wtext)
            words_out.append({"word": wtext, "start": float(start), "end": float(end)})

    per_sec_text = [" ".join(ws) for ws in per_sec]
    payload = {
        "ok": True,
        "kind": "asr_per_sec",
        "wav_path": str(wav_path),
        "num_segments": int(num_segments),
        "model_size": str(model_size),
        "device": str(device),
        "compute_type": str(ct),
        "language": language,
        "elapsed_s": float(time.time() - t0),
        "words": words_out,
        "per_sec_text": per_sec_text,
    }
    _write_json(cache_path, payload)
    return per_sec_text, RelevanceArtifact(
        kind="asr_per_sec",
        cache_path=cache_path,
        num_segments=int(num_segments),
        ok=True,
        details={"cached": False, "elapsed_s": payload["elapsed_s"], "words": len(words_out)},
    )


def load_or_compute_clap_audio_embeddings(
    *,
    wav_path: Path,
    num_segments: int,
    cache_path: Path,
    model_name: str = "laion/clap-htsat-fused",
    device: str = "cpu",
    dtype: str = "float32",
    batch_size: int = 16,
    pretrained: bool = True,
) -> tuple[np.ndarray, RelevanceArtifact]:
    """
    Compute CLAP audio embeddings per second and cache to npz.

    Cache schema:
      - emb: float32 [T, D]
      - meta.json sidecar with config (model_name/device/dtype)
    """
    cache_path = Path(cache_path)
    meta_path = cache_path.with_suffix(".json")
    if cache_path.exists() and meta_path.exists():
        with np.load(cache_path) as z:
            emb = np.asarray(z["emb"], dtype=np.float32)
        if emb.ndim == 2 and int(emb.shape[0]) >= int(num_segments):
            return emb[: int(num_segments)], RelevanceArtifact(
                kind="clap_audio_emb",
                cache_path=cache_path,
                num_segments=int(num_segments),
                ok=True,
                details={"cached": True, "meta": json.loads(meta_path.read_text(encoding="utf-8"))},
            )

    from avs.audio.clap_probe import ClapProbe, ClapProbeConfig

    t0 = time.time()
    key = (str(model_name), str(device), str(dtype), bool(pretrained))
    probe = _CLAP_PROBE_CACHE.get(key)
    if probe is None:
        probe = ClapProbe(ClapProbeConfig(model_name=str(model_name), pretrained=bool(pretrained), device=str(device), dtype=str(dtype)))
        _CLAP_PROBE_CACHE[key] = probe
    # Use a batched implementation to avoid huge processor inputs on long videos.
    emb = probe.audio_embeddings_per_second(wav_path, num_segments=int(num_segments), batch_size=int(batch_size))
    emb = np.asarray(emb, dtype=np.float32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, emb=emb)
    meta = {
        "ok": True,
        "kind": "clap_audio_emb",
        "wav_path": str(wav_path),
        "num_segments": int(num_segments),
        "model_name": str(model_name),
        "device": str(device),
        "dtype": str(dtype),
        "batch_size": int(batch_size),
        "pretrained": bool(pretrained),
        "elapsed_s": float(time.time() - t0),
        "shape": [int(x) for x in emb.shape],
    }
    _write_json(meta_path, meta)
    return emb, RelevanceArtifact(
        kind="clap_audio_emb",
        cache_path=cache_path,
        num_segments=int(num_segments),
        ok=True,
        details={"cached": False, "elapsed_s": meta["elapsed_s"], "shape": meta["shape"]},
    )


def clap_query_relevance_from_embeddings(audio_emb: np.ndarray, query: str, *, model_name: str = "laion/clap-htsat-fused", device: str = "cpu") -> list[float]:
    """
    Compute per-second CLAP query relevance: cos(audio_emb[t], text_emb(query)).
    """
    from avs.audio.clap_probe import ClapProbe, ClapProbeConfig

    key = (str(model_name), str(device), "float32", True)
    probe = _CLAP_PROBE_CACHE.get(key)
    if probe is None:
        probe = ClapProbe(ClapProbeConfig(model_name=str(model_name), pretrained=True, device=str(device), dtype="float32"))
        _CLAP_PROBE_CACHE[key] = probe
    text_emb = probe.text_embeddings([str(query)])  # [1, D]
    a = np.asarray(audio_emb, dtype=np.float32)
    t = np.asarray(text_emb[0], dtype=np.float32)
    # audio_emb and text_emb are already L2-normalized in ClapProbe.
    sims = a @ t.reshape(-1, 1)
    sims = sims.reshape(-1)
    return [float(x) for x in sims.tolist()]
