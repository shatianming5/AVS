from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from shared.schemas import StyleFeatures, TextBlock
from worker.pipeline.cache import load_stage_payload, save_stage_payload, sha256_text, stage_cache_path


_HEDGES = ["may", "might", "likely", "potentially", "could", "suggest", "appears", "appear"]
_CONNECTORS = [
    "however",
    "nevertheless",
    "yet",
    "but",
    "therefore",
    "thus",
    "moreover",
    "furthermore",
    "in summary",
    "in this paper",
]

STYLE_CACHE_VERSION = 1


def extract_style_features(intro_blocks: list[TextBlock]) -> StyleFeatures:
    texts = [b.clean_text or b.text or "" for b in intro_blocks]
    joined = " ".join(texts)
    words = re.findall(r"[A-Za-z']+", joined.lower())
    word_count = max(1, len(words))

    # sentence lengths (word count)
    sentences = [s.strip() for s in re.split(r"[.!?]+", joined) if s.strip()]
    sent_lens = [len(re.findall(r"[A-Za-z']+", s)) for s in sentences if s]
    sent_lens_sorted = sorted(sent_lens) if sent_lens else []

    def pct(p: float) -> int:
        if not sent_lens_sorted:
            return 0
        idx = int(round((len(sent_lens_sorted) - 1) * p))
        return int(sent_lens_sorted[max(0, min(len(sent_lens_sorted) - 1, idx))])

    sentence_len_stats = {
        "count": len(sent_lens_sorted),
        "mean": (sum(sent_lens_sorted) / len(sent_lens_sorted)) if sent_lens_sorted else 0.0,
        "p50": pct(0.50),
        "p90": pct(0.90),
    }

    # hedging
    hedge_counts = Counter(w for w in words if w in _HEDGES)
    hedging_profile = {
        "counts": dict(hedge_counts),
        "per_1k_words": {k: (v / word_count) * 1000.0 for k, v in hedge_counts.items()},
        "total_per_1k_words": (sum(hedge_counts.values()) / word_count) * 1000.0,
    }

    # connectors
    connector_counts = Counter()
    lower = joined.lower()
    for c in _CONNECTORS:
        connector_counts[c] = len(re.findall(rf"\b{re.escape(c)}\b", lower))
    connector_profile = {
        "counts": dict(connector_counts),
        "top": [k for k, _v in connector_counts.most_common(5) if _v > 0],
    }

    # voice
    we = len(re.findall(r"\bwe\b", lower))
    this_paper = len(re.findall(r"\bthis paper\b", lower))
    voice_profile = {
        "we_per_1k_words": (we / word_count) * 1000.0,
        "this_paper_per_1k_words": (this_paper / word_count) * 1000.0,
    }

    # citations
    par_cites = []
    for b in intro_blocks:
        t = (b.clean_text or b.text or "").strip()
        cites = len(re.findall(r"\[[0-9, -]+\]", t)) + len(re.findall(r"\([^)]+\)", t))
        par_cites.append(cites)
    citation_density = {
        "mean_per_paragraph": (sum(par_cites) / len(par_cites)) if par_cites else 0.0,
        "per_paragraph": par_cites,
    }

    return StyleFeatures(
        sentence_len_stats=sentence_len_stats,
        hedging_profile=hedging_profile,
        connector_profile=connector_profile,
        voice_profile=voice_profile,
        citation_density=citation_density,
    )


def extract_style_features_cached(*, pdf_id: str, intro_blocks: list[TextBlock], data_dir: Path) -> tuple[StyleFeatures, bool]:
    fingerprint = "\n".join(f"{b.block_id}\t{(b.clean_text or b.text or '').strip()}" for b in intro_blocks)
    input_hash = sha256_text(fingerprint)
    path = stage_cache_path(data_dir=data_dir, stage="style_features", key=pdf_id)
    payload = load_stage_payload(path=path, version=STYLE_CACHE_VERSION, input_hash=input_hash)
    if isinstance(payload, dict):
        try:
            return StyleFeatures.model_validate(payload), True
        except Exception:  # noqa: BLE001
            pass

    features = extract_style_features(intro_blocks)
    save_stage_payload(path=path, version=STYLE_CACHE_VERSION, input_hash=input_hash, payload=features.model_dump(mode="json"))
    return features, False
