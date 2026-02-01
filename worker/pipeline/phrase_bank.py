from __future__ import annotations

from collections import Counter

from shared.schemas import StyleFeatures


def build_phrase_bank(style_by_pdf: dict[str, StyleFeatures]) -> dict:
    connectors = Counter()
    hedges = Counter()

    for s in style_by_pdf.values():
        c_counts = (s.connector_profile or {}).get("counts") or {}
        for k, v in c_counts.items():
            connectors[str(k)] += int(v)
        h_counts = (s.hedging_profile or {}).get("counts") or {}
        for k, v in h_counts.items():
            hedges[str(k)] += int(v)

    connector_top = [k for k, v in connectors.most_common(10) if v > 0]
    hedge_top = [k for k, v in hedges.most_common(10) if v > 0]

    return {
        "connectors": connector_top,
        "hedges": hedge_top,
    }

