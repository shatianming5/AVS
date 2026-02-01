from __future__ import annotations

from dataclasses import dataclass


def _lcs(a: list[str], b: list[str]) -> list[str]:
    dp: list[list[int]] = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) - 1, -1, -1):
        for j in range(len(b) - 1, -1, -1):
            if a[i] == b[j]:
                dp[i][j] = 1 + dp[i + 1][j + 1]
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    i = j = 0
    out: list[str] = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return out


def consensus_sequence(seqs: list[list[str]]) -> list[str]:
    if not seqs:
        return []
    out = seqs[0]
    for s in seqs[1:]:
        out = _lcs(out, s)
    return out


def align_move_sequences(seqs_by_pdf: dict[str, list[str]]) -> dict:
    seqs = [s for s in seqs_by_pdf.values() if s]
    cons = consensus_sequence(seqs)
    if not cons:
        cons = ["Context", "Problem", "Gap", "Approach", "Contribution", "Roadmap"]

    # strength: average LCS ratio
    ratios: list[float] = []
    for s in seqs:
        l = len(_lcs(cons, s))
        denom = max(1, max(len(cons), len(s)))
        ratios.append(l / denom)
    strength = sum(ratios) / len(ratios) if ratios else 0.0

    # per-pdf greedy alignment indices
    per_pdf: dict[str, list[dict]] = {}
    for pdf_id, s in seqs_by_pdf.items():
        j = 0
        mapping: list[dict] = []
        for lab in cons:
            found = None
            for k in range(j, len(s)):
                if s[k] == lab:
                    found = k
                    j = k + 1
                    break
            mapping.append({"label": lab, "matched": found is not None, "span_index": found})
        per_pdf[pdf_id] = mapping

    # optional labels: labels appearing in sequences but not in consensus
    extras: dict[str, int] = {}
    cons_set = set(cons)
    for s in seqs:
        for lab in s:
            if lab not in cons_set:
                extras[lab] = extras.get(lab, 0) + 1
    optional = [{"label": k, "count": v} for k, v in sorted(extras.items(), key=lambda kv: (-kv[1], kv[0]))]

    return {
        "consensus_sequence": cons,
        "strength_score": float(max(0.0, min(1.0, strength))),
        "optional_labels": optional,
        "per_pdf_alignment": per_pdf,
    }

