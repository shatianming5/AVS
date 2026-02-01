from __future__ import annotations


def _ratio(numer: int, denom: int) -> float:
    if denom <= 0:
        return 0.0
    return float(numer) / float(denom)


def llm_health_metrics(*, llm_stats: dict) -> dict:
    total = llm_stats.get("total_items") or {}
    missing = llm_stats.get("missing_items") or {}
    repair = llm_stats.get("repair_used") or {}

    moves_total = int(total.get("moves_blocks") or 0)
    moves_missing = int(missing.get("moves_blocks") or 0)

    bp_rules_total = int(total.get("blueprint_rules") or 0)
    bp_rules_missing = int(missing.get("blueprint_rules") or 0)

    sb_items_total = int(total.get("storyboard_items") or 0)
    sb_items_missing = int(missing.get("storyboard_items") or 0)

    return {
        "moves_return_coverage": _ratio(moves_total - moves_missing, moves_total),
        "blueprint_rules_with_evidence_ratio": _ratio(bp_rules_total - bp_rules_missing, bp_rules_total),
        "storyboard_items_with_evidence_ratio": _ratio(sb_items_total - sb_items_missing, sb_items_total),
        "invalid_supporting_block_ids": {
            "moves_missing_blocks": moves_missing,
            "blueprint_rules_missing_blocks": int(missing.get("blueprint_missing_block_ids") or 0),
            "storyboard_missing_block_ids": int(missing.get("storyboard_missing_block_ids") or 0),
        },
        "repair_used": {
            "moves": bool(repair.get("moves", False)),
            "blueprint": bool(repair.get("blueprint", False)),
            "storyboard": bool(repair.get("storyboard", False)),
        },
    }

