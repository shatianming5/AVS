from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LlmRunStats:
    repair_used: dict[str, bool] = field(default_factory=dict)
    total_items: dict[str, int] = field(default_factory=dict)
    missing_items: dict[str, int] = field(default_factory=dict)


_STATS = LlmRunStats()


def reset_llm_stats() -> None:
    global _STATS
    _STATS = LlmRunStats()


def mark_repair_used(stage: str) -> None:
    _STATS.repair_used[str(stage)] = True


def add_total(stage: str, n: int) -> None:
    k = str(stage)
    _STATS.total_items[k] = int(_STATS.total_items.get(k, 0)) + int(n)


def add_missing(stage: str, n: int) -> None:
    k = str(stage)
    _STATS.missing_items[k] = int(_STATS.missing_items.get(k, 0)) + int(n)


def snapshot_llm_stats() -> dict:
    return {
        "repair_used": dict(_STATS.repair_used),
        "total_items": dict(_STATS.total_items),
        "missing_items": dict(_STATS.missing_items),
    }

