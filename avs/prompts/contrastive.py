from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnchorWindow:
    start_s: float
    end_s: float


def _merge_windows(windows: list[AnchorWindow]) -> list[AnchorWindow]:
    if not windows:
        return []
    ws = sorted(windows, key=lambda w: (w.start_s, w.end_s))
    out: list[AnchorWindow] = [ws[0]]
    for w in ws[1:]:
        last = out[-1]
        if w.start_s <= last.end_s:
            out[-1] = AnchorWindow(start_s=last.start_s, end_s=max(last.end_s, w.end_s))
        else:
            out.append(w)
    return out


def anchor_windows_seconds(anchors_s: list[float], *, delta_s: float = 0.0) -> list[AnchorWindow]:
    windows = [AnchorWindow(start_s=float(a) - float(delta_s), end_s=float(a) + float(delta_s)) for a in anchors_s]
    windows = [AnchorWindow(start_s=max(0.0, w.start_s), end_s=max(0.0, w.end_s)) for w in windows]
    return _merge_windows(windows)


def format_multi_choice(choices: list[str]) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lines: list[str] = []
    for i, c in enumerate(choices):
        prefix = letters[i] if i < len(letters) else f"({i})"
        lines.append(f"{prefix}. {c}")
    return "\n".join(lines)


def build_contrastive_prompt(
    *,
    question_text: str,
    choices: list[str],
    anchors_s: list[float],
    delta_s: float = 0.0,
) -> str:
    windows = anchor_windows_seconds(anchors_s, delta_s=delta_s)
    if windows:
        window_str = ", ".join([f"[{w.start_s:.2f}s, {w.end_s:.2f}s]" for w in windows])
    else:
        window_str = "(none)"

    choices_block = format_multi_choice(choices)
    return "\n".join(
        [
            "You are an audio-visual QA assistant.",
            "Important: Only use visual evidence within the given audio-anchor time windows. Ignore the rest of the video.",
            f"Audio-anchor windows: {window_str}",
            "",
            f"Question: {question_text}",
            "Choices:",
            choices_block,
            "",
            "Answer with the single best choice letter, then 1-2 sentences of justification that references the window.",
        ]
    )

