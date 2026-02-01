from __future__ import annotations

import re


def clean_text_preserve(raw_text: str) -> str:
    """
    Conservative cleaning:
    - join hyphenated line breaks ("exam-\nple" -> "example")
    - replace newlines with spaces
    - collapse whitespace
    """
    t = raw_text or ""
    # hyphen line breaks: letter + "-\n" + letter
    t = re.sub(r"([A-Za-z])-\n([A-Za-z])", r"\1\2", t)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = t.replace("\n", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t
