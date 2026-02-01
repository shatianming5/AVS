from __future__ import annotations


def test_extract_json_object_from_plain_text() -> None:
    from shared.llm_client import _extract_json  # noqa: WPS437

    assert _extract_json('{"a": 1, "b": "x"}') == {"a": 1, "b": "x"}


def test_extract_json_object_from_wrapped_text() -> None:
    from shared.llm_client import _extract_json  # noqa: WPS437

    txt = "Sure! Here you go:\n```json\n{\n  \"ok\": true,\n  \"n\": 2\n}\n```\nThanks."
    assert _extract_json(txt) == {"ok": True, "n": 2}

