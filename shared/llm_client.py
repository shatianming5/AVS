from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time
from urllib.parse import urlparse

import httpx


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    v = raw.strip().lower()
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off", ""}:
        return False
    return default


def llm_required() -> bool:
    return _env_bool("PAPER_SKILL_REQUIRE_LLM", True)


def llm_json_mode_enabled() -> bool:
    return _env_bool("PAPER_SKILL_LLM_JSON_MODE", True)


def redact_base_url(base_url: str) -> str:
    p = urlparse(base_url)
    if not p.scheme or not p.netloc:
        return base_url
    return f"{p.scheme}://{p.netloc}"


@dataclass(frozen=True)
class LlmConfig:
    base_url: str
    model: str
    timeout_s: float
    max_retries: int
    temperature: float
    api_key: str | None = None


_DISCOVERED_MODEL: dict[str, str] = {}


def _discover_chat_model(*, base_url: str, api_key: str | None, timeout_s: float) -> str | None:
    b = base_url.rstrip("/")
    key = f"{b}|{api_key or ''}"
    if key in _DISCOVERED_MODEL:
        return _DISCOVERED_MODEL[key]

    url = _join_url(b, "/models")
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        timeout = min(5.0, max(2.0, float(timeout_s)))
        with httpx.Client(timeout=timeout) as client:
            res = client.get(url, headers=headers)
        if res.status_code >= 400:
            return None
        data = res.json()
    except Exception:  # noqa: BLE001
        return None

    items = data.get("data") if isinstance(data, dict) else None
    if not isinstance(items, list) or not items:
        return None

    ids: list[str] = []
    for it in items:
        if isinstance(it, dict) and isinstance(it.get("id"), str):
            ids.append(it["id"])
    if not ids:
        return None

    def is_chat_like(mid: str) -> bool:
        m = mid.lower()
        if any(bad in m for bad in ["embedding", "whisper", "tts", "dall-e", "image", "moderation"]):
            return False
        return True

    chat_candidates = [m for m in ids if is_chat_like(m)]
    if not chat_candidates:
        return None

    preferred = [m for m in chat_candidates if any(k in m.lower() for k in ["gpt", "instruct", "chat"])]
    chosen = preferred[0] if preferred else chat_candidates[0]
    _DISCOVERED_MODEL[key] = chosen
    return chosen


def load_llm_config() -> LlmConfig | None:
    base_url = (
        os.environ.get("PAPER_SKILL_LLM_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
        or "http://127.0.0.1:8001/v1"
    ).strip()
    timeout_s = float((os.environ.get("PAPER_SKILL_LLM_TIMEOUT_S") or "60").strip())
    max_retries = int((os.environ.get("PAPER_SKILL_LLM_MAX_RETRIES") or "2").strip())
    temperature = float((os.environ.get("PAPER_SKILL_LLM_TEMPERATURE") or "0.2").strip())
    api_key = (os.environ.get("PAPER_SKILL_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip() or None

    model = (
        os.environ.get("PAPER_SKILL_LLM_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or os.environ.get("OPENAI_API_MODEL")
        or os.environ.get("CHAT_MODEL")
        or ""
    ).strip()
    if not model:
        discovered = _discover_chat_model(base_url=base_url, api_key=api_key, timeout_s=timeout_s)
        if discovered:
            model = discovered
        else:
            return None

    return LlmConfig(
        base_url=base_url.rstrip("/"),
        model=model,
        timeout_s=timeout_s,
        max_retries=max(0, max_retries),
        temperature=temperature,
        api_key=api_key,
    )


def require_llm_config() -> LlmConfig:
    cfg = load_llm_config()
    if cfg is None:
        raise RuntimeError("PAPER_SKILL_LLM_MODEL is required (PAPER_SKILL_REQUIRE_LLM=1).")
    return cfg


def _join_url(base_url: str, path: str) -> str:
    b = base_url.rstrip("/")
    p = (path or "").lstrip("/")
    return f"{b}/{p}"


def _extract_json(text: str) -> object:
    s = (text or "").strip()
    if not s:
        raise ValueError("Empty LLM response text.")
    try:
        return json.loads(s)
    except Exception:  # noqa: BLE001
        pass

    # Best-effort extraction: first {...} or [...]
    obj_start = s.find("{")
    obj_end = s.rfind("}")
    if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
        try:
            return json.loads(s[obj_start : obj_end + 1])
        except Exception:  # noqa: BLE001
            pass

    arr_start = s.find("[")
    arr_end = s.rfind("]")
    if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
        try:
            return json.loads(s[arr_start : arr_end + 1])
        except Exception:  # noqa: BLE001
            pass

    raise ValueError(f"Could not extract JSON from response (head): {s[:240]!r}")


def _assistant_content(data: dict) -> str:
    choices = data.get("choices") or []
    if isinstance(choices, list) and choices:
        c0 = choices[0]
        if isinstance(c0, dict):
            msg = c0.get("message")
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return msg["content"]
            if isinstance(c0.get("text"), str):
                return c0["text"]
    if isinstance(data.get("content"), str):
        return data["content"]
    raise ValueError("Missing assistant content in LLM response.")


class LlmHttpError(RuntimeError):
    def __init__(self, status_code: int, body: str):
        super().__init__(f"LLM HTTP {status_code}: {body[:400]}")
        self.status_code = int(status_code)
        self.body = body


def chat_completions_json(*, cfg: LlmConfig, messages: list[dict], schema_name: str) -> dict:
    url = _join_url(cfg.base_url, "/chat/completions")
    base_payload = {
        "model": cfg.model,
        "messages": messages,
        "temperature": cfg.temperature,
    }
    payloads = [base_payload]
    if llm_json_mode_enabled():
        payloads = [
            {**base_payload, "response_format": {"type": "json_object"}},
            base_payload,
        ]
    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"

    last_err: Exception | None = None
    for payload_i, payload in enumerate(payloads):
        for attempt in range(cfg.max_retries + 1):
            try:
                with httpx.Client(timeout=cfg.timeout_s) as client:
                    res = client.post(url, json=payload, headers=headers)
                if res.status_code >= 400:
                    raise LlmHttpError(res.status_code, res.text)
                data = res.json()
                content = _assistant_content(data)
                parsed = _extract_json(content)
                if not isinstance(parsed, dict):
                    raise ValueError(f"LLM returned non-object JSON for {schema_name}: {type(parsed).__name__}")
                return parsed
            except LlmHttpError as e:
                last_err = e
                # Fallback for gateways that don't support `response_format`.
                if payload_i == 0 and len(payloads) > 1 and e.status_code in {400, 422}:
                    break
                if attempt >= cfg.max_retries:
                    break
                time.sleep(min(1.0, 0.2 * (2**attempt)))
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt >= cfg.max_retries:
                    break
                time.sleep(min(1.0, 0.2 * (2**attempt)))

        if payload_i == len(payloads) - 1:
            break

    raise RuntimeError(f"LLM call failed for schema={schema_name}: {last_err}") from last_err
