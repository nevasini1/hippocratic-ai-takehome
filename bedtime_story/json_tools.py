"""Small, dependency-free helpers for structured LLM responses."""

from __future__ import annotations

import json
from typing import Any

from .models import JudgeFormatError, JudgeReport


def extract_json_object(text: str) -> dict[str, Any]:
    """Return the first complete JSON object without greedy brace matching."""

    if not isinstance(text, str) or not text.strip():
        raise JudgeFormatError("Judge returned an empty response.")

    candidate = text.strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3:
            candidate = "\n".join(lines[1:-1]).strip()

    # A JSON array or scalar is a schema violation. We still tolerate a short
    # prose preamble because older models occasionally add one around JSON.
    if candidate[:1] in {"[", '"'} or candidate in {"true", "false", "null"}:
        raise JudgeFormatError("Judge output must be a top-level JSON object.")

    decoder = json.JSONDecoder()
    starts = [0] if candidate.startswith("{") else []
    starts.extend(index for index, char in enumerate(candidate) if char == "{")

    tried: set[int] = set()
    for start in starts:
        if start in tried:
            continue
        tried.add(start)
        try:
            value, _ = decoder.raw_decode(candidate, start)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(value, dict):
            return value

    raise JudgeFormatError("Judge did not return a valid JSON object.")


def parse_judge_report(text: str) -> JudgeReport:
    return JudgeReport.from_mapping(extract_json_object(text))
