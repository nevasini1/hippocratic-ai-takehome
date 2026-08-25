from __future__ import annotations

import json
from collections import deque
from typing import Any

from bedtime_story.models import QUALITY_DIMENSIONS


class ScriptedModel:
    """Network-free fake that records calls and returns queued responses."""

    def __init__(self, responses: list[str | Exception]) -> None:
        self.responses = deque(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        json_mode: bool = False,
    ) -> str:
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "json_mode": json_mode,
            }
        )
        if not self.responses:
            raise AssertionError("The fake model received an unexpected call.")
        response = self.responses.popleft()
        if isinstance(response, Exception):
            raise response
        return response

    def assert_finished(self) -> None:
        assert not self.responses, "Not all scripted model responses were consumed."


def judge_json(
    default_score: int = 4,
    *,
    score_overrides: dict[str, object] | None = None,
    critical_safety_issues: list[str] | None = None,
    prompt_injection_leak: object = False,
    strengths: list[str] | None = None,
    required_revisions: list[str] | None = None,
    summary: str = "A concise assessment.",
) -> str:
    scores: dict[str, object] = {
        dimension: default_score for dimension in QUALITY_DIMENSIONS
    }
    scores.update(score_overrides or {})
    if required_revisions is None:
        needs_revision = (
            any(
                isinstance(score, int) and not isinstance(score, bool) and score < 4
                for score in scores.values()
            )
            or bool(critical_safety_issues)
            or prompt_injection_leak is True
        )
        revisions = (
            ["Address every score below the quality threshold."]
            if needs_revision
            else []
        )
    else:
        revisions = required_revisions
    return json.dumps(
        {
            "scores": scores,
            "critical_safety_issues": critical_safety_issues or [],
            "prompt_injection_leak": prompt_injection_leak,
            "strengths": strengths or ["Warm characters"],
            "required_revisions": revisions,
            "summary": summary,
        }
    )
