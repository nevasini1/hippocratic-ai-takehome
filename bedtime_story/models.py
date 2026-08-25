"""Validated data models for the bedtime-story pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Any, Mapping


MAX_REQUEST_CHARS = 1_000
MAX_FEEDBACK_CHARS = 500

LENGTH_WORD_RANGES: dict[str, tuple[int, int]] = {
    "short": (250, 400),
    "medium": (450, 700),
    "long": (700, 1_000),
}

QUALITY_DIMENSIONS = (
    "request_adherence",
    "age_appropriateness",
    "story_arc",
    "engagement",
    "language_clarity",
    "bedtime_tone",
    "emotional_safety",
)


class JudgeFormatError(ValueError):
    """Raised when the judge response does not match the required schema."""


class InputValidationError(ValueError):
    """Raised for invalid listener-controlled CLI data."""


@dataclass(frozen=True)
class StorySpec:
    """A normalized request supplied to every role in the pipeline."""

    request: str
    age: int = 7
    length: str = "medium"

    def __post_init__(self) -> None:
        if not isinstance(self.request, str):
            raise InputValidationError("The story request must be text.")
        if not isinstance(self.length, str):
            raise InputValidationError("Length must be text.")
        request = self.request.strip()
        length = self.length.strip().lower()

        if not request:
            raise InputValidationError("The story request cannot be empty.")
        if len(request) > MAX_REQUEST_CHARS:
            raise InputValidationError(
                f"The story request must be {MAX_REQUEST_CHARS} characters or fewer."
            )
        if isinstance(self.age, bool) or not isinstance(self.age, int):
            raise InputValidationError("Age must be a whole number from 5 through 10.")
        if not 5 <= self.age <= 10:
            raise InputValidationError("Age must be from 5 through 10.")
        if length not in LENGTH_WORD_RANGES:
            choices = ", ".join(LENGTH_WORD_RANGES)
            raise InputValidationError(f"Length must be one of: {choices}.")

        object.__setattr__(self, "request", request)
        object.__setattr__(self, "length", length)

    @property
    def word_range(self) -> tuple[int, int]:
        return LENGTH_WORD_RANGES[self.length]

    def as_prompt_data(self) -> dict[str, Any]:
        minimum, maximum = self.word_range
        return {
            "story_request": self.request,
            "target_age": self.age,
            "length": self.length,
            "target_word_range": {"minimum": minimum, "maximum": maximum},
        }


@dataclass(frozen=True)
class JudgeReport:
    """A conservatively validated scorecard returned by the LLM judge."""

    scores: Mapping[str, int]
    critical_safety_issues: tuple[str, ...]
    prompt_injection_leak: bool
    strengths: tuple[str, ...]
    required_revisions: tuple[str, ...]
    summary: str

    @classmethod
    def from_mapping(cls, value: object) -> "JudgeReport":
        if not isinstance(value, Mapping):
            raise JudgeFormatError("Judge output must be a JSON object.")

        expected_fields = {
            "scores",
            "critical_safety_issues",
            "prompt_injection_leak",
            "strengths",
            "required_revisions",
            "summary",
        }
        if set(value) != expected_fields:
            raise JudgeFormatError("Judge output has missing or unexpected fields.")

        raw_scores = value.get("scores")
        if not isinstance(raw_scores, Mapping):
            raise JudgeFormatError("Judge output must contain a scores object.")
        if set(raw_scores) != set(QUALITY_DIMENSIONS):
            raise JudgeFormatError(
                "Judge scores have missing or unexpected dimensions."
            )

        scores: dict[str, int] = {}
        for dimension in QUALITY_DIMENSIONS:
            score = raw_scores.get(dimension)
            # bool is an int subclass in Python, so reject it explicitly.
            if isinstance(score, bool) or not isinstance(score, int):
                raise JudgeFormatError(
                    f"Score for {dimension!r} must be an integer from 1 to 5."
                )
            if not 1 <= score <= 5:
                raise JudgeFormatError(
                    f"Score for {dimension!r} must be an integer from 1 to 5."
                )
            scores[dimension] = score

        critical_safety_issues = _string_tuple(
            value.get("critical_safety_issues"), "critical_safety_issues"
        )
        strengths = _string_tuple(value.get("strengths"), "strengths")
        required_revisions = _string_tuple(
            value.get("required_revisions"), "required_revisions"
        )

        prompt_injection_leak = value.get("prompt_injection_leak")
        if not isinstance(prompt_injection_leak, bool):
            raise JudgeFormatError("prompt_injection_leak must be true or false.")

        summary = value.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            raise JudgeFormatError("summary must be a non-empty string.")

        needs_revision = (
            any(score < 4 for score in scores.values())
            or bool(critical_safety_issues)
            or prompt_injection_leak
        )
        if needs_revision and not required_revisions:
            raise JudgeFormatError(
                "A failing judge report must include at least one required revision."
            )

        return cls(
            scores=scores,
            critical_safety_issues=critical_safety_issues,
            prompt_injection_leak=prompt_injection_leak,
            strengths=strengths,
            required_revisions=required_revisions,
            summary=summary.strip(),
        )

    @property
    def average_score(self) -> float:
        return fmean(self.scores.values())

    @property
    def minimum_score(self) -> int:
        return min(self.scores.values())

    @property
    def is_safe(self) -> bool:
        """Allow a sub-threshold story into best-safe selection, never approval."""

        return (
            not self.critical_safety_issues
            and not self.prompt_injection_leak
            and self.scores["age_appropriateness"] >= 4
            and self.scores["emotional_safety"] >= 4
        )

    @property
    def passes(self) -> bool:
        """Compute approval locally rather than trusting a model verdict."""

        return self.is_safe and all(score >= 4 for score in self.scores.values())


@dataclass(frozen=True)
class StoryCandidate:
    story: str
    report: JudgeReport
    revision_number: int


@dataclass(frozen=True)
class StoryResult:
    spec: StorySpec
    story: str
    report: JudgeReport | None
    accepted: bool
    used_fallback: bool
    revision_count: int
    feedback_rounds: int
    candidates: tuple[StoryCandidate, ...]
    quality_note: str = ""


def validate_feedback(feedback: str) -> str:
    if not isinstance(feedback, str):
        raise InputValidationError("Feedback must be text.")
    normalized = feedback.strip()
    if not normalized:
        raise InputValidationError("Feedback cannot be empty.")
    if len(normalized) > MAX_FEEDBACK_CHARS:
        raise InputValidationError(
            f"Feedback must be {MAX_FEEDBACK_CHARS} characters or fewer."
        )
    return normalized


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise JudgeFormatError(f"{field_name} must be a JSON array of strings.")

    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise JudgeFormatError(
                f"Every item in {field_name} must be a non-empty string."
            )
        result.append(item.strip())
    return tuple(result)
