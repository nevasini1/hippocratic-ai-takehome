from __future__ import annotations

import json

import pytest

from bedtime_story.json_tools import extract_json_object, parse_judge_report
from bedtime_story.models import (
    MAX_FEEDBACK_CHARS,
    MAX_REQUEST_CHARS,
    JudgeFormatError,
    StorySpec,
    validate_feedback,
)
from tests.support import judge_json


def test_story_spec_normalizes_input_and_preserves_unicode() -> None:
    spec = StorySpec("  A café dragon named José  ", age=8, length="SHORT")

    assert spec.request == "A café dragon named José"
    assert spec.length == "short"
    assert spec.word_range == (250, 400)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"request": "   "}, "cannot be empty"),
        ({"request": "x" * (MAX_REQUEST_CHARS + 1)}, "characters or fewer"),
        ({"request": "story", "age": 4}, "from 5 through 10"),
        ({"request": "story", "age": True}, "whole number"),
        ({"request": "story", "length": "epic"}, "Length must be one of"),
        ({"request": None}, "must be text"),
    ],
)
def test_story_spec_rejects_invalid_input(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        StorySpec(**kwargs)  # type: ignore[arg-type]


def test_judge_parser_accepts_plain_fenced_and_prefaced_objects() -> None:
    raw = judge_json()

    assert parse_judge_report(raw).passes
    assert parse_judge_report(f"```json\n{raw}\n```").passes
    assert parse_judge_report(f"Here is the scorecard:\n{raw}").passes


def test_json_extractor_does_not_greedily_combine_objects() -> None:
    assert extract_json_object('note {"first": 1} and {"second": 2}') == {"first": 1}


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "not json",
        '[{"scores": {}}]',
        '"a string"',
        '{"scores": ',
    ],
)
def test_judge_parser_rejects_non_object_or_malformed_output(raw: str) -> None:
    with pytest.raises(JudgeFormatError):
        parse_judge_report(raw)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value["scores"].pop("engagement"),
        lambda value: value["scores"].update({"engagement": True}),
        lambda value: value["scores"].update({"engagement": 6}),
        lambda value: value["scores"].update({"engagement": float("nan")}),
        lambda value: value["scores"].update({"engagement": float("inf")}),
        lambda value: value.update({"prompt_injection_leak": "false"}),
        lambda value: value.update({"strengths": [""]}),
        lambda value: value.update({"summary": ""}),
        lambda value: value.update({"unexpected": "field"}),
        lambda value: value["scores"].update({"unexpected": 4}),
    ],
)
def test_judge_parser_rejects_schema_violations(mutator) -> None:
    value = json.loads(judge_json())
    mutator(value)

    with pytest.raises(JudgeFormatError):
        parse_judge_report(json.dumps(value))


def test_pass_is_computed_locally_from_scores_and_safety() -> None:
    low_score = parse_judge_report(judge_json(score_overrides={"engagement": 3}))
    unsafe = parse_judge_report(judge_json(critical_safety_issues=["Graphic injury"]))
    not_verified_safe = parse_judge_report(
        judge_json(score_overrides={"emotional_safety": 3})
    )

    assert low_score.is_safe
    assert not low_score.passes
    assert not unsafe.is_safe
    assert not unsafe.passes
    assert not not_verified_safe.is_safe
    assert not not_verified_safe.passes


def test_failing_report_requires_an_actionable_revision() -> None:
    raw = judge_json(
        score_overrides={"story_arc": 3},
        required_revisions=[],
    )

    with pytest.raises(JudgeFormatError, match="required revision"):
        parse_judge_report(raw)


def test_feedback_validation_is_bounded() -> None:
    assert validate_feedback("  make the cat funnier  ") == "make the cat funnier"
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_feedback(" ")
    with pytest.raises(ValueError, match="characters or fewer"):
        validate_feedback("x" * (MAX_FEEDBACK_CHARS + 1))
