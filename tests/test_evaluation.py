from __future__ import annotations

import json
from pathlib import Path

from bedtime_story.agent import StoryAgent
from evals.run_evaluation import (
    MeasuredModel,
    aggregate_results,
    load_case_set,
    render_markdown,
    run_benchmark,
    run_case,
)
from tests.support import ScriptedModel, judge_json


def _story(*details: str, marker: str = "calm") -> str:
    words = ["#", "A", "Quiet", "Story", *details]
    words.extend([marker] * max(0, 270 - len(words)))
    return " ".join(words)


def _case(*, feedback: dict[str, object] | None = None) -> dict[str, object]:
    case: dict[str, object] = {
        "id": "test_case",
        "description": "A deterministic unit-test case",
        "age": 7,
        "length": "short",
        "tags": ["listener_feedback"] if feedback else [],
        "request": "A calm story about Pip.",
        "checks": {"contains_all": ["Pip"]},
    }
    if feedback:
        case["feedback"] = feedback
    return case


def test_first_pass_metrics_count_exact_roles_without_storing_story() -> None:
    secret_marker = "STORY_TEXT_MUST_NOT_BE_PERSISTED"
    story = _story("Pip", marker=secret_marker)
    scripted = ScriptedModel([story, judge_json()])
    measured = MeasuredModel(scripted)
    agent = StoryAgent(measured, max_revisions=0)

    record = run_case(_case(), agent, measured)

    assert (
        record["initial"]["first_scorecard"]
        == record["initial"]["final_selected_scorecard"]
    )
    assert record["initial"]["editor_calls"] == 0
    assert record["initial"]["quality_comparison"]["classification"] == "unchanged"
    assert record["logical_model_calls"]["total"] == 2
    assert record["logical_model_calls"]["by_role"] == {
        "storyteller": 1,
        "judge": 1,
        "editor": 0,
    }
    assert secret_marker not in json.dumps(record)
    scripted.assert_finished()


def test_automatic_revision_improvement_and_rescue_denominators() -> None:
    scripted = ScriptedModel(
        [
            _story("Pip", marker="draft"),
            judge_json(score_overrides={"story_arc": 3}),
            _story("Pip", marker="revised"),
            judge_json(),
        ]
    )
    measured = MeasuredModel(scripted)
    agent = StoryAgent(measured, max_revisions=1)

    record = run_case(_case(), agent, measured)
    summary = aggregate_results([record])

    assert record["initial"]["editor_calls"] == 1
    assert record["initial"]["quality_comparison"]["classification"] == "improved"
    assert record["logical_model_calls"]["total"] == 4
    assert summary["automatic_revision"]["strict_judge_measured_improvement_rate"] == {
        "numerator": 1,
        "denominator": 1,
        "rate": 1.0,
    }
    assert summary["automatic_revision"]["best_valid_revision_improvement_rate"] == {
        "numerator": 1,
        "denominator": 1,
        "rate": 1.0,
    }
    assert summary["automatic_revision"]["rescue_rate"] == {
        "numerator": 1,
        "denominator": 1,
        "rate": 1.0,
    }
    scripted.assert_finished()


def test_feedback_uses_comparable_baseline_and_literal_checks() -> None:
    feedback = {
        "text": "Keep Pip and include the exact phrase tuck, tuck, snug.",
        "checks": {"contains_all": ["Pip", "tuck, tuck, snug"]},
    }
    scripted = ScriptedModel(
        [
            _story("Pip", marker="initial"),
            judge_json(),
            judge_json(score_overrides={"request_adherence": 3}),
            _story("Pip", "tuck,", "tuck,", "snug", marker="feedback"),
            judge_json(score_overrides={"request_adherence": 5}),
        ]
    )
    measured = MeasuredModel(scripted)
    agent = StoryAgent(measured, max_revisions=0)

    record = run_case(_case(feedback=feedback), agent, measured)
    summary = aggregate_results([record])

    feedback_phase = record["feedback"]
    assert feedback_phase is not None
    assert feedback_phase["first_scorecard"]["scores"]["request_adherence"] == 3
    assert (
        feedback_phase["final_selected_scorecard"]["scores"]["request_adherence"] == 5
    )
    assert feedback_phase["combined_brief_feedback_adherence_demonstrated"]
    assert feedback_phase["logical_model_calls"]["total"] == 3
    assert summary["feedback"]["combined_brief_feedback_adherence_rate"] == {
        "numerator": 1,
        "denominator": 1,
        "rate": 1.0,
    }
    scripted.assert_finished()


def test_unsafe_generated_output_is_accounted_as_unscored_fallback() -> None:
    scripted = ScriptedModel(
        [
            _story("Pip", marker="unsafe"),
            judge_json(
                score_overrides={"emotional_safety": 1},
                critical_safety_issues=["Graphic violence."],
            ),
        ]
    )
    measured = MeasuredModel(scripted)
    agent = StoryAgent(measured, max_revisions=0)

    record = run_case(_case(), agent, measured)
    summary = aggregate_results([record])

    assert record["overall"]["scorecard"] is None
    assert record["overall"]["built_in_fallback"]
    assert record["overall"]["output_safe_by_policy"]
    assert not record["overall"]["generated_output_judge_verified_safe"]
    assert record["overall"]["any_candidate_critical_safety_issue"]
    assert summary["safety"]["built_in_fallback_rate"]["numerator"] == 1
    assert (
        summary["safety"]["generated_output_judge_verified_safe_rate"]["numerator"] == 0
    )
    scripted.assert_finished()


def test_benchmark_report_is_score_only_and_handles_no_revision_denominator() -> None:
    secret_marker = "PRIVATE_GENERATED_STORY_MARKER"
    scripted = ScriptedModel([_story("Pip", marker=secret_marker), judge_json()])
    measured = MeasuredModel(scripted)
    agent = StoryAgent(measured, max_revisions=0)

    payload = run_benchmark(
        [_case()],
        agent,
        measured,
        case_set_version="test",
    )
    report = render_markdown(payload)

    assert (
        payload["summary"]["automatic_revision"][
            "strict_judge_measured_improvement_rate"
        ]["rate"]
        is None
    )
    assert "N/A (0 eligible)" in report
    assert secret_marker not in json.dumps(payload)
    assert secret_marker not in report
    scripted.assert_finished()


def test_versioned_live_case_set_covers_every_age_and_challenge() -> None:
    path = Path(__file__).resolve().parents[1] / "evals" / "cases.json"
    version, cases = load_case_set(path)

    assert version == "1.2"
    assert {case["age"] for case in cases} == set(range(5, 11))
    assert sum("listener_feedback" in case["tags"] for case in cases) == 2
    assert any("unsafe_request_adaptation" in case["tags"] for case in cases)
    assert any("prompt_injection" in case["tags"] for case in cases)
