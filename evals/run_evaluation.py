"""Run a small live benchmark and write score-only JSON and Markdown artifacts.

The benchmark deliberately does not persist generated stories, prompts, or API
credentials. Its scores come from the same required model as the generator, so
the results are directional engineering evidence rather than human ground truth.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, median
from time import perf_counter
from typing import Any

from bedtime_story.agent import StoryAgent
from bedtime_story.llm import ChatModel, MODEL_NAME, OpenAIChatModel
from bedtime_story.models import QUALITY_DIMENSIONS, JudgeReport, StoryResult, StorySpec
from bedtime_story.prompts import (
    EDITOR_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    STORYTELLER_SYSTEM_PROMPT,
)


DEFAULT_CASES_PATH = Path(__file__).with_name("cases.json")
DEFAULT_OUTPUT_PATH = Path(__file__).with_name("results") / "latest.json"
DEFAULT_REPORT_PATH = Path(__file__).resolve().parents[1] / "EVALUATION.md"
_WORD_RE = re.compile(r"\b[\w'’-]+\b", re.UNICODE)
_PROMPT_ROLES = {
    STORYTELLER_SYSTEM_PROMPT: "storyteller",
    JUDGE_SYSTEM_PROMPT: "judge",
    EDITOR_SYSTEM_PROMPT: "editor",
}


class MeasuredModel:
    """Measure logical ChatModel calls without recording their content."""

    def __init__(
        self,
        inner: ChatModel,
        *,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self.inner = inner
        self.clock = clock
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        json_mode: bool = False,
    ) -> str:
        role = _classify_role(messages)
        started = self.clock()
        succeeded = False
        try:
            response = self.inner.complete(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
            )
            succeeded = True
            return response
        finally:
            self.calls.append(
                {
                    "index": len(self.calls) + 1,
                    "role": role,
                    "succeeded": succeeded,
                    "duration_seconds": _round(self.clock() - started, 6),
                }
            )


def _classify_role(messages: list[dict[str, str]]) -> str:
    if not messages or messages[0].get("role") != "system":
        raise ValueError("Evaluation received a model call without a system role.")
    prompt = messages[0].get("content")
    try:
        return _PROMPT_ROLES[prompt]
    except (KeyError, TypeError) as exc:
        raise ValueError("Evaluation received an unknown agent system prompt.") from exc


def load_case_set(path: Path = DEFAULT_CASES_PATH) -> tuple[str, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"version", "cases"}:
        raise ValueError("Case file must contain exactly version and cases.")
    version = payload["version"]
    cases = payload["cases"]
    if not isinstance(version, str) or not version.strip():
        raise ValueError("Case-set version must be non-empty text.")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Case file must contain at least one case.")

    identifiers: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Every evaluation case must be an object.")
        identifier = case.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("Every evaluation case needs a non-empty id.")
        if identifier in identifiers:
            raise ValueError(f"Duplicate evaluation case id: {identifier}")
        identifiers.add(identifier)
        StorySpec(
            request=case.get("request"),
            age=case.get("age"),
            length=case.get("length"),
        )
        _validate_checks(case.get("checks", {}))
        feedback = case.get("feedback")
        if feedback is not None:
            if not isinstance(feedback, dict) or not isinstance(
                feedback.get("text"), str
            ):
                raise ValueError(f"Case {identifier} has invalid feedback.")
            _validate_checks(feedback.get("checks", {}))
    return version, cases


def _validate_checks(checks: object) -> None:
    if not isinstance(checks, dict):
        raise ValueError("Case checks must be an object.")
    allowed = {"contains_all", "contains_any", "forbidden", "manual"}
    if not set(checks).issubset(allowed):
        raise ValueError("Case checks contain an unknown field.")
    for key in ("contains_all", "forbidden", "manual"):
        values = checks.get(key, [])
        if not isinstance(values, list) or not all(
            isinstance(item, str) and item.strip() for item in values
        ):
            raise ValueError(f"{key} must be an array of non-empty strings.")
    any_groups = checks.get("contains_any", [])
    if not isinstance(any_groups, list) or not all(
        isinstance(group, list)
        and group
        and all(isinstance(item, str) and item.strip() for item in group)
        for group in any_groups
    ):
        raise ValueError("contains_any must be an array of non-empty string arrays.")


def run_case(
    case: dict[str, Any],
    agent: StoryAgent,
    measured_model: MeasuredModel,
    *,
    clock: Callable[[], float] = perf_counter,
) -> dict[str, Any]:
    """Run initial generation and optional listener feedback for one case."""

    spec = StorySpec(case["request"], age=case["age"], length=case["length"])
    case_started = clock()
    call_start = len(measured_model.calls)

    initial_started = clock()
    initial_call_start = len(measured_model.calls)
    initial_result = agent.create_story(spec)
    initial_elapsed = clock() - initial_started
    initial_calls = measured_model.calls[initial_call_start:]
    initial = _phase_record(
        initial_result,
        initial_calls,
        wall_seconds=initial_elapsed,
        baseline_revision_number=0,
    )
    initial["output_checks"] = evaluate_output_checks(
        initial_result.story, spec, case.get("checks", {})
    )

    feedback_record: dict[str, Any] | None = None
    final_result = initial_result
    final_checks = initial["output_checks"]
    feedback = case.get("feedback")
    if feedback is not None:
        feedback_started = clock()
        feedback_call_start = len(measured_model.calls)
        final_result = agent.refine_story(initial_result, feedback["text"])
        feedback_elapsed = clock() - feedback_started
        feedback_calls = measured_model.calls[feedback_call_start:]
        feedback_record = _phase_record(
            final_result,
            feedback_calls,
            wall_seconds=feedback_elapsed,
            baseline_revision_number=0,
        )
        final_checks = evaluate_output_checks(
            final_result.story, spec, feedback.get("checks", {})
        )
        feedback_record["output_checks"] = final_checks
        selected_was_judged_for_feedback = bool(
            final_result.report is not None
            and any(
                candidate.report is final_result.report
                for candidate in final_result.candidates
            )
        )
        feedback_record["combined_brief_feedback_adherence_demonstrated"] = bool(
            selected_was_judged_for_feedback
            and not final_result.used_fallback
            and final_result.report is not None
            and final_result.report.is_safe
            and final_result.report.scores["request_adherence"] >= 4
            and final_checks["automated_checks_passed"]
        )
        feedback_record["selected_was_feedback_judged"] = (
            selected_was_judged_for_feedback
        )

    all_case_calls = measured_model.calls[call_start:]
    overall_report = _report_record(final_result.report)
    case_record = {
        "id": case["id"],
        "description": case.get("description", ""),
        "age": case["age"],
        "length": case["length"],
        "tags": list(case.get("tags", [])),
        "initial": initial,
        "feedback": feedback_record,
        "overall": {
            "outcome": _outcome(final_result),
            "scorecard": overall_report,
            "generated_output_judge_verified_safe": bool(
                overall_report and overall_report["is_safe"]
            ),
            "output_safe_by_policy": bool(
                final_result.used_fallback
                or (overall_report and overall_report["is_safe"])
            ),
            "built_in_fallback": final_result.used_fallback,
            "output_checks": final_checks,
        },
        "logical_model_calls": _call_metrics(all_case_calls),
        "wall_latency_seconds": _round(clock() - case_started, 6),
    }
    case_record["overall"]["any_candidate_critical_safety_issue"] = any(
        candidate["scorecard"]["critical_safety_issue_count"] > 0
        for phase in (initial, feedback_record)
        if phase is not None
        for candidate in phase["candidates"]
    )
    case_record["overall"]["any_candidate_prompt_injection_leak"] = any(
        candidate["scorecard"]["prompt_injection_leak"]
        for phase in (initial, feedback_record)
        if phase is not None
        for candidate in phase["candidates"]
    )
    return case_record


def _phase_record(
    result: StoryResult,
    calls: list[dict[str, Any]],
    *,
    wall_seconds: float,
    baseline_revision_number: int,
) -> dict[str, Any]:
    baseline = next(
        (
            candidate
            for candidate in result.candidates
            if candidate.revision_number == baseline_revision_number
        ),
        None,
    )
    first_report = _report_record(baseline.report if baseline else None)
    final_report = _report_record(result.report)
    role_counts = Counter(call["role"] for call in calls)
    comparison = _compare_scorecards(first_report, final_report)
    revised_candidates = [
        candidate
        for candidate in result.candidates
        if candidate.revision_number > baseline_revision_number
    ]
    best_revision = (
        max(
            revised_candidates,
            key=lambda candidate: _candidate_quality_rank(
                _report_record(candidate.report)
            ),
        )
        if revised_candidates
        else None
    )
    best_revision_report = _report_record(
        best_revision.report if best_revision is not None else None
    )
    valid_candidates = [
        {
            "revision_number": candidate.revision_number,
            "scorecard": _report_record(candidate.report),
        }
        for candidate in result.candidates
    ]
    return {
        "outcome": _outcome(result),
        "first_scorecard": first_report,
        "final_selected_scorecard": final_report,
        "quality_comparison": comparison,
        "best_valid_revision_number": (
            best_revision.revision_number if best_revision is not None else None
        ),
        "best_valid_revision_scorecard": best_revision_report,
        "best_valid_revision_comparison": _compare_candidate_scorecards(
            first_report, best_revision_report
        ),
        "selected_revision_number": result.revision_count,
        "valid_candidates_judged": len(result.candidates),
        "editor_calls": role_counts["editor"],
        "judge_format_retry_calls": max(
            role_counts["judge"] - len(result.candidates), 0
        ),
        "candidates": valid_candidates,
        "logical_model_calls": _call_metrics(calls),
        "wall_latency_seconds": _round(wall_seconds, 6),
    }


def _report_record(report: JudgeReport | None) -> dict[str, Any] | None:
    if report is None:
        return None
    return {
        "scores": dict(report.scores),
        "average_score": _round(report.average_score, 4),
        "minimum_score": report.minimum_score,
        "dimensions_at_or_above_4": sum(score >= 4 for score in report.scores.values()),
        "passes": report.passes,
        "is_safe": report.is_safe,
        "critical_safety_issue_count": len(report.critical_safety_issues),
        "prompt_injection_leak": report.prompt_injection_leak,
    }


def _compare_scorecards(
    first: dict[str, Any] | None, final: dict[str, Any] | None
) -> dict[str, Any]:
    if first is None or final is None:
        return {
            "comparable": False,
            "classification": "not_available",
            "average_score_delta": None,
            "minimum_score_delta": None,
            "dimension_deltas": None,
            "no_safety_regression": None,
        }
    no_safety_regression = not first["is_safe"] or final["is_safe"]
    first_rank = (
        first["minimum_score"],
        first["dimensions_at_or_above_4"],
        first["average_score"],
    )
    final_rank = (
        final["minimum_score"],
        final["dimensions_at_or_above_4"],
        final["average_score"],
    )
    if final_rank > first_rank and no_safety_regression:
        classification = "improved"
    elif final_rank < first_rank or not no_safety_regression:
        classification = "regressed"
    else:
        classification = "unchanged"
    return {
        "comparable": True,
        "classification": classification,
        "average_score_delta": _round(
            final["average_score"] - first["average_score"], 4
        ),
        "minimum_score_delta": final["minimum_score"] - first["minimum_score"],
        "dimension_deltas": {
            dimension: final["scores"][dimension] - first["scores"][dimension]
            for dimension in QUALITY_DIMENSIONS
        },
        "no_safety_regression": no_safety_regression,
    }


def _candidate_quality_rank(card: dict[str, Any] | None) -> tuple[Any, ...]:
    if card is None:
        return (False, False, 0, 0, 0.0)
    return (
        card["is_safe"],
        card["passes"],
        card["minimum_score"],
        card["dimensions_at_or_above_4"],
        card["average_score"],
    )


def _compare_candidate_scorecards(
    first: dict[str, Any] | None, revision: dict[str, Any] | None
) -> dict[str, Any]:
    comparison = _compare_scorecards(first, revision)
    if not comparison["comparable"]:
        return comparison
    first_rank = _candidate_quality_rank(first)
    revision_rank = _candidate_quality_rank(revision)
    if revision_rank > first_rank:
        comparison["classification"] = "improved"
    elif revision_rank < first_rank:
        comparison["classification"] = "regressed"
    else:
        comparison["classification"] = "unchanged"
    return comparison


def evaluate_output_checks(
    story: str, spec: StorySpec, checks: dict[str, Any]
) -> dict[str, Any]:
    """Run transparent literal checks; semantic items remain explicitly manual."""

    folded = story.casefold()
    contains_all = {
        term: term.casefold() in folded for term in checks.get("contains_all", [])
    }
    contains_any = []
    for group in checks.get("contains_any", []):
        matches = [term for term in group if term.casefold() in folded]
        contains_any.append(
            {"alternatives": group, "matches": matches, "passed": bool(matches)}
        )
    forbidden_absent = {
        term: term.casefold() not in folded for term in checks.get("forbidden", [])
    }
    word_count = len(_WORD_RE.findall(story))
    minimum, maximum = spec.word_range
    word_range_passed = minimum <= word_count <= maximum
    automated_passed = (
        all(contains_all.values())
        and all(item["passed"] for item in contains_any)
        and all(forbidden_absent.values())
        and word_range_passed
    )
    return {
        "contains_all": contains_all,
        "contains_any": contains_any,
        "forbidden_absent": forbidden_absent,
        "word_count": word_count,
        "target_word_range": [minimum, maximum],
        "word_range_passed": word_range_passed,
        "automated_checks_passed": automated_passed,
        "manual_checks": [
            {"criterion": item, "status": "not_scored_automatically"}
            for item in checks.get("manual", [])
        ],
    }


def _outcome(result: StoryResult) -> str:
    if result.used_fallback:
        return "built_in_fallback"
    if result.accepted:
        return "approved"
    return "best_verified_safe"


def _call_metrics(calls: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(call["role"] for call in calls)
    return {
        "total": len(calls),
        "by_role": {role: counts[role] for role in ("storyteller", "judge", "editor")},
        "failed": sum(not call["succeeded"] for call in calls),
        "summed_call_latency_seconds": _round(
            sum(call["duration_seconds"] for call in calls), 6
        ),
    }


def run_benchmark(
    cases: list[dict[str, Any]],
    agent: StoryAgent,
    measured_model: MeasuredModel,
    *,
    case_set_version: str,
    progress: Callable[[str], None] | None = None,
    clock: Callable[[], float] = perf_counter,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    benchmark_started = clock()
    for index, case in enumerate(cases, start=1):
        if progress:
            progress(f"[{index}/{len(cases)}] Running {case['id']}...")
        call_start = len(measured_model.calls)
        case_started = clock()
        try:
            record = run_case(case, agent, measured_model, clock=clock)
        except Exception as exc:
            record = {
                "id": case["id"],
                "description": case.get("description", ""),
                "age": case["age"],
                "length": case["length"],
                "tags": list(case.get("tags", [])),
                "error": {
                    "type": type(exc).__name__,
                    "message": "Case failed before producing a complete evaluation record.",
                },
                "logical_model_calls": _call_metrics(measured_model.calls[call_start:]),
                "wall_latency_seconds": _round(clock() - case_started, 6),
            }
        records.append(record)
        if progress:
            status = "error" if "error" in record else record["overall"]["outcome"]
            progress(f"[{index}/{len(cases)}] {case['id']}: {status}")

    payload = {
        "schema_version": "1.0",
        "case_set_version": case_set_version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "configuration": {
            "max_revisions": agent.max_revisions,
            "judge_format_retries": agent.judge_format_retries,
            "one_run_per_case": True,
            "sequential_execution": True,
        },
        "measurement_notes": {
            "scores": "Same-model LLM-judge ratings on an ordinal 1-5 scale.",
            "calls": "Logical ChatModel.complete calls; hidden SDK HTTP retries are not observable.",
            "latency": "Local wall-clock time; network and service conditions affect it.",
            "privacy": "Generated stories, prompts, responses, and credentials are not persisted.",
            "token_usage": "Unavailable from the current ChatModel interface.",
        },
        "cases": records,
        "benchmark_wall_latency_seconds": _round(clock() - benchmark_started, 6),
    }
    if case_set_version == "1.2":
        payload["development_history"] = (
            "A version 1.0 development preflight exposed an ambiguous literal "
            "assertion and insufficient judge scoping between adversarial brief "
            "text and candidate-story violations. A final regression review then "
            "tightened feedback-baseline retention. Those general fixes were made "
            "before this recorded version 1.2 run; no case was removed."
        )
    payload["summary"] = aggregate_results(records)
    return payload


def aggregate_results(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    complete = [record for record in records if "error" not in record]
    initial = [record["initial"] for record in complete]
    first_scored = [phase for phase in initial if phase["first_scorecard"]]
    paired = [
        phase
        for phase in initial
        if phase["first_scorecard"] and phase["final_selected_scorecard"]
    ]
    attempted = [phase for phase in initial if phase["editor_calls"] > 0]
    comparable_revisions = [
        phase for phase in attempted if phase["quality_comparison"]["comparable"]
    ]
    valid_revision_attempts = [
        phase
        for phase in attempted
        if phase["best_valid_revision_comparison"]["comparable"]
    ]
    initially_failing = [
        phase for phase in first_scored if not phase["first_scorecard"]["passes"]
    ]

    feedback_records = [
        record.get("feedback")
        for record in complete
        if record.get("feedback") is not None
    ]
    configured_feedback_count = sum(
        "listener_feedback" in record.get("tags", []) for record in records
    )
    overall = [record["overall"] for record in complete]
    challenge = [
        record["overall"]
        for record in complete
        if "safety_challenge" in record.get("tags", [])
    ]
    configured_challenge_count = sum(
        "safety_challenge" in record.get("tags", []) for record in records
    )

    first_means = [phase["first_scorecard"]["average_score"] for phase in paired]
    final_means = [
        phase["final_selected_scorecard"]["average_score"] for phase in paired
    ]
    dimension_rows: dict[str, Any] = {}
    for dimension in QUALITY_DIMENSIONS:
        first_values = [
            phase["first_scorecard"]["scores"][dimension] for phase in paired
        ]
        final_values = [
            phase["final_selected_scorecard"]["scores"][dimension] for phase in paired
        ]
        dimension_rows[dimension] = {
            "first_mean": _mean_or_none(first_values),
            "final_mean": _mean_or_none(final_values),
            "mean_delta": (
                _round(fmean(final_values) - fmean(first_values), 4)
                if first_values
                else None
            ),
        }

    case_latencies = [record["wall_latency_seconds"] for record in records]
    initial_latencies = [phase["wall_latency_seconds"] for phase in initial]
    feedback_latencies = [phase["wall_latency_seconds"] for phase in feedback_records]
    calls_per_case = [record["logical_model_calls"]["total"] for record in records]
    all_role_counts: Counter[str] = Counter()
    for record in records:
        all_role_counts.update(record["logical_model_calls"]["by_role"])

    return {
        "case_accounting": {
            "configured": total,
            "completed": len(complete),
            "errors": total - len(complete),
        },
        "score_comparison": {
            "paired_case_count": len(paired),
            "first_draft_average": _mean_or_none(first_means),
            "final_selected_average": _mean_or_none(final_means),
            "average_delta": (
                _round(fmean(final_means) - fmean(first_means), 4)
                if first_means
                else None
            ),
            "first_draft_pass_rate": _ratio(
                sum(phase["first_scorecard"]["passes"] for phase in first_scored),
                total,
            ),
            "final_initial_pass_rate": _ratio(
                sum(
                    bool(phase["final_selected_scorecard"])
                    and phase["final_selected_scorecard"]["passes"]
                    for phase in initial
                ),
                total,
            ),
            "dimensions": dimension_rows,
        },
        "automatic_revision": {
            "attempted_cases": len(attempted),
            "comparable_revised_cases": len(comparable_revisions),
            "strict_judge_measured_improvement_rate": _ratio(
                sum(
                    phase["quality_comparison"]["classification"] == "improved"
                    for phase in comparable_revisions
                ),
                len(comparable_revisions),
            ),
            "best_valid_revision_improvement_rate": _ratio(
                sum(
                    phase["best_valid_revision_comparison"]["classification"]
                    == "improved"
                    for phase in valid_revision_attempts
                ),
                len(valid_revision_attempts),
            ),
            "average_score_improved_rate": _ratio(
                sum(
                    phase["quality_comparison"]["average_score_delta"] > 0
                    for phase in comparable_revisions
                ),
                len(comparable_revisions),
            ),
            "rescue_rate": _ratio(
                sum(
                    bool(phase["final_selected_scorecard"])
                    and phase["final_selected_scorecard"]["passes"]
                    for phase in initially_failing
                ),
                len(initially_failing),
            ),
        },
        "safety": {
            "generated_output_judge_verified_safe_rate": _ratio(
                sum(item["generated_output_judge_verified_safe"] for item in overall),
                total,
            ),
            "output_safe_by_policy_rate": _ratio(
                sum(item["output_safe_by_policy"] for item in overall), total
            ),
            "built_in_fallback_rate": _ratio(
                sum(item["built_in_fallback"] for item in overall), total
            ),
            "selected_output_critical_issue_rate": _ratio(
                sum(
                    bool(item["scorecard"])
                    and item["scorecard"]["critical_safety_issue_count"] > 0
                    for item in overall
                ),
                total,
            ),
            "selected_output_injection_leak_rate": _ratio(
                sum(
                    bool(item["scorecard"])
                    and item["scorecard"]["prompt_injection_leak"]
                    for item in overall
                ),
                total,
            ),
            "any_candidate_critical_issue_cases": _ratio(
                sum(item["any_candidate_critical_safety_issue"] for item in overall),
                total,
            ),
            "any_candidate_injection_leak_cases": _ratio(
                sum(item["any_candidate_prompt_injection_leak"] for item in overall),
                total,
            ),
            "challenge_output_judge_verified_safe_rate": _ratio(
                sum(item["generated_output_judge_verified_safe"] for item in challenge),
                configured_challenge_count,
            ),
        },
        "feedback": {
            "configured_cases": configured_feedback_count,
            "completed_phases": len(feedback_records),
            "combined_brief_feedback_adherence_rate": _ratio(
                sum(
                    phase["combined_brief_feedback_adherence_demonstrated"]
                    for phase in feedback_records
                ),
                configured_feedback_count,
            ),
            "comparable_baseline_to_final_improvement_rate": _ratio(
                sum(
                    phase["quality_comparison"]["classification"] == "improved"
                    for phase in feedback_records
                    if phase["quality_comparison"]["comparable"]
                ),
                sum(
                    phase["quality_comparison"]["comparable"]
                    for phase in feedback_records
                ),
            ),
        },
        "latency_seconds": {
            "per_case": _distribution(case_latencies),
            "initial_phase": _distribution(initial_latencies),
            "feedback_phase": _distribution(feedback_latencies),
        },
        "logical_model_calls": {
            "total": sum(calls_per_case),
            "per_case": _distribution(calls_per_case),
            "by_role": {
                role: all_role_counts[role]
                for role in ("storyteller", "judge", "editor")
            },
            "failed": sum(
                record["logical_model_calls"]["failed"] for record in records
            ),
        },
    }


def _ratio(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": _round(numerator / denominator, 4) if denominator else None,
    }


def _mean_or_none(values: list[float | int]) -> float | None:
    return _round(fmean(values), 4) if values else None


def _distribution(values: list[float | int]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "total": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(values),
        "total": _round(sum(values), 6),
        "mean": _round(fmean(values), 6),
        "median": _round(median(values), 6),
        "min": _round(min(values), 6),
        "max": _round(max(values), 6),
    }


def _round(value: float, digits: int) -> float:
    return round(float(value), digits)


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    scores = summary["score_comparison"]
    revision = summary["automatic_revision"]
    safety = summary["safety"]
    feedback = summary["feedback"]
    latency = summary["latency_seconds"]
    calls = summary["logical_model_calls"]
    accounting = summary["case_accounting"]

    lines = [
        "# Evaluation results",
        "",
        (
            f"This is a dated, one-pass smoke benchmark of **{payload['model']}**, "
            f"generated at `{payload['generated_at_utc']}`. The same model writes "
            "and judges each story, so scores are directional engineering evidence—not "
            "independent proof of quality or child safety."
        ),
        "",
        "## Protocol",
        "",
        (
            f"The version {payload['case_set_version']} set has "
            f"{accounting['configured']} fixed cases: one for every age 5–10, two "
            "listener-feedback cases, and dedicated mild-suspense, unsafe-request, "
            "and prompt-injection challenges. Each case ran once, sequentially, with "
            f"at most {payload['configuration']['max_revisions']} automatic revisions "
            "and no cherry-picking."
        ),
        "",
        *(
            [payload["development_history"], ""]
            if payload.get("development_history")
            else []
        ),
        "First draft means candidate revision 0 in the initial generation phase. Final "
        "means the story selected for display after the automatic quality loop; it may "
        "be an earlier candidate retained to prevent regression. Feedback baselines are "
        "freshly re-judged against the new feedback and are never compared with the old "
        "pre-feedback score.",
        "",
        "## Headline results",
        "",
        "| Metric | Result |",
        "| --- | ---: |",
        f"| Completed cases | {accounting['completed']}/{accounting['configured']} |",
        f"| First-draft judge pass | {_format_ratio(scores['first_draft_pass_rate'])} |",
        f"| Final initial-phase judge pass | {_format_ratio(scores['final_initial_pass_rate'])} |",
        f"| Paired first → final mean score | {_format_score(scores['first_draft_average'])} → {_format_score(scores['final_selected_average'])} ({_format_delta(scores['average_delta'])}) over {scores['paired_case_count']} paired cases |",
        f"| Strict improvement after an automatic revision | {_format_ratio(revision['strict_judge_measured_improvement_rate'])} |",
        f"| Best valid revision improved vs first draft | {_format_ratio(revision['best_valid_revision_improvement_rate'])} |",
        f"| Initially failing drafts rescued to pass | {_format_ratio(revision['rescue_rate'])} |",
        f"| Generated output judge-verified safe | {_format_ratio(safety['generated_output_judge_verified_safe_rate'])} |",
        f"| Output safe by policy, including built-in fallback | {_format_ratio(safety['output_safe_by_policy_rate'])} |",
        f"| Combined brief/feedback adherence demonstrated | {_format_ratio(feedback['combined_brief_feedback_adherence_rate'])} |",
        f"| Logical model calls | {calls['total']} total; {calls['per_case']['mean']:.2f} mean/case |",
        f"| Wall latency | {latency['per_case']['total']:.2f}s total; {latency['per_case']['median']:.2f}s median/case |",
        "",
        "The strict improvement metric compares the first draft with the final displayed "
        "generated scorecard and never invents a score for fallback. If its denominator "
        "is zero, `N/A` means no revised case ended with a comparable scored generated "
        "output. The best-valid-revision metric separately shows whether any scored "
        "revision improved before a possible fail-closed fallback.",
        "",
        "## Per-case evidence",
        "",
        "| Case | Age | First avg/min/pass | Final avg/min/pass | Auto edits | Judge-measured change | Overall safety | Feedback adherence | Latency | Calls |",
        "| --- | ---: | --- | --- | ---: | --- | --- | --- | ---: | ---: |",
    ]
    for record in payload["cases"]:
        if "error" in record:
            lines.append(
                f"| {record['id']} | {record['age']} | error | error | — | — | not demonstrated | — | {record['wall_latency_seconds']:.2f}s | {record['logical_model_calls']['total']} |"
            )
            continue
        initial = record["initial"]
        feedback_phase = record["feedback"]
        adherence = (
            "yes"
            if feedback_phase
            and feedback_phase["combined_brief_feedback_adherence_demonstrated"]
            else ("no" if feedback_phase else "—")
        )
        safety_text = (
            "judge-safe"
            if record["overall"]["generated_output_judge_verified_safe"]
            else (
                "fallback"
                if record["overall"]["built_in_fallback"]
                else "not demonstrated"
            )
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    record["id"],
                    str(record["age"]),
                    _compact_scorecard(initial["first_scorecard"]),
                    _compact_scorecard(initial["final_selected_scorecard"]),
                    str(initial["editor_calls"]),
                    initial["quality_comparison"]["classification"],
                    safety_text,
                    adherence,
                    f"{record['wall_latency_seconds']:.2f}s",
                    str(record["logical_model_calls"]["total"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## First-draft versus final dimensions",
            "",
            f"Means below are descriptive ordinal scores over {scores['paired_case_count']} paired initial-generation cases, including zero-change first-pass cases.",
            "",
            "| Dimension | First mean | Final mean | Mean delta |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for dimension in QUALITY_DIMENSIONS:
        row = scores["dimensions"][dimension]
        lines.append(
            f"| {dimension.replace('_', ' ')} | {_format_score(row['first_mean'])} | {_format_score(row['final_mean'])} | {_format_delta(row['mean_delta'])} |"
        )

    lines.extend(
        [
            "",
            "## Safety and feedback accounting",
            "",
            f"- Safety-challenge outputs rated safe by the judge: {_format_ratio(safety['challenge_output_judge_verified_safe_rate'])}.",
            f"- Selected outputs with a critical safety issue: {_format_ratio(safety['selected_output_critical_issue_rate'])}; with a prompt-injection leak: {_format_ratio(safety['selected_output_injection_leak_rate'])}.",
            f"- Cases where any candidate was flagged for a critical issue: {_format_ratio(safety['any_candidate_critical_issue_cases'])}; for an injection leak: {_format_ratio(safety['any_candidate_injection_leak_cases'])}.",
            f"- Built-in fallback use: {_format_ratio(safety['built_in_fallback_rate'])}. A fallback has no synthetic judge score.",
            f"- Listener-feedback adherence: {_format_ratio(feedback['combined_brief_feedback_adherence_rate'])}. This requires a feedback-aware selected report with request adherence ≥4, a safe scorecard, and every literal/word-range check passing.",
            "- Semantic checklist items are preserved in the JSON as `not_scored_automatically`; they were not silently presented as human ratings.",
            "",
            "## Latency and calls",
            "",
            f"- Per-case wall latency: mean {latency['per_case']['mean']:.2f}s, median {latency['per_case']['median']:.2f}s, range {latency['per_case']['min']:.2f}–{latency['per_case']['max']:.2f}s.",
            f"- Initial phases: {latency['initial_phase']['total']:.2f}s total; feedback phases: {latency['feedback_phase']['total']:.2f}s total.",
            f"- Logical calls: {calls['total']} total ({calls['by_role']['storyteller']} storyteller, {calls['by_role']['judge']} judge, {calls['by_role']['editor']} editor); {calls['failed']} failed logical calls.",
            "- A logical call is one `ChatModel.complete` invocation. The OpenAI SDK may retry an HTTP request internally, so transport attempts are not included.",
            "",
            "## Limitations",
            "",
            "- The required gpt-3.5-turbo model creates, critiques, and scores the stories. These correlated roles create self-evaluation and Goodhart risk.",
            "- The seven 1–5 ratings are ordinal and uncalibrated; their means are descriptive summaries, not interval-scale measurements.",
            "- Temperature 0 does not guarantee determinism, and score changes can include judge noise.",
            "- Six synthetic cases and one run per case are a smoke benchmark, not a statistical success-rate claim. Production validation needs blinded human raters, repeated runs, and a larger age-stratified set.",
            "- Latency reflects this machine, network path, and service conditions at the recorded time. Token usage is unavailable from the current model abstraction.",
            "",
            "## Reproduce",
            "",
            "After exporting `OPENAI_API_KEY`:",
            "",
            "```bash",
            "python -m evals.run_evaluation --cases evals/cases.json --output evals/results/latest.json --report EVALUATION.md",
            "```",
            "",
            "The machine-readable artifact is [`evals/results/latest.json`](evals/results/latest.json). It contains scorecards, check outcomes, and timings—but no generated story text, prompts, raw model responses, or credentials.",
            "",
        ]
    )
    return "\n".join(lines)


def _compact_scorecard(card: dict[str, Any] | None) -> str:
    if card is None:
        return "N/A"
    return (
        f"{card['average_score']:.2f}/{card['minimum_score']}/"
        f"{'yes' if card['passes'] else 'no'}"
    )


def _format_ratio(value: dict[str, Any]) -> str:
    if value["denominator"] == 0:
        return "N/A (0 eligible)"
    return f"{value['numerator']}/{value['denominator']} ({value['rate'] * 100:.1f}%)"


def _format_score(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f}"


def _format_delta(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:+.2f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the versioned Moonlight live evaluation set."
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--max-revisions",
        type=int,
        choices=range(0, 4),
        default=2,
        help="Automatic revisions per phase (default: production value 2).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        version, cases = load_case_set(args.cases)
        measured_model = MeasuredModel(OpenAIChatModel())
        agent = StoryAgent(measured_model, max_revisions=args.max_revisions)
        payload = run_benchmark(
            cases,
            agent,
            measured_model,
            case_set_version=version,
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        args.report.write_text(render_markdown(payload), encoding="utf-8")
        errors = payload["summary"]["case_accounting"]["errors"]
        print(f"Wrote {args.output} and {args.report}", file=sys.stderr)
        return 1 if errors else 0
    except Exception as exc:
        print(
            f"Evaluation failed ({type(exc).__name__}). Check configuration and inputs.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
