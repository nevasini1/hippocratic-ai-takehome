"""Bounded storyteller → judge → editor orchestration."""

from __future__ import annotations

from .json_tools import parse_judge_report
from .llm import ChatModel
from .models import (
    JudgeFormatError,
    JudgeReport,
    StoryCandidate,
    StoryResult,
    StorySpec,
    validate_feedback,
)
from .prompts import editor_messages, judge_messages, storyteller_messages


SAFE_FALLBACK_STORY = """# The Little Lantern's Goodnight

At the edge of a quiet meadow stood a little blue lantern named Luma. Each evening, Luma helped the fireflies find their favorite flowers and showed the crickets where to set up their tiny music stands.

One night, a soft cloud covered the moon. "What if my light is not bright enough?" Luma whispered.

An old moth named Miri settled beside her. "A gentle light can still show the next step," she said.

So Luma took one small step through the silver grass. Her glow found a firefly waiting under a leaf. Together their lights grew warmer. Then they found two crickets beside a smooth stone, and soon four tiny lights bobbed across the meadow like sleepy stars.

They did not light the whole world at once. They lit one dewdrop, one flower, and one friendly face at a time. Before long, everyone had reached the cozy hollow beneath the willow tree.

Inside, the crickets placed their music stands in a neat half-moon. The fireflies hung glowing beads of dew above them. Luma noticed that every small light had added something the others could not, and the hollow felt brighter because they had traveled together.

The cloud drifted on, and the moon returned. Luma yawned a golden little yawn.

"You were right," she told Miri. "The next step was enough."

The fireflies dimmed their glow. The crickets played their softest song. Luma curled beneath a clover blanket while the meadow breathed in and out around her.

Miri tucked the blanket beneath Luma's round blue edge. One by one, the meadow friends whispered goodnight—to the flowers, to the willow, to the moon, and to one another. Luma's glow became as small and peaceful as a candle seen from far away.

And under the patient moon, every small light rested until morning."""


class StoryAgent:
    """Create and refine stories while enforcing a finite quality-control loop."""

    def __init__(
        self,
        model: ChatModel,
        *,
        max_revisions: int = 2,
        judge_format_retries: int = 1,
    ) -> None:
        if (
            isinstance(max_revisions, bool)
            or not isinstance(max_revisions, int)
            or not 0 <= max_revisions <= 3
        ):
            raise ValueError("max_revisions must be a whole number from 0 to 3.")
        if (
            isinstance(judge_format_retries, bool)
            or not isinstance(judge_format_retries, int)
            or not 0 <= judge_format_retries <= 2
        ):
            raise ValueError("judge_format_retries must be a whole number from 0 to 2.")
        self._model = model
        self.max_revisions = max_revisions
        self.judge_format_retries = judge_format_retries

    def create_story(self, spec: StorySpec) -> StoryResult:
        draft = self._model.complete(
            storyteller_messages(spec),
            temperature=0.8,
            max_tokens=2_200,
        )
        return self._quality_loop(
            spec,
            draft,
            first_revision_number=0,
            feedback_rounds=0,
        )

    def refine_story(self, result: StoryResult, feedback: str) -> StoryResult:
        """Apply listener feedback, then put the result through the same judge."""

        normalized_feedback = validate_feedback(feedback)
        baseline_report: JudgeReport | None = None
        try:
            # Re-score the prior story against the new feedback so its old score
            # is not compared with a revision evaluated under a different brief.
            baseline_report = self._judge(
                result.spec,
                result.story,
                listener_feedback=normalized_feedback,
            )
        except JudgeFormatError:
            # Feedback can still be attempted using the last valid review. The
            # revised candidate must receive a valid new judgment before display.
            pass

        revised = self._model.complete(
            editor_messages(
                result.spec,
                result.story,
                baseline_report or result.report,
                listener_feedback=normalized_feedback,
            ),
            temperature=0.5,
            max_tokens=2_200,
        )
        return self._quality_loop(
            result.spec,
            revised,
            first_revision_number=1,
            feedback_rounds=result.feedback_rounds + 1,
            listener_feedback=normalized_feedback,
            previous_safe_result=result,
            initial_candidates=(
                (StoryCandidate(result.story, baseline_report, 0),)
                if baseline_report is not None
                else ()
            ),
        )

    def _quality_loop(
        self,
        spec: StorySpec,
        first_story: str,
        *,
        first_revision_number: int,
        feedback_rounds: int,
        listener_feedback: str | None = None,
        previous_safe_result: StoryResult | None = None,
        initial_candidates: tuple[StoryCandidate, ...] = (),
    ) -> StoryResult:
        candidates = list(initial_candidates)
        story = first_story

        # max_revisions means N additional drafts after the first candidate.
        for additional_revision in range(self.max_revisions + 1):
            revision_number = first_revision_number + additional_revision
            try:
                report = self._judge(
                    spec,
                    story,
                    listener_feedback=listener_feedback,
                )
            except JudgeFormatError:
                return self._best_safe_or_fallback(
                    spec,
                    candidates,
                    revision_count=revision_number,
                    feedback_rounds=feedback_rounds,
                    previous_safe_result=previous_safe_result,
                    note=(
                        "The judge returned invalid structured output after a retry; "
                        "a previously verified safe story was used."
                    ),
                )

            candidate = StoryCandidate(story, report, revision_number)
            candidates.append(candidate)
            if report.passes:
                # A feedback-aware baseline may already pass. Keep it when a
                # later passing edit scores worse; equal-quality edits still
                # win the final tie-break so harmless requested changes apply.
                passing_candidates = [item for item in candidates if item.report.passes]
                selected = max(passing_candidates, key=_candidate_quality_key)
                return StoryResult(
                    spec=spec,
                    story=selected.story,
                    report=selected.report,
                    accepted=True,
                    used_fallback=False,
                    revision_count=selected.revision_number,
                    feedback_rounds=feedback_rounds,
                    candidates=tuple(candidates),
                )

            if additional_revision < self.max_revisions:
                story = self._model.complete(
                    editor_messages(
                        spec,
                        story,
                        report,
                        listener_feedback=listener_feedback,
                    ),
                    temperature=0.5,
                    max_tokens=2_200,
                )

        return self._best_safe_or_fallback(
            spec,
            candidates,
            revision_count=first_revision_number + self.max_revisions,
            feedback_rounds=feedback_rounds,
            previous_safe_result=previous_safe_result,
            note=(
                "The revision limit was reached; the highest-scoring verified safe "
                "candidate was used."
            ),
        )

    def _judge(
        self,
        spec: StorySpec,
        story: str,
        *,
        listener_feedback: str | None,
    ) -> JudgeReport:
        last_error: JudgeFormatError | None = None
        previous_invalid_output: str | None = None
        for _ in range(self.judge_format_retries + 1):
            raw_report = self._model.complete(
                judge_messages(
                    spec,
                    story,
                    listener_feedback=listener_feedback,
                    previous_invalid_output=previous_invalid_output,
                    validation_error=str(last_error) if last_error else None,
                ),
                temperature=0.0,
                max_tokens=900,
                json_mode=True,
            )
            try:
                return parse_judge_report(raw_report)
            except JudgeFormatError as exc:
                last_error = exc
                # Limit retry context so a pathological model response cannot
                # expand the next prompt without bound. It remains quoted data.
                previous_invalid_output = raw_report[:4_000]
        assert last_error is not None
        raise last_error

    def _best_safe_or_fallback(
        self,
        spec: StorySpec,
        candidates: list[StoryCandidate],
        *,
        revision_count: int,
        feedback_rounds: int,
        previous_safe_result: StoryResult | None,
        note: str,
    ) -> StoryResult:
        safe_candidates = [
            candidate for candidate in candidates if candidate.report.is_safe
        ]
        if safe_candidates:
            best = max(safe_candidates, key=_candidate_quality_key)
            return StoryResult(
                spec=spec,
                story=best.story,
                report=best.report,
                accepted=False,
                used_fallback=False,
                revision_count=best.revision_number,
                feedback_rounds=feedback_rounds,
                candidates=tuple(candidates),
                quality_note=note,
            )

        if previous_safe_result is not None and (
            previous_safe_result.used_fallback
            or (
                previous_safe_result.report is not None
                and previous_safe_result.report.is_safe
            )
        ):
            return StoryResult(
                spec=spec,
                story=previous_safe_result.story,
                report=previous_safe_result.report,
                accepted=False,
                used_fallback=previous_safe_result.used_fallback,
                revision_count=0,
                feedback_rounds=feedback_rounds,
                candidates=tuple(candidates),
                quality_note=(
                    f"{note} The prior safe version was kept instead of an "
                    "unverified revision."
                ),
            )

        return StoryResult(
            spec=spec,
            story=SAFE_FALLBACK_STORY,
            report=None,
            accepted=False,
            used_fallback=True,
            revision_count=revision_count,
            feedback_rounds=feedback_rounds,
            candidates=tuple(candidates),
            quality_note=(
                f"{note} No generated candidate passed the safety gate, so the "
                "built-in safe fallback was used."
            ),
        )


def _candidate_quality_key(candidate: StoryCandidate) -> tuple[int, int, float, int]:
    """Match the documented balanced-quality ordering with latest as tie-break."""

    return (
        candidate.report.minimum_score,
        sum(score >= 4 for score in candidate.report.scores.values()),
        candidate.report.average_score,
        candidate.revision_number,
    )
