"""Command-line entry point for the Hippocratic AI bedtime-story assignment.

With two more hours, I would expand the included smoke benchmark into repeated,
age-stratified runs and collect blinded ratings from parents or children's-
fiction editors to calibrate the same-model judge. I would also add opt-in token
telemetry, production dashboards, and adversarial red-team cases without storing
story text or other listener-controlled content.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from typing import TextIO

from bedtime_story import InputValidationError, StoryAgent, StoryResult, StorySpec
from bedtime_story.llm import ChatModel, ConfigurationError, OpenAIChatModel


MAX_FEEDBACK_ROUNDS = 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an age-appropriate bedtime story, judge it, and revise it "
            "until it meets a structured quality bar."
        )
    )
    parser.add_argument(
        "--request",
        help="Story idea. If omitted, the program asks interactively.",
    )
    parser.add_argument(
        "--age",
        type=int,
        choices=range(5, 11),
        default=7,
        metavar="5-10",
        help="Target listener age (default: 7).",
    )
    parser.add_argument(
        "--length",
        choices=("short", "medium", "long"),
        default="medium",
        help="Approximate story length (default: medium).",
    )
    parser.add_argument(
        "--max-revisions",
        type=int,
        choices=range(0, 4),
        default=2,
        metavar="0-3",
        help="Maximum automatic revisions after each first candidate (default: 2).",
    )
    parser.add_argument(
        "--show-evaluation",
        action="store_true",
        help="Print the final scorecard and revision metadata after the story.",
    )
    parser.add_argument(
        "--feedback",
        action="store_true",
        help="Ask for listener changes even when --request is supplied.",
    )
    return parser


def run(
    argv: Sequence[str] | None = None,
    *,
    model_factory: Callable[[], ChatModel] = OpenAIChatModel,
    input_fn: Callable[[str], str] = input,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    args = build_parser().parse_args(argv)
    interactive_request = args.request is None

    try:
        request = args.request
        if request is None:
            print("Moonlight Bedtime Storyteller", file=stdout)
            request = input_fn("What kind of story would you like? ")

        spec = StorySpec(request=request, age=args.age, length=args.length)
        model = model_factory()
        agent = StoryAgent(model, max_revisions=args.max_revisions)

        print("Creating, reviewing, and polishing your story...", file=stderr)
        result = agent.create_story(spec)
        _print_result(result, args.show_evaluation, stdout)

        if interactive_request or args.feedback:
            _feedback_loop(
                agent,
                result,
                show_evaluation=args.show_evaluation,
                input_fn=input_fn,
                stdout=stdout,
                stderr=stderr,
            )
        return 0
    except (EOFError, KeyboardInterrupt):
        print("\nGoodnight!", file=stderr)
        return 130
    except (InputValidationError, ConfigurationError) as exc:
        print(f"Error: {exc}", file=stderr)
        return 2
    except Exception as exc:  # The CLI should not expose request bodies or secrets.
        print(
            "Error: the OpenAI request failed "
            f"({type(exc).__name__}). Check your API key, model access, and network, "
            "then try again.",
            file=stderr,
        )
        return 1


def _feedback_loop(
    agent: StoryAgent,
    result: StoryResult,
    *,
    show_evaluation: bool,
    input_fn: Callable[[str], str],
    stdout: TextIO,
    stderr: TextIO,
) -> StoryResult:
    for _ in range(MAX_FEEDBACK_ROUNDS):
        feedback = input_fn(
            "\nPress Enter to finish, or describe a change "
            "(for example, 'make it funnier'): "
        ).strip()
        if not feedback or feedback.lower() in {"done", "quit", "q"}:
            return result

        try:
            print("Applying your feedback and re-checking the story...", file=stderr)
            result = agent.refine_story(result, feedback)
        except InputValidationError as exc:
            print(f"Feedback not applied: {exc}", file=stderr)
            continue
        _print_result(result, show_evaluation, stdout)

    print(
        f"Reached the limit of {MAX_FEEDBACK_ROUNDS} feedback rounds for this session.",
        file=stderr,
    )
    return result


def _print_result(result: StoryResult, show_evaluation: bool, stdout: TextIO) -> None:
    print(f"\n{result.story}\n", file=stdout)
    if show_evaluation:
        print(_format_evaluation(result), file=stdout)


def _format_evaluation(result: StoryResult) -> str:
    if result.used_fallback:
        status = "safe fallback"
    elif result.accepted:
        status = "approved"
    else:
        status = "best verified-safe candidate"

    lines = [
        "--- Quality report ---",
        f"Status: {status}",
        f"Candidates judged: {len(result.candidates)}",
        f"Revision number selected: {result.revision_count}",
        f"Listener feedback rounds: {result.feedback_rounds}",
    ]
    if result.report is not None:
        for name, score in result.report.scores.items():
            lines.append(f"{name.replace('_', ' ').title()}: {score}/5")
        lines.append(f"Average: {result.report.average_score:.2f}/5")
        lines.append(f"Judge summary: {result.report.summary}")
    if result.quality_note:
        lines.append(f"Note: {result.quality_note}")
    return "\n".join(lines)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
