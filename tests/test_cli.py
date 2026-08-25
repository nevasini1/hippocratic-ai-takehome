from __future__ import annotations

from io import StringIO

from bedtime_story.llm import ConfigurationError
from main import run
from tests.support import ScriptedModel, judge_json


def test_noninteractive_cli_prints_story_and_optional_scorecard() -> None:
    model = ScriptedModel(["# A Good Story", judge_json(default_score=5)])
    stdout = StringIO()
    stderr = StringIO()

    status = run(
        [
            "--request",
            "A sleepy fox",
            "--age",
            "6",
            "--length",
            "short",
            "--max-revisions",
            "0",
            "--show-evaluation",
        ],
        model_factory=lambda: model,
        stdout=stdout,
        stderr=stderr,
    )

    assert status == 0
    assert "# A Good Story" in stdout.getvalue()
    assert "Status: approved" in stdout.getvalue()
    assert "Average: 5.00/5" in stdout.getvalue()
    assert "Creating, reviewing" in stderr.getvalue()
    model.assert_finished()


def test_empty_request_is_rejected_before_model_construction() -> None:
    created = False

    def factory():
        nonlocal created
        created = True
        return ScriptedModel([])

    stderr = StringIO()
    status = run(
        ["--request", "   "],
        model_factory=factory,
        stdout=StringIO(),
        stderr=stderr,
    )

    assert status == 2
    assert not created
    assert "cannot be empty" in stderr.getvalue()


def test_configuration_error_is_friendly() -> None:
    def factory():
        raise ConfigurationError("OPENAI_API_KEY is not set.")

    stderr = StringIO()
    status = run(
        ["--request", "A moon rabbit"],
        model_factory=factory,
        stdout=StringIO(),
        stderr=stderr,
    )

    assert status == 2
    assert stderr.getvalue().strip() == "Error: OPENAI_API_KEY is not set."


def test_api_error_does_not_echo_exception_message_or_key_material() -> None:
    # SDKs can raise ValueError for malformed parameters. It must not be confused
    # with our listener-input validation or echo potentially sensitive details.
    model = ScriptedModel([ValueError("private-key-material")])
    stderr = StringIO()

    status = run(
        ["--request", "A moon rabbit"],
        model_factory=lambda: model,
        stdout=StringIO(),
        stderr=stderr,
    )

    assert status == 1
    assert "ValueError" in stderr.getvalue()
    assert "private-key-material" not in stderr.getvalue()


def test_interactive_feedback_revises_and_rejudges_story() -> None:
    model = ScriptedModel(
        [
            "Original story",
            judge_json(default_score=5),
            judge_json(
                default_score=5,
                score_overrides={"request_adherence": 3},
            ),
            "Funnier story",
            judge_json(default_score=5),
        ]
    )
    answers = iter(["A polite dragon", "make the dragon funnier", ""])
    stdout = StringIO()

    status = run(
        ["--max-revisions", "0"],
        model_factory=lambda: model,
        input_fn=lambda _prompt: next(answers),
        stdout=stdout,
        stderr=StringIO(),
    )

    assert status == 0
    assert "Original story" in stdout.getvalue()
    assert "Funnier story" in stdout.getvalue()
    assert len(model.calls) == 5
    model.assert_finished()
