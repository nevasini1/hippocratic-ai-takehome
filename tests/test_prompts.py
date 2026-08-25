from bedtime_story.models import QUALITY_DIMENSIONS
from bedtime_story.prompts import (
    EDITOR_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    STORYTELLER_SYSTEM_PROMPT,
)


def test_judge_prompt_does_not_prime_placeholder_scores() -> None:
    """A numeric JSON template made GPT-3.5 copy 1 into every score."""

    for dimension in QUALITY_DIMENSIONS:
        assert f'"{dimension}": 1' not in JUDGE_SYSTEM_PROMPT
        assert dimension in JUDGE_SYSTEM_PROMPT

    assert "5 = exceptional" in JUDGE_SYSTEM_PROMPT
    assert "Do not copy a default score" in JUDGE_SYSTEM_PROMPT


def test_adversarial_brief_is_distinguished_from_candidate_violation() -> None:
    assert "only from the candidate text" in JUDGE_SYSTEM_PROMPT
    assert "mere presence in the brief is not" in JUDGE_SYSTEM_PROMPT
    assert "omit those parts entirely" in STORYTELLER_SYSTEM_PROMPT
    assert "remove the offending material entirely" in EDITOR_SYSTEM_PROMPT
    assert "target word range" in EDITOR_SYSTEM_PROMPT
