from __future__ import annotations

from types import SimpleNamespace

import pytest

from bedtime_story.llm import (
    MODEL_NAME,
    ConfigurationError,
    ModelResponseError,
    OpenAIChatModel,
)


class FakeCompletions:
    def __init__(
        self,
        content: str | None = "hello",
        *,
        finish_reason: str = "stop",
    ) -> None:
        self.content = content
        self.finish_reason = finish_reason
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.content),
                    finish_reason=self.finish_reason,
                )
            ]
        )


def fake_client(completions: FakeCompletions):
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def test_every_sdk_call_uses_assignment_model_and_json_mode() -> None:
    completions = FakeCompletions("  result  ")
    model = OpenAIChatModel(client=fake_client(completions))

    output = model.complete(
        [{"role": "user", "content": "hello"}],
        temperature=0.0,
        max_tokens=100,
        json_mode=True,
    )

    assert MODEL_NAME == "gpt-3.5-turbo"
    assert output == "result"
    assert completions.requests[0]["model"] == "gpt-3.5-turbo"
    assert completions.requests[0]["response_format"] == {"type": "json_object"}
    assert "OPENAI_MODEL" not in completions.requests[0]


def test_missing_api_key_fails_before_an_sdk_request(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY is not set"):
        OpenAIChatModel()


@pytest.mark.parametrize("content", [None, "", "   "])
def test_empty_model_content_is_rejected(content: str | None) -> None:
    model = OpenAIChatModel(client=fake_client(FakeCompletions(content)))

    with pytest.raises(ModelResponseError, match="empty response"):
        model.complete(
            [{"role": "user", "content": "hello"}],
            temperature=0.0,
            max_tokens=100,
        )


def test_truncated_model_content_is_rejected() -> None:
    model = OpenAIChatModel(
        client=fake_client(FakeCompletions("partial story", finish_reason="length"))
    )

    with pytest.raises(ModelResponseError, match="before completing"):
        model.complete(
            [{"role": "user", "content": "hello"}],
            temperature=0.8,
            max_tokens=100,
        )
