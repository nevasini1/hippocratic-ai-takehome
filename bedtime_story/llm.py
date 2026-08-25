"""OpenAI Chat Completions adapter used by every logical agent role."""

from __future__ import annotations

import os
from typing import Any, Protocol

from openai import OpenAI


# Assignment constraint: every storyteller/judge/editor call uses this model.
# It is intentionally not configurable through the CLI or environment.
MODEL_NAME = "gpt-3.5-turbo"


class ConfigurationError(RuntimeError):
    """Raised when local configuration is incomplete."""


class ModelResponseError(RuntimeError):
    """Raised when the SDK returns no usable text."""


class ChatModel(Protocol):
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        json_mode: bool = False,
    ) -> str:
        """Return one non-empty model message."""


class OpenAIChatModel:
    """Thin, testable adapter around the current OpenAI Python SDK."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        client: Any | None = None,
    ) -> None:
        if client is not None:
            self._client = client
            return

        key = (api_key or os.getenv("OPENAI_API_KEY", "")).strip()
        if not key:
            raise ConfigurationError(
                "OPENAI_API_KEY is not set. Export it in your shell before running."
            )
        self._client = OpenAI(api_key=key, timeout=60.0, max_retries=2)

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        json_mode: bool = False,
    ) -> str:
        request: dict[str, Any] = {
            "model": MODEL_NAME,
            "messages": messages,
            "stream": False,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            request["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(**request)
        try:
            choice = response.choices[0]
            content = choice.message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise ModelResponseError(
                "The model returned an unexpected response."
            ) from exc

        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason not in {None, "stop"}:
            raise ModelResponseError(
                "The model stopped before completing a usable response."
            )
        if not isinstance(content, str) or not content.strip():
            raise ModelResponseError("The model returned an empty response.")
        return content.strip()
