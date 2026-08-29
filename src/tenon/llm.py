"""LLM client: talks to any OpenAI-compatible chat-completions endpoint.

Only the `openai` SDK is used for the wire protocol. Retry policy, usage
accounting, and the known protocol pitfalls (verbatim assistant replay,
tool_call_id pairing) are hand-written here — the agent loop builds on
these guarantees.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

from openai import APIError, APIConnectionError, APITimeoutError, OpenAI, RateLimitError

from .config import Config

MAX_RETRIES = 5          # retries for 429 / 5xx / timeouts / connection errors
BACKOFF_BASE_S = 1.0     # exponential base; full jitter applied on top
BACKOFF_CAP_S = 60.0
API_TIMEOUT_S = 300.0    # single-request timeout (slow reasoning models)


class ContextLengthExceeded(Exception):
    """The request exceeded the model's context window.

    Never retried — the agent loop catches this and triggers compression.
    """


@dataclass
class ToolCall:
    id: str
    name: str
    arguments_json: str


@dataclass
class AssistantMessage:
    """One assistant turn, wrapping the raw SDK message."""

    _raw: dict                  # verbatim dict; replayed into history as-is
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)

    def as_message(self) -> dict:
        """Return the message exactly as received.

        The chat-completions protocol requires the assistant turn (including
        its tool_calls) to be replayed verbatim before tool results follow.
        """
        return self._raw


@dataclass
class UsageStats:
    """Accumulated token usage — feeds the cost-budget termination layer
    and the /cost display."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, usage) -> None:
        self.calls += 1
        if usage is not None:  # some gateways omit usage
            self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0


def _is_context_length_error(exc: APIError) -> bool:
    code = getattr(exc, "code", None) or ""
    body = str(getattr(exc, "body", "") or "")
    text = f"{code} {exc} {body}".lower()
    return "context" in text and ("length" in text or "window" in text or "maximum" in text)


class LLMClient:
    def __init__(self, config: Config):
        self._config = config
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=API_TIMEOUT_S,
        )
        self.usage = UsageStats()

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        stream: bool = False,
    ) -> AssistantMessage:
        """One non-streaming chat-completion round trip, with retries."""
        if stream:
            raise NotImplementedError("streaming arrives with the interactive REPL")
        kwargs: dict = {"model": self._config.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        resp = self._create_with_retry(kwargs)
        self.usage.add(resp.usage)
        msg = resp.choices[0].message
        tool_calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments_json=tc.function.arguments)
            for tc in (msg.tool_calls or [])
        ]
        return AssistantMessage(
            _raw=msg.model_dump(exclude_none=True),
            text=msg.content or "",
            tool_calls=tool_calls,
        )

    def _create_with_retry(self, kwargs: dict):
        delay = BACKOFF_BASE_S
        for attempt in range(MAX_RETRIES + 1):
            try:
                return self._client.chat.completions.create(**kwargs)
            except RateLimitError as exc:
                if attempt == MAX_RETRIES:
                    raise
                self._sleep(self._retry_after(exc) or delay)
            except (APITimeoutError, APIConnectionError):
                if attempt == MAX_RETRIES:
                    raise
                self._sleep(delay)
            except APIError as exc:
                if _is_context_length_error(exc):
                    raise ContextLengthExceeded(str(exc)) from exc
                status = getattr(exc, "status_code", None)
                retryable = status is not None and status >= 500
                if not retryable or attempt == MAX_RETRIES:
                    raise
                self._sleep(delay)
            delay = min(delay * 2, BACKOFF_CAP_S)
        raise RuntimeError("unreachable")  # pragma: no cover

    @staticmethod
    def _retry_after(exc: RateLimitError) -> float | None:
        try:
            value = exc.response.headers.get("Retry-After")
            return float(value) if value is not None else None
        except (AttributeError, ValueError):
            return None

    @staticmethod
    def _sleep(delay: float) -> None:
        time.sleep(random.uniform(0, delay))  # full jitter
