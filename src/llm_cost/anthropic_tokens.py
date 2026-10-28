"""Exact token counts for Claude models through the Anthropic token counting API."""

from __future__ import annotations

from typing import Any

DEFAULT_MODEL = "claude-opus-5-5"


def count_tokens_anthropic(text: str, model: str = DEFAULT_MODEL, client: Any = None) -> int:
    """Count the input tokens of ``text`` sent as one user message to ``model``.

    Calls ``client.messages.count_tokens`` (free, but it needs network access and
    Anthropic credentials). When ``client`` is omitted an ``anthropic.Anthropic()``
    client is created, which reads credentials from the environment as usual.
    Requires the ``anthropic`` extra: ``pip install "llm-cost[anthropic]"``.
    """
    if client is None:
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise RuntimeError(
                "exact counting needs the anthropic SDK: pip install 'llm-cost[anthropic]'"
            ) from exc
        client = Anthropic()
    response = client.messages.count_tokens(
        model=model,
        messages=[{"role": "user", "content": text}],
    )
    return int(response.input_tokens)
