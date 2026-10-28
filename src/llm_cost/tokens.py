"""Offline token estimation."""

from __future__ import annotations

import math

DEFAULT_CHARS_PER_TOKEN = 4.0


def estimate_tokens(text: str, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN) -> int:
    """Estimate the token count of ``text`` without calling any API.

    Uses a characters-per-token ratio (about 4 for English prose). It is a rough
    estimate: code and non-English text usually need more tokens per character.
    Use ``count_tokens_anthropic`` for an exact count for Claude models.
    """
    if not chars_per_token > 0 or math.isinf(chars_per_token):
        raise ValueError("chars_per_token must be a positive finite number")
    if not text:
        return 0
    return math.ceil(len(text) / chars_per_token)
