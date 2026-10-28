"""Model prices in USD per million tokens, and loading of custom price files."""

from __future__ import annotations

import difflib
import json
from collections.abc import Mapping
from dataclasses import dataclass, fields
from decimal import Decimal
from pathlib import Path

PRICES_AS_OF = "2026-10-06"
PRICES_SOURCE = "https://www.anthropic.com/pricing#api"

# Default multipliers on the input price when a model has no explicit cache prices.
CACHE_READ_MULTIPLIER = Decimal("0.1")
CACHE_WRITE_MULTIPLIER = Decimal("1.25")  # 5-minute TTL


_DECIMAL_FIELDS = ("input", "output", "cache_read", "cache_write", "long_input", "long_output")


def _to_decimal(model: str, name: str, value: object) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, int | float | str | Decimal):
        raise ValueError(f"{model}: '{name}' must be a number")
    try:
        # str() avoids binary float artefacts (Decimal(0.1) is not exactly 0.1).
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except ArithmeticError as exc:
        raise ValueError(f"{model}: '{name}' must be a number") from exc
    if not result.is_finite():
        raise ValueError(f"{model}: '{name}' must be a finite number")
    if result < 0:
        raise ValueError(f"{model}: '{name}' must not be negative")
    return result


@dataclass(frozen=True)
class ModelPrice:
    """Prices for one model, in USD per million tokens.

    ``long_input_threshold`` models tiered pricing: when a request's prompt has more
    input tokens than the threshold, the ``long_*`` prices apply to the whole request.
    Cache prices in that tier are scaled by the same factor as the input price.

    Numeric fields accept ``int``, ``float``, ``str`` or ``Decimal`` and are stored as
    ``Decimal``.
    """

    provider: str
    model: str
    input: Decimal
    output: Decimal
    cache_read: Decimal | None = None
    cache_write: Decimal | None = None
    long_input_threshold: int | None = None
    long_input: Decimal | None = None
    long_output: Decimal | None = None

    def __post_init__(self) -> None:
        # Accept ints, floats and numeric strings, but store Decimal so cost math is exact.
        for name in _DECIMAL_FIELDS:
            value = getattr(self, name)
            if value is None:
                continue
            object.__setattr__(self, name, _to_decimal(self.model, name, value))
        threshold = self.long_input_threshold
        if threshold is not None and (
            isinstance(threshold, bool) or not isinstance(threshold, int) or threshold < 0
        ):
            raise ValueError(f"{self.model}: long_input_threshold must be a non-negative integer")
        long_fields = (self.long_input_threshold, self.long_input, self.long_output)
        if any(v is not None for v in long_fields) and any(v is None for v in long_fields):
            raise ValueError(
                f"{self.model}: long_input_threshold, long_input and long_output "
                "must be set together"
            )

    @property
    def effective_cache_read(self) -> Decimal:
        if self.cache_read is not None:
            return self.cache_read
        return self.input * CACHE_READ_MULTIPLIER

    @property
    def effective_cache_write(self) -> Decimal:
        if self.cache_write is not None:
            return self.cache_write
        return self.input * CACHE_WRITE_MULTIPLIER


def _p(provider: str, model: str, input: str, output: str, **kw: str | int) -> ModelPrice:
    extra = {k: (v if isinstance(v, int) else Decimal(v)) for k, v in kw.items()}
    return ModelPrice(provider, model, Decimal(input), Decimal(output), **extra)


# First-party Anthropic API list prices. Other providers can be added with a JSON file
# (see ``load_prices``) so the bundled table only holds prices we can keep up to date.
DEFAULT_PRICES: dict[str, ModelPrice] = {
    p.model: p
    for p in [
        _p("anthropic", "claude-fable-5-1", "10", "50", cache_read="0.25"),
        _p("anthropic", "claude-fable-5", "10", "50", cache_read="1"),
        _p("anthropic", "claude-opus-5-5", "4", "20", cache_read="0.20"),
        _p("anthropic", "claude-opus-5", "5", "25"),
        _p("anthropic", "claude-opus-4-8", "5", "25"),
        _p("anthropic", "claude-opus-4-7", "5", "25"),
        _p("anthropic", "claude-opus-4-6", "5", "25"),
        _p("anthropic", "claude-sonnet-5-5", "2", "10", cache_read="0.20"),
        _p("anthropic", "claude-sonnet-5", "2", "10"),
        _p("anthropic", "claude-sonnet-4-6", "3", "15"),
        _p(
            "anthropic",
            "claude-haiku-5-5",
            "0.10",
            "0.50",
            long_input_threshold=100_000,
            long_input="0.50",
            long_output="2.50",
        ),
        _p("anthropic", "claude-haiku-4-5", "1", "5"),
    ]
}


def _parse_entry(model: str, raw: object) -> ModelPrice:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{model}: expected an object with prices")
    allowed = {f.name for f in fields(ModelPrice)} - {"model"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"{model}: unknown field(s) {', '.join(sorted(unknown))}")
    for required in ("input", "output"):
        if required not in raw:
            raise ValueError(f"{model}: missing required field '{required}'")
    provider = raw.get("provider", "custom")
    if not isinstance(provider, str) or not provider:
        raise ValueError(f"{model}: 'provider' must be a non-empty string")
    kwargs = {k: v for k, v in raw.items() if v is not None and k != "provider"}
    return ModelPrice(model=model, provider=provider, **kwargs)


def load_prices(
    path: str | Path, base: Mapping[str, ModelPrice] | None = None
) -> dict[str, ModelPrice]:
    """Load prices from a JSON file and merge them over ``base`` (default: bundled prices).

    The file maps model names to price objects, in USD per million tokens::

        {"gpt-example": {"provider": "openai", "input": 2.5, "output": 10}}
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("price file must contain a JSON object mapping model names to prices")
    prices = dict(DEFAULT_PRICES if base is None else base)
    for model, raw in data.items():
        prices[model] = _parse_entry(model, raw)
    return prices


class UnknownModelError(KeyError):
    """Raised when a model has no known price."""

    def __init__(self, model: str, suggestions: list[str]) -> None:
        self.model = model
        self.suggestions = suggestions
        super().__init__(model)

    def __str__(self) -> str:
        msg = f"unknown model '{self.model}'"
        if self.suggestions:
            msg += f"; did you mean: {', '.join(self.suggestions)}?"
        return msg


def get_price(model: str, prices: Mapping[str, ModelPrice] | None = None) -> ModelPrice:
    """Return the price entry for ``model`` or raise ``UnknownModelError`` with suggestions."""
    table = DEFAULT_PRICES if prices is None else prices
    try:
        return table[model]
    except KeyError:
        suggestions = difflib.get_close_matches(model, list(table), n=3, cutoff=0.5)
        raise UnknownModelError(model, suggestions) from None
