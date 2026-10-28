"""Cost calculation from token counts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal

from llm_cost.pricing import ModelPrice, get_price

_MILLION = Decimal(1_000_000)


@dataclass(frozen=True)
class CostBreakdown:
    """Cost of one request in USD, split by token type."""

    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    input_cost: Decimal
    output_cost: Decimal
    cache_read_cost: Decimal
    cache_write_cost: Decimal
    long_context: bool

    @property
    def total(self) -> Decimal:
        return self.input_cost + self.output_cost + self.cache_read_cost + self.cache_write_cost

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "input_cost": float(self.input_cost),
            "output_cost": float(self.output_cost),
            "cache_read_cost": float(self.cache_read_cost),
            "cache_write_cost": float(self.cache_write_cost),
            "total": float(self.total),
            "long_context": self.long_context,
        }


def estimate_cost(
    model: str | ModelPrice,
    input_tokens: int,
    output_tokens: int = 0,
    *,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    prices: Mapping[str, ModelPrice] | None = None,
) -> CostBreakdown:
    """Compute the cost of a request.

    ``input_tokens`` is the uncached part of the prompt, as reported by the API's
    ``usage.input_tokens``; cache reads and writes are billed separately.
    """
    for name, value in (
        ("input_tokens", input_tokens),
        ("output_tokens", output_tokens),
        ("cache_read_tokens", cache_read_tokens),
        ("cache_write_tokens", cache_write_tokens),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}")

    price = model if isinstance(model, ModelPrice) else get_price(model, prices)
    in_rate, out_rate = price.input, price.output
    cache_read_rate, cache_write_rate = price.effective_cache_read, price.effective_cache_write

    prompt_tokens = input_tokens + cache_read_tokens + cache_write_tokens
    long_context = (
        price.long_input_threshold is not None and prompt_tokens > price.long_input_threshold
    )
    if long_context:
        assert price.long_input is not None and price.long_output is not None
        scale = price.long_input / price.input if price.input else Decimal(1)
        in_rate, out_rate = price.long_input, price.long_output
        cache_read_rate *= scale
        cache_write_rate *= scale

    return CostBreakdown(
        model=price.model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        input_cost=input_tokens * in_rate / _MILLION,
        output_cost=output_tokens * out_rate / _MILLION,
        cache_read_cost=cache_read_tokens * cache_read_rate / _MILLION,
        cache_write_cost=cache_write_tokens * cache_write_rate / _MILLION,
        long_context=long_context,
    )
