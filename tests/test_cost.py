from decimal import Decimal

import pytest

from llm_cost import ModelPrice, UnknownModelError, estimate_cost


def test_input_and_output_cost():
    cost = estimate_cost("claude-opus-5-5", 1_000_000, 500_000)
    assert cost.input_cost == Decimal("4")
    assert cost.output_cost == Decimal("10")
    assert cost.total == Decimal("14")
    assert cost.long_context is False


def test_cache_tokens_are_billed_separately():
    cost = estimate_cost(
        "claude-sonnet-4-6",
        input_tokens=0,
        cache_read_tokens=1_000_000,
        cache_write_tokens=1_000_000,
    )
    assert cost.cache_read_cost == Decimal("0.3")
    assert cost.cache_write_cost == Decimal("3.75")
    assert cost.total == Decimal("4.05")


def test_long_context_tier_applies_above_threshold():
    below = estimate_cost("claude-haiku-5-5", 100_000, 1_000)
    above = estimate_cost("claude-haiku-5-5", 100_001, 1_000)
    assert below.long_context is False
    assert below.input_cost == Decimal("0.01")
    assert above.long_context is True
    assert above.input_cost == Decimal("100001") * Decimal("0.50") / Decimal(1_000_000)
    assert above.output_cost == Decimal("0.0025")


def test_cached_tokens_count_towards_long_context_threshold():
    cost = estimate_cost("claude-haiku-5-5", 10, cache_read_tokens=100_000)
    assert cost.long_context is True
    # Cache reads are scaled by the same factor as the input price (0.50 / 0.10 = 5x).
    assert cost.cache_read_cost == Decimal("100000") * Decimal("0.05") / Decimal(1_000_000)


def test_accepts_model_price_directly():
    price = ModelPrice("acme", "m", Decimal("1"), Decimal("2"))
    cost = estimate_cost(price, 2_000_000, 1_000_000)
    assert cost.model == "m"
    assert cost.total == Decimal("4")


def test_custom_price_table():
    prices = {"m": ModelPrice("acme", "m", Decimal("1"), Decimal("1"))}
    assert estimate_cost("m", 1_000_000, prices=prices).total == Decimal("1")
    with pytest.raises(UnknownModelError):
        estimate_cost("claude-opus-5-5", 1, prices=prices)


@pytest.mark.parametrize("bad", [-1, 1.5, True, "10"])
def test_rejects_invalid_token_counts(bad):
    with pytest.raises(ValueError, match="non-negative integer"):
        estimate_cost("claude-opus-5-5", bad)


def test_to_dict_is_json_friendly():
    data = estimate_cost("claude-opus-5-5", 1000, 1000).to_dict()
    assert data["total"] == pytest.approx(0.024)
    assert data["model"] == "claude-opus-5-5"
    assert isinstance(data["input_cost"], float)
