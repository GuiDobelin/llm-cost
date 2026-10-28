import json
from decimal import Decimal

import pytest

from llm_cost import DEFAULT_PRICES, ModelPrice, UnknownModelError, get_price, load_prices


def test_default_prices_are_keyed_by_model_name():
    for name, price in DEFAULT_PRICES.items():
        assert price.model == name
        assert price.input > 0 and price.output > 0


def test_get_price_known_model():
    price = get_price("claude-opus-5-5")
    assert price.input == Decimal("4")
    assert price.output == Decimal("20")
    assert price.effective_cache_read == Decimal("0.20")


def test_cache_prices_default_to_multipliers():
    price = get_price("claude-sonnet-4-6")
    assert price.effective_cache_read == Decimal("0.3")
    assert price.effective_cache_write == Decimal("3.75")


def test_get_price_unknown_model_suggests_close_matches():
    with pytest.raises(UnknownModelError) as exc:
        get_price("claude-opus-55")
    assert "claude-opus-5-5" in exc.value.suggestions
    assert "did you mean" in str(exc.value)


def test_get_price_unknown_model_without_suggestions():
    with pytest.raises(UnknownModelError) as exc:
        get_price("zzz")
    assert str(exc.value) == "unknown model 'zzz'"


def test_load_prices_merges_over_defaults(tmp_path):
    path = tmp_path / "prices.json"
    path.write_text(
        json.dumps(
            {
                "my-model": {"provider": "acme", "input": 0.1, "output": 0.4},
                "claude-opus-5-5": {"provider": "anthropic", "input": 3, "output": 15},
            }
        )
    )
    prices = load_prices(path)
    assert prices["my-model"] == ModelPrice("acme", "my-model", Decimal("0.1"), Decimal("0.4"))
    assert prices["claude-opus-5-5"].input == Decimal("3")
    assert "claude-haiku-5-5" in prices
    # The bundled table itself is not modified.
    assert DEFAULT_PRICES["claude-opus-5-5"].input == Decimal("4")


def test_load_prices_with_explicit_base(tmp_path):
    path = tmp_path / "prices.json"
    path.write_text(json.dumps({"a": {"input": 1, "output": 2}}))
    prices = load_prices(path, base={})
    assert list(prices) == ["a"]
    assert prices["a"].provider == "custom"


def test_load_prices_tiered_entry(tmp_path):
    path = tmp_path / "prices.json"
    entry = {
        "input": 1,
        "output": 2,
        "long_input_threshold": 1000,
        "long_input": 2,
        "long_output": 4,
    }
    path.write_text(json.dumps({"tiered": entry}))
    assert load_prices(path)["tiered"].long_input_threshold == 1000


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("[]", "JSON object"),
        ('{"m": 3}', "expected an object"),
        ('{"m": {"input": 1}}', "missing required field 'output'"),
        ('{"m": {"input": 1, "output": 2, "colour": 1}}', "unknown field"),
        ('{"m": {"input": "abc", "output": 2}}', "must be a number"),
        ('{"m": {"input": true, "output": 2}}', "must be a number"),
        ('{"m": {"input": "NaN", "output": 2}}', "finite"),
        ('{"m": {"input": -1, "output": 2}}', "must not be negative"),
        ('{"m": {"input": 1, "output": 2, "long_input_threshold": 5}}', "set together"),
        (
            '{"m": {"input": 1, "output": 2, "long_input_threshold": 1.5,'
            ' "long_input": 1, "long_output": 1}}',
            "non-negative integer",
        ),
    ],
)
def test_load_prices_rejects_invalid_files(tmp_path, content, message):
    path = tmp_path / "prices.json"
    path.write_text(content)
    with pytest.raises(ValueError, match=message):
        load_prices(path)


def test_model_price_coerces_numbers_to_decimal():
    price = ModelPrice("acme", "m", 1.5, 2, cache_read="0.1")
    assert price.input == Decimal("1.5")
    assert price.output == Decimal("2")
    assert price.cache_read == Decimal("0.1")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"input": -1.0}, "must not be negative"),
        ({"input": float("inf")}, "finite"),
        ({"input": True}, "must be a number"),
        ({"long_input_threshold": -5, "long_input": 1, "long_output": 1}, "non-negative"),
    ],
)
def test_model_price_constructor_validates(kwargs, message):
    args = {"provider": "acme", "model": "m", "input": 1, "output": 2} | kwargs
    with pytest.raises(ValueError, match=message):
        ModelPrice(**args)


@pytest.mark.parametrize("provider", [None, 3, ""])
def test_load_prices_rejects_bad_provider(tmp_path, provider):
    path = tmp_path / "prices.json"
    path.write_text(json.dumps({"m": {"provider": provider, "input": 1, "output": 2}}))
    with pytest.raises(ValueError, match="provider"):
        load_prices(path)
