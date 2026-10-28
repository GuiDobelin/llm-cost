import math

import pytest

from llm_cost import estimate_tokens


def test_empty_text_has_no_tokens():
    assert estimate_tokens("") == 0


def test_rounds_up():
    assert estimate_tokens("abcde") == 2
    assert estimate_tokens("abcd") == 1


def test_custom_ratio():
    assert estimate_tokens("a" * 30, chars_per_token=3) == 10


@pytest.mark.parametrize("ratio", [0, -1, math.inf, math.nan])
def test_rejects_invalid_ratio(ratio):
    with pytest.raises(ValueError):
        estimate_tokens("text", chars_per_token=ratio)
