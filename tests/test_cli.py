import io
import json
from types import SimpleNamespace

import pytest

from llm_cost import __version__
from llm_cost.cli import main


class FakeMessages:
    def __init__(self, tokens):
        self.tokens = tokens
        self.calls = []

    def count_tokens(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(input_tokens=self.tokens)


class FakeClient:
    def __init__(self, tokens=1234):
        self.messages = FakeMessages(tokens)


@pytest.fixture
def tty_stdin(monkeypatch):
    """Simulate an interactive terminal with nothing piped in."""
    fake = io.StringIO("")
    fake.isatty = lambda: True
    monkeypatch.setattr("sys.stdin", fake)


def run(capsys, *argv, **kwargs):
    code = main(list(argv), **kwargs)
    out, err = capsys.readouterr()
    return code, out, err


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_no_command_prints_help(capsys):
    code, out, _ = run(capsys)
    assert code == 0
    assert "estimate" in out and "compare" in out


def test_models_lists_bundled_prices(capsys):
    code, out, _ = run(capsys, "models")
    assert code == 0
    assert "claude-opus-5-5" in out
    assert "$4.00" in out


def test_models_json_and_provider_filter(capsys, tmp_path):
    prices = tmp_path / "p.json"
    prices.write_text(json.dumps({"x-1": {"provider": "acme", "input": 1, "output": 2}}))
    code, out, _ = run(capsys, "models", "--prices", str(prices), "--provider", "acme", "--json")
    assert code == 0
    assert json.loads(out) == [
        {
            "provider": "acme",
            "model": "x-1",
            "input": 1.0,
            "output": 2.0,
            "cache_read": 0.1,
            "cache_write": 1.25,
            "long_input_threshold": None,
            "long_input": None,
            "long_output": None,
        }
    ]


def test_models_unknown_provider(capsys):
    code, _, err = run(capsys, "models", "--provider", "nobody")
    assert code == 1
    assert "no models for provider 'nobody'" in err


def test_estimate_from_text(capsys, tty_stdin):
    code, out, _ = run(capsys, "estimate", "a" * 400, "-o", "100", "--json")
    data = json.loads(out)
    assert code == 0
    assert data["input_tokens"] == 100
    assert data["input_source"] == "estimated"
    assert data["total"] == pytest.approx(100 * 4e-6 + 100 * 20e-6)


def test_estimate_text_output(capsys, tty_stdin):
    code, out, _ = run(capsys, "estimate", "-i", "1000", "-o", "1000", "-m", "claude-sonnet-5-5")
    assert code == 0
    assert "claude-sonnet-5-5 (anthropic)" in out
    assert "1,000 (given)" in out
    assert "$0.012000" in out
    assert "cache read" not in out


def test_estimate_with_cache_and_long_context(capsys, tty_stdin):
    code, out, _ = run(
        capsys, "estimate", "-m", "claude-haiku-5-5", "-i", "10", "--cache-read", "200000"
    )
    assert code == 0
    assert "cache read cost" in out
    assert "long-context pricing applied (prompt > 100,000 tokens)" in out


def test_estimate_from_file(capsys, tmp_path, tty_stdin):
    path = tmp_path / "prompt.txt"
    path.write_text("b" * 40, encoding="utf-8")
    code, out, _ = run(capsys, "estimate", "-f", str(path), "--json")
    assert code == 0
    assert json.loads(out)["input_tokens"] == 10


def test_estimate_from_stdin(capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("c" * 8))
    code, out, _ = run(capsys, "estimate", "--json")
    assert code == 0
    assert json.loads(out)["input_tokens"] == 2


def test_estimate_dash_reads_stdin(capsys, monkeypatch):
    fake = io.StringIO("d" * 12)
    fake.isatty = lambda: True
    monkeypatch.setattr("sys.stdin", fake)
    code, out, _ = run(capsys, "estimate", "-", "--chars-per-token", "3", "--json")
    assert code == 0
    assert json.loads(out)["input_tokens"] == 4


def test_estimate_exact_uses_client(capsys, tty_stdin):
    client = FakeClient(tokens=42)
    code, out, _ = run(capsys, "estimate", "hello", "--exact", "--json", client=client)
    assert code == 0
    data = json.loads(out)
    assert data["input_tokens"] == 42
    assert data["input_source"] == "exact"
    assert client.messages.calls == [
        {"model": "claude-opus-5-5", "messages": [{"role": "user", "content": "hello"}]}
    ]


def test_estimate_exact_without_sdk(capsys, monkeypatch, tty_stdin):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "anthropic":
            raise ImportError("no anthropic")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    code, _, err = run(capsys, "estimate", "hello", "--exact")
    assert code == 1
    assert "llm-cost[anthropic]" in err


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["estimate"], "no input"),
        (["estimate", "x", "-f", "y"], "not both"),
        (["estimate", "x", "-i", "5"], "not both"),
        (["estimate", "-i", "5", "--exact"], "--exact"),
        (["estimate", "-f", "/nonexistent/file.txt"], "cannot read"),
        (["estimate", "-i", "5", "-m", "claude-opus-55"], "did you mean"),
        (["estimate", "-i", "5", "--prices", "/nonexistent.json"], "cannot read price file"),
    ],
)
def test_estimate_errors(capsys, tty_stdin, argv, message):
    code, _, err = run(capsys, *argv)
    assert code == 1
    assert message in err


def test_invalid_price_file(capsys, tmp_path, tty_stdin):
    path = tmp_path / "bad.json"
    path.write_text('{"m": {"input": 1}}')
    code, _, err = run(capsys, "models", "--prices", str(path))
    assert code == 1
    assert "invalid price file" in err


@pytest.mark.parametrize("bad", ["-1", "abc"])
def test_rejects_bad_token_arguments(capsys, bad):
    with pytest.raises(SystemExit) as exc:
        main(["estimate", "-i", bad])
    assert exc.value.code == 2


def test_compare_sorts_by_total(capsys, tty_stdin):
    code, out, _ = run(capsys, "compare", "-i", "10000", "-o", "1000", "--json")
    assert code == 0
    data = json.loads(out)
    totals = [r["total"] for r in data["results"]]
    assert totals == sorted(totals)
    assert data["results"][0]["model"] == "claude-haiku-5-5"
    assert data["input_source"] == "given"


def test_compare_text_output(capsys, tty_stdin):
    code, out, _ = run(capsys, "compare", "-i", "20000", "-o", "1000", "--provider", "anthropic")
    assert code == 0
    lines = out.splitlines()
    assert lines[0].split() == ["model", "provider", "input", "output", "total"]
    assert lines[2].startswith("claude-haiku-5-5")
    assert "20,000 input tokens (given)" in out


def test_models_shows_long_context_tier(capsys):
    code, out, _ = run(capsys, "models")
    assert code == 0
    assert "claude-haiku-5-5: prompts over 100,000 tokens cost $0.50 input / $2.50 output" in out


def test_empty_stdin_is_an_error(capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    code, _, err = run(capsys, "estimate")
    assert code == 1
    assert "no input" in err


def test_undecodable_stdin_is_an_error(capsys, monkeypatch):
    raw = io.TextIOWrapper(io.BytesIO(b"\xff\xfe"), encoding="utf-8")
    monkeypatch.setattr("sys.stdin", raw)
    code, _, err = run(capsys, "estimate")
    assert code == 1
    assert "cannot read stdin" in err


def test_exact_rejects_non_anthropic_models(capsys, tmp_path, tty_stdin):
    prices = tmp_path / "p.json"
    prices.write_text(json.dumps({"x-1": {"provider": "acme", "input": 1, "output": 2}}))
    client = FakeClient()
    code, _, err = run(
        capsys, "estimate", "hi", "--exact", "-m", "x-1", "--prices", str(prices), client=client
    )
    assert code == 1
    assert "only supports Anthropic models" in err
    assert client.messages.calls == []


def test_exact_reports_api_errors(capsys, tty_stdin):
    class FailingMessages:
        def count_tokens(self, **kwargs):
            raise ConnectionError("network unreachable")

    client = SimpleNamespace(messages=FailingMessages())
    code, _, err = run(capsys, "estimate", "hi", "--exact", client=client)
    assert code == 1
    assert "token counting failed: network unreachable" in err
