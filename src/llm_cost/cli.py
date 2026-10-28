import argparse
import json
import sys
from collections.abc import Mapping
from decimal import Decimal
from typing import Any

from llm_cost import __version__
from llm_cost.anthropic_tokens import DEFAULT_MODEL, count_tokens_anthropic
from llm_cost.cost import CostBreakdown, estimate_cost
from llm_cost.pricing import (
    DEFAULT_PRICES,
    PRICES_AS_OF,
    ModelPrice,
    UnknownModelError,
    get_price,
    load_prices,
)
from llm_cost.tokens import DEFAULT_CHARS_PER_TOKEN, estimate_tokens


class CLIError(Exception):
    """A user-facing error: printed to stderr, exit code 1."""


def _non_negative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from None
    if number < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return number


def _positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from None
    if not 0 < number < float("inf"):
        raise argparse.ArgumentTypeError("must be a positive number")
    return number


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--prices",
        metavar="FILE",
        help="JSON file with extra or overriding prices (USD per 1M tokens)",
    )
    parser.add_argument("--json", action="store_true", help="print JSON instead of text")


def _add_input(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "text", nargs="?", help="prompt text; use '-' or pipe into stdin to read stdin"
    )
    parser.add_argument("-f", "--file", help="read the prompt from a file")
    parser.add_argument(
        "-i",
        "--input-tokens",
        type=_non_negative_int,
        help="use this input token count instead of counting text",
    )
    parser.add_argument(
        "-o",
        "--output-tokens",
        type=_non_negative_int,
        default=0,
        help="expected output tokens (default: 0)",
    )
    parser.add_argument(
        "--chars-per-token",
        type=_positive_float,
        default=DEFAULT_CHARS_PER_TOKEN,
        help=f"ratio for the offline estimate (default: {DEFAULT_CHARS_PER_TOKEN:g})",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-cost",
        description="Estimate LLM token counts and API cost per provider and model.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    models = sub.add_parser("models", help="list known models and their prices")
    models.add_argument("--provider", help="only show models from this provider")
    _add_common(models)

    estimate = sub.add_parser("estimate", help="estimate the cost of one request")
    _add_input(estimate)
    estimate.add_argument(
        "-m", "--model", default=DEFAULT_MODEL, help=f"model name (default: {DEFAULT_MODEL})"
    )
    estimate.add_argument(
        "--cache-read", type=_non_negative_int, default=0, help="tokens read from the cache"
    )
    estimate.add_argument(
        "--cache-write", type=_non_negative_int, default=0, help="tokens written to the cache"
    )
    estimate.add_argument(
        "--exact",
        action="store_true",
        help="count tokens with the Anthropic API (Claude models; needs the anthropic extra)",
    )
    _add_common(estimate)

    compare = sub.add_parser("compare", help="compare the cost of one request across models")
    _add_input(compare)
    compare.add_argument("--provider", help="only compare models from this provider")
    _add_common(compare)
    return parser


def _load_table(path: str | None) -> Mapping[str, ModelPrice]:
    if path is None:
        return DEFAULT_PRICES
    try:
        return load_prices(path)
    except OSError as exc:
        raise CLIError(f"cannot read price file: {exc}") from exc
    except ValueError as exc:
        raise CLIError(f"invalid price file {path}: {exc}") from exc


def _read_text(args: argparse.Namespace) -> str:
    if args.text is not None and args.file is not None:
        raise CLIError("give either TEXT or --file, not both")
    if args.file is not None:
        try:
            with open(args.file, encoding="utf-8") as fh:
                return fh.read()
        except (OSError, UnicodeDecodeError) as exc:
            raise CLIError(f"cannot read {args.file}: {exc}") from exc
    no_input = "no input: pass TEXT, --file, --input-tokens or pipe text into stdin"
    if args.text == "-" or (args.text is None and not sys.stdin.isatty()):
        try:
            text = sys.stdin.read()
        except UnicodeDecodeError as exc:
            raise CLIError(f"cannot read stdin: {exc}") from exc
        if not text and args.text is None:
            raise CLIError(no_input)
        return text
    if args.text is None:
        raise CLIError(no_input)
    return args.text


def _input_tokens(
    args: argparse.Namespace, price: ModelPrice | None, client: Any = None
) -> tuple[int, str]:
    """Return the input token count and how it was obtained."""
    if args.input_tokens is not None:
        if getattr(args, "exact", False):
            raise CLIError("--exact counts text; it cannot be combined with --input-tokens")
        if args.text is not None or args.file is not None:
            raise CLIError("give either text input or --input-tokens, not both")
        return args.input_tokens, "given"
    text = _read_text(args)
    if getattr(args, "exact", False):
        if price is None or price.provider != "anthropic":
            raise CLIError("--exact only supports Anthropic models")
        try:
            return count_tokens_anthropic(text, price.model, client=client), "exact"
        except RuntimeError as exc:
            raise CLIError(str(exc)) from exc
        except Exception as exc:
            # anthropic.APIError and friends: report them without a traceback.
            raise CLIError(f"token counting failed: {exc}") from exc
    return estimate_tokens(text, args.chars_per_token), "estimated"


def _usd(amount: Decimal) -> str:
    return f"${amount:,.6f}"


def _prices_note(path: str | None) -> str:
    note = f"bundled prices as of {PRICES_AS_OF}"
    return f"{note}, with overrides from {path}" if path else note


def _filter(table: Mapping[str, ModelPrice], provider: str | None) -> list[ModelPrice]:
    rows = [p for p in table.values() if provider is None or p.provider == provider]
    if not rows:
        raise CLIError(f"no models for provider '{provider}'")
    return rows


def _print_table(headers: list[str], rows: list[list[str]], right: set[int]) -> None:
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def fmt(cells: list[str]) -> str:
        parts = [
            c.rjust(w) if i in right else c.ljust(w)
            for i, (c, w) in enumerate(zip(cells, widths, strict=True))
        ]
        return "  ".join(parts).rstrip()

    print(fmt(headers))
    print(fmt(["-" * w for w in widths]))
    for row in rows:
        print(fmt(row))


def _cmd_models(args: argparse.Namespace) -> None:
    rows = sorted(
        _filter(_load_table(args.prices), args.provider), key=lambda p: (p.provider, p.model)
    )
    if args.json:
        data = [
            {
                "provider": p.provider,
                "model": p.model,
                "input": float(p.input),
                "output": float(p.output),
                "cache_read": float(p.effective_cache_read),
                "cache_write": float(p.effective_cache_write),
                "long_input_threshold": p.long_input_threshold,
                "long_input": None if p.long_input is None else float(p.long_input),
                "long_output": None if p.long_output is None else float(p.long_output),
            }
            for p in rows
        ]
        print(json.dumps(data, indent=2))
        return
    _print_table(
        ["provider", "model", "input/1M", "output/1M", "cache read/1M", "cache write/1M"],
        [
            [
                p.provider,
                p.model,
                f"${p.input:,.2f}",
                f"${p.output:,.2f}",
                f"${p.effective_cache_read:,.3f}",
                f"${p.effective_cache_write:,.3f}",
            ]
            for p in rows
        ],
        right={2, 3, 4, 5},
    )
    print(f"\n{_prices_note(args.prices)}.")
    for p in rows:
        if p.long_input_threshold is not None:
            print(
                f"{p.model}: prompts over {p.long_input_threshold:,} tokens cost "
                f"${p.long_input:,.2f} input / ${p.long_output:,.2f} output per 1M."
            )


def _cmd_estimate(args: argparse.Namespace, client: Any = None) -> None:
    table = _load_table(args.prices)
    price = get_price(args.model, table)
    tokens, source = _input_tokens(args, price, client=client)
    cost = estimate_cost(
        price,
        tokens,
        args.output_tokens,
        cache_read_tokens=args.cache_read,
        cache_write_tokens=args.cache_write,
    )
    if args.json:
        print(
            json.dumps(
                cost.to_dict() | {"provider": price.provider, "input_source": source}, indent=2
            )
        )
        return
    _print_estimate(cost, price, source, args.prices)


def _print_estimate(
    cost: CostBreakdown, price: ModelPrice, source: str, prices: str | None
) -> None:
    lines = [
        ("model", f"{cost.model} ({price.provider})"),
        ("input tokens", f"{cost.input_tokens:,} ({source})"),
        ("output tokens", f"{cost.output_tokens:,}"),
    ]
    if cost.cache_read_tokens or cost.cache_write_tokens:
        lines += [
            ("cache read", f"{cost.cache_read_tokens:,}"),
            ("cache write", f"{cost.cache_write_tokens:,}"),
        ]
    lines += [("input cost", _usd(cost.input_cost)), ("output cost", _usd(cost.output_cost))]
    if cost.cache_read_tokens or cost.cache_write_tokens:
        lines += [
            ("cache read cost", _usd(cost.cache_read_cost)),
            ("cache write cost", _usd(cost.cache_write_cost)),
        ]
    lines.append(("total", _usd(cost.total)))
    width = max(len(k) for k, _ in lines)
    for key, value in lines:
        print(f"{key.ljust(width)}  {value}")
    if cost.long_context:
        print(f"long-context pricing applied (prompt > {price.long_input_threshold:,} tokens)")
    print(_prices_note(prices))


def _cmd_compare(args: argparse.Namespace) -> None:
    rows = _filter(_load_table(args.prices), args.provider)
    tokens, source = _input_tokens(args, None)
    costs = sorted(
        ((p, estimate_cost(p, tokens, args.output_tokens)) for p in rows),
        key=lambda pc: (pc[1].total, pc[0].model),
    )
    if args.json:
        data = [c.to_dict() | {"provider": p.provider} for p, c in costs]
        print(json.dumps({"input_source": source, "results": data}, indent=2))
        return
    _print_table(
        ["model", "provider", "input", "output", "total"],
        [
            [p.model, p.provider, _usd(c.input_cost), _usd(c.output_cost), _usd(c.total)]
            for p, c in costs
        ],
        right={2, 3, 4},
    )
    print(
        f"\n{tokens:,} input tokens ({source}), {args.output_tokens:,} output tokens. "
        f"{_prices_note(args.prices)}."
    )


def main(argv: list[str] | None = None, *, client: Any = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        if args.command == "models":
            _cmd_models(args)
        elif args.command == "estimate":
            _cmd_estimate(args, client=client)
        else:
            _cmd_compare(args)
    except (CLIError, UnknownModelError) as exc:
        print(f"llm-cost: error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
