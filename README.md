# llm-cost

**Estimate LLM token counts and API cost per provider and model, from the CLI or Python**

[![CI](https://github.com/GuiDobelin/llm-cost/actions/workflows/ci.yml/badge.svg)](https://github.com/GuiDobelin/llm-cost/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

"How much will this prompt cost?" comes up when you pick a model, size a batch job, or
show users a per-question cost. `llm-cost` gives a fast answer offline: it estimates the
token count, applies the model's prices (including prompt-cache reads and writes and
long-context tiers), and returns an exact `Decimal` breakdown. When you need the exact
count for a Claude model, it calls Anthropic's free token counting endpoint. It has no
runtime dependencies.

```text
$ llm-cost --help
usage: llm-cost [-h] [--version] COMMAND ...

Estimate LLM token counts and API cost per provider and model.

positional arguments:
  COMMAND
    models    list known models and their prices
    estimate  estimate the cost of one request
    compare   compare the cost of one request across models
```

## Install

```bash
pip install "git+https://github.com/GuiDobelin/llm-cost"
# with exact token counting for Claude models:
pip install "llm-cost[anthropic] @ git+https://github.com/GuiDobelin/llm-cost"
```

## Usage

### CLI

Estimate one request from a file (stdin, a `TEXT` argument and `--input-tokens` also work):

```text
$ llm-cost estimate -f prompt.txt -o 400 -m claude-sonnet-5-5
model          claude-sonnet-5-5 (anthropic)
input tokens   38 (estimated)
output tokens  400
input cost     $0.000076
output cost    $0.004000
total          $0.004076
bundled prices as of 2026-10-06
```

Include prompt caching, for example in an agent loop where most of the prompt is cached:

```text
$ llm-cost estimate -m claude-opus-5-5 -i 2000 --cache-read 48000 -o 800
model             claude-opus-5-5 (anthropic)
input tokens      2,000 (given)
output tokens     800
cache read        48,000
cache write       0
input cost        $0.008000
output cost       $0.016000
cache read cost   $0.009600
cache write cost  $0.000000
total             $0.033600
bundled prices as of 2026-10-06
```

Compare the same request across models, cheapest first:

```text
$ llm-cost compare -i 50000 -o 2000 | head -8
model              provider       input     output      total
-----------------  ---------  ---------  ---------  ---------
claude-haiku-5-5   anthropic  $0.005000  $0.001000  $0.006000
claude-haiku-4-5   anthropic  $0.050000  $0.010000  $0.060000
claude-sonnet-5    anthropic  $0.100000  $0.020000  $0.120000
claude-sonnet-5-5  anthropic  $0.100000  $0.020000  $0.120000
claude-sonnet-4-6  anthropic  $0.150000  $0.030000  $0.180000
claude-opus-5-5    anthropic  $0.200000  $0.040000  $0.240000
```

Other commands and flags:

- `llm-cost models` lists every known model with its input, output and cache prices.
- `--exact` (on `estimate`) counts tokens with Anthropic's `count_tokens` API instead of
  the offline estimate. It needs the `anthropic` extra and credentials, as the SDK reads them.
- `--json` prints machine-readable output on every command.
- `--chars-per-token` tunes the offline estimate (default 4, a fair value for English prose).

### Custom prices and other providers

The bundled table holds first-party Anthropic API list prices, with the date they were
last checked. You can add any provider, or override a price, with a JSON file in USD per
million tokens (see [`examples/prices.example.json`](examples/prices.example.json)):

```json
{
  "my-hosted-model": {"provider": "self-hosted", "input": 0.15, "output": 0.60}
}
```

```text
$ llm-cost compare -i 50000 -o 2000 --prices examples/prices.example.json --provider self-hosted
model            provider         input     output      total
---------------  -----------  ---------  ---------  ---------
my-hosted-model  self-hosted  $0.007500  $0.001200  $0.008700
```

The optional fields are `cache_read` and `cache_write` (defaults: 0.1x and 1.25x the input
price), and `long_input_threshold` with `long_input` and `long_output` for tiered pricing.

### Python

```python
from llm_cost import estimate_cost, estimate_tokens

tokens = estimate_tokens("Summarize this quarterly report in three bullet points. " * 200)
cost = estimate_cost("claude-opus-5-5", tokens, output_tokens=500)
print(tokens, cost.total)  # 2800 0.0212
```

`estimate_cost` also takes `cache_read_tokens`, `cache_write_tokens` and a `prices` table
(from `load_prices("prices.json")`). It returns a `CostBreakdown` with per-part costs and
`.to_dict()`. For an exact count with Claude:

```python
from llm_cost.anthropic_tokens import count_tokens_anthropic

count_tokens_anthropic(open("prompt.txt").read(), model="claude-opus-5-5")
```

## Accuracy notes

- The offline estimate is a characters-per-token heuristic. Each provider and model family
  uses its own tokenizer, and code or non-English text usually takes more tokens. Use it
  for budgeting, and use `--exact` (or your provider's usage report) for billing.
- Prices change. Check the date printed with every estimate, and override with `--prices`
  when needed. Partner platforms (Bedrock, Vertex AI) bill Claude at their own rates.
- Cache writes use the 5-minute TTL price. Batch discounts aren't modelled yet.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check . && ruff format --check .
```

Tests run offline. The exact-count path is tested with a fake client.

## Ecosystem

Part of a set of small tools around [citerag](https://github.com/GuiDobelin/citerag), a self-hostable
RAG API that answers questions about your documents with verifiable citations.

## License

[MIT](LICENSE)
