# llm-cost

**Estimate LLM token counts and API cost per provider and model, from the CLI or Python**

[![CI](https://github.com/GuiDobelin/llm-cost/actions/workflows/ci.yml/badge.svg)](https://github.com/GuiDobelin/llm-cost/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

<!-- TODO: one paragraph on the problem this solves and who it is for. -->

```text
$ llm-cost --help
```

## Install

```bash
pip install "git+https://github.com/GuiDobelin/llm-cost"
```

## Usage

### CLI

```bash
llm-cost ...
```

### Python

```python
import llm_cost
```

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check . && ruff format --check .
```

## Ecosystem

Part of a set of small tools around [citerag](https://github.com/GuiDobelin/citerag), a self-hostable
RAG API that answers questions about your documents with verifiable citations.

## License

[MIT](LICENSE)
