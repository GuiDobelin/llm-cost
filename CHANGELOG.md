# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-10-28

### Added

- Bundled first-party Anthropic API prices (as of 2026-10-06), including prompt-cache and
  long-context tier pricing.
- JSON price files to add other providers or override prices (`load_prices`, `--prices`).
- `estimate_cost()` with an exact `Decimal` breakdown, and `estimate_tokens()` for offline estimates.
- Exact token counts for Claude models with `count_tokens_anthropic()` (optional `anthropic` extra).
- CLI: `llm-cost models`, `llm-cost estimate` and `llm-cost compare`, all with `--json`.
