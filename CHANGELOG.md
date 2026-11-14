# Changelog

All notable changes to this project are documented in this file. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-11-14

### Added

- YAML benchmark configuration with per-model overrides of shared defaults.
- Multi-provider model support through LiteLLM (OpenAI, Anthropic, Ollama and other supported backends).
- Deterministic mock provider that runs benchmarks without API keys or network access.
- JSONL dataset support with `id`, `prompt`, `metric`, `reference` and optional `system` fields, validated before execution.
- Correction metrics: `exact_match`, `contains` and `regex`, with documented normalization rules.
- Benchmark engine with repeated trials, bounded concurrency, configurable seed, per-request latency, token and cost capture, and per-trial error recording.
- Per-model statistics: request counts, success rate, score over evaluable responses, median and P95 latency, token totals, estimated cost and error breakdown, with unavailable values distinct from zero.
- CLI (`minieval-bench`) with `validate`, `run` and `report` commands and Rich table output.
- JSON and Markdown reports plus a run manifest (effective parameters, models, version, UTC date, seed, dataset SHA-256 fingerprint) containing no secrets.
- Test suite running fully offline and Ruff lint configuration.
- GitHub Actions workflow running tests and lint on Ubuntu and Windows.
