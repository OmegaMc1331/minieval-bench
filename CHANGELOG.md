# Changelog

All notable changes to this project are documented in this file. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-11-14

### Security

- Provider error messages are sanitized before being stored, exported or displayed: API keys, bearer tokens, passwords, URL credentials and query strings are masked, and messages are truncated to 400 characters. Exception types remain available for statistics.
- Documented that result files contain prompts and responses verbatim and must be reviewed before publication.

### Fixed

- Token and cost totals no longer present partial sums as complete totals: each aggregate now carries the number of requests that reported it, displayed as `value (reported/succeeded)` when partial.
- `KeyboardInterrupt` now cancels pending requests immediately instead of waiting for the whole queue to drain; `run` exits with code 130 and writes no reports.

### Improved

- Distinct statistical indicators with explicit denominators: request success rate, answer accuracy, evaluation coverage and overall pass rate. `score` is retained as an alias of `answer_accuracy`.
- Per-repeat seeds derived deterministically from the global seed (`sha256("<seed>:<repeat>")`), recorded in the manifest; all models receive the same seed for a given entry and repeat when the provider supports it.
- Deterministic execution plan reconstructible from the configuration and the seed, with optional `shuffle` default so no model is systematically served first; each trial records its `plan_index`.
- Request latency, queue delay and total run duration are measured and reported separately.
- Bounded retries for transient provider errors only (408, 409, 429, 500, 502, 503, 504, timeouts, connection errors): at most 2 retries, exponential backoff, `Retry-After` honored up to 30 seconds. Authentication and configuration errors are never retried.

### Compatibility

- CLI commands, YAML configuration and dataset formats of v0.1.0 are unchanged; `shuffle` is a new optional field defaulting to `false`.
- Reports produced by v0.1.0 remain readable; indicators derivable from old fields are computed on load, unavailable data is shown as `n/a`.

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
