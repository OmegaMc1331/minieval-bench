# Changelog

All notable changes to this project are documented in this file. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - 2026-11-15

Reliability, security and evaluation-integrity corrective release. The full defect analysis is in `docs/V0.3.1_AUDIT.md`.

### Fixed

- Code evaluation verdicts can no longer be forged by evaluated programs: the in-container harness is split into a trusted controller process and a worker process, the host parses only the controller's verdict, reported test identities must match the dataset tests exactly, and zero-executed-test runs cannot produce a perfect score. Twelve adversarial scenarios (forged verdicts, injected tests, report-pipe fd spray, worker kill, malformed output, ...) are covered by real Docker tests.
- Docker output capture is bounded during capture on both the host and inside the container: exceeding `max_output_bytes` kills the producer and yields a distinct `output_overflow` outcome instead of buffering unbounded output; containers are force-removed on every exit path including timeouts, overflows and exceptions. Measured on the development host: 512 MB of container output against a 64 KB cap kept the host working set within 0.7 MB of baseline.
- `latency_ms` now measures the model call only. Trials record `generation_ms`, `evaluation_ms`, `total_ms` and `queue_ms` separately; reports, statistics, exports, the dashboard and comparisons label each duration, and runs produced before 0.3.1 are marked as legacy (mixed generation and evaluation latency) rather than recomputed.
- Judge call admission is atomic under concurrency: `max_calls` can no longer be exceeded by parallel workers, and every evaluation records calls, responses, valid and invalid verdicts, network errors and provider retries. Billed calls with unusable verdicts contribute their tokens and cost, with a known-cost count for coverage.
- `answer_accuracy` covers deterministic metrics only; JSON Schema conformance, code evaluation pass rate, code test totals and judge scores get separate fields with explicit per-evaluator denominators instead of one mixed population.
- `evaluate` verifies question matching before reusing stored answers: identical dataset fingerprint or per-entry prompt/reference fingerprints from the embedded snapshot; changed prompts are conflicts kept unchanged, unverifiable matches are flagged. Reevaluated runs record source-run and reevaluation provenance as separate manifest blocks and never modify the originals.
- `compare` warns about differing judge rubrics or judge models, unknown judge configuration in pre-0.3.1 runs, differing code evaluation settings and mixed latency semantics; two-run comparisons over an identical dataset include per-model paired differences with a bootstrap interval and explicit unit counts.
- Pairwise outcomes distinguish confirmed wins, ties, mixed verdicts (win + tie), disagreements, incomplete, judge errors and unavailable comparisons; both verdicts are kept, provider errors are masked, and judge calls honor `max_calls` with per-judgment token and cost usage.
- JSON `null` responses parse and validate against schemas that allow `null` (parse failures are signalled by the error channel, not a `None` value); rubrics reject duplicate criterion ids.

### Security

- The default code evaluation image is pinned by digest (`python:3.12-slim@sha256:5e2bdb...`), verified against the adversarial suite; the update procedure is documented in the audit. Unavailable Docker still yields `unavailable` outcomes and never falls back to host execution.
- Threat model and residual risks of the code-evaluation sandbox are documented in `docs/V0.3.1_AUDIT.md`: the verdict channel and decision are trusted, while per-test outcomes rely on the worker process integrity (exec-based in-process isolation cannot be made airtight).

### Improved

- Judge budget policy is explicit: `budget_usd` is a stop-on-observed guard announced at run start, with a plan-time refusal when the output-token lower bound alone exceeds it (input-token cost cannot be pre-bounded).
- Statistical analysis reports units, excluded question ids and the conditional nature of the pass-rate interval, with a note that intervals are not significance tests; single-question, identical-value and disjoint-dataset edge cases are covered.
- CSV export adds generation/evaluation/total latency columns and a separate `evaluations.csv` (evaluator type, status, scores, judgments, evaluation tokens and cost, code test totals, JSON error counts) with formula neutralization and empty cells for missing values.
- The HTML dashboard shows separated latencies, a per-model evaluator matrix, judge cost coverage, reevaluation provenance and excluded questions; the comparison report gains paired differences and judge configuration sections; pairwise tallies show mixed verdicts and usage coverage.
- A manual `docker-integration` GitHub Actions workflow runs the sandbox, adversarial and output-limit tests plus a leftover-container check on a dedicated runner; `pytest-cov` joins the dev dependencies (95% line coverage offline).

### Compatibility

- All commands, YAML configuration, datasets, CSV exports and report formats remain usable; every schema change is additive (`generation_ms`, `evaluation_ms`, `total_ms`, per-evaluator statistics fields, manifest `judge`/`source_run`/`reevaluation` blocks, payload `evaluatorsByModel`).
- `latency_ms` changes meaning for new runs (generation only instead of generation plus evaluation); pre-0.3.1 results are displayed as recorded and flagged legacy. `answer_accuracy` no longer includes JSON Schema or code evaluations on mixed datasets (new dedicated fields carry them); the audit documents the previous definitions.
- Reports from 0.1.0, 0.1.1, 0.2.0 and 0.3.0 load, render and export with fixtures covering each release.

### Known Limitations

- The code-evaluation worker's per-test outcomes can in principle be forged by a malicious submission with in-process introspection; the verdict channel and validation remain trusted (see the audit's threat model).
- Per-test timeouts rely on `SIGALRM` inside Linux containers; other kernels fall back to the wall-clock timeout.
- `budget_usd` cannot pre-bound input-token cost: judging stops once the observed spend reaches it, so an in-flight call may overshoot.
- Estimated costs can differ from actual provider billing.

## [0.3.0] - 2026-11-14

### Added

- LLM-as-a-judge evaluation (`metric: judge`): configurable judge through the existing LiteLLM provider stack (or a deterministic mock judge), YAML rubrics with weighted criteria and scales, validated JSON verdicts (missing or out-of-bounds scores become evaluation failures, never zero scores), overall scores computed by MiniEval from normalized weighted criteria, repeated judgments with per-criterion means and standard deviations, and full transparency (judge model, rubric fingerprint and version, judgment counts, failures, judge tokens and cost).
- Judge budgets: planned judge calls are computed and printed before the run; `max_calls` refuses to start when the plan exceeds the cap and stops judging when reached; `budget_usd` requires judge pricing (unknown pricing refuses the run instead of guessing) and stops judging once the accumulated cost crosses it.
- Judge calibration (`calibrate` command): scores a calibration dataset with expected per-criterion scores and reports mean error and disagreements per criterion.
- Blind pairwise comparison (`pairwise` command): two anonymized responses judged twice with positions swapped, verdicts mapped back to the original models, contradictions kept as disagreements, standalone HTML page with side-by-side responses and per-model tallies. No Elo or tournament ranking.
- Structured output evaluation (`metric: json_schema`): JSON Schema Draft 2020-12 validation with a documented extraction policy (direct parse, else first balanced block), detailed `path: message` errors including resolved paths for missing required properties, remote `$ref` URLs rejected at dataset load.
- Code evaluation (`metric: code_tests`): generated Python executed only inside a restricted Docker container (no network, read-only root filesystem, non-root user, all capabilities dropped, no-new-privileges, memory/cpu/pid limits, noexec tmpfs, automatic removal, host-side wall-clock timeout); submission and tests piped over stdin without shell interpolation; per-test alarm timeouts; timeouts, runtime errors, syntax errors and resource-limit kills distinguished; Docker unavailability reported as `unavailable`, never executed on the host and never counted as a model failure.
- Statistical analysis: per-model dispersion (mean, median, standard deviation, min, max, count) and percentile bootstrap confidence intervals over question-level units (repeats averaged first), plus paired per-question score differences in run comparisons; no interval with fewer than two questions and no significance claims.
- `evaluate` command: re-evaluates stored responses with new evaluators without calling the candidate models, writing to a separate directory and leaving originals untouched.
- Evaluation records on every trial (evaluator, status, verdict, score, details, tokens, cost) with evaluator summaries and separated evaluation costs in JSON, Markdown and the HTML report; five new HTML report sections (evaluation overview, judge scores, structured output, code evaluation, statistical analysis) and evaluation details in the trial explorer.
- Offline examples: judge benchmark with rubric and calibration dataset, JSON Schema benchmark, containerized Python code benchmark.

### Improved

- Dataset entries: `reference` is now required only for the metrics that use it; `json_schema` and `tests` fields validated at load (schema validity, test module syntax).
- Docker integration tests (8 scenarios including infinite loop, memory bomb, blocked network and blocked filesystem writes) run only with `MINIEVAL_DOCKER_TESTS=1` and a reachable daemon; the standard suite stays fully offline.

### Security

- Untrusted model responses are framed as data in judge and pairwise prompts and cannot alter the rubric or output format; the residual prompt-injection risk is documented.
- Generated code never executes on the host; no sandbox bypass exists, and Docker-dependent tests are opt-in.
- Judge and pairwise error messages pass through the same secret-masking pipeline as provider errors.

### Compatibility

- All v0.2.0 (and earlier) commands, configuration, datasets, results files, reports, CSV, history and comparisons are unchanged; every schema change is additive with safe defaults, and old results render in the new HTML report with unavailable data shown as `n/a`.
- Subjective judge scores never replace or alter `answer_accuracy` or any existing statistic.

## [0.2.0] - 2026-11-14

### Added

- Self-contained interactive HTML report (`report.html`) generated by every `run` (disable with `--no-html`) or regenerated with `report --format html`. Sections: overview, model performance with selectable indicator, per-model latency histograms with exact bin counts, token and cost charts with coverage, trial explorer, error breakdown and run metadata. No server, no network access, no CDN, vanilla HTML/CSS/JS.
- Trial explorer: per-trial table with search, filters (model, outcome, error type), sorting, pagination and a detail panel showing the full response, sanitized error, and the prompt and reference when the run used the new `run --snapshot` option, which embeds the dataset in `results.json`.
- `compare` command producing a standalone HTML comparison of two or more runs: highlighted condition differences, methodological warnings (changed datasets, model configuration drift, differing seeds, repeats, concurrency, versions), per-model indicator tables with absolute and relative deltas (relative only when defined), grouped bar charts.
- `export` command writing individual trials to CSV (UTF-8, missing values kept empty, spreadsheet formula injection neutralized).
- `history` command listing past runs found under a directory from their manifests, sorted by date.
- Light and dark themes following the system preference with manual toggle, responsive layout, keyboard navigation and text equivalents for charts.

### Improved

- Performance of report generation verified from 10 to 10,000 trials (10,000 trials render in well under a second of Python time into a file of a few hundred KB); the trial explorer paginates to bound DOM size.

### Compatibility

- All v0.1.x commands, YAML configuration, dataset formats and JSON schemas are unchanged; new JSON fields (`dataset` snapshot in `results.json`) are additive and optional. Reports from v0.1.0 and v0.1.1 render in the HTML report with unavailable data shown as `n/a`.
- Frontend resources ship inside the wheel; the HTML report works from an installed package without the source repository.

### Security

- Untrusted content (prompts, responses, error messages, model names) is rendered as text only, never as HTML; the embedded JSON payload escapes `</script` sequences; the report performs no network requests. CSV cells that could be interpreted as formulas are prefixed.

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
