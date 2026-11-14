# MiniEval

MiniEval is a lightweight Python framework for benchmarking and comparing language models. It runs the same dataset against several models through LiteLLM, scores responses with deterministic metrics, and produces reproducible JSON and Markdown reports. A deterministic mock provider is included, so a full benchmark can run without any API key.

## Features

- Multi-model benchmarks configured from a single YAML file
- LiteLLM providers: OpenAI, Anthropic, Ollama and other supported backends
- Deterministic mock provider for offline runs and testing
- JSONL datasets with per-entry metrics: `exact_match`, `contains`, `regex`
- Repeated trials, configurable concurrency and seed
- Per-request latency, token usage and cost capture
- Median and P95 latency, success rate, score and error breakdown per model
- JSON and Markdown reports with a run manifest (version, UTC date, seed, dataset fingerprint); unavailable values are distinguished from zero
- CLI with Rich table output: `validate`, `run`, `report`

## Requirements

- Python 3.12 or newer
- Windows, Linux or macOS
- An API key in the environment only when calling real providers (for example `OPENAI_API_KEY`); the mock provider needs nothing

## Installation

From source with uv:

```bash
git clone https://github.com/OmegaMc1331/minieval-bench.git
cd minieval-bench
uv sync
uv run minieval-bench --help
```

As a standalone tool installed from GitHub:

```bash
uv tool install git+https://github.com/OmegaMc1331/minieval-bench
```

With pip:

```bash
pip install git+https://github.com/OmegaMc1331/minieval-bench
```

## Quick start without an API key

The repository ships a benchmark that uses only the mock provider:

```bash
uv run minieval-bench validate --config benchmarks/example.yaml
uv run minieval-bench run --config benchmarks/example.yaml --output results/
uv run minieval-bench report --input results/
```

Example output of `run` on the bundled benchmark (seed 42, 6 entries, 3 repeats, 2 models):

```text
                                                    Benchmark results
┌─────────────┬──────────┬────┬────────┬─────────┬───────┬─────────────┬──────────┬───────────┬────────────┬────────────┐
│ Model       │ Requests │ OK │ Failed │ Success │ Score │ Median (ms) │ P95 (ms) │ In tokens │ Out tokens │ Cost (USD) │
├─────────────┼──────────┼────┼────────┼─────────┼───────┼─────────────┼──────────┼───────────┼────────────┼────────────┤
│ mock-free   │       18 │ 18 │      0 │    100% │ 0.500 │        29.0 │     37.7 │        96 │         72 │        n/a │
│ mock-priced │       18 │ 18 │      0 │    100% │ 0.500 │        27.5 │     45.0 │        96 │         72 │   0.000156 │
└─────────────┴──────────┴────┴────────┴─────────┴───────┴─────────────┴──────────┴───────────┴────────────┴────────────┘
```

The score of 0.500 comes from the dataset itself: three entries match the mock response format (`mock-` plus eight hex characters) and three deliberately do not.

## Configuration

A benchmark is described in one YAML file:

```yaml
benchmark: my-benchmark
dataset: datasets/my-dataset.jsonl

defaults:
  repeats: 3
  concurrency: 4
  seed: 42
  timeout: 60
  max_tokens: 512
  temperature: 0.2

models:
  - name: gpt-4o-mini
    model: openai/gpt-4o-mini
  - name: claude-haiku
    model: anthropic/claude-3-5-haiku-latest
    temperature: 0.0
  - name: local-llama
    model: ollama/llama3.1
    timeout: 120
  - name: mock-free
    provider: mock
```

Dataset paths are resolved relative to the config file. Every field under `defaults` applies to all models and can be overridden per model.

### Model fields

| Field | Description |
| --- | --- |
| `name` | Display name used in reports; must be unique |
| `provider` | `litellm` (default) or `mock` |
| `model` | LiteLLM model string, for example `openai/gpt-4o-mini`; defaults to `name` for mock models |
| `temperature` | Sampling temperature, forwarded when set |
| `max_tokens` | Maximum completion tokens |
| `timeout` | Request timeout in seconds |
| `drop_params` | Have LiteLLM drop parameters the provider rejects instead of failing the request |
| `input_cost_per_1m`, `output_cost_per_1m` | Optional USD prices per one million tokens, used for cost estimation |

API keys are never stored in the configuration or written to reports. LiteLLM reads them from the environment (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, ...).

A parameter a provider rejects (for example `temperature` on some reasoning models) fails the affected requests and shows up in the error breakdown; it is never silently ignored. Set `drop_params: true` on the model to have LiteLLM drop unsupported parameters instead.

### Defaults fields

| Field | Default | Description |
| --- | --- |
| `repeats` | 1 | Number of times each entry is sent to each model |
| `concurrency` | 4 | Maximum simultaneous requests across the whole run |
| `seed` | 0 | Seed forwarded to providers and used by the mock provider |
| `timeout` | 60 | Request timeout in seconds |
| `max_tokens` | none | Default maximum completion tokens |
| `temperature` | none | Default sampling temperature |

## Dataset format

Datasets are JSONL files, one JSON object per line:

```jsonl
{"id": "capital-fr", "prompt": "What is the capital of France? Answer with the city name only.", "metric": "exact_match", "reference": "Paris"}
{"id": "vowel-word", "prompt": "Name a word containing all five vowels.", "metric": "contains", "reference": "europe"}
{"id": "hex-id", "prompt": "Give an identifier in hex format.", "metric": "regex", "reference": "^[0-9a-f]{8}$", "system": "Answer as briefly as possible."}
```

| Field | Required | Description |
| --- | --- | --- |
| `id` | yes | Unique entry identifier |
| `prompt` | yes | User prompt |
| `metric` | yes | One of `exact_match`, `contains`, `regex` |
| `reference` | yes | Expected answer, contained text, or regular expression |
| `system` | no | System prompt for this entry |

Validation rules: ids must be unique, regex references must compile, unknown fields are rejected. Errors are reported with the file name and line number.

## Metrics

- `exact_match`: the response equals the reference after normalization.
- `contains`: the reference appears in the response after normalization.
- `regex`: the reference is a regular expression matched with `re.search` against the raw response.

Normalization lowercases the text, strips surrounding whitespace, removes characters that are neither word characters nor whitespace, and collapses internal whitespace. `regex` is matched against the unmodified response.

Limits: these metrics are objective but narrow. They suit prompts with a single verifiable answer or a required output format. Open-ended generation has no reference to compare against, and MiniEval deliberately does not invent a correctness score for it; leave such prompts out of scored datasets or express the expected format as a `regex`.

## Commands

```bash
minieval-bench validate --config benchmarks/example.yaml
minieval-bench run --config benchmarks/example.yaml --output results/
minieval-bench report --input results/
minieval-bench --version
```

`run` exits with code 1 when every request failed, so it can gate CI jobs.

## Reports

`run` writes three files:

- `manifest.json`: effective parameters, models, MiniEval version, UTC date, seed, dataset path and SHA-256 fingerprint
- `results.json`: manifest, per-model statistics and every individual trial
- `report.md`: the same statistics as Markdown

Example `report.md`:

```markdown
# MiniEval report: mock-demo

- MiniEval version: 0.1.0
- Created (UTC): 2026-11-14T00:12:00.925692+00:00
- Seed: 42
- Repeats: 3
- Concurrency: 4
- Dataset: `datasets/example.jsonl` (6 entries, sha256 `3ee24699...`)

## Results

| Model | Requests | OK | Failed | Score | Median (ms) | P95 (ms) | In tokens | Out tokens | Cost (USD) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mock-free | 18 | 18 | 0 | 0.500 | 29.0 | 37.7 | 96 | 72 | n/a |
| mock-priced | 18 | 18 | 0 | 0.500 | 27.5 | 45.0 | 96 | 72 | 0.000156 |

Score is the pass rate over evaluable responses (successful requests).
`n/a` marks a value that is unavailable, as opposed to a real zero.
```

Statistics per model: request count, succeeded and failed requests, success rate, score over evaluable responses, median and P95 latency over successful requests, input and output tokens, estimated cost, and error counts by type. `n/a` (JSON `null`) means unavailable: no successful request, no token counts returned, or unknown pricing. It never means zero.

Cost is computed from `input_cost_per_1m` and `output_cost_per_1m` when both are set; otherwise LiteLLM's bundled price map is consulted. When neither source knows the price, the cost is `n/a`.

## Development

```bash
uv sync
uv run ruff check .
uv run pytest
```

The test suite runs fully offline using the mock provider.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT, see [LICENSE](LICENSE).
