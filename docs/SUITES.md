# Benchmark Suites

MiniEval 0.4.0 organizes benchmarks into **suites**: versioned sets of tasks
executed under one protocol. This document describes the suite format, the
registry, the built-in catalog and how to create your own suite.

## Quick tour

```bash
minieval-bench suite list
minieval-bench suite info mini-general
minieval-bench suite validate mini-general
minieval-bench suite run mini-general --config models.yaml --output results/mini-general
minieval-bench suite run mini-general --config models.yaml --dry-run
minieval-bench suite verify results/mini-general
minieval-bench suite compare --run results/run-a --run results/run-b
minieval-bench suite export --input results/mini-general --output mini-general.zip
```

The models file lists the candidates and reuses the standard model
configuration:

```yaml
models:
  - name: gpt-4o-mini
    provider: litellm
    model: openai/gpt-4o-mini
  - name: local-mock
    provider: mock
```

## Registry

Suites come from two places:

1. built-in suites packaged inside MiniEval (`builtin:…` origin);
2. user directories listed in the `MINIEVAL_SUITE_PATH` environment variable
   (`os.pathsep`-separated) or passed with `--path`.

Discovery only reads `suite.yaml` (and the datasets for `info`/`validate`);
nothing in a suite is ever executed to discover, validate or plan it. A suite
id defined in several locations is an error, reported explicitly.

## Suite format

A suite is a directory:

```text
my-suite/
├── suite.yaml
└── datasets/
    └── questions.jsonl
```

`suite.yaml`:

```yaml
id: my-suite                  # lowercase letters, digits, hyphens
version: 1.0.0                # semantic version X.Y.Z
name: My Suite
description: What it measures, and what it does not.
license: MIT                  # SPDX identifier or custom text
source: https://example.com/my-suite   # optional provenance
author: Example Org                    # optional

defaults:                     # protocol parameters (optional fields)
  repeats: 3
  seed: 42
  concurrency: 2
  timeout: 60

statistics:                   # optional, engine defaults if omitted
  bootstrap_iterations: 1000
  confidence_level: 0.95

evaluators:                   # optional; needed for judge / code_tests entries
  judge:
    provider: litellm
    model: openai/gpt-4o-mini
    rubric: rubrics/general.yaml
  code:
    runtime: docker

tasks:
  - id: core-questions        # unique per suite
    dataset: datasets/questions.jsonl   # relative .jsonl path, inside the suite dir
    weight: 2.0               # positive; used by macro aggregation
    expected_metric: exact_match        # optional cross-check of the dataset
```

Validation enforces: unique task ids, semver versions, relative `.jsonl`
dataset references that stay inside the suite directory (no absolute paths,
no `..`, no symlink escape), unique entry ids, metric-specific requirements
(regex compilation, JSON Schema validity, test-module syntax), expected-metric
consistency, evaluator presence for `judge`/`code_tests` entries, and rubric
loadability. Datasets use the standard JSONL entry format (see the README),
with the optional `category` field used by stratified sampling.

## Parameter inheritance

Priority order:

1. suite protocol (datasets, weights, metrics, evaluators) — never silently
   overridable;
2. suite `defaults`;
3. user CLI overrides (`--repeats`, `--seed`, `--concurrency`, `--timeout`,
   `--max-tokens`, `--temperature`, `--shuffle/--no-shuffle`, sampling
   options);
4. engine defaults.

Every user override is recorded in the suite manifest and in
`benchmark.lock.json`, and the run is flagged `protocol_modified`: it is never
presented as the official suite protocol.

## Run structure

```text
results/my-suite/
├── suite-manifest.json       # run metadata, statuses, protocol flags
├── benchmark.lock.json       # deterministic protocol fingerprint
├── suite-results.json        # per-task statuses and stats
├── suite-report.md           # global markdown report
├── suite-report.html         # global standalone report
└── tasks/
    └── core-questions/       # one engine run per task (standard format)
        ├── manifest.json
        ├── results.json
        ├── report.md
        └── report.html
```

Task statuses: `completed`, `partially_completed`, `unavailable` (missing
dataset, evaluator or dependency — never scored as zero),
`infrastructure_failed`, `not_executed` (after an interruption). A suite with
any non-completed task is not reported as fully validated.

`--resume` reuses a stored task result only when its section of the stored
lockfile matches the recomputed one; anything else is re-executed. Task
directories are written atomically (staging directory then rename).

## Aggregation

Scores are aggregated per **metric family** only:
`deterministic` (exact_match/contains/regex), `json_schema`, `code_tests`,
`judge`. Families are never merged into a global score.

- **macro**: weighted mean of per-task rates (task `weight`, normalized);
- **micro**: pooled successes over pooled trials.

Each aggregate shows the number of trials and how many tasks of the family it
covers; unavailable tasks contribute no observations, not zeros. Judge means
depend on the rubric scale and are only comparable across runs with the same
rubric fingerprint.

## Built-in catalog

| Suite | Version | Tasks | Entries | Metrics | Notes |
| --- | --- | --- | --- | --- | --- |
| mini-general | 1.0.0 | 5 | 58 | exact_match, regex | arithmetic, logic, strings, stable facts, formats |
| mini-instruction | 1.0.0 | 5 | 34 | exact_match, regex | formats, extraction, transformations, classification, syntax |
| mini-structured | 1.0.0 | 3 | 17 | json_schema | objects, arrays, nested constraints |
| mini-code | 0.1.0 | 1 | 6 | code_tests | experimental; requires `--allow-code-execution` and Docker |

All questions are original and their answers were computed or hand-verified.
These datasets exercise the protocol with objectively checkable answers; they
are **not** statistically representative evaluations of model quality, and the
suite descriptions state this scope. `mini-code` keeps every v0.3.1 code
safety guarantee: pinned image digest, restricted runtime, explicit
activation; see the V0.3.1 audit for the residual tampering risk inside the
worker process.

## Creating a suite

1. Create a directory with a unique kebab-case id.
2. Author datasets as JSONL (see `dataset import` for converting CSV/JSON).
3. Write `suite.yaml`, referencing the datasets with relative paths.
4. Validate: `minieval-bench suite validate my-suite --path <parent-dir>`.
5. Dry-run with your models file, then run.

Publishing a suite requires a license and, for redistributed data, verified
redistribution rights; MiniEval never invents a license and refuses to assume
one (see [DATASETS.md](DATASETS.md)).
