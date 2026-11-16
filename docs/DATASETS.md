# Datasets

MiniEval datasets are JSONL files: one JSON object per line, validated by the
engine before any run. This document covers the entry format, the import
commands and the rules for external (Hugging Face) sources.

## Entry format

```json
{"id": "q-123", "prompt": "What is 2 + 2? Reply with the number only.",
 "metric": "exact_match", "reference": "4", "system": null,
 "category": "arithmetic"}
```

Fields: `id` (unique), `prompt`, `metric` (`exact_match`, `contains`,
`regex`, `json_schema`, `code_tests`, `judge`), `reference` (required for the
deterministic metrics), `system` (optional), `json_schema` (for
`json_schema`), `tests` (Python module for `code_tests`), `category`
(optional, used by stratified sampling). UTF-8 encoding is enforced.

## Commands

```bash
minieval-bench dataset list
minieval-bench dataset validate datasets/custom.jsonl
minieval-bench dataset import --source ./data/questions.csv --output datasets/questions.jsonl
minieval-bench dataset import-hf --dataset org/name --revision <40-char-sha> \
  --file data/train.jsonl --output datasets/train.jsonl --license apache-2.0
```

## Local import

CSV sources need the columns `id`, `prompt`, `metric` plus the optional
`reference`, `system`, `category` columns. CSV supports the deterministic
metrics and `judge` only: `json_schema` and `code_tests` entries must be
authored as JSONL directly (the error message says so). JSON sources are
arrays of entry objects.

Transformations are deterministic and recorded in a provenance sidecar
(`<output>.provenance.json`): empty CSV cells become null, fields are
re-serialized with sorted keys, row order is kept. The importer fails on the
first invalid row with its position — nothing is skipped silently, references
are never invented, and an open-ended task is never converted to exact match
(the metric column is mandatory). Encoding must be UTF-8; duplicates, missing
references and invalid metrics abort the import.

Archives (`.zip`, `.tar`, …) and unknown formats are refused explicitly:
extract manually and import the JSONL/CSV file.

## Hugging Face import

`dataset import-hf` downloads exactly one file of a public dataset:

- `--revision` must be a full 40-character commit sha. Branch names such as
  `main` are rejected: they move, and a moving reference is not reproducible.
- `--file` must be a `.jsonl` or `.csv` inside the repository; parquet is
  refused (it would require an extra dependency), archives are refused.
- Downloads are capped (`--max-bytes`, 64 MiB by default).
- Nothing is ever executed from the remote source, no script is run, no
  dependency is installed, and no archive is extracted.

Provenance recorded next to the output: dataset id, exact revision and file,
resolve URL, SHA-256 of the downloaded bytes and of the output, import date,
transformations and MiniEval version.

The license is read from the dataset card at the pinned revision when
present; `--license` records a license you verified yourself. When neither is
available the import succeeds but reports `license: unknown` with a warning:
verify redistribution rights before sharing the content. MiniEval never
invents a license.

Downloads are cached under `~/.cache/minieval-bench/hf/<dataset>/<revision>/`
(override with `MINIEVAL_CACHE_DIR`) and reused offline, including the
license. `--refresh-cache` forces a new download.

## Redistributing data

Before committing or shipping a dataset:

- verify the licence permits redistribution, and its attribution obligations;
- record the source, revision and license in the provenance or suite
  metadata;
- if rights cannot be established, keep the content out of the repository
  and document the import steps so users can fetch it themselves where
  allowed.

External datasets may contain personal or confidential information. Review
the content before publishing results: MiniEval reports embed prompts and
responses verbatim when responses are included.

## Built-in suite datasets

The datasets of the built-in suites are original, MIT-licensed and validated
with the same rules; see [SUITES.md](SUITES.md) for their catalog and scope.
