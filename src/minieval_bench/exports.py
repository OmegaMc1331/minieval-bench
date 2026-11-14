from __future__ import annotations

import csv
from pathlib import Path

from minieval_bench.models import TrialResult

CSV_COLUMNS = [
    "model",
    "entry_id",
    "repeat",
    "ok",
    "passed",
    "latency_ms",
    "generation_ms",
    "evaluation_ms",
    "total_ms",
    "queue_ms",
    "input_tokens",
    "output_tokens",
    "cost_usd",
    "error_type",
    "error_message",
]

FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def csv_value(value) -> str:
    if value is None:
        return ""
    if value is True:
        return "true"
    if value is False:
        return "false"
    return str(value)


def neutralize_formula(value: str) -> str:
    if value.startswith(FORMULA_PREFIXES):
        return "'" + value
    return value


def write_trials_csv(output_path: Path, trials: list[TrialResult]) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for trial in trials:
            writer.writerow(
                neutralize_formula(csv_value(getattr(trial, column)))
                for column in CSV_COLUMNS
            )
    return len(trials)
