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

EVALUATION_COLUMNS = [
    "model",
    "entry_id",
    "repeat",
    "evaluator",
    "status",
    "passed",
    "score",
    "verdict",
    "judgments",
    "evaluation_input_tokens",
    "evaluation_output_tokens",
    "evaluation_cost_usd",
    "code_tests_passed",
    "code_tests_total",
    "json_errors",
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


def evaluation_row(trial: TrialResult) -> dict:
    evaluation = trial.evaluation
    details = evaluation.details or {}
    row = {
        "model": trial.model,
        "entry_id": trial.entry_id,
        "repeat": trial.repeat,
        "evaluator": evaluation.evaluator,
        "status": evaluation.status,
        "passed": evaluation.passed,
        "score": evaluation.score,
        "verdict": evaluation.verdict,
        "judgments": details.get("judgments") if evaluation.evaluator == "judge" else None,
        "evaluation_input_tokens": evaluation.input_tokens,
        "evaluation_output_tokens": evaluation.output_tokens,
        "evaluation_cost_usd": evaluation.cost_usd,
        "code_tests_passed": (
            details.get("passed") if evaluation.evaluator == "code_tests" else None
        ),
        "code_tests_total": (
            details.get("total") if evaluation.evaluator == "code_tests" else None
        ),
        "json_errors": (
            len(details.get("errors") or [])
            if evaluation.evaluator == "json_schema"
            else None
        ),
        "error_message": evaluation.error_message,
    }
    return row


def write_evaluations_csv(output_path: Path, trials: list[TrialResult]) -> int:
    evaluated = [trial for trial in trials if trial.evaluation is not None]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(EVALUATION_COLUMNS)
        for trial in evaluated:
            row = evaluation_row(trial)
            writer.writerow(
                neutralize_formula(csv_value(row.get(column)))
                for column in EVALUATION_COLUMNS
            )
    return len(evaluated)
