from __future__ import annotations

import random
import statistics

from minieval_bench.models import StatisticsConfig, TrialResult

MINIMUM_UNITS = 2


def numeric_summary(values: list[float]) -> dict:
    if not values:
        return {
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "count": 0,
        }
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "std": statistics.stdev(values) if len(values) >= MINIMUM_UNITS else None,
        "min": min(values),
        "max": max(values),
        "count": len(values),
    }


def bootstrap_interval(
    unit_values: list[float],
    iterations: int,
    confidence_level: float,
    seed: int,
) -> dict:
    summary = {"low": None, "high": None, "method": "percentile bootstrap over question means"}
    if len(unit_values) < MINIMUM_UNITS:
        return summary
    rng = random.Random(seed)
    count = len(unit_values)
    means = []
    for _ in range(iterations):
        sample = [unit_values[rng.randrange(count)] for _ in range(count)]
        means.append(statistics.mean(sample))
    means.sort()
    alpha = (1 - confidence_level) / 2
    low_index = max(0, int(alpha * iterations) - 1)
    high_index = min(iterations - 1, int((1 - alpha) * iterations))
    summary["low"] = means[low_index]
    summary["high"] = means[high_index]
    return summary


def question_pass_rates(trials: list[TrialResult]) -> list[float]:
    by_question: dict[str, list[int]] = {}
    for trial in trials:
        if not trial.ok or trial.passed is None:
            continue
        by_question.setdefault(trial.entry_id, []).append(1 if trial.passed else 0)
    return [statistics.mean(values) for values in by_question.values()]


def question_latencies(trials: list[TrialResult]) -> list[float]:
    by_question: dict[str, list[float]] = {}
    for trial in trials:
        if not trial.ok:
            continue
        latency = trial.generation_ms
        if latency is None:
            latency = trial.latency_ms
        by_question.setdefault(trial.entry_id, []).append(latency)
    return [statistics.mean(values) for values in by_question.values()]


def question_judge_scores(trials: list[TrialResult]) -> list[float]:
    by_question: dict[str, list[float]] = {}
    for trial in trials:
        if not trial.ok or trial.evaluation is None:
            continue
        if trial.evaluation.evaluator != "judge" or trial.evaluation.score is None:
            continue
        by_question.setdefault(trial.entry_id, []).append(trial.evaluation.score)
    return [statistics.mean(values) for values in by_question.values()]


def analyze_model_statistics(
    trials: list[TrialResult], config: StatisticsConfig, seed: int
) -> list[dict]:
    grouped: dict[str, list[TrialResult]] = {}
    for trial in trials:
        grouped.setdefault(trial.model, []).append(trial)

    analyses = []
    for model in sorted(grouped):
        model_trials = grouped[model]
        pass_units = question_pass_rates(model_trials)
        latency_units = question_latencies(model_trials)
        judge_units = question_judge_scores(model_trials)
        analyses.append(
            {
                "model": model,
                "pass_rate": {
                    "summary": numeric_summary(pass_units),
                    "ci95": bootstrap_interval(
                        pass_units,
                        config.bootstrap_iterations,
                        config.confidence_level,
                        seed,
                    ),
                },
                "latency_ms": {
                    "summary": numeric_summary(latency_units),
                    "ci95": bootstrap_interval(
                        latency_units,
                        config.bootstrap_iterations,
                        config.confidence_level,
                        seed + 1,
                    ),
                },
                "judge_score": {
                    "summary": numeric_summary(judge_units),
                    "ci95": bootstrap_interval(
                        judge_units,
                        config.bootstrap_iterations,
                        config.confidence_level,
                        seed + 2,
                    ),
                },
                "resampling": {
                    "unit": "question (repeats averaged first)",
                    "iterations": config.bootstrap_iterations,
                    "confidence_level": config.confidence_level,
                },
            }
        )
    return analyses


def paired_question_pass_rates(
    trials_a: list[TrialResult], trials_b: list[TrialResult]
) -> list[float]:
    def rates_by_question(trials: list[TrialResult]) -> dict[str, float]:
        grouped: dict[str, list[int]] = {}
        for trial in trials:
            if not trial.ok or trial.passed is None:
                continue
            grouped.setdefault(trial.entry_id, []).append(1 if trial.passed else 0)
        return {question: statistics.mean(values) for question, values in grouped.items()}

    rates_a = rates_by_question(trials_a)
    rates_b = rates_by_question(trials_b)
    common = sorted(set(rates_a) & set(rates_b))
    return [rates_b[question] - rates_a[question] for question in common]


def paired_comparison(
    trials_a: list[TrialResult],
    trials_b: list[TrialResult],
    config: StatisticsConfig,
    seed: int,
) -> dict:
    differences = paired_question_pass_rates(trials_a, trials_b)
    return {
        "differences_summary": numeric_summary(differences),
        "ci95": bootstrap_interval(
            differences, config.bootstrap_iterations, config.confidence_level, seed
        ),
        "units": len(differences),
        "resampling": {
            "unit": "question (repeats averaged first)",
            "iterations": config.bootstrap_iterations,
            "confidence_level": config.confidence_level,
        },
    }


def summarize_evaluations(trials: list[TrialResult]) -> list[dict]:
    grouped: dict[str, list[TrialResult]] = {}
    for trial in trials:
        if not trial.ok:
            continue
        evaluator = trial.evaluation.evaluator if trial.evaluation else "deterministic"
        grouped.setdefault(evaluator, []).append(trial)

    summaries = []
    for evaluator in sorted(grouped):
        evaluator_trials = grouped[evaluator]
        completed = [
            trial
            for trial in evaluator_trials
            if trial.evaluation is None or trial.evaluation.status == "completed"
        ]
        passed = sum(1 for trial in completed if _passed_value(trial) is True)
        binary = [trial for trial in completed if _passed_value(trial) is not None]
        scores = [
            trial.evaluation.score
            for trial in completed
            if trial.evaluation is not None and trial.evaluation.score is not None
        ]
        summaries.append(
            {
                "evaluator": evaluator,
                "trials": len(evaluator_trials),
                "completed": len(completed),
                "passed": passed,
                "failed_binary": sum(
                    1 for trial in binary if _passed_value(trial) is False
                ),
                "evaluator_failed": sum(
                    1
                    for trial in evaluator_trials
                    if trial.evaluation and trial.evaluation.status == "failed"
                ),
                "evaluator_unavailable": sum(
                    1
                    for trial in evaluator_trials
                    if trial.evaluation and trial.evaluation.status == "unavailable"
                ),
                "binary_pass_rate": passed / len(binary) if binary else None,
                "mean_score": statistics.mean(scores) if scores else None,
                "score_count": len(scores),
                "input_tokens": _sum_reported(
                    evaluator_trials, lambda trial: trial.evaluation.input_tokens
                ),
                "output_tokens": _sum_reported(
                    evaluator_trials, lambda trial: trial.evaluation.output_tokens
                ),
                "cost_usd": _sum_reported(
                    evaluator_trials, lambda trial: trial.evaluation.cost_usd
                ),
            }
        )
    return summaries


def _passed_value(trial: TrialResult) -> bool | None:
    if trial.evaluation is None:
        return trial.passed
    return trial.evaluation.passed


def per_model_evaluator_summaries(trials: list[TrialResult]) -> list[dict]:
    grouped: dict[tuple[str, str], list[TrialResult]] = {}
    for trial in trials:
        if not trial.ok:
            continue
        evaluator = trial.evaluation.evaluator if trial.evaluation else "deterministic"
        grouped.setdefault((trial.model, evaluator), []).append(trial)

    summaries = []
    for (model, evaluator) in sorted(grouped):
        evaluator_trials = grouped[(model, evaluator)]
        completed = [
            trial
            for trial in evaluator_trials
            if trial.evaluation is None or trial.evaluation.status == "completed"
        ]
        binary = [trial for trial in completed if _passed_value(trial) is not None]
        passed = sum(1 for trial in binary if _passed_value(trial) is True)
        scores = [
            trial.evaluation.score
            for trial in completed
            if trial.evaluation is not None and trial.evaluation.score is not None
        ]
        summaries.append(
            {
                "model": model,
                "evaluator": evaluator,
                "trials": len(evaluator_trials),
                "completed": len(completed),
                "passed": passed,
                "binary": len(binary),
                "binary_pass_rate": passed / len(binary) if binary else None,
                "mean_score": statistics.mean(scores) if scores else None,
                "score_count": len(scores),
            }
        )
    return summaries


def _sum_reported(trials: list[TrialResult], pick) -> float | int | None:
    values = [
        pick(trial)
        for trial in trials
        if trial.evaluation is not None and pick(trial) is not None
    ]
    if not values:
        return None
    return sum(values)
