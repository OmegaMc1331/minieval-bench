from __future__ import annotations

import math
import statistics
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime

from rich.progress import Progress

from minieval_bench.metrics import evaluate
from minieval_bench.models import (
    BenchmarkConfig,
    DatasetEntry,
    ModelConfig,
    ModelStats,
    TrialResult,
)
from minieval_bench.providers import LitellmProvider, MockProvider, build_provider


def run_benchmark(
    config: BenchmarkConfig, entries: list[DatasetEntry], show_progress: bool = True
) -> tuple[list[TrialResult], datetime]:
    started_at = datetime.now(UTC)
    providers = {model.name: build_provider(model, config.defaults.seed) for model in config.models}
    jobs = [
        (model, entry, repeat)
        for model in config.models
        for entry in entries
        for repeat in range(config.defaults.repeats)
    ]

    trials: list[TrialResult] = []
    with ThreadPoolExecutor(max_workers=config.defaults.concurrency) as executor:
        futures = [
            executor.submit(run_trial, providers[model.name], model, entry, repeat)
            for model, entry, repeat in jobs
        ]
        if show_progress:
            with Progress(transient=True) as progress:
                task = progress.add_task("Running requests", total=len(jobs))
                for future in as_completed(futures):
                    trials.append(future.result())
                    progress.advance(task)
        else:
            for future in as_completed(futures):
                trials.append(future.result())

    trials.sort(key=lambda trial: (trial.model, trial.entry_id, trial.repeat))
    return trials, started_at


def run_trial(
    provider: MockProvider | LitellmProvider, model: ModelConfig, entry: DatasetEntry, repeat: int
) -> TrialResult:
    started = time.perf_counter()
    try:
        response = provider.complete(entry.prompt, entry.system)
    except Exception as exc:
        return TrialResult(
            model=model.name,
            entry_id=entry.id,
            repeat=repeat,
            ok=False,
            response=None,
            latency_ms=(time.perf_counter() - started) * 1000,
            input_tokens=None,
            output_tokens=None,
            cost_usd=None,
            error_type=type(exc).__name__,
            error_message=str(exc),
            passed=None,
        )
    return TrialResult(
        model=model.name,
        entry_id=entry.id,
        repeat=repeat,
        ok=True,
        response=response.text,
        latency_ms=(time.perf_counter() - started) * 1000,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cost_usd=response.cost_usd,
        error_type=None,
        error_message=None,
        passed=evaluate(entry.metric, response.text, entry.reference),
    )


def compute_stats(trials: list[TrialResult]) -> list[ModelStats]:
    grouped: dict[str, list[TrialResult]] = {}
    for trial in trials:
        grouped.setdefault(trial.model, []).append(trial)

    stats = []
    for model in sorted(grouped):
        model_trials = grouped[model]
        succeeded = [trial for trial in model_trials if trial.ok]
        latencies = [trial.latency_ms for trial in succeeded]
        passed = sum(1 for trial in succeeded if trial.passed)
        errors: dict[str, int] = {}
        for trial in model_trials:
            if trial.error_type:
                errors[trial.error_type] = errors.get(trial.error_type, 0) + 1
        stats.append(
            ModelStats(
                model=model,
                requests=len(model_trials),
                succeeded=len(succeeded),
                failed=len(model_trials) - len(succeeded),
                success_rate=len(succeeded) / len(model_trials),
                evaluable=len(succeeded),
                passed=passed,
                score=passed / len(succeeded) if succeeded else None,
                median_latency_ms=statistics.median(latencies) if latencies else None,
                p95_latency_ms=percentile(latencies, 95) if latencies else None,
                input_tokens=sum_available([trial.input_tokens for trial in succeeded]),
                output_tokens=sum_available([trial.output_tokens for trial in succeeded]),
                cost_usd=sum_available([trial.cost_usd for trial in succeeded]),
                errors=errors,
            )
        )
    return stats


def percentile(values: Sequence[float], pct: float) -> float:
    ordered = sorted(values)
    rank = math.ceil(pct / 100 * len(ordered))
    return ordered[rank - 1]


def sum_available(values: Sequence[int | float | None]) -> int | float | None:
    if all(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)
