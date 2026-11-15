from __future__ import annotations

import hashlib
import math
import random
import statistics
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime

from rich.progress import Progress

from minieval_bench.evaluation import (
    build_code_runner_from_config,
    build_judge_from_config,
    evaluate_response,
)
from minieval_bench.models import (
    BenchmarkConfig,
    DatasetEntry,
    ModelConfig,
    ModelStats,
    TrialResult,
)
from minieval_bench.providers import (
    LitellmProvider,
    MockProvider,
    build_provider,
    safe_error_message,
)


@dataclass
class PlannedTrial:
    model: ModelConfig
    entry: DatasetEntry
    repeat: int
    seed: int
    plan_index: int = 0


def repeat_seed(seed: int, repeat: int) -> int:
    payload = f"{seed}:{repeat}".encode()
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def build_plan(config: BenchmarkConfig, entries: list[DatasetEntry]) -> list[PlannedTrial]:
    plan = [
        PlannedTrial(
            model=model,
            entry=entry,
            repeat=repeat,
            seed=repeat_seed(config.defaults.seed, repeat),
        )
        for model in config.models
        for entry in entries
        for repeat in range(config.defaults.repeats)
    ]
    if config.defaults.shuffle:
        random.Random(config.defaults.seed).shuffle(plan)
    for plan_index, item in enumerate(plan):
        item.plan_index = plan_index
    return plan


def run_benchmark(
    config: BenchmarkConfig, entries: list[DatasetEntry], show_progress: bool = True
) -> tuple[list[TrialResult], datetime, float]:
    started_at = datetime.now(UTC)
    started = time.perf_counter()
    providers = {model.name: build_provider(model) for model in config.models}
    judge = build_judge_from_config(config, config.defaults.seed)
    code_runner = build_code_runner_from_config(config)
    plan = build_plan(config, entries)

    trials: list[TrialResult] = []
    executor = ThreadPoolExecutor(max_workers=config.defaults.concurrency)
    futures = [
        executor.submit(
            run_trial,
            providers[item.model.name],
            item,
            time.perf_counter(),
            judge,
            code_runner,
        )
        for item in plan
    ]
    try:
        if show_progress:
            with Progress(transient=True) as progress:
                task = progress.add_task("Running requests", total=len(plan))
                for future in as_completed(futures):
                    trials.append(future.result())
                    progress.advance(task)
        else:
            for future in as_completed(futures):
                trials.append(future.result())
    except KeyboardInterrupt:
        executor.shutdown(cancel_futures=True)
        raise
    executor.shutdown(wait=True)

    duration_seconds = time.perf_counter() - started
    trials.sort(key=lambda trial: (trial.model, trial.entry_id, trial.repeat))
    return trials, started_at, duration_seconds


def run_trial(
    provider: MockProvider | LitellmProvider,
    planned: PlannedTrial,
    queued_at: float,
    judge=None,
    code_runner=None,
) -> TrialResult:
    started = time.perf_counter()
    queue_ms = (started - queued_at) * 1000
    generation_started = time.perf_counter()
    try:
        response = provider.complete(
            planned.entry.prompt, planned.entry.system, planned.seed
        )
    except Exception as exc:
        generation_ms = (time.perf_counter() - generation_started) * 1000
        return TrialResult(
            model=planned.model.name,
            entry_id=planned.entry.id,
            repeat=planned.repeat,
            ok=False,
            response=None,
            latency_ms=generation_ms,
            input_tokens=None,
            output_tokens=None,
            cost_usd=None,
            error_type=type(exc).__name__,
            error_message=safe_error_message(exc),
            passed=None,
            queue_ms=queue_ms,
            plan_index=planned.plan_index,
            generation_ms=generation_ms,
            evaluation_ms=None,
            total_ms=generation_ms,
        )
    generation_ms = (time.perf_counter() - generation_started) * 1000
    evaluation_started = time.perf_counter()
    outcome = evaluate_response(
        planned.entry, response.text, judge=judge, code_runner=code_runner
    )
    evaluation_ms = (time.perf_counter() - evaluation_started) * 1000
    total_ms = generation_ms + evaluation_ms
    return TrialResult(
        model=planned.model.name,
        entry_id=planned.entry.id,
        repeat=planned.repeat,
        ok=True,
        response=response.text,
        latency_ms=generation_ms,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cost_usd=response.cost_usd,
        error_type=None,
        error_message=None,
        passed=outcome.passed,
        queue_ms=queue_ms,
        plan_index=planned.plan_index,
        evaluation=outcome,
        generation_ms=generation_ms,
        evaluation_ms=evaluation_ms,
        total_ms=total_ms,
    )


def is_binary_evaluated(trial: TrialResult) -> bool:
    if trial.evaluation is None:
        return True
    return trial.evaluation.status == "completed" and trial.evaluation.passed is not None


def evaluator_name(trial: TrialResult) -> str:
    return trial.evaluation.evaluator if trial.evaluation else "deterministic"


def binary_rate(trials: list[TrialResult]) -> float | None:
    binary = [trial for trial in trials if is_binary_evaluated(trial)]
    if not binary:
        return None
    passed = sum(1 for trial in binary if trial.passed)
    return passed / len(binary)


def code_test_totals(trials: list[TrialResult]) -> tuple[int | None, int | None]:
    passed_total = 0
    test_total = 0
    reported = False
    for trial in trials:
        if not trial.evaluation or not trial.evaluation.details:
            continue
        details = trial.evaluation.details
        if trial.evaluation.status == "completed" and isinstance(details.get("total"), int):
            passed_total += details["passed"]
            test_total += details["total"]
            reported = True
    if not reported:
        return None, None
    return passed_total, test_total


def compute_stats(trials: list[TrialResult]) -> list[ModelStats]:
    grouped: dict[str, list[TrialResult]] = {}
    for trial in trials:
        grouped.setdefault(trial.model, []).append(trial)

    stats = []
    for model in sorted(grouped):
        model_trials = grouped[model]
        succeeded = [trial for trial in model_trials if trial.ok]
        binary = [trial for trial in succeeded if is_binary_evaluated(trial)]
        by_evaluator: dict[str, list[TrialResult]] = {}
        for trial in succeeded:
            by_evaluator.setdefault(evaluator_name(trial), []).append(trial)
        latencies = [
            trial.generation_ms if trial.generation_ms is not None else trial.latency_ms
            for trial in succeeded
        ]
        evaluation_latencies = [
            trial.evaluation_ms for trial in succeeded if trial.evaluation_ms is not None
        ]
        total_latencies = [
            trial.total_ms for trial in succeeded if trial.total_ms is not None
        ]
        queue_delays = [trial.queue_ms for trial in model_trials if trial.queue_ms is not None]
        passed = sum(1 for trial in binary if trial.passed)
        errors: dict[str, int] = {}
        for trial in model_trials:
            if trial.error_type:
                errors[trial.error_type] = errors.get(trial.error_type, 0) + 1
        input_tokens, input_reported = available_total(
            [trial.input_tokens for trial in succeeded]
        )
        output_tokens, output_reported = available_total(
            [trial.output_tokens for trial in succeeded]
        )
        cost_usd, cost_reported = available_total([trial.cost_usd for trial in succeeded])
        evaluated = [trial for trial in model_trials if trial.evaluation is not None]
        evaluation_input_tokens, _ = available_total(
            [trial.evaluation.input_tokens for trial in evaluated]
        )
        evaluation_output_tokens, _ = available_total(
            [trial.evaluation.output_tokens for trial in evaluated]
        )
        evaluation_cost, evaluation_cost_reported = available_total(
            [trial.evaluation.cost_usd for trial in evaluated]
        )

        deterministic_trials = by_evaluator.get("deterministic", [])
        answer_accuracy = binary_rate(deterministic_trials)
        json_schema_trials = by_evaluator.get("json_schema", [])
        code_trials = by_evaluator.get("code_tests", [])
        judge_trials = by_evaluator.get("judge", [])
        judge_scores = [
            trial.evaluation.score
            for trial in judge_trials
            if trial.evaluation
            and trial.evaluation.status == "completed"
            and trial.evaluation.score is not None
        ]
        code_passed, code_total = code_test_totals(code_trials)
        stats.append(
            ModelStats(
                model=model,
                requests=len(model_trials),
                succeeded=len(succeeded),
                failed=len(model_trials) - len(succeeded),
                success_rate=len(succeeded) / len(model_trials),
                evaluable=len(binary),
                passed=passed,
                score=answer_accuracy,
                median_latency_ms=statistics.median(latencies) if latencies else None,
                p95_latency_ms=percentile(latencies, 95) if latencies else None,
                median_evaluation_ms=(
                    statistics.median(evaluation_latencies) if evaluation_latencies else None
                ),
                median_total_ms=statistics.median(total_latencies) if total_latencies else None,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
                errors=errors,
                answer_accuracy=answer_accuracy,
                evaluation_coverage=len(binary) / len(model_trials),
                overall_pass_rate=passed / len(model_trials),
                incorrect=len(binary) - passed,
                median_queue_ms=statistics.median(queue_delays) if queue_delays else None,
                input_tokens_reported=input_reported,
                output_tokens_reported=output_reported,
                cost_reported=cost_reported,
                evaluator_failed=sum(
                    1 for trial in evaluated if trial.evaluation.status == "failed"
                ),
                evaluator_unavailable=sum(
                    1 for trial in evaluated if trial.evaluation.status == "unavailable"
                ),
                evaluation_input_tokens=evaluation_input_tokens,
                evaluation_output_tokens=evaluation_output_tokens,
                evaluation_cost_usd=evaluation_cost,
                evaluation_cost_reported=evaluation_cost_reported,
                deterministic_trials=len(deterministic_trials),
                json_schema_conformance=binary_rate(json_schema_trials),
                json_schema_trials=len(json_schema_trials),
                code_eval_pass_rate=binary_rate(code_trials),
                code_eval_trials=len(code_trials),
                code_tests_passed=code_passed,
                code_tests_total=code_total,
                judge_score_mean=statistics.mean(judge_scores) if judge_scores else None,
                judge_evaluations=len(judge_scores),
                judge_trials=len(judge_trials),
            )
        )
    return stats


def percentile(values: Sequence[float], pct: float) -> float:
    ordered = sorted(values)
    rank = math.ceil(pct / 100 * len(ordered))
    return ordered[rank - 1]


def available_total(
    values: Sequence[int | float | None],
) -> tuple[int | float | None, int]:
    reported = [value for value in values if value is not None]
    if not reported:
        return None, 0
    return sum(reported), len(reported)
