from __future__ import annotations

from pathlib import Path

from minieval_bench.evaluation.code import DockerCodeRunner
from minieval_bench.evaluation.judge import (
    DEFAULT_RUBRIC,
    BaseJudge,
    JudgeError,
    Rubric,
    build_judge,
    load_rubric,
)
from minieval_bench.evaluation.structured import evaluate_json_schema
from minieval_bench.metrics import evaluate as evaluate_metric
from minieval_bench.models import (
    BenchmarkConfig,
    DatasetEntry,
    EvaluationOutcome,
)

DETERMINISTIC_METRICS = ("exact_match", "contains", "regex")


def evaluate_response(
    entry: DatasetEntry,
    response: str,
    judge: BaseJudge | None = None,
    code_runner: DockerCodeRunner | None = None,
) -> EvaluationOutcome:
    if entry.metric in DETERMINISTIC_METRICS:
        passed = evaluate_metric(entry.metric, response, entry.reference or "")
        return EvaluationOutcome(
            evaluator="deterministic",
            status="completed",
            passed=passed,
            score=1.0 if passed else 0.0,
        )
    if entry.metric == "json_schema":
        return evaluate_json_schema(response, entry.json_schema or {})
    if entry.metric == "code_tests":
        if code_runner is None:
            return unavailable("code evaluation is not enabled: configure evaluators.code")
        return code_runner.run(response, entry.tests or "")
    if entry.metric == "judge":
        if judge is None:
            return unavailable("judge evaluation is not enabled: configure evaluators.judge")
        return judge.evaluate(entry.prompt, entry.reference, response)
    return EvaluationOutcome(
        evaluator="unknown",
        status="failed",
        error_message=f"unknown metric: {entry.metric}",
    )


def unavailable(message: str) -> EvaluationOutcome:
    return EvaluationOutcome(evaluator="unknown", status="unavailable", error_message=message)


def build_judge_from_config(config: BenchmarkConfig, seed: int) -> BaseJudge | None:
    judge_config = config.evaluators.judge if config.evaluators else None
    if judge_config is None:
        return None
    rubric = DEFAULT_RUBRIC
    if judge_config.rubric:
        rubric = load_rubric(Path(judge_config.rubric))
    return build_judge(judge_config, rubric, seed)


def build_code_runner_from_config(config: BenchmarkConfig) -> DockerCodeRunner | None:
    code_config = config.evaluators.code if config.evaluators else None
    if code_config is None:
        return None
    return DockerCodeRunner(code_config)


def plan_judge_calls(config: BenchmarkConfig, entries: list[DatasetEntry]) -> int:
    judge_config = config.evaluators.judge if config.evaluators else None
    if judge_config is None:
        return 0
    judge_entries = sum(1 for entry in entries if entry.metric == "judge")
    return judge_entries * len(config.models) * config.defaults.repeats * judge_config.repeats


__all__ = [
    "DEFAULT_RUBRIC",
    "BaseJudge",
    "DockerCodeRunner",
    "JudgeError",
    "Rubric",
    "build_code_runner_from_config",
    "build_judge",
    "build_judge_from_config",
    "evaluate_response",
    "load_rubric",
    "plan_judge_calls",
]
