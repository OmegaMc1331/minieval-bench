from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import ValidationError

from minieval_bench.config import apply_defaults, format_validation_error
from minieval_bench.evaluation import build_judge_from_config, plan_judge_calls
from minieval_bench.evaluation.code import docker_available
from minieval_bench.evaluation.judge import JudgeError
from minieval_bench.models import (
    BenchmarkConfig,
    DatasetEntry,
    Defaults,
    Manifest,
    ModelConfig,
    ModelStats,
    TrialResult,
)
from minieval_bench.reports import build_manifest, load_results, write_reports
from minieval_bench.runner import compute_stats, run_benchmark
from minieval_bench.suites.lockfile import (
    LOCKFILE_NAME,
    build_lockfile,
    read_lockfile,
    task_lock_section,
    write_lockfile,
)
from minieval_bench.suites.models import SuiteTask
from minieval_bench.suites.registry import BUILTIN_ORIGIN, LoadedSuite
from minieval_bench.suites.sampling import SamplingError, SamplingPlan, apply_sampling

SUITE_MANIFEST_NAME = "suite-manifest.json"
SUITE_RESULTS_NAME = "suite-results.json"
TASKS_DIR_NAME = "tasks"


class SuiteRunError(Exception):
    pass


@dataclass
class UserOverrides:
    repeats: int | None = None
    seed: int | None = None
    concurrency: int | None = None
    timeout: float | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    shuffle: bool | None = None

    def apply(self, defaults: Defaults) -> Defaults:
        values = {
            key: value
            for key, value in vars(self).items()
            if value is not None
        }
        return defaults.model_copy(update=values)

    def as_dict(self) -> dict:
        return {key: value for key, value in vars(self).items() if value is not None}


@dataclass
class SuiteRunOptions:
    models_path: Path
    output_dir: Path
    overrides: UserOverrides = field(default_factory=UserOverrides)
    sampling: SamplingPlan = field(default_factory=SamplingPlan)
    resume: bool = False
    max_calls: int | None = None
    allow_code_execution: bool = False


@dataclass
class TaskPlan:
    task: SuiteTask
    entries: list[DatasetEntry]
    dataset_sha256: str
    sampling_record: dict
    config: BenchmarkConfig
    lock_section: dict

    @property
    def planned_calls(self) -> int:
        return len(self.entries) * self.config.defaults.repeats * len(self.config.models)

    @property
    def planned_judge_calls(self) -> int:
        return plan_judge_calls(self.config, self.entries)

    @property
    def needs_code_runtime(self) -> bool:
        return any(entry.metric == "code_tests" for entry in self.entries)


@dataclass
class TaskRunResult:
    task_id: str
    status: str
    weight: float
    planned_calls: int
    executed_calls: int = 0
    reused: bool = False
    error: str | None = None
    duration_seconds: float | None = None
    manifest: Manifest | None = None
    stats: list[ModelStats] = field(default_factory=list)
    trials: list[TrialResult] = field(default_factory=list)


@dataclass
class SuiteRunOutcome:
    loaded: LoadedSuite
    output_dir: Path
    lockfile: dict
    models: list[ModelConfig]
    effective_defaults: Defaults
    user_overrides: dict
    sampling_plan: dict
    tasks: list[TaskRunResult]
    started_at: datetime
    duration_seconds: float
    interrupted: bool
    resume_used: bool


def load_models_file(path: Path) -> list[ModelConfig]:
    if not path.is_file():
        raise SuiteRunError(f"models file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SuiteRunError(f"invalid YAML in models file {path}: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("models"), list):
        raise SuiteRunError(f"models file {path} must contain a 'models' list")
    try:
        models = [ModelConfig.model_validate(item) for item in raw["models"]]
    except ValidationError as exc:
        raise SuiteRunError(
            f"invalid models file {path}: {format_validation_error(exc)}"
        ) from exc
    if not models:
        raise SuiteRunError(f"models file {path} defines no model")
    names = [model.name for model in models]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise SuiteRunError(f"duplicate model names: {', '.join(duplicates)}")
    for model in models:
        if model.provider == "litellm" and not model.model:
            raise SuiteRunError(
                f"model '{model.name}' uses provider 'litellm' and requires a model string"
            )
    return models


def build_task_config(
    loaded: LoadedSuite,
    task: SuiteTask,
    entries: list[DatasetEntry],
    models: list[ModelConfig],
    effective_defaults: Defaults,
) -> BenchmarkConfig:
    definition = loaded.definition
    config = BenchmarkConfig(
        benchmark=f"{definition.id}/{task.id}",
        dataset=dataset_display_path(loaded, task),
        models=[model.model_copy(deep=True) for model in models],
        defaults=effective_defaults,
        evaluators=(
            definition.evaluators.model_copy(deep=True)
            if definition.evaluators
            else None
        ),
        statistics=definition.statistics,
    )
    apply_defaults(config)
    return config


def dataset_display_path(loaded: LoadedSuite, task: SuiteTask) -> str:
    if loaded.origin == BUILTIN_ORIGIN:
        return f"{loaded.location}/{task.dataset}"
    return str(Path(loaded.location) / task.dataset)


def prepare_run(
    loaded: LoadedSuite, options: SuiteRunOptions
) -> tuple[list[TaskPlan], list[ModelConfig], Defaults, dict]:
    models = load_models_file(options.models_path)
    return prepare_run_with_models(loaded, options, models)


def prepare_run_with_models(
    loaded: LoadedSuite, options: SuiteRunOptions, models: list[ModelConfig]
) -> tuple[list[TaskPlan], list[ModelConfig], Defaults, dict]:
    effective_defaults = options.overrides.apply(loaded.definition.defaults)
    plans = []
    for task in loaded.definition.tasks:
        dataset = loaded.datasets[task.id]
        try:
            selected, sampling_record = apply_sampling(
                dataset.entries, dataset.sha256, options.sampling
            )
        except SamplingError as exc:
            raise SuiteRunError(f"task '{task.id}': {exc}") from exc
        config = build_task_config(loaded, task, selected, models, effective_defaults)
        plans.append(
            TaskPlan(
                task=task,
                entries=selected,
                dataset_sha256=dataset.sha256,
                sampling_record=sampling_record,
                config=config,
                lock_section=task_lock_section(
                    task, dataset.sha256, dataset.entries, sampling_record
                ),
            )
        )
    return plans, models, effective_defaults, options.overrides.as_dict()


def preflight_problems(
    loaded: LoadedSuite, plans: list[TaskPlan], options: SuiteRunOptions
) -> list[str]:
    problems = []
    total_calls = sum(plan.planned_calls for plan in plans)
    if options.max_calls is not None and total_calls > options.max_calls:
        problems.append(
            f"planned candidate calls ({total_calls}) exceed --max-calls "
            f"({options.max_calls})"
        )
    code_needed = any(plan.needs_code_runtime for plan in plans)
    if code_needed and not options.allow_code_execution:
        problems.append(
            "the suite contains code_tests entries: pass --allow-code-execution to run "
            "them inside the restricted Docker runtime"
        )
    for plan in plans:
        if plan.planned_judge_calls == 0:
            continue
        judge = build_judge_from_config(plan.config, plan.config.defaults.seed)
        if judge is None:
            continue
        refusal = judge.plan_check(plan.planned_judge_calls)
        if refusal:
            problems.append(f"task '{plan.task.id}': {refusal}")
    return problems


def dry_run_details(plans: list[TaskPlan]) -> dict:
    code_needed = any(plan.needs_code_runtime for plan in plans)
    return {
        "docker_required": code_needed,
        "docker_available": docker_available() if code_needed else None,
        "max_cost_upper_bounds": {
            plan.task.id: {
                model.name: estimated_cost_upper_bound(plan, model)
                for model in plan.config.models
            }
            for plan in plans
        },
    }


def estimated_cost_upper_bound(plan: TaskPlan, model: ModelConfig) -> float | None:
    if model.input_cost_per_1m is None or model.output_cost_per_1m is None:
        return None
    if model.max_tokens is None:
        return None
    input_words = sum(
        len((entry.system or "").split()) + len(entry.prompt.split())
        for entry in plan.entries
    )
    calls = len(plan.entries) * plan.config.defaults.repeats
    input_tokens = input_words * calls
    output_tokens = model.max_tokens * calls
    input_cost = input_tokens * model.input_cost_per_1m
    output_cost = output_tokens * model.output_cost_per_1m
    return (input_cost + output_cost) / 1_000_000


def run_suite(loaded: LoadedSuite, options: SuiteRunOptions) -> SuiteRunOutcome:
    plans, models, effective_defaults, user_overrides = prepare_run(loaded, options)
    lockfile = build_lockfile(
        loaded,
        models,
        effective_defaults,
        user_overrides,
        [plan.lock_section for plan in plans],
    )
    started_at = datetime.now(UTC)
    started = time.perf_counter()
    results: list[TaskRunResult] = []
    interrupted = False
    for plan in plans:
        if interrupted:
            results.append(
                TaskRunResult(
                    task_id=plan.task.id,
                    status="not_executed",
                    weight=plan.task.weight,
                    planned_calls=plan.planned_calls,
                    error="skipped after an earlier interruption",
                )
            )
            continue
        if options.resume and stored_task_matches(options.output_dir, plan):
            try:
                results.append(load_reused_result(plan, options.output_dir))
                continue
            except Exception:
                pass
        try:
            results.append(execute_task(plan, options))
        except KeyboardInterrupt:
            interrupted = True
            results.append(
                TaskRunResult(
                    task_id=plan.task.id,
                    status="not_executed",
                    weight=plan.task.weight,
                    planned_calls=plan.planned_calls,
                    error="interrupted before completion; partial trials were discarded",
                )
            )
    duration_seconds = time.perf_counter() - started
    outcome = SuiteRunOutcome(
        loaded=loaded,
        output_dir=options.output_dir,
        lockfile=lockfile,
        models=models,
        effective_defaults=effective_defaults,
        user_overrides=user_overrides,
        sampling_plan=sampling_plan_dict(options.sampling),
        tasks=results,
        started_at=started_at,
        duration_seconds=duration_seconds,
        interrupted=interrupted,
        resume_used=options.resume,
    )
    write_suite_artefacts(outcome)
    return outcome


def sampling_plan_dict(plan: SamplingPlan) -> dict:
    return {
        "method": plan.method,
        "count": plan.count,
        "seed": plan.seed,
    }


def execute_task(plan: TaskPlan, options: SuiteRunOptions) -> TaskRunResult:
    if plan.needs_code_runtime and not options.allow_code_execution:
        return TaskRunResult(
            task_id=plan.task.id,
            status="unavailable",
            weight=plan.task.weight,
            planned_calls=plan.planned_calls,
            error="code evaluation requires --allow-code-execution",
        )
    try:
        trials, started_at, duration = run_benchmark(plan.config, plan.entries, show_progress=False)
    except JudgeError as exc:
        return TaskRunResult(
            task_id=plan.task.id,
            status="unavailable",
            weight=plan.task.weight,
            planned_calls=plan.planned_calls,
            error=str(exc),
        )
    except Exception as exc:
        return TaskRunResult(
            task_id=plan.task.id,
            status="infrastructure_failed",
            weight=plan.task.weight,
            planned_calls=plan.planned_calls,
            error=f"{type(exc).__name__}: {exc}",
        )
    stats = compute_stats(trials)
    manifest = build_manifest(
        plan.config, plan.entries, started_at, duration, dataset_sha256=plan.dataset_sha256
    )
    write_task_reports(options.output_dir, plan, manifest, stats, trials)
    return TaskRunResult(
        task_id=plan.task.id,
        status=task_status(plan, trials),
        weight=plan.task.weight,
        planned_calls=plan.planned_calls,
        executed_calls=len(trials),
        duration_seconds=duration,
        manifest=manifest,
        stats=stats,
        trials=trials,
    )


def task_status(plan: TaskPlan, trials: list[TrialResult]) -> str:
    if len(trials) != plan.planned_calls:
        return "partially_completed"
    if any(not trial.ok for trial in trials):
        return "partially_completed"
    if any(
        trial.evaluation is not None and trial.evaluation.status in ("failed", "unavailable")
        for trial in trials
    ):
        return "partially_completed"
    return "completed"


def write_task_reports(
    output_dir: Path,
    plan: TaskPlan,
    manifest: Manifest,
    stats: list[ModelStats],
    trials: list[TrialResult],
) -> Path:
    tasks_dir = output_dir / TASKS_DIR_NAME
    final_dir = tasks_dir / plan.task.id
    staging_dir = tasks_dir / f".staging-{plan.task.id}"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    write_reports(staging_dir, manifest, stats, trials, html=True)
    if final_dir.exists():
        shutil.rmtree(final_dir)
    staging_dir.rename(final_dir)
    return final_dir


def stored_task_matches(output_dir: Path, plan: TaskPlan) -> bool:
    lock_path = output_dir / LOCKFILE_NAME
    if not lock_path.is_file():
        return False
    results_path = output_dir / TASKS_DIR_NAME / plan.task.id / "results.json"
    if not results_path.is_file():
        return False
    try:
        stored = read_lockfile(lock_path)
    except ValueError:
        return False
    stored_sections = {
        section.get("id"): section for section in stored.get("tasks", [])
    }
    return stored_sections.get(plan.task.id) == plan.lock_section


def load_reused_result(plan: TaskPlan, output_dir: Path) -> TaskRunResult:
    task_dir = output_dir / TASKS_DIR_NAME / plan.task.id
    manifest, stats, trials = load_results(task_dir)
    return TaskRunResult(
        task_id=plan.task.id,
        status=task_status(plan, trials),
        weight=plan.task.weight,
        planned_calls=plan.planned_calls,
        executed_calls=len(trials),
        reused=True,
        duration_seconds=manifest.duration_seconds,
        manifest=manifest,
        stats=stats,
        trials=trials,
    )


def write_suite_artefacts(outcome: SuiteRunOutcome) -> None:
    output_dir = outcome.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_lockfile(output_dir / LOCKFILE_NAME, outcome.lockfile)
    write_json_atomic(
        output_dir / SUITE_MANIFEST_NAME, suite_manifest_payload(outcome)
    )
    write_json_atomic(output_dir / SUITE_RESULTS_NAME, suite_results_payload(outcome))


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def suite_manifest_payload(outcome: SuiteRunOutcome) -> dict:
    lockfile = outcome.lockfile
    return {
        "minieval_version": lockfile["minieval_version"],
        "suite": {
            **lockfile["suite"],
            "origin": outcome.loaded.origin,
            "location": outcome.loaded.location,
        },
        "created_at_utc": outcome.started_at.isoformat(),
        "duration_seconds": outcome.duration_seconds,
        "generation": lockfile["generation"],
        "sampling": outcome.sampling_plan,
        "statistics": lockfile["statistics"],
        "evaluators": lockfile["evaluators"],
        "protocol_modified": bool(outcome.user_overrides),
        "sampled": outcome.sampling_plan.get("method") != "none",
        "interrupted": outcome.interrupted,
        "resume_used": outcome.resume_used,
        "models": [model.model_dump(mode="json") for model in outcome.models],
        "tasks": [
            {
                "id": result.task_id,
                "status": result.status,
                "weight": result.weight,
                "planned_calls": result.planned_calls,
                "executed_calls": result.executed_calls,
                "reused": result.reused,
                "error": result.error,
                "duration_seconds": result.duration_seconds,
                "dataset_sha256": next(
                    section["dataset"]["sha256"]
                    for section in lockfile["tasks"]
                    if section["id"] == result.task_id
                ),
            }
            for result in outcome.tasks
        ],
    }


def suite_results_payload(outcome: SuiteRunOutcome) -> dict:
    return {
        "tasks": [
            {
                "id": result.task_id,
                "status": result.status,
                "weight": result.weight,
                "reused": result.reused,
                "stats": [item.model_dump(mode="json") for item in result.stats],
            }
            for result in outcome.tasks
        ]
    }
