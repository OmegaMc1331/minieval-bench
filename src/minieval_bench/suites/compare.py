from __future__ import annotations

from pathlib import Path

from minieval_bench.evaluation.statistics import (
    binary_pass_value,
    judge_score_value,
    paired_comparison,
)
from minieval_bench.suites.lockfile import lockfile_differences, statistics_from_lockfile
from minieval_bench.suites.reports import (
    FAMILY_FIELDS,
    FAMILY_LABELS,
    SuiteReportError,
    SuiteRunData,
    TaskRunData,
    load_suite_run,
)

PAIRED_METRICS = [
    ("pass rate (conditional on evaluation)", binary_pass_value, None),
    ("judge score", judge_score_value, "judge"),
]


def build_suite_comparison_payload(run_dirs: list[Path]) -> dict:
    runs = [load_suite_run(run_dir) for run_dir in run_dirs]
    labels = unique_labels(runs)
    differences: list[str] = []
    for other in runs[1:]:
        differences.extend(lockfile_differences(runs[0].lockfile, other.lockfile))
    protocol_equal = not differences

    task_ids = [task.task_id for task in runs[0].tasks]
    for other in runs[1:]:
        other_ids = [task.task_id for task in other.tasks]
        if other_ids != task_ids:
            protocol_equal = False
            differences.append(
                f"task lists differ between runs: {task_ids} vs {other_ids}"
            )

    payload = {
        "kind": "suite-compare",
        "runs": [
            {
                "label": label,
                "suite": run.manifest.get("suite", {}).get("id"),
                "version": run.manifest.get("suite", {}).get("version"),
                "created_at_utc": run.manifest.get("created_at_utc"),
                "path": str(run.path),
                "statuses": run.manifest.get("tasks", []),
            }
            for label, run in zip(labels, runs, strict=True)
        ],
        "protocolEqual": protocol_equal,
        "differences": differences,
    }
    if protocol_equal:
        payload["tasks"] = comparison_tables(runs)
        payload["paired"] = paired_analyses(runs)
    return payload


def unique_labels(runs: list[SuiteRunData]) -> list[str]:
    labels = [run.manifest.get("suite", {}).get("id") or "suite" for run in runs]
    unique = []
    for index, label in enumerate(labels):
        if labels.count(label) > 1:
            unique.append(f"{label} ({index + 1})")
        else:
            unique.append(label)
    return unique


def comparison_tables(runs: list[SuiteRunData]) -> list[dict]:
    tables = []
    for task in runs[0].tasks:
        families = sorted(
            task.families(), key=lambda family: list(FAMILY_LABELS).index(family)
        )
        rows = []
        for model in ordered_models(runs, task.task_id):
            values: dict[str, list] = {}
            for family in families:
                rate_field, _ = FAMILY_FIELDS[family]
                per_run = []
                for run in runs:
                    run_task = find_task(run, task.task_id)
                    stats = (
                        next((item for item in run_task.stats if item.model == model), None)
                        if run_task
                        else None
                    )
                    per_run.append(getattr(stats, rate_field) if stats else None)
                values[family] = per_run
            rows.append({"model": model, "values": values})
        latency = []
        for model in ordered_models(runs, task.task_id):
            per_run = []
            for run in runs:
                run_task = find_task(run, task.task_id)
                stats = (
                    next((item for item in run_task.stats if item.model == model), None)
                    if run_task
                    else None
                )
                per_run.append(stats.median_total_ms if stats else None)
            latency.append({"model": model, "values": per_run})
        tables.append(
            {
                "id": task.task_id,
                "families": [
                    {"family": family, "label": FAMILY_LABELS[family]} for family in families
                ],
                "rows": rows,
                "latency": latency,
            }
        )
    return tables


def find_task(run: SuiteRunData, task_id: str) -> TaskRunData | None:
    return next((task for task in run.tasks if task.task_id == task_id), None)


def ordered_models(runs: list[SuiteRunData], task_id: str) -> list[str]:
    ordered = [model.get("name") for model in runs[0].manifest.get("models", [])]
    observed = set()
    for run in runs:
        task = find_task(run, task_id)
        if task:
            observed.update(item.model for item in task.stats)
    return ordered + sorted(observed - set(ordered))


def paired_analyses(runs: list[SuiteRunData]) -> list[dict]:
    if len(runs) != 2:
        return []
    first, second = runs
    statistics_config = statistics_from_lockfile(first.lockfile)
    seed = first.lockfile.get("generation", {}).get("defaults", {}).get("seed", 0)
    analyses = []
    for task in first.tasks:
        other_task = find_task(second, task.task_id)
        if other_task is None:
            continue
        for model in ordered_models(runs, task.task_id):
            trials_a = [trial for trial in task.trials if trial.model == model]
            trials_b = [trial for trial in other_task.trials if trial.model == model]
            if not trials_a or not trials_b:
                continue
            for metric, selector, required_family in PAIRED_METRICS:
                if required_family and required_family not in task.families():
                    continue
                comparison = paired_comparison(
                    trials_a, trials_b, statistics_config, seed, metric, selector
                )
                if comparison["units"] == 0:
                    continue
                analyses.append({"task": task.task_id, "model": model, **comparison})
    return analyses


__all__ = ["build_suite_comparison_payload", "SuiteReportError", "load_suite_run"]
