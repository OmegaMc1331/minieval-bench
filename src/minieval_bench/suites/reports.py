from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from minieval_bench.html_report import render_standalone
from minieval_bench.models import Manifest, ModelStats, TrialResult
from minieval_bench.suites.lockfile import LOCKFILE_NAME, read_lockfile
from minieval_bench.suites.runner import SUITE_MANIFEST_NAME

SUITE_REPORT_MD = "suite-report.md"
SUITE_REPORT_HTML = "suite-report.html"
SUITE_RESULTS_NAME = "suite-results.json"

METRIC_FAMILIES = {
    "exact_match": "deterministic",
    "contains": "deterministic",
    "regex": "deterministic",
    "json_schema": "json_schema",
    "code_tests": "code_tests",
    "judge": "judge",
}
FAMILY_LABELS = {
    "deterministic": "Deterministic accuracy",
    "json_schema": "JSON Schema conformance",
    "code_tests": "Code test pass rate",
    "judge": "Judge score (mean)",
}
FAMILY_FIELDS = {
    "deterministic": ("answer_accuracy", "deterministic_trials"),
    "json_schema": ("json_schema_conformance", "json_schema_trials"),
    "code_tests": ("code_eval_pass_rate", "code_eval_trials"),
    "judge": ("judge_score_mean", "judge_evaluations"),
}


class SuiteReportError(Exception):
    pass


@dataclass
class TaskRunData:
    task_id: str
    status: str
    weight: float
    metrics: dict[str, int]
    entry_count: int
    planned_calls: int
    executed_calls: int
    reused: bool
    error: str | None
    duration_seconds: float | None
    manifest: Manifest | None = None
    stats: list[ModelStats] = field(default_factory=list)
    trials: list[TrialResult] = field(default_factory=list)

    def families(self) -> set[str]:
        return {METRIC_FAMILIES[metric] for metric in self.metrics}


@dataclass
class SuiteRunData:
    manifest: dict
    lockfile: dict
    tasks: list[TaskRunData]
    path: Path

    def models(self) -> list[str]:
        ordered = [model.get("name") for model in self.manifest.get("models", [])]
        observed = {item.model for task in self.tasks for item in task.stats}
        return ordered + sorted(observed - set(ordered))


def load_suite_run(input_dir: Path) -> SuiteRunData:
    manifest_path = input_dir / SUITE_MANIFEST_NAME
    lock_path = input_dir / LOCKFILE_NAME
    if not manifest_path.is_file():
        raise SuiteReportError(f"no {SUITE_MANIFEST_NAME} in {input_dir}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        lockfile = read_lockfile(lock_path) if lock_path.is_file() else {}
    except (OSError, ValueError) as exc:
        raise SuiteReportError(f"cannot read the suite run in {input_dir}: {exc}") from exc

    manifest_tasks = {section.get("id"): section for section in manifest.get("tasks", [])}
    tasks: list[TaskRunData] = []
    for section in lockfile.get("tasks", []):
        task_id = section.get("id")
        meta = manifest_tasks.get(task_id, {})
        task = TaskRunData(
            task_id=task_id,
            status=meta.get("status", "not_executed"),
            weight=section.get("weight", 1.0),
            metrics=section.get("metrics", {}),
            entry_count=section.get("dataset", {}).get("entry_count", 0),
            planned_calls=meta.get("planned_calls", 0),
            executed_calls=meta.get("executed_calls", 0),
            reused=meta.get("reused", False),
            error=meta.get("error"),
            duration_seconds=meta.get("duration_seconds"),
        )
        task_dir = input_dir / "tasks" / str(task_id)
        results_path = task_dir / "results.json"
        if results_path.is_file():
            from minieval_bench.reports import load_results

            task_manifest, stats, trials = load_results(task_dir)
            task.manifest = task_manifest
            task.stats = stats
            task.trials = trials
            task.executed_calls = len(trials)
        tasks.append(task)
    return SuiteRunData(manifest=manifest, lockfile=lockfile, tasks=tasks, path=input_dir)


def family_aggregates(run: SuiteRunData) -> list[dict]:
    families_present = sorted(
        {family for task in run.tasks for family in task.families()},
        key=lambda family: list(FAMILY_LABELS).index(family),
    )
    aggregates = []
    for family in families_present:
        family_tasks = [task for task in run.tasks if family in task.families()]
        rate_field, trials_field = FAMILY_FIELDS[family]
        per_model = []
        for model in run.models():
            covered: list[tuple[float, float]] = []
            trials_total = 0
            passed_total = 0.0
            observed_trials = 0
            for task in family_tasks:
                stats = next((item for item in task.stats if item.model == model), None)
                if stats is None:
                    continue
                trials = getattr(stats, trials_field) or 0
                rate = getattr(stats, rate_field)
                trials_total += trials
                if rate is not None and trials:
                    covered.append((rate, task.weight))
                    passed_total += rate * trials
                    observed_trials += trials
            per_model.append(
                {
                    "model": model,
                    "macro": weighted_mean(covered) if covered else None,
                    "micro": passed_total / observed_trials if observed_trials else None,
                    "trials": trials_total,
                    "tasksCovered": len(covered),
                    "tasksTotal": len(family_tasks),
                    "format": "rate" if family != "judge" else "score",
                }
            )
        aggregates.append(
            {
                "family": family,
                "label": FAMILY_LABELS[family],
                "tasks": len(family_tasks),
                "perModel": per_model,
            }
        )
    return aggregates


def weighted_mean(rates_and_weights: list[tuple[float, float]]) -> float:
    total_weight = sum(weight for _, weight in rates_and_weights)
    return sum(rate * weight for rate, weight in rates_and_weights) / total_weight


def consumption_totals(run: SuiteRunData) -> list[dict]:
    rows = []
    for model in run.models():
        input_tokens = 0
        output_tokens = 0
        cost = 0.0
        cost_reported = 0
        succeeded = 0
        evaluation_cost = 0.0
        evaluation_cost_reported = 0
        partial = False
        for task in run.tasks:
            stats = next((item for item in task.stats if item.model == model), None)
            if stats is None:
                continue
            succeeded += stats.succeeded
            input_tokens += stats.input_tokens or 0
            output_tokens += stats.output_tokens or 0
            if stats.cost_usd is not None:
                cost += stats.cost_usd
                cost_reported += stats.cost_reported or 0
                if (stats.cost_reported or 0) < stats.succeeded:
                    partial = True
            else:
                partial = True
            if stats.evaluation_cost_usd is not None:
                evaluation_cost += stats.evaluation_cost_usd
                evaluation_cost_reported += stats.evaluation_cost_reported or 0
        rows.append(
            {
                "model": model,
                "input_tokens": input_tokens or None,
                "output_tokens": output_tokens or None,
                "cost_usd": cost if cost_reported else None,
                "cost_partial": partial,
                "cost_reported": cost_reported or None,
                "succeeded": succeeded,
                "evaluation_cost_usd": evaluation_cost if evaluation_cost_reported else None,
                "evaluation_cost_reported": evaluation_cost_reported or None,
            }
        )
    return rows


def latency_rows(run: SuiteRunData) -> list[dict]:
    rows = []
    for task in run.tasks:
        for stats in task.stats:
            rows.append(
                {
                    "task": task.task_id,
                    "model": stats.model,
                    "median_generation_ms": stats.median_latency_ms,
                    "median_evaluation_ms": stats.median_evaluation_ms,
                    "median_total_ms": stats.median_total_ms,
                }
            )
    return rows


def suite_overview(run: SuiteRunData) -> dict:
    statuses: dict[str, int] = {}
    for task in run.tasks:
        statuses[task.status] = statuses.get(task.status, 0) + 1
    return {
        "suite": run.manifest.get("suite", {}),
        "created_at_utc": run.manifest.get("created_at_utc"),
        "models": [model.get("name") for model in run.manifest.get("models", [])],
        "task_count": len(run.tasks),
        "entry_count": sum(task.entry_count for task in run.tasks),
        "planned_calls": sum(task.planned_calls for task in run.tasks),
        "executed_calls": sum(task.executed_calls for task in run.tasks),
        "duration_seconds": run.manifest.get("duration_seconds"),
        "statuses": statuses,
        "interrupted": run.manifest.get("interrupted", False),
        "protocol_modified": run.manifest.get("protocol_modified", False),
        "sampled": run.manifest.get("sampled", False),
        "resume_used": run.manifest.get("resume_used", False),
    }


def build_suite_payload(run: SuiteRunData) -> dict:
    return {
        "kind": "suite",
        "manifest": run.manifest,
        "overview": suite_overview(run),
        "sampling": sampling_summary(run),
        "tasks": [task_payload(task) for task in run.tasks],
        "familyAggregates": family_aggregates(run),
        "latency": latency_rows(run),
        "consumption": consumption_totals(run),
    }


def sampling_summary(run: SuiteRunData) -> dict:
    sampling = {"method": "none", "selections": {}}
    for section in run.lockfile.get("tasks", []):
        record = section.get("sampling", {})
        if record.get("method", "none") != "none":
            sampling["method"] = record.get("method")
            sampling["selections"][section.get("id")] = {
                "selected": record.get("selected_count"),
                "total": section.get("dataset", {}).get("entry_count"),
                "seed": record.get("seed"),
                "selected_ids": record.get("selected_ids", []),
            }
    return sampling


def task_payload(task: TaskRunData) -> dict:
    return {
        "id": task.task_id,
        "status": task.status,
        "weight": task.weight,
        "metrics": task.metrics,
        "entryCount": task.entry_count,
        "plannedCalls": task.planned_calls,
        "executedCalls": task.executed_calls,
        "reused": task.reused,
        "error": task.error,
        "durationSeconds": task.duration_seconds,
        "stats": [item.model_dump(mode="json") for item in task.stats],
    }


def render_suite_html(run: SuiteRunData) -> str:
    return render_standalone(build_suite_payload(run), "suite.html", "suite.js")


def write_suite_reports(run: SuiteRunData, html: bool = True) -> list[Path]:
    written = []
    markdown_path = run.path / SUITE_REPORT_MD
    markdown_path.write_text(suite_markdown_report(run), encoding="utf-8")
    written.append(markdown_path)
    if html:
        html_path = run.path / SUITE_REPORT_HTML
        html_path.write_text(render_suite_html(run), encoding="utf-8")
        written.append(html_path)
    return written


def rate_cell(value: float | None) -> str:
    return "n/a" if value is None else format(value, ".1%")


def score_cell(value: float | None) -> str:
    return "n/a" if value is None else format(value, ".3f")


def family_cell(value: float | None, family: str) -> str:
    return score_cell(value) if family == "judge" else rate_cell(value)


def suite_markdown_report(run: SuiteRunData) -> str:
    overview = suite_overview(run)
    suite_block = overview["suite"]
    suite_title = f"{suite_block.get('id')} {suite_block.get('version', '')}".rstrip()
    title = f"# MiniEval suite report: {suite_title}"
    task_statuses = ", ".join(
        f"{status} x{count}" for status, count in sorted(overview["statuses"].items())
    )
    duration = cell(overview["duration_seconds"])
    lines = [
        title,
        "",
        f"- MiniEval version: {run.manifest.get('minieval_version', 'n/a')}",
        f"- Created (UTC): {overview['created_at_utc'] or 'n/a'}",
        f"- Models: {', '.join(overview['models']) or 'n/a'}",
        f"- Tasks: {overview['task_count']} ({task_statuses})",
        f"- Questions: {overview['entry_count']}",
        f"- Requests executed: {overview['executed_calls']}"
        f" / {overview['planned_calls']} planned",
        f"- Duration (s): {duration}",
        f"- License: {suite_block.get('license', 'n/a')}",
        "",
    ]
    if overview["interrupted"]:
        lines.append(
            "> The run was interrupted: task results below are partial and the suite is"
            " not fully validated.\n"
        )
    if overview["protocol_modified"]:
        overrides = run.manifest.get("generation", {}).get("user_overrides", {})
        lines.append(
            f"> Protocol modified from the official suite by user overrides: {overrides}.\n"
        )
    if overview["sampled"]:
        lines.append(
            "> Sampling was applied: scores describe a deterministic subset, not the"
            " full suite.\n"
        )

    lines += [
        "## Task breakdown",
        "",
        "| Task | Status | Weight | Metric(s) | Questions | Calls | Error |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for task in run.tasks:
        metrics = ", ".join(f"{name} x{count}" for name, count in sorted(task.metrics.items()))
        lines.append(
            f"| {task.task_id} | {task.status} | {task.weight} | {metrics or 'n/a'} "
            f"| {task.entry_count} | {task.executed_calls}/{task.planned_calls} "
            f"| {task.error or ''} |"
        )
    lines.append("")

    lines += [
        "## Model comparison per task",
        "",
        "Answer accuracy covers deterministic metrics only; each evaluator keeps its own"
        " denominator. Judge means depend on the rubric scale.",
        "",
    ]
    for task in run.tasks:
        if not task.stats:
            continue
        lines += [
            f"### {task.task_id}",
            "",
            "| Model | Success | Accuracy | JSON | Code | Judge |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for stats in task.stats:
            json_cell = (
                rate_cell(stats.json_schema_conformance)
                if stats.json_schema_trials
                else "n/a"
            )
            code_cell = (
                rate_cell(stats.code_eval_pass_rate) if stats.code_eval_trials else "n/a"
            )
            judge_cell = (
                score_cell(stats.judge_score_mean) if stats.judge_trials else "n/a"
            )
            lines.append(
                f"| {stats.model} | {rate_cell(stats.success_rate)} "
                f"| {rate_cell(stats.answer_accuracy)} "
                f"| {json_cell} | {code_cell} | {judge_cell} |"
            )
        lines.append("")

    lines += [
        "## Family aggregates",
        "",
        "Aggregates never mix metric families. Macro averages task rates with suite"
        " weights; micro pools trials. Unavailable tasks contribute no observations and"
        " no zeros.",
        "",
        "| Family | Model | Macro | Micro | Trials | Tasks covered |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for aggregate in family_aggregates(run):
        for item in aggregate["perModel"]:
            lines.append(
                f"| {aggregate['label']} | {item['model']} "
                f"| {family_cell(item['macro'], aggregate['family'])} "
                f"| {family_cell(item['micro'], aggregate['family'])} "
                f"| {item['trials']} | {item['tasksCovered']}/{aggregate['tasks']} |"
            )
    lines.append("")

    lines += [
        "## Latency",
        "",
        "Generation covers the model call (retries included), evaluation the evaluator"
        " run, total both.",
        "",
        "| Task | Model | Median gen (ms) | Median eval (ms) | Median total (ms) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in latency_rows(run):
        lines.append(
            f"| {row['task']} | {row['model']} "
            f"| {cell(row['median_generation_ms'])} "
            f"| {cell(row['median_evaluation_ms'])} "
            f"| {cell(row['median_total_ms'])} |"
        )
    lines.append("")

    lines += [
        "## Tokens and costs",
        "",
        "| Model | Input tokens | Output tokens | Generation cost (USD) | Evaluation cost (USD) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in consumption_totals(run):
        cost = "n/a" if row["cost_usd"] is None else format(row["cost_usd"], ".6f")
        if row["cost_usd"] is not None and row["cost_partial"]:
            cost += " (partial)"
        evaluation_cost = (
            "n/a" if row["evaluation_cost_usd"] is None
            else format(row["evaluation_cost_usd"], ".6f")
        )
        if row["evaluation_cost_usd"] is not None and (
            row["evaluation_cost_reported"] or 0
        ) < row["succeeded"]:
            evaluation_cost += " (partial)"
        lines.append(
            f"| {row['model']} | {int_cell(row['input_tokens'])} "
            f"| {int_cell(row['output_tokens'])} | {cost} | {evaluation_cost} |"
        )
    lines.append("")

    fingerprint = run.lockfile.get("suite", {}).get("definition_fingerprint", "n/a")
    lines += [
        "## Reproducibility",
        "",
        f"- Suite definition fingerprint: `{fingerprint}`",
        f"- Protocol lockfile: `{LOCKFILE_NAME}` beside this report"
        " (`suite verify` re-checks it)",
    ]
    sampling = sampling_summary(run)
    if sampling["method"] != "none":
        selections = "; ".join(
            f"{task_id}: {values['selected']}/{values['total']}"
            for task_id, values in sorted(sampling["selections"].items())
        )
        lines.append(f"- Sampling: {sampling['method']} ({selections})")
    lines += [
        "",
        "Protocol reproducibility covers data, parameters and evaluators. It does not"
        " bind hosted models: a provider can change server-side behaviour.",
        "",
    ]
    return "\n".join(lines)


def cell(value: float | None) -> str:
    return "n/a" if value is None else format(value, ".1f")


def int_cell(value: int | None) -> str:
    return "n/a" if value is None else format(value, ",")
