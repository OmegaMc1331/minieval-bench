from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from rich.table import Table

from minieval_bench import __version__
from minieval_bench.evaluation.judge import DEFAULT_RUBRIC, load_rubric
from minieval_bench.html_report import build_analysis, write_html_report
from minieval_bench.models import (
    BenchmarkConfig,
    DatasetEntry,
    Manifest,
    ModelStats,
    TrialResult,
)
from minieval_bench.runner import repeat_seed


def dataset_fingerprint(dataset_path: Path) -> str:
    return hashlib.sha256(dataset_path.read_bytes()).hexdigest()


def entry_fingerprint(entry: DatasetEntry) -> str:
    canonical = json.dumps(
        {
            "prompt": entry.prompt,
            "reference": entry.reference,
            "system": entry.system,
            "metric": entry.metric,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def judge_manifest_block(config: BenchmarkConfig) -> dict | None:
    judge_config = config.evaluators.judge if config.evaluators else None
    if judge_config is None:
        return None
    rubric = DEFAULT_RUBRIC
    if judge_config.rubric:
        rubric = load_rubric(Path(judge_config.rubric))
    return {
        "provider": judge_config.provider,
        "model": judge_config.model,
        "repeats": judge_config.repeats,
        "rubric_fingerprint": rubric.fingerprint(),
        "rubric_version": rubric.version,
        "criteria": [
            {
                "id": criterion.id,
                "scale_min": criterion.scale_min,
                "scale_max": criterion.scale_max,
                "weight": criterion.weight,
            }
            for criterion in rubric.criteria
        ],
    }


def build_manifest(
    config: BenchmarkConfig,
    entries: list[DatasetEntry],
    started_at: datetime,
    duration_seconds: float,
) -> Manifest:
    return Manifest(
        minieval_version=__version__,
        benchmark=config.benchmark,
        created_at_utc=started_at.isoformat(),
        seed=config.defaults.seed,
        repeats=config.defaults.repeats,
        concurrency=config.defaults.concurrency,
        dataset_path=config.dataset,
        dataset_sha256=dataset_fingerprint(Path(config.dataset)),
        dataset_entries=len(entries),
        models=config.models,
        duration_seconds=duration_seconds,
        trial_seeds=[
            repeat_seed(config.defaults.seed, repeat)
            for repeat in range(config.defaults.repeats)
        ],
        shuffle=config.defaults.shuffle,
        evaluators=config.evaluators.model_dump(mode="json") if config.evaluators else None,
        statistics=config.statistics.model_dump(mode="json"),
        judge=judge_manifest_block(config),
    )


def write_reports(
    output_dir: Path,
    manifest: Manifest,
    stats: list[ModelStats],
    trials: list[TrialResult],
    dataset: list[DatasetEntry] | None = None,
    html: bool = True,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    results_path = output_dir / "results.json"
    report_path = output_dir / "report.md"

    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    statistics, evaluator_summaries, evaluator_by_model = build_analysis(manifest, trials)
    payload = {
        "manifest": manifest.model_dump(mode="json"),
        "stats": [item.model_dump(mode="json") for item in stats],
        "trials": [trial.model_dump(mode="json") for trial in trials],
        "statistics": statistics,
        "evaluators": evaluator_summaries,
        "evaluatorsByModel": evaluator_by_model,
    }
    if dataset is not None:
        payload["dataset"] = [entry.model_dump(mode="json") for entry in dataset]
    results_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(markdown_report(manifest, stats), encoding="utf-8")
    written = [manifest_path, results_path, report_path]
    if html:
        html_path = output_dir / "report.html"
        write_html_report(html_path, manifest, stats, trials, dataset)
        written.append(html_path)
    return written


def load_results(
    input_dir: Path,
) -> tuple[Manifest, list[ModelStats], list[TrialResult]]:
    results_path = input_dir / "results.json"
    if not results_path.is_file():
        raise FileNotFoundError(f"no results.json in {input_dir}")
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    manifest = Manifest.model_validate(payload["manifest"])
    stats = [ModelStats.model_validate(item) for item in payload["stats"]]
    trials = [TrialResult.model_validate(item) for item in payload.get("trials", [])]
    return manifest, stats, trials


def load_dataset_snapshot(input_dir: Path) -> list[DatasetEntry] | None:
    results_path = input_dir / "results.json"
    if not results_path.is_file():
        return None
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    if payload.get("dataset") is None:
        return None
    return [DatasetEntry.model_validate(item) for item in payload["dataset"]]


def match_dataset_entries(
    source_manifest: Manifest,
    snapshot: list[DatasetEntry] | None,
    dataset_identical: bool,
    entries: list[DatasetEntry],
) -> dict:
    matching = {
        "basis": "dataset-identical",
        "verified": {entry.id for entry in entries},
        "unverified": set(),
        "conflicts": {},
    }
    if dataset_identical:
        return matching
    if snapshot is None:
        return {
            "basis": "entry-id-only",
            "verified": set(),
            "unverified": {entry.id for entry in entries},
            "conflicts": {},
        }
    old_by_id = {entry.id: entry for entry in snapshot}
    verified = set()
    unverified = set()
    conflicts = {}
    for entry in entries:
        old_entry = old_by_id.get(entry.id)
        if old_entry is None:
            unverified.add(entry.id)
        elif entry_fingerprint(old_entry) != entry_fingerprint(entry):
            conflicts[entry.id] = "prompt, reference, system or metric changed since the source run"
        else:
            verified.add(entry.id)
    return {
        "basis": "snapshot-fingerprints",
        "verified": verified,
        "unverified": unverified,
        "conflicts": conflicts,
    }


def markdown_report(manifest: Manifest, stats: list[ModelStats]) -> str:
    lines = [
        f"# MiniEval report: {manifest.benchmark}",
        "",
        f"- MiniEval version: {manifest.minieval_version}",
        f"- Created (UTC): {manifest.created_at_utc}",
        f"- Seed: {manifest.seed}",
        f"- Repeats: {manifest.repeats}",
        f"- Concurrency: {manifest.concurrency}",
        f"- Shuffle: {manifest.shuffle if manifest.shuffle is not None else 'n/a'}",
        f"- Duration (s): {cell(manifest.duration_seconds, '.1f')}",
        f"- Dataset: `{manifest.dataset_path}` "
        f"({manifest.dataset_entries} entries, sha256 `{manifest.dataset_sha256}`)",
        "",
        "## Results",
        "",
        "| Model | Requests | OK | Failed | Success | Accuracy | Coverage | Overall |"
        " Median gen (ms) | P95 gen (ms) | Median eval (ms) | Median total (ms) |"
        " Queue (ms) | In tokens | Out tokens | Cost (USD) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
        " --- | --- | --- | --- | --- |",
    ]
    for model_stats in stats:
        cells = [
            model_stats.model,
            str(model_stats.requests),
            str(model_stats.succeeded),
            str(model_stats.failed),
            rate(model_stats.success_rate),
            rate(model_stats.answer_accuracy),
            rate(model_stats.evaluation_coverage),
            rate(model_stats.overall_pass_rate),
            cell(model_stats.median_latency_ms, ".1f"),
            cell(model_stats.p95_latency_ms, ".1f"),
            cell(model_stats.median_evaluation_ms, ".1f"),
            cell(model_stats.median_total_ms, ".1f"),
            cell(model_stats.median_queue_ms, ".1f"),
            consumption_cell(
                model_stats.input_tokens,
                model_stats.input_tokens_reported,
                model_stats.succeeded,
                ",",
            ),
            consumption_cell(
                model_stats.output_tokens,
                model_stats.output_tokens_reported,
                model_stats.succeeded,
                ",",
            ),
            consumption_cell(
                model_stats.cost_usd, model_stats.cost_reported, model_stats.succeeded, ".6f"
            ),
        ]
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        "Indicators, with their explicit denominators:",
        "",
        "- **Success** = requests completed without provider error / planned requests.",
        "- **Accuracy** = correct answers / evaluable answers (successful requests).",
        "- **Coverage** = evaluated trials / planned requests.",
        "- **Overall** = correct answers / planned requests; failed requests count as not passed.",
        "",
        "Latency columns measure the model call only (provider retries included)."
        " Evaluation latency measures the evaluator run and total latency covers"
        " generation plus evaluation. Reports written before 0.3.1 stored a single"
        " `latency_ms` mixing both; those values are shown as-is.",
        "",
        "Token and cost totals are sums over the requests that reported them; "
        "`value (r/s)` marks a partial total covering r of s successful requests. "
        "`n/a` marks a value that is unavailable, as opposed to a real zero.",
        "",
    ]

    evaluation_rows = [
        model_stats
        for model_stats in stats
        if model_stats.evaluator_failed
        or model_stats.evaluator_unavailable
        or model_stats.evaluation_cost_usd is not None
    ]
    if evaluation_rows:
        lines += [
            "## Evaluation overhead",
            "",
            "| Model | Evaluator failures | Unavailable | Evaluation cost (USD) |",
            "| --- | --- | --- | --- |",
        ]
        for model_stats in evaluation_rows:
            lines.append(
                f"| {model_stats.model} | {model_stats.evaluator_failed} "
                f"| {model_stats.evaluator_unavailable} "
                f"| {cell(model_stats.evaluation_cost_usd, '.6f')} |"
            )
        lines.append("")

    breakdown_rows = [
        model_stats
        for model_stats in stats
        if model_stats.json_schema_trials
        or model_stats.code_eval_trials
        or model_stats.judge_trials
        or model_stats.evaluator_failed
        or model_stats.evaluator_unavailable
    ]
    if breakdown_rows:
        lines += [
            "## Evaluation breakdown",
            "",
            "Accuracy above covers deterministic metrics only. Each evaluator below has"
            " its own denominator (binary trials of that evaluator).",
            "",
            "| Model | Deterministic | Answer accuracy | JSON conformance | Code pass"
            " | Code tests | Judge score | Judge judged/eligible |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for model_stats in breakdown_rows:
            json_cell = rate_with_count(
                model_stats.json_schema_conformance, model_stats.json_schema_trials
            )
            code_cell = rate_with_count(
                model_stats.code_eval_pass_rate, model_stats.code_eval_trials
            )
            judged = count_cell(model_stats.judge_evaluations)
            eligible = count_cell(model_stats.judge_trials)
            judge_cell = f"{judged} / {eligible}"
            lines.append(
                f"| {model_stats.model} "
                f"| {count_cell(model_stats.deterministic_trials)} "
                f"| {rate(model_stats.answer_accuracy)} "
                f"| {json_cell} "
                f"| {code_cell} "
                f"| {tests_cell(model_stats)} "
                f"| {score_cell(model_stats.judge_score_mean)} "
                f"| {judge_cell} |"
            )
        lines.append("")

    error_rows = [
        (model_stats.model, error, str(count))
        for model_stats in stats
        for error, count in sorted(model_stats.errors.items())
    ]
    if error_rows:
        lines += ["## Errors", "", "| Model | Error | Count |", "| --- | --- | --- |"]
        lines += ["| " + " | ".join(row) + " |" for row in error_rows]
        lines.append("")

    return "\n".join(lines)


def cell(value: int | float | None, format_spec: str) -> str:
    return "n/a" if value is None else format(value, format_spec)


def rate(value: float | None) -> str:
    return "n/a" if value is None else format(value, ".1%")


def count_cell(value: int | None) -> str:
    return "0" if value == 0 else ("n/a" if value is None else str(value))


def rate_with_count(value: float | None, trials: int | None) -> str:
    if value is None:
        return "n/a (0)" if trials == 0 else "n/a"
    return f"{format(value, '.1%')} ({trials})"


def tests_cell(model_stats: ModelStats) -> str:
    if model_stats.code_tests_total is None:
        return "n/a"
    return f"{model_stats.code_tests_passed}/{model_stats.code_tests_total}"


def score_cell(value: float | None) -> str:
    return "n/a" if value is None else format(value, ".3f")


def consumption_cell(
    value: int | float | None, reported: int | None, succeeded: int, format_spec: str
) -> str:
    if value is None:
        return "n/a"
    if reported is None or reported >= succeeded:
        return format(value, format_spec)
    return f"{format(value, format_spec)} ({reported}/{succeeded})"


def stats_table(stats: list[ModelStats]) -> Table:
    table = Table(title="Benchmark results")
    table.add_column("Model")
    table.add_column("Requests", justify="right")
    table.add_column("OK", justify="right")
    table.add_column("Failed", justify="right")
    table.add_column("Success", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("Coverage", justify="right")
    table.add_column("Overall", justify="right")
    table.add_column("Median gen (ms)", justify="right")
    table.add_column("P95 gen (ms)", justify="right")
    table.add_column("Median eval (ms)", justify="right")
    table.add_column("Median total (ms)", justify="right")
    table.add_column("Queue (ms)", justify="right")
    table.add_column("In tokens", justify="right")
    table.add_column("Out tokens", justify="right")
    table.add_column("Cost (USD)", justify="right")
    for model_stats in stats:
        table.add_row(
            model_stats.model,
            str(model_stats.requests),
            str(model_stats.succeeded),
            str(model_stats.failed),
            rate(model_stats.success_rate),
            rate(model_stats.answer_accuracy),
            rate(model_stats.evaluation_coverage),
            rate(model_stats.overall_pass_rate),
            cell(model_stats.median_latency_ms, ".1f"),
            cell(model_stats.p95_latency_ms, ".1f"),
            cell(model_stats.median_evaluation_ms, ".1f"),
            cell(model_stats.median_total_ms, ".1f"),
            cell(model_stats.median_queue_ms, ".1f"),
            consumption_cell(
                model_stats.input_tokens,
                model_stats.input_tokens_reported,
                model_stats.succeeded,
                ",",
            ),
            consumption_cell(
                model_stats.output_tokens,
                model_stats.output_tokens_reported,
                model_stats.succeeded,
                ",",
            ),
            consumption_cell(
                model_stats.cost_usd, model_stats.cost_reported, model_stats.succeeded, ".6f"
            ),
        )
    return table


def evaluation_breakdown_table(stats: list[ModelStats]) -> Table | None:
    rows = [
        model_stats
        for model_stats in stats
        if model_stats.json_schema_trials
        or model_stats.code_eval_trials
        or model_stats.judge_trials
        or model_stats.evaluator_failed
        or model_stats.evaluator_unavailable
    ]
    if not rows:
        return None
    table = Table(title="Evaluation breakdown (per evaluator)")
    table.add_column("Model")
    table.add_column("Deterministic", justify="right")
    table.add_column("Answer accuracy", justify="right")
    table.add_column("JSON conformance", justify="right")
    table.add_column("Code pass", justify="right")
    table.add_column("Code tests", justify="right")
    table.add_column("Judge score", justify="right")
    table.add_column("Judge judged/eligible", justify="right")
    for model_stats in rows:
        table.add_row(
            model_stats.model,
            count_cell(model_stats.deterministic_trials),
            rate(model_stats.answer_accuracy),
            rate_with_count(model_stats.json_schema_conformance, model_stats.json_schema_trials),
            rate_with_count(model_stats.code_eval_pass_rate, model_stats.code_eval_trials),
            tests_cell(model_stats),
            score_cell(model_stats.judge_score_mean),
            f"{count_cell(model_stats.judge_evaluations)} / {count_cell(model_stats.judge_trials)}",
        )
    return table


def errors_table(stats: list[ModelStats]) -> Table | None:
    rows = [
        (model_stats.model, error, str(count))
        for model_stats in stats
        for error, count in sorted(model_stats.errors.items())
    ]
    if not rows:
        return None
    table = Table(title="Errors")
    table.add_column("Model")
    table.add_column("Error")
    table.add_column("Count", justify="right")
    for row in rows:
        table.add_row(*row)
    return table
