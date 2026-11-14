from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from rich.table import Table

from minieval_bench import __version__
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
    )


def write_reports(
    output_dir: Path, manifest: Manifest, stats: list[ModelStats], trials: list[TrialResult]
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    results_path = output_dir / "results.json"
    report_path = output_dir / "report.md"

    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    payload = {
        "manifest": manifest.model_dump(mode="json"),
        "stats": [item.model_dump(mode="json") for item in stats],
        "trials": [trial.model_dump(mode="json") for trial in trials],
    }
    results_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(markdown_report(manifest, stats), encoding="utf-8")
    return [manifest_path, results_path, report_path]


def load_results(input_dir: Path) -> tuple[Manifest, list[ModelStats]]:
    results_path = input_dir / "results.json"
    if not results_path.is_file():
        raise FileNotFoundError(f"no results.json in {input_dir}")
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    manifest = Manifest.model_validate(payload["manifest"])
    stats = [ModelStats.model_validate(item) for item in payload["stats"]]
    return manifest, stats


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
        " Median (ms) | P95 (ms) | Queue (ms) | In tokens | Out tokens | Cost (USD) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
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
        "Token and cost totals are sums over the requests that reported them; "
        "`value (r/s)` marks a partial total covering r of s successful requests. "
        "`n/a` marks an unavailable value, as opposed to a real zero.",
        "",
    ]

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
    table.add_column("Median (ms)", justify="right")
    table.add_column("P95 (ms)", justify="right")
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
