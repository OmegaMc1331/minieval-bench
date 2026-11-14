from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from minieval_bench import __version__
from minieval_bench.config import ConfigError, DatasetError, load_config, load_dataset
from minieval_bench.models import BenchmarkConfig, DatasetEntry
from minieval_bench.reports import (
    build_manifest,
    errors_table,
    load_results,
    stats_table,
    write_reports,
)
from minieval_bench.runner import compute_stats, run_benchmark

app = typer.Typer(
    no_args_is_help=True,
    help="Run reproducible benchmarks and compare language models.",
)
console = Console()

ConfigArg = Annotated[Path, typer.Option("--config", help="Path to the benchmark YAML file.")]


def print_version(value: bool) -> None:
    if value:
        console.print(f"minieval-bench {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=print_version,
            is_eager=True,
            help="Show the MiniEval version and exit.",
        ),
    ] = False,
) -> None:
    pass


@app.command()
def validate(config: ConfigArg) -> None:
    """Validate a benchmark configuration and its dataset."""
    benchmark_config, entries = load_benchmark(config)

    metric_counts: dict[str, int] = {}
    for entry in entries:
        metric_counts[entry.metric] = metric_counts.get(entry.metric, 0) + 1

    model_count = len(benchmark_config.models)
    request_count = len(entries) * benchmark_config.defaults.repeats * model_count
    console.print(
        f"[green]Configuration is valid:[/green] {model_count} model(s), "
        f"{len(entries)} entries, {request_count} request(s) planned"
    )
    metrics_summary = ", ".join(f"{name} x{count}" for name, count in sorted(metric_counts.items()))
    console.print(f"Metrics: {metrics_summary}")
    console.print(models_table(benchmark_config))


@app.command()
def run(
    config: ConfigArg,
    output_dir: Annotated[
        Path, typer.Option("--output", help="Directory where reports are written.")
    ] = Path("results"),
) -> None:
    """Run a benchmark and write JSON and Markdown reports."""
    benchmark_config, entries = load_benchmark(config)

    try:
        trials, started_at, duration_seconds = run_benchmark(benchmark_config, entries)
    except KeyboardInterrupt:
        console.print("[red]Interrupted. No reports were written.[/red]")
        raise typer.Exit(130) from None
    stats = compute_stats(trials)
    manifest = build_manifest(benchmark_config, entries, started_at, duration_seconds)
    written = write_reports(output_dir, manifest, stats, trials)

    console.print(stats_table(stats))
    errors = errors_table(stats)
    if errors:
        console.print(errors)
    for path in written:
        console.print(f"wrote {path}")

    if stats and all(model_stats.succeeded == 0 for model_stats in stats):
        console.print(
            "[red]Every request failed. Check provider credentials and configuration.[/red]"
        )
        raise typer.Exit(1)


@app.command()
def report(
    input_dir: Annotated[
        Path, typer.Option("--input", help="Directory containing a results.json file.")
    ],
) -> None:
    """Display a report from a previous benchmark run."""
    try:
        manifest, stats = load_results(input_dir)
    except (FileNotFoundError, json.JSONDecodeError, ValidationError) as exc:
        console.print(f"[red]Cannot read report from {input_dir}: {exc}[/red]")
        raise typer.Exit(1) from exc

    manifest_table = Table(title="Run manifest", show_header=False)
    manifest_table.add_column("Field", style="dim")
    manifest_table.add_column("Value")
    manifest_table.add_row("Benchmark", manifest.benchmark)
    manifest_table.add_row("MiniEval version", manifest.minieval_version)
    manifest_table.add_row("Created (UTC)", manifest.created_at_utc)
    manifest_table.add_row("Seed", str(manifest.seed))
    manifest_table.add_row("Trial seeds", format_seeds(manifest.trial_seeds))
    manifest_table.add_row("Repeats", str(manifest.repeats))
    manifest_table.add_row("Concurrency", str(manifest.concurrency))
    manifest_table.add_row("Shuffle", "n/a" if manifest.shuffle is None else str(manifest.shuffle))
    manifest_table.add_row("Duration (s)", "n/a" if manifest.duration_seconds is None
                           else format(manifest.duration_seconds, ".1f"))
    manifest_table.add_row("Dataset", manifest.dataset_path)
    manifest_table.add_row("Dataset sha256", manifest.dataset_sha256)
    manifest_table.add_row("Entries", str(manifest.dataset_entries))

    console.print(manifest_table)
    console.print(stats_table(stats))
    errors = errors_table(stats)
    if errors:
        console.print(errors)


def load_benchmark(config_path: Path) -> tuple[BenchmarkConfig, list[DatasetEntry]]:
    try:
        benchmark_config = load_config(config_path)
        entries = load_dataset(Path(benchmark_config.dataset))
    except (ConfigError, DatasetError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    return benchmark_config, entries


def format_seeds(trial_seeds: list[int] | None) -> str:
    if trial_seeds is None:
        return "n/a"
    return ", ".join(str(seed) for seed in trial_seeds)


def models_table(benchmark_config: BenchmarkConfig) -> Table:
    table = Table(title="Models")
    table.add_column("Name")
    table.add_column("Provider")
    table.add_column("Model string")
    table.add_column("Temperature", justify="right")
    table.add_column("Max tokens", justify="right")
    table.add_column("Timeout (s)", justify="right")
    table.add_column("Cost / 1M tokens (in / out)", justify="right")
    for model in benchmark_config.models:
        cost = (
            f"{model.input_cost_per_1m} / {model.output_cost_per_1m}"
            if model.input_cost_per_1m is not None and model.output_cost_per_1m is not None
            else "n/a"
        )
        table.add_row(
            model.name,
            model.provider,
            model.model or "",
            "n/a" if model.temperature is None else str(model.temperature),
            "n/a" if model.max_tokens is None else str(model.max_tokens),
            "n/a" if model.timeout is None else str(model.timeout),
            cost,
        )
    return table


if __name__ == "__main__":
    app()
