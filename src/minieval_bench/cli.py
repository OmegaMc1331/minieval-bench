from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from minieval_bench import __version__
from minieval_bench.comparison import build_comparison_payload
from minieval_bench.config import ConfigError, DatasetError, load_config, load_dataset
from minieval_bench.exports import write_trials_csv
from minieval_bench.history import find_runs
from minieval_bench.html_report import render_standalone, write_html_report
from minieval_bench.models import BenchmarkConfig, DatasetEntry
from minieval_bench.reports import (
    build_manifest,
    errors_table,
    load_dataset_snapshot,
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
    no_html: Annotated[
        bool, typer.Option("--no-html", help="Skip the interactive HTML report.")
    ] = False,
    snapshot: Annotated[
        bool,
        typer.Option(
            "--snapshot",
            help="Embed the dataset (prompts and references) in results.json and the HTML report.",
        ),
    ] = False,
) -> None:
    """Run a benchmark and write JSON, Markdown and HTML reports."""
    benchmark_config, entries = load_benchmark(config)

    try:
        trials, started_at, duration_seconds = run_benchmark(benchmark_config, entries)
    except KeyboardInterrupt:
        console.print("[red]Interrupted. No reports were written.[/red]")
        raise typer.Exit(130) from None
    stats = compute_stats(trials)
    manifest = build_manifest(benchmark_config, entries, started_at, duration_seconds)
    written = write_reports(
        output_dir,
        manifest,
        stats,
        trials,
        dataset=entries if snapshot else None,
        html=not no_html,
    )

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
    report_format: Annotated[
        str,
        typer.Option("--format", help="Output format: terminal or html."),
    ] = "terminal",
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Path of the HTML file when --format html is used."),
    ] = None,
) -> None:
    """Display a report from a previous benchmark run."""
    if report_format not in ("terminal", "html"):
        console.print(f"[red]Unknown format '{report_format}': expected terminal or html.[/red]")
        raise typer.Exit(1)

    try:
        manifest, stats, trials = load_results(input_dir)
    except (FileNotFoundError, json.JSONDecodeError, ValidationError) as exc:
        console.print(f"[red]Cannot read report from {input_dir}: {exc}[/red]")
        raise typer.Exit(1) from exc

    if report_format == "html":
        output_path = output or input_dir / "report.html"
        dataset = load_dataset_snapshot(input_dir)
        write_html_report(output_path, manifest, stats, trials, dataset)
        console.print(f"wrote {output_path}")
        return

    console.print(manifest_table(manifest))
    console.print(stats_table(stats))
    errors = errors_table(stats)
    if errors:
        console.print(errors)


@app.command()
def compare(
    runs_arg: Annotated[
        list[Path],
        typer.Option("--run", help="Results directory of a benchmark run (repeat the option)."),
    ],
    output: Annotated[
        Path, typer.Option("--output", help="Path of the comparison HTML report.")
    ] = Path("comparison.html"),
) -> None:
    """Compare two or more benchmark runs in a standalone HTML report."""
    if len(runs_arg) < 2:
        console.print("[red]compare needs at least two --run directories.[/red]")
        raise typer.Exit(1)

    try:
        payload = build_comparison_payload(runs_arg)
    except (FileNotFoundError, json.JSONDecodeError, ValidationError) as exc:
        console.print(f"[red]Cannot compare runs: {exc}[/red]")
        raise typer.Exit(1) from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_standalone(payload, "compare.html", "compare.js"), encoding="utf-8"
    )
    for warning in payload["warnings"]:
        console.print(f"[yellow]{warning}[/yellow]")
    console.print(f"wrote {output}")


@app.command()
def export(
    input_dir: Annotated[
        Path, typer.Option("--input", help="Directory containing a results.json file.")
    ],
    export_format: Annotated[
        str, typer.Option("--format", help="Export format: csv.")
    ] = "csv",
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Path of the CSV file (defaults to trials.csv in --input)."),
    ] = None,
) -> None:
    """Export individual trial results to CSV."""
    if export_format != "csv":
        console.print(f"[red]Unsupported format '{export_format}': only csv is available.[/red]")
        raise typer.Exit(1)

    try:
        _, _, trials = load_results(input_dir)
    except (FileNotFoundError, json.JSONDecodeError, ValidationError) as exc:
        console.print(f"[red]Cannot export from {input_dir}: {exc}[/red]")
        raise typer.Exit(1) from exc

    output_path = output or input_dir / "trials.csv"
    row_count = write_trials_csv(output_path, trials)
    console.print(f"wrote {output_path} ({row_count} trials)")


@app.command()
def history(
    root: Annotated[
        Path, typer.Option("--root", help="Directory scanned recursively for benchmark runs.")
    ] = Path("results"),
) -> None:
    """List benchmark runs found under a directory."""
    records = find_runs(root)
    if not records:
        console.print(f"No benchmark runs found under {root}")
        return

    table = Table(title=f"Benchmark runs under {root}")
    table.add_column("Benchmark")
    table.add_column("Executed (UTC)")
    table.add_column("Models", justify="right")
    table.add_column("Planned requests", justify="right")
    table.add_column("Version")
    table.add_column("Location")
    for record in records:
        table.add_row(
            record.benchmark,
            record.created_at_utc,
            str(record.model_count),
            str(record.planned_requests),
            record.minieval_version,
            str(record.path),
        )
    console.print(table)


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


def manifest_table(manifest) -> Table:
    table = Table(title="Run manifest", show_header=False)
    table.add_column("Field", style="dim")
    table.add_column("Value")
    table.add_row("Benchmark", manifest.benchmark)
    table.add_row("MiniEval version", manifest.minieval_version)
    table.add_row("Created (UTC)", manifest.created_at_utc)
    table.add_row("Seed", str(manifest.seed))
    table.add_row("Trial seeds", format_seeds(manifest.trial_seeds))
    table.add_row("Repeats", str(manifest.repeats))
    table.add_row("Concurrency", str(manifest.concurrency))
    table.add_row("Shuffle", "n/a" if manifest.shuffle is None else str(manifest.shuffle))
    table.add_row(
        "Duration (s)",
        "n/a" if manifest.duration_seconds is None else format(manifest.duration_seconds, ".1f"),
    )
    table.add_row("Dataset", manifest.dataset_path)
    table.add_row("Dataset sha256", manifest.dataset_sha256)
    table.add_row("Entries", str(manifest.dataset_entries))
    return table


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
