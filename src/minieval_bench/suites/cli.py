from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from minieval_bench.suites.models import SuiteError
from minieval_bench.suites.registry import (
    LoadedSuite,
    discover_suites,
    load_suite,
    validate_suite,
)

suite_app = typer.Typer(
    no_args_is_help=True,
    help="Discover, run, verify and compare benchmark suites.",
)
console = Console()

PathsOption = Annotated[
    list[Path] | None,
    typer.Option(
        "--path",
        help="Extra directory scanned for suites, in addition to the built-in and"
        " MINIEVAL_SUITE_PATH locations (repeat the option).",
    ),
]


def resolve_suite(suite_id: str, paths: list[Path] | None) -> LoadedSuite:
    try:
        return load_suite(suite_id, paths)
    except (SuiteError, OSError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc


@suite_app.command("list")
def list_suites(paths: PathsOption = None) -> None:
    """List the suites available in the registry."""
    summaries = discover_suites(paths)
    if not summaries:
        console.print(
            "No suites found. Built-in suites ship with MiniEval; add directories to"
            " the MINIEVAL_SUITE_PATH environment variable or pass --path."
        )
        return
    table = Table(title="Benchmark suites")
    table.add_column("Suite")
    table.add_column("Version")
    table.add_column("Name")
    table.add_column("Tasks", justify="right")
    table.add_column("Origin")
    table.add_column("Status")
    for summary in summaries:
        table.add_row(
            summary.suite_id,
            summary.version or "n/a",
            summary.name or "n/a",
            "n/a" if summary.task_count is None else str(summary.task_count),
            summary.origin,
            "ok" if summary.error is None else f"error: {summary.error}",
            style=None if summary.error is None else "red",
        )
    console.print(table)


@suite_app.command("info")
def info(
    suite_id: Annotated[str, typer.Argument(help="Suite identifier.")],
    paths: PathsOption = None,
) -> None:
    """Show the version, tasks, metrics, datasets and run conditions of a suite."""
    loaded = resolve_suite(suite_id, paths)
    definition = loaded.definition

    header = Table(title=f"Suite {definition.id}", show_header=False)
    header.add_column("Field", style="dim")
    header.add_column("Value")
    header.add_row("Version", definition.version)
    header.add_row("Name", definition.name)
    header.add_row("Description", definition.description)
    header.add_row("License", definition.license)
    header.add_row("Source", definition.source or "n/a")
    header.add_row("Author", definition.author or "n/a")
    header.add_row("Origin", loaded.location)
    defaults = definition.defaults
    header.add_row(
        "Run conditions",
        f"repeats {defaults.repeats}, seed {defaults.seed}, concurrency"
        f" {defaults.concurrency}, timeout {defaults.timeout}s, shuffle"
        f" {defaults.shuffle}",
    )
    console.print(header)

    tasks = Table(title="Tasks")
    tasks.add_column("Task")
    tasks.add_column("Entries", justify="right")
    tasks.add_column("Metrics")
    tasks.add_column("Weight", justify="right")
    tasks.add_column("Dataset")
    for task in definition.tasks:
        dataset = loaded.datasets[task.id]
        metrics = ", ".join(
            f"{name} x{count}" for name, count in sorted(dataset.metric_counts().items())
        )
        tasks.add_row(
            task.id,
            str(len(dataset.entries)),
            metrics,
            str(task.weight),
            f"{dataset.reference} (sha256 {dataset.sha256[:12]}…)",
        )
    console.print(tasks)

    evaluators = definition.evaluators
    if evaluators is None:
        console.print("Evaluators: deterministic metrics only.")
        return
    evaluator_rows = []
    if evaluators.judge:
        judge = evaluators.judge
        evaluator_rows.append(
            f"judge: provider {judge.provider}, model {judge.model or 'n/a'},"
            f" {judge.repeats} judgment(s) per response, rubric"
            f" {judge.rubric or 'default'}"
        )
    if evaluators.code:
        code = evaluators.code
        evaluator_rows.append(
            f"code: runtime {code.runtime}, image {code.image}, timeout"
            f" {code.timeout_seconds}s, memory {code.memory_mb} MB"
        )
    console.print("Evaluators:\n  " + "\n  ".join(evaluator_rows))


@suite_app.command("validate")
def validate(
    suite_id: Annotated[str, typer.Argument(help="Suite identifier.")],
    paths: PathsOption = None,
) -> None:
    """Check that a suite can be used: references, metrics, evaluators, licenses."""
    loaded = resolve_suite(suite_id, paths)
    errors = validate_suite(loaded)
    if errors:
        for error in errors:
            console.print(f"[red]{error}[/red]")
        console.print(f"[red]Suite '{suite_id}' is not usable: {len(errors)} error(s).[/red]")
        raise typer.Exit(1)
    total_entries = sum(len(dataset.entries) for dataset in loaded.datasets.values())
    console.print(
        f"[green]Suite '{suite_id}' is valid:[/green] {len(loaded.definition.tasks)}"
        f" task(s), {total_entries} entries, license {loaded.definition.license}."
    )
