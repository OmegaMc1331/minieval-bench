from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from minieval_bench.models import ModelConfig
from minieval_bench.suites.lockfile import (
    LOCKFILE_NAME,
    build_lockfile,
    lockfile_differences,
    read_lockfile,
)
from minieval_bench.suites.models import SuiteError
from minieval_bench.suites.registry import (
    LoadedSuite,
    discover_suites,
    load_suite,
    validate_suite,
)
from minieval_bench.suites.reports import (
    SuiteReportError,
    load_suite_run,
    render_suite_html,
    suite_markdown_report,
    write_suite_reports,
)
from minieval_bench.suites.runner import (
    SUITE_MANIFEST_NAME,
    SuiteRunError,
    SuiteRunOptions,
    UserOverrides,
    dry_run_details,
    preflight_problems,
    prepare_run,
    prepare_run_with_models,
    run_suite,
)
from minieval_bench.suites.sampling import SamplingPlan

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


@suite_app.command("run")
def run(
    suite_id: Annotated[str, typer.Argument(help="Suite identifier.")],
    config: Annotated[
        Path, typer.Option("--config", help="YAML file listing the candidate models.")
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option("--output", help="Directory for suite results (default results/<id>)."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print the execution plan without calling any provider."),
    ] = False,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume",
            help="Reuse stored task results whose protocol matches the lockfile.",
        ),
    ] = False,
    max_calls: Annotated[
        int | None,
        typer.Option("--max-calls", help="Refuse to plan more candidate calls than this."),
    ] = None,
    allow_code_execution: Annotated[
        bool,
        typer.Option(
            "--allow-code-execution",
            help="Enable code_tests tasks, executed inside the restricted Docker runtime.",
        ),
    ] = False,
    repeats: Annotated[int | None, typer.Option(help="Override the suite repeats.")] = None,
    seed: Annotated[int | None, typer.Option(help="Override the suite seed.")] = None,
    concurrency: Annotated[
        int | None, typer.Option(help="Override the suite concurrency.")
    ] = None,
    timeout: Annotated[float | None, typer.Option(help="Override the request timeout (s).")] = None,
    max_tokens: Annotated[
        int | None, typer.Option(help="Override the maximum response tokens.")
    ] = None,
    temperature: Annotated[
        float | None, typer.Option(help="Override the generation temperature.")
    ] = None,
    shuffle: Annotated[
        bool | None,
        typer.Option("--shuffle/--no-shuffle", help="Override the suite shuffle setting."),
    ] = None,
    sample: Annotated[
        int | None,
        typer.Option("--sample", help="Run on a deterministic subset of N entries per task."),
    ] = None,
    sample_seed: Annotated[
        int | None, typer.Option("--sample-seed", help="Seed of the subset selection.")
    ] = None,
    sample_mode: Annotated[
        str,
        typer.Option("--sample-mode", help="Subset selection mode: random or stratified."),
    ] = "random",
    paths: PathsOption = None,
) -> None:
    """Run every task of a suite over the configured candidate models."""
    loaded = resolve_suite(suite_id, paths)
    errors = validate_suite(loaded)
    if errors:
        for error in errors:
            console.print(f"[red]{error}[/red]")
        raise typer.Exit(1)

    output = output_dir or Path("results") / loaded.definition.id
    options = SuiteRunOptions(
        models_path=config,
        output_dir=output,
        overrides=UserOverrides(
            repeats=repeats,
            seed=seed,
            concurrency=concurrency,
            timeout=timeout,
            max_tokens=max_tokens,
            temperature=temperature,
            shuffle=shuffle,
        ),
        sampling=SamplingPlan(
            method="none" if sample is None else sample_mode,
            count=sample,
            seed=sample_seed,
        ),
        resume=resume,
        max_calls=max_calls,
        allow_code_execution=allow_code_execution,
    )

    try:
        plans, models, effective_defaults, user_overrides = prepare_run(loaded, options)
    except SuiteRunError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    problems = preflight_problems(loaded, plans, options)
    if problems:
        for problem in problems:
            console.print(f"[red]{problem}[/red]")
        raise typer.Exit(1)

    if dry_run:
        print_dry_run(loaded, plans, models, effective_defaults, user_overrides, options)
        return

    outcome = run_suite(loaded, options)

    table = Table(title=f"Suite run: {loaded.definition.id}")
    table.add_column("Task")
    table.add_column("Status")
    table.add_column("Calls", justify="right")
    table.add_column("Reused")
    table.add_column("Detail")
    for result in outcome.tasks:
        table.add_row(
            result.task_id,
            status_style(result.status),
            f"{result.executed_calls}/{result.planned_calls}",
            "yes" if result.reused else "no",
            result.error or "",
        )
    console.print(table)
    written_reports = []
    try:
        written_reports = write_suite_reports(load_suite_run(output))
    except SuiteReportError as exc:
        console.print(f"[yellow]suite report could not be written: {exc}[/yellow]")
    for path in (
        output / SUITE_MANIFEST_NAME,
        output / LOCKFILE_NAME,
        output / "suite-results.json",
        *written_reports,
    ):
        console.print(f"wrote {path}")
    if outcome.interrupted:
        console.print(
            "[red]Interrupted: completed tasks were kept, the interrupted task and the "
            "remaining ones were marked not_executed.[/red]"
        )
        raise typer.Exit(130)


@suite_app.command("report")
def report(
    input_dir: Annotated[
        Path, typer.Option("--input", help="Directory of a suite run.")
    ],
    report_format: Annotated[
        str, typer.Option("--format", help="Output format: terminal or html.")
    ] = "terminal",
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Path of the HTML file when --format html is used."),
    ] = None,
) -> None:
    """Display or regenerate the global report of a suite run."""
    if report_format not in ("terminal", "html"):
        console.print(f"[red]Unknown format '{report_format}': expected terminal or html.[/red]")
        raise typer.Exit(1)
    try:
        run = load_suite_run(input_dir)
    except SuiteReportError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    if report_format == "html":
        target = output or input_dir / "suite-report.html"
        target.write_text(
            render_suite_html(run), encoding="utf-8"
        )
        console.print(f"wrote {target}")
        return

    from rich.markdown import Markdown

    console.print(Markdown(suite_markdown_report(run)))


@suite_app.command("verify")
def verify(
    input_dir: Annotated[
        Path, typer.Argument(help="Directory of a suite run to verify.")
    ],
    paths: PathsOption = None,
) -> None:
    """Check a suite run against the current suite, datasets and protocol."""
    manifest_path = input_dir / SUITE_MANIFEST_NAME
    lock_path = input_dir / LOCKFILE_NAME
    for path in (manifest_path, lock_path):
        if not path.is_file():
            console.print(f"[red]missing {path.name} in {input_dir}[/red]")
            raise typer.Exit(1)
    try:
        stored_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stored_lock = read_lockfile(lock_path)
    except ValueError as exc:
        console.print(f"[red]cannot read the run artefacts: {exc}[/red]")
        raise typer.Exit(1) from exc

    suite_block = stored_manifest.get("suite", {})
    suite_id = suite_block.get("id")
    try:
        loaded = load_suite(suite_id, paths)
    except (SuiteError, OSError) as exc:
        console.print(
            f"[red]suite '{suite_id}' is not available in the registry: {exc}[/red]"
        )
        raise typer.Exit(1) from exc

    generation = stored_manifest.get("generation", {})
    try:
        models = [ModelConfig.model_validate(item) for item in generation.get("models", [])]
    except Exception as exc:
        console.print(f"[red]stored model configuration is unreadable: {exc}[/red]")
        raise typer.Exit(1) from exc
    options = SuiteRunOptions(
        models_path=Path("unused"),
        output_dir=input_dir,
        overrides=UserOverrides(**(generation.get("user_overrides") or {})),
        sampling=SamplingPlan(**(stored_manifest.get("sampling") or {"method": "none"})),
    )
    try:
        plans, models, effective_defaults, user_overrides = prepare_run_with_models(
            loaded, options, models
        )
    except SuiteRunError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    current_lock = build_lockfile(
        loaded,
        models,
        effective_defaults,
        user_overrides,
        [plan.lock_section for plan in plans],
    )

    differences = lockfile_differences(stored_lock, current_lock)
    if differences:
        for difference in differences:
            console.print(f"[red]{difference}[/red]")
        console.print(
            f"[red]{len(differences)} difference(s): the run does not match the current"
            " protocol.[/red]"
        )
        raise typer.Exit(1)
    console.print(
        f"[green]Run in {input_dir} matches the current protocol of suite"
        f" '{suite_id}' {loaded.definition.version}.[/green]"
    )
    console.print(
        "This verifies the protocol and the data, not that a hosted model behaves"
        " identically: provider-side changes are outside MiniEval's control."
    )


def status_style(status: str) -> str:
    if status == "completed":
        return f"[green]{status}[/green]"
    if status in ("unavailable", "infrastructure_failed", "not_executed"):
        return f"[red]{status}[/red]"
    return f"[yellow]{status}[/yellow]"


def print_dry_run(loaded, plans, models, effective_defaults, user_overrides, options) -> None:
    definition = loaded.definition
    console.print(
        f"Dry run for suite [bold]{definition.id}[/bold] {definition.version}"
        f" ({len(plans)} task(s), {len(models)} model(s))"
    )
    table = Table(title="Execution plan")
    table.add_column("Task")
    table.add_column("Entries", justify="right")
    table.add_column("Repeats", justify="right")
    table.add_column("Calls", justify="right")
    table.add_column("Judge calls", justify="right")
    table.add_column("Metrics")
    for plan in plans:
        metrics = ", ".join(
            f"{name} x{count}"
            for name, count in sorted(plan.lock_section["metrics"].items())
        )
        table.add_row(
            plan.task.id,
            str(len(plan.entries)),
            str(plan.config.defaults.repeats),
            str(plan.planned_calls),
            str(plan.planned_judge_calls or "-"),
            metrics,
        )
    console.print(table)
    total_calls = sum(plan.planned_calls for plan in plans)
    total_judge = sum(plan.planned_judge_calls for plan in plans)
    console.print(
        f"Conditions: repeats {effective_defaults.repeats}, seed"
        f" {effective_defaults.seed}, concurrency {effective_defaults.concurrency},"
        f" timeout {effective_defaults.timeout}s, shuffle {effective_defaults.shuffle}"
    )
    console.print(f"Planned candidate calls: {total_calls}; judge calls: {total_judge}")

    details = dry_run_details(plans)
    if details["docker_required"]:
        availability = "available" if details["docker_available"] else "not available"
        console.print(
            f"[yellow]Docker runtime required for code_tests tasks; daemon is"
            f" {availability}.[/yellow]"
        )
        if not options.allow_code_execution:
            console.print(
                "[yellow]code_tests tasks stay unavailable without"
                " --allow-code-execution.[/yellow]"
            )
    cost_rows = []
    for task_id, by_model in details["max_cost_upper_bounds"].items():
        for model_name, bound in by_model.items():
            cost_rows.append((task_id, model_name, bound))
    if any(bound is not None for _, _, bound in cost_rows):
        cost_table = Table(title="Cost upper bounds (heuristic, only with configured pricing)")
        cost_table.add_column("Task")
        cost_table.add_column("Model")
        cost_table.add_column("Max cost (USD)", justify="right")
        for task_id, model_name, bound in cost_rows:
            cost_table.add_row(task_id, model_name, "n/a" if bound is None else f"{bound:.4f}")
        console.print(cost_table)
        console.print(
            "Bounds assume every response reaches max_tokens; actual costs are usually"
            " lower. n/a marks unknown pricing or an unset max_tokens."
        )

    if user_overrides:
        console.print(
            f"[yellow]User overrides: {user_overrides}; the run is not strictly the"
            " official suite protocol.[/yellow]"
        )
    if options.sampling.method != "none":
        console.print(
            f"[yellow]Sampling: {options.sampling.method}, {options.sampling.count}"
            " entries per task; results describe this deterministic subset, not the"
            " full suite.[/yellow]"
        )
    console.print("Dry run: nothing was executed and nothing was written.")
