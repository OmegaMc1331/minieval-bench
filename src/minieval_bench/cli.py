from __future__ import annotations

import json
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from minieval_bench import __version__
from minieval_bench.comparison import build_comparison_payload
from minieval_bench.config import ConfigError, DatasetError, load_config, load_dataset
from minieval_bench.evaluation import (
    build_code_runner_from_config,
    build_judge_from_config,
    evaluate_response,
    plan_judge_calls,
)
from minieval_bench.evaluation.judge import JudgeError, load_calibration
from minieval_bench.evaluation.pairwise import (
    PairwiseError,
    build_pairwise_judge,
    pairwise_summary,
    run_pairwise,
)
from minieval_bench.exports import write_trials_csv
from minieval_bench.history import find_runs
from minieval_bench.html_report import render_standalone, write_html_report
from minieval_bench.models import BenchmarkConfig, DatasetEntry
from minieval_bench.reports import (
    build_manifest,
    dataset_fingerprint,
    errors_table,
    evaluation_breakdown_table,
    judge_manifest_block,
    load_dataset_snapshot,
    load_results,
    match_dataset_entries,
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
        check_judge_budget(benchmark_config, entries)
    except JudgeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    try:
        trials, started_at, duration_seconds = run_benchmark(benchmark_config, entries)
    except KeyboardInterrupt:
        console.print("[red]Interrupted. No reports were written.[/red]")
        raise typer.Exit(130) from None
    except JudgeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
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
    breakdown = evaluation_breakdown_table(stats)
    if breakdown:
        console.print(breakdown)
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


def check_judge_budget(benchmark_config, entries) -> None:
    planned = plan_judge_calls(benchmark_config, entries)
    if planned == 0:
        return
    judge = build_judge_from_config(benchmark_config, benchmark_config.defaults.seed)
    refusal = judge.plan_check(planned) if judge else None
    if refusal:
        raise JudgeError(refusal)
    console.print(
        f"Judge enabled: {planned} call(s) planned "
        f"({benchmark_config.evaluators.judge.repeats} judgment(s) per response)"
    )
    if judge and not judge.pricing_known():
        console.print(
            "[yellow]Judge pricing is unknown: "
            "the cost of evaluation cannot be guaranteed.[/yellow]"
        )
    if judge and benchmark_config.evaluators.judge.budget_usd is not None:
        console.print(
            "budget_usd is a stop-on-observed guard: judging stops once the recorded spend "
            "reaches it. Input-token cost cannot be pre-bounded, so a single in-flight call "
            "may overshoot it."
        )


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
    breakdown = evaluation_breakdown_table(stats)
    if breakdown:
        console.print(breakdown)
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


@app.command()
def evaluate(
    input_dir: Annotated[
        Path, typer.Option("--input", help="Directory containing a results.json file.")
    ],
    config: ConfigArg,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output", help="Directory for the re-evaluated reports."),
    ] = None,
) -> None:
    """Re-evaluate stored responses without calling the candidate models."""
    benchmark_config, entries = load_benchmark(config)
    try:
        manifest, _, trials = load_results(input_dir)
    except (FileNotFoundError, json.JSONDecodeError, ValidationError) as exc:
        console.print(f"[red]Cannot read results from {input_dir}: {exc}[/red]")
        raise typer.Exit(1) from exc

    entries_by_id = {entry.id: entry for entry in entries}
    missing = sorted({trial.entry_id for trial in trials} - set(entries_by_id))
    if missing:
        console.print(
            f"[yellow]{len(missing)} trial entry id(s) not in the dataset are kept unchanged: "
            f"{', '.join(missing[:5])}{'…' if len(missing) > 5 else ''}[/yellow]"
        )

    snapshot = load_dataset_snapshot(input_dir)
    new_dataset_sha = dataset_fingerprint(Path(benchmark_config.dataset))
    matching = match_dataset_entries(
        manifest, snapshot, manifest.dataset_sha256 == new_dataset_sha, entries
    )
    conflicting_ids = set(matching["conflicts"])
    if conflicting_ids:
        console.print(
            f"[red]{len(conflicting_ids)} entry id(s) changed since the source run and are kept"
            f" unchanged (their stored answers may not match the new questions): "
            f"{', '.join(sorted(conflicting_ids)[:5])}"
            f"{'…' if len(conflicting_ids) > 5 else ''}[/red]"
        )
    if matching["unverified"]:
        console.print(
            f"[yellow]{len(matching['unverified'])} entry id(s) cannot be verified against the"
            " source run (matching by id only); their stored answers may not match the"
            " current questions.[/yellow]"
        )

    try:
        judge = build_judge_from_config(benchmark_config, benchmark_config.defaults.seed)
    except JudgeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    if judge is not None:
        planned = sum(
            1
            for trial in trials
            if trial.ok and entries_by_id.get(trial.entry_id) is not None
            and entries_by_id[trial.entry_id].metric == "judge"
            and trial.entry_id not in conflicting_ids
        ) * benchmark_config.evaluators.judge.repeats
        refusal = judge.plan_check(planned)
        if refusal:
            console.print(f"[red]{refusal}[/red]")
            raise typer.Exit(1)
        console.print(f"Judge enabled: {planned} call(s) planned for re-evaluation")
    code_runner = build_code_runner_from_config(benchmark_config)

    reevaluation_started = datetime.now(UTC)
    reevaluated = []
    for trial in trials:
        entry = entries_by_id.get(trial.entry_id)
        if (
            not trial.ok
            or entry is None
            or trial.entry_id in conflicting_ids
        ):
            reevaluated.append(trial)
            continue
        outcome = evaluate_response(entry, trial.response or "", judge, code_runner)
        reevaluated.append(
            trial.model_copy(update={"evaluation": outcome, "passed": outcome.passed})
        )
    reevaluation_duration = (datetime.now(UTC) - reevaluation_started).total_seconds()

    stats = compute_stats(reevaluated)
    source_manifest = manifest
    reevaluation_manifest = build_manifest(
        benchmark_config, entries, reevaluation_started, reevaluation_duration
    )
    reevaluation_manifest = reevaluation_manifest.model_copy(
        update={
            "source_run": {
                "benchmark": source_manifest.benchmark,
                "created_at_utc": source_manifest.created_at_utc,
                "minieval_version": source_manifest.minieval_version,
                "seed": source_manifest.seed,
                "dataset_sha256": source_manifest.dataset_sha256,
                "models": [
                    model.model_dump(mode="json") for model in source_manifest.models
                ],
                "path": str(input_dir),
            },
            "reevaluation": {
                "minieval_version": __version__,
                "created_at_utc": reevaluation_started.isoformat(),
                "duration_seconds": reevaluation_duration,
                "dataset_sha256": new_dataset_sha,
                "evaluators": (
                    benchmark_config.evaluators.model_dump(mode="json")
                    if benchmark_config.evaluators
                    else None
                ),
                "judge": judge_manifest_block(benchmark_config),
                "matching": {
                    "basis": matching["basis"],
                    "verified": len(matching["verified"]),
                    "unverified": len(matching["unverified"]),
                    "conflicts": sorted(conflicting_ids),
                },
            },
        }
    )
    destination = output_dir or input_dir / "reevaluated"
    written = write_reports(
        destination, reevaluation_manifest, stats, reevaluated, dataset=entries
    )

    console.print(manifest_table(reevaluation_manifest))
    console.print(stats_table(stats))
    for path in written:
        console.print(f"wrote {path}")
    console.print(f"original results in {input_dir} were left untouched")


@app.command()
def pairwise(
    input_dir: Annotated[
        Path, typer.Option("--input", help="Directory containing a results.json file.")
    ],
    config: ConfigArg,
    models_arg: Annotated[
        list[str] | None,
        typer.Option("--models", help="The two model names to compare (repeat the option)."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Path of the pairwise HTML report."),
    ] = None,
) -> None:
    """Compare two models blindly, with position swapping, on a completed run."""
    benchmark_config, entries = load_benchmark(config)
    try:
        manifest, _, trials = load_results(input_dir)
    except (FileNotFoundError, json.JSONDecodeError, ValidationError) as exc:
        console.print(f"[red]Cannot read results from {input_dir}: {exc}[/red]")
        raise typer.Exit(1) from exc

    judge_config = benchmark_config.evaluators.judge if benchmark_config.evaluators else None
    if judge_config is None:
        console.print("[red]pairwise requires an evaluators.judge block in the config.[/red]")
        raise typer.Exit(1)

    available_models = sorted({trial.model for trial in trials})
    if models_arg:
        if len(models_arg) != 2 or any(name not in available_models for name in models_arg):
            console.print(
                f"[red]--models needs exactly two names among: {', '.join(available_models)}[/red]"
            )
            raise typer.Exit(1)
        model_a, model_b = models_arg
    else:
        if len(available_models) < 2:
            console.print("[red]the run contains fewer than two models.[/red]")
            raise typer.Exit(1)
        model_a, model_b = available_models[:2]

    try:
        judge = build_pairwise_judge(judge_config, benchmark_config.defaults.seed)
    except PairwiseError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    prompt_by_entry = {entry.id: entry.prompt for entry in entries}
    records = run_pairwise(
        trials, prompt_by_entry, model_a, model_b, judge, benchmark_config.defaults.seed
    )
    summary = pairwise_summary(records, model_a, model_b)
    payload = {
        "kind": "pairwise",
        "manifest": manifest.model_dump(mode="json"),
        "modelA": model_a,
        "modelB": model_b,
        "summary": summary,
        "records": [record.model_dump(mode="json") for record in records],
    }

    html_path = output or input_dir / "pairwise.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(
        render_standalone(payload, "pairwise.html", "pairwise.js"), encoding="utf-8"
    )
    json_path = html_path.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    table = Table(title=f"Blind pairwise: {model_a} vs {model_b}")
    table.add_column("Measure")
    table.add_column("Count", justify="right")
    for measure, count in summary.items():
        table.add_row(measure, str(count))
    console.print(table)
    console.print(f"wrote {html_path}")
    console.print(f"wrote {json_path}")


@app.command()
def calibrate(
    config: ConfigArg,
    output: Annotated[
        Path, typer.Option("--output", help="Path of the calibration JSON report.")
    ] = Path("calibration.json"),
) -> None:
    """Compare judge verdicts against a calibration dataset with expected scores."""
    benchmark_config, _ = load_benchmark(config)
    judge_config = benchmark_config.evaluators.judge if benchmark_config.evaluators else None
    if judge_config is None:
        console.print("[red]calibrate requires an evaluators.judge block in the config.[/red]")
        raise typer.Exit(1)
    if not judge_config.calibration:
        console.print("[red]calibrate requires judge.calibration pointing to a JSONL file.[/red]")
        raise typer.Exit(1)

    try:
        items = load_calibration(Path(judge_config.calibration))
        judge = build_judge_from_config(benchmark_config, benchmark_config.defaults.seed)
        if judge is None:
            raise JudgeError("judge is not configured")
    except JudgeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    criterion_ids = [criterion.id for criterion in judge.rubric.criteria]
    unknown = sorted(
        {key for item in items for key in item.expected} - set(criterion_ids)
    )
    if unknown:
        console.print(
            f"[red]calibration references unknown criteria: {', '.join(unknown)}[/red]"
        )
        raise typer.Exit(1)

    results = []
    for item in items:
        outcome = judge.evaluate(item.prompt, None, item.response)
        judged = {}
        if outcome.status == "completed" and outcome.details:
            judged = {
                criterion_id: values["mean"]
                for criterion_id, values in outcome.details.get("criteria", {}).items()
            }
        results.append(
            {
                "id": item.id,
                "expected": item.expected,
                "judged": judged,
                "status": outcome.status,
                "error": outcome.error_message,
            }
        )

    by_criterion = {}
    for criterion_id in criterion_ids:
        differences = []
        for item, result in zip(items, results, strict=True):
            if criterion_id in result["judged"] and criterion_id in item.expected:
                differences.append(result["judged"][criterion_id] - item.expected[criterion_id])
        by_criterion[criterion_id] = {
            "items": len(differences),
            "mean_error": statistics.mean(differences) if differences else None,
            "disagreements": sum(1 for difference in differences if abs(difference) >= 1),
        }

    report = {
        "judge_model": judge.judge_model_name(),
        "rubric_fingerprint": judge.rubric.fingerprint(),
        "criteria": by_criterion,
        "items": results,
        "note": (
            "Agreement with calibration references does not prove judge validity outside "
            "these examples; disagreement highlights specific weaknesses."
        ),
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    table = Table(title="Judge calibration")
    table.add_column("Criterion")
    table.add_column("Items", justify="right")
    table.add_column("Mean error", justify="right")
    table.add_column("Disagreements", justify="right")
    for criterion_id, values in by_criterion.items():
        table.add_row(
            criterion_id,
            str(values["items"]),
            "n/a" if values["mean_error"] is None else f"{values['mean_error']:+.2f}",
            str(values["disagreements"]),
        )
    console.print(table)
    console.print(f"wrote {output}")


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
    if manifest.judge:
        table.add_row("Judge model", manifest.judge.get("model") or manifest.judge.get("provider"))
        table.add_row("Judge rubric", str(manifest.judge.get("rubric_fingerprint")))
    if manifest.source_run:
        source = manifest.source_run
        table.add_row("Source run", source.get("path", "n/a"))
        table.add_row("Source benchmark", source.get("benchmark", "n/a"))
        table.add_row("Source version", source.get("minieval_version", "n/a"))
        table.add_row("Source dataset sha256", source.get("dataset_sha256", "n/a"))
    if manifest.reevaluation:
        reevaluation = manifest.reevaluation
        matching = reevaluation.get("matching", {})
        table.add_row(
            "Reevaluation matching",
            f"{matching.get('basis', 'n/a')}: {matching.get('verified', 0)} verified, "
            f"{matching.get('unverified', 0)} unverified, "
            f"{len(matching.get('conflicts', []))} conflict(s)",
        )
        table.add_row("Reevaluation dataset sha256", reevaluation.get("dataset_sha256", "n/a"))
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
