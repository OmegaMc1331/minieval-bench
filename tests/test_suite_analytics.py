from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from minieval_bench.cli import app
from minieval_bench.suites.registry import load_suite_dir
from minieval_bench.suites.reports import (
    build_suite_payload,
    family_aggregates,
    load_suite_run,
    write_suite_reports,
)
from minieval_bench.suites.runner import SuiteRunOptions, run_suite

runner = CliRunner()
WIDE = {"COLUMNS": "220"}


def write_jsonl(path: Path, entries: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n", encoding="utf-8"
    )
    return path


def write_suite(root: Path, tasks: list[dict], evaluators: dict | None = None) -> Path:
    definition = {
        "id": "analytics-suite",
        "version": "1.0.0",
        "name": "Analytics",
        "description": "Mixed metric families.",
        "license": "MIT",
        "defaults": {"repeats": 1, "seed": 1, "concurrency": 2},
        "tasks": tasks,
    }
    if evaluators is not None:
        definition["evaluators"] = evaluators
    suite_dir = root / "suites" / "analytics-suite"
    suite_dir.mkdir(parents=True)
    (suite_dir / "suite.yaml").write_text(yaml.safe_dump(definition), encoding="utf-8")
    return suite_dir


def mixed_suite(root: Path) -> Path:
    suite_dir = write_suite(
        root,
        [
            {"id": "always-pass", "dataset": "datasets/pass.jsonl", "weight": 2.0},
            {"id": "always-fail", "dataset": "datasets/fail.jsonl", "weight": 1.0},
            {
                "id": "structured",
                "dataset": "datasets/structured.jsonl",
                "weight": 1.0,
            },
            {
                "id": "code",
                "dataset": "datasets/code.jsonl",
                "weight": 1.0,
            },
        ],
        evaluators={"code": {"runtime": "docker"}},
    )
    write_jsonl(
        suite_dir / "datasets" / "pass.jsonl",
        [
            {"id": "p1", "prompt": "a", "metric": "contains", "reference": "mock"},
        ],
    )
    write_jsonl(
        suite_dir / "datasets" / "fail.jsonl",
        [
            {"id": "f1", "prompt": "b", "metric": "contains", "reference": "zzz"},
            {"id": "f2", "prompt": "c", "metric": "contains", "reference": "zzz"},
            {"id": "f3", "prompt": "d", "metric": "contains", "reference": "zzz"},
        ],
    )
    write_jsonl(
        suite_dir / "datasets" / "structured.jsonl",
        [
            {
                "id": "s1",
                "prompt": "Return an object.",
                "metric": "json_schema",
                "json_schema": {"type": "object", "required": ["name"]},
            }
        ],
    )
    write_jsonl(
        suite_dir / "datasets" / "code.jsonl",
        [
            {
                "id": "c1",
                "prompt": "Write add.",
                "metric": "code_tests",
                "tests": "def test_add():\n    assert add(1, 1) == 2\n",
            }
        ],
    )
    return suite_dir


def run_mixed(tmp_path: Path, suite_dir: Path) -> Path:
    models_file = tmp_path / "models.yaml"
    models_file.write_text(
        "models:\n  - name: mock-a\n    provider: mock\n", encoding="utf-8"
    )
    output = tmp_path / "out"
    options = SuiteRunOptions(models_path=models_file, output_dir=output)
    run_suite(load_suite_dir(suite_dir), options)
    return output


def test_family_aggregates_separate_families(tmp_path: Path) -> None:
    output = run_mixed(tmp_path, mixed_suite(tmp_path))
    run = load_suite_run(output)

    aggregates = family_aggregates(run)
    families = [aggregate["family"] for aggregate in aggregates]
    assert families == ["deterministic", "json_schema", "code_tests"]

    deterministic = aggregates[0]
    model_row = deterministic["perModel"][0]
    assert model_row["model"] == "mock-a"
    assert model_row["macro"] is not None
    assert abs(model_row["macro"] - 2.0 / 3.0) < 1e-9
    assert abs(model_row["micro"] - 1.0 / 4.0) < 1e-9
    assert model_row["trials"] == 4
    assert model_row["tasksCovered"] == 2
    assert deterministic["tasks"] == 2

    json_family = aggregates[1]
    assert json_family["perModel"][0]["trials"] == 1
    code_family = aggregates[2]
    assert code_family["perModel"][0]["trials"] == 0
    assert code_family["perModel"][0]["macro"] is None
    assert code_family["tasks"] == 1


def test_unavailable_task_never_counts_as_zero(tmp_path: Path) -> None:
    output = run_mixed(tmp_path, mixed_suite(tmp_path))
    run = load_suite_run(output)

    code_task = next(task for task in run.tasks if task.task_id == "code")
    assert code_task.status == "unavailable"
    assert code_task.stats == []

    payload = build_suite_payload(run)
    code_family = next(
        aggregate
        for aggregate in payload["familyAggregates"]
        if aggregate["family"] == "code_tests"
    )
    assert code_family["perModel"][0]["macro"] is None
    assert code_family["perModel"][0]["trials"] == 0


def test_json_schema_conformance_reported(tmp_path: Path) -> None:
    output = run_mixed(tmp_path, mixed_suite(tmp_path))
    run = load_suite_run(output)

    structured = next(task for task in run.tasks if task.task_id == "structured")
    stats = structured.stats[0]
    assert stats.json_schema_trials == 1
    assert stats.json_schema_conformance == 0.0


def test_suite_reports_are_written(tmp_path: Path) -> None:
    output = run_mixed(tmp_path, mixed_suite(tmp_path))
    run = load_suite_run(output)

    written = write_suite_reports(run)

    assert [path.name for path in written] == ["suite-report.md", "suite-report.html"]
    markdown = (output / "suite-report.md").read_text(encoding="utf-8")
    assert "# MiniEval suite report: analytics-suite 1.0.0" in markdown
    assert "## Task breakdown" in markdown
    assert "## Family aggregates" in markdown
    assert "unavailable" in markdown
    assert "## Reproducibility" in markdown

    html = (output / "suite-report.html").read_text(encoding="utf-8")
    assert "<script id=\"report-data\" type=\"application/json\">" in html
    assert "src=" not in html
    assert 'href="http' not in html
    start = html.index("application/json\">") + len("application/json\">")
    end = html.index("</script>", start)
    payload = json.loads(html[start:end])
    assert payload["kind"] == "suite"
    assert len(payload["tasks"]) == 4
    assert payload["overview"]["task_count"] == 4


def test_payload_marks_sampling_and_overrides(tmp_path: Path) -> None:
    suite_dir = mixed_suite(tmp_path)
    models_file = tmp_path / "models.yaml"
    models_file.write_text(
        "models:\n  - name: mock-a\n    provider: mock\n", encoding="utf-8"
    )
    output = tmp_path / "out"
    from minieval_bench.suites.runner import UserOverrides
    from minieval_bench.suites.sampling import SamplingPlan

    options = SuiteRunOptions(
        models_path=models_file,
        output_dir=output,
        overrides=UserOverrides(repeats=1),
        sampling=SamplingPlan(method="random", count=1, seed=4),
    )
    run_suite(load_suite_dir(suite_dir), options)
    payload = build_suite_payload(load_suite_run(output))

    assert payload["overview"]["sampled"] is True
    assert payload["overview"]["protocol_modified"] is True
    assert payload["sampling"]["method"] == "random"
    selections = payload["sampling"]["selections"]
    assert selections["always-pass"]["selected"] == 1
    assert selections["always-fail"]["selected"] == 1


def test_consumption_totals(tmp_path: Path) -> None:
    suite_dir = mixed_suite(tmp_path)
    models_file = tmp_path / "models.yaml"
    models_file.write_text(
        "models:\n  - name: mock-priced\n    provider: mock\n"
        "    input_cost_per_1m: 0.5\n    output_cost_per_1m: 1.5\n",
        encoding="utf-8",
    )
    output = tmp_path / "out"
    run_suite(load_suite_dir(suite_dir), SuiteRunOptions(models_path=models_file, output_dir=output))

    payload = build_suite_payload(load_suite_run(output))
    row = payload["consumption"][0]
    assert row["model"] == "mock-priced"
    assert row["input_tokens"] is not None and row["input_tokens"] > 0
    assert row["cost_usd"] is not None and row["cost_usd"] > 0
    assert row["cost_partial"] is False


def test_suite_report_cli(tmp_path: Path) -> None:
    output = run_mixed(tmp_path, mixed_suite(tmp_path))
    for name in ("suite-report.md", "suite-report.html"):
        report_path = output / name
        if report_path.exists():
            report_path.unlink()

    result = runner.invoke(
        app, ["suite", "report", "--input", str(output), "--format", "terminal"], env=WIDE
    )
    assert result.exit_code == 0
    assert "Family aggregates" in result.output

    result = runner.invoke(
        app, ["suite", "report", "--input", str(output), "--format", "html"], env=WIDE
    )
    assert result.exit_code == 0
    assert (output / "suite-report.html").is_file()

    result = runner.invoke(
        app,
        ["suite", "report", "--input", str(output), "--format", "xml"],
        env=WIDE,
    )
    assert result.exit_code == 1


def test_suite_report_rejects_non_suite_directory(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["suite", "report", "--input", str(tmp_path)], env=WIDE
    )
    assert result.exit_code == 1
    assert "no suite-manifest.json" in result.output
