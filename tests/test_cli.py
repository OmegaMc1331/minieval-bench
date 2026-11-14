from __future__ import annotations

import json

import yaml
from typer.testing import CliRunner

from conftest import EXAMPLE_CONFIG
from minieval_bench.cli import app

runner = CliRunner()
WIDE = {"COLUMNS": "200"}


def test_version_flag():
    result = runner.invoke(app, ["--version"], env=WIDE)

    assert result.exit_code == 0
    assert "0.2.0" in result.output


def test_validate_accepts_example_config():
    result = runner.invoke(app, ["validate", "--config", str(EXAMPLE_CONFIG)], env=WIDE)

    assert result.exit_code == 0
    assert "Configuration is valid" in result.output
    assert "mock-free" in result.output


def test_validate_rejects_missing_config(tmp_path):
    result = runner.invoke(app, ["validate", "--config", str(tmp_path / "absent.yaml")], env=WIDE)

    assert result.exit_code == 1
    assert "not found" in result.output


def test_run_writes_reports(tmp_path):
    output_dir = tmp_path / "out"
    result = runner.invoke(
        app,
        ["run", "--config", str(EXAMPLE_CONFIG), "--output", str(output_dir)],
        env=WIDE,
    )

    assert result.exit_code == 0
    results = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    assert len(results["trials"]) == 2 * 6 * 3
    assert {item["model"] for item in results["stats"]} == {"mock-free", "mock-priced"}
    assert (output_dir / "report.md").is_file()
    assert (output_dir / "manifest.json").is_file()


def test_run_reports_missing_dataset(tmp_path, mock_config):
    mock_config["dataset"] = str(tmp_path / "absent.jsonl")
    config_path = tmp_path / "bench.yaml"
    config_path.write_text(yaml.safe_dump(mock_config), encoding="utf-8")

    result = runner.invoke(app, ["run", "--config", str(config_path)], env=WIDE)

    assert result.exit_code == 1
    assert "not found" in result.output


def test_report_displays_previous_run(tmp_path):
    output_dir = tmp_path / "out"
    run_result = runner.invoke(
        app,
        ["run", "--config", str(EXAMPLE_CONFIG), "--output", str(output_dir)],
        env=WIDE,
    )
    assert run_result.exit_code == 0

    result = runner.invoke(app, ["report", "--input", str(output_dir)], env=WIDE)

    assert result.exit_code == 0
    assert "mock-free" in result.output
    assert "Run manifest" in result.output


def test_report_fails_without_results(tmp_path):
    result = runner.invoke(app, ["report", "--input", str(tmp_path)], env=WIDE)

    assert result.exit_code == 1
    assert "Cannot read report" in result.output


def test_run_writes_html_report_by_default(tmp_path):
    output_dir = tmp_path / "out"
    result = runner.invoke(
        app, ["run", "--config", str(EXAMPLE_CONFIG), "--output", str(output_dir)], env=WIDE
    )

    assert result.exit_code == 0
    html = (output_dir / "report.html").read_text(encoding="utf-8")
    assert "<!doctype html>" in html.lower()
    assert "mock-demo" in html


def test_run_no_html_skips_html_report(tmp_path):
    output_dir = tmp_path / "out"
    result = runner.invoke(
        app,
        ["run", "--config", str(EXAMPLE_CONFIG), "--output", str(output_dir), "--no-html"],
        env=WIDE,
    )

    assert result.exit_code == 0
    assert not (output_dir / "report.html").exists()
    assert (output_dir / "results.json").is_file()


def test_run_snapshot_embeds_dataset(tmp_path):
    output_dir = tmp_path / "out"
    result = runner.invoke(
        app,
        ["run", "--config", str(EXAMPLE_CONFIG), "--output", str(output_dir), "--snapshot"],
        env=WIDE,
    )

    assert result.exit_code == 0
    results = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    assert len(results["dataset"]) == 6
    assert results["dataset"][0]["prompt"]


def test_report_html_format(tmp_path):
    output_dir = tmp_path / "out"
    runner.invoke(
        app, ["run", "--config", str(EXAMPLE_CONFIG), "--output", str(output_dir)], env=WIDE
    )

    result = runner.invoke(
        app,
        ["report", "--input", str(output_dir), "--format", "html"],
        env=WIDE,
    )

    assert result.exit_code == 0
    assert (output_dir / "report.html").is_file()


def test_report_rejects_unknown_format(tmp_path):
    output_dir = tmp_path / "out"
    runner.invoke(
        app, ["run", "--config", str(EXAMPLE_CONFIG), "--output", str(output_dir), "--no-html"],
        env=WIDE,
    )

    result = runner.invoke(
        app, ["report", "--input", str(output_dir), "--format", "pdf"], env=WIDE
    )

    assert result.exit_code == 1
    assert "Unknown format" in result.output


def test_compare_writes_html_report(tmp_path):
    first = tmp_path / "run-a"
    second = tmp_path / "run-b"
    for directory in (first, second):
        runner.invoke(
            app, ["run", "--config", str(EXAMPLE_CONFIG), "--output", str(directory)], env=WIDE
        )
    comparison = tmp_path / "comparison.html"

    result = runner.invoke(
        app,
        [
            "compare",
            "--run", str(first),
            "--run", str(second),
            "--output", str(comparison),
        ],
        env=WIDE,
    )

    assert result.exit_code == 0
    html = comparison.read_text(encoding="utf-8")
    assert "mock-demo" in html
    assert "Model comparison" in html


def test_compare_warns_on_different_datasets(tmp_path, mock_config, valid_entries, write_benchmark):
    first = tmp_path / "run-a"
    second = tmp_path / "run-b"
    mock_config["defaults"]["seed"] = 1
    config_a = write_benchmark(mock_config, valid_entries)
    runner.invoke(
        app,
        ["run", "--config", str(config_a), "--output", str(first), "--no-html"],
        env=WIDE,
    )
    valid_entries[0]["prompt"] = "changed prompt"
    config_b = write_benchmark({**mock_config, "benchmark": "run-b"}, valid_entries)
    runner.invoke(
        app,
        ["run", "--config", str(config_b), "--output", str(second), "--no-html"],
        env=WIDE,
    )

    result = runner.invoke(
        app,
        [
            "compare",
            "--run", str(first),
            "--run", str(second),
            "--output", str(tmp_path / "comparison.html"),
        ],
        env=WIDE,
    )

    assert result.exit_code == 0
    assert "Datasets differ" in result.output
    assert (tmp_path / "comparison.html").is_file()


def test_compare_requires_two_runs(tmp_path):
    result = runner.invoke(app, ["compare", "--run", str(tmp_path)], env=WIDE)

    assert result.exit_code == 1
    assert "at least two" in result.output


def test_export_writes_csv(tmp_path):
    output_dir = tmp_path / "out"
    runner.invoke(
        app, ["run", "--config", str(EXAMPLE_CONFIG), "--output", str(output_dir), "--no-html"],
        env=WIDE,
    )

    result = runner.invoke(
        app,
        ["export", "--input", str(output_dir), "--format", "csv"],
        env=WIDE,
    )

    assert result.exit_code == 0
    csv_path = output_dir / "trials.csv"
    assert csv_path.is_file()
    content = csv_path.read_text(encoding="utf-8")
    assert content.splitlines()[0].startswith("model,entry_id,repeat")
    assert len(content.splitlines()) == 37


def test_export_rejects_unknown_format(tmp_path):
    output_dir = tmp_path / "out"
    runner.invoke(
        app, ["run", "--config", str(EXAMPLE_CONFIG), "--output", str(output_dir), "--no-html"],
        env=WIDE,
    )

    result = runner.invoke(
        app, ["export", "--input", str(output_dir), "--format", "xlsx"], env=WIDE
    )

    assert result.exit_code == 1
    assert "Unsupported format" in result.output


def test_history_lists_runs(tmp_path):
    for name in ("run-a", "run-b"):
        runner.invoke(
            app,
            ["run", "--config", str(EXAMPLE_CONFIG), "--output", str(tmp_path / name), "--no-html"],
            env=WIDE,
        )

    result = runner.invoke(app, ["history", "--root", str(tmp_path)], env=WIDE)

    assert result.exit_code == 0
    assert "mock-demo" in result.output
    assert "run-a" in result.output


def test_history_finds_nothing(tmp_path):
    result = runner.invoke(app, ["history", "--root", str(tmp_path / "empty")], env=WIDE)

    assert result.exit_code == 0
    assert "No benchmark runs found" in result.output


def test_report_reads_v010_results(tmp_path):
    from test_reports import OLD_RESULTS

    results_dir = tmp_path / "legacy"
    results_dir.mkdir()
    (results_dir / "results.json").write_text(json.dumps(OLD_RESULTS), encoding="utf-8")

    result = runner.invoke(app, ["report", "--input", str(results_dir)], env=WIDE)

    assert result.exit_code == 0
    assert "old-mock" in result.output
    assert "n/a" in result.output
