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
    assert "0.1.0" in result.output


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
