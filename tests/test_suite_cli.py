from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

from minieval_bench.cli import app

runner = CliRunner()
WIDE = {"COLUMNS": "220"}


def write_user_suite(root: Path, suite_id: str = "cli-suite") -> Path:
    suite_dir = root / suite_id
    dataset_dir = suite_dir / "datasets"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "a.jsonl").write_text(
        '{"id": "a1", "prompt": "Say one", "metric": "exact_match", "reference": "one"}\n'
        '{"id": "a2", "prompt": "Say two", "metric": "exact_match", "reference": "two"}\n',
        encoding="utf-8",
    )
    definition = {
        "id": suite_id,
        "version": "1.2.0",
        "name": "CLI Suite",
        "description": "Suite exercised through the CLI.",
        "license": "MIT",
        "defaults": {"repeats": 2, "seed": 3, "concurrency": 1},
        "tasks": [
            {
                "id": "task-a",
                "dataset": "datasets/a.jsonl",
                "weight": 1.5,
                "expected_metric": "exact_match",
            }
        ],
    }
    (suite_dir / "suite.yaml").write_text(yaml.safe_dump(definition), encoding="utf-8")
    return suite_dir.parent


def test_suite_list_shows_user_suites(tmp_path: Path) -> None:
    suites_root = write_user_suite(tmp_path)
    result = runner.invoke(app, ["suite", "list", "--path", str(suites_root)], env=WIDE)

    assert result.exit_code == 0
    assert "cli-suite" in result.output
    assert "1.2.0" in result.output
    assert "ok" in result.output


def test_suite_list_without_suites_prints_guidance(tmp_path: Path) -> None:
    result = runner.invoke(app, ["suite", "list", "--path", str(tmp_path / "empty")], env=WIDE)

    assert result.exit_code == 0
    assert "No suites found" in result.output


def test_suite_info_shows_tasks_and_conditions(tmp_path: Path) -> None:
    suites_root = write_user_suite(tmp_path)
    result = runner.invoke(
        app, ["suite", "info", "cli-suite", "--path", str(suites_root)], env=WIDE
    )

    assert result.exit_code == 0
    assert "Version" in result.output
    assert "1.2.0" in result.output
    assert "task-a" in result.output
    assert "exact_match x2" in result.output
    assert "repeats 2, seed 3" in result.output
    assert "MIT" in result.output
    assert "deterministic metrics only" in result.output


def test_suite_info_unknown_suite_fails(tmp_path: Path) -> None:
    result = runner.invoke(app, ["suite", "info", "ghost", "--path", str(tmp_path)], env=WIDE)

    assert result.exit_code == 1
    assert "no suite 'ghost'" in result.output


def test_suite_validate_accepts_valid_suite(tmp_path: Path) -> None:
    suites_root = write_user_suite(tmp_path)
    result = runner.invoke(
        app, ["suite", "validate", "cli-suite", "--path", str(suites_root)], env=WIDE
    )

    assert result.exit_code == 0
    assert "is valid" in result.output
    assert "2 entries" in result.output


def test_suite_validate_rejects_metric_mismatch(tmp_path: Path) -> None:
    suites_root = write_user_suite(tmp_path, "mismatch-suite")
    dataset = suites_root / "mismatch-suite" / "datasets" / "a.jsonl"
    dataset.write_text(
        '{"id": "a1", "prompt": "Say one", "metric": "contains", "reference": "one"}\n',
        encoding="utf-8",
    )
    result = runner.invoke(
        app, ["suite", "validate", "mismatch-suite", "--path", str(suites_root)], env=WIDE
    )

    assert result.exit_code == 1
    assert "expects metric 'exact_match'" in result.output


def test_suite_list_marks_malformed_suite(tmp_path: Path) -> None:
    suites_root = write_user_suite(tmp_path)
    broken = suites_root / "broken"
    broken.mkdir()
    (broken / "suite.yaml").write_text("id: [unclosed", encoding="utf-8")
    result = runner.invoke(app, ["suite", "list", "--path", str(suites_root)], env=WIDE)

    assert result.exit_code == 0
    assert "broken" in result.output
    assert "error" in result.output
