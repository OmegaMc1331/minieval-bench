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


def test_suite_list_always_includes_builtin_suites(tmp_path: Path) -> None:
    result = runner.invoke(app, ["suite", "list", "--path", str(tmp_path / "empty")], env=WIDE)

    assert result.exit_code == 0
    assert "mini-general" in result.output
    assert "mini-code" in result.output


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


def write_models_file(tmp_path: Path) -> Path:
    path = tmp_path / "models.yaml"
    path.write_text(
        "models:\n  - name: mock-a\n    provider: mock\n", encoding="utf-8"
    )
    return path


def test_suite_run_cli_executes_and_writes_artefacts(tmp_path: Path) -> None:
    suites_root = write_user_suite(tmp_path, "run-suite")
    models = write_models_file(tmp_path)
    output = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "suite", "run", "run-suite",
            "--config", str(models),
            "--output", str(output),
            "--path", str(suites_root),
        ],
        env=WIDE,
    )

    assert result.exit_code == 0
    assert "completed" in result.output
    assert (output / "suite-manifest.json").is_file()
    assert (output / "benchmark.lock.json").is_file()
    assert (output / "suite-results.json").is_file()
    assert (output / "tasks" / "task-a" / "results.json").is_file()


def test_suite_run_cli_dry_run_writes_nothing(tmp_path: Path) -> None:
    suites_root = write_user_suite(tmp_path, "dry-suite")
    models = write_models_file(tmp_path)
    output = tmp_path / "dry-out"
    result = runner.invoke(
        app,
        [
            "suite", "run", "dry-suite",
            "--config", str(models),
            "--output", str(output),
            "--dry-run",
            "--path", str(suites_root),
        ],
        env=WIDE,
    )

    assert result.exit_code == 0
    assert "Dry run" in result.output
    assert "Planned candidate calls: 4" in result.output
    assert "nothing was executed and nothing was written" in result.output
    assert not output.exists()


def test_suite_run_cli_dry_run_flags_overrides_and_sampling(tmp_path: Path) -> None:
    suites_root = write_user_suite(tmp_path, "dry-suite")
    models = write_models_file(tmp_path)
    result = runner.invoke(
        app,
        [
            "suite", "run", "dry-suite",
            "--config", str(models),
            "--output", str(tmp_path / "o"),
            "--dry-run",
            "--repeats", "1",
            "--sample", "1",
            "--sample-seed", "3",
            "--path", str(suites_root),
        ],
        env=WIDE,
    )

    assert result.exit_code == 0
    assert "User overrides: {'repeats': 1}" in result.output
    assert "Sampling: random, 1 entries per task" in result.output
    assert "Planned candidate calls: 1" in result.output


def test_suite_run_cli_rejects_invalid_suite(tmp_path: Path) -> None:
    suites_root = write_user_suite(tmp_path, "bad-suite")
    dataset = suites_root / "bad-suite" / "datasets" / "a.jsonl"
    dataset.write_text(
        '{"id": "a1", "prompt": "one", "metric": "contains", "reference": "one"}\n',
        encoding="utf-8",
    )
    models = write_models_file(tmp_path)
    result = runner.invoke(
        app,
        [
            "suite", "run", "bad-suite",
            "--config", str(models),
            "--output", str(tmp_path / "o"),
            "--path", str(suites_root),
        ],
        env=WIDE,
    )

    assert result.exit_code == 1
    assert "expects metric 'exact_match'" in result.output


def test_suite_run_cli_enforces_max_calls(tmp_path: Path) -> None:
    suites_root = write_user_suite(tmp_path, "capped-suite")
    models = write_models_file(tmp_path)
    result = runner.invoke(
        app,
        [
            "suite", "run", "capped-suite",
            "--config", str(models),
            "--output", str(tmp_path / "o"),
            "--max-calls", "3",
            "--path", str(suites_root),
        ],
        env=WIDE,
    )

    assert result.exit_code == 1
    assert "exceed --max-calls" in result.output


def test_suite_run_cli_requires_models_file(tmp_path: Path) -> None:
    suites_root = write_user_suite(tmp_path, "nosuite")
    result = runner.invoke(
        app,
        [
            "suite", "run", "nosuite",
            "--config", str(tmp_path / "absent.yaml"),
            "--output", str(tmp_path / "o"),
            "--path", str(suites_root),
        ],
        env=WIDE,
    )

    assert result.exit_code == 1
    assert "models file not found" in result.output
