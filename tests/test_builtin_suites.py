from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from minieval_bench.cli import app
from minieval_bench.suites.registry import (
    builtin_suites_root,
    discover_suites,
    load_suite,
    validate_suite,
)

runner = CliRunner()
WIDE = {"COLUMNS": "220"}

EXPECTED_SUITES = {
    "mini-general": "1.0.0",
    "mini-instruction": "1.0.0",
    "mini-structured": "1.0.0",
    "mini-code": "0.1.0",
}


def test_builtin_suites_are_discovered() -> None:
    summaries = discover_suites()
    by_id = {summary.suite_id: summary for summary in summaries}
    for suite_id, version in EXPECTED_SUITES.items():
        assert suite_id in by_id, suite_id
        assert by_id[suite_id].error is None
        assert by_id[suite_id].version == version
        assert by_id[suite_id].origin == "builtin"


@pytest.mark.parametrize("suite_id", sorted(EXPECTED_SUITES))
def test_builtin_suites_validate(suite_id: str) -> None:
    loaded = load_suite(suite_id)
    assert validate_suite(loaded) == []
    for task in loaded.definition.tasks:
        assert len(loaded.datasets[task.id].entries) >= 5
    assert loaded.definition.license
    assert loaded.definition.description


def test_builtin_suite_contents_are_original_and_safe() -> None:
    forbidden = ("sk-", "api_key", "password", "Bearer ")
    for path in sorted(Path(builtin_suites_root()).rglob("*")):
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            for marker in forbidden:
                assert marker not in text, (path, marker)


def test_mini_general_runs_offline_with_mock_provider(tmp_path: Path) -> None:
    models = tmp_path / "models.yaml"
    models.write_text(
        "models:\n  - name: mock-a\n    provider: mock\n", encoding="utf-8"
    )
    output = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "suite", "run", "mini-general",
            "--config", str(models),
            "--output", str(output),
            "--repeats", "1",
            "--concurrency", "4",
        ],
        env=WIDE,
    )
    assert result.exit_code == 0
    payload = json.loads((output / "suite-manifest.json").read_text(encoding="utf-8"))
    statuses = {task["status"] for task in payload["tasks"]}
    assert statuses == {"completed"}
    assert payload["generation"]["defaults"]["repeats"] == 1
    lock = json.loads((output / "benchmark.lock.json").read_text(encoding="utf-8"))
    assert len(lock["tasks"]) == 5
    assert (output / "suite-report.html").is_file()

    verify = runner.invoke(app, ["suite", "verify", str(output)], env=WIDE)
    assert verify.exit_code == 0


def test_mini_code_requires_explicit_activation(tmp_path: Path) -> None:
    models = tmp_path / "models.yaml"
    models.write_text(
        "models:\n  - name: mock-a\n    provider: mock\n", encoding="utf-8"
    )
    result = runner.invoke(
        app,
        [
            "suite", "run", "mini-code",
            "--config", str(models),
            "--output", str(tmp_path / "out"),
        ],
        env=WIDE,
    )
    assert result.exit_code == 1
    assert "--allow-code-execution" in result.output


def test_dry_run_works_offline_for_builtin_suite(tmp_path: Path) -> None:
    models = tmp_path / "models.yaml"
    models.write_text(
        "models:\n  - name: mock-a\n    provider: mock\n", encoding="utf-8"
    )
    result = runner.invoke(
        app,
        [
            "suite", "run", "mini-instruction",
            "--config", str(models),
            "--output", str(tmp_path / "out"),
            "--dry-run",
        ],
        env=WIDE,
    )
    assert result.exit_code == 0
    assert "Planned candidate calls: 68" in result.output
    assert "nothing was executed and nothing was written" in result.output


def test_builtin_stratified_sampling(tmp_path: Path) -> None:
    loaded = load_suite("mini-general")
    from minieval_bench.suites.sampling import SamplingError, SamplingPlan, apply_sampling

    dataset = loaded.datasets["arithmetic"]
    _, record = apply_sampling(
        dataset.entries, dataset.sha256, SamplingPlan(method="stratified", count=8, seed=1)
    )
    assert record["selected_count"] == 8

    code = load_suite("mini-code")
    functions = code.datasets["functions"]
    with pytest.raises(SamplingError, match="category"):
        apply_sampling(
            functions.entries,
            functions.sha256,
            SamplingPlan(method="stratified", count=2, seed=1),
        )
