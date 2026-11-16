from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from minieval_bench.cli import app
from minieval_bench.suites.lockfile import lockfile_differences
from minieval_bench.suites.registry import load_suite_dir
from minieval_bench.suites.runner import SuiteRunOptions, prepare_run, run_suite
from minieval_bench.suites.sampling import SamplingPlan

runner = CliRunner()
WIDE = {"COLUMNS": "220"}

MODELS_YAML = "models:\n  - name: mock-a\n    provider: mock\n"


def write_jsonl(path: Path, entries: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n", encoding="utf-8"
    )
    return path


def write_schema_suite(root: Path) -> Path:
    suite_dir = root / "suites" / "schema-suite"
    definition = {
        "id": "schema-suite",
        "version": "1.0.0",
        "name": "Schema",
        "description": "Structured output task.",
        "license": "MIT",
        "tasks": [{"id": "objects", "dataset": "datasets/objects.jsonl"}],
    }
    suite_dir.mkdir(parents=True)
    (suite_dir / "suite.yaml").write_text(yaml.safe_dump(definition), encoding="utf-8")
    write_jsonl(
        suite_dir / "datasets" / "objects.jsonl",
        [
            {
                "id": "o1",
                "prompt": "Return a user object.",
                "metric": "json_schema",
                "json_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            }
        ],
    )
    return suite_dir


def current_lock(suite_dir: Path, tmp_path: Path, sampling: SamplingPlan | None = None):
    loaded = load_suite_dir(suite_dir)
    models_file = tmp_path / "models.yaml"
    models_file.write_text(MODELS_YAML, encoding="utf-8")
    options = SuiteRunOptions(
        models_path=models_file,
        output_dir=tmp_path / "out",
        sampling=sampling or SamplingPlan(),
    )
    plans, models, effective, overrides = prepare_run(loaded, options)
    from minieval_bench.suites.lockfile import build_lockfile

    return build_lockfile(
        loaded, models, effective, overrides, [plan.lock_section for plan in plans]
    )


def test_lockfile_detects_dataset_change(tmp_path: Path) -> None:
    suite_dir = write_schema_suite(tmp_path)
    stored = current_lock(suite_dir, tmp_path)
    write_jsonl(
        suite_dir / "datasets" / "objects.jsonl",
        [
            {
                "id": "o1",
                "prompt": "Return a user object, revised.",
                "metric": "json_schema",
                "json_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            }
        ],
    )
    changed = current_lock(suite_dir, tmp_path)
    differences = lockfile_differences(stored, changed)
    assert any("dataset content changed" in difference for difference in differences)


def test_lockfile_detects_schema_change(tmp_path: Path) -> None:
    suite_dir = write_schema_suite(tmp_path)
    stored = current_lock(suite_dir, tmp_path)
    write_jsonl(
        suite_dir / "datasets" / "objects.jsonl",
        [
            {
                "id": "o1",
                "prompt": "Return a user object.",
                "metric": "json_schema",
                "json_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                    "required": ["name"],
                },
            }
        ],
    )
    changed = current_lock(suite_dir, tmp_path)
    differences = lockfile_differences(stored, changed)
    assert any("json_schema of entry 'o1' changed" in d for d in differences)


def test_lockfile_detects_sampling_change(tmp_path: Path) -> None:
    suite_dir = write_schema_suite(tmp_path)
    stored = current_lock(suite_dir, tmp_path, SamplingPlan(method="random", count=1, seed=1))
    changed = current_lock(suite_dir, tmp_path, SamplingPlan(method="random", count=1, seed=2))
    differences = lockfile_differences(stored, changed)
    assert any("sampling selection changed" in difference for difference in differences)


def test_lockfile_detects_rubric_change(tmp_path: Path) -> None:
    suite_dir = tmp_path / "suites" / "judge-suite"
    rubric_path = tmp_path / "rubric.yaml"
    rubric_path.write_text(
        "version: 1\ncriteria:\n  - id: clarity\n    description: Is it clear\n"
        "    scale_min: 1\n    scale_max: 5\n    weight: 1.0\n",
        encoding="utf-8",
    )
    definition = {
        "id": "judge-suite",
        "version": "1.0.0",
        "name": "Judge",
        "description": "Subjective task.",
        "license": "MIT",
        "tasks": [{"id": "answers", "dataset": "datasets/answers.jsonl"}],
        "evaluators": {"judge": {"provider": "mock", "rubric": str(rubric_path)}},
    }
    suite_dir.mkdir(parents=True)
    (suite_dir / "suite.yaml").write_text(yaml.safe_dump(definition), encoding="utf-8")
    write_jsonl(
        suite_dir / "datasets" / "answers.jsonl",
        [{"id": "j1", "prompt": "Say something.", "metric": "judge"}],
    )

    stored = current_lock(suite_dir, tmp_path)
    rubric_path.write_text(
        "version: 2\ncriteria:\n  - id: clarity\n    description: Is it clear\n"
        "    scale_min: 1\n    scale_max: 5\n    weight: 2.0\n",
        encoding="utf-8",
    )
    changed = current_lock(suite_dir, tmp_path)
    differences = lockfile_differences(stored, changed)
    assert any("evaluator configuration changed" in difference for difference in differences)


def test_lockfile_detects_code_tests_change(tmp_path: Path) -> None:
    suite_dir = tmp_path / "suites" / "code-suite"
    definition = {
        "id": "code-suite",
        "version": "1.0.0",
        "name": "Code",
        "description": "d",
        "license": "MIT",
        "tasks": [{"id": "code", "dataset": "datasets/code.jsonl"}],
        "evaluators": {"code": {"runtime": "docker"}},
    }
    suite_dir.mkdir(parents=True)
    (suite_dir / "suite.yaml").write_text(yaml.safe_dump(definition), encoding="utf-8")
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
    stored = current_lock(suite_dir, tmp_path)
    write_jsonl(
        suite_dir / "datasets" / "code.jsonl",
        [
            {
                "id": "c1",
                "prompt": "Write add.",
                "metric": "code_tests",
                "tests": "def test_add():\n    assert add(2, 2) == 4\n",
            }
        ],
    )
    changed = current_lock(suite_dir, tmp_path)
    differences = lockfile_differences(stored, changed)
    assert any("code_test of entry 'c1' changed" in d for d in differences)


def test_lockfile_detects_weight_and_version_changes(tmp_path: Path) -> None:
    suite_dir = write_schema_suite(tmp_path)
    stored = current_lock(suite_dir, tmp_path)
    stored_modified = json.loads(json.dumps(stored))
    stored_modified["suite"]["version"] = "1.1.0"
    stored_modified["tasks"][0]["weight"] = 3.0
    differences = lockfile_differences(stored_modified, stored)
    assert any("suite version changed" in d for d in differences)
    assert any("weight changed" in d for d in differences)


def test_lockfile_detects_model_and_defaults_changes(tmp_path: Path) -> None:
    suite_dir = write_schema_suite(tmp_path)
    stored = current_lock(suite_dir, tmp_path)
    stored_modified = json.loads(json.dumps(stored))
    stored_modified["generation"]["models"][0]["temperature"] = 0.3
    differences = lockfile_differences(stored_modified, stored)
    assert any("model configuration changed" in d for d in differences)

    stored_modified = json.loads(json.dumps(stored))
    stored_modified["generation"]["defaults"]["repeats"] = 9
    differences = lockfile_differences(stored_modified, stored)
    assert any("run parameters changed" in d for d in differences)


def test_suite_run_then_verify_roundtrip(tmp_path: Path) -> None:
    suite_dir = write_schema_suite(tmp_path)
    suites_root = suite_dir.parent
    models_file = tmp_path / "models.yaml"
    models_file.write_text(MODELS_YAML, encoding="utf-8")
    output = tmp_path / "results" / "schema-suite"
    options = SuiteRunOptions(models_path=models_file, output_dir=output)
    run_suite(load_suite_dir(suite_dir), options)

    result = runner.invoke(
        app,
        ["suite", "verify", str(output), "--path", str(suites_root)],
        env=WIDE,
    )
    assert result.exit_code == 0
    assert "matches the current protocol" in result.output


def test_suite_verify_detects_dataset_modification(tmp_path: Path) -> None:
    suite_dir = write_schema_suite(tmp_path)
    suites_root = suite_dir.parent
    models_file = tmp_path / "models.yaml"
    models_file.write_text(MODELS_YAML, encoding="utf-8")
    output = tmp_path / "results" / "schema-suite"
    options = SuiteRunOptions(models_path=models_file, output_dir=output)
    run_suite(load_suite_dir(suite_dir), options)

    write_jsonl(
        suite_dir / "datasets" / "objects.jsonl",
        [
            {
                "id": "o1",
                "prompt": "Return a user object, edited.",
                "metric": "json_schema",
                "json_schema": {"type": "object", "required": []},
            }
        ],
    )
    result = runner.invoke(
        app,
        ["suite", "verify", str(output), "--path", str(suites_root)],
        env=WIDE,
    )
    assert result.exit_code == 1
    assert "dataset content changed" in result.output


def test_suite_verify_reports_missing_suite(tmp_path: Path) -> None:
    suite_dir = write_schema_suite(tmp_path)
    models_file = tmp_path / "models.yaml"
    models_file.write_text(MODELS_YAML, encoding="utf-8")
    output = tmp_path / "results" / "schema-suite"
    options = SuiteRunOptions(models_path=models_file, output_dir=output)
    run_suite(load_suite_dir(suite_dir), options)

    result = runner.invoke(
        app,
        ["suite", "verify", str(output), "--path", str(tmp_path / "empty")],
        env=WIDE,
    )
    assert result.exit_code == 1
    assert "not available in the registry" in result.output


def test_suite_verify_requires_artefacts(tmp_path: Path) -> None:
    result = runner.invoke(app, ["suite", "verify", str(tmp_path)], env=WIDE)
    assert result.exit_code == 1
    assert "missing" in result.output
