from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from minieval_bench.suites.models import SuiteDefinition, SuiteError
from minieval_bench.suites.registry import (
    discover_suites,
    load_suite,
    load_suite_dir,
    validate_suite,
)


def write_dataset(path: Path, entries: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n", encoding="utf-8"
    )
    return path


def write_suite(
    root: Path,
    suite_id: str = "demo-suite",
    tasks: list[dict] | None = None,
    **overrides,
) -> Path:
    definition = {
        "id": suite_id,
        "version": "1.0.0",
        "name": "Demo Suite",
        "description": "Suite used by tests.",
        "license": "MIT",
        "defaults": {"repeats": 2, "seed": 7, "concurrency": 2},
        "tasks": tasks
        or [
            {
                "id": "task-a",
                "dataset": "datasets/a.jsonl",
                "weight": 1.0,
                "expected_metric": "exact_match",
            }
        ],
    }
    definition.update(overrides)
    suite_dir = root / suite_id
    suite_dir.mkdir(parents=True)
    (suite_dir / "suite.yaml").write_text(yaml.safe_dump(definition), encoding="utf-8")
    write_dataset(
        suite_dir / "datasets" / "a.jsonl",
        [
            {"id": "a1", "prompt": "one", "metric": "exact_match", "reference": "1"},
            {"id": "a2", "prompt": "two", "metric": "exact_match", "reference": "2"},
        ],
    )
    return suite_dir


@pytest.fixture
def suite_root(tmp_path: Path) -> Path:
    return tmp_path / "suites"


def test_discovery_finds_user_suites(suite_root: Path) -> None:
    write_suite(suite_root)
    summaries = discover_suites([suite_root])
    user_summaries = [s for s in summaries if s.origin == str(suite_root)]
    assert [summary.suite_id for summary in user_summaries] == ["demo-suite"]
    summary = user_summaries[0]
    assert summary.version == "1.0.0"
    assert summary.task_count == 1
    assert summary.error is None
    assert summary.origin == str(suite_root)


def test_discovery_reports_malformed_suites(suite_root: Path) -> None:
    suite_dir = write_suite(suite_root, suite_id="broken-suite")
    (suite_dir / "suite.yaml").write_text(
        "id: broken-suite\nversion: not-semver\nname: Broken\ndescription: d\n"
        "license: MIT\ntasks:\n- id: t\n  dataset: datasets/a.jsonl\n",
        encoding="utf-8",
    )
    summaries = discover_suites([suite_root])
    broken = next(summary for summary in summaries if summary.suite_id == "broken-suite")
    assert broken.error is not None
    assert "version" in broken.error


def test_discovery_reports_duplicate_ids(suite_root: Path) -> None:
    write_suite(suite_root)
    other_root = suite_root.parent / "more-suites"
    write_suite(other_root)
    summaries = discover_suites([suite_root, other_root])
    duplicated = [
        summary
        for summary in summaries
        if summary.origin in (str(suite_root), str(other_root))
    ]
    assert len(duplicated) == 2
    assert all(
        summary.error == "suite id is defined in several registry locations"
        for summary in duplicated
    )


def test_discovery_ignores_directories_without_definition(suite_root: Path) -> None:
    (suite_root / "not-a-suite").mkdir(parents=True)
    write_suite(suite_root)
    summaries = discover_suites([suite_root])
    user_summaries = [s for s in summaries if s.origin == str(suite_root)]
    assert [summary.suite_id for summary in user_summaries] == ["demo-suite"]


def test_load_suite_reads_datasets(suite_root: Path) -> None:
    write_suite(suite_root)
    loaded = load_suite("demo-suite", [suite_root])
    assert loaded.definition.id == "demo-suite"
    assert loaded.datasets["task-a"].sha256
    assert [entry.id for entry in loaded.datasets["task-a"].entries] == ["a1", "a2"]
    assert validate_suite(loaded) == []


def test_load_suite_unknown_id_lists_available(suite_root: Path) -> None:
    write_suite(suite_root)
    with pytest.raises(SuiteError, match="available suites: demo-suite"):
        load_suite("missing-suite", [suite_root])


def test_load_suite_rejects_ambiguous_id(suite_root: Path) -> None:
    write_suite(suite_root)
    write_suite(suite_root.parent / "other", suite_id="demo-suite")
    with pytest.raises(SuiteError, match="several registry locations"):
        load_suite("demo-suite", [suite_root, suite_root.parent / "other"])


def test_task_ids_must_be_unique(suite_root: Path) -> None:
    with pytest.raises(ValueError, match="duplicate task ids"):
        SuiteDefinition.model_validate(
            {
                "id": "demo",
                "version": "1.0.0",
                "name": "Demo",
                "description": "d",
                "license": "MIT",
                "tasks": [
                    {"id": "same", "dataset": "datasets/a.jsonl"},
                    {"id": "same", "dataset": "datasets/b.jsonl"},
                ],
            }
        )


def test_version_must_be_semver() -> None:
    with pytest.raises(ValueError, match="semantic version"):
        SuiteDefinition.model_validate(
            {
                "id": "demo",
                "version": "1.0",
                "name": "Demo",
                "description": "d",
                "license": "MIT",
                "tasks": [{"id": "t", "dataset": "datasets/a.jsonl"}],
            }
        )


@pytest.mark.parametrize(
    "reference",
    ["/etc/passwd", "../escape.jsonl", "datasets/../../escape.jsonl", "suite.yaml"],
)
def test_unsafe_dataset_references_are_rejected(reference: str) -> None:
    with pytest.raises(ValueError, match="relative"):
        SuiteDefinition.model_validate(
            {
                "id": "demo",
                "version": "1.0.0",
                "name": "Demo",
                "description": "d",
                "license": "MIT",
                "tasks": [{"id": "t", "dataset": reference}],
            }
        )


def test_symlink_escape_is_detected(suite_root: Path) -> None:
    suite_dir = write_suite(suite_root)
    outside = suite_root.parent / "outside.jsonl"
    write_dataset(
        outside, [{"id": "x", "prompt": "p", "metric": "exact_match", "reference": "r"}]
    )
    link = suite_dir / "datasets" / "linked.jsonl"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks require a privilege on this Windows account")
    definition = {
        "id": "linked-suite",
        "version": "1.0.0",
        "name": "Linked",
        "description": "d",
        "license": "MIT",
        "tasks": [{"id": "t", "dataset": "datasets/linked.jsonl"}],
    }
    link_dir = suite_root / "linked-suite"
    link_dir.mkdir()
    (link_dir / "suite.yaml").write_text(yaml.safe_dump(definition), encoding="utf-8")
    with pytest.raises(SuiteError, match="escapes the suite directory"):
        load_suite_dir(link_dir)


def test_missing_dataset_reference_reports_task(suite_root: Path) -> None:
    suite_dir = write_suite(suite_root, suite_id="missing-dataset")
    (suite_dir / "datasets" / "a.jsonl").unlink()
    with pytest.raises(SuiteError, match="dataset file not found"):
        load_suite("missing-dataset", [suite_root])


def test_expected_metric_mismatch_is_a_validation_error(suite_root: Path) -> None:
    suite_dir = write_suite(suite_root)
    write_dataset(
        suite_dir / "datasets" / "a.jsonl",
        [{"id": "a1", "prompt": "one", "metric": "contains", "reference": "1"}],
    )
    loaded = load_suite("demo-suite", [suite_root])
    errors = validate_suite(loaded)
    assert any("expects metric 'exact_match'" in error for error in errors)


def test_code_tests_without_code_evaluator_is_a_validation_error(suite_root: Path) -> None:
    suite_dir = write_suite(
        suite_root,
        tasks=[{"id": "code", "dataset": "datasets/code.jsonl", "expected_metric": "code_tests"}],
    )
    write_dataset(
        suite_dir / "datasets" / "code.jsonl",
        [
            {
                "id": "c1",
                "prompt": "return 1",
                "metric": "code_tests",
                "tests": "def test_one():\n    assert True\n",
            }
        ],
    )
    loaded = load_suite("demo-suite", [suite_root])
    errors = validate_suite(loaded)
    assert any("evaluators.code" in error for error in errors)


def test_definition_fingerprint_is_stable_and_content_sensitive(suite_root: Path) -> None:
    first_dir = write_suite(suite_root, suite_id="fp-a")
    second_dir = write_suite(suite_root.parent / "fp-root", suite_id="fp-a")
    first = load_suite_dir(first_dir)
    second = load_suite_dir(second_dir)
    assert first.definition.fingerprint() == second.definition.fingerprint()

    changed = load_suite_dir(first_dir)
    changed.definition.tasks[0].weight = 2.0
    assert changed.definition.fingerprint() != first.definition.fingerprint()


def test_category_field_is_optional_and_counted(suite_root: Path) -> None:
    suite_dir = write_suite(suite_root)
    write_dataset(
        suite_dir / "datasets" / "a.jsonl",
        [
            {
                "id": "a1",
                "prompt": "one",
                "metric": "exact_match",
                "reference": "1",
                "category": "math",
            },
            {
                "id": "a2",
                "prompt": "two",
                "metric": "exact_match",
                "reference": "2",
                "category": "words",
            },
        ],
    )
    loaded = load_suite("demo-suite", [suite_root])
    dataset = loaded.datasets["task-a"]
    assert dataset.category_counts() == {"math": 1, "words": 1}
