from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from minieval_bench.cli import app
from minieval_bench.suites.compare import build_suite_comparison_payload
from minieval_bench.suites.export import SuiteExportError, export_suite_archive
from minieval_bench.suites.registry import load_suite_dir
from minieval_bench.suites.runner import SuiteRunOptions, run_suite
from minieval_bench.suites.sampling import SamplingPlan

runner = CliRunner()
WIDE = {"COLUMNS": "220"}


def write_jsonl(path: Path, entries: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n", encoding="utf-8"
    )
    return path


def write_compare_suite(root: Path) -> Path:
    suite_dir = root / "suites" / "compare-suite"
    definition = {
        "id": "compare-suite",
        "version": "1.0.0",
        "name": "Compare",
        "description": "Two deterministic tasks.",
        "license": "MIT",
        "defaults": {"repeats": 2, "seed": 3, "concurrency": 2},
        "tasks": [
            {"id": "pass", "dataset": "datasets/pass.jsonl", "weight": 1.0},
            {"id": "fail", "dataset": "datasets/fail.jsonl", "weight": 1.0},
        ],
    }
    suite_dir.mkdir(parents=True)
    (suite_dir / "suite.yaml").write_text(yaml.safe_dump(definition), encoding="utf-8")
    write_jsonl(
        suite_dir / "datasets" / "pass.jsonl",
        [
            {"id": "p1", "prompt": "a", "metric": "contains", "reference": "mock"},
            {"id": "p2", "prompt": "b", "metric": "contains", "reference": "mock"},
        ],
    )
    write_jsonl(
        suite_dir / "datasets" / "fail.jsonl",
        [{"id": "f1", "prompt": "c", "metric": "contains", "reference": "zzz"}],
    )
    return suite_dir


def run_once(
    tmp_path: Path, suite_dir: Path, output_name: str, sampling: SamplingPlan | None = None
) -> Path:
    models_file = tmp_path / "models.yaml"
    if not models_file.exists():
        models_file.write_text(
            "models:\n  - name: mock-a\n    provider: mock\n", encoding="utf-8"
        )
    output = tmp_path / output_name
    options = SuiteRunOptions(
        models_path=models_file,
        output_dir=output,
        sampling=sampling or SamplingPlan(),
    )
    run_suite(load_suite_dir(suite_dir), options)
    return output


def test_identical_protocols_produce_detailed_comparison(tmp_path: Path) -> None:
    suite_dir = write_compare_suite(tmp_path)
    first = run_once(tmp_path, suite_dir, "run-a")
    second = run_once(tmp_path, suite_dir, "run-b")

    payload = build_suite_comparison_payload([first, second])

    assert payload["protocolEqual"] is True
    assert payload["differences"] == []
    assert [task["id"] for task in payload["tasks"]] == ["pass", "fail"]
    pass_task = payload["tasks"][0]
    row = pass_task["rows"][0]
    assert row["model"] == "mock-a"
    values = row["values"]["deterministic"]
    assert len(values) == 2
    assert all(value == 1.0 for value in values)
    paired = payload["paired"]
    assert all(
        entry["units"] == (2 if entry["task"] == "pass" else 1) for entry in paired
    )
    assert {entry["task"] for entry in paired} == {"pass", "fail"}


def test_different_sampling_blocks_comparison(tmp_path: Path) -> None:
    suite_dir = write_compare_suite(tmp_path)
    first = run_once(tmp_path, suite_dir, "full-run")
    second = run_once(
        tmp_path,
        suite_dir,
        "sampled-run",
        SamplingPlan(method="random", count=1, seed=2),
    )

    payload = build_suite_comparison_payload([first, second])

    assert payload["protocolEqual"] is False
    assert any("sampling" in difference for difference in payload["differences"])
    assert "tasks" not in payload


def test_dataset_change_blocks_comparison(tmp_path: Path) -> None:
    suite_dir = write_compare_suite(tmp_path)
    first = run_once(tmp_path, suite_dir, "before")
    write_jsonl(
        suite_dir / "datasets" / "fail.jsonl",
        [{"id": "f1", "prompt": "changed", "metric": "contains", "reference": "zzz"}],
    )
    second = run_once(tmp_path, suite_dir, "after")

    payload = build_suite_comparison_payload([first, second])

    assert payload["protocolEqual"] is False
    assert any("dataset content changed" in d for d in payload["differences"])
    assert "tasks" not in payload


def test_model_change_blocks_comparison(tmp_path: Path) -> None:
    suite_dir = write_compare_suite(tmp_path)
    first = run_once(tmp_path, suite_dir, "model-a")
    other_models = tmp_path / "models-b.yaml"
    other_models.write_text(
        "models:\n  - name: mock-a\n    provider: mock\n    temperature: 0.7\n",
        encoding="utf-8",
    )
    output = tmp_path / "model-b"
    run_suite(
        load_suite_dir(suite_dir),
        SuiteRunOptions(models_path=other_models, output_dir=output),
    )

    payload = build_suite_comparison_payload([first, output])

    assert payload["protocolEqual"] is False
    assert any("model configuration changed" in d for d in payload["differences"])


def test_suite_version_change_blocks_comparison(tmp_path: Path) -> None:
    suite_dir = write_compare_suite(tmp_path)
    first = run_once(tmp_path, suite_dir, "v1")
    definition = yaml.safe_load((suite_dir / "suite.yaml").read_text(encoding="utf-8"))
    definition["version"] = "1.1.0"
    (suite_dir / "suite.yaml").write_text(yaml.safe_dump(definition), encoding="utf-8")
    second = run_once(tmp_path, suite_dir, "v2")

    payload = build_suite_comparison_payload([first, second])

    assert payload["protocolEqual"] is False
    assert any("suite version changed" in d for d in payload["differences"])


def test_insufficient_results_yield_no_paired(tmp_path: Path) -> None:
    suite_dir = write_compare_suite(tmp_path)
    first = run_once(tmp_path, suite_dir, "ok-run")
    empty_models = tmp_path / "empty-models.yaml"

    from minieval_bench.suites.runner import UserOverrides

    second = tmp_path / "failed-run"
    failing = tmp_path / "failing-models.yaml"
    failing.write_text("models:\n  - name: mock-a\n    provider: mock\n", encoding="utf-8")
    empty_models.unlink(missing_ok=True)
    output = second
    options = SuiteRunOptions(
        models_path=failing,
        output_dir=output,
        overrides=UserOverrides(repeats=1),
    )
    import minieval_bench.runner as engine_runner

    class FailingProvider:
        def __init__(self, model):
            self.model = model

        def complete(self, prompt, system, seed):
            raise ConnectionError("network down")

    original = engine_runner.build_provider
    engine_runner.build_provider = lambda model: FailingProvider(model)
    try:
        run_suite(load_suite_dir(suite_dir), options)
    finally:
        engine_runner.build_provider = original

    payload = build_suite_comparison_payload([first, second])
    assert payload["protocolEqual"] is False
    assert any("run parameters changed" in d for d in payload["differences"])


def test_suite_compare_cli_identical(tmp_path: Path) -> None:
    suite_dir = write_compare_suite(tmp_path)
    first = run_once(tmp_path, suite_dir, "cli-a")
    second = run_once(tmp_path, suite_dir, "cli-b")
    output = tmp_path / "comparison.html"

    result = runner.invoke(
        app,
        [
            "suite", "compare",
            "--run", str(first),
            "--run", str(second),
            "--output", str(output),
        ],
        env=WIDE,
    )

    assert result.exit_code == 0
    assert "Protocols are identical" in result.output
    html = output.read_text(encoding="utf-8")
    assert "suite-compare" not in html[html.index("</title>"):html.index("</title>") + 40]
    assert 'type="application/json"' in html


def test_suite_compare_cli_different_protocols(tmp_path: Path) -> None:
    suite_dir = write_compare_suite(tmp_path)
    first = run_once(tmp_path, suite_dir, "cmp-a")
    second = run_once(
        tmp_path, suite_dir, "cmp-b", SamplingPlan(method="random", count=1, seed=9)
    )
    output = tmp_path / "comparison.html"

    result = runner.invoke(
        app,
        [
            "suite", "compare",
            "--run", str(first),
            "--run", str(second),
            "--output", str(output),
        ],
        env=WIDE,
    )

    assert result.exit_code == 0
    assert "Protocols differ" in result.output
    assert any("sampling" in line for line in result.output.splitlines())


def test_suite_export_minimal_mode(tmp_path: Path) -> None:
    suite_dir = write_compare_suite(tmp_path)
    output_dir = run_once(tmp_path, suite_dir, "export-run")
    archive = tmp_path / "export.zip"

    summary = export_suite_archive(output_dir, archive)

    assert summary["include_responses"] is False
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        assert "suite-manifest.json" in names
        assert "benchmark.lock.json" in names
        assert "checksums.json" in names
        assert "tasks/pass/stats.json" in names
        assert "tasks/pass/results.json" not in names
        checksums = json.loads(zf.read("checksums.json"))
        import hashlib

        for name, digest in checksums.items():
            assert hashlib.sha256(zf.read(name)).hexdigest() == digest
        metadata = json.loads(zf.read("export-metadata.json"))
        assert metadata["include_responses"] is False
        stats = json.loads(zf.read("tasks/pass/stats.json"))
        assert "trials" not in stats
        assert stats["stats"]


def test_suite_export_with_responses(tmp_path: Path) -> None:
    suite_dir = write_compare_suite(tmp_path)
    output_dir = run_once(tmp_path, suite_dir, "export-resp")
    archive = tmp_path / "export-resp.zip"

    summary = export_suite_archive(output_dir, archive, include_responses=True)

    assert summary["include_responses"] is True
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        assert "tasks/pass/results.json" in names
        results = json.loads(zf.read("tasks/pass/results.json"))
        assert "trials" in results


def test_suite_export_refuses_secrets(tmp_path: Path) -> None:
    suite_dir = write_compare_suite(tmp_path)
    output_dir = run_once(tmp_path, suite_dir, "secret-run")
    results_path = output_dir / "tasks" / "pass" / "results.json"
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    payload["trials"][0]["response"] = "my key is sk-ABCDEFGHIJKLMNO"
    results_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SuiteExportError, match="credential"):
        export_suite_archive(output_dir, tmp_path / "secret.zip", include_responses=True)


def test_suite_export_cli(tmp_path: Path) -> None:
    suite_dir = write_compare_suite(tmp_path)
    output_dir = run_once(tmp_path, suite_dir, "cli-export")
    archive = tmp_path / "cli.zip"

    result = runner.invoke(
        app,
        ["suite", "export", "--input", str(output_dir), "--output", str(archive)],
        env=WIDE,
    )
    assert result.exit_code == 0
    assert "Minimal mode" in result.output
    assert archive.is_file()

    result = runner.invoke(
        app,
        [
            "suite", "export", "--input", str(output_dir),
            "--output", str(tmp_path / "cli-full.zip"),
            "--include-responses",
        ],
        env=WIDE,
    )
    assert result.exit_code == 0
    assert "Individual responses are included" in result.output

    result = runner.invoke(
        app,
        ["suite", "export", "--input", str(tmp_path / "missing"), "--output", str(archive)],
        env=WIDE,
    )
    assert result.exit_code == 1
