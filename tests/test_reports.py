from __future__ import annotations

import json

import pytest

from minieval_bench import __version__
from minieval_bench.models import Manifest, ModelConfig, ModelStats
from minieval_bench.reports import (
    build_manifest,
    dataset_fingerprint,
    load_results,
    markdown_report,
    write_reports,
)
from minieval_bench.runner import compute_stats, run_benchmark
from test_runner import load_written

OLD_RESULTS = {
    "manifest": {
        "minieval_version": "0.1.0",
        "benchmark": "legacy",
        "created_at_utc": "2026-11-01T00:00:00+00:00",
        "seed": 7,
        "repeats": 2,
        "concurrency": 2,
        "dataset_path": "legacy.jsonl",
        "dataset_sha256": "0" * 64,
        "dataset_entries": 2,
        "models": [{"name": "old-mock", "provider": "mock"}],
    },
    "stats": [
        {
            "model": "old-mock",
            "requests": 4,
            "succeeded": 3,
            "failed": 1,
            "success_rate": 0.75,
            "evaluable": 3,
            "passed": 2,
            "score": 2 / 3,
            "median_latency_ms": 10.0,
            "p95_latency_ms": 20.0,
            "input_tokens": 30,
            "output_tokens": 12,
            "cost_usd": None,
            "errors": {"TimeoutError": 1},
        }
    ],
    "trials": [],
}


@pytest.fixture
def written_reports(tmp_path, write_benchmark, mock_config, valid_entries):
    config, entries = load_written(write_benchmark(mock_config, valid_entries))
    trials, started_at, duration = run_benchmark(config, entries, show_progress=False)
    manifest = build_manifest(config, entries, started_at, duration)
    stats = compute_stats(trials)
    output_dir = tmp_path / "results"
    write_reports(output_dir, manifest, stats, trials)
    return output_dir, manifest, stats, trials


def test_write_reports_creates_json_and_markdown(written_reports):
    output_dir, manifest, stats, _ = written_reports

    assert (output_dir / "manifest.json").is_file()
    assert (output_dir / "report.md").is_file()

    results = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    assert results["manifest"]["benchmark"] == manifest.benchmark
    assert len(results["stats"]) == len(stats)
    assert len(results["trials"]) > 0


def test_manifest_records_version_seed_seeds_and_duration(written_reports):
    output_dir, _, _, _ = written_reports

    saved = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    assert saved["minieval_version"] == __version__
    assert saved["seed"] == 1
    assert saved["repeats"] == 2
    assert saved["concurrency"] == 2
    assert saved["dataset_entries"] == 3
    assert len(saved["dataset_sha256"]) == 64
    assert saved["created_at_utc"].endswith("+00:00")
    assert saved["duration_seconds"] > 0
    assert len(saved["trial_seeds"]) == 2
    assert saved["trial_seeds"][0] != saved["trial_seeds"][1]
    assert saved["shuffle"] is False


def test_stats_json_contains_new_indicators_and_coverage(written_reports):
    output_dir, _, _, _ = written_reports

    results = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    model_stats = results["stats"][0]
    assert model_stats["answer_accuracy"] == model_stats["score"]
    assert model_stats["evaluation_coverage"] == model_stats["success_rate"]
    assert model_stats["overall_pass_rate"] is not None
    assert model_stats["input_tokens_reported"] == model_stats["succeeded"]
    assert model_stats["cost_reported"] == 0
    assert model_stats["cost_usd"] is None


def test_reports_contain_no_secrets(written_reports):
    output_dir, _, _, _ = written_reports

    for path in output_dir.iterdir():
        content = path.read_text(encoding="utf-8").lower()
        for secret_marker in ("api_key", "apikey", "authorization", "sk-", "bearer"):
            assert secret_marker not in content


def test_markdown_report_marks_partial_consumption():
    manifest = make_manifest()
    stats = [make_stats(input_tokens=7, input_tokens_reported=7, succeeded=10)]

    markdown = markdown_report(manifest, stats)

    assert "7 (7/10)" in markdown
    assert "**Overall** = correct answers / planned requests" in markdown


def test_markdown_report_omits_error_section_without_errors():
    markdown = markdown_report(make_manifest(), [make_stats(errors={})])

    assert "## Errors" not in markdown


def test_markdown_report_renders_error_breakdown():
    markdown = markdown_report(make_manifest(), [make_stats()])

    assert "## Errors" in markdown
    assert "TimeoutError" in markdown


def test_load_results_roundtrip(written_reports):
    output_dir, manifest, stats, _ = written_reports

    loaded_manifest, loaded_stats = load_results(output_dir)

    assert loaded_manifest == manifest
    assert loaded_stats == stats


def test_load_results_derives_indicators_from_v010_reports(tmp_path):
    results_dir = tmp_path / "legacy"
    results_dir.mkdir()
    (results_dir / "results.json").write_text(json.dumps(OLD_RESULTS), encoding="utf-8")

    manifest, stats = load_results(results_dir)

    assert manifest.minieval_version == "0.1.0"
    assert manifest.duration_seconds is None
    assert manifest.trial_seeds is None
    model_stats = stats[0]
    assert model_stats.answer_accuracy == pytest.approx(2 / 3)
    assert model_stats.evaluation_coverage == pytest.approx(0.75)
    assert model_stats.overall_pass_rate == pytest.approx(0.5)
    assert model_stats.incorrect == 1
    assert model_stats.input_tokens_reported is None
    assert model_stats.cost_reported is None


def test_load_results_requires_results_json(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_results(tmp_path)


def test_dataset_fingerprint_tracks_content(tmp_path):
    dataset = tmp_path / "d.jsonl"
    dataset.write_text("a\n", encoding="utf-8")
    first = dataset_fingerprint(dataset)
    dataset.write_text("b\n", encoding="utf-8")

    assert dataset_fingerprint(dataset) != first


def make_manifest() -> Manifest:
    return Manifest(
        minieval_version=__version__,
        benchmark="unit",
        created_at_utc="2026-01-01T00:00:00+00:00",
        seed=1,
        repeats=1,
        concurrency=1,
        dataset_path="dataset.jsonl",
        dataset_sha256="0" * 64,
        dataset_entries=1,
        models=[ModelConfig(name="m", provider="mock")],
    )


def make_stats(**overrides) -> ModelStats:
    fields = {
        "model": "m",
        "requests": 2,
        "succeeded": 1,
        "failed": 1,
        "success_rate": 0.5,
        "evaluable": 1,
        "passed": 1,
        "score": 1.0,
        "median_latency_ms": 10.0,
        "p95_latency_ms": 10.0,
        "input_tokens": 5,
        "output_tokens": 5,
        "cost_usd": None,
        "errors": {"TimeoutError": 1},
    }
    return ModelStats(**{**fields, **overrides})
