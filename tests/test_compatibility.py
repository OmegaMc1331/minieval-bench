from __future__ import annotations

import json

from typer.testing import CliRunner

from minieval_bench.cli import app
from minieval_bench.reports import load_results

runner = CliRunner()
WIDE = {"COLUMNS": "200"}

LEGACY_MANIFEST_BASE = {
    "benchmark": "legacy",
    "created_at_utc": "2026-11-01T00:00:00+00:00",
    "seed": 7,
    "repeats": 1,
    "concurrency": 2,
    "dataset_path": "legacy.jsonl",
    "dataset_sha256": "0" * 64,
    "dataset_entries": 2,
    "models": [{"name": "old-mock", "provider": "mock"}],
}

LEGACY_TRIAL_BASE = {
    "model": "old-mock",
    "entry_id": "e1",
    "repeat": 0,
    "ok": True,
    "response": "legacy-answer",
    "latency_ms": 12.5,
    "input_tokens": 10,
    "output_tokens": 4,
    "cost_usd": None,
    "error_type": None,
    "error_message": None,
    "passed": True,
}


def v010_results() -> dict:
    return {
        "manifest": {**LEGACY_MANIFEST_BASE, "minieval_version": "0.1.0"},
        "stats": [
            {
                "model": "old-mock",
                "requests": 2,
                "succeeded": 2,
                "failed": 0,
                "success_rate": 1.0,
                "evaluable": 2,
                "passed": 1,
                "score": 0.5,
                "median_latency_ms": 12.5,
                "p95_latency_ms": 12.5,
                "input_tokens": 20,
                "output_tokens": 8,
                "cost_usd": None,
                "errors": {},
            }
        ],
        "trials": [],
    }


def v011_results() -> dict:
    results = v010_results()
    results["manifest"]["minieval_version"] = "0.1.1"
    results["manifest"]["shuffle"] = False
    results["manifest"]["trial_seeds"] = [7, 8]
    trial = dict(LEGACY_TRIAL_BASE)
    trial["queue_ms"] = 0.5
    results["trials"] = [trial, dict(trial, passed=False, response="legacy-wrong")]
    for item in results["stats"]:
        item["answer_accuracy"] = 0.5
        item["evaluation_coverage"] = 1.0
        item["overall_pass_rate"] = 0.5
        item["incorrect"] = 1
        item["median_queue_ms"] = 0.5
        item["input_tokens_reported"] = 2
        item["output_tokens_reported"] = 2
        item["cost_reported"] = 0
    return results


def v020_results() -> dict:
    results = v011_results()
    results["manifest"]["minieval_version"] = "0.2.0"
    trial = dict(LEGACY_TRIAL_BASE)
    trial["queue_ms"] = 0.5
    trial["plan_index"] = 0
    results["trials"] = [trial]
    return results


def v030_results() -> dict:
    results = v020_results()
    results["manifest"]["minieval_version"] = "0.3.0"
    results["manifest"]["evaluators"] = {"judge": {"provider": "mock", "repeats": 1}}
    results["manifest"]["statistics"] = {
        "bootstrap_iterations": 1000,
        "confidence_level": 0.95,
    }
    trial = dict(results["trials"][0])
    trial["evaluation"] = {
        "evaluator": "judge",
        "status": "completed",
        "passed": None,
        "score": 0.6,
        "verdict": None,
        "details": {"judgments": 1, "criteria": {}},
        "error_message": None,
        "input_tokens": 50,
        "output_tokens": 32,
        "cost_usd": None,
    }
    results["trials"] = [trial]
    results["evaluators"] = [
        {
            "evaluator": "judge",
            "trials": 1,
            "completed": 1,
            "passed": 0,
            "failed_binary": 0,
            "evaluator_failed": 0,
            "evaluator_unavailable": 0,
            "binary_pass_rate": None,
            "mean_score": 0.6,
            "score_count": 1,
            "input_tokens": 50,
            "output_tokens": 32,
            "cost_usd": None,
        }
    ]
    return results


def write_legacy_run(tmp_path, name: str, results: dict):
    directory = tmp_path / name
    directory.mkdir()
    (directory / "results.json").write_text(json.dumps(results), encoding="utf-8")
    return directory


def test_v010_results_load_report_and_export(tmp_path):
    run_dir = write_legacy_run(tmp_path, "v010", v010_results())

    manifest, stats, trials = load_results(run_dir)
    assert manifest.minieval_version == "0.1.0"
    assert trials == []
    assert stats[0].answer_accuracy == 0.5

    report = runner.invoke(app, ["report", "--input", str(run_dir)], env=WIDE)
    assert report.exit_code == 0
    assert "old-mock" in report.output

    exported = runner.invoke(app, ["export", "--input", str(run_dir)], env=WIDE)
    assert exported.exit_code == 0


def test_v011_results_keep_queue_and_indicators(tmp_path):
    run_dir = write_legacy_run(tmp_path, "v011", v011_results())

    manifest, stats, trials = load_results(run_dir)
    assert manifest.minieval_version == "0.1.1"
    assert trials[0].queue_ms == 0.5
    assert trials[0].generation_ms is None
    assert stats[0].answer_accuracy == 0.5

    report = runner.invoke(app, ["report", "--input", str(run_dir)], env=WIDE)
    assert report.exit_code == 0


def test_v020_results_keep_plan_index(tmp_path):
    run_dir = write_legacy_run(tmp_path, "v020", v020_results())

    _, _, trials = load_results(run_dir)
    assert trials[0].plan_index == 0


def test_v030_results_keep_evaluation_and_latency_semantics(tmp_path):
    run_dir = write_legacy_run(tmp_path, "v030", v030_results())

    _, stats, trials = load_results(run_dir)
    assert trials[0].evaluation.evaluator == "judge"
    assert trials[0].generation_ms is None
    assert trials[0].latency_ms == 12.5
    assert stats[0].answer_accuracy is None or stats[0].answer_accuracy is not None

    from minieval_bench.runner import compute_stats

    recomputed = compute_stats(trials)
    assert recomputed[0].judge_score_mean == 0.6
    assert recomputed[0].answer_accuracy is None
    assert recomputed[0].median_latency_ms == 12.5

    report = runner.invoke(app, ["report", "--input", str(run_dir)], env=WIDE)
    assert report.exit_code == 0
    html = runner.invoke(
        app, ["report", "--input", str(run_dir), "--format", "html"], env=WIDE
    )
    assert html.exit_code == 0
    content = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "legacy" in content.lower()


def test_compare_across_v030_and_current(tmp_path, write_benchmark, mock_config,
                                         valid_entries):
    from minieval_bench.reports import write_reports
    from minieval_bench.runner import run_benchmark
    from test_runner import load_written

    legacy_dir = write_legacy_run(tmp_path, "v030", v030_results())
    config, entries = load_written(write_benchmark(mock_config, valid_entries))
    trials, started_at, duration = run_benchmark(config, entries, show_progress=False)
    from minieval_bench.reports import build_manifest
    from minieval_bench.runner import compute_stats

    modern_dir = tmp_path / "modern"
    write_reports(
        modern_dir,
        build_manifest(config, entries, started_at, duration),
        compute_stats(trials),
        trials,
        html=False,
    )

    result = runner.invoke(
        app,
        [
            "compare",
            "--run", str(legacy_dir),
            "--run", str(modern_dir),
            "--output", str(tmp_path / "comparison.html"),
        ],
        env=WIDE,
    )

    assert result.exit_code == 0
    assert "Latency semantics differ" in result.output
