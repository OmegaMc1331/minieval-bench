from __future__ import annotations

import json
from pathlib import Path

import pytest

from minieval_bench import __version__
from minieval_bench.comparison import build_comparison_payload
from minieval_bench.models import Manifest, ModelConfig, ModelStats
from minieval_bench.reports import write_reports
from test_reports import OLD_RESULTS as LEGACY_RESULTS


def make_manifest(**overrides) -> Manifest:
    fields = {
        "minieval_version": __version__,
        "benchmark": "unit",
        "created_at_utc": "2026-11-14T00:00:00+00:00",
        "seed": 42,
        "repeats": 2,
        "concurrency": 2,
        "dataset_path": "dataset.jsonl",
        "dataset_sha256": "a" * 64,
        "dataset_entries": 3,
        "models": [ModelConfig(name="model-a", provider="mock")],
        "duration_seconds": 1.0,
        "trial_seeds": [1, 2],
        "shuffle": False,
    }
    return Manifest(**{**fields, **overrides})


def make_stats(**overrides) -> ModelStats:
    fields = {
        "model": "model-a",
        "requests": 6,
        "succeeded": 6,
        "failed": 0,
        "success_rate": 1.0,
        "evaluable": 6,
        "passed": 3,
        "score": 0.5,
        "median_latency_ms": 20.0,
        "p95_latency_ms": 40.0,
        "input_tokens": 100,
        "output_tokens": 50,
        "cost_usd": 0.001,
        "errors": {},
        "answer_accuracy": 0.5,
        "evaluation_coverage": 1.0,
        "overall_pass_rate": 0.5,
        "incorrect": 3,
        "median_queue_ms": 1.0,
        "input_tokens_reported": 6,
        "output_tokens_reported": 6,
        "cost_reported": 6,
    }
    return ModelStats(**{**fields, **overrides})


def write_run(directory: Path, manifest: Manifest, stats: list[ModelStats]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    write_reports(directory, manifest, stats, [], html=False)
    return directory


def indicator(payload: dict, model: str, key: str) -> dict:
    target = next(item for item in payload["commonModels"] if item["model"] == model)
    return next(item for item in target["indicators"] if item["key"] == key)


def test_comparable_runs_produce_deltas_without_warnings(tmp_path):
    first = write_run(tmp_path / "a", make_manifest(created_at_utc="2026-11-14T01:00:00+00:00"),
                      [make_stats()])
    second = write_run(tmp_path / "b", make_manifest(created_at_utc="2026-11-14T02:00:00+00:00"),
                       [make_stats(answer_accuracy=0.75, median_latency_ms=30.0)])

    payload = build_comparison_payload([first, second])

    assert payload["warnings"] == []
    assert [run["label"] for run in payload["runs"]] == ["unit (1)", "unit (2)"]
    accuracy = indicator(payload, "model-a", "answer_accuracy")
    assert accuracy["values"] == [0.5, 0.75]
    assert accuracy["deltaAbs"] == pytest.approx(0.25)
    assert accuracy["deltaRel"] == pytest.approx(0.5)
    latency = indicator(payload, "model-a", "median_latency_ms")
    assert latency["deltaAbs"] == pytest.approx(10.0)


def test_different_datasets_trigger_warning(tmp_path):
    first = write_run(tmp_path / "a", make_manifest(), [make_stats()])
    second = write_run(
        tmp_path / "b",
        make_manifest(dataset_sha256="b" * 64, benchmark="other"),
        [make_stats()],
    )

    payload = build_comparison_payload([first, second])

    assert any("Datasets differ" in warning for warning in payload["warnings"])


def test_different_model_configuration_trigger_warning(tmp_path):
    models_a = [ModelConfig(name="model-a", provider="mock", model="model-a")]
    models_b = [
        ModelConfig(name="model-a", provider="mock", model="model-a", temperature=0.7)
    ]
    first = write_run(tmp_path / "a", make_manifest(models=models_a), [make_stats()])
    second = write_run(tmp_path / "b", make_manifest(models=models_b), [make_stats()])

    payload = build_comparison_payload([first, second])

    assert any("different configuration" in warning for warning in payload["warnings"])
    target = payload["commonModels"][0]
    assert target["configConsistent"] is False
    assert target["configFields"] == ["temperature"]


def test_differing_conditions_and_versions_are_reported(tmp_path):
    first = write_run(tmp_path / "a", make_manifest(), [make_stats()])
    second = write_run(
        tmp_path / "b",
        make_manifest(seed=7, repeats=3, concurrency=8, minieval_version="0.1.1"),
        [make_stats()],
    )

    payload = build_comparison_payload([first, second])

    joined = " ".join(payload["warnings"])
    assert "seed" in joined
    assert "repeats" in joined
    assert "concurrency" in joined
    assert "0.1.1" in joined


def test_missing_values_do_not_produce_deltas(tmp_path):
    first = write_run(tmp_path / "a", make_manifest(), [make_stats(cost_usd=None)])
    second = write_run(tmp_path / "b", make_manifest(benchmark="other"), [make_stats()])

    payload = build_comparison_payload([first, second])

    cost = indicator(payload, "model-a", "cost_usd")
    assert cost["deltaAbs"] is None
    assert cost["deltaRel"] is None


def test_zero_reference_value_has_no_relative_delta(tmp_path):
    first = write_run(tmp_path / "a", make_manifest(), [make_stats(cost_usd=0.0)])
    second = write_run(tmp_path / "b", make_manifest(benchmark="other"), [make_stats(cost_usd=0.5)])

    payload = build_comparison_payload([first, second])

    cost = indicator(payload, "model-a", "cost_usd")
    assert cost["deltaAbs"] == pytest.approx(0.5)
    assert cost["deltaRel"] is None


def test_all_failed_runs_keep_indicators_unavailable(tmp_path):
    failed_stats = make_stats(
        requests=2,
        succeeded=0,
        failed=2,
        success_rate=0.0,
        evaluable=0,
        passed=0,
        score=None,
        median_latency_ms=None,
        p95_latency_ms=None,
        input_tokens=None,
        output_tokens=None,
        cost_usd=None,
        answer_accuracy=None,
        overall_pass_rate=0.0,
        incorrect=0,
        input_tokens_reported=0,
        output_tokens_reported=0,
        cost_reported=0,
    )
    first = write_run(tmp_path / "a", make_manifest(), [failed_stats])
    second = write_run(tmp_path / "b", make_manifest(benchmark="other"), [failed_stats])

    payload = build_comparison_payload([first, second])

    accuracy = indicator(payload, "model-a", "answer_accuracy")
    assert accuracy["values"] == [None, None]
    assert accuracy["deltaAbs"] is None
    assert payload["warnings"] == []


def test_three_runs_compare_first_against_last(tmp_path):
    runs = []
    for index, accuracy in enumerate([0.2, 0.5, 0.8]):
        runs.append(
            write_run(
                tmp_path / f"run{index}",
                make_manifest(benchmark=f"run-{index}"),
                [make_stats(answer_accuracy=accuracy)],
            )
        )

    payload = build_comparison_payload(runs)

    accuracy = indicator(payload, "model-a", "answer_accuracy")
    assert accuracy["values"] == [0.2, 0.5, 0.8]
    assert accuracy["deltaAbs"] == pytest.approx(0.6)
    assert accuracy["deltaRel"] == pytest.approx(3.0)


def test_models_not_present_everywhere_are_listed(tmp_path):
    first = write_run(
        tmp_path / "a",
        make_manifest(models=[ModelConfig(name="model-a", provider="mock")]),
        [make_stats()],
    )
    second = write_run(
        tmp_path / "b",
        make_manifest(
            benchmark="other",
            models=[
                ModelConfig(name="model-a", provider="mock"),
                ModelConfig(name="model-b", provider="mock"),
            ],
        ),
        [make_stats(), make_stats(model="model-b")],
    )

    payload = build_comparison_payload([first, second])

    assert [item["model"] for item in payload["commonModels"]] == ["model-a"]
    assert payload["modelsOnlyIn"] == {payload["runs"][1]["label"]: ["model-b"]}


def test_no_common_models_warns(tmp_path):
    first = write_run(
        tmp_path / "a",
        make_manifest(models=[ModelConfig(name="model-a", provider="mock")]),
        [make_stats()],
    )
    second = write_run(
        tmp_path / "b",
        make_manifest(
            benchmark="other",
            models=[ModelConfig(name="model-z", provider="mock")],
        ),
        [make_stats(model="model-z")],
    )

    payload = build_comparison_payload([first, second])

    assert payload["commonModels"] == []
    assert any("No model name is common" in warning for warning in payload["warnings"])


def test_comparison_accepts_v010_results(tmp_path):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "results.json").write_text(json.dumps(LEGACY_RESULTS), encoding="utf-8")
    modern = write_run(
        tmp_path / "modern",
        make_manifest(models=[ModelConfig(name="old-mock", provider="mock")]),
        [make_stats(model="old-mock")],
    )

    payload = build_comparison_payload([legacy, modern])

    assert len(payload["runs"]) == 2
    accuracy = indicator(payload, "old-mock", "answer_accuracy")
    assert accuracy["values"][0] == pytest.approx(2 / 3)
