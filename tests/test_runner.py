from __future__ import annotations

from pathlib import Path

import pytest

from minieval_bench.config import load_config, load_dataset
from minieval_bench.models import DatasetEntry, EvaluationOutcome, ModelConfig
from minieval_bench.providers import MockProvider, ProviderResponse
from minieval_bench.runner import (
    PlannedTrial,
    build_plan,
    compute_stats,
    repeat_seed,
    run_benchmark,
    run_trial,
)


class FailingProvider:
    def complete(self, prompt: str, system: str | None, seed: int) -> ProviderResponse:
        raise RuntimeError("provider exploded")


class InterruptingProvider:
    def complete(self, prompt: str, system: str | None, seed: int) -> ProviderResponse:
        raise KeyboardInterrupt


def load_written(config_path: Path):
    config = load_config(config_path)
    return config, load_dataset(Path(config.dataset))


def make_entry(**overrides) -> DatasetEntry:
    fields = {"id": "e1", "prompt": "p", "metric": "regex", "reference": "^mock-[0-9a-f]+$"}
    return DatasetEntry(**{**fields, **overrides})


def make_planned(model: ModelConfig, entry: DatasetEntry, repeat: int = 0) -> PlannedTrial:
    return PlannedTrial(model=model, entry=entry, repeat=repeat, seed=repeat_seed(1, repeat))


def test_run_benchmark_runs_every_combination(write_benchmark, mock_config, valid_entries):
    mock_config["models"] = [
        {"name": "mock-a", "provider": "mock"},
        {"name": "mock-b", "provider": "mock"},
    ]
    config, entries = load_written(write_benchmark(mock_config, valid_entries))

    trials, _, _ = run_benchmark(config, entries, show_progress=False)

    assert len(trials) == 2 * 3 * 2
    assert all(trial.ok for trial in trials)
    assert {trial.model for trial in trials} == {"mock-a", "mock-b"}


def test_run_benchmark_is_deterministic_with_same_seed(write_benchmark, mock_config, valid_entries):
    config_path = write_benchmark(mock_config, valid_entries)

    def collect():
        config, entries = load_written(config_path)
        trials, _, _ = run_benchmark(config, entries, show_progress=False)
        return [
            (trial.model, trial.entry_id, trial.repeat, trial.passed, trial.response)
            for trial in trials
        ]

    assert collect() == collect()


def test_run_benchmark_changes_with_seed(write_benchmark, mock_config, valid_entries):
    def responses(seed: int) -> set[str | None]:
        mock_config["defaults"]["seed"] = seed
        config, entries = load_written(write_benchmark(mock_config, valid_entries))
        trials, _, _ = run_benchmark(config, entries, show_progress=False)
        return {trial.response for trial in trials}

    assert responses(1) != responses(2)


def test_run_benchmark_uses_distinct_seeds_per_repeat(write_benchmark, mock_config, valid_entries):
    config, entries = load_written(write_benchmark(mock_config, valid_entries))

    trials, _, _ = run_benchmark(config, entries, show_progress=False)

    responses_by_repeat: dict[int, set[str]] = {0: set(), 1: set()}
    for trial in trials:
        responses_by_repeat[trial.repeat].add(trial.response)
    assert responses_by_repeat[0].isdisjoint(responses_by_repeat[1])


def test_run_benchmark_records_queue_delay_and_plan_index(
    write_benchmark, mock_config, valid_entries
):
    config, entries = load_written(write_benchmark(mock_config, valid_entries))

    trials, _, _ = run_benchmark(config, entries, show_progress=False)

    assert all(trial.queue_ms is not None and trial.queue_ms >= 0 for trial in trials)
    assert sorted(trial.plan_index for trial in trials) == list(range(len(trials)))


def test_run_benchmark_sorts_trials(write_benchmark, mock_config, valid_entries):
    config, entries = load_written(write_benchmark(mock_config, valid_entries))

    trials, _, _ = run_benchmark(config, entries, show_progress=False)

    keys = [(trial.model, trial.entry_id, trial.repeat) for trial in trials]
    assert keys == sorted(keys)


def test_run_benchmark_priced_model_reports_cost(write_benchmark, mock_config, valid_entries):
    mock_config["models"] = [
        {"name": "free", "provider": "mock"},
        {"name": "priced", "provider": "mock", "input_cost_per_1m": 1.0, "output_cost_per_1m": 2.0},
    ]
    config, entries = load_written(write_benchmark(mock_config, valid_entries))

    trials, _, _ = run_benchmark(config, entries, show_progress=False)

    by_model: dict[str, list] = {}
    for trial in trials:
        by_model.setdefault(trial.model, []).append(trial)
    assert all(trial.cost_usd is None for trial in by_model["free"])
    assert all(trial.cost_usd is not None and trial.cost_usd > 0 for trial in by_model["priced"])


def test_repeat_seed_is_stable_and_distinct_per_repeat():
    assert repeat_seed(42, 0) == repeat_seed(42, 0)
    assert repeat_seed(42, 0) != repeat_seed(42, 1)
    assert repeat_seed(42, 1) != repeat_seed(43, 1)
    assert 0 <= repeat_seed(42, 0) < 2**32


def test_build_plan_is_reproducible_from_config_and_seed(
    write_benchmark, mock_config, valid_entries
):
    config_path = write_benchmark(mock_config, valid_entries)

    def plan_keys():
        config, entries = load_written(config_path)
        return [
            (item.model.name, item.entry.id, item.repeat, item.seed, item.plan_index)
            for item in build_plan(config, entries)
        ]

    assert plan_keys() == plan_keys()


def test_build_plan_shuffles_deterministically(write_benchmark, mock_config, valid_entries):
    mock_config["defaults"]["shuffle"] = True
    shuffled_path = write_benchmark(mock_config, valid_entries)
    shuffled_config, entries = load_written(shuffled_path)
    shuffled_first = build_plan(shuffled_config, entries)
    shuffled_second = build_plan(shuffled_config, entries)

    mock_config["defaults"]["shuffle"] = False
    ordered_config, _ = load_written(write_benchmark(mock_config, valid_entries))
    ordered = build_plan(ordered_config, entries)

    def keys(plan):
        return [(item.model.name, item.entry.id, item.repeat) for item in plan]

    assert keys(shuffled_first) == keys(shuffled_second)
    assert keys(shuffled_first) != keys(ordered)
    assert sorted(item.plan_index for item in shuffled_first) == list(range(len(shuffled_first)))


def test_run_benchmark_propagates_user_interrupt(
    write_benchmark, mock_config, valid_entries, monkeypatch
):
    monkeypatch.setattr(
        "minieval_bench.runner.build_provider", lambda model: InterruptingProvider()
    )
    config, entries = load_written(write_benchmark(mock_config, valid_entries))

    with pytest.raises(KeyboardInterrupt):
        run_benchmark(config, entries, show_progress=False)


def test_run_trial_records_provider_failure():
    model = ModelConfig(name="broken", provider="mock")
    planned = make_planned(model, make_entry())

    trial = run_trial(FailingProvider(), planned, queued_at=0.0)

    assert not trial.ok
    assert trial.error_type == "RuntimeError"
    assert "provider exploded" in trial.error_message
    assert trial.passed is None
    assert trial.response is None
    assert trial.latency_ms >= 0
    assert trial.queue_ms >= 0
    assert trial.plan_index == 0


def test_run_trial_evaluates_successful_response():
    model = ModelConfig(name="mock-a", provider="mock")
    planned = make_planned(model, make_entry())

    trial = run_trial(MockProvider(model), planned, queued_at=0.0)

    assert trial.ok
    assert trial.passed is True
    assert trial.input_tokens is not None
    assert trial.error_type is None


class SlowEvaluatorRunner:
    def run(self, code: str, tests: str):
        import time

        time.sleep(0.15)
        return EvaluationOutcome(
            evaluator="code_tests", status="completed", passed=True, score=1.0
        )


def test_run_trial_separates_generation_and_evaluation_latency():
    model = ModelConfig(name="mock-a", provider="mock")
    entry = make_entry(
        metric="code_tests", tests="def test_add():\n    assert True"
    )
    planned = make_planned(model, entry)

    trial = run_trial(
        MockProvider(model), planned, queued_at=0.0, code_runner=SlowEvaluatorRunner()
    )

    assert trial.generation_ms is not None
    assert trial.evaluation_ms is not None
    assert trial.evaluation_ms >= 140
    assert trial.generation_ms < 100
    assert trial.total_ms == pytest.approx(trial.generation_ms + trial.evaluation_ms)
    assert trial.latency_ms == pytest.approx(trial.generation_ms)
    assert trial.latency_ms < 100


def test_run_trial_failed_request_keeps_generation_latency():
    model = ModelConfig(name="broken", provider="mock")
    planned = make_planned(model, make_entry())

    trial = run_trial(FailingProvider(), planned, queued_at=0.0)

    assert trial.generation_ms is not None
    assert trial.latency_ms == pytest.approx(trial.generation_ms)
    assert trial.evaluation_ms is None
    assert trial.total_ms == pytest.approx(trial.generation_ms)


def test_compute_stats_uses_generation_latency_for_new_runs():
    trial = run_trial(
        MockProvider(ModelConfig(name="mock-a", provider="mock")),
        make_planned(ModelConfig(name="mock-a", provider="mock"), make_entry()),
        queued_at=0.0,
    )

    stats = compute_stats([trial])

    assert stats[0].median_latency_ms == pytest.approx(trial.generation_ms)
    assert stats[0].median_evaluation_ms is not None
    assert stats[0].median_total_ms is not None


def test_compute_stats_falls_back_to_legacy_latency():
    legacy = run_trial(
        MockProvider(ModelConfig(name="mock-a", provider="mock")),
        make_planned(ModelConfig(name="mock-a", provider="mock"), make_entry()),
        queued_at=0.0,
    )
    legacy = legacy.model_copy(
        update={"generation_ms": None, "evaluation_ms": None, "total_ms": None}
    )

    stats = compute_stats([legacy])

    assert stats[0].median_latency_ms == pytest.approx(legacy.latency_ms)
    assert stats[0].median_evaluation_ms is None
    assert stats[0].median_total_ms is None
