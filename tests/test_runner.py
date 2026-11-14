from __future__ import annotations

from pathlib import Path

from minieval_bench.config import load_config, load_dataset
from minieval_bench.models import DatasetEntry, ModelConfig
from minieval_bench.providers import MockProvider, ProviderResponse
from minieval_bench.runner import run_benchmark, run_trial


class FailingProvider:
    def complete(self, prompt: str, system: str | None) -> ProviderResponse:
        raise RuntimeError("provider exploded")


def load_written(config_path: Path):
    config = load_config(config_path)
    return config, load_dataset(Path(config.dataset))


def make_entry(**overrides) -> DatasetEntry:
    fields = {"id": "e1", "prompt": "p", "metric": "regex", "reference": "^mock-[0-9a-f]{8}$"}
    return DatasetEntry(**fields, **overrides)


def test_run_benchmark_runs_every_combination(write_benchmark, mock_config, valid_entries):
    mock_config["models"] = [
        {"name": "mock-a", "provider": "mock"},
        {"name": "mock-b", "provider": "mock"},
    ]
    config, entries = load_written(write_benchmark(mock_config, valid_entries))

    trials, _ = run_benchmark(config, entries, show_progress=False)

    assert len(trials) == 2 * 3 * 2
    assert all(trial.ok for trial in trials)
    assert {trial.model for trial in trials} == {"mock-a", "mock-b"}


def test_run_benchmark_is_deterministic_with_same_seed(write_benchmark, mock_config, valid_entries):
    config_path = write_benchmark(mock_config, valid_entries)

    def collect():
        config, entries = load_written(config_path)
        trials, _ = run_benchmark(config, entries, show_progress=False)
        return [(trial.model, trial.entry_id, trial.repeat, trial.passed) for trial in trials]

    assert collect() == collect()


def test_run_benchmark_changes_with_seed(write_benchmark, mock_config, valid_entries):
    def responses(seed: int) -> set[str | None]:
        mock_config["defaults"]["seed"] = seed
        config, entries = load_written(write_benchmark(mock_config, valid_entries))
        trials, _ = run_benchmark(config, entries, show_progress=False)
        return {trial.response for trial in trials}

    assert responses(1) != responses(2)


def test_run_benchmark_sorts_trials(write_benchmark, mock_config, valid_entries):
    config, entries = load_written(write_benchmark(mock_config, valid_entries))

    trials, _ = run_benchmark(config, entries, show_progress=False)

    keys = [(trial.model, trial.entry_id, trial.repeat) for trial in trials]
    assert keys == sorted(keys)


def test_run_benchmark_priced_model_reports_cost(write_benchmark, mock_config, valid_entries):
    mock_config["models"] = [
        {"name": "free", "provider": "mock"},
        {"name": "priced", "provider": "mock", "input_cost_per_1m": 1.0, "output_cost_per_1m": 2.0},
    ]
    config, entries = load_written(write_benchmark(mock_config, valid_entries))

    trials, _ = run_benchmark(config, entries, show_progress=False)

    by_model: dict[str, list] = {}
    for trial in trials:
        by_model.setdefault(trial.model, []).append(trial)
    assert all(trial.cost_usd is None for trial in by_model["free"])
    assert all(trial.cost_usd is not None and trial.cost_usd > 0 for trial in by_model["priced"])


def test_run_trial_records_provider_failure():
    model = ModelConfig(name="broken", provider="mock")
    entry = make_entry()

    trial = run_trial(FailingProvider(), model, entry, repeat=0)

    assert not trial.ok
    assert trial.error_type == "RuntimeError"
    assert trial.error_message == "provider exploded"
    assert trial.passed is None
    assert trial.response is None
    assert trial.latency_ms >= 0


def test_run_trial_evaluates_successful_response():
    model = ModelConfig(name="mock-a", provider="mock")
    entry = make_entry()

    trial = run_trial(MockProvider(model, seed=1), model, entry, repeat=0)

    assert trial.ok
    assert trial.passed is True
    assert trial.input_tokens is not None
    assert trial.error_type is None
