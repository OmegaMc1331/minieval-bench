from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from minieval_bench.config import ConfigError, DatasetError, load_config, load_dataset


def write_yaml(tmp_path: Path, config: dict) -> Path:
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_load_config_applies_defaults(write_benchmark, mock_config, valid_entries):
    config_path = write_benchmark(mock_config, valid_entries)

    config = load_config(config_path)

    assert config.benchmark == "unit"
    assert config.defaults.repeats == 2
    model = config.models[0]
    assert model.timeout == 30
    assert model.model == "mock-a"
    assert config.dataset == str((config_path.parent / "dataset.jsonl").resolve())


def test_load_config_missing_file():
    with pytest.raises(ConfigError, match="not found"):
        load_config(Path("does-not-exist.yaml"))


def test_load_config_invalid_yaml(tmp_path):
    config_path = tmp_path / "broken.yaml"
    config_path.write_text("benchmark: [unclosed", encoding="utf-8")

    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(config_path)


def test_load_config_requires_at_least_one_model(write_benchmark, mock_config, valid_entries):
    mock_config["models"] = []
    config_path = write_benchmark(mock_config, valid_entries)

    with pytest.raises(ConfigError, match="models"):
        load_config(config_path)


def test_load_config_rejects_duplicate_model_names(write_benchmark, mock_config, valid_entries):
    mock_config["models"] = [
        {"name": "twin", "provider": "mock"},
        {"name": "twin", "provider": "mock"},
    ]
    config_path = write_benchmark(mock_config, valid_entries)

    with pytest.raises(ConfigError, match="duplicate model names: twin"):
        load_config(config_path)


def test_load_config_rejects_unknown_keys(write_benchmark, mock_config, valid_entries):
    mock_config["unknown_option"] = True
    config_path = write_benchmark(mock_config, valid_entries)

    with pytest.raises(ConfigError, match="unknown_option"):
        load_config(config_path)


def test_load_config_requires_model_string_for_litellm(write_benchmark, mock_config, valid_entries):
    mock_config["models"] = [{"name": "gpt", "provider": "litellm"}]
    config_path = write_benchmark(mock_config, valid_entries)

    with pytest.raises(ConfigError, match="requires a model string"):
        load_config(config_path)


def test_load_config_rejects_invalid_repeats(write_benchmark, mock_config, valid_entries):
    mock_config["defaults"]["repeats"] = 0
    config_path = write_benchmark(mock_config, valid_entries)

    with pytest.raises(ConfigError, match="repeats"):
        load_config(config_path)


def test_load_config_resolves_relative_dataset_path(tmp_path, mock_config):
    dataset_path = tmp_path / "data.jsonl"
    dataset_path.write_text(
        '{"id": "e1", "prompt": "p", "metric": "contains", "reference": "x"}\n', encoding="utf-8"
    )
    mock_config["dataset"] = "data.jsonl"
    config_path = write_yaml(tmp_path, mock_config)

    config = load_config(config_path)

    assert config.dataset == str(dataset_path.resolve())
    assert len(load_dataset(Path(config.dataset))) == 1


def test_load_dataset_missing_file(tmp_path, mock_config):
    mock_config["dataset"] = "absent.jsonl"
    config_path = write_yaml(tmp_path, mock_config)

    config = load_config(config_path)

    with pytest.raises(DatasetError, match="not found"):
        load_dataset(Path(config.dataset))
