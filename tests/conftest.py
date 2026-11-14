from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = REPO_ROOT / "benchmarks" / "example.yaml"


def write_jsonl(path: Path, entries: list[dict]) -> Path:
    content = "\n".join(json.dumps(entry) for entry in entries) + "\n"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def valid_entries() -> list[dict]:
    return [
        {"id": "e1", "prompt": "first prompt", "metric": "regex", "reference": "^mock-[0-9a-f]+$"},
        {"id": "e2", "prompt": "second prompt", "metric": "contains", "reference": "mock"},
        {"id": "e3", "prompt": "third prompt", "metric": "exact_match", "reference": "unreachable"},
    ]


@pytest.fixture
def mock_config() -> dict:
    return {
        "benchmark": "unit",
        "models": [{"name": "mock-a", "provider": "mock"}],
        "defaults": {"repeats": 2, "concurrency": 2, "seed": 1, "timeout": 30},
    }


@pytest.fixture
def write_benchmark(tmp_path):
    def _write(config: dict, entries: list[dict]) -> Path:
        dataset_path = write_jsonl(tmp_path / "dataset.jsonl", entries)
        config_path = tmp_path / "benchmark.yaml"
        merged = dict(config, dataset=str(dataset_path))
        config_path.write_text(yaml.safe_dump(merged), encoding="utf-8")
        return config_path

    return _write
