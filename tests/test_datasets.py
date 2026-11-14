from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import write_jsonl
from minieval_bench.config import DatasetError, load_dataset


def write_lines(tmp_path: Path, lines: list[str]) -> Path:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dataset_path


def test_load_dataset_reads_entries_and_optional_system(tmp_path):
    entries = [
        {"id": "a", "prompt": "pa", "metric": "exact_match", "reference": "x"},
        {"id": "b", "prompt": "pb", "metric": "contains", "reference": "y", "system": "be strict"},
    ]
    dataset_path = write_jsonl(tmp_path / "dataset.jsonl", entries)

    loaded = load_dataset(dataset_path)

    assert [entry.id for entry in loaded] == ["a", "b"]
    assert loaded[1].system == "be strict"
    assert loaded[0].system is None


def test_load_dataset_skips_blank_lines(tmp_path):
    dataset_path = write_lines(
        tmp_path,
        [
            json.dumps({"id": "a", "prompt": "p", "metric": "contains", "reference": "x"}),
            "",
            json.dumps({"id": "b", "prompt": "p", "metric": "contains", "reference": "x"}),
        ],
    )

    assert len(load_dataset(dataset_path)) == 2


def test_load_dataset_rejects_empty_file(tmp_path):
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text("", encoding="utf-8")

    with pytest.raises(DatasetError, match="empty"):
        load_dataset(dataset_path)


def test_load_dataset_rejects_invalid_json_with_line_number(tmp_path):
    dataset_path = write_lines(
        tmp_path,
        [
            json.dumps({"id": "a", "prompt": "p", "metric": "contains", "reference": "x"}),
            "{not json",
        ],
    )

    with pytest.raises(DatasetError, match=r"dataset\.jsonl:2"):
        load_dataset(dataset_path)


def test_load_dataset_requires_reference(tmp_path):
    entries = [{"id": "a", "prompt": "p", "metric": "contains"}]
    dataset_path = write_jsonl(tmp_path / "dataset.jsonl", entries)

    with pytest.raises(DatasetError, match="reference"):
        load_dataset(dataset_path)


def test_load_dataset_requires_known_metric(tmp_path):
    entries = [{"id": "a", "prompt": "p", "metric": "bleu", "reference": "x"}]
    dataset_path = write_jsonl(tmp_path / "dataset.jsonl", entries)

    with pytest.raises(DatasetError, match="metric"):
        load_dataset(dataset_path)


def test_load_dataset_rejects_duplicate_ids(tmp_path):
    entries = [
        {"id": "a", "prompt": "p1", "metric": "contains", "reference": "x"},
        {"id": "a", "prompt": "p2", "metric": "contains", "reference": "x"},
    ]
    dataset_path = write_jsonl(tmp_path / "dataset.jsonl", entries)

    with pytest.raises(DatasetError, match="duplicate entry id 'a'"):
        load_dataset(dataset_path)


def test_load_dataset_rejects_invalid_regex_reference(tmp_path):
    entries = [{"id": "a", "prompt": "p", "metric": "regex", "reference": "([unclosed"}]
    dataset_path = write_jsonl(tmp_path / "dataset.jsonl", entries)

    with pytest.raises(DatasetError, match="invalid regex"):
        load_dataset(dataset_path)


def test_load_dataset_rejects_unknown_fields(tmp_path):
    entries = [{"id": "a", "prompt": "p", "metric": "contains", "reference": "x", "weight": 2}]
    dataset_path = write_jsonl(tmp_path / "dataset.jsonl", entries)

    with pytest.raises(DatasetError, match="weight"):
        load_dataset(dataset_path)
