from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from minieval_bench.cli import app
from minieval_bench.datasets import (
    DatasetImportError,
    hf_file_url,
    import_huggingface,
    import_local_file,
)

runner = CliRunner()
WIDE = {"COLUMNS": "220"}


class FakeResponse:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self, size: int = -1) -> bytes:
        return self._data if size < 0 else self._data[:size]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None


CSV_CONTENT = (
    "id,prompt,metric,reference,category\n"
    'q1,"What is 2+2?",exact_match,4,arithmetic\n'
    'q2,"Name a primary color.",contains,red,general\n'
)


def write_csv(tmp_path: Path, content: str = CSV_CONTENT, name: str = "source.csv") -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_csv_import_produces_valid_dataset(tmp_path: Path) -> None:
    source = write_csv(tmp_path)
    output = tmp_path / "imported.jsonl"

    report = import_local_file(source, output)

    assert report.entry_count == 2
    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["id"] == "q1"
    assert first["metric"] == "exact_match"
    assert first["reference"] == "4"
    assert first["category"] == "arithmetic"
    assert report.output_sha256
    sidecar = output.with_name(output.name + ".provenance.json")
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    assert provenance["source"]["kind"] == "csv"
    assert provenance["source"]["sha256"] == report.source_sha256
    assert provenance["output"]["entries"] == 2


def test_import_is_deterministic(tmp_path: Path) -> None:
    source = write_csv(tmp_path)
    first_output = tmp_path / "first.jsonl"
    second_output = tmp_path / "second.jsonl"

    import_local_file(source, first_output)
    import_local_file(source, second_output)

    assert first_output.read_bytes() == second_output.read_bytes()


def test_csv_import_maps_empty_cells_to_null(tmp_path: Path) -> None:
    source = write_csv(
        tmp_path,
        "id,prompt,metric,reference,system,category\n"
        "j1,Say something.,judge,,,\n",
    )
    output = tmp_path / "judge.jsonl"

    import_local_file(source, output)

    entry = json.loads(output.read_text(encoding="utf-8"))
    assert entry["reference"] is None
    assert entry["system"] is None
    assert entry["category"] is None


def test_csv_import_rejects_missing_columns(tmp_path: Path) -> None:
    source = write_csv(tmp_path, "id,prompt\n1,hello\n")
    with pytest.raises(DatasetImportError, match="missing column"):
        import_local_file(source, tmp_path / "out.jsonl")


def test_csv_import_rejects_structured_metrics(tmp_path: Path) -> None:
    source = write_csv(tmp_path, "id,prompt,metric\ns1,Make JSON,json_schema\n")
    with pytest.raises(DatasetImportError, match="author JSONL entries"):
        import_local_file(source, tmp_path / "out.jsonl")


def test_csv_import_rejects_duplicate_ids(tmp_path: Path) -> None:
    source = write_csv(
        tmp_path,
        "id,prompt,metric,reference\ndup,one,exact_match,1\ndup,two,exact_match,2\n",
    )
    with pytest.raises(DatasetImportError, match="duplicate entry id 'dup'"):
        import_local_file(source, tmp_path / "out.jsonl")


def test_csv_import_rejects_missing_reference(tmp_path: Path) -> None:
    source = write_csv(tmp_path, "id,prompt,metric,reference\nq1,one,exact_match,\n")
    with pytest.raises(DatasetImportError, match="requires a reference"):
        import_local_file(source, tmp_path / "out.jsonl")


def test_import_rejects_invalid_utf8(tmp_path: Path) -> None:
    source = tmp_path / "latin.csv"
    source.write_bytes(b"id,prompt,metric\nq1,caf\xe9,contains\n")
    with pytest.raises(DatasetImportError, match="not valid UTF-8"):
        import_local_file(source, tmp_path / "out.jsonl")


def test_json_array_import(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            [
                {"id": "a", "prompt": "one", "metric": "exact_match", "reference": "1"},
                {"id": "b", "prompt": "two", "metric": "contains", "reference": "o"},
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "out.jsonl"

    report = import_local_file(source, output)

    assert report.entry_count == 2
    lines = output.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["id"] == "a"


def test_json_import_rejects_non_array(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"id": "a"}), encoding="utf-8")
    with pytest.raises(DatasetImportError, match="must be an array"):
        import_local_file(source, tmp_path / "out.jsonl")


def test_import_refuses_archives(tmp_path: Path) -> None:
    source = tmp_path / "archive.zip"
    source.write_bytes(b"PK")
    with pytest.raises(DatasetImportError, match="archives are not supported"):
        import_local_file(source, tmp_path / "out.jsonl")


def test_import_refuses_unknown_suffix(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    source.write_bytes(b"...")
    with pytest.raises(DatasetImportError, match="unsupported source format"):
        import_local_file(source, tmp_path / "out.jsonl")


def test_hf_revision_must_be_commit_sha(tmp_path: Path) -> None:
    with pytest.raises(DatasetImportError, match="40-character commit sha"):
        import_huggingface(
            dataset_id="org/data",
            revision="main",
            filename="data.jsonl",
            output=tmp_path / "out.jsonl",
        )


def test_hf_rejects_traversal_and_bad_ids(tmp_path: Path) -> None:
    with pytest.raises(DatasetImportError, match="invalid file reference"):
        import_huggingface(
            dataset_id="org/data",
            revision="a" * 40,
            filename="../secret.jsonl",
            output=tmp_path / "out.jsonl",
        )
    with pytest.raises(DatasetImportError, match="invalid dataset id"):
        import_huggingface(
            dataset_id="no-slash",
            revision="a" * 40,
            filename="data.jsonl",
            output=tmp_path / "out.jsonl",
        )


def test_hf_rejects_unsupported_file_types(tmp_path: Path) -> None:
    with pytest.raises(DatasetImportError, match="parquet"):
        import_huggingface(
            dataset_id="org/data",
            revision="a" * 40,
            filename="data.parquet",
            output=tmp_path / "out.jsonl",
        )
    with pytest.raises(DatasetImportError, match="archives"):
        import_huggingface(
            dataset_id="org/data",
            revision="a" * 40,
            filename="data.zip",
            output=tmp_path / "out.jsonl",
        )


def test_hf_url_format() -> None:
    url = hf_file_url("org/data", "b" * 40, "split/train.jsonl")
    assert url == f"https://huggingface.co/datasets/org/data/resolve/{'b' * 40}/split/train.jsonl"


def test_hf_import_with_fake_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from minieval_bench import datasets as datasets_module

    revision = "c" * 40
    payload = '{"id": "h1", "prompt": "one", "metric": "exact_match", "reference": "1"}\n'
    card = "---\nlicense: apache-2.0\n---\n# dataset\n"
    calls: list[str] = []

    def fake_urlopen(url: str, timeout: int = 60):
        calls.append(url)
        if url.endswith("README.md"):
            return FakeResponse(card.encode())
        return FakeResponse(payload.encode())

    monkeypatch.setattr(datasets_module, "urlopen", fake_urlopen)
    monkeypatch.setenv("MINIEVAL_CACHE_DIR", str(tmp_path / "cache"))
    output = tmp_path / "hf.jsonl"

    report = import_huggingface(
        dataset_id="org/data", revision=revision, filename="data.jsonl", output=output
    )

    assert report.entry_count == 1
    assert report.license_value == "apache-2.0"
    assert report.license_basis == "dataset card at the pinned revision"
    assert json.loads(output.read_text(encoding="utf-8"))["id"] == "h1"
    provenance = json.loads(
        output.with_name(output.name + ".provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["source"]["kind"] == "huggingface"
    assert revision in provenance["source"]["reference"]

    calls.clear()
    second_output = tmp_path / "hf-again.jsonl"
    import_huggingface(
        dataset_id="org/data", revision=revision, filename="data.jsonl", output=second_output
    )
    assert calls == []
    assert second_output.read_bytes() == output.read_bytes()


def test_hf_import_prefers_declared_license(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from minieval_bench import datasets as datasets_module

    payload = '{"id": "h1", "prompt": "one", "metric": "exact_match", "reference": "1"}\n'

    def fake_urlopen(url: str, timeout: int = 60):
        if url.endswith("README.md"):
            return FakeResponse(b"")
        return FakeResponse(payload.encode())

    monkeypatch.setattr(datasets_module, "urlopen", fake_urlopen)
    monkeypatch.setenv("MINIEVAL_CACHE_DIR", str(tmp_path / "cache"))
    report = import_huggingface(
        dataset_id="org/data",
        revision="d" * 40,
        filename="data.jsonl",
        output=tmp_path / "out.jsonl",
        license_declared="cc-by-4.0",
    )
    assert report.license_value == "cc-by-4.0"
    assert report.license_basis == "user-declared"


def test_hf_import_enforces_size_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from minieval_bench import datasets as datasets_module

    monkeypatch.setattr(
        datasets_module,
        "urlopen",
        lambda url, timeout=60: FakeResponse(b"x" * 100),
    )
    monkeypatch.setenv("MINIEVAL_CACHE_DIR", str(tmp_path / "cache"))
    with pytest.raises(DatasetImportError, match="size limit"):
        import_huggingface(
            dataset_id="org/data",
            revision="e" * 40,
            filename="data.jsonl",
            output=tmp_path / "out.jsonl",
            max_bytes=10,
        )


def test_dataset_validate_cli(tmp_path: Path) -> None:
    dataset = tmp_path / "ok.jsonl"
    dataset.write_text(
        '{"id": "a", "prompt": "one", "metric": "exact_match", "reference": "1"}\n',
        encoding="utf-8",
    )
    result = runner.invoke(app, ["dataset", "validate", str(dataset)], env=WIDE)
    assert result.exit_code == 0
    assert "Dataset is valid" in result.output

    broken = tmp_path / "broken.jsonl"
    broken.write_text(
        '{"id": "a", "prompt": "one", "metric": "exact_match", "reference": "1"}\n'
        '{"id": "a", "prompt": "two", "metric": "exact_match", "reference": "2"}\n',
        encoding="utf-8",
    )
    result = runner.invoke(app, ["dataset", "validate", str(broken)], env=WIDE)
    assert result.exit_code == 1
    assert "duplicate entry id" in result.output


def test_dataset_import_cli(tmp_path: Path) -> None:
    source = write_csv(tmp_path)
    output = tmp_path / "imported.jsonl"
    result = runner.invoke(
        app,
        [
            "dataset", "import",
            "--source", str(source),
            "--output", str(output),
        ],
        env=WIDE,
    )
    assert result.exit_code == 0
    assert "Imported 2 entries" in result.output
    assert output.is_file()
    assert (tmp_path / "imported.jsonl.provenance.json").is_file()
