from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from minieval_bench import __version__
from minieval_bench.config import DatasetError, parse_dataset_content
from minieval_bench.models import DatasetEntry

CSV_SUPPORTED_METRICS = ("exact_match", "contains", "regex", "judge")
JSON_SUFFIXES = (".jsonl", ".ndjson", ".json")
ARCHIVE_SUFFIXES = (".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar")
HF_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
HF_DATASET_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
HF_BASE_URL = "https://huggingface.co/datasets"
HF_SUPPORTED_SUFFIXES = (".jsonl", ".csv")
DEFAULT_MAX_DOWNLOAD_BYTES = 64 * 1024 * 1024
PROVENANCE_SUFFIX = ".provenance.json"


class DatasetImportError(Exception):
    pass


@dataclass
class ImportReport:
    source_kind: str
    source_reference: str
    source_sha256: str
    output_path: Path
    output_sha256: str
    entry_count: int
    license_value: str | None
    license_basis: str | None


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cache_dir() -> Path:
    override = os.environ.get("MINIEVAL_CACHE_DIR")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "minieval-bench"


def dataset_summary(entries: list[DatasetEntry]) -> dict:
    metric_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    for entry in entries:
        metric_counts[entry.metric] = metric_counts.get(entry.metric, 0) + 1
        if entry.category is not None:
            category_counts[entry.category] = category_counts.get(entry.category, 0) + 1
    return {
        "entries": len(entries),
        "metrics": metric_counts,
        "categories": category_counts or None,
    }


def import_local_file(source: Path, output: Path, delimiter: str = ",") -> ImportReport:
    if not source.is_file():
        raise DatasetImportError(f"source file not found: {source}")
    if source.suffix.lower() in ARCHIVE_SUFFIXES:
        raise DatasetImportError(
            f"archives are not supported ({source.suffix}): extract it manually and "
            "import the JSONL or CSV file"
        )
    if source.suffix.lower() == ".csv":
        return import_csv_file(source, output, delimiter)
    if source.suffix.lower() in JSON_SUFFIXES:
        return import_json_file(source, output)
    raise DatasetImportError(
        f"unsupported source format '{source.suffix}': expected .jsonl, .json or .csv"
    )


def import_csv_file(source: Path, output: Path, delimiter: str = ",") -> ImportReport:
    raw_entries = read_csv_entries(source, delimiter)
    report = write_imported_entries(
        source_kind="csv",
        source_reference=str(source.resolve()),
        source_sha256=sha256_file(source),
        raw_entries=raw_entries,
        output=output,
        transformations=[
            f"parsed CSV with delimiter '{delimiter}'",
            "empty cells mapped to null",
            "fields written with sorted keys, rows kept in file order",
        ],
    )
    return report


def read_csv_entries(source: Path, delimiter: str) -> list[dict]:
    try:
        text = source.read_bytes().decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DatasetImportError(f"{source}: source is not valid UTF-8") from exc
    reader = csv.DictReader(text.splitlines(), delimiter=delimiter)
    required = {"id", "prompt", "metric"}
    missing_columns = required - set(reader.fieldnames or [])
    if missing_columns:
        raise DatasetImportError(
            f"{source}: missing column(s): {', '.join(sorted(missing_columns))}"
        )
    raw_entries = []
    for row_number, row in enumerate(reader, 2):
        cleaned = {}
        for key, value in row.items():
            if key is None:
                raise DatasetImportError(
                    f"{source}: row {row_number} has more cells than the header"
                )
            cleaned[key] = value if value != "" else None
        metric = cleaned.get("metric")
        if metric not in CSV_SUPPORTED_METRICS:
            raise DatasetImportError(
                f"{source}: row {row_number}: metric '{metric}' cannot be imported from"
                " CSV; author JSONL entries for json_schema and code_tests"
            )
        raw_entries.append(cleaned)
    return raw_entries


def import_json_file(source: Path, output: Path) -> ImportReport:
    try:
        text = source.read_bytes().decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DatasetImportError(f"{source}: source is not valid UTF-8") from exc
    if source.suffix.lower() in (".jsonl", ".ndjson"):
        raw_entries = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise DatasetImportError(f"{source}: JSON source must be an array of entries")
        raw_entries = payload
    for position, raw_entry in enumerate(raw_entries, 1):
        if not isinstance(raw_entry, dict):
            raise DatasetImportError(
                f"{source}: entry {position} is not a JSON object"
            )
    return write_imported_entries(
        source_kind="json",
        source_reference=str(source.resolve()),
        source_sha256=sha256_file(source),
        raw_entries=raw_entries,
        output=output,
        transformations=[
            "JSON entries re-serialized with sorted keys, order kept",
        ],
    )


def write_imported_entries(
    source_kind: str,
    source_reference: str,
    source_sha256: str,
    raw_entries: list[dict],
    output: Path,
    transformations: list[str],
    license_value: str | None = None,
    license_basis: str | None = None,
) -> ImportReport:
    if not raw_entries:
        raise DatasetImportError(f"{source_reference}: no entries found to import")
    lines = []
    for position, raw_entry in enumerate(raw_entries, 1):
        try:
            entry = DatasetEntry.model_validate(raw_entry)
        except Exception as exc:
            raise DatasetImportError(
                f"{source_reference}: entry {position} is invalid: {exc}"
            ) from exc
        lines.append(json.dumps(entry.model_dump(mode="json"), sort_keys=True))
    content = "\n".join(lines) + "\n"
    try:
        parse_dataset_content(content, str(output))
    except DatasetError as exc:
        raise DatasetImportError(f"{source_reference}: {exc}") from exc
    if output.parent:
        output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8", newline="\n")
    report = ImportReport(
        source_kind=source_kind,
        source_reference=source_reference,
        source_sha256=source_sha256,
        output_path=output,
        output_sha256=hashlib.sha256(content.encode()).hexdigest(),
        entry_count=len(raw_entries),
        license_value=license_value,
        license_basis=license_basis,
    )
    write_provenance(report, transformations)
    return report


def write_provenance(report: ImportReport, transformations: list[str]) -> None:
    provenance = {
        "minieval_version": __version__,
        "imported_at_utc": datetime.now(UTC).isoformat(),
        "source": {
            "kind": report.source_kind,
            "reference": report.source_reference,
            "sha256": report.source_sha256,
        },
        "output": {
            "path": str(report.output_path),
            "sha256": report.output_sha256,
            "entries": report.entry_count,
        },
        "transformations": transformations,
        "license": {
            "value": report.license_value,
            "basis": report.license_basis,
        },
    }
    sidecar = report.output_path.with_name(
        report.output_path.name + PROVENANCE_SUFFIX
    )
    sidecar.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


def import_huggingface(
    dataset_id: str,
    revision: str,
    filename: str,
    output: Path,
    license_declared: str | None = None,
    max_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES,
    refresh_cache: bool = False,
) -> ImportReport:
    if not HF_DATASET_ID_PATTERN.match(dataset_id):
        raise DatasetImportError(
            f"invalid dataset id '{dataset_id}': expected '<namespace>/<name>'"
        )
    if not HF_REVISION_PATTERN.match(revision):
        raise DatasetImportError(
            "revision must be a full 40-character commit sha: branch names such as"
            " 'main' move and are not reproducible references"
        )
    if filename.startswith(("/", "\\")) or ".." in Path(filename).parts:
        raise DatasetImportError(f"invalid file reference '{filename}'")
    suffix = Path(filename).suffix.lower()
    if suffix in ARCHIVE_SUFFIXES:
        raise DatasetImportError(
            f"archives are not supported ({suffix}); pick a JSONL or CSV file"
        )
    if suffix not in HF_SUPPORTED_SUFFIXES:
        raise DatasetImportError(
            f"unsupported file type '{suffix}': only .jsonl and .csv can be imported"
            " without extra dependencies (parquet would require pyarrow)"
        )

    raw_path = hf_cache_path(dataset_id, revision, filename)
    if refresh_cache or not raw_path.is_file():
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        download_file(hf_file_url(dataset_id, revision, filename), raw_path, max_bytes)

    license_cache = raw_path.with_name(raw_path.name + ".license")
    if license_cache.is_file():
        card_license = license_cache.read_text(encoding="utf-8") or None
    else:
        card_license = read_hf_card_license(dataset_id, revision)
        license_cache.write_text(card_license or "", encoding="utf-8")
    license_value = license_declared or card_license
    license_basis = None
    if license_declared:
        license_basis = "user-declared"
    elif card_license:
        license_basis = "dataset card at the pinned revision"

    source_reference = hf_file_url(dataset_id, revision, filename)
    source_sha256 = sha256_file(raw_path)
    if suffix == ".csv":
        raw_entries = read_csv_entries(raw_path, ",")
        transformations = [
            "downloaded from Hugging Face at the pinned commit",
            "parsed CSV with delimiter ','",
            "empty cells mapped to null",
            "fields written with sorted keys, rows kept in file order",
        ]
    else:
        try:
            text = raw_path.read_bytes().decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise DatasetImportError(f"{filename}: downloaded file is not valid UTF-8") from exc
        raw_entries = [json.loads(line) for line in text.splitlines() if line.strip()]
        transformations = [
            "downloaded from Hugging Face at the pinned commit",
            "JSONL entries re-serialized with sorted keys, order kept",
        ]
    return write_imported_entries(
        source_kind="huggingface",
        source_reference=source_reference,
        source_sha256=source_sha256,
        raw_entries=raw_entries,
        output=output,
        transformations=transformations,
        license_value=license_value,
        license_basis=license_basis,
    )


def hf_file_url(dataset_id: str, revision: str, filename: str) -> str:
    return f"{HF_BASE_URL}/{dataset_id}/resolve/{revision}/{filename}"


def hf_cache_path(dataset_id: str, revision: str, filename: str) -> Path:
    safe_name = filename.replace("/", "__")
    return cache_dir() / "hf" / dataset_id / revision / safe_name


def download_file(url: str, destination: Path, max_bytes: int) -> None:
    try:
        with urlopen(url, timeout=60) as response:
            data = response.read(max_bytes + 1)
    except HTTPError as exc:
        raise DatasetImportError(f"download failed for {url}: status {exc.code}") from exc
    except URLError as exc:
        raise DatasetImportError(f"download failed for {url}: {exc.reason}") from exc
    if len(data) > max_bytes:
        raise DatasetImportError(
            f"download exceeds the size limit ({max_bytes} bytes): pass --max-bytes"
            " explicitly after checking the source"
        )
    if not data:
        raise DatasetImportError(f"downloaded file is empty: {url}")
    temporary = destination.with_name(destination.name + ".part")
    temporary.write_bytes(data)
    temporary.replace(destination)


def read_hf_card_license(dataset_id: str, revision: str) -> str | None:
    card_url = f"{HF_BASE_URL}/{dataset_id}/resolve/{revision}/README.md"
    try:
        with urlopen(card_url, timeout=30) as response:
            text = response.read(1_000_000).decode("utf-8", errors="replace")
    except (HTTPError, URLError, OSError):
        return None
    for line in text.splitlines():
        match = re.match(r"^license:\s*(.+?)\s*$", line)
        if match:
            return match.group(1)
    return None


def validate_dataset_file(path: Path) -> tuple[list[DatasetEntry], str]:
    from minieval_bench.config import load_dataset

    try:
        entries = load_dataset(path)
    except DatasetError as exc:
        raise DatasetImportError(str(exc)) from exc
    return entries, sha256_file(path)
