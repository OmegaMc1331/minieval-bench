from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from minieval_bench.models import Manifest


@dataclass
class RunRecord:
    benchmark: str
    created_at_utc: str
    model_count: int
    planned_requests: int
    minieval_version: str
    path: Path


def find_runs(root: Path) -> list[RunRecord]:
    run_dirs = {path.parent for path in root.rglob("manifest.json")}
    run_dirs |= {path.parent for path in root.rglob("results.json")}

    records = []
    for run_dir in run_dirs:
        manifest = load_manifest(run_dir)
        if manifest is None:
            continue
        planned_requests = (
            manifest.dataset_entries * manifest.repeats * len(manifest.models)
        )
        records.append(
            RunRecord(
                benchmark=manifest.benchmark,
                created_at_utc=manifest.created_at_utc,
                model_count=len(manifest.models),
                planned_requests=planned_requests,
                minieval_version=manifest.minieval_version,
                path=run_dir,
            )
        )
    records.sort(key=lambda record: record.created_at_utc, reverse=True)
    return records


def load_manifest(run_dir: Path) -> Manifest | None:
    manifest_path = run_dir / "manifest.json"
    try:
        if manifest_path.is_file():
            return Manifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        results_path = run_dir / "results.json"
        if results_path.is_file():
            payload = json.loads(results_path.read_text(encoding="utf-8"))
            return Manifest.model_validate(payload["manifest"])
    except (OSError, ValueError, KeyError):
        return None
    return None
