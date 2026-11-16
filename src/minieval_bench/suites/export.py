from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from minieval_bench import __version__
from minieval_bench.providers import mask_secrets
from minieval_bench.suites.reports import SuiteRunData, load_suite_run

EXPORT_METADATA_NAME = "export-metadata.json"
CHECKSUMS_NAME = "checksums.json"


class SuiteExportError(Exception):
    pass


def export_suite_archive(
    input_dir: Path, output_path: Path, include_responses: bool = False
) -> dict:
    run = load_suite_run(input_dir)
    members: dict[str, bytes] = {}
    for name in (
        "suite-manifest.json",
        "benchmark.lock.json",
        "suite-results.json",
        "suite-report.md",
        "suite-report.html",
    ):
        candidate = input_dir / name
        if candidate.is_file():
            members[name] = candidate.read_bytes()

    for task in run.tasks:
        task_results = input_dir / "tasks" / task.task_id / "results.json"
        if not task_results.is_file():
            continue
        summary = task_summary_payload(task_results)
        members[f"tasks/{task.task_id}/stats.json"] = json_bytes(summary)
        if include_responses:
            members[f"tasks/{task.task_id}/results.json"] = task_results.read_bytes()

    if include_responses:
        scan_for_secrets(members, run)

    checksums = {
        name: hashlib.sha256(content).hexdigest() for name, content in sorted(members.items())
    }
    members[CHECKSUMS_NAME] = json_bytes(checksums)
    members[EXPORT_METADATA_NAME] = json_bytes(
        {
            "minieval_version": __version__,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "include_responses": include_responses,
            "suite": run.manifest.get("suite", {}),
            "note": (
                "checksums.json verifies member integrity against this archive; it does"
                " not authenticate the producer."
            ),
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return {
        "path": output_path,
        "members": len(members),
        "include_responses": include_responses,
        "entry_count": sum(task.entry_count for task in run.tasks),
    }


def task_summary_payload(task_results: Path) -> dict:
    payload = json.loads(task_results.read_text(encoding="utf-8"))
    return {
        "manifest": payload.get("manifest"),
        "stats": payload.get("stats", []),
    }


def json_bytes(payload: dict) -> bytes:
    return (json.dumps(payload, indent=2) + "\n").encode()


RENDERED_REPORT_NAME = "suite-report.html"


def scan_for_secrets(members: dict[str, bytes], run: SuiteRunData) -> None:
    for name, content in members.items():
        if name == RENDERED_REPORT_NAME:
            continue
        text = content.decode("utf-8", errors="replace")
        if mask_secrets(text) != text:
            raise SuiteExportError(
                f"refusing to export '{name}': it contains text that looks like a"
                " credential; remove the secret or export without"
                " --include-responses"
            )
    for task in run.tasks:
        for trial in task.trials:
            for field in (trial.response, trial.error_message):
                if field and mask_secrets(field) != field:
                    raise SuiteExportError(
                        "refusing to export responses: task"
                        f" '{task.task_id}' contains text that looks like a"
                        " credential; export without --include-responses"
                    )
