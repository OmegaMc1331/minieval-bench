from __future__ import annotations

from minieval_bench import __version__
from minieval_bench.history import find_runs
from test_comparison import make_manifest, write_run


def test_find_runs_lists_runs_sorted_by_date_desc(tmp_path):
    write_run(tmp_path / "late", make_manifest(created_at_utc="2026-11-14T02:00:00+00:00"),
              [])
    write_run(tmp_path / "early", make_manifest(created_at_utc="2026-11-14T01:00:00+00:00"),
              [])

    records = find_runs(tmp_path)

    assert [record.path.name for record in records] == ["late", "early"]
    record = records[0]
    assert record.benchmark == "unit"
    assert record.model_count == 1
    assert record.planned_requests == 3 * 2 * 1
    assert record.minieval_version == __version__


def test_find_runs_walks_subdirectories(tmp_path):
    write_run(tmp_path / "group" / "run-a", make_manifest(), [])
    write_run(tmp_path / "group" / "run-b", make_manifest(benchmark="other"), [])

    records = find_runs(tmp_path)

    assert {record.benchmark for record in records} == {"unit", "other"}


def test_find_runs_reads_results_json_without_manifest(tmp_path):
    import json

    run_dir = tmp_path / "results-only"
    run_dir.mkdir()
    payload = {
        "manifest": make_manifest().model_dump(mode="json"),
        "stats": [],
        "trials": [],
    }
    (run_dir / "results.json").write_text(json.dumps(payload), encoding="utf-8")

    records = find_runs(tmp_path)

    assert len(records) == 1
    assert records[0].path == run_dir


def test_find_runs_ignores_incomplete_and_broken_directories(tmp_path):
    (tmp_path / "empty").mkdir()
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "manifest.json").write_text("{not json", encoding="utf-8")
    wrong = tmp_path / "wrong"
    wrong.mkdir()
    (wrong / "manifest.json").write_text('{"unexpected": true}', encoding="utf-8")

    assert find_runs(tmp_path) == []
