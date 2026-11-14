from __future__ import annotations

import json
import time

from minieval_bench.html_report import (
    build_run_payload,
    read_template,
    render_standalone,
    write_html_report,
)
from minieval_bench.models import DatasetEntry, TrialResult
from minieval_bench.reports import load_results
from test_reports import OLD_RESULTS, make_manifest, make_stats
from test_stats import make_trial

HOSTILE_RESPONSE = '<script>alert("xss")</script><img src=x onerror="alert(1)">'
HOSTILE_ERROR = "</script><script>alert(2)</script>"
HOSTILE_PROMPT = "\"'>&<iframe src=javascript:alert(3)></iframe>é中🙂"


def hostile_trials() -> list[TrialResult]:
    return [
        make_trial(model="model-a", passed=True),
        TrialResult(
            model="model-a",
            entry_id="e2",
            repeat=1,
            ok=False,
            response=None,
            latency_ms=1.0,
            input_tokens=None,
            output_tokens=None,
            cost_usd=None,
            error_type="APIError",
            error_message=HOSTILE_ERROR,
            passed=None,
            queue_ms=0.0,
            plan_index=1,
        ),
    ]


def hostile_entries() -> list[DatasetEntry]:
    return [
        DatasetEntry(
            id="e1",
            prompt=HOSTILE_PROMPT,
            metric="contains",
            reference='"quoted & <tagged>',
        )
    ]


def test_packaged_templates_are_readable():
    html = read_template("report.html")
    assert "<!doctype html>" in html.lower()
    assert read_template("styles.css").count(":root") >= 1
    assert "buildPage" in read_template("dashboard.js")
    assert "buildPage" in read_template("compare.js")


def test_render_standalone_inlines_everything():
    html = render_standalone({"kind": "run", "manifest": {"benchmark": "unit"}},
                             "report.html", "dashboard.js")

    assert "--c1" in html
    assert "buildPage" in html
    assert "/*__STYLES__*/" not in html
    assert "__DATA__" not in html
    embedded = html.split('id="report-data" type="application/json">')[1].split("</script>")[0]
    assert json.loads(embedded.replace("<\\/", "</"))["kind"] == "run"


def test_render_standalone_has_no_external_requests():
    html = render_standalone({"kind": "run"}, "report.html", "dashboard.js")

    assert "http://" not in html
    assert "https://" not in html
    assert "<script src" not in html.lower()
    assert "<link" not in html.lower()


def test_report_resists_script_injection(tmp_path):
    trials = [
        TrialResult(
            model="model-a",
            entry_id="e1",
            repeat=0,
            ok=True,
            response=HOSTILE_RESPONSE,
            latency_ms=2.0,
            input_tokens=3,
            output_tokens=4,
            cost_usd=None,
            error_type=None,
            error_message=None,
            passed=True,
            queue_ms=0.0,
            plan_index=0,
        ),
        hostile_trials()[1],
    ]
    stats = [make_stats()]
    output = tmp_path / "report.html"

    write_html_report(output, make_manifest(), stats, trials, hostile_entries())

    html = output.read_text(encoding="utf-8")
    assert "</script><script>" not in html
    assert html.count("</script>") == 2
    embedded = html.split('id="report-data" type="application/json">')[1].split("</script>")[0]
    payload = json.loads(embedded.replace("<\\/", "</"))
    assert HOSTILE_RESPONSE in payload["responses"][0]
    assert payload["dataset"][0]["prompt"] == HOSTILE_PROMPT


def test_build_run_payload_uses_interned_indexes():
    trials = [
        make_trial(model="model-a"),
        make_trial(model="model-b", passed=False),
        make_trial(model="model-a"),
    ]

    payload = build_run_payload(make_manifest(), [make_stats(model="model-a")], trials, None)

    assert payload["models"] == ["model-a", "model-b"]
    assert payload["entryIds"] == ["e1"]
    assert payload["trials"][0][0] == 0
    assert payload["trials"][1][0] == 1
    assert len(payload["responses"]) == 3
    assert payload["dataset"] is None


def test_build_run_payload_keeps_null_and_failed_data():
    payload = build_run_payload(
        make_manifest(), [make_stats()], hostile_trials(), None
    )

    failed = payload["trials"][1]
    assert failed[3] is False
    assert failed[4] is None
    assert failed[7] is None
    assert payload["errorTypes"] == ["APIError"]
    assert failed[10] == 0
    assert payload["responses"][1] is None
    assert payload["errorMessages"][1] == HOSTILE_ERROR


def test_html_report_for_v010_results(tmp_path):
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    (legacy_dir / "results.json").write_text(json.dumps(OLD_RESULTS), encoding="utf-8")
    manifest, stats, trials = load_results(legacy_dir)

    output = tmp_path / "legacy.html"
    write_html_report(output, manifest, stats, trials, None)

    html = output.read_text(encoding="utf-8")
    assert "legacy" in html
    embedded = html.split('id="report-data" type="application/json">')[1]
    embedded = embedded.split("</script>")[0].replace("<\\/", "</")
    assert json.loads(embedded)["manifest"]["minieval_version"] == "0.1.0"


def test_html_generation_scales_to_large_trial_counts(tmp_path):
    trials = []
    for index in range(10_000):
        trial = make_trial(model="model-a", latency_ms=5.0 + (index % 50))
        trial.response = f"mock-{index:08d}"
        trial.entry_id = f"entry-{index % 1000:04d}"
        trial.queue_ms = float(index % 7)
        trials.append(trial)

    output = tmp_path / "large.html"
    started = time.perf_counter()
    write_html_report(output, make_manifest(), [make_stats()], trials, None)
    duration = time.perf_counter() - started

    assert duration < 30
    assert output.stat().st_size < 50 * 1024 * 1024
