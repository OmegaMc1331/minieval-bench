from __future__ import annotations

import json

from typer.testing import CliRunner

from conftest import EXAMPLE_CONFIG, REPO_ROOT, write_jsonl
from minieval_bench.cli import app

runner = CliRunner()
WIDE = {"COLUMNS": "200"}

JUDGE_CONFIG = REPO_ROOT / "benchmarks" / "judge-mock.yaml"
SCHEMA_CONFIG = REPO_ROOT / "benchmarks" / "json-schema.yaml"
CODE_CONFIG = REPO_ROOT / "benchmarks" / "code-python.yaml"


def test_run_classic_benchmark_records_deterministic_evaluations(tmp_path):
    output_dir = tmp_path / "classic"
    result = runner.invoke(
        app, ["run", "--config", str(EXAMPLE_CONFIG), "--output", str(output_dir)], env=WIDE
    )

    assert result.exit_code == 0
    results = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    evaluations = [trial["evaluation"] for trial in results["trials"]]
    assert all(evaluation["evaluator"] == "deterministic" for evaluation in evaluations)
    assert all(evaluation["status"] == "completed" for evaluation in evaluations)
    assert results["evaluators"][0]["evaluator"] == "deterministic"
    assert results["statistics"][0]["pass_rate"]["summary"]["count"] > 0


def test_run_with_mock_judge_produces_judge_scores(tmp_path):
    output_dir = tmp_path / "judged"
    result = runner.invoke(
        app, ["run", "--config", str(JUDGE_CONFIG), "--output", str(output_dir)], env=WIDE
    )

    assert result.exit_code == 0
    assert "Judge enabled" in result.output
    assert "24 call(s) planned" in result.output

    results = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    judged = [trial for trial in results["trials"] if trial["evaluation"]["evaluator"] == "judge"]
    assert len(judged) == 3 * 2 * 2
    for trial in judged:
        evaluation = trial["evaluation"]
        assert evaluation["status"] == "completed"
        assert evaluation["passed"] is None
        assert 0.0 <= evaluation["score"] <= 1.0
        assert evaluation["details"]["judgments"] == 2
        assert evaluation["cost_usd"] is not None

    summaries = {item["evaluator"]: item for item in results["evaluators"]}
    assert summaries["judge"]["mean_score"] is not None
    assert summaries["judge"]["cost_usd"] > 0

    stats_by_model = {item["model"]: item for item in results["stats"]}
    for model_stats in stats_by_model.values():
        assert model_stats["evaluable"] == 0
        assert model_stats["answer_accuracy"] is None
        assert model_stats["evaluation_coverage"] == 0.0

    html = (output_dir / "report.html").read_text(encoding="utf-8")
    assert "Judge scores" in html or "judge" in html


def test_run_with_json_schema_reports_conformance(tmp_path):
    output_dir = tmp_path / "schema"
    result = runner.invoke(
        app, ["run", "--config", str(SCHEMA_CONFIG), "--output", str(output_dir)], env=WIDE
    )

    assert result.exit_code == 0
    results = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    evaluations = [trial["evaluation"] for trial in results["trials"]]
    assert all(evaluation["evaluator"] == "json_schema" for evaluation in evaluations)

    for evaluation in evaluations:
        assert evaluation["status"] == "completed"
        assert evaluation["passed"] is False
        assert evaluation["details"]["json_valid"] is False
        assert evaluation["details"]["errors"][0]["path"] == "$"

    summary = results["evaluators"][0]
    assert summary["evaluator"] == "json_schema"
    assert summary["trials"] == 3
    assert summary["passed"] == 0
    assert summary["mean_score"] == 0.0

    html = (output_dir / "report.html").read_text(encoding="utf-8")
    assert "Structural conformance" in html


def test_json_schema_accepts_valid_json_responses():
    from minieval_bench.evaluation.structured import evaluate_json_schema

    schema = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
    outcome = evaluate_json_schema('{"n": 3}', schema)

    assert outcome.passed is True
    assert outcome.details["json_valid"] is True


def test_run_with_code_tests_reports_unavailable_without_docker(tmp_path, monkeypatch):
    import minieval_bench.evaluation.code as code_module

    monkeypatch.setattr(
        code_module, "docker_available", lambda: (False, "docker daemon is not reachable")
    )
    output_dir = tmp_path / "code"
    result = runner.invoke(
        app, ["run", "--config", str(CODE_CONFIG), "--output", str(output_dir)], env=WIDE
    )

    assert result.exit_code == 0
    results = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    evaluations = [trial["evaluation"] for trial in results["trials"]]
    assert all(evaluation["evaluator"] == "code_tests" for evaluation in evaluations)
    assert all(evaluation["status"] == "unavailable" for evaluation in evaluations)
    assert all("unavailable" in evaluation["error_message"] for evaluation in evaluations)

    stats = results["stats"][0]
    assert stats["evaluator_unavailable"] == 2
    assert stats["evaluable"] == 0
    assert stats["answer_accuracy"] is None


def test_run_refuses_judge_plan_above_max_calls(tmp_path):
    config_dir = tmp_path / "bench"
    config_dir.mkdir()
    write_jsonl(tmp_path / "dataset.jsonl", [
        {"id": f"e{index}", "prompt": "p", "metric": "judge"} for index in range(4)
    ])
    config = """
benchmark: capped
dataset: ../dataset.jsonl
defaults:
  repeats: 2
models:
  - name: mock-candidate
    provider: mock
evaluators:
  judge:
    provider: mock
    repeats: 2
    max_calls: 4
"""
    config_path = config_dir / "capped.yaml"
    config_path.write_text(config, encoding="utf-8")

    result = runner.invoke(
        app, ["run", "--config", str(config_path), "--output", str(tmp_path / "out")], env=WIDE
    )

    assert result.exit_code == 1
    assert "exceed max_calls" in result.output


def test_run_refuses_monetary_budget_without_pricing(tmp_path):
    config_dir = tmp_path / "bench"
    config_dir.mkdir()
    write_jsonl(tmp_path / "dataset.jsonl", [
        {"id": "e1", "prompt": "p", "metric": "judge"},
    ])
    config = """
benchmark: budgeted
dataset: ../dataset.jsonl
models:
  - name: mock-candidate
    provider: mock
evaluators:
  judge:
    provider: mock
    budget_usd: 1.0
"""
    config_path = config_dir / "budget.yaml"
    config_path.write_text(config, encoding="utf-8")

    result = runner.invoke(
        app, ["run", "--config", str(config_path), "--output", str(tmp_path / "out")], env=WIDE
    )

    assert result.exit_code == 1
    assert "budget_usd requires judge pricing" in result.output


def test_evaluate_reevaluates_without_new_generation(tmp_path, monkeypatch):
    import minieval_bench.evaluation.code as code_module

    monkeypatch.setattr(
        code_module, "docker_available", lambda: (False, "docker daemon is not reachable")
    )
    run_dir = tmp_path / "run"
    runner.invoke(
        app, ["run", "--config", str(SCHEMA_CONFIG), "--output", str(run_dir), "--no-html"],
        env=WIDE,
    )
    original = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))

    result = runner.invoke(
        app,
        ["evaluate", "--input", str(run_dir), "--config", str(SCHEMA_CONFIG)],
        env=WIDE,
    )

    assert result.exit_code == 0
    reevaluated_dir = run_dir / "reevaluated"
    reevaluated = json.loads((reevaluated_dir / "results.json").read_text(encoding="utf-8"))

    assert [trial["response"] for trial in reevaluated["trials"]] == [
        trial["response"] for trial in original["trials"]
    ]
    assert [trial["latency_ms"] for trial in reevaluated["trials"]] == [
        trial["latency_ms"] for trial in original["trials"]
    ]
    assert all(trial["evaluation"] for trial in reevaluated["trials"])
    assert "left untouched" in result.output


def test_evaluate_reports_missing_entries(tmp_path, mock_config, valid_entries, write_benchmark):
    run_dir = tmp_path / "run"
    runner.invoke(
        app,
        ["run", "--config", str(write_benchmark(mock_config, valid_entries)),
         "--output", str(run_dir), "--no-html"],
        env=WIDE,
    )
    changed_entries = [dict(entry, id=entry["id"] + "-x") for entry in valid_entries]
    changed_config = write_benchmark(mock_config, changed_entries)

    result = runner.invoke(
        app, ["evaluate", "--input", str(run_dir), "--config", str(changed_config)], env=WIDE
    )

    assert result.exit_code == 0
    assert "kept unchanged" in result.output


def test_pairwise_runs_blind_comparison_with_mock_judge(tmp_path):
    run_dir = tmp_path / "run"
    runner.invoke(
        app,
        ["run", "--config", str(JUDGE_CONFIG), "--output", str(run_dir), "--no-html"],
        env=WIDE,
    )

    result = runner.invoke(
        app, ["pairwise", "--input", str(run_dir), "--config", str(JUDGE_CONFIG)], env=WIDE
    )

    assert result.exit_code == 0
    assert "mock-candidate" in result.output
    payload = json.loads((run_dir / "pairwise.json").read_text(encoding="utf-8"))
    assert payload["summary"]["comparisons"] == 3 * 2
    assert payload["modelA"] != payload["modelB"]
    html = (run_dir / "pairwise.html").read_text(encoding="utf-8")
    assert "pairwise" in html.lower()
    assert "id=\"report-data\"" in html
    for record in payload["records"]:
        assert len(record["judgments"]) == 2
        assert record["judgments"][0]["order"] == "first"
        assert record["judgments"][1]["order"] == "second"
        assert all(
            judgment["preferred"] in (payload["modelA"], payload["modelB"], "tie")
            for judgment in record["judgments"]
        )


def test_pairwise_requires_judge_configuration(
    tmp_path, mock_config, valid_entries, write_benchmark
):
    run_dir = tmp_path / "run"
    config_path = write_benchmark(mock_config, valid_entries)
    runner.invoke(
        app,
        ["run", "--config", str(config_path), "--output", str(run_dir), "--no-html"],
        env=WIDE,
    )

    result = runner.invoke(
        app, ["pairwise", "--input", str(run_dir), "--config", str(config_path)], env=WIDE
    )

    assert result.exit_code == 1
    assert "evaluators.judge" in result.output


def test_pairwise_validates_model_names(tmp_path):
    run_dir = tmp_path / "run"
    runner.invoke(
        app, ["run", "--config", str(JUDGE_CONFIG), "--output", str(run_dir), "--no-html"],
        env=WIDE,
    )

    result = runner.invoke(
        app,
        [
            "pairwise", "--input", str(run_dir), "--config", str(JUDGE_CONFIG),
            "--models", "mock-candidate",
        ],
        env=WIDE,
    )

    assert result.exit_code == 1
    assert "exactly two" in result.output


def test_calibrate_compares_mock_judge_to_references(tmp_path):
    config = """
benchmark: calibration-demo
dataset: ../datasets/judge-mock.jsonl
models:
  - name: mock-candidate
    provider: mock
evaluators:
  judge:
    provider: mock
    rubric: ../rubrics/general.yaml
    calibration: ../datasets/judge-calibration.jsonl
"""
    config_dir = tmp_path / "bench"
    config_dir.mkdir()
    (tmp_path / "datasets").mkdir()
    (tmp_path / "rubrics").mkdir()
    import shutil

    shutil.copy(REPO_ROOT / "rubrics" / "general.yaml", tmp_path / "rubrics" / "general.yaml")
    shutil.copy(
        REPO_ROOT / "datasets" / "judge-mock.jsonl", tmp_path / "datasets" / "judge-mock.jsonl"
    )
    shutil.copy(
        REPO_ROOT / "datasets" / "judge-calibration.jsonl",
        tmp_path / "datasets" / "judge-calibration.jsonl",
    )
    config_path = config_dir / "calibration.yaml"
    config_path.write_text(config, encoding="utf-8")
    output = tmp_path / "calibration.json"

    result = runner.invoke(
        app,
        ["calibrate", "--config", str(config_path), "--output", str(output)],
        env=WIDE,
    )

    assert result.exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert set(report["criteria"]) == {
        "correctness", "relevance", "instruction_following", "clarity"
    }
    assert report["criteria"]["correctness"]["items"] == 2
    assert len(report["items"]) == 2
    assert "does not prove judge validity" in report["note"]
    assert "Judge calibration" in result.output


def test_calibrate_requires_calibration_file(tmp_path):
    result = runner.invoke(app, ["calibrate", "--config", str(JUDGE_CONFIG)], env=WIDE)

    assert result.exit_code == 1
    assert "judge.calibration" in result.output


def test_evaluate_records_provenance_and_verified_matching(tmp_path):
    run_dir = tmp_path / "run"
    runner.invoke(
        app, ["run", "--config", str(SCHEMA_CONFIG), "--output", str(run_dir), "--snapshot",
              "--no-html"],
        env=WIDE,
    )

    result = runner.invoke(
        app, ["evaluate", "--input", str(run_dir), "--config", str(SCHEMA_CONFIG)], env=WIDE
    )

    assert result.exit_code == 0
    reevaluated = json.loads(
        (run_dir / "reevaluated" / "results.json").read_text(encoding="utf-8")
    )
    manifest = reevaluated["manifest"]
    assert manifest["source_run"]["dataset_sha256"] == manifest["reevaluation"]["dataset_sha256"]
    assert manifest["reevaluation"]["matching"]["basis"] == "dataset-identical"
    assert manifest["reevaluation"]["matching"]["verified"] > 0
    assert manifest["reevaluation"]["matching"]["conflicts"] == []
    assert "Source run" in result.output


def test_evaluate_flags_changed_prompts_as_conflicts(tmp_path):
    run_dir = tmp_path / "run"
    runner.invoke(
        app, ["run", "--config", str(SCHEMA_CONFIG), "--output", str(run_dir), "--snapshot",
              "--no-html"],
        env=WIDE,
    )
    original = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    schema_path = tmp_path / "schema-changed.jsonl"
    first = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))["dataset"][0]
    changed = dict(first, prompt=first["prompt"] + " with an extra instruction")
    schema_path.write_text(json.dumps(changed) + "\n", encoding="utf-8")
    config_path = tmp_path / "changed.yaml"
    config_path.write_text(
        f"benchmark: changed\ndataset: {schema_path}\nmodels:\n  - name: mock-demo\n"
        "    provider: mock\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app, ["evaluate", "--input", str(run_dir), "--config", str(config_path)], env=WIDE
    )

    assert result.exit_code == 0
    assert "changed since the source run" in result.output
    reevaluated = json.loads(
        (run_dir / "reevaluated" / "results.json").read_text(encoding="utf-8")
    )
    matching = reevaluated["manifest"]["reevaluation"]["matching"]
    assert matching["conflicts"] == [first["id"]]
    assert reevaluated["trials"][0]["evaluation"] == original["trials"][0]["evaluation"]


def test_evaluate_warns_when_matching_is_unverified(tmp_path):
    run_dir = tmp_path / "run"
    runner.invoke(
        app, ["run", "--config", str(SCHEMA_CONFIG), "--output", str(run_dir), "--no-html"],
        env=WIDE,
    )
    schema_path = tmp_path / "schema-other.jsonl"
    schema_path.write_text(
        '{"id": "c1", "prompt": "different question entirely", "metric": "json_schema",'
        ' "json_schema": {"type": "object"}}\n',
        encoding="utf-8",
    )
    config_path = tmp_path / "other.yaml"
    config_path.write_text(
        f"benchmark: other\ndataset: {schema_path}\nmodels:\n  - name: mock-demo\n"
        "    provider: mock\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app, ["evaluate", "--input", str(run_dir), "--config", str(config_path)], env=WIDE
    )

    assert result.exit_code == 0
    assert "cannot be verified" in result.output
    reevaluated = json.loads(
        (run_dir / "reevaluated" / "results.json").read_text(encoding="utf-8")
    )
    matching = reevaluated["manifest"]["reevaluation"]["matching"]
    assert matching["basis"] == "entry-id-only"
    assert matching["unverified"] >= 1


def test_run_manifest_records_judge_rubric_fingerprint(tmp_path):
    run_dir = tmp_path / "run"
    runner.invoke(
        app, ["run", "--config", str(JUDGE_CONFIG), "--output", str(run_dir), "--no-html"],
        env=WIDE,
    )

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["judge"]["model"] is None
    assert manifest["judge"]["provider"] == "mock"
    assert len(manifest["judge"]["rubric_fingerprint"]) == 64
    assert [c["id"] for c in manifest["judge"]["criteria"]] == [
        "correctness", "relevance", "instruction_following", "clarity"
    ]
