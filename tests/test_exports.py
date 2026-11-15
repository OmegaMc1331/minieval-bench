from __future__ import annotations

import csv

from minieval_bench.exports import (
    CSV_COLUMNS,
    csv_value,
    neutralize_formula,
    write_evaluations_csv,
    write_trials_csv,
)
from minieval_bench.models import EvaluationOutcome
from test_stats import make_trial


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


def test_csv_value_conventions():
    assert csv_value(None) == ""
    assert csv_value(True) == "true"
    assert csv_value(False) == "false"
    assert csv_value(1.5) == "1.5"
    assert csv_value("plain") == "plain"


def test_neutralize_formula_prefixes():
    assert neutralize_formula("=SUM(A1:A2)") == "'=SUM(A1:A2)"
    assert neutralize_formula("+1") == "'+1"
    assert neutralize_formula("-2+3") == "'-2+3"
    assert neutralize_formula("@cmd") == "'@cmd"
    assert neutralize_formula("\tTAB") == "'\tTAB"
    assert neutralize_formula("safe") == "safe"
    assert neutralize_formula("2 - 3") == "2 - 3"


def test_export_writes_header_and_rows(tmp_path):
    trials = [make_trial(model="model-a"), make_trial(model="model-a", ok=False, error_type="Boom")]
    output = tmp_path / "trials.csv"

    count = write_trials_csv(output, trials)

    rows = read_rows(output)
    assert count == 2
    assert rows[0] == CSV_COLUMNS
    assert rows[1][0] == "model-a"
    assert rows[1][3] == "true"
    assert rows[1][4] == "true"
    assert rows[2][3] == "false"
    assert rows[2][4] == ""
    assert rows[2][CSV_COLUMNS.index("error_type")] == "Boom"


def test_export_preserves_missing_values_as_empty(tmp_path):
    trials = [make_trial(input_tokens=None, output_tokens=None, cost_usd=None)]
    output = tmp_path / "trials.csv"

    write_trials_csv(output, trials)

    row = read_rows(output)[1]
    assert row[CSV_COLUMNS.index("input_tokens")] == ""
    assert row[CSV_COLUMNS.index("cost_usd")] == ""
    assert row[CSV_COLUMNS.index("queue_ms")] == ""


def test_export_handles_special_characters_and_formulas(tmp_path):
    tricky = make_trial(error_type="E", ok=False)
    tricky.error_message = '=SUM(A1)\nLine2, "quoted"; é中🙂'
    output = tmp_path / "trials.csv"

    write_trials_csv(output, [tricky])

    row = read_rows(output)[1]
    assert row[CSV_COLUMNS.index("error_message")] == "'=SUM(A1)\nLine2, \"quoted\"; é中🙂"


def test_export_is_utf8_encoded(tmp_path):
    trial = make_trial()
    trial.entry_id = "entrée-é中"
    output = tmp_path / "trials.csv"

    write_trials_csv(output, [trial])

    raw = output.read_bytes()
    assert "entrée-é中".encode() in raw




def test_evaluations_csv_extracts_per_evaluator_fields(tmp_path):
    judged = make_trial()
    judged.evaluation = EvaluationOutcome(
        evaluator="judge",
        status="failed",
        passed=None,
        error_message="judge returned invalid JSON",
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.01,
        details={"judgments": 0},
    )
    coded = make_trial()
    coded.evaluation = EvaluationOutcome(
        evaluator="code_tests", status="completed", passed=True, score=1.0,
        details={"passed": 2, "total": 2},
    )
    schemad = make_trial()
    schemad.entry_id = "e3"
    schemad.evaluation = EvaluationOutcome(
        evaluator="json_schema", status="completed", passed=False, score=0.0,
        details={"errors": [{"path": "$.a", "message": "required"}]},
    )
    output = tmp_path / "evaluations.csv"

    count = write_evaluations_csv(output, [judged, coded, schemad, make_trial()])

    assert count == 3
    rows = read_rows(output)
    assert rows[0][0:5] == ["model", "entry_id", "repeat", "evaluator", "status"]
    judged_row = rows[1]
    assert judged_row[EVALUATION_INDEX("evaluator")] == "judge"
    assert judged_row[EVALUATION_INDEX("judgments")] == "0"
    assert judged_row[EVALUATION_INDEX("evaluation_input_tokens")] == "100"
    assert judged_row[EVALUATION_INDEX("evaluation_cost_usd")] == "0.01"
    assert judged_row[EVALUATION_INDEX("code_tests_passed")] == ""
    coded_row = rows[2]
    assert coded_row[EVALUATION_INDEX("code_tests_passed")] == "2"
    assert coded_row[EVALUATION_INDEX("code_tests_total")] == "2"
    schema_row = rows[3]
    assert schema_row[EVALUATION_INDEX("json_errors")] == "1"


def EVALUATION_INDEX(column: str) -> int:
    from minieval_bench.exports import EVALUATION_COLUMNS

    return EVALUATION_COLUMNS.index(column)


def test_export_cli_writes_evaluations_file(tmp_path):
    from typer.testing import CliRunner

    from conftest import EXAMPLE_CONFIG
    from minieval_bench.cli import app

    cli = CliRunner()
    output_dir = tmp_path / "out"
    cli.invoke(app, ["run", "--config", str(EXAMPLE_CONFIG), "--output", str(output_dir),
                     "--no-html"], env={"COLUMNS": "200"})

    result = cli.invoke(app, ["export", "--input", str(output_dir)], env={"COLUMNS": "200"})

    assert result.exit_code == 0
    evaluations = output_dir / "evaluations.csv"
    assert evaluations.is_file()
    content = evaluations.read_text(encoding="utf-8")
    assert "deterministic" in content
