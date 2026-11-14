from __future__ import annotations

import csv

from minieval_bench.exports import CSV_COLUMNS, csv_value, neutralize_formula, write_trials_csv
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
