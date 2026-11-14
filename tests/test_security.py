from __future__ import annotations

from typer.testing import CliRunner

from minieval_bench.cli import app
from minieval_bench.models import Manifest, ModelConfig
from minieval_bench.providers import mask_secrets, safe_error_message
from minieval_bench.reports import write_reports
from minieval_bench.runner import PlannedTrial, compute_stats, run_trial
from test_runner import make_entry

FAKE_OPENAI_KEY = "sk-TESTKEY123456789abcdef"
FAKE_HEADER_TOKEN = "leaker123456789"
FAKE_URL_SECRET = "SECRETQUERYVALUE123"
FAKE_PASSWORD = "hunter2password"


def test_mask_secrets_hides_openai_style_keys():
    assert FAKE_OPENAI_KEY not in mask_secrets(f"call failed with key {FAKE_OPENAI_KEY}")


def test_mask_secrets_keeps_field_names_and_hides_values():
    masked = mask_secrets(
        f"api_key={FAKE_OPENAI_KEY} token=abc123 password={FAKE_PASSWORD}"
    )

    assert FAKE_OPENAI_KEY not in masked
    assert FAKE_PASSWORD not in masked
    assert "api_key=" in masked
    assert "token=" in masked
    assert "password=" in masked


def test_mask_secrets_hides_bearer_tokens():
    masked = mask_secrets(f"Authorization: Bearer {FAKE_HEADER_TOKEN}")

    assert FAKE_HEADER_TOKEN not in masked


def test_mask_secrets_hides_url_credentials_and_query_parameters():
    masked = mask_secrets(
        f"GET https://user:{FAKE_PASSWORD}@api.example.com/v1?key={FAKE_URL_SECRET} failed"
    )

    assert FAKE_PASSWORD not in masked
    assert FAKE_URL_SECRET not in masked
    assert "api.example.com" in masked


def test_safe_error_message_keeps_status_and_masks_details():
    class LeakyAPIError(Exception):
        status_code = 401

    exc = LeakyAPIError(f"invalid api_key={FAKE_OPENAI_KEY}")

    message = safe_error_message(exc)

    assert message.startswith("status 401")
    assert FAKE_OPENAI_KEY not in message


def test_safe_error_message_truncates_long_messages():
    message = safe_error_message(Exception("x" * 2000))

    assert len(message) <= 400
    assert message.endswith("...")


def test_safe_error_message_without_details():
    assert safe_error_message(Exception()) == "no details"


class LeakyProvider:
    def complete(self, prompt: str, system: str | None, seed: int):
        raise ConnectionError(
            f"request to https://user:{FAKE_PASSWORD}@api.example.com/v1?key={FAKE_URL_SECRET} "
            f"failed; api_key={FAKE_OPENAI_KEY}; Authorization: Bearer {FAKE_HEADER_TOKEN}"
        )


def leaked_trial():
    model = ModelConfig(name="leaky", provider="litellm", model="openai/x")
    planned = PlannedTrial(model=model, entry=make_entry(), repeat=0, seed=1)
    return run_trial(LeakyProvider(), planned, queued_at=0.0)


def test_run_trial_sanitizes_error_messages():
    trial = leaked_trial()

    assert trial.ok is False
    assert trial.error_type == "ConnectionError"
    for secret in (FAKE_OPENAI_KEY, FAKE_PASSWORD, FAKE_URL_SECRET, FAKE_HEADER_TOKEN):
        assert secret not in trial.error_message
    assert "api.example.com" in trial.error_message


def test_exports_and_terminal_output_contain_no_secrets(tmp_path):
    trial = leaked_trial()
    stats = compute_stats([trial])
    manifest = Manifest(
        minieval_version="0.1.1",
        benchmark="leaky",
        created_at_utc="2026-11-14T00:00:00+00:00",
        seed=1,
        repeats=1,
        concurrency=1,
        dataset_path="dataset.jsonl",
        dataset_sha256="0" * 64,
        dataset_entries=1,
        models=[ModelConfig(name="leaky", provider="litellm", model="openai/x")],
    )
    output_dir = tmp_path / "results"
    write_reports(output_dir, manifest, stats, [trial])

    for path in output_dir.iterdir():
        content = path.read_text(encoding="utf-8")
        for secret in (FAKE_OPENAI_KEY, FAKE_PASSWORD, FAKE_URL_SECRET, FAKE_HEADER_TOKEN):
            assert secret not in content, f"{secret} leaked into {path.name}"

    result = CliRunner().invoke(app, ["report", "--input", str(output_dir)], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    for secret in (FAKE_OPENAI_KEY, FAKE_PASSWORD, FAKE_URL_SECRET, FAKE_HEADER_TOKEN):
        assert secret not in result.output
