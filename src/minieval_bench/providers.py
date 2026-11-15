from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass

from minieval_bench.models import ModelConfig

MAX_REQUEST_ATTEMPTS = 3
TRANSIENT_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504})
MAX_RETRY_AFTER_SECONDS = 30.0
ERROR_MESSAGE_LIMIT = 400

SECRET_PATTERNS = [
    (re.compile(r"sk-[A-Za-z0-9_-]{6,}"), "***"),
    (re.compile(r"(?i)\bbearer\s+\S+"), "Bearer ***"),
    (re.compile(r"(https?://)[^\s/:@]+:[^\s/@]+@"), r"\1***@"),
    (re.compile(r"\?[^\s'\"]+"), "?***"),
    (
        re.compile(r"(?i)\b(api[_-]?key|token|password|secret|authorization)\b(\s*[:=]\s*)\S+"),
        r"\1\2***",
    ),
]


@dataclass
class ProviderResponse:
    text: str
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    attempts: int = 1


def mask_secrets(text: str) -> str:
    for pattern, replacement in SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def safe_error_message(exc: Exception) -> str:
    parts = []
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        parts.append(f"status {status_code}")
    message = str(exc).strip()
    if message:
        parts.append(mask_secrets(message))
    description = "; ".join(parts) or "no details"
    if len(description) > ERROR_MESSAGE_LIMIT:
        description = description[: ERROR_MESSAGE_LIMIT - 3] + "..."
    return description


def is_transient_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code in TRANSIENT_STATUS_CODES
    if isinstance(exc, TimeoutError | ConnectionError):
        return True
    name = type(exc).__name__.lower()
    return "timeout" in name or "connection" in name


def retry_delay_seconds(exc: Exception, attempt: int) -> float:
    retry_after = getattr(exc, "retry_after", None)
    if isinstance(retry_after, int | float) and 0 < retry_after <= MAX_RETRY_AFTER_SECONDS:
        return float(retry_after)
    return min(0.5 * 2**attempt, 8.0)


def cost_from_pricing(
    model: ModelConfig, input_tokens: int | None, output_tokens: int | None
) -> float | None:
    if model.input_cost_per_1m is None or model.output_cost_per_1m is None:
        return None
    if input_tokens is None or output_tokens is None:
        return None
    input_cost = input_tokens * model.input_cost_per_1m
    output_cost = output_tokens * model.output_cost_per_1m
    return (input_cost + output_cost) / 1_000_000


class MockProvider:
    def __init__(self, model: ModelConfig) -> None:
        self.model = model

    def complete(self, prompt: str, system: str | None, seed: int) -> ProviderResponse:
        payload = f"{self.model.model}|{system}|{prompt}|{seed}".encode()
        digest = hashlib.sha256(payload).hexdigest()
        time.sleep((int(digest[:4], 16) % 40 + 5) / 1000)
        input_tokens = len((system or "").split()) + len(prompt.split())
        output_tokens = 4
        return ProviderResponse(
            text=f"mock-{digest[:8]}",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_from_pricing(self.model, input_tokens, output_tokens),
        )


class LitellmProvider:
    def __init__(self, model: ModelConfig) -> None:
        self.model = model

    def complete(self, prompt: str, system: str | None, seed: int) -> ProviderResponse:
        import litellm

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict = {"timeout": self.model.timeout or 60, "seed": seed}
        if self.model.temperature is not None:
            kwargs["temperature"] = self.model.temperature
        if self.model.max_tokens is not None:
            kwargs["max_tokens"] = self.model.max_tokens
        if self.model.drop_params:
            kwargs["drop_params"] = True

        response, attempts = self._completion_with_retries(litellm, messages, kwargs)

        text = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        output_tokens = getattr(usage, "completion_tokens", None) if usage else None
        cost = cost_from_pricing(self.model, input_tokens, output_tokens)
        if cost is None:
            cost = self._fallback_cost(litellm, response)
        return ProviderResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            attempts=attempts,
        )

    def _completion_with_retries(self, litellm, messages: list[dict], kwargs: dict):
        for attempt in range(MAX_REQUEST_ATTEMPTS):
            try:
                return (
                    litellm.completion(model=self.model.model, messages=messages, **kwargs),
                    attempt + 1,
                )
            except Exception as exc:
                if attempt == MAX_REQUEST_ATTEMPTS - 1 or not is_transient_error(exc):
                    raise
                time.sleep(retry_delay_seconds(exc, attempt))
        raise RuntimeError("unreachable")

    def _fallback_cost(self, litellm, response) -> float | None:
        try:
            return litellm.completion_cost(response)
        except Exception:
            return None


def build_provider(model: ModelConfig) -> MockProvider | LitellmProvider:
    if model.provider == "mock":
        return MockProvider(model)
    return LitellmProvider(model)
