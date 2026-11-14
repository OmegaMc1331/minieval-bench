from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

from minieval_bench.models import ModelConfig


@dataclass
class ProviderResponse:
    text: str
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None


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
    def __init__(self, model: ModelConfig, seed: int) -> None:
        self.model = model
        self.seed = seed

    def complete(self, prompt: str, system: str | None) -> ProviderResponse:
        payload = f"{self.model.model}|{system}|{prompt}|{self.seed}".encode()
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
    def __init__(self, model: ModelConfig, seed: int) -> None:
        self.model = model
        self.seed = seed

    def complete(self, prompt: str, system: str | None) -> ProviderResponse:
        import litellm

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict = {"timeout": self.model.timeout or 60, "seed": self.seed}
        if self.model.temperature is not None:
            kwargs["temperature"] = self.model.temperature
        if self.model.max_tokens is not None:
            kwargs["max_tokens"] = self.model.max_tokens
        if self.model.drop_params:
            kwargs["drop_params"] = True

        response = litellm.completion(model=self.model.model, messages=messages, **kwargs)

        text = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        output_tokens = getattr(usage, "completion_tokens", None) if usage else None
        cost = cost_from_pricing(self.model, input_tokens, output_tokens)
        if cost is None:
            cost = self._fallback_cost(response)
        return ProviderResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )

    def _fallback_cost(self, response) -> float | None:
        import litellm

        try:
            return litellm.completion_cost(response)
        except Exception:
            return None


def build_provider(model: ModelConfig, seed: int) -> MockProvider | LitellmProvider:
    if model.provider == "mock":
        return MockProvider(model, seed)
    return LitellmProvider(model, seed)
