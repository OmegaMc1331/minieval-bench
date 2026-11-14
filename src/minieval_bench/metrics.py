from __future__ import annotations

import re


def normalize(text: str) -> str:
    lowered = text.strip().lower()
    without_punctuation = re.sub(r"[^\w\s]", "", lowered)
    return re.sub(r"\s+", " ", without_punctuation)


def exact_match(response: str, reference: str) -> bool:
    return normalize(response) == normalize(reference)


def contains(response: str, reference: str) -> bool:
    return normalize(reference) in normalize(response)


def evaluate(metric: str, response: str, reference: str) -> bool:
    if metric == "exact_match":
        return exact_match(response, reference)
    if metric == "contains":
        return contains(response, reference)
    if metric == "regex":
        return re.search(reference, response) is not None
    raise ValueError(f"unknown metric: {metric}")
