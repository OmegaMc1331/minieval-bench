from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from minieval_bench.models import DatasetEntry


class SamplingError(Exception):
    pass


@dataclass
class SamplingPlan:
    method: str = "none"
    count: int | None = None
    seed: int | None = None


def apply_sampling(
    entries: list[DatasetEntry], dataset_sha256: str, plan: SamplingPlan
) -> tuple[list[DatasetEntry], dict]:
    if plan.method == "none":
        return entries, {"method": "none"}
    if plan.method not in ("random", "stratified"):
        raise SamplingError(f"unknown sampling method '{plan.method}'")
    if plan.count is None or plan.count <= 0:
        raise SamplingError("--sample requires a positive number of entries")
    ordered = sorted(entries, key=lambda entry: entry.id)
    record = {
        "method": plan.method,
        "seed": plan.seed if plan.seed is not None else 0,
        "source_sha256": dataset_sha256,
    }
    if plan.count >= len(ordered):
        selected = ordered
    else:
        rng = random.Random(sample_seed(record["seed"], dataset_sha256))
        if plan.method == "random":
            selected = sample_random(ordered, plan.count, rng)
        else:
            selected = sample_stratified(ordered, plan.count, rng)
    record["requested_count"] = plan.count
    record["selected_count"] = len(selected)
    record["selected_ids"] = [entry.id for entry in selected]
    return selected, record


def sample_seed(seed: int, dataset_sha256: str) -> int:
    payload = f"{seed}:{dataset_sha256}".encode()
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def sample_random(
    entries: list[DatasetEntry], count: int, rng: random.Random
) -> list[DatasetEntry]:
    picked = rng.sample(entries, count)
    return sorted(picked, key=lambda entry: entry.id)


def sample_stratified(
    entries: list[DatasetEntry], count: int, rng: random.Random
) -> list[DatasetEntry]:
    by_category: dict[str, list[DatasetEntry]] = {}
    for entry in entries:
        if entry.category is None:
            raise SamplingError(
                "stratified sampling requires a category on every entry; "
                f"entry '{entry.id}' has none"
            )
        by_category.setdefault(entry.category, []).append(entry)
    allocations = proportional_allocation(
        {category: len(members) for category, members in by_category.items()}, count
    )
    picked: list[DatasetEntry] = []
    for category in sorted(by_category):
        picked.extend(rng.sample(by_category[category], allocations[category]))
    return sorted(picked, key=lambda entry: entry.id)


def proportional_allocation(sizes: dict[str, int], total: int) -> dict[str, int]:
    population = sum(sizes.values())
    exact = {key: total * size / population for key, size in sizes.items()}
    allocation = {key: int(value) for key, value in exact.items()}
    remainder = total - sum(allocation.values())
    order = sorted(sizes, key=lambda key: (-(exact[key] - allocation[key]), key))
    for key in order[:remainder]:
        allocation[key] += 1
    return allocation
