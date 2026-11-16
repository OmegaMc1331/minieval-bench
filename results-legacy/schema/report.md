# MiniEval report: json-schema-demo

- MiniEval version: 0.4.0
- Created (UTC): 2026-11-16T21:46:23.083145+00:00
- Seed: 42
- Repeats: 1
- Concurrency: 2
- Shuffle: False
- Duration (s): 0.1
- Dataset: `D:\MiniEval\datasets\json-schema.jsonl` (3 entries, sha256 `7f71dfcdcffcf81197b08d8c25fd9919d06978cb5546435e69d307a1492c693c`)

## Results

| Model | Requests | OK | Failed | Success | Accuracy | Coverage | Overall | Median gen (ms) | P95 gen (ms) | Median eval (ms) | Median total (ms) | Queue (ms) | In tokens | Out tokens | Cost (USD) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mock-candidate | 3 | 3 | 0 | 100.0% | n/a | 100.0% | 0.0% | 33.1 | 40.5 | 0.1 | 33.2 | 1.8 | 27 | 12 | n/a |

Indicators, with their explicit denominators:

- **Success** = requests completed without provider error / planned requests.
- **Accuracy** = correct answers / evaluable answers (successful requests).
- **Coverage** = evaluated trials / planned requests.
- **Overall** = correct answers / planned requests; failed requests count as not passed.

Latency columns measure the model call only (provider retries included). Evaluation latency measures the evaluator run and total latency covers generation plus evaluation. Reports written before 0.3.1 stored a single `latency_ms` mixing both; those values are shown as-is.

Token and cost totals are sums over the requests that reported them; `value (r/s)` marks a partial total covering r of s successful requests. `n/a` marks a value that is unavailable, as opposed to a real zero.

## Evaluation breakdown

Accuracy above covers deterministic metrics only. Each evaluator below has its own denominator (binary trials of that evaluator).

| Model | Deterministic | Answer accuracy | JSON conformance | Code pass | Code tests | Judge score | Judge judged/eligible |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mock-candidate | 0 | n/a | 0.0% (3) | n/a (0) | n/a | n/a | 0 / 0 |
