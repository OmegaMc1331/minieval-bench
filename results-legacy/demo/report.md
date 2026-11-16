# MiniEval report: mock-demo

- MiniEval version: 0.4.0
- Created (UTC): 2026-11-16T21:46:18.450739+00:00
- Seed: 42
- Repeats: 3
- Concurrency: 4
- Shuffle: False
- Duration (s): 0.3
- Dataset: `D:\MiniEval\datasets\example.jsonl` (6 entries, sha256 `3ee246991f64a0a7c3bb97749a1b997ea0a36f42a642cbe66bd9811d45c1c93d`)

## Results

| Model | Requests | OK | Failed | Success | Accuracy | Coverage | Overall | Median gen (ms) | P95 gen (ms) | Median eval (ms) | Median total (ms) | Queue (ms) | In tokens | Out tokens | Cost (USD) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mock-free | 18 | 18 | 0 | 100.0% | 50.0% | 100.0% | 50.0% | 25.8 | 40.5 | 0.1 | 25.8 | 39.8 | 96 | 72 | n/a |
| mock-priced | 18 | 18 | 0 | 100.0% | 50.0% | 100.0% | 50.0% | 32.1 | 42.5 | 0.1 | 32.1 | 153.6 | 96 | 72 | 0.000156 |

Indicators, with their explicit denominators:

- **Success** = requests completed without provider error / planned requests.
- **Accuracy** = correct answers / evaluable answers (successful requests).
- **Coverage** = evaluated trials / planned requests.
- **Overall** = correct answers / planned requests; failed requests count as not passed.

Latency columns measure the model call only (provider retries included). Evaluation latency measures the evaluator run and total latency covers generation plus evaluation. Reports written before 0.3.1 stored a single `latency_ms` mixing both; those values are shown as-is.

Token and cost totals are sums over the requests that reported them; `value (r/s)` marks a partial total covering r of s successful requests. `n/a` marks a value that is unavailable, as opposed to a real zero.
