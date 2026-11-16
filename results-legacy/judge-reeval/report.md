# MiniEval report: judge-mock-demo

- MiniEval version: 0.4.0
- Created (UTC): 2026-11-16T21:46:38.919915+00:00
- Seed: 42
- Repeats: 2
- Concurrency: 2
- Shuffle: False
- Duration (s): 0.0
- Dataset: `D:\MiniEval\datasets\judge-mock.jsonl` (3 entries, sha256 `4e10ca91c866af284a04ecfdaf89a20fdffb9affa385e6c2955a0828c2937175`)

## Results

| Model | Requests | OK | Failed | Success | Accuracy | Coverage | Overall | Median gen (ms) | P95 gen (ms) | Median eval (ms) | Median total (ms) | Queue (ms) | In tokens | Out tokens | Cost (USD) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mock-candidate | 6 | 6 | 0 | 100.0% | n/a | 0.0% | 0.0% | 30.8 | 43.4 | 0.6 | 31.5 | 31.3 | 48 | 24 | n/a |
| mock-priced | 6 | 6 | 0 | 100.0% | n/a | 0.0% | 0.0% | 32.9 | 40.8 | 0.5 | 33.4 | 123.5 | 48 | 24 | 0.000060 |

Indicators, with their explicit denominators:

- **Success** = requests completed without provider error / planned requests.
- **Accuracy** = correct answers / evaluable answers (successful requests).
- **Coverage** = evaluated trials / planned requests.
- **Overall** = correct answers / planned requests; failed requests count as not passed.

Latency columns measure the model call only (provider retries included). Evaluation latency measures the evaluator run and total latency covers generation plus evaluation. Reports written before 0.3.1 stored a single `latency_ms` mixing both; those values are shown as-is.

Token and cost totals are sums over the requests that reported them; `value (r/s)` marks a partial total covering r of s successful requests. `n/a` marks a value that is unavailable, as opposed to a real zero.

## Evaluation overhead

| Model | Evaluator failures | Unavailable | Evaluation cost (USD) |
| --- | --- | --- | --- |
| mock-candidate | 0 | 0 | 0.000624 |
| mock-priced | 0 | 0 | 0.000624 |

## Evaluation breakdown

Accuracy above covers deterministic metrics only. Each evaluator below has its own denominator (binary trials of that evaluator).

| Model | Deterministic | Answer accuracy | JSON conformance | Code pass | Code tests | Judge score | Judge judged/eligible |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mock-candidate | 0 | n/a | n/a (0) | n/a (0) | n/a | 0.483 | 6 / 6 |
| mock-priced | 0 | n/a | n/a (0) | n/a (0) | n/a | 0.471 | 6 / 6 |
