# Reproducibility

MiniEval 0.4.0 makes benchmark runs reproducible at the **protocol level**:
the same suite, datasets, sampling, models and parameters produce the same
execution plan. This document explains what is pinned, how to verify it, and
what reproducibility cannot cover.

## The lockfile

Every `suite run` writes `benchmark.lock.json` next to the results. It is a
deterministic JSON document: no timestamps, no absolute paths, sorted keys.
Identical protocol inputs produce byte-identical lockfiles.

It pins:

- the MiniEval version;
- the suite id, version and a SHA-256 fingerprint of the whole suite
  definition (tasks, weights, defaults, evaluators, statistics);
- per task: the dataset reference and content SHA-256, the sampling record
  (method, seed, selected ids), the metric census, and fingerprints of every
  JSON schema and code-test module;
- the effective run parameters, the list of user overrides, and the full model
  configurations;
- the evaluator blocks: judge rubric fingerprint, rubric version, criteria,
  model and repeats; the complete code runtime configuration including the
  pinned image digest;
- the statistics configuration.

## Verifying a run

```bash
minieval-bench suite verify results/my-suite/
```

The command reloads the suite from the registry (using the run's recorded
models and options), recomputes the lockfile and reports every difference:
dataset content changed, sampling selection changed, schema or test module
changed, rubric or evaluator configuration changed, model parameters changed,
suite version drift, added or removed tasks, or the suite being unavailable.
It never calls a provider.

Modification detection also drives `--resume`: a stored task result is reused
only when its lock section still matches.

## Comparing runs

`suite compare` first compares the lockfiles. Identical protocols get
per-task score and latency differences plus paired per-question statistics
(reuses the v0.3.1 bootstrap machinery; the interval is not a significance
test). Different protocols get precise warnings, no deltas and no ranking.

## Limits

- **Protocol reproducibility is not output reproducibility.** A hosted model
  can change server-side (quantization, replacement, routing); identical
  protocol inputs do not guarantee identical responses. The lockfile pins what
  MiniEval controls: data, parameters, evaluators.
- Provider-side sampling temperature zero still does not guarantee
  determinism for every provider.
- Latencies and costs depend on the machine, the network and the provider
  load; they are reported per run and compared descriptively.
- Docker-based code evaluation pins the image by digest; the host kernel and
  Docker engine remain outside the fingerprint.
- SHA-256 fingerprints verify integrity against a reference value. They do
  not authenticate the producer of a file; no cryptographic signature is
  implemented in this version.

## Sampling

`--sample N --sample-seed S` selects a deterministic subset per task: entries
are sorted by id and drawn with a RNG seeded from
`sha256("<seed>:<dataset-sha256>")`, independently of the global Python RNG.
`--sample-mode stratified` requires a `category` on every entry and allocates
the sample proportionally per category (largest-remainder method).

The method, seed, count, selected ids and source fingerprint are recorded in
the lockfile and in the report. A sampled run is labelled as a subset: its
scores are never presented as the full suite score, and any sampling
difference between two runs blocks their comparison.
