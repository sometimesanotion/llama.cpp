# Benchmarking the decision engine

This is the practical guide for measuring `POST /v1/decision` and the engine behind
it. It covers what to measure, how to reproduce it, which frozen artifacts own the
results, and how not to fool yourself with a number.

The rule that governs everything here: a producer concentration score
(`confidence`, `certainty`) is never a benchmark gate and never a benchmark
result. It is reported telemetry. All benchmarks below measure either speed
(task value), outcome correctness (task value), or reproducibility (which is its
own axis).

Diagram of the axes:

```
speed          -> prefill_ms, scoring_ms, total_ms, cache_hit, rounds, rows
task value     -> winner agreement, Brier, ECE, fork byte equality, framing choice
reproducibility-> frozen readout baselines, calibration ledger, frozen backend flags
```

## 1. What to measure

| Question | Tool | Artifact |
|---|---|---|
| How fast is a decision warm and cold? | `timings` on the `/v1/decision` response | response body |
| How fast is it end to end? | `timings` on the `/v1/decision` response | response body |
| Does the fork reproduce the parent state? | `test-decision-engine` fork oracle | `[fork control]` log line |
| Does a refactor move the readout? | `test-decision-engine --check-readout` | `readout_{cpu,gpu}_baseline.json` |
| Is the model accurate on a labeled set? | `test_decision_accuracy.py` | `accuracy_report.json` |
| Does a decision starve chat or `/slots`? | `test_decision_admission.py` | `fairness.json` bound |
| Is a heuristic sign-off still current? | `test-decision-engine` calibration tests | `calibration.json` |

## 2. Freeze the environment before you measure

A decision number is only comparable to another decision number when the model,
quantization, prompt template, and backend flags match. The frozen readout and
calibration artifacts record the flags they were measured under.

The frozen set:

| Flag | Value used by the recorded baselines | Why it must be fixed |
|---|---|---|
| model + quantization | recorded as the last two path components | weights move every probability |
| `template_hash` | `make_prefix_tag(...)` | the framed prompt moves the label boundary |
| `kv_unified` | `true` | decisions fork on a unified cache |
| `swa_full` | `false` | sliding-window layout changes retained cells |
| `n_ctx` | `2048` in the server decision tests, server defaults otherwise | changes whether a request fits |
| `n_batch` / `n_ubatch` | `512` / `512` | batch split changes rounding and pass count |
| `n_seq_max` | `10` | number of live branches |
| `gpu_layers` | `-1` | CPU vs GPU changes producer numerics |
| `flash_attn` | `false` | ROCm FA is not bit-reproducible across runs |

If any of these changes, prior speed and accuracy numbers are stale. Re-measure
before you compare.

Two lanes:

- GPU lane: the only lane for speed, fork exactness on a real layout, and the
  model matrix. Set `LLAMA_DECISION_TEST_MODEL` to the GGUF under test.
- CPU lane: host-runnable tests and the frozen CPU readout oracle. It loads the
  generated dummy model from the `generate-models` fixture, or falls back to
  `LLAMA_DECISION_TEST_MODEL`.

## 3. Decision latency

The unified shape reports its own phases on every response `timings` object
(`prefill_ms`, `scoring_ms`, `total_ms`, `rounds`, `rows`,
`per_decision_ms`). Warm and cold are compared through `cache_hit` (true only on
the warm call) and the cold cost in the first run. `rows` is the number of
scored branch rows and `rounds` the number of decode waves: the letter readout
always scores the exact tree, so `rows` is the trie's divergence-node count.

A quick latency probe:

```sh
curl -s http://localhost:8096/v1/decision -H 'Content-Type: application/json' -d @request.json \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["timings"])'
```

Compare the same request with and without `"cache_prompt": true` to separate
the cold prefill cost from the warm path. A cold run on one branch against a
warm run on another is the classic false-regression trap.

## 4. End-to-end server latency

The response `timings` object is the engine report (unified `questions` shape);
there is no separate engine-side tool, so engine cost and server cost are the
same phases:

```sh
curl -s http://localhost:8096/v1/decision -H 'Content-Type: application/json' -d @request.json \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["timings"])'
```

With `"diagnostics": true` the same response also reports `prefill_ms`,
`scoring_ms`, `suffix_tokens`, and `common_suffix_tokens` under `diagnostics`.

## 5. Accuracy and framing

The labeled corpus is `tests/decision-baseline/accuracy_corpus.json`. It has 16
letter (Jev) cases, each with an `expected` answer. It can
optionally be replaced by a Jev-distill corpus: point `LLAMA_DECISION_CORPUS` at
a `.jsonl` file (or a directory of them) whose rows carry
`id/kind/options/target/state/question`. Each line is one self-contained row, so
`shuf -n2000 file.jsonl` per file yields a valid random sample; the harness
derives the winner from `target` and scores Brier against the soft distribution.
`LLAMA_DECISION_CORPUS_MAX` caps the loaded cases (default 100; 0 = all).

The harness is one-shot stateless by default and uses a modest context
(`LLAMA_SERVER_TEST_CTX`, default 8192; 2048 fits the committed corpus but corpus
states can need more). The slot-session framing is opt-in with
`LLAMA_DECISION_SESSION_FRAMING=1`; it is not part of the default benchmark run.

```sh
LLAMA_SERVER_BIN=build/bin/llama-server \
LLAMA_SERVER_TEST_MODEL=<model.gguf> \
LLAMA_DECISION_ACCURACY_REPORT=tests/decision-baseline/accuracy_report.json \
python3 tools/server/tests/test_decision_accuracy.py

# same run over a sampled Jev-distill corpus
LLAMA_DECISION_CORPUS=/ai/data/decision/SargeDev/jev-distill-corpus-v3 \
LLAMA_DECISION_CORPUS_MAX=1000 \
python3 tools/server/tests/test_decision_accuracy.py
```

The harness is measurement only. It never asserts a minimum accuracy, so a weak
model cannot fail the suite; with no model or server binary it skips (exit 0).
It reports, per model:

- `winner_agreement`: fraction of cases whose argmax equals `expected`.
- `brier`: mean squared error of the distribution against the one-hot label.
- `ece_certainty` and `ece_confidence`: expected calibration error of the two
  concentration scores. These are calibration measurements, not gates.
- `stateless_local_confidence`: the same cases under `confidence_profile: "local"`,
  to show the profile changes only the reported number.
- `stateless_permutations2` and `permutations_gain`: winner-agreement and Brier
  delta of a 2-pass order de-bias run against the 1-pass default.
- `session`: the same evidence prefilled on a chat slot and answered through the
  session fork, plus `framing_winner` chosen by winner agreement, Brier as the
  tie-break. Confidence never chooses the framing.

Update the committed report only with a matching experiment:

```sh
LLAMA_DECISION_ACCURACY_REPORT=tests/decision-baseline/accuracy_report.json \
  python3 tools/server/tests/test_decision_accuracy.py
```

Each run upserts that model's block; other models are preserved.

## 6. Fork exactness (the byte oracle)

Fork correctness is never judged by matching the argmax. A branch that shares a
recurrent tail can agree on the winner while its probability vector is stale, so
the oracle compares state bytes.

```sh
LLAMA_DECISION_TEST_MODEL=<model.gguf> \
  ./build/bin/test-decision-engine "decision engine harness(\\..*fork.*)?"
```

The oracle decodes a prompt, forks a branch through `engine::fork_into`, decodes
one token, and asserts the branch state bytes equal a full `restore` fork of the
same parent. The control test prints the plain `seq_cp` characterization as
`[fork control] <lane>: ... N of M state bytes differ, max logit delta ...` and
does not assert on it. Dense attention is the byte-identical control; LFM2 is the
layout that breaks a plain `seq_cp`.

The layout matrix (fresh cache, prior sequences, 1-2 branches) is only
load-bearing where a partial fork can drift, so it runs in full only on
recurrent/hybrid models. A dense model runs a single fresh-layout smoke check
instead: `seq_cp` shares only attention cells, so the full matrix is a
tautology there and costs CI time without adding signal.

The ctest labels wrap the same filters:

```sh
ctest --test-dir build --output-on-failure -R "decision|fork|permut|calibration"
```

## 7. Readout reproducibility baselines

The frozen readout baseline is the deterministic core of the committed corpus:
per-question probabilities, winners, `confidence = 1 - H/log K`, `certainty =
max p`, and label-pool size. The timing block is recorded for context
and is excluded from the byte diff, because it moves every run.

```sh
# record (GPU needs LLAMA_DECISION_TEST_MODEL; CPU uses the generated model)
./build/bin/test-decision-engine --record-readout gpu
./build/bin/test-decision-engine --record-readout cpu

# check: recompute and fail on any deterministic byte change
./build/bin/test-decision-engine --check-readout gpu
./build/bin/test-decision-engine --check-readout cpu
```

The suite gate chooses whichever frozen baseline matches the model available in
the lane. A mismatch prints the first differing line.

## 8. Fairness under chat

`tests/decision-baseline/fairness.json` holds a pre-registered bound fixed
before measurement. `test_decision_admission.py` enforces it:

- `/slots` must answer within `slots_during_decision_ms` (250 ms) while a
  multi-second decision is in flight. This is the cooperative-yield guarantee.
- A chat request fired during a decision must finish within
  `decision_ms + chat_latency_factor * warm_decision_ms + chat_latency_margin_ms`
  (factor 2.0, margin 500 ms).

```sh
LLAMA_SERVER_BIN=build/bin/llama-server \
LLAMA_SERVER_TEST_NGL=99 \
LLAMA_SERVER_TEST_MODEL=<model.gguf> \
python3 tools/server/tests/test_decision_admission.py
```

Changing the bound is a code change: update `fairness.json` and re-measure.
`test_decision_envelope.py` carries the session-fork and permutations-profile
checks and is run the same way.

## 9. Refreshing a frozen artifact

Do this only together with the code change that moved it. Never rewrite a
baseline to make a test pass without understanding the drift.

| Artifact | Refresh command | Scope |
|---|---|---|
| `calibration.json` rows | `test-decision-engine --write-calibration-rows` | host only, no GGUF |
| `calibration.json` full | `LLAMA_DECISION_TEST_MODEL=... test-decision-engine --write-calibration` | rows + model measurements |
| `baseline.json` CPU oracle | `test-decision-engine --write-cpu-oracle` | needs the generated model |
| `readout_gpu_baseline.json` | `LLAMA_DECISION_TEST_MODEL=... test-decision-engine --record-readout gpu` | GPU lane |
| `readout_cpu_baseline.json` | `test-decision-engine --record-readout cpu` | CPU lane |
| `decision_letter.golden.json` | `LLAMA_DECISION_TEST_MODEL=... test-decision-engine --write-decision-golden` | letter readout |
| `accuracy_report.json` | `LLAMA_DECISION_ACCURACY_REPORT=... test_decision_accuracy.py` | accuracy |
| `fairness.json` bound | manual, with a matching code change | fairness |

The rows-only writer preserves the existing model measurements; the full writer
re-measures them and uses `n_ctx=8192, gpu_layers=0`. Every calibration row
carries `production_gate: false`; a confidence row may not gate admission,
caching, routing, or persistence.

## 10. Reference model matrix

Run the speed, accuracy, and session gates on the six-model matrix:

| model | attention | role |
|---|---|---|
| lfm2.5-350m | hybrid SSM | fork exactness case, small |
| lfm2.5-2.6b | hybrid SSM | fork exactness case, main |
| qwen3.5-2b | hybrid SSM | fork exactness case |
| qwen3.5-9b | hybrid SSM | scale case |
| gemma-4-e4b | dense | byte-identical control |
| gemma-4-12b | dense | byte-identical control |

Point `LLAMA_DECISION_TEST_MODEL` at each model and rerun the relevant command.
Dense gemma models must stay byte-identical to the previous answers; LFM2 is the
layout that breaks plain `seq_cp` and is where the partial hybrid fork earns its
place.

## 11. Pitfalls

- Do not compare probabilities bit-for-bit across backends. Producer numerics
  reorder reductions on CUDA, Metal, and ROCm, and flash attention changes them
  further. Compare with a tolerance; keep bit equality for the state-byte fork
  oracle and the deterministic readout core.
- Do not compare a cold run against a warm run. Use `cache_hit` and the cold
  prefill cost from the response `timings`.
- Do not present `confidence`/`certainty` as accuracy. They measure
  concentration, not correctness, and they never gate anything.
- `usage.output_tokens` is always 0. A non-zero value is a bug, not a benchmark
  result.
- A changed `template_hash`, quantization, or any frozen backend flag
  invalidates every prior parity, calibration, and accuracy claim. Re-measure.
- The accuracy harness is not a gate. Read `winner_agreement`, `Brier`, and `ECE`
  as measurements; do not turn one into a pass/fail threshold without a
  pre-registered bound equivalent to `fairness.json`.

## 12. Quick reference

```sh
# decision suites
ctest --test-dir build --output-on-failure -R "decision|fork|permut|calibration"

# server accuracy
LLAMA_SERVER_BIN=build/bin/llama-server LLAMA_SERVER_TEST_MODEL=<model.gguf> \
  python3 tools/server/tests/test_decision_accuracy.py

# server envelope + session + fairness
LLAMA_SERVER_BIN=build/bin/llama-server LLAMA_SERVER_TEST_NGL=99 LLAMA_SERVER_TEST_MODEL=<model.gguf> \
  python3 tools/server/tests/test_decision_envelope.py
LLAMA_SERVER_BIN=build/bin/llama-server LLAMA_SERVER_TEST_NGL=99 LLAMA_SERVER_TEST_MODEL=<model.gguf> \
  python3 tools/server/tests/test_decision_admission.py

# frozen readout
LLAMA_DECISION_TEST_MODEL=<model.gguf> ./build/bin/test-decision-engine --record-readout gpu
LLAMA_DECISION_TEST_MODEL=<model.gguf> ./build/bin/test-decision-engine --check-readout gpu
```
