# Benchmarking and validating the decision engine

This is the practical guide for measuring and validating `POST /v1/decision`, the
decision engine behind it, and the live-session substrate it reads. It covers
what to measure, how to reproduce it, which frozen artifacts own the results,
how to validate isolation from `/v1/chat`, and how not to fool yourself with a
number.

The rule that governs everything here: a producer concentration score
(`confidence`, `certainty`) is never a benchmark gate and never a benchmark
result. It is reported telemetry. All benchmarks below measure either speed
(task value), outcome correctness (task value), or reproducibility (which is its
own axis).

```
speed           -> prefill_ms, scoring_ms, total_ms, cache_hit, rounds, rows
task value      -> winner agreement, Brier, ECE, fork byte equality, framing choice
reproducibility -> frozen readout baselines, calibration ledger, frozen backend flags
coexistence     -> /slots and chat latency bounds, chat byte-stability, source invariance
```

## 0. The surface at a glance

| Layer | Entry point | Validates | Artifact |
|---|---|---|---|
| Engine unit/oracle | `build/bin/test-decision-engine` | schema, assemble, fork byte exactness, session registry, device state | `[fork control]` log, `tests/fixtures/decision/*` |
| ctest | `ctest -L decision` etc. | the same binary under the CI labels | ctest labels |
| Server accuracy | `tools/server/tests/test_decision_accuracy.py` | labeled-corpus winner/Brier/ECE, framing | `tests/decision-baseline/accuracy_report.json` |
| Server envelope | `tools/server/tests/test_decision_envelope.py` | wire shape, errors, sessions, handles, adapters, snapshots | response bodies |
| Server admission | `tools/server/tests/test_decision_admission.py` | 413/429/529, capacity, chat/decision KV integrity, fairness | `tests/decision-baseline/fairness.json` |
| Server coexistence | `tools/server/tests/test_decision_session_concurrency.py` | session decision vs a generating chat, in-flight refusal, concurrent decisions | measured latencies printed |
| Temperature driver | `tools/parallel-decision/tests/test_temperature.py` | temperature/confidence/provenance subset | ctest `test-decision-temperature` |
| Frozen state | `tests/decision-baseline/*.json` | reproducibility gates | readout, session, calibration, baseline |

The C++ harness compiles the engine sources directly (`tests/CMakeLists.txt:262-301`),
so it does not need `LLAMA_BUILD_SERVER`. The server scripts drive a real
`llama-server`; they are standalone drivers (each has a `main()` and no
`test_*` functions), so `pytest`/`tests.sh` do not run them. Invoke them
directly, or add them to CI explicitly.

## 1. Build and environment

### CPU build (host-runnable suites)

```sh
cmake -B build -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=ON
cmake --build build -j"$(nproc)"
```

### ROCm/HIP build (the only lane for speed, GPU fork exactness, and the model matrix)

`build-amd.sh` is the canonical ROCm build (HIP, `gfx1100` by default, backend
loaded dynamically):

```sh
GPU_TARGETS=gfx1100 BUILD_DIR=build-rocm ./build-amd.sh
```

Run anything against it with the backend directory on the library path; the
server and test binaries dlopen `libggml-hip.so` from there:

```sh
export LD_LIBRARY_PATH="$PWD/build-rocm/bin"
```

A build without a GPU backend still passes, but the GPU lanes skip. Confirm the
GPU is live by watching for `ROCm0` in the context logs when a model is loaded.

### Environment variables

| Variable | Used by | Meaning |
|---|---|---|
| `LLAMA_DECISION_TEST_MODEL` | `test-decision-engine` | GGUF for the GPU lanes and the model-bound baselines |
| `LLAMA_DECISION_TEST_BIN` | `test_temperature.py` | exact path to the built test binary (injected by ctest) |
| `LLAMA_SERVER_BIN` | all server scripts | path to `llama-server` (default `build/bin/llama-server`) |
| `LLAMA_SERVER_TEST_MODEL` | all server scripts | model under test, tried first; the bundled small models are fallbacks |
| `LLAMA_SERVER_TEST_NGL` | server scripts | GPU layers (default `99`) |
| `LLAMA_SERVER_TEST_CTX` | accuracy | context size (default `8192`) |
| `LLAMA_DECISION_CORPUS` / `_MAX` | accuracy | override corpus dir/file and the case cap (default 100; 0 = all) |
| `LLAMA_DECISION_ACCURACY_GATE` | accuracy | set `1` to fail the run when the holdout is below the pre-registered floors |
| `LLAMA_DECISION_ACCURACY_SPLIT` | accuracy | narrow the run to `calibration` or `holdout` |
| `LLAMA_DECISION_ACCURACY_REPORT` | accuracy | where to upsert the report block |
| `LLAMA_DECISION_SESSION_FRAMING` | accuracy | set `1` to add the slot-session framing |
| `LLAMA_DECISION_MAX_BODY` / `_MAX_QUEUE` | admission | override the tiny admission caps |
| `LLAMA_SERVER_TEST_ALLOW_SKIP` | admission | allow a clean skip when no model supports labels |

`LLAMA_FLASH_ATTN_TYPE_DISABLED` / `_ENABLED` are C enum values in `llama.h`,
not environment variables; the harness sets flash attention off by default for
bit-reproducibility.

## 2. Freeze the environment before you measure

A decision number is comparable to another decision number only when the model,
quantization, prompt template, and backend flags match. The frozen readout and
calibration artifacts record the exact flags they were measured under
(`tests/decision-baseline/readout_*_baseline.json`, `calibration.json`).

| Flag | Value used by the recorded baselines | Why it must be fixed |
|---|---|---|
| model + quantization | recorded as the last path components | weights move every probability |
| `template_hash` | `make_prefix_tag(...)` | the framed prompt moves the label boundary |
| `kv_unified` | `true` | decisions fork on a unified cache |
| `swa_full` | `false` | sliding-window layout changes retained cells |
| `n_ctx` | 8192 in the server suites; 2048/512 in bounded sweeps | changes whether a request fits |
| `n_batch` / `n_ubatch` | `512` / `512` | batch split changes rounding and pass count |
| `n_seq_max` | `10` (`n_parallel + n_seq_decision`) | number of live branches |
| `gpu_layers` | `-1` (server), 0 in some calibration runs | CPU vs GPU changes producer numerics |
| `flash_attn` | `false` in the engine oracle; the production preset records it `on` | ROCm FA is not bit-reproducible across runs |
| `cache_type_k` / `cache_type_v` | production preset records `q8_0` | KV quantization changes producer numerics |

If any of these changes, prior speed and accuracy numbers are stale. Re-measure
before you compare.

### Lanes

- GPU lane: the only lane for speed, fork exactness on a real layout, the model
  matrix, and the session backends. Set `LLAMA_DECISION_TEST_MODEL` and run a
  HIP build.
- CPU lane: host-runnable tests and the frozen CPU readout/session oracles. It
  loads the generated `qwen35-dense.gguf` dummy model from the `generate-models`
  fixture (a hybrid arch, so the recurrent paths are exercised), or falls back to
  `LLAMA_DECISION_TEST_MODEL`.

## 3. Unifying measurements across shapes

`test-decision-engine` is one binary with three subcommands and a filter:

```sh
# run the whole harness
./build/bin/test-decision-engine

# run only sub-tests whose full name matches a regex (the name is
# "decision engine harness.<subtest>")
./build/bin/test-decision-engine "decision engine harness(\..*fork.*)?"

# freeze/refresh writers (see section 10)
./build/bin/test-decision-engine --write-golden
./build/bin/test-decision-engine --write-cpu-oracle
./build/bin/test-decision-engine --write-decision-golden
./build/bin/test-decision-engine --write-calibration
./build/bin/test-decision-engine --write-calibration-rows
./build/bin/test-decision-engine --record-readout   cpu|gpu
./build/bin/test-decision-engine --check-readout    cpu|gpu
```

The GPU lanes skip with a reason when no GPU backend is present or when
`LLAMA_DECISION_TEST_MODEL` is unset, so the CPU lane never fails open.

## 4. Decision latency

The unified shape reports its own phases on every response `timings` object
(`prefill_ms`, `scoring_ms`, `total_ms`, `rounds`, `rows`, `per_decision_ms`).
Warm and cold are separated by `cache_hit` (true only on the warm call) and the
cold prefill cost in the first run. `rows` is the number of scored branch rows
and `rounds` the number of decode waves; the letter readout always scores the
exact tree, so `rows` is the trie's divergence-node count.

```sh
curl -s http://localhost:8096/v1/decision -H 'Content-Type: application/json' -d @request.json \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["timings"])'
```

With `"diagnostics": true` the same response also reports `prefill_ms`,
`scoring_ms`, `suffix_tokens`, and `common_suffix_tokens` under `diagnostics`.
Compare the same request with and without `"cache_prompt": true` to separate the
cold prefill cost from the warm path. A cold run on one branch against a warm
run on another is the classic false-regression trap.

## 5. Accuracy and framing

The labeled corpus is `tests/decision-baseline/accuracy_corpus.json` (letter
cases with an `expected` answer, and with a `calibration` / `holdout` `split`
fixed by a hash of the case id). The split is a property of the fixture, not of
the run, so adding cases never moves an existing case between splits and any
reader reproduces the same assignment. It can optionally be replaced by a
Jev-distill corpus: point `LLAMA_DECISION_CORPUS` at a `.jsonl` file (or a
directory of them) whose rows carry `id/kind/options/target/state/question`.
Each line is one self-contained row, so `shuf -n2000 file.jsonl` per file yields
a valid random sample; the harness derives the winner from `target` and scores
Brier against the soft distribution. `LLAMA_DECISION_CORPUS_MAX` caps the loaded
cases (default 100; 0 = all).

The harness is one-shot stateless by default and uses a modest context
(`LLAMA_SERVER_TEST_CTX`, default 8192). The slot-session framing is opt-in with
`LLAMA_DECISION_SESSION_FRAMING=1`; it is not part of the default run.

```sh
LLAMA_SERVER_BIN=build-rocm/bin/llama-server \
LLAMA_SERVER_TEST_MODEL=<model.gguf> \
LLAMA_DECISION_ACCURACY_REPORT=tests/decision-baseline/accuracy_report.json \
python3 tools/server/tests/test_decision_accuracy.py

# same run over a sampled Jev-distill corpus
LLAMA_DECISION_CORPUS=/ai/data/decision/SargeDev/jev-distill-corpus-v3 \
LLAMA_DECISION_CORPUS_MAX=1000 \
python3 tools/server/tests/test_decision_accuracy.py
```

The harness has two modes. The default run is measurement only: it reports
numbers and gates nothing, so a weak model cannot fail the suite; with no model
or server binary it skips (exit 0). With `LLAMA_DECISION_ACCURACY_GATE=1` it
additionally evaluates the **holdout** split against `GATE_FLOORS`, the
pre-registered per-model winner-agreement floors in the harness, and fails the
run when a model is below its floor. The floors are frozen from the letter
readout's own measured holdout agreement on the committed corpus, two
percentage points below each measurement; Brier and ECE are reported and are
never part of a floor. `LLAMA_DECISION_ACCURACY_SPLIT` narrows a run to one
split (`calibration` or `holdout`) so a floor can be derived on the calibration
cases and then checked on the holdout.

It reports, per model:

- `winner_agreement`: fraction of cases whose argmax equals `expected`.
- `brier`: mean squared error of the distribution against the one-hot label.
- `ece_certainty` and `ece_confidence`: expected calibration error of the two
  concentration scores. These are calibration measurements, not gates.
- `stateless_local_confidence`: the same cases under `confidence_profile: "local"`.
- `stateless_permutations2` and `permutations_gain`: winner-agreement and Brier
  delta of a 2-pass order de-bias run against the 1-pass default.
- `session`: the same evidence prefilled on a chat slot and answered through the
  session fork, plus `framing_winner` chosen by winner agreement, Brier as the
  tie-break. Confidence never chooses the framing.
- `readout.letter` and `readout.delta`: the per-case raw results of the readout
  that produced the numbers, and the paired readout comparison. A server that
  answered under a different readout than the harness expects is refused, not
  measured.

Update the committed report only with a matching experiment; each run upserts
that model's block and preserves the others.

## 6. Fork exactness (the byte oracle)

Fork correctness is never judged by matching the argmax. A branch that shares a
recurrent tail can agree on the winner while its probability vector is stale, so
the oracle compares state bytes.

```sh
LLAMA_DECISION_TEST_MODEL=<model.gguf> \
  ./build/bin/test-decision-engine "decision engine harness(\..*fork.*)?"
```

The oracle decodes a prompt, forks a branch through `engine::fork_into`, decodes
one token, and asserts the branch state bytes equal a full `restore` fork of the
same parent. The control test prints the plain `seq_cp` characterization as
`[fork control] <lane>: ... N of M state bytes differ, max logit delta ...` and
does not assert on it. Dense attention is the byte-identical control; LFM2 is the
layout that breaks a plain `seq_cp` (the hybrid partial fork is what restores
exactness there).

The layout matrix (fresh cache, prior sequences, 1-2 branches) is only
load-bearing where a partial fork can drift, so it runs in full only on
recurrent/hybrid models. A dense model runs a single fresh-layout smoke check
plus the full `copy/restore/hybrid` strategy comparison on GPU. Verified on GPU
(ROCm gfx1100): `copy` passes on dense, skips on recurrent; nested partial forks
run only on recurrent/hybrid.

## 7. Session substrate validation (C++)

The engine's fork path has its own oracle, independent of the server:

| Test | What it asserts |
|---|---|
| `test_session_fork` | a session fork of a prefilled context matches a full re-prefill |
| `test_pool_seq_lifecycle` | engine pool sequences are released after a decision |

The server's session path (the sidecar token snapshot: capture, eager store,
turn advance/release, TTL/budget eviction) is covered by the HTTP suites
(`tools/server/tests/test_decision_session_concurrency.py` and
`test_decision_envelope.py`).

Source invariance is structural, not just asserted: `llama_memory_seq_rm` removes
only the caller's membership, `find_slot` never reuses a non-empty non-SWA cell,
and a recurrent partial load does `seq_rm(dst)` first. The oracle records the
source bytes before and after a fork and requires equality.

```sh
# engine fork/session behavior
LLAMA_DECISION_TEST_MODEL=<model.gguf> \
  ./build/bin/test-decision-engine "decision engine harness(\..*session.*)?"
```

## 8. Readout reproducibility baselines

The frozen readout baseline is the deterministic core of the committed corpus:
per-question probabilities, winners, `confidence = 1 - H/log K`, `certainty =
max p`, and label-pool size. Timing is recorded for context and excluded from
the byte diff.

```sh
# record (GPU needs LLAMA_DECISION_TEST_MODEL; CPU uses the generated model)
./build/bin/test-decision-engine --record-readout gpu
./build/bin/test-decision-engine --record-readout cpu

# check: recompute and fail on any deterministic byte change
./build/bin/test-decision-engine --check-readout gpu
./build/bin/test-decision-engine --check-readout cpu
```

The in-suite gate (`test_readout_baseline`) chooses whichever frozen baseline
matches the model available in the lane; a mismatch prints the first differing
line.

## 9. Coexistence with chat (what is asserted)

The decision engine and chat share one context, one scheduler thread, and one
unified KV pool. The guarantees are isolation (no corruption/eviction), not
parallelism. The suites validate the guarantee at two levels.

Engine/server correctness:
- `test_decision_admission.py` `run_chat_decision_integrity`: a decision fired
  mid-chat must leave the chat answer byte-identical to a quiet run.
- `test_decision_admission.py` `run_recurrent_kv_integrity`: chat, decision,
  identical chat byte-identical on recurrent/hybrid models.
- `test_decision_session_concurrency.py`: session decision while another slot
  generates (answer stable), in-flight same-slot decision refused, session
  survives an intervening chat (`cache_idle_slots`), concurrent session
  decisions agree.

Fairness bound (`tests/decision-baseline/fairness.json`), fixed before
measurement and enforced by `test_decision_admission.py`:
- `/slots` must answer within `slots_during_decision_ms` (250 ms) while a
  multi-second decision is in flight (the cooperative-yield guarantee).
- A chat fired during a decision must finish within
  `decision_ms + factor * warm_decision_ms + margin` (factor 2.0, margin 500 ms).

```sh
LLAMA_SERVER_BIN=build-rocm/bin/llama-server \
LLAMA_SERVER_TEST_NGL=99 \
LLAMA_SERVER_TEST_MODEL=<model.gguf> \
python3 tools/server/tests/test_decision_admission.py

LLAMA_SERVER_BIN=build-rocm/bin/llama-server \
LLAMA_SERVER_TEST_MODEL=<model.gguf> \
python3 tools/server/tests/test_decision_session_concurrency.py
```

Changing the bound is a code change: update `fairness.json` and re-measure.

### What coexistence costs (measured, not asserted)

A decision does not run in parallel with chat: the scheduler processes the
decision before the next `update_slots`, and chat decode is declined during the
yield. It is serialized compute interleaved at a decode-chunk boundary. Measured
on ROCm gfx1100 (warm chat + concurrent heavy session decision):

| model | quiet chat | chat + concurrent decision | quiet decision | concurrent decision |
|---|---|---|---|---|
| lfm2.5-350m | 310 ms | 323 ms | 13 ms | 14 ms |
| qwen3.5-2b | 924 ms | 941 ms | 44 ms | 46 ms |
| gemma-4-e4b | 1514 ms | 1583 ms | 52 ms | 58 ms |

The chat absorbs roughly the decision's duration; the decision itself returns at
its own cost. The latency ratio is machine dependent and is reported by the
coexistence test, not asserted.

## 10. Refreshing a frozen artifact

Do this only together with the code change that moved it. Never rewrite a
baseline to make a test pass without understanding the drift.

| Artifact | Refresh command | Scope |
|---|---|---|
| `calibration.json` rows | `test-decision-engine --write-calibration-rows` | host only |
| `calibration.json` full | `LLAMA_DECISION_TEST_MODEL=... test-decision-engine --write-calibration` | rows + model measurements |
| `baseline.json` CPU oracle | `test-decision-engine --write-cpu-oracle` | needs the generated model |
| `readout_gpu_baseline.json` | `LLAMA_DECISION_TEST_MODEL=... test-decision-engine --record-readout gpu` | GPU lane |
| `readout_cpu_baseline.json` | `test-decision-engine --record-readout cpu` | CPU lane |
| `decision_letter.golden.json` | `LLAMA_DECISION_TEST_MODEL=... test-decision-engine --write-decision-golden` | letter readout |
| `decision_basic.golden.json` | `test-decision-engine --write-golden` | shape/assemble |
| `accuracy_report.json` | `LLAMA_DECISION_ACCURACY_REPORT=... test_decision_accuracy.py` | accuracy |
| `fairness.json` bound | manual, with a matching code change | fairness |

The rows-only calibration writer preserves existing model measurements; the full
writer re-measures them. Every calibration row carries `production_gate: false`;
a confidence row may not gate admission, caching, routing, or persistence.

## 11. Reference model matrix

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
place. On GPU, run the fork and session subsets for each model before trusting a
change; the recorded baselines are single-machine (ROCm, FA off) and are not a
cross-backend guarantee.

## 12. ctest registration

```sh
# everything decision-related
ctest --test-dir build --output-on-failure -L "decision|calibration|fork|permut|temperature"

# by name
ctest --test-dir build --output-on-failure -R "decision|fork|permut|calibration|temperature"
```

| ctest name | Command |
|---|---|
| `test-decision-engine` | the full harness, labels `decision;main` |
| `test-decision-calibration` | `"decision engine harness(\..*calibration.*)?"` |
| `test-decision-fork` | `"decision engine harness(\..*fork.*)?"` |
| `test-decision-permut` | `"decision engine harness(\..*permut.*)?"` |
| `test-decision-temperature` | `test_temperature.py`, needs Python 3 and `LLAMA_DECISION_TEST_BIN` |

`test-decision-engine` requires the `generate-models` fixture (the generated CPU
models) and always runs in the standard CPU CI lane. The model-bound GPU lanes
inside it stay skipped unless a GPU backend and `LLAMA_DECISION_TEST_MODEL` are
present.

## 13. Pitfalls

- Do not compare probabilities bit-for-bit across backends. Producer numerics
  reorder reductions on CUDA, Metal, and ROCm, and flash attention changes them
  further. Compare with a tolerance; keep bit equality for the state-byte fork
  oracle and the deterministic readout/session cores.
- Do not compare a cold run against a warm run. Use `cache_hit` and the cold
  prefill cost from the response `timings`.
- Do not compare the first server decode against later runs. On GPU the first
  batch can pick a different shape and flip a near-tie; warm the path once before
  taking a baseline. `test_decision_session_concurrency.py` does this.
- Do not use a tiny base model to judge framing agreement. On `stories15M` the
  stateless and session framings flip a near-tie winner; the envelope's
  exact-winner session assertion is meaningful on a real model only.
- Implicit `id_slot` sessions are fragile: any other chat task's
  `cache_idle_slots` clears the idle slot, after which a decision re-captures if
  the state is resident or refuses. Use a first-class `session_id` with
  `capture_on_turn_complete`, or the `clone`/`file` backend, when a session must
  outlive an intervening chat.
- A session captured while a LoRA adapter was active is scored by a decision
  decode that runs on the base model. The registry refuses a scope *change*, not
  a base decode of an adapted capture; validate adapter + session usage
  explicitly if you rely on it.
- Do not present `confidence`/`certainty` as accuracy. They measure
  concentration, not correctness, and they never gate anything.
- `usage.output_tokens` is always 0. A non-zero value is a bug, not a benchmark
  result.
- A changed `template_hash`, quantization, or any frozen backend flag
  invalidates every prior parity, calibration, and accuracy claim. Re-measure.
- The accuracy harness measures by default and gates only under
  `LLAMA_DECISION_ACCURACY_GATE=1`, against floors pre-registered in
  `GATE_FLOORS`. A floor for a new model or a new corpus is derived on the
  calibration split first; never set one from a holdout run, and never read
  `winner_agreement`, `Brier`, or `ECE` from a producer `confidence`.
- The server scripts are standalone; `pytest`/`tests.sh` do not execute them.
  Run them directly, or wire them into CI.

## 14. Quick reference

```sh
# --- CPU ---
cmake -B build -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=ON && cmake --build build -j
ctest --test-dir build --output-on-failure -L "decision|calibration|fork|permut|temperature"

# --- GPU (ROCm) ---
GPU_TARGETS=gfx1100 BUILD_DIR=build-rocm ./build-amd.sh
export LD_LIBRARY_PATH="$PWD/build-rocm/bin"
MODEL=/path/to/model.gguf

# engine oracles (GPU lanes on)
LLAMA_DECISION_TEST_MODEL=$MODEL ./build-rocm/bin/test-decision-engine "decision engine harness(\..*fork.*)?"
LLAMA_DECISION_TEST_MODEL=$MODEL ./build-rocm/bin/test-decision-engine "decision engine harness(\..*session.*)?"

# frozen readout baselines
LLAMA_DECISION_TEST_MODEL=$MODEL ./build-rocm/bin/test-decision-engine --check-readout gpu

# --- server suites ---
export LLAMA_SERVER_BIN=$PWD/build-rocm/bin/llama-server
export LLAMA_SERVER_TEST_MODEL=$MODEL
python3 tools/server/tests/test_decision_envelope.py
python3 tools/server/tests/test_decision_admission.py
python3 tools/server/tests/test_decision_session_concurrency.py
python3 tools/server/tests/test_decision_accuracy.py
```
