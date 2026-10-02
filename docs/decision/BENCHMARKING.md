# Benchmarking and validating the decision engine

This is the practical guide for measuring and validating `POST /v1/decision`,
the engine behind it, and the live-session substrate it reads. It covers what to
measure, how to reproduce it, which frozen artifacts own the result, how to
validate isolation from `/v1/chat`, and how not to fool yourself with a number.

The contract being validated is `API.md`. This document is not normative about
behaviour; it is normative about *method*.

The rule that governs everything here: a producer concentration score
(`confidence`, `certainty`) is never a benchmark gate and never a benchmark
result. It is reported telemetry. Every benchmark below measures one of three
things, and they are kept apart on purpose.

```
accuracy       -> winner agreement, Brier, ECE                 (against ground truth)
speed          -> prefill_ms, scoring_ms, rounds, rows, cache_hit, warm_hit
stability      -> chat byte-stability, slot-release, lifecycle, admission, memory
task value     -> fork byte equality, capacity preflight, warm-tier admission
```

Reproducibility is its own axis and is owned by the frozen artifacts
(`tests/decision-baseline/*.json`): a dense model is bit-reproducible, a
recurrent or hybrid one is not, and the rule for comparing a repeat is in
`API.md` section 3.1.

## 0. The surface at a glance

| Layer | Entry point | Validates | Artifact |
|---|---|---|---|
| Engine harness | `build/bin/test-decision-engine` | envelope, both front-ends, fork byte exactness, capacity, warm tier, label pool, golden responses | `tests/fixtures/decision/*.golden.json` |
| ctest | `ctest -R "test-decision"` | the same binary, filtered, under CI | ctest labels |
| Accuracy | `tools/server/tests/test_decision_accuracy.py` | labeled-corpus winner/Brier/ECE, confidence profiles, permutation gain, framing | `tests/decision-baseline/accuracy_report.json` |
| Envelope and errors | `tools/server/tests/test_decision_envelope.py` | wire shape, the full status set, sessions, handles, adapters, snapshots | response bodies |
| Jev conformance | `tools/server/tests/test_decision_systemone.py` | `/v1/systemone` against the transcribed Jev spec, and route parity | response bodies |
| Generic surface | `tools/server/tests/test_decision_generic.py` | the `schema` shape: wide fields, typed grids, aggregates, knobs | response bodies |
| Admission and isolation | `tools/server/tests/test_decision_admission.py` | 413/429/529/422/503, capacity sweep, chat byte-integrity, sidecar no-stall | `tests/decision-baseline/fairness.json` |
| Sessions and lifecycle | `tools/server/tests/test_decision_session_concurrency.py` | session decisions vs generating chat, TTL reaper, byte-budget eviction, identity, warm tier | printed counters and statuses |
| Agentic mixed load | `tools/server/tests/test_decision_agentic.py` | 4 chat agents + 4 stateless + 1 session + 2 churn workers on one GPU | printed gaps |
| Temperature | `tools/parallel-decision/tests/test_temperature.py` | temperature/confidence/provenance subset | ctest `test-decision-temperature` |
| Policy | `tests/test-common-instances.cpp` | the session-store eviction policy over a POD view, in C++ | assertions |

The C++ harness compiles the engine sources directly
(`tests/CMakeLists.txt`, the `test-decision-engine` block), so it does not need
`LLAMA_BUILD_SERVER`. The server scripts drive a real `llama-server`; they are
standalone drivers, so `pytest`/`tests.sh` do not run them. Invoke them
directly, or wire them into CI explicitly.

**`ctest -L decision` is not the decision gate.** It selects only 2 tests
(`test-generate-models`, `test-decision-engine`); the other four carry the
`calibration`, `fork`, `permut` and `temperature` labels. Use
`ctest -R "test-decision"`, which selects all of them.

## 1. Build and environment

### CPU build (host-runnable suites)

```sh
cmake -B build -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=ON
cmake --build build -j"$(nproc)"
```

A CPU build passes, and the model-bound GPU lanes inside the harness skip with a
reason rather than failing open. It is the right lane for the envelope, the
golden responses, the fork exactness controls and the CPU reproducibility
baseline.

### ROCm/HIP build (the only lane for speed and the model matrix)

`build-amd.sh` is the canonical ROCm build (HIP, `gfx1100` by default, backend
loaded dynamically):

```sh
GPU_TARGETS=gfx1100 BUILD_DIR=build-rocm ./build-amd.sh
export LD_LIBRARY_PATH="$PWD/build-rocm/bin"
```

Speed, GPU fork exactness, the accuracy matrix and the agentic harness all need
this lane. Confirm the GPU is live by watching for `ROCm0` in the context logs
when a model is loaded.

### Environment variables

| Variable | Used by | Meaning |
|---|---|---|
| `LLAMA_DECISION_TEST_MODEL` | `test-decision-engine` | GGUF for the GPU lanes and the model-bound baselines |
| `LLAMA_DECISION_TEST_BIN` | `test_temperature.py` | exact path to the built test binary (injected by ctest) |
| `LLAMA_SERVER_BIN` | all server scripts | path to `llama-server` |
| `LLAMA_SERVER_TEST_MODEL` | all server scripts | model under test; the bundled small models are fallbacks |
| `LLAMA_SERVER_TEST_NGL` | server scripts | GPU layers (default `99`) |
| `LLAMA_SERVER_TEST_CTX` | accuracy | context size (default `8192`) |
| `LLAMA_DECISION_CORPUS` / `_MAX` | accuracy | override corpus dir/file and the case cap (default 100; 0 = all) |
| `LLAMA_DECISION_ACCURACY_GATE` | accuracy | set `1` to fail the run when a holdout is below its floor |
| `LLAMA_DECISION_ACCURACY_SPLIT` | accuracy | narrow the run to `calibration` or `holdout` |
| `LLAMA_DECISION_ACCURACY_REPORT` | accuracy | where to upsert the report block |
| `LLAMA_DECISION_SESSION_FRAMING` | accuracy | set `1` to add the slot-session framing |
| `LLAMA_DECISION_MAX_BODY` / `_MAX_QUEUE` | admission | override the admission caps |
| `LLAMA_DECISION_POOL_CAP` | any | clamp the realized answer-label pool; a `choice` wider than the clamp is a 422 |
| `LLAMA_DECISION_FORK` | any | force `copy`/`restore`/`hybrid`/`auto`; `copy` on a recurrent model is a 400 |
| `LLAMA_SERVER_TEST_ALLOW_SKIP` | admission | turn "no candidate model supports letter labels" from a failure into a clean skip |

**The server scripts default `LLAMA_SERVER_BIN` to `build/bin/llama-server`.**
If that directory holds a stale binary you will measure the wrong server and
report failures that do not exist in the tree. Always export it explicitly:

```sh
export LLAMA_SERVER_BIN=$PWD/build-rocm/bin/llama-server
```

`LLAMA_FLASH_ATTN_TYPE_DISABLED` / `_ENABLED` are C enum values in `llama.h`,
not environment variables; the harness sets flash attention off by default where
bit-reproducibility is required.

## 2. Freeze the environment before you measure

A decision number is comparable to another decision number only when the model,
quantization, prompt template, and backend flags match. The frozen readout and
calibration artifacts record the flags they were measured under
(`tests/decision-baseline/readout_*_baseline.json`, `calibration.json`).

| Flag | Value used by the recorded baselines | Why it must be fixed |
|---|---|---|
| model + quantization | recorded as the last path components | weights move every probability |
| `template_hash` | `make_prefix_tag(...)` | the framed prompt moves the label boundary |
| `kv_unified` | `true` on the decision executor | decisions fork on a unified cache |
| `swa_full` | `false` | sliding-window layout changes retained cells |
| `n_ctx` | 8192 in the server suites | changes whether a request fits |
| `n_batch` / `n_ubatch` | `512` / `512` | batch split changes rounding and pass count |
| `n_seq_decision` | the `--decision-seqs` value | number of live branches the plan may use |
| `gpu_layers` | 99 in the server suites | CPU vs GPU changes producer numerics |
| `flash_attn` | off where the baseline is bit-compared; the production preset records it on | ROCm FA is not bit-reproducible across runs |
| `cache_type_k` / `cache_type_v` | production preset records `q8_0` | KV quantization changes producer numerics |

If any of these changes, prior speed and accuracy numbers are stale. Re-measure
before you compare.

## 3. Selecting a subset of the harness

`test-decision-engine` is one binary with a name filter over its sub-cases. A
case's full name is `decision engine harness.<sub-case>`, so:

```sh
# the whole harness
./build/bin/test-decision-engine

# only cases whose name matches
./build/bin/test-decision-engine "decision engine harness(\..*fork.*)?"
./build/bin/test-decision-engine "decision engine harness(\..*calibration.*)?"

# the baseline writers (section 10)
./build/bin/test-decision-engine --write-golden
./build/bin/test-decision-engine --write-cpu-oracle
./build/bin/test-decision-engine --write-decision-golden
./build/bin/test-decision-engine --write-calibration
./build/bin/test-decision-engine --write-calibration-rows
./build/bin/test-decision-engine --record-readout   cpu|gpu
./build/bin/test-decision-engine --check-readout    cpu|gpu
```

The GPU lanes skip with a reason when no GPU backend is present or when
`LLAMA_DECISION_TEST_MODEL` is unset, so the CPU lane never fails open. A case
that cannot be evaluated on the available model **skips; it never xfails.**

## 4. Speed

The response carries a `timings` object (`prefill_ms`, `scoring_ms`, `total_ms`,
`rounds`, `rows`, `per_decision_ms`) and, under `diagnostics`, the prefix
counters that separate a warm call from a cold one.

```sh
curl -s http://localhost:8096/v1/decision -H 'Content-Type: application/json' \
  -H '...' -d '{"model":"m","diagnostics":true,"state":"...","questions":{...}}' \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["timings"], d["usage"])'
```

`timings` is **not** on the default envelope. It appears when the request sets
`diagnostics: true`, and on a session fork. Asking for it without either raises
`KeyError`; that is the harness being strict about the default envelope, not a
missing field.

What each number is:

* `prefill_ms` - everything that had to be decoded before scoring, including a
  cold prefill paid on a warm-tier miss and a resident-prefix restore. It is
  *not* the whole-request time.
* `scoring_ms` - the branch waves only.
* `rounds` - decode waves. `rows` - scored branch rows. For the letter readout
  `rows` is the trie's divergence-node count, so it grows with the option count
  and the option surface form, not with the question count.
* `per_decision_ms` - `total_ms` divided by the number of contexts.

Method rules:

* Separate cold from warm. `prefill_ms` on the first call versus later calls is
  the signal; `state_cache_hit` and `cached_tokens` name it, and a session
  response adds `warm_hit`. A cold run compared against a warm run is the
  classic false-regression trap.
* Warm the path before taking a baseline. On GPU the first batch can pick a
  different kernel shape and flip a near-tie.
* Measure the engine, not the HTTP round trip, when comparing fork strategies;
  the transport and the first-token cost are identical across variants.
* Any latency claim must name the model, the quantization, the flash-attention
  setting and the backend. A bare millisecond figure is not a result.

## 5. Accuracy

The labeled corpus is `tests/decision-baseline/accuracy_corpus.json` (letter
cases with an `expected` answer, and with a `calibration` / `holdout` `split`
fixed by a hash of the case id). The split is a property of the fixture, not of
the run, so adding cases never moves an existing case between splits and any
reader reproduces the same assignment. It can optionally be replaced by a
Jev-distill corpus: point `LLAMA_DECISION_CORPUS` at a `.jsonl` file (or a
directory of them) whose rows carry `id/kind/options/target/state/question`.
`LLAMA_DECISION_CORPUS_MAX` caps the loaded cases (default 100; 0 = all).

```sh
export LLAMA_SERVER_BIN=$PWD/build-rocm/bin/llama-server
LLAMA_SERVER_TEST_MODEL=<model.gguf> \
LLAMA_DECISION_ACCURACY_GATE=1 \
LLAMA_DECISION_ACCURACY_REPORT=tests/decision-baseline/accuracy_report.json \
python3 tools/server/tests/test_decision_accuracy.py
```

The harness has two modes. The default run is measurement only: it reports
numbers and gates nothing, so a weak model cannot fail the suite; with no model
or server binary it skips (exit 0). With `LLAMA_DECISION_ACCURACY_GATE=1` it
additionally evaluates the **holdout** split against `GATE_FLOORS`, the
pre-registered per-model winner-agreement floors:

| model | floor |
|---|---|
| `lfm2.5-350m-gguf/latest.gguf` | 0.35 |
| `qwen3.5-2b-gguf/ud-q5_k_xl.gguf` | 0.72 |
| `gemma-4-e4b-it-gguf/latest.gguf` | 0.73 |

`LLAMA_DECISION_ACCURACY_SPLIT` narrows a run to one split, which is how a floor
for a new model or corpus must be derived: on `calibration`, then checked on
`holdout`. Never set a floor from a holdout run.

Reported per model:

- `winner_agreement` - fraction of cases whose argmax equals `expected`.
- `brier` - mean squared error of the distribution against the one-hot label.
- `ece_certainty` and `ece_confidence` - expected calibration error of the two
  concentration scores. Measurements, never gates.
- `stateless_local_confidence` - the same cases under `confidence_profile: "local"`.
- `stateless_permutations2` and `permutations_gain` - winner agreement and Brier
  delta of a 2-pass order-debias run against the 1-pass default.
- `session` - the same evidence prefilled on a chat slot and answered through the
  session snapshot, plus `framing_winner` chosen by winner agreement with Brier
  as the tie-break. Concentration never chooses the framing.
- `latency` (mean / p50 / p95) and `warm` - reported alongside, never gated.
- `readout.letter` and `readout.delta` - the per-case raw results of the readout
  that produced the numbers, and the paired readout comparison.

**`readout.delta` is currently degenerate.** Only one readout exists in this
build, so the paired comparison is that readout against itself: exactly zero with
a zero-width interval, and the report says so in `delta.compared`. The rejected
alternative and the margins it was judged against survive in the frozen
`readout_decision` block, and `OBJECTIVE_MULTI_CONTEXT.md` section 9 records what
that measurement actually covered. Do not read `readout.delta` as a comparison.

Each run upserts only that model's block in the report and preserves the others.

## 6. Stability and coexistence with chat

Every decision runs on the internal `__decision__` sidecar executor: its own
context and its own scheduler thread. Chat contexts carry no decision sequences
and are not forced into the unified KV cache. So the guarantee is **isolation by
construction**, and what the suites assert is that it holds in practice and that
the two APIs do not perturb each other's answers.

That is a change from the removed in-context lane, where a decision ran on the
chat context and chat decode was declined for its duration. Two consequences for
methodology:

* A decision and a chat turn now **run concurrently on separate contexts** and
  contend for the same device. Device contention is the interesting variable,
  not serialization, so a latency comparison between quiet and concurrent runs
  measures GPU sharing.
* The "cooperative yield between decision waves" is no longer what keeps
  `/slots` responsive to a *chat* context; it is what keeps the sidecar's own
  scheduler responsive. The bound in `fairness.json`
  (`slots_during_decision_ms`, 250 ms) is still asserted and still meaningful,
  but the artifact's `rationale` text for the chat-latency bound still describes
  the shared-context lane. **Read the bound, not the rationale.**

What is asserted:

- `test_decision_admission.py` `run_chat_decision_integrity`: a decision fired
  mid-chat leaves the chat answer byte-identical to a quiet run.
- `test_decision_admission.py` `run_recurrent_kv_integrity`: chat, decision,
  identical chat byte-identical on recurrent/hybrid models.
- `test_decision_admission.py` `run_sidecar_no_stall`: the chat stream keeps its
  inter-token cadence while a decision runs.
- `test_decision_admission.py` `run_deadline_control`: with
  `--decision-timeout-ms` set low, a decision that stays in flight past the
  server's 1 s poll boundary answers 503 with `Retry-After` and no partial
  answer, and a control group proves a normal decision under the default
  timeout is never cancelled. The deadline is only checked at that poll
  boundary, so the suite **probes** a heavy request first and *skips* - never
  fails - on a model whose worst decision is too fast to reach it.
- `test_decision_session_concurrency.py`: a session decision while another slot
  generates leaves the chat answer unchanged, an in-flight same-slot decision is
  refused, a first-class session survives an intervening `cache_idle_slots`, and
  concurrent decisions on one session agree.
- `test_decision_agentic.py`: 4 streaming chat agents, 4 stateless decision
  workers, 1 session worker and 2 churn workers concurrently on one GPU; no
  unexpected 413/422/429/499/529, every chat agent answers, and each chat agent's
  max inter-token gap stays under the printed bound.

```sh
export LLAMA_SERVER_BIN=$PWD/build-rocm/bin/llama-server
export LLAMA_SERVER_TEST_NGL=99
export LLAMA_SERVER_TEST_MODEL=<model.gguf>
python3 tools/server/tests/test_decision_admission.py
python3 tools/server/tests/test_decision_session_concurrency.py
python3 tools/server/tests/test_decision_agentic.py
```

The latency ratio between quiet and concurrent runs is machine dependent and is
**reported, not asserted**; the correctness properties are asserted. Changing the
`fairness.json` bound is a code change and requires a re-measured gate.

### Memory

The sidecar holds non-resident token snapshots: no reserved sequence between
decisions, no arena, so there is nothing like `arena_used` to leak. Under mixed
chat plus decision load, RSS growth is normally the stock prompt cache
(`cache_ram_mib` default 8192, `cache_idle_slots` on). With `--cache-ram 0` the
same workload should be flat and stateless decisions should add zero growth.
Check that before suspecting a leak.

## 7. Fork exactness (the byte oracle)

Fork correctness is never judged by matching the argmax. A branch that shares a
recurrent tail can agree on the winner while its probability vector is stale, so
the oracle compares state bytes.

```sh
LLAMA_DECISION_TEST_MODEL=<model.gguf> \
  ./build/bin/test-decision-engine "decision engine harness(\..*fork.*)?"
```

The oracle decodes a prompt, forks a branch through the engine's fork path,
decodes one token, and asserts the branch state bytes equal a full `restore`
fork of the same parent. It prints the plain `seq_cp` characterization as
`[fork control] <lane>: ... N of M state bytes differ, max logit delta ...` and
does not assert on it - that line is the measurement, the assertion is the
byte-equality check. Dense attention is the byte-identical control; LFM2 is the
layout that breaks a plain `seq_cp`, which is what the partial hybrid fork
exists for.

The layout matrix (fresh cache, prior sequences, 1-2 branches) runs in full only
on recurrent/hybrid models. A dense model runs a single fresh-layout smoke check
plus the full `copy`/`restore`/`hybrid` strategy comparison on GPU.

Related assertions in the same subset, all task value: that the default fork
keeps the reference answers, that a strategy fork equals a full restore, that a
restored prefix reproduces the cold fork's winner and option set (never its last
bits), that a single-question bypass picks the same winner and restores the
trunk byte-for-byte, and that a multi-round greedy plan is unchanged by the
bypass.

## 8. Session lifecycle

The session store is policy over an owned token snapshot, and it is validated at
three levels:

| Level | What it covers |
|---|---|
| `tests/test-common-instances.cpp` | the eviction and accounting policy over a POD view: LRU order, the pinned / leased / removed / replacing skip rules, the charged-bytes rule, and the no-victim-means-refused case |
| `test_decision_engine` (`...session...`) | the engine's own fork-of-a-token-list path against a full re-prefill, on both the CPU and the GPU model |
| `test_decision_session_concurrency.py` | the wire behaviour: eager capture, implicit and first-class handles, TTL reaping, byte-budget eviction, snapshot identity under concurrency, and the warm tier |

`tests/test-common-instances.cpp` is the right home for the policy assertions
because `test-decision-engine` compiles no server code, and because a wire-only
test cannot cheaply reach an LRU order or a skip rule.

Two lifecycle rules that are worth stating when reading a result, because both
are easy to test wrongly:

* **TTL is a reaper, not a read filter.** It runs at the store-create and
  store-resolve points, never from a timer thread, so a test that creates a
  reference with a short TTL and then races a decision against it is testing a
  clock, not the reaper. Set the TTL while the lease is held instead.
* **A leased reference is skipped, not deferred.** It is never reaped late. To
  observe the lease deterministically, wait for the resolve to stamp
  `last_used_ms` rather than assuming a decode duration.

## 9. Reproducibility baselines

The frozen readout baseline is the deterministic core of the committed corpus:
per-question probabilities, winners, `confidence`, `certainty`, and label-pool
size. Timing is recorded for context and excluded from the byte diff. A dense
model is compared byte-for-byte; a recurrent or hybrid one is not, for the
reason in `API.md` section 3.1.

```sh
# record (GPU needs LLAMA_DECISION_TEST_MODEL; CPU uses the generated model)
./build/bin/test-decision-engine --record-readout gpu
./build/bin/test-decision-engine --record-readout cpu

# check: recompute and fail on any deterministic byte change
./build/bin/test-decision-engine --check-readout gpu
./build/bin/test-decision-engine --check-readout cpu
```

The in-suite gate picks whichever frozen baseline matches the model available in
the lane; a mismatch prints the first differing line.

**Deleting a member that a frozen baseline records breaks this gate.** The
comparison is byte-for-byte on the serialized core, so a removed field has to be
removed from every baseline file that carries it, deliberately and in the same
change.

## 10. Refreshing a frozen artifact

Do this only together with the code change that moved it. Never rewrite a
baseline to make a test pass without understanding the drift.

| Artifact | Refresh command | Scope |
|---|---|---|
| `calibration.json` rows | `test-decision-engine --write-calibration-rows` | host only; preserves model measurements |
| `calibration.json` full | `LLAMA_DECISION_TEST_MODEL=... test-decision-engine --write-calibration` | rows **and** model measurements - re-measures them |
| `baseline.json` CPU oracle | `test-decision-engine --write-cpu-oracle` | needs the generated model |
| `readout_gpu_baseline.json` | `LLAMA_DECISION_TEST_MODEL=... test-decision-engine --record-readout gpu` | GPU lane |
| `readout_cpu_baseline.json` | `test-decision-engine --record-readout cpu` | CPU lane |
| `decision_letter.golden.json` | `LLAMA_DECISION_TEST_MODEL=... test-decision-engine --write-decision-golden` | letter readout |
| `decision_basic.golden.json` | `test-decision-engine --write-golden` | shape/assemble |
| `accuracy_report.json` | `LLAMA_DECISION_ACCURACY_REPORT=... test_decision_accuracy.py` | accuracy |
| `fairness.json` bound | manual, with a matching code change | responsiveness |

The full calibration writer re-measures `model_measurements` for the model you
point it at, so it will overwrite the block recorded for a different model.
Back the file up, or write to a copy. Every calibration row carries
`production_gate: false`; a concentration row may not gate admission, caching,
routing, or persistence.

## 11. Reference model matrix

Three models, chosen because they exercise three different cache and fork paths.
A green run on one is not a green run.

| model | attention | what it is the control for |
|---|---|---|
| `lfm2.5-350m` | recurrent / hybrid SSM | the layout that breaks a plain `seq_cp`; the prefix-restore drift |
| `qwen3.5-2b` | hybrid | partial-format fork and the warm restore at scale |
| `gemma-4-e4b` | dense, sliding window, reasoning | bit-identical from the first call |

```sh
for m in "$LFM" "$QWEN" "$GEMMA"; do
  LLAMA_DECISION_TEST_MODEL="$m" ./build/bin/test-decision-engine || break
  LLAMA_DECISION_TEST_MODEL="$m" ./build/bin/test-decision-engine \
    "decision engine harness(\..*fork.*)?"
done
```

Dense models must stay byte-identical to the previous answers. LFM2 is where the
partial hybrid fork earns its place. The recorded baselines are single-machine
(ROCm, flash attention off) and are **not** a cross-backend guarantee.

## 12. ctest registration

```sh
# every decision test (the label alone is not enough - see section 0)
ctest --test-dir build --output-on-failure -R "test-decision"

# the engine oracles on their own
ctest --test-dir build --output-on-failure -R "test-decision-fork|test-decision-calibration"

# the regression suites a decision change can break
ctest --test-dir build --output-on-failure \
  -R "test-common-instances|test-server-instances-snapshots|test-save-load-state|test-recurrent-state-rollback"
```

| ctest name | Command | Labels |
|---|---|---|
| `test-decision-engine` | the full harness | `decision;main` |
| `test-decision-calibration` | `"decision engine harness(\..*calibration.*)?"` | `calibration` |
| `test-decision-fork` | `"decision engine harness(\..*fork.*)?"` | `fork` |
| `test-decision-permut` | `"decision engine harness(\..*permut.*)?"` | `permut` |
| `test-decision-temperature` | `test_temperature.py` | `temperature` |

`test-decision-engine` requires the `generate-models` fixture (the generated CPU
models) and always runs in the standard CPU CI lane. The model-bound GPU lanes
inside it stay skipped unless a GPU backend and `LLAMA_DECISION_TEST_MODEL` are
present.

`test-recurrent-state-rollback` fails on lfm2.5 and qwen3.5 on both CPU and GPU.
The identical failure reproduces on upstream master, so it is a pre-existing
upstream bug. Treat it as a known failure and confirm the **failure set** is
unchanged rather than treating a new entry as a regression.

## 13. Pitfalls

- Do not compare probabilities bit-for-bit across backends. Producer numerics
  reorder reductions on CUDA, Metal and ROCm, and flash attention changes them
  further. Compare with a tolerance; keep bit equality for the state-byte fork
  oracle and the deterministic golden responses.
- Do not compare a cold run against a warm run. Use `prefill_ms`,
  `state_cache_hit` and `warm_hit`.
- Do not compare the first server decode against later runs. On GPU the first
  batch can pick a different shape and flip a near-tie; warm the path once before
  taking a baseline.
- Do not look for `timings` on a default response. It is `diagnostics`-only.
- Do not use a tiny base model to judge framing agreement. On a 15M-parameter
  fixture the stateless and session framings flip a near-tie winner; the
  exact-winner session assertion is meaningful on a real model only. Run the
  suites against a reference model before believing a failure.
- Do not read `readout.delta` as a readout comparison. One readout exists in this
  build, so it is that readout against itself.
- Do not treat a session TTL as something a read observes. It is a reaper, and a
  GET on an expired-but-unreaped handle still answers 200.
- Do not test an adapter-scoped session by assuming a base-model decode. The
  captured scope is resolved and applied to the executor, and the reported
  `adapter_scope` is the scope that actually conditioned the replay.
- Do not present `confidence`/`certainty` as accuracy. They measure
  concentration, not correctness, and they never gate anything.
- `usage.output_tokens` is always 0. A non-zero value is a bug, not a benchmark
  result.
- A changed `template_hash`, quantization, or any frozen backend flag invalidates
  every prior parity, calibration and accuracy claim. Re-measure.
- The server scripts are standalone; `pytest`/`tests.sh` do not execute them.
  Run them directly, or wire them into CI.
- Export `LLAMA_SERVER_BIN` before running a server suite, or you will test a
  stale binary in `build/`.

## 14. Quick reference

```sh
# --- CPU ---
cmake -B build -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=ON && cmake --build build -j
ctest --test-dir build --output-on-failure -R "test-decision"

# --- GPU (ROCm) ---
GPU_TARGETS=gfx1100 BUILD_DIR=build-rocm ./build-amd.sh
export LD_LIBRARY_PATH="$PWD/build-rocm/bin"
MODEL=/path/to/model.gguf

# engine oracles (GPU lanes on)
LLAMA_DECISION_TEST_MODEL=$MODEL ./build-rocm/bin/test-decision-engine
LLAMA_DECISION_TEST_MODEL=$MODEL ./build-rocm/bin/test-decision-engine \
  "decision engine harness(\..*fork.*)?"

# frozen readout baseline
LLAMA_DECISION_TEST_MODEL=$MODEL ./build-rocm/bin/test-decision-engine --check-readout gpu

# --- server suites ---
export LLAMA_SERVER_BIN=$PWD/build-rocm/bin/llama-server
export LLAMA_SERVER_TEST_NGL=99
export LLAMA_SERVER_TEST_MODEL=$MODEL
python3 tools/server/tests/test_decision_envelope.py     # wire shape, errors, sessions
python3 tools/server/tests/test_decision_systemone.py    # Jev conformance
python3 tools/server/tests/test_decision_generic.py     # the schema shape
python3 tools/server/tests/test_decision_admission.py   # admission, isolation
python3 tools/server/tests/test_decision_session_concurrency.py  # lifecycle
python3 tools/server/tests/test_decision_agentic.py      # mixed agentic load
python3 tools/server/tests/test_decision_accuracy.py     # accuracy (gated)
```