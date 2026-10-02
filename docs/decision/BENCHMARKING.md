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

### 4.1 Field compilation cost

`compile_fields` is off the request clock (it never decodes), but at the documented
limits it can dominate the CPU time a request costs before the GPU does any work.
It holds a per-engine token cache, an exact-field dedup, and a candidate-path
collision check. Measured on the `qwen3.5-2b` tokenizer with the CPU scaffold, by
compiling the field sets below through the env-gated `compile_fields cost probe`
in `tests/test-decision-engine.cpp` (point `LLAMA_DECISION_BENCH_MODEL` at a real
tokenizer and filter the harness to the probe):

| workload | before | after |
|---|---|---|
| typical: 8 questions, 1 pass, 4 options, cold cache | 0.28 ms | 0.29 ms |
| typical, warm cache | 0.01 ms | 0.01-0.02 ms |
| worst: 256 questions, 8 passes, 255 options | 3962 ms | 3893 ms |

The typical case does not regress, which is the acceptance condition. The worst
case does not move within run-to-run noise, and the reason is worth recording: the
token cache reports **522236 misses and 4 hits** on that field set, because every
`(suffix, candidate)` pair is distinct, so the cache cannot hit there and the
eviction policy only changes how the overflow is discarded. The wall time is
dominated by the O(K^2) candidate-path collision check and the trie build, which
are deliberately left in place; see the follow-up note in the roadmap. The cache
and dedup changes are therefore a bounded-memory and expected-comparison-count fix,
not a wall-time win at that size. A workload whose tokenizations repeat is where
the cache policy pays; the warm row is that case.

**Current tree baseline.** Re-measured with the probe on the ROCm build with the
`qwen3.5-2b` tokenizer: the typical case is 0.42 ms cold (32 tokenizations, 0
hits) and the worst case is 5006 ms (522240 tokenizations, 0 hits). The worst
case is dominated by the O(K^2) candidate-path collision loop in
`engine::compile_fields`, not by the token cache or the dedup, because every
`(suffix, candidate)` pair is distinct at that size.

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

### 8.1 The scheduler acknowledgement budget

After a session decision is dispatched, a pool-owned release worker waits for the
sidecar's scheduler to acknowledge that the task is finished, then drops the lease
on the store entry. The client response does not wait for any of it. That wait is
bounded by `DECISION_SCHEDULER_ACK_BUDGET_MS` (`common/common.h`), and the
measurement behind the constant is recorded here.

**What it measures.** The interval from posting the acknowledgement op to its
return. It is a liveness property of the executor and nothing else; it is never
compared against a reported distribution measure, and it is deliberately not
derived from `--decision-timeout-ms`.

**How to reproduce.** Start `llama-server` with `-lv 5` (the duration is logged at
debug level as `decision drain took Nms of a Bms budget`) and run a session
decision. A stateless decision takes no drain at all, which is itself the first
control: the instrument is not on that path.

Reference lane, ROCm on an RX 7900 XT (gfx1100), `--decision-seqs 8`, the
concurrency cap at its default of 4 unless noted:

| workload | lfm2.5-350m (recurrent) | qwen3.5-2b (hybrid) | gemma-4-e4b-it (dense) |
|---|---|---|---|
| clean session decision, one at a time | p50 0 ms, max 1 ms | p50 0 ms, max 0 ms | p50 0 ms, max 1 ms |
| 4 concurrent decisions on one session, 3 questions | p50 32 ms, p99 95 ms, max 95 ms | p50 126 ms, p99 378 ms, max 378 ms | p50 263 ms, p99 786 ms, max 786 ms |
| 4 concurrent, 32 five-level questions and a long prefix | - | - | p50 868 ms, p99 2602 ms, max 2602 ms |
| 16 concurrent with 3 decision sequences | - | - | max 2404 ms (the drain tail) |

**The control group that must not time out** is the first row: a decision that
completes normally on the smallest model, and on the largest at the configured
`--decision-seqs`, both drain in about a millisecond. If either approaches the
budget, the budget is wrong, not the workload.

**What sets the tail** is the second and third rows, and it is not the cancelled
decision. A client that disconnects mid-decode has its task cancelled out of the
queue before it runs, so its drain returns immediately (measured 0 ms at cancel
delays from 50 ms to 1.5 s on a 707 ms decision). The tail is the *siblings*: a
drain queues behind whatever else is on the single sidecar scheduler thread, so
its length scales with the remaining decisions in flight, roughly one to three
times a single decision on a saturated queue.

**How the constant was chosen.** The measured p99.9 of the heaviest workload
above is 2602 ms. The budget is **30000 ms**, about 11x that, which leaves room
for a slower host while staying inside the regime where the wait is a bounded
management-plane wait rather than a wedged thread. It is not a multiple of the
decision deadline: that deadline is producer-facing and may already be spent by
the time the drain runs, which is exactly why reusing it would prove nothing.

**The failure mode is inverted on purpose.** A drain that cannot be observed
inside the budget keeps the lease, the entry's adapter references and the
transient snapshot key, and logs at error level. An unobserved task is
indistinguishable from a running one, so the only safe reading of "the wait was
too long" is "we could not prove the reader is finished". The cost is a retained
reference - visible in `GET /v1/session/{id}` as `bytes` and in the store's
counters, and clearable by `DELETE` - rather than adapter memory a running
decode still points at.

One detail worth knowing before trusting the number: the op's own deadline is
only consulted where the result queue's poll expires, and on a busy context
other results keep waking that poll, so a deadline can pass without the wait
returning. The budget is therefore enforced against the measured elapsed time as
well as the returned result, and both must agree that the scheduler answered
inside it.

### 8.2 Baseline findings: the release drain and the compile path

Two costs are measured before the release path and the candidate-path check are
changed, so a later comparison has a fixed before.

**Release drain.** A session decision holds a lease through
`decision_lease_guard`. Its destructor calls
`release_decision_snapshot_after_dispatch` on the HTTP thread, which posts
`decision_drain_ack` through `instance_op` and waits up to
`scheduler_ack_budget_ms()`. Only when the sidecar's scheduler is observed does it
erase the transient `decision_snapshot_resolve_` key, decrement the lease and call
`release_decision_refs`; an unobserved wait keeps all three. The wait therefore
sits on the client's critical path after the answer exists, and the measured tail
is the table in 8.1.

**Collision check.** `engine::compile_fields` runs a pairwise loop over the
candidate `paths` of every field and rejects a field whose options tokenize to a
colliding path. `decision_field::build_nodes` computes the same relation on the
trie it builds immediately after, so the pairwise loop is a second implementation
of the invariant. At the documented worst case the loop dominates the compile
wall time in 4.1.

### 8.3 Release worker calibration

The post-dispatch drain and the store release run on one pool-owned, non-scheduler
worker behind a bounded FIFO (`server_instances::release_loop`). Four thresholds
govern it, and all four bound **executor liveness or queue capacity**. None of
them is a quality gate: no reported `confidence`, `certainty` or any other
distribution measure is compared against any of them, and none of them caches,
routes, admits or persists an answer.

A queued job carries a backoff deadline rather than making the worker sleep, and
the worker runs the first job whose deadline has passed. That is what keeps one
job's backoff off the whole queue. A job that cannot observe the scheduler is
never abandoned: after the short ladder it is reported once and parked on the long
interval, still retrying, because a reference and the snapshot it holds are
retained memory until something observes the reader finishing.

| Constant | Frozen value | Bounds |
|---|---|---|
| `release_fast_attempts` | 5 | short-ladder attempts before a job is reported as retained and moved to the long interval |
| `release_retry_backoff_ms` | 50 | a short-ladder retry's backoff |
| `release_retained_retry_ms` | 60000 | a parked job's retry interval until the scheduler is observed |
| `max_release_jobs` | 128 | deepest pending queue; a full queue falls back to the inline release rather than dropping a lease |

**Where to read them.** `GET /v1/session/{id}` reports the worker's counters in a
nested `counters.release` block, next to the session store's own counters:

```json
"counters": { "n_snapshots": 1, "n_reuses": 16, "n_releases": 0,
              "n_sessions": 1, "bytes_total": 3188, "n_pending": 0,
              "release": { "jobs": 16, "retries": 0, "retained": 0, "inline": 0,
                           "queue_high": 14, "bytes_pending": 0, "bytes_high": 47820 } }
```

`jobs` are leases whose drain was observed and whose references were released;
`retries` are unobserved drains that were re-queued (including the one that parks a
job); `retained` are jobs that spent the short ladder and are now parked, still
holding their lease, their adapter references and their transient snapshot key;
`inline` are leases released on the HTTP thread because the queue was stopped or
full; `queue_high` and `bytes_high` are high-water marks and `bytes_pending` the
snapshot bytes held by unfinished jobs right now. A `bytes_pending` that never
returns to zero is a retained snapshot, which is the one number on this path that
represents memory rather than work.

**How to reproduce.** Both arms run inside
`tools/server/tests/test_decision_session_concurrency.py` and print their own
measurements:

```sh
export LLAMA_SERVER_BIN=$PWD/build-decision/bin/llama-server
LLAMA_SERVER_TEST_MODEL=$MODEL python3 tools/server/tests/test_decision_session_concurrency.py
```

The workload is `_drain_scenario`: one session, then 16 concurrent decisions of 32
five-level questions against a long prefix with 3 sidecar sequences and
`LLAMA_DECISION_MAX_QUEUE=32`.

**Control group, which must not fire.** The shipped budget observes every drain on
its first attempt, so a normal session workload must produce **zero** retries,
zero retained leases and zero inline fallbacks. This is asserted, not merely
measured: if any of the three is nonzero the shipped budget is mistuned and the
test fails rather than absorbing the regression. Reference lane, ROCm:

| | lfm2.5-350m | qwen3.5-2b | gemma-4-e4b-it |
|---|---|---|---|
| jobs / retries / retained / inline | 16 / 0 / 0 / 0 | 16 / 0 / 0 / 0 | 16 / 0 / 0 / 0 |
| retry rate per job | 0.00 | 0.00 | 0.00 |
| peak pending queue depth | 13-14 | 14 | 14 |
| peak pending snapshot bytes | 44632-47820 | 47700 | 47880 |
| pending bytes after the storm settles | 0 | 0 | 0 |
| transient resolve keys vs. pre-storm baseline | equal | equal | equal |

The two marked ranges are arrival-timing dependent (the drain op is FIFO behind
the sidecar's own queue, so the depth depends on how far the storm got before the
worker caught up); the control properties that must not move are the three
zero-valued rows.

**Stress arm, which must fire.** The forced arm sets
`LLAMA_DECISION_SCHEDULER_ACK_BUDGET_MS=-1`, the never-observe sentinel, so every
drain fails and every job walks the short ladder and parks:

| | lfm2.5-350m | qwen3.5-2b | gemma-4-e4b-it |
|---|---|---|---|
| parked leases | 16 | 16 | 16 |
| retries | 80 | 80 | 80 |
| pending snapshot bytes still held | 51008 | 50880 | 51072 |
| peak pending queue depth | 16 | 16 | 16 |

`80 = 16 x release_fast_attempts`, so every job parks exactly at the end of its
short ladder, and the retained bytes are all 16 snapshots: the failure is fully
accounted rather than only logged. The queue depth is the whole storm at once,
which is the point of moving the backoff into a deadline - before that change this
arm measured a depth of 1 to 7, because a worker sleeping inside one job's backoff
serialised the queue behind it and the depth was an artifact of the sleep rather
than a count of the leases actually held.

**How the constants were frozen.**

* *Queue cap = 128.* The pending depth cannot exceed the number of leases held at
  once, and the pool's own admission gate bounds that at `2 x
  --decision-max-queue` (the 529 threshold; above the cap a decision is 429). The
  reference lane ran with `LLAMA_DECISION_MAX_QUEUE=32`, so the largest admissible
  burst is 64 and the measured peak was 13-14. 128 is 2x the largest admissible
  burst and 9x the measured peak, so the inline fallback - which reintroduces the
  pre-change client-side wait - is unreachable in normal operation. R4's
  provisional 64 was exactly the admissible burst at this queue setting, leaving no
  headroom; 128 removes that edge.
* *Short ladder = 5 attempts.* The condition being retried is a sidecar scheduler
  briefly behind its own queue, and the drain op is FIFO behind the decision tasks,
  so the useful question is how long a job may wait for the queue to move. Five
  attempts is 250 ms of short backoff on top of the first attempt. The measured
  retry rate of 0.00 per job says the ladder is a tail mechanism, so the tail is
  what bounds it. The ladder no longer terminates a job: it only decides when a job
  is *reported* and moved to the long interval.
* *Backoff = 50 ms.* The queue moves in units of one scheduler task, and a single
  decision measures 17 ms / 59 ms / 83 ms on the three reference models, so 50 ms
  is one to three task times: long enough that the retry lands after the queue has
  moved, short enough that it stays far inside the 30 s budget. Nothing in the
  measurement supports a longer pause, and a longer one only widens the window in
  which a pending job holds its snapshot.
* *Parked interval = 60 s.* The condition that parks a job is the drain op's result
  not arriving inside a 30 s budget, which is not a condition that clears in
  milliseconds. 60 s trades self-heal latency against the one resource a parked job
  still costs: each retry holds the single worker for up to one budget, so a
  permanently unobservable scheduler keeps the worker busy roughly half the time
  while it retries. That is the deliberate backstop - the queue cap and the inline
  fallback sit behind it, so even that case degrades to a slower release rather than
  to a dropped lease. An operator erase does not wait for the interval: it wakes the
  parked job immediately, which the stress arm asserts (a growth in `retries`
  inside 5 s, against a 60 s interval, can only be the wake).

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