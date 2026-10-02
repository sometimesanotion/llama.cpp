# Objective: multiple context windows with concurrent chat and decisions

Non-normative overview. It states what the merge of the instance-pool branch and
the decision branch was for, how the pieces fit, what actually landed, and what
"reliable and performant" means when testing it. The normative contracts live
in:

- `docs/decision/API.md` - the `/v1/decision` wire contract, and the combined
  generic + Jev request shape.
- `LLAMA_INSTANCES.md` - the instance-pool contract (one process, many windows).
- `docs/decision/README.md` - what the decision endpoint is for, and why it is fast.
- `docs/decision/OBJECTIVE.md` - why the engine looks the way it does (historical).
- `docs/decision/BENCHMARKING.md` - how to measure and validate it.

If this overview disagrees with `API.md` or `LLAMA_INSTANCES.md`, those win.

> **Status: the in-context decision lane has been removed.** Decisions now always
> run on the internal `__decision__` sidecar executor, on a single-context server
> as well as a pool, so a decision never shares a context with chat. The
> `--decision-instance` flag is gone with it, and so is the session arena and the
> `host`/`clone`/`file` backends. Sections describing the legacy shared-context
> lane (2.2's second half, 3.2, 4's second half, and the section 12 history) are
> kept only as design history; they no longer describe the server.

---

## 1. The combined goal

A single `llama-server` process should serve an agentic team, not one chat
session. That means three things at once:

1. **Load a model's weights once, then serve several independent context
   windows** ("instances"), each with its own size, parameters, KV cache,
   compute buffer and scheduler thread, so different agents do not share or
   evict one another's KV.
2. **Serve the decision API alongside the chat API**, including decisions that
   read a chat turn the agent has already produced, without that decision
   writing, stalling or resizing the chat window it read.
3. **Make the above reliable and measurable under concurrent agentic load**: no
   corruption, no cross-agent KV interference, bounded admission, and known,
   repeatable latency behaviour.

The merge combines two previously separate efforts:

| | from | what it contributed |
|---|---|---|
| the multi-context / instance-pool work | `server_instances` | one process, many independently sized windows, lazy materialisation, the pool API, adapter scoping per instance, the internal-executor concept |
| the decision-engine and session-substrate work | `parallel-decision` | the scoring engine, the Jev and generic front-ends, the two readouts, the live-session token snapshots, the admission and error surface |

The interesting part is where they meet, and it is one sentence: **a decision is
not a chat request, so it must not run on a chat window.** The instance pool is
what makes that expressible - it already knew how to own a context, schedule it
and hide it from the user, so the decision stack got a home of its own (the
sidecar executor) instead of a reserved slice of somebody else's window. Every
placement decision below follows from that.

The rest of this document describes how those meet, what survived, and what did
not.

---

## 2. The two concurrency axes

There are two independent levels of concurrency. Keep them separate in your head.

### 2.1 Between context windows (instances)

`server_instances` owns a pool. Each `--instance NAME:ctx=N[:parallel=M][:group=G]`
is a named context window with its own KV cache, compute buffer, scheduler
thread, and LoRA set. The model weights are loaded once and shared by all
instances.

- Instances are registered without allocating KV. They materialise lazily on
  the first request that targets them (`ensure_built_instance`). An unused
  window costs nothing but its registry entry.
- Each instance has its own scheduler thread and its own `server_context_impl`,
  so two instances can be dispatched concurrently at the HTTP/task level.
- Compute is not necessarily parallel at the device: on one GPU, contexts
  serialize at the backend. More instances means more VRAM, not free
  throughput. Do not claim parallel throughput without measuring it. The
  measured shape of that trade for decisions is in section 3.1.
- Routing is by `model`, `instance`, or `id_slot` in the body or query. The pool
  id (`base`) alone means "the default instance, or the sole instance, or
  ambiguous (400)".
- An instance flagged `internal` is undeletable and is never in a user group. The
  decision sidecar executor is exactly that, and that is why chat can never land
  on it and it can never be named as a session's owner.

### 2.2 Decisions vs chat context

With the sidecar executor (section 3.1) there is no in-context decision layer at
all: a chat context holds only its own chat slots, and decision traffic runs on
the sidecar's separate context. That is why decision work can neither stall nor
write a chat window. It is also why the two APIs now *run concurrently* and
contend for the device, which is a different thing to measure than the old
serialized lane (see `BENCHMARKING.md` section 6).

What the sidecar forces on its own context: `kv_unified`, because branches share
cells by metadata and a metadata-sharing fork only works in a unified cache.
That requirement does **not** propagate to chat instances.

**The legacy single-context lane, as history.** The design this replaced ran
decisions on the same context as chat, using disjoint sequence-id ranges inside
it:

```
chat slots:            [0, n_parallel)
decision engine pool:  [n_parallel, n_parallel + n_seq_decision)
session arena:         [n_parallel + n_seq_decision, ... + n_seq_arena)
```

A decision there ran inside `queue_tasks.yield_to_queue(...)` on the chat
context's scheduler thread; while it yielded, the scheduler declined every task
type except metrics and slot-get, and declined tasks were parked and replayed
FIFO. Its hard contract was that **chat on that context stalls for the whole
decision duration** - deliberate, but it made a multi-second decision a
multi-second chat stall, and a concurrent decode could perturb a chat token
through batch geometry. `--decision-arena-seqs` sized its session arena. None of
it exists in the tree.

---

## 3. Where decisions run

The decision sidecar executor is the single, default placement, and it is not
optional. Setting `--decision-seqs N` (N >= 3) is what turns the decision API on,
and it is what makes `params.decision_sidecar` true; with that set the pool
registers an internal, undeletable `__decision__` instance - its own context and
its own scheduler thread - and **every** decision, stateless and session, runs
there. A single-context server with no `--instance` at all gets one too, because
the whole point is that a decision never lands on a context chat is using. Chat
contexts carry no decision sequences and are never forced into the unified KV
cache, so a decision cannot stall or write a chat window. There is no fallback
placement and none may be added.

### 3.1 The sidecar executor (default)

- Cost: one extra context plus compute buffer once the sidecar is built on the
  first decision demand.
- Behavior: chat is never stalled by decision work. The sidecar competes with
  chat contexts for the GPU, so decision latency depends on total device load.
- The sidecar keeps its own resident-prefix warm tier
  (`--decision-warm-budget-mb`), so repeat decisions on a turn fork a kept
  prefix instead of re-prefilling.
- The sidecar is the only executor. `--decision-instance NAME`, which used to
  disable it in favour of the shared-context executor, has been removed.

Measured shape (4 agents, 4 stateless workers, 1 session worker, 2 churn
workers, all three target models):

- Small model (lfm2.5-350m): sidecar removes the decision-induced chat tail
  (chat max gap 611 ms -> 166 ms) at the cost of slower decisions.
- Mid model (qwen3.5-2b): sidecar greatly improves chat tail (max
  1641 ms -> 280 ms) but decisions are roughly 1.5-2x slower.
- Larger model (gemma-4-e4b): device-bound; sidecar gives little chat benefit
  and slows decisions. Four contexts contending for one GPU is the bottleneck.

### 3.2 Legacy shared context (removed)

This lane ran decisions on the same context as chat via the engine's reserved
sequence ranges and the decision yield, so chat on that context stalled for the
decision duration and any concurrent decode could perturb a chat token through
batch geometry. It is removed: the sidecar is the only executor, and
`--decision-instance` is gone. A single-context server now gets a sidecar
instance of its own.

The retained-turn registry, the host/clone/file session backends, the session
arena (`--decision-arena-seqs`), and the flags that configured them only ever
served this lane. They are removed from the tree.

### 3.3 Session decisions

A request may carry `id_slot` or a first-class `session_id` to answer about a
turn the slot already decoded, instead of re-prefilling the transcript.

- At create (or first use), the server takes an EAGER owned token snapshot of
  the completed turn - the token list plus adapter scope - through a read-only
  pool op on the owning instance. The snapshot is replayed into the sidecar
  context on demand (the `tokens` backend). Only owned copies cross the
  boundary: no KV pointer, sequence id, or context handle leaves the owning
  instance.
- Later decisions reuse the snapshot rather than the live slot, so the source
  slot can be cleared and reused by `cache_idle_slots`.
- Identity is a strong content hash over the token list plus the adapter scope,
  the turn counter, and the source slot. A changed turn or scope is 422; a
  token snapshot is model-epoch independent (a reload does not stale it).
- The source slot is read-only; the source slot's KV is never written by the
  decision.
- The sidecar executor offers only the `tokens` backend; the retired
  `clone`/`file` backends are a 501 capability refusal. The in-context session
  store (arena, memory epoch, retained-turn registry) has been removed.

---

## 4. Sequence and KV layout

The sidecar executor has its own context, so its decision engine pool lives in
its own KV cells, never in a chat context's. A chat context holds only its chat
slots. Weights are shared across all instances; each instance (chat and sidecar)
has its own KV cells. What the sidecar's context holds:

- the persistent **prefix snapshot** sequence, which survives between requests so
  a matching prefix is never re-decoded;
- the **resident warm prefixes** the warm tier keeps (`--decision-warm-budget-mb`),
  so a repeat decision on the same turn forks a kept prefix instead of
  re-prefilling;
- the transient **trunk** and **branch** sequences of the decision in flight.

A stateless decision prefills into the pool, scores, then removes its transient
sequences; only the prefix snapshot and any warm slot survive. A session
decision replays an owned token snapshot into the same pool, so it holds no cells
between decisions either - there is no session arena, by design, and nothing like
`arena_used` to leak.

**A capacity preflight estimates peak KV use and refuses with 422 rather than
ever partially overwriting the cache.** The refusal is deliberately placed
*before* any sequence is allocated, so an over-budget request never costs a
resident warm prefix that another request was relying on. `llama_decode`
returning 1 is also mapped to a client error rather than a partial answer.

**The legacy single-context layout, as history.** The design this replaced had
the engine sharing the chat context, in reserved sequence ranges above the chat
slots:

```
one context with --decision-seqs 4, --parallel 1, arena default 1

seq 0                     chat slot 0
seq 1 .. 4                decision engine pool (seq 1 is the persistent prefix snapshot)
seq 5                     session arena (materialized only during a session decision, host/file)
```

with a `host`/`file` session reference holding no cells between decisions (an
arena sequence allocated for the decision and freed on release) and a `clone`
reference sharing the source attention cells and pinning them for the session
lifetime. None of it exists in the tree.

---

## 5. The concurrency contract

Guaranteed:

- **Decisions never corrupt chat KV.** With the sidecar they run on a separate
  context, so chat's slots and their cells are not merely protected by
  convention - they are not addressable by a decision at all. On top of that the
  engine removes its transient sequences on every exit path, and admission
  bounds how many run at once.
- A decision reads a chat slot **read-only**: the source slot's KV is never
  written by a decision, and a decision never decodes on the slot's context.
- Admission bounds concurrent decisions, not their duration: 413 (body too
  large), 429 / 529 (queue full), 499 (client cancel), 503 + `Retry-After`
  (server-side deadline via `--decision-timeout-ms`, or a pool transient),
  422 (over-budget or
  semantically invalid).
- Chat and decisions **run concurrently on separate contexts** rather than
  serializing. The cost is device contention, which is measurable; the old
  serialization was not a safety property.

Not guaranteed:

- **Device-level isolation.** On one GPU the sidecar and the chat contexts
  contend, so a decision's latency depends on total device load, and a chat
  stream's inter-token cadence can be affected by a large decision. That is a
  throughput trade, measured in `BENCHMARKING.md` section 6, not a corruption
  risk.
- No cancellation of a decision during teardown. A long decision delays
  `DELETE`/resize on that instance until the abort deadline; then the management
  op may return 503 with the instance intact. Client disconnect during a
  decision is handled promptly (the engine stops), but a session capture already
  in flight completes.
- Session decisions do not cross context windows: a session handle belongs to the
  instance that owns its slot, and that instance is named on every get / patch /
  delete.

The measured shape of the sidecar trade, from the agentic mixed-load harness
(4 streaming chat agents, 4 stateless decision workers, 1 session worker, 2 churn
workers, one GPU, all three reference models):

- Small model (lfm2.5-350m): the sidecar removes the decision-induced chat tail
  (chat max inter-token gap 611 ms -> 166 ms) at the cost of slower decisions.
- Mid model (qwen3.5-2b): it greatly improves the chat tail (max 1641 ms ->
  280 ms) but decisions are roughly 1.5-2x slower.
- Larger model (gemma-4-e4b): device-bound; it gives little chat benefit and
  slows decisions. Four contexts contending for one GPU is the bottleneck.

---

## 6. Routing rules

`/v1/decision` and `/v1/session` are dispatched by one predicate,
`decision_request_is_session_pinned`, which asks a single question: does this
request name a retained turn - `session_id` or `id_slot`, in the body or in the
query string? Nothing else is computed from the request, and nothing is
computed from a stamped pool id: the `instance` the pool writes into a forwarded
body is echo-only and never selects a target.

| request | what the pool does |
| --- | --- |
| stateless decision (no `session_id`, no `id_slot`) | dispatch straight to the decision sidecar executor (the internal `__decision__`). `model` and `instance` are echo-only, so a bare pool id, a Jev alias (`jev-latest`/`jev-preview`) and an unknown `model` all answer on the sidecar; `model` is still REQUIRED by the contract |
| `session_id` or `id_slot` | resolve the owning instance, attach an owned token snapshot of that slot's completed turn, then dispatch to the sidecar |

The owner of a session-pinned request is resolved by `decision_owning_instance`:
an explicit `instance` wins, otherwise a `model` of the form `base:NAME` names
one. A group is refused, because the request addresses one instance's slot and a
group is not an instance. The internal `__decision__` instance is refused as an
owner (it has no chat slot to snapshot), an unknown name is refused, and a bare
pool id or no target resolves to the default instance, matching stateless
routing. The decision then runs on the sidecar *from the snapshot*, never on the
owning instance, so the source slot is only ever read.

Session handles are instance-local: get, patch and delete must name the owning
instance, the same way create did.

See `API.md` sections 2.4 and 2.6.

---

## 7. Configuration surface you will use

Everything the decision API needs, and nothing about it is optional once
`--decision-seqs` is set. `API.md` section 5.1 owns the environment overrides
that sit on top of these.

| flag | default | what it does |
|---|---|---|
| `--instance NAME:ctx=N:parallel=M:group=G[:default]` | - | one named context window per agent |
| `--decision-seqs N` | 0 (disabled) | **turns the decision API on.** Sizes the decision sequence pool (N >= 3: prefix snapshot, trunk, one branch) and, by being non-zero, is what registers the sidecar executor. Forces `kv_unified` on the sidecar context only, never on a chat instance |
| `--decision-sidecar-ctx N` | 0 | sidecar window size; 0 means the largest configured instance window, so the longest turn a chat instance can produce still replays |
| `--decision-sidecar-prebuild` | off | build the sidecar context at startup instead of on the first decision |
| `--decision-max-queue N` | 4 | concurrent decision cap; past it 429, past twice it 529 |
| `--decision-timeout-ms N` | 60000 | server-side deadline for a whole decision, calibrated as `max(30 s, 4x p99 cold-prefill)` on the reference GPU; on expiry 503 + `Retry-After`, never a partial answer. `0` disables it |
| `--decision-warm-budget-mb N` | 0 (off) | KV budget for the sidecar's resident session warm prefixes. With a positive budget the sidecar keeps recent session turns resident and forks them on a repeat instead of re-prefilling. A hit answers the same winner and option set as a miss on every model; on a recurrent or hybrid model the host-state restore can move the reported concentration by up to ~0.05, which is the documented repeatability rule, never the answer. A warm fork is admitted only if its peak fits beside every *other* resident prefix; one that does not is a 422 before the cache is touched |
| `--decision-session-ttl MS` | 0 (no expiry) | default TTL for created sessions. A **reaper**, not a read filter: it runs at the create and resolve points, never from a timer thread |
| `--decision-session-budget-mb N` | 0 (unlimited) | byte budget for the sidecar's owned token snapshots. Pressure evicts the least-recently-used unpinned, unleased reference and retries; a create is refused only when every remaining reference is pinned or in flight. The unit is mebibytes of token bytes, so the smallest expressible budget is 1 MiB = 262144 tokens |
| `--decision-temperature FILE` | - | calibrated temperatures plus provenance; refused if the provenance does not match the running model, quantization, template and backend flags |
| `--decision-contract HASH` | - | pin the contract identity (tokenizer + prompt template + label code); a mismatch refuses the decision path with a plain 501 |
| `--decision-permutations N` | 1 | server default order-debias passes for requests that omit `permutations` |
| `--jinja`, `--metrics` | - | `--metrics` is required for `/metrics`; without it that endpoint is 501 by design |

Tests require GPU offload: `-ngl 99`.

---

## 8. What reliable and performant means here

### 8.1 Test protocol

Build GPU-only (no CUDA/Vulkan on the reference machine), then run the engine
harness and the seven server suites against **all three** reference models -
lfm2.5-350m (recurrent), qwen3.5-2b (hybrid), gemma-4-e4b (dense, sliding
window, reasoning). They exercise different fork and cache paths, so a green run
on one is not a green run.

```sh
export LLAMA_SERVER_BIN=$PWD/build/bin/llama-server      # the ROCm build, always explicit

LLAMA_DECISION_TEST_MODEL=$MODEL ./build/bin/test-decision-engine
ctest --test-dir build -R "test-decision" --output-on-failure

for suite in envelope systemone generic admission session_concurrency agentic accuracy; do
  LLAMA_SERVER_TEST_MODEL=$MODEL python3 "tools/server/tests/test_decision_${suite}.py" || break
done

# the regression suites a decision change can break
ctest --test-dir build -R \
  "test-server-instances-snapshots|test-common-instances|test-save-load-state|test-recurrent-state-rollback" \
  --output-on-failure
```

`BENCHMARKING.md` owns the methodology behind each of those, the frozen
artifacts that own a number, and the known upstream failure in
`test-recurrent-state-rollback`.

### 8.2 Acceptance signals

- No crashes, no context shifts, no KV corruption.
- No unexpected 413/422/429/499/529; expected 422s only for genuinely
  over-budget or semantically invalid requests.
- Session churn does not monotonically grow memory: the sidecar holds only
  non-resident token snapshots, so there is no arena to leak, and under a
  configured budget or TTL the store reclaims references rather than growing.
- Chat answers are byte-identical whether or not a decision ran; a decision
  fired mid-chat leaves the chat answer unchanged.
- Chat is never *stalled* by a decision - the sidecar is a different context.
  Chat and decisions may still contend for the GPU, so device-bound latency
  interaction is measured rather than asserted.
- Other instances stay within their undisturbed latency under decision load.
- Master-vs-branch default path stays identical: with `--decision-seqs` and
  `--instance` absent, `/props`, chat output, and `n_seq_max` match master; the
  only divergence is `/v1/decision` returning 501 "decisions are disabled"
  (`DECISION_DISABLED_MESSAGE`, `ERROR_TYPE_NOT_SUPPORTED`) instead of master's
  404. `DECISION_DISABLED_MESSAGE` (`tools/server/server-common.h`) is the single
  source of the refusal wording; the pool and a single-context server both read it.

### 8.3 Interpreting memory

RSS growth during mixed chat+decision load is normally the stock prompt cache
(`cache_ram_mib` default 8192, `cache_idle_slots` on). With `--cache-ram 0` the
same workload is flat and stateless decisions add zero growth. Check that before
suspecting a leak. The sidecar's own resident memory is bounded by its context
size plus the warm budget, both explicit.

---

## 9. Current status and known gaps

Verified at the merge commit (2026-09):

- The decision suites, admission and session concurrency passed on all three
  models. The accuracy harness measures the labeled corpus on all three and
  gates the holdout split against per-model floors, so "passes" says something
  about answer quality here; see the quality gap below for what it does not fix.
- The instance and snapshot C++ suites, and save/load, passed.
- Cross-context host-format state transfer (a capture from one context loading
  into a context with different `n_ctx`/`n_seq_max`) matched to zero logit
  difference on all three models.

Per-milestone assertion counts and gate results are recorded in the roadmap
annotations rather than here; they move every time the harness grows, and a
number in this file would be stale within a day.

> M0-M9 completion (2026-09-29): the simplification roadmap
> `ROADMAP_20260929_SIMPLIFY.md` has landed. The decision sidecar executor is now
> the single decision executor, unconditionally once `--decision-seqs` is set
> (pool or no pool); sessions are eager token snapshots replayed on the sidecar
> (mechanism B); chat contexts carry no decision sequences and are never forced
> into the unified KV cache; the warm tier (`--decision-warm-budget-mb`) and the
> timeout/queue admission (`--decision-timeout-ms`, `--decision-max-queue`) are
> calibrated; per-instance and sidecar VRAM are reported in both `/instances` and
> `/props`. The old shared-context machinery (`session_registry`/
> `session_store`, the arena/backend flags) has been removed from the tree. The
> gates in that checklist are green on all three models; a `--cache-ram 0` soak
> shows flat RSS and no sequence drift.

> Red-team remediation (2026-10-01): this stack was reviewed against its three
> must-keep features and against the normative contract, and the findings were
> remediated. The plan, the reasoning behind each fix and the per-milestone
> measured outcome are in `ROADMAP_20261001_REDTEAM.md` and its checklist; that
> is the record, not this section. What belongs here is the two things a reader
> of the system needs: the accuracy numbers did not move, and two performance
> questions were settled by measurement rather than by argument.
>
> **Accuracy did not move, and that is the result rather than a non-event.** On
> the ROCm reference build the letter readout's winner agreement on the frozen
> 240-case corpus, measured before the work and again after it, is 0.3917 /
> holdout 0.3770 on `lfm2.5-350m`, 0.7375 / 0.7459 on `qwen3.5-2b` and 0.7542 /
> 0.7541 on `gemma-4-e4b` - identical to four decimal places on all three. Every
> change was on the task-value axis (exactness, capacity, memory, identity, error
> handling) and none of them touched how a distribution is produced, so that is
> what the frozen per-model floors in `GATE_FLOORS` are there to catch.
>
> Two performance questions are now settled by numbers. The single-branch decode
> bypass recovered **no** measurable compute on any of the three models - median
> and p95 `scoring_ms` ratios at 2, 16 and 64 options all sit within a few
> percent of 1.0 - so the fork strategies it would have enabled stay off. The
> resident-prefix cache **was** worth keeping: a hit beats a cold prefill by
> 12.6x / 23.9x / 58.6x at about 1k / 8k / 32k prefix tokens on `qwen3.5-2b` and
> 1.3x / 17.4x / 45.6x on `lfm2.5-350m`, so the host-state restore on recurrent
> and hybrid models stayed and removing it was rejected. One measurement behind
> that cache is fixed now and was wrong before: the reported `prefill_ms` used to
> exclude a cold prefill paid on a warm-miss.

Known gaps, in rough priority order:

0. **Jev answer quality on the lfm2.5 family is at chance, and the letter
   readout's framing is why it cannot be fixed from outside.** On the frozen
   240-case corpus (`tests/decision-baseline/accuracy_corpus.json`) the letter
   readout's holdout `winner_agreement` is **0.377** on `lfm2.5-350m`, against
   **0.746** (`qwen3.5-2b`) and **0.754** (`gemma-4-e4b`) on the same cases. A
   uniform guesser over the same option spaces scores **0.372**, so lfm2.5 is
   indistinguishable from chance while the control band is not: this is the
   model family, not the corpus or the path. The numbers are measured against
   ground truth and are gated - `GATE_FLOORS` in
   `tools/server/tests/test_decision_accuracy.py` freezes each model's holdout
   agreement (0.35 / 0.72 / 0.73), so a regression fails instead of passing
   quietly.

   **What was measured and rejected.** Scoring the answer *values* through the
   generic schema framing (one enum field per question over the wire values,
   reusing `compiled_schema` / `field_input`) instead of one letter label per
   option. On the same 122 holdout cases it scored **0.270 / 0.680 / 0.730** -
   worse on every model, including the two controls - at 1.4x the warm p50
   latency. The non-inferiority margin was frozen from the control models'
   calibration-split bootstrap intervals before any holdout number was read; the
   decision record is `READOUT_DECISION` in that harness and `readout_decision`
   in `tests/decision-baseline/accuracy_report.json`. So the earlier note that
   "the generic front-end scores 0.75 and 1.0 on the same two lfm models" does
   not survive a 240-case corpus - it came from four cases in a 12-case one, and
   no old accuracy figure on this branch is comparable to a new one.

   The mechanism is positional, not an implementation defect: the two framings
   compile to byte-identical scoring fields and the same engine scores them, but
   the schema framing puts the question in the shared prefix and shows only
   `"q0":` at the answer position, while the letter framing puts the question and
   its options immediately before `Answer:`. A future attempt has to beat 0.377
   on lfm2.5 on this corpus; it cannot be reasoned about from the framing
   "value scoring is more principled", because that framing is the one that
   measured worse.

   **That measurement can no longer be reproduced, and it is not evidence about
   the generic front-end.** Two things about it need saying plainly, so the next
   reader does not rebuild a deletion argument out of it.
   * It is **dead**. The code that produced the value framing has been deleted, so
     `paired_readout_delta` in the accuracy harness is now degenerate by
     construction: with one readout in the build it is that readout against
     itself, exactly zero with a zero-width interval, and the report says so in
     `delta.compared`. Only the frozen `readout_decision` block still carries the
     original margins and verdict.
   * It measured the wrong question. What ran was **one enum field per case over
     the wire values** - 240 distinct single-field schemas, each its own request,
     with no per-field record and no multi-field catalogue competing for the same
     prefix. That is sound evidence against the value readout as a **drop-in
     replacement for letter labels**, which is exactly what was asked of it. It is
     **not** evidence about the generic front-end as a deployed feature, because
     a real deployment does neither: it sends one stable schema, so the prefix
     cache hits and the measured ~1.4x cost largely evaporates, and it sends many
     fields per request, which is the regime this measurement never exercised.
     A multi-field measurement on a stable schema is a separate, still-unmade
     experiment.

   The front-end is kept, on the user's decision, for schema compilation. Keep it
   on that basis; do not reopen it on the basis of these numbers.
1. `test-recurrent-state-rollback` fails on lfm2.5-350m and qwen3.5-2b on CPU
   and GPU. The identical failure reproduces on upstream master, so it is a
   pre-existing upstream bug, not a regression from this merge. Do not treat it
   as a green gate until it is fixed upstream.

The earlier gaps (sidecar default routing, lazy-session invalidation by
idle-slot purging, and session-in-sidecar) are resolved by the M0-M9 work above:
routing is the single session-pinned predicate of section 6, sessions are
captured eagerly, and sessions run on the sidecar as token snapshots.

---

## 10. Reading order for a new agent

1. `docs/decision/README.md`, for what the endpoint is for and why it is fast.
2. This file, for the goals of the merge and the mental model.
3. `docs/decision/API.md`, for the wire contract of the combined generic and Jev
   shape. It is normative; where this file disagrees, it wins.
4. `LLAMA_INSTANCES.md`, for the instance pool (windows, routing, adapters,
   snapshots).
5. `docs/decision/BENCHMARKING.md`, for how to measure and validate any claim.
6. `docs/decision/JEV-API.md`, if you are writing a Jev client.
7. `docs/decision/OBJECTIVE.md`, for the historical engine comparison and the
   reasoning behind the fork strategy.
8. Source entry points: `tools/server/server-instances.{h,cpp}` (the pool, the
   sidecar, the session store), `tools/server/server-context.{h,cpp}` (the
   decision handler and the routing), `tools/parallel-decision/decision-engine.{h,cpp}`
   (the scoring engine), `tools/parallel-decision/decision-protocol.{h,cpp}` (one
   envelope, both shapes). The legacy-lane session registry and store have been
   deleted.

---

## 11. Glossary

- Instance / context window: a named, independently sized KV + compute context
  built from shared weights, with its own scheduler thread.
- Internal instance: an `internal` pool instance - undeletable and in no user
  group. The decision sidecar executor is the only one, and chat can never land
  on it.
- Sidecar: the decision executor - the internal `__decision__` pool instance
  with its own context and scheduler thread. It owns every decision, on a
  single-context server as well as a pool, and is never written or stalled by
  chat.
- Shared context / legacy lane: removed. Decisions ran on the same context as
  chat, so chat stalled for the decision and a concurrent decode could perturb a
  chat token through batch geometry.
- Engine pool: the decision sequences inside a context. In the sidecar's context
  they are its only contents: the persistent prefix snapshot, any resident warm
  prefixes, and the transient trunks and branches of a decision in flight.
- Session arena: removed. The sidecar's session references are non-resident token
  snapshots, so there is nothing to reserve between decisions.
- Yield: the mechanism that lets a decision run on its scheduler thread while
  only `/metrics` and `/slots` are answered. Still used inside a decision, on
  whichever context that decision runs on.
- Token snapshot: the owned copy of a completed turn's tokens and adapter scope
  taken from the owning instance through a read-only op and replayed into the
  sidecar context; the sidecar's only session backend.
- Non-resident: a session reference that holds no context cells between
  decisions.
- Fork: the strategy that gives each branch a copy-on-write view of the parent
  sequence (`copy` / `restore` / `hybrid`, chosen by `select_fork`).
- Task value vs producer concentration: two different axes. Fork-state equality,
  capacity preflights, adapter-scope application and memory budgets are task
  value. `confidence` and `certainty` describe how peaked a distribution is and
  never gate anything. See `API.md` section 6.

---

## 12. Simplification findings (2026-09-29)

Non-normative. This section records the architecture review that led to
`ROADMAP_20260929_SIMPLIFY.md`. **That roadmap is not in the tree** - it was a
planning artifact and has since been removed, so this section and the M0-M9
status note above are the surviving record of it. Where the review disagrees
with the rest of this file, the simplification target wins for new work; the old
behavior stays documented until the prune phases land. As of the M0-M9 landing,
the simplification target is the current implementation (section 3): the sidecar
executor with token snapshots is the single decision placement, so the
shared-context machinery below is the legacy lane described in sections 2.2/3.2/4.

### 12.1 Branch correction

`_codacus_parallel_decision` is not itself a sidecar. It reserves decision
sequences inside the chat context (`--decision-seqs` grows `n_seq_max`), decodes
on the same scheduler thread, and shares the prompt's KV cells with
`llama_memory_seq_cp`. A separate standalone worker process drove it, and that
worker has since been removed from the tree. The "decision sidecar with its own
context" called for by the simplification target does not exist on either source
branch; it is new work built on the instance pool from
`_multi_context_adapter_ds`.

The current branch adds the decision stack on top of the instance pool (77 files,
about 24k lines over `_multi_context_adapter_ds`): the Jev and generic
frontends, the readouts, the session registry with `host`/`clone`/`file`
backends, the reserved decision and arena sequence ranges, the decision yield,
and the admission/error surface. The simplified target keeps the scoring engine,
the frontends, admission, and the adapter-scope idea, and replaces the KV
sharing with a sidecar plus a read-only token snapshot.

### 12.2 Snapshot mechanism experiment (A vs B)

Setup: one model, a chat context and a sidecar context, a synthetic turn of
2k/8k/32k tokens, a 16-token question head decoded with logits, GPU-only
(`-ngl 99`). A = `llama_state_seq_get_data_ext` on the chat context, then
`llama_state_seq_set_data_ext` on the sidecar. B = the sidecar re-prefills the
turn's token list (cold) or forks an already-resident prefix (warm). "Shared"
is the question decode on the chat context with the prefix resident, which is
the current on-demand suffix cost. Times in ms; maxdiff is against the shared
continuation.

| model | N | A capture (chat thread) | A load | A total | B cold | B warm | shared | maxdiff A | maxdiff B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lfm2.5-350m | 2k | 5.7 | 4.5 | 14.0 | 59.8 | 4.1 | 3.8 | 0 | 0 |
| lfm2.5-350m | 8k | 26.4 | 16.2 | 46.5 | 206.5 | 4.4 | 4.1 | 0 | 0 |
| lfm2.5-350m | 32k | 83.0 | 66.8 | 154.2 | 1198.2 | 5.5 | 5.0 | 0 | 0 |
| qwen3.5-2b | 2k | 11.6 | 8.8 | 34.3 | 236.6 | 11.4 | 10.5 | 0 | 0 |
| qwen3.5-2b | 8k | 28.6 | 20.5 | 59.8 | 886.3 | 11.9 | 10.9 | 0 | 0 |
| qwen3.5-2b | 32k | 92.2 | 70.6 | 174.8 | 4198.6 | 13.8 | 12.2 | 0 | 0 |
| gemma-4-e4b | 2k | 14.8 | 10.1 | 50.5 | 503.4 | 46.9 | 24.4 | 1.29 | 0 |
| gemma-4-e4b | 8k | 35.1 | 25.4 | 85.5 | 2127.2 | 50.5 | 26.0 | 1.07 | 0 |
| gemma-4-e4b | 32k | 103.1 | 87.0 | 221.9 | 16316.4 | 67.3 | 42.2 | 0.885 | 0 |

Warm measurement note: for lfm and qwen the warm column is a metadata fork
(`llama_memory_seq_cp`) of the resident prefix plus the question decode; for
gemma it is question-tail removal plus re-decode, because the raw fork in the
bench harness does not reproduce the engine's sliding-window fork handling. The
production warm path is the engine's own fork and is calibrated in M7 of the
roadmap.

Findings:

- B is bit-exact against the shared continuation on every model and size
  (B-S = 0). A is exact on lfm and qwen, and exact on gemma only at prefixes
  within the 512-token sliding window; beyond it, A-S is about 1.0 logit, which
  can flip a token. Host-format byte capture does not round-trip a windowed
  cache into another context exactly.
- B cold is expensive on the sidecar (gemma 32k: 16.3 s) but it never touches
  the chat context and is paid once per turn; after the first decision a
  resident prefix makes follow-up decisions as cheap as the current shared
  suffix (B warm; lfm/qwen measured, gemma to be measured by the engine in the
  warm-cache calibration milestone).
- A's capture is a recurring tax on the chat thread (83-103 ms per 32k turn)
  and holds 384-532 MB of host bytes per 32k turn; B holds only the token list
  (about 128 KB). A also forces eager capture at turn completion, because the
  stock `cache_idle_slots` purge invalidates a lazy capture before the decision
  arrives (finding F1).

Decision rule, as applied: a mechanism is admissible only if it reproduces the
source context's continuation exactly on every model (a task-correctness gate,
not a producer-confidence gate). A fails that gate on the SWA model, so B is
the only mechanism. The remaining choices in the design are performance or
capacity gates and are calibrated in their own milestones. Producer confidence
in the engine output is reported and never gates routing, caching, or
persistence.

#### 12.2.1 Reproduction run (M0 gate)

Harness: a standalone snapshot-measurement binary that lived in
`tools/parallel-decision/` and was **deleted when the experiment ended**, so it
cannot be re-run and there is no source to look for; the recorded results below
are kept and are the only record of it. GPU-only (`-ngl 99`), one chat context
and one sidecar context with identical params (`kv_unified`, `swa_full=false`,
flash attention off for bit-exactness). The turn is tokenized from a fixed
synthetic paragraph; the 16-token question head is tokenized from a fixed
sentence. "A" is `llama_state_seq_get_data_ext`/`set_data_ext` between the two
contexts, "B cold" is a from-scratch re-prefill of the same token list on the
sidecar, "B warm" is a `llama_memory_seq_cp` fork of a clean turn-only prefix
plus the head decode, and "shared" is the head decode on the chat context with
the prefix resident. Times are ms; maxdiff is the max |logit| gap of the
mechanism continuation against the shared continuation. Rerun on 2026-09-29 at
N=2048:

| model | N | A capture | A load | A total | B cold | B warm | shared | maxdiff A | maxdiff B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lfm2.5-350m | 2k | 21.9 | 4.6 | 31.8 | 61.7 | 14.8 | 17.5 | 0 | 0 |
| qwen3.5-2b | 2k | 49.6 | 9.0 | 75.7 | 236.4 | 57.2 | 27.0 | 0 | 0 |
| gemma-4-e4b | 2k | 125.2 | 9.9 | 162.3 | 544.9 | 167.2 | 38.2 | 0.9691 | 0 |

The qualitative findings from the recorded table reproduce: B is bit-exact on
every model (maxdiff 0); A is exact on lfm and qwen but inexact on gemma beyond
its sliding window (maxdiff about 0.97, against 1.29 recorded; same order, same
verdict). A capture is a recurring tax on the chat thread (22-125 ms here at 2k,
growing with the turn). Absolute timings differ from the recorded table because
that run used different batch/context settings and a different machine; the
correctness gates (maxdiff) match, which is the load-bearing result. The warm
maxdiff is reported as 0 for lfm/qwen only when the fork copies a clean
turn-only prefix; forking a seq that already decoded the head shares the
recurrent tail cell and is not a valid warm measurement (see the engine's own
fork, calibrated in M7).

### 12.3 Target shape in one paragraph

One sidecar executor owns the decision engine, its own context, and one
scheduler thread. Sessions are token snapshots: at create or first use the
server copies the completed turn's tokens and adapter scope out of the owning
instance (a bounded read, no KV), and the sidecar re-prefills them into its own
KV on demand. A resident-prefix cache in the sidecar turns repeated decisions on
one turn into the current fork-and-decode cost; it is a performance layer only.
Chat contexts lose every decision sequence range, the forced unified KV, the
yield, and the session arena, so decisions can neither stall nor write them.
The full milestone plan is in `ROADMAP_20260929_SIMPLIFY.md`.
