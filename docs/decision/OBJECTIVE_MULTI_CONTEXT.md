# Objective: multiple context windows with concurrent chat and decisions

Non-normative overview. It explains what this merged branch is for, how the
pieces fit, and what "reliable and performant" means when testing it. The
normative contracts live in:

- `docs/decision/API.md` - the `/v1/decision` wire contract.
- `LLAMA_INSTANCES.md` - the instance-pool contract (one process, many windows).
- `docs/decision/README.md` - decision-engine design notes and KV background.
- `docs/decision/OBJECTIVE.md` - why the engine looks the way it does.
- `docs/decision/BENCHMARKING.md` - how to measure it.

If this overview disagrees with `API.md` or `LLAMA_INSTANCES.md`, those win.

> **Status: the in-context decision lane has been removed.** Decisions now always
> run on the internal `__decision__` sidecar executor, on a single-context server
> as well as a pool, so a decision never shares a context with chat. The
> `--decision-instance` flag is gone with it. Sections describing the legacy
> shared-context lane (3.2, and the legacy-lane notes elsewhere) are kept below
> only as design history; they no longer describe the server.

---

## 1. The combined goal

A single `llama-server` process should serve an agentic team, not one chat
session. That means three things at once:

1. Load a model's weights once, then serve several independent context windows
   ("instances"), each with its own size and parameters, so different agents do
   not share or evict one another's KV.
2. Serve the decision API (`/v1/decision`, `/v1/session`) alongside the chat
   API, including decisions that read an existing chat turn. A decision either
   runs on a context that also serves chat (sharing its KV cache), or on a
   dedicated context (a "sidecar") so it does not stall chat.
3. Make the above reliable and measurable under concurrent agentic load: no
   corruption, no cross-agent KV interference, bounded admission, and known,
   repeatable latency behavior.

The merge combines two previously separate efforts:

- the multi-context / instance-pool work (`server_instances`), and
- the decision-engine + session-registry work (`parallel-decision`).

The rest of this document describes how those meet.

---

## 2. The two concurrency axes

There are two independent levels of concurrency. Keep them separate in your head.

### 2.1 Between context windows (instances)

`server_instances` owns a pool. Each `--instance NAME:ctx=N[:parallel=M][:group=G]`
is a named context window with its own KV cache, compute buffer, scheduler
thread, and LoRA set. The model weights are loaded once and shared by all
instances.

- Instances are registered without allocating KV. They materialize lazily on
  the first request that targets them (`ensure_built_instance`). An unused
  window costs nothing but its registry entry.
- Each instance has its own scheduler thread and its own `server_context_impl`,
  so two instances can be dispatched concurrently at the HTTP/task level.
- Compute is not necessarily parallel at the device: on one GPU, contexts
  serialize at the backend. More instances means more VRAM, not free
  throughput. Do not claim parallel throughput without measuring it.
- Routing is by `model`, `instance`, `snapshot`, or `id_slot` in the body or
  query. The pool id (`base`) alone means "the default instance, or the sole
  instance, or ambiguous (400)".

### 2.2 Decisions vs chat context

With the sidecar executor (section 3.1), there is no in-context decision layer
at all: a chat context holds only its own chat slots, and decision traffic runs
on the sidecar's separate context. This is why decision work can neither stall
nor write a chat window.

The legacy single-context lane (section 3.2) keeps the old shared layout:
disjoint ranges of sequence ids inside one context.

- chat slots: `[0, n_parallel)`
- decision engine pool: `[n_parallel, n_parallel + n_seq_decision)`
- session registry arena: `[n_parallel + n_seq_decision, ... + n_seq_arena)`

A decision on that shared context runs inside `queue_tasks.yield_to_queue(...)`
on that context's scheduler thread. While it yields, the scheduler declines every
task type except `SERVER_TASK_TYPE_METRICS` and `SERVER_TASK_TYPE_SLOT_GET`;
declined tasks are parked and replayed FIFO after the yield. A CANCEL posted
during the yield still removes its target.

The hard contract that follows (legacy lane only): **chat on the same context is
stalled for the whole decision duration.** This is deliberate, not a bug. Other
instances are unaffected.

`--decision-seqs N` (N >= 3) enables the engine and forces `kv_unified` (in the
legacy lane; the sidecar forces it only on its own context).
`--decision-arena-seqs` sizes the session arena (legacy lane, default
`n_parallel`).

---

## 3. Where decisions run

The decision sidecar executor is the single, default placement. When
`--decision-seqs` is set and a pool exists (`--instance`), the pool registers an
internal, undeletable `__decision__` instance - its own context and scheduler
thread - and every decision (stateless and session) runs there. Chat contexts
carry no decision sequences and are never forced into the unified KV cache, so a
decision cannot stall or write a chat window.

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
has its own KV cells.

Legacy single-context lane (no pool): the engine shares the chat context, using
reserved sequence ranges above the chat slots:

```
one context (one instance) with --decision-seqs 4, --parallel 1, arena default 1

seq 0                     chat slot 0
seq 1 .. 4                decision engine pool (seq 1 is the persistent prefix snapshot)
seq 5                     session arena (materialized only during a session decision, host/file)
```

- A stateless decision prefills into the engine pool, scores, then removes its
  transient sequences. Only the prefix snapshot survives for reuse.
- A `host`/`file` session reference holds no cells between decisions; an arena
  sequence is allocated for the decision and freed on release.
- A `clone` session reference shares the source attention cells and pins them
  for the session lifetime (opt-in; not available on sliding-window models).
- A capacity preflight estimates peak use and returns 422 rather than partially
  overwriting the cache; `llama_decode` returning 1 is also mapped to 422.

---

## 5. The concurrency contract

Guaranteed:

- Decisions never corrupt chat KV: with the sidecar they run on a separate
  context; in the legacy shared lane they use disjoint sequence ranges, a
  read-only source slot, cleanup on every exit, and admission limits.
- Only `/metrics` and `/slots` are served in a legacy shared context while a
  decision yields.
- Declined tasks are parked and replayed FIFO, and are not lost or reordered.
- Admission bounds concurrent decisions, not their duration: codes 413 (body too
  large), 429 / 529 (queue full), 499 (client cancel), 503 + Retry-After
  (server-side deadline via `--decision-timeout-ms`).

Measured, legacy shared-context lane (chat already streaming):

| model | decision | largest chat inter-token gap | other-instance chat |
| --- | --- | --- | --- |
| lfm2.5-350m | 270 ms | 269.7 ms | unaffected |
| qwen3.5-2b | 498 ms | 503.0 ms | unaffected |
| gemma-4-e4b | 1005 ms | 1008.3 ms | unaffected |

The stall equals the decision duration and applies only to a decision running on
a chat context (the removed legacy lane). The sidecar executor is the only one
left, so no decision ever stalls chat.

Not guaranteed:

- No cancellation of a decision during teardown. A long decision delays
  `DELETE`/resize on that instance until the abort deadline; then the
  management op may return 503 with the instance intact. Client
  disconnect during a decision is handled promptly (the engine stops), but a
  session capture already in flight completes.
- Session decisions do not cross context windows (in the legacy shared lane).

---

## 6. Routing rules

Placement of a decision is decided by an explicit `target_specified` signal
computed from the ORIGINAL request fields, never from a stamped pool id: the
pool id echoed into a decision body is echo-only and never selects a target.

| request target | result |
| --- | --- |
| stateless decision (no placement) | the decision sidecar executor (the internal `__decision__`) |
| `instance: "X"` | instance X |
| `model: "base:X"` | instance X |
| `model: "base:latest:X"` | instance X |
| `model: "base"` / no target (stateless decision) | the sidecar executor |
| `model: "base:GROUP"` (group) | any free member; refused for session-pinned requests (400) |
| `session_id` or `id_slot` | the owning instance's slot (the decision then runs on the sidecar from an owned token snapshot); cross-instance is refused |
| unknown `model` (stateless) | answered on the sidecar (echo-only) |

On the sidecar executor a stateless `model` is echo-only and never places the
request. See `API.md` section 2.6.

---

## 7. Configuration surface you will use

- Instances: `--instance NAME:ctx=N:parallel=M:group=G[:default]`.
- Decisions: `--decision-seqs N` (N >= 3) with a pool selects the sidecar
  executor; it forces `kv_unified` on the sidecar context only, never on chat
  instances.
- Sidecar executor: `--decision-sidecar-ctx N`, `--decision-sidecar-prebuild`.
- Sidecar admission/timeout: `--decision-timeout-ms N`,
  `--decision-max-queue N`.
- Sidecar warm tier: `--decision-warm-budget-mb N` (KV budget for the resident
  session warm prefixes; 0 = off, default). With a positive budget the sidecar
  keeps recent session turns resident and forks them on a repeat instead of
  re-prefilling; a hit is wire-identical to a miss on every model and only
  qwen's recurrent warm-restore can move the reported concentration by up to ~0.05.
- Session limits: `--decision-session-ttl`, `--decision-session-budget-mb`
  (token-snapshot byte budget for the sidecar store; 0 = unlimited).
- Template and metrics: `--jinja`, `--metrics` (the latter is required for
  `/metrics`; without it the endpoint is 501 by design).
- Tests require GPU offload: `-ngl 99`.

---

## 8. What reliable and performant means here

### 8.1 Test protocol

Build GPU-only (no CUDA/Vulkan on the reference machine), then run:

- `LLAMA_DECISION_TEST_MODEL=<model> ./build/bin/test-decision-engine`
- `tools/server/tests/test_decision_envelope.py`
- `tools/server/tests/test_decision_admission.py`
- `tools/server/tests/test_decision_session_concurrency.py`
- `tools/server/tests/test_decision_accuracy.py`
- `test-server-instances-snapshots`, `test-common-instances`,
  `test-save-load-state`, `test-recurrent-state-rollback`

Use all three models: lfm2.5-350m (hybrid/recurrent), qwen3.5-2b (hybrid),
gemma-4-e4b (dense, sliding-window, reasoning). They exercise different fork
and cache paths, so a green run on one is not a green run.

### 8.2 Acceptance signals

- No crashes, no context shifts, no KV corruption.
- No unexpected 413/422/429/499/529; expected 422s only for genuinely
  over-budget or semantically invalid requests.
- Session churn does not monotonically grow memory: the sidecar holds only
  non-resident token snapshots, so there is no arena to leak (the legacy lane's
  `arena_used` returns to 0 after each non-resident session decision).
- Chat inter-token gap is not decision-attributable on any instance (sidecar
  executor), and only metrics/slots are served during a legacy shared-lane
  yield.
- Other instances stay within their undisturbed latency under decision load.
- Master-vs-branch default path stays identical: with `--decision-seqs` and
  `--instance` absent, `/props`, chat output, and `n_seq_max` match master; the
  only divergence is `/v1/decision` returning 400 "decisions are disabled"
  instead of master's 404.

### 8.3 Interpreting memory

RSS growth during mixed chat+decision load is normally the stock prompt cache
(`cache_ram_mib` default 8192, `cache_idle_slots` on). With `--cache-ram 0` the
same workload is flat and stateless decisions add zero growth. Check that before
suspecting a leak.

---

## 9. Current status and known gaps (2026-09)

Verified working at the merge commit:

- The decision suites, admission and session concurrency pass on all three
  models. The accuracy harness measures the labeled corpus on all three and
  gates the holdout split against per-model floors, so "passes" says something
  about answer quality here; see the quality gap below for what it does not fix.
- `test-decision-engine` passes (lfm 1850, qwen 1845, gemma 1768 assertions).
- The instance and snapshot C++ suites, and save/load (126/126), pass.
- Cross-context host-format state transfer (a capture from one context loading
  into a context with different `n_ctx`/`n_seq_max`) matches to zero logit
  difference on all three models.

> M0-M9 completion (2026-09-29): the simplification roadmap
> `ROADMAP_20260929_SIMPLIFY.md` has landed. The decision sidecar executor is now
> the single decision executor (opt-in via `--decision-seqs` with a pool, the
> default when a pool exists); sessions are eager token snapshots replayed on the
> sidecar (mechanism B), chat contexts carry no decision sequences and are never
> forced into the unified KV cache in sidecar mode; the warm tier
> (`--decision-warm-budget-mb`) and the timeout/queue admission
> (`--decision-timeout-ms`, `--decision-max-queue`) are calibrated; per-instance
> and sidecar VRAM are reported in both `/instances` and `/props`. The old
> shared-context machinery (`session_registry`/`session_store`, the arena/backend
> flags) has been removed from the tree. The
> gates in the checklist are green on all three models; `--cache-ram 0` soak shows
> flat RSS and no sequence drift.

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
1. `test-recurrent-state-rollback` fails on lfm2.5-350m and qwen3.5-2b on CPU
   and GPU. The identical failure reproduces on upstream master, so it is a
   pre-existing upstream bug, not a regression from this merge. Do not treat it
   as a green gate until it is fixed upstream.

The earlier gaps (sidecar default routing, lazy-session invalidation by
idle-slot purging, and session-in-sidecar) are resolved by the M0-M9 work above:
routing now uses an explicit `target_specified` signal, sessions are captured
eagerly, and sessions run on the sidecar as token snapshots.

---

## 10. Reading order for a new agent

1. This file, for the goals and the mental model.
2. `LLAMA_INSTANCES.md`, for the instance pool (windows, routing, adapters,
   snapshots).
3. `docs/decision/API.md`, for the decision wire contract.
4. `docs/decision/README.md`, for how the decision engine uses KV.
5. `docs/decision/OBJECTIVE.md`, for the engine tradeoffs and fork strategy.
6. Source entry points: `tools/server/server-instances.{h,cpp}`,
   `tools/server/server-context.{h,cpp}` (handles, sequence layout),
   `tools/parallel-decision/decision-engine.{h,cpp}`. The legacy-lane session
   registry and store have been deleted.

---

## 11. Glossary

- Instance / context window: a named, independently sized KV + compute context
  built from shared weights, with its own scheduler thread.
- Sidecar: the decision executor - the internal `__decision__` pool instance
  with its own context and scheduler thread. It owns every decision, on a
  single-context server as well as a pool, and is never written or stalled by
  chat.
- Shared context / legacy lane: removed. Decisions ran on the same context as
  chat, so chat stalled for the decision and a concurrent decode could perturb a
  chat token through batch geometry.
- Engine pool: reserved decision sequences inside a context, above the chat
  slots (legacy shared lane, and the sidecar's own context).
- Session arena: reserved sequences used to materialize a retained-turn
  reference for the duration of a session decision (legacy shared lane only;
  the sidecar uses non-resident token snapshots).
- Yield: the mechanism that lets a decision run on the scheduler thread while
  only `/metrics` and `/slots` are answered (legacy shared lane).
- Token snapshot: the owned copy of a completed turn's tokens and adapter scope
  taken from the owning instance and replayed into the sidecar context; the
  sidecar's only session backend.
- Non-resident: a session reference that holds no context cells between
  decisions (token snapshots, and legacy `host` bytes in RAM / `file` on disk).
- Fork: the strategy that gives each branch a copy-on-write view of the parent
  sequence (`copy` / `restore` / `hybrid`, chosen by `select_fork`).

---

## 12. Simplification findings (2026-09-29)

Non-normative. This section records the architecture review that led to
`ROADMAP_20260929_SIMPLIFY.md`. Where it disagrees with the rest of this file,
the simplification target wins for new work; the old behavior stays documented
until the prune phases land. As of the M0-M9 landing, the simplification target
is the current implementation (section 3): the sidecar executor with token
snapshots is the single decision placement, so the shared-context machinery
below is the legacy lane described in sections 2.2/3.2/4.

### 12.1 Branch correction

`_codacus_parallel_decision` is not itself a sidecar. It reserves decision
sequences inside the chat context (`--decision-seqs` grows `n_seq_max`), decodes
on the same scheduler thread, and shares the prompt's KV cells with
`llama_memory_seq_cp`. The `llama-parallel-decision` worker is a separate
process. The "decision sidecar with its own context" called for by the
simplification target does not exist on either source branch; it is new work
built on the instance pool from `_multi_context_adapter_ds`.

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

Harness (removed after the experiment; the recorded results below are kept):
`tools/parallel-decision/bench-snapshot.cpp`, GPU-only (`-ngl 99`), one chat context and
one sidecar context with identical params (`kv_unified`, `swa_full=false`,
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
