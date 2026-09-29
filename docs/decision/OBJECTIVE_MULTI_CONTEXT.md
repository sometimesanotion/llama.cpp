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

### 2.2 Chat vs decision inside one context

Inside one context there is a second layer: disjoint ranges of sequence ids.

- chat slots: `[0, n_parallel)`
- decision engine pool: `[n_parallel, n_parallel + n_seq_decision)`
- session registry arena: `[n_parallel + n_seq_decision, ... + n_seq_arena)`

A decision on a shared context runs inside `queue_tasks.yield_to_queue(...)` on
that context's scheduler thread. While it yields, the scheduler declines every
task type except `SERVER_TASK_TYPE_METRICS` and `SERVER_TASK_TYPE_SLOT_GET`;
declined tasks are parked and replayed FIFO after the yield. A CANCEL posted
during the yield still removes its target.

The hard contract that follows: **chat on the same context is stalled for the
whole decision duration.** This is deliberate, not a bug. Other instances are
unaffected. See section 5 for measured numbers.

`--decision-seqs N` (N >= 3) enables the engine and forces `kv_unified`.
`--decision-arena-seqs` sizes the session arena (default `n_parallel`).

---

## 3. Two ways to place decisions

### 3.1 Shared context (default)

Stateless and session decisions run on the routed instance's context.

- Cost: no extra context, no extra compute buffer. Uses the engine's reserved
  sequences and, for sessions, transient arena cells.
- Behavior: chat on that instance stalls for the decision; chat on other
  instances does not.
- Good when: decisions are short relative to chat inter-token time, or when the
  instance has no concurrent chat, or when saving VRAM matters.

### 3.2 Sidecar context (`--decision-instance NAME`)

A dedicated instance, with its own context and scheduler thread, receives
stateless decisions that name no instance.

- Cost: one extra context plus compute buffer once it is built.
- Behavior: chat elsewhere is not stalled by those decisions. Decision latency
  can be worse because the sidecar competes with chat contexts for the GPU.
- Constraint: `NAME` must be an exact configured instance; a group is refused at
  startup.
- Session decisions are currently pinned to the owning instance and never
  group-routed. Cross-context session transfer is proven feasible but not
  implemented (section 9).

Measured shape (4 agents, 4 stateless workers, 1 session worker, 2 churn
workers, all three target models):

- Small model (lfm2.5-350m): sidecar removes the decision-induced chat tail
  (chat max gap 611 ms -> 166 ms) at the cost of slower decisions.
- Mid model (qwen3.5-2b): sidecar greatly improves chat tail (max
  1641 ms -> 280 ms) but decisions are roughly 1.5-2x slower.
- Larger model (gemma-4-e4b): device-bound; sidecar gives little chat benefit
  and slows decisions. Four contexts contending for one GPU is the bottleneck.

So the shared-vs-sidecar choice is a policy decision, not a correctness one.

### 3.3 Session decisions

A request may carry `id_slot` or a first-class `session_id` to answer about a
turn the slot already decoded, instead of re-prefilling the transcript.

- On the first decision for a turn, the server captures the turn into an owned
  reference. Backends: `host` (bytes in RAM, non-resident), `clone` (metadata
  cell reference; not available on sliding-window models), `file` (on disk).
- Later decisions fork the reference rather than the live slot, so the source
  slot can be cleared and reused.
- Identity is content hash + turn + adapter scope + memory epoch. A changed
  turn or scope is 422; a stale memory epoch is 409.
- The source slot is read-only; the source slot's KV is never written by the
  decision.

---

## 4. Sequence and KV layout in one context

```
one context (one instance) with --decision-seqs 4, --parallel 1, arena default 1

seq 0                     chat slot 0
seq 1 .. 4                decision engine pool (seq 1 is the persistent prefix snapshot)
seq 5                     session arena (materialized only during a session decision, host/file)
```

- Weights are shared across all instances; each instance has its own KV cells.
- A stateless decision prefills into the engine pool, scores, then removes its
  transient sequences. Only the prefix snapshot survives for reuse.
- A `host`/`file` session reference holds no cells between decisions; an arena
  sequence is allocated for the decision and freed on release.
- A `clone` session reference shares the source attention cells and pins them
  for the session lifetime (opt-in).
- A capacity preflight estimates peak use and returns 422 rather than partially
  overwriting the cache; `llama_decode` returning 1 is also mapped to 422.

---

## 5. The concurrency contract

Guaranteed:

- Decisions never corrupt chat KV: disjoint sequence ranges, a read-only source
  slot, cleanup on every exit, and admission limits.
- Only `/metrics` and `/slots` are served in a context while a decision yields.
- Declined tasks are parked and replayed FIFO, and are not lost or reordered.
- Admission bounds concurrent decisions, not their duration: codes 413 (body too
  large), 429 / 529 (queue full), 499 (client cancel).

Measured (same-context, chat already streaming):

| model | decision | largest chat inter-token gap | other-instance chat |
| --- | --- | --- | --- |
| lfm2.5-350m | 270 ms | 269.7 ms | unaffected |
| qwen3.5-2b | 498 ms | 503.0 ms | unaffected |
| gemma-4-e4b | 1005 ms | 1008.3 ms | unaffected |

The stall equals the decision duration. Plan agentic workloads accordingly:
short frequent decisions on a shared context, or a sidecar, or a dedicated
instance per decision-heavy agent.

Not guaranteed:

- No cancellation of a decision during teardown. A long decision delays
  `DELETE`/resize on that instance until the abort deadline; then the
  management op may return 503 with the instance intact. Client
  disconnect during a decision is handled promptly (the engine stops), but a
  session capture already in flight completes.
- Session decisions do not cross context windows.

---

## 6. Routing rules

| request target | result |
| --- | --- |
| `instance: "X"` | instance X |
| `model: "base:X"` | instance X |
| `model: "base:latest:X"` | instance X |
| `model: "base"` / no target (stateless decision) | `--decision-instance` if set, else default/sole instance, else 400 ambiguous |
| `model: "base:GROUP"` (group) | any free member; refused for session-pinned requests (400) |
| `session_id` or `id_slot` | the owning instance's slot; cross-instance is refused |
| unknown `model` | 400 model not found |

Note: `handle_post_decision` stamps the pool id onto an untargeted body, so a
bare pool id must be treated as "no target" for the sidecar default to work.
See the status note in section 9.

---

## 7. Configuration surface you will use

- Instances: `--instance NAME:ctx=N:parallel=M:group=G[:default]`.
- Decisions: `--decision-seqs N` (N >= 3, forces unified KV).
- Session arena: `--decision-arena-seqs N`.
- Sidecar: `--decision-instance NAME`.
- Session backend: `--decision-session-backend host|clone|file`.
- Session limits: `--decision-session-ttl`, `--decision-session-budget-mb`,
  `--decision-session-persist DIR`.
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
- `arena_used` returns to 0 after each non-resident session decision; session
  churn does not monotonically grow memory.
- Same-context stall matches the decision duration and only metrics/slots are
  served during the yield.
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

- The decision suites, admission, session concurrency, and accuracy harness pass
  on all three models.
- `test-decision-engine` passes (lfm 1816, qwen 1811, gemma 1734 assertions).
- The instance and snapshot C++ suites, and save/load (126/126), pass.
- Cross-context host-format state transfer (a capture from one context loading
  into a context with different `n_ctx`/`n_seq_max`) matches to zero logit
  difference on all three models.

Known gaps, in rough priority order:

1. Sidecar default routing needs a one-line fix: a bare pool id inside an
   untargeted decision body currently prevents the `--decision-instance` fill,
   so the request 400s "ambiguous" or goes to the default instance. The fix is
   to treat a bare pool id as "no target" in the decision dispatch. (A local
   validated fix exists in this working tree; confirm it is in your base.)
2. Lazy sessions are invalidated by stock idle-slot purging: with the default
   policy a session is captured on first use, and any later task on any slot
   saves and clears idle slots (`cache_idle_slots`), so the first decision can
   return 422 "no decoded state". Create with
   `policy.capture_on_turn_complete=true`, or make eager capture the default.
3. `--decision-session-persist DIR` concatenates `DIR` and the filename without
   a path separator, so files land next to the directory. `--slot-save-path` is
   normalized; this flag is not.
4. A failed session materialize pins one arena sequence until the session is
   deleted (the error path does not free the sequence). Low severity, but it can
   accumulate under hostile or corrupt inputs.
5. Session-in-sidecar is the main open design question. Cross-context transfer
   is proven, so it is implementable: capture on the owning context's thread,
   transfer bytes, materialize and decode on the sidecar, applying the captured
   adapter scope. Today sessions are pinned to the owning context.
6. `test-recurrent-state-rollback` fails on lfm2.5-350m and qwen3.5-2b on CPU
   and GPU. The identical failure reproduces on upstream master, so it is a
   pre-existing upstream bug, not a regression from this merge. Do not treat it
   as a green gate until it is fixed upstream.

---

## 10. Reading order for a new agent

1. This file, for the goals and the mental model.
2. `LLAMA_INSTANCES.md`, for the instance pool (windows, routing, adapters,
   snapshots).
3. `docs/decision/API.md`, for the decision wire contract.
4. `docs/decision/README.md`, for how the decision engine uses KV.
5. `docs/decision/OBJECTIVE.md`, for the engine tradeoffs and fork strategy.
6. Source entry points: `tools/server/server-instances.{h,cpp}`,
   `tools/server/server-context.{h,cpp}` (yield, handles, sequence layout),
   `tools/parallel-decision/decision-engine.{h,cpp}`,
   `tools/parallel-decision/session-registry.{h,cpp}`,
   `tools/parallel-decision/session-store.{h,cpp}`.

---

## 11. Glossary

- Instance / context window: a named, independently sized KV + compute context
  built from shared weights, with its own scheduler thread.
- Sidecar: an instance dedicated to stateless decisions via
  `--decision-instance`.
- Shared context: decisions run on the same context as chat, so chat stalls for
  the decision.
- Engine pool: reserved decision sequences inside a context, above the chat
  slots.
- Session arena: reserved sequences used to materialize a retained-turn
  reference for the duration of a session decision.
- Yield: the mechanism that lets a decision run on the scheduler thread while
  only `/metrics` and `/slots` are answered.
- Non-resident: a session reference that holds no context cells between
  decisions (`host` bytes in RAM, `file` on disk).
- Fork: the strategy that gives each branch a copy-on-write view of the parent
  sequence (`copy` / `restore` / `hybrid`, chosen by `select_fork`).
