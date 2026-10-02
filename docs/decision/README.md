# The decision endpoint

This is the introduction. It says what `POST /v1/decision` is for, how it is
shaped, why it is fast, and where every claim is documented in full.

| document | what it owns |
|---|---|
| `API.md` | **the contract**: routes, the combined request shape, the response envelope, errors, limits, and the environment overrides. Normative. |
| `JEV-API.md` | the upstream Jev "System One" specification, transcribed. Reference only; the conformance gate is written against it. |
| `BENCHMARKING.md` | how to measure and validate: accuracy, speed, stability, and the frozen artifacts that own a number. |
| `OBJECTIVE_MULTI_CONTEXT.md` | the instance-pool and decision-merge goals, and where the design landed. Non-normative. |
| `OBJECTIVE.md` | the historical comparison against a reference decision engine, kept as the reasoning behind several choices. |
| `tools/parallel-decision/README.md` | the engine and the two front-ends, one level down. |

## What it does

Not "generate text", but "pick an answer from a fixed list". You hand the server
an opaque evidence document and a set of typed questions; it returns one
closed-world distribution per question and generates nothing at all
(`usage.output_tokens` is always 0).

```
POST /v1/decision
state + N questions -> N answer distributions, no sampling
```

This is classification work - routing, triage, structured extraction, RAG
ranking - where what matters is which option wins, not whether the prose is
fluent. The endpoint exists to make that cheap and repeatable.

The state is treated as **data**, never as instructions: it is framed as
evidence, `<` is escaped, and the prompt tells the model that state content is
not instructions.

## One endpoint, two front-ends, one envelope

`/v1/decision` serves both request shapes, selected by the body's top-level
shape. A body carrying both is a 400.

| body has | front-end | answered with |
|---|---|---|
| `questions` | the Jev shape: `state` (or `contexts`) plus a map of typed questions | `answers`, keyed by question id |
| `schema` | the generic typed-schema shape: a JSON Schema or compact typed fields | the same `answers` map, keyed by field name, each answer extended with `value` |

The Jev question types are `noul` (binary), `choice` (one of 2-255 options),
`score` (an ordered level on a 2-10 scale), plus the numeric `integer` /
`number` extension over a typed value grid. The generic field types are `enum`,
`boolean`, `integer`, `number`.

Both front-ends compile to the *same* `field_input[]` plan and terminate at the
*same* engine, one envelope parser, one request type. That is structural, not a
convention: there is no second scorer, no second evidence validator and no
second copy of the producer knobs, so the two contracts cannot drift. The
generic front-end earns its place by expressing what the letter readout cannot:
a field wider than the model's realized answer-label pool, and typed value grids
with numeric aggregates.

`POST /v1/systemone` is the strict Jev contract alone and refuses a `schema`
body with a 400 naming `/v1/decision`, so a Jev client works by swapping a base
URL and nothing else. `POST /decision` is a deprecated alias for the canonical
handler.

## Why it is fast

Four things, in order of how much they matter.

1. **One batched forward pass.** The engine lays out every question, every
   candidate and every branch (each candidate's unique tail tokens) as separate
   sequences, decodes them all in a single batched `llama_decode`, and reads one
   scored row per branch. Each branch costs one row of a *shared* full-vocabulary
   matmul rather than a decode step of its own, so the cost grows with the number
   of scored rows instead of with the number of sequential steps - which is why a
   64-option question is not 64 times a 1-option one.
2. **A dedicated executor.** Every decision, stateless and session, runs on the
   internal `__decision__` sidecar executor: its own context and its own
   scheduler thread. Chat contexts are never written, stalled or resized by
   decision work. The engine still reads **full-vocabulary logits** - there is no
   classifier-only context and no answer head - but on a context that chat does
   not share, so a decision cannot corrupt chat KV state and a long decision
   cannot stall a long chat turn.
3. **Cheap branch forking.** Evaluating many branches at once needs many copies
   of the same "so far" context. The branch keeps sequence state on the device
   and stages device-to-device copies on the backend's stream behind a single
   synchronisation, instead of one synchronous copy per tensor, and moves
   transposed cache regions as one bulk strided transfer rather than thousands
   of tiny ones.
4. **Prefix caching.** The framed instructions prefix is cached per request
   identity and reused, so a repeat request does not re-decode it. On a
   recurrent or hybrid model a resident prefix is restored from a host-format
   snapshot, which is far cheaper than a cold prefill; `API.md` section 3.1
   records the measured ratio and the drift that costs.

Plus the bookkeeping that makes it safe to run in production: a peak-KV
preflight that refuses an over-budget request **before** any cache is touched,
admission control, a contract hash that refuses to serve a decision whose
tokenizer or prompt-template identity has changed, an optional calibrated
temperature file with provenance, and a per-model answer-label pool.

## Where the numbers are

There are no headline latencies in this file. Speed, accuracy and stability are
measured, and the numbers live with the method that produced them in
`BENCHMARKING.md` and in the frozen artifacts under `tests/decision-baseline/`.
A number quoted without its model, quantization, prompt template and backend
flags is not comparable to any other number, which is why this document quotes
none.

## Live-session decisions

A request carrying `id_slot`, or a first-class `session_id`, is answered about a
chat slot that already holds decoded state. The pool takes an **owned token
snapshot** of that slot's completed turn - the token list plus the enabled
adapter scope - through a read-only op on the owning instance's scheduler, so no
KV pointer, sequence id or context handle leaves that instance. The sidecar
replays the owned tokens into its own context and scores there. The source slot
is only ever read, never forked and never written.

Later decisions in the same turn reuse the snapshot, so the answer survives the
origin slot being cleared and reused by `cache_idle_slots`. One retained turn
per slot; the opaque `turn` tag pins it and a mismatched `turn` is a 422, never
a silent answer about a different turn. A `session_id` outlives the reused
`id_slot` with a create / get / patch / erase lifecycle over `/v1/session`,
which also owns the store's byte budget and TTL policy.

## What it will refuse

422 for a semantic or capacity error, 413 for an over-size body, 429/529 and 503
with `Retry-After` for a full queue or a transient failure, 499 when the
client disconnects, 501 when decisions are not enabled or the
loaded model cannot serve them at all (no usable single-token answer
labels, or a pinned contract that does not match), and 500 for an internal
failure. The path never truncates: an over-limit request is rejected, never
silently clipped. `API.md` section 4 owns the full table.

Two rules the rest of the system is built on:

- **Producer concentration never gates anything.** `confidence` and `certainty`
  are pure functions of the returned `probabilities`. They describe how peaked an
  answer distribution is, not whether it is right - a model can be maximally
  confident and wrong. They never gate admission, caching, routing or
  persistence. Everything that does gate is on the other axis: fork-state
  equality, capacity preflights, adapter-scope application, snapshot identity,
  memory budgets.
- **Probabilities are conditional on the options you supplied.** They do not
  measure the chance that every option is wrong. Say so in every model card and
  every API document.