# Jev-Compatible Decision API (`/v1/decision`) - Specification

Status: implemented. This is the normative contract: routes, request, response,
errors, and limits. Non-normative design notes, the backend survey, and the
benchmark narrative live in `README.md`.

The endpoint naming is the one deliberate difference from Jev, and it is
settled:

- `POST /v1/decision` is the canonical route.
- `POST /decision` is a deprecated alias for the same handler.
- `/v1/systemone` is Jev's own public path. It is NOT registered on this
  server. "Jev-compatible" means the request and response fields and their
  semantics match, so a client written for Jev is pointed here with only a
  base-URL/path change.

Other locked shapes: dual readout (letter labels for Jev, trie for generic
schemas); adapters strictly optional; temperature `T=1.0` default with a
per-type override; fork by `llama_memory_seq_cp` for attention plus a
`PARTIAL_ONLY` byte copy of the recurrent state, with a full
`llama_state_seq` save/restore as the fallback; probabilities are closed-world
and never gated on confidence.

---

## 1. What the API is

One POST. No generation. A caller supplies opaque `state` (evidence, treated
as DATA, never as instructions) plus `DECISION_MIN_QUESTIONS` to
`DECISION_MAX_QUESTIONS` independently-scored typed questions. The server
returns one closed-world distribution per question, assembled by code.
`output_tokens` is always 0; no text is sampled.

```
POST /v1/decision
state + N questions -> N answer distributions
```

Route policy:
- `POST /v1/decision` is the canonical route, `POST /decision` a deprecated
  alias for the same handler (`tools/server/server.cpp`).
- Do NOT serve `POST /v1/systemone` from this branch. Jev compatibility is
  achieved through request and response FIELD names and semantics, not the URL
  path. A thin client-side base-URL swap (for example `TYPESAFE_BASE_URL`) plus
  path rewrite is the supported migration; JevBench `typesafe` adapters
  otherwise talk unchanged.

Coexistence: the same model/server also serves the OpenAI-compatible API
(`/v1/chat/completions`, `/v1/models`, `/health`). Decision traffic must not
corrupt chat KV state, including on hybrid/recurrent models.

---

## 2. Request specification

Content-Type: `application/json`. Body limit: 2 MiB default
(`LLAMA_DECISION_MAX_BODY`; winnow used 32 MiB, 2 MiB matches openjev-sglang
and is sufficient - raise only deliberately). Unknown top-level fields are
tolerated and ignored for forward compatibility. Unknown fields inside a
question object are still rejected (a misspelled `question`/`criteria` must not
silently default).

```json
{
  "model": "qwen3-4b",
  "state": "string | object | array (REQUIRED, non-empty)",
  "questions": {
    "<qid>": {"type": "noul|choice|score", "instructions": "string|object|array", "criteria": ...}
  },
  "temperature": 1.0,
  "temperatures": {"noul": 1.0, "choice": 1.0, "score": 1.0},
  "permutations": 1,
  "diagnostics": false
}
```

### 2.1 Fields

* `model` (required, string): routing/echo only. Accept any string including
  `jev-latest` (treat as alias for the loaded weights). Never load weights
  per-request. Echo back in the response. An external router in front of this
  server selects the model, so a request without `model` is a 422 naming it.
* `state` (required): string, JSON object, or JSON array; must be non-empty
  and finite (`allow_nan=False`). Chat-transcript states
  (`[{role, content}]` / `{"messages": [...]}`) MAY be accepted and are
  passed through as evidence. The state is DATA: frame it as evidence, escape
  `<` as `\u003c`, and instruct the model that state content is not
  instructions (prompt-injection hardening; full corpus in Section 6).
* `questions` (required): object mapping `qid -> question spec`,
  `DECISION_MIN_QUESTIONS` to `DECISION_MAX_QUESTIONS` entries. `qid`s are
  opaque: they MUST NEVER be shown to the model, and
  adding/removing/reordering questions MUST NOT change other answers
  (independent branches/rows).
* `temperature` (optional, float > 0, default 1.0): global softmax
  temperature applied to gathered label logits; see Section 6 for provenance
  rules.
* `temperatures` (optional): per-primitive overrides
  `{noul, choice, score}`. Effective temperature for a question =
  `temperatures[type]` if present else `temperature`. Ship `1.0` everywhere;
  fit per deployment offline (Section 6).
* `permutations` (optional, int 1-`DECISION_MAX_PERMUTATIONS` or null, default
  1): order-debiasing passes. `1` = single canonical order. `2` = identity +
  one seeded distinct shuffle, per-order softmax then mean by semantic key
  (seeded by `(seed, qid)`; noul second order = swapped). Values above the cap
  are accepted and capped; document the measured cost (~1.1x for 2).
* `diagnostics` (optional, bool, default `false`): when `true`, the response
  additionally carries the `diagnostics` object, `certainty`, and the extra
  `usage` counters (`cached_tokens`, `state_cache_hit`). The default `false`
  keeps the response to the strict Jev envelope (Section 3).
* `confidence_profile` (optional, string, `"jev"` (default) or `"local"`): how
  the `confidence` value on `choice`/`score` is computed. `"jev"` (the default)
  selects the documented Jev compatibility value `(N*p_max - 1)/(N - 1)`, clamped
  to [0,1], a rescaled winner share built from `certainty`; it matches Jev's
  official Choice confidence and is conservative at the low end (uniform -> 0),
  so it is the right basis for fallback gating. Jev leaves the Score definition
  open above three levels, so the same monotone rule is the stated local value
  there. `"local"` selects `1 - H/log K` (normalized inverse entropy), which reads
  the whole distribution and calibrates better on some model families (for example
  Gemma). The profile changes only the reported concentration, never an answer or
  a probability; it never gates anything on its own.

Server flag: `--decision-permutations N` (env `LLAMA_ARG_DECISION_PERMUTATIONS`,
default 1) sets the pass count for requests that omit `permutations`; an explicit
request field always wins and the pass cap still applies.

Server executor: when `--decision-seqs` is set, every decision runs on the
decision sidecar executor - an internal, undeletable pool instance with its own
context and scheduler thread, built on first decision demand. A single-context
server gets one too, so a decision never runs on a chat context. Chat contexts
are never written, stalled, or resized by decision work. The sidecar is the only
executor.

Local-only notes: `model` is required here (the external router selects it) and echoed back
verbatim, matching Jev. `GET /v1/models` keeps the OpenAI model-list shape
(`{"object":"list","data":[...]}`), not Jev's `{"models":[...]}` shape.

### 2.2 Question types (canonical names)

Accept `bool` as an alias for `noul` and `scale` as an alias for `score` on
INPUT; always emit canonical `noul`/`choice`/`score` on output.

* `instructions` REQUIRED and non-null on all three types (string, object, or
  array); a question with null or missing `instructions` is a 422 naming
  `instructions`.
* `noul`: binary judgment.
  `criteria` optional: `{"true": "desc", "false": "desc"}` (descriptions may
  be strings; treat missing as null desc). Internally two options
  `["false","true"]` (or lettered `A:yes/B:no` - equivalent after mapping).
* `choice`: categorical selection.
  `criteria` REQUIRED: object `{key: desc|null}`, keys in object
  order. Model sees description, or key when null. Answer uses the KEY.
* `score`: ordered rating, zero-based.
  `criteria` REQUIRED: ordered array of `DECISION_MIN_OPTIONS` to
  `DECISION_MAX_SCORE_LEVELS` level descriptions, low -> high (equivalently
  `{"0": desc, ...}` legend dict on input). Model sees each level; answer is
  the expected index.

Every count limit is listed once in Section 5 and owned by a single constant;
the validator, the response, and this document read the same value. Option
IDs/keys must be unique. Over-limit requests are rejected (never truncated).
Reject empty `criteria` where required. Structured `instructions`/`criteria`
values are rendered into the prompt, never silently stringified; `legend`
echoes the ORIGINAL structured values so they round-trip.

Each option line is rendered by `format_option_line` (one function) as
`label: <key> - <description>`, where the label is the single letter the model
answers with; when the rendered description is empty the line is
`label: <key>` only (no trailing separator). Keys are `false`/`true` for noul,
the option names for choice, and `"0".."K-1"` for score.

### 2.3 Unified shape: questions + state or contexts

The wire converges on Jev's shape: a map of typed questions (`noul` /
`choice` / `score`), answered against one `state` (Jev) or, as this branch's
extension, against a list of `contexts` in one batched pass. The numeric types
(`integer` / `number`) are this branch's extension over Jev: a range generates a
typed grid, answered with the winning value and an optional scalar `aggregate`.

```json
{"state": "...", "questions": {"qid": {"type": "noul|choice|score|integer|number",
    "instructions": ..., "criteria": ...}},
 "temperature": 1.0, "temperatures": {"noul": 1.0, "choice": 1.0, "score": 1.0,
                                      "integer": 1.0, "number": 1.0},
 "confidence_profile": "jev|local", "permutations": 1}
```

* `questions`: 1-`DECISION_MAX_QUESTIONS` entries, Jev-typed. Each question
  carries its own `instructions` (string/object/array) and `criteria`
  (noul `{true,false}` map, choice option->description map up to the label-pool
  cap, score ordered-level array). Question keys are the answer keys.
* Numeric questions: `type: "integer"` needs integer `minimum` and `maximum`;
  `type: "number"` needs numeric `minimum`, `maximum` and `step` (or
  `multipleOf`, not both). The bounds define a 2-`DECISION_MAX_NUMERIC_VALUES`
  ascending grid, shown to the model and scored exactly like a choice. The grid
  must include both ends (`multipleOf` must divide the range). An optional
  `aggregate` (`"mode"` default, `"median"`, `"mean"`) adds a scalar summary to
  the answer; it never changes `value`.
* Evidence: exactly one of `state` (string/object/array, Jev) or `contexts`
  (1-`DECISION_MAX_CONTEXTS` non-empty strings, answered in order against the
  same questions). Session fork scores exactly one context.
* `temperature` (global, float > 0, default 1.0) with per-type `temperatures`
  overrides (the map accepts `integer` and `number` keys too); `confidence_profile` (`"jev"` default / `"local"`); `permutations`
  (1-8, order-de-biasing, capped, not on a session fork). All match Section 2.1
  semantics.
* Response, single `state`: `{model, answers: {qid: {...}}, usage, timings}`.
  `answers` is the strict Jev envelope (Section 2.2); `timings` is additive.
* Response, `contexts`: `{model, contexts: [{answers, usage}, ...], timings}`.
  The per-context `answers`/`usage` match the single-state shapes; `timings` is
  batch level. The `contexts` key is the one extension over Jev.

This is one of two request front-ends served on `/v1/decision`; the other is
the generic typed-schema form (Section 2.5). A body is dispatched on its
top-level shape: `questions` selects the Jev front-end, `schema` selects the
generic front-end, and a body carrying both is a 400. Both terminate at the
same engine, so the two shapes cannot drift.

### 2.4 Live-session request (`id_slot`, `session_id`, `session_pos`, `turn`)

The unified shape accepts an optional `id_slot` (int) so the decision is answered about
a chat slot that already holds decoded state, without re-prefilling the
transcript. A first-class `session_id` (string) is the alternative: a server-side
session handle that owns its slot and turn, decoupled from the reused `id_slot`.
`session_id` and `id_slot` are mutually exclusive; a body carrying both is a 400.
`session_pos` (int) optionally pins the continuation position; it
requires a session and must equal the session's next position exactly, otherwise
the request is a 422 (a shifted position is never scored silently). `turn` (string)
optionally pins the retained-turn identity; a mismatched `turn` is a 422, never a
silent answer about a different turn.

* The slot must exist, not be released, and hold decoded state. These are
  capability checks only: an unknown, released, or empty slot is a 4xx with a
  reason. The server never inspects a confidence value to accept or reject a
  session.
* A session scores exactly one context: a `contexts` request with more than one
  entry is a 400. The readout appends the questions as a fresh user turn
  rendered through the slot's chat template; the transcript is not re-injected
  as `state`, and the request `state` is still required by the shape but not
  re-decoded.
* Session readout runs full-logits on the decision sidecar executor's context,
  like the stateless path. There is no classifier context and no answer head.
* A session fork is exact: `copy` is refused on a recurrent/hybrid model (it
  would not reproduce the slot state), `hybrid`/`restore`/`auto` are accepted.
* The source slot is never mutated: a decision reads it only to take an owned
  token snapshot (a bounded copy of the completed turn's tokens and adapter
  scope). After the request, chat on that slot and a repeat decision both keep
  working.
* The response reports the fork additively: `session_fork: true`, `source_slot`,
  and `session_pos` (and `turn` when the request set one). With
  `diagnostics: true`, `diagnostics.permutations` reports the pass count used.

#### First-class session handles (`/v1/session`)

A client may create a server-side session that outlives the slot's current turn
and survives the slot's KV being cleared and reused:

- `POST /v1/session` - create. Body: `{"id_slot": N, "turn": "...",
  "policy": {"pinned": bool, "ttl_ms": N, "max_turns": N}}`. Returns
  `session_id`, `id_slot`, `turn`, `backend` (`"tokens"`), and `captured`.
  `id_slot` alone keeps working byte-identically; `session_id` is additive.
  Capture is eager: the completed turn's tokens and adapter scope are copied out
  of the owning instance at create, so there is no lazy window for an idle-slot
  purge to invalidate. `capture_on_turn_complete` is accepted as always true.
  The legacy `host`/`clone`/`file` backends are not available on the sidecar
  executor: `clone` and `file` are a 501 capability refusal, never a silent
  fallback, and `host` is accepted as the token snapshot.
- `GET /v1/session/{id}` - status: `session_id`, `id_slot`, `turn`, `backend`
  (`"tokens"`), `pinned`, `ttl_ms`, `captured`, `bytes`, `created_ms`,
  `last_used_ms`, and the trigger counters (`n_snapshots`, `n_reuses`,
  `n_releases`, `n_sessions`, `bytes_total`). The arena counters
  (`arena_used`/`arena_capacity`) are removed: a token snapshot holds no reserved
  sequence between decisions, so there is no arena to report.
- `PATCH /v1/session/{id}` - set `pinned` and/or `ttl_ms`.
- `DELETE /v1/session/{id}` - erase, releasing the owned reference.

`session_id` is mutually exclusive with `id_slot` in a decision request (400 when
both appear). A session handle is bound to one source slot; when the slot's turn
ends the session ends with it.

#### Session substrate

A session decision never forks the live slot. The pool keeps a decision-owned
token store keyed by (instance, slot, turn) that owns one eager token snapshot
per retained turn:

- The snapshot is taken through a read-only pool op on the owning instance's
  scheduler: it copies the completed turn's tokens and the enabled adapter scope
  (path/scale list). Only owned copies cross the boundary - no KV pointer,
  sequence id, or context handle leaves the owning instance. This is the on-demand
  capture trigger: a turn that is never queried produces no reference. Later
  decisions in the same turn reuse the snapshot instead of the slot, so the source
  survives the slot's KV being cleared and reused by `cache_idle_slots`.
- The retained-turn policy keeps one reference per slot. When the slot decodes
  past the reference's position (a completed new turn), the reference is released
  before the next decision. A slot with no decoded state (no `pos_max`) cannot
  have advanced, so a cleared slot still matches its reference.
- A session decision runs on the sidecar executor: it re-prefills the owned
  token list into the executor's own context (mechanism B, token replay) and
  scores the question head there. The adapter scope is resolved from the pool
  adapter registry and applied to the executor context, so K/V and queries are
  conditioning-consistent with the captured scope. A stateless decision stays
  scoped to the base model. Diagnostics report the scope actually used in
  `adapter_scope`.
- A decision on an in-flight slot (`is_processing()`) is a 422: it measures
  task validity (is the turn complete), never confidence.
- Identity is a strong content hash over the token list plus the adapter scope,
  the turn counter, and the source slot. A request whose pinned `turn` or
  `session_pos` does not match the snapshot is a 422; a session whose source
  turn advanced is stale (422), never answered from an old turn. A token
  snapshot is model-epoch independent: a model reload does not make it stale.
- Lifecycle: a reference is released by `SLOT_ERASE`, by the slot's release
  callback, by an expired TTL (`ttl_ms`, reaped only when a session is older than
  its TTL and never when pinned or held by an in-flight decision), and, under
  byte-budget pressure (`--decision-session-budget-mb`), by LRU eviction of the
  least-recently-used unpinned, unleased reference. A pinned or in-flight
  reference is never evicted. The default budget is unlimited and the default TTL
  is 0 (no expiry), so a deployment that never sets them sees no eviction.
- Window persistence: `POST /slots/{id}?action=save` co-writes a session manifest
  sidecar next to the slot file when a session exists. The manifest is bound to
  the slot file by its content hash and carries the session identity, policy,
  backend, and the reference state. `POST /slots/{id}?action=restore` reads the
  sidecar: a matching manifest rebinds the session to the restored window, a
  mismatched or foreign one is unresolvable (dropped, never served), and a slot
  restored without a sidecar drops any retained reference - a restore never
  leaves a stale retained turn. The slot file format is unchanged.

### 2.5 Generic typed-schema front-end (`schema`)

The second producer of engine fields, additive to the Jev contract. A body with
a top-level `schema` (and no `questions`) selects this front-end; both present
is a 400. The generic shape accepts either compact typed fields or a JSON
Schema object with `properties`:

```json
{"model": "qwen3-4b", "instructions": "Answer each field.",
 "schema": {"category": {"type": "enum", "choices": ["billing","technical"],
                         "description": "Ticket type"},
            "urgent": {"type": "boolean", "description": "Needs urgent handling?"},
            "count": {"type": "integer", "minimum": 1, "maximum": 5}},
 "state": "Customer asks for a refund of $42, order arrived broken."}
```

* Field types: `enum` (choices list), `boolean`, `integer` (integer
  `minimum`/`maximum`), `number` (numeric `minimum`/`maximum`/`step`, or
  `multipleOf` in a JSON Schema). Each field is compiled to the same
  `field_input` the Jev front-end produces, and scored by the same engine.
  Numeric fields take `aggregate` (`mode` default, `median`, `mean`).
* Response: `{model, results: [...], usage, timings}`. Each result matches one
  context (or the single `state`): `{decision: {field: value, ...},
  fields: {field: {value, probability, scored_nodes, tree}}, usage}`. Numeric
  fields add `interval_p10_p90` and `aggregate` to their field record.
* Options: `mode` (`auto` default, `tree`, `greedy`), `tree_max` (default 128),
  `cache_prompt` (default true). Evidence (`state`, `contexts`, or a session
  `id_slot`/`turn`) is orthogonal to the front-end and behaves as in Section 2.3.
* The generic front-end reuses the Jev engine, label pool, temperature and
  permutation machinery; it never re-implements field compilation or branch
  scoring.

### 2.6 Multi-instance routing

Every decision runs on the decision sidecar executor (the internal
`__decision__` instance), so decision traffic never writes, stalls, or resizes a
chat context. Placement is decided from the ORIGINAL request fields, never from
a stamped pool id; the pool id echoed into a decision body is echo-only and never
selects a target. The `instance` / `model` fields keep their routing role for the
parts that still need a target:

* A **stateless** decision carries no placement: the request is dispatched to
  the sidecar executor directly. `model` is echo-only and never selects a
  placement target. A bare pool id, a Jev alias (`jev-latest`/`jev-preview`), or
  an unknown model all answer on the sidecar; `model` itself is still REQUIRED by
  the contract (a body without it is a 422). `instance` is likewise echo-only on
  the stateless sidecar path.
* A **live-session** decision (`id_slot`/`session_id`) names the instance that
  owns the source slot (never a group, 400). The pool takes an owned token
  snapshot of that slot's completed turn through a read-only op and the decision
  then runs on the sidecar executor; the source slot is never decoded by the
  decision.
* `/v1/session` create names the owning instance (never a group, 400); a session
  handle is instance-local, so get/patch/delete must name its owning instance.

Sidecar executor flags (server side):

* `--decision-sidecar-ctx N` - sidecar context size (default 0 = the largest
  effective instance window, so the longest turn a chat instance can produce
  still replays; a compiled plan is sized by the request, so no smaller fixed
  default is safe).
* `--decision-sidecar-prebuild` - build the sidecar context eagerly at startup
  instead of lazily on the first decision.
* `--decision-timeout-ms N` - server-side deadline for a whole decision (default
  0 = none); on expiry the request answers 503 + Retry-After, never a partial
  answer.
* `--decision-max-queue N` - concurrent decision cap (default 4); beyond it the
  request answers 429, above twice it 529. `LLAMA_DECISION_MAX_BODY` and the
  env-var overrides still apply.
* `--decision-warm-budget-mb N` - KV budget (mebibytes) for the sidecar's resident
  session warm prefixes (default 0 = the warm tier is off). With a positive budget
  the sidecar reserves warm sequences (roughly `budget / (per-cell bytes x n_ctx)`,
  capped at 8); the first decision on a session turn cold-prefills it into a kept
  resident sequence and a follow-up decision on the same turn forks that prefix
  instead of re-prefilling. A hit is wire-identical to a miss on every model; on the
  qwen hybrid the recurrent warm-restore can drift the reported score concentration
  by up to ~0.05 (the documented producer-numerics matter), never the winner. The
  budget is a rough VRAM cap, not exact accounting.

---

## 3. Response specification

The default response is exactly the Jev envelope:

```json
{
  "model": "<echo>",
  "answers": {
    "<qid>": {"type": "noul", "noul": 0.0-1.0}
            | {"type": "choice", "choice": "<key>",
               "probabilities": {"<key>": p, ...},
               "confidence": c}
            | {"type": "score", "score": 0.0-(K-1),
               "probabilities": {"0": p, ...}, "legend": {"0": "desc", ...},
               "confidence": c}
            | {"type": "integer"|"number", "value": <typed grid value>,
               "probabilities": {"<value>": p, ...},
               "confidence": c, "aggregate": s}
  },
  "usage": {"input_tokens": N, "output_tokens": 0}
}
```

With `diagnostics: true` the same answers are returned with additive fields:
`certainty` on choice/score/numeric, the `diagnostics` object, the extra `usage`
counters, the `timings` object, and the score/numeric spread summaries
(`median`, `interval_p10_p90`). The answers themselves are byte-identical
either way.

```json
{
  "usage": {"input_tokens": N, "output_tokens": 0,
            "cached_tokens": M, "state_cache_hit": true|false},
  "timings": {"prefill_ms": ..., "scoring_ms": ..., "total_ms": ...,
              "rounds": ..., "rows": ...}
}
```

### 3.1 Semantics (normative)

* `noul.noul = P(true)` in [0,1]. No `probabilities`/`confidence` required.
  `noul` MUST NEVER carry `confidence`.
* `choice.choice = argmax(probabilities)`; `probabilities` keyed by option
  KEY, sums to 1.
* `score.score = sum(i * p_i)` (expected zero-based index, float in
  [0, K-1]); `probabilities` keys are STRINGS `"0".."K-1"`; `legend` echoes
  input criteria in order with string keys.
* `integer`/`number`: `value` is the winning grid value as a typed JSON number
  (mode); `probabilities` keys are the grid values rendered at fixed width
  (no float noise), sums to 1. The optional `aggregate` is the scalar summary
  of the same distribution when the question set one: `mean = sum(p_i * v_i)`,
  `median` is the value-space weighted quantile at 0.5, `mode` is `value`.
  `aggregate` is additive and never changes `value`.
* `confidence = (N*p_max - 1)/(N - 1)`, clamped to [0,1] (the default). This is
  Jev's documented Choice confidence, a linear rescale of the winner's share
  `certainty = max(p)`: uniform -> 0, one-hot -> 1. Because it is built from the
  winner's share and is conservative at the low end, it is the recommended basis
  for fallback gating. Jev leaves the Score confidence above 3 levels
  undocumented, so the same monotone rule is the stated local value there.
  `confidence_profile: "local"` selects `1 - H(p)/log(K)` (normalized inverse
  entropy) instead, which reads the whole distribution and calibrates better on
  some families; the profile changes only the reported number. Clamp to [0,1].
* `certainty = max(p)` (the winner's share). `confidence` is always returned on
  `choice`, `score` and numeric questions; `certainty` is additive and returned
  when `diagnostics` is true. Both measure concentration of the answer
  distribution; they are not
  calibrated correctness and never gate admission, caching, routing, or
  persistence on their own.
* Score `median` and `interval_p10_p90` (skew-robust spread summaries) are
  additive and only present when `diagnostics: true`; the default score answer
  is the strict Jev `{type, score, probabilities, legend, confidence}` shape.
  Numeric answers report the same two summaries, over the grid values, under
  `diagnostics: true`.
* The `timings` object is additive and only present when `diagnostics: true`
  (or on a session fork, which reports its diagnostics additively).
* `probabilities` are CONDITIONAL on the supplied options (they are a
  conditional option score, not a calibrated correctness). `confidence`/
  `certainty` measure CONCENTRATION, not correctness. Document this in every
  model card and API doc.
* `usage.output_tokens` MUST be 0 (warmup/branch tokens are accounting-only).
  `usage.input_tokens` includes cache hits + warmup and counts a shared prefix
  ONCE. The default `usage` carries only `input_tokens` and `output_tokens`;
  with `diagnostics: true` it also carries `cached_tokens` and
  `state_cache_hit`, which expose prefix-cache behavior.
* When `diagnostics: true`, the response carries one additive object beyond the
  frozen Jev shape. `diagnostics` reports the readout identity (`contract_hash`,
  `prompt_version`, `model`, `quantization`, `template_hash`, `backend_flags`,
  `label_pool_size`), the adapter scope (`adapters_configured`,
  `adapter_scope`), and timings (`prefill_ms`, `scoring_ms`, `suffix_tokens`,
  `common_suffix_tokens`). These are additive and never change an answer. The
  default response omits them.

### 3.2 Worked example

Request:

```json
{"model": "qwen3-4b", "state": "Customer asks for a refund of $42, order arrived broken.",
 "questions": {
   "refund": {"type": "noul", "instructions": "Should we refund?",
              "criteria": {"true": "yes, refund", "false": "no refund"}},
   "dept": {"type": "choice", "instructions": "Route the ticket.",
            "criteria": {"billing": "payment/refund issues", "support": "how-to help"}},
   "urgency": {"type": "score", "instructions": "Rate urgency.",
               "criteria": ["low", "medium", "high"]}}}
```

Response (default envelope):

```json
{"model": "qwen3-4b",
 "answers": {
   "refund": {"type": "noul", "noul": 0.9995},
   "dept": {"type": "choice", "choice": "billing",
            "probabilities": {"billing": 0.9999, "support": 0.0001},
            "confidence": 0.999},
   "urgency": {"type": "score", "score": 1.66,
               "probabilities": {"0": 0.08, "1": 0.18, "2": 0.74},
               "legend": {"0": "low", "1": "medium", "2": "high"},
               "confidence": 0.33}},
 "usage": {"input_tokens": 412, "output_tokens": 0}}
```

The same request with `"diagnostics": true` keeps the answers byte-identical
and adds `certainty`, the `diagnostics` object, and
the extra usage counters:

```json
{"model": "qwen3-4b",
 "answers": {
   "dept": {"type": "choice", "choice": "billing",
            "probabilities": {"billing": 0.9999, "support": 0.0001},
            "confidence": 0.999, "certainty": 0.9999}},
 "usage": {"input_tokens": 412, "output_tokens": 0,
           "cached_tokens": 180, "state_cache_hit": false},
 "diagnostics": {"contract_hash": "...", "adapters_configured": false, "adapter_scope": "base"}}
```

---

## 4. Errors and headers

`handle_decision` (`tools/server/server-context.cpp`) distinguishes the error
classes by exception type, and the server maps each through
`format_error_response` (`tools/server/server-common.cpp`) to an HTTP status.
Malformed JSON and unusable values/types are 400; well-formed but semantically
invalid decision content is 422 (including an unknown field inside a question
object); unknown top-level fields are ignored; over-limit capacity is 413 or
422; a full queue is 429 or 529 with `Retry-After`; a cancel or client
disconnect is 499; an unavailable model (no usable labels, contract mismatch)
is 501. The full contract follows.

### 4.1 Full target error contract

| HTTP | `type` | When | Notes |
|---|---|---|---|
| 400 | `invalid_request_error` | malformed JSON; unusable route/role/args | do NOT use for semantic question errors |
| 401 | `authentication_error` | missing/invalid API key when auth is configured | |
| 403 | `permission_error` | authenticated but not allowed | |
| 404 | `not_found_error` | unknown model/route | in router mode the decision route follows the existing proxy behavior, like the other routes |
| 413 | `payload_too_large` | request body over the configured cap | must reject before decode |
| 415 | `unsupported_media_type` | `Content-Type` is not `application/json` | enforced by the shared HTTP layer |
| 422 | `invalid_request_error` (semantic) | valid JSON but invalid decision schema: bad question type, `DECISION_MIN_QUESTIONS`-`DECISION_MAX_QUESTIONS` questions, `DECISION_MIN_OPTIONS`-`DECISION_MAX_CHOICE_OPTIONS` options, `DECISION_MIN_OPTIONS`-`DECISION_MAX_SCORE_LEVELS` score levels, duplicate keys, missing `instructions`, missing required `criteria`, unknown field inside a question | Jev semantic-failure code; never downgrade to 400. Unknown top-level fields are ignored, not rejected |
| 429 | `rate_limit_error` | decision queue full | include `Retry-After` |
| 499 | `client_closed_request` | client disconnected / cancelled mid-evaluation | cancel siblings; never report as a normal answer |
| 500 | `server_error` | inference/engine failure | reset engine state |
| 501 | `not_supported_error` | feature not enabled/available | existing enum value |
| 503 | `unavailable_error` | shutting down / no service | include `Retry-After` when transient |
| 529 | `overloaded_error` | server overloaded | include `Retry-After`; Jev uses this |

Error body shape (unchanged from existing server):
`{"code": <http int>, "message": <string>, "type": <string>}`.

Type-string provenance: the `type` strings live in `format_error_response`
(`tools/server/server-common.cpp`). 400/401/403/404/500/501/503 predate this
work; 413/422/429/499/529 were added for the decision contract. The HTTP code
is normative; if a future implementation uses a different `type` string it must
document the mapping.

Notes:
- 422 vs 400 is a deliberate split: malformed syntax is 400, semantically
  invalid decision content is 422. A client must be able to tell them apart.
- Over-limit capacity is rejected BEFORE decode; the path never truncates and
  never silently clips.
- 413 comes from the body cap (`LLAMA_DECISION_MAX_BODY`), 429/529 from the
  concurrent-request cap (`LLAMA_DECISION_MAX_QUEUE`); both can be overridden
  for tests.
- The control cases (requests that must NOT be rejected) are covered by the
  server tests, alongside the fixtures for every row above.

### 4.2 Headers

* `Retry-After` on 429/529 (implemented).
* Cancel/disconnect aborts the evaluation and never emits a partial answer
  (implemented; a client that is already gone cannot observe the 499, so the
  observable guarantee is that no partial answer is produced).
* `x-decision-latency-ms` and `Server-Timing: prepare,prefill,branches` are
  proposed instrumentation, not currently emitted. Per-request timings are
  available via the response `timings` object.

---

## 5. Limits

Every limit is owned by one constant; the validator and this table read the
same value.

| Limit | Owner |
|---|---|
| questions per request | `DECISION_MIN_QUESTIONS`-`DECISION_MAX_QUESTIONS` |
| `noul` options | fixed 2 |
| `choice` options per question | `DECISION_MIN_OPTIONS`-`DECISION_MAX_CHOICE_OPTIONS` |
| `score` levels per question | `DECISION_MIN_OPTIONS`-`DECISION_MAX_SCORE_LEVELS` |
| order-de-bias passes | 1 default, capped at `DECISION_MAX_PERMUTATIONS` |
| contexts per generic request | `DECISION_MAX_CONTEXTS` |
| answer-label pool | `LABEL_POOL_CAP` (equal to `DECISION_MAX_CHOICE_OPTIONS`) |
| request body | `LLAMA_DECISION_MAX_BODY` (default 2 MiB) |
| concurrent decisions | `--decision-max-queue` (default 4, env `LLAMA_DECISION_MAX_QUEUE`), then 429/529; `--decision-timeout-ms` (env `LLAMA_DECISION_TIMEOUT_MS`) sets the server-side deadline |
| retained session references | one per chat slot; token snapshots hold only the owned token list plus adapter scope (`bytes` reports that), `--decision-session-budget-mb` sets the byte budget (0 = unlimited) and `--decision-session-ttl` sets the default expiry (0 = none) |
| session residency | token snapshots are non-resident by design: they hold no context cells between decisions (no arena, no `arena_used`) |
| multi-instance routing | `instance`/`model` body or query fields; sessions pin to the owning instance, groups are refused for session requests; all decisions run on the decision sidecar executor |
| trie fields / values per field | 1-32 fields, 1-255 values |

The protocol option cap is `DECISION_MAX_CHOICE_OPTIONS` (255); the label-pool
cap is `LABEL_POOL_CAP` (255), equal so every option can get a label. The
REALIZED label pool is model-dependent: the tokenizer must resolve each label
as a single token at the answer boundary, so a model may yield fewer than 255.
The realized size is reported as `diagnostics.label_pool_size` and
recorded in the calibration ledger. A request whose widest question needs more
labels than the realized pool is a 422, never a truncated option set and never
a silent text-mode workaround: the user must pick a model whose tokenizer
resolves enough single tokens (a capable model resolves the full 255-label pool
over A-Z, a-z, 0-9, printable ASCII symbols, and accented Latin/Greek/Cyrillic
single characters).

---

## 6. Calibration and acceptance

Two axes, never conflated:
- CONFIDENCE (producer self-doubt): `certainty` and the winner share. A model
  can be confident and still wrong.
- TASK VALUE (outcome correctness/economy): `tree_max`, hoist length, wave
  packing, single-question bypass, dedup, temperature fit quality
  (NLL/Brier/ECE), permutation gain.

Rules:
* Ship `T=1.0` + per-type overrides. A non-default temperature ships with
  PROVENANCE (model hash, quantization, template hash, backend flags) and is
  refused on mismatch. Temperature is argmax-invariant: it changes
  probabilities/thresholds, not winners. Provide an offline refit script
  (NLL/Brier per primitive on own traffic). Expect quant/backend swaps to move
  ECE; refit per deployment.
* Order bias: `permutations` is a seeded distinct shuffle with the mean taken
  by semantic key. Default stays 1; `--decision-permutations` raises the server
  default for deployments that opt in.
* Fork exactness: a branch must reproduce the parent's state. The fork oracle
  decodes a branch and compares its state bytes against a full `restore` fork of
  the same parent, so a plain `seq_cp` that only matches the argmax still fails.
  The default fork is the partial hybrid (`seq_cp` attention + `PARTIAL_ONLY`
  recurrent copy); a full restore stays available and is the automatic fallback
  when the partial format is unavailable. Acceptance compares with
  TOLERANCE, never bit-equality. Pin `tokenizer + template + label code` in a
  contract/startup hash and refuse on mismatch. Every parity/calibration claim
  inherits the frozen backend flag set recorded at baseline (FA type, K/V
  cache types, `kv_unified`, `swa_full`, `n_ubatch`, threads); changing flags
  invalidates the claim.
* Prompt-injection corpus for `safe_data`: `<|turn>`, `{REASON:`, `__media__`,
  backticks, and nested arrays/objects.
* Diagnostics to log per request: `prefix/suffix tokens`, `cache_hit`, `waves`,
  `queue_ms`. These catch prompt-drift and readout drift before users do.
* Policy: the model's `confidence`/`certainty` NEVER gates admission,
  caching, routing, or persistence.

---

## 7. Glossary

- **state**: the evidence document; opaque to the server, treated as DATA.
- **question**: one typed decision over a `state`; identified by an opaque
  `qid` that is never shown to the model.
- **option / candidate**: one allowed answer value for a question.
- **label**: a single-token letter (`A`, `B`, ... up to `BL`) the model emits
  instead of the option text; probability comes from that token's logit.
- **readout**: how option probabilities are extracted. Letter readout uses one
  label token per option; trie readout scores the model's own token paths for
  surface-form values.
- **branch / trunk / prefix**: one KV sequence per scored path; a trunk is
  `shared prefix + context`; the prefix is the cacheable head.
- **session fork**: answering `id_slot` or `session_id` by replaying the slot's
  owned token snapshot on the decision sidecar executor instead of re-prefilling
  `state`; the source slot is read-only.
- **session reference / retained turn**: the owned token snapshot of a completed
  turn's tokens and adapter scope, taken eagerly at session create (or on the
  first `id_slot` decision for that turn), reused by later decisions in the same
  turn, and released when the turn advances. One retained turn per slot. `turn`
  is the opaque client tag that pins the reference.
- **session handle / `session_id`**: a first-class, server-side session identity
  decoupled from the reused `id_slot`, with an explicit create/query/pin/erase
  lifecycle over `/v1/session`.
- **session backend**: how a retained turn is referenced. The sidecar executor
  uses the single `tokens` backend: the owned token list plus adapter scope,
  replayed into the executor's context on demand. `clone`/`file` are a 501
  capability refusal; a non-selectable backend is never a silent fallback.
- **session manifest**: the sidecar a slot save co-writes, bound to the slot file
  by its content hash, that a slot restore uses to rebind or refuse the retained
  reference - a restore never answers from a stale turn.
- **decision sidecar executor**: the internal pool instance that owns every
  decision, stateless and session, on its own context and scheduler thread; chat
  contexts are never written, stalled, or resized by decision work.
- **producer confidence vs task value**: two DIFFERENT axes (Section 6). Never
  use a confidence number to gate caching/admission/correctness.
- **closed-world probabilities**: `probabilities` are conditional on the
  supplied options; they do not measure the chance that all options are wrong.
- **contract hash**: a hash over the tokenizer identity, the framed prompt
  template, and the label code. If it changes, any earlier calibration is
  stale. Logged at startup and returned as `diagnostics.contract_hash`.
