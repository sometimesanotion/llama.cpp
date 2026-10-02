# parallel-decision

Answer a finite JSON schema in one batched forward pass, instead of generating the JSON token by token.

Every field of a schema has a fixed set of allowed values (enums, booleans, bounded integers, numbers on a grid).
After the context, each field's allowed values are scored as token paths that fork from the same KV cache, so all
fields are answered in one `llama_decode` and cannot see each other. Each answer comes back with a probability, and
the JSON object is assembled by code, so it always matches the schema.

This directory holds the decision engine and its two request front-ends, and nothing executable. The engine
(`decision-engine.*`, `labels.*`, `decision-protocol.*`, `letter_readout.*`, `generic_frontend.*`) is built as a
static library that only `llama-server` links, through `POST /v1/decision` and `POST /v1/systemone`; there is no
CLI and no standalone worker. `tests/` holds one Python case, `test_temperature.py`, which `ctest` runs as
`test-decision-temperature`.

> **Removed, so nobody goes looking for it.** This directory used to carry three executables and now carries
> none: a standalone CLI that drove the engine outside the server (dropped with the in-context decision lane), a
> state-transfer benchmark that compared host-snapshot capture against a sidecar re-prefill (deleted once its
> experiment was recorded), and an earlier scratch driver for engine throughput. All three sources are gone from
> the tree, so nothing here builds them and no binary in `build/` is one of them. Their numbers live in
> `docs/decision/OBJECTIVE_MULTI_CONTEXT.md` section 12.2 and `docs/decision/BENCHMARKING.md`; to measure now,
> drive the server over HTTP instead of looking for a harness.

## Build

Same as llama.cpp:

```bash
cmake -B build -DGGML_CUDA=ON        # or plain `cmake -B build` for CPU / Metal
cmake --build build --config Release -j
```

## Run the server

`--decision-seqs N` turns the decision API on and registers the decision sidecar executor: an
internal, undeletable pool instance with its own context and its own scheduler thread, sized to
hold the shared prefix, one context in flight and one branch per scored path. Every decision runs
there, stateless and session alike, so a decision can never write, stall or resize a chat context.

```bash
./build/bin/llama-server -m model.gguf -ngl 99 -fa on -c 32768 --decision-seqs 24 --port 8096
```

A decision has no classifier-only context and no answer head: the readout reads full logits from the
sidecar's own context, so answering a question costs no duplicated chat KV. The engine estimates the
request's peak KV use (the cached prefix plus every live question branch) before any decode. A
request that does not fit is rejected with `422`, returns no decision, and leaves no partial state;
the next request still succeeds. The decision path never truncates a prompt.

Without `--decision-seqs` there is no sidecar, and every decision and session route answers `501`
`not_supported_error` naming the flag. The refusal happens before any dispatch, so it never touches
chat state.

With a presets file, one loaded model serves chat and decisions:

```ini
[*]
ngl = 99
fa = on
jinja = 1
parallel = 1
cache-type-k = q8_0
cache-type-v = q8_0
decision-seqs = 24

[gemma-4-12b]
model = ./models/gemma-4-12b-it-UD-Q4_K_XL.gguf
ctx-size = 32768
ubatch-size = 512
decision-seqs = 12
```

```bash
./build/bin/llama-server --models-preset models.ini --models-max 1 --port 8096
```

How many decision sequences a model affords depends on its attention, and the budget is the sidecar's
own: it is sized from `--decision-seqs`, not from the chat instance's `n_parallel`. A plain-attention
model shares the sidecar's cells across its sequences, so many sequences cost almost nothing. A
sliding-window model (Gemma) allocates its window per sequence, so keep it low (12 on a 12 GB card).
Hybrid models with recurrent layers work, but llama.cpp splits their batches per sequence length, so
branches run in several passes instead of one.

`--decision-sidecar-ctx N` sizes the sidecar window; `0` (the default) means the largest configured
instance window, so the longest turn a chat instance can produce still replays. `--decision-seqs`
forces `kv_unified` on the sidecar context only, never on a chat instance.

Set `"permutations": N` (default 1, capped at 8) to de-bias option order: pass 0 keeps the caller's
order and each later pass presents the same options in a distinct order seeded by the question id,
then the per-pass distributions are averaged by option key. Two passes cost about 1.1x and pull a
position-biased model toward the balanced answer; the noul second pass is the swap. The default
single pass is byte-identical to a request without the field. A deployment can raise the default for
requests that omit the field with `--decision-permutations N` (`LLAMA_ARG_DECISION_PERMUTATIONS`); an
explicit request value always wins.

Branches fork the cached prefix three ways. `copy` uses `llama_memory_seq_cp`; it shares attention cells by metadata and is exact
only for dense attention, because a recurrent/hybrid model's SSM/conv state is not carried by those cells. `restore` saves and
reloads the whole sequence state with `llama_state_seq_get/set_data`; it is exact everywhere but copies the attention prefix per
branch. `hybrid` is `seq_cp` for attention plus a `PARTIAL_ONLY` byte copy of the recurrent state into the child's own cell: exact
everywhere and cheap on hybrid models. The engine picks per model (`auto` = hybrid on recurrent/hybrid when the partial format is
available, else restore; copy on dense attention); set `LLAMA_DECISION_FORK=copy|restore|hybrid|auto` to force it. There is no
request field for this: the fork is a property of the model, not of the question. On a sliding-window model the copy is clamped
to the retained window so branch memory does not grow.

Fork correctness is judged by a byte-level oracle, not by matching the argmax: the engine's tests decode a branch and compare its
state bytes against a full `restore` fork of the same parent. A plain `seq_cp` fork that shares a recurrent tail can agree on the
winner while its probabilities are stale, so the oracle is the guarantee and the hybrid fork is its enforcement.

## POST /v1/decision, generic `schema` shape

A body with a top-level `schema` (and no `questions`) selects the generic front-end. `model` is required, exactly as on
the Jev shape. `contexts` is a list of 1-256 strings; they share one schema, one set of instructions and one cached
prefix, and come back in the same order.

```bash
curl http://localhost:8096/v1/decision -H "Content-Type: application/json" -d '{
  "model": "gemma-4-e4b",
  "instructions": "Answer each question about this support request from its state.",
  "schema": {
    "category": {"type": "enum", "choices": ["billing","technical","cancellation","other"],
                 "description": "What type of support request is this?"},
    "urgent":   {"type": "boolean", "description": "Does this need urgent handling?"},
    "priority": {"type": "enum", "choices": ["low","medium","high","critical"],
                 "description": "Rate support priority."}
  },
  "contexts": ["I was charged twice and need this fixed today."]
}'
```

```json
{
  "model": "gemma-4-e4b",
  "answers": {
    "category": {"type": "enum", "value": "billing", "confidence": 1.0,
                 "probabilities": {"billing": 1.0, "technical": 4.679e-08,
                                   "cancellation": 1.649e-08, "other": 1.799e-06},
                 "legend": {"billing": "billing", "technical": "technical",
                            "cancellation": "cancellation", "other": "other"},
                 "scored": "tree", "scored_nodes": 1},
    "urgent":   {"type": "boolean", "value": true, "confidence": 0.9998,
                 "probabilities": {"true": 0.9999, "false": 9.038e-05},
                 "legend": {"true": true, "false": false},
                 "scored": "tree", "scored_nodes": 1},
    "priority": {"type": "enum", "value": "critical", "confidence": 0.4478,
                 "probabilities": {"low": 1.894e-08, "medium": 7.021e-06,
                                   "high": 0.4142, "critical": 0.5858},
                 "legend": {"low": "low", "medium": "medium",
                            "high": "high", "critical": "critical"},
                 "scored": "tree", "scored_nodes": 1}
  },
  "usage": {"input_tokens": 137, "output_tokens": 0}
}
```

That is a real capture from `gemma-4-e4b` on the ROCm build (`-ngl 99`, `--decision-seqs 24`, first decision, cold
prefix), with the probabilities and `confidence` shortened to four significant digits for width; the server returns
full double precision. The digits are one run: a repeat returns the same keys and the same winners, and the reported
concentration can move a little (see the repeatability rule in `docs/decision/API.md` section 3.1), so read this for
the keys, not for the digits:

* The envelope is the **Jev one, extended**, and it is built by the same emitter for both shapes: the same `model`,
  `usage`, `timings`, `diagnostics` object and session-fork fields. There is no `object`, no `results[]`, no `fields`,
  no `decision`: the top level is `model`, `answers`, `usage`, and each answer is the Jev key set plus `value` and
  `scored`. A Jev client reads the same keys it always has, and switching a field from a `score` to a typed `number`
  changes the answer's shape and nothing else.
* `probabilities` is a **map**, keyed by every allowed value, summing to 1 - never a singular `probability`. `tree` is
  the string `"tree"` or `"argmax"` inside `scored`, never a boolean.
* One context answers at the top level as `answers`. Several use the `contexts` array of
  `{"answers": ..., "usage": ...}` objects, one per context, in request order.
* `usage` carries only `input_tokens` and `output_tokens` by default, exactly as on the Jev shape. `"diagnostics": true`
  adds the full counters (`cached_tokens`, `state_cache_hit`), the `timings` object and the whole `diagnostics` object at
  the top level - the same ones the Jev shape reports, with `prompt_version` naming the schema prompt - and adds
  `certainty` to every field plus `interval_p10_p90` and `aggregate` to a **numeric** one. A session decision reports the
  same fork fields a Jev session decision does. It never changes an answer.

The normative version of all of this is `docs/decision/API.md` section 2.5; where this README and that document
disagree, that document wins.

### Schema

Compact fields, or a JSON Schema object with `properties`:

| type | keys | notes |
|---|---|---|
| `enum` | `choices` (or `enum`) | 1-255 values |
| `boolean` | - | true / false |
| `integer` | `minimum`, `maximum` | 1-255 values |
| `number` | `minimum`, `maximum`, `step` (`multipleOf` in JSON Schema) | fixed-width decimals |

Every field also needs a `description`: it is what the model is told the field means, so a compact field without one
is a 422 naming the field. In the JSON Schema form the meaning already lives in the property's own
`description`/`title`, so it is not repeated and not required there.

Numeric fields take `aggregate`: `mode` (default), `median` or `mean`.

### Options

| field | default | meaning |
|---|---|---|
| `instructions` | `""` | prepended to the generated field catalogue; cached with it |
| `mode` | `auto` | `tree` scores every divergence node and returns exact probabilities; `greedy` walks the trie; `auto` picks tree up to `tree_max` values |
| `tree_max` | 128 | per-field switch between tree and greedy |
| `cache_prompt` | true | reuse the cached instructions + schema prefix |

## Decision shape (`state` + `questions`)

`POST /v1/decision` also accepts the decision shape: one `state` and 1-256 typed `questions`
(`noul` yes/no, `choice` pick-one, `score` ordered rating, plus the numeric extension
`integer`/`number` grids). Answers come back as one closed
distribution per question, with `output_tokens` always 0:

```json
{"state": "...",
 "questions": {"refund": {"type": "noul", "instructions": "refund?"},
               "dept": {"type": "choice", "instructions": "route", "criteria": {"billing": "payment", "support": "help"}},
               "urgency": {"type": "score", "instructions": "urgency", "criteria": ["low", "medium", "high"]},
               "age": {"type": "integer", "instructions": "age in years", "minimum": 18, "maximum": 60},
               "amount": {"type": "number", "instructions": "refund amount", "minimum": 0, "maximum": 100,
                          "step": 10, "aggregate": "mean"}}}
```

```json
{"answers": {"refund": {"type": "noul", "noul": 0.99},
             "dept": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.9, "support": 0.1},
                      "confidence": 0.53},
             "urgency": {"type": "score", "score": 1.6, "probabilities": {"0": 0.05, "1": 0.3, "2": 0.65},
                         "legend": {"0": "low", "1": "medium", "2": "high"},
                         "confidence": 0.28},
             "age": {"type": "integer", "value": 45, "probabilities": {"18": 0.01, "...": "...", "45": 0.87},
                     "confidence": 0.86},
             "amount": {"type": "number", "value": 10.0, "probabilities": {"0": 0.01, "...": "...", "10": 0.9},
                        "confidence": 0.88, "aggregate": 11.4}}}
```

A numeric question's bounds generate an ascending 2-255 value grid, scored by label exactly
like a choice; the answer's `value` is the winning grid value (a typed JSON number), and an
optional `aggregate` (`mean`/`median`/`mode`) adds a scalar summary of the same distribution
without changing `value`. See `docs/decision/API.md` for the full contract.

`instructions` is required and non-null on every question; choice keys and score levels keep object
order, and each option line sent to the model is rendered by `format_option_line` (one function) as
`label: <key> - <description>`, dropping the ` - ` and the description when the rendered description
is empty. Unknown top-level request fields are ignored; unknown fields inside a question object are a 422.

By default the response is the strict Jev envelope: `answers` plus
`usage{input_tokens,output_tokens:0}`. Pass `"diagnostics": true` to additionally get `certainty`,
the `diagnostics` object, and the extra usage counters
(`cached_tokens`, `state_cache_hit`). The answers themselves are identical either way.

Both shapes are served by `POST /v1/decision`, the canonical route. `POST /decision` is a deprecated
alias for the same handler; use `/v1/decision`. `model` is required on both (a body without it is a 422) and is
echoed back verbatim; `GET /v1/models` keeps the OpenAI list shape, not Jev's.

### Live session (`id_slot`, `session_id`)

Both shapes accept `id_slot` (and optional `session_pos` and `turn`) to answer about a chat slot
that already holds decoded state, so the transcript is not re-prefilled. A first-class `session_id`
is the alternative: a server-side handle decoupled from the reused `id_slot`, mutually exclusive
with it (a body carrying both is a 422), with a create/query/pin/erase lifecycle over
`POST/GET/PATCH/DELETE /v1/session`. The slot must exist and hold state; a `session_pos` that does
not exactly continue it is a 422. The generic shape scores one context per session request; the Jev
shape appends the questions as a fresh user turn through the slot's chat template and runs full
logits on the sidecar executor. The pool answers through an owned token snapshot: on the first
decision for the slot's current turn it copies the completed turn's tokens and adapter scope out of
the owning instance through a read-only op, and the sidecar re-prefills them into its own context
(`tokens` is the only backend; the retired `clone`/`file` backends are a 501). Later decisions in
the same turn reuse the snapshot instead of the slot, and the reference is released when the slot
decodes past it (a new completed turn). One retained turn per slot; the opaque `turn` tag pins it,
and a mismatched `turn` is a 422. A decision on an in-flight slot is a 422. The source slot is
never mutated, and the response reports `session_fork`, `source_slot` and `session_pos` additively.
A session decision decodes under the snapshot's captured adapter scope; a stateless decision stays
base-scoped. A slot save or restore carries only the slot's token and KV state; a retained session
is a live sidecar handle and is not part of a slot file. Under a configured byte budget or TTL, the
store evicts the least-recently-used unpinned, unleased reference and reaps expired unpinned
references; the defaults never evict.

In multi-instance mode (`--instance`), routing differs by shape, and the difference is the whole
point of the sidecar. A **stateless** decision carries no placement at all: `model` and `instance`
are echo-only, they never select a target, and the request is dispatched straight to the internal
decision sidecar executor. A **live-session** decision and a `/v1/session` create must name the
instance that owns the source slot, through the same `instance`/`model` fields chat uses, because
the snapshot is taken from that instance's scheduler; a group is refused. The decision itself still
runs on the sidecar, so the same slot id in another instance is a different, empty slot.

### Limits and errors

The full contract lives in `docs/decision/API.md` (Sections 4 and 5); the
validator and that document read the same `DECISION_*`/`LABEL_POOL_CAP`
constants. In short: 422 for semantic errors (empty state, unknown field
inside a question, missing/non-null `instructions`, over-limit options, a
`session_id` together with `id_slot`), 400 for a body carrying both `schema`
and `questions`, 413 for the body cap, 429/529 with `Retry-After` for the queue
cap, 503 with `Retry-After` when the server-side deadline expires or a pool
transient fails, 499 on client disconnect, and 501 when decisions are not
enabled or the model cannot serve them. The path
never truncates: an over-limit request is rejected, never silently clipped.

### Prefix cache

The shared prefix (instructions, schema, state) is cached per request tag. A cached entry is stored
in the self-contained host format, because a device-format entry references a context staging buffer
that the next save reuses. The active request keeps a device-format copy for its own trunks and
branches, so a warm prefix restores device-to-device while a later save cannot corrupt an older
cache entry.

### Usage and diagnostics

`usage` always reports `input_tokens` (state + cached prefix) and `output_tokens` (always 0, nothing
is generated). With `"diagnostics": true` it also reports `cached_tokens` and `state_cache_hit`;
every answer additionally carries `certainty`, and the response carries
`diagnostics.contract_hash` (see below). These are
for inspection only; they never change an answer, and they are omitted from the default envelope.

### Contract hash and diagnostics

At first use the server computes a contract hash over the tokenizer identity, the framed prompt
template, the label code, and the prompt version, logs it, and returns it as
`diagnostics.contract_hash`. Pass `--decision-contract HASH` to pin it: if the running contract
differs, the decision path is refused with a plain 501 instead of serving stale calibration.

### Frozen backend flags

Every parity and calibration claim is only valid under the backend flag set recorded in
`tests/decision-baseline/calibration.json` (flash-attention setting, K/V cache types, `kv_unified`,
`swa_full`, `n_ubatch`, threads). Change any of them and re-run the calibration gate.

## Temperature and confidence

`temperature` (default 1.0) and per-type `temperatures` scale the label logits before the
softmax. Temperature never changes the winner; it only changes how sharply the distribution
is concentrated.

`confidence` is, by default, the certainty-based Jev value `(N*p_max - 1)/(N - 1)`
(rescaled winner share: uniform -> 0, one-hot -> 1, conservative at the low end),
which matches Jev's official Choice confidence and is the recommended basis for
fallback gating. `certainty` is `max(p)` (the winner's share). A request with
`confidence_profile: "local"` reports the entropy form `1 - H/log(K)` instead,
which reads the whole distribution and calibrates better on some families; the
profile changes only the reported number, not an answer or a probability. Both
values measure how concentrated the answer is. They are **not**
calibrated correctness, and they are **not** accuracy. The probabilities are
conditional on the options you supplied: if the right answer is not among them,
the distribution still sums to 1 over the wrong set. Never gate admission,
caching, routing or persistence on `confidence` or `certainty` alone, and never
present them as probability of being correct. `tests/decision-baseline/accuracy_report.json`
reports winner agreement, Brier and ECE per model and framing; confidence never gates any of it.

Admission is not part of this axis. The engine's peak-KV preflight check
accepts or rejects each request by its token footprint alone, never by
`confidence`/`certainty`. A low-confidence and a high-confidence request of the same length get the
same outcome.

A calibrated temperature is deployment-specific. `--decision-temperature FILE` loads a JSON
profile `{"temperatures": {"noul": ..., "choice": ..., "score": ...}, "provenance": {"model": ...,
"quantization": ..., "template_hash": ..., "backend_flags": ...}}`. If any temperature differs
from 1.0 and the recorded provenance does not match the running model, quantization, prompt
template and backend flags, the server refuses the profile instead of silently applying it.
With no file the default stays 1.0.

### Two front-ends, one engine

The two wire shapes are both producers of the engine's `field_input[]`, so they
share the prefix cache, the branch scorer, the softmax and the SWA clamp:

- The Jev shape: `state` (or `contexts` for the multi-context extension) plus
  typed `questions`, scored as declared answer labels and returned as one closed
  distribution per typed question.
- The generic `schema` shape: a JSON Schema or compact typed fields compiled by
  `generic_frontend` into the same `field_input[]`, returned as a typed record.

A body with `questions` selects the Jev front-end; a body with `schema` selects
the generic front-end; a body carrying both is a 400. The letter readout is a
thin layer over the trie scorer, not a second implementation, and the response
envelope around the two is built once (`server_context::decision_response`), so a
field added to it appears on both shapes at once.

Letter labels are resolved at the framed answer boundary, not in isolation. A SentencePiece /
`add_space_prefix` vocabulary tokenizes a bare `A` as the space-prefixed form in isolation but
emits the bare token after the tail, so the pool is built from the tail and the letter readout
works on both SentencePiece and BPE tokenizers.

The label pool is single-character-first: A-Z, a-z, 0-9, printable ASCII symbols, then accented
Latin, Greek and Cyrillic single characters, then the two-character composed fallback. A model
whose tokenizer resolves the single characters as single tokens realizes the full 255-label pool
(`LABEL_POOL_CAP`, equal to `DECISION_MAX_CHOICE_OPTIONS`). The realized pool is the hard
capacity for a model: a request whose widest question needs more labels than the realized pool is
a 422, never a truncated option set and never a silent text-mode workaround - the user must pick
a model whose tokenizer resolves enough single tokens.

## Model card snippet

```yaml
model: <base gguf>
task: single-pass decision / classification over a supplied state
readout: answer labels resolved at the framed answer tail (SentencePiece and BPE)
context: shared prefix + one state per request, on the sidecar executor's own context; branches forked
  on that context's unified KV cache, never on a chat context
output: probability distributions only, output_tokens always 0, closed over the supplied options
confidence: Jev value (N*p_max-1)/(N-1) by default, opt-in 1 - H/log(K); certainty: max(p); concentration, NOT calibrated accuracy
calibration: deployment-specific; valid only under the recorded model, quantization, template hash and backend flags
```

## A UI for it

[decision-playground](https://github.com/thecodacus/decision-playground) is a browser-only playground: it talks
straight to your llama-server, runs a decision and the same question as a chat completion side by side with live
timers, and has a small game whose agents decide through the endpoint.
