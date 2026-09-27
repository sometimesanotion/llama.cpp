# parallel-decision

Answer a finite JSON schema in one batched forward pass, instead of generating the JSON token by token.

Every field of a schema has a fixed set of allowed values (enums, booleans, bounded integers, numbers on a grid).
After the context, each field's allowed values are scored as token paths that fork from the same KV cache, so all
fields are answered in one `llama_decode` and cannot see each other. Each answer comes back with a probability, and
the JSON object is assembled by code, so it always matches the schema.

This directory holds the engine (`decision-engine.*`), a CLI (`llama-parallel-decision`), and the engine is also
served by `llama-server` as `POST /v1/decision`.

## Build

Same as llama.cpp:

```bash
cmake -B build -DGGML_CUDA=ON        # or plain `cmake -B build` for CPU / Metal
cmake --build build --config Release -j
```

## Run the server

`--decision-seqs N` reserves the sequence slots the decisions need: one holds the cached instructions, one per context
in flight, the rest are the parallel questions. It also switches the KV cache to unified, which is what lets the
branches share the context's cells.

```bash
./build/bin/llama-server -m model.gguf -ngl 99 -fa on -c 32768 --decision-seqs 24 --port 8096
```

The readout runs full logits on the shared chat context. There is no
classifier-only context and no answer head: every decision reads output rows
from the one context chat uses, so a decision cannot duplicate the KV cache.
The decision sequences sit above the chat slots (n_parallel ..
n_parallel + n_seq_decision). The engine estimates the request's peak KV use
(the cached instructions plus every live question branch) before any decode.
A request that does not fit is rejected with `422`, returns no decision, and
leaves no partial state; the next request still succeeds. The decision path
never truncates a prompt.

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

How many sequences a model affords depends on its attention. A plain-attention model shares the context's cells, so
128 sequences cost almost nothing. A sliding-window model (Gemma) allocates its window per sequence, so keep it low
(12 on a 12 GB card). Hybrid models with recurrent layers work, but llama.cpp splits their batches per sequence
length, so branches run in several passes instead of one.

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
available, else restore; copy on dense attention); set `"fork"` on a `contexts`/`schema` request, or
`LLAMA_DECISION_FORK=copy|restore|hybrid|auto`, to force it. On a sliding-window model the copy is clamped to the retained window
so branch memory does not grow.

Fork correctness is judged by a byte-level oracle, not by matching the argmax: the engine's tests decode a branch and compare its
state bytes against a full `restore` fork of the same parent. A plain `seq_cp` fork that shares a recurrent tail can agree on the
winner while its probabilities are stale, so the oracle is the guarantee and the hybrid fork is its enforcement.

## POST /v1/decision

`contexts` is a list of 1-256 strings. They share one schema, one set of instructions, and one cached prefix; results
come back in the same order.

```bash
curl http://localhost:8096/v1/decision -H "Content-Type: application/json" -d '{
  "model": "gemma-4-12b",
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
  "object": "decision",
  "results": [
    {
      "decision": {"category": "billing", "urgent": true, "priority": "high"},
      "fields": {
        "category": {"value": "billing",  "probability": 1.0,  "scored_nodes": 1, "tree": true},
        "urgent":   {"value": true,       "probability": 1.0,  "scored_nodes": 1, "tree": true},
        "priority": {"value": "high",     "probability": 0.74, "scored_nodes": 1, "tree": true}
      },
      "usage": {"context_tokens": 21, "scored_rows": 14}
    }
  ],
  "usage": {"prompt_tokens": 137, "cached_tokens": 116, "context_tokens": 21, "scored_rows": 14},
  "timings": {"prefill_ms": 50.7, "scoring_ms": 50.0, "total_ms": 100.7, "rounds": 1, "per_decision_ms": 100.7}
}
```

(That response is a real one: Gemma 4 12B on an RTX 3060, warm cache.)

### Schema

Compact fields, or a JSON Schema object with `properties`:

| type | keys | notes |
|---|---|---|
| `enum` | `choices` (or `enum`) | 1-255 values |
| `boolean` | - | true / false |
| `integer` | `minimum`, `maximum` | 1-255 values |
| `number` | `minimum`, `maximum`, `step` (`multipleOf` in JSON Schema) | fixed-width decimals |

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
alias for the same handler; use `/v1/decision`. `model` is optional and echoed back verbatim;
`GET /v1/models` keeps the OpenAI list shape, not Jev's.

### Live session (`id_slot`)

Both shapes accept `id_slot` (and optional `session_pos` and `turn`) to answer about a chat slot
that already holds decoded state, so the transcript is not re-prefilled. The slot must exist and
hold state; a `session_pos` that does not exactly continue it is a 422. The generic shape scores
one context per session request; the Jev shape appends the questions as a fresh user turn through
the slot's chat template and runs full logits on the shared context. The server answers through an
owned snapshot: on the first decision for the slot's current turn it serializes the slot's decoded
state into a decision-owned arena sequence (`--decision-arena-seqs`), later decisions in the same
turn fork the arena sequence, and the snapshot is released when the slot decodes past it (a new
completed turn). One retained turn per slot; the opaque `turn` tag pins it, and a mismatched
`turn` is a 422. A decision on an in-flight slot is a 422. The source slot
is never mutated, and the response reports `session_fork`, `source_slot` and `session_pos`
additively.

### Limits and errors

The full contract lives in `docs/decision/API.md` (Sections 4 and 5); the
validator and that document read the same `DECISION_*`/`LABEL_POOL_CAP`
constants. In short: 422 for semantic errors (empty state, unknown field
inside a question, missing/non-null `instructions`, over-limit options),
413 for the body cap, 429/529 with `Retry-After` for the queue cap, 499 on
client disconnect, and 501 when the model cannot serve decisions. The path
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
thin layer over the trie scorer, not a second implementation.

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
context: shared prefix + one state per request; branches forked on a unified KV cache
output: probability distributions only, output_tokens always 0, closed over the supplied options
confidence: Jev value (N*p_max-1)/(N-1) by default, opt-in 1 - H/log(K); certainty: max(p); concentration, NOT calibrated accuracy
calibration: deployment-specific; valid only under the recorded model, quantization, template hash and backend flags
```

## CLI

`llama-parallel-decision` runs the same engine from a worker process (stdin/stdout protocol, one JSON request per
line). Environment: `DECIDE_TREE`, `DECIDE_TREE_MAX`, `DECIDE_NSEQ`, `DECIDE_SPLIT_BOUNDARY`.
It is a developer tool and is off by default; build it with `-DLLAMA_BUILD_DECISION_CLI=ON`.

## A UI for it

[decision-playground](https://github.com/thecodacus/decision-playground) is a browser-only playground: it talks
straight to your llama-server, runs a decision and the same question as a chat completion side by side with live
timers, and has a small game whose agents decide through the endpoint.
