> Non-normative design notes, KV background, and benchmarks. The normative
> contract is `docs/decision/API.md`.

## What the /v1/decision endpoint is really trying to do:

At its core, this work adds a new kind of query to llama-server:
/v1/decision, which is not "generate text" but "pick an answer from a fixed
list." Instead of asking the model to produce a sentence, you hand it a JSON
object — a state (the situation being judged) and a set of questions, each
with a small finite set of allowed answers (true/false, a choice among
options, or a score level).  The model scores each allowed answer and
returns probabilities, e.g.  "this ticket is 92% a refund question."

The request has two mutually exclusive front-ends over one shared engine (a
body using both is a 400). The primary "state + questions" shape carries a
`state` and a `questions` map; each question carries required `instructions`
and is typed `noul` (true/false), `choice` (pick one of 2-255 options),
`score` (pick a 0..K-1 level on a 2-10 scale), or the numeric `integer` /
`number` extension, and the server scores one
next-token choice over verified single-token letter labels, sharing one
framed state prefix across all questions.  The generic typed-schema shape
("schema") supplies a JSON Schema or compact typed fields
(boolean/enum/integer/number) and walks the schema's value trie in one
batched pass.  Both front-ends terminate at the same `field_input[]` plan, so
the two contracts cannot drift.  A request may also ask for a permutation
de-bias pass (`permutations: N`), which shuffles option order and averages,
so no one fixed ordering biases the scores.  A request may also carry an
`id_slot` or a first-class `session_id` (and optional `turn`) to answer about a
live chat slot through an owned reference, without re-prefilling the
transcript.

This is classification-style work — routing, triage, RAG ranking, structured
extraction — where you care about which option wins, not about fluent prose. 
The whole feature is an attempt to make that cheap and deterministic, in
three ways:

1.  One batched forward pass instead of many.  The engine lays out every
question, every candidate, and every branch (each candidate's unique tail
tokens) as separate sequences, decodes them all in a single batched
llama_decode, and reads one scored row per branch.  A decision costs roughly
one decode step, no matter how many options exist.

2.  Full-vocabulary logits on the shared context.  The engine reads one
output row per branch from the shared context's logits.  There is no second
classifier-only context and no answer head: the feature serves every request
with the one context chat uses, so a decision cannot duplicate the KV cache.

3.  Faster branch forking.  To evaluate many branches at once, the engine
needs many copies of the same "so far" context.  The branch makes
saving/restoring a sequence's state much cheaper: it keeps the state on the
GPU and stages the device-to-device copies on the backend's stream with a
single sync, instead of one synchronous copy (and one cudaStreamSynchronize)
per tensor.

Plus a lot of bookkeeping that makes this safe in production: a preflight
check that rejects requests that can't fit the context, admission control
(413/429/529), a contract hash that refuses to serve a decision if the
tokenizer/prompt-template identity changed, an optional calibrated
temperature file, and per-model caching of the answer-label pool.

Beyond 413/429/529, the endpoint returns 422 for a semantic or capacity
error, 499 when the client disconnects, and 501 when the
loaded model cannot serve decisions at all (for example its vocabulary has no
usable single-token answer labels, or a pinned contract does not match).

The optional temperature profile is loaded with `--decision-temperature
FILE`: it maps `noul`/`choice`/`score` temperatures to non-default values,
and is only honored when the file's recorded provenance (model, quantization,
template hash, backend flags) matches the running configuration — a stale
profile is a server configuration error, never silently applied.  The
contract identity can be pinned with `--decision-contract HASH`; a mismatch
refuses the decision path.  The default response is the strict Jev envelope
(`model`, `answers`, `usage` with only input/output tokens); passing
`"diagnostics": true` adds the `diagnostics` identity (contract_hash,
prompt_version, timings, and the provenance of the readout) and `certainty`,
so callers can see exactly how an answer was produced without changing the
answers themselves.

## How the KV cache interacts with /v1/decision

The KV cache is where the model stores what it has seen so far.  The
decision engine leans on it heavily, and the server forces the "unified" KV
cache when decisions are enabled (--decision-seqs N automatically sets
kv_unified = true).  In a unified cache, all live sequences draw from one
shared pool of cells, and — critically — the code's seq_cp can make a branch
share the same physical cells as its parent instead of copying them, as long
as they're in the same stream.  So the layout for a decision request looks
like this:

- A dedicated snapshot sequence holds the static prefix (system prompt +
chat template up to the question).  It persists across requests, so a repeat
request with the same prefix is a cache hit: the prefix is never re-decoded. 

- Each trunk sequence forks off the snapshot (shares its cells) and decodes
one context (the state) plus a common suffix head.

- Each branch sequence forks off its trunk and decodes only its own unique tail
  — the few tokens that distinguish one candidate answer from another — writing
  only those new cells into the shared pool.

- At the scored position, the engine reads one output row per branch, so the
  whole request is served by: one prefix decode (on first use), one decode per
  context, and one batched decode for the branch tails.

The decision sequences live above the chat slots (ids `n_parallel` ..
`n_parallel + n_seq_decision`), on the same shared context chat uses.  There
is no separate decision context.

> Sidecar note (roadmap M0-M9): this shared-context description is historical.
> With `--decision-seqs` set, decisions run on the internal `__decision__`
> sidecar executor - its own context and scheduler thread, forced `kv_unified`
> only on itself - and chat contexts carry no decision sequences. Sessions are
> eager token snapshots replayed on the sidecar, which is the only executor.
> The shared-context engine pool, arena, yield, and session registry that this
> section describes have been removed from the tree.

## Is the KV cache updated by decision queries?

Yes — decisions are real decodes and they do write to the KV cache.  The
model's forward pass on the trunk context and on each branch tail populates
new KV cells, exactly like generation would.  But the branch is careful
about whose cache it touches and what survives:

- Decisions write into the same unified pool as chat, but into
    reserved sequence ids above the chat slots (n_parallel ..  n_parallel +
    n_seq_decision), so chat's slots and their KV are never overwritten.

  - After scoring, the engine removes the branch and trunk sequences
    (llama_memory_seq_rm), reclaiming their cells.  Only the snapshot
    sequence's prefix survives, so the next matching query can reuse it.

  - A session decision replays the slot's owned token snapshot (the sidecar's
    `tokens` backend: the owned token list plus adapter scope, re-prefilled into
    the sidecar context on demand; `clone`/`file` are a 501 capability refusal),
    never the live slot: the source slot's KV is never read for scoring and
    never written. There is no in-context session lane: it was removed with the
    session registry and the arena.

  - A preflight check estimates peak KV use and returns 422 rather than ever
    partially overwriting the cache, and a cancelled request leaves the pool
    dirty but the next decision clears it before reuse.

So the short answer: decisions do update the KV cache, but transiently, in a
reserved range, with self-cleanup - the design's whole point is that a
decision never disturbs the state chat depends on.

## Live-session decisions (owned token snapshot)

A request with `id_slot` or a first-class `session_id` answers about a chat
slot that already holds decoded state.  The server does not fork the live slot
and does not re-prefill the transcript.  Instead the pool captures the slot's
completed turn as an owned token snapshot: the token list plus the enabled
adapter scope, copied out through a read-only op on the owning instance's
scheduler, so no KV pointer, sequence id, or context handle leaves that
instance.  The sidecar executor re-prefills the owned tokens into its own
context and scores there.  Later decisions in the same turn reuse the snapshot,
so the answer survives the origin slot being cleared and reused by
`cache_idle_slots`.  One retained turn per slot: when the slot decodes past the
reference's position (a new completed turn), the reference is released before
the next decision.  The optional `turn` tag pins the retained turn; a
mismatched `turn` is a 422, never a silent answer about a different turn.  The
trigger is a cost decision (fire on the first decision for a turn, reuse the
snapshot), never a confidence decision.  A decision on an in-flight slot is a
422 (the turn is not complete).

`session_id` handles live outside the reused `id_slot` with a
create/query/pin/erase lifecycle over `/v1/session`.  Under a configured byte
budget (`--decision-session-budget-mb`), the store evicts the
least-recently-used unpinned, unleased reference to fit a capture, and a
configured TTL reaps expired unpinned references; the defaults (unlimited
budget, no expiry) never evict.  A slot save or restore carries only the slot's
token and KV state: a retained session is a live sidecar handle and is not part
of a slot file.

## What the benchmarks say: /v1/decision versus chat

The reported warm numbers (GPU, ROCm 7900 XT, flash attention off for
bit-reproducibility) for the full-logits readout are:

- LFM2.5-350M ~70 ms, LFM2.5-2.6B ~108 ms, Qwen3.5-2B 118 ms, Qwen3.5-9B 252
  ms, Gemma-e4b 70 ms, Gemma-4-12B-QAT 129 ms.

Read against chat, the key structural difference is cost per token.  A chat
completion is a loop: decode one token, sample it, feed it back, repeat —
the cost is proportional to the number of generated tokens (tens to
hundreds).  A decision is a single batched decode that produces every
candidate's score at once, plus a prefill for the context.  So a decision's
latency is roughly one token-generation step, and it does not scale with the
number of options or fields — all branches ride in the same batch.  That's
why even a 9B model answers in ~250 ms warm, while producing a 50-token chat
reply from the same model would take roughly an order of magnitude longer.

## Three optimizations drive the gap:

- the batched branch decode turns every candidate's score into one shared
  decode step (the full-vocabulary logits matmul is the cost, and it is
  shared across all branches);

- the async state restore turns per-tensor synchronous copies into one
  staged stream drain (measured ~4x on this operation); and

- prefix caching removes the prefill cost on repeat calls.

One honest caveat documented with the numbers: these are warm timings with a
cached prefix, on a specific GPU, and with flash attention disabled on ROCm
because it is not bit-reproducible.
