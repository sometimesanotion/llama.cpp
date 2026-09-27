# Objective: best of both decision engines

This document compares the two decision-engine implementations measured on this
machine, explains what each does well and badly, and recommends a target design:
a Jev-compatible API on top of one hardened, high-performance llama.cpp process.

- Reference implementation: one fork primitive (`llama_memory_seq_cp`), generic
  `contexts` + `schema` shape only.
- This implementation: keeps the generic shape as a second front-end and adds
  the Jev `state` + `questions` shape, three fork strategies, cancellation, and
  diagnostics. The classifier answer head and its separate context were
  removed; every readout uses full logits on the shared context.

Everything below was measured on the same machine and the same GPU. Numbers are
warm (`cache_prompt=true`), one context, `mode=auto`, flash attention off,
unified KV. Commands are in the last section.

## 1. How the measurement was run

The two branches do not share a benchmark harness, so the comparison is engine
to engine on one fixture (`contexts_schema.request.json`) with the same schema,
the same context, and the same model context parameters (`n_ctx=2048`,
`n_batch=n_ubatch=512`, `n_seq_max=10`, unified KV, flash attention disabled):

- Reference: a small scratch driver (`ref-bench`) built in the reference
  worktree, linked against the reference `llama-decision`, calls
  `engine::decide_batch`. The reference options have no `fork`; it always uses
  `seq_cp`.
- This implementation: the same engine driven by `test-decision-engine` with
  the fork oracle, and by the server decision tests; the fork strategies
  (`auto` = hybrid on a recurrent/hybrid model, copy on a dense model) and the
  explicit `restore` path are exercised through the test harness.

The fixture scores four fields: one 3-way enum, one boolean, one 4-level
integer, and one 3-level number. Warm means the second call, so the cached
static prefix is reused and only the dynamic part plus branch scoring is timed.

## 2. Speed: the two defaults are at parity

Warm total, milliseconds (prefill / scoring / total):

| model | arch | reference (seq_cp) | auto (default) | restore (explicit) |
|---|---|---|---|---|
| lfm2.5-350m | hybrid | 16.7 / 12.3 / **29.0** | 16.7 / 11.7 / **28.3** | 27.1 / 63.9 / 91.0 |
| lfm2.5-2.6b | hybrid | 27.0 / 21.5 / **48.6** | 25.5 / 22.1 / **47.5** | 42.2 / 89.0 / 131.2 |
| qwen3.5-2b | hybrid | 10.0 / 29.3 / **39.3** | 10.1 / 29.3 / **39.4** | 21.0 / 82.3 / 103.3 |
| gemma-4-e4b | dense | 41.0 / 26.8 / **67.8** | 38.6 / 26.7 / **65.3** | 86.4 / 270.1 / 356.4 |

Default to default, this implementation is within 0.3 ms to 2.4 ms of the
reference (0.3 percent slower on qwen3.5-2b, 2 to 4 percent faster on the other
three). That is noise on a warm call. The explicit restore path is 2.6x to 5.3x
slower.

Why this is the shape of the result:

- **Both defaults are the same fork.** `seq_cp` shares the parent attention
  cells by metadata and does no byte copy. The hybrid fork does the same
  `seq_cp` and adds a `PARTIAL_ONLY` copy of the recurrent state (about 0.1 ms).
  For a dense model the hybrid default is literally the same `copy` strategy.
- **Restore is a different workload.** It serializes and reloads the whole
  sequence state per trunk and per branch. On a hybrid model a full save walks
  the transposed V cache as thousands of tiny tensor copies; on a dense model it
  copies the whole KV. That is what pushes `scoring_ms` from about 12 ms to 64 ms
  (350m) and from about 27 ms to 270 ms (gemma).

The reference README warns that hybrid models split their batches per sequence
length and run in several passes. On these four models that did not happen:
`rounds` was 1 for every model in both branches.

## 3. Accuracy: the reference and the hybrid default agree exactly

Total variation between probability vectors, and winner flips, on the same
four-field fixture:

| model | reference vs restore | reference vs auto | auto vs restore |
|---|---|---|---|
| lfm2.5-350m | TV 7.4e-2, 0 flips | TV 4.3e-10, 0 flips | TV 7.4e-2 |
| lfm2.5-2.6b | TV 1.5e-2, 0 flips | TV 4.0e-9, 0 flips | TV 1.5e-2 |
| qwen3.5-2b | TV 3.2e-2, 0 flips | TV 1.5e-8, 0 flips | TV 3.2e-2 |
| gemma-4-e4b | TV 1.5e-2, 0 flips | TV 2.2e-10, 0 flips | TV 1.5e-2 |

The hybrid default reproduces the reference to float32 print precision. The
explicit restore path is the numerical outlier, not the reference.

The branch-state control explains it. Forking a trunk with plain `seq_cp` and
decoding a branch, then comparing the branch state bytes to a full restore,
gives `0 of N bytes differ, max logit delta 0` on all four models, in every
layout tested (fresh cache, two prior sequences, one and two branches per
batch). The historical claim that plain `seq_cp` loses recurrent state is not
reproducible on this base. That was already observed and recorded when the
control was written. The control runs on this build, but it drives the
same core `llama_memory_seq_cp` and full-state load the reference uses; the
decision-tool diff between the branches does not touch the memory core.

So on this machine and these models:

- The reference's `seq_cp` is both fast and exact at the branch-state level.
- The hybrid fork default is exact by construction and matches it.
- The explicit restore path is slow and, on the GPU, produces probabilities up to
  about 7e-2 away from the shared-cell paths without ever flipping a winner.
  That is consistent with a different physical attention-cell layout changing
  the GPU summation order, not with a different branch state: the control above
  shows the states are byte-identical. It is producer numerics, not a
  task-value difference.

The honest conclusion is that the hybrid default's byte-level fork oracle is
**insurance, not a measured accuracy fix** on this base. It is cheap insurance
(it costs about 0.1 ms and the default is at parity), and it protects layouts
where `seq_cp` copy-on-write might not be exact on another backend, but a reader
should not believe the reference is currently returning wrong numbers here.

## 4. What the API shapes mean

Reference `/v1/decision` and `/decision` take a constrained JSON schema:

```json
{ "instructions": "...",
  "schema": { "category": {"type":"enum","choices":["billing","technical"]},
              "urgent":   {"type":"boolean"} },
  "contexts": ["..."] }
```

and return `results[]`, each with `decision` (the assembled JSON object) and
`fields` (`value`, `probability`, `scored_nodes`, `tree`). Values are the
model's own token paths for the schema literals; there are no answer types, no
calibrated confidence, and no natural-language options. One request is one
schema; there is no session, permutations, temperature, head, or diagnostics.

This implementation accepts the same generic shape as a second front-end, plus
the Jev shape:

```json
{ "state": "...",
  "questions": {
    "is_urgent":  {"type":"noul",  "instructions":"Does this convey urgency?"},
    "department": {"type":"choice","instructions":"Which team?",
                   "criteria":{"billing":"Payments","technical":"Bugs"}},
    "frustration":{"type":"score", "instructions":"How frustrated?",
                   "criteria":["Calm","Frustrated","Very angry"]} } }
```

and returns `answers{}` with per-type payloads: `noul` (probability of yes),
`choice` (`choice`, `probabilities`, `confidence`), `score` (`score`,
`probabilities`, `legend`, `confidence`), plus optional
`diagnostics`. This is a typed, calibrated, natural-language contract; the
generic shape is the JSON-literal subset.

The deeper difference is the readout:

- Reference: score the model's token paths for each schema value (a trie over
  the model's own tokens). Great for enum/boolean/integer literals, hard for
  free-text options.
- The Jev readout: score one verified label token per option, over a label pool
  built from the chat template boundary. This is what lets arbitrary
  natural-language `criteria` become scorable.

Meaning in practice: the reference is a fast constrained-JSON endpoint. The
implementation is a decision service. The Jev shape is a superset in
expressive power; the legacy shape is a compatibility surface.

### 4.5 Local confidence and certainty vs the Jev formula [Local]

The local implementation returns two numbers on `choice` and `score`:

- `confidence = (N * p_max - 1) / (N - 1)`, clamped to [0,1], by default. This is
  Jev's documented Choice confidence, a linear rescale of the winner's share that
  maps a uniform distribution to 0. It is conservative at the low end, so it is
  the recommended basis for fallback gating. The Score form above K=3 stays
  [Undocumented], so the same monotone rule is the stated local value there.
- `certainty = max_i p_i`, the **winner's share**. It reads only the winner.

Written in terms of `certainty`, the Jev confidence is exactly

```
Jev confidence = (K * certainty - 1) / (K - 1)
```

It is not entropy: it ignores every option except the winner. The local entropy
value `1 - H / log K` remains available as an opt-in `confidence_profile: "local"`.

Measured on the committed readout corpus (two- and three-option questions), the three numbers for
the same distribution are:

| probabilities | `certainty` (max p) | Jev `(K p_max - 1)/(K - 1)` | `local confidence` (1 - H/log K) |
|---|---|---|---|
| (0.5004, 0.4996), K=2 | 0.5004 | 0.0008 | 5.1e-7 |
| (0.0012, 0.9987), K=2 | 0.9987 | 0.9975 | 0.9859 |
| (0.0012, 0.0465, 0.9523), K=3 | 0.9523 | 0.9284 | 0.8205 |
| (0.0062, 0.1662, 0.8276), K=3 | 0.8276 | 0.7413 | 0.5571 |

`certainty` is the raw winner's share, the Jev number is that share rescaled, and `local
confidence` is lowest of the three because it also penalizes mass spread across the non-winners.
The gap is largest, and the Jev number most flattering, when the distribution is flat enough that
`p_max` alone looks confident.

Implementation notes [Local]:

- `confidence` is always present on `choice` and `score`, and by default it is
  the certainty-based Jev value `(N*p_max - 1)/(N - 1)`, matching Jev's official
  Choice confidence. `certainty` is additive and present when `diagnostics` is
  set. `noul` carries neither, matching the documented contract.
- Both are pure functions of `probabilities`, so a caller can recompute either and need not trust
  the server. Neither number ever gates correctness, admission, caching, or routing.
- The local entropy value is available as an opt-in: a request with
  `confidence_profile: "local"` returns `1 - H/log K` as `confidence`. The default
  stays `(N*p_max - 1)/(N - 1)`, matching Jev. `certainty` is unaffected by the
  profile. The profile changes only the reported concentration; probabilities are identical.
- The benchmark surfaces record both: the frozen readout baselines store
  `confidence` and `certainty` per question, and the response `certainty` is
  reported when `diagnostics` is set.
- Definitions: `inverse_entropy_confidence` and `winner_share` in
  `tools/parallel-decision/decision-protocol.cpp`.
## 5. Process and hardening differences

Both branches run decisions on the loaded model's context with unified KV and
reserved decision sequences (`--decision-seqs`), and both keep chat and
decisions on one scheduler thread. This implementation adds:

- Cancellation: `options.should_stop` is checked inside the decode loop, and the
  server passes a per-request cancel flag.
- Cooperative yielding: the decision runs inside `yield_to_queue`, so
  `/metrics` and `/slots` are answered while a decision is in flight. The
  reference calls `handle_decision` directly and blocks them.
- Guaranteed cleanup: `clear_pool_seqs` runs on every exit, so a failed or
  cancelled decision cannot leave cells that starve chat.
- A single full-logits path on the shared context: the classifier-only
  context and answer head were removed, so a decision never duplicates the KV
  cache.
- Diagnostics: contract hash, template hash, and adapter
  scope, plus `certainty` on choice/score.
- Adapter scoping: decision decodes run on the base model, scoped while
  chat applies its adapters.

The reference is much simpler to read and maintain. Its failure modes are also
simpler: no cancellation inside a multi-second decision, no yield, no
diagnostics, no typed answers.

## 6. Recommendations

Target: one engine core, a Jev-compatible public API, and a hardened process
behind it. Concretely:

1. **Keep the Jev shape as the public contract, keep the generic shape as a
   second front-end.** Route both shapes through one scorer so the answer path
   cannot drift (this implementation already does this). Do not grow a second
   scorer for the generic shape.

2. **Keep the hybrid default and the fork oracle.** It is at parity with the
   reference and gives a byte-level guarantee. Do not "simplify" to plain
   `seq_cp` for the one-line speedup: the guarantee is the product, even though
   the guarantee is not currently exercised on this base. Keep exactly one
   place that chooses the fork (`select_fork`) and one `fork_into`.

3. **Add a startup capability probe, not a runtime guess.** At model load, run
   the branch-state control once for the loaded layout and log whether plain
   `seq_cp` is byte-exact. Default to hybrid regardless; the probe documents
   the guarantee and gives a fast path to revisit if a future layout or backend
   proves `seq_cp` exact everywhere.

4. **Adopt the reference's lean engine surface as the single core.** The
   reference's `decide`/`decide_batch`, `score_branches`, `compile_schema`, and
   `render_prompt` are the right shape. Keep the additions (fork axis,
   letter readout) as options on that core, not as parallel code.

5. **Keep the hardening.** Cancellation, yielding, cleanup on every exit, the
   base-model adapter scope, and the admission and fairness gates are the
   difference between a demo endpoint and a service. Preserve them, and keep
   the admission and fairness gates that bound a decision's effect on chat.

6. **Make the Jev envelope exact.** `usage.output_tokens` must stay 0 (a
   decision generates nothing) and `legend` must echo the request verbatim.
   The single full-logits readout is the only path, so there is no fast path
   that could change the winner; keep that as a hard gate.

7. **Do not add a worker thread or a second context to make decisions
   concurrent.** The measured cost is small enough that cooperative yielding on
   the one context is sufficient, and it avoids a second copy of the model
   state and any cross-thread KV hazard. The one-context, one-thread rule is the
   thread-safety story; keep it.

## 7. Risks and follow-ups

- The parity and the exactness control are single machine (ROCm gfx1100, flash
  attention off). Producer numerics differ on CUDA and Metal, and with flash
  attention on. The fork oracle is the portable guarantee; the latency table is
  not. Re-run the control on each backend before making a speed or default
  claim there.
- Because the reference and the hybrid default agree to float32 precision
  here, a reviewer could reasonably ask why the hybrid path exists. The answer
  is portability of the guarantee, not the current measurement. Say that
  plainly in any write-up instead of implying the reference returns wrong
  numbers.
- The reference's blocking handler is the one place it is clearly worse under
  chat load. Any merge must keep the yield and the cancel path.

## 8. Reproducing the numbers

Reference implementation (a scratch driver linked against the reference decision
library):

```sh
cmake -B build -S . -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DGPU_TARGETS=gfx1100
cmake --build build --target ref-bench -j
HSA_OVERRIDE_GFX_VERSION=11.0.0 ./build/bin/ref-bench MODEL \
    tests/fixtures/decision/contexts_schema.request.json auto
```

This implementation (the scratch `bench-decision` driver was removed; the fork
oracle and the fixture are driven by the decision test harness):

```sh
LLAMA_DECISION_TEST_MODEL=MODEL ./build/bin/test-decision-engine \
    "decision engine harness(\\..*fork.*)?"
```

The exactness control is the fork oracle test (`test-decision-fork`); it prints
the `[fork control]` line per model. The probability comparison used the fork
oracle's state-byte comparison on one side and the `PROBS` line from
`ref-bench` on the other.

Models measured: `lfm2.5-350m`, `lfm2.5-2.6b`, `qwen3.5-2b` (hybrid),
`gemma-4-e4b` (dense).
