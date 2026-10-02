# Jev API Reference (TypeSafe "System One" evaluation endpoint)

> Reference only: the normative contract for this server is
> `docs/decision/API.md`.

Reference for replicating the wire shape of the Jev API. Compiled from https://docs.typesafe.ai/api and its linked pages (see [Sources](#sources)), current as of 2026-09-24, model `jev-1.13.0`.

Conventions in this document:
- **[Doc]**: stated in the official docs.
- **[Derived]**: not stated verbatim, but consistent with every documented example (arithmetic checked against them).
- **[Undocumented]**: the docs do not say. Do not invent behavior; pick a defensible choice and treat it as your own.

---

## 1. Overview

Jev is a "System One" model: given a `state` (content) and a map of typed `questions`, it returns one typed, probability-based `answer` per question, under the same ids. All questions in a request are evaluated independently and in parallel against the same state; no answer is context for another. Answers are constrained to the options/levels supplied (never free text).

Three question types: `noul` (yes/no probability), `choice` (one of N options), `score` (position on ordered levels).

---

## 2. Endpoints

| Method | URL | Purpose |
|---|---|---|
| `POST` | `https://api.typesafe.ai/v1/systemone` | Evaluate state against questions |
| `GET` | `https://api.typesafe.ai/v1/models` | List model names/aliases the account can use |

Base URL: `https://api.typesafe.ai`. Python SDK reads `TYPESAFE_API_KEY`, `TYPESAFE_BASE_URL`, `TYPESAFE_DEFAULT_MODEL`.

### Headers

```http
Authorization: Bearer <API_KEY>
Content-Type: application/json
```

Response headers referenced by the SDK docs: `x-typesafe-request-id`, `Retry-After`, `retry-after-ms`.

---

## 3. `POST /v1/systemone`

### 3.1 Request body

```ts
type EntryType = string | object | unknown[] | null;   // "JSON structure"; see 3.3

interface SystemOneRequest {
  state: string | object | unknown[];                  // required
  model: string;                                       // required
  questions: Record<string, Question>;                 // required; keys are caller-chosen ids
}

type Question = NoulQuestion | ChoiceQuestion | ScoreQuestion;

interface NoulQuestion {
  type: "noul";
  instructions: EntryType;                             // required
  criteria?: { true?: EntryType; false?: EntryType };  // optional
}

interface ChoiceQuestion {
  type: "choice";
  instructions: EntryType;                             // required
  criteria: Record<string, EntryType>;                 // required; option name -> description (null allowed). Max 255 options.
}

interface ScoreQuestion {
  type: "score";
  instructions: EntryType;                             // required
  criteria: EntryType[];                               // required; ordered levels. Should have >= 2; API accepts up to 10.
}
```

Field rules [Doc]:
- `state`: plain string for text, or JSON object/array (chat logs, records, application state). Text only: no images/audio/video. Arrays are arrays of text values.
- `model`: required over HTTP. Use `"jev-latest"` (SDK default) or a versioned id (`"jev-1.13.0"`). See [Models](#5-models).
- `questions` keys: chosen by the caller, returned verbatim as the answer keys. [Doc] **The key is not sent to the model and not used in inference.** (Keys can be code-generated, e.g. `same_as_record_18`.) [Local] The "not used in inference" half holds on this server's default path only: with `permutations: 1` the key reaches nothing but the answer map, and renaming a key is byte-identical down to `diagnostics.suffix_tokens`. At `permutations >= 2` the key seeds the option order (`permutation_order(k, qid, o)`), so a renamed key can reorder the options and change the answer - measured on the reference ROCm build, up to 0.216 of reported concentration on a hybrid model at two passes. A client that never sets `permutations` gets the Jev behaviour exactly; see `docs/decision/API.md` section 2.1 for `permutations`.
- `instructions`: required on all three types. Docs' shared type table lists `null` as allowed for `EntryType`, while the API page marks `instructions` required. [Undocumented] whether `null` instructions is accepted; treat as required non-null.
- Noul `criteria`: optional. `true` = what a yes (value near 1) means; `false` = what a no (near 0) means.
- Choice `criteria`: option name -> description; use `null` when the option name needs no extra detail. Descriptions may be nested objects/arrays (e.g. taxonomy subtrees).
- Score `criteria`: array position is the level number, starting at 0. The model sees descriptions only, not level numbers or neighbors; each level is judged independently.

### Numeric extension (not part of Jev)

This branch adds two question types on top of the Jev set. They are optional
extensions, additive to the request and response; a Jev-only client never sees
them.

```ts
interface IntegerQuestion {
  type: "integer";
  instructions: EntryType;              // required
  minimum: number;                      // required, integer
  maximum: number;                      // required, integer, >= minimum
  aggregate?: "mode" | "median" | "mean";
}

interface NumberQuestion {
  type: "number";
  instructions: EntryType;              // required
  minimum: number;                      // required
  maximum: number;                      // required, >= minimum
  step?: number;                        // required unless multipleOf is set; > 0
  multipleOf?: number;                  // alias for step; provide one, not both
  aggregate?: "mode" | "median" | "mean";
}
```

The bounds define an ascending 2-255 value grid (shown to the model and scored
exactly like a `choice`); `multipleOf`/`step` must divide the range so the grid
includes both ends. Values render at fixed decimal width (no float noise).
Answers are `{type, value, probabilities, confidence}` where `value` is the
winning grid value as a typed JSON number (mode) and `probabilities` keys are
the rendered values. An `aggregate` adds a scalar summary of the same
distribution (`mean = sum(p_i * v_i)`, `median` is the value-space weighted
quantile at 0.5, `mode` is `value`); it never changes `value`. Under
`diagnostics: true` the numeric answer also carries `certainty`, `median`, and
`interval_p10_p90`, matching the `score` spread summaries.
- Duplicate/empty option names, empty `questions`, and max question count: [Undocumented].

### 3.2 Question-time semantics the model relies on [Doc]

- Option names **and** descriptions (Choice), level descriptions (Score), `instructions`, and `criteria` are all sent to the model; question ids are not.
- A question can point at part of a structured `state` by name in backticks with a dot/index path, e.g. `` Does `ticket.messages[0].text` request a refund? ``.
- Instruction object field names (`question`, `focus`, `what`, `not_for`, `examples`, ...) are **not reserved**; they are arbitrary labels the model sees along with the values.
- A Noul `instructions` may be phrased as a question or a statement to judge; high value = yes/true.

### 3.3 Structured `EntryType` locations [Doc]

`string | object | array | null` is accepted in: `instructions` (all types), Choice `criteria` values, Score `criteria` entries, Noul `criteria.true` / `criteria.false`.

Example (object instructions with referenced data):
```json
"instructions": {
  "potential_duplicate": { "name": "John Smith", "location": "Oakland, California", "last_employer": "Google" },
  "question": "Is the resume for the same person as `potential_duplicate`?"
}
```

### 3.4 Minimal request

```json
{
  "state": "Help! My payouts have been failing for 3 days.",
  "model": "jev-latest",
  "questions": {
    "is_urgent": { "type": "noul", "instructions": "Does this convey urgency?" }
  }
}
```

### 3.5 Full request (all types)

```json
{
  "state": "Help! My payouts have been failing for 3 days.",
  "model": "jev-latest",
  "questions": {
    "is_urgent": {
      "type": "noul",
      "instructions": "Does this convey urgency?",
      "criteria": { "true": "Explicitly time-sensitive", "false": "No urgency expressed" }
    },
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "billing": "Payments, invoicing, refunds",
        "technical": "Bugs, outages, integrations",
        "sales": "Pricing, upgrades, new accounts"
      }
    },
    "frustration": {
      "type": "score",
      "instructions": "How frustrated is the customer?",
      "criteria": ["Calm", "Frustrated", "Very angry"]
    }
  }
}
```

---

### 3.6 Response body (HTTP 200)

```ts
interface SystemOneResponse {
  model: string;                        // versioned id that actually answered, e.g. "jev-1.13.0" (never the alias)
  answers: Record<string, Answer>;      // one per question, same keys as request `questions`
  usage: { input_tokens: number; output_tokens: number };  // integers
}

type Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer;

interface NoulAnswer {
  type: "noul";
  noul: number;                         // 0..1, probability of yes. NO confidence field.
}

interface ChoiceAnswer {
  type: "choice";
  choice: string;                       // highest-probability option
  probabilities: Record<string, number>;  // EVERY option -> probability; sums to 1
  confidence: number;                   // 0..1
}

interface ScoreAnswer {
  type: "score";
  score: number;                        // probability-weighted level position; can fall between levels
  legend: Record<string, EntryType>;    // level index (string) -> the level's description, echoed back
  probabilities: Record<string, number>;  // level index (string) -> probability; sums to 1
  confidence: number;                   // 0..1
}
```

Notes:
- `legend` values echo the request's level description exactly as sent. The API page types them as strings; with structured (object) levels the docs' own example returns objects, e.g. `"0": {"what": "...", "examples": ["..."]}`. [Doc, from Score page]
- Probabilities are decimals (e.g. `0.0`, `0.57`, `1.0`); shown rounded to ~2 decimals in docs. JSON key order of `probabilities` is not stable (examples show differing order from the request). Do not depend on it.
- Wire keys for Score `probabilities`/`legend` are **strings** `"0"`, `"1"`, ... (the Python SDK converts them to ints).
- `usage.output_tokens` are reported but not billed.

### 3.7 Answer examples

Noul:
```json
{ "model": "jev-1.13.0",
  "answers": { "is_urgent": { "type": "noul", "noul": 0.95 } },
  "usage": { "input_tokens": 296, "output_tokens": 20 } }
```

Choice:
```json
{ "model": "jev-1.13.0",
  "answers": { "department": {
    "type": "choice", "choice": "billing",
    "probabilities": { "billing": 0.88, "technical": 0.12, "sales": 0.0 },
    "confidence": 0.81 } },
  "usage": { "input_tokens": 318, "output_tokens": 34 } }
```

Score:
```json
{ "model": "jev-1.13.0",
  "answers": { "frustration": {
    "type": "score", "score": 1.05,
    "legend": { "0": "Calm", "1": "Frustrated", "2": "Very angry" },
    "probabilities": { "0": 0.0, "1": 0.95, "2": 0.05 },
    "confidence": 0.92 } },
  "usage": { "input_tokens": 304, "output_tokens": 18 } }
```

Score with structured levels (legend echoes objects):
```json
"legend": {
  "0": { "what": "Cosmetic; no impact to functionality", "examples": ["typo in a label", "misaligned icon"] },
  "1": { "what": "Broken or degraded feature, but workaround exists", "examples": ["export fails in one browser but works in another"] },
  "2": { "what": "Blocking issue; no workaround exists", "examples": ["cannot log in", "data loss"] }
}
```

---

## 4. Answer math

### 4.1 Noul [Doc]
`noul` = P(yes) in [0,1]. Near 1 strong yes, near 0 strong no, near 0.5 uncertain. Only two outcomes, so no `confidence`. Callers threshold it themselves.

### 4.2 Choice [Doc + Derived]
- `choice` = argmax of `probabilities`. Tie behavior [Undocumented].
- `probabilities` covers all options and sums to 1.
- `confidence`: Docs say it is "derived from the probabilities": 1.0 when all mass is on one option, lower as mass spreads. The docs' 3-option demo uses `(3 * p_max - 1) / 2`. **[Derived]** the general form `confidence = (N * p_max - 1) / (N - 1)`, clamped to [0,1], reproduces every documented Choice example (2-decimal rounding), including N=3, 4, and 5:
  - p_max 0.61, N=3 -> 0.42 (doc 0.42); p_max 0.40, N=4 -> 0.20 (doc 0.20); p_max 0.74, N=5 -> 0.675 (doc 0.67); p_max 0.84, N=3 -> 0.76 (doc 0.76).
  - Minor drift exists (p_max 0.88, N=3 gives 0.82; doc example shows 0.81), plausibly from unrounded internal probabilities.

### 4.3 Score [Doc + Derived]
- `score = sum_i (i * probabilities[i])`, with `i` the 0-based level index. Range is 0 to `len(criteria) - 1`. Example: `0*0.0 + 1*0.57 + 2*0.43 = 1.43`.
- `legend[str(i)]` = `criteria[i]`.
- `confidence`: same idea as Choice (derived from `probabilities`). The 3-level formula above matches all documented 3-level Score examples (e.g. 0.57/0.43 -> 0.35; 0.74/0.26 -> 0.61; 0.91/0.09 -> 0.87). **It does NOT match the documented 4- and 5-level examples** (probabilities 0.48/0.52 over 4 levels are documented as confidence 0.52 while the formula gives 0.36; 0.86/0.14 over 5 levels documented as 0.89 vs formula 0.825). The exact Score confidence definition is [Undocumented]; for N=2..3 use the formula, and otherwise choose your own monotone "peakedness" measure and do not claim parity.
- Different distributions can give the same score; `probabilities` + `confidence` disambiguate.

### 4.4 Confidence guidance [Doc]
Callers gate actions on `confidence` (docs suggest a floor around 0.5 and stricter thresholds for high-stakes actions, e.g. >0.9). It describes the model's certainty, not correctness. Docs also say `confidence` is a default convenience and callers may compute their own from `probabilities`.

### 4.5 Local confidence and certainty vs the Jev formula [Local]

The local implementation returns two numbers on `choice` and `score`:

- `confidence = (N * p_max - 1) / (N - 1)`, clamped to [0,1]. This is Jev's
  documented Choice confidence, a linear rescale of the winner's share: it maps
  a uniform distribution to 0 and a one-hot distribution to 1, and is
  conservative at the low end, so it is the recommended basis for fallback
  gating. The Score form above K=3 stays [Undocumented], so the same monotone
  rule is the stated local value there.
- `certainty = max_i p_i`, the **winner's share**. It reads only the winner.

The Jev Choice confidence is therefore `(K * certainty - 1) / (K - 1)`, exactly
as in Section 4.2. It is not entropy: it ignores every option except the winner.
The score form above K=3 stays [Undocumented], so no exact parity claim is
possible there either.

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
- The local entropy value `1 - H/logK` remains available as an opt-in: a request with
  `confidence_profile: "local"` returns it as `confidence` (the same monotone rule above K=3,
  where Jev is undocumented). The default stays `(N*p_max - 1)/(N - 1)`, matching Jev.
  `certainty` is unaffected by the profile. The profile changes only the reported
  concentration; probabilities are identical.
- The benchmark surfaces record both: the frozen readout baselines store
  `confidence` and `certainty` per question, and the response reports
  `certainty` when `diagnostics` is set.
- Definitions: `inverse_entropy_confidence`, `winner_share` and `jev_winner_share_confidence` in
  `tools/parallel-decision/decision-protocol.cpp`.

---

## 5. Models

Only one model family is documented: **`jev-1.13.0`**.

| Property | Value |
|---|---|
| Versioned id | `jev-1.13.0` |
| Aliases | `jev-latest` -> `jev-1.13.0` (latest stable official release; SDK default). `jev-preview` -> `jev-1.13.0` (latest release incl. previews; currently same model, no preview build). |
| Input | Text only: string, JSON object, or array of text values |
| Context | 64k tokens per request (state + all questions); 32k tokens for state + the single longest question |
| Price | Per input token: $42 / billion tokens ($0.042 / million). Output tokens free. |
| Rate limits | 250,000 tokens/second and 1,200 requests/minute (docs: "adjusting dynamically", may change without notice) |
| Language | English primary and best; other languages (incl. CJK) accepted with lower accuracy |
| Customization | None per-account (no fine-tuning); all shaping happens through `state`, `instructions`, `criteria` |

Behavior [Doc]:
- Aliases move when releases ship; the response `model` field always reports the resolved versioned id.
- Versioned ids are accepted in `model` whether or not `GET /v1/models` lists them.
- Behavior for an unknown model name, and behavior when the context limit is exceeded: [Undocumented].
- Only requests exceeding either rate limit return 429.

### `GET /v1/models`

```bash
curl https://api.typesafe.ai/v1/models -H "Authorization: Bearer $TYPESAFE_API_KEY"
```
Response shape:
```ts
{ models: Array<{ name: string; description: string; release_date: string }> }
```
Currently lists aliases (`jev-latest`, `jev-preview`). Exact `release_date` format and current values: [Undocumented].

---

## 6. Errors

JSON body describing the failure. Exact body schema is [Undocumented]; the SDK exposes "the server's JSON error body, plain response text, or none".

| Status | Meaning | Retry? |
|---|---|---|
| 401 Unauthorized | Missing or invalid API key | No |
| 422 Unprocessable Entity | Request failed validation (missing required field, malformed question); body names the offending field | No |
| 429 Too Many Requests | Rate limit exceeded (token/s or requests/min); may carry `Retry-After` / `retry-after-ms` | Yes, exponential backoff |
| 529 Overloaded | Temporarily overloaded | Yes, exponential backoff |

Additional statuses the Python SDK maps to typed errors [Doc, SDK]: 400 Bad Request, 403 Permission Denied, 404 Not Found, other 5xx Internal Server Error.

### Official Python SDK client behavior (useful for compatible clients)
- Default retry set: HTTP 408, 429, and all 5xx; also retries connection errors and timeouts.
- `max_retries=2` (after initial attempt), backoff starts 0.5s, doubles, capped at 5.0s, jitter 0.25 (fraction subtracted), honors `Retry-After` and `retry-after-ms`.
- Default per-HTTP-operation timeout 10.0s; total retry budget per call 30.0s.
- Response validation error path example: `answers.tone.confidence`.

---

## 7. Wire-compatibility checklist for a replica

1. Accept `POST /v1/systemone` with Bearer auth; validate `state`, `model`, `questions` as required (422 on failure with a body naming the field).
2. Accept `state` as string | object | array; accept `instructions`/criteria descriptions as string | object | array | null.
3. Echo answers under the caller's keys; never leak/require the keys inside the model prompt. [Local] This is exact at the default `permutations: 1`; a multi-pass request additionally seeds the option order from the key, which is an opt-in local extension (see 3.1).
4. Return `model` as the resolved versioned id (map aliases like `jev-latest` -> your resolved id), `answers`, and `usage.{input_tokens, output_tokens}` (integers).
5. Noul: `{type, noul}` only. Choice: `{type, choice, probabilities (all options, sums to 1), confidence}`. Score: `{type, score, legend, probabilities, confidence}` with string level keys `"0".."n-1"`.
6. `score = sum(i * p_i)`; `choice = argmax(p)`; `legend[i]` = original level description object (unchanged, including objects).
7. Enforce/document limits: Choice <= 255 options; Score 2..10 levels; questions independent; evaluate every question against the full state.
8. Emit 401/422/429/529 as documented; send `Retry-After` on 429/529 so retrying clients behave.
9. Because callers gate on `confidence`, a local model that cannot produce calibrated distributions should produce conservative (lower) confidence; Jev's docs treat confidence as the basis for fallback/escalation decisions.

---

## 8. Not documented (do not assume Jev behavior)

- Error response JSON schema; exact validation error text.
- Maximum number of questions per request; maximum `state` size beyond the token budgets above.
- Handling of empty `questions`, duplicate/empty option names, `criteria` with 0 or 1 Score levels (docs: "should have at least two"), or `null` `instructions`.
- Tie-breaking in `choice`; exact Score `confidence` definition for N > 3; decimal precision of returned floats.
- Streaming, idempotency keys, versioning headers, request-size limits, or other endpoints (docs show only the two above).
- Unknown-model error status; context-overflow error status.

---

## Sources

- API reference: https://docs.typesafe.ai/api
- Models (+ `GET /v1/models`): https://docs.typesafe.ai/models
- Primitives: https://docs.typesafe.ai/primitives, /primitives/choice, /primitives/score, /primitives/noul, /primitives/advanced
- State: https://docs.typesafe.ai/concepts/state
- Confidence: https://docs.typesafe.ai/confidence
- Python SDK exceptions, retries, constants: https://docs.typesafe.ai/sdk/python/api/exceptions, /sdk/python/api/retries, /sdk/python/api/constants
- Full index: https://docs.typesafe.ai/llms.txt
- Known model quirks (not reviewed here): https://docs.typesafe.ai/model-jaggedness/jev-1.13
