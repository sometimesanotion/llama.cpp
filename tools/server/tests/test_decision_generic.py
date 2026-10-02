#!/usr/bin/env python3
"""The generic `schema` decision surface, as the refactor's characterization gate.

The Jev `questions` shape and the generic `schema` shape are two front-ends over one engine. The
generic shape exists because it can express what the Jev letter readout cannot: a field wider
than the tokenizer's realized answer-label pool, typed value grids, and numeric aggregates. Those
are the capabilities meant to survive, so they are pinned here before any refactor moves them.

This suite is about the contract, not the weights. It asserts the structural invariants that hold
for any model (a value is always an allowed value, a probability is in [0,1], the value band is
ordered and inside the range, `mode` equals the winning value, one result per context) plus the
rejection surface. It never asserts a particular answer, so it does not fail on a weak model.

Skips cleanly (exit 0) when the server binary or a model is missing.
"""

import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Server, http, prefill_slot, supports_letter_labels, check, LETTER_SYSTEM,
# SERVER_BIN, MODEL_CANDIDATES
_env_spec = importlib.util.spec_from_file_location("decision_envelope", os.path.join(HERE, "test_decision_envelope.py"))
env = importlib.util.module_from_spec(_env_spec)
_env_spec.loader.exec_module(env)

STATE = "The service was unreachable for ten minutes before the backup took over."

# The realized answer-label pool is shrunk through the documented override so the wide-domain
# contrast below runs against any model, instead of only one whose tokenizer resolves few
# single-token labels. The pool is built once, on the first Jev decision.
POOL_CAP = 8


def body_for(schema, **extra):
    out = {"model": "test", "schema": schema}
    # `contexts` replaces the single state: the request shape accepts one or the other. The key is
    # chosen after the pop, or the lookup would always see the consumed entry.
    contexts = extra.pop("contexts", None)
    out["contexts" if contexts is not None else "state"] = STATE if contexts is None else contexts
    out.update(extra)
    return out


def post_schema(server, schema, **extra):
    return server.post("/v1/decision", json.dumps(body_for(schema, **extra)))


# The realized answer-label pool for this model, taken from a Jev decision's diagnostics. The
# generic shape exists precisely to go past it, so the contrast below needs the real number.
def label_pool_size(server):
    payload = {
        "model": "test", "state": STATE, "diagnostics": True,
        "questions": {"q": {"type": "noul", "instructions": "ok?"}},
    }
    status, text = server.post("/v1/decision", json.dumps(payload))
    if status != 200:
        return None
    return json.loads(text).get("diagnostics", {}).get("label_pool_size")


def check_record(record, values, numeric, label, diagnostics=False):
    """The shape and invariants every scored field must satisfy, whatever its type.

    The generic path extends the Jev answer rather than replacing it, so a record carries the Jev
    keys (type, confidence, a full probabilities map, a legend) plus the generic `value`. The
    spread summaries are diagnostics-only, matching the Jev score discipline.
    """
    env.check(isinstance(record, dict), f"{label}: the field record is an object")
    for key in ("type", "value", "confidence", "probabilities", "legend", "scored"):
        env.check(key in record, f"{label}: the field record carries {key}")
    env.check(any(v == record["value"] for v in values),
              f"{label}: the value is one of the allowed values ({record['value']})")
    env.check(0.0 <= record["confidence"] <= 1.0, f"{label}: confidence is in [0,1]")

    # Jev probability discipline: a map over every allowed value that sums to 1
    probs = record["probabilities"]
    env.check(isinstance(probs, dict) and len(probs) == len(values),
              f"{label}: the probability map covers every allowed value ({sorted(probs)})")
    env.check(abs(sum(probs.values()) - 1.0) < 1e-6, f"{label}: the probabilities sum to 1")
    env.check(all(0.0 <= p <= 1.0 for p in probs.values()), f"{label}: every probability is in [0,1]")
    env.check(record["scored"] in ("tree", "argmax"), f"{label}: the scored mode is reported")
    if record["scored"] == "argmax":
        env.check(sum(1 for p in probs.values() if p > 0.0) == 1,
                  f"{label}: a greedily scored field is the point mass it chose")

    # the legend is the Jev ScoreAnswer shape: the same keys, mapped to their values
    env.check(set(record["legend"]) == set(probs), f"{label}: the legend covers the same keys")

    # off-envelope fields never appear unless asked for
    for key in ("interval_p10_p90", "aggregate"):
        if not diagnostics:
            env.check(key not in record, f"{label}: {key} is diagnostics-only")
    if not numeric or not diagnostics:
        return record
    env.check("interval_p10_p90" in record, f"{label}: a numeric field carries interval_p10_p90")
    env.check("aggregate" in record, f"{label}: a numeric field carries aggregate")
    band = record["interval_p10_p90"]
    env.check(isinstance(band, list) and len(band) == 2, f"{label}: the band has two entries")
    lo, hi = band
    numbers = [v for v in values if isinstance(v, (int, float))]
    env.check(lo <= hi, f"{label}: the band is ordered ({lo} <= {hi})")
    env.check(min(numbers) <= lo and hi <= max(numbers), f"{label}: the band is inside the value range")
    return record


def run_checks(server):
    # 1. The response envelope. The generic path extends the Jev envelope, so the default answer is
    #    exactly the Jev shape: model + answers + a reduced usage, with no timings and no
    #    off-envelope counters. The additive counters and timings are opt-in.
    status, text = post_schema(server, {
        "active":   {"type": "boolean", "description": "is the incident active"},
        "severity": {"type": "enum", "description": "incident severity", "enum": ["low", "medium", "high"]},
    })
    env.check(status == 200, f"typed schema status {status}: {text[:200]}")
    if status != 200:
        return
    doc = json.loads(text)
    env.check(set(doc) == {"model", "answers", "usage"},
              f"the default envelope is the Jev envelope: {sorted(doc)}")
    env.check(isinstance(doc["answers"], dict), "answers is a map keyed by field name")
    env.check(set(doc["answers"]) == {"active", "severity"}, "every declared field is answered")
    env.check(set(doc["usage"]) == {"input_tokens", "output_tokens"},
              f"the strict usage is input/output only: {sorted(doc['usage'])}")
    env.check(doc["usage"]["output_tokens"] == 0, "a decision still generates nothing")

    # the opt-in carries the additive counters, timings, and the per-field spread summaries
    status, text = post_schema(server, {
        "active":   {"type": "boolean", "description": "is the incident active"},
        "severity": {"type": "enum", "description": "incident severity", "enum": ["low", "medium", "high"]},
        "sev":      {"type": "integer", "description": "severity", "minimum": 0, "maximum": 4},
    }, diagnostics=True)
    env.check(status == 200, f"the diagnostics status {status}: {text[:200]}")
    if status == 200:
        ddoc = json.loads(text)
        for key in ("prefill_ms", "scoring_ms", "total_ms", "rounds", "rows", "per_decision_ms"):
            env.check(key in ddoc["timings"], f"timings carry {key}")
        for key in ("input_tokens", "output_tokens", "cached_tokens", "state_cache_hit"):
            env.check(key in ddoc["usage"], f"the diagnostics usage carries {key}")

    # 2. A boolean and an enum, in the Jev answer shape.
    check_record(doc["answers"]["active"], [True, False], False, "boolean")
    check_record(doc["answers"]["severity"], ["low", "medium", "high"], False, "enum")

    # 3. Numeric grids: the inclusive integer range and the stepped number grid.
    status, text = post_schema(server, {
        "count":  {"type": "integer", "description": "affected rows", "minimum": 1, "maximum": 4},
        "impact": {"type": "number", "description": "impact factor", "minimum": 0.0, "maximum": 1.0,
                   "step": 0.5},
    })
    env.check(status == 200, f"numeric schema status {status}: {text[:200]}")
    if status == 200:
        fields = json.loads(text)["answers"]
        rec = check_record(fields["count"], list(range(1, 5)), True, "integer")
        env.check(isinstance(rec["value"], int), f"an integer field yields an integer value ({rec['value']})")
        rec = check_record(fields["impact"], [0.0, 0.5, 1.0], True, "number")
        env.check(isinstance(rec["value"], (int, float)) and not isinstance(rec["value"], bool),
                  f"a number field yields a numeric value ({rec['value']})")

    # 4. The aggregates, which are diagnostics-only, so the default numeric answer stays on the
    #    envelope. `mode` is the default and leaves the aggregate on the winning value; `median` and
    #    `mean` are value-space reductions, so they stay inside the value range.
    for aggregate, in (("mode",), ("median",), ("mean",)):
        status, text = post_schema(server, {
            "sev": {"type": "integer", "description": "severity", "minimum": 0, "maximum": 4,
                    "aggregate": aggregate},
        }, diagnostics=True)
        env.check(status == 200, f"the {aggregate} aggregate status {status}: {text[:200]}")
        if status != 200:
            continue
        rec = check_record(json.loads(text)["answers"]["sev"], list(range(5)), True, aggregate,
                           diagnostics=True)
        env.check(0 <= rec["aggregate"] <= 4, f"the {aggregate} aggregate is inside the value range")
        if aggregate == "mode":
            env.check(rec["aggregate"] == rec["value"], "the mode aggregate is the winning value")
    status, text = post_schema(server, {
        "sev": {"type": "integer", "description": "severity", "minimum": 0, "maximum": 4},
    }, diagnostics=True)
    env.check(status == 200, f"the default aggregate status {status}: {text[:200]}")
    if status == 200:
        rec = json.loads(text)["answers"]["sev"]
        env.check(rec["aggregate"] == rec["value"], "the default aggregate is mode")

    # 5. The capability that earns this front-end its place: a field wider than the tokenizer's
    #    realized answer-label pool. The Jev letter readout needs one label per option and refuses
    #    to go past the pool; the generic shape scores the values themselves and has no such bound.
    #    The pool is shrunk with the documented override so this runs on any model rather than only
    #    on one whose tokenizer happens to resolve few single-token labels.
    pool = label_pool_size(server)
    env.check(pool is not None, "the realized answer-label pool is reported")
    if pool is None:
        return
    env.check(pool == POOL_CAP, f"the pool override is honored ({pool} labels, asked for {POOL_CAP})")
    wide = [f"option-{i}" for i in range(POOL_CAP + 8)]
    env.check(len(wide) > pool, f"the test domain is wider than the pool ({len(wide)} > {pool})")
    status, text = post_schema(server, {
        "wide": {"type": "enum", "description": "a wide catalogue", "enum": wide},
    })
    env.check(status == 200, f"a {len(wide)}-value enum is served with only {pool} labels: "
                             f"{status} {text[:200]}")
    if status == 200:
        rec = json.loads(text)["answers"]["wide"]
        env.check(rec["value"] in wide, f"the wide field's value is one of its choices ({rec['value']})")
    # the same domain as a Jev choice is refused for want of labels: this is the whole contrast
    jev = {"q": {"type": "choice", "instructions": "which one?",
                 "criteria": {k: "" for k in wide}}}
    status, text = server.post("/v1/decision", json.dumps(
        {"model": "test", "state": STATE, "questions": jev}))
    env.check(status == 422, f"the Jev choice over the same {len(wide)} domain is refused: "
                             f"{status} {text[:160]}")
    env.check("label" in text, f"the refusal names the answer label: {text[:160]}")

    # 6. The JSON Schema body form: `properties`, a description is optional, `multipleOf` is the
    #    numeric step key.
    status, text = post_schema(server, {
        "type": "object",
        "properties": {
            "active": {"type": "boolean"},
            "level":  {"type": "integer", "minimum": 0, "maximum": 2},
            "grade":  {"type": "number", "minimum": 0.0, "maximum": 1.0, "multipleOf": 0.25},
        },
    })
    env.check(status == 200, f"the JSON Schema body status {status}: {text[:200]}")
    if status == 200:
        fields = json.loads(text)["answers"]
        env.check(set(fields) == {"active", "level", "grade"},
                  f"the properties object is the catalogue: {sorted(fields)}")
        check_record(fields["level"], [0, 1, 2], True, "json-schema integer")

    # 7. Many fields at once: the catalogue is scored in one batched pass and every field is scored.
    many = {f"f{i}": {"type": "boolean", "description": f"flag {i}"} for i in range(16)}
    status, text = post_schema(server, many)
    env.check(status == 200, f"a 16-field schema status {status}: {text[:200]}")
    if status == 200:
        env.check(len(json.loads(text)["answers"]) == 16, "all 16 fields are scored")

    # 8. Several contexts in one request: one result per context, in request order.
    contexts = ["The service was unreachable for ten minutes.",
                "A user could not log in for an hour.",
                "Nothing unusual happened today."]
    status, text = post_schema(server, {
        "severity": {"type": "enum", "description": "incident severity", "enum": ["low", "medium", "high"]},
    }, contexts=contexts)
    env.check(status == 200, f"the multi-context status {status}: {text[:200]}")
    if status == 200:
        # several contexts use the documented Jev `contexts` array, not a second list shape
        mdoc = json.loads(text)
        env.check(set(mdoc) == {"model", "contexts"},
                  f"the multi-context envelope is the Jev one: {sorted(mdoc)}")
        ctxts = mdoc["contexts"]
        env.check(len(ctxts) == len(contexts),
                  f"one entry per context ({len(ctxts)} vs {len(contexts)})")
        env.check(all(set(c) == {"answers", "usage"} for c in ctxts),
                  "each context entry carries answers and usage")
        env.check(all(set(c["answers"]) == {"severity"} for c in ctxts),
                  "every context scores the whole catalogue")
        env.check(all(set(c["usage"]) == {"input_tokens", "output_tokens"} for c in ctxts),
                  "each context carries the strict usage")

    # 9. The echoed model follows the same contract as the Jev shape.
    for requested, resolves in (("jev-latest", True), ("jev-preview", True), ("my-model-v9", False), ("", True)):
        status, text = server.post("/v1/decision", json.dumps(
            body_for({"a": {"type": "boolean", "description": "d"}}, model=requested)))
        env.check(status == 200, f"the model echo for {requested!r} status {status}: {text[:160]}")
        if status != 200:
            continue
        echoed = json.loads(text)["model"]
        if resolves:
            env.check(echoed != requested, f"the alias {requested!r} resolves to the loaded model")
        else:
            env.check(echoed == requested, f"a concrete id is echoed verbatim ({echoed})")

    # 10. The rejection surface: a malformed schema is a client error, and a body carrying both
    #     shapes cannot be dispatched at all.
    for schema, needle in (
        ({}, "1-32 fields"),
        ({"a": {"type": "boolean"}}, "needs a description"),
        ({"a": {"type": "wat", "description": "d"}}, "boolean, enum, integer and number"),
        ({"a": {"type": "enum", "description": "d"}}, "need a list of choices"),
        ({"a": {"type": "enum", "description": "d", "enum": ["x", "x"]}}, "duplicate"),
        ({"a": {"type": "integer", "description": "d", "minimum": 0, "maximum": 300}}, "1-255 values"),
        ({"a": {"type": "integer", "description": "d", "minimum": 0}}, "minimum and maximum"),
        ({"a": {"type": "number", "description": "d", "minimum": 0.0, "maximum": 1.0, "step": -1}}, "positive step"),
        ({"a": {"type": "boolean", "description": "d", "aggregate": "mean"}}, "median/mean for numeric"),
    ):
        status, text = post_schema(server, schema)
        env.check(status == 422, f"a malformed schema is a 422: {status} {text[:160]}")
        env.check(needle in text, f"the refusal names the rule {needle!r}: {text[:160]}")

    status, text = post_schema(server, {"a": {"type": "boolean", "description": "d"}},
                               questions={"q": {"type": "noul", "instructions": "x"}})
    env.check(status == 400, f"a body carrying both shapes is a 400: {status} {text[:160]}")
    env.check("not both" in text, f"the refusal names the mutual exclusion: {text[:160]}")

    # 10b. A field of the wrong JSON type is invalid decision content, not malformed syntax: it
    #     answers 422 with the field's own name in the message. The control below proves the
    #     syntax class is untouched: a truncated body is still 400.
    for schema, needle in (
        ({"a": {"type": 5, "description": "d"}}, "type must be a string"),
        ({"a": {"type": "enum", "description": 7}}, "description must be a string"),
        ({"a": {"type": "boolean", "description": "d", "aggregate": 3}}, "aggregate must be a string"),
        ({"a": {"type": "boolean", "description": "d", "x-aggregate": []}}, "x-aggregate must be a string"),
        ({"a": {"type": "integer", "description": "d", "minimum": "0", "maximum": 2}},
         "minimum must be an integer"),
        ({"a": {"type": "integer", "description": "d", "minimum": 0, "maximum": 2.5}},
         "maximum must be an integer"),
        ({"a": {"type": "number", "description": "d", "minimum": "0", "maximum": 1, "step": 0.5}},
         "minimum must be a number"),
        ({"a": {"type": "number", "description": "d", "minimum": 0, "maximum": "1", "step": 0.5}},
         "maximum must be a number"),
        ({"a": {"type": "number", "description": "d", "minimum": 0, "maximum": 1, "step": True}},
         "step must be a number"),
    ):
        status, text = post_schema(server, schema)
        env.check(status == 422, f"a wrong-typed field is a 422: {status} {text[:160]}")
        env.check(needle in text, f"the refusal names the field {needle!r}: {text[:160]}")

    for payload, needle in (
        ({"model": 5, "state": STATE, "schema": {"a": {"type": "boolean", "description": "d"}}},
         "model must be a string"),
        ({"model": "t", "state": STATE, "instructions": 7,
          "schema": {"a": {"type": "boolean", "description": "d"}}}, "instructions must be a string"),
        ({"model": "t", "state": STATE, "schema": {"a": {"type": "boolean", "description": "d"}},
          "mode": 1}, "mode must be a string"),
        ({"model": "t", "state": STATE, "schema": {"a": {"type": "boolean", "description": "d"}},
          "tree_max": "8"}, "tree_max must be an integer"),
        ({"model": "t", "state": STATE, "schema": {"a": {"type": "boolean", "description": "d"}},
          "cache_prompt": "yes"}, "cache_prompt must be a boolean"),
        ({"model": "t", "state": STATE, "schema": {"a": {"type": "boolean", "description": "d"}},
          "diagnostics": "yes"}, "diagnostics must be a boolean"),
    ):
        status, text = server.post("/v1/decision", json.dumps(payload))
        env.check(status == 422, f"a wrong-typed request field is a 422: {status} {text[:160]}")
        env.check(needle in text, f"the refusal names the field {needle!r}: {text[:160]}")

    status, text = server.post("/v1/decision", '{"model": "t", "state":')
    env.check(status == 400, f"a truncated body stays a 400: {status} {text[:160]}")

    status, text = post_schema(server, {"a": {"type": "boolean", "description": "d"}})
    env.check(status == 200, f"the control body still answers 200: {status} {text[:160]}")

    for payload, needle in (
        ({"state": STATE, "schema": {"a": {"type": "boolean", "description": "d"}}}, "model is required"),
        ({"model": "t", "schema": "nope"}, "must be an object"),
        ({"model": "t", "schema": {"a": {"type": "boolean", "description": "d"}}}, "state"),
    ):
        status, text = server.post("/v1/decision", json.dumps(payload))
        env.check(status == 422, f"a missing request part is a 422: {status} {text[:160]}")
        env.check(needle in text, f"the refusal names {needle!r}: {text[:160]}")

    # 10. The knobs the generic shape shares with the Jev shape. They used to be parsed and then
    #     ignored here; both shapes now resolve them the same way, so each accepted value must move
    #     the answer in the direction it names. Nothing below asserts a particular answer, only that
    #     the knob is honoured.
    schema = {
        "active":   {"type": "boolean", "description": "is the incident active"},
        "severity": {"type": "enum", "description": "incident severity", "enum": ["low", "medium", "high"]},
        "count":    {"type": "integer", "description": "affected rows", "minimum": 1, "maximum": 4},
    }

    def answers_of(**extra):
        status, text = post_schema(server, schema, **extra)
        env.check(status == 200, f"the {sorted(extra)} status {status}: {text[:200]}")
        return json.loads(text)["answers"] if status == 200 else None

    # a forced greedy mode scores the field by argmax, so it reports the point mass it chose
    greedy = answers_of(mode="greedy", diagnostics=True)
    if greedy is not None:
        check_record(greedy["active"], [True, False], False, "greedy boolean", diagnostics=True)
        env.check(all(rec["scored"] == "argmax" for rec in greedy.values()),
                  f"mode greedy scores every field by argmax: "
                  f"{ {k: v['scored'] for k, v in greedy.items()} }")
    # a tree_max below the widest value space moves that field to greedy and leaves the narrow ones
    narrow = answers_of(mode="auto", tree_max=2)
    if narrow is not None:
        check_record(narrow["severity"], ["low", "medium", "high"], False, "tree_max enum")
        env.check(narrow["severity"]["scored"] == "argmax",
                  f"tree_max 2 puts a three-value field on the greedy path: {narrow['severity']['scored']}")
        env.check(narrow["active"]["scored"] == "tree",
                  f"tree_max 2 leaves a two-value field on the trie: {narrow['active']['scored']}")
    # a forced tree mode keeps the distribution however narrow the bound
    tree = answers_of(mode="tree", tree_max=2)
    if tree is not None:
        env.check(all(rec["scored"] == "tree" for rec in tree.values()),
                  f"mode tree keeps every field on the trie: { {k: v['scored'] for k, v in tree.items()} }")

    # the reported profile follows the request, and never moves a probability
    default_answers = answers_of()
    jev_answers = answers_of(confidence_profile="jev")
    local_answers = answers_of(confidence_profile="local")
    if default_answers and jev_answers and local_answers:
        env.check([jev_answers[k]["confidence"] for k in sorted(jev_answers)] ==
                  [default_answers[k]["confidence"] for k in sorted(default_answers)],
                  "an explicit jev profile is what the default already reports")
        env.check(any(abs(local_answers[k]["confidence"] - default_answers[k]["confidence"]) > 1e-9
                      for k in local_answers),
                  "the local profile reports a different concentration")

    # temperature is argmax-invariant but not distribution-invariant: a colder field must flatten
    hot = answers_of(temperature=1.0)
    cold = answers_of(temperature=0.05)
    if hot and cold:
        env.check(all(hot[k]["value"] == cold[k]["value"] for k in hot),
                  "a temperature change does not move a winner")
        env.check(any(cold[k]["confidence"] != hot[k]["confidence"] for k in hot),
                  "a colder temperature sharpens the reported concentration")

    # per-type temperatures resolve against the field's own type, not globally: only the numeric
    # fields read the `integer` override
    typed = answers_of(temperatures={"integer": 0.05})
    if typed and default_answers:
        numeric_changed = [k for k in typed
                           if typed[k]["type"] in ("integer", "number")
                           and typed[k]["probabilities"] != default_answers[k]["probabilities"]]
        other_unchanged = [k for k in typed
                           if typed[k]["type"] not in ("integer", "number")
                           and typed[k]["probabilities"] == default_answers[k]["probabilities"]]
        env.check(numeric_changed, f"the integer override moved the numeric fields: {sorted(typed)}")
        env.check(other_unchanged,
                  f"the integer override left the other types alone: {sorted(typed)}")
    # an unlisted type key is still refused: the wire vocabulary is the five primitives
    status, text = post_schema(server, schema, temperatures={"boolean": 0.5})
    env.check(status == 422, f"an unlisted temperatures key is refused: {status} {text[:160]}")
    env.check("unknown field" in text, f"the refusal names the rule: {text[:160]}")

    # order de-biasing: more passes is more work, visible in the reported rounds, and the answer is
    # still a well-formed record over the same value space
    one_status, one_text = post_schema(server, schema, diagnostics=True)
    env.check(one_status == 200, f"the one-pass status {one_status}: {one_text[:200]}")
    two_status, two_text = post_schema(server, schema, permutations=2, diagnostics=True)
    env.check(two_status == 200, f"the two-pass status {two_status}: {two_text[:200]}")
    if one_status == 200 and two_status == 200:
        one_doc, two_doc = json.loads(one_text), json.loads(two_text)
        allowed = {
            "active":   [True, False],
            "severity": ["low", "medium", "high"],
            "count":    list(range(1, 5)),
        }
        env.check(two_doc["timings"]["rounds"] >= one_doc["timings"]["rounds"],
                  f"two passes score at least as many rounds as one "
                  f"({one_doc['timings']['rounds']} -> {two_doc['timings']['rounds']})")
        for name, rec in two_doc["answers"].items():
            check_record(rec, allowed[name], rec["type"] in ("integer", "number"), f"two-pass {name}",
                         diagnostics=True)
            env.check(set(rec["probabilities"]) == set(one_doc["answers"][name]["probabilities"]),
                      f"two passes cover the same value space as one: {sorted(rec['probabilities'])}")

    # 11. A session: the schema is answered against a prefilled slot without re-prefilling it.
    env.prefill_slot(server, 0, env.LETTER_SYSTEM, "State:\n" + STATE + "\n")
    schema = {"severity": {"type": "enum", "description": "incident severity",
                           "enum": ["low", "medium", "high"]}}
    status, text = post_schema(server, schema, id_slot=0)
    env.check(status == 200, f"a schema session decision status {status}: {text[:200]}")
    if status == 200:
        sdoc = json.loads(text)
        env.check("answers" in sdoc, "a session answer is the Jev envelope")
        env.check("contexts" not in sdoc, "a session scores exactly one context")
    status, text = post_schema(server, schema, id_slot=1)
    env.check(status in (400, 422), f"an empty slot is refused: {status} {text[:160]}")

    # 12. A session handle: the same schema answered from a first-class session.
    status, text = server.post("/v1/session", json.dumps({"id_slot": 0, "turn": "t1"}))
    env.check(status == 200, f"the session create status {status}: {text[:200]}")
    if status == 200:
        sid = json.loads(text)["session_id"]
        status, text = post_schema(server, schema, session_id=sid)
        env.check(status == 200, f"a schema decision by session_id status {status}: {text[:200]}")
        status, text = post_schema(server, schema, session_id=sid, turn="wrong")
        env.check(status == 422, f"a mismatched turn is a 422: {status} {text[:160]}")
        env.check("turn" in text, f"the refusal names turn: {text[:160]}")


# The envelope key set, i.e. everything outside the answer records themselves. Both front-ends answer
# through one envelope, so a caller switching a field from a Jev primitive to a typed schema sees
# the answer's shape change and nothing else move. This is the superset claim, spelled out as a key
# comparison rather than prose.
def envelope_keys(doc):
    keys = set(doc)
    keys.discard("answers")
    return keys


# The Jev questions that mean the same thing as `SYM_SCHEMA` below, so the two shapes can be asked
# the same question about the same evidence and their envelopes compared.
SYM_QUESTIONS = {
    "severity": {"type": "choice", "instructions": "incident severity",
                 "criteria": {"low": "low", "medium": "medium", "high": "high"}},
    "count": {"type": "score", "instructions": "affected rows",
              "criteria": ["0", "1", "2", "3"]},
}

SYM_SCHEMA = {
    "severity": {"type": "enum", "description": "incident severity", "enum": ["low", "medium", "high"]},
    "count": {"type": "integer", "description": "affected rows", "minimum": 0, "maximum": 3},
}


def run_envelope_checks(server):
    """The generic shape reports the same envelope as the Jev shape, not a thinner one.

    The answer records genuinely differ - a typed `value` over a field's value space instead of a
    Jev `choice`/`score` - and that difference is the feature. Everything around them is identical
    work that both front-ends must report identically, or a client using the escape hatch has to
    accept a worse envelope than a Jev client does.
    """
    # 1. The opt-in diagnostics object: the same keys the Jev shape reports, and its own prompt
    #    version, because the two shapes frame different prompts over the same engine.
    status, text = post_schema(server, SYM_SCHEMA, diagnostics=True)
    env.check(status == 200, f"the diagnostics status {status}: {text[:200]}")
    if status != 200:
        return
    diag_doc = json.loads(text)
    for key in ("contract_hash", "prompt_version", "prefill_ms", "scoring_ms", "suffix_tokens",
                "common_suffix_tokens", "leaf_suffix_tokens", "label_pool_size", "permutations",
                "adapters_configured", "adapter_scope", "model", "quantization", "template_hash",
                "backend_flags", "token_cache_hits", "token_cache_misses"):
        env.check(key in diag_doc["diagnostics"], f"the generic diagnostics carry {key}")
    env.check(diag_doc["diagnostics"]["prompt_version"] == "schema-v1",
              f"the generic readout names its own prompt version: "
              f"{diag_doc['diagnostics'].get('prompt_version')}")
    env.check(diag_doc["diagnostics"]["adapter_scope"] == "base",
              "a stateless decision decodes on the base model")
    env.check(diag_doc["diagnostics"]["suffix_tokens"] > 0,
              f"the generic plan reports its suffix tokens: {diag_doc['diagnostics']['suffix_tokens']}")

    # 2. `certainty` is additive: absent by default, present on a numeric field under diagnostics,
    #    and equal to the winner's share of the reported distribution.
    plain_status, plain_text = post_schema(server, SYM_SCHEMA)
    env.check(plain_status == 200, f"the default status {plain_status}: {plain_text[:200]}")
    if plain_status != 200:
        return
    plain = json.loads(plain_text)
    env.check("certainty" not in plain["answers"]["count"],
              "certainty is diagnostics-only on a generic field")
    rec = diag_doc["answers"]["count"]
    env.check("certainty" in rec, "a numeric field carries certainty under diagnostics")
    env.check(abs(rec["certainty"] - max(rec["probabilities"].values())) < 1e-9,
              f"certainty is the winner's share: {rec.get('certainty')} vs {rec['probabilities']}")

    # 3. timings: additive with the opt-in, absent otherwise. Checked here as well as in the
    #    structural pass above because it is the envelope, not the answer, that carries them.
    env.check("timings" in diag_doc, "the opt-in response carries timings")
    env.check("timings" not in plain, "the default response carries no timings")

    # 4. Session symmetry: the same schema answered from a session reports the fork the Jev shape
    #    reports for the same session. This is the check that fails before the shared envelope
    #    exists: the generic branch had no diagnostics block and no session fields at all.
    env.prefill_slot(server, 0, env.LETTER_SYSTEM, "State:\n" + STATE + "\n")
    status, g_session_text = post_schema(server, SYM_SCHEMA, id_slot=0)
    env.check(status == 200, f"the generic session status {status}: {g_session_text[:200]}")
    jev_body = {"model": "test", "state": STATE, "questions": SYM_QUESTIONS, "id_slot": 0}
    status, j_session_text = server.post("/v1/decision", json.dumps(jev_body))
    env.check(status == 200, f"the Jev session status {status}: {j_session_text[:200]}")
    if status != 200:
        return
    g_session = json.loads(g_session_text)
    j_session = json.loads(j_session_text)
    env.check(g_session.get("session_fork") is True,
              f"a generic session decision reports its fork: {sorted(g_session)}")
    env.check(envelope_keys(g_session) == envelope_keys(j_session),
              f"the two shapes report the same session envelope: "
              f"{sorted(envelope_keys(g_session))} vs {sorted(envelope_keys(j_session))}")
    env.check(g_session["source_slot"] == j_session["source_slot"],
              f"both shapes name the same source slot: {g_session['source_slot']} vs "
              f"{j_session['source_slot']}")
    env.check(g_session["session_pos"] == j_session["session_pos"],
              f"both shapes report the same fork position: {g_session['session_pos']} vs "
              f"{j_session['session_pos']}")
    env.check(g_session["usage"]["output_tokens"] == 0,
              "a generic session decision still generates nothing")

    # 5. Cross-shape golden: one schema and one question set that mean the same thing, answered on
    #    the same server, share the envelope key set outside the answer records. The default and the
    #    opt-in are both compared, because both are gates on the same emitter.
    for extra, label in (({}, "default"), ({"diagnostics": True}, "diagnostics")):
        status, g_text = post_schema(server, SYM_SCHEMA, **extra)
        jev_body = dict({"model": "test", "state": STATE, "questions": SYM_QUESTIONS}, **extra)
        j_status, j_text = server.post("/v1/decision", json.dumps(jev_body))
        env.check(status == 200 and j_status == 200,
                  f"the cross-shape pair ({label}) answered: {status} {j_status}")
        if status != 200 or j_status != 200:
            continue
        g_doc, j_doc = json.loads(g_text), json.loads(j_text)
        env.check(envelope_keys(g_doc) == envelope_keys(j_doc),
                  f"the two shapes share one {label} envelope: "
                  f"{sorted(envelope_keys(g_doc))} vs {sorted(envelope_keys(j_doc))}")
        env.check(g_doc["usage"]["output_tokens"] == j_doc["usage"]["output_tokens"] == 0,
                  "both shapes report a zero output count")

    # 6. One entry in a `contexts` request still answers with the `contexts` array. The key names
    #    the request shape, not how many answers came back, and both shapes agree on that: a
    #    single-state request answers `answers`, a `contexts` request answers `contexts`.
    for shape, request in (("schema", body_for(SYM_SCHEMA, contexts=[STATE])),
                           ("questions", {"model": "test", "contexts": [STATE],
                                         "questions": SYM_QUESTIONS})):
        status, text = server.post("/v1/decision", json.dumps(request))
        env.check(status == 200, f"the one-context {shape} request answered: {status} {text[:160]}")
        if status != 200:
            continue
        doc = json.loads(text)
        env.check(set(doc) == {"model", "contexts"},
                  f"a one-entry {shape} contexts request answers with the array: {sorted(doc)}")
        env.check(len(doc["contexts"]) == 1, f"the {shape} array carries its one entry")


def sidecar_args():
    # the sidecar executor is the default decision placement when a pool exists, so the session
    # checks below run against the configuration the branch actually deploys
    return ["--instance", "main:ctx=8192:parallel=2:default",
            "--instance", "other:ctx=512:parallel=1",
            "--slots", "--jinja", "--slot-save-path", tempfile.mkdtemp()]


def run_prefix_checks(model):
    """The generic shape stays on the superset route whatever the deployment mounts it under.

    This is the anti-regression half of the mount-point checks: routing a request by anything
    other than the route that was invoked can either refuse the generic shape by mistake or
    admit it on the strict route. Only the superset may serve it.
    """
    schema = {"dept": {"type": "enum", "description": "owning team",
                       "enum": ["billing", "technical"]}}
    for prefix in ("", "/api", "/api/v1"):
        server = env.Server(model, sidecar_args(), api_prefix=prefix)
        try:
            server.start()
        except Exception as e:  # noqa: BLE001
            server.stop()
            print(f"skip prefix {prefix!r}: {e}")
            return
        try:
            status, text = server.post(prefix + "/v1/decision", json.dumps(body_for(schema)))
            env.check(status == 200,
                      f"prefix {prefix!r}: the superset route serves a schema body: {status} {text[:160]}")
            if status != 200:
                continue
            doc = json.loads(text)
            env.check(set(doc) == {"model", "answers", "usage"},
                      f"prefix {prefix!r}: the schema answer is the Jev envelope: {sorted(doc)}")
            env.check(set(doc["answers"]) == {"dept"},
                      f"prefix {prefix!r}: every field is answered ({sorted(doc['answers'])})")
            check_record(doc["answers"]["dept"], ["billing", "technical"], False,
                         f"prefix {prefix!r} dept")

            status, text = server.post(prefix + "/v1/systemone", json.dumps(body_for(schema)))
            env.check(status == 400,
                      f"prefix {prefix!r}: the strict route still refuses a schema body: {status} {text[:160]}")
        finally:
            server.stop()


def main():
    if not os.path.isfile(env.SERVER_BIN):
        print(f"SKIP: server binary not found at {env.SERVER_BIN}")
        return 0
    candidates = [m for m in env.MODEL_CANDIDATES if m and os.path.isfile(m)]
    if not candidates:
        print("SKIP: no test model; set LLAMA_SERVER_TEST_MODEL")
        return 0
    previous_cap = os.environ.get("LLAMA_DECISION_POOL_CAP")
    os.environ["LLAMA_DECISION_POOL_CAP"] = str(POOL_CAP)
    try:
        return run_models(candidates)
    finally:
        if previous_cap is None:
            os.environ.pop("LLAMA_DECISION_POOL_CAP", None)
        else:
            os.environ["LLAMA_DECISION_POOL_CAP"] = previous_cap


def run_models(candidates):
    for model in candidates:
        server = env.Server(model, sidecar_args())
        try:
            server.start()
        except Exception as e:  # noqa: BLE001
            server.stop()
            print(f"skip {os.path.basename(model)}: {e}")
            continue
        if not env.supports_letter_labels(server):
            server.stop()
            print(f"skip {os.path.basename(model)}: no usable answer labels")
            continue
        try:
            run_checks(server)
            run_envelope_checks(server)
        except Exception as e:  # noqa: BLE001
            server.stop()
            print(f"FAIL: generic schema checks: {e}")
            return 1
        # one model resident at a time: the mount-point checks each start their own server
        server.stop()
        try:
            run_prefix_checks(model)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL: generic schema mount-point checks: {e}")
            return 1
        print("decision generic schema checks passed")
        return 0
    print("SKIP: no candidate model supports letter labels; set LLAMA_SERVER_TEST_MODEL")
    return 0


def test_decision_generic():
    """pytest entry point for the checks above.

    The script's return code is the verdict. A missing binary or model is a skip; a run that
    completes the suite is a pass.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = main()
    out = buf.getvalue()
    print(out, end="")
    if rc != 0:
        pytest.fail(out.strip() or "the generic schema checks failed")
    if "SKIP" in out and "passed" not in out:
        pytest.skip(out.strip().splitlines()[-1])


if __name__ == "__main__":
    sys.exit(main())
