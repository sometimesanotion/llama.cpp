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


def check_record(record, values, numeric, label):
    """The shape and the invariants every scored field must satisfy, whatever its type."""
    env.check(isinstance(record, dict), f"{label}: the field record is an object")
    for key in ("value", "probability", "scored_nodes", "tree"):
        env.check(key in record, f"{label}: the field record carries {key}")
    env.check(any(v == record["value"] for v in values),
              f"{label}: the value is one of the allowed values ({record['value']})")
    env.check(0.0 <= record["probability"] <= 1.0, f"{label}: the probability is in [0,1]")
    if not numeric:
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
    # 1. The response envelope: a results list, one entry per context, with the token accounting
    #    and timings the Jev shape also reports.
    status, text = post_schema(server, {
        "active":   {"type": "boolean", "description": "is the incident active"},
        "severity": {"type": "enum", "description": "incident severity", "enum": ["low", "medium", "high"]},
    })
    env.check(status == 200, f"typed schema status {status}: {text[:200]}")
    if status != 200:
        return
    doc = json.loads(text)
    env.check("model" in doc and "results" in doc, f"the envelope carries model and results: {list(doc)}")
    env.check(isinstance(doc["results"], list) and len(doc["results"]) == 1, "one result per context")
    for key in ("input_tokens", "output_tokens", "cached_tokens", "state_cache_hit"):
        env.check(key in doc["usage"], f"usage carries {key}")
    env.check(doc["usage"]["output_tokens"] == 0, "a decision still generates nothing")
    for key in ("prefill_ms", "scoring_ms", "total_ms", "rounds", "rows", "per_decision_ms"):
        env.check(key in doc["timings"], f"timings carry {key}")
    result = doc["results"][0]
    env.check("decision" in result and "fields" in result, "a result carries decision and fields")
    for key in ("context_tokens", "scored_rows"):
        env.check(key in result["usage"], f"the per-result usage carries {key}")

    # 2. A boolean and an enum: the value is echoed into the decision and typed in the record.
    env.check(result["decision"]["active"] in (True, False), "the decision holds the boolean value")
    env.check(result["fields"]["active"]["value"] == result["decision"]["active"],
              "the decision echoes the field record")
    check_record(result["fields"]["active"], [True, False], False, "boolean")
    check_record(result["fields"]["severity"], ["low", "medium", "high"], False, "enum")

    # 3. Numeric grids: the inclusive integer range and the stepped number grid.
    status, text = post_schema(server, {
        "count":  {"type": "integer", "description": "affected rows", "minimum": 1, "maximum": 4},
        "impact": {"type": "number", "description": "impact factor", "minimum": 0.0, "maximum": 1.0,
                   "step": 0.5},
    })
    env.check(status == 200, f"numeric schema status {status}: {text[:200]}")
    if status == 200:
        fields = json.loads(text)["results"][0]["fields"]
        rec = check_record(fields["count"], list(range(1, 5)), True, "integer")
        env.check(isinstance(rec["value"], int), f"an integer field yields an integer value ({rec['value']})")
        rec = check_record(fields["impact"], [0.0, 0.5, 1.0], True, "number")
        env.check(isinstance(rec["value"], (int, float)) and not isinstance(rec["value"], bool),
                  f"a number field yields a numeric value ({rec['value']})")

    # 4. The aggregates. `mode` is the default and leaves the aggregate on the winning value;
    #    `median` and `mean` are value-space reductions, so they stay inside the value range and
    #    keep reporting the band.
    for aggregate, in (("mode",), ("median",), ("mean",)):
        status, text = post_schema(server, {
            "sev": {"type": "integer", "description": "severity", "minimum": 0, "maximum": 4,
                    "aggregate": aggregate},
        })
        env.check(status == 200, f"the {aggregate} aggregate status {status}: {text[:200]}")
        if status != 200:
            continue
        rec = check_record(json.loads(text)["results"][0]["fields"]["sev"], list(range(5)), True, aggregate)
        env.check(0 <= rec["aggregate"] <= 4, f"the {aggregate} aggregate is inside the value range")
        if aggregate == "mode":
            env.check(rec["aggregate"] == rec["value"], "the mode aggregate is the winning value")
    status, text = post_schema(server, {
        "sev": {"type": "integer", "description": "severity", "minimum": 0, "maximum": 4},
    })
    env.check(status == 200, f"the default aggregate status {status}: {text[:200]}")
    if status == 200:
        rec = json.loads(text)["results"][0]["fields"]["sev"]
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
        rec = json.loads(text)["results"][0]["fields"]["wide"]
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
        fields = json.loads(text)["results"][0]["fields"]
        env.check(set(fields) == {"active", "level", "grade"},
                  f"the properties object is the catalogue: {sorted(fields)}")
        check_record(fields["level"], [0, 1, 2], True, "json-schema integer")

    # 7. Many fields at once: the catalogue is scored in one batched pass and every field is scored.
    many = {f"f{i}": {"type": "boolean", "description": f"flag {i}"} for i in range(16)}
    status, text = post_schema(server, many)
    env.check(status == 200, f"a 16-field schema status {status}: {text[:200]}")
    if status == 200:
        env.check(len(json.loads(text)["results"][0]["fields"]) == 16, "all 16 fields are scored")

    # 8. Several contexts in one request: one result per context, in request order.
    contexts = ["The service was unreachable for ten minutes.",
                "A user could not log in for an hour.",
                "Nothing unusual happened today."]
    status, text = post_schema(server, {
        "severity": {"type": "enum", "description": "incident severity", "enum": ["low", "medium", "high"]},
    }, contexts=contexts)
    env.check(status == 200, f"the multi-context status {status}: {text[:200]}")
    if status == 200:
        results = json.loads(text)["results"]
        env.check(len(results) == len(contexts),
                  f"one result per context ({len(results)} vs {len(contexts)})")
        env.check(all(set(r["fields"]) == {"severity"} for r in results),
                  "every context scores the whole catalogue")

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

    for payload, needle in (
        ({"state": STATE, "schema": {"a": {"type": "boolean", "description": "d"}}}, "model is required"),
        ({"model": "t", "schema": "nope"}, "must be an object"),
        ({"model": "t", "schema": {"a": {"type": "boolean", "description": "d"}}}, "state"),
    ):
        status, text = server.post("/v1/decision", json.dumps(payload))
        env.check(status == 422, f"a missing request part is a 422: {status} {text[:160]}")
        env.check(needle in text, f"the refusal names {needle!r}: {text[:160]}")

    # 11. A session: the schema is answered against a prefilled slot without re-prefilling it.
    env.prefill_slot(server, 0, env.LETTER_SYSTEM, "State:\n" + STATE + "\n")
    schema = {"severity": {"type": "enum", "description": "incident severity",
                           "enum": ["low", "medium", "high"]}}
    status, text = post_schema(server, schema, id_slot=0)
    env.check(status == 200, f"a schema session decision status {status}: {text[:200]}")
    if status == 200:
        env.check(len(json.loads(text)["results"]) == 1, "a session scores exactly one context")
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


def sidecar_args():
    # the sidecar executor is the default decision placement when a pool exists, so the session
    # checks below run against the configuration the branch actually deploys
    return ["--instance", "main:ctx=8192:parallel=2:default",
            "--instance", "other:ctx=512:parallel=1",
            "--slots", "--jinja", "--slot-save-path", tempfile.mkdtemp()]


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
        except Exception as e:  # noqa: BLE001
            server.stop()
            print(f"FAIL: generic schema checks: {e}")
            return 1
        server.stop()
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
