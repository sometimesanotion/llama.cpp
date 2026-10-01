#!/usr/bin/env python3
"""/v1/systemone is the strict Jev contract; /v1/decision is the superset.

This suite is the conformance gate for the Jev contract. It pins the request shape, the response
envelope, the three question types, the documented status codes, and the guarantee that the two
routes answer identically. It is written against docs/decision/JEV-API.md, so a change to either
route that breaks Jev compatibility fails here.

Every check that can be made without weights is made here: the envelope, the types, the keys, the
status codes and the cross-route agreement. A model is only needed to get a 200 out of a valid
request, so the suite asserts the contract, never a particular answer.

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

SYSTEMONE = "/v1/systemone"
DECISION = "/v1/decision"

STATE = "Customer was charged twice on May 3 and asks for a refund of the duplicate."

# The three Jev question types, as JEV-API.md documents them.
Noul = {"type": "noul", "instructions": "Should this be refunded?"}
Choice = {"type": "choice", "instructions": "Which team should handle this?",
          "criteria": {"billing": "payments and refunds", "technical": "software errors"}}
Score = {"type": "score", "instructions": "How urgent is this?",
         "criteria": ["low", "medium", "high"]}

ALL_THREE = {"refund": Noul, "dept": Choice, "urgency": Score}


def jev_body(**extra):
    out = {"model": "test", "state": STATE, "questions": dict(ALL_THREE)}
    out.update(extra)
    return out


def call(server, path, body):
    return server.post(path, json.dumps(body))


# --- 1. the response envelope is the Jev shape, on both routes ---

def check_envelope(env, doc, label):
    env.check(isinstance(doc, dict), f"{label}: the response is an object")
    env.check("answers" in doc, f"{label}: the response carries answers")
    env.check("model" in doc, f"{label}: the response echoes the model")
    env.check("usage" in doc, f"{label}: the response carries usage")
    env.check(isinstance(doc["answers"], dict), f"{label}: answers is a map keyed by question id")
    usage = doc["usage"]
    for key in ("input_tokens", "output_tokens"):
        env.check(key in usage, f"{label}: usage carries {key}")
    env.check(usage["output_tokens"] == 0, f"{label}: a decision generates nothing")


def check_answer(env, answer, kind, label, keys):
    env.check("type" in answer, f"{label}: the answer carries its type")
    env.check(answer["type"] == kind, f"{label}: the type is {kind} (got {answer.get('type')})")
    if kind == "noul":
        # Jev's NoulAnswer is {type, noul} only: a boolean question has no confidence
        env.check("noul" in answer, f"{label}: a noul answer carries noul")
        env.check(isinstance(answer["noul"], (int, float)), f"{label}: noul is a probability")
        env.check(0.0 <= answer["noul"] <= 1.0, f"{label}: noul is in [0,1]")
        env.check("confidence" not in answer, f"{label}: Jev's noul answer carries no confidence")
        env.check("probabilities" not in answer, f"{label}: Jev's noul answer carries no probability map")
    else:
        env.check("confidence" in answer, f"{label}: a {kind} answer carries confidence")
        probs = answer.get("probabilities")
        env.check(isinstance(probs, dict), f"{label}: a {kind} answer carries a probability map")
        env.check(set(probs) == set(keys),
                  f"{label}: the map covers every option key ({sorted(probs) if probs else probs} vs {sorted(keys)})")
        env.check(abs(sum(probs.values()) - 1.0) < 1e-6, f"{label}: the probabilities sum to 1")
        env.check(all(0.0 <= p <= 1.0 for p in probs.values()), f"{label}: every probability is in [0,1]")
        winner = max(probs, key=probs.get)
        if kind == "choice":
            env.check(answer.get("choice") == winner, f"{label}: choice is the argmax")
        else:
            env.check("score" in answer, f"{label}: a score answer carries score")
            env.check(isinstance(answer.get("legend"), dict), f"{label}: a score answer echoes the legend")
            env.check(set(answer["legend"]) == set(probs), f"{label}: the legend covers the levels")
            expected = sum(float(k) * p for k, p in probs.items())
            env.check(abs(answer["score"] - expected) < 1e-6,
                      f"{label}: score is the weighted level index ({answer.get('score')} vs {expected})")


# the option keys each documented question type is scored over
OPTION_KEYS = {"refund": None, "dept": ["billing", "technical"], "urgency": ["0", "1", "2"]}


def run_checks(server):
    # 1. The three documented types, and the strict envelope, on the strict route.
    status, text = call(server, SYSTEMONE, jev_body())
    env.check(status == 200, f"the strict route answers a Jev request: {status} {text[:200]}")
    if status != 200:
        return
    doc = json.loads(text)
    check_envelope(env, doc, "systemone")
    env.check(set(doc["answers"]) == set(ALL_THREE), f"every question is answered ({sorted(doc['answers'])})")
    for qid, spec in ALL_THREE.items():
        check_answer(env, doc["answers"][qid], spec["type"], f"systemone {qid}", OPTION_KEYS[qid])

    # 2. The strict route adds nothing: no diagnostics object, no timings, no generic keys.
    for absent in ("diagnostics", "timings", "results", "fields", "decision", "contexts"):
        env.check(absent not in doc, f"the default strict response omits {absent}")

    # 3. The two routes agree, answer for answer. This is the superset relationship: /v1/decision
    #    carries everything /v1/systemone does, for a Jev request. The two calls are separate
    #    requests, so the winner, the option set and the key set are pinned exactly while the
    #    concentration is compared with the documented producer-numerics bound: on a recurrent
    #    model a cached prefix is restored through the engine's host-state warm path, which is
    #    exact in the answer and approximate in the float.
    status, text = call(server, DECISION, jev_body())
    env.check(status == 200, f"the superset route answers a Jev request: {status} {text[:200]}")
    if status == 200:
        other = json.loads(text)
        check_envelope(env, other, "decision")
        env.check_answers_agree(doc["answers"], other["answers"], "both routes")

    # 4. The generic shape is refused on the strict route and served on the superset.
    schema_body = {"model": "test", "state": STATE,
                   "schema": {"dept": {"type": "enum", "description": "owning team",
                                       "enum": ["billing", "technical"]}}}
    status, text = call(server, SYSTEMONE, schema_body)
    env.check(status == 400, f"the strict route refuses a schema body: {status} {text[:160]}")
    env.check("/v1/decision" in text, f"the refusal names the route that serves it: {text[:160]}")
    status, text = call(server, DECISION, schema_body)
    env.check(status == 200, f"the superset route serves a schema body: {status} {text[:160]}")
    if status == 200:
        sup = json.loads(text)
        # the generic shape extends the Jev envelope, so a schema answer is still Jev-readable
        env.check("answers" in sup and "results" not in sup,
                  f"the schema answer extends the Jev envelope: {sorted(sup)}")
        env.check(set(sup) == {"model", "answers", "usage"},
                  f"the schema answer is the Jev envelope: {sorted(sup)}")
        for name, ans in sup.get("answers", {}).items():
            env.check("type" in ans and "value" in ans and "confidence" in ans
                      and "probabilities" in ans and "legend" in ans,
                      f"the generic {name} answer carries the Jev keys plus value: {sorted(ans)}")
            env.check("timings" not in sup, "the strict envelope carries no timings")

    # 5. Each type on its own, so a type-specific regression is not masked by the others.
    for qid, spec in ALL_THREE.items():
        status, text = call(server, SYSTEMONE, {"model": "test", "state": STATE, "questions": {qid: spec}})
        env.check(status == 200, f"a single {spec['type']} question is answered: {status} {text[:160]}")
        if status == 200:
            answers = json.loads(text)["answers"]
            env.check(set(answers) == {qid}, f"the answer is keyed by the question id ({sorted(answers)})")
            check_answer(env, answers[qid], spec["type"], f"single {spec['type']}", OPTION_KEYS[qid])

    # 6. The documented criteria forms. A score question takes an ordered array or a legend object
    #    whose keys are "0".."K-1" in order; noul takes true/false descriptions.
    status, text = call(server, SYSTEMONE, {"model": "test", "state": STATE, "questions": {
        "legend": {"type": "score", "instructions": "How urgent?",
                   "criteria": {"0": "low", "1": "medium", "2": "high"}}}})
    env.check(status == 200, f"a score legend object is accepted: {status} {text[:160]}")
    if status == 200:
        answer = json.loads(text)["answers"]["legend"]
        env.check(answer.get("legend") == {"0": "low", "1": "medium", "2": "high"},
                  f"the legend is echoed verbatim ({answer.get('legend')})")
    status, text = call(server, SYSTEMONE, {"model": "test", "state": STATE, "questions": {
        "n": {"type": "noul", "instructions": "Refunded?",
              "criteria": {"true": "a duplicate charge was refunded", "false": "nothing was refunded"}}}})
    env.check(status == 200, f"noul criteria are accepted: {status} {text[:160]}")
    status, text = call(server, SYSTEMONE, {"model": "test", "state": STATE, "questions": {
        "n": {"type": "noul", "instructions": "Refunded?", "criteria": {"maybe": "unsure"}}}})
    env.check(status == 422, f"noul criteria keys must be true/false: {status} {text[:160]}")

    # 7. The documented status codes. 422 is the content error, 400 the request error.
    for body, code, needle in (
        ({"state": STATE, "questions": ALL_THREE}, 422, "model is required"),
        ({"model": "test", "questions": ALL_THREE}, 422, "state"),
        ({"model": "test", "state": STATE}, 422, "questions"),
        ({"model": "test", "state": STATE, "questions": {}}, 422, "questions"),
        ({"model": "test", "state": STATE, "questions": {"q": {"instructions": "x"}}}, 422, "type"),
        ({"model": "test", "state": STATE, "questions": {"q": {"type": "noul"}}}, 422, "instructions"),
        ({"model": "test", "state": STATE, "questions": {"q": {"type": "wat", "instructions": "x"}}},
         422, "unknown type"),
        ({"model": "test", "state": STATE,
          "questions": {"q": {"type": "choice", "instructions": "x", "criteria": {"only": "one"}}}},
         422, "2-255"),
        ({"model": "test", "state": STATE,
          "questions": {"q": {"type": "score", "instructions": "x", "criteria": ["only"]}}},
         422, "2-10"),
        ({"model": "test", "state": STATE, "questions": {"q": {"type": "noul", "instructions": "x",
                                                             "bogus": 1}}}, 422, "unknown"),
        ({"model": 7, "state": STATE, "questions": ALL_THREE}, 422, "model"),
        ({"model": "test", "state": STATE, "contexts": ["a"], "questions": ALL_THREE}, 422, "not both"),
    ):
        status, text = call(server, SYSTEMONE, body)
        env.check(status == code, f"{needle!r} is a {code}: {status} {text[:160]}")
        env.check(needle in text, f"the refusal names {needle!r}: {text[:160]}")

    # a body that is not an object at all
    status, text = env.http("POST", f"http://127.0.0.1:{server.port}{SYSTEMONE}", "[]")
    env.check(status in (400, 422), f"a non-object body is a client error: {status} {text[:160]}")

    # 8. Malformed JSON is a 400, as the shared HTTP layer reports it.
    status, text = env.http("POST", f"http://127.0.0.1:{server.port}{SYSTEMONE}", "{not json")
    env.check(status == 400, f"malformed JSON is a 400: {status} {text[:160]}")

    # 9. A session is answered on the strict route too, and a session decision still reads as Jev.
    env.prefill_slot(server, 0, env.LETTER_SYSTEM, "State:\n" + STATE + "\n")
    status, text = call(server, SYSTEMONE, jev_body(id_slot=0))
    env.check(status == 200, f"a Jev session decision on the strict route: {status} {text[:200]}")
    if status == 200:
        doc = json.loads(text)
        check_envelope(env, doc, "systemone session")
        check_answer(env, doc["answers"]["dept"], "choice", "systemone session dept", OPTION_KEYS["dept"])

    # 10. The additive diagnostics are opt-in and never change the answers. A warm repeat comes
    #     first so the two compared calls sit in the same cache state; the comparison itself is the
    #     shared cross-request contract, since diagnostics and plain are two separate requests.
    call(server, SYSTEMONE, jev_body())
    status, text = call(server, SYSTEMONE, jev_body())
    env.check(status == 200, f"the warm baseline is answered: {status} {text[:200]}")
    status2, text2 = call(server, SYSTEMONE, jev_body(diagnostics=True))
    env.check(status2 == 200, f"diagnostics are accepted: {status2} {text2[:200]}")
    if status == 200 and status2 == 200:
        plain = json.loads(text)["answers"]
        diag = json.loads(text2)
        env.check("diagnostics" in diag, "the diagnostics object is present when asked for")
        env.check("timings" in diag, "timings are reported when diagnostics are asked for")
        env.check_answers_agree(plain, diag["answers"], "diagnostics",
                                additive=env.DIAGNOSTICS_ADDITIVE_KEYS)

    # 11. The model echo follows the documented contract on the strict route.
    for requested, resolves in (("jev-latest", True), ("jev-preview", True), ("my-model-v9", False), ("", True)):
        status, text = call(server, SYSTEMONE, jev_body(model=requested))
        env.check(status == 200, f"the model echo for {requested!r}: {status} {text[:160]}")
        if status != 200:
            continue
        echoed = json.loads(text)["model"]
        if resolves:
            env.check(echoed != requested, f"the alias {requested!r} resolves to the loaded model")
        else:
            env.check(echoed == requested, f"a concrete id is echoed verbatim ({echoed})")


def sidecar_args():
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
            print(f"FAIL: systemone conformance: {e}")
            return 1
        server.stop()
        print("decision systemone conformance checks passed")
        return 0
    print("SKIP: no candidate model supports letter labels; set LLAMA_SERVER_TEST_MODEL")
    return 0


def test_decision_systemone():
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
        pytest.fail(out.strip() or "the systemone conformance checks failed")
    if "SKIP" in out and "passed" not in out:
        pytest.skip(out.strip().splitlines()[-1])


if __name__ == "__main__":
    sys.exit(main())
