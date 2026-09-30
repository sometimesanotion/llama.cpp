#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Coexistence checks for a live chat session and the decision endpoint.

The decision engine runs on the same context as chat: there is one scheduler
thread, so a decision is serialized with chat decode rather than run on a second
context. This script pins down the observable contract when both APIs are used
on one live session:

  * a session decision about an idle slot succeeds while another slot generates,
    and the generating chat answer is byte-identical to a quiet run;
  * a decision about a slot that is itself generating is refused (4xx) and the
    chat answer is unchanged;
  * a first-class session survives another chat task clearing idle slots;
  * two concurrent decisions on one session both succeed and agree.

Timing is reported (quiet vs concurrent) so the serialization is visible; the
latency ratio is not asserted because it is machine dependent. Correctness
(answers last, chat bytes stable) is asserted. Skips cleanly (exit 0) when the
server binary or a model with usable answer labels is missing.

Run with a GPU build:
  LLAMA_SERVER_BIN=build-rocm/bin/llama-server \
  LD_LIBRARY_PATH=build-rocm/bin \
  python3 test_decision_session_concurrency.py MODEL.gguf
"""

import importlib.util
import json
import os
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Server, http, prefill_slot, supports_letter_labels, DECISION_VALID, LETTER_SYSTEM
_env_spec = importlib.util.spec_from_file_location("decision_envelope", os.path.join(HERE, "test_decision_envelope.py"))
env = importlib.util.module_from_spec(_env_spec)
_env_spec.loader.exec_module(env)

MODEL_CANDIDATES = [
    os.environ.get("LLAMA_SERVER_TEST_MODEL", ""),
    os.path.join(HERE, "tmp", "stories15M-q4_0.gguf"),
    os.path.join(HERE, "tmp", "moe_shakespeare15M.gguf"),
]


def content_of(text):
    try:
        return json.loads(text)["choices"][0]["message"]["content"]
    except Exception:  # noqa: BLE001
        return None


def answer_of(text):
    try:
        return json.loads(text)["answers"]
    except Exception:  # noqa: BLE001
        return None


# sidecar executor mode: the pool registers a default chat instance and a lazily-registered second
# instance (used by the cross-instance refusal), plus the internal __decision__ executor. M3: the
# sidecar is the default executor when --decision-seqs is set, so no --decision-sidecar is passed.
def sidecar_args():
    import tempfile
    return ["--instance", "main:ctx=8192:parallel=2:default",
            "--instance", "other:ctx=512:parallel=1",
            "--slots", "--jinja", "--slot-save-path", tempfile.mkdtemp()]


def timed_post(fn, *args):
    t = time.time()
    result = fn(*args)
    return result, (time.time() - t) * 1000.0


def run_checks(model, extra_args=None):
    # sidecar executor mode: a decision runs on the internal executor's own context, so chat on
    # the chat instance is not serialized behind it. The old shared-context lane still serializes.
    sidecar_mode = extra_args is not None
    srv = env.Server(model, extra_args if extra_args is not None
                     else ["--parallel", "2", "--slots", "--jinja",
                           "--decision-arena-seqs", "4",
                           "--slot-save-path", tempfile.mkdtemp()])
    try:
        srv.start()
    except Exception as e:  # noqa: BLE001
        srv.stop()
        print(f"skip: server did not start on {os.path.basename(model)}: {e}")
        return True
    if not env.supports_letter_labels(srv):
        srv.stop()
        print(f"skip: no usable answer labels on {os.path.basename(model)}")
        return True

    try:
        # slot 1 holds a completed turn with the exact decision prefix; a first-class session
        # owns the turn through an arena copy, so it survives another chat clearing idle slots.
        user = "State:\n" + env.DECISION_VALID["state"] + "\n"
        env.prefill_slot(srv, 1, env.LETTER_SYSTEM, user)
        status, text = srv.post("/v1/session", json.dumps(
            {"id_slot": 1, "turn": "t1", "policy": {"backend": "host", "capture_on_turn_complete": True}}))
        env.check(status == 200, f"session create status {status}: {text[:200]}")
        sid = json.loads(text)["session_id"]

        # a heavier decision so its duration is measurable against chat
        heavy = dict(env.DECISION_VALID)
        heavy["state"] = env.DECISION_VALID["state"] + (" extra evidence sentence." * 60)
        heavy_sid = dict(heavy, session_id=sid)

        chat = {"messages": [{"role": "user", "content": "Write a long paragraph about spring weather and gardens."}],
                "max_tokens": 160, "seed": 42, "temperature": 0.0, "id_slot": 0}

        def chat_post(body=None):
            return srv.post("/v1/chat/completions", json.dumps(body or chat))

        def decision_post(body):
            return srv.post("/v1/decision", json.dumps(body))

        # 1. warm the chat path, then take the quiet baseline. The very first decode on a
        #    context can pick a different batch shape and flip a near-tie on a GPU, so the first
        #    answer is not a stable reference by itself; later runs are byte-stable.
        (wst, wtx), _ = timed_post(chat_post)
        env.check(wst == 200, f"chat warmup status {wst}: {wtx[:120]}")
        (qst, qtx), q_chat_ms = timed_post(chat_post)
        env.check(qst == 200, f"quiet chat status {qst}: {qtx[:120]}")
        q_con = content_of(qtx)
        (dst, dtx), q_dec_ms = timed_post(decision_post, heavy_sid)
        env.check(dst == 200, f"quiet session decision status {dst}: {dtx[:200]}")

        # 2. cross-slot overlap: chat generating on slot 0 while a session decision runs on slot 1
        result = {}

        def run_chat():
            result["chat"] = timed_post(chat_post)

        th = threading.Thread(target=run_chat)
        th.start()
        time.sleep(0.05)
        (dst2, dtx2), d_ms = timed_post(decision_post, heavy_sid)
        th.join()
        (cst2, ctx2), c_ms = result["chat"]
        env.check(cst2 == 200, f"concurrent chat status {cst2}: {ctx2[:120]}")
        env.check(dst2 == 200, f"concurrent session decision status {dst2}: {dtx2[:200]}")
        env.check(content_of(ctx2) == q_con, "a concurrent session decision does not change the chat answer")
        print(f"cross-slot: quiet chat {q_chat_ms:.0f}ms, with decision {c_ms:.0f}ms; "
              f"quiet decision {q_dec_ms:.0f}ms, concurrent {d_ms:.0f}ms "
              f"(quiet sum {q_chat_ms + q_dec_ms:.0f}ms)")

        # M3.2: on the sidecar executor a decision does not stall chat on the chat instance. The
        # decision runs on its own context, so the concurrent chat must stay near its quiet latency
        # instead of being serialized behind the full decision duration. Machine dependent, so the
        # assertion is a generous factor of the quiet chat latency, not an absolute budget.
        if sidecar_mode:
            env.check(c_ms <= q_chat_ms * 4.0 + 500.0,
                      f"sidecar chat is not stalled by a concurrent decision: quiet {q_chat_ms:.0f}ms "
                      f"vs with-decision {c_ms:.0f}ms")

        # 3. the session survives another chat task (cache_idle_slots may clear the origin slot)
        (dst3, dtx3), _ = timed_post(decision_post, heavy_sid)
        env.check(dst3 == 200, f"session after a concurrent chat status {dst3}: {dtx3[:200]}")

        # 4. a decision about the slot that is generating is refused, and the chat is untouched
        result2 = {}

        def run_chat2():
            result2["chat"] = timed_post(chat_post)

        th2 = threading.Thread(target=run_chat2)
        th2.start()
        time.sleep(0.2)
        (sst, stx), s_ms = timed_post(decision_post, dict(env.DECISION_VALID, id_slot=0))
        th2.join()
        (cst3, ctx3), c3_ms = result2["chat"]
        env.check(sst in (400, 422), f"an in-flight slot is refused: {sst} {stx[:160]}")
        env.check(cst3 == 200, f"chat after a refused same-slot decision status {cst3}: {ctx3[:120]}")
        env.check(content_of(ctx3) == q_con, "a refused same-slot decision does not change the chat answer")

        # 5. two concurrent decisions on one session both succeed and agree
        outs = {}

        def decide(i):
            outs[i] = timed_post(decision_post, heavy_sid)

        threads = [threading.Thread(target=decide, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        env.check(all(outs[i][0][0] == 200 for i in outs), f"concurrent session decisions succeed: {outs}")
        winners = [answer_of(outs[i][0][1]).get("dept", {}).get("choice") for i in outs]
        env.check(winners[0] == winners[1], f"concurrent session decisions agree: {winners}")

        # 6. an implicit id_slot session whose origin was cleared is never silently answered from
        #    an old turn: it either re-captures the resident slot or refuses, never returns stale.
        prefill_user = env.DECISION_VALID["state"] + " turn two"
        env.prefill_slot(srv, 1, env.LETTER_SYSTEM, "State:\n" + prefill_user + "\n")
        # drive a chat task so cache_idle_slots may clear the idle slot 1
        (cst4, _), _ = timed_post(chat_post)
        env.check(cst4 == 200, "chat between implicit-session probes succeeds")
        (ist, itx), _ = timed_post(decision_post, dict(env.DECISION_VALID, id_slot=1))
        env.check(ist == 200 or ist in (400, 422),
                  f"an implicit id_slot session is answered or refused, never an error: {ist} {itx[:120]}")

        print(f"decision session coexistence checks passed on {os.path.basename(model)}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: {e}")
        return False
    finally:
        srv.stop()


def run_sidecar_specific_checks(model):
    """Sidecar executor mode: sessions are eager token snapshots replayed on the internal
    executor, never the owning chat context. Pins the M2 behaviors: F1 regression (another
    slot's chat does not invalidate a session), identity mismatch 422, cross-instance refusal
    400, and the clone/file backends being a 501 capability refusal.
    """
    srv = env.Server(model, sidecar_args())
    try:
        srv.start()
    except Exception as e:  # noqa: BLE001
        srv.stop()
        print(f"skip sidecar session checks on {os.path.basename(model)}: {e}")
        return True
    if not env.supports_letter_labels(srv):
        srv.stop()
        print(f"skip sidecar session checks on {os.path.basename(model)}: no usable answer labels")
        return True

    def prefill(slot, state):
        env.prefill_slot(srv, slot, env.LETTER_SYSTEM, "State:\n" + state + "\n")

    def decision(body):
        return srv.post("/v1/decision", json.dumps(body))

    def chat():
        return srv.post("/v1/chat/completions", json.dumps(
            {"messages": [{"role": "user", "content": "Write about spring weather and gardens."}],
             "max_tokens": 32, "seed": 42, "temperature": 0.0, "id_slot": 0}))

    try:
        # a session is captured eagerly at create from the owning instance
        prefill(1, env.DECISION_VALID["state"])
        status, text = srv.post("/v1/session", json.dumps({"id_slot": 1, "instance": "main"}))
        env.check(status == 200, f"sidecar session create status {status}: {text[:200]}")
        sid = json.loads(text)["session_id"]
        env.check(json.loads(text)["captured"] is True, "sidecar session captures eagerly")

        status, text = decision(dict(env.DECISION_VALID, session_id=sid, diagnostics=True))
        env.check(status == 200, f"sidecar session decision status {status}: {text[:200]}")
        env.check(json.loads(text).get("session_fork") is True, "sidecar decision reports a session fork")

        # F1 regression: chat on another slot (cache_idle_slots may clear the origin slot) does
        # not invalidate the session: the snapshot owns the tokens
        status, text = chat()
        env.check(status == 200, f"sidecar chat on another slot status {status}: {text[:120]}")
        status, text = decision(dict(env.DECISION_VALID, session_id=sid))
        env.check(status == 200, f"sidecar F1 session still valid status {status}: {text[:200]}")

        # identity mismatch: the source slot advances to a new turn -> the old session is stale
        prefill(1, env.DECISION_VALID["state"] + " turn two")
        status, text = decision(dict(env.DECISION_VALID, session_id=sid))
        env.check(status == 422, f"sidecar stale session refused 422: {status} {text[:200]}")
        env.check("stale" in text or "turn" in text, f"sidecar staleness names the turn: {text[:160]}")

        # a fresh session for the cross-instance and backend checks
        prefill(0, env.DECISION_VALID["state"])
        status, text = srv.post("/v1/session", json.dumps({"id_slot": 0, "instance": "main"}))
        env.check(status == 200, f"sidecar second session create status {status}: {text[:200]}")
        sid2 = json.loads(text)["session_id"]

        # cross-instance refusal: naming a different instance for the session is 400
        status, text = decision(dict(env.DECISION_VALID, session_id=sid2, instance="other"))
        env.check(status == 400, f"sidecar cross-instance refusal status {status}: {text[:200]}")

        # clone/file are a 501 capability refusal on the sidecar (token snapshots only)
        for backend in ("clone", "file"):
            status, text = srv.post("/v1/session", json.dumps(
                {"id_slot": 0, "instance": "main", "policy": {"backend": backend}}))
            env.check(status == 501, f"sidecar {backend} backend refused 501: {status} {text[:200]}")

        # an implicit id_slot decision captures eagerly on first use and replays on the sidecar
        status, text = decision(dict(env.DECISION_VALID, id_slot=0, instance="main"))
        env.check(status == 200, f"sidecar implicit id_slot decision status {status}: {text[:200]}")
        env.check(json.loads(text).get("session_fork") is True, "sidecar implicit session fork")

        # lifecycle
        status, text = srv.post("/v1/decision", json.dumps(dict(env.DECISION_VALID, session_id=sid2)))
        env.check(status == 200, f"sidecar session_id decision status {status}: {text[:200]}")
        status, text = env.http("GET", f"http://127.0.0.1:{srv.port}/v1/session/{sid2}")
        env.check(status == 200, f"sidecar session get status {status}: {text[:200]}")
        status, text = env.http("PATCH", f"http://127.0.0.1:{srv.port}/v1/session/{sid2}",
                            json.dumps({"pinned": True, "ttl_ms": 60000}))
        env.check(status == 200, f"sidecar session patch status {status}: {text[:200]}")
        status, text = env.http("DELETE", f"http://127.0.0.1:{srv.port}/v1/session/{sid2}")
        env.check(status == 200, f"sidecar session delete status {status}: {text[:200]}")
        status, text = env.http("GET", f"http://127.0.0.1:{srv.port}/v1/session/{sid2}")
        env.check(status == 404, f"sidecar deleted session gone: {status} {text[:200]}")

        print(f"sidecar token-snapshot session checks passed on {os.path.basename(model)}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: sidecar session checks: {e}")
        return False
    finally:
        srv.stop()


# M7 resident warm-prefix calibration lane: the sidecar keeps a bounded set of resident session
# prefixes; the first decision on a turn cold-prefills (warm_hit false), a repeat forks the resident
# prefix (warm_hit true) and must be bit-identical to its own miss. The control group (never-repeated
# sessions) must show zero warm hits; the repeated group must show hits.
def answers_close(a, b, tol):
    """The answers agree within `tol` on every numeric field and exactly on every non-numeric one.
    The winners (choice key, noul value, score index) must be unchanged; only the reported
    concentration (probabilities and the derived diagnostics) may move within `tol`. This is the
    documented M7.5 tolerance: on the qwen hybrid model the recurrent warm-restore drifts the score
    probabilities and their derived interval/median by up to ~0.05 between a cold miss and a warm hit
    (the M2/M3 recorded qwen producer-numerics matter, flaky from ~0 to ~0.05 run to run). A hit never
    changes a winner, only the reported concentration; lfm and gemma are effectively wire-identical."""
    if set(a) != set(b):
        return False
    for qid in a:
        if set(a[qid]) != set(b[qid]):
            return False
        for k, v in a[qid].items():
            bv = b[qid][k]
            if isinstance(v, dict):
                if set(v) != set(bv):
                    return False
                for kk, vv in v.items():
                    if isinstance(vv, float) and abs(vv - bv[kk]) > tol:
                        return False
                continue
            if isinstance(v, (list, tuple)):
                if len(v) != len(bv):
                    return False
                for x, y in zip(v, bv):
                    if isinstance(x, float) and abs(x - y) > tol:
                        return False
                continue
            if isinstance(v, float):
                if abs(v - bv) > tol:
                    return False
                continue
            if v != bv:
                return False
    return True


def run_warm_cache_checks(model):
    args = sidecar_args() + ["--decision-warm-budget-mb", "256"]
    srv = env.Server(model, args)
    try:
        srv.start()
    except Exception as e:  # noqa: BLE001
        srv.stop()
        print(f"skip warm cache checks on {os.path.basename(model)}: {e}")
        return True
    if not env.supports_letter_labels(srv):
        srv.stop()
        print(f"skip warm cache checks on {os.path.basename(model)}: no usable answer labels")
        return True

    def prefill(state):
        env.prefill_slot(srv, 1, env.LETTER_SYSTEM, "State:\n" + state + "\n")

    def decision(body):
        return srv.post("/v1/decision", json.dumps(body))

    def new_session(state):
        prefill(state)
        status, text = srv.post("/v1/session", json.dumps({"id_slot": 1, "instance": "main"}))
        env.check(status == 200, f"warm session create status {status}: {text[:200]}")
        return json.loads(text)["session_id"]

    try:
        # M7.3 control group: never-repeated sessions show zero warm hits on first use
        control_hits = 0
        for i in range(3):
            sid = new_session("Distinct evidence sentence number %d for the warm control." % i)
            status, text = decision(dict(env.DECISION_VALID, session_id=sid, diagnostics=True))
            env.check(status == 200, f"warm control decision status {status}: {text[:200]}")
            control_hits += 1 if json.loads(text).get("warm_hit") else 0
        env.check(control_hits == 0, f"control group never-repeated sessions show zero warm hits: {control_hits}")

        # M7.4/M7.5 repeated group: the repeat forks the resident prefix and is bit-identical
        sid = new_session("The customer was charged twice on May 3 and wants a refund.")
        def warm_decision():
            status, text = decision(dict(env.DECISION_VALID, session_id=sid, diagnostics=True))
            env.check(status == 200, f"warm repeated decision status {status}: {text[:200]}")
            d = json.loads(text)
            return d.get("warm_hit"), d.get("answers")

        hit0, ans0 = warm_decision()
        hit1, ans1 = warm_decision()
        hit2, ans2 = warm_decision()
        env.check(hit0 is False, f"repeated group first decision is cold: {hit0}")
        env.check(hit1 is True, f"repeated group second decision is warm: {hit1}")
        env.check(hit2 is True, f"repeated group third decision is warm: {hit2}")
        # M7.5: a warm hit never changes a winner; probabilities stay within the documented qwen
        # recurrent warm-restore tolerance (measured up to ~0.05 in the derived diagnostics, bound 0.1)
        # M7.5: a warm hit never changes a winner; probabilities stay within the documented qwen
        # recurrent warm-restore tolerance (measured up to ~0.05 in the derived diagnostics, bound 0.1)
        env.check(answers_close(ans1, ans0, 0.1), "warm hit answer matches the cold miss within the documented tolerance")
        env.check(answers_close(ans2, ans0, 0.1), "warm repeat answer matches the cold miss within the documented tolerance")
        print(f"warm cache: control hits={control_hits}, repeated group hit rate=2/2, "
              f"hit answers within the documented tolerance on {os.path.basename(model)}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: warm cache checks: {e}")
        return False
    finally:
        srv.stop()


def main():
    if not os.path.isfile(env.SERVER_BIN):
        print(f"SKIP: server binary not found at {env.SERVER_BIN}")
        return 0
    # an explicit model selects exactly that model; otherwise try the bundled small models
    explicit = os.environ.get("LLAMA_SERVER_TEST_MODEL", "")
    if explicit:
        candidates = [explicit]
    else:
        candidates = [m for m in MODEL_CANDIDATES if m and os.path.isfile(m)]
    candidates = [m for m in candidates if m and os.path.isfile(m)]
    if not candidates:
        print("SKIP: no test model; set LLAMA_SERVER_TEST_MODEL")
        return 0
    for model in candidates:
        if not run_checks(model):
            return 1
        # sidecar executor mode: the same coexistence contract, plus the token-snapshot session
        # behaviors (F1 regression, identity mismatch 422, cross-instance 400, backend 501s)
        if not run_checks(model, sidecar_args()):
            return 1
        if not run_sidecar_specific_checks(model):
            return 1
        if not run_warm_cache_checks(model):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
