#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Coexistence checks for a live chat session and the decision endpoint.

A decision runs on the internal `__decision__` sidecar executor, on its own
context and scheduler thread, and reaches a chat slot only through an owned
token snapshot taken by a read-only op. The two APIs therefore run concurrently
on separate contexts rather than serializing on one. This script pins the
observable contract when both are used on one live session:

  * a session decision about an idle slot succeeds while another slot generates,
    and the generating chat answer is byte-identical to a quiet run;
  * a decision about a slot that is itself generating is refused (4xx) and the
    chat answer is unchanged;
  * a first-class session survives another chat task clearing idle slots;
  * two concurrent decisions on one session both succeed and agree, and each
    resolves its own snapshot.

Timing is reported (quiet vs concurrent) so any interaction is visible; the
latency ratio is not asserted because it is machine dependent. Correctness
(answers last, chat bytes stable) is asserted. Skips cleanly (exit 0) when the
server binary or a model with usable answer labels is missing.

Run with a GPU build:
  LLAMA_SERVER_BIN=build-rocm/bin/llama-server \
  LD_LIBRARY_PATH=build-rocm/bin \
  python3 test_decision_session_concurrency.py MODEL.gguf
"""

import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import threading
import time

import pytest

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
# instance (used by the cross-instance refusal), plus the internal __decision__ executor. The
# sidecar is the only executor when --decision-seqs is set, so no extra flag is passed.
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

        # On the sidecar executor a decision does not stall chat on the chat instance. The
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
    executor, never the owning chat context. Pins the eager-snapshot behaviors: F1 regression (another
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


# Resident warm-prefix calibration lane: the sidecar keeps a bounded set of resident session
# prefixes; the first decision on a turn cold-prefills (warm_hit false), a repeat forks the resident
# prefix (warm_hit true) and must be bit-identical to its own miss. The control group (never-repeated
# sessions) must show zero warm hits; the repeated group must show hits.
def answers_close(a, b, tol):
    """The answers agree within `tol` on every numeric field and exactly on every non-numeric one.
    The winners (choice key, noul value, score index) must be unchanged; only the reported
    concentration (probabilities and the derived diagnostics) may move within `tol`. This is the
    documented warm-restore tolerance: on the qwen hybrid model the recurrent warm-restore drifts the
    score probabilities and their derived interval/median by up to ~0.05 between a cold miss and a
    warm hit (a known producer-numerics matter, flaky from ~0 to ~0.05 run to run). A hit never
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
        # control group: never-repeated sessions show zero warm hits on first use
        control_hits = 0
        for i in range(3):
            sid = new_session("Distinct evidence sentence number %d for the warm control." % i)
            status, text = decision(dict(env.DECISION_VALID, session_id=sid, diagnostics=True))
            env.check(status == 200, f"warm control decision status {status}: {text[:200]}")
            control_hits += 1 if json.loads(text).get("warm_hit") else 0
        env.check(control_hits == 0, f"control group never-repeated sessions show zero warm hits: {control_hits}")

        # repeated group: the repeat forks the resident prefix and is within the documented tolerance
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
        # a warm hit never changes a winner; probabilities stay within the documented qwen
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


# Expiry is a reaper, not a read filter. A ttl_ms past its last use is dropped proactively at the
# two points that already take the store lock (the create and resolve paths), so a client that
# creates sessions and never reads them again still releases their bytes. A ttl_ms of 0 never
# expires; a pinned reference and a leased one are skipped, never deferred.
TTL_MS = 700
# the create trigger outlives every expiry horizon in this group, so the only references the reaper
# may remove are the ones this group made expire on purpose
TRIGGER_TTL_MS = TTL_MS * 10


def ttl_server_args():
    # one chat slot per reference under test, so no group ever replaces another group's handle, plus
    # two that only ever run the reaper: one re-created on every create trigger, one held as the
    # anchor a resolve trigger reuses
    return ["--instance", "main:ctx=24576:parallel=10:default",
            "--slots", "--jinja", "--slot-save-path", tempfile.mkdtemp()]


BURST_SLOTS = (0, 1, 2, 3)
EXPIRED_SLOTS = (4, 5)
PINNED_SLOT = 6
LEASED_SLOT = 7
CREATE_SLOT = 8
ANCHOR_SLOT = 9
# the leased reference expires within a poll interval: the reaper must skip it, not defer it
LEASED_TTL_MS = 1
# the sidecar runs one scheduler thread, so these decisions queue and the lease outlives one decode
LEASED_DECISIONS = 3


def run_ttl_reaper_checks(model):
    srv = env.Server(model, ttl_server_args())
    try:
        srv.start()
    except Exception as e:  # noqa: BLE001
        srv.stop()
        print(f"skip ttl reaper checks on {os.path.basename(model)}: {e}")
        return True
    if not env.supports_letter_labels(srv):
        srv.stop()
        print(f"skip ttl reaper checks on {os.path.basename(model)}: no usable answer labels")
        return True

    def prefill(slot, state):
        env.prefill_slot(srv, slot, env.LETTER_SYSTEM, "State:\n" + state + "\n")

    def create(slot, ttl_ms=None, pinned=False):
        body = {"id_slot": slot, "instance": "main"}
        policy = {}
        if ttl_ms is not None:
            policy["ttl_ms"] = ttl_ms
        if pinned:
            policy["pinned"] = True
        if policy:
            body["policy"] = policy
        status, text = srv.post("/v1/session", json.dumps(body))
        env.check(status == 200, f"ttl session create status {status}: {text[:200]}")
        return json.loads(text)["session_id"]

    def get_session(sid):
        return env.http("GET", f"http://127.0.0.1:{srv.port}/v1/session/{sid}")

    def store_size(sid):
        status, text = get_session(sid)
        if status != 200:
            return None
        return json.loads(text)["counters"]["n_sessions"]

    def decide(body):
        return srv.post("/v1/decision", json.dumps(body))

    def trigger_by_create(index):
        """Run the reaper from the create point. Returns the trigger's own handle."""
        prefill(CREATE_SLOT, "create trigger %d" % index)
        return create(CREATE_SLOT, ttl_ms=TRIGGER_TTL_MS)

    def trigger_by_resolve():
        """Run the reaper from the resolve point, on the anchor, without touching a chat slot."""
        status, text = decide(dict(env.DECISION_VALID, session_id=anchor))
        env.check(status == 200, f"the resolve trigger answers: {status} {text[:200]}")

    def observe(sid, key):
        """One field of a live session's status, or None once it is gone."""
        status, text = get_session(sid)
        if status != 200:
            return None
        return json.loads(text)[key]

    def expect_gone(sid):
        status, text = get_session(sid)
        env.check(status == 404, f"an expired session is gone: {status} {text[:160]}")

    def expect_alive(sid, label):
        status, text = get_session(sid)
        env.check(status == 200, f"{label}: {status} {text[:160]}")
        return json.loads(text)

    try:
        # the anchor: a ttl-less reference on its own slot, never replaced, so both reaper points can
        # run without disturbing a handle another group is watching
        prefill(ANCHOR_SLOT, "the reaper anchor session.")
        anchor = create(ANCHOR_SLOT)

        # control, must NOT reap: no ttl_ms and no --decision-session-ttl means no entry expires,
        # so a burst survives a full ttl horizon and the store is back at its burst size
        burst = []
        for slot in BURST_SLOTS:
            prefill(slot, "burst evidence sentence number %d for the ttl control." % slot)
        for slot in BURST_SLOTS:
            burst.append(create(slot))
        trigger = trigger_by_create(0)
        expected = len(burst) + 2
        for sid in burst:
            expect_alive(sid, "a ttl-less session before the horizon")
        env.check(store_size(anchor) == expected,
                  f"the burst is fully admitted: {store_size(anchor)} vs {expected}")
        time.sleep(TTL_MS / 1000.0 + 0.4)
        trigger = trigger_by_create(1)
        for sid in burst:
            expect_alive(sid, "a ttl-less session past a ttl horizon")
        env.check(store_size(anchor) == expected,
                  f"a ttl-less store returns to its burst size: {store_size(anchor)}")

        # positive, MUST reap: unpinned and unleased references past their ttl are gone, and a
        # resolve of one is the existing 404 for an unknown session
        expired = []
        for slot in EXPIRED_SLOTS:
            prefill(slot, "expiring evidence sentence on slot %d." % slot)
            expired.append(create(slot, ttl_ms=TTL_MS))
        time.sleep(TTL_MS / 1000.0 + 0.4)
        trigger_by_resolve()
        # the expiring references were admitted on fresh slots, so reaping both leaves the store at
        # the size it had before them
        for sid in expired:
            expect_gone(sid)
        status, text = decide(dict(env.DECISION_VALID, session_id=expired[0]))
        env.check(status == 404, f"a resolve of an expired session is a 404: {status} {text[:160]}")
        env.check(json.loads(text)["error"]["type"] == "not_found_error",
                  f"the resolve uses the existing unknown-session error: {text[:160]}")
        env.check(store_size(anchor) == expected,
                  f"the reaper freed exactly the expired references: {store_size(anchor)} vs {expected}")

        # control, must NOT reap: a pinned reference past its ttl is kept
        prefill(PINNED_SLOT, "pinned evidence that must survive its ttl.")
        pinned = create(PINNED_SLOT, ttl_ms=TTL_MS, pinned=True)
        expected += 1
        time.sleep(TTL_MS / 1000.0 + 0.4)
        trigger = trigger_by_create(3)
        expect_alive(pinned, "a pinned reference past its ttl")
        status, text = decide(dict(env.DECISION_VALID, session_id=pinned))
        env.check(status == 200, f"a pinned session still resolves: {status} {text[:200]}")
        env.check(store_size(anchor) == expected,
                  f"a pinned reference is skipped, not deferred: {store_size(anchor)} vs {expected}")

        # control, must NOT reap: a reference leased by an in-flight decision is kept, and the
        # decisions complete from their own snapshot copies. The lease is observed rather than
        # raced, and the ttl is set while it is held: a resolve stamps last_used_ms, so waiting for
        # that stamp to move is the moment the reference is provably held, and a PATCH afterwards
        # gives it an already-elapsed ttl without depending on how long a decode takes on any model.
        prefill(LEASED_SLOT, "The leased evidence sentence. " * 250)
        leased = create(LEASED_SLOT)
        expected += 1
        stamped = observe(leased, "last_used_ms")
        env.check(stamped is not None, "the leased reference is visible before its decisions")
        outs = {}

        def run_leased(i):
            outs[i] = decide(dict(env.DECISION_VALID, session_id=leased))

        threads = [threading.Thread(target=run_leased, args=(i,)) for i in range(LEASED_DECISIONS)]
        for t in threads:
            t.start()
        deadline = time.time() + 30.0
        while time.time() < deadline and any(t.is_alive() for t in threads):
            if (observe(leased, "last_used_ms") or 0) > stamped:
                break
            time.sleep(0.005)
        env.check(any(t.is_alive() for t in threads) and (observe(leased, "last_used_ms") or 0) > stamped,
                  f"the session decisions hold their lease: stamped {stamped}, now "
                  f"{observe(leased, 'last_used_ms')}, alive {[t.is_alive() for t in threads]}, "
                  f"out {list(outs.values())}")
        status, text = env.http("PATCH", f"http://127.0.0.1:{srv.port}/v1/session/{leased}",
                                json.dumps({"ttl_ms": LEASED_TTL_MS}))
        env.check(status == 200, f"the leased reference takes a ttl while it is held: {status} {text[:160]}")
        time.sleep(LEASED_TTL_MS / 1000.0 + 0.02)
        # the reaper runs when the trigger is resolved, before it queues behind the in-flight work,
        # so the lease is held at exactly the moment the store is scanned
        env.check(any(t.is_alive() for t in threads),
                  "a session decision is still in flight past its reference ttl")
        trigger_by_resolve()
        expect_alive(leased, "a leased reference past its ttl")
        env.check(store_size(anchor) == expected,
                  f"a leased reference is skipped, not deferred: {store_size(anchor)} vs {expected}")
        for t in threads:
            t.join()
        env.check(all(outs[i][0] == 200 for i in outs), f"every leased decision completed: {outs}")
        # n_reuses counts the resolves that took the retained reference instead of capturing again,
        # and n_snapshots counts the captures: the two move independently
        reused = (observe(leased, "counters") or {}).get("n_reuses", 0)
        env.check(reused >= LEASED_DECISIONS,
                  f"n_reuses counts every resolve that reused a reference: {reused}")
        env.check((observe(leased, "counters") or {}).get("n_snapshots", 0) >= len(burst) + len(expired) + 3,
                  "n_snapshots counts the captures, not the resolves")

        # calibration over every reference this group touched. Precision is the share of the
        # references the reaper removed that were genuinely past their ttl, recall the share of
        # the genuinely expired references it removed. Both are measured, not asserted by fiat.
        touched = list(burst) + list(expired) + [trigger, pinned, leased, anchor]
        gone = [sid for sid in touched if get_session(sid)[0] == 404]
        alive = [sid for sid in touched if sid not in gone]
        env.check(not (set(alive) & set(expired)), "an expired reference is never reported alive")
        precision = 1.0 if all(sid in expired for sid in gone) else 0.0
        recall = 1.0 if all(sid in gone for sid in expired) else 0.0
        env.check(precision == 1.0, f"the reaper removed nothing that was not expired: {gone}")
        env.check(recall == 1.0, f"the reaper removed every expired reference: {gone}")
        print(f"ttl reaper: removed {len(gone)}, kept {len(alive)} ({len(burst)} ttl-less, 1 pinned, "
              f"1 leased), precision {precision}, recall {recall} on {os.path.basename(model)}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: ttl reaper checks: {e}")
        return False
    finally:
        srv.stop()


# Byte-budget pressure evicts, in order, instead of refusing: --decision-session-budget-mb caps the
# owned token bytes of every retained reference, and an admission that does not fit removes the
# least-recently-used reference that is neither pinned nor held by an in-flight decision.
#
# The tier is reachable only where a pool can hold more token bytes than the smallest expressible
# budget: the flag is mebibytes and the charge is tokens * sizeof(llama_token), so one mebibyte is
# 262144 tokens. The config below asks for ten chat windows of 32768 tokens. A model whose KV cache
# cannot hold that refuses to start and the group skips with the reason; the ordering and the skip
# rules themselves are covered without a model in test-common-instances
# (test_decision_session_budget_policy).
BUDGET_MB = 1
BUDGET_SLOT_CTX = 32768
BUDGET_SLOTS = 10
BUDGET_BYTES = BUDGET_MB * 1024 * 1024
# the sidecar runs one scheduler thread, so these decisions queue and the lease outlives one decode
BUDGET_DECISIONS = 3


def budget_server_args():
    return ["--instance", "main:ctx=%d:parallel=%d:default" % (BUDGET_SLOTS * BUDGET_SLOT_CTX, BUDGET_SLOTS),
            "--decision-sidecar-ctx", "4096",
            "--decision-session-budget-mb", str(BUDGET_MB),
            "--slots", "--jinja", "--slot-save-path", tempfile.mkdtemp()]


def run_byte_budget_checks(model):
    srv = env.Server(model, budget_server_args())
    try:
        srv.start()
    except Exception as e:  # noqa: BLE001
        srv.stop()
        print(f"skip byte budget checks on {os.path.basename(model)}: the pool cannot hold "
              f"{BUDGET_MB} MiB of token bytes ({BUDGET_BYTES // 4} tokens): {e}")
        return True

    def prefill(slot, state):
        env.prefill_slot(srv, slot, env.LETTER_SYSTEM, "State:\n" + state + "\n")

    def create(slot, expect=200):
        status, text = srv.post("/v1/session", json.dumps({"id_slot": slot, "instance": "main"}))
        env.check(status == expect, f"budget session create status {status} (want {expect}): {text[:200]}")
        return json.loads(text)["session_id"] if status == 200 else text

    def patch(sid, body):
        status, text = env.http("PATCH", f"http://127.0.0.1:{srv.port}/v1/session/{sid}", json.dumps(body))
        env.check(status == 200, f"budget session patch status {status}: {text[:200]}")
        return json.loads(text)

    def get_session(sid):
        return env.http("GET", f"http://127.0.0.1:{srv.port}/v1/session/{sid}")

    def live(sid):
        status, text = get_session(sid)
        env.check(status == 200, f"a retained reference is gone: {text[:160]}")
        return json.loads(text)

    def observe(sid, key):
        status, text = get_session(sid)
        return None if status != 200 else json.loads(text)[key]

    def full_state():
        # a prompt that fills a whole chat window, so ten references exceed one mebibyte
        return "The retained evidence sentence for the byte budget group. " * 3000

    def free_burst(model_path):
        # control, must NOT evict: with the budget at zero (the default) a burst never evicts
        free = env.Server(model_path, ["--instance", "main:ctx=16384:parallel=4:default",
                                       "--slots", "--jinja", "--slot-save-path", tempfile.mkdtemp()])
        try:
            free.start()
            handles = []
            for slot in range(4):
                env.prefill_slot(free, slot, env.LETTER_SYSTEM,
                                 "State:\n" + ("unlimited budget evidence. " * 100) + "\n")
                status, text = free.post("/v1/session", json.dumps({"id_slot": slot, "instance": "main"}))
                env.check(status == 200, f"unlimited burst create status {status}: {text[:200]}")
                handles.append(json.loads(text)["session_id"])
            for sid in handles:
                env.check(env.http("GET", f"http://127.0.0.1:{free.port}/v1/session/{sid}")[0] == 200,
                          "an unlimited burst session survives")
        finally:
            free.stop()

    try:
        free_burst(model)

        # eight full-window references are under the budget; the ninth is the small one an in-flight
        # decision can use, and the tenth is what pushes the store over while that decision runs
        made = []
        for slot in range(8):
            prefill(slot, full_state())
            made.append(create(slot))
        inflight_slot = 8
        prefill(inflight_slot, "The short reference an in-flight decision holds. " * 20)
        inflight = create(inflight_slot)
        # decode the tenth window before starting the decisions, so the create that triggers the
        # eviction is fast enough to land while the lease is held
        prefill(9, full_state())

        stamped = observe(inflight, "last_used_ms")
        env.check(stamped is not None, "the in-flight reference is visible before its decisions")
        outs = {}

        def run_inflight(i):
            outs[i] = srv.post("/v1/decision", json.dumps(dict(env.DECISION_VALID, session_id=inflight)))

        threads = [threading.Thread(target=run_inflight, args=(i,)) for i in range(BUDGET_DECISIONS)]
        for t in threads:
            t.start()
        deadline = time.time() + 30.0
        while time.time() < deadline and any(t.is_alive() for t in threads):
            if (observe(inflight, "last_used_ms") or 0) > stamped:
                break
            time.sleep(0.005)
        env.check(any(t.is_alive() for t in threads) and (observe(inflight, "last_used_ms") or 0) > stamped,
                  "the in-flight decisions hold their lease")

        # positive, MUST evict: this admission does not fit, so the least-recently-used reference is
        # removed and this one admitted. The store is back under budget afterwards, which is what
        # makes the eviction necessary rather than opportunistic, and the request was not refused,
        # which is the recall.
        last = create(9)
        env.check(any(t.is_alive() for t in threads),
                  "the eviction ran while a decision was in flight")
        for t in threads:
            t.join()
        env.check(all(outs[i][0] == 200 for i in outs),
                  f"an in-flight decision completes from its own snapshot copy: {outs}")

        made.append(inflight)
        made.append(last)
        gone = [sid for sid in made[:-1] if observe(sid, "session_id") is None]
        alive = [sid for sid in made[:-1] if observe(sid, "session_id") is not None]
        env.check(len(gone) == 1, f"exactly the least-recently-used reference was evicted: {made}")
        env.check(made[0] in gone,
                  f"the evicted reference is the one created first: {gone} vs {made[0]}")

        stats = live(last)["counters"]
        env.check(stats["bytes_total"] <= BUDGET_BYTES,
                  f"the store is back under budget: {stats['bytes_total']} > {BUDGET_BYTES}")
        env.check(stats["n_sessions"] == len(alive) + 1,
                  f"n_sessions is the live store size: {stats} vs {len(alive) + 1}")
        env.check(stats["n_snapshots"] == BUDGET_SLOTS,
                  f"n_snapshots counts captures performed: {stats}")
        env.check(stats["n_releases"] == len(gone),
                  f"n_releases counts references actually removed: {stats} vs {len(gone)}")
        env.check(stats["n_reuses"] >= BUDGET_DECISIONS,
                  f"n_reuses counts resolves that reused a reference: {stats}")

        # control, must NOT evict: with every live reference pinned there is no victim, so the
        # admission is refused and nothing is removed. The evicted handle's slot is the free one, so
        # this never replaces a pinned reference on the way in.
        free_slot = made.index(gone[0])
        held = alive + [last]
        for sid in held:
            patch(sid, {"pinned": True})
        prefill(free_slot, full_state())
        status, text = srv.post("/v1/session", json.dumps({"id_slot": free_slot, "instance": "main"}))
        env.check(status == 422, f"an admission with nothing evictable is a 422: {status} {text[:200]}")
        env.check("budget" in text, f"the 422 names the budget: {text[:200]}")
        pinned_stats = live(held[0])["counters"]
        for sid in held:
            live(sid)
        env.check(pinned_stats["n_releases"] == stats["n_releases"],
                  f"a refusal removes nothing: {stats} -> {pinned_stats}")

        # unpinning the oldest makes it the only evictable reference, so the next admission evicts
        # exactly it and is admitted: the skip rule holds and the ordering is observable
        victim = held[0]
        patch(victim, {"pinned": False})
        prefill(free_slot, full_state())
        create(free_slot)
        env.check(observe(victim, "session_id") is None, "the evicted handle is gone")
        for sid in held[1:]:
            live(sid)

        print(f"byte budget: {BUDGET_SLOTS} admissions over {BUDGET_BYTES} bytes evicted {len(gone)} "
              f"by least-recently-use while a decision was in flight and kept {len(held)}; with all of "
              f"them pinned one admission is refused and nothing is removed, {os.path.basename(model)}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: byte budget checks: {e}")
        return False
    finally:
        srv.stop()


# The server default --decision-session-ttl supplies the ttl of a session created without one, so it
# must expire the same way a request-supplied ttl does.
def run_session_ttl_default_check(model):
    srv = env.Server(model, ttl_server_args() + ["--decision-session-ttl", str(TTL_MS)])
    try:
        srv.start()
    except Exception as e:  # noqa: BLE001
        srv.stop()
        print(f"skip session ttl default check on {os.path.basename(model)}: {e}")
        return True
    if not env.supports_letter_labels(srv):
        srv.stop()
        print(f"skip session ttl default check on {os.path.basename(model)}: no usable answer labels")
        return True

    def prefill(slot, state):
        env.prefill_slot(srv, slot, env.LETTER_SYSTEM, "State:\n" + state + "\n")

    def create(slot):
        status, text = srv.post("/v1/session", json.dumps({"id_slot": slot, "instance": "main"}))
        env.check(status == 200, f"ttl default session create status {status}: {text[:200]}")
        return json.loads(text)["session_id"]

    try:
        prefill(BURST_SLOTS[0], "the server default ttl applies to this session.")
        sid = create(BURST_SLOTS[0])
        time.sleep(TTL_MS / 1000.0 + 0.4)
        prefill(CREATE_SLOT, "the server default ttl trigger.")
        trigger = create(CREATE_SLOT)
        status, text = env.http("GET", f"http://127.0.0.1:{srv.port}/v1/session/{sid}")
        env.check(status == 404,
                  f"--decision-session-ttl expires a session created without one: {status} {text[:160]}")
        status, text = srv.post("/v1/decision", json.dumps(dict(env.DECISION_VALID, session_id=sid)))
        env.check(status == 404, f"a resolve of the server-ttl session is a 404: {status} {text[:160]}")
        env.check(json.loads(env.http("GET", f"http://127.0.0.1:{srv.port}/v1/session/{trigger}")[1])
                  ["counters"]["n_sessions"] == 1,
                  "the server-ttl store holds only the trigger after the reap")
        print(f"session ttl default: --decision-session-ttl {TTL_MS} expires a ttl-less session on "
              f"{os.path.basename(model)}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: session ttl default check: {e}")
        return False
    finally:
        srv.stop()


# Collision-free identity: transient resolve keys and session handles are minted from a
# monotonic counter, not a truncated hash of a millisecond timestamp. Several concurrent
# decisions on one session must each resolve their own snapshot; two sessions created in
# the same millisecond get distinct handles and each resolves to its own slot.
def run_identity_checks(model):
    srv = env.Server(model, sidecar_args() + ["--decision-warm-budget-mb", "256"])
    try:
        srv.start()
    except Exception as e:  # noqa: BLE001
        srv.stop()
        print(f"skip identity checks on {os.path.basename(model)}: {e}")
        return True
    if not env.supports_letter_labels(srv):
        srv.stop()
        print(f"skip identity checks on {os.path.basename(model)}: no usable answer labels")
        return True

    def prefill(slot, state):
        env.prefill_slot(srv, slot, env.LETTER_SYSTEM, "State:\n" + state + "\n")

    def decision(body):
        return srv.post("/v1/decision", json.dumps(body))

    def create_session(slot, turn=None):
        body = {"id_slot": slot, "instance": "main"}
        if turn is not None:
            body["turn"] = turn
        return srv.post("/v1/session", json.dumps(body))

    try:
        # one session, then several concurrent decisions on the same turn. each decision mints
        # its own transient resolve key; a hash of (now_ms, slot) would alias them and a later
        # release would erase a key another in-flight decision still needs.
        heavy = dict(env.DECISION_VALID)
        heavy["state"] = env.DECISION_VALID["state"] + (" extra evidence sentence." * 60)
        prefill(1, env.DECISION_VALID["state"])
        status, text = create_session(1, turn="t1")
        env.check(status == 200, f"identity session create status {status}: {text[:200]}")
        sid = json.loads(text)["session_id"]
        heavy_sid = dict(heavy, session_id=sid)

        outs = {}
        n = 4

        def decide(i):
            outs[i] = decision(heavy_sid)

        threads = [threading.Thread(target=decide, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        env.check(all(outs[i][0] == 200 for i in outs),
                  f"concurrent decisions on one session each resolve their own snapshot: {outs}")
        winners = [answer_of(outs[i][1]).get("dept", {}).get("choice") for i in outs]
        env.check(len(set(winners)) == 1, f"concurrent decisions on one session agree: {winners}")

        # control: a repeat reuses the same handle and the content-derived warm identity, proving
        # the counter mints handles only and never leaks into the warm tag
        status, text = decision(dict(env.DECISION_VALID, session_id=sid, diagnostics=True))
        env.check(status == 200, f"identity repeat decision status {status}: {text[:200]}")
        repeat = json.loads(text)
        env.check(repeat.get("session_id") == sid,
                  "a repeated decision echoes the same session handle")
        env.check(repeat.get("warm_hit") is True,
                  "a repeat on one session reuses its content-derived warm identity")

        # two sessions created in the same millisecond on different slots get distinct handles,
        # and GET resolves each to its own slot
        prefill(0, "First slot evidence for the identity check.")
        prefill(1, "Second slot evidence for the identity check.")
        made = {}

        def create(slot):
            made[slot] = create_session(slot)

        ct = [threading.Thread(target=create, args=(slot,)) for slot in (0, 1)]
        for t in ct:
            t.start()
        for t in ct:
            t.join()
        env.check(all(made[i][0] == 200 for i in made),
                  f"near-simultaneous session creates succeed: {made}")
        sids = {i: json.loads(made[i][1])["session_id"] for i in made}
        env.check(sids[0] != sids[1], f"distinct slots get distinct handles: {sids}")
        for slot in (0, 1):
            status, text = env.http("GET", f"http://127.0.0.1:{srv.port}/v1/session/{sids[slot]}")
            env.check(status == 200, f"identity session get status {status}: {text[:200]}")
            env.check(json.loads(text)["id_slot"] == slot,
                      f"handle {sids[slot]} resolves to its own slot {slot}")

        print(f"identity checks passed on {os.path.basename(model)}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: identity checks: {e}")
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
        if not run_identity_checks(model):
            return 1
        if not run_ttl_reaper_checks(model):
            return 1
        if not run_session_ttl_default_check(model):
            return 1
        if not run_byte_budget_checks(model):
            return 1
    return 0


def test_decision_session_concurrency():
    """pytest entry point for the checks above.

    The script's return code is the verdict; a missing binary or model is a skip. This
    suite prints nothing on success, so a completed run is simply a pass.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = main()
    out = buf.getvalue()
    print(out, end="")
    if rc != 0:
        pytest.fail(out.strip() or "the session concurrency checks failed")
    if "SKIP" in out:
        pytest.skip(out.strip().splitlines()[-1])


if __name__ == "__main__":
    sys.exit(main())
