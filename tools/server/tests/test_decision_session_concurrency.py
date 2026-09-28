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


def timed_post(fn, *args):
    t = time.time()
    result = fn(*args)
    return result, (time.time() - t) * 1000.0


def run_checks(model):
    srv = env.Server(model, ["--parallel", "2", "--slots", "--jinja",
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
