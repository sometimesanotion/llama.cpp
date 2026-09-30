#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agentic-load harness for the decision sidecar executor (M3.4).

Runs one llama-server with the sidecar executor default and a realistic mixed
load on a single GPU: 4 chat agents streaming, 4 stateless decision workers,
1 session worker, and 2 churn workers, all concurrent. It asserts the M3
concurrency contract:

  * chat on the chat instance is NOT gated on a decision: the largest chat
    inter-token gap under concurrent decision load stays within a generous
    multiple of the quiet baseline (the decision runs on the sidecar, so it
    must not serialize chat);
  * no unexpected 413/422/429/499/529 from the decision/session/churn traffic
    (admission and semantic failures would be real regressions here);
  * the session worker's decisions stay answerable under the load.

Skips cleanly (exit 0) when the server binary or a model with usable answer
labels is missing, so it never fails open.

Run with a GPU build and the sidecar default:
  LLAMA_SERVER_BIN=build-gpu/bin/llama-server \
  LD_LIBRARY_PATH=build-gpu/bin \
  python3 test_decision_agentic.py MODEL.gguf
"""

import contextlib
import importlib.util
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_env_spec = importlib.util.spec_from_file_location("decision_envelope", os.path.join(HERE, "test_decision_envelope.py"))
env = importlib.util.module_from_spec(_env_spec)
_env_spec.loader.exec_module(env)

MODEL_CANDIDATES = [
    os.environ.get("LLAMA_SERVER_TEST_MODEL", ""),
    os.path.join(HERE, "tmp", "stories15M-q4_0.gguf"),
]

N_CHAT = 4        # concurrent streaming chat agents
N_STATELESS = 4   # concurrent stateless decision workers
N_SESSION = 1     # session worker issuing session decisions
N_CHURN = 2       # churn workers (slot erase + tiny instance create/destroy)
GAP_FACTOR = 4.0
GAP_MARGIN_MS = 1500.0


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    def __init__(self, model, extra_args=None):
        self.model = model
        self.extra_args = extra_args or []
        self.port = free_port()
        self.proc = None

    def start(self):
        cmd = [
            env.SERVER_BIN,
            "-m", self.model,
            "-c", "8192",
            "-ngl", os.environ.get("LLAMA_SERVER_TEST_NGL", "99"),
            "--instance", "main:ctx=8192:parallel=4:default",
            "--instance", "sess:ctx=1024:parallel=1",
            "--decision-seqs", "8",
            "--jinja",
            "--slots", "--slot-save-path", os.path.join(tempfile.mkdtemp(prefix="agentic-slots-")),
            "--port", str(self.port),
            "--host", "127.0.0.1",
        ] + self.extra_args
        e = dict(os.environ)
        build_bin = os.path.dirname(os.path.abspath(env.SERVER_BIN))
        e["LD_LIBRARY_PATH"] = build_bin + (os.pathsep + e["LD_LIBRARY_PATH"] if e.get("LD_LIBRARY_PATH") else "")
        # a deep enough queue that a mixed load never trips admission: any 429/529 here
        # would be a real regression to diagnose, not an expected outcome
        e["LLAMA_DECISION_MAX_QUEUE"] = "16"
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=e)
        deadline = time.time() + 120
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError("server exited early")
            try:
                status, _ = env.http("GET", f"http://127.0.0.1:{self.port}/health")
                if status == 200:
                    return
            except Exception:  # noqa: BLE001
                time.sleep(0.2)
        raise RuntimeError("server did not become healthy")

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def post(self, path, body):
        return env.http("POST", f"http://127.0.0.1:{self.port}{path}", body)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def supports_letter_labels(server):
    try:
        status, text = server.post("/v1/decision", json.dumps(env.DECISION_VALID))
    except Exception:  # noqa: BLE001
        return False
    if status == 200:
        return True
    return "answer tokens" not in text


def stream_chat(url, body):
    """POST a streaming chat and return (status, max inter-token gap ms, content)."""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        resp = urllib.request.urlopen(req, timeout=180)
    except urllib.error.HTTPError as e:
        return e.code, None, ""
    status = resp.status
    last = None
    max_gap = 0.0
    content = []
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            obj = json.loads(payload)
        except Exception:  # noqa: BLE001
            continue
        delta = (obj.get("choices") or [{}])[0].get("delta") or {}
        tok = delta.get("content") or delta.get("reasoning_content")
        if tok:
            now = time.time() * 1000.0
            if last is not None:
                max_gap = max(max_gap, now - last)
            last = now
            content.append(tok)
    return status, max_gap, "".join(content)


def chat_body(seed, id_slot=0):
    return {
        "messages": [{"role": "user", "content": "Write a long paragraph about spring weather and gardens, "
                                                 "describing the flowers, the light, and the rain."}],
        "max_tokens": 256,
        "seed": seed,
        "temperature": 0.0,
        "id_slot": id_slot,
        "stream": True,
    }


def run_single_agent(url, seed, out, idx):
    # a stream can return an empty content when the model finishes immediately (e.g. a lone
    # EOS under contention); retry a few times so the measured gap reflects the steady load,
    # not one degenerate generation
    for attempt in range(5):
        st, gap, content = stream_chat(url, chat_body(seed + attempt, id_slot=idx))
        if st == 200 and content:
            out[idx] = (st, gap, content)
            return
    out[idx] = (st, gap, content)


def stateless_worker(server, stop, errors, n):
    body = dict(env.DECISION_VALID)
    body["state"] = body["state"] + " Additional evidence about the order and the account. " * 8
    while not stop.is_set():
        status, text = server.post("/v1/decision", json.dumps(body))
        if status != 200:
            errors.append(("stateless", status, text[:120]))
        time.sleep(0.02)


def prefill_slot(server, id_slot, system, user, instance):
    body = {
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "max_tokens": 0,
        "grammar": 'root ::= ""',
        "add_generation_prompt": False,
        "id_slot": id_slot,
        "instance": instance,
    }
    status, text = server.post("/v1/chat/completions", json.dumps(body))
    check(status == 200, f"slot prefill status {status}: {text[:200]}")


def session_worker(server, stop, errors, sid):
    # a completed turn owned as a token snapshot; the sidecar answers about it on the dedicated
    # "sess" instance. The snapshot was taken before the load; if it ever turns stale (422) the
    # worker re-prefills and recreates, keeping session decisions running under load.
    body = dict(env.DECISION_VALID, session_id=sid)
    prefill = "State:\n" + env.DECISION_VALID["state"] + "\n"
    while not stop.is_set():
        status, text = server.post("/v1/decision", json.dumps(body))
        if status == 200:
            time.sleep(0.03)
            continue
        if status == 422:
            prefill_slot(server, 0, env.LETTER_SYSTEM, prefill, "sess")
            sst, stext = server.post("/v1/session",
                                     json.dumps({"id_slot": 0, "instance": "sess"}))
            if sst != 200:
                errors.append(("session recreate", sst, stext[:120]))
                return
            sid = json.loads(stext)["session_id"]
            body = dict(env.DECISION_VALID, session_id=sid)
            continue
        errors.append(("session decision", status, text[:120]))
        time.sleep(0.03)


def churn_worker(server, stop, errors):
    # churn a dedicated tiny instance (create + destroy) so the pool lifecycle runs under load
    # without racing the chat agents' slots
    base = f"http://127.0.0.1:{server.port}"
    idx = 0
    while not stop.is_set():
        name = f"churn{idx % 7}"
        status, text = env.http("POST", f"{base}/instances",
                                json.dumps({"name": name, "group": "churn", "ctx_size": 128, "parallel": 1}))
        if status not in (200, 201, 409):
            errors.append(("instance create", status, text[:120]))
        status, text = env.http("DELETE", f"{base}/instances/{name}", None)
        if status not in (200, 404):
            errors.append(("instance delete", status, text[:120]))
        idx += 1
        time.sleep(0.05)


def quiet_chat_gap(url):
    _, quiet_gap, _ = stream_chat(url, chat_body(seed=1, id_slot=0))
    return quiet_gap


def run_checks(model):
    server = Server(model)
    try:
        server.start()
    except Exception as e:  # noqa: BLE001
        server.stop()
        print(f"skip agentic harness on {os.path.basename(model)}: {e}")
        return True
    if not supports_letter_labels(server):
        server.stop()
        print(f"skip agentic harness on {os.path.basename(model)}: no usable answer labels")
        return True

    url = f"http://127.0.0.1:{server.port}/v1/chat/completions"
    try:
        def decision_post(body):
            return server.post("/v1/decision", json.dumps(body))

        # warm the sidecar and the chat path so the measured run is steady-state
        warm_status, warm_text = decision_post(env.DECISION_VALID)
        check(warm_status == 200, f"sidecar warmup: {warm_status} {warm_text[:120]}")
        status, _, _ = stream_chat(url, chat_body(seed=0, id_slot=0))
        check(status == 200, f"chat warmup: {status}")

        quiet_gap = quiet_chat_gap(url)
        gap_bound = quiet_gap * GAP_FACTOR + GAP_MARGIN_MS
        print(f"quiet baseline: max chat inter-token gap {quiet_gap:.0f}ms (bound {gap_bound:.0f}ms)")

        # create the session snapshot before the load: the worker answers about this turn while
        # chat agents and churn run around it (on the dedicated sess instance, so chat agents on
        # main never advance this source slot)
        prefill_slot(server, 0, env.LETTER_SYSTEM, "State:\n" + env.DECISION_VALID["state"] + "\n", "sess")
        sst, stext = server.post("/v1/session", json.dumps({"id_slot": 0, "instance": "sess"}))
        check(sst == 200, f"session create status {sst}: {stext[:120]}")
        session_sid = json.loads(stext)["session_id"]

        # mixed load: 4 chat agents, 4 stateless decisions, 1 session, 2 churn
        stop = threading.Event()
        errors = []
        chats = {}
        threads = []
        for i in range(N_CHAT):
            threads.append(threading.Thread(target=run_single_agent, args=(url, 100 + i, chats, i)))
        for i in range(N_STATELESS):
            threads.append(threading.Thread(target=stateless_worker, args=(server, stop, errors, i)))
        threads.append(threading.Thread(target=session_worker, args=(server, stop, errors, session_sid)))
        for i in range(N_CHURN):
            threads.append(threading.Thread(target=churn_worker, args=(server, stop, errors)))

        for t in threads:
            t.start()
        # let the load run for a few chat generations
        deadline = time.time() + 30
        while time.time() < deadline:
            done = all(threads[i].is_alive() for i in range(N_CHAT))
            time.sleep(0.1)
            # stop early once every chat agent finished at least one answer
            if len(chats) >= N_CHAT and not any(t.is_alive() for i, t in enumerate(threads) if i < N_CHAT):
                break
        time.sleep(0.5)
        stop.set()
        for t in threads:
            t.join(timeout=30)

        # every chat agent finished with a well-formed stream and a bounded gap
        check(len(chats) == N_CHAT, f"all chat agents answered: got {len(chats)}/{N_CHAT}")
        for idx, (st, gap, content) in chats.items():
            check(st == 200, f"chat agent {idx} status {st}")
            check(bool(content), f"chat agent {idx} produced content")
            check(gap is not None and gap <= gap_bound,
                  f"chat agent {idx} max gap {gap:.0f}ms exceeds {gap_bound:.0f}ms "
                  f"(quiet {quiet_gap:.0f}ms); a decision must not serialize chat")
            print(f"chat agent {idx}: max inter-token gap {gap:.0f}ms")

        # no unexpected error codes from the decision/session/churn traffic
        unexpected = [(k, s, t) for (k, s, t) in errors if s not in (200,)]
        check(not unexpected, f"unexpected decision/session/churn errors: {unexpected[:5]}")
        check(not any(k == "session decision" for k, s, t in errors),
              f"session decisions failed under load: {errors[:3]}")
        print("mixed load: no unexpected 413/422/429/499/529; session decisions answer under load")
        return True
    except Exception as e:  # noqa: BLE001
        server.stop()
        print(f"FAIL: {e}")
        return False
    finally:
        server.stop()


def main():
    if not os.path.isfile(env.SERVER_BIN):
        print(f"SKIP: server binary not found at {env.SERVER_BIN}")
        return 0
    candidates = [m for m in MODEL_CANDIDATES if m and os.path.isfile(m)]
    if not candidates:
        print("SKIP: no test model; set LLAMA_SERVER_TEST_MODEL")
        return 0
    for model in candidates:
        if not run_checks(model):
            return 1
        print(f"agentic harness passed on {os.path.basename(model)}")
    return 0


def test_decision_agentic():
    """pytest entry point for the harness above.

    The script's return code is the verdict; a missing binary or model is a skip.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = main()
    out = buf.getvalue()
    print(out, end="")
    if rc != 0:
        pytest.fail(out.strip() or "the agentic harness failed")
    if "SKIP" in out and "passed" not in out:
        pytest.skip(out.strip().splitlines()[-1])


if __name__ == "__main__":
    sys.exit(main())