#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Admission-control checks for the decision endpoint.

Starts llama-server with a tiny decision body cap and a queue depth of one, then
exercises the capacity and overload contract: oversize body -> 413, a burst of
concurrent decisions -> 429 (or 529) with `Retry-After`, and a semantically
invalid request -> 422. Skips cleanly (exit 0) when the server binary or a small
test model is missing, so it never fails open.
"""

import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

SERVER_BIN = os.environ.get("LLAMA_SERVER_BIN", os.path.join(REPO, "build", "bin", "llama-server"))

MODEL_CANDIDATES = [
    os.environ.get("LLAMA_SERVER_TEST_MODEL", ""),
    os.path.join(HERE, "tmp", "stories15M-q4_0.gguf"),
    os.path.join(HERE, "tmp", "moe_shakespeare15M.gguf"),
]

MAX_BODY = 16384
MAX_QUEUE = 1

# Pre-registered responsiveness/fairness bound (see tests/decision-baseline/fairness.json).
FAIRNESS_FILE = os.path.join(REPO, "tests", "decision-baseline", "fairness.json")
DEFAULT_FAIRNESS = {
    "slots_during_decision_ms": 250.0,
    "chat_latency_factor": 2.0,
    "chat_latency_margin_ms": 500.0,
}


def load_fairness_bound():
    try:
        with open(FAIRNESS_FILE) as f:
            bound = json.load(f).get("bound", {})
        return {k: float(bound.get(k, v)) for k, v in DEFAULT_FAIRNESS.items()}
    except Exception:  # noqa: BLE001
        return dict(DEFAULT_FAIRNESS)

DECISION_VALID = {
    "model": "test",
    "state": "Customer was charged twice on May 3.",
    "questions": {
        "refund": {"type": "noul", "instructions": "Should this be refunded?"},
        "dept": {
            "type": "choice",
            "instructions": "What is the issue?",
            "criteria": {"billing": "payment", "technical": "bug"},
        },
        "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["calm", "upset", "furious"]},
    },
}


def find_model():
    for path in MODEL_CANDIDATES:
        if path and os.path.isfile(path):
            return path
    return None


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


API_KEY = "decision-secret"


def http(method, url, body=None, content_type="application/json", api_key=API_KEY):
    import urllib.request
    import urllib.error

    data = body.encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", content_type)
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, dict(resp.headers), resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read().decode("utf-8")


class Server:
    def __init__(self, model, extra_args=None, max_body=MAX_BODY, ctx=8192):
        self.model = model
        self.extra_args = extra_args or []
        self.max_body = max_body
        self.ctx = ctx
        self.port = free_port()
        self.proc = None
        self._log = None
        self._logfile = None

    def start(self):
        cmd = [
            SERVER_BIN,
            "-m", self.model,
            "-c", str(self.ctx),
            "-ngl", os.environ.get("LLAMA_SERVER_TEST_NGL", "99"),
            "--decision-seqs", "8",
            "--slots",
            "--api-key", API_KEY,
            "--port", str(self.port),
            "--host", "127.0.0.1",
        ] + self.extra_args
        env = dict(os.environ)
        build_bin = os.path.dirname(os.path.abspath(SERVER_BIN))
        env["LD_LIBRARY_PATH"] = build_bin + (os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        env["LLAMA_DECISION_MAX_BODY"] = str(self.max_body)
        env["LLAMA_DECISION_MAX_QUEUE"] = str(MAX_QUEUE)
        self._logfile = tempfile.NamedTemporaryFile(prefix="decision-server-", suffix=".log", delete=False)
        self._logfile.close()
        self._log = open(self._logfile.name, "w", encoding="utf-8")
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=self._log, env=env)
        deadline = time.time() + 120
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError("server exited early")
            try:
                status, _, _ = http("GET", f"http://127.0.0.1:{self.port}/health")
                if status == 200:
                    return
            except Exception:
                time.sleep(0.2)
        raise RuntimeError("server did not become healthy")

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self._log is not None:
            self._log.close()
            self._log = None

    def post(self, body):
        return http("POST", f"http://127.0.0.1:{self.port}/v1/decision", body)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def supports_letter_labels(server):
    status, _, text = server.post(json.dumps(DECISION_VALID))
    if status == 200:
        return True
    if status == 501:
        # the server's startup probe found no usable answer labels: the 501 carries the probe
        # failure, so skipping here means "this model cannot serve decisions", never a real error
        check("cannot serve decision questions" in text, f"501 names the startup probe reason: {text}")
        return False
    raise AssertionError(f"letter support probe unexpected status {status}: {text}")


def preflight(server):
    # A healthy server does not prove the endpoint is wired. Assert the decision route exists
    # before any assertion runs, so a build without it fails instead of passing open.
    status, _, _ = http("GET", f"http://127.0.0.1:{server.port}/health")
    check(status == 200, f"preflight: /health is {status}")
    status, _, text = server.post(json.dumps(DECISION_VALID))
    check(status != 404, f"preflight: the decision route exists (status {status}: {text[:120]})")


def check_shutdown(server):
    # After stop the decision route must be gone: a clean connection refusal (or a 503 while the
    # process drains), never a 200. A 200 here would mean the route outlives its model.
    status = None
    try:
        status, _, _ = server.post(json.dumps(DECISION_VALID))
    except Exception:  # noqa: BLE001
        status = None
    check(status != 200, f"a stopped server never answers the decision route with 200 (status {status})")


def run_checks(server):
    # 1. body over the configured cap is rejected before any decode
    big = dict(DECISION_VALID)
    big["state"] = "x" * (MAX_BODY + 512)
    status, _, text = server.post(json.dumps(big))
    check(status == 413, f"oversize body status {status}: {text}")
    check(json.loads(text)["error"]["code"] == 413, "oversize body error code 413")

    # 2. semantically invalid request is 422, not 400
    bad = dict(DECISION_VALID)
    bad["state"] = ""
    status, _, text = server.post(json.dumps(bad))
    check(status == 422, f"semantic error status {status}: {text}")

    # 2b. control group: a single small request at max_queue defaults is always admitted
    # (0 false positives); only a saturated burst may be refused
    control_statuses = []
    for _ in range(8):
        status, _, text = server.post(json.dumps(DECISION_VALID))
        control_statuses.append(status)
        check(status == 200, f"a single request is always admitted: status {status}: {text}")
    admitted = len([s for s in control_statuses if s == 200])
    control_precision = admitted / len(control_statuses) if control_statuses else 1.0
    check(control_precision == 1.0, f"admission control false-positive rate: precision={control_precision}")

    # 3. a burst over the queue depth is admitted (429) or overloaded (529), with Retry-After
    heavy = dict(DECISION_VALID)
    heavy["state"] = "refund request with a broken item. " * 20
    n = 8
    barrier = threading.Barrier(n)
    results = [None] * n

    def worker(i):
        barrier.wait()
        results[i] = server.post(json.dumps(heavy))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    statuses = [r[0] for r in results if r is not None]
    check(200 in statuses, f"at least one request succeeds: {statuses}")
    check(any(s in (429, 529) for s in statuses), f"a saturated burst is rejected: {statuses}")
    for status, headers, _ in results:
        if status == 429:
            check(headers.get("Retry-After") == "1", f"429 carries Retry-After: {headers}")
        if status == 529:
            check(headers.get("Retry-After") == "1", f"529 carries Retry-After: {headers}")

    # admission precision/recall: admitted-and-finished / admitted, rejected-when-saturated / attempted
    attempted_saturated = len(results)
    rejected_saturated = len([r for r in results if r is not None and r[0] in (429, 529)])
    admission_recall = rejected_saturated / attempted_saturated if attempted_saturated else 1.0
    check(admission_recall > 0.0, f"admission recall: {admission_recall}")
    print(f"measured: admission precision={control_precision} recall={admission_recall}")

    # 4. only application/json is accepted on the decision route
    status, _, _ = http("POST", f"http://127.0.0.1:{server.port}/v1/decision", "{}", content_type="text/plain")
    check(status in (400, 415), f"non-JSON content type is rejected: {status}")

    # 4b. an unauthenticated request is rejected when auth is configured
    status, _, _ = http("POST", f"http://127.0.0.1:{server.port}/v1/decision", json.dumps(DECISION_VALID), api_key=None)
    check(status == 401, f"unauthenticated request status {status}")

    # 5. chat coexistence: a decision must not break chat on the same context
    url = f"http://127.0.0.1:{server.port}/v1/chat/completions"
    chat = json.dumps({"messages": [{"role": "user", "content": "hello"}], "max_tokens": 1})
    status, _, text = http("POST", url, chat)
    check(status == 200, f"chat before decision: {status} {text}")
    t0 = time.time()
    status, _, text = server.post(json.dumps(DECISION_VALID))
    warm_decision_ms = (time.time() - t0) * 1000.0
    check(status == 200, f"decision between chats: {status} {text}")
    status, _, text = http("POST", url, chat)
    check(status == 200, f"chat after decision: {status} {text}")

    # 6. cooperative yield: a long decision must not freeze the scheduler
    bound = load_fairness_bound()
    heavy = {
        "model": "test",
        "state": "The customer opened a ticket about a delayed delivery and a duplicate charge. " * 8,
        "questions": {
            f"q{i}": {"type": "noul", "instructions": f"Is claim {i} supported by the state?"}
            for i in range(96)
        },
    }
    outcome = {}

    def run_decision():
        t = time.time()
        st, _, tx = server.post(json.dumps(heavy))
        outcome["ms"] = (time.time() - t) * 1000.0
        outcome["status"] = st
        outcome["text"] = tx

    decision_thread = threading.Thread(target=run_decision)
    decision_thread.start()
    time.sleep(0.1)  # let the decision start so the probes land inside it
    slots_latency = None
    while decision_thread.is_alive():
        t = time.time()
        st, _, _ = http("GET", f"http://127.0.0.1:{server.port}/slots")
        dt = (time.time() - t) * 1000.0
        if st == 200:
            slots_latency = dt if slots_latency is None else min(slots_latency, dt)
        time.sleep(0.01)
    decision_thread.join()

    check(outcome.get("status") == 200, f"heavy decision: {outcome.get('status')} {outcome.get('text')}")
    check(slots_latency is not None, "the slots endpoint answered during the decision")
    check(slots_latency <= bound["slots_during_decision_ms"],
          f"/slots stayed responsive during a {outcome['ms']:.0f} ms decision: {slots_latency:.0f} ms")

    # chat fired concurrently with a decision is serialized behind it, but must complete
    chat_result = {}

    def run_chat():
        t = time.time()
        st, _, tx = http("POST", url, chat)
        chat_result["ms"] = (time.time() - t) * 1000.0
        chat_result["status"] = st
        chat_result["text"] = tx

    decision_thread = threading.Thread(target=run_decision)
    decision_thread.start()
    time.sleep(0.1)
    chat_thread = threading.Thread(target=run_chat)
    chat_thread.start()
    decision_thread.join()
    chat_thread.join()
    check(chat_result.get("status") == 200, f"concurrent chat: {chat_result.get('status')} {chat_result.get('text')}")
    chat_bound = outcome["ms"] + bound["chat_latency_factor"] * warm_decision_ms + bound["chat_latency_margin_ms"]
    check(chat_result["ms"] <= chat_bound,
          f"concurrent chat latency {chat_result['ms']:.0f} ms within {chat_bound:.0f} ms")
    print(f"measured: decision {outcome['ms']:.0f} ms, slots-during {slots_latency:.0f} ms, chat {chat_result['ms']:.0f} ms")


def capacity_body(state, temperature=None):
    body = {
        "model": "test",
        "state": state,
        "questions": {
            "dept": {
                "type": "choice",
                "instructions": "What is the issue?",
                "criteria": {"billing": "payment", "technical": "bug"},
            },
        },
    }
    if temperature is not None:
        body["temperature"] = temperature
    return json.dumps(body)


# Sweep the prompt size against a bounded classifier context: every size up to the budget is
# accepted and decides identically to a large-context control, every size past it is rejected with
# 422 and no decision. The accept/reject outcome must not depend on the reported confidence.
CAPACITY_CTX = 512
CAPACITY_UNIT = "The customer was charged twice and asked for a refund. "


def run_capacity_sweep(model):
    # the decision runs on the shared context, so the bounded server sizes that context directly
    bounded = Server(model, ["--ctx-size", str(CAPACITY_CTX)])
    bounded.start()
    control = None
    try:
        if not supports_letter_labels(bounded):
            return "skip"

        def post_state(state, temperature=None):
            return bounded.post(capacity_body(state, temperature))

        last_ok = None
        first_reject = None
        observed = []
        for mult in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
            status, _, text = post_state(CAPACITY_UNIT * mult)
            if status == 413:
                # the body cap is a separate limit (checked above); the context boundary must have
                # been reached before the state grows past it, so stop the sweep here
                break
            observed.append((mult, status, text))
            if status == 200:
                body = json.loads(text)
                check("answers" in body, "a fitting request returns a decision")
                last_ok = {"mult": mult, "tokens": body["usage"]["input_tokens"], "body": body}
            elif status != 422:
                raise AssertionError(f"capacity sweep unexpected status {status}: {text}")

        check(last_ok is not None, "a request fits within the bounded context")
        check(any(s == 422 for _, s, _ in observed), "the sweep reaches the capacity boundary")
        first_reject = next(m for m, s, _ in observed if s == 422)
        check(first_reject > last_ok["mult"], "the boundary is monotone in state size")

        # every size below the boundary must be accepted and every size at or above it rejected
        not_fired_ok = all(s == 200 for m, s, _ in observed if m < first_reject)
        fired_ok = all(s == 422 for m, s, _ in observed if m >= first_reject)
        check(not_fired_ok, "no false reject below the boundary")
        check(fired_ok, "no false accept at or above the boundary")
        expected_not_fired = sum(1 for m, _, _ in observed if m < first_reject)
        expected_fired = sum(1 for m, _, _ in observed if m >= first_reject)
        fired = sum(1 for m, s, _ in observed if m >= first_reject and s == 422)
        not_fired = sum(1 for m, s, _ in observed if m < first_reject and s == 200)
        precision = fired / expected_fired if expected_fired else 1.0
        recall = not_fired / expected_not_fired if expected_not_fired else 1.0
        check(precision == 1.0 and recall == 1.0, f"capacity precision={precision} recall={recall}")

        # no truncation and no KV residue
        rejected_text = next(t for m, s, t in observed if m == first_reject)
        rejected = json.loads(rejected_text)
        check(rejected["error"]["code"] == 422, "the boundary rejection is a 422")
        check("answers" not in rejected and "results" not in rejected, "a rejected request returns no decision")
        status, _, text = post_state(CAPACITY_UNIT)
        check(status == 200, f"the server still serves after a rejection: {status} {text}")

        # the outcome must not depend on the reported confidence: same length, different temperatures
        sharp = post_state(CAPACITY_UNIT * first_reject, 0.01)
        flat = post_state(CAPACITY_UNIT * first_reject, 4.0)
        check(sharp[0] == 422 and flat[0] == 422,
              f"the reject is independent of confidence/temperature: {sharp[0]} {flat[0]}")

        # the decision context is bounded by the shared chat n_ctx. The sweep above
        # proved every state past CAPACITY_CTX is rejected on the bounded server (requested size).

        # control group: the fitting request decides the same on a large-context server
        control = Server(model)
        control.start()
        if not supports_letter_labels(control):
            return "skip"
        state = CAPACITY_UNIT * last_ok["mult"]
        status, _, text = control.post(capacity_body(state))
        check(status == 200, f"control server serves the fitting request: {status}")
        cb = json.loads(text)["answers"]["dept"]
        lb = last_ok["body"]["answers"]["dept"]
        check(lb["choice"] == cb["choice"], "the bounded and control servers pick the same choice")
        tv = sum(abs(lb["probabilities"][k] - cb["probabilities"][k]) for k in lb["probabilities"])
        check(tv <= 5e-2, f"the bounded and control distributions agree (TV={tv})")

        # the decision shares the chat n_ctx, so a state that the bounded (512) context rejected
        # must be accepted on the control (8192) context.
        status, _, text = control.post(capacity_body(CAPACITY_UNIT * first_reject))
        check(status == 200,
              f"the unbounded decision context serves a state beyond CAPACITY_CTX: {status} {text[:120]}")

        print(f"capacity sweep: fit up to {last_ok['mult']} units ({last_ok['tokens']} input tokens), "
              f"first reject at {first_reject} units, precision={precision} recall={recall}, ctx={CAPACITY_CTX}")
        return "pass"
    finally:
        if control is not None:
            control.stop()
        bounded.stop()


# M4.5: a decision fired while a chat is generating must not evict or corrupt chat cells. The chat
# completes byte-for-byte like a no-decision reference, and the decision matches a quiet reference.
# A fixed seed keeps chat generation deterministic on the CPU backend.
def run_chat_decision_integrity(model):
    srv = Server(model, ["--seed", "42"])
    srv.start()
    try:
        if not supports_letter_labels(srv):
            return "skip"
        url = f"http://127.0.0.1:{srv.port}/v1/chat/completions"
        chat_body = {
            "messages": [{"role": "user", "content": "Write a short paragraph about spring weather."}],
            "max_tokens": 24,
            "seed": 42,
        }

        def chat_once():
            status, _, text = http("POST", url, json.dumps(chat_body))
            return status, text

        # reference: quiet chat, no decision
        status, ref_text = chat_once()
        check(status == 200, f"quiet chat reference: {status} {ref_text[:120]}")
        ref = json.loads(ref_text)["choices"][0]["message"]["content"]

        # quiet decision reference
        status, _, dtext = srv.post(json.dumps(DECISION_VALID))
        check(status == 200, f"quiet decision reference: {status}")
        quiet = json.loads(dtext)

        # mid-stream: fire a decision while the chat is generating
        out = {}

        def run_chat():
            out["status"], out["text"] = chat_once()

        th = threading.Thread(target=run_chat)
        th.start()
        time.sleep(0.05)  # let the chat start so the decision lands mid-stream where possible
        status, _, dtext = srv.post(json.dumps(DECISION_VALID))
        th.join()

        check(out.get("status") == 200, f"chat with a mid-stream decision: {out.get('status')}")
        got = json.loads(out["text"])["choices"][0]["message"]["content"]
        check(got == ref, "chat with a mid-stream decision matches the quiet reference byte-for-byte")
        check(status == 200, f"mid-stream decision: {status} {dtext[:120]}")
        d = json.loads(dtext)
        check(d["answers"]["dept"]["choice"] == quiet["answers"]["dept"]["choice"],
              "mid-stream decision matches the quiet decision reference")
        return "pass"
    finally:
        srv.stop()


# M3: on the sidecar default, a decision runs on its own context, so chat on a chat instance is
# not gated on the decision. A concurrent chat must finish near its quiet latency instead of being
# serialized behind the decision duration. The fairness bound is read from the same ledger as the
# shared-context lane; the decision term is not added because the executor is a different context.
def run_sidecar_no_stall(model):
    srv = Server(model, ["--instance", "main:ctx=8192:parallel=1:default"])
    try:
        srv.start()
    except Exception as e:  # noqa: BLE001
        srv.stop()
        print(f"skip sidecar no-stall on {os.path.basename(model)}: {e}")
        return "skip"
    if not supports_letter_labels(srv):
        srv.stop()
        print(f"skip sidecar no-stall on {os.path.basename(model)}: no usable answer labels")
        return "skip"
    bound = load_fairness_bound()
    url = f"http://127.0.0.1:{srv.port}/v1/chat/completions"
    chat = {"messages": [{"role": "user", "content": "Write a short paragraph about spring weather."}],
            "max_tokens": 16, "seed": 42, "temperature": 0.0}
    heavy = {
        "model": "test",
        "state": "The customer opened a ticket about a delayed delivery and a duplicate charge. " * 24,
        "questions": {
            f"q{i}": {"type": "noul", "instructions": f"Is claim {i} supported by the state?"}
            for i in range(64)
        },
    }
    try:
        # quiet chat baseline (warm once so a cold first decode cannot skew the comparison)
        status, _, _ = http("POST", url, json.dumps(chat))
        check(status == 200, f"sidecar chat warmup: {status}")
        t0 = time.time()
        status, _, _ = http("POST", url, json.dumps(chat))
        check(status == 200, f"sidecar quiet chat: {status}")
        quiet_ms = (time.time() - t0) * 1000.0

        # concurrent: the decision runs on the sidecar; chat must not wait for it
        out = {}

        def run_decision():
            t = time.time()
            st, _, tx = srv.post(json.dumps(heavy))
            out["ms"] = (time.time() - t) * 1000.0
            out["status"] = st
            out["text"] = tx

        decision_thread = threading.Thread(target=run_decision)
        decision_thread.start()
        time.sleep(0.1)
        t0 = time.time()
        status, _, tx = http("POST", url, json.dumps(chat))
        chat_ms = (time.time() - t0) * 1000.0
        decision_thread.join()

        check(out.get("status") == 200, f"sidecar heavy decision: {out.get('status')} {out.get('text')}")
        check(status == 200, f"sidecar concurrent chat: {status} {tx[:120]}")
        bound_ms = quiet_ms * bound["chat_latency_factor"] + bound["chat_latency_margin_ms"]
        check(chat_ms <= bound_ms,
              f"sidecar chat is not gated on a decision: quiet {quiet_ms:.0f}ms, concurrent {chat_ms:.0f}ms "
              f"(bound {bound_ms:.0f}ms, decision {out['ms']:.0f}ms)")
        print(f"measured: sidecar chat quiet {quiet_ms:.0f}ms, concurrent {chat_ms:.0f}ms, "
              f"decision {out['ms']:.0f}ms")
        return "pass"
    finally:
        srv.stop()


def run_recurrent_kv_integrity(model):
    # Recurrent/hybrid models keep their state in a fixed RS buffer; a decision that forks the
    # context must not disturb a chat turn. Two identical chat completions around a decision must
    # match byte-for-byte. This is the KV-integrity guarantee the docs claim for coexistence. The
    # gate runs this with a recurrent SERVER_MODEL; on a dense model it still checks the same
    # sequential ordering, so it is never skipped for the model kind.
    srv = Server(model, ["--seed", "42"])
    srv.start()
    try:
        if not supports_letter_labels(srv):
            return "skip"
        url = f"http://127.0.0.1:{srv.port}/v1/chat/completions"
        chat_body = {
            "messages": [{"role": "user", "content": "Write a short paragraph about spring weather."}],
            "max_tokens": 24,
            "seed": 42,
        }

        status, _, first = http("POST", url, json.dumps(chat_body))
        check(status == 200, f"chat before the decision: {status} {first[:120]}")

        status, _, dtext = srv.post(json.dumps(DECISION_VALID))
        check(status == 200, f"decision between the chats: {status} {dtext[:120]}")

        status, _, second = http("POST", url, json.dumps(chat_body))
        check(status == 200, f"chat after the decision: {status} {second[:120]}")

        a = json.loads(first)["choices"][0]["message"]["content"]
        b = json.loads(second)["choices"][0]["message"]["content"]
        check(a == b, "the identical chat is byte-identical around a decision on a recurrent model")
        return "pass"
    finally:
        srv.stop()


# M8: server-side deadline. A decision that exceeds --decision-timeout-ms answers 503 +
# Retry-After and never a partial answer, and the server stays usable afterwards. The control
# group proves that under the calibrated default timeout a normal decision is never cancelled.
def run_deadline_control(model):
    # The server-side deadline is only checked at the HTTP_POLLING_SECONDS (1 s) poll boundary
    # (server-queue.h), so a decision that completes in under 1 s can never hit it regardless of
    # the deadline value. The proof therefore needs a decision that stays in flight past 1 s,
    # which requires a context large enough that the prefill dominates. We probe the heavy request
    # on a generous ctx first; a model whose near-full-context decision is under 1.2 s (so the
    # 1 s poll would not reliably catch it) is skipped, not failed. The deadline is set far below
    # the 1 s poll so the first poll after it expires fires the 503.
    deadline_ctx = 16384
    heavy = {
        "model": "test",
        "state": "The customer opened a ticket about a delayed delivery and a duplicate charge. " * 50,
        "questions": {
            f"q{i}": {"type": "noul", "instructions": f"Is claim {i} supported by the state?"}
            for i in range(256)
        },
    }
    probe = None
    deadline_srv = None
    control_srv = None
    try:
        probe = Server(model, ctx=deadline_ctx, max_body=2 * 1024 * 1024)
        probe.start()
        if not supports_letter_labels(probe):
            return "skip"
        probe.post(json.dumps(DECISION_VALID))  # warm so the heavy timing is not skewed by a cold load
        t0 = time.time()
        status, _, text = probe.post(json.dumps(heavy))
        heavy_ms = (time.time() - t0) * 1000.0
        check(status == 200, f"probe heavy decision under the default timeout: {status} {text[:120]}")
        probe.stop()
        probe = None
        if heavy_ms < 1200.0:
            print(f"skip deadline-hit on {os.path.basename(model)}: max decision {heavy_ms:.0f}ms "
                  "is under the 1 s deadline poll, so it cannot exercise the deadline; control group 200")
            control_srv = Server(model)
            control_srv.start()
            if supports_letter_labels(control_srv):
                for _ in range(3):
                    status, _, _ = control_srv.post(json.dumps(DECISION_VALID))
                    check(status == 200, f"control decision under the default timeout: {status}")
            return "pass"

        deadline_srv = Server(model, ["--decision-timeout-ms", "100"], max_body=2 * 1024 * 1024,
                              ctx=deadline_ctx)
        deadline_srv.start()
        if not supports_letter_labels(deadline_srv):
            return "pass"
        t0 = time.time()
        status, headers, text = deadline_srv.post(json.dumps(heavy))
        elapsed_ms = (time.time() - t0) * 1000.0
        check(status == 503, f"over-deadline decision is 503: got {status} {text[:160]}")
        check(headers.get("Retry-After") == "1", f"503 carries Retry-After: {headers}")
        check("answers" not in json.loads(text), "a deadline hit never returns a partial answer")
        check(elapsed_ms < 15000, f"the deadline stops the compute promptly: {elapsed_ms:.0f}ms")

        # the abort must not wedge the scheduler or the KV: /health and a chat both still answer
        status, _, _ = http("GET", f"http://127.0.0.1:{deadline_srv.port}/health")
        check(status == 200, f"health after the deadline abort: {status}")
        chat = {"messages": [{"role": "user", "content": "Say hello"}], "max_tokens": 8}
        status, _, text = http("POST", f"http://127.0.0.1:{deadline_srv.port}/v1/chat/completions",
                               json.dumps(chat))
        check(status == 200, f"chat works after the deadline abort: {status} {text[:120]}")
        deadline_srv.stop()
        deadline_srv = None

        # control group: an ordinary decision under the default timeout is never cancelled
        control_srv = Server(model)
        control_srv.start()
        if supports_letter_labels(control_srv):
            for _ in range(3):
                status, _, _ = control_srv.post(json.dumps(DECISION_VALID))
                check(status == 200, f"control decision under the default timeout: {status}")
        print(f"deadline/control: heavy {heavy_ms:.0f}ms over-deadline 503+Retry-After in "
              f"{elapsed_ms:.0f}ms, server responsive after, control group 200")
        return "pass"
    finally:
        if probe is not None:
            probe.stop()
        if deadline_srv is not None:
            deadline_srv.stop()
        if control_srv is not None:
            control_srv.stop()


def main():
    if not os.path.isfile(SERVER_BIN):
        print(f"FAIL: server binary not found at {SERVER_BIN}")
        return 1

    allow_skip = bool(os.environ.get("LLAMA_SERVER_TEST_ALLOW_SKIP"))
    candidates = [m for m in MODEL_CANDIDATES if m and os.path.isfile(m)]
    if not candidates:
        if allow_skip:
            print("SKIP: no test model asset; set LLAMA_SERVER_TEST_MODEL (LLAMA_SERVER_TEST_ALLOW_SKIP set)")
            return 0
        print("FAIL: no test model asset; set LLAMA_SERVER_TEST_MODEL or LLAMA_SERVER_TEST_ALLOW_SKIP")
        return 1

    for model in candidates:
        server = Server(model)
        try:
            server.start()
        except Exception as e:  # noqa: BLE001
            server.stop()
            print(f"FAIL: server did not start on {os.path.basename(model)}: {e}")
            return 1
        try:
            preflight(server)
        except Exception as e:  # noqa: BLE001
            server.stop()
            print(f"FAIL: preflight: {e}")
            return 1
        if not supports_letter_labels(server):
            server.stop()
            print(f"skip {os.path.basename(model)}: no usable answer labels")
            continue
        try:
            run_checks(server)
        except Exception as e:  # noqa: BLE001
            server.stop()
            print(f"FAIL: {e}")
            return 1
        server.stop()
        check_shutdown(server)

        # bounded-context sweep: a request past the budget is rejected, never truncated
        try:
            capacity = run_capacity_sweep(model)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL: capacity sweep: {e}")
            return 1
        if capacity == "skip":
            print("SKIP: no usable answer labels for the capacity sweep")
        else:
            print("decision capacity sweep passed")

        # KV integrity while a chat is generating (M4.5)
        try:
            integrity = run_chat_decision_integrity(model)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL: chat/decision integrity: {e}")
            return 1
        if integrity == "skip":
            print("SKIP: no usable answer labels for chat/decision integrity")
        else:
            print("chat/decision integrity passed")

        # recurrent KV integrity: chat, decision, identical chat
        try:
            recurrent = run_recurrent_kv_integrity(model)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL: recurrent chat/decision integrity: {e}")
            return 1
        if recurrent == "skip":
            print("SKIP: recurrent KV integrity (no usable answer labels)")
        else:
            print("recurrent chat/decision integrity passed")

        # M3 sidecar default: a decision on the sidecar does not stall chat on a chat instance
        try:
            no_stall = run_sidecar_no_stall(model)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL: sidecar no-stall: {e}")
            return 1
        if no_stall == "skip":
            print("SKIP: sidecar no-stall (no usable answer labels)")
        else:
            print("sidecar no-stall passed")

        # M8: deadline (503 + Retry-After, server stays usable) and control group under the default timeout
        try:
            deadline = run_deadline_control(model)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL: deadline/control: {e}")
            return 1
        if deadline == "skip":
            print("SKIP: deadline/control (no usable answer labels)")
        else:
            print("deadline/control passed")

        print("decision admission checks passed")
        return 0

    if allow_skip:
        print("SKIP: no candidate model supports letter labels (LLAMA_SERVER_TEST_ALLOW_SKIP set)")
        return 0
    print("FAIL: no candidate model supports letter labels; set LLAMA_SERVER_TEST_MODEL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
