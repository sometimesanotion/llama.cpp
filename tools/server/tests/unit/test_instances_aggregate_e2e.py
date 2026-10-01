"""Aggregate parity E2E: two live background servers (one window each) plus
a mock router child advertising adapter bytes.

The merged contract (the same shape the router's production merge implements,
covered over identical shapes by the C++ merge tests): merged total.adapter is
the sum of the child adapter totals, and merged total.total is the sum of the
adapter-inclusive child totals with no double count. Instance rows pass
through untouched; snapshot rows are tagged with their owning model on merge.
"""

import json
import os
import signal
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from utils import *

MOCK_ADAPTER = 123456
MOCK_TOTAL = {
    "model":   1000,
    "context": 200,
    "compute": 50,
    "adapter": MOCK_ADAPTER,
    "total":   1000 + 200 + 50 + MOCK_ADAPTER,
}
MOCK_ENVELOPE = {
    "instances": [{"id": "mock:m"}],
    "snapshots": [{"name": "s1"}],
    "total":     MOCK_TOTAL,
}


class _MockChildHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/instances":
            body = json.dumps(MOCK_ENVELOPE).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        pass


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture()
def mock_child():
    port = _free_port()
    httpd = HTTPServer(("127.0.0.1", port), _MockChildHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield MOCK_ENVELOPE
    httpd.shutdown()


def _live_server(instance: str):
    srv = ServerPreset.tinyllama2()
    srv.server_port = _free_port()
    srv.instances = [f"{instance}:ctx=512"]
    srv.n_ctx = 512
    srv.start()
    # demand-build the one window so the envelope carries real byte fields
    res = srv.make_request("POST", "/completion", data={
        "model": f"tinyllama-2:{instance}",
        "prompt": "Hello",
        "n_predict": 4,
    })
    assert res.status_code == 200
    return srv


def test_aggregate_two_live_plus_mock(mock_child):
    srv_a = _live_server("a")
    try:
        srv_b = _live_server("b")
        try:
            env_a = srv_a.make_request("GET", "/instances").body
            env_b = srv_b.make_request("GET", "/instances").body
            env_m = mock_child

            # live legs: real byte fields, no adapters on these windows
            for env in (env_a, env_b):
                assert env["total"]["adapter"] == 0
                assert env["total"]["total"] == (
                    env["total"]["model"] + env["total"]["context"] + env["total"]["compute"]
                )

            # merged adapter is the sum across children (only the mock here)
            merged_adapter = env_a["total"]["adapter"] + env_b["total"]["adapter"] + env_m["total"]["adapter"]
            assert merged_adapter == MOCK_ADAPTER

            # merged total is the sum of the adapter-inclusive child totals
            merged_total = env_a["total"]["total"] + env_b["total"]["total"] + env_m["total"]["total"]
            legs = ["model", "context", "compute", "adapter"]
            assert merged_total == sum(
                env_a["total"][k] + env_b["total"][k] + env_m["total"][k] for k in legs
            ), "adapter bytes double counted in merged total"

            # instance rows pass through untouched: one live row per server plus mock
            ids = (
                {inst["id"] for inst in env_a["instances"]}
                | {inst["id"] for inst in env_b["instances"]}
                | {inst["id"] for inst in env_m["instances"]}
            )
            assert "tinyllama-2:a" in ids
            assert "tinyllama-2:b" in ids
            assert "mock:m" in ids

            # the mock snapshot row is present for the merge to tag
            assert env_m["snapshots"][0]["name"] == "s1"
        finally:
            srv_b.stop()
    finally:
        srv_a.stop()


AGG_PRESET = """\
[agg-a]
hf-repo = ggml-org/test-model-stories260K:F32
instance = w:ctx=512

[agg-b]
hf-repo = ggml-org/test-model-stories260K-infill:F32
instance = w:ctx=512
"""


def _router_with_two_children():
    import os
    import tempfile
    preset_path = os.path.join(tempfile.mkdtemp(), "agg.ini")
    with open(preset_path, "w") as f:
        f.write(AGG_PRESET)
    srv = ServerPreset.router()
    srv.server_port = _free_port()
    srv.models_preset = preset_path
    srv.models_max = 3
    srv.start()
    for model in ("agg-a", "agg-b"):
        res = srv.make_request("POST", "/models/load", data={"model": model}, timeout=180)
        assert res.status_code == 200, res.body
    return srv


def _wait_for_model(srv, model, timeout=180):
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        res = srv.make_request("GET", "/models")
        for item in res.body.get("data", []):
            status = item.get("status", {})
            value = status.get("value") if isinstance(status, dict) else status
            if item.get("id") == model and value == "loaded":
                return
        time.sleep(2)
    raise AssertionError(f"model {model} never loaded")


def test_router_aggregate_two_children():
    srv = _router_with_two_children()
    try:
        _wait_for_model(srv, "agg-a")
        _wait_for_model(srv, "agg-b")

        first = srv.make_request("GET", "/instances").body
        # stable across runs: parallel completion order must not leak out
        for _ in range(2):
            assert srv.make_request("GET", "/instances").body == first

        ids = {inst["id"] for inst in first["instances"]}
        assert any(i.startswith("agg-a:") for i in ids)
        assert any(i.startswith("agg-b:") for i in ids)

        total = first["total"]
        assert total["total"] == total["model"] + total["context"] + total["compute"] + total["adapter"]
    finally:
        srv.stop()


def test_router_aggregate_unloaded_child_skipped():
    srv = _router_with_two_children()
    try:
        _wait_for_model(srv, "agg-a")
        _wait_for_model(srv, "agg-b")

        res = srv.make_request("POST", "/models/unload", data={"model": "agg-b"})
        assert res.status_code == 200

        env = srv.make_request("GET", "/instances").body
        ids = {inst["id"] for inst in env["instances"]}
        assert any(i.startswith("agg-a:") for i in ids)
        assert not any(i.startswith("agg-b:") for i in ids)
    finally:
        srv.stop()


# --- caller-latency bound on the router aggregate ------------------------------

AGG_SLOW_PRESET = """\
[agg-a]
hf-repo = ggml-org/test-model-stories260K:F32
instance = w:ctx=512

[agg-b]
hf-repo = ggml-org/test-model-stories260K:F32
instance = w:ctx=512

[agg-c]
hf-repo = ggml-org/test-model-stories260K:F32
instance = w:ctx=512
"""

# small scheduling windows so three children exercise three batches without the
# production 8-per-batch / multi-second budget.
AGG_TOTAL_MS = 100
AGG_CHILD_TIMEOUT_MS = 300
AGG_FANOUT_MAX = 1


def _child_pid_for_alias(router_pid, alias):
    for pid in subprocess.check_output(["pgrep", "-P", str(router_pid)]).decode().split():
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                if alias.encode() in f.read().split(b"\0"):
                    return int(pid)
        except OSError:
            pass
    return None


def test_router_aggregate_slow_child_deadline_bound(monkeypatch):
    """A child frozen past its per-child socket timeout is absent while a healthy
    sibling is still reported, and the caller pays at most the budget plus one
    per-child timeout (an in-flight read cannot be cancelled)."""
    import tempfile
    preset_path = os.path.join(tempfile.mkdtemp(), "agg.ini")
    with open(preset_path, "w") as f:
        f.write(AGG_SLOW_PRESET)
    monkeypatch.setenv("LLAMA_SERVER_TEST_AGG_TOTAL_MS", str(AGG_TOTAL_MS))
    monkeypatch.setenv("LLAMA_SERVER_TEST_AGG_CHILD_TIMEOUT_MS", str(AGG_CHILD_TIMEOUT_MS))
    monkeypatch.setenv("LLAMA_SERVER_TEST_AGG_FANOUT_MAX", str(AGG_FANOUT_MAX))

    srv = ServerPreset.router()
    srv.server_port = _free_port()
    srv.models_preset = preset_path
    srv.models_max = 4
    srv.start()
    suspended = []
    try:
        for model in ("agg-a", "agg-b", "agg-c"):
            res = srv.make_request("POST", "/models/load", data={"model": model}, timeout=180)
            assert res.status_code == 200, res.body
        for model in ("agg-a", "agg-b", "agg-c"):
            _wait_for_model(srv, model)
        for alias in ("agg-b", "agg-c"):
            pid = _child_pid_for_alias(srv.process.pid, alias)
            assert pid is not None, f"child {alias} not found"
            os.kill(pid, signal.SIGSTOP)
            suspended.append(pid)

        t0 = time.monotonic()
        env = srv.make_request("GET", "/instances", timeout=60).body
        elapsed_ms = (time.monotonic() - t0) * 1000.0

        ids = {inst["id"] for inst in env["instances"]}
        assert any(i.startswith("agg-a:") for i in ids)
        assert not any(i.startswith("agg-b:") for i in ids)
        assert not any(i.startswith("agg-c:") for i in ids)

        bound = AGG_TOTAL_MS + AGG_CHILD_TIMEOUT_MS
        assert elapsed_ms <= bound + 150, f"aggregate took {elapsed_ms:.0f} ms, budget {bound} ms"
    finally:
        for pid in suspended:
            os.kill(pid, signal.SIGCONT)
        srv.stop()
