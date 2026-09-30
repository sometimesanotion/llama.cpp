#include "server-instances.h"
#include "server-snapshot.h"
#include "server-models.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#undef NDEBUG
#include <cassert>

// per-instance on-disk layout: <slot_save_path>/<model_key>/<instance>/<snapshot>.bin
static void test_layout_paths() {
    const std::string inst = server_snapshot_instance_path("data/slots/", "code", "ledger", "work");
    assert(inst == "data/slots/code/ledger/work.bin");

    // trailing slash on the root is optional
    const std::string inst_noslash = server_snapshot_instance_path("data/slots", "code", "ledger", "work");
    assert(inst_noslash == "data/slots/code/ledger/work.bin");

    const std::string dir = server_snapshot_instance_dir("data/slots/", "code", "ledger");
    assert(dir == "data/slots/code/ledger");

    // legacy flat layout (migration read path)
    const std::string leg = server_snapshot_legacy_path("data/slots/", "code", "work");
    assert(leg == "data/slots/code/work.bin");

    // instances sharing a snapshot name never share a file
    const std::string other = server_snapshot_instance_path("data/slots/", "code", "scratch", "work");
    assert(other == "data/slots/code/scratch/work.bin");
    assert(other != inst);
}

// router aggregate merge: instance rows pass through, snapshots gain their
// owning model, totals sum with 64-bit saturation
static void test_merge() {
    json a = {
        { "instances", json::array({ json{ { "id", "m1:default" } } }) },
        { "snapshots", json::array({ json{ { "name", "s1" } } }) },
        { "total",     json{ { "model", 100 }, { "context", 10 }, { "compute", 5 }, { "total", 115 } } },
    };
    json b = {
        { "instances", json::array({ json{ { "id", "m2:default" } } }) },
        { "snapshots", json::array({ json{ { "name", "s1" } } }) },
        { "total",     json{ { "model", 200 }, { "context", 20 }, { "compute", 7 }, { "total", 227 } } },
    };
    const json out = server_models_merge_instances({ { "m1", a }, { "m2", b } });

    assert(out["instances"].size() == 2);
    assert(out["instances"][0]["id"] == "m1:default");
    assert(out["instances"][1]["id"] == "m2:default");

    // same snapshot name on two children stays distinct via the model tag
    assert(out["snapshots"].size() == 2);
    assert(out["snapshots"][0]["model"] == "m1");
    assert(out["snapshots"][1]["model"] == "m2");

    assert(out["total"]["model"] == 300);
    assert(out["total"]["context"] == 30);
    assert(out["total"]["compute"] == 12);
    assert(out["total"]["total"] == 342);
}

static void test_merge_tolerates_shape_drift() {
    // bare-array envelope (pre-envelope forks) and missing totals contribute
    // rows without breaking the sums
    json bare = json::array({ json{ { "id", "m3:x" } } });
    json nototal = {
        { "instances", json::array() },
        { "snapshots", json::array() },
    };
    const json out = server_models_merge_instances({ { "m3", bare }, { "m4", nototal } });
    assert(out["instances"].size() == 0);  // non-object envelope contributes nothing
    assert(out["snapshots"].size() == 0);
    assert(out["total"]["model"] == 0);
    assert(out["total"]["total"] == 0);
}

static void test_merge_saturates() {    json big = {
        { "instances", json::array() },
        { "snapshots", json::array() },
        { "total",     json{ { "model", UINT64_MAX }, { "context", 0 }, { "compute", 0 }, { "total", UINT64_MAX } } },
    };
    json small = {
        { "instances", json::array() },
        { "snapshots", json::array() },
        { "total",     json{ { "model", 1 }, { "context", 0 }, { "compute", 0 }, { "total", 1 } } },
    };
    const json out = server_models_merge_instances({ { "big", big }, { "small", small } });
    assert(out["total"]["model"] == UINT64_MAX);
    assert(out["total"]["total"] == UINT64_MAX);
}

// merged adapter bytes are additive across children; the merged total sums the
// child totals verbatim (each already adapter-inclusive), never re-adding.
static void test_merge_adapter_additive() {
    json a = {
        { "instances", json::array() },
        { "snapshots", json::array() },
        { "total",     json{ { "model", 100 }, { "context", 10 }, { "compute", 5 }, { "adapter", 20 }, { "total", 135 } } },
    };
    json b = {
        { "instances", json::array() },
        { "snapshots", json::array() },
        { "total",     json{ { "model", 200 }, { "context", 20 }, { "compute", 7 }, { "adapter", 30 }, { "total", 257 } } },
    };
    const json out = server_models_merge_instances({ { "m1", a }, { "m2", b } });
    const json total = out["total"];
    assert(total.is_object() && total.size() == 5);
    assert(total["model"] == 300);
    assert(total["context"] == 30);
    assert(total["compute"] == 12);
    assert(total["adapter"] == 50);
    assert(total["total"] == 392);
    assert(total["total"] == total["model"].get<uint64_t>() + total["context"].get<uint64_t>() +
                             total["compute"].get<uint64_t>() + total["adapter"].get<uint64_t>());
}

// a child without the adapter key (pre-adapter fork) contributes zero adapter
// bytes without breaking the sums.
static void test_merge_adapter_missing_key() {
    json a = {
        { "instances", json::array() },
        { "snapshots", json::array() },
        { "total",     json{ { "model", 100 }, { "context", 10 }, { "compute", 5 }, { "adapter", 20 }, { "total", 135 } } },
    };
    json b = {
        { "instances", json::array() },
        { "snapshots", json::array() },
        { "total",     json{ { "model", 200 }, { "context", 20 }, { "compute", 7 }, { "total", 227 } } },
    };
    const json out = server_models_merge_instances({ { "m1", a }, { "m2", b } });
    assert(out["total"]["adapter"] == 20);
    assert(out["total"]["total"] == 362);
}

// as_u64 edges: negatives, floats, strings and missing keys read as zero;
// the unsigned limit survives the round trip.
static void test_merge_as_u64_edges() {
    json a = {
        { "instances", json::array() },
        { "snapshots", json::array() },
        { "total",     json{ { "model", -5 }, { "context", 1.5 }, { "compute", "x" }, { "adapter", UINT64_MAX }, { "total", 0 } } },
    };
    const json out = server_models_merge_instances({ { "m1", a } });
    assert(out["total"]["model"] == 0);
    assert(out["total"]["context"] == 0);
    assert(out["total"]["compute"] == 0);
    assert(out["total"]["adapter"] == UINT64_MAX);
}

// the fan-out collector: fast children are collected, slow ones skip at the
// deadline, failing fetches skip, and the output sorts by model name no
// matter the completion order. two slow tasks run concurrently (max, not
// sum): a serial implementation needs both sleeps back to back.
static void test_fanout_collect_deadline_skip() {
    server_model_meta slow_a;
    slow_a.name = "a";
    slow_a.port = 2;
    server_model_meta slow_b;
    slow_b.name = "d";
    slow_b.port = 5;
    server_model_meta fast_c;
    fast_c.name = "c";
    fast_c.port = 3;
    server_model_meta fast_b;
    fast_b.name = "b";
    fast_b.port = 4;
    instances_fetch_fn fetch = [](const server_model_meta & meta)
            -> std::optional<std::pair<std::string, json>> {
        if (meta.name == "a" || meta.name == "d") {
            std::this_thread::sleep_for(std::chrono::milliseconds(1500));
            return std::make_pair(meta.name, json{ { "instances", json::array() } });
        }
        if (meta.name == "c") {
            return std::nullopt;
        }
        return std::make_pair(meta.name, json{ { "instances", json::array() } });
    };
    const int64_t t0 = ggml_time_ms();
    auto out = instances_fanout_collect({ slow_a, fast_c, fast_b, slow_b }, fetch, t0 + 400);
    const int64_t wall = ggml_time_ms() - t0;
    // slow pair skipped at the deadline, failing fetch skipped, fast collected
    assert(out.size() == 1);
    assert(out[0].first == "b");
    // concurrent, not serial: both 1500ms sleeps overlap (a serial collector
    // needs 3000ms+ for the slow pair alone, plus abandoned-task joins)
    assert(wall < 2600);

    // deterministic order: names sorted even though submission order was not
    instances_fetch_fn quick = [](const server_model_meta & meta)
            -> std::optional<std::pair<std::string, json>> {
        return std::make_pair(meta.name, json{ { "instances", json::array() } });
    };
    server_model_meta m1;
    m1.name = "zulu";
    server_model_meta m2;
    m2.name = "alpha";
    auto ordered = instances_fanout_collect({ m1, m2 }, quick, ggml_time_ms() + 5000);
    assert(ordered.size() == 2);
    assert(ordered[0].first == "alpha");
    assert(ordered[1].first == "zulu");
}

// golden for the /instances envelope shape: exact key sets and value types as
// served today. later changes must keep this green unless they state a new shape.
static void test_instances_envelope_shape() {
    server_instances mgr;
    mgr.base_name = "tinyllama-2";
    mgr.params.slot_save_path = "";

    auto inst = std::make_shared<server_instance>();
    inst->cfg.name       = "ledger";
    inst->cfg.group      = "ledger";
    inst->cfg.pinned     = true;
    inst->cfg.is_default = true;
    inst->effective.n_ctx      = 512;
    inst->effective.n_parallel = 1;
    inst->built = false;
    mgr.instances.push_back(inst);

    static const std::function<bool()> no_stop = []() { return false; };
    server_http_req req { {}, {}, "/instances", "", "", {}, no_stop };

    auto res = mgr.handle_get_instances(req);
    assert(res->status == 200);

    const json body = json::parse(res->data);
    assert(body.is_object());
    assert(body.size() == 3);
    assert(body.contains("instances"));
    assert(body.contains("snapshots"));
    assert(body.contains("total"));

    assert(body["instances"].is_array());
    assert(body["instances"].size() == 1);
    assert(body["snapshots"].is_array());

    const json total = body["total"];
    assert(total.is_object());
    assert(total.size() == 5);
    assert(total.contains("model") && total["model"].is_number_integer());
    assert(total.contains("context") && total["context"].is_number_integer());
    assert(total.contains("compute") && total["compute"].is_number_integer());
    assert(total.contains("adapter") && total["adapter"].is_number_integer());
    assert(total.contains("total") && total["total"].is_number_integer());

    const json row = body["instances"][0];
    assert(row.is_object());
    const std::vector<std::string> keys = {
        "id", "aliases", "group", "n_ctx", "parallel", "pinned", "is_default", "internal",
        "state", "model_bytes", "context_bytes", "compute_bytes", "adapter_bytes", "total_bytes",
        "vram_bytes", "last_used", "last_used_epoch",
    };
    assert(row.size() == keys.size());
    for (const auto & key : keys) {
        assert(row.contains(key));
    }
    assert(row["id"].is_string());
    assert(row["aliases"].is_array());
    assert(row["group"].is_string());
    assert(row["n_ctx"].is_number());
    assert(row["parallel"].is_number());
    assert(row["pinned"].is_boolean());
    assert(row["is_default"].is_boolean());
    assert(row["internal"].is_boolean());
    assert(row["state"].is_string());
    assert(row["model_bytes"].is_number_integer());
    assert(row["context_bytes"].is_number_integer());
    assert(row["compute_bytes"].is_number_integer());
    assert(row["adapter_bytes"].is_number_integer());
    assert(row["total_bytes"].is_number_integer());
    assert(row["vram_bytes"].is_number_integer());
    assert(row["last_used"].is_number());
    assert(row["last_used_epoch"].is_number());
}

// ordering over routing candidates: fewest busy wins, then least recently used,
// then registration order. full or slot-less members are never picked.
static void test_pick_best_candidate() {
    // busy-first: the idle member wins over a busier one regardless of stamps
    assert(server_instances::pick_best_candidate({
        { 0, 1, 1, 100 },
        { 1, 1, 0, 900 },
    }) == std::optional<size_t>(1));

    // then least-recently-used among equally busy members
    assert(server_instances::pick_best_candidate({
        { 0, 2, 1, 900 },
        { 1, 2, 1, 100 },
    }) == std::optional<size_t>(1));

    // then registration order on a full tie: members are pushed in registration
    // order, and a strict compare keeps the first one seen
    assert(server_instances::pick_best_candidate({
        { 0, 1, 0, 500 },
        { 1, 1, 0, 500 },
    }) == std::optional<size_t>(0));
    assert(server_instances::pick_best_candidate({
        { 1, 1, 0, 500 },
        { 0, 1, 0, 500 },
    }) == std::optional<size_t>(1));

    // all-busy group excludes every member
    assert(!server_instances::pick_best_candidate({
        { 0, 1, 1, 100 },
        { 1, 2, 2, 200 },
    }).has_value());

    // control: a fresh unbuilt member (no slots) never outranks a used one,
    // even when the used member is the only one with a window
    assert(server_instances::pick_best_candidate({
        { 0, 0, 0, -1 },
        { 1, 1, 0, 700 },
    }) == std::optional<size_t>(1));

    // control: a fresh built member (never used) is picked before a used one
    // at equal busyness; the cold member spreads the first load
    assert(server_instances::pick_best_candidate({
        { 0, 1, 0, 700 },
        { 1, 1, 0, -1 },
    }) == std::optional<size_t>(1));
}

// the pure unbuilt cases of the n_ctx display rule: an unbuilt window reports the
// size published in the lock-free display cache (explicit sizes as requested,
// 0 while inheriting until the weights are loaded; the model default comes from a
// live pool, covered by the server golden).
static void test_displayed_n_ctx_unbuilt() {
    server_instances mgr; // weights never loaded
    server_instance  inst;
    inst.built.store(false, std::memory_order_relaxed);
    inst.n_parallel_effective.store(1, std::memory_order_relaxed);

    inst.n_ctx_effective.store(256, std::memory_order_relaxed);
    assert(mgr.displayed_n_ctx(inst) == 256);

    inst.n_ctx_effective.store(0, std::memory_order_relaxed);
    assert(mgr.displayed_n_ctx(inst) == 0);
}

// pool I/O worker lifecycle: no job can be posted to a stopped or
// never-started worker. post returns nullopt there instead of a future
// that would never complete (callers answer the existing retriable 503).
static void test_snapshot_io_post_lifecycle() {
    server_instances mgr;

    // before start: no worker exists, the post refuses fast
    assert(!mgr.snapshot_io_post([]() {}).has_value());

    mgr.start_io_worker();
    // double start is a no-op: still exactly one worker draining the queue
    mgr.start_io_worker();
    auto fut = mgr.snapshot_io_post([]() {});
    assert(fut.has_value());
    fut->wait();
    fut->get();

    mgr.stop_io_worker();
    // after stop: the post refuses again, never a dangling future
    assert(!mgr.snapshot_io_post([]() {}).has_value());

    // double stop is a no-op (and so is the terminate() in the destructor)
    mgr.stop_io_worker();
}

// hashed write keys: identities that sanitize alike never share a directory,
// while the legacy mapping is unchanged so old files stay readable.
static void test_snapshot_model_keys() {
    // collision control group: sanitization twins get distinct write dirs
    const std::string slash = server_snapshot_model_key_hashed("a/b");
    const std::string colon = server_snapshot_model_key_hashed("a:b");
    const std::string plain = server_snapshot_model_key_hashed("a_b");
    assert(slash != colon && slash != plain && colon != plain);

    // each write key extends the legacy mapping with a hash suffix
    assert(slash.substr(0, 4) == "a_b-");
    assert(colon.substr(0, 4) == "a_b-");

    // deterministic across calls
    assert(server_snapshot_model_key_hashed("org/model") == server_snapshot_model_key_hashed("org/model"));

    // the legacy mapping is unchanged (the read fallback for old files)
    assert(server_snapshot_model_key("a/b") == "a_b");
    assert(server_snapshot_model_key("a:b") == "a_b");
    assert(server_snapshot_model_key("tinyllama-2") == "tinyllama-2");
}

// resolve order: hashed-key scoped file wins, then the previous-key scoped
// file, then the legacy flat file; "" when none exists.
static void test_snapshot_resolve_key_fallback() {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path root = fs::temp_directory_path(ec) / "llama-m8-resolve";
    fs::remove_all(root, ec);

    server_instances mgr;
    mgr.base_name = "a/b";  // previous key "a_b", write key "a_b-<hash>"
    mgr.params.slot_save_path = root.string();

    const std::string p_new  = mgr.snapshot_instance_path("ledger", "work");
    const std::string p_prev = mgr.snapshot_instance_path_prev("ledger", "work");
    const std::string p_leg  = mgr.snapshot_legacy_path("work");
    assert(p_new != p_prev);

    const auto touch = [&](const std::string & p) {
        fs::create_directories(fs::path(p).parent_path(), ec);
        std::ofstream out(p, std::ios::binary);
        out.put('x');
    };

    assert(mgr.resolve_snapshot_path("ledger", "work").empty());

    touch(p_leg);
    assert(mgr.resolve_snapshot_path("ledger", "work") == p_leg);

    touch(p_prev);
    assert(mgr.resolve_snapshot_path("ledger", "work") == p_prev);

    touch(p_new);
    assert(mgr.resolve_snapshot_path("ledger", "work") == p_new);

    fs::remove_all(root, ec);
}

// firing rule for the huge-implicit-window guardrail: warns iff the window
// inherits its size (n_ctx == 0) on a large-train model. the null-model case
// never reaches the predicate (the caller returns first).
static void test_huge_implicit_ctx_predicate() {
    // must fire: inherit on a large-train model, including the threshold itself
    assert(server_should_warn_huge_implicit_ctx(0, 32768));
    assert(server_should_warn_huge_implicit_ctx(0, 262144));

    // must not fire: explicit size, however small or large
    assert(!server_should_warn_huge_implicit_ctx(1, 262144));
    assert(!server_should_warn_huge_implicit_ctx(512, 262144));
    assert(!server_should_warn_huge_implicit_ctx(2147483647, 262144));

    // must not fire: inherit on a small-train model, including just below threshold
    assert(!server_should_warn_huge_implicit_ctx(0, 0));
    assert(!server_should_warn_huge_implicit_ctx(0, 4096));
    assert(!server_should_warn_huge_implicit_ctx(0, 32767));
}

// snapshot switch budget: 2 concurrent switches are allowed, the 3rd is
// rejected. the deadline-free rejections are exact (no clock or sleep).
static void test_snapshot_switch_budget() {
    server_instances mgr;

    // within budget: both switches acquire, never a 503
    server_instances::switch_guard first(mgr, INT64_MAX);
    server_instances::switch_guard second(mgr, INT64_MAX);
    assert(first.acquired && second.acquired);

    // over budget with no time left to wait: rejected immediately
    server_instances::switch_guard third(mgr, 0);
    assert(!third.acquired);
}

// snapshot I/O queue budget: max_io_jobs (4) queued writes are accepted, the
// 5th is rejected with the retriable nullopt. a pinned worker job makes the
// drain deterministic (no clock or sleep on the test thread).
static void test_snapshot_io_queue_budget() {
    server_instances mgr;
    mgr.start_io_worker();

    // pin the worker inside one job so nothing drains while the queue fills
    std::promise<void> entered;
    std::atomic<bool>  release{false};
    auto pin = mgr.snapshot_io_post([&]() {
        entered.set_value();
        while (!release.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });
    assert(pin.has_value());
    entered.get_future().wait();

    // four queued posts accepted...
    std::vector<std::future<void>> queued;
    for (int i = 0; i < 4; i++) {
        auto fut = mgr.snapshot_io_post([]() {});
        assert(fut.has_value());
        queued.push_back(std::move(*fut));
    }
    // ...the 5th queued write is rejected
    assert(!mgr.snapshot_io_post([]() {}).has_value());

    release.store(true);
    pin->wait();
    pin->get();
    for (auto & fut : queued) {
        fut.wait();
        fut.get();
    }
    mgr.stop_io_worker();
}

// test-only helper: capture an envelope as a canonical string so later
// refactors compare byte-for-byte.
static std::string instances_canonical(const json & envelope) {
    return envelope.dump();
}

// canonical capture is stable: two reads of one pool produce identical bytes.
static void test_envelope_canonical_stable() {
    server_instances mgr;
    mgr.base_name = "tinyllama-2";
    mgr.params.slot_save_path = "";

    auto inst = std::make_shared<server_instance>();
    inst->cfg.name       = "ledger";
    inst->cfg.group      = "ledger";
    inst->cfg.pinned     = true;
    inst->cfg.is_default = true;
    inst->effective.n_ctx      = 512;
    inst->effective.n_parallel = 1;
    inst->built = false;
    mgr.instances.push_back(inst);

    static const std::function<bool()> no_stop = []() { return false; };
    server_http_req req { {}, {}, "/instances", "", "", {}, no_stop };

    auto first  = mgr.handle_get_instances(req);
    auto second = mgr.handle_get_instances(req);
    assert(first->status == 200 && second->status == 200);
    assert(instances_canonical(json::parse(first->data)) == instances_canonical(json::parse(second->data)));

    const json body = json::parse(first->data);
    assert(body.is_object() && body.size() == 3);
}

// merge contract: instance rows pass through untouched, snapshot rows gain
// their owning model and keep adapter_fp, totals sum the five current keys.
static void test_merge_exact_shape() {
    const json row_a = {
        { "id", "m1:default" }, { "adapter_bytes", 16 }, { "state", "loaded" },
    };
    const json snap_a = {
        { "name", "s1" }, { "size", 10 }, { "mtime", 7 },
        { "n_ctx_seq", 64 }, { "adapter_fp", "abcdef" }, { "instance", "ledger" },
    };
    json a = {
        { "instances", json::array({ row_a }) },
        { "snapshots", json::array({ snap_a }) },
        { "total",     json{ { "model", 100 }, { "context", 10 }, { "compute", 5 }, { "adapter", 16 }, { "total", 131 } } },
    };
    json b = {
        { "instances", json::array({ json{ { "id", "m2:default" } } }) },
        { "snapshots", json::array({ json{ { "name", "s1" } } }) },
        { "total",     json{ { "model", 200 }, { "context", 20 }, { "compute", 7 }, { "total", 227 } } },
    };
    const json out = server_models_merge_instances({ { "m1", a }, { "m2", b } });

    // instance rows are untouched copies
    assert(out["instances"].size() == 2);
    assert(out["instances"][0] == row_a);
    assert(out["instances"][1]["id"] == "m2:default");

    // snapshot rows gain the owning model, keep every other field
    assert(out["snapshots"].size() == 2);
    assert(out["snapshots"][0]["model"] == "m1");
    assert(out["snapshots"][0]["adapter_fp"] == "abcdef");
    assert(out["snapshots"][0]["instance"] == "ledger");
    assert(out["snapshots"][1]["model"] == "m2");

    // totals carry exactly the five current keys and sum additively; a child
    // without the adapter key contributes zero adapter bytes
    const json total = out["total"];
    assert(total.is_object() && total.size() == 5);
    assert(total["model"] == 300);
    assert(total["context"] == 30);
    assert(total["compute"] == 12);
    assert(total["adapter"] == 16);
    assert(total["total"] == 358);
}

// snapshot row shape in the live envelope: one file on disk surfaces one row
// with exactly {name, size, mtime, n_ctx_seq, adapter_fp, instance}.
static void test_envelope_snapshot_row_shape() {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path root = fs::temp_directory_path(ec) / "llama-m0-envelope-snap";
    fs::remove_all(root, ec);

    server_instances mgr;
    mgr.base_name = "m";
    mgr.params.slot_save_path = root.string() + "/";

    auto inst = std::make_shared<server_instance>();
    inst->cfg.name       = "ledger";
    inst->cfg.group      = "ledger";
    inst->effective.n_ctx      = 512;
    inst->effective.n_parallel = 1;
    inst->built = false;
    mgr.instances.push_back(inst);

    server_snapshot_data data;
    data.n_ctx_seq  = 64;
    data.adapter_fp = "abcdef";
    data.tokens     = { 1, 2, 3 };
    data.kv         = { 4, 5, 6 };
    const std::string path = mgr.snapshot_instance_path("ledger", "work");
    fs::create_directories(fs::path(path).parent_path(), ec);
    assert(server_snapshot_write(path, data));

    static const std::function<bool()> no_stop = []() { return false; };
    server_http_req req { {}, {}, "/instances", "", "", {}, no_stop };
    auto res = mgr.handle_get_instances(req);
    assert(res->status == 200);

    const json body = json::parse(res->data);
    assert(body["snapshots"].is_array() && body["snapshots"].size() == 1);
    const json row = body["snapshots"][0];
    assert(row.is_object() && row.size() == 7);
    assert(row["name"] == "work");
    assert(row["n_ctx_seq"] == 64);
    assert(row["adapter_fp"] == "abcdef");
    assert(row["version"] == 2);
    assert(row["instance"] == "ledger");
    assert(row["size"].is_number_integer());
    assert(row["mtime"].is_number_integer());

    fs::remove_all(root, ec);
}

// the internal decision sidecar executor: reserved name, internal marker in the envelope,
// undeletable, unresizable, not createable, and VRAM reported like any other window.
static void test_sidecar_registration_and_guards() {
    server_instances mgr;
    mgr.base_name = "m";
    mgr.params.slot_save_path = "";

    auto sidecar = std::make_shared<server_instance>();
    sidecar->cfg.name  = server_instances::decision_sidecar_name();
    sidecar->cfg.group = server_instances::decision_sidecar_name();
    sidecar->internal  = true;
    sidecar->effective.n_ctx      = 512;
    sidecar->effective.n_parallel = 1;
    mgr.instances.push_back(sidecar);

    static const std::function<bool()> no_stop = []() { return false; };

    // the envelope reports the sidecar with its internal marker; unbuilt it owns zero bytes
    server_http_req inst_req { {}, {}, "/instances", "", "", {}, no_stop };
    auto res = mgr.handle_get_instances(inst_req);
    assert(res->status == 200);
    const json body = json::parse(res->data);
    assert(body["instances"].is_array() && body["instances"].size() == 1);
    const json row = body["instances"][0];
    assert(row["internal"] == true);
    assert(row["id"] == "m:__decision__");
    assert(row.contains("context_bytes") && row.contains("compute_bytes") && row.contains("vram_bytes"));
    // VRAM reporting follows the shared rule: context + compute, like every other window
    assert(row["vram_bytes"] == row["context_bytes"].get<uint64_t>() + row["compute_bytes"].get<uint64_t>());
    assert(row["context_bytes"] == 0 && row["compute_bytes"] == 0);

    // the executor is manager-owned: delete, resize and a same-name create are all refused
    server_http_req del_req { {}, {}, "/instances/__decision__", "", "", {}, no_stop };
    del_req.params["name"] = server_instances::decision_sidecar_name();
    assert(mgr.handle_delete_instance(del_req)->status == 400);

    server_http_req resize_req { {}, {}, "/instances/__decision__/resize", "", "", {}, no_stop };
    resize_req.params["name"] = server_instances::decision_sidecar_name();
    resize_req.body           = "{\"ctx_size\": 1024}";
    assert(mgr.handle_post_instance_resize(resize_req)->status == 400);

    server_http_req create_req { {}, {}, "/instances", "", "", {}, no_stop };
    create_req.body = "{\"name\": \"__decision__\", \"group\": \"g\"}";
    assert(mgr.handle_post_instances(create_req)->status == 409);
}

// concurrent first demands for one window collapse onto exactly one build:
// ensure_built_instance serializes on mutex_mgmt, so 8 racing /props calls share
// one context allocation. needs a real model to allocate a window; skips without one.
static void test_sidecar_build_once_concurrent(const std::string & model_path) {
    if (model_path.empty()) {
        printf("test-sidecar-build-once: no model file; skip (set LLAMA_DECISION_TEST_MODEL or pass -m)\n");
        return;
    }
    common_params params;
    params.model.path = model_path;
    params.n_ctx      = 256;
    params.n_parallel = 1;
    params.warmup     = false;
    params.cpuparams.n_threads       = 4; // hermetic: -1 segfaults in some container toolchains
    params.cpuparams_batch.n_threads = 4;
    params.n_seq_decision   = 8;
    params.decision_sidecar = true;

    common_instance chat;
    chat.name       = "chat";
    chat.group      = "chat";
    chat.ctx_size   = 256;
    chat.parallel   = 1;
    chat.is_default = true;
    params.instances.push_back(chat);

    int builds = 0;
    server_instances mgr;
    mgr.set_context_builder([&builds, &mgr](server_instance & inst) {
        ++builds;
        return mgr.build_context_default(inst);
    });
    assert(mgr.load(params));

    // the sidecar is registered lazily: reserved name, internal, unbuilt, parallel 1
    assert(mgr.instances.size() == 2);
    const auto & sidecar = mgr.instances[1];
    assert(sidecar->cfg.name == server_instances::decision_sidecar_name());
    assert(sidecar->internal);
    assert(!sidecar->built.load(std::memory_order_acquire));
    assert(sidecar->effective.n_parallel == 1);

    // 8 racing first requests on the default window collapse to one build
    static const std::function<bool()> no_stop = []() { return false; };
    server_http_req props_req { {}, {}, "/props", "", "", {}, no_stop };
    std::atomic<int> ok{0};
    std::vector<std::thread> threads;
    for (int i = 0; i < 8; ++i) {
        threads.emplace_back([&]() {
            if (mgr.handle_get_props(props_req)->status == 200) {
                ok.fetch_add(1);
            }
        });
    }
    for (auto & th : threads) {
        th.join();
    }
    assert(ok.load() == 8);
    assert(builds == 1);
    assert(mgr.instances[0]->built.load(std::memory_order_acquire));

    mgr.terminate();
}

int main(int argc, char ** argv) {
    std::string model_path;
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == "-m") {
            model_path = argv[i + 1];
        }
    }
    if (model_path.empty() && std::getenv("LLAMA_DECISION_TEST_MODEL") != nullptr) {
        model_path = std::getenv("LLAMA_DECISION_TEST_MODEL");
    }

    test_layout_paths();
    test_merge();
    test_merge_tolerates_shape_drift();
    test_merge_saturates();
    test_merge_adapter_additive();
    test_merge_adapter_missing_key();
    test_merge_as_u64_edges();
    test_fanout_collect_deadline_skip();
    test_instances_envelope_shape();
    test_pick_best_candidate();
    test_displayed_n_ctx_unbuilt();
    test_snapshot_io_post_lifecycle();
    test_snapshot_model_keys();
    test_snapshot_resolve_key_fallback();
    test_huge_implicit_ctx_predicate();
    test_snapshot_switch_budget();
    test_snapshot_io_queue_budget();
    test_envelope_canonical_stable();
    test_merge_exact_shape();
    test_envelope_snapshot_row_shape();
    test_sidecar_registration_and_guards();
    test_sidecar_build_once_concurrent(model_path);
    return 0;
}
