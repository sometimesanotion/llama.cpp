#include "common.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"
#include "server-instances.h"
#include "server-models.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <optional>
#include <string>
#include <thread>
#include <vector>

// capture the build mode BEFORE NDEBUG is undefined for assert() below: the
// library under test is compiled in the same configuration, so this picks the
// release-only cold-reload branch (debug builds must abort, so their body is compiled out)
#ifdef NDEBUG
#define TEST_LIBRARY_NDEBUG 1
#endif

#undef NDEBUG
#include <cassert>

#include <cstdio>
#include <cstdlib>

static void test_instances_parse_round_trip() {
    const std::vector<common_instance> instances = common_instances_parse(
        "swarm0:group=swarm:ctx=16384:parallel=1,"
        "swarm1:group=swarm:ctx=16384,"
        "ledger:ctx=65536:pinned:default,"
        "scratch:ctx=65536:pinned");

    assert(instances.size() == 4);

    const common_instance & swarm0 = instances[0];
    assert(swarm0.name == "swarm0");
    assert(swarm0.group == "swarm");
    assert(swarm0.ctx_size == 16384);
    assert(swarm0.parallel == 1);
    assert(!swarm0.pinned);
    assert(!swarm0.is_default);

    const common_instance & swarm1 = instances[1];
    assert(swarm1.name == "swarm1");
    assert(swarm1.group == "swarm");
    assert(swarm1.ctx_size == 16384);
    assert(swarm1.parallel == 0);
    assert(!swarm1.pinned);
    assert(!swarm1.is_default);

    const common_instance & ledger = instances[2];
    assert(ledger.name == "ledger");
    assert(ledger.group == "ledger"); // default group == name
    assert(ledger.ctx_size == 65536);
    assert(ledger.pinned);
    assert(ledger.is_default);

    const common_instance & scratch = instances[3];
    assert(scratch.name == "scratch");
    assert(scratch.group == "scratch");
    assert(scratch.ctx_size == 65536);
    assert(scratch.pinned);
    assert(!scratch.is_default);

    // round-trip: to_string then parse again must yield identical instances
    const std::string s = common_instances_to_string(instances);
    const auto reparsed = common_instances_parse(s);
    assert(reparsed.size() == instances.size());
    for (size_t i = 0; i < instances.size(); ++i) {
        assert(reparsed[i].name == instances[i].name);
        assert(reparsed[i].group == instances[i].group);
        assert(reparsed[i].ctx_size == instances[i].ctx_size);
        assert(reparsed[i].parallel == instances[i].parallel);
        assert(reparsed[i].is_default == instances[i].is_default);
        assert(reparsed[i].pinned == instances[i].pinned);
    }

    // empty spec yields no instances
    assert(common_instances_parse("").empty());
}

static void expect_parse_error(const std::string & spec) {
    bool threw = false;
    try {
        common_instances_parse(spec);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    if (!threw) {
        fprintf(stderr, "expected parse error for '%s'\n", spec.c_str());
        assert(false);
    }
}

static void test_instances_parse_errors() {
    expect_parse_error(":ctx=8");                // empty name
    expect_parse_error("foo:ctx=abc");           // non-integer ctx
    expect_parse_error("foo:ctx=-5");            // negative ctx
    expect_parse_error("foo:parallel=-1");       // negative parallel
    expect_parse_error("foo:unknown");           // unknown option
    expect_parse_error("foo::pinned");           // empty option
    expect_parse_error("foo:pinned=x");          // pinned takes no value
    expect_parse_error("foo:sleep=0");           // sleep is not supported in this branch
    expect_parse_error("foo,foo");               // duplicate instance name
    expect_parse_error("foo:group=bar,bar");     // group collides with instance name
    // name/group character class: rejected by common_instance_validate
    expect_parse_error("a=b:ctx=8");             // '=' in name
    expect_parse_error("a/b:ctx=8");             // '/' in name
    expect_parse_error("a b:ctx=8");             // space in name
    expect_parse_error("a:b:ctx=8");             // ':' in name (also a grammar separator)
    expect_parse_error("foo:group=a/b");         // '/' in group
    expect_parse_error("foo:group=a=b");         // '=' in group
    // 'latest' is a reserved routing token, never a legal name or group
    expect_parse_error("latest");                // reserved name
    expect_parse_error("latest:group=g");        // reserved name with a group
    expect_parse_error("foo:group=latest");      // reserved group
}

static void expect_parse_error_like(const std::string & spec, const std::string & needle) {
    try {
        common_instances_parse(spec);
    } catch (const std::invalid_argument & e) {
        if (std::string(e.what()).find(needle) == std::string::npos) {
            fprintf(stderr, "error for '%s' did not mention '%s': %s\n", spec.c_str(), needle.c_str(), e.what());
            assert(false);
        }
        return;
    }
    fprintf(stderr, "expected parse error for '%s'\n", spec.c_str());
    assert(false);
}

// numeric options reject trailing garbage with a full-consume parse;
// non-finite scales name the scale; whitespace around '=' stays leniency.
static void test_instance_numerics_strict() {
    // trailing garbage is rejected, not silently accepted
    expect_parse_error("a:ctx=8192xyz");
    expect_parse_error("a:parallel=2abc");
    expect_parse_error("a:ctx=0x10");
    expect_parse_error("a:ctx=12.5");
    expect_parse_error("a:ctx=");
    expect_parse_error("a:ctx=9999999999999999999999");
    // control group: well-formed numerics keep parsing
    const auto insts = common_instances_parse("a:ctx=8192:parallel=4");
    assert(insts.size() == 1);
    assert(insts[0].ctx_size == 8192);
    assert(insts[0].parallel == 4);
    // scale lookahead keeps working: 'pinned' is not a scale
    const auto lora = common_instances_parse("a:lora=./x.gguf:pinned");
    assert(lora.size() == 1);
    assert(lora[0].lora.size() == 1);
    assert(lora[0].lora[0].second == 1.0f);
    assert(lora[0].pinned);
    // non-finite scales name the scale, not the option
    expect_parse_error_like("a:lora=./x.gguf:nan", "lora scale");
    expect_parse_error_like("a:lora=./x.gguf:inf", "lora scale");
    expect_parse_error_like("a:lora=./x.gguf:-inf", "lora scale");
    // whitespace around '=' is leniency, never an unknown option
    const auto spaced = common_instances_parse("name:group =G");
    assert(spaced.size() == 1);
    assert(spaced[0].group == "G");
}

// duplicate names and name/group collisions are validated over the whole
// accumulated list, not just within one spec: repeated flags concatenate
// parses, so the check must run on the merged result.
static void test_instances_validate_all_cross_spec() {
    // duplicate names spread across two specs throw
    {
        std::vector<common_instance> merged;
        for (const auto & inst : common_instances_parse("a")) {
            merged.push_back(inst);
        }
        for (const auto & inst : common_instances_parse("a")) {
            merged.push_back(inst);
        }
        bool threw = false;
        try {
            common_instances_validate_all(merged);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        assert(threw);
    }
    // a group in one spec colliding with a name in another throws
    {
        std::vector<common_instance> merged;
        for (const auto & inst : common_instances_parse("x:group=y")) {
            merged.push_back(inst);
        }
        for (const auto & inst : common_instances_parse("y")) {
            merged.push_back(inst);
        }
        bool threw = false;
        try {
            common_instances_validate_all(merged);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        assert(threw);
    }
    // control group: distinct names and groups validate clean
    {
        std::vector<common_instance> merged;
        for (const auto & inst : common_instances_parse("a")) {
            merged.push_back(inst);
        }
        for (const auto & inst : common_instances_parse("b:group=g")) {
            merged.push_back(inst);
        }
        common_instances_validate_all(merged);
        common_instances_validate_all({});
    }
}

// the collision report names the actually-colliding pair in both orders:
// {group=x} vs {name=y} and {name=x} vs {group=x} each name x twice.
static void test_instances_collision_message_names_pair() {
    {
        std::vector<common_instance> merged;
        for (const auto & inst : common_instances_parse("a:group=y")) {
            merged.push_back(inst);
        }
        for (const auto & inst : common_instances_parse("y")) {
            merged.push_back(inst);
        }
        try {
            common_instances_validate_all(merged);
            assert(false);
        } catch (const std::invalid_argument & e) {
            const std::string msg = e.what();
            assert(msg.find("'y'") != std::string::npos);
        }
    }
    {
        std::vector<common_instance> merged;
        for (const auto & inst : common_instances_parse("a:group=x")) {
            merged.push_back(inst);
        }
        for (const auto & inst : common_instances_parse("b:group=a")) {
            merged.push_back(inst);
        }
        try {
            common_instances_validate_all(merged);
            assert(false);
        } catch (const std::invalid_argument & e) {
            // the colliding pair is group 'a' vs name 'a'; 'x' and 'b' are
            // bystanders that must not appear
            const std::string msg = e.what();
            assert(msg.find("'a'") != std::string::npos);
            assert(msg.find("'x'") == std::string::npos);
            assert(msg.find("'b'") == std::string::npos);
        }
    }
}

// adapter scales survive a grammar round trip bit-exactly: the text form
// must carry full float precision, matching the raw-bits fingerprint input.
static void test_instance_lora_scale_round_trip() {
    for (const char * text : { "0.1", "0.5", "0.12345679", "0.001", "2" }) {
        const auto insts = common_instances_parse(std::string("a:lora=./x.gguf:") + text);
        assert(insts.size() == 1);
        assert(insts[0].lora.size() == 1);
        const float scale = insts[0].lora[0].second;
        const auto back = common_instances_parse(common_instances_to_string(insts));
        assert(back.size() == 1);
        assert(back[0].lora.size() == 1);
        assert(back[0].lora[0].second == scale);
    }
    // the default scale prints bare (no ':1' suffix) and re-parses to 1.0
    const auto bare = common_instances_parse("a:lora=./x.gguf");
    assert(common_instances_to_string(bare) == "a:lora=./x.gguf");
}

// the explicit instance field overrides any instance/group component of the
// model id, including the absent component of a bare base id and the
// reserved latest pin. the pool is hand-built (no weights) because name
// resolution only reads identity config.
static void test_resolve_honors_explicit_instance() {
    server_instances mgr;
    mgr.base_name = "tinyllama-2";
    for (const char * name : { "a", "b", "c" }) {
        auto inst = std::make_shared<server_instance>();
        inst->cfg.name  = name;
        inst->cfg.group = (std::string(name) == "c") ? "c" : "g";
        if (std::string(name) == "b") {
            inst->cfg.is_default = true;
        }
        mgr.instances.push_back(inst);
    }
    std::string error;
    const auto inst_name = [&](const server_instances::resolve_target & t) {
        assert(t.kind == server_instances::target_kind::INSTANCE);
        assert(t.inst);
        return t.inst->cfg.name;
    };

    // bare base plus an explicit instance resolves to that instance, not default
    assert(inst_name(mgr.resolve("tinyllama-2", "a", error)) == "a");
    // an explicit group routes by group policy
    {
        const auto t = mgr.resolve("tinyllama-2", "g", error);
        assert(t.kind == server_instances::target_kind::GROUP);
        assert(t.group == "g");
    }
    // latest plus an explicit instance resolves to that instance, not default
    assert(inst_name(mgr.resolve("tinyllama-2:latest", "a", error)) == "a");
    // control group: no override keeps the long-standing behavior
    assert(inst_name(mgr.resolve("tinyllama-2", "", error)) == "b");
    assert(inst_name(mgr.resolve("tinyllama-2:latest", "", error)) == "b");
    assert(inst_name(mgr.resolve("tinyllama-2:a", "b", error)) == "b");
    assert(inst_name(mgr.resolve("tinyllama-2:latest:b", "a", error)) == "a");
    assert(inst_name(mgr.resolve("", "a", error)) == "a");
    // unknown override still misses, on the generation-routing tier
    assert(mgr.resolve("tinyllama-2", "nope", error).kind == server_instances::target_kind::NONE);
    assert(!error.empty());
    // a foreign pool name is rejected even when an override is present: the
    // base guard runs before any override is considered
    assert(mgr.resolve("other-pool", "a", error).kind == server_instances::target_kind::NONE);

    mgr.terminate();
}

static void test_instances_parse_valid_names() {    // control group: previously valid names stay valid, including near-misses
    // of the reserved token (the reservation is the exact string "latest")
    for (const char * spec : {
            "a",
            "swarm0:group=swarm:ctx=512",
            "ledger:ctx=512:pinned:default",
            "Latest",
            "latest1",
            "my-latest-thing",
        }) {
        const auto insts = common_instances_parse(spec);
        assert(insts.size() == 1);
    }
}

static void test_instance_params() {
    common_params base;
    base.n_ctx = 1024;
    base.n_parallel = 4;

    common_instance inst;
    inst.name = "worker";
    inst.group = "swarm";

    // no overrides: inherit ctx, but parallel defaults to 1 (never the base value)
    common_params p = common_instance_params(base, inst);
    assert(p.n_ctx == 1024);
    assert(p.n_parallel == 1);

    // explicit overrides
    inst.ctx_size = 8192;
    inst.parallel = 2;
    p = common_instance_params(base, inst);
    assert(p.n_ctx == 8192);
    assert(p.n_parallel == 2);

    // zero ctx_size keeps the base value; zero parallel keeps 1
    common_params base2;
    base2.n_ctx = 2048;
    base2.n_parallel = 8;
    common_instance empty_inst;
    empty_inst.name = "bare";
    p = common_instance_params(base2, empty_inst);
    assert(p.n_ctx == 2048);
    assert(p.n_parallel == 1); // divergence from _swarm_api: never inherits base.n_parallel
}

static void test_instances_lora_grammar() {    // repeatable lora= with explicit and default scales
    auto v = common_instances_parse("a:lora=./x.gguf:0.5:lora=./y.gguf,b:lora=./z.gguf");
    assert(v.size() == 2);
    assert(v[0].lora.size() == 2);
    assert(v[0].lora[0].first == "./x.gguf" && v[0].lora[0].second == 0.5f);
    assert(v[0].lora[1].first == "./y.gguf" && v[0].lora[1].second == 1.0f);
    assert(v[1].lora.size() == 1);
    assert(v[1].lora[0].first == "./z.gguf" && v[1].lora[0].second == 1.0f);

    // a non-float component after lora= is not a scale: default applies, comp parses on
    auto w = common_instances_parse("a:lora=./x.gguf:pinned");
    assert(w.size() == 1 && w[0].lora.size() == 1);
    assert(w[0].lora[0].second == 1.0f && w[0].pinned);

    // no lora= means an empty list (inherit the base --lora set)
    auto p = common_instances_parse("plain:ctx=512");
    assert(p.size() == 1 && p[0].lora.empty());

    // invalid scales and paths are parse errors
    expect_parse_error("a:lora=");            // empty path
    expect_parse_error("a:lora=./x.gguf:0");  // zero scale
    expect_parse_error("a:lora=./x.gguf:-2"); // negative scale
    expect_parse_error("a:lora=./x.gguf:lora=./x.gguf"); // duplicate path
}

// multiple lora= entries keep order and scales; non-finite scales are rejected.
static void test_instances_lora_multi_scale() {
    auto v = common_instances_parse("a:lora=./x.gguf:0.5:lora=./y.gguf:lora=./z.gguf:2.0");
    assert(v.size() == 1 && v[0].lora.size() == 3);
    assert(v[0].lora[0].first == "./x.gguf" && v[0].lora[0].second == 0.5f);
    assert(v[0].lora[1].first == "./y.gguf" && v[0].lora[1].second == 1.0f);
    assert(v[0].lora[2].first == "./z.gguf" && v[0].lora[2].second == 2.0f);

    expect_parse_error("a:lora=./x.gguf:nan"); // not a positive finite scale
    expect_parse_error("a:lora=./x.gguf:inf");
    expect_parse_error("a:lora=./x.gguf:-inf");
}

// separator discipline: ',' splits instances so a parsed path never contains
// one; a ':' inside the path breaks the option parse.
static void test_instances_lora_separators() {
    auto v = common_instances_parse("a:lora=./x,y.gguf");
    assert(v.size() == 2);
    assert(v[0].lora.size() == 1 && v[0].lora[0].first == "./x");
    assert(v[1].name == "y.gguf" && v[1].lora.empty());
    expect_parse_error("a:lora=./x:0.5.gguf"); // ':' inside the path
}

// specs written before lora= existed still parse with an empty adapter list.
static void test_instances_lora_compat() {
    auto v = common_instances_parse("swarm0:group=swarm:ctx=16384:parallel=1,ledger:ctx=65536:pinned:default");
    assert(v.size() == 2);
    assert(v[0].lora.empty() && v[1].lora.empty());
}

// to_string emits the adapter list and the result reparses to the same list.
static void test_instances_lora_round_trip() {
    auto v = common_instances_parse("a:lora=./x.gguf:0.5:lora=./y.gguf,b:ctx=512");
    assert(v.size() == 2);
    const std::string s = common_instances_to_string(v);
    assert(s.find(":lora=./x.gguf:") != std::string::npos);
    assert(s.find(":lora=./y.gguf") != std::string::npos);
    // default scale prints bare (no scale suffix after the path)
    assert(s.find(":lora=./y.gguf:") == std::string::npos);

    const auto reparsed = common_instances_parse(s);
    assert(reparsed.size() == 2);
    assert(reparsed[0].lora.size() == 2);
    assert(reparsed[0].lora[0].first == "./x.gguf" && reparsed[0].lora[0].second == 0.5f);
    assert(reparsed[0].lora[1].first == "./y.gguf" && reparsed[0].lora[1].second == 1.0f);
    assert(reparsed[1].lora.empty());
}

// a space can never survive the CLI layer, so adapter paths with spaces are
// rejected at validation.
static void test_instances_lora_validate() {
    expect_parse_error("a:lora=./x y.gguf");
    common_instance inst;
    inst.name  = "a";
    inst.group = "a";
    inst.lora.emplace_back("./x y.gguf", 1.0f);
    bool threw = false;
    try {
        common_instance_validate(inst);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    assert(threw);
}

static void test_instance_params_lora() {
    common_params base;
    base.lora_adapters.push_back({ "base.gguf", 1.0f, "", "", nullptr });

    // empty inst.lora inherits the base set untouched
    common_instance inherit;
    inherit.name = "inherit";
    common_params p = common_instance_params(base, inherit);
    assert(p.lora_adapters.size() == 1);
    assert(p.lora_adapters[0].path == "base.gguf" && p.lora_adapters[0].scale == 1.0f);

    // non-empty inst.lora REPLACES the base set; ptrs stay null for the pool
    common_instance over;
    over.name = "over";
    over.lora = { { "./a.gguf", 0.5f }, { "./b.gguf", 2.0f } };
    p = common_instance_params(base, over);
    assert(p.lora_adapters.size() == 2);
    assert(p.lora_adapters[0].path == "./a.gguf" && p.lora_adapters[0].scale == 0.5f);
    assert(p.lora_adapters[1].path == "./b.gguf" && p.lora_adapters[1].scale == 2.0f);
    assert(p.lora_adapters[0].ptr == nullptr && p.lora_adapters[1].ptr == nullptr);
}

static void test_lora_fingerprint() {
    std::vector<common_adapter_lora_info> empty;
    assert(common_lora_fingerprint(empty).empty());

    // stable and order-independent; ptr is never an input
    std::vector<common_adapter_lora_info> a = {
        { "b.gguf", 1.0f, "", "", nullptr },
        { "a.gguf", 0.5f, "", "", (llama_adapter_lora *) 0x1234 },
    };
    std::vector<common_adapter_lora_info> b = {
        { "a.gguf", 0.5f, "", "", nullptr },
        { "b.gguf", 1.0f, "", "", nullptr },
    };
    assert(!common_lora_fingerprint(a).empty());
    assert(common_lora_fingerprint(a) == common_lora_fingerprint(b));

    // scale is part of the identity
    std::vector<common_adapter_lora_info> c = {
        { "a.gguf", 1.0f, "", "", nullptr },
    };
    std::vector<common_adapter_lora_info> d = {
        { "a.gguf", 0.5f, "", "", nullptr },
    };
    assert(common_lora_fingerprint(c) != common_lora_fingerprint(d));
}

static void test_adapter_buf_size_null() {
    assert(llama_adapter_lora_buf_size(nullptr) == 0);
}

static void test_borrowed_model(const common_params & base) {
    common_params params = base;
    params.n_ctx = 256;
    params.n_parallel = 1;
    params.warmup = false;

    auto model_init = common_init_from_params(params, true);
    llama_model * model = model_init->model();
    if (model == nullptr) {
        fprintf(stderr, "failed to load model\n");
        exit(1);
    }
    assert(model_init->context() == nullptr);

    // build two contexts from the SAME model with different ctx sizes
    // note: the KV cache pads n_ctx_seq to a multiple of 256
    common_instance inst_a;
    inst_a.name = "a";
    inst_a.ctx_size = 512;
    common_instance inst_b;
    inst_b.name = "b";
    inst_b.ctx_size = 1024;

    common_params p_a = common_instance_params(params, inst_a);
    common_params p_b = common_instance_params(params, inst_b);

    std::vector<common_init_result_ptr> ctxs;
    ctxs.push_back(common_init_from_model_params(p_a, model));
    ctxs.push_back(common_init_from_model_params(p_b, model));

    assert(ctxs[0]->context() != nullptr);
    assert(ctxs[1]->context() != nullptr);
    assert(ctxs[0]->model() == model); // borrowed: model() returns the shared model
    assert(ctxs[1]->model() == model);
    assert(llama_n_ctx(ctxs[0]->context()) == 512);
    assert(llama_n_ctx(ctxs[1]->context()) == 1024);

    // samplers are initialized on the borrowed path too
    assert(ctxs[0]->sampler(0) != nullptr);
    assert(ctxs[1]->sampler(0) != nullptr);

    // destroy the contexts; the model must survive
    ctxs.clear();

    const llama_vocab * vocab = llama_model_get_vocab(model);
    assert(llama_vocab_n_tokens(vocab) > 0);

    // a fresh context can still be built from the model
    auto ctx3 = common_init_from_model_params(p_a, model);
    assert(ctx3->context() != nullptr);
    assert(ctx3->model() == model);

    // model_init is destroyed last; the model is freed there
}

// a demand-driven build followed by repeated start_loops() must leave exactly one
// scheduler thread: a second start would assign over a joinable thread and abort.
static void test_demand_build_starts_one_loop(const common_params & base) {
    common_params params = base;
    params.n_ctx      = 256;
    params.n_parallel = 1;
    params.warmup     = false;

    common_instance cfg;
    cfg.name     = "solo";
    cfg.group    = "solo";
    cfg.ctx_size = 256;
    cfg.parallel = 1;
    params.instances.push_back(cfg);

    int builds = 0;
    server_instances mgr;
    mgr.set_context_builder([&builds, &mgr](server_instance & inst) {
        ++builds;
        return mgr.build_context_default(inst);
    });

    assert(mgr.load(params));
    assert(builds == 0);
    assert(!mgr.instances.front()->built);

    // a no-model handler forces the demand-driven build of the default instance
    static const std::function<bool()> no_stop = []() { return false; };
    server_http_req req { {}, {}, "/props", "", "", {}, no_stop };
    auto res = mgr.handle_get_props(req);
    assert(res->status == 200);
    assert(builds == 1);
    assert(mgr.instances.front()->built);
    assert(mgr.instances.front()->loop_started);

    // repeated starts are no-ops: no second thread, no abort, no extra build
    mgr.start_loops();
    mgr.start_loops();
    assert(builds == 1);
    assert(mgr.instances.front()->loop_started);

    mgr.terminate();
}

// control: start_loops() on a pool of only unbuilt windows starts nothing.
static void test_start_loops_skips_unbuilt(const common_params & base) {
    common_params params = base;
    params.n_ctx      = 256;
    params.n_parallel = 1;
    params.warmup     = false;

    common_instance cfg;
    cfg.name     = "cold";
    cfg.group    = "cold";
    cfg.ctx_size = 256;
    cfg.parallel = 1;
    params.instances.push_back(cfg);

    server_instances mgr;
    assert(mgr.load(params));
    assert(!mgr.instances.front()->built);

    mgr.start_loops();
    assert(!mgr.instances.front()->built);
    assert(!mgr.instances.front()->loop_started);
    assert(!mgr.instances.front()->ctx_server);
    assert(!mgr.instances.front()->loop_thread.joinable());

    mgr.terminate();
}

// resize is a manager-owned teardown + rebuild: success leaves a fresh window
// with exactly one scheduler, failure leaves a well-defined unbuilt window
// that a later demand retries at the new size.
static void test_resize_teardown_rebuild(const common_params & base) {
    common_params params = base;
    params.n_ctx      = 256;
    params.n_parallel = 1;
    params.warmup     = false;

    common_instance cfg;
    cfg.name     = "r";
    cfg.group    = "r";
    cfg.ctx_size = 256;
    cfg.parallel = 1;
    params.instances.push_back(cfg);

    int builds = 0;
    server_instances mgr;
    mgr.set_context_builder([&builds, &mgr](server_instance & inst) {
        ++builds;
        return mgr.build_context_default(inst);
    });
    assert(mgr.load(params));

    static const std::function<bool()> no_stop = []() { return false; };
    server_http_req props_req { {}, {}, "/props", "", "", {}, no_stop };
    auto resize_req = [&](int32_t n_ctx) {
        return server_http_req { { { "name", "r" } }, {}, "/instances/r/resize", "",
                                 safe_json_to_str({ { "ctx_size", n_ctx } }), {}, no_stop };
    };

    // demand-build the 256 window first
    assert(mgr.handle_get_props(props_req)->status == 200);
    assert(builds == 1);

    // successful resize: fresh 512 window, exactly one scheduler
    auto res = mgr.handle_post_instance_resize(resize_req(512));
    assert(res->status == 200);
    assert(json::parse(res->data)["n_ctx"].get<int>() == 512);
    assert(builds == 2);
    const auto inst = mgr.instances.front();
    assert(inst->built && inst->loop_started && inst->loop_thread.joinable());
    assert(inst->effective.n_ctx == 512);
    mgr.start_loops();
    mgr.start_loops();
    assert(builds == 2);
    assert(mgr.handle_get_props(props_req)->status == 200);

    // program the builder to fail: the resize 507s and leaves a clean unbuilt
    // window (null context/routes, no joinable thread) at the new size
    mgr.set_context_builder([](server_instance &) { return false; });
    res = mgr.handle_post_instance_resize(resize_req(1024));
    assert(res->status == 507);
    assert(!inst->built && !inst->ctx_server && !inst->routes);
    assert(!inst->loop_thread.joinable());
    assert(inst->cfg.ctx_size == 1024);
    assert(inst->effective.n_ctx == 1024);

    // a later demand retries at the new size and serves again
    mgr.set_context_builder(nullptr);
    assert(mgr.handle_get_props(props_req)->status == 200);
    assert(builds == 2);
    assert(inst->built && inst->loop_started && inst->loop_thread.joinable());
    assert(inst->effective.n_ctx == 1024);

    mgr.terminate();
}

// the display caches (n_ctx_effective / n_parallel_effective) track the
// requested size across register -> unbuilt resize -> lazy build -> built resize,
// so a lock-free read never needs the mutable `effective` params.
static void test_display_cache_transitions(const common_params & base) {
    common_params params = base;
    params.n_ctx      = 128;
    params.n_parallel = 1;
    params.warmup     = false;

    common_instance cfg;
    cfg.name     = "a";
    cfg.group    = "a";
    cfg.ctx_size = 256;
    cfg.parallel = 1;
    params.instances.push_back(cfg);

    server_instances mgr;
    assert(mgr.load(params));
    const auto inst = mgr.instances.front();

    static const std::function<bool()> no_stop = []() { return false; };
    server_http_req props_req { {}, {}, "/props", "", "", {}, no_stop };
    auto resize_req = [&](int32_t n_ctx) {
        return server_http_req { { { "name", "a" } }, {}, "/instances/a/resize", "",
                                 safe_json_to_str({ { "ctx_size", n_ctx } }), {}, no_stop };
    };

    // registered but unbuilt: the cache reports the requested size
    assert(!inst->built.load());
    assert(mgr.displayed_n_ctx(*inst) == 256);
    assert(inst->n_parallel_effective.load() == 1);

    // unbuilt resize: the declaration changes and the display follows it
    auto res = mgr.handle_post_instance_resize(resize_req(512));
    assert(res->status == 200);
    assert(!inst->built.load());
    assert(mgr.displayed_n_ctx(*inst) == 512);

    // lazy build: the cache must equal the real context size
    assert(mgr.handle_get_props(props_req)->status == 200);
    assert(inst->built.load());
    assert(mgr.displayed_n_ctx(*inst) == inst->ctx_server->get_n_ctx());
    assert(inst->n_ctx_effective.load() == inst->ctx_server->get_n_ctx());

    // built resize: the cache follows the rebuilt context. the context size is
    // padded by the backend to a 256 multiple, so compare against get_n_ctx().
    res = mgr.handle_post_instance_resize(resize_req(256));
    assert(res->status == 200);
    assert(inst->built.load());
    assert(mgr.displayed_n_ctx(*inst) == 256);
    assert(mgr.displayed_n_ctx(*inst) == inst->ctx_server->get_n_ctx());
    assert(inst->n_ctx_effective.load() == inst->ctx_server->get_n_ctx());

    mgr.terminate();
}

// readers hammer the lock-free display paths while writers create, demand,
// resize and destroy instances. the atomic `built` flag and the display caches
// must keep every returned row consistent (no torn context, n_ctx >= 0,
// parallel >= 1) and leave the pool back at its single default member.
static void test_built_publish_stress(const common_params & base) {
    common_params params = base;
    params.n_ctx      = 128;
    params.n_parallel = 1;
    params.warmup     = false;
    params.endpoint_slots = true;

    common_instance cfg;
    cfg.name       = "base";
    cfg.group      = "base";
    cfg.ctx_size   = 128;
    cfg.parallel   = 1;
    cfg.is_default = true;
    params.instances.push_back(cfg);

    server_instances mgr;
    assert(mgr.load(params));

    static const std::function<bool()> no_stop = []() { return false; };
    std::atomic<bool> stop{false};
    std::atomic<int>  errors{0};

    const auto req = [&](const char * path) {
        return server_http_req { {}, {}, path, "", "", {}, no_stop };
    };

    auto reader = [&]() {
        while (!stop.load(std::memory_order_relaxed)) {
            try {
                auto m = mgr.handle_get_models(req("/models"));
                if (m->status == 200) {
                    json body = json::parse(m->data);
                    for (auto & d : body["data"]) {
                        if (d["n_ctx"].get<int>() < 0 || d["parallel"].get<int>() < 1) {
                            ++errors;
                        }
                    }
                }
                auto p = mgr.handle_get_props(req("/props"));
                if (p->status == 200) {
                    json body = json::parse(p->data);
                    for (auto & it : body["instances"]) {
                        if (it["n_ctx"].get<int>() < 0) {
                            ++errors;
                        }
                    }
                }
                json env = json::parse(mgr.handle_get_instances(req("/instances"))->data);
                for (auto & it : env["instances"]) {
                    if (it["n_ctx"].get<int>() < 0 || it["parallel"].get<int>() < 1) {
                        ++errors;
                    }
                }
            } catch (const std::exception &) {
                ++errors;
            }
        }
    };

    auto writer = [&](int id) {
        for (int k = 0; k < 15; ++k) {
            const std::string name = "w" + std::to_string(id) + "_" + std::to_string(k);
            server_http_req create { {}, {}, "/instances", "",
                safe_json_to_str({ { "name", name }, { "ctx_size", 128 }, { "group", "g" } }), {}, no_stop };
            if (mgr.handle_post_instances(create)->status != 201) {
                ++errors;
                continue;
            }
            std::string error;
            auto target = mgr.resolve(name, "", error);
            if (target.kind == server_instances::target_kind::INSTANCE) {
                // the targeted /slots path materializes the window (and must not be a
                // mere display read); the status is irrelevant here
                server_http_req demand { { { "instance", name } }, {}, "/slots", "", "", {}, no_stop };
                mgr.handle_get_slots(demand);
            }
            server_http_req resize { { { "name", name } }, {}, "/instances/" + name + "/resize", "",
                safe_json_to_str({ { "ctx_size", 192 } }), {}, no_stop };
            mgr.handle_post_instance_resize(resize);
            server_http_req del { { { "name", name } }, {}, "/instances/" + name, "", "", {}, no_stop };
            mgr.handle_delete_instance(del);
        }
    };

    std::vector<std::thread> readers;
    for (int i = 0; i < 3; ++i) {
        readers.emplace_back(reader);
    }
    std::thread wa(writer, 0);
    std::thread wb(writer, 1);
    wa.join();
    wb.join();
    stop.store(true, std::memory_order_relaxed);
    for (auto & t : readers) {
        t.join();
    }

    assert(errors.load() == 0);

    // every writer destroyed its instances: only the default remains
    json env = json::parse(mgr.handle_get_instances(req("/instances"))->data);
    assert(env["instances"].size() == 1);

    mgr.terminate();
}

// a cold reload that finds the adapter registry non-empty is a refcount bug.
// the debug build aborts at the source, so this body only exists in release
// builds (NDEBUG), where the contract is a logged 507 and a surviving process.
static void test_cold_reload_registry_mismatch(const common_params & base) {
#ifdef TEST_LIBRARY_NDEBUG
    common_params params = base;
    params.n_ctx      = 256;
    params.n_parallel = 1;
    params.warmup     = false;

    common_instance cfg;
    cfg.name     = "a";
    cfg.group    = "a";
    cfg.ctx_size = 256;
    cfg.parallel = 1;
    params.instances.push_back(cfg);

    server_instances mgr;
    assert(mgr.load(params));
    assert(mgr.model != nullptr);

    // a leaked registry entry (never attached, never released). the release path
    // only inspects emptiness, so a null adapter is enough and keeps teardown safe
    mgr.adapter_registry["leaked"] = server_instances::adapter_registry_entry{};

    // force the cold state without going through destroy's registry drain
    mgr.model = nullptr;

    static const std::function<bool()> no_stop = []() { return false; };
    server_http_req create { {}, {}, "/instances", "",
        safe_json_to_str({ { "name", "b" }, { "ctx_size", 256 } }), {}, no_stop };
    auto res = mgr.handle_post_instances(create);
    assert(res->status == 507);
    assert(res->data.find("inconsistent") != std::string::npos);

    mgr.terminate();
#else
    (void) base;
#endif
}

// pool simulation: the pool loads the adapter once against the shared model, then
// hands the ptr to init_from_model. the bogus path proves the skip guard: if init
// tried to load-and-own again it would fail on the bogus file and return no context.
static void test_borrowed_model_adapter_skip(const common_params & base, const std::string & adapter_path) {
    common_params params = base;
    params.n_ctx = 256;
    params.n_parallel = 1;
    params.warmup = false;

    auto model_init = common_init_from_params(params, true);
    llama_model * model = model_init->model();
    assert(model != nullptr);

    // the pool's single load (outside init_from_model)
    llama_adapter_lora * pool_adapter = llama_adapter_lora_init(model, adapter_path.c_str());
    if (pool_adapter == nullptr) {
        fprintf(stderr, "WARNING: cannot load adapter '%s' on this model, skipping.\n", adapter_path.c_str());
        return;
    }
    assert(llama_adapter_lora_buf_size(pool_adapter) > 0);

    common_params p = params;
    p.lora_adapters = { { "/nonexistent/bogus.gguf", 0.5f, "", "", pool_adapter } };
    auto ctx = common_init_from_model_params(p, model);
    assert(ctx->context() != nullptr); // the bogus path was never touched
    assert(p.lora_adapters[0].ptr == pool_adapter);

    // teardown: the context never owned the adapter (no double-free below), the
    // pool frees its single load here
    ctx.reset();
    llama_adapter_lora_free(pool_adapter);
}

// golden for one built and one unbuilt window through the live envelope:
// exact row key set, loaded/unloaded states, zero bytes while unbuilt, and
// the total_bytes/vram_bytes identities. runs against the real model fixture.
static void test_instances_envelope_built_unbuilt(const common_params & base) {
    common_params params = base;
    params.n_ctx      = 256;
    params.n_parallel = 1;
    params.warmup     = false;

    common_instance cfg_cold;
    cfg_cold.name     = "cold";
    cfg_cold.group    = "cold";
    cfg_cold.ctx_size = 256;
    cfg_cold.parallel = 1;
    common_instance cfg_warm;
    cfg_warm.name        = "warm";
    cfg_warm.group       = "warm";
    cfg_warm.ctx_size    = 256;
    cfg_warm.parallel    = 1;
    cfg_warm.is_default  = true;
    params.instances.push_back(cfg_cold);
    params.instances.push_back(cfg_warm);

    server_instances mgr;
    assert(mgr.load(params));

    static const std::function<bool()> no_stop = []() { return false; };
    server_http_req list_req { {}, {}, "/instances", "", "", {}, no_stop };
    server_http_req props_req { {}, {}, "/props", "", "", {}, no_stop };

    // before any demand both windows are registered but unbuilt
    {
        auto res = mgr.handle_get_instances(list_req);
        assert(res->status == 200);
        const json body = json::parse(res->data);
        assert(body["instances"].size() == 2);
        for (const auto & row : body["instances"]) {
            assert(row["state"] == "unloaded");
            assert(row["context_bytes"] == 0);
            assert(row["compute_bytes"] == 0);
            assert(row["model_bytes"] == 0);
            assert(row["adapter_bytes"] == 0);
            assert(row["total_bytes"] == 0);
            assert(row["vram_bytes"] == 0);
            assert(row["last_used"] == -1);
            assert(row["last_used_epoch"] == -1);
        }
        const json total = body["total"];
        assert(total["model"] == 0 && total["context"] == 0 && total["compute"] == 0);
    }

    // one demand builds exactly one window: warm loads, cold stays unbuilt
    assert(mgr.handle_get_props(props_req)->status == 200);
    {
        auto res = mgr.handle_get_instances(list_req);
        assert(res->status == 200);
        const json body = json::parse(res->data);
        assert(body["instances"].size() == 2);

        const json * cold = nullptr;
        const json * warm = nullptr;
        for (const auto & row : body["instances"]) {
            const std::string id = row["id"].get<std::string>();
            if (id.find(":cold") != std::string::npos) {
                cold = &row;
            } else if (id.find(":warm") != std::string::npos) {
                warm = &row;
            }
        }
        assert(cold != nullptr && warm != nullptr);
        assert((*cold)["state"] == "unloaded");
        assert((*cold)["context_bytes"] == 0);
        assert((*warm)["state"] == "loaded");
        assert((*warm)["n_ctx"] == 256);
        assert((*warm)["parallel"] == 1);
        assert((*warm)["is_default"] == true);
        // byte identities hold on the built row too
        const uint64_t m  = (*warm)["model_bytes"].get<uint64_t>();
        const uint64_t cx = (*warm)["context_bytes"].get<uint64_t>();
        const uint64_t co = (*warm)["compute_bytes"].get<uint64_t>();
        assert((*warm)["total_bytes"] == m + cx + co);
        assert((*warm)["vram_bytes"] == cx + co);
    }

    mgr.terminate();
}

// cancelling a still-queued task removes it before any scheduler runs it:
// the handler never fires and no result is delivered.
static void test_queue_stop_cancels_pending() {
    static_assert(HTTP_POLLING_SECONDS == 1, "polling cadence for scheduler waits");
    server_queue                                queue;
    server_result_queue<server_task_result_ptr> response;
    std::atomic<bool> executed{false};
    queue.on_new_task([&](server_task && task, bool) -> bool {
        if (task.type == SERVER_TASK_TYPE_INSTANCE_OP) {
            executed.store(true);
        }
        return true;
    });
    queue.on_update_slots([]() {});

    server_response_reader rd(queue, response, 0);
    server_task task(SERVER_TASK_TYPE_INSTANCE_OP);
    task.id = rd.get_new_id();
    task.instance_op = []() -> json { return json{ { "ok", true } }; };
    rd.post_task(std::move(task));

    rd.stop(); // no scheduler is running: the task is still queued

    std::thread loop([&]() { queue.start_loop(-1); });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    queue.terminate();
    loop.join();

    assert(!executed.load());
    assert(rd.next([]() { return true; }) == nullptr);
}

#if 0
// compile check (keep disabled): a result handle whose element type has no int
// id must fail on the queue's static_assert (uncomment to verify)
struct test_queue_result_bad_id {
    std::string id;
};
static void test_queue_wrong_type_rejected() {
    server_result_queue<std::unique_ptr<test_queue_result_bad_id>> queue;
    (void) queue;
}
#endif

// a timed-out manager-to-scheduler task must not run late: with the scheduler
// held by a blocking op, a short-deadline swap times out, and after the
// blocker releases the installed set is unchanged (the pending task was
// cancelled, not left to commit).
static void test_scheduler_timeout_cancels_pending(const common_params & base) {
    common_params params = base;
    params.n_ctx      = 256;
    params.n_parallel = 1;
    params.warmup     = false;

    common_instance cfg;
    cfg.name        = "solo";
    cfg.group       = "solo";
    cfg.ctx_size    = 256;
    cfg.parallel    = 1;
    cfg.is_default  = true;
    params.instances.push_back(cfg);

    server_instances mgr;
    assert(mgr.load(params));

    static const std::function<bool()> no_stop = []() { return false; };
    server_http_req props_req { {}, {}, "/props", "", "", {}, no_stop };
    assert(mgr.handle_get_props(props_req)->status == 200);
    server_context * ctx = mgr.instances.front()->ctx_server.get();
    assert(ctx != nullptr);

    std::atomic<bool> entered{false};
    std::atomic<bool> release{false};
    std::thread blocker([&]() {
        ctx->instance_op([&]() -> json {
            entered.store(true);
            while (!release.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            return json{ { "success", true } };
        });
    });
    while (!entered.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // the scheduler is occupied: the swap must time out. the desired set is a
    // null-ptr entry that touches no file, so only its installation is observed.
    std::vector<common_adapter_lora_info> desired = { { "m1-test.gguf", 1.0f, "", "", nullptr } };
    auto timed_out = ctx->set_lora_adapters(desired, ggml_time_ms() + 50);
    assert(timed_out == nullptr);

    release.store(true);
    blocker.join();

    server_http_req lora_req { {}, {}, "/lora-adapters", "", "", {}, no_stop };
    auto lora_res = mgr.handle_get_lora_adapters(lora_req);
    assert(lora_res->status == 200);
    assert(json::parse(lora_res->data).empty());

    mgr.terminate();
}

// bounded read of the scheduler's installed adapter list through the
// existing GET_LORA task: a set list reads back with the same paths, scales
// and pointers, and re-setting the identical list is a no-op that leaves the
// pointers identical.
static void test_get_lora_adapters_round_trip(const common_params & base) {
    common_params params = base;
    params.n_ctx      = 256;
    params.n_parallel = 1;
    params.warmup     = false;

    common_instance cfg;
    cfg.name        = "solo";
    cfg.group       = "solo";
    cfg.ctx_size    = 256;
    cfg.parallel    = 1;
    cfg.is_default  = true;
    params.instances.push_back(cfg);

    server_instances mgr;
    assert(mgr.load(params));

    static const std::function<bool()> no_stop = []() { return false; };
    server_http_req props_req { {}, {}, "/props", "", "", {}, no_stop };
    assert(mgr.handle_get_props(props_req)->status == 200);
    server_context * ctx = mgr.instances.front()->ctx_server.get();
    assert(ctx != nullptr);

    const int64_t no_deadline = ggml_time_ms() + 5000;

    auto empty = ctx->get_lora_adapters(no_deadline);
    assert(empty.has_value() && empty->empty());

    // null-ptr entries touch no file: only the list plumbing is observed
    std::vector<common_adapter_lora_info> list = {
        { "m2-a.gguf", 0.5f, "", "", nullptr },
        { "m2-b.gguf", 1.0f, "", "", nullptr },
    };
    assert(ctx->set_lora_adapters(list, no_deadline) != nullptr);

    auto got = ctx->get_lora_adapters(no_deadline);
    assert(got.has_value() && got->size() == 2);
    assert((*got)[0].path == "m2-a.gguf" && (*got)[0].scale == 0.5f && (*got)[0].ptr == nullptr);
    assert((*got)[1].path == "m2-b.gguf" && (*got)[1].scale == 1.0f && (*got)[1].ptr == nullptr);

    // identical list: the scheduler short-circuits, pointers stay identical
    assert(ctx->set_lora_adapters(*got, no_deadline) != nullptr);
    auto got2 = ctx->get_lora_adapters(no_deadline);
    assert(got2.has_value() && got2->size() == 2);
    assert((*got2)[0].ptr == (*got)[0].ptr && (*got2)[1].ptr == (*got)[1].ptr);
    assert((*got2)[0].path == (*got)[0].path && (*got2)[1].path == (*got)[1].path);

    mgr.terminate();
}

// test-only: synthesize a minimal loadable LoRA GGUF for `model`: one rank-8
// pair on blk.0.attn_q.weight with deterministic data. hermetic: no network,
// no fixture file.
static std::string test_write_tiny_lora(llama_model * model, const std::string & path) {
    const int n_embd = llama_model_n_embd(model);
    const int rank   = 8;
    ggml_init_params gparams = { 16 * 1024 * 1024, nullptr, false };
    ggml_context * gctx = ggml_init(gparams);
    assert(gctx != nullptr);
    ggml_tensor * a = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, n_embd, rank);
    ggml_tensor * b = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, rank, n_embd);
    ggml_set_name(a, "blk.0.attn_q.weight.lora_a");
    ggml_set_name(b, "blk.0.attn_q.weight.lora_b");
    for (int64_t i = 0; i < ggml_nelements(a); i++) {
        ((float *) a->data)[i] = 0.001f * (float) (i % 7);
    }
    for (int64_t i = 0; i < ggml_nelements(b); i++) {
        ((float *) b->data)[i] = 0.001f * (float) (i % 5);
    }
    gguf_context * ggu = gguf_init_empty();
    gguf_set_val_str(ggu, "general.type", "adapter");
    gguf_set_val_str(ggu, "general.architecture", "llama");
    gguf_set_val_str(ggu, "adapter.type", "lora");
    gguf_set_val_f32(ggu, "adapter.lora.alpha", 8.0f);
    gguf_add_tensor(ggu, a);
    gguf_add_tensor(ggu, b);
    const bool ok = gguf_write_to_file(ggu, path.c_str(), false);
    gguf_free(ggu);
    ggml_free(gctx);
    assert(ok);
    return path;
}

// test-only: load the shared weights model-only to size the synthetic LoRAs,
// then write two files: file1 and a byte-identical file2 under a distinct
// path (the registry keys on path, so file2 is a second entry).
static void test_write_lora_pair(const common_params & base, const std::string & dir,
                                 std::string & file1, std::string & file2) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(dir, ec);

    common_params params = base;
    auto model_init = common_init_from_params(params, true);
    llama_model * model = model_init->model();
    assert(model != nullptr);

    file1 = (fs::path(dir) / "m3-a.gguf").string();
    file2 = (fs::path(dir) / "m3-b.gguf").string();
    test_write_tiny_lora(model, file1);
    fs::copy_file(file1, file2, ec);
    assert(!ec);
}

static server_http_req test_req(const std::map<std::string, std::string> & params,
                                const std::string & path,
                                const std::string & body) {
    static const std::function<bool()> no_stop = []() { return false; };
    return server_http_req { params, {}, path, "", body, {}, no_stop };
}

static std::shared_ptr<server_instance> test_find(server_instances & mgr, const std::string & name) {
    for (const auto & inst : mgr.instances) {
        if (inst->cfg.name == name) {
            return inst;
        }
    }
    return nullptr;
}

// teardown liveness: destroying an instance whose scheduler is wedged answers
// 503 within the compose deadline instead of hanging, leaves the instance
// intact, and releases the management plane so an unrelated op completes.
static void test_destroy_bounded_on_wedged_scheduler(const common_params & base) {
    common_params params = base;
    params.n_ctx      = 256;
    params.n_parallel = 1;
    params.warmup     = false;

    common_instance victim;
    victim.name        = "victim";
    victim.group       = "victim";
    victim.ctx_size    = 256;
    victim.parallel    = 1;
    victim.is_default  = true;
    common_instance witness;
    witness.name     = "w";
    witness.group    = "w";
    witness.ctx_size = 256;
    witness.parallel = 1;
    params.instances.push_back(victim);
    params.instances.push_back(witness);

    server_instances mgr;
    assert(mgr.load(params));

    static const std::function<bool()> no_stop = []() { return false; };
    server_http_req props_req { {}, {}, "/props", "", "", {}, no_stop };
    assert(mgr.handle_get_props(props_req)->status == 200); // builds the default (victim)
    server_context * ctx = test_find(mgr, "victim")->ctx_server.get();
    assert(ctx != nullptr);

    // wedge the victim scheduler with a blocking op
    std::atomic<bool> entered{false};
    std::atomic<bool> release{false};
    std::thread blocker([&]() {
        ctx->instance_op([&]() -> json {
            entered.store(true);
            while (!release.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            return json{ { "success", true } };
        });
    });
    while (!entered.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // the destroy must 503 within the compose deadline (1s floor at ctx 256),
    // not hang; the witness pin queued behind it completes once the
    // management plane is released
    std::atomic<int>     destroy_status{0};
    std::atomic<int64_t> destroy_ms{0};
    std::thread destroyer([&]() {
        const int64_t t0 = ggml_time_ms();
        auto res = mgr.handle_delete_instance(test_req({ { "name", "victim" } }, "/instances/victim", ""));
        destroy_ms.store(ggml_time_ms() - t0);
        destroy_status.store(res->status);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(200)); // let the destroy block in abort
    const int64_t t1 = ggml_time_ms();
    auto pin_res = mgr.handle_post_instance_pin(test_req({ { "name", "w" } }, "/instances/w/pin", ""));
    const int64_t pin_ms = ggml_time_ms() - t1;
    assert(pin_res->status == 200);
    destroyer.join();

    assert(destroy_status.load() == 503);
    // bounded: about one deadline, far from forever
    assert(destroy_ms.load() >= 500 && destroy_ms.load() < 8000);
    // the pin waited out the bounded stall, then completed
    assert(pin_ms < 8000);
    fprintf(stdout, "wedged teardown: destroy status = %d in %lld ms, queued pin in %lld ms\n",
        destroy_status.load(), (long long) destroy_ms.load(), (long long) pin_ms);
    // the instance is intact: still registered, still built
    assert(test_find(mgr, "victim") != nullptr);
    assert(test_find(mgr, "victim")->built);

    // unwedge: the same destroy now succeeds, proving nothing was torn down
    release.store(true);
    blocker.join();
    assert(mgr.handle_delete_instance(test_req({ { "name", "victim" } }, "/instances/victim", ""))->status == 200);
    assert(test_find(mgr, "victim") == nullptr);

    mgr.terminate();
}

// calibration control group: on a healthy scheduler every management op
// succeeds fast, so the bounded wait never fires. runs identically before
// and after the deadline is introduced.
static void test_management_healthy_control_fast(const common_params & base) {
    common_params params = base;
    params.n_ctx      = 256;
    params.n_parallel = 1;
    params.warmup     = false;

    common_instance solo;
    solo.name       = "solo";
    solo.group      = "solo";
    solo.ctx_size   = 256;
    solo.parallel   = 1;
    solo.is_default = true;
    params.instances.push_back(solo);

    server_instances mgr;
    assert(mgr.load(params));

    static const std::function<bool()> no_stop = []() { return false; };
    server_http_req props_req { {}, {}, "/props", "", "", {}, no_stop };
    assert(mgr.handle_get_props(props_req)->status == 200);

    // create / pin / resize / destroy: all 200, each far below the compose
    // deadline (1s floor), so a bounded wait can never mistake them for stuck
    int64_t worst_ms = 0;
    for (int i = 0; i < 3; ++i) {
        const std::string name = "x" + std::to_string(i);
        int64_t t0 = ggml_time_ms();
        auto created = mgr.handle_post_instances(test_req({}, "/instances",
            safe_json_to_str({ { "name", name }, { "ctx_size", 256 } })));
        assert(created->status == 201);
        worst_ms = std::max(worst_ms, ggml_time_ms() - t0);
        assert(ggml_time_ms() - t0 < 8000);

        t0 = ggml_time_ms();
        assert(mgr.handle_post_instance_pin(test_req({ { "name", name } }, "/instances/" + name + "/pin", ""))->status == 200);
        worst_ms = std::max(worst_ms, ggml_time_ms() - t0);
        assert(ggml_time_ms() - t0 < 8000);

        t0 = ggml_time_ms();
        assert(mgr.handle_delete_instance(test_req({ { "name", name } }, "/instances/" + name, ""))->status == 200);
        worst_ms = std::max(worst_ms, ggml_time_ms() - t0);
        assert(ggml_time_ms() - t0 < 8000);
    }

    int64_t t0 = ggml_time_ms();
    auto resized = mgr.handle_post_instance_resize(test_req({ { "name", "solo" } }, "/instances/solo/resize",
        safe_json_to_str({ { "ctx_size", 512 } })));
    assert(resized->status == 200);
    worst_ms = std::max(worst_ms, ggml_time_ms() - t0);
    assert(ggml_time_ms() - t0 < 8000);
    fprintf(stdout, "healthy control: 10 ops, 0 retriable, worst op %lld ms\n", (long long) worst_ms);

    mgr.terminate();
}

// pool sharing: a declared adapter resolves to the pool's entry on
// demand-build (ptr non-null, installed identical), and attaching the same
// file to a second instance observes the same ptr. a distinct file, even with
// identical bytes, resolves to a distinct ptr. read-after-post agreement holds.
static void test_pool_adapter_shared_ptr(const common_params & base) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const std::string dir = (fs::temp_directory_path(ec) / "llama-m3-shared").string();
    fs::remove_all(dir, ec);
    std::string file1;
    std::string file2;
    test_write_lora_pair(base, dir, file1, file2);

    common_params params = base;
    params.n_ctx      = 256;
    params.n_parallel = 1;
    params.warmup     = false;

    common_instance cfga;
    cfga.name        = "a";
    cfga.group       = "a";
    cfga.ctx_size    = 256;
    cfga.parallel    = 1;
    cfga.is_default  = true;
    cfga.lora.emplace_back(file1, 1.0f);
    params.instances.push_back(cfga);

    server_instances mgr;
    assert(mgr.load(params));

    static const std::function<bool()> no_stop = []() { return false; };
    server_http_req props_req { {}, {}, "/props", "", "", {}, no_stop };
    assert(mgr.handle_get_props(props_req)->status == 200);

    auto inst_a = test_find(mgr, "a");
    assert(inst_a && inst_a->built);
    assert(inst_a->effective.lora_adapters.size() == 1);
    llama_adapter_lora * pool_ptr = inst_a->effective.lora_adapters[0].ptr;
    assert(pool_ptr != nullptr);

    const int64_t deadline = ggml_time_ms() + 5000;
    auto installed_a = inst_a->ctx_server->get_lora_adapters(deadline);
    assert(installed_a.has_value() && installed_a->size() == 1);
    assert((*installed_a)[0].ptr == pool_ptr);

    // second instance shares the file: the same registry entry, the same ptr
    auto create = mgr.handle_post_instances(test_req({}, "/instances",
        safe_json_to_str({ { "name", "b" }, { "ctx_size", 256 } })));
    assert(create->status == 201);
    auto attach = mgr.handle_post_instance_adapters(test_req({ { "name", "b" } },
        "/instances/b/adapters", safe_json_to_str({ { "path", file1 }, { "scale", 1.0f } })));
    assert(attach->status == 200);

    auto inst_b = test_find(mgr, "b");
    assert(inst_b && inst_b->built);
    auto installed_b = inst_b->ctx_server->get_lora_adapters(deadline);
    assert(installed_b.has_value() && installed_b->size() == 1);
    assert((*installed_b)[0].ptr == pool_ptr);

    // distinct file: a distinct entry with a distinct ptr
    attach = mgr.handle_post_instance_adapters(test_req({ { "name", "b" } },
        "/instances/b/adapters", safe_json_to_str({ { "path", file2 }, { "scale", 0.5f } })));
    assert(attach->status == 200);
    installed_b = inst_b->ctx_server->get_lora_adapters(deadline);
    assert(installed_b.has_value() && installed_b->size() == 2);
    assert((*installed_b)[1].ptr != nullptr);
    assert((*installed_b)[1].ptr != pool_ptr);

    mgr.terminate();
    fs::remove_all(dir, ec);
}

// fingerprint over pool refs: stable, order-independent, never a ptr input,
// and "" for the empty set.
static void test_pool_adapter_fingerprint(const common_params & base) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const std::string dir = (fs::temp_directory_path(ec) / "llama-m3-fp").string();
    fs::remove_all(dir, ec);
    std::string file1;
    std::string file2;
    test_write_lora_pair(base, dir, file1, file2);

    assert(common_lora_fingerprint({}).empty());

    common_params params = base;
    params.n_ctx      = 256;
    params.n_parallel = 1;
    params.warmup     = false;

    common_instance cfga;
    cfga.name        = "a";
    cfga.group       = "a";
    cfga.ctx_size    = 256;
    cfga.parallel    = 1;
    cfga.is_default  = true;
    cfga.lora.emplace_back(file1, 1.0f);
    cfga.lora.emplace_back(file2, 0.5f);
    params.instances.push_back(cfga);

    server_instances mgr;
    assert(mgr.load(params));

    static const std::function<bool()> no_stop = []() { return false; };
    server_http_req props_req { {}, {}, "/props", "", "", {}, no_stop };
    assert(mgr.handle_get_props(props_req)->status == 200);

    auto inst_a = test_find(mgr, "a");
    assert(inst_a && inst_a->built);
    const std::string fp = common_lora_fingerprint(inst_a->effective.lora_adapters);
    assert(!fp.empty());

    // order-independent: the same set listed backwards hashes alike
    auto swapped = inst_a->effective.lora_adapters;
    std::swap(swapped[0], swapped[1]);
    assert(common_lora_fingerprint(swapped) == fp);

    // ptr never an input: the same paths/scales with null ptrs hash alike
    auto nulled = inst_a->effective.lora_adapters;
    for (auto & la : nulled) {
        la.ptr = nullptr;
    }
    assert(common_lora_fingerprint(nulled) == fp);

    mgr.terminate();
    fs::remove_all(dir, ec);
}

// detach then re-attach reuses the surviving registry entry: the ptr is
// identical (no reload) while another instance holds its ref.
static void test_pool_adapter_detach_reattach(const common_params & base) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const std::string dir = (fs::temp_directory_path(ec) / "llama-m3-reattach").string();
    fs::remove_all(dir, ec);
    std::string file1;
    std::string file2;
    test_write_lora_pair(base, dir, file1, file2);

    common_params params = base;
    params.n_ctx      = 256;
    params.n_parallel = 1;
    params.warmup     = false;

    server_instances mgr;
    assert(mgr.load(params));

    const int64_t deadline = ggml_time_ms() + 5000;
    assert(mgr.handle_post_instances(test_req({}, "/instances",
        safe_json_to_str({ { "name", "a" }, { "ctx_size", 256 } })))->status == 201);
    assert(mgr.handle_post_instances(test_req({}, "/instances",
        safe_json_to_str({ { "name", "b" }, { "ctx_size", 256 } })))->status == 201);
    assert(mgr.handle_post_instance_adapters(test_req({ { "name", "a" } },
        "/instances/a/adapters", safe_json_to_str({ { "path", file1 } })))->status == 200);
    assert(mgr.handle_post_instance_adapters(test_req({ { "name", "b" } },
        "/instances/b/adapters", safe_json_to_str({ { "path", file1 } })))->status == 200);

    auto inst_a = test_find(mgr, "a");
    assert(inst_a && inst_a->built);
    llama_adapter_lora * before = inst_a->effective.lora_adapters[0].ptr;
    assert(before != nullptr);

    assert(mgr.handle_delete_instance_adapters(test_req({ { "name", "a" } },
        "/instances/a/adapters", safe_json_to_str({ { "path", file1 } })))->status == 200);
    auto get = mgr.handle_get_instance_adapters(test_req({ { "name", "a" } }, "/instances/a/adapters", ""));
    assert(get->status == 200 && json::parse(get->data).empty());

    // b still holds its ref, so the entry survives and the re-attach reuses it
    assert(mgr.handle_post_instance_adapters(test_req({ { "name", "a" } },
        "/instances/a/adapters", safe_json_to_str({ { "path", file1 } })))->status == 200);
    assert(inst_a->effective.lora_adapters.size() == 1);
    assert(inst_a->effective.lora_adapters[0].ptr == before);
    auto installed = inst_a->ctx_server->get_lora_adapters(deadline);
    assert(installed.has_value() && installed->size() == 1);
    assert((*installed)[0].ptr == before);

    mgr.terminate();
    fs::remove_all(dir, ec);
}

// a scale-only re-attach updates the scale in place: no refcount bump, no
// duplicate entry, and the scheduler serves the new scale. pins the swap
// core's acquired/release pairing for the update path.
static void test_pool_adapter_scale_update_no_bump(const common_params & base) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const std::string dir = (fs::temp_directory_path(ec) / "llama-m8-scale").string();
    fs::remove_all(dir, ec);
    std::string file1;
    std::string file2;
    test_write_lora_pair(base, dir, file1, file2);

    common_params params = base;
    params.n_ctx      = 256;
    params.n_parallel = 1;
    params.warmup     = false;

    server_instances mgr;
    assert(mgr.load(params));

    assert(mgr.handle_post_instances(test_req({}, "/instances",
        safe_json_to_str({ { "name", "a" }, { "ctx_size", 256 } })))->status == 201);
    assert(mgr.handle_post_instance_adapters(test_req({ { "name", "a" } },
        "/instances/a/adapters", safe_json_to_str({ { "path", file1 }, { "scale", 0.5 } })))->status == 200);

    auto inst_a = test_find(mgr, "a");
    assert(inst_a && inst_a->built);
    assert(inst_a->effective.lora_adapters.size() == 1);
    assert(mgr.adapter_registry.size() == 1);
    assert(mgr.adapter_registry.begin()->second.refcount == 1);

    // same file, new scale: still one entry, still one ref
    assert(mgr.handle_post_instance_adapters(test_req({ { "name", "a" } },
        "/instances/a/adapters", safe_json_to_str({ { "path", file1 }, { "scale", 1.0 } })))->status == 200);
    assert(inst_a->effective.lora_adapters.size() == 1);
    assert(inst_a->effective.lora_adapters[0].scale == 1.0f);
    assert(mgr.adapter_registry.size() == 1);
    assert(mgr.adapter_registry.begin()->second.refcount == 1);

    // detach drops to zero and frees the entry while the model stays alive
    assert(mgr.handle_delete_instance_adapters(test_req({ { "name", "a" } },
        "/instances/a/adapters", safe_json_to_str({ { "path", file1 } })))->status == 200);
    assert(inst_a->effective.lora_adapters.empty());
    assert(mgr.adapter_registry.empty());

    mgr.terminate();
    fs::remove_all(dir, ec);
}

// a detach racing a snapshot save: both serialize on the management mutex, so
// the fingerprint read never races the set mutation (TSan asserts this).
static void test_pool_adapter_concurrent_detach_snapshot(const common_params & base) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const std::string dir = (fs::temp_directory_path(ec) / "llama-m3-race").string();
    fs::remove_all(dir, ec);
    std::string file1;
    std::string file2;
    test_write_lora_pair(base, dir, file1, file2);

    common_params params = base;
    params.n_ctx          = 256;
    params.n_parallel     = 1;
    params.warmup         = false;
    params.slot_save_path = (fs::path(dir) / "slots").string() + "/";

    server_instances mgr;
    assert(mgr.load(params));

    assert(mgr.handle_post_instances(test_req({}, "/instances",
        safe_json_to_str({ { "name", "a" }, { "ctx_size", 256 } })))->status == 201);
    assert(mgr.handle_post_instance_adapters(test_req({ { "name", "a" } },
        "/instances/a/adapters", safe_json_to_str({ { "path", file1 } })))->status == 200);
    mgr.start_loops();

    std::atomic<int> saves_ok{0};
    std::thread saver([&]() {
        for (int i = 0; i < 20; i++) {
            auto res = mgr.handle_post_instance_snapshot(test_req({ { "name", "a" } },
                "/instances/a/snapshot", safe_json_to_str({ { "name", "w1" } })));
            if (res->status == 201) {
                saves_ok++;
            }
        }
    });
    auto detach = mgr.handle_delete_instance_adapters(test_req({ { "name", "a" } },
        "/instances/a/adapters", safe_json_to_str({ { "path", file1 } })));
    assert(detach->status == 200);
    saver.join();
    assert(saves_ok.load() == 20);

    mgr.terminate();
    fs::remove_all(dir, ec);
}

// resize rebuilds the window from the pool: the installed set is the same
// borrowed entry, not a reload. a failing factory maps to the adapter 400.
static void test_pool_adapter_resize_borrowed(const common_params & base) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const std::string dir = (fs::temp_directory_path(ec) / "llama-m3-resize").string();
    fs::remove_all(dir, ec);
    std::string file1;
    std::string file2;
    test_write_lora_pair(base, dir, file1, file2);

    common_params params = base;
    params.n_ctx      = 256;
    params.n_parallel = 1;
    params.warmup     = false;

    common_instance cfga;
    cfga.name        = "a";
    cfga.group       = "a";
    cfga.ctx_size    = 256;
    cfga.parallel    = 1;
    cfga.is_default  = true;
    params.instances.push_back(cfga);

    server_instances mgr;
    assert(mgr.load(params));

    static const std::function<bool()> no_stop = []() { return false; };
    server_http_req props_req { {}, {}, "/props", "", "", {}, no_stop };
    assert(mgr.handle_get_props(props_req)->status == 200);

    const int64_t deadline = ggml_time_ms() + 5000;
    assert(mgr.handle_post_instance_adapters(test_req({ { "name", "a" } },
        "/instances/a/adapters", safe_json_to_str({ { "path", file1 } })))->status == 200);

    auto inst_a = test_find(mgr, "a");
    assert(inst_a && inst_a->built);
    llama_adapter_lora * before = inst_a->effective.lora_adapters[0].ptr;
    assert(before != nullptr);

    auto resize = [&](int32_t n_ctx) {
        return mgr.handle_post_instance_resize(test_req({ { "name", "a" } },
            "/instances/a/resize", safe_json_to_str({ { "ctx_size", n_ctx } })));
    };
    assert(resize(512)->status == 200);
    assert(inst_a->built);
    assert(inst_a->effective.lora_adapters.size() == 1);
    assert(inst_a->effective.lora_adapters[0].ptr == before);
    auto installed = inst_a->ctx_server->get_lora_adapters(deadline);
    assert(installed.has_value() && installed->size() == 1);
    assert((*installed)[0].ptr == before);

    // fail-to-load factory: the adapter error maps to 400, never 507
    mgr.set_context_builder([](server_instance & inst) {
        inst.adapter_failed = true;
        return false;
    });
    assert(resize(1024)->status == 400);
    assert(!inst_a->built);

    // recovery re-resolves from the declaration on the next demand
    mgr.set_context_builder(nullptr);
    assert(resize(256)->status == 200);
    assert(mgr.handle_get_props(props_req)->status == 200);
    assert(inst_a->built);
    assert(inst_a->effective.lora_adapters.size() == 1);
    assert(inst_a->effective.lora_adapters[0].ptr != nullptr);
    installed = inst_a->ctx_server->get_lora_adapters(deadline);
    assert(installed.has_value() && installed->size() == 1);
    assert((*installed)[0].ptr == inst_a->effective.lora_adapters[0].ptr);

    mgr.terminate();
    fs::remove_all(dir, ec);
}

// byte identities with an attached adapter: total adds every leg including
// adapters, vram stays context+compute, and the envelope totals agree.
static void test_adapter_bytes_identities(const common_params & base) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const std::string dir = (fs::temp_directory_path(ec) / "llama-m5-ident").string();
    fs::remove_all(dir, ec);
    std::string file1;
    std::string file2;
    test_write_lora_pair(base, dir, file1, file2);

    common_params params = base;
    params.n_ctx      = 256;
    params.n_parallel = 1;
    params.warmup     = false;

    common_instance cfga;
    cfga.name        = "a";
    cfga.group       = "a";
    cfga.ctx_size    = 256;
    cfga.parallel    = 1;
    cfga.is_default  = true;
    cfga.lora.emplace_back(file1, 1.0f);
    params.instances.push_back(cfga);

    server_instances mgr;
    assert(mgr.load(params));

    static const std::function<bool()> no_stop = []() { return false; };
    server_http_req props_req { {}, {}, "/props", "", "", {}, no_stop };
    assert(mgr.handle_get_props(props_req)->status == 200);

    server_http_req list_req { {}, {}, "/instances", "", "", {}, no_stop };
    auto res = mgr.handle_get_instances(list_req);
    assert(res->status == 200);
    const json body = json::parse(res->data);
    assert(body["instances"].size() == 1);
    const json row = body["instances"][0];

    const uint64_t m  = row["model_bytes"].get<uint64_t>();
    const uint64_t cx = row["context_bytes"].get<uint64_t>();
    const uint64_t co = row["compute_bytes"].get<uint64_t>();
    const uint64_t ad = row["adapter_bytes"].get<uint64_t>();
    assert(ad > 0);
    assert(ad == llama_adapter_lora_buf_size(test_find(mgr, "a")->effective.lora_adapters[0].ptr));
    assert(row["total_bytes"] == m + cx + co + ad);
    assert(row["vram_bytes"] == cx + co);

    const json total = body["total"];
    assert(total["adapter"].get<uint64_t>() >= ad);
    assert(total["total"] == total["model"].get<uint64_t>() + total["context"].get<uint64_t>() +
                             total["compute"].get<uint64_t>() + total["adapter"].get<uint64_t>());

    mgr.terminate();
    fs::remove_all(dir, ec);
}

// an unbuilt window owns no buffers: every byte field reads zero, even with
// declared adapters or an attach recorded while unbuilt.
static void test_adapter_bytes_unbuilt_zero(const common_params & base) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const std::string dir = (fs::temp_directory_path(ec) / "llama-m5-unbuilt").string();
    fs::remove_all(dir, ec);
    std::string file1;
    std::string file2;
    test_write_lora_pair(base, dir, file1, file2);

    common_params params = base;
    params.n_ctx      = 256;
    params.n_parallel = 1;
    params.warmup     = false;

    common_instance cfga;
    cfga.name     = "a";
    cfga.group    = "a";
    cfga.ctx_size = 256;
    cfga.parallel = 1;
    cfga.lora.emplace_back(file1, 1.0f);
    params.instances.push_back(cfga);

    server_instances mgr;
    assert(mgr.load(params));

    // attach while unbuilt: declaration only, still no buffers
    assert(mgr.handle_post_instance_adapters(test_req({ { "name", "a" } },
        "/instances/a/adapters", safe_json_to_str({ { "path", file2 } })))->status == 200);

    static const std::function<bool()> no_stop = []() { return false; };
    server_http_req list_req { {}, {}, "/instances", "", "", {}, no_stop };
    auto res = mgr.handle_get_instances(list_req);
    assert(res->status == 200);
    const json body = json::parse(res->data);
    assert(body["instances"].size() == 1);
    const json row = body["instances"][0];
    assert(row["state"] == "unloaded");
    assert(row["model_bytes"] == 0 && row["context_bytes"] == 0);
    assert(row["compute_bytes"] == 0 && row["adapter_bytes"] == 0);
    assert(row["total_bytes"] == 0 && row["vram_bytes"] == 0);

    mgr.terminate();
    fs::remove_all(dir, ec);
}

// pool sharing counts per row but once per pool: two instances on one file
// each report the full entry size, the envelope adapter total counts one file.
static void test_adapter_bytes_pool_shared(const common_params & base) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const std::string dir = (fs::temp_directory_path(ec) / "llama-m5-pool").string();
    fs::remove_all(dir, ec);
    std::string file1;
    std::string file2;
    test_write_lora_pair(base, dir, file1, file2);

    common_params params = base;
    params.n_ctx      = 256;
    params.n_parallel = 1;
    params.warmup     = false;

    server_instances mgr;
    assert(mgr.load(params));

    assert(mgr.handle_post_instances(test_req({}, "/instances",
        safe_json_to_str({ { "name", "a" }, { "ctx_size", 256 } })))->status == 201);
    assert(mgr.handle_post_instances(test_req({}, "/instances",
        safe_json_to_str({ { "name", "b" }, { "ctx_size", 256 } })))->status == 201);
    assert(mgr.handle_post_instance_adapters(test_req({ { "name", "a" } },
        "/instances/a/adapters", safe_json_to_str({ { "path", file1 } })))->status == 200);
    assert(mgr.handle_post_instance_adapters(test_req({ { "name", "b" } },
        "/instances/b/adapters", safe_json_to_str({ { "path", file1 } })))->status == 200);

    const uint64_t entry = llama_adapter_lora_buf_size(test_find(mgr, "a")->effective.lora_adapters[0].ptr);
    assert(entry > 0);

    static const std::function<bool()> no_stop = []() { return false; };
    server_http_req list_req { {}, {}, "/instances", "", "", {}, no_stop };
    const json body = json::parse(mgr.handle_get_instances(list_req)->data);
    // the legacy default (built at load with no adapters) plus a and b
    assert(body["instances"].size() == 3);
    for (const auto & row : body["instances"]) {
        const std::string id = row["id"].get<std::string>();
        if (id.size() >= 2 && (id.compare(id.size() - 2, 2, ":a") == 0 ||
                               id.compare(id.size() - 2, 2, ":b") == 0)) {
            assert(row["adapter_bytes"] == entry);
        } else {
            assert(row["adapter_bytes"] == 0);
        }
    }
    assert(body["total"]["adapter"] == entry);

    mgr.terminate();
    fs::remove_all(dir, ec);
}

// allocation failure (never adapter failure) answers 507 with the device
// memory text on both create and resize.
static void test_oom_text_create_resize(const common_params & base) {
    common_params params = base;
    params.n_ctx      = 256;
    params.n_parallel = 1;
    params.warmup     = false;

    common_instance cfga;
    cfga.name     = "a";
    cfga.group    = "a";
    cfga.ctx_size = 256;
    cfga.parallel = 1;
    params.instances.push_back(cfga);

    server_instances mgr;
    mgr.set_context_builder([](server_instance &) { return false; });
    assert(mgr.load(params));

    auto create = mgr.handle_post_instances(test_req({}, "/instances",
        safe_json_to_str({ { "name", "b" }, { "ctx_size", 256 } })));
    assert(create->status == 507);
    assert(create->data.find("not enough device memory") != std::string::npos);

    mgr.set_context_builder(nullptr);
    static const std::function<bool()> no_stop = []() { return false; };
    server_http_req props_req { {}, {}, "/props", "", "", {}, no_stop };
    assert(mgr.handle_get_props(props_req)->status == 200);

    mgr.set_context_builder([](server_instance &) { return false; });
    auto resize = mgr.handle_post_instance_resize(test_req({ { "name", "a" } },
        "/instances/a/resize", safe_json_to_str({ { "ctx_size", 512 } })));
    assert(resize->status == 507);
    assert(resize->data.find("not enough device memory") != std::string::npos);

    mgr.terminate();
}

// test-only: write a version-1 snapshot file (no adapter_fp field) with the
// given payload. the layout is the documented on-disk contract: magic, version,
// n_ctx_seq, n_tokens, kv_size, tokens, kv.
static void test_write_snapshot_v1(const std::string & path, int32_t n_ctx_seq,
                                   const llama_tokens & tokens, const std::vector<uint8_t> & kv) {
    std::ofstream out(path, std::ios::binary);
    assert(out.is_open());
    const uint32_t magic   = 0x534C5041;
    const uint32_t version = 1;
    const int32_t  n_tokens = (int32_t) tokens.size();
    const uint64_t kv_size  = (uint64_t) kv.size();
    out.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char *>(&version), sizeof(version));
    out.write(reinterpret_cast<const char *>(&n_ctx_seq), sizeof(n_ctx_seq));
    out.write(reinterpret_cast<const char *>(&n_tokens), sizeof(n_tokens));
    out.write(reinterpret_cast<const char *>(&kv_size), sizeof(kv_size));
    out.write(reinterpret_cast<const char *>(tokens.data()), n_tokens * (std::streamsize) sizeof(llama_token));
    out.write(reinterpret_cast<const char *>(kv.data()), (std::streamsize) kv.size());
    out.close();
    assert(out.good());
}

// spelling normalization: two spellings of one file resolve to one registry
// entry (same ptr while held, one row) and one fingerprint at equal scales.
static void test_instances_lora_normalize(const common_params & base) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const std::string dir = (fs::temp_directory_path(ec) / "llama-m7-norm").string();
    fs::remove_all(dir, ec);
    std::string file1;
    std::string file2;
    test_write_lora_pair(base, dir, file1, file2);

    common_params params = base;
    params.n_ctx      = 256;
    params.n_parallel = 1;
    params.warmup     = false;

    server_instances mgr;
    assert(mgr.load(params));

    assert(mgr.handle_post_instances(test_req({}, "/instances",
        safe_json_to_str({ { "name", "a" }, { "ctx_size", 256 } })))->status == 201);

    // second spelling of the same file: the dot-segment collapses to file1
    const std::string alt = (fs::path(dir) / "." / "m3-a.gguf").string();
    assert(mgr.handle_post_instance_adapters(test_req({ { "name", "a" } },
        "/instances/a/adapters", safe_json_to_str({ { "path", file1 } })))->status == 200);
    auto inst_a = test_find(mgr, "a");
    assert(inst_a && inst_a->built);
    llama_adapter_lora * first_ptr = inst_a->effective.lora_adapters[0].ptr;
    assert(first_ptr != nullptr);
    const std::string first_fp = common_lora_fingerprint(inst_a->effective.lora_adapters);
    assert(!first_fp.empty());

    // re-attach under the other spelling: the same entry, scale updated, one row
    assert(mgr.handle_post_instance_adapters(test_req({ { "name", "a" } },
        "/instances/a/adapters", safe_json_to_str({ { "path", alt }, { "scale", 0.5f } })))->status == 200);
    assert(inst_a->effective.lora_adapters.size() == 1);
    assert(inst_a->effective.lora_adapters[0].ptr == first_ptr);
    assert(inst_a->effective.lora_adapters[0].scale == 0.5f);
    auto get = mgr.handle_get_instance_adapters(test_req({ { "name", "a" } }, "/instances/a/adapters", ""));
    assert(get->status == 200 && json::parse(get->data).size() == 1);

    // same spelling-independent fingerprint at equal scales
    assert(mgr.handle_delete_instance_adapters(test_req({ { "name", "a" } },
        "/instances/a/adapters", safe_json_to_str({ { "path", file1 } })))->status == 200);
    assert(mgr.handle_post_instance_adapters(test_req({ { "name", "a" } },
        "/instances/a/adapters", safe_json_to_str({ { "path", alt } })))->status == 200);
    assert(inst_a->effective.lora_adapters.size() == 1);
    assert(common_lora_fingerprint(inst_a->effective.lora_adapters) == first_fp);

    mgr.terminate();
    fs::remove_all(dir, ec);
}

// setup: pool with a slot-save dir, one built instance with file1 attached,
// loops (scheduler + pool I/O worker) running. returns the adapter file path.
static std::string test_m6_setup(server_instances & mgr, const common_params & base, const std::string & dir) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::remove_all(dir, ec);
    std::string file1;
    std::string file2;
    test_write_lora_pair(base, dir, file1, file2);

    common_params params = base;
    params.n_ctx          = 256;
    params.n_parallel     = 1;
    params.warmup         = false;
    params.slot_save_path = (fs::path(dir) / "slots").string() + "/";
    assert(mgr.load(params));

    assert(mgr.handle_post_instances(test_req({}, "/instances",
        safe_json_to_str({ { "name", "a" }, { "ctx_size", 256 } })))->status == 201);
    assert(mgr.handle_post_instance_adapters(test_req({ { "name", "a" } },
        "/instances/a/adapters", safe_json_to_str({ { "path", file1 } })))->status == 200);
    mgr.start_loops();
    return file1;
}

static server_http_res_ptr test_m6_switch(server_instances & mgr, const std::string & snapshot) {
    auto forward = [](server_routes & routes, const server_http_req & req) {
        return routes.get_props(req);
    };
    return mgr.dispatch(test_req({}, "/completion", safe_json_to_str({
        { "model",    mgr.base_name + ":a" },
        { "snapshot", snapshot },
    })), forward);
}

// the save stamps the live adapter fingerprint into the file.
static void test_snapshot_records_fp(const common_params & base) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const std::string dir = (fs::temp_directory_path(ec) / "llama-m6-record").string();

    server_instances mgr;
    test_m6_setup(mgr, base, dir);

    assert(mgr.handle_post_instance_snapshot(test_req({ { "name", "a" } },
        "/instances/a/snapshot", safe_json_to_str({ { "name", "w1" } })))->status == 201);

    auto inst = test_find(mgr, "a");
    assert(inst && inst->built);
    const std::string live_fp = common_lora_fingerprint(inst->effective.lora_adapters);
    assert(!live_fp.empty());

    auto st = server_snapshot_read_status(mgr.snapshot_instance_path("a", "w1"));
    assert(st.status == server_snapshot_status::OK && st.data.has_value());
    assert(st.data->adapter_fp == live_fp);

    mgr.terminate();
    fs::remove_all(dir, ec);
}

// a snapshot saved under a different adapter set is rejected on switch.
static void test_snapshot_mismatch_400(const common_params & base) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const std::string dir = (fs::temp_directory_path(ec) / "llama-m6-mismatch").string();

    server_instances mgr;
    test_m6_setup(mgr, base, dir);

    assert(mgr.handle_post_instance_snapshot(test_req({ { "name", "a" } },
        "/instances/a/snapshot", safe_json_to_str({ { "name", "w1" } })))->status == 201);

    // w2 carries w1's valid KV under a foreign fingerprint
    auto st = server_snapshot_read_status(mgr.snapshot_instance_path("a", "w1"));
    assert(st.status == server_snapshot_status::OK && st.data.has_value());
    server_snapshot_data foreign = std::move(*st.data);
    foreign.adapter_fp = "0123456789abcdef";
    assert(server_snapshot_write(mgr.snapshot_instance_path("a", "w2"), foreign));

    auto res = test_m6_switch(mgr, "w2");
    assert(res->status == 400);
    assert(res->data.find("adapter") != std::string::npos);

    mgr.terminate();
    fs::remove_all(dir, ec);
}

// a pre-v2 file (no fingerprint at all) restores with a warning, not an error.
static void test_snapshot_prev2_allows(const common_params & base) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const std::string dir = (fs::temp_directory_path(ec) / "llama-m6-prev2").string();

    server_instances mgr;
    test_m6_setup(mgr, base, dir);

    assert(mgr.handle_post_instance_snapshot(test_req({ { "name", "a" } },
        "/instances/a/snapshot", safe_json_to_str({ { "name", "w1" } })))->status == 201);

    // w3: w1's valid KV as a real v1 file (no fp field on disk)
    auto st = server_snapshot_read_status(mgr.snapshot_instance_path("a", "w1"));
    assert(st.status == server_snapshot_status::OK && st.data.has_value());
    test_write_snapshot_v1(mgr.snapshot_instance_path("a", "w3"),
                           st.data->n_ctx_seq, st.data->tokens, st.data->kv);
    auto check = server_snapshot_read_status(mgr.snapshot_instance_path("a", "w3"));
    assert(check.status == server_snapshot_status::OK && check.data.has_value());
    assert(check.data->adapter_fp.empty());

    auto res = test_m6_switch(mgr, "w3");
    assert(res->status == 200);

    mgr.terminate();
    fs::remove_all(dir, ec);
}

// adapter-free variant of the m6 pool: the same window with no adapters, so
// the live fingerprint is empty ("") - the exact-match target of a v2-empty file.
static void test_snapshot_no_adapter_setup(server_instances & mgr, const common_params & base, const std::string & dir) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::remove_all(dir, ec);

    common_params params = base;
    params.n_ctx          = 256;
    params.n_parallel     = 1;
    params.warmup         = false;
    params.slot_save_path = (fs::path(dir) / "slots").string() + "/";
    assert(mgr.load(params));

    assert(mgr.handle_post_instances(test_req({}, "/instances",
        safe_json_to_str({ { "name", "a" }, { "ctx_size", 256 } })))->status == 201);
    mgr.start_loops();
}

// a v2 file saved with no adapters restores onto an adapter-free window:
// empty matches empty. golden before and after the strict gate.
static void test_snapshot_v2_empty_on_empty_allows(const common_params & base) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const std::string dir = (fs::temp_directory_path(ec) / "llama-m9-empty-ok").string();

    server_instances mgr;
    test_snapshot_no_adapter_setup(mgr, base, dir);

    assert(mgr.handle_post_instance_snapshot(test_req({ { "name", "a" } },
        "/instances/a/snapshot", safe_json_to_str({ { "name", "w1" } })))->status == 201);
    assert(mgr.handle_post_instance_snapshot(test_req({ { "name", "a" } },
        "/instances/a/snapshot", safe_json_to_str({ { "name", "w2" } })))->status == 201);

    auto st = server_snapshot_read_status(mgr.snapshot_instance_path("a", "w1"));
    assert(st.status == server_snapshot_status::OK && st.data.has_value());
    assert(st.data->adapter_fp.empty());

    assert(test_m6_switch(mgr, "w1")->status == 200);
    assert(test_m6_switch(mgr, "w2")->status == 200);

    mgr.terminate();
    fs::remove_all(dir, ec);
}

// a v2 file saved with no adapters must NOT restore onto an adapter-equipped
// window: the empty fingerprint means "zero adapters", which mismatches.
static void test_snapshot_v2_empty_on_nonempty_400s(const common_params & base) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const std::string dir = (fs::temp_directory_path(ec) / "llama-m9-empty-strict").string();

    server_instances mgr;
    test_m6_setup(mgr, base, dir);

    assert(mgr.handle_post_instance_snapshot(test_req({ { "name", "a" } },
        "/instances/a/snapshot", safe_json_to_str({ { "name", "w1" } })))->status == 201);

    // w4: w1's valid KV re-stamped empty (a v2 file with fp_len 0)
    auto st = server_snapshot_read_status(mgr.snapshot_instance_path("a", "w1"));
    assert(st.status == server_snapshot_status::OK && st.data.has_value());
    assert(!st.data->adapter_fp.empty());
    server_snapshot_data no_fp = std::move(*st.data);
    no_fp.adapter_fp.clear();
    assert(server_snapshot_write(mgr.snapshot_instance_path("a", "w4"), no_fp));

    auto res = test_m6_switch(mgr, "w4");
    assert(res->status == 400);
    assert(res->data.find("adapter") != std::string::npos);

    mgr.terminate();
    fs::remove_all(dir, ec);
}

// adapter drift between save and restore is rejected: save attached, detach,
// then restoring the attached-fp snapshot fails.
static void test_snapshot_drift_400(const common_params & base) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const std::string dir = (fs::temp_directory_path(ec) / "llama-m6-drift").string();

    server_instances mgr;
    const std::string file1 = test_m6_setup(mgr, base, dir);

    assert(mgr.handle_post_instance_snapshot(test_req({ { "name", "a" } },
        "/instances/a/snapshot", safe_json_to_str({ { "name", "w1" } })))->status == 201);
    assert(mgr.handle_delete_instance_adapters(test_req({ { "name", "a" } },
        "/instances/a/adapters", safe_json_to_str({ { "path", file1 } })))->status == 200);

    auto res = test_m6_switch(mgr, "w1");
    assert(res->status == 400);
    assert(res->data.find("adapter") != std::string::npos);

    mgr.terminate();
    fs::remove_all(dir, ec);
}

// live aggregate parity: two live managers (one with a real attached adapter)
// plus a mock child envelope advertising adapter bytes, merged through the
// production router merge. merged adapter is the sum; merged total is the sum
// of the adapter-inclusive child totals, never re-added.
static void test_live_merge_adapter_parity(const common_params & base) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const std::string dir = (fs::temp_directory_path(ec) / "llama-m13-live").string();
    fs::remove_all(dir, ec);
    std::string file1;
    std::string file2;
    test_write_lora_pair(base, dir, file1, file2);

    static const std::function<bool()> no_stop = []() { return false; };
    server_http_req list_req { {}, {}, "/instances", "", "", {}, no_stop };

    // live server one: a window with a real attached adapter
    common_params params_a = base;
    params_a.n_ctx      = 256;
    params_a.n_parallel = 1;
    params_a.warmup     = false;
    server_instances mgr_a;
    assert(mgr_a.load(params_a));
    assert(mgr_a.handle_post_instances(test_req({}, "/instances",
        safe_json_to_str({ { "name", "a" }, { "ctx_size", 256 } })))->status == 201);
    assert(mgr_a.handle_post_instance_adapters(test_req({ { "name", "a" } },
        "/instances/a/adapters", safe_json_to_str({ { "path", file1 } })))->status == 200);
    const json env_a = json::parse(mgr_a.handle_get_instances(list_req)->data);
    assert(env_a["total"]["adapter"].get<uint64_t>() > 0);

    // live server two: a plain window, no adapters
    common_params params_b = base;
    params_b.n_ctx      = 256;
    params_b.n_parallel = 1;
    params_b.warmup     = false;
    server_instances mgr_b;
    assert(mgr_b.load(params_b));
    assert(mgr_b.handle_post_instances(test_req({}, "/instances",
        safe_json_to_str({ { "name", "b" }, { "ctx_size", 256 } })))->status == 201);
    const json env_b = json::parse(mgr_b.handle_get_instances(list_req)->data);
    assert(env_b["total"]["adapter"] == 0);

    // mock router child: a canned envelope advertising adapter bytes
    const json env_m = {
        { "instances", json::array({ json{ { "id", "mock:m" } } }) },
        { "snapshots", json::array({ json{ { "name", "s1" } } }) },
        { "total",     json{ { "model", 50 }, { "context", 6 }, { "compute", 3 }, { "adapter", 40 }, { "total", 99 } } },
    };

    const json out = server_models_merge_instances({
        { mgr_a.base_name, env_a },
        { mgr_b.base_name, env_b },
        { "mock",          env_m },
    });

    const uint64_t ad_a = env_a["total"]["adapter"].get<uint64_t>();
    const uint64_t t_a  = env_a["total"]["total"].get<uint64_t>();
    const uint64_t t_b  = env_b["total"]["total"].get<uint64_t>();
    const json total = out["total"];
    assert(total["adapter"] == ad_a + 40);
    assert(total["total"] == t_a + t_b + 99);
    assert(total["total"] == total["model"].get<uint64_t>() + total["context"].get<uint64_t>() +
                             total["compute"].get<uint64_t>() + total["adapter"].get<uint64_t>());
    // rows: live rows pass through, the mock snapshot gains its model tag
    assert(out["instances"].size() == env_a["instances"].size() + env_b["instances"].size() + 1);
    assert(out["snapshots"].size() == 1);
    assert(out["snapshots"][0]["model"] == "mock");

    mgr_a.terminate();
    mgr_b.terminate();
    fs::remove_all(dir, ec);
}

// calibration matrix for the aggregate fan-out scheduling policy. the
// collector takes an injectable fetch, so the policy is measured directly with
// no HTTP. every number is caller latency (task-value), never child health;
// nothing is cached or persisted, and a skipped child is simply absent.
static void test_fanout_calibration() {
    // fan-out width 2 keeps the batch schedule deterministic with small K
    setenv("LLAMA_SERVER_TEST_AGG_FANOUT_MAX", "2", 1);

    auto make_targets = [](int k) {
        std::vector<server_model_meta> targets;
        for (int i = 0; i < k; ++i) {
            server_model_meta m;
            // zero-padded so name sort == index order
            m.name = "c" + std::to_string(100 + i);
            targets.push_back(m);
        }
        return targets;
    };

    // healthy control: K fast children, none may be skipped, output sorted
    {
        const int     K        = 8;
        const int64_t fetch_ms = 15;
        auto          targets  = make_targets(K);
        instances_fetch_fn fetch = [fetch_ms](const server_model_meta & m) -> std::optional<std::pair<std::string, json>> {
            std::this_thread::sleep_for(std::chrono::milliseconds(fetch_ms));
            return std::make_pair(m.name, json{ { "id", m.name } });
        };

        std::vector<double> lat;
        lat.reserve(50);
        for (int run = 0; run < 50; ++run) {
            const int64_t t0 = ggml_time_us();
            auto          rows = instances_fanout_collect(targets, fetch, ggml_time_ms() + 5000);
            lat.push_back((ggml_time_us() - t0) / 1000.0);

            assert((int) rows.size() == K);  // skip-precision 1.0
            for (int i = 0; i < K; ++i) {
                assert(rows[i].first == targets[i].name);  // deterministic sort
            }
        }
        std::sort(lat.begin(), lat.end());
        const double p50    = lat[lat.size() / 2];
        const double p95    = lat[(size_t) (lat.size() * 0.95)];
        const double serial = K * fetch_ms;
        assert(p95 <= serial * 1.20);
        fprintf(stdout, "fanout healthy control: K=%d p50=%.2f ms p95=%.2f ms serial=%.2f ms\n", K, p50, p95, serial);
    }

    // one poisoned child: emulates the real socket timeout by sleeping past the
    // per-child cap and returning nullopt; exactly one absence, K-1 live rows
    {
        const int K       = 4;
        auto      targets = make_targets(K);
        instances_fetch_fn fetch = [](const server_model_meta & m) -> std::optional<std::pair<std::string, json>> {
            if (m.name == "c101") {
                std::this_thread::sleep_for(std::chrono::milliseconds(300));
                return std::nullopt;  // timed out
            }
            return std::make_pair(m.name, json{ { "id", m.name } });
        };
        auto rows = instances_fanout_collect(targets, fetch, ggml_time_ms() + 5000);
        assert(rows.size() == K - 1);  // skip-recall 1.0
        for (const auto & r : rows) {
            assert(r.first != "c101");
        }
        fprintf(stdout, "fanout poisoned child: %zu/%d live rows, one absence\n", rows.size(), K);
    }

    // budget exhaustion: a fetch that ignores the per-child cap and sleeps past
    // the global budget; the call must return within budget + one per-child
    // timeout + slack, and must never hang
    {
        const int     K         = 4;
        const int64_t budget_ms = 100;
        const int64_t slow_ms   = 500;
        auto          targets   = make_targets(K);
        instances_fetch_fn fetch = [slow_ms](const server_model_meta & m) -> std::optional<std::pair<std::string, json>> {
            if (m.name == "c101") {
                std::this_thread::sleep_for(std::chrono::milliseconds(slow_ms));
            }
            return std::make_pair(m.name, json{ { "id", m.name } });
        };
        const int64_t t0      = ggml_time_us();
        auto          rows    = instances_fanout_collect(targets, fetch, ggml_time_ms() + budget_ms);
        const double  elapsed = (ggml_time_us() - t0) / 1000.0;
        // no batch after the budget, but an in-flight read cannot be cancelled
        assert(elapsed <= budget_ms + slow_ms + 250);
        bool has_fast = false;
        for (const auto & r : rows) {
            has_fast = has_fast || r.first == "c100";
        }
        assert(has_fast);
        fprintf(stdout, "fanout budget exhaustion: elapsed=%.2f ms bound=%.2f ms\n", elapsed,
                (double) (budget_ms + slow_ms));
    }

    unsetenv("LLAMA_SERVER_TEST_AGG_FANOUT_MAX");
}

// /v1/models uses the same per-member skip-and-continue policy as /slots. a
// failing built member yields a marker row instead of failing the whole
// aggregate; the healthy envelope and the slots path are unchanged.
static void test_models_skip_and_continue(const common_params & base) {
    common_params params = base;
    params.n_ctx          = 256;
    params.n_parallel     = 1;
    params.warmup         = false;
    params.endpoint_slots = true;
    for (const char * name : { "a", "b" }) {
        common_instance cfg;
        cfg.name     = name;
        cfg.group    = name;
        cfg.ctx_size = 256;
        cfg.parallel = 1;
        params.instances.push_back(cfg);
    }

    server_instances mgr;
    assert(mgr.load(params));

    static const std::function<bool()> no_stop = []() { return false; };
    auto demand = [&](const char * name) {
        server_http_req req { { { "instance", name } }, {}, "/slots", "", "", {}, no_stop };
        assert(mgr.handle_get_slots(req)->status == 200);
    };
    demand("a");
    demand("b");

    server_http_req models_req { {}, {}, "/models", "", "", {}, no_stop };

    // healthy envelope is unchanged
    auto healthy = mgr.handle_get_models(models_req);
    assert(healthy->status == 200);
    json h = json::parse(healthy->data);
    assert(h.contains("models") && h.contains("object") && h.contains("data"));
    assert(h["object"] == "list");
    assert(h["data"].size() == 2);
    for (auto & d : h["data"]) {
        assert(d["status"] == "loaded");
        assert(!d.contains("error"));
    }

    // force exactly one member's route to non-200
    mgr.aggregate_route_hook = [](server_instance & inst,
                                  const server_http_req &) -> std::optional<server_http_res_ptr> {
        if (inst.cfg.name == "b") {
            auto r  = std::make_unique<server_http_res>();
            r->status = 500;
            r->data   = safe_json_to_str({ { "error", { { "message", "forced" } } } });
            return r;
        }
        return std::nullopt;
    };

    auto partial = mgr.handle_get_models(models_req);
    assert(partial->status == 200);
    json p = json::parse(partial->data);
    assert(p["data"].size() == 2);
    int loaded = 0;
    int markers = 0;
    for (auto & d : p["data"]) {
        if (d.contains("error")) {
            ++markers;
            assert(d["instance"] == "b");
        } else {
            ++loaded;
            assert(d["status"] == "loaded");
        }
    }
    assert(loaded == 1 && markers == 1);
    assert(p["models"].size() >= 1);  // the healthy member's model entry survived

    // slots path unchanged: the same forced failure still yields a marker
    server_http_req slots_req { {}, {}, "/slots", "", "", {}, no_stop };
    auto slots = mgr.handle_get_slots(slots_req);
    assert(slots->status == 200);
    json s = json::parse(slots->data);
    int slot_markers = 0;
    int slot_rows    = 0;
    for (auto & row : s) {
        if (row.contains("error")) {
            ++slot_markers;
            assert(row["instance"] == "b");
        } else {
            ++slot_rows;
            assert(row.contains("instance"));
        }
    }
    assert(slot_markers == 1);
    assert(slot_rows >= 1);

    mgr.aggregate_route_hook = nullptr;
    mgr.terminate();
}

// GET /instances must not hold mutex_mgmt across the on-disk snapshot
// listing, or slow storage would serialize create/destroy/resize behind every
// router poll. a smoke guard: the listing is made slow with many snapshot files
// and a concurrent management op must not wait for it.
static void test_instances_listing_off_management_lock(const common_params & base) {
    namespace fs = std::filesystem;

    common_params params = base;
    params.n_ctx      = 256;
    params.n_parallel = 1;
    params.warmup     = false;

    common_instance cfg;
    cfg.name     = "a";
    cfg.group    = "a";
    cfg.ctx_size = 256;
    cfg.parallel = 1;
    params.instances.push_back(cfg);

    const std::string dir = (fs::temp_directory_path() / ("llama_m8_" + std::to_string(ggml_time_us()))).string();
    params.slot_save_path = dir + "/";

    server_instances mgr;
    assert(mgr.load(params));

    // populate the per-instance snapshot dir with many header-parse targets
    const fs::path snap_dir = fs::path(mgr.snapshot_instance_path("a", "probe")).parent_path();
    fs::create_directories(snap_dir);
    const int N = 4000;
    for (int i = 0; i < N; ++i) {
        std::ofstream out(snap_dir / ("s" + std::to_string(i) + ".bin"), std::ios::binary);
    }

    static const std::function<bool()> no_stop = []() { return false; };
    server_http_req list_req { {}, {}, "/instances", "", "", {}, no_stop };

    int64_t listing_ms = 0;
    std::thread lister([&]() {
        const int64_t t0 = ggml_time_us();
        auto          res = mgr.handle_get_instances(list_req);
        listing_ms        = (ggml_time_us() - t0) / 1000;
        assert(res->status == 200);
    });

    // let the lister acquire mutex_mgmt and enter the snapshot listing
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    const int64_t t0 = ggml_time_us();
    auto pin = mgr.handle_post_instance_pin(
        server_http_req { { { "name", "a" } }, {}, "/instances/a/pin", "", "", {}, no_stop });
    const int64_t pin_ms = (ggml_time_us() - t0) / 1000;
    assert(pin->status == 200);

    lister.join();

    // the listing must actually be the slow side of the race
    assert(listing_ms > 20);
    // the management op must not wait for the listing (half the listing is a
    // generous margin: without the fix the pin waits out the remaining listing)
    assert(pin_ms * 2 < listing_ms);
    fprintf(stdout, "instances listing off mgmt lock: listing=%lld ms pin=%lld ms\n", (long long) listing_ms,
            (long long) pin_ms);

    std::error_code ec;
    fs::remove_all(dir, ec);
    mgr.terminate();
}

// start_loops must start the pool I/O worker, so a server that advertises
// ready only after start_loops can never hand out the I/O-busy 503 to the first
// snapshot switch. observable through the manager: a post succeeds only once the
// worker runs.
static void test_start_loops_starts_io_worker() {
    server_instances mgr;
    assert(!mgr.snapshot_io_post([]() {}).has_value());  // no worker before the single start
    mgr.start_loops();
    auto fut = mgr.snapshot_io_post([]() {});
    assert(fut.has_value());
    fut->wait();
    fut->get();
    mgr.terminate();
}

// The session store's byte-budget policy, without a server, a model or a budget flag. The wire
// suite can only reach the evicting path when a pool holds more token bytes than the smallest
// expressible budget (--decision-session-budget-mb is in mebibytes of tokens.size() * 4, so one
// mebibyte is 262144 tokens), which a mid-size dense model cannot host. These cases therefore carry
// the ordering and the skip rules, and the wire suite carries the wiring.
static void test_decision_session_budget_policy() {
    const auto ref = [](size_t bytes, int64_t last_used_ms, bool pinned = false, bool leased = false,
                        bool removed = false, bool replacing = false) {
        decision_session_ref r;
        r.bytes        = bytes;
        r.last_used_ms = last_used_ms;
        r.pinned       = pinned;
        r.leased       = leased;
        r.removed      = removed;
        r.replacing    = replacing;
        return r;
    };

    // the charged total is every live reference; a reference whose turn ended and one that is about
    // to be replaced are not charged, a pinned or in-flight one is still charged
    std::vector<decision_session_ref> refs = {
        ref(100, 10, /* pinned = */ false, /* leased = */ false, /* removed = */ false, /* replacing = */ false),
        ref(200, 20, /* pinned = */ true,  /* leased = */ false, /* removed = */ false, /* replacing = */ false),
        ref(300, 5,  /* pinned = */ false, /* leased = */ true,  /* removed = */ false, /* replacing = */ false),
        ref(400, 1,  /* pinned = */ false, /* leased = */ false, /* removed = */ true,  /* replacing = */ false),
        ref(500, 2,  /* pinned = */ false, /* leased = */ false, /* removed = */ false, /* replacing = */ true),
    };
    assert(decision_session_total_bytes(refs) == 600); // 100 + 200 + 300

    // least recently used, and never a pinned, leased, removed or replaced reference
    assert(pick_decision_session_victim(refs) == 0);

    // with two evictable references the least recently used one wins, not the first in store order
    refs[2].leased = false;
    assert(pick_decision_session_victim(refs) == 2);
    refs[2].leased = true;

    // every remaining reference is held, so there is no victim and the admission must be refused
    refs[0].pinned = true;
    assert(pick_decision_session_victim(refs) == std::nullopt);

    // a held reference is skipped, not deferred: the next evictable one is chosen instead
    refs.push_back(ref(600, 99, /* pinned = */ false, /* leased = */ false, /* removed = */ false, /* replacing = */ false));
    assert(pick_decision_session_victim(refs) == 5);

    // the budget is mebibytes of owned token bytes, so the smallest expressible budget is 1 MiB and
    // the whole tier needs that many tokens to be reachable at all
    assert(1024u * 1024u / sizeof(llama_token) == 262144);

    // an empty store has no victim, so an over-budget admission on an empty store is a refusal
    assert(pick_decision_session_victim({}) == std::nullopt);
    assert(decision_session_total_bytes({}) == 0);
}

// Whether a stored reference may be retired right now, without a server, a model or a lease. This
// is the rule every path that drops a reference shares, so it is expressed here as a pure function
// and the pool only performs what it returns. It is task validity - whether the store may be
// mutated - and reads no answer.
static void test_decision_session_retirement_policy() {
    // nothing holds an ordinary reference, so it retires now and its adapter refs are released
    assert(pick_decision_session_retirement(/* present = */ true, /* leased = */ false) ==
           decision_session_retirement::now);

    // an in-flight decision is never a victim: the reference is marked removed and finalized when
    // the last lease drops, so the reader finishes before its tokens and refs are freed
    assert(pick_decision_session_retirement(/* present = */ true, /* leased = */ true) ==
           decision_session_retirement::deferred);

    // a key the store does not hold is a no-op, whatever the caller's view of the lease says
    assert(pick_decision_session_retirement(/* present = */ false, /* leased = */ false) ==
           decision_session_retirement::absent);
    assert(pick_decision_session_retirement(/* present = */ false, /* leased = */ true) ==
           decision_session_retirement::absent);

    // pinning and expiry decide candidacy, retirement decides death. The two views share one type,
    // so table them together: a pinned unleased reference is never a victim but is retirable once
    // something else picked it, and a leased one is skipped by both.
    decision_session_ref pinned;
    pinned.pinned = true;
    decision_session_ref leased;
    leased.leased = true;

    std::vector<decision_session_ref> refs = { pinned, leased };
    assert(pick_decision_session_victim(refs) == std::nullopt);
    assert(pick_decision_session_retirement(true, pinned.leased) == decision_session_retirement::now);
    assert(pick_decision_session_retirement(true, leased.leased) == decision_session_retirement::deferred);

    // unpinning makes it a victim; the lease still outlives the eviction
    pinned.pinned = false;
    refs[0]       = pinned;
    assert(pick_decision_session_victim(refs) == 0);
    assert(pick_decision_session_retirement(true, refs[0].leased) == decision_session_retirement::now);
}

// The byte budget charges a view of the store and GET /v1/session reports a view of the same store,
// so the two must agree on what a reference costs and on whether the store still holds it. They read
// the same projection and the same liveness rule, so this table pins that agreement: every reference
// the reported count skips is also one the budget does not charge, and no other.
static void test_decision_session_projection_agreement() {
    const auto ref = [](size_t bytes, bool pinned = false, bool leased = false, bool removed = false,
                        bool replacing = false) {
        decision_session_ref r;
        r.bytes     = bytes;
        r.pinned    = pinned;
        r.leased    = leased;
        r.removed   = removed;
        r.replacing = replacing;
        return r;
    };

    // every combination of the four facts that decide liveness and evictability, one reference each
    for (int bits = 0; bits < 16; ++bits) {
        const bool   pinned    = bits & 1;
        const bool   leased    = bits & 2;
        const bool   removed   = bits & 4;
        const bool   replacing = bits & 8;
        const size_t bytes     = 100 * (size_t) (bits + 1);

        const std::vector<decision_session_ref> one = {
            ref(bytes, pinned, leased, removed, replacing),
        };

        // the count the status reports and the bytes the budget charges come from one predicate
        assert(decision_session_live_count(one) == (decision_session_live(one[0]) ? 1 : 0));
        assert(decision_session_total_bytes(one) == (decision_session_live(one[0]) ? bytes : 0));

        // a held reference is charged and counted but is never a victim, so the two questions the
        // budget asks - what does this cost, may this go - stay independent
        assert(decision_session_live(one[0]) == !(removed || replacing));
        if (decision_session_live(one[0]) && !pinned && !leased) {
            assert(pick_decision_session_victim(one) == 0);
        } else if (!decision_session_live(one[0]) || pinned || leased) {
            assert(pick_decision_session_victim(one) == std::nullopt);
        }
    }

    // a store holding both kinds reports the live one and charges only for it
    const std::vector<decision_session_ref> mixed = {
        ref(100),                                                                      // live, evictable
        ref(200, /* pinned = */ true),                                                 // live, held
        ref(300, /* pinned = */ false, /* leased = */ false, /* removed = */ true),    // gone, awaiting its lease
        ref(400, /* pinned = */ false, /* leased = */ false, /* replacing = */ true),  // about to be replaced
    };
    assert(decision_session_live_count(mixed) == 2);
    assert(decision_session_total_bytes(mixed) == 300);
    assert(pick_decision_session_victim(mixed) == 0);

    // an empty store reports and charges nothing, which is what an untouched counter read must show
    assert(decision_session_live_count({}) == 0);
    assert(decision_session_total_bytes({}) == 0);
}

int main(int argc, char ** argv) {
    test_instances_parse_round_trip();
    test_instances_lora_multi_scale();
    test_instances_lora_separators();
    test_instances_lora_compat();
    test_instances_lora_round_trip();
    test_instances_lora_validate();
    test_instances_parse_errors();
    test_instance_numerics_strict();
    test_instances_validate_all_cross_spec();
    test_instances_collision_message_names_pair();
    test_instance_lora_scale_round_trip();
    test_instances_parse_valid_names();
    test_instance_params();
    test_instances_lora_grammar();
    test_instance_params_lora();
    test_lora_fingerprint();
    test_adapter_buf_size_null();
    test_resolve_honors_explicit_instance();
    test_fanout_calibration();
    test_start_loops_starts_io_worker();
    test_decision_session_budget_policy();
    test_decision_session_retirement_policy();
    test_decision_session_projection_agreement();

    common_params params;
    std::string   adapter_path;
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == "-m") {
            params.model.path = argv[i + 1];
        }
        if (std::string(argv[i]) == "--lora") {
            adapter_path = argv[i + 1];
        }
    }
    // hermetic thread count: the -1 default resolves through the host backend
    // registry, which segfaults in some container toolchains
    params.cpuparams.n_threads       = 4;
    params.cpuparams_batch.n_threads = 4;

    // These tests exercise the pool over chat instances only and never enable the decision
    // endpoint, so they state the sidecar off the way arg parsing does: decision_sidecar is
    // derived from --decision-seqs, and a server without it runs no sidecar. Hand-built params
    // that never went through arg parsing keep the struct default (true), which pairs a sidecar
    // with zero decision sequences and is exactly the combination server_instances::load
    // refuses - so it has to be stated here rather than inherited.
    params.decision_sidecar = false;

    if (params.model.path.empty()) {
        fprintf(stderr, "WARNING: no model file provided. Set LLAMACPP_TEST_MODELFILE=<gguf_model_path> to run the borrowed-model test.\n");
        return 0;
    }

    // params are built by hand here (no CLI parse), so resolve the thread
    // counts the same way parsing would; leaving -1 crashes context init
    if (params.cpuparams.n_threads < 0) {
        params.cpuparams.n_threads = common_cpu_get_num_math();
    }
    if (params.cpuparams_batch.n_threads < 0) {
        params.cpuparams_batch.n_threads = common_cpu_get_num_math();
    }

    ggml_backend_load_all();
    test_queue_stop_cancels_pending();
    test_get_lora_adapters_round_trip(params);
    test_pool_adapter_shared_ptr(params);
    test_pool_adapter_fingerprint(params);
    test_pool_adapter_detach_reattach(params);
    test_pool_adapter_scale_update_no_bump(params);
    test_pool_adapter_concurrent_detach_snapshot(params);
    test_pool_adapter_resize_borrowed(params);
    test_adapter_bytes_identities(params);
    test_adapter_bytes_unbuilt_zero(params);
    test_adapter_bytes_pool_shared(params);
    test_oom_text_create_resize(params);
    test_snapshot_records_fp(params);
    test_snapshot_mismatch_400(params);
    test_snapshot_prev2_allows(params);
    test_snapshot_drift_400(params);
    test_snapshot_v2_empty_on_empty_allows(params);
    test_snapshot_v2_empty_on_nonempty_400s(params);
    test_instances_lora_normalize(params);
    test_live_merge_adapter_parity(params);
    test_models_skip_and_continue(params);
    test_instances_listing_off_management_lock(params);
    test_borrowed_model(params);
    test_scheduler_timeout_cancels_pending(params);
    test_destroy_bounded_on_wedged_scheduler(params);
    test_management_healthy_control_fast(params);
    test_instances_envelope_built_unbuilt(params);
    test_demand_build_starts_one_loop(params);
    test_start_loops_skips_unbuilt(params);
    test_resize_teardown_rebuild(params);
    test_display_cache_transitions(params);
    test_built_publish_stress(params);
    test_cold_reload_registry_mismatch(params);
    if (!adapter_path.empty()) {
        test_borrowed_model_adapter_skip(params, adapter_path);
    } else {
        fprintf(stderr, "WARNING: no adapter file provided. Pass --lora <adapter_gguf> to run the adapter skip test.\n");
    }

    fprintf(stdout, "%s: all tests passed\n", __func__);
    return 0;
}
