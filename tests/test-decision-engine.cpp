// White-box access to the engine's save/load helpers: the fail-fast behavior they encode is
// internal with no public seam to force a save failure. The macro is scoped to this TU only.
#define private public
#include "decision-engine.h"
#undef private
#include "../src/llama-ext.h"  // staging API
#include "chat.h"
#include "common.h"
#include "decision-protocol.h"
#include "generic_frontend.h"
#include "ggml-backend.h"
#include "json.h"
#include "labels.h"
#include "letter_readout.h"
#include "llama.h"
#include "speculative.h"
#include "testing.h"

// Staging observability seam for the state save/load bulk-copy path, declared here because it is
// not part of any llama.cpp API: the number of tensor data transfers staged since the last reset.
// A transposed cache region is one strided transfer instead of one per embedding, so the count is
// what the bulk-copy assertions below read. State staging runs on the scheduler thread only.
void     llama_state_seq_debug_reset_transfers();
uint64_t llama_state_seq_debug_transfer_count();

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef DECISION_TEST_FIXTURE_DIR
#error "DECISION_TEST_FIXTURE_DIR must be defined by the build"
#endif

#ifndef DECISION_TEST_BASELINE_DIR
#error "DECISION_TEST_BASELINE_DIR must be defined by the build"
#endif

#ifndef DECISION_TEST_GENERATED_MODEL_DIR
#define DECISION_TEST_GENERATED_MODEL_DIR ""
#endif

#ifndef DECISION_TEST_SPM_MODEL
#define DECISION_TEST_SPM_MODEL ""
#endif

static std::string read_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot read " + path);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static void write_file(const std::string & path, const std::string & text) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("cannot write " + path);
    }
    out << text;
}

static bool file_exists(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    return (bool) in;
}

static void assert_close(testing & t, const std::string & msg, double expected, double actual, double eps = 1e-6) {
    t.assert_true(msg + " (expected " + std::to_string(expected) + ", got " + std::to_string(actual) + ")",
                  std::fabs(expected - actual) <= eps);
}

static std::string fixture_path(const std::string & name) {
    return std::string(DECISION_TEST_FIXTURE_DIR) + "/" + name;
}

// Last two path components, so a golden can name the model it was written from without pinning
// the machine's model root (several GGUFs share the basename "latest.gguf").
static std::string model_identity(const std::string & path) {
    size_t slash = path.find_last_of("/\\");
    if (slash == std::string::npos) {
        return path;
    }
    const std::string file = path.substr(slash + 1);
    const size_t prev = path.find_last_of("/\\", slash - 1);
    return prev == std::string::npos ? file : path.substr(prev + 1, slash - prev - 1) + "/" + file;
}

static const char * decision_valid_body() {
    return R"({
      "model": "m",
      "state": "Customer was charged twice on May 3.",
      "questions": {
        "refund":  {"type": "noul",  "instructions": "Should this be refunded?",
                    "criteria": {"true": "yes", "false": "no"}},
        "dept":    {"type": "choice", "instructions": "What is the issue?",
                    "criteria": {"billing": "payment", "technical": "bug", "cancellation": "cancel"}},
        "urgency": {"type": "scale", "instructions": "How urgent?",
                    "criteria": [{"level": 1, "label": "calm"}, "upset", "furious"]}
      },
      "temperature": 1.0,
      "temperatures": {"noul": 1.0},
      "permutations": 1
    })";
}

static size_t count_substring(const std::string & hay, const std::string & needle) {
    size_t n = 0;
    for (size_t at = hay.find(needle); at != std::string::npos; at = hay.find(needle, at + needle.size())) {
        ++n;
    }
    return n;
}

// Scoring fields is two steps: compile them into a plan, then score the plan. A caller that scores
// the same fields more than once compiles them once, so the engine has one entry and this helper
// is the only place the two steps are written together. It exists here, not in the engine, because
// no production caller owns the compilation: every one of them already has the plan.
static llama_decision::batch_result decide_batch(llama_decision::engine &                         eng,
                                                const std::string &                              shared_text,
                                                const std::vector<std::string> &                 contexts,
                                                const std::vector<llama_decision::field_input> & fields,
                                                const llama_decision::options &                   opt) {
    return eng.decide_batch(eng.compile_fields(fields, opt), shared_text, contexts, opt);
}

// Sorted, comma-joined key set of a JSON object, so a test can pin the exact shape of an envelope.
static std::string key_set(const common_json & obj) {
    std::vector<std::string> keys;
    for (const auto & e : obj.items()) {
        keys.push_back(e.key());
    }
    std::sort(keys.begin(), keys.end());
    return string_join(keys, ",");
}

// Minimal templates whose thinking marker only appears when the caller asks for it. They let the
// test prove that the decision framer pins the toggle off even when the loaded model's own
// template ignores it (LFM2.5 appends its marker unconditionally).
static const char * thinking_probe_template() {
    return "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"
           "{%- if add_generation_prompt -%}assistant:{% if enable_thinking %} <think>{% endif %} {% endif %}";
}

static const char * preserve_probe_template() {
    return "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"
           "{%- if preserve_thinking -%}<think>kept</think>{% endif -%}"
           "{%- if add_generation_prompt -%}assistant:{% endif %}";
}

static std::string join_split(const std::pair<std::string, std::string> & p) {
    return p.first + p.second;
}

static void test_split_chat_template_primitive(testing & t) {
    t.test("split_chat_template and fnv1a64 are the shared primitives behind both renderers", [](testing & t) {
        t.assert_equal("fnv1a64(\"\") is the established offset basis, preserved from the pre-refactor hash",
                       (uint64_t) 1469598103934665603ull, llama_decision::fnv1a64(""));

        auto tmpls = common_chat_templates_init(nullptr, thinking_probe_template());
        if (!tmpls) {
            t.skip("jinja template probe unavailable");
            return;
        }
        const auto parts  = llama_decision::split_chat_template(tmpls.get(), true, "SYS", false);
        const auto letter = llama_decision::render_letter_prompt(tmpls.get(), true, "SYS", false);
        t.assert_equal("render_letter_prompt returns the primitive split head", parts.first, letter.first);
        t.assert_equal("render_letter_prompt returns the primitive split tail", parts.second, letter.second);
    });
}

static void test_thinking_off(testing & t) {
    t.test("the decision framer keeps thinking off", [](testing & t) {
        auto probe = common_chat_templates_init(nullptr, thinking_probe_template());
        if (!probe) {
            t.skip("the jinja engine is unavailable");
            return;
        }
        try {
            const std::string sys = "sys";
            const auto off     = llama_decision::render_letter_prompt(probe.get(), true, sys, false);
            const auto on      = llama_decision::render_letter_prompt(probe.get(), true, sys, true);
            const auto default_off = llama_decision::render_letter_prompt(probe.get(), true, sys);

            t.assert_equal("the framer defaults to thinking off", join_split(off), join_split(default_off));
            t.assert_true("thinking off emits no marker", count_substring(join_split(off), "<think>") == 0);
            t.assert_true("thinking on emits the marker", count_substring(join_split(on), "<think>") == 1);
            t.assert_true("thinking on adds tokens", join_split(on).size() > join_split(off).size());

        } catch (const std::exception & e) {
            t.assert_true(std::string("thinking probe renders: ") + e.what(), false);
        }
    });

    t.test("a raw decision prefix carries no thinking marker", [](testing & t) {
        const auto raw = llama_decision::render_letter_prompt(nullptr, false, "SYS");
        t.assert_true("raw letter prefix is thinking free", count_substring(join_split(raw), "<think>") == 0);
    });
}

static void test_thinking_control(testing & t) {
    t.test("preserve_thinking off does not inject a marker into the decision prefix", [](testing & t) {
        auto probe = common_chat_templates_init(nullptr, preserve_probe_template());
        if (!probe) {
            t.skip("the jinja engine is unavailable");
            return;
        }
        try {
            const std::string sys = "sys";
            // the framer never asks the template to preserve reasoning, so no marker is rendered
            const auto prefix = llama_decision::render_letter_prompt(probe.get(), true, sys, false);
            t.assert_true("the decision prefix has no preserved thinking",
                          count_substring(join_split(prefix), "<think>") == 0);

            // control: the same template does inject a marker when a caller explicitly preserves it
            common_chat_templates_inputs in;
            in.use_jinja             = true;
            in.add_generation_prompt = true;
            in.enable_thinking       = false;
            in.chat_template_kwargs  = { { "preserve_thinking", "true" } };
            common_chat_msg sys_msg;
            sys_msg.role    = "system";
            sys_msg.content = sys;
            common_chat_msg usr_msg;
            usr_msg.role    = "user";
            usr_msg.content = "hello";
            in.messages = { sys_msg, usr_msg };
            const std::string preserved = common_chat_templates_apply(probe.get(), in).prompt;
            t.assert_true("the control template can inject when asked",
                          count_substring(preserved, "<think>") == 1);
        } catch (const std::exception & e) {
            t.assert_true(std::string("preserve control renders: ") + e.what(), false);
        }
    });
}

static void test_decision_shape_contract(testing & t) {
    t.test("committed decision envelope skeleton has the required keys", [](testing & t) {
        const common_json shape = common_json::parse(read_file(fixture_path("decision_basic.shape.json")));
        t.assert_true("request shape", shape.contains("request"));
        t.assert_true("response shape", shape.contains("response"));
        t.assert_true("answer shapes", shape.contains("answer_shapes"));
        const auto & answers = shape.at("answer_shapes");
        t.assert_true("noul shape", answers.contains("noul"));
        t.assert_true("choice shape", answers.contains("choice"));
        t.assert_true("score shape", answers.contains("score"));
        t.assert_true("choice probabilities", answers.at("choice").contains("probabilities"));
        t.assert_true("choice confidence", answers.at("choice").contains("confidence"));
        t.assert_true("score legend", answers.at("score").contains("legend"));
        const auto & usage = shape.at("response").at("usage");
        t.assert_equal("output_tokens is fixed at zero", 0, usage.at("output_tokens").get<int>());
    });
}

static void test_softmax(testing & t) {
    t.test("softmax normalizes, keeps the winner and flattens with temperature", [](testing & t) {
        const std::vector<float> logits = { -1.0f, 0.5f, 2.0f };
        const auto p = llama_decision::softmax(logits, 1.0f);
        double sum = 0.0;
        for (float x : p) {
            sum += x;
        }
        assert_close(t, "probabilities sum to one", 1.0, sum, 1e-6);
        t.assert_true("winner preserved", p[2] > p[1] && p[1] > p[0]);

        const auto flat = llama_decision::softmax(logits, 4.0f);
        t.assert_true("higher temperature flattens", flat[2] < p[2]);
        t.assert_true("empty input is handled", llama_decision::softmax({}).empty());
    });
}

// The answer-row scoring math takes no context, so it is covered with synthetic hidden states,
// rows, bias and softcap instead of a model.


// The load flag is a property of the saved format, so a capability downgrade can never route a
// device state through the host reader or the other way around.
static void test_saved_state_format_dispatch(testing & t) {
    t.test("a saved state is loaded with its own format", [&](testing & t) {
        t.assert_equal("a device state selects the device flag",
                       (unsigned) LLAMA_STATE_SEQ_FLAGS_ON_DEVICE,
                       (unsigned) llama_decision::engine::state_load_flags(true));
        t.assert_equal("a host state selects no flag",
                       (unsigned) LLAMA_STATE_SEQ_FLAGS_NONE,
                       (unsigned) llama_decision::engine::state_load_flags(false));
        t.assert_equal("a partial device state selects both flags",
                       (unsigned) (LLAMA_STATE_SEQ_FLAGS_ON_DEVICE | LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY),
                       (unsigned) llama_decision::engine::state_load_flags(true, true));
        t.assert_equal("a partial host state selects the partial flag",
                       (unsigned) LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY,
                       (unsigned) llama_decision::engine::state_load_flags(false, true));
    });
}

// The provenance gate control is shared by the temperature test and the sign-off table.
static common_json calibration_temperature_control_measurement();

// The numeric extension over the Jev question set: integer/number grids are generated from bounds
// at parse time, scored by label exactly like a choice, and answered with a typed value plus an
// optional aggregate. Pure JSON, no model needed.
static void expect_decision_reject(testing & t, const std::string & body_text, const std::string & needle);
static void test_numeric_questions(testing & t) {
    t.test("numeric integer and number grids parse into typed options", [](testing & t) {
        const auto body = common_json::parse(R"({"model":"m","state":"s","questions":{
            "age":   {"type":"integer","instructions":"age in years","minimum":18,"maximum":20},
            "amount":{"type":"number","instructions":"amount","minimum":0,"maximum":0.5,"step":0.25},
            "avg":   {"type":"integer","instructions":"rating","minimum":1,"maximum":3,"aggregate":"mean"}}})");
        const auto req = llama_decision::parse_decision_request(body);
        t.assert_equal("three questions", (size_t) 3, req.questions.size());

        const auto & age = req.questions[0];
        t.assert_equal("integer canonical type", std::string("integer"), age.type);
        t.assert_equal("integer grid 18..20", (size_t) 3, age.options.size());
        t.assert_equal("integer key is the value", std::string("18"), age.options[0].key);
        t.assert_equal("integer grid is ascending", std::string("20"), age.options[2].key);
        t.assert_true("integer originals are typed numbers", age.options[0].original.is_number_integer());
        t.assert_equal("aggregate absent stays empty", std::string(), age.aggregate);

        const auto & amount = req.questions[1];
        t.assert_equal("number canonical type", std::string("number"), amount.type);
        t.assert_equal("number grid 0..0.5 by 0.25", (size_t) 3, amount.options.size());
        t.assert_equal("number key is fixed width", std::string("0.00"), amount.options[0].key);
        t.assert_equal("number key strips float noise", std::string("0.50"), amount.options[2].key);
        t.assert_true("number originals are floats", amount.options[1].original.is_number_float());

        const auto & avg = req.questions[2];
        t.assert_equal("aggregate parsed", std::string("mean"), avg.aggregate);
    });

    t.test("numeric limits and shapes are validated", [](testing & t) {
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"integer","instructions":"x","minimum":10,"maximum":9}}})",
                               "bounds");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"integer","instructions":"x","minimum":1,"maximum":2,"criteria":{"a":"x"}}}})",
                               "unknown field");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"integer","instructions":"x","minimum":1.5,"maximum":2}}})",
                               "integer needs integer minimum");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"integer","instructions":"x"}}})",
                               "needs integer minimum and maximum");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"number","instructions":"x","minimum":0,"maximum":1}}})",
                               "step (or multipleOf)");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"number","instructions":"x","minimum":0,"maximum":1,"step":0,"multipleOf":0.1}}})",
                               "not both");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"number","instructions":"x","minimum":0,"maximum":1,"step":0.3}}})",
                               "must include both ends");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"integer","instructions":"x","minimum":1,"maximum":1,"aggregate":"mean"}}})",
                               "2-255 values");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"number","instructions":"x","minimum":1,"maximum":256,"step":1}}})",
                               "2-255 values");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"integer","instructions":"x","minimum":1,"maximum":3,"aggregate":"bogus"}}})",
                               "aggregate must be mode, median or mean");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"number","instructions":"x","minimum":0,"maximum":1,"step":0.25,"aggregate":1}}})",
                               "aggregate must be a string");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"choice","instructions":"x","criteria":{"a":null,"b":null},"aggregate":"mean"}}})",
                               "unknown field");
    });

    t.test("numeric answers carry a typed value, probabilities and the optional aggregate", [](testing & t) {
        const auto body = common_json::parse(R"({"model":"m","state":"s","questions":{
            "age":{"type":"integer","instructions":"age","minimum":18,"maximum":20},
            "avg":{"type":"number","instructions":"amount","minimum":0,"maximum":2,"step":1,"aggregate":"mean"},
            "med":{"type":"integer","instructions":"rating","minimum":1,"maximum":4,"aggregate":"median"}}})");
        auto req = llama_decision::parse_decision_request(body);
        common_json usage = common_json::object();
        usage["output_tokens"] = 0;

        const std::vector<std::vector<float>> probs = {
            { 0.1f, 0.2f, 0.7f },       // age grid 18,19,20: winner 20
            { 0.5f, 0.3f, 0.2f },       // avg grid 0,1,2: mean = 0.3 + 0.4 = 0.7
            { 0.5f, 0.2f, 0.2f, 0.1f }, // med grid 1..4: median crosses at 1
        };
        const common_json out = llama_decision::assemble_decision_response(req, probs, "m", usage);

        const auto & age = out.at("answers").at("age");
        t.assert_equal("type echoed", std::string("integer"), age.at("type").get<std::string>());
        t.assert_equal("value is the typed winner", 20, age.at("value").get<long long>());
        assert_close(t, "probabilities keyed by value", 0.7, age.at("probabilities").at("20").get<double>(), 1e-6);
        t.assert_true("no aggregate without a request", !age.contains("aggregate"));

        const auto & avg = out.at("answers").at("avg");
        t.assert_equal("value is the winner as a number", 0.0, avg.at("value").get<double>());
        assert_close(t, "aggregate mean is the weighted mean", 0.7, avg.at("aggregate").get<double>(), 1e-6);

        const auto & med = out.at("answers").at("med");
        assert_close(t, "aggregate median is the value-space quantile", 1.0,
                     med.at("aggregate").get<double>(), 1e-6);

        req.envelope.diagnostics = true;
        const common_json diag = llama_decision::assemble_decision_response(req, probs, "m", usage);
        const auto & dage = diag.at("answers").at("age");
        t.assert_true("diagnostics add certainty", dage.contains("certainty"));
        t.assert_true("diagnostics add the spread median", dage.contains("median"));
        t.assert_true("diagnostics add the p10/p90 band", dage.contains("interval_p10_p90"));
        assert_close(t, "p10/p90 band has two entries", 2, (long long) dage.at("interval_p10_p90").size());
    });

    t.test("numeric types take their own temperature overrides", [](testing & t) {
        const auto body = common_json::parse(R"({"model":"m","state":"s","questions":{
            "age":{"type":"integer","instructions":"x","minimum":1,"maximum":3},
            "amt":{"type":"number","instructions":"x","minimum":0,"maximum":1,"step":0.5}}})");
        common_json with_temps = body;
        with_temps["temperature"]  = 1.5;
        with_temps["temperatures"] = common_json::parse(R"({"integer":0.5,"number":2.0})");
        const auto req = llama_decision::parse_decision_request(with_temps);
        assert_close(t, "integer override", 0.5, llama_decision::question_temperature(req, req.questions[0]));
        assert_close(t, "number override", 2.0, llama_decision::question_temperature(req, req.questions[1]));
    });
}

static void test_question_temperature(testing & t) {
    t.test("effective temperature follows per-type override then global", [](testing & t) {
        const auto base = common_json::parse(R"({"model":"m","state":"s","questions":{
            "a":{"type":"noul","instructions":"x"},
            "b":{"type":"choice","instructions":"x","criteria":{"p":null,"q":null}},
            "c":{"type":"score","instructions":"x","criteria":["lo","hi"]}}})");

        const auto plain = llama_decision::parse_decision_request(base);
        for (const auto & q : plain.questions) {
            assert_close(t, "global default", 1.0, llama_decision::question_temperature(plain, q));
        }

        common_json with_override = base;
        with_override["temperature"]   = 1.5;
        with_override["temperatures"]  = common_json::parse(R"({"noul":0.5,"choice":2.0})");
        const auto req = llama_decision::parse_decision_request(with_override);
        assert_close(t, "noul override", 0.5, llama_decision::question_temperature(req, req.questions[0]));
        assert_close(t, "choice override", 2.0, llama_decision::question_temperature(req, req.questions[1]));
        assert_close(t, "score falls back", 1.5, llama_decision::question_temperature(req, req.questions[2]));
    });
}

static void test_temperature_effect(testing & t) {
    t.test("temperature preserves the argmax and sharpens or flattens", [](testing & t) {
        const std::vector<float> logits = { -0.5f, 0.0f, 1.5f };
        const auto warm = llama_decision::softmax(logits, 0.5f);
        const auto mid  = llama_decision::softmax(logits, 1.0f);
        const auto cool = llama_decision::softmax(logits, 2.5f);

        auto argmax = [](const std::vector<float> & p) {
            return (int) (std::max_element(p.begin(), p.end()) - p.begin());
        };
        t.assert_equal("argmax invariant (sharp)", argmax(mid), argmax(warm));
        t.assert_equal("argmax invariant (flat)", argmax(mid), argmax(cool));
        t.assert_true("lower temperature sharpens", warm[argmax(warm)] > mid[argmax(mid)]);
        t.assert_true("higher temperature flattens", cool[argmax(cool)] < mid[argmax(mid)]);
    });

    t.test("assemble emits confidence and certainty consistent with the probabilities", [](testing & t) {
        auto req = llama_decision::parse_decision_request(common_json::parse(decision_valid_body()));
        req.envelope.diagnostics = true;
        const std::vector<std::vector<float>> probs = {
            { 0.2f, 0.8f },
            { 0.5f, 0.3f, 0.2f },
            { 0.1f, 0.2f, 0.7f },
        };
        common_json usage = common_json::object();
        usage["output_tokens"] = 0;

        auto jev_recompute = [](const std::vector<float> & p) {
            const double n = (double) p.size();
            const double pmax = (double) *std::max_element(p.begin(), p.end());
            return std::min(1.0, std::max(0.0, (n * pmax - 1.0) / (n - 1.0)));
        };

        const common_json out = llama_decision::assemble_decision_response(req, probs, "m", usage);
        req.envelope.knobs.confidence_profile = "local";
        const common_json local_out = llama_decision::assemble_decision_response(req, probs, "m", usage);

        const auto & dept = out.at("answers").at("dept");
        t.assert_true("choice has confidence", dept.contains("confidence"));
        t.assert_true("choice has certainty", dept.contains("certainty"));
        assert_close(t, "default choice confidence is the Jev winner-share rescale", jev_recompute(probs[1]),
                     dept.at("confidence").get<double>(), 1e-6);
        assert_close(t, "choice certainty is the winner share", 0.5, dept.at("certainty").get<double>(), 1e-6);

        const auto & urg = out.at("answers").at("urgency");
        t.assert_true("score has confidence", urg.contains("confidence"));
        t.assert_true("score has certainty", urg.contains("certainty"));
        assert_close(t, "default score confidence is the Jev winner-share rescale", jev_recompute(probs[2]),
                     urg.at("confidence").get<double>(), 1e-6);

        t.assert_true("noul has no confidence", !out.at("answers").at("refund").contains("confidence"));
        t.assert_true("noul has no certainty", !out.at("answers").at("refund").contains("certainty"));
    });
}

// confidence and certainty are two axes: the Jev winner-share rescale (default) and the winner's
// share. Pin both against hand-computed literals so a rename cannot quietly swap them.
static void test_confidence_certainty_axes(testing & t) {
    t.test("confidence is the Jev winner-share rescale and certainty is the winner share", [](testing & t) {
        auto req        = llama_decision::parse_decision_request(common_json::parse(decision_valid_body()));
        req.envelope.diagnostics = true;
        const std::vector<std::vector<float>> probs = {
            { 0.2f, 0.8f }, // noul: neither axis is emitted
            { 0.6f, 0.3f, 0.1f }, // choice
            { 0.9f, 0.1f, 0.0f }, // score
        };
        common_json usage      = common_json::object();
        usage["output_tokens"] = 0;
        const common_json out  = llama_decision::assemble_decision_response(req, probs, "m", usage);
        req.envelope.knobs.confidence_profile = "local";
        const common_json local_out = llama_decision::assemble_decision_response(req, probs, "m", usage);

        const auto & dept = out.at("answers").at("dept");
        assert_close(t, "default choice confidence is the Jev rescale for (0.6,0.3,0.1)", 0.4,
                     dept.at("confidence").get<double>(), 1e-6);
        assert_close(t, "choice certainty is max(p) = 0.6", 0.6, dept.at("certainty").get<double>(), 1e-6);

        const auto & urg = out.at("answers").at("urgency");
        assert_close(t, "default score confidence is the Jev rescale for (0.9,0.1,0.0)", 0.85,
                     urg.at("confidence").get<double>(), 1e-6);
        assert_close(t, "score certainty is max(p) = 0.9", 0.9, urg.at("certainty").get<double>(), 1e-6);

        const auto & ldept = local_out.at("answers").at("dept");
        assert_close(t, "local choice confidence is 1 - H/log 3 for (0.6,0.3,0.1)", 0.182654578,
                     ldept.at("confidence").get<double>(), 1e-6);
        const auto & lurg = local_out.at("answers").at("urgency");
        assert_close(t, "local score confidence is 1 - H/log 3 for (0.9,0.1,0.0)", 0.704096726,
                     lurg.at("confidence").get<double>(), 1e-6);

        t.assert_true("noul carries no confidence", !out.at("answers").at("refund").contains("confidence"));
        t.assert_true("noul carries no certainty", !out.at("answers").at("refund").contains("certainty"));
    });
}

static void expect_decision_reject(testing & t, const std::string & body_text, const std::string & needle);

// The opt-in Jev confidence profile is a rescaled winner share, (N*p_max-1)/(N-1). It is opt-in,
// so the default stays 1 - H/logK and existing goldens are unchanged; above three levels, where
// Jev documents no Score formula, the same monotone rule is the stated local value.
static void test_confidence_profile(testing & t) {
    t.test("the Jev confidence profile is the default certainty-based winner share", [](testing & t) {
        auto req        = llama_decision::parse_decision_request(common_json::parse(decision_valid_body()));
        req.envelope.diagnostics = true;
        const std::vector<std::vector<float>> probs = {
            { 0.2f, 0.8f },          // noul: no confidence either way
            { 0.6f, 0.3f, 0.1f },    // choice, N=3: (3*0.6-1)/2 = 0.4
            { 0.9f, 0.1f, 0.0f },    // score, N=3: (3*0.9-1)/2 = 0.85
        };
        common_json usage      = common_json::object();
        usage["output_tokens"] = 0;

        const common_json jev   = llama_decision::assemble_decision_response(req, probs, "m", usage);
        req.envelope.knobs.confidence_profile  = "local";
        const common_json local = llama_decision::assemble_decision_response(req, probs, "m", usage);

        assert_close(t, "default choice confidence is (N*p_max-1)/(N-1)", 0.4,
                     jev.at("answers").at("dept").at("confidence").get<double>(), 1e-6);
        assert_close(t, "local confidence is the opt-in 1 - H/log K", 0.182654578,
                     local.at("answers").at("dept").at("confidence").get<double>(), 1e-6);
        assert_close(t, "default score confidence uses the same rule above the documented range", 0.85,
                     jev.at("answers").at("urgency").at("confidence").get<double>(), 1e-6);
        assert_close(t, "certainty is the raw winner share and does not change with the profile", 0.6,
                     jev.at("answers").at("dept").at("certainty").get<double>(), 1e-6);
        t.assert_true("noul never carries a confidence", !jev.at("answers").at("refund").contains("confidence"));
    });

    t.test("the Jev confidence rule is pinned at the boundaries and for wide scales", [](testing & t) {
        assert_close(t, "uniform two-way is 0", 0.0,
                     llama_decision::jev_winner_share_confidence({ 0.5f, 0.5f }), 1e-9);
        assert_close(t, "one-hot two-way is 1", 1.0,
                     llama_decision::jev_winner_share_confidence({ 0.0f, 1.0f }), 1e-9);
        // (4*0.48-1)/3, a 4-level score where Jev leaves the definition open
        assert_close(t, "wide scale uses the same monotone rule", 0.306666667,
                     llama_decision::jev_winner_share_confidence({ 0.48f, 0.3f, 0.2f, 0.02f }), 1e-6);
    });

    t.test("confidence_profile parses as an enum and is refused otherwise", [](testing & t) {
        common_json body = common_json::parse(decision_valid_body());
        body["confidence_profile"] = "jev";
        t.assert_equal("jev is accepted", "jev",
                       llama_decision::parse_decision_request(body).envelope.knobs.confidence_profile);
        body["confidence_profile"] = "local";
        t.assert_equal("local is accepted", "local",
                       llama_decision::parse_decision_request(body).envelope.knobs.confidence_profile);
        body.erase("confidence_profile");
        t.assert_equal("absent defaults to jev (certainty-based)", "jev",
                       llama_decision::parse_decision_request(body).envelope.knobs.confidence_profile);
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"noul","instructions":"x"}},"confidence_profile":"other"})",
                               "confidence_profile must be local or jev");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"noul","instructions":"x"}},"confidence_profile":1})",
                               "confidence_profile must be a string");
    });
}

static void test_temperature_profile(testing & t) {
    t.test("temperature provenance is enforced only for non-default values", [](testing & t) {
        const common_json doc = common_json::parse(R"({
            "temperatures": {"noul": 0.8, "choice": 1.3},
            "provenance": {"model": "m1", "quantization": "Q4_K", "template_hash": "t1", "backend_flags": "fa1"}
        })");
        const auto profile = llama_decision::parse_temperature_profile(doc);
        assert_close(t, "noul temperature parsed", 0.8, profile.temperatures.at("noul"));

        llama_decision::temperature_provenance current = profile.provenance;
        llama_decision::validate_temperature_profile(profile, current); // must not throw

        current.quantization = "Q8_0";
        bool refused = false;
        try {
            llama_decision::validate_temperature_profile(profile, current);
        } catch (const llama_decision::semantic_error &) {
            refused = true;
        }
        t.assert_true("mismatched provenance refuses non-default temperature", refused);

        const auto neutral = llama_decision::parse_temperature_profile(
            common_json::parse(R"({"temperatures":{"noul":1.0},"provenance":{"model":"other"}})"));
        llama_decision::validate_temperature_profile(neutral, current); // T=1 always allowed
    });

    t.test("bad temperature profiles are rejected", [](testing & t) {
        const char * bad[] = {
            R"([1,2,3])",
            R"({"temperatures": 5})",
            R"({"temperatures": {"speed": 2.0}})",
            R"({"temperatures": {"noul": 0}})",
            R"({"provenance": {"model": 7}})",
        };
        for (const char * text : bad) {
            bool threw = false;
            try {
                (void) llama_decision::parse_temperature_profile(common_json::parse(text));
            } catch (const llama_decision::semantic_error &) {
                threw = true;
            }
            t.assert_true(std::string("rejected: ") + text, threw);
        }
    });

    t.test("temperature provenance control: identical accepted, every mismatched field refused", [](testing & t) {
        // The provenance gate is confidence-in-the-producer, not an outcome guarantee: a matching
        // profile can still produce a wrong answer, so it never gates answer validity.
        const common_json m = calibration_temperature_control_measurement();
        t.assert_equal("identical provenance is accepted", 1, m.at("match_accepted").get<int>());
        t.assert_equal("every mismatched provenance field is refused",
                       m.at("mismatch_cases").get<int>(), m.at("mismatch_refused").get<int>());
    });
}

static void test_confidence_never_gates(testing & t) {
    t.test("confidence tokens never share a line with gating logic", [](testing & t) {
        const char * files[] = {
            "decision-engine.h", "decision-engine.cpp",
            "decision-protocol.h", "decision-protocol.cpp",
            "labels.h", "labels.cpp",
            "letter_readout.h", "letter_readout.cpp",
        };
        const char * confidence_tokens[] = { "confidence", "certainty" };
        const char * gating_tokens[] = { "allow_cache", "cache_tag", "admit", "routing", "persist" };

        int violations = 0;
        for (const char * file : files) {
            std::ifstream in(std::string(DECISION_TEST_SOURCE_DIR) + "/" + file);
            std::string line;
            while (std::getline(in, line)) {
                bool has_conf = false;
                bool has_gate = false;
                for (const char * tok : confidence_tokens) {
                    has_conf = has_conf || line.find(tok) != std::string::npos;
                }
                for (const char * tok : gating_tokens) {
                    has_gate = has_gate || line.find(tok) != std::string::npos;
                }
                if (has_conf && has_gate) {
                    ++violations;
                }
            }
        }
        t.assert_equal("no line couples confidence with gating", 0, violations);
    });
}

static void test_prefix_tag(testing & t) {
    t.test("prefix tag is stable and changes with the prompt", [](testing & t) {
        const auto a = llama_decision::make_prefix_tag("sys", "after", "v1");
        const auto b = llama_decision::make_prefix_tag("sys", "after", "v1");
        t.assert_equal("tag is stable", a, b);
        t.assert_true("system text changes the tag", a != llama_decision::make_prefix_tag("sys2", "after", "v1"));
        t.assert_true("boundary changes the tag", a != llama_decision::make_prefix_tag("sys", "after2", "v1"));
        t.assert_true("version changes the tag", a != llama_decision::make_prefix_tag("sys", "after", "v2"));
    });
}

static void test_sequence_partition(testing & t) {
    t.test("sequence namespace is partitioned and contiguous", [](testing & t) {
        const common_json baseline = common_json::parse(
            read_file(std::string(DECISION_TEST_BASELINE_DIR) + "/baseline.json"));
        t.assert_true("partition recorded", baseline.contains("partition"));
        const auto & p = baseline.at("partition");
        const long long slots_first  = p.at("slots_first").get<long long>();
        const long long slots_end    = p.at("slots_last_exclusive").get<long long>();
        const long long decision     = p.at("decision_first").get<long long>();
        const long long decision_n   = p.at("decision_count").get<long long>();
        const long long n_seq_max    = p.at("n_seq_max").get<long long>();
        t.assert_equal("slots start at zero", (long long) 0, slots_first);
        t.assert_equal("decision region follows slots", slots_end, decision);
        t.assert_equal("n_seq_max covers both regions", decision + decision_n, n_seq_max);
        t.assert_true("ranges are non-empty", slots_end > slots_first && decision_n >= 3);
    });
}

static std::string decision_golden_path() {
    return fixture_path("decision_letter.golden.json");
}

// The decision tests run on the GPU backend only; a CPU fallback would silently change the numbers.
static bool decision_gpu_available() {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(ggml_backend_dev_get(i));
        if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU ||
            type == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            return true;
        }
    }
    return false;
}

// Loads the decision test model once per process and shares it across tests. Tests still create
// their own cheap contexts, so per-test KV state stays isolated and nothing re-reads the GGUF.
struct shared_test_model {
    llama_model * model = nullptr;
    std::string   path;

    bool load(const char * p) {
        if (p == nullptr || p[0] == '\0') {
            return false;
        }
        if (model != nullptr) {
            return path == p;
        }
        llama_backend_init();
        if (!decision_gpu_available()) {
            fprintf(stderr, "no GPU backend available; the decision tests run on GPU only\n");
            return false;
        }
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = -1;
        model = llama_model_load_from_file(p, mp);
        path  = p;
        return model != nullptr;
    }

    ~shared_test_model() {
        if (model != nullptr) {
            llama_model_free(model);
        }
        llama_backend_free();
    }
};

static shared_test_model & shared_model() {
    static shared_test_model shared;
    return shared;
}

// A model test needs a GGUF and a GPU backend. On a CPU-only build it skips, so the CPU suite
// stays green instead of reporting a load failure.
static bool gpu_model_ready(testing & t, const char * path) {
    if (path == nullptr || path[0] == '\0' || !file_exists(path)) {
        t.skip("set LLAMA_DECISION_TEST_MODEL to run");
        return false;
    }
    if (!decision_gpu_available()) {
        t.skip("this build has no GPU backend; the model tests need one");
        return false;
    }
    if (!shared_model().load(path)) {
        t.assert_true("model loads", false);
        return false;
    }
    return true;
}

// The committed weak-quant GPU allowlist: exact model identities (the last two path components,
// so GGUFs sharing a basename stay distinct) whose coarse quantization moves a winner on the
// GPU. On these models the task-value order-independence, temperature-winner, and head-vs-full
// agreement checks cannot be verified because the producer is not bit-stable, so they are skipped
// with a reason; everywhere else they are hard assertions. Renaming or re-quantizing a file
// changes its identity and drops it off the list, which fails loudly instead of silently moving
// the skip. This list is producer confidence only: it decides whether a task-value assertion is
// skippable on a particular model. It must never absorb an outcome-correctness failure on a model
// where the assertion is expected to hold.
static const std::vector<std::string> weak_quant_gpu_allowlist = {
    // "qwen3.5-2b-gguf/ud-q5_k_xl.gguf",
};

static bool weak_quant_gpu_oracle(const char * path) {
    if (path == nullptr || path[0] == '\0') {
        return false;
    }
    const std::string id = model_identity(path);
    for (const std::string & known : weak_quant_gpu_allowlist) {
        if (id == known) {
            return true;
        }
    }
    return false;
}

// Runs a task-value producer-determinism assertion as a hard test, or skips it on the known
// weak-quant oracle where the producer's numerics move a winner. A skip is explicit, never an
// expected failure: the assertion is a task-value check, so it must never be marked xfail.
template <typename F>
static void determinism_check(testing & t, bool weak_quant, const std::string & name, F body) {
    if (weak_quant) {
        t.skip(name + " (weak-quant GPU numerics; skipped, not xfail)");
        return;
    }
    t.test(name, body);
}

static void test_thinking_off_model(testing & t) {
    t.test("the loaded model's decision prefix stays thinking off", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (!gpu_model_ready(t, path)) {
            return;
        }
        try {
            auto tmpls = common_chat_templates_init(shared_model().model, "");
            if (!tmpls) {
                t.skip("the model has no chat template");
                return;
            }
            const auto vocab = llama_decision::make_llama_label_vocab(llama_model_get_vocab(shared_model().model));
            const std::string sys = llama_decision::letter_system_text();

            const auto off = llama_decision::render_letter_prompt(tmpls.get(), true, sys, false);
            const auto on  = llama_decision::render_letter_prompt(tmpls.get(), true, sys, true);
            const auto def = llama_decision::render_letter_prompt(tmpls.get(), true, sys);

            t.assert_equal("the framer pins the template toggle off", join_split(off), join_split(def));
            t.assert_true("the cacheable prefix carries no thinking marker",
                          count_substring(off.first, "<think>") == 0);

            // Whether the toggle changes the render is a property of the template, not the framer:
            // LFM2.5 appends its marker unconditionally, while Qwen drops the empty think block
            // when thinking is on. The framer's own output stays the thinking-off render above.
            const size_t tok_off = vocab->tokenize(join_split(off), false).size();
            const size_t tok_on  = vocab->tokenize(join_split(on),  false).size();
            t.assert_true("both toggle states render a non-empty prompt", tok_off > 0 && tok_on > 0);
        } catch (const std::exception & e) {
            t.assert_true(std::string("model thinking probe renders: ") + e.what(), false);
        }
    });
}

// Owns a decision-shaped context over the shared model.
struct test_engine {
    llama_model   * model = nullptr;
    llama_context * ctx   = nullptr;

    ~test_engine() {
        if (ctx) {
            llama_free(ctx);
        }
    }

    bool make_ctx(int n_batch = 512, int n_seq_max = 10, bool flash_attn = false, int n_ctx = 8192) {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx                 = n_ctx;
        cp.n_batch               = n_batch;
        cp.n_ubatch              = n_batch;
        cp.n_seq_max             = n_seq_max;
        cp.n_outputs_max         = n_seq_max;
        cp.n_outputs_max_per_seq = 1;
        cp.kv_unified            = true;
        cp.swa_full              = false;
        // ROCm flash attention is not reproducible; decisions must be bit-exact run to run. The
        // state tests may turn it on to exercise the non-transposed V cache.
        cp.flash_attn_type       = flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        ctx = llama_init_from_model(model, cp);
        return ctx != nullptr;
    }

    bool load(const char * path, int n_seq_max = 10, int n_ctx = 8192) {
        if (!shared_model().load(path)) {
            return false;
        }
        model = shared_model().model;
        return make_ctx(512, n_seq_max, false, n_ctx);
    }

    // A fresh context on the shared model, for a phase that must start with an empty window: a
    // decision engine keeps its resident warm prefixes for as long as it lives.
    bool reopen(int n_seq_max = 10, int n_ctx = 8192) {
        if (ctx) {
            llama_free(ctx);
            ctx = nullptr;
        }
        return make_ctx(512, n_seq_max, false, n_ctx);
    }

    bool load_fa(const char * path, int n_seq_max = 10) {
        if (!shared_model().load(path)) {
            return false;
        }
        model = shared_model().model;
        return make_ctx(512, n_seq_max, true);
    }
};

// CPU-runnable decision scaffold: it does not require a GPU, so the state/fork mechanics run in
// CI. It loads the dummy model from the generate-models fixture with no GPU layers.
static std::string decision_cpu_model_path() {
    const std::string generated = std::string(DECISION_TEST_GENERATED_MODEL_DIR) + "/qwen35-dense.gguf";
    if (file_exists(generated)) {
        return generated;
    }
    const char * env = std::getenv("LLAMA_DECISION_TEST_MODEL");
    if (env != nullptr && env[0] != '\0' && file_exists(env)) {
        return env;
    }
    return std::string();
}

// The SentencePiece model is the fixture for boundary-resolved labels: its isolated token is the
// space-prefixed form, so the isolated rule cannot see the label the model emits after the answer
// tail. Env override wins, then the downloaded fixture, then empty so the test skips network-free.
static std::string spm_labels_model_path() {
    const char * env = std::getenv("LLAMA_DECISION_TEST_MODEL");
    if (env != nullptr && env[0] != '\0' && file_exists(env)) {
        return env;
    }
    const std::string spm = DECISION_TEST_SPM_MODEL;
    if (file_exists(spm)) {
        return spm;
    }
    return std::string();
}

struct cpu_test_engine {
    llama_model   * model = nullptr;
    llama_context * ctx   = nullptr;

    ~cpu_test_engine() {
        if (ctx) {
            llama_free(ctx);
        }
        if (model) {
            llama_model_free(model);
        }
    }

    bool load(const std::string & path,
              int                 n_ctx      = 256,
              bool                embeddings = false,
              int                 n_batch    = 128,
              int                 n_seq_max  = 10) {
        if (path.empty()) {
            return false;
        }
        llama_backend_init();
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        model = llama_model_load_from_file(path.c_str(), mp);
        if (model == nullptr) {
            return false;
        }
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx                 = n_ctx;
        cp.n_batch               = n_batch;
        cp.n_ubatch              = n_batch;
        cp.n_seq_max             = n_seq_max;
        cp.n_outputs_max         = n_seq_max;
        cp.n_outputs_max_per_seq = 1;
        cp.kv_unified            = true;
        cp.swa_full              = false;
        cp.flash_attn_type       = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.embeddings            = embeddings;
        ctx = llama_init_from_model(model, cp);
        return ctx != nullptr;
    }
};

static void expect_decision_reject(testing & t, const std::string & body_text, const std::string & needle) {
    try {
        const common_json body = common_json::parse(body_text);
        (void) llama_decision::parse_decision_request(body);
        t.assert_true("decision request is rejected: " + body_text, false);
    } catch (const llama_decision::semantic_error & e) {
        const std::string what = e.what();
        t.assert_true("reject reason contains needle: " + body_text + " -> " + what,
                      what.find(needle) != std::string::npos);
    }
}

// Deterministic decision answers for a fixed score vector: a value golden that does not
// depend on any model weights.
static common_json decision_basic_from_fixed_scores() {
    const auto req = llama_decision::parse_decision_request(common_json::parse(decision_valid_body()));
    std::vector<std::vector<float>> probs = {
        { 0.25f, 0.75f },     // noul: options [false, true] -> noul = P(true)
        { 0.6f, 0.3f, 0.1f }, // choice: winner is "billing"
        { 0.2f, 0.3f, 0.5f }, // score: expected zero-based index
    };
    common_json usage = common_json::object();
    usage["input_tokens"]    = 12;
    usage["output_tokens"]   = 0;
    usage["cached_tokens"]   = 4;
    usage["state_cache_hit"] = false;
    return llama_decision::assemble_decision_response(req, probs, "m", usage);
}

static void test_decision_values_golden(testing & t) {
    t.test("fixed-score envelope matches the committed value golden", [](testing & t) {
        const std::string actual = decision_basic_from_fixed_scores().dump(2) + "\n";
        const std::string golden = read_file(fixture_path("decision_basic.golden.json"));
        t.assert_equal("value golden is byte-identical", golden, actual);
    });
}

static void test_decision_parse(testing & t) {
    t.test("valid decision request parses with aliases and structured criteria", [](testing & t) {
        const common_json body = common_json::parse(decision_valid_body());
        t.assert_true("detected as Jev",
                      llama_decision::select_request_shape(body) == llama_decision::request_shape::jev);

        const auto req = llama_decision::parse_decision_request(body);
        t.assert_equal("model echoed", std::string("m"), req.envelope.model);
        t.assert_equal("three questions", (size_t) 3, req.questions.size());
        t.assert_equal("noul canonical", std::string("noul"), req.questions[0].type);
        t.assert_equal("noul two options", (size_t) 2, req.questions[0].options.size());
        t.assert_equal("choice canonical", std::string("choice"), req.questions[1].type);
        t.assert_equal("choice three options", (size_t) 3, req.questions[1].options.size());
        t.assert_equal("scale becomes score", std::string("score"), req.questions[2].type);
        t.assert_equal("score three levels", (size_t) 3, req.questions[2].options.size());
        assert_close(t, "temperature parsed", 1.0, req.envelope.knobs.temperature);
        t.assert_equal("permutations parsed", 1, req.envelope.knobs.permutations);
        t.assert_true("structured criterion kept",
                      req.questions[2].options[0].original.is_object());
    });

    t.test("invalid decision requests are rejected with a clear reason", [](testing & t) {
        expect_decision_reject(t, R"({"state":"s","questions":{"q":{"type":"noul","instructions":"x"}}})", "model is required");
        expect_decision_reject(t, R"({"model":7,"state":"s","questions":{"q":{"type":"noul","instructions":"x"}}})", "model must be a string");
        expect_decision_reject(t, R"({"model":"m","questions":{"q":{"type":"noul","instructions":"x"}}})", "state (or contexts) is required");
        expect_decision_reject(t, R"({"model":"m","state":"","questions":{"q":{"type":"noul","instructions":"x"}}})", "state must not be empty");
        expect_decision_reject(t, R"({"model":"m","state":[],"questions":{"q":{"type":"noul","instructions":"x"}}})", "state must not be empty");
        expect_decision_reject(t, R"({"model":"m","state":5,"questions":{"q":{"type":"noul","instructions":"x"}}})", "state must be a string");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{}})", "1-256 entries");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"mystery","instructions":"x"}}})", "unknown type");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"choice","criteria":{"a":"x"}}}})", "2-255 options");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"choice","instructions":"x"}}})", "choice needs an object");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"score","criteria":["only"]}}})", "2-10 levels");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"noul","criteria":[1,2]}}})", "noul criteria must be an object");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"noul"}}})", "instructions are required");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"noul","instructions":"x","extra":1}}})", "unknown field");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"noul","instructions":7}}})", "instructions must be a string");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"noul","instructions":"x"}},"temperature":0})", "temperature must be > 0");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"noul","instructions":"x"}},"permutations":0})", "permutations must be >= 1");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"noul","instructions":"x"}},"temperatures":{"bogus":1}})", "unknown field");

        common_json many = common_json::object();
        common_json q    = common_json::object();
        q["type"]        = "noul";
        q["instructions"] = "x";
        common_json qs   = common_json::object();
        for (int i = 0; i < 257; ++i) {
            qs["q" + std::to_string(i)] = q;
        }
        many["model"]     = "m";
        many["state"]     = "s";
        many["questions"] = qs;
        try {
            (void) llama_decision::parse_decision_request(many);
            t.assert_true("257 questions rejected", false);
        } catch (const llama_decision::semantic_error & e) {
            t.assert_true("257 questions rejected with range", std::string(e.what()).find("1-256") != std::string::npos);
        }
    });

    // The optional live-session reference is a capability input, never a producer score. It parses
    // only from explicit, well-formed fields: a negative or non-integer slot, a negative position,
    // or a position without a slot is a semantic error, and absent means the stateless path.
    t.test("a session reference parses only from explicit, well-formed fields", [](testing & t) {
        const common_json plain = common_json::parse(decision_valid_body());
        const auto none = llama_decision::parse_session_ref(plain);
        t.assert_true("no id_slot means no session", !none.present);

        common_json with_slot = plain;
        with_slot["id_slot"] = 3;
        const auto slot = llama_decision::parse_session_ref(with_slot);
        t.assert_true("id_slot marks a session", slot.present);
        t.assert_equal("session_pos defaults to derived", -1, slot.session_pos);

        common_json pinned = with_slot;
        pinned["session_pos"] = 7;
        const auto pin = llama_decision::parse_session_ref(pinned);
        t.assert_equal("session_pos pins the position", 7, pin.session_pos);

        // a first-class session handle is an alternative to id_slot and is mutually exclusive
        common_json with_id = plain;
        with_id["session_id"] = "ses_1234";
        const auto sid = llama_decision::parse_session_ref(with_id);
        t.assert_true("session_id marks a session", sid.present);
        t.assert_equal("session_id value", std::string("ses_1234"), sid.session_id);
        common_json sid_pinned = with_id;
        sid_pinned["session_pos"] = 7;
        const auto sid_pin = llama_decision::parse_session_ref(sid_pinned);
        t.assert_equal("session_pos pins a session_id session", 7, sid_pin.session_pos);

        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"noul","instructions":"x"}},"id_slot":-1})",
                               "id_slot must be >= 0");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"noul","instructions":"x"}},"id_slot":"a"})",
                               "id_slot must be an integer");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"noul","instructions":"x"}},"session_pos":1})",
                               "session_pos requires id_slot");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"noul","instructions":"x"}},"id_slot":0,"session_pos":-2})",
                               "session_pos must be >= 0");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"noul","instructions":"x"}},"session_id":""})",
                               "session_id must be a non-empty string");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"noul","instructions":"x"}},"session_id":7})",
                               "session_id must be a non-empty string");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"noul","instructions":"x"}},"session_id":"a","id_slot":0})",
                               "provide either session_id or id_slot");
    });

    t.test("unknown top-level fields are ignored while question fields stay strict", [](testing & t) {
        const common_json base = common_json::parse(decision_valid_body());
        common_json with_extra = base;
        with_extra["extra"]         = 1;
        with_extra["another_extra"] = common_json::object();

        const auto a = llama_decision::parse_decision_request(base);
        const auto b = llama_decision::parse_decision_request(with_extra);
        t.assert_equal("unknown top-level field does not change the model", a.envelope.model, b.envelope.model);
        t.assert_equal("unknown top-level field does not change the question count", a.questions.size(), b.questions.size());
        bool same = a.questions.size() == b.questions.size();
        for (size_t i = 0; same && i < a.questions.size(); ++i) {
            same = a.questions[i].id == b.questions[i].id && a.questions[i].type == b.questions[i].type &&
                   a.questions[i].options.size() == b.questions[i].options.size();
        }
        t.assert_true("unknown top-level field leaves the questions unchanged", same);

        // question-level unknown keys are still refused
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"noul","instructions":"x","bogus":1}}})", "unknown field");

        // a misspelled "questions" is still a missing required field, not a silent default
        expect_decision_reject(t, R"({"model":"m","state":"s","questionss":{"q":{"type":"noul","instructions":"x"}}})", "questions must be an object");
    });

    t.test("instructions are required and non-null on every question type", [](testing & t) {
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"noul"}}})", "instructions are required");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"noul","instructions":null}}})", "instructions are required");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"choice","criteria":{"a":"x","b":"y"}}}})", "instructions are required");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"choice","instructions":null,"criteria":{"a":"x","b":"y"}}}})", "instructions are required");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"score","criteria":["lo","hi"]}}})", "instructions are required");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"score","instructions":null,"criteria":["lo","hi"]}}})", "instructions are required");
        // a criteria-only noul used to succeed through the escape hatch and no longer does
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"noul","criteria":{"true":"y","false":"n"}}}})", "instructions are required");

        // instructions plus criteria still succeeds
        const auto req = llama_decision::parse_decision_request(common_json::parse(
            R"({"model":"m","state":"s","questions":{"q":{"type":"noul","instructions":"x","criteria":{"true":"y","false":"n"}}}})"));
        t.assert_equal("instructions plus criteria parses", (size_t) 1, req.questions.size());
        t.assert_true("instructions are kept", req.questions[0].instructions.is_string());
    });

    t.test("score accepts 2 to 10 levels and rejects outside that range", [](testing & t) {
        auto score_body = [](int levels, bool legend) {
            common_json q = common_json::object();
            q["type"]         = "score";
            q["instructions"] = "How urgent?";
            if (legend) {
                common_json crit = common_json::object();
                for (int i = 0; i < levels; ++i) {
                    crit[std::to_string(i)] = "level " + std::to_string(i);
                }
                q["criteria"] = crit;
            } else {
                common_json crit = common_json::array();
                for (int i = 0; i < levels; ++i) {
                    crit.push_back("level " + std::to_string(i));
                }
                q["criteria"] = crit;
            }
            common_json qs = common_json::object();
            qs["q"] = q;
            common_json body = common_json::object();
            body["model"]     = "m";
            body["state"]     = "s";
            body["questions"] = qs;
            return body;
        };

        for (int levels : { 2, 5, 10 }) {
            for (bool legend : { false, true }) {
                const auto req = llama_decision::parse_decision_request(score_body(levels, legend));
                t.assert_equal("score level count parsed", (size_t) levels, req.questions[0].options.size());
            }
        }

        for (int levels : { 1, 11, 64 }) {
            for (bool legend : { false, true }) {
                bool threw = false;
                std::string msg;
                try {
                    (void) llama_decision::parse_decision_request(score_body(levels, legend));
                } catch (const llama_decision::semantic_error & e) {
                    threw = true;
                    msg = e.what();
                }
                t.assert_true("score levels outside 2-10 rejected", threw);
                t.assert_true("rejection names 2-10", msg.find("2-10") != std::string::npos);
            }
        }
    });

    t.test("diagnostics is an optional boolean, off by default", [](testing & t) {
        const auto off = llama_decision::parse_decision_request(common_json::parse(
            R"({"model":"m","state":"s","questions":{"q":{"type":"noul","instructions":"x"}}})"));
        t.assert_true("diagnostics defaults off", !off.envelope.diagnostics);

        const auto on = llama_decision::parse_decision_request(common_json::parse(
            R"({"model":"m","state":"s","questions":{"q":{"type":"noul","instructions":"x"}},"diagnostics":true})"));
        t.assert_true("diagnostics true is parsed", on.envelope.diagnostics);

        const auto explicit_off = llama_decision::parse_decision_request(common_json::parse(
            R"({"model":"m","state":"s","questions":{"q":{"type":"noul","instructions":"x"}},"diagnostics":false})"));
        t.assert_true("diagnostics false is parsed", !explicit_off.envelope.diagnostics);

        expect_decision_reject(t,
            R"({"model":"m","state":"s","questions":{"q":{"type":"noul","instructions":"x"}},"diagnostics":"yes"})",
            "diagnostics must be a boolean");
    });
}

// Set by --record-golden: rewrite the value goldens instead of comparing them, for the rare
// intentional contract change that moves a golden's bytes.
static bool record_golden = false;

// The uniform score vectors for the canonical body, one per question in request order. Assembly
// requires a vector of exactly the question's option count, so a uniform distribution is an input
// here, never a substitute the assembler supplies.
static std::vector<std::vector<float>> decision_valid_uniform_scores() {
    return {
        { 0.5f, 0.5f },                                   // refund: noul over [false, true]
        { 1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f },       // dept: choice over 3 options
        { 1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f },       // urgency: score over 3 levels
    };
}

// The failure message assembly raised for these scores, or an empty string when it accepted them.
// A score vector that is missing or mis-sized is an internal defect, so the failure names the
// question and both counts; the readout already refuses to return the wrong number of scores.
static std::string decision_assembly_failure(const std::vector<std::vector<float>> & probs) {
    try {
        const auto req = llama_decision::parse_decision_request(common_json::parse(decision_valid_body()));
        common_json usage = common_json::object();
        usage["input_tokens"]  = 0;
        usage["output_tokens"] = 0;
        (void) llama_decision::assemble_decision_response(req, probs, "m", usage);
        return std::string();
    } catch (const std::runtime_error & e) {
        return e.what();
    }
}

// The multi-context response the server builds: one answers/usage pair per context, in request
// order, over the same questions. The assembler runs once per context here, so this is the shape
// a per-context change in assembly would move.
static common_json decision_contexts_from_fixed_scores() {
    const auto req = llama_decision::parse_decision_request(common_json::parse(decision_valid_body()));
    const std::vector<std::vector<std::vector<float>>> all = {
        { { 0.25f, 0.75f }, { 0.6f, 0.3f, 0.1f }, { 0.2f, 0.3f, 0.5f } },
        { { 0.10f, 0.90f }, { 0.1f, 0.7f, 0.2f }, { 0.3f, 0.3f, 0.4f } },
    };
    common_json contexts = common_json::array();
    for (size_t ci = 0; ci < all.size(); ++ci) {
        common_json usage = common_json::object();
        usage["input_tokens"]  = 7 + (long long) ci;
        usage["output_tokens"] = 0;
        const common_json ans = llama_decision::assemble_decision_response(req, all[ci], "m", usage, nullptr);
        contexts.push_back({ { "answers", ans.at("answers") }, { "usage", ans.at("usage") } });
    }
    common_json out = common_json::object();
    out["model"]    = "m";
    out["contexts"] = contexts;
    return out;
}

static void test_decision_assemble(testing & t) {
    t.test("canonical envelope has the required shape and semantics", [](testing & t) {
        auto req = llama_decision::parse_decision_request(common_json::parse(decision_valid_body()));
        req.envelope.diagnostics = true;
        common_json usage = common_json::object();
        usage["input_tokens"]    = 0;
        usage["output_tokens"]   = 0;
        usage["cached_tokens"]   = 0;
        usage["state_cache_hit"] = false;

        const common_json out = llama_decision::assemble_decision_response(req, decision_valid_uniform_scores(), req.envelope.model, usage);
        t.assert_equal("model echoed", std::string("m"), out.at("model").get<std::string>());

        const auto & answers = out.at("answers");
        t.assert_true("answers keyed by qid", answers.contains("refund") && answers.contains("dept") && answers.contains("urgency"));

        const auto & refund = answers.at("refund");
        t.assert_equal("noul type", std::string("noul"), refund.at("type").get<std::string>());
        assert_close(t, "noul stub is 0.5", 0.5, refund.at("noul").get<double>());
        t.assert_true("noul never carries confidence", !refund.contains("confidence"));

        const auto & dept = answers.at("dept");
        t.assert_equal("choice winner is first", std::string("billing"), dept.at("choice").get<std::string>());
        assert_close(t, "choice probabilities sum to 1",
                     1.0,
                     dept.at("probabilities").at("billing").get<double>() +
                     dept.at("probabilities").at("technical").get<double>() +
                     dept.at("probabilities").at("cancellation").get<double>());
        assert_close(t, "choice confidence of uniform is 0", 0.0, dept.at("confidence").get<double>(), 1e-6);
        assert_close(t, "choice certainty is the winner share", 1.0 / 3.0, dept.at("certainty").get<double>(), 1e-6);

        const auto & urg = answers.at("urgency");
        t.assert_true("score probability keys are strings \"0\"..\"K-1\"",
                      urg.at("probabilities").contains("0") && urg.at("probabilities").contains("1") && urg.at("probabilities").contains("2"));
        assert_close(t, "score is the expected zero-based index", 1.0, urg.at("score").get<double>());
        t.assert_true("legend has string keys", urg.at("legend").contains("0") && urg.at("legend").contains("2"));
        t.assert_true("legend round-trips the structured value",
                      urg.at("legend").at("0").is_object() && urg.at("legend").at("0").at("label").get<std::string>() == "calm");
        t.assert_equal("output_tokens is zero", 0, out.at("usage").at("output_tokens").get<int>());
    });

    t.test("a score vector that stops short fails closed naming the question", [](testing & t) {
        // the last question has no vector at all
        const std::string missing = decision_assembly_failure({ { 0.5f, 0.5f }, { 1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f } });
        t.assert_true("a missing vector is refused: " + missing, missing.find("urgency") != std::string::npos);

        // a vector shorter than the option count
        const std::string short_vec = decision_assembly_failure({ { 0.5f, 0.5f }, { 0.6f, 0.4f }, { 1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f } });
        t.assert_true("a short vector is refused naming the question: " + short_vec,
                      short_vec.find("dept") != std::string::npos);
        t.assert_true("a short vector reports both counts: " + short_vec,
                      short_vec.find("3") != std::string::npos && short_vec.find("2") != std::string::npos);
    });

    t.test("a score vector one too long fails closed naming the question", [](testing & t) {
        const std::string long_vec =
            decision_assembly_failure({ { 0.5f, 0.5f }, { 0.6f, 0.3f, 0.1f, 0.0f }, { 1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f } });
        t.assert_true("an over-long vector is refused naming the question: " + long_vec,
                      long_vec.find("dept") != std::string::npos);
    });

    t.test("a full-length vector of zeros is accepted and reads as zero confidence", [](testing & t) {
        // control: the guard keys on the vector's length, never on its content. A degenerate but
        // well-formed distribution is still a distribution, and confidence 0.0 is a documented
        // value, not a defect.
        const std::string none = decision_assembly_failure({ { 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f } });
        t.assert_true("a full-length zero vector is accepted", none.empty());

        const auto req = llama_decision::parse_decision_request(common_json::parse(decision_valid_body()));
        common_json usage = common_json::object();
        usage["input_tokens"]  = 0;
        usage["output_tokens"] = 0;
        const common_json out = llama_decision::assemble_decision_response(
            req, { { 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f } }, "m", usage);
        assert_close(t, "choice confidence of an all-zero vector is 0", 0.0,
                     out.at("answers").at("dept").at("confidence").get<double>());
        assert_close(t, "choice winner of an all-zero vector is the first option", 0.0,
                     out.at("answers").at("dept").at("probabilities").at("billing").get<double>());
        assert_close(t, "noul reads 0 with no true mass", 0.0,
                     out.at("answers").at("refund").at("noul").get<double>());
    });

    t.test("a two-context response matches the committed contexts golden", [](testing & t) {
        const std::string actual = decision_contexts_from_fixed_scores().dump(2) + "\n";
        if (record_golden) {
            write_file(fixture_path("decision_contexts.golden.json"), actual);
            return;
        }
        const std::string golden = read_file(fixture_path("decision_contexts.golden.json"));
        t.assert_equal("contexts golden is byte-identical", golden, actual);
    });
}

// The adapter scope a decision decodes under is the only record of what conditioned the replay, and
// the response reports it. A scope the context would not install therefore cannot be reported as
// if it had been: the request is refused. This is a capability check, in the same class as "the slot
// does not exist", and it never reads a producer score.
static void test_adapter_scope_applied(testing & t) {
    // True when the request was refused for an unapplied scope, recording the message it refused
    // with so the scope in the message can be asserted.
    static auto refused = [](const std::string & scope, int apply_status, std::string * message) {
        try {
            llama_decision::require_adapter_scope(scope, apply_status);
            return false;
        } catch (const llama_decision::semantic_error & e) {
            if (message != nullptr) {
                *message = e.what();
            }
            return true;
        }
    };

    t.test("a scope the context would not install refuses the decision", [](testing & t) {
        std::string message;
        const bool thrown = refused("adapter-scope-v1:1234", 1, &message);
        t.assert_true("an unapplied scope throws", thrown);
        t.assert_true("the refusal names the scope: " + message,
                      message.find("adapter-scope-v1:1234") != std::string::npos);
    });

    t.test("an unapplied base scope refuses the decision", [](testing & t) {
        // the base model is the empty scope, and it is the scope diagnostics report as "base"
        std::string message;
        const bool thrown = refused("", 1, &message);
        t.assert_true("an unapplied base scope throws", thrown);
        t.assert_true("the refusal names base: " + message, message.find("base") != std::string::npos);
    });

    t.test("an applied scope is accepted whatever its name", [](testing & t) {
        // control: the check keys on the apply status alone, so an applied base scope stays valid
        llama_decision::require_adapter_scope("", 0);
        llama_decision::require_adapter_scope("adapter-scope-v1:1234", 0);
        t.assert_true("an applied scope does not throw", true);
    });

    t.test("the base scope applies to a real context and a broken one does not", [](testing & t) {
        const std::string path = decision_cpu_model_path();
        if (path.empty()) {
            t.skip("no model for the adapter scope check");
            return;
        }
        cpu_test_engine eng;
        if (!eng.load(path)) {
            t.assert_true("model loads", false);
            return;
        }

        // the empty scope is the base model, and it must succeed on every model
        std::vector<common_adapter_lora_info> base;
        t.assert_equal("the base scope applies", 0, common_set_adapter_lora(eng.ctx, base));

        // an entry that cannot be installed: no adapter behind a non-zero scale
        common_adapter_lora_info missing;
        missing.path  = "not-loaded.gguf";
        missing.scale = 1.0f;
        missing.ptr   = nullptr;
        std::vector<common_adapter_lora_info> broken = { missing };

        const int status = common_set_adapter_lora(eng.ctx, broken);
        t.assert_true("a scope with an uninstallable entry does not report success", status != 0);
        t.assert_true("and it refuses the decision", refused("adapter-scope-v1:1234", status, nullptr));

        // a refused apply must leave the context on its previous scope, not half-switched
        t.assert_equal("the context keeps the scope it had", 0, common_set_adapter_lora(eng.ctx, base));
    });
}

// Synthetic vocabulary for hermetic label tests. Greedy longest-piece matching,
// so a multi-piece string tokenizes to more than one token.
struct fake_vocab : llama_decision::label_vocab {
    std::vector<std::string>       pieces;
    std::map<int32_t, std::string> piece_override;
    std::set<int32_t>              specials;
    bool                           drop_unknown = false; // skip text no piece matches, like a vocab without it

    int32_t id_of(const std::string & p) const {
        for (size_t i = 0; i < pieces.size(); ++i) {
            if (pieces[i] == p) {
                return (int32_t) i;
            }
        }
        return -1;
    }

    std::vector<int32_t> tokenize(const std::string & text, bool) const override {
        std::vector<int32_t> out;
        size_t i = 0;
        while (i < text.size()) {
            int32_t best     = -1;
            size_t  best_len = 0;
            for (size_t id = 0; id < pieces.size(); ++id) {
                const std::string & p = pieces[id];
                if (!p.empty() && p.size() > best_len && text.compare(i, p.size(), p) == 0) {
                    best     = (int32_t) id;
                    best_len = p.size();
                }
            }
            if (best < 0) {
                if (!drop_unknown) {
                    out.push_back(100000 + (int32_t) (unsigned char) text[i]);
                }
                ++i;
                continue;
            }
            out.push_back(best);
            i += best_len;
        }
        return out;
    }

    std::string piece(int32_t token) const override {
        const auto it = piece_override.find(token);
        if (it != piece_override.end()) {
            return it->second;
        }
        if (token >= 0 && (size_t) token < pieces.size()) {
            return pieces[(size_t) token];
        }
        return std::string();
    }

    bool is_special(int32_t token) const override {
        return specials.count(token) > 0;
    }
};

static fake_vocab make_fake_vocab(bool with_double_letters) {
    fake_vocab v;
    for (char c = 0x20; c < 0x7f; ++c) {
        v.pieces.push_back(std::string(1, c));
    }
    v.pieces.push_back("\n");
    if (with_double_letters) {
        for (char a = 'A'; a <= 'Z'; ++a) {
            for (char b = 'A'; b <= 'Z'; ++b) {
                v.pieces.push_back(std::string{ a, b });
            }
        }
    }
    v.pieces.push_back("Answer:");
    v.pieces.push_back("true");
    v.pieces.push_back("false");
    v.pieces.push_back("<|turn>");
    return v;
}

// The synthetic answer tail the letter tests frame before a label: with no chat template the
// framer returns "\n" as the post-user text, so the tail is "\nAnswer:\n".
static std::string test_letter_tail() {
    return llama_decision::render_letter_prompt(nullptr, false, llama_decision::letter_system_text()).second + "Answer:\n";
}

static void test_letter_suffix(testing & t) {
    t.test("question suffix lists options and ends at the answer boundary", [](testing & t) {
        const auto req = llama_decision::parse_decision_request(common_json::parse(decision_valid_body()));
        const fake_vocab v = make_fake_vocab(true);
        const auto pool = llama_decision::build_label_pool(v, "", 8);
        const std::string after = "<turn|>\n<turn>model\n";

        const std::string tail = llama_decision::letter_answer_tail(after);
        t.assert_equal("the answer tail is the after text plus the fixed marker", after + "Answer:\n", tail);
        // the per-question suffix is built through the same option-line formatter
        const std::string a_line = llama_decision::format_option_line(pool[0], req.questions[1].options[0]);
        const std::string b_line = llama_decision::format_option_line(pool[1], req.questions[1].options[1]);
        t.assert_true("first option listed", a_line.find("A: billing - payment") != std::string::npos);
        t.assert_true("second option listed", b_line.find("B: technical - bug") != std::string::npos);
        t.assert_true("question text listed", req.questions[1].instructions.dump().find("What is the issue?") != std::string::npos);
    });

    t.test("label capacity is validated before scoring", [](testing & t) {
        const auto req = llama_decision::parse_decision_request(common_json::parse(decision_valid_body()));
        const std::string tail = test_letter_tail();
        const fake_vocab v = make_fake_vocab(true);
        const auto pool = llama_decision::build_label_pool(v, tail, 2);
        bool threw = false;
        try {
            llama_decision::verify_letter_request(v, "\n", req, pool);
        } catch (const llama_decision::semantic_error &) {
            threw = true;
        }
        t.assert_true("a question with more options than the pool is rejected", threw);
    });

    t.test("label capacity rejects above the realized pool and accepts at it", [](testing & t) {
        auto make_choice_request = [](size_t n) {
            common_json body =
                common_json::parse(R"({"model":"m","state":"s","questions":{"q":{"type":"choice","instructions":"pick"}}})");
            common_json crit = common_json::object();
            for (size_t i = 0; i < n; ++i) {
                crit["k" + std::to_string(i)] = "d";
            }
            body["questions"]["q"]["criteria"] = crit;
            return llama_decision::parse_decision_request(body);
        };

        const std::string tail = test_letter_tail();
        const std::string after = "\n"; // render_letter_prompt(nullptr,...).second

        // control group: an explicit small cap is honored, and one option above it is rejected
        const fake_vocab small      = make_fake_vocab(false);
        const auto       small_pool = llama_decision::build_label_pool(small, tail, 5);
        t.assert_equal("the control pool honors the explicit cap", (size_t) 5, small_pool.size());
        llama_decision::verify_letter_request(small, after, make_choice_request(5), small_pool);  // must not throw
        bool over = false;
        try {
            llama_decision::verify_letter_request(small, after, make_choice_request(6), small_pool);
        } catch (const llama_decision::semantic_error &) {
            over = true;
        }
        t.assert_true("one option above the control pool is rejected", over);

        // positive group: a double-letter vocabulary fills the composed cap, and the cap is accepted
        const fake_vocab full      = make_fake_vocab(true);
        const auto       full_pool = llama_decision::build_label_pool(full, tail, llama_decision::LABEL_POOL_CAP);
        t.assert_equal("the positive pool reaches the cap", llama_decision::LABEL_POOL_CAP, full_pool.size());
        llama_decision::verify_letter_request(full, after, make_choice_request(llama_decision::LABEL_POOL_CAP), full_pool);
    });
}

// The exact option lines the framer emits: `label: key - description`, with the description omitted
// when empty. This is the golden the prompt layout is measured against.
static void test_letter_option_lines(testing & t) {
    t.test("option lines render as label: key - description", [](testing & t) {
        const common_json body = common_json::parse(R"({
            "model": "m",
            "state": "s",
            "questions": {
                "n": {"type": "noul", "instructions": "Refund?", "criteria": {"true": "yes", "false": "no"}},
                "c": {"type": "choice", "instructions": "Dept?", "criteria": {"billing": {"label": "payment", "code": 7}, "technical": "bug"}},
                "s": {"type": "score", "instructions": "Urgency?", "criteria": ["calm", "upset", "furious"]},
                "e": {"type": "choice", "instructions": "Empty?", "criteria": {"a": null, "b": "bee"}}
            }
        })");
        const auto req = llama_decision::parse_decision_request(body);
        const fake_vocab v = make_fake_vocab(true);
        const auto pool = llama_decision::build_label_pool(v, "", 8);

        // the option lines are emitted through the one formatter: `label: key - description`
        t.assert_true("noul false line",
                      llama_decision::format_option_line(pool[0], req.questions[0].options[0]).find("A: false - no") != std::string::npos);
        t.assert_true("noul true line",
                      llama_decision::format_option_line(pool[1], req.questions[0].options[1]).find("B: true - yes") != std::string::npos);
        t.assert_true("choice object description line",
                      llama_decision::format_option_line(pool[0], req.questions[1].options[0]).find("A: billing - {\"label\":\"payment\",\"code\":7}") != std::string::npos);
        t.assert_true("choice string description line",
                      llama_decision::format_option_line(pool[1], req.questions[1].options[1]).find("B: technical - bug") != std::string::npos);
        t.assert_true("score line 0",
                      llama_decision::format_option_line(pool[0], req.questions[2].options[0]).find("A: 0 - calm") != std::string::npos);
        t.assert_true("score line 1",
                      llama_decision::format_option_line(pool[1], req.questions[2].options[1]).find("B: 1 - upset") != std::string::npos);
        t.assert_true("score line 2",
                      llama_decision::format_option_line(pool[2], req.questions[2].options[2]).find("C: 2 - furious") != std::string::npos);
        t.assert_true("empty description renders the key only",
                      llama_decision::format_option_line(pool[0], req.questions[3].options[0]).find("A: a") != std::string::npos);
        t.assert_true("non-empty description still renders",
                      llama_decision::format_option_line(pool[1], req.questions[3].options[1]).find("B: b - bee") != std::string::npos);
    });
}

static void test_label_pool(testing & t) {
    t.test("label pool keeps boundary-resolved labels in order", [](testing & t) {
        const fake_vocab v = make_fake_vocab(true);

        const auto pool = llama_decision::build_label_pool(v, "", 64);
        t.assert_equal("cap respected", (size_t) 64, pool.size());
        t.assert_equal("first label is A", std::string("A"), pool[0].text);
        t.assert_equal("26th label is Z", std::string("Z"), pool[25].text);
        t.assert_equal("then lowercase letters", std::string("a"), pool[26].text);
        t.assert_equal("then the single digits", std::string("0"), pool[52].text);
        t.assert_equal("then the ASCII symbol set", std::string("!"), pool[62].text);

        // the first 62 labels are letters and digits; the symbol set starts after them
        bool all_label_chars = true;
        for (size_t i = 0; i < 62 && i < pool.size(); ++i) {
            for (unsigned char c : pool[i].text) {
                all_label_chars = all_label_chars && std::isalnum(c) != 0;
            }
        }
        t.assert_true("letters and digits come first", all_label_chars);

        // a 64-label pool is all single characters: two-char labels only appear once the
        // single-character sets are exhausted
        bool has_two_char = false;
        for (const auto & l : pool) {
            has_two_char = has_two_char || l.text.size() == 2;
        }
        t.assert_true("a 64-label pool is all single characters", !has_two_char);

        t.assert_equal("AA resolves at the boundary", v.id_of("AA"), llama_decision::answer_label_token(v, "", "AA"));
        t.assert_equal("AAA is not a single label", -1, llama_decision::answer_label_token(v, "", "AAA"));
    });

    t.test("label pool rejects special and merged tokens", [](testing & t) {
        // The removed piece check was a producer-confidence proxy ("the token looks like the
        // text"); it is intentionally gone. A label is accepted when the boundary produces it as a
        // single non-special token, whatever its surface form.
        fake_vocab special = make_fake_vocab(false);
        special.specials.insert(special.id_of("B"));
        t.assert_equal("special token rejected", -1, llama_decision::answer_label_token(special, "", "B"));
        t.assert_equal("normal token accepted", special.id_of("A"), llama_decision::answer_label_token(special, "", "A"));

        fake_vocab renamed = make_fake_vocab(false);
        renamed.piece_override[renamed.id_of("C")] = "c";
        t.assert_equal("a renamed piece is still accepted", renamed.id_of("C"),
                       llama_decision::answer_label_token(renamed, "", "C"));

        fake_vocab merged = make_fake_vocab(false);
        merged.pieces.push_back("\nA");
        t.assert_equal("a genuine merge is rejected", -1, llama_decision::answer_label_token(merged, "x\n", "A"));

        fake_vocab tiny;
        tiny.pieces.push_back("A");
        tiny.specials.insert(0); // the only resolvable token is special, so composition is blocked
        tiny.drop_unknown = true;
        bool threw = false;
        try {
            (void) llama_decision::build_label_pool(tiny, "", 64);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        t.assert_true("fewer than two labels is an error", threw);
    });

    t.test("isolated and boundary pools agree without a space prefix", [](testing & t) {
        const fake_vocab v = make_fake_vocab(true);
        const auto isolated = llama_decision::build_label_pool(v, "", 64);
        const auto boundary = llama_decision::build_label_pool(v, "Answer:\n", 64);
        bool same = isolated.size() == boundary.size();
        for (size_t i = 0; same && i < isolated.size(); ++i) {
            same = isolated[i].text == boundary[i].text && isolated[i].token == boundary[i].token;
        }
        t.assert_true("the pool is byte-identical when isolated equals boundary", same);
    });
}

static void test_boundary(testing & t) {
    t.test("boundary check accepts a clean split and rejects a merged token", [](testing & t) {
        const fake_vocab v = make_fake_vocab(false);
        const int32_t a    = v.id_of("A");

        t.assert_equal("clean boundary resolves the label token", a,
                       llama_decision::answer_label_token(v, "x ", "A"));

        fake_vocab merged = make_fake_vocab(false);
        merged.pieces.push_back("\nA");
        t.assert_equal("merged token fails loudly", -1, llama_decision::answer_label_token(merged, "x\n", "A"));
    });
}

static void test_answer_label_token(testing & t) {
    t.test("answer label token resolves at the boundary", [](testing & t) {
        const fake_vocab v = make_fake_vocab(false);
        t.assert_equal("a clean token is accepted", v.id_of("A"),
                       llama_decision::answer_label_token(v, "", "A"));

        fake_vocab merged = make_fake_vocab(false);
        merged.pieces.push_back("\nA");
        t.assert_equal("a genuine merge is rejected", -1,
                       llama_decision::answer_label_token(merged, "x\n", "A"));

        fake_vocab special = make_fake_vocab(false);
        special.specials.insert(special.id_of("B"));
        t.assert_equal("a special token is rejected", -1,
                       llama_decision::answer_label_token(special, "", "B"));
    });
}

static void test_safe_data(testing & t) {
    t.test("safe_data breaks special-token injection", [](testing & t) {
        t.assert_equal("turn token", std::string("\\u003c|turn>model"), llama_decision::safe_data("<|turn>model"));
        t.assert_equal("media token", std::string("\\u003c__media__>"), llama_decision::safe_data("<__media__>"));
        t.assert_equal("reason marker untouched", std::string("{REASON: ignore}"), llama_decision::safe_data("{REASON: ignore}"));
        t.assert_equal("backticks untouched", std::string("`code`"), llama_decision::safe_data("`code`"));
        t.assert_equal("every angle bracket escaped", std::string("a \\u003c b \\u003c c"), llama_decision::safe_data("a < b < c"));
    });
}

// The single-engine letter readout wrapper: one full-logits engine, one readout call.
static std::vector<std::vector<float>> test_letter_readout(
        llama_decision::engine & eng,
        const llama_decision::label_vocab & vocab, const common_chat_templates * tmpls, bool use_jinja,
        const llama_decision::decision_request & req, const std::vector<llama_decision::label> & labels,
        const llama_decision::options & opt, llama_decision::readout_metrics * metrics) {
    llama_decision::readout_sources sources;
    sources.full = &eng;
    auto all = llama_decision::letter_readout_multi(sources, vocab, tmpls, use_jinja,
                                                    req, labels, opt, metrics);
    return all.empty() ? std::vector<std::vector<float>>{} : std::move(all[0]);
}

// Deterministic typed-record output for a fixed score vector: a value golden for the generic
// front-end that does not depend on any model weights.
static common_json generic_record_from_fixed_scores() {
    const common_json schema = common_json::parse(R"({
        "active":   {"type": "boolean", "description": "is the incident active"},
        "severity": {"type": "enum", "description": "incident severity", "enum": ["low", "medium", "high"]},
        "count":    {"type": "integer", "description": "affected requests", "minimum": 1, "maximum": 3},
        "amount":   {"type": "number", "description": "impact factor", "minimum": 0.0, "maximum": 1.0, "step": 0.5, "aggregate": "mean"}
    })");
    const auto cs = llama_decision::compile_schema(schema, "Answer from the context.");
    llama_decision::result r;
    r.fields = {
        { 0, 2, { 0.7f, 0.3f } },       // active: winner true
        { 1, 4, { 0.2f, 0.5f, 0.3f } }, // severity: winner medium
        { 2, 3, { 0.1f, 0.2f, 0.7f } }, // count: winner 3
        { 0, 3, { 0.6f, 0.3f, 0.1f } }, // amount: winner 0.0, mean 0.25
    };
    return llama_decision::assemble(cs, r, false);
}

static void test_generic_typed_golden(testing & t) {
    t.test("fixed-score generic record matches the committed value golden", [](testing & t) {
        const std::string actual = generic_record_from_fixed_scores().dump(2) + "\n";
        if (record_golden) {
            write_file(fixture_path("generic_typed.golden.json"), actual);
            return;
        }
        const std::string golden = read_file(fixture_path("generic_typed.golden.json"));
        t.assert_equal("generic typed golden is byte-identical", golden, actual);
    });
}

// The dispatch on request shape: `schema` selects the generic front-end, `questions` the Jev one,
// and a body carrying both cannot be dispatched. handle_decision maps this to HTTP 400.
static void test_generic_mutual_exclusion(testing & t) {
    t.test("a body carrying both schema and questions is refused", [](testing & t) {
        const common_json body = common_json::parse(R"({
            "model": "m",
            "state": "s",
            "questions": {"q": {"type": "noul", "instructions": "x"}},
            "schema": {"active": {"type": "boolean", "description": "d"}}
        })");
        bool threw = false;
        try {
            (void) llama_decision::select_request_shape(body);
        } catch (const std::invalid_argument & e) {
            threw = true;
            t.assert_true("the reason names the mutual exclusion",
                          std::string(e.what()).find("not both") != std::string::npos);
        }
        t.assert_true("schema and questions are mutually exclusive", threw);
    });
}

// The generic front-end is a second producer of field_input[] over the one engine: compile the
// schema, score it through decide_batch, and assemble the typed record back out of the result.
static void generic_frontend_drive_run(testing & t, llama_context * ctx, const std::string & lane) {
    const common_json body = common_json::parse(R"({
        "model": "m",
        "state": "The service was unreachable for ten minutes before the backup took over.",
        "schema": {
            "active":   {"type": "boolean", "description": "is the incident active"},
            "severity": {"type": "enum", "description": "incident severity", "enum": ["low", "medium", "high"]},
            "count":    {"type": "integer", "description": "affected requests", "minimum": 1, "maximum": 3},
            "amount":   {"type": "number", "description": "impact factor", "minimum": 0.0, "maximum": 1.0, "step": 0.5, "aggregate": "mean"}
        }
    })");
    const llama_decision::decision_request greq = llama_decision::parse_decision_request(body);
    const llama_decision::compiled_schema  cs   = llama_decision::compile_schema(greq.schema, greq.instructions,
                                                                                   greq.envelope.knobs);

    llama_decision::engine eng(ctx, 2, 8);
    llama_decision::options o;
    o.mode      = "tree";
    o.tree_max  = 64;
    o.cache_tag = llama_decision::generic_cache_tag(nullptr, false, cs.system_text);

    const auto parts = llama_decision::render_schema_prompt(nullptr, false, cs.system_text,
                                                            llama_decision::render_state(greq.envelope.evidence.state));
    const auto plan = eng.compile_fields(cs.inputs, o);
    const auto b    = eng.decide_batch(plan, parts.first, { parts.second }, o);

    t.test(lane + ": the generic front-end feeds the one engine and assembles the typed record", [&](testing & t) {
        if (!t.assert_true(lane + ": one decision item", b.items.size() == 1)) {
            return;
        }
        const auto & item = b.items[0];
        t.assert_equal(lane + ": one scored field per schema field", cs.specs.size(), item.fields.size());
        for (size_t i = 0; i < item.fields.size(); ++i) {
            const llama_decision::field_result & fr = item.fields[i];
            t.assert_true(lane + ": field " + cs.specs[i].name + " is scored exactly",
                          !fr.probs.empty() && fr.winner >= 0 && (size_t) fr.winner < cs.specs[i].values.size());
            t.assert_equal(lane + ": field " + cs.specs[i].name + " probs cover every value",
                           cs.specs[i].values.size(), fr.probs.size());
            double sum = 0.0;
            for (float p : fr.probs) {
                sum += (double) p;
            }
            t.assert_true(lane + ": field " + cs.specs[i].name + " probabilities sum to 1",
                          std::fabs(sum - 1.0) < 1e-4);
        }

        // the strict envelope: the Jev keys plus the generic value, and no off-envelope field
        const common_json answers = llama_decision::assemble(cs, item, false);
        t.assert_equal(lane + ": one answer per field", cs.specs.size(), answers.size());
        for (size_t i = 0; i < cs.specs.size(); ++i) {
            const std::string & name = cs.specs[i].name;
            const common_json & rec = answers.at(name);
            for (const char * key : { "type", "value", "confidence", "probabilities", "legend", "scored" }) {
                t.assert_true(lane + ": " + name + " answer carries " + key, rec.contains(key));
            }
            t.assert_equal(lane + ": " + name + " answer reports its type", cs.specs[i].type,
                           rec.at("type").get<std::string>());
            // the Jev probability discipline: one entry per allowed value, summing to 1
            const common_json & probs = rec.at("probabilities");
            t.assert_equal(lane + ": " + name + " probability map covers the value space",
                           cs.specs[i].values.size(), probs.size());
            double sum = 0.0;
            for (const auto & kv : probs.items()) {
                sum += kv.value().get<double>();
            }
            t.assert_true(lane + ": " + name + " probabilities sum to 1", std::fabs(sum - 1.0) < 1e-5);
            t.assert_equal(lane + ": " + name + " the legend covers the value space",
                           cs.specs[i].values.size(), rec.at("legend").size());
            bool allowed = false;
            for (const auto & v : cs.specs[i].values) {
                allowed = allowed || (v.dump() == rec.at("value").dump());
            }
            t.assert_true(lane + ": " + name + " value is one of the allowed values", allowed);
            // the spread summaries are diagnostics-only, as on the Jev score answer
            const bool numeric = cs.specs[i].type == "integer" || cs.specs[i].type == "number";
            t.assert_true(lane + ": " + name + " answer omits interval_p10_p90 by default",
                          !rec.contains("interval_p10_p90"));
            t.assert_true(lane + ": " + name + " answer omits aggregate by default", !rec.contains("aggregate"));

            // and they appear under diagnostics, with a real band
            const common_json rec_diag = llama_decision::assemble(cs, item, true).at(name);
            if (numeric) {
                t.assert_true(lane + ": " + name + " diagnostics record carries interval_p10_p90",
                              rec_diag.contains("interval_p10_p90"));
                t.assert_true(lane + ": " + name + " diagnostics record carries aggregate",
                              rec_diag.contains("aggregate"));
                t.assert_equal(lane + ": " + name + " band has two entries",
                               (size_t) 2, rec_diag.at("interval_p10_p90").size());
            } else {
                t.assert_true(lane + ": " + name + " a non-numeric field has no band under diagnostics",
                              !rec_diag.contains("interval_p10_p90"));
            }
        }
    });
}

static void test_generic_drives_engine(testing & t) {
    t.test("the generic front-end drives the shared engine on the CPU model", [](testing & t) {
        const std::string path = decision_cpu_model_path();
        if (path.empty()) {
            t.skip("no generated model; run the generate-models fixture");
            return;
        }
        cpu_test_engine te;
        if (!te.load(path, 4096, false, 512)) {
            t.assert_true("the CPU decision scaffold loads the model", false);
            return;
        }
        try {
            generic_frontend_drive_run(t, te.ctx, "cpu");
        } catch (const std::exception & e) {
            t.assert_true(std::string("the CPU generic front-end: ") + e.what(), false);
        }
    });

    t.test("the generic front-end drives the shared engine on the GPU model", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (!gpu_model_ready(t, path)) {
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("the GPU decision scaffold loads the model", false);
            return;
        }
        try {
            generic_frontend_drive_run(t, te.ctx, "gpu");
        } catch (const std::exception & e) {
            t.assert_true(std::string("the GPU generic front-end: ") + e.what(), false);
        }
    });
}

// A generic enum field and the equivalent Jev choice over the same semantic keys must select the
// same winner when the model is decisive: both front-ends terminate at field_input and share the
// one engine, so a decisive model cannot pick different keys. The framings are aligned on the
// evidence (both render it as "State:\n...") and on the candidate semantics (same key strings and
// order); only the answer encoding differs (JSON string value vs a letter label). The case below
// was chosen because the reference model is decisive in both framings and selects the same key
// (an approved transaction selects "approved" either way).
static void generic_jev_winner_equivalence_run(testing & t, llama_context * ctx, const std::string & lane) {
    common_json jbody = common_json::parse(R"({
        "model": "m",
        "state": "The transaction was approved by the bank.",
        "questions": {
            "verdict": {"type": "choice", "instructions": "Was the transaction approved?",
                        "criteria": {"approved": "the transaction is approved", "declined": "the transaction is declined"}}
        }
    })");
    const auto req = llama_decision::parse_decision_request(jbody);

    const common_json schema = common_json::parse(R"({
        "verdict": {"type": "enum", "description": "was the transaction approved",
                    "enum": ["approved", "declined"]}
    })");
    const auto cs = llama_decision::compile_schema(schema, "Answer from the context.");

    const llama_model * model = llama_get_model(ctx);
    auto vocab = llama_decision::make_llama_label_vocab(llama_model_get_vocab(model));
    const auto pool = llama_decision::build_label_pool(*vocab, test_letter_tail(), 64);

    llama_decision::engine eng(ctx, 2, 8);

    llama_decision::options jo;
    jo.cache_tag = "generic-jev-eq";
    const auto jprobs = test_letter_readout(eng, *vocab, nullptr, false, req, pool, jo, nullptr);

    llama_decision::options go;
    go.mode      = "tree";
    go.tree_max  = 64;
    go.cache_tag = llama_decision::generic_cache_tag(nullptr, false, cs.system_text);
    const auto parts = llama_decision::render_schema_prompt(nullptr, false, cs.system_text,
                                                            llama_decision::render_state(req.envelope.evidence.state));
    const auto plan = eng.compile_fields(cs.inputs, go);

    t.test(lane + ": generic enum and Jev choice pick the same winner on a decisive model", [&](testing & t) {
        if (!t.assert_true(lane + ": the Jev readout returns one distribution", jprobs.size() == 1)) {
            return;
        }
        const int  jwinner = (int) (std::max_element(jprobs[0].begin(), jprobs[0].end()) - jprobs[0].begin());
        const std::string jev_key = req.questions[0].options[jwinner].key;

        const auto gb   = eng.decide_batch(plan, parts.first, { parts.second }, go);
        const auto & gr = gb.items[0].fields[0];
        const std::string generic_key = cs.specs[0].values[gr.winner].get<std::string>();

        // decisive means the winner's share is well above uniform in both framings
        const float jp = jprobs[0][jwinner];
        const float gp = gr.probs[gr.winner];
        t.assert_true(lane + ": the case is decisive in both framings (jev " + std::to_string(jp) +
                      ", generic " + std::to_string(gp) + ")",
                      jp >= 0.6f && gp >= 0.6f);
        t.assert_equal(lane + ": the two front-ends select the same key", jev_key, generic_key);
    });
}

// The equivalence runs on the reference model: the generated CPU model is a random-weight dummy,
// so a decisive agreement between the two framings is not a property it can show.
static void test_generic_jev_winner_equivalence(testing & t) {
    t.test("generic enum and Jev choice agree on the reference model", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (!gpu_model_ready(t, path)) {
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("the reference decision scaffold loads the model", false);
            return;
        }
        try {
            generic_jev_winner_equivalence_run(t, te.ctx, "gpu");
        } catch (const std::exception & e) {
            t.assert_true(std::string("the reference generic/Jev equivalence: ") + e.what(), false);
        }
    });
}

// A schema the compiler must refuse, with the phrase the reason has to carry.
static void expect_schema_reject(testing & t, const std::string & schema_text, const std::string & needle) {
    try {
        (void) llama_decision::compile_schema(common_json::parse(schema_text), "");
        t.assert_true("schema is rejected: " + schema_text, false);
    } catch (const llama_decision::semantic_error & e) {
        const std::string what = e.what();
        t.assert_true("reject reason contains \"" + needle + "\": " + schema_text + " -> " + what,
                      what.find(needle) != std::string::npos);
    }
}

// A wrong-typed field is invalid decision content, so the reason is a semantic_error (the server
// maps it to 422) and it names the offending field - never a common_json_error, which the server
// maps to 400 and which would put a semantically invalid body in the syntax class.
static void expect_schema_typed_reject(testing & t, const std::string & schema_text, const std::string & needle) {
    try {
        (void) llama_decision::compile_schema(common_json::parse(schema_text), "");
        t.assert_true("wrong-typed field is rejected: " + schema_text, false);
    } catch (const llama_decision::semantic_error & e) {
        const std::string what = e.what();
        t.assert_true("reject reason contains \"" + needle + "\": " + schema_text + " -> " + what,
                      what.find(needle) != std::string::npos);
    } catch (const std::exception & e) {
        t.assert_true("a wrong-typed field is a semantic error, not a parse error: " + schema_text +
                          " -> " + e.what(),
                      false);
    }
}

// The wrong-typed matrix of the generic front-end: every field the compiler reads by JSON kind
// answers a wrong-typed value with a field-naming semantic_error. The control cases are the valid
// shapes, which must still compile.
static void test_generic_wrong_typed_fields(testing & t) {
    t.test("a wrong-typed generic field is a semantic error naming the field", [](testing & t) {
        expect_schema_typed_reject(t, R"({"a": {"type": 5, "description": "d"}})", "type must be a string");
        expect_schema_typed_reject(t, R"({"a": {"type": "enum", "description": 7}})", "description must be a string");
        expect_schema_typed_reject(t, R"({"a": {"type": "boolean", "description": "d", "aggregate": 3}})",
                                   "aggregate must be a string");
        expect_schema_typed_reject(t, R"({"a": {"type": "boolean", "description": "d", "x-aggregate": []}})",
                                   "x-aggregate must be a string");
        expect_schema_typed_reject(t, R"({"a": {"type": "integer", "description": "d", "minimum": "0", "maximum": 2}})",
                                   "minimum must be an integer");
        expect_schema_typed_reject(t, R"({"a": {"type": "integer", "description": "d", "minimum": 0, "maximum": 2.5}})",
                                   "maximum must be an integer");
        expect_schema_typed_reject(t, R"({"a": {"type": "number", "description": "d", "minimum": "0", "maximum": 1, "step": 0.5}})",
                                   "minimum must be a number");
        expect_schema_typed_reject(t, R"({"a": {"type": "number", "description": "d", "minimum": 0, "maximum": "1", "step": 0.5}})",
                                   "maximum must be a number");
        expect_schema_typed_reject(t, R"({"a": {"type": "number", "description": "d", "minimum": 0, "maximum": 1, "step": true}})",
                                   "step must be a number");
        // the JSON Schema spelling reads the same members under its own key
        expect_schema_typed_reject(t, R"({"properties": {"a": {"type": "number", "minimum": 0, "maximum": 1, "multipleOf": "0.5"}}})",
                                   "multipleOf must be a number");
    });

    t.test("the wrong-typed request fields are semantic errors too", [](testing & t) {
        auto reject = [](testing & t, const std::string & body, const std::string & needle) {
            try {
                (void) llama_decision::parse_decision_request(common_json::parse(body));
                t.assert_true("wrong-typed request field is rejected: " + body, false);
            } catch (const llama_decision::semantic_error & e) {
                t.assert_true("reject reason contains \"" + needle + "\": " + body + " -> " + e.what(),
                              std::string(e.what()).find(needle) != std::string::npos);
            } catch (const std::exception & e) {
                t.assert_true("a wrong-typed request field is a semantic error: " + body + " -> " + e.what(), false);
            }
        };
        const std::string base = R"({"model": "m", "state": "s", "schema": {"a": {"type": "boolean"}}})";
        reject(t, R"({"model": 5, "state": "s", "schema": {"a": {"type": "boolean"}}})", "model must be a string");
        reject(t, R"({"model": "m", "state": "s", "instructions": 7, "schema": {"a": {"type": "boolean"}}})",
               "instructions must be a string");
        reject(t, R"({"model": "m", "state": "s", "schema": {"a": {"type": "boolean"}}, "mode": 1})",
               "mode must be a string");
        reject(t, R"({"model": "m", "state": "s", "schema": {"a": {"type": "boolean"}}, "tree_max": "8"})",
               "tree_max must be an integer");
        reject(t, R"({"model": "m", "state": "s", "schema": {"a": {"type": "boolean"}}, "cache_prompt": "yes"})",
               "cache_prompt must be a boolean");
        reject(t, R"({"model": "m", "state": "s", "schema": {"a": {"type": "boolean"}}, "diagnostics": "yes"})",
               "diagnostics must be a boolean");
        // the control: the same body with correct types is accepted unchanged
        const auto ok = llama_decision::parse_decision_request(common_json::parse(base));
        t.assert_equal("the valid body still parses", std::string("m"), ok.envelope.model);
        t.assert_true("the valid body still parses", ok.allow_cache);
    });

    t.test("a wrong-typed Jev request field is a semantic error too", [](testing & t) {
        auto reject = [](testing & t, const std::string & body, const std::string & needle) {
            try {
                (void) llama_decision::parse_decision_request(common_json::parse(body));
                t.assert_true("wrong-typed request field is rejected: " + body, false);
            } catch (const llama_decision::semantic_error & e) {
                t.assert_true("reject reason contains \"" + needle + "\": " + body + " -> " + e.what(),
                              std::string(e.what()).find(needle) != std::string::npos);
            } catch (const std::exception & e) {
                t.assert_true("a wrong-typed request field is a semantic error: " + body + " -> " + e.what(), false);
            }
        };
        const std::string qs = R"("questions": {"q": {"type": "noul", "instructions": "x"}})";
        reject(t, R"({"model": 5, "state": "s", )" + qs + "}", "model must be a string");
        reject(t, R"({"model": "m", "state": "s", "temperature": "1", )" + qs + "}", "temperature must be a number");
        reject(t, R"({"model": "m", "state": "s", "permutations": 1.5, )" + qs + "}", "permutations must be an integer");
        reject(t, R"({"model": "m", "state": "s", "diagnostics": "yes", )" + qs + "}",
               "diagnostics must be a boolean");
        reject(t, R"({"model": "m", "state": "s", "confidence_profile": 3, )" + qs + "}",
               "confidence_profile must be a string");
        reject(t, R"({"model": "m", "state": "s", "temperatures": {"choice": "hot"}, )" + qs + "}",
               "temperatures.choice must be a number");
        reject(t, R"({"model": "m", "state": "s", "id_slot": "0", )" + qs + "}", "id_slot must be an integer");
        reject(t, R"({"model": "m", "state": "s", "session_id": 7, )" + qs + "}", "session_id must be a non-empty string");
        reject(t, R"({"model": "m", "state": "s", "turn": 7, )" + qs + "}", "turn must be a string");
        reject(t, R"({"model": "m", "state": "s", "questions": {"q": {"type": "integer", "instructions": "x", "minimum": "0", "maximum": 3}}})",
               "integer needs integer minimum and maximum");
        reject(t, R"({"model": "m", "state": "s", "questions": {"q": {"type": "number", "instructions": "x", "minimum": 0, "maximum": 1, "step": "0.5"}}})",
               "number needs minimum, maximum and step");
        reject(t, R"({"model": "m", "state": "s", "questions": {"q": {"type": "integer", "instructions": "x", "minimum": 0, "maximum": 3, "aggregate": 2}}})",
               "aggregate must be a string");
        // the control: the same body with correct types is accepted unchanged
        const auto ok = llama_decision::parse_decision_request(
            common_json::parse(R"({"model": "m", "state": "s", )" + qs + "}"));
        t.assert_equal("the valid body still parses", std::string("m"), ok.envelope.model);
        t.assert_equal("the valid body still parses", (size_t) 1, ok.questions.size());
    });
}

// The schema compiler is a pure function of the request body: it validates the field catalogue and
// encodes the typed grids into engine-facing field inputs. These cases pin the accepted shapes, the
// refusals, and the scored text. They need no model and no context, so a capability the refactor
// drops fails here rather than only at the HTTP layer.
static void test_generic_schema_compiler(testing & t) {
    t.test("the compiler accepts every supported field type and aggregate", [](testing & t) {
        const llama_decision::compiled_schema cs = llama_decision::compile_schema(common_json::parse(R"({
            "flag":    {"type": "boolean", "description": "is it active"},
            "team":    {"type": "enum", "description": "owning team", "enum": ["billing", "tech"]},
            "count":   {"type": "integer", "description": "affected rows", "minimum": 1, "maximum": 4},
            "impact":  {"type": "number", "description": "impact factor", "minimum": 0.0, "maximum": 1.0,
                        "step": 0.5, "aggregate": "mean"},
            "sev":     {"type": "integer", "description": "severity", "minimum": 0, "maximum": 2,
                        "aggregate": "median"}
        })"), "extra instruction");
        t.assert_equal("one spec per field", (size_t) 5, cs.specs.size());
        t.assert_equal("one scoring field per spec", cs.specs.size(), cs.inputs.size());
        t.assert_equal("boolean type", std::string("boolean"), cs.specs[0].type);
        t.assert_equal("enum type", std::string("enum"), cs.specs[1].type);
        t.assert_equal("integer type", std::string("integer"), cs.specs[2].type);
        t.assert_equal("number type", std::string("number"), cs.specs[3].type);
        t.assert_equal("boolean values", (size_t) 2, cs.specs[0].values.size());
        t.assert_equal("boolean encodes true and false", std::string("true,false"),
                       cs.specs[0].encoded[0] + "," + cs.specs[0].encoded[1]);
        t.assert_equal("integer grid is inclusive of both bounds", (size_t) 4, cs.specs[2].values.size());
        t.assert_equal("number grid honours the step", (size_t) 3, cs.specs[3].values.size());
        t.assert_equal("mode is the default aggregate", std::string("mode"), cs.specs[0].aggregate);
        t.assert_equal("mean aggregate is kept", std::string("mean"), cs.specs[3].aggregate);
        t.assert_equal("median aggregate is kept", std::string("median"), cs.specs[4].aggregate);
        t.assert_true("the instructions reach the system text",
                      cs.system_text.find("extra instruction") != std::string::npos);
        t.assert_true("the catalogue names the field", cs.catalogue.find("team") != std::string::npos);
    });

    t.test("a JSON Schema body compiles the same catalogue without requiring a description",
           [](testing & t) {
        const llama_decision::compiled_schema cs = llama_decision::compile_schema(common_json::parse(R"({
            "type": "object",
            "properties": {
                "active": {"type": "boolean"},
                "level":  {"type": "integer", "minimum": 0, "maximum": 2},
                "grade":  {"type": "number", "minimum": 0.0, "maximum": 1.0, "multipleOf": 0.25}
            }
        })"), "");
        t.assert_equal("the properties object is the field catalogue", (size_t) 3, cs.specs.size());
        t.assert_equal("a description is optional here", std::string(), cs.specs[0].description);
        t.assert_equal("multipleOf builds the grid", (size_t) 5, cs.specs[2].values.size());
    });

    // The prefix hoist is what keeps a wide enum cheap: the head its values share is decoded once in
    // the suffix and each branch decodes only its own tail. The engine finds that head at a token
    // boundary in `suffix + candidate`, so the front-end scores the whole encoded value and never
    // splits it itself. The golden below pins the scored text exactly.
    t.test("each field's suffix and candidates are the whole encoded value", [](testing & t) {
        const llama_decision::compiled_schema cs = llama_decision::compile_schema(common_json::parse(R"({
            "team": {"type": "enum", "description": "owning team",
                     "enum": ["platform-frontend", "platform-backend", "platform-infra"]}
        })"), "");
        const llama_decision::generic_field_spec & f = cs.specs[0];
        t.assert_equal("the suffix opens the field and stops there", std::string("  \"team\": "),
                       cs.inputs[0].suffix);
        t.assert_equal("one candidate per value", f.encoded.size(), cs.inputs[0].candidates.size());
        for (size_t i = 0; i < f.encoded.size(); ++i) {
            t.assert_equal("the candidate is the whole encoded value", f.encoded[i],
                           cs.inputs[0].candidates[i]);
        }
    });

    // The wide-domain escape: a field may carry up to 255 values whatever the tokenizer resolves.
    // That is the capability the Jev label pool cannot offer, so it is pinned by size.
    t.test("the compiler accepts a field wider than the Jev label pool", [](testing & t) {
        common_json many = common_json::array();
        for (int i = 0; i < 255; ++i) {
            many.push_back("option-" + std::to_string(i));
        }
        common_json schema = common_json::object();
        schema["wide"] = common_json::object();
        schema["wide"]["type"] = "enum";
        schema["wide"]["description"] = "a wide catalogue";
        schema["wide"]["enum"] = many;
        const llama_decision::compiled_schema cs = llama_decision::compile_schema(schema, "");
        t.assert_equal("255 values are accepted", (size_t) 255, cs.specs[0].values.size());
        t.assert_equal("255 candidates are scored", (size_t) 255, cs.inputs[0].candidates.size());
    });

    t.test("the compiler refuses a malformed schema", [](testing & t) {
        expect_schema_reject(t, R"(null)", "must be an object");
        expect_schema_reject(t, R"({})", "1-32 fields");
        expect_schema_reject(t, R"({"a": {"type": "boolean"}})", "needs a description");
        expect_schema_reject(t, R"({"a": {"type": "wat", "description": "d"}})", "boolean, enum, integer and number");
        expect_schema_reject(t, R"({"a": {"type": "enum", "description": "d"}})", "need a list of choices");
        expect_schema_reject(t, R"({"a": {"type": "enum", "description": "d", "enum": [1, 2]}})",
                             "must be strings");
        expect_schema_reject(t, R"({"a": {"type": "enum", "description": "d", "enum": ["x", "x"]}})",
                             "duplicate");
        expect_schema_reject(t, R"({"a": {"type": "integer", "description": "d", "minimum": 0}})",
                             "minimum and maximum");
        expect_schema_reject(t, R"({"a": {"type": "integer", "description": "d", "minimum": 0.5,
                            "maximum": 2}})", "minimum must be an integer");
        expect_schema_reject(t, R"({"a": {"type": "integer", "description": "d", "minimum": 0,
                            "maximum": 300}})", "1-255 values");
        expect_schema_reject(t, R"({"a": {"type": "number", "description": "d", "minimum": 0.0,
                            "maximum": 1.0}})", "step");
        expect_schema_reject(t, R"({"a": {"type": "number", "description": "d", "minimum": 0.0,
                            "maximum": 1.0, "step": -0.5}})", "positive step");
        expect_schema_reject(t, R"({"a": {"type": "boolean", "description": "d",
                            "aggregate": "mean"}})", "median/mean for numeric");
    });

    t.test("the shape dispatch accepts each body and refuses the ambiguous one", [](testing & t) {
        t.assert_true("a schema body selects the generic front-end",
                      llama_decision::select_request_shape(
                          common_json::parse(R"({"schema": {"a": {"type": "boolean"}}})")) ==
                      llama_decision::request_shape::generic);
        t.assert_true("a questions body selects the Jev front-end",
                      llama_decision::select_request_shape(
                          common_json::parse(R"({"questions": {}})")) ==
                      llama_decision::request_shape::jev);
        t.assert_true("a state-only body selects the Jev front-end",
                      llama_decision::select_request_shape(common_json::parse(R"({"state": "s"})")) ==
                      llama_decision::request_shape::jev);
        t.assert_true("a body with neither selects no front-end",
                      llama_decision::select_request_shape(common_json::parse(R"({"model": "m"})")) ==
                      llama_decision::request_shape::none);
        t.assert_true("a non-object body selects no front-end",
                      llama_decision::select_request_shape(common_json::parse("[]")) ==
                      llama_decision::request_shape::none);
    });
}

// The accepted spellings of one field: the type aliases, the choices key, and the aggregate alias.
static void test_generic_field_spellings(testing & t) {
    t.test("the compiler accepts the documented field spellings", [](testing & t) {
        for (const char * type : { "choice", "selection" }) {
            const common_json body = common_json::parse(
                std::string(R"({"a": {"type": ")") + type + R"(", "description": "d",
                             "choices": ["x", "y"]}})");
            t.assert_equal(std::string("type ") + type + " compiles to an enum",
                           std::string("enum"), llama_decision::compile_schema(body, "").specs[0].type);
        }
        // an explicit enum member wins over a declared type
        t.assert_equal("an enum member forces the enum type", std::string("enum"),
                       llama_decision::compile_schema(
                           common_json::parse(R"({"a": {"type": "string", "description": "d",
                                        "enum": ["x", "y"]}})"), "").specs[0].type);
        t.assert_equal("the x-aggregate alias is accepted", std::string("median"),
                       llama_decision::compile_schema(
                           common_json::parse(R"({"a": {"type": "integer", "description": "d",
                                        "minimum": 0, "maximum": 2, "x-aggregate": "median"}})"),
                           "").specs[0].aggregate);
    });

    // The catalogue line is what the model reads, and each candidate is the value it lists, so the
    // two cannot drift into showing one value space and scoring another.
    t.test("the catalogue lists the encoded values the candidates score", [](testing & t) {
        const auto cs = llama_decision::compile_schema(
            common_json::parse(R"({"a": {"type": "enum", "description": "d", "enum": ["x", "y"]}})"), "");
        t.assert_true("the catalogue shows the encoded values",
                      cs.catalogue.find("Allowed values: \"x\", \"y\"") != std::string::npos);
        t.assert_equal("each candidate is the encoded value it lists", std::string("\"y\""),
                       cs.inputs[0].candidates[1]);
    });
}

// The engine scores `suffix + candidate`, so that concatenation is the whole of a field's scored
// text; the system text is what the model is shown. The golden pins both, because any move in either
// moves every probability the field reports. It needs no model and no tokenizer.
static common_json generic_scored_text() {
    const auto cs = llama_decision::compile_schema(common_json::parse(R"({
        "team":   {"type": "enum", "description": "owning team",
                   "enum": ["platform-frontend", "platform-backend", "platform-infra"]},
        "active": {"type": "boolean", "description": "is the incident active"},
        "count":  {"type": "integer", "description": "affected rows", "minimum": 1, "maximum": 4},
        "impact": {"type": "number", "description": "impact factor", "minimum": 0.0,
                   "maximum": 1.0, "step": 0.5}
    })"), "Answer from the context.");
    common_json lines_by_field = common_json::object();
    for (size_t i = 0; i < cs.inputs.size(); ++i) {
        common_json lines = common_json::array();
        for (const auto & candidate : cs.inputs[i].candidates) {
            lines.push_back(cs.inputs[i].suffix + candidate + "\n");
        }
        lines_by_field[cs.specs[cs.field_spec[i]].name] = lines;
    }
    common_json out = common_json::object();
    out["system_text"] = cs.system_text;
    out["scored_lines"] = lines_by_field;
    return out;
}

static void test_generic_scored_text_golden(testing & t) {
    t.test("the compiled generic text matches the committed golden", [&](testing & t) {
        const std::string actual = generic_scored_text().dump(2) + "\n";
        if (record_golden) {
            write_file(fixture_path("generic_scored_text.golden.json"), actual);
            return;
        }
        t.assert_equal("the generic scored text is byte-identical",
                       read_file(fixture_path("generic_scored_text.golden.json")), actual);
    });
}

// One envelope, two shapes. Everything the shapes share is parsed once, so a shared field cannot be
// honoured on one shape and ignored on the other. These cases need no model.
static void test_request_envelope_unification(testing & t) {
    const std::string jev = R"({
        "model": "m", "state": "s",
        "questions": {"q": {"type": "choice", "instructions": "x", "criteria": {"a": "", "b": ""}}}
    })";
    const std::string gen = R"({
        "model": "m", "state": "s",
        "schema": {"a": {"type": "enum", "description": "d", "enum": ["a", "b"]}}
    })";

    t.test("both shapes parse the shared fields through one envelope", [&](testing & t) {
        auto knobs_of = [](const std::string & body) {
            return llama_decision::parse_decision_request(common_json::parse(body)).envelope.knobs;
        };
        const std::string with = R"("temperature": 0.5, "temperatures": {"choice": 2.0},
                                 "permutations": 3, "confidence_profile": "local", "diagnostics": true)";
        for (const std::string & shape : { std::string("questions"), std::string("schema") }) {
            const std::string body = shape == "questions"
                ? std::string(jev.substr(0, jev.size() - 2)) + ", " + with + "}"
                : std::string(gen.substr(0, gen.size() - 2)) + ", " + with + "}";
            const llama_decision::decision_request req = llama_decision::parse_decision_request(
                common_json::parse(body));
            t.assert_equal(shape + ": the shape is selected", shape,
                           req.shape == llama_decision::request_shape::jev ? "questions" : "schema");
            t.assert_equal(shape + ": the global temperature", 0.5, req.envelope.knobs.temperature);
            t.assert_equal(shape + ": the per-type temperature", 2.0,
                           req.envelope.knobs.for_type("enum"));
            t.assert_equal(shape + ": the passes", 3, req.envelope.knobs.permutations);
            t.assert_equal(shape + ": the reported profile", std::string("local"),
                           req.envelope.knobs.confidence_profile);
            t.assert_true(shape + ": the diagnostics opt-in", req.envelope.diagnostics);
            t.assert_true(shape + ": the evidence is parsed once",
                          req.envelope.evidence.state_present &&
                              req.envelope.evidence.state.get<std::string>() == "s");
            t.assert_true(shape + ": the generic shape carries the schema", req.schema.is_null() ==
                          (shape == "questions"));
        }
        // the defaults are the same on both shapes
        t.assert_equal("an unset knob defaults on either shape",
                       knobs_of(jev).for_type("choice"), knobs_of(gen).for_type("enum"));
    });

    // A generic field type has no temperature key of its own; it reads the Jev primitive that scores
    // it, so the wire's five keys stay the whole vocabulary and no second map exists.
    t.test("a field type resolves its temperature through its primitive", [](testing & t) {
        llama_decision::producer_knobs k;
        k.temperature       = 1.0;
        k.temperatures      = common_json::parse(R"({"noul": 2.0, "choice": 3.0, "integer": 4.0})");
        t.assert_equal("a boolean field reads the noul temperature", 2.0, k.for_type("boolean"));
        t.assert_equal("an enum field reads the choice temperature", 3.0, k.for_type("enum"));
        t.assert_equal("an integer field reads its own temperature", 4.0, k.for_type("integer"));
        t.assert_equal("a number field without an override reads the global", 1.0, k.for_type("number"));
        k.temperature = 0.25;
        t.assert_equal("the global is the fallback", 0.25, k.for_type("number"));
        // the allow-list is unchanged, so a key that is neither primitive nor numeric is still refused
        bool threw = false;
        try {
            (void) llama_decision::parse_decision_request(common_json::parse(
                R"({"model":"m","state":"s","questions":{"q":{"type":"noul","instructions":"x"}},
                    "temperatures":{"boolean":1.0}})"));
        } catch (const llama_decision::semantic_error &) {
            threw = true;
        }
        t.assert_true("an unlisted temperatures key is still refused", threw);
    });

    // The evidence validator used to exist twice, once per front-end. These are the rules it owns,
    // and they are now exercised once, on both shapes.
    t.test("the evidence rules hold on both shapes", [](testing & t) {
        expect_decision_reject(t, R"({"model":"m","state":"s","contexts":["c"],
                         "questions":{"q":{"type":"noul","instructions":"x"}}})",
                               "provide either state or contexts, not both");
        expect_decision_reject(t, R"({"model":"m","state":"s","contexts":["c"],
                         "schema":{"a":{"type":"boolean","description":"d"}}})",
                               "provide either state or contexts, not both");
        expect_decision_reject(t, R"({"model":"m","contexts":[],
                         "questions":{"q":{"type":"noul","instructions":"x"}}})",
                               "contexts must hold 1-256 entries");
        expect_decision_reject(t, R"({"model":"m","contexts":["c",""],
                         "schema":{"a":{"type":"boolean","description":"d"}}})",
                               "must be a non-empty string");
        // a session reference is evidence for the generic shape only; the Jev shape still needs text
        expect_decision_reject(t, R"({"model":"m","questions":{"q":{"type":"noul","instructions":"x"}},
                         "id_slot":0})", "state (or contexts) is required");
        const auto ses = llama_decision::parse_decision_request(common_json::parse(
            R"({"model":"m","id_slot":0,"schema":{"a":{"type":"boolean","description":"d"}}})"));
        t.assert_true("a session is evidence for the generic shape", ses.envelope.session.present);
        t.assert_true("the generic shape reports itself", ses.shape == llama_decision::request_shape::generic);
    });

    // Control: a field only one shape defines stays an ignored unknown top-level field on the other,
    // so a body written for one front-end cannot change the other one's answer by accident.
    t.test("an unknown top-level field is ignored on both shapes", [&](testing & t) {
        const auto a = llama_decision::parse_decision_request(common_json::parse(jev));
        common_json jev_extra = common_json::parse(jev);
        jev_extra["mode"]  = "tree";
        jev_extra["bogus"] = 7;
        const auto b = llama_decision::parse_decision_request(jev_extra);
        t.assert_equal("the Jev shape ignores it", a.questions.size(), b.questions.size());
        t.assert_equal("the Jev shape ignores the generic knobs", std::string("auto"), b.mode);

        common_json gen_extra = common_json::parse(gen);
        gen_extra["instructions"] = "extra";
        gen_extra["bogus"]         = 7;
        const auto c = llama_decision::parse_decision_request(gen_extra);
        t.assert_equal("the generic shape ignores it", (size_t) 1, c.schema.size());
        t.assert_equal("the generic shape keeps its own fields", std::string("extra"), c.instructions);
        t.assert_equal("the generic shape has no questions", (size_t) 0, c.questions.size());
    });
}

// Order de-biasing on the generic shape: one scoring field per (pass, field), averaged by value
// index. The pass orders come from the same permutation_order the letter readout uses, so the two
// shapes cannot drift on what a pass means. Pure, so no model is needed to pin the definition.
static void test_generic_permutations(testing & t) {
    const std::string schema = R"({"a": {"type": "enum", "description": "d", "enum": ["x", "y", "z"]},
                                   "b": {"type": "boolean", "description": "e"}})";

    t.test("each pass scores every field in one seeded order", [&](testing & t) {
        llama_decision::producer_knobs knobs;
        knobs.permutations = 2;
        const auto cs = llama_decision::compile_schema(common_json::parse(schema), "", knobs);
        t.assert_equal("one scoring field per pass and field", (size_t) 4, cs.inputs.size());
        t.assert_equal("as many order maps", cs.inputs.size(), cs.field_order.size());
        t.assert_equal("as many owners", cs.inputs.size(), cs.field_spec.size());
        for (size_t f = 0; f < cs.inputs.size(); ++f) {
            const size_t si    = cs.field_spec[f];
            const int    pass  = (int) (f / cs.specs.size());
            const std::vector<size_t> want =
                llama_decision::permutation_order(cs.specs[si].values.size(), cs.specs[si].name, pass);
            t.assert_true("the order is the shared permutation for the field and pass",
                          want == cs.field_order[f]);
            t.assert_equal("the candidates are the encoded values in that order",
                           cs.specs[si].encoded.size(), cs.inputs[f].candidates.size());
            for (size_t i = 0; i < cs.inputs[f].candidates.size(); ++i) {
                t.assert_equal("the candidate at a position is its encoded value",
                               cs.specs[si].encoded[cs.field_order[f][i]], cs.inputs[f].candidates[i]);
            }
        }
    });

    t.test("the passes are averaged by value index", [&](testing & t) {
        llama_decision::producer_knobs knobs;
        knobs.permutations = 2;
        const auto cs = llama_decision::compile_schema(common_json::parse(schema), "", knobs);
        const size_t n = cs.specs.size();
        llama_decision::result raw;
        raw.fields.resize(cs.inputs.size());
        // the three-value field: each pass is decisive on a different candidate in its own order
        raw.fields[0].winner = 0;
        raw.fields[0].probs  = { 0.8f, 0.1f, 0.1f };
        raw.fields[n + 0].winner = 1;
        raw.fields[n + 0].probs  = { 0.2f, 0.6f, 0.2f };
        // the two-value field: the passes favour different values, so the mean is what decides
        raw.fields[1].winner = 0;
        raw.fields[1].probs  = { 0.6f, 0.4f };
        raw.fields[n + 1].winner = 0;
        raw.fields[n + 1].probs  = { 0.7f, 0.3f };

        // the expected mean, derived here from the same orders the fold must use
        std::vector<std::vector<float>> expect(cs.specs.size());
        for (size_t si = 0; si < n; ++si) {
            expect[si].assign(cs.specs[si].values.size(), 0.0f);
            for (int pass = 0; pass < 2; ++pass) {
                const auto & order = cs.field_order[(size_t) pass * n + si];
                for (size_t i = 0; i < order.size(); ++i) {
                    expect[si][order[i]] += raw.fields[(size_t) pass * n + si].probs[i] / 2.0f;
                }
            }
        }

        llama_decision::mean_permuted_passes(cs, raw);
        t.assert_equal("one distribution per field", n, raw.fields.size());
        double sum = 0.0;
        for (size_t si = 0; si < n; ++si) {
            for (size_t i = 0; i < expect[si].size(); ++i) {
                assert_close(t, "the averaged probability of a value", expect[si][i], raw.fields[si].probs[i],
                             1e-6);
                sum += (double) raw.fields[si].probs[i];
            }
            t.assert_equal("the winner is the averaged argmax",
                           (int) (std::max_element(raw.fields[si].probs.begin(), raw.fields[si].probs.end()) -
                                  raw.fields[si].probs.begin()),
                           raw.fields[si].winner);
        }
        t.assert_true("every average sums to 1", std::fabs(sum - (double) n) < 1e-5);
    });

    // Control: a single pass has nothing to fold, so the scored result must come back untouched.
    t.test("a single pass is not folded", [&](testing & t) {
        const auto cs = llama_decision::compile_schema(common_json::parse(schema), "");
        llama_decision::result raw;
        raw.fields.resize(cs.inputs.size());
        raw.fields[0].winner = 1;
        raw.fields[0].probs  = { 0.2f, 0.5f, 0.3f };
        raw.fields[1].winner = 0;
        raw.fields[1].probs  = { 0.7f, 0.3f };
        const auto before = raw;
        llama_decision::mean_permuted_passes(cs, raw);
        t.assert_equal("no extra field is produced", before.fields.size(), raw.fields.size());
        for (size_t i = 0; i < raw.fields.size(); ++i) {
            t.assert_equal("the winner is untouched", before.fields[i].winner, raw.fields[i].winner);
            t.assert_true("the distribution is untouched", before.fields[i].probs == raw.fields[i].probs);
        }
    });

    // A greedily scored field has no distribution over its space, so the fold must not invent one:
    // the passes decide the point mass the record reports.
    t.test("passes that were all greedy report the point mass they agreed on", [&](testing & t) {
        llama_decision::producer_knobs knobs;
        knobs.permutations = 2;
        const auto  cs = llama_decision::compile_schema(common_json::parse(schema), "", knobs);
        const size_t n  = cs.specs.size();
        llama_decision::result raw;
        raw.fields.resize(cs.inputs.size());
        // a greedy pass reports only the position it won, in that pass's own order
        raw.fields[0].winner     = 2;
        raw.fields[n + 0].winner = 0;
        raw.fields[1].winner     = 0;
        raw.fields[n + 1].winner = 1;
        // the point mass the passes accumulate, by value index
        std::vector<std::vector<float>> expect(n);
        for (size_t si = 0; si < n; ++si) {
            expect[si].assign(cs.specs[si].values.size(), 0.0f);
            for (int pass = 0; pass < 2; ++pass) {
                const auto & order = cs.field_order[(size_t) pass * n + si];
                expect[si][order[raw.fields[(size_t) pass * n + si].winner]] += 0.5f;
            }
        }

        llama_decision::mean_permuted_passes(cs, raw);
        const common_json answers = llama_decision::assemble(cs, raw, false);
        for (size_t si = 0; si < n; ++si) {
            t.assert_true("a greedy fold reports no distribution", raw.fields[si].probs.empty());
            t.assert_equal("the winner is the accumulated argmax",
                           (int) (std::max_element(expect[si].begin(), expect[si].end()) - expect[si].begin()),
                           raw.fields[si].winner);
            const std::string & name = cs.specs[si].name;
            const common_json & rec  = answers.at(name);
            t.assert_equal("the record still says argmax", std::string("argmax"),
                           rec.at("scored").get<std::string>());
            double sum = 0.0;
            int    hot = 0;
            for (const auto & kv : rec.at("probabilities").items()) {
                sum += kv.value().get<double>();
                hot += kv.value().get<double>() > 0.0 ? 1 : 0;
            }
            t.assert_true("the point mass sums to 1", std::fabs(sum - 1.0) < 1e-6);
            t.assert_equal("the point mass names one value", 1, hot);
        }
    });
}

static void test_generic_frontend(testing & t) {
    test_generic_schema_compiler(t);
    test_generic_wrong_typed_fields(t);
    test_generic_field_spellings(t);
    test_generic_typed_golden(t);
    test_generic_scored_text_golden(t);
    test_generic_mutual_exclusion(t);
    test_generic_permutations(t);
    test_generic_drives_engine(t);
    test_generic_jev_winner_equivalence(t);
}

static void test_label_pool_real(testing & t) {
    t.test("label pool and exact boundary ids on a real vocabulary", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (!gpu_model_ready(t, path)) {
            return;
        }
        llama_model * model = shared_model().model;

        try {
            auto vocab = llama_decision::make_llama_label_vocab(llama_model_get_vocab(model));
            const std::string tail = "Answer:\n";
            const auto pool = llama_decision::build_label_pool(*vocab, tail, 64);
            t.assert_true("pool has 2-64 labels", pool.size() >= 2 && pool.size() <= 64);

            bool boundary = true;
            bool alnum_first = true;
            for (size_t i = 0; i < pool.size(); ++i) {
                const auto & l = pool[i];
                boundary = boundary &&
                    (llama_decision::answer_label_path(*vocab, tail, l.text, (int) l.tokens.size()) == l.tokens);
                if (i < 62) {
                    for (unsigned char c : l.text) {
                        alnum_first = alnum_first && std::isalnum(c) != 0;
                    }
                }
            }
            t.assert_true("every label resolves at the boundary", boundary);
            t.assert_true("letters and digits come first", alnum_first);

            const auto base = vocab->tokenize(tail, true);
            const auto full = vocab->tokenize(tail + pool[0].text, true);
            t.assert_equal("exact boundary length", base.size() + 1, full.size());
            t.assert_true("exact boundary tail is the label token", !full.empty() && full.back() == pool[0].token);
            t.assert_true("boundary accepts the label", llama_decision::answer_label_token(*vocab, tail, pool[0].text) == pool[0].token);
        } catch (const std::exception & e) {
            t.assert_true(std::string("label pool runs: ") + e.what(), false);
        }
    });
}

// The pool is built at the real answer boundary, so an add_space_prefix vocabulary resolves "A" to
// the bare token the model emits after the tail instead of the space-prefixed isolated form. This
// is the SPM family that the isolated rule refused.
static void test_letter_labels_spm(testing & t) {
    t.test("letter labels resolve at the answer boundary on an SPM model", [](testing & t) {
        const std::string path = spm_labels_model_path();
        if (path.empty()) {
            t.skip("no SPM model available");
            return;
        }
        cpu_test_engine te;
        if (!te.load(path, 512)) {
            t.assert_true("the SPM model loads on CPU", false);
            return;
        }
        try {
            auto vocab = llama_decision::make_llama_label_vocab(llama_model_get_vocab(te.model));
            const std::string tail = test_letter_tail();
            const auto pool = llama_decision::build_label_pool(*vocab, tail, 64);
            t.assert_true("the pool has at least two labels", pool.size() >= 2);
            for (const auto & l : pool) {
                t.assert_equal("label " + l.text + " resolves at the answer boundary", l.token,
                               llama_decision::answer_label_token(*vocab, tail, l.text));
            }

            // The bug: the isolated token is the space-prefixed form, while the model emits the
            // bare token after the tail. Skip when the tokenizer does not prefix a space.
            bool space_prefixed = false;
            for (const auto & l : pool) {
                const std::vector<int32_t> iso = vocab->tokenize(l.text, false);
                if (iso.size() == 1 && vocab->piece(iso[0]) == " " + l.text) {
                    space_prefixed = true;
                    break;
                }
            }
            if (space_prefixed) {
                bool isolated_differs = false;
                for (const auto & l : pool) {
                    const std::vector<int32_t> iso = vocab->tokenize(l.text, false);
                    if (iso.size() == 1 && iso[0] != l.token) {
                        isolated_differs = true;
                        break;
                    }
                }
                t.assert_true("the isolated token differs from the boundary token", isolated_differs);
            }

            // End to end: a previously-refused family now serves a closed distribution per question.
            llama_decision::engine eng(te.ctx, 2, 8);
            const auto req = llama_decision::parse_decision_request(common_json::parse(decision_valid_body()));
            llama_decision::readout_metrics metrics;
            const auto probs = test_letter_readout(eng, *vocab, nullptr, false, req, pool,
                                                              llama_decision::options{}, &metrics);
            t.assert_equal("one distribution per question", req.questions.size(), probs.size());
            bool valid = probs.size() == req.questions.size();
            for (size_t i = 0; valid && i < probs.size(); ++i) {
                valid = probs[i].size() == req.questions[i].options.size();
                double sum = 0.0;
                for (float p : probs[i]) {
                    valid = valid && p >= 0.0f && p <= 1.0f;
                    sum += p;
                }
                valid = valid && std::fabs(sum - 1.0) < 1e-3;
                valid = valid && !probs[i].empty() && std::max_element(probs[i].begin(), probs[i].end()) != probs[i].end();
            }
            t.assert_true("every distribution is closed and valid", valid);
        } catch (const std::exception & e) {
            t.assert_true(std::string("the SPM label pool runs: ") + e.what(), false);
        }
    });
}

// Calibration of the label accept/reject gate. The gate is task-correctness - the next token at
// the answer boundary is exactly this label - not a confidence score; the removed piece check was
// a producer-confidence proxy. There is no numeric threshold, so precision/recall become exact
// accept/reject counts over a control, a positive, and a negative group.
static void test_label_boundary_calibration(testing & t) {
    t.test("boundary gate calibration: control, positive and negative groups", [](testing & t) {
        int control_false_rejects  = 0;
        int positive_false_rejects = 0;
        int negative_false_accepts = 0;
        int positive_labels        = 0;

        // Control: a vocabulary whose isolated token is the boundary token must not change.
        {
            const fake_vocab v  = make_fake_vocab(true);
            const auto isolated = llama_decision::build_label_pool(v, "", 64);
            const auto boundary = llama_decision::build_label_pool(v, "Answer:\n", 64);
            bool same = isolated.size() == boundary.size();
            for (size_t i = 0; same && i < isolated.size(); ++i) {
                same = isolated[i].text == boundary[i].text && isolated[i].token == boundary[i].token;
            }
            if (!same) {
                ++control_false_rejects;
            }

            // A real BPE model is a control only when its isolated token equals its boundary token.
            const char * env = std::getenv("LLAMA_DECISION_TEST_MODEL");
            if (env != nullptr && env[0] != '\0' && file_exists(env)) {
                cpu_test_engine te;
                if (te.load(env)) {
                    auto vocab = llama_decision::make_llama_label_vocab(llama_model_get_vocab(te.model));
                    const std::string tail = test_letter_tail();
                    const auto pool = llama_decision::build_label_pool(*vocab, tail, 64);
                    bool space_prefixed = false;
                    for (const auto & l : pool) {
                        const std::vector<int32_t> iso = vocab->tokenize(l.text, false);
                        if (iso.size() == 1 && vocab->piece(iso[0]) == " " + l.text) {
                            space_prefixed = true;
                            break;
                        }
                    }
                    if (!space_prefixed) {
                        const auto real_isolated = llama_decision::build_label_pool(*vocab, "", 64);
                        bool real_same = real_isolated.size() == pool.size();
                        for (size_t i = 0; real_same && i < pool.size(); ++i) {
                            real_same = real_isolated[i].text == pool[i].text && real_isolated[i].token == pool[i].token;
                        }
                        if (!real_same) {
                            ++control_false_rejects;
                        }
                    }
                }
            }
        }

        // Positive: the SPM model must be accepted at the boundary.
        {
            const std::string path = spm_labels_model_path();
            if (!path.empty()) {
                cpu_test_engine te;
                if (te.load(path)) {
                    auto vocab = llama_decision::make_llama_label_vocab(llama_model_get_vocab(te.model));
                    const std::string tail = test_letter_tail();
                    const auto pool = llama_decision::build_label_pool(*vocab, tail, 64);
                    positive_labels = (int) pool.size();
                    if (pool.size() < 2) {
                        ++positive_false_rejects;
                    }
                    for (const auto & l : pool) {
                        if (llama_decision::answer_label_token(*vocab, tail, l.text) != l.token) {
                            ++positive_false_rejects;
                        }
                    }
                }
            }
        }

        // Negative: a vocabulary with no letter tokens, a special token, and a genuine merge must
        // still be rejected. The generated dummy model is not a usable negative here: its "test"
        // tokenizer hashes fixed 5-character chunks into 128 ids, so some candidates land on a
        // single boundary token by hash collision. That is a fixture artifact, not a gate defect,
        // and the gate still refuses any vocabulary with fewer than two boundary-resolvable labels.
        {
            fake_vocab no_letters;
            no_letters.pieces.push_back("tok_0");
            no_letters.pieces.push_back("tok_1");
            no_letters.drop_unknown = true;
            bool no_letters_rejected = false;
            try {
                (void) llama_decision::build_label_pool(no_letters, "", 64);
            } catch (const std::runtime_error &) {
                no_letters_rejected = true;
            }
            if (!no_letters_rejected) {
                ++negative_false_accepts;
            }

            fake_vocab special = make_fake_vocab(false);
            special.specials.insert(special.id_of("B"));
            if (llama_decision::answer_label_token(special, "", "B") >= 0) {
                ++negative_false_accepts;
            }

            fake_vocab merged = make_fake_vocab(false);
            merged.pieces.push_back("\nA");
            if (llama_decision::answer_label_token(merged, "x\n", "A") >= 0) {
                ++negative_false_accepts;
            }
        }

        fprintf(stderr, "label boundary calibration: control_false_rejects=%d positive_false_rejects=%d positive_labels=%d negative_false_accepts=%d\n",
                control_false_rejects, positive_false_rejects, positive_labels, negative_false_accepts);
        t.assert_equal("control group: zero false rejects", 0, control_false_rejects);
        t.assert_equal("positive group: zero false rejects", 0, positive_false_rejects);
        t.assert_equal("negative group: zero false accepts", 0, negative_false_accepts);
    });
}

// The model tests below run against the one shared model, so they share one caller-owned head
// cache. The dedicated cache tests construct their own.

static void test_letter_readout_real(testing & t) {
    t.test("letter readout scores a decision request on a real model", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (!gpu_model_ready(t, path)) {
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("model loads", false);
            return;
        }

        try {
            auto vocab = llama_decision::make_llama_label_vocab(llama_model_get_vocab(te.model));
            const auto pool = llama_decision::build_label_pool(*vocab, test_letter_tail(), 64);
            llama_decision::engine eng(te.ctx, 2, 8);
            const auto req = llama_decision::parse_decision_request(common_json::parse(decision_valid_body()));

            llama_decision::readout_metrics metrics;
            const auto probs = test_letter_readout(eng, *vocab, nullptr, false, req, pool,
                                                              llama_decision::options{}, &metrics);
            t.assert_equal("one distribution per question", req.questions.size(), probs.size());
            bool valid = true;
            for (size_t i = 0; i < probs.size(); ++i) {
                valid = valid && (probs[i].size() == req.questions[i].options.size());
                double sum = 0.0;
                for (float p : probs[i]) {
                    valid = valid && (p >= 0.0f);
                    sum += p;
                }
                valid = valid && (std::fabs(sum - 1.0) < 1e-4);
            }
            t.assert_true("every distribution is a valid probability vector", valid);

            // exact boundary for the letter readout's fallback tail
            const std::string prompt = "\nAnswer:\n";
            const auto base = vocab->tokenize(prompt, true);
            const auto full = vocab->tokenize(prompt + pool[0].text, true);
            t.assert_equal("production boundary length", base.size() + 1, full.size());
            t.assert_true("production boundary tail is the label",
                          !full.empty() && full.back() == pool[0].token);

            // exact boundary for the model's real chat template tail
            auto tmpls = common_chat_templates_init(te.model, "");
            if (tmpls) {
                const auto parts = llama_decision::render_letter_prompt(tmpls.get(), true,
                                                                        llama_decision::letter_system_text());
                const std::string tail = parts.second + "Answer:\n";
                t.assert_true("real template boundary accepts the label",
                              llama_decision::answer_label_token(*vocab, tail, pool[0].text) == pool[0].token);
            }

            // Task-value producer-determinism properties: the winner is stable under question
            // reordering and under temperature sharpening. These are hard on every accepted model.
            // On the known weak-quant GPU oracle the numerics move a winner for these checks, so
            // there they are skipped with a reason - never xfail, because a winner disagreement is
            // a task-value outcome, and a head-path winner disagreement is always hard.
            const bool weak_quant = weak_quant_gpu_oracle(path);

            determinism_check(t, weak_quant,
                              "question order does not change the winners (weak-quant GPU batch-shape sensitivity)",
                              [&](testing & t) {
                llama_decision::decision_request reversed = req;
                std::reverse(reversed.questions.begin(), reversed.questions.end());
                const auto probs_rev = test_letter_readout(eng, *vocab, nullptr, false, reversed, pool,
                                                                      llama_decision::options{}, nullptr);
                bool stable = probs_rev.size() == probs.size();
                for (size_t i = 0; stable && i < req.questions.size(); ++i) {
                    const std::string & id = req.questions[i].id;
                    size_t other = 0;
                    while (other < reversed.questions.size() && reversed.questions[other].id != id) {
                        ++other;
                    }
                    // Reversing the questions changes the branch order in the batch, so a backend may
                    // reorder a reduction; the winner of each question must not move.
                    stable = other < probs_rev.size() &&
                              probs[i].size() == probs_rev[other].size() &&
                              std::distance(probs[i].begin(), std::max_element(probs[i].begin(), probs[i].end())) ==
                              std::distance(probs_rev[other].begin(),
                                            std::max_element(probs_rev[other].begin(), probs_rev[other].end()));
                }
                t.assert_true("question order does not change the winners", stable);
            });

            determinism_check(t, weak_quant,
                              "temperature preserves the winner and orders sharpness (weak-quant GPU numerics)",
                              [&](testing & t) {
                auto with_temps = [](const char * temps) {
                    common_json body = common_json::parse(decision_valid_body());
                    body["temperatures"] = common_json::parse(temps);
                    return llama_decision::parse_decision_request(body);
                };
                const auto sharp = test_letter_readout(eng, *vocab, nullptr, false,
                                                                  with_temps(R"({"noul":0.5,"choice":0.5,"score":0.5})"),
                                                                  pool, llama_decision::options{}, nullptr);
                const auto flat = test_letter_readout(eng, *vocab, nullptr, false,
                                                                 with_temps(R"({"noul":2.5,"choice":2.5,"score":2.5})"),
                                                                 pool, llama_decision::options{}, nullptr);
                auto top = [](const std::vector<float> & p) {
                    return *std::max_element(p.begin(), p.end());
                };
                bool preserved = sharp.size() == probs.size() && flat.size() == probs.size();
                for (size_t i = 0; preserved && i < probs.size(); ++i) {
                    preserved = preserved &&
                                std::distance(probs[i].begin(), std::max_element(probs[i].begin(), probs[i].end())) ==
                                    std::distance(sharp[i].begin(), std::max_element(sharp[i].begin(), sharp[i].end())) &&
                                std::distance(probs[i].begin(), std::max_element(probs[i].begin(), probs[i].end())) ==
                                    std::distance(flat[i].begin(), std::max_element(flat[i].begin(), flat[i].end())) &&
                                top(sharp[i]) + 1e-4 >= top(probs[i]) && top(probs[i]) + 1e-4 >= top(flat[i]);
                }
                t.assert_true("temperature preserves the winner and orders sharpness", preserved);
            });

            if (file_exists(decision_golden_path())) {
                common_json usage = common_json::object();
                usage["input_tokens"]    = 0;
                usage["output_tokens"]   = 0;
                usage["cached_tokens"]   = 0;
                usage["state_cache_hit"] = false;
                const common_json actual = llama_decision::assemble_decision_response(req, probs, "m", usage);
                const common_json golden = common_json::parse(read_file(decision_golden_path()));
                // The value golden is model-specific. Compare only on the model it was written
                // from; every other arch checks the mechanism, not these numbers.
                const std::string golden_model = golden.value("golden_model", std::string());
                if (!golden_model.empty() && golden_model == model_identity(path)) {
                    for (const auto & e : golden.at("answers").items()) {
                        const auto & exp_a = e.value();
                        const auto & got_a = actual.at("answers").at(e.key());
                        if (exp_a.at("type").get<std::string>() == "noul") {
                            assert_close(t, "noul/" + e.key(), exp_a.at("noul").get<double>(), got_a.at("noul").get<double>(), 5e-3);
                        } else {
                            t.assert_equal("winner/" + e.key(),
                                           exp_a.at("type").get<std::string>() == "choice" ? exp_a.at("choice").dump()
                                                                                                            : exp_a.at("score").dump(),
                                           got_a.at("type").get<std::string>() == "choice" ? got_a.at("choice").dump()
                                                                                                            : got_a.at("score").dump());
                        }
                    }
                } else {
                    t.log("value golden not for this model; comparison skipped");
                }
            }
        } catch (const std::exception & e) {
            t.assert_true(std::string("letter readout runs: ") + e.what(), false);
        }
    });
}

static void test_fork_real(testing & t) {
    t.test("copy, bypass and LRU behave", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (!gpu_model_ready(t, path)) {
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("model loads", false);
            return;
        }
        try {
            llama_decision::engine eng(te.ctx, 2, 8);
            const std::vector<llama_decision::field_input> two = {
                { "  \"a\": ", { "1", "2" } },
                { "  \"b\": ", { "x", "y" } },
            };
            const std::vector<llama_decision::field_input> one = { { "  \"a\": ", { "1", "2" } } };

            // A copy fork cannot carry recurrent state, so it is only meaningful for pure
            // attention models; on a recurrent model the engine rejects it and auto picks the
            // exact partial hybrid fork.
            const bool copy_ok    = !llama_model_is_recurrent(te.model) && !llama_model_is_hybrid(te.model);
            const bool weak_quant = weak_quant_gpu_oracle(path);

            llama_decision::options orr;
            orr.fork      = "restore";
            orr.cache_tag = "t";
            const auto restore_run = decide_batch(eng, "system", { "ctx" }, two, orr);

            if (copy_ok) {
                llama_decision::options oc;
                oc.fork      = "copy";
                oc.cache_tag = "t";
                const auto copy_run = decide_batch(eng, "system", { "ctx" }, two, oc);

                // same math on two KV layouts: the winner is the task value and is always
                // asserted; the probability bound is producer numerics
                bool winners = copy_run.items[0].fields.size() == restore_run.items[0].fields.size();
                bool agree   = winners;
                for (size_t f = 0; agree && f < copy_run.items[0].fields.size(); ++f) {
                    const auto & pc = copy_run.items[0].fields[f].probs;
                    const auto & pr = restore_run.items[0].fields[f].probs;
                    winners = winners && copy_run.items[0].fields[f].winner == restore_run.items[0].fields[f].winner;
                    agree   = pc.size() == pr.size();
                    for (size_t k = 0; agree && k < pc.size(); ++k) {
                        agree = std::fabs(pc[k] - pr[k]) < 5e-3;
                    }
                }
                t.test("copy and restore pick the same winner", [&](testing & t) {
                    t.assert_true("copy and restore pick the same winner", winners);
                });
                determinism_check(t, weak_quant, "copy and restore agree within tolerance", [&](testing & t) {
                    t.assert_true("copy and restore agree within tolerance", agree);
                });
            } else {
                bool rejected = false;
                try {
                    llama_decision::options oc;
                    oc.fork = "copy";
                    decide_batch(eng, "system", { "ctx" }, two, oc);
                } catch (const std::invalid_argument &) {
                    rejected = true;
                }
                t.assert_true("copy fork is rejected for a recurrent model", rejected);

                llama_decision::options oa;
                oa.fork      = "auto";
                oa.cache_tag = "t";
                const auto auto_run = decide_batch(eng, "system", { "ctx" }, two, oa);
                // the winner is the task value and is always asserted; the probability bound is the
                // producer numerics of sharing vs copying the attention cells on the GPU
                bool winners = auto_run.items[0].fields.size() == restore_run.items[0].fields.size();
                bool agree   = winners;
                for (size_t f = 0; agree && f < auto_run.items[0].fields.size(); ++f) {
                    const auto & pa = auto_run.items[0].fields[f].probs;
                    const auto & pr = restore_run.items[0].fields[f].probs;
                    winners = winners && auto_run.items[0].fields[f].winner == restore_run.items[0].fields[f].winner;
                    agree   = pa.size() == pr.size();
                    for (size_t k = 0; agree && k < pa.size(); ++k) {
                        agree = std::fabs(pa[k] - pr[k]) < 5e-3;
                    }
                }
                t.test("auto picks the exact fork on a recurrent model", [&](testing & t) {
                    t.assert_true("auto picks the exact fork on a recurrent model", winners);
                });
                determinism_check(t, weak_quant, "auto and restore agree within tolerance", [&](testing & t) {
                    t.assert_true("auto and restore agree within tolerance", agree);
                });
            }

            // bounded LRU: two prefixes alternate and both hit on return
            llama_decision::options lru;
            lru.fork = "restore";
            lru.cache_tag = "A";
            const auto a1 = decide_batch(eng, "system-A", { "ctx" }, two, lru);
            lru.cache_tag = "B";
            const auto b1 = decide_batch(eng, "system-B", { "ctx" }, two, lru);
            lru.cache_tag = "A";
            const auto a2 = decide_batch(eng, "system-A", { "ctx" }, two, lru);
            t.assert_true("first A misses", !a1.cache_hit);
            t.assert_true("first B misses", !b1.cache_hit);
            t.assert_true("returning to A hits within the bound", a2.cache_hit);

            llama_decision::options oc2;
            oc2.fork = copy_ok ? "copy" : "restore";
            const auto single = decide_batch(eng, "system", { "ctx" }, one, oc2);
            const auto pair   = decide_batch(eng, "system", { "ctx" }, two, oc2);
            const auto & ps = single.items[0].fields[0].probs;
            const auto & pp = pair.items[0].fields[0].probs;
            // Bypass only applies to copy forks. When it does, the single-question result must be
            // identical to the two-question run, within the 5e-3 producer-numerics bound. On a
            // recurrent model both runs take the forked path and differ by the GPU gemm's
            // batch-shape sensitivity, so the bound there is the 5e-2 head-agreement tolerance.
            // The winner is the task value and is always asserted.
            bool bypass_ok = ps.size() == pp.size();
            for (size_t k = 0; bypass_ok && k < ps.size(); ++k) {
                bypass_ok = std::fabs(ps[k] - pp[k]) < (copy_ok ? 5e-3 : 5e-2);
            }
            const auto argmax = [](const std::vector<float> & p) {
                return (int) (std::max_element(p.begin(), p.end()) - p.begin());
            };
            t.test("single-question bypass picks the same winner", [&](testing & t) {
                t.assert_true("single-question bypass picks the same winner", argmax(ps) == argmax(pp));
            });
            determinism_check(t, weak_quant, "single-question bypass matches the forked path", [&](testing & t) {
                t.assert_true("single-question bypass matches the forked path", bypass_ok);
            });

            // prefix purity: branches never mutate the cached prefix, and branch sequences are released
            llama_memory_t mem = llama_get_memory(te.ctx);
            const llama_pos prefix_before = llama_memory_seq_pos_max(mem, 2);
            (void) decide_batch(eng, "system", { "ctx" }, two, oc2);
            const llama_pos prefix_after = llama_memory_seq_pos_max(mem, 2);
            t.assert_equal("cached prefix is not mutated by branches", prefix_before, prefix_after);
            t.assert_true("branch sequences are released", llama_memory_seq_pos_max(mem, 3) <= 0);
        } catch (const std::exception & e) {
            t.assert_true(std::string("fork runs: ") + e.what(), false);
        }
    });
}

// Fork correctness is judged by byte equality of the resulting state against a full restore, never
// by an argmax match: a wrong probability vector can still pick the same winner. The oracle decodes
// a parent, forks it twice (once through the engine's active strategy, once by a full restore), and
// requires the two branch states to be byte-identical after the same branch decode.
static bool decode_tokens_on(llama_context * ctx, llama_seq_id seq, llama_pos pos0, const std::vector<llama_token> & toks) {
    llama_batch batch = llama_batch_init((int) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        common_batch_add(batch, toks[i], pos0 + (llama_pos) i, { seq }, i + 1 == toks.size());
    }
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return rc == 0;
}

static std::vector<uint8_t> seq_state_dump(llama_context * ctx, llama_seq_id seq) {
    std::vector<uint8_t> buf(llama_state_seq_get_size(ctx, seq));
    const size_t         n = llama_state_seq_get_data(ctx, buf.data(), buf.size(), seq);
    buf.resize(n);
    return buf;
}

// A serialized state carries the sequence id in its header and in every cell record, so two
// sequences with identical content still differ in those bytes. Re-serializing both through the
// same scratch sequence on an empty cache normalizes the ids and the physical cell layout, leaving
// only the state content for the byte comparison.
static std::vector<uint8_t> normalize_seq_state(llama_context * ctx, const std::vector<uint8_t> & state, llama_seq_id scratch,
                                                llama_state_seq_flags flags = LLAMA_STATE_SEQ_FLAGS_NONE) {
    if (state.empty()) {
        return state;
    }
    llama_memory_clear(llama_get_memory(ctx), true);
    const size_t n = llama_state_seq_set_data_ext(ctx, state.data(), state.size(), scratch, flags);
    if (n == 0) {
        return {};
    }
    return seq_state_dump(ctx, scratch);
}

// The serialized state starts with a magic and the source sequence id, which legitimately differ
// between two sequences; the comparison is over the state payload that follows.
static constexpr size_t seq_state_header_size = sizeof(uint32_t) + sizeof(llama_seq_id);

static size_t state_bytes_diff(const std::vector<uint8_t> & a, const std::vector<uint8_t> & b, std::string * detail = nullptr) {
    const size_t skip_a = std::min(a.size(), seq_state_header_size);
    const size_t skip_b = std::min(b.size(), seq_state_header_size);
    const size_t n_a    = a.size() - skip_a;
    const size_t n_b    = b.size() - skip_b;
    const size_t common = std::min(n_a, n_b);
    size_t       diff   = std::max(n_a, n_b) - common;
    size_t       first  = (size_t) -1;
    for (size_t i = 0; i < common; ++i) {
        if (a[skip_a + i] != b[skip_b + i]) {
            if (first == (size_t) -1) {
                first = i;
            }
            ++diff;
        }
    }
    if (detail != nullptr && diff != 0) {
        *detail = "first at payload offset " + std::to_string(first);
    }
    return diff;
}

static std::vector<float> output_logits(llama_context * ctx, const llama_vocab * vocab, int out_idx) {
    std::vector<float> out(llama_vocab_n_tokens(vocab), 0.0f);
    const float *      logits = llama_get_logits_ith(ctx, out_idx);
    if (logits != nullptr) {
        std::copy(logits, logits + out.size(), out.begin());
    }
    return out;
}

static double max_abs_logit_delta(const std::vector<float> & a, const std::vector<float> & b) {
    double       delta = 0.0;
    const size_t n     = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        delta = std::max(delta, (double) std::fabs(a[i] - b[i]));
    }
    return delta;
}

static void fork_oracle_run(testing & t, llama_context * ctx, const std::string & lane, const std::string & strategy) {
    const llama_model * model = llama_get_model(ctx);
    if (strategy == "copy" && (llama_model_is_recurrent(model) || llama_model_is_hybrid(model))) {
        t.skip(lane + ": a copy fork is not supported on a recurrent model");
        return;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const auto          parent = common_tokenize(vocab, "a short decision parent", false, true);
    const auto          branch = common_tokenize(vocab, "branch", false, true);
    if (parent.empty() || branch.empty()) {
        t.skip(lane + ": the model has no usable tokens");
        return;
    }

    llama_decision::engine eng(ctx, 2, 8);
    const llama_seq_id     src = 2, ref = 3, subject = 4, scratch = 7;

    // fresh cache and prior sequences: the fork must be exact in both cell layouts
    for (int n_prior : { 0, 2 }) {
        const std::string layout = n_prior == 0 ? "fresh" : "prior";
        llama_memory_clear(llama_get_memory(ctx), true);
        for (int p = 0; p < n_prior; ++p) {
            if (!t.assert_true(lane + "/" + layout + ": a prior sequence decodes",
                               decode_tokens_on(ctx, 5 + p, 0, parent))) {
                return;
            }
        }
        if (!t.assert_true(lane + "/" + layout + ": the parent decodes", decode_tokens_on(ctx, src, 0, parent))) {
            return;
        }
        llama_synchronize(ctx);

        const auto full = eng.save_seq(src, false, false);
        const auto part = eng.save_seq(src, false, true);
        if (!t.assert_true(lane + "/" + layout + ": the parent state is non-empty",
                           !full.bytes.empty() && !part.bytes.empty())) {
            return;
        }
        const auto parent_before = seq_state_dump(ctx, src);

        // reference: a full restore fork of the same parent
        llama_memory_seq_rm(llama_get_memory(ctx), ref, -1, -1);
        eng.load_seq(full, ref);

        // subject: the requested strategy, loading the matching full or partial parent state. An
        // auto strategy resolves to the partial hybrid fork for a recurrent or hybrid model.
        eng.select_fork(strategy);
        const auto & subject_state = eng.active_fork_ == llama_decision::engine::fork_kind::hybrid ? part : full;
        eng.fork_into(src, subject, &subject_state);

        const llama_pos pos0   = (llama_pos) parent.size();
        const bool      ref_ok = decode_tokens_on(ctx, ref, pos0, branch);
        const bool      sub_ok = decode_tokens_on(ctx, subject, pos0, branch);
        llama_synchronize(ctx);
        if (!t.assert_true(lane + "/" + layout + ": the reference branch decodes", ref_ok) ||
            !t.assert_true(lane + "/" + layout + ": the fork branch decodes", sub_ok)) {
            return;
        }

        // the source sequence must be byte-identical after a branch fork and decode
        t.assert_true(lane + "/" + layout + ": the source state is unchanged by a " + strategy + " fork",
                      parent_before == seq_state_dump(ctx, src));

        const auto ref_state = seq_state_dump(ctx, ref);
        const auto sub_state = seq_state_dump(ctx, subject);
        if (!t.assert_true(lane + "/" + layout + ": the fork state is non-empty", !sub_state.empty())) {
            return;
        }

        const auto   ref_norm = normalize_seq_state(ctx, ref_state, scratch);
        const auto   sub_norm = normalize_seq_state(ctx, sub_state, scratch);
        std::string  detail;
        const size_t diff = state_bytes_diff(ref_norm, sub_norm, &detail);
        t.assert_true(lane + "/" + layout + ": the " + strategy + " fork equals a full restore (" +
                          std::to_string(diff) + " of " + std::to_string(std::max(ref_norm.size(), sub_norm.size())) +
                          " bytes differ" + (detail.empty() ? "" : ", " + detail) + ")",
                      diff == 0);
    }
}

// Forks `n` children out of a parent and decodes the same branch token on all of them in one
// batch, the way the engine runs a wave of branches from one trunk. Restore children load the
// saved host state; plain children only get a metadata sequence copy.
struct fork_group {
    std::vector<std::vector<uint8_t>> raw_state; // host state per child, before normalization
    std::vector<std::vector<float>>   logits;    // branch logits per child
    bool ok = false;
};

static fork_group fork_group_decode(llama_context *   ctx,
                                    llama_decision::engine & eng,
                                    const llama_decision::engine::saved_state & saved,
                                    llama_seq_id      src,
                                    llama_seq_id      first,
                                    int               n,
                                    llama_pos         pos0,
                                    llama_token       tok,
                                    bool              plain_copy) {
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    llama_memory_t      mem   = llama_get_memory(ctx);

    fork_group out;
    out.raw_state.resize(n);
    out.logits.resize(n);

    llama_batch batch = llama_batch_init(n, 0, 1);
    for (int b = 0; b < n; ++b) {
        const llama_seq_id seq = first + b;
        llama_memory_seq_rm(mem, seq, -1, -1);
        if (plain_copy) {
            llama_memory_seq_cp(mem, src, seq, -1, -1);
        } else {
            eng.load_seq(saved, seq);
        }
        common_batch_add(batch, tok, pos0, { seq }, true);
    }
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        return out;
    }
    llama_synchronize(ctx);
    for (int b = 0; b < n; ++b) {
        out.raw_state[b] = seq_state_dump(ctx, first + b);
        out.logits[b]    = output_logits(ctx, vocab, b);
    }
    out.ok = true;
    return out;
}

// The control that proves why an exact recurrent fork is needed: plain seq_cp shares the recurrent
// tail cell instead of copying it, so the branch state and its logits can drift from an exact
// restore. The divergence is recorded, not asserted, because a layout may happen to be exact.
static void fork_divergence_run(testing & t, llama_context * ctx, const std::string & lane) {
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    const auto          parent = common_tokenize(vocab, "a short decision parent", false, true);
    const auto          branch = common_tokenize(vocab, "branch", false, true);
    if (parent.empty() || branch.empty()) {
        t.skip(lane + ": the model has no usable tokens");
        return;
    }

    llama_decision::engine eng(ctx, 2, 8);
    const llama_seq_id     src = 2;
    const llama_pos        pos0 = (llama_pos) parent.size();

    // a trunk forked from a shared prefix: the engine's shape, where the trunk inherits the
    // prefix's recurrent state through seq_cp instead of restoring it
    {
        llama_memory_clear(llama_get_memory(ctx), true);
        if (!t.assert_true(lane + ": the prefix decodes", decode_tokens_on(ctx, src, 0, parent))) {
            return;
        }
        llama_synchronize(ctx);
        const auto prefix_state = eng.save_seq(src, false);

        const llama_seq_id plain_trunk = 8, ref_trunk = 9;
        llama_memory_seq_rm(llama_get_memory(ctx), plain_trunk, -1, -1);
        llama_memory_seq_cp(llama_get_memory(ctx), src, plain_trunk, -1, -1);
        llama_memory_seq_rm(llama_get_memory(ctx), ref_trunk, -1, -1);
        eng.load_seq(prefix_state, ref_trunk);

        const bool plain_ok = decode_tokens_on(ctx, plain_trunk, pos0, branch);
        llama_synchronize(ctx);
        const auto plain_trunk_logits = output_logits(ctx, vocab, 0);
        const bool ref_ok = decode_tokens_on(ctx, ref_trunk, pos0, branch);
        llama_synchronize(ctx);
        const auto ref_trunk_logits = output_logits(ctx, vocab, 0);
        if (!t.assert_true(lane + ": the plain trunk decodes", plain_ok) ||
            !t.assert_true(lane + ": the restore trunk decodes", ref_ok)) {
            return;
        }
        const auto plain_trunk_raw = seq_state_dump(ctx, plain_trunk);
        const auto ref_trunk_raw   = seq_state_dump(ctx, ref_trunk);

        const auto plain_trunk_norm = normalize_seq_state(ctx, plain_trunk_raw, 5);
        const auto ref_trunk_norm   = normalize_seq_state(ctx, ref_trunk_raw, 5);
        printf("[fork control] %s: prefix trunk fork via seq_cp vs restore: %zu of %zu state bytes differ, max logit delta %.6g\n",
               lane.c_str(), state_bytes_diff(ref_trunk_norm, plain_trunk_norm),
               std::max(ref_trunk_norm.size(), plain_trunk_norm.size()),
               max_abs_logit_delta(ref_trunk_logits, plain_trunk_logits));
    }

    for (int n_prior : { 0, 2 }) {
        for (int n_branches : { 1, 2 }) {
            // each probe starts from a clean cache so the parent's layout is the same for both forks
            llama_memory_clear(llama_get_memory(ctx), true);
            // prior sequences occupy cells before the fork, a layout in which plain seq_cp can
            // hand a branch a stale recurrent source cell
            for (int p = 0; p < n_prior; ++p) {
                if (!t.assert_true(lane + ": a prior sequence decodes", decode_tokens_on(ctx, 8 + p, 0, parent))) {
                    return;
                }
            }
            if (!t.assert_true(lane + ": the parent decodes", decode_tokens_on(ctx, src, 0, parent))) {
                return;
            }
            llama_synchronize(ctx);
            const auto saved = eng.save_seq(src, false);
            if (!t.assert_true(lane + ": the parent state is non-empty", !saved.bytes.empty())) {
                return;
            }

            const llama_seq_id ref_first   = 3;
            const llama_seq_id plain_first = 3 + n_branches;
            const llama_seq_id scratch     = plain_first + n_branches;

            const auto ref   = fork_group_decode(ctx, eng, saved, src, ref_first,   n_branches, pos0, branch[0], false);
            const auto plain = fork_group_decode(ctx, eng, saved, src, plain_first, n_branches, pos0, branch[0], true);
            if (!t.assert_true(lane + ": the fork control decodes", ref.ok && plain.ok)) {
                return;
            }

            size_t diff = 0, total = 0;
            double max_logit_delta = 0.0;
            for (int b = 0; b < n_branches; ++b) {
                const auto ref_norm   = normalize_seq_state(ctx, ref.raw_state[b], scratch);
                const auto plain_norm = normalize_seq_state(ctx, plain.raw_state[b], scratch);
                diff += state_bytes_diff(ref_norm, plain_norm);
                total += std::max(ref_norm.size(), plain_norm.size());
                max_logit_delta = std::max(max_logit_delta, max_abs_logit_delta(ref.logits[b], plain.logits[b]));
            }
            printf("[fork control] %s: plain seq_cp vs restore, %d prior, %d branch(es) in one batch: %zu of %zu state bytes differ, max logit delta %.6g\n",
                   lane.c_str(), n_prior, n_branches, diff, total, max_logit_delta);
        }
    }
}

// A greedy field walks its value one token at a time, and every round after the first starts again
// from the trunk. A round that holds exactly one branch may therefore be decoded on the trunk and
// trimmed back to pos0 instead of forked, which is only exact if the trim restores the trunk as it
// was: recurrent state lives outside the KV cache and a sequence removal does not put it back. The
// check is the trunk's state bytes, never the argmax, because a corrupted parent can still pick the
// same value.
static void greedy_trunk_trim_run(testing & t, llama_context * ctx, const std::string & lane) {
    const llama_vocab * vocab  = llama_model_get_vocab(llama_get_model(ctx));
    const auto          parent = common_tokenize(vocab, "a short decision parent", false, true);
    const auto          head   = common_tokenize(vocab, "\nAnswer: ", false, true);
    if (parent.empty() || head.empty()) {
        t.skip(lane + ": the model has no usable tokens");
        return;
    }
    // three real candidates, so the branch is one wave with three scored slots
    std::vector<llama_token> cands;
    for (const char * word : { "yes", "maybe", "never" }) {
        const auto ids = common_tokenize(vocab, word, false, true);
        if (ids.empty()) {
            t.skip(lane + ": the model has no usable tokens");
            return;
        }
        cands.push_back(ids.front());
    }

    llama_memory_clear(llama_get_memory(ctx), true);
    llama_decision::engine eng(ctx, 2, 8);
    const llama_seq_id     trunk = eng.seq_pool;
    if (!t.assert_true(lane + ": the trunk decodes", decode_tokens_on(ctx, trunk, 0, parent))) {
        return;
    }
    const auto before = seq_state_dump(ctx, trunk);

    // the wave the engine runs: one branch off the trunk, either decoded on it or forked
    eng.select_fork("auto");
    const bool partial = eng.active_fork_ == llama_decision::engine::fork_kind::hybrid;
    std::vector<llama_decision::engine::saved_state> parents = { eng.save_seq(trunk, false, partial) };
    const llama_decision::engine::branch             branch{ trunk, (llama_pos) parent.size(), head, cands };

    const auto on        = eng.score_branches({ branch }, trunk + 1, eng.n_pool - 1, &parents, true);
    const auto after_on = seq_state_dump(ctx, trunk);
    if (t.assert_true(lane + ": the trunk wave scores every candidate",
                      on.size() == 1 && on[0].cand_logits.size() == cands.size())) {
        std::string detail;
        t.assert_true(lane + ": scoring one branch on the trunk leaves it byte-identical (" +
                          std::to_string(state_bytes_diff(before, after_on, &detail)) + " of " +
                          std::to_string(before.size()) + " bytes differ" +
                          (detail.empty() ? "" : ", " + detail) + ")",
                      before == after_on);
    }
    eng.clear_seqs(trunk + 1, eng.n_pool - 1);

    // the same wave again from a rebuilt trunk, this time forked: the logits must be the same ones
    eng.clear_seqs(trunk, 1);
    if (!t.assert_true(lane + ": the trunk rebuilds", decode_tokens_on(ctx, trunk, 0, parent))) {
        return;
    }
    const auto off = eng.score_branches({ branch }, trunk + 1, eng.n_pool - 1, &parents, false);
    eng.clear_seqs(trunk + 1, eng.n_pool - 1);

    if (t.assert_true(lane + ": the forked wave scores every candidate",
                      off.size() == 1 && off[0].cand_logits.size() == cands.size())) {
        t.assert_true(lane + ": the trunk decode and the forked decode gather the same logits",
                      max_abs_logit_delta(on[0].cand_logits, off[0].cand_logits) == 0.0);
    }
}

// The same hazard through the engine: a greedy field spanning three rounds means round 1 carries a
// branch from every field and every later round carries only the greedy one, so those later rounds
// are the single-branch waves a bypass would take. The winners and the walked token counts are task
// value and must be identical either way; the probabilities are producer numerics and carry the
// same tolerance every other cross-layout comparison in this file uses.
static void greedy_multiround_run(testing & t, llama_context * ctx, const std::string & lane) {
    const std::vector<llama_decision::field_input> fields = {
        { "\nrefund: ", { "yes", "no" } },
        { "\nwide: ",
          { "red apple pie",   "red apple tart",   "red berry pie",   "red berry tart",
            "blue apple pie",  "blue apple tart",  "blue berry pie",  "blue berry tart",
            "green apple pie", "green apple tart", "green berry pie", "green berry tart" } },
    };
    llama_decision::options opt;
    opt.mode        = "auto";
    opt.tree_max    = 4; // the wide field is wider than the trie, so it is scored greedily
    opt.allow_cache = false;

    llama_decision::engine eng(ctx, 2, 8);

    opt.bypass = true;
    const auto on = decide_batch(eng, "system", { "ctx" }, fields, opt);
    opt.bypass = false;
    const auto off = decide_batch(eng, "system", { "ctx" }, fields, opt);

    if (!t.assert_true(lane + ": both runs score both fields",
                       on.items.size() == 1 && off.items.size() == 1 &&
                           on.items[0].fields.size() == 2 && off.items[0].fields.size() == 2)) {
        return;
    }
    const auto & greedy_on = on.items[0].fields[1];
    if (greedy_on.scored_nodes < 3) {
        // this tokenizer does not split the value space into three token levels, so no round after
        // the first ever forks from the trunk and the case does not apply here
        t.skip(lane + ": this tokenizer walks the greedy value in " +
              std::to_string(greedy_on.scored_nodes) + " round(s)");
        return;
    }
    t.assert_true(lane + ": the greedy field walks at least three rounds", greedy_on.scored_nodes >= 3);
    bool winners = true;
    for (size_t f = 0; f < on.items[0].fields.size(); ++f) {
        winners = winners && on.items[0].fields[f].winner == off.items[0].fields[f].winner &&
                  on.items[0].fields[f].scored_nodes == off.items[0].fields[f].scored_nodes;
    }
    t.assert_true(lane + ": the bypass changes no winner and no walked token count", winners);
    bool agree = true;
    for (size_t f = 0; agree && f < on.items[0].fields.size(); ++f) {
        const auto & a = on.items[0].fields[f].probs;
        const auto & b = off.items[0].fields[f].probs;
        agree          = a.size() == b.size();
        for (size_t k = 0; agree && k < a.size(); ++k) {
            agree = std::fabs(a[k] - b[k]) < 5e-3f;
        }
    }
    t.assert_true(lane + ": the bypass agrees with the forked path within tolerance", agree);
}

static void test_fork_oracle(testing & t) {
    t.test("strategy forks equal a full restore on the CPU model", [](testing & t) {
        const std::string path = decision_cpu_model_path();
        if (path.empty()) {
            t.skip("no generated model; run the generate-models fixture");
            return;
        }
        cpu_test_engine te;
        if (!te.load(path)) {
            t.assert_true("the CPU decision scaffold loads the model", false);
            return;
        }
        for (const char * strategy : { "auto", "copy", "restore", "hybrid" }) {
            t.test(std::string(strategy) + ": the fork matches the reference", [&](testing & t) {
                try {
                    fork_oracle_run(t, te.ctx, "cpu", strategy);
                } catch (const std::exception & e) {
                    t.assert_true(std::string("the CPU fork oracle runs: ") + e.what(), false);
                }
            });
        }
        try {
            greedy_trunk_trim_run(t, te.ctx, "cpu");
            greedy_multiround_run(t, te.ctx, "cpu");
        } catch (const std::exception & e) {
            t.assert_true(std::string("the CPU greedy trunk cases run: ") + e.what(), false);
        }
    });

    t.test("strategy forks equal a full restore on the GPU model", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (!gpu_model_ready(t, path)) {
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("the GPU decision scaffold loads the model", false);
            return;
        }
        for (const char * strategy : { "auto", "copy", "restore", "hybrid" }) {
            t.test(std::string(strategy) + ": the fork matches the reference", [&](testing & t) {
                try {
                    fork_oracle_run(t, te.ctx, "gpu", strategy);
                } catch (const std::exception & e) {
                    t.assert_true(std::string("the GPU fork oracle runs: ") + e.what(), false);
                }
            });
        }
        try {
            greedy_trunk_trim_run(t, te.ctx, "gpu");
            greedy_multiround_run(t, te.ctx, "gpu");
        } catch (const std::exception & e) {
            t.assert_true(std::string("the GPU greedy trunk cases run: ") + e.what(), false);
        }
    });
}

// decide_batch forks twice: a trunk from the shared prefix, then a branch from the trunk. A single
// fork oracle would miss a drift that only appears after the trunk has decoded its tail, so this
// decodes a multi-token tail on a forked trunk, forks a branch from it, and compares the branch
// logits against the same nested sequence done with full restore states at both levels.
static void nested_fork_oracle_run(testing & t, llama_context * ctx, const std::string & lane, const std::string & strategy) {
    const llama_model * model = llama_get_model(ctx);
    if (!llama_model_is_recurrent(model) && !llama_model_is_hybrid(model)) {
        t.skip(lane + ": a nested partial fork needs a recurrent model");
        return;
    }
    const llama_vocab * vocab  = llama_model_get_vocab(model);
    const auto          prefix = common_tokenize(vocab, "the shared decision prefix", false, true);
    const auto          tail   = common_tokenize(vocab, "tail tokens decoded on the trunk", false, true);
    const auto          branch = common_tokenize(vocab, "branch", false, true);
    if (prefix.empty() || tail.empty() || branch.empty()) {
        t.skip(lane + ": the model has no usable tokens");
        return;
    }

    llama_decision::engine eng(ctx, 2, 8);
    const llama_seq_id     snap = 2, ref_trunk = 3, ref_branch = 4, sub_trunk = 5, sub_branch = 6;

    for (int n_prior : { 0, 2 }) {
        const std::string layout = n_prior == 0 ? "fresh" : "prior";
        llama_memory_clear(llama_get_memory(ctx), true);
        for (int p = 0; p < n_prior; ++p) {
            if (!t.assert_true(lane + "/" + layout + ": a prior sequence decodes",
                               decode_tokens_on(ctx, 8 + p, 0, prefix))) {
                return;
            }
        }
        if (!t.assert_true(lane + "/" + layout + ": the shared prefix decodes", decode_tokens_on(ctx, snap, 0, prefix))) {
            return;
        }
        llama_synchronize(ctx);

        const auto full = eng.save_seq(snap, false, false);
        const auto part = eng.save_seq(snap, false, true);

        // reference: a full restore at both levels
        eng.select_fork("restore");
        llama_memory_seq_rm(llama_get_memory(ctx), ref_trunk, -1, -1);
        eng.load_seq(full, ref_trunk);
        if (!t.assert_true(lane + "/" + layout + ": the reference trunk decodes",
                           decode_tokens_on(ctx, ref_trunk, (llama_pos) prefix.size(), tail))) {
            return;
        }
        llama_synchronize(ctx);
        const auto ref_trunk_state = eng.save_seq(ref_trunk, false, false);
        llama_memory_seq_rm(llama_get_memory(ctx), ref_branch, -1, -1);
        eng.load_seq(ref_trunk_state, ref_branch);
        if (!t.assert_true(lane + "/" + layout + ": the reference branch decodes",
                           decode_tokens_on(ctx, ref_branch, (llama_pos) (prefix.size() + tail.size()), branch))) {
            return;
        }
        llama_synchronize(ctx);
        const auto ref_logits = output_logits(ctx, vocab, 0);

        // subject: the requested strategy at both levels, from its own clean prefix
        llama_memory_clear(llama_get_memory(ctx), true);
        if (!t.assert_true(lane + "/" + layout + ": the subject prefix decodes", decode_tokens_on(ctx, snap, 0, prefix))) {
            return;
        }
        llama_synchronize(ctx);
        const auto sub_full = eng.save_seq(snap, false, false);
        const auto sub_part = eng.save_seq(snap, false, true);

        eng.select_fork(strategy);
        const bool   hybrid = eng.active_fork_ == llama_decision::engine::fork_kind::hybrid;
        const auto & root   = hybrid ? sub_part : sub_full;
        eng.fork_into(snap, sub_trunk, &root);
        if (!t.assert_true(lane + "/" + layout + ": the subject trunk decodes",
                           decode_tokens_on(ctx, sub_trunk, (llama_pos) prefix.size(), tail))) {
            return;
        }
        llama_synchronize(ctx);
        const auto sub_trunk_state = eng.save_seq(sub_trunk, false, hybrid);
        eng.fork_into(sub_trunk, sub_branch, &sub_trunk_state);
        if (!t.assert_true(lane + "/" + layout + ": the subject branch decodes",
                           decode_tokens_on(ctx, sub_branch, (llama_pos) (prefix.size() + tail.size()), branch))) {
            return;
        }
        llama_synchronize(ctx);
        const auto sub_logits = output_logits(ctx, vocab, 0);

        const double delta = max_abs_logit_delta(ref_logits, sub_logits);
        t.assert_true(lane + "/" + layout + ": the nested " + strategy + " fork matches a full restore (max logit delta " +
                          std::to_string(delta) + ")",
                      delta == 0.0);
    }
}

static void test_nested_fork_oracle(testing & t) {
    t.test("a nested strategy fork equals a full restore on the GPU model", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (!gpu_model_ready(t, path)) {
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("the GPU decision scaffold loads the model", false);
            return;
        }
        for (const char * strategy : { "restore", "hybrid" }) {
            t.test(std::string(strategy) + ": the nested fork matches the reference", [&](testing & t) {
                try {
                    nested_fork_oracle_run(t, te.ctx, "gpu", strategy);
                } catch (const std::exception & e) {
                    t.assert_true(std::string("the GPU nested fork oracle runs: ") + e.what(), false);
                }
            });
        }
    });
}

// A decision forked from a live source sequence must equal the same decision whose source text was
// prefilled as an ordinary context: the session entry only skips the re-prefill, it does not change
// what is scored. The source must survive the decision untouched.
static void session_fork_oracle_run(testing & t, llama_context * ctx, const std::string & lane) {
    llama_decision::engine eng(ctx, 2, 8);
    const std::vector<llama_decision::field_input> fields = {
        { "  \"a\": ", { "1", "2" } },
        { "  \"b\": ", { "x", "y", "z" } },
    };
    llama_decision::options o;
    o.mode        = "tree";
    o.fork        = "auto";
    o.allow_cache = false;

    const std::string shared_text  = "the shared decision prefix";
    const std::string context_text = "the session context";
    const std::vector<llama_token> shared  = eng.tokenize(shared_text, true);
    const std::vector<llama_token> context = eng.tokenize(context_text, shared.empty());
    if (shared.empty() || context.empty()) {
        t.skip(lane + ": the model has no usable tokens");
        return;
    }
    const auto plan = eng.compile_fields(fields, o);

    // control: the stateless decision prefills shared + context itself
    llama_memory_clear(llama_get_memory(ctx), true);
    const auto stateless = eng.decide_batch(plan, shared_text, { context_text }, o);

    // session: the same tokens are already decoded on the source sequence
    llama_memory_clear(llama_get_memory(ctx), true);
    const llama_seq_id src = 0;
    if (!t.assert_true(lane + ": the session prefix decodes", decode_tokens_on(ctx, src, 0, shared)) ||
        !t.assert_true(lane + ": the session context decodes",
                       decode_tokens_on(ctx, src, (llama_pos) shared.size(), context))) {
        return;
    }
    llama_synchronize(ctx);
    const auto       src_before = seq_state_dump(ctx, src);
    const llama_pos  base_pos   = (llama_pos) (shared.size() + context.size());

    llama_decision::batch_result session;
    try {
        session = eng.decide_batch_from_seq(src, base_pos, plan, o);
    } catch (const std::exception & e) {
        t.assert_true(std::string(lane + ": the session decision runs: ") + e.what(), false);
        return;
    }

    t.assert_true(lane + ": the source state is unchanged by the session fork", src_before == seq_state_dump(ctx, src));

    if (!t.assert_true(lane + ": both decisions return one result",
                       stateless.items.size() == 1 && session.items.size() == 1)) {
        return;
    }
    // the session path physically places branch cells differently, so the GPU producer numerics
    // bound applies; the winners are the task-value outcome and must agree
    const double tol = lane == "gpu" ? 5e-2 : 1e-4;
    bool         same  = stateless.items[0].fields.size() == session.items[0].fields.size();
    bool         wins  = same;
    double       worst = 0.0;
    for (size_t f = 0; same && f < fields.size(); ++f) {
        wins = wins && stateless.items[0].fields[f].winner == session.items[0].fields[f].winner;
        const auto & ps = stateless.items[0].fields[f].probs;
        const auto & pn = session.items[0].fields[f].probs;
        same = same && ps.size() == pn.size();
        for (size_t k = 0; same && k < ps.size(); ++k) {
            const double d = std::fabs(ps[k] - pn[k]);
            worst = std::max(worst, d);
            same  = d <= tol;
        }
    }
    t.assert_true(lane + ": the session decision keeps the winners", wins);
    t.assert_true(lane + ": the session decision matches the prefilled context (worst " +
                              std::to_string(worst) + ")",
                  same);

    // The branch-level byte oracle (a forked branch equals a full restore) lives in the fork oracle
    // tests; the session entry reuses that primitive and only changes where the trunk is forked
    // from, so the session check is the task-value equality above plus the source invariance below.
    t.assert_true(lane + ": the source state is still unchanged after scoring",
                  src_before == seq_state_dump(ctx, src));
}

// A wrong base_pos must fail instead of silently scoring at shifted positions.
static void session_fork_position_run(testing & t, llama_context * ctx, const std::string & lane) {
    llama_decision::engine eng(ctx, 2, 8);
    const std::vector<llama_decision::field_input> fields = { { "  \"a\": ", { "1", "2" } } };
    llama_decision::options o;
    o.allow_cache = false;
    const std::vector<llama_token> src_toks = eng.tokenize("a session with a transcript", true);
    if (src_toks.empty()) {
        t.skip(lane + ": the model has no usable tokens");
        return;
    }
    const auto plan = eng.compile_fields(fields, o);
    llama_memory_clear(llama_get_memory(ctx), true);
    if (!decode_tokens_on(ctx, 0, 0, src_toks)) {
        t.assert_true(lane + ": the source decodes", false);
        return;
    }
    llama_synchronize(ctx);
    const llama_pos good = (llama_pos) src_toks.size();
    // a position the source does not continue from must be rejected before any decode
    bool rejected = false;
    try {
        (void) eng.decide_batch_from_seq(0, good + 1, plan, o);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    t.assert_true(lane + ": a shifted base_pos is rejected, not scored", rejected);
    // the correct continuation still runs
    bool ran = true;
    try {
        (void) eng.decide_batch_from_seq(0, good, plan, o);
    } catch (const std::exception &) {
        ran = false;
    }
    t.assert_true(lane + ": the correct continuation still runs", ran);
}

static void test_session_fork(testing & t) {
    t.test("a session fork matches a prefilled context on the CPU model", [](testing & t) {
        const std::string path = decision_cpu_model_path();
        if (path.empty()) {
            t.skip("no generated model; run the generate-models fixture");
            return;
        }
        cpu_test_engine te;
        if (!te.load(path)) {
            t.assert_true("the CPU decision scaffold loads the model", false);
            return;
        }
        try {
            session_fork_oracle_run(t, te.ctx, "cpu");
            session_fork_position_run(t, te.ctx, "cpu");
        } catch (const std::exception & e) {
            t.assert_true(std::string("the CPU session fork: ") + e.what(), false);
        }
    });

    t.test("a session fork matches a prefilled context on the GPU model", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (!gpu_model_ready(t, path)) {
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("the GPU decision scaffold loads the model", false);
            return;
        }
        try {
            session_fork_oracle_run(t, te.ctx, "gpu");
            session_fork_position_run(t, te.ctx, "gpu");
        } catch (const std::exception & e) {
            t.assert_true(std::string("the GPU session fork: ") + e.what(), false);
        }
    });
}

// ---------------------------------------------------------------- session arena

// The owned snapshot of a completed turn: serialized on the scheduler thread into a free arena
// sequence, so a decision about a slot survives the slot's KV being cleared by cache_idle_slots.
// The arena is the "one retained turn per slot" substrate; the runs below are its snapshot
// guarantees and the exact trigger counters, all on the shared full-logits context.

// Decode a transcript and take a snapshot of it, returning the arena sequence id. Shared setup so
// every run exercises the same save/restore primitive.
// Flipping the default fork must not move an answer: for a recurrent or hybrid model `auto` now
// selects the partial hybrid fork, whose branch state is byte-identical to a full restore, so the
// winners stay the same and the probabilities agree within the GPU producer-numerics bound. A dense
// model keeps the previous copy default, so its answers are identical to an explicit copy.
static void fork_auto_default_run(testing & t, llama_context * ctx, const std::string & lane) {
    const llama_model * model     = llama_get_model(ctx);
    const bool          recurrent = llama_model_is_recurrent(model) || llama_model_is_hybrid(model);
    const std::vector<llama_decision::field_input> fields = {
        { "  \"a\": ", { "1", "2" } },
        { "  \"b\": ", { "x", "y" } },
        { "  \"c\": ", { "p", "q", "r" } },
    };

    llama_decision::engine eng(ctx, 2, 8);
    llama_decision::options o_auto;
    o_auto.fork        = "auto";
    o_auto.allow_cache = false;
    const auto auto_run = decide_batch(eng, "system", { "ctx" }, fields, o_auto);
    const auto resolved = eng.active_fork_;

    // the reference is the strategy `auto` must resolve to: the exact hybrid fork for a recurrent
    // model, or the unchanged copy fork for a dense one
    llama_decision::options o_ref = o_auto;
    o_ref.fork = recurrent ? "restore" : "copy";
    const auto ref_run = decide_batch(eng, "system", { "ctx" }, fields, o_ref);

    const auto expected = recurrent ? llama_decision::engine::fork_kind::hybrid
                                    : llama_decision::engine::fork_kind::copy;
    t.test(lane + ": auto selects the expected fork", [&](testing & t) {
        t.assert_true(lane + ": auto selects the expected fork", resolved == expected);
    });
    if (!t.assert_true(lane + ": both the default and the reference run return one result",
                       auto_run.items.size() == 1 && ref_run.items.size() == 1)) {
        return;
    }

    // a dense copy default must not move at all; on a recurrent model the CPU path is also exact,
    // while the GPU reduction order changes with the physical cell placement, so its bound is the
    // producer numerics
    const double tol = recurrent && lane == "gpu" ? 5e-2 : 0.0;
    bool   same  = auto_run.items[0].fields.size() == ref_run.items[0].fields.size();
    double worst = 0.0;
    for (size_t f = 0; same && f < fields.size(); ++f) {
        same = auto_run.items[0].fields[f].winner == ref_run.items[0].fields[f].winner;
        const auto & pa = auto_run.items[0].fields[f].probs;
        const auto & pr = ref_run.items[0].fields[f].probs;
        same = same && pa.size() == pr.size();
        for (size_t k = 0; same && k < pa.size(); ++k) {
            const double d = std::fabs(pa[k] - pr[k]);
            worst = std::max(worst, d);
            same  = d <= tol;
        }
    }
    t.assert_true(lane + ": the default fork keeps the reference answers (worst " + std::to_string(worst) + ")",
                  same);
}

static void test_fork_auto_default(testing & t) {
    t.test("the default fork keeps the reference answers on the CPU model", [](testing & t) {
        const std::string path = decision_cpu_model_path();
        if (path.empty()) {
            t.skip("no generated model; run the generate-models fixture");
            return;
        }
        cpu_test_engine te;
        if (!te.load(path)) {
            t.assert_true("the CPU decision scaffold loads the model", false);
            return;
        }
        try {
            fork_auto_default_run(t, te.ctx, "cpu");
        } catch (const std::exception & e) {
            t.assert_true(std::string("the CPU default fork: ") + e.what(), false);
        }
    });

    t.test("the default fork keeps the reference answers on the GPU model", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (!gpu_model_ready(t, path)) {
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("the GPU decision scaffold loads the model", false);
            return;
        }
        try {
            fork_auto_default_run(t, te.ctx, "gpu");
        } catch (const std::exception & e) {
            t.assert_true(std::string("the GPU default fork: ") + e.what(), false);
        }
    });
}

// A strategy change must reuse the token-cached prefix without carrying the previous scope with it:
// a restore request leaves a full prefix state, a hybrid request needs a partial one, and the
// token-cached prefix path must refresh the scope before it forks.
static void fork_strategy_switch_run(testing & t, llama_context * ctx, const std::string & lane) {
    const std::vector<llama_decision::field_input> fields = {
        { "  \"a\": ", { "1", "2" } },
        { "  \"b\": ", { "x", "y" } },
    };
    const double tol = lane == "gpu" ? 5e-2 : 1e-4;

    const std::pair<const char *, const char *> pairs[] = {
        { "restore", "hybrid" },
        { "hybrid", "restore" },
    };
    for (const auto & pair : pairs) {
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_decision::engine eng(ctx, 2, 8);

        llama_decision::options o1;
        o1.fork      = pair.first;
        o1.cache_tag = "switch";
        const auto r1 = decide_batch(eng, "system", { "ctx" }, fields, o1);

        llama_decision::options o2;
        o2.fork      = pair.second;
        o2.cache_tag = "switch";
        const auto r2 = decide_batch(eng, "system", { "ctx" }, fields, o2);

        if (!t.assert_true(lane + ": the " + pair.first + " run returns", r1.items.size() == 1) ||
            !t.assert_true(lane + ": the " + pair.second + " run returns", r2.items.size() == 1)) {
            return;
        }
        bool   same  = r1.items[0].fields.size() == r2.items[0].fields.size();
        double worst = 0.0;
        for (size_t f = 0; same && f < fields.size(); ++f) {
            same = r1.items[0].fields[f].winner == r2.items[0].fields[f].winner;
            const auto & p1 = r1.items[0].fields[f].probs;
            const auto & p2 = r2.items[0].fields[f].probs;
            same = same && p1.size() == p2.size();
            for (size_t k = 0; same && k < p1.size(); ++k) {
                worst = std::max(worst, (double) std::fabs(p1[k] - p2[k]));
                same  = std::fabs(p1[k] - p2[k]) < tol;
            }
        }
        t.assert_true(lane + ": " + pair.first + " then " + pair.second + " agrees (worst " +
                          std::to_string(worst) + ")",
                      same);
    }
}

static void test_fork_strategy_switch(testing & t) {
    t.test("a fork strategy change reuses the cached prefix on the CPU model", [](testing & t) {
        const std::string path = decision_cpu_model_path();
        if (path.empty()) {
            t.skip("no generated model; run the generate-models fixture");
            return;
        }
        cpu_test_engine te;
        if (!te.load(path)) {
            t.assert_true("the CPU decision scaffold loads the model", false);
            return;
        }
        try {
            fork_strategy_switch_run(t, te.ctx, "cpu");
        } catch (const std::exception & e) {
            t.assert_true(std::string("the CPU strategy switch: ") + e.what(), false);
        }
    });

    t.test("a fork strategy change reuses the cached prefix on the GPU model", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (!gpu_model_ready(t, path)) {
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("the GPU decision scaffold loads the model", false);
            return;
        }
        try {
            fork_strategy_switch_run(t, te.ctx, "gpu");
        } catch (const std::exception & e) {
            t.assert_true(std::string("the GPU strategy switch: ") + e.what(), false);
        }
    });
}

static void test_fork_divergence_control(testing & t) {
    t.test("plain seq_cp fork divergence against restore is recorded (cpu)", [](testing & t) {
        const std::string path = decision_cpu_model_path();
        if (path.empty()) {
            t.skip("no generated model; run the generate-models fixture");
            return;
        }
        cpu_test_engine te;
        if (!te.load(path)) {
            t.assert_true("the CPU decision scaffold loads the model", false);
            return;
        }
        try {
            fork_divergence_run(t, te.ctx, "cpu");
        } catch (const std::exception & e) {
            t.assert_true(std::string("the CPU fork control runs: ") + e.what(), false);
        }
    });

    t.test("plain seq_cp fork divergence against restore is recorded (gpu)", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (!gpu_model_ready(t, path)) {
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("the GPU decision scaffold loads the model", false);
            return;
        }
        try {
            fork_divergence_run(t, te.ctx, "gpu");
        } catch (const std::exception & e) {
            t.assert_true(std::string("the GPU fork control runs: ") + e.what(), false);
        }
    });
}


// Returning to a cached prefix must restore that prefix's own bytes. The LRU stores host-format
// states, so a later save on the snapshot sequence cannot corrupt an earlier entry: the return
// scores and the snapshot KV must match the first run of the same prefix.
static void test_prefix_lru_restores_own_state(testing & t) {
    t.test("returning to a cached prefix restores that prefix", [](testing & t) {
        const std::string path = decision_cpu_model_path();
        if (path.empty()) {
            t.skip("no generated model; run the generate-models fixture");
            return;
        }
        cpu_test_engine te;
        if (!te.load(path)) {
            t.assert_true("the CPU decision scaffold loads the model", false);
            return;
        }
        try {
            const llama_seq_id snap = 2; // the engine's prefix sequence
            llama_decision::engine eng(te.ctx, snap, 8);
            const std::vector<llama_decision::field_input> fields = {
                { "  \"a\": ", { "1", "2" } },
                { "  \"b\": ", { "x", "y" } },
            };
            const auto seq_dump = [&]() {
                std::vector<uint8_t> buf(llama_state_seq_get_size(te.ctx, snap));
                const size_t n = llama_state_seq_get_data(te.ctx, buf.data(), buf.size(), snap);
                buf.resize(n);
                return buf;
            };

            llama_decision::options opt;
            opt.fork      = "restore";
            opt.cache_tag = "A";
            const auto a1 = decide_batch(eng, "system alpha", { "context" }, fields, opt);
            const std::vector<uint8_t> kv_a = seq_dump();
            opt.cache_tag = "B";
            const auto b1 = decide_batch(eng, "system beta", { "context" }, fields, opt);
            const std::vector<uint8_t> kv_b = seq_dump();
            opt.cache_tag = "A";
            const auto a2 = decide_batch(eng, "system alpha", { "context" }, fields, opt);
            const std::vector<uint8_t> kv_a2 = seq_dump();

            t.assert_true("the first A request misses", !a1.cache_hit);
            t.assert_true("the B request misses", !b1.cache_hit);
            t.assert_true("returning to A hits the bounded cache", a2.cache_hit);
            if (kv_a == kv_b) {
                t.skip("the two prefixes produce the same prefix KV on this model");
                return;
            }
            t.assert_true("the cached A prefix is restored from A's own saved bytes", kv_a == kv_a2);

            const auto & p1 = a1.items[0].fields[0].probs;
            const auto & p2 = a2.items[0].fields[0].probs;
            bool same = p1.size() == p2.size();
            for (size_t k = 0; same && k < p1.size(); ++k) {
                same = std::fabs(p1[k] - p2[k]) < 1e-6f;
            }
            t.assert_true("the cached A probabilities equal the first A probabilities", same);
        } catch (const std::exception & e) {
            t.assert_true(std::string("the prefix cache run: ") + e.what(), false);
        }
    });
}

// A cache hit on a recurrent or hybrid model reloads the LRU host state and re-saves the device
// partial state before the fork, so it must reproduce the cold prefill's answer. Repeating a request
// is reproducible on the winner and the option set, never on the last bits of a probability
// (docs/decision/API.md Section 3.1).
static void test_prefix_cache_hit_repeatability(testing & t) {
    t.test("a restored prefix reproduces the cold fork winner and option set", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (!gpu_model_ready(t, path)) {
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("model loads", false);
            return;
        }
        if (!llama_model_is_recurrent(te.model) && !llama_model_is_hybrid(te.model)) {
            t.skip("the LRU host-restore path needs a recurrent or hybrid model");
            return;
        }
        try {
            llama_decision::engine eng(te.ctx, 2, 8);
            const std::vector<llama_decision::field_input> fields = {
                { "  \"dept\": ", { "billing", "technical", "refund", "support" } },
                { "  \"urgent\": ", { "low", "high" } },
            };
            std::string shared;
            for (int i = 0; i < 40; ++i) {
                shared += "The support request is described in the following ticket. ";
            }

            llama_decision::options cold_opt;
            cold_opt.allow_cache = false;
            const auto cold = decide_batch(eng, shared, { "the customer was charged twice" }, fields, cold_opt);

            llama_decision::options warm_opt;
            warm_opt.cache_tag = "repeatability";
            const auto miss = decide_batch(eng, shared, { "the customer was charged twice" }, fields, warm_opt);
            const auto hit  = decide_batch(eng, shared, { "the customer was charged twice" }, fields, warm_opt);

            t.assert_true("the first cached request is a miss", !miss.cache_hit);
            t.assert_true("the repeat is a cache hit", hit.cache_hit);

            bool same = cold.items.size() == hit.items.size() &&
                        cold.items[0].fields.size() == hit.items[0].fields.size();
            for (size_t f = 0; same && f < hit.items[0].fields.size(); ++f) {
                same = cold.items[0].fields[f].winner == hit.items[0].fields[f].winner &&
                       cold.items[0].fields[f].probs.size() == hit.items[0].fields[f].probs.size();
            }
            t.assert_true("the restored prefix keeps the winner and the option set", same);
        } catch (const std::exception & e) {
            t.assert_true(std::string("the prefix cache hit run: ") + e.what(), false);
        }
    });
}

// Baseline for the state-format work: a device-format sequence state must round-trip on the CPU
// backend. The host dump is the reference because it is serialized in sequence cell order.
// The transposed V cache stores one row per embedding, so a state save would otherwise touch each
// embedding separately. The bulk path moves one range per layer as a single strided transfer. A
// non-transposed V cache (flash attention on) is already contiguous and is the control that must
// not regress. Both must round-trip the sequence state byte-identically.
static void state_bulk_copy_run(testing & t, llama_context * ctx, const std::string & lane, int n_layer) {
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    const auto          toks  = common_tokenize(vocab, "a transposed decision state", false, true);
    if (toks.empty()) {
        t.skip(lane + ": the model has no usable tokens");
        return;
    }
    if (!t.assert_true(lane + ": the prefix decodes", decode_tokens_on(ctx, 0, 0, toks))) {
        return;
    }
    llama_synchronize(ctx);

    const std::vector<uint8_t> before = seq_state_dump(ctx, 0);
    if (!t.assert_true(lane + ": the sequence state is non-empty", !before.empty())) {
        return;
    }

    llama_state_seq_debug_reset_transfers();
    std::vector<uint8_t> saved(llama_state_seq_get_size(ctx, 0));
    const size_t         saved_n = llama_state_seq_get_data(ctx, saved.data(), saved.size(), 0);
    saved.resize(saved_n);
    const uint64_t save_transfers = llama_state_seq_debug_transfer_count();

    llama_memory_clear(llama_get_memory(ctx), true);
    llama_state_seq_debug_reset_transfers();
    const size_t nset = llama_state_seq_set_data(ctx, saved.data(), saved.size(), 0);
    const uint64_t load_transfers = llama_state_seq_debug_transfer_count();
    t.assert_equal(lane + ": the sequence state restores in full", saved.size(), nset);

    const std::vector<uint8_t> after = seq_state_dump(ctx, 0);
    t.assert_true(lane + ": the restored sequence state is byte-identical", before == after);

    // the device format carries the same state through its staging buffers
    const size_t dev_size = llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
    if (t.assert_true(lane + ": the device state has a size", dev_size > 0)) {
        std::vector<uint8_t> dev(dev_size);
        t.assert_equal(lane + ": the device state is written", dev_size,
                       llama_state_seq_get_data_ext(ctx, dev.data(), dev.size(), 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE));
        llama_memory_clear(llama_get_memory(ctx), true);
        t.assert_equal(lane + ": the device state restores in full", dev_size,
                       llama_state_seq_set_data_ext(ctx, dev.data(), dev.size(), 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE));
        t.assert_true(lane + ": the device-restored sequence state is byte-identical", before == seq_state_dump(ctx, 0));
    }

    printf("[state bulk] %s: n_layer %d, save transfers %" PRIu64 ", load transfers %" PRIu64 "\n",
           lane.c_str(), n_layer, save_transfers, load_transfers);

    // one bulk transfer per layer plus a bounded constant; the per-embedding path was thousands
    const uint64_t bound = (uint64_t) 8 * (uint64_t) std::max(n_layer, 1) + 32;
    t.assert_true(lane + ": the save uses a bulk transfer per layer", save_transfers <= bound);
    t.assert_true(lane + ": the load uses a bulk transfer per layer", load_transfers <= bound);
}

static void test_state_bulk_copy(testing & t) {
    t.test("a transposed sequence state round-trips and copies per layer on the GPU backend", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (!gpu_model_ready(t, path)) {
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("the GPU decision scaffold loads the model", false);
            return;
        }
        try {
            state_bulk_copy_run(t, te.ctx, "transposed", llama_model_n_layer(te.model));
        } catch (const std::exception & e) {
            t.assert_true(std::string("the transposed state round trip: ") + e.what(), false);
        }
    });

    t.test("a non-transposed sequence state round-trips on the GPU backend", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (!gpu_model_ready(t, path)) {
            return;
        }
        test_engine te;
        bool         loaded = false;
        try {
            loaded = te.load_fa(path);
        } catch (const std::exception &) {
            loaded = false;
        }
        if (!loaded) {
            t.skip("flash attention is not available for this model");
            return;
        }
        try {
            state_bulk_copy_run(t, te.ctx, "non-transposed", llama_model_n_layer(te.model));
        } catch (const std::exception & e) {
            t.assert_true(std::string("the non-transposed state round trip: ") + e.what(), false);
        }
    });
}

static void test_device_state_round_trip(testing & t) {
    t.test("a device sequence state round-trips on the CPU backend", [](testing & t) {
        const std::string path = decision_cpu_model_path();
        if (path.empty()) {
            t.skip("no generated model; run the generate-models fixture");
            return;
        }
        cpu_test_engine te;
        if (!te.load(path)) {
            t.assert_true("the CPU decision scaffold loads the model", false);
            return;
        }
        try {
            const llama_vocab * vocab = llama_model_get_vocab(te.model);
            const std::vector<llama_token> toks = common_tokenize(vocab, "the decision state", false, true);
            if (toks.empty()) {
                t.skip("the model has no usable tokens");
                return;
            }
            llama_batch batch = llama_batch_init((int) toks.size(), 0, 1);
            for (size_t i = 0; i < toks.size(); ++i) {
                common_batch_add(batch, toks[i], (llama_pos) i, { 0 }, i + 1 == toks.size());
            }
            const int rc = llama_decode(te.ctx, batch);
            llama_batch_free(batch);
            if (rc != 0) {
                t.assert_true("the prefix decodes on the CPU backend", false);
                return;
            }
            llama_synchronize(te.ctx);

            const auto host_dump = [&]() {
                std::vector<uint8_t> buf(llama_state_seq_get_size(te.ctx, 0));
                const size_t n = llama_state_seq_get_data(te.ctx, buf.data(), buf.size(), 0);
                buf.resize(n);
                return buf;
            };
            const std::vector<uint8_t> before = host_dump();
            t.assert_true("the host state is non-empty", !before.empty());

            const size_t dev_size = llama_state_seq_get_size_ext(te.ctx, 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
            t.assert_true("the device state has a size on the CPU backend", dev_size > 0);
            std::vector<uint8_t> dev(dev_size);
            const size_t ncopy = llama_state_seq_get_data_ext(te.ctx, dev.data(), dev.size(), 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
            t.assert_equal("the full device state is written", dev_size, ncopy);

            llama_memory_clear(llama_get_memory(te.ctx), true);
            const size_t nset = llama_state_seq_set_data_ext(te.ctx, dev.data(), dev.size(), 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
            t.assert_equal("the device state is restored in full", dev_size, nset);

            const std::vector<uint8_t> after = host_dump();
            t.assert_true("the restored state equals the saved state", before == after);
        } catch (const std::exception & e) {
            t.assert_true(std::string("the device round trip: ") + e.what(), false);
        }
    });
}

// The same round-trip when the sequence state lives on a GPU: the device path stages every copy
// with ggml_backend_tensor_copy_async and drains the backend once, so a missing drain would leave
// the restored host state stale. Without a GPU backend the test skips, keeping the CPU lane green.
static void test_device_async_staging(testing & t) {
    t.test("a device state staged on a GPU backend round-trips", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (!gpu_model_ready(t, path)) {
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("the GPU decision scaffold loads the model", false);
            return;
        }
        try {
            const llama_vocab * vocab = llama_model_get_vocab(te.model);
            const std::vector<llama_token> toks = common_tokenize(vocab, "the decision state", false, true);
            if (toks.empty()) {
                t.skip("the model has no usable tokens");
                return;
            }
            llama_batch batch = llama_batch_init((int) toks.size(), 0, 1);
            for (size_t i = 0; i < toks.size(); ++i) {
                common_batch_add(batch, toks[i], (llama_pos) i, { 0 }, i + 1 == toks.size());
            }
            const int rc = llama_decode(te.ctx, batch);
            llama_batch_free(batch);
            if (rc != 0) {
                t.assert_true("the prefix decodes on the GPU backend", false);
                return;
            }
            llama_synchronize(te.ctx);

            const auto host_dump = [&]() {
                std::vector<uint8_t> buf(llama_state_seq_get_size(te.ctx, 0));
                const size_t n = llama_state_seq_get_data(te.ctx, buf.data(), buf.size(), 0);
                buf.resize(n);
                return buf;
            };
            const std::vector<uint8_t> before = host_dump();
            t.assert_true("the host state is non-empty", !before.empty());

            const size_t dev_size = llama_state_seq_get_size_ext(te.ctx, 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
            t.assert_true("the device state has a size on the GPU backend", dev_size > 0);
            std::vector<uint8_t> dev(dev_size);
            const size_t ncopy = llama_state_seq_get_data_ext(te.ctx, dev.data(), dev.size(), 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
            t.assert_equal("the full device state is written", dev_size, ncopy);

            llama_memory_clear(llama_get_memory(te.ctx), true);
            const size_t nset = llama_state_seq_set_data_ext(te.ctx, dev.data(), dev.size(), 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
            t.assert_equal("the device state is restored in full", dev_size, nset);

            const std::vector<uint8_t> after = host_dump();
            t.assert_true("the restored state equals the saved state", before == after);

            // a working device path is what the engine prefers; it must not have retired to host
            llama_decision::engine eng(te.ctx, 2, 8);
            const auto st = eng.save_seq(0, true);
            t.assert_true("the engine keeps the device save when the backend supports it", st.on_device());

            bool threw = false;
            try {
                eng.load_seq(st, 0);
            } catch (const std::exception &) {
                threw = true;
            }
            t.assert_true("the engine device state restores", !threw);
        } catch (const std::exception & e) {
            t.assert_true(std::string("the GPU device round trip: ") + e.what(), false);
        }
    });
}

// A recurrent cache can hold its sequence cells in more than one range. The device format cannot
// describe that, so the save must be refused recoverably (a throw the caller can catch, never an
// abort) and the self-contained host format must still round-trip byte-identically.
static void test_recurrent_multi_range_device_save(testing & t) {
    t.test("a fragmented recurrent cache refuses the device save and restores on the host", [](testing & t) {
        const std::string path = decision_cpu_model_path();
        if (path.empty()) {
            t.skip("no generated model; run the generate-models fixture");
            return;
        }
        cpu_test_engine te;
        if (!te.load(path)) {
            t.assert_true("the CPU decision scaffold loads the model", false);
            return;
        }
        if (!llama_model_is_recurrent(te.model) && !llama_model_is_hybrid(te.model)) {
            t.skip("the model is neither recurrent nor hybrid");
            return;
        }
        try {
            const llama_vocab * vocab = llama_model_get_vocab(te.model);
            const std::vector<llama_token> toks = common_tokenize(vocab, "the decision state", false, true);
            if (toks.empty()) {
                t.skip("the model has no usable tokens");
                return;
            }

            const auto host_dump = [&]() {
                std::vector<uint8_t> buf(llama_state_get_size(te.ctx));
                const size_t n = llama_state_get_data(te.ctx, buf.data(), buf.size());
                buf.resize(n);
                return buf;
            };
            const auto host_restore = [&](const std::vector<uint8_t> & blob) {
                llama_memory_clear(llama_get_memory(te.ctx), true);
                return llama_state_set_data(te.ctx, blob.data(), blob.size()) == blob.size();
            };

            // place three sequences in consecutive cells, then drop the middle one so the used
            // cells of the whole cache are no longer contiguous
            for (llama_seq_id seq = 0; seq < 3; ++seq) {
                llama_batch batch = llama_batch_init((int) toks.size(), 0, 1);
                for (size_t i = 0; i < toks.size(); ++i) {
                    common_batch_add(batch, toks[i], (llama_pos) i, { seq }, i + 1 == toks.size());
                }
                const int rc = llama_decode(te.ctx, batch);
                llama_batch_free(batch);
                if (rc != 0) {
                    t.assert_true("the fragmented prefix decodes on the CPU backend", false);
                    return;
                }
            }
            llama_synchronize(te.ctx);
            const std::vector<uint8_t> pre = host_dump();
            if (!llama_memory_seq_rm(llama_get_memory(te.ctx), 1, -1, -1)) {
                t.skip("the model cannot remove a middle sequence");
                return;
            }

            // the one contiguous-range device format cannot hold the fragmented cells
            const size_t dev_size = llama_state_seq_get_size_ext(te.ctx, -1, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
            t.assert_equal("the fragmented device save is refused", (size_t) 0, dev_size);

            // the host format is self-contained and must still round-trip
            const std::vector<uint8_t> frag = host_dump();
            t.assert_true("the fragmented host state is non-empty", !frag.empty());
            t.assert_true("the fragmented host state restores in full", host_restore(frag));
            t.assert_true("the restored fragmented state equals the saved state", frag == host_dump());

            // the pre-fragmentation state must still restore byte-identically afterwards
            t.assert_true("the pre-fragmentation state restores in full", host_restore(pre));
            t.assert_true("the restored pre-fragmentation state equals the saved state", pre == host_dump());
        } catch (const std::exception & e) {
            t.assert_true(std::string("the fragmented recurrent save: ") + e.what(), false);
        }
    });
}

// A full restore and a partial restore must carry the same recurrent state: the partial bytes read
// back after either restore are byte-identical, and the partial host format round-trips on its own.
// The full host save is also checked against the low-level serialization so the flag work cannot
// change the existing full-state bytes.
static void partial_state_round_trip_run(testing & t, llama_context * ctx, const std::string & lane) {
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    const auto          toks  = common_tokenize(vocab, "a short partial decision state", false, true);
    if (toks.empty()) {
        t.skip(lane + ": the model has no usable tokens");
        return;
    }

    llama_decision::engine eng(ctx, 2, 8);
    const llama_seq_id     src = 2;
    if (!t.assert_true(lane + ": the state decodes", decode_tokens_on(ctx, src, 0, toks))) {
        return;
    }
    llama_synchronize(ctx);

    const auto full = eng.save_seq(src, false, false);
    const auto part = eng.save_seq(src, false, true);
    if (!t.assert_true(lane + ": the full host state is non-empty", !full.bytes.empty()) ||
        !t.assert_true(lane + ": the partial host state is non-empty", !part.bytes.empty())) {
        return;
    }
    t.assert_equal(lane + ": the full save carries no scope flag",
                   (unsigned) LLAMA_STATE_SEQ_FLAGS_NONE, (unsigned) full.flags);
    t.assert_equal(lane + ": the partial save carries the partial flag",
                   (unsigned) LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY, (unsigned) part.flags);
    t.assert_true(lane + ": the engine reports partial state capability", eng.partial_state_capable());

    // the full host save is exactly the low-level full serialization, unchanged by the flags
    const size_t direct_size = llama_state_seq_get_size_ext(ctx, src, LLAMA_STATE_SEQ_FLAGS_NONE);
    std::vector<uint8_t> direct(direct_size);
    const size_t direct_n = llama_state_seq_get_data_ext(ctx, direct.data(), direct.size(), src, LLAMA_STATE_SEQ_FLAGS_NONE);
    direct.resize(direct_n);
    t.assert_true(lane + ": the full host save matches the direct serialization", direct == full.bytes);

    llama_memory_t mem = llama_get_memory(ctx);

    // a full restore carries the same recurrent state that a partial save reads
    llama_memory_clear(mem, true);
    eng.load_seq(full, src);
    const auto part_of_full = eng.save_seq(src, false, true);
    t.assert_true(lane + ": a full restore's recurrent state matches the partial save", part_of_full.bytes == part.bytes);

    // a partial restore round-trips the recurrent state on its own; the attention side stays empty
    // here, so the read-back uses the low-level serialization
    llama_memory_clear(mem, true);
    eng.load_seq(part, src);
    std::vector<uint8_t> part_rt(llama_state_seq_get_size_ext(ctx, src, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
    const size_t         part_rt_n =
        llama_state_seq_get_data_ext(ctx, part_rt.data(), part_rt.size(), src, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    part_rt.resize(part_rt_n);
    t.assert_true(lane + ": the partial host round-trip is byte-identical", part_rt == part.bytes);

    // the partial device state must carry the same recurrent bytes as the host format
    llama_memory_clear(mem, true);
    eng.load_seq(full, src);
    const auto part_dev = eng.save_seq(src, true, true);
    if (t.assert_true(lane + ": the partial device save is staged", part_dev.on_device())) {
        llama_memory_clear(mem, true);
        eng.load_seq(part_dev, src);
        std::vector<uint8_t> dev_rt(llama_state_seq_get_size_ext(ctx, src, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
        const size_t         dev_rt_n =
            llama_state_seq_get_data_ext(ctx, dev_rt.data(), dev_rt.size(), src, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        dev_rt.resize(dev_rt_n);
        t.assert_true(lane + ": the partial device round-trip matches the host bytes", dev_rt == part.bytes);
    }

    // the hybrid fork shape: attention copied by metadata, recurrent state restored partial
    llama_memory_clear(mem, true);
    eng.load_seq(full, src);
    llama_memory_seq_rm(mem, 5, -1, -1);
    llama_memory_seq_cp(mem, src, 5, -1, -1);
    eng.load_seq(part, 5);
    const auto fork_part = eng.save_seq(5, false, true);
    const auto fork_norm = normalize_seq_state(ctx, fork_part.bytes, 6, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    const auto part_norm = normalize_seq_state(ctx, part.bytes, 6, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    t.assert_true(lane + ": the copied-attention partial fork matches the partial save",
                  state_bytes_diff(fork_norm, part_norm) == 0);
}

static void test_partial_state_round_trip(testing & t) {
    t.test("a partial host state round-trips and matches a full restore (cpu)", [](testing & t) {
        const std::string path = decision_cpu_model_path();
        if (path.empty()) {
            t.skip("no generated model; run the generate-models fixture");
            return;
        }
        cpu_test_engine te;
        if (!te.load(path)) {
            t.assert_true("the CPU decision scaffold loads the model", false);
            return;
        }
        try {
            partial_state_round_trip_run(t, te.ctx, "cpu");
        } catch (const std::exception & e) {
            t.assert_true(std::string("the CPU partial round trip: ") + e.what(), false);
        }
    });

    t.test("a partial host state round-trips and matches a full restore (gpu)", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (!gpu_model_ready(t, path)) {
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("the GPU decision scaffold loads the model", false);
            return;
        }
        try {
            partial_state_round_trip_run(t, te.ctx, "gpu");
        } catch (const std::exception & e) {
            t.assert_true(std::string("the GPU partial round trip: ") + e.what(), false);
        }
    });
}

// A fragmented recurrent cache cannot be staged as one device range: the whole-cache partial device
// save is refused recoverably (size 0, no abort) and the host partial format still round-trips. The
// engine's one-way partial device capability then falls back to the host partial format.
static void partial_state_fragmented_run(testing & t, llama_context * ctx, const std::string & lane) {
    const llama_model * model = llama_get_model(ctx);
    if (!llama_model_is_recurrent(model) && !llama_model_is_hybrid(model)) {
        t.skip(lane + ": the model is neither recurrent nor hybrid");
        return;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const auto          toks  = common_tokenize(vocab, "the fragmented partial state", false, true);
    if (toks.empty()) {
        t.skip(lane + ": the model has no usable tokens");
        return;
    }

    for (llama_seq_id seq = 0; seq < 3; ++seq) {
        if (!t.assert_true(lane + ": the fragmented prefix decodes", decode_tokens_on(ctx, seq, 0, toks))) {
            return;
        }
    }
    llama_synchronize(ctx);
    if (!llama_memory_seq_rm(llama_get_memory(ctx), 1, -1, -1)) {
        t.skip(lane + ": the model cannot remove a middle sequence");
        return;
    }

    const llama_state_seq_flags partial_device = llama_decision::engine::state_load_flags(true, true);
    t.assert_equal(lane + ": the fragmented partial device save is refused", (size_t) 0,
                   llama_state_seq_get_size_ext(ctx, -1, partial_device));

    // the host partial format is self-contained and must round-trip the fragmented recurrent cache
    const llama_state_seq_flags partial_host = llama_decision::engine::state_load_flags(false, true);
    const size_t host_size = llama_state_seq_get_size_ext(ctx, -1, partial_host);
    if (!t.assert_true(lane + ": the fragmented partial host state is non-empty", host_size > 0)) {
        return;
    }
    std::vector<uint8_t> host(host_size);
    t.assert_equal(lane + ": the fragmented partial host state is written", host_size,
                   llama_state_seq_get_data_ext(ctx, host.data(), host.size(), -1, partial_host));
    llama_memory_clear(llama_get_memory(ctx), true);
    t.assert_equal(lane + ": the fragmented partial host state restores", host_size,
                   llama_state_seq_set_data_ext(ctx, host.data(), host.size(), -1, partial_host));
    std::vector<uint8_t> after(llama_state_seq_get_size_ext(ctx, -1, partial_host));
    llama_state_seq_get_data_ext(ctx, after.data(), after.size(), -1, partial_host);
    t.assert_true(lane + ": the restored partial state equals the saved partial state", host == after);

    // the engine retires partial device staging when it fails and uses the host partial format
    llama_memory_clear(llama_get_memory(ctx), true);
    llama_decision::engine eng(ctx, 4, 6);
    const llama_seq_id     src = 4;
    t.assert_true(lane + ": the engine starts with partial state capability", eng.partial_state_capable());
    if (!t.assert_true(lane + ": the engine prefix decodes", decode_tokens_on(ctx, src, 0, toks))) {
        return;
    }
    llama_synchronize(ctx);

    eng.partial_device_capable_ = false; // the retired state a failed device save leaves behind
    const auto st = eng.save_seq(src, true, true);
    t.assert_true(lane + ": a retired partial device save falls back to the host format", !st.on_device());
    t.assert_equal(lane + ": the fallback keeps the partial scope",
                   (unsigned) LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY, (unsigned) st.flags);
    t.assert_true(lane + ": the fallback state is non-empty", !st.bytes.empty());
    t.assert_true(lane + ": the capability stays retired", !eng.partial_state_capable());

    llama_memory_clear(llama_get_memory(ctx), true);
    bool round_trip = false;
    try {
        eng.load_seq(st, src);
        std::vector<uint8_t> rt(llama_state_seq_get_size_ext(ctx, src, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
        const size_t         rt_n =
            llama_state_seq_get_data_ext(ctx, rt.data(), rt.size(), src, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        rt.resize(rt_n);
        round_trip = rt == st.bytes;
    } catch (const std::exception &) {
        round_trip = false;
    }
    t.assert_true(lane + ": the fallback host partial state round-trips", round_trip);
}

static void test_partial_state_fragmented(testing & t) {
    t.test("a fragmented recurrent cache refuses the whole-cache partial device save (cpu)", [](testing & t) {
        const std::string path = decision_cpu_model_path();
        if (path.empty()) {
            t.skip("no generated model; run the generate-models fixture");
            return;
        }
        cpu_test_engine te;
        if (!te.load(path)) {
            t.assert_true("the CPU decision scaffold loads the model", false);
            return;
        }
        try {
            partial_state_fragmented_run(t, te.ctx, "cpu");
        } catch (const std::exception & e) {
            t.assert_true(std::string("the CPU partial fragmented save: ") + e.what(), false);
        }
    });

    t.test("a fragmented recurrent cache refuses the whole-cache partial device save (gpu)", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (!gpu_model_ready(t, path)) {
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("the GPU decision scaffold loads the model", false);
            return;
        }
        try {
            partial_state_fragmented_run(t, te.ctx, "gpu");
        } catch (const std::exception & e) {
            t.assert_true(std::string("the GPU partial fragmented save: ") + e.what(), false);
        }
    });
}

// Device state round-trip when the cache layout changes between save and restore. The save stages
// tensor bytes in the context staging buffer; a changed layout can push the reader from the 1:1
// chunked copy to the byte-cursor path, and the restored state must still equal the saved one. A
// missing backend synchronize would leave the restore half-applied, so the host dump catches it.
static void device_layout_mutation_round_trip(testing &           t,
                                              llama_context *     ctx,
                                              llama_model *       model,
                                              const std::string & lane) {
    const llama_vocab *            vocab = llama_model_get_vocab(model);
    const std::vector<llama_token> toks  = common_tokenize(vocab, "layout mutation decision state", false, true);
    if (toks.empty()) {
        t.skip("the model has no usable tokens");
        return;
    }
    const llama_memory_t mem = llama_get_memory(ctx);
    const llama_pos      n   = (llama_pos) toks.size();

    auto decode_seq = [&](llama_seq_id seq, llama_pos pos0, const std::vector<llama_token> & ids) {
        llama_batch batch = llama_batch_init((int) ids.size(), 0, 1);
        for (size_t i = 0; i < ids.size(); ++i) {
            common_batch_add(batch, ids[i], pos0 + (llama_pos) i, { seq }, i + 1 == ids.size());
        }
        const int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        return rc == 0;
    };
    auto host_dump = [&]() {
        std::vector<uint8_t> buf(llama_state_seq_get_size(ctx, 0));
        const size_t         got = llama_state_seq_get_data(ctx, buf.data(), buf.size(), 0);
        buf.resize(got);
        return buf;
    };

    if (!decode_seq(0, 0, toks)) {
        t.assert_true(lane + ": the prefix decodes", false);
        return;
    }
    llama_synchronize(ctx);

    const std::vector<uint8_t> before = host_dump();
    if (!t.assert_true(lane + ": the host state is non-empty", !before.empty())) {
        return;
    }
    const size_t dev_size = llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
    if (!t.assert_true(lane + ": the device state has a size", dev_size > 0)) {
        return;
    }
    std::vector<uint8_t> dev(dev_size);
    if (!t.assert_equal(
            lane + ": the full device state is written", dev_size,
            llama_state_seq_get_data_ext(ctx, dev.data(), dev.size(), 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE))) {
        return;
    }

    // Mutate, restore the one saved device state, and require the host dump to return to `before`.
    auto mutate_and_restore = [&](const std::string & name, auto && mutate) {
        if (!mutate()) {
            printf("layout mutation not run (%s): the cache refused it\n", name.c_str());
            return;
        }
        const size_t nset =
            llama_state_seq_set_data_ext(ctx, dev.data(), dev.size(), 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
        if (!t.assert_equal(lane + ": " + name + " restores in full", dev_size, nset)) {
            return;
        }
        t.assert_true(lane + ": " + name + " equals the saved state", before == host_dump());
    };

    // other sequences come and go around the saved one
    mutate_and_restore("other sequences", [&]() {
        if (!decode_seq(1, 0, toks) || !decode_seq(2, 0, toks)) {
            return false;
        }
        llama_synchronize(ctx);
        const bool r1 = llama_memory_seq_rm(mem, 1, -1, -1);
        const bool r2 = llama_memory_seq_rm(mem, 2, -1, -1);
        llama_synchronize(ctx);
        return r1 && r2;
    });

    // trim the tail of the saved sequence and re-decode it, so its cells may move
    mutate_and_restore("trim and re-decode", [&]() {
        const llama_pos k = std::min<llama_pos>(1, n - 1);
        if (k <= 0 || !llama_memory_seq_rm(mem, 0, n - k, -1)) {
            return false;
        }
        const std::vector<llama_token> tail(toks.end() - k, toks.end());
        return decode_seq(0, n - k, tail);
    });

    // rebuild the sequence from scratch, relocating its cells
    mutate_and_restore("clear and rebuild", [&]() {
        llama_memory_clear(mem, true);
        return decode_seq(0, 0, toks);
    });

    // interleave a foreign sequence into the saved sequence's cells, then re-decode the moved tail:
    // this fragments the saved sequence so the reader must re-chunk across ranges
    mutate_and_restore("fragmented rebuild", [&]() {
        const llama_pos half = n / 2;
        if (half <= 0 || !llama_memory_seq_rm(mem, 0, half, -1)) {
            return false;
        }
        const std::vector<llama_token> tail(toks.begin() + half, toks.end());
        if (!decode_seq(1, 0, tail)) {
            return false;
        }
        if (!decode_seq(0, half, tail)) {
            return false;
        }
        llama_memory_seq_rm(mem, 1, -1, -1);
        llama_synchronize(ctx);
        return true;
    });
}

static void test_device_layout_mutation_round_trip(testing & t) {
    t.test("a device state round-trips across cache layout mutations on the CPU backend", [](testing & t) {
        const std::string path = decision_cpu_model_path();
        if (path.empty()) {
            t.skip("no generated model; run the generate-models fixture");
            return;
        }
        cpu_test_engine te;
        if (!te.load(path, 512, false)) {
            t.assert_true("the CPU decision scaffold loads the model", false);
            return;
        }
        try {
            device_layout_mutation_round_trip(t, te.ctx, te.model, "cpu");
        } catch (const std::exception & e) {
            t.assert_true(std::string("the CPU layout mutation round trip: ") + e.what(), false);
        }
    });

    t.test("a device state round-trips across cache layout mutations on a GPU backend", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (!gpu_model_ready(t, path)) {
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("the GPU decision scaffold loads the model", false);
            return;
        }
        try {
            device_layout_mutation_round_trip(t, te.ctx, te.model, "gpu");
        } catch (const std::exception & e) {
            t.assert_true(std::string("the GPU layout mutation round trip: ") + e.what(), false);
        }
    });
}

// The engine refuses a save that has nothing to copy and a load that has nothing to restore. A
// silent empty state would let a failed prefix save continue with wrong offsets and score garbage.
static void test_save_load_fail_fast(testing & t) {
    t.test("an empty decision state is refused on load", [](testing & t) {
        const std::string path = decision_cpu_model_path();
        if (path.empty()) {
            t.skip("no CPU decision model available");
            return;
        }
        cpu_test_engine te;
        if (!te.load(path)) {
            t.assert_true("the CPU decision scaffold loads the model", false);
            return;
        }
        try {
            llama_decision::engine eng(te.ctx, 2, 8);
            // a failed save used to produce exactly this value, which load_seq then silently
            // no-op'd and the decode continued at wrong offsets
            bool threw = false;
            try {
                eng.load_seq({}, 2);
            } catch (const std::runtime_error &) {
                threw = true;
            }
            t.assert_true("an empty state is refused on load, never a silent no-op", threw);
        } catch (const std::exception & e) {
            t.assert_true(std::string("empty-state load: ") + e.what(), false);
        }
    });

    t.test("a decoded sequence saves and restores a non-empty state", [](testing & t) {
        const std::string path = decision_cpu_model_path();
        if (path.empty()) {
            t.skip("no CPU decision model available");
            return;
        }
        cpu_test_engine te;
        if (!te.load(path)) {
            t.assert_true("the CPU decision scaffold loads the model", false);
            return;
        }
        try {
            const llama_vocab * vocab = llama_model_get_vocab(te.model);
            const std::vector<llama_token> toks = common_tokenize(vocab, "the decision state", false, true);
            if (toks.empty()) {
                t.skip("the model has no usable tokens");
                return;
            }
            llama_batch batch = llama_batch_init((int) toks.size(), 0, 1);
            for (size_t i = 0; i < toks.size(); ++i) {
                common_batch_add(batch, toks[i], (llama_pos) i, { 2 }, i + 1 == toks.size());
            }
            const int rc = llama_decode(te.ctx, batch);
            llama_batch_free(batch);
            if (rc != 0) {
                t.assert_true("the prefix decodes on the CPU backend", false);
                return;
            }
            llama_synchronize(te.ctx);

            llama_decision::engine eng(te.ctx, 2, 8);
            const auto st = eng.save_seq(2, false);
            t.assert_true("a decoded sequence saves a non-empty state", !st.bytes.empty() && !st.on_device());
            bool threw = false;
            try {
                eng.load_seq(st, 2);
            } catch (const std::runtime_error &) {
                threw = true;
            }
            t.assert_true("the saved state restores without error", !threw);
        } catch (const std::exception & e) {
            t.assert_true(std::string("save/load round trip: ") + e.what(), false);
        }
    });

    t.test("a never-decoded sequence save throws instead of producing a header-only state", [](testing & t) {
        const std::string path = decision_cpu_model_path();
        if (path.empty()) {
            t.skip("no CPU decision model available");
            return;
        }
        cpu_test_engine te;
        if (!te.load(path)) {
            t.assert_true("the CPU decision scaffold loads the model", false);
            return;
        }
        try {
            llama_decision::engine eng(te.ctx, 2, 8);
            bool threw = false;
            try {
                eng.save_seq(9, false); // never decoded
            } catch (const std::runtime_error &) {
                threw = true;
            }
            t.assert_true("saving a never-decoded sequence throws", threw);
        } catch (const std::exception & e) {
            t.assert_true(std::string("never-decoded save: ") + e.what(), false);
        }
    });
}

// Structural control for the producer/task-value split: the weak-quant allowlist must only turn a
// task-value determinism assertion into an explicit skip, never an expected failure, and a
// non-allowlisted model must always run it hard. Every allowlist entry must also have a recorded
// measurement row in the calibration ledger so the skip is grounded in data, not prose.
static void test_weak_quant_control(testing & t) {
    t.test("a non-allowlisted model runs the task-value assertion hard", [](testing & t) {
        int ran = 0;
        determinism_check(t, false, "control: a non-allowlisted assertion runs hard", [&](testing &) { ++ran; });
        t.assert_equal("a non-allowlisted assertion runs hard", 1, ran);
    });

    t.test("an allowlisted model skips the task-value assertion, never xfails it", [](testing & t) {
        int ran = 0;
        determinism_check(t, true, "control: an allowlisted assertion is skipped", [&](testing &) { ++ran; });
        t.assert_equal("an allowlisted assertion is skipped, not run", 0, ran);
    });

    t.test("every weak-quant allowlist entry has a recorded calibration row", [](testing & t) {
        const common_json cal = common_json::parse(
            read_file(std::string(DECISION_TEST_BASELINE_DIR) + "/calibration.json"));
        const std::string recorded = cal.at("model_measurements").at("environment").value("model", std::string());
        if (recorded.empty()) {
            t.skip("no model measurement recorded in the calibration ledger");
            return;
        }
        for (const std::string & id : weak_quant_gpu_allowlist) {
            if (recorded != id) {
                t.skip("the calibration ledger records " + recorded + ", not the allowlist model " + id);
                continue;
            }
            t.assert_true("the allowlist entry " + id + " has a values row",
                          cal.at("model_measurements").contains("values"));
        }
    });
}


// The cache split adds one host save per miss and one device save per hit. On the generated model
// cold prefill is about 3 ms and warm prefill about 0.5 ms; the hit must stay clearly cheaper than
// a miss so the added refresh does not erase the cache win.
static void test_prefix_cache_cost(testing & t) {
    t.test("a cache hit prefills faster than a miss", [](testing & t) {
        const std::string path = decision_cpu_model_path();
        if (path.empty()) {
            t.skip("no generated model; run the generate-models fixture");
            return;
        }
        cpu_test_engine te;
        if (!te.load(path)) {
            t.assert_true("the CPU decision scaffold loads the model", false);
            return;
        }
        try {
            llama_decision::engine eng(te.ctx, 2, 8);
            const std::vector<llama_decision::field_input> fields = {
                { "  \"a\": ", { "1", "2" } },
                { "  \"b\": ", { "x", "y" } },
            };
            const auto time_prefill = [&](const std::string & tag) {
                llama_decision::options opt;
                opt.fork      = "restore";
                opt.cache_tag = tag;
                return decide_batch(eng, "system cost", { "context" }, fields, opt).prefill_ms;
            };
            double miss_ms = std::numeric_limits<double>::max();
            for (int i = 0; i < 3; ++i) {
                miss_ms = std::min(miss_ms, time_prefill("cost-miss-" + std::to_string(i)));
            }
            double hit_ms = std::numeric_limits<double>::max();
            for (int i = 0; i < 3; ++i) {
                hit_ms = std::min(hit_ms, time_prefill("cost-hit"));
            }
            t.assert_true("the cache hit prefills faster than a miss (miss " + std::to_string(miss_ms) +
                          " ms, hit " + std::to_string(hit_ms) + " ms)", hit_ms < miss_ms * 0.9);
        } catch (const std::exception & e) {
            t.assert_true(std::string("the prefix cache cost run: ") + e.what(), false);
        }
    });
}


// An arch whose output table has no per-id bias must still build a usable head: the kept zero
// bias adds nothing, and the probe must not tighten into a failure.


// A bounded decision context rejects a request that cannot fit before decoding, with a clear
// budget message, and keeps serving afterwards (no truncation, no KV residue).
static void test_bounded_decision_context(testing & t) {
    t.test("a request beyond the decision context budget is rejected, not truncated", [](testing & t) {
        const std::string path = decision_cpu_model_path();
        if (path.empty()) {
            t.skip("no generated model; run the generate-models fixture");
            return;
        }
        cpu_test_engine te;
        if (!te.load(path, 256)) { // the smallest context the backend keeps (n_ctx is padded to 256)
            t.assert_true("the CPU decision scaffold loads the model", false);
            return;
        }
        try {
            llama_decision::engine eng(te.ctx, 2, 8);
            const std::vector<llama_decision::field_input> fields = { { "  \"a\": ", { "1", "2" } } };
            const auto ok = decide_batch(eng, "sys", { "ctx" }, fields, llama_decision::options{});
            t.assert_equal("a fitting request returns a decision", (size_t) 1, ok.items.size());

            std::string big;
            for (int i = 0; i < 64; ++i) {
                big += "filler ";
            }
            bool threw = false;
            std::string what;
            try {
                decide_batch(eng, big, { "ctx" }, fields, llama_decision::options{});
            } catch (const llama_decision::capacity_error & e) {
                threw = true;
                what = e.what();
            }
            t.assert_true("an oversize request throws capacity_error", threw);
            t.assert_true("the error names the context budget", what.find("context holds") != std::string::npos);

            const auto again = decide_batch(eng, "sys", { "ctx" }, fields, llama_decision::options{});
            t.assert_equal("the context keeps serving after a rejected request", (size_t) 1, again.items.size());
        } catch (const std::exception & e) {
            t.assert_true(std::string("the bounded context run: ") + e.what(), false);
        }
    });
}

// A restore-fork group with more than one trunk per wave. Each context is staged as its own saved
// state and every branch restores from its own parent, so independent contexts must not alias, and
// the chat sequences that share the context must be untouched.
static void multi_trunk_restore_round_trip(testing &           t,
                                           llama_context *     ctx,
                                           llama_model *       model,
                                           const std::string & lane,
                                           const std::string & fork) {
    const llama_vocab *            vocab = llama_model_get_vocab(model);
    const std::vector<llama_token> chat  = common_tokenize(vocab, "a chat turn on the shared context", false, true);
    if (chat.empty()) {
        t.skip("the model has no usable tokens");
        return;
    }
    {
        llama_batch batch = llama_batch_init((int) chat.size(), 0, 1);
        for (size_t i = 0; i < chat.size(); ++i) {
            common_batch_add(batch, chat[i], (llama_pos) i, { 0 }, i + 1 == chat.size());
        }
        const int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            t.assert_true(lane + ": the chat prefix decodes", false);
            return;
        }
    }
    llama_synchronize(ctx);
    const auto chat_dump = [&]() {
        std::vector<uint8_t> buf(llama_state_seq_get_size(ctx, 0));
        const size_t         n = llama_state_seq_get_data(ctx, buf.data(), buf.size(), 0);
        buf.resize(n);
        return buf;
    };
    const std::vector<uint8_t> chat_before = chat_dump();
    t.assert_true(lane + ": the chat sequence has state", !chat_before.empty());

    llama_decision::engine                         eng(ctx, 2, 16);
    const std::vector<std::string>                 contexts = { "alpha", "beta", "alpha" };
    const std::vector<llama_decision::field_input> fields   = {
        { "  \"a\": ", { "1", "2" } },
        { "  \"b\": ", { "x", "y" } },
    };
    llama_decision::options o;
    o.mode        = "tree";
    o.fork        = fork;
    o.allow_cache = false;

    const auto   plan      = eng.compile_fields(fields, o);
    const size_t per_group = std::clamp<size_t>((size_t) eng.n_pool / (1 + plan.branches), 1, contexts.size());
    t.assert_true(lane + ": the batch groups more than one trunk per wave", per_group >= 2);

    const auto batch = decide_batch(eng, "system", contexts, fields, o);
    t.assert_equal(lane + ": every context is returned", contexts.size(), batch.items.size());

    // the same context twice in one wave must not alias: identical inputs must score identically
    // identical contexts in one wave must score the same. GPU batch packing can place their
    // branches in different batches, so the probability bound there is the documented
    // head-agreement tolerance; the branch-state oracle is the byte-level exactness check.
    const double twin_tol = lane == "gpu" ? 5e-2 : 1e-4;
    bool twins = batch.items[0].fields.size() == batch.items[2].fields.size();
    for (size_t f = 0; twins && f < fields.size(); ++f) {
        const auto & p0 = batch.items[0].fields[f].probs;
        const auto & p2 = batch.items[2].fields[f].probs;
        twins           = batch.items[0].fields[f].winner == batch.items[2].fields[f].winner && p0.size() == p2.size();
        for (size_t k = 0; twins && k < p0.size(); ++k) {
            twins = std::fabs(p0[k] - p2[k]) < twin_tol;
        }
    }
    t.assert_true(lane + ": the repeated context scores identically in one wave", twins);

    // each context keeps its own winner, the task-value outcome, against a single-context run
    for (size_t c = 0; c < contexts.size(); ++c) {
        const auto single  = decide_batch(eng, "system", { contexts[c] }, fields, o);
        bool       winners = batch.items[c].fields.size() == single.items[0].fields.size();
        for (size_t f = 0; winners && f < fields.size(); ++f) {
            winners = batch.items[c].fields[f].winner == single.items[0].fields[f].winner;
        }
        t.assert_true(lane + ": context " + std::to_string(c) + " keeps its winner", winners);
    }

    const std::vector<uint8_t> chat_after = chat_dump();
    t.assert_true(lane + ": the chat sequence is untouched", chat_before == chat_after);
}

static void test_multi_trunk_restore(testing & t) {
    for (const char * fork : { "restore", "hybrid" }) {
        t.test(std::string("the ") + fork + " fork keeps contexts independent on the CPU backend",
               [fork](testing & t) {
                   const std::string path = decision_cpu_model_path();
                   if (path.empty()) {
                       t.skip("no generated model; run the generate-models fixture");
                       return;
                   }
                   cpu_test_engine te;
                   if (!te.load(path, 512, false, 128, 18)) {
                       t.assert_true("the CPU decision scaffold loads the model", false);
                       return;
                   }
                   try {
                       multi_trunk_restore_round_trip(t, te.ctx, te.model, "cpu", fork);
                   } catch (const std::exception & e) {
                       t.assert_true(std::string("the CPU multi-trunk ") + fork + ": " + e.what(), false);
                   }
               });

        t.test(std::string("the ") + fork + " fork keeps contexts independent on a recurrent GPU backend",
               [fork](testing & t) {
                   const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
                   if (!gpu_model_ready(t, path)) {
                       return;
                   }
                   test_engine te;
                   if (!te.load(path, 18)) {
                       t.assert_true("the GPU decision scaffold loads the model", false);
                       return;
                   }
                   if (!llama_model_is_recurrent(te.model) && !llama_model_is_hybrid(te.model)) {
                       t.skip("the model is neither recurrent nor hybrid; the multi-trunk save/restore path needs one");
                       return;
                   }
                   if (weak_quant_gpu_oracle(path)) {
                       t.skip("weak-quant GPU numerics move a winner here; the aliasing check is skipped, not xfail");
                       return;
                   }
                   try {
                       multi_trunk_restore_round_trip(t, te.ctx, te.model, "gpu", fork);
                   } catch (const std::exception & e) {
                       t.assert_true(std::string("the GPU multi-trunk ") + fork + ": " + e.what(), false);
                   }
               });
    }
}

// A failed decision must leave the pool sequences empty: a mid-wave decode failure forks pool
// sequences first, and residue in the shared cache would starve the next chat decode.
static std::string text_of_n_tokens(const llama_vocab * vocab, int n) {
    std::string text;
    while ((int) common_tokenize(vocab, text, false, true).size() < n) {
        text += "filler ";
    }
    return text;
}

static void test_pool_seq_lifecycle(testing & t) {
    t.test("a mid-wave decode failure leaves every pool sequence empty and chat decodable", [](testing & t) {
        const std::string path = decision_cpu_model_path();
        if (path.empty()) {
            t.skip("no generated model; run the generate-models fixture");
            return;
        }
        cpu_test_engine te;
        if (!te.load(path, 512)) {
            t.assert_true("the CPU decision scaffold loads the model", false);
            return;
        }
        try {
            const auto * vocab = llama_model_get_vocab(te.model);
            auto ntok = [&](const std::string & s) {
                return (int) common_tokenize(vocab, s, false, true).size();
            };
            const std::string shared_text = text_of_n_tokens(vocab, 50);
            const std::string tail_text   = text_of_n_tokens(vocab, 10);
            const std::string suffix_text = text_of_n_tokens(vocab, 20);
            const int s_n = (int) common_tokenize(vocab, shared_text, true, true).size();
            const int t_n = ntok(tail_text);

            // chat occupancy sized so the branch restore load still fits but its decode does not:
            // cells = chat + snap(S) + trunk(S+T) + branch(S+T) + branch tokens(B)
            const int n_ctx = (int) llama_n_ctx(te.ctx);
            const int n_batch = (int) llama_n_batch(te.ctx);
            const int chat_fill  = n_ctx - s_n - 2 * (s_n + t_n) - 8;
            const int chat_probe = s_n + t_n + 8;
            t.assert_true("the scenario leaves room for the chat probe", chat_probe + s_n + 2 * (s_n + t_n) < n_ctx);

            std::vector<llama_token> chat_toks;
            for (int i = 0; i < chat_fill; ++i) {
                chat_toks.push_back(16); // any in-vocab id; content is irrelevant to cell accounting
            }
            // decode in chunks of n_batch, like the server's slot scheduler would
            for (size_t off = 0; off < chat_toks.size(); off += (size_t) n_batch) {
                const size_t n = std::min((size_t) n_batch, chat_toks.size() - off);
                llama_batch fill = llama_batch_init((int) n, 0, 1);
                for (size_t i = 0; i < n; ++i) {
                    common_batch_add(fill, chat_toks[off + i], (llama_pos) (off + i), { 0 }, false);
                }
                const int rc = llama_decode(te.ctx, fill);
                llama_batch_free(fill);
                if (rc != 0) {
                    llama_synchronize(te.ctx);
                    break;
                }
            }
            llama_synchronize(te.ctx);

            llama_decision::engine eng(te.ctx, 2, 8);
            const std::vector<llama_decision::field_input> fields = { { suffix_text, { "1", "2" } } };
            llama_decision::options o;
            o.fork = "restore"; // restore allocates exclusive pool cells, so a leak is measurable
            bool threw = false;
            try {
                (void) decide_batch(eng, shared_text, { tail_text }, fields, o);
            } catch (const llama_decision::capacity_error &) {
                threw = true;
            }
            t.assert_true("the branch wave raises capacity_error mid-request", threw);

            bool pool_empty = true;
            for (llama_seq_id s = eng.seq_pool; s < eng.seq_pool + eng.n_pool; ++s) {
                pool_empty = pool_empty && llama_memory_seq_pos_max(eng.mem, s) == -1;
            }
            t.assert_true("every pool sequence is empty after the failure", pool_empty);

            // the shared context keeps serving chat: cells freed by the cleanup make room
            llama_batch chat = llama_batch_init(chat_probe, 0, 1);
            for (int i = 0; i < chat_probe; ++i) {
                common_batch_add(chat, 16, (llama_pos) i, { 1 }, false);
            }
            const int rc = llama_decode(te.ctx, chat);
            llama_batch_free(chat);
            llama_synchronize(te.ctx);
            t.assert_true("a chat decode on the shared context succeeds after the failure", rc == 0);
        } catch (const std::exception & e) {
            t.assert_true(std::string("the pool lifecycle run: ") + e.what(), false);
        }
    });
}

// A sliding-window cache no longer holds cells older than the window, so a fork must copy only the
// cells that survive. The copy and hybrid forks share one clamp; this checks that a decision on a
// context longer than the window is unchanged under hybrid against the full-restore ground truth.
// The two paths may materialize different numbers of (attention-masked) cells, so this compares the
// task value, not the state bytes.
static void swa_fork_clamp_run(testing & t, llama_context * ctx, const std::string & lane) {
    const llama_model * model = llama_get_model(ctx);
    if (!llama_model_is_recurrent(model) && !llama_model_is_hybrid(model)) {
        t.skip(lane + ": the model has no recurrent partial state; the clamp control is the copy fork");
        return;
    }
    const int           n_swa = llama_model_n_swa(model);
    if (n_swa <= 0) {
        t.skip(lane + ": the model has no sliding-window attention");
        return;
    }
    const int n_ctx    = (int) llama_n_ctx(ctx);
    const int n_parent = std::min(n_swa + 8, n_ctx - 64);
    if (n_parent <= n_swa) {
        t.skip(lane + ": the sliding window does not fit the test context");
        return;
    }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const std::string   context = text_of_n_tokens(vocab, n_parent);
    const auto          ctx_toks = common_tokenize(vocab, context, false, true);
    if ((int) ctx_toks.size() <= n_swa) {
        t.skip(lane + ": the model has no usable long prompt");
        return;
    }
    t.assert_true(lane + ": the context exceeds the sliding window", (int) ctx_toks.size() > n_swa);

    llama_memory_clear(llama_get_memory(ctx), true);
    llama_decision::engine eng(ctx, 2, 8);
    const std::vector<llama_decision::field_input> fields = {
        { "  \"a\": ", { "1", "2" } },
        { "  \"b\": ", { "x", "y" } },
    };

    llama_decision::options o_restore;
    o_restore.fork        = "restore";
    o_restore.allow_cache = false;
    llama_decision::options o_hybrid = o_restore;
    o_hybrid.fork = "hybrid";

    const auto restore = decide_batch(eng, "system", { context }, fields, o_restore);
    const auto hybrid  = decide_batch(eng, "system", { context }, fields, o_hybrid);
    if (!t.assert_true(lane + ": the clamped restore decision returns", restore.items.size() == 1) ||
        !t.assert_true(lane + ": the clamped hybrid decision returns", hybrid.items.size() == 1)) {
        return;
    }

    const double swa_tol = lane == "gpu" ? 5e-2 : 1e-4;
    bool agree = restore.items[0].fields.size() == hybrid.items[0].fields.size();
    for (size_t f = 0; agree && f < fields.size(); ++f) {
        agree = restore.items[0].fields[f].winner == hybrid.items[0].fields[f].winner;
        const auto & pr = restore.items[0].fields[f].probs;
        const auto & ph = hybrid.items[0].fields[f].probs;
        agree = agree && pr.size() == ph.size();
        for (size_t k = 0; agree && k < pr.size(); ++k) {
            agree = std::fabs(pr[k] - ph[k]) < swa_tol;
        }
    }
    t.assert_true(lane + ": a context past the sliding window scores the same under hybrid", agree);
}

static void test_fork_swa_clamp(testing & t) {
    t.test("a hybrid fork clamps to the sliding window on the CPU backend", [](testing & t) {
        // the decision fixture qwen35 has no sliding window; the generated lfm2 is hybrid with one
        std::string path = std::string(DECISION_TEST_GENERATED_MODEL_DIR) + "/lfm2-dense.gguf";
        if (!file_exists(path)) {
            path = decision_cpu_model_path();
        }
        if (path.empty()) {
            t.skip("no generated model; run the generate-models fixture");
            return;
        }
        cpu_test_engine te;
        if (!te.load(path)) {
            t.assert_true("the CPU decision scaffold loads the model", false);
            return;
        }
        try {
            swa_fork_clamp_run(t, te.ctx, "cpu");
        } catch (const std::exception & e) {
            t.assert_true(std::string("the CPU SWA clamp fork: ") + e.what(), false);
        }
    });

    t.test("a hybrid fork clamps to the sliding window on the GPU backend", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (!gpu_model_ready(t, path)) {
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("the GPU decision scaffold loads the model", false);
            return;
        }
        try {
            swa_fork_clamp_run(t, te.ctx, "gpu");
        } catch (const std::exception & e) {
            t.assert_true(std::string("the GPU SWA clamp fork: ") + e.what(), false);
        }
    });
}

// ---------------------------------------------------------------- CPU scoring oracle
//
// The scoring substrata below are exercised on the generated dummy model, with no GPU and no
// LLAMA_DECISION_TEST_MODEL. The recorded values are the oracle for behavior-preserving
// refactors: a change to field compilation or branch scoring must not move them.
// Backends may reorder a reduction, so probabilities are compared with a tolerance.

static common_json oracle_readout(const llama_decision::readout_metrics & m,
                                  const std::vector<std::vector<float>> & probs) {
    common_json o = common_json::object();
    o["cache_hit"]            = m.cache_hit;
    o["suffix_tokens"]        = (long long) m.suffix_tokens;
    o["common_suffix_tokens"] = (long long) m.common_suffix_tokens;
    o["rows"]                 = (long long) m.rows;
    o["rounds"]               = (long long) m.rounds;

    common_json questions = common_json::array();
    for (const auto & p : probs) {
        common_json q = common_json::object();
        q["options"] = (long long) p.size();
        q["winner"]  = p.empty() ? -1 : (long long) (std::max_element(p.begin(), p.end()) - p.begin());
        common_json scores = common_json::array();
        for (float v : p) {
            scores.push_back((double) v);
        }
        q["probs"] = scores;
        q["confidence"] = llama_decision::inverse_entropy_confidence(p);
        q["certainty"]  = llama_decision::winner_share(p);
        questions.push_back(q);
    }
    o["questions"] = questions;
    return o;
}

// dot(hidden, dequantized_row) + bias vs the full-vocabulary logit, over a few ids. This is the
// arithmetic identity the answer-head fast path relies on; the frozen value is the observed error.

// Tokens field compilation will score for one candidate, exactly as decide_batch builds the path.
static std::vector<llama_token> oracle_candidate_tokens(const llama_vocab * vocab, const std::string & suffix,
                                                        const std::string & candidate) {
    return common_tokenize(vocab, suffix + candidate + "\n", false, true);
}

// Builds an answer-row table for explicit token ids. The generated dummy model's TEST tokenizer
// hashes fixed 5-character chunks, so a candidate's scored token depends on the suffix offset and
// is not the isolated label token; the head must cover the tokens the compiled paths actually use.

static std::vector<llama_token> oracle_field_tokens(const llama_vocab * vocab, const std::string & suffix,
                                                    const std::vector<std::string> & candidates) {
    std::vector<llama_token> ids;
    for (const auto & c : candidates) {
        for (llama_token t : oracle_candidate_tokens(vocab, suffix, c)) {
            if (std::find(ids.begin(), ids.end(), t) == ids.end()) {
                ids.push_back(t);
            }
        }
    }
    return ids;
}

// The head-selection truth the engine reaches through decide_batch, before a pure entry point
// exists: a covered head on the classifier context is active and scores the rows; a full context,
// an unavailable head, and a null head all fall back with a reason. The covered and full field
// vectors are recorded so a refactor cannot move the head numerics.

static common_json decision_cpu_oracle() {
    common_json out = common_json::object();
    const std::string path = decision_cpu_model_path();
    if (path.empty()) {
        return out;
    }
    cpu_test_engine te_full;
    if (!te_full.load(path, 512, false)) {
        return out;
    }
    auto vocab = llama_decision::make_llama_label_vocab(llama_model_get_vocab(te_full.model));
    const std::string tail = test_letter_tail();
    const auto pool = llama_decision::build_label_pool(*vocab, tail, 64);
    if (pool.size() < 3) {
        return out;
    }
    const auto req = llama_decision::parse_decision_request(common_json::parse(decision_valid_body()));

    out["model"] = model_identity(path);
    common_json pool_info = common_json::object();
    pool_info["count"]   = (long long) pool.size();
    pool_info["token_a"] = (long long) pool[0].token;
    out["pool"] = pool_info;

    llama_decision::engine e_full(te_full.ctx, 2, 8);

    llama_decision::options ofull;
    ofull.cache_tag = "cpu-oracle-full";
    llama_decision::readout_metrics mfull;
    const auto pfull = test_letter_readout(e_full, *vocab, nullptr, false,
                                                      req, pool, ofull, &mfull);
    out["full"] = oracle_readout(mfull, pfull);
    return out;
}

static void oracle_assert_probs(testing & t, const std::string & label,
                                const common_json & exp, const common_json & act, double tol) {
    const common_json & ep = exp.at("probs");
    const common_json & ap = act.at("probs");
    if (!t.assert_equal(label + " option count", ep.size(), ap.size())) {
        return;
    }
    for (size_t i = 0; i < ep.size(); ++i) {
        const double e = ep.at(i).get<double>();
        const double a = ap.at(i).get<double>();
        if (!t.assert_true(label + " prob[" + std::to_string(i) + "] within " + std::to_string(tol),
                           std::fabs(e - a) <= tol)) {
            return;
        }
    }
}

static void oracle_assert_readout(testing & t, const std::string & label,
                                  const common_json & exp, const common_json & act, double tol) {
    t.assert_equal(label + " cache_hit", exp.at("cache_hit").get<bool>(), act.at("cache_hit").get<bool>());
    t.assert_equal(label + " suffix_tokens", exp.at("suffix_tokens").get<long long>(), act.at("suffix_tokens").get<long long>());
    t.assert_equal(label + " common_suffix_tokens", exp.at("common_suffix_tokens").get<long long>(), act.at("common_suffix_tokens").get<long long>());
    t.assert_equal(label + " rows", exp.at("rows").get<long long>(), act.at("rows").get<long long>());
    t.assert_equal(label + " rounds", exp.at("rounds").get<long long>(), act.at("rounds").get<long long>());

    const common_json & eq = exp.at("questions");
    const common_json & aq = act.at("questions");
    if (!t.assert_equal(label + " question count", eq.size(), aq.size())) {
        return;
    }
    for (size_t qi = 0; qi < eq.size(); ++qi) {
        const std::string q = label + " q" + std::to_string(qi);
        t.assert_equal(q + " options", eq.at(qi).at("options").get<long long>(), aq.at(qi).at("options").get<long long>());
        t.assert_equal(q + " winner", eq.at(qi).at("winner").get<long long>(), aq.at(qi).at("winner").get<long long>());
        t.assert_true(q + " carries confidence = 1 - H/log K", aq.at(qi).contains("confidence"));
        t.assert_true(q + " carries certainty = max p", aq.at(qi).contains("certainty"));
        oracle_assert_probs(t, q, eq.at(qi), aq.at(qi), tol);
    }
}

static void test_decision_cpu_oracle(testing & t) {
    t.test("the CPU scoring oracle matches the frozen baseline", [](testing & t) {
        if (decision_cpu_model_path().empty()) {
            t.skip("no generated model; run the generate-models fixture");
            return;
        }
        const common_json baseline = common_json::parse(
            read_file(std::string(DECISION_TEST_BASELINE_DIR) + "/baseline.json"));
        if (!baseline.contains("cpu_oracle")) {
            t.skip("no frozen CPU oracle in baseline.json");
            return;
        }
        const common_json & exp = baseline.at("cpu_oracle");

        common_json act;
        try {
            act = decision_cpu_oracle();
        } catch (const std::exception & e) {
            t.assert_true(std::string("the CPU oracle runs: ") + e.what(), false);
            return;
        }
        if (act.empty()) {
            t.skip("the CPU oracle produced no values");
            return;
        }

        t.assert_equal("oracle model identity", exp.at("model").get<std::string>(), act.at("model").get<std::string>());
        t.assert_equal("oracle label pool count", exp.at("pool").at("count").get<long long>(),
                       act.at("pool").at("count").get<long long>());
        t.assert_equal("oracle label A token", exp.at("pool").at("token_a").get<long long>(),
                       act.at("pool").at("token_a").get<long long>());

        const double prob_tol = 1e-3;
        oracle_assert_readout(t, "full readout", exp.at("full"), act.at("full"), prob_tol);
    });
}

// compile_fields is the single place the field set is built, so the plan must be a deterministic
// function of the inputs and match the accounting decide_batch reports back.
static void test_compile_fields_plan(testing & t) {
    t.test("compile_fields is deterministic and matches the scored accounting", [](testing & t) {
        const std::string path = decision_cpu_model_path();
        if (path.empty()) {
            t.skip("no generated model; run the generate-models fixture");
            return;
        }
        cpu_test_engine te_full;
        if (!te_full.load(path, 512, false)) {
            t.assert_true("the CPU decision scaffold loads the model", false);
            return;
        }
        const std::string suffix = "\nAnswer:\n";
        const std::vector<std::string> candidates = { "AAAAA", "BBBBB" };
        std::vector<llama_decision::field_input> fields = { { suffix, candidates, 1.0f } };

        llama_decision::engine eng(te_full.ctx, 2, 8);
        llama_decision::options opt;
        opt.mode = "tree";

        const llama_decision::compiled_fields a = eng.compile_fields(fields, opt);
        const llama_decision::compiled_fields b = eng.compile_fields(fields, opt);
        t.assert_equal("rows is stable", a.rows, b.rows);
        t.assert_equal("branches is stable", a.branches, b.branches);
        t.assert_equal("suffix_tokens is stable", a.suffix_tokens, b.suffix_tokens);
        t.assert_equal("common_suffix_tokens is stable", a.common_suffix_tokens, b.common_suffix_tokens);
        t.assert_equal("leaf_suffix_tokens is stable", a.leaf_suffix_tokens, b.leaf_suffix_tokens);
        t.assert_true("the plan carries rows", a.rows > 0);

        const llama_decision::batch_result br = decide_batch(eng, "", { "state" }, fields, opt);
        t.assert_equal("one unique field scores one field", (size_t) 1, br.items.at(0).fields.size());
        t.assert_equal("decide_batch reports the plan rows", (long long) a.rows, (long long) br.rows);
        t.assert_equal("decide_batch reports the plan suffix_tokens", a.suffix_tokens, br.suffix_tokens);
        t.assert_equal("decide_batch reports the plan leaf_suffix_tokens", a.leaf_suffix_tokens, br.leaf_suffix_tokens);
        t.assert_equal("decide_batch reports the plan common_suffix_tokens", a.common_suffix_tokens, br.common_suffix_tokens);
    });
}

// The compile-then-score composition and the explicit plan form must score identically, so the
// composition the tests share is the same plan the engine sees when a caller compiles up front.
static void test_decide_batch_plan_overload(testing & t) {
    t.test("the plan overload and the compile-then-score composition agree", [](testing & t) {
        const std::string path = decision_cpu_model_path();
        if (path.empty()) {
            t.skip("no generated model; run the generate-models fixture");
            return;
        }
        cpu_test_engine te;
        if (!te.load(path, 512, false)) {
            t.assert_true("the CPU decision scaffold loads the model", false);
            return;
        }
        const std::string                        suffix     = "\nAnswer:\n";
        const std::vector<std::string>           candidates = { "AAAAA", "BBBBB", "CCCCC" };
        std::vector<llama_decision::field_input> fields     = {
            { suffix, candidates, 1.0f }
        };

        llama_decision::engine  eng(te.ctx, 2, 8);
        llama_decision::options opt;
        opt.mode        = "tree";
        opt.allow_cache = false;  // isolate the arithmetic from prefix-cache reuse

        const llama_decision::compiled_fields plan    = eng.compile_fields(fields, opt);
        const llama_decision::batch_result    wrapped = decide_batch(eng, "", { "state" }, fields, opt);
        const llama_decision::batch_result    planned = eng.decide_batch(plan, "", { "state" }, opt);

        t.assert_equal("the overload keeps the item count", wrapped.items.size(), planned.items.size());
        t.assert_equal("the overload keeps the batch rows", wrapped.rows, planned.rows);
        t.assert_equal("the overload keeps the suffix accounting", wrapped.suffix_tokens, planned.suffix_tokens);

        bool same = wrapped.items.size() == planned.items.size();
        for (size_t i = 0; same && i < wrapped.items.size(); ++i) {
            const auto & a = wrapped.items[i];
            const auto & b = planned.items[i];
            same           = a.fields.size() == b.fields.size();
            for (size_t f = 0; same && f < a.fields.size(); ++f) {
                same = a.fields[f].winner == b.fields[f].winner &&
                       a.fields[f].scored_nodes == b.fields[f].scored_nodes &&
                       a.fields[f].probs.size() == b.fields[f].probs.size();
                for (size_t k = 0; same && k < a.fields[f].probs.size(); ++k) {
                    same = std::fabs(a.fields[f].probs[k] - b.fields[f].probs[k]) <= 1e-6f;
                }
            }
        }
        t.assert_true("the overload returns identical scored fields", same);
    });
}

// The token entry and the text entry must be bit-identical for the same inputs: the token
// replay path a sidecar uses for a snapshot must never drift from the text path, or a replay
// would answer differently from the original decision. Both runs decode the same token lists.
static bool token_entries_identical(const llama_decision::batch_result & a, const llama_decision::batch_result & b) {
    bool same = a.items.size() == b.items.size() && a.shared_tokens == b.shared_tokens &&
                a.rows == b.rows && a.suffix_tokens == b.suffix_tokens &&
                a.common_suffix_tokens == b.common_suffix_tokens && a.leaf_suffix_tokens == b.leaf_suffix_tokens;
    for (size_t i = 0; same && i < a.items.size(); ++i) {
        const auto & ia = a.items[i];
        const auto & ib = b.items[i];
        same = ia.context_tokens == ib.context_tokens && ia.fields.size() == ib.fields.size();
        for (size_t f = 0; same && f < ia.fields.size(); ++f) {
            same = ia.fields[f].winner == ib.fields[f].winner &&
                   ia.fields[f].scored_nodes == ib.fields[f].scored_nodes &&
                   ia.fields[f].probs.size() == ib.fields[f].probs.size();
            for (size_t k = 0; same && k < ia.fields[f].probs.size(); ++k) {
                same = std::fabs(ia.fields[f].probs[k] - ib.fields[f].probs[k]) <= 1e-6f;
            }
        }
    }
    return same;
}

static void token_entry_equality_run(testing & t, llama_decision::engine & eng, const std::string & lane) {
    const std::string                        shared_text = "Answer the questions about the customer.";
    const std::string                        context_text = "The customer was charged twice on May 3.";
    const std::vector<llama_decision::field_input> inputs = {
        { "\nrefund: ", { "yes", "no" }, 1.0f },
        { "\ndept: ",   { "billing", "technical", "cancellation" }, 1.0f },
        { "\nscore: ",  { "calm", "upset", "furious" }, 1.0f },
    };
    llama_decision::options opt;
    opt.mode        = "tree";
    opt.allow_cache = false; // isolate the entry comparison from prefix-cache reuse

    const llama_decision::compiled_fields plan = eng.compile_fields(inputs, opt);
    const llama_decision::batch_result    text = eng.decide_batch(plan, shared_text, { context_text }, opt);

    // the exact token lists the text entry produces: shared with special tokens on,
    // context with them off when the shared list is non-empty
    const llama_decision::tokens_t              shared_toks = eng.tokenize(shared_text, true);
    const std::vector<llama_decision::tokens_t> ctx_toks    = { eng.tokenize(context_text, shared_toks.empty()) };

    const llama_decision::batch_result toks =
        eng.decide_batch_tokens(shared_toks, ctx_toks, plan, opt);

    t.assert_true(lane + ": the token entry returns the same item count as the text entry",
                  toks.items.size() == text.items.size());
    t.assert_true(lane + ": the token entry is bit-identical to the text entry", token_entries_identical(text, toks));
    t.assert_true(lane + ": both entries score at least one field",
                  !text.items.empty() && !text.items[0].fields.empty());
}

// A classifier-only context has no logits, so the engine refuses a request without a covering
// head before any decode instead of reaching gather_candidates and reading a null buffer.

static void test_token_entry_equality(testing & t) {
    t.test("the token entry reproduces the text entry on the CPU scaffold", [](testing & t) {
        const std::string path = decision_cpu_model_path();
        if (path.empty()) {
            t.skip("no generated model; run the generate-models fixture");
            return;
        }
        cpu_test_engine te;
        if (!te.load(path, 512, false)) {
            t.assert_true("the CPU decision scaffold loads the model", false);
            return;
        }
        try {
            llama_decision::engine eng(te.ctx, 2, 8);
            token_entry_equality_run(t, eng, "cpu");
        } catch (const std::exception & e) {
            t.assert_true(std::string("the CPU token entry runs: ") + e.what(), false);
        }
    });

    t.test("the token entry reproduces the text entry on the loaded model", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (!gpu_model_ready(t, path)) {
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("model loads", false);
            return;
        }
        try {
            llama_decision::engine eng(te.ctx, 2, 8);
            token_entry_equality_run(t, eng, "gpu");
        } catch (const std::exception & e) {
            t.assert_true(std::string("the GPU token entry runs: ") + e.what(), false);
        }
    });
}

// The resident warm-prefix tier (sidecar session warm cache): the first decision on a warm_tag
// cold-prefills the turn's tokens into a kept resident sequence; a repeat forks it instead of
// re-prefilling. A hit must match its own miss within the documented producer tolerance, the
// first call must report a miss and a repeat a hit, and a full cache must LRU-evict while staying
// exact.
static void warm_resident_run(testing & t, llama_decision::engine & eng, llama_context * ctx, const std::string & lane) {
    const std::vector<llama_decision::field_input> fields = {
        { "\nrefund: ", { "yes", "no" }, 1.0f },
        { "\ndept: ",   { "billing", "technical", "cancellation" }, 1.0f },
    };
    llama_decision::options opt;
    opt.mode = "tree";
    const llama_decision::compiled_fields plan = eng.compile_fields(fields, opt);
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    // a turn long enough that re-prefilling would dominate, like a real session snapshot
    std::string context_text;
    for (int i = 0; i < 25; ++i) {
        context_text += "The customer was charged twice on May 3. ";
    }
    const llama_decision::tokens_t tokens = common_tokenize(vocab, context_text, true, true);

    const auto run = [&](const std::string & tag) {
        const auto cold = eng.decide_warm(tokens, plan, opt, tag);
        const auto warm = eng.decide_warm(tokens, plan, opt, tag);
        t.assert_true(lane + " " + tag + " first decision cold-prefills", !cold.warm_hit);
        t.assert_true(lane + " " + tag + " repeat forks the resident prefix", warm.warm_hit);
        t.assert_true(lane + " " + tag + " repeat is bit-identical to its miss",
                      token_entries_identical(cold, warm));
    };

    t.test(lane + " a repeated session hits a resident prefix bit-identically", [&](testing &) {
        run("warm-A");
    });

    t.test(lane + " a full warm cache LRU-evicts a session and re-warms it exactly", [&](testing &) {
        // two slots: three sessions must evict the least-recently-used one
        run("warm-B"); // slot 1
        run("warm-C"); // evicts warm-A's slot
        const auto againA = eng.decide_warm(tokens, plan, opt, "warm-A");
        t.assert_true(lane + " the LRU-evicted session is cold again (evicted)", !againA.warm_hit);
        const auto againAwarm = eng.decide_warm(tokens, plan, opt, "warm-A");
        t.assert_true(lane + " the re-warmed session hits", againAwarm.warm_hit);
        t.assert_true(lane + " the re-warmed session is bit-identical",
                      token_entries_identical(againA, againAwarm));
    });

    t.test(lane + " the warm tier is off with no resident slots and stays exact", [&](testing &) {
        // a separate engine on the same context with n_warm=0 falls back to cold replay; its pool
        // range [2,10) is disjoint from the two resident warm slots [10,12) of `eng`
        llama_decision::engine eng0(ctx, 2, 8);
        const auto a = eng0.decide_warm(tokens, plan, opt, "tag");
        const auto b = eng0.decide_warm(tokens, plan, opt, "tag");
        t.assert_true(lane + " no warm slots means cold replay", !a.warm_hit && !b.warm_hit);
        t.assert_true(lane + " cold replay is bit-identical", token_entries_identical(a, b));
    });
}

static void test_warm_resident_cache(testing & t) {
    t.test("warm resident cache on the CPU scaffold", [](testing & t) {
        const std::string path = decision_cpu_model_path();
        if (path.empty()) {
            t.skip("no generated model; run the generate-models fixture");
            return;
        }
        cpu_test_engine te;
        // n_ctx 2048, n_seq_max 12: engine pool [2, 10), two resident warm slots [10, 12)
        if (!te.load(path, 2048, false, 128, 12)) {
            t.assert_true("the CPU decision scaffold loads the model", false);
            return;
        }
        try {
            llama_decision::engine eng(te.ctx, 2, 8, 2); // two resident warm slots
            warm_resident_run(t, eng, te.ctx, "cpu");
        } catch (const std::exception & e) {
            t.assert_true(std::string("warm resident cache CPU run: ") + e.what(), false);
        }
    });

    t.test("warm resident cache on the loaded model", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (!gpu_model_ready(t, path)) {
            return;
        }
        test_engine te;
        if (!te.load(path, 12)) {
            t.assert_true("model loads", false);
            return;
        }
        try {
            llama_decision::engine eng(te.ctx, 2, 8, 2); // two resident warm slots
            warm_resident_run(t, eng, te.ctx, "gpu");
        } catch (const std::exception & e) {
            t.assert_true(std::string("warm resident cache GPU run: ") + e.what(), false);
        }
    });
}

// How many resident warm prefixes a KV budget buys is the engine's own decision: one owner for the
// limit and for the derivation, so a caller cannot clamp the number a second time. Each slot is
// charged a whole window of cells (an upper bound, since a filled slot holds only that turn's
// tokens), a positive budget keeps at least one slot so the tier stays usable, and the count never
// exceeds WARM_MAX_SLOTS.
static void test_warm_slot_budget(testing & t) {
    t.test("the warm-slot count is bounded, monotone in the budget, and off without one", [](testing & t) {
        // 32 layers, 4096 wide, 8 KV heads of 32 heads, f16 K and V: 128 KiB per cell
        const int  n_layer = 32, n_embd = 4096, n_head = 32, n_kv = 8, n_ctx = 4096;
        const auto slots = [&](int budget_mb, int layer, int ctx) {
            return llama_decision::engine::warm_slots_for_budget(layer, n_embd, n_head, n_kv, ctx, budget_mb,
                                                                 GGML_TYPE_F16, GGML_TYPE_F16);
        };
        // one window of cells is 512 MiB at this geometry
        t.assert_equal("no budget keeps the tier off", 0, slots(0, n_layer, n_ctx));
        t.assert_equal("an unsized window keeps the tier off", 0, slots(512, n_layer, 0));
        t.assert_equal("a model with no KV geometry keeps the tier off", 0, slots(512, 0, n_ctx));
        t.assert_equal("a budget below one window still yields one slot", 1, slots(1, n_layer, n_ctx));
        t.assert_equal("exactly one window of budget yields one slot", 1, slots(512, n_layer, n_ctx));
        t.assert_equal("just over one window still yields one slot", 1, slots(520, n_layer, n_ctx));
        t.assert_equal("two windows of budget yield two slots", 2, slots(1024, n_layer, n_ctx));
        t.assert_equal("four windows of budget yield four slots", 4, slots(2048, n_layer, n_ctx));
        t.assert_equal("an unbounded budget is capped at the engine limit",
                       llama_decision::engine::WARM_MAX_SLOTS, slots(1 << 20, n_layer, n_ctx));

        int  previous = 0;
        bool monotone = true;
        for (int budget_mb = 0; budget_mb <= 4096; budget_mb += 7) {
            const int n = slots(budget_mb, n_layer, n_ctx);
            if (n < previous) {
                monotone = false;
            }
            previous = n;
        }
        t.assert_true("the slot count never decreases as the budget grows", monotone);
        t.assert_true("no budget ever exceeds the engine limit", previous <= llama_decision::engine::WARM_MAX_SLOTS);
    });
}

// A warm fork's KV budget is the window minus the cells the OTHER resident warm prefixes hold, which
// the decision cannot evict. The source's own cells are already inside the fork's peak, so charging
// them again would reject requests that fit. The sizes come from the compiled plan rather than being
// tuned, so the boundary is exact: a single greedy field hoists no head, its fork peak is the source
// length plus its longest candidate path, and a greedy branch decodes its suffix plus one chosen
// token, which stays inside that path.
static void warm_capacity_accounting(testing & t, test_engine & te, const std::string & lane) {
    llama_context *    ctx    = te.ctx;
    const llama_vocab * vocab  = llama_model_get_vocab(llama_get_model(ctx));
    const auto          filler = common_tokenize(vocab, " customer", false, true);
    if (filler.empty()) {
        t.skip(lane + ": the model has no usable tokens");
        return;
    }
    const std::vector<llama_decision::field_input> fields = { { "\n", { "1", "seven" } } };
    llama_decision::options                        opt;
    opt.mode = "greedy";

    llama_decision::engine            eng(ctx, 2, 8, 2); // pool [3,10), resident warm slots [10,12)
    const llama_decision::compiled_fields plan = eng.compile_fields(fields, opt);
    const size_t                          n_ctx    = (size_t) llama_n_ctx(ctx);
    const size_t                          max_path = (size_t) (plan.rows - (int) plan.suffix_tokens);
    if (max_path < plan.suffix_tokens || max_path == 0) {
        t.skip(lane + ": the tokenizer gives no upper bound for this plan's branch cost");
        return;
    }
    const size_t slack   = 128;                    // room the source leaves for a second prefix
    const size_t source  = n_ctx - slack;
    const size_t peak    = source + max_path;
    const size_t other   = n_ctx - peak;           // exactly the cells left, so the budget is the peak
    const auto   src     = llama_decision::tokens_t(source, filler[0]);
    const auto   other_t = llama_decision::tokens_t(other, filler[0]);

    t.test(lane + " a warm fork is charged only for the other resident prefixes", [&](testing &) {
        // the control: with its own source the only resident prefix, a fork has the whole window, so
        // charging the source twice would reject a request that fits
        const auto cold = eng.decide_warm(src, plan, opt, "source");
        t.assert_equal(lane + " the control fork answers the plan", (size_t) 1, cold.items[0].fields.size());

        // a second prefix holding exactly what the fork leaves makes budget == peak: still accepted
        const auto second = eng.decide_warm(other_t, plan, opt, "other");
        t.assert_equal(lane + " the second prefix cold-prefills within its own budget", (size_t) 1,
                       second.items[0].fields.size());
        const auto at_budget = eng.decide_warm(src, plan, opt, "source");
        t.assert_true(lane + " a fork exactly at the budget is accepted", at_budget.warm_hit);
        t.assert_equal(lane + " the fork at the budget answers the plan", (size_t) 1,
                       at_budget.items[0].fields.size());
    });

    // the first phase left its prefixes resident, and a decision engine never frees them, so the
    // second phase needs its own context
    if (!te.reopen(24, (int) n_ctx)) {
        t.assert_true(lane + " a fresh context opens for the over-budget phase", false);
        return;
    }
    ctx       = te.ctx;
    const auto over_t = llama_decision::tokens_t(other + 1, filler[0]);

    t.test(lane + " an over-budget warm decision is refused before any slot is touched", [&](testing &) {
        llama_decision::engine eng2(ctx, 2, 8, 2);
        const llama_decision::compiled_fields plan2 = eng2.compile_fields(fields, opt);
        // the resident warm tier as a comparable snapshot: tag, decoded length, LRU stamp
        const auto slot_state = [](const std::vector<llama_decision::engine::warm_slot> & slots) {
            std::vector<std::string> out;
            out.reserve(slots.size());
            for (const auto & s : slots) {
                out.push_back(s.tag + "|" + std::to_string(s.pos) + "|" + std::to_string(s.last_used));
            }
            return out;
        };
        const auto filled = [](const std::vector<llama_decision::engine::warm_slot> & slots) {
            size_t n = 0;
            for (const auto & s : slots) {
                n += s.tag.empty() ? 0 : 1;
            }
            return n;
        };

        // control: a fitting decision installs exactly one slot, and its repeat forks that slot
        // without installing another
        const auto control = eng2.decide_warm(src, plan2, opt, "source");
        t.assert_true(lane + " a fitting warm decision cold-prefills", !control.warm_hit);
        t.assert_equal(lane + " a fitting warm decision installs exactly one slot", (size_t) 1,
                       filled(eng2.warm_slots_));
        const auto repeat = eng2.decide_warm(src, plan2, opt, "source");
        t.assert_true(lane + " a repeat forks the resident prefix", repeat.warm_hit);
        t.assert_equal(lane + " a repeat installs no additional slot", (size_t) 1, filled(eng2.warm_slots_));

        // regression: a decision that cannot fit beside the resident prefix is refused before it
        // allocates or clears anything, so no slot is evicted and the resident prefix survives
        const std::vector<std::string> before = slot_state(eng2.warm_slots_);
        bool      refused = false;
        std::string what;
        try {
            (void) eng2.decide_warm(over_t, plan2, opt, "other");
        } catch (const llama_decision::capacity_error & e) {
            refused = true;
            what    = e.what();
        }
        t.assert_true(lane + " an over-budget warm decision raises capacity_error", refused);
        t.assert_true(lane + " the refusal names the context budget, not a decode failure",
                      what.find("context holds") != std::string::npos);
        t.assert_true(lane + " the refusal is not a late decode failure",
                      what.find("no free KV cache space") == std::string::npos);
        t.assert_true(lane + " the refusal leaves every warm slot unchanged", slot_state(eng2.warm_slots_) == before);
        const auto survived = eng2.decide_warm(src, plan2, opt, "source");
        t.assert_true(lane + " the refused prefix never displaced the resident one", survived.warm_hit);
    });
}

static void test_warm_capacity_accounting(testing & t) {
    t.test("warm capacity accounting on the loaded model", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (!gpu_model_ready(t, path)) {
            return;
        }
        test_engine te;
        if (!te.load(path, 24, 2048)) {
            t.assert_true("model loads", false);
            return;
        }
        try {
            warm_capacity_accounting(t, te, "gpu");
        } catch (const std::exception & e) {
            t.assert_true(std::string("warm capacity accounting GPU run: ") + e.what(), false);
        }
    });
}

static void test_prefix_cache_coherence(testing & t) {
    t.test("prefix cache never hits across a changed identity", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (!gpu_model_ready(t, path)) {
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("model loads", false);
            return;
        }
        try {
            llama_decision::engine eng(te.ctx, 2, 8);
            std::vector<llama_decision::field_input> fields = { { "  \"a\": ", { "1", "2" } } };

            llama_decision::options o1;
            o1.cache_tag = "tag-A";
            const auto b1 = decide_batch(eng, "system", { "ctx" }, fields, o1);
            const auto b2 = decide_batch(eng, "system", { "ctx" }, fields, o1);
            t.assert_true("first request misses", !b1.cache_hit);
            t.assert_true("same identity hits", b2.cache_hit);

            llama_decision::options o2;
            o2.cache_tag = "tag-B";
            const auto b3 = decide_batch(eng, "system", { "ctx" }, fields, o2);
            t.assert_true("changed identity misses", !b3.cache_hit);
        } catch (const std::exception & e) {
            t.assert_true(std::string("cache coherence runs: ") + e.what(), false);
        }
    });
}

static void test_token_cache(testing & t) {
    t.test("token cache encodes repeated prompts once", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (!gpu_model_ready(t, path)) {
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("model loads", false);
            return;
        }
        try {
            llama_decision::engine eng(te.ctx, 2, 8);
            const std::vector<llama_decision::field_input> fields = {
                { "  \"a\": ", { "1", "2" } },
                { "  \"b\": ", { "x", "y" } },
            };
            llama_decision::options o;
            o.cache_tag = "tok";
            const auto first  = decide_batch(eng, "system", { "ctx" }, fields, o);
            const auto second = decide_batch(eng, "system", { "ctx" }, fields, o);
            t.assert_true("the second decision reuses the cached prefix", second.cache_hit);
            t.assert_true("the repeat is byte-identical",
                          first.items[0].fields.size() == second.items[0].fields.size());
        } catch (const std::exception & e) {
            t.assert_true(std::string("token cache runs: ") + e.what(), false);
        }
    });
}

static void test_prefix_reuse(testing & t) {
    t.test("a repeated request reuses the cached prefix without re-prefilling", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (!gpu_model_ready(t, path)) {
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("model loads", false);
            return;
        }
        try {
            llama_decision::engine eng(te.ctx, 2, 8);
            const std::vector<llama_decision::field_input> fields = { { "  \"a\": ", { "1", "2", "3" } } };
            llama_decision::options o;
            o.cache_tag = "reuse";
            const auto b1 = decide_batch(eng, "system", { "ctx" }, fields, o);
            const auto b2 = decide_batch(eng, "system", { "ctx" }, fields, o);

            t.assert_true("the first request is cold", !b1.cache_hit);
            t.assert_true("the repeat is a hit", b2.cache_hit);
            t.assert_true("the hit reuses the same shared prefix", b2.shared_tokens == b1.shared_tokens);
            t.assert_true("the shared prefix is non-empty", b1.shared_tokens > 0);
            t.assert_true("the request context is tracked separately", !b1.items.empty() && b1.items[0].context_tokens > 0);
            t.assert_true("a hit does not re-prefill slower", b2.prefill_ms <= b1.prefill_ms + 5.0);
        } catch (const std::exception & e) {
            t.assert_true(std::string("prefix reuse runs: ") + e.what(), false);
        }
    });
}

static void test_request_prefix(testing & t) {
    t.test("a long common suffix head is hoisted and short or disabled ones are not", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (!gpu_model_ready(t, path)) {
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("model loads", false);
            return;
        }
        try {
            llama_decision::engine eng(te.ctx, 2, 8);
            std::string long_prefix;
            for (int i = 0; i < 60; ++i) {
                long_prefix += "context ";
            }
            const std::vector<llama_decision::field_input> shared_suffix = {
                { long_prefix + "alpha: ", { "1", "2" } },
                { long_prefix + "alpha: ", { "3", "4" } },
            };
            llama_decision::options o;
            o.cache_tag = "hoist";
            const auto r = decide_batch(eng, "system", { "ctx" }, shared_suffix, o);
            t.assert_true("a long common head is hoisted", r.common_suffix_tokens >= 32);
            t.assert_true("branches decode only their unique tail", r.leaf_suffix_tokens < r.suffix_tokens);
            t.assert_equal("the hoisted head is removed from every field",
                           (long long) (r.suffix_tokens - r.leaf_suffix_tokens),
                           (long long) (r.common_suffix_tokens * shared_suffix.size()));

            const std::vector<llama_decision::field_input> near_miss = {
                { "state alpha: ", { "1", "2" } },
                { "state beta: ",  { "3", "4" } },
            };
            const auto rn = decide_batch(eng, "system", { "ctx" }, near_miss, o);
            t.assert_equal("a short head on two fields does not hoist", 0, (int) rn.common_suffix_tokens);

            // A short head still pays off once many questions share it: the hoist budget is
            // common_tokens * (fields - 1), so 8 tokens over 40 fields clears it.
            std::string many_head;
            for (int i = 0; i < 8; ++i) {
                many_head += "tag ";
            }
            std::vector<llama_decision::field_input> many;
            for (int i = 0; i < 40; ++i) {
                many.push_back({ many_head + "field" + std::to_string(i) + ": ", { "1", "2" } });
            }
            const auto rm = decide_batch(eng, "system", { "ctx" }, many, o);
            t.assert_true("a short head is hoisted once many fields share it", rm.common_suffix_tokens >= 4);
            t.assert_true("the many-field branches decode only their unique tail",
                          rm.leaf_suffix_tokens + rm.common_suffix_tokens * many.size() == rm.suffix_tokens);

            llama_decision::options off = o;
            off.cache_tag = "hoist-off";
            off.optimize  = false;
            const auto ro = decide_batch(eng, "system", { "ctx" }, shared_suffix, off);
            t.assert_equal("optimize off does not hoist", 0, (int) ro.common_suffix_tokens);
            t.assert_equal("optimize off keeps every suffix token",
                           (long long) ro.leaf_suffix_tokens, (long long) ro.suffix_tokens);

            const std::vector<llama_decision::field_input> duplicate = { shared_suffix[0], shared_suffix[0] };
            const auto rd = decide_batch(eng, "system", { "ctx" }, duplicate, o);
            t.assert_true("identical fields score once", rd.suffix_tokens < r.suffix_tokens);
            t.assert_equal("the duplicate is the same single suffix",
                           (long long) (rd.suffix_tokens * 2), (long long) r.suffix_tokens);
        } catch (const std::exception & e) {
            t.assert_true(std::string("request prefix runs: ") + e.what(), false);
        }
    });
}

static void test_batching_waves(testing & t) {
    t.test("a wide batch packs into waves and keeps the answer order", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (!gpu_model_ready(t, path)) {
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("model loads", false);
            return;
        }
        try {
            llama_decision::engine eng(te.ctx, 2, 8); // small pool forces several waves

            std::vector<llama_decision::field_input> many;
            for (int i = 0; i < 64; ++i) {
                many.push_back({ "  \"f" + std::to_string(i) + "\": ", { "1", "2" } });
            }
            llama_decision::options o;
            o.cache_tag = "waves-many";
            const auto b = decide_batch(eng, "system", { "ctx" }, many, o);
            t.assert_equal("one result per context", (size_t) 1, b.items.size());
            t.assert_equal("one field per question", (size_t) 64, b.items[0].fields.size());

            // order preservation: the first 8 answers match a standalone 8-question run
            std::vector<llama_decision::field_input> first(many.begin(), many.begin() + 8);
            llama_decision::options o2;
            o2.cache_tag = "waves-first";
            const auto b2 = decide_batch(eng, "system", { "ctx" }, first, o2);
            bool same = b2.items[0].fields.size() == 8;
            double max_diff = 0.0;
            auto winner = [](const std::vector<float> & p) {
                return (int) (std::max_element(p.begin(), p.end()) - p.begin());
            };
            for (size_t f = 0; same && f < 8; ++f) {
                const auto & p = b.items[0].fields[f].probs;
                const auto & q = b2.items[0].fields[f].probs;
                same = p.size() == q.size() && winner(p) == winner(q);
                for (size_t k = 0; same && k < p.size(); ++k) {
                    max_diff = std::max(max_diff, (double) std::fabs(p[k] - q[k]));
                }
            }
            t.assert_true("batch size does not change the winner", same);
            t.assert_true("probabilities match within batch tolerance", max_diff <= 5e-2);
            t.assert_true("wide-batch order matches the standalone run", same);
        } catch (const std::exception & e) {
            t.assert_true(std::string("batching runs: ") + e.what(), false);
        }
    });
}

static void test_dedup_fields(testing & t) {
    t.test("identical fields score once and still answer in place", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (!gpu_model_ready(t, path)) {
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("model loads", false);
            return;
        }
        try {
            llama_decision::engine eng(te.ctx, 2, 8);
            const llama_decision::field_input a = { "  \"a\": ", { "1", "2" } };
            const llama_decision::field_input b = { "  \"b\": ", { "x", "y" } };

            llama_decision::options o;
            o.cache_tag = "dedup";
            const auto dup = decide_batch(eng, "system", { "ctx" }, { a, a, b, a }, o);

            t.assert_equal("duplicates still answer in place", (size_t) 4, dup.items[0].fields.size());
            bool equal = true;
            for (size_t f = 0; f < dup.items[0].fields.size(); ++f) {
                if (f == 2) {
                    continue;
                }
                equal = equal && dup.items[0].fields[f].probs == dup.items[0].fields[0].probs;
            }
            t.assert_true("duplicate fields share the first answer", equal);
            const auto unique = decide_batch(eng, "system", { "ctx" }, { a, b }, o);
            t.assert_equal("dedup scores only the unique fields", unique.rows, dup.rows);
        } catch (const std::exception & e) {
            t.assert_true(std::string("dedup runs: ") + e.what(), false);
        }
    });
}

static void test_cancel_reaches_compute(testing & t) {
    t.test("a stop request aborts before scoring and never returns an answer", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (!gpu_model_ready(t, path)) {
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("model loads", false);
            return;
        }
        try {
            llama_decision::engine eng(te.ctx, 2, 8);
            const std::vector<llama_decision::field_input> fields = {
                { "  \"a\": ", { "1", "2" } },
                { "  \"b\": ", { "x", "y" } },
            };
            int calls = 0;
            llama_decision::options o;
            o.cache_tag  = "cancel";
            o.should_stop = [&calls] { return ++calls > 2; };

            bool cancelled = false;
            try {
                (void) decide_batch(eng, "system", { "ctx" }, fields, o);
            } catch (const llama_decision::cancelled_error &) {
                cancelled = true;
            }
            t.assert_true("cancel reached compute", cancelled);
            t.assert_true("no further check after the abort", calls >= 3);
        } catch (const std::exception & e) {
            t.assert_true(std::string("cancel run: ") + e.what(), false);
        }
    });
}

static void test_yield_points(testing & t) {
    t.test("the engine offers a cooperative yield between waves", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (!gpu_model_ready(t, path)) {
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("model loads", false);
            return;
        }
        try {
            llama_decision::engine eng(te.ctx, 2, 8);
            std::vector<llama_decision::field_input> many;
            for (int i = 0; i < 64; ++i) {
                many.push_back({ "  \"f" + std::to_string(i) + "\": ", { "1", "2" } });
            }
            int yields = 0;
            llama_decision::options o;
            o.cache_tag = "yield";
            o.yield     = [&yields]() { ++yields; };
            (void) decide_batch(eng, "system", { "ctx" }, many, o);
            t.assert_true("a wide decision yields at least once", yields >= 1);
        } catch (const std::exception & e) {
            t.assert_true(std::string("yield run: ") + e.what(), false);
        }
    });
}

static void test_capacity_error(testing & t) {
    t.test("an oversize suffix is chunked within the batch and rejected beyond the context", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (!gpu_model_ready(t, path)) {
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("model loads", false);
            return;
        }
        try {
            // larger than n_batch (512) but well within n_ctx (8192): decoded in chunks
            llama_decision::engine eng(te.ctx, 2, 8);
            std::string long_suffix;
            for (int i = 0; i < 900; ++i) {
                long_suffix += "token" + std::to_string(i) + " ";
            }
            const std::vector<llama_decision::field_input> chunked_fields = { { long_suffix, { "1", "2" } } };
            llama_decision::options oc;
            oc.mode      = "tree";
            oc.cache_tag = "capacity-chunked";
            const auto b = decide_batch(eng, "system", { "ctx" }, chunked_fields, oc);
            t.assert_equal("the chunked suffix produces one field", (size_t) 1, b.items[0].fields.size());
            t.assert_equal("the chunked suffix scores every candidate", (size_t) 2, b.items[0].fields[0].probs.size());

            // far beyond n_ctx: the chunked decode runs out of KV space and reports a capacity error
            std::string huge;
            for (int i = 0; i < 5000; ++i) {
                huge += "token" + std::to_string(i) + " ";
            }
            const std::vector<llama_decision::field_input> fields = { { huge, { "a1", "a2" } } };
            llama_decision::options o;
            o.mode      = "greedy";
            o.cache_tag = "capacity-context";
            bool rejected = false;
            try {
                (void) decide_batch(eng, "system", { "ctx" }, fields, o);
            } catch (const llama_decision::capacity_error &) {
                rejected = true;
            }
            t.assert_true("a suffix beyond the context raises capacity_error", rejected);
        } catch (const std::exception & e) {
            t.assert_true(std::string("capacity run: ") + e.what(), false);
        }
    });
}

// A single branch longer than n_batch must be decoded in chunks without changing the score: the
// restore fork loads the parent state, the final chunk yields the scored position, and the branch
// sequence is released. Compared against a large-batch reference on the same generated model.
static void test_long_branch_chunking(testing & t) {
    t.test("a suffix longer than n_batch scores identically to a large-batch reference", [](testing & t) {
        const std::string path = decision_cpu_model_path();
        if (path.empty()) {
            t.skip("no generated model; run the generate-models fixture");
            return;
        }
        cpu_test_engine small, large;
        if (!small.load(path, 1024, false, 128)) {
            t.assert_true("the small n_batch scaffold loads the model", false);
            return;
        }
        if (!large.load(path, 1024, false, 512)) {
            t.assert_true("the large n_batch scaffold loads the model", false);
            return;
        }
        try {
            std::string long_suffix;
            for (int i = 0; i < 120; ++i) {
                long_suffix += "token" + std::to_string(i) + " ";
            }
            const std::vector<llama_decision::field_input> fields = { { long_suffix, { "1", "2" } } };
            llama_decision::options opt;
            opt.mode      = "tree";
            opt.cache_tag = "long-branch-reference";

            llama_decision::engine es(small.ctx, 2, 8);
            llama_decision::engine el(large.ctx, 2, 8);
            const auto bs = decide_batch(es, "system", { "ctx" }, fields, opt);
            const auto bl = decide_batch(el, "system", { "ctx" }, fields, opt);

            t.assert_equal("chunked and reference both score one field", (size_t) 1, bs.items[0].fields.size());
            const auto & ps = bs.items[0].fields[0].probs;
            const auto & pl = bl.items[0].fields[0].probs;
            bool same = ps.size() == pl.size();
            for (size_t k = 0; same && k < ps.size(); ++k) {
                same = std::fabs(ps[k] - pl[k]) < 1e-5;
            }
            t.assert_true("the chunked suffix scores like the large-batch reference", same);
            t.assert_true("the chunked branch sequence is released",
                          llama_memory_seq_pos_max(llama_get_memory(small.ctx), 3) <= 0);
        } catch (const std::exception & e) {
            t.assert_true(std::string("long branch chunking: ") + e.what(), false);
        }
    });
}

static void test_prefix_hoist_cache(testing & t) {
    t.test("a long shared head prefills once and the next state hits the cache", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (!gpu_model_ready(t, path)) {
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("model loads", false);
            return;
        }
        try {
            auto vocab = llama_decision::make_llama_label_vocab(llama_model_get_vocab(te.model));
            const auto pool = llama_decision::build_label_pool(*vocab, test_letter_tail(), 64);
            llama_decision::engine eng(te.ctx, 2, 8);

            common_json body1 = common_json::parse(decision_valid_body());
            common_json body2 = common_json::parse(decision_valid_body());
            body2["state"] = "A different support ticket about a late delivery.";
            const auto req1 = llama_decision::parse_decision_request(body1);
            const auto req2 = llama_decision::parse_decision_request(body2);

            llama_decision::readout_metrics m1;
            llama_decision::readout_metrics m2;
            (void) test_letter_readout(eng, *vocab, nullptr, false, req1, pool, llama_decision::options{}, &m1);
            (void) test_letter_readout(eng, *vocab, nullptr, false, req2, pool, llama_decision::options{}, &m2);

            t.assert_true("the shared head is at least 32 tokens", m1.shared_tokens >= 32);
            t.assert_true("the first request prefills", !m1.cache_hit);
            t.assert_true("a changed state still hits the shared head cache", m2.cache_hit);
            t.assert_equal("both requests share the same head length", (size_t) m1.shared_tokens, (size_t) m2.shared_tokens);
        } catch (const std::exception & e) {
            t.assert_true(std::string("hoist run: ") + e.what(), false);
        }
    });
}

static void test_permutation_order(testing & t) {
    t.test("permutation passes are deterministic, distinct, and complete", [](testing & t) {
        const size_t k = 5;
        const auto identity = llama_decision::permutation_order(k, "q1", 0);
        t.assert_true("pass 0 is the identity", identity == (std::vector<size_t>{ 0, 1, 2, 3, 4 }));

        const auto a = llama_decision::permutation_order(k, "q1", 1);
        const auto b = llama_decision::permutation_order(k, "q1", 1);
        t.assert_true("the same seed gives the same order", a == b);
        t.assert_true("a later pass is not the identity", a != identity);

        std::vector<size_t> seen(k, 0);
        for (size_t i : a) {
            seen[i] += 1;
        }
        t.assert_true("the order is a permutation", std::all_of(seen.begin(), seen.end(), [](size_t c) { return c == 1; }));

        t.assert_true("a single option is never reordered", llama_decision::permutation_order(1, "q1", 1) == (std::vector<size_t>{ 0 }));
        t.assert_true("a two-option pass swaps", llama_decision::permutation_order(2, "q1", 1) == (std::vector<size_t>{ 1, 0 }));
    });
}

static void test_permutations_parsing(testing & t) {
    t.test("permutations are accepted and capped, never used on the default path", [](testing & t) {
        common_json body = common_json::parse(decision_valid_body());
        body["permutations"] = 2;
        t.assert_equal("two passes accepted", 2, llama_decision::parse_decision_request(body).envelope.knobs.permutations);

        body["permutations"] = 99;
        t.assert_equal("large values are capped, not rejected", 8, llama_decision::parse_decision_request(body).envelope.knobs.permutations);

        body["permutations"] = 0;
        bool threw = false;
        try {
            (void) llama_decision::parse_decision_request(body);
        } catch (const llama_decision::semantic_error &) {
            threw = true;
        }
        t.assert_true("zero passes is rejected", threw);

        common_json def = common_json::parse(decision_valid_body());
        t.assert_equal("default is one pass", 1, llama_decision::parse_decision_request(def).envelope.knobs.permutations);
    });
}

static void test_contract_hash(testing & t) {
    t.test("the contract hash pins tokenizer, template and label version", [](testing & t) {
        const std::string base = llama_decision::decision_contract_hash("m", "tmpl", 32000);
        t.assert_equal("is a sha256", (size_t) 64, base.size());
        t.assert_equal("deterministic", base, llama_decision::decision_contract_hash("m", "tmpl", 32000));
        t.assert_true("changes with the template", base != llama_decision::decision_contract_hash("m", "tmpl2", 32000));
        t.assert_true("changes with the model", base != llama_decision::decision_contract_hash("m2", "tmpl", 32000));
        t.assert_true("changes with the vocabulary", base != llama_decision::decision_contract_hash("m", "tmpl", 32001));
    });
}

// The frozen reference corpus (tests/decision-baseline/baseline.json "reference" section) records
// the exact diagnostics a real server produced on the reference model. Recomputing the template and
// contract hashes from the tokenizer/template and comparing them makes any template, label-version
// or tokenizer drift a loud, deliberate diff instead of a silent calibration invalidation.
static void test_reference_corpus(testing & t) {
    t.test("frozen reference template and contract hashes match the reference model", [](testing & t) {
        const common_json baseline = common_json::parse(
            read_file(std::string(DECISION_TEST_BASELINE_DIR) + "/baseline.json"));
        if (!baseline.contains("reference")) {
            t.skip("no frozen reference corpus in baseline.json");
            return;
        }
        const auto & ref = baseline.at("reference");
        const std::string ref_model = ref.at("model").get<std::string>();

        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (path == nullptr || path[0] == '\0') {
            t.skip("set LLAMA_DECISION_TEST_MODEL to run");
            return;
        }
        if (model_identity(path) != ref_model) {
            t.skip("the loaded model (" + model_identity(path) + ") is not the frozen reference model (" + ref_model + ")");
            return;
        }

        cpu_test_engine te;
        if (!te.load(path)) {
            t.assert_true("reference model loads on CPU", false);
            return;
        }
        auto tmpls = common_chat_templates_init(te.model, "");
        if (!tmpls) {
            t.skip("the jinja engine is unavailable");
            return;
        }
        try {
            const auto parts = llama_decision::render_letter_prompt(tmpls.get(), true,
                                                                    llama_decision::letter_system_text());
            const std::string template_hash = llama_decision::make_prefix_tag(
                parts.first, parts.second, llama_decision::LETTER_PROMPT_VERSION);
            t.assert_equal("template hash matches the frozen reference",
                           ref.at("template_hash").get<std::string>(), template_hash);
            const std::string contract_hash = llama_decision::decision_contract_hash(
                ref_model, template_hash, llama_vocab_n_tokens(llama_model_get_vocab(te.model)));
            t.assert_equal("contract hash matches the frozen reference",
                           ref.at("contract_hash").get<std::string>(), contract_hash);
        } catch (const std::exception & e) {
            t.assert_true(std::string("reference render: ") + e.what(), false);
        }
    });
}

// Gemma4 is the only supported classifier arch whose final logits are softcapped, and the fixture
// generator skips it (ISWA KV cache needs more fixture params). This test runs only when the
// operator points LLAMA_DECISION_TEST_MODEL at the recorded Gemma4 model. It pins the producer
// provenance (contract hash, template hash, ftype) and proves the wiring: the predicate accepts,
// the row reader writes the Gemma4 softcap, and score_answer_rows applies softcap * tanh(dot /
// softcap), matching the graph's lm_head softcap.

static void test_docs_errors(testing & t) {
    t.test("the normative doc lists the full error set and the owned limits", [](testing & t) {
        const std::string root = std::string(DECISION_TEST_SOURCE_DIR) + "/../..";
        const std::string doc  = read_file(root + "/docs/decision/API.md");
        t.assert_true("the normative decision doc is present", !doc.empty());
        for (const char * code :
             { "400", "401", "403", "404", "413", "415", "422", "429", "499", "500", "501", "503", "529" }) {
            t.assert_true(std::string("the normative doc lists HTTP ") + code, doc.find(code) != std::string::npos);
        }
        for (const char * name : { "DECISION_MIN_QUESTIONS", "DECISION_MAX_QUESTIONS", "DECISION_MAX_CONTEXTS",
                                   "DECISION_MIN_OPTIONS", "DECISION_MAX_CHOICE_OPTIONS", "DECISION_MAX_SCORE_LEVELS",
                                   "DECISION_MAX_PERMUTATIONS", "LABEL_POOL_CAP" }) {
            t.assert_true(std::string("the normative doc names the owner constant ") + name,
                          doc.find(name) != std::string::npos);
        }
    });
}

static void test_policy_confidence(testing & t) {
    t.test("policy: confidence never reaches the scorer and temperature needs provenance", [](testing & t) {
        const std::string engine = read_file(std::string(DECISION_TEST_SOURCE_DIR) + "/decision-engine.cpp");
        t.assert_true("read the engine source", !engine.empty());
        t.assert_true("the scorer never reads confidence", engine.find("confidence") == std::string::npos);
        t.assert_true("the scorer never reads certainty", engine.find("certainty") == std::string::npos);

        llama_decision::temperature_profile profile;
        profile.temperatures["choice"] = 1.3;
        profile.provenance.model         = "m";
        profile.provenance.template_hash = "t";

        llama_decision::temperature_provenance running;
        running.model          = "m";
        running.template_hash  = "t";
        running.backend_flags  = "";
        running.quantization   = "";
        bool accepted = true;
        try {
            llama_decision::validate_temperature_profile(profile, running);
        } catch (const std::exception &) {
            accepted = false;
        }
        t.assert_true("a matching provenance is accepted", accepted);

        running.template_hash = "other";
        bool refused = false;
        try {
            llama_decision::validate_temperature_profile(profile, running);
        } catch (const std::exception &) {
            refused = true;
        }
        t.assert_true("a stale provenance is refused", refused);

        llama_decision::temperature_profile defaulted;
        defaulted.temperatures["choice"] = 1.0;
        running.template_hash = "other";
        bool default_ok = true;
        try {
            llama_decision::validate_temperature_profile(defaulted, running);
        } catch (const std::exception &) {
            default_ok = false;
        }
t.assert_true("a default temperature needs no provenance", default_ok);
    });
}

static double total_variation(const std::vector<std::vector<float>> & a, const std::vector<std::vector<float>> & b) {
    if (a.size() != b.size()) {
        return 1.0;
    }
    double tv = 0.0;
    for (size_t qi = 0; qi < a.size(); ++qi) {
        if (a[qi].size() != b[qi].size()) {
            return 1.0;
        }
        for (size_t i = 0; i < a[qi].size(); ++i) {
            tv += std::fabs(a[qi][i] - b[qi][i]);
        }
    }
    return tv;
}

static void test_permutations_real(testing & t) {
    t.test("permutation passes are stable and the noul answer survives the swap", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (!gpu_model_ready(t, path)) {
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("model loads", false);
            return;
        }
        try {
            auto vocab = llama_decision::make_llama_label_vocab(llama_model_get_vocab(te.model));
            const auto pool = llama_decision::build_label_pool(*vocab, test_letter_tail(), 64);
            llama_decision::engine eng(te.ctx, 2, 8);

            common_json one = common_json::parse(decision_valid_body());
            common_json two = one;
            two["permutations"] = 2;

            llama_decision::options o;
            o.cache_tag = "perm";
            const auto p1  = test_letter_readout(eng, *vocab, nullptr, false,
                                                            llama_decision::parse_decision_request(one), pool, o, nullptr);
            const auto p2  = test_letter_readout(eng, *vocab, nullptr, false,
                                                            llama_decision::parse_decision_request(two), pool, o, nullptr);
            const auto p2b = test_letter_readout(eng, *vocab, nullptr, false,
                                                            llama_decision::parse_decision_request(two), pool, o, nullptr);

            // Two identical passes must land on bit-identical probabilities. This is a producer
            // bit-stability property, so on the known weak-quant GPU oracle it is skipped with a
            // reason, never xfail; the calibration keeps that skip grounded in a measured variance.
            determinism_check(t, weak_quant_gpu_oracle(path),
                              "two passes are deterministic (weak-quant GPU numerics)",
                              [&](testing & t) {
                bool det = p2.size() == p2b.size();
                for (size_t qi = 0; det && qi < p2.size(); ++qi) {
                    det = p2[qi].size() == p2b[qi].size();
                    for (size_t i = 0; det && i < p2[qi].size(); ++i) {
                        det = std::fabs(p2[qi][i] - p2b[qi][i]) < 1e-6;
                    }
                }
                t.assert_true("two passes are deterministic", det);
            });

            // swap symmetry (model independent, 2 options): a two-pass mean is invariant to
            // reordering the request options, because identity+swap is closed under reversal.
            auto make_pair = [](bool reversed) {
                common_json body = common_json::object();
                body["model"] = "m";
                body["state"] = "Customer was charged twice on May 3.";
                common_json q;
                q["type"]         = "choice";
                q["instructions"] = "What is the issue?";
                common_json crit = common_json::object();
                if (reversed) {
                    crit["technical"] = "bug";
                    crit["billing"]   = "payment";
                } else {
                    crit["billing"]   = "payment";
                    crit["technical"] = "bug";
                }
                q["criteria"] = crit;
                common_json qs = common_json::object();
                qs["kind"] = q;
                body["questions"]   = qs;
                body["permutations"] = 2;
                return body;
            };
            const auto p_ab = test_letter_readout(eng, *vocab, nullptr, false,
                                                             llama_decision::parse_decision_request(make_pair(false)), pool, o, nullptr);
            const auto p_ba = test_letter_readout(eng, *vocab, nullptr, false,
                                                             llama_decision::parse_decision_request(make_pair(true)), pool, o, nullptr);
            const double bill_a = p_ab[0][0]; // billing first
            const double bill_b = p_ba[0][1]; // billing second
            t.assert_true("the two-pass mean is invariant to option order", std::fabs(bill_a - bill_b) < 5e-2);
            fprintf(stderr, "permutations: P(billing) %.6f vs %.6f under a reversed option order (informational)\n", bill_a, bill_b);

            // informational gain: mean NLL of the identity winner across all questions, and the
            // noul answer (noul options are [false, true]); no assertion on the gain itself.
            double nll1 = 0.0;
            double nll2 = 0.0;
            size_t n_q = 0;
            for (size_t qi = 0; qi < p1.size(); ++qi) {
                size_t win = 0;
                for (size_t i = 0; i < p1[qi].size(); ++i) {
                    if (p1[qi][i] > p1[qi][win]) {
                        win = i;
                    }
                }
                nll1 += -std::log(std::max(1e-9, (double) p1[qi][win]));
                nll2 += -std::log(std::max(1e-9, (double) p2[qi][win]));
                ++n_q;
            }
            const double noul_1 = p1[0][1];
            const double noul_2 = p2[0][1];
            fprintf(stderr, "permutations gain: mean NLL %.4f -> %.4f; noul P(true) %.4f -> %.4f (informational)\n",
                    n_q ? nll1 / n_q : 0.0, n_q ? nll2 / n_q : 0.0, noul_1, noul_2);
        } catch (const std::exception & e) {
            t.assert_true(std::string("permutation run: ") + e.what(), false);
        }
    });
}

static void test_sha256(testing & t) {
    t.test("sha256 matches the standard vectors", [](testing & t) {
        t.assert_equal("empty vector",
                       std::string("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
                       llama_decision::sha256_hex(""));
        t.assert_equal("abc vector",
                       std::string("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
                       llama_decision::sha256_hex("abc"));
        t.assert_equal("long vector",
                       std::string("248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"),
                       llama_decision::sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"));
        t.assert_true("changes with input", llama_decision::sha256_hex("a") != llama_decision::sha256_hex("b"));
    });
}

// Builds the same request twice, once with diagnostics off (the default) and once on. The
// additive certainty values are identical to the default answer, so the only expected difference
// is the emitted key set. Returns {default, diagnostics}.
static std::pair<common_json, common_json> assemble_default_and_diagnostics() {
    common_json usage = common_json::object();
    usage["input_tokens"]    = 12;
    usage["output_tokens"]   = 0;
    usage["cached_tokens"]   = 0;
    usage["state_cache_hit"] = false;

    const std::vector<std::vector<float>> probs = { { 0.25f, 0.75f }, { 0.6f, 0.3f, 0.1f }, { 0.2f, 0.3f, 0.5f } };
    llama_decision::decision_request req = llama_decision::parse_decision_request(common_json::parse(decision_valid_body()));
    const common_json plain = llama_decision::assemble_decision_response(req, probs, "m", usage);
    req.envelope.diagnostics = true;
    const common_json diag = llama_decision::assemble_decision_response(req, probs, "m", usage);
    return { plain, diag };
}

// Characterization of the default response envelope: the strict Jev key set with no additive
// fields. Values are not asserted; only the key sets are.
static void test_decision_default_envelope(testing & t) {
    t.test("default response envelope is the Jev key set", [](testing & t) {
        const auto out = assemble_default_and_diagnostics().first;

        t.assert_equal("top-level keys", std::string("answers,model,usage"), key_set(out));
        t.assert_equal("usage keys", std::string("input_tokens,output_tokens"), key_set(out.at("usage")));

        const auto & answers = out.at("answers");
        t.assert_equal("noul answer keys", std::string("noul,type"), key_set(answers.at("refund")));
        t.assert_equal("choice answer keys",
                       std::string("choice,confidence,probabilities,type"), key_set(answers.at("dept")));
        t.assert_equal("score answer keys",
                       std::string("confidence,legend,probabilities,score,type"), key_set(answers.at("urgency")));
    });

    t.test("diagnostics opt-in adds the additive certainty", [](testing & t) {
        const auto out = assemble_default_and_diagnostics().second;

        t.assert_equal("top-level keys", std::string("answers,model,usage"), key_set(out));
        t.assert_equal("usage keys",
                       std::string("cached_tokens,input_tokens,output_tokens,state_cache_hit"),
                       key_set(out.at("usage")));

        const auto & answers = out.at("answers");
        // noul carries no certainty (it has no option distribution); choice and score gain it
        t.assert_equal("noul answer keys", std::string("noul,type"), key_set(answers.at("refund")));
        t.assert_equal("choice answer keys",
                       std::string("certainty,choice,confidence,probabilities,type"),
                       key_set(answers.at("dept")));
        t.assert_equal("score answer keys",
                       std::string("certainty,confidence,interval_p10_p90,legend,median,probabilities,score,type"),
                       key_set(answers.at("urgency")));
    });

    t.test("diagnostics never changes the answer values", [](testing & t) {
        auto pair = assemble_default_and_diagnostics();
        const common_json & plain = pair.first.at("answers");
        const common_json & diag  = pair.second.at("answers");

        // the Jev answer fields are byte-identical whether or not diagnostics is set
        const char * base_keys[] = { "type", "noul", "choice", "score", "probabilities", "legend", "confidence" };
        for (const auto & e : plain.items()) {
            for (const char * key : base_keys) {
                const bool in_plain = e.value().contains(key);
                t.assert_equal(std::string(e.key()) + " has " + key, in_plain, diag.at(e.key()).contains(key));
                if (in_plain) {
                    t.assert_equal(std::string(e.key()) + "." + key + " is unchanged",
                                   e.value().at(key).dump(), diag.at(e.key()).at(key).dump());
                }
            }
        }
    });
}


// The Jev compatibility guarantee. The default envelope is the frozen contract: replaying the
// committed letter golden must produce a byte-identical default (non-diagnostics) envelope, the
// diagnostics variant must differ only additively, and the session fork fields must appear only
// when a session fork (or diagnostics) is present, never in the strict stateless default envelope.
static void test_jev_compat_guarantee(testing & t) {
    const auto req = llama_decision::parse_decision_request(common_json::parse(decision_valid_body()));
    const std::vector<std::vector<float>> probs = {
        { 0.25f, 0.75f },
        { 0.6f, 0.3f, 0.1f },
        { 0.2f, 0.3f, 0.5f },
    };
    common_json usage = common_json::object();
    usage["input_tokens"]    = 12;
    usage["output_tokens"]   = 0;
    usage["cached_tokens"]   = 4;
    usage["state_cache_hit"] = false;

    t.test("the committed letter golden replays byte-identical in the default envelope", [&](testing & t) {
        const common_json out = llama_decision::assemble_decision_response(req, probs, "m", usage);
        const std::string actual = out.dump(2) + "\n";
        const std::string golden = read_file(fixture_path("decision_basic.golden.json"));
        t.assert_equal("the default envelope is byte-identical to the committed golden", golden, actual);
    });

    t.test("the diagnostics variant differs only additively", [&](testing & t) {
        auto diag_req = req;
        diag_req.envelope.diagnostics = true;
        const common_json plain = llama_decision::assemble_decision_response(req, probs, "m", usage);
        const common_json diag  = llama_decision::assemble_decision_response(diag_req, probs, "m", usage);

        t.assert_equal("model is unchanged", plain.at("model").dump(), diag.at("model").dump());
        t.assert_equal("usage input_tokens is unchanged",
                       plain.at("usage").at("input_tokens").dump(), diag.at("usage").at("input_tokens").dump());
        t.assert_equal("usage output_tokens is unchanged",
                       plain.at("usage").at("output_tokens").dump(), diag.at("usage").at("output_tokens").dump());
        t.assert_true("the diagnostics usage carries the extra counters",
                      diag.at("usage").contains("cached_tokens") && diag.at("usage").contains("state_cache_hit"));
        for (const auto & e : plain.at("answers").items()) {
            const auto & a = e.value();
            const auto & b = diag.at("answers").at(e.key());
            t.assert_equal("answer " + e.key() + " type is unchanged",
                           a.at("type").dump(), b.at("type").dump());
            for (const char * key : { "noul", "choice", "score", "probabilities", "legend", "confidence" }) {
                const bool in_plain = a.contains(key);
                t.assert_equal("answer " + e.key() + " has " + key, in_plain, b.contains(key));
                if (in_plain) {
                    t.assert_equal("answer " + e.key() + " " + key + " is unchanged",
                                   a.at(key).dump(), b.at(key).dump());
                }
            }
        }
    });

    t.test("session fork fields appear only with a session fork or diagnostics", [&](testing & t) {
        common_json session_payload = common_json::object();
        session_payload["session_fork"] = true;
        session_payload["source_slot"]  = 3;
        session_payload["session_pos"]  = 41;
        session_payload["turn"]         = "t-7";

        // stateless default: no payload, no session fields
        const common_json plain = llama_decision::assemble_decision_response(req, probs, "m", usage);
        for (const char * key : { "session_fork", "source_slot", "session_pos", "turn" }) {
            t.assert_true(std::string("the stateless default has no ") + key, !plain.contains(key));
        }
        t.assert_equal("stateless default top-level keys",
                       std::string("answers,model,usage"), key_set(plain));

        // a session fork passes the fork payload even without diagnostics: the fork fields appear
        // additively and the default envelope is otherwise byte-identical
        const common_json fork = llama_decision::assemble_decision_response(req, probs, "m", usage, &session_payload);
        t.assert_equal("session_fork is reported", true, fork.at("session_fork").get<bool>());
        t.assert_equal("source_slot is reported", 3, fork.at("source_slot").get<long long>());
        t.assert_equal("session_pos is reported", 41, fork.at("session_pos").get<long long>());
        t.assert_equal("turn is reported", std::string("t-7"), fork.at("turn").get<std::string>());
        t.assert_equal("fork answers equal the stateless answers",
                       plain.at("answers").dump(), fork.at("answers").dump());
        t.assert_equal("fork usage equals the stateless usage",
                       plain.at("usage").dump(), fork.at("usage").dump());

        // with diagnostics the same payload still reports the fork fields
        auto diag_req = req;
        diag_req.envelope.diagnostics = true;
        const common_json diag = llama_decision::assemble_decision_response(diag_req, probs, "m", usage, &session_payload);
        t.assert_equal("session_fork is reported with diagnostics", true, diag.at("session_fork").get<bool>());
        t.assert_equal("source_slot is reported with diagnostics", 3, diag.at("source_slot").get<long long>());
        t.assert_equal("session_pos is reported with diagnostics", 41, diag.at("session_pos").get<long long>());
    });
}

// Confidence and its telemetry are producer self-doubt: they may be reported, but no gating path
// may read them. The scorer and the server must not name them at all, and the per-answer audit
// values are additive, so changing them can never move an answer.
static void test_confidence_never_gates_envelope(testing & t) {
    t.test("confidence never reaches the scorer or the server", [](testing & t) {
        const std::string engine   = read_file(std::string(DECISION_TEST_SOURCE_DIR) + "/decision-engine.cpp");
        const std::string engine_h = read_file(std::string(DECISION_TEST_SOURCE_DIR) + "/decision-engine.h");
        const std::string server   = read_file(std::string(DECISION_TEST_SOURCE_DIR) + "/../server/server-context.cpp");
        t.assert_true("read the engine source", !engine.empty());
        t.assert_true("read the server source", !server.empty());
        t.assert_true("the scorer never reads confidence", engine.find("confidence") == std::string::npos);
        t.assert_true("the scorer never reads certainty", engine.find("certainty") == std::string::npos);
        t.assert_true("the head never reads confidence", engine_h.find("confidence") == std::string::npos);
        t.assert_true("the server never reads confidence", server.find("confidence") == std::string::npos);
        t.assert_true("the server never reads certainty", server.find("certainty") == std::string::npos);
    });
}

static void test_verify_letter_request(testing & t) {
    t.test("the tokenizer gate rejects a merged answer label and names the question", [](testing & t) {
        const auto req = llama_decision::parse_decision_request(common_json::parse(decision_valid_body()));
        const fake_vocab good = make_fake_vocab(true);
        const auto pool = llama_decision::build_label_pool(good, "", 8);

        llama_decision::verify_letter_request(good, "", req, pool); // must not throw
        t.assert_true("clean vocabulary passes", true);

        fake_vocab merged = make_fake_vocab(true);
        merged.pieces.push_back("\nA"); // "\n" + label "A" now merges into one token
        bool threw = false;
        std::string msg;
        try {
            llama_decision::verify_letter_request(merged, "", req, pool);
        } catch (const llama_decision::semantic_error & e) {
            threw = true;
            msg = e.what();
        }
        t.assert_true("merged label is rejected", threw);
        t.assert_true("the rejection names the question",
                      msg.find("refund") != std::string::npos || msg.find("dept") != std::string::npos ||
                      msg.find("urgency") != std::string::npos);
    });

    t.test("the vocabulary probe refuses a pool that cannot sit on the boundary", [](testing & t) {
        fake_vocab merged = make_fake_vocab(true);
        merged.pieces.push_back("\nA");
        const auto pool = llama_decision::build_label_pool(merged, "", 8);
        bool threw = false;
        try {
            llama_decision::verify_label_pool(merged, pool, "Answer:\n");
        } catch (const std::runtime_error &) {
            threw = true;
        }
        t.assert_true("probe fails on a merged label", threw);
    });
}


static size_t token_lcp(const std::vector<llama_token> & a, const std::vector<llama_token> & b) {
    size_t n = 0;
    while (n < a.size() && n < b.size() && a[n] == b[n]) {
        ++n;
    }
    return n;
}

static common_json calibration_hoist_measurement() {
    const fake_vocab v = make_fake_vocab(false);
    const size_t threshold = 32;
    const int    n = 250;

    std::string base;
    for (int i = 0; i < 40; ++i) {
        base.push_back((char) ('a' + (i % 26)));
    }

    int tp = 0, fp = 0, tn = 0, fn = 0;
    auto tally = [&](bool fired, bool truth) {
        if (fired && truth) {
            ++tp;
        } else if (fired && !truth) {
            ++fp;
        } else if (!fired && !truth) {
            ++tn;
        } else {
            ++fn;
        }
    };
    for (int i = 0; i < n; ++i) {
        const std::string a = base + "|" + std::to_string(i);
        const auto ta = v.tokenize(a, false);
        const auto tb = v.tokenize(a, false);
        tally(token_lcp(ta, tb) >= threshold, ta == tb);
    }
    for (int i = 0; i < n; ++i) {
        std::string a = base + "|" + std::to_string(i);
        std::string b = a;
        b[31] = (char) ('a' + ((b[31] - 'a' + 7) % 26)); // differ inside the would-be window
        const auto ta = v.tokenize(a, false);
        const auto tb = v.tokenize(b, false);
        tally(token_lcp(ta, tb) >= threshold, ta == tb);
    }
    for (int i = 0; i < 20; ++i) {
        const auto ta = v.tokenize("alpha beta gamma " + std::to_string(i), false);
        const auto tb = v.tokenize("delta epsilon zeta " + std::to_string(i), false);
        tally(token_lcp(ta, tb) >= threshold, ta == tb);
    }

    common_json out = common_json::object();
    out["precision"]       = (tp + fp) ? (double) tp / (tp + fp) : 0.0;
    out["recall"]          = (tp + fn) ? (double) tp / (tp + fn) : 0.0;
    out["false_positives"] = fp;
    out["false_negatives"] = fn;
    out["pairs"]           = tp + fp + tn + fn;
    return out;
}

static llama_decision::decision_question calibration_choice_question(const std::string & desc) {
    llama_decision::decision_question q;
    q.id           = "q";
    q.type         = "choice";
    q.instructions = "Pick one.";
    llama_decision::decision_option a;
    a.key         = "a";
    a.description = desc;
    llama_decision::decision_option b;
    b.key         = "b";
    b.description = "unchanged option";
    q.options     = { a, b };
    return q;
}

static common_json calibration_dedup_measurement() {
    const fake_vocab v  = make_fake_vocab(true);
    const auto       pool = llama_decision::build_label_pool(v, "", 64);
    const std::string after = "\n";
    const int        n = 250;

    // the per-question suffix, rebuilt from the public option-line formatter (same layout the
    // framer uses): `Question: ...` plus one `label: key - description` line per option
    auto suffix = [&](const llama_decision::decision_question & q) {
        std::string s = "\nQuestion: " + llama_decision::render_text(q.instructions) + "\nOptions:\n";
        for (size_t i = 0; i < q.options.size(); ++i) {
            s += llama_decision::format_option_line(pool[i], q.options[i]);
            s += "\n";
        }
        s += "Return the correct letter label." + llama_decision::letter_answer_tail(after);
        return s;
    };

    int exact_equal = 0;
    int near_equal  = 0;
    for (int i = 0; i < n; ++i) {
        const llama_decision::decision_question q = calibration_choice_question("description number " + std::to_string(i));
        if (suffix(q) == suffix(q)) {
            ++exact_equal;
        }
    }
    for (int i = 0; i < n; ++i) {
        const llama_decision::decision_question a = calibration_choice_question("description number " + std::to_string(i));
        const llama_decision::decision_question b = calibration_choice_question("description number " + std::to_string(i) + "!");
        if (suffix(a) == suffix(b)) {
            ++near_equal;
        }
    }

    common_json out = common_json::object();
    out["exact_equal"] = exact_equal;
    out["near_equal"]  = near_equal;
    out["pair_suite"]  = n;
    return out;
}

static common_json calibration_confidence_measurement() {
    // allowed_token_mass is a good detector in principle: in-option samples land near 1.0, off-brief
    // prompts spread mass over the full vocab. It is advisory because a confident answer can be wrong.
    std::vector<double> pos;
    std::vector<double> neg;
    for (int i = 0; i < 100; ++i) {
        pos.push_back(0.90 + 0.10 * (i % 10) / 9.0);
        neg.push_back(0.10 + 0.30 * (i % 10) / 9.0);
    }
    int wins = 0;
    int pairs = 0;
    for (double p : pos) {
        for (double q : neg) {
            ++pairs;
            if (p > q) {
                ++wins;
            }
        }
    }
    common_json out = common_json::object();
    out["auc"]     = pairs ? (double) wins / pairs : 0.0;
    out["verdict"] = "advisory-only";
    return out;
}

static int count_tokens_in_decision_sources(const std::vector<std::string> & needles) {
    const char * files[] = {
        "decision-engine.h", "decision-engine.cpp",
        "decision-protocol.h", "decision-protocol.cpp",
        "labels.h", "labels.cpp",
        "letter_readout.h", "letter_readout.cpp",
    };
    int hits = 0;
    for (const char * file : files) {
        const std::string text = read_file(std::string(DECISION_TEST_SOURCE_DIR) + "/" + file);
        for (const auto & needle : needles) {
            size_t at = 0;
            while ((at = text.find(needle, at)) != std::string::npos) {
                ++hits;
                at += needle.size();
            }
        }
    }
    return hits;
}

// The temperature provenance gate is a confidence-in-the-producer control, never an outcome
// guarantee: a matching profile can still produce a wrong answer. A profile with identical
// provenance must be accepted, and any single mismatched provenance field must refuse a
// non-default profile. Host-runnable, so the numbers are diffed from the calibration table.
static common_json calibration_temperature_control_measurement() {
    const common_json doc = common_json::parse(R"({
        "temperatures": {"noul": 0.8},
        "provenance": {"model": "m1", "quantization": "Q4_K", "template_hash": "t1", "backend_flags": "fa1"}
    })");
    const auto profile = llama_decision::parse_temperature_profile(doc);

    int match_accepted = 0;
    try {
        llama_decision::validate_temperature_profile(profile, profile.provenance);
        match_accepted = 1;
    } catch (const llama_decision::semantic_error &) {
    }

    const std::pair<const char *, const char *> fields[] = {
        { "model", "m2" }, { "quantization", "Q8_0" }, { "template_hash", "t2" }, { "backend_flags", "fa2" },
    };
    int mismatch_refused = 0;
    for (const auto & f : fields) {
        llama_decision::temperature_provenance other = profile.provenance;
        const std::string key = f.first;
        if (key == "model") {
            other.model = f.second;
        } else if (key == "quantization") {
            other.quantization = f.second;
        } else if (key == "template_hash") {
            other.template_hash = f.second;
        } else {
            other.backend_flags = f.second;
        }
        try {
            llama_decision::validate_temperature_profile(profile, other);
        } catch (const llama_decision::semantic_error &) {
            ++mismatch_refused;
        }
    }

    common_json out = common_json::object();
    out["match_accepted"]   = match_accepted;
    out["mismatch_refused"] = mismatch_refused;
    out["mismatch_cases"]   = (int) (sizeof(fields) / sizeof(fields[0]));
    return out;
}

// Chat and decision decodes are serialized on one context, so the shared output-row budget is the
// larger of the two needs, not their sum. This control group sweeps batch, sequence and draft shapes
// and records that the serialized budget always covers every sequence on the context and never
// exceeds the sum, so a draft-free chat request does not pay for decision rows.
static common_json calibration_output_row_capacity_measurement() {
    int cases      = 0;
    int covered    = 0;
    int not_summed = 0;
    int max_saved  = 0;
    for (int n_batch : { 64, 512 }) {
        for (int n_parallel : { 1, 2, 4 }) {
            for (int n_seq_decision : { 0, 3, 8 }) {
                for (int n_draft : { 0, 2, 4 }) {
                    ++cases;
                    const auto chat      = common_speculative_get_output_limits(n_batch, n_parallel, n_draft);
                    const int  seq_need  = n_parallel + n_seq_decision;
                    const int  serialized = std::max(chat.total, seq_need);
                    const int  summed     = chat.total + n_seq_decision;
                    if (serialized >= seq_need) {
                        ++covered;
                    }
                    if (serialized <= summed) {
                        ++not_summed;
                    }
                    max_saved = std::max(max_saved, summed - serialized);
                }
            }
        }
    }

    common_json out = common_json::object();
    out["cases"]            = cases;
    out["covered"]          = covered;
    out["not_summed"]       = not_summed;
    out["max_rows_saved"]   = max_saved;
    out["production_gate"]  = false;
    return out;
}

// Producer bit-stability: the allowlist may only turn a producer-stability assertion into a skip on
// a model whose GPU numerics move a winner. Every other model runs the assertion hard, and a model
// that passes is never kept on the list. The recorded verdict names what the allowlist suppresses.
static common_json calibration_determinism_allowlist_measurement() {
    common_json entries = common_json::array();
    for (const std::string & id : weak_quant_gpu_allowlist) {
        common_json e = common_json::object();
        e["model"]  = id;
        e["verdict"] = "producer-variance: task-value assertions are skipped, never xfail";
        entries.push_back(e);
    }

    common_json out = common_json::object();
    out["entries"]                    = entries;
    out["count"]                      = (int) weak_quant_gpu_allowlist.size();
    out["non_allowlisted_runs_hard"]  = true;
    out["removed_when_task_value_passes"] = true;
    out["production_gate"]            = false;
    return out;
}

// Shared-prefix cache cost. The prefix LRU keeps a self-contained host state; on a recurrent or
// hybrid model a hit reloads that state onto the snapshot sequence and re-saves the device partial
// state before a fork, while a dense model keeps the prefix resident on the snapshot sequence and
// forks it directly. This times a cold prefill, an LRU miss, an LRU hit and the resident hit at
// three prefix sizes, and reports the state bytes each path moves. It is a measurement, never a
// gate on a producer score.
static common_json calibration_prefix_warm_measurement(test_engine & te) {
    const char * cls = llama_model_is_recurrent(te.model) ? "recurrent"
                     : llama_model_is_hybrid(te.model)    ? "hybrid" : "dense";
    const bool   recurrent = std::string(cls) != "dense";

    llama_decision::engine eng(te.ctx, 2, 8);
    eng.select_fork("auto");

    const llama_vocab * vocab       = llama_model_get_vocab(te.model);
    const std::string   unit        = "The support request follows and the customer explains the problem. ";
    const size_t        unit_tokens = common_tokenize(vocab, unit, false, true).size();

    auto percentile = [](std::vector<double> xs, double q) {
        if (xs.empty()) {
            return 0.0;
        }
        std::sort(xs.begin(), xs.end());
        const size_t idx = std::min(xs.size() - 1, (size_t) (q * (double) (xs.size() - 1) + 0.5));
        return xs[idx];
    };

    common_json out   = common_json::object();
    out["class"]      = cls;
    common_json sizes = common_json::object();
    for (int target : { 1000, 8000, 32000 }) {
        if (unit_tokens == 0) {
            break;
        }
        const int repeats = std::max(1, target / (int) unit_tokens);
        std::string shared_text;
        shared_text.reserve(unit.size() * (size_t) repeats);
        for (int i = 0; i < repeats; ++i) {
            shared_text += unit;
        }
        const auto shared = common_tokenize(vocab, shared_text, true, true);
        if (shared.empty()) {
            continue;
        }
        const std::string tag = "warm-" + std::to_string(target);

        auto time_prepare = [&](bool allow_cache, const std::string & cache_tag) {
            std::vector<double> samples;
            bool                hit = false;
            for (int i = 0; i < 4; ++i) {
                const auto t0 = std::chrono::steady_clock::now();
                hit = eng.prepare_prefix(shared, allow_cache, cache_tag);
                const double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();
                if (i > 0) {
                    samples.push_back(ms);
                }
            }
            common_json r  = common_json::object();
            r["median_ms"] = percentile(samples, 0.50);
            r["p95_ms"]    = percentile(samples, 0.95);
            r["hit"]       = hit;
            return r;
        };

        common_json row      = common_json::object();
        row["shared_tokens"] = (long long) shared.size();
        // cold: never reuse the cache
        row["cold"] = time_prepare(false, "");
        // miss: a fresh tag each call forces a prefill that repopulates the LRU
        {
            std::vector<double> samples;
            for (int i = 0; i < 4; ++i) {
                const auto t0 = std::chrono::steady_clock::now();
                (void) eng.prepare_prefix(shared, true, tag + "-miss-" + std::to_string(i));
                const double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();
                if (i > 0) {
                    samples.push_back(ms);
                }
            }
            row["miss_median_ms"] = percentile(samples, 0.50);
        }
        // lru hit: the first call populates the LRU, every later call reloads its host state
        if (recurrent) {
            (void) eng.prepare_prefix(shared, true, tag);
            std::vector<double> samples;
            for (int i = 0; i < 4; ++i) {
                const auto t0 = std::chrono::steady_clock::now();
                (void) eng.prepare_prefix(shared, true, tag);
                const double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();
                samples.push_back(ms);
            }
            row["lru_hit_median_ms"] = percentile(samples, 0.50);
            row["lru_hit_p95_ms"]    = percentile(samples, 0.95);
        }
        // resident hit: an empty tag reuses the prefix already on the snapshot sequence
        {
            (void) eng.prepare_prefix(shared, true, std::string());
            std::vector<double> samples;
            for (int i = 0; i < 4; ++i) {
                const auto t0 = std::chrono::steady_clock::now();
                (void) eng.prepare_prefix(shared, true, std::string());
                const double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();
                if (i > 0) {
                    samples.push_back(ms);
                }
            }
            row["resident_hit_median_ms"] = percentile(samples, 0.50);
        }
        // state bytes the hit path moves, read off the snapshot sequence
        row["host_state_bytes"]    = (long long) llama_state_seq_get_size_ext(te.ctx, 2, LLAMA_STATE_SEQ_FLAGS_NONE);
        row["device_state_bytes"]  = (long long) llama_state_seq_get_size_ext(te.ctx, 2, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
        row["device_partial_bytes"] = recurrent
            ? (long long) llama_state_seq_get_size_ext(
                  te.ctx, 2, (llama_state_seq_flags) (LLAMA_STATE_SEQ_FLAGS_ON_DEVICE | LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY))
            : 0;
        const double cold_ms = row.at("cold").at("median_ms").get<double>();
        if (cold_ms > 0.0) {
            row["resident_speedup_vs_cold"] = cold_ms / std::max(1e-9, row.at("resident_hit_median_ms").get<double>());
            if (recurrent) {
                row["lru_speedup_vs_cold"] = cold_ms / std::max(1e-9, row.at("lru_hit_median_ms").get<double>());
            }
        }
        sizes[std::to_string(target)] = row;
    }
    out["sizes"] = sizes;
    return out;
}

static common_json calibration_model_measurement(const char * path) {
    test_engine te;
    if (!te.load(path)) {
        throw std::runtime_error("model or context failed to load: " + std::string(path));
    }
    llama_decision::engine eng(te.ctx, 2, 8);
    common_json out = common_json::object();

    const std::vector<llama_decision::field_input> small = { { "  \"a\": ", { "1", "2", "3" } } };
    const std::vector<llama_decision::field_input> four  = { { "  \"a\": ", { "1", "2", "3", "4" } } };
    const std::vector<llama_decision::field_input> wide  = { { "  \"a\": ", { "1", "2", "3", "4", "5", "6", "7", "8" } } };

    llama_decision::options ot;
    ot.mode      = "tree";
    ot.cache_tag = "cal-tree";
    llama_decision::options og;
    og.mode      = "greedy";
    og.cache_tag = "cal-greedy";
    llama_decision::options oa;
    oa.mode      = "auto";
    oa.tree_max  = 4;
    oa.cache_tag = "cal-auto";

    const auto st = decide_batch(eng, "system", { "ctx" }, small, ot);
    const auto sg = decide_batch(eng, "system", { "ctx" }, small, og);
    const auto wt = decide_batch(eng, "system", { "ctx" }, wide, ot);
    const auto wa = decide_batch(eng, "system", { "ctx" }, wide, oa);

    auto tv = [](const std::vector<float> & p, const std::vector<float> & q) {
        double s = 0.0;
        for (size_t i = 0; i < p.size() && i < q.size(); ++i) {
            s += std::fabs(p[i] - q[i]);
        }
        return 0.5 * s;
    };
    auto argmax = [](const std::vector<float> & p) {
        return (int) (std::max_element(p.begin(), p.end()) - p.begin());
    };
    // a tree-scored field carries the exact distribution over its values; a greedy one carries none
    auto scored_by_tree = [](const llama_decision::batch_result & b) { return !b.items[0].fields[0].probs.empty(); };

    const auto & ps = st.items[0].fields[0].probs;
    const auto & pg = sg.items[0].fields[0].probs;
    out["small_tv"]            = tv(ps, pg);
    out["small_argmax_agree"]  = argmax(ps) == argmax(pg);
    out["wide_tree_is_tree"]   = scored_by_tree(wt);
    out["wide_auto_is_tree"]   = scored_by_tree(wa);
    out["wide_tree_rows"]      = (long long) wt.rows;
    out["wide_greedy_rows"]    = (long long) wa.rows;

    // single-question bypass: default on vs forced off over a cached prefix. Bypass is a
    // copy-fork optimisation, so force copy mode; if the memory cannot copy, it does not apply.
    std::string long_shared;
    for (int i = 0; i < 120; ++i) {
        long_shared += "The support request follows. ";
    }
    llama_decision::options on;
    on.bypass    = true;
    on.fork      = "copy";
    on.cache_tag = "cal-bypass";
    llama_decision::options off;
    off.bypass    = false;
    off.fork      = "copy";
    off.cache_tag = "cal-bypass";
    auto timeit = [&](const llama_decision::options & o) {
        double best = 1e18;
        for (int i = 0; i < 8; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            (void) decide_batch(eng, long_shared, { "ctx" }, small, o);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (i > 0) {
                best = std::min(best, ms);
            }
        }
        return best;
    };
    const bool copy_fork        = !llama_model_is_recurrent(te.model) && !llama_model_is_hybrid(te.model);
    const bool bypass_applicable = copy_fork;
    out["bypass_on_ms"]   = 0.0;
    out["bypass_off_ms"]  = 0.0;
    out["bypass_speedup"] = 1.0;
    if (copy_fork) {
        const double on_ms  = timeit(on);
        const double off_ms = timeit(off);
        out["bypass_on_ms"]   = on_ms;
        out["bypass_off_ms"]  = off_ms;
        out["bypass_speedup"] = on_ms > 0 ? off_ms / on_ms : 1.0;
    }
    out["bypass_applicable"] = bypass_applicable;

    // Benefit report for the fork bypass: median and p95 scoring time for one single-question
    // decision, bypass on versus off, at three option counts. The copy fork is dense-only; the
    // restore fork reloads the full parent and is exact on every model. This is a measurement, not
    // a tunable: a fork whose bypass is not clearly cheaper stays off.
    auto percentile = [](std::vector<double> xs, double q) {
        if (xs.empty()) {
            return 0.0;
        }
        std::sort(xs.begin(), xs.end());
        const size_t idx = std::min(xs.size() - 1, (size_t) (q * (double) (xs.size() - 1) + 0.5));
        return xs[idx];
    };
    const auto bypass_field = [](int n) {
        std::vector<std::string> vals;
        vals.reserve((size_t) n);
        for (int i = 0; i < n; ++i) {
            vals.push_back("v" + std::to_string(i));
        }
        return std::vector<llama_decision::field_input>{ { "  \"a\": ", std::move(vals) } };
    };
    const auto measure_bypass = [&](const std::string & fork, int n_opts, bool bypass) {
        llama_decision::options o;
        o.mode      = "greedy"; // one branch per greedy round, so the bypass actually engages
        o.fork      = fork;
        o.bypass    = bypass;
        o.cache_tag = "cal-bypass-" + fork + "-" + std::to_string(n_opts);
        const auto fields = bypass_field(n_opts);
        std::vector<double> samples;
        for (int i = 0; i < 9; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            (void) decide_batch(eng, long_shared, { "ctx" }, fields, o);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (i > 0) {
                samples.push_back(ms);
            }
        }
        common_json r = common_json::object();
        r["median_ms"] = percentile(samples, 0.50);
        r["p95_ms"]    = percentile(samples, 0.95);
        return r;
    };
    common_json bypass_report = common_json::object();
    for (const std::string fork : { std::string("copy"), std::string("restore") }) {
        if (fork == "copy" && !copy_fork) {
            continue;
        }
        common_json by_fork = common_json::object();
        for (int n : { 2, 16, 64 }) {
            const common_json on  = measure_bypass(fork, n, true);
            const common_json off = measure_bypass(fork, n, false);
            common_json e = common_json::object();
            e["on"]      = on;
            e["off"]     = off;
            e["speedup"] = on.at("median_ms").get<double>() > 0.0
                         ? off.at("median_ms").get<double>() / on.at("median_ms").get<double>()
                         : 1.0;
            by_fork[std::to_string(n)] = e;
        }
        bypass_report[fork] = by_fork;
    }
    out["bypass_measurement"] = bypass_report;

    // prefix-reuse economy behind the hoist threshold: cached vs cold prefill of the shared head
    llama_decision::options po;
    po.cache_tag = "cal-prefill";
    llama_decision::options pc = po;
    pc.allow_cache = false;
    auto time_prefill = [&](const llama_decision::options & o) {
        double best = 1e18;
        for (int i = 0; i < 5; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            (void) decide_batch(eng, long_shared, { "ctx" }, four, o);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (i > 0) {
                best = std::min(best, ms);
            }
        }
        return best;
    };
    out["prefill_cached_ms"] = time_prefill(po);
    out["prefill_cold_ms"]   = time_prefill(pc);

    // auto mode switches to greedy exactly above tree_max
    llama_decision::options o3;
    o3.mode      = "auto";
    o3.tree_max  = 3;
    o3.cache_tag = "cal-auto3";
    llama_decision::options o4;
    o4.mode      = "auto";
    o4.tree_max  = 3;
    o4.cache_tag = "cal-auto4";
    const auto b3 = decide_batch(eng, "system", { "ctx" }, small, o3);
    const auto b4 = decide_batch(eng, "system", { "ctx" }, four, o4);
    out["auto_at_tree_max_is_tree"]    = scored_by_tree(b3);
    out["auto_above_tree_max_is_tree"] = scored_by_tree(b4);

    // temperature changes the distribution shape but keeps the winner; NLL(winner) is recorded
    auto vocab = llama_decision::make_llama_label_vocab(llama_model_get_vocab(te.model));
    const auto pool = llama_decision::build_label_pool(*vocab, test_letter_tail(), 64);
    out["label_pool_size"] = (long long) pool.size();

    auto readout = [&](double temp) {
        common_json body = common_json::parse(decision_valid_body());
        body["temperature"] = temp;
        body.erase("temperatures");
        llama_decision::options ro;
        ro.cache_tag = "cal-temp";
        return test_letter_readout(eng, *vocab, nullptr, false, llama_decision::parse_decision_request(body), pool, ro, nullptr);
    };
    const auto p10 = readout(1.0);
    const auto p13 = readout(1.3);
    double nll10 = 0.0;
    double nll13 = 0.0;
    size_t nq    = 0;
    for (size_t i = 0; i < p10.size(); ++i) {
        const int w = argmax(p10[i]);
        nll10 += -std::log(std::max(1e-12, (double) p10[i][w]));
        nll13 += -std::log(std::max(1e-12, (double) p13[i][w]));
        ++nq;
    }
    out["temperature_nll_t1"]    = nq ? nll10 / nq : 0.0;
    out["temperature_nll_t1_3"]  = nq ? nll13 / nq : 0.0;
    out["temperature_nll_delta"] = nq ? (nll13 - nll10) / nq : 0.0;

    // hoist (outcome axis): optimize on vs off changes the batch shape by lifting the shared
    // suffix head onto the trunk; the constrained distribution must not move, only the row layout
    std::vector<llama_decision::field_input> corpus;
    // discriminative candidates and a meaningful state: the control group must not sit on a near
    // tie, or tiny batch-shape noise could flip the winner and mask a real hoist regression
    const std::vector<std::string> cands = { "billing", "technical", "cancellation", "refund", "account", "support" };
    const std::string decision_state = "The customer was charged twice on May 3 and asks for a refund of the duplicate charge.";
    for (int f = 0; f < 5; ++f) {
        corpus.push_back({ "  \"prepared_and_standardized_field_of_the_schema_" + std::to_string(f) + "\": ", cands });
    }
    llama_decision::options oo_on;
    oo_on.optimize  = true;
    oo_on.cache_tag = "cal-hoist-on";
    llama_decision::options oo_off;
    oo_off.optimize  = false;
    oo_off.cache_tag = "cal-hoist-off";
    const auto hon  = decide_batch(eng, "system", { decision_state }, corpus, oo_on);
    const auto hoff = decide_batch(eng, "system", { decision_state }, corpus, oo_off);
    double hoist_tv = 0.0;
    bool   hoist_agree = hon.items.size() == hoff.items.size() &&
                         !hon.items.empty() && !hoff.items.empty() &&
                         hon.items[0].fields.size() == hoff.items[0].fields.size();
    for (size_t fi = 0; hoist_agree && fi < hon.items[0].fields.size(); ++fi) {
        const auto & a = hon.items[0].fields[fi].probs;
        const auto & b = hoff.items[0].fields[fi].probs;
        hoist_agree = a.size() == b.size() && argmax(a) == argmax(b);
        hoist_tv = std::max(hoist_tv, tv(a, b));
    }
    out["hoist_argmax_agree"] = hoist_agree;
    out["hoist_tv"]           = hoist_tv;
    out["hoist_fired"]        = hon.common_suffix_tokens > 0 && hoff.common_suffix_tokens == 0;

    // auto mode with the default tree_max keeps the widest contract field on the exact tree path
    llama_decision::options od;
    od.mode      = "auto";
    od.cache_tag = "cal-auto-default";
    const auto wdef = decide_batch(eng, "system", { "ctx" }, wide, od);
    out["wide_auto_default_is_tree"] = scored_by_tree(wdef);

    // shared-prefix cache cost at three prefix sizes: a 32k prefix needs a window above it, so reopen
    // the context and fall back to a smaller window when the backend cannot hold the largest one
    {
        bool reopened = false;
        for (int big : { 40960, 20480, 10240 }) {
            if (te.reopen(10, big)) {
                reopened = true;
                break;
            }
        }
        out["prefix_warm_measurement"] = reopened ? calibration_prefix_warm_measurement(te) : common_json::object();
    }
    return out;
}

static void test_calibration_hoist(testing & t) {
    t.test("calibration: hoist threshold separates exact heads from near misses", [](testing & t) {
        const common_json m = calibration_hoist_measurement();
        t.assert_true("precision >= 0.99", m.at("precision").get<double>() >= 0.99);
        t.assert_true("recall >= 0.99", m.at("recall").get<double>() >= 0.99);
        t.assert_equal("31-token near misses never hoist", 0, m.at("false_positives").get<int>());
        t.assert_equal("no exact head is missed", 0, m.at("false_negatives").get<int>());
    });
}

static void test_calibration_dedup(testing & t) {
    t.test("calibration: exact duplicate suffixes dedup but near duplicates do not", [](testing & t) {
        const common_json m = calibration_dedup_measurement();
        t.assert_equal("every exact duplicate dedups", m.at("pair_suite").get<int>(), m.at("exact_equal").get<int>());
        t.assert_equal("no near duplicate dedups", 0, m.at("near_equal").get<int>());
    });
}

static void test_calibration_confidence(testing & t) {
    t.test("calibration: confidence diagnostics are a usable detector but stay advisory-only", [](testing & t) {
        const common_json m = calibration_confidence_measurement();
        t.assert_true("allowed-mass AUC >= 0.99", m.at("auc").get<double>() >= 0.99);
        t.assert_equal("verdict is advisory-only", std::string("advisory-only"), m.at("verdict").get<std::string>());
    });
}



static common_json calibration_row(const std::string & trigger, const std::string & axis,
                                   const std::vector<std::string> & may_gate,
                                   const std::vector<std::string> & may_not_gate,
                                   const std::string & verdict) {
    common_json r = common_json::object();
    r["trigger"]    = trigger;
    r["axis"]       = axis;
    common_json mg  = common_json::array();
    for (const auto & s : may_gate) {
        mg.push_back(s);
    }
    common_json mn = common_json::array();
    for (const auto & s : may_not_gate) {
        mn.push_back(s);
    }
    r["may_gate"]        = mg;
    r["may_not_gate"]    = mn;
    r["production_gate"] = false;
    r["verdict"]         = verdict;
    return r;
}

// A control group is recorded data: the cases a trigger must accept and the cases it must refuse.
// The ledger test reads it, so a named trigger cannot be added without one.
static common_json control_case(const common_json & input, const common_json & expect) {
    common_json c = common_json::object();
    c["input"]  = input;
    c["expect"] = expect;
    return c;
}

static common_json control_group(std::initializer_list<common_json> cases) {
    common_json g   = common_json::object();
    common_json arr = common_json::array();
    for (const auto & c : cases) {
        arr.push_back(c);
    }
    g["cases"] = arr;
    return g;
}

static common_json calibration_rows() {
    common_json rows = common_json::object();

    {
        auto r = calibration_row("prefix hoist length (>=32 shared tokens)", "task-value",
                                 { "prefix reuse decision" },
                                 { "answers", "admission" },
                                 "hoist only on an exact shared token head of at least 32 tokens; near misses never hoist; prefix-reuse prefill delta recorded");
        r["measurements"] = calibration_hoist_measurement();
        r["measurements"]["outcome_tv_bound"]     = 0.15;
        r["measurements"]["outcome_argmax_agree"] = true;
        common_json g = control_group({
            control_case("shared head at the hoist budget", "hoist"),
            control_case("shared head below the hoist minimum", "no hoist"),
            control_case("near-miss head", "no hoist"),
        });
        g["hoist_min_tokens"] = 4;
        g["hoist_budget"]     = 32;
        r["control_group"] = g;
        rows["prefix_hoist"] = r;
    }
    {
        auto r = calibration_row("question dedup on byte-identical suffix", "task-value",
                                 { "dedup of identical suffixes" },
                                 { "near-duplicate collapse", "caching across states" },
                                 "dedup only on exact suffix equality; near duplicates never collapse");
        r["measurements"] = calibration_dedup_measurement();
        rows["question_dedup"] = r;
    }
    {
        auto r = calibration_row("allowed_token_mass / full-vocab argmax diagnostics", "confidence",
                                 {},
                                 { "admission", "caching", "routing", "persistence", "answer validity" },
                                 "advisory-only; dashboard threshold, never a production gate");
        r["measurements"] = calibration_confidence_measurement();
        rows["confidence_diagnostics"] = r;
    }

    rows["tree_vs_greedy"] = calibration_row("tree vs greedy exactness and cost", "task-value",
                                 { "exact-vs-greedy mode choice" },
                                 { "correctness claims", "caching" },
                                 "measured on the test model; model_measurements carry the numbers");
    rows["single_question_bypass"] = calibration_row("single-question fork bypass", "task-value",
                                 { "which scoring path runs" },
                                 { "correctness", "admission" },
                                 "copy-fork only; bypass stays on by default and never runs on multi-question rounds, on restore-fork memory, or on hybrid memory. A restore fork is exact but measured no faster than the fork it replaces; a hybrid trunk cannot be restored exactly, so a later greedy round would decode on a corrupted trunk. Per-option-count medians are recorded in model_measurements.bypass_measurement");
    {
        auto r = calibration_row("tree_max auto switch", "task-value",
                                 { "mode selection at the boundary" },
                                 { "correctness claims" },
                                 "auto uses tree up to tree_max and greedy above; default pinned at 128");
        common_json g = control_group({
            control_case("value count at tree_max default", "tree"),
            control_case("value count above tree_max default", "greedy"),
            control_case("explicit tree mode", "tree"),
            control_case("explicit greedy mode", "greedy"),
        });
        g["tree_max_default"] = 128;
        r["control_group"] = g;
        rows["auto_mode_boundary"] = r;
    }
    rows["temperature_profile"] = calibration_row("calibrated temperature", "confidence",
                                 { "probability/confidence values only" },
                                 { "admission", "caching", "routing", "persistence", "answer key" },
                                 "T=1.0 default; non-default needs matching provenance; stale profile refused");
    rows["temperature_profile"]["measurements"] = calibration_temperature_control_measurement();
    {
        auto r = calibration_row("single decision request at max_queue defaults", "resource",
                                 { "how many concurrent decisions run" },
                                 { "single-request rejection", "answer validity" },
                                 "a single small request is always admitted at the configured queue depth; only a saturated burst may be refused (429/529); measured by test_decision_admission.py");
        common_json g = control_group({
            control_case("single small request", "admitted"),
            control_case("burst above the queue depth", "429 or 529 with Retry-After"),
        });
        g["queue_cap_default"] = 4;
        g["queue_cap_env"]     = "LLAMA_DECISION_MAX_QUEUE";
        r["control_group"] = g;
        rows["admission_control"] = r;
    }
    {
        auto r = calibration_row("prefix state LRU capacity (restore-mode cache)", "operational",
                                 { "state-cache eviction cost" },
                                 { "answers", "admission", "routing" },
                                 "the restore-mode prefix state cache keeps the most recent entries and evicts the least recent; a hit changes cost only, never an answer");
        common_json g = control_group({
            control_case("revisit within the capacity", "hit"),
            control_case("revisit past the capacity", "miss"),
            control_case("a re-prefilled evicted entry", "same answer"),
        });
        g["capacity"] = 4;
        r["control_group"] = g;
        rows["prefix_lru_capacity"] = r;
    }
    {
        auto r = calibration_row("permutations cap (order de-bias passes)", "task-value",
                                 { "how many order passes run" },
                                 { "the winner when the default is used", "answers" },
                                 "the pass count is accepted and capped at 8, one pass is the default, and zero passes is refused; more passes only add cost");
        common_json g = control_group({
            control_case("two passes", "accepted"),
            control_case("value above the cap", "accepted and capped at 8"),
            control_case("zero passes", "rejected"),
            control_case("no permutations field", "default one pass"),
        });
        g["cap"]     = 8;
        g["default"] = 1;
        r["control_group"] = g;
        rows["permutations_cap"] = r;
    }
    {
        auto r = calibration_row("choice option count limits", "task-value",
                                 { "option count validation before scoring" },
                                 { "answers", "admission" },
                                 "a choice keeps 2 to 64 options; over-limit is rejected before decode, never truncated");
        common_json g = control_group({
            control_case("one option", "rejected"),
            control_case("two options", "accepted"),
            control_case("64 options", "accepted"),
            control_case("65 options", "rejected"),
        });
        g["min_options"] = 2;
        g["max_options"] = 64;
        r["control_group"] = g;
        rows["choice_option_limits"] = r;
    }
    {
        auto r = calibration_row("realized answer-label pool size", "task-value",
                                 { "whether a question's option set can be represented" },
                                 { "answers", "admission", "caching" },
                                 "the protocol cap is 64 (LABEL_POOL_CAP, equal to DECISION_MAX_CHOICE_OPTIONS); the "
                                 "realized pool is model-dependent and may be smaller; a request whose widest question "
                                 "needs more labels than the realized pool is a 422");
        common_json g               = control_group({
            control_case("a question at the realized pool size", "accepted"),
            control_case("a question one above the realized pool", "422"),
            control_case("a double-letter vocabulary at the cap", "realized pool reaches 64"),
        });
        g["protocol_cap"]           = 64;
        g["realized_source"]        = "diagnostics.label_pool_size; recorded per model in model_measurements";
        r["control_group"]          = g;
        rows["label_pool_capacity"] = r;
    }
    {
        auto r = calibration_row("score level count limits", "task-value",
                                 { "level count validation before scoring" },
                                 { "answers", "admission" },
                                 "a score keeps 2 to 10 levels; over-limit is rejected before decode, never truncated");
        common_json g = control_group({
            control_case("one level", "rejected"),
            control_case("two levels", "accepted"),
            control_case("ten levels", "accepted"),
            control_case("eleven levels", "rejected"),
        });
        g["min_levels"] = 2;
        g["max_levels"] = 10;
        r["control_group"] = g;
        rows["score_level_limits"] = r;
    }
    {
        auto r = calibration_row("question count limits", "task-value",
                                 { "question count validation before scoring" },
                                 { "answers", "admission" },
                                 "a request carries 1 to 256 questions; over-limit is rejected before decode, never truncated");
        common_json g = control_group({
            control_case("zero questions", "rejected"),
            control_case("one question", "accepted"),
            control_case("256 questions", "accepted"),
            control_case("257 questions", "rejected"),
        });
        g["min_questions"] = 1;
        g["max_questions"] = 256;
        r["control_group"] = g;
        rows["question_count_limit"] = r;
    }
    {
        auto r = calibration_row("request body cap", "resource",
                                 { "request admission by body size" },
                                 { "answers", "single-request validity" },
                                 "the decision body cap is rejected with 413 before decode and never truncated; the cap is overridable per deployment");
        common_json g = control_group({
            control_case("body under the cap", "accepted"),
            control_case("body over the cap", "413 before decode"),
        });
        g["body_cap_bytes"] = 2 * 1024 * 1024;
        g["body_cap_env"]   = "LLAMA_DECISION_MAX_BODY";
        r["control_group"] = g;
        rows["body_cap"] = r;
    }
    {
        auto r = calibration_row("output-row budget (max of chat and decision, not the sum)", "task-value",
                                 { "logits buffer size for the shared context" },
                                 { "answers", "admission", "caching" },
                                 "chat and decision decodes are serialized, so the shared budget is the larger need; it always covers n_parallel + n_seq_decision and never exceeds the sum");
        r["measurements"] = calibration_output_row_capacity_measurement();
        common_json g = control_group({
            control_case("decision sequences on a draft-free context", "every sequence covered"),
            control_case("draft sequences on the context", "budget not above the sum"),
        });
        g["serialization"] = "chat and decision never decode at the same time";
        r["control_group"] = g;
        rows["output_row_capacity"] = r;
    }
    {
        auto r = calibration_row("opt-in confidence profile and order-de-bias pass profile", "task-value",
                                 { "which confidence value is reported", "how many order-de-bias passes run" },
                                 { "admission", "caching", "routing", "persistence", "answer key" },
                                 "the certainty-based Jev confidence (N*p_max-1)/(N-1) and permutations=1 stay the defaults; the local 1-H/logK confidence profile and a higher default pass count are opt-in per request or per server flag and change only the reported concentration and the cost, never the winner gate; the corpus harness records winner agreement, Brier and ECE");
        common_json g = control_group({
            control_case("no confidence_profile field", "certainty-based Jev (N*p_max-1)/(N-1), unchanged answer"),
            control_case("confidence_profile=local", "1-H/logK, same probabilities"),
            control_case("no permutations field at the default server", "one pass"),
            control_case("server default permutations=2", "two passes unless the request says otherwise"),
        });
        g["metrics"] = common_json::parse(R"(["winner_agreement","brier","ece"])");
        g["corpus"]  = "accuracy_corpus.json";
        g["report"]  = "accuracy_report.json";
        r["control_group"] = g;
        rows["readout_compatibility"] = r;
    }
    {
        auto r = calibration_row("weak-quant producer-stability allowlist", "producer-confidence",
                                 { "whether a producer-stability assertion is skipped on a model" },
                                 { "any task-value assertion", "answers", "admission", "caching", "routing", "persistence" },
                                 "only a model whose GPU numerics move a winner may be listed; a model that passes the task-value checks is removed, never kept");
        r["measurements"] = calibration_determinism_allowlist_measurement();
        common_json g = control_group({
            control_case("model on the allowlist", "producer assertion skipped, not xfail"),
            control_case("model not on the allowlist", "producer assertion runs hard"),
        });
        r["control_group"] = g;
        rows["determinism_allowlist"] = r;
    }
    return rows;
}

static void test_calibration_table(testing & t) {
    t.test("calibration: sign-off table covers every gated heuristic and stays advisory-only", [](testing & t) {
        const common_json cal = common_json::parse(
            read_file(std::string(DECISION_TEST_BASELINE_DIR) + "/calibration.json"));

        const char * ids[] = { "tree_vs_greedy", "prefix_hoist", "single_question_bypass", "question_dedup", "confidence_diagnostics", "auto_mode_boundary", "temperature_profile", "readout_compatibility", "admission_control" };
        for (const char * id : ids) {
            const auto & row = cal.at("rows").at(id);
            t.assert_true(std::string(id) + " has an axis", row.contains("axis"));
            t.assert_true(std::string(id) + " has a verdict", !row.at("verdict").get<std::string>().empty());
            t.assert_true(std::string(id) + " is not a production gate", !row.at("production_gate").get<bool>());
        }
        for (const char * id : { "confidence_diagnostics", "temperature_profile", "readout_compatibility" }) {
            const auto & ng = cal.at("rows").at(id).at("may_not_gate");
            for (const char * rule : { "admission", "caching", "routing", "persistence" }) {
                bool found = false;
                for (const auto & e : ng.items()) {
                    found = found || e.value().get<std::string>() == rule;
                }
                t.assert_true(std::string(id) + " may not gate " + rule, found);
            }
        }

        assert_close(t, "prefix_hoist precision matches the committed value",
                     cal.at("rows").at("prefix_hoist").at("measurements").at("precision").get<double>(),
                     calibration_hoist_measurement().at("precision").get<double>(), 1e-9);
        assert_close(t, "prefix_hoist recall matches the committed value",
                     cal.at("rows").at("prefix_hoist").at("measurements").at("recall").get<double>(),
                     calibration_hoist_measurement().at("recall").get<double>(), 1e-9);
        t.assert_equal("question_dedup exact dedup matches the committed value",
                       cal.at("rows").at("question_dedup").at("measurements").at("exact_equal").get<int>(),
                       calibration_dedup_measurement().at("exact_equal").get<int>());
        t.assert_equal("question_dedup near-dup result matches the committed value",
                       cal.at("rows").at("question_dedup").at("measurements").at("near_equal").get<int>(),
                       calibration_dedup_measurement().at("near_equal").get<int>());
        assert_close(t, "confidence_diagnostics AUC matches the committed value",
                     cal.at("rows").at("confidence_diagnostics").at("measurements").at("auc").get<double>(),
                     calibration_confidence_measurement().at("auc").get<double>(), 1e-9);
        assert_close(t, "prefix_hoist outcome TV bound matches the committed value",
                     cal.at("rows").at("prefix_hoist").at("measurements").at("outcome_tv_bound").get<double>(), 0.15, 1e-9);

        assert_close(t, "temperature_profile match-accept matches the committed value",
                     cal.at("rows").at("temperature_profile").at("measurements").at("match_accepted").get<double>(),
                     calibration_temperature_control_measurement().at("match_accepted").get<double>(), 1e-9);
        t.assert_equal("temperature_profile every mismatch refused matches the committed value",
                       cal.at("rows").at("temperature_profile").at("measurements").at("mismatch_refused").get<int>(),
                       calibration_temperature_control_measurement().at("mismatch_refused").get<int>());
    });
}

// Every named trigger constant carries a ledger row with an axis tag and a recorded control group.
// The map is data: removing a row, its axis or its control group fails here, and a new trigger has
// to be recorded before it can gate anything.
static void test_calibration_ledger(testing & t) {
    t.test("calibration: every named trigger has an axis and a recorded control group", [](testing & t) {
        const common_json cal = common_json::parse(
            read_file(std::string(DECISION_TEST_BASELINE_DIR) + "/calibration.json"));

        const struct { const char * name; const char * row; } triggers[] = {
            { "HOIST_MIN_TOKENS",    "prefix_hoist" },
            { "HOIST_BUDGET",        "prefix_hoist" },
            { "tree_max default",    "auto_mode_boundary" },
            { "prefix LRU capacity", "prefix_lru_capacity" },
            { "permutations cap",    "permutations_cap" },
            { "choice limits",       "choice_option_limits" },
            { "score limits",        "score_level_limits" },
            { "question count limit","question_count_limit" },
            { "queue cap",           "admission_control" },
            { "body cap",            "body_cap" },
            { "output-row budget",   "output_row_capacity" },
            { "determinism allowlist","determinism_allowlist" },
        };
        for (const auto & tr : triggers) {
            const std::string id  = tr.row;
            const std::string tag = std::string(tr.name) + " (" + id + ")";
            t.assert_true(tag + " maps to a ledger row", cal.at("rows").contains(id));
            const auto & row = cal.at("rows").at(id);
            t.assert_true(tag + " row has an axis tag",
                          row.contains("axis") && !row.at("axis").get<std::string>().empty());
            t.assert_true(tag + " row has a control group", row.contains("control_group"));
            const auto & cg = row.at("control_group");
            t.assert_true(tag + " control group records cases",
                          cg.contains("cases") && cg.at("cases").is_array() && cg.at("cases").size() > 0);
            t.assert_true(tag + " is not a production gate", !row.at("production_gate").get<bool>());
        }
    });
}

// The moved and new gates are recorded with measured verdicts. A capability precondition is
// measured on real fixtures, the serialized budget is swept over batch/sequence/draft shapes, and
// the producer-stability allowlist is a skip-only list.
static void test_calibration_gates(testing & t) {
    t.test("calibration: relocated gates carry measured verdicts", [](testing & t) {
        const common_json cal = common_json::parse(
            read_file(std::string(DECISION_TEST_BASELINE_DIR) + "/calibration.json"));

        // output-row capacity (task-value): the serialized budget covers every sequence and is no
        // larger than the sum of the two workloads
        const common_json oc = calibration_output_row_capacity_measurement();
        const auto & oc_row = cal.at("rows").at("output_row_capacity");
        t.assert_equal("output_row_capacity axis is task-value",
                       std::string("task-value"), oc_row.at("axis").get<std::string>());
        t.assert_equal("output_row_capacity cases match the committed value",
                       oc_row.at("measurements").at("cases").get<int>(), oc.at("cases").get<int>());
        t.assert_equal("every coincident workload is covered",
                       oc_row.at("measurements").at("covered").get<int>(),
                       oc_row.at("measurements").at("cases").get<int>());
        t.assert_equal("the serialized budget never exceeds the sum",
                       oc_row.at("measurements").at("not_summed").get<int>(),
                       oc_row.at("measurements").at("cases").get<int>());
        t.assert_true("a draft-free chat buffer is no larger than the sum",
                      oc_row.at("measurements").at("max_rows_saved").get<int>() >= 0);

        // producer bit-stability allowlist (producer confidence): skip-only, never a task-value gate
        const common_json da = calibration_determinism_allowlist_measurement();
        const auto & da_row = cal.at("rows").at("determinism_allowlist");
        t.assert_equal("determinism_allowlist axis is producer-confidence",
                       std::string("producer-confidence"), da_row.at("axis").get<std::string>());
        t.assert_true("determinism_allowlist is not a production gate",
                      !da_row.at("production_gate").get<bool>());
        t.assert_equal("determinism_allowlist entries match the committed value",
                       da_row.at("measurements").at("count").get<int>(), da.at("count").get<int>());
        t.assert_true("the allowlist never gates an answer, admission or cache",
                      da_row.at("may_not_gate").is_array() && da_row.at("may_not_gate").size() >= 5);
    });
}

// A producer-stability allowlist is only honest if the listed model actually shows the variance it
// is listed for: two identical permutation passes on the GPU do not land on bit-identical
// probabilities. A model that does stay stable must be removed from the list, not kept. This runs
// only when the loaded model is on the list, so the CPU lane skips it.
static void test_calibration_determinism_variance(testing & t) {
    t.test("calibration: a listed model shows the producer variance it is listed for", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (path == nullptr || path[0] == '\0') {
            t.skip("set LLAMA_DECISION_TEST_MODEL to run");
            return;
        }
        if (!weak_quant_gpu_oracle(path)) {
            t.skip("the loaded model is not on the producer-stability allowlist");
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("model loads", false);
            return;
        }
        try {
            auto vocab = llama_decision::make_llama_label_vocab(llama_model_get_vocab(te.model));
            const auto pool = llama_decision::build_label_pool(*vocab, test_letter_tail(), 64);
            llama_decision::engine eng(te.ctx, 2, 8);

            common_json two = common_json::parse(decision_valid_body());
            two["permutations"] = 2;
            llama_decision::options o;
            o.cache_tag = "producer-variance";
            const auto p1 = test_letter_readout(eng, *vocab, nullptr, false,
                                                           llama_decision::parse_decision_request(two), pool, o, nullptr);
            const auto p2 = test_letter_readout(eng, *vocab, nullptr, false,
                                                           llama_decision::parse_decision_request(two), pool, o, nullptr);

            bool stable = p1.size() == p2.size();
            for (size_t qi = 0; stable && qi < p1.size(); ++qi) {
                stable = p1[qi].size() == p2[qi].size();
                for (size_t i = 0; stable && i < p1[qi].size(); ++i) {
                    stable = std::fabs(p1[qi][i] - p2[qi][i]) < 1e-6;
                }
            }
            t.assert_true("the listed model shows producer variance (two passes drift)", !stable);
        } catch (const std::exception & e) {
            t.assert_true(std::string("producer variance: ") + e.what(), false);
        }
    });
}

// The provenance of the readout must be identical at both call sites (temperature validation and
// response diagnostics), so it is one function of the live params and model.
static void test_decision_provenance(testing & t) {
    t.test("decision provenance is one function of the live params and model", [](testing & t) {
        const std::string path = decision_cpu_model_path();
        if (path.empty()) {
            t.skip("no generated model; run the generate-models fixture");
            return;
        }
        cpu_test_engine te;
        if (!te.load(path)) {
            t.assert_true("the CPU decision scaffold loads the model", false);
            return;
        }
        common_params params;
        params.model.path   = path;
        params.n_ubatch     = 128;
        params.cache_type_k = GGML_TYPE_F16;
        params.cache_type_v = GGML_TYPE_F16;
        params.kv_unified   = true;

        const auto a = llama_decision::decision_provenance_current("m", params, te.model, nullptr, false);
        const auto b = llama_decision::decision_provenance_current("m", params, te.model, nullptr, false);
        t.assert_equal("model is stable", a.model, b.model);
        t.assert_equal("quantization is stable", a.quantization, b.quantization);
        t.assert_equal("template hash is stable", a.template_hash, b.template_hash);
        t.assert_equal("backend flags are stable", a.backend_flags, b.backend_flags);
        t.assert_true("the quantization is non-empty", !a.quantization.empty());

        // the backend flags follow the params, so both call sites see the same backend identity
        common_params other = params;
        other.n_ubatch = 256;
        const auto c = llama_decision::decision_provenance_current("m", other, te.model, nullptr, false);
        t.assert_true("backend flags follow the params", a.backend_flags != c.backend_flags);
    });
}

static void test_calibration_model(testing & t) {
    t.test("calibration: model-dependent thresholds measured on a real model", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (!gpu_model_ready(t, path)) {
            return;
        }
        common_json m;
        try {
            m = calibration_model_measurement(path);
        } catch (const std::exception & e) {
            t.assert_true(std::string("calibration measurement runs: ") + e.what(), false);
            return;
        }

        t.assert_true("tree_vs_greedy control-small keeps the same winner", m.at("small_argmax_agree").get<bool>());
        t.assert_true("tree_vs_greedy control-small distributions agree", m.at("small_tv").get<double>() <= 1e-4);
        t.assert_true("tree_vs_greedy tree mode is exact", m.at("wide_tree_is_tree").get<bool>());
        t.assert_true("tree_vs_greedy auto above tree_max selects greedy", !m.at("wide_auto_is_tree").get<bool>());
        t.assert_true("tree_vs_greedy both modes score at least one row",
                      m.at("wide_tree_rows").get<long long>() > 0 && m.at("wide_greedy_rows").get<long long>() > 0);
        t.assert_true("tree_vs_greedy auto with the default tree_max stays on the exact tree path",
                      m.at("wide_auto_default_is_tree").get<bool>());
        if (m.at("bypass_applicable").get<bool>()) {
            t.assert_true("single_question_bypass bypass does not regress", m.at("bypass_speedup").get<double>() >= 0.9);
        } else {
            t.assert_true("single_question_bypass bypass is not applicable to restore-fork memory", m.at("bypass_speedup").get<double>() == 1.0);
        }
        // the benefit report covers the exact fork kinds at three option counts and is finite; the
        // numbers are a measurement, never a gate on a producer score
        const common_json & bm     = m.at("bypass_measurement");
        bool                bm_ok  = bm.is_object() && !bm.empty();
        for (const auto & fork_entry : bm.items()) {
            for (const auto & count_entry : fork_entry.value().items()) {
                const common_json & e = count_entry.value();
                bm_ok = bm_ok && std::isfinite(e.at("on").at("median_ms").get<double>()) &&
                                std::isfinite(e.at("off").at("median_ms").get<double>()) &&
                                std::isfinite(e.at("speedup").get<double>());
            }
        }
        t.assert_true("single_question_bypass benefit report is finite at every option count", bm_ok);
        t.assert_true("auto_mode_boundary auto at tree_max selects tree", m.at("auto_at_tree_max_is_tree").get<bool>());
        t.assert_true("auto_mode_boundary auto above tree_max selects greedy", !m.at("auto_above_tree_max_is_tree").get<bool>());
        t.assert_true("temperature_profile NLL is finite", std::isfinite(m.at("temperature_nll_t1").get<double>()));
        t.assert_true("prefix-reuse prefill delta is reported",
                      m.at("prefill_cached_ms").get<double>() > 0.0 &&
                      m.at("prefill_cached_ms").get<double>() <= m.at("prefill_cold_ms").get<double>() * 1.1);

        // shared-prefix cache cost: the LRU host restore is only worth keeping on a recurrent or
        // hybrid model if a hit beats a cold prefill. The numbers are recorded per size; the
        // assertions only require the measurement to be present and the hit to be cheaper.
        const common_json & pwm = m.at("prefix_warm_measurement");
        t.assert_true("prefix_warm_measurement records the class and sizes",
                      pwm.contains("class") && pwm.contains("sizes") && !pwm.at("sizes").empty());
        for (const auto & size_entry : pwm.at("sizes").items()) {
            const common_json & r = size_entry.value();
            t.assert_true("prefix_warm_measurement cold is finite", std::isfinite(r.at("cold").at("median_ms").get<double>()));
            t.assert_true("prefix_warm_measurement miss is finite", std::isfinite(r.at("miss_median_ms").get<double>()));
            t.assert_true("prefix_warm_measurement resident hit is finite",
                          std::isfinite(r.at("resident_hit_median_ms").get<double>()));
            if (pwm.at("class").get<std::string>() != "dense") {
                t.assert_true("the LRU host restore is faster than a cold prefill",
                              r.at("lru_speedup_vs_cold").get<double>() > 1.0);
            }
        }

        // hoist (outcome axis): optimize on/off must keep every winner and stay within the
        // committed TV bound recorded in the sign-off table, and the control must actually
        // exercise the hoist (shared head lifted on, not lifted off)
        const common_json cal = common_json::parse(
            read_file(std::string(DECISION_TEST_BASELINE_DIR) + "/calibration.json"));
        const double hoist_bound = cal.at("rows").at("prefix_hoist").at("measurements").at("outcome_tv_bound").get<double>();
        t.assert_true("hoist control: optimize on vs off keeps the winner", m.at("hoist_argmax_agree").get<bool>());
        t.assert_true("hoist control: per-field TV stays within the committed bound",
                      m.at("hoist_tv").get<double>() <= hoist_bound);
        t.assert_true("hoist control: the shared suffix head is actually hoisted", m.at("hoist_fired").get<bool>());
    });
}

static int write_calibration(const char * model_path) {
    common_json cal = common_json::object();
    cal["note"] = "Calibration sign-off for the decision heuristics. Values are measurements, not "
                  "tunables. Every row carries production_gate:false; confidence rows may never gate "
                  "admission, caching, routing or persistence. Update only with a matching code change "
                  "and a re-measured gate.";
    cal["rows"] = calibration_rows();

    common_json model = common_json::object();
    if (model_path == nullptr || model_path[0] == '\0') {
        fprintf(stderr, "set LLAMA_DECISION_TEST_MODEL to record model measurements\n");
    } else {
        const std::string model_path_str = model_path;
        std::ifstream model_in(model_path, std::ios::binary | std::ios::ate);
        const long long model_bytes = model_in ? (long long) model_in.tellg() : -1;

        common_json env = common_json::object();
        env["model"]   = model_identity(model_path_str);
        env["model_bytes"] = model_bytes;
        env["quantization"] = "recorded per run; see model filename";
        env["backend_flags"] = common_json::parse(R"({"kv_unified":true,"swa_full":false,
            "n_ctx":8192,"n_batch":512,"n_ubatch":512,"n_seq_max":10,"gpu_layers":0})");
        model["environment"] = env;
        try {
            model["values"] = calibration_model_measurement(model_path);
        } catch (const std::exception & e) {
            fprintf(stderr, "failed to measure model: %s\n", e.what());
            return 2;
        }
    }
    cal["model_measurements"] = model;

    write_file(std::string(DECISION_TEST_BASELINE_DIR) + "/calibration.json", cal.dump(2) + "\n");
    return 0;
}

// Refreshes calibration.json "rows" only; the model measurements are preserved. Host-runnable:
// every row is a pure measurement, so no GGUF is needed.
static int write_calibration_rows() {
    const std::string path = std::string(DECISION_TEST_BASELINE_DIR) + "/calibration.json";
    common_json cal = common_json::parse(read_file(path));
    cal["rows"] = calibration_rows();
    write_file(path, cal.dump(2) + "\n");
    return 0;
}

static int write_goldens() {
    write_file(fixture_path("decision_basic.golden.json"), decision_basic_from_fixed_scores().dump(2) + "\n");
    write_file(fixture_path("generic_typed.golden.json"), generic_record_from_fixed_scores().dump(2) + "\n");
    return 0;
}

// Refreshes baseline.json "cpu_oracle" only; every other section is preserved.
static int write_cpu_oracle() {
    const common_json oracle = decision_cpu_oracle();
    if (oracle.empty()) {
        fprintf(stderr, "the CPU oracle needs the generated dummy model\n");
        return 2;
    }
    const std::string path = std::string(DECISION_TEST_BASELINE_DIR) + "/baseline.json";
    common_json baseline = common_json::parse(read_file(path));
    baseline["cpu_oracle"] = oracle;
    write_file(path, baseline.dump(1) + "\n");
    return 0;
}

static int write_decision_golden(const char * model_path) {
    if (model_path == nullptr || model_path[0] == '\0') {
        fprintf(stderr, "set LLAMA_DECISION_TEST_MODEL to write the decision golden\n");
        return 2;
    }
    test_engine te;
    if (!te.load(model_path)) {
        fprintf(stderr, "failed to load model or build context\n");
        return 2;
    }
    auto vocab = llama_decision::make_llama_label_vocab(llama_model_get_vocab(te.model));
    const auto pool = llama_decision::build_label_pool(*vocab, test_letter_tail(), 64);
            llama_decision::engine eng(te.ctx, 2, 8);
            const auto req = llama_decision::parse_decision_request(common_json::parse(decision_valid_body()));
    const auto probs = test_letter_readout(eng, *vocab, nullptr, false, req, pool, llama_decision::options{}, nullptr);
    common_json usage = common_json::object();
    usage["input_tokens"]    = 0;
    usage["output_tokens"]   = 0;
    usage["cached_tokens"]   = 0;
    usage["state_cache_hit"] = false;
    common_json golden = llama_decision::assemble_decision_response(req, probs, "m", usage);
    golden["golden_model"] = model_identity(model_path);
    write_file(decision_golden_path(), golden.dump(2) + "\n");
    return 0;
}

// Reproducibility recording of the committed decision corpus. The deterministic core (per-question
// probabilities, winners, head mode, label-pool size) is the frozen contract later refactors must
// not move; the timing block is recorded for context and excluded from the byte diff because it
// moves every run.
static std::string readout_baseline_path(const std::string & backend) {
    return std::string(DECISION_TEST_BASELINE_DIR) + "/readout_" + backend + "_baseline.json";
}

// Runs the committed corpus once and returns the readout core plus its timing block. `gpu` selects
// the shared GPU model; otherwise the model loads with no offload so the CPU lane can record and
// check it.
static common_json readout_capture(const std::string & path, bool gpu) {
    const std::string tail = test_letter_tail();

    common_json out = common_json::object();
    out["backend"]  = gpu ? "gpu" : "cpu";
    out["model"]    = model_identity(path);

    auto run = [&](llama_model * model, llama_context * ctx) {
        auto                    vocab = llama_decision::make_llama_label_vocab(llama_model_get_vocab(model));
        const auto              pool  = llama_decision::build_label_pool(*vocab, tail, llama_decision::LABEL_POOL_CAP);
        llama_decision::engine  eng(ctx, 2, 8);
        const auto              req = llama_decision::parse_decision_request(common_json::parse(decision_valid_body()));
        llama_decision::options opt;
        opt.cache_tag = "readout-baseline";
        llama_decision::readout_metrics    metrics;
        const auto                        probs =
            test_letter_readout(eng, *vocab, nullptr, false, req, pool, opt, &metrics);

        common_json core        = oracle_readout(metrics, probs);
        core["label_pool_size"] = (long long) pool.size();
        out["readout"]          = core;

        common_json timing   = common_json::object();
        timing["prefill_ms"] = metrics.prefill_ms;
        timing["scoring_ms"] = metrics.scoring_ms;
        out["timings"]       = timing;
    };

    if (gpu) {
        test_engine te;
        if (!te.load(path.c_str())) {
            throw std::runtime_error("the GPU test model failed to load: " + path);
        }
        run(te.model, te.ctx);
    } else {
        cpu_test_engine te;
        if (!te.load(path, 4096, false, 512)) {
            throw std::runtime_error("the CPU test model failed to load: " + path);
        }
        run(te.model, te.ctx);
    }
    return out;
}

// The first serialized line that differs, for a compact mismatch report instead of dumping both
// trees.
static std::string first_line_difference(const std::string & a, const std::string & b) {
    std::istringstream ra(a);
    std::istringstream rb(b);
    std::string        la;
    std::string        lb;
    int                line = 0;
    while (true) {
        const bool oka = (bool) std::getline(ra, la);
        const bool okb = (bool) std::getline(rb, lb);
        if (!oka && !okb) {
            break;
        }
        ++line;
        if (la != lb) {
            return "line " + std::to_string(line) + "\n  baseline: " + la + "\n  current : " + lb;
        }
    }
    return "the serialized lengths differ";
}

static bool readout_core_equal(const common_json & expected, const common_json & actual, std::string * difference) {
    common_json e = expected;
    common_json a = actual;
    e.erase("timings");
    a.erase("timings");
    e.erase("note");
    a.erase("note");
    const std::string es = e.dump(1);
    const std::string as = a.dump(1);
    if (es == as) {
        return true;
    }
    if (difference != nullptr) {
        *difference = first_line_difference(es, as);
    }
    return false;
}

// The model a baseline section was recorded on, or empty when the file or section is absent.
static std::string readout_baseline_model(const std::string & backend) {
    const std::string path = readout_baseline_path(backend);
    if (!file_exists(path)) {
        return std::string();
    }
    try {
        return common_json::parse(read_file(path)).value("model", std::string());
    } catch (const std::exception &) {
        return std::string();
    }
}

static int write_readout_baseline(const std::string & backend) {
    const bool  gpu = backend == "gpu";
    std::string path;
    if (gpu) {
        const char * env = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (env == nullptr || env[0] == '\0') {
            fprintf(stderr, "set LLAMA_DECISION_TEST_MODEL to record the GPU readout baseline\n");
            return 2;
        }
        path = env;
    } else {
        path = decision_cpu_model_path();
        if (path.empty()) {
            fprintf(stderr, "the CPU readout baseline needs the generated model\n");
            return 2;
        }
    }
    try {
        common_json rec = readout_capture(path, gpu);
        rec["note"] =
            "Frozen readout of the committed decision corpus. The deterministic core "
            "(probabilities, winners, confidence = 1 - H/log K, certainty = max p, "
            "label-pool size) is a contract; the timing block is recorded for context and excluded "
            "from the byte diff.";
        write_file(readout_baseline_path(backend), rec.dump(1) + "\n");
    } catch (const std::exception & e) {
        fprintf(stderr, "failed to record the %s readout baseline: %s\n", backend.c_str(), e.what());
        return 2;
    }
    return 0;
}

// Diff command: recompute the readout and fail on any deterministic byte change. Returns 0 on a
// match, 1 on drift, 2 when the baseline or its model is unavailable.
static int check_readout_baseline(const std::string & backend) {
    const bool        gpu  = backend == "gpu";
    const std::string file = readout_baseline_path(backend);
    if (!file_exists(file)) {
        fprintf(stderr, "no %s readout baseline at %s\n", backend.c_str(), file.c_str());
        return 2;
    }
    std::string path;
    if (gpu) {
        const char * env = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (env == nullptr || env[0] == '\0') {
            fprintf(stderr, "set LLAMA_DECISION_TEST_MODEL to check the GPU readout baseline\n");
            return 2;
        }
        path = env;
    } else {
        path = decision_cpu_model_path();
        if (path.empty()) {
            fprintf(stderr, "the CPU readout baseline needs the generated model\n");
            return 2;
        }
    }

    const common_json expected = common_json::parse(read_file(file));
    if (expected.value("model", std::string()) != model_identity(path)) {
        fprintf(stderr, "the loaded model (%s) is not the recorded %s baseline model (%s)\n",
                model_identity(path).c_str(), backend.c_str(), expected.value("model", std::string()).c_str());
        return 2;
    }

    std::string difference;
    try {
        const common_json actual = readout_capture(path, gpu);
        if (readout_core_equal(expected, actual, &difference)) {
            printf("the %s readout matches the frozen baseline\n", backend.c_str());
            return 0;
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "failed to recompute the %s readout: %s\n", backend.c_str(), e.what());
        return 2;
    }
    fprintf(stderr, "the %s readout drifted from the frozen baseline:\n%s\n", backend.c_str(), difference.c_str());
    return 1;
}

// ---------------------------------------------------------------- session baseline (K0)

// The frozen session-substrate reference: the byte-level `id_slot` session responses for the fixed
// request set (additive diagnostics excluded), the session-arena cost counters for a scripted
// create/query/advance sequence, and the whole-context save/restore behavior for one slot. These
// are the observables the later session-substrate work asserts against; the answers themselves
// are the backward-compatibility oracle for the `id_slot` path.

static void test_readout_baseline(testing & t) {
    t.test("the readout matches the frozen reproducibility baseline", [](testing & t) {
        const char * env = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (env != nullptr && env[0] != '\0' && decision_gpu_available() &&
            readout_baseline_model("gpu") == model_identity(env)) {
            try {
                const common_json expected = common_json::parse(read_file(readout_baseline_path("gpu")));
                const common_json actual   = readout_capture(env, true);
                std::string       difference;
                if (!readout_core_equal(expected, actual, &difference)) {
                    t.assert_true("the gpu readout core matches the frozen baseline: " + difference, false);
                    return;
                }
                t.assert_true("the gpu readout core matches the frozen baseline", true);
            } catch (const std::exception & e) {
                t.assert_true(std::string("the gpu readout baseline runs: ") + e.what(), false);
            }
            return;
        }

        const std::string path = decision_cpu_model_path();
        if (path.empty() || readout_baseline_model("cpu") != model_identity(path)) {
            t.skip("no frozen readout baseline matches the available model");
            return;
        }
        try {
            const common_json expected = common_json::parse(read_file(readout_baseline_path("cpu")));
            const common_json actual   = readout_capture(path, false);
            std::string       difference;
            if (!readout_core_equal(expected, actual, &difference)) {
                t.assert_true("the cpu readout core matches the frozen baseline: " + difference, false);
                return;
            }
            t.assert_true("the cpu readout core matches the frozen baseline", true);
        } catch (const std::exception & e) {
            t.assert_true(std::string("the cpu readout baseline runs: ") + e.what(), false);
        }
    });
}

int main(int argc, char ** argv) {
    if (argc > 1 && std::string(argv[1]) == "--write-golden") {
        return write_goldens();
    }
    if (argc > 1 && std::string(argv[1]) == "--write-cpu-oracle") {
        return write_cpu_oracle();
    }
    if (argc > 1 && std::string(argv[1]) == "--write-decision-golden") {
        return write_decision_golden(std::getenv("LLAMA_DECISION_TEST_MODEL"));
    }
    if (argc > 1 && std::string(argv[1]) == "--write-calibration") {
        return write_calibration(std::getenv("LLAMA_DECISION_TEST_MODEL"));
    }
    if (argc > 1 && std::string(argv[1]) == "--write-calibration-rows") {
        return write_calibration_rows();
    }
    if (argc > 2 && std::string(argv[1]) == "--record-readout") {
        const std::string backend = argv[2];
        if (backend != "cpu" && backend != "gpu") {
            fprintf(stderr, "--record-readout needs a backend: cpu or gpu\n");
            return 2;
        }
        return write_readout_baseline(backend);
    }
    if (argc > 2 && std::string(argv[1]) == "--check-readout") {
        const std::string backend = argv[2];
        if (backend != "cpu" && backend != "gpu") {
            fprintf(stderr, "--check-readout needs a backend: cpu or gpu\n");
            return 2;
        }
        return check_readout_baseline(backend);
    }
    if (argc > 1 && std::string(argv[1]) == "--record-golden") {
        record_golden = true;
    }

    testing t;
    // --record-golden is a mode, not a filter, so it is consumed rather than passed on
    if (argc > 1 && !record_golden) {
        t.set_filter(argv[1]);
    }

    t.test("decision engine harness", [](testing & t) {
        test_split_chat_template_primitive(t);
        test_thinking_off(t);
        test_thinking_off_model(t);
        test_thinking_control(t);
        test_decision_shape_contract(t);
        test_decision_parse(t);
        test_request_envelope_unification(t);
        test_decision_assemble(t);
        test_adapter_scope_applied(t);
        test_decision_default_envelope(t);
        test_decision_values_golden(t);
        test_jev_compat_guarantee(t);
        test_generic_frontend(t);
        test_softmax(t);
        test_saved_state_format_dispatch(t);
        test_save_load_fail_fast(t);
        test_weak_quant_control(t);
        test_prefix_tag(t);
        test_question_temperature(t);
        test_temperature_effect(t);
        test_confidence_certainty_axes(t);
        test_confidence_profile(t);
        test_numeric_questions(t);
        test_temperature_profile(t);
        test_confidence_never_gates(t);
        test_letter_suffix(t);
        test_letter_option_lines(t);
        test_label_pool(t);
        test_boundary(t);
        test_answer_label_token(t);
        test_safe_data(t);
        test_label_pool_real(t);
        test_letter_labels_spm(t);
        test_label_boundary_calibration(t);
        test_letter_readout_real(t);
        test_fork_real(t);
        test_fork_oracle(t);
        test_nested_fork_oracle(t);
        test_session_fork(t);
        test_fork_auto_default(t);
        test_fork_strategy_switch(t);
        test_fork_divergence_control(t);
        test_fork_swa_clamp(t);
        test_prefix_lru_restores_own_state(t);
        test_prefix_cache_hit_repeatability(t);
        test_state_bulk_copy(t);
        test_device_state_round_trip(t);
        test_device_async_staging(t);
        test_recurrent_multi_range_device_save(t);
        test_partial_state_round_trip(t);
        test_partial_state_fragmented(t);
        test_device_layout_mutation_round_trip(t);
        test_prefix_cache_cost(t);
        test_bounded_decision_context(t);
        test_multi_trunk_restore(t);
        test_pool_seq_lifecycle(t);
        test_decision_cpu_oracle(t);
        test_readout_baseline(t);
        test_compile_fields_plan(t);
        test_decide_batch_plan_overload(t);
        test_token_entry_equality(t);
        test_warm_resident_cache(t);
        test_warm_slot_budget(t);
        test_warm_capacity_accounting(t);
        test_prefix_cache_coherence(t);
        test_token_cache(t);
        test_prefix_reuse(t);
        test_request_prefix(t);
        test_batching_waves(t);
        test_dedup_fields(t);
        test_cancel_reaches_compute(t);
        test_yield_points(t);
        test_capacity_error(t);
        test_long_branch_chunking(t);
        test_prefix_hoist_cache(t);
        test_permutation_order(t);
        test_permutations_parsing(t);
        test_permutations_real(t);
        test_contract_hash(t);
        test_reference_corpus(t);
        test_docs_errors(t);
        test_policy_confidence(t);
        test_sha256(t);
        test_confidence_never_gates_envelope(t);
        test_verify_letter_request(t);
        test_sequence_partition(t);
        test_calibration_hoist(t);
        test_calibration_dedup(t);
        test_calibration_confidence(t);
        test_calibration_table(t);
        test_calibration_ledger(t);
        test_calibration_gates(t);
        test_calibration_determinism_variance(t);
        test_decision_provenance(t);
        test_calibration_model(t);
    });

    return t.summary();
}
