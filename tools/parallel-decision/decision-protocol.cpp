#include "decision-protocol.h"

#include "chat.h"
#include "labels.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace llama_decision {

uint64_t fnv1a64(const std::string & s) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ull;
    }
    return h;
}

// The stable per-(id, pass) shuffle seed. Distinct passes get distinct orders; pass 0 is identity.
static uint64_t permutation_seed(const std::string & id, int pass) {
    uint64_t h = fnv1a64(id);
    h ^= (uint64_t) (uint32_t) pass * 0x9e3779b97f4a7c15ull;
    h *= 1099511628211ull;
    return h != 0 ? h : 1;
}

// A seeded distinct permutation of `count` indices for `id` at `pass`. Pass 0 (and any pass on a
// single option) is the identity; a later pass is a Fisher-Yates shuffle that is forced to differ
// from the identity so every pass actually de-biases order. Shared by the Jev and generic readouts.
std::vector<size_t> permutation_order(size_t count, const std::string & id, int pass) {
    std::vector<size_t> order(count);
    for (size_t i = 0; i < count; ++i) {
        order[i] = i;
    }
    if (pass <= 0 || count < 2) {
        return order;
    }
    uint64_t s = permutation_seed(id, pass);
    for (size_t i = count - 1; i > 0; --i) {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        std::swap(order[i], order[(size_t) (s % (i + 1))]);
    }
    bool identity = true;
    for (size_t i = 0; i < count; ++i) {
        if (order[i] != i) {
            identity = false;
            break;
        }
    }
    if (identity) {
        std::swap(order[count - 2], order[count - 1]); // a pass must actually reorder to de-bias
    }
    return order;
}

std::pair<std::string, std::string> split_chat_template(const common_chat_templates * tmpls, bool use_jinja,
                                                        const std::string & system_text, bool enable_thinking) {
    static const std::string sentinel = "\x1f<<decision-context>>\x1f";
    common_chat_templates_inputs in;
    in.use_jinja             = use_jinja;
    in.add_generation_prompt = true;
    in.enable_thinking       = enable_thinking;
    common_chat_msg sys;
    sys.role    = "system";
    sys.content = system_text;
    common_chat_msg usr;
    usr.role    = "user";
    usr.content = sentinel;
    in.messages = { sys, usr };
    const std::string prompt = common_chat_templates_apply(tmpls, in).prompt;
    const size_t at = prompt.find(sentinel);
    if (at == std::string::npos) {
        throw std::runtime_error("the chat template did not keep the user message");
    }
    return { prompt.substr(0, at), prompt.substr(at + sentinel.size()) };
}

// Structured criteria values keep their shape as text for the model; null is empty.
std::string render_text(const common_json & v) {
    if (v.is_string()) {
        return v.get<std::string>();
    }
    if (v.is_null()) {
        return std::string();
    }
    return v.dump();
}

namespace {

bool is_textual(const common_json & v) {
    return v.is_string() || v.is_object() || v.is_array();
}

// The number of decimal places a value needs to round-trip, so a grid built by
// `lo + i * step` can be rendered with fixed width and no float noise ("0.3",
// never "0.30000000000000004").
int decimal_places(double x) {
    for (int k = 0; k <= 9; ++k) {
        const double v = x * std::pow(10.0, k);
        if (std::fabs(v - std::llround(v)) < 1e-9 * std::max(1.0, std::fabs(v))) {
            return k;
        }
    }
    return 9;
}

void check_allowed_keys(const common_json & obj, std::initializer_list<const char *> allowed, const std::string & where) {
    for (const auto & e : obj.items()) {
        bool ok = false;
        for (const char * k : allowed) {
            if (e.key() == k) {
                ok = true;
                break;
            }
        }
        if (!ok) {
            throw semantic_error(where + ": unknown field \"" + e.key() + "\"");
        }
    }
}

std::string canonical_type(const std::string & t) {
    if (t == "bool") {
        return "noul";
    }
    if (t == "scale") {
        return "score";
    }
    return t;
}

// Validates a "temperatures" object: known keys only, each a number > 0. When `out` is given the
// parsed values are stored; otherwise the object is only checked. Shared by the request parser and
// the standalone temperature profile so the two cannot drift.
void read_temperatures(const common_json & temps, std::map<std::string, double> * out) {
    if (!temps.is_object()) {
        throw semantic_error("temperatures must be an object");
    }
    for (const auto & e : temps.items()) {
        if (e.key() != "noul" && e.key() != "choice" && e.key() != "score" &&
            e.key() != "integer" && e.key() != "number") {
            throw semantic_error("temperatures: unknown field \"" + e.key() + "\"");
        }
        if (!e.value().is_number() || !(e.value().get<double>() > 0.0)) {
            throw semantic_error("temperatures." + e.key() + " must be a number > 0");
        }
        if (out != nullptr) {
            (*out)[e.key()] = e.value().get<double>();
        }
    }
}

// A score is an expected zero-based index, so the level count is the number of criteria entries.
// Shared by the array and legend-object forms; 2 is the smallest meaningful scale and 10 bounds the
// answer-row fan-out.
void check_score_levels(const std::string & id, size_t levels) {
    if (levels < DECISION_MIN_OPTIONS || levels > DECISION_MAX_SCORE_LEVELS) {
        throw semantic_error("question \"" + id + "\": score needs " + std::to_string(DECISION_MIN_OPTIONS) +
                             "-" + std::to_string(DECISION_MAX_SCORE_LEVELS) + " levels");
    }
}

void validate_state(const common_json & s) {
    if (s.is_string()) {
        if (s.get<std::string>().empty()) {
            throw semantic_error("state must not be empty");
        }
        return;
    }
    if (s.is_array() || s.is_object()) {
        if (s.size() == 0) {
            throw semantic_error("state must not be empty");
        }
        return;
    }
    throw semantic_error("state must be a string, object or array");
}

decision_question parse_question(const std::string & id, const common_json & spec) {
    if (!spec.is_object()) {
        throw semantic_error("question \"" + id + "\" must be an object");
    }

    if (!spec.contains("type") || !spec.at("type").is_string()) {
        throw semantic_error("question \"" + id + "\" needs a string \"type\"");
    }
    const std::string q_type = canonical_type(spec.at("type").get<std::string>());
    const bool numeric = q_type == "integer" || q_type == "number";
    if (q_type != "noul" && q_type != "choice" && q_type != "score" && !numeric) {
        throw semantic_error("question \"" + id + "\": unknown type \"" + q_type + "\"");
    }

    if (numeric) {
        check_allowed_keys(spec, { "type", "instructions", "minimum", "maximum", "step", "multipleOf", "aggregate" },
                           "question \"" + id + "\"");
    } else {
        check_allowed_keys(spec, { "type", "instructions", "criteria" }, "question \"" + id + "\"");
    }

    decision_question q;
    q.id          = id;
    q.type        = q_type;
    q.has_criteria = spec.contains("criteria") && !spec.at("criteria").is_null();

    if (spec.contains("instructions")) {
        const common_json & ins = spec.at("instructions");
        if (!ins.is_null() && !is_textual(ins)) {
            throw semantic_error("question \"" + id + "\": instructions must be a string, object or array");
        }
        q.instructions = ins;
    }

    if (q.type == "noul") {
        decision_option no;
        decision_option yes;
        no.key  = "false";
        yes.key = "true";
        if (q.has_criteria) {
            const common_json & crit = spec.at("criteria");
            if (!crit.is_object()) {
                throw semantic_error("question \"" + id + "\": noul criteria must be an object");
            }
            for (const auto & e : crit.items()) {
                if (e.key() != "true" && e.key() != "false") {
                    throw semantic_error("question \"" + id + "\": noul criteria keys must be \"true\" and \"false\"");
                }
            }
            if (crit.contains("false")) {
                no.description = render_text(crit.at("false"));
            }
            if (crit.contains("true")) {
                yes.description = render_text(crit.at("true"));
            }
        }
        q.options = { no, yes };
    } else if (q.type == "choice") {
        if (!q.has_criteria || !spec.at("criteria").is_object()) {
            throw semantic_error("question \"" + id + "\": choice needs an object \"criteria\"");
        }
        const common_json & crit = spec.at("criteria");
        if (crit.size() < DECISION_MIN_OPTIONS || crit.size() > DECISION_MAX_CHOICE_OPTIONS) {
            throw semantic_error("question \"" + id + "\": choice needs " + std::to_string(DECISION_MIN_OPTIONS) +
                                 "-" + std::to_string(DECISION_MAX_CHOICE_OPTIONS) + " options");
        }
        for (const auto & e : crit.items()) {
            if (e.key().empty()) {
                throw semantic_error("question \"" + id + "\": option keys must not be empty");
            }
            decision_option o;
            o.key         = e.key();
            o.description = render_text(e.value());
            o.original    = e.value();
            q.options.push_back(o);
        }
    } else if (q.type == "score") {
        if (!q.has_criteria) {
            throw semantic_error("question \"" + id + "\": score needs \"criteria\"");
        }
        const common_json & crit = spec.at("criteria");
        if (crit.is_array()) {
            check_score_levels(id, crit.size());
            for (size_t i = 0; i < crit.size(); ++i) {
                decision_option o;
                o.key         = std::to_string(i);
                o.description = render_text(crit.at(i));
                o.original    = crit.at(i);
                q.options.push_back(o);
            }
        } else if (crit.is_object()) {
            check_score_levels(id, crit.size());
            size_t i = 0;
            for (const auto & e : crit.items()) {
                if (e.key() != std::to_string(i)) {
                    throw semantic_error("question \"" + id + "\": score legend keys must be \"0\"..\"K-1\" in order");
                }
                decision_option o;
                o.key         = std::to_string(i);
                o.description = render_text(e.value());
                o.original    = e.value();
                q.options.push_back(o);
                ++i;
            }
        } else {
            throw semantic_error("question \"" + id + "\": score criteria must be an array or a legend object");
        }
    } else if (q.type == "integer") {
        // A range-generated typed grid: minimum..maximum inclusive, scored by label exactly like a
        // choice. The typed value rides in `original`, so the answer and the aggregates are numbers.
        if (!spec.contains("minimum") || !spec.at("minimum").is_number_integer() ||
            !spec.contains("maximum") || !spec.at("maximum").is_number_integer()) {
            throw semantic_error("question \"" + id + "\": integer needs integer minimum and maximum");
        }
        const long long lo = spec.at("minimum").get<long long>();
        const long long hi = spec.at("maximum").get<long long>();
        const size_t    n  = (size_t) (hi - lo) + 1;
        if (hi < lo || n < DECISION_MIN_OPTIONS || n > DECISION_MAX_NUMERIC_VALUES) {
            throw semantic_error("question \"" + id + "\": integer bounds must define " +
                                 std::to_string(DECISION_MIN_OPTIONS) + "-" +
                                 std::to_string(DECISION_MAX_NUMERIC_VALUES) + " values");
        }
        for (long long v = lo; v <= hi; ++v) {
            decision_option o;
            o.key      = std::to_string(v);
            o.original = common_json(v);
            q.options.push_back(std::move(o));
        }
    } else { // number
        if (spec.contains("step") && spec.contains("multipleOf")) {
            throw semantic_error("question \"" + id + "\": provide step or multipleOf, not both");
        }
        const char * step_key = spec.contains("step") ? "step" : (spec.contains("multipleOf") ? "multipleOf" : nullptr);
        if (step_key == nullptr ||
            !spec.contains("minimum") || !spec.contains("maximum") ||
            !spec.at("minimum").is_number() || !spec.at("maximum").is_number() ||
            !spec.at(step_key).is_number()) {
            throw semantic_error("question \"" + id + "\": number needs minimum, maximum and step (or multipleOf)");
        }
        const double lo   = spec.at("minimum").get<double>();
        const double hi   = spec.at("maximum").get<double>();
        const double step = spec.at(step_key).get<double>();
        if (!(step > 0) || !(hi >= lo)) {
            throw semantic_error("question \"" + id + "\": number needs ordered bounds and a positive step");
        }
        // the shared grid encoder (D5): fixed-width text, both ends included, no float noise
        const std::vector<numeric_grid_value> grid = numeric_grid(lo, hi, step);
        for (const auto & gv : grid) {
            decision_option o;
            o.key      = gv.text;
            o.original = common_json(gv.value);
            q.options.push_back(std::move(o));
        }
    }

    if (q.type == "integer" || q.type == "number") {
        if (spec.contains("aggregate") && !spec.at("aggregate").is_null()) {
            if (!spec.at("aggregate").is_string()) {
                throw semantic_error("question \"" + id + "\": aggregate must be a string");
            }
            q.aggregate = spec.at("aggregate").get<std::string>();
            if (q.aggregate != "mode" && q.aggregate != "median" && q.aggregate != "mean") {
                throw semantic_error("question \"" + id + "\": aggregate must be mode, median or mean");
            }
        }
    }

    if (!spec.contains("instructions") || spec.at("instructions").is_null()) {
        throw semantic_error("question \"" + id + "\": instructions are required");
    }
    return q;
}

} // namespace

// Normalized inverse entropy, 1 - H/log K. The producer's self-doubt axis: it measures how
// concentrated the distribution is, never whether the winner is correct.
double inverse_entropy_confidence(const std::vector<float> & p) {
    const double k = (double) p.size();
    if (k <= 1.0) {
        return 1.0;
    }
    double h = 0.0;
    for (float x : p) {
        if (x > 0.0f) {
            h -= (double) x * std::log((double) x);
        }
    }
    const double v = 1.0 - h / std::log(k);
    return std::min(1.0, std::max(0.0, v));
}

// The winner's share, max(p). A separate axis from the entropy confidence: two distributions can
// share a winner share but differ in shape.
double winner_share(const std::vector<float> & p) {
    return p.empty() ? 0.0 : (double) *std::max_element(p.begin(), p.end());
}

// The Jev Choice confidence, (N*p_max - 1)/(N - 1), clamped. Jev documents it for Choice and for
// Score at 2..3 levels; above three the definition is undocumented, so the same rule is the stated
// local value. N below 2 has no meaningful rescale, so a single option is fully confident.
double jev_winner_share_confidence(const std::vector<float> & p) {
    const double n = (double) p.size();
    if (n <= 1.0) {
        return 1.0;
    }
    const double v = (n * winner_share(p) - 1.0) / (n - 1.0);
    return std::min(1.0, std::max(0.0, v));
}

// The two producer-concentration numbers for a distribution, keyed for JSON. `confidence` is the
// Jev value or the opt-in entropy value per `confidence_profile`; `certainty` is the winner's
// share. Shared by the Jev and generic readouts so the profile selection lives in one place.
common_json concentration_metrics(const std::vector<float> & p, const std::string & confidence_profile) {
    common_json out = common_json::object();
    out["confidence"] = confidence_profile == "local" ? inverse_entropy_confidence(p)
                                                      : jev_winner_share_confidence(p);
    out["certainty"]  = winner_share(p);
    return out;
}

bool temperature_provenance::operator==(const temperature_provenance & other) const {
    return model == other.model && quantization == other.quantization &&
           template_hash == other.template_hash && backend_flags == other.backend_flags;
}

temperature_profile parse_temperature_profile(const common_json & doc) {
    if (!doc.is_object()) {
        throw semantic_error("temperature profile must be an object");
    }
    temperature_profile profile;
    if (doc.contains("provenance")) {
        const common_json & prov = doc.at("provenance");
        if (!prov.is_object()) {
            throw semantic_error("provenance must be an object");
        }
        auto read = [&prov](const char * key, std::string & out) {
            if (prov.contains(key)) {
                if (!prov.at(key).is_string()) {
                    throw semantic_error(std::string("provenance.") + key + " must be a string");
                }
                out = prov.at(key).get<std::string>();
            }
        };
        read("model", profile.provenance.model);
        read("quantization", profile.provenance.quantization);
        read("template_hash", profile.provenance.template_hash);
        read("backend_flags", profile.provenance.backend_flags);
    }
    if (doc.contains("temperatures")) {
        read_temperatures(doc.at("temperatures"), &profile.temperatures);
    }
    return profile;
}

void validate_temperature_profile(const temperature_profile & profile, const temperature_provenance & current) {
    bool any_non_default = false;
    for (const auto & kv : profile.temperatures) {
        if (kv.second != 1.0) {
            any_non_default = true;
        }
    }
    if (!any_non_default) {
        return;
    }
    if (profile.provenance != current) {
        throw semantic_error("temperature profile provenance does not match the running configuration");
    }
}

double question_temperature(const decision_request & req, const decision_question & q) {
    if (req.temperatures.is_object() && req.temperatures.contains(q.type)) {
        const common_json & v = req.temperatures.at(q.type);
        if (v.is_number()) {
            return v.get<double>();
        }
    }
    return req.temperature;
}

session_ref parse_session_ref(const common_json & body) {
    session_ref ref;
    if (!body.is_object()) {
        return ref;
    }
    if (body.contains("session_id") && !body.at("session_id").is_null()) {
        const common_json & v = body.at("session_id");
        if (!v.is_string() || v.get<std::string>().empty()) {
            throw semantic_error("session_id must be a non-empty string");
        }
        ref.session_id = v.get<std::string>();
        ref.present    = true;
    }
    if (body.contains("id_slot") && !body.at("id_slot").is_null()) {
        const common_json & v = body.at("id_slot");
        if (!v.is_number_integer()) {
            throw semantic_error("id_slot must be an integer");
        }
        if (!ref.session_id.empty()) {
            throw semantic_error("provide either session_id or id_slot, not both");
        }
        ref.id_slot = (int) v.get<long long>();
        if (ref.id_slot < 0) {
            throw semantic_error("id_slot must be >= 0");
        }
        ref.present = true;
    }
    if (body.contains("session_pos") && !body.at("session_pos").is_null()) {
        const common_json & v = body.at("session_pos");
        if (!v.is_number_integer()) {
            throw semantic_error("session_pos must be an integer");
        }
        ref.session_pos = (int) v.get<long long>();
        if (ref.session_pos < 0) {
            throw semantic_error("session_pos must be >= 0");
        }
        if (!ref.present) {
            throw semantic_error("session_pos requires id_slot or session_id");
        }
    }
    if (body.contains("turn") && !body.at("turn").is_null()) {
        const common_json & v = body.at("turn");
        if (!v.is_string()) {
            throw semantic_error("turn must be a string");
        }
        ref.turn = v.get<std::string>();
    }
    return ref;
}

decision_evidence parse_evidence(const common_json & body) {
    decision_evidence out;
    if (!body.is_object()) {
        return out;
    }
    if (body.contains("contexts") && !body.at("contexts").is_null()) {
        if (!body.at("contexts").is_array() || body.at("contexts").empty() ||
            body.at("contexts").size() > DECISION_MAX_CONTEXTS) {
            throw semantic_error("contexts must hold 1-" + std::to_string(DECISION_MAX_CONTEXTS) + " entries");
        }
        if (body.contains("state") && !body.at("state").is_null()) {
            throw semantic_error("provide either state or contexts, not both");
        }
        for (const auto & c : body.at("contexts")) {
            if (!c.is_string() || c.get<std::string>().empty()) {
                throw semantic_error("every entry of contexts must be a non-empty string");
            }
            out.contexts.push_back(c.get<std::string>());
        }
        return out;
    }
    if (!body.contains("state") || body.at("state").is_null()) {
        return out;
    }
    validate_state(body.at("state"));
    out.state_present = true;
    out.state         = body.at("state");
    return out;
}

std::string render_state(const common_json & state) {
    const std::string text = state.is_string() ? state.get<std::string>() : state.dump();
    return "State:\n" + safe_data(text) + "\n";
}

std::vector<numeric_grid_value> numeric_grid(double lo, double hi, double step) {
    const double    count = (hi - lo) / step;
    const long long n     = std::llround(count);
    if (std::fabs(count - (double) n) > 1e-7) {
        throw semantic_error("the number grid must include both ends");
    }
    if (n < 0 || n + 1 < (long long) DECISION_MIN_OPTIONS || n + 1 > (long long) DECISION_MAX_NUMERIC_VALUES) {
        throw semantic_error("the number grid must hold " + std::to_string(DECISION_MIN_OPTIONS) + "-" +
                             std::to_string(DECISION_MAX_NUMERIC_VALUES) + " values");
    }
    const int places = std::max({ decimal_places(lo), decimal_places(hi), decimal_places(step) });
    std::vector<numeric_grid_value> out;
    out.reserve((size_t) n + 1);
    for (long long i = 0; i <= n; ++i) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.*f", places, lo + (double) i * step);
        const double v = std::strtod(buf, nullptr); // drop float noise: 0.1 + 0.2 -> "0.3"
        out.push_back({ v, buf });
    }
    return out;
}

decision_request parse_decision_request(const common_json & body) {
    if (!body.is_object()) {
        throw semantic_error("request must be an object");
    }
    // Unknown top-level fields are tolerated for Jev compatibility; unknown fields inside a
    // question are still refused by parse_question.

    decision_request req;

    // The model is required: an external router in front of this server selects the model, and
    // the response echoes it back. Jev requires it too, so a missing field is a 422 naming it.
    if (!body.contains("model") || body.at("model").is_null()) {
        throw semantic_error("model is required");
    }
    if (!body.at("model").is_string()) {
        throw semantic_error("model must be a string");
    }
    req.model = body.at("model").get<std::string>();

    if (body.contains("contexts") && !body.at("contexts").is_null()) {
        if (!body.at("contexts").is_array() || body.at("contexts").empty() ||
            body.at("contexts").size() > DECISION_MAX_CONTEXTS) {
            throw semantic_error("contexts must hold 1-" + std::to_string(DECISION_MAX_CONTEXTS) + " entries");
        }
        if (body.contains("state") && !body.at("state").is_null()) {
            throw semantic_error("provide either state or contexts, not both");
        }
        for (const auto & c : body.at("contexts")) {
            if (!c.is_string() || c.get<std::string>().empty()) {
                throw semantic_error("every entry of contexts must be a non-empty string");
            }
            req.contexts.push_back(c);
        }
    } else {
        if (!body.contains("state")) {
            throw semantic_error("state (or contexts) is required");
        }
        req.state = body.at("state");
        validate_state(req.state);
    }

    if (!body.contains("questions") || !body.at("questions").is_object()) {
        throw semantic_error("questions must be an object");
    }
    const common_json & qs = body.at("questions");
    if (qs.size() < DECISION_MIN_QUESTIONS || qs.size() > DECISION_MAX_QUESTIONS) {
        throw semantic_error("questions must hold " + std::to_string(DECISION_MIN_QUESTIONS) + "-" +
                             std::to_string(DECISION_MAX_QUESTIONS) + " entries");
    }
    for (const auto & e : qs.items()) {
        req.questions.push_back(parse_question(e.key(), e.value()));
    }

    if (body.contains("temperature") && !body.at("temperature").is_null()) {
        if (!body.at("temperature").is_number()) {
            throw semantic_error("temperature must be a number");
        }
        req.temperature = body.at("temperature").get<double>();
        if (!(req.temperature > 0.0)) {
            throw semantic_error("temperature must be > 0");
        }
    }

    if (body.contains("temperatures") && !body.at("temperatures").is_null()) {
        read_temperatures(body.at("temperatures"), nullptr);
        req.temperatures = body.at("temperatures");
    }

    if (body.contains("permutations") && !body.at("permutations").is_null()) {
        const common_json & v = body.at("permutations");
        if (!v.is_number_integer()) {
            throw semantic_error("permutations must be an integer");
        }
        req.permutations = (int) v.get<long long>();
        if (req.permutations < 1) {
            throw semantic_error("permutations must be >= 1");
        }
        if (req.permutations > DECISION_MAX_PERMUTATIONS) {
            req.permutations = DECISION_MAX_PERMUTATIONS; // accepted but capped: more passes only add cost
        }
    }

    if (body.contains("diagnostics") && !body.at("diagnostics").is_null()) {
        if (!body.at("diagnostics").is_boolean()) {
            throw semantic_error("diagnostics must be a boolean");
        }
        req.diagnostics = body.at("diagnostics").get<bool>();
    }

    if (body.contains("confidence_profile") && !body.at("confidence_profile").is_null()) {
        if (!body.at("confidence_profile").is_string()) {
            throw semantic_error("confidence_profile must be a string");
        }
        req.confidence_profile = body.at("confidence_profile").get<std::string>();
        if (req.confidence_profile != "local" && req.confidence_profile != "jev") {
            throw semantic_error("confidence_profile must be local or jev");
        }
    }

    req.session = parse_session_ref(body);

    return req;
}

std::vector<std::vector<float>> uniform_probs(const decision_request & req) {
    std::vector<std::vector<float>> out;
    out.reserve(req.questions.size());
    for (const auto & q : req.questions) {
        const float p = q.options.empty() ? 0.0f : 1.0f / (float) q.options.size();
        out.emplace_back(q.options.size(), p);
    }
    return out;
}

std::string sha256_hex(const std::string & text) {
    static const uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    };
    auto rotr = [](uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };

    std::vector<uint8_t> msg(text.begin(), text.end());
    const uint64_t bit_len = (uint64_t) msg.size() * 8;
    msg.push_back(0x80);
    while (msg.size() % 64 != 56) {
        msg.push_back(0);
    }
    for (int i = 7; i >= 0; --i) {
        msg.push_back((uint8_t) ((bit_len >> (8 * i)) & 0xff));
    }

    uint32_t h[8] = { 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };
    for (size_t off = 0; off < msg.size(); off += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = ((uint32_t) msg[off + 4 * i] << 24) | ((uint32_t) msg[off + 4 * i + 1] << 16) |
                   ((uint32_t) msg[off + 4 * i + 2] << 8) | (uint32_t) msg[off + 4 * i + 3];
        }
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t t1 = hh + S1 + ch + k[i] + w[i];
            const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = S0 + maj;
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }
    char buf[65];
    for (int i = 0; i < 8; ++i) {
        std::snprintf(buf + 8 * i, 9, "%08x", h[i]);
    }
    return std::string(buf);
}

// The ordered level index at quantile `q` of a score distribution over levels 0..K-1, with linear
// interpolation across the mass-bearing level. Jev's `score` is the mean; this is the skew-robust
// spread summary the branch adds (median, and the p10/p90 band).
static double score_quantile(const std::vector<float> & p, double q) {
    double cum = 0.0;
    for (size_t i = 0; i < p.size(); ++i) {
        const double prev = cum;
        cum += p[i];
        if (cum >= q) {
            if (i > 0 && p[i] > 0.0) {
                return (double) (i - 1) + (q - prev) / (double) p[i];
            }
            return (double) i;
        }
    }
    return (double) (p.size() - 1);
}

// The typed numeric value of a numeric-grid option. The grid stores the value as the option's
// `original` (a JSON integer or float), so the answer and the aggregates are real numbers.
static double option_number(const decision_option & o) {
    if (o.original.is_number_integer()) {
        return (double) o.original.get<long long>();
    }
    return o.original.get<double>();
}

// Value-space weighted quantile of a numeric distribution: the same interpolation the score
// quantile does, but over the actual grid values instead of level indices.
double value_quantile(const std::vector<float> & p, const std::vector<double> & values, double q) {
    double cum = 0.0;
    for (size_t i = 0; i < p.size(); ++i) {
        const double prev = cum;
        cum += p[i];
        if (cum >= q) {
            if (i > 0 && p[i] > 0.0) {
                const double t = (q - prev) / (double) p[i];
                return values[i - 1] + t * (values[i] - values[i - 1]);
            }
            return values[i];
        }
    }
    return values.back();
}

common_json assemble_decision_response(const decision_request & req,
                                  const std::vector<std::vector<float>> & probs,
                                  const std::string & model,
                                  const common_json & usage,
                                  const common_json * diagnostics) {
    const auto uniform = uniform_probs(req);

    common_json answers = common_json::object();
    for (size_t qi = 0; qi < req.questions.size(); ++qi) {
        const decision_question & q = req.questions[qi];
        std::vector<float> p = (qi < probs.size() && !probs[qi].empty()) ? probs[qi] : uniform[qi];
        if (p.size() != q.options.size()) {
            p = uniform[qi];
        }

        common_json a = common_json::object();
        a["type"] = q.type;

        if (q.type == "noul") {
            float p_true = 0.0f;
            for (size_t i = 0; i < q.options.size(); ++i) {
                if (q.options[i].key == "true") {
                    p_true = p[i];
                }
            }
            a["noul"] = (double) p_true;
        } else {
            size_t best = 0;
            common_json probs_obj = common_json::object();
            for (size_t i = 0; i < q.options.size(); ++i) {
                probs_obj[q.options[i].key] = (double) p[i];
                if (p[i] > p[best]) {
                    best = i;
                }
            }
const common_json conc = concentration_metrics(p, req.confidence_profile);
            a["confidence"] = conc.at("confidence");
            if (req.diagnostics) {
                a["certainty"] = conc.at("certainty");             // max(p); additive
            }

            if (q.type == "choice") {
                a["choice"]        = q.options[best].key;
                a["probabilities"] = probs_obj;
            } else if (q.type == "score") {
                double expected = 0.0;
                common_json legend = common_json::object();
                for (size_t i = 0; i < q.options.size(); ++i) {
                    expected += (double) i * (double) p[i];
                    legend[q.options[i].key] = q.options[i].original;
                }
                a["score"]           = expected;
                // additive spread summaries over Jev: the skew-robust median and the 10th-90th
                // percentile band. They are diagnostics-only, so the default envelope stays the
                // strict Jev ScoreAnswer shape.
                if (req.diagnostics) {
                    common_json band = common_json::array();
                    band.push_back(score_quantile(p, 0.10));
                    band.push_back(score_quantile(p, 0.90));
                    a["median"]          = score_quantile(p, 0.5);
                    a["interval_p10_p90"] = band;
                }
                a["probabilities"] = probs_obj;
                a["legend"]        = legend;
            } else { // integer | number: the numeric extension over Jev
                std::vector<double> values;
                values.reserve(q.options.size());
                for (const auto & o : q.options) {
                    values.push_back(option_number(o));
                }
                // `value` is always the winner (mode), a typed grid value; the aggregate is the
                // optional scalar summary. Both are pure functions of the same distribution.
                a["value"]         = q.options[best].original;
                a["probabilities"] = probs_obj;
                if (!q.aggregate.empty()) {
                    double agg = values[best]; // mode
                    if (q.aggregate == "median") {
                        agg = value_quantile(p, values, 0.5);
                    } else if (q.aggregate == "mean") {
                        agg = 0.0;
                        for (size_t i = 0; i < p.size(); ++i) {
                            agg += (double) p[i] * values[i];
                        }
                    }
                    a["aggregate"] = agg;
                }
                if (req.diagnostics) {
                    common_json band = common_json::array();
                    band.push_back(value_quantile(p, values, 0.10));
                    band.push_back(value_quantile(p, values, 0.90));
                    a["median"]          = value_quantile(p, values, 0.5);
                    a["interval_p10_p90"] = band;
                }
            }
        }
        answers[q.id] = a;
    }

    common_json out = common_json::object();
    out["model"]   = model;
    out["answers"] = answers;
    // The strict Jev envelope carries only input/output tokens. The extra counters are additive
    // diagnostics, so drop them unless the caller opted in.
    if (req.diagnostics) {
        out["usage"] = usage;
    } else {
        common_json jev_usage = common_json::object();
        if (usage.contains("input_tokens")) {
            jev_usage["input_tokens"] = usage.at("input_tokens");
        }
        if (usage.contains("output_tokens")) {
            jev_usage["output_tokens"] = usage.at("output_tokens");
        }
        out["usage"] = jev_usage;
    }
    // The diagnostics object and the session fork fields are additive too. The caller hands over a
    // ready payload and decides whether it is emitted: the server passes it for a diagnostics
    // request and for a session fork (which reports its fork fields additively, even without
    // `diagnostics: true`). A null payload means the strict Jev default envelope.
    if (diagnostics != nullptr) {
        for (auto it = diagnostics->begin(); it != diagnostics->end(); ++it) {
            out[it.key()] = it.value();
        }
    }
    return out;
}

} // namespace llama_decision
