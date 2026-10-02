#include "generic_frontend.h"

#include "chat.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace llama_decision {

namespace {

std::string json_text(const std::string & s) {
    return common_json(s).dump();
}

// The typed double as a JSON number: an integral value stays integral so a whole-valued quantile
// or aggregate serializes without a ".0".
common_json json_number(double x) {
    const long long r = std::llround(x);
    if (std::fabs(x - (double) r) < 1e-9) {
        return common_json(r);
    }
    return common_json(x);
}

// Builds one field spec from a schema entry. `json_schema` selects the JSON Schema field naming
// and the numeric step key ("multipleOf" vs "step"). Values are 1-255 and unique; the numeric
// grid reuses the shared numeric_grid encoder (D5).
generic_field_spec make_field(const std::string & name, const std::string & type, const std::string & description,
                              const common_json & spec, bool json_schema) {
    generic_field_spec f;
    f.name        = name;
    f.description = description;
    const std::string kind = type;
    if (kind == "boolean") {
        f.type    = "boolean";
        f.values  = { common_json(true), common_json(false) };
        f.encoded = { "true", "false" };
    } else if (kind == "enum" || kind == "choice" || kind == "selection") {
        const char * key = spec.contains("enum") ? "enum" : "choices";
        if (!spec.contains(key) || !spec.at(key).is_array()) {
            throw semantic_error("field \"" + name + "\": enum fields need a list of choices");
        }
        f.type = "enum";
        for (const auto & c : spec.at(key)) {
            if (!c.is_string()) {
                throw semantic_error("field \"" + name + "\": enum choices must be strings");
            }
            const std::string v = c.get<std::string>();
            f.values.push_back(common_json(v));
            f.encoded.push_back(json_text(v));
        }
    } else if (kind == "integer") {
        const std::string               where = "field \"" + name + "\": ";
        const std::optional<long long> lo     = read_integer(spec, "minimum", where);
        const std::optional<long long> hi     = read_integer(spec, "maximum", where);
        if (!lo || !hi) {
            throw semantic_error("field \"" + name + "\": integer fields need integer minimum and maximum");
        }
        if (*hi < *lo || *hi - *lo + 1 > (long long) DECISION_MAX_NUMERIC_VALUES) {
            throw semantic_error("field \"" + name + "\": integer bounds must define 1-" +
                                 std::to_string(DECISION_MAX_NUMERIC_VALUES) + " values");
        }
        f.type = "integer";
        for (const auto & gv : numeric_grid((double) *lo, (double) *hi, 1.0)) {
            f.values.push_back(common_json((long long) gv.value));
            f.encoded.push_back(gv.text);
        }
    } else if (kind == "number") {
        const char *                step_key = json_schema ? "multipleOf" : "step";
        const std::string           where    = "field \"" + name + "\": ";
        const std::optional<double> lo       = read_number(spec, "minimum", where);
        const std::optional<double> hi       = read_number(spec, "maximum", where);
        const std::optional<double> step     = read_number(spec, step_key, where);
        if (!lo || !hi || !step) {
            throw semantic_error("field \"" + name + "\": number fields need minimum, maximum and " + step_key);
        }
        if (!(*step > 0) || !(*hi >= *lo)) {
            throw semantic_error("field \"" + name + "\": number needs ordered bounds and a positive step");
        }
        f.type = "number";
        for (const auto & gv : numeric_grid(*lo, *hi, *step)) {
            f.values.push_back(common_json(gv.value));
            f.encoded.push_back(gv.text);
        }
    } else {
        throw semantic_error("field \"" + name + "\": supported types are boolean, enum, integer and number");
    }
    if (f.values.empty() || f.values.size() > DECISION_MAX_NUMERIC_VALUES) {
        throw semantic_error("field \"" + name + "\" needs 1-" + std::to_string(DECISION_MAX_NUMERIC_VALUES) +
                             " allowed values");
    }
    for (size_t a = 0; a < f.encoded.size(); ++a) {
        for (size_t b = 0; b < a; ++b) {
            if (f.encoded[a] == f.encoded[b]) {
                throw semantic_error("field \"" + name + "\" has duplicate allowed values");
            }
        }
    }
    const std::string where = "field \"" + name + "\": ";
    std::string       agg   = read_string(spec, "x-aggregate", where).value_or("mode");
    agg                      = read_string(spec, "aggregate", where).value_or(agg);
    const bool numeric = f.type == "integer" || f.type == "number";
    if (agg != "mode" && !(numeric && (agg == "median" || agg == "mean"))) {
        throw semantic_error("field \"" + name + "\": aggregate must be mode, or median/mean for numeric fields");
    }
    f.aggregate = agg;

    return f;
}

// The numeric grid values as doubles, index-aligned with the encoded values.
std::vector<double> numeric_values(const generic_field_spec & f) {
    std::vector<double> out;
    out.reserve(f.values.size());
    for (const auto & v : f.values) {
        out.push_back(v.is_number_integer() ? (double) v.get<long long>() : v.get<double>());
    }
    return out;
}

} // namespace

compiled_schema compile_schema(const common_json & schema, const std::string & instructions,
                               const producer_knobs & knobs) {
    if (!schema.is_object()) {
        throw semantic_error("\"schema\" must be an object");
    }
    compiled_schema cs;
    cs.knobs = knobs;
    const bool json_schema = schema.contains("properties");
    const common_json & props = json_schema ? schema.at("properties") : schema;
    if (!props.is_object() || props.size() < 1 || props.size() > 32) {
        throw semantic_error("the schema must define 1-32 fields");
    }
    for (const auto & e : props.items()) {
        const common_json & spec = e.value();
        if (!spec.is_object()) {
            throw semantic_error("field \"" + e.key() + "\" must be an object");
        }
        const std::string where = "field \"" + e.key() + "\": ";
        std::string       type  = read_string(spec, "type", where).value_or(std::string());
        if (spec.contains("enum")) {
            type = "enum";
        }
        std::string description = read_string(spec, "description", where).value_or(std::string());
        if (!json_schema && description.empty()) {
            throw semantic_error("field \"" + e.key() + "\" needs a description");
        }
        cs.specs.push_back(make_field(e.key(), type, description, spec, json_schema));
    }

    std::string catalog;
    for (const auto & f : cs.specs) {
        std::string allowed;
        for (size_t i = 0; i < f.encoded.size(); ++i) {
            allowed += (i ? ", " : "") + f.encoded[i];
        }
        catalog += (catalog.empty() ? "" : "\n") + json_text(f.name) +
                   (f.description.empty() ? "" : ": " + f.description) + "\nAllowed values: " + allowed;
    }
    cs.catalogue   = catalog;
    cs.system_text = "Select the requested field value from its allowed values, based on the context. "
                     "Respond with the JSON value only.\n\nFields:\n" + catalog + "\n" + instructions;

    // One scoring field per (pass, field), pass-major. The candidate texts are the whole encoded
    // value: the engine splits the shared head of `suffix + candidate` at a token boundary, so a
    // value-space prefix needs no second, character-level split here.
    cs.passes = std::max(1, knobs.permutations);
    cs.field_order.reserve(cs.specs.size() * (size_t) cs.passes);
    cs.field_spec.reserve(cs.specs.size() * (size_t) cs.passes);
    cs.inputs.reserve(cs.specs.size() * (size_t) cs.passes);
    for (int pass = 0; pass < cs.passes; ++pass) {
        for (size_t si = 0; si < cs.specs.size(); ++si) {
            const generic_field_spec &   f    = cs.specs[si];
            const std::vector<size_t>    order = permutation_order(f.encoded.size(), f.name, pass);
            field_input in;
            in.suffix      = "  " + json_text(f.name) + ": ";
            in.temperature = (float) knobs.for_type(f.type);
            in.candidates.reserve(order.size());
            for (size_t i : order) {
                in.candidates.push_back(f.encoded[i]);
            }
            cs.inputs.push_back(std::move(in));
            cs.field_order.push_back(order);
            cs.field_spec.push_back(si);
        }
    }
    return cs;
}

std::vector<field_input> compiled_schema::scoring_inputs(const std::string & head) const {
    if (head.empty()) {
        return inputs;
    }
    std::vector<field_input> out;
    out.reserve(inputs.size());
    for (size_t f = 0; f < inputs.size(); ++f) {
        field_input in  = inputs[f];
        in.suffix        = head + in.suffix;
        out.push_back(std::move(in));
    }
    return out;
}

void mean_permuted_passes(const compiled_schema & cs, result & r) {
    const size_t n_specs = cs.specs.size();
    if (cs.passes <= 1 || r.fields.size() != n_specs * (size_t) cs.passes) {
        return;
    }
    std::vector<field_result> folded(n_specs);
    for (size_t si = 0; si < n_specs; ++si) {
        const size_t n_values = cs.specs[si].values.size();
        std::vector<float> acc(n_values, 0.0f);
        bool scored_distribution = false;
        int  nodes              = 0;
        for (int pass = 0; pass < cs.passes; ++pass) {
            const field_result &       fr    = r.fields[(size_t) pass * n_specs + si];
            const std::vector<size_t> & order = cs.field_order[(size_t) pass * n_specs + si];
            nodes += fr.scored_nodes;
            if (fr.probs.size() == n_values && order.size() == n_values) {
                scored_distribution = true;
                for (size_t i = 0; i < n_values; ++i) {
                    acc[order[i]] += fr.probs[i] / (float) cs.passes;
                }
            } else if (fr.winner >= 0 && (size_t) fr.winner < order.size()) {
                acc[order[fr.winner]] += 1.0f / (float) cs.passes;
            }
        }
        folded[si].scored_nodes = nodes;
        folded[si].winner       = acc.empty() ? -1
                                             : (int) (std::max_element(acc.begin(), acc.end()) - acc.begin());
        // passes that were all greedy produced no distribution, so none is reported: the record
        // shows the point mass the passes agreed on, as it does for a single greedy pass
        folded[si].probs = scored_distribution ? std::move(acc) : std::vector<float>();
    }
    r.fields = std::move(folded);
}

// The probability/legend key for an allowed value. A string value keys itself; anything else keys
// by its JSON literal, so a boolean is "true" and a grid point is "2" - never a quoted JSON string.
static std::string generic_value_key(const common_json & v) {
    return v.is_string() ? v.get<std::string>() : v.dump();
}

// One scored field as a Jev-shaped answer. The generic path *extends* the Jev answer rather than
// replacing it, so a client written against the Jev envelope reads the same keys:
//   type          the generic field type (boolean | enum | integer | number)
//   value         the selected typed value - the generic extension over a Jev choice/score key
//   confidence    the Jev winner-share, from the same helper the Jev path uses
//   probabilities the Jev map: every allowed value keyed by its value, summing to 1
//   legend        the value space, as a Jev ScoreAnswer legend
//   scored        "tree" for a real distribution, "argmax" for a greedy-scored field
// The winner's share `certainty` and the spread summaries (interval_p10_p90, aggregate) are
// diagnostics-only, matching the Jev path, so the default answer carries no off-envelope field.
common_json generic_field_record(const generic_field_spec & spec, const field_result & fr,
                                 const std::string & confidence_profile, bool diagnostics) {
    const int idx = fr.winner;
    if (idx < 0 || idx >= (int) spec.values.size()) {
        throw std::runtime_error("field \"" + spec.name + "\" has no selected value");
    }
    const bool numeric = spec.type == "integer" || spec.type == "number";
    const bool have_dist = fr.probs.size() == spec.values.size();

    common_json probs_obj = common_json::object();
    common_json legend    = common_json::object();
    std::vector<float>    p;
    for (size_t i = 0; i < spec.values.size(); ++i) {
        legend[generic_value_key(spec.values[i])] = spec.values[i];
        // Every allowed value gets a key, so the Jev map invariant holds even for a greedily
        // scored field: its non-winning values are the zeros of the point mass.
        probs_obj[generic_value_key(spec.values[i])] = have_dist ? (double) fr.probs[i] : 0.0;
    }
    if (have_dist) {
        p = fr.probs;
    } else {
        // A field scored greedily (more allowed values than the tree bound) has no distribution
        // over the space, only the winning path's share. Report the point mass it actually chose
        // rather than fabricating a spread, and say so in `scored`.
        p.assign(spec.values.size(), 0.0f);
        p[idx] = 1.0f;
        probs_obj[generic_value_key(spec.values[idx])] = 1.0;
    }

    common_json f = common_json::object();
    f["type"]         = spec.type;
    f["value"]        = spec.values[idx];
    const common_json conc = concentration_metrics(p, confidence_profile);
    f["confidence"]   = conc.at("confidence");
    f["probabilities"] = probs_obj;
    f["legend"]       = legend;
    f["scored"]       = have_dist ? "tree" : "argmax";
    f["scored_nodes"] = fr.scored_nodes;
    if (diagnostics) {
        f["certainty"] = conc.at("certainty");  // max(p); additive, as on the Jev path
    }

    if (diagnostics && numeric && have_dist) {
        // the numeric spread summary and aggregate reuse the Jev value-space quantile over the grid
        const std::vector<double> values = numeric_values(spec);
        common_json interval = common_json::array();
        interval.push_back(json_number(value_quantile(fr.probs, values, 0.10)));
        interval.push_back(json_number(value_quantile(fr.probs, values, 0.90)));
        f["interval_p10_p90"] = interval;
        double agg = values[idx]; // mode
        if (spec.aggregate == "median") {
            agg = value_quantile(fr.probs, values, 0.5);
        } else if (spec.aggregate == "mean") {
            agg = 0.0;
            for (size_t i = 0; i < fr.probs.size(); ++i) {
                agg += (double) fr.probs[i] * values[i];
            }
        }
        f["aggregate"] = json_number(agg);
    }
    return f;
}

// One context's scored fields as a Jev `answers` map, keyed by field name. A single context emits
// this at the top level; several contexts are wrapped in the documented `contexts` array. The order
// -de-bias passes, if any, are folded first, so this stays the single owner of the value space.
common_json assemble(const compiled_schema & cs, const result & r, bool diagnostics) {
    common_json answers = common_json::object();
    for (size_t i = 0; i < cs.specs.size(); ++i) {
        if (i >= r.fields.size()) {
            throw std::runtime_error("the scored result is missing field \"" + cs.specs[i].name + "\"");
        }
        answers[cs.specs[i].name] =
            generic_field_record(cs.specs[i], r.fields[i], cs.knobs.confidence_profile, diagnostics);
    }
    return answers;
}

std::pair<std::string, std::string> render_schema_prompt(const common_chat_templates * tmpls, bool use_jinja,
                                                         const std::string & system_text, const std::string & context) {
    if (tmpls == nullptr) {
        return { system_text + "\nContext:\n", context + "\nOutput:\n{\n" };
    }
    const auto split = split_chat_template(tmpls, use_jinja, system_text, false);
    return { split.first, context + split.second + "{\n" };
}

std::string generic_cache_tag(const common_chat_templates * tmpls, bool use_jinja,
                              const std::string & system_text) {
    const auto split = tmpls != nullptr ? split_chat_template(tmpls, use_jinja, system_text, false)
                                        : std::make_pair(system_text + "\n", std::string("\n"));
    return make_prefix_tag(system_text, split.second, GENERIC_PROMPT_VERSION);
}

std::string generic_template_hash(const common_chat_templates * tmpls, bool use_jinja) {
    // rendered with an empty system text, so the tag covers the chat template's own shape and the
    // schema prompt version without the request's field catalogue
    const auto parts = render_schema_prompt(tmpls, use_jinja, std::string(), std::string());
    return make_prefix_tag(std::string(), parts.second, GENERIC_PROMPT_VERSION);
}

std::string generic_contract_hash(const std::string & model_name, const std::string & template_hash, int vocab_size) {
    return sha256_hex("decision-contract-v1|" + template_hash + "|" + std::string(GENERIC_PROMPT_VERSION) + "|" +
                      model_name + "|" + std::to_string(vocab_size));
}

std::vector<field_input> session_field_inputs(const compiled_schema & cs,
                                              const std::string & before, const std::string & after) {
    // the fresh user turn carries the catalogue, then the JSON answer opens after the assistant
    // open (`after`); every suffix shares this framing so the plan hoists it
    return cs.scoring_inputs(before + "\n" + cs.catalogue + "\n" + after + "{\n");
}

} // namespace llama_decision