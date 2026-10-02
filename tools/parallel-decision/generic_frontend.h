#pragma once

// Generic typed-schema front-end: the `schema` request shape, compiled into the same engine-facing
// fields the Jev letter readout scores. Both shapes share one request envelope and one engine
// (see decision-protocol.h), so this front-end differs only in how it compiles fields and assembles
// the typed record; it never re-implements field compilation, temperature resolution or branch
// scoring.

#include "decision-engine.h"
#include "decision-protocol.h"

#include <string>
#include <utility>
#include <vector>

struct common_chat_templates;

namespace llama_decision {

// Bump when the generic schema prompt layout changes; it is part of the prefix cache identity.
inline constexpr const char * GENERIC_PROMPT_VERSION = "schema-v1";

// One field of a generic schema: its name, type, description, aggregate and the typed values.
struct generic_field_spec {
    std::string              name;
    std::string              type;      // boolean | enum | integer | number
    std::string              description;
    std::string              aggregate; // mode | median | mean (median/mean: numeric only)
    std::vector<common_json> values;    // typed values; index = candidate index
    std::vector<std::string> encoded;   // fixed-width JSON text of each value
};

// The compiled generic schema: the cacheable system text, the field catalogue, the knobs it was
// compiled and will be reported under, and the engine-facing scoring fields (a second producer of
// field_input[]). There is one scoring field per (pass, field) in pass-major order, so an
// order-de-bias average is a fold over the passes of one field rather than a second plan.
struct compiled_schema {
    std::string                     system_text; // fixed instructions + field catalogue (cacheable)
    std::string                     catalogue;   // per-field catalogue, reused by the session framing
    std::vector<generic_field_spec> specs;
    producer_knobs                  knobs;
    int                             passes = 1;  // order-de-bias passes each field was scored in
    std::vector<field_input>        inputs;         // the stateless scoring fields, pass-major
    std::vector<std::vector<size_t>> field_order;   // per scoring field: candidate position -> value index
    std::vector<size_t>              field_spec;    // per scoring field: the owning field index

    // The scoring fields for one framing. `head` is prepended to every field's suffix: the stateless
    // framing passes nothing, the session framing passes the fresh user turn. One builder, so the
    // two framings cannot score different text for the same schema. The order maps above describe
    // the passes and are framing-independent.
    std::vector<field_input> scoring_inputs(const std::string & head) const;
};

// Accepts compact field specs {"name": {"type": ..., ...}} or a JSON Schema object with
// "properties" (boolean, string+enum, integer min/max, number min/max/step or multipleOf).
// Each field resolves its softmax temperature from `knobs` by its own declared type, exactly as a
// Jev question does. Pure: tokenizes nothing and touches no context. Throws semantic_error on a
// malformed schema. The defaults compile the same fields as an unknobbed request.
compiled_schema compile_schema(const common_json & schema, const std::string & instructions,
                               const producer_knobs & knobs = {});

// Folds the permuted passes of one scored context into one distribution per field, averaged by value
// index. The same mean-by-semantic-key the letter readout takes, so the two shapes cannot drift.
// A field whose passes were all scored greedily keeps no distribution, and is reported as the point
// mass the passes agreed on. A single pass has nothing to fold and is left alone.
void mean_permuted_passes(const compiled_schema & cs, result & r);

// One scored field as a Jev-shaped answer: type, value, confidence, a full probabilities map keyed
// by value, the value-space legend, and the `scored` mode. Under `diagnostics` a numeric field also
// reports interval_p10_p90 and the applied aggregate. Pure.
common_json generic_field_record(const generic_field_spec & spec, const field_result & fr,
                                 const std::string & confidence_profile, bool diagnostics);

// One context's fields as a Jev `answers` map keyed by field name. Pure.
common_json assemble(const compiled_schema & cs, const result & r, bool diagnostics);

// Renders the schema prompt with the chat template (thinking off) and splits it into the static
// prefix and the per-request part: the context, the user-turn close, the assistant open and the
// opening brace of the JSON answer. Reuses split_chat_template, so this front-end has no sentinel
// logic of its own (D6).
std::pair<std::string, std::string> render_schema_prompt(const common_chat_templates * tmpls, bool use_jinja,
                                                         const std::string & system_text, const std::string & context);

// The prefix-cache identity of the generic schema prompt: prompt version + chat template shape +
// the system text. Used to reject a cache hit produced under a different schema or template.
std::string generic_cache_tag(const common_chat_templates * tmpls, bool use_jinja,
                              const std::string & system_text);

// Session-framed field inputs: the schema catalogue is rendered as a fresh user turn (`before` /
// `after` from split_user_turn) so a live-session fork answers the schema without re-prefilling
// the transcript. The full framing lives in every suffix, so the plan's shared head is decoded
// once on the trunk and the branches diverge at the field name.
std::vector<field_input> session_field_inputs(const compiled_schema & cs,
                                              const std::string & before, const std::string & after);

} // namespace llama_decision