#pragma once

// Generic typed-schema front-end: the second producer of field_input[] over the shared engine.
// The Jev front-end (letter readout) and this front-end are the two request shapes: `schema`
// selects the generic front-end, `questions` selects Jev, and a body carrying both is refused.
// This front-end compiles a schema into engine-facing scoring fields and assembles the typed
// record from the scores; it never re-implements field compilation or branch scoring.

#include "decision-engine.h"
#include "decision-protocol.h"

#include <string>
#include <utility>
#include <vector>

struct common_chat_templates;

namespace llama_decision {

// Bump when the generic schema prompt layout changes; it is part of the prefix cache identity.
inline constexpr const char * GENERIC_PROMPT_VERSION = "schema-v1";

// Which front-end a decision request body selects.
enum class request_shape { none, jev, generic };

// The request shape a body selects. Mutual exclusion is a request error: a body carrying both
// `schema` and `questions` cannot be dispatched and throws std::invalid_argument (the server maps
// it to 400). `state`/`contexts`/`id_slot` are evidence, orthogonal to the front-end.
request_shape select_request_shape(const common_json & body);

// One field of a generic schema: its name, type, description, aggregate and the typed values.
struct generic_field_spec {
    std::string              name;
    std::string              type;      // boolean | enum | integer | number
    std::string              description;
    std::string              aggregate; // mode | median | mean (median/mean: numeric only)
    std::string              common;    // leading text shared by every encoded value, fixed in the suffix
    std::vector<common_json> values;    // typed values; index = candidate index
    std::vector<std::string> encoded;   // fixed-width JSON text of each value
};

// The compiled generic schema: the cacheable system text, the field catalogue and the
// engine-facing scoring fields (a second producer of field_input[]).
struct compiled_schema {
    std::string                     system_text; // fixed instructions + field catalogue (cacheable)
    std::string                     catalogue;   // per-field catalogue, reused by the session framing
    std::vector<generic_field_spec> specs;
    std::vector<field_input>        inputs;      // suffix + candidate texts the engine scores
};

// Accepts compact field specs {"name": {"type": ..., ...}} or a JSON Schema object with
// "properties" (boolean, string+enum, integer min/max, number min/max/step or multipleOf).
// Pure: tokenizes nothing and touches no context. Throws semantic_error on a malformed schema.
compiled_schema compile_schema(const common_json & schema, const std::string & instructions);

// The typed record for one scored field: value, probability, scored_nodes, tree. A numeric field
// also reports the value-space interval_p10_p90 and the applied aggregate (additive: the value
// stays the winner). Pure.
common_json generic_field_record(const generic_field_spec & spec, const field_result & fr);

// {"decision": {...}, "fields": {...}} from a scored result. Pure.
common_json assemble(const compiled_schema & cs, const result & r);

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

// A generic request as parsed from the body.
struct generic_request {
    std::string         model;
    common_json         schema;
    std::string         instructions;
    decision_evidence   evidence;
    session_ref         session;
    std::string         mode = "auto";  // auto | tree | greedy
    size_t              tree_max = 128;
    bool                allow_cache = true;
    std::string         fork = "auto"; // auto | copy | restore | hybrid
};

// Parses and validates a generic request body. Throws semantic_error on invalid content.
generic_request parse_generic_request(const common_json & body);

// Session-framed field inputs: the schema catalogue is rendered as a fresh user turn (`before` /
// `after` from split_user_turn) so a live-session fork answers the schema without re-prefilling
// the transcript. The full framing lives in every suffix, so the plan's shared head is decoded
// once on the trunk and the branches diverge at the field name.
std::vector<field_input> session_field_inputs(const compiled_schema & cs,
                                              const std::string & before, const std::string & after);

} // namespace llama_decision