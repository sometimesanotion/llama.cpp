#pragma once

// Decision request/response bridge: the wire shape on one side, the internal
// per-question option lists on the other. Pure JSON logic, no llama calls, so it
// can be unit tested without a model.

#include "json.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct common_chat_templates;

namespace llama_decision {

// Request shape limits. A request over a limit is rejected before any decode; it is never
// truncated. The choice option cap matches Jev's and the composed answer-label pool cap, so every
// option gets a label (a label is a 1-2 token path over A-Z and 0-9).
inline constexpr size_t DECISION_MIN_QUESTIONS        = 1;
inline constexpr size_t DECISION_MAX_QUESTIONS        = 256;
inline constexpr size_t DECISION_MAX_CONTEXTS         = 256;
inline constexpr size_t DECISION_MIN_OPTIONS          = 2;
inline constexpr size_t DECISION_MAX_CHOICE_OPTIONS   = 255;
inline constexpr size_t DECISION_MAX_SCORE_LEVELS     = 10;
inline constexpr size_t DECISION_MAX_NUMERIC_VALUES   = 255;
inline constexpr int    DECISION_MAX_PERMUTATIONS     = 8;

// Valid JSON but invalid decision content (bad type, limits, missing fields).
// The server maps this to HTTP 422; malformed JSON stays a parse error (400).
struct semantic_error : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};

// Typed readers for one member of a request object, one per JSON kind. A member that is absent or
// null reads as "not supplied" (nullopt); a member that is present with the wrong JSON type throws
// semantic_error naming it, so a wrong-typed decision field is semantic invalidity (422) and never
// a malformed body (400). `where` prefixes the name in the message ("field \"priority\": ").
std::optional<std::string> read_string(const common_json & obj, const std::string & key, const std::string & where = "");
std::optional<bool>        read_bool(const common_json & obj, const std::string & key, const std::string & where = "");
std::optional<long long>   read_integer(const common_json & obj, const std::string & key, const std::string & where = "");
std::optional<double>      read_number(const common_json & obj, const std::string & key, const std::string & where = "");

// FNV-1a 64 over the bytes of `s` (offset basis 1469598103934665603, prime 1099511628211).
// The single hash primitive behind the decision prefix tag and the permutation seed.
uint64_t fnv1a64(const std::string & s);

// A seeded distinct permutation of `count` indices for `id` at `pass`; pass 0 (and any pass on a
// single option) is the identity. Shared by the Jev and generic readouts for order-de-biasing.
std::vector<size_t> permutation_order(size_t count, const std::string & id, int pass);

// The producer concentration scores, pure functions of an option distribution. `confidence`
// (default) is the Jev compatibility value `(N*p_max - 1)/(N - 1)` (see
// jev_winner_share_confidence); `certainty` is the winner's share max(p), the quantity Jev's
// documented Choice confidence is derived from. The opt-in entropy form 1 - H/log K is available
// via confidence_profile "local". Neither number measures whether the winner is correct.
double inverse_entropy_confidence(const std::vector<float> & p);
double winner_share(const std::vector<float> & p);

// The opt-in Jev compatibility confidence: `(N*p_max - 1)/(N - 1)` clamped to [0,1], a rescaled
// winner's share. N is the option count. Jev documents this for Choice and for Score with 2..3
// levels; above three the definition is undocumented, so the same monotone rule is the stated
// local value. It reads only the winner, unlike the entropy confidence.
double jev_winner_share_confidence(const std::vector<float> & p);

// The two producer-concentration numbers for a distribution, keyed for JSON: `confidence` (Jev or
// opt-in entropy per `confidence_profile`) and `certainty` (winner's share). Shared by the Jev and
// generic readouts.
common_json concentration_metrics(const std::vector<float> & p, const std::string & confidence_profile);

// Renders the chat template with a sentinel user message and returns the text before and after
// the sentinel. `render_prompt` and `render_letter_prompt` share it and differ only in how they
// use the split; `tmpls` must be non-null.
std::pair<std::string, std::string> split_chat_template(const common_chat_templates * tmpls,
                                                        bool use_jinja,
                                                        const std::string & system_text,
                                                        bool enable_thinking);

// The running model cannot serve the decision path at all (for example its vocabulary has no
// usable single-token answer labels). The server maps this to HTTP 501, never a client error.
struct unsupported_error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// One allowed answer of a question.
struct decision_option {
    std::string key;         // choice key, level index string, or "true"/"false"
    std::string description; // text shown to the model
    common_json original;    // original criterion value, echoed by `legend`
};

struct decision_question {
    std::string              id;
    std::string              type;         // canonical: noul | choice | score | integer | number
    common_json              instructions; // string/object/array, may be null
    std::vector<decision_option>  options;
    bool                     has_criteria = false;
    std::string              aggregate;    // numeric only: "" | mode | median | mean
};

// A request may name a live chat slot to answer about, so the transcript is not re-prefilled.
// `present` is true only when id_slot or session_id is supplied; `session_pos` is the source's
// next position when pinned by the caller and -1 when the server derives it from the slot. `turn`
// is an opaque client tag that must match the slot's retained snapshot; a mismatch is a 409/422,
// never a silent answer about a different turn. All are capability inputs: the slot must exist,
// hold decoded state, and the position must continue it exactly. `session_id` names a first-class
// server-side session handle and is mutually exclusive with `id_slot`. The slot number itself is
// resolved from the request body by the pool, which is the component that owns the slot table, so
// only the presence of a slot reference travels here.
struct session_ref {
    bool        present     = false;
    std::string session_id; // first-class session handle; mutually exclusive with id_slot
    int         session_pos = -1;
    std::string turn;       // opaque turn tag, matched against the retained snapshot
};

// The evidence source shared by the Jev and generic front-ends: exactly one of a single `state`
// (string/object/array) or a list of `contexts` (1-DECISION_MAX_CONTEXTS non-empty strings). The
// session reference is orthogonal to the evidence text.
struct decision_evidence {
    bool                   state_present = false;
    common_json            state;         // single evidence document; unused when contexts is set
    std::vector<std::string> contexts;    // multi-context extension, answered in order
};

// Parses the evidence fields of a decision body. Throws semantic_error on invalid content; the
// caller checks `state_present` for the "state or contexts required" rule because a session request
// may carry neither. The single evidence validator: a body is evidence-checked here and nowhere
// else, so the two request shapes cannot drift on the "state or contexts, not both" rule.
decision_evidence parse_evidence(const common_json & body);

// Which front-end a decision request body selects. Dispatch is on the top-level shape: `schema`
// selects the generic front-end, `questions` (or a bare `state`) the Jev one. Mutual exclusion is a
// request error: a body carrying both `schema` and `questions` cannot be dispatched and throws
// std::invalid_argument (the server maps it to 400). `state`/`contexts`/`id_slot` are evidence,
// orthogonal to the front-end.
enum class request_shape { none, jev, generic };

request_shape select_request_shape(const common_json & body);

// The producer-side knobs every request shape shares: how the gathered label logits are softened,
// how many order-de-bias passes are averaged, and which confidence profile the answer reports. A
// value rather than a body fragment, so the letter readout and the schema compiler resolve the same
// temperature from the same fields.
struct producer_knobs {
    double      temperature       = 1.0;   // global softmax temperature on the gathered label logits
    common_json temperatures;               // per-type overrides, the wire object
    int         permutations      = 1;      // order-de-bias passes, mean taken by semantic key
    std::string confidence_profile = "jev";  // reported concentration only; it gates nothing

    // The effective softmax temperature for a question or field of `type`: the per-type override
    // when there is one, else the global. A type outside the Jev primitives maps onto its
    // primitive - a generic `boolean` is a `noul`, an `enum` a `choice` - so one rule serves both
    // shapes and the wire's five type keys stay the whole vocabulary.
    double for_type(const std::string & type) const;
};

// Everything a decision request carries regardless of its shape: the echoed model, the evidence,
// the session reference, the producer knobs and the diagnostics opt-in. One envelope, so the two
// shapes cannot disagree about a shared field.
struct decision_envelope {
    std::string       model;
    decision_evidence evidence;
    session_ref       session;
    producer_knobs    knobs;
    bool              diagnostics = false; // emit additive and diagnostics fields
};

// A parsed decision request: the shared envelope, the shape that selected it, and the fields only
// that shape defines. `questions` belongs to the Jev front-end; `schema` and the schema-only knobs
// to the generic one. Both terminate at the same engine.
struct decision_request {
    decision_envelope           envelope;
    request_shape               shape = request_shape::none;
    std::vector<decision_question> questions; // Jev shape
    common_json                 schema;      // generic shape
    std::string                 instructions;
    std::string                 mode     = "auto"; // generic: auto | tree | greedy
    size_t                      tree_max = 128;
    bool                        allow_cache = true;
};

// Renders a state (string/object/array) as prompt evidence: the text is framed as data and every
// "<" is escaped so chat-template special tokens cannot be injected from the evidence.
std::string render_state(const common_json & state);

// One grid value of a numeric question: the round-tripped double and its fixed-width JSON text.
// Every grid value has the same shape, so the encoded set renders without float noise.
struct numeric_grid_value {
    double      value;
    std::string text;
};

// The ascending numeric grid [lo, hi] at `step` (or multipleOf), including both ends. The shared
// encoder behind the Jev `integer`/`number` extensions and the generic `number` field, so the two
// front-ends cannot drift (D5). Throws semantic_error when the step does not divide the range or
// the grid is outside 2-DECISION_MAX_NUMERIC_VALUES values.
std::vector<numeric_grid_value> numeric_grid(double lo, double hi, double step);

// Value-space weighted quantile of a numeric distribution over the grid values: the same
// interpolation the score quantile uses, over actual values instead of level indices. Shared by
// the Jev numeric answers and the generic record's interval/aggregate.
double value_quantile(const std::vector<float> & p, const std::vector<double> & values, double q);

// Renders a string/object/array instruction or criterion value to prompt text.
std::string render_text(const common_json & value);

// Effective softmax temperature for a question: per-type override, else the global value.
double question_temperature(const decision_request & req, const decision_question & q);

// Reads the optional live-session reference shared by the Jev and generic shapes. Throws
// semantic_error when a field has the wrong type, when id_slot is negative, or when session_pos
// is supplied without id_slot.
session_ref parse_session_ref(const common_json & body);

// Parses the envelope both request shapes share. Calls the existing parse_evidence, read_temperatures
// and parse_session_ref, so the shared fields are validated exactly once per body whatever the shape.
// Throws semantic_error on invalid content.
decision_envelope parse_decision_envelope(const common_json & body);

// Parses one request body of either shape: the envelope, then the shape the body selects, then that
// shape's own fields. Every other top-level field is ignored, so a typo cannot change an answer.
// Throws semantic_error on invalid decision content and std::invalid_argument on a body that
// carries both `questions` and `schema`.
decision_request parse_decision_request(const common_json & body);

// A decision decodes under an adapter scope and reports the scope it used, so a scope the context
// would not install cannot be reported as if it had been. `apply_status` is what installing `scope`
// returned (0 = installed); the base model is the empty scope. Throws semantic_error, so the request
// is refused as 422 in the same class as "the slot does not exist": the requested conditioning is not
// available. It never reads a producer score.
void require_adapter_scope(const std::string & scope, int apply_status);

// Identity a calibrated temperature profile was fitted against. It is never used
// to gate answers; it only stops a profile fitted on one deployment from silently
// applying to another.
struct temperature_provenance {
    std::string model;
    std::string quantization;
    std::string template_hash;
    std::string backend_flags;

    bool operator==(const temperature_provenance & other) const;
    bool operator!=(const temperature_provenance & other) const { return !(*this == other); }
};

struct temperature_profile {
    std::map<std::string, double> temperatures; // per type: noul | choice | score
    temperature_provenance        provenance;
};

// Parses {"temperatures": {...}, "provenance": {...}}. Throws semantic_error on bad shape.
temperature_profile parse_temperature_profile(const common_json & doc);

// Throws semantic_error when any non-default temperature would run under a provenance
// that does not match the running configuration. T=1.0 is always allowed.
void validate_temperature_profile(const temperature_profile & profile, const temperature_provenance & current);

// Lowercase hex SHA-256 of the given bytes. Used for the decision contract hash.
std::string sha256_hex(const std::string & text);

// One context's Jev answers: the `answers` map assemble_decision_response emits, without the
// envelope around it. A caller that builds its own envelope - the server's, shared by both
// front-ends - uses this instead of running the Jev envelope and taking it apart. `probs` is
// index-aligned with req.questions and carries exactly one score per option; a missing or
// mis-sized vector is an internal defect and throws rather than being answered with a distribution
// nobody scored. This is the single owner of the Jev answer record, and of the default-vs-
// diagnostics rule for its per-answer additive fields.
common_json assemble_answers(const decision_request & req, const std::vector<std::vector<float>> & probs);

// The strict Jev `usage` object: input_tokens and output_tokens only. The prefix-cache counters are
// additive diagnostics, so they are dropped unless the caller opted in. One definition, reached by
// the Jev assembler and by the response envelope both front-ends answer through, so the two shapes
// cannot disagree on which usage keys the strict envelope carries.
common_json jev_usage(const common_json & usage, bool diagnostics);

// The single-state shape answers exactly one answer map. When the batch produced none - a session
// with no decoded context - the one owner of the empty case answers an empty object, so the shared
// envelope never indexes an empty vector and neither front-end re-derives a fallback. A `contexts`
// request does not use this: it answers the documented array even when it carries one entry.
common_json single_state_answers(const std::vector<common_json> & answers_per_context);

// Canonical decision response. The per-answer additive fields (certainty, the extra usage
// counters) are emitted only when `req.diagnostics` is set. The optional `diagnostics` payload is
// additive and merged whenever the caller provides it: the server passes it for a diagnostics
// request and for a session fork (which reports its fork fields additively, even without
// `diagnostics: true`); a null payload keeps the strict Jev default envelope. `probs` is
// index-aligned with req.questions as for assemble_answers.
common_json assemble_decision_response(const decision_request & req,
                                  const std::vector<std::vector<float>> & probs,
                                  const std::string & model,
                                  const common_json & usage,
                                  const common_json * diagnostics = nullptr);

} // namespace llama_decision
