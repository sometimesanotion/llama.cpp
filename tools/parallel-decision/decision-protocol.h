#pragma once

// Decision request/response bridge: the wire shape on one side, the internal
// per-question option lists on the other. Pure JSON logic, no llama calls, so it
// can be unit tested without a model.

#include "json.h"

#include <cstddef>
#include <cstdint>
#include <map>
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

// A retained-turn reference outlived the memory epoch it was captured under (a model reload or a
// whole-context load/clear). The server maps this to HTTP 409 (stale session); it is NEVER a
// semantic_error (422): "stale" and "invalid request" are different outcomes, and a stale
// reference is never answered from old state.
struct stale_error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

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

// A referenced object (for example an unknown or already-deleted session id) does not exist. The
// server maps this to HTTP 404, the not-found family, distinct from a 422 semantic error.
struct not_found_error : std::runtime_error {
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
// server-side session handle and is mutually exclusive with `id_slot`.
struct session_ref {
    bool        present     = false;
    int         id_slot     = -1;
    std::string session_id; // first-class session handle; mutually exclusive with id_slot
    int         session_pos = -1;
    std::string turn;       // opaque turn tag, matched against the retained snapshot
};

struct decision_request {
    std::string               model;
    common_json               state;       // single evidence document (Jev); unused when `contexts` is set
    std::vector<common_json>  contexts;    // multi-context extension: the same questions against each
    std::vector<decision_question> questions;
    double                    temperature = 1.0;
    common_json               temperatures; // object or null
    int                       permutations = 1;
    std::string               confidence_profile = "jev"; // "jev" (certainty-based, Jev default) | "local" (1 - H/logK)
    bool                      diagnostics = false; // emit additive and diagnostics fields
    session_ref               session;
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
// may carry neither.
decision_evidence parse_evidence(const common_json & body);

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

// Throws semantic_error on any invalid decision content.
decision_request parse_decision_request(const common_json & body);

// Uniform distributions, one per question, sized to its option count.
std::vector<std::vector<float>> uniform_probs(const decision_request & req);

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

// Canonical decision response. The per-answer additive fields (certainty, the extra usage
// counters) are emitted only when `req.diagnostics` is set. The optional `diagnostics` payload is
// additive and merged whenever the caller provides it: the server passes it for a diagnostics
// request and for a session fork (which reports its fork fields additively, even without
// `diagnostics: true`); a null payload keeps the strict Jev default envelope. `probs` is
// index-aligned with req.questions and their options; a missing or empty entry falls back to a
// uniform distribution. This assembler is the single owner of the default-vs-diagnostics envelope.
common_json assemble_decision_response(const decision_request & req,
                                  const std::vector<std::vector<float>> & probs,
                                  const std::string & model,
                                  const common_json & usage,
                                  const common_json * diagnostics = nullptr);

} // namespace llama_decision
