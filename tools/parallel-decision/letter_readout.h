#pragma once

// The readout for the Jev `questions` shape: it scores each question as one next-token choice over
// the verified label pool, sharing one framed state prefix across all questions. It is built on the
// same engine and the same branch scorer as the trie readout the generic front-end uses.

#include "decision-engine.h"
#include "decision-protocol.h"
#include "labels.h"

#include <string>
#include <utility>
#include <vector>

struct common_params;

namespace llama_decision {

// Bump when the letter prompt layout changes; it is part of the prefix cache identity.
inline constexpr const char * LETTER_PROMPT_VERSION = "letter-v2";

// Identity of the decision readout contract: the tokenizer identity, the framed prompt template,
// and the label code. A template edit or a version bump changes it, so a mismatched expected hash
// means the calibration is stale and the decision path must refuse it.
std::string decision_contract_hash(const std::string & model_name, const std::string & template_hash, int vocab_size);

// Quantization label of the loaded model: the general.quantization_version / file_type / type
// metadata, falling back to the model file name. Shared by the provenance helper and the bench.
std::string decision_quantization(const llama_model * model, const std::string & fallback_path);

// Provenance of the decision readout running now: model identity, quantization, prompt template
// hash, and backend flags. Derived from the live params and model so the temperature validation
// and the response diagnostics cannot drift apart.
temperature_provenance decision_provenance_current(const std::string & model_name,
                                                   const common_params & params,
                                                   const llama_model * model,
                                                   const common_chat_templates * tmpls, bool use_jinja);

// The fixed system instruction used by the letter readout.
const char * letter_system_text();

// The assistant-answer tail a label follows: `after` plus the fixed "Answer:\n" marker. One
// definition, so the framer, the per-request gate and the server cannot drift apart.
std::string letter_answer_tail(const std::string & after);

struct readout_metrics {
    bool   cache_hit      = false;
    bool   warm_hit       = false; // a resident warm prefix was forked instead of a cold replay
    size_t shared_tokens  = 0;
    size_t context_tokens = 0;         // sum over contexts
    std::vector<size_t> per_context_tokens; // one entry per context
    int    rows           = 0;
    int    rounds         = 0;
    double prefill_ms     = 0;
    double scoring_ms     = 0;
    size_t      label_pool_size      = 0;  // realized answer-label pool for this model, <= LABEL_POOL_CAP
    size_t suffix_tokens        = 0; // unique question suffixes after dedup
    size_t common_suffix_tokens = 0; // suffix head hoisted onto the shared trunk
    size_t leaf_suffix_tokens   = 0; // what each branch actually decodes
};

// Splits the rendered chat prompt at the user message: `first` is the cacheable
// system/user prefix, `second` is the text after the user content (user turn end
// plus assistant turn start, without any answer instruction). The decision readout
// is thinking-off; `enable_thinking` stays false unless a caller explicitly opts in.
std::pair<std::string, std::string> render_letter_prompt(const common_chat_templates * tmpls, bool use_jinja,
                                                         const std::string & system_text,
                                                         bool enable_thinking = false);

// User-turn markers for a standalone user message: the text before and after the user content,
// including the generation prompt. Appending a decision turn to a live session must not re-render
// the transcript or add a second system message, so the turn is built from this split alone.
std::pair<std::string, std::string> split_user_turn(const common_chat_templates * tmpls, bool use_jinja,
                                                    bool enable_thinking = false);

// The one option-value formatter (`key`, plus ` - description` only when the rendered description
// is non-empty), so an empty description can never leave a trailing separator that would change the
// prompt layout.
std::string format_option_value(const decision_option & opt);

// The one option-line formatter (`label: ` plus the formatted value). The framer builds every scored
// line through this, so the prompt layout has a single source.
std::string format_option_line(const label & l, const decision_option & opt);

// Startup vocabulary probe: every pooled label must be the single non-special token the answer
// tail produces, so the scored slot sits on a clean prompt boundary. Throws std::runtime_error
// with a clear reason when it does not, so the caller can refuse the letter path instead of
// scoring a merged token.
void verify_label_pool(const label_vocab & vocab, const std::vector<label> & labels, const std::string & tail);

// Per-request tokenizer gate: the same boundary check applied to the labels the request actually
// uses. A mismatch throws semantic_error naming the question (HTTP 422), never a silent score.
// This is also the hard capacity check: a question whose widest option count exceeds the realized
// label pool is rejected, never silently truncated or worked around.
void verify_letter_request(const label_vocab & vocab, const std::string & tail,
                           const decision_request & req, const std::vector<label> & labels);

// A session to answer about instead of a stateless prompt: the caller owns the completed turn's
// token list (the sidecar executor's token snapshot), so the readout re-prefills it and continues
// from its end; the source instance is never touched.
struct session_source {
    const tokens_t * tokens = nullptr; // owned token snapshot to replay
    std::string warm_tag;              // resident warm identity (session content hash); empty = cold replay
};

// The context a letter request runs on: the shared full-logits engine. When `session` is set the
// readout replays the owned token snapshot and ignores the state text.
struct readout_sources {
    engine *    full       = nullptr;  // shared context (full-vocabulary logits)
    const session_source * session = nullptr; // token-snapshot session source, null for a stateless readout
};

// The multi-context form: the same questions scored against every context of a `contexts`
// request (or the single `state`, when that is set) in one batched pass. Returns one
// probability matrix per context (question x option), in request order. `metrics` is batch
// level; `per_context_tokens` reports each context's token count. A session fork scores exactly
// one context.
std::vector<std::vector<std::vector<float>>> letter_readout_multi(const readout_sources & sources,
                                                                  const label_vocab & vocab,
                                                                  const common_chat_templates * tmpls, bool use_jinja,
                                                                  const decision_request & req,
                                                                  const std::vector<label> & labels,
                                                                  const options & opt,
readout_metrics * metrics = nullptr);

} // namespace llama_decision
