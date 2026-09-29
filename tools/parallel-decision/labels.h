#pragma once

// Answer-label handling for scored decision readouts.
//
// A label is one answer ("A".."Z", "0".."9", "AA".."ZZ", "A0".."Z9", ...) that
// maps to a 1-2 token path at the assistant-answer boundary in the model
// vocabulary. The tokenizer is abstracted so the pool and boundary logic can
// be unit tested without a model.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct llama_vocab;

namespace llama_decision {

struct label_vocab {
    virtual ~label_vocab() = default;
    virtual std::vector<int32_t> tokenize(const std::string & text, bool parse_special) const = 0;
    virtual std::string piece(int32_t token) const = 0;
    virtual bool is_special(int32_t token) const = 0;
};

struct label {
    std::string          text;
    std::vector<int32_t> tokens; // 1-2 tokens the answer tail produces for `text`
    int32_t              token  = -1; // first token when the path is length 1, else -1 (multi-token)
};

// Adapter over a llama.cpp vocabulary. Owns nothing; the vocab must outlive it.
std::unique_ptr<label_vocab> make_llama_label_vocab(const llama_vocab * vocab);

// The token the model emits for `text` at the answer boundary `tail`, else -1. The label is the
// single extra token of tokenize(tail + text) when the tokens before it equal tokenize(tail) and
// that token is non-special. This is the only correct authority for a label token: a
// SentencePiece vocabulary may resolve the isolated form to a different token than the one the
// model emits after the framed tail.
int32_t answer_label_token(const label_vocab & vocab, const std::string & tail, const std::string & text);

// The token path (1..max_len tokens) the model emits for `text` at the answer boundary `tail`,
// else empty. Same authority as answer_label_token, generalized to multi-token answers: the
// tokens before the extra count must equal tokenize(tail) and none of the extra tokens may be
// special. The composed label pool uses this so a two-character label works whether the
// tokenizer emits it as one token or as two.
std::vector<int32_t> answer_label_path(const label_vocab & vocab, const std::string & tail,
                                       const std::string & text, int max_len);

// Single characters first (A-Z, a-z, 0-9, printable ASCII symbols, then accented Latin, Greek
// and Cyrillic), then the two-character AA-ZZ, A0-Z9, 0A-9Z, 00-99, each kept only when the
// boundary resolution yields a unique, non-special 1-2 token path; at most `cap` labels. A model
// whose tokenizer resolves the single characters as single tokens realizes the full 255 labels.
// Throws std::runtime_error if fewer than 2.
inline constexpr size_t LABEL_POOL_CAP = 255;

std::vector<label> build_label_pool(const label_vocab & vocab, const std::string & tail, size_t cap = LABEL_POOL_CAP);

// Escapes "<" so state or option text cannot inject chat-template special
// tokens (for example "<|turn>" or "<__media__>").
std::string safe_data(const std::string & text);

} // namespace llama_decision
