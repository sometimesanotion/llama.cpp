#pragma once

// Parallel constrained decisions for finite JSON schemas, shared by llama-parallel-decision
// and llama-server's /decision endpoint.
//
// Every field of a schema has a finite set of allowed values. After a shared context, each
// field's value is scored as token paths following that field's own suffix; every scored path
// runs as its own sequence forked from the context, so all fields are evaluated in one batched
// llama_decode and cannot see each other. Attention cells are shared by metadata; recurrent state
// is copied from a saved partial state. Small fields score every divergence node of their token
// trie at once and return the exact constrained distribution; larger fields walk the trie greedily.

#include "llama.h"
#include "json.h"

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct common_chat_templates;

namespace llama_decision {

using tokens_t = std::vector<llama_token>;

// The request needs more rows or sequences than the batch or pool can hold. The server maps
// this to 422: the request is well formed but too large for the configured budget.
struct capacity_error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// The caller asked to stop before the evaluation finished. The server maps this to 499.
struct cancelled_error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// One field as the scorer sees it: the text before its value and the allowed value texts.
struct field_input {
    std::string              suffix;              // e.g.  '  "fire": '
    std::vector<std::string> candidates;          // allowed values, with the suffix's shared prefix removed
    float                    temperature = 1.0f;  // softmax temperature for this field's score
};

struct options {
    std::string mode           = "auto"; // auto: tree up to tree_max values, else greedy; tree; greedy
    size_t      tree_max       = 128;
    bool        split_boundary = false;  // legacy: tokenise suffix and values separately
    bool        allow_cache    = true;   // reuse the cached static prefix when it matches
    std::string cache_tag;               // optional: cache is only reused when the tag also matches
    std::string fork           = "auto"; // auto | copy | restore | hybrid: how branches fork the prefix
    bool        bypass         = true;   // skip the fork when a round has exactly one branch
    bool        optimize       = true;   // dedup identical fields and hoist a long common suffix head
    std::function<bool()> should_stop;   // optional: checked before every decode and between waves
    std::function<void()> yield;         // optional: cooperative yield point between waves
};

// Numerically stable softmax with an optional temperature; T=1 matches the trie's
// exact normalization so the legacy path stays bit-identical.
std::vector<float> softmax(const std::vector<float> & logits, float temperature = 1.0f);

// The compiled scoring plan for one request: every field's token trie, deduplicated, with a shared
// suffix head hoisted onto the trunk. Opaque: the trie layout is private. It is a pure function of
// (inputs, options) and the vocabulary, so scoring reuses the same plan instead of re-deriving the
// fields.
struct compiled_fields {
    compiled_fields();
    ~compiled_fields();
    compiled_fields(compiled_fields &&) noexcept;
    compiled_fields & operator=(compiled_fields &&) noexcept;
    compiled_fields(const compiled_fields &) = delete;
    compiled_fields & operator=(const compiled_fields &) = delete;

    size_t field_count          = 0; // unique fields after dedup
    size_t suffix_tokens        = 0; // per-field suffix tokens before the shared head is hoisted
    size_t common_suffix_tokens = 0; // suffix head hoisted onto every trunk
    size_t leaf_suffix_tokens   = 0; // what each branch decodes after the hoist
    int    rows                 = 0; // batch rows the plan needs
    int    branches             = 0; // round-1 branches of one context

    struct impl;
    std::unique_ptr<impl> p;
};

// Deterministic identity of a rendered static prefix: prompt version + chat template
// shape + the prefix text. Used to reject a cache hit produced under a different prompt.
std::string make_prefix_tag(const std::string & system_text, const std::string & after,
                            const std::string & prompt_version);

struct field_result {
    int                winner       = -1;
    float              path_score   = 1.0f;
    int                scored_nodes = 0;
    bool               tree         = false;
    std::vector<float> probs;            // tree fields: probability of every allowed value
};

struct result {
    std::vector<field_result> fields;
    bool   cache_hit      = false;
    size_t shared_tokens  = 0;
    size_t context_tokens = 0;
    int    rows           = 0;
    int    rounds         = 0;
    double prefill_ms     = 0;
    double scoring_ms     = 0;
    // suffix token accounting: unique field suffixes after dedup, the head hoisted onto the trunk,
    // and what each branch actually decodes
    size_t suffix_tokens        = 0;
    size_t common_suffix_tokens = 0;
    size_t leaf_suffix_tokens   = 0;
};

// Several contexts decided against one schema and one cached prefix. Items carry fields,
// context_tokens and rows; timings and cache state cover the whole batch.
struct batch_result {
    std::vector<result> items;
    bool   cache_hit     = false;
    size_t shared_tokens = 0;
    int    rows          = 0;
    int    rounds        = 0;
    double prefill_ms    = 0;
    double scoring_ms    = 0;
    size_t suffix_tokens        = 0;
    size_t common_suffix_tokens = 0;
    size_t leaf_suffix_tokens   = 0;
};

// Scores decisions on an existing context with the sequence ids [seq_base, seq_base + n_seqs):
// one keeps the cached static prefix; the rest hold one trunk (prefix + context) per context in
// flight, then branches. The context needs a unified KV cache so branches share the trunk's cells.
class engine {
  public:
    engine(llama_context * ctx, llama_seq_id seq_base, int n_seqs);

    // The single constructor for the flags a save writes and a load uses. Pure: it reads only the
    // requested format and scope, never the engine capability, so a capability downgrade can only
    // change how a state is saved, never how saved bytes are read.
    static llama_state_seq_flags state_load_flags(bool on_device, bool partial = false);

    // True while the engine may save a partial (recurrent-only) state to device staging. A failed
    // partial device save retires this one-way capability for the process; the host format remains.
    bool partial_state_capable() const { return partial_device_capable_; }

    // True when `requested` selects a fork that reproduces the parent's state exactly on this model.
    // A copy fork shares attention cells by metadata, which drops the recurrent state of a
    // recurrent/hybrid model, so it is exact only for dense attention. Restore and hybrid are exact
    // everywhere, and auto never picks copy on a recurrent model. This is a capability predicate: it
    // never inspects a producer concentration score.
    bool session_fork_supported(const std::string & requested) const {
        if (requested == "copy") {
            return probe_fork_ == fork_kind::copy;
        }
        return true;
    }

    const llama_model * get_model() const { return model; }

    // Compiles field inputs into the scoring plan. Pure: it tokenizes (through the per-engine
    // cache) and lays out the trie, but never touches the context or the KV cache.
    compiled_fields compile_fields(const std::vector<field_input> & inputs, const options & opt) const;

    result decide(const std::string & shared_text, const std::string & context_text,
                  const std::vector<field_input> & fields, const options & opt);

    // Contexts are prefilled together and their branches scored together, in groups sized to fit
    // the sequence budget; results keep the order of the contexts.
    batch_result decide_batch(const std::string & shared_text, const std::vector<std::string> & contexts,
                              const std::vector<field_input> & fields, const options & opt);

    // Same as above, but scores a plan the caller already compiled. The fields are compiled once
    // instead of twice. A null plan is a caller error.
    batch_result decide_batch(const compiled_fields &          plan,
                              const std::string &              shared_text,
                              const std::vector<std::string> & contexts,
                              const options &                  opt);

    // Scores a plan against a sequence that is already decoded (`src`), so a caller answers about a
    // live session without re-prefilling its transcript. One trunk is forked from `src` and decodes
    // only the plan's shared suffix head, prefixed by `tail_before_common`; the branches then start
    // at `base_pos` plus that head. `src` is never removed or decoded, so its cells survive the
    // decision. `base_pos` must be the source's next position; any other value is rejected so a
    // wrong caller cannot score at shifted positions.
    batch_result decide_batch_from_seq(llama_seq_id src, llama_pos base_pos, const compiled_fields & plan,
                                       const options & opt, const std::string & tail_before_common = "");

    // Remove every cell of `count` sequences starting at `first` in this engine's memory.
    void clear_seqs(llama_seq_id first, int count);

    // Clear the pool sequences branches are forked into. The cached prefix (seq_snap) is never
    // touched, so prefix cache reuse survives a cleanup.
    void clear_pool_seqs();

  private:
    struct prompt_part {
        const tokens_t * toks;
        llama_pos        pos0;
        llama_seq_id     seq;
    };
    struct branch {
        llama_seq_id trunk;
        llama_pos    pos0;
        tokens_t     toks;
        tokens_t     cands;
    };
    // A trunk already prefilled with its head, ready for the shared wave loop. `trunk` is the pool
    // sequence seq_pool + i, so the saved parent states stay indexed by their offset from seq_pool.
    struct trunk_run {
        llama_seq_id trunk;
        llama_pos    branch_pos;     // position where the branch suffixes start
        size_t       out_index;      // which out.items entry this trunk fills
        size_t       context_tokens; // per-request context size to report
    };
    struct branch_score {
        std::vector<float> cand_logits;
    };

    // copy shares the attention cells by metadata; restore reloads a full saved state; hybrid copies
    // attention cells by metadata and additionally restores the recurrent part from a partial state.
    enum class fork_kind { copy, restore, hybrid };

    // The device and host formats and the state scope (full vs recurrent-only) are distinct, and the
    // flags travel with the bytes, so a load never guesses. The device format stages tensor bytes in
    // the context staging buffer and is not self-contained: it is valid only until something saves
    // over that buffer. The host format is self-contained and outlives the request. Empty state is
    // invalid.
    struct saved_state {
        std::vector<uint8_t>   bytes;
        llama_state_seq_flags  flags = LLAMA_STATE_SEQ_FLAGS_NONE;

        bool on_device() const { return (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) != 0; }
    };

    llama_context     * ctx;
    const llama_model * model;
    const llama_vocab * vocab;
    llama_memory_t      mem;
    llama_seq_id        seq_snap, seq_pool;
    int                 n_pool;
    tokens_t            cached;
    std::string         cached_tag;

    mutable std::unordered_map<std::string, tokens_t> token_cache_;
    static constexpr size_t                           token_cache_limit_ = 1024;

    std::function<bool()> stop_;
    std::function<void()> yield_;
    fork_kind           probe_fork_;
    fork_kind           active_fork_ = fork_kind::copy;
    llama_pos           swa_         = 0; // sliding-window size, 0 = none

    // prefix_state_ is the only device-format state and is valid only for the current request;
    // LRU entries are host format so they outlive the saves that reuse the device staging buffer
    saved_state         prefix_state_;

    // One-way capabilities: device staging is used while a device save works. A device save failure
    // retires the capability for its scope for the process and falls back to the self-contained host
    // format. A device load failure is fatal, because the host bytes are not present to fall back
    // to. The flags select the save format; they are never a content check.
    mutable bool device_capable_         = true;
    mutable bool partial_device_capable_ = true;

    // Device staging is keyed by source sequence, so a device save at one scope invalidates that
    // sequence's earlier device state at any scope. Remember the last device scope per source
    // sequence and use the host format when a device save would mix scopes on one sequence.
    mutable std::unordered_map<llama_seq_id, bool> device_scope_;

    struct prefix_entry {
        std::string tag;
        saved_state state;
    };
    std::vector<prefix_entry> prefix_lru_; // restore mode, most-recent first; entries are host format
    size_t                    prefix_lru_capacity_ = 4;

    tokens_t tokenize(const std::string & text, bool add_special) const;
    void     check_cancel() const;
    void     decode_parts(const std::vector<prompt_part> & parts);
    bool     prepare_prefix(const tokens_t & shared, bool allow_cache, const std::string & tag);

    saved_state save_seq(llama_seq_id seq, bool prefer_device, bool partial = false) const;
    void        load_seq(const saved_state & state, llama_seq_id seq) const;
    void        fork_into(llama_seq_id src, llama_seq_id dst, const saved_state * src_state);
    void        select_fork(const std::string & requested);

    const saved_state * parent_state_for(const std::vector<saved_state> * parent_states, llama_seq_id trunk) const;

    std::vector<branch_score> score_branches(const std::vector<branch> & branches, llama_seq_id first, int n_free,
                                             const std::vector<saved_state> * parent_states,
                                             bool allow_bypass);

    // One wave of already-prefilled trunks: save each trunk's parent state, run the trie and greedy
    // rounds, and write each trunk's result into out.items. Both decide_batch entry points share it,
    // so branch scoring has exactly one implementation.
    void run_trunk_wave(batch_result & out, const compiled_fields & plan, const std::vector<trunk_run> & runs,
                        bool allow_bypass);

    void gather_candidates(int out_idx, const tokens_t & cands, branch_score & out);
};

} // namespace llama_decision
