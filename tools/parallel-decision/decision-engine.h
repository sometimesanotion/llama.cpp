#pragma once

// Parallel constrained decisions for finite JSON schemas, served by llama-server's /v1/decision
// and /v1/systemone endpoints.
//
// Every field of a schema has a finite set of allowed values. After a shared context, each
// field's value is scored as token paths following that field's own suffix; every scored path
// runs as its own sequence forked from the context, so all fields are evaluated in one batched
// llama_decode and cannot see each other. Attention cells are shared by metadata; recurrent state
// is copied from a saved partial state. Small fields score every divergence node of their token
// trie at once and return the exact constrained distribution; larger fields walk the trie greedily.

#include "json.h"
#include "llama.h"

#include <deque>
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

    size_t suffix_tokens        = 0;  // per-field suffix tokens before the shared head is hoisted
    size_t common_suffix_tokens = 0;  // suffix head hoisted onto every trunk
    size_t leaf_suffix_tokens   = 0;  // what each branch decodes after the hoist
    int    rows                 = 0;  // batch rows the plan needs
    int    branches             = 0;  // round-1 branches of one context

    // The token cache's behavior while this plan was compiled. The cache is a pure function of
    // (text, add_special) over the vocabulary, so these count cost, never correctness; they exist
    // so an eviction policy can be read in production instead of guessed at.
    size_t token_cache_hits   = 0;
    size_t token_cache_misses = 0;

    struct impl;
    std::unique_ptr<impl> p;
};

// Deterministic identity of a rendered static prefix: prompt version + chat template
// shape + the prefix text. Used to reject a cache hit produced under a different prompt.
std::string make_prefix_tag(const std::string & system_text, const std::string & after,
                            const std::string & prompt_version);

struct field_result {
    int                winner       = -1;
    int                scored_nodes = 0;
    std::vector<float> probs;            // tree fields: probability of every allowed value
};

struct result {
    std::vector<field_result> fields;
    size_t context_tokens = 0;
};

// Several contexts decided against one schema and one cached prefix. Items carry fields,
// context_tokens and rows; timings and cache state cover the whole batch.
struct batch_result {
    std::vector<result> items;
    bool                cache_hit            = false;
    bool                warm_hit             = false;  // a resident warm prefix was forked instead of a cold prefill
    size_t              shared_tokens        = 0;
    int                 rows                 = 0;
    int                 rounds               = 0;
    double              prefill_ms           = 0;
    double              scoring_ms           = 0;
    size_t              suffix_tokens        = 0;
    size_t              common_suffix_tokens = 0;
    size_t              leaf_suffix_tokens   = 0;
    size_t              token_cache_hits     = 0;
    size_t              token_cache_misses   = 0;
};

// The token accounting one context of a decision cost: its own decoded evidence plus the shared
// prefix, which is counted once per context. A decision generates nothing, so the output count is
// always zero, and a prefix-cache hit charges the cached prefix as its cached tokens. Reported over
// the engine's own record, so both front-ends report the same numbers for the same work.
common_json decision_usage(const batch_result & batch, size_t context_index);

// The batch timings, additive with `diagnostics` like every other reported number. `rounds` counts
// the scoring passes a multi-round batch needed and `rows` the branch rows it decoded, which is
// what makes a request's cost legible next to its answer.
common_json decision_timings(const batch_result & batch);

// Scores decisions on an existing context with the sequence ids [seq_base, seq_base + n_seqs):
// one keeps the cached static prefix; the rest hold one trunk (prefix + context) per context in
// flight, then branches. The context needs a unified KV cache so branches share the trunk's cells.
class engine {
  public:
    engine(llama_context * ctx, llama_seq_id seq_base, int n_seqs, int n_warm = 0);

    // Resident warm slots a KV budget buys, from the model KV geometry and the cache types. The
    // engine owns the derivation as well as the limit, so a caller passes geometry and never clamps
    // the result again. Each slot is charged one whole window of cells, which is an upper bound:
    // decode_keep clears the sequence and decodes from pos 0, so a filled slot holds only that
    // turn's tokens. A budget below one full window still yields one slot so the tier stays usable,
    // and a budget of 0, an unsized window, or a model with no KV geometry disables the tier.
    static int warm_slots_for_budget(int n_layer, int n_embd, int n_head, int n_kv, int n_ctx, int budget_mb,
                                     ggml_type type_k, ggml_type type_v);

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

    // Compiles field inputs into the scoring plan. Pure: it tokenizes (through the per-engine
    // cache) and lays out the trie, but never touches the context or the KV cache.
    compiled_fields compile_fields(const std::vector<field_input> & inputs, const options & opt) const;

    // Contexts are prefilled together and their branches scored together, in groups sized to fit
    // the sequence budget; results keep the order of the contexts. The plan is compiled by the
    // caller through compile_fields, so a caller that scores the same fields twice compiles them
    // once. A null plan is a caller error.
    batch_result decide_batch(const compiled_fields &          plan,
                              const std::string &              shared_text,
                              const std::vector<std::string> & contexts,
                              const options &                  opt);

    // Scores a plan against pre-tokenized contexts: the token-replay entry a sidecar uses for a
    // token snapshot, where the caller owns the token lists. `shared` and `contexts` are the exact
    // token lists the text entry would produce (shared with special tokens on, contexts with
    // special tokens off when `shared` is non-empty), so both entries are bit-identical for the
    // same inputs. A null plan is a caller error.
    batch_result decide_batch_tokens(const tokens_t &              shared,
                                     const std::vector<tokens_t> & contexts,
                                     const compiled_fields &       plan,
                                     const options &               opt);

    // Scores a plan against a sequence that is already decoded (`src`), so a caller answers about a
    // live session without re-prefilling its transcript. One trunk is forked from `src` and decodes
    // only the plan's shared suffix head, prefixed by `tail_before_common`; the branches then start
    // at `base_pos` plus that head. `src` is never removed or decoded, so its cells survive the
    // decision. `base_pos` must be the source's next position; any other value is rejected so a
    // wrong caller cannot score at shifted positions.
    batch_result decide_batch_from_seq(llama_seq_id src, llama_pos base_pos, const compiled_fields & plan,
                                       const options & opt, const std::string & tail_before_common = "");

    // Token-replay entry with a resident warm prefix (the sidecar's session warm cache). The first
    // call for a `warm_tag` cold-prefills `tokens` into a reserved resident sequence and keeps it;
    // a later call for the same `warm_tag` forks that resident prefix instead of re-prefilling, so
    // repeated decisions on one turn cost only the suffix decode. A miss and a hit are bit-identical
    // (both fork a prefix that holds exactly `tokens`). When the warm tier is off (no resident slots,
    // `warm_tag` empty, or caching disabled) it falls back to a cold `decide_batch_tokens` replay.
    batch_result decide_warm(const tokens_t & tokens, const compiled_fields & plan, const options & opt,
                             const std::string & warm_tag);

  private:
    // The most resident warm prefixes an engine keeps. The engine owns the number, so a caller that
    // sizes sequences above the pool asks for a count here instead of inventing a second limit.
    static constexpr int WARM_MAX_SLOTS = 8;

    // KV cells the resident warm prefixes hold, skipping `except`: the sequence a fork reads from,
    // whose own cells that fork's peak already counts (pass -1 to charge every slot). The prefixes
    // stay resident for the whole decision, so both capacity preflights subtract them from the
    // window.
    size_t resident_warm_cells(llama_seq_id except) const;

    // Peak KV cells one decision needs: `persistent` cells held once (a cached prefix or a session
    // source), one trunk per group member, and the branch wave decoded above the trunks. Pure plan
    // geometry: it never reads the memory, so a caller may run it before allocating any sequence.
    static size_t peak_kv_cells(size_t persistent, size_t group, size_t trunk_len,
                                size_t branch_wave, size_t branch_len);

    // Longest branch path a plan decodes, in tokens. Pure plan geometry.
    static size_t max_branch_tokens(const compiled_fields & plan);

    // Refuse a decision whose peak exceeds the window minus the resident warm prefixes it cannot
    // evict. `resident_except` is the sequence the fork reads from, whose own cells the peak already
    // counts (pass -1 to charge every slot). Both entry points and `decide_warm` share this check.
    void check_decision_capacity(size_t peak, llama_seq_id resident_except) const;

    // The single constructor for the flags a save writes and a load uses. Pure: it reads only the
    // requested format and scope, never the engine capability, so a capability downgrade can only
    // change how a state is saved, never how saved bytes are read.
    static llama_state_seq_flags state_load_flags(bool on_device, bool partial = false);

    // True while the engine may save a partial (recurrent-only) state to device staging. A failed
    // partial device save retires this one-way capability for the process; the host format remains.
    bool partial_state_capable() const { return partial_device_capable_; }

    // Remove every cell of `count` sequences starting at `first` in this engine's memory.
    void clear_seqs(llama_seq_id first, int count);

    // Clear the pool sequences branches are forked into. The cached prefix (seq_snap) is never
    // touched, so prefix cache reuse survives a cleanup.
    void clear_pool_seqs();

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

    // Resident warm prefixes (sidecar session warm cache): a bounded set of sequences above the
    // engine pool, each holding one session turn's token list decoded and kept resident. Only the
    // owner's scheduler thread touches them. Touched only through decide_warm.
    int n_warm_ = 0;

    struct warm_slot {
        std::string tag;             // warm identity (session content hash); empty = free
        llama_pos   pos       = -1;  // = tokens.size() when filled
        int64_t     last_used = 0;
    };

    std::vector<warm_slot> warm_slots_;
    int64_t                warm_clock_ = 0;

    mutable std::unordered_map<std::string, tokens_t> token_cache_;
    // Insertion order of token_cache_ keys, so eviction drops the oldest entry instead of the whole
    // map. A miss clears nothing: the entries that would have hit stay resident. Because the cache
    // is a pure function of (text, add_special) over the vocabulary, eviction can only change cost.
    mutable std::deque<std::string>                   token_cache_order_;
    mutable size_t                                    token_cache_hits_   = 0;
    mutable size_t                                    token_cache_misses_ = 0;
    static constexpr size_t                           token_cache_limit_  = 1024;

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

    // resident warm prefix helpers: keep `tokens` decoded on a reserved warm sequence, and pick a
    // slot (free first, else least-recently-used) for a new warm identity
    void decode_keep(llama_seq_id seq, const tokens_t & tokens);
    int  warm_slot_for(const std::string & tag) const;
    int  alloc_warm_slot();
    void touch_warm(int idx);

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
