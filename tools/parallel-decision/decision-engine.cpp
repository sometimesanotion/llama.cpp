#include "decision-engine.h"
#include "decision-protocol.h"

#include "chat.h"
#include "common.h"
#include "labels.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace llama_decision {

namespace {

double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

struct decision_field {
    tokens_t              suffix;
    std::vector<tokens_t> paths;

    std::vector<int> active;  // greedy walk
    tokens_t         chosen;

    std::vector<tokens_t> node_prefix;   // tree: trie nodes with more than one allowed next token
    std::vector<tokens_t> node_options;
    std::vector<float>    probs;

    bool  use_tree     = false;
    int   winner       = -1;
    float path_score   = 1.0f;
    int   scored_nodes = 0;
    float temperature  = 1.0f;

    decision_field(tokens_t s, std::vector<tokens_t> p) : suffix(std::move(s)), paths(std::move(p)) {
        for (int i = 0; i < (int) paths.size(); ++i) {
            active.push_back(i);
        }
    }

    // Greedy: follow single-path edges for free; return the options at the next divergence.
    tokens_t options() {
        while (active.size() > 1) {
            const size_t depth = chosen.size();
            tokens_t opts;
            for (int i : active) {
                if (depth >= paths[i].size()) {
                    throw std::runtime_error("candidate paths must be distinct and terminated");
                }
                if (std::find(opts.begin(), opts.end(), paths[i][depth]) == opts.end()) {
                    opts.push_back(paths[i][depth]);
                }
            }
            if (opts.size() > 1) {
                return opts;
            }
            chosen.push_back(opts[0]);
        }
        winner = active.empty() ? -1 : active[0];
        return {};
    }

    void select(llama_token tok, float p) {
        const size_t depth = chosen.size();
        std::vector<int> remaining;
        for (int i : active) {
            if (paths[i][depth] == tok) {
                remaining.push_back(i);
            }
        }
        active = remaining;
        chosen.push_back(tok);
        path_score *= p;
        scored_nodes += 1;
    }

    void build_nodes() {
        std::vector<std::pair<std::vector<int>, size_t>> stack;
        std::vector<int> root(paths.size());
        for (int i = 0; i < (int) paths.size(); ++i) {
            root[i] = i;
        }
        stack.push_back({ root, 0 });
        while (!stack.empty()) {
            auto [act, depth] = stack.back();
            stack.pop_back();
            if (act.size() <= 1) {
                continue;
            }
            tokens_t opts;
            for (int i : act) {
                if (depth >= paths[i].size()) {
                    throw std::runtime_error("candidate paths must be distinct and terminated");
                }
                if (std::find(opts.begin(), opts.end(), paths[i][depth]) == opts.end()) {
                    opts.push_back(paths[i][depth]);
                }
            }
            if (opts.size() > 1) {
                node_prefix.emplace_back(paths[act[0]].begin(), paths[act[0]].begin() + depth);
                node_options.push_back(opts);
            }
            for (llama_token tok : opts) {
                std::vector<int> sub;
                for (int i : act) {
                    if (paths[i][depth] == tok) {
                        sub.push_back(i);
                    }
                }
                stack.push_back({ sub, depth + 1 });
            }
        }
    }

    int tree_rows() const {
        int n = 0;
        for (const auto & p : node_prefix) {
            n += (int) (suffix.size() + p.size());
        }
        return n;
    }

    // Exact constrained distribution: log-softmax at each node over its allowed tokens,
    // summed along every candidate path, normalised over candidates.
    void finish_tree(const std::vector<std::vector<float>> & node_scores) {
        std::vector<std::vector<float>> node_logp;
        for (const auto & s : node_scores) {
            const float mx = *std::max_element(s.begin(), s.end());
            double z = 0;
            for (float x : s) {
                z += std::exp(x - mx);
            }
            const float lz = mx + (float) std::log(z);
            std::vector<float> lp;
            for (float x : s) {
                lp.push_back(x - lz);
            }
            node_logp.push_back(lp);
        }
        std::vector<float> path_lp(paths.size(), 0.0f);
        for (size_t i = 0; i < paths.size(); ++i) {
            for (size_t n = 0; n < node_prefix.size(); ++n) {
                const auto & pre = node_prefix[n];
                if (pre.size() >= paths[i].size() || !std::equal(pre.begin(), pre.end(), paths[i].begin())) {
                    continue;
                }
                const auto & opts = node_options[n];
                const auto   it   = std::find(opts.begin(), opts.end(), paths[i][pre.size()]);
                if (it == opts.end()) {
                    throw std::runtime_error("candidate token missing from its trie node");
                }
                path_lp[i] += node_logp[n][it - opts.begin()];
            }
        }
        const int best = (int) (std::max_element(path_lp.begin(), path_lp.end()) - path_lp.begin());
        probs          = softmax(path_lp, temperature);
        winner         = best;
        path_score     = probs[best];
        scored_nodes   = (int) node_prefix.size();
    }
};

} // namespace

// The plan's private layout: the deduplicated fields plus the input-to-field map and the suffix
// head that was hoisted onto the trunk. Hidden here so the trie stays an implementation detail.
struct compiled_fields::impl {
    std::vector<decision_field> fields;
    std::vector<size_t>         field_first; // input field -> scored field (identical fields share one)
    tokens_t                    plan_common; // shared suffix head hoisted onto every trunk
};

compiled_fields::compiled_fields() = default;
compiled_fields::~compiled_fields() = default;
compiled_fields::compiled_fields(compiled_fields &&) noexcept = default;
compiled_fields & compiled_fields::operator=(compiled_fields &&) noexcept = default;

std::vector<float> softmax(const std::vector<float> & logits, float temperature) {
    std::vector<float> out(logits.size(), 0.0f);
    if (logits.empty()) {
        return out;
    }
    if (!(temperature > 0.0f)) {
        temperature = 1.0f;
    }
    const float mx = *std::max_element(logits.begin(), logits.end());
    double      z  = 0.0;
    for (float x : logits) {
        z += std::exp((double) ((x - mx) / temperature));
    }
    for (size_t i = 0; i < logits.size(); ++i) {
        out[i] = (float) (std::exp((double) ((logits[i] - mx) / temperature)) / z);
    }
    return out;
}

std::string make_prefix_tag(const std::string & system_text, const std::string & after,
                            const std::string & prompt_version) {
    // the trailing separator byte keeps the same chained FNV-1a as the pre-refactor mixing
    const uint64_t h = fnv1a64(prompt_version + "\x1f" + system_text + "\x1f" + after + "\x1f");
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%016llx", (unsigned long long) h);
    return std::string("decision-prefix-v1:") + buf;
}

// ---------------------------------------------------------------- engine

engine::engine(llama_context * ctx, llama_seq_id seq_base, int n_seqs)
    : ctx(ctx), model(llama_get_model(ctx)), vocab(llama_model_get_vocab(llama_get_model(ctx))),
      mem(llama_get_memory(ctx)), seq_snap(seq_base), seq_pool(seq_base + 1), n_pool(n_seqs - 1) {
    if (n_seqs < 3) {
        throw std::invalid_argument("a decision engine needs at least 3 sequences");
    }
    const int n_swa = llama_model_n_swa(model);
    if (n_swa > 0) {
        swa_ = (llama_pos) n_swa;
    }
    // Recurrent and hybrid memory keeps state outside the KV cache, so an attention-only copy
    // drops it. The hybrid fork copies attention cells by metadata and the recurrent state from a
    // partial save; the full restore stays the fallback when the partial format is unavailable.
    const bool recurrent = llama_model_is_recurrent(model) || llama_model_is_hybrid(model);
    probe_fork_ = recurrent ? (partial_state_capable() ? fork_kind::hybrid : fork_kind::restore) : fork_kind::copy;
}

void engine::select_fork(const std::string & requested) {
    if (requested == "copy") {
        // A copy fork only moves attention KV cells. Recurrent state (e.g. LFM2's short
        // convolution) lives outside the KV cache, so copy would silently drop it.
        if (probe_fork_ != fork_kind::copy) {
            throw std::invalid_argument("fork \"copy\" is not supported by a recurrent model; use auto or restore");
        }
        active_fork_ = fork_kind::copy;
    } else if (requested == "restore") {
        active_fork_ = fork_kind::restore;
    } else if (requested == "hybrid") {
        active_fork_ = fork_kind::hybrid;
    } else if (requested == "auto") {
        // The partial state format is the fast path for a recurrent model; fall back to the full
        // restore once a failed partial save retires the capability for the process.
        active_fork_ = probe_fork_;
        if (active_fork_ == fork_kind::hybrid && !partial_state_capable()) {
            active_fork_ = fork_kind::restore;
        }
    } else {
        throw std::invalid_argument("fork must be auto, copy, restore or hybrid");
    }
}

llama_state_seq_flags engine::state_load_flags(bool on_device, bool partial) {
    llama_state_seq_flags flags = LLAMA_STATE_SEQ_FLAGS_NONE;
    if (on_device) {
        flags |= LLAMA_STATE_SEQ_FLAGS_ON_DEVICE;
    }
    if (partial) {
        flags |= LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    }
    return flags;
}

engine::saved_state engine::save_seq(llama_seq_id seq, bool prefer_device, bool partial) const {
    // A sequence that was never decoded has no state; saving it must be an error, never a
    // header-only state that a later load would restore as nothing. Occupancy is authoritative:
    // the sequence state writer always emits at least a header, so its byte count cannot tell an
    // empty sequence apart from a decoded one.
    if (llama_memory_seq_pos_max(mem, seq) < 0) {
        throw std::runtime_error("failed to save a decision sequence state: no state is available");
    }
    // The device path stages the tensor bytes in the context staging buffer and returns only a
    // small metadata header on the host, so a saved device state is not self-contained. Prefer it
    // only where the state is consumed before the next save; otherwise use the self-contained host
    // format. A failed device save retires the device path for its scope for the process lifetime.
    const auto                          scope      = device_scope_.find(seq);
    const bool                          scope_ok   = scope == device_scope_.end() || scope->second == partial;
    const bool                          capable    = partial ? partial_device_capable_ : device_capable_;
    const llama_state_seq_flags         device     = state_load_flags(true, partial);
    if (prefer_device && capable && scope_ok) {
        const size_t size = llama_state_seq_get_size_ext(ctx, seq, device);
        if (size != 0) {
            std::vector<uint8_t> buf(size);
            if (llama_state_seq_get_data_ext(ctx, buf.data(), buf.size(), seq, device) == size) {
                device_scope_[seq] = partial;
                return { std::move(buf), device };
            }
        }
        if (partial) {
            partial_device_capable_ = false;
        } else {
            device_capable_ = false;
        }
    }
    const llama_state_seq_flags host = state_load_flags(false, partial);
    const size_t size = llama_state_seq_get_size_ext(ctx, seq, host);
    std::vector<uint8_t> buf(size);
    if (size == 0 || llama_state_seq_get_data_ext(ctx, buf.data(), buf.size(), seq, host) != size) {
        throw std::runtime_error("failed to save a decision sequence state");
    }
    return { std::move(buf), host };
}

void engine::load_seq(const saved_state & state, llama_seq_id seq) const {
    if (state.bytes.empty()) {
        // save_seq refuses to produce an empty state, so this can only be a caller bug; fail
        // loudly instead of silently restoring nothing.
        throw std::runtime_error("cannot load an empty decision sequence state");
    }
    // The format and the scope belong to the value: the device flag is never substituted for the
    // host flag or the other way around, so a stale engine flag cannot misread the bytes. A failed
    // device load is fatal because the host bytes are not present to fall back to.
    const size_t n = llama_state_seq_set_data_ext(ctx, state.bytes.data(), state.bytes.size(), seq, state.flags);
    if (n == 0) {
        throw std::runtime_error(state.on_device() ? "failed to restore a device decision sequence state"
                                                   : "failed to restore a decision sequence state");
    }
}

void engine::fork_into(llama_seq_id src, llama_seq_id dst, const saved_state * src_state) {
    llama_memory_seq_rm(mem, dst, -1, -1);
    if (active_fork_ == fork_kind::restore) {
        if (src_state == nullptr) {
            throw std::runtime_error("a restore fork needs a saved parent state");
        }
        load_seq(*src_state, dst);
        return;
    }
    // A sliding-window cache no longer holds cells older than the window; copy only what survives.
    llama_pos p0 = -1;
    if (swa_ > 0) {
        const llama_pos pmax = llama_memory_seq_pos_max(mem, src);
        if (pmax >= 0 && pmax + 1 > swa_) {
            p0 = pmax + 1 - swa_;
        }
    }
    llama_memory_seq_cp(mem, src, dst, p0, -1);
    if (active_fork_ == fork_kind::hybrid) {
        // The shared attention cells are only metadata; the recurrent part lives outside the KV
        // cache and must be copied from a partial parent state. An empty state means the parent has
        // no decoded state yet, so the copy above is all there is.
        if (src_state == nullptr) {
            throw std::runtime_error("a hybrid fork needs a partial parent state");
        }
        if (!src_state->bytes.empty()) {
            if ((src_state->flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
                throw std::runtime_error("a hybrid fork needs a partial parent state");
            }
            load_seq(*src_state, dst);
        }
    }
}

// The saved parent state a branch fork loads: none for a copy fork, otherwise the trunk's entry in
// the wave's saved states. Looked up once so every fork site shares one bounds check.
const engine::saved_state * engine::parent_state_for(const std::vector<saved_state> * parent_states, llama_seq_id trunk) const {
    if (active_fork_ == fork_kind::copy) {
        return nullptr;
    }
    if (parent_states == nullptr) {
        throw capacity_error("a save/restore fork needs parent states");
    }
    const size_t idx = (size_t) (trunk - seq_pool);
    if (idx >= parent_states->size()) {
        throw capacity_error("missing parent state for a decision branch");
    }
    return &(*parent_states)[idx];
}

void engine::check_cancel() const {
    if (stop_ && stop_()) {
        throw cancelled_error("the decision was cancelled");
    }
}

tokens_t engine::tokenize(const std::string & text, bool add_special) const {
    const std::string key = (add_special ? "\x01" : "\x00") + text;
    const auto it = token_cache_.find(key);
    if (it != token_cache_.end()) {
        return it->second;
    }
    tokens_t toks = common_tokenize(vocab, text, add_special, /*parse_special=*/ true);
    // a chat template may already start with the BOS text; keep a single BOS
    const llama_token bos = llama_vocab_bos(vocab);
    if (toks.size() >= 2 && toks[0] == bos && toks[1] == bos) {
        toks.erase(toks.begin());
    }
    if (token_cache_.size() >= token_cache_limit_) {
        token_cache_.clear();
    }
    token_cache_.emplace(key, toks);
    return toks;
}

void engine::clear_seqs(llama_seq_id first, int count) {
    for (int i = 0; i < count; ++i) {
        llama_memory_seq_rm(mem, first + (llama_seq_id) i, -1, -1);
    }
}

void engine::clear_pool_seqs() {
    clear_seqs(seq_pool, n_pool);
}

// Decode several prompts, each on its own sequence, packed into as few batches as n_batch allows.
void engine::decode_parts(const std::vector<prompt_part> & parts) {
    const int n_batch = (int) llama_n_batch(ctx);
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    auto flush = [&]() {
        check_cancel();
        const int rc = batch.n_tokens > 0 ? llama_decode(ctx, batch) : 0;
        common_batch_clear(batch);
        if (rc != 0) {
            llama_batch_free(batch);
            throw capacity_error(rc == 1 ? "no free KV cache space for the decision prompt"
                                         : "llama_decode failed on the decision prompt (" + std::to_string(rc) + ")");
        }
    };
    for (const auto & p : parts) {
        for (size_t i = 0; i < p.toks->size(); ++i) {
            if (batch.n_tokens == n_batch) {
                flush();
            }
            common_batch_add(batch, (*p.toks)[i], p.pos0 + (llama_pos) i, { p.seq }, false);
        }
    }
    flush();
    llama_batch_free(batch);
}

// Restore (or build) the cached static prefix on seq_snap. Only this engine's own sequences are
// touched, so it can share a context with other users (e.g. server slots).
bool engine::prepare_prefix(const tokens_t & shared, bool allow_cache, const std::string & tag) {
    // The hybrid fork copies attention by sequence metadata and loads only the recurrent part from a
    // partial state, so its prefix state is partial. The LRU checkpoint stays a self-contained full
    // host state: restoring it must rebuild both the attention cells a trunk copy shares and the
    // recurrent state the partial prefix save reads.
    const bool partial_state = active_fork_ == fork_kind::hybrid;
    const bool restorable    = active_fork_ == fork_kind::restore || partial_state;

    if (allow_cache && !shared.empty() && restorable && !tag.empty()) {
        for (size_t i = 0; i < prefix_lru_.size(); ++i) {
            if (prefix_lru_[i].tag == tag) {
                prefix_entry entry = std::move(prefix_lru_[i]);
                prefix_lru_.erase(prefix_lru_.begin() + (long) i);
                clear_seqs(seq_snap, n_pool + 1); // the state moves to seq_snap, so everything resets
                // the LRU entry is host format; refresh the device copy for the current request only
                load_seq(entry.state, seq_snap);
                prefix_state_ = save_seq(seq_snap, true, partial_state);
                cached        = shared;
                cached_tag    = tag;
                // move the entry back to the front; the state bytes are not copied
                prefix_lru_.insert(prefix_lru_.begin(), std::move(entry));
                return true;
            }
        }
    } else {
        const bool tag_ok = tag.empty() || tag == cached_tag;
        if (allow_cache && tag_ok && !shared.empty() && shared == cached &&
            llama_memory_seq_pos_max(mem, seq_snap) == (llama_pos) cached.size() - 1) {
            // The cached prefix on seq_snap survives across requests, but a strategy change means
            // the prefix state a fork loads must be refreshed for the new scope.
            if (restorable) {
                const bool have_partial = (prefix_state_.flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != 0;
                if (prefix_state_.bytes.empty() || have_partial != partial_state) {
                    prefix_state_ = save_seq(seq_snap, true, partial_state);
                }
            }
            return true;
        }
    }

    clear_seqs(seq_snap, n_pool + 1);
    cached.clear();
    cached_tag.clear();
    prefix_state_ = saved_state{};
    if (!shared.empty()) {
        decode_parts({ { &shared, 0, seq_snap } });
        cached     = shared;
        cached_tag = tag;
        if (restorable) {
            prefix_state_ = save_seq(seq_snap, true, partial_state);
            if (!tag.empty()) {
                // the LRU entry must outlive the request, so it uses the self-contained host format
                saved_state host_state = save_seq(seq_snap, false, false);
                assert(!host_state.on_device());
                prefix_lru_.insert(prefix_lru_.begin(), { tag, std::move(host_state) });
                while (prefix_lru_.size() > prefix_lru_capacity_) {
                    prefix_lru_.pop_back();
                }
            }
        }
    }
    return false;
}

// Score each branch as its own sequence forked from its trunk; return each branch's last-token
// logits restricted to its candidate tokens. Groups are bounded by free sequences and batch rows.
std::vector<engine::branch_score> engine::score_branches(const std::vector<branch> & branches, llama_seq_id first, int n_free,
                                                         const std::vector<saved_state> * parent_states,
                                                         bool allow_bypass) {
    std::vector<branch_score> result(branches.size());

    // A single branch does not need its own sequence: decode it on the trunk and trim afterwards.
    // Only bypass when the branch fits one batch; an oversize branch falls through to the chunked
    // path, which rejects it as a capacity error instead of overflowing llama_decode.
    if (allow_bypass && branches.size() == 1 && active_fork_ == fork_kind::copy &&
        branches[0].toks.size() <= (size_t) llama_n_batch(ctx)) {
        const llama_seq_id seq  = branches[0].trunk;
        const auto &       toks = branches[0].toks;
        llama_batch batch = llama_batch_init((int) toks.size(), 0, 1);
        for (size_t i = 0; i < toks.size(); ++i) {
            common_batch_add(batch, toks[i], branches[0].pos0 + (llama_pos) i, { seq }, i + 1 == toks.size());
        }
        check_cancel();
        const int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            throw capacity_error(rc == 1 ? "no free KV cache space for the decision branch"
                                         : "llama_decode failed on the decision branch (" + std::to_string(rc) + ")");
        }
        llama_synchronize(ctx); // the scored rows are on the host only after the async decode drains
        gather_candidates((int) toks.size() - 1, branches[0].cands, result[0]);
        llama_memory_seq_rm(mem, seq, branches[0].pos0, -1); // the trunk keeps its prefix and context only
        llama_synchronize(ctx);
        return result;
    }

    const size_t batch_cap = active_fork_ == fork_kind::restore ? 512u : (size_t) llama_n_batch(ctx);
    const int max_rows = (int) std::min((size_t) llama_n_batch(ctx), batch_cap);
    size_t start = 0;
    while (start < branches.size()) {
        size_t end  = start;
        int    rows = 0;
        while (end < branches.size() && (int) (end - start) < n_free && rows + (int) branches[end].toks.size() <= max_rows) {
            rows += (int) branches[end].toks.size();
            ++end;
        }
        if (end == start) {
            // One branch is larger than a single batch: decode it alone in chunks within its own
            // sequence. A suffix is teacher-forced, so splitting it across decodes keeps the same
            // state and the final chunk yields the scored position.
            const size_t b = start;
            const llama_seq_id seq = first;
            fork_into(branches[b].trunk, seq, parent_state_for(parent_states, branches[b].trunk));
            const auto & toks = branches[b].toks;
            for (size_t i = 0; i < toks.size(); i += (size_t) max_rows) {
                const size_t n = std::min((size_t) max_rows, toks.size() - i);
                llama_batch batch = llama_batch_init((int) n, 0, 1);
                int out_idx = -1;
                for (size_t j = 0; j < n; ++j) {
                    const bool last = i + j + 1 == toks.size();
                    if (last) {
                        out_idx = batch.n_tokens;
                    }
                    common_batch_add(batch, toks[i + j], branches[b].pos0 + (llama_pos) (i + j), { seq }, last);
                }
                check_cancel();
                const int rc = llama_decode(ctx, batch);
                llama_batch_free(batch);
                if (rc != 0) {
                    throw capacity_error(rc == 1 ? "no free KV cache space for the decision branch"
                                                 : "llama_decode failed on the decision branch (" + std::to_string(rc) + ")");
                }
                if (out_idx >= 0) {
                    llama_synchronize(ctx);
                    gather_candidates(out_idx, branches[b].cands, result[b]);
                }
            }
            clear_seqs(first, 1);
            llama_synchronize(ctx);
            ++start;
            continue;
        }
        llama_batch batch = llama_batch_init(rows, 0, 1);
        std::vector<int> out_idx;
        for (size_t b = start; b < end; ++b) {
            const llama_seq_id seq = first + (llama_seq_id) (b - start);
            fork_into(branches[b].trunk, seq, parent_state_for(parent_states, branches[b].trunk));
            const auto & toks = branches[b].toks;
            for (size_t i = 0; i < toks.size(); ++i) {
                const bool last = i + 1 == toks.size();
                if (last) {
                    out_idx.push_back(batch.n_tokens);
                }
                common_batch_add(batch, toks[i], branches[b].pos0 + (llama_pos) i, { seq }, last);
            }
        }
        check_cancel();
        const int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            throw capacity_error(rc == 1 ? "no free KV cache space for the decision branches"
                                         : "llama_decode failed on the decision branches (" + std::to_string(rc) + ")");
        }
        llama_synchronize(ctx); // the scored rows are on the host only after the async decode drains
        for (size_t b = start; b < end; ++b) {
            gather_candidates(out_idx[b - start], branches[b].cands, result[b]);
        }
        clear_seqs(first, (int) (end - start));
        llama_synchronize(ctx);
        start = end;
    }
    return result;
}

void engine::gather_candidates(int out_idx, const tokens_t & cands, branch_score & out) {
    const float * logits = llama_get_logits_ith(ctx, out_idx);
    if (logits == nullptr) {
        throw std::runtime_error("no logits for the scored decision position");
    }
    for (llama_token t : cands) {
        out.cand_logits.push_back(logits[t]);
    }
}

result engine::decide(const std::string & shared_text, const std::string & context_text,
                      const std::vector<field_input> & inputs, const options & opt) {
    batch_result b = decide_batch(shared_text, { context_text }, inputs, opt);
    result r = std::move(b.items[0]);
    r.cache_hit     = b.cache_hit;
    r.shared_tokens = b.shared_tokens;
    r.rounds        = b.rounds;
    r.prefill_ms    = b.prefill_ms;
    r.scoring_ms    = b.scoring_ms;
    return r;
}

compiled_fields engine::compile_fields(const std::vector<field_input> & inputs, const options & opt) const {
    compiled_fields plan;
    plan.p = std::make_unique<compiled_fields::impl>();
    std::vector<decision_field> & fields      = plan.p->fields;
    std::vector<size_t> &         field_first = plan.p->field_first;

    int branches = 0; // round-1 branches of one context
    for (const auto & in : inputs) {
        const size_t n = in.candidates.size();
        if (n < 1 || n > 255) {
            throw std::invalid_argument("each field needs 1-255 allowed values");
        }
        tokens_t suffix;
        std::vector<tokens_t> paths;
        if (opt.split_boundary) {
            for (const auto & c : in.candidates) {
                paths.push_back(tokenize(c + "\n", false));
            }
            suffix = tokenize(in.suffix, false);
        } else {
            // Tokenise each complete "suffix + value + terminator" and split at the longest token
            // prefix shared by every candidate: the value's first token is exactly what the model
            // would write itself (e.g. `":` then ` true`, not `": ` then `true`).
            std::vector<tokens_t> seqs;
            for (const auto & c : in.candidates) {
                seqs.push_back(tokenize(in.suffix + c + "\n", false));
            }
            size_t common = seqs[0].size();
            for (const auto & s : seqs) {
                size_t c = 0;
                while (c < std::min(common, s.size()) && s[c] == seqs[0][c]) {
                    ++c;
                }
                common = std::min({ common, c, s.size() - 1 });
            }
            suffix.assign(seqs[0].begin(), seqs[0].begin() + common);
            for (const auto & s : seqs) {
                paths.emplace_back(s.begin() + common, s.end());
            }
        }
        if (suffix.empty()) {
            throw std::invalid_argument("a field suffix must not be empty");
        }
        for (size_t a = 0; a < paths.size(); ++a) {
            for (size_t b = 0; b < a; ++b) {
                const size_t m = std::min(paths[a].size(), paths[b].size());
                if (std::equal(paths[a].begin(), paths[a].begin() + m, paths[b].begin())) {
                    throw std::invalid_argument("two allowed values tokenise to colliding paths");
                }
            }
        }
        decision_field field(suffix, paths);
        field.temperature = in.temperature;
        field.build_nodes();
        field.use_tree = opt.mode == "tree" ? true : opt.mode == "greedy" ? false : n <= opt.tree_max;
        // exact-token dedup: identical fields (suffix, paths, temperature, mode) score once
        size_t canon = fields.size();
        for (size_t u = 0; u < fields.size(); ++u) {
            if (fields[u].temperature == field.temperature && fields[u].use_tree == field.use_tree &&
                fields[u].suffix == field.suffix && fields[u].paths == field.paths) {
                canon = u;
                break;
            }
        }
        if (canon == fields.size()) {
            branches += field.use_tree ? (int) field.node_prefix.size() : 1;
            fields.push_back(std::move(field));
        }
        field_first.push_back(canon);
    }

    // Hoist a suffix head shared by every field onto the trunk, so each branch decodes only its
    // unique tail. The saving is `common` tokens on every branch but the trunk, so a short head
    // still pays off once many questions share it.
    constexpr size_t HOIST_MIN_TOKENS = 4;  // below this the extra trunk decode is not worth it
    constexpr size_t HOIST_BUDGET    = 32;  // tokens saved across all branches before hoisting
    size_t suffix_tokens = 0;
    for (const auto & fd : fields) {
        suffix_tokens += fd.suffix.size();
    }
    tokens_t plan_common;
    if (opt.optimize && fields.size() > 1) {
        size_t common = fields[0].suffix.size() - 1; // keep at least one token per branch
        for (const auto & fd : fields) {
            common = std::min(common, fd.suffix.size() - 1);
            size_t j = 0;
            while (j < common && fields[0].suffix[j] == fd.suffix[j]) {
                ++j;
            }
            common = j;
        }
        if (common >= HOIST_MIN_TOKENS && common * (fields.size() - 1) >= HOIST_BUDGET) {
            plan_common.assign(fields[0].suffix.begin(), fields[0].suffix.begin() + common);
            for (auto & fd : fields) {
                fd.suffix.erase(fd.suffix.begin(), fd.suffix.begin() + common);
            }
        }
    }
    size_t leaf_suffix_tokens = 0;
    int    total              = 0;
    for (const auto & fd : fields) {
        leaf_suffix_tokens += fd.suffix.size();
        size_t max_path = 0;
        for (const auto & p : fd.paths) {
            max_path = std::max(max_path, p.size());
        }
        total += fd.use_tree ? fd.tree_rows() : (int) (fd.suffix.size() + max_path);
    }

    plan.field_count          = fields.size();
    plan.suffix_tokens        = suffix_tokens;
    plan.common_suffix_tokens = plan_common.size();
    plan.leaf_suffix_tokens   = leaf_suffix_tokens;
    plan.rows                 = total;
    plan.branches             = branches;
    plan.p->plan_common       = std::move(plan_common);
    return plan;
}

batch_result engine::decide_batch(const std::string & shared_text, const std::vector<std::string> & contexts,
                                  const std::vector<field_input> & inputs, const options & opt) {
    return decide_batch(compile_fields(inputs, opt), shared_text, contexts, opt);
}

batch_result engine::decide_batch(const compiled_fields &          plan,
                                  const std::string &              shared_text,
                                  const std::vector<std::string> & contexts,
                                  const options &                  opt) {
    if (opt.mode != "auto" && opt.mode != "tree" && opt.mode != "greedy") {
        throw std::invalid_argument("mode must be auto, tree or greedy");
    }
    if (contexts.empty()) {
        throw std::invalid_argument("a decision needs at least one context");
    }
    if (plan.p == nullptr) {
        throw std::runtime_error("the decision plan is empty");
    }
    select_fork(opt.fork);
    stop_  = opt.should_stop;
    yield_ = opt.yield;
    llama_synchronize(ctx); // drain any work left by the previous decision before reusing sequences
    check_cancel();
    const tokens_t shared = tokenize(shared_text, true);
    std::vector<tokens_t> prefixes;
    for (const auto & text : contexts) {
        prefixes.push_back(tokenize(text, shared.empty()));
        if (prefixes.back().empty()) {
            throw std::invalid_argument("the decision context must not be empty");
        }
    }

    const std::vector<decision_field> & fields          = plan.p->fields;
    const tokens_t &                    plan_common     = plan.p->plan_common;
    const int                           total           = plan.rows;
    const int                           branches        = plan.branches;
    const size_t                        suffix_tokens   = plan.suffix_tokens;
    const size_t                        leaf_suffix_tokens = plan.leaf_suffix_tokens;

    // every trunk decodes its context plus the hoisted suffix head; branches start after it
    std::vector<tokens_t> tails = prefixes;
    if (!plan_common.empty()) {
        for (auto & t : tails) {
            t.insert(t.end(), plan_common.begin(), plan_common.end());
        }
    }

    batch_result out;
    out.shared_tokens        = shared.size();
    out.rows                 = total * (int) contexts.size();
    out.suffix_tokens        = suffix_tokens;
    out.common_suffix_tokens = plan_common.size();
    out.leaf_suffix_tokens   = leaf_suffix_tokens;
    out.items.resize(contexts.size());

    size_t max_tail = 0;
    for (const auto & t : tails) {
        max_tail = std::max(max_tail, t.size());
    }
    size_t max_branch = 0;
    for (const auto & fd : fields) {
        for (const auto & p : fd.paths) {
            max_branch = std::max(max_branch, p.size());
        }
    }

    // Bounded decision context: reject a request whose peak KV use cannot fit before touching the
    // cache, so the failure is a clean client error. The peak estimate uses the full n_ctx as the
    // budget and ignores resident chat cells (no free-cell API exists), so it is conservative and
    // may reject a request that would fit; the decode-time rc==1 path is the actual guarantee
    // against a partial restore. Never truncate.
    const size_t per_group = std::clamp<size_t>(n_pool / (1 + branches), 1, contexts.size());
    const size_t group     = std::min(per_group, contexts.size());
    const size_t n_free    = (size_t) std::max(0, n_pool - (int) group);
    const size_t branch_wave = std::min(n_free, (size_t) branches * group);
    const size_t trunk_len   = shared.size() + max_tail;
    const size_t peak        = shared.size()
                             + group * trunk_len
                             + branch_wave * (trunk_len + max_branch);
    const size_t budget      = (size_t) llama_n_ctx(ctx);
    if (peak > budget) {
        throw capacity_error("decision context budget exceeded: the request needs up to " + std::to_string(peak) +
                             " tokens but the context holds " + std::to_string(budget) +
                             " (raise --ctx-size)");
    }

    const auto t0 = std::chrono::steady_clock::now();
    out.cache_hit = prepare_prefix(shared, opt.allow_cache, opt.cache_tag);
    out.prefill_ms += ms_since(t0);

    // Every exit from here must leave the pool sequences empty: a failed or cancelled decision
    // must not leave cells in the shared cache to starve chat decodes. The cached prefix on
    // seq_snap is kept, so prefix cache reuse survives.
    struct pool_cleanup {
        engine * e;
        ~pool_cleanup() { e->clear_pool_seqs(); }
    } cleanup{this};

    // each context in a group holds one trunk sequence; the rest of the pool scores branches
    for (size_t g0 = 0; g0 < contexts.size(); g0 += per_group) {
        if (yield_) {
            yield_();
        }
        const size_t n_group = std::min(per_group, contexts.size() - g0);

        const auto tp = std::chrono::steady_clock::now();
        std::vector<prompt_part> parts;
        for (size_t i = 0; i < n_group; ++i) {
            const llama_seq_id trunk = seq_pool + (llama_seq_id) i;
            if (active_fork_ == fork_kind::restore) {
                llama_memory_seq_rm(mem, trunk, -1, -1);
                if (!shared.empty()) {
                    load_seq(prefix_state_, trunk);
                }
            } else {
                // copy ignores the state; hybrid loads the partial prefix into the trunk it copied
                fork_into(seq_snap, trunk, &prefix_state_);
            }
            parts.push_back({ &tails[g0 + i], (llama_pos) shared.size(), trunk });
        }
        decode_parts(parts);
        llama_synchronize(ctx); // llama_decode is asynchronous: wait for the prefill so its time isn't billed to scoring
        out.prefill_ms += ms_since(tp);

        std::vector<trunk_run> runs;
        runs.reserve(n_group);
        for (size_t i = 0; i < n_group; ++i) {
            runs.push_back({ seq_pool + (llama_seq_id) i,
                             (llama_pos) (shared.size() + tails[g0 + i].size()),
                             g0 + i, prefixes[g0 + i].size() });
        }
        run_trunk_wave(out, plan, runs, opt.bypass);
    }
    return out;
}

// The one scoring loop shared by every decide entry point. It assumes each run's trunk has already
// been forked and prefilled and that its head was decoded at the position before branch_pos.
void engine::run_trunk_wave(batch_result & out, const compiled_fields & plan, const std::vector<trunk_run> & runs,
                            bool allow_bypass) {
    const std::vector<decision_field> & fields             = plan.p->fields;
    const std::vector<size_t> &         field_first        = plan.p->field_first;
    const int                           total              = plan.rows;
    const size_t                        suffix_tokens      = plan.suffix_tokens;
    const size_t                        leaf_suffix_tokens = plan.leaf_suffix_tokens;
    const size_t                        n_group            = runs.size();
    if (n_group == 0) {
        return;
    }

    const bool save_trunks = active_fork_ == fork_kind::restore || active_fork_ == fork_kind::hybrid;
    std::vector<saved_state> trunk_states;
    if (save_trunks) {
        const bool partial = active_fork_ == fork_kind::hybrid;
        trunk_states.reserve(n_group);
        for (size_t i = 0; i < n_group; ++i) {
            assert(runs[i].trunk == seq_pool + (llama_seq_id) i);
            trunk_states.push_back(save_seq(seq_pool + (llama_seq_id) i, true, partial));
        }
    }

    // round 1 carries every tree node and each greedy field's first step; later rounds only
    // continue greedy fields that are still open
    const auto ts = std::chrono::steady_clock::now();
    std::vector<std::vector<decision_field>> state(n_group, fields);
    bool first = true;
    while (true) {
        check_cancel();
        if (yield_) {
            yield_(); // let the caller serve light requests between waves
        }
        std::vector<branch> todo;
        std::vector<std::pair<size_t, size_t>> owner; // (trunk in wave, field)
        for (size_t i = 0; i < n_group; ++i) {
            const llama_seq_id trunk = runs[i].trunk;
            const llama_pos    pos0  = runs[i].branch_pos;
            for (size_t f = 0; f < state[i].size(); ++f) {
                auto & fd = state[i][f];
                if (fd.use_tree) {
                    if (first) {
                        for (size_t n = 0; n < fd.node_prefix.size(); ++n) {
                            tokens_t ids = fd.suffix;
                            ids.insert(ids.end(), fd.node_prefix[n].begin(), fd.node_prefix[n].end());
                            todo.push_back({ trunk, pos0, ids, fd.node_options[n] });
                            owner.push_back({ i, f });
                        }
                    }
                } else {
                    tokens_t opts = fd.options();
                    if (!opts.empty()) {
                        tokens_t ids = fd.suffix;
                        ids.insert(ids.end(), fd.chosen.begin(), fd.chosen.end());
                        todo.push_back({ trunk, pos0, ids, opts });
                        owner.push_back({ i, f });
                    }
                }
            }
        }
        if (todo.empty()) {
            break;
        }
        const auto scores = score_branches(todo, seq_pool + (llama_seq_id) n_group, n_pool - (int) n_group,
                                            trunk_states.empty() ? nullptr : &trunk_states, allow_bypass && todo.size() == 1);
        out.rounds += 1;
        std::vector<std::vector<std::vector<std::vector<float>>>> tree_scores(n_group, std::vector<std::vector<std::vector<float>>>(fields.size()));
        for (size_t row = 0; row < owner.size(); ++row) {
            const auto [i, f] = owner[row];
            auto & fd = state[i][f];
            if (fd.use_tree) {
                tree_scores[i][f].push_back(scores[row].cand_logits);
            } else {
                const auto & s    = scores[row].cand_logits;
                const auto   p    = softmax(s, fd.temperature);
                const int    best = (int) (std::max_element(s.begin(), s.end()) - s.begin());
                fd.select(todo[row].cands[best], p[best]);
            }
        }
        if (first) {
            for (size_t i = 0; i < n_group; ++i) {
                for (size_t f = 0; f < fields.size(); ++f) {
                    if (state[i][f].use_tree) {
                        state[i][f].finish_tree(tree_scores[i][f]);
                    }
                }
            }
        }
        first = false;
    }
    clear_seqs(seq_pool, (int) n_group);
    for (size_t i = 0; i < n_group; ++i) {
        result & r = out.items[runs[i].out_index];
        r.context_tokens       = runs[i].context_tokens;
        r.rows                 = total;
        r.suffix_tokens        = suffix_tokens;
        r.common_suffix_tokens = plan.common_suffix_tokens;
        r.leaf_suffix_tokens   = leaf_suffix_tokens;
        std::vector<field_result> scored;
        scored.reserve(state[i].size());
        for (auto & fd : state[i]) {
            if (fd.use_tree && fd.probs.empty()) {
                fd.finish_tree({});
            }
            scored.push_back({ fd.winner, fd.path_score, fd.scored_nodes, fd.use_tree, fd.probs });
        }
        r.fields.resize(field_first.size());
        for (size_t f = 0; f < field_first.size(); ++f) {
            r.fields[f] = scored[field_first[f]];
        }
    }
    llama_synchronize(ctx); // the pool sequences are clear before the next group or call reuses them
    out.scoring_ms += ms_since(ts);
}

batch_result engine::decide_batch_from_seq(llama_seq_id src, llama_pos base_pos, const compiled_fields & plan,
                                           const options & opt, const std::string & tail_before_common) {
    if (opt.mode != "auto" && opt.mode != "tree" && opt.mode != "greedy") {
        throw std::invalid_argument("mode must be auto, tree or greedy");
    }
    if (plan.p == nullptr) {
        throw std::runtime_error("the decision plan is empty");
    }
    // The source is the session the caller wants answered; a sequence with no decoded state has
    // nothing to fork, which is a caller error rather than an empty result.
    const llama_pos src_pos_max = llama_memory_seq_pos_max(mem, src);
    if (src_pos_max < 0) {
        throw std::invalid_argument("the session source sequence has no decoded state");
    }
    // The head must continue at the source's next position. A wrong value would decode over the
    // source's own cells, so reject it instead of shifting the branch positions silently.
    if (base_pos != src_pos_max + 1) {
        throw std::invalid_argument("the session fork must continue at the source sequence's next position");
    }
    select_fork(opt.fork);
    stop_  = opt.should_stop;
    yield_ = opt.yield;
    llama_synchronize(ctx);
    check_cancel();

    // only the plan's suffix head is left to decode: the source already carries the session text
    tokens_t head = tokenize(tail_before_common, false);
    head.insert(head.end(), plan.p->plan_common.begin(), plan.p->plan_common.end());

    batch_result out;
    out.shared_tokens        = 0;
    out.rows                 = plan.rows;
    out.suffix_tokens        = plan.suffix_tokens;
    out.common_suffix_tokens = plan.common_suffix_tokens;
    out.leaf_suffix_tokens   = plan.leaf_suffix_tokens;
    out.items.resize(1);

    size_t max_branch = 0;
    for (const auto & fd : plan.p->fields) {
        for (const auto & p : fd.paths) {
            max_branch = std::max(max_branch, p.size());
        }
    }
    // The source occupies cells up to base_pos and the trunk copies them, so the peak is the
    // source plus the head and the branch suffixes decoded above it. Reject an over-budget request
    // before touching the cache; the decode rc==1 path is the actual guarantee.
    const size_t branch_wave = std::min((size_t) n_pool, (size_t) plan.branches);
    const size_t peak        = (size_t) std::max<llama_pos>(base_pos, 0) + head.size()
                             + branch_wave * (head.size() + max_branch);
    const size_t budget      = (size_t) llama_n_ctx(ctx);
    if (peak > budget) {
        throw capacity_error("decision context budget exceeded: the request needs up to " + std::to_string(peak) +
                             " tokens but the context holds " + std::to_string(budget) +
                             " (raise --ctx-size)");
    }

    // A restore or hybrid fork loads the parent from a saved state; a copy fork ignores it. Saving
    // the live source is read-only for `src`.
    saved_state src_state;
    if (active_fork_ != fork_kind::copy) {
        src_state = save_seq(src, true, active_fork_ == fork_kind::hybrid);
    }

    struct pool_cleanup {
        engine * e;
        ~pool_cleanup() { e->clear_pool_seqs(); }
    } cleanup{this};

    const auto tp = std::chrono::steady_clock::now();
    const llama_seq_id trunk = seq_pool;
    fork_into(src, trunk, &src_state);
    if (!head.empty()) {
        const std::vector<prompt_part> parts = { { &head, base_pos, trunk } };
        decode_parts(parts);
    }
    llama_synchronize(ctx);
    out.prefill_ms += ms_since(tp);

    const std::vector<trunk_run> runs = {
        { trunk, base_pos + (llama_pos) head.size(), 0, (size_t) std::max<llama_pos>(base_pos, 0) },
    };
    run_trunk_wave(out, plan, runs, opt.bypass);

    return out;
}

} // namespace llama_decision
