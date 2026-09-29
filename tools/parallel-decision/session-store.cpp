#include "session-store.h"

#include "llama.h"

#include <cstdio>
#include <vector>

namespace llama_decision {

// The shared reserved-sequence arena over [base, base + n_arena): one bitmap, owned by the
// registry, so host, clone, and file captures never collide on the same sequence id.
class basic_session_arena final : public session_arena {
  public:
    basic_session_arena(llama_seq_id base, int n_arena) : base_(base), n_arena_(n_arena), used_(n_arena, false) {
    }

    llama_seq_id alloc() override {
        for (int i = 0; i < n_arena_; ++i) {
            if (!used_[i]) {
                used_[i] = true;
                return base_ + (llama_seq_id) i;
            }
        }
        return -1; // arena full
    }

    void free_seq(llama_seq_id seq) override {
        if (seq >= base_ && seq < base_ + n_arena_) {
            used_[seq - base_] = false;
        }
    }

    size_t capacity() const override { return (size_t) n_arena_; }

    size_t used() const override {
        size_t n = 0;
        for (bool b : used_) {
            if (b) {
                ++n;
            }
        }
        return n;
    }

  private:
    llama_seq_id      base_;
    int               n_arena_;
    std::vector<bool> used_;
};

std::unique_ptr<session_arena> make_session_arena(llama_seq_id base, int n_arena) {
    return std::make_unique<basic_session_arena>(base, n_arena);
}

// The host backend: the moved session_arena behavior. Serializes the source sequence's decoded
// state in the self-contained host format (llama_state_seq_get_data_ext) and restores it into the
// registry-assigned arena sequence (llama_state_seq_set_data_ext), so the reference owns its cells
// and the origin can be cleared and reused.
class host_state_store final : public session_store {
  public:
    explicit host_state_store(const session_store_config & cfg) : cfg_(cfg) {
    }

    bool capture(const capture_request & req, capture_handle & out) override {
        if (cfg_.ctx == nullptr || req.src < 0) {
            return false;
        }
        // Host-format state is self-contained and survives the source being cleared. It is held
        // in RAM, not in a context sequence, so a retained turn never competes with chat cells.
        const size_t size = llama_state_seq_get_size_ext(cfg_.ctx, req.src, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (size == 0) {
            return false;
        }
        out.owned_state.resize(size);
        if (llama_state_seq_get_data_ext(cfg_.ctx, out.owned_state.data(), out.owned_state.size(), req.src,
                                         LLAMA_STATE_SEQ_FLAGS_NONE) != size) {
            out.owned_state.clear();
            return false;
        }
        out.backend      = session_backend::host;
        out.epoch        = 0; // the memory epoch is owned and stamped by the registry
        out.pos          = req.base_pos;
        out.resident_seq = -1;
        out.n_bytes      = size;
        out.identity     = req.identity;
        out.locator.clear();
        return true;
    }

    bool materialize(capture_handle & h, llama_seq_id & seq) override {
        if (h.resident_seq >= 0) {
            seq = h.resident_seq; // a resident handle (manifest-restored or legacy) is already loaded
            return true;
        }
        if (cfg_.ctx == nullptr || h.owned_state.empty()) {
            return false;
        }
        // One arena sequence per in-flight materialization, so two materialized references never
        // collide. unmaterialize/release free it again, which keeps a retained turn out of the
        // chat context's cells between decisions.
        if (h.materialized_seq < 0) {
            h.materialized_seq = cfg_.arena != nullptr ? cfg_.arena->alloc() : -1;
            if (h.materialized_seq < 0) {
                return false; // arena full
            }
        }
        llama_memory_seq_rm(llama_get_memory(cfg_.ctx), h.materialized_seq, -1, -1);
        if (llama_state_seq_set_data_ext(cfg_.ctx, h.owned_state.data(), h.owned_state.size(), h.materialized_seq,
                                         LLAMA_STATE_SEQ_FLAGS_NONE) != h.owned_state.size()) {
            return false;
        }
        seq = h.materialized_seq;
        return true;
    }

    void unmaterialize(capture_handle & h) override {
        if (cfg_.ctx != nullptr && h.materialized_seq >= 0) {
            llama_memory_seq_rm(llama_get_memory(cfg_.ctx), h.materialized_seq, -1, -1);
        }
        if (h.materialized_seq >= 0 && cfg_.arena != nullptr) {
            cfg_.arena->free_seq(h.materialized_seq);
        }
        h.materialized_seq = -1;
    }

    void release(capture_handle & h) override {
        if (cfg_.ctx != nullptr && h.resident_seq >= 0) {
            llama_memory_seq_rm(llama_get_memory(cfg_.ctx), h.resident_seq, -1, -1);
        }
        unmaterialize(h);
        h = capture_handle{};
    }

    size_t bytes(const capture_handle & h) const override {
        return h.n_bytes;
    }

    void on_memory_epoch(uint64_t) override {
        // the owned bytes are independent of the memory generation; the registry refuses
        // epoch-stale captures and materialize loads them into a fresh arena sequence on demand
    }

  private:
    session_store_config cfg_;
};

// The clone backend: a metadata-only cell reference for the source's attention cells, plus an owned
// copy of the recurrent/hybrid part (a partial state, exactly like the engine's hybrid fork). The
// resident sequence shares the source's cells, so a capture pins them (they stay non-empty) until
// the reference is released. The clone never writes a cell; it only adds a sequence reference.
class seq_clone_store final : public session_store {
  public:
    explicit seq_clone_store(const session_store_config & cfg) : cfg_(cfg) {
    }

    bool capture(const capture_request & req, capture_handle & out) override {
        if (cfg_.ctx == nullptr || req.dst < 0) {
            return false;
        }
        llama_context *  ctx = cfg_.ctx;
        const llama_memory_t mem = llama_get_memory(ctx);
        const llama_seq_id    dst = req.dst;

        // A metadata-only cell reference: share the source's cells with the resident sequence. The
        // source is never modified, and a later clear of the source leaves the pinned cells intact.
        llama_memory_seq_rm(mem, dst, -1, -1);
        llama_memory_seq_cp(mem, req.src, dst, 0, -1);

        // The recurrent/hybrid part lives outside the KV cache: copy it from a partial state so the
        // reference owns it. A dense model has no partial state, so only the shared cells remain.
        const llama_model * model = llama_get_model(ctx);
        size_t              partial_bytes = 0;
        if (llama_model_is_recurrent(model) || llama_model_is_hybrid(model)) {
            const size_t psize = llama_state_seq_get_size_ext(ctx, req.src, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            if (psize != 0) {
                std::vector<uint8_t> buf(psize);
                if (llama_state_seq_get_data_ext(ctx, buf.data(), buf.size(), req.src, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != psize) {
                    llama_memory_seq_rm(mem, dst, -1, -1);
                    return false;
                }
                if (llama_state_seq_set_data_ext(ctx, buf.data(), buf.size(), dst, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != psize) {
                    llama_memory_seq_rm(mem, dst, -1, -1);
                    return false;
                }
                partial_bytes = psize;
            }
        }
        out.backend      = session_backend::clone;
        out.epoch        = 0;
        out.pos          = req.base_pos;
        out.resident_seq = dst;
        out.n_bytes      = partial_bytes; // shared cells hold no extra bytes; the recurrent copy does
        out.identity     = req.identity;
        out.locator.clear();
        return true;
    }

    bool materialize(capture_handle & h, llama_seq_id & seq) override {
        seq = h.resident_seq;
        return seq >= 0;
    }

    void release(capture_handle & h) override {
        const llama_seq_id arena_seq = h.resident_seq;
        h = capture_handle{};
        if (cfg_.ctx != nullptr && arena_seq >= 0) {
            llama_memory_seq_rm(llama_get_memory(cfg_.ctx), arena_seq, -1, -1);
        }
    }

    size_t bytes(const capture_handle & h) const override {
        return h.n_bytes;
    }

    void on_memory_epoch(uint64_t) override {
        // a whole-context load clears every cell; the registry refuses epoch-stale captures, and a
        // later release only removes a sequence reference from cells that may already be gone
    }

  private:
    session_store_config cfg_;
};

// The file backend: the turn state lives on disk, so a session does not occupy an arena sequence at
// capture (only the bytes). Materialize loads the file into an arena sequence allocated for the
// in-flight decision; unmaterialize/release free it again, so a retained turn is non-resident.
class file_state_store final : public session_store {
  public:
    explicit file_state_store(const session_store_config & cfg) : cfg_(cfg) {
    }

    bool capture(const capture_request & req, capture_handle & out) override {
        if (cfg_.ctx == nullptr || cfg_.file_dir.empty() || req.src < 0) {
            return false;
        }
        // Path hygiene: the filename derives from the first-class session handle (or, for an
        // implicit slot-keyed session, the source slot and the content hash). A client never
        // supplies a path.
        std::string stem = req.session_id;
        if (stem.empty()) {
            stem = "slot-" + std::to_string((long long) req.src) + "-" + req.identity.content_hash;
        }
        const std::string path = cfg_.file_dir + "session-" + stem + ".gguf";
        // The state bytes are the KV; the identity (content hash + turn + scope) travels in the
        // registry and the manifest, so the file needs no prompt tokens.
        if (llama_state_seq_save_file(cfg_.ctx, path.c_str(), req.src, nullptr, 0) == 0) {
            return false;
        }
        size_t n_bytes = 0;
        FILE * f = std::fopen(path.c_str(), "rb");
        if (f != nullptr) {
            if (std::fseek(f, 0, SEEK_END) == 0) {
                n_bytes = (size_t) std::ftell(f);
            }
            std::fclose(f);
        }
        if (n_bytes == 0) {
            std::remove(path.c_str());
            return false;
        }
        out.backend      = session_backend::file;
        out.epoch        = 0;
        out.pos          = req.base_pos;
        out.resident_seq = -1; // no arena sequence held at capture
        out.n_bytes      = n_bytes;
        out.identity     = req.identity;
        out.locator      = path;
        return true;
    }

    bool materialize(capture_handle & h, llama_seq_id & seq) override {
        if (cfg_.ctx == nullptr || h.locator.empty()) {
            return false;
        }
        // One arena sequence per in-flight materialization, so two materialized references never
        // collide; unmaterialize/release free it again.
        if (h.materialized_seq < 0) {
            h.materialized_seq = cfg_.arena != nullptr ? cfg_.arena->alloc() : -1;
            if (h.materialized_seq < 0) {
                return false; // arena full
            }
        }
        // Query the token count first (the two-pass load reports the count, then restores the
        // state); the capture wrote no prompt tokens, so the count is zero and the second pass
        // loads the KV into the sequence.
        size_t n_packed = 0;
        if (llama_state_seq_load_file(cfg_.ctx, h.locator.c_str(), h.materialized_seq, nullptr, 0, &n_packed) == 0) {
            return false;
        }
        std::vector<llama_token> toks(std::max<size_t>(1, n_packed));
        size_t                   n_read = 0;
        if (llama_state_seq_load_file(cfg_.ctx, h.locator.c_str(), h.materialized_seq, toks.data(), toks.size(), &n_read) == 0) {
            return false;
        }
        seq = h.materialized_seq;
        return true;
    }

    void unmaterialize(capture_handle & h) override {
        if (cfg_.ctx != nullptr && h.materialized_seq >= 0) {
            llama_memory_seq_rm(llama_get_memory(cfg_.ctx), h.materialized_seq, -1, -1);
        }
        if (h.materialized_seq >= 0 && cfg_.arena != nullptr) {
            cfg_.arena->free_seq(h.materialized_seq);
        }
        h.materialized_seq = -1;
    }

    void release(capture_handle & h) override {
        if (!h.locator.empty()) {
            std::remove(h.locator.c_str());
        }
        unmaterialize(h);
        h = capture_handle{};
    }

    size_t bytes(const capture_handle & h) const override {
        return h.n_bytes;
    }

    void on_memory_epoch(uint64_t) override {
        // the bytes on disk are independent of the memory generation; the registry refuses
        // epoch-stale captures, and the next materialize reloads the file into an arena sequence
    }

  private:
    session_store_config cfg_;
};

bool clone_backend_capable(const llama_model * model) {
    if (model == nullptr) {
        return false;
    }
    // A sliding-window cache evicts the oldest cells, so a cell-reference clone would decay
    // silently. Dense unified attention and recurrent/hybrid (whose recurrent part the clone copies
    // from a partial state) are capable.
    if (llama_model_n_swa(model) > 0) {
        return false;
    }
    return true;
}

std::unique_ptr<session_store> make_session_store(session_backend backend, const session_store_config & cfg) {
    switch (backend) {
        case session_backend::host:
            return std::make_unique<host_state_store>(cfg);
        case session_backend::clone:
            return cfg.ctx != nullptr && clone_backend_capable(llama_get_model(cfg.ctx))
                ? std::unique_ptr<session_store>(new seq_clone_store(cfg)) : nullptr;
        case session_backend::file:
            return !cfg.file_dir.empty()
                ? std::unique_ptr<session_store>(new file_state_store(cfg)) : nullptr;
    }
    return nullptr;
}

} // namespace llama_decision