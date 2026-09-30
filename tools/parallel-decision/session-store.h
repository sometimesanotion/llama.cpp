#pragma once

// The session reference substrate: one interface for the reference backends that hold an owned copy
// of a completed chat turn, so a decision about a slot survives the slot's KV being cleared and
// reused by cache_idle_slots. A backend only knows how to capture, materialize, and release one
// reference; the session_registry is the single accounting/lifecycle authority.

#include "llama.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace llama_decision {

// The reference backends. `host` holds the self-contained host-format state bytes in RAM and
// restores them into a reserved arena sequence only for the duration of a decision, so a retained
// turn never competes with chat for context cells. `clone` is a metadata-only cell-reference
// backend for dense unified attention (plus a partial recurrent copy for recurrent/hybrid layers);
// `file` keeps the turn state on disk and loads it on demand.
enum class session_backend {
    clone,
    host,
    file,
};

// The stable name of a backend, for the session API and diagnostics.
inline const char * session_backend_name(session_backend backend) {
    switch (backend) {
        case session_backend::clone: return "clone";
        case session_backend::host:  return "host";
        case session_backend::file:  return "file";
    }
    return "unknown";
}

// Identity of a captured turn. `content_hash` is a hash of the captured token prefix (reuse the
// FNV-1a helper behind make_prefix_tag); `turn` is a monotonic per-session turn counter;
// `adapter_scope` names the active adapter set (empty = base model).
struct session_identity {
    std::string content_hash;   // FNV-1a of the captured token prefix
    uint64_t    turn = 0;       // monotonic per-session turn counter
    std::string adapter_scope;  // active adapter set identity; "" = base

    bool operator==(const session_identity & o) const {
        return content_hash == o.content_hash && turn == o.turn && adapter_scope == o.adapter_scope;
    }
    bool operator!=(const session_identity & o) const { return !(*this == o); }
};

// How a session is captured and kept. Defaults preserve today's behavior: host backend, capture on
// the first decision, not pinned, no TTL.
struct session_policy {
    session_backend backend                  = session_backend::host;
    bool            capture_on_turn_complete = false;
    bool            pinned                   = false;
    int64_t         ttl_ms                   = 0; // 0 = no expiry
    size_t          max_turns                = 0; // 0 = unlimited
};

// What a backend is asked to capture: the source sequence, the base position the readout continues
// from, the client turn tag, the computed identity, the first-class session handle, and the arena
// sequence the registry assigned for the capture to materialize into.
struct capture_request {
    llama_seq_id     src = -1;
    llama_pos        base_pos = -1;
    std::string      turn_tag;
    session_identity identity;
    std::string      session_id;   // first-class session handle; "" for an implicit slot-keyed session
    llama_seq_id     dst = -1;     // registry-assigned arena sequence (clone); -1 for host/file
};

// The result of a capture: enough for the registry to materialize and release the reference. The
// backend-specific state lives in `owned_state`, `resident_seq`, and `locator`.
struct capture_handle {
    session_backend  backend = session_backend::host;
    uint64_t         epoch = 0;          // memory epoch at capture; checked on resolve
    llama_pos        pos = -1;           // base position captured
    llama_seq_id     resident_seq = -1;  // clone: the sequence holding the reference; host/file: -1
    llama_seq_id     materialized_seq = -1; // host/file: the arena sequence loaded for the in-flight decision; clone: -1
    size_t           n_bytes = 0;        // bytes held (registry budget accounting)
    session_identity identity;
    std::string      locator;            // file backend: path; others: ""
    std::vector<uint8_t> owned_state;    // host: the self-contained state bytes; others: empty
};

// The shared reserved-sequence arena owned by the session registry. Every backend draws its
// resident/scratch sequences here, so host, clone, and file captures never collide on the same
// sequence id. alloc() returns -1 when the pool is exhausted (a capacity refusal, never a borrow).
class session_arena {
  public:
    virtual ~session_arena() = default;

    virtual llama_seq_id alloc() = 0;
    virtual void         free_seq(llama_seq_id seq) = 0;
    virtual size_t       capacity() const = 0;
    virtual size_t       used() const = 0;
};

// How a backend is configured: the shared arena and the writable directory for the file backend.
struct session_store_config {
    llama_context * ctx = nullptr;
    session_arena * arena = nullptr;    // shared reserved-sequence allocator (registry-owned)
    std::string     file_dir;           // writable directory for the file backend; "" = unsupported
};

// The reference backend interface. A backend knows one reference at a time: how to capture it from
// a source sequence, materialize it into a forkable sequence, release it, and report its size.
class session_store {
  public:
    virtual ~session_store() = default;

    // Capture `req.src`'s decoded state into the registry-assigned sequence `req.dst` (host/clone)
    // or into a file under the configured directory (file). Returns true on success; false when the
    // store can hold no further reference (full) or the source has no state to capture. Never
    // modifies the source.
    virtual bool capture(const capture_request & req, capture_handle & out) = 0;

    // Materialize a captured reference: `seq` receives a forkable sequence id, allocated from the
    // shared arena and recorded in the handle. Returns false when the reference can no longer be
    // materialized (arena full, bytes gone).
    virtual bool materialize(capture_handle & h, llama_seq_id & seq) = 0;

    // Drop the materialized cells after a decision, so a non-resident backend holds no context
    // memory between decisions. The reference itself stays captured and can be materialized again.
    // A backend whose cells are shared by definition (clone) or held by the store (file) may treat
    // this as a no-op.
    virtual void unmaterialize(capture_handle & h) {
        (void) h;
    }

    // Release a captured reference, freeing its owned resources.
    virtual void release(capture_handle & h) = 0;

    // The bytes a captured reference holds (registry budget accounting).
    virtual size_t bytes(const capture_handle & h) const = 0;

    // The memory epoch changed: drop or re-derive references (an epoch-bumped capture is stale).
    virtual void on_memory_epoch(uint64_t epoch) = 0;
};

// The reserved-sequence arena over [base, base + n_arena): the pool every backend draws its
// resident/scratch sequences from.
std::unique_ptr<session_arena> make_session_arena(llama_seq_id base, int n_arena);

// True when the clone backend is capable on this model: dense unified attention, or recurrent/hybrid
// (whose recurrent part the clone copies from a partial state). A sliding-window cache is never
// capable: it evicts the oldest cells, so a cell-reference clone would decay silently. Pure: reads
// only the model layout, never a producer concentration score.
bool clone_backend_capable(const llama_model * model);

// Creates the store instance for a selectable backend, or nullptr when the backend is not selectable
// on this model/context/configuration (clone: not capable; file: no writable directory). The
// registry treats a nullptr as a capability refusal.
std::unique_ptr<session_store> make_session_store(session_backend backend, const session_store_config & cfg);

} // namespace llama_decision