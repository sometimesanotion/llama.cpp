#pragma once

// The session registry: the single accounting/lifecycle authority for retained-turn references. It
// owns the store instances, the per-slot session records, the first-class session handles, the
// exact trigger counters, and the resolve paths that id_slot and session_id decisions route
// through. A backend only knows how to capture, materialize, and release one reference; the
// registry decides when a reference is taken, reused, released, or refused.

#include "decision-engine.h"
#include "decision-protocol.h"
#include "session-store.h"

#include "llama.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace llama_decision {

// One retained-turn session. `id` is the first-class session handle ("" while the session is
// implicitly slot-keyed and only reachable by id_slot). `identity` is the captured turn's
// identity (content hash + turn + adapter scope): the single currentness check. A session is
// bound to one source chat slot; when the slot's turn ends the session ends with it.
struct decision_session {
    std::string      id;          // first-class session handle; "" for an implicit slot-keyed session
    session_policy   policy;
    capture_handle   capture;
    session_identity identity;    // recorded at capture; the single currentness check
    int64_t          created_ms = 0;
    int64_t          last_used_ms = 0;
    std::string      turn;        // opaque client turn tag
    int              id_slot = -1; // the source chat slot
    llama_pos        pos = -1;    // the base position the readout continues from
    bool             captured = false; // the reference is materialized in the store
    int              lease_count = 0;  // >0: held by an in-flight decision; eviction skips it
};

// The outcome of a successful resolve: the forkable sequence, the base position the readout
// continues from, the retained turn tag, and the session handle.
struct resolved_session {
    llama_seq_id seq = -1;
    llama_pos    pos = -1;
    std::string  turn;
    std::string  session_id;
};

// A serialized session reference: the manifest JSON plus the host-format state bytes that back a
// host or clone capture. For a file capture `state` is empty and the manifest's locator names the
// on-disk state file. `serialize` produces it; the server co-writes it next to the chat-window blob
// and passes it back to `deserialize` on a restore.
struct session_manifest_data {
    std::string          manifest; // JSON: identity, policy, backend, locator, bound hash
    std::vector<uint8_t> state;    // host/clone: host-format state bytes; file: empty
};

// The outcome of restoring a saved session manifest.
enum class session_restore_status {
    restored,     // the session was rebuilt and can be resolved
    unresolvable, // the manifest exists but its bound hash, content hash, or slot id does not
                  // match the restored window; the capture is dropped and never served
    absent,       // no manifest for this slot
};

// What create() needs: the source slot, its decoded token prefix (for the content hash), the
// slot's active adapter scope, the client turn tag, and the capture/lifetime policy.
struct session_create_request {
    int            id_slot = -1;
    std::vector<llama_token> prefix;
    std::string    adapter_scope;
    std::string    turn;
    session_policy policy;
};

class session_registry {
  public:
    // `epoch` is the memory epoch this registry generation starts at (0 for a fresh model). It is
    // bumped by server_decision_state::reset and forwarded here by on_memory_epoch; captures record
    // the epoch they were taken under and resolve refuses on mismatch with stale_error (HTTP 409).
    // `file_dir` is a writable directory for the file backend ("" = unsupported); `budget_bytes` is
    // the byte budget for all retained references (0 = unlimited; a capture over it is refused).
    session_registry(llama_context * ctx, llama_seq_id base_seq, int n_arena, int n_slots, uint64_t epoch = 0,
                     const std::string & file_dir = "", size_t budget_bytes = 0);

    // Resolve the retained session for an id_slot decision, reproducing today's behavior: find or
    // capture the owned reference for the slot's current turn, discard a stale one, and return the
    // forkable sequence. `prefix` is the slot's decoded token prefix (used for the content-hash
    // identity check); `pos_max` is the slot's current decoded next position (or -1 when the slot
    // holds no decoded state); `adapter_scope` is the slot's active adapter set identity ("" =
    // base). `session_pos` is the caller-pinned continuation position (-1 when not pinned); `turn`
    // is the client's opaque turn tag. Throws semantic_error on a turn/session_pos mismatch, on a
    // content or adapter-scope identity mismatch for a pinned reference, and on no decoded state;
    // stale_error (HTTP 409) when the retained reference was captured under a different memory
    // epoch; capacity_error when the store is full.
    resolved_session resolve_slot(int id_slot, const std::vector<llama_token> & prefix, llama_pos pos_max,
                                  int session_pos, const std::string & turn,
                                  const std::string & adapter_scope);

    // Create a first-class session for a slot's completed turn and return its handle. The identity
    // is recorded from `req.prefix`; with capture_on_turn_complete the reference is captured now,
    // otherwise on the first resolve.
    std::string create(const session_create_request & req);

    // The first-class session for a handle, or nullptr when unknown.
    const decision_session * find(const std::string & session_id) const;

    // Resolve a first-class session: identity-check it against the slot's current prefix and
    // adapter scope, capture it on first use, and return the forkable sequence. A stale or unknown
    // session is refused, never answered from an old turn. Throws stale_error (HTTP 409) when the
    // session was created/captured under a different memory epoch; semantic_error on a content or
    // adapter-scope identity mismatch; capacity_error when the store is full.
    resolved_session resolve(const std::string & session_id, const std::vector<llama_token> & prefix,
                             const std::string & adapter_scope);

    // Erase a first-class session, releasing its reference. Returns false when unknown.
    bool erase(const std::string & session_id);

    // Patch a first-class session's lifetime policy: `set_pinned`/`set_ttl` select which fields
    // to write. Returns false when unknown.
    bool patch(const std::string & session_id, bool set_pinned, bool pinned, bool set_ttl, int64_t ttl_ms);

    // A chat turn on the slot completed: capture the eager (capture_on_turn_complete) sessions for
    // it, and drop any retained session whose turn is over.
    void on_turn_complete(int id_slot, const std::vector<llama_token> & prefix, const std::string & adapter_scope);

    // The slot released or erased its turn: drop its retained session.
    void on_slot_release(int id_slot);

    // The memory epoch changed: forward to every store so each drops or re-derives its references.
    void on_memory_epoch(uint64_t epoch);

    // Reap every expired session: a session whose ttl_ms is positive and whose age exceeds it is
    // dropped, never a pinned or leased session, and never a session younger than its TTL.
    // Returns the number reaped.
    int reap_expired();

    // Hold a lease on a slot's retained session: eviction and reaping skip it while an in-flight
    // decision materializes it. Returns false when the slot holds no session.
    bool lease(int id_slot);
    void unlease(int id_slot);

    // LRU eviction counters: how many references were evicted to fit a capture, and how many were
    // reaped by TTL.
    int n_evictions() const { return n_evictions_; }
    int n_ttl_reaps() const { return n_ttl_reaps_; }

    // The retained session for a slot, or nullptr when the slot holds none.
    const decision_session * find_by_slot(int id_slot) const;

    // Whether a slot's retained session is still the current turn: true when the session's identity
    // (content hash + turn + adapter scope) and memory epoch match the slot's current prefix. An
    // empty prefix (a cleared slot) has no content to compare, so only the epoch is checked. Pure:
    // reads no slot state.
    bool is_current(const decision_session * sess, const std::vector<llama_token> & prefix,
                    const std::string & adapter_scope) const;

    // The FNV-1a content hash of a decoded token prefix: the identity of the captured turn, so a
    // clear plus a same-length re-prefill of different content is refused instead of answered.
    static std::string content_hash_of(const std::vector<llama_token> & prefix);

    // The FNV-1a content hash of a chat-window blob (the per-slot state file bytes): the binding a
    // session manifest is bound to, so a restore whose blob differs is refused instead of answered.
    static std::string blob_hash_of(const std::string & blob);

    // Serialize the retained session for `id_slot` into a manifest bound to `bound_hash` (the
    // content hash of the chat-window blob the manifest will sit next to). `state` carries the
    // host-format state bytes that back a host or clone capture; a file capture keeps its locator
    // and `state` stays empty. Returns an empty manifest when the slot holds no session or a host
    // or clone capture holds no state to persist.
    session_manifest_data serialize(int id_slot, const std::string & bound_hash) const;

    // Restore a serialized session for `id_slot` after a chat-window restore. `slot_prefix` is the
    // token prefix the slot holds after the restore; `bound_hash` is the content hash of the
    // chat-window blob the manifest was saved next to. A manifest whose bound hash, content hash,
    // or slot id does not match is unresolvable: any capture for the slot is dropped and never
    // served. An empty manifest is absent.
    session_restore_status deserialize(int id_slot, const session_manifest_data & m,
                                       const std::vector<llama_token> & slot_prefix,
                                       const std::string & bound_hash);

    size_t n_sessions() const { return n_sessions_; }
    size_t bytes() const      { return bytes_; }
    int    capacity() const   { return arena_ != nullptr ? (int) arena_->capacity() : 0; }
    size_t arena_used() const { return arena_ != nullptr ? arena_->used() : 0; }
    int    n_snapshots() const { return n_snapshots_; }
    int    n_reuses() const    { return n_reuses_; }
    int    n_releases() const  { return n_releases_; }

  private:
    session_store * get_store(session_backend backend);
    void release_capture(decision_session * sess);
    bool capture_now(decision_session * sess, const std::vector<llama_token> & prefix, const std::string & adapter_scope);
    void drop_session(int id_slot);
    void release_slot_session(int id_slot);
    bool evict_lru(decision_session * keep, size_t needed_bytes);
    int  slot_of(const std::string & session_id) const;
    bool current_turn(const decision_session * sess, const std::vector<llama_token> & prefix, const std::string & adapter_scope) const;
    std::string issue_session_id();

    llama_context * ctx_ = nullptr;
    int             n_slots_ = 0;
    std::unique_ptr<session_arena> arena_; // the shared reserved-sequence pool (the single allocator)
    std::string     file_dir_;             // writable directory for the file backend; "" = unsupported
    size_t          budget_bytes_ = 0;     // byte budget for all retained references; 0 = unlimited
    std::map<session_backend, std::unique_ptr<session_store>> stores_;
    std::vector<std::unique_ptr<decision_session>> slots_; // per slot id
    std::vector<uint64_t>                         slot_turns_; // per slot monotonic turn counter
    std::map<std::string, int>                    session_slot_; // session handle -> slot id
    uint64_t  epoch_ = 0;      // memory epoch; captures record it and resolve refuses on mismatch
    int       n_snapshots_ = 0;
    int       n_reuses_    = 0;
    int       n_releases_  = 0;
    int       n_evictions_ = 0; // LRU evictions to fit a capture
    int       n_ttl_reaps_ = 0; // TTL-expired reaps
    size_t    n_sessions_  = 0;
    size_t    bytes_       = 0;
    uint64_t  session_id_seed_ = 0;
};

} // namespace llama_decision