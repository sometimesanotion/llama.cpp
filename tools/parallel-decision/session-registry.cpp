#include "session-registry.h"

#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>

namespace llama_decision {

static int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

// The manifest sidecar embeds the one host-state serializer's bytes for every backend, so a
// restore rebuilds the reference without the source slot (or, for file, the original capture
// being released). These helpers read and write the raw bytes a sidecar carries.
static std::vector<uint8_t> read_binary_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return {};
    }
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

static bool write_binary_file(const std::string & path, const std::vector<uint8_t> & bytes) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }
    out.write((const char *) bytes.data(), (std::streamsize) bytes.size());
    return (bool) out;
}

session_registry::session_registry(llama_context * ctx, llama_seq_id base_seq, int n_arena, int n_slots, uint64_t epoch,
                                   const std::string & file_dir, size_t budget_bytes)
    : ctx_(ctx), n_slots_(n_slots),
      arena_(make_session_arena(base_seq, n_arena)), file_dir_(file_dir), budget_bytes_(budget_bytes),
      slots_(n_slots), slot_turns_(n_slots, 0), epoch_(epoch) {
    session_id_seed_ = (uint64_t) now_ms();
}

std::string session_registry::content_hash_of(const std::vector<llama_token> & prefix) {
    std::string bytes;
    bytes.reserve(prefix.size() * sizeof(llama_token));
    for (llama_token tok : prefix) {
        const uint32_t t = (uint32_t) tok;
        for (int b = 0; b < 4; ++b) {
            bytes.push_back((char) ((t >> (8 * b)) & 0xff));
        }
    }
    const uint64_t h = fnv1a64(bytes);
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%016llx", (unsigned long long) h);
    return std::string("session-content-v1:") + buf;
}

std::string session_registry::blob_hash_of(const std::string & blob) {
    const uint64_t h = fnv1a64(blob);
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%016llx", (unsigned long long) h);
    return std::string("window-blob-v1:") + buf;
}

session_manifest_data session_registry::serialize(int id_slot, const std::string & bound_hash) const {
    session_manifest_data out;
    if (id_slot < 0 || id_slot >= n_slots_ || slots_[id_slot] == nullptr) {
        return out;
    }
    const decision_session * sess = slots_[id_slot].get();

    // A host or clone capture persists its state through the one host-state serializer: the
    // resident sequence's self-contained bytes, so a restore can re-materialize the turn without
    // the source slot. A file capture embeds its on-disk bytes the same way, so the sidecar is
    // self-contained and a restore rewrites the file when the original capture is released.
    size_t n_bytes = sess->capture.n_bytes;
    if (sess->captured) {
        if (sess->policy.backend == session_backend::file) {
            if (sess->capture.locator.empty()) {
                return session_manifest_data{};
            }
            out.state = read_binary_file(sess->capture.locator);
            if (out.state.empty()) {
                return session_manifest_data{};
            }
            n_bytes = out.state.size();
        } else if (!sess->capture.owned_state.empty()) {
            // the host backend holds the self-contained bytes directly
            out.state = sess->capture.owned_state;
            n_bytes   = out.state.size();
        } else {
            if (sess->capture.resident_seq < 0) {
                return session_manifest_data{};
            }
            const size_t size = llama_state_seq_get_size_ext(ctx_, sess->capture.resident_seq, LLAMA_STATE_SEQ_FLAGS_NONE);
            if (size == 0) {
                return session_manifest_data{};
            }
            out.state.resize(size);
            if (llama_state_seq_get_data_ext(ctx_, out.state.data(), out.state.size(),
                                             sess->capture.resident_seq, LLAMA_STATE_SEQ_FLAGS_NONE) != size) {
                return session_manifest_data{};
            }
            n_bytes = size;
        }
    }

    common_json s = common_json::object();
    s["id"]      = sess->id;
    s["turn"]    = sess->turn;
    s["pos"]     = (long long) sess->pos;
    s["backend"] = session_backend_name(sess->policy.backend);
    s["captured"] = sess->captured;
    s["n_bytes"] = (long long) n_bytes;
    if (sess->policy.backend == session_backend::file) {
        s["locator"] = sess->capture.locator;
    }
    common_json identity = common_json::object();
    identity["content_hash"]  = sess->identity.content_hash;
    identity["turn"]          = (long long) sess->identity.turn;
    identity["adapter_scope"] = sess->identity.adapter_scope;
    s["identity"] = identity;
    common_json policy = common_json::object();
    policy["backend"]                  = session_backend_name(sess->policy.backend);
    policy["capture_on_turn_complete"] = sess->policy.capture_on_turn_complete;
    policy["pinned"]                   = sess->policy.pinned;
    policy["ttl_ms"]                   = (long long) sess->policy.ttl_ms;
    policy["max_turns"]                = (long long) sess->policy.max_turns;
    s["policy"] = policy;

    common_json manifest = common_json::object();
    manifest["version"]    = 1;
    manifest["bound_hash"] = bound_hash;
    manifest["epoch"]      = (long long) sess->capture.epoch;
    manifest["id_slot"]    = id_slot;
    manifest["session"]    = s;
    out.manifest = manifest.dump();
    return out;
}

static session_backend session_backend_parse(const std::string & name) {
    if (name == "clone") {
        return session_backend::clone;
    }
    if (name == "file") {
        return session_backend::file;
    }
    return session_backend::host;
}

session_restore_status session_registry::deserialize(int id_slot, const session_manifest_data & m,
                                                     const std::vector<llama_token> & slot_prefix,
                                                     const std::string & bound_hash) {
    if (id_slot < 0 || id_slot >= n_slots_) {
        return session_restore_status::unresolvable;
    }
    if (m.manifest.empty()) {
        return session_restore_status::absent;
    }
    common_json manifest;
    try {
        manifest = common_json::parse(m.manifest);
    } catch (const common_json_error &) {
        // a malformed sidecar is never trusted: drop any capture and refuse
        drop_session(id_slot);
        return session_restore_status::unresolvable;
    }
    // The manifest is bound to the chat-window blob it was saved next to; a different blob, or a
    // manifest meant for a different slot, is a different window and is never served.
    if (manifest.value("bound_hash", std::string()) != bound_hash ||
        manifest.value("id_slot", -1) != id_slot) {
        drop_session(id_slot);
        return session_restore_status::unresolvable;
    }
    const common_json s = manifest.value("session", common_json::object());
    const common_json identity_json = s.value("identity", common_json::object());
    session_identity identity;
    identity.content_hash  = identity_json.value("content_hash", std::string());
    identity.turn          = (uint64_t) identity_json.value("turn", (long long) 0);
    identity.adapter_scope = identity_json.value("adapter_scope", std::string());
    // The per-session content hash is the restored slot's token identity: a window whose tokens
    // differ from the captured turn is unresolvable, never answered from the old turn.
    if (!identity.content_hash.empty() && identity.content_hash != content_hash_of(slot_prefix)) {
        drop_session(id_slot);
        return session_restore_status::unresolvable;
    }

    const session_backend backend = session_backend_parse(s.value("backend", std::string("host")));
    const bool             want_captured = s.value("captured", false);
    const std::string      file_locator  = backend == session_backend::file ? s.value("locator", std::string()) : std::string();
    // A reference that cannot be rebuilt (no state bytes, no on-disk file) is unresolvable. It is
    // checked before the pre-restore capture is released, so a broken manifest never destroys a
    // still-valid reference silently.
    if (want_captured && (m.state.empty() || (backend == session_backend::file && file_locator.empty()))) {
        drop_session(id_slot);
        return session_restore_status::unresolvable;
    }

    // A manifest restore does not advance the slot's turn: the restored window is the very turn the
    // manifest saved, so the slot's turn counter is re-synced to the manifest's recorded turn and
    // any pre-restore capture for the slot is released without counting a turn completion.
    release_slot_session(id_slot);
    slot_turns_[id_slot] = identity.turn;
    auto created = std::make_unique<decision_session>();
    created->id       = s.value("id", std::string());
    created->id_slot  = id_slot;
    created->turn     = s.value("turn", std::string());
    created->pos      = (llama_pos) s.value("pos", (long long) -1);
    created->created_ms  = now_ms();
    created->last_used_ms = created->created_ms;
    created->identity = identity;
    const common_json policy_json = s.value("policy", common_json::object());
    created->policy.backend                  = backend;
    created->policy.capture_on_turn_complete = policy_json.value("capture_on_turn_complete", false);
    created->policy.pinned                   = policy_json.value("pinned", false);
    created->policy.ttl_ms                   = policy_json.value("ttl_ms", (long long) 0);
    created->policy.max_turns                = (size_t) policy_json.value("max_turns", (long long) 0);
    created->capture.epoch = epoch_; // a restored reference is bound to the current memory generation

    if (want_captured) {
        if (backend == session_backend::file) {
            // The file backend's on-disk state is the one serializer's bytes (the save-file
            // payload); the release above removed the original capture's file, so the embedded
            // bytes rewrite it. The reference then materializes exactly like a fresh file capture.
            if (!write_binary_file(file_locator, m.state)) {
                return session_restore_status::unresolvable;
            }
            created->capture.backend      = session_backend::file;
            created->capture.pos          = created->pos;
            created->capture.n_bytes      = m.state.size();
            created->capture.identity     = identity;
            created->capture.locator      = file_locator;
            created->captured             = true;
        } else if (backend == session_backend::host) {
            // The host backend holds the manifest's host-format bytes in RAM; it loads them into
            // its scratch only while a decision is served, so the restore costs no context cells.
            created->capture.backend      = session_backend::host;
            created->capture.pos          = created->pos;
            created->capture.n_bytes      = m.state.size();
            created->capture.owned_state  = m.state;
            created->capture.identity     = identity;
            created->captured             = true;
        } else {
            // clone re-derives its metadata reference into a resident arena sequence
            const llama_seq_id dst = arena_->alloc();
            if (dst < 0) {
                return session_restore_status::unresolvable; // the arena is full; nothing is half-bound
            }
            if (llama_state_seq_set_data_ext(ctx_, m.state.data(), m.state.size(), dst,
                                             LLAMA_STATE_SEQ_FLAGS_NONE) != m.state.size()) {
                arena_->free_seq(dst);
                return session_restore_status::unresolvable;
            }
            created->capture.backend      = session_backend::clone;
            created->capture.pos          = created->pos;
            created->capture.resident_seq = dst;
            created->capture.n_bytes      = m.state.size();
            created->capture.identity     = identity;
            created->captured             = true;
        }
        bytes_ += created->capture.n_bytes;
    }
    slots_[id_slot] = std::move(created);
    if (!slots_[id_slot]->id.empty()) {
        session_slot_[slots_[id_slot]->id] = id_slot;
        ++n_sessions_;
    }
    if (slots_[id_slot]->captured) {
        ++n_snapshots_; // a manifest restore re-materializes the reference: one capture event
    }
    return session_restore_status::restored;
}

std::string session_registry::issue_session_id() {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "ses_%016llx", (unsigned long long) (session_id_seed_++));
    return buf;
}

session_store * session_registry::get_store(session_backend backend) {
    auto it = stores_.find(backend);
    if (it != stores_.end()) {
        return it->second.get();
    }
    session_store_config cfg;
    cfg.ctx         = ctx_;
    cfg.arena       = arena_.get();
    cfg.file_dir    = file_dir_;
    auto store = make_session_store(backend, cfg);
    if (store == nullptr) {
        return nullptr;
    }
    session_store * raw = store.get();
    stores_.emplace(backend, std::move(store));
    return raw;
}

bool session_registry::capture_now(decision_session * sess, const std::vector<llama_token> & prefix,
                                   const std::string & adapter_scope) {
    if (sess == nullptr || sess->id_slot < 0 || sess->id_slot >= n_slots_) {
        return false;
    }
    reap_expired();
    session_identity id;
    id.content_hash  = content_hash_of(prefix);
    id.turn          = slot_turns_[sess->id_slot];
    id.adapter_scope = adapter_scope;
    capture_request req;
    req.src        = (llama_seq_id) sess->id_slot;
    req.base_pos   = sess->pos;
    req.turn_tag   = sess->turn;
    req.identity   = id;
    req.session_id = sess->id;
    session_store * store = get_store(sess->policy.backend);
    if (store == nullptr) {
        return false;
    }
    // A clone capture materializes into a reserved arena sequence the registry assigns, so every
    // backend draws from one pool and none can collide. Host (RAM bytes) and file (disk bytes)
    // captures hold no arena sequence at capture; each store loads its own scratch only while a
    // decision is being served, so a retained turn is non-resident between decisions.
    if (sess->policy.backend == session_backend::clone) {
        req.dst = arena_->alloc();
        if (req.dst < 0) {
            return false; // arena full
        }
    }
    if (!store->capture(req, sess->capture)) {
        if (req.dst >= 0) {
            arena_->free_seq(req.dst);
        }
        return false;
    }
    // The budget is enforced on the actual capture bytes. Under pressure, the store evicts the
    // least-recently-used unpinned, unleased references (never the one being captured) to make
    // room, exactly like a cache; a capture that still does not fit is released at once, so a
    // partial allocation never persists.
    const size_t held = store->bytes(sess->capture);
    if (budget_bytes_ != 0 && bytes_ + held > budget_bytes_) {
        if (!evict_lru(sess, held)) {
            store->release(sess->capture);
            if (req.dst >= 0) {
                arena_->free_seq(req.dst);
            }
            sess->captured = false;
            return false;
        }
    }
    sess->capture.epoch = epoch_; // the registry stamps the capture epoch; the store only fills the bytes
    sess->identity = id;
    sess->captured = true;
    bytes_ += held;
    ++n_snapshots_;
    return true;
}

void session_registry::release_capture(decision_session * sess) {
    if (sess == nullptr || !sess->captured) {
        return;
    }
    session_store * store = get_store(sess->capture.backend);
    if (store != nullptr) {
        bytes_ -= store->bytes(sess->capture);
        const llama_seq_id resident = sess->capture.resident_seq;
        store->release(sess->capture);
        if (resident >= 0) {
            arena_->free_seq(resident);
        }
    }
    sess->captured = false;
    ++n_releases_;
}

void session_registry::release_slot_session(int id_slot) {
    if (id_slot < 0 || id_slot >= n_slots_ || slots_[id_slot] == nullptr) {
        return;
    }
    decision_session * sess = slots_[id_slot].get();
    release_capture(sess);
    if (!sess->id.empty()) {
        session_slot_.erase(sess->id);
        if (n_sessions_ > 0) {
            --n_sessions_;
        }
    }
    slots_[id_slot].reset();
}

void session_registry::drop_session(int id_slot) {
    release_slot_session(id_slot);
    ++slot_turns_[id_slot];
}

int session_registry::slot_of(const std::string & session_id) const {
    auto it = session_slot_.find(session_id);
    if (it == session_slot_.end()) {
        return -1;
    }
    return it->second;
}

bool session_registry::current_turn(const decision_session * sess, const std::vector<llama_token> & prefix,
                                    const std::string & adapter_scope) const {
    if (sess == nullptr) {
        return false;
    }
    session_identity cur;
    cur.content_hash  = content_hash_of(prefix);
    cur.turn          = slot_turns_[sess->id_slot];
    cur.adapter_scope = adapter_scope;
    return sess->identity == cur && sess->capture.epoch == epoch_;
}

static resolved_session materialize_session(llama_decision::session_store * store, decision_session * sess) {
    resolved_session out;
    llama_seq_id seq = -1;
    if (store == nullptr || !store->materialize(sess->capture, seq)) {
        throw semantic_error("the session cannot be materialized");
    }
    out.seq          = seq;
    out.pos          = sess->pos;
    out.turn         = sess->turn;
    out.session_id   = sess->id;
    return out;
}

resolved_session session_registry::resolve_slot(int id_slot, const std::vector<llama_token> & prefix, llama_pos pos_max,
                                                int session_pos, const std::string & turn,
                                                const std::string & adapter_scope) {
    if (id_slot < 0 || id_slot >= n_slots_) {
        throw std::invalid_argument("id_slot " + std::to_string(id_slot) + " is out of range [0, " +
                                    std::to_string(n_slots_) + ")");
    }
    reap_expired();
    decision_session * sess = slots_[id_slot].get();
    if (sess != nullptr) {
        // An opaque turn tag pins the session; a mismatch is a client error, never a silent answer
        // about a different turn.
        if (!turn.empty() && !sess->turn.empty() && turn != sess->turn) {
            throw semantic_error(
                "turn " + turn + " does not match the retained turn of id_slot " +
                std::to_string(id_slot) + " (" + sess->turn + ")");
        }
        // The single identity advance check: a captured turn is current only when its content hash,
        // turn, adapter scope, and memory epoch all match the current slot. A cleared slot (pos_max
        // < 0) has no content to compare, so the reference survives the origin being cleared.
        const bool content_ok = pos_max < 0 || sess->identity.content_hash == content_hash_of(prefix);
        const bool epoch_ok   = sess->capture.epoch == epoch_;
        const bool scope_ok   = sess->identity.adapter_scope == adapter_scope;
        if (!epoch_ok) {
            // A capture from a previous memory generation is stale, never a semantic error and never
            // answered from the old state. The route maps this to HTTP 409.
            throw stale_error(
                "the retained turn of id_slot " + std::to_string(id_slot) +
                " is stale: the memory epoch changed since capture; re-capture or create a new session");
        }
        if (!content_ok || !scope_ok) {
            // A pinned reference (a turn tag or a first-class session) is refused, never re-answered
            // from a different turn; a changed adapter scope is always refused, because the captured
            // state cannot represent the current scope. A plain id_slot with no pin follows today's
            // behavior: the turn advanced, so the stale reference is discarded and re-taken.
            const bool pinned = !turn.empty() || !sess->id.empty();
            if (pinned || !scope_ok) {
                throw semantic_error(
                    "the retained turn of id_slot " + std::to_string(id_slot) +
                    " is no longer current (the slot content or adapter scope changed); "
                    "re-capture or create a new session");
            }
            drop_session(id_slot);
            sess = nullptr;
        } else {
            sess->last_used_ms = now_ms();
        }
    }
    if (sess == nullptr) {
        // First decision for the current turn: take the owned reference now.
        if (pos_max < 0) {
            throw semantic_error(
                "id_slot " + std::to_string(id_slot) + " has no decoded state to fork");
        }
        const llama_pos pos = pos_max + 1;
        if (session_pos >= 0 && (llama_pos) session_pos != pos) {
            throw semantic_error(
                "session_pos " + std::to_string(session_pos) + " does not continue id_slot " +
                std::to_string(id_slot) + " (expected " + std::to_string(pos) + ")");
        }
        auto created = std::make_unique<decision_session>();
        created->id_slot  = id_slot;
        created->turn     = turn;
        created->policy   = session_policy{};
        created->pos      = pos;
        created->created_ms = now_ms();
        created->last_used_ms = created->created_ms;
        created->identity.content_hash  = content_hash_of(prefix);
        created->identity.turn          = slot_turns_[id_slot];
        created->identity.adapter_scope = adapter_scope;
        if (!capture_now(created.get(), prefix, adapter_scope)) {
            throw capacity_error(
                "the decision session arena is full; start the server with --decision-arena-seqs N");
        }
        sess = created.get();
        slots_[id_slot] = std::move(created);
    } else {
        // Same turn as the retained reference: fork the resident sequence, not the live slot. The
        // continuation is the captured position, so a pinned session_pos must match it exactly.
        if (session_pos >= 0 && (llama_pos) session_pos != sess->pos) {
            throw semantic_error(
                "session_pos " + std::to_string(session_pos) + " does not continue id_slot " +
                std::to_string(id_slot) + " (expected " + std::to_string(sess->pos) + ")");
        }
        ++n_reuses_;
    }
    session_store * store = get_store(sess->capture.backend);
    return materialize_session(store, sess);
}

std::string session_registry::create(const session_create_request & req) {
    if (req.id_slot < 0 || req.id_slot >= n_slots_) {
        throw std::invalid_argument("id_slot " + std::to_string(req.id_slot) + " is out of range [0, " +
                                    std::to_string(n_slots_) + ")");
    }
    reap_expired();
    // A backend that is not selectable on this model/context is a capability refusal, not a budget
    // one; clone is gated on the model layout and file on a writable directory.
    if (get_store(req.policy.backend) == nullptr) {
        throw unsupported_error(
            "the requested decision session backend is not supported on this model/context");
    }
    if (req.prefix.empty()) {
        throw semantic_error("id_slot " + std::to_string(req.id_slot) + " has no decoded state to capture");
    }
    // One retained turn per slot: creating a session replaces any reference the slot already holds.
    drop_session(req.id_slot);
    auto created = std::make_unique<decision_session>();
    created->id       = issue_session_id();
    created->id_slot  = req.id_slot;
    created->turn     = req.turn;
    created->policy   = req.policy;
    created->pos      = (llama_pos) req.prefix.size();
    created->created_ms = now_ms();
    created->last_used_ms = created->created_ms;
    created->identity.content_hash  = content_hash_of(req.prefix);
    created->identity.turn          = slot_turns_[req.id_slot];
    created->identity.adapter_scope = req.adapter_scope;
    created->capture.epoch          = epoch_; // a lazy session is bound to the epoch it was created under
    if (req.policy.capture_on_turn_complete) {
        // The turn is already complete at create: an eager session captures its state now.
        if (!capture_now(created.get(), req.prefix, req.adapter_scope)) {
            throw capacity_error(
                "the decision session arena is full; start the server with --decision-arena-seqs N");
        }
    }
    const std::string id = created->id;
    slots_[req.id_slot] = std::move(created);
    session_slot_[id]   = req.id_slot;
    ++n_sessions_;
    return id;
}

const decision_session * session_registry::find(const std::string & session_id) const {
    const int slot = slot_of(session_id);
    if (slot < 0 || slot >= n_slots_ || slots_[slot] == nullptr || slots_[slot]->id != session_id) {
        return nullptr;
    }
    return slots_[slot].get();
}

resolved_session session_registry::resolve(const std::string & session_id, const std::vector<llama_token> & prefix,
                                           const std::string & adapter_scope) {
    reap_expired();
    const int slot = slot_of(session_id);
    if (slot < 0) {
        throw std::invalid_argument("session " + session_id + " does not exist");
    }
    decision_session * sess = slots_[slot].get();
    if (sess == nullptr || sess->id != session_id) {
        throw std::invalid_argument("session " + session_id + " does not exist");
    }
    sess->last_used_ms = now_ms();
    // The session is bound to the epoch it was created/captured under; a later memory generation is
    // stale (409), never a semantic error and never answered from the old state.
    if (sess->capture.epoch != epoch_) {
        throw stale_error(
            "session " + session_id + " is stale: the memory epoch changed since capture; "
            "re-create the session");
    }
    if (!sess->captured) {
        // A lazy session captures on its first resolve: the source slot must still hold the turn.
        if (prefix.empty()) {
            throw semantic_error("session " + session_id + " has no decoded state to capture");
        }
        if (!current_turn(sess, prefix, adapter_scope)) {
            throw semantic_error(
                "session " + session_id + " is stale: the source slot content or adapter scope "
                "changed since capture");
        }
        if (!capture_now(sess, prefix, adapter_scope)) {
            throw capacity_error(
                "the decision session arena is full; start the server with --decision-arena-seqs N");
        }
    } else {
        // The owned reference survives a cache_idle_slots clear of its slot: an empty prefix has no
        // content to compare. A slot that holds different content is a different turn: refuse.
        if (!prefix.empty() && !current_turn(sess, prefix, adapter_scope)) {
            throw semantic_error(
                "session " + session_id + " is stale: the source slot content or adapter scope "
                "changed since capture");
        }
    }
    session_store * store = get_store(sess->capture.backend);
    return materialize_session(store, sess);
}

bool session_registry::erase(const std::string & session_id) {
    const int slot = slot_of(session_id);
    if (slot < 0) {
        return false;
    }
    decision_session * sess = slots_[slot].get();
    if (sess == nullptr || sess->id != session_id) {
        return false;
    }
    drop_session(slot);
    return true;
}

bool session_registry::patch(const std::string & session_id, bool set_pinned, bool pinned,
                             bool set_ttl, int64_t ttl_ms) {
    const int slot = slot_of(session_id);
    if (slot < 0) {
        return false;
    }
    decision_session * sess = slots_[slot].get();
    if (sess == nullptr || sess->id != session_id) {
        return false;
    }
    if (set_pinned) {
        sess->policy.pinned = pinned;
    }
    if (set_ttl) {
        sess->policy.ttl_ms = ttl_ms;
    }
    return true;
}

void session_registry::on_turn_complete(int id_slot, const std::vector<llama_token> & prefix,
                                        const std::string & adapter_scope) {
    if (id_slot < 0 || id_slot >= n_slots_ || slots_[id_slot] == nullptr) {
        return;
    }
    decision_session * sess = slots_[id_slot].get();
    if (sess->policy.capture_on_turn_complete && !sess->captured) {
        // An eager session captures the state of the turn that just completed.
        capture_now(sess, prefix, adapter_scope);
        return;
    }
    // The retained session's turn is over.
    drop_session(id_slot);
}

void session_registry::on_slot_release(int id_slot) {
    drop_session(id_slot);
}

int session_registry::reap_expired() {
    const int64_t now = now_ms();
    int           n  = 0;
    for (int slot = 0; slot < n_slots_; ++slot) {
        decision_session * sess = slots_[slot].get();
        // The reaper never touches a session younger than its TTL, never a pinned session, and
        // never one held by an in-flight decision.
        if (sess == nullptr || sess->policy.pinned || sess->lease_count > 0 || sess->policy.ttl_ms <= 0) {
            continue;
        }
        if (now - sess->last_used_ms > sess->policy.ttl_ms) {
            release_slot_session(slot);
            ++n_ttl_reaps_;
            ++n;
        }
    }
    return n;
}

bool session_registry::lease(int id_slot) {
    if (id_slot < 0 || id_slot >= n_slots_ || slots_[id_slot] == nullptr) {
        return false;
    }
    ++slots_[id_slot]->lease_count;
    return true;
}

void session_registry::unlease(int id_slot) {
    if (id_slot < 0 || id_slot >= n_slots_ || slots_[id_slot] == nullptr) {
        return;
    }
    decision_session * sess = slots_[id_slot].get();
    if (sess->lease_count > 0) {
        --sess->lease_count;
    }
    // the last lease dropped: a non-resident backend frees the cells it loaded for the decision,
    // so a retained turn holds no context memory between decisions
    if (sess->lease_count == 0 && sess->captured) {
        session_store * store = get_store(sess->capture.backend);
        if (store != nullptr) {
            store->unmaterialize(sess->capture);
        }
    }
}

// Evict the least-recently-used unpinned, unleased references (never `keep`, never a pinned or
// leased session) until a capture of `needed_bytes` fits the byte budget. Returns true when the
// budget now fits, false when the pressure cannot be relieved (the caller then refuses).
bool session_registry::evict_lru(decision_session * keep, size_t needed_bytes) {
    if (budget_bytes_ == 0) {
        return true; // unlimited budget
    }
    std::vector<int> evictable;
    for (int slot = 0; slot < n_slots_; ++slot) {
        decision_session * sess = slots_[slot].get();
        if (sess == nullptr || sess == keep || sess->policy.pinned || sess->lease_count > 0) {
            continue;
        }
        evictable.push_back(slot);
    }
    std::sort(evictable.begin(), evictable.end(), [&](int a, int b) {
        return slots_[a]->last_used_ms < slots_[b]->last_used_ms;
    });
    for (int slot : evictable) {
        if (bytes_ + needed_bytes <= budget_bytes_) {
            break;
        }
        release_slot_session(slot);
        ++n_evictions_;
    }
    return bytes_ + needed_bytes <= budget_bytes_;
}

void session_registry::on_memory_epoch(uint64_t epoch) {
    epoch_ = epoch;
    for (auto & kv : stores_) {
        kv.second->on_memory_epoch(epoch);
    }
}

const decision_session * session_registry::find_by_slot(int id_slot) const {
    if (id_slot < 0 || id_slot >= n_slots_) {
        return nullptr;
    }
    return slots_[id_slot].get();
}

} // namespace llama_decision