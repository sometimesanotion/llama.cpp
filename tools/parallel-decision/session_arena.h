#pragma once

// Decision-owned arena for live-session snapshots: a pool of reserved sequences that hold
// owned copies of completed chat turns, so a decision about a slot survives the slot's KV
// being cleared and reused by cache_idle_slots. A snapshot serializes the slot's decoded
// state in the self-contained host format and restores it into a free arena sequence; a
// later decision forks the arena sequence instead of the live slot. One retained turn per
// slot, detected by the slot's decoded position at snapshot time.

#include "llama.h"

#include <string>
#include <vector>

namespace llama_decision {

// One retained turn: the owned copy of a chat slot's decoded state at the moment the first
// decision for that turn took its snapshot. `pos` is the next position captured (the base
// the readout continues from); the arena sequence holds the state bytes up to pos - 1.
struct session_snapshot {
    bool        active   = false;
    std::string turn;            // opaque client tag ("" when the client sent none)
    llama_seq_id arena_seq = -1; // sequence holding the restored owned copy
    llama_pos   pos       = -1;  // next position captured at snapshot time
    size_t      n_bytes   = 0;   // serialized state size (for cost accounting)
};

// The session arena: a decision-owned pool of reserved sequences. Snapshot serializes the
// source sequence's decoded state (llama_state_seq_get_data_ext, host format) and restores
// it into a free arena sequence (llama_state_seq_set_data_ext), so the arena owns its
// cells and the origin can be cleared and reused. One retained turn per slot; the arena is
// full when every sequence holds a snapshot, which the caller maps to a 422.
class session_arena {
  public:
    session_arena() = default;

    // Reserve arena sequences [base, base + n_arena) in the context, tracking one turn per
    // slot (slots indexed [0, n_slots)).
    session_arena(llama_context * ctx, llama_seq_id base, int n_arena, int n_slots);

    // The retained snapshot for a slot, or nullptr when the slot holds none.
    const session_snapshot * find(int slot_id) const;

    // Reuse the retained snapshot for a slot: returns the arena sequence id to fork (and
    // counts the reuse for the cost calibration), or -1 when the slot holds no snapshot.
    llama_seq_id reuse(int slot_id);

    // Serialize `src`'s decoded state (host format) and restore it into a free arena
    // sequence for `slot`, replacing any retained turn the slot already had. Returns the
    // arena sequence id, or -1 when the arena is full. The source is not modified.
    llama_seq_id snapshot(int slot_id, llama_seq_id src, llama_pos pos, const std::string & turn);

    // Free the arena sequence held by `slot` (the retained turn is discarded).
    void release(int slot_id);

    // Release every retained snapshot (model reset / shutdown).
    void reset();

    int capacity() const { return n_arena_; }

    // Exact trigger counters for the cost calibration: snapshots taken, arena sequences
    // reused, and snapshots released. These are the observable fire/advance counts the
    // cost heuristic is calibrated against; they never gate anything.
    int n_snapshots() const { return n_snapshots_; }
    int n_reuses() const    { return n_reuses_; }
    int n_releases() const  { return n_releases_; }

    // Whether a slot's retained snapshot is still the current turn: true when the slot's
    // current decoded next position (`pos`, -1 when the slot holds no decoded state)
    // still matches the snapshot's captured position. A cleared slot (-1) cannot have
    // advanced, so the snapshot is still current; a slot that decoded past the snapshot
    // has advanced to a new turn. Pure: reads no slot state itself.
    static bool snapshot_is_current(const session_snapshot * snap, llama_pos pos) {
        if (snap == nullptr || !snap->active) {
            return false;
        }
        return pos < 0 || pos == snap->pos;
    }

  private:
    llama_context * ctx_ = nullptr;
    llama_seq_id    base_ = -1;
    int             n_arena_ = 0;
    std::vector<bool> used_;          // per arena sequence: holds a snapshot
    std::vector<session_snapshot> slots_; // per slot id
    int n_snapshots_ = 0;
    int n_reuses_    = 0;
    int n_releases_  = 0;
};

} // namespace llama_decision