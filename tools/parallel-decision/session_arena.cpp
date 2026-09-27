#include "session_arena.h"

namespace llama_decision {

session_arena::session_arena(llama_context * ctx, llama_seq_id base, int n_arena, int n_slots)
    : ctx_(ctx), base_(base), n_arena_(n_arena), used_(n_arena, false), slots_(n_slots) {
}

const session_snapshot * session_arena::find(int slot_id) const {
    if (slot_id < 0 || slot_id >= (int) slots_.size()) {
        return nullptr;
    }
    return slots_[slot_id].active ? &slots_[slot_id] : nullptr;
}

llama_seq_id session_arena::reuse(int slot_id) {
    const session_snapshot * snap = find(slot_id);
    if (snap == nullptr) {
        return -1;
    }
    ++n_reuses_;
    return snap->arena_seq;
}

llama_seq_id session_arena::snapshot(int slot_id, llama_seq_id src, llama_pos pos, const std::string & turn) {
    if (slot_id < 0 || slot_id >= (int) slots_.size()) {
        return -1;
    }
    release(slot_id);
    if (ctx_ == nullptr) {
        return -1;
    }
    llama_seq_id arena_seq = -1;
    for (int i = 0; i < n_arena_; ++i) {
        if (!used_[i]) {
            arena_seq = base_ + (llama_seq_id) i;
            used_[i]  = true;
            break;
        }
    }
    if (arena_seq < 0) {
        return -1; // arena full
    }
    // Host-format state is self-contained and survives the source being cleared.
    const size_t size = llama_state_seq_get_size_ext(ctx_, src, LLAMA_STATE_SEQ_FLAGS_NONE);
    if (size == 0) {
        used_[arena_seq - base_] = false;
        return -1;
    }
    std::vector<uint8_t> buf(size);
    if (llama_state_seq_get_data_ext(ctx_, buf.data(), buf.size(), src, LLAMA_STATE_SEQ_FLAGS_NONE) != size) {
        used_[arena_seq - base_] = false;
        return -1;
    }
    if (llama_state_seq_set_data_ext(ctx_, buf.data(), buf.size(), arena_seq, LLAMA_STATE_SEQ_FLAGS_NONE) != size) {
        used_[arena_seq - base_] = false;
        return -1;
    }
    slots_[slot_id] = { true, turn, arena_seq, pos, size };
    ++n_snapshots_;
    return arena_seq;
}

void session_arena::release(int slot_id) {
    if (slot_id < 0 || slot_id >= (int) slots_.size() || !slots_[slot_id].active) {
        return;
    }
    const llama_seq_id arena_seq = slots_[slot_id].arena_seq;
    slots_[slot_id] = session_snapshot{};
    if (ctx_ != nullptr && arena_seq >= base_ && arena_seq < base_ + n_arena_) {
        llama_memory_seq_rm(llama_get_memory(ctx_), arena_seq, -1, -1);
        used_[arena_seq - base_] = false;
    }
    ++n_releases_;
}

void session_arena::reset() {
    for (int slot = 0; slot < (int) slots_.size(); ++slot) {
        release(slot);
    }
    n_snapshots_ = 0;
    n_reuses_    = 0;
    n_releases_  = 0;
}

} // namespace llama_decision