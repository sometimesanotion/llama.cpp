#pragma once

// Owns every piece of /v1/decision state that is tied to one loaded model. The server context
// holds exactly one of these and calls reset() whenever the model is freed, so a sleep->wake
// reload cannot leave a stale vocab or contract pointing at dead memory.

#include "letter_readout.h"
#include "session-registry.h"

#include "common.h"
#include "llama.h"

#include <memory>
#include <string>
#include <vector>

struct server_decision_state {
    // One live adapter-scope predicate: the decision decode answers for the base model, and
    // the registry the server applies to chat batches is the single source of truth. Entries
    // with a zero scale are registered but disabled, matching the enabled-adapter convention
    // used by the slot scheduler.
    bool adapters_configured(const std::vector<common_adapter_lora_info> & loras) const {
        for (const auto & lora : loras) {
            if (lora.scale > 0.0f) {
                return true;
            }
        }
        return false;
    }

    // One engine on the shared full-logits context (seq ids above the chat slots).
    std::unique_ptr<llama_decision::engine> decision_letter_engine;
    std::unique_ptr<llama_decision::label_vocab> decision_label_vocab; // letter readout, built once per model
    std::vector<llama_decision::label>      decision_labels;
    std::string                             decision_label_error; // set when the vocabulary probe fails
    std::string     decision_contract;    // identity of the decision readout contract
    bool            decision_temp_loaded = false;
    llama_decision::temperature_profile   decision_temp_profile;

    // Retained-turn registry for live-session decisions: one owned reference per chat slot, so a
    // decision about a slot survives the slot's KV being cleared by cache_idle_slots. The registry
    // sequences sit above the engine's pool; it is created on first use and reset with the rest.
    std::unique_ptr<llama_decision::session_registry> decision_sessions;

    // The memory epoch: a monotonically increasing counter that identifies the current KV/memory
    // generation. Every retained-turn capture records the epoch it was taken under, and a resolve
    // refuses a capture from a different generation (HTTP 409 stale). It is bumped by reset()
    // (model free/reload) and by server_context::on_memory_invalidated() on a whole-context
    // replace/load; the registry is created lazily with the current value.
    uint64_t memory_epoch = 0;

    // Total reset: every member that is a function of the loaded model is dropped, so a fresh
    // model after a reload rebuilds the vocab, labels and contract from scratch. Any member added
    // here is covered by construction.
    void reset() {
        ++memory_epoch; // a reload is a new memory generation: every capture from the old one is stale
        decision_letter_engine.reset();
        decision_label_vocab.reset();
        decision_labels.clear();
        decision_label_error.clear();
        decision_contract.clear();
        decision_temp_loaded = false;
        decision_temp_profile = {};
        decision_sessions.reset();
    }

    // The whole context was replaced or loaded without a model free/reload (a whole-context
    // restore or memory clear): bump the memory epoch and tell the live registry, so every retained
    // capture from the previous generation is refused as stale (409) instead of served from old
    // state. A registry created after this call starts at the current epoch.
    void on_memory_invalidated() {
        ++memory_epoch;
        if (decision_sessions) {
            decision_sessions->on_memory_epoch(memory_epoch);
        }
    }
};
