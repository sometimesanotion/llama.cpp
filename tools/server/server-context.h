#pragma once

#include "server-http.h"
#include "server-task.h"
#include "server-queue.h"

#include "json.h"

#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <set>

struct server_context_impl; // private implementation

struct server_context_meta {
    std::string build_info;
    std::string model_name;
    std::set<std::string> model_aliases;
    std::set<std::string> model_tags;
    std::string model_path;
    bool has_mtmd;
    bool has_inp_image;
    bool has_inp_audio;
    bool has_inp_video;
    json json_ui_settings;
    int slot_n_ctx;
    enum llama_pooling_type pooling_type;

    // chat params
    server_chat_params & chat_params;
    std::map<std::string, bool> chat_template_caps;

    // tokens
    std::string bos_token_str;
    std::string eos_token_str;
    llama_token fim_pre_token;
    llama_token fim_sub_token;
    llama_token fim_mid_token;
    llama_token fim_pad_token;
    llama_token fim_rep_token;
    llama_token fim_sep_token;

    // sampling
    std::vector<llama_logit_bias> logit_bias_eog;

    // model meta
    enum llama_vocab_type model_vocab_type;
    int32_t model_vocab_n_tokens;
    int32_t model_n_ctx_train;
    int32_t model_n_embd_inp;
    uint64_t model_n_params;
    uint64_t model_size;
    std::string model_ftype;
};

enum server_state {
    SERVER_STATE_DOWNLOADING,
    SERVER_STATE_LOADING,
    SERVER_STATE_READY,
    SERVER_STATE_SLEEPING,
};

static std::string server_state_to_str(server_state state) {
    switch (state) {
        case SERVER_STATE_DOWNLOADING: return "downloading";
        case SERVER_STATE_LOADING:     return "loading";
        case SERVER_STATE_READY:       return "ready";
        case SERVER_STATE_SLEEPING:    return "sleeping";
        default: GGML_ASSERT(false && "invalid server_state");
    }
}

static server_state server_state_from_str(const std::string & str) {
    if (str == "downloading") return SERVER_STATE_DOWNLOADING;
    if (str == "loading")     return SERVER_STATE_LOADING;
    if (str == "ready")       return SERVER_STATE_READY;
    if (str == "sleeping")    return SERVER_STATE_SLEEPING;
    GGML_ASSERT(false && "invalid server_state string");
}

using server_state_callback_t = std::function<void(server_state, json /* payload */)>;

struct server_context {
    std::unique_ptr<server_context_impl> impl;

    server_context();
    ~server_context();

    // load the model and initialize llama_context
    // returns true on success
    // when `shared_model` is non-null, weights are borrowed (owned externally); only this
    // instance's context + compute buffers are created
    bool load_model(common_params & params, llama_model * shared_model = nullptr);

    // this function will block main thread until termination
    void start_loop();

    // terminate main loop (will unblock start_loop)
    void terminate();

    // get the underlaying llama_context, can return nullptr if sleeping
    // not thread-safe, should only be used from the main thread
    llama_context * get_llama_context() const;

    // get a new response reader, used by CLI application
    server_response_reader get_response_reader();

    // get server metadata (read-only), can only be called after load_model()
    // not thread-safe, should only be used from the main thread
    server_context_meta get_meta() const;

    // note: must be set before load_model() is called
    void set_state_callback(server_state_callback_t callback);

    // adjust the advertised model name / aliases after load (multi-instance mode);
    // not thread-safe, only used by the manager during setup
    void set_model_name(const std::string & name);
    void set_model_aliases(const std::set<std::string> & aliases);

    // race-free aggregate of this context's slot activity, published by the
    // scheduler thread (see server_context_stats). best-effort and possibly
    // stale; the instance's own queue is the authority and defers a request
    // when the race is lost.
    server_context_stats get_stats() const;

    // whole-window context size, cached at load (equals llama_n_ctx then).
    // race-free; prefer this over get_slot_n_ctx() * n_parallel, which can
    // disagree with llama_n_ctx after rounding/recapping.
    int32_t get_n_ctx() const;

    // manager-only: be notified on the scheduler thread whenever a slot becomes idle,
    // used to wake requests waiting for a free instance in a group
    void set_slot_release_callback(std::function<void(int /* id_slot */)> callback);

    // manager-only (sidecar executor): install the resolver the decision route calls with the
    // snapshot key embedded in a routed session body, to fetch the pool-owned token snapshot for
    // the in-flight request. the pool owns the store and the adapter refs; the snapshot is a
    // read-only owned copy that stays valid for the task.
    using decision_snapshot_resolver_fn = std::function<std::shared_ptr<server_decision_snapshot>(const std::string & key)>;
    void set_decision_snapshot_resolver(decision_snapshot_resolver_fn resolver);

    // manager-only: two-phase snapshot switching. slot_save_copy() copies the slot KV to
    // a host buffer on the scheduler thread (bounded GPU->host transfer, no file I/O); the
    // manager writes that buffer to disk on its pool I/O worker. slot_restore_apply() applies
    // a host buffer on the scheduler (bounded host->GPU transfer); the manager read the file on
    // its pool I/O worker. a busy slot fails with a retriable ERROR_TYPE_UNAVAILABLE result, and
    // a failed restore clears the slot to empty (never a partially-loaded KV). deadline_ms is the
    // KV-size-scaled compose deadline (ms since epoch); -1 waits forever. returns nullptr on timeout.
    server_task_result_ptr slot_save_copy(int id_slot, int64_t deadline_ms = -1);
    server_task_result_ptr slot_restore_apply(int id_slot, std::vector<uint8_t> buffer, llama_tokens tokens, int64_t deadline_ms = -1);

    // manager-only: copy a completed turn's tokens and enabled adapter scope off the scheduler
    // thread. the result is an owned copy (tokens + path/scale lora list, no pointers); the source
    // slot is read-only and its KV is never touched. a processing slot or one without decoded
    // state is an error result. same deadline contract as instance_op.
    server_task_result_ptr slot_decision_snapshot(int id_slot, int64_t deadline_ms = -1);

    // replace the instance's adapter set. the caller must hold the pool's
    // instance_drain_guard (no slot processing, no interleaving save/restore) and
    // pool ownership of every adapter in the list (each raw ptr must be a live
    // registry entry the caller holds a ref for, so the tensor cannot be freed
    // during the swap). runs the swap on the
    // scheduler thread and WAITS for it with a deadline; returns the task result
    // (null on timeout). post directly to this context's queue, never through
    // pool dispatch (the drain guard's removing flag would reject it).
    server_task_result_ptr set_lora_adapters(std::vector<common_adapter_lora_info> adapters,
                                             int64_t deadline_ms);

    // bounded read of the scheduler's installed adapter list, through the same
    // choke point as the writers. nullopt on timeout (not observed, distinct
    // from an observed empty set). no new task type: reuses GET_LORA.
    std::optional<std::vector<common_adapter_lora_info>> get_lora_adapters(int64_t deadline_ms);

    // manager-only: run `op` on this instance's scheduler thread, serialized with all
    // other tasks (context lifetime is not thread-safe). `op` returns the JSON payload
    // and throws with a message to signal an error; the caller gets a
    // server_task_result_instance (or an error result). deadline_ms bounds the
    // wait (-1 waits forever, preserving the legacy callers); null on timeout.
    server_task_result_ptr instance_op(const std::function<json()> & op, int64_t deadline_ms = -1);

    // manager-only: abort all in-flight slot tasks with an error result (used before
    // destroy/resize so no HTTP reader is left hanging). same deadline contract
    // as instance_op: teardown paths pass a compose budget so a stalled
    // scheduler answers 503 with the instance intact instead of wedging the
    // management plane (which is held across the call).
    server_task_result_ptr abort_slots(const std::string & reason, int64_t deadline_ms = -1);

    // manager-only reporting (read-only, best-effort across scheduler thread)
    int get_slot_n_ctx() const;
    size_t get_model_bytes() const;   // model bytes (identical for every instance; count once), 0 when not loaded
    size_t get_context_bytes() const; // KV bytes, 0 when not loaded
    size_t get_compute_bytes() const; // compute buffer bytes, 0 when not loaded

private:
    // single choke point for every manager->scheduler op: posts the task and waits on the
    // result; the should_stop predicate aborts the wait (e.g. on a deadline)
    server_task_result_ptr run_scheduler_task(server_task && task, const std::function<bool()> & should_stop);
};


// forward declarations
struct server_res_generator;

struct server_routes {
    server_routes(const common_params & params, server_context & ctx_server);

    void init_routes();

    // note: this is not thread-safe and can only when ctx_http.is_ready is false
    void update_meta(const server_context & ctx_server) {
        this->meta = std::make_unique<server_context_meta>(ctx_server.get_meta());
    }

    // handlers using lambda function, so that they can capture `this` without `std::bind`
    // they won't be called until ctx_http.is_ready is set to true
    server_http_context::handler_t get_health;
    server_http_context::handler_t get_metrics;
    server_http_context::handler_t get_slots;
    server_http_context::handler_t post_slots;
    server_http_context::handler_t get_props;
    server_http_context::handler_t post_props;
    server_http_context::handler_t post_infill;
    server_http_context::handler_t post_completions;
    server_http_context::handler_t post_completions_oai;
    server_http_context::handler_t post_chat_completions;
    server_http_context::handler_t post_chat_completions_tok;
    server_http_context::handler_t post_control;
    server_http_context::handler_t post_responses_oai;
    server_http_context::handler_t post_responses_tok_oai;
    server_http_context::handler_t post_transcriptions_oai;
    server_http_context::handler_t post_anthropic_messages;
    server_http_context::handler_t post_anthropic_count_tokens;
    server_http_context::handler_t post_apply_template;
    server_http_context::handler_t get_models;
    server_http_context::handler_t post_tokenize;
    server_http_context::handler_t post_detokenize;
    server_http_context::handler_t post_embeddings;
    server_http_context::handler_t post_embeddings_oai;
    server_http_context::handler_t post_rerank;
    server_http_context::handler_t post_decision;
    // /v1/systemone: the Jev contract only. /v1/decision is the superset and also serves the
    // generic `schema` shape; the two share one handler and differ only in the shapes accepted.
    server_http_context::handler_t post_systemone;
    server_http_context::handler_t post_session;
    server_http_context::handler_t get_session;
    server_http_context::handler_t delete_session;
    server_http_context::handler_t patch_session;
    server_http_context::handler_t get_lora_adapters;
    server_http_context::handler_t post_lora_adapters;

    // to be used in router mode
    json get_model_info() const;

private:
    std::unique_ptr<server_res_generator> handle_completions_impl(
            const server_http_req & req,
            server_task_type type,
            const json & data,
            const std::vector<raw_buffer> & files,
            task_response_type res_type);
    std::unique_ptr<server_res_generator> handle_slots_save(const server_http_req & req, int id_slot);
    std::unique_ptr<server_res_generator> handle_slots_restore(const server_http_req & req, int id_slot);
    std::unique_ptr<server_res_generator> handle_slots_erase(const server_http_req &, int id_slot);
    std::unique_ptr<server_res_generator> handle_embeddings_impl(const server_http_req & req, task_response_type res_type);
    std::unique_ptr<server_res_generator> handle_count_tokens(const llama_vocab * vocab, mtmd_context * mctx, const mtmd_helper_init_opt & init_opt, const server_http_req & req, task_response_type res_type);

    // using unique_ptr to allow late initialization of const
    std::unique_ptr<const server_context_meta> meta;

    const common_params & params;
    server_context_impl & ctx_server;

    server_queue & queue_tasks;
    server_result_queue<server_task_result_ptr> & queue_results;
    std::unique_ptr<server_res_generator> create_response(bool bypass_sleep = false);

    // decision admission: bound concurrent decision requests so a burst cannot pile up work
    std::atomic<int> decision_inflight{0};
    size_t           decision_max_body  = 2u * 1024u * 1024u;
    int              decision_max_queue = 4;

    // cached responses, to be used during sleep
    std::mutex     mutex_cache;
    json           cached_models  = nullptr;
    json           cached_props   = nullptr;
    server_metrics cached_metrics;
    // set when a scrape during sleep already reported the throughput buckets
    bool           should_reset_buckets = false;
    // call right before sleep to update the cached responses
    void update_cached_responses(bool is_sleeping);
};
