#include "server-instances.h"

#include "decision-engine.h"
#include "decision-protocol.h"

#include "common.h"
#include "log.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <set>
#include <thread>

#define IST_INF(fmt, ...) LOG_INF("inst %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define IST_WRN(fmt, ...) LOG_WRN("inst %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define IST_ERR(fmt, ...) LOG_ERR("inst %12.*s: " fmt, 12, __func__, __VA_ARGS__)

// wall-clock ms for the decision session store's created/last-used stamps
static int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

// The number of resident warm-prefix slots for the decision sidecar, derived from
// --decision-warm-budget-mb. The engine owns the number and the derivation
// (engine::warm_slots_for_budget), so nothing here clamps it a second time.
static int decision_warm_slots(const llama_model * model, const common_params & params) {
    return llama_decision::engine::warm_slots_for_budget(llama_model_n_layer(model), llama_model_n_embd(model),
                                                         llama_model_n_head(model), llama_model_n_head_kv(model),
                                                         params.n_ctx, params.decision_warm_budget_mb,
                                                         params.cache_type_k, params.cache_type_v);
}

// size the per-slot snapshot locks: one heap-allocated mutex per slot so the vector
// survives resize() (std::mutex is not movable) and switching different slots of one
// instance do not serialize each other.
static void server_instance_size_slot_locks(std::vector<std::unique_ptr<std::mutex>> & locks, int n_parallel) {
    locks.clear();
    locks.reserve((size_t) n_parallel);
    for (int i = 0; i < n_parallel; ++i) {
        locks.push_back(std::make_unique<std::mutex>());
    }
}

// guardrail: a context window this large that nobody asked for explicitly almost
// always means the train-ctx default fired on a long-context model (e.g. a 9B model
// with 262144 train ctx silently reserving ~10 GiB of KV on a 21 GiB card). warn
// loudly, before the allocation, with the remedy. explicit sizes stay silent: the
// operator asked for those.
static constexpr int32_t HUGE_CTX_WARN_TOKENS = 32768;

bool server_should_warn_huge_implicit_ctx(int32_t effective_n_ctx, int32_t n_ctx_train) {
    return effective_n_ctx == 0 && n_ctx_train >= HUGE_CTX_WARN_TOKENS;
}

static void warn_if_huge_implicit_ctx(const llama_model * model, const common_params & effective, const char * instance_name) {
    if (model == nullptr) {
        return; // nothing to compare against: not the footgun
    }
    if (!server_should_warn_huge_implicit_ctx(effective.n_ctx, llama_model_n_ctx_train(model))) {
        return; // explicit size, or a train context too small to matter
    }
    const int32_t n_ctx_train = llama_model_n_ctx_train(model);
    // rough KV footprint for the message: layers * kv_heads * head_dim * K+V * cache
    // bytes. head_dim ~= n_embd / n_head holds for the GQA families this branch serves;
    // hybrid (SSM) state is extra, so this is a lower bound.
    const int64_t n_layer   = llama_model_n_layer(model);
    const int64_t n_head    = llama_model_n_head(model);
    const int64_t head_dim  = n_head > 0 ? llama_model_n_embd(model) / n_head : 0;
    const int64_t kv_elems  = (int64_t) n_ctx_train * n_layer * llama_model_n_head_kv(model) * head_dim * 2;
    const double  bytes_per = (double) ggml_type_size(effective.cache_type_k) / (double) ggml_blck_size(effective.cache_type_k);
    IST_WRN("instance '%s' has no explicit context size and inherits the full train context (%d tokens, KV cache alone ~%.1f GiB): "
            "pass --ctx-size or instance ctx= to avoid filling device memory\n",
            instance_name, n_ctx_train, (double) kv_elems * bytes_per / 1073741824.0);
}

// read-back agreement for an adapter-set switch: the scheduler's committed set
// must equal the manager's expected list. on drift the expected set is posted
// once more; false = still diverged or a bound expired (caller answers
// retriable). fingerprints are computed on demand, so a successful re-apply
// makes every later fingerprint observe the corrected set.
static bool verify_attached(server_instance & inst, int64_t deadline_ms) {
    // effective.lora_adapters: callers hold mutex_mgmt and the instance_drain_guard
    auto installed = inst.ctx_server->get_lora_adapters(deadline_ms);
    if (installed && are_lora_sets_identical(*installed, inst.effective.lora_adapters)) {
        return true;
    }
    auto result = inst.ctx_server->set_lora_adapters(inst.effective.lora_adapters, deadline_ms);
    if (!result || result->is_error()) {
        return false;
    }
    installed = inst.ctx_server->get_lora_adapters(deadline_ms);
    return installed && are_lora_sets_identical(*installed, inst.effective.lora_adapters);
}

// RAII exclusive access for destroy/resize: set removing = true, wait for in-flight
// dispatches to drain, then restore the flag on scope exit.
struct instance_drain_guard {
    server_instances &              mgr;
    std::shared_ptr<server_instance> inst;

    instance_drain_guard(server_instances & m, std::shared_ptr<server_instance> i)
        : mgr(m), inst(std::move(i)) {
        std::unique_lock<std::mutex> lock(mgr.mutex_dispatch);
        inst->removing = true;
        mgr.cond_dispatch.notify_all();
        mgr.cond_dispatch.wait(lock, [&]() { return inst->n_active_dispatch == 0; });
    }

    ~instance_drain_guard() {
        std::lock_guard<std::mutex> lock(mgr.mutex_dispatch);
        inst->removing = false;
        mgr.cond_dispatch.notify_all();
    }
};

// RAII count of a direct route call on an instance (aggregate / management handlers
// that address an instance without going through dispatch()). destroy/resize drain on
// n_active_dispatch, so the guard keeps the scheduler alive until the route call
// finishes. the check-and-increment is atomic with the `running` flip in destroy, so a
// stale shared_ptr is rejected (acquired == false) instead of posting to a dead queue.
struct active_route_guard {
    server_instances & mgr;
    server_instance &  inst;
    bool               acquired = false;

    active_route_guard(server_instances & m, server_instance & i) : mgr(m), inst(i) {
        std::lock_guard<std::mutex> lock(mgr.mutex_dispatch);
        if (inst.removing || !inst.running) {
            return;
        }
        inst.n_active_dispatch++;
        acquired = true;
    }

    ~active_route_guard() {
        if (acquired) {
            std::lock_guard<std::mutex> lock(mgr.mutex_dispatch);
            inst.n_active_dispatch--;
            mgr.cond_dispatch.notify_all();
        }
    }
};

bool server_instances::load(const common_params & params) {
    this->params = params;

    // fail fast on duplicate names or name/group collisions accumulated
    // across repeated flags or built programmatically: registration below
    // assumes every name and group reference is unambiguous
    try {
        common_instances_validate_all(params.instances);
    } catch (const std::invalid_argument & e) {
        IST_ERR("invalid instance configuration: %s\n", e.what());
        return false;
    }
    // the sidecar executor name is reserved for the pool itself; a user instance
    // claiming it would shadow the routing target
    for (const auto & cfg : params.instances) {
        if (cfg.name == decision_sidecar_name()) {
            IST_ERR("instance name '%s' is reserved for the decision sidecar executor\n", decision_sidecar_name());
            return false;
        }
    }

    // the internal sidecar executor needs a decision engine to justify its context
    if (params.decision_sidecar && params.n_seq_decision < 3) {
        IST_ERR("%s", "the decision sidecar executor requires --decision-seqs N (N >= 3)\n");
        return false;
    }

    // each instance inherits the base sleep value into its own scheduler loop
    // with no manager visibility (no sleeping state, no countdown), so refuse
    // the combination instead of sleeping silently. idle reclamation is
    // imperative DELETE /instances/:name.
    if (!params.instances.empty() && params.sleep_idle_seconds >= 0) {
        IST_ERR("%s", "instances do not support --sleep-idle-seconds (would sleep per-instance with no manager visibility); run with sleep disabled\n");
        return false;
    }

    // the pool identity is the first alias (or the model name or the file basename)
    if (!params.model_alias.empty()) {
        base_name = *params.model_alias.begin();
    } else if (!params.model.get_name().empty()) {
        base_name = params.model.get_name();
    } else {
        base_name = std::filesystem::path(params.model.path).filename().string();
    }
    IST_INF("pool identity: base = '%s'\n", base_name.c_str());

    // load the shared weights exactly once
    common_params model_params = params;
    model_init                 = common_init_from_params(model_params, true);
    model                      = model_init ? model_init->model() : nullptr;
    if (model == nullptr) {
        IST_ERR("failed to load model weights '%s'\n", params.model.path.c_str());
        return false;
    }
    IST_INF("loaded shared model weights '%s'\n", params.model.path.c_str());
    train_ctx_cached.store(llama_model_n_ctx_train(model), std::memory_order_relaxed);

    std::vector<common_instance> inst_cfgs = params.instances;
    if (inst_cfgs.empty()) {
        // single default instance, so a bare <base> request always has a target. it takes the
        // global --parallel/-c, preserving stock single-context startup behavior, because
        // common_instance_params never falls back to the base values for a declared instance.
        common_instance inst;
        inst.name       = "default";
        inst.group      = "default";
        inst.is_default = true;
        inst.parallel   = params.n_parallel;
        inst_cfgs.push_back(std::move(inst));
    }

    instances.reserve(inst_cfgs.size());

    // register every configured instance WITHOUT materializing it: each context window
    // (KV + compute) is allocated on first demand for that instance, so declaring N
    // instances never costs N windows upfront. the weights loaded above are the only
    // thing this function loads.
    const bool legacy_default = params.instances.empty();
    for (const auto & cfg : inst_cfgs) {
        auto inst       = std::make_shared<server_instance>();
        inst->cfg       = cfg;
        inst->effective = common_instance_params(params, cfg);

        if (inst->effective.n_parallel < 1) {
            IST_WRN("instance '%s' has no valid n_parallel, defaulting to 1\n", cfg.name.c_str());
            inst->effective.n_parallel = 1;
        }
        if (inst->effective.n_ctx == 0) {
            IST_INF("instance '%s' inherits the model's default context size\n", cfg.name.c_str());
        }

        // guardrail at registration: a huge implicit window is almost never intended,
        // and this is the first line an operator sees for it (before any demand builds it)
        warn_if_huge_implicit_ctx(model, inst->effective, cfg.name.c_str());

        // publish the display caches before the instance is reachable from the vector
        inst->n_ctx_effective.store(inst->effective.n_ctx, std::memory_order_relaxed);
        inst->n_parallel_effective.store(inst->effective.n_parallel, std::memory_order_relaxed);

        instances.push_back(std::move(inst));

        IST_INF("instance '%s' (group '%s', ctx = %d, parallel = %d%s%s) registered, builds on first demand\n",
                cfg.name.c_str(), cfg.group.c_str(), instances.back()->effective.n_ctx,
                instances.back()->effective.n_parallel, cfg.pinned ? ", pinned" : "",
                cfg.is_default ? ", default" : "");
    }

    // the decision sidecar executor: an internal, undeletable executor instance with its own
    // context and scheduler thread. lazy built on the first decision that routes to it. sized to
    // --decision-sidecar-ctx, else the largest configured window, so the longest turn a
    // chat instance can produce still replays on the sidecar. never the default, never in
    // any user group, so chat never lands on it.
    if (params.decision_sidecar) {
        common_instance sidecar;
        sidecar.name     = decision_sidecar_name();
        sidecar.group    = decision_sidecar_name();
        sidecar.ctx_size = params.decision_sidecar_ctx;
        if (sidecar.ctx_size == 0) {
            // the effective window, so an instance that inherits -c still bounds the sidecar.
            // A compiled plan is sized by the request (questions x branch depth), which is not
            // known here, so any fixed default refuses requests a larger window would answer.
            for (const auto & cfg : inst_cfgs) {
                sidecar.ctx_size = std::max(sidecar.ctx_size, common_instance_params(params, cfg).n_ctx);
            }
        }
        sidecar.parallel = 1;

        auto inst       = std::make_shared<server_instance>();
        inst->cfg       = sidecar;
        inst->internal  = true;
        inst->effective = common_instance_params(params, sidecar);
        inst->n_ctx_effective.store(inst->effective.n_ctx, std::memory_order_relaxed);
        inst->n_parallel_effective.store(1, std::memory_order_relaxed);
        instances.push_back(std::move(inst));
        IST_INF("decision sidecar '%s' (ctx = %d, parallel = 1, decision-seqs = %d) registered, builds on the first stateless decision\n",
                decision_sidecar_name(), instances.back()->effective.n_ctx, params.n_seq_decision);
        // --decision-sidecar-prebuild: materialize the sidecar window now instead of on the first
        // decision, so a decision-heavy deployment pays the build cost once at startup
        if (params.decision_sidecar_prebuild) {
            if (auto err = ensure_built_instance(instances.back())) {
                IST_ERR("failed to prebuild decision sidecar '%s'\n", decision_sidecar_name());
                return false;
            }
        }
    }

    // legacy drop-in: with no instances configured the single default instance is built
    // eagerly, preserving stock single-context startup behavior (exactly one window).
    if (legacy_default) {
        if (auto err = ensure_built_instance(instances.front())) {
            IST_ERR("failed to build default instance '%s'\n", instances.front()->cfg.name.c_str());
            return false;
        }
    }

    return true;
}

//
// name resolution
//

// caller must hold mutex_dispatch (the instances vector may be mutated by management ops)
server_instances::resolve_target server_instances::resolve_instance_or_group(const std::string & target,
                                                                             std::string &       error) const {
    resolve_target res;

    // exact instance name wins over group
    for (const auto & inst : instances) {
        if (inst->cfg.name == target) {
            res.kind = target_kind::INSTANCE;
            res.inst = inst;
            return res;
        }
    }
    for (const auto & inst : instances) {
        if (inst->cfg.group == target) {
            res.kind  = target_kind::GROUP;
            res.group = target;
            return res;
        }
    }

    error = "model or instance not found: '" + target + "'";
    return res;
}

server_instances::resolve_target server_instances::resolve(const std::string & model_id,
                                                           const std::string & explicit_instance,
                                                           std::string &       error) const {
    resolve_target              res;
    std::lock_guard<std::mutex> lock(mutex_dispatch);

    const auto pick_default = [&]() {
        for (const auto & inst : instances) {
            if (inst->cfg.is_default) {
                res.kind = target_kind::INSTANCE;
                res.inst = inst;
                return true;
            }
        }
        // an internal window (the decision sidecar) is never a default target: a bare pool id
        // must keep resolving to the sole configured instance even when the sidecar is present
        std::shared_ptr<server_instance> sole;
        for (const auto & inst : instances) {
            if (inst->internal) {
                continue;
            }
            if (sole) {
                sole = nullptr;
                break;
            }
            sole = inst;
        }
        if (sole) {
            res.kind = target_kind::INSTANCE;
            res.inst = sole;
            return true;
        }
        return false;
    };

    // default instance for empty model id; an explicit instance field still takes precedence
    if (model_id.empty()) {
        if (!explicit_instance.empty()) {
            return resolve_instance_or_group(explicit_instance, error);
        }
        if (pick_default()) {
            return res;
        }
        error = "ambiguous: multiple instances and no default, specify one via 'instance' or the model id";
        return res;
    }

    const auto comps = string_split<std::string>(model_id, ':');

    if (comps[0] != base_name) {
        // this process serves exactly one pool; everything else belongs to another server
        error = "model not found on this child: '" + model_id + "'";
        return res;
    }

    // <base> alone: the default instance (or the sole instance, or ambiguous).
    // the explicit `instance` request field overrides the absent component.
    if (comps.size() == 1) {
        if (!explicit_instance.empty()) {
            return resolve_instance_or_group(explicit_instance, error);
        }
        if (pick_default()) {
            return res;
        }
        error = "ambiguous: multiple instances and no default for '" + model_id + "'";
        return res;
    }

    // <base>:latest -> the default instance
    if (comps.size() == 2) {
        std::string target;
        if (comps[1] == "latest") {
            // the explicit `instance` request field overrides the reserved pin
            if (!explicit_instance.empty()) {
                return resolve_instance_or_group(explicit_instance, error);
            }
            if (pick_default()) {
                return res;
            }
            error = "ambiguous: multiple instances and no default for '" + model_id + "'";
            return res;
        }
        target = comps[1];

        // the explicit `instance` request field overrides the model's instance/group component
        if (!explicit_instance.empty()) {
            target = explicit_instance;
        }
        return resolve_instance_or_group(target, error);
    }

    // <base>:latest:<name|group> -> that member. 'latest' is reserved, so a
    // middle component holding anything else cannot be a legal id; more than
    // three components is never a legal id either.
    if (comps.size() == 3) {
        if (comps[1] != "latest") {
            error = "model not found: '" + model_id + "'";
            return res;
        }
        std::string target = comps[2];

        // the explicit `instance` request field overrides the model's instance/group component
        if (!explicit_instance.empty()) {
            target = explicit_instance;
        }
        return resolve_instance_or_group(target, error);
    }

    error = "model not found: '" + model_id + "'";
    return res;
}

// single reduction of an instance's slot activity, shared by group routing and
// last-used reporting: unbuilt windows report never-used stamps; built windows
// report the scheduler-published aggregates (max stamps, busy count). this is
// the one place the fresh policy is defined: last_used_us == -1 sorts a cold
// member before any used member among equally-busy candidates.
static server_context_stats instance_stats(const server_instance & inst) {
    if (!inst.built.load(std::memory_order_acquire)) {
        return server_context_stats{};
    }
    return inst.ctx_server->get_stats();
}

std::optional<size_t> server_instances::pick_best_candidate(const std::vector<route_candidate> & members) {
    std::optional<size_t> best;
    int                   best_busy = INT32_MAX;
    int64_t               best_lru  = INT64_MAX;

    for (const auto & cand : members) {
        // unbuilt (no slots) and fully busy members are never candidates; when
        // every member is busy the caller waits for a slot release instead
        if (cand.n_slots <= 0 || cand.n_busy >= cand.n_slots) {
            continue;
        }

        // prefer fewer busy slots, then least-recently-used; tie-break by
        // instance order (strict compare keeps the first registration)
        bool better = false;
        if (!best) {
            better = true;
        } else if (cand.n_busy != best_busy) {
            better = cand.n_busy < best_busy;
        } else {
            better = cand.last_used_us < best_lru;
        }

        if (better) {
            best      = cand.index;
            best_busy = cand.n_busy;
            best_lru  = cand.last_used_us;
        }
    }
    return best;
}

std::optional<size_t> server_instances::pick_best_available(const std::string & group) const {
    // caller must hold mutex_dispatch
    std::vector<route_candidate> members;
    for (size_t i = 0; i < instances.size(); ++i) {
        const server_instance & inst = *instances[i];
        // removing = a management op owns the instance; running = false once a destroy or
        // a pool-wide shutdown has begun, so a group waiter never picks a dying instance.
        // unbuilt members have no slots yet; group demand materializes the first one
        // (see dispatch_group) instead of picking it here.
        if (inst.cfg.group != group || inst.removing || !inst.running || !inst.built.load(std::memory_order_acquire)) {
            continue;
        }

        const server_context_stats stats = instance_stats(inst);
        members.push_back({ i, inst.n_parallel_effective.load(std::memory_order_relaxed), stats.n_processing, stats.last_used_us });
    }
    return pick_best_candidate(members);
}

//
// request dispatch
//

server_http_res_ptr server_instances::dispatch(const server_http_req & req, const forward_fn & forward) {
    return dispatch(req, forward, dispatch_options{});
}

server_http_res_ptr server_instances::dispatch(const server_http_req & req, const forward_fn & forward,
                                               const dispatch_options & opt) {
    std::string model_id;
    std::string instance_field;
    std::string snapshot;
    int         id_slot = -1;

    try {
        json body = json::parse(req.body);
        if (body.is_object()) {
            model_id       = json_value(body, "model", std::string());
            instance_field = json_value(body, "instance", std::string());
            snapshot       = json_value(body, "snapshot", std::string());
            id_slot        = json_value(body, "id_slot", -1);
        }
    } catch (const std::exception &) {
        // a malformed body is reported by the instance's own handler
    }

    if (model_id.empty()) {
        model_id = req.get_param("model");
    }
    if (instance_field.empty()) {
        instance_field = req.get_param("instance");
    }
    if (snapshot.empty()) {
        snapshot = req.get_param("snapshot");
    }
    if (id_slot < 0) {
        const std::string & s = req.get_param("id_slot");
        if (!s.empty()) {
            try {
                id_slot = std::stoi(s);
            } catch (const std::exception &) {
            }
        }
    }

    std::string          error;
    const resolve_target target = resolve(model_id, instance_field, error);
    switch (target.kind) {
        case target_kind::INSTANCE:
            return dispatch_instance(req, target.inst, snapshot, id_slot, forward);
        case target_kind::GROUP:
            if (opt.require_instance) {
                return make_error("this request addresses one instance's slot; route it with the "
                                  "'instance' or 'model' field, not a group", ERROR_TYPE_INVALID_REQUEST);
            }
            return dispatch_group(req, target.group, snapshot, id_slot, forward);
        case target_kind::NONE:
            return make_error(error, ERROR_TYPE_INVALID_REQUEST);
    }
    return make_error("unreachable", ERROR_TYPE_SERVER);
}

server_http_res_ptr server_instances::dispatch_instance(const server_http_req &                  req,
                                                        const std::shared_ptr<server_instance> & inst,
                                                        const std::string &                      snapshot,
                                                        int                                      id_slot,
                                                        const forward_fn &                       forward) {
    // first demand for this window materializes it: exactly one context is ever built
    // per demand, never the whole pool upfront
    if (auto err = ensure_built_instance(inst)) {
        return err;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_dispatch);
        inst->n_active_dispatch++;
        // check under the same lock that increments the count, so a destroy that starts
        // after this point must drain this dispatch before stopping the scheduler
        if (inst->removing || !inst->running) {
            inst->n_active_dispatch--;
            return make_error("instance '" + inst->cfg.name + "' is being reconfigured", ERROR_TYPE_UNAVAILABLE);
        }
    }

    const auto release_dispatch = [this, &inst]() {
        std::lock_guard<std::mutex> lock(mutex_dispatch);
        inst->n_active_dispatch--;
        cond_dispatch.notify_all();
    };

    server_http_res_ptr err;
    if (!snapshot.empty()) {
        // a plain request leaves the slot (and any bound snapshot) untouched; only a
        // named snapshot can change the binding
        err = apply_snapshot(*inst, snapshot, id_slot);
    }
    if (err) {
        release_dispatch();
        return err;
    }

    try {
        server_http_res_ptr res = forward(*inst->routes, req);
        release_dispatch();
        return res;
    } catch (...) {
        // a handler must not throw (ex_wrapper guarantees it at the HTTP layer), but keep the
        // dispatch counter balanced anyway so a destroy is never blocked on a leaked count
        release_dispatch();
        throw;
    }
}

// group dispatch waits for a free member instead of dispatching onto a queue.
//
// NOTIFY COVERAGE: the wait is predicate-based; every 0->capacity transition that
// affects pick_best_available must notify cond_dispatch under mutex_dispatch. with the
// predicate wait the deadline is the only bound; a missed notify degrades to the 503
// deadline, never an indefinite hang. notify sources:
//   - slot-release callback (every instance, via the instance-creation helper)
//   - create_instance  after push
//   - destroy_instance after erase and after clearing `removing`
//   - resize_instance  after the removing toggles
//   - the drain guard on both set and clear of `removing`
//   - release_dispatch on every dispatch exit
server_http_res_ptr server_instances::dispatch_group(const server_http_req & req,
                                                     const std::string &     group,
                                                     const std::string &     snapshot,
                                                     int                     id_slot,
                                                     const forward_fn &      forward) {
    // a single steady-clock deadline computed once. "wait forever" is time_point::max(),
    // never INT64_MAX arithmetic (which would overflow to a negative deadline and 503
    // a busy-but-eventually-free group immediately).
    const bool wait_forever = params.instance_wait_seconds < 0;
    const auto deadline     = wait_forever
                                  ? std::chrono::steady_clock::time_point::max()
                                  : std::chrono::steady_clock::now() +
                                        std::chrono::seconds(params.instance_wait_seconds);

    while (true) {
        std::shared_ptr<server_instance> inst;
        std::shared_ptr<server_instance> unbuilt;
        {
            std::lock_guard<std::mutex> lock(mutex_dispatch);
            const auto                  best = pick_best_available(group);
            if (best) {
                inst = instances[*best];
                inst->n_active_dispatch++;
                // the removing/running check stays under the same lock as the count, so a
                // destroy that begins after this point must drain this dispatch first
                if (inst->removing || !inst->running) {
                    inst->n_active_dispatch--;
                    cond_dispatch.notify_all();
                    inst.reset();  // re-pick
                }
            } else {
                // no built member has a free slot: materialize the first registered
                // member instead of waiting. one demand builds exactly one window.
                for (const auto & it : instances) {
                    if (it->cfg.group == group && !it->built.load(std::memory_order_acquire) && !it->removing && it->running) {
                        unbuilt = it;
                        break;
                    }
                }
            }
        }

        if (unbuilt) {
            // a persistently failing build degrades to the 503 deadline below, never a hang
            // and never a hot spin: only a successful build re-picks immediately
            if (auto err = ensure_built_instance(unbuilt)) {
                IST_WRN("on-demand build of instance '%s' failed, waiting for a free member\n",
                        unbuilt->cfg.name.c_str());
            } else {
                continue;
            }
        }

        if (inst) {
            const auto release_dispatch = [this, &inst]() {
                std::lock_guard<std::mutex> lock(mutex_dispatch);
                inst->n_active_dispatch--;
                cond_dispatch.notify_all();
            };

            server_http_res_ptr err;
            if (!snapshot.empty()) {
                // see dispatch_instance: a plain request never unbinds a slot
                err = apply_snapshot(*inst, snapshot, id_slot);
            }
            if (err) {
                release_dispatch();
                return err;
            }

            try {
                server_http_res_ptr res = forward(*inst->routes, req);
                release_dispatch();
                return res;
            } catch (...) {
                release_dispatch();
                throw;
            }
        }

        // predicate wait (no poll): re-picks under mutex_dispatch on every notify
        std::unique_lock<std::mutex> lock(mutex_dispatch);
        const bool                   ready = cond_dispatch.wait_until(lock, deadline, [&]() {
            return pick_best_available(group).has_value();
        });
        if (!ready) {
            return make_error("no free instance in group '" + group + "'", ERROR_TYPE_UNAVAILABLE);
        }
    }
}

void server_instances::clear_slot_binding(server_instance & inst, int id_slot) {
    if (id_slot < 0) {
        id_slot = 0;
    }
    std::lock_guard<std::mutex> lock(mutex_dispatch);
    if ((size_t) id_slot < inst.slot_snapshots.size()) {
        inst.slot_snapshots[id_slot].clear();
    }
}

//
// two-phase snapshot compose
//

int64_t server_instances::snapshot_deadline_ms(const server_instance & inst) {
    // 1s floor + 1s per 64k context, an approximation of the KV byte cost. the
    // transfer itself is bounded by the scheduler tasks; this bounds the whole switch.
    return ggml_time_ms() + 1000 + (int64_t) inst.ctx_server->get_slot_n_ctx() / 65536 * 1000;
}

server_instances::switch_guard::switch_guard(server_instances & m, int64_t deadline_ms) : mgr(m) {
    std::unique_lock<std::mutex> lock(mgr.mutex_switch);
    const int64_t                remain = deadline_ms - ggml_time_ms();
    if (remain <= 0) {
        return;
    }
    const bool ok = mgr.cond_switch.wait_for(lock, std::chrono::milliseconds(remain), [&]() {
        return mgr.n_active_switches < max_concurrent_switches;
    });
    if (ok) {
        mgr.n_active_switches++;
        acquired = true;
    }
}

server_instances::switch_guard::~switch_guard() {
    // notify only when the count changed: a failed acquisition waited out its
    // deadline without taking a slot, so there is nothing to hand over
    if (acquired) {
        std::lock_guard<std::mutex> lock(mgr.mutex_switch);
        mgr.n_active_switches--;
        mgr.cond_switch.notify_all();
    }
}

server_instances::server_snapshot_read_result server_instances::snapshot_io_read(const std::string & path,
                                                                                  int64_t             deadline_ms) {
    auto result = std::make_shared<std::optional<server_snapshot_read_out>>();
    auto future = snapshot_io_post([result, path]() {
        *result = server_snapshot_read_status(path);
    });
    if (!future) {
        // the I/O queue is full (hard-bound on queued host memory); a retriable 503
        server_snapshot_read_result busy;
        busy.busy = true;
        return busy;
    }
    const int64_t remain = deadline_ms - ggml_time_ms();
    if (remain <= 0 || future->wait_for(std::chrono::milliseconds(remain)) != std::future_status::ready) {
        server_snapshot_read_result timed_out;
        timed_out.timed_out = true;
        return timed_out;
    }
    future->get();
    server_snapshot_read_result done;
    done.status = (*result)->status;
    done.data   = std::move((*result)->data);
    return done;
}

server_instances::server_snapshot_write_result server_instances::snapshot_io_write(const std::string & path,
                                                                                    server_snapshot_data data,
                                                                                    int64_t             deadline_ms) {
    auto result = std::make_shared<bool>(false);
    auto future = snapshot_io_post([result, path, data = std::move(data)]() {
        *result = server_snapshot_write(path, data);
        if (!*result) {
            IST_WRN("failed to write snapshot '%s'\n", path.c_str());
        }
    });
    if (!future) {
        // the I/O queue is full; the slot must not be bound to a snapshot whose file was
        // never written
        server_snapshot_write_result busy;
        busy.busy = true;
        return busy;
    }
    const int64_t remain = deadline_ms - ggml_time_ms();
    if (remain <= 0 || future->wait_for(std::chrono::milliseconds(remain)) != std::future_status::ready) {
        // timed out: the write still completes in the background on the pool I/O worker;
        // the caller must not bind the slot to this snapshot, it may not exist on disk yet
        server_snapshot_write_result timed_out;
        timed_out.timed_out = true;
        return timed_out;
    }
    future->get();
    server_snapshot_write_result done;
    done.ok = *result;
    return done;
}

server_instances::kv_copy_out server_instances::kv_copy_to_data(server_instance &   inst,
                                                                  int                 id_slot,
                                                                  int64_t             deadline_ms,
                                                                  const std::string & timeout_msg) {
    kv_copy_out out;
    auto result = inst.ctx_server->slot_save_copy(id_slot, deadline_ms);
    if (!result) {
        out.error = make_error(timeout_msg, ERROR_TYPE_UNAVAILABLE);
        return out;
    }
    if (result->is_error()) {
        // a busy slot fails with a retriable 503 instead of deferring
        out.error = make_error_from_result(*result);
        return out;
    }
    auto * copy = dynamic_cast<server_task_result_slot_copy *>(result.get());
    GGML_ASSERT(copy != nullptr);
    out.data.n_ctx_seq = inst.ctx_server->get_slot_n_ctx();
    out.data.tokens    = copy->tokens;
    out.data.kv        = std::move(copy->buffer);
    // record the adapter set this KV was computed under; a restore under a
    // different set is rejected. the fingerprint read is serialized against
    // attach/detach: the drain waits out this dispatch's n_active_dispatch.
    out.data.adapter_fp = common_lora_fingerprint(inst.effective.lora_adapters);
    out.ok = true;
    return out;
}

server_http_res_ptr server_instances::apply_snapshot(server_instance &   inst,
                                                     const std::string & snapshot,
                                                     int                 id_slot) {
    if (!fs_validate_filename(snapshot)) {
        return make_error("invalid snapshot name", ERROR_TYPE_INVALID_REQUEST);
    }
    if (params.slot_save_path.empty()) {
        return make_error("snapshot switching requires --slot-save-path", ERROR_TYPE_NOT_SUPPORTED);
    }
    if (id_slot < 0) {
        id_slot = 0;
    }
    if ((size_t) id_slot >= inst.slot_snapshots.size()) {
        return make_error("invalid slot id", ERROR_TYPE_INVALID_REQUEST);
    }

    // per-instance resolve: the instance-scoped file wins, the legacy flat
    // file is the migration fallback ("" when neither exists, read as 404).
    const std::string filepath = resolve_snapshot_path(inst.cfg.name, snapshot);
    if (filepath.empty()) {
        return make_error("snapshot not found: '" + snapshot + "'", ERROR_TYPE_NOT_FOUND);
    }

    // 1. per-slot lock: switching different slots of one instance do not serialize
    std::lock_guard<std::mutex> slot_lock(*inst.mutex_snapshot[id_slot]);

    // 2. acquire the switch semaphore, bounded by the compose deadline
    const int64_t deadline_ms = snapshot_deadline_ms(inst);
    switch_guard  sw_guard(*this, deadline_ms);
    if (!sw_guard.acquired) {
        return make_error("too many concurrent snapshot switches, retry", ERROR_TYPE_UNAVAILABLE);
    }

    // 3. read the current binding under mutex_dispatch. a slot already bound to this
    //    snapshot serves immediately from the live KV.
    std::string current;
    {
        std::lock_guard<std::mutex> lock(mutex_dispatch);
        current = inst.slot_snapshots[id_slot];
        if (current == snapshot) {
            // already bound: continue the conversation from the live KV
            return nullptr;
        }
    }

    // 4. persist the slot's current KV (if any) before switching away from it.
    //    the write runs awaited on the single FIFO pool I/O worker, which
    //    serializes all file ops; the deadline bounds the wait (see below).
    if (!current.empty()) {
        auto copied = kv_copy_to_data(inst, id_slot, deadline_ms, "snapshot switch timed out while saving");
        if (!copied.ok) {
            return std::move(copied.error);
        }
        server_snapshot_data data = std::move(copied.data);
        // save the old snapshot back where it was read from: the
        // instance-scoped file when present, else the legacy flat file it was
        // restored from (migration), else the instance-scoped path. the old
        // snapshot file must match the slot's KV before we switch away,
        // otherwise a later restore of `current` silently loses the
        // conversation that happened while it was bound. on failure the switch
        // is aborted and the slot keeps its live KV bound to `current`.
        std::string cur_path = resolve_snapshot_path(inst.cfg.name, current);
        if (cur_path.empty()) {
            cur_path = snapshot_instance_path(inst.cfg.name, current);
        }
        auto write = snapshot_io_write(cur_path, std::move(data), deadline_ms);
        if (write.busy) {
            return make_error("snapshot io worker is busy, retry", ERROR_TYPE_UNAVAILABLE);
        }
        if (write.timed_out) {
            return make_error("snapshot switch timed out while saving", ERROR_TYPE_UNAVAILABLE);
        }
        if (!write.ok) {
            return make_error("failed to save snapshot '" + current + "' to disk", ERROR_TYPE_SERVER);
        }
    }

    // 5. MISSING (404) vs CORRUPT (400) is decided by the snapshot module, not an
    //    HTTP-thread existence check (no TOCTOU with a concurrent fire-and-forget write).
    auto read = snapshot_io_read(filepath, deadline_ms);
    if (read.busy) {
        return make_error("snapshot io worker is busy, retry", ERROR_TYPE_UNAVAILABLE);
    }
    if (read.timed_out) {
        return make_error("snapshot read timed out", ERROR_TYPE_UNAVAILABLE);
    }
    if (read.status == server_snapshot_status::MISSING) {
        return make_error("snapshot not found: '" + snapshot + "'", ERROR_TYPE_NOT_FOUND);
    }
    if (read.status == server_snapshot_status::CORRUPT || !read.data) {
        return make_error("corrupt or unreadable snapshot: '" + snapshot + "'", ERROR_TYPE_INVALID_REQUEST);
    }

    // 6. a snapshot saved under a different context size is incompatible with this slot
    if (read.data->n_ctx_seq != inst.ctx_server->get_slot_n_ctx()) {
        return make_error("snapshot context size does not match this slot", ERROR_TYPE_INVALID_REQUEST);
    }

    // 6b. a snapshot saved under a different adapter set is incompatible with this
    // instance: the KV was computed under different tensors. the file version
    // decides what an empty fingerprint means: v1 never had the field
    // (warn-allow, legacy), while v2 always stamps it, so a v2 empty fp means
    // "saved with zero adapters" and must match exactly. the fingerprint read
    // is serialized against attach/detach (see step 4).
    if (!read.data->adapter_fp.empty() || read.data->version > 1) {
        const std::string current_fp = common_lora_fingerprint(inst.effective.lora_adapters);
        if (read.data->adapter_fp != current_fp) {
            return make_error("snapshot adapter set does not match this instance", ERROR_TYPE_INVALID_REQUEST);
        }
    } else {
        IST_WRN("snapshot '%s' has no adapter fingerprint (pre-v2 file), skipping the adapter check\n",
                snapshot.c_str());
    }

    // 7. apply the host KV buffer on the scheduler; busy/timeout -> 503, apply failure
    //    clears the slot so it is never left partially loaded
    auto result = inst.ctx_server->slot_restore_apply(id_slot, std::move(read.data->kv), read.data->tokens, deadline_ms);
    if (!result) {
        clear_slot_binding(inst, id_slot);
        return make_error("snapshot switch timed out while restoring", ERROR_TYPE_UNAVAILABLE);
    }
    if (result->is_error()) {
        clear_slot_binding(inst, id_slot);
        return make_error_from_result(*result);
    }

    // 8. bind on success
    {
        std::lock_guard<std::mutex> lock(mutex_dispatch);
        inst.slot_snapshots[id_slot] = snapshot;
    }
    return nullptr;
}

//
// management API
//

std::shared_ptr<server_instance> server_instances::get_instance(const std::string & name) const {
    std::lock_guard<std::mutex> lock(mutex_dispatch);
    for (const auto & inst : instances) {
        if (inst->cfg.name == name) {
            return inst;
        }
    }
    return nullptr;
}

std::vector<std::shared_ptr<server_instance>> server_instances::snapshot_instances() const {
    std::lock_guard<std::mutex> lock(mutex_dispatch);
    return instances;
}

std::string server_instances::instance_id(const server_instance & inst) const {
    return base_name + ":" + inst.cfg.name;
}

std::set<std::string> server_instances::instance_aliases(const std::string & name, const std::string & group) const {
    std::set<std::string> aliases;
    aliases.insert(base_name);
    aliases.insert(base_name + ":" + group);
    aliases.insert(base_name + ":latest");
    aliases.insert(base_name + ":latest:" + name);
    aliases.insert(base_name + ":latest:" + group);
    return aliases;
}

void server_instances::apply_identity(server_instance & inst) {
    inst.ctx_server->set_model_name(instance_id(inst));
    inst.ctx_server->set_model_aliases(instance_aliases(inst.cfg.name, inst.cfg.group));
}

int32_t server_instances::displayed_n_ctx(const server_instance & inst) const {
    if (inst.built.load(std::memory_order_acquire)) {
        return inst.ctx_server->get_n_ctx();
    }
    const int32_t cached = inst.n_ctx_effective.load(std::memory_order_relaxed);
    if (cached > 0) {
        return cached;
    }
    return train_ctx_cached.load(std::memory_order_relaxed);
}

// most recent slot use across an instance (max t_last_used over its slots), -1 when unused
// (or unbuilt: a registered window has no slots yet)
static int64_t instance_last_used(const server_instance & inst) {
    return instance_stats(inst).last_used_us;
}

// wall-clock twin of instance_last_used: unix epoch seconds of the most recent slot
// release, -1 when unused or unbuilt. safe to compare against the caller's own clock
// and across processes, unlike the monotonic last_used.
static int64_t instance_last_used_epoch(const server_instance & inst) {
    return instance_stats(inst).last_used_wall_s;
}

json server_instances::instance_to_json(const server_instance & inst) const {
    // adapter accounting reads effective.lora_adapters: every caller holds mutex_mgmt
    // (create/resize/pin paths) or takes it first (get_instances_json below)
    const uint64_t adapter_bytes = instance_adapter_bytes(inst);
    // an unbuilt (registered but never demanded) window owns no buffers yet
    if (!inst.built.load(std::memory_order_acquire)) {
        return instance_to_json(inst, 0, 0, 0, adapter_bytes);
    }
    // all memory fields derive from the three instance getters (single source of truth)
    const uint64_t model_bytes   = inst.ctx_server->get_model_bytes();
    const uint64_t context_bytes = inst.ctx_server->get_context_bytes();
    const uint64_t compute_bytes = inst.ctx_server->get_compute_bytes();
    return instance_to_json(inst, model_bytes, context_bytes, compute_bytes, adapter_bytes);
}

json server_instances::instance_to_json(const server_instance & inst,
                                        uint64_t                model_bytes,
                                        uint64_t                context_bytes,
                                        uint64_t                compute_bytes,
                                        uint64_t                adapter_bytes) const {
    const int64_t t_last_used = instance_last_used(inst);
    const int64_t t_last_used_epoch = instance_last_used_epoch(inst);

    return json{
        { "id", instance_id(inst) },
        { "aliases", instance_aliases(inst.cfg.name, inst.cfg.group) },
        { "group", inst.cfg.group },
        { "n_ctx", displayed_n_ctx(inst) },
        { "parallel", inst.n_parallel_effective.load(std::memory_order_relaxed) },
        { "pinned", inst.cfg.pinned },
        { "is_default", inst.cfg.is_default },
        { "internal", inst.internal },
        // unbuilt = registered but never demanded; its window (and bytes) do not exist yet
        { "state", inst.built.load(std::memory_order_acquire) ? "loaded" : "unloaded" },
        // memory breakdown
        { "model_bytes", model_bytes },
        { "context_bytes", context_bytes },
        { "compute_bytes", compute_bytes },
        // this instance's own adapter footprint (a shared entry counts toward every
        // instance referencing it: evicting any one of them does not free it)
        { "adapter_bytes", adapter_bytes },
        // every leg of the window adds up, adapters included
        { "total_bytes", model_bytes + context_bytes + compute_bytes + adapter_bytes },
        // documented alias kept for the list() contract: context + compute
        { "vram_bytes", context_bytes + compute_bytes },
        { "last_used", t_last_used },
        { "last_used_epoch", t_last_used_epoch },
    };
}

json server_instances::get_instances_json() const {
    json     instances_arr = json::array();
    // 64-bit sums; the shared weights are counted once per pool
    uint64_t total_model   = 0;
    uint64_t total_context = 0;
    uint64_t total_compute = 0;
    uint64_t total_adapter = 0;
    bool     model_counted = false;

    // mgmt first (lock order mgmt -> dispatch): adapter accounting reads the
    // registry and effective.lora_adapters, both mgmt-guarded. the scope covers
    // exactly the instance rows, the totals and the registry sum; the snapshot
    // listing below does file I/O and must NOT hold this lock, because
    // GET /instances is the router's polling endpoint and slow storage would
    // otherwise serialize create/destroy/resize behind every poll.
    {
        std::lock_guard<std::mutex> mgmt_lock(mutex_mgmt);
        {
            std::lock_guard<std::mutex> lock(mutex_dispatch);
            for (const auto & inst : instances) {
                // unbuilt windows own no buffers; they contribute identity with zero bytes.
                // both locks are held here (mgmt -> dispatch), so the context is stable
                const bool     is_built      = inst->built.load(std::memory_order_acquire);
                const uint64_t model_bytes   = is_built ? inst->ctx_server->get_model_bytes()   : 0;
                const uint64_t context_bytes = is_built ? inst->ctx_server->get_context_bytes() : 0;
                const uint64_t compute_bytes = is_built ? inst->ctx_server->get_compute_bytes() : 0;
                const uint64_t adapter_bytes = instance_adapter_bytes(*inst);

                instances_arr.push_back(instance_to_json(*inst, model_bytes, context_bytes, compute_bytes, adapter_bytes));

                if (!model_counted && model_bytes > 0) {
                    total_model += model_bytes;
                    model_counted = true;
                }
                total_context += context_bytes;
                total_compute += compute_bytes;
            }
        }

        // pool-wide adapter footprint: each registry file counted once
        for (const auto & kv : adapter_registry) {
            total_adapter += llama_adapter_lora_buf_size(kv.second.adapter.get());
        }
    }

    // snapshots are on-disk and independent of the live instances; list them so a cold
    // (weight-unloaded) pool's KV snapshots stay discoverable for reactivation.
    // this reads only params and base_name, both immutable after load(), so it runs
    // unlocked and never serializes management behind slow snapshot storage.
    json snapshots = pool_snapshots_json();

    return json{
        { "instances", std::move(instances_arr) },
        { "snapshots", std::move(snapshots) },
        { "total",
         {
              { "model", total_model },
              { "context", total_context },
              { "compute", total_compute },
              { "adapter", total_adapter },
              { "total", total_model + total_context + total_compute + total_adapter },
          }                                     },
    };
}

// one snapshot row builder for both JSON producers: {name, size, mtime,
// n_ctx_seq, adapter_fp, instance}. the tag is the owning instance name, or
// null for legacy flat files.
static void push_snapshot_row(std::vector<json> & entries, const server_snapshot_meta & meta, const json & instance_tag) {
    entries.push_back({
        { "name",       meta.name       },
        { "size",       meta.size       },
        { "mtime",      meta.mtime      },
        { "n_ctx_seq",  meta.n_ctx_seq  },
        { "adapter_fp", meta.adapter_fp },
        { "version",    meta.version    },
        { "instance",   instance_tag    },
    });
}

// list one model-key directory: legacy flat files (null instance tag) plus every
// instance subdirectory (a snapshot outlives its instance, so every subdirectory
// is listed for cold-pool discoverability, not just live instances). `seen`
// dedupes scoped entries already listed from the newer (hashed) key so a
// re-saved snapshot appears once; legacy rows predate hashing and are never
// deduped.
static void list_snapshot_key_dir(const std::string & dir,
                                  bool include_legacy,
                                  const std::function<void(const server_snapshot_meta &, const json &)> & push,
                                  std::set<std::string> & seen) {
    if (include_legacy) {
        for (const auto & meta : server_snapshot_list(dir)) {
            push(meta, nullptr);
        }
    }
    std::error_code          ec;
    std::vector<std::string> subdirs;
    std::filesystem::directory_iterator it(dir, ec);
    if (!ec) {
        for (const auto & entry : it) {
            if (ec) {
                break;
            }
            if (entry.is_directory(ec)) {
                subdirs.push_back(entry.path().filename().string());
            }
        }
    }
    std::sort(subdirs.begin(), subdirs.end());
    for (const auto & sub : subdirs) {
        for (const auto & meta : server_snapshot_list(dir + "/" + sub)) {
            if (!seen.insert(sub + '\0' + meta.name).second) {
                continue;
            }
            push(meta, sub);
        }
    }
}

// null-safe string field read for ordering (common_json::value() throws
// on a null, and legacy snapshot entries are explicitly tagged null).
static std::string snapshot_str_field(const json & e, const std::string & key) {
    if (!e.is_object() || !e.contains(key) || !e.at(key).is_string()) {
        return "";
    }
    return e.at(key).get<std::string>();
}

json server_instances::pool_snapshots_json() const {
    std::vector<json> entries;
    if (!params.slot_save_path.empty()) {
        const std::string model_key     = server_snapshot_model_key(base_name);
        const std::string model_key_new = server_snapshot_model_key_hashed(base_name);
        const std::string dir           = params.slot_save_path + model_key;
        const std::string dir_new       = params.slot_save_path + model_key_new;
        const auto push = [&entries](const server_snapshot_meta & meta, const json & instance_tag) {
            push_snapshot_row(entries, meta, instance_tag);
        };
        // a snapshot re-saved after the key change exists under both keys; the
        // hashed key is listed first so its row wins and the older one is skipped.
        // legacy flat files predate hashing and live only under the old key.
        std::set<std::string> seen_scoped;
        list_snapshot_key_dir(dir_new, false, push, seen_scoped);
        // legacy flat files (pre-per-instance layout), tagged with a null instance
        list_snapshot_key_dir(dir, true, push, seen_scoped);
    }
    // deterministic envelope: order by (instance, name), legacy (null) first
    std::sort(entries.begin(), entries.end(), [](const json & a, const json & b) {
        const std::string ai = snapshot_str_field(a, "instance");
        const std::string bi = snapshot_str_field(b, "instance");
        return ai != bi ? ai < bi : snapshot_str_field(a, "name") < snapshot_str_field(b, "name");
    });
    json snapshots = json::array();
    for (auto & e : entries) {
        snapshots.push_back(std::move(e));
    }
    return snapshots;
}

// snapshots visible to one instance: its own instance-scoped directory (new key
// first, then the previous key) plus the legacy flat files (migration read
// path, tagged with a null instance).
json server_instances::instance_snapshots_json(const std::string & instance) const {
    std::vector<json> entries;
    if (!params.slot_save_path.empty()) {
        const std::string model_key     = server_snapshot_model_key(base_name);
        const std::string model_key_new = server_snapshot_model_key_hashed(base_name);
        const std::string dir           = params.slot_save_path + model_key;
        const std::string dir_new       = params.slot_save_path + model_key_new;
        const auto push_scoped = [&entries, &instance](const server_snapshot_meta & meta) {
            push_snapshot_row(entries, meta, instance);
        };
        // the hashed key wins over the previous key for re-saved snapshots
        std::set<std::string> seen;
        for (const auto & meta : server_snapshot_list(dir_new + "/" + instance)) {
            seen.insert(meta.name);
            push_scoped(meta);
        }
        for (const auto & meta : server_snapshot_list(dir + "/" + instance)) {
            if (!seen.insert(meta.name).second) {
                continue;
            }
            push_scoped(meta);
        }
        for (const auto & meta : server_snapshot_list(dir)) {
            push_snapshot_row(entries, meta, nullptr);
        }
    }
    std::stable_sort(entries.begin(), entries.end(), [](const json & a, const json & b) {
        return snapshot_str_field(a, "name") < snapshot_str_field(b, "name");
    });
    json snapshots = json::array();
    for (auto & e : entries) {
        snapshots.push_back(std::move(e));
    }
    return snapshots;
}

std::string server_instances::snapshot_instance_path(const std::string & instance,
                                                     const std::string & snapshot) const {
    // the only write target: the hashed key, so identities that sanitize alike
    // never share a directory
    return server_snapshot_instance_path(params.slot_save_path, server_snapshot_model_key_hashed(base_name),
                                         instance, snapshot);
}

std::string server_instances::snapshot_instance_path_prev(const std::string & instance,
                                                          const std::string & snapshot) const {
    // read fallback for files written before key hashing (same shape as the
    // scoped -> legacy fallback below)
    return server_snapshot_instance_path(params.slot_save_path, server_snapshot_model_key(base_name),
                                         instance, snapshot);
}

std::string server_instances::snapshot_legacy_path(const std::string & snapshot) const {
    return server_snapshot_legacy_path(params.slot_save_path, server_snapshot_model_key(base_name), snapshot);
}

// resolve a snapshot for read (and save-back): the hashed-key scoped file wins,
// then the previous-key scoped file, then the legacy flat file (migration
// fallbacks). "" when none exists.
std::string server_instances::resolve_snapshot_path(const std::string & instance,
                                                    const std::string & snapshot) const {
    std::error_code ec;
    const std::string inst_path = snapshot_instance_path(instance, snapshot);
    if (std::filesystem::exists(inst_path, ec)) {
        return inst_path;
    }
    const std::string prev_path = snapshot_instance_path_prev(instance, snapshot);
    if (std::filesystem::exists(prev_path, ec)) {
        return prev_path;
    }
    const std::string leg_path = snapshot_legacy_path(snapshot);
    if (std::filesystem::exists(leg_path, ec)) {
        return leg_path;
    }
    return "";
}

server_instances::server_instances(context_builder_fn builder) {
    set_context_builder(std::move(builder));
}

std::string server_instances::adapter_key(const std::string & path) {
    return std::filesystem::absolute(path).lexically_normal().string();
}

std::optional<std::string> server_instances::try_adapter_key(const std::string & path) {
    // absolute() is purely lexical on common platforms, so an embedded NUL
    // would sail through into registry keys, fingerprints and declarations
    // that can never load. reject it here, before any of that happens.
    if (path.find('\0') != std::string::npos) {
        return std::nullopt;
    }
    try {
        return std::filesystem::absolute(path).lexically_normal().string();
    } catch (const std::filesystem::filesystem_error &) {
        return std::nullopt;
    }
}

llama_adapter_lora * server_instances::ensure_adapter(const std::string & path) {
    const auto key_opt = try_adapter_key(path);
    if (!key_opt) {
        return nullptr;
    }
    const std::string key = *key_opt;
    auto it = adapter_registry.find(key);
    if (it != adapter_registry.end()) {
        it->second.refcount++;
        return it->second.adapter.get();
    }
    llama_adapter_lora_ptr adapter;
    adapter.reset(llama_adapter_lora_init(model, key.c_str()));
    if (adapter == nullptr) {
        IST_ERR("failed to load lora adapter '%s'\n", key.c_str());
        return nullptr;
    }
    llama_adapter_lora * ptr = adapter.get();
    adapter_registry[key] = { std::move(adapter), 1 };
    return ptr;
}

void server_instances::release_adapter(const std::string & key) {
    auto it = adapter_registry.find(key);
    if (it == adapter_registry.end()) {
        return;
    }
    if (--it->second.refcount == 0) {
        adapter_registry.erase(it);
    }
}

void server_instances::release_adapter_set(const std::vector<common_adapter_lora_info> & loras) {
    for (const auto & la : loras) {
        if (la.ptr != nullptr) {
            release_adapter(adapter_key(la.path));
        }
    }
}

std::optional<std::vector<common_adapter_lora_info>> server_instances::resolve_adapter_set(const common_instance & cfg) {
    std::vector<common_adapter_lora_info> resolved = common_instance_params(params, cfg).lora_adapters;
    for (auto & la : resolved) {
        llama_adapter_lora * ptr = ensure_adapter(la.path);
        if (ptr == nullptr) {
            release_adapter_set(resolved);
            return std::nullopt;
        }
        // canonical key as path: one spelling per file, so the snapshot
        // fingerprint is stable across restarts and re-resolution
        la.path = adapter_key(la.path);
        la.ptr  = ptr;
        // the report-only meta travels with the mirror so a GET shape stays
        // consistent whether the entry came from a build or a refresh
        common_adapter_lora_fill_meta(la);
    }
    return resolved;
}

uint64_t server_instances::instance_adapter_bytes(const server_instance & inst) const {
    uint64_t bytes = 0;
    for (const auto & la : inst.effective.lora_adapters) {
        bytes += llama_adapter_lora_buf_size(la.ptr);
    }
    return bytes;
}

void server_instances::set_context_builder(context_builder_fn builder) {
    if (builder) {
        context_builder = std::move(builder);
    } else {
        context_builder = [this](server_instance & inst) { return build_context_default(inst); };
    }
}

void server_instances::start_instance_loop_locked(server_instance & inst) {
    // non-recursive mutex: try_lock fails exactly when the caller holds it
    assert(mutex_dispatch.try_lock() == false);
    if (inst.loop_started) {
        return;
    }
    inst.loop_thread = std::thread([&inst]() { inst.ctx_server->start_loop(); });
    inst.loop_started = true;
}

bool server_instances::build_context_default(server_instance & inst) {
    // resolve the adapter set into pool-owned entries BEFORE the context load:
    // init_from_model skips (never re-loads) entries with a non-null ptr.
    // the registry access relies on the caller holding mutex_mgmt.
    auto resolved = resolve_adapter_set(inst.cfg);
    if (!resolved) {
        inst.adapter_failed = true;
        return false;
    }
    inst.effective.lora_adapters = std::move(*resolved);

    // shared ownership of the pool's weights: the context borrows `model`, so this copy
    // guarantees the model outlives the context (and any transient shared_ptr reference
    // to this instance held by an aggregate handler) even after the pool frees its copy
    inst.model_owner = model_init;

    // allocate only this instance's KV + compute buffers from the already-loaded weights
    inst.ctx_server = std::make_unique<server_context>();
    if (!inst.ctx_server->load_model(inst.effective, model)) {
        // drop the refs taken above and restore the declared set so a later
        // demand retries the same adapters (never a dangling ptr)
        release_adapter_set(inst.effective.lora_adapters);
        inst.effective.lora_adapters = common_instance_params(params, inst.cfg).lora_adapters;
        inst.ctx_server.reset();
        inst.model_owner.reset();
        return false;
    }

    apply_identity(inst);

    inst.slot_snapshots.resize((size_t) inst.effective.n_parallel);
    server_instance_size_slot_locks(inst.mutex_snapshot, inst.effective.n_parallel);

    inst.routes = std::make_unique<server_routes>(inst.effective, *inst.ctx_server);
    inst.routes->update_meta(*inst.ctx_server);

    // wake group waiters on the first 0->capacity transition; a completed turn also advances the
    // slot's decision turn counter, so a session whose turn is over is marked for removal
    inst.ctx_server->set_slot_release_callback([this, name = inst.cfg.name](int id_slot) {
        on_decision_slot_release(name, id_slot);
        std::lock_guard<std::mutex> lock(mutex_dispatch);
        cond_dispatch.notify_all();
    });

    // the internal sidecar executor resolves routed token-snapshot sessions from the pool store
    if (inst.internal) {
        inst.ctx_server->set_decision_snapshot_resolver([this](const std::string & key) {
            return decision_snapshot_by_key(key);
        });
    }

    return true;
}

bool server_instances::build_context_into(server_instance & inst) {
    inst.effective = common_instance_params(params, inst.cfg);
    inst.adapter_failed = false;

    // instances never sleep: the pool has no residency state machine and
    // reports no sleeping state, so an inherited idle timeout would park a
    // scheduler with zero manager visibility. load() already refuses the
    // combination at startup; strip it here too for runtime-created windows.
    inst.effective.sleep_idle_seconds = -1;

    if (inst.effective.n_parallel < 1) {
        IST_WRN("instance '%s' has no valid n_parallel, defaulting to 1\n", inst.cfg.name.c_str());
        inst.effective.n_parallel = 1;
    }

    // the internal decision sidecar is the one executor that owns a decision engine pool: it
    // restores the decision sequences and unified KV that chat instances no longer carry,
    // so its own context is sized for the engine pool while every chat instance is stock. the
    // resident warm-prefix slots sit above the pool and hold session prefixes resident.
    if (inst.internal) {
        inst.effective.n_seq_decision = params.n_seq_decision;
        inst.effective.n_seq_warm     = decision_warm_slots(model, inst.effective);
        inst.effective.kv_unified     = true;
    }

    if (inst.effective.n_ctx == 0) {
        IST_INF("instance '%s' inherits the model's default context size\n", inst.cfg.name.c_str());
    }

    // guardrail before the allocation: a huge implicit window is almost never intended
    warn_if_huge_implicit_ctx(model, inst.effective, inst.cfg.name.c_str());

    // publish the display caches for the unbuilt window (and again after a successful
    // build, where they match the context exactly); the build below is not observable
    // through `built` until the caller release-stores true
    inst.n_ctx_effective.store(inst.effective.n_ctx, std::memory_order_relaxed);
    inst.n_parallel_effective.store(inst.effective.n_parallel, std::memory_order_relaxed);

    return context_builder(inst);
}

void server_instances::teardown_instance_context(server_instance & inst) {
    // close the barrier first: a lock-free reader that acquire-loads false after
    // this point never touches ctx_server, and one that still sees true is held off
    // by the drain guard the caller holds (`removing`), which also waits out every
    // in-flight dispatch before the context is freed below
    inst.built.store(false, std::memory_order_release);
    // abort first so in-flight generations finish promptly; the drain guard the
    // caller holds then waits only for stragglers, never a never-ending decode.
    // this abort stays unbounded on purpose: the only caller (resize) runs a
    // bounded abort first and only reaches here once the scheduler proved
    // live, so this second abort is a fast-path no-op, not a stall risk.
    if (inst.ctx_server) {
        inst.ctx_server->abort_slots("instance '" + inst.cfg.name + "' resized");
        inst.ctx_server->terminate();
    }
    // never join under mutex_dispatch: a manager thread holding the dispatch
    // lock while the scheduler tears down would deadlock the teardown path
    bool do_join;
    {
        std::lock_guard<std::mutex> lock(mutex_dispatch);
        do_join               = !inst.scheduler_joined;
        inst.scheduler_joined = true;
    }
    if (do_join && inst.loop_thread.joinable()) {
        inst.loop_thread.join();
    }
    inst.ctx_server.reset();
    inst.routes.reset();
    inst.model_owner.reset();
    // drop this window's adapter refs while the model is still alive; the rebuild
    // re-resolves from cfg. restore the declared set so no dangling ptr survives
    // when the rebuild fails and the window stays unbuilt.
    release_adapter_set(inst.effective.lora_adapters);
    inst.effective.lora_adapters = common_instance_params(params, inst.cfg).lora_adapters;
    inst.slot_snapshots.clear();
    inst.mutex_snapshot.clear();
    inst.loop_started = false;
}

std::shared_ptr<server_instance> server_instances::build_instance(const common_instance & cfg, bool & adapter_failed) {
    auto inst = std::make_shared<server_instance>();
    inst->cfg = cfg;

    if (!build_context_into(*inst)) {
        adapter_failed = inst->adapter_failed;
        return nullptr;
    }

    adapter_failed = false;
    return inst;
}

// materialize one registered instance on first demand. the expensive work (weight
// reload, context alloc) runs under mutex_mgmt with NO other lock held, so concurrent
// first demands for one instance collapse onto a single build and a reload can never
// race create_instance's reload (lock order everywhere is mgmt -> dispatch). the built
// context is installed under mutex_dispatch; terminate() null-guards unbuilt instances
// and create_instance's shutdown check is mirrored here, so a build that loses a race
// with teardown frees what it allocated and reports unavailable instead of leaking a
// scheduler thread.
server_http_res_ptr server_instances::cold_reload_locked() {
    // reload the shared weights when the pool went cold (last instance destroyed)
    if (model != nullptr) {
        return nullptr;
    }
    // the registry drains at delete-last, so a cold pool never holds stale entries:
    // inherited adapters re-resolve from the config, not from old pointers. a
    // non-empty registry here means a refcount accounting bug. debug builds abort
    // loudly at the source; release builds must not take a live server down for an
    // internal accounting bug, so they log and refuse with a retriable 507.
    if (!adapter_registry.empty()) {
#ifndef NDEBUG
        GGML_ABORT("adapter registry not drained before cold reload");
#else
        IST_ERR("cold reload with %zu leaked adapter entries, refusing\n", adapter_registry.size());
        return make_error(507, "insufficient_memory_error", "adapter registry is inconsistent, restart the server");
#endif
    }
    common_params model_params = params;
    model_init                 = common_init_from_params(model_params, true);
    model                      = model_init ? model_init->model() : nullptr;
    if (model == nullptr) {
        IST_ERR("failed to reload model weights '%s'\n", params.model.path.c_str());
        return make_error(507, "insufficient_memory_error", "failed to reload shared weights");
    }
    IST_INF("reloaded shared model weights '%s'\n", params.model.path.c_str());
    train_ctx_cached.store(llama_model_n_ctx_train(model), std::memory_order_relaxed);
    return nullptr;
}

void server_instances::drop_built_context_locked(server_instance & inst) {
    // the scheduler was never started, so there is no thread to join: drop the
    // fresh context with no thread to join, release the refs the build took
    // and restore the declared set for a later retry
    inst.routes.reset();
    inst.ctx_server->terminate();
    inst.ctx_server.reset();
    inst.model_owner.reset();
    release_adapter_set(inst.effective.lora_adapters);
    inst.effective.lora_adapters = common_instance_params(params, inst.cfg).lora_adapters;
}

server_http_res_ptr server_instances::ensure_built_instance(const std::shared_ptr<server_instance> & inst) {
    std::lock_guard<std::mutex> mgmt_lock(mutex_mgmt);

    {
        std::lock_guard<std::mutex> lock(mutex_dispatch);
        if (inst->built.load(std::memory_order_acquire)) {
            return nullptr;
        }
        bool registered = false;
        for (const auto & it : instances) {
            if (it == inst) {
                registered = true;
                break;
            }
        }
        if (!registered) {
            return make_error("instance '" + inst->cfg.name + "' no longer exists", ERROR_TYPE_NOT_FOUND);
        }
        if (inst->removing || !inst->running || terminated) {
            return make_error("instance '" + inst->cfg.name + "' is being reconfigured", ERROR_TYPE_UNAVAILABLE);
        }
    }

    // reload the shared weights when the pool went cold (last instance destroyed)
    if (auto err = cold_reload_locked()) {
        return err;
    }

    // materialize the window: effective params are recomputed from cfg, then the
    // injected builder allocates (the default builder resolves the adapter set
    // into pool-owned entries first; a bad adapter path is a 400, never a 507).
    // expensive work runs with no dispatch lock held; the install below is gated
    // on a teardown race, mirroring create_instance's shutdown check.
    if (!build_context_into(*inst)) {
        if (inst->adapter_failed) {
            return make_error("failed to load lora adapter for instance '" + inst->cfg.name + "'", ERROR_TYPE_INVALID_REQUEST);
        }
        IST_ERR("failed to allocate instance '%s', shared model stays loaded\n", inst->cfg.name.c_str());
        return make_error(507, "insufficient_memory_error",
                          "failed to allocate instance '" + inst->cfg.name + "', not enough device memory");
    }

    {
        std::lock_guard<std::mutex> lock(mutex_dispatch);
        if (inst->removing || !inst->running || terminated) {
            // lost a race with teardown after the build: drop the fresh context,
            // nothing is installed, no scheduler is started.
            drop_built_context_locked(*inst);
            return make_error("instance '" + inst->cfg.name + "' is being reconfigured", ERROR_TYPE_UNAVAILABLE);
        }
        // start the scheduler only now that the instance is fully installed, so a
        // racing terminate() cannot leak the thread. the release-store publishes the
        // context writes above to any lock-free reader that acquire-loads built.
        start_instance_loop_locked(*inst);
        inst->built.store(true, std::memory_order_release);
        cond_dispatch.notify_all();  // wake group waiters so the new window can be picked
    }

    IST_INF("instance '%s' (group '%s', ctx = %d, parallel = %d) built on demand\n", inst->cfg.name.c_str(),
            inst->cfg.group.c_str(), inst->effective.n_ctx, inst->effective.n_parallel);
    return nullptr;
}

server_http_res_ptr server_instances::create_instance(const common_instance & cfg) {
    // management ops serialize on mutex_mgmt, so the duplicate check below stays
    // authoritative through the push at the bottom (no concurrent create/destroy can
    // interleave) and the weight reload never races a last-instance unload
    std::lock_guard<std::mutex> mgmt_lock(mutex_mgmt);

    // the sidecar name is reserved for the pool's internal executor
    if (cfg.name == decision_sidecar_name()) {
        return make_error(409, "invalid_request_error",
                          "instance name is reserved: '" + cfg.name + "'");
    }

    // reject duplicates before any expensive weight reload
    {
        std::lock_guard<std::mutex> lock(mutex_dispatch);
        for (const auto & inst : instances) {
            if (inst->cfg.name == cfg.name) {
                return make_error(409, "invalid_request_error", "instance already exists: '" + cfg.name + "'");
            }
            if (inst->cfg.group == cfg.name || inst->cfg.name == cfg.group) {
                return make_error(409, "invalid_request_error",
                                  "instance name collides with an existing group: '" + cfg.name + "'");
            }
        }
    }

    // reload the shared weights first when the pool went cold (last instance destroyed)
    if (auto err = cold_reload_locked()) {
        return err;
    }

    bool adapter_failed = false;
    auto inst = build_instance(cfg, adapter_failed);
    if (!inst) {
        if (adapter_failed) {
            return make_error("failed to load lora adapter for instance '" + cfg.name + "'", ERROR_TYPE_INVALID_REQUEST);
        }
        IST_ERR("failed to allocate instance '%s', shared model stays loaded\n", cfg.name.c_str());
        return make_error(507, "insufficient_memory_error",
                          "failed to allocate instance '" + cfg.name + "', not enough device memory");
    }

    IST_INF("creating instance '%s' (group '%s', ctx = %d, parallel = %d)\n", cfg.name.c_str(), cfg.group.c_str(),
            inst->effective.n_ctx, inst->effective.n_parallel);

    {
        std::lock_guard<std::mutex> lock(mutex_dispatch);
        if (terminated) {
            // the server is shutting down mid-create: the scheduler was never
            // started, so drop the fresh context with no thread to join
            drop_built_context_locked(*inst);
            return make_error(503, "unavailable_error", "server is shutting down");
        }
        // explicitly created means explicitly demanded: push first, then start the
        // scheduler under the same lock, so terminate() cannot slip between them
        // and leak a joinable thread. release-store publishes the fresh context.
        inst->built.store(true, std::memory_order_release);
        instances.push_back(inst);
        start_instance_loop_locked(*inst);
        cond_dispatch.notify_all();  // wake group waiters so the new member can be picked
    }

    IST_INF("instance '%s' created at runtime\n", cfg.name.c_str());
    return make_ok(instance_to_json(*inst), 201);
}

server_http_res_ptr server_instances::destroy_instance(const std::string & name, bool) {
    // serialized with the other management ops so this can never race a resize or a
    // concurrent create/destroy of the same (or any) instance
    std::lock_guard<std::mutex> mgmt_lock(mutex_mgmt);

    // the internal sidecar executor is manager-owned: it exists for the pool's lifetime
    // and is recreated on demand; a delete would only break the decision routing
    if (name == decision_sidecar_name()) {
        return make_error(400, "invalid_request_error", "the decision sidecar executor is internal and cannot be deleted");
    }

    // pinned is advisory in this branch; the force flag is accepted and ignored
    std::shared_ptr<server_instance> inst;
    size_t inst_pos = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_dispatch);
        for (size_t i = 0; i < instances.size(); ++i) {
            if (instances[i]->cfg.name != name) {
                continue;
            }
            inst = instances[i];
            inst_pos = i;
            instances.erase(instances.begin() + i);
            // flip running under the same lock as the erase: any dispatch that incremented
            // before this point is drained by the guard below, any dispatch that checks
            // after sees running == false and never posts to the about-to-stop scheduler
            inst->running = false;
            cond_dispatch.notify_all();  // a group waiter must re-pick without this member
            break;
        }
    }

    if (!inst) {
        return make_error("instance not found: '" + name + "'", ERROR_TYPE_NOT_FOUND);
    }

    // an unbuilt window never started a scheduler and owns no context: nothing to
    // abort, drain or join (no dispatch can be inside it: ensure_built_instance only
    // reports success once built is set)
    if (inst->built.load(std::memory_order_acquire)) {
        // abort in-flight generation, then drain the remaining dispatched requests so no
        // HTTP reader is left hanging when the scheduler is stopped below.
        // bounded: the management lock is held across this call, so a stalled
        // scheduler answers 503 instead of wedging every other management op
        // behind it. on timeout the unlink above is rolled back (same position,
        // running flipped back) so the instance is fully intact and a retry
        // finds it where it always was; without the rollback the last local
        // reference would tear down a live scheduler here.
        if (!inst->ctx_server->abort_slots("instance '" + name + "' evicted", snapshot_deadline_ms(*inst))) {
            {
                std::lock_guard<std::mutex> lock(mutex_dispatch);
                inst->running = true;
                instances.insert(instances.begin() + std::min(inst_pos, instances.size()), inst);
                cond_dispatch.notify_all();  // group waiters re-pick with this member
            }
            return make_error("timed out aborting in-flight work, retry", ERROR_TYPE_UNAVAILABLE);
        }

        {
            instance_drain_guard guard(*this, inst);
            inst->ctx_server->terminate();
            bool do_join;
            {
                std::lock_guard<std::mutex> lock(mutex_dispatch);
                do_join               = !inst->scheduler_joined;
                inst->scheduler_joined = true;
            }
            if (do_join && inst->loop_thread.joinable()) {
                inst->loop_thread.join();
            }
        }
    }

    // release this manager reference now; the weights are NOT freed here if any other
    // reference is still alive. each instance holds a shared copy of the pool model
    // (model_owner) that outlives its context, so the model is freed only when the last
    // reference -- including a transient shared_ptr held by an in-flight aggregate
    // handler -- actually drops, never while a context can still dereference it.
    // drop this instance's adapter refs first: entries at refcount 0 free here while
    // the model is still alive (an unbuilt window holds no refs; its null ptrs skip).
    release_adapter_set(inst->effective.lora_adapters);
    inst.reset();

    // delete-last: no live instances remain, so free the shared weights
    bool last_instance;
    {
        std::lock_guard<std::mutex> lock(mutex_dispatch);
        last_instance = instances.empty();
    }
    if (last_instance) {
        // drain the registry while the model is still alive (llama_adapter_lora_free
        // erases each adapter from model->loras). freeing the model first would leave
        // the registry unique_ptrs dangling. empty by construction (every instance
        // released its refs above), so this only guards against refcount bugs.
        adapter_registry.clear();
        model_init.reset();
        model = nullptr;
        IST_INF("pool '%s' has no instances left, shared weights unloaded\n", base_name.c_str());
    }

    IST_INF("instance '%s' destroyed\n", name.c_str());
    return make_ok({
        { "success", true }
    });
}

server_http_res_ptr server_instances::resize_instance(const std::string & name, int32_t new_ctx) {
    if (new_ctx <= 0) {
        return make_error("ctx_size must be positive", ERROR_TYPE_INVALID_REQUEST);
    }
    // the sidecar sizes itself from --decision-sidecar-ctx at registration; a runtime
    // resize would silently change the replay budget for every decision
    if (name == decision_sidecar_name()) {
        return make_error(400, "invalid_request_error", "the decision sidecar executor is internal and cannot be resized");
    }

    // serialized with the other management ops: a resize and a destroy of the same
    // instance can no longer race (destroy must never stop the queue under resize)
    std::lock_guard<std::mutex> mgmt_lock(mutex_mgmt);

    std::shared_ptr<server_instance> inst;
    {
        std::lock_guard<std::mutex> lock(mutex_dispatch);
        for (const auto & it : instances) {
            if (it->cfg.name == name) {
                inst = it;
                break;
            }
        }
        if (!inst) {
            return make_error("instance not found: '" + name + "'", ERROR_TYPE_NOT_FOUND);
        }
        // an unbuilt window has no context to rebuild: record the new size, it applies
        // when the window materializes on first demand. the display cache follows the
        // declaration so a concurrent reader never sees the old size
        if (!inst->built.load(std::memory_order_acquire)) {
            inst->cfg.ctx_size = new_ctx;
            inst->effective     = common_instance_params(params, inst->cfg);
            inst->n_ctx_effective.store(inst->effective.n_ctx, std::memory_order_relaxed);
            inst->n_parallel_effective.store(inst->effective.n_parallel, std::memory_order_relaxed);
            IST_INF("instance '%s' resized to ctx = %d (applies on first demand)\n", name.c_str(), new_ctx);
            return make_ok(instance_to_json(*inst));
        }
    }

    // exclusive access while the old context is torn down and the new one is
    // built at the requested size. the guard rejects new dispatches and drains
    // in-flight ones; its destructor restores `removing` even when the rebuild
    // fails or throws.
    // abort in-flight generations before draining: the drain below would otherwise
    // wait forever on a never-ending decode (same order as destroy_instance).
    // bounded like destroy: a stalled scheduler answers 503 with the window
    // untouched instead of stalling the management plane.
    if (!inst->ctx_server->abort_slots("instance '" + name + "' resized", snapshot_deadline_ms(*inst))) {
        return make_error("timed out aborting in-flight work, retry", ERROR_TYPE_UNAVAILABLE);
    }
    {
        instance_drain_guard guard(*this, inst);
        // pin the post-resize adapter entries BEFORE the teardown drops the old
        // refs: the overlapping refs keep shared entries alive, so the rebuild
        // borrows them instead of reloading the files. a resolve failure answers
        // 400 with the window untouched (never a destructive teardown for a
        // predictable failure).
        const int32_t prev_ctx = inst->cfg.ctx_size;
        inst->cfg.ctx_size = new_ctx;
        auto pinned = resolve_adapter_set(inst->cfg);
        if (!pinned) {
            inst->cfg.ctx_size = prev_ctx;
            return make_error("failed to load lora adapter for instance '" + name + "'", ERROR_TYPE_INVALID_REQUEST);
        }
        teardown_instance_context(*inst);

        if (!build_context_into(*inst)) {
            // the rebuild failed: a well-defined unbuilt window at the new size.
            // a later demand retries the build instead of serving a dangling context.
            // teardown already released the old refs; a resolve failure took none.
            release_adapter_set(*pinned);
            IST_ERR("failed to resize instance '%s' to ctx = %d, leaving it unbuilt\n", name.c_str(), new_ctx);
            if (inst->adapter_failed) {
                return make_error("failed to load lora adapter for instance '" + name + "'", ERROR_TYPE_INVALID_REQUEST);
            }
            return make_error(507, "insufficient_memory_error",
                              "failed to resize instance '" + name + "', not enough device memory");
        }
        // the build resolved its own refs; drop the pins
        release_adapter_set(*pinned);

        // the window was rebuilt (bindings were dropped with the old context);
        // start the scheduler exactly once under the dispatch lock. the release-store
        // publishes the rebuilt context to lock-free readers.
        std::lock_guard<std::mutex> lock(mutex_dispatch);
        start_instance_loop_locked(*inst);
        inst->built.store(true, std::memory_order_release);
        cond_dispatch.notify_all();
    }

    IST_INF("instance '%s' resized to ctx = %d\n", name.c_str(), new_ctx);
    return make_ok({ { "n_ctx", displayed_n_ctx(*inst) } });
}

server_http_res_ptr server_instances::set_instance_pinned(const std::string & name, bool pinned) {
    std::lock_guard<std::mutex> mgmt_lock(mutex_mgmt);
    std::shared_ptr<server_instance> inst;
    {
        std::lock_guard<std::mutex> lock(mutex_dispatch);
        for (const auto & it : instances) {
            if (it->cfg.name == name) {
                inst = it;
                break;
            }
        }
        if (!inst) {
            return make_error("instance not found: '" + name + "'", ERROR_TYPE_NOT_FOUND);
        }
        inst->cfg.pinned = pinned;
    }

    IST_INF("instance '%s' %s\n", name.c_str(), pinned ? "pinned" : "unpinned");
    return make_ok(instance_to_json(*inst));
}

//
// HTTP handlers
//
// endpoints carrying a `model` are dispatched to the owning instance; endpoints without
// one (tokenize, props, health, ...) run on the default instance. aggregate endpoints
// (/models, /slots) merge the per-instance state.
//

server_http_res_ptr server_instances::handle_get_health(const server_http_req & req) {
    auto inst = default_instance();
    if (!inst) {
        // a cold/empty pool is healthy but has nothing loaded; a liveness probe must be
        // able to distinguish cold from dead
        return make_ok({
            { "status",    "ok"      },
            { "instances", 0         },
        });
    }
    // a registered-but-undemanded default is healthy too, and probing it must NOT
    // materialize its window: orchestrators poll /health constantly, and a probe
    // is not a demand. the guard is taken BEFORE any field is read, so the shared_ptr
    // cannot outlive a concurrent destroy/resize into a torn context.
    active_route_guard guard(*this, *inst);
    if (!guard.acquired) {
        return make_error("instance '" + inst->cfg.name + "' is being reconfigured", ERROR_TYPE_UNAVAILABLE);
    }
    if (!inst->built.load(std::memory_order_acquire)) {
        return make_ok({
            { "status",    "ok"      },
            { "instances", (int) snapshot_instances().size() },
        });
    }
    return inst->routes->get_health(req);
}

server_http_res_ptr server_instances::handle_get_metrics(const server_http_req & req) {
    // a scrape is an observability read, never a demand: it must not materialize
    // an unbuilt window. unlike /slots there is no aggregate form, because the
    // metrics are context-local counters and merging them would invent a gauge
    // policy; no target therefore means the default instance.
    const std::string model_id       = req.get_param("model");
    const std::string instance_field = req.get_param("instance");

    std::shared_ptr<server_instance> inst;
    if (model_id.empty() && instance_field.empty()) {
        inst = default_instance();
    } else {
        std::string          error;
        const resolve_target target = resolve(model_id, instance_field, error);
        if (target.kind != target_kind::INSTANCE) {
            return make_error(error.empty() ? "invalid instance for metrics" : error, ERROR_TYPE_INVALID_REQUEST);
        }
        inst = target.inst;
    }
    if (!inst) {
        return make_error("no instances loaded", ERROR_TYPE_NOT_FOUND);
    }
    // the target window does not exist yet: render nothing rather than allocate
    // it as a scrape side effect. this is an exact outcome condition, not a
    // health or confidence score. the guard is taken first so a concurrent
    // destroy/resize cannot turn this read into a torn context dereference.
    active_route_guard guard(*this, *inst);
    if (!guard.acquired) {
        return make_error("instance '" + inst->cfg.name + "' is being reconfigured", ERROR_TYPE_UNAVAILABLE);
    }
    if (!inst->built.load(std::memory_order_acquire)) {
        return make_error("instance '" + inst->cfg.name + "' has no loaded context", ERROR_TYPE_NOT_FOUND);
    }
    return inst->routes->get_metrics(req);
}

json server_instances::instance_error_marker(const server_http_res_ptr & res, const std::string & name) {
    json marker = { { "instance", name } };
    try {
        const json body = json::parse(res->data);
        marker["error"] = (body.is_object() && body.contains("error")) ? body.at("error") : body;
    } catch (const std::exception &) {
        marker["error"] = res->status;
    }
    return marker;
}

std::vector<server_instances::per_instance_route_result> server_instances::collect_instance_routes(
        const server_http_req & req,
        server_http_context::handler_t server_routes::* handler) {
    std::vector<per_instance_route_result> out;
    for (const auto & inst : snapshot_instances()) {
        // guard first, then read `built`: a concurrent destroy/resize refuses here
        // instead of letting the field read race its teardown
        active_route_guard guard(*this, *inst);
        if (!guard.acquired) {
            continue;  // being destroyed/resized; not reported at all
        }
        per_instance_route_result row;
        row.inst  = inst;
        // a registered-but-undemanded window is never materialized by a listing;
        // the caller decides whether to show it as unloaded (models) or skip it (slots)
        row.built = inst->built.load(std::memory_order_acquire);
        if (!row.built) {
            out.push_back(std::move(row));
            continue;
        }
        // the hook is a test-only seam (null in production)
        server_http_res_ptr res;
        if (aggregate_route_hook) {
            if (auto forced = aggregate_route_hook(*inst, req)) {
                res = std::move(*forced);
            }
        }
        if (!res) {
            res = (inst->routes.get()->*handler)(req);
        }
        if (res->status != 200) {
            row.error_marker = instance_error_marker(res, inst->cfg.name);
        } else {
            try {
                row.body = json::parse(res->data);
            } catch (const std::exception &) {
                // a 200 whose body cannot be parsed is still a failing member
                row.error_marker = instance_error_marker(res, inst->cfg.name);
            }
        }
        out.push_back(std::move(row));
    }
    return out;
}

server_http_res_ptr server_instances::handle_get_slots(const server_http_req & req) {
    std::string model_id       = req.get_param("model");
    std::string instance_field = req.get_param("instance");

    // no target: aggregate the slots of every instance, tagged with the instance name
    if (model_id.empty() && instance_field.empty()) {
        json all_slots = json::array();
        for (auto & row : collect_instance_routes(req, &server_routes::get_slots)) {
            if (!row.built) {
                continue;  // unbuilt windows have no slots yet; listing must not build them
            }
            if (!row.body) {
                // a dying member must not fail the whole aggregate: skip it with a
                // marker row carrying the instance (router-visible, never silent)
                all_slots.push_back(std::move(row.error_marker));
                continue;
            }
            for (auto & slot : *row.body) {
                slot["instance"] = row.inst->cfg.name;
                all_slots.push_back(std::move(slot));
            }
        }
        return make_ok(all_slots);
    }

    std::string          error;
    const resolve_target target = resolve(model_id, instance_field, error);
    if (target.kind != target_kind::INSTANCE) {
        return make_error(error.empty() ? "invalid instance for slots" : error, ERROR_TYPE_INVALID_REQUEST);
    }
    if (auto err = ensure_built_instance(target.inst)) {
        return err;
    }
    active_route_guard guard(*this, *target.inst);
    if (!guard.acquired) {
        return make_error("instance '" + target.inst->cfg.name + "' is being reconfigured", ERROR_TYPE_UNAVAILABLE);
    }
    return target.inst->routes->get_slots(req);
}

server_http_res_ptr server_instances::handle_post_slots(const server_http_req & req) {
    // resolve the target instance from an optional model/instance field in the body,
    // defaulting to the default instance
    std::string model_id;
    std::string instance_field;
    try {
        json body = json::parse(req.body);
        if (body.is_object()) {
            model_id       = json_value(body, "model", std::string());
            instance_field = json_value(body, "instance", std::string());
        }
    } catch (const std::exception &) {
        // a malformed body is reported by the instance's own handler
    }
    if (model_id.empty()) {
        model_id = req.get_param("model");
    }
    if (instance_field.empty()) {
        instance_field = req.get_param("instance");
    }

    std::string          error;
    const resolve_target target = resolve(model_id, instance_field, error);
    if (target.kind != target_kind::INSTANCE) {
        return make_error(error.empty() ? "invalid instance for slot action" : error, ERROR_TYPE_INVALID_REQUEST);
    }
    if (auto err = ensure_built_instance(target.inst)) {
        return err;
    }
    active_route_guard guard(*this, *target.inst);
    if (!guard.acquired) {
        return make_error("instance '" + target.inst->cfg.name + "' is being reconfigured", ERROR_TYPE_UNAVAILABLE);
    }
    return target.inst->routes->post_slots(req);
}

server_http_res_ptr server_instances::handle_get_props(const server_http_req & req) {
    auto inst = default_instance();
    if (!inst) {
        return make_error("no instances loaded", ERROR_TYPE_SERVER);
    }
    if (auto err = ensure_built_instance(inst)) {
        return err;
    }
    active_route_guard guard(*this, *inst);
    if (!guard.acquired) {
        return make_error("instance '" + inst->cfg.name + "' is being reconfigured", ERROR_TYPE_UNAVAILABLE);
    }
    auto res = inst->routes->get_props(req);
    if (res->status != 200) {
        return res;
    }
    try {
        json                        props         = json::parse(res->data);
        int                         total_slots   = 0;
        json                        instances_arr = json::array();
        std::lock_guard<std::mutex> lock(mutex_dispatch);
        for (const auto & it : instances) {
            // the display cache is lock-free; cfg.name/group are immutable after registration.
            // memory fields are size accessors on the context (they lock only mutex_mem, an
            // independent lock, so no lock-order concern under mutex_dispatch). an unbuilt
            // window owns no buffers yet and reports zero bytes.
            const bool     is_built      = it->built.load(std::memory_order_acquire);
            const uint64_t model_bytes   = is_built ? it->ctx_server->get_model_bytes()   : 0;
            const uint64_t context_bytes = is_built ? it->ctx_server->get_context_bytes() : 0;
            const uint64_t compute_bytes = is_built ? it->ctx_server->get_compute_bytes() : 0;
            total_slots += it->n_parallel_effective.load(std::memory_order_relaxed);
            instances_arr.push_back({
                { "name",          it->cfg.name        },
                { "group",         it->cfg.group       },
                { "n_ctx",         displayed_n_ctx(*it) },
                // per-instance memory, sidecar included; matches the /instances breakdown
                { "model_bytes",   model_bytes   },
                { "context_bytes", context_bytes },
                { "compute_bytes", compute_bytes },
                { "vram_bytes",    context_bytes + compute_bytes },
                { "total_bytes",   model_bytes + context_bytes + compute_bytes },
                { "state",         is_built ? "loaded" : "unloaded" },
            });
        }
        props["total_slots"] = total_slots;
        props["instances"]   = instances_arr;
        res->data            = safe_json_to_str(props);
    } catch (const std::exception & e) {
        IST_WRN("failed to merge /props: %s\n", e.what());
    }
    return res;
}

server_http_res_ptr server_instances::handle_post_props(const server_http_req & req) {
    return default_instance_forward(req, &server_routes::post_props);
}

server_http_res_ptr server_instances::handle_post_infill(const server_http_req & req) {
    return dispatch(req, [](server_routes & routes, const server_http_req & req) { return routes.post_infill(req); });
}

server_http_res_ptr server_instances::handle_post_completions(const server_http_req & req) {
    return dispatch(req,
                    [](server_routes & routes, const server_http_req & req) { return routes.post_completions(req); });
}

server_http_res_ptr server_instances::handle_post_completions_oai(const server_http_req & req) {
    return dispatch(
        req, [](server_routes & routes, const server_http_req & req) { return routes.post_completions_oai(req); });
}

server_http_res_ptr server_instances::handle_post_chat_completions(const server_http_req & req) {
    return dispatch(
        req, [](server_routes & routes, const server_http_req & req) { return routes.post_chat_completions(req); });
}

server_http_res_ptr server_instances::handle_post_chat_completions_tok(const server_http_req & req) {
    return dispatch(
        req, [](server_routes & routes, const server_http_req & req) { return routes.post_chat_completions_tok(req); });
}

server_http_res_ptr server_instances::handle_post_control(const server_http_req & req) {
    return default_instance_forward(req, &server_routes::post_control);
}

server_http_res_ptr server_instances::handle_post_responses_oai(const server_http_req & req) {
    return dispatch(req,
                    [](server_routes & routes, const server_http_req & req) { return routes.post_responses_oai(req); });
}

server_http_res_ptr server_instances::handle_post_responses_tok_oai(const server_http_req & req) {
    return dispatch(
        req, [](server_routes & routes, const server_http_req & req) { return routes.post_responses_tok_oai(req); });
}

server_http_res_ptr server_instances::handle_post_transcriptions_oai(const server_http_req & req) {
    return dispatch(
        req, [](server_routes & routes, const server_http_req & req) { return routes.post_transcriptions_oai(req); });
}

server_http_res_ptr server_instances::handle_post_anthropic_messages(const server_http_req & req) {
    return dispatch(
        req, [](server_routes & routes, const server_http_req & req) { return routes.post_anthropic_messages(req); });
}

server_http_res_ptr server_instances::handle_post_anthropic_count_tokens(const server_http_req & req) {
    return dispatch(req, [](server_routes & routes, const server_http_req & req) {
        return routes.post_anthropic_count_tokens(req);
    });
}

server_http_res_ptr server_instances::handle_post_apply_template(const server_http_req & req) {
    return default_instance_forward(req, &server_routes::post_apply_template);
}

server_http_res_ptr server_instances::handle_get_models(const server_http_req & req) {
    // one entry per instance: reuse each instance's get_models handler and merge, keeping
    // the legacy /models shape ({"models": [...], "object": "list", "data": [...]})
    json models = json::array();
    json data   = json::array();
    for (auto & row : collect_instance_routes(req, &server_routes::get_models)) {
        // a registered-but-undemanded window is listed without building it: a mere
        // listing must never materialize a context. the display caches are lock-free
        if (!row.built) {
            data.push_back({
                { "id",       instance_id(*row.inst)    },
                { "n_ctx",    displayed_n_ctx(*row.inst) },
                { "parallel", row.inst->n_parallel_effective.load(std::memory_order_relaxed) },
                { "status",   "unloaded"                },
            });
            continue;
        }
        // a failing built member no longer fails the whole aggregate: emit a marker
        // row carrying the instance (router-visible, never silent)
        if (!row.body) {
            data.push_back(std::move(row.error_marker));
            continue;
        }
        for (auto & m : (*row.body)["models"]) {
            models.push_back(std::move(m));
        }
        for (auto & d : (*row.body)["data"]) {
            d["n_ctx"]    = displayed_n_ctx(*row.inst);
            d["parallel"] = row.inst->n_parallel_effective.load(std::memory_order_relaxed);
            d["status"]   = "loaded";
            data.push_back(std::move(d));
        }
    }
    return make_ok({
        { "models", models },
        { "object", "list" },
        { "data",   data   }
    });
}

server_http_res_ptr server_instances::handle_post_tokenize(const server_http_req & req) {
    return default_instance_forward(req, &server_routes::post_tokenize);
}

server_http_res_ptr server_instances::handle_post_detokenize(const server_http_req & req) {
    return default_instance_forward(req, &server_routes::post_detokenize);
}

server_http_res_ptr server_instances::handle_post_embeddings(const server_http_req & req) {
    return dispatch(req,
                    [](server_routes & routes, const server_http_req & req) { return routes.post_embeddings(req); });
}

server_http_res_ptr server_instances::handle_post_embeddings_oai(const server_http_req & req) {
    return dispatch(
        req, [](server_routes & routes, const server_http_req & req) { return routes.post_embeddings_oai(req); });
}

server_http_res_ptr server_instances::handle_post_rerank(const server_http_req & req) {
    return dispatch(req, [](server_routes & routes, const server_http_req & req) { return routes.post_rerank(req); });
}

// A decision request that carries a live-session reference must be pinned to the instance that
// owns the slot: the retained turn lives in one context and is not portable to another. Only a
// body/query slot or session handle marks a request as session-pinned; a malformed body is
// reported by the owning instance's handler, never here.
// does the ORIGINAL request (before handle_post_decision stamps a pool id) name a routing
// target? a request is targeted when it carries instance / snapshot / session / slot fields, or
// a model field that is not the bare pool id and not a Jev alias. a bare pool id or a Jev alias
// is echo-only: it never decides placement, so it is not a target.
static bool decision_request_target_specified(const server_http_req & req, const std::string & base_name) {
    auto model_names_target = [&](const std::string & model) {
        return !model.empty() && model != base_name && model != "jev-latest" && model != "jev-preview";
    };
    try {
        const json body = json::parse(req.body);
        if (body.is_object()) {
            if (!json_value(body, "instance", std::string()).empty()) return true;
            if (!json_value(body, "snapshot", std::string()).empty()) return true;
            if (model_names_target(json_value(body, "model", std::string()))) return true;
        }
    } catch (const std::exception &) {
        // not targeted here: the instance handler reports the parse error
    }
    if (!req.get_param("instance").empty()) return true;
    if (!req.get_param("snapshot").empty()) return true;
    return model_names_target(req.get_param("model"));
}

static bool decision_request_is_session_pinned(const server_http_req & req) {
    try {
        const json body = json::parse(req.body);
        if (body.is_object()) {
            if (!json_value(body, "session_id", std::string()).empty()) {
                return true;
            }
            if (json_value(body, "id_slot", -1) >= 0) {
                return true;
            }
        }
    } catch (const std::exception &) {
        // not session-pinned here: the instance handler reports the parse error
    }
    if (!req.get_param("session_id").empty()) {
        return true;
    }
    const std::string & id_slot = req.get_param("id_slot");
    if (!id_slot.empty()) {
        try {
            return std::stoi(id_slot) >= 0;
        } catch (const std::exception &) {
        }
    }
    return false;
}

server_http_res_ptr server_instances::handle_post_decision(const server_http_req & req) {
    // the sidecar executor owns every decision, stateless and session; the pool resolves a token
    // snapshot for a session decision before routing it there (see handle_post_decision_sidecar)
    return handle_post_decision_sidecar(req);
}

// sidecar executor mode: every decision runs on the internal executor instance. the model field
// is echo-only (a decision never picks a placement target); a session-pinned request attaches an
// owned token snapshot captured from the owning instance, then routes to the sidecar.
server_http_res_ptr server_instances::handle_post_decision_sidecar(const server_http_req & req) {
    auto sidecar = get_instance(decision_sidecar_name());
    if (sidecar == nullptr) {
        return make_error("the decision sidecar executor is not registered", ERROR_TYPE_SERVER);
    }
    const auto forward = [](server_routes & routes, const server_http_req & req) { return routes.post_decision(req); };

    server_http_req routed = req;
    json body;
    try {
        body = json::parse(req.body);
    } catch (const std::exception &) {
        // a malformed body is reported by the sidecar's own handler
        return dispatch_instance(routed, sidecar, "", -1, forward);
    }
    if (body.is_object() && req.path == DECISION_JEV_PATH) {
        body[DECISION_JEV_ONLY_KEY] = true;
        routed.body                  = body.dump();
    }
    if (body.is_object() && decision_request_is_session_pinned(req)) {
        // the session fields may arrive in the query string (as chat routing allows); merge them
        // into the body so the snapshot resolver and the sidecar parser see one source
        if (json_value(body, "session_id", std::string()).empty()) {
            const std::string qsid = req.get_param("session_id");
            if (!qsid.empty()) {
                body["session_id"] = qsid;
            }
        }
        if (json_value(body, "id_slot", -1) < 0) {
            const std::string qslot = req.get_param("id_slot");
            if (!qslot.empty()) {
                try {
                    body["id_slot"] = std::stoi(qslot);
                } catch (const std::exception &) {
                }
            }
        }
        std::pair<std::string, int> lease_key;
        if (server_http_res_ptr err = attach_decision_snapshot(body, routed, lease_key)) {
            return err;
        }
        // dispatch, then release the store lease once the (possibly cancelled) task is done
        auto res = dispatch_instance(routed, sidecar, "", -1, forward);
        std::string snap_key;
        try {
            const json rb = json::parse(routed.body);
            snap_key = rb.value("__decision_snapshot_key", std::string());
        } catch (const std::exception &) {
        }
        release_decision_snapshot_after_dispatch(snap_key, lease_key);
        return res;
    }
    // stateless: the model field is echo-only and the sidecar parser validates it; the executor
    // is reached directly, so nothing needs to be rewritten for routing
    return dispatch_instance(routed, sidecar, "", -1, forward);
}

server_http_res_ptr server_instances::handle_post_session(const server_http_req & req) {
    if (params.decision_sidecar) {
        return handle_post_session_sidecar(req);
    }
    dispatch_options opt;
    opt.require_instance = true;
    return dispatch(req, [](server_routes & routes, const server_http_req & req) { return routes.post_session(req); }, opt);
}

server_http_res_ptr server_instances::handle_get_session(const server_http_req & req) {
    if (params.decision_sidecar) {
        return handle_get_session_sidecar(req);
    }
    dispatch_options opt;
    opt.require_instance = true;
    return dispatch(req, [](server_routes & routes, const server_http_req & req) { return routes.get_session(req); }, opt);
}

server_http_res_ptr server_instances::handle_delete_session(const server_http_req & req) {
    if (params.decision_sidecar) {
        return handle_delete_session_sidecar(req);
    }
    dispatch_options opt;
    opt.require_instance = true;
    return dispatch(req, [](server_routes & routes, const server_http_req & req) { return routes.delete_session(req); }, opt);
}

server_http_res_ptr server_instances::handle_patch_session(const server_http_req & req) {
    if (params.decision_sidecar) {
        return handle_patch_session_sidecar(req);
    }
    dispatch_options opt;
    opt.require_instance = true;
    return dispatch(req, [](server_routes & routes, const server_http_req & req) { return routes.patch_session(req); }, opt);
}

// --- sidecar executor session handlers (token-snapshot store, pool-level) ---

server_http_res_ptr server_instances::handle_post_session_sidecar(const server_http_req & req) {
    json body;
    try {
        body = json::parse(req.body);
    } catch (const common_json_error & e) {
        return make_error("invalid JSON: " + std::string(e.what()), ERROR_TYPE_INVALID_REQUEST);
    }
    if (!body.is_object() || !body.contains("id_slot")) {
        return make_error("a session create needs an id_slot", ERROR_TYPE_INVALID_REQUEST);
    }
    const int id_slot = body.value("id_slot", -1);
    if (id_slot < 0) {
        return make_error("id_slot must be non-negative", ERROR_TYPE_INVALID_REQUEST);
    }
    // the owning instance must be named (a group is refused: the slot lives in one context)
    std::shared_ptr<server_instance> inst;
    std::string error;
    if (server_http_res_ptr err = decision_owning_instance(
            body.value("instance", std::string()), body.value("model", std::string()), inst, error)) {
        return err;
    }

    // policy: the only backend is the token snapshot. clone/file were host/recurrent-state
    // backends of the removed in-context registry and are a capability refusal (501), never a
    // silent fallback. capture_on_turn_complete is accepted as always true: every create captures
    // eagerly, so there is no lazy window for cache_idle_slots to clear.
    bool    pinned = false;
    int64_t ttl_ms = 0;
    if (body.contains("policy") && body.at("policy").is_object()) {
        const json & p = body.at("policy");
        if (p.contains("backend")) {
            const std::string backend = p.at("backend").get<std::string>();
            if (backend == "clone" || backend == "file") {
                return make_error("the '" + backend + "' session backend is not supported by the sidecar "
                                  "executor; only token snapshots are", ERROR_TYPE_NOT_SUPPORTED);
            }
        }
        if (p.contains("pinned")) {
            pinned = p.at("pinned").get<bool>();
        }
        if (p.contains("ttl_ms")) {
            ttl_ms = p.at("ttl_ms").get<int64_t>();
            if (ttl_ms < 0) {
                return make_error("ttl_ms must be >= 0", ERROR_TYPE_INVALID_REQUEST_SEMANTIC);
            }
        }
        // the token store keeps one snapshot per slot, so there is no per-session turn count to
        // cap. refusing beats accepting a limit that cannot be enforced.
        if (const auto max_turns = llama_decision::read_integer(p, "max_turns", "policy.")) {
            if (*max_turns > 0) {
                return make_error("max_turns is not supported: the session store keeps one token snapshot "
                                  "per slot, so there is no per-session turn limit",
                                  ERROR_TYPE_INVALID_REQUEST_SEMANTIC);
            }
        }
    }
    // the server default --decision-session-ttl applies when the request omits it (0 = no expiry)
    if (ttl_ms == 0) {
        ttl_ms = params.decision_session_ttl_ms;
    }

    // eager read-only capture on the owning instance's scheduler (no decoded state or a still-
    // processing slot is refused by the op)
    std::unique_ptr<server_task_result_decision_snapshot> op_res;
    if (server_http_res_ptr err = decision_snapshot_op(inst, id_slot, op_res)) {
        return err;
    }
    if (op_res->tokens.empty()) {
        return make_error("id_slot " + std::to_string(id_slot) + " has no decoded state to capture",
                          ERROR_TYPE_INVALID_REQUEST_SEMANTIC);
    }

    const std::string scope_str = decision_adapter_scope_of(op_res->lora_scope);
    std::vector<common_adapter_lora_info> loras;
    if (!op_res->lora_scope.empty()) {
        std::lock_guard<std::mutex> mgmt_lock(mutex_mgmt);
        try {
            loras = resolve_decision_lora_scope(op_res->lora_scope);
        } catch (const std::exception & e) {
            return make_error(e.what(), ERROR_TYPE_INVALID_REQUEST);
        }
    }

    const std::pair<std::string, int> key = std::make_pair(inst->cfg.name, id_slot);
    std::string session_id;
    std::vector<common_adapter_lora_info> stale_loras;
    {
        std::lock_guard<std::mutex> lock(mutex_decision_sessions);
        // creating a session for a slot replaces any reference the slot already holds; its refs
        // are released after this lock is dropped (never across the store lock)
        stale_loras = finalize_decision_session_locked(key);
        // token-snapshot budget: --decision-session-budget-mb caps the total owned token bytes
        // across all retained references (0 = unlimited). an over-budget create is refused, never
        // truncated; the LRU eviction tier that frees a reference under pressure is the warm tier.
        if (params.decision_session_budget_mb > 0) {
            const size_t budget = (size_t) params.decision_session_budget_mb * 1024u * 1024u;
            size_t total = op_res->tokens.size() * sizeof(llama_token);
            for (const auto & kv : decision_sessions_) {
                if (kv.second.removed) {
                    continue;
                }
                total += kv.second.tokens.size() * sizeof(llama_token);
            }
            if (total > budget) {
                return make_error("creating this session snapshot would exceed the --decision-session-budget-mb "
                                  "budget (" + std::to_string(total) + " > " + std::to_string(budget) + " bytes); "
                                  "delete sessions or raise the budget", ERROR_TYPE_INVALID_REQUEST_SEMANTIC);
            }
        }
        decision_session_entry entry;
        entry.instance      = inst->cfg.name;
        entry.id_slot       = id_slot;
        entry.turn          = body.value("turn", std::string());
        entry.base_pos      = op_res->base_pos;
        entry.tokens        = std::move(op_res->tokens);
        entry.loras         = loras;
        entry.adapter_scope = scope_str;
        entry.content_hash  = decision_content_hash_of(entry.tokens, op_res->lora_scope);
        entry.turn_counter  = decision_slot_turns_[key];
        entry.created_ms    = now_ms();
        entry.last_used_ms  = entry.created_ms;
        entry.pinned        = pinned;
        entry.ttl_ms        = ttl_ms;
        session_id = "ses_" + std::to_string(llama_decision::fnv1a64(
            std::to_string(entry.created_ms) + decision_session_key(inst->cfg.name, id_slot))) ;
        entry.session_id = session_id;
        decision_sessions_[key] = std::move(entry);
        decision_session_index_[session_id] = key;
    }
    if (!stale_loras.empty()) {
        std::lock_guard<std::mutex> mgmt_lock(mutex_mgmt);
        release_adapter_set(stale_loras);
    }

    json out;
    out["session_id"] = session_id;
    out["id_slot"]    = id_slot;
    out["turn"]       = body.value("turn", std::string());
    out["backend"]    = "tokens";
    out["captured"]   = true;
    return make_ok(out);
}

server_http_res_ptr server_instances::handle_get_session_sidecar(const server_http_req & req) {
    const std::string sid = req.get_param("session_id");
    if (sid.empty()) {
        return make_error("missing session_id", ERROR_TYPE_INVALID_REQUEST);
    }
    json out;
    {
        std::lock_guard<std::mutex> lock(mutex_decision_sessions);
        auto idx = decision_session_index_.find(sid);
        if (idx == decision_session_index_.end()) {
            return make_error("session " + sid + " does not exist", ERROR_TYPE_NOT_FOUND);
        }
        auto it = decision_sessions_.find(idx->second);
        if (it == decision_sessions_.end()) {
            return make_error("session " + sid + " does not exist", ERROR_TYPE_NOT_FOUND);
        }
        const decision_session_entry & entry = it->second;
        out["session_id"]   = sid;
        out["id_slot"]      = entry.id_slot;
        out["instance"]     = entry.instance;
        out["turn"]         = entry.turn;
        out["backend"]      = "tokens";
        out["pinned"]       = entry.pinned;
        out["ttl_ms"]       = (long long) entry.ttl_ms;
        out["captured"]     = true;
        out["bytes"]        = (long long) (entry.tokens.size() * sizeof(llama_token));
        out["created_ms"]   = (long long) entry.created_ms;
        out["last_used_ms"] = (long long) entry.last_used_ms;
        out["counters"]     = {
            { "n_snapshots", (long long) decision_sessions_.size() },
            { "n_reuses",    0LL },
            { "n_releases",  0LL },
            { "n_sessions",  (long long) decision_sessions_.size() },
            { "bytes_total", (long long) entry.tokens.size() * sizeof(llama_token) },
        };
    }
    return make_ok(out);
}

server_http_res_ptr server_instances::handle_delete_session_sidecar(const server_http_req & req) {
    const std::string sid = req.get_param("session_id");
    if (sid.empty()) {
        return make_error("missing session_id", ERROR_TYPE_INVALID_REQUEST);
    }
    std::pair<std::string, int> key;
    bool leased = false;
    {
        std::lock_guard<std::mutex> lock(mutex_decision_sessions);
        auto idx = decision_session_index_.find(sid);
        if (idx == decision_session_index_.end()) {
            return make_error("session " + sid + " does not exist", ERROR_TYPE_NOT_FOUND);
        }
        key = idx->second;
        auto it = decision_sessions_.find(key);
        if (it == decision_sessions_.end()) {
            return make_error("session " + sid + " does not exist", ERROR_TYPE_NOT_FOUND);
        }
        if (it->second.lease_count > 0) {
            // an in-flight sidecar decision holds the snapshot; drop the handle now and let the
            // finalize (ref release) happen when the last lease is dropped by the dispatch drain
            decision_session_index_.erase(sid);
            it->second.session_id.clear();
            it->second.removed = true;
            leased = true;
        }
    }
    if (!leased) {
        erase_decision_session(key);
    }
    json out;
    out["session_id"] = sid;
    out["erased"]     = true;
    return make_ok(out);
}

server_http_res_ptr server_instances::handle_patch_session_sidecar(const server_http_req & req) {
    const std::string sid = req.get_param("session_id");
    if (sid.empty()) {
        return make_error("missing session_id", ERROR_TYPE_INVALID_REQUEST);
    }
    json body;
    try {
        body = json::parse(req.body);
    } catch (const common_json_error & e) {
        return make_error("invalid JSON: " + std::string(e.what()), ERROR_TYPE_INVALID_REQUEST);
    }
    bool set_pinned = false, pinned = false;
    bool set_ttl = false;
    int64_t ttl_ms = 0;
    if (body.contains("pinned")) {
        set_pinned = true;
        pinned     = body.at("pinned").get<bool>();
    }
    if (body.contains("ttl_ms")) {
        set_ttl = true;
        ttl_ms  = body.at("ttl_ms").get<int64_t>();
        if (ttl_ms < 0) {
            return make_error("ttl_ms must be >= 0", ERROR_TYPE_INVALID_REQUEST_SEMANTIC);
        }
    }
    if (!set_pinned && !set_ttl) {
        return make_error("patch needs a pinned or ttl_ms field", ERROR_TYPE_INVALID_REQUEST_SEMANTIC);
    }
    json out;
    {
        std::lock_guard<std::mutex> lock(mutex_decision_sessions);
        auto idx = decision_session_index_.find(sid);
        if (idx == decision_session_index_.end()) {
            return make_error("session " + sid + " does not exist", ERROR_TYPE_NOT_FOUND);
        }
        auto it = decision_sessions_.find(idx->second);
        if (it == decision_sessions_.end()) {
            return make_error("session " + sid + " does not exist", ERROR_TYPE_NOT_FOUND);
        }
        decision_session_entry & entry = it->second;
        if (set_pinned) {
            entry.pinned = pinned;
        }
        if (set_ttl) {
            entry.ttl_ms = ttl_ms;
        }
        out["session_id"] = sid;
        out["pinned"]     = entry.pinned;
        out["ttl_ms"]     = (long long) entry.ttl_ms;
    }
    return make_ok(out);
}

// --- decision session token store (sidecar executor) ---

std::string server_instances::decision_session_key(const std::string & instance, int id_slot) {
    return instance + ":" + std::to_string(id_slot);
}

std::string server_instances::decision_adapter_scope_of(const std::vector<std::pair<std::string, float>> & scope) {
    if (scope.empty()) {
        return std::string();
    }
    std::string s;
    for (const auto & p : scope) {
        s += p.first + "@" + std::to_string(p.second) + ";";
    }
    return "adapter-scope-v1:" + llama_decision::sha256_hex(s);
}

// Warm identity for the sidecar, hashed with sha256 because it keys only this store's own warm
// tier. The registry's manifest path hash (session-registry.cpp) is fnv1a64 and is deliberately a
// different function over a different string. The two never exchange hashes: a warm_tag is compared
// only against another warm_tag in this map, and a manifest hash only against another manifest hash,
// so the split is currently harmless. Unifying them would change every warm tag and invalidate the
// resident tier.


std::string server_instances::decision_content_hash_of(const std::vector<llama_token> & tokens,
                                                       const std::vector<std::pair<std::string, float>> & scope) {
    std::string bytes;
    bytes.reserve(tokens.size() * sizeof(llama_token) + 16);
    for (llama_token tok : tokens) {
        const uint32_t t = (uint32_t) tok;
        for (int b = 0; b < 4; ++b) {
            bytes.push_back((char) ((t >> (8 * b)) & 0xff));
        }
    }
    for (const auto & p : scope) {
        bytes += p.first;
        bytes.push_back('@');
        const uint32_t bits = (uint32_t) p.second;
        bytes.push_back((char) (bits & 0xff));
        bytes.push_back((char) ((bits >> 8) & 0xff));
        bytes.push_back((char) ((bits >> 16) & 0xff));
        bytes.push_back((char) ((bits >> 24) & 0xff));
    }
    return std::string("session-tokens-v1:") + llama_decision::sha256_hex(bytes);
}

std::vector<common_adapter_lora_info> server_instances::resolve_decision_lora_scope(
        const std::vector<std::pair<std::string, float>> & scope) {
    std::vector<common_adapter_lora_info> out;
    for (const auto & p : scope) {
        common_adapter_lora_info la;
        la.path  = p.first;
        la.scale = p.second;
        llama_adapter_lora * ptr = ensure_adapter(p.first);
        if (ptr == nullptr) {
            release_adapter_set(out);
            throw std::runtime_error("failed to resolve the session adapter scope: " + p.first);
        }
        la.ptr = ptr;
        out.push_back(std::move(la));
    }
    return out;
}

std::vector<common_adapter_lora_info> server_instances::finalize_decision_session_locked(
        const std::pair<std::string, int> & key) {
    auto it = decision_sessions_.find(key);
    if (it == decision_sessions_.end()) {
        return {};
    }
    decision_session_entry entry = std::move(it->second);
    decision_sessions_.erase(it);
    if (!entry.session_id.empty()) {
        decision_session_index_.erase(entry.session_id);
    }
    // the refs are released by the caller AFTER the store lock is dropped: releasing needs
    // mutex_mgmt, and this store lock may be held on a scheduler thread (a release hook) where a
    // management op waiting on this scheduler thread also holds mutex_mgmt
    return entry.loras;
}

// erase a store entry and release its adapter refs. call only from an HTTP thread (never a
// scheduler thread), and never while holding mutex_dispatch.
void server_instances::erase_decision_session(const std::pair<std::string, int> & key) {
    std::vector<common_adapter_lora_info> loras;
    {
        std::lock_guard<std::mutex> lock(mutex_decision_sessions);
        loras = finalize_decision_session_locked(key);
    }
    if (!loras.empty()) {
        std::lock_guard<std::mutex> mgmt_lock(mutex_mgmt);
        release_adapter_set(loras);
    }
}

void server_instances::on_decision_slot_release(const std::string & instance, int id_slot) {
    const auto key = std::make_pair(instance, id_slot);
    std::lock_guard<std::mutex> lock(mutex_decision_sessions);
    ++decision_slot_turns_[key];
    auto it = decision_sessions_.find(key);
    if (it == decision_sessions_.end()) {
        return;
    }
    // the session's turn is over: mark it for removal. the actual ref release happens on an HTTP
    // thread (a later resolve / erase), never on this scheduler thread, because releasing an
    // adapter ref needs mutex_mgmt which a management op holds while waiting on this very thread.
    it->second.removed = true;
}

void server_instances::decision_sessions_clear() {
    std::vector<common_adapter_lora_info> loras;
    {
        std::lock_guard<std::mutex> lock(mutex_decision_sessions);
        for (auto & kv : decision_sessions_) {
            auto & entry = kv.second;
            loras.insert(loras.end(), entry.loras.begin(), entry.loras.end());
        }
        decision_sessions_.clear();
        decision_session_index_.clear();
        decision_slot_turns_.clear();
        decision_snapshot_resolve_.clear();
    }
    if (!loras.empty()) {
        std::lock_guard<std::mutex> mgmt_lock(mutex_mgmt);
        release_adapter_set(loras);
    }
}

std::shared_ptr<server_decision_snapshot> server_instances::decision_snapshot_by_key(const std::string & key) {
    std::lock_guard<std::mutex> lock(mutex_decision_sessions);
    auto it = decision_snapshot_resolve_.find(key);
    return it != decision_snapshot_resolve_.end() ? it->second : nullptr;
}

server_http_res_ptr server_instances::decision_owning_instance(const std::string & instance_field,
                                                               const std::string & model_field,
                                                               std::shared_ptr<server_instance> & inst_out,
                                                               std::string & error) const {
    // an explicit instance wins; otherwise the model id may name one (base:NAME). a group or an
    // ambiguous target is refused: a session decision addresses one instance's slot. a bare pool
    // id (or no target) resolves to the default instance, matching stateless routing.
    std::string target = instance_field;
    if (target.empty() && !model_field.empty()) {
        const auto comps = string_split<std::string>(model_field, ':');
        if (comps.size() >= 2 && comps[0] == base_name) {
            target = comps.back();
        }
    }
    if (target.empty() || target == "latest") {
        auto inst = default_instance();
        if (inst == nullptr) {
            error = "a session decision must name the owning instance ('instance' or 'model' field)";
            return make_error(error, ERROR_TYPE_INVALID_REQUEST);
        }
        inst_out = inst;
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(mutex_dispatch);
    for (const auto & inst : instances) {
        if (inst->cfg.name == target) {
            if (inst->internal) {
                error = "the decision sidecar executor has no chat slot to snapshot";
                return make_error(error, ERROR_TYPE_INVALID_REQUEST);
            }
            inst_out = inst;
            return nullptr;
        }
        if (inst->cfg.group == target) {
            error = "this request addresses one instance's slot; route it with the 'instance' or "
                    "'model' field, not a group";
            return make_error(error, ERROR_TYPE_INVALID_REQUEST);
        }
    }
    error = "model or instance not found: '" + target + "'";
    return make_error(error, ERROR_TYPE_INVALID_REQUEST);
}

server_http_res_ptr server_instances::decision_snapshot_op(const std::shared_ptr<server_instance> & inst, int id_slot,
                                                           std::unique_ptr<server_task_result_decision_snapshot> & out) {
    if (auto err = ensure_built_instance(inst)) {
        return err;
    }
    auto res = inst->ctx_server->slot_decision_snapshot(id_slot);
    if (res == nullptr) {
        return make_error("the decision snapshot timed out on instance '" + inst->cfg.name + "'", ERROR_TYPE_UNAVAILABLE);
    }
    if (res->is_error()) {
        return make_error_from_result(*res);
    }
    auto * snap = dynamic_cast<server_task_result_decision_snapshot *>(res.get());
    if (snap == nullptr) {
        return make_error("unexpected decision snapshot result", ERROR_TYPE_SERVER);
    }
    // take ownership so the caller can read the tokens/lora scope after this function returns
    out.reset(static_cast<server_task_result_decision_snapshot *>(res.release()));
    return nullptr;
}

server_http_res_ptr server_instances::attach_decision_snapshot(const json & body, server_http_req & routed,
                                                               std::pair<std::string, int> & lease_out) {
    const std::string session_id = body.value("session_id", std::string());
    const int         id_slot    = body.value("id_slot", -1);
    const std::string instance_field = body.value("instance", std::string());
    const std::string model_field    = body.value("model", std::string());

    std::shared_ptr<server_decision_snapshot> snap = std::make_shared<server_decision_snapshot>();
    std::pair<std::string, int> key;
    bool leased = false;

    if (!session_id.empty()) {
        // first-class session: the pool store is the single owner
        std::lock_guard<std::mutex> lock(mutex_decision_sessions);
        auto idx = decision_session_index_.find(session_id);
        if (idx == decision_session_index_.end()) {
            return make_error("session " + session_id + " does not exist", ERROR_TYPE_NOT_FOUND);
        }
        key = idx->second;
        auto it = decision_sessions_.find(key);
        if (it == decision_sessions_.end()) {
            return make_error("session " + session_id + " does not exist", ERROR_TYPE_NOT_FOUND);
        }
        decision_session_entry & entry = it->second;
        if (entry.removed || decision_slot_turns_[key] != entry.turn_counter) {
            // the slot advanced past the captured turn: never answer from an old turn
            return make_error("session " + session_id + " is stale: the source slot's turn ended; "
                              "create a new session", ERROR_TYPE_INVALID_REQUEST_SEMANTIC);
        }
        if (!instance_field.empty() && instance_field != entry.instance) {
            return make_error("session " + session_id + " belongs to instance '" + entry.instance +
                              "', not '" + instance_field + "'", ERROR_TYPE_INVALID_REQUEST);
        }
        if (!model_field.empty() && model_field != base_name) {
            const auto comps = string_split<std::string>(model_field, ':');
            if (comps.size() >= 2 && comps[0] == base_name && comps.back() != entry.instance) {
                return make_error("session " + session_id + " belongs to instance '" + entry.instance +
                                  "', not '" + comps.back() + "'", ERROR_TYPE_INVALID_REQUEST);
            }
        }
        snap->tokens        = entry.tokens;
        snap->loras         = entry.loras;
        snap->adapter_scope = entry.adapter_scope;
        snap->source_slot   = entry.id_slot;
        snap->turn          = entry.turn;
        snap->session_id    = session_id;
        snap->base_pos      = entry.base_pos;
        snap->warm_tag      = entry.content_hash;
        entry.last_used_ms  = now_ms();
        ++entry.lease_count;
        leased = true;
    } else {
        // implicit id_slot session: eager capture through a read-only op on the owning instance
        std::shared_ptr<server_instance> inst;
        std::string error;
        if (server_http_res_ptr err = decision_owning_instance(instance_field, model_field, inst, error)) {
            return err;
        }
        key = std::make_pair(inst->cfg.name, id_slot);

        // reuse the eager snapshot for the current turn; otherwise capture now (no lazy window)
        std::unique_ptr<server_task_result_decision_snapshot> op_res;
        {
            std::lock_guard<std::mutex> lock(mutex_decision_sessions);
            auto it = decision_sessions_.find(key);
            if (it != decision_sessions_.end() && !it->second.removed &&
                decision_slot_turns_[key] == it->second.turn_counter) {
                decision_session_entry & entry = it->second;
                snap->tokens        = entry.tokens;
                snap->loras         = entry.loras;
                snap->adapter_scope = entry.adapter_scope;
                snap->source_slot   = id_slot;
                snap->turn          = entry.turn;
                snap->base_pos      = entry.base_pos;
                snap->warm_tag      = entry.content_hash;
                entry.last_used_ms  = now_ms();
                ++entry.lease_count;
                leased = true;
            }
        }
        if (!leased) {
            // the store mutex is never held across an instance_op
            if (server_http_res_ptr err = decision_snapshot_op(inst, id_slot, op_res)) {
                return err;
            }
            if (op_res->tokens.empty()) {
                return make_error("id_slot " + std::to_string(id_slot) + " has no decoded state to snapshot",
                                  ERROR_TYPE_INVALID_REQUEST_SEMANTIC);
            }
            const std::string scope_str = decision_adapter_scope_of(op_res->lora_scope);
            std::vector<common_adapter_lora_info> loras;
            if (!op_res->lora_scope.empty()) {
                std::lock_guard<std::mutex> mgmt_lock(mutex_mgmt);
                try {
                    loras = resolve_decision_lora_scope(op_res->lora_scope);
                } catch (const std::exception & e) {
                    return make_error(e.what(), ERROR_TYPE_INVALID_REQUEST);
                }
            }
            std::vector<common_adapter_lora_info> stale_loras_out;
            {
                std::lock_guard<std::mutex> lock(mutex_decision_sessions);
                // a slot release may have advanced the turn while the op ran; drop the stale entry
                // (its refs are released after this lock is dropped, never across the store lock)
                std::vector<common_adapter_lora_info> stale_loras;
                auto it = decision_sessions_.find(key);
                if (it != decision_sessions_.end() && it->second.removed) {
                    stale_loras = finalize_decision_session_locked(key);
                }
                decision_session_entry entry;
                // token-snapshot budget: the capture path holds the same --decision-session-budget-mb
                // cap as the create path, so a client that only ever sends id_slot-pinned decisions
                // cannot grow the store past it. The entry at this key is about to be replaced, so it
                // is excluded from the total rather than counted twice.
                if (params.decision_session_budget_mb > 0) {
                    const size_t budget = (size_t) params.decision_session_budget_mb * 1024u * 1024u;
                    size_t total = op_res->tokens.size() * sizeof(llama_token);
                    for (const auto & kv : decision_sessions_) {
                        if (kv.second.removed || kv.first == key) {
                            continue;
                        }
                        total += kv.second.tokens.size() * sizeof(llama_token);
                    }
                    if (total > budget) {
                        return make_error("capturing this id_slot snapshot would exceed the --decision-session-budget-mb "
                                          "budget (" + std::to_string(total) + " > " + std::to_string(budget) + " bytes); "
                                          "delete sessions or raise the budget", ERROR_TYPE_INVALID_REQUEST_SEMANTIC);
                    }
                }
                entry.instance       = inst->cfg.name;
                entry.id_slot        = id_slot;
                entry.base_pos       = op_res->base_pos;
                entry.tokens         = std::move(op_res->tokens);
                entry.loras          = loras;
                entry.adapter_scope  = scope_str;
                entry.content_hash   = decision_content_hash_of(entry.tokens, op_res->lora_scope);
                entry.turn_counter   = decision_slot_turns_[key];
                entry.created_ms     = now_ms();
                entry.last_used_ms   = entry.created_ms;
                decision_sessions_[key] = std::move(entry);
                decision_session_entry & stored = decision_sessions_[key];
                snap->tokens        = stored.tokens;
                snap->loras         = stored.loras;
                snap->adapter_scope = stored.adapter_scope;
                snap->source_slot   = id_slot;
                snap->base_pos      = stored.base_pos;
                snap->warm_tag      = stored.content_hash;
                ++stored.lease_count;
                leased = true;
                if (!stale_loras.empty()) {
                    stale_loras_out = std::move(stale_loras);
                }
            }
            if (!stale_loras_out.empty()) {
                std::lock_guard<std::mutex> mgmt_lock(mutex_mgmt);
                release_adapter_set(stale_loras_out);
            }
        }
    }
    if (!leased) {
        return make_error("the decision session could not be resolved", ERROR_TYPE_SERVER);
    }

    // embed a transient resolve key so the sidecar route can fetch the owned snapshot; the pool
    // holds the lease (and the adapter refs) until release_decision_snapshot_after_dispatch
    char keybuf[32];
    std::snprintf(keybuf, sizeof(keybuf), "snp_%08llx", (unsigned long long) (llama_decision::fnv1a64(
        std::to_string(now_ms()) + decision_session_key(key.first, key.second))));
    const std::string snap_key = keybuf;
    {
        std::lock_guard<std::mutex> lock(mutex_decision_sessions);
        decision_snapshot_resolve_[snap_key] = snap;
    }
    lease_out = key;
    json rb = body;
    // the decision runs on the sidecar executor whatever instance the client named: the source
    // instance has no resolver and must not decode the snapshot. the model field is echo-only and
    // stays untouched, so the sidecar's contract parser (which requires a model) and its echo both
    // see the caller's value.
    rb["instance"] = decision_sidecar_name();
    rb["__decision_snapshot_key"] = snap_key;
    routed.body = rb.dump();
    return nullptr;
}

// After a sidecar dispatch: wait for the sidecar's scheduler to fully finish the (possibly
// cancelled) decision task, then drop the lease on the store entry so its adapter refs can be
// released. The FIFO sync op runs after the decision task, so its return proves the task is done
// even when the HTTP side returned early on a client cancel.
void server_instances::release_decision_snapshot_after_dispatch(const std::string & snap_key,
                                                                const std::pair<std::string, int> & entry_key) {
    auto sidecar = get_instance(decision_sidecar_name());
    if (sidecar != nullptr && sidecar->built.load(std::memory_order_acquire)) {
        sidecar->ctx_server->instance_op([]() { return json{{"drain", true}}; });
    }
    std::vector<common_adapter_lora_info> removed_loras;
    {
        std::lock_guard<std::mutex> lock(mutex_decision_sessions);
        decision_snapshot_resolve_.erase(snap_key);
        auto it = decision_sessions_.find(entry_key);
        if (it == decision_sessions_.end()) {
            return;
        }
        decision_session_entry & entry = it->second;
        if (entry.lease_count > 0) {
            --entry.lease_count;
        }
        if (entry.removed && entry.lease_count == 0) {
            removed_loras = finalize_decision_session_locked(entry_key);
        }
    }
    if (!removed_loras.empty()) {
        std::lock_guard<std::mutex> mgmt_lock(mutex_mgmt);
        release_adapter_set(removed_loras);
    }
}

server_http_res_ptr server_instances::handle_get_lora_adapters(const server_http_req & req) {
    return default_instance_forward(req, &server_routes::get_lora_adapters);
}

server_http_res_ptr server_instances::handle_post_lora_adapters(const server_http_req & req) {
    // the legacy writer commits to the scheduler through the stock handler (its
    // body grammar and zero-for-unlisted scale semantics stay the only
    // implementation). the manager mirror is then refreshed from the scheduler,
    // so the next attach does not re-apply a stale scale. the drain is exclusive:
    // no dispatch may read the effective list (apply_snapshot fingerprint) while
    // the refresh below rewrites it, so the write is never torn.
    std::lock_guard<std::mutex> mgmt_lock(mutex_mgmt);

    auto inst = default_instance();
    if (!inst) {
        return make_error("no instances loaded", ERROR_TYPE_NOT_FOUND);
    }
    // the legacy grammar needs a scheduler to apply to; an unbuilt default is the
    // same tier as save-to-unbuilt (404)
    if (!inst->built.load(std::memory_order_acquire)) {
        return make_error("instance '" + inst->cfg.name + "' has no loaded context", ERROR_TYPE_NOT_FOUND);
    }
    // like swap_adapter_set: management ops are serialized by mutex_mgmt, so no
    // two drains contend; the scheduler posts run while removing (direct queue)
    instance_drain_guard guard(*this, inst);
    auto res = inst->routes->post_lora_adapters(req);
    if (res->status != 200) {
        return res;
    }
    if (!refresh_effective_from_scheduler(*inst, snapshot_deadline_ms(*inst))) {
        return make_error("could not read the committed adapter set", ERROR_TYPE_UNAVAILABLE);
    }
    // the scale is part of the snapshot fingerprint: bound slots would otherwise
    // keep serving KV computed under the old adapters
    {
        std::lock_guard<std::mutex> lock(mutex_dispatch);
        inst->slot_snapshots.assign(inst->slot_snapshots.size(), std::string());
    }
    return res;
}

//
// management API handlers
//

server_http_res_ptr server_instances::handle_get_instances(const server_http_req &) {
    return make_ok(get_instances_json());
}

server_http_res_ptr server_instances::handle_post_instances(const server_http_req & req) {
    json body;
    try {
        body = json::parse(req.body);
    } catch (const std::exception &) {
        return make_error("invalid JSON body", ERROR_TYPE_INVALID_REQUEST);
    }
    if (!body.is_object()) {
        return make_error("body must be a JSON object", ERROR_TYPE_INVALID_REQUEST);
    }

    common_instance cfg;
    cfg.name       = json_value(body, "name", std::string());
    cfg.group      = json_value(body, "group", std::string());
    cfg.ctx_size   = json_value(body, "ctx_size", 0);
    cfg.parallel   = json_value(body, "parallel", 0);
    cfg.pinned     = json_value(body, "pinned", false);
    cfg.is_default = json_value(body, "default", false);

    if (cfg.name.empty()) {
        return make_error("'name' is required", ERROR_TYPE_INVALID_REQUEST);
    }
    if (cfg.group.empty()) {
        cfg.group = cfg.name;
    }
    if (cfg.ctx_size < 0 || cfg.parallel < 0) {
        return make_error("'ctx_size' and 'parallel' must be non-negative", ERROR_TYPE_INVALID_REQUEST);
    }

    try {
        common_instance_validate(cfg);
    } catch (const std::invalid_argument & e) {
        return make_error(e.what(), ERROR_TYPE_INVALID_REQUEST);
    }

    return create_instance(cfg);
}

server_http_res_ptr server_instances::handle_post_instance_pin(const server_http_req & req) {
    return set_instance_pinned(req.get_param("name"), true);
}

server_http_res_ptr server_instances::handle_post_instance_unpin(const server_http_req & req) {
    return set_instance_pinned(req.get_param("name"), false);
}

server_http_res_ptr server_instances::handle_post_instance_resize(const server_http_req & req) {
    json body;
    try {
        body = json::parse(req.body);
    } catch (const std::exception &) {
        return make_error("invalid JSON body", ERROR_TYPE_INVALID_REQUEST);
    }
    const int32_t new_ctx = json_value(body, "ctx_size", 0);
    return resize_instance(req.get_param("name"), new_ctx);
}

server_http_res_ptr server_instances::handle_delete_instance(const server_http_req & req) {
    // force is accepted for compatibility and ignored (nothing enforces pinned)
    return destroy_instance(req.get_param("name"), !req.get_param("force").empty());
}

bool server_instances::refresh_effective_from_scheduler(server_instance & inst, int64_t deadline_ms) {
    if (!inst.ctx_server) {
        return false;
    }
    auto committed = inst.ctx_server->get_lora_adapters(deadline_ms);
    if (!committed) {
        return false;
    }
    // one committed list feeds both forms: the effective list (ptrs, meta) and
    // the grammar list (path + scale) can therefore never drift apart
    inst.effective.lora_adapters = std::move(*committed);
    inst.cfg.lora.clear();
    inst.cfg.lora.reserve(inst.effective.lora_adapters.size());
    for (const auto & la : inst.effective.lora_adapters) {
        inst.cfg.lora.emplace_back(la.path, la.scale);
    }
    return true;
}

server_http_res_ptr server_instances::swap_adapter_set(const std::shared_ptr<server_instance> & inst,
                                                       const adapter_set_mutator &               mutate) {
    // exclusive access while the set changes: the drain rejects new dispatches and
    // waits out in-flight generations, so the swap below never frees adapter tensors
    // under a decoding slot (use-after-free) nor corrupts its output. the drain
    // comes before the mutation because apply_snapshot reads the effective list
    // without the management lock while its dispatch is counted.
    instance_drain_guard guard(*this, inst);

    // the scheduler owns the live set (a legacy POST writes it directly), so the
    // mirror is refreshed before the mutation: an attach/detach is then additive
    // over what the scheduler actually holds instead of reverting a legacy scale.
    if (!refresh_effective_from_scheduler(*inst, snapshot_deadline_ms(*inst))) {
        return make_error("could not read the committed adapter set", ERROR_TYPE_UNAVAILABLE);
    }

    // rollback copies: the lists stay untouched until the synchronous apply succeeds
    const auto cfg_prev       = inst->cfg.lora;
    const auto effective_prev = inst->effective.lora_adapters;

    adapter_set_delta delta;
    if (auto err = mutate(inst->cfg.lora, inst->effective.lora_adapters, delta)) {
        return err;
    }

    // synchronous apply, posted directly to this instance's queue (never through
    // dispatch(), which the guard's removing flag would reject). the drain makes
    // the scheduler quiescent, so the task runs immediately; the deadline only
    // trips on a stalled scheduler.
    auto result = inst->ctx_server->set_lora_adapters(inst->effective.lora_adapters, snapshot_deadline_ms(*inst));
    if (!result || result->is_error()) {
        inst->cfg.lora                = cfg_prev;
        inst->effective.lora_adapters = effective_prev;
        release_adapter_set(delta.acquired);
        return make_error("timed out applying the adapter set", ERROR_TYPE_UNAVAILABLE);
    }
    // read-back agreement: the committed set must equal the expected list.
    // no rollback here: the set above succeeded, so the lists already describe
    // the scheduler; a failed read-back is answered retriable and the idempotent
    // retry re-verifies.
    if (!verify_attached(*inst, snapshot_deadline_ms(*inst))) {
        return make_error("could not verify the adapter set", ERROR_TYPE_UNAVAILABLE);
    }
    release_adapter_set(delta.released); // frees at refcount 0 while the model is alive

    // the swap invalidated every slot's live KV (the scheduler cleared the prompts)
    // AND the recorded bindings: bound slots would otherwise keep serving KV computed
    // under the old adapters (resize semantics)
    {
        std::lock_guard<std::mutex> lock(mutex_dispatch);
        inst->slot_snapshots.assign(inst->slot_snapshots.size(), std::string());
    }
    return nullptr;
}

server_http_res_ptr server_instances::handle_post_instance_adapters(const server_http_req & req) {
    json body;
    try {
        body = json::parse(req.body);
    } catch (const std::exception &) {
        return make_error("invalid JSON body", ERROR_TYPE_INVALID_REQUEST);
    }
    if (!body.is_object()) {
        return make_error("body must be a JSON object", ERROR_TYPE_INVALID_REQUEST);
    }
    const std::string path  = json_value(body, "path", std::string());
    const float         scale = json_value(body, "scale", 1.0f);
    if (path.empty()) {
        return make_error("'path' is required", ERROR_TYPE_INVALID_REQUEST);
    }
    // no separator restriction here: the body is JSON, so ':' and ',' need no
    // escaping (the --instance grammar still rejects them as separators).
    if (!(scale > 0.0f) || !std::isfinite(scale)) {
        return make_error("'scale' must be a positive finite number", ERROR_TYPE_INVALID_REQUEST);
    }
    // the message carries no echo of the path: a malformed path may hold
    // bytes (such as NUL) that truncate log lines and confuse readers
    const auto key_opt = try_adapter_key(path);
    if (!key_opt) {
        return make_error("invalid adapter path", ERROR_TYPE_INVALID_REQUEST);
    }
    const std::string key = *key_opt;

    // serialized with the other management ops; dispatches never take this mutex, so
    // the drain below cannot deadlock (resize precedent, same lock order)
    std::lock_guard<std::mutex> mgmt_lock(mutex_mgmt);

    auto inst = get_instance(req.get_param("name"));
    if (!inst) {
        return make_error("instance not found: '" + req.get_param("name") + "'", ERROR_TYPE_NOT_FOUND);
    }

    // an unbuilt window has no scheduler to apply to: record the declaration, it
    // resolves (takes refs) when the window materializes. matches the resize-on-
    // unbuilt precedent, and avoids forcing a window allocation for a bookkeeping op.
    if (!inst->built.load(std::memory_order_acquire)) {
        bool present = false;
        for (auto & gl : inst->cfg.lora) {
            if (adapter_key(gl.first) == key) {
                gl = { key, scale };
                present = true;
            }
        }
        if (!present) {
            inst->cfg.lora.emplace_back(key, scale);
        }
        inst->effective.lora_adapters.clear();
        for (const auto & gl : inst->cfg.lora) {
            inst->effective.lora_adapters.push_back({ gl.first, gl.second, "", "", nullptr });
        }
        return make_ok(instance_to_json(*inst));
    }

    // a built window swaps through the shared core: the mutator below only
    // edits the two lists (upsert + scale), everything else lives in one place
    adapter_set_mutator upsert = [&](std::vector<std::pair<std::string, float>> & cfg_lora,
                                     std::vector<common_adapter_lora_info> &     effective,
                                     adapter_set_delta &                         delta) -> server_http_res_ptr {
        bool present = false;
        for (auto & la : effective) {
            if (adapter_key(la.path) == key) {
                la.scale = scale; // re-attach updates the scale; the ref is already held
                present  = true;
            }
        }
        if (!present) {
            llama_adapter_lora * ptr = ensure_adapter(key);
            if (ptr == nullptr) {
                return make_error("failed to load lora adapter '" + key + "'", ERROR_TYPE_INVALID_REQUEST);
            }
            delta.acquired.push_back({ key, scale, "", "", ptr });
            effective.push_back({ key, scale, "", "", ptr });
        }
        // cfg.lora tracks the same set in grammar form; matched independently so the
        // two lists can never diverge even if a previous op left them inconsistent
        bool cfg_present = false;
        for (auto & gl : cfg_lora) {
            if (adapter_key(gl.first) == key) {
                gl           = { key, scale };
                cfg_present  = true;
            }
        }
        if (!cfg_present) {
            cfg_lora.emplace_back(key, scale);
        }
        return nullptr;
    };
    if (auto err = swap_adapter_set(inst, upsert)) {
        return err;
    }

    IST_INF("instance '%s' attached adapter '%s' (scale = %g)\n", inst->cfg.name.c_str(), key.c_str(), scale);
    return make_ok(instance_to_json(*inst));
}

server_http_res_ptr server_instances::handle_delete_instance_adapters(const server_http_req & req) {
    // filesystem paths contain '/', so no :path URL segment can hold them: the path
    // arrives in the JSON body, with ?path= accepted as an alias for empty bodies
    std::string path = req.get_param("path");
    if (!req.body.empty()) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (const std::exception &) {
            return make_error("invalid JSON body", ERROR_TYPE_INVALID_REQUEST);
        }
        if (!body.is_object()) {
            return make_error("body must be a JSON object", ERROR_TYPE_INVALID_REQUEST);
        }
        const std::string body_path = json_value(body, "path", std::string());
        if (!body_path.empty()) {
            path = body_path;
        }
    }
    if (path.empty()) {
        return make_error("'path' is required (body or ?path= query)", ERROR_TYPE_INVALID_REQUEST);
    }
    // same boundary as attach: a malformed path is a caller bug (400),
    // answered before any instance lookup, drain, or registry touch
    const auto key_opt = try_adapter_key(path);
    if (!key_opt) {
        return make_error("invalid adapter path", ERROR_TYPE_INVALID_REQUEST);
    }
    const std::string key = *key_opt;

    std::lock_guard<std::mutex> mgmt_lock(mutex_mgmt);

    auto inst = get_instance(req.get_param("name"));
    if (!inst) {
        return make_error("instance not found: '" + req.get_param("name") + "'", ERROR_TYPE_NOT_FOUND);
    }

    // an unbuilt window has no scheduler and no in-flight dispatch can be inside
    // it: mutate the declaration lists directly (no drain needed).
    if (!inst->built.load(std::memory_order_acquire)) {
        std::vector<common_adapter_lora_info> removed;
        {
            auto & eff = inst->effective.lora_adapters;
            for (auto it = eff.begin(); it != eff.end();) {
                if (adapter_key(it->path) == key) {
                    removed.push_back(*it);
                    it = eff.erase(it);
                } else {
                    ++it;
                }
            }
        }
        if (removed.empty()) {
            return make_error("adapter not attached to instance '" + inst->cfg.name + "'", ERROR_TYPE_NOT_FOUND);
        }
        {
            auto & cfg_lora = inst->cfg.lora;
            for (auto it = cfg_lora.begin(); it != cfg_lora.end();) {
                if (adapter_key(it->first) == key) {
                    it = cfg_lora.erase(it);
                } else {
                    ++it;
                }
            }
        }
        release_adapter_set(removed); // no-op: unbuilt entries hold no refs
        IST_INF("instance '%s' detached adapter '%s'\n", inst->cfg.name.c_str(), key.c_str());
        return make_ok({
            { "success", true }
        });
    }

    // fast 404 before the drain: an adapter that was never attached stalls no
    // traffic and holds the management plane only for the scan below. the
    // scan inside the drain stays authoritative: a racing attach can only add
    // the key (turning this into a success), and management ops already
    // serialize on the management lock, so the two scans cannot disagree.
    {
        bool attached = false;
        for (const auto & la : inst->effective.lora_adapters) {
            if (adapter_key(la.path) == key) {
                attached = true;
                break;
            }
        }
        if (!attached) {
            return make_error("adapter not attached to instance '" + inst->cfg.name + "'", ERROR_TYPE_NOT_FOUND);
        }
    }

    // a built window swaps through the shared core: the mutator below only
    // edits the two lists (erase), everything else lives in one place. the
    // peek above already 404s the never-attached case, so the miss return
    // below is unreachable in practice, kept as the authoritative check.
    adapter_set_mutator erase = [&](std::vector<std::pair<std::string, float>> & cfg_lora,
                                    std::vector<common_adapter_lora_info> &     effective,
                                    adapter_set_delta &                         delta) -> server_http_res_ptr {
        // remove the key from both lists (they are kept in sync); the removed
        // entries carry the refs to drop on success
        for (auto it = effective.begin(); it != effective.end();) {
            if (adapter_key(it->path) == key) {
                delta.released.push_back(*it);
                it = effective.erase(it);
            } else {
                ++it;
            }
        }
        if (delta.released.empty()) {
            return make_error("adapter not attached to instance '" + inst->cfg.name + "'", ERROR_TYPE_NOT_FOUND);
        }
        for (auto it = cfg_lora.begin(); it != cfg_lora.end();) {
            if (adapter_key(it->first) == key) {
                it = cfg_lora.erase(it);
            } else {
                ++it;
            }
        }
        return nullptr;
    };
    if (auto err = swap_adapter_set(inst, erase)) {
        return err;
    }

    IST_INF("instance '%s' detached adapter '%s'\n", inst->cfg.name.c_str(), key.c_str());
    return make_ok({
        { "success", true }
    });
}

server_http_res_ptr server_instances::handle_get_instance_adapters(const server_http_req & req) {
    // reader side of the locking invariant: the effective list mutates under mutex_mgmt
    std::lock_guard<std::mutex> mgmt_lock(mutex_mgmt);

    auto inst = get_instance(req.get_param("name"));
    if (!inst) {
        return make_error("instance not found: '" + req.get_param("name") + "'", ERROR_TYPE_NOT_FOUND);
    }

    json adapters = json::array();
    for (const auto & la : inst->effective.lora_adapters) {
        adapters.push_back({
            { "path",  la.path  },
            { "scale", la.scale },
        });
    }
    return make_ok(adapters);
}

server_http_res_ptr server_instances::handle_post_instance_snapshot(const server_http_req & req) {
    // POST /instances/:name/snapshot  body { "name": "<snapshot>", "id_slot": N? }
    // saves one slot's KV into the instance's own snapshot namespace
    // (<slot_save_path>/<model_key>/<instance>/<snapshot>.bin). id_slot selects
    // the slot (default 0); snapshots are per-instance, never silently slot 0
    // of another instance's namespace. serialized with resize/destroy so the
    // slot-save task on the scheduler never races a context rebuild that
    // destroys the very context it is copying from
    std::lock_guard<std::mutex> mgmt_lock(mutex_mgmt);

    const std::string name = req.get_param("name");
    json              body;
    try {
        body = json::parse(req.body);
    } catch (const std::exception &) {
        return make_error("invalid JSON body", ERROR_TYPE_INVALID_REQUEST);
    }
    const std::string snapshot = json_value(body, "name", std::string());
    const int         id_slot  = json_value(body, "id_slot", 0);
    if (!fs_validate_filename(snapshot)) {
        return make_error("invalid snapshot name", ERROR_TYPE_INVALID_REQUEST);
    }
    if (params.slot_save_path.empty()) {
        return make_error("snapshot management requires --slot-save-path", ERROR_TYPE_NOT_SUPPORTED);
    }

    auto inst = get_instance(name);
    if (!inst) {
        return make_error("instance not found: '" + name + "'", ERROR_TYPE_NOT_FOUND);
    }
    // saving needs live KV; unlike a generation request, a save must not materialize
    // a window as a side effect
    if (!inst->built.load(std::memory_order_acquire)) {
        return make_error("instance '" + name + "' has no loaded context", ERROR_TYPE_NOT_FOUND);
    }
    if (id_slot < 0 || (size_t) id_slot >= inst->slot_snapshots.size()) {
        return make_error("invalid slot id", ERROR_TYPE_INVALID_REQUEST);
    }

    const std::string dir = server_snapshot_instance_dir(params.slot_save_path,
                                                         server_snapshot_model_key_hashed(base_name), name);
    std::error_code   ec;
    std::filesystem::create_directories(dir, ec);

    const std::string filepath = snapshot_instance_path(name, snapshot);

    // the same compose pipeline as a request-time switch: per-slot lock, switch
    // semaphore, deadline-bounded KV copy on the scheduler, then an awaited
    // file write on the pool I/O worker
    {
        std::lock_guard<std::mutex> slot_lock(*inst->mutex_snapshot[id_slot]);
        const int64_t               deadline_ms = snapshot_deadline_ms(*inst);
        switch_guard                sw_guard(*this, deadline_ms);
        if (!sw_guard.acquired) {
            return make_error("too many concurrent snapshot switches, retry", ERROR_TYPE_UNAVAILABLE);
        }

        auto copied = kv_copy_to_data(*inst, id_slot, deadline_ms, "snapshot save timed out");
        if (!copied.ok) {
            return std::move(copied.error);
        }
        server_snapshot_data data = std::move(copied.data);
        // await the write so a failed save never binds the slot to a file that does not
        // exist on disk (the slot KV is unchanged either way)
        auto write = snapshot_io_write(filepath, std::move(data), deadline_ms);
        if (write.busy) {
            return make_error("snapshot io worker is busy, retry", ERROR_TYPE_UNAVAILABLE);
        }
        if (write.timed_out) {
            return make_error("snapshot save timed out", ERROR_TYPE_UNAVAILABLE);
        }
        if (!write.ok) {
            return make_error("failed to write snapshot '" + snapshot + "' to disk", ERROR_TYPE_SERVER);
        }

        // the instance-scoped file is now authoritative: drop a legacy flat
        // file of the same name (migration dedup) so listings never show the
        // snapshot twice and future reads cannot ambiguate.
        {
            std::error_code lec;
            std::filesystem::remove(snapshot_legacy_path(snapshot), lec);
        }

        // the slot's KV now matches the snapshot content, so bind it to the snapshot
        std::lock_guard<std::mutex> lock(mutex_dispatch);
        inst->slot_snapshots[id_slot] = snapshot;

        // copy metadata: same shape as server_task_result_slot_copy::to_json
        // (that object was consumed building data above; a move preserves sizes)
        return make_ok({ { "id_slot", id_slot }, { "n_tokens", data.tokens.size() }, { "n_bytes", data.kv.size() } }, 201);
    }
}

server_http_res_ptr server_instances::handle_get_instance_snapshots(const server_http_req & req) {
    const std::string name = req.get_param("name");
    auto              inst = get_instance(name);
    if (!inst) {
        return make_error("instance not found: '" + name + "'", ERROR_TYPE_NOT_FOUND);
    }
    if (params.slot_save_path.empty()) {
        return make_error("snapshot management requires --slot-save-path", ERROR_TYPE_NOT_SUPPORTED);
    }

    return make_ok({
        { "snapshots", instance_snapshots_json(name) }
    });
}

server_http_res_ptr server_instances::handle_delete_instance_snapshot(const server_http_req & req) {
    // serialized with resize so a binding cleanup can never race a context rebuild that
    // re-allocates the slot-snapshot bookkeeping
    std::lock_guard<std::mutex> mgmt_lock(mutex_mgmt);

    const std::string name     = req.get_param("name");
    const std::string snapshot = req.get_param("snapshot");
    if (!fs_validate_filename(snapshot)) {
        return make_error("invalid snapshot name", ERROR_TYPE_INVALID_REQUEST);
    }
    if (params.slot_save_path.empty()) {
        return make_error("snapshot management requires --slot-save-path", ERROR_TYPE_NOT_SUPPORTED);
    }

    auto inst = get_instance(name);
    if (!inst) {
        return make_error("instance not found: '" + name + "'", ERROR_TYPE_NOT_FOUND);
    }

    // per-instance delete: the hashed-key scoped file, the previous-key scoped
    // file (pre-hashing saves), then the legacy flat file (migration). each
    // removed file unbinds slots: the named instance's slots for its own files,
    // every instance's slots for a legacy file (which any instance may have
    // restored before scoping existed).
    const std::string inst_path = snapshot_instance_path(name, snapshot);
    const std::string prev_path = snapshot_instance_path_prev(name, snapshot);
    const std::string leg_path  = snapshot_legacy_path(snapshot);

    std::error_code ec;
    const bool removed_inst = std::filesystem::remove(inst_path, ec);
    ec.clear();
    const bool removed_prev = std::filesystem::remove(prev_path, ec);
    ec.clear();
    const bool removed_leg = std::filesystem::remove(leg_path, ec);

    if (!removed_inst && !removed_prev && !removed_leg) {
        return make_error("snapshot not found: '" + snapshot + "'", ERROR_TYPE_NOT_FOUND);
    }

    // unbind any slot that was bound to the deleted snapshot
    {
        std::lock_guard<std::mutex> lock(mutex_dispatch);
        if (removed_leg) {
            for (const auto & it : instances) {
                for (auto & s : it->slot_snapshots) {
                    if (s == snapshot) {
                        s.clear();
                    }
                }
            }
        } else {
            for (auto & s : inst->slot_snapshots) {
                if (s == snapshot) {
                    s.clear();
                }
            }
        }
    }

    return make_ok({
        { "success", true }
    });
}

//
// lifecycle
//

void server_instances::start_loops() {
    // idempotent and safe to call after demand-driven builds: only instances that
    // own a context and never started a loop get one. serialized on mutex_dispatch
    // so a racing demand build cannot interleave a second start (assigning over a
    // joinable thread would std::terminate).
    std::lock_guard<std::mutex> lock(mutex_dispatch);
    if (terminated) {
        return;
    }
    for (const auto & inst : instances) {
        // unbuilt windows have no scheduler yet; their loop starts on first demand
        if (!inst->ctx_server) {
            continue;
        }
        start_instance_loop_locked(*inst);
    }
    start_io_worker();
}

server_instances::~server_instances() {
    // release the decision session store's adapter refs before the shared model is freed
    decision_sessions_clear();
    // last-resort shutdown on any exit path (e.g. an exception after start_loops): joins
    // every scheduler thread so no joinable thread survives pool destruction (a joinable
    // std::thread destructor would std::terminate the process). a no-op after a normal
    // terminate(), which the flag below makes one-shot.
    terminate();
}

void server_instances::terminate() {
    // one-shot: the signal handler, the main path, and the destructor may each call
    // terminate(); only the first call tears anything down. runs under mutex_dispatch so
    // no concurrent create_instance can slip a new instance past the teardown.
    std::vector<std::shared_ptr<server_instance>> live;
    {
        std::lock_guard<std::mutex> lock(mutex_dispatch);
        if (terminated) {
            return;
        }
        terminated = true;
        // reject any dispatch that arrives during teardown (group waiters re-pick and
        // find nothing)
        for (const auto & inst : instances) {
            inst->running = false;
        }
        cond_dispatch.notify_all();
        // snapshot so teardown never iterates a vector that a management op may mutate
        live = instances;
    }

    for (const auto & inst : live) {
        // unbuilt windows never started a scheduler and own no context
        if (inst->ctx_server) {
            inst->ctx_server->terminate();
        }
    }
    for (const auto & inst : live) {
        bool do_join;
        {
            std::lock_guard<std::mutex> lock(mutex_dispatch);
            do_join               = !inst->scheduler_joined;
            inst->scheduler_joined = true;
        }
        if (do_join && inst->loop_thread.joinable()) {
            inst->loop_thread.join();
        }
    }
    stop_io_worker();

    // drain the adapter registry under mutex_mgmt with NO dispatch lock held (the
    // joins above already ran lock-free): every scheduler is terminated, so no
    // params_base.lora_adapters raw ptr can still be dereferenced, and the pool's
    // model_init is still alive so llama_adapter_lora_free can erase from model->loras.
    {
        std::lock_guard<std::mutex> mgmt_lock(mutex_mgmt);
        adapter_registry.clear();
    }
}

// pool snapshot I/O worker: a single FIFO thread owns every on-disk snapshot read/write,
// so a snapshot switch never does file I/O on a scheduler thread and a client disconnect
// cannot interrupt an in-flight write. jobs are bounded by the switch semaphore.
void server_instances::io_loop() {
    while (true) {
        std::packaged_task<void()> task;
        {
            std::unique_lock<std::mutex> lock(mutex_io);
            cond_io.wait(lock, [&]() { return io_stop || !io_jobs.empty(); });
            if (io_jobs.empty()) {
                return;  // io_stop set and the queue is drained
            }
            task = std::move(io_jobs.front());
            io_jobs.pop_front();
        }
        task();
    }
}

std::optional<std::future<void>> server_instances::snapshot_io_post(std::function<void()> && fn) {
    std::packaged_task<void()> task(std::move(fn));
    auto                       future = task.get_future();
    {
        std::lock_guard<std::mutex> lock(mutex_io);
        // no worker, no queue: before the single start or after the single stop a
        // posted job would never run, so refuse fast with a retriable error
        if (!io_running) {
            return std::nullopt;
        }
        // hard bound on queued jobs: each queued write holds a full KV host buffer, so a
        // full queue rejects with a retriable error instead of accumulating host memory
        if (io_jobs.size() >= max_io_jobs) {
            return std::nullopt;
        }
        io_jobs.push_back(std::move(task));
    }
    cond_io.notify_one();
    return future;
}

void server_instances::start_io_worker() {
    std::lock_guard<std::mutex> lock(mutex_io);
    if (io_running) {
        return;
    }
    io_stop    = false;
    io_running = true;
    if (!io_thread.joinable()) {
        io_thread = std::thread([this]() { io_loop(); });
    }
}

void server_instances::stop_io_worker() {
    {
        std::lock_guard<std::mutex> lock(mutex_io);
        if (!io_running) {
            return;
        }
        io_running = false;
        io_stop    = true;
        cond_io.notify_all();
    }
    // the loop drains any pending jobs before it returns, so an HTTP thread waiting
    // on a job future is never left hanging at shutdown
    if (io_thread.joinable()) {
        io_thread.join();
    }
}

std::shared_ptr<server_instance> server_instances::default_instance() {
    // one implementation (see the const overload); the dispatch mutex is mutable
    return static_cast<const server_instances *>(this)->default_instance();
}

std::shared_ptr<server_instance> server_instances::default_instance() const {
    std::lock_guard<std::mutex> lock(mutex_dispatch);
    for (const auto & inst : instances) {
        if (inst->cfg.is_default) {
            return inst;
        }
    }
    return instances.empty() ? nullptr : instances.front();
}

// the shared prologue of every default-instance endpoint: resolve the default,
// build it on demand, guard against a racing destroy/resize, then run the
// instance's own route handler. custom endpoints (health, props, slots,
// models) keep their own bodies.
server_http_res_ptr server_instances::default_instance_forward(const server_http_req & req,
                                                               server_http_context::handler_t server_routes::* method) {
    auto inst = default_instance();
    if (!inst) {
        return make_error("no instances loaded", ERROR_TYPE_SERVER);
    }
    if (auto err = ensure_built_instance(inst)) {
        return err;
    }
    active_route_guard guard(*this, *inst);
    if (!guard.acquired) {
        return make_error("instance '" + inst->cfg.name + "' is being reconfigured", ERROR_TYPE_UNAVAILABLE);
    }
    return (inst->routes.get()->*method)(req);
}

server_http_res_ptr server_instances::make_error(const std::string & message, error_type type) const {
    auto res          = std::make_unique<server_http_res>();
    res->status       = error_status(type);
    res->content_type = "application/json; charset=utf-8";
    res->data         = safe_json_to_str({
        { "error", format_error_response(message, type) }
    });
    return res;
}

server_http_res_ptr server_instances::make_error_from_result(server_task_result & result) {
    auto * err = dynamic_cast<server_task_result_error *>(&result);
    // the error class travels with the result: the body is the class's own render and
    // the status comes from the same table, never from a field of that body. a result
    // that is not an error means the caller reached an error branch unexpectedly.
    auto res          = std::make_unique<server_http_res>();
    res->status       = err != nullptr ? error_status(err->err_type) : error_status(ERROR_TYPE_SERVER);
    res->content_type = "application/json; charset=utf-8";
    res->data         = safe_json_to_str({
        { "error", err != nullptr ? err->to_json() :
                                    format_error_response("unexpected instance task result", ERROR_TYPE_SERVER) }
    });
    return res;
}

server_http_res_ptr server_instances::make_error(int                 code,
                                                 const std::string & type,
                                                 const std::string & message) const {
    auto res          = std::make_unique<server_http_res>();
    res->status       = code;
    res->content_type = "application/json; charset=utf-8";
    res->data         = safe_json_to_str({
        { "error", { { "code", code }, { "message", message }, { "type", type } } }
    });
    return res;
}

server_http_res_ptr server_instances::make_ok(const json & data, int status) const {
    auto res          = std::make_unique<server_http_res>();
    res->status       = status;
    res->content_type = "application/json; charset=utf-8";
    res->data         = safe_json_to_str(data);
    return res;
}
