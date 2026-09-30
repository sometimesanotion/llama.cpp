// A/B snapshot-mechanism bench (the M0 decision-record experiment).
//
// One model, two contexts with identical config: a "chat" context that owns the
// turn, and a "sidecar" context that answers. A synthetic turn of N tokens is
// decoded on the chat context, then a 16-token question head is decoded with
// logits. The head logits on the chat context with the prefix resident are the
// "shared" continuation. Two mechanisms reproduce that continuation on the
// sidecar:
//
//   A (host bytes): llama_state_seq_get_data_ext on the chat context, then
//   llama_state_seq_set_data_ext on the sidecar, then the head decode.
//
//   B (token replay): the sidecar re-prefills the turn token list (cold), or
//   forks the already-resident prefix (warm), then the head decode.
//
// Times are ms; maxdiff is the max |logit| difference of the mechanism's head
// continuation against the shared one. See docs/decision/OBJECTIVE_MULTI_CONTEXT.md
// section 12.2 for the recorded table this reproduces.

#include "common.h"
#include "ggml-backend.h"
#include "json.h"
#include "llama.h"

#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

static double now_ms() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration<double, std::milli>(now).count();
}

static double ms_elapsed(double start_ms) {
    return now_ms() - start_ms;
}

struct bench_opts {
    std::string model;
    std::vector<int> ctx_sizes = { 2048 };  // turn token counts
    int head_tokens = 16;
    int n_gpu_layers = 99;
    int reps = 1;       // runs per turn size (cold-prefill percentile measurement, M8)
    bool json_out = false;
    bool cold_only = false; // measure only the cold-prefill cost (B cold); M8 timeout calibration
};

struct loaded {
    llama_model * model = nullptr;
    llama_context * ctx_chat = nullptr;
    llama_context * ctx_side = nullptr;

    bool load(const bench_opts & bo) {
        llama_backend_init();
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = bo.n_gpu_layers;
        model = llama_model_load_from_file(bo.model.c_str(), mp);
        if (!model) {
            fprintf(stderr, "bench-snapshot: failed to load model %s\n", bo.model.c_str());
            return false;
        }
        const int n_ctx = *std::max_element(bo.ctx_sizes.begin(), bo.ctx_sizes.end()) + bo.head_tokens + 64;
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx             = n_ctx;
        cp.n_batch           = 512;
        cp.n_ubatch          = 512;
        cp.n_seq_max         = 8;
        cp.n_outputs_max     = 8;
        cp.n_outputs_max_per_seq = 1;
        cp.kv_unified        = true;
        cp.swa_full          = false;
        // ROCm flash attention is not bit-reproducible; the mechanism comparison must be exact.
        cp.flash_attn_type   = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        ctx_chat = llama_init_from_model(model, cp);
        if (!ctx_chat) {
            fprintf(stderr, "bench-snapshot: failed to create chat context\n");
            return false;
        }
        ctx_side = llama_init_from_model(model, cp);
        if (!ctx_side) {
            fprintf(stderr, "bench-snapshot: failed to create sidecar context\n");
            return false;
        }
        return true;
    }

    ~loaded() {
        if (ctx_side) llama_free(ctx_side);
        if (ctx_chat) llama_free(ctx_chat);
        if (model) llama_model_free(model);
        llama_backend_free();
    }
};

static const char * turn_base() {
    return "The decision engine reads a short evidence document and answers a fixed set of typed "
           "questions by scoring single-token labels at the answer boundary. ";
}

static std::vector<llama_token> turn_tokens(const llama_vocab * vocab, int n) {
    std::string text;
    while (true) {
        text += turn_base();
        std::vector<llama_token> toks = common_tokenize(vocab, text, false, false);
        if ((int) toks.size() >= n) {
            toks.resize(n);
            return toks;
        }
    }
}

static std::vector<llama_token> head_tokens(const llama_vocab * vocab, int n) {
    std::vector<llama_token> toks;
    while ((int) toks.size() < n) {
        std::vector<llama_token> part = common_tokenize(
            vocab, " Based on the evidence above, answer each question with one label.", false, false);
        for (llama_token t : part) {
            toks.push_back(t);
            if ((int) toks.size() == n) {
                break;
            }
        }
    }
    return toks;
}

// decode `toks` on `seq` starting at `pos0`; when logits_out is non-null the
// last token requests logits and the continuation is copied into it. chunks at
// n_batch so a long turn never exceeds the context's logical batch size.
static bool decode_on(llama_context * ctx, llama_seq_id seq, llama_pos pos0,
                      const std::vector<llama_token> & toks, std::vector<float> * logits_out) {
    const int n_batch = (int) llama_n_batch(ctx);
    for (size_t beg = 0; beg < toks.size(); beg += n_batch) {
        const size_t n = std::min((size_t) n_batch, toks.size() - beg);
        const bool last_chunk = beg + n == toks.size();
        llama_batch batch = llama_batch_init((int) n, 0, 1);
        for (size_t i = 0; i < n; ++i) {
            const bool wants_logits = logits_out != nullptr && last_chunk && i + 1 == n;
            common_batch_add(batch, toks[beg + i], pos0 + (llama_pos) (beg + i), { seq }, wants_logits);
        }
        const int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            fprintf(stderr, "bench-snapshot: decode failed rc=%d seq=%d pos0=%d n=%zu wants_logits=%d\n",
                    rc, (int) seq, (int) pos0, toks.size(), logits_out != nullptr ? 1 : 0);
            return false;
        }
    }
    if (logits_out != nullptr) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        if (logits == nullptr) {
            return false;
        }
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
        logits_out->assign(logits, logits + n_vocab);
    }
    return true;
}

static double max_abs_logit_delta(const std::vector<float> & a, const std::vector<float> & b) {
    double delta = 0.0;
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        delta = std::max(delta, (double) std::fabs(a[i] - b[i]));
    }
    return delta;
}

// percentile of a sorted sample (0..100). the sample must be non-empty.
static double pct(const std::vector<double> & sorted, double p) {
    const size_t n = sorted.size();
    if (n == 1) {
        return sorted[0];
    }
    const double idx = (p / 100.0) * (double) (n - 1);
    const size_t lo = (size_t) std::floor(idx);
    const size_t hi = (size_t) std::ceil(idx);
    const double frac = idx - (double) lo;
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

static const char * model_identity(const std::string & path) {
    static std::string cached;
    size_t slash = path.find_last_of("/\\");
    if (slash == std::string::npos) {
        cached = path;
    } else {
        const std::string file = path.substr(slash + 1);
        const size_t prev = path.find_last_of("/\\", slash - 1);
        cached = prev == std::string::npos ? file : path.substr(prev + 1, slash - prev - 1) + "/" + file;
    }
    return cached.c_str();
}

// One A/B row for a single turn size. Fills out with the measured times and
// maxdiffs; returns false on a hard failure.
static bool run_size(loaded & ld, int n, int head_n, common_json & out) {
    const llama_vocab * vocab = llama_model_get_vocab(ld.model);
    const std::vector<llama_token> turn  = turn_tokens(vocab, n);
    const std::vector<llama_token> head  = head_tokens(vocab, head_n);
    if ((int) turn.size() != n || (int) head.size() != head_n) {
        return false;
    }

    // clear both contexts so a rep is measured on a fresh prefix (--reps repeats run_size)
    llama_memory_clear(llama_get_memory(ld.ctx_chat), true);
    llama_memory_clear(llama_get_memory(ld.ctx_side), true);

    // --- shared: chat context with the prefix resident ---
    if (!decode_on(ld.ctx_chat, 0, 0, turn, nullptr)) {
        return false;
    }

    // --- A capture: host bytes of the turn prefix, before the head decode ---
    const size_t a_size = llama_state_seq_get_size_ext(ld.ctx_chat, 0, LLAMA_STATE_SEQ_FLAGS_NONE);
    std::vector<uint8_t> a_bytes(a_size);
    const double t_a_cap = now_ms();
    const size_t a_got = llama_state_seq_get_data_ext(ld.ctx_chat, a_bytes.data(), a_bytes.size(), 0,
                                                      LLAMA_STATE_SEQ_FLAGS_NONE);
    const double a_capture_ms = ms_elapsed(t_a_cap);
    if (a_got != a_bytes.size()) {
        fprintf(stderr, "bench-snapshot: mechanism A capture returned %zu of %zu bytes\n", a_got, a_bytes.size());
        return false;
    }

    // shared continuation: the head decode with the prefix resident
    std::vector<float> shared;
    const double t0 = now_ms();
    if (!decode_on(ld.ctx_chat, 0, (llama_pos) n, head, &shared)) {
        return false;
    }
    const double shared_ms = ms_elapsed(t0);

    // --- A load: host bytes into the sidecar ---
    llama_memory_clear(llama_get_memory(ld.ctx_side), true);
    const double t_a_load = now_ms();
    const size_t a_set = llama_state_seq_set_data_ext(ld.ctx_side, a_bytes.data(), a_bytes.size(), 0,
                                                      LLAMA_STATE_SEQ_FLAGS_NONE);
    const double a_load_ms = ms_elapsed(t_a_load);
    if (a_set != a_bytes.size()) {
        fprintf(stderr, "bench-snapshot: mechanism A load returned %zu of %zu bytes\n", a_set, a_bytes.size());
        return false;
    }
    std::vector<float> a_cont;
    const double t_a_head = now_ms();
    if (!decode_on(ld.ctx_side, 0, (llama_pos) n, head, &a_cont)) {
        return false;
    }
    const double a_head_ms = ms_elapsed(t_a_head);
    const double maxdiff_a = max_abs_logit_delta(shared, a_cont);

    // --- B cold: token replay prefill (measured first, on a fresh prefix) ---
    llama_memory_clear(llama_get_memory(ld.ctx_side), true);
    const double t_b_pre = now_ms();
    if (!decode_on(ld.ctx_side, 0, 0, turn, nullptr)) {
        return false;
    }
    const double b_prefill_ms = ms_elapsed(t_b_pre);
    std::vector<float> b_cold;
    const double t_b_head = now_ms();
    if (!decode_on(ld.ctx_side, 0, (llama_pos) n, head, &b_cold)) {
        return false;
    }
    const double b_head_ms = ms_elapsed(t_b_head);
    const double maxdiff_b_cold = max_abs_logit_delta(shared, b_cold);

    // --- B warm: fork a fresh turn-only resident prefix [0, n); the fork and
    //     head decode never share a seq with a previous head decode, so the
    //     recurrent state the fork copies is a clean turn state ---
    llama_memory_clear(llama_get_memory(ld.ctx_side), true);
    if (!decode_on(ld.ctx_side, 0, 0, turn, nullptr)) {
        return false;
    }
    const double t_b_fork = now_ms();
    llama_memory_seq_cp(llama_get_memory(ld.ctx_side), 0, 1, 0, n);
    const double b_fork_ms = ms_elapsed(t_b_fork);
    std::vector<float> b_warm;
    const double t_b_warm_head = now_ms();
    if (!decode_on(ld.ctx_side, 1, (llama_pos) n, head, &b_warm)) {
        return false;
    }
    const double b_warm_head_ms = ms_elapsed(t_b_warm_head);
    const double maxdiff_b_warm = max_abs_logit_delta(shared, b_warm);

    out["N"] = n;
    out["a_capture_ms"] = a_capture_ms;
    out["a_load_ms"] = a_load_ms;
    out["a_head_ms"] = a_head_ms;
    out["a_total_ms"] = a_capture_ms + a_load_ms + a_head_ms;
    out["b_cold_ms"] = b_prefill_ms + b_head_ms;
    out["b_warm_ms"] = b_fork_ms + b_warm_head_ms;
    out["shared_ms"] = shared_ms;
    out["maxdiff_a"] = maxdiff_a;
    out["maxdiff_b_cold"] = maxdiff_b_cold;
    out["maxdiff_b_warm"] = maxdiff_b_warm;
    return true;
}

// Cold-only measurement (M8): how long a decision pays on the sidecar to replay
// a turn of n tokens and score the head. This is the number the
// --decision-timeout-ms default is calibrated from. It avoids the A capture and
// warm fork, which are large/sliding-window-fragile and not needed for the
// deadline rule.
static bool run_cold(loaded & ld, int n, int head_n, common_json & out) {
    const llama_vocab * vocab = llama_model_get_vocab(ld.model);
    const std::vector<llama_token> turn = turn_tokens(vocab, n);
    const std::vector<llama_token> head = head_tokens(vocab, head_n);
    if ((int) turn.size() != n || (int) head.size() != head_n) {
        return false;
    }
    llama_memory_clear(llama_get_memory(ld.ctx_side), true);
    const double t0 = now_ms();
    if (!decode_on(ld.ctx_side, 0, 0, turn, nullptr)) {
        return false;
    }
    if (!decode_on(ld.ctx_side, 0, (llama_pos) n, head, nullptr)) {
        return false;
    }
    out["cold_ms"] = ms_elapsed(t0);
    return true;
}

static void usage() {
    fprintf(stderr,
            "usage: bench-snapshot <model.gguf> [--ctx 2048[,8192[,32768]]] [--head 16] [--ngl 99] [--reps N] [--json]\n"
            "  --reps N  run each turn size N times and report cold-prefill p50/p95/p99 (M8 calibration)\n"
            "  --cold-only  measure only the cold-prefill cost (B cold); avoids the warm fork that is\n"
            "               sliding-window-fragile at large sizes and is not needed for the deadline rule\n");
}

int main(int argc, char ** argv) {
    bench_opts bo;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--ctx") {
            if (i + 1 >= argc) { usage(); return 2; }
            bo.ctx_sizes.clear();
            std::string list = argv[++i];
            size_t at = 0;
            while (at <= list.size()) {
                size_t comma = list.find(',', at);
                bo.ctx_sizes.push_back(std::stoi(list.substr(at, comma == std::string::npos ? std::string::npos : comma - at)));
                if (comma == std::string::npos) break;
                at = comma + 1;
            }
        } else if (arg == "--head") {
            if (i + 1 >= argc) { usage(); return 2; }
            bo.head_tokens = std::atoi(argv[++i]);
        } else if (arg == "--ngl") {
            if (i + 1 >= argc) { usage(); return 2; }
            bo.n_gpu_layers = std::atoi(argv[++i]);
        } else if (arg == "--reps") {
            if (i + 1 >= argc) { usage(); return 2; }
            bo.reps = std::atoi(argv[++i]);
            if (bo.reps < 1) {
                fprintf(stderr, "bench-snapshot: --reps must be >= 1\n");
                return 2;
            }
        } else if (arg == "--cold-only") {
            bo.cold_only = true;
        } else if (arg == "--json") {
            bo.json_out = true;
        } else if (arg == "-h" || arg == "--help") {
            usage();
            return 0;
        } else if (!arg.empty() && arg[0] == '-') {
            usage();
            return 2;
        } else {
            bo.model = arg;
        }
    }
    if (bo.model.empty()) {
        usage();
        return 2;
    }

    loaded ld;
    if (!ld.load(bo)) {
        return 1;
    }

    common_json rows = common_json::array();
    for (int n : bo.ctx_sizes) {
        // b_cold_ms / cold_ms is the cold-prefill cost a decision pays on the sidecar
        // for a turn of n tokens; M8 calibrates the --decision-timeout-ms default from
        // its p50/p95/p99 across reps.
        std::vector<double> cold;
        for (int rep = 0; rep < bo.reps; ++rep) {
            common_json row = common_json::object();
            row["model"] = model_identity(bo.model);
            row["head"] = bo.head_tokens;
            row["rep"] = rep;
            const bool ok = bo.cold_only
                ? run_cold(ld, n, bo.head_tokens, row)
                : run_size(ld, n, bo.head_tokens, row);
            if (!ok) {
                fprintf(stderr, "bench-snapshot: run for N=%d rep=%d failed\n", n, rep);
                return 1;
            }
            cold.push_back(bo.cold_only ? row["cold_ms"].get<double>() : row["b_cold_ms"].get<double>());
            rows.push_back(row);
        }
        if (bo.reps > 1) {
            std::sort(cold.begin(), cold.end());
            const double p50 = pct(cold, 50.0);
            const double p95 = pct(cold, 95.0);
            const double p99 = pct(cold, 99.0);
            if (bo.json_out) {
                common_json p = common_json::object();
                p["N"] = n;
                p["model"] = model_identity(bo.model);
                p["reps"] = bo.reps;
                p["cold_p50_ms"] = p50;
                p["cold_p95_ms"] = p95;
                p["cold_p99_ms"] = p99;
                p["cold_max_ms"] = cold.back();
                rows.push_back(p);
            } else {
                printf("| %s | %d | %d | %.1f | %.1f | %.1f | %.1f |\n",
                       model_identity(bo.model), n, bo.reps, p50, p95, p99, cold.back());
            }
        }
    }

    if (bo.reps == 1) {
        if (bo.json_out) {
            common_json out = common_json::object();
            out["rows"] = rows;
            std::cout << out.dump(2) << "\n";
        } else if (!bo.cold_only) {
            printf("| model | N | A capture | A load | A total | B cold | B warm | shared | maxdiff A | maxdiff B cold | maxdiff B warm |\n");
            printf("|---|---|---|---|---|---|---|---|---|---|---|\n");
            for (size_t i = 0; i < rows.size(); ++i) {
                const common_json & r = rows[i];
                printf("| %s | %d | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.4f | %.4f | %.4f |\n",
                       r["model"].get<std::string>().c_str(), r["N"].get<int>(),
                       r["a_capture_ms"].get<double>(), r["a_load_ms"].get<double>(),
                       r["a_total_ms"].get<double>(), r["b_cold_ms"].get<double>(),
                       r["b_warm_ms"].get<double>(), r["shared_ms"].get<double>(),
                       r["maxdiff_a"].get<double>(), r["maxdiff_b_cold"].get<double>(),
                       r["maxdiff_b_warm"].get<double>());
            }
        } else {
            printf("| model | N | cold |\n|---|---|---|\n");
            for (size_t i = 0; i < rows.size(); ++i) {
                const common_json & r = rows[i];
                printf("| %s | %d | %.1f |\n",
                       r["model"].get<std::string>().c_str(), r["N"].get<int>(),
                       r["cold_ms"].get<double>());
            }
        }
    } else if (!bo.json_out) {
        printf("| model | N | reps | cold p50 | cold p95 | cold p99 | cold max |\n");
        printf("|---|---|---|---|---|---|---|\n");
        // the percentile rows were already printed in the loop above
    }
    return 0;
}