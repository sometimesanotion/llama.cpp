#include "arg.h"
#include "common.h"
#include "download.h"
#include "llama.h"
#include "speculative.h"

#include <cmath>
#include <limits>
#include <string>
#include <vector>
#include <sstream>
#include <unordered_set>

#undef NDEBUG
#include <cassert>

static void test(void) {
    common_params params;

    auto assert_output_limits = [](int32_t n_batch, int32_t n_parallel, int32_t n_draft,
                                   int32_t total, int32_t per_seq) {
        const auto limits = common_speculative_get_output_limits(n_batch, n_parallel, n_draft);
        assert(limits.total == total);
        assert(limits.per_seq == per_seq);
    };

    assert_output_limits(16, 2,  3, 8, 4);
    assert_output_limits(16, 2, -1, 2, 1);
    assert_output_limits( 6, 2,  3, 6, 4);
    assert_output_limits( 2, 1,  3, 2, 2);
    assert_output_limits(
            std::numeric_limits<int32_t>::max(),
            std::numeric_limits<int32_t>::max(),
            std::numeric_limits<int32_t>::max(),
            std::numeric_limits<int32_t>::max(),
            std::numeric_limits<int32_t>::max());

    {
        common_params_speculative spec;
        spec.synth_len = 3.4;

        auto assert_invalid = [](const common_params_speculative & value, int32_t n_max) {
            try {
                common_speculative_synth_rates_resolve(&value, n_max);
                assert(false);
            } catch (const std::invalid_argument &) {
            }
        };

        const auto rates = common_speculative_synth_rates_resolve(&spec, 4);
        assert(rates.size() == 4);
        assert(std::abs(rates[0] - 0.80581) < 1e-5);
        assert(std::abs(rates[1] - 0.64933) < 1e-5);
        assert(std::abs(rates[2] - 0.52323) < 1e-5);
        assert(std::abs(rates[3] - 0.42163) < 1e-5);
        assert(std::abs(1.0 + rates[0] + rates[1] + rates[2] + rates[3] - 3.4) < 1e-8);

        spec.synth_len = 1.0;
        assert(common_speculative_synth_rates_resolve(&spec, 4) == std::vector<double>({0.0, 0.0, 0.0, 0.0}));

        spec.synth_len = 5.0;
        assert(common_speculative_synth_rates_resolve(&spec, 4) == std::vector<double>({1.0, 1.0, 1.0, 1.0}));

        spec.synth_len = 5.1;
        assert_invalid(spec, 4);

        spec.synth_len = std::numeric_limits<double>::quiet_NaN();
        assert_invalid(spec, 4);

        spec.synth_len = 0.0;
        assert_invalid(spec, 4);

        spec.synth_len = -1.0;
        spec.synth_rates = {0.8, 0.6, 0.4};
        assert_invalid(spec, 4);

        spec.synth_rates = {0.8, 0.6, 0.4, 0.2};
        assert(common_speculative_synth_rates_resolve(&spec, 4) == spec.synth_rates);

        spec.synth_rates = {0.8, 0.9, 0.4, 0.2};
        assert_invalid(spec, 4);

        spec.synth_rates = {0.8, std::numeric_limits<double>::quiet_NaN(), 0.4, 0.2};
        assert_invalid(spec, 4);

        spec.synth_rates = {0.8, 0.6, 0.4, -0.2};
        assert_invalid(spec, 4);

        spec.synth_rates = {0.8, 0.6, 0.4, 0.2};
        spec.synth_len = 3.0;
        assert_invalid(spec, 4);
    }

    {
        common_params base;
        base.n_parallel = 4;
        base.n_outputs_max_per_seq = 8;

        const auto draft = common_base_params_to_speculative(base);
        assert(draft.n_outputs_max == 4);
        assert(draft.n_outputs_max_per_seq == 1);
    }

    printf("test-arg-parser: make sure there is no duplicated arguments in any examples\n\n");
    for (int ex = 0; ex < LLAMA_EXAMPLE_COUNT; ex++) {
        try {
            auto ctx_arg = common_params_parser_init(params, (enum llama_example)ex);
            common_params_add_preset_options(ctx_arg.options);
            std::unordered_set<std::string> seen_args;
            std::unordered_set<std::string> seen_env_vars;
            for (const auto & opt : ctx_arg.options) {
                // check for args duplications
                for (const auto & arg : opt.get_args()) {
                    if (seen_args.find(arg) == seen_args.end()) {
                        seen_args.insert(arg);
                    } else {
                        fprintf(stderr, "test-arg-parser: found different handlers for the same argument: %s", arg.c_str());
                        exit(1);
                    }
                }
                // check for env var duplications
                for (const auto & env : opt.get_env()) {
                    if (seen_env_vars.find(env) == seen_env_vars.end()) {
                        seen_env_vars.insert(env);
                    } else {
                        fprintf(stderr, "test-arg-parser: found different handlers for the same env var: %s", env.c_str());
                        exit(1);
                    }
                }

                // exclude spec args from this check
                // ref: https://github.com/ggml-org/llama.cpp/pull/22397
                const bool skip = opt.is_spec;

                // ensure shorter argument precedes longer argument
                if (!skip && opt.args.size() > 1) {
                    const std::string first(opt.args.front());
                    const std::string last(opt.args.back());

                    if (first.length() > last.length()) {
                        fprintf(stderr, "test-arg-parser: shorter argument should come before longer one: %s, %s\n",
                                first.c_str(), last.c_str());
                        assert(false);
                    }
                }

                // same check for negated arguments
                if (opt.args_neg.size() > 1) {
                    const std::string first(opt.args_neg.front());
                    const std::string last(opt.args_neg.back());

                    if (first.length() > last.length()) {
                        fprintf(stderr, "test-arg-parser: shorter negated argument should come before longer one: %s, %s\n",
                                first.c_str(), last.c_str());
                        assert(false);
                    }
                }
            }
        } catch (std::exception & e) {
            printf("%s\n", e.what());
            assert(false);
        }
    }

    auto list_str_to_char = [](std::vector<std::string> & argv) -> std::vector<char *> {
        std::vector<char *> res;
        for (auto & arg : argv) {
            res.push_back(const_cast<char *>(arg.data()));
        }
        return res;
    };

    std::vector<std::string> argv;

    printf("test-arg-parser: test invalid usage\n\n");

    // missing value
    argv = {"binary_name", "-m"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    // wrong value (int)
    argv = {"binary_name", "-ngl", "hello"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    // wrong value (enum)
    argv = {"binary_name", "-sm", "hello"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    {
        common_params penalty_params;
        assert(penalty_params.sampling.penalty_last_n == 64);
        assert(penalty_params.sampling.dry_penalty_last_n == 64);

        argv = {"binary_name", "--repeat-last-n", "-1"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--dry-penalty-last-n", "-1"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "0"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "-1"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "nan"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "inf"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "-inf"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        const char * penalty_options[] = {"--frequency-penalty", "--presence-penalty"};
        const char * nonfinite_values[] = {"nan", "inf", "-inf"};
        for (const char * option : penalty_options) {
            for (const char * value : nonfinite_values) {
                argv = {"binary_name", option, value};
                assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));
            }
        }
    }

    // non-existence arg in specific example (--draft cannot be used outside llama-speculative)
    argv = {"binary_name", "--draft", "123"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_EMBEDDING));

    argv = {"binary_name", "-lm", "hello"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    // repeated --instance flags accumulate into one list, so duplicates and
    // name/group collisions across flags fail just like comma-separated ones
    {
        common_params inst_params;
        argv = {"binary_name", "--instance", "a", "--instance", "a"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), inst_params, LLAMA_EXAMPLE_SERVER));

        argv = {"binary_name", "--instance", "x:group=y", "--instance", "y"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), inst_params, LLAMA_EXAMPLE_SERVER));

        argv = {"binary_name", "--instance", "a", "--instance", "b"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), inst_params, LLAMA_EXAMPLE_SERVER));
        assert(inst_params.instances.size() == 2);
    }

    printf("test-arg-parser: test valid usage\n\n");

    argv = {"binary_name", "-m", "model_file.gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "model_file.gguf");

    argv = {"binary_name", "-t", "1234"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.cpuparams.n_threads == 1234);

    argv = {"binary_name", "--verbose"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.verbosity > 1);

    argv = {"binary_name", "-m", "abc.gguf", "--predict", "6789", "--batch-size", "9090"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "abc.gguf");
    assert(params.n_predict == 6789);
    assert(params.n_batch == 9090);

    // --draft cannot be used outside llama-speculative
    argv = {"binary_name", "--spec-draft-n-max", "123"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SPECULATIVE));
    assert(params.speculative.draft.n_max == 123);

    {
        common_params synth_params;
        argv = {"binary_name", "--spec-synth-len", "3.4"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), synth_params, LLAMA_EXAMPLE_SERVER));
        assert(synth_params.speculative.synth_len == 3.4);
    }

    {
        common_params synth_params;
        argv = {"binary_name", "--spec-synth-rates", "0.8,0.6,0.2"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), synth_params, LLAMA_EXAMPLE_SERVER));
        assert(synth_params.speculative.synth_rates == std::vector<double>({0.8, 0.6, 0.2}));
    }

    {
        common_params synth_params;
        argv = {"binary_name", "--spec-synth-len", "3.4x"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), synth_params, LLAMA_EXAMPLE_SERVER));
    }

    {
        common_params kv_params;
        argv = {"binary_name", "--kv-unified-per-slot", "2048"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), kv_params, LLAMA_EXAMPLE_SERVER));
        assert(kv_params.kv_unified_per_slot == 2048);
    }

    argv = {"binary_name", "-lm", "none"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_NONE);

    argv = {"binary_name", "-lm", "mmap"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MMAP);

    argv = {"binary_name", "-lm", "mlock"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MLOCK);

    argv = {"binary_name", "-lm", "mmap+mlock"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK);

    argv = {"binary_name", "-lm", "dio"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_DIRECT_IO);

    printf("test-arg-parser: test decision flags\n\n");

    {
        common_params dec_params;

        // every decision flag without --decision-seqs is a usage error
        const char * decision_flags[][2] = {
            {"--decision-temperature", "temps.json"},
            {"--decision-contract", "abc123"},
        };
        for (const auto & flag : decision_flags) {
            argv = {"binary_name", flag[0], flag[1]};
            assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), dec_params, LLAMA_EXAMPLE_SERVER));
        }

        // with --decision-seqs they parse and land in params
        argv = {"binary_name", "--decision-seqs", "8"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), dec_params, LLAMA_EXAMPLE_SERVER));
        assert(dec_params.n_seq_decision == 8);

        argv = {"binary_name", "--decision-seqs", "8", "--decision-temperature", "temps.json", "--decision-contract", "abc123"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), dec_params, LLAMA_EXAMPLE_SERVER));
        assert(dec_params.decision_temperature == "temps.json");
        assert(dec_params.decision_contract == "abc123");

        // below the minimum is also a usage error
        argv = {"binary_name", "--decision-seqs", "2"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), dec_params, LLAMA_EXAMPLE_SERVER));
    }

    printf("test-arg-parser: test decision sidecar flags\n\n");

    {
        common_params sc_params;

        // the sidecar executor needs --decision-seqs and a pool to live in
        argv = {"binary_name", "--decision-sidecar"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), sc_params, LLAMA_EXAMPLE_SERVER));
        argv = {"binary_name", "--decision-sidecar-ctx", "4096"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), sc_params, LLAMA_EXAMPLE_SERVER));
        argv = {"binary_name", "--decision-seqs", "8", "--decision-sidecar"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), sc_params, LLAMA_EXAMPLE_SERVER));

        // with the pool they parse and land in params
        argv = {"binary_name", "--decision-seqs", "8", "--decision-sidecar", "--decision-sidecar-ctx", "4096", "--instance", "a:ctx=512"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), sc_params, LLAMA_EXAMPLE_SERVER));
        assert(sc_params.decision_sidecar == true);
        assert(sc_params.decision_sidecar_ctx == 4096);

        // M5.5: the sidecar prebuild, decision timeout and max-queue flags parse and land in params
        argv = {"binary_name", "--decision-seqs", "8", "--decision-sidecar-prebuild",
                "--decision-timeout-ms", "30000", "--decision-max-queue", "8", "--instance", "sc:ctx=512"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), sc_params, LLAMA_EXAMPLE_SERVER));
        assert(sc_params.decision_sidecar_prebuild == true);
        assert(sc_params.decision_timeout_ms == 30000);
        assert(sc_params.decision_max_queue == 8);

        // M7: the resident warm-prefix budget flag parses and lands; a negative budget is rejected
        argv = {"binary_name", "--decision-seqs", "8", "--decision-warm-budget-mb", "256", "--instance", "w:ctx=512"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), sc_params, LLAMA_EXAMPLE_SERVER));
        assert(sc_params.decision_warm_budget_mb == 256);
        argv = {"binary_name", "--decision-seqs", "8", "--decision-warm-budget-mb", "-1", "--instance", "w:ctx=512"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), sc_params, LLAMA_EXAMPLE_SERVER));

        // a negative timeout is rejected; max-queue below 1 is rejected
        argv = {"binary_name", "--decision-seqs", "8", "--decision-timeout-ms", "-1", "--instance", "sc:ctx=512"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), sc_params, LLAMA_EXAMPLE_SERVER));
        argv = {"binary_name", "--decision-seqs", "8", "--decision-max-queue", "0", "--instance", "sc:ctx=512"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), sc_params, LLAMA_EXAMPLE_SERVER));

        // a negative ctx is rejected
        argv = {"binary_name", "--decision-seqs", "8", "--decision-sidecar-ctx", "-1", "--instance", "a:ctx=512"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), sc_params, LLAMA_EXAMPLE_SERVER));

        // the sidecar and a named decision instance are two placements for the same decisions
        argv = {"binary_name", "--decision-seqs", "8", "--decision-sidecar", "--decision-instance", "a", "--instance", "a:ctx=512"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), sc_params, LLAMA_EXAMPLE_SERVER));
    }

    printf("test-arg-parser: test decision sidecar default (M3)\n\n");

    {
        // with a pool present, the sidecar is the default executor when --decision-seqs is set
        common_params m3_params;
        argv = {"binary_name", "--decision-seqs", "8", "--instance", "a:ctx=512"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), m3_params, LLAMA_EXAMPLE_SERVER));
        assert(m3_params.decision_sidecar == true);

        // M4: the temporary hidden --no-decision-sidecar flag is gone; the sidecar is the only
        // pool executor, so an unknown flag is rejected rather than keeping the legacy path
        common_params m3_off;
        argv = {"binary_name", "--decision-seqs", "8", "--no-decision-sidecar", "--instance", "a:ctx=512"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), m3_off, LLAMA_EXAMPLE_SERVER));

        // a single-context server has no pool to host the sidecar: legacy path stays the default
        common_params m3_single;
        argv = {"binary_name", "--decision-seqs", "8"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), m3_single, LLAMA_EXAMPLE_SERVER));
        assert(m3_single.decision_sidecar == false);

        // a named decision instance also selects the legacy executor, not the sidecar default
        common_params m3_named;
        argv = {"binary_name", "--decision-seqs", "8", "--decision-instance", "a", "--instance", "a:ctx=512"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), m3_named, LLAMA_EXAMPLE_SERVER));
        assert(m3_named.decision_sidecar == false);
    }

    printf("test-arg-parser: test decision KV-mode contract\n\n");

    {
        // decision sequences need the unified KV cache: an explicit --no-kv-unified conflicts
        common_params kv_params;
        argv = {"binary_name", "--decision-seqs", "8", "--no-kv-unified"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), kv_params, LLAMA_EXAMPLE_SERVER));

        // an explicit --kv-unified is accepted
        argv = {"binary_name", "--decision-seqs", "8", "--kv-unified"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), kv_params, LLAMA_EXAMPLE_SERVER));
        assert(kv_params.kv_unified == true);

        // without an explicit choice, parse succeeds and the server auto-enables the unified cache
        common_params auto_params;
        argv = {"binary_name", "--decision-seqs", "8"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), auto_params, LLAMA_EXAMPLE_SERVER));
        assert(auto_params.kv_unified == false);
    }

    // multi-value args (CSV)
    argv = {"binary_name", "--lora", "file1.gguf,\"file2,2.gguf\",\"file3\"\"3\"\".gguf\",file4\".gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.lora_adapters.size() == 4);
    assert(params.lora_adapters[0].path == "file1.gguf");
    assert(params.lora_adapters[1].path == "file2,2.gguf");
    assert(params.lora_adapters[2].path == "file3\"3\".gguf");
    assert(params.lora_adapters[3].path == "file4\".gguf");

// skip this part on windows, because setenv is not supported
#ifdef _WIN32
    printf("test-arg-parser: skip on windows build\n");
#else
    printf("test-arg-parser: test environment variables (valid + invalid usages)\n\n");

    setenv("LLAMA_ARG_THREADS", "blah", true);
    argv = {"binary_name"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    setenv("LLAMA_ARG_MODEL", "blah.gguf", true);
    setenv("LLAMA_ARG_THREADS", "1010", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "blah.gguf");
    assert(params.cpuparams.n_threads == 1010);

    setenv("LLAMA_ARG_LOAD_MODE", "blah", true);
    argv = {"binary_name"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    setenv("LLAMA_ARG_LOAD_MODE", "mmap", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MMAP);

    setenv("LLAMA_ARG_LOAD_MODE", "mlock", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MLOCK);

    setenv("LLAMA_ARG_LOAD_MODE", "mmap+mlock", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK);

    setenv("LLAMA_ARG_LOAD_MODE", "dio", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_DIRECT_IO);

    printf("test-arg-parser: test negated environment variables\n\n");

    setenv("LLAMA_ARG_LOAD_MODE", "none", true);
    setenv("LLAMA_ARG_NO_PERF", "1", true); // legacy format
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_NONE);
    assert(params.no_perf == true);

    printf("test-arg-parser: test environment variables being overwritten\n\n");

    setenv("LLAMA_ARG_MODEL", "blah.gguf", true);
    setenv("LLAMA_ARG_THREADS", "1010", true);
    argv = {"binary_name", "-m", "overwritten.gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "overwritten.gguf");
    assert(params.cpuparams.n_threads == 1010);

    {
        common_params kv_params;
        setenv("LLAMA_ARG_KV_UNIFIED_PER_SLOT", "2048", true);
        argv = {"binary_name"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), kv_params, LLAMA_EXAMPLE_SERVER));
        assert(kv_params.kv_unified_per_slot == 2048);
        unsetenv("LLAMA_ARG_KV_UNIFIED_PER_SLOT");
    }
#endif // _WIN32

    printf("test-arg-parser: test download functions\n\n");
    const char * GOOD_URL = "http://ggml.ai/";
    const char * BAD_URL  = "http://ggml.ai/404";

    {
        printf("test-arg-parser: test good URL\n\n");
        auto res = common_remote_get_content(GOOD_URL, {});
        assert(res.first == 200);
        assert(res.second.size() > 0);
        std::string str(res.second.data(), res.second.size());
        assert(str.find("llama.cpp") != std::string::npos);
    }

    {
        printf("test-arg-parser: test bad URL\n\n");
        auto res = common_remote_get_content(BAD_URL, {});
        assert(res.first == 404);
    }

    {
        printf("test-arg-parser: test max size error\n");
        common_remote_params params;
        params.max_size = 1;
        try {
            common_remote_get_content(GOOD_URL, params);
            assert(false && "it should throw an error");
        } catch (std::exception & e) {
            printf("  expected error: %s\n\n", e.what());
        }
    }

    printf("test-arg-parser: all tests OK\n\n");
}

int main(void) {
    try {
        test();
    } catch (std::exception & e) {
        fprintf(stderr, "test-arg-parser: exception: %s\n", e.what());
        return 1;
    }
    return 0;
}
