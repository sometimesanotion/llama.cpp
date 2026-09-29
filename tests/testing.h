#pragma once

#include "common.h"

#include <chrono>
#include <exception>
#include <iostream>
#include <string>
#include <regex>
#include <vector>

struct testing {
    std::ostream &out;
    std::vector<std::string> stack;
    std::regex filter;
    bool filter_tests = false;
    bool throw_exception = false;
    bool verbose = false;
    int tests = 0;
    int assertions = 0;
    int failures = 0;
    int unnamed = 0;
    int exceptions = 0;
    int skipped = 0;
    int expected_failures = 0;
    int unexpected_passes = 0;

    // set by skip(), read by the innermost test()
    bool skip_current = false;
    std::string skip_reason;

    static constexpr std::size_t status_column = 80;

    explicit testing(std::ostream &os = std::cout) : out(os) {}

    std::string indent() const {
        if (stack.empty()) {
            return "";
        }
        return std::string((stack.size() - 1) * 2, ' ');
    }

    std::string full_name() const {
        return string_join(stack, ".");
    }

    void log(const std::string & msg) {
        if (verbose) {
            out << indent() << "  " << msg << "\n";
        }
    }

    void set_filter(const std::string & re) {
        filter = std::regex(re);
        filter_tests = true;
    }

    bool should_run() const {
        if (filter_tests) {
            if (!std::regex_match(full_name(), filter)) {
                return false;
            }
        }
        return true;
    }

    template <typename F>
    void run_with_exceptions(F &&f, const char *ctx) {
        try {
            f();
        } catch (const std::exception &e) {
            ++failures;
            ++exceptions;
            out << indent() << "UNHANDLED EXCEPTION (" << ctx << "): " << e.what() << "\n";
            if (throw_exception) {
                throw;
            }
        } catch (...) {
            ++failures;
            ++exceptions;
            out << indent() << "UNHANDLED EXCEPTION (" << ctx << "): unknown\n";
            if (throw_exception) {
                throw;
            }
        }
    }

    void skip(const std::string &reason = "") {
        skip_current = true;
        skip_reason  = reason;
    }

    void print_result(const std::string &label, int new_failures, int new_assertions, const std::string &extra = "", bool was_skipped = false, const char * status_override = nullptr) const {
        std::string line = indent() + label;

        std::string details;
        if (new_assertions > 0) {
            if (new_failures == 0) {
                details = std::to_string(new_assertions) + " assertion(s)";
            } else {
                details = std::to_string(new_failures) + " of " +
                          std::to_string(new_assertions) + " assertion(s) failed";
            }
        }
        if (!extra.empty()) {
            if (!details.empty()) {
                details += ", ";
            }
            details += extra;
        }

        if (!details.empty()) {
            line += " (" + details + ")";
        }

        const std::string status = status_override != nullptr ? status_override
            : (new_failures != 0 ? "[FAIL]" : (was_skipped ? "[SKIP]" : "[PASS]"));

        if (line.size() + 1 < status_column) {
            line.append(status_column - line.size(), ' ');
        } else {
            line.push_back(' ');
        }

        out << line << status << "\n";
    }

    template <typename F>
    void test(const std::string &name, F f) {
        stack.push_back(name);
        if (!should_run()) {
            stack.pop_back();
            return;
        }

        ++tests;
        out << indent() << name << "\n";

        int before_failures   = failures;
        int before_assertions = assertions;

        // do not let a skipped subtest also mark its parent as skipped
        bool        outer_skip        = skip_current;
        std::string outer_skip_reason = skip_reason;
        skip_current = false;
        skip_reason.clear();

        run_with_exceptions([&] { f(*this); }, "test");

        int new_failures   = failures   - before_failures;
        int new_assertions = assertions - before_assertions;

        bool was_skipped = skip_current && new_failures == 0;
        if (was_skipped) {
            ++skipped;
        }

        print_result(name, new_failures, new_assertions, was_skipped ? skip_reason : "", was_skipped);

        skip_current = outer_skip;
        skip_reason  = outer_skip_reason;

        stack.pop_back();
    }

    template <typename F>
    void test(F f) {
        test("test #" + std::to_string(++unnamed), f);
    }

    // A subtest that is known to fail on the current code. It stays green while the expected
    // failure is present, and turns red when it passes, so the marker is removed once fixed.
    // An unexpected pass (XPASS) is a hard failure, not a silent success. xfail marks
    // producer-certainty (bit-stability) properties only; it must never absorb a task-value
    // failure such as a winner or outcome disagreement.
    template <typename F>
    void xfail(const std::string & name, F f) {
        stack.push_back(name);
        if (!should_run()) {
            stack.pop_back();
            return;
        }

        ++tests;
        out << indent() << name << "\n";

        int before_failures   = failures;
        int before_assertions = assertions;

        bool        outer_skip        = skip_current;
        std::string outer_skip_reason = skip_reason;
        skip_current = false;
        skip_reason.clear();

        run_with_exceptions([&] { f(*this); }, "test");

        int new_failures   = failures   - before_failures;
        int new_assertions = assertions - before_assertions;

        if (skip_current && new_failures == 0) {
            ++skipped;
            print_result(name, 0, new_assertions, skip_reason, true);
        } else if (new_failures > 0) {
            // the known failure is present: drop it so the suite stays green
            failures = before_failures;
            ++expected_failures;
            print_result(name, 0, new_assertions, "known failure, not counted", false, "[XFAIL]");
        } else {
            ++unexpected_passes;
            ++failures;
            print_result(name, 1, new_assertions, "passed unexpectedly; remove the expected-failure marker", false, "[XPASS]");
        }

        skip_current = outer_skip;
        skip_reason  = outer_skip_reason;

        stack.pop_back();
    }

    template <typename F>
    void bench(const std::string &name, F f, int iterations = 100) {
        stack.push_back(name);
        if (!should_run()) {
            stack.pop_back();
            return;
        }

        ++tests;
        out << indent() << "[bench] " << name << "\n";

        int before_failures   = failures;
        int before_assertions = assertions;

        using clock = std::chrono::high_resolution_clock;

        std::chrono::microseconds duration(0);

        run_with_exceptions([&] {
            for (auto i = 0; i < iterations; i++) {
                auto start = clock::now();
                f();
                duration += std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - start);
            }
        }, "bench");

        auto avg_elapsed   = duration.count() / iterations;
        auto avg_elapsed_s = std::chrono::duration_cast<std::chrono::duration<double>>(duration).count() / iterations;
        auto rate = (avg_elapsed_s > 0.0) ? (1.0 / avg_elapsed_s) : 0.0;

        int new_failures   = failures   - before_failures;
        int new_assertions = assertions - before_assertions;

        std::string extra =
            "n=" + std::to_string(iterations) +
            " avg=" + std::to_string(avg_elapsed) + "us" +
            " rate=" + std::to_string(int(rate)) + "/s";

        print_result("[bench] " + name, new_failures, new_assertions, extra);

        stack.pop_back();
    }

    template <typename F>
    void bench(F f, int iterations = 100) {
        bench("bench #" + std::to_string(++unnamed), f, iterations);
    }

    // Assertions
    bool assert_true(bool cond) {
        return assert_true("", cond);
    }

    bool assert_true(const std::string &msg, bool cond) {
        ++assertions;
        if (!cond) {
            ++failures;
            out << indent() << "ASSERTION FAILED";
            if (!msg.empty()) {
                out << " : " << msg;
            }
            out << "\n";
            return false;
        }
        return true;
    }

    template <typename A, typename B>
    bool assert_equal(const A &expected, const B &actual) {
        return assert_equal("", expected, actual);
    }

    template <typename A, typename B>
    bool assert_equal(const std::string &msg, const A &expected, const B &actual) {
        ++assertions;
        if (!(actual == expected)) {
            ++failures;
            out << indent() << "ASSERT EQUAL FAILED";
            if (!msg.empty()) {
                out << " : " << msg;
            }
            out << "\n";

            out << indent() << "  expected: " << expected << "\n";
            out << indent() << "  actual  : " << actual << "\n";
            return false;
        }
        return true;
    }

    // Print summary and return an exit code
    int summary() const {
        out << "\n";
        out << "tests      : " << tests << "\n";
        out << "assertions : " << assertions << "\n";
        out << "failures   : " << failures << "\n";
        out << "exceptions : " << exceptions << "\n";
        out << "skipped    : " << skipped << "\n";
        if (expected_failures > 0 || unexpected_passes > 0) {
            out << "xfail      : " << expected_failures << "\n";
            out << "xpass      : " << unexpected_passes << "\n";
        }
        return failures == 0 ? 0 : 1;
    }
};
