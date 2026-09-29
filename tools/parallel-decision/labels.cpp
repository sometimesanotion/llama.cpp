#include "labels.h"

#include "llama.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

namespace llama_decision {

namespace {

class llama_label_vocab : public label_vocab {
public:
    explicit llama_label_vocab(const llama_vocab * vocab) : vocab(vocab) {}

    std::vector<int32_t> tokenize(const std::string & text, bool parse_special) const override {
        const int n = llama_tokenize(vocab, text.data(), (int) text.size(), nullptr, 0, false, parse_special);
        if (n == 0) {
            return {};
        }
        std::vector<llama_token> tokens((size_t) std::abs(n));
        const int m = llama_tokenize(vocab, text.data(), (int) text.size(), tokens.data(), (int) tokens.size(), false, parse_special);
        if (m < 0) {
            throw std::runtime_error("tokenization failed");
        }
        tokens.resize((size_t) m);
        return std::vector<int32_t>(tokens.begin(), tokens.end());
    }

    std::string piece(int32_t token) const override {
        char buf[128];
        const int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, false);
        if (n < 0) {
            return {};
        }
        return std::string(buf, (size_t) n);
    }

    bool is_special(int32_t token) const override {
        return llama_vocab_is_control(vocab, token) || llama_vocab_is_eog(vocab, token);
    }

private:
    const llama_vocab * vocab;
};

} // namespace

std::unique_ptr<label_vocab> make_llama_label_vocab(const llama_vocab * vocab) {
    return std::make_unique<llama_label_vocab>(vocab);
}

int32_t answer_label_token(const label_vocab & vocab, const std::string & tail, const std::string & text) {
    const std::vector<int32_t> path = answer_label_path(vocab, tail, text, 1);
    return path.size() == 1 ? path[0] : -1;
}

std::vector<int32_t> answer_label_path(const label_vocab & vocab, const std::string & tail,
                                       const std::string & text, int max_len) {
    const std::vector<int32_t> with_text = vocab.tokenize(tail + text, true);
    const std::vector<int32_t> tail_only = vocab.tokenize(tail, true);
    if (with_text.size() <= tail_only.size()) {
        return {};
    }
    const size_t extra = with_text.size() - tail_only.size();
    if ((int) extra > max_len) {
        return {};
    }
    for (size_t i = 0; i < tail_only.size(); ++i) {
        if (with_text[i] != tail_only[i]) {
            return {};
        }
    }
    std::vector<int32_t> path(with_text.begin() + (ptrdiff_t) tail_only.size(), with_text.end());
    for (const int32_t token : path) {
        if (vocab.is_special(token)) {
            return {};
        }
    }
    return path;
}

std::vector<label> build_label_pool(const label_vocab & vocab, const std::string & tail, size_t cap) {
    // Single-character candidates first, ordered for readability: A-Z, a-z, 0-9, the base64
    // extras and safe ASCII symbols, then accented Latin, Greek and Cyrillic single characters.
    // The two-character composed labels are the fallback for a tokenizer that cannot resolve
    // enough single characters as single tokens. The realized pool is model-dependent: every
    // kept label must resolve as a unique, non-special 1-2 token path at the answer boundary.
    std::vector<std::string> candidates;
    auto one = [&](char c) { candidates.push_back(std::string(1, c)); };
    for (char a = 'A'; a <= 'Z'; ++a) {
        one(a);
    }
    for (char a = 'a'; a <= 'z'; ++a) {
        one(a);
    }
    for (char d = '0'; d <= '9'; ++d) {
        one(d);
    }
    // printable ASCII symbols that are unambiguous as a label and safe in the option line
    for (char c : std::string("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")) {
        one(c);
    }
    // accented Latin and common Greek/Cyrillic single characters, so a tokenizer that keeps them
    // as single tokens can realize the full label pool
    for (const char * s : {
            "\xC3\x80","\xC3\x81","\xC3\x82","\xC3\x83","\xC3\x84","\xC3\x85","\xC3\x86","\xC3\x87",
            "\xC3\x88","\xC3\x89","\xC3\x8A","\xC3\x8B","\xC3\x8C","\xC3\x8D","\xC3\x8E","\xC3\x8F",
            "\xC3\x90","\xC3\x91","\xC3\x92","\xC3\x93","\xC3\x94","\xC3\x95","\xC3\x96","\xC3\x97",
            "\xC3\x98","\xC3\x99","\xC3\x9A","\xC3\x9B","\xC3\x9C","\xC3\x9D","\xC3\x9E","\xC3\x9F",
            "\xC3\xA0","\xC3\xA1","\xC3\xA2","\xC3\xA3","\xC3\xA4","\xC3\xA5","\xC3\xA6","\xC3\xA7",
            "\xC3\xA8","\xC3\xA9","\xC3\xAA","\xC3\xAB","\xC3\xAC","\xC3\xAD","\xC3\xAE","\xC3\xAF",
            "\xC3\xB0","\xC3\xB1","\xC3\xB2","\xC3\xB3","\xC3\xB4","\xC3\xB5","\xC3\xB6","\xC3\xB7",
            "\xC3\xB8","\xC3\xB9","\xC3\xBA","\xC3\xBB","\xC3\xBC","\xC3\xBD","\xC3\xBE","\xC3\xBF",
            "\xCE\x91","\xCE\x92","\xCE\x93","\xCE\x94","\xCE\x95","\xCE\x96","\xCE\x97","\xCE\x98",
            "\xCE\x99","\xCE\x9A","\xCE\x9B","\xCE\x9C","\xCE\x9D","\xCE\x9E","\xCE\x9F","\xCE\xA0",
            "\xCE\xA1","\xCE\xA3","\xCE\xA4","\xCE\xA5","\xCE\xA6","\xCE\xA7","\xCE\xA8","\xCE\xA9",
            "\xCE\xB1","\xCE\xB2","\xCE\xB3","\xCE\xB4","\xCE\xB5","\xCE\xB6","\xCE\xB7","\xCE\xB8",
            "\xCE\xB9","\xCE\xBA","\xCE\xBB","\xCE\xBC","\xCE\xBD","\xCE\xBE","\xCE\xBF","\xCF\x80",
            "\xCF\x81","\xCF\x82","\xCF\x83","\xCF\x84","\xCF\x85","\xCF\x86","\xCF\x87","\xCF\x88",
            "\xCF\x89","\xD0\x90","\xD0\x91","\xD0\x92","\xD0\x93","\xD0\x94","\xD0\x95","\xD0\x96",
            "\xD0\x97","\xD0\x98","\xD0\x99","\xD0\x9A","\xD0\x9B","\xD0\x9C","\xD0\x9D","\xD0\x9E",
            "\xD0\x9F","\xD0\xA0","\xD0\xA1","\xD0\xA2","\xD0\xA3","\xD0\xA4","\xD0\xA5","\xD0\xA6",
            "\xD0\xA7","\xD0\xA8","\xD0\xA9","\xD0\xAA","\xD0\xAB","\xD0\xAC","\xD0\xAD","\xD0\xAE",
            "\xD0\xAF","\xD0\xB0","\xD0\xB1","\xD0\xB2","\xD0\xB3","\xD0\xB4","\xD0\xB5","\xD0\xB6",
            "\xD0\xB7","\xD0\xB8","\xD0\xB9","\xD0\xBA","\xD0\xBB","\xD0\xBC","\xD0\xBD","\xD0\xBE",
            "\xD0\xBF","\xD1\x80","\xD1\x81","\xD1\x82","\xD1\x83","\xD1\x84","\xD1\x85","\xD1\x86",
            "\xD1\x87","\xD1\x88","\xD1\x89","\xD1\x8A","\xD1\x8B","\xD1\x8C","\xD1\x8D","\xD1\x8E",
            "\xD1\x8F" }) {
        candidates.push_back(s);
    }
    auto two = [&](char a, char b) { candidates.push_back(std::string{ a, b }); };
    for (char a = 'A'; a <= 'Z'; ++a) {
        for (char b = 'A'; b <= 'Z'; ++b) {
            two(a, b); // AA..ZZ
        }
    }
    for (char a = 'A'; a <= 'Z'; ++a) {
        for (char d = '0'; d <= '9'; ++d) {
            two(a, d); // A0..Z9
        }
    }
    for (char d = '0'; d <= '9'; ++d) {
        for (char a = 'A'; a <= 'Z'; ++a) {
            two(d, a); // 0A..9Z
        }
    }
    for (char d1 = '0'; d1 <= '9'; ++d1) {
        for (char d2 = '0'; d2 <= '9'; ++d2) {
            two(d1, d2); // 00..99
        }
    }

    std::vector<label> pool;
    auto add = [&](const std::string & text) {
        const std::vector<int32_t> path = answer_label_path(vocab, tail, text, 2);
        if (path.empty()) {
            return;
        }
        for (const label & l : pool) {
            if (l.tokens == path) {
                return;
            }
        }
        label l;
        l.text   = text;
        l.tokens = path;
        l.token  = path.size() == 1 ? path[0] : -1;
        pool.push_back(std::move(l));
    };
    // single-token labels first, then two-token ones: the answer head can only score length-1
    // paths, so ordering the pool this way keeps the cheap head path available for as many
    // options as the tokenizer's single-token coverage allows
    for (const std::string & text : candidates) {
        add(text);
    }
    std::stable_partition(pool.begin(), pool.end(), [](const label & l) { return l.tokens.size() == 1; });
    if (pool.size() > cap) {
        pool.resize(cap);
    }

    if (pool.size() < 2) {
        throw std::runtime_error("no suitable answer tokens in the vocabulary");
    }
    return pool;
}

std::string safe_data(const std::string & text) {
    std::string out;
    out.reserve(text.size());
    for (char c : text) {
        if (c == '<') {
            out += "\\u003c";
        } else {
            out += c;
        }
    }
    return out;
}

} // namespace llama_decision
