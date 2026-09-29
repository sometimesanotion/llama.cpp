// File and environment helpers shared by the decision developer tools.
#pragma once

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace decision_file {

inline std::string env_str(const char * name) {
    const char * v = std::getenv(name);
    return v ? v : "";
}

inline std::string read_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot read " + path);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

} // namespace decision_file
