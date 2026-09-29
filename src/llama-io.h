#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

struct ggml_tensor;

class llama_io_write_i {
public:
    llama_io_write_i() = default;
    virtual ~llama_io_write_i() = default;

    virtual void write(const void * src, size_t size) = 0;
    virtual void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // Write n_rows rows of row_size bytes each, taken from tensor at offset + i*row_stride.
    // The rows are appended to the stream contiguously. This is the bulk form of a transposed
    // cache region, where one row is one embedding across the cells and row_stride is the
    // embedding stride. A row count of 1 is equivalent to write_tensor.
    virtual void write_tensor_strided(ggml_tensor * tensor, size_t offset, size_t row_size, size_t n_rows, size_t row_stride) = 0;

    // bytes written so far
    virtual size_t n_bytes() = 0;

    void write_string(const std::string & str);
};

class llama_io_read_i {
public:
    llama_io_read_i() = default;
    virtual ~llama_io_read_i() = default;

    virtual void read(void * dst, size_t size) = 0;
    virtual void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // Read n_rows rows of row_size bytes each from the stream, which are contiguous, and place
    // them in tensor at offset + i*row_stride. The inverse of write_tensor_strided.
    virtual void read_tensor_strided(ggml_tensor * tensor, size_t offset, size_t row_size, size_t n_rows, size_t row_stride) = 0;

    // bytes read so far
    virtual size_t n_bytes() = 0;

    void read_string(std::string & str);
};
