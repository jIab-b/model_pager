#pragma once

#include <cstddef>
#include <cstdint>

static constexpr std::size_t PAGER_PAGE_BYTES = 64u * 1024u;

extern "C" {
    void runtime_init(int device_id, int compute_streams, int io_streams);
    void runtime_shutdown();

    int model_register(const char* path,
                       const std::size_t* file_offsets,
                       const std::size_t* sizes,
                       int count);
    void model_close(int handle);
    std::size_t model_tensor_size(int handle, int index);
    int gds_available();

    void model_read_into(int handle,
                         int index,
                         void* dst_device_ptr,
                         std::size_t bytes,
                         int io_stream);

    void model_read_into_batch(int handle,
                               const int* indices,
                               void* const* dst_device_ptrs,
                               const std::size_t* bytes,
                               int n,
                               int io_stream);

    void wait_all();

    void get_memory_stats(std::size_t* gpu_allocated,
                          std::size_t* gpu_free);
}


