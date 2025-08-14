#pragma once
#include <cstddef>

// Page granularity (64 KiB)
constexpr std::size_t PAGER_PAGE_BYTES = 64 * 1024;

extern "C" {
    // pager core
    void* pager_reserve(std::size_t bytes);
    void  pager_prefetch(const void* pages, int n);
    void  pager_evict(const void* pages, int n);

    // single-model API
    void  model_reserve(std::size_t bytes);
    void  model_prefetch();
    void  model_evict();

    // page table & staging
    void  model_set_weights_layout(const std::size_t* file_offsets,
                                   const std::size_t* sizes,
                                   int count);
    void  model_stage_file(const char* path, std::size_t chunk_bytes);
    std::size_t model_planned_bytes();

    // memory stats
    void  get_memory_stats(std::size_t* uma_reserved,
                           std::size_t* uma_used,
                           std::size_t* gpu_allocated,
                           std::size_t* gpu_free);
}
