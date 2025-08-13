#pragma once
#include <cuda.h>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <cuda_runtime.h>

// Size of one logical page (64 KiB)
constexpr std::size_t PAGER_PAGE_BYTES = 64 * 1024;

struct PageEntry {
    const char* module_name;
    std::size_t offset;
    std::size_t size;
    void* um_base;
    int num_pages;
};

struct ModelTable {
    const char* id;
    void* um_base;
    std::size_t total_bytes;
    std::vector<PageEntry> pages;
};

extern "C" {
    void* pager_reserve(std::size_t bytes);
    void  pager_prefetch(const void* pages, int n);
    void  pager_evict(const void* pages, int n);
    void  register_model(const char* id,
                        const char* safetensors_path,
                        int n_entries,
                        const char** module_names,
                        const std::size_t* offsets,
                        const std::size_t* sizes);
    void  acquire_pages(const char* id);
    void  release_pages(const char* id);
    void  update_schedule(const char** ids, const int* priorities, int n);
    void  process_schedule();
    void  get_memory_stats(std::size_t* uma_reserved,
                           std::size_t* uma_used,
                           std::size_t* gpu_allocated,
                           std::size_t* gpu_free);
}
