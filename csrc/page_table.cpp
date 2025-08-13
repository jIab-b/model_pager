#include "page_table.hpp"
#include <cuda_runtime.h>
#include <mutex>

static void* g_model_um_base = nullptr;
static std::size_t g_model_total_bytes = 0;
static std::mutex g_model_mutex;

extern "C" void model_reserve(std::size_t bytes) {
    std::lock_guard<std::mutex> lk(g_model_mutex);
    g_model_total_bytes = bytes;
    g_model_um_base = pager_reserve(bytes);
}

extern "C" void model_prefetch() {
    std::lock_guard<std::mutex> lk(g_model_mutex);
    if (!g_model_um_base) return;
    int n = (g_model_total_bytes + PAGER_PAGE_BYTES - 1) / PAGER_PAGE_BYTES;
    pager_prefetch(g_model_um_base, n);
}

extern "C" void model_evict() {
    std::lock_guard<std::mutex> lk(g_model_mutex);
    if (!g_model_um_base) return;
    int n = (g_model_total_bytes + PAGER_PAGE_BYTES - 1) / PAGER_PAGE_BYTES;
    pager_evict(g_model_um_base, n);
}

extern "C" void get_memory_stats(std::size_t* uma_reserved,
                                  std::size_t* uma_used,
                                  std::size_t* gpu_allocated,
                                  std::size_t* gpu_free) {
    size_t free_b, total_b;
    cudaMemGetInfo(&free_b, &total_b);
    *gpu_free = free_b;
    *gpu_allocated = total_b - free_b;
    std::lock_guard<std::mutex> lk(g_model_mutex);
    *uma_reserved = g_model_total_bytes;
    *uma_used = g_model_total_bytes;
}
