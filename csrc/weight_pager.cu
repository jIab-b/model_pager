#include "weight_pager.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <mutex>

// PyBind11 export is handled in um_tensor.cpp to avoid duplicate module symbol.

namespace {
static std::mutex g_pool_mtx;
static void*      g_pool_ptr  = nullptr;
static std::size_t g_pool_cap = 0;
static std::size_t g_pool_off = 0;

void ensure_pool(std::size_t min_bytes) {
    if (g_pool_ptr && g_pool_cap - g_pool_off >= min_bytes) return;
    // enlarge (double or at least min_bytes)
    std::size_t new_cap = g_pool_cap ? g_pool_cap * 2 : 1 << 26; // start 64MB
    while (new_cap - g_pool_off < min_bytes) new_cap *= 2;

    void* new_ptr = nullptr;
    cudaError_t err = cudaMallocManaged(&new_ptr, new_cap, cudaMemAttachGlobal);
    if (err != cudaSuccess) {
        printf("[pager] cudaMallocManaged failed: %s\n", cudaGetErrorString(err));
        return;
    }
    if (g_pool_ptr) {
        cudaMemcpy(new_ptr, g_pool_ptr, g_pool_off, cudaMemcpyDefault);
        cudaFree(g_pool_ptr);
    }
    g_pool_ptr = new_ptr;
    g_pool_cap = new_cap;
    printf("[pager] pool resize: %zu bytes\n", g_pool_cap);
}
} // anon

extern "C" void* pager_reserve(std::size_t bytes) {
    std::lock_guard<std::mutex> lk(g_pool_mtx);
    std::size_t aligned = (bytes + PAGER_PAGE_BYTES - 1) & ~(PAGER_PAGE_BYTES - 1);
    ensure_pool(aligned);
    void* ptr = static_cast<char*>(g_pool_ptr) + g_pool_off;
    g_pool_off += aligned;
    printf("[pager] reserve %zu bytes → %p (off=%zu)\n", aligned, ptr, g_pool_off);
    return ptr;
}

extern "C" void pager_prefetch(const void* pages, int n) {
    int dev;
    cudaGetDevice(&dev);
    std::size_t bytes = static_cast<std::size_t>(n) * PAGER_PAGE_BYTES;
    cudaMemPrefetchAsync(const_cast<void*>(pages), bytes, dev, 0);
}

extern "C" void pager_evict(const void* pages, int n) {
    std::size_t bytes = static_cast<std::size_t>(n) * PAGER_PAGE_BYTES;
    cudaMemPrefetchAsync(const_cast<void*>(pages), bytes, cudaCpuDeviceId, 0);
}
