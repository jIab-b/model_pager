#include "page_table.hpp"
#include <cuda_runtime.h>
#include <mutex>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <algorithm>

static void* g_model_um_base = nullptr;
static std::size_t g_model_total_bytes = 0;
static std::mutex g_model_mutex;

static std::vector<std::size_t> g_file_offsets;
static std::vector<std::size_t> g_sizes;
static std::vector<std::size_t> g_um_offsets;
static std::size_t g_planned_total_bytes = 0;

static inline std::size_t align_up(std::size_t x, std::size_t a) {
    return (x + a - 1) / a * a;
}

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

extern "C" void model_set_weights_layout(const std::size_t* file_offsets,
                                          const std::size_t* sizes,
                                          int count) {
    std::lock_guard<std::mutex> lk(g_model_mutex);
    g_file_offsets.assign(file_offsets, file_offsets + count);
    g_sizes.assign(sizes, sizes + count);
    std::vector<int> idx(count);
    for (int i = 0; i < count; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b){ return g_file_offsets[a] < g_file_offsets[b]; });
    g_um_offsets.assign(count, 0);
    std::size_t off = 0;
    for (int k = 0; k < count; ++k) {
        int i = idx[k];
        off = align_up(off, PAGER_PAGE_BYTES);
        g_um_offsets[i] = off;
        off += g_sizes[i];
    }
    g_planned_total_bytes = align_up(off, PAGER_PAGE_BYTES);
}

extern "C" void model_stage_file(const char* path, std::size_t chunk_bytes) {
    std::lock_guard<std::mutex> lk(g_model_mutex);
    if (!g_model_um_base || g_um_offsets.empty() || g_sizes.empty()) return;
    std::ifstream in(path, std::ios::binary);
    if (!in) return;
    std::vector<char> buf;
    buf.resize(chunk_bytes ? chunk_bytes : (8u << 20));
    const std::size_t n = g_sizes.size();
    for (std::size_t i = 0; i < n; ++i) {
        std::size_t remain = g_sizes[i];
        std::size_t file_off = g_file_offsets[i];
        std::size_t um_off = g_um_offsets[i];
        char* dst = static_cast<char*>(g_model_um_base) + um_off;
        while (remain) {
            std::size_t chunk = std::min(remain, static_cast<std::size_t>(buf.size()));
            in.seekg(static_cast<std::streamoff>(file_off), std::ios::beg);
            in.read(buf.data(), static_cast<std::streamsize>(chunk));
            std::size_t got = static_cast<std::size_t>(in.gcount());
            if (got == 0) return;
            std::memcpy(dst, buf.data(), got);
            remain -= got;
            file_off += got;
            dst += got;
        }
    }
}

extern "C" std::size_t model_planned_bytes() {
    std::lock_guard<std::mutex> lk(g_model_mutex);
    return g_planned_total_bytes;
}

// removed model_touch
