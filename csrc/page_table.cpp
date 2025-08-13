#include "page_table.hpp"
#include <cuda_runtime.h>
#include <algorithm>

static std::vector<ModelTable> g_models;

static ModelTable* find_model(const char* id) {
    for (auto &mt : g_models)
        if (std::string(mt.id) == id)
            return &mt;
    return nullptr;
}

extern "C" void register_model(const char* id,
                               const char* safetensors_path,
                               int n_entries,
                               const char** module_names,
                               const std::size_t* offsets,
                               const std::size_t* sizes) {
    ModelTable mt;
    mt.id = id;
    mt.total_bytes = 0;
    for (int i = 0; i < n_entries; ++i)
        mt.total_bytes += sizes[i];
    mt.um_base = pager_reserve(mt.total_bytes);
    mt.pages.reserve(n_entries);
    for (int i = 0; i < n_entries; ++i) {
        PageEntry pe{module_names[i], offsets[i], sizes[i], mt.um_base, 0};
        mt.pages.push_back(pe);
    }
    g_models.push_back(mt);
}

extern "C" void acquire_pages(const char* id) {
    ModelTable* mt = find_model(id);
    if (!mt) return;
    for (auto &pe : mt->pages) {
        std::size_t page_off = (pe.offset / PAGER_PAGE_BYTES) * PAGER_PAGE_BYTES;
        int n = (pe.size + PAGER_PAGE_BYTES - 1) / PAGER_PAGE_BYTES;
        void* ptr = static_cast<char*>(mt->um_base) + page_off;
        pager_prefetch(ptr, n);
    }
}

extern "C" void release_pages(const char* id) {
    ModelTable* mt = find_model(id);
    if (!mt) return;
    for (auto &pe : mt->pages) {
        std::size_t page_off = (pe.offset / PAGER_PAGE_BYTES) * PAGER_PAGE_BYTES;
        int n = (pe.size + PAGER_PAGE_BYTES - 1) / PAGER_PAGE_BYTES;
        void* ptr = static_cast<char*>(mt->um_base) + page_off;
        pager_evict(ptr, n);
    }
}

extern "C" void update_schedule(const char** ids, const int* priorities, int n) {
    // no-op
}

extern "C" void process_schedule() {
    // no-op
}

extern "C" void get_memory_stats(std::size_t* uma_reserved,
                                  std::size_t* uma_used,
                                  std::size_t* gpu_allocated,
                                  std::size_t* gpu_free) {
    size_t free_b, total_b;
    cudaMemGetInfo(&free_b, &total_b);
    *gpu_free = free_b;
    *gpu_allocated = total_b - free_b;
    std::size_t reserved = 0;
    for (auto &mt : g_models)
        reserved += mt.total_bytes;
    *uma_reserved = reserved;
    *uma_used = reserved;
}
