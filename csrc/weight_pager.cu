#include "weight_pager.hpp"
#include <cuda_runtime.h>
#include <cstdio>

extern "C" void* pager_reserve(std::size_t bytes) {
    printf("[pager] reserve %zu bytes (stub)\n", bytes);
    return nullptr;
}

extern "C" void pager_prefetch(const void*, int) { /* no-op */ }
extern "C" void pager_evict   (const void*, int) { /* no-op */ }
