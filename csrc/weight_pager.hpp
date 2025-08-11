#pragma once
#include <cuda.h>
#include <cstdint>
#include <cstddef>

// Size of one logical page (64 KiB)
constexpr std::size_t PAGER_PAGE_BYTES = 64 * 1024;

extern "C" {
// Reserve `bytes` of unified-memory managed space and return device pointer.
void* pager_reserve(std::size_t bytes);
// Prefetch `n` pages (n * PAGER_PAGE_BYTES) starting at `pages` into current GPU.
void  pager_prefetch(const void* pages, int n);
// Evict `n` pages back to host.
void  pager_evict(const void* pages, int n);
}
