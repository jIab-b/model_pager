#pragma once
#include <cuda.h>
#include <cstdint>

extern "C" {
void* pager_reserve(std::size_t bytes);
void  pager_prefetch(const void* pages, int n);
void  pager_evict(const void* pages, int n);
}
