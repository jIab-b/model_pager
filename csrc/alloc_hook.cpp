#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <atomic>
#include <mutex>

namespace {
std::atomic<int64_t> g_allocated{0};
int64_t g_cap = -1;

void* raw_alloc(size_t nbytes) {
    if (g_cap > 0 && g_allocated.load() + static_cast<int64_t>(nbytes) > g_cap) {
        throw std::runtime_error("VRAM cap exceeded (alloc hook)");
    }
    auto* ptr = c10::cuda::CUDACachingAllocator::raw_alloc(nbytes);
    g_allocated += static_cast<int64_t>(nbytes);
    return ptr;
}

void raw_delete(void* ptr) {
    size_t sz = c10::cuda::CUDACachingAllocator::memorySize(ptr);
    g_allocated -= static_cast<int64_t>(sz);
    c10::cuda::CUDACachingAllocator::raw_delete(ptr);
}

} // anon

void set_cap(int64_t bytes) {
    g_cap = bytes;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("set_cap", &set_cap, "Set VRAM cap (bytes)");
}
