#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <atomic>
#include <mutex>

namespace {
std::atomic<int64_t> g_allocated{0};
int64_t g_cap = -1;

void* real_alloc(size_t nbytes) {
    return c10::cuda::CUDACachingAllocator::raw_alloc(nbytes);
}

void real_free(void* ptr) {
    c10::cuda::CUDACachingAllocator::raw_delete(ptr);
}

void* cap_alloc(size_t nbytes) {
    if (g_cap > 0 && g_allocated.load() + static_cast<int64_t>(nbytes) > g_cap) {
        // try freeing cached blocks first
        c10::cuda::CUDACachingAllocator::emptyCache();
        if (g_allocated.load() + static_cast<int64_t>(nbytes) > g_cap) {
            throw std::runtime_error("VRAM cap exceeded (alloc hook)");
        }
    }
    void* ptr = real_alloc(nbytes);
    g_allocated += static_cast<int64_t>(nbytes);
    return ptr;
}

void cap_free(void* ptr) {
    size_t sz = c10::cuda::CUDACachingAllocator::memorySize(ptr);
    g_allocated -= static_cast<int64_t>(sz);
    real_free(ptr);
}

bool installed = false;

void install_cap_allocator() {
    if (!installed) {
        c10::cuda::CUDACachingAllocator::raw_alloc = cap_alloc;
        c10::cuda::CUDACachingAllocator::raw_delete = cap_free;
        installed = true;
    }
}
} // anon

void set_cap(int64_t bytes) {
    g_cap = bytes;
    install_cap_allocator();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("set_cap", &set_cap, "Set VRAM cap (bytes) and install allocator");
}
