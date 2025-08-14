#include "page_table.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {
struct ModelEntry {
    int fd;
    std::vector<std::size_t> offsets;
    std::vector<std::size_t> sizes;
};

static std::mutex g_mutex;
static int g_device = 0;
static std::vector<cudaStream_t> g_compute_streams;
static std::vector<cudaStream_t> g_io_streams;
static std::vector<ModelEntry> g_models;
static std::size_t g_chunk_bytes = 8u << 20; // 8 MiB default chunk size

struct HBuf {
    void* p[2] = {nullptr, nullptr};
    cudaEvent_t ev[2] = {nullptr, nullptr};
    bool inited = false;
};
static std::vector<HBuf> g_io_hbufs;

static inline bool valid_handle(int h) {
    return h >= 0 && static_cast<std::size_t>(h) < g_models.size() && g_models[h].fd >= 0;
}

static void* alloc_pinned(std::size_t bytes) {
    void* ptr = nullptr;
    cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault);
    return ptr;
}

static void free_pinned(void* ptr) {
    if (ptr) cudaFreeHost(ptr);
}
} // namespace

extern "C" void runtime_init(int device_id, int compute_streams, int io_streams) {
    std::lock_guard<std::mutex> lk(g_mutex);
    g_device = device_id;
    cudaSetDevice(g_device);
    g_compute_streams.resize(std::max(0, compute_streams));
    for (auto& s : g_compute_streams) cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    g_io_streams.resize(std::max(1, io_streams));
    for (auto& s : g_io_streams) cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    g_io_hbufs.resize(g_io_streams.size());
    for (size_t i = 0; i < g_io_hbufs.size(); ++i) {
        HBuf& hb = g_io_hbufs[i];
        for (int k = 0; k < 2; ++k) {
            hb.p[k] = alloc_pinned(g_chunk_bytes);
            cudaEventCreateWithFlags(&hb.ev[k], cudaEventDisableTiming);
        }
        hb.inited = true;
    }
}

extern "C" void runtime_shutdown() {
    std::lock_guard<std::mutex> lk(g_mutex);
    for (auto& s : g_compute_streams) cudaStreamDestroy(s);
    g_compute_streams.clear();
    for (auto& s : g_io_streams) cudaStreamDestroy(s);
    g_io_streams.clear();
    for (auto& hb : g_io_hbufs) {
        for (int k = 0; k < 2; ++k) {
            if (hb.ev[k]) { cudaEventDestroy(hb.ev[k]); hb.ev[k] = nullptr; }
            if (hb.p[k]) { free_pinned(hb.p[k]); hb.p[k] = nullptr; }
        }
        hb.inited = false;
    }
    g_io_hbufs.clear();
    for (auto& m : g_models) {
        if (m.fd >= 0) close(m.fd);
        m.fd = -1;
    }
    g_models.clear();
}

extern "C" int model_register(const char* path,
                               const std::size_t* file_offsets,
                               const std::size_t* sizes,
                               int count) {
    std::lock_guard<std::mutex> lk(g_mutex);
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return -1;
    (void)posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);
    (void)posix_fadvise(fd, 0, 0, POSIX_FADV_NOREUSE);
    ModelEntry e;
    e.fd = fd;
    e.offsets.assign(file_offsets, file_offsets + count);
    e.sizes.assign(sizes, sizes + count);
    g_models.push_back(std::move(e));
    return static_cast<int>(g_models.size() - 1);
}

extern "C" void model_close(int handle) {
    std::lock_guard<std::mutex> lk(g_mutex);
    if (!valid_handle(handle)) return;
    auto& m = g_models[handle];
    if (m.fd >= 0) close(m.fd);
    m.fd = -1;
    m.offsets.clear();
    m.sizes.clear();
}

extern "C" std::size_t model_tensor_size(int handle, int index) {
    std::lock_guard<std::mutex> lk(g_mutex);
    if (!valid_handle(handle)) return 0;
    const auto& m = g_models[handle];
    if (index < 0 || static_cast<std::size_t>(index) >= m.sizes.size()) return 0;
    return m.sizes[static_cast<std::size_t>(index)];
}

extern "C" int gds_available() {
    return 0;
}

extern "C" void model_read_into(int handle,
                                 int index,
                                 void* dst_device_ptr,
                                 std::size_t bytes,
                                 int io_stream) {
    std::lock_guard<std::mutex> lk(g_mutex);
    if (!valid_handle(handle)) return;
    auto& m = g_models[handle];
    if (index < 0 || static_cast<std::size_t>(index) >= m.offsets.size()) return;
    std::size_t off = m.offsets[static_cast<std::size_t>(index)];
    std::size_t to_read = std::min<std::size_t>(bytes, m.sizes[static_cast<std::size_t>(index)]);

    int stream_idx = std::min<int>(std::max(0, io_stream), static_cast<int>(g_io_streams.size()) - 1);
    HBuf& hb = g_io_hbufs[static_cast<size_t>(stream_idx)];
    char* dptr = static_cast<char*>(dst_device_ptr);
    std::size_t remaining = to_read;
    std::size_t file_off = off;
    std::size_t dev_off = 0;
    int toggle = 0;
    while (remaining) {
        if (hb.ev[toggle]) cudaEventSynchronize(hb.ev[toggle]);
        std::size_t chunk = std::min<std::size_t>(remaining, g_chunk_bytes);
        off_t this_off = static_cast<off_t>(file_off);
        ssize_t got = pread(m.fd, hb.p[toggle], static_cast<size_t>(chunk), this_off);
        if (got <= 0) break;
        cudaMemcpyAsync(dptr + dev_off, hb.p[toggle], static_cast<size_t>(got), cudaMemcpyHostToDevice, g_io_streams[stream_idx]);
        cudaEventRecord(hb.ev[toggle], g_io_streams[stream_idx]);
        (void)posix_fadvise(m.fd, this_off, static_cast<off_t>(got), POSIX_FADV_DONTNEED);
        remaining -= static_cast<std::size_t>(got);
        file_off += static_cast<std::size_t>(got);
        dev_off  += static_cast<std::size_t>(got);
        toggle ^= 1;
    }
}

extern "C" void model_read_into_batch(int handle,
                                       const int* indices,
                                       void* const* dst_device_ptrs,
                                       const std::size_t* bytes,
                                       int n,
                                       int io_stream) {
    for (int i = 0; i < n; ++i) {
        model_read_into(handle, indices[i], dst_device_ptrs[i], bytes[i], io_stream);
    }
}

extern "C" void wait_all() {
    std::lock_guard<std::mutex> lk(g_mutex);
    for (auto& s : g_compute_streams) cudaStreamSynchronize(s);
    for (auto& s : g_io_streams) cudaStreamSynchronize(s);
}

extern "C" void get_memory_stats(std::size_t* gpu_allocated,
                                  std::size_t* gpu_free) {
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    if (gpu_free) *gpu_free = free_b;
    if (gpu_allocated) *gpu_allocated = total_b - free_b;
}


