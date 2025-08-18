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
#include <unordered_map>
#include <list>

namespace {
struct ModelEntry {
    int fd;
    std::vector<std::size_t> offsets;
    std::vector<std::size_t> sizes;
};

static std::mutex g_mutex;
static std::vector<std::mutex> g_io_mutexes;
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

struct GpuCacheItem {
    void* dptr = nullptr;
    std::size_t bytes = 0;
    cudaEvent_t ready = nullptr;
};
struct CpuCacheItem {
    void* hptr = nullptr;
    std::size_t bytes = 0;
};
static std::unordered_map<long long, GpuCacheItem> g_gpu_cache;
static std::unordered_map<long long, CpuCacheItem> g_cpu_cache;
static std::list<std::pair<int,int>> g_lru;
static std::unordered_map<long long, std::list<std::pair<int,int>>::iterator> g_lru_pos;
static std::size_t g_gpu_budget = 0, g_gpu_used = 0;
static std::size_t g_cpu_budget = 0, g_cpu_used = 0;

static inline long long key_of(int handle, int index) {
    return (static_cast<long long>(handle) << 32) | static_cast<unsigned int>(index);
}

static void gpu_cache_evict_until(std::size_t need_bytes) {
    while (g_gpu_used + need_bytes > g_gpu_budget && !g_lru.empty()) {
        auto back = g_lru.back();
        long long k = key_of(back.first, back.second);
        auto it = g_gpu_cache.find(k);
        if (it != g_gpu_cache.end()) {
            cudaFree(it->second.dptr);
            g_gpu_used -= it->second.bytes;
            g_gpu_cache.erase(it);
        }
        auto itc = g_cpu_cache.find(k);
        if (itc != g_cpu_cache.end() && g_cpu_used > g_cpu_budget) {
            if (itc->second.hptr) cudaFreeHost(itc->second.hptr);
            g_cpu_used -= itc->second.bytes;
            g_cpu_cache.erase(itc);
        }
        g_lru.pop_back();
    }
}

static void cpu_cache_evict_until(std::size_t need_bytes) {
    while (g_cpu_used + need_bytes > g_cpu_budget && !g_lru.empty()) {
        auto back = g_lru.back();
        long long k = key_of(back.first, back.second);
        auto it = g_cpu_cache.find(k);
        if (it != g_cpu_cache.end()) {
            if (it->second.hptr) cudaFreeHost(it->second.hptr);
            g_cpu_used -= it->second.bytes;
            g_cpu_cache.erase(it);
        }
        g_lru_pos.erase(k);
        g_lru.pop_back();
    }
}

static inline void lru_touch(int handle, int index) {
    long long k = key_of(handle, index);
    auto it = g_lru_pos.find(k);
    if (it != g_lru_pos.end()) {
        g_lru.erase(it->second);
        it->second = g_lru.insert(g_lru.begin(), std::make_pair(handle, index));
    } else {
        auto pos = g_lru.insert(g_lru.begin(), std::make_pair(handle, index));
        g_lru_pos[k] = pos;
    }
}

static inline void lru_erase(long long k) {
    auto it = g_lru_pos.find(k);
    if (it != g_lru_pos.end()) {
        g_lru.erase(it->second);
        g_lru_pos.erase(it);
    }
}

static void read_from_disk_into(int fd, std::size_t file_off, void* dst_device_ptr, std::size_t bytes, int stream_idx) {
    std::lock_guard<std::mutex> lk(g_io_mutexes[static_cast<size_t>(stream_idx)]);
    HBuf& hb = g_io_hbufs[static_cast<size_t>(stream_idx)];
    char* dptr = static_cast<char*>(dst_device_ptr);
    std::size_t remaining = bytes;
    std::size_t dev_off = 0;
    std::size_t cur_off = file_off;
    int toggle = 0;
    while (remaining) {
        if (hb.ev[toggle]) cudaEventSynchronize(hb.ev[toggle]);
        std::size_t chunk = std::min<std::size_t>(remaining, g_chunk_bytes);
        off_t this_off = static_cast<off_t>(cur_off);
        ssize_t got = pread(fd, hb.p[toggle], static_cast<size_t>(chunk), this_off);
        if (got <= 0) break;
        cudaMemcpyAsync(dptr + dev_off, hb.p[toggle], static_cast<size_t>(got), cudaMemcpyHostToDevice, g_io_streams[stream_idx]);
        cudaEventRecord(hb.ev[toggle], g_io_streams[stream_idx]);
        (void)posix_fadvise(fd, this_off, static_cast<off_t>(got), POSIX_FADV_DONTNEED);
        remaining -= static_cast<std::size_t>(got);
        cur_off += static_cast<std::size_t>(got);
        dev_off  += static_cast<std::size_t>(got);
        toggle ^= 1;
    }
}

static void read_from_disk_into_with_cpu_cache(int fd, std::size_t file_off, void* hcache_ptr, void* dst_device_ptr, std::size_t bytes, int stream_idx) {
    char* dptr = static_cast<char*>(dst_device_ptr);
    char* hptr = static_cast<char*>(hcache_ptr);
    std::size_t remaining = bytes;
    std::size_t dev_off = 0;
    std::size_t cur_off = file_off;
    while (remaining) {
        std::size_t chunk = std::min<std::size_t>(remaining, g_chunk_bytes);
        off_t this_off = static_cast<off_t>(cur_off);
        ssize_t got = pread(fd, hptr + dev_off, static_cast<size_t>(chunk), this_off);
        if (got <= 0) break;
        cudaMemcpyAsync(dptr + dev_off, hptr + dev_off, static_cast<size_t>(got), cudaMemcpyHostToDevice, g_io_streams[stream_idx]);
        (void)posix_fadvise(fd, this_off, static_cast<off_t>(got), POSIX_FADV_DONTNEED);
        remaining -= static_cast<std::size_t>(got);
        cur_off += static_cast<std::size_t>(got);
        dev_off  += static_cast<std::size_t>(got);
    }
}

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
    g_io_mutexes.resize(g_io_streams.size());
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
    g_gpu_cache.clear();
    g_cpu_cache.clear();
    g_lru.clear();
    g_gpu_used = g_cpu_used = 0;
}

extern "C" void set_chunk_bytes(std::size_t bytes) {
    std::lock_guard<std::mutex> lk(g_mutex);
    if (bytes >= (1u<<20)) g_chunk_bytes = bytes;
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
    int fd = -1; std::size_t file_off = 0; std::size_t to_read = 0;
    {
        std::lock_guard<std::mutex> lk(g_mutex);
        if (!valid_handle(handle)) return;
        auto& m = g_models[handle];
        if (index < 0 || static_cast<std::size_t>(index) >= m.offsets.size()) return;
        file_off = m.offsets[static_cast<std::size_t>(index)];
        to_read = std::min<std::size_t>(bytes, m.sizes[static_cast<std::size_t>(index)]);
        fd = m.fd;
    }
    int stream_idx = std::min<int>(std::max(0, io_stream), static_cast<int>(g_io_streams.size()) - 1);

    long long k = key_of(handle, index);
    {
        std::lock_guard<std::mutex> lk(g_mutex);
        auto git = g_gpu_cache.find(k);
        if (git != g_gpu_cache.end() && git->second.bytes == to_read && git->second.dptr) {
            cudaMemcpyAsync(dst_device_ptr, git->second.dptr, to_read, cudaMemcpyDeviceToDevice, g_io_streams[stream_idx]);
            lru_touch(handle, index);
            return;
        }
        auto cit = g_cpu_cache.find(k);
        if (cit != g_cpu_cache.end() && cit->second.bytes == to_read && cit->second.hptr) {
            cudaMemcpyAsync(dst_device_ptr, cit->second.hptr, to_read, cudaMemcpyHostToDevice, g_io_streams[stream_idx]);
            lru_touch(handle, index);
            return;
        }
    }

    bool cached_gpu = false, cached_cpu = false;
    void* gpu_cache_ptr = nullptr;
    void* cpu_cache_ptr = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_mutex);
        if (g_gpu_budget >= to_read && to_read > 0) {
            if (g_gpu_used + to_read > g_gpu_budget) gpu_cache_evict_until(to_read);
            if (g_gpu_used + to_read <= g_gpu_budget) {
                if (cudaMalloc(&gpu_cache_ptr, to_read) == cudaSuccess && gpu_cache_ptr) {
                    g_gpu_used += to_read;
                    cached_gpu = true;
                }
            }
        }
        if (!cached_gpu && g_cpu_budget >= to_read && to_read > 0) {
            if (g_cpu_used + to_read > g_cpu_budget) cpu_cache_evict_until(to_read);
            if (g_cpu_used + to_read <= g_cpu_budget) {
                cpu_cache_ptr = alloc_pinned(to_read);
                if (cpu_cache_ptr) {
                    g_cpu_used += to_read;
                    cached_cpu = true;
                }
            }
        }
    }

    if (cached_gpu && gpu_cache_ptr) {
        read_from_disk_into(fd, file_off, gpu_cache_ptr, to_read, stream_idx);
        cudaMemcpyAsync(dst_device_ptr, gpu_cache_ptr, to_read, cudaMemcpyDeviceToDevice, g_io_streams[stream_idx]);
        std::lock_guard<std::mutex> lk(g_mutex);
        g_gpu_cache[k] = {gpu_cache_ptr, to_read, nullptr};
        lru_touch(handle, index);
        return;
    }

    if (cached_cpu && cpu_cache_ptr) {
        read_from_disk_into_with_cpu_cache(fd, file_off, cpu_cache_ptr, dst_device_ptr, to_read, stream_idx);
        std::lock_guard<std::mutex> lk(g_mutex);
        g_cpu_cache[k] = {cpu_cache_ptr, to_read};
        lru_touch(handle, index);
        return;
    }

    read_from_disk_into(fd, file_off, dst_device_ptr, to_read, stream_idx);
}

extern "C" void model_read_into_sched(int handle,
                                       int index,
                                       void* dst_device_ptr,
                                       std::size_t bytes,
                                       int io_stream,
                                       int compute_stream) {
    model_read_into(handle, index, dst_device_ptr, bytes, io_stream);
    int sidx = std::min<int>(std::max(0, io_stream), static_cast<int>(g_io_streams.size()) - 1);
    int cidx = std::min<int>(std::max(0, compute_stream), static_cast<int>(g_compute_streams.size()) - 1);
    cudaEvent_t ev; cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
    cudaEventRecord(ev, g_io_streams[sidx]);
    cudaStreamWaitEvent(g_compute_streams[cidx], ev, 0);
    cudaEventDestroy(ev);
}

extern "C" void model_read_into_batch(int handle,
                                       const int* indices,
                                       void* const* dst_device_ptrs,
                                       const std::size_t* bytes,
                                       int n,
                                       int io_stream) {
    for (int i = 0; i < n; ++i) model_read_into(handle, indices[i], dst_device_ptrs[i], bytes[i], io_stream);
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

extern "C" void cache_configure(std::size_t gpu_bytes, std::size_t cpu_pinned_bytes) {
    std::lock_guard<std::mutex> lk(g_mutex);
    g_gpu_budget = gpu_bytes;
    g_cpu_budget = cpu_pinned_bytes;
}

extern "C" void cache_clear(int level) {
    std::lock_guard<std::mutex> lk(g_mutex);
    if (level == 0 || level == 2) {
        for (auto& kv : g_gpu_cache) if (kv.second.dptr) cudaFree(kv.second.dptr);
        g_gpu_cache.clear();
        g_gpu_used = 0;
    }
    if (level == 1 || level == 2) {
        for (auto& kv : g_cpu_cache) if (kv.second.hptr) cudaFreeHost(kv.second.hptr);
        g_cpu_cache.clear();
        g_cpu_used = 0;
    }
    g_lru.clear();
    g_lru_pos.clear();
}

extern "C" void cache_stats(std::size_t* gpu_used, int* gpu_items, std::size_t* cpu_used, int* cpu_items) {
    std::lock_guard<std::mutex> lk(g_mutex);
    if (gpu_used) *gpu_used = g_gpu_used;
    if (gpu_items) *gpu_items = static_cast<int>(g_gpu_cache.size());
    if (cpu_used) *cpu_used = g_cpu_used;
    if (cpu_items) *cpu_items = static_cast<int>(g_cpu_cache.size());
}


