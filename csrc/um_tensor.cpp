#include <torch/extension.h>
#include "page_table.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>

namespace py = pybind11;

// NOTE: This is a **prototype** helper that wraps a slice of the Unified
// Memory pool (allocated with pager_reserve) as a torch::Tensor without
// copying.  Only contiguous FP16 / FP32 tensors are handled for now.
torch::Tensor tensor_from_um(void* ptr, std::vector<int64_t> sizes, c10::ScalarType dtype) {
    int64_t numel = 1;
    for (auto s : sizes) numel *= s;
    c10::TensorOptions opts = torch::TensorOptions().dtype(dtype).device(torch::kCUDA).requires_grad(false);
    auto deleter = [](void* /*unused*/) {};
    return torch::from_blob(ptr, c10::IntArrayRef(sizes), deleter, opts);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tensor_from_um",    &tensor_from_um,    "Create Tensor from UM ptr");
    m.def("pager_reserve",     &pager_reserve,     "Reserve unified-memory bytes");
    m.def("pager_prefetch",    &pager_prefetch,    "Prefetch pages to GPU");
    m.def("pager_evict",       &pager_evict,       "Evict pages to CPU");
    m.def("acquire_pages",     [](const std::string &id) { acquire_pages(id.c_str()); }, "Acquire pages for model");
    m.def("release_pages",     [](const std::string &id) { release_pages(id.c_str()); }, "Release pages for model");
    m.def("register_model",    [](const std::string &id, const std::string &path, const std::vector<std::string> &module_names, const std::vector<size_t> &offsets, const std::vector<size_t> &sizes) {
        std::vector<const char*> names_c;
        names_c.reserve(module_names.size());
        for (const auto &s : module_names) names_c.push_back(s.c_str());
        register_model(id.c_str(), path.c_str(), (int)names_c.size(), names_c.data(), offsets.data(), sizes.data());
    }, "Register model with page table");
    m.def("update_schedule",   [](const std::vector<std::string> &ids, const std::vector<int> &priorities) {
        std::vector<const char*> ids_c;
        ids_c.reserve(ids.size());
        for (const auto &s : ids) ids_c.push_back(s.c_str());
        update_schedule(ids_c.data(), priorities.data(), (int)ids_c.size());
    }, "Update scheduling priorities");
    m.def("process_schedule",  &process_schedule,  "Process scheduling tick");
    m.def("get_memory_stats",  []() {
        size_t uma_reserved, uma_used, gpu_allocated, gpu_free;
        get_memory_stats(&uma_reserved, &uma_used, &gpu_allocated, &gpu_free);
        return py::make_tuple(uma_reserved, uma_used, gpu_allocated, gpu_free);
    }, "Get UMA and GPU memory stats");
}
