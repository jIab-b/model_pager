#include <torch/extension.h>
#include <torch/script.h>
#include "page_table.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <unordered_map>
#include <vector>
#include <string>

namespace py = pybind11;

static std::unordered_map<std::string, torch::jit::script::Module> g_jit_modules;

// Wrap UM ptr as Tensor
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

    m.def("model_reserve",     &model_reserve,     "Reserve UMA for entire model");
    m.def("model_prefetch",    &model_prefetch,    "Prefetch UMA pages for model");
    m.def("model_evict",       &model_evict,       "Evict UMA pages for model");

    m.def("get_memory_stats",  []() {
        size_t r,u,a,f;
        get_memory_stats(&r,&u,&a,&f);
        return py::make_tuple(r,u,a,f);
    }, "Get UMA and GPU memory stats");

    m.def("load_module", [](const std::string &name, const std::string &path) {
        torch::jit::script::Module module = torch::jit::load(path);
        module.eval();
        g_jit_modules[name] = std::move(module);
    }, "Load TorchScript module");

    m.def("launch_module", [](const std::string &name, const std::vector<at::Tensor> &inputs) {
        auto it = g_jit_modules.find(name);
        if (it == g_jit_modules.end())
            throw std::runtime_error("Module not loaded: " + name);
        std::vector<c10::IValue> ivals;
        ivals.reserve(inputs.size());
        for (const auto &t : inputs) ivals.emplace_back(t);
        at::IValue out = it->second.forward(ivals);
        return out.toTensor();
    }, "Launch a loaded TorchScript module");
}
