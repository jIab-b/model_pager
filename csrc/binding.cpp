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
static std::unordered_map<std::string, py::function> g_py_kernels;

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
    
    m.def("model_set_weights_layout", [](py::list file_offsets, py::list sizes) {
        const int count = static_cast<int>(py::len(file_offsets));
        if (count != static_cast<int>(py::len(sizes))) throw std::runtime_error("layout arrays mismatch");
        std::vector<std::size_t> offs(count), szs(count);
        for (int i = 0; i < count; ++i) {
            offs[i] = file_offsets[i].cast<std::size_t>();
            szs[i] = sizes[i].cast<std::size_t>();
        }
        model_set_weights_layout(offs.data(), szs.data(), count);
    }, "Set per-tensor file offsets and sizes for UMA layout");
    m.def("model_stage_file", [](const std::string &path, std::size_t chunk_bytes) {
        model_stage_file(path.c_str(), chunk_bytes);
    }, "Stage weights from safetensors file into UMA");
    m.def("model_planned_bytes", &model_planned_bytes, "Get planned UMA total bytes");

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

    m.def("register_kernel", [](const std::string &name, py::function fn) {
        g_py_kernels[name] = std::move(fn);
    }, "Register a Python callable as a kernel for a module name");

    m.def("launch_kernel", [](const std::string &name, const std::vector<at::Tensor> &inputs) {
        auto it = g_py_kernels.find(name);
        if (it == g_py_kernels.end())
            throw std::runtime_error("Kernel not registered: " + name);
        py::tuple args(inputs.size());
        for (size_t i = 0; i < inputs.size(); ++i) args[i] = py::cast(inputs[i]);
        py::object out = it->second(*args);
        return out.cast<at::Tensor>();
    }, "Launch a registered Python kernel by name");
}
