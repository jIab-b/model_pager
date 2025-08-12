#include <torch/extension.h>
#include "weight_pager.hpp"

// NOTE: This is a **prototype** helper that wraps a slice of the Unified
// Memory pool (allocated with pager_reserve) as a torch::Tensor without
// copying.  Only contiguous FP16 / FP32 tensors are handled for now.

torch::Tensor tensor_from_um(void* ptr, std::vector<int64_t> sizes, c10::ScalarType dtype) {
    // Compute number of elements
    int64_t numel = 1;
    for (auto s : sizes) numel *= s;

    // Build TensorOptions (cuda device, given dtype)
    c10::TensorOptions opts = torch::TensorOptions().dtype(dtype).device(torch::kCUDA).requires_grad(false);

    // Wrap the raw pointer without taking ownership (no-op deleter)
    auto deleter = [](void* /*unused*/) {};

    auto tensor = torch::from_blob(ptr, c10::IntArrayRef(sizes), deleter, opts);

    return tensor;
}

#include <pybind11/pybind11.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tensor_from_um", &tensor_from_um, "Create Tensor from UM ptr");
    m.def("pager_reserve",  &pager_reserve,  "Reserve unified-memory bytes");
    m.def("pager_prefetch", &pager_prefetch, "Prefetch pages to GPU");
    m.def("pager_evict",    &pager_evict,    "Evict pages to CPU");
}
