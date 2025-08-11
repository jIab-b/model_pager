#include <torch/extension.h>
#include "weight_pager.hpp"

// NOTE: This is a **prototype** helper that wraps a slice of the Unified
// Memory pool (allocated with pager_reserve) as a torch::Tensor without
// copying.  Only contiguous FP16 / FP32 tensors are handled for now.

torch::Tensor tensor_from_um(void* ptr, std::vector<int64_t> sizes, c10::ScalarType dtype) {
    c10::TensorOptions opts = torch::TensorOptions().dtype(dtype).device(torch::kCUDA);
    auto storage = c10::Storage(c10::Storage::use_byte_size_t(),
                                torch::numel(sizes) * (dtype == c10::ScalarType::Half ? 2 : 4),
                                c10::IntrusivePtr<c10::StorageImpl>::reclaim(nullptr), /*resizable*/false);
    // Placeholder: real implementation must create StorageImpl that owns ptr.
    TORCH_WARN("tensor_from_um is a stub – returns empty tensor");
    return torch::empty({0}, opts);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tensor_from_um", &tensor_from_um, "Create Tensor from UM ptr (stub)");
}
