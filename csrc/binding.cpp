#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "page_table.hpp"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("runtime_init", &runtime_init, py::arg("device_id"), py::arg("compute_streams") = 2, py::arg("io_streams") = 2);
    m.def("runtime_shutdown", &runtime_shutdown);
    m.def("set_chunk_bytes", &set_chunk_bytes);
    m.def("cache_configure", &cache_configure);
    m.def("cache_clear", &cache_clear);
    m.def("cache_stats", []( ){
        std::size_t gu=0, cu=0; int gi=0, ci=0; cache_stats(&gu, &gi, &cu, &ci); return py::make_tuple(gu, gi, cu, ci);
    });

    m.def("model_register", [](const std::string& path,
                                 const std::vector<std::size_t>& offsets,
                                 const std::vector<std::size_t>& sizes) {
        return model_register(path.c_str(), offsets.data(), sizes.data(), static_cast<int>(sizes.size()));
    });
    m.def("model_close", &model_close);
    m.def("model_tensor_size", &model_tensor_size);
    m.def("gds_available", &gds_available);

    m.def("model_read_into", [](int handle, int index, at::Tensor dst, py::object io_stream_opt) {
        TORCH_CHECK(dst.is_cuda(), "dst must be CUDA tensor");
        void* ptr = dst.data_ptr();
        std::size_t bytes = static_cast<std::size_t>(dst.nbytes());
        int io_stream = io_stream_opt.is_none() ? 0 : io_stream_opt.cast<int>();
        model_read_into(handle, index, ptr, bytes, io_stream);
    });

    m.def("model_read_into_sched", [](int handle, int index, at::Tensor dst, int io_stream, int compute_stream) {
        TORCH_CHECK(dst.is_cuda(), "dst must be CUDA tensor");
        void* ptr = dst.data_ptr();
        std::size_t bytes = static_cast<std::size_t>(dst.nbytes());
        model_read_into_sched(handle, index, ptr, bytes, io_stream, compute_stream);
    });

    m.def("model_read_into_batch", [](int handle, const std::vector<int>& indices, 
        const std::vector<at::Tensor>& dsts, py::object io_stream_opt) {
        TORCH_CHECK(indices.size() == dsts.size(), "indices and dsts size mismatch");
        std::vector<void*> ptrs(dsts.size());
        std::vector<std::size_t> bytes(dsts.size());
        for (size_t i = 0; i < dsts.size(); ++i) {
            TORCH_CHECK(dsts[i].is_cuda(), "dst ", i, " must be CUDA tensor");
            ptrs[i] = dsts[i].data_ptr();
            bytes[i] = static_cast<std::size_t>(dsts[i].nbytes());
        }
        int io_stream = io_stream_opt.is_none() ? 0 : io_stream_opt.cast<int>();
        model_read_into_batch(handle, indices.data(), (void* const*)ptrs.data(), bytes.data(), 
            static_cast<int>(indices.size()), io_stream);
    });

    m.def("wait_all", &wait_all);
    m.def("get_memory_stats", [](){
        std::size_t ga=0, gf=0; get_memory_stats(&ga, &gf); return py::make_tuple(ga, gf);
    });
}


