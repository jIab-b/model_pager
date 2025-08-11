from torch.utils.cpp_extension import load
import os, pathlib

def load_native_libs():
    root = pathlib.Path(__file__).parent.parent / "csrc"
    weight_pager = load(
        name="weight_pager",
        sources=[str(root / "weight_pager.cu")],
        extra_cflags=["-O3", "-std=c++17", "-lineinfo"],
        extra_cuda_cflags=["--expt-relaxed-constexpr"],
        verbose=False,
    )
    activation_pool = load(
        name="activation_pool",
        sources=[str(root / "activation_pool.cpp")],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=False,
    )
    return weight_pager, activation_pool
