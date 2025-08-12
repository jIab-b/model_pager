from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="model_pager_exts",
    ext_modules=[
        CUDAExtension(
            "weight_pager_ext",
            [
                "csrc/weight_pager.cu",
                "csrc/um_tensor.cpp",
            ],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
